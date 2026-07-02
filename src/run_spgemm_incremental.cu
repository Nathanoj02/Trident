//
// run_spgemm_incremental - static-recompute baseline for the dynamic-vs-static
// distributed SpGEMM comparison
#include "hns_spgemm.cuh"
#include "sparse_summa.cuh"
#include "test_utils.cuh"
#include "incremental_updates.hpp"

#include <mpi.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <cctype>

// Resolve "N" into an absolute nnz count.
static std::size_t resolve_update_size(const std::string& spec, std::size_t total) {
    std::size_t b = 0, e = spec.size();
    while (b < e && std::isspace((unsigned char)spec[b]))     ++b;
    while (e > b && std::isspace((unsigned char)spec[e - 1])) --e;
    const std::string s = spec.substr(b, e - b);
    if (s.empty()) return 0;
    if (s.back() == '%') {
        const double pct = std::stod(s.substr(0, s.size() - 1));
        std::size_t out = (std::size_t)std::llround(pct / 100.0 * (double)total);
        if (out == 0 && pct > 0.0) out = 1;
        return out;
    }
    return (std::size_t)std::stoull(s);
}

using IT = int32_t;
using VT = float;

struct IncArgs {
    const char* matA = nullptr;
    const char* matB = nullptr;
    int p_r = 1, p_c = 1;
    Implementation impl = Implementation::SUMMA;
    const char* impl_str = "summa";
    uint64_t seed = 1;
    std::size_t batches = 10;
    std::size_t warmup = 3;          // leading batches excluded from timing stats
    std::string update_size_spec = "0"; // "N" or "N%" of global nnz (required)
    std::size_t update_size = 0;         // resolved absolute (filled after matrix read)
    bool static_b = false;
    std::size_t c_remote_size = 6442450944ull;
    const char* csv_path = nullptr;  // if set, write per-batch CSV here (rank 0)
};

static Implementation parse_impl(const char* s) {
    if (!strcmp(s, "async"))        return Implementation::ASYNC;
    if (!strcmp(s, "workstealing")) return Implementation::WORKSTEALING;
    if (!strcmp(s, "summa"))        return Implementation::SUMMA;
    return Implementation::SUMMA;
}

static void parse_inc_args(int argc, char** argv, IncArgs& a) {
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* def) -> const char* {
            return (i + 1 < argc) ? argv[++i] : def;
        };
        if      (!strcmp(argv[i], "--matA"))        a.matA = next(nullptr);
        else if (!strcmp(argv[i], "--matB"))        a.matB = next(nullptr);
        else if (!strcmp(argv[i], "--2D-pgrid")) {
            const char* g = next("1x1");
            sscanf(g, "%dx%d", &a.p_r, &a.p_c);
        }
        else if (!strcmp(argv[i], "--impl")) { a.impl_str = next("summa"); a.impl = parse_impl(a.impl_str); }
        else if (!strcmp(argv[i], "--seed"))        a.seed = strtoull(next("1"), nullptr, 10);
        else if (!strcmp(argv[i], "-F") || !strcmp(argv[i], "--fix-updates"))
            a.batches = strtoull(next("10"), nullptr, 10);
        else if (!strcmp(argv[i], "-p") || !strcmp(argv[i], "--update-size"))
            a.update_size_spec = next("0");
        else if (!strcmp(argv[i], "-w") || !strcmp(argv[i], "--warmup"))
            a.warmup = strtoull(next("3"), nullptr, 10);
        else if (!strcmp(argv[i], "--csv"))         a.csv_path = next(nullptr);
        else if (!strcmp(argv[i], "--static"))      a.static_b = true;
        else if (!strcmp(argv[i], "--c-remote-size"))
            a.c_remote_size = strtoull(next("6442450944"), nullptr, 10);
    }
}

// Extract a DistCusparseCSX's local CSR (device) to a host HCsr, sorted by (row, col)
static tinc::HCsr<IT, VT> extract_band(DistCusparseCSX<IT, VT>* d) {
    mmio::CSX<IT, VT>* m = d->csx->mat;
    const IT n = m->nrows, mc = m->ncols, nz = m->nnz;
    tinc::HCsr<IT, VT> b(n, mc);
    b.col.resize(nz);
    b.val.resize(nz);
    CUDA_CHECK(cudaMemcpy(b.row_off.data(), m->ptr_vec, sizeof(IT) * (n + 1), cudaMemcpyDeviceToHost));
    if (nz > 0) {
        CUDA_CHECK(cudaMemcpy(b.col.data(), m->idx_vec, sizeof(IT) * nz, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.val.data(), m->val,     sizeof(VT) * nz, cudaMemcpyDeviceToHost));
    }
    for (IT r = 0; r < n; ++r) {
        const IT lo = b.row_off[r], hi = b.row_off[r + 1];
        std::vector<std::pair<IT, VT>> e;
        e.reserve(hi - lo);
        for (IT j = lo; j < hi; ++j) e.emplace_back(b.col[j], b.val[j]);
        std::sort(e.begin(), e.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
        for (IT j = lo; j < hi; ++j) { b.col[j] = e[j - lo].first; b.val[j] = e[j - lo].second; }
    }
    return b;
}

// Build a fresh DistCusparseCSX (device) from a host HCsr, reusing `part`.
static DistCusparseCSX<IT, VT>* build_dist(const tinc::HCsr<IT, VT>& h, Partitioning* part) {
    const IT nz = h.nnz();
    mmio::COO<IT, VT>* coo = mmio::COO_create<IT, VT>(h.n, h.m, nz, true);
    IT idx = 0;
    for (IT r = 0; r < h.n; ++r)
        for (IT j = h.row_off[r]; j < h.row_off[r + 1]; ++j) {
            coo->row[idx] = r; coo->col[idx] = h.col[j]; coo->val[idx] = h.val[j]; ++idx;
        }
    mmio::CSX<IT, VT>* csx = coo_to_row_csx_contig<IT, VT>(coo);   // device CSX
    mmio::COO_destroy(&coo);
    CusparseCSX<IT, VT>* cc = new CusparseCSX<IT, VT>(csx);
    return new DistCusparseCSX<IT, VT>(cc, part);
}

int main(int argc, char** argv) {
#ifdef KOKKOS
    Kokkos::initialize(argc, argv);
    {
#endif
    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    IncArgs args;
    parse_inc_args(argc, argv, args);
    const int nprocpergroup = world_size / (args.p_r * args.p_c);

    if (world_rank == 0) {
        std::printf("---- run_spgemm_incremental ----\n");
        std::printf("A=%s B=%s grid=%dx%dx%d impl=%s seed=%llu batches=%zu warmup=%zu update_size=%s static_b=%d\n",
                    args.matA, args.matB, args.p_r, args.p_c, nprocpergroup,
                    args.impl_str, (unsigned long long)args.seed, args.batches,
                    args.warmup, args.update_size_spec.c_str(), (int)args.static_b);
        std::fflush(stdout);
    }

    int gpn = 0;
    CUDA_CHECK(cudaGetDeviceCount(&gpn));
    CUDA_CHECK(cudaSetDevice(world_rank % gpn));

    // ---- read base operands (full local tiles) ----
    mmio::Matrix_Metadata metaA, metaB;
    dmmio::DCOO<IT, VT>* dcoo_A = dmmio::DCOO_read<IT, VT>(
        args.matA, world_size, world_rank, args.p_r, args.p_c, nprocpergroup,
        dmmio::PartitioningType::Naive, dmmio::Operation::None, true, &metaA, MASK_SIZE, false);
    dmmio::DCOO<IT, VT>* dcoo_B = dmmio::DCOO_read<IT, VT>(
        args.matB, world_size, world_rank, args.p_r, args.p_c, nprocpergroup,
        dmmio::PartitioningType::Naive, dmmio::Operation::None, true, &metaB, MASK_SIZE, false,
        dcoo_A->permutation);

    DistCusparseCSX<IT, VT>* base_A = new DistCusparseCSX<IT, VT>(dcoo_A, mmio::MajorDim::ROWS);
    DistCusparseCSX<IT, VT>* base_B = new DistCusparseCSX<IT, VT>(dcoo_B, mmio::MajorDim::ROWS);
    Partitioning* part_A = base_A->partitioning;
    Partitioning* part_B = base_B->partitioning;

    tinc::HCsr<IT, VT> band_A = extract_band(base_A);
    tinc::HCsr<IT, VT> band_B = extract_band(base_B);

    // ---- resolve update-size spec ----
    {
        unsigned long long loc = (unsigned long long)band_A.nnz(), glob = 0;
        MPI_Allreduce(&loc, &glob, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        args.update_size = resolve_update_size(args.update_size_spec, (std::size_t)glob);
        if (world_rank == 0)
            std::printf("update_size: %s -> %zu (global A nnz %llu)\n",
                        args.update_size_spec.c_str(), args.update_size, glob);
    }

    // ---- split bands into start + disjoint update batches ----
    auto alloc_a = tinc::hamilton_alloc<IT>(band_A.nnz(), args.update_size, args.batches, world_size, MPI_COMM_WORLD);
    auto alloc_b = tinc::hamilton_alloc<IT>(band_B.nnz(), args.update_size, args.batches, world_size, MPI_COMM_WORLD);
    const IT local_upd_a = alloc_a[world_rank];
    const IT local_upd_b = args.static_b ? IT{0} : alloc_b[world_rank];

    const uint64_t s_a = args.seed + 1009ull * (uint64_t)world_rank + 1;
    const uint64_t s_b = args.seed + 1009ull * (uint64_t)world_rank + 2;

    auto [start_A, upd_A] = tinc::generate_disjoint_updates<IT, VT>(band_A, args.batches, local_upd_a, s_a);
    tinc::HCsr<IT, VT> start_B;
    std::vector<tinc::HCsr<IT, VT>> upd_B;
    if (args.static_b) {
        start_B = band_B;                                  // B fixed at full band
        upd_B.assign(args.batches, tinc::HCsr<IT, VT>(band_B.n, band_B.m));
    } else {
        auto pr = tinc::generate_disjoint_updates<IT, VT>(band_B, args.batches, local_upd_b, s_b);
        start_B = std::move(pr.first);
        upd_B   = std::move(pr.second);
    }

    ThreadPool pool(2);

    tinc::HCsr<IT, VT> A_acc = start_A;
    tinc::HCsr<IT, VT> B_acc = start_B;

    // per-batch records (populated on rank 0 only)
    struct BatchRec {
        int64_t nnz_a, nnz_b, nnz_c;
        double  spgemm_ms;
        uint64_t csum_a, csum_b;
    };
    std::vector<BatchRec> recs;
    if (world_rank == 0) recs.reserve(args.batches);

    for (std::size_t b = 0; b < args.batches; ++b) {
        // insert this batch's new coordinates
        A_acc = tinc::apply_update(A_acc, upd_A[b]);
        if (!args.static_b) B_acc = tinc::apply_update(B_acc, upd_B[b]);

        // operand checksums (order-independent sum across ranks)
        uint64_t csa = tinc::checksum(A_acc), csb = tinc::checksum(B_acc);
        MPI_Allreduce(MPI_IN_PLACE, &csa, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &csb, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

        DistCusparseCSX<IT, VT>* dA = build_dist(A_acc, part_A);
        DistCusparseCSX<IT, VT>* dB = build_dist(B_acc, part_B);

        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();

        DistCusparseCSX<IT, VT>* dC = (args.impl == Implementation::SUMMA)
            ? sparse_summa(dA, dB)
            : hns_spgemm_main<IT, VT>(dA, dB, args.impl, pool, nullptr, args.c_remote_size, false, false);

        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        MPI_Allreduce(MPI_IN_PLACE, &ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        int64_t nnz_a = A_acc.nnz(), nnz_b = B_acc.nnz();
        int64_t nnz_c = (dC ? (int64_t)dC->getLocalNnz() : 0);
        MPI_Allreduce(MPI_IN_PLACE, &nnz_a, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &nnz_b, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &nnz_c, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

        if (world_rank == 0) {
            recs.push_back({nnz_a, nnz_b, nnz_c, ms, csa, csb});
            const bool warm = (b < args.warmup);
            std::printf("[batch %2zu/%zu]%s spgemm=%8.3f ms  nnz_c=%lld\n",
                        b + 1, args.batches, warm ? " (warmup)" : "        ",
                        ms, (long long)nnz_c);
            std::fflush(stdout);
        }

        if (dC) delete dC;
        delete dA;
        delete dB;
    }

    // ---- timing summary + CSV (rank 0) ----
    if (world_rank == 0) {
        const char* mat = extract_matrix_name(args.matA);

        // optional CSV for the analysis notebook (unified per-batch schema)
        if (args.csv_path) {
            std::ofstream csv(args.csv_path);
            if (csv) {
                csv << "bench,impl,matrix,nproc,p_r,p_c,node_size,seed,batches,warmup,"
                       "update_size,batch_idx,is_warmup,nnz_a_global,nnz_b_global,"
                       "nnz_c_global,spgemm_ms,csum_a_global,csum_b_global\n";
                for (std::size_t b = 0; b < recs.size(); ++b) {
                    const auto& r = recs[b];
                    csv << "trident_static," << args.impl_str << ',' << mat << ','
                        << world_size << ',' << args.p_r << ',' << args.p_c << ','
                        << nprocpergroup << ',' << args.seed << ',' << args.batches << ','
                        << args.warmup << ',' << args.update_size << ',' << b << ','
                        << (b < args.warmup ? 1 : 0) << ',' << r.nnz_a << ',' << r.nnz_b << ','
                        << r.nnz_c << ',' << r.spgemm_ms << ',' << r.csum_a << ',' << r.csum_b << '\n';
                }
                std::printf("csv written: %s (%zu rows)\n", args.csv_path, recs.size());
            } else {
                std::fprintf(stderr, "WARN: could not open csv path '%s'\n", args.csv_path);
            }
        }

        // stats over measured (non-warmup) batches
        const std::size_t n = recs.size();
        const std::size_t w = std::min(args.warmup, n);
        const std::size_t m = n - w;            // measured count
        double total_all = 0.0, total_meas = 0.0;
        for (std::size_t b = 0; b < n; ++b) {
            total_all += recs[b].spgemm_ms;
            if (b >= w) total_meas += recs[b].spgemm_ms;
        }
        double mean = 0.0, sd = 0.0, mn = 0.0, mx = 0.0;
        if (m > 0) {
            mean = total_meas / (double)m;
            mn = mx = recs[w].spgemm_ms;
            for (std::size_t b = w; b < n; ++b) {
                double x = recs[b].spgemm_ms;
                sd += (x - mean) * (x - mean);
                mn = std::min(mn, x);
                mx = std::max(mx, x);
            }
            sd = (m > 1) ? std::sqrt(sd / (double)(m - 1)) : 0.0;   // sample std (ddof=1)
        }

        std::printf("\n==== run_spgemm_incremental summary ====\n");
        std::printf("matrix=%s impl=%s grid=%dx%dx%d nproc=%d seed=%llu\n",
                    mat, args.impl_str, args.p_r, args.p_c, nprocpergroup,
                    world_size, (unsigned long long)args.seed);
        std::printf("batches=%zu warmup=%zu measured=%zu update_size=%zu static_b=%d\n",
                    n, w, m, args.update_size, (int)args.static_b);
        if (n) std::printf("final nnz: a=%lld b=%lld c=%lld\n",
                           (long long)recs[n-1].nnz_a, (long long)recs[n-1].nnz_b,
                           (long long)recs[n-1].nnz_c);
        std::printf("total spgemm: all=%.3f ms  measured=%.3f ms\n", total_all, total_meas);
        std::printf("per-batch spgemm (measured n=%zu): mean=%.3f ms  std=%.3f ms  min=%.3f  max=%.3f\n",
                    m, mean, sd, mn, mx);
        std::printf("========================================\n");
        std::fflush(stdout);
    }

    delete base_A;
    delete base_B;
    dmmio::DCOO_destroy(&dcoo_A);
    dmmio::DCOO_destroy(&dcoo_B);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#ifdef KOKKOS
    }
    Kokkos::finalize();
#endif
    return 0;
}
