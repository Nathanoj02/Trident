#pragma once
//
// Incremental-update generation for the dynamic-vs-static SpGEMM comparison.
#include <vector>
#include <utility>
#include <tuple>
#include <random>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <mpi.h>

namespace tinc {

// Minimal host CSR (tile-local indices). row_off has n+1 entries.
template<typename IT, typename VT>
struct HCsr {
    IT n = 0, m = 0;
    std::vector<IT> row_off;   // size n+1
    std::vector<IT> col;       // size nnz
    std::vector<VT> val;       // size nnz

    HCsr() = default;
    HCsr(IT n_, IT m_) : n(n_), m(m_), row_off(static_cast<std::size_t>(n_) + 1, 0) {}

    IT nnz() const { return row_off.empty() ? IT{0} : row_off.back(); }
};

template<typename T> inline MPI_Datatype mpi_index_type();
template<> inline MPI_Datatype mpi_index_type<int32_t>()  { return MPI_INT32_T;  }
template<> inline MPI_Datatype mpi_index_type<uint32_t>() { return MPI_UINT32_T; }
template<> inline MPI_Datatype mpi_index_type<int64_t>()  { return MPI_INT64_T;  }
template<> inline MPI_Datatype mpi_index_type<uint64_t>() { return MPI_UINT64_T; }

// Largest-remainder (Hamilton) allocation of `global_upd` per-batch updates
// across ranks, weighted by `band_nnz`
template<typename IT>
std::vector<IT> hamilton_alloc(IT band_nnz, std::size_t global_upd,
                               std::size_t batches, int world_size, MPI_Comm comm)
{
    std::vector<IT> nnz_per_rank(world_size);
    MPI_Allgather(&band_nnz, 1, mpi_index_type<IT>(),
                  nnz_per_rank.data(), 1, mpi_index_type<IT>(), comm);

    std::size_t global_nnz = 0;
    for (auto x : nnz_per_rank) global_nnz += static_cast<std::size_t>(x);

    std::vector<IT> alloc(world_size, IT{0});
    if (global_nnz == 0 || global_upd == 0) return alloc;

    std::vector<std::size_t> rems(world_size);
    std::size_t total_floor = 0;
    for (int r = 0; r < world_size; ++r) {
        const std::size_t prod = global_upd * static_cast<std::size_t>(nnz_per_rank[r]);
        const std::size_t fl   = prod / global_nnz;
        const std::size_t rm   = prod - fl * global_nnz;
        alloc[r] = static_cast<IT>(fl);
        rems[r]  = rm;
        total_floor += fl;
    }

    std::size_t residual = (global_upd > total_floor) ? (global_upd - total_floor) : 0;
    if (residual > 0) {
        std::vector<int> order(world_size);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (rems[a] != rems[b]) return rems[a] > rems[b];
            return a < b;
        });
        for (std::size_t i = 0; i < residual && static_cast<int>(i) < world_size; ++i)
            alloc[order[i]] += 1;
    }

    if (batches == 0) return alloc;

    std::vector<IT> cap(world_size);
    for (int r = 0; r < world_size; ++r)
        cap[r] = static_cast<IT>(static_cast<std::size_t>(nnz_per_rank[r]) / batches);

    std::size_t spillover = 0;
    for (int r = 0; r < world_size; ++r) {
        if (alloc[r] > cap[r]) {
            spillover += static_cast<std::size_t>(alloc[r] - cap[r]);
            alloc[r]   = cap[r];
        }
    }
    if (spillover == 0) return alloc;

    std::vector<int> order(world_size);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        const std::size_t ha = static_cast<std::size_t>(cap[a] - alloc[a]);
        const std::size_t hb = static_cast<std::size_t>(cap[b] - alloc[b]);
        if (ha != hb) return ha > hb;
        return a < b;
    });
    for (int r : order) {
        if (spillover == 0) break;
        const std::size_t headroom = static_cast<std::size_t>(cap[r] - alloc[r]);
        const std::size_t take     = std::min(headroom, spillover);
        alloc[r]  += static_cast<IT>(take);
        spillover -= take;
    }
    return alloc;
}

// Build a (row, col-ascending) sorted CSR from elements[lo, hi)
template<typename IT, typename VT>
HCsr<IT, VT> build_slice(const std::vector<std::tuple<IT, IT, VT>>& elements,
                         std::size_t lo, std::size_t hi, IT n, IT m)
{
    HCsr<IT, VT> out(n, m);
    std::vector<std::vector<std::pair<IT, VT>>> rows(n);
    for (std::size_t j = lo; j < hi; ++j) {
        const auto& [r, c, v] = elements[j];
        rows[r].emplace_back(c, v);
    }
    IT idx = 0;
    out.col.reserve(hi - lo);
    out.val.reserve(hi - lo);
    for (IT r = 0; r < n; ++r) {
        std::sort(rows[r].begin(), rows[r].end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& [c, v] : rows[r]) { out.col.push_back(c); out.val.push_back(v); ++idx; }
        out.row_off[r + 1] = idx;
    }
    return out;
}

// Flatten band into row-major (row, col-ascending) element list
template<typename IT, typename VT>
std::vector<std::tuple<IT, IT, VT>> flatten(const HCsr<IT, VT>& band)
{
    std::vector<std::tuple<IT, IT, VT>> elements;
    elements.reserve(band.nnz());
    for (IT r = 0; r < band.n; ++r)
        for (IT j = band.row_off[r]; j < band.row_off[r + 1]; ++j)
            elements.emplace_back(r, band.col[j], band.val[j]);
    return elements;
}

// Split band into {start, updates}; updates.size() == batches, disjoint, each
// of size local_upd (last clamped); start = remainder. start + all updates == band
template<typename IT, typename VT>
std::pair<HCsr<IT, VT>, std::vector<HCsr<IT, VT>>>
generate_disjoint_updates(const HCsr<IT, VT>& band, std::size_t batches,
                          IT local_upd, uint64_t seed)
{
    const IT n = band.n, m = band.m;
    auto elements = flatten(band);

    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
    std::shuffle(elements.begin(), elements.end(), rng);

    const std::size_t nnz   = elements.size();
    const std::size_t want  = (local_upd > 0) ? static_cast<std::size_t>(local_upd) * batches : 0;
    const std::size_t upd_n = std::min(want, nnz);

    std::vector<HCsr<IT, VT>> updates;
    updates.reserve(batches);
    for (std::size_t b = 0; b < batches; ++b) {
        const std::size_t lo = std::min(b * static_cast<std::size_t>(local_upd), upd_n);
        const std::size_t hi = std::min(lo + static_cast<std::size_t>(local_upd), upd_n);
        updates.push_back(build_slice(elements, lo, hi, n, m));
    }
    HCsr<IT, VT> start = build_slice(elements, upd_n, nnz, n, m);
    return std::make_pair(std::move(start), std::move(updates));
}

// base += delta (sum on duplicate (r,c)); output sorted. For disjoint inserts
// no collision occurs.
template<typename IT, typename VT>
HCsr<IT, VT> apply_update(const HCsr<IT, VT>& base, const HCsr<IT, VT>& delta)
{
    const IT n = base.n, m = base.m;
    std::vector<std::unordered_map<IT, VT>> rows(n);
    for (IT r = 0; r < n; ++r)
        for (IT j = base.row_off[r]; j < base.row_off[r + 1]; ++j)
            rows[r][base.col[j]] = base.val[j];
    for (IT r = 0; r < delta.n; ++r)
        for (IT j = delta.row_off[r]; j < delta.row_off[r + 1]; ++j)
            rows[r][delta.col[j]] += delta.val[j];

    HCsr<IT, VT> out(n, m);
    IT idx = 0;
    for (IT r = 0; r < n; ++r) {
        std::vector<std::pair<IT, VT>> e(rows[r].begin(), rows[r].end());
        std::sort(e.begin(), e.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& [c, v] : e) { out.col.push_back(c); out.val.push_back(v); ++idx; }
        out.row_off[r + 1] = idx;
    }
    return out;
}

// Order-independent checksum of (r, c, bits(v)) over the matrix
template<typename IT, typename VT>
uint64_t checksum(const HCsr<IT, VT>& a)
{
    uint64_t acc = 1469598103934665603ull; // FNV offset, used as an accumulator base
    for (IT r = 0; r < a.n; ++r) {
        for (IT j = a.row_off[r]; j < a.row_off[r + 1]; ++j) {
            uint32_t vb; std::memcpy(&vb, &a.val[j], sizeof(uint32_t));
            uint64_t h = (static_cast<uint64_t>(static_cast<uint32_t>(r)) << 40)
                       ^ (static_cast<uint64_t>(static_cast<uint32_t>(a.col[j])) << 16)
                       ^ static_cast<uint64_t>(vb);
            h *= 1099511628211ull;          // FNV prime mix
            acc += h;                        // sum -> order independent
        }
    }
    return acc;
}

} // namespace tinc
