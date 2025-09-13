#include <mpi.h>
#include <iostream>
#include <variant>

#include <mmio/io.h>
#include <mmio/mmio.h>
#include <mmio/utils.h>
#include <dmmio/dio.h>
#include <dmmio/dmmio.h>
#include <dmmio/dutils.h>
#include <dmmio/partitioning.h>

#include <ccutils/mpi/mpi_macros.h>

// CHECK IF STILL REQUIRED
#include <cstdint>
#include <string.h>
#include <memory>
// -----------------------

#include "../include/test_utils.cuh"

#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_CooMatrix.hpp>
#include <KokkosSparse_CcsMatrix.hpp>
#include <KokkosSparse_coo2crs.hpp>
#include <KokkosSparse_crs2ccs.hpp>
#include <KokkosSparse_ccs2crs.hpp>

#define OPSTR(X) ((X == dmmio::Operation::None) ? ("None") : ("Transpose") )

namespace KokkosWrap {
    enum class MajorDim {
        ROWS, // i.e. CSR
        COLS  // i.e. CSC
    };

    template <typename IT, typename VT>
    struct Matrix {
        dmmio::Partitioning* partitioning;  // still raw pointer to external partitioning

        // Instead of a raw union, use std::variant for safety
        std::variant<
            KokkosSparse::CrsMatrix<VT, IT, Kokkos::DefaultExecutionSpace, void, IT>,
            KokkosSparse::CcsMatrix<VT, IT, Kokkos::DefaultExecutionSpace, void, IT>
        > storage;

        MajorDim layout;  // which one we actually hold

        // ---- Constructor ----
        Matrix(dmmio::DCOO<IT, VT>* dcoo, MajorDim T)
            : partitioning(dcoo->partitioning), layout(T)
        {
            int nnz = dcoo->coo->nnz;
            Kokkos::View<IT*> row_d("row", nnz);
            Kokkos::View<IT*> col_d("col", nnz);
            Kokkos::View<VT*> val_d("val", nnz);

            auto row_h = Kokkos::create_mirror_view(row_d);
            auto col_h = Kokkos::create_mirror_view(col_d);
            auto val_h = Kokkos::create_mirror_view(val_d);

            for (int i = 0; i < nnz; i++) {
                row_h(i) = dcoo->coo->row[i];
                col_h(i) = dcoo->coo->col[i];
                val_h(i) = static_cast<VT>(1.0); // TODO: use real values if available
            }

            Kokkos::deep_copy(row_d, row_h);
            Kokkos::deep_copy(col_d, col_h);
            Kokkos::deep_copy(val_d, val_h);

            // Build COO
            KokkosSparse::CooMatrix<VT, IT, Kokkos::DefaultExecutionSpace, void, IT> M(
                dcoo->coo->nrows, dcoo->coo->ncols, row_d, col_d, val_d);
            Kokkos::fence();

            // --- Step 1: COO → CSR (row-major) ---
            KokkosSparse::CrsMatrix<VT, IT, Kokkos::DefaultExecutionSpace, void, IT> csr(
                M.numRows(), M.numCols(), M.entries(), M.rowIndices(), M.values());
            Kokkos::View<IT*> row_map("row_map", M.numRows()+1);
            KokkosSparse::coo2crs(&csr.graph.row_map, M.rowIndices(), M.numEntries(), M.numRows());

            // --- Step 2: Decide layout ---
            if (T == MajorDim::ROWS) {
                storage = csr; // keep CSR
            } else {
                // CSR → CSC (column-major)
                KokkosSparse::CcsMatrix<VT, IT, Kokkos::DefaultExecutionSpace, void, IT> csc(
                    M.numCols(), M.numEntries());
                KokkosSparse::crs2ccs(&csc, csr);
                storage = csc; // store CSC
            }
        }
    };
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &name_len);

    std::cout << "Hello world from processor " << processor_name
            << ", rank " << world_rank << " out of " << world_size << " processors\n";
    MPI_Barrier(MPI_COMM_WORLD);

    Config * config = (Config *)(malloc(sizeof(Config)));
    parse_args(argc, argv, config);

    // Set the process grid parameeters
    int nprocrows = config->nprocrows, nproccols = config->nproccols, nprocpergroup = world_size/(nprocrows*nproccols);

    // Compute all the respective partitioning types
    dmmio::Operation Aop, Bop, Cop;
    dmmio::PartitioningType Apart, Bpart, Cpart;

    Aop   = dmmio::Operation::None;
    Apart = (dmmio::PartitioningType)config->part_num;

    if (Apart == dmmio::PartitioningType::Naive) {
        Bpart = dmmio::PartitioningType::Naive;
        Cpart = dmmio::PartitioningType::Naive;

        Bop   = dmmio::Operation::None;
        Cop   = dmmio::Operation::None;
    } else {
        // TODO wirite it for other partitionings and check it
        fprintf(stderr, "Partitioning different by Naive are not supported yet.\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }

    // Reading the distribuited matrices
    std::string A_mtx_path = (std::string) config->matpathA;
    mmio::Matrix_Metadata *meta_A = new mmio::Matrix_Metadata();
    dmmio::DCOO<uint32_t, float> *dcoo_A = dmmio::DCOO_read<uint32_t, float>(
        A_mtx_path.c_str(),
        world_size, world_rank,
        nprocrows, nproccols, nprocpergroup,
        Apart, Aop,
        false, meta_A
    );

    std::string B_mtx_path = (std::string) config->matpathA;
    mmio::Matrix_Metadata *meta_B = new mmio::Matrix_Metadata();
    dmmio::DCOO<uint32_t, float> *dcoo_B = dmmio::DCOO_read<uint32_t, float>(
        A_mtx_path.c_str(),
        world_size, world_rank,
        nprocrows, nproccols, nprocpergroup,
        Apart, Bop,
        false, meta_B
    );

    // Some prints
    if (world_rank == 0) {
      std::cout << "A matrix file path: " << config->matpathA << std::endl;
      std::cout << "B matrix file path: " << config->matpathB << std::endl;
      std::cout << "Number of processes per row: "  << nprocrows     << std::endl;
      std::cout << "Number of processes per col: "  << nproccols     << std::endl;
      std::cout << "Number of processes per node: " << nprocpergroup << std::endl;
    }

    if (world_rank == 0) {
      fprintf(stdout, "\n================= Hierarchical Partitioning ==================\n");
      fprintf(stdout, "A partitioning: %s, operand: %s\n", dcoo_A->partitioning->type_str, OPSTR(dcoo_A->partitioning->op));
      fprintf(stdout, "B partitioning: %s, operand: %s\n", dcoo_B->partitioning->type_str, OPSTR(dcoo_B->partitioning->op));
      fprintf(stdout, "C partitioning: %s, operand: %s\n", "????", OPSTR(Cop));
    }

    MPI_Barrier(MPI_COMM_WORLD);
// ==== End of Setup ====

    // TODO: check all the grids are the same or compatible (do we support rectangular grids?)
    dmmio::ProcessGrid *Cgrid = dcoo_A->partitioning->grid; // NOTE BUG TODO: we nead to think something for the grid managing.
    // dmmio::utils::ProcessGrid_print(dcoo_A->partitioning->grid);
    dmmio::utils::ProcessGrid_graph(dcoo_A->partitioning->grid, stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    Kokkos::initialize(argc, argv);
    // auto *kokkos_A = dmmio2kokkos(dcoo_A);
    // auto *kokkos_B = dmmio2kokkos(dcoo_B);
    KokkosWrap::Matrix<uint32_t, float> kokkos_A(dcoo_A, KokkosWrap::MajorDim::COLS);
    KokkosWrap::Matrix<uint32_t, float> kokkos_B(dcoo_B, KokkosWrap::MajorDim::ROWS);

    // delete kokkos_A;
    // delete kokkos_B;
    Kokkos::finalize();

    delete meta_A;
    delete meta_B;
    dmmio::DCOO_destroy(&dcoo_A);
    dmmio::DCOO_destroy(&dcoo_B);

    MPI_Finalize();
    return(0);
}
