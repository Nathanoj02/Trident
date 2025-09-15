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
#include "../ccutils/include/ccutils/cuda/cuda_utils.hpp"

#include <KokkosWrap_tmpfix.hpp>
#include "KokkosSparse_spgemm.hpp"

#define OPSTR(X) ((X == dmmio::Operation::None) ? ("None") : ("Transpose") )


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
    dmmio::utils::ProcessGrid_graph(dcoo_A->partitioning->grid, stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    dmmio::partitioning::indextransform::transformCoo::global2group(dcoo_A);
    dmmio::partitioning::indextransform::transformCoo::global2group(dcoo_B);

    Kokkos::initialize(argc, argv);
    {
        KokkosWrap::Matrix<int32_t, uint32_t, float> kokkos_A(dcoo_A, KokkosWrap::MajorDim::COLS);
        KokkosWrap::Matrix<int32_t, uint32_t, float> kokkos_B(dcoo_B, KokkosWrap::MajorDim::ROWS);

        // Simulate a local multiplication only A(i,j)*B(i,j) tiles are multiplied
        using csr_matrix_type = typename KokkosSparse::CrsMatrix<float, int32_t, Kokkos::DefaultExecutionSpace, void, int32_t>;
        using csc_matrix_type = typename KokkosSparse::CcsMatrix<float, int32_t, Kokkos::DefaultExecutionSpace, void, int32_t>;

        auto tmp_A = KokkosSparse::ccs2crs(std::get<csc_matrix_type>(kokkos_A.storage));
        auto tmp_B = std::get<csr_matrix_type>(kokkos_B.storage);

        csr_matrix_type C = KokkosSparse::spgemm<csr_matrix_type>(tmp_A, false, tmp_B, false);

        // Keep the C raw pointers
        mmio::CSR<int32_t, float> d_out_C = KokkosWrap::rawptr_get(C);

        // Copy C to host (just to dbg)
        mmio::CSR<int32_t, float>* h_out_C  = mmio::CSR_create<int32_t, float>(C.numRows(), C.numCols(), C.nnz(), true);
        d2h_copy(h_out_C->row_ptr, C.numRows(), d_out_C.row_ptr);
        d2h_copy(h_out_C->col_idx, C.nnz(),     d_out_C.col_idx);
        d2h_copy(h_out_C->val,     C.nnz(),     d_out_C.val);

        mmio::utils::CSR_print_as_dense(h_out_C, "h_out_C");

        // kokkos_A and kokkos_B are automatically freed since in scope
    }
    Kokkos::finalize();

    delete meta_A;
    delete meta_B;
    dmmio::DCOO_destroy(&dcoo_A);
    dmmio::DCOO_destroy(&dcoo_B);

    MPI_Finalize();
    return(0);
}
