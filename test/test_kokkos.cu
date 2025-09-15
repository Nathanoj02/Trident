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

#define D2D_COPY(TY, D, S, N)                                                   \
    CUDA_CHECK(cudaMalloc(&(D), sizeof(TY)*(N)));                               \
    CUDA_CHECK(cudaMemcpy(D, D, sizeof(TY)*(N), cudaMemcpyDeviceToDevice));


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

        // ----- Simulate the communication by simply copy the A and B raw pointers -----
        mmio::CSC<int32_t, float> *tmp_csc;
        mmio::CSR<int32_t, float> *tmp_csr;

        float   *recv_A_valuevec;
        int32_t *recv_A_indexvec, *recv_A_ptrvec;
        tmp_csc = &(std::get<mmio::CSC<int32_t, float>>(kokkos_A.dev_mmio));
        int32_t recv_A_nrows = tmp_csc->nrows, recv_A_ncols = tmp_csc->ncols, recv_A_nnz = tmp_csc->nnz;
        D2D_COPY(int32_t, recv_A_ptrvec,   tmp_csc->col_ptr, recv_A_ncols)
        D2D_COPY(int32_t, recv_A_indexvec, tmp_csc->row_idx, recv_A_nnz)
        D2D_COPY(float,   recv_A_valuevec, tmp_csc->val,     recv_A_nnz)

        float   *recv_B_valuevec;
        int32_t *recv_B_ptrvec, *recv_B_indexvec;
        tmp_csr = &(std::get<mmio::CSR<int32_t, float>>(kokkos_B.dev_mmio));
        int32_t recv_B_nrows = tmp_csr->nrows, recv_B_ncols = tmp_csr->ncols, recv_B_nnz = tmp_csr->nnz;
        D2D_COPY(int32_t, recv_B_ptrvec,   tmp_csr->row_ptr, recv_B_nrows)
        D2D_COPY(int32_t, recv_B_indexvec, tmp_csr->col_idx, recv_B_nnz)
        D2D_COPY(float,   recv_B_valuevec, tmp_csr->val,     recv_B_nnz)
        // ------------------------------------------------------------------------------

        // ----- Parse 'receved' row pointers to kokkos structures -----
        using ordinal_view_t = Kokkos::View<int32_t*>;
        using values_view_t  = Kokkos::View<float*>;

        ordinal_view_t colmap(recv_A_ptrvec, recv_A_ncols + 1);
        ordinal_view_t rowidx(recv_A_indexvec, recv_A_nnz);
        values_view_t  valuesA(recv_A_valuevec, recv_A_nnz);

        KokkosSparse::CcsMatrix<float, int32_t, Kokkos::DefaultExecutionSpace, void, int32_t> recv_A("recv_A",
                                        recv_A_nrows, recv_A_ncols, recv_A_nnz,
                                        valuesA,
                                        colmap,
                                        rowidx
        );

        ordinal_view_t rowmap(recv_B_ptrvec, recv_B_nrows + 1);
        ordinal_view_t colidx(recv_B_indexvec, recv_B_nnz);
        values_view_t  valuesB(recv_B_valuevec, recv_B_nnz);

        KokkosSparse::CrsMatrix<float, int32_t, Kokkos::DefaultExecutionSpace, void, int32_t> recv_B("recv_B",
                                        recv_B_nrows, recv_B_ncols, recv_B_nnz,
                                        valuesB,
                                        rowmap,
                                        colidx
        );
        // -------------------------------------------------------------

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
