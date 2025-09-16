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
#define TEST(B) (B) ? \
    (fprintf(stdout, "%sPASSED%s\n", GREEN, RESET)) : \
    (fprintf(stderr, "%sNOT PASSED%s\n", RED, RESET)) ;

template <typename KIT, typename DIT, typename VT>
mmio::CSR<KIT, VT>* simulate_csr_comm (KokkosWrap::DistribuitedMatrix<KIT, DIT, VT> kokkos_wrap) {
    mmio::CSR<KIT, VT> *tmp_csr = &(std::get<mmio::CSR<KIT, VT>>(kokkos_wrap.dev_mmio));
    KIT nrows = tmp_csr->nrows, ncols = tmp_csr->ncols, nnz = tmp_csr->nnz;

    mmio::CSR<KIT, VT> *out = mmio::CSR_create<KIT, VT>(nrows, ncols, nnz, true);
    CUDA_CHECK(cudaMemcpy(out->col_idx, tmp_csr->col_idx, sizeof(KIT)* nnz,      cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(out->row_ptr, tmp_csr->row_ptr, sizeof(KIT)*(nrows+1), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(out->val,     tmp_csr->val,     sizeof(VT) * nnz,      cudaMemcpyDeviceToDevice));

    return(out);
}

template <typename KIT, typename DIT, typename VT>
mmio::CSC<KIT, VT>* simulate_csc_comm (KokkosWrap::DistribuitedMatrix<KIT, DIT, VT> kokkos_wrap) {
    mmio::CSC<KIT, VT> *tmp_csr = &(std::get<mmio::CSC<KIT, VT>>(kokkos_wrap.dev_mmio));
    KIT nrows = tmp_csr->nrows, ncols = tmp_csr->ncols, nnz = tmp_csr->nnz;

    mmio::CSC<KIT, VT> *out = mmio::CSC_create<KIT, VT>(nrows, ncols, nnz, true);
    CUDA_CHECK(cudaMemcpy(out->col_ptr, tmp_csr->col_ptr, sizeof(KIT)*(ncols+1), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(out->row_idx, tmp_csr->row_idx, sizeof(KIT)* nnz,      cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(out->val,     tmp_csr->val,     sizeof(VT) * nnz,      cudaMemcpyDeviceToDevice));

    return(out);
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
    dmmio::DCOO<int32_t, float> *dcoo_A = dmmio::DCOO_read<int32_t, float>(
        A_mtx_path.c_str(),
        world_size, world_rank,
        nprocrows, nproccols, nprocpergroup,
        Apart, Aop,
        false, meta_A
    );

    std::string B_mtx_path = (std::string) config->matpathB;
    mmio::Matrix_Metadata *meta_B = new mmio::Matrix_Metadata();
    dmmio::DCOO<int32_t, float> *dcoo_B = dmmio::DCOO_read<int32_t, float>(
        B_mtx_path.c_str(),
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

    MPI_ALL_PRINT(
        mmio::utils::COO_print_as_dense(dcoo_A->coo, "dcoo_A", fp);
        mmio::utils::COO_print_as_dense(dcoo_B->coo, "dcoo_B", fp);
    )

    mmio::CSR<int32_t, float>* h_out_C;
    Kokkos::initialize(argc, argv);
    {
        KokkosWrap::DistribuitedMatrix<int32_t, int32_t, float> kokkos_A(dcoo_A, KokkosWrap::MajorDim::COLS);
        KokkosWrap::DistribuitedMatrix<int32_t, int32_t, float> kokkos_B(dcoo_B, KokkosWrap::MajorDim::ROWS);

        // ----- Simulate a local multiplication only A(i,j)*B(i,j) tiles are multiplied -----

        // Simulate the communication by simply copy the A and B raw pointers
        mmio::CSC<int32_t, float> *tmp_recv_csc = simulate_csc_comm(kokkos_A);
        mmio::CSR<int32_t, float> *tmp_recv_csr = simulate_csr_comm(kokkos_B);

        // Parse 'receved' row pointers to kokkos structures
        KokkosWrap::LocalMatrix<int32_t, int32_t, float> compute_A(tmp_recv_csc);
        KokkosWrap::LocalMatrix<int32_t, int32_t, float> compute_B(tmp_recv_csr);

        // Compute the local spgemm and spadd
        KokkosWrap::LocalMatrix<int32_t, int32_t, float> C;
        KokkosWrap::LocalMatrix<int32_t, int32_t, float>::sp_mma(compute_A, compute_B, C);

        // Performing twice just to test the aggregation
        KokkosWrap::LocalMatrix<int32_t, int32_t, float>::sp_mma(compute_A, compute_B, C);

        // -----------------------------------------------------------------------------------

        // Keep the C raw pointers
        mmio::CSR<int32_t, float> d_out_C = KokkosWrap::rawptr_get(C);

        // Copy C to host (just to dbg)
        h_out_C  = mmio::CSR_create<int32_t, float>(d_out_C.nrows, d_out_C.ncols, d_out_C.nnz, true);
        d2h_copy(h_out_C->row_ptr, d_out_C.nrows+1, d_out_C.row_ptr);
        d2h_copy(h_out_C->col_idx, d_out_C.nnz,     d_out_C.col_idx);
        d2h_copy(h_out_C->val,     d_out_C.nnz,     d_out_C.val);

        MPI_ALL_PRINT(mmio::utils::CSR_print_as_dense(h_out_C, "h_out_C", fp);)

        // kokkos_A and kokkos_B are automatically freed since in scope
    }
    Kokkos::finalize();

    /*  TEST CORRECTNESS BUG (test not works)
     * It can also be in the mmio::DENSE structure/API since I developed them for this

    mmio::DENSE<int32_t, float>* A = mmio::coo2dense(dcoo_A->coo);
    dmmio::DCOO_destroy(&dcoo_A);
    delete meta_A;

    mmio::DENSE<int32_t, float>* B = mmio::coo2dense(dcoo_B->coo);
    dmmio::DCOO_destroy(&dcoo_B);
    delete meta_B;

    mmio::DENSE<int32_t, float>* check_C = matmul(A, B);
    // delete A;
    // delete B;

    mmio::DENSE<int32_t, float>* algo_C = mmio::csr2dense(h_out_C);
    TEST((*algo_C) == (*check_C))
    */

    MPI_Finalize();
    return(0);
}
