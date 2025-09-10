#include <mpi.h>
#include <iostream>

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

#define OPSTR(X) ((X == dmmio::Operation::None) ? ("None") : ("Transpose") )

typedef struct {
    int rowidx;
    int colidx;
    int flag;
} RemoteTile;

typedef struct {
    int rowidx;
    int colidx;
} LocalTile;

// TMP for debug
typedef struct {
    int rowidx;
    int colidx;
    int nodeidx;
} Recever;

template<typename IT, typename VT>
int LocalTilePrint(dmmio::DCOO<IT, VT> *M, FILE* fp) {
    const dmmio::ProcessGrid *grid = M->partitioning->grid;

    int row_size    = grid->row_size;
    int col_size    = grid->col_size;
    int node_size   = grid->node_size;

    int my_row  = grid->row_rank;
    int my_col  = grid->col_rank;
    int my_node = grid->node_rank;

    // Header row: column labels
    fprintf(fp, "Rank %d (row=%d, col=%d, node=%d)\n", grid->global_rank, my_row, my_col, my_node);
    fprintf(fp, "         ");
    for (int col = 0; col < row_size; ++col) {
        fprintf(fp, " col %-2d ", col);
    }
    fprintf(fp, "\n");

    // Top border
    fprintf(fp, "       ");
    for (int col = 0; col < row_size; ++col) {
        fprintf(fp, " -------");
    }
    fprintf(fp, "\n");

    // For each row
    for (int row = 0; row < col_size; ++row) {
        for (int node = 0; node < node_size; ++node) {
            if (node == 0) {
                fprintf(fp, "row %-2d |", row);
            } else {
                fprintf(fp, "       |");
            }

            for (int col = 0; col < row_size; ++col) {
                if (row == my_row && col == my_col && node == my_node) {
                    fprintf(fp, " XXXXX |");
                } else {
                    fprintf(fp, "       |");
                }
            }
            fprintf(fp, "\n");
        }

        // Separator line
        fprintf(fp, "       ");
        for (int col = 0; col < row_size; ++col) {
            fprintf(fp, " -------");
        }
        fprintf(fp, "\n");
    }

    return 0;
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

    MPI_ALL_PRINT(mmio::utils::COO_print_as_dense(dcoo_A->coo, std::string("Rank ") + std::to_string(world_rank), fp))

    uint32_t local_nnz = dcoo_A->coo->nnz;

    LocalTile  local_C_tile;
    RemoteTile remote_A_tile, remote_B_tile;
    local_C_tile.rowidx = Cgrid->col_rank; // col_rank is the grid's rowidx
    local_C_tile.colidx = Cgrid->row_rank; // row_rank is the grid's colidx
    int common_grd_size = dcoo_A->partitioning->grid->row_size; // == dcoo_B->partitioning->grid->col_size

    // NOTE: this will be fixed during all the computation
    remote_A_tile.rowidx = local_C_tile.rowidx; // Every process will receve only A tails in himself row_comm
    remote_B_tile.colidx = local_C_tile.colidx; // Every process will receve only B tails in himself col_comm

    // This is the initialization stragger, these will be increased each round
    remote_A_tile.colidx = (local_C_tile.colidx + local_C_tile.rowidx) % common_grd_size; // Stragger left
    remote_B_tile.rowidx = (local_C_tile.rowidx + local_C_tile.colidx) % common_grd_size; // Stragger down

    const int n_iters = dcoo_A->partitioning->grid->row_size; // This must be equal to dcoo_B->partitioning->grid->col_size
    for (int iter = 0; iter < n_iters; iter++)
    {
        Recever A_recever, B_recever;
        A_recever.rowidx  = remote_A_tile.rowidx;
        A_recever.colidx  = remote_A_tile.colidx;
        A_recever.nodeidx = Cgrid->node_rank;

        B_recever.rowidx  = remote_B_tile.rowidx;
        B_recever.colidx  = remote_B_tile.colidx;
        B_recever.nodeidx = Cgrid->node_rank;

        /* ======================== Internode communication ======================= */

        // TIMER_START(0);
        // myLocalCSR * B_node = internode_communication(B, peer_process_send, peer_process_recv, iter_row_send, log, csh);
        // TIMER_STOP(0);
        // ADD_TO_MY_TIMER(tim, "mpi_internode", 0);


        /* ======================== Intranode communication ======================= */

        // myLocalCSR * B_final;
        //
        // TIMER_START(0);
        // node_aggregator_all(B_node, grid, &B_final, csh, 1, 1);
        //
        // if (peer_process_send != grid->rank && B_node != NULL)
        //     free_local_csr(B_node);
        //
        // TIMER_STOP(0);
        // ADD_TO_MY_TIMER(tim, "mpi_intranode", 0);

        /* ======================== Local SpGEMM ======================= */

        if (world_rank == 0) fprintf(stdout, "====================================================== Round %d ======================================================\n", iter);

        MPI_ALL_PRINT(
          fprintf(fp, "Process (%d,%d,%d) is performing A(%d,%d,%d) x B(%d,%d,%d)\n",
                  Cgrid->col_rank, Cgrid->row_rank, Cgrid->node_rank,
                  A_recever.rowidx, A_recever.colidx, A_recever.nodeidx,
                  B_recever.rowidx, B_recever.colidx, B_recever.nodeidx
          );
          LocalTilePrint(dcoo_A, fp);
        )

        // NOTE: put kokkos here
        // TIMER_START(0);
        // CUSPARSE_CHECK(cusparseCreateCsr(&(B_final->cusparse_spmat),
        //                                 B_final->nrows, B_final->ncols,
        //                                 B_final->nnz,
        //                                 B_final->d_row, B_final->d_col, B_final->d_values,
        //                                 CUSPARSE_INDEX_32I,
        //                                 CUSPARSE_INDEX_32I,
        //                                 CUSPARSE_INDEX_BASE_ZERO,
        //                                 CUDA_R_32F)); //TODO: Allow different datatypes
        //
        // C_tosend = local_spgemm_reuse(A_loc, B_final, &handle);
        // TIMER_STOP(0);
        // ADD_TO_MY_TIMER(tim, "local_SpGEMM", 0);
        //
        //
        // if (B_final != NULL)
        //     free_local_csr(B_final);

        // Round shift
        remote_A_tile.colidx = (remote_A_tile.colidx + 1) % common_grd_size; // ShiftLeft
        remote_B_tile.rowidx = (remote_B_tile.rowidx + 1) % common_grd_size; // ShiftDown
        MPI_Barrier(MPI_COMM_WORLD);
    }


    delete meta_A;
    delete meta_B;
    dmmio::DCOO_destroy(&dcoo_A);
    dmmio::DCOO_destroy(&dcoo_B);

    MPI_Finalize();
    return(0);
}
