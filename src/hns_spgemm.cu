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
#include <vector>
// -----------------------

#include "../include/test_utils.cuh"

#define OPSTR(X) ((X == dmmio::Operation::None) ? ("None") : ("Transpose") )

typedef struct {
    int rowidx;
    int colidx;
    int nodeidx;
    int flag;
} RemoteTile;

typedef struct {
    int rowidx;
    int colidx;
    int nodeidx;
} LocalTile;

template<typename TT>
__host__ int checkTileIncluded (std::vector<TT*> tiles, int rowidx, int colidx, int nodeidx) {
    int flag = 0;
    for (size_t i=0; i<tiles.size(); i++) {
        TT* tile = tiles[i];
        if (tile->rowidx == rowidx && tile->colidx == colidx && tile->nodeidx == nodeidx) {
            flag = 1;
            break;
        }
    }
    return(flag);
}

template<typename TT>
__host__ void LocalTilePrint(std::vector<TT*> tiles, dmmio::ProcessGrid *grid, FILE* fp) {
    int row_size    = grid->row_size;
    int col_size    = grid->col_size;
    int node_size   = grid->node_size;

    // Header row: column labels
    fprintf(fp, "Rank %d ", grid->global_rank);
    for (size_t i=0; i<tiles.size(); i++) {
        TT* tile = tiles[i];
        fprintf(fp, " (row=%d, col=%d, node=%d)", tile->rowidx, tile->colidx, tile->nodeidx);
    }
    fprintf(fp, "\n");

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
                if (checkTileIncluded(tiles, row, col, node)) {
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
}

// template<typename LTT, typename RTT> // BUG with mpi print macro
__host__
void LocalTilePrintTriple(std::vector<LocalTile*> tilesC, dmmio::ProcessGrid *gridC,
                        std::vector<RemoteTile*> tilesA, dmmio::ProcessGrid *gridA,
                        std::vector<RemoteTile*> tilesB, dmmio::ProcessGrid *gridB,
                        FILE* fp) {

    // Print header
    fprintf(fp, "C +=  A x B\n\n");

    // Extract grid sizes
    int row_sizeC  = gridC->row_size;
    int col_sizeC  = gridC->col_size;
    int node_sizeC = gridC->node_size;

    int row_sizeA  = gridA->row_size;
    int col_sizeA  = gridA->col_size;
    int node_sizeA = gridA->node_size;

    int row_sizeB  = gridB->row_size;
    int col_sizeB  = gridB->col_size;
    int node_sizeB = gridB->node_size;

    // For simplicity, assume all have the same dimensions
    int row_size   = row_sizeC;
    int col_size   = col_sizeC;
    int node_size  = node_sizeC;

    // Header row: only for C
    fprintf(fp, "         ");
    for (int col = 0; col < row_size; ++col) {
        fprintf(fp, " col %-2d ", col);
    }
    fprintf(fp, "      ");
    for (int col = 0; col < row_size; ++col) {
        fprintf(fp, " col %-2d ", col);
    }
    fprintf(fp, "      ");
    for (int col = 0; col < row_size; ++col) {
        fprintf(fp, " col %-2d ", col);
    }
    fprintf(fp, "\n");

    // Top border
    auto print_border = [&](int count) {
        fprintf(fp, "       ");
        for (int col = 0; col < count; ++col) {
            fprintf(fp, " -------");
        }
    };

    print_border(row_size);
    print_border(row_size);
    print_border(row_size);
    fprintf(fp, "\n");

    // For each row
    for (int row = 0; row < col_size; ++row) {
        for (int node = 0; node < node_size; ++node) {
            // ---- C ----
            if (node == 0) {
                fprintf(fp, "row %-2d |", row);
            } else {
                fprintf(fp, "       |");
            }
            for (int col = 0; col < row_size; ++col) {
                if (checkTileIncluded(tilesC, row, col, node)) {
                    fprintf(fp, " XXXXX |");
                } else {
                    fprintf(fp, "       |");
                }
            }

            fprintf(fp, "      ");

            // ---- A ---- (no row labels)
            fprintf(fp, "|");
            for (int col = 0; col < row_size; ++col) {
                if (checkTileIncluded(tilesA, row, col, node)) {
                    fprintf(fp, " XXXXX |");
                } else {
                    fprintf(fp, "       |");
                }
            }

            fprintf(fp, "      ");

            // ---- B ---- (no row labels)
            fprintf(fp, "|");
            for (int col = 0; col < row_size; ++col) {
                if (checkTileIncluded(tilesB, row, col, node)) {
                    fprintf(fp, " XXXXX |");
                } else {
                    fprintf(fp, "       |");
                }
            }

            fprintf(fp, "\n");
        }

        // Separator lines
        print_border(row_size);
        print_border(row_size);
        print_border(row_size);
        fprintf(fp, "\n");
    }
}

__host__
void print_comm_info (MPI_Comm comm, FILE *fp) {
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    char name[MPI_MAX_OBJECT_NAME];
    int name_length;
    MPI_Comm_get_name(comm, name, &name_length);
    fprintf(fp, "The communicator %s contains %d MPI processes.\n", name, comm_size);
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

    // MPI_ALL_PRINT(mmio::utils::COO_print_as_dense(dcoo_A->coo, std::string("Rank ") + std::to_string(world_rank), fp))

    uint32_t local_nnz = dcoo_A->coo->nnz;

    int node_size = Cgrid->node_size; // NOTE: every grid must have the same node size!!
    RemoteTile remote_A_tile, remote_B_tile;
    LocalTile  local_C_tile, local_A_tile, local_B_tile; // A and be will be exposed with RDMA
    int common_grd_size = dcoo_A->partitioning->grid->row_size; // == dcoo_B->partitioning->grid->col_size
    local_C_tile.rowidx  = Cgrid->col_rank;  // col_rank is the grid's rowidx
    local_C_tile.colidx  = Cgrid->row_rank;  // row_rank is the grid's colidx
    local_C_tile.nodeidx = Cgrid->node_rank;

    local_A_tile.rowidx  = dcoo_A->partitioning->grid->col_rank;  // col_rank is the grid's rowidx
    local_A_tile.colidx  = dcoo_A->partitioning->grid->row_rank;  // row_rank is the grid's colidx
    local_A_tile.nodeidx = dcoo_A->partitioning->grid->node_rank;

    local_B_tile.rowidx  = dcoo_B->partitioning->grid->col_rank;  // col_rank is the grid's rowidx
    local_B_tile.colidx  = dcoo_B->partitioning->grid->row_rank;  // row_rank is the grid's colidx
    local_B_tile.nodeidx = dcoo_B->partitioning->grid->node_rank;

    int *tmp_row_recv_buff = (int*)malloc(sizeof(int)*dcoo_A->partitioning->grid->row_size);
    int *tmp_col_recv_buff = (int*)malloc(sizeof(int)*dcoo_B->partitioning->grid->col_size);
    int *tmp_node_recv_buff = (int*)malloc(sizeof(int)*Cgrid->node_size);
    MPI_Allgather(&(Cgrid->global_rank), 1, MPI_INT, tmp_row_recv_buff, 1, MPI_INT, dcoo_A->partitioning->grid->row_comm);
    MPI_Allgather(&(Cgrid->global_rank), 1, MPI_INT, tmp_col_recv_buff, 1, MPI_INT, dcoo_B->partitioning->grid->col_comm);
    MPI_Allgather(&(Cgrid->global_rank), 1, MPI_INT, tmp_node_recv_buff, 1, MPI_INT, Cgrid->node_comm);

    /*
    MPI_ALL_PRINT(
        if (dcoo_B->partitioning->grid->col_rank == 0) {
            fprintf(fp, "Ranks in col_comm: ");
            for (int i=0; i<dcoo_B->partitioning->grid->col_size; i++) fprintf(fp, " %d", tmp_col_recv_buff[i]);
            fprintf(fp, "\n");
        }
        if (dcoo_A->partitioning->grid->row_rank == 0) {
            fprintf(fp, "Ranks in row_comm: ");
            for (int i=0; i<dcoo_A->partitioning->grid->row_size; i++) fprintf(fp, " %d", tmp_row_recv_buff[i]);
            fprintf(fp, "\n");
        }
        if (Cgrid->node_rank == 0) {
            fprintf(fp, "Ranks in node_comm: ");
            for (int i=0; i<Cgrid->node_size; i++) fprintf(fp, " %d", tmp_node_recv_buff[i]);
            fprintf(fp, "\n");
        }
    )*/

    MPI_Win win_A, win_B;
    MPI_Win_create(&remote_A_tile, sizeof(RemoteTile), sizeof(RemoteTile), MPI_INFO_NULL, dcoo_A->partitioning->grid->row_comm, &win_A);
    MPI_Win_fence(0, win_A);
    MPI_Win_create(&remote_B_tile, sizeof(RemoteTile), sizeof(RemoteTile), MPI_INFO_NULL, dcoo_B->partitioning->grid->col_comm, &win_B);
    MPI_Win_fence(0, win_B);

    int psend_A, psend_B;
    MPI_Win win_psend_A, win_psend_B;
    int MPI_result = MPI_Win_create(&psend_A, sizeof(int), sizeof(int), MPI_INFO_NULL, dcoo_A->partitioning->grid->row_comm, &win_psend_A);
    if (MPI_result != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, __LINE__);
    MPI_Win_fence(0, win_psend_A);
    MPI_result = MPI_Win_create(&psend_B, sizeof(int), sizeof(int), MPI_INFO_NULL, dcoo_B->partitioning->grid->col_comm, &win_psend_B);
    if (MPI_result != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, __LINE__);
    MPI_Win_fence(0, win_psend_B);
    MPI_Barrier(MPI_COMM_WORLD);

    // NOTE: only for debug print
    std::vector<RemoteTile> allAtiles, allBtiles;
    std::vector<LocalTile*> allCtiles_ptr;
    allCtiles_ptr.push_back(&local_C_tile);
    // --------------------------

    /* remote_A_tile.rowidx and remote_B_tile.colidx are fixed since each process only require A tails and B tails which are respectivelly
     * on is own same row and column. Moreover, both remote_A_tile.nodeidx and remote_B_tile.nodeidx are fixed and same to Cgrid->node_rank;
     * the only difference is that with the intranode communication I will also gat remote_B_tile from the other nodeidx.
     *
     * The only value that will change iteration by iteration will be the remote_A_tile.colidx and remote_B_tile.rowidx.
     */
    int colAtoGet = (local_C_tile.colidx + local_C_tile.rowidx) % common_grd_size; // Stragger left
    int rowBtoGet = (local_C_tile.rowidx + local_C_tile.colidx) % common_grd_size; // Stragger down

    const int n_iters = dcoo_A->partitioning->grid->row_size; // This must be equal to dcoo_B->partitioning->grid->col_size
    for (int iter = 0; iter < n_iters; iter++)
    {
        /* ======================== Internode communication ======================= */

        // TIMER_START(0);
        // myLocalCSR * B_node = internode_communication(B, peer_process_send, peer_process_recv, iter_row_send, log, csh);
        // TIMER_STOP(0);
        // ADD_TO_MY_TIMER(tim, "mpi_internode", 0);

        MPI_Put(&(Cgrid->row_rank), 1, MPI_INT, colAtoGet, 0, 1, MPI_INT, win_psend_A);
        MPI_Win_fence(0, win_psend_A);
        MPI_Put(&(Cgrid->col_rank), 1, MPI_INT, rowBtoGet, 0, 1, MPI_INT, win_psend_B);
        MPI_Win_fence(0, win_psend_B);

        MPI_Barrier(MPI_COMM_WORLD);

        RemoteTile fetchTile;
        fetchTile.rowidx = local_A_tile.rowidx;
        fetchTile.colidx = local_A_tile.colidx;
        fetchTile.nodeidx = local_A_tile.nodeidx;
        MPI_Put(&fetchTile, sizeof(RemoteTile), MPI_BYTE, psend_A, 0, sizeof(RemoteTile), MPI_BYTE, win_A);
        MPI_Win_fence(0, win_A);

        fetchTile.rowidx = local_B_tile.rowidx;
        fetchTile.colidx = local_B_tile.colidx;
        fetchTile.nodeidx = local_B_tile.nodeidx;
        MPI_Put(&fetchTile, sizeof(RemoteTile), MPI_BYTE, psend_B, 0, sizeof(RemoteTile), MPI_BYTE, win_B);
        MPI_Win_fence(0, win_B);
        MPI_Barrier(MPI_COMM_WORLD);


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

        RemoteTile *intranode_recv_buff = (RemoteTile*)malloc(sizeof(RemoteTile)*node_size);
        MPI_Allgather(&remote_B_tile, sizeof(RemoteTile), MPI_BYTE, intranode_recv_buff, sizeof(RemoteTile), MPI_BYTE, dcoo_B->partitioning->grid->node_comm);

        /* ======================== Local SpGEMM ======================= */

        if (world_rank == 0) fprintf(stdout, "====================================================== Round %d ======================================================\n", iter);

        std::vector<LocalTile*>  Ctiles = {&local_C_tile};
        std::vector<RemoteTile*> Atiles = {&remote_A_tile};
        std::vector<RemoteTile*> Btiles;
        for (int i=0; i<node_size; i++) Btiles.push_back(&(intranode_recv_buff[i]));
        FILE *fp = stdout;
        MPI_PROCESS_PRINT(MPI_COMM_WORLD, Cgrid->global_size - 1,
        // MPI_ALL_PRINT(
          fprintf(fp, "[%d] iter %d, psend_A: %d, psend_B: %d\n", Cgrid->global_rank, iter, psend_A, psend_B);
          fprintf(fp, "Process (%d,%d,%d) is performing A(%d,%d,%d) x B(%d,%d,%d)\n",
                  Cgrid->col_rank, Cgrid->row_rank, Cgrid->node_rank,
                  remote_A_tile.rowidx, remote_A_tile.colidx, remote_A_tile.nodeidx,
                  remote_B_tile.rowidx, remote_B_tile.colidx, remote_B_tile.nodeidx
          );
          // LocalTilePrint<RemoteTile>(Atiles, Cgrid, fp);
          LocalTilePrintTriple(Ctiles, Cgrid,
                        Atiles, dcoo_A->partitioning->grid,
                        Btiles, dcoo_B->partitioning->grid,
                        fp);
        )
        MPI_Barrier(MPI_COMM_WORLD);

        allAtiles.push_back(remote_A_tile);
        for (int i=0; i<node_size; i++) allBtiles.push_back(intranode_recv_buff[i]);

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

        free(intranode_recv_buff);

        // Round shift
        colAtoGet = (colAtoGet + 1) % common_grd_size; // ShiftLeft
        rowBtoGet = (rowBtoGet + 1) % common_grd_size; // ShiftDown
        MPI_Barrier(MPI_COMM_WORLD);
    }

    std::vector<RemoteTile*> allAtiles_ptr, allBtiles_ptr;
    for (int i=0; i<allAtiles.size(); i++) allAtiles_ptr.push_back(&(allAtiles[i]));
    for (int i=0; i<allBtiles.size(); i++) allBtiles_ptr.push_back(&(allBtiles[i]));
    MPI_ALL_PRINT(
        fprintf(fp, "All computed multiplications by process (%d,%d,%d)\n",
                Cgrid->col_rank, Cgrid->row_rank, Cgrid->node_rank
        );
        LocalTilePrintTriple(allCtiles_ptr, Cgrid,
                    allAtiles_ptr, dcoo_A->partitioning->grid,
                    allBtiles_ptr, dcoo_B->partitioning->grid,
                    fp);
    )
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_free(&win_A);
    MPI_Win_free(&win_B);

    delete meta_A;
    delete meta_B;
    dmmio::DCOO_destroy(&dcoo_A);
    dmmio::DCOO_destroy(&dcoo_B);

    MPI_Finalize();
    return(0);
}
