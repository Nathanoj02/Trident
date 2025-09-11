#include "hns_spgemm.h"
#include "message_queue.cuh"
#include "tile_holder.cuh"

MPIDataTypeCache mpidtc; //fix linker error

// template<typename LTT, typename RTT> // BUG with mpi print macro
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
                // if (row == tileC->rowidx && col == tileC->colidx && node == tileC->nodeidx) {
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
                // if (row == tileA->rowidx && col == tileA->colidx && node == tileA->nodeidx) {
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
                // if (row == tileB->rowidx && col == tileB->colidx && node == tileB->nodeidx) {
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

void print_comm_info (MPI_Comm comm, FILE *fp) {
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    char name[MPI_MAX_OBJECT_NAME];
    int name_length;
    MPI_Comm_get_name(comm, name, &name_length);
    fprintf(fp, "The communicator %s contains %d MPI processes.\n", name, comm_size);
}


template <typename IT, typename VT>
dmmio::DCOO<IT, VT> * hns_spgemm_main(dmmio::DCOO<IT, VT>* dcoo_A, dmmio::DCOO<IT, VT>* dcoo_B) 
{

    // TODO: check all the grids are the same or compatible (do we support rectangular grids?)
    dmmio::ProcessGrid *Cgrid = dcoo_A->partitioning->grid; // NOTE BUG TODO: we nead to think something for the grid managing.
    dmmio::utils::ProcessGrid_graph(dcoo_A->partitioning->grid, stdout);
    MPI_Barrier(MPI_COMM_WORLD);

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

    int colAtoGet = (local_C_tile.colidx + local_C_tile.rowidx) % common_grd_size; // Stragger left
    int rowBtoGet = (local_C_tile.rowidx + local_C_tile.colidx) % common_grd_size; // Stragger down


    const int n_iters = dcoo_A->partitioning->grid->row_size; // This must be equal to dcoo_B->partitioning->grid->col_size
    

    // Message queue setup -- these will contain indices of the processes that request tiles of A and tiles of B
    MessageQueue<int> A_queue(n_iters, Cgrid->row_comm); 
    MessageQueue<int> B_queue(n_iters, Cgrid->col_comm); 


    // Get max nnz for A and B tiles 
    IT A_max_nnz = dcoo_A->coo->nnz;
    MPI_Allreduce(MPI_IN_PLACE, &A_max_nnz, 1, MPIType<IT>(), MPI_MAX, Cgrid->row_comm);

    IT B_max_nnz = dcoo_B->coo->nnz;
    MPI_Allreduce(MPI_IN_PLACE, &B_max_nnz, 1, MPIType<IT>(), MPI_MAX, Cgrid->col_comm);


    // Tile holders for A and B -- these are buffers that remote processes will write tiles to
    TileHolder<IT, VT> A_holder(dcoo_A->partitioning->local_rows, (IT)A_max_nnz*1.5, Cgrid->row_comm);
    TileHolder<IT, VT> B_holder(dcoo_B->partitioning->local_rows, (IT)B_max_nnz*1.5, Cgrid->col_comm);


    for (int iter = 0; iter < n_iters; iter++)
    {

        RemoteTile fetchTile;
        fetchTile.rowidx = local_A_tile.rowidx;
        fetchTile.colidx = local_A_tile.colidx;
        fetchTile.nodeidx = local_A_tile.nodeidx;

        fetchTile.rowidx = local_B_tile.rowidx;
        fetchTile.colidx = local_B_tile.colidx;
        fetchTile.nodeidx = local_B_tile.nodeidx;

        // Round shift
        colAtoGet = (colAtoGet + 1) % common_grd_size; // ShiftLeft
        rowBtoGet = (rowBtoGet + 1) % common_grd_size; // ShiftDown
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

}

template dmmio::DCOO<uint32_t, float> * hns_spgemm_main<uint32_t, float>(dmmio::DCOO<uint32_t, float>* dcoo_A, dmmio::DCOO<uint32_t, float>* dcoo_B);
