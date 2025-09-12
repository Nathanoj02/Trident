#pragma once
#include "common.h"


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

// template<typename LTT, typename RTT> // BUG with mpi print macro
void LocalTilePrintTriple(std::vector<LocalTile*> tilesC, dmmio::ProcessGrid *gridC,
                        std::vector<RemoteTile*> tilesA, dmmio::ProcessGrid *gridA,
                        std::vector<RemoteTile*> tilesB, dmmio::ProcessGrid *gridB,
                        FILE* fp) 
{

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


template<typename TT>
void LocalTilePrint(std::vector<TT*> tiles, dmmio::ProcessGrid *grid, FILE* fp) {
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
                // if (row == my_row && col == my_col && node == my_node) {
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
