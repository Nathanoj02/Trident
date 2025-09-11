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
int checkTileIncluded (std::vector<TT*> tiles, int rowidx, int colidx, int nodeidx) {
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


template <typename IT, typename VT>
dmmio::DCOO<IT, VT> * hns_spgemm_main(dmmio::DCOO<IT, VT>* dcoo_A, 
                     dmmio::DCOO<IT, VT>* dcoo_B);
