#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <MatrixMarket_Tpetra.hpp>
#include <TpetraExt_MatrixMatrix.hpp>
#include <Tpetra_MultiVector.hpp>

#include <ccutils/timers.h>
#include <ccutils/cuda/cuda_utils.hpp>
#include <ccutils/mpi/mpi_macros.h>

#include <mpi.h>
#include <string.h>
#include <sstream>
#include <fstream>
#include <vector>

#include "../include/mcl/args.hpp"

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *                          MCL TRILINOS
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
using matrix_t = Tpetra::CrsMatrix<float, int32_t, long long>;
using graph_t = Tpetra::CrsGraph<int32_t, long long>;
using GO = long long;
using SC = float;
using LO = int32_t;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::Comm;
using reader_t = Tpetra::MatrixMarket::Reader<matrix_t>;

/*
 * Prune entries with absolute value <= tol.
 * NOTE: This no longer squares values (inflation is done separately).
 */
RCP<matrix_t> prune_small(RCP<matrix_t>&& A, const float tol) {
    auto rowMap = A->getRowMap();
    const LO nrows = rowMap->getLocalNumElements();
    
    // Single pass: count non-zeros per row while extracting data
    std::vector<size_t> new_rowptrs(nrows, 0);
    std::vector<std::vector<LO>> row_colinds(nrows);
    std::vector<std::vector<SC>> row_values(nrows);
    
    // Pre-allocate to avoid reallocations (assume ~same sparsity)
    for (LO row = 0; row < nrows; ++row) {
        size_t row_nnz = A->getNumEntriesInLocalRow(row);
        row_colinds[row].reserve(row_nnz);
        row_values[row].reserve(row_nnz);
    }
    
    // Single pass through matrix
    for (LO row = 0; row < nrows; ++row) {
        size_t numEntries = A->getNumEntriesInLocalRow(row);
        if (numEntries == 0) continue;
        
        typename matrix_t::nonconst_local_inds_host_view_type indices("inds", numEntries);
        typename matrix_t::nonconst_values_host_view_type values("vals", numEntries);
        
        size_t got = 0;
        A->getLocalRowCopy(row, indices, values, got);
        
        for (size_t j = 0; j < got; ++j) {
            if (std::abs(values(j)) > tol) {
                row_colinds[row].push_back(indices(j));
                row_values[row].push_back(values(j));
            }
        }
        new_rowptrs[row] = row_colinds[row].size();
    }
    
    // Create new matrix with proper sizing
    RCP<matrix_t> Anew = rcp(new matrix_t(
        A->getRowMap(), 
        A->getColMap(), 
        Teuchos::ArrayView<const size_t>(new_rowptrs.data(), new_rowptrs.size())
    ));
    
    // Insert all values (already filtered)
    for (LO row = 0; row < nrows; ++row) {
        if (!row_colinds[row].empty()) {
            Anew->insertLocalValues(
                row, 
                row_colinds[row].size(), 
                row_values[row].data(), 
                row_colinds[row].data()
            );
        }
    }
    
    Anew->fillComplete(A->getDomainMap(), A->getRangeMap());
    
    return Anew;
}

/*
 * Read matrix (unchanged)
 */
RCP<matrix_t> read_trilinos(const char * matpath, RCP<const Comm<int>>& comm) {
    std::ifstream ifs;
    ifs.open(matpath);
    std::string banner;
    std::getline(ifs, banner);
    if (banner.find("pattern") != std::string::npos) {
        RCP<graph_t> A_graph = reader_t::readSparseGraphFile(matpath, comm);
        RCP<matrix_t> A_result = rcp(new matrix_t(A_graph));
        A_result->fillComplete();
        A_result->setAllToScalar((SC)1.0);
        return A_result;
    }

    return reader_t::readSparseFile(matpath, comm);
}

/*
 * Print matrix (unchanged)
 */
void print_matrix(const RCP<matrix_t> &M, const int myid) {
    auto rowMap = M->getRowMap();
    auto colMap = M->getColMap();
    const LO localNumRows = rowMap->getLocalNumElements();

    const GO numGlobalRows = M->getGlobalNumRows();
    const GO numGlobalCols = M->getGlobalNumCols();

    std::vector<std::vector<SC>> localBlock(localNumRows,
                                                std::vector<SC>(numGlobalCols, 0.0));

    for (LO localRow = 0; localRow < localNumRows; ++localRow) {
        size_t numEntries = M->getNumEntriesInLocalRow(localRow);

        using local_inds_view_t = typename matrix_t::nonconst_local_inds_host_view_type;
        using values_view_t     = typename matrix_t::nonconst_values_host_view_type;

        local_inds_view_t indices("indices", numEntries);
        values_view_t     values("values", numEntries);
        size_t actualNumEntries = 0;
        M->getLocalRowCopy(localRow, indices, values, actualNumEntries);

        const GO globalRow = rowMap->getGlobalElement(localRow);

        for (size_t j = 0; j < actualNumEntries; ++j) {
            const GO globalCol = colMap->getGlobalElement(indices(j));
            if (globalCol >= 0 && globalCol < numGlobalCols)
                localBlock[localRow][globalCol] = values(j);
        }
    }

    for (int rank = 0; rank < M->getComm()->getSize(); ++rank) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (myid == rank) {
            for (LO localRow = 0; localRow < localNumRows; ++localRow) {
                const GO globalRow = rowMap->getGlobalElement(localRow);
                printf("[%4lld] ", static_cast<long long>(globalRow));
                for (GO col = 0; col < numGlobalCols; ++col)
                    printf("%8.4f ", localBlock[localRow][col]);
                printf("\n");
            }
            fflush(stdout);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

/*
 * Set diagonal (self loops) to 1.0 in-place.
 * If diagonal position exists, replace the value; otherwise insert it.
 */
void set_self_loops(RCP<matrix_t> &A) {
    auto rowMap = A->getRowMap();
    auto colMap = A->getColMap();
    const LO localNumRows = rowMap->getLocalNumElements();
    SC one = static_cast<SC>(1.0);
    
    // Count entries per row (will be same or +1 if diagonal doesn't exist)
    std::vector<size_t> entriesPerRow(localNumRows);
    std::vector<bool> hasDiag(localNumRows, false);
    
    for (LO lrow = 0; lrow < localNumRows; ++lrow) {
        GO grow = rowMap->getGlobalElement(lrow);
        LO lcol = colMap->getLocalElement(grow);
        
        size_t numEntries = A->getNumEntriesInLocalRow(lrow);
        entriesPerRow[lrow] = numEntries;
        
        if (lcol != Teuchos::OrdinalTraits<LO>::invalid() && numEntries > 0) {
            typename matrix_t::nonconst_local_inds_host_view_type indices("i", numEntries);
            typename matrix_t::nonconst_values_host_view_type values("v", numEntries);
            size_t got = 0;
            A->getLocalRowCopy(lrow, indices, values, got);
            
            for (size_t j = 0; j < got; ++j) {
                if (indices(j) == lcol) {
                    hasDiag[lrow] = true;
                    break;
                }
            }
            
            // If diagonal doesn't exist, we'll need one more entry
            if (!hasDiag[lrow]) {
                entriesPerRow[lrow]++;
            }
        } else if (lcol != Teuchos::OrdinalTraits<LO>::invalid()) {
            // Row is empty but diagonal is valid - add it
            entriesPerRow[lrow] = 1;
        }
    }
    
    // Create new matrix
    RCP<matrix_t> Anew = rcp(new matrix_t(
        rowMap, colMap,
        Teuchos::ArrayView<const size_t>(entriesPerRow.data(), entriesPerRow.size())
    ));
    
    // Fill new matrix
    for (LO lrow = 0; lrow < localNumRows; ++lrow) {
        GO grow = rowMap->getGlobalElement(lrow);
        LO lcol = colMap->getLocalElement(grow);
        
        size_t numEntries = A->getNumEntriesInLocalRow(lrow);
        
        if (numEntries == 0) {
            // Empty row - just add diagonal if valid
            if (lcol != Teuchos::OrdinalTraits<LO>::invalid()) {
                Anew->insertLocalValues(lrow, 1, &one, &lcol);
            }
        } else {
            typename matrix_t::nonconst_local_inds_host_view_type indices("i", numEntries);
            typename matrix_t::nonconst_values_host_view_type values("v", numEntries);
            size_t got = 0;
            A->getLocalRowCopy(lrow, indices, values, got);
            
            std::vector<LO> newIndices;
            std::vector<SC> newValues;
            newIndices.reserve(got + 1);
            newValues.reserve(got + 1);
            
            bool diagInserted = false;
            for (size_t j = 0; j < got; ++j) {
                if (indices(j) == lcol) {
                    // Replace diagonal with 1.0
                    newIndices.push_back(indices(j));
                    newValues.push_back(one);
                    diagInserted = true;
                } else {
                    // Keep existing entry
                    newIndices.push_back(indices(j));
                    newValues.push_back(values(j));
                }
            }
            
            // Add diagonal if it didn't exist
            if (!diagInserted && lcol != Teuchos::OrdinalTraits<LO>::invalid()) {
                newIndices.push_back(lcol);
                newValues.push_back(one);
            }
            
            if (!newIndices.empty()) {
                Anew->insertLocalValues(lrow, newIndices.size(), 
                                       newValues.data(), newIndices.data());
            }
        }
    }
    
    Anew->fillComplete(A->getDomainMap(), A->getRangeMap());
    A = Anew;
}

/*
 * Normalize matrix by columns so that each column sums to 1.
 * This uses: col_sums = A^T * ones, where ones is a vector of ones sized by number of rows.
 */
void normalize_columns(RCP<matrix_t> &A) {
    using vector_t = Tpetra::Vector<SC, LO, GO>;
    using mv_t     = Tpetra::MultiVector<SC, LO, GO>;

    auto rowMap = A->getRowMap();
    auto domainMap = A->getDomainMap();  // This is what you actually need

    // Step 1: compute column sums
    RCP<mv_t> ones = rcp(new mv_t(rowMap, 1));
    ones->putScalar(1.0);

    RCP<mv_t> col_sums_mv = rcp(new mv_t(domainMap, 1));  // Use domainMap!
    A->apply(*ones, *col_sums_mv, Teuchos::TRANS);

    // Step 2: copy and invert values
    RCP<vector_t> col_sums = rcp(new vector_t(domainMap));  // Use domainMap!
    {
        auto mv_h = col_sums_mv->getLocalViewHost(Tpetra::Access::ReadOnly);
        auto vec_h = col_sums->getLocalViewHost(Tpetra::Access::ReadWrite);
        const size_t nLocalCols = vec_h.extent(0);
        for (size_t i = 0; i < nLocalCols; ++i) {
            SC s = mv_h(i,0);
            vec_h(i,0) = (s > 1e-20) ? 1.0 / s : 0.0;
        }
    }

    // Step 3: scale columns
    A->rightScale(*col_sums);
}

/*
 * Inflation: raise each entry to power r (for MCL r is typically >1; here r=2)
 * Then normalize by columns to restore stochastic property.
 * This performs the operation in-place.
 */
void inflation(RCP<matrix_t> &A, int myid, const float power = 2.0f) {
    // Ensure matrix is in correct state at start
    bool wasFillComplete = A->isFillComplete();
    if (wasFillComplete) {
        A->resumeFill();
    }
    
    auto rowMap = A->getRowMap();
    const LO localNumRows = rowMap->getLocalNumElements();

    for (LO lrow = 0; lrow < localNumRows; ++lrow) {
        size_t numEntries = A->getNumEntriesInLocalRow(lrow);
        if (numEntries == 0) continue;
        
        typename matrix_t::nonconst_local_inds_host_view_type indices("inds", numEntries);
        typename matrix_t::nonconst_values_host_view_type values("vals", numEntries);
        
        size_t got = 0;
        A->getLocalRowCopy(lrow, indices, values, got);
        
        for (size_t j = 0; j < got; ++j) {
            values(j) = std::pow(values(j), power);
        }
        
        A->replaceLocalValues(lrow, got, values.data(), indices.data());
    }
    
    // Ensure fillComplete is called before normalize_columns
    if (A->isFillActive()) {
        A->fillComplete(A->getDomainMap(), A->getRangeMap());
    }

    #ifdef DEBUG
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        if(myid==0) printf("--- A_next (after inflation squaring) ---\n");
        print_matrix(A, myid);
    #endif
    
    normalize_columns(A);
}

/*
 * Main MCL routine - updated to use normalize/inflate/prune
 */
void mcl_trilinos(int argc, char *argv[], const int myid, const MLCArgs *args) {
    Tpetra::ScopeGuard scope(&argc, &argv);
    {
        RCP<const Comm<int>> comm = Tpetra::getDefaultComm();
        RCP<matrix_t> A1,A2;
        if(myid==0) fprintf(stdout, "Reading matrix\n");
        A1 = read_trilinos(args->mtx_path, comm);
        A2 = rcp(new matrix_t(*A1));
        if(myid==0) fprintf(stdout, "Matrix read, OK\n");

        // OPTIONAL: set diagonal elements to 1 if requested (CLI flag)
        if (args->add_diag) {
            if(myid==0) fprintf(stdout, "Adding self-loops (diag = 1)\n");
            set_self_loops(A1);
            A2 = rcp(new matrix_t(*A1));
        }
        
        #ifdef DEBUG
            fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD);
            if(myid==0) printf("--- Initial Matrix ---\n");
            print_matrix(A1, myid);
        #endif

        // Normalize initially to create a stochastic matrix (columns sum to 1)
        if(myid==0) fprintf(stdout, "Normalizing initial matrix (columns sum to 1)\n");
        normalize_columns(A1);
        A2 = rcp(new matrix_t(*A1)); // keep A2 consistent

        #ifdef DEBUG
            fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD);
            if(myid==0) printf("--- Initial Normalized Matrix ---\n");
            print_matrix(A1, myid);
        #endif

        std::string timer_prefix = "Timer rank " + std::to_string(myid);
        int iter = 0;
        std::vector<GO> nnz(args->max_iter, 0);
        std::vector<GO> nnz_pruned(args->max_iter, 0);

        // Timer
        CPU_TIMER_DEF(mcl_spgemm)
        CPU_TIMER_INIT(mcl)
        while (iter < args->max_iter) {
            if(myid==0)fprintf(stdout, "\n===================== ITERATION %d ======================\n", iter);
            fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD);

            // Expansion: square the matrix (A_next = A1 * A2)
            CPU_TIMER_START(mcl_spgemm);
            RCP<matrix_t> A_next = rcp(new matrix_t(A1->getRowMap(), 0));
            Tpetra::MatrixMatrix::Multiply(*A1, false, *A2, false, *A_next, true);
            CPU_TIMER_STOP(mcl_spgemm)
            TIMER_PRINT_LAST_WPREFIX_STR(mcl_spgemm, timer_prefix.c_str())
            #ifdef DEBUG
                fflush(stdout);
                MPI_Barrier(MPI_COMM_WORLD);
                if(myid==0) printf("--- A_next (after expansion) ---\n");
                print_matrix(A_next, myid);
            #endif

            nnz[iter] = A_next->getGlobalNumEntries();

            // Inflation: raise entries to power (2) and normalize by column
            inflation(A_next, myid, 2.0f);

            #ifdef DEBUG
                fflush(stdout);
                MPI_Barrier(MPI_COMM_WORLD);
                if(myid==0) printf("--- A_next (after inflation) ---\n");
                print_matrix(A_next, myid);
            #endif

            // Prune small values
            auto A_next_pruned = prune_small(std::move(A_next), args->pruning_tol);
            nnz_pruned[iter] = A_next_pruned->getGlobalNumEntries();
            #ifdef DEBUG
                fflush(stdout);
                MPI_Barrier(MPI_COMM_WORLD);
                if(myid==0) printf("--- A_next pruned ---\n");
                print_matrix(A_next_pruned, myid);
            #endif

            // Prepare for next iteration
            A1 = A_next_pruned;
            A2 = A1; // both point to same matrix for next expansion
            ++iter;

            fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        CPU_TIMER_STOP(mcl)

        if(myid==0) {
            printf("\n===================== Done MCL in %d iterations ======================\n", iter);
            printf("NNZ A_next:            ");
            for (size_t i = 0; i < iter; ++i) printf("%8lu ", nnz[i]);
            printf("\nNNZ A_next post prune: ");
            for (size_t i = 0; i < iter; ++i) printf("%8lu ", nnz_pruned[i]);
            printf("\n");
        }
        FLUSH_WAIT(500000)
        TIMER_PRINT_WPREFIX_STR(mcl, timer_prefix.c_str())
    }
}


int main(int argc, char *argv[]) {
    int myid, ntask;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &ntask);

    MLCArgs *args = (MLCArgs*)malloc(sizeof(MLCArgs));
    parse_args(argc, argv, args);
    if(myid == 0) {
        printf("\n================= TRILINOS MCL ==================\n");
        printf("Matrix: %s\n", args->mtx_name);
        printf("Max iterations: %u\n", args->max_iter);
        printf("Pruning tolerance: %f\n", args->pruning_tol);
        printf("Add diag (self loops): %d\n", args->add_diag);
    }
    FLUSH_WAIT(200000);

    mcl_trilinos(argc, argv, myid, args);

    free(args);
    MPI_Finalize();
    return(EXIT_SUCCESS);
}
