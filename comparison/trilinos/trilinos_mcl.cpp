#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <MatrixMarket_Tpetra.hpp>
#include <TpetraExt_MatrixMatrix.hpp>

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
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *                          MCL TRILINOS
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

RCP<matrix_t> prune_square(RCP<matrix_t>&& A, const float tol) {
    const GO nnz = A->getLocalNumEntries();
    const GO nrows =  A->getRowMap()->getLocalNumElements();

    auto A_loc = A->getLocalMatrixDevice();
    float * values = d2h_copy(A_loc.values.data(), nnz);
    int32_t * colinds = d2h_copy(A_loc.graph.entries.data(), nnz);
    size_t offset = 0;
    GO pruned_nnz = 0;
    std::vector<size_t> new_rowptrs(nrows, 0);
    for (LO row = 0; row < nrows; row++) {
        size_t row_nnz = A->getNumEntriesInLocalRow(row);

        size_t count = 0;
        for (size_t k=0; k<row_nnz; k++) {
            if (abs(values[k + offset]) > tol) {
                new_rowptrs[row] += 1;
            }
            count++;
        }

        offset += row_nnz;
        pruned_nnz += count;
    }

	RCP<matrix_t> Anew = rcp(new matrix_t(A->getRowMap(), A->getColMap(), Teuchos::ArrayView<size_t>(new_rowptrs)));

    offset = 0;
    for (LO row = 0; row < nrows; row++) {
        std::vector<LO> new_colinds;
        std::vector<SC> new_values;
        GO global_row = A->getRowMap()->getGlobalElement(row);
        size_t row_nnz = A->getNumEntriesInLocalRow(row);
        for (size_t k=0; k<row_nnz; k++) {
            if (abs(values[k + offset]) > tol) {
                new_colinds.push_back(colinds[k + offset]);
                new_values.push_back((values[k + offset] * values[k + offset])); // Square
            }
        }

        //Anew->insertGlobalValues(global_row, new_colinds.size(), new_values.data(), new_colinds.data());
        Anew->insertLocalValues(row, new_colinds.size(), new_values.data(), new_colinds.data());

        offset += row_nnz;
    }
    free(colinds);
    free(values);

    //Anew->fillComplete(A->getColMap(), A->getRowMap());
    Anew->fillComplete();

    return Anew;
}

RCP<matrix_t> read_trilinos(const char * matpath, RCP<const Comm<int>>& comm) {
    // First, is it a pattern matrix?
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

void print_matrix(RCP<matrix_t> &M, const int myid) {
    M->fillComplete();
    auto rowMap = M->getRowMap();
    auto colMap = M->getColMap();
    LO localNumRows = rowMap->getLocalNumElements();

    for (LO localRow = 0; localRow < localNumRows; ++localRow) {
        GO globalRow = rowMap->getGlobalElement(localRow);
        size_t numEntries = M->getNumEntriesInLocalRow(localRow);

        // Allocate Kokkos host views
        using local_inds_view_t = typename matrix_t::nonconst_local_inds_host_view_type;
        using values_view_t     = typename matrix_t::nonconst_values_host_view_type;

        local_inds_view_t indices("indices", numEntries);
        values_view_t     values("values", numEntries);

        size_t actualNumEntries = 0;
        M->getLocalRowCopy(localRow, indices, values, actualNumEntries);

        std::vector<SC> row(colMap->getGlobalNumElements(), 0.0);
        for (size_t j = 0; j < actualNumEntries; ++j) {
            GO globalCol = colMap->getGlobalElement(indices(j));
            if (globalCol < (GO)row.size())
                row[globalCol] = values(j);
        }

        if (myid == 0) {
            for (size_t j = 0; j < row.size(); ++j)
                printf("%.6g%s", row[j], (j + 1 == row.size()) ? "\n" : " ");
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void mcl_trilinos(const int myid, const MLCArgs *args) {
    Tpetra::ScopeGuard scope(NULL, NULL);
    {
        RCP<const Comm<int>> comm = Tpetra::getDefaultComm();
        RCP<matrix_t> A1,A2;
        A1 = read_trilinos(args->mtx_path, comm);
        A2 = rcp(new matrix_t(*A1));

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

            // Neighbors 
            CPU_TIMER_START(mcl_spgemm);
            RCP<matrix_t> A_next = rcp(new matrix_t(A1->getRowMap(), 0));
            Tpetra::MatrixMatrix::Multiply(*A1, false, *A2, false, *A_next, true);
            CPU_TIMER_STOP(mcl_spgemm)
            TIMER_PRINT_LAST(mcl_spgemm)
            #ifdef DEBUG
                if(myid==0) printf("--- A_next ---");
                print_matrix(A_next, myid);
            #endif


            nnz[iter] = A_next->getGlobalNumEntries();
            
            auto A_next_pruned = prune_square(std::move(A_next), args->pruning_tol);
            nnz_pruned[iter] = A_next_pruned->getGlobalNumEntries();
            #ifdef DEBUG
                if(myid==0) printf("--- A_next pruned ---");
                print_matrix(A_next_pruned, myid);
            #endif

            A1 = A_next_pruned;
            A2 = A_next_pruned;
            ++iter;
            MPI_Barrier(MPI_COMM_WORLD);
        }
        CPU_TIMER_STOP(mcl)

        if(myid==0) {
            printf("NNZ A_next: ");
            for (size_t i = 0; i < iter; ++i) printf("%lu ", nnz[i]);
            printf("\nNNZ A_next post prune: ");
            for (size_t i = 0; i < iter; ++i) printf("%lu ", nnz_pruned[i]);
            printf("\n===================== Done MCL in %d iterations ======================\n", iter);
        }
        FLUSH_WAIT(500000)
        TIMER_PRINT(mcl)
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
        printf("Pruning tolerance: %s\n", args->pruning_tol);
    }
    FLUSH_WAIT(200000);

    mcl_trilinos(myid, args);

    free(args);
	MPI_Finalize();
	return(EXIT_SUCCESS);
}

// Usage example:
// srun --nodes 1 --qos debug --time 00:01:00 --constraint gpu --ntasks 4 --gpus 4 --account m4646_g comparison/build_trilinos/trilinos_mcl --mtx small_matrices/mcl_and_bfs_test_graph.mtx --pruning_tol 0.01 --max_iter 2