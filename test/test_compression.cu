#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_adjacent_difference.cuh>
#include <iostream>

#include <cub/cub.cuh>

#include <vector>
#include <iomanip>

#include "common.h"
#include "utils.cuh"
#include "test_utils.cuh"
#include "spacomm.cuh"

#define DEBUG

// -------------------
// For the compression
// -------------------

template<typename IT, typename VT>
void printCSRMatrix(const std::vector<IT>& colIdx,
                    const std::vector<VT>& values,
                    const std::vector<IT>& rowPtr) {
    int nrows = rowPtr.size() - 1;
    int ncols = 0;

    // Find maximum column index to determine number of columns
    for (int c : colIdx) {
        if (c > ncols) ncols = c;
    }
    ncols += 1; // since indices start at 0

    for (int r = 0; r < nrows; r++) {
        std::vector<std::string> row(ncols, " - ");

        // Fill row with values
        for (int j = rowPtr[r]; j < rowPtr[r + 1]; j++) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1) << values[j]; // e.g. "3.0"
            row[colIdx[j]] = ss.str();
        }

        // Print row
        for (int c = 0; c < ncols; c++) {
            std::cout << std::setw(4) << row[c] << " ";
        }
        std::cout << "\n";
    }
}

template<typename IT, typename VT>
void printCSCMatrix(const std::vector<IT>& rowIdx,
                    const std::vector<VT>& values,
                    const std::vector<IT>& colPtr) {
    int ncols = colPtr.size() - 1;
    int nrows = 0;

    // Find maximum row index to determine number of rows
    for (int r : rowIdx) {
        if (r > nrows) nrows = r;
    }
    nrows += 1; // since indices start at 0

    for (int r = 0; r < nrows; r++) {
        std::vector<std::string> row(ncols, " - ");

        // Fill row r with values
        for (int c = 0; c < ncols; c++) {
            for (int j = colPtr[c]; j < colPtr[c + 1]; j++) {
                if (rowIdx[j] == r) {
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(1) << values[j];
                    row[c] = ss.str();
                }
            }
        }

        // Print row
        for (int c = 0; c < ncols; c++) {
            std::cout << std::setw(4) << row[c] << " ";
        }
        std::cout << "\n";
    }
}


template<typename T>
void printEntriesByRow(const thrust::host_vector<T>& colIdx,
                       const thrust::host_vector<int>& rowPtr) {
    int nrows = rowPtr.size() - 1;
    for (int r = 0; r < nrows; r++) {
        for (int j = rowPtr[r]; j < rowPtr[r + 1]; j++) {
            std::cout << colIdx[j] << " ";
        }
        if (r < nrows - 1) std::cout << "| ";
    }
    std::cout << "\n";
}

int tmp_test1 () {
    // Example input
    thrust::host_vector<float> h_val{1.0,
                                   2.0,   3.0,  4.0,
                                   5.0,   6.0,  7.0,  8.0,  9.0, 10.0,
                                   11.0, 12.0, 13.0, 14.0, 15.0,
                                   16.0, 17.0, 18.0, 19.0, 20.0,
                                   21.0, 22.0,
                                   23.0, 24.0, 25.0,
                                   26.0, 27.0
    };
    thrust::host_vector<int> h_col{0,
                                 1, 3, 7,
                                 0, 1, 4, 7, 8, 10,
                                 0, 2, 4, 6, 8,
                                 1, 3, 5, 7, 9,
                                 2, 8,
                                 1, 5, 7,
                                 0, 10
    };
    thrust::host_vector<int> h_row{0, 1, 4, 10, 15, 20, 22, 25, 27}; // 8 segments
    thrust::host_vector<int> h_mask(2); // 8 bits total (2 words * MASK_SIZE=4)

    printCSRMatrix(std::vector<int>(h_col.begin(), h_col.end()),
                   std::vector<float>(h_val.begin(), h_val.end()),
                   std::vector<int>(h_row.begin(), h_row.end())
    );
    printEntriesByRow(h_col, h_row);
    printEntriesByRow(h_val, h_row);
    fflush(stdout);

    // condition mask:
    // let's enable segments {0,1,4,7} for fun
    // word0 covers segments [0..3], word1 covers [4..7]
    h_mask[0] = (1 << 0) | (1 << 1);       // bits 0 and 1 set
    h_mask[1] = (1 << 0) | (1 << 3);       // bits 4 and 7 set

    int n = h_col.size();
    int m = h_row.size();

    std::cout << "Input vector v: ";
    for (int x : h_col) std::cout << x << " ";
    std::cout << "\nSearch vector s: ";
    for (int x : h_row) std::cout << x << " ";
    std::cout << "\nBitmask vector c: ";
    for (int x : h_mask) std::cout << x << " ";
    std::cout << "\n";
    fflush(stdout);

    // Copy to device
    thrust::device_vector<float> d_val  = h_val;
    thrust::device_vector<int>   d_col  = h_col;
    thrust::device_vector<int>   d_row  = h_row;
    thrust::device_vector<BMASK_TYPE>   d_mask = h_mask;
    // return(0); // DEBUG

    int* new_col;
    int num_selected = SpaComm::select_entries<int, int>(
            thrust::raw_pointer_cast(d_col.data()), n,
            thrust::raw_pointer_cast(d_row.data()), m,
            thrust::raw_pointer_cast(d_mask.data()),
            &new_col
    );
    {
        int *tmp = (int*)malloc(sizeof(int)*num_selected);
        cudaMemcpy(tmp, new_col, sizeof(int)*num_selected, cudaMemcpyDeviceToHost);

        std::cout << "[out] Selected " << num_selected << " values:\n";
        for (int i=0; i<num_selected; i++) std::cout << tmp[i] << " ";
        std::cout << "\n";
        free(tmp);
    }
    fflush(stdout);

    float* new_val;
    int num_selected_val = SpaComm::select_entries<float, int>(
            thrust::raw_pointer_cast(d_val.data()), n,
            thrust::raw_pointer_cast(d_row.data()), m,
            thrust::raw_pointer_cast(d_mask.data()),
            &new_val
    );
    {
        float *tmp = (float*)malloc(sizeof(float)*num_selected_val);
        cudaMemcpy(tmp, new_val, sizeof(float)*num_selected_val, cudaMemcpyDeviceToHost);

        std::cout << "[out] Selected " << num_selected_val << " values:\n";
        for (int i=0; i<num_selected_val; i++) std::cout << tmp[i] << " ";
        std::cout << "\n";
        free(tmp);
    }
    fflush(stdout);

    if (num_selected != num_selected_val) {
        fprintf(stderr, "Error: num_selected_val (%d) differ from num_selected (%d)!\n", num_selected_val, num_selected);
        exit(__LINE__);
    }

    // Part for row pointer
    int *new_row = SpaComm::select_ptrs(thrust::raw_pointer_cast(d_row.data()), m, thrust::raw_pointer_cast(d_mask.data()));
    fflush(stdout);

    // return(0); // DEBUG

    std::cout << "Test at line " << __LINE__ << std::endl; fflush(stdout);

    thrust::device_ptr<float> d_val_ptr(new_val);
    thrust::device_ptr<int>   d_col_ptr(new_col);
    thrust::device_ptr<int>   d_row_ptr(new_row);

    std::cout << "Test at line " << __LINE__ << std::endl; fflush(stdout);

    std::cout << "new_val ptr: " << new_val << ", new_col ptr: " << new_col
          << ", d_row ptr: " << new_row
          << ", num_selected: " << num_selected
          << ", m: " << m << std::endl;

    // Copy to host vectors
    thrust::host_vector<float> h_new_val(num_selected);
    thrust::host_vector<int>   h_new_col(num_selected);
    thrust::host_vector<int>   h_new_row(m);

    thrust::copy(d_val_ptr, d_val_ptr + num_selected, h_new_val.begin());
    thrust::copy(d_col_ptr, d_col_ptr + num_selected, h_new_col.begin());
    thrust::copy(d_row_ptr, d_row_ptr + m, h_new_row.begin());

    printCSRMatrix(std::vector<int>(h_new_col.begin(), h_new_col.end()),
                   std::vector<float>(h_new_val.begin(), h_new_val.end()),
                   std::vector<int>(h_new_row.begin(), h_new_row.end())
    );
    fflush(stdout);

    return 0;
}

bool div(int i, int j, int k) { return((i%j)==k); }
bool even(int i) { return(div(i, 2, 0)); }
bool  odd(int i) { return(div(i, 2, 1)); }
bool zeroOthree(int i) { return(div(i, 3, 0)); }
bool oneOthree(int i)  { return(div(i, 3, 1)); }
bool twoOthree(int i)  { return(div(i, 3, 2)); }

typedef bool(*ConditionFn)(int);
ConditionFn syntetic_selections(int i) {

}

template<typename IT, typename VT>
mmio::CSX<IT, VT>* gen_syntetic_matrix(int seed, int n, mmio::MajorDim layout, bool print=false) {

    bool(*empty_row_condition)(int);
    switch(seed) {
        case 0:
            empty_row_condition = even;
            break;
        case 1:
            empty_row_condition = odd;
            break;
        case 2:
            empty_row_condition = twoOthree;
            break;
        case 3:
            empty_row_condition = oneOthree;
            break;
        case 4:
            empty_row_condition = zeroOthree;
            break;
        default:
            fprintf(stderr, "Unrecognized seed (%d)\n", seed);
            exit(__LINE__);
    }

    std::vector<IT> vec_ptr(n+1);
    vec_ptr[0] = 0;
    IT nnz = 0;

    for (int i=0; i<n; i++) {
        if (empty_row_condition(i)) {
            IT row_nnz = (i<(n/2)) ? (i+1) : (n-i);
            vec_ptr[i+1] = vec_ptr[i] + row_nnz;
            nnz += row_nnz;
        } else {
            vec_ptr[i+1] = vec_ptr[i];
        }
    }

    std::vector<IT> vec_idx(nnz);
    std::vector<VT> vec_val(nnz);
    for (int i=0; i<n; i++) {
        for (int j=0; j<(vec_ptr[i+1] - vec_ptr[i]); j++) {
            vec_idx[vec_ptr[i] + j] = (i<(n/2)) ? (i + j) : (n - j - 1) ;
            vec_val[vec_ptr[i] + j] = (vec_ptr[i] + j) * 1.0 ;
        }
    }

    if (print) {
        std::cout << "---------- Syntetic matrix ( " << seed << ", " << n << ") ----------" << std::endl;

        if (layout == mmio::MajorDim::ROWS) {
            std::cout << "----- CSR -----" << std::endl;
            printCSRMatrix(vec_idx, vec_val, vec_ptr);
        } else {
            std::cout << "----- CSC -----" << std::endl;
            printCSCMatrix(vec_idx, vec_val, vec_ptr);
        }
    }

    mmio::CSX<IT,VT> *csx = (mmio::CSX<IT,VT>*)malloc(sizeof(mmio::CSX<IT,VT>));
    csx->majordim = layout;
    csx->nnz      = nnz;
    csx->nrows    = n;
    csx->ncols    = n;
    csx->val     = (VT*)malloc(nnz * sizeof(VT));
    csx->idx_vec = (IT*)malloc(nnz * sizeof(IT));
    csx->ptr_vec = (IT*)malloc((n+1) * sizeof(IT));

    std::copy(vec_val.begin(), vec_val.end(), csx->val);
    std::copy(vec_idx.begin(), vec_idx.end(), csx->idx_vec);
    std::copy(vec_ptr.begin(), vec_ptr.end(), csx->ptr_vec);

    return(csx);

}

int check_mod_pattern(const BMASK_TYPE* mask, int n, int k, int m) {
    for (int byte_idx = 0; byte_idx < n; ++byte_idx) {
        BMASK_TYPE expected = 0;
        // Build expected pattern for this byte
        for (int bit = 0; bit < 8; ++bit) {
            int global_bit = byte_idx * 8 + bit;
            if ((global_bit % m) == k) {
                expected |= (1 << bit);
            }
        }
        // Compare against actual
        if (mask[byte_idx] != expected) {
            return 0;
        }
    }
    return 1;
}

#define NGPU_PER_NODE 4

int main(int argc, char ** argv) {

    MPI_Init(&argc, &argv);

    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // cudaSetDevice(world_rank%NGPU_PER_NODE);
    MPI_Barrier(MPI_COMM_WORLD);

    // MPI_Abort(MPI_COMM_WORLD, 1); // Debug

    Config * config = (Config *)(malloc(sizeof(Config)));
    parse_args(argc, argv, config);
    MPI_Barrier(MPI_COMM_WORLD);

    // MPI_Abort(MPI_COMM_WORLD, 1); // Debug

    // Set the process grid parameeters
    int nprocrows = config->nprocrows, nproccols = config->nproccols, nprocpergroup = world_size/(nprocrows*nproccols);
    dmmio::ProcessGrid *grid = dmmio::io::ProcessGrid_create(nprocrows, nproccols, nprocpergroup);

    // MPI_Abort(MPI_COMM_WORLD, 1); // Debug

    int skip = config->spcomm; // I use spcomm as flag for skip the first experiments
    if (skip) goto parallel_exps;
    // ----------------------------------------------------------------------------------------------------

   if (world_rank == 0) {
        thrust::device_vector<int> test_vector{0, 0, 1, 1, 3, 3, 6, 6};

        thrust::device_vector<int> A_rowptr{0, 1, 2, 2, 3, 4, 4, 5, 6};
        thrust::device_vector<int> B_colptr{0, 0, 1, 2, 2, 3, 4, 4, 5};

        // Apply the same function to both
        BMASK_TYPE *result_test = SpaComm::gen_bitmask(thrust::raw_pointer_cast(test_vector.data()), test_vector.size()-1, MASK_SIZE);
        // return(0); // MPI_Abort(MPI_COMM_WORLD, 1); // Debug

        BMASK_TYPE *resultA     = SpaComm::gen_bitmask(thrust::raw_pointer_cast(A_rowptr.data()), A_rowptr.size()-1, MASK_SIZE);
        BMASK_TYPE *resultB     = SpaComm::gen_bitmask(thrust::raw_pointer_cast(B_colptr.data()), B_colptr.size()-1, MASK_SIZE);

        int num_segments_test = ((test_vector.size()-1)%MASK_SIZE == 0) ? ((test_vector.size()-1)/MASK_SIZE) : (((test_vector.size()-1)/MASK_SIZE)+1);
        int num_segments_A    = ((A_rowptr.size()-1)%MASK_SIZE == 0) ? ((A_rowptr.size()-1)/MASK_SIZE) : (((A_rowptr.size()-1)/MASK_SIZE)+1);
        int num_segments_B    = ((B_colptr.size()-1)%MASK_SIZE == 0) ? ((B_colptr.size()-1)/MASK_SIZE) : (((B_colptr.size()-1)/MASK_SIZE)+1);
        std::vector<BMASK_TYPE> h_test(num_segments_test);
        std::vector<BMASK_TYPE> h_resultA(num_segments_A);
        std::vector<BMASK_TYPE> h_resultB(num_segments_B);

        CHECK_CUDA(cudaMemcpy(h_test.data(),    result_test, sizeof(BMASK_TYPE) * num_segments_test, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_resultA.data(), resultA,     sizeof(BMASK_TYPE) * num_segments_A, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_resultB.data(), resultB,     sizeof(BMASK_TYPE) * num_segments_B, cudaMemcpyDeviceToHost));

        std::cout << "Result test: ";
        for (auto v : h_test) std::cout << static_cast<int>(v) << " ";
        std::cout << "\nResult A: ";
        for (auto v : h_resultA) std::cout << static_cast<int>(v) << " ";
        std::cout << "\nResult B: ";
        for (auto v : h_resultB) std::cout << static_cast<int>(v) << " ";
        std::cout << std::endl;

        ASSERT(num_segments_A == num_segments_B, "For the intersection num_segments_A == num_segments_B");
        BMASK_TYPE *intersection = SpaComm::intersect_bitmasks(resultA, resultB, num_segments_A);
        // return(0); // DEBUG

        std::vector<BMASK_TYPE> h_intersection(num_segments_A);
        cudaMemcpy(h_intersection.data(), intersection, sizeof(BMASK_TYPE) * num_segments_A, cudaMemcpyDeviceToHost);

        std::cout << "Intersection: ";
        for (auto v : h_intersection) std::cout << static_cast<int>(v) << " ";
        std::cout << std::endl;

        std::cout << "----- Test for compression -----" << std::endl;
        tmp_test1();

        std::cout << "----- Test syntetic matrices -----" << std::endl;
        for (int i=0; i<8; i++) free(gen_syntetic_matrix<int,float>(i%5, 8, (i<=4) ? (mmio::MajorDim::ROWS) : (mmio::MajorDim::COLS), true));
   }
   MPI_Barrier(MPI_COMM_WORLD);

    // ----------------------------------------------------------------------------------------------------
    parallel_exps:
    if (world_rank==0) std::cout << "-------------------- Parallel part --------------------" << std::endl; fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    dmmio::utils::ProcessGrid_graph(grid, stdout);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Here I am generating the matrices in the way that:
     *                     B:
     *                     ----------------------
     *                     | 2%3  | 1%3  | 0%3  |
     *                     ----------------------
     *                     | 2%3  | 1%3  | 0%3  |
     *                     ----------------------
     *                        ||    ||
     * A:                     \/    \/
     * ---------------     ----------------------
     * | even | even | --> | 2%6  | 4%6  | 0%6  |
     * ---------------     ----------------------
     * | odd  | odd  | --> | 5%6  | 1%6  | 3%6  |
     * ---------------     ----------------------
     *
     */

    mmio::CSX<int,float> *A_csx = gen_syntetic_matrix<int,float>((grid->col_rank%2)   , 16, mmio::MajorDim::COLS);
    mmio::CSX<int,float> *B_csx = gen_syntetic_matrix<int,float>((grid->row_rank%3) +2, 16, mmio::MajorDim::ROWS);

    // MPI_ALL_PRINT(
    //     fprintf(fp, "global_rank: %d, row_rank: %d, col_rank: %d, node_rank: %d\n",
    //             grid->global_rank, grid->row_rank, grid->col_rank, grid->node_rank
    //     );
    //     mmio::utils::CSX_print_as_dense(A_csx, "A matrix", fp);
    //     mmio::utils::CSX_print_as_dense(B_csx, "B matrix", fp);
    // )

    moveCSX2device(A_csx);
    moveCSX2device(B_csx);

    if (world_rank==0) std::cout << "----- Test spcomm_2D -----" << std::endl; fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    fflush(stdout);

    BMASK_TYPE *col_filters, *row_filters;
    SpaComm::spcomm_2D(A_csx, B_csx, grid, &col_filters, &row_filters);

    int k = A_csx->ncols; // We know  A_csx->ncols == B_csx->nrows
    int mask_size = ((k%MASK_SIZE)==0) ? (k/MASK_SIZE) : ((k/MASK_SIZE)+1) ;
    int filter_size = mask_size * grid->row_size ;

    std::vector<BMASK_TYPE> h_col_filters(filter_size);
    std::vector<BMASK_TYPE> h_row_filters(filter_size);

    CHECK_CUDA(cudaMemcpy(h_col_filters.data(), col_filters, sizeof(BMASK_TYPE) * filter_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_row_filters.data(), row_filters, sizeof(BMASK_TYPE) * filter_size, cudaMemcpyDeviceToHost));

    BMASK_TYPE *A_map = SpaComm::gen_bitmask(A_csx->ptr_vec, A_csx->ncols, MASK_SIZE);
    BMASK_TYPE *B_map = SpaComm::gen_bitmask(B_csx->ptr_vec, B_csx->nrows, MASK_SIZE);

    move2host(&A_map, mask_size);
    move2host(&B_map, mask_size);

    MPI_ALL_PRINT(
        fprintf(fp, "A[%d,%d] bitmask: ", grid->col_rank, grid->row_rank);
        SpaComm::printBit_left2right(A_map, mask_size, fp);
        fprintf(fp, "\n");

        fprintf(fp, "B[%d,%d] bitmask: ", grid->col_rank, grid->row_rank);
        SpaComm::printBit_left2right(B_map, mask_size, fp);
        fprintf(fp, "\n");

        for (int i=0; i<grid->row_size; i++) {
            fprintf(fp, "Col_filters for C[%d,%d,%d] = A[%d,%d] * B[%d, %d]: ",
                    grid->col_rank, i, grid->row_rank,
                    grid->col_rank, grid->row_rank,
                    grid->row_rank, i
            );
            SpaComm::printBit_left2right(h_col_filters.data()+(i*mask_size), mask_size, fp);

            fprintf(fp, "\nRow_filters for C[%d,%d,%d] = A[%d,%d] * B[%d, %d]: ",
                    i, grid->row_rank, grid->col_rank,
                    i, grid->col_rank,
                    grid->col_rank, grid->row_rank
            );
            SpaComm::printBit_left2right(h_row_filters.data()+(i*mask_size), mask_size, fp);
            fprintf(fp, "\n");
        }
    )

    int input_check = 1, global_check;
    if ((grid->col_rank % 2) == 0) input_check = (input_check && check_mod_pattern(A_map, mask_size, 0, 2));
    if ((grid->col_rank % 2) == 1) input_check = (input_check && check_mod_pattern(A_map, mask_size, 1, 2));
    if ((grid->row_rank % 3) == 0) input_check = (input_check && check_mod_pattern(B_map, mask_size, 2, 3));
    if ((grid->row_rank % 3) == 1) input_check = (input_check && check_mod_pattern(B_map, mask_size, 1, 3));
    if ((grid->row_rank % 3) == 2) input_check = (input_check && check_mod_pattern(B_map, mask_size, 0, 3));
    MPI_Allreduce(&input_check, &global_check, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (world_rank==0) { if(global_check) fprintf(stdout, "Input check passed\n"); else fprintf(stderr, "ERROR on input check\n"); }

    int output_check = 1;
    if ((grid->col_rank % 2) == 0) {
        output_check = (output_check && check_mod_pattern(h_col_filters.data()              , mask_size, 2, 6));
        output_check = (output_check && check_mod_pattern(h_col_filters.data() +   mask_size, mask_size, 4, 6));
        output_check = (output_check && check_mod_pattern(h_col_filters.data() + 2*mask_size, mask_size, 0, 6));
    }
    if ((grid->col_rank % 2) == 1) {
        output_check = (output_check && check_mod_pattern(h_col_filters.data()              , mask_size, 5, 6));
        output_check = (output_check && check_mod_pattern(h_col_filters.data() +   mask_size, mask_size, 1, 6));
        output_check = (output_check && check_mod_pattern(h_col_filters.data() + 2*mask_size, mask_size, 3, 6));
    }
    if ((grid->row_rank % 3) == 0) {
        output_check = (output_check && check_mod_pattern(h_row_filters.data()            , mask_size, 2, 6));
        output_check = (output_check && check_mod_pattern(h_row_filters.data() + mask_size, mask_size, 5, 6));
    }
    if ((grid->row_rank % 3) == 1) {
        output_check = (output_check && check_mod_pattern(h_row_filters.data()            , mask_size, 4, 6));
        output_check = (output_check && check_mod_pattern(h_row_filters.data() + mask_size, mask_size, 1, 6));
    }
    if ((grid->row_rank % 3) == 2) {
        output_check = (output_check && check_mod_pattern(h_row_filters.data()            , mask_size, 0, 6));
        output_check = (output_check && check_mod_pattern(h_row_filters.data() + mask_size, mask_size, 3, 6));
    }

    MPI_Allreduce(&output_check, &global_check, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (world_rank==0) { if(global_check) fprintf(stdout, "Output check passed\n"); else fprintf(stderr, "ERROR on output check\n"); }

    return 0;
}

