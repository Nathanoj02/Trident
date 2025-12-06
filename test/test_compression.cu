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
#define NREP 1

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

template <typename IT, typename VT>
mmio::CSX<IT,VT>* cpmat_d2h (mmio::CSX<IT,VT> *devicemat) {
    IT nnz   = devicemat->nnz;
    IT nrows = devicemat->nrows;
    IT ncols = devicemat->ncols;
    mmio::MajorDim majd = devicemat->majordim;
    uint64_t ptrsize = (majd==mmio::MajorDim::ROWS) ? ( devicemat->nrows + 1 ) : ( devicemat->ncols + 1 );

    VT *h_val     = (VT*)malloc(sizeof(VT) * nnz);
    IT *h_idx_vec = (IT*)malloc(sizeof(IT) * nnz);
    IT *h_ptr_vec = (IT*)malloc(sizeof(IT) * ptrsize);
    cudaMemcpy(h_ptr_vec, devicemat->ptr_vec, sizeof(IT) * ptrsize, cudaMemcpyDefault);
    cudaMemcpy(h_idx_vec, devicemat->idx_vec, sizeof(VT) * nnz,     cudaMemcpyDefault);
    cudaMemcpy(h_val,     devicemat->val,     sizeof(VT) * nnz,     cudaMemcpyDefault);

    mmio::CSX<IT,VT> *hostmat = mmio::CSX_create(nrows, ncols, nnz, majd,
        h_ptr_vec,
        h_idx_vec,
        h_val
    );
    return(hostmat);
}

enum SynteticMatrixSeed {
    EVEN,
    ODD,
    TWOOTHREE,
    ONEOTHREE,
    ZEROOTHREE
};

const char* to_string(SynteticMatrixSeed seed) {
    switch(seed) {
        case EVEN:
            return("even");
        case ODD:
            return("odd");
        case TWOOTHREE:
            return("twoOthree");
        case ONEOTHREE:
            return("oneOthree");
        case ZEROOTHREE:
            return("zeroOthree");
        default:
            fprintf(stderr, "Unrecognized seed (%d)\n", seed);
            exit(__LINE__);
    }
}

template<typename IT, typename VT>
// mmio::CSX<IT, VT>* gen_syntetic_matrix(int seed, int n, mmio::MajorDim layout, bool print=false) {
mmio::CSX<IT, VT>* gen_syntetic_matrix(SynteticMatrixSeed seed, dmmio::ProcessGrid *grid, int scale_factor, mmio::MajorDim layout, bool print=false) {

    int nrows = scale_factor;
    int ncols = grid->node_size * scale_factor;
    int total_rows = grid->node_size * scale_factor;

    bool(*empty_row_condition)(int);
    switch(seed) {
        case EVEN:
            empty_row_condition = even;
            break;
        case ODD:
            empty_row_condition = odd;
            break;
        case TWOOTHREE:
            empty_row_condition = twoOthree;
            break;
        case ONEOTHREE:
            empty_row_condition = oneOthree;
            break;
        case ZEROOTHREE:
            empty_row_condition = zeroOthree;
            break;
        default:
            fprintf(stderr, "Unrecognized seed (%d)\n", seed);
            exit(__LINE__);
    }

    int ptr_size = (layout == mmio::MajorDim::ROWS) ? (nrows+1) : (ncols+1) ;
    std::vector<IT> vec_ptr(ptr_size);
    vec_ptr[0] = 0;
    IT nnz = 0;

    // MPI_Barrier(MPI_COMM_WORLD);
    // if (grid->global_rank==0) std::cout << __func__ << " initialized" << std::endl; fflush(stdout);
    // sleep(1);
    // MPI_Barrier(MPI_COMM_WORLD);

    if (layout == mmio::MajorDim::ROWS) {
        for (int i=0; i<nrows; i++) {
            int grp_i = (grid->node_rank)*scale_factor + i;
            if (empty_row_condition(grp_i)) {
                IT row_nnz = (grp_i<(ncols/2)) ? (grp_i+1) : (ncols-grp_i);
                vec_ptr[i+1] = vec_ptr[i] + row_nnz;
                nnz += row_nnz;
            } else {
                vec_ptr[i+1] = vec_ptr[i];
            }
        }
    } else {
        for (int i=0; i<ncols; i++) {
            if (empty_row_condition(i)) {
                IT col_nnz = (i<(total_rows/2)) ? (i+1) : (total_rows-i);
                vec_ptr[i+1] = vec_ptr[i] + col_nnz;
                nnz += col_nnz;
            } else {
                vec_ptr[i+1] = vec_ptr[i];
            }
        }
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    // if (grid->global_rank==0) std::cout << __func__ << " ptr generated, nnz: " << nnz << std::endl; fflush(stdout);
    // sleep(1);
    // MPI_Barrier(MPI_COMM_WORLD);

    std::vector<IT> vec_idx(nnz);
    std::vector<VT> vec_val(nnz);
    if (layout == mmio::MajorDim::ROWS) {
        for (int i=0; i<nrows; i++) {
            int grp_i = (grid->node_rank)*scale_factor + i;
            for (int j=0; j<(vec_ptr[i+1] - vec_ptr[i]); j++) {
                vec_idx[vec_ptr[i] + j] = (grp_i<(ncols/2)) ? (grp_i + j) : (ncols - j - 1) ;
                vec_val[vec_ptr[i] + j] = (vec_ptr[i] + j) * 1.0 ;
            }
        }
    } else {
        for (int i=0; i<ncols; i++) {
            for (int j=0; j<(vec_ptr[i+1] - vec_ptr[i]); j++) {
                vec_idx[vec_ptr[i] + j] = (i<(total_rows/2)) ? (i + j) : (total_rows - j - 1) ;
                vec_val[vec_ptr[i] + j] = (vec_ptr[i] + j) * 1.0 ;
            }
        }
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    // if (grid->global_rank==0) std::cout << __func__ << " nnz generated" << std::endl; fflush(stdout);
    // sleep(1);
    // MPI_Barrier(MPI_COMM_WORLD);

    if (layout == mmio::MajorDim::COLS) {
        // Filter nnz wrt range
        IT filtered_nnz = 0;
        IT lower_range = (grid->node_rank)*scale_factor, upper_range = lower_range + scale_factor;
        for (IT value : vec_idx) {
            if ((value >= lower_range) && (value < upper_range)) filtered_nnz++;
        }

        IT k = 0;
        std::vector<IT> new_vec_ptr(ptr_size);
        std::vector<IT> new_vec_idx(filtered_nnz);
        std::vector<VT> new_vec_val(filtered_nnz);
        for (int i=0; i<ptr_size; i++) new_vec_ptr[i] = 0;
        for (int i=0; i<ncols; i++) {
            for (int j=0; j<(vec_ptr[i+1] - vec_ptr[i]); j++) {
                IT value = vec_idx[vec_ptr[i] + j];
                if ((value >= lower_range) && (value < upper_range)) {
                    new_vec_ptr[i]++; // NOTE still not a proper vec_ptr
                    new_vec_idx[k] = vec_idx[vec_ptr[i] + j];
                    new_vec_val[k] = vec_val[vec_ptr[i] + j];
                    k++;
                }
            }
        }

        vec_ptr[0] = 0;
        for (int i=1; i<ptr_size; i++) vec_ptr[i] = vec_ptr[i-1] + new_vec_ptr[i-1];
        for (int i=0; i<filtered_nnz; i++) {
            vec_idx[i] = new_vec_idx[i] - lower_range;
            vec_val[i] = new_vec_val[i];
        }
        nnz = filtered_nnz;
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    // if (grid->global_rank==0) std::cout << __func__ << " nnz filtered" << std::endl; fflush(stdout);
    // sleep(1);
    // MPI_Barrier(MPI_COMM_WORLD);

    if (print) {
        std::cout << "---------- Syntetic matrix ( " << seed << ", " << scale_factor << ") ----------" << std::endl;

        if (layout == mmio::MajorDim::ROWS) {
            std::cout << "----- CSR -----" << std::endl;
            printCSRMatrix(vec_idx, vec_val, vec_ptr);
        } else {
            std::cout << "----- CSC -----" << std::endl;
            printCSCMatrix(vec_idx, vec_val, vec_ptr);
        }
    }

    mmio::CSX<IT,VT> *csx = CSX_create_contig_device(nrows, ncols, nnz, layout,
                                                     vec_idx.data(), vec_ptr.data(), vec_val.data()
    );

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

    int expectedA_fn(int col_rank, int iter) {
        int col_mod  = col_rank % 2;
        int iter_mod =     iter % 3;

        if (col_mod == 0) {
            if (iter_mod==0) return(2);
            if (iter_mod==1) return(4);
            if (iter_mod==2) return(0);
        } else {
            if (iter_mod==0) return(5);
            if (iter_mod==1) return(1);
            if (iter_mod==2) return(3);
        }
    }

    int expectedB_fn(int row_rank, int iter) {
        int row_mod = row_rank % 3;
        int iter_mod =    iter % 2;

        if (row_mod == 0) {
            if (iter_mod==0) return(2);
            if (iter_mod==1) return(5);
        } else if (row_mod == 1) {
            if (iter_mod==0) return(4);
            if (iter_mod==1) return(1);
        } else {
            if (iter_mod==0) return(0);
            if (iter_mod==1) return(3);
        }
    }

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

    int skip = config->Acsc; // I use spcomm as flag for skip the first experiments
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
        for (int i=0; i<8; i++) free(gen_syntetic_matrix<int,float>(static_cast<SynteticMatrixSeed>(i%5), grid, 8, (i<=4) ? (mmio::MajorDim::ROWS) : (mmio::MajorDim::COLS), true));
   }
   MPI_Barrier(MPI_COMM_WORLD);

    // ----------------------------------------------------------------------------------------------------
    parallel_exps:
    skip = config->spcomm;
    if (skip) goto simulated_spgemm;
    {
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


    SynteticMatrixSeed Aseed = static_cast<SynteticMatrixSeed>(grid->col_rank%2);
    SynteticMatrixSeed Bseed = static_cast<SynteticMatrixSeed>((grid->row_rank%3)+2);
    mmio::CSX<int,float> *A_csx = gen_syntetic_matrix<int,float>(Aseed, grid, 16, mmio::MajorDim::COLS);
    mmio::CSX<int,float> *B_csx = gen_syntetic_matrix<int,float>(Bseed, grid, 16, mmio::MajorDim::ROWS);

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
    }

    simulated_spgemm:
    {
        if (world_rank==0) std::cout << "-------------------- Simulation part --------------------" << std::endl; fflush(stdout);
        dmmio::utils::ProcessGrid_graph(grid, stdout);

        // Simulated inputs
        int scale_factor = 16/(grid->node_size);
        if ((scale_factor % MASK_SIZE) != 0) scale_factor += (scale_factor % MASK_SIZE);
        SynteticMatrixSeed Aseed = static_cast<SynteticMatrixSeed>( grid->col_rank%2);
        SynteticMatrixSeed Bseed = static_cast<SynteticMatrixSeed>((grid->row_rank%3) + 2);
        mmio::CSX<int,float> *A_csx = gen_syntetic_matrix<int,float>(Aseed, grid, scale_factor, mmio::MajorDim::COLS);
        mmio::CSX<int,float> *B_csx = gen_syntetic_matrix<int,float>(Bseed, grid, scale_factor, mmio::MajorDim::ROWS);

        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank==0) std::cout << "Generated input on contiguous GPU" << std::endl; fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        sleep(1);

        if (config->verbose > 1) {
            mmio::CSX<int,float> *h_A_csx = cpmat_d2h(A_csx);
            mmio::CSX<int,float> *h_B_csx = cpmat_d2h(B_csx);
            MPI_Barrier(MPI_COMM_WORLD);
            if (world_rank==0) std::cout << "Tmp copy back to the CPU" << std::endl; fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD);
            sleep(1);

            MPI_ALL_PRINT(
                fprintf(fp, "Aseed: %s, Bseed: %s, scale_factor: %d\n", to_string(Aseed), to_string(Bseed), scale_factor);
                mmio::utils::CSX_print_as_dense(h_A_csx, "Input A", fp);
                mmio::utils::CSX_print_as_dense(h_B_csx, "Input B", fp);
            )

            mmio::CSX_destroy(&h_A_csx);
            mmio::CSX_destroy(&h_B_csx);
        }
        MPI_Barrier(MPI_COMM_WORLD);


        if (world_rank==0) printf("Beginning spgemm -- implementation: %s\n", config->impl_str);
        for (int repetition=0; repetition<NREP; repetition++)
        {
            if (world_rank==0)
            {
                printf("----- Repetition %d out of %d -----\n", repetition, NREP);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            sleep(1);

            int common_grid_size = grid->row_size; // This must be equal to kwd_B->...->col_size
            const int n_iters = common_grid_size;

            // Indices of tiles to fetch in the first iteration from each communicator
            int colAtoGet = (grid->row_rank + grid->col_rank) % common_grid_size; // Stragger left
            int rowBtoGet = (grid->col_rank + grid->row_rank) % common_grid_size; // Stragger down

            // Sparsity pattern communication
            if (world_rank==0) std::cout << "Start of Sparsity pattern communication... " << std::endl; fflush(stdout);
            SpaComm::SpaCommHandler<int32_t, float> *spcomm_data = new SpaComm::SpaCommHandler<int32_t, float>(A_csx, B_csx, grid);
            MPI_Barrier(MPI_COMM_WORLD);
            if (world_rank==0) std::cout << "End of Sparsity pattern communication" << std::endl; fflush(stdout);

            // ----- Check correctness -----
            int mask_size   = spcomm_data->mask_len;
            int filter_size = mask_size * grid->row_size ;
            std::vector<BMASK_TYPE> h_col_filters(filter_size);
            std::vector<BMASK_TYPE> h_row_filters(filter_size);
            CHECK_CUDA(cudaMemcpy(h_col_filters.data(), spcomm_data->A_column_filters, sizeof(BMASK_TYPE) * filter_size, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_row_filters.data(), spcomm_data->B_rows_filters, sizeof(BMASK_TYPE) * filter_size, cudaMemcpyDeviceToHost));

            int output_check = 1, global_check;
            for (int k=0; k<common_grid_size; k++) {
                BMASK_TYPE *Afilter = h_col_filters.data() + (k*mask_size);
                BMASK_TYPE *Bfilter = h_row_filters.data() + (k*mask_size);
                int expectedA = expectedA_fn(grid->col_rank, k);
                int expectedB = expectedB_fn(grid->row_rank, k);
                int checkA    = check_mod_pattern(Afilter, mask_size, expectedA, 6);
                int checkB    = check_mod_pattern(Bfilter, mask_size, expectedB, 6);

                if ( config->verbose ) {
                    MPI_ALL_PRINT(
                        fprintf(fp, "Aseed: %s, Bseed: %s\n", to_string(Aseed), to_string(Bseed));
                        fprintf(fp, "Filters %d: (%d,%d)\n", k, checkA, checkB);
                        SpaComm::printBit_left2right(Afilter, mask_size, fp); fprintf(fp, " exp %d\n", expectedA);
                        SpaComm::printBit_left2right(Bfilter, mask_size, fp); fprintf(fp, " exp %d\n", expectedB);
                    )
                }
                output_check = (output_check && checkA);
                output_check = (output_check && checkB);
            }
            MPI_Allreduce(&output_check, &global_check, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
            if (world_rank==0) { if(global_check) fprintf(stdout, "Check passed\n"); else fprintf(stderr, "ERROR on check\n"); }
            // ------------------------------

            SpaComm::SpaCommBuffers<int32_t, float> *buffA = new SpaComm::SpaCommBuffers<int32_t, float>(A_csx);
            SpaComm::SpaCommBuffers<int32_t, float> *buffB = new SpaComm::SpaCommBuffers<int32_t, float>(B_csx);

            // Main loop
            for (int iter = 0; iter < n_iters; iter++)
            {
                if (grid->global_rank == 0)
                {
                    std::cout<<"--- Iteration " << iter << " ---" << std::endl;
                }
                sleep(1);
                MPI_Barrier(MPI_COMM_WORLD);

                if (world_rank==0) std::cout << "Compressing A and B... " << std::endl; fflush(stdout);
                mmio::CSX<int32_t, float> *csxtosendA = spcomm_data->Compress(A_csx, iter, buffA, 0);
                mmio::CSX<int32_t, float> *csxtosendB = spcomm_data->Compress(B_csx, iter, buffB, 0);

                MPI_Barrier(MPI_COMM_WORLD);
                if (world_rank==0) std::cout << "tile compressed" << std::endl; fflush(stdout);

                BMASK_TYPE *A_map = SpaComm::gen_bitmask(csxtosendA->ptr_vec, csxtosendA->ncols, MASK_SIZE);
                BMASK_TYPE *B_map = SpaComm::gen_bitmask(csxtosendB->ptr_vec, csxtosendB->nrows, MASK_SIZE);

                if (grid->node_size > 1) {
                    BMASK_TYPE *tmp_A_map, *tmp_B_map;
                    int mask_size = spcomm_data->mask_len;
                    int sub_mask_size = spcomm_data->sub_mask_len;
                    CUDA_CHECK(cudaMalloc(&tmp_A_map, sizeof(BMASK_TYPE)*mask_size));
                    CUDA_CHECK(cudaMalloc(&tmp_B_map, sizeof(BMASK_TYPE)*mask_size));
                    MPI_Allreduce(A_map, tmp_A_map, mask_size, MPI_BMASK_TYPE, MPI_BOR, grid->node_comm);
                    MPI_Allgather(B_map, sub_mask_size, MPI_BMASK_TYPE, tmp_B_map, sub_mask_size, MPI_BMASK_TYPE, grid->node_comm);
                    CUDA_CHECK(cudaFree(A_map));
                    CUDA_CHECK(cudaFree(B_map));
                    A_map = tmp_A_map;
                    B_map = tmp_B_map;
                }

                MPI_Barrier(MPI_COMM_WORLD);
                if (world_rank==0) std::cout << "Bitmasks computed" << std::endl; fflush(stdout);

                int expectedA = expectedA_fn(grid->col_rank, iter);
                int expectedB = expectedB_fn(grid->row_rank, iter);

                // Copy masks to host
                BMASK_TYPE *h_A_map = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*mask_size);
                BMASK_TYPE *h_B_map = (BMASK_TYPE*)malloc(sizeof(BMASK_TYPE)*mask_size);
                cudaMemcpy(h_A_map, A_map, sizeof(BMASK_TYPE)*mask_size, cudaMemcpyDefault);
                cudaMemcpy(h_B_map, B_map, sizeof(BMASK_TYPE)*mask_size, cudaMemcpyDefault);

                // Computing correctness check
                int checkA = check_mod_pattern(h_A_map, mask_size, expectedA, 6);
                int checkB = check_mod_pattern(h_B_map, mask_size, expectedB, 6);
                output_check = (output_check && checkA);
                output_check = (output_check && checkB);

                MPI_Barrier(MPI_COMM_WORLD);
                if (world_rank==0) std::cout << "Bitmasks checked: " << output_check << std::endl; fflush(stdout);
                sleep(1);
                MPI_Barrier(MPI_COMM_WORLD);

                if (config->verbose) {
                    MPI_ALL_PRINT(
                        fprintf(fp, "[%d] %p %p\n", world_rank, A_map, B_map);
                        SpaComm::printBit_left2right(h_A_map, mask_size, fp); fprintf(fp, "\n");
                        SpaComm::printBit_left2right(h_B_map, mask_size, fp); fprintf(fp, "\n");

                        int int_A_map = 0; for (int i=0;i<mask_size;i++) int_A_map += static_cast<int>(h_A_map[i]);
                        int int_B_map = 0; for (int i=0;i<mask_size;i++) int_B_map += static_cast<int>(h_B_map[i]);

                        fprintf(fp, "Expected A: %d (mod 6) --> %d\n", expectedA, checkA);
                        fprintf(fp, "Expected B: %d (mod 6) --> %d\n", expectedB, checkB);
                    )
                }

                if ( config->verbose ) {
                    mmio::CSX<int,float> *h_csxtosendA = cpmat_d2h(csxtosendA);
                    mmio::CSX<int,float> *h_csxtosendB = cpmat_d2h(csxtosendB);
                    MPI_ALL_PRINT(
                        fprintf(fp, "Expected A: %d (mod 6) --> %d\n", expectedA, checkA);
                        mmio::utils::CSX_print_as_dense(h_csxtosendA, "Filtered A", fp);
                        fprintf(fp, "Expected B: %d (mod 6) --> %d\n", expectedB, checkB);
                        mmio::utils::CSX_print_as_dense(h_csxtosendB, "Filtered B", fp);
                    )
                    mmio::CSX_destroy(&h_csxtosendA);
                    mmio::CSX_destroy(&h_csxtosendB);
                }

                CUDA_CHECK(cudaFree(A_map));
                CUDA_CHECK(cudaFree(B_map));
                free(h_A_map);
                free(h_B_map);

                if (world_rank==0) std::cout << "Maps freed" << std::endl; fflush(stdout);

                // Round shift
                colAtoGet = (colAtoGet + 1) % common_grid_size; // ShiftLeft
                rowBtoGet = (rowBtoGet + 1) % common_grid_size; // ShiftDown
            }
            MPI_Allreduce(&output_check, &global_check, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            if (world_rank==0)
            {
                printf("----- -------------------- -----\n");
                if(global_check)
                    fprintf(stdout, "Check passed\n");
                else
                    fprintf(stderr, "ERROR on check\n");
            }
            sleep(1);
            MPI_Barrier(MPI_COMM_WORLD);

            // TODO: free here the buffers...
        }
        if (world_rank==0) printf("Done spgemm\n");
    }

    return 0;
}

