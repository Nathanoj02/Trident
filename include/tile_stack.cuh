#pragma once
#include "common.h"


using namespace mmio;

template <typename IT, typename VT>
struct TileStack
{

    static_assert( sizeof(IT) == sizeof(VT) ); // This is necessary for now to avoid alignment issues in the stack

    static const MPI_Datatype MPI_IDX = MPIType<IT>();
    static const MPI_Datatype MPI_VAL = MPIType<VT>();


    /****************************************************
     *                                                  *
     *                  CONSTRUCTORS                    *
     *                                                  *
     ****************************************************/

    TileStack(const size_t buf_size, MPI_Comm comm):
        buf_size(buf_size), comm(comm)
    {
        ASSERT( ( (buf_size % sizeof(VT) == 0) && buf_size % sizeof(IT) == 0 ), 
                "Error: the buffer size must evenly divide sizeof(VT) and sizeof(IT) for alignment purposes");

        MPI_Comm_rank(comm, &rank);

        CUDA_CHECK(cudaMalloc(&d_win_buf, buf_size));
        MPI_Win_create(d_win_buf, buf_size, sizeof(uint8_t), MPI_INFO_NULL, comm, &win);

        tail = 0;
        MPI_Win_create(&tail, sizeof(size_t), sizeof(size_t), MPI_INFO_NULL, comm, &tail_win);

        MPI_Barrier(comm);
    }


    /****************************************************
     *                                                  *
     *                 MEMBER FUNCTIONS                 *
     *                                                  *
     ****************************************************/


    /* Stack messages are stored in the following fashion:
     *      [....|vals|padding|colinds|rowptrs|...]
     *
     *          vals: sizeof(VT) * nnz
     *          padding: if sizeof(IT) == 4 and sizeof(VT) == 8, there may need to be 4 bytes of padding to ensure everything is aligned
     *                   if (sizeof(rowptrs) + sizeof(colinds) is not divisible by 8, then 4 bytes of paddding are added
     *          colinds: sizeof(IT) * nnz
     *          rowptrs: sizeof(IT) * (nrows + 1) -- the last element is nnz
     *
     * In general, the stack looks like this
     *      [Message | Message | ... |Message| ... ]
     *      0                                ^tail buf_size 
     * So the tail points to the last element of the stack, which means you need to decrement from the tail if you want to access elements
     */

    inline size_t padding_size(const size_t rowptrs_sz, const size_t colinds_sz) const
    {
        //TODO -- isn't needed unless we have sizeof(VT) != sizeof(IT)
        return 0; 
    }


    inline size_t rowptrs_size(CSR<IT, VT>& tile) const
    {
        return sizeof(IT) * (tile.nrows + 1);
    }
    

    inline size_t rowptrs_size(const IT nrows) const
    {
        return sizeof(IT) * (nrows + 1);
    }


    inline size_t colinds_size(CSR<IT, VT>& tile) const
    {
        return sizeof(IT) * (tile.nnz);
    }


    inline size_t vals_size(CSR<IT, VT>& tile) const
    {
        return sizeof(VT) * (tile.nnz);
    }


    void push_tile(CSR<IT, VT>& tile, const int target)
    {
        // Size of buffers
        const size_t tile_rowptrs_size = rowptrs_size(tile);
        const size_t tile_colinds_size = colinds_size(tile);
        const size_t tile_vals_size = vals_size(tile);
        const size_t padding_tile_size = padding_size(tile_rowptrs_size, tile_colinds_size);
        const size_t total_msg_size = tile_rowptrs_size + tile_colinds_size + tile_vals_size + padding_tile_size;


        // Lock both windows
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, 0, win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, 0, tail_win);


        // First, get and accumulate tail of stack
        size_t stack_tail; // This will store the old value of the tail
        MPI_Fetch_and_op(&total_msg_size, &stack_tail, MPI_UNSIGNED_LONG, target, 0, MPI_SUM, tail_win);
        MPI_Win_flush(target, tail_win);

        if (stack_tail + total_msg_size >= buf_size)
        {
            std::cerr<<"ERROR: Stack tail: "<<stack_tail<<", total_msg_size:" <<total_msg_size<<
                ", buf_size: "<<buf_size<<"-- stack_tail + total_msg_size >= buf_size"<<std::endl;
            std::abort();
        }


        // Now, put the tile on the stack
        MPI_Put((uint8_t *)(tile.val), tile_vals_size, MPI_BYTE, target, stack_tail, tile_vals_size, MPI_BYTE, win);
        MPI_Put((uint8_t *)(tile.col_idx), tile_colinds_size, MPI_BYTE, target, stack_tail + tile_vals_size + padding_tile_size, tile_colinds_size, MPI_BYTE, win);
        MPI_Put((uint8_t *)(tile.row_ptr), tile_rowptrs_size, MPI_BYTE, target, stack_tail + tile_vals_size + padding_tile_size + tile_colinds_size, tile_rowptrs_size, MPI_BYTE, win);

        // End the RMA epoch
        MPI_Win_unlock(target, win);
        MPI_Win_unlock(target, tail_win);
    }


    CSR<IT, VT> pop_tile(IT nrows, IT ncols)
    {
        CSR<IT, VT> tile;

        if (tail == 0)
        {
            return tile; //return empty
        }

        tile.nrows = nrows;
        tile.ncols = ncols;

        const size_t tile_rowptrs_size = rowptrs_size(tile);

        CUDA_CHECK(cudaMalloc(&(tile.row_ptr), tile_rowptrs_size));

        // Begin RMA epoch
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, tail_win);


        // Synchronize to get all local updates
        // TODO: This might be slow... maybe it's better to call this every so often
        MPI_Win_sync(win);
        MPI_Win_sync(tail_win);


        // First, get nnz
        d2h_copy(&(tile.nnz), 1, (IT *)(d_win_buf + tail - sizeof(IT))); //TODO: avoid this

        // Calculate other sizes
        const size_t tile_colinds_size = colinds_size(tile);
        const size_t tile_vals_size = vals_size(tile);
        const size_t padding_tile_size = padding_size(tile_rowptrs_size, tile_colinds_size);
        const size_t total_msg_size = tile_rowptrs_size + tile_colinds_size + tile_vals_size + padding_tile_size;


        // Now we can malloc the other parts of the tile
        CUDA_CHECK(cudaMalloc(&(tile.col_idx), tile_colinds_size));
        CUDA_CHECK(cudaMalloc(&(tile.val), tile_vals_size));


        // And now we can copy the rest of the tile 
        IT * d_rowptrs_ptr = (IT *)(d_win_buf + tail - tile_rowptrs_size);
        IT * d_colinds_ptr = (IT *)(d_win_buf + tail - tile_rowptrs_size - tile_colinds_size);
        VT * d_vals_ptr = (VT *)(d_win_buf + tail - tile_rowptrs_size - tile_colinds_size - padding_tile_size - tile_vals_size);
        CUDA_CHECK(cudaMemcpyAsync(tile.row_ptr, d_rowptrs_ptr, tile_rowptrs_size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpyAsync(tile.col_idx, d_colinds_ptr, tile_colinds_size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpyAsync(tile.val, d_vals_ptr, tile_vals_size, cudaMemcpyDeviceToDevice));


        // Decrement the stack tail
        tail -= total_msg_size;
        MPI_Win_sync(tail_win);


        // End the RMA epoch
        MPI_Win_unlock(rank, win);
        MPI_Win_unlock(rank, tail_win);


        // Return the tile
        return tile;
    }


    /****************************************************
     *                                                  *
     *                  DESTRUCTORS                     *
     *                                                  *
     ****************************************************/

    ~TileStack()
    {
        MPI_Win_free(&win);
        MPI_Win_free(&tail_win);
        CUDA_FREE_SAFE(d_win_buf);
    }




    /****************************************************
     *                                                  *
     *                  DATA MEMBERS                    *
     *                                                  *
     ****************************************************/

    // Main stack 
    MPI_Win win; 
    uint8_t * d_win_buf;
    size_t buf_size; 

    // Stack metadata
    MPI_Win tail_win;
    size_t tail;

    // MPI metadata
    int rank;
    MPI_Comm comm;

};
