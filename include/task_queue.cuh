#pragma once



#include "common.h"
#include "KokkosWrap.hpp"

using namespace KokkosWrap;

template <typename IT, typename VT>
struct LocalSpGEMMTask
{

    using LocalMatrix = LocalMatrix<IT, IT, VT>;
    using KokkosCrs = LocalMatrix::KokkosCrs;

    MPI_Datatype MPI_TASK = MPIType<LocalSpGEMMTask<IT, VT>>; 

    int colAtoGet;
    int rowBtoGet;
    int iter;


    LocalSpGEMMTask():
        colAtoGet(-1),
        rowBtoGet(-1),
        iter(-1),
    {}



    LocalMatrix execute(DistCusparse<IT, VT> * A, DistCusparse<IT, VT> * B,
                        TileHolder<IT, VT>& A_holder, TileHolder<IT, VT>& B_holder,
                        MessageQueue<int>& A_queue, MessageQueue<int>& B_queue,
                        int row_rank, int col_rank, 
                        dmmio::ProcessGrid * grid, 
                        cudaStream_t& stream,
                        CsxBuffers<IT, VT>* conversion_buffs,
                        CsxBuffers<IT, VT>* gather_buffs)
    {

        // Ask for remote tile
        A_queue.notify(&row_rank, colAtoGet, iter);
        B_queue.notify(&col_rank, rowBtoGet, iter);


        // Receive remote tiles
        IT A_tile_nnz, B_tile_nnz;
        MPI_Request reqs[2];
        if (rowBtoGet != grid->col_rank) 
        {
            B_tile_nnz = B_holder.recv_tile_contig(rowBtoGet, &reqs[1]);
        } 
        else 
        {
            B_tile_nnz = B_holder.copy_device_local_csx(B->csx->mat, stream);
        }

        if (colAtoGet != grid->row_rank) 
        {
            A_tile_nnz = A_holder.recv_tile_contig(colAtoGet, &reqs[0]);
        } 
        else 
        {
            A_tile_nnz = A_holder.copy_device_local_csx(A->csx->mat, stream);
        }

        if (rowBtoGet != grid->col_rank && B_tile_nnz > 0)
        {
            MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
        }


        if (colAtoGet != grid->row_rank && A_tile_nnz > 0)
        {
            MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
        }

        CUDA_SYNC(stream);


        // Make remote A
        CusparseCSX<IT, VT> * A_remote = new CusparseCSX<IT,VT>(&handle, 
                                                                A_holder.form_mmiocsx(A->csx->nrows(), 
                                                                                      A->csx->ncols(), 
                                                                                      A_tile_nnz, 
                                                                                      dist_A->csx->mat->majordim), 
                                                                conversion_buffs);
        CUDA_SYNC(stream);


        // Make remote B
        CusparseCSX<IT, VT> * B_node = new CusparseCSX<IT, VT>(B_holder.node_allgather_mmiocsx(B->csx->nrows(), B->csx->ncols(), B_tile_nnz, grid, gather_buffs));
        CUDA_SYNC(stream);


        // Local partition of C
        LocalMatrix<IT, IT, VT> C_p;


        // Local SpGEMM
        if (A_remote->nnz() > 0 && B_node->nnz() > 0)
        {
            LocalMatrix<IT, IT, VT> A_p(A_remote->mat);
            LocalMatrix<IT, IT, VT> B_p(B_node->mat);
            LocalMatrix<IT, IT, VT>::spgemm(A_p, B_p, C_p);
        }
        CUDA_SYNC(stream);


        // Return result
        return C_p;
    }
                 


};


template <typename Task>
struct TaskQueue
{
    int ntasks;
    int local_ntasks;
    int * ncompleted;
    Task * tasks;
    int * claimed;
    dmmio::ProcessGrid * grid;
    MPI_Win task_win;
    MPI_Win claimed_win;
    MPI_Win ncompleted_win;

    static constexpr ncomplete_owner = 0;
    static constexpr coordinator_rank = 0;


    TaskQueue(dmmio::ProcessGrid * grid, int row_rank, int col_rank):
        ntasks(grid->global_size * grid->row_size), local_ntasks(grid->row_size), 
        tasks(new Tasks[grid->row_size]), claimed(new int[grid->row_size]),
        grid(grid),
    {
        // Initialize my tasks
        // TODO: This is not general
        int col_id = (grid->row_rank + grid->col_rank) % grid->row_size;
        int row_id = (grid->row_rank + grid->col_rank) % grid->row_size;
        for (int i=0; i<local_ntasks; i++)
        {
            tasks[i].colAtoGet = col_id;
            tasks[i].rowBtoGet = row_id;
            tasks[i].iter = i;
            col_id = (col_id + 1) % grid->row_size;
            row_id = (row_id + 1) % grid->row_size;
            claimed[i] = -1;
        }


        // Seed RNG
        srand( (unsigned)time(NULL) ); 


        ntasks = local_ntasks * grid->global_size;
        ncompleted = new int(0);


        // Create MPI Windows 
        MPI_Win_create(tasks, sizeof(Task) * local_ntasks, sizeof(Task), MPI_INFO_NULL, grid->world_comm, &task_win);
        MPI_Win_create(claimed, sizeof(int) * local_ntasks, sizeof(int), MPI_INFO_NULL, grid->world_comm, &claimed_win);
        MPI_Win_create(ncompleted, sizeof(int), sizeof(int), MPI_INFO_NULL, grid->world_comm, &claimed_win);

        MPI_Barrier(grid->world_comm);
    }



    inline int get_random_coordinator()
    {
        int rng = rand() % grid->global_size;
        return (rng / grid->node_size) * grid->node_size + coordinator_rank; 
    }
    


    inline bool is_coordinator()
    {
        return grid->global_rank % grid->node_size == coordinator_rank;
    }



    inline bool is_coordinator(int rank)
    {
        return rank % grid->node_size == coordinator_rank;
    }



    Task * pop_random_task()
    {
        int target_rank = get_random_coordinator();
        int offset = rand() % grid->row_size;
        return pop_task(target_rank, offset);
    }



    Task * pop_local_task(int offset)
    {
        return pop_task(grid->global_rank, offset);
    }



    Task * pop_task(int rank, int offset)
    {
        assert(rank < grid->global_size);
        assert(offset < grid->row_size);
        Task * result;
        if (is_coordinator())
        {
            result = pop_task_coordinator(rank, offset);
        }
        else
        {
            result = new Task;
        }
        MPI_Bcast(result, 1, MPI_TASK, 0, grid->node_comm);
        return result;
    }



    Task * pop_task_coordinator(int rank, int offset)
    {

        assert(is_coordinator(rank));

        // Claim a task, if no one else already has
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, claimed_win);
        int negone = -1;
        int result;
        MPI_Compare_and_swap(&(rank), &negone, &result, MPI_INT, rank, offset, claimed_win);
        MPI_Win_unlock(rank, claimed_win);

        if (result == -1)
        {
            return nullptr;
        }


        // Increment global complete task count
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, ncomplete_owner, 0, ncomplete_win);
        int one = 1;
        MPI_Accumulate(&one, 1, MPI_INT, ncomplete_owner, 0, 1, MPI_INT, MPI_SUM, ncomplete_win);
        MPI_Win_unlock(ncomplete_owner, ncomplete_win);


        // Get the task
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, task_win);
        Task * result = new Task;
        MPI_Get(result, 1, MPI_TASK, rank, offset, 1, MPI_TASK, task_win);
        MPI_Win_unlock(rank, task_win);


        // Done
        return result;
    }



    int check_n_complete()
    {
        int result;
        MPI_Win_lock(MPI_LOCK_SHARED, ncomplete_owner, MPI_MODE_NOCHECK, ncomplete_win);
        MPI_Get(&result, 1, MPI_INT, ncomplete_owner, 0, 1, MPI_INT, ncomplete_win);
        MPI_Win_unlock(ncomplete_owner, ncomplete_win);
        return result;
    }

    

    ~TaskQueue()
    {
        MPI_Win_free(&task_win);
        MPI_Win_free(&claimed_win);
        MPI_Win_free(&ncomplete_win);
        delete[] tasks;
        delete[] claimed;
        delete ncomplete;
        MPI_Barrier(grid->world_comm);
    }
};










