#pragma once

#include "KokkosWrap.hpp"
#include "cusparse_helpers.cuh"

using namespace KokkosWrap;

template <typename IT, typename VT>
struct LocalSpGEMMTask
{

    using LocalMatrix = LocalMatrix<IT, IT, VT>;
    using KokkosCrs = LocalMatrix::KokkosCrs;

    MPI_Datatype MPI_TASK; 

    int row_rank_A;
    int col_rank_A;
    int row_rank_B;
    int col_rank_B;
    int owner;
    int iter;


    LocalSpGEMMTask():
        row_rank_A(-1),
        col_rank_A(-1),
        row_rank_B(-1),
        col_rank_B(-1),
        MPI_TASK(MPIType<LocalSpGEMMTask<IT, VT>>())
    {}



    inline int ranks_to_global(dmmio::ProcessGrid * grid, int row_rank, int col_rank)
    {
        return (row_rank) * (grid->node_size * grid->col_size) + (col_rank * grid->node_size) + grid->node_rank;
    }


    inline int ranks_to_global_A(dmmio::ProcessGrid * grid)
    {
        return ranks_to_global(grid, row_rank_A, col_rank_A);
    }



    inline int ranks_to_global_B(dmmio::ProcessGrid * grid)
    {
        return ranks_to_global(grid, row_rank_B, col_rank_B);
    }



    LocalMatrix execute(DistCusparseCSX<IT, VT> * A, DistCusparseCSX<IT, VT> * B,
                        TileHolder<IT, VT>& A_holder, TileHolder<IT, VT>& B_holder,
                        MessageQueue<int>& A_queue, MessageQueue<int>& B_queue,
                        dmmio::ProcessGrid * grid, 
                        cudaStream_t& stream,
                        cusparseHandle_t& handle,
                        CsxBuffers<IT, VT>* conversion_buffs,
                        CsxBuffers<IT, VT>* gather_buffs)
    {

        int A_rank = ranks_to_global_A(grid);
        int B_rank = ranks_to_global_B(grid);

        // Ask for remote tile
        A_queue.notify(&(grid->global_rank), A_rank, iter);
        B_queue.notify(&(grid->global_rank), B_rank, iter);


        // Receive remote tiles
        IT A_tile_nnz, B_tile_nnz;
        MPI_Request reqs[2];
        if (B_rank != grid->global_rank) 
        {
            B_tile_nnz = B_holder.recv_tile_contig(B_rank, &reqs[1]);
        } 
        else 
        {
            B_tile_nnz = B_holder.copy_device_local_csx(B->csx->mat, stream);
        }

        if (A_rank != grid->global_rank) 
        {
            A_tile_nnz = A_holder.recv_tile_contig(A_rank, &reqs[0]);
        } 
        else 
        {
            A_tile_nnz = A_holder.copy_device_local_csx(A->csx->mat, stream);
        }

        if (grid->global_rank != B_rank && B_tile_nnz > 0)
        {
            MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
        }


        if (grid->global_rank != A_rank && A_tile_nnz > 0)
        {
            MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
        }

        CUDA_SYNC(stream);


        // Make remote A
        CusparseCSX<IT, VT> * A_remote = new CusparseCSX<IT,VT>(&handle, 
                                                                A_holder.form_mmiocsx(A->csx->nrows(), 
                                                                                      A->csx->ncols(), 
                                                                                      A_tile_nnz, 
                                                                                      A->csx->mat->majordim), 
                                                                conversion_buffs);
        CUDA_SYNC(stream);


        // Make remote B
        CusparseCSX<IT, VT> * B_node = new CusparseCSX<IT, VT>(B_holder.node_allgather_mmiocsx(B->csx->nrows(), B->csx->ncols(), B_tile_nnz, grid, gather_buffs));
        CUDA_SYNC(stream);


        // Local partition of C
        LocalMatrix C_p;


        // Local SpGEMM
        if (A_remote->nnz() > 0 && B_node->nnz() > 0)
        {
            LocalMatrix A_p(A_remote->mat);
            LocalMatrix B_p(B_node->mat);
            C_p = LocalMatrix::spgemm(A_p, B_p);
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
    int * ncomplete;
    Task * tasks;
    int * claimed;
    dmmio::ProcessGrid * grid;
    MPI_Win task_win;
    MPI_Win claimed_win;
    MPI_Win ncomplete_win;

    static constexpr int ncomplete_owner = 0;
    static constexpr int coordinator_rank = 0;


    TaskQueue(dmmio::ProcessGrid * grid, int row_rank, int col_rank):
        ntasks(grid->global_size * grid->row_size), local_ntasks(grid->row_size), 
        tasks(new Task[grid->row_size]), claimed(new int[grid->row_size]),
        grid(grid)
    {
        // Initialize my tasks
        // TODO: This is not general
        int col_id = (grid->row_rank + grid->col_rank) % grid->row_size;
        int row_id = (grid->row_rank + grid->col_rank) % grid->row_size;
        for (int i=0; i<local_ntasks; i++)
        {
            tasks[i].col_rank_A = col_rank; //rank in process column
            tasks[i].row_rank_A = col_id; //rank in process row
            tasks[i].col_rank_B = row_id; 
            tasks[i].row_rank_B = row_rank;
            tasks[i].iter = i;
            tasks[i].owner = grid->global_rank;
            col_id = (col_id + 1) % grid->row_size;
            row_id = (row_id + 1) % grid->row_size;
            claimed[i] = -1;
        }


        // Seed RNG
        srand( (unsigned)time(NULL) ); 


        ntasks = local_ntasks * grid->global_size;
        ncomplete = new int(0);


        // Create MPI Windows 
        MPI_Win_create(tasks, sizeof(Task) * local_ntasks, sizeof(Task), MPI_INFO_NULL, grid->world_comm, &task_win);
        MPI_Win_create(claimed, sizeof(int) * local_ntasks, sizeof(int), MPI_INFO_NULL, grid->world_comm, &claimed_win);
        MPI_Win_create(ncomplete, sizeof(int), sizeof(int), MPI_INFO_NULL, grid->world_comm, &ncomplete_win);

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
        MPI_Bcast(result, 1, result->MPI_TASK, 0, grid->node_comm);
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

        if (result != -1)
        {
            return nullptr;
        }


        // Get the task
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, task_win);
        Task * task = new Task;
        MPI_Get(task, 1, task->MPI_TASK, rank, offset, 1, task->MPI_TASK, task_win);
        MPI_Win_unlock(rank, task_win);


        // Done
        return task;
    }



    void inc_n_complete()
    {
        // Increment global complete task count
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, ncomplete_owner, 0, ncomplete_win);
        int one = 1;
        MPI_Accumulate(&one, 1, MPI_INT, ncomplete_owner, 0, 1, MPI_INT, MPI_SUM, ncomplete_win);
        MPI_Win_unlock(ncomplete_owner, ncomplete_win);
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










