#pragma once

#include "KokkosWrap.hpp"
#include "cusparse_helpers.cuh"

#include <unordered_set>

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
        owner(-1),
        iter(-1),
        MPI_TASK(MPIType<LocalSpGEMMTask<IT, VT>>())
    {}


    inline bool is_valid()
    {
        return owner > -1;
    }


    inline int ranks_to_global(dmmio::ProcessGrid * grid, int row_rank, int col_rank)
    {
        return (col_rank) * (grid->node_size * grid->row_size) + (row_rank * grid->node_size) + grid->node_rank;
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

        //print_rkn("Doing local spgemm\n");

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
                 

    std::string to_str()
    {
        std::stringstream ss;
        ss << "\tOwner: " << owner << "\n"
           << "\tRowRankA: " << row_rank_A << "\n"
           << "\tColRankA: " << col_rank_A << "\n"
           << "\tRowRankB: " << row_rank_B << "\n"
           << "\tColRankB: " << col_rank_B << "\n"
           << "\tIteration: " << iter << "\n";
        return ss.str();
    }


};


template <typename Task>
struct TaskQueue
{
    int ntasks;
    int local_ntasks;
    int local_offset;
    int ncomplete;
    Task * tasks;
    dmmio::ProcessGrid * grid;
    MPI_Win task_win;
    MPI_Win ncomplete_win;
    MPI_Win local_offset_win;

    std::unordered_set<int> finished_ranks;

    static constexpr int master_rank = 0;
    static constexpr int coordinator_rank = 0;


    TaskQueue(dmmio::ProcessGrid * grid, int row_rank, int col_rank):
        ntasks(grid->global_size * grid->row_size), local_ntasks(grid->row_size), 
        tasks(new Task[grid->row_size]),
        grid(grid), local_offset(0), ncomplete(0)
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
        }


        // Seed RNG
        srand( (unsigned)time(NULL) + grid->global_rank ); 


        ntasks = local_ntasks * grid->row_size * grid->col_size;


        // Create MPI Windows 
        MPI_Win_create(tasks, sizeof(Task) * local_ntasks, sizeof(Task), MPI_INFO_NULL, grid->world_comm, &task_win);
        MPI_Win_create(&ncomplete, sizeof(int), sizeof(int), MPI_INFO_NULL, grid->world_comm, &ncomplete_win);
        MPI_Win_create(&local_offset, sizeof(int), sizeof(int), MPI_INFO_NULL, grid->world_comm, &local_offset_win);

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
        int target_rank;
        if (is_coordinator())
        {
            // Try to get a rank that, to my knowledge, is not done
            for (int i=0; i<grid->global_size; i++)
            {
                target_rank = get_random_coordinator();
                if (!finished_ranks.contains(target_rank))
                {
                    break;
                }
            }
        }
        MPI_Bcast(&target_rank, 1, MPI_INT, coordinator_rank, grid->node_comm);
        return pop_task(target_rank);
    }



    Task * pop_local_task()
    {
        return pop_task((grid->global_rank / grid->node_size) * grid->node_size);
    }



    Task * pop_task(int rank)
    {
        assert(rank < grid->global_size);
        assert(is_coordinator(rank));


        Task * result;
        if (is_coordinator())
        {
            result = pop_task_coordinator(rank);
        }
        else
        {
            result = new Task;
        }

        MPI_Bcast(result, 1, result->MPI_TASK, coordinator_rank, grid->node_comm);
        result->owner += (result->owner == -1) ? 0 : grid->node_rank;
        return result;
    }



    Task * pop_task_coordinator(int rank)
    {

        assert(is_coordinator(rank));

        Task * task = new Task;

        // Claim task at the top of rank's queue
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, local_offset_win);
        int one = 1;
        int offset;
        MPI_Fetch_and_op(&one, &offset, MPI_INT, rank, 0, MPI_SUM, local_offset_win);
        MPI_Win_unlock(rank, local_offset_win);

        
        // If not less than local task count, it's invalid
        if (offset >= local_ntasks)
        {
            finished_ranks.emplace(rank);
            return task;
        }
        // Also, if I got the last task, I know this rank is finished
        else if (offset == (local_ntasks-1)) 
        {
            finished_ranks.emplace(rank);
        }


        // Get the task
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, task_win);
        MPI_Get(task, 1, task->MPI_TASK, rank, offset, 1, task->MPI_TASK, task_win);
        MPI_Win_unlock(rank, task_win);


        // Done
        return task;
    }



    void inc_n_complete()
    {
        // Increment global complete task count
        if (is_coordinator())
        {
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, master_rank, 0, ncomplete_win);
            int one = 1;
            MPI_Accumulate(&one, 1, MPI_INT, master_rank, 0, 1, MPI_INT, MPI_SUM, ncomplete_win);
            MPI_Win_unlock(master_rank, ncomplete_win);
        }
    }



    int check_n_complete()
    {
        int result;
        if (is_coordinator())
        {
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, master_rank, 0, ncomplete_win);
            MPI_Get(&result, 1, MPI_INT, master_rank, 0, 1, MPI_INT, ncomplete_win);
            MPI_Win_unlock(master_rank, ncomplete_win);
        }
        MPI_Bcast(&result, 1, MPI_INT, coordinator_rank, grid->node_comm);
        return result;
    }

    

    ~TaskQueue()
    {
        MPI_Win_free(&task_win);
        MPI_Win_free(&ncomplete_win);
        MPI_Win_free(&local_offset_win);
        delete[] tasks;
        MPI_Barrier(grid->world_comm);
    }
};










