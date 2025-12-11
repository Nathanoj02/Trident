#include "hns_spgemm.cuh"
#include "sparse_summa.cuh"
#include "test_utils.cuh"

#include "mcl/args.hpp"

#include <ccutils/timers.h>

typedef int32_t IT;
typedef float VT;

void prune()
{
}


void inflation()
{
}


void normalize()
{
}


int main(int argc, char ** argv)
{
    Kokkos::initialize(argc, argv);
    {
    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);

    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MLCArgs *args = (MLCArgs*)malloc(sizeof(MLCArgs));
    parse_args(argc, argv, args);
    if(world_rank == 0) 
    {
        printf("\n================= TRILINOS MCL ==================\n");
        printf("Matrix: %s\n", args->mtx_name);
        printf("Max iterations: %u\n", args->max_iter);
        printf("Pruning tolerance: %f\n", args->pruning_tol);
        printf("Node size: %d\n", args->node_size);
    }
    FLUSH_WAIT(200000);

    MPI_Barrier(MPI_COMM_WORLD);


    // Set the process grid parameeters
    int nprocrows = world_size / args->node_size;
    int nproccols = nprocrows;

    // Compute all the respective partitioning types
    dmmio::Operation Aop, Bop;
    dmmio::PartitioningType Apart;

    Aop   = dmmio::Operation::None;
    Apart = dmmio::PartitioningType::Naive;
    Bop   = dmmio::Operation::None;

    // Reading the distributed matrices
    std::string A_mtx_path = (std::string) args->mtx_path;
    mmio::Matrix_Metadata *meta_A = new mmio::Matrix_Metadata();
    dmmio::DCOO<int32_t, float> *dcoo_A = dmmio::DCOO_read<int32_t, float>(
        A_mtx_path.c_str(),
        world_size, world_rank,
        nprocrows, nproccols, args->node_size,
        Apart, Aop,
        true, meta_A,
        MASK_SIZE
    );

    int gpn;
    CUDA_CHECK(cudaGetDeviceCount(&gpn));
    CUDA_CHECK(cudaSetDevice(world_rank % gpn));

    dmmio::utils::ProcessGrid_graph(dcoo_A->partitioning->grid, stdout, true);
    MPI_Barrier(MPI_COMM_WORLD);

    {

        DistCusparseCSX<int32_t, float> * dist_A1 = new DistCusparseCSX<int32_t, float>(dcoo_A, mmio::MajorDim::ROWS);
        DistCusparseCSX<int32_t, float> * dist_A2 = new DistCusparseCSX<int32_t, float>(dcoo_A, mmio::MajorDim::ROWS);
        DistCusparseCSX<int32_t, float> * dist_An;

        ThreadPool pool(2);

        std::vector<IT> nnz(args->max_iter, 0);
        std::vector<IT> nnz_pruned(args->max_iter, 0);
        int iter = 0;
        while (iter < args->max_iter) 
        {
            if(world_rank==0)fprintf(stdout, "\n===================== ITERATION %d ======================\n", iter);
            fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD);

            // Normalize

            // Expansion 
            if (args->impl == Implementation::SUMMA)
            {
                dist_An = sparse_summa(dist_A1, dist_A2);
            }
            else
            {
                dist_An = hns_spgemm_main<int32_t, float>(dist_A1, dist_A2, args->impl, pool, nullptr, 0);
            }
            MPI_Barrier(MPI_COMM_WORLD);


            // Inflation


            // Prune



            dist_A1 = dist_An;
            dist_A2 = dist_A1;


            delete dist_A1;
            delete dist_A2;
            iter++;
            fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        if(world_rank==0) 
        {
            printf("\n===================== Done MCL in %d iterations ======================\n", iter);
            printf("NNZ A_next:            ");
            for (size_t i = 0; i < iter; ++i) printf("%8lu ", nnz[i]);
            printf("\nNNZ A_next post prune: ");
            for (size_t i = 0; i < iter; ++i) printf("%8lu ", nnz_pruned[i]);
            printf("\n");
        }

    }

    dmmio::DCOO_destroy(&dcoo_A);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

#ifdef KOKKOS
    }
    Kokkos::finalize();
#endif
}









