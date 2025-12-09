#include "hns_spgemm.cuh"
#include "test_utils.cuh"

#include <ccutils/timers.h>

#define PRINT_MYDEV { \
    int dev; \
    cudaError_t err = cudaGetDevice(&dev); \
    if (err == cudaSuccess) { \
        printf("[%d] Rank %d is currently on CUDA device %d\n", __LINE__, world_rank, dev); \
    } else { \
        printf("cudaGetDevice failed: %s\n", cudaGetErrorString(err)); \
    } \
}

#define LOGFILE

int main(int argc, char ** argv)
{
#ifdef KOKKOS
    Kokkos::initialize(argc, argv);
    {
#endif
    const char* env = std::getenv("SLURM_LOCALID");
    int slurm_local_id = (env != nullptr) ? std::atoi(env) : 0;


    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);

    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank==0)
    {
        std::cout<<CYAN<<"----Running HnS-SpGEMM----"<<RESET<<std::endl;
    }

    Config * config = (Config *)(malloc(sizeof(Config)));
    parse_args(argc, argv, config);

    MPI_Barrier(MPI_COMM_WORLD);


    // Set the process grid parameeters
    int nprocrows = config->nprocrows, nproccols = config->nproccols, nprocpergroup = world_size/(nprocrows*nproccols);

    // Compute all the respective partitioning types
    dmmio::Operation Aop, Bop;
    dmmio::PartitioningType Apart;

    Aop   = dmmio::Operation::None;
    Apart = dmmio::PartitioningType::Naive;

    if (Apart == dmmio::PartitioningType::Naive) 
    {
        Bop   = dmmio::Operation::None;
    } 
    else 
    {
        fprintf(stderr, "Partitioning different by Naive are not supported yet.\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }

    if (world_rank == 0)
    {
      std::cout << "A matrix file path: " << config->matpathA << std::endl;
      std::cout << "B matrix file path: " << config->matpathB << std::endl;
      std::cout << "Number of processes per row: "  << nprocrows     << std::endl;
      std::cout << "Number of processes per col: "  << nproccols     << std::endl;
      std::cout << "Number of processes per node: " << nprocpergroup << std::endl;
      std::cout << "Chosen implementation: " << config->impl_str <<  std::endl;
      std::cout << "A stored in CSC format: " << config->Acsc << std::endl;
      std::cout << "Spcomm enabled: " << config->spcomm << " (It require --Acsc)" << std::endl;
      std::cout << "Compression threshold: " << COMP_THRESHOLD << " B" << std::endl;
      std::cout << "C_remote_size: " << config->c_remote_size << std::endl;
      std::cout << "Permute: " << config->permute << std::endl;
#ifdef ACCUM_THREAD
      std::cout << "Accumulation thread activated" << std::endl;
#endif
    }

    if (config->skip_ws)
    {
        if (world_rank == 0)
        {
            std::cout << "Workstealing skipped" << std::endl;
        }
    }

    // Checks on the input params
    {
        if (config->spcomm && (!config->Acsc)) 
        {
            if (world_rank == 0) fprintf(stderr, "Error: --spcomm requires --Acsc\n");
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Abort(MPI_COMM_WORLD, __LINE__);
        }

#ifndef SKIP_SPGEMM
        if(config->skip_spgemm) 
        {
            if (world_rank == 0) fprintf(stderr, "WARNING: --skip-spgemm is a DEBUG ONLY flag, local SpGEMM computation will be skipped\n");
        }
#else
        if (world_rank == 0) fprintf(stderr, "WARNING: -DSKIP_SPGEMM is a DEBUG ONLY flag, local SpGEMM computation will be skipped\n");
#endif

    }

    // Reading the distributed matrices
    std::string A_mtx_path = (std::string) config->matpathA;
    mmio::Matrix_Metadata *meta_A = new mmio::Matrix_Metadata();
    dmmio::DCOO<int32_t, float> *dcoo_A = dmmio::DCOO_read<int32_t, float>(
        A_mtx_path.c_str(),
        world_size, world_rank,
        nprocrows, nproccols, nprocpergroup,
        Apart, Aop,
        true, meta_A,
        MASK_SIZE,
        config->permute
    );

    std::string B_mtx_path = (std::string) config->matpathB;
    mmio::Matrix_Metadata *meta_B = new mmio::Matrix_Metadata();
    dmmio::DCOO<int32_t, float> *dcoo_B = dmmio::DCOO_read<int32_t, float>(
        B_mtx_path.c_str(),
        world_size, world_rank,
        nprocrows, nproccols, nprocpergroup,
        Apart, Bop,
        true, meta_B,
        MASK_SIZE,
        config->permute,
        dcoo_A->permutation
    );

    if (world_rank==0) fprintf(stdout, "A matrix: %dx%d, B matrix: %dx%d\n", dcoo_A->coo->nrows, dcoo_A->coo->ncols, dcoo_B->coo->nrows, dcoo_B->coo->ncols); fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    int gpn;
    CUDA_CHECK(cudaGetDeviceCount(&gpn));
    CUDA_CHECK(cudaSetDevice(world_rank % gpn));


    dmmio::utils::ProcessGrid_graph(dcoo_A->partitioning->grid, stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    {
        if (world_rank==0) printf("Beginning conversion\n");
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);

        mmio::MajorDim A_maj = (config->Acsc) ? (mmio::MajorDim::COLS) : (mmio::MajorDim::ROWS) ;

        DistCusparseCSX<int32_t, float> * dist_A = new DistCusparseCSX<int32_t, float>(dcoo_A, A_maj);
        DistCusparseCSX<int32_t, float> * dist_B = new DistCusparseCSX<int32_t, float>(dcoo_B, mmio::MajorDim::ROWS);
        DistCusparseCSX<int32_t, float> * dist_C;

        if (world_rank==0)  printf("Done conversion\n");
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);

        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);


        CPU_TIMER_DEF(spgemm);
        CPU_TIMER_DEF(spacomm);

#ifdef NVTX_PROFILING
        cudaProfilerStart();
        NVTX_PUSH_RANGE("spcomm",1);
#endif

        CPU_TIMER_START(spacomm);

        SpaComm::SpaCommHandler<int32_t, float> *spcomm_data = nullptr;
        if (config->spcomm) 
        {
            spcomm_data = new SpaComm::SpaCommHandler<int32_t, float>(dist_A->csx->mat, dist_B->csx->mat, dist_A->partitioning->grid);
        } 
        else 
        {
            spcomm_data = nullptr;
        }

        CPU_TIMER_STOP(spacomm);

        if (world_rank==0)
        {
            TIMER_PRINT(spacomm);
        }

#ifdef NVTX_PROFILING
        NVTX_POP_RANGE;
#endif


        CPU_TIMER_START(spgemm);

        // Gen thread pool (mostly for profiling)
        ThreadPool pool(2);

        //const int niters = 6;
        const int niters = 6;

        if (world_rank==0) printf("Beginning spgemm -- implementation: %s\n", config->impl_str);

        for (int i=0; i<niters; i++) 
        {
            if (world_rank==0) printf("STARTING spgemm round: %d\n", i);
            fflush(stdout);
            sleep(0.2);

#ifdef NVTX_PROFILING
            NVTX_PUSH_RANGE("spgemm",1);
#endif

            MPI_Barrier(MPI_COMM_WORLD);
            hns_spgemm_main<int32_t, float>(dist_A, dist_B, config->impl, pool, spcomm_data, config->c_remote_size, config->skip_spgemm, config->skip_ws);
            MPI_Barrier(MPI_COMM_WORLD);

#ifdef NVTX_PROFILING
            NVTX_POP_RANGE;
#endif

            fflush(stdout);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

#ifdef NVTX_PROFILING
        cudaProfilerStop();
#endif
        if (spcomm_data != nullptr) delete spcomm_data;
        delete dist_A;
        delete dist_B;
    }

    dmmio::DCOO_destroy(&dcoo_A);
    dmmio::DCOO_destroy(&dcoo_B);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#ifdef KOKKOS
    }
    Kokkos::finalize();
#endif
}
