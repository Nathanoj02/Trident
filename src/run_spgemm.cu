#include "hns_spgemm.cuh"
#include "hns_spgemm_get.cuh"
#include "test_utils.cuh"
#include <ccutils/timers.h>
#include <Kokkos_Core.hpp>

#define PRINT_MYDEV { \
    int dev; \
    cudaError_t err = cudaGetDevice(&dev); \
    if (err == cudaSuccess) { \
        printf("[%d] Rank %d is currently on CUDA device %d\n", __LINE__, world_rank, dev); \
    } else { \
        printf("cudaGetDevice failed: %s\n", cudaGetErrorString(err)); \
    } \
}


int main(int argc, char ** argv)
{
    Kokkos::initialize(argc, argv);
    {

    const char* env = std::getenv("SLURM_LOCALID");
    int slurm_local_id = (env != nullptr) ? std::atoi(env) : 0;

    int numDevices;
    //cudaError_t err = cudaGetDeviceCount(&numDevices);
    //int mydev = slurm_local_id % numDevices;
    //            err = cudaSetDevice(mydev);

    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);

    //int dev;
    //err = cudaGetDevice(&dev);

    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank==0)
    {
        std::cout<<CYAN<<"----Running HnS-SpGEMM----"<<RESET<<std::endl;
    }

    std::string logname("log_rk_" + std::to_string(world_rank) + ".out");
    FILE * logfile = fopen(logname.c_str(), "w");

    Config * config = (Config *)(malloc(sizeof(Config)));
    parse_args(argc, argv, config);

    int name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &name_len);

    MPI_Barrier(MPI_COMM_WORLD);


    // Set the process grid parameeters
    int nprocrows = config->nprocrows, nproccols = config->nproccols, nprocpergroup = world_size/(nprocrows*nproccols);

    // Compute all the respective partitioning types
    dmmio::Operation Aop, Bop, Cop;
    dmmio::PartitioningType Apart, Bpart, Cpart;

    Aop   = dmmio::Operation::None;
    Apart = dmmio::PartitioningType::Naive;

    if (Apart == dmmio::PartitioningType::Naive) {
        Bpart = dmmio::PartitioningType::Naive;
        Cpart = dmmio::PartitioningType::Naive;

        Bop   = dmmio::Operation::None;
        Cop   = dmmio::Operation::None;
    } else {
        // TODO wirite it for other partitionings and check it
        fprintf(stderr, "Partitioning different by Naive are not supported yet.\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }

    // Some prints
    if (world_rank == 0)
    {
      std::cout << "A matrix file path: " << config->matpathA << std::endl;
      std::cout << "B matrix file path: " << config->matpathB << std::endl;
      std::cout << "Number of processes per row: "  << nprocrows     << std::endl;
      std::cout << "Number of processes per col: "  << nproccols     << std::endl;
      std::cout << "Number of processes per node: " << nprocpergroup << std::endl;
      std::cout << "Chosen implementation: " << config->impl << "(main use MPI_Put)" << std::endl;
      std::cout << "A stored in CSC format: " << config->Acsc << std::endl;
      std::cout << "Spcomm enabled: " << config->spcomm << " (It require --Acsc)" << std::endl;
    }

    // Checks on the input params
    {
        if (config->spcomm && (!config->Acsc)) {
            if (world_rank == 0) fprintf(stderr, "Error: --spcomm requires --Acsc\n");
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Abort(MPI_COMM_WORLD, __LINE__);
        }
        // NOTE: strcmp(a,b) return 0 if a == b, meaning that if a==b than 'if(strcmp(a,b))' is false
        if (strcmp(config->impl, "get") && strcmp(config->impl, "main")) {
            if (world_rank == 0) fprintf(stderr, "Error: supported implementations are main or get (not %s)\n", config->impl);
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Abort(MPI_COMM_WORLD, __LINE__);
        }
        if (!strcmp(config->impl, "get") && (config->Acsc || config->spcomm)) {
            if (world_rank == 0) fprintf(stderr, "Error: --spcomm and --Acsc are not supported with --impl get\n");
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Abort(MPI_COMM_WORLD, __LINE__);
        }

        // TODO
        if ( config->spcomm && nprocpergroup > 1) {
            if (world_rank == 0) fprintf(stderr, "Error: on 3D grids, --spcomm is not supported yet\n");
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Abort(MPI_COMM_WORLD, __LINE__);
        }
    }

    // Reading the distribuited matrices
    std::string A_mtx_path = (std::string) config->matpathA;
    mmio::Matrix_Metadata *meta_A = new mmio::Matrix_Metadata();
    dmmio::DCOO<int32_t, float> *dcoo_A = dmmio::DCOO_read<int32_t, float>(
        A_mtx_path.c_str(),
        world_size, world_rank,
        nprocrows, nproccols, nprocpergroup,
        Apart, Aop,
        false, meta_A
    );

    std::string B_mtx_path = (std::string) config->matpathB;
    mmio::Matrix_Metadata *meta_B = new mmio::Matrix_Metadata();
    dmmio::DCOO<int32_t, float> *dcoo_B = dmmio::DCOO_read<int32_t, float>(
        A_mtx_path.c_str(),
        world_size, world_rank,
        nprocrows, nproccols, nprocpergroup,
        Apart, Bop,
        false, meta_B
    );

    if (dcoo_A->partitioning->grid->node_size > 1)
    {
        cudaSetDevice(dcoo_A->partitioning->grid->node_rank);
    }
    // This causes a bunch of complaints related to IPC -- not sure what these are or if they matter

    dmmio::utils::ProcessGrid_graph(dcoo_A->partitioning->grid, stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    {
        if (world_rank==0) printf("Beginning conversion\n");
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);

        mmio::MajorDim A_maj = (config->Acsc) ? (mmio::MajorDim::COLS) : (mmio::MajorDim::ROWS) ;
        KokkosWrap::DistribuitedMatrix<int32_t, int32_t, float> wrapped_A(dcoo_A, A_maj);
        KokkosWrap::DistribuitedMatrix<int32_t, int32_t, float> wrapped_B(dcoo_B, mmio::MajorDim::ROWS);

        if (world_rank==0)  printf("Done conversion\n");
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);



        mmio::CSX<int32_t, float> *dist_C;
        CPU_TIMER_DEF(spgemm);
        CPU_TIMER_DEF(spacomm);
        if (world_rank==0) printf("Beginning spgemm -- implementation: %s\n", config->impl);
        for (int i=0; i<50; i++) 
        {
            CPU_TIMER_START(spacomm);

            SpaComm::SpaCommHandler<int32_t, float> *spcomm_data;
            if (config->spcomm) {
                spcomm_data = new SpaComm::SpaCommHandler<int32_t, float>(wrapped_A, wrapped_B);


                // ----- debug -----
                /*{
                    size_t size = (spcomm_data->mask_len)*(wrapped_A.partitioning->grid->row_size);
                    int8_t* hcolmaps = (int8_t*)malloc(sizeof(int8_t)*size);
                    cudaMemcpy(hcolmaps, spcomm_data->A_column_filters, size, cudaMemcpyDeviceToHost);
                    SpaComm::printBit_left2right(hcolmaps, size, stdout);
                    free(hcolmaps);
                }*/
                // -----------------
            } else {
                spcomm_data = nullptr;
            }

            CPU_TIMER_STOP(spacomm);
            if (world_rank==0)
            {
                TIMER_PRINT(spacomm);
            }
/*
            CPU_TIMER_START(spgemm);

            if (!strcmp(config->impl, "main"))
            {
                dist_C = hns_spgemm_main<int32_t, float>(wrapped_A, wrapped_B, spcomm_data);
            }
            else if (!strcmp(config->impl, "get"))
            {
                dist_C = hns_spgemm_get<int32_t, float>(wrapped_A, wrapped_B);
            }

            CPU_TIMER_STOP(spgemm);
            if (world_rank==0)
            {
                TIMER_PRINT(spgemm);
            }
            delete dist_C;
*/
        }
        if (world_rank==0) printf("Done spgemm\n");
    }

    dmmio::DCOO_destroy(&dcoo_A);
    dmmio::DCOO_destroy(&dcoo_B);

    fclose(logfile);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    }
    Kokkos::finalize();
}
