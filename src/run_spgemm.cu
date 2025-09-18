#include "hns_spgemm.cuh"
#include "test_utils.cuh"
#include <ccutils/timers.h>



int main(int argc, char ** argv)
{

    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);


    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

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
    Apart = (dmmio::PartitioningType)config->part_num;

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

    cudaSetDevice(dcoo_A->partitioning->grid->node_rank);
    // This causes a bunch of complaints related to IPC -- not sure what these are or if they matter
    Kokkos::initialize(argc, argv);


    // Some prints
    if (world_rank == 0) 
    {
      std::cout << "A matrix file path: " << config->matpathA << std::endl;
      std::cout << "B matrix file path: " << config->matpathB << std::endl;
      std::cout << "Number of processes per row: "  << nprocrows     << std::endl;
      std::cout << "Number of processes per col: "  << nproccols     << std::endl;
      std::cout << "Number of processes per node: " << nprocpergroup << std::endl;
    }

    dmmio::utils::ProcessGrid_graph(dcoo_A->partitioning->grid, stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    {
        MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Beginning conversion\n"));
        fflush(stdout);

        dmmio::partitioning::indextransform::transformCoo::global2group(dcoo_A);
        dmmio::partitioning::indextransform::transformCoo::global2group(dcoo_B);
        KokkosWrap::DistribuitedMatrix<int32_t, int32_t, float> wrapped_A(dcoo_A, mmio::MajorDim::COLS);
        KokkosWrap::DistribuitedMatrix<int32_t, int32_t, float> wrapped_B(dcoo_B, mmio::MajorDim::ROWS);

        MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Done conversion\n"));
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);

        CPU_TIMER_DEF(spgemm);

        MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Beginning spgemm computations\n"));
        for (int i=0; i<50; i++)
        {
            CPU_TIMER_START(spgemm);
            // DistCSR<int32_t, float> * dist_C = hns_spgemm_main(dist_A, dist_B);
            mmio::CSX<int32_t, float> *dist_C = hns_spgemm_main<int32_t, float>(wrapped_A, wrapped_B);
            CPU_TIMER_STOP(spgemm);
            if (world_rank==0)
            {
                TIMER_PRINT(spgemm);
            }
            delete dist_C;
        }
        MPI_PROCESS_PRINT(MPI_COMM_WORLD, 0, printf("Done spgemm computations\n"));

        // delete dist_A;
        // delete dist_B;
    }

    dmmio::DCOO_destroy(&dcoo_A);
    dmmio::DCOO_destroy(&dcoo_B);

    Kokkos::finalize();
    fclose(logfile);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
