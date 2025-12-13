#include "hns_spgemm.cuh"
#include "sparse_summa.cuh"
#include "test_utils.cuh"
#include "utils.cuh"

#undef DETAILED_TIMERS

#include "mcl/args.hpp"

#include <ccutils/timers.h>
#include <ccutils/cuda/cuda_utils.hpp>

typedef int32_t IT;
typedef float VT;


CusparseCSX<IT, VT> * to_gpu_csx(mmio::CSX<IT, VT> * cpu_csx)
{

    IT * d_colinds = h2d_copy(cpu_csx->idx_vec, cpu_csx->nnz);
    IT * d_rowptrs = h2d_copy(cpu_csx->ptr_vec, cpu_csx->nrows + 1);
    VT * d_vals = h2d_copy(cpu_csx->val, cpu_csx->nnz);

    mmio::CSX<IT, VT> * gpu_csx = CSX_create_contig_device<IT, VT>(cpu_csx->nrows, 
                                                     cpu_csx->ncols,
                                                     cpu_csx->nnz,
                                                     mmio::MajorDim::ROWS,
                                                     d_colinds, d_rowptrs,
                                                     d_vals);

    return new CusparseCSX<IT, VT>(gpu_csx);
}


mmio::CSX<IT, VT> * to_cpu_csx(CusparseCSX<IT, VT> * cusparse_csx)
{
    mmio::CSX<IT, VT> * gpu_csx = cusparse_csx->mat;

    IT * h_colinds = d2h_copy(gpu_csx->idx_vec, gpu_csx->nnz);
    IT * h_rowptrs = d2h_copy(gpu_csx->ptr_vec, gpu_csx->nrows + 1);
    VT * h_vals = d2h_copy(gpu_csx->val, gpu_csx->nnz);


    mmio::CSX<IT, VT> * cpu_csx = mmio::CSX_create<IT, VT>(gpu_csx->nrows, 
                                                            gpu_csx->ncols,
                                                            gpu_csx->nnz,
                                                            mmio::MajorDim::ROWS,
                                                            h_rowptrs, h_colinds, 
                                                            h_vals);

    return cpu_csx;
}



void prune(DistCusparseCSX<IT, VT> * dist_csx, VT tol)
{

    CusparseCSX<IT, VT> * local_csx = dist_csx->csx;
    mmio::CSX<IT, VT> * cpu_csx = to_cpu_csx(local_csx);
    local_csx->explicit_free();
    std::vector<IT> p_colinds;
    std::vector<IT> p_rowptrs(cpu_csx->nrows+1, 0);
    std::vector<VT> p_vals;


    for (int i=0; i<cpu_csx->nrows; i++)
    {
        IT row_nnz = 0;
        for (int j=cpu_csx->ptr_vec[i]; j<cpu_csx->ptr_vec[i+1]; j++)
        {
            VT val = cpu_csx->val[j];
            IT cid = cpu_csx->idx_vec[j];
            if (std::abs(val) > tol)
            {
                p_vals.push_back(val);
                p_colinds.push_back(cid);
                row_nnz++;
            }
        }
        p_rowptrs[i+1] = row_nnz;
    }

    std::inclusive_scan(p_rowptrs.begin() + 1, p_rowptrs.end(), p_rowptrs.begin() + 1);
    IT tot_nnz = p_rowptrs[cpu_csx->nrows];

    mmio::CSX<IT, VT> * gpu_csx = CSX_create_contig_device<IT, VT>(cpu_csx->nrows, cpu_csx->ncols, tot_nnz, mmio::MajorDim::ROWS, p_colinds.data(), p_rowptrs.data(), p_vals.data());


    mmio::CSX_destroy<IT, VT>(&cpu_csx);

    dist_csx->csx = new CusparseCSX<IT, VT>(gpu_csx);
}



void normalize(DistCusparseCSX<IT, VT> * dist_csx)
{

    CusparseCSX<IT, VT> * local_csx = dist_csx->csx;
    auto cpu_csx = to_cpu_csx(local_csx);

    std::vector<IT> row_sums(cpu_csx->nrows, 0);

    for (IT i = 0; i < cpu_csx->nrows; i++)
    {
        for (int j=cpu_csx->ptr_vec[i]; j<cpu_csx->ptr_vec[i+1]; j++)
        {
            row_sums[i] += cpu_csx->val[j];
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, row_sums.data(), row_sums.size(),
                  MPI_INT32_T, MPI_SUM, MPI_COMM_WORLD);

    for (IT i = 0; i < cpu_csx->nrows; i++)
    {
        IT s = row_sums[i];
        for (int j=cpu_csx->ptr_vec[i]; j<cpu_csx->ptr_vec[i+1]; j++)
        {
            cpu_csx->val[j] = (s > 1e-20) ? 1.0 / s : 0.0;
        }
    }

    h2d_copy(local_csx->mat->val, local_csx->mat->nnz, cpu_csx->val);
    mmio::CSX_destroy<IT, VT>(&cpu_csx);
}


void inflation(DistCusparseCSX<IT, VT> * dist_csx, const float power = 2.0f)
{
    CusparseCSX<IT, VT> * local_csx = dist_csx->csx;
    mmio::CSX<IT, VT> * cpu_csx = to_cpu_csx(local_csx);


    for (int i=0; i<cpu_csx->nrows; i++)
    {
        for (int j=cpu_csx->ptr_vec[i]; j<cpu_csx->ptr_vec[i+1]; j++)
        {
            cpu_csx->val[j] = std::pow(cpu_csx->val[j], power);
        }
    }

    mmio::CSX_destroy<IT, VT>(&cpu_csx);

    normalize(dist_csx);
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
    int nprocrows = (int)std::ceil(std::sqrt(world_size / args->node_size));
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
    dmmio::DCOO<int32_t, float> *dcoo_A1 = dmmio::DCOO_read<int32_t, float>(
        A_mtx_path.c_str(),
        world_size, world_rank,
        nprocrows, nproccols, args->node_size,
        Apart, Aop,
        true, meta_A,
        MASK_SIZE, false, nullptr, true
    );

    dmmio::DCOO<int32_t, float> *dcoo_A2 = dmmio::DCOO_read<int32_t, float>(
        A_mtx_path.c_str(),
        world_size, world_rank,
        nprocrows, nproccols, args->node_size,
        Apart, Aop,
        true, meta_A,
        MASK_SIZE, false, nullptr, true
    );
    
    if (world_rank == 0)
    {
        printf("Done IO\n");
        fflush(stdout);
    }

    int gpn;
    CUDA_CHECK(cudaGetDeviceCount(&gpn));
    CUDA_CHECK(cudaSetDevice(world_rank % gpn));

    dmmio::utils::ProcessGrid_graph(dcoo_A1->partitioning->grid, stdout, true);
    MPI_Barrier(MPI_COMM_WORLD);

    {

        DistCusparseCSX<int32_t, float> * dist_A1 = new DistCusparseCSX<int32_t, float>(dcoo_A1, mmio::MajorDim::ROWS);
        DistCusparseCSX<int32_t, float> * dist_A2 = new DistCusparseCSX<int32_t, float>(dcoo_A2, mmio::MajorDim::ROWS);
        DistCusparseCSX<int32_t, float> * dist_An;


        ThreadPool pool(2);

        std::vector<IT> nnz(args->max_iter, 0);
        std::vector<IT> nnz_pruned(args->max_iter, 0);
        int iter = 0;

        // Normalize
        print_rk0("First normalization\n");
        normalize(dist_A1);

        while (iter < args->max_iter) 
        {
            if(world_rank==0)fprintf(stdout, "\n===================== ITERATION %d ======================\n", iter);
            fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD);


            // Expansion 
            print_rk0("Expansion\n");
            if (!strcmp(args->impl, "summa"))
            {
                dist_An = sparse_summa(dist_A1, dist_A2);
            }
            else
            {
                dist_An = hns_spgemm_main<int32_t, float>(dist_A1, dist_A2, Implementation::ASYNC, pool, nullptr, 0);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            print_rk0("Done expansion\n");

            nnz[iter] = dist_An->getGlobalNnz();

            // Inflation
            print_rk0("Inflation\n");
            inflation(dist_An);


            // Prune
            print_rk0("Pruning\n");
            prune(dist_An, args->pruning_tol);

            nnz_pruned[iter] = dist_An->getGlobalNnz();

            delete dist_A1;

            // They are the same for subsequent iterations
            if (iter == 0)
            {
                delete dist_A2;
            }


            dist_A1 = dist_An;
            dist_A2 = dist_A1;


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

    dmmio::DCOO_destroy(&dcoo_A1);
    dmmio::DCOO_destroy(&dcoo_A2);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

#ifdef KOKKOS
    }
    Kokkos::finalize();
#endif
}









