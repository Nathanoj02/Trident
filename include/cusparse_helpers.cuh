#pragma once

#include "common.h"
#include "utils.cuh"

#include <dmmio/dmmio.h>
#include <dmmio/partitioning.h>



using namespace mmio;
using namespace dmmio;

template <typename IT, typename VT>
struct Triple
{
    IT row;
    IT col;
    VT val;
};

template <typename IT, typename VT>
CSX<IT,VT>* coo_to_row_csx_contig(COO<IT, VT> * coo)
{
    using Tr = Triple<IT, VT>;

    // Sort by row
    std::vector<Tr> triples(coo->nnz);
    for (IT i=0; i<coo->nnz; i++)
    {
        triples[i].row = coo->row[i];
        triples[i].col = coo->col[i];
        triples[i].val = coo->val[i];
    }


    std::sort(triples.begin(), triples.end(),
        [](auto& t1, auto& t2)
        {
            return t1.row < t2.row;
        }
    );


    // First convert the local COO representation to a CSR representation on the host
    std::vector<IT> rowptrs(coo->nrows + 1, 0);
    std::for_each(triples.begin(), triples.end(),
        [&](auto& t)
        {
            rowptrs[t.row+1]++;
        }
    );

    std::vector<IT> colinds(coo->nnz);
    std::transform(triples.begin(), triples.end(),
                   colinds.begin(),
        [&](auto& t)
        {
            return t.col;
        }
    );

    std::vector<VT> vals(coo->nnz);
    std::transform(triples.begin(), triples.end(),
                   vals.begin(),
        [&](auto& t)
        {
            return t.val;
        }
    );


    std::inclusive_scan(rowptrs.begin() + 1, rowptrs.end(), rowptrs.begin() + 1);


    // Convert to a mmio row_major CSX matrix (i.e. a csr csx)
    return(CSX_create_contig_device(coo->nrows, coo->ncols, coo->nnz, mmio::MajorDim::ROWS,
                            colinds.data(), rowptrs.data(), vals.data()));
}


template <typename IT, typename VT>
CSX<IT,VT>* coo_to_col_csx_contig(COO<IT, VT> * coo)
{
    using Tr = Triple<IT, VT>;

    // Sort by row
    std::vector<Tr> triples(coo->nnz);
    for (IT i=0; i<coo->nnz; i++)
    {
        triples[i].row = coo->row[i];
        triples[i].col = coo->col[i];
        triples[i].val = coo->val[i];
    }

    std::sort(triples.begin(), triples.end(),
        [](auto& t1, auto& t2)
        {
            return t1.col < t2.col;
        }
    );

    // First convert the local COO representation to a CSR representation on the host
    std::vector<IT> colptrs(coo->ncols + 1, 0);
    std::for_each(triples.begin(), triples.end(),
        [&](auto& t)
        {
            colptrs[t.col+1]++;
        }
    );

    std::vector<IT> rowinds(coo->nnz);
    std::transform(triples.begin(), triples.end(),
                   rowinds.begin(),
        [&](auto& t)
        {
            return t.row;
        }
    );

    std::vector<VT> vals(coo->nnz);
    std::transform(triples.begin(), triples.end(),
                   vals.begin(),
        [&](auto& t)
        {
            return t.val;
        }
    );

    std::inclusive_scan(colptrs.begin() + 1, colptrs.end(), colptrs.begin() + 1);


    // Convert to a csx matrix
    return(CSX_create_contig_device(coo->nrows, coo->ncols, coo->nnz, mmio::MajorDim::COLS,
                            rowinds.data(), colptrs.data(), vals.data()));
}


template <typename IT, typename VT>
struct CusparseCSX
{

    enum class State
    {
        Mat,
        Buffs, 
        Null
    };

    cusparseSpMatDescr_t descr;
    cusparseMatDescr_t mat_descr; // nice names
    CSX<IT, VT> * mat;
    CsxBuffers<IT, VT> * buffers;
    State state;

    CusparseCSX():
        mat(nullptr), buffers(nullptr), state(State::Null)
    {}

    CusparseCSX(CSX<IT, VT> * mat): 
        mat(mat), buffers(nullptr), state(State::Mat)
    {
        if (mat->majordim == MajorDim::ROWS)
        {
            CUSPARSE_CHECK(cusparseCreateCsr(&descr,
                                             mat->nrows,
                                             mat->ncols,
                                             mat->nnz,
                                             mat->ptr_vec,
                                             mat->idx_vec,
                                             mat->val,
                                             CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO,
                                             CUDA_R_32F));
            CUSPARSE_CHECK(cusparseCreateMatDescr(&mat_descr));
        }
    }


    CusparseCSX(cusparseHandle_t* handle, 
                CSX<IT, VT> * _mat,
                CsxBuffers<IT, VT> * conversion_buffers):
        mat(nullptr), buffers(nullptr), state(State::Mat)
    {
        if (_mat->majordim == MajorDim::COLS)
        {
            mat = cusparse_csc_to_csr(handle, _mat, conversion_buffers);
        }
        else
        {
            mat = _mat;
        }
        CUSPARSE_CHECK(cusparseCreateCsr(&descr,
                                         mat->nrows,
                                         mat->ncols,
                                         mat->nnz,
                                         mat->ptr_vec,
                                         mat->idx_vec,
                                         mat->val,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateMatDescr(&mat_descr));
    }


    CusparseCSX(CsxBuffers<IT, VT> * buffers):
        mat(nullptr), buffers(buffers), state(State::Buffs)
    {

        CUSPARSE_CHECK(cusparseCreateCsr(&descr,
                                         buffers->ptr_dim-1,
                                         buffers->other_dim,
                                         buffers->nnz,
                                         buffers->d_node_rowptrs,
                                         buffers->d_node_colinds,
                                         buffers->d_node_vals,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateMatDescr(&mat_descr));
    }



    inline bool is_mat()
    {
        return state == State::Mat;
    }


    inline bool is_buffs()
    {
        return state == State::Buffs;
    }


    inline bool is_null()
    {
        return state == State::Null;
    }



    void validate_csr()
    {
        assert(is_mat());

        IT * h_colinds = d2h_copy(mat->idx_vec, mat->nnz);
        IT * h_rowptrs = d2h_copy(mat->ptr_vec, mat->nrows+1);


        // Validate colinds
        bool colinds_valid = true;
        for (IT i=0; i<mat->nnz;i++)
        {
            if (h_colinds[i] < 0 || h_colinds[i] >= mat->ncols)
            {
                colinds_valid = false;
                break;
            }
        }
        

        // Validate rowptrs
        bool rowptrs_valid = true;
        for (IT i=0; i<mat->nrows; i++)
        {
            if (i==0)
            {
                rowptrs_valid = (h_rowptrs[0] == 0);
            }
            rowptrs_valid = (h_rowptrs[i] <= h_rowptrs[i+1]) && rowptrs_valid;
        }
        rowptrs_valid = rowptrs_valid && (h_rowptrs[mat->nrows] == mat->nnz);



        printf("Colinds valid: %d, rowptrs_valid: %d\n", colinds_valid, rowptrs_valid);
        fflush(stdout);


        free(h_colinds);
        free(h_rowptrs);
    }


    inline IT nrows()
    {
        if (is_mat())
        {
            return mat->nrows;
        }
        else if (is_buffs())
        {
            return buffers->ptr_dim-1;
        }
        else
        {
            return 0;
        }
    }


    inline IT ncols()
    {
        if (is_mat())
        {
            return mat->ncols;
        }
        else if (is_buffs())
        {
            return buffers->other_dim;
        }
        else
        {
            return 0;
        }
    }


    inline IT nnz()
    {
        if (is_mat())
        {
            return mat->nnz;
        }
        else if (is_buffs())
        {
            return buffers->nnz;
        }
        else
        {
            return 0;
        }
    }


    void explicit_free()
    {
        if (is_mat())
        {
            CSX_destroy_device(&mat);
        }
        else if (is_buffs())
        {
            buffers->explicitFree();
        }
    }


    void explicit_free_async()
    {
        if (is_mat())
        {
            CSX_destroy_device(&mat);
        }
        else if (is_buffs())
        {
            buffers->explicitFreeAsync();
        }
    }


    void assert_mat(const char * name)
    {
        std::stringstream ss;
        ss << "Exepcted "<<name<< " underlying storage to be a mmio::CSX object";
        assert( is_mat() && ss.str().c_str());
    }


    void assert_buffs(const char * name)
    {
        std::stringstream ss;
        ss << "Exepcted "<<name<< " underlying storage to be a CsxBuffers object";
        assert( is_buffs() && ss.str().c_str());
    }


    ~CusparseCSX()
    {
        if (is_null())
        {
            return;
        }

        if (is_mat())
        {
            CUSPARSE_CHECK(cusparseDestroySpMat(descr));
            CUSPARSE_CHECK(cusparseDestroyMatDescr(mat_descr));

            if (mat->majordim == MajorDim::COLS)
            {
                CSX_destroy_device<IT, VT>(&mat);
            }
        }

        if (is_buffs()) 
        {
            buffers->explicitFree();
        }
    }
};


template <typename IT, typename VT>
struct DistCusparseCSX
{

    Partitioning * partitioning;
    CusparseCSX<IT, VT> * csx;

    DistCusparseCSX(){}

    DistCusparseCSX(CusparseCSX<IT, VT> * mat, Partitioning * part):
        csx(mat), partitioning(part)
    {}


    DistCusparseCSX(dmmio::DCOO<IT, VT> * dcoo, MajorDim T):
        partitioning(dcoo->partitioning)
    {
        using namespace dmmio::partitioning::indextransform;


        while (dcoo->coo->nrows % (dcoo->partitioning->grid->col_size * dcoo->partitioning->grid->node_size * MASK_SIZE) != 0 &&
                dcoo->coo->nrows % (dcoo->partitioning->grid->row_size * MASK_SIZE))
        {
            dcoo->coo->nrows++;
        }

        while (dcoo->coo->ncols % (dcoo->partitioning->grid->col_size * dcoo->partitioning->grid->node_size * MASK_SIZE) != 0 &&
                dcoo->coo->ncols % (dcoo->partitioning->grid->row_size * MASK_SIZE))
        {
            dcoo->coo->ncols++;
        }

        IT max_dim = max(dcoo->coo->ncols, dcoo->coo->nrows);


        dcoo->coo->nrows /= (dcoo->partitioning->grid->col_size * dcoo->partitioning->grid->node_size);
        dcoo->coo->ncols /= dcoo->partitioning->grid->row_size;


        for (IT i=0; i<dcoo->coo->nnz; i++)
        {
            dcoo->coo->row[i] = global2local::row(dcoo->partitioning, dcoo->coo->row[i]);
            dcoo->coo->col[i] = global2local::col(dcoo->partitioning, dcoo->coo->col[i]);
        }


        auto coo = dcoo->coo;

        CSX<IT, VT> * mat;

        // --- Step 2: Decide layout ---
        if (T == MajorDim::ROWS) 
        {
            mat = coo_to_row_csx_contig(coo);
        } 
        else 
        {
            mat = coo_to_col_csx_contig(coo);
        }
        
        csx = new CusparseCSX<IT, VT>(mat);

        CUDA_CHECK(cudaDeviceSynchronize());
    }


    IT getLocalPtrvecsize()
    {
        if (csx->is_buffs())
        {
            return csx->nrows();
        }
        if (csx->mat->majordim == MajorDim::ROWS)
        {
            return csx->nrows();
        }
        else
        {
            return csx->ncols();
        }
    }


    inline IT getLocalNrows()
    {
        return csx->nrows();
    }


    inline IT getLocalNcols()
    {
        return csx->ncols();
    }


    inline IT getLocalNnz()
    {
        return csx->nnz();
    }


};


template <typename IT, typename VT>
CSX<IT, VT> * cusparse_csc_to_csr(cusparseHandle_t* handle, CSX<IT, VT> * csc, CsxBuffers<IT, VT> * buffers)
{
    cudaStream_t * stream = buffers->stream;

    // Variables
    VT * d_vals = csc->val;
    IT * d_rowinds = csc->idx_vec;
    IT * d_colptrs = csc->ptr_vec;

    auto nnz = csc->nnz;
    auto nrows = csc->nrows;
    auto ncols = csc->ncols;


    // CSX result
    CSX<IT, VT> * csr = new CSX<IT, VT>;
    csr->majordim = MajorDim::ROWS;
    csr->nnz = nnz;
    csr->nrows = nrows;
    csr->ncols = ncols;


    // CSR pointers
    VT * d_csr_vals;
    IT * d_colinds, * d_rowptrs;

    if (buffers == nullptr) 
    {
        CUDA_CHECK(cudaMalloc(&d_csr_vals, sizeof(VT) * nnz));
        CUDA_CHECK(cudaMalloc(&d_colinds, sizeof(IT) * nnz));
        CUDA_CHECK(cudaMalloc(&d_rowptrs, sizeof(IT) * (nrows + 1)));
    } 
    else 
    {
        buffers->ensure_async(nnz, nrows + 1, ncols);
        CUDA_SYNC(*stream);
        d_csr_vals = buffers->d_node_vals;
        d_colinds  = buffers->d_node_colinds;
        d_rowptrs  = buffers->d_node_rowptrs;
    }

    csr->val = d_csr_vals;
    csr->ptr_vec = d_rowptrs;
    csr->idx_vec = d_colinds;

    // First, use cusparse to convert the csc pointers into raw CSR pointers
    // cusparse does not have a csc->csr, it only has csr->csc
    // we can trick it into doing csc->csr by pretending our csc
    // is a transposed csr matrix
    size_t buff_size = 0;
    void * d_buff = nullptr;
    CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(*handle,
                                                ncols, nrows,
                                                nnz,
                                                d_vals,
                                                d_colptrs,
                                                d_rowinds,
                                                d_csr_vals,
                                                d_rowptrs,
                                                d_colinds,
                                                CUDA_R_32F,
                                                CUSPARSE_ACTION_NUMERIC,
                                                CUSPARSE_INDEX_BASE_ZERO,
                                                CUSPARSE_CSR2CSC_ALG_DEFAULT,
                                                &buff_size));
    CUDA_SYNC(*stream);

    if (buffers == nullptr) 
    {
        CUDA_CHECK(cudaMalloc(&d_buff, buff_size));
    } 
    else 
    {
        buffers->ensure_tmp_async(buff_size);
        CUDA_SYNC(*stream);
        d_buff = buffers->tmp_buffers[0].tmp_buffer;
    }

    CUSPARSE_CHECK(cusparseCsr2cscEx2(*handle,
                                    ncols, nrows,
                                    nnz,
                                    d_vals,
                                    d_colptrs,
                                    d_rowinds,
                                    d_csr_vals,
                                    d_rowptrs,
                                    d_colinds,
                                    CUDA_R_32F,
                                    CUSPARSE_ACTION_NUMERIC,
                                    CUSPARSE_INDEX_BASE_ZERO,
                                    CUSPARSE_CSR2CSC_ALG_DEFAULT,
                                    d_buff));
    CUDA_SYNC(*stream);

    if (buffers == nullptr) { CUDA_FREE_SAFE(d_buff); }

    return csr;
}



template <typename IT, typename VT>
int cusparse_spmma(cusparseHandle_t* handle,
                   CusparseCSX<IT, VT>* A, 
                   CusparseCSX<IT, VT>* B, 
                   CusparseCSX<IT, VT>** C_prod,
                   CusparseCSX<IT, VT>** C_accum,
                   CusparseCSX<IT, VT>** C_local,
                   bool memefficient=true)
{

#ifdef NVTX_PROFILING
    NVTX_PUSH_RANGE("cuspMult",0);
#endif

    // C_prod = AB
    // TODO: heuristic for switching between these
    int did;
    if (memefficient)
    {
        did = cusparse_spgemm_mem(handle, A, B, *C_prod);
    }
    else
    {
        did = cusparse_spgemm_reuse(handle, A, B, *C_prod);
    }

#ifdef NVTX_PROFILING
    NVTX_POP_RANGE;
#endif


    if (did == 0)
    {
        return 0;
    }


#ifdef NVTX_PROFILING
    NVTX_PUSH_RANGE("cuspAggr",0);
#endif

    // C_accum = C_local + C_prod
    cusparse_spgeam(handle, *C_prod, *C_local, *C_accum);

#ifdef NVTX_PROFILING
    NVTX_POP_RANGE;
#endif

    // C_local now points to underlying C_accum storage
    std::swap(*C_local, *C_accum);

    return 1;
}


template <typename IT, typename VT>
void cusparse_spgeam(cusparseHandle_t * handle,
                     CusparseCSX<IT, VT> * C_prod,
                     CusparseCSX<IT, VT> * C_local,
                     CusparseCSX<IT, VT> * C_accum)
{

    C_prod->assert_buffs("C_prod");
    C_local->assert_buffs("C_local");
    C_accum->assert_buffs("C_accum");

    CsxBuffers<IT, VT> * C_prod_buffs = C_prod->buffers;
    CsxBuffers<IT, VT> * C_local_buffs = C_local->buffers;
    CsxBuffers<IT, VT> * C_accum_buffs = C_accum->buffers;

    cudaStream_t * stream = C_prod_buffs->stream;
    assert(stream != nullptr);

    //par_print("C_prod_nnz: %lu, C_accum_nnz: %lu, C_local_nnz: %lu\n",
    //            C_prod_buffs->nnz, C_accum_buffs->nnz, C_local_buffs->nnz);

    int m = C_prod->nrows();
    int n = C_prod->ncols();

    float alpha = 1.0;
    float beta = 1.0;

    size_t buf_size = 0;

    CUSPARSE_CHECK( cusparseScsrgeam2_bufferSizeExt(*handle,
                                                    m, n, 
                                                    &alpha,
                                                    C_prod->mat_descr,
                                                    C_prod_buffs->nnz,
                                                    C_prod_buffs->d_node_vals,
                                                    (const int*) C_prod_buffs->d_node_rowptrs,
                                                    (const int*) C_prod_buffs->d_node_colinds,
                                                    &beta,
                                                    C_local->mat_descr,
                                                    C_local_buffs->nnz,
                                                    C_local_buffs->d_node_vals,
                                                    (const int*) C_local_buffs->d_node_rowptrs,
                                                    (const int*) C_local_buffs->d_node_colinds,
                                                    C_accum->mat_descr,
                                                    C_accum_buffs->d_node_vals,
                                                    (const int*) C_accum_buffs->d_node_rowptrs,
                                                    (const int*) C_accum_buffs->d_node_colinds,
                                                    &buf_size) );
    CUDA_SYNC(*stream);

    C_accum_buffs->ensure_tmp_async(buf_size);
    CUDA_SYNC(*stream);

    int nnz_accum = 0;
    CUSPARSE_CHECK( cusparseXcsrgeam2Nnz(*handle,
                                         m, n, 
                                         C_prod->mat_descr,
                                         C_prod_buffs->nnz,
                                         C_prod_buffs->d_node_rowptrs,
                                         C_prod_buffs->d_node_colinds,
                                         C_local->mat_descr,
                                         C_local_buffs->nnz,
                                         C_local_buffs->d_node_rowptrs,
                                         C_local_buffs->d_node_colinds,
                                         C_accum->mat_descr,
                                         C_accum_buffs->d_node_rowptrs,
                                         &nnz_accum, 
                                         C_accum_buffs->tmp_buffers[0].tmp_buffer) );
    CUDA_SYNC(*stream);

    C_accum_buffs->ensure_async(nnz_accum, m+1, n);
    CUDA_SYNC(*stream);

    CUSPARSE_CHECK( cusparseScsrgeam2(*handle,
                                      m, n,
                                      &alpha,
                                      C_prod->mat_descr,
                                      C_prod_buffs->nnz,
                                      C_prod_buffs->d_node_vals,
                                      C_prod_buffs->d_node_rowptrs,
                                      C_prod_buffs->d_node_colinds,
                                      &beta,
                                      C_local->mat_descr,
                                      C_local_buffs->nnz,
                                      C_local_buffs->d_node_vals,
                                      C_local_buffs->d_node_rowptrs,
                                      C_local_buffs->d_node_colinds,
                                      C_accum->mat_descr,
                                      C_accum_buffs->d_node_vals,
                                      C_accum_buffs->d_node_rowptrs,
                                      C_accum_buffs->d_node_colinds,
                                      C_accum_buffs->tmp_buffers[0].tmp_buffer) );
    CUDA_SYNC(*stream);

    //par_print("Post accumulation: C_prod_nnz: %lu, C_accum_nnz: %lu, C_local_nnz: %lu\n",
    //            C_prod_buffs->nnz, C_accum_buffs->nnz, C_local_buffs->nnz);
}


template <typename IT, typename  VT>
void cusparse_spgemm(cusparseHandle_t& handle,
                     CusparseCSX<IT, VT>* A, 
                     CusparseCSX<IT, VT>* B, 
                     CusparseCSX<IT, VT>* C)
{

    // Check to make sure underlying storage is ok
    A->assert_mat("A");
    B->assert_mat("B");
    C->assert_buffs("C");


    A->validate_csr();
    B->validate_csr();


    CsxBuffers<IT, VT> * buffers = C->buffers;

    assert(buffers->nbufs >= 2 && "Need at least 2 temporary buffers");

    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_DEFAULT;

    float alpha = 1.0;
    float beta = 1.0;

    cusparseSpGEMMDescr_t descr;
    CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&descr));

    size_t buf_size1;

    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle,
                                                 op, op, 
                                                 &alpha, 
                                                 A->descr,
                                                 B->descr,
                                                 &beta,
                                                 C->descr,
                                                 CUDA_R_32F,
                                                 alg,
                                                 descr, &buf_size1,
                                                 NULL));
    buffers->ensure_tmp(buf_size1); 
    void * d_buf1 = buffers->tmp_buffers[0].tmp_buffer;


    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(handle,
                                                 op, op, 
                                                 &alpha, 
                                                 A->descr,
                                                 B->descr,
                                                 &beta,
                                                 C->descr,
                                                 CUDA_R_32F,
                                                 alg,
                                                 descr, &buf_size1,
                                                 d_buf1));

    size_t buf_size2;
    CUSPARSE_CHECK(cusparseSpGEMM_compute(handle,
                                          op, op,
                                          &alpha,
                                          A->descr,
                                          B->descr,
                                          &beta,
                                          C->descr,
                                          CUDA_R_32F,
                                          alg,
                                          descr, &buf_size2,
                                          NULL));


    int64_t num_prods = 0;
    CUSPARSE_CHECK(cusparseSpGEMM_getNumProducts(descr, &num_prods));

    print_gpu_mem();
    par_print("FLOPS(C) = %lu\nBuffer Size 2 = %zu\n", num_prods,buf_size2);

    buffers->ensure_tmp(buf_size2, 1);
    void * d_buf2 = buffers->tmp_buffers[1].tmp_buffer;

    CUSPARSE_CHECK(cusparseSpGEMM_compute(handle,
                                          op, op,
                                          &alpha,
                                          A->descr,
                                          B->descr,
                                          &beta,
                                          C->descr,
                                          CUDA_R_32F,
                                          alg,
                                          descr, &buf_size2,
                                          d_buf2));

    par_print("done compute");

    int64_t Cnrows, Cncols, Cnnz;
    CUSPARSE_CHECK(cusparseSpMatGetSize(C->descr, &Cnrows, &Cncols, &Cnnz));

    buffers->ensure(Cnnz, Cnrows+1, Cncols);
    print_gpu_mem();

    CUSPARSE_CHECK(cusparseCsrSetPointers(C->descr, buffers->d_node_rowptrs, buffers->d_node_colinds, buffers->d_node_vals));

    CUSPARSE_CHECK(cusparseSpGEMM_copy(handle, op, op, &alpha,
                                       A->descr, B->descr, &beta,
                                       C->descr,
                                       CUDA_R_32F, alg, descr));

    CUSPARSE_CHECK(cusparseSpGEMM_destroyDescr(descr));

}


//TODO: Sync only on the cusparse stream
template <typename IT, typename VT>
int cusparse_spgemm_reuse(cusparseHandle_t * handle,
                          CusparseCSX<IT, VT>* A,
                          CusparseCSX<IT, VT>* B,
                          CusparseCSX<IT, VT>* C)
{
    // Check to make sure underlying storage is ok
    A->assert_mat("A");
    B->assert_mat("B");
    C->assert_buffs("C");

    CsxBuffers<IT, VT> * buffers = C->buffers;
    cudaStream_t * stream = buffers->stream;
    assert(stream != nullptr);

    assert(buffers->nbufs >= 5 && "Need at least 5 temporary buffers for SpGEMMreuse");

    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_DEFAULT;

    float alpha = 1.0;
    float beta = 0.0;

    cusparseSpGEMMDescr_t descr;
    CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&descr));
    CUDA_SYNC(*stream);

    // === Phase 1: Work Estimation ===
    size_t buf_size1;
    CUSPARSE_CHECK(cusparseSpGEMMreuse_workEstimation(*handle,
                                                      op, op,
                                                      A->descr,
                                                      B->descr,
                                                      C->descr,
                                                      alg,
                                                      descr,
                                                      &buf_size1,
                                                      NULL));
    CUDA_SYNC(*stream);

    buffers->ensure_tmp_async(buf_size1);
    CUDA_SYNC(*stream);
    void * d_buf1 = buffers->tmp_buffers[0].tmp_buffer;

    CUSPARSE_CHECK(cusparseSpGEMMreuse_workEstimation(*handle,
                                                      op, op,
                                                      A->descr,
                                                      B->descr,
                                                      C->descr,
                                                      alg,
                                                      descr,
                                                      &buf_size1,
                                                      d_buf1));
    CUDA_SYNC(*stream);

    int64_t num_prods = 0;
    CUSPARSE_CHECK(cusparseSpGEMM_getNumProducts(descr, &num_prods));
    CUDA_SYNC(*stream);

    if (num_prods == 0)
    {
        return 0;
    }

    // === Phase 2: Determine NNZ and sparsity structure ===
    size_t buf_size2, buf_size3, buf_size4;
    CUSPARSE_CHECK(cusparseSpGEMMreuse_nnz(*handle,
                                           op, op,
                                           A->descr,
                                           B->descr,
                                           C->descr,
                                           alg,
                                           descr,
                                           &buf_size2, NULL,
                                           &buf_size3, NULL,
                                           &buf_size4, NULL));
    CUDA_SYNC(*stream);

    buffers->ensure_tmp_async(buf_size2,  1);
    CUDA_SYNC(*stream);
    buffers->ensure_tmp_async(buf_size3,  2);
    CUDA_SYNC(*stream);
    buffers->ensure_tmp_async(buf_size4,  3);
    CUDA_SYNC(*stream);


    void * d_buf2 = buffers->tmp_buffers[1].tmp_buffer;
    void * d_buf3 = buffers->tmp_buffers[2].tmp_buffer;
    void * d_buf4 = buffers->tmp_buffers[3].tmp_buffer;

    CUSPARSE_CHECK(cusparseSpGEMMreuse_nnz(*handle,
                                           op, op,
                                           A->descr,
                                           B->descr,
                                           C->descr,
                                           alg,
                                           descr,
                                           &buf_size2, d_buf2,
                                           &buf_size3, d_buf3,
                                           &buf_size4, d_buf4));
    CUDA_SYNC(*stream);

    buffers->free_tmp_async(0);
    buffers->free_tmp_async(1);

    CUDA_SYNC(*stream);

    // Get result matrix dimensions and allocate output
    int64_t Cnrows, Cncols, Cnnz;
    CUSPARSE_CHECK(cusparseSpMatGetSize(C->descr, &Cnrows, &Cncols, &Cnnz));

    buffers->ensure_async(Cnnz, Cnrows+1, Cncols);
    CUDA_SYNC(*stream);

    CUSPARSE_CHECK(cusparseCsrSetPointers(C->descr, buffers->d_node_rowptrs,
                                          buffers->d_node_colinds, buffers->d_node_vals));

    // === Phase 3: Copy (prepare internal structures) ===
    size_t buf_size5;
    CUSPARSE_CHECK(cusparseSpGEMMreuse_copy(*handle,
                                            op, op,
                                            A->descr,
                                            B->descr,
                                            C->descr,
                                            alg,
                                            descr,
                                            &buf_size5, NULL));
    CUDA_SYNC(*stream);


    buffers->ensure_tmp_async(buf_size5,  4);
    CUDA_SYNC(*stream);
    void * d_buf5 = buffers->tmp_buffers[4].tmp_buffer;

    CUSPARSE_CHECK(cusparseSpGEMMreuse_copy(*handle,
                                            op, op,
                                            A->descr,
                                            B->descr,
                                            C->descr,
                                            alg,
                                            descr,
                                            &buf_size5, d_buf5));
    CUDA_SYNC(*stream);

    // === Phase 4: Compute ===
    CUSPARSE_CHECK(cusparseSpGEMMreuse_compute(*handle,
                                               op, op,
                                               &alpha,
                                               A->descr,
                                               B->descr,
                                               &beta,
                                               C->descr,
                                               CUDA_R_32F,
                                               alg,
                                               descr));
    CUDA_SYNC(*stream);

    CUSPARSE_CHECK(cusparseSpGEMM_destroyDescr(descr));
    CUDA_SYNC(*stream);

    buffers->free_tmp_async(2);
    buffers->free_tmp_async(3);
    buffers->free_tmp_async(4);
    CUDA_SYNC(*stream);

    return 1;
}


template <typename IT, typename VT>
int cusparse_spgemm_mem(cusparseHandle_t * handle,
                        CusparseCSX<IT, VT>* A,
                        CusparseCSX<IT, VT>* B,
                        CusparseCSX<IT, VT>* C,
                        float chunk_fraction = 0.2f)
{
    // Check to make sure underlying storage is ok
    A->assert_mat("A");
    B->assert_mat("B");
    C->assert_buffs("C");

    CsxBuffers<IT, VT> * buffers = C->buffers;
    cudaStream_t * stream = buffers->stream;

    assert(stream != nullptr);
    assert(buffers->nbufs >= 3 && "Need at least 3 temporary buffers for memory-efficient SpGEMM");

    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_ALG3;

    float alpha = 1.0;
    float beta = 0.0;

    cusparseSpGEMMDescr_t descr;
    CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&descr));

    // === Phase 1: Work Estimation ===
    size_t buf_size1;
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(*handle,
                                                 op, op,
                                                 &alpha,
                                                 A->descr,
                                                 B->descr,
                                                 &beta,
                                                 C->descr,
                                                 CUDA_R_32F,
                                                 alg,
                                                 descr, &buf_size1,
                                                 NULL));
    CUDA_SYNC(*stream);

    buffers->ensure_tmp_async(buf_size1);
    CUDA_SYNC(*stream);
    void * d_buf1 = buffers->tmp_buffers[0].tmp_buffer;

    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(*handle,
                                                 op, op,
                                                 &alpha,
                                                 A->descr,
                                                 B->descr,
                                                 &beta,
                                                 C->descr,
                                                 CUDA_R_32F,
                                                 alg,
                                                 descr, &buf_size1,
                                                 d_buf1));
    CUDA_SYNC(*stream);

    int64_t num_prods = 0;
    CUSPARSE_CHECK(cusparseSpGEMM_getNumProducts(descr, &num_prods));

    if (num_prods == 0)
    {
        return 0;
    }

    // === Phase 2: Memory Estimation (memory-efficient approach) ===
    size_t buf_size2, buf_size3;

    // First call to determine buffer size for memory estimation
    CUSPARSE_CHECK(cusparseSpGEMM_estimateMemory(*handle,
                                                 op, op,
                                                 &alpha,
                                                 A->descr,
                                                 B->descr,
                                                 &beta,
                                                 C->descr,
                                                 CUDA_R_32F,
                                                 alg,
                                                 descr,
                                                 chunk_fraction,
                                                 &buf_size3,
                                                 NULL,
                                                 NULL));
    CUDA_SYNC(*stream);

    buffers->ensure_tmp_async(buf_size3, 2);
    CUDA_SYNC(*stream);
    void * d_buf3 = buffers->tmp_buffers[2].tmp_buffer;

    // Second call to perform memory estimation and get compute buffer size
    CUSPARSE_CHECK(cusparseSpGEMM_estimateMemory(*handle,
                                                 op, op,
                                                 &alpha,
                                                 A->descr,
                                                 B->descr,
                                                 &beta,
                                                 C->descr,
                                                 CUDA_R_32F,
                                                 alg,
                                                 descr,
                                                 chunk_fraction,
                                                 &buf_size3,
                                                 d_buf3,
                                                 &buf_size2));
    CUDA_SYNC(*stream);
    buffers->free_tmp_async(2);
    CUDA_SYNC(*stream);

    // Allocate compute buffer
    buffers->ensure_tmp_async(buf_size2, 1);
    CUDA_SYNC(*stream);
    void * d_buf2 = buffers->tmp_buffers[1].tmp_buffer;

    // === Phase 3: Compute ===
    CUSPARSE_CHECK(cusparseSpGEMM_compute(*handle,
                                          op, op,
                                          &alpha,
                                          A->descr,
                                          B->descr,
                                          &beta,
                                          C->descr,
                                          CUDA_R_32F,
                                          alg,
                                          descr, &buf_size2,
                                          d_buf2));
    CUDA_SYNC(*stream);

    // === Phase 4: Get result size and allocate output ===
    int64_t Cnrows, Cncols, Cnnz;
    CUSPARSE_CHECK(cusparseSpMatGetSize(C->descr, &Cnrows, &Cncols, &Cnnz));

    buffers->ensure_async(Cnnz, Cnrows+1, Cncols);
    CUDA_SYNC(*stream);

    CUSPARSE_CHECK(cusparseCsrSetPointers(C->descr, buffers->d_node_rowptrs,
                                          buffers->d_node_colinds, buffers->d_node_vals));

    // === Phase 5: Copy results ===
    CUSPARSE_CHECK(cusparseSpGEMM_copy(*handle, op, op, &alpha,
                                       A->descr, B->descr, &beta,
                                       C->descr,
                                       CUDA_R_32F, alg, descr));
    CUDA_SYNC(*stream);
    buffers->free_tmp_async(1);
    buffers->free_tmp_async(3);
    CUDA_SYNC(*stream);

    CUSPARSE_CHECK(cusparseSpGEMM_destroyDescr(descr));

    return 1;
}


