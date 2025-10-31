#pragma once

#include "common.h"
#include "utils.cuh"
#include "kokkos_helpers.cuh"

#include <dmmio/dmmio.h>
#include <dmmio/partitioning.h>



using namespace mmio;
using namespace dmmio;

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


    CusparseCSX(cusparseHandle_t& handle, 
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

        if (is_mat() && mat->majordim==MajorDim::ROWS)
        {
            CUSPARSE_CHECK(cusparseDestroySpMat(descr));
            CUSPARSE_CHECK(cusparseDestroyMatDescr(mat_descr));
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

        while (dcoo->coo->nrows % (dcoo->partitioning->grid->col_size * dcoo->partitioning->grid->node_size * MASK_SIZE) != 0)
        {
            dcoo->coo->nrows++;
        }

        while (dcoo->coo->ncols % (dcoo->partitioning->grid->row_size * MASK_SIZE) != 0)
        {
            dcoo->coo->ncols++;
        }

        IT max_dim = max(dcoo->coo->ncols, dcoo->coo->nrows);

        dcoo->coo->ncols = max_dim;
        dcoo->coo->nrows = max_dim;
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
            mat = coo_to_row_csx(coo);
        } 
        else 
        {
            mat = coo_to_col_csx(coo);
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


    void explicit_free()
    {
        csx->explicit_free();
    }
    
};


template <typename IT, typename VT>
CSX<IT, VT> * cusparse_csc_to_csr(cusparseHandle_t& handle, CSX<IT, VT> * csc, CsxBuffers<IT, VT> * buffers)
{
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
        buffers->ensure(nnz, nrows + 1, ncols);
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
    CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(handle,
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
    if (buffers == nullptr) 
    {
        CUDA_CHECK(cudaMalloc(&d_buff, buff_size));
    } 
    else 
    {
        buffers->ensure_tmp(buff_size);
        d_buff = buffers->tmp_buffers[0].tmp_buffer;
    }

    CUSPARSE_CHECK(cusparseCsr2cscEx2(handle,
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

    if (buffers == nullptr) { CUDA_FREE_SAFE(d_buff); }

    return csr;
}



template <typename IT, typename VT>
void cusparse_spmma(cusparseHandle_t& handle,
                     CusparseCSX<IT, VT>* A, 
                     CusparseCSX<IT, VT>* B, 
                     CusparseCSX<IT, VT>* C_prod,
                     CusparseCSX<IT, VT>* C_accum,
                     CusparseCSX<IT, VT>* C_local,
                     const bool accum)
{
    // C_prod = AB
    cusparse_spgemm(handle, A, B, C_prod);


    // Do not accumulate, C_local will point to underlying C_prod
    if (!accum)
    {
        std::swap(C_local, C_prod);
        return;
    }


    // C_accum = C_local + C_prod
    cusparse_spgeam(handle, C_prod, C_local, C_accum);

    // C_local now points to underlying C_accum storage
    std::swap(C_local, C_accum);

}


template <typename IT, typename VT>
void cusparse_spgeam(cusparseHandle_t& handle,
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

    int m = C_prod->nrows();
    int n = C_prod->ncols();

    float alpha = 1.0;
    float beta = 1.0;

    size_t buf_size = 0;

    CUSPARSE_CHECK( cusparseScsrgeam2_bufferSizeExt(handle,
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

    C_accum_buffs->ensure_tmp(buf_size);

    int nnz_accum = 0;
    CUSPARSE_CHECK( cusparseXcsrgeam2Nnz(handle,
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

    C_accum_buffs->ensure(nnz_accum, m+1, n);

    CUSPARSE_CHECK( cusparseScsrgeam2(handle,
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
    buffers->ensure_tmp(buf_size1, 0); 
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

    int64_t Cnrows, Cncols, Cnnz;
    CUSPARSE_CHECK(cusparseSpMatGetSize(C->descr, &Cnrows, &Cncols, &Cnnz));

    buffers->ensure(Cnnz, Cnrows+1, Cncols);

    CUSPARSE_CHECK(cusparseCsrSetPointers(C->descr, buffers->d_node_rowptrs, buffers->d_node_colinds, buffers->d_node_vals));

    CUSPARSE_CHECK(cusparseSpGEMM_copy(handle, op, op, &alpha,
                                       A->descr, B->descr, &beta,
                                       C->descr,
                                       CUDA_R_32F, alg, descr));

    CUSPARSE_CHECK(cusparseSpGEMM_destroyDescr(descr));

}
                     

