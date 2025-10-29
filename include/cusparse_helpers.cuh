#pragma once

#include "common.h"
#include "utils.cuh"



using namespace mmio;


template <typename IT, VT>
struct CusparseCSX
{
    cusparseSpMatDescr_t descr;
    CSX<IT, VT> * mat;
    CsxBuffers<IT, VT> * buffers;
    State state;

    CusparseCSX():
        mat(nullptr), buffers(nullptr), state(State::Null)
    {}

    CusparseCSX(CSX<IT, VT> * mat): 
        mat(mat), buffers(nullptr), state(State::Mat)
    {
        assert(mat->majordim == MajorDim::ROWS);
        CHECK_CUSPARSE(cusparseCreateCsr(&descr,
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
    }


    CusparseCSX(CsxBuffers<IT, VT> * buffers):
        mat(nullptr), buffers(buffers), state(State::Buffs)
    {
        CHECK_CUSPARSE(cusparseCreateCsr(&descr,
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
    }


    enum class State
    {
        Mat,
        Buffs, 
        Null
    };


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

        CHECK_CUSPARSE(cusparseDestroySpMat(descr));
    }
};


template <typename IT, VT>
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
                                                    C_prod->descr,
                                                    C_prod_buffs->nnz,
                                                    C_prod_buffs->d_node_vals,
                                                    C_prod_buffs->d_node_rowptrs,
                                                    C_prod_buffs->d_node_colinds,
                                                    &beta,
                                                    C_local_buffs->descr,
                                                    C_local_buffs->nnz,
                                                    C_local_buffs->d_node_vals,
                                                    C_local_buffs->d_node_rowptrs,
                                                    C_local_buffs->d_node_colinds,
                                                    C_accum_buffs->descr,
                                                    C_accum_buffs->nnz,
                                                    C_accum_buffs->d_node_vals,
                                                    C_accum_buffs->d_node_rowptrs,
                                                    C_accum_buffs->d_node_colinds,
                                                    &buf_size) );

    C_accum_buffs->ensure_tmp(buf_size);

    size_t nnz_accum = 0;

    CUSPARSE_CHECK( cusparseXcsrgeam2Nnz(handle,
                                         m, n, 
                                         C_prod->descr,
                                         C_prod_buffs->nnz,
                                         C_prod_buffs->d_node_rowptrs,
                                         C_prod_buffs->d_node_colinds,
                                         C_local_buffs->descr,
                                         C_local_buffs->nnz,
                                         C_local_buffs->d_node_rowptrs,
                                         C_local_buffs->d_node_colinds,
                                         C_accum_buffs->descr,
                                         C_accum_buffs->d_node_rowptrs,
                                         &nnz, 
                                         C_accum_buffs->tmp_buffers[0]) );

    C_accum_buffs->ensure(nnz_accum, m+1);

    CUSPARSE_CHECK( cusparseScsrgeam2(handle,
                                      m, n,
                                      &alpha,
                                      C_prod->descr,
                                      C_prod_buffs->nnz,
                                      C_prod_buffs->d_node_vals,
                                      C_prod_buffs->d_node_rowptrs,
                                      C_prod_buffs->d_node_colinds,
                                      &beta,
                                      C_local_buffs->descr,
                                      C_local_buffs->nnz,
                                      C_local_buffs->d_node_vals,
                                      C_local_buffs->d_node_rowptrs,
                                      C_local_buffs->d_node_colinds,
                                      C_accum_buffs->descr,
                                      C_accum_buffs->nnz,
                                      C_accum_buffs->d_node_vals,
                                      C_accum_buffs->d_node_rowptrs,
                                      C_accum_buffs->d_node_colinds,
                                      C_accum_buffs->tmp_buffers[0]) );
}


template <typename IT, VT>
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


template <typename IT, VT>
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
    void * d_buf1 = buffers->tmp_buffers[0];

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
    void * d_buf2 = buffers->tmp_buffers[1];
    CHECK_CUSPARSE(cusparseSpGEMM_Compute(handle,
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

    CHECK_CUSPARSE(cusparseSpGEMM_Compute(handle,
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
    CHECK_CUSPARSE(cusparseSpMatGetSize(C->descr, &Cnrows, &Cncols, &Cnnz));

    buffers->ensure(Cnnz, Cnrows+1);

    CHECK_CUSPARSE(cusparseCsrSetPointers(C->descr, buffers->d_node_rowptrs, buffers->d_node_colinds, buffers->d_node_vals));

    CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, op, op, &alpha,
                                       A->descr, B->descr, &beta,
                                       C->descr,
                                       CUDA_R_32F, alg, descr));

    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(descr));

}
                     

