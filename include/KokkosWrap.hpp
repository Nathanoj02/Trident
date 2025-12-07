#pragma once

// KokkosWrap.hpp
#include <dmmio/dmmio.h>
#include <dmmio/partitioning.h>

#include <variant>
#include <cstdint>
#include <memory>

#include <cusparse_v2.h>


#include "KokkosWrapTriple.hpp"
#include "kokkos_helpers.cuh"


using MajorDim = mmio::MajorDim;

namespace KokkosWrap {

    template <typename KIT, typename DIT, typename VT>
    struct DistribuitedMatrix {
        dmmio::Partitioning* partitioning;  // still raw pointer to external partitioning
        // Instead of a raw union, use std::variant for safety
        // std::variant<
        //     KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT>,
        //     KokkosSparse::CcsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT>
        // > storage;

        mmio::CSX<KIT, VT>* mmio_csx;

        // ---- Constructor ----
        DistribuitedMatrix(dmmio::DCOO<DIT, VT>* dcoo, MajorDim T);

        // ------ Methods ------
        KIT getLocalNnz();
        KIT getLocalNcols();
        KIT getLocalNrows();
        KIT getLocalPtrvecsize();
    };

    template <typename KIT, typename DIT, typename VT>
    struct LocalMatrix {

        using KokkosCrs = KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT>;
        using KokkosCcs = KokkosSparse::CcsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT>;

        bool initialized;
        KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> storage;

        // ---- Constructor ----
        LocalMatrix();
        LocalMatrix(cusparseHandle_t& handle, mmio::CSX<DIT, VT>* mmio_csx, CsxBuffers<DIT,VT> *buffs=nullptr);
        LocalMatrix(mmio::CSR<DIT, VT>* mmio_csr);
        LocalMatrix(mmio::CSX<DIT, VT>* mmio_csx);
        LocalMatrix(DIT nrows, DIT ncols, DIT nnz, DIT* ptrvec, DIT* idxvec, VT* values, MajorDim layout);

        // In-place SpGEMM: C = C + A*B
        static void sp_mma(const LocalMatrix& A, const LocalMatrix& B, LocalMatrix& C);

        // In-place SpGEMM: C = A*B
        static LocalMatrix spgemm(const LocalMatrix& A, const LocalMatrix& B);

        // In-place Spadd
        static void spadd(const LocalMatrix& A, LocalMatrix& C);

        // Conversion
        KokkosCrs csc_to_csr(cusparseHandle_t& handle, mmio::CSX<DIT, VT> * mmio_csx, CsxBuffers<DIT,VT> *buffs=nullptr);

        // Change access structure
        mmio::CSX<DIT,VT>* get_csx(void);

        // Free memory & decostructor
        void freeBuffers();
        ~LocalMatrix();
    };


    //template <typename IT, typename VT>
    //struct Triple
    //{
    //    IT row;
    //    IT col;
    //    VT val;
    //};


    template<typename KIT, typename VT>
    mmio::CSX<KIT, VT>* rawptr_get(KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT>& kokkos_csr) {
        auto val_view     = kokkos_csr.values;
        auto rowmap_view  = kokkos_csr.graph.row_map;
        auto entries_view = kokkos_csr.graph.entries;

        mmio::CSX<KIT, VT> *csx = (mmio::CSX<KIT, VT>*)malloc(sizeof(mmio::CSX<KIT, VT>));
        csx->majordim = MajorDim::ROWS;
        csx->nnz      = kokkos_csr.nnz();
        csx->nrows    = kokkos_csr.numRows();
        csx->ncols    = kokkos_csr.numCols();
        csx->val      = kokkos_csr.values.data();
        csx->ptr_vec  = const_cast<int32_t*>(rowmap_view.data());
        csx->idx_vec  = const_cast<int32_t*>(entries_view.data());

        return(csx);
    }

    template<typename KIT, typename VT>
    mmio::CSX<KIT, VT>* rawptr_get(KokkosSparse::CcsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT>& kokkos_csc) {
        auto val_view     = kokkos_csc.values;
        auto colmap_view  = kokkos_csc.graph.col_map;
        auto entries_view = kokkos_csc.graph.entries;

        mmio::CSX<KIT, VT> *csx = (mmio::CSX<KIT, VT>*)malloc(sizeof(mmio::CSX<KIT, VT>));
        csx->majordim = MajorDim::COLS;
        csx->nnz      = kokkos_csc.nnz();
        csx->nrows    = kokkos_csc.numRows();
        csx->ncols    = kokkos_csc.numCols();
        csx->val      = val_view.data();
        csx->ptr_vec  = const_cast<int32_t*>(colmap_view.data());
        csx->idx_vec  = const_cast<int32_t*>(entries_view.data());

        return(csx);
    }

    template<typename KIT, typename VT>
    mmio::CSX<KIT, VT>* rawptr_get(LocalMatrix<KIT, KIT, VT>& kokkos_csr) {
        return(rawptr_get(kokkos_csr.storage));
    }


    template <typename IT, typename VT>
    mmio::CSX<IT, VT> * csx_gen(IT nrows, IT ncols, IT nnz, IT* ptrvec, IT* idxvec, VT* values, MajorDim layout) {
        using dmmiocsx = typename mmio::CSX<IT, VT>;
        dmmiocsx *csx  = (dmmiocsx*)malloc(sizeof(dmmiocsx));

        csx->majordim  = layout;
        csx->nnz       = nnz;
        csx->nrows     = nrows;
        csx->ncols     = ncols;
        csx->val       = values;
        csx->ptr_vec   = ptrvec;
        csx->idx_vec   = idxvec;

        return(csx);
    }


    template <typename KIT, typename DIT, typename VT>
    DistribuitedMatrix<KIT,DIT,VT>::DistribuitedMatrix(dmmio::DCOO<DIT, VT>* dcoo, MajorDim T)
            : partitioning(dcoo->partitioning)
    {
        static_assert(std::is_same<KIT, DIT>::value);
        using namespace dmmio::partitioning::indextransform;
        while (dcoo->coo->nrows % (dcoo->partitioning->grid->col_size * dcoo->partitioning->grid->node_size * MASK_SIZE) != 0)
        {
            dcoo->coo->nrows++;
        }

        while (dcoo->coo->ncols % (dcoo->partitioning->grid->row_size * MASK_SIZE) != 0)
        {
            dcoo->coo->ncols++;
        }
        KIT max_dim = max(dcoo->coo->ncols, dcoo->coo->nrows);
        dcoo->coo->ncols = max_dim;
        dcoo->coo->nrows = max_dim;
        dcoo->coo->nrows /= (dcoo->partitioning->grid->col_size * dcoo->partitioning->grid->node_size);
        dcoo->coo->ncols /= dcoo->partitioning->grid->row_size;

        for (KIT i=0; i<dcoo->coo->nnz; i++)
        {
            dcoo->coo->row[i] = global2local::row(dcoo->partitioning, dcoo->coo->row[i]);
            dcoo->coo->col[i] = global2local::col(dcoo->partitioning, dcoo->coo->col[i]);
        }

        auto coo = dcoo->coo;

        // --- Step 2: Decide layout ---
        if (T == MajorDim::ROWS) 
        {
            mmio_csx = coo_to_row_csx(coo);
        } 
        else 
        {
            mmio_csx = coo_to_col_csx(coo);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    template <typename KIT, typename DIT, typename VT>
    KIT DistribuitedMatrix<KIT,DIT,VT>::getLocalNnz()   { return(mmio_csx->nnz);   }

    template <typename KIT, typename DIT, typename VT>
    KIT DistribuitedMatrix<KIT,DIT,VT>::getLocalNcols() { return(mmio_csx->ncols); }

    template <typename KIT, typename DIT, typename VT>
    KIT DistribuitedMatrix<KIT,DIT,VT>::getLocalNrows() { return(mmio_csx->nrows); }

    template <typename KIT, typename DIT, typename VT>
    KIT DistribuitedMatrix<KIT,DIT,VT>::getLocalPtrvecsize() {
        if (mmio_csx->majordim == MajorDim::ROWS)
            return(mmio_csx->nrows); // Put +1 here?
        else
            return(mmio_csx->ncols); // Put +1 here?
    }

    template <typename KIT, typename DIT, typename VT>
    LocalMatrix<KIT,DIT,VT>::LocalMatrix() : storage(), initialized(false) {}

    template <typename KIT, typename DIT, typename VT>
    LocalMatrix<KIT,DIT,VT>::LocalMatrix(cusparseHandle_t& handle, mmio::CSX<DIT, VT>* mmio_csx, CsxBuffers<DIT,VT> *buffs)
        :initialized(true)
    {
        static_assert(std::is_same<KIT, DIT>::value);

        using ordinal_view_t = Kokkos::View<KIT*, Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        using values_view_t  = Kokkos::View<VT*,   Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        // TODO check KIT == DIT or find a way to a static cast
        if (mmio_csx->majordim == MajorDim::ROWS) {
            ordinal_view_t rowmap(mmio_csx->ptr_vec, mmio_csx->nrows + 1);
            ordinal_view_t colidx(mmio_csx->idx_vec, mmio_csx->nnz);
            values_view_t  values(mmio_csx->val,     mmio_csx->nnz);

            KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> kokkos_csr("kokkos_csr",
                                            mmio_csx->nrows, mmio_csx->ncols, mmio_csx->nnz,
                                            values,
                                            rowmap,
                                            colidx
            );

            storage = kokkos_csr;
        } else {
            storage = csc_to_csr(handle, mmio_csx, buffs);
        }
    }


    template <typename KIT, typename DIT, typename VT>
    LocalMatrix<KIT,DIT,VT>::LocalMatrix(mmio::CSR<DIT, VT>* mmio_csr) 
        :initialized(true)
    {
        static_assert(std::is_same<KIT, DIT>::value);

        using ordinal_view_t = Kokkos::View<KIT*, Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        using values_view_t  = Kokkos::View<VT*,   Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        // TODO check KIT == DIT or find a way to a static cast
        ordinal_view_t rowmap(mmio_csr->row_ptr, mmio_csr->nrows + 1);
        ordinal_view_t colidx(mmio_csr->col_idx, mmio_csr->nnz);
        values_view_t  values(mmio_csr->val,     mmio_csr->nnz);

        KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> kokkos_csr("kokkos_csr",
                                        mmio_csr->nrows, mmio_csr->ncols, mmio_csr->nnz,
                                        values,
                                        rowmap,
                                        colidx
        );

        storage = kokkos_csr;
    }


    template <typename KIT, typename DIT, typename VT>
    LocalMatrix<KIT,DIT,VT>::LocalMatrix(mmio::CSX<DIT, VT>* mmio_csx) 
        :initialized(true)
    {
        static_assert(std::is_same<KIT, DIT>::value);

        using ordinal_view_t = Kokkos::View<KIT*, Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        using values_view_t  = Kokkos::View<VT*,   Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        // TODO check KIT == DIT or find a way to a static cast
        ordinal_view_t rowmap(mmio_csx->ptr_vec, mmio_csx->nrows + 1);
        ordinal_view_t colidx(mmio_csx->idx_vec, mmio_csx->nnz);
        values_view_t  values(mmio_csx->val,     mmio_csx->nnz);

        KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> kokkos_csr("kokkos_csr",
                                        mmio_csx->nrows, mmio_csx->ncols, mmio_csx->nnz,
                                        values,
                                        rowmap,
                                        colidx
        );

        storage = kokkos_csr;
    }

    template <typename KIT, typename DIT, typename VT>
    mmio::CSX<DIT,VT>* LocalMatrix<KIT,DIT,VT>::get_csx(void) {
        return(rawptr_get(storage));
    }


    template <typename KIT, typename DIT, typename VT>
    void LocalMatrix<KIT,DIT,VT>::freeBuffers() {
        if (initialized) {
            CUDA_FREE_SAFE((void*)storage.graph.row_map.data());
            CUDA_FREE_SAFE((void*)storage.graph.entries.data());
            CUDA_FREE_SAFE((void*)storage.values.data());
            initialized = false;
        }
    }

    template <typename KIT, typename DIT, typename VT>
    LocalMatrix<KIT,DIT,VT>::~LocalMatrix() {
        // Nothing — views are unmanaged, so Kokkos won't free them
        // Assume the caller manages lifetime of ptr_vec, idx_vec, val
    }


    template <typename KIT, typename DIT, typename VT>
    LocalMatrix<KIT,DIT,VT>::LocalMatrix(DIT nrows, DIT ncols, DIT nnz, DIT* ptrvec, DIT* idxvec, VT* values, MajorDim layout)
    : LocalMatrix(csx_gen<DIT, VT>((nrows, ncols, nnz, ptrvec, idxvec, values, layout))) { }

    template <typename KIT, typename DIT, typename VT>
    void LocalMatrix<KIT,DIT,VT>::sp_mma(const LocalMatrix& A, const LocalMatrix& B, LocalMatrix& C) 
    {
        using csr_matrix_type = typename KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT>;

        CUDA_TIMER_DEF(spadd_time);
        CUDA_TIMER_DEF(spm_time);

        // Create KokkosKernelHandle
        using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
            KIT, KIT, VT,
            Kokkos::DefaultExecutionSpace,
            typename Kokkos::DefaultExecutionSpace::memory_space,
            typename Kokkos::DefaultExecutionSpace::memory_space>;

        CUDA_TIMER_START_DEFAULT(spm_time);
        csr_matrix_type product = KokkosSparse::spgemm<csr_matrix_type>(A.storage, false, B.storage, false);
        CUDA_TIMER_STOP(spm_time);

        if (C.initialized == false) 
        {
            C.storage = product;
            C.initialized = true;
        } 
        else 
        {
            csr_matrix_type accumulator;

            CUDA_TIMER_START_DEFAULT(spadd_time);
            KernelHandle kh;
            kh.create_spadd_handle(false);

            KokkosSparse::spadd_symbolic(&kh, product, C.storage, accumulator);
            KokkosSparse::spadd_numeric(&kh, 1.0, product, 1.0, C.storage, accumulator);
            kh.destroy_spadd_handle();

            C.storage = accumulator;
            CUDA_TIMER_STOP(spadd_time);

        }
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        char tmpstr[100];
        sprintf(tmpstr, "[process %d]", rank);
        TIMER_PRINT_WPREFIX_STR(spadd_time, tmpstr)
        TIMER_PRINT_WPREFIX_STR(spm_time, tmpstr)

    }


    template <typename KIT, typename DIT, typename VT>
    LocalMatrix<KIT, DIT, VT> LocalMatrix<KIT,DIT,VT>::spgemm(const LocalMatrix& A, const LocalMatrix& B) 
    {
        CUDA_TIMER_DEF(spm_time);

        LocalMatrix<KIT, DIT, VT> result;

        CUDA_TIMER_START_DEFAULT(spm_time);
        LocalMatrix<KIT, DIT, VT>::KokkosCrs product = KokkosSparse::spgemm<LocalMatrix<KIT,DIT,VT>::KokkosCrs>(A.storage, false, B.storage, false);
        CUDA_TIMER_STOP(spm_time);

        result.storage = product;
        result.initialized = true;


        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        char tmpstr[100];
        sprintf(tmpstr, "[process %d]", rank);
        TIMER_PRINT_WPREFIX_STR(spm_time, tmpstr)

        return result;
    }


    template <typename KIT, typename DIT, typename VT>
    void LocalMatrix<KIT,DIT,VT>::spadd(const LocalMatrix& A, LocalMatrix& C) 
    {
        CUDA_TIMER_DEF(spadd_time);
        CUDA_TIMER_START_DEFAULT(spadd_time)

        if (A.storage.nnz() == 0)
        {
        }
        else if (!C.initialized)
        {
            C.storage = A.storage;
            C.initialized = true;
        }
        else
        {
            // Create KokkosKernelHandle
            using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
                KIT, KIT, VT,
                Kokkos::DefaultExecutionSpace,
                typename Kokkos::DefaultExecutionSpace::memory_space,
                typename Kokkos::DefaultExecutionSpace::memory_space>;


            LocalMatrix<KIT, DIT, VT>::KokkosCrs accumulator;

            KernelHandle kh;
            kh.create_spadd_handle(false);

            KokkosSparse::spadd_symbolic(&kh, A.storage, C.storage, accumulator);
            KokkosSparse::spadd_numeric(&kh, 1.0, A.storage, 1.0, C.storage, accumulator);
            kh.destroy_spadd_handle();

            C.storage = accumulator;
            C.initialized = true;
        }

        CUDA_TIMER_STOP(spadd_time);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        char tmpstr[100];
        sprintf(tmpstr, "[process %d]", rank);
        TIMER_PRINT_WPREFIX_STR(spadd_time, tmpstr)
    }

    // Convert Kokkos CSC matrix to CRS matrix
    template <typename KIT, typename DIT, typename VT>
    LocalMatrix<KIT, DIT, VT>::KokkosCrs LocalMatrix<KIT, DIT, VT>::csc_to_csr(cusparseHandle_t& handle, mmio::CSX<DIT, VT> * csc, CsxBuffers<DIT,VT> *buffers)
    {
        // Type aliases
        using ordinal_view_t = Kokkos::View<KIT*, Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        using values_view_t  = Kokkos::View<VT*,   Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;


        // Variables
        VT * d_vals = csc->val;
        KIT * d_rowinds = csc->idx_vec;
        KIT * d_colptrs = csc->ptr_vec;

        auto nnz = csc->nnz;
        auto nrows = csc->nrows;
        auto ncols = csc->ncols;


        // CSR pointers
        VT * d_csr_vals;
        KIT * d_colinds, * d_rowptrs;

        if (buffers == nullptr) {
            CUDA_CHECK(cudaMalloc(&d_csr_vals, sizeof(VT) * nnz));
            CUDA_CHECK(cudaMalloc(&d_colinds, sizeof(KIT) * nnz));
            CUDA_CHECK(cudaMalloc(&d_rowptrs, sizeof(KIT) * (nrows + 1)));
        } else {
            buffers->ensure(nnz, nrows + 1, ncols);
            d_csr_vals = buffers->d_node_vals;
            d_colinds  = buffers->d_node_colinds;
            d_rowptrs  = buffers->d_node_rowptrs;
        }


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
        if (buffers == nullptr) {
            CUDA_CHECK(cudaMalloc(&d_buff, buff_size));
        } else {
            buffers->ensure_tmp(buff_size);
            d_buff = buffers->tmp_buffer.tmp_buffer;
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


        // Make the KokkosCRS matrix 
        ordinal_view_t rowmap(d_rowptrs, nrows + 1);
        ordinal_view_t colidx(d_colinds, nnz);
        values_view_t  values(d_csr_vals, nnz);
        typename LocalMatrix<KIT, DIT, VT>::KokkosCrs kokkos_csr("kokkos_csr",
                                                        nrows, ncols, nnz,
                                                        values,
                                                        rowmap,
                                                        colidx);

        return kokkos_csr;
    }


}

