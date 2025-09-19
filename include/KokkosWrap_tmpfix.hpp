#pragma once

// KokkosWrap.hpp
#include <dmmio/dmmio.h>

#include <variant>
#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_CcsMatrix.hpp>

using MajorDim = mmio::MajorDim;

namespace KokkosWrap {

    template <typename KIT, typename DIT, typename VT>
    struct DistribuitedMatrix {
        dmmio::Partitioning* partitioning;  // still raw pointer to external partitioning

        // Instead of a raw union, use std::variant for safety
        std::variant<
            KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT>,
            KokkosSparse::CcsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT>
        > storage;

        // MajorDim layout;  // which one we actually hold // Now inside the CSX

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
        bool initialized;
        KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> storage;

        // ---- Constructor ----
        LocalMatrix();
        LocalMatrix(mmio::CSX<DIT, VT>* mmio_csx);
        LocalMatrix(DIT nrows, DIT ncols, DIT nnz, DIT* ptrvec, DIT* idxvec, VT* values, MajorDim layout);

        // In-place SpGEMM: C = C + A*B
        static void sp_mma(const LocalMatrix& A, const LocalMatrix& B, LocalMatrix& C);

        // Free memory & decostructor
        void freeBuffers();
        ~LocalMatrix();
    };

    // These two function are exposed temporary for the test_kokkos C matrix
    template<typename KIT, typename VT>
    mmio::CSX<KIT, VT>* rawptr_get(KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> kokkos_csr);

    template<typename KIT, typename VT>
    mmio::CSX<KIT, VT>* rawptr_get(KokkosSparse::CcsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> kokkos_csc);

    template<typename KIT, typename VT>
    mmio::CSX<KIT, VT>* rawptr_get(LocalMatrix<KIT, KIT, VT> kokkos_csr);
}

// KokkosWrap.cpp
#include <variant>
#include <dmmio/partitioning.h>

#include "../include/KokkosWrap.hpp"

// CHECK IF STILL REQUIRED
#include <cstdint>
#include <memory>
// -----------------------

#include <KokkosSparse_CooMatrix.hpp>
#include <KokkosSparse_coo2crs.hpp>
#include <KokkosSparse_crs2ccs.hpp>
#include <KokkosSparse_ccs2crs.hpp>
#include <KokkosSparse_spadd.hpp>

namespace KokkosWrap {

    template<typename KIT, typename VT>
    mmio::CSX<KIT, VT>* rawptr_get(KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> kokkos_csr) {
        auto val_view     = kokkos_csr.values;
        auto rowmap_view  = kokkos_csr.graph.row_map;
        auto entries_view = kokkos_csr.graph.entries;

        mmio::CSX<KIT, VT> *csx = (mmio::CSX<KIT, VT>*)malloc(sizeof(mmio::CSX<KIT, VT>));
        csx->majordim = MajorDim::ROWS;
        csx->nnz      = kokkos_csr.nnz();
        csx->nrows    = kokkos_csr.numRows();
        csx->ncols    = kokkos_csr.numCols();
        csx->val      = val_view.data();
        csx->ptr_vec  = const_cast<int32_t*>(rowmap_view.data());
        csx->idx_vec  = const_cast<int32_t*>(entries_view.data());

        return(csx);
    }

    template<typename KIT, typename VT>
    mmio::CSX<KIT, VT>* rawptr_get(KokkosSparse::CcsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> kokkos_csc) {
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
    mmio::CSX<KIT, VT>* rawptr_get(LocalMatrix<KIT, KIT, VT> kokkos_csr) {
        return(rawptr_get(kokkos_csr.storage));
    }

    template <typename KIT, typename DIT, typename VT>
    DistribuitedMatrix<KIT,DIT,VT>::DistribuitedMatrix(dmmio::DCOO<DIT, VT>* dcoo, MajorDim T)
            : partitioning(dcoo->partitioning)
    {
        int nnz = dcoo->coo->nnz;
        Kokkos::View<KIT*> row_d("row", nnz);
        Kokkos::View<KIT*> col_d("col", nnz);
        Kokkos::View<VT*> val_d("val", nnz);

        auto row_h = Kokkos::create_mirror_view(row_d);
        auto col_h = Kokkos::create_mirror_view(col_d);
        auto val_h = Kokkos::create_mirror_view(val_d);

        for (int i = 0; i < nnz; i++) {
            row_h(i) = static_cast<KIT>(dcoo->coo->row[i]);
            col_h(i) = static_cast<KIT>(dcoo->coo->col[i]);
            if (dcoo->coo->val == nullptr)
                val_h(i) = static_cast<VT>(1.0); // TODO: use real values if available
            else
                val_h(i) = static_cast<VT>(dcoo->coo->val[i]);
        }

        Kokkos::deep_copy(row_d, row_h);
        Kokkos::deep_copy(col_d, col_h);
        Kokkos::deep_copy(val_d, val_h);

        // --- Step 1: COO → CSR (row-major) ---
        auto csr = KokkosSparse::coo2crs(dcoo->coo->nrows, dcoo->coo->ncols, row_d, col_d, val_d);

        // --- Step 2: Decide layout ---
        if (T == MajorDim::ROWS) {
            storage   = csr; // keep CSR
            mmio_csx  = rawptr_get(csr);
        } else {
            // CSR → CSC (column-major)
            auto csc  = KokkosSparse::crs2ccs(csr);
            storage   = csc; // store CSC
            mmio_csx  = rawptr_get(csc);
        }
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
    LocalMatrix<KIT,DIT,VT>::LocalMatrix(mmio::CSX<DIT, VT>* mmio_csx) : initialized(true)
    {
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
            ordinal_view_t colmap(mmio_csx->ptr_vec, mmio_csx->ncols + 1);
            ordinal_view_t rowidx(mmio_csx->idx_vec, mmio_csx->nnz);
            values_view_t  values(mmio_csx->val,     mmio_csx->nnz);

            KokkosSparse::CcsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> kokkos_csc("kokkos_csc",
                                            mmio_csx->nrows, mmio_csx->ncols, mmio_csx->nnz,
                                            values,
                                            colmap,
                                            rowidx
            );

            storage = KokkosSparse::ccs2crs(kokkos_csc);
        }
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
    LocalMatrix<KIT,DIT,VT>::LocalMatrix(DIT nrows, DIT ncols, DIT nnz, DIT* ptrvec, DIT* idxvec, VT* values, MajorDim layout)
    : LocalMatrix(csx_gen<DIT, VT>((nrows, ncols, nnz, ptrvec, idxvec, values, layout))) { }

    template <typename KIT, typename DIT, typename VT>
    void LocalMatrix<KIT,DIT,VT>::sp_mma(const LocalMatrix& A, const LocalMatrix& B, LocalMatrix& C) {

        using csr_matrix_type = typename KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT>;
        csr_matrix_type product = KokkosSparse::spgemm<csr_matrix_type>(A.storage, false, B.storage, false);

        if (C.initialized == false) {
            C.storage = product;
            C.initialized = true;
        } else {
            csr_matrix_type accumulator;

            // Create KokkosKernelHandle
            using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
                KokkosKernels::default_size_type, KIT, VT,
                Kokkos::DefaultExecutionSpace,
                typename Kokkos::DefaultExecutionSpace::memory_space,
                typename Kokkos::DefaultExecutionSpace::memory_space>;


            KernelHandle kh;
            kh.create_spadd_handle(false);

            KokkosSparse::spadd_symbolic(&kh, product, C.storage, accumulator);
            KokkosSparse::spadd_numeric(&kh, 1.0, product, 1.0, C.storage, accumulator);
            kh.destroy_spadd_handle();

            C.storage = accumulator;
        }
    }

    template struct DistribuitedMatrix<int32_t, uint32_t, double>;
    template struct DistribuitedMatrix<int32_t, uint32_t, float>;
}

