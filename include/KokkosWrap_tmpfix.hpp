// KokkosWrap.hpp
#include <dmmio/dmmio.h>

#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_CcsMatrix.hpp>

namespace KokkosWrap {
    enum class MajorDim {
        ROWS, // i.e. CSR
        COLS  // i.e. CSC
    };

    template <typename KIT, typename DIT, typename VT>
    struct Matrix {
        dmmio::Partitioning* partitioning;  // still raw pointer to external partitioning

        // Instead of a raw union, use std::variant for safety
        std::variant<
            KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT>,
            KokkosSparse::CcsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT>
        > storage;

        MajorDim layout;  // which one we actually hold

        // mmio structure to access the raw pointers
        std::variant<
            mmio::CSR<KIT, VT>,
            mmio::CSC<KIT, VT>
        > dev_mmio;

        // ---- Constructor ----
        Matrix(dmmio::DCOO<DIT, VT>* dcoo, MajorDim T);
    };

    // These two function are exposed temporary for the test_kokkos C matrix
    template<typename KIT, typename VT>
    mmio::CSR<KIT, VT> rawptr_get(KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> kokkos_csr);

    template<typename KIT, typename VT>
    mmio::CSC<KIT, VT> rawptr_get(KokkosSparse::CcsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> kokkos_csc);
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

namespace KokkosWrap {

    template<typename KIT, typename VT>
    mmio::CSR<KIT, VT> rawptr_get(KokkosSparse::CrsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> kokkos_csr) {
        auto val_view     = kokkos_csr.values;
        auto rowmap_view  = kokkos_csr.graph.row_map;
        auto entries_view = kokkos_csr.graph.entries;

        mmio::CSR<KIT, VT> csr;
        csr.nnz     = kokkos_csr.nnz();
        csr.nrows   = kokkos_csr.numRows();
        csr.ncols   = kokkos_csr.numCols();
        csr.val     = val_view.data();
        csr.row_ptr = const_cast<int32_t*>(rowmap_view.data());
        csr.col_idx = const_cast<int32_t*>(entries_view.data());

        return(csr);
    }

    template<typename KIT, typename VT>
    mmio::CSC<KIT, VT> rawptr_get(KokkosSparse::CcsMatrix<VT, KIT, Kokkos::DefaultExecutionSpace, void, KIT> kokkos_csc) {
        auto val_view     = kokkos_csc.values;
        auto colmap_view  = kokkos_csc.graph.col_map;
        auto entries_view = kokkos_csc.graph.entries;

        mmio::CSC<KIT, VT> csc;
        csc.nnz     = kokkos_csc.nnz();
        csc.nrows   = kokkos_csc.numRows();
        csc.ncols   = kokkos_csc.numCols();
        csc.val     = val_view.data();
        csc.col_ptr = const_cast<int32_t*>(colmap_view.data());
        csc.row_idx = const_cast<int32_t*>(entries_view.data());

        return(csc);
    }

    template <typename KIT, typename DIT, typename VT>
    Matrix<KIT,DIT,VT>::Matrix(dmmio::DCOO<DIT, VT>* dcoo, MajorDim T)
            : partitioning(dcoo->partitioning), layout(T)
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
            val_h(i) = static_cast<VT>(1.0); // TODO: use real values if available
        }

        Kokkos::deep_copy(row_d, row_h);
        Kokkos::deep_copy(col_d, col_h);
        Kokkos::deep_copy(val_d, val_h);

        // --- Step 1: COO → CSR (row-major) ---
        auto csr = KokkosSparse::coo2crs(dcoo->coo->nrows, dcoo->coo->ncols, row_d, col_d, val_d);

        // --- Step 2: Decide layout ---
        if (T == MajorDim::ROWS) {
            storage   = csr; // keep CSR
            dev_mmio  = rawptr_get(csr);
        } else {
            // CSR → CSC (column-major)
            auto csc  = KokkosSparse::crs2ccs(csr);
            storage   = csc; // store CSC
            dev_mmio  = rawptr_get(csc);
        }
    }

    template struct Matrix<int32_t, uint32_t, double>;
    template struct Matrix<int32_t, uint32_t, float>;
}

