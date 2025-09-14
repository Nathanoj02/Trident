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

        // ---- Constructor ----
        Matrix(dmmio::DCOO<DIT, VT>* dcoo, MajorDim T);
    };
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
            storage = csr; // keep CSR
        } else {
            // CSR → CSC (column-major)
            auto csc = KokkosSparse::crs2ccs(csr);
            storage = csc; // store CSC
        }
    }

    template struct Matrix<int32_t, uint32_t, double>;
    template struct Matrix<int32_t, uint32_t, float>;
}

