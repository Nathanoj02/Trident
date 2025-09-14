/*
#ifndef KOKKOS_WRAP_H
#define KOKKOS_WRAP_H

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

#endif // KOKKOS_WRAP_H
*/
