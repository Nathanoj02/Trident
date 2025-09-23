#pragma once
#include "common.h"
#include "KokkosWrap.hpp"




template <typename IT, typename VT>
using KWrapDMat = typename KokkosWrap::DistribuitedMatrix<IT, IT, VT>;

template <typename IT, typename VT>
using KWrapLMat = typename KokkosWrap::LocalMatrix<IT, IT, VT>;

template <typename IT, typename VT>
mmio::CSX<IT, VT> * hns_spgemm_get(KWrapDMat<IT, VT>& kwd_A, KWrapDMat<IT, VT>& kwd_B);



