#pragma once

namespace KokkosWrap {

template <typename IT, typename VT>
struct Triple {
    IT row;
    IT col;
    VT val;
};

}

