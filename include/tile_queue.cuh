#pragma once
#include "common.h"



template <typename IT, typename VT>
class TileQueue
{


public:

    TileQueue(const size_t init_size):
        size(init_size)
    {
    }


private:
    size_t size;

};
