#include "tile_queue.cuh"
#include "test_utils.cuh"


using IT = uint32_t;
using VT = float;


int main(int argc, char ** argv)
{

    static constexpr size_t queue_size = 1e9;

    TileQueue<IT, VT> queue(queue_size);

    return 0;
}
