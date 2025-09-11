#include "tile_stack.cuh"
#include "test_utils.cuh"


using IT = uint32_t;
using VT = float;


int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    static constexpr size_t stack_size = 8 * 1e6;

    TileStack<IT, VT> stack(stack_size, MPI_COMM_WORLD);
    mmio::CSR<IT, VT> tile;
    stack.push_tile(tile, 0);
    auto r_tile = stack.pop_tile(0,0);

    MPI_Finalize();
    return 0;
}
