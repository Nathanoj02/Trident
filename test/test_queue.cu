#include "test_utils.cuh"
#include "message_queue.cuh"


using IT = uint32_t;
using VT = float;


int main(int argc, char ** argv)
{
    int req;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &req);

    std::cout<<"MPI thread support: "<<req<<std::endl;

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    static const size_t queue_size = nranks;

    MessageQueue<int> queue(queue_size, MPI_COMM_WORLD);
    queue.wait();
    queue.notify(NULL, 0, 0);


    MPI_Finalize();
    return 0;
}
