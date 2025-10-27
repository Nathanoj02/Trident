#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include <common.h>

// You have to be sure target != myrank
#define MYREQUEST(MYRK, TARGET) ((TARGET<MYRK) ? (TARGET) : (TARGET-1))

#define NREP 10
#define WARMUPS 3

// P2P Allgatherv for device buffers
template<typename DT>
void P2P_Allgatherv(const DT* send_buf,
                    int count_send,
                    MPI_Datatype datatype_send,
                    DT* recv_buf,
                    const int* count_recv,
                    const int* displacements,
                    MPI_Datatype datatype_recv,
                    MPI_Comm mpi_communicator,
                    cudaStream_t stream = 0,
                    int my_rank = -1,
                    int comm_size = -1)
{

#ifdef DEBUG_P2P_ALLGATHERV
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif

    if (my_rank == -1)
        MPI_Comm_rank(mpi_communicator, &my_rank);

    if (comm_size == -1)
        MPI_Comm_size(mpi_communicator, &comm_size);

    int tag;
    int nreqs = 2 * (comm_size-1);

    MPI_Status  *statuses = (MPI_Status*) malloc(nreqs * sizeof(MPI_Status));
    MPI_Request *requests = (MPI_Request*)malloc(nreqs * sizeof(MPI_Request));

    for (int i = 0; i < comm_size; i++) {
        if (i != my_rank) {
            // send
            tag = my_rank * comm_size + i;
            MPI_Isend(send_buf, count_send, datatype_send, i, tag, mpi_communicator, &(requests[MYREQUEST(my_rank, i)]));

#ifdef DEBUG_P2P_ALLGATHERV
            fprintf(stdout, "[%d, %d] sent to %d with tag %d\n", world_rank, my_rank, i, tag);
            fflush(stdout);
#endif

            // recv
            tag = i * comm_size + my_rank;
            MPI_Irecv(recv_buf + displacements[i], count_recv[i], datatype_recv, i, tag,
                      mpi_communicator, &(requests[comm_size-1 + MYREQUEST(my_rank, i)]));

#ifdef DEBUG_P2P_ALLGATHERV
            fprintf(stdout, "[%d, %d] received from %d with tag %d\n", world_rank, my_rank, i, tag);
            fflush(stdout);
#endif
        } else {
            // Local copy: device to device
            CUDA_CHECK(cudaMemcpyAsync(recv_buf + displacements[i],
                                  send_buf,
                                  count_send * sizeof(DT),
                                  cudaMemcpyDeviceToDevice,
                                  stream));

#ifdef DEBUG_P2P_ALLGATHERV
            fprintf(stdout, "[%d, %d] performed self communication with D2D copy\n", world_rank, my_rank);
            fflush(stdout);
#endif
        }
    }

    MPI_Waitall(nreqs, requests, statuses);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_STATUS_CHECK(nreqs, statuses, mpi_communicator);

    free(requests);
    free(statuses);
}

void print_comm_info(MPI_Comm comm) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dev_id;
    CUDA_CHECK(cudaGetDevice(&dev_id));

    MPI_ALL_PRINT(
        fprintf(fp, "World rank %d: comm_rank=%d comm_size=%d, device_id=%d\n",
            world_rank, rank, size, dev_id);
    )
    MPI_Barrier(MPI_COMM_WORLD);
}


#define DTYPE int
#define MPIDTYPE MPI_INT

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int deviceCount, dev_id, my_dev;
    cudaGetDeviceCount(&deviceCount);
    my_dev =world_rank % deviceCount;

    // Just to set all the GPUs
    for (int i=0; i<deviceCount; i++) {
        CUDA_CHECK(cudaSetDevice(i));
    }

    CUDA_CHECK(cudaSetDevice(my_dev));
    CUDA_CHECK(cudaGetDevice(&dev_id));

    if (my_dev != dev_id) {
        fprintf(stderr, "[%d] error in device assign\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_ALL_PRINT(
        fprintf(fp, "Rank %d has device %d assigned\n", world_rank, dev_id);
    )

    // ---------------------------------------------------------------------------------

    // First communicator: all processes in the same node
    MPI_Comm intranode_comm;
    int color_intranode = world_rank / deviceCount;
    MPI_Comm_split(MPI_COMM_WORLD, color_intranode, world_rank, &intranode_comm);

    // Second communicator: round-robin grouping by "local device id"
    MPI_Comm internode_comm;
    int color_internode = world_rank % deviceCount;
    MPI_Comm_split(MPI_COMM_WORLD, color_internode, world_rank, &internode_comm);

    // Third communicator: all processes in a single communicator
    MPI_Comm singleton_comm;
    int color_singleton = world_rank;
    MPI_Comm_split(MPI_COMM_WORLD, color_singleton, world_rank, &singleton_comm);

    int nameid = 0;
    std::vector<const char*> names = {"world", "intranode", "internode", "singleton"};
    std::vector<MPI_Comm> comms = {MPI_COMM_WORLD, intranode_comm, internode_comm, singleton_comm};

    for (MPI_Comm communicator : comms ) {
        int comm_size, comm_rank;
        MPI_Comm_rank(communicator, &comm_rank);
        MPI_Comm_size(communicator, &comm_size);
        const char* myname = names[nameid];


#ifdef NVTX_PROFILING
        char mytagsttring[50];
        sprintf(mytagsttring, "Test_%s_comm", myname);
        NVTX_PUSH_RANGE(mytagsttring, 1);
#endif

        // ----------- Init sending buffers -----------
        // Example: each rank sends variable size buffers
        int base_size = 1 << 20;
        DTYPE *d_send_buf, *d_recv_buf;
        int nelements = 4 * base_size * ((world_rank%2)+ 1);
        CUDA_CHECK(cudaMalloc(&d_send_buf, nelements * sizeof(DTYPE)));

        // host init
        DTYPE *h_send_buff = (DTYPE*)malloc(sizeof(DTYPE)*nelements);
        for (int i = 0; i < nelements; i++) h_send_buff[i] = world_rank;
        CUDA_CHECK(cudaMemcpy(d_send_buf, h_send_buff, nelements * sizeof(DTYPE), cudaMemcpyHostToDevice));

        if (base_size < 20) {
            MPI_ALL_PRINT(
                fprintf(fp, "send_buffer(%d): ", nelements);
                for (int i=0; i<nelements; i++) fprintf(fp, " %d", h_send_buff[i]);
                fprintf(fp, "\n");
            )
        }

        // ----------- Compute recv buffer and displacements -----------
        int *recv_counts = (int*)malloc(sizeof(int)*comm_size);
        MPI_Allgather(&nelements, 1, MPI_INT, recv_counts, 1, MPI_INT, communicator);

        int total_recv_count = recv_counts[comm_size-1];
        int *displacements = (int*)malloc(comm_size * sizeof(int));

        displacements[0] = 0;
        for (int i = 0; i < comm_size-1; i++) {
            displacements[i+1] = displacements[i]+recv_counts[i];
            total_recv_count += recv_counts[i];
        }

        CUDA_CHECK(cudaMalloc(&d_recv_buf, total_recv_count * sizeof(DTYPE)));
        CUDA_CHECK(cudaMemset(d_recv_buf, 0, total_recv_count * sizeof(DTYPE)));

        // ----------- P2P test -----------

        for (int r=-WARMUPS; r<NREP; r++) {
#ifdef NVTX_PROFILING
            NVTX_PUSH_RANGE("P2P_Allgatherv", 0);
#endif

            // Run custom P2P allgatherv
            P2P_Allgatherv(d_send_buf, nelements, MPIDTYPE, d_recv_buf, recv_counts, displacements, MPIDTYPE, communicator,
                        0, comm_rank, comm_size);

#ifdef NVTX_PROFILING
            NVTX_POP_RANGE;
#endif
        }

        // Copy result back to host
        DTYPE *h_p2p = (DTYPE*)malloc(total_recv_count * sizeof(DTYPE));
        CUDA_CHECK(cudaMemcpy(h_p2p, d_recv_buf, total_recv_count * sizeof(DTYPE), cudaMemcpyDeviceToHost));

        if (base_size < 20) {
            MPI_ALL_PRINT(
                fprintf(fp, "send_buffer(%d): ", total_recv_count);
                for (int i=0; i<total_recv_count; i++) fprintf(fp, " %d", h_p2p[i]);
                fprintf(fp, "\n");
            )
        }

        // Compare with MPI_Allgatherv (on host)
        DTYPE *h_allgatherv = (DTYPE*)malloc(total_recv_count * sizeof(DTYPE));

#ifdef NVTX_PROFILING
        NVTX_PUSH_RANGE("host_Allgatherv", 0);
#endif

        MPI_Allgatherv(h_send_buff, nelements, MPIDTYPE,
                    h_allgatherv, recv_counts, displacements, MPIDTYPE,
                    communicator);

#ifdef NVTX_PROFILING
        NVTX_POP_RANGE;
#endif

        // Check correctness
        bool ok = true;
        for (int i = 0; i < total_recv_count; i++) {
            if (h_p2p[i] != h_allgatherv[i]) {
                fprintf(stderr, "Rank %d mismatch at %d: p2p=%d allg=%d\n",
                        world_rank, i, h_p2p[i], h_allgatherv[i]);
                ok = false;
                break;
            }
        }

        if (world_rank == 0) {
            if (ok) printf("P2P_Allgatherv matches MPI_Allgatherv ✅\n");
            else    printf("Mismatch ❌\n");
        }

        // ---- test with cuda-aware MPI ----
        CUDA_CHECK(cudaMemset(d_recv_buf, 0, total_recv_count * sizeof(DTYPE)));

        for (int r=-WARMUPS; r<NREP; r++) {
#ifdef NVTX_PROFILING
            NVTX_PUSH_RANGE("device_Allgatherv", 0);
#endif

            MPI_Allgatherv(d_send_buf, nelements, MPIDTYPE,
                        d_recv_buf, recv_counts, displacements, MPIDTYPE,
                        communicator);

#ifdef NVTX_PROFILING
            NVTX_POP_RANGE;
#endif
        }

        // Cleanup
        free(h_send_buff); free(displacements); free(recv_counts);
        free(h_p2p); free(h_allgatherv);
        CUDA_CHECK(cudaFree(d_send_buf));
        CUDA_CHECK(cudaFree(d_recv_buf));

#ifdef NVTX_PROFILING
        NVTX_POP_RANGE;
#endif
        nameid++;
    }

    MPI_Finalize();
    return 0;
}

