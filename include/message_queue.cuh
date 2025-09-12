#pragma once
#include "common.h"


template <typename Message>
struct MessageQueue
{

    static constexpr std::size_t msg_size = sizeof(Message);


    MessageQueue(const size_t size, MPI_Comm comm):
        size(size), comm(comm), serviced(0)
    {

        messages = new Message[size];
        MPI_Comm_rank(comm, &rank);
        MPI_MESSAGE = MPIType<Message>();

        MPI_Win_create(messages, sizeof(Message) * size, sizeof(Message), MPI_INFO_NULL, comm, &msg_win);
        MPI_Win_lock_all(0, msg_win);
    }


    Message wait()
    {
        while (serviced < size)
        {
            MPI_Win_sync(msg_win);
            for (int i=0; i<size; i++)
            {
                if (messages[i] != -1)
                {
                    messages[i] = -1;
                    serviced++;
                    return messages[i];
                }
            }
        }
        return -1;
    }


    void notify(Message * msg, const int target, const size_t offset)
    {
        MPI_Accumulate(msg, 1, MPI_MESSAGE, target, offset, 1, MPI_MESSAGE, MPI_REPLACE, msg_win);
        MPI_Win_flush(rank, msg_win);
    }


    ~MessageQueue()
    {
        MPI_Win_free(&msg_win);
        delete[] messages;
    }

    MPI_Win msg_win;
    Message * messages;
    std::size_t size, serviced;
    int rank;
    MPI_Comm comm;
    MPI_Datatype MPI_MESSAGE;

};
