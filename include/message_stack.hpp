#pragma once
#include "common.h"


template <typename Message>
struct MessageStack
{

    static constexpr std::size_t msg_size = sizeof(Message);


    MessageStack(const size_t size, MPI_Comm comm):
        size(size), comm(comm), tail(0)
    {

        messages = new Message[size];
        MPI_Comm_rank(comm, &rank);
        MPI_MESSAGE = MPIType<Message>();

        MPI_Win_create(messages, sizeof(Message) * size, sizeof(Message), MPI_INFO_NULL, comm, &msg_win);
        MPI_Win_create(&tail, sizeof(size_t), sizeof(size_t), MPI_INFO_NULL, comm, &tail_win);
    }


    void push(const int target, Message * msg)
    {
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, 0, msg_win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, 0, tail_win);

        int one = 1;
        std::size_t remote_tail;
        MPI_Fetch_and_op(&one, &remote_tail, MPI_INT, target, 0, MPI_SUM, tail_win);
        MPI_Win_flush(target, tail_win);

        MPI_Put(msg, 1, MPI_MESSAGE, 1, remote_tail, 1, MPI_MESSAGE, msg_win);

        MPI_Win_unlock(target, msg_win);
        MPI_Win_unlock(target, tail_win);
    }


    Message * pop()
    {
        if (tail <= 0)
        {
            return nullptr;
        }

        Message * msg = new Message();

        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, msg_win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, tail_win);

        MPI_Win_sync(tail_win);
        MPI_Win_sync(msg_win);

        memcpy(msg, messages + (tail - 1), sizeof(Message));
        tail -= 1;

        // TODO: Do I need these?
        MPI_Win_sync(tail_win);
        MPI_Win_sync(msg_win);

        MPI_Win_unlock(rank, msg_win);
        MPI_Win_unlock(rank, tail_win);

        return msg;
    }


    ~MessageStack()
    {
        MPI_Win_free(&msg_win);
        MPI_Win_free(&tail_win);
        delete[] messages;
    }

    MPI_Win msg_win, tail_win;
    Message * messages;
    std::size_t size, tail;
    int rank;
    MPI_Comm comm;
    MPI_Datatype MPI_MESSAGE;

};
