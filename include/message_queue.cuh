#pragma once
#include "common.h"
#include <atomic>


template <typename Message>
struct MessageQueue
{

    static constexpr std::size_t msg_size = sizeof(Message);


    struct Packet
    {
        int target;
        int offset;
        Message * msg;
        std::atomic<bool> ready;
    };


    MessageQueue(const size_t size, MPI_Comm comm):
        size(size), comm(comm),serviced(0)
    {

        packet = new Packet{-1,-1, nullptr, false};
        done_notify = false;

        messages = new Message[size];
        for (size_t i=0; i<size; i++)
        {
            messages[i] = -1;
        }
        MPI_Comm_rank(comm, &rank);
        MPI_MESSAGE = MPIType<Message>();

        MPI_Win_create(messages, sizeof(Message) * size, sizeof(Message), MPI_INFO_NULL, comm, &msg_win);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, msg_win);
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
                    Message result = messages[i];
                    messages[i] = -1;
                    MPI_Win_sync(msg_win);
                    serviced++;
                    return result;
                }

            }
        }
        return -2;
    }


    Message poll()
    {
        MPI_Win_sync(msg_win);
        MPI_Win_flush_all(msg_win);
        for (int i=0; i<size; i++)
        {
            if (messages[i] != -1)
            {
                Message result = messages[i];
                messages[i] = -1;
                serviced++;
                return result;
            }
        }
        return -2;
    }


    void notify(Message * msg, const int target, const size_t offset)
    {
        MPI_Accumulate(msg, 1, MPI_MESSAGE, target, offset, 1, MPI_MESSAGE, MPI_REPLACE, msg_win);
        MPI_Win_flush(target, msg_win);
    }


    void local_notify(Message * msg, const int target, const size_t offset)
    {
        packet->offset = offset;
        packet->target = target;
        packet->msg = msg;
        packet->ready.store(true, std::memory_order_release);
    }


    void poll_notify()
    {
        if (packet->ready.load(std::memory_order_acquire))
        {
            notify(packet->msg, packet->target, packet->offset);
        }
    }


    void tell_done_notifying()
    {
        done_notify = true;
    }


    bool done()
    {
        return (done_notify && serviced >= size);
    }


    ~MessageQueue()
    {
        MPI_Win_unlock_all(msg_win);
        MPI_Win_free(&msg_win);
        delete packet;
        delete[] messages;
    }

    MPI_Win msg_win;
    Message * messages;
    std::size_t size, serviced;
    int rank;
    MPI_Comm comm;
    MPI_Datatype MPI_MESSAGE;
    bool done_notify;

    Packet * packet;

};
