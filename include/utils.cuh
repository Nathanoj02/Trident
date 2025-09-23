#pragma once
#include "common.h"
#define CUSPARSE_CHECK(call) do {                                    \
    cusparseStatus_t err = call;                                     \
    if (err != CUSPARSE_STATUS_SUCCESS) {                            \
        fprintf(stderr, "cuSPARSE error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cusparseGetErrorString(err));    \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while(0)


#define FLUSH_WAIT(s) \
    do { \
        fflush(stdout); \
        sleep(s); \
    } while(0);

#define OPSTR(X) ((X == dmmio::Operation::None) ? ("None") : ("Transpose") )

typedef struct {
    int rowidx;
    int colidx;
    int nodeidx;
    int flag;
} RemoteTile;

typedef struct {
    int rowidx;
    int colidx;
    int nodeidx;
} LocalTile;


template <typename T, typename... Args>
void print_h_arr(T * h_arr, const uint32_t n, const char * prefix, Args... args)
{
    std::cout<<'\n';
    fprintf(stdout, prefix, args...);
    for (uint32_t i=0; i<n; i++)
    {
        std::cout<<h_arr[i]<<',';
    }
    std::cout<<'\n';
    FLUSH_WAIT(0.5);
}


template <typename T, typename... Args>
void print_d_arr(T * d_arr, const uint32_t n, const char * prefix, Args... args)
{
    T * h_arr = (T*)d2h_copy(d_arr, n);
    print_h_arr(h_arr, n, prefix, args...);
    free(h_arr);
}


template <typename... Args>
void log_write(FILE * file, const char * msg, Args... args)
{
    fprintf(file, "\n");
    fprintf(file, msg, args...);
    fprintf(file, "\n");
    fflush(file);
}


template <typename T, typename... Args>
void log_h_arr(FILE * file, T * buf, size_t n, const char * prefix, Args... args)
{
    int limit = 100;
    fprintf(file, "\n");
    fprintf(file, prefix, args...);
    fprintf(file, "\n");
    fflush(file);
    for (size_t i=0; i<n; i++)
    {
        if (i < limit || (n-i) < limit)
        {
            if constexpr (std::is_integral<T>::value)
                fprintf(file, "%d, ", buf[i]);
            if constexpr (std::is_floating_point<T>::value)
                fprintf(file, "%f, ", buf[i]);
        }
        else
        {
            fprintf(file, "...");
            i = (n - limit);
        }
    }
    fprintf(file, "\n");
    fflush(file);
}

template <typename T, typename... Args>
void log_h_arr_unlimited(FILE * file, T * buf, size_t n, const char * prefix, Args... args)
{
    fprintf(file, "\n");
    fprintf(file, prefix, args...);
    fprintf(file, "\n");
    for (size_t i=0; i<n; i++)
    {
        if constexpr (std::is_integral<T>::value)
        {
            fprintf(file, "%d, ", buf[i]);
        }
        if constexpr (std::is_floating_point<T>::value)
        {
            fprintf(file, "%f, ", buf[i]);
        }
        if ((i + 1) % 50 == 0)
        {
            fprintf(file, "\n");
        }
    }
    fprintf(file, "\n");
    fflush(file);
}


template <typename T, typename... Args>
void log_d_arr(FILE * file, T * d_buf, size_t n, const char * prefix, Args... args)
{
    T * h_buf = (T*)d2h_copy(d_buf, n);
    log_h_arr<T, Args...>(file, h_buf, n, prefix, args...);
    freeMem(h_buf);
}


template <typename T, typename... Args>
void log_d_arr_unlimited(FILE * file, T * d_buf, size_t n, const char * prefix, Args... args)
{
    T * h_buf = d2h_copy(d_buf, n);
    log_h_arr_unlimited<T, Args...>(file, h_buf, n, prefix, args...);
    freeMem(h_buf);
}

template <typename... Args>
void print_rk0(dmmio::ProcessGrid * grid, const char * msg, Args... args)
{
    if (grid->global_rank==0) 
    {
        fprintf(stdout, "\n");
        fprintf(stdout, msg, args...);
        fprintf(stdout, "\n");
    }
    FLUSH_WAIT(0.5);
}


template <typename... Args>
void print_rkn(dmmio::ProcessGrid * grid, const char * msg, Args... args)
{
    print_rkn(grid->global_rank, msg, args...);
}


template <typename... Args>
void print_rkn(int rank, const char * msg, Args... args)
{
    fprintf(stdout, "\n" GREEN "Process %d --- " RESET, rank);
    fprintf(stdout, msg, args...);
    fprintf(stdout, "\n");
    FLUSH_WAIT(0.5);
}


template <typename... Args>
void par_print(const char * str, Args... args)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);
    fprintf(stdout, "---Process %d---\n", rank);
    fprintf(stdout, str, args...);
    fprintf(stdout, "----------------\n");
    fflush(stdout);
    sleep(1);
    MPI_Barrier(MPI_COMM_WORLD);
}
