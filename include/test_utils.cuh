
#ifndef TEST_UTILS_CUH
#define TEST_UTILS_CUH

#include "common.h"

#ifdef USE_NVTX
#include "../include/nvtxmacro.h"
COUNTER_DEF(0);
#endif

#define NUM_TRIALS 50


enum ExperimentType
{
    BASELINE,
    SPCOMM,
    OVERLAP,
    PETSC,
    TRILINOS,
    COMBBLAS,
};

ExperimentType strtoextype(const char * s)
{
    if (!strcmp(s, "baseline"))
    {
        return BASELINE;
    }
    if (!strcmp(s, "spcomm"))
    {
        return SPCOMM;
    }
    if (!strcmp(s, "overlap"))
    {
        return OVERLAP;
    }
    if (!strcmp(s, "petsc"))
    {
        return PETSC;
    }
    if (!strcmp(s, "trilinos"))
    {
        return TRILINOS;
    }
    if (!strcmp(s, "combblas"))
    {
        return COMBBLAS;
    }
    fprintf(stderr, "Error: %s is not a valid value for --type\n", s);
    exit(EXIT_FAILURE);
}

enum spcomm_impl
{
    ALLGATHERV_ROWPTRS,
    ALLGATHERV_NOROWPTRS,
    ALLGATHERV_HYBRID,
    ALLTOALLV_ROWPTRS,
    ALLTOALLV_NOROWPTRS
} typedef SpcommImpl;


SpcommImpl strtoimpl(const char * str)
{
    if (!strcmp(str, "allgatherv_rowptrs"))
    {
        return ALLGATHERV_ROWPTRS;
    }
    if (!strcmp(str, "allgatherv_norowptrs"))
    {
        return ALLGATHERV_NOROWPTRS;
    }
    if (!strcmp(str, "allgatherv_hybrid"))
    {
        return ALLGATHERV_HYBRID;
    }
    if (!strcmp(str, "alltoallv_rowptrs"))
    {
        return ALLTOALLV_ROWPTRS;
    }
    if (!strcmp(str, "alltoallv_norowptrs"))
    {
        return ALLTOALLV_NOROWPTRS;
    }
    fprintf(stderr, "Error: %s is not a valid value for --impl\n", str);
    exit(EXIT_FAILURE);
}


enum bfs_impl
{
    BFS_OURS,
    BFS_TRILINOS
} typedef BfsImpl;


BfsImpl strtobfsimpl(const char * str)
{
    if (!strcmp(str, "ours"))
    {
        return BFS_OURS;
    }
    if (!strcmp(str, "trilinos"))
    {
        return BFS_TRILINOS;
    }
    fprintf(stderr, "Error: %s is not a valid value for --bfs_impl\n", str);
    exit(EXIT_FAILURE);
}


struct config
{
    ExperimentType type;
    const char * type_str;
    bool do_check;
    bool transpose_B;
    int part_num;

    int nprocrows;
    int nproccols;

    const char * matnameA;
    const char * matnameB;
    const char * matnameC;

    const char * matpathA;
    const char * matpathB;
    const char * matpathC;

    const char * file_suffix;

    bool scramble;
    SpcommImpl impl;

    uint64_t d;
    uint64_t seed;
    BfsImpl bfs_impl;
    const char * bfs_implstr;

    float tol;
    BfsImpl mcl_impl;
    const char * mcl_implstr;

} typedef Config;


char* extract_matrix_name(const char* filepath) 
{
    // Find the last occurrence of '/'
    const char* lastSlash = strrchr(filepath, '/');
    if (!lastSlash) {
        lastSlash = filepath; // No slash found, start from the beginning
    } else {
        lastSlash++; // Move past the '/'
    }

    // Find the occurrence of '.mtx'
    const char* dotMtx = strstr(lastSlash, ".");
    if (!dotMtx) {
        return NULL; // '.mtx' not found, invalid format
    }

    // Calculate the length of the matrix name
    size_t nameLen = dotMtx - lastSlash;

    // Allocate memory for the matrix name (+1 for null terminator)
    char* matrixName = (char*)malloc(nameLen + 1);
    if (!matrixName) {
        return NULL; // Memory allocation failed
    }

    // Copy the matrix name
    strncpy(matrixName, lastSlash, nameLen);
    matrixName[nameLen] = '\0'; // Null-terminate the string

    return matrixName;
}


void parse_args(int argc, char ** argv, Config * config)
{
    config->do_check = false;
    config->part_num = 0;
    config->transpose_B = false;
    config->file_suffix = "none";
    config->type_str = "none";
    config->scramble = false;

    config->nprocrows = 1;
    config->nproccols = 1;

    int inc = 2;
    for (int i=1; i<argc; i+=inc)
    {
        inc = 2;
        const char * argname = (argv[i]);

        if (!strcmp(argname, "--matA"))
        {
            config->matpathA = argv[i+1];
            config->matnameA = extract_matrix_name(config->matpathA);
        }
        else if (!strcmp(argname, "--matB"))
        {
            config->matpathB= argv[i+1];
            config->matnameB = extract_matrix_name(config->matpathB);
        }
        else if (!strcmp(argname, "--matC"))
        {
            config->matpathC= argv[i+1];
            config->do_check = true;
            config->matnameC = extract_matrix_name(config->matpathC);
        }
        else if (!strcmp(argname, "--type"))
        {
            config->type_str = argv[i+1];
            config->type = strtoextype(argv[i+1]);
        }
        else if (!strcmp(argname, "--fsuffix"))
        {
            config->file_suffix = argv[i+1];
        }
        else if (!strcmp(argname, "--part"))
        {
            config->part_num = atoi(argv[i+1]);
        }
        else if (!strcmp(argname, "--impl"))
        {
            config->impl = strtoimpl(argv[i+1]);
        }
        else if (!strcmp(argname, "--transB"))
        {
            config->transpose_B = true;
            inc = 1;
        }
        else if (!strcmp(argname, "--scramble"))
        {
            config->scramble = true;
            inc = 1;
        }
        else if (!strcmp(argname, "--bfs_impl"))
        {
            config->bfs_implstr = argv[i+1];
            config->bfs_impl = strtobfsimpl(argv[i+1]);
        }
        else if (!strcmp(argname, "--d"))
        {
            config->d = atoll(argv[i+1]);
        }
        else if (!strcmp(argname, "--seed"))
        {
            config->seed = atoll(argv[i+1]);
        }
        else if (!strcmp(argname, "--mcl_impl"))
        {
            config->mcl_implstr = argv[i+1];
            config->mcl_impl = strtobfsimpl(argv[i+1]);
        }
        else if (!strcmp(argname, "--tol"))
        {
            config->tol = (float)atof(argv[i+1]);
        }

        else if (!strcmp(argname, "--2D-pgrid"))
        {
            const char * grid_str = argv[i+1];
            char * x_pos = (char*)strchr(grid_str, 'x');

            if (x_pos == NULL) {
                fprintf(stderr, "Invalid format for --2D-pgrid. Expected NxM (e.g., 2x3).\n");
                exit(EXIT_FAILURE);
            }

            // Split the string
            *x_pos = '\0';  // Temporarily null-terminate at 'x'
            config->nprocrows = atoi(grid_str);
            config->nproccols = atoi(x_pos + 1);
            *x_pos = 'x';  // Restore string if needed elsewhere

            // Optional: add validation
            if (config->nprocrows <= 0 || config->nproccols <= 0) {
                fprintf(stderr, "Invalid values for --2D-pgrid: both N and M must be > 0.\n");
                exit(EXIT_FAILURE);
            }
        }

    }

}


#endif
