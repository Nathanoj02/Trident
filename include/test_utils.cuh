
#ifndef TEST_UTILS_CUH
#define TEST_UTILS_CUH

#include "common.h"

#ifdef USE_NVTX
#include "../include/nvtxmacro.h"
COUNTER_DEF(0);
#endif

#define NUM_TRIALS 50



struct config
{
    const char * impl_str;
    Implementation impl;
    bool skip_spgemm;
    bool verbose;
    bool spcomm;
    bool Acsc;
    bool mem_efficient;

    int nprocrows;
    int nproccols;

    const char * matnameA;
    const char * matnameB;
    const char * matnameC;

    const char * matpathA;
    const char * matpathB;
    const char * matpathC;


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
    config->skip_spgemm = false;
    config->verbose     = false;
    config->spcomm      = false;
    config->Acsc        = false;
    config->impl_str  = "none";
    config->nprocrows = 1;
    config->nproccols = 1;
    config->mem_efficient = false;

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
            config->matnameC = extract_matrix_name(config->matpathC);
        }
        else if (!strcmp(argname, "--spcomm"))
        {
            config->spcomm = true;
            inc = 1;
        }
        else if (!strcmp(argname, "--Acsc"))
        {
            config->Acsc = true;
            inc = 1;
        }
        else if (!strcmp(argname, "--skip-spgemm"))
        {
            config->skip_spgemm = true;
            inc = 1;
        }
        else if (!strcmp(argname, "--verbose"))
        {
            config->verbose = true;
            inc = 1;
        }
        else if (!strcmp(argname, "--impl"))
        {
            config->impl_str = argv[i+1];
        }
        else if (!strcmp(argname, "--mem-efficient"))
        {
            config->mem_efficient = true;
            inc = 1;
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

    if (!strcmp(config->impl_str, "get")) config->impl = Implementation::GET;
    else if (!strcmp(config->impl_str, "put")) config->impl = Implementation::PUT;
    else if (!strcmp(config->impl_str, "sendrecv")) config->impl = Implementation::SENDRECV;
    else if (!strcmp(config->impl_str, "none")) config->impl = STDIMPL;
    else {
        fprintf(stdout, "Error: unrecognized implementation %s, valid implementations are:\n\t%s\n\t%s\n\t%s\n",
                config->impl_str, "get", "put", "sendrecv");
        exit(__LINE__);
    }

}


#endif
