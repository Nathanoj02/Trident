
#include <stdint.h>
#include "../common.h"

struct MLCArgs {
    char *mtx_path;
    char *mtx_name;
    uint32_t max_iter;
    float pruning_tol;
    bool add_diag;
    int node_size;
    Implementation impl;
};

char* mcl_extract_matrix_name(const char* filepath) {
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

void parse_args(int argc, char **argv, MLCArgs *args) {
    args->max_iter      = 10;
    args->pruning_tol   = 0.01f;

    int inc = 2;
    for (int i=1; i<argc; i+=inc) {
        inc = 2;
        const char * argname = (argv[i]);

        if (!strcmp(argname, "--mtx")) {
            args->mtx_path = argv[i+1];
            args->mtx_name = mcl_extract_matrix_name(args->mtx_path);
        } else if (!strcmp(argname, "--pruning-tol")) {
            args->pruning_tol = atof(argv[i+1]);
        } else if (!strcmp(argname, "--max-iter")) {
            args->max_iter = atoi(argv[i+1]);
        } else if (!strcmp(argname, "--add-self-loops")) {
            args->add_diag = true;
            inc = 1;
        }
    }

}
