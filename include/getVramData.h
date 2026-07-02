#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include "common.h"

#include <vector>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

struct MyMemData {
    size_t current_free_mem;
    size_t current_total_mem;

    const char*         counter_name;
    size_t              trial_total_mem;
    std::vector<size_t> trial_free_mem;
    std::vector<char*>  trial_names;
    size_t              internal_counter;

    // Constructor
    MyMemData(const char* input_counter_name)
        : counter_name(input_counter_name),
          trial_total_mem(0),
          internal_counter(0)
    {
        CUDA_CHECK(cudaMemGetInfo(&current_free_mem, &current_total_mem));
        trial_total_mem = current_total_mem;
#ifdef DETAILED_TIMERS
        fprintf(stdout, "Memory counter %s -- total memory %zu\n", counter_name, trial_total_mem);
#endif
    }

    // Destructor (frees allocated trial name copies)
    ~MyMemData() {
        for (char* name : trial_names) {
            delete[] name;
        }
    }

    // Append measurement
    void append_measure(const char* input_trial_name) {
        CUDA_CHECK(cudaMemGetInfo(&current_free_mem, &current_total_mem));

        if (current_total_mem != trial_total_mem) {
            fprintf(stderr,
                    "Error at insertion %zu (%s) in counter %s: total memory mismatch (%zu != %zu)\n",
                    internal_counter,
                    input_trial_name,
                    counter_name,
                    current_total_mem,
                    trial_total_mem);
        }

        // Deep copy of name
        size_t len = std::strlen(input_trial_name) + 1;
        char* name_copy = new char[len];
        std::memcpy(name_copy, input_trial_name, len);

        trial_names.push_back(name_copy);
        trial_free_mem.push_back(current_free_mem);
        internal_counter++;

#ifdef DETAILED_TIMERS
        fprintf(stdout, "Memory counter %s -- appended trial %s with %zu free memory\n", counter_name, input_trial_name, current_free_mem);
#endif
    }

    // Printing function
    void print(void) {
        printf("Memory Counter: %s\n", counter_name);
        printf("Total Memory: %zu bytes\n", trial_total_mem);
        printf("---------------------------------\n");

        for (size_t i = 0; i < trial_free_mem.size(); ++i) {
            printf("[%zu] %-20s Free Memory: %zu bytes\n",
                   i,
                   trial_names[i],
                   trial_free_mem[i]);
        }

        printf("---------------------------------\n");
    }

    std::string short_print(void) {
        std::string tmp;

        tmp += "{";
        tmp += counter_name;
        tmp += ", ";
        tmp += std::to_string(trial_total_mem);

        for (size_t i = 0; i < trial_free_mem.size(); ++i) {
            tmp += ", {";
            tmp += trial_names[i];
            tmp += ", ";
            tmp += std::to_string(trial_free_mem[i]);
            tmp += "}";
        }

        tmp += "}";

        return tmp;
    }

};
