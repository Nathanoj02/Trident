# CMAKE SETUP
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_ARCHITECTURES "80")
set(CMAKE_CXX_STANDARD 20)
project(HnS_SpGEMM LANGUAGES CXX CUDA)


# FIND CUDATOOLKIT
find_package(CUDAToolkit REQUIRED COMPONENTS cusparse)
message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})
message(STATUS "CUDA include: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


# FIND MPI 
find_package(MPI REQUIRED COMPONENTS CXX)


# FIND KOKKOS
#find_package(Kokkos REQUIRED)
#find_package(KokkosKernels REQUIRED)


# ADD INCLUDE DIRECTORY
include_directories(include)


# DMMIO SETUP
set(CCUTILS_ENABLE_MPI ON CACHE BOOL "Enable MPI support in ccutils")
set(CCUTILS_ENABLE_CUDA ON CACHE BOOL "Enable CUDA support in ccutils")
set(DMMIO_TOOLS ON CACHE BOOL "Enable dmmio tools")
set(DMMIO_ENABLE_MPI ON CACHE BOOL "Enable MPI support in dmmio")
add_subdirectory(distributed_mmio)


# CCUTILS SETUP
if(NOT TARGET ccutils)
    add_subdirectory(ccutils)
endif()


# CXX FLAGS
#set(CMAKE_CXX_FLAGS 
#    -O3
#    -Wall 
#    -march=native 
#    -ffast-math
#    -ftree-vectorize
#    -pthread)


# THREADS
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)


# CUDA FLAGS
set(CMAKE_CUDA_FLAGS "--extended-lambda --expt-relaxed-constexpr -O3")


# MAIN HNS_SPGEMM LIBRARY
add_library(hns_spgemm
            src/hns_spgemm.cu
            )
target_link_libraries(hns_spgemm PUBLIC
                        MPI::MPI_CXX 
                        dmmio 
                        ccutils::ccutils ccutils::ccutils_cuda ccutils::ccutils_mpi
                        Threads::Threads
                        CUDA::cusparse)


# MACRO FOR ADDING TESTS
macro(add_test name)
    add_executable(${name} test/${name}.cu)
    target_link_libraries(${name} PRIVATE 
                            MPI::MPI_CXX 
                            dmmio 
                            ccutils::ccutils ccutils::ccutils_cuda ccutils::ccutils_mpi 
                            hns_spgemm 
                            Threads::Threads 
                            CUDA::cusparse)
endmacro()


# TESTS
add_test(test_queue)
add_test(test_kokkos)
add_test(test_compression)
add_test(test_cuda_mpi)
add_test(test_p2pallgatherv)

add_executable(test_p2p_mthread test/test_p2p_mthread.cpp)
target_link_libraries(test_p2p_mthread PUBLIC OpenMP::OpenMP_CXX MPI::MPI_CXX)

# MACRO FOR ADDING OTHER EXECUTABLES
macro(add_src name)
    add_executable(${name} src/${name}.cu)
    target_link_libraries(${name} PRIVATE 
                            MPI::MPI_CXX 
                            dmmio 
                            ccutils::ccutils ccutils::ccutils_cuda ccutils::ccutils_mpi 
                            hns_spgemm 
                            Threads::Threads 
                            CUDA::cusparse)
    target_compile_options(${name} PRIVATE -O3 -Wall)
endmacro()


# EXECUTABLES
add_src(run_spgemm)
add_src(scratch)

# MACRO FOR ADDING PROFILING EXECUTABLES
macro(add_prof name folder)
    add_executable(${name}_prof ${folder}/${name}.cu)
    target_link_libraries(${name}_prof PRIVATE
                            MPI::MPI_CXX
                            dmmio
                            ccutils::ccutils ccutils::ccutils_cuda ccutils::ccutils_mpi
                            hns_spgemm
                            Threads::Threads
                            CUDA::cusparse)
    target_compile_options(${name}_prof PUBLIC -Wall -lnvToolsExt)
    target_compile_definitions(${name}_prof PUBLIC NVTX_PROFILING)
endmacro()

#PROFILING
add_prof(run_spgemm src)
add_prof(test_p2pallgatherv test)


option(ENABLE_NVTX_PROFILING "Enable NVTX profiling" OFF)

if(ENABLE_NVTX_PROFILING)
    target_compile_definitions(hns_spgemm PUBLIC NVTX_PROFILING)
endif()

