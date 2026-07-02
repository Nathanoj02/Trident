# CMAKE SETUP
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_ARCHITECTURES "80;90")
set(CMAKE_CXX_STANDARD 20)
project(HnS_SpGEMM LANGUAGES CXX CUDA)
set(CMAKE_CUDA_FLAGS 
"-ccbin ${CMAKE_CXX_COMPILER} \
-Wno-deprecated-declarations \
-Xcompiler -O3 \
-Xcompiler -fopenmp \
--extended-lambda --expt-relaxed-constexpr -O3")


# FIND CUDATOOLKIT
find_package(CUDAToolkit REQUIRED COMPONENTS cusparse)
message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})
message(STATUS "CUDA include: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# FIND MPI 
find_package(MPI REQUIRED COMPONENTS CXX)

# FIND KOKKOS
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)

# NCCL: set via env NCCL_ROOT or -DNCCL_ROOT=...
if(NOT NCCL_ROOT)
    if(DEFINED ENV{NCCL_ROOT})
        set(NCCL_ROOT "$ENV{NCCL_ROOT}")
    else()
        message(FATAL_ERROR "NCCL_ROOT not set. Pass -DNCCL_ROOT=<path> or export NCCL_ROOT (dir with include/nccl.h + lib/libnccl.so).")
    endif()
endif()

set(NCCL_INCLUDE_DIR "${NCCL_ROOT}/include")
set(NCCL_LIBRARY "${NCCL_ROOT}/lib/libnccl.so")

if(NOT EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
    message(FATAL_ERROR "NCCL header not found: ${NCCL_INCLUDE_DIR}/nccl.h")
endif()

if(NOT EXISTS "${NCCL_LIBRARY}")
    message(FATAL_ERROR "NCCL library not found: ${NCCL_LIBRARY}")
endif()

message(STATUS "Using NCCL include: ${NCCL_INCLUDE_DIR}")
message(STATUS "Using NCCL library: ${NCCL_LIBRARY}")

add_library(NCCL::NCCL INTERFACE IMPORTED)
set_target_properties(NCCL::NCCL PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${NCCL_LIBRARY}"
)

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
#set(CMAKE_CUDA_FLAGS "--extended-lambda --expt-relaxed-constexpr -O3")
#set(CMAKE_CUDA_FLAGS "--extended-lambda --expt-relaxed-constexpr")


# MAIN HNS_SPGEMM LIBRARIES
set(SOURCES src/hns_spgemm.cu 
             src/sparse_summa.cu 
             src/hns_spgemm_1d.cu)
add_library(hns_spgemm_accumthread ${SOURCES})
target_link_libraries(hns_spgemm_accumthread PUBLIC
                      MPI::MPI_CXX 
                      dmmio 
                      ccutils::ccutils ccutils::ccutils_cuda ccutils::ccutils_mpi
                      KokkosKernels::kokkoskernels Kokkos::kokkos
                      Threads::Threads
                      CUDA::cusparse
                      NCCL::NCCL)
target_compile_definitions(hns_spgemm_accumthread PUBLIC ACCUM_THREAD)


add_library(hns_spgemm ${SOURCES})
target_link_libraries(hns_spgemm PUBLIC
                      MPI::MPI_CXX 
                      dmmio 
                      ccutils::ccutils ccutils::ccutils_cuda ccutils::ccutils_mpi
                      KokkosKernels::kokkoskernels Kokkos::kokkos
                      Threads::Threads
                      CUDA::cusparse
                      NCCL::NCCL)


find_library(NVTX_LIBRARY
    NAMES nvToolsExt
    PATHS ${CUDAToolkit_LIBRARY_DIR} ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
          /usr/local/cuda/lib64 /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/lib64
)
if(NVTX_LIBRARY)
    message(STATUS "Found NVTX: ${NVTX_LIBRARY}")
else()
    message(STATUS "NVTX compiled lib not found (CUDA 12 NVTX3 header-only) — building without it")
    set(NVTX_LIBRARY "" CACHE STRING "" FORCE)
endif()


# MACRO FOR ADDING OTHER EXECUTABLES
macro(add_src src_name exec_name hns_libname)
    add_executable(${exec_name} src/${src_name}.cu)
    target_link_libraries(${exec_name} PRIVATE
    		MPI::MPI_CXX
    		dmmio
    		ccutils::ccutils ccutils::ccutils_cuda ccutils::ccutils_mpi
            KokkosKernels::kokkoskernels Kokkos::kokkos
            ${hns_libname}
    		Threads::Threads
            CUDA::cusparse
            NCCL::NCCL
    		${NVTX_LIBRARY}
	)
    target_compile_options(${exec_name} PRIVATE -O3 -Wall)
endmacro()


# EXECUTABLES
add_src(run_spgemm run_spgemm hns_spgemm)
add_src(run_spgemm_incremental run_spgemm_incremental hns_spgemm)
add_src(mcl mcl hns_spgemm)
add_src(run_spgemm run_spgemm_accumthread hns_spgemm_accumthread)

option(ENABLE_NVTX_PROFILING "Enable NVTX profiling" OFF)

if(ENABLE_NVTX_PROFILING)
    target_compile_definitions(hns_spgemm_deb PUBLIC NVTX_PROFILING)
endif()


# Trilinos/CombBLAS baselines - off by default (need extra deps; not required for run_spgemm)
option(BUILD_COMPARISON "Build comparison baselines (Trilinos, CombBLAS)" OFF)
if(BUILD_COMPARISON)
    add_subdirectory(comparison)
endif()

