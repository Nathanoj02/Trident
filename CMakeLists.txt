# CMAKE SETUP
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_ARCHITECTURES "80")
set(CMAKE_CXX_STANDARD 20)
project(HnS_SpGEMM LANGUAGES CXX CUDA)



# FIND CUDATOOLKIT
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})
message(STATUS "CUDA include: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


# FIND MPI 
find_package(MPI REQUIRED COMPONENTS CXX)


# FIND KOKKOS
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)



# ADD INCLUDE DIRECTORY
include_directories(include)


# DMMIO SETUP
set(DMMIO_TOOLS ON CACHE BOOL "Enable dmmio tools")
set(DMMIO_ENABLE_MPI ON CACHE BOOL "Enable MPI support in dmmio")
add_subdirectory(distributed_mmio)


# CCUTILS SETUP
if(NOT TARGET ccutils)
    set(CCUTILS_ENABLE_MPI ON CACHE BOOL "Enable MPI support in ccutils")
    set(CCUTILS_ENABLE_CUDA ON CACHE BOOL "Enable CUDA support in ccutils")
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


# CUDA FLAGS
set(CMAKE_CUDA_FLAGS "--extended-lambda --expt-relaxed-constexpr -ccbin=CC -O3")


# MAIN HNS_SPGEMM LIBRARY
add_library(hns_spgemm
            src/hns_spgemm.cu)
target_link_libraries(hns_spgemm PUBLIC MPI::MPI_CXX dmmio ccutils::ccutils ccutils::ccutils_mpi KokkosKernels::kokkoskernels Kokkos::kokkos Threads::Threads CUDA::cusparse)


# MACRO FOR ADDING TESTS
macro(add_test name)
    add_executable(${name} test/${name}.cu)
    target_link_libraries(${name} PRIVATE MPI::MPI_CXX dmmio ccutils::ccutils ccutils::ccutils_mpi hns_spgemm KokkosKernels::kokkoskernels Kokkos::kokkos Threads::Threads CUDA::cusparse)
endmacro()


# TESTS
add_test(test_queue)
add_test(test_kokkos)


# MACRO FOR ADDING OTHER EXECUTABLES
macro(add_src name)
    add_executable(${name} src/${name}.cu)
    target_link_libraries(${name} PRIVATE MPI::MPI_CXX dmmio ccutils::ccutils ccutils::ccutils_mpi hns_spgemm KokkosKernels::kokkoskernels Kokkos::kokkos Threads::Threads CUDA::cusparse)
    #target_compile_options(${name} PRIVATE -O3 -Wall)
endmacro()


# EXECUTABLES
add_src(run_spgemm)
add_src(scratch)
