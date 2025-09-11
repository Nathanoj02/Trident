cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_ARCHITECTURES "80")
project(HnS_SpGEMM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_compile_definitions(CCUTILS_ENABLE_CUDA)

# FIND CUDATOOLKIT
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})
message(STATUS "CUDA include: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


# FIND MPI 
find_package(MPI REQUIRED COMPONENTS CXX)


# ADD INCLUDE DIRECTORY
include_directories(include)

# DMMIO SETUP
set(DMMIO_ENABLE_MPI ON CACHE BOOL "Enable MPI support in dmmio")
add_subdirectory(distributed_mmio)


# CCUTILS SETUP
if(NOT TARGET ccutils)
    set(CCUTILS_ENABLE_MPI ON CACHE BOOL "Enable MPI support in ccutils")
    set(CCUTILS_ENABLE_CUDA ON CACHE BOOL "Enable CUDA support in ccutils")
    add_subdirectory(ccutils)
endif()


find_package(Kokkos REQUIRED)


# CXX FLAGS
set(CXX_FLAGS 
    -O3
    -Wall 
    -march=native 
    -ffast-math
    -ftree-vectorize)

# CUDA FLAGS
set(CMAKE_CUDA_FLAGS "--extended-lambda --expt-relaxed-constexpr -ccbin=CC")


# MACRO FOR ADDING TESTS
macro(add_test name)
    add_executable(${name} test/${name}.cu)
    target_link_libraries(${name} PRIVATE MPI::MPI_CXX dmmio ccutils::ccutils ccutils::ccutils_mpi)
endmacro()


# Keep this here for now since it is in src
add_executable(hns_spgemm src/hns_spgemm.cu)
target_link_libraries(hns_spgemm PRIVATE MPI::MPI_CXX dmmio ccutils::ccutils ccutils::ccutils_mpi Kokkos::kokkos)

# Move to test if we want to add kokkos to all the tests
add_executable(test_kokkos test/test_kokkos.cu)
target_link_libraries(test_kokkos PRIVATE MPI::MPI_CXX dmmio ccutils::ccutils ccutils::ccutils_mpi Kokkos::kokkos)

add_test(test_stack)
