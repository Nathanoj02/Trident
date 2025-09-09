cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_ARCHITECTURES "80")
project(HnS_SpGEMM LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})
message(STATUS "CUDA include: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(include)

find_package(MPI REQUIRED COMPONENTS CXX)

set(DMMIO_ENABLE_MPI ON CACHE BOOL "Enable MPI support in dmmio")
add_subdirectory(distributed_mmio)

if(NOT TARGET ccutils)
    set(CCUTILS_ENABLE_MPI ON CACHE BOOL "Enable MPI support in ccutils")
    add_subdirectory(ccutils)
endif()

add_executable(hns_spgemm src/hns_spgemm.cu)
target_link_libraries(hns_spgemm PRIVATE MPI::MPI_CXX dmmio ccutils::ccutils ccutils::ccutils_mpi)
