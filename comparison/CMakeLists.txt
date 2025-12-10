

# Library setup

find_package(Trilinos REQUIRED COMPONENTS Teuchos Tpetra)

#set(COMBBLAS_INCLUDE "$ENV{COMBBLAS_DIR}/include")
#set(COMBBLAS_LIB "$ENV{COMBBLAS_DIR}/lib")


message("\nFound Trilinos! Here are the details: ")
message("   Trilinos_DIR = ${Trilinos_DIR}")
message("   Trilinos_VERSION = ${Trilinos_VERSION}")
message("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
message("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
message("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
message("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
message("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
message("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
message("End of Trilinos details\n")
set(Tpetra_DIR ${Trilinos_DIR}/../Tpetra/)
set(Teuchos_DIR ${Trilinos_DIR}/../Teuchos/)
message(${Tpetra_DIR})
find_package(Tpetra REQUIRED Tpetra)
find_package(Teuchos REQUIRED Teuchos)


set(COMMON_LIBS Tpetra::all_libs Teuchos::all_libs MPI::MPI_CXX ccutils::ccutils ccutils::ccutils_cuda ccutils::ccutils_mpi dmmio)
set(COMMON_FLAGS -O3) # -target-accel=nvidia80")

#Trilinos
add_executable(trilinos_spgemm trilinos/trilinos_spgemm.cpp)
target_link_libraries(trilinos_spgemm PRIVATE ${COMMON_LIBS})
target_compile_options(trilinos_spgemm PRIVATE -O3)

add_executable(trilinos_mcl trilinos/trilinos_mcl.cpp)
target_link_libraries(trilinos_mcl PRIVATE ${COMMON_LIBS})
target_include_directories(trilinos_mcl PRIVATE include/mcl)
target_compile_options(trilinos_mcl PRIVATE ${COMMON_FLAGS})

add_executable(trilinos_mcl_dbg trilinos/trilinos_mcl.cpp)
target_link_libraries(trilinos_mcl_dbg PRIVATE ${COMMON_LIBS})
target_include_directories(trilinos_mcl_dbg PRIVATE include/mcl)
target_compile_options(trilinos_mcl_dbg PRIVATE ${COMMON_FLAGS} -DDEBUG=1)



# Combblas
#add_executable(combblas_spgemm combblas/combblas_spgemm.cu)
#target_link_directories(combblas_spgemm PRIVATE ${COMBBLAS_LIB} )
#target_link_libraries(combblas_spgemm PRIVATE -lCombBLAS ccutils::ccutils_cuda ccutils::ccutils_mpi)
#target_include_directories(combblas_spgemm PRIVATE ${COMBBLAS_INCLUDE} $ENV{COMBBLAS_DIR}/../gspgemm $ENV{COMBBLAS_DIR}/../gspgemm/nsparse $ENV{COMBBLAS_DIR}/../gspgemm/rmerge $ENV{COMBBLAS_DIR}/../gspgemm/rmerge/SPMMProject $ENV{COMBBLAS_DIR}/../gspgemm/cusplibrary)
#target_compile_options(combblas_spgemm PRIVATE ${COMMON_FLAGS} -fopenmp)




