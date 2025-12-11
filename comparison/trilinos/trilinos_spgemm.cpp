#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
//#include <MatrixMarket_Tpetra.hpp>
#include <TpetraExt_MatrixMatrix.hpp>
#include <Kokkos_Core.hpp>

#include <ccutils/timers.h>

#include <dmmio/dmmio.h>
#include <dmmio/dio.h>
#include <dmmio/partitioning.h>

#include <cmath>
#include <vector>
#include "unistd.h"

#define MASK_SIZE 1


using scalar_t = float; // Tpetra::CrsMatrix<>::scalar_type;
using GO = long long;
using LO = int32_t;

// Use the default Kokkos device (will be CUDA if compiled with CUDA support)
using device_t = Kokkos::DefaultExecutionSpace::device_type;
using node_t = Tpetra::Map<>::node_type;

using matrix_t = Tpetra::CrsMatrix<scalar_t, LO, GO, node_t>;
using graph_t = Tpetra::CrsGraph<LO, GO, node_t>;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::Comm;
using map_t = Tpetra::Map<LO, GO, node_t>;
//using reader_t = Tpetra::MatrixMarket::Reader<matrix_t>;
using mmio_csr_t = mmio::CSR<LO, scalar_t>;
using mmio_coo_t = mmio::COO<LO, scalar_t>;
using mmio_dcoo_t = dmmio::DCOO<LO, scalar_t>;

void map_inds(mmio_dcoo_t * dcoo, int np)
{

    dcoo->coo->nrows /= np;

    for (GO i=0; i<dcoo->coo->nnz; i++)
    {
        dcoo->coo->row[i] %= dcoo->coo->nrows;
    }

}

// Verify that a local CSR matrix is correctly formed
// Returns true if valid, false otherwise. Prints detailed error info.
bool verify_local_csr(const size_t* row_ptrs, const LO* col_idx, const scalar_t* vals,
                      LO local_nrows, GO global_ncols, GO local_nnz, int rank)
{
    bool valid = true;

    // Check 1: row_ptrs[0] should be 0
    if (row_ptrs[0] != 0) {
        std::cerr << "[Rank " << rank << "] ERROR: row_ptrs[0] = " << row_ptrs[0] << ", expected 0" << std::endl;
        valid = false;
    }

    // Check 2: row_ptrs[local_nrows] should equal local_nnz
    if (row_ptrs[local_nrows] != static_cast<size_t>(local_nnz)) {
        std::cerr << "[Rank " << rank << "] ERROR: row_ptrs[" << local_nrows << "] = " << row_ptrs[local_nrows]
                  << ", expected local_nnz = " << local_nnz << std::endl;
        valid = false;
    }

    // Check 3: row_ptrs should be monotonically non-decreasing
    for (LO i = 0; i < local_nrows; i++) {
        if (row_ptrs[i] > row_ptrs[i + 1]) {
            std::cerr << "[Rank " << rank << "] ERROR: row_ptrs not monotonic at row " << i
                      << ": row_ptrs[" << i << "] = " << row_ptrs[i]
                      << " > row_ptrs[" << (i+1) << "] = " << row_ptrs[i + 1] << std::endl;
            valid = false;
            break; // Don't spam with all errors
        }
    }

    // Check 4: All column indices should be in valid range [0, global_ncols)
    size_t invalid_col_count = 0;
    LO first_invalid_row = -1;
    LO first_invalid_col = -1;
    for (LO row = 0; row < local_nrows; row++) {
        for (size_t j = row_ptrs[row]; j < row_ptrs[row + 1]; j++) {
            if (col_idx[j] < 0 || col_idx[j] >= global_ncols) {
                if (invalid_col_count == 0) {
                    first_invalid_row = row;
                    first_invalid_col = col_idx[j];
                }
                invalid_col_count++;
            }
        }
    }
    if (invalid_col_count > 0) {
        std::cerr << "[Rank " << rank << "] ERROR: " << invalid_col_count << " column indices out of range [0, "
                  << global_ncols << "). First invalid: row " << first_invalid_row
                  << ", col_idx = " << first_invalid_col << std::endl;
        valid = false;
    }

    // Check 5: Check for NaN or Inf values
    size_t nan_count = 0;
    size_t inf_count = 0;
    for (GO j = 0; j < local_nnz; j++) {
        if (std::isnan(vals[j])) nan_count++;
        if (std::isinf(vals[j])) inf_count++;
    }
    if (nan_count > 0) {
        std::cerr << "[Rank " << rank << "] WARNING: " << nan_count << " NaN values in matrix" << std::endl;
    }
    if (inf_count > 0) {
        std::cerr << "[Rank " << rank << "] WARNING: " << inf_count << " Inf values in matrix" << std::endl;
    }

    // Print summary
    std::cout << "[Rank " << rank << "] CSR verification: local_nrows=" << local_nrows
              << ", local_nnz=" << local_nnz << ", global_ncols=" << global_ncols;
    if (local_nrows > 0) {
        // Compute average nnz per row
        double avg_nnz = static_cast<double>(local_nnz) / local_nrows;
        std::cout << ", avg_nnz_per_row=" << avg_nnz;

        // Find min/max row lengths
        size_t min_row_len = row_ptrs[1] - row_ptrs[0];
        size_t max_row_len = min_row_len;
        for (LO i = 0; i < local_nrows; i++) {
            size_t row_len = row_ptrs[i + 1] - row_ptrs[i];
            min_row_len = std::min(min_row_len, row_len);
            max_row_len = std::max(max_row_len, row_len);
        }
        std::cout << ", min_row_len=" << min_row_len << ", max_row_len=" << max_row_len;
    }
    std::cout << ", valid=" << (valid ? "YES" : "NO") << std::endl;

    return valid;
}


Teuchos::RCP<matrix_t> read_fast(const char * filename, const Teuchos::RCP<const Teuchos::Comm<int>>& comm, LO ** perm_vec, bool permute=false)
{

    int np, rank;
    rank = comm->getRank();
    np = comm->getSize();

    std::cout << "permute: " << permute << std::endl;
    fflush(stdout);

    mmio::Matrix_Metadata meta;
    mmio_dcoo_t * dcoo;
    if (*perm_vec==nullptr)
    {
        dcoo = dmmio::DCOO_read<LO, scalar_t>(filename, np, rank, 1, 1, np, dmmio::PartitioningType::Naive, dmmio::Operation::None, true, &meta, MASK_SIZE, permute);
    }
    else
    {
        dcoo = dmmio::DCOO_read<LO, scalar_t>(filename, np, rank, 1, 1, np, dmmio::PartitioningType::Naive, dmmio::Operation::None, true, &meta, MASK_SIZE, permute, *perm_vec);
    }


    while ((dcoo->coo->nrows * MASK_SIZE) % np != 0)
    {
        dcoo->coo->nrows++;
    }

    if (rank==0)
    {
        std::cout << "Done with dmmio read" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    GO glob_nrows = dcoo->coo->nrows;

    if (*perm_vec == nullptr && permute)
    {
        *perm_vec = new LO[glob_nrows];
        memcpy(*perm_vec, dcoo->permutation, sizeof(LO) * glob_nrows);
    }

    map_inds(dcoo, np);

    std::cout << "Global rows: " << glob_nrows << ", local rows: " << dcoo->coo->nrows << std::endl;

    mmio_coo_t * coo = dcoo->coo;
    mmio_csr_t * csr = mmio::COO2CSR(coo);

    if (rank==0)
    {
        std::cout << "Done with csr conversion" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Convert to size_t row pointers on the host first for verification
    std::vector<size_t> row_ptrs_host(coo->nrows + 1);
    for (LO i = 0; i <= coo->nrows; i++) {
        row_ptrs_host[i] = static_cast<size_t>(csr->row_ptr[i]);
    }

    // Verify local CSR is correctly formed before passing to Tpetra
    bool csr_valid = verify_local_csr(row_ptrs_host.data(), csr->col_idx, csr->val,
                                       coo->nrows, glob_nrows, coo->nnz, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if (!csr_valid) {
        std::cerr << "[Rank " << rank << "] CSR verification failed for " << filename << std::endl;
    }

    Teuchos::RCP<const map_t> row_map = Teuchos::rcp(new map_t(glob_nrows, (LO)coo->nrows, 0, comm));
    // For SpGEMM, domain map should match the distribution (same as row_map for square matrices)
    Teuchos::RCP<const map_t> domain_map = row_map;
    Teuchos::RCP<const map_t> range_map = row_map;

    if (rank==0)
    {
        std::cout << "Done with making row maps" << std::endl;
        std::cout << "Kokkos execution space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Create matrix using row-by-row insertion (Tpetra handles device copy)
    // First, compute the max number of entries per row for allocation
    size_t maxEntriesPerRow = 0;
    for (LO i = 0; i < coo->nrows; i++) {
        size_t rowLen = row_ptrs_host[i+1] - row_ptrs_host[i];
        if (rowLen > maxEntriesPerRow) maxEntriesPerRow = rowLen;
    }

    Teuchos::RCP<matrix_t> mat = Teuchos::rcp(new matrix_t(row_map, maxEntriesPerRow));

    // Insert entries row by row
    for (LO localRow = 0; localRow < coo->nrows; localRow++) {
        size_t rowStart = row_ptrs_host[localRow];
        size_t rowEnd = row_ptrs_host[localRow + 1];
        size_t numEntries = rowEnd - rowStart;

        if (numEntries > 0) {
            // Get global row index
            GO globalRow = row_map->getGlobalElement(localRow);

            // Create views of the column indices and values for this row
            Teuchos::ArrayView<const LO> colInds(&csr->col_idx[rowStart], numEntries);
            Teuchos::ArrayView<const scalar_t> vals(&csr->val[rowStart], numEntries);

            // Convert local column indices to global
            std::vector<GO> globalColInds(numEntries);
            for (size_t j = 0; j < numEntries; j++) {
                globalColInds[j] = static_cast<GO>(csr->col_idx[rowStart + j]);
            }
            Teuchos::ArrayView<const GO> globalColIndsView(globalColInds.data(), numEntries);

            mat->insertGlobalValues(globalRow, globalColIndsView, vals);
        }
    }

    mat->fillComplete(domain_map, range_map);

    if (rank==0)
    {
        std::cout << "Done with making matrix" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    dmmio::DCOO_destroy(&dcoo);
    
    return mat;
}

//RCP<matrix_t> read_trilinos(const char * matpath, RCP<const Comm<int>>& comm)
//{
//    // First, is it a pattern matrix?
//    std::ifstream ifs;
//    ifs.open(matpath);
//    std::string banner;
//    std::getline(ifs, banner);
//    if (banner.find("pattern") != std::string::npos)
//    {
//        RCP<graph_t> A_graph = reader_t::readSparseGraphFile(matpath, comm);
//        RCP<matrix_t> A_result = rcp(new matrix_t(A_graph));
//        A_result->fillComplete();
//        A_result->setAllToScalar((SC)1.0);
//        return A_result;
//    }
//
//    return reader_t::readSparseFile(matpath, comm);
//}




// Return a pointer (RCP is like std::shared_ptr) to an output stream.
// It prints on Process 0 of the given MPI communicator, but ignores
// all output on other MPI processes.
Teuchos::RCP<Teuchos::FancyOStream> getOutputStream (const Teuchos::Comm<int>& comm) {
  using Teuchos::getFancyOStream;

  const int myRank = comm.getRank ();
  if (myRank == 0) {
    // Process 0 of the given communicator prints to std::cout.
    return getFancyOStream (Teuchos::rcpFromRef (std::cout));
  }
  else {
    // A "black hole output stream" ignores all output directed to it.
    return getFancyOStream (Teuchos::rcp (new Teuchos::oblackholestream ()));
  }
}

int main(int narg, char *arg[]) {
  // Create Tpetra scope (calls Kokkos::Initialize and Tpetra::Initialize
  Tpetra::ScopeGuard scope(&narg, &arg);
  {
    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get a default communicator:  MPI_COMM_WORLD
    const Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();

    // Output stream 'out' will ignore output not from Process 0.
    Teuchos::RCP<Teuchos::FancyOStream> pOut = getOutputStream(*comm);
    Teuchos::FancyOStream& out = *pOut;

    // Command-line options for this example:  filename, distribution, etc.
    // You can specify options any way you like
    Teuchos::CommandLineProcessor cmdp(false,true);

    std::string filenameA = "";
    cmdp.setOption("matA", &filenameA, "Path and filename of the matrix A to be read.");

    std::string filenameB = "";
    cmdp.setOption("matB", &filenameB, "Path and filename of the matrix B to be read.");

    std::string filenameC = "";
    //cmdp.setOption("matC", &filenameC, "Path and filename of the matrix C file to write.");

    std::string distribution ="1D";
    cmdp.setOption("distribution", &distribution, "Parallel distribution to use: 1D, 2D, LowerTriangularBlock, MMFile");

    bool randomize = false;
    cmdp.setOption("randomize", "norandomize", &randomize, "Randomly permute the matrix rows and columns");

    bool binary = false;  
    //cmdp.setOption("binary", "mtx", &binary, "Reading a binary file instead of a matrix market file");

    int chunkSize = 10000;
    cmdp.setOption("chunksize", &chunkSize, "Number of edges to be read and broadcasted at once");

    if (cmdp.parse(narg,arg)!=Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      return -1;
    }

    // Load the options into a Teuchos::Parameter list
    Teuchos::ParameterList params;
    params.set("distribution", distribution);
    params.set("randomize", randomize);
    params.set("binary", binary);
    params.set("chunkSize", (size_t)chunkSize);

    // Check if CUDA is the backend
    out << "Kokkos is running on: " << Kokkos::DefaultExecutionSpace().name() << std::endl;
    out << "Random permutation: " << params.get<bool>("randomize") << std::endl;


    // Call readSparseFile to read the file
    Teuchos::RCP<matrix_t> Amat, Bmat;
    try {
      //using reader_t = Tpetra::MatrixMarket::Reader<matrix_t>;
      //Amat = read_trilinos(filenameA.c_str(), comm);
      //Bmat = read_trilinos(filenameB.c_str(), comm);
      LO * perm_vec = nullptr;
      Amat = read_fast(filenameA.c_str(), comm, &perm_vec, randomize);
      Bmat = read_fast(filenameB.c_str(), comm, &perm_vec, randomize);
    } catch (std::exception &e) {
      out << ":  matrix reading failed " << filenameA << std::endl;
      out << e.what() << std::endl;
      throw e;
    }

    //if (perm_vec != nullptr) delete[] perm_vec;


    // The resulting matrix is ready to use
    Teuchos::FancyOStream foo(Teuchos::rcp(&std::cout,false));
    out << "Matrix A:" << std::endl;
    Amat->describe(foo, Teuchos::VERB_LOW);
    out << std::endl << std::endl << "Matrix B:" << std::endl;
    Bmat->describe(foo, Teuchos::VERB_LOW);
    out << std::endl << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    fflush(stdout);

    // Multiply
    CPU_TIMER_DEF(spgemm);
    for (int i=0; i<10; i++)
    {
        Teuchos::RCP<matrix_t> Cmat = Teuchos::rcp(new matrix_t(Amat->getRowMap(), 0));
        CPU_TIMER_START(spgemm);
        Tpetra::MatrixMatrix::Multiply(*Amat, false, *Bmat, false, *Cmat, true);
        MPI_Barrier(MPI_COMM_WORLD);
        CPU_TIMER_STOP(spgemm);

        if (world_rank==0)
        {
            TIMER_PRINT_LAST(spgemm);
        }
        sleep(2);
        if (i==9)
        {
            Cmat->describe(foo, Teuchos::VERB_LOW);
        }
    }

    if (world_rank==0) 
    {
        printf("Done spgemm\n");
        TIMER_PRINT(spgemm);
    }

    // Write C to MatrixMarket file
    //Tpetra::MatrixMarket::Writer<matrix_t>::writeSparseFile(filenameC, Cmat);
    //out << "Matrix C has been written to " << filenameC << std::endl;
  }

  return 0;
}
