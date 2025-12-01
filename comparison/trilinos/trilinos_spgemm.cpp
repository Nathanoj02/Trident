#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <MatrixMarket_Tpetra.hpp>
#include <TpetraExt_MatrixMatrix.hpp>

#include <ccutils/timers.h>

//#include <dmmio/dmmio.h>
//#include <dmmio/dio.h>
//#include <dmmio/partitioning.h>

#include "unistd.h"

#define MASK_SIZE 8


using scalar_t = float; // Tpetra::CrsMatrix<>::scalar_type;
using matrix_t = Tpetra::CrsMatrix<float, int32_t, long long>;
using graph_t = Tpetra::CrsGraph<int32_t, long long>;
using GO = long long;
using SC = float;
using LO = int32_t;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::Comm;
using reader_t = Tpetra::MatrixMarket::Reader<matrix_t>;
//using mmio_csr_t = mmio::CSR<int32_t, scalar_t>;
//using mmio_coo_t = mmio::COO<int32_t, scalar_t>;
//using mmio_dcoo_t = dmmio::DCOO<int32_t, scalar_t>;

//Teuchos::RCP<matrix_t> read_fast(const char * filename, const Teuchos::Comm<int>& comm)
//{
//
//    int np, rank;
//    rank = comm.getRank();
//    np = comm.getSize();
//
//
//    mmio_dcoo_t * dcoo = dmmio::DCOO_read(filename, np, rank, np, 1, 1, dmmio::PartitioningType::Naive, dmmio::Operation::None, true, nullptr, MASK_SIZE);
//
//    mmio_coo_t * coo = dcoo->coo;
//    mmio_csr_t * csr = mmio::COO2CSR(coo);
//
//    matrix_t mat(
//
//    Teuchos::RCP<matrix_t> mat;
//    
//}

RCP<matrix_t> read_trilinos(const char * matpath, RCP<const Comm<int>>& comm)
{
    // First, is it a pattern matrix?
    std::ifstream ifs;
    ifs.open(matpath);
    std::string banner;
    std::getline(ifs, banner);
    if (banner.find("pattern") != std::string::npos)
    {
        RCP<graph_t> A_graph = reader_t::readSparseGraphFile(matpath, comm);
        RCP<matrix_t> A_result = rcp(new matrix_t(A_graph));
        A_result->fillComplete();
        A_result->setAllToScalar((SC)1.0);
        return A_result;
    }

    return reader_t::readSparseFile(matpath, comm);
}




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
    Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();

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
    //cmdp.setOption("randomize", "norandomize", &randomize, "Randomly permute the matrix rows and columns");

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

    // Call readSparseFile to read the file
    using scalar_t = float; // Tpetra::CrsMatrix<>::scalar_type;
    using matrix_t = Tpetra::CrsMatrix<scalar_t>;
    Teuchos::RCP<matrix_t> Amat, Bmat;
    try {
      //using reader_t = Tpetra::MatrixMarket::Reader<matrix_t>;
      Amat = read_trilinos(filenameA.c_str(), comm);
      Bmat = read_trilinos(filenameB.c_str(), comm);
    } catch (std::exception &e) {
      out << ":  matrix reading failed " << filenameA << std::endl;
      out << e.what() << std::endl;
      throw e;
    }

    // The resulting matrix is ready to use
    Teuchos::FancyOStream foo(Teuchos::rcp(&std::cout,false));
    out << "Matrix A:" << std::endl;
    Amat->describe(foo, Teuchos::VERB_LOW);
    out << std::endl << std::endl << "Matrix B:" << std::endl;
    Bmat->describe(foo, Teuchos::VERB_LOW);
    out << std::endl << std::endl;

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
    }

    if (world_rank==0) 
    {
        printf("Done spgemm\n");
        TIMER_PRINT(spgemm);
    }

    //out << "Matrix C=AxB:" << std::endl;
    //Cmat->describe(foo, Teuchos::VERB_HIGH);

    // Write C to MatrixMarket file
    //Tpetra::MatrixMarket::Writer<matrix_t>::writeSparseFile(filenameC, Cmat);
    //out << "Matrix C has been written to " << filenameC << std::endl;
  }

  return 0;
}
