#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <MatrixMarket_Tpetra.hpp>
#include <TpetraExt_MatrixMatrix.hpp>

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
    // Get a default communicator:  MPI_COMM_WORLD
    const Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();

    // Output stream 'out' will ignore output not from Process 0.
    Teuchos::RCP<Teuchos::FancyOStream> pOut = getOutputStream(*comm);
    Teuchos::FancyOStream& out = *pOut;

    // Command-line options for this example:  filename, distribution, etc.
    // You can specify options any way you like
    Teuchos::CommandLineProcessor cmdp(false,true);

    std::string filenameA = "";
    cmdp.setOption("mtxA", &filenameA, "Path and filename of the matrix A to be read.");

    std::string filenameB = "";
    cmdp.setOption("mtxB", &filenameB, "Path and filename of the matrix B to be read.");

    std::string filenameC = "";
    cmdp.setOption("mtxC", &filenameC, "Path and filename of the matrix C file to write.");

    std::string distribution ="1D";
    cmdp.setOption("distribution", &distribution, "Parallel distribution to use: 1D, 2D, LowerTriangularBlock, MMFile");

    bool randomize = false;
    cmdp.setOption("randomize", "norandomize", &randomize, "Randomly permute the matrix rows and columns");

    bool binary = false;  
    cmdp.setOption("binary", "mtx", &binary, "Reading a binary file instead of a matrix market file");

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
      using reader_t = Tpetra::MatrixMarket::Reader<matrix_t>;
      Amat = reader_t::readSparseFile(filenameA, comm, params);
      Bmat = reader_t::readSparseFile(filenameB, comm, params);
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
    Teuchos::RCP<matrix_t> Cmat = Teuchos::rcp(new matrix_t(Amat->getRowMap(), 0));
    Tpetra::MatrixMatrix::Multiply(*Amat, false, *Bmat, false, *Cmat, true);

    out << "Matrix C=AxB:" << std::endl;
    Cmat->describe(foo, Teuchos::VERB_HIGH);

    // Write C to MatrixMarket file
    Tpetra::MatrixMarket::Writer<matrix_t>::writeSparseFile(filenameC, Cmat);
    out << "Matrix C has been written to " << filenameC << std::endl;
  }

  return 0;
}