#ifndef COMMON_H
#define COMMON_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>

#include <cstdint>
#include <memory>
#include <iostream>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <numeric>
#include <thread>

#include <ccutils/colors.h>
#include <ccutils/mpi/mpi_macros.h>
#include <ccutils/macros.h>
#include <ccutils/cuda/cuda_macros.h>
#include <ccutils/cuda/cuda_utils.hpp>

#include <mmio/io.h>
#include <mmio/mmio.h>
#include <mmio/utils.h>
#include <dmmio/dio.h>
#include <dmmio/dmmio.h>
#include <dmmio/dutils.h>
#include <dmmio/partitioning.h>

#include "MPIType.h"

#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cusparse.h>

#include <cub/cub.cuh>

#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spgemm.hpp>
#include <KokkosSparse_spadd.hpp>


#endif
