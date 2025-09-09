#ifndef COMMON_H
#define COMMON_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <type_traits>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>

#include <ccutils/colors.h>
#include "MPIType.h"

#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cusparse.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/adjacent_difference.h>
#include <thrust/merge.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/distance.h>
#include <thrust/set_operations.h>
#include <thrust/sequence.h>

//AcSpGEMM headers
//#include "CSR.h"
//#include "COO.h"
//#include "Vector.h"
//#include "dCSR.h"
//#include "dVector.h"
//#include "Multiply.h"
//#include "Transpose.h"
//#include "Compare.h"

#endif
