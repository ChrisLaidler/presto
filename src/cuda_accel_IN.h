#ifndef CUDA_ACCEL_IN_INCLUDED
#define CUDA_ACCEL_IN_INCLUDED

#include <algorithm>

#include <cufft.h>
#include <cufftXt.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>

extern "C"
{
#define __float128 long double
#include "accel.h"
}

#include "cuda_utils.h"
#include "cuda_accel_utils.h"

#define LN2  0.693147180559945309417232121458176568075500134360255254120680


extern int    cuMedianBuffSz;

__host__ void normAndSpread(cudaStream_t inpStream, cuFFdotBatch* batch, uint stack );

#endif // CUDA_ACCEL_IN_INCLUDED

