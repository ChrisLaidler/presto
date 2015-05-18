#ifndef CUDA_ACCEL_IN_INCLUDED
#define CUDA_ACCEL_IN_INCLUDED

#include <algorithm>

#include <cufft.h>
#include <cufftXt.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>

extern "C"
{
#define __float128 long double
#include "accel.h"
}

#include "cuda_utils.h"
#include "cuda_accel_utils.h"

#define LN2  0.693147180559945309417232121458176568075500134360255254120680f


/** in-place bitonic sort float array in shared memory  .
 * @param data A pointer to an shared memory array containing elements to be sorted.
 * @param arrayLength The number of elements in the array
 * @param trdId the index of the calling thread (1 thread for 2 items in data)
 * @param noThread The number of thread that are sorting this data
 * @param dir direction to sort data ( 1 -> smallest to largest AND -1 -> largest to smallest )
 *
 * This is an in-place bitonic sort.
 * This is very fast for small numbers of items, ie; when they can all fit in shared memory, or generally are less that 1K or 2K
 *
 * It has a constant performance of \f$ O\left(n\ \log^2 n \right)\f$ where n is the number of items to be sorted.
 * It only works on shared memory as it requires synchronisation.
 *
 * Each thread counts for to items in the array, as each thread performs comparisons between to elements.
 * Generally there is ~48.0 KBytes of shared memory, thus could sort up to 12288 items. However there is a
 * maximum of 1024 thread per block, thus if there are more that 2048 threads each thread must do multiple comparisons at
 * each step. These are refereed to as batches.
 *
 */
__device__ void bitonicSort(float *data, const uint arrayLength, const uint trdId, const uint noThread, const int dir = 1);

/** Calculate the median of float values  .
 *
 * @param data
 * @param arrayLength
 * @param output
 * @param noSections
 * @param median
 * @param dir
 * @return
 */
template< int bufferSz>
__device__ float cuMedianBySection(float *data, float *smBuffer, uint arrayLength, int dir = 1);

/** Calculate the median of float values  .
 *
 * @param data
 * @param arrayLength
 * @param output
 * @param noSections
 * @param median
 * @param dir
 * @return
 */
template< int bufferSz>
__device__ float cuMedian(float *smBuffer, uint arrayLength, int dir = 1);

__host__ void normAndSpread_f(cudaStream_t inpStream, cuFFdotBatch* batch, uint stack );

#endif // CUDA_ACCEL_IN_INCLUDED

