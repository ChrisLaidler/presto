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

//======================================= CUFFT callbacks =================================================\\

extern  __device__ cufftCallbackLoadC d_loadCallbackPtr;
extern  __device__ cufftCallbackStoreC d_storeCallbackPtr;

/** CFFT Callback function to convolve the input before the main IFFT
 *
 * @param dataIn
 * @param offset
 * @param callerInfo
 * @param sharedPtr
 * @return
 */
__device__ cufftComplex CB_ConvolveInput( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr);

/** CFFT Callback function to calculate power and save value after main IFFT
 *
 * @param dataIn
 * @param offset
 * @param callerInfo
 * @param sharedPtr
 * @return
 */
__device__ void CB_PowerOut( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr);

/** Convolution kernel - Just write 0 to all locations
 */
__host__  void convolveffdot00_f(cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack);

__host__  void convolveffdot02_f(cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack);

/** Convolution kernel - One thread per f-∂f pixel in a stack  .
 */
__host__  void convolveffdot10_f(cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack);

/** Convolution kernel - One thread per f-∂f pixel
 */
__global__ void convolveffdot1(fcomplexcu *ffdot, const int width, const int stride, const int height, const fcomplexcu *data, const fcomplexcu *kernels);

/** Convolution kernel - One thread per r location (input FFT)
 * Each thread reads one input value and loops down over the kernels
 */
__global__ void convolveffdot31(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, const fcomplexcu *kernels);

/** Convolution kernel - One thread per r location loop down z - Texture memory
 * Each thread reads one input value and loops down over the kernels
 */
__global__ void convolveffdot36(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, fCplxTex kerTex);

/** Convolution kernel - All r and interlaced of Z
 */
__global__ void convolveffdot37(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, const fcomplexcu *kernels);

/** Convolution kernel - Each thread handles blocks of x looping down over y
 * Each thread reads one input value and loops down over the kernels
 */
template <int noBatch >
__global__ void convolveffdot38(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, const fcomplexcu *kernels);

/** Convolution kernel - Convolve a stack with a kernel - multi-step  .
 * Each thread loops down a column of the plain
 * Reads the input and convolves it with the kernel and writes result to plain
 */
//__host__ void convolveffdot41_f(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, const fcomplexcu *kernels, const fcomplexcu *datas, fcomplexcu *ffdot, const int width, const int stride, iHarmList heights, const int stackHeight, cHarmList kerDat, fCplxTex kerTex, int noSteps, int noPlns, int FLAGS );


/** Convolution kernel - Convolve a stack with a kernel - multi-step - Loop ( Pln - Y - step )  .
 * Each thread loops down a column of the plain
 * Reads the input and convolves it with the kernel and writes result to plain
 */
__host__  void convolveffdot41_f(cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack);

/** Convolution kernel - Convolve a stack with a kernel - multi-step - Use Constant memory  .
 * Each thread loops down a column of the plain
 * Reads the input and convolves it with the kernel and writes result to plain
 */
__host__  void convolveffdot42_f(cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack);

/** Convolution kernel - Convolve a stack with a kernel - multi-step - Use Constant memory  .
 * Each thread loops down a column of the plain
 * Reads the input and convolves it with the kernel and writes result to plain
 */
__host__  void convolveffdot43_f(cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack);

/** Convolution kernel - Convolve an entire batch with convolution kernel  .
 * Each thread loops down a column of the plains and convolves input with kernel and writes result to plain
 */
__host__  void convolveffdot50_f(cudaStream_t cnvlStream, cuFFdotBatch* batch);

/** Convolution kernel - Convolve a multi-step stack - using a 1 plain convolution kernel  .
 * Split the stack into overlapping sections and read the shared kernel values once and convolve with all relevant values
 */
__host__ void convolveffdot71_f(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, const fcomplexcu *kernel, const fcomplexcu *datas, cHarmList ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, fCplxTex kerTex, iHarmList zUp, iHarmList zDn, const uint noSteps, const uint noPlns, uint FLAGS );

/** Convolution kernel - Convolve a multi-step stack - using a 1 plain convolution kernel  .
 * Split the stack into overlapping sections and read the shared kernel values once and convolve with all relevant values
 */
__host__  void convolveffdot72_f(cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack);

