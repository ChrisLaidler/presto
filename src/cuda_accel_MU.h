#include <algorithm>

#include <cufft.h>
#include <cufftXt.h>

#if CUDA_VERSION >= 7050 // Half precision
#include <cuda_fp16.h>
#endif

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

//======================================= Constant memory =================================================\\

//extern __device__ __constant__ float*       PLN_START;
//extern __device__ __constant__ uint         PLN_STRIDE;


//======================================= CUFFT callbacks =================================================\\

extern  __device__ cufftCallbackLoadC d_loadCallbackPtr;
extern  __device__ cufftCallbackStoreC d_storePow_f;
extern  __device__ cufftCallbackStoreC d_storeInmemRow;
extern  __device__ cufftCallbackStoreC d_storeInmemPln;


/** CFFT Callback function to multiply the input before the main IFFT
 *
 * @param dataIn
 * @param offset
 * @param callerInfo
 * @param sharedPtr
 * @return
 */
__device__ cufftComplex CB_MultiplyInput( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr);

/** CFFT Callback function to calculate power and save value after main IFFT
 *
 * @param dataIn
 * @param offset
 * @param callerInfo
 * @param sharedPtr
 * @return
 */
__device__ void CB_PowerOut_f( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr);

/** CFFT Callback function to calculate power and save value after main IFFT
 *
 * @param dataIn
 * @param offset
 * @param callerInfo
 * @param sharedPtr
 * @return
 */
__device__ void CB_PowerOut_h( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr);

/** CFFT Callback function to calculate power and save value to device in memory plain after main IFFT
 *
 * @param dataIn
 * @param offset
 * @param callerInfo
 * @param sharedPtr
 * @return
 */
__device__ void CB_PowerOutInmem_ROW( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr);

/** CFFT Callback function to calculate power and save value to device in memory plain after main IFFT
 *
 * @param dataIn
 * @param offset
 * @param callerInfo
 * @param sharedPtr
 * @return
 */
__device__ void CB_PowerOutInmem_PLN( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr);

/** Multiplication kernel - Just write 0 to all locations
 */
__host__  void mult00(cudaStream_t multStream, cuFFdotBatch* batch, uint stack);

//__host__  void mult02_f(cudaStream_t multStream, cuFFdotBatch* batch, uint stack);

/** Multiplication kernel - One thread per f-∂f pixel in a stack  .
 */
//__host__  void mult21_f(cudaStream_t multStream, cuFFdotBatch* batch, uint stack);

/** Multiplication kernel - One thread per f-∂f pixel
 */
//__global__ void mult1(fcomplexcu *ffdot, const int width, const int stride, const int height, const fcomplexcu *data, const fcomplexcu *kernels);


/** Multiplication kernel - All r and interlaced of Z
 */
//__global__ void mult37(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, const fcomplexcu *kernels);

/** Multiplication kernel - Each thread handles blocks of x looping down over y
 * Each thread reads one input value and loops down over the kernels
 */
//template <int noBatch >
//__global__ void mult38(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, const fcomplexcu *kernels);

/** Multiplication kernel - Multiply a stack with a kernel - multi-step  .
 * Each thread loops down a column of the plain
 * Reads the input and multiplies it with the kernel and writes result to plain
 */
//__host__ void mult11_f(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, const fcomplexcu *kernels, const fcomplexcu *datas, fcomplexcu *ffdot, const int width, const int stride, iHarmList heights, const int stackHeight, cHarmList kerDat, fCplxTex kerTex, int noSteps, int noPlns, int FLAGS );


/** Multiplication kernel - Multiply a stack with a kernel - multi-step - Use Constant memory  .
 * Each thread loops down a column of the plain
 * Reads the input and multiplies it with the kernel and writes result to plain
 */
__host__  void mult21_f(cudaStream_t multStream, cuFFdotBatch* batch, uint stack);

/** Multiplication kernel - Multiply a stack with a kernel - multi-step - Loop ( Pln - Y - step )  .
 * Each thread loops down a column of the plain
 * Reads the input and multiplies it with the kernel and writes result to plain
 */
__host__  void mult22_f(cudaStream_t multStream, cuFFdotBatch* batch, uint stack);

/** Multiplication kernel - Multiply a stack with a kernel - multi-step - Use Constant memory  .
 * Each thread loops down a column of the plain
 * Reads the input and multiplies it with the kernel and writes result to plain
 */
__host__  void mult23_f(cudaStream_t multStream, cuFFdotBatch* batch, uint stack);

__host__  void mult24(cudaStream_t multStream, cuFFdotBatch* batch, uint stack);

/** Multiplication kernel - One plain at a time
 * Each thread reads one input value and loops down over the kernels
 */
__host__  void mult10(cuFFdotBatch* batch);

/** Multiplication kernel - One thread per r location (input FFT)
 * Each thread reads one input value and loops down over the kernels
 */
__global__ void mult11(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, const fcomplexcu *kernels);

/** Multiplication kernel - One thread per r location loop down z - Texture memory
 * Each thread reads one input value and loops down over the kernels
 */
__global__ void mult12(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, fCplxTex kerTex);

/** Multiplication kernel - Multiply an entire batch with Multiplication kernel  .
 * Each thread loops down a column of the plains and multiplies input with kernel and writes result to plain
 */
__host__  void mult30_f(cudaStream_t multStream, cuFFdotBatch* batch);

/** Multiplication kernel - Multiply a multi-step stack - using a 1 plain Multiplication kernel  .
 * Split the stack into overlapping sections and read the shared kernel values once and multiply with all relevant values
 */
//__host__ void mult71_f(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, const fcomplexcu *kernel, const fcomplexcu *datas, cHarmList ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, fCplxTex kerTex, iHarmList zUp, iHarmList zDn, const uint noSteps, const uint noPlns, uint FLAGS );

/** Multiplication kernel - Multiply a multi-step stack - using a 1 plain Multiplication kernel  .
 * Split the stack into overlapping sections and read the shared kernel values once and multiply with all relevant values
 */
//__host__  void mult72_f(cudaStream_t multStream, cuFFdotBatch* batch, uint stack);

