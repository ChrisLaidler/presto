#include <algorithm>

#include <cufft.h>
#include <cufftXt.h>

#if CUDA_VERSION >= 7050   // Half precision header  .
#include <cuda_fp16.h>
#endif

#include <thrust/sort.h>
#include <thrust/device_vector.h>

extern "C"
{
#define __float128 long double
#include "accel.h"
}

#include "cuda_utils.h"
#include "cuda_accel_utils.h"

//=========================================== Defines =====================================================\\

#define CPY_WIDTH 512  // 256  512 768

#if CUDA_VERSION >= 6050  // CUFFT callbacks type defines

extern  __device__ cufftCallbackLoadC d_loadConst;
extern  __device__ cufftCallbackLoadC d_loadRead;
extern  __device__ cufftCallbackLoadC d_loadInp;
extern  __device__ cufftCallbackLoadC d_loadInp0;
extern  __device__ cufftCallbackLoadC d_loadInp1;
extern  __device__ cufftCallbackLoadC d_loadInp2;
extern  __device__ cufftCallbackLoadC d_loadInp3;
extern  __device__ cufftCallbackLoadC d_loadCallbackPtr;

extern  __device__ cufftCallbackStoreC d_storePow_f;
extern  __device__ cufftCallbackStoreC d_storeInmemRow;
extern  __device__ cufftCallbackStoreC d_storeInmemPln;

#endif

//======================================= Constant memory =================================================\\


//======================================= CUFFT callbacks =================================================\\


#if CUDA_VERSION >= 6050 // CUFFT callbacks, only implemented in CUDA 6.5

/** CUFFT callback kernel to simply return constant value  .
 */
__device__ cufftComplex CB_RetConst( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr);

/** CUFFT callback kernel to simply read data  .
 */
__device__ cufftComplex CB_RetValue( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr);

/** CUFFT callback kernel to simply read data  .
 */
__device__ cufftComplex CB_readInp( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr);

/** CUFFT callback kernel to simply read data  .
 */
__device__ cufftComplex CB_readInp0( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr);

/** CUFFT callback kernel to simply read data  .
 */
__device__ cufftComplex CB_readInp1( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr);

/** CUFFT callback kernel to simply read data  .
 */
__device__ cufftComplex CB_readInp2( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr);

/** CUFFT callback kernel to simply read data  .
 */
__device__ cufftComplex CB_readInp3( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr);

/** CFFT Callback function to multiply the input before the main IFFT  .
 *
 * @param dataIn
 * @param offset
 * @param callerInfo
 * @param sharedPtr
 * @return
 */
__device__ cufftComplex CB_MultiplyInput( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr);

/** CFFT Callback function to calculate power and save value after main IFFT  .
 *
 * @param dataIn
 * @param offset
 * @param callerInfo
 * @param sharedPtr
 * @return
 */
__device__ void CB_PowerOut_f( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr);

/** CUFFT callback kernel to calculate and store float powers in the in-memory plane, after the FFT  .
 *  CallerInfo is passed as the address of the first element of the first step in the inmem plane
 *
 */
__device__ void CB_InmemOutRow_f( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr);

/** CUFFT callback kernel to calculate and store float powers in the in-memory plane, after the FFT  .
 *  CallerInfo is passed as the address of the first element of the first step in the inmem plane
 *
 */
__device__ void CB_InmemOutPln_f( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr);

/** CFFT Callback function to calculate power and save value after main IFFT  .
 *
 * @param dataIn
 * @param offset
 * @param callerInfo
 * @param sharedPtr
 * @return
 */
__device__ void CB_PowerOut_h( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr);

/** CUFFT callback kernel to calculate and store half powers in the in-memory plane, after the FFT  .
 *  CallerInfo is passed as the address of the first element of the first step in the inmem plane
 *  Assumes row interleaved data
 *
 */
__device__ void CB_InmemOutRow_h( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr);

/** CUFFT callback kernel to calculate and store half powers in the in-memory plane, after the FFT  .
 *  CallerInfo is passed as the address of the first element of the first step in the inmem plane
 *  Assumes plane interleaved data
 *
 */
__device__ void CB_InmemOutPln_h( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr);

/** CFFT Callback function to calculate power and save value to device in memory plane after main IFFT  .
 *
 * @param dataIn
 * @param offset
 * @param callerInfo
 * @param sharedPtr
 * @return
 */
__device__ void CB_PowerOutInmem_ROW( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr);

/** CFFT Callback function to calculate power and save value to device in memory plane after main IFFT  .
 *
 * @param dataIn
 * @param offset
 * @param callerInfo
 * @param sharedPtr
 * @return
 */
__device__ void CB_PowerOutInmem_PLN( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr);

#endif

/** Get a pointer to the location of the first element of the output of the CUFFT  .
 *
 */
void* getCBwriteLocation(cuFFdotBatch* batch, cuFfdotStack* cStack);

/** Set CUFFT store FFT callback  .
 *
 */
void setCB(cuFFdotBatch* batch, cuFfdotStack* cStack);

/** Multiplication kernel - Just write 0 to all locations
 */
__host__  void mult00(cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack);

/** Multiplication kernel - All planes of a stack indevidually - One thread per r location (input FFT)  .
 * Each thread reads one input value and loops down over the kernels
 */
__host__  void mult11(cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack);

/** Multiplication kernel - Multiply a stack with a kernel - multi-step - Use Constant memory  .
 * Each thread loops down a column of the plane
 * Reads the input and multiplies it with the kernel and writes result to plane
 */
__host__  void mult21(cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack);

/** Multiplication kernel - Multiply a stack with a kernel - multi-step - Loop ( Pln - Y - step )  .
 * Each thread loops down a column of the plane
 * Reads the input and multiplies it with the kernel and writes result to plane
 */
__host__  void mult22(cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack);

/** Multiplication kernel - Multiply a stack with a kernel - multi-step - Use Constant memory  .
 * Each thread loops down a column of the plane
 * Reads the input and multiplies it with the kernel and writes result to plane
 */
__host__  void mult23(cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack);

/** DEPRICTAED this is a testing multiplication function
 *
 * @param multStream
 * @param batch
 * @param stack
 */
__host__  void mult24(cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack);

/** Multiplication kernel - Multiply an entire batch with Multiplication kernel  .
 * Each thread loops down a column of the planes and multiplies input with kernel and writes result to plane
 */
__host__  void mult31(cudaStream_t multStream, cuFFdotBatch* batch);
