#include <algorithm>

#include <cufft.h>
#include <cufftXt.h>

#if CUDART_VERSION >= 7050   // Half precision header  .
#include <cuda_fp16.h>
#endif

#include <thrust/sort.h>
#include <thrust/device_vector.h>

extern "C"
{
#include "accel.h"
}

#include "cuda_utils.h"
#include "cuda_accel_utils.h"

//=========================================== Defines =====================================================\\

#define CPY_WIDTH 512  // 256  512 768

/** Multiplication kernel - Just write 0 to all locations
 */
__host__  void mult00(cudaStream_t multStream, cuCgPlan* plan, cuFfdotStack* cStack);

/** Multiplication kernel - All planes of a stack individually - One thread per r location (input FFT)  .
 * Each thread reads one input value and loops down over the kernels
 */
__host__  void mult11(cudaStream_t multStream, cuCgPlan* plan, cuFfdotStack* cStack);

/** Multiplication kernel - Multiply a stack with a kernel - multi-segment - Use Constant memory  .
 * Each thread loops down a column of the plane
 * Reads the input and multiplies it with the kernel and writes result to plane
 */
__host__  void mult21(cudaStream_t multStream, cuCgPlan* plan, cuFfdotStack* cStack);

/** Multiplication kernel - Multiply a stack with a kernel - multi-segment - Loop ( Pln - Y - segment )  .
 * Each thread loops down a column of the plane
 * Reads the input and multiplies it with the kernel and writes result to plane
 */
__host__  void mult22(cudaStream_t multStream, cuCgPlan* plan, cuFfdotStack* cStack);

/** Multiplication kernel - Multiply a stack with a kernel - multi-segment - Use Constant memory  .
 * Each thread loops down a column of the plane
 * Reads the input and multiplies it with the kernel and writes result to plane
 */
__host__  void mult23(cudaStream_t multStream, cuCgPlan* plan, cuFfdotStack* cStack);

/** DEPRICTAED this is a testing multiplication function
 *
 * @param multStream
 * @param plan
 * @param stack
 */
__host__  void mult24(cudaStream_t multStream, cuCgPlan* plan, cuFfdotStack* cStack);

/** Multiplication kernel - Multiply an entire plan with Multiplication kernel  .
 * Each thread loops down a column of the planes and multiplies input with kernel and writes result to plane
 */
__host__  void mult31(cudaStream_t multStream, cuCgPlan* plan);
