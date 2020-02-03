#include "cuda_accel.h"

#ifndef	CUDA_ACCEL_FFT
#define	CUDA_ACCEL_FFT

/**  iFFT a specific stack  .
 *
 * @param plan
 * @param cStack
 * @param pStack
 */
void iFFT_Stack(cuCgPlan* plan, cuFfdotStack* cStack, cuFfdotStack* pStack = NULL);


#endif	// CUDA_ACCEL_FFT
