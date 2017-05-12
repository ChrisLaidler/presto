/**   cuda_accel_gen.h
 *
 * This file contains all the common defines and specific to cuda
 *
 */

#ifndef	CUDA_ACCEL_GEN
#define	CUDA_ACCEL_GEN

#include "cuda_utils.h"
#include "candTree.h"

void createFFTPlans(cuFFdotBatch* batch, presto_fft_type type);

void createRvals(cuFFdotBatch* batch, rVals** rLev1, rVals**** rAraays );

int setConstStkInfo(stackInfo* h_inf, int noStacks,  cudaStream_t stream);

void freeRvals(cuFFdotBatch* batch, rVals** rLev1, rVals**** rAraays );

void freeCuAccel(cuPlnInfo* mInf);

bool compare(cuFFdotBatch* batch, confSpecsGen* conf);

#endif	// CUDA_ACCEL_GEN
