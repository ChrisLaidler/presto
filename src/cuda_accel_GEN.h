/**   cuda_accel_gen.h
 *
 * This file contains all the common defines and specific to cuda
 *
 */

#ifndef	CUDA_ACCEL_GEN
#define	CUDA_ACCEL_GEN

#include "cuda_utils.h"
#include "candTree.h"

void createFFTPlans(cuCgPlan* plan, presto_fft_type type);

void createRvals(cuCgPlan* plan, rVals** rLev1, rVals**** rAraays );

int setConstStkInfo(stackInfo* h_inf, int noStacks,  cudaStream_t stream);

void freeRvals(cuCgPlan* plan, rVals** rLev1, rVals**** rAraays );

void freeCuAccel(cuCgInfo* mInf);

bool compare(cuCgPlan* plan, confSpecsCG* conf);

#endif	// CUDA_ACCEL_GEN
