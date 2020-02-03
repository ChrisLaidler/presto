#ifndef CUDA_ACCEL_SS_INCLUDED
#define CUDA_ACCEL_SS_INCLUDED

#include <cufft.h>
#include <algorithm>

#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"


//====================================== Constant variables  ===============================================\\

extern  __device__ const float FRAC_STAGE[16]       ;
extern  __device__ const float STP_STAGE[16]        ;
extern             const float HARM_FRAC_STAGE[16]  ;
extern  __device__ const float FRAC_HARM[16]        ;
extern  __device__ const short STAGE[5][2]          ;
extern  __device__ const short NO_HARMS[5]          ;

__host__ void sum_and_searchCU00  ( cudaStream_t stream, cuCgPlan* plan );

__host__ void sum_and_searchCU31  ( cudaStream_t stream, cuCgPlan* plan );

__host__ void cg_sum_and_search_inmem (cuCgPlan* plan );

int procesCanidate(cuCgPlan* plan, double rr, double zz, double poww, double sig, int stage, int numharm );

#endif
