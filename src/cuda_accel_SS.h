#ifndef CUDA_ACCEL_SS_INCLUDED
#define CUDA_ACCEL_SS_INCLUDED

#include <cufft.h>
#include <algorithm>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>

#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"

//====================================== Constant variables  ===============================================\\

extern __device__ const float FRAC_STAGE[16]    ;
extern            const float h_FRAC_STAGE[16]  ;
extern __device__ const float FRAC_HARM[16]     ;
extern __device__ const int STAGE[5][2]         ; //     =  { {0,0}, {1,1}, {2,3}, {4,7}, {8,15} } ;
extern __device__ const int CHUNKSZE[5]         ; //     =  { 4, 8, 8, 8, 8 } ;


__host__ __device__ double candidate_sigma_cu(double poww, int numharm, long long numindep);

__host__ void add_and_searchCU00  ( cudaStream_t stream, cuFFdotBatch* batch );

__host__ void add_and_searchCU31  ( cudaStream_t stream, cuFFdotBatch* batch );

__host__ void add_and_searchCU32  ( cudaStream_t stream, cuFFdotBatch* batch );

__host__ void add_and_searchCU33  ( cudaStream_t stream, cuFFdotBatch* batch );

__host__ void add_and_searchCU3_PT_f ( cudaStream_t stream, cuFFdotBatch* batch );

__host__ void add_and_maxCU31_f   ( dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream,cuSearchList searchList, float* d_cands, uint* d_sem, int base, float* rLows, int noSteps, const uint noStages, uint FLAGS );

template<int noStages, int canMethoud> __global__ void add_and_searchCU4(cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base);

#endif
