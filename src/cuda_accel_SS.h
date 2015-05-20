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

//======================================= Constant memory =================================================\\

extern __device__ __constant__ int        YINDS[MAX_YINDS];             ///<
extern __device__ __constant__ float      YINDS_F[MAX_YINDS];           ///<
extern __device__ __constant__ float      POWERCUT[MAX_HARM_NO];        ///<
extern __device__ __constant__ float      NUMINDEP[MAX_HARM_NO];        ///<

extern __device__ __constant__ int        HEIGHT[MAX_HARM_NO];          ///< Plain heights in stage order
extern __device__ __constant__ int        STRIDE[MAX_HARM_NO];          ///< Plain strides in stage order
extern __device__ __constant__ int        HWIDTH[MAX_HARM_NO];          ///< Plain half width in stage order

//====================================== Constant variables  ===============================================\\

extern __device__ const float FRAC[16]  ; //     =  {1.0f, 0.5f, 0.25f, 0.75f, 0.125f, 0.375f, 0.625f, 0.875f, 0.0625f, 0.1875f, 0.3125f, 0.4375f, 0.5625f, 0.6875f, 0.8125f, 0.9375f } ;
extern __device__ const int STAGE[5][2] ; //     =  { {0,0}, {1,1}, {2,3}, {4,7}, {8,15} } ;
extern __device__ const int CHUNKSZE[5] ; //     =  { 4, 8, 8, 8, 8 } ;


__host__ __device__ double candidate_sigma_cu(double poww, int numharm, long long numindep);

__host__ void add_and_searchCU3_PT_f(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch );

__host__ void add_and_searchCU31_f  (dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream,cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base, float* rLows, int noSteps, const uint noStages, uint FLAGS );

__host__ void add_and_searchCU311_f (dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch );

__host__ void add_and_maxCU31_f(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream,cuSearchList searchList, float* d_cands, uint* d_sem, int base, float* rLows, int noSteps, const uint noStages, uint FLAGS );

template<int noStages, int canMethoud> __global__ void add_and_searchCU4(cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base);

#endif
