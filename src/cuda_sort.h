#ifndef CUDA_SORT_H
#define CUDA_SORT_H

template <typename T, int noEls>
__device__ void bitonicSort_mem(T *data);

template <typename T, int noEls, int noArr>
__device__ void bitonicSort_reg(T *val);

template <typename T, const int noSort>
__device__ void bitonicSort_SM(T *data);



template< int noEls >
__device__ float cuOrderStatPow2_radix(float *val, int offset, int printVals);

template <typename T, int noEls, int noArr>
__device__ T cuOrderStatPow2_sort(T *val, int os);

template <typename T, int noEls>
__device__ void cuOrderStatPow2_sort_SM(T *data, int os);

template <typename T, int noArr>
__device__ inline T getValue(T *val, const int os)
{
  __shared__ T oreerSta;

  const int noInWarp	= noArr*32;
  const int tid		= threadIdx.y * blockDim.x + threadIdx.x;       /// Thread ID in block (flat index)
  const int wId		= tid / 32;
  const int laneId	= tid % 32;

  int tw 		= os / noInWarp;
  int rest		= ( os - noInWarp * tw );
  int tl		= rest % 32 ;
  int ta		= rest / 32 ;

  if ( laneId == tl )
  {
    if ( wId == tw )
    {
      oreerSta = val[ta];
    }
  }

  __syncthreads();

  return oreerSta;
}

#endif // CUDA_SORT_H
