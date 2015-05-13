#include "cuda_accel_CV.h"

__global__ void convolveffdot1(fcomplexcu *ffdot, const int width, const int stride, const int height, const fcomplexcu *data, const fcomplexcu *kernels)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  fcomplexcu ker;

  if (ix < width && iy < height)
  {
    // Calculate flat index
    const int idx = iy * stride + ix;

    ker = kernels[idx];
    ker.r /= (float) width;
    ker.i /= (float) width;
  }
}
