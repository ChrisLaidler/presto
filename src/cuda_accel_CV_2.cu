#include "cuda_accel_CV.h"

/** Convolution kernel - All r and contiguous blocks of Z
 */
__global__ void convolveffdot2(fcomplexcu *ffdot, uint width, uint stride, uint height, fcomplexcu *data, fcomplexcu *kernels, uint number)
{
  const uint ix = blockIdx.x * blockDim.x + threadIdx.x;
  const uint iy = blockIdx.y * blockDim.y + threadIdx.y;

  if (ix < width)
  {
    fcomplexcu ker;
    uint idx;

    const float inpReal = data[ix].r / (float) width;
    const float inpImag = data[ix].i / (float) width;

    for (int i = iy * number; i < (iy + 1) * number; i++)
    {
      if (i < height)
      {
        idx = i * stride + ix;
        ker = kernels[idx];
        ffdot[idx].r = (inpReal * ker.r + inpImag * ker.i) ;
        ffdot[idx].i = (inpImag * ker.r - inpReal * ker.i) ;
      }
    }
  }
}
