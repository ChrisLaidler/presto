#include "cuda_accel_MU.h"

/** Convolution kernel - One thread per r location (input FFT)
 * Each thread reads one input value and loops down over the kernels
 */
__global__ void mult11(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, const fcomplexcu *kernels)
{
  const int bidx  = threadIdx.y * CNV_DIMX + threadIdx.x;
  const int tid   = blockIdx.x  * CNV_DIMX * CNV_DIMY     + bidx;

  if (tid < width)    // Clip
  {
    fcomplexcu ker;   // item from kernel
    int idx = 0;      // flat index

    const float inpReal = data[tid].r / (float) width;
    const float inpImag = data[tid].i / (float) width;

    // Stride the input and output
    kernels += tid;
    ffdot   += tid;

    //#pragma unroll
    for (int y = 0; y < height; y++)
    {
      idx = y * stride;

      ker = kernels[idx];

      ffdot[idx].r = (inpReal * ker.r + inpImag * ker.i);
      ffdot[idx].i = (inpImag * ker.r - inpReal * ker.i);
    } 
  }
}
