#include "cuda_accel_CV.h"

__global__ void convolveffdot37(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, const fcomplexcu *kernels)
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;

  if (tid < width)
  {
    //fcomplexcu dat = data[tid];
    fcomplexcu ker[CNV_WORK], res[CNV_WORK];

    //dat.r /= (float) width;
    //dat.i /= (float) width;
    const float inpReal = data[tid].r / (float) width;
    const float inpImag = data[tid].i / (float) width;

    // Stride the input and output
    kernels += tid;
    ffdot   += tid;

    for (int y = 0; y < height; y += CNV_WORK)
    {
      for (int d = 0; d < CNV_WORK; d++)
      {
        ker[d] = kernels[(y + d) * stride];
      }

      for (int d = 0; d < CNV_WORK; d++)
      {
        res[d].r = (inpReal * ker[d].r + inpImag * ker[d].i);
        res[d].i = (inpImag * ker[d].r - inpReal * ker[d].i);
      }

      for (int d = 0; d < CNV_WORK; d++)
      {
        ffdot[(y + d) * stride] = res[d];
      }
    }
  }
}

