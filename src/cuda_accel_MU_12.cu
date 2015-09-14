#include "cuda_accel_MU.h"

__global__ void mult12(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, fCplxTex kerTex)
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;

  if (tid < width)
  {
    float2 kker;
    int idx;

    const float inpReal = data[tid].r / (float) width;
    const float inpImag = data[tid].i / (float) width;

    ffdot   += tid;

    for (int y = 0; y < height; y++)
    {
      idx   = y * stride;
      kker  = tex2D < float2 > (kerTex, tid, y);

      ffdot[idx].r = (inpReal * kker.x + inpImag * kker.y);
      ffdot[idx].i = (inpImag * kker.x - inpReal * kker.y);
    }
  }
}
