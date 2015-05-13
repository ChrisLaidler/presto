#include "cuda_accel_CV.h"

/** Convolution kernel - Convolve a stack with a Stack sized kernel - single step
 * Each thread loops down a column of the plains and convolves input with kernel and writes result to plain
 */
template<uint FLAGS, uint no>
__global__ void convolveffdot4(const fcomplexcu *kernels, const fcomplexcu *datas, fcomplexcu *ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, fCplxTex kerTex)
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;

  if ( tid < width )
  {
    int iy = 0;         // y index
    int idx;            // flat index

    fcomplexcu ker;     // kernel data
    float2 kker;        // kernel data (texture memory)
    fcomplexcu dat[no]; // set of input data for this thread

    // Stride
    kernels += tid;
    ffdot   += tid;
    datas   += tid;

    int pHeight = 0;

    // Read all the input data
#pragma unroll
    for (int n = 0; n < no; n++)
    {
      dat[n] = datas[ n * stride] ;
      dat[n].r /= (float) width ;
      dat[n].i /= (float) width ;
    }

#pragma unroll
    for (int n = 0; n < no; n++)                      // Loop through the plains
    {
      for (; iy < pHeight + heights.val[n]; iy++)     // Loop over the plain
      {
        idx = (iy) * stride ;

        if ( FLAGS & FLAG_CNV_TEX )
        {
          kker = tex2D < float2 > (kerTex, tid, iy);
          ffdot[idx].r = (dat[n].r * kker.x + dat[n].i * kker.y );
          ffdot[idx].i = (dat[n].i * kker.x - dat[n].r * kker.y );
        }
        else
        {
          ker = kernels[idx];
          ffdot[idx].r = (dat[n].r * ker.r + dat[n].i * ker.i);
          ffdot[idx].i = (dat[n].i * ker.r - dat[n].r * ker.i);
        }
      }
      pHeight += heights.val[n];
    }
  }
}

