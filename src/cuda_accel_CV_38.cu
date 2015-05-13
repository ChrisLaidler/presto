#include "cuda_accel_CV.h"

template <int noBatch >
__global__ void convolveffdot38(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, const fcomplexcu *kernels)
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;
  int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;

  int idx[noBatch];
  fcomplexcu dat[noBatch];

#pragma unroll
  for ( int i = 0; i < noBatch ; i++)
  {
    tid = blockIdx.x * CNV_DIMX * CNV_DIMY * noBatch + bidx*(i+1);
    if (tid < width)
    {
      idx[i] = tid;
      dat[i] = data[tid];
      dat[i].r /= (float) width;
      dat[i].i /= (float) width;
    }
    else
    {
      idx[i] = -1;
      //tid = -1;
    }
  }

  //if (tid < width)
  {
    fcomplexcu ker;
    int idxv = 0;

    // Stride
    //kernels += tid;
    //ffdot   += tid;

    // Loop down over the kernels
    for (int y = 0; y < height; y++)
    {
      idxv = y * stride;
#pragma unroll
      for ( int i = 0; i < noBatch ; i++)
      {
        if ( idx[i] >= 0 )
        {
          ker = kernels[idxv+idx[i]];

          ffdot[idxv+idx[i]].r = (dat[i].r * ker.r + dat[i].i * ker.i);//(float) width;
          ffdot[idxv+idx[i]].i = (dat[i].i * ker.r - dat[i].r * ker.i);//(float) width;
        }
      }
    }
  }
}

