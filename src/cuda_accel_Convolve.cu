/**
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation version 3.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */


#include <cufft.h>
#include <algorithm>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
//#include <cub/cub.cuh>
#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>


extern "C"
{
#define __float128 long double
#include "accel.h"
}

#include "cuda_utils.h"
#include "cuda_accel_utils.h"

/** Convolution kernel - One thread per f-∂f pixel
 */
__global__ void convolveffdot(fcomplexcu *ffdot, const int width, const int stride, const int height, const fcomplexcu *data, const fcomplexcu *kernels)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  fcomplexcu dat, ker;

  if (ix < width && iy < height)
  {
    // Calculate flat index
    const int idx = iy * stride + ix;

    ker = kernels[idx];
    ker.r /= (float) width;
    ker.i /= (float) width;
    dat = data[ix];
  }
}

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

/** Convolution kernel - One thread per r location (input FFT)
 * Each thread reads one input value and loops down over the kernels
 * NOTE: this is the same as 35
 */
__global__ void convolveffdot3(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, const fcomplexcu *kernels)
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;
  const int tid = blockIdx.x * CNV_DIMX * CNV_DIMY + bidx;

  if (tid < width)
  {
    fcomplexcu dat = data[tid];
    fcomplexcu ker;
    int idx;

    dat.r /= (float) width;
    dat.i /= (float) width;

    kernels += tid;
    ffdot += tid;

    for (int y = 0; y < height; y++)
    {
      idx = y * stride;
      ker = kernels[idx];
      ffdot[idx].r = (dat.r * ker.r + dat.i * ker.i);
      ffdot[idx].i = (dat.i * ker.r - dat.r * ker.i);
    }
  }
}

/** Convolution kernel - One thread per r location (input FFT) loop down z
 * Each thread reads one input value and loops down over the kernels
 */
__global__ void convolveffdot35(fcomplexcu *ffdot, uint width, uint stride, const uint height, const fcomplexcu *data, const fcomplexcu *kernels)
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;

  if (tid < width) // Clip
  {
    fcomplexcu ker; // item from kernel
    int idx = 0;    // flat index

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

/** Convolution kernel - All r and interlaced of Z
 */
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

/** Convolution kernel - One thread per r location loop down z - Texture memory
 * Each thread reads one input value and loops down over the kernels
 */
__global__ void convolveffdot36(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, fCplxTex kerTex)
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

/** Convolution kernel - Each thread handles blocks of x looping down over y
 * Each thread reads one input value and loops down over the kernels
 */
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

/** Convolution kernel - Convolve a stack with a Stack sized kernel - multi-step
 * Each thread loops down a column of the plains and convolves input with kernel and writes result to plain
 */
#if TEMPLATE_CONVOLVE == 1
template<uint FLAGS, uint noPlns, uint noSteps>
__global__ void convolveffdot41(const fcomplexcu *kernels, const fcomplexcu *datas, fcomplexcu *ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, cHarmList kerDat, fCplxTex kerTex )
#else
template<uint FLAGS, uint noPlns>
__global__ void convolveffdot41(const fcomplexcu *kernels, const fcomplexcu *datas, fcomplexcu *ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, cHarmList kerDat, fCplxTex kerTex, const uint noSteps )
#endif
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;

  if ( tid < width )
  {
    int iy = 0;                       // y index
    int idx;                          // flat index

    fcomplexcu ker;                   // kernel data
#if TEMPLATE_CONVOLVE == 1
    fcomplexcu dat[noPlns*noSteps];   // set of input data for this thread, this should be dat[noPlns*noSteps];
#else
    fcomplexcu dat[noPlns*MAX_STEPS]; // set of input data for this thread, this should be dat[noPlns*noSteps];
#endif

    // Stride
    kernels += tid;
    ffdot   += tid;
    datas   += tid;

    int pHeight = 0;

    for (int plnNo = 0; plnNo < noPlns; plnNo++)          // Loop through the plains
    {
      kerDat.val[plnNo] += tid;
    }

    // Read the input data
#if TEMPLATE_CONVOLVE == 1
#pragma unroll
#endif
    for (int n = 0; n < noPlns*noSteps; n++)
    {
      dat[n]           = datas[ n * stride ] ;
      dat[n].r        /= (float) width ;
      dat[n].i        /= (float) width ;
    }

#ifndef DEBUG
#pragma unroll
#endif
    for (int plnNo = 0; plnNo < noPlns; plnNo++)      // Loop through the plains
    {
      for (iy = 0; iy < heights.val[plnNo]; iy++)     // Loop over the plain
      {
        idx = (iy) * stride ;
        int sy = iy+pHeight;

        if ( FLAGS & FLAG_CNV_TEX )
        {
          ker   = *(fcomplexcu*)& tex2D < float2 > (kerTex, tid, sy);
        }
        else
        {
          ker   = kerDat.val[plnNo][idx];
        }

#if TEMPLATE_CONVOLVE == 1
#pragma unroll
#endif
        for ( int step = 0; step < noSteps; step++)   // Loop over steps
        {
          // Calculate indices
          if      ( FLAGS & FLAG_STP_ROW )
          {
            idx  = ( sy * noSteps + step) * stride ;
          }
          else if ( FLAGS & FLAG_STP_PLN )
          {
            idx  = ( sy + heights.val[plnNo]*step) * stride ;
          }
          else if ( FLAGS & FLAG_STP_STK )
          {
            idx  = ( sy + stackHeight*step) * stride ;
          }

          const int ox = plnNo*noSteps+step;

          // Convolve
          ffdot[idx].r = (dat[ox].r * ker.r + dat[ox].i * ker.i);
          ffdot[idx].i = (dat[ox].i * ker.r - dat[ox].r * ker.i);
        }
      }
      pHeight += heights.val[plnNo];
    }
  }
}

template<uint FLAGS, uint noPlns >
__host__ void convolveffdot41_s(dim3 dimBlock, dim3 dimGrid, int i1, cudaStream_t cnvlStream, const fcomplexcu *kernels, const fcomplexcu *datas, fcomplexcu *ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, cHarmList kerDat, fCplxTex kerTex, const uint noSteps )
{
#if TEMPLATE_CONVOLVE == 1
  switch (noSteps)
  {
  case 1:
    convolveffdot41<FLAGS,noPlns,1><<<dimGrid,  dimBlock, i1, cnvlStream >>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
    break;
  case 2:
    convolveffdot41<FLAGS,noPlns,2> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
      break;
  case 3:
    convolveffdot41<FLAGS,noPlns,3> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
      break;
  case 4:
    convolveffdot41<FLAGS,noPlns,4> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
      break;
  case 5:
    convolveffdot41<FLAGS,noPlns,5> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
      break;
  case 6:
    convolveffdot41<FLAGS,noPlns,6> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
      break;
  case 7:
    convolveffdot41<FLAGS,noPlns,7> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
      break;
  case 8:
    convolveffdot41<FLAGS,noPlns,8> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
      break;
  //case 9:
  //  convolveffdot41<FLAGS,noPlns,9> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
  //    break;
  //case 10:
  //  convolveffdot41<FLAGS,noPlns,10> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
  //    break;
  //case 11:
  //  convolveffdot41<FLAGS,noPlns,11> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
  //    break;
  default:
    fprintf(stderr, "ERROR: convolveffdot41 has not been templated for %i steps\n", noSteps);
    exit(EXIT_FAILURE);
  }

#else
  convolveffdot41<FLAGS,noPlns> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps);
#endif
}

template<uint FLAGS >
__host__ void convolveffdot41_p(dim3 dimBlock, dim3 dimGrid, int i1, cudaStream_t cnvlStream, const fcomplexcu *kernels, const fcomplexcu *datas, fcomplexcu *ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, cHarmList kerDat, fCplxTex kerTex, uint noSteps, uint noPlns )
{
  switch (noPlns)
  {
    case 1:
      convolveffdot41_s<FLAGS,1> (dimBlock, dimGrid, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps );
      break;
    case 2:
      convolveffdot41_s<FLAGS,2> (dimBlock, dimGrid, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps );
      break;
    case 3:
      convolveffdot41_s<FLAGS,3> (dimBlock, dimGrid, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps );
      break;
    case 4:
      convolveffdot41_s<FLAGS,4> (dimBlock, dimGrid, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps );
      break;
    case 5:
      convolveffdot41_s<FLAGS,5> (dimBlock, dimGrid, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps );
      break;
    case 6:
      convolveffdot41_s<FLAGS,6> (dimBlock, dimGrid, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps );
      break;
    case 7:
      convolveffdot41_s<FLAGS,7> (dimBlock, dimGrid, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps );
      break;
    case 8:
      convolveffdot41_s<FLAGS,8> (dimBlock, dimGrid, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps );
      break;
    case 9:
      convolveffdot41_s<FLAGS,9> (dimBlock, dimGrid, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps );
      break;
    default:
      fprintf(stderr, "ERROR: convolveffdot41 has not been templated for %i plains\n", noPlns);
      exit(EXIT_FAILURE);
  }
}

__host__ void convolveffdot41_f(dim3 dimBlock, dim3 dimGrid, int i1, cudaStream_t cnvlStream, const fcomplexcu *kernels, const fcomplexcu *datas, fcomplexcu *ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, cHarmList kerDat, fCplxTex kerTex, uint noSteps, uint noPlns, uint FLAGS )
{
  if ( FLAGS & FLAG_CNV_TEX )
  {
    /*
    if      (FLAGS & FLAG_STP_ROW )
      convolveffdot41_p <FLAG_CNV_TEX | FLAG_STP_ROW> (dimBlock, dimGrid, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps, noPlns );
    else if ( FLAGS & FLAG_STP_PLN )
      convolveffdot41_p <FLAG_CNV_TEX | FLAG_STP_PLN> (dimBlock, dimGrid, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps, noPlns );
    //else if ( FLAGS & FLAG_STP_STK )
    //  convolveffdot41_p <FLAG_CNV_TEX | FLAG_STP_STK> (dimBlock, dimGrid, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps, noPlns );
    else
    {
      fprintf(stderr, "ERROR: convolveffdot41 has not been templated for \n", noPlns);
      exit(EXIT_FAILURE);
    }
    */
  }
  else
  {
      if      ( FLAGS & FLAG_STP_ROW )
        convolveffdot41_p< FLAG_STP_ROW > (dimBlock, dimGrid, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps, noPlns );
      else if ( FLAGS & FLAG_STP_PLN )
        convolveffdot41_p< FLAG_STP_PLN > (dimBlock, dimGrid, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps, noPlns );
      //else if ( FLAGS & FLAG_STP_STK )
      //  convolveffdot41_p< FLAG_STP_STK | FLAG_STP_ROW > (dimBlock, dimGrid, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps, noPlns );
      else
      {
        fprintf(stderr, "ERROR: convolveffdot41 has not been templated for %i plains\n", noPlns);
        exit(EXIT_FAILURE);
      }
  }
}

/** Convolution kernel - Convolve an entire family (Stack List) with family convolution kernel
 * Each thread loops down a column of the plains and convolves input with kernel and writes result to plain
 */
template<uint FLAGS >
__global__ void convolveffdot5(const fcomplexcu *kernels, const fcomplexcu *datas, fcomplexcu *ffdot, iHarmList widths, iHarmList strides, iHarmList heights, uint no, uint hh, uint noSteps, uint step )
{
  const int ix = blockIdx.x * CNV_DIMX * CNV_DIMY + CNV_DIMX * threadIdx.y + threadIdx.x;

  int iy      = 0;
  int height  = 0;
  fcomplexcu ker, dat;

  // Stride
  kernels += ix;
  ffdot   += ix;
  datas   += ix;
  int tid  = 0;
  int idx  = 0;
  int stride;

  if ( FLAGS & FLAG_STP_ROW )
  {
    ffdot  += ix + step * strides.val[0]; ;
  }
  else if ( FLAGS & FLAG_STP_PLN )
  {
    ffdot += ix ; //+ step * strides.val[0] * heights.val[0];
  }

  for (int n = 0; n < no; n++)
  {
    if (ix < widths.val[n])
    {
      stride   = strides.val[n];
      dat      = datas[step*stride];
      datas    += stride*noSteps;

      dat.r   /= (float) widths.val[n] ;
      dat.i   /= (float) widths.val[n] ;

      height  += heights.val[n];

      if ( FLAGS & FLAG_STP_PLN )
      {
        ffdot += step * strides.val[n] * heights.val[n];
      }

      for (; iy < height; iy++)
      {
        ker           =  kernels[tid];
        tid          += stride;


        // Convolve
        ffdot[idx].r = (dat.r * ker.r + dat.i * ker.i);
        ffdot[idx].i = (dat.i * ker.r - dat.r * ker.i);


        //ffdot[tid].r  =  (dat.r * ker.r + dat.i * ker.i);
        //ffdot[tid].i  =  (dat.i * ker.r - dat.r * ker.i);



        // Calculate indices
        if ( FLAGS & FLAG_STP_ROW )
        {
          ffdot  += noSteps * stride;
        }
        else if ( FLAGS & FLAG_STP_PLN )
        {
          ffdot += stride;
        }
      }
      if ( FLAGS & FLAG_STP_PLN )
      {
        ffdot += ( noSteps - 1 - step ) * strides.val[n] * heights.val[n];
      }
    }
  }
}

/** Convolution kernel - Convolve a stack - using a 1 plain convolution kernel
 * Split the stack into overlapping sections and read the shared kernel values once and convolve with all relevant values
 * NOTE: does not with multi step stacks
 */
template<uint FLAGS, uint no>
__global__ void convolveffdot6(const fcomplexcu *kernels, const fcomplexcu *datas, cHarmList ffdot, const int width, const uint stride, iHarmList heights, const uint hh, fCplxTex kerTex, iHarmList zUp, iHarmList zDn )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;

  if ( tid < width )
  {
    int iy = 0;             // y index
    int iy2;                // Upper Y value
    int idx1, idx2;         // flat index
    int plnNo;
    int s = 0;

    if ( no == 1 )
    {
      fcomplexcu ker;     // kernel data
      fcomplexcu dat;     // set of input data for this thread

      // Stride
      kernels         += tid;
      ffdot.val[0]    += tid;
      datas           += tid;

      dat = datas[ 0 ] ;
      dat.r /= (float) width ;
      dat.i /= (float) width ;

      for (; iy < heights.val[0]; iy++)     // Loop over the plain
      {
        idx1 = (iy) * stride ;

        if ( FLAGS & FLAG_CNV_TEX )
        {
          ker   = *(fcomplexcu*)& tex2D < float2 > (kerTex, tid, iy);
        }
        else
        {
          ker   = kernels[idx1];
        }

        ffdot.val[0][idx1].r = (dat.r * ker.r + dat.i * ker.i);
        ffdot.val[0][idx1].i = (dat.i * ker.r - dat.r * ker.i);
      }

    }
    else
    {
      const int noSteps = 2;
      fcomplexcu ker1[noSteps];
      fcomplexcu ker2[1];
      fcomplexcu dat[no];     // set of input data for this thread

      // Stride
      kernels += tid;
      datas   += tid;

#pragma unroll
      for (int n = 0; n < no; n++)
      {
        dat[n]          = datas[ /* tid + */ n * stride] ;
        dat[n].r        /= (float) width ;
        dat[n].i        /= (float) width ;

        ffdot.val[n]    += tid;
      }

      // Loop up through steps
#pragma unroll
      for ( int step = 0; step < no - 1; step++ )
      {
        for ( iy = zUp.val[step]; iy < zUp.val[step+1] ; iy++ ) // Loop over the z values for the current sub plans
        {
          //int mx = noSteps;

          //if ( (iy + mx) > zUp.val[step+1] )
            //mx = zUp.val[step+1] - iy;

//#pragma unroll
          //for ( int s = 0; s < mx ; s++ )
          {
            iy2   = iy  + zDn.val[no-2-step] ;
            idx1  = iy  * stride ;
            idx2  = iy2 * stride ;

            // Read the kernel value
            if ( FLAGS & FLAG_CNV_TEX )
            {
              ker1[s]  = *(fcomplexcu*)& tex2D < float2 > (kerTex, tid, iy  );
              ker2[s]  = *(fcomplexcu*)& tex2D < float2 > (kerTex, tid, iy2 );
            }
            else
            {
              ker1[s]  = kernels[idx1];
              ker2[s]  = kernels[idx2];
            }
          }

#pragma unroll
          for ( int dd = 0; dd < step+1; dd++) // Loop over sub plains
          {
            //for ( int s = 0; s < mx ; s++ )
            {
              idx1  = ( iy  - zUp.val[dd] ) * stride ; // TODO: can do the dd offset at the beginning
              idx2  = ( iy2 - zUp.val[dd] ) * stride ; // TODO: can do the dd offset at the beginning

              plnNo = dd;

              (ffdot.val[plnNo])[idx1].r = (dat[plnNo].r * ker1[s].r + dat[plnNo].i * ker1[s].i);
              (ffdot.val[plnNo])[idx1].i = (dat[plnNo].i * ker1[s].r - dat[plnNo].r * ker1[s].i);

              (ffdot.val[plnNo])[idx2].r = (dat[plnNo].r * ker2[s].r + dat[plnNo].i * ker2[s].i);
              (ffdot.val[plnNo])[idx2].i = (dat[plnNo].i * ker2[s].r - dat[plnNo].r * ker2[s].i);
            }
          }
        }
      }

      // Loop through the centre block
      for ( iy = zUp.val[no-1]; iy < zDn.val[0] ; iy += noSteps )
      {
        int mx = noSteps;

        if ( (iy + mx) > zDn.val[0] )
          mx = zDn.val[0] - iy;

        for ( int s = 0; s < mx ; s++ )
        {
          idx1 = ( iy + s ) * stride ;

          // Read the kernel value
          if ( FLAGS & FLAG_CNV_TEX )
          {
            ker1[s]   = *((fcomplexcu*)& tex2D < float2 > (kerTex, tid, ( iy + s ) )) ;
          }
          else
          {
            ker1[s]    = kernels[idx1] ;
          }
        }

#pragma unroll
        for ( int dd = 0; dd < no; dd++) // Loop over sub plains
        {
          for ( int s = 0; s < mx ; s++ )
          {
            idx1 = ( (iy+s) -zUp.val[dd] ) * stride ; // TODO: can do the dd offset at the beginning

            (ffdot.val[dd])[idx1].r = (dat[dd].r * ker1[s].r + dat[dd].i * ker1[s].i);
            (ffdot.val[dd])[idx1].i = (dat[dd].i * ker1[s].r - dat[dd].r * ker1[s].i);
          }
        }
      }
    }
  }
}

/** Convolution kernel - Convolve a multi-step stack - using a 1 plain convolution kernel
 * Split the stack into overlapping sections and read the shared kernel values once and convolve with all relevant values
 */

#if TEMPLATE_CONVOLVE == 1
template<uint FLAGS, uint noPlns, uint noSteps>
__global__ void convolveffdot7(const fcomplexcu *kernel, const fcomplexcu *datas, cHarmList ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, fCplxTex kerTex, iHarmList zUp, iHarmList zDn )
#else
template<uint FLAGS, uint noPlns>
__global__ void convolveffdot7(const fcomplexcu *kernel, const fcomplexcu *datas, cHarmList ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, fCplxTex kerTex, iHarmList zUp, iHarmList zDn, uint noSteps )
#endif
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;

  if ( tid < width )
  {
    int iyTop = 0;                    // Kernel y index of top section
    int iyBot;                        // Kernel y index of bottom section
    int idxTop, idxBot;               // Plain  y index of top & bottom of section
#if TEMPLATE_CONVOLVE == 1
    fcomplexcu dat[noPlns*noSteps];   // set of input data for this thread, this should be dat[noPlns*noSteps];
#else
    fcomplexcu dat[noPlns*MAX_STEPS];   // set of input data for this thread, this should be dat[noPlns*noSteps];
#endif

    // Stride
    kernel          += tid;
    datas           += tid;

    // Shift the plain data to the correct x offset
#pragma unroll
    for (int n = 0; n < noPlns; n++)
      ffdot.val[n]    += tid;

    // Read the input data
#if TEMPLATE_CONVOLVE == 1
#pragma unroll
#endif
    for (int n = 0; n < noPlns*noSteps; n++)
    {
      dat[n]          = datas[ n * stride ] ;
      dat[n].r        /= (float) width ;
      dat[n].i        /= (float) width ;
    }

    if ( noPlns == 1 )
    {
      fcomplexcu ker;     // kernel data

      for (; iyTop < heights.val[0]; iyTop++)     // Loop over the plain
      {
        idxTop = (iyTop) * stride ;

        if ( FLAGS & FLAG_CNV_TEX )
        {
          ker   = *(fcomplexcu*)& tex2D < float2 > (kerTex, tid, iyTop);
        }
        else
        {
          ker   = kernel[idxTop];
        }

        // Now convolve the kernel element with the relevant plain elements
#if TEMPLATE_CONVOLVE == 1
#pragma unroll
#endif
        for ( int step = 0; step < noSteps; step++)   // Loop over steps
        {
          // Calculate indices
          if ( FLAGS & FLAG_STP_ROW )
          {
            idxTop  = (( (iyTop ) ) * noSteps + step) * stride ;
          }
          else if ( FLAGS & FLAG_STP_PLN )
          {
            idxTop  = (( (iyTop )  ) + heights.val[0]*step) * stride ;
          }
          else if ( FLAGS & FLAG_STP_STK )
          {
            idxTop  = (( (iyTop )  ) + stackHeight*step) * stride ;
          }

          ffdot.val[0][idxTop].r = (dat[step].r * ker.r + dat[step].i * ker.i);
          ffdot.val[0][idxTop].i = (dat[step].i * ker.r - dat[step].r * ker.i);
        }
      }
    }
    else
    {
      fcomplexcu kerTop[1];
      fcomplexcu kerBot[1];

      // Loop through sections - read kernel value - convolve with plain values
#pragma unroll
      for ( int section = 0; section < noPlns - 1; section++ )
      {
        for ( iyTop = zUp.val[section]; iyTop < zUp.val[section+1] ; iyTop++ ) // Loop over the z values for the kernel for this this section
        {
          iyBot   = iyTop + zDn.val[noPlns-2-section] ;
          idxTop  = iyTop * stride ;
          idxBot  = iyBot * stride ;

          FOLD  // Read the kernel value
          {
            if ( FLAGS & FLAG_CNV_TEX )
            {
              kerTop[0]  = *(fcomplexcu*)& tex2D < float2 > (kerTex, tid, iyTop );
              kerBot[0]  = *(fcomplexcu*)& tex2D < float2 > (kerTex, tid, iyBot );
            }
            else
            {
              kerTop[0]  = kernel[idxTop];
              kerBot[0]  = kernel[idxBot];
            }
          }

          // Loop over all plain values and convolve
#pragma unroll
          for ( int plnNo = 0; plnNo < section+1; plnNo++)  // Loop over sub plains
          {
#if TEMPLATE_CONVOLVE == 1
#pragma unroll
#endif
            for ( int step = 0; step < noSteps; step++)     // Loop over steps
            {
              // Calculate indices
              if      ( FLAGS & FLAG_STP_ROW )
              {
                idxTop  = (( iyTop - zUp.val[plnNo] ) * noSteps + step) * stride ;
                idxBot  = (( iyBot - zUp.val[plnNo] ) * noSteps + step) * stride ;
              }
              else if ( FLAGS & FLAG_STP_PLN )
              {
                idxTop  = (( iyTop - zUp.val[plnNo] ) + heights.val[plnNo]*step) * stride ;
                idxBot  = (( iyBot - zUp.val[plnNo] ) + heights.val[plnNo]*step) * stride ;
              }
              else if ( FLAGS & FLAG_STP_STK )
              {
                idxTop  = (( iyTop - zUp.val[plnNo] ) + stackHeight*step) * stride ;
                idxBot  = (( iyBot - zUp.val[plnNo] ) + stackHeight*step) * stride ;
              }

              // Convolve
              (ffdot.val[plnNo])[idxTop].r = (dat[plnNo*noSteps+step].r * kerTop[0].r + dat[plnNo*noSteps+step].i * kerTop[0].i);
              (ffdot.val[plnNo])[idxTop].i = (dat[plnNo*noSteps+step].i * kerTop[0].r - dat[plnNo*noSteps+step].r * kerTop[0].i);

              (ffdot.val[plnNo])[idxBot].r = (dat[plnNo*noSteps+step].r * kerBot[0].r + dat[plnNo*noSteps+step].i * kerBot[0].i);
              (ffdot.val[plnNo])[idxBot].i = (dat[plnNo*noSteps+step].i * kerBot[0].r - dat[plnNo*noSteps+step].r * kerBot[0].i);
            }
          }
        }
      }

      // Loop through the centre block - convolve with chunks plain values
      // I tested reading in "chunks" of kernel and then looping but this made no improvement
      for ( iyTop = zUp.val[noPlns-1]; iyTop < zDn.val[0] ; iyTop += 1 )
      {
        FOLD // Read kernel value
        {
          idxTop = ( iyTop ) * stride ;

          // Read the kernel value
          if ( FLAGS & FLAG_CNV_TEX )
          {
            kerTop[0]   = *((fcomplexcu*)& tex2D < float2 > (kerTex, tid, ( iyTop ) )) ;
          }
          else
          {
            kerTop[0]   = kernel[idxTop] ;
          }
        }

        // Now convolve the kernel element with the relevant plain elements
#pragma unroll
        for ( int plnNo = 0; plnNo < noPlns; plnNo++)   // Loop over plains
        {
#if TEMPLATE_CONVOLVE == 1
#pragma unroll
#endif
          for ( int step = 0; step < noSteps; step++)   // Loop over steps
          {
            // Calculate indices
            if ( FLAGS & FLAG_STP_ROW )
            {
              idxTop  = (( (iyTop )  - zUp.val[plnNo] ) * noSteps + step) * stride ;
            }
            else if ( FLAGS & FLAG_STP_PLN )
            {
              idxTop  = (( (iyTop )  - zUp.val[plnNo] ) + heights.val[plnNo]*step) * stride ;
            }
            else if ( FLAGS & FLAG_STP_STK )
            {
              idxTop  = (( (iyTop )  - zUp.val[plnNo] ) + stackHeight*step) * stride ;
            }

            // Convolve
            (ffdot.val[plnNo])[idxTop].r = (dat[plnNo*noSteps+step].r * kerTop[0].r + dat[plnNo*noSteps+step].i * kerTop[0].i);
            (ffdot.val[plnNo])[idxTop].i = (dat[plnNo*noSteps+step].i * kerTop[0].r - dat[plnNo*noSteps+step].r * kerTop[0].i);
          }
        }
      }
    }
  }
}

template<uint FLAGS, uint noPlns >
__host__ void convolveffdot7_s(dim3 dimBlock, dim3 dimGrid, int i1, cudaStream_t cnvlStream, const fcomplexcu *kernel, const fcomplexcu *datas, cHarmList ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, fCplxTex kerTex, iHarmList zUp, iHarmList zDn, const uint noSteps )
{
#if TEMPLATE_CONVOLVE == 1
  switch (noSteps)
  {
  case 1:
    convolveffdot7<FLAGS,noPlns,1><<<dimGrid,  dimBlock, i1, cnvlStream >>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
    break;
  case 2:
    convolveffdot7<FLAGS,noPlns,2> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
      break;
  case 3:
    convolveffdot7<FLAGS,noPlns,3> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
      break;
  case 4:
    convolveffdot7<FLAGS,noPlns,4> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
      break;
  case 5:
    convolveffdot7<FLAGS,noPlns,5> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
      break;
  case 6:
    convolveffdot7<FLAGS,noPlns,6> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
      break;
  case 7:
    convolveffdot7<FLAGS,noPlns,7> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
      break;
  case 8:
    convolveffdot7<FLAGS,noPlns,8> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
      break;
  //case 9:
  //  convolveffdot7<FLAGS,noPlns,9> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
  //    break;
  //case 10:
  //  convolveffdot7<FLAGS,noPlns,10> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
   //   break;
  //case 11:
  //  convolveffdot7<FLAGS,noPlns,11> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
  //    break;
  //case MAX_STEPS:
  //  convolveffdot7<FLAGS,noPlns,MAX_STEPS> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
  //    break;
  default:
    fprintf(stderr, "ERROR: convolveffdot7 has not been templated for %i steps\n", noSteps);
    exit(EXIT_FAILURE);
  }
#else
  convolveffdot7<FLAGS,noPlns> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps);
#endif
}

template<uint FLAGS >
__host__ void convolveffdot7_p(dim3 dimBlock, dim3 dimGrid, int i1, cudaStream_t cnvlStream, const fcomplexcu *kernel, const fcomplexcu *datas, cHarmList ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, fCplxTex kerTex, iHarmList zUp, iHarmList zDn, const uint noSteps, const uint noPlns )
{
  switch (noPlns)
  {
    case 1:
      convolveffdot7_s<FLAGS,1> (dimBlock, dimGrid, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps );
      break;
    case 2:
      convolveffdot7_s<FLAGS,2> (dimBlock, dimGrid, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps );
      break;
    case 3:
      convolveffdot7_s<FLAGS,3> (dimBlock, dimGrid, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps );
      break;
    case 4:
      convolveffdot7_s<FLAGS,4> (dimBlock, dimGrid, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps );
      break;
    case 5:
      convolveffdot7_s<FLAGS,5> (dimBlock, dimGrid, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps );
      break;
    case 6:
      convolveffdot7_s<FLAGS,6> (dimBlock, dimGrid, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps );
      break;
    case 7:
      convolveffdot7_s<FLAGS,7> (dimBlock, dimGrid, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps );
      break;
    case 8:
      convolveffdot7_s<FLAGS,8> (dimBlock, dimGrid, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps );
      break;
    case 9:
      convolveffdot7_s<FLAGS,9> (dimBlock, dimGrid, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps );
      break;
    default:
      fprintf(stderr, "ERROR: convolveffdot7 has not been templated for %i plains\n", noPlns);
      exit(EXIT_FAILURE);
  }
}

__host__ void convolveffdot7_f(dim3 dimBlock, dim3 dimGrid, int i1, cudaStream_t cnvlStream, const fcomplexcu *kernel, const fcomplexcu *datas, cHarmList ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, fCplxTex kerTex, iHarmList zUp, iHarmList zDn, const uint noSteps, const uint noPlns, uint FLAGS )
{
  if ( FLAGS & FLAG_CNV_TEX )
  {
    if      ( FLAGS & FLAG_STP_ROW )
      convolveffdot7_p<FLAG_CNV_TEX | FLAG_STP_ROW> (dimBlock, dimGrid, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps, noPlns );
    else if ( FLAGS & FLAG_STP_PLN )
      convolveffdot7_p<FLAG_CNV_TEX | FLAG_STP_PLN> (dimBlock, dimGrid, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps, noPlns );
    //else if ( FLAGS & FLAG_STP_STK )
    //  convolveffdot7_p<FLAG_CNV_TEX | FLAG_STP_STK> (dimBlock, dimGrid, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps, noPlns );
    else
    {
      fprintf(stderr, "ERROR: convolveffdot7 has not been templated for flag combination. \n", noPlns);
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    if      ( FLAGS & FLAG_STP_ROW )
      convolveffdot7_p< FLAG_STP_ROW> (dimBlock, dimGrid, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps, noPlns );
    else if ( FLAGS & FLAG_STP_PLN )
      convolveffdot7_p< FLAG_STP_PLN> (dimBlock, dimGrid, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps, noPlns );
    //else if ( FLAGS & FLAG_STP_STK )
    //  convolveffdot7_p< FLAG_STP_STK> (dimBlock, dimGrid, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps, noPlns );
    else
    {
      fprintf(stderr, "ERROR: convolveffdot7 has not been templated for flag combination.\n", noPlns);
      exit(EXIT_FAILURE);
    }
  }
}

void convolveStack(cuStackList* plains, accelobs * obs, GSList** cands)
{
  dim3 dimBlock, dimGrid;

  nvtxRangePush("Convolve & FFT");

  FOLD // Convolve
  {
    dimBlock.x = CNV_DIMX;   // in my experience 16 is almost always best (half warp)
    dimBlock.y = CNV_DIMY;   // in my experience 16 is almost always best (half warp)

    int harm = 0;

    // In my limited testing I found convolving each plain separately works fastest so it is the "default"
    if      ( plains->flag & FLAG_CNV_STK ) // Do the convolutions one stack  at a time
    {
      // Convolve this entire stack in one block
      for (int ss = 0; ss< plains->noStacks; ss++)
      {
        cuFfdotStack* cStack = &plains->stacks[ss];
        iHarmList hlist;
        cHarmList plainsDat;
        cHarmList kerDat;
        iHarmList zUp;
        iHarmList zDn;

        for (int i = 0; i < cStack->noInStack; i++)     // Loop over plains to determine where they start
        {
          hlist.val[i]      =  cStack->harmInf[i].height;
          plainsDat.val[i]  =  cStack->plains[i].d_plainData;
          kerDat.val[i]     =  cStack->kernels[i].d_kerData;

          zUp.val[i]        =  cStack->zUp[i];
          zDn.val[i]        =  cStack->zDn[i];
        }

        dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY ));
        dimGrid.y = 1;

        // Synchronisation
        CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->cnvlStream, cStack->prepComp,0),     "Waiting for GPU to be ready to copy data to device.");  // Need input data
        CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->cnvlStream, plains->searchComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plain so search must be compete


        if ( plains->flag & FLAG_CNV_1KER )
        {
          // NOTE convolveffdot41 seams fatser and has been adapted for multi-step so now using it
          // Will have to put in some flag to call convolveffdot41 if desired
          //convolveffdot41_f(dimGrid, dimBlock, 0, cStack->cnvlStream, cStack->d_kerData, cStack->d_iData, cStack->d_plainData, cStack->width, cStack->stride, hlist, cStack->height, kerDat, cStack->kerDatTex, plains->noSteps, cStack->noInStack, plains->flag );

          convolveffdot7_f(dimGrid, dimBlock, 0, cStack->cnvlStream, cStack->d_kerData, cStack->d_iData, plainsDat, cStack->width, cStack->stride, hlist, cStack->height, cStack->kerDatTex, zUp, zDn, plains->noSteps, cStack->noInStack, plains->flag );
        }
        else
        {
          convolveffdot41_f(dimGrid, dimBlock, 0, cStack->cnvlStream, cStack->d_kerData, cStack->d_iData, cStack->d_plainData, cStack->width, cStack->stride, hlist, cStack->height, kerDat, cStack->kerDatTex, plains->noSteps, cStack->noInStack, plains->flag );
        }

        // Run message
        CUDA_SAFE_CALL(cudaGetLastError(), "Error at kernel launch (convolveffdot7)");

        // Synchronisation
        cudaEventRecord(cStack->convComp, cStack->cnvlStream);
      }
    }
    else if ( plains->flag & FLAG_CNV_FAM ) // Do the convolutions one family at a time
    {
      iHarmList hlist;
      iHarmList slist;
      iHarmList wlist;
      int heights = 0;
      for (int ss = 0; ss< plains->noHarms; ss++)
      {
        cuHarmInfo* cHarm = &plains->hInfos[ss];

        hlist.val[ss] = cHarm->height;
        slist.val[ss] = cHarm->stride;
        wlist.val[ss] = cHarm->width;
        heights += cHarm->height;
      }

      dimGrid.x = ceil(plains->hInfos[0].width / (float) ( CNV_DIMX * CNV_DIMY ));
      dimGrid.y = 1;

      cudaStreamWaitEvent(plains->stacks[0].cnvlStream, plains->searchComp, 0);

      if ( ( (plains->flag & FLAG_STP_ROW) || (plains->flag & FLAG_STP_PLN) ) && !( plains->flag & FLAG_CNV_1KER) )
      {
        for ( int stp = 0; stp < plains->noSteps; stp++ )
        {
          if (plains->flag & FLAG_STP_ROW)
            convolveffdot5<FLAG_STP_ROW><<<dimGrid, dimBlock, 0, plains->stacks[0].cnvlStream>>>(plains->d_kerData, plains->d_iData, plains->d_plainData, wlist, slist, hlist, plains->noHarms, heights, plains->noSteps, stp);
          else
            convolveffdot5<FLAG_STP_PLN><<<dimGrid, dimBlock, 0, plains->stacks[0].cnvlStream>>>(plains->d_kerData, plains->d_iData, plains->d_plainData, wlist, slist, hlist, plains->noHarms, heights, plains->noSteps, stp);
        }
      }
      else
      {
        fprintf(stderr,"ERROR: Family convolutions require stack interleaved data and plain sized kernels \n");
        exit(EXIT_FAILURE);
      }

      // Run message
      CUDA_SAFE_CALL(cudaGetLastError(), "Error at kernel launch");

      for (int ss = 0; ss< plains->noStacks; ss++)
      {
        cuFfdotStack* cStack = &plains->stacks[ss];
        cudaEventRecord(cStack->convComp, cStack->cnvlStream);
      }
    }
    else // ( plains->flag & FLAG_CNV_PLN ) // Do the convolutions one plain  at a time
    {
      // NOTE: The use of FLAG_CNV_1KER in this section will be handled because we are using the "kernels" pointers to the complex data

      //for (int ss = plains->noStacks-1; ss >= 0; ss-- )
      for (int ss = 0; ss< plains->noStacks; ss++)              // Loop through Stacks
      {
        cuFfdotStack* cStack = &plains->stacks[ss];

        // Do some Synchronisation
        CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->cnvlStream, cStack->prepComp,0),     "Waiting for GPU to be ready to copy data to device.");  // Need input data
        CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->cnvlStream, plains->searchComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plain so search must be compete

        fcomplexcu* d_plainData;    // The complex f-∂f plain data
        fcomplexcu* d_iData;        // The complex input array


        for (int si = 0; si< cStack->noInStack; si++)         // Loop through plains in stack
        {
          cudaStream_t sst = cStack->cnvlStream;              // NB: This is really the only option because of synchronisation issues
          //cudaStream_t sst = plains->stacks[harm%(plains->noStacks)].cnvlStream;
          //cudaStream_t sst = plains->stacks[ss+harm%(plains->noStacks-ss)].cnvlStream;

          cuHarmInfo* cHInfo    = &cStack->harmInf[si];     // The current harmonic we are working on
          cuFFdot*    cPlain    = &cStack->plains[si];      // The current f-∂f plain

          dimGrid.x = ceil(cHInfo->width / (float) ( CNV_DIMX * CNV_DIMY ));
          dimGrid.y = 1;

          // Do some Synchronisation
          CUDA_SAFE_CALL(cudaStreamWaitEvent(sst, cStack->prepComp,0),     "Waiting for GPU to be ready to copy data to device.");  // Need input data
          CUDA_SAFE_CALL(cudaStreamWaitEvent(sst, plains->searchComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plain so search must be compete

          for (int sti = 0; sti < plains->noSteps; sti++)         // Loop through Steps
          {
            d_iData       = cPlain->d_iData + cHInfo->stride * sti;

            if      ( plains->flag & FLAG_STP_ROW )
            {
              fprintf(stderr,"ERROR: Cannot do single plain convolutions with row interleave multi step stacks.\n");
              exit(EXIT_FAILURE);
            }
            else if ( plains->flag & FLAG_STP_PLN )
              d_plainData = cPlain->d_plainData + sti * cHInfo->height * cHInfo->stride;   // Shift by plain height
            else if ( plains->flag & FLAG_STP_STK )
              d_plainData = cPlain->d_plainData + sti * cStack->height * cHInfo->stride;   // Shift by stack height
            else
              d_plainData   = cPlain->d_plainData;  // If nothing is specified just use plain data

            if ( plains->flag & FLAG_CNV_TEX )
              convolveffdot36<<<dimGrid, dimBlock, 0, sst>>>(d_plainData, cHInfo->width, cHInfo->stride, cHInfo->height, d_iData, cPlain->kernel->kerDatTex);
            else
              convolveffdot35<<<dimGrid, dimBlock, 0, sst>>>(d_plainData, cHInfo->width, cHInfo->stride, cHInfo->height, d_iData, cPlain->kernel->d_kerData);

            // Run message
            CUDA_SAFE_CALL(cudaGetLastError(), "Error at convolution kernel launch");

            harm++;
          }
        }

        // Synchronise
        cudaEventRecord(cStack->convComp, cStack->cnvlStream);
      }
    }
  }

  if ( DBG_PLN01 ) // Print debug info
  {
    for (int ss = 0; ss < plains->noStacks; ss++)
    {
      cuFfdotStack* cStack = &plains->stacks[ss];
      CUDA_SAFE_CALL(cudaStreamSynchronize(cStack->cnvlStream),"");
    }

    for (int ss = 0; ss < plains->noHarms; ss++) // Print
    {
      cuFFdot* cPlain     = &plains->plains[plains->pIdx[ss]];
      printf("\nGPU Convolved h:%i   f: %f\n",ss,cPlain->harmInf->harmFrac);
      printData_cu(plains, plains->flag, plains->pIdx[ss], 10, 1);
      CUDA_SAFE_CALL(cudaStreamSynchronize(0),"");
    }
  }

  FOLD // FFT
  {
    // Copy fft data to device
    //for (int ss = plains->noStacks-1; ss >= 0; ss-- )
    for (int ss = 0; ss< plains->noStacks; ss++)
    {
      cuFfdotStack* cStack = &plains->stacks[ss];

      // Synchronise
      cudaStreamWaitEvent(cStack->fftPStream, cStack->convComp, 0);

      // Do the FFT
#pragma omp critical
      {
        CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with cnvlStream.");
        CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) cStack->d_plainData, (cufftComplex *) cStack->d_plainData, CUFFT_INVERSE),"Error executing CUFFT plan.");
      }

      // Synchronise
      cudaEventRecord(cStack->plnComp, cStack->fftPStream);
    }
  }

  if ( DBG_PLN02 )  // Print debug info
  {
    for (int ss = 0; ss < plains->noHarms; ss++) // Print
    {
      cuFFdot* cPlain     = &plains->plains[plains->pIdx[ss]];
      printf("\nGPU Post convolve  & FFT h:%i   f: %f\n",ss, cPlain->harmInf->harmFrac);
      printData_cu(plains, plains->flag, plains->pIdx[ss], 10, 1);
      CUDA_SAFE_CALL(cudaStreamSynchronize(0),"");
    }
  }

  if ( DBG_PLTPLN06 ) // Draw the plain
  {
    char fname[1024];
    for (int i = 0; i< plains->noHarms; i++)
    {
      sprintf(fname, "./%08.0f_pln_%02i_%04.02f_GPU.png", plains->rLow, i, plains->hInfos[plains->pIdx[i]].harmFrac);
      drawPlainCmplx(plains->plains[plains->pIdx[i]].d_plainData, fname, plains->hInfos[plains->pIdx[i]].stride, plains->hInfos[plains->pIdx[i]].height );
    }
  }

  plains->haveCData = 1;

  nvtxRangePop();
}
