#include "cuda_accel_CV.h"

/** Convolution kernel - Convolve an entire batch with convolution kernel  .
 * Each thread loops down a column of the plains and convolves input with kernel and writes result to plain
 */
/*
template<uint FLAGS, typename hType, int noHars>
__global__ void convolveffdot5_ok(const fcomplexcu *kernels, const fcomplexcu *datas, fcomplexcu *ffdot, iHarmList widths, iHarmList strides, iHarmList heights, uint no, uint hh, uint noSteps, uint step )
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
*/


template<uint FLAGS, int noSteps>
__global__ void convolveffdot5_k(const fcomplexcu* __restrict__ kernels, const fcomplexcu* __restrict__ datas, fcomplexcu* __restrict__ ffdot, int noPlains)
{
  const int ix = blockIdx.x * CNV_DIMX * CNV_DIMY + CNV_DIMX * threadIdx.y + threadIdx.x;

  fcomplexcu input[noSteps];

  // Stride
  ffdot   += ix;
  datas   += ix;

  for (int n = 0; n < noPlains; n++)                  // Loop over plains  .
  {
    const int stride   = STRIDE_FAM_ORDER[n];
    const int height   = HEIGHT_FAM_ORDER[n];
    fcomplexcu* ker    = KERNEL_FAM_ORDER[n] + ix;

    if ( ix < stride )
    {
      // read input for each step into registers
      for (int step = 0; step < noSteps; step++)      // Loop over plains  .
      {
        input[step]      = datas[step*stride];

        // Normalise
        input[step].r   /= (float) stride ;
        input[step].i   /= (float) stride ;
      }

      // Stride input data
      datas        += stride*noSteps;

      for (int iy = 0; iy < height; iy++)           // Loop over individual plain  .
      {
        const int plnOffset = iy*stride;
        const int PlnStride = height*stride;

        // Convolve and write data
        for (int step = 0; step < noSteps; step++)  // Loop over steps  .
        {
          // Convolve
          fcomplexcu val;
          val.r = (input[step].r * ker->r + input[step].i * ker->i);
          val.i = (input[step].i * ker->r - input[step].r * ker->i);

          if      ( FLAGS & FLAG_STP_ROW )
          {
            *ffdot = val;
            ffdot += stride;  // Stride output pointer to next plain
          }
          else if ( FLAGS & FLAG_STP_PLN )
          {
            ffdot[plnOffset + step*PlnStride ] = val;
          }
        }

        // Stride kernel to next row
        ker += stride;
      }

      if ( FLAGS & FLAG_STP_PLN ) 	                // Stride output pointer to next plain  .
      {
        ffdot += noSteps*height*stride;
      }
    }
  }
}

template<int FLAGS>
__host__  void convolveffdot5_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, cuFFdotBatch* batch)
{
  switch (batch->noSteps)
  {
    case 1:
    {
      convolveffdot5_k<FLAGS,1><<<dimGrid, dimBlock, i1, cnvlStream>>>(batch->d_kerData , batch->d_iData, batch->d_plainData, batch->noHarms);
      break;
    }
    case 2:
    {
      convolveffdot5_k<FLAGS,2><<<dimGrid, dimBlock, i1, cnvlStream>>>(batch->d_kerData , batch->d_iData, batch->d_plainData, batch->noHarms);
      break;
    }
    case 3:
    {
      convolveffdot5_k<FLAGS,3><<<dimGrid, dimBlock, i1, cnvlStream>>>(batch->d_kerData , batch->d_iData, batch->d_plainData, batch->noHarms);
      break;
    }
    case 4:
    {
      convolveffdot5_k<FLAGS,4><<<dimGrid, dimBlock, i1, cnvlStream>>>(batch->d_kerData , batch->d_iData, batch->d_plainData, batch->noHarms);
      break;
    }
    case 5:
    {
      convolveffdot5_k<FLAGS,5><<<dimGrid, dimBlock, i1, cnvlStream>>>(batch->d_kerData , batch->d_iData, batch->d_plainData, batch->noHarms);
      break;
    }
    case 6:
    {
      convolveffdot5_k<FLAGS,6><<<dimGrid, dimBlock, i1, cnvlStream>>>(batch->d_kerData , batch->d_iData, batch->d_plainData, batch->noHarms);
      break;
    }
    case 7:
    {
      convolveffdot5_k<FLAGS,7><<<dimGrid, dimBlock, i1, cnvlStream>>>(batch->d_kerData , batch->d_iData, batch->d_plainData, batch->noHarms);
      break;
    }
    case 8:
    {
      convolveffdot5_k<FLAGS,8><<<dimGrid, dimBlock, i1, cnvlStream>>>(batch->d_kerData , batch->d_iData, batch->d_plainData, batch->noHarms);
      break;
    }
    default:
    {
      fprintf(stderr, "ERROR: convolveffdot5 has not been templated for %lu steps\n", batch->noSteps);
      exit(EXIT_FAILURE);
    }
  }
}

__host__  void convolveffdot5_f(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, cuFFdotBatch* batch)
{
  if      ( batch->flag & FLAG_STP_ROW )
    convolveffdot5_s<FLAG_STP_ROW>(dimGrid, dimBlock, i1, cnvlStream, batch);
  else if ( batch->flag & FLAG_STP_PLN )
    convolveffdot5_s<FLAG_STP_PLN>(dimGrid, dimBlock, i1, cnvlStream, batch);
  else
  {
    fprintf(stderr, "ERROR: convolveffdot5 has not been templated for layout.\n");
    exit(EXIT_FAILURE);
  }
}
