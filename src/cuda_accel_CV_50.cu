#include "cuda_accel_CV.h"

/** Convolution kernel - Convolve an entire batch with convolution kernel  .
 * Each thread loops down a column of the plains and convolves input with kernel and writes result to plain
 */
template<uint FLAGS, int noSteps>
__global__ void convolveffdot50_k(const fcomplexcu* __restrict__ kernels, const fcomplexcu* __restrict__ datas, fcomplexcu* __restrict__ ffdot, int noPlains)
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

          if      ( FLAGS & FLAG_ITLV_ROW )
          {
            *ffdot = val;
            ffdot += stride;  // Stride output pointer to next plain
          }
          else if ( FLAGS & FLAG_ITLV_PLN )
          {
            ffdot[plnOffset + step*PlnStride ] = val;
          }
        }

        // Stride kernel to next row
        ker += stride;
      }

      if ( FLAGS & FLAG_ITLV_PLN ) 	                // Stride output pointer to next plain  .
      {
        ffdot += noSteps*height*stride;
      }
    }
  }
}

template<int FLAGS>
__host__  void convolveffdot50_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, cuFFdotBatch* batch)
{
  switch (batch->noSteps)
  {
    case 1:
    {
      convolveffdot50_k<FLAGS,1><<<dimGrid, dimBlock, i1, cnvlStream>>>(batch->d_kerData , batch->d_iData, batch->d_plainData, batch->noHarms);
      break;
    }
    case 2:
    {
      convolveffdot50_k<FLAGS,2><<<dimGrid, dimBlock, i1, cnvlStream>>>(batch->d_kerData , batch->d_iData, batch->d_plainData, batch->noHarms);
      break;
    }
    case 3:
    {
      convolveffdot50_k<FLAGS,3><<<dimGrid, dimBlock, i1, cnvlStream>>>(batch->d_kerData , batch->d_iData, batch->d_plainData, batch->noHarms);
      break;
    }
    case 4:
    {
      convolveffdot50_k<FLAGS,4><<<dimGrid, dimBlock, i1, cnvlStream>>>(batch->d_kerData , batch->d_iData, batch->d_plainData, batch->noHarms);
      break;
    }
    case 5:
    {
      convolveffdot50_k<FLAGS,5><<<dimGrid, dimBlock, i1, cnvlStream>>>(batch->d_kerData , batch->d_iData, batch->d_plainData, batch->noHarms);
      break;
    }
    case 6:
    {
      convolveffdot50_k<FLAGS,6><<<dimGrid, dimBlock, i1, cnvlStream>>>(batch->d_kerData , batch->d_iData, batch->d_plainData, batch->noHarms);
      break;
    }
    case 7:
    {
      convolveffdot50_k<FLAGS,7><<<dimGrid, dimBlock, i1, cnvlStream>>>(batch->d_kerData , batch->d_iData, batch->d_plainData, batch->noHarms);
      break;
    }
    case 8:
    {
      convolveffdot50_k<FLAGS,8><<<dimGrid, dimBlock, i1, cnvlStream>>>(batch->d_kerData , batch->d_iData, batch->d_plainData, batch->noHarms);
      break;
    }
    default:
    {
      fprintf(stderr, "ERROR: convolveffdot5 has not been templated for %lu steps\n", batch->noSteps);
      exit(EXIT_FAILURE);
    }
  }
}

__host__  void convolveffdot50_f(cudaStream_t cnvlStream, cuFFdotBatch* batch)
{
  dim3 dimGrid, dimBlock;

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil(batch->hInfos[0].width / (float) ( CNV_DIMX * CNV_DIMY ));
  dimGrid.y = 1;

  if      ( batch->flag & FLAG_ITLV_ROW )
    convolveffdot50_s<FLAG_ITLV_ROW>(dimGrid, dimBlock, 0, cnvlStream, batch);
  else if ( batch->flag & FLAG_ITLV_PLN )
    convolveffdot50_s<FLAG_ITLV_PLN>(dimGrid, dimBlock, 0, cnvlStream, batch);
  else
  {
    fprintf(stderr, "ERROR: convolveffdot5 has not been templated for layout.\n");
    exit(EXIT_FAILURE);
  }
}
