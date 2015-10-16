#include "cuda_accel_MU.h"

/** Multiplication kernel - Multiply an entire batch with convolution kernel  .
 * Each thread loops down a column of the planes and multiplies input with kernel and writes result to plane
 */
template<uint FLAGS, int noSteps>
__global__ void mult30_k(const __restrict__ fcomplexcu* kernels, const __restrict__ fcomplexcu* datas, __restrict__ fcomplexcu* ffdot, int noPlanes)
{
  const int ix = blockIdx.x * CNV_DIMX * CNV_DIMY + CNV_DIMX * threadIdx.y + threadIdx.x;

  fcomplexcu input[noSteps];

  // Stride
  ffdot   += ix;
  datas   += ix;

  for (int n = 0; n < noPlanes; n++)                  // Loop over planes  .
  {
    const int stride   = STRIDE_HARM[n];
    const int height   = HEIGHT_HARM[n];
    fcomplexcu* ker    = KERNEL_HARM[n] + ix;

    if ( ix < stride )
    {
      // read input for each step into registers
      for (int step = 0; step < noSteps; step++)      // Loop over planes  .
      {
        input[step]      = datas[step*stride];

        // Normalise
        input[step].r   /= (float) stride ;
        input[step].i   /= (float) stride ;
      }

      // Stride input data
      datas        += stride*noSteps;

      short   lDepth  = ceilf(height/(float)gridDim.y);
      short   y0      = lDepth*blockIdx.y;
      short   y1      = MIN(y0+lDepth, height);

      //for (int iy = 0; iy < height; iy++)           // Loop over individual plane  .
      for (int iy = y0; iy < y1; iy++)              // Loop over individual plane  .
      {
        const int plnOffset = iy*stride;
        const int PlnStride = height*stride;

        // Multiply and write data
        for (int step = 0; step < noSteps; step++)  // Loop over steps  .
        {
          // Multiply
          fcomplexcu val;
          val.r = (input[step].r * ker->r + input[step].i * ker->i);
          val.i = (input[step].i * ker->r - input[step].r * ker->i);

          if      ( FLAGS & FLAG_ITLV_ROW )
          {
            *ffdot = val;
            ffdot += stride;  // Stride output pointer to next plane
          }
          else
          {
            ffdot[plnOffset + step*PlnStride ] = val;
          }
        }

        // Stride kernel to next row
        ker += stride;
      }

      if ( !(FLAGS & FLAG_ITLV_ROW) ) 	                // Stride output pointer to next plane  .
      {
        ffdot += noSteps*height*stride;
      }

    }
  }
}

template<int FLAGS>
__host__  void mult30_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuFFdotBatch* batch)
{
  switch (batch->noSteps)
  {
    case 1:
    {
      mult30_k<FLAGS,1><<<dimGrid, dimBlock, i1, multStream>>>(batch->d_kerData , batch->d_iData, batch->d_planeMult, batch->noHarms);
      break;
    }
    case 2:
    {
      mult30_k<FLAGS,2><<<dimGrid, dimBlock, i1, multStream>>>(batch->d_kerData , batch->d_iData, batch->d_planeMult, batch->noHarms);
      break;
    }
    case 3:
    {
      mult30_k<FLAGS,3><<<dimGrid, dimBlock, i1, multStream>>>(batch->d_kerData , batch->d_iData, batch->d_planeMult, batch->noHarms);
      break;
    }
    case 4:
    {
      mult30_k<FLAGS,4><<<dimGrid, dimBlock, i1, multStream>>>(batch->d_kerData , batch->d_iData, batch->d_planeMult, batch->noHarms);
      break;
    }
    case 5:
    {
      mult30_k<FLAGS,5><<<dimGrid, dimBlock, i1, multStream>>>(batch->d_kerData , batch->d_iData, batch->d_planeMult, batch->noHarms);
      break;
    }
    case 6:
    {
      mult30_k<FLAGS,6><<<dimGrid, dimBlock, i1, multStream>>>(batch->d_kerData , batch->d_iData, batch->d_planeMult, batch->noHarms);
      break;
    }
    case 7:
    {
      mult30_k<FLAGS,7><<<dimGrid, dimBlock, i1, multStream>>>(batch->d_kerData , batch->d_iData, batch->d_planeMult, batch->noHarms);
      break;
    }
    case 8:
    {
      mult30_k<FLAGS,8><<<dimGrid, dimBlock, i1, multStream>>>(batch->d_kerData , batch->d_iData, batch->d_planeMult, batch->noHarms);
      break;
    }
    default:
    {
      fprintf(stderr, "ERROR: mult5 has not been templated for %i steps\n", batch->noSteps);
      exit(EXIT_FAILURE);
    }
  }
}

__host__  void mult30_f(cudaStream_t multStream, cuFFdotBatch* batch)
{
  dim3 dimGrid, dimBlock;

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil(batch->hInfos[0].width / (float) ( CNV_DIMX * CNV_DIMY ));
  dimGrid.y = batch->mulSlices;

  if      ( batch->flag & FLAG_ITLV_ROW )
    mult30_s<FLAG_ITLV_ROW>(dimGrid, dimBlock, 0, multStream, batch);
  else
    mult30_s<0>(dimGrid, dimBlock, 0, multStream, batch);
}
