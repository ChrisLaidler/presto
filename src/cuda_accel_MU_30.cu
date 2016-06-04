#include "cuda_accel_MU.h"

/** Multiplication kernel - Multiply an entire batch with convolution kernel  .
 * Each thread loops down a column of the planes and multiplies input with kernel and writes result to plane
 */
template<int64_t FLAGS, int noSteps>
__global__ void mult30_k(const __restrict__ fcomplexcu* kernels, const __restrict__ fcomplexcu* datas, __restrict__ fcomplexcu* ffdot, int noPlanes)
{
  const int ix = blockIdx.x * CNV_DIMX * CNV_DIMY + CNV_DIMX * threadIdx.y + threadIdx.x;

  fcomplexcu input[noSteps];

  // Stride
  ffdot   += ix;
  datas   += ix;

  int pHeight = 0;

  for (int n = 0; n < noPlanes; n++)                  // Loop over planes  .
  {
    const int stride      = STRIDE_HARM[n];

    if ( ix < stride )
    {
      const int plnHeight = HEIGHT_HARM[n];
      const int plnStride = plnHeight*stride;
      const short lDepth  = ceilf(plnHeight/(float)gridDim.y);
      const short y0      = lDepth*blockIdx.y;
      const short y1      = MIN(y0+lDepth, plnHeight);
      fcomplexcu* ker     = (fcomplexcu*)KERNEL_HARM[n] + y0 * stride + ix;

      // read input for each step into registers
      for (int step = 0; step < noSteps; step++)      // Loop over planes  .
      {
        input[step]       = datas[step*stride];

        // Normalise
        input[step].r    /= (float) stride ;
        input[step].i    /= (float) stride ;
      }

      // Stride input data
      datas              += stride*noSteps;

      for (int planeY = y0; planeY < y1; planeY++)              // Loop over individual plane  .
      {
        int off1;
        FOLD // Calculate partial offset  .
        {
          if      ( FLAGS & FLAG_ITLV_ROW )
          {
            off1  = pHeight + planeY*noSteps*stride;
          }
          else
          {
            off1  = pHeight + planeY*stride;
          }
        }

        // Multiply and write data
        for (int step = 0; step < noSteps; step++)  // Loop over steps  .
        {
          // 
          fcomplexcu out;
          fcomplexcu ipd = input[step];

          // Calculate index
          int idx = 0;
          FOLD // Calculate indices  .
          {
            if      ( FLAGS & FLAG_ITLV_ROW )
            {
              idx  = off1 + step * stride;
            }
            else
            {
              idx  = off1 + step * plnStride;
            }
          }

	  // Multiply
#if CORRECT_MULT
          // This is the "correct" version
          out.r = (ipd.r * ker->r - ipd.i * ker->i);
          out.i = (ipd.r * ker->i + ipd.i * ker->r);
#else
          // This is the version accelsearch uses, ( added for comparison )
          out.r = (ipd.r * ker->r + ipd.i * ker->i);
          out.i = (ipd.i * ker->r - ipd.r * ker->i);
#endif

          // Write the actual value
          ffdot[idx] = out;
        }

        // Stride kernel to next row
        ker += stride;
      }
      
      // Track plane offsett
      pHeight += noSteps*plnHeight*stride;
    }
  }
}

template<int64_t FLAGS>
__host__  void mult30_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuFFdotBatch* batch)
{
  switch (batch->noSteps)
  {
    case 1:
    {
      mult30_k<FLAGS,1><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)batch->d_kerData , batch->d_iData, (fcomplexcu*)batch->d_planeMult, batch->noGenHarms);
      break;
    }
    case 2:
    {
      mult30_k<FLAGS,2><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)batch->d_kerData , batch->d_iData, (fcomplexcu*)batch->d_planeMult, batch->noGenHarms);
      break;
    }
    case 3:
    {
      mult30_k<FLAGS,3><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)batch->d_kerData , batch->d_iData, (fcomplexcu*)batch->d_planeMult, batch->noGenHarms);
      break;
    }
    case 4:
    {
      mult30_k<FLAGS,4><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)batch->d_kerData , batch->d_iData, (fcomplexcu*)batch->d_planeMult, batch->noGenHarms);
      break;
    }
    case 5:
    {
      mult30_k<FLAGS,5><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)batch->d_kerData , batch->d_iData, (fcomplexcu*)batch->d_planeMult, batch->noGenHarms);
      break;
    }
    case 6:
    {
      mult30_k<FLAGS,6><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)batch->d_kerData , batch->d_iData, (fcomplexcu*)batch->d_planeMult, batch->noGenHarms);
      break;
    }
    case 7:
    {
      mult30_k<FLAGS,7><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)batch->d_kerData , batch->d_iData, (fcomplexcu*)batch->d_planeMult, batch->noGenHarms);
      break;
    }
    case 8:
    {
      mult30_k<FLAGS,8><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)batch->d_kerData , batch->d_iData, (fcomplexcu*)batch->d_planeMult, batch->noGenHarms);
      break;
    }
    default:
    {
      fprintf(stderr, "ERROR: mult5 has not been templated for %i steps\n", batch->noSteps);
      exit(EXIT_FAILURE);
    }
  }
}

__host__  void mult30(cudaStream_t multStream, cuFFdotBatch* batch)
{
  dim3 dimGrid, dimBlock;

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil(batch->hInfos[0].width / (float) ( CNV_DIMX * CNV_DIMY ));
  dimGrid.y = batch->mulSlices;

  if      ( batch->flags & FLAG_ITLV_ROW )
    mult30_s<FLAG_ITLV_ROW>(dimGrid, dimBlock, 0, multStream, batch);
  else
    mult30_s<0>(dimGrid, dimBlock, 0, multStream, batch);
}
