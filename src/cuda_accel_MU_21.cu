#include "cuda_accel_MU.h"

/** Multiplication kernel - Multiply a stack with a kernel - multi-step - Loop ( Y - Pln - step )  .
 * Each thread loops down a column of the plane
 * Reads the input and multiplies it with the kernel and writes result to plane
 */
template<int64_t FLAGS, int noSteps, int noPlns>
__global__ void mult21_k(const __restrict__ fcomplexcu* kernels, const __restrict__ fcomplexcu* inpData, __restrict__ fcomplexcu* ffdot, const int width, const int stride, const int firstPlane )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;          /// Block ID - flat index
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;      /// Global thread ID - flat index ie column index of stack

  if ( tid < width )  // Valid thread  .
  {
    const int   kerHeight = HEIGHT_HARM[firstPlane];              // The size of the kernel
    fcomplexcu  inpDat[noPlns][noSteps];                          // Set of input data for this thread/column

    int     lDepth  = ceilf(kerHeight/(float)gridDim.y);
    int     y0      = lDepth*blockIdx.y;
    int     y1      = MIN(y0+lDepth, kerHeight);

    FOLD // Stride, kernel, input data & output data  .
    {
      kernels += tid;
      ffdot   += tid;
      inpData += tid;
    }

    FOLD // Read all input data  .
    {
      for ( int step = 0; step < noSteps; step++ )
      {
        for ( int pln = 0; pln < noPlns; pln++ )                  // Loop through the planes  .
        {
          fcomplexcu ipd        = inpData[ (int)(pln*noSteps*stride + step*stride) ];
          ipd.r                 /= (float) width;
          ipd.i                 /= (float) width;
          inpDat[pln][step]     = ipd;
        }
      }
    }

    for ( int kerY = y0; kerY < y1; kerY++ )                      // Loop through the kernel  .
    {
      fcomplexcu ker;                                             // kernel data
      int pHeight = 0;                                            // Height of previous data in the stack

      FOLD // Read the kernel value  .
      {
        ker   = kernels[kerY*stride];
      }

      for (int pln = 0; pln < noPlns; pln++)                      // Loop through the planes  .
      {
        const int plnHeight     = HEIGHT_HARM[firstPlane + pln];
        const int kerYOffset    = (kerHeight - plnHeight)/2;
        const int planeY        = kerY - kerYOffset;
        const int ns2           = plnHeight * stride;

        if( planeY >= 0 && planeY < plnHeight )
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

          for ( int step = 0; step < noSteps; ++step )            // Loop over steps .
          {
            int idx;

            FOLD // Calculate indices  .
            {
              if      ( FLAGS & FLAG_ITLV_ROW )
              {
                idx  = off1 + step * stride;
              }
              else
              {
                idx  = off1 + step * ns2;
              }
            }

            FOLD // Multiply  .
            {
              fcomplexcu ipd = inpDat[pln][step];
              fcomplexcu out;
              out.r = (ipd.r * ker.r + ipd.i * ker.i);
              out.i = (ipd.i * ker.r - ipd.r * ker.i);
              ffdot[idx] = out;
            }
          }
        }

        pHeight += plnHeight * noSteps * stride;                  // Set striding value for next plane
      }
    }
  }
}

template<int64_t FLAGS, int noSteps>
__host__  void mult21_p(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  int offset            = cStack->startIdx;

  switch (cStack->noInStack)
  {
    case 1:
    {
      mult21_k<FLAGS,noSteps,1><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 2:
    {
      mult21_k<FLAGS,noSteps,2><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 3:
    {
      mult21_k<FLAGS,noSteps,3><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 4:
    {
      mult21_k<FLAGS,noSteps,4><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 5:
    {
      mult21_k<FLAGS,noSteps,5><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 6:
    {
      mult21_k<FLAGS,noSteps,6><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 7:
    {
      mult21_k<FLAGS,noSteps,7><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 8:
    {
      mult21_k<FLAGS,noSteps,8><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 9:
    {
      mult21_k<FLAGS,noSteps,9><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    default:
    {
      fprintf(stderr, "ERROR: mult21 has not been templated for %i planes in a stack.\n",cStack->noInStack);
      exit(EXIT_FAILURE);
    }
  }
}

template<int64_t FLAGS>
__host__  void mult21_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  switch (batch->noSteps)
  {
    case 1:
    {
      mult21_p<FLAGS,1>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
    case 2:
    {
      mult21_p<FLAGS,2>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
    case 3:
    {
      mult21_p<FLAGS,3>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
    case 4:
    {
      mult21_p<FLAGS,4>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
    case 5:
    {
      mult21_p<FLAGS,5>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
    case 6:
    {
      mult21_p<FLAGS,6>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
    case 7:
    {
      mult21_p<FLAGS,7>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
    case 8:
    {
      mult21_p<FLAGS,8>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
    default:
    {
      fprintf(stderr, "ERROR: mult21 has not been templated for %i steps\n", batch->noSteps);
      exit(EXIT_FAILURE);
    }
  }
}

__host__  void mult21(cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  dim3 dimGrid, dimBlock;

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY ));
  dimGrid.y = cStack->mulSlices;

  if      ( batch->flags & FLAG_ITLV_ROW )
    mult21_s<FLAG_ITLV_ROW>(dimGrid, dimBlock, 0, multStream, batch, cStack);
  else
    mult21_s<0>(dimGrid, dimBlock, 0, multStream, batch, cStack);
}
