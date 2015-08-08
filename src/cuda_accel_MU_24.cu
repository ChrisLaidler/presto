#include "cuda_accel_MU.h"

/** Multiplication kernel - All multiplications for a stack - Uses registers to store sections of the kernel - Loop ( chunk (read ker) - plan - Y - step ) .
 * Each thread loops down a column of the plain
 * Reads the input and multiples it with the kernel and writes result to plain
 */
template<int FLAGS, int noSteps, int noPlns, int chunkSZ>
__global__ void mult24_k(const __restrict__ fcomplexcu* kernels, const __restrict__ fcomplexcu* inpData, __restrict__ fcomplexcu* ffdot, const int width, const int stride, const int firstPlain )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;          /// Block ID - flat index
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;      /// Global thread ID - flat index ie column index of stack

  if ( tid < width )  // Valid thread  .
  {
    const int kerHeight = HEIGHT_HARM[firstPlain];       // The size of the kernel

    register fcomplexcu  kers[chunkSZ];

    FOLD  // Stride, kernel, input data & output data  .
    {
      kernels += tid;
      ffdot   += tid;
      inpData += tid;
    }

    short   lDepth  = ceilf(kerHeight/(float)gridDim.y);
    short   y0      = lDepth*blockIdx.y;
    short   y1      = MIN(y0+lDepth, kerHeight);

    for ( int y = y0; y < y1; y+=chunkSZ )
    {
      const int c1  = MIN(chunkSZ,kerHeight-y);

      int pHeight   = 0;

      FOLD // Read kernel  .
      {
        for ( int i = 0; i < c1 ; i++)
        {
          kers[i]= kernels[(y+i)*stride];
        }
      }

      for (int pln = 0; pln < noPlns; pln++)                  // Loop through the plains of the stack  .
      {
        const int plnHeight     = HEIGHT_HARM[firstPlain + pln];
        const int kerYOffset    = (kerHeight - plnHeight)/2;

        const int yP0           = MAX(y - kerYOffset,0);
        const int yP1           = MIN(y + c1 - kerYOffset, plnHeight);

        const int kerAddd       = MAX(0, kerYOffset - y);

        const int ns2           = plnHeight * stride;

        __restrict__ fcomplexcu inpDat[noSteps];              // Set of input data for this thread/column

        FOLD // Read all input data  .
        {
          // NOTE: I tested reading the input for plains and steps (2 loops above) but that was slower, here uses less registers as well.

          for (int step = 0; step < noSteps; step++)
          {
            fcomplexcu ipd        = inpData[ (int)(pln*noSteps*stride + step*stride) ];
            ipd.r                 /= (float) width;
            ipd.i                 /= (float) width;
            inpDat[step]     = ipd;
          }
        }

        for (int plainY = yP0; plainY < yP1; plainY++)          // Loop over the individual plain  .
        {
          int cy = plainY - yP0 + kerAddd;
          fcomplexcu ker;

          FOLD // Read the kernel value  .
          {
            ker = kers[cy];
          }

          int offsetPart1;

          FOLD // Calculate partial offset  .
          {
            if      ( FLAGS & FLAG_ITLV_ROW )
            {
              offsetPart1  = pHeight + plainY*noSteps*stride;
            }
            else if ( FLAGS & FLAG_ITLV_PLN )
            {
              offsetPart1  = pHeight + plainY*stride;
            }
          }

          for ( int step = 0; step < noSteps; step++ )        // Loop over steps .
          {
            int idx;

            FOLD // Calculate offset  .
            {
              if      ( FLAGS & FLAG_ITLV_ROW )
              {
                idx  = offsetPart1 + step * stride;
              }
              else if ( FLAGS & FLAG_ITLV_PLN )
              {
                idx  = offsetPart1 + step * ns2;
              }
            }

            FOLD // Multiply  .
            {
              fcomplexcu ipd = inpDat[step];
              fcomplexcu out;
              out.r = (ipd.r * ker.r + ipd.i * ker.i);
              out.i = (ipd.i * ker.r - ipd.r * ker.i);
              ffdot[idx] = out;
            }
          }
        }

        pHeight += plnHeight * noSteps * stride;              // Set striding value for next plain
      }
    }
  }
}

template<int FLAGS, int noSteps, int chunkSZ>
__host__  void mult24_p(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t stream, cuFFdotBatch* batch, uint stack)
{
  cuFfdotStack* cStack  = &batch->stacks[stack];
  int offset            = cStack->startIdx;

  switch (cStack->noInStack)
  {
    case 1	:
    {
      //cudaFuncSetCacheConfig(mult24_k<FLAGS,noSteps,1>, cudaFuncCachePreferL1);
      mult24_k<FLAGS,noSteps,1, chunkSZ><<<dimGrid, dimBlock, i1, stream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 2	:
    {
      //cudaFuncSetCacheConfig(mult24_k<FLAGS,noSteps,2>, cudaFuncCachePreferL1);
      mult24_k<FLAGS,noSteps,2, chunkSZ><<<dimGrid, dimBlock, i1, stream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 3	:
    {
      //cudaFuncSetCacheConfig(mult24_k<FLAGS,noSteps,3>, cudaFuncCachePreferL1);
      mult24_k<FLAGS,noSteps,3, chunkSZ><<<dimGrid, dimBlock, i1, stream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 4	:
    {
      //cudaFuncSetCacheConfig(mult24_k<FLAGS,noSteps,4>, cudaFuncCachePreferL1);
      mult24_k<FLAGS,noSteps,4, chunkSZ><<<dimGrid, dimBlock, i1, stream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 5	:
    {
      //cudaFuncSetCacheConfig(mult24_k<FLAGS,noSteps,5>, cudaFuncCachePreferL1);
      mult24_k<FLAGS,noSteps,5, chunkSZ><<<dimGrid, dimBlock, i1, stream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 6	:
    {
      //cudaFuncSetCacheConfig(mult24_k<FLAGS,noSteps,6>, cudaFuncCachePreferL1);
      mult24_k<FLAGS,noSteps,6, chunkSZ><<<dimGrid, dimBlock, i1, stream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 7	:
    {
      //cudaFuncSetCacheConfig(mult24_k<FLAGS,noSteps,7>, cudaFuncCachePreferL1);
      mult24_k<FLAGS,noSteps,7, chunkSZ><<<dimGrid, dimBlock, i1, stream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 8	:
    {
      //cudaFuncSetCacheConfig(mult24_k<FLAGS,noSteps,8>, cudaFuncCachePreferL1);
      mult24_k<FLAGS,noSteps,8, chunkSZ><<<dimGrid, dimBlock, i1, stream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 9	:
    {
      //cudaFuncSetCacheConfig(mult24_k<FLAGS,noSteps,9>, cudaFuncCachePreferL1);
      mult24_k<FLAGS,noSteps,9, chunkSZ><<<dimGrid, dimBlock, i1, stream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    default	:
    {
      fprintf(stderr, "ERROR: mult24 has not been templated for %i plains in a stack.\n",cStack->noInStack);
      exit(EXIT_FAILURE);
    }
  }
}

template<int FLAGS, int noSteps>
__host__  void mult21_c(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t stream, cuFFdotBatch* batch, uint stack)
{
  switch (globalInt01)
  {
    case 1:
    {
      mult24_p<FLAGS, noSteps,1>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 2:
    {
      mult24_p<FLAGS, noSteps,2>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
//    case 3:
//    {
//      mult24_p<FLAGS, noSteps,3>(dimGrid, dimBlock, i1, stream, batch, stack);
//      break;
//    }
    case 4:
    {
      mult24_p<FLAGS, noSteps,4>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
//    case 5:
//    {
//      mult24_p<FLAGS, noSteps,5>(dimGrid, dimBlock, i1, stream, batch, stack);
//      break;
//    }
    case 6:
    {
      mult24_p<FLAGS, noSteps,6>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
//    case 7:
//    {
//      mult24_p<FLAGS, noSteps,7>(dimGrid, dimBlock, i1, stream, batch, stack);
//      break;
//    }
    case 8:
    {
      mult24_p<FLAGS, noSteps,8>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
//    case 9:
//    {
//      mult24_p<FLAGS, noSteps,9>(dimGrid, dimBlock, i1, stream, batch, stack);
//      break;
//    }
    case 10:
    {
      mult24_p<FLAGS, noSteps,10>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 12:
    {
      mult24_p<FLAGS, noSteps,12>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 14:
    {
      mult24_p<FLAGS, noSteps,14>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 16:
    {
      mult24_p<FLAGS, noSteps,16>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 18:
    {
      mult24_p<FLAGS, noSteps,18>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 20:
    {
      mult24_p<FLAGS, noSteps,20>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 24:
    {
      mult24_p<FLAGS, noSteps,24>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    default:
      fprintf(stderr, "ERROR: %s has not been templated for %i chunk size.\n", __FUNCTION__, globalInt01);
      exit(EXIT_FAILURE);
  }
}

template<int FLAGS>
__host__  void mult24_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t stream, cuFFdotBatch* batch, uint stack)
{
  switch (batch->noSteps)
  {
    case 1	:
    {
      mult21_c<FLAGS,1>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 2	:
    {
      mult21_c<FLAGS,2>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 3	:
    {
      mult21_c<FLAGS,3>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 4	:
    {
      mult21_c<FLAGS,4>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 5	:
    {
      mult21_c<FLAGS,5>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 6	:
    {
      mult21_c<FLAGS,6>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 7	:
    {
      mult21_c<FLAGS,7>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 8	:
    {
      mult21_c<FLAGS,8>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    default	:
    {
      fprintf(stderr, "ERROR: convolveffdot42 has not been templated for %i steps\n", batch->noSteps);
      exit(EXIT_FAILURE);
    }
  }
}

__host__  void mult24(cudaStream_t stream, cuFFdotBatch* batch, uint stack)
{
  dim3 dimGrid, dimBlock;

  cuFfdotStack* cStack = &batch->stacks[stack];

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY ));
  dimGrid.y = cStack->noMulSlices;


  if      ( batch->flag & FLAG_ITLV_ROW )
    mult24_s<FLAG_ITLV_ROW>(dimGrid, dimBlock, 0, stream, batch, stack);
  else if ( batch->flag & FLAG_ITLV_PLN )
    mult24_s<FLAG_ITLV_PLN>(dimGrid, dimBlock, 0, stream, batch, stack);
  else
  {
    fprintf(stderr, "ERROR: mult24 has not been templated for layout.\n");
    exit(EXIT_FAILURE);
  }
}
