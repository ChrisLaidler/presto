#include "cuda_accel_CV.h"

/** Convolution kernel - Convolve a stack with a kernel - multi-step  .
 * Each thread loops down a column of the plain
 * Reads the input and convolves it with the kernel and writes result to plain
 */
template<int FLAGS, int noSteps, int noPlns>
__global__ void convolveffdot43_k(const fcomplexcu* __restrict__ kernels, const fcomplexcu* __restrict__ inpData, fcomplexcu* __restrict__ ffdot, const int width, const int stride, const int firstPlain )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;          /// Block ID - flat index
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;      /// Global thread ID - flat index ie column index of stack

  __shared__ fcomplexcu ker_sm[CNV_DIMX*CNV_DIMY*CNV_WORK];

  if ( tid < width )  // Valid thread  .
  {
    int idx;                                      /// flat index of output plain
    int pHeight = 0;                              /// Height of previous data in the stack
    fcomplexcu ker;                               /// kernel data

    const int kerHeight = HEIGHT_FAM_ORDER[firstPlain];       // The size of the kernel
    const int bStride = CNV_DIMX*CNV_DIMY;

    FOLD  // Stride, kernel, input data & output data  .
    {
      kernels += tid;
      ffdot   += tid;
      inpData += tid;
    }

    __restrict__ fcomplexcu inpDat[noSteps][noPlns];          // Set of input data for this thread/column

    FOLD // Read input data for this plain  .
    {
      for (int step = 0; step < noSteps; step++)
      {
        for (int pln = 0; pln < noPlns; pln++)                // Loop through the plains  .
        {
          inpDat[step][pln]     = inpData[ (int)(pln*noSteps*stride + step*stride) ];
          inpDat[step][pln].r   /= (float) width;
          inpDat[step][pln].i   /= (float) width;
        }
      }
    }

    int noChunks = ceil(kerHeight / (float)CNV_WORK);

    for (int chunk = 0; chunk < noChunks; chunk++ )
    {
      const int c0  = chunk*CNV_WORK;
      const int c1  = MIN(CNV_WORK,kerHeight-c0);
      pHeight       = 0;

      /*
      if ( tid == 0 )
      {
        printf("\n");
        printf("chunk %02i  c0: %02i  len %i  \n", chunk, c0, c1);
      }
      */

      FOLD // Load kernel data into SM
      {
        for( int c = 0; c < c1; c++ )
        {
          ker_sm[c*bStride + bidx] = kernels[(c0+c)*stride];
        }
      }

      for (int pln = 0; pln < noPlns; pln++)                    // Loop through the plains  .
      {
        const int plnHeight     = HEIGHT_FAM_ORDER[firstPlain + pln];
        const int kerYOffset    = (kerHeight - plnHeight)/2;
        const int ns2           = plnHeight * stride;

        const int p0            = MAX(c0 - kerYOffset,0);
        const int p1            = MIN(c0 + c1 - kerYOffset, plnHeight);
        const int kerAddd       = MAX(0,c1-p1);

        /*
        if ( tid == 0 )
        {
          if ( p1 > p0 )
            printf("pln %02i  kerYOffset: %02i  kerAddd %02i  p0 %03i  p1 %03i \n", pln, kerYOffset, kerAddd, p0, p1 );
        }
        */

        //Fout
        {

        for (int plainY = p0; plainY < p1; plainY++)            // Loop over the individual plain  .
        {
          int offsetPart1;

          FOLD // Read the kernel value  .
          {
            //ker   = kernels[(kerYOffset+plainY)*stride];
            ker   = ker_sm[(plainY-p0+kerAddd)*bStride + bidx];
          }

          FOLD // Calculate partial offset  .
          {
            if      ( FLAGS & FLAG_STP_ROW )
            {
              offsetPart1  = pHeight + plainY*noSteps*stride;
            }
            else if ( FLAGS & FLAG_STP_PLN )
            {
              offsetPart1  = pHeight + plainY*stride;
            }
          }

          for ( int step = 0; step < noSteps; ++step )          // Loop over steps .
          {
            FOLD // Calculate offset  .
            {
              if      ( FLAGS & FLAG_STP_ROW )
              {
                idx  = offsetPart1 + step * stride;
              }
              else if ( FLAGS & FLAG_STP_PLN )
              {
                idx  = offsetPart1 + step * ns2;
              }
            }

            FOLD // Convolve  .
            {
              ffdot[idx].r = (inpDat[step][pln].r * ker.r + inpDat[step][pln].i * ker.i);
              ffdot[idx].i = (inpDat[step][pln].i * ker.r - inpDat[step][pln].r * ker.i);
            }
          }
        }

        pHeight += plnHeight * noSteps * stride;
      }

      }
    }
  }
}

template<int FLAGS, int noSteps>
__host__  void convolveffdot43_p(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack)
{
  int offset = 0;
  for ( int i = 0; i < stack; i++)
  {
    offset += batch->stacks[i].noInStack;
  }
  cuFfdotStack* cStack = &batch->stacks[stack];

  switch (cStack->noInStack)
  {
    case 1:
    {
      convolveffdot43_k<FLAGS,noSteps,1><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->inpStride, offset);
      break;
    }
    case 2:
    {
      convolveffdot43_k<FLAGS,noSteps,2><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->inpStride, offset);
      break;
    }
    case 3:
    {
      convolveffdot43_k<FLAGS,noSteps,3><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->inpStride, offset);
      break;
    }
    case 4:
    {
      convolveffdot43_k<FLAGS,noSteps,4><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->inpStride, offset);
      break;
    }
    case 5:
    {
      convolveffdot43_k<FLAGS,noSteps,5><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->inpStride, offset);
      break;
    }
    case 6:
    {
      convolveffdot43_k<FLAGS,noSteps,6><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->inpStride, offset);
      break;
    }
    case 7:
    {
      convolveffdot43_k<FLAGS,noSteps,7><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->inpStride, offset);
      break;
    }
    case 8:
    {
      convolveffdot43_k<FLAGS,noSteps,8><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->inpStride, offset);
      break;
    }
    default:
    {
      fprintf(stderr, "ERROR: convolveffdot43 has not been templated for %i plains in a stack.\n",cStack->noInStack);
      exit(EXIT_FAILURE);
    }
  }
}

template<int FLAGS>
__host__  void convolveffdot43_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack)
{

  switch (batch->noSteps)
  {
    case 1:
    {
      convolveffdot43_p<FLAGS,1>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 2:
    {
      convolveffdot43_p<FLAGS,2>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 3:
    {
      convolveffdot43_p<FLAGS,3>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 4:
    {
      convolveffdot43_p<FLAGS,4>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 5:
    {
      convolveffdot43_p<FLAGS,5>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 6:
    {
      convolveffdot43_p<FLAGS,6>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 7:
    {
      convolveffdot43_p<FLAGS,7>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 8:
    {
      convolveffdot43_p<FLAGS,8>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    default:
    {
      fprintf(stderr, "ERROR: convolveffdot42 has not been templated for %lu steps\n", batch->noSteps);
      exit(EXIT_FAILURE);
    }
  }
}

__host__  void convolveffdot43_f(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack)
{
  if      ( batch->flag & FLAG_STP_ROW )
    convolveffdot43_s<FLAG_STP_ROW>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
  else if ( batch->flag & FLAG_STP_PLN )
    convolveffdot43_s<FLAG_STP_PLN>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
  else
  {
    fprintf(stderr, "ERROR: convolveffdot43 has not been templated for layout.\n");
    exit(EXIT_FAILURE);
  }
}
