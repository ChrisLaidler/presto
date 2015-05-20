#include "cuda_accel_CV.h"


/** Convolution kernel - Convolve a stack with a kernel - multi-step - Loop ( Y - Pln - step )  .
 * Each thread loops down a column of the plain
 * Reads the input and convolves it with the kernel and writes result to plain
 */
template<int FLAGS, int noSteps, int noPlns>
__global__ void convolveffdot42_k(const fcomplexcu* __restrict__ kernels, const fcomplexcu* __restrict__ inpData, fcomplexcu* __restrict__ ffdot, const int width, const int stride, const int firstPlain )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;          /// Block ID - flat index
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;      /// Global thread ID - flat index ie column index of stack

  if ( tid < width )  // Valid thread  .
  {
    int idx;                                      /// flat index of output plain
    int pHeight = 0;                              /// Height of previous data in the stack
    fcomplexcu ker;                               /// kernel data

    FOLD  // Stride, kernel, input data & output data  .
    {
      kernels += tid;
      ffdot   += tid;
      inpData += tid;
    }

    __restrict__ fcomplexcu inpDat[noSteps][noPlns];          // Set of input data for this thread/column
    const int kerHeight = HEIGHT_FAM_ORDER[firstPlain];       // The size of the kernel

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

    for (int y = 0; y < kerHeight; y++)                       // Loop through the kernel .
    {
      FOLD // Read the kernel value  .
      {
        ker   = kernels[y*stride];
      }

      pHeight = 0;

      for (int pln = 0; pln < noPlns; pln++)                  // Loop through the plains  .
      {
        const int plnHeight     = HEIGHT_FAM_ORDER[firstPlain + pln];
        const int kerYOffset    = (kerHeight - plnHeight)/2;
        const int plainY        = y - kerYOffset;
        const int ns2           = plnHeight * stride;
        int off1;

        if( plainY >= 0 && plainY < plnHeight )
        {
          FOLD // Calculate partial offset  .
          {
            if      ( FLAGS & FLAG_ITLV_ROW )
            {
              off1  = pHeight + plainY*noSteps*stride;
            }
            else if ( FLAGS & FLAG_ITLV_PLN )
            {
              off1  = pHeight + plainY*stride;
            }
          }

          for ( int step = 0; step < noSteps; ++step )        // Loop over steps .
          {
            FOLD // Calculate indices  .
            {
              if      ( FLAGS & FLAG_ITLV_ROW )
              {
                idx  = off1 + step * stride;
              }
              else if ( FLAGS & FLAG_ITLV_PLN )
              {
                idx  = off1 + step * ns2;
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

template<int FLAGS, int noSteps>
__host__  void convolveffdot42_p(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack)
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
      convolveffdot42_k<FLAGS,noSteps,1><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 2:
    {
      convolveffdot42_k<FLAGS,noSteps,2><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 3:
    {
      convolveffdot42_k<FLAGS,noSteps,3><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 4:
    {
      convolveffdot42_k<FLAGS,noSteps,4><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 5:
    {
      convolveffdot42_k<FLAGS,noSteps,5><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 6:
    {
      convolveffdot42_k<FLAGS,noSteps,6><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 7:
    {
      convolveffdot42_k<FLAGS,noSteps,7><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 8:
    {
      convolveffdot42_k<FLAGS,noSteps,8><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 9:
    {
      convolveffdot42_k<FLAGS,noSteps,9><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    default:
    {
      fprintf(stderr, "ERROR: convolveffdot42 has not been templated for %i plains in a stack.\n",cStack->noInStack);
      exit(EXIT_FAILURE);
    }
  }
}

template<int FLAGS>
__host__  void convolveffdot42_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack)
{

  switch (batch->noSteps)
  {
    case 1:
    {
      convolveffdot42_p<FLAGS,1>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 2:
    {
      convolveffdot42_p<FLAGS,2>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 3:
    {
      convolveffdot42_p<FLAGS,3>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 4:
    {
      convolveffdot42_p<FLAGS,4>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 5:
    {
      convolveffdot42_p<FLAGS,5>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 6:
    {
      convolveffdot42_p<FLAGS,6>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 7:
    {
      convolveffdot42_p<FLAGS,7>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 8:
    {
      convolveffdot42_p<FLAGS,8>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    default:
    {
      fprintf(stderr, "ERROR: convolveffdot42 has not been templated for %lu steps\n", batch->noSteps);
      exit(EXIT_FAILURE);
    }
  }
}

__host__  void convolveffdot42_f(cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack)
{
  dim3 dimGrid, dimBlock;

  cuFfdotStack* cStack = &batch->stacks[stack];

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY ));
  dimGrid.y = 1;

  if      ( batch->flag & FLAG_ITLV_ROW )
    convolveffdot42_s<FLAG_ITLV_ROW>(dimGrid, dimBlock, 0, cnvlStream, batch, stack);
  else if ( batch->flag & FLAG_ITLV_PLN )
    convolveffdot42_s<FLAG_ITLV_PLN>(dimGrid, dimBlock, 0, cnvlStream, batch, stack);
  else
  {
    fprintf(stderr, "ERROR: convolveffdot42 has not been templated for layout.\n");
    exit(EXIT_FAILURE);
  }
}
