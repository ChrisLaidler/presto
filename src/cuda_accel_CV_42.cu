#include "cuda_accel_CV.h"

/** Convolution kernel - Convolve a stack with a kernel - multi-step  .
 * Each thread loops down a column of the plain
 * Reads the input and convolves it with the kernel and writes result to plain
 */
template<int FLAGS, int noSteps>
__global__ void convolveffdot42_k(const fcomplexcu* __restrict__ kernels, const fcomplexcu* __restrict__ inpData, fcomplexcu* __restrict__ ffdot, const int width, const int stride, int noPlns, const int firstPlain )
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

    __restrict__ fcomplexcu inpDat[noSteps];                  // Set of input data for this thread/column

    for (int pln = 0; pln < noPlns; pln++)                    // Loop through the plains  .
    {
      const int plnStrd = pln*stride*noSteps;

      // Read input data for this plain
      for (int step = 0; step < noSteps; step++)
      {
        inpDat[step]           = inpData[ (int)(plnStrd + step*stride) ];
        inpDat[step].r        /= (float) width;
        inpDat[step].i        /= (float) width;
      }

      const int plnHeight     = HEIGHT_FAM_ORDER[firstPlain + pln];
      const int kerYOffset    = (HEIGHT_FAM_ORDER[firstPlain] - plnHeight)/2;
      const int ns2           = plnHeight * stride;

      for (int plainY = 0; plainY < plnHeight; plainY++)      // Loop over the individual plain  .
      {

        FOLD // Read the kernel value  .
        {
          ker   = kernels[(kerYOffset+plainY)*stride];
        }

        int off1;

        // Calculate partial offset
        if      ( FLAGS & FLAG_STP_ROW )
        {
          off1  = pHeight + plainY*noSteps*stride;
        }
        else if ( FLAGS & FLAG_STP_PLN )
        {
          off1  = pHeight + plainY*stride;
        }

        for ( int step = 0; step < noSteps; ++step )          // Loop over steps .
        {
          FOLD // Calculate indices  .
          {
            if      ( FLAGS & FLAG_STP_ROW )
            {
              idx  = off1 + step * stride;
            }
            else if ( FLAGS & FLAG_STP_PLN )
            {
              idx  = off1 + step * ns2;
            }
          }

          // Convolve
          ffdot[idx].r = (inpDat[step].r * ker.r + inpDat[step].i * ker.i);
          ffdot[idx].i = (inpDat[step].i * ker.r - inpDat[step].r * ker.i);
        }
      }

      pHeight += plnHeight * noSteps * stride;
    }
  }
}

template<int FLAGS>
__host__  void convolveffdot42_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack)
{
  int offset = 0;
  for ( int i = 0; i < stack; i++)
  {
    offset += batch->stacks[i].noInStack;
  }
  cuFfdotStack* cStack = &batch->stacks[stack];

  switch (batch->noSteps)
  {
    case 1:
    {
      convolveffdot42_k<FLAGS,1><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->inpStride, cStack->noInStack, offset);
      break;
    }
    case 2:
    {
      convolveffdot42_k<FLAGS,2><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->inpStride, cStack->noInStack, offset);
      break;
    }
    case 3:
    {
      convolveffdot42_k<FLAGS,3><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->inpStride, cStack->noInStack, offset);
      break;
    }
    case 4:
    {
      convolveffdot42_k<FLAGS,4><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->inpStride, cStack->noInStack, offset);
      break;
    }
    case 5:
    {
      convolveffdot42_k<FLAGS,5><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->inpStride, cStack->noInStack, offset);
      break;
    }
    case 6:
    {
      convolveffdot42_k<FLAGS,6><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->inpStride, cStack->noInStack, offset);
      break;
    }
    case 7:
    {
      convolveffdot42_k<FLAGS,7><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->inpStride, cStack->noInStack, offset);
      break;
    }
    case 8:
    {
      convolveffdot42_k<FLAGS,8><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->inpStride, cStack->noInStack, offset);
      break;
    }
    default:
    {
      fprintf(stderr, "ERROR: convolveffdot42 has not been templated for %lu steps\n", batch->noSteps);
      exit(EXIT_FAILURE);
    }
  }
}

__host__  void convolveffdot42_f(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack)
{
  if      ( batch->flag & FLAG_STP_ROW )
    convolveffdot42_s<FLAG_STP_ROW>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
  else if ( batch->flag & FLAG_STP_PLN )
    convolveffdot42_s<FLAG_STP_PLN>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
  else
  {
    fprintf(stderr, "ERROR: convolveffdot42 has not been templated for layout.\n");
    exit(EXIT_FAILURE);
  }
}
