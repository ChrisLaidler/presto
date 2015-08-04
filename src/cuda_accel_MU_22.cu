#include "cuda_accel_MU.h"

/** Multiplication kernel - Multiply a stack with a kernel - multi-step - Loop ( Pln - Y - step )  .
 * Each thread loops down a column of the plain
 * Reads the input and multiplies it with the kernel and writes result to plain
 */
template<int FLAGS, int noSteps>
__global__ void mult22_k(const __restrict__ fcomplexcu*  kernels, const __restrict__ fcomplexcu*  inpData, __restrict__ fcomplexcu* ffdot, const int width, const int stride, int noPlns, const int firstPlain )
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
      const int plnStrd       = pln*stride*noSteps;
      const int plnHeight     = HEIGHT_HARM[firstPlain + pln];
      const int kerYOffset    = (HEIGHT_HARM[firstPlain] - plnHeight)/2;
      const int ns2           = plnHeight * stride;

      FOLD // Read input data for this plain  .
      {
        for (int step = 0; step < noSteps; step++)
        {
          fcomplexcu inp      = inpData[ (int)(plnStrd + step*stride) ];
          inp.r               /= (float) width;
          inp.i               /= (float) width;
          inpDat[step]        = inp;
        }
      }

      short   lDepth  = ceilf(plnHeight/(float)gridDim.y);
      short   y0      = lDepth*blockIdx.y;
      short   y1      = MIN(y0+lDepth, plnHeight);

      for (int plainY = y0; plainY < y1; plainY++)      // Loop over the individual plain  .
      {
        FOLD // Read the kernel value  .
        {
          ker   = kernels[(kerYOffset+plainY)*stride];
        }

        int off1;

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

        for ( int step = 0; step < noSteps; ++step )          // Loop over steps .
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

          FOLD // Multiply  .
          {
            //ffdot[idx].r = (inpDat[step].r * ker.r + inpDat[step].i * ker.i);
            //ffdot[idx].i = (inpDat[step].i * ker.r - inpDat[step].r * ker.i);

            fcomplexcu ipd = inpDat[step];
            fcomplexcu out;
            out.r = (ipd.r * ker.r + ipd.i * ker.i);
            out.i = (ipd.i * ker.r - ipd.r * ker.i);
            ffdot[idx] = out;
          }
        }
      }

      pHeight += plnHeight * noSteps * stride;
    }
  }
}

template<int FLAGS>
__host__  void mult22_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuFFdotBatch* batch, uint stack)
{
  cuFfdotStack* cStack  = &batch->stacks[stack];
  int offset            = cStack->startIdx;

  switch (batch->noSteps)
  {
    case 1:
    {
      mult22_k<FLAGS,1><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
    case 2:
    {
      mult22_k<FLAGS,2><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
    case 3:
    {
      mult22_k<FLAGS,3><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
    case 4:
    {
      mult22_k<FLAGS,4><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
    case 5:
    {
      mult22_k<FLAGS,5><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
    case 6:
    {
      mult22_k<FLAGS,6><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
    case 7:
    {
      mult22_k<FLAGS,7><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
    case 8:
    {
      mult22_k<FLAGS,8><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
    default:
    {
      fprintf(stderr, "ERROR: mult22 has not been templated for %lu steps\n", batch->noSteps);
      exit(EXIT_FAILURE);
    }
  }
}

__host__  void mult22_f(cudaStream_t multStream, cuFFdotBatch* batch, uint stack)
{
  dim3 dimGrid, dimBlock;

  cuFfdotStack* cStack = &batch->stacks[stack];

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY ));
  dimGrid.y = cStack->noMulSlices;

  if      ( batch->flag & FLAG_ITLV_ROW )
    mult22_s<FLAG_ITLV_ROW>(dimGrid, dimBlock, 0, multStream, batch, stack);
  else if ( batch->flag & FLAG_ITLV_PLN )
    mult22_s<FLAG_ITLV_PLN>(dimGrid, dimBlock, 0, multStream, batch, stack);
  else
  {
    fprintf(stderr, "ERROR: mult22 has not been templated for layout.\n");
    exit(EXIT_FAILURE);
  }
}
