#include "cuda_accel_MU.h"

/** Kernel for testing best possible performance - Just write to ffdot plain - 1 thread per complex value
 *
 * @param kernels
 * @param inpData
 * @param ffdot
 * @param width
 * @param height
 * @param stride
 * @param noSteps
 * @param kerHeight
 */
__global__ void mult00_k(const __restrict__ fcomplexcu* kernels, const __restrict__ fcomplexcu*  inpData, __restrict__ fcomplexcu* ffdot, const int width, const int height, const int stride, const int noSteps, const int noPlns, int kerHeight )
{
  const int ix = blockIdx.x * CNV_DIMX + threadIdx.x;
  const int iy = blockIdx.y * CNV_DIMY + threadIdx.y;

  fcomplexcu ker;                                 /// kernel data
  uint nHeight = height * noSteps;

  ker.i = 0;
  ker.r = 0;

  if (ix < width && iy < nHeight)
  {
    // Calculate flat index
    const int idx = iy * stride + ix;

    ffdot[idx] = ker;
  }
}

__global__ void mult01_k(const __restrict__ fcomplexcu* kernels, const __restrict__ fcomplexcu* inpData, __restrict__ fcomplexcu* ffdot, const int width, const int height, const int stride, const int noSteps, const int noPlns, int kerHeight )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;          /// Block ID - flat index
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;      /// Global thread ID - flat index ie column index of stack

  fcomplexcu ker;                                 /// kernel data

  if ( tid < width )  // Valid thread  .
  {
    int idx;                                      /// flat index of output plain

    FOLD  // Stride, kernel, input data & output data  .
    {
      kernels += tid;
      ffdot   += tid;
      inpData += tid;
    }

    FOLD // Read input data  .
    {
      for (int step = 0; step < noSteps; step++)
      {
        for (int pln = 0; pln < noPlns; pln++)                // Loop through the plains  .
        {
          fcomplexcu ipd        = inpData[ (int)(pln*noSteps*stride + step*stride) ];

          if ( ipd.r < 0 && ipd.r > 0 )   // Required so as to not optimise out
          {
            printf("ipd \n");
          }
        }
      }
    }

    FOLD // Read kernel  .
    {
      for (int k = 0; k < kerHeight; k++ )
      {
        idx   = k * stride;
        ker   = kernels[idx];

        if ( ker.r < 0 && ker.r > 0 )     // Required so as to not optimise out
        {
          printf("ker \n");
        }
      }
    }

    ker.i = 0;
    ker.r = 0;

    uint nHeight = height * noSteps;

    FOLD // Write data to plains  .
    {
      for (int y = 0; y < nHeight; y++ )
      {
        idx  = y * stride;

        FOLD // Write  .
        {
          ffdot[idx] = ker;
        }
      }
    }
  }
}

/** Kernel for testing best possible performance - Just write to ffdot plain - Each thread loops down over column  .
 *
 * @param kernels
 * @param inpData
 * @param ffdot
 * @param width
 * @param height
 * @param stride
 * @param noSteps
 * @param kerHeight
 */
__host__  void mult00_f(cudaStream_t multStream, cuFFdotBatch* batch, uint stack)
{
  dim3 dimGrid, dimBlock;

  cuFfdotStack* cStack = &batch->stacks[stack];

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  if (0)
  {
    dimGrid.x = ceil(cStack->width                    / (float) ( CNV_DIMX ));
    dimGrid.y = ceil(cStack->height*batch->noSteps    / (float) ( CNV_DIMX ));

    mult00_k<<<dimGrid, dimBlock, 0, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->height, cStack->strideCmplx, batch->noSteps, cStack->noInStack, cStack->kerHeigth);
  }
  else
  {
    dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY ));
    dimGrid.y = 1;

    mult01_k<<<dimGrid, dimBlock, 0, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->height, cStack->strideCmplx, batch->noSteps, cStack->noInStack, cStack->kerHeigth);
  }

}

//-----------------------------------------//


/** Multiplication kernel - Multiply a stack with a kernel - multi-step - Loop ( Pln - Y - step )  .
 * Each thread loops down a column of the plain
 * Reads the input and multiplies it with the kernel and writes result to plain
 */
template<int FLAGS, int noSteps>
__global__ void mult02_k(const fcomplexcu* __restrict__ kernels, const fcomplexcu* __restrict__ inpData, fcomplexcu* __restrict__ ffdot, const int width, const int stride, int noPlns, const int firstPlain )
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
      const int plnHeight     = HEIGHT_FAM_ORDER[firstPlain + pln];
      const int kerYOffset    = (HEIGHT_FAM_ORDER[firstPlain] - plnHeight)/2;
      const int ns2           = plnHeight * stride;

      FOLD // Read input data for this plain
      {
        for (int step = 0; step < noSteps; step++)
        {
          fcomplexcu inp    = inpData[ (int)(plnStrd + step*stride) ];
          inp.r             /= (float) width;
          inp.i             /= (float) width;
          inpDat[step]      = inp;
        }
      }

      for (int plainY = 0; plainY < plnHeight; plainY++)      // Loop over the individual plain  .
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

          fcomplexcu kv;
          FOLD // Multiply  .
          {
            kv.r = (inpDat[step].r * ker.r + inpDat[step].i * ker.i);
            kv.i = (inpDat[step].i * ker.r - inpDat[step].r * ker.i);
          }

          //ker.r       = tid;
          //ker.i       = plainY;
          //ffdot[idx]  = ker;
          //ffdot[idx]  = inpDat[step];
          //kv.r = width;
          ffdot[idx]  = kv;
        }
      }

      pHeight += plnHeight * noSteps * stride;
    }
  }
}

template<int FLAGS>
__host__  void mult02_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuFFdotBatch* batch, uint stack)
{
  cuFfdotStack* cStack  = &batch->stacks[stack];
  int offset            = cStack->startIdx;

  switch (batch->noSteps)
  {
    case 1:
    {
      mult02_k<FLAGS,1><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
    case 2:
    {
      mult02_k<FLAGS,2><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
    case 3:
    {
      mult02_k<FLAGS,3><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
    case 4:
    {
      mult02_k<FLAGS,4><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
    case 5:
    {
      mult02_k<FLAGS,5><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
    case 6:
    {
      mult02_k<FLAGS,6><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
    case 7:
    {
      mult02_k<FLAGS,7><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
    case 8:
    {
      mult02_k<FLAGS,8><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
    default:
    {
      fprintf(stderr, "ERROR: mult11 has not been templated for %lu steps\n", batch->noSteps);
      exit(EXIT_FAILURE);
    }
  }
}

__host__  void mult02_f(cudaStream_t multStream, cuFFdotBatch* batch, uint stack)
{
  dim3 dimGrid, dimBlock;

  cuFfdotStack* cStack = &batch->stacks[stack];

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY ));
  dimGrid.y = 1;

  if      ( batch->flag & FLAG_ITLV_ROW )
    mult02_s<FLAG_ITLV_ROW>(dimGrid, dimBlock, 0, multStream, batch, stack);
  else if ( batch->flag & FLAG_ITLV_PLN )
    mult02_s<FLAG_ITLV_PLN>(dimGrid, dimBlock, 0, multStream, batch, stack);
  else
  {
    fprintf(stderr, "ERROR: mult11 has not been templated for layout.\n");
    exit(EXIT_FAILURE);
  }
}
