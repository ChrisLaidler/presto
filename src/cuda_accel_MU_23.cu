#include "cuda_accel_MU.h"

/** Multiplication kernel - All multiplications for a stack - Uses registers to store sections of the kernel - Loop ( chunk (read ker) - plan - Y - step ) .
 * Each thread loops down a column of the plain
 * Reads the input and multiples it with the kernel and writes result to plain
 */
template<int FLAGS, int noSteps, int noPlns>
__global__ void mult23_k(const __restrict__ fcomplexcu* kernels, const __restrict__ fcomplexcu* inpData, __restrict__ fcomplexcu* ffdot, const int width, const int stride, const int firstPlain )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;          /// Block ID - flat index
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;      /// Global thread ID - flat index ie column index of stack

  const int cv_chunkSZ = 12;

  if ( tid < width )  // Valid thread  .
  {
    const int kerHeight = HEIGHT_HARM[firstPlain];       // The size of the kernel

    short   lDepth      = ceilf(kerHeight/(float)gridDim.y);
    short   y0          = lDepth*blockIdx.y;
    short   y1          = MIN(y0+lDepth, kerHeight);

    FOLD  // Stride, kernel, input data & output data  .
    {
      kernels += tid;
      ffdot   += tid;
      inpData += tid;
    }

    //int noChunks = ceilf(kerHeight / (float)cv_chunkSZ);
    //for ( int chunk = 0; chunk < noChunks; chunk++ )

    for ( int y = y0; y < y1; y+=cv_chunkSZ )
    {
      //const int c0  = chunk*cv_chunkSZ;
      //const int c1  = MIN(cv_chunkSZ,kerHeight-c0);

      const int c0  = y;
      const int c1  = MIN(cv_chunkSZ,kerHeight-y);
      int pHeight   = 0;

      register fcomplexcu k0   = kernels[(c0+0 )*stride];
      register fcomplexcu k1   = kernels[(c0+1 )*stride];
      register fcomplexcu k2   = kernels[(c0+2 )*stride];
      register fcomplexcu k3   = kernels[(c0+3 )*stride];
      register fcomplexcu k4   = kernels[(c0+4 )*stride];
      register fcomplexcu k5   = kernels[(c0+5 )*stride];
      register fcomplexcu k6   = kernels[(c0+6 )*stride];
      register fcomplexcu k7   = kernels[(c0+7 )*stride];
      register fcomplexcu k8   = kernels[(c0+8 )*stride];
      register fcomplexcu k9   = kernels[(c0+9 )*stride];
      register fcomplexcu k10  = kernels[(c0+10)*stride];
      register fcomplexcu k11  = kernels[(c0+11)*stride];

      for (int pln = 0; pln < noPlns; pln++)                  // Loop through the plains of the stack  .
      {
        const int plnHeight     = HEIGHT_HARM[firstPlain + pln];
        const int kerYOffset    = (kerHeight - plnHeight)/2;

        const int p0            = MAX(c0 - kerYOffset,0);
        const int p1            = MIN(c0 + c1 - kerYOffset, plnHeight);

        const int kerAddd       = MAX(0, kerYOffset - c0);

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

        for (int plainY = p0; plainY < p1; plainY++)          // Loop over the individual plain  .
        {
          int y = plainY - p0 + kerAddd;
          fcomplexcu ker;

          FOLD // Read the kernel value  .
          {
            switch(y)
            {
              case 0	:
              {
                ker = k0;
                break;
              }
              case 1	:
              {
                ker = k1;
                break;
              }
              case 2	:
              {
                ker = k2;
                break;
              }
              case 3	:
              {
                ker = k3;
                break;
              }
              case 4	:
              {
                ker = k4;
                break;
              }
              case 5	:
              {
                ker = k5;
                break;
              }
              case 6	:
              {
                ker = k6;
                break;
              }
              case 7	:
              {
                ker = k7;
                break;
              }
              case 8	:
              {
                ker = k8;
                break;
              }
              case 9	:
              {
                ker = k9;
                break;
              }
              case 10	:
              {
                ker = k10;
                break;
              }
              case 11	:
              {
                ker = k11;
                break;
              }
            }

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

          for ( int step = 0; step < noSteps; step++ )        // Loop over steps  .
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

template<int FLAGS, int noSteps>
__host__  void mult23_p(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuFFdotBatch* batch, uint stack)
{
  cuFfdotStack* cStack  = &batch->stacks[stack];
  int offset            = cStack->startIdx;

  switch (cStack->noInStack)
  {
    case 1	:
    {
      //cudaFuncSetCacheConfig(mult23_k<FLAGS,noSteps,1>, cudaFuncCachePreferL1);
      mult23_k<FLAGS,noSteps,1><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 2	:
    {
      //cudaFuncSetCacheConfig(mult23_k<FLAGS,noSteps,2>, cudaFuncCachePreferL1);
      mult23_k<FLAGS,noSteps,2><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 3	:
    {
      //cudaFuncSetCacheConfig(mult23_k<FLAGS,noSteps,3>, cudaFuncCachePreferL1);
      mult23_k<FLAGS,noSteps,3><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 4	:
    {
      //cudaFuncSetCacheConfig(mult23_k<FLAGS,noSteps,4>, cudaFuncCachePreferL1);
      mult23_k<FLAGS,noSteps,4><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 5	:
    {
      //cudaFuncSetCacheConfig(mult23_k<FLAGS,noSteps,5>, cudaFuncCachePreferL1);
      mult23_k<FLAGS,noSteps,5><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 6	:
    {
      //cudaFuncSetCacheConfig(mult23_k<FLAGS,noSteps,6>, cudaFuncCachePreferL1);
      mult23_k<FLAGS,noSteps,6><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 7	:
    {
      //cudaFuncSetCacheConfig(mult23_k<FLAGS,noSteps,7>, cudaFuncCachePreferL1);
      mult23_k<FLAGS,noSteps,7><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 8	:
    {
      //cudaFuncSetCacheConfig(mult23_k<FLAGS,noSteps,8>, cudaFuncCachePreferL1);
      mult23_k<FLAGS,noSteps,8><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 9	:
    {
      //cudaFuncSetCacheConfig(mult23_k<FLAGS,noSteps,9>, cudaFuncCachePreferL1);
      mult23_k<FLAGS,noSteps,9><<<dimGrid, dimBlock, i1, multStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    default	:
    {
      fprintf(stderr, "ERROR: mult23 has not been templated for %i plains in a stack.\n",cStack->noInStack);
      exit(EXIT_FAILURE);
    }
  }
}

template<int FLAGS>
__host__  void mult23_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuFFdotBatch* batch, uint stack)
{
  switch (batch->noSteps)
  {
    case 1	:
    {
      mult23_p<FLAGS,1>(dimGrid, dimBlock, i1, multStream, batch, stack);
      break;
    }
    case 2	:
    {
      mult23_p<FLAGS,2>(dimGrid, dimBlock, i1, multStream, batch, stack);
      break;
    }
    case 3	:
    {
      mult23_p<FLAGS,3>(dimGrid, dimBlock, i1, multStream, batch, stack);
      break;
    }
    case 4	:
    {
      mult23_p<FLAGS,4>(dimGrid, dimBlock, i1, multStream, batch, stack);
      break;
    }
    case 5	:
    {
      mult23_p<FLAGS,5>(dimGrid, dimBlock, i1, multStream, batch, stack);
      break;
    }
    case 6	:
    {
      mult23_p<FLAGS,6>(dimGrid, dimBlock, i1, multStream, batch, stack);
      break;
    }
    case 7	:
    {
      mult23_p<FLAGS,7>(dimGrid, dimBlock, i1, multStream, batch, stack);
      break;
    }
    case 8	:
    {
      mult23_p<FLAGS,8>(dimGrid, dimBlock, i1, multStream, batch, stack);
      break;
    }
    default	:
    {
      fprintf(stderr, "ERROR: convolveffdot42 has not been templated for %i steps\n", batch->noSteps);
      exit(EXIT_FAILURE);
    }
  }
}

__host__  void mult23_f(cudaStream_t multStream, cuFFdotBatch* batch, uint stack)
{
  dim3 dimGrid, dimBlock;

  cuFfdotStack* cStack = &batch->stacks[stack];

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY )) ;
  dimGrid.y = cStack->mulSlices ;


  if      ( batch->flag & FLAG_ITLV_ROW )
    mult23_s<FLAG_ITLV_ROW>(dimGrid, dimBlock, 0, multStream, batch, stack);
  else if ( batch->flag & FLAG_ITLV_PLN )
    mult23_s<FLAG_ITLV_PLN>(dimGrid, dimBlock, 0, multStream, batch, stack);
  else
  {
    fprintf(stderr, "ERROR: mult23 has not been templated for layout.\n");
    exit(EXIT_FAILURE);
  }
}
