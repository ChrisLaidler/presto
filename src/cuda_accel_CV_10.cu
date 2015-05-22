#include "cuda_accel_CV.h"

template<int FLAGS, int noSteps, int noPlns>
__global__ void convolveffdot10_f(const __restrict__ fcomplexcu* kernels, const __restrict__ fcomplexcu* inpData, __restrict__ fcomplexcu* ffdot, const int width, const int stride, const int firstPlain )
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  const int kerHeight   = HEIGHT_FAM_ORDER[firstPlain];       // The size of the kernel

  if ( ix < width && iy < kerHeight )
  {
    // Calculate flat index
    int pHeight           = 0;
    fcomplexcu ker;

    FOLD  // Stride, kernel, input data & output data  .
    {
      kernels += ix;
      ffdot   += ix;
      inpData += ix;
    }

    FOLD // Read the kernel  .
    {
      int idx = iy * stride;
      ker   = kernels[idx];
      ker.r /= (float) width;
      ker.i /= (float) width;
    }

    for (int pln = 0; pln < noPlns; pln++)                    // Loop through the plains of the stack  .
    {
      const int plnHeight     = HEIGHT_FAM_ORDER[firstPlain + pln];
      const int kerYOffset    = (kerHeight - plnHeight)/2;
      const int plainY        = iy - kerYOffset;
      const int ns2           = plnHeight  * stride;

      if ( plainY >= 0 && plainY < plnHeight )
      {
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

        for ( int step = 0; step < noSteps; step++ )          // Loop over steps .
        {
          fcomplexcu ipd      = inpData[ (int)(pln*noSteps*stride + step*stride) ];

          int idx;
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
            ffdot[idx].r = ipd.r * ker.r + ipd.i * ker.i;
            ffdot[idx].i = ipd.i * ker.r - ipd.r * ker.i;
          }
        }
      }

      pHeight += plnHeight * noSteps * stride;
    }
  }
}

template<int FLAGS, int noSteps>
__host__  void convolveffdot10_p(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack)
{
  cuFfdotStack* cStack  = &batch->stacks[stack];
  int offset            = cStack->startIdx;

  switch (cStack->noInStack)
  {
    case 1  :
    {
      cudaFuncSetCacheConfig(convolveffdot10_f<FLAGS,noSteps,1>, cudaFuncCachePreferL1);
      convolveffdot10_f<FLAGS,noSteps,1><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 2  :
    {
      cudaFuncSetCacheConfig(convolveffdot10_f<FLAGS,noSteps,2>, cudaFuncCachePreferL1);
      convolveffdot10_f<FLAGS,noSteps,2><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 3  :
    {
      cudaFuncSetCacheConfig(convolveffdot10_f<FLAGS,noSteps,3>, cudaFuncCachePreferL1);
      convolveffdot10_f<FLAGS,noSteps,3><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 4  :
    {
      cudaFuncSetCacheConfig(convolveffdot10_f<FLAGS,noSteps,4>, cudaFuncCachePreferL1);
      convolveffdot10_f<FLAGS,noSteps,4><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 5  :
    {
      cudaFuncSetCacheConfig(convolveffdot10_f<FLAGS,noSteps,5>, cudaFuncCachePreferL1);
      convolveffdot10_f<FLAGS,noSteps,5><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 6  :
    {
      cudaFuncSetCacheConfig(convolveffdot10_f<FLAGS,noSteps,6>, cudaFuncCachePreferL1);
      convolveffdot10_f<FLAGS,noSteps,6><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 7  :
    {
      cudaFuncSetCacheConfig(convolveffdot10_f<FLAGS,noSteps,7>, cudaFuncCachePreferL1);
      convolveffdot10_f<FLAGS,noSteps,7><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 8  :
    {
      cudaFuncSetCacheConfig(convolveffdot10_f<FLAGS,noSteps,8>, cudaFuncCachePreferL1);
      convolveffdot10_f<FLAGS,noSteps,8><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 9  :
    {
      cudaFuncSetCacheConfig(convolveffdot10_f<FLAGS,noSteps,9>, cudaFuncCachePreferL1);
      convolveffdot10_f<FLAGS,noSteps,9><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    default :
    {
      fprintf(stderr, "ERROR: convolveffdot10 has not been templated for %i plains in a stack.\n",cStack->noInStack);
      exit(EXIT_FAILURE);
    }
  }
}

template<int FLAGS>
__host__  void convolveffdot10_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack)
{
  switch ( batch->noSteps )
  {
    case 1  :
    {
      convolveffdot10_p<FLAGS,1>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 2  :
    {
      convolveffdot10_p<FLAGS,2>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 3  :
    {
      convolveffdot10_p<FLAGS,3>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 4  :
    {
      convolveffdot10_p<FLAGS,4>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 5  :
    {
      convolveffdot10_p<FLAGS,5>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 6  :
    {
      convolveffdot10_p<FLAGS,6>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 7  :
    {
      convolveffdot10_p<FLAGS,7>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 8  :
    {
      convolveffdot10_p<FLAGS,8>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    default :
    {
      fprintf(stderr, "ERROR: convolveffdot42 has not been templated for %lu steps\n", batch->noSteps);
      exit(EXIT_FAILURE);
    }
  }
}

__host__  void convolveffdot10_f(cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack)
{
  dim3 dimGrid, dimBlock;

  cuFfdotStack* cStack = &batch->stacks[stack];

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil( cStack->width     / (float) ( CNV_DIMX ) );
  dimGrid.y = ceil( cStack->kerHeigth / (float) ( CNV_DIMY ) );

  if      ( batch->flag & FLAG_ITLV_ROW )
    convolveffdot10_s<FLAG_ITLV_ROW>(dimGrid, dimBlock, 0, cnvlStream, batch, stack);
  else if ( batch->flag & FLAG_ITLV_PLN )
    convolveffdot10_s<FLAG_ITLV_PLN>(dimGrid, dimBlock, 0, cnvlStream, batch, stack);
  else
  {
    fprintf(stderr, "ERROR: convolveffdot10 has not been templated for layout.\n");
    exit(EXIT_FAILURE);
  }
}
