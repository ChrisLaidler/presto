/** @file cuda_accel_MU_21.cu
 *  @brief The implementation of the stack multiplication kernel v1
 *
 *  @author Chris Laidler
 *  @bug No known bugs.
 *
 *  Change Log
 *
 *  [0.0.01] []
 *    Beginning of change log
 *    Working version un-numbed
 *
 *  [0.0.01] [2017-02-24]
 *     Added preprocessor directives for steps and chunks
 *
 */
 
#include "cuda_accel_MU.h"

#ifdef WITH_MUL_21

/** Multiplication kernel - Multiply a stack with a kernel - multi-step - Loop ( Y - Pln - step )  .
 * Each thread loops down a column of the plane
 * Reads the input and multiplies it with the kernel and writes result to plane
 */
template<int64_t FLAGS, int noSteps, int noPlns>
__global__ void mult21_k(const float2* __restrict__ kernels, const float2* __restrict__ inpData, float2* __restrict__ ffdot, const int width, const int stride, const int firstPlane )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;		/// Block ID - flat index
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;		/// Global thread ID - flat index ie column index of stack

  if ( tid < width )  // Valid thread  .
  {
    const int   kerHeight = HEIGHT_HARM[firstPlane];			// The size of the kernel
    float2  inpDat[noPlns][noSteps];				// Set of input data for this thread/column

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
        for ( int pln = 0; pln < noPlns; pln++ )			// Loop through the planes  .
        {
          float2 ipd        = inpData[ (int)(pln*noSteps*stride + step*stride) ];
          ipd.x                 /= (float) width;
          ipd.y                 /= (float) width;
          inpDat[pln][step]     = ipd;
        }
      }
    }

    for ( int kerY = y0; kerY < y1; kerY++ )				// Loop through the kernel  .
    {
      float2 ker;							// kernel data
      int pHeight = 0;							// Height of previous data in the stack

      FOLD // Read the kernel value  .
      {
        ker   = kernels[kerY*stride];
      }

      for (int pln = 0; pln < noPlns; pln++)				// Loop through the planes  .
      {
        const int plnHeight     = HEIGHT_HARM[firstPlane + pln];
        const int kerYOffset    = KERNEL_OFF_HARM[firstPlane + pln];
        const int planeY        = kerY - kerYOffset;
#ifdef WITH_ITLV_PLN
        const int ns2           = plnHeight * stride;
#endif

        if( planeY >= 0 && planeY < plnHeight )
        {
          int off1;

          FOLD // Calculate partial offset  .
          {
            if      ( FLAGS & FLAG_ITLV_ROW )
            {
              off1  = pHeight + planeY*noSteps*stride;
            }
#ifdef WITH_ITLV_PLN
            else
            {
              off1  = pHeight + planeY*stride;
            }
#endif
          }

          for ( int step = 0; step < noSteps; ++step )			// Loop over steps .
          {
            int idx;

            FOLD // Calculate indices  .
            {
              if      ( FLAGS & FLAG_ITLV_ROW )
              {
                idx  = off1 + step * stride;
              }
#ifdef WITH_ITLV_PLN
              else
              {
                idx  = off1 + step * ns2;
              }
#endif
            }

            FOLD // Multiply  .
            {
              float2 ipd = inpDat[pln][step];
              float2 out;

#if CORRECT_MULT
              // This is the "correct" version
              out.x = (ipd.x * ker.x - ipd.y * ker.y);
              out.y = (ipd.x * ker.y + ipd.y * ker.x);
#else
              // This is the version accelsearch uses, ( added for comparison )
              out.x = (ipd.x * ker.x + ipd.y * ker.y);
              out.y = (ipd.y * ker.x - ipd.x * ker.y);
#endif
              ffdot[idx] = out;
            }
          }
        }

        pHeight += plnHeight * noSteps * stride;			// Set striding value for next plane
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
      mult21_k<FLAGS,noSteps,1><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 2:
    {
      mult21_k<FLAGS,noSteps,2><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 3:
    {
      mult21_k<FLAGS,noSteps,3><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 4:
    {
      mult21_k<FLAGS,noSteps,4><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 5:
    {
      mult21_k<FLAGS,noSteps,5><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 6:
    {
      mult21_k<FLAGS,noSteps,6><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 7:
    {
      mult21_k<FLAGS,noSteps,7><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 8:
    {
      mult21_k<FLAGS,noSteps,8><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 9:
    {
      mult21_k<FLAGS,noSteps,9><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
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
#if MIN_STEPS <= 1  and MAX_STEPS >= 1
    case 1:
    {
      mult21_p<FLAGS,1>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 2  and MAX_STEPS >= 2
    case 2:
    {
      mult21_p<FLAGS,2>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 3  and MAX_STEPS >= 3
    case 3:
    {
      mult21_p<FLAGS,3>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 4  and MAX_STEPS >= 4
    case 4:
    {
      mult21_p<FLAGS,4>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 5  and MAX_STEPS >= 5
    case 5:
    {
      mult21_p<FLAGS,5>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 6  and MAX_STEPS >= 6
    case 6:
    {
      mult21_p<FLAGS,6>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 7  and MAX_STEPS >= 7
    case 7:
    {
      mult21_p<FLAGS,7>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 8  and MAX_STEPS >= 8
    case 8:
    {
      mult21_p<FLAGS,8>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 9  and MAX_STEPS >= 9
    case 9:
    {
      mult21_p<FLAGS,9>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 10 and MAX_STEPS >= 10
    case 10:
    {
      mult21_p<FLAGS,10>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 11 and MAX_STEPS >= 11
    case 11:
    {
      mult21_p<FLAGS,11>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 12 and MAX_STEPS >= 12
    case 12:
    {
      mult21_p<FLAGS,12>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

    default:
    {
      if      ( batch->noSteps < MIN_STEPS )
	fprintf(stderr, "ERROR: In %s, # steps (%i) less than the compiled minimum %i.\n", __FUNCTION__, batch->noSteps, MIN_STEPS );
      else if ( batch->noSteps > MAX_STEPS )
	fprintf(stderr, "ERROR: In %s, # steps (%i) greater than the compiled maximum %i.\n", __FUNCTION__, batch->noSteps, MIN_STEPS );
      else
	fprintf(stderr, "ERROR: %s has not been templated for %i steps.\n", __FUNCTION__, batch->noSteps);

      exit(EXIT_FAILURE);
    }
  }
}

#endif

__host__  void mult21(cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack)
{
#ifdef WITH_MUL_21

  dim3 dimGrid, dimBlock;

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY ));
  dimGrid.y = cStack->mulSlices;

  if      ( batch->flags & FLAG_ITLV_ROW )
    mult21_s<FLAG_ITLV_ROW>(dimGrid, dimBlock, 0, multStream, batch, cStack);
#ifdef WITH_ITLV_PLN
  else
    mult21_s<0>(dimGrid, dimBlock, 0, multStream, batch, cStack);
#else
  else
  {
    fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
    exit(EXIT_FAILURE);
  }
#endif

#else
  EXIT_DIRECTIVE("WITH_MUL_21");
#endif
}
