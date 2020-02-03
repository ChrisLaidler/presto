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
 *     Added preprocessor directives for segments and chunks
 *
 */
 
#include "cuda_accel_MU.h"

#ifdef WITH_MUL_21

/** Multiplication kernel - Multiply a stack with a kernel - multi-segment - Loop ( Y - Pln - segment )  .
 * Each thread loops down a column of the plane
 * Reads the input and multiplies it with the kernel and writes result to plane
 */
template<int64_t FLAGS, int noSegments, int noPlns>
__global__ void mult21_k(const float2* __restrict__ kernels, const float2* __restrict__ inpData, float2* __restrict__ ffdot, const int width, const int stride, const int firstPlane )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;		/// Block ID - flat index
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;		/// Global thread ID - flat index ie column index of stack

  if ( tid < width )  // Valid thread  .
  {
    const int   kerHeight = HEIGHT_HARM[firstPlane];			// The size of the kernel
    float2      inpDat[noPlns][noSegments];				// Set of input data for this thread/column

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
      for ( int sIdx = 0; sIdx < noSegments; sIdx++ )
      {
        for ( int pln = 0; pln < noPlns; pln++ )			// Loop through the planes  .
        {
          float2 ipd			= inpData[ (int)(pln*noSegments*stride + sIdx*stride) ];
          ipd.x				/= (float) width;
          ipd.y				/= (float) width;
          inpDat[pln][sIdx]	= ipd;
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
              off1  = pHeight + planeY*noSegments*stride;
            }
#ifdef WITH_ITLV_PLN
            else
            {
              off1  = pHeight + planeY*stride;
            }
#endif
          }

          for ( int sIdx = 0; sIdx < noSegments; ++sIdx )		// Loop over segments .
          {
            int idx;

            FOLD // Calculate indices  .
            {
              if      ( FLAGS & FLAG_ITLV_ROW )
              {
                idx  = off1 + sIdx * stride;
              }
#ifdef WITH_ITLV_PLN
              else
              {
                idx  = off1 + sIdx * ns2;
              }
#endif
            }

            FOLD // Multiply  .
            {
              float2 ipd = inpDat[pln][sIdx];
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

        pHeight += plnHeight * noSegments * stride;			// Set striding value for next plane
      }
    }
  }
}

template<int64_t FLAGS, int noSwgments>
__host__  void mult21_p(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuCgPlan* plan, cuFfdotStack* cStack)
{
  int offset            = cStack->startIdx;

  switch (cStack->noInStack)
  {
    case 1:
    {
      mult21_k<FLAGS,noSwgments,1><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 2:
    {
      mult21_k<FLAGS,noSwgments,2><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 3:
    {
      mult21_k<FLAGS,noSwgments,3><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 4:
    {
      mult21_k<FLAGS,noSwgments,4><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 5:
    {
      mult21_k<FLAGS,noSwgments,5><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 6:
    {
      mult21_k<FLAGS,noSwgments,6><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 7:
    {
      mult21_k<FLAGS,noSwgments,7><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 8:
    {
      mult21_k<FLAGS,noSwgments,8><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 9:
    {
      mult21_k<FLAGS,noSwgments,9><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
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
__host__  void mult21_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuCgPlan* plan, cuFfdotStack* cStack)
{
  switch (plan->noSegments)
  {
#if MIN_SEGMENTS <= 1  and MAX_SEGMENTS >= 1
    case 1:
    {
      mult21_p<FLAGS,1>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTS <= 2  and MAX_SEGMENTS >= 2
    case 2:
    {
      mult21_p<FLAGS,2>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTS <= 3  and MAX_SEGMENTS >= 3
    case 3:
    {
      mult21_p<FLAGS,3>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTS <= 4  and MAX_SEGMENTS >= 4
    case 4:
    {
      mult21_p<FLAGS,4>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTS <= 5  and MAX_SEGMENTS >= 5
    case 5:
    {
      mult21_p<FLAGS,5>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTS <= 6  and MAX_SEGMENTS >= 6
    case 6:
    {
      mult21_p<FLAGS,6>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTS <= 7  and MAX_SEGMENTS >= 7
    case 7:
    {
      mult21_p<FLAGS,7>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTS <= 8  and MAX_SEGMENTS >= 8
    case 8:
    {
      mult21_p<FLAGS,8>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTS <= 9  and MAX_SEGMENTS >= 9
    case 9:
    {
      mult21_p<FLAGS,9>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTS <= 10 and MAX_SEGMENTS >= 10
    case 10:
    {
      mult21_p<FLAGS,10>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTS <= 11 and MAX_SEGMENTS >= 11
    case 11:
    {
      mult21_p<FLAGS,11>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTS <= 12 and MAX_SEGMENTS >= 12
    case 12:
    {
      mult21_p<FLAGS,12>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

    default:
    {
      if      ( plan->noSegments < MIN_SEGMENTS )
	fprintf(stderr, "ERROR: In %s, # segments (%i) less than the compiled minimum %i.\n", __FUNCTION__, plan->noSegments, MIN_SEGMENTS );
      else if ( plan->noSegments > MAX_SEGMENTS )
	fprintf(stderr, "ERROR: In %s, # segments (%i) greater than the compiled maximum %i.\n", __FUNCTION__, plan->noSegments, MIN_SEGMENTS );
      else
	fprintf(stderr, "ERROR: %s has not been templated for %i segments.\n", __FUNCTION__, plan->noSegments);

      exit(EXIT_FAILURE);
    }
  }
}

#endif

__host__  void mult21(cudaStream_t multStream, cuCgPlan* plan, cuFfdotStack* cStack)
{
#ifdef WITH_MUL_21

  dim3 dimGrid, dimBlock;

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY ));
  dimGrid.y = cStack->mulSlices;

  if      ( plan->flags & FLAG_ITLV_ROW )
    mult21_s<FLAG_ITLV_ROW>(dimGrid, dimBlock, 0, multStream, plan, cStack);
#ifdef WITH_ITLV_PLN
  else
    mult21_s<0>(dimGrid, dimBlock, 0, multStream, plan, cStack);
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
