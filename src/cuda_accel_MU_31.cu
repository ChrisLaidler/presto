/** @file cuda_accel_MU_31.cu
 *  @brief The implementation of the family multiplication kernel v1
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

#ifdef WITH_MUL_31

/** Multiplication kernel - Multiply an entire plan with convolution kernel  .
 * Each thread loops down a column of the planes and multiplies input with kernel and writes result to plane
 */
template<int64_t FLAGS, int noSegments>
__global__ void mult31_k(const __restrict__ float2* kernels, const __restrict__ float2* datas, __restrict__ float2* ffdot, int noPlanes)
{
  const int ix = blockIdx.x * CNV_DIMX * CNV_DIMY + CNV_DIMX * threadIdx.y + threadIdx.x;

  float2 input[noSegments];

  // Stride
  ffdot   += ix;
  datas   += ix;

  int pHeight = 0;

  for (int n = 0; n < noPlanes; n++)					// Loop over planes  .
  {
    const int stride      = STRIDE_HARM[n];

    if ( ix < stride )
    {
      const int plnHeight = HEIGHT_HARM[n];
      const short lDepth  = ceilf(plnHeight/(float)gridDim.y);
      const short y0      = lDepth*blockIdx.y;
      const short y1      = MIN(y0+lDepth, plnHeight);
      float2* ker         = (float2*)KERNEL_HARM[n] + y0 * stride + ix;

#ifdef WITH_ITLV_PLN
      const int plnStride = plnHeight*stride;
#endif

      // read input for each segment into registers
      for (int sIdx = 0; sIdx < noSegments; sIdx++)			// Loop over planes  .
      {
        input[sIdx]       = datas[sIdx*stride];

        // Normalise
        input[sIdx].x    /= (float) stride ;
        input[sIdx].y    /= (float) stride ;
      }

      // Stride input data
      datas              += stride*noSegments;

      for (int planeY = y0; planeY < y1; planeY++)			// Loop over individual plane  .
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

        // Multiply and write data
        for (int sIdx = 0; sIdx < noSegments; sIdx++)			// Loop over segments  .
        {
          //
          float2 out;
          float2 ipd = input[sIdx];

          // Calculate index
          int idx = 0;
          FOLD // Calculate indices  .
          {
            if      ( FLAGS & FLAG_ITLV_ROW )
            {
              idx  = off1 + sIdx * stride;
            }
#ifdef WITH_ITLV_PLN
            else
            {
              idx  = off1 + sIdx * plnStride;
            }
#endif
          }

	  // Multiply
#if CORRECT_MULT
          // This is the "correct" version
          out.x = (ipd.x * ker->x - ipd.y * ker->y);
          out.y = (ipd.x * ker->y + ipd.y * ker->x);
#else
          // This is the version accelsearch uses, ( added for comparison )
          out.x = (ipd.x * ker->x + ipd.y * ker->y);
          out.y = (ipd.y * ker->x - ipd.x * ker->y);
#endif

          // Write the actual value
          ffdot[idx] = out;
        }

        // Stride kernel to next row
        ker += stride;
      }

      // Track plane offset
      pHeight += noSegments*plnHeight*stride;
    }
  }
}

template<int64_t FLAGS>
__host__  void mult31_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuCgPlan* plan)
{
  switch (plan->noSegments)
  {
#if MIN_SEGMENTS <= 1  and MAX_SEGMENTS >= 1
    case 1:
    {
      mult31_k<FLAGS,1><<<dimGrid, dimBlock, i1, multStream>>>((float2*)plan->d_kerData, (float2*)plan->d_iData, (float2*)plan->d_planeCplx, plan->noGenHarms);
      break;
    }
#endif

#if MIN_SEGMENTS <= 2  and MAX_SEGMENTS >= 2
    case 2:
    {
      mult31_k<FLAGS,2><<<dimGrid, dimBlock, i1, multStream>>>((float2*)plan->d_kerData, (float2*)plan->d_iData, (float2*)plan->d_planeCplx, plan->noGenHarms);
      break;
    }
#endif

#if MIN_SEGMENTS <= 3  and MAX_SEGMENTS >= 3
    case 3:
    {
      mult31_k<FLAGS,3><<<dimGrid, dimBlock, i1, multStream>>>((float2*)plan->d_kerData, (float2*)plan->d_iData, (float2*)plan->d_planeCplx, plan->noGenHarms);
      break;
    }
#endif

#if MIN_SEGMENTS <= 4  and MAX_SEGMENTS >= 4
    case 4:
    {
      mult31_k<FLAGS,4><<<dimGrid, dimBlock, i1, multStream>>>((float2*)plan->d_kerData, (float2*)plan->d_iData, (float2*)plan->d_planeCplx, plan->noGenHarms);
      break;
    }
#endif

#if MIN_SEGMENTS <= 5  and MAX_SEGMENTS >= 5
    case 5:
    {
      mult31_k<FLAGS,5><<<dimGrid, dimBlock, i1, multStream>>>((float2*)plan->d_kerData, (float2*)plan->d_iData, (float2*)plan->d_planeCplx, plan->noGenHarms);
      break;
    }
#endif

#if MIN_SEGMENTS <= 6  and MAX_SEGMENTS >= 6
    case 6:
    {
      mult31_k<FLAGS,6><<<dimGrid, dimBlock, i1, multStream>>>((float2*)plan->d_kerData, (float2*)plan->d_iData, (float2*)plan->d_planeCplx, plan->noGenHarms);
      break;
    }
#endif

#if MIN_SEGMENTS <= 7  and MAX_SEGMENTS >= 7
    case 7:
    {
      mult31_k<FLAGS,7><<<dimGrid, dimBlock, i1, multStream>>>((float2*)plan->d_kerData, (float2*)plan->d_iData, (float2*)plan->d_planeCplx, plan->noGenHarms);
      break;
    }
#endif

#if MIN_SEGMENTS <= 8  and MAX_SEGMENTS >= 8
    case 8:
    {
      mult31_k<FLAGS,8><<<dimGrid, dimBlock, i1, multStream>>>((float2*)plan->d_kerData, (float2*)plan->d_iData, (float2*)plan->d_planeCplx, plan->noGenHarms);
      break;
    }
#endif

#if MIN_SEGMENTS <= 9  and MAX_SEGMENTS >= 9
    case 9:
    {
      mult31_k<FLAGS,9><<<dimGrid, dimBlock, i1, multStream>>>((float2*)plan->d_kerData, (float2*)plan->d_iData, (float2*)plan->d_planeCplx, plan->noGenHarms);
      break;
    }
#endif

#if MIN_SEGMENTS <= 10 and MAX_SEGMENTS >= 10
    case 10:
    {
      mult31_k<FLAGS,10><<<dimGrid, dimBlock, i1, multStream>>>((float2*)plan->d_kerData, (float2*)plan->d_iData, (float2*)plan->d_planeCplx, plan->noGenHarms);
      break;
    }
#endif

#if MIN_SEGMENTS <= 11 and MAX_SEGMENTS >= 11
    case 11:
    {
      mult31_k<FLAGS,11><<<dimGrid, dimBlock, i1, multStream>>>((float2*)plan->d_kerData, (float2*)plan->d_iData, (float2*)plan->d_planeCplx, plan->noGenHarms);
      break;
    }
#endif

#if MIN_SEGMENTS <= 12 and MAX_SEGMENTS >= 12
    case 12:
    {
      mult31_k<FLAGS,12><<<dimGrid, dimBlock, i1, multStream>>>((float2*)plan->d_kerData, (float2*)plan->d_iData, (float2*)plan->d_planeCplx, plan->noGenHarms);
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

#endif	// WITH_MUL_31

__host__  void mult31(cudaStream_t multStream, cuCgPlan* plan)
{
#ifdef WITH_MUL_31

  dim3 dimGrid, dimBlock;

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil(plan->hInfos[0].width / (float) ( CNV_DIMX * CNV_DIMY ));
  dimGrid.y = plan->mulSlices;

  if      ( plan->flags & FLAG_ITLV_ROW )
    mult31_s<FLAG_ITLV_ROW>(dimGrid, dimBlock, 0, multStream, plan);
#ifdef WITH_ITLV_PLN
  else
    mult31_s<0>(dimGrid, dimBlock, 0, multStream, plan);
#else
  else
  {
    fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
    exit(EXIT_FAILURE);
  }
#endif

#else
  EXIT_DIRECTIVE("WITH_MUL_31");
#endif
}
