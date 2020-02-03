/** @file cuda_accel_MU_23.cu
 *  @brief The implementation of the stack multiplication kernel v3
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
 *	Added preprocessor directives for segments and chunks
 *
 *  [0.0.01] [2017-03-25]
 *	Added templating for multiplication chunks
 *
 *  [2017-03-10]
 *	Fixed bug in templating
 *
 */

#include "cuda_accel_MU.h"

#ifdef WITH_MUL_23

/** Multiplication kernel - All multiplications for a stack - Uses registers to store sections of the kernel - Loop ( chunk (read ker) - plan - Y - segment ) .
 * Each thread loops down a column of the plane
 * Reads the input and multiples it with the kernel and writes result to plane
 */
template<int64_t FLAGS, int noSegments, int noPlns, const int cunkSize>
__global__ void mult23_k(const __restrict__ float2* kernels, const __restrict__ float2* inpData, __restrict__ float2* ffdot, const int width, const int stride, const int firstPlane )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;		/// Block ID - flat index
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;		/// Global thread ID - flat index ie column index of stack

  if ( tid < width )  // Valid thread  .
  {
    const int kerHeight = HEIGHT_HARM[firstPlane];			// The size of the kernel

    short   lDepth      = ceilf(kerHeight/(float)gridDim.y);
    short   y0          = lDepth*blockIdx.y;
    short   y1          = MIN(y0+lDepth, kerHeight);

    FOLD  // Stride, kernel, input data & output data  .
    {
      kernels += tid;
      ffdot   += tid;
      inpData += tid;
    }

    for ( int y = y0; y < y1; y+= cunkSize )
    {
      const int c0  = y;
      const int c1  = MIN(cunkSize, kerHeight-y);
      int pHeight   = 0;

      float2 kerChunk[cunkSize];

      for ( int i = 0; i < cunkSize; i++)
      {
	kerChunk[i]   = kernels[(c0+i )*stride];
      }

      for (int pln = 0; pln < noPlns; pln++)				// Loop through the planes of the stack  .
      {
        const int plnHeight     = HEIGHT_HARM[firstPlane + pln];
        const int kerYOffset    = KERNEL_OFF_HARM[firstPlane + pln];

        const int p0            = MAX(c0 - kerYOffset,0);
        const int p1            = MIN(c0 + c1 - kerYOffset, plnHeight);

        const int kerAddd       = MAX(0, kerYOffset - c0);

#ifdef WITH_ITLV_PLN
        const int ns2           = plnHeight * stride;
#endif

        __restrict__ float2 inpDat[noSegments];				// Set of input data for this thread/column
        FOLD // Read all input data  .
        {
          // NOTE: I tested reading the input for planes and segments (2 loops above) but that was slower, here uses less registers as well.

          for (int sIdx = 0; sIdx < noSegments; sIdx++)
          {
            float2 ipd	= inpData[ (int)(pln*noSegments*stride + sIdx*stride) ];
            ipd.x		/= (float) width;
            ipd.y		/= (float) width;
            inpDat[sIdx]	= ipd;
          }
        }

        for (int planeY = p0; planeY < p1; planeY++)			// Loop over the individual plane  .
        {
          int y = planeY - p0 + kerAddd;
          float2 ker = kerChunk[y];

          int offsetPart1;
          FOLD // Calculate partial offset  .
          {
            if      ( FLAGS & FLAG_ITLV_ROW )
            {
              offsetPart1  = pHeight + planeY*noSegments*stride;
            }
#ifdef WITH_ITLV_PLN
            else
            {
              offsetPart1  = pHeight + planeY*stride;
            }
#endif
          }

          for ( int sIdx = 0; sIdx < noSegments; sIdx++ )		// Loop over segments  .
          {
            int idx;

            FOLD // Calculate offset  .
            {
              if      ( FLAGS & FLAG_ITLV_ROW )
              {
                idx  = offsetPart1 + sIdx * stride;
              }
#ifdef WITH_ITLV_PLN
              else
              {
                idx  = offsetPart1 +sIdx  * ns2;
              }
#endif
            }

            FOLD // Multiply  .
            {
              float2 ipd = inpDat[sIdx];
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

template<int64_t FLAGS, int noSegments, int noPlns>
__host__  void mult23_c(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuCgPlan* plan, cuFfdotStack* cStack)
{
  int offset            = cStack->startIdx;

   switch (cStack->mulChunk)
   {
#if MIN_MUL_CHUNK <= 1  and MAX_MUL_CHUNK >= 1
    case 1 :
    {
      mult23_k<FLAGS,noSegments,noPlns,1><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 2  and MAX_MUL_CHUNK >= 2
    case 2 :
    {
      mult23_k<FLAGS,noSegments,noPlns,2><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 3  and MAX_MUL_CHUNK >= 3
    case 3 :
    {
      mult23_k<FLAGS,noSegments,noPlns,3><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 4  and MAX_MUL_CHUNK >= 4
    case 4 :
    {
      mult23_k<FLAGS,noSegments,noPlns,4><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 5  and MAX_MUL_CHUNK >= 5
    case 5 :
    {
      mult23_k<FLAGS,noSegments,noPlns,5><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 6  and MAX_MUL_CHUNK >= 6
    case 6 :
    {
      mult23_k<FLAGS,noSegments,noPlns,6><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 7  and MAX_MUL_CHUNK >= 7
    case 7 :
    {
      mult23_k<FLAGS,noSegments,noPlns,7><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 8  and MAX_MUL_CHUNK >= 8
    case 8 :
    {
      mult23_k<FLAGS,noSegments,noPlns,8><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 9  and MAX_MUL_CHUNK >= 9
    case 9 :
    {
      mult23_k<FLAGS,noSegments,noPlns,9><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 10  and MAX_MUL_CHUNK >= 10
    case 10 :
    {
      mult23_k<FLAGS,noSegments,noPlns,10><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 11  and MAX_MUL_CHUNK >= 11
    case 11 :
    {
      mult23_k<FLAGS,noSegments,noPlns,11><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 12  and MAX_MUL_CHUNK >= 12
    case 12 :
    {
      mult23_k<FLAGS,noSegments,noPlns,12><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

    default:
    {
      if ( cStack->mulChunk < MIN_MUL_CHUNK )
	fprintf(stderr, "ERROR: In %s, chunk size (%i) less than the compiled minimum %i.\n", __FUNCTION__, cStack->mulChunk, MIN_MUL_CHUNK );
      else if ( plan->ssChunk > MAX_MUL_CHUNK )
	fprintf(stderr, "ERROR: In %s, chunk size (%i) greater than the compiled maximum %i.\n", __FUNCTION__, cStack->mulChunk, MAX_MUL_CHUNK );
      else
	fprintf(stderr, "ERROR: %s has not been templated for %i chunk size.\n", __FUNCTION__, cStack->mulChunk);

      exit(EXIT_FAILURE);
    }

   }
}

template<int64_t FLAGS, int noSegments>
__host__  void mult23_p(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuCgPlan* plan, cuFfdotStack* cStack)
{
   switch (cStack->noInStack)
  {
    case 1	:
    {
      mult23_c<FLAGS,noSegments,1>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
    case 2	:
    {
      mult23_c<FLAGS,noSegments,2>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
    case 3	:
    {
      mult23_c<FLAGS,noSegments,3>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
    case 4	:
    {
      mult23_c<FLAGS,noSegments,4>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
    case 5	:
    {
      mult23_c<FLAGS,noSegments,5>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
    case 6	:
    {
      mult23_c<FLAGS,noSegments,6>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
    case 7	:
    {
      mult23_c<FLAGS,noSegments,7>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
    case 8	:
    {
      mult23_c<FLAGS,noSegments,8>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
    case 9	:
    {
      mult23_c<FLAGS,noSegments,9>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
    default	:
    {
      fprintf(stderr, "ERROR: mult23 has not been templated for %i planes in a stack.\n", cStack->noInStack);
      exit(EXIT_FAILURE);
    }
  }
}

template<int64_t FLAGS>
__host__  void mult23_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuCgPlan* plan, cuFfdotStack* cStack)
{
  switch (plan->noSegments)
  {
#if MIN_SEGMENTSS <= 1  and MAX_SEGMENTSS >= 1
    case 1:
    {
      mult23_p<FLAGS,1>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTSS <= 2  and MAX_SEGMENTSS >= 2
    case 2:
    {
      mult23_p<FLAGS,2>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTSS <= 3  and MAX_SEGMENTSS >= 3
    case 3:
    {
      mult23_p<FLAGS,3>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTSS <= 4  and MAX_SEGMENTSS >= 4
    case 4:
    {
      mult23_p<FLAGS,4>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTSS <= 5  and MAX_SEGMENTSS >= 5
    case 5:
    {
      mult23_p<FLAGS,5>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTSS <= 6  and MAX_SEGMENTSS >= 6
    case 6:
    {
      mult23_p<FLAGS,6>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTSS <= 7  and MAX_SEGMENTSS >= 7
    case 7:
    {
      mult23_p<FLAGS,7>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTSS <= 8  and MAX_SEGMENTSS >= 8
    case 8:
    {
      mult23_p<FLAGS,8>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTSS <= 9  and MAX_SEGMENTSS >= 9
    case 9:
    {
      mult23_p<FLAGS,9>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTSS <= 10 and MAX_SEGMENTSS >= 10
    case 10:
    {
      mult23_p<FLAGS,10>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTSS <= 11 and MAX_SEGMENTSS >= 11
    case 11:
    {
      mult23_p<FLAGS,11>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

#if MIN_SEGMENTSS <= 12 and MAX_SEGMENTSS >= 12
    case 12:
    {
      mult23_p<FLAGS,12>(dimGrid, dimBlock, i1, multStream, plan, cStack);
      break;
    }
#endif

    default:
    {
      if      ( plan->noSegments < MIN_SEGMENTS )
	fprintf(stderr, "ERROR: In %s, # segments (%i) less than the compiled minimum %i.\n", __FUNCTION__, plan->noSegments, MIN_SEGMENTS );
      else if ( plan->noSegments > MAX_SEGMENTS )
	fprintf(stderr, "ERROR: In %s, # segment (%i) greater than the compiled maximum %i.\n", __FUNCTION__, plan->noSegments, MIN_SEGMENTS );
      else
	fprintf(stderr, "ERROR: %s has not been templated for %i segment.\n", __FUNCTION__, plan->noSegments);

      exit(EXIT_FAILURE);
    }
  }
}

#endif	// WITH_MUL_23

__host__  void mult23(cudaStream_t multStream, cuCgPlan* plan, cuFfdotStack* cStack)
{
#ifdef WITH_MUL_23
  dim3 dimGrid, dimBlock;

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY )) ;
  dimGrid.y = cStack->mulSlices ;

  if      ( plan->flags & FLAG_ITLV_ROW )
    mult23_s<FLAG_ITLV_ROW>(dimGrid, dimBlock, 0, multStream, plan, cStack);
#ifdef WITH_ITLV_PLN
  else
    mult23_s<0>(dimGrid, dimBlock, 0, multStream, plan, cStack);
#else
  else
  {
    fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
    exit(EXIT_FAILURE);
  }
#endif

#else
  EXIT_DIRECTIVE("WITH_MUL_23");
#endif
}
