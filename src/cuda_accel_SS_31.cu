/** @file cuda_accel_SS_31.cu
 *  @brief The implementation of the standard sum and search kernel
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
 *  [0.0.01] [2017-02-12]
 *     Added per block counts of candidates found
 *
 *  [0.0.02] [2017-02-15]
 *     Fixed an inexplicable bug with the autonomic add
 *     Added capability to optionally do count
 *     
 *  [0.0.03] [2017-02-24]
 *     Added preprocessor directives for segments and chunks
 *
 *  [2017-05-05]
 *	Kernel optimisations with a focus on reducing register use
 *
 */


#include "cuda_accel_SS.h"

#define SS31_X		16			// X Thread Block
#define SS31_Y		8			// Y Thread Block
#define SS31BS		(SS31_X*SS31_Y)


__device__ inline int getOffset(const int stage, const int sIdx, const int strd1, const int oStride, const int sid)
{
  return stage*gridDim.y*strd1 + blockIdx.y*strd1 + sIdx*ALEN + sid ;		// 1 - This is the original method that "packs" the segments into contiguous sections
  //return stage*gridDim.y*strd1 + blockIdx.y*strd1 + sIdx*oStride + sid ;	// 2
}

#ifdef WITH_SAS_31

/** Sum and Search - loop down - column max - multi-segment - segment outer .
 *
 * The first thread (sid 0) loops down the first "good" column of the plane and sums and searches
 * It writes its results to the 0 spot in the results array
 * The strided padding at the end of the array can be use to count per block counts
 *
 * @param width		The width of a single segment (this is usually accellen)
 * @param d_cands	Address of device output memory
 * @param oStride	The stride (in candPZs) of the output memory (this is a per segment stride) there is a funky diagram to show how the output memory is laid out
 *
 * @param powersArr
 */
template<typename T, int64_t FLAGS, const int noStages, const int noHarms, typename pArr, const int cunkSize, const int noSegments>
__global__ void
sum_and_searchCU31_k(const uint width, candPZs* d_cands, const int oStride, pArr powersArr, int* d_counts)
{
  const int tidx	= threadIdx.y * SS31_X     +  threadIdx.x;	///< Thread index within in the block
  const int sid		= blockIdx.x  * SS31BS     +  tidx;		///< The index in the segment where 0 is the first 'good' column in the fundamental plane

#ifdef WITH_SAS_COUNT	// Zero SM  .
  const int bidx	= blockIdx.y * gridDim.x   +  blockIdx.x;	///< Block index
  uint 		conts	= 0;						///< Per thread count of candidates found
  __shared__ uint  cnt;							///< Block count of candidates

  if ( (tidx == 0) && d_counts )
  {
    cnt = 0;
  }
#endif

  if ( sid < width )
  {
    const int zeroHeight  = HEIGHT_STAGE[0];

    // This is why the parameters need to be templated
    candPZs	candLists	[noStages][noSegments];			///< Best candidates found thus far - This "waists" registers as the short is padded up, I tried separate arrays but that was slower
    float	powers		[cunkSize][noSegments];			///< Registers to hold values to increase mem cache hits

    FOLD // Prep - Initialise the x indices & set candidates to 0  .
    {
      // Calculate the x indices or create a pointer offset by the correct amount
      for ( int harm = 0; harm < noHarms; harm++ )			// loop over harmonic  .
      {
	//// NB NOTE: the indexing below assume each plane starts on a integer thus the fundamental must start on a multiple of noHarms
	int   ix		= lround_t( sid*FRAC_STAGE[harm] ) + PSTART_STAGE[harm] ;

	// Stride plane pointers
	powersArr.val[harm]	= (void*)( (T*)powersArr.val[harm] + ix );
      }

      FOLD  // Set the local and return candidate powers to zero  .
      {
	int xStride = noSegments*oStride ;

	for ( int stage = 0; stage < noStages; stage++ )
	{
	  for ( int sIdx = 0; sIdx < noSegments; sIdx++)		// Loop over segments  .
	  {
	    candLists[stage][sIdx].value = 0 ;
	    d_cands[getOffset(stage, sIdx, xStride, oStride, sid) ].value = 0;
	  }
	}
      }
    }

    FOLD // Sum & Search - Ignore contaminated ends sid to starts at correct spot  .
    {
      short	lDepth	= ceilf(zeroHeight/(float)gridDim.y);
      short	y0	= lDepth*blockIdx.y;
      short	y1	= MIN(y0+lDepth, zeroHeight);

      for( short y = y0; y < y1 ; y += cunkSize )			// loop over chunks  .
      {
	FOLD // Initialise powers for each section column to 0  .
	{
	  for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )		// Loop over powers  .
	  {
	    for ( int sIdx = 0; sIdx < noSegments; sIdx++ )		// Loop over segments  .
	    {
	      powers[yPlus][sIdx]	= 0 ;
	    }
	  }
	}

	FOLD // Loop over stages, sum and search  .
	{
	  for ( int stage = 0 ; stage < noStages; stage++)		// Loop over stages  .
	  {
	    short start = STAGE[stage][0] ;
	    short end   = STAGE[stage][1] ;

	    FOLD // Create a section of summed powers one for each segment  .
	    {
	      for ( int harm = start; harm <= end; harm++ )		// Loop over harmonics (plan) in this stage  .
	      {
		short	iyP	= -1;					///< yIndex of the last power read from DRAM
		T	pow[noSegments];				///< A buffer of the previous powers

		for( short yPlus = 0; yPlus < cunkSize; yPlus++ )	// Loop over the chunk  .
		{
		  short iy1	= YINDS[ (zeroHeight+INDS_BUFF)*harm + y + yPlus ];
		  int iy2;

		  if ( (iyP != iy1) )					// Only read power if it is not the same as the previous  .
		  {
		    for ( int sIdx = 0; sIdx < noSegments; sIdx++)	// Loop over segments  .
		    {
		      FOLD // Calculate index  .
		      {
#ifdef WITH_ITLV_PLN
			if        ( FLAGS & FLAG_ITLV_ROW )
#endif
			{
			  iy2     = ( iy1 * noSegments + sIdx ) * STRIDE_STAGE[harm];
			}
#ifdef WITH_ITLV_PLN
			else
			{
			  iy2     = ( iy1 + sIdx * HEIGHT_STAGE[harm] ) * STRIDE_STAGE[harm] ;
			}
#endif
		      }

		      FOLD // Read powers  .
		      {
			pow[sIdx] = ((T*)powersArr[harm])[iy2];
		      }
		    }

		    iyP = iy1;
		  }

		  FOLD // Accumulate powers  .
		  {
		    for ( short sIdx = 0; sIdx < noSegments; sIdx++)	// Loop over segments  .
		    {
		      powers[yPlus][sIdx] += getFloat(pow[sIdx]);
		    }
		  }
		}
	      }
	    }

	    FOLD // Search set of powers  .
	    {
	      for ( short sIdx = 0; sIdx < noSegments; sIdx++)		// Loop over segments  .
	      {
		float pow;
		float maxP = POWERCUT_STAGE[stage];
		short maxI;

		for( short yPlus = 0; yPlus < cunkSize ; yPlus++ )	// Loop over section  .
		{
		  pow = powers[yPlus][sIdx];

		  if  ( pow > maxP )
		  {
		    short idx = y + yPlus;

		    if ( idx < y1 )
		    {
		      maxP = pow;
		      maxI = idx;
		    }
		  }
		}

		if  (  maxP > POWERCUT_STAGE[stage] )
		{
		  if ( maxP > candLists[stage][sIdx].value )
		  {
		    // This is our new max!
		    candLists[stage][sIdx].value	= maxP;
		    candLists[stage][sIdx].z		= maxI;
		  }
		}
	      }
	    }
	  }
	}
      }
    }

    FOLD // Write results back to DRAM and calculate sigma if needed  .
    {
      int xStride = noSegments*oStride ;

      for ( int stage = 0 ; stage < noStages; stage++)			// Loop over stages  .
      {
	for ( int sIdx = 0; sIdx < noSegments; sIdx++)			// Loop over segments  .
	{
	  if  ( candLists[stage][sIdx].value > POWERCUT_STAGE[stage] )
	  {
	    // Write to DRAM
	    d_cands[getOffset(stage, sIdx, xStride, oStride, sid) ] = candLists[stage][sIdx];

#ifdef WITH_SAS_COUNT	// Count thread cands  .
	    conts++;
#endif
	  }
	}
      }
    }
  }

#ifdef WITH_SAS_COUNT	// Counts using SM  .
  if ( d_counts )
  {
    // NOTE: Could do an initial warp level recurse here but not really necessary

    conts = warpReduceSum<int>(conts);

    __syncthreads();			// Make sure cnt has been zeroed

    if ( (__lidd() == 1) && conts)	// Increment block specific counts in SM  .
    {
      atomicAdd(&cnt, conts);
    }

    __syncthreads();			// Make sure autonomic adds are viable

    if ( tidx == 0 )			// Write SM count back to main memory  .
    {
      d_counts[bidx] = cnt;
    }
  }
#endif
}

template< typename T, int64_t FLAGS, int noStages, const int noHarms, typename pArr, const int cunkSize>
__host__ void sum_and_searchCU31_s(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuCgPlan* plan )
{
  pArr   powers;
  int* d_cnts	= NULL;

  for (int i = 0; i < noHarms; i++)
  {
    int sIdx        = plan->cuSrch->sIdx[i];
    powers.val[i]   = plan->planes[sIdx].d_planePowr;
  }

#ifdef WITH_SAS_COUNT
  if ( plan->flags & FLAG_SS_COUNT)
  {
    d_cnts	= (int*)((char*)plan->d_outData1 + plan->candDataSize);
  }
#endif

  switch (plan->noSegments)
  {

#if MIN_SEGMENTS <= 1  and MAX_SEGMENTS >= 1
    case 1:
    {
      sum_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize, 1><<<dimGrid,  dimBlock, 0, stream >>>(plan->accelLen, (candPZs*)plan->d_outData1, plan->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_SEGMENTS <= 2  and MAX_SEGMENTS >= 2
    case 2:
    {
      sum_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize, 2><<<dimGrid,  dimBlock, 0, stream >>>(plan->accelLen, (candPZs*)plan->d_outData1, plan->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_SEGMENTS <= 3  and MAX_SEGMENTS >= 3
    case 3:
    {
      sum_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize, 3><<<dimGrid,  dimBlock, 0, stream >>>(plan->accelLen, (candPZs*)plan->d_outData1, plan->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_SEGMENTS <= 4  and MAX_SEGMENTS >= 4
    case 4:
    {
      sum_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize, 4><<<dimGrid,  dimBlock, 0, stream >>>(plan->accelLen, (candPZs*)plan->d_outData1, plan->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_SEGMENTS <= 5  and MAX_SEGMENTS >= 5
    case 5:
    {
      sum_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize, 5><<<dimGrid,  dimBlock, 0, stream >>>(plan->accelLen, (candPZs*)plan->d_outData1, plan->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_SEGMENTS <= 6  and MAX_SEGMENTS >= 6
    case 6:
    {
      sum_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize, 6><<<dimGrid,  dimBlock, 0, stream >>>(plan->accelLen, (candPZs*)plan->d_outData1, plan->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_SEGMENTS <= 7  and MAX_SEGMENTS >= 7
    case 7:
    {
      sum_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize, 7><<<dimGrid,  dimBlock, 0, stream >>>(plan->accelLen, (candPZs*)plan->d_outData1, plan->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_SEGMENTS <= 8  and MAX_SEGMENTS >= 8
    case 8:
    {
      sum_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize, 8><<<dimGrid,  dimBlock, 0, stream >>>(plan->accelLen, (candPZs*)plan->d_outData1, plan->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_SEGMENTS <= 9  and MAX_SEGMENTS >= 9
    case 9:
    {
      sum_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize, 9><<<dimGrid,  dimBlock, 0, stream >>>(plan->accelLen, (candPZs*)plan->d_outData1, plan->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_SEGMENTS <= 10 and MAX_SEGMENTS >= 10
    case 10:
    {
      sum_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize, 10><<<dimGrid,  dimBlock, 0, stream >>>(plan->accelLen, (candPZs*)plan->d_outData1, plan->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_SEGMENTS <= 11 and MAX_SEGMENTS >= 11
    case 11:
    {
      sum_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize, 11><<<dimGrid,  dimBlock, 0, stream >>>(plan->accelLen, (candPZs*)plan->d_outData1, plan->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_SEGMENTS <= 12 and MAX_SEGMENTS >= 12
    case 12:
    {
      sum_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize,12><<<dimGrid,  dimBlock, 0, stream >>>(plan->accelLen, (candPZs*)plan->d_outData1, plan->strideOut, powers, d_cnts);
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

template< typename T, int64_t FLAGS, int noStages, const int noHarms, typename pArr>
__host__ void sum_and_searchCU31_c(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuCgPlan* plan )
{
  switch ( plan->ssChunk )
  {
#if MIN_SAS_CHUNK <= 1  and MAX_SAS_CHUNK >= 1
    case 1 :
    {
      sum_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 1>(dimGrid, dimBlock, stream, plan);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 2  and MAX_SAS_CHUNK >= 2
    case 2 :
    {
      sum_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 2>(dimGrid, dimBlock, stream, plan);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 3  and MAX_SAS_CHUNK >= 3
    case 3 :
    {
      sum_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 3>(dimGrid, dimBlock, stream, plan);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 4  and MAX_SAS_CHUNK >= 4
    case 4 :
    {
      sum_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 4>(dimGrid, dimBlock, stream, plan);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 5  and MAX_SAS_CHUNK >= 5
    case 5 :
    {
      sum_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 5>(dimGrid, dimBlock, stream, plan);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 6  and MAX_SAS_CHUNK >= 6
    case 6 :
    {
      sum_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 6>(dimGrid, dimBlock, stream, plan);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 7  and MAX_SAS_CHUNK >= 7
    case 7 :
    {
      sum_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 7>(dimGrid, dimBlock, stream, plan);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 8  and MAX_SAS_CHUNK >= 8
    case 8 :
    {
      sum_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 8>(dimGrid, dimBlock, stream, plan);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 9  and MAX_SAS_CHUNK >= 9
    case 9 :
    {
      sum_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 9>(dimGrid, dimBlock, stream, plan);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 10 and MAX_SAS_CHUNK >= 10
    case 10 :
    {
      sum_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 10>(dimGrid, dimBlock, stream, plan);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 11 and MAX_SAS_CHUNK >= 11
    case 11 :
    {
      sum_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 11>(dimGrid, dimBlock, stream, plan);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 12 and MAX_SAS_CHUNK >= 12
    case 12 :
    {
      sum_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 12>(dimGrid, dimBlock, stream, plan);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 13 and MAX_SAS_CHUNK >= 13
    case 13 :
    {
      sum_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 13>(dimGrid, dimBlock, stream, plan);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 14 and MAX_SAS_CHUNK >= 14
    case 14 :
    {
      sum_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 14>(dimGrid, dimBlock, stream, plan);
      break;
    }
#endif

    default:
    {
      if ( plan->ssChunk < MIN_SAS_CHUNK )
	fprintf(stderr, "ERROR: In %s, chunk size (%i) less than the compiled minimum %i.\n", __FUNCTION__, plan->ssChunk, MIN_SAS_CHUNK );
      else if ( plan->ssChunk > MAX_SAS_CHUNK )
	fprintf(stderr, "ERROR: In %s, chunk size (%i) greater than the compiled maximum %i.\n", __FUNCTION__, plan->ssChunk, MIN_SAS_CHUNK );
      else
	fprintf(stderr, "ERROR: %s has not been templated for %i chunk size.\n", __FUNCTION__, plan->ssChunk);

      exit(EXIT_FAILURE);
    }
  }
}

template<typename T, int64_t FLAGS >
__host__ void sum_and_searchCU31_p(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuCgPlan* plan )
{
  const int noStages = plan->noHarmStages;

  switch (noStages)
  {
    case 1 :
    {
      sum_and_searchCU31_c< T, FLAGS, 1, 1, ptr01>(dimGrid, dimBlock, stream, plan);
      break;
    }
    case 2 :
    {
      sum_and_searchCU31_c< T, FLAGS, 2, 2, ptr02>(dimGrid, dimBlock, stream, plan);
      break;
    }
    case 3 :
    {
      sum_and_searchCU31_c< T, FLAGS, 3, 4, ptr04>(dimGrid, dimBlock, stream, plan);
      break;
    }
    case 4 :
    {
      sum_and_searchCU31_c< T, FLAGS, 4, 8, ptr08>(dimGrid, dimBlock, stream, plan);
      break;
    }
    case 5 :
    {
      sum_and_searchCU31_c< T, FLAGS, 5, 16, ptr16>(dimGrid, dimBlock, stream, plan);
      break;
    }
    default:
      fprintf(stderr, "ERROR: %s has not been templated for %i stages\n", __FUNCTION__, noStages);
      exit(EXIT_FAILURE);
  }
}

template<typename T>
__host__ void sum_and_searchCU31_f(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuCgPlan* plan )
{
  const int64_t FLAGS = plan->flags;

  FOLD // Call flag template  .
  {
    if      ( FLAGS & FLAG_ITLV_ROW )
    {
      sum_and_searchCU31_p<T, FLAG_ITLV_ROW>    (dimGrid, dimBlock, stream, plan);
    }
#ifdef WITH_ITLV_PLN
    else
    {
      sum_and_searchCU31_p<T, 0>                (dimGrid, dimBlock, stream, plan);
    }
#else
    else
    {
      fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
      exit(EXIT_FAILURE);
    }
#endif
  }
}

#endif	// WITH_SAS_31

__host__ void sum_and_searchCU31( cudaStream_t stream, cuCgPlan* plan )
{
#ifdef 	WITH_SAS_31

  dim3 dimBlock, dimGrid;

  rVals* rVal = &(*plan->rAraays)[plan->rActive][0][0];

  dimBlock.x		= SS31_X;
  dimBlock.y		= SS31_Y;

  float bw		= SS31BS;
  float ww		= plan->strideOut / ( bw );

  dimGrid.x		= ceil(ww);
  dimGrid.y		= plan->ssSlices;

  rVal->noBlocks	= dimGrid.x * dimGrid.y;

  FOLD // Check first segment for divisibility  .
  {
    // Note: Could do all segments?
    double devisNo	= plan->noGenHarms;
    for ( int sIdx = 0; sIdx < plan->noSegments; sIdx++)
    {
      rVals* rVal2 = &(*plan->rAraays)[plan->rActive][sIdx][0];

      double firsrR	= rVal2->drlo;
      double rem = fmod(firsrR, devisNo);

      if ( fabs(rem) >= 1e-6 )
      {
	printf("ERROR: Invalid r-value in %s, value not divisabe %.3f %% %.3f = %.3f  \n",__FUNCTION__, firsrR, devisNo, rem);
	exit(EXIT_FAILURE);
      }
    }
  }

  if( rVal->noBlocks > MAX_SAS_BLKS )
  {
    fprintf(stderr, "ERROR: Too many blocks in sum and search kernel, try reducing SS_SLICES %i > %i. (in function %s in %s )\n", rVal->noBlocks, MAX_SAS_BLKS, __FUNCTION__, __FILE__);
    exit(EXIT_FAILURE);
  }

  infoMSG(7,7," SAS 3.1 - no ThreadBlocks %i  width %i  stride %i  remainder: %i ", dimBlock.x * dimBlock.y, plan->accelLen, plan->strideOut,  (plan->strideOut-plan->accelLen)*2);

  if      ( plan->flags & FLAG_POW_HALF	)		// CUFFT callbacks using half powers  .
  {
#ifdef	WITH_HALF_RESCISION_POWERS
#if 	CUDART_VERSION >= 7050
    sum_and_searchCU31_f<half>        (dimGrid, dimBlock, stream, plan );
#else	// CUDART_VERSION
    fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
    exit(EXIT_FAILURE);
#endif	// CUDART_VERSION
#else	// WITH_HALF_RESCISION_POWERS
    EXIT_DIRECTIVE("WITH_HALF_RESCISION_POWERS");
#endif	// WITH_HALF_RESCISION_POWERS
  }
  else if ( plan->flags & FLAG_CUFFT_CB_POW	)	// CUFFT callbacks use float powers  .
  {
#ifdef	WITH_SINGLE_RESCISION_POWERS
    sum_and_searchCU31_f<float>       (dimGrid, dimBlock, stream, plan );
#else	// WITH_SINGLE_RESCISION_POWERS
    EXIT_DIRECTIVE("WITH_HALF_RESCISION_POWERS");
#endif	// WITH_SINGLE_RESCISION_POWERS
  }
  else							// NO CUFFT callbacks so use complex values  .
  {
#ifdef	WITH_COMPLEX_POWERS
    sum_and_searchCU31_f<float2>      (dimGrid, dimBlock, stream, plan );
#else	// WITH_COMPLEX_POWERS
    EXIT_DIRECTIVE("WITH_COMPLEX_POWERS");
#endif	// WITH_COMPLEX_POWERS
  }

#else	// WITH_SAS_31
  EXIT_DIRECTIVE("WITH_SAS_31");
#endif	// WITH_SAS_31
}
