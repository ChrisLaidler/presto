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
 *     Added preprocessor directives for steps and chunks
 *
 *  [2017-05-05]
 *	Kernel optimisations with a focus on reducing register use
 *
 */


#include "cuda_accel_SS.h"

#define SS31_X		16			// X Thread Block
#define SS31_Y		8			// Y Thread Block
#define SS31BS		(SS31_X*SS31_Y)


__device__ inline int getOffset(const int stage, const int step, const int strd1, const int oStride, const int sid)
{
  return stage*gridDim.y*strd1 + blockIdx.y*strd1 + step*ALEN + sid ;		// 1 - This is the original method that "packs" the steps into contiguous sections
  //return stage*gridDim.y*strd1 + blockIdx.y*strd1 + step*oStride + sid ;	// 2
}

#ifdef WITH_SAS_31

/** Sum and Search - loop down - column max - multi-step - step outer .
 *
 * The first thread (sid 0) loops down the first "good" column of the plane and sums and searches
 * It writes its results to the 0 spot in the results array
 * The strided padding at the end of the array can be use to count per block counts
 *
 * @param width		The width of a single step (this is usually accellen)
 * @param d_cands	Address of device output memory
 * @param oStride	The stride (in candPZs) of the output memory (this is a per step stride) there is a funky diagram to show how the output memory is laid out
 *
 * @param powersArr
 */
template<typename T, int64_t FLAGS, const int noStages, const int noHarms, typename pArr, const int cunkSize, const int noSteps>
__global__ void
add_and_searchCU31_k(const uint width, candPZs* d_cands, const int oStride, pArr powersArr, int* d_counts)
{
  const int tidx	= threadIdx.y * SS31_X     +  threadIdx.x;	///< Thread index within in the block
  const int sid		= blockIdx.x  * SS31BS     +  tidx;		///< The index in the step where 0 is the first 'good' column in the fundamental plane

#ifdef WITH_SAS_COUNT	// Zero SM  .
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
    candPZs	candLists	[noStages][noSteps];			///< Best candidates found thus far - This "waists" registers as the short is padded up, I tried separate arrays but that was slower
    float	powers		[cunkSize][noSteps];			///< Registers to hold values to increase mem cache hits

    FOLD // Prep - Initialise the x indices & set candidates to 0  .
    {
      // Calculate the x indices or create a pointer offset by the correct amount
      for ( int harm = 0; harm < noHarms; harm++ )			// loop over harmonic  .
      {
	//// NB NOTE: the indexing below assume each plane starts on a multiple of noHarms
	int   ix		= lround_t( sid*FRAC_STAGE[harm] ) + PSTART_STAGE[harm] ;

	// Stride plane pointers
	powersArr.val[harm]	= (void*)( (T*)powersArr.val[harm] + ix );
      }

      FOLD  // Set the local and return candidate powers to zero  .
      {
	int xStride = noSteps*oStride ;

	for ( int stage = 0; stage < noStages; stage++ )
	{
	  for ( int step = 0; step < noSteps; step++)			// Loop over steps  .
	  {
	    candLists[stage][step].value = 0 ;
	    d_cands[getOffset(stage, step, xStride, oStride, sid) ].value = 0;
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
	    for ( int step = 0; step < noSteps; step++ )		// Loop over steps  .
	    {
	      powers[yPlus][step]	= 0 ;
	    }
	  }
	}

	FOLD // Loop over stages, sum and search  .
	{
	  for ( int stage = 0 ; stage < noStages; stage++)		// Loop over stages  .
	  {
	    short start = STAGE[stage][0] ;
	    short end   = STAGE[stage][1] ;

	    FOLD // Create a section of summed powers one for each step  .
	    {
	      for ( int harm = start; harm <= end; harm++ )		// Loop over harmonics (batch) in this stage  .
	      {
		short	iyP	= -1;					///< yIndex of the last power read from DRAM
		T	pow[noSteps];					///< A buffer of the previous powers

		for( short yPlus = 0; yPlus < cunkSize; yPlus++ )	// Loop over the chunk  .
		{
		  short iy1	= YINDS[ (zeroHeight+INDS_BUFF)*harm + y + yPlus ];
		  int iy2;

		  if ( (iyP != iy1) )					// Only read power if it is not the same as the previous  .
		  {
		    for ( int step = 0; step < noSteps; step++)		// Loop over steps  .
		    {
		      FOLD // Calculate index  .
		      {
#ifdef WITH_ITLV_PLN
			if        ( FLAGS & FLAG_ITLV_ROW )
#endif
			{
			  iy2     = ( iy1 * noSteps + step ) * STRIDE_STAGE[harm];
			}
#ifdef WITH_ITLV_PLN
			else
			{
			  iy2     = ( iy1 + step * HEIGHT_STAGE[harm] ) * STRIDE_STAGE[harm] ;
			}
#endif
		      }

		      FOLD // Read powers  .
		      {
			pow[step] = ((T*)powersArr[harm])[iy2];
		      }
		    }

		    iyP = iy1;
		  }

		  FOLD // Accumulate powers  .
		  {
		    for ( short step = 0; step < noSteps; step++)	// Loop over steps  .
		    {
		      powers[yPlus][step] += getFloat(pow[step]);
		    }
		  }
		}
	      }
	    }

	    FOLD // Search set of powers  .
	    {
	      for ( short step = 0; step < noSteps; step++)		// Loop over steps  .
	      {
		float pow;
		float maxP = POWERCUT_STAGE[stage];
		short maxI;

		for( short yPlus = 0; yPlus < cunkSize ; yPlus++ )	// Loop over section  .
		{
		  pow = powers[yPlus][step];

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
		  if ( maxP > candLists[stage][step].value )
		  {
		    // This is our new max!
		    candLists[stage][step].value	= maxP;
		    candLists[stage][step].z		= maxI;
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
      int xStride = noSteps*oStride ;

      for ( int stage = 0 ; stage < noStages; stage++)			// Loop over stages  .
      {
	for ( int step = 0; step < noSteps; step++)			// Loop over steps  .
	{
	  if  ( candLists[stage][step].value > POWERCUT_STAGE[stage] )
	  {
	    // Write to DRAM
	    d_cands[getOffset(stage, step, xStride, oStride, sid) ] = candLists[stage][step];

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

    __syncthreads();			// Make sure cnt has been zeroed

    if ( conts)				// Increment block specific counts in SM  .
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
__host__ void add_and_searchCU31_s(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  pArr   powers;
  int* d_cnts	= NULL;

  for (int i = 0; i < noHarms; i++)
  {
    int sIdx        = batch->cuSrch->sIdx[i];
    powers.val[i]   = batch->planes[sIdx].d_planePowr;
  }

#ifdef WITH_SAS_COUNT
  if ( batch->flags & FLAG_SS_COUNT)
  {
    d_cnts	= (int*)((char*)batch->d_outData1 + batch->cndDataSize);
  }
#endif

  switch (batch->noSteps)
  {

#if MIN_STEPS <= 1  and MAX_STEPS >= 1
    case 1:
    {
      add_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize,1><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_STEPS <= 2  and MAX_STEPS >= 2
    case 2:
    {
      add_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize,2><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_STEPS <= 3  and MAX_STEPS >= 3
    case 3:
    {
      add_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize,3><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_STEPS <= 4  and MAX_STEPS >= 4
    case 4:
    {
      add_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize,4><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_STEPS <= 5  and MAX_STEPS >= 5
    case 5:
    {
      add_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize,5><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_STEPS <= 6  and MAX_STEPS >= 6
    case 6:
    {
      add_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize,6><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_STEPS <= 7  and MAX_STEPS >= 7
    case 7:
    {
      add_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize,7><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_STEPS <= 8  and MAX_STEPS >= 8
    case 8:
    {
      add_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize,8><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_STEPS <= 9  and MAX_STEPS >= 9
    case 9:
    {
      add_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize,9><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_STEPS <= 10 and MAX_STEPS >= 10
    case 10:
    {
      add_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize,10><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_STEPS <= 11 and MAX_STEPS >= 11
    case 11:
    {
      add_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize,11><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
#endif

#if MIN_STEPS <= 12 and MAX_STEPS >= 12
    case 12:
    {
      add_and_searchCU31_k< T, FLAGS, noStages, noHarms, pArr, cunkSize,12><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
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

template< typename T, int64_t FLAGS, int noStages, const int noHarms, typename pArr>
__host__ void add_and_searchCU31_c(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  switch ( batch->ssChunk )
  {
#if MIN_SAS_CHUNK <= 1  and MAX_SAS_CHUNK >= 1
    case 1 :
    {
      add_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 1>(dimGrid, dimBlock, stream, batch);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 2  and MAX_SAS_CHUNK >= 2
    case 2 :
    {
      add_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 2>(dimGrid, dimBlock, stream, batch);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 3  and MAX_SAS_CHUNK >= 3
    case 3 :
    {
      add_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 3>(dimGrid, dimBlock, stream, batch);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 4  and MAX_SAS_CHUNK >= 4
    case 4 :
    {
      add_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 4>(dimGrid, dimBlock, stream, batch);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 5  and MAX_SAS_CHUNK >= 5
    case 5 :
    {
      add_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 5>(dimGrid, dimBlock, stream, batch);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 6  and MAX_SAS_CHUNK >= 6
    case 6 :
    {
      add_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 6>(dimGrid, dimBlock, stream, batch);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 7  and MAX_SAS_CHUNK >= 7
    case 7 :
    {
      add_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 7>(dimGrid, dimBlock, stream, batch);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 8  and MAX_SAS_CHUNK >= 8
    case 8 :
    {
      add_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 8>(dimGrid, dimBlock, stream, batch);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 9  and MAX_SAS_CHUNK >= 9
    case 9 :
    {
      add_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 9>(dimGrid, dimBlock, stream, batch);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 10 and MAX_SAS_CHUNK >= 10
    case 10 :
    {
      add_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 10>(dimGrid, dimBlock, stream, batch);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 11 and MAX_SAS_CHUNK >= 11
    case 11 :
    {
      add_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 11>(dimGrid, dimBlock, stream, batch);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 12 and MAX_SAS_CHUNK >= 12
    case 12 :
    {
      add_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 12>(dimGrid, dimBlock, stream, batch);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 13 and MAX_SAS_CHUNK >= 13
    case 13 :
    {
      add_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 13>(dimGrid, dimBlock, stream, batch);
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 14 and MAX_SAS_CHUNK >= 14
    case 14 :
    {
      add_and_searchCU31_s< T, FLAGS, noStages, noHarms, pArr, 14>(dimGrid, dimBlock, stream, batch);
      break;
    }
#endif

    default:
    {
      if ( batch->ssChunk < MIN_SAS_CHUNK )
	fprintf(stderr, "ERROR: In %s, chunk size (%i) less than the compiled minimum %i.\n", __FUNCTION__, batch->ssChunk, MIN_SAS_CHUNK );
      else if ( batch->ssChunk > MAX_SAS_CHUNK )
	fprintf(stderr, "ERROR: In %s, chunk size (%i) greater than the compiled maximum %i.\n", __FUNCTION__, batch->ssChunk, MIN_SAS_CHUNK );
      else
	fprintf(stderr, "ERROR: %s has not been templated for %i chunk size.\n", __FUNCTION__, batch->ssChunk);

      exit(EXIT_FAILURE);
    }
  }
}

template<typename T, int64_t FLAGS >
__host__ void add_and_searchCU31_p(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int noStages = batch->noHarmStages;

  switch (noStages)
  {
    case 1 :
    {
      add_and_searchCU31_c< T, FLAGS, 1, 1, ptr01>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 2 :
    {
      add_and_searchCU31_c< T, FLAGS, 2, 2, ptr02>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 3 :
    {
      add_and_searchCU31_c< T, FLAGS, 3, 4, ptr04>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 4 :
    {
      add_and_searchCU31_c< T, FLAGS, 4, 8, ptr08>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 5 :
    {
      add_and_searchCU31_c< T, FLAGS, 5, 16, ptr16>(dimGrid, dimBlock, stream, batch);
      break;
    }
    default:
      fprintf(stderr, "ERROR: %s has not been templated for %i stages\n", __FUNCTION__, noStages);
      exit(EXIT_FAILURE);
  }
}

template<typename T>
__host__ void add_and_searchCU31_f(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int64_t FLAGS = batch->flags;

  FOLD // Call flag template  .
  {
    if      ( FLAGS & FLAG_ITLV_ROW )
    {
      add_and_searchCU31_p<T, FLAG_ITLV_ROW>    (dimGrid, dimBlock, stream, batch);
    }
#ifdef WITH_ITLV_PLN
    else
    {
      add_and_searchCU31_p<T, 0>                (dimGrid, dimBlock, stream, batch);
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

__host__ void add_and_searchCU31( cudaStream_t stream, cuFFdotBatch* batch )
{
#ifdef 	WITH_SAS_31

  dim3 dimBlock, dimGrid;

  rVals* rVal = &(*batch->rAraays)[batch->rActive][0][0];

  dimBlock.x		= SS31_X;
  dimBlock.y		= SS31_Y;

  float bw		= SS31BS;
  float ww		= batch->strideOut / ( bw );

  dimGrid.x		= ceil(ww);
  dimGrid.y		= batch->ssSlices;

  rVal->noBlocks	= dimGrid.x * dimGrid.y;

  if( rVal->noBlocks > MAX_SAS_BLKS )
  {
    fprintf(stderr, "ERROR: Too many blocks in sum and search kernel, try reducing SS_SLICES %i > %i. (in function %s in %s )\n", rVal->noBlocks, MAX_SAS_BLKS, __FUNCTION__, __FILE__);
    exit(EXIT_FAILURE);
  }

  infoMSG(7,7," no ThreadBlocks %i  width %i  stride %i  remainder: %i ", dimBlock.x * dimBlock.y, batch->accelLen, batch->strideOut,  (batch->strideOut-batch->accelLen)*2);

  if      ( batch->flags & FLAG_POW_HALF	)	// CUFFT callbacks using half powers  .
  {
#ifdef	WITH_HALF_RESCISION_POWERS
#if 	CUDA_VERSION >= 7050
    add_and_searchCU31_f<half>        (dimGrid, dimBlock, stream, batch );
#else	// CUDA_VERSION
    fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
    exit(EXIT_FAILURE);
#endif	// CUDA_VERSION
#else	// WITH_HALF_RESCISION_POWERS
    EXIT_DIRECTIVE("WITH_HALF_RESCISION_POWERS");
#endif	// WITH_HALF_RESCISION_POWERS
  }
  else if ( batch->flags & FLAG_CUFFT_CB_POW	)	// CUFFT callbacks use float powers  .
  {
#ifdef	WITH_SINGLE_RESCISION_POWERS
    add_and_searchCU31_f<float>       (dimGrid, dimBlock, stream, batch );
#else	// WITH_SINGLE_RESCISION_POWERS
    EXIT_DIRECTIVE("WITH_HALF_RESCISION_POWERS");
#endif	// WITH_SINGLE_RESCISION_POWERS
  }
  else							// NO CUFFT callbacks so use complex values  .
  {
#ifdef	WITH_COMPLEX_POWERS
    add_and_searchCU31_f<fcomplexcu>  (dimGrid, dimBlock, stream, batch );
#else	// WITH_COMPLEX_POWERS
    EXIT_DIRECTIVE("WITH_COMPLEX_POWERS");
#endif	// WITH_COMPLEX_POWERS
  }

#else	// WITH_SAS_31
  EXIT_DIRECTIVE("WITH_SAS_31");
#endif	// WITH_SAS_31
}
