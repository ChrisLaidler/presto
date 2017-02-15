/** @file cuda_accel_SS_31.cu
 *  @brief The implimentation fo the standard sum and search kernel
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
 *     Added per block counts of canidates found
 *
 *  [0.0.02] [2017-02-15]
 *     Fixed an inexplicable bug with the autonomic add
 */


 #include "cuda_accel_SS.h"

#define SS31_X		16			// X Thread Block
#define SS31_Y		8			// Y Thread Block
#define SS31BS		(SS31_X*SS31_Y)


__device__ inline int getOffset(const int stage, const int step, const int strd1, const int oStride, const int sid)
{
  //return stage*gridDim.y*strd1 + blockIdx.y*strd1 + step*ALEN + sid ;		// 1 - This is the orrigional methoud that "packs" the steps into contiguous sections
  return stage*gridDim.y*strd1 + blockIdx.y*strd1 + step*oStride + sid ;	// 2

}

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
template<typename T, int64_t FLAGS, const int noStages, const int noHarms, const int cunkSize, const int noSteps>
__global__ void add_and_searchCU31(const uint width, candPZs* d_cands, const int oStride, vHarmList powersArr, int* d_counts)
{
  const int bidx	= blockIdx.y * gridDim.x  +  blockIdx.x;	///< Block index
  const int tidx	= threadIdx.y * SS31_X  +  threadIdx.x;		///< Thread index within in the block
  const int sid		= blockIdx.x  * SS31BS  +  tidx;		///< The index in the step where 0 is the first 'good' column in the fundamental plane

  uint 		conts	= 0;						///< Per thread count of candidates found
  __shared__ uint  cnt;							///< Block count of candidates

  FOLD  // Zero SM  .
  {
    if ( tidx == 0 )
    {
      cnt = 0;
    }
  }

  if ( sid < width )
  {
    const int zeroHeight  = HEIGHT_STAGE[0];

    // This is why the parameters need to be templated
    int             inds	[noHarms];
    candPZs         candLists	[noStages][noSteps];
    float           powers	[noSteps][cunkSize];			///< registers to hold values to increase mem cache hits
    T*              array	[noHarms];				///< A pointer array

    FOLD // Set the values of the pointer array  .
    {
      for ( int i = 0; i < noHarms; i++)
      {
	array[i] = (T*)powersArr[i];
      }
    }

    FOLD // Prep - Initialise the x indices & set candidates to 0  .
    {
      // Calculate the x indices or create a pointer offset by the correct amount
      for ( int harm = 0; harm < noHarms; harm++ )			// loop over harmonic  .
      {
	//// NB NOTE: the indexing below assume each plane starts on a multiple of noHarms
	int   ix        = lround_t( sid*FRAC_STAGE[harm] ) + PSTART_STAGE[harm] ;
	inds[harm]      = ix;
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
      short   lDepth  = ceilf(zeroHeight/(float)gridDim.y);
      short   y0      = lDepth*blockIdx.y;
      short   y1      = MIN(y0+lDepth, zeroHeight);

      for( short y = y0; y < y1 ; y += cunkSize )			// loop over chunks  .
      {
	FOLD // Initialise powers for each section column to 0  .
	{
	  for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )		// Loop over powers  .
	  {
	    for ( int step = 0; step < noSteps; step++ )		// Loop over steps  .
	    {
	      powers[step][yPlus]       = 0 ;
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
		//float*  t     = powersArr[harm];
		int     ix1   = inds[harm] ;
		int     ix2   = ix1;
		short   iyP   = -1;
		float   pow[noSteps];

		for( short yPlus = 0; yPlus < cunkSize; yPlus++ )	// Loop over the chunk  .
		{
		  short trm     = y + yPlus ;				///< True Y index in plane
		  short iy1     = YINDS[ (zeroHeight+INDS_BUFF)*harm + trm ];
		  //  OR
		  //int iy1     = roundf( (HEIGHT_STAGE[harm]-1.0)*trm/(float)(zeroHeight-1.0) ) ;

		  int iy2;

		  if ( (iyP != iy1) )					// Only read power if it is not the same as the previous  .
		  {
		    for ( int step = 0; step < noSteps; step++)		// Loop over steps  .
		    {
		      FOLD // Calculate index  .
		      {
			if        ( FLAGS & FLAG_ITLV_ROW )
			{
			  ix2     = ix1 + step    * STRIDE_STAGE[harm] ;
			  iy2     = iy1 * noSteps * STRIDE_STAGE[harm];
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
			pow[step] = getPower(array[harm], iy2 + ix2 );
		      }
		    }

		    iyP = iy1;
		  }

		  FOLD // Accumulate powers  .
		  {
		    for ( short step = 0; step < noSteps; step++)	// Loop over steps  .
		    {
		      powers[step][yPlus] += pow[step];
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
		  pow = powers[step][yPlus];

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
		    candLists[stage][step].value  = maxP;
		    candLists[stage][step].z      = maxI;
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

      for ( int step = 0; step < noSteps; step++)			// Loop over steps  .
      {
	for ( int stage = 0 ; stage < noStages; stage++)		// Loop over stages  .
	{
	  if  ( candLists[stage][step].value > POWERCUT_STAGE[stage] )
	  {
	    // Write to DRAM
	    d_cands[getOffset(stage, step, xStride, oStride, sid) ] = candLists[stage][step];
	    conts++;
	  }
	}
      }
    }
  }

  FOLD // Counts using SM  .
  {
    // NOTE: Could do an inital warp level recuse here but not really nessesary

    __syncthreads();			// Make sure cnt has been zeroed

    if ( conts)				// Increment block specific counts in SM  .
    {
      atomicAdd(&cnt, conts);
    }

    __syncthreads();			// Make sure autonomic adds are viable

    if ( (tidx == 0) && cnt )		// Write SM count back to main memory  .
    {
      d_counts[bidx] = cnt;
    }
  }
}

template< typename T, int64_t FLAGS, int noStages, const int noHarms, const int cunkSize>
__host__ void add_and_searchCU31_q(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int noSteps = batch->noSteps ;

  vHarmList   powers;

  for (int i = 0; i < noHarms; i++)
  {
    int sIdx        = batch->cuSrch->sIdx[i];
    powers.val[i]   = batch->planes[sIdx].d_planePowr;
  }

  int* d_cnts	= (int*)((char*)batch->d_outData1 + batch->cndDataSize);

  switch (noSteps)
  {
    case 1:
    {
      add_and_searchCU31< T, FLAGS,noStages,noHarms,cunkSize,1><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
    case 2:
    {
      add_and_searchCU31< T, FLAGS,noStages,noHarms,cunkSize,2><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
    case 3:
    {
      add_and_searchCU31< T, FLAGS,noStages,noHarms,cunkSize,3><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
    case 4:
    {
      add_and_searchCU31< T, FLAGS,noStages,noHarms,cunkSize,4><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
    case 5:
    {
      add_and_searchCU31< T, FLAGS,noStages,noHarms,cunkSize,5><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
    case 6:
    {
      add_and_searchCU31< T, FLAGS,noStages,noHarms,cunkSize,6><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
    case 7:
    {
      add_and_searchCU31< T, FLAGS,noStages,noHarms,cunkSize,7><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
    case 8:
    {
      add_and_searchCU31< T, FLAGS,noStages,noHarms,cunkSize,8><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, d_cnts);
      break;
    }
    default:
      fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for %i steps\n", noSteps);
      exit(EXIT_FAILURE);
  }
}

template< typename T, int64_t FLAGS, int noStages, const int noHarms>
__host__ void add_and_searchCU31_c(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  switch ( batch->ssChunk )
  {
    case 1 :
    {
      add_and_searchCU31_q< T, FLAGS,noStages,noHarms,1>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 2 :
    {
      add_and_searchCU31_q< T, FLAGS,noStages,noHarms,2>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 3 :
    {
      add_and_searchCU31_q< T, FLAGS,noStages,noHarms,3>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 4 :
    {
      add_and_searchCU31_q< T, FLAGS,noStages,noHarms,4>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 5 :
    {
      add_and_searchCU31_q< T, FLAGS,noStages,noHarms,5>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 6 :
    {
      add_and_searchCU31_q< T, FLAGS,noStages,noHarms,6>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 7 :
    {
      add_and_searchCU31_q< T, FLAGS,noStages,noHarms,7>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 8 :
    {
      add_and_searchCU31_q< T, FLAGS,noStages,noHarms,8>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 9 :
    {
      add_and_searchCU31_q< T, FLAGS,noStages,noHarms,9>(dimGrid, dimBlock, stream, batch);
      break;
    }
    //    case 10:
    //    {
    //      add_and_searchCU31_q< T, FLAGS,noStages,noHarms,10>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    //    case 12:
    //    {
    //      add_and_searchCU31_q< T, FLAGS,noStages,noHarms,12>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    //    case 14:
    //    {
    //      add_and_searchCU31_q< T, FLAGS,noStages,noHarms,14>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    //    case 16:
    //    {
    //      add_and_searchCU31_q< T, FLAGS,noStages,noHarms,16>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    //    case 18:
    //    {
    //      add_and_searchCU31_q< T, FLAGS,noStages,noHarms,18>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    //    case 20:
    //    {
    //      add_and_searchCU31_q< T, FLAGS,noStages,noHarms,20>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    //    case 24:
    //    {
    //      add_and_searchCU31_q< T, FLAGS,noStages,noHarms,24>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    default:
      fprintf(stderr, "ERROR: %s has not been templated for %i chunk size.\n", __FUNCTION__, batch->ssChunk);
      exit(EXIT_FAILURE);
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
      add_and_searchCU31_c< T, FLAGS,1,1>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 2 :
    {
      add_and_searchCU31_c< T, FLAGS,2,2>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 3 :
    {
      add_and_searchCU31_c< T, FLAGS,3,4>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 4 :
    {
      add_and_searchCU31_c< T, FLAGS,4,8>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 5 :
    {
      add_and_searchCU31_c< T, FLAGS,5,16>(dimGrid, dimBlock, stream, batch);
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

__host__ void add_and_searchCU31( cudaStream_t stream, cuFFdotBatch* batch )
{
  dim3 dimBlock, dimGrid;

  rVals* rVal = &(*batch->rAraays)[batch->rActive][0][0];

  dimBlock.x		= SS31_X;
  dimBlock.y		= SS31_Y;

  float bw		= SS31BS;
  //float ww		= batch->accelLen / ( bw );
  float ww		= batch->strideOut / ( bw );	// DBG

  dimGrid.x		= ceil(ww);
  dimGrid.y		= batch->ssSlices;

  rVal->noBlocks	= dimGrid.x * dimGrid.y;

  if( rVal->noBlocks > MAX_SAS_BLKS )
  {
    fprintf(stderr, "ERROR: Too many blocks in sum and search kernel, try reducing SS_SLICES %i > %i. (in function %s in %s )\n", rVal->noBlocks, MAX_SAS_BLKS, __FUNCTION__, __FILE__);
    exit(EXIT_FAILURE);
  }

  infoMSG(7,7," no ThreadBlocks %i  width %i  stride %i  remainder: %i ", dimBlock.x * dimBlock.y, batch->accelLen, batch->strideOut,  (batch->strideOut-batch->accelLen)*2);

  if      ( batch->flags & FLAG_POW_HALF         )
  {
#if CUDA_VERSION >= 7050
    add_and_searchCU31_f<half>        (dimGrid, dimBlock, stream, batch );
#else
    fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
    exit(EXIT_FAILURE);
#endif
  }
  else if ( batch->flags & FLAG_CUFFT_CB_POW )
  {
    add_and_searchCU31_f<float>       (dimGrid, dimBlock, stream, batch );
  }
  else
  {
    add_and_searchCU31_f<fcomplexcu>  (dimGrid, dimBlock, stream, batch );
  }
}
