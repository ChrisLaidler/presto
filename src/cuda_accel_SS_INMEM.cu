/** @file cuda_accel_SS.cu
 *  @brief The implementation of the in-memory sum and search kernel
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
 */

#include "cuda_accel_SS.h"

#define SSIM_X		16						///< X Thread Block
#define SSIM_Y		16						///< Y Thread Block
#define SSIMBS		(SSIM_X*SSIM_Y)

#ifdef WITH_SAS_IM

template<typename T, const int noStages, const int noHarms, const int cunkSize>
__global__ void searchINMEM_k(T* read, int iStride, int oStride, int firstBin, int start, int end, candPZs* d_cands, int* d_counts )
{
  //
  const int tidx	= threadIdx.y * SSIM_X     + threadIdx.x;	///< Thread index within in the block
  const int sid		= blockIdx.x  * SSIMBS     + tidx;		///< The index in the step where 0 is the first 'good' column in the fundamental plane
  const int zeroHeight	= HEIGHT_STAGE[0];				///< The height of the fundamental plane

  int		inds      [noHarms];					///< x-indices of for each harmonic
  candPZs	candLists [noStages];					///< Device memory to store results in
  float		powers    [cunkSize];					///< registers to hold values to increase mem cache hits

  int		idx	= start + sid ;					///< The global index of the thread in the plane
  int		len	= end - start;					///< The total number of columns being handled by this kernel

#ifdef WITH_SAS_COUNT	// Zero SM  .
  const int 	bidx	= blockIdx.y  * gridDim.x  + blockIdx.x;	///< Block index
  uint 		conts = 0;						///< Per thread count of candidates found
  __shared__ uint  cnt;							///< Block count of candidates
  if ( (tidx == 0) && d_counts )
  {
    cnt = 0;
  }
#endif

  if ( sid < len )
  {
    FOLD // Set the local and return candidate powers to zero  .
    {
      for ( int stage = 0; stage < noStages; stage++ )
      {
	candLists[stage].value = 0 ;
	d_cands[stage*gridDim.y*oStride + blockIdx.y*oStride + sid].value = 0;
      }
    }

    FOLD // Prep - Initialise the x indices  .
    {
      FOLD 	// Calculate the x indices or create a pointer offset by the correct amount  .
      {
	for ( int harm = 0; harm < noHarms; harm++ )			// Loop over harmonics (batch) in this stage  .
	{
	  // TODO: check if float has large enough "integer" precision
	  int  ix	= lroundf( idx*FRAC_STAGE[harm] ) - firstBin;
	  inds[harm]	= ix;
	}
      }
    }

    FOLD // Sum & Search - Ignore contaminated ends sid to starts at correct spot  .
    {
      int   lDepth        = ceilf(zeroHeight/(float)gridDim.y);
      int   y0            = lDepth*blockIdx.y;
      int   y1            = MIN(y0+lDepth, zeroHeight);
      int   yIndsChnksz   = zeroHeight+INDS_BUFF;

      for( int y = y0; y < y1 ; y += cunkSize )                       // loop over chunks  .
      {
	FOLD // Initialise chunk of powers to zero .
	{
	  for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )            // Loop over powers  .
	    powers[yPlus] = 0;
	}

	FOLD // Loop over other stages, sum and search  .
	{
	  for ( int stage = 0 ; stage < noStages; stage++)            // Loop over stages  .
	  {
	    int start = STAGE[stage][0] ;
	    int end   = STAGE[stage][1] ;

	    FOLD	// Create a section of summed powers one for each step  .
	    {
	      for ( int harm = start; harm <= end; harm++ )         // Loop over harmonics (batch) in this stage  .
	      {
		int     ix1   = inds[harm] ;

		if ( ix1 >= 0 ) // Valid stage
		{
		  int   iyP       = -1;                             // The previous y-index used
		  float pow       = 0 ;
		  const int   yIndsStride = yIndsChnksz*harm;

		  for( int yPlus = 0; yPlus < cunkSize; yPlus++ )   // Loop over the chunk  .
		  {
		    int yPln     = y + yPlus ;                      ///< True Y index in plane

		    // Don't check yPln against zeroHeight, YINDS contains a buffer at the end, only do the check later
		    int iy1     = YINDS[ yIndsStride + yPln ];

		    if ( iyP != iy1 ) // Only read power if it is not the same as the previous  .
		    {
		      unsigned long long izz = iy1*iStride + ix1 ;
		      pow = getLong(read, izz );

		      iyP = iy1;
		    }

		    FOLD // // Accumulate powers  .
		    {
		      powers[yPlus] += pow;
		    }
		  }
		}
	      }
	    }

	    FOLD // Search set of powers  .
	    {
	      float pow;
	      float maxP = POWERCUT_STAGE[stage];
	      int maxI;

	      for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )    // Loop over section  .
	      {
		pow = powers[yPlus];
		if  ( pow > maxP )
		{
		  int idx = y + yPlus;
		  if ( idx < y1 )
		  {
		    maxP = pow;
		    maxI = idx;
		  }
		}
	      }

	      if  (  maxP > POWERCUT_STAGE[stage] )
	      {
		if ( maxP > candLists[stage].value )
		{
		  // This is our new max!
		  candLists[stage].value  = maxP;
		  candLists[stage].z      = maxI;
		}
	      }
	    }
	  }
	}
      }
    }

    FOLD // Write results back to DRAM and calculate sigma if needed  .
    {
      for ( int stage = 0 ; stage < noStages; stage++)      // Loop over stages  .
      {
	if  ( candLists[stage].value > POWERCUT_STAGE[stage] )
	{
	  // Write to DRAM
	  d_cands[stage*gridDim.y*oStride + blockIdx.y*oStride + sid] = candLists[stage];

#ifdef WITH_SAS_COUNT	// Counts using SM  .
	  conts++;
#endif
	}
      }
    }
  }

#ifdef WITH_SAS_COUNT	// Counts using SM  .
  if ( d_counts )
  {
    // NOTE: Could do an initial warp level recurse here but not really necessary

    __syncthreads();			// Make sure cnt has been zeroed

    if ( conts)			// Increment block specific counts in SM  .
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

template<typename T, const int noStages, const int noHarms>
__host__ void searchINMEM_c(cuFFdotBatch* batch )
{
  dim3 dimBlock, dimGrid;

  rVals* rVal = &(*batch->rAraays)[batch->rActive][0][0];

  FOLD // Check if we can use the specific data types in the kernel  .
  {
    // Check the length of the addressable
    double lastBin_d  = batch->cuSrch->sSpec->searchRLow*batch->conf->noResPerBin + batch->cuSrch->inmemStride ;
    double maxUint    = std::numeric_limits<int>::max();
    if ( maxUint <= lastBin_d )
    {
      fprintf(stderr, "ERROR: There is not enough precision in int in %s in %s.\n", __FUNCTION__, __FILE__ );
      exit(EXIT_FAILURE);
    }

    lastBin_d        = batch->cuSrch->inmemStride * batch->hInfos->noZ;
    double maxInt    = std::numeric_limits<unsigned long long>::max();
    if ( maxInt <= lastBin_d )
    {
      fprintf(stderr, "ERROR: There is not enough precision in unsigned long long in %s in %s.\n", __FUNCTION__, __FILE__ );
      exit(EXIT_FAILURE);
    }

  }

  int firstBin  = batch->cuSrch->sSpec->searchRLow * batch->conf->noResPerBin ;
  int start     = rVal->drlo * batch->conf->noResPerBin ;
  int end       = start + rVal->numrs;
  int noBins    = end - start;
  int* d_cnts	= NULL;

  infoMSG(6,6,"%i harms summed - r from %i to %i (%i)\n", noHarms, start, end, noBins);

  infoMSG(7,7,"Saving results in %p  counts in %p", batch->d_outData1, d_cnts);

  dimBlock.x    = SSIM_X;
  dimBlock.y    = SSIM_Y;

  dimGrid.y     = batch->ssSlices;
  dimGrid.x     = ceil(noBins / (float) SSIMBS );

  rVal->noBlocks = dimGrid.x * dimGrid.y;

  if( rVal->noBlocks > MAX_SAS_BLKS )
  {
    fprintf(stderr, "ERROR: Too many blocks in sum and search kernel, try reducing SS_INMEM_SZ or SS_SLICES %i > %i. (in function %s in %s )\n", rVal->noBlocks, MAX_SAS_BLKS, __FUNCTION__, __FILE__);
    exit(EXIT_FAILURE);
  }

#ifdef WITH_SAS_COUNT
  if ( batch->flags & FLAG_SS_COUNT)
  {
    d_cnts	= (int*)((char*)batch->d_outData1 + batch->cndDataSize);
  }
#endif

  switch ( batch->ssChunk )
  {
#if MIN_SAS_CHUNK <= 1  and MAX_SAS_CHUNK >= 1
    case 1 :
    {
      searchINMEM_k<T,noStages,noHarms,1><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->cuSrch->d_planeFull, batch->cuSrch->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1, d_cnts );
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 2  and MAX_SAS_CHUNK >= 2
    case 2 :
    {
      searchINMEM_k<T,noStages,noHarms,2><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->cuSrch->d_planeFull, batch->cuSrch->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1, d_cnts );
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 3  and MAX_SAS_CHUNK >= 3
    case 3 :
    {
      searchINMEM_k<T,noStages,noHarms,3><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->cuSrch->d_planeFull, batch->cuSrch->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1, d_cnts );
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 4  and MAX_SAS_CHUNK >= 4
    case 4 :
    {
      searchINMEM_k<T,noStages,noHarms,4><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->cuSrch->d_planeFull, batch->cuSrch->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1, d_cnts );
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 5  and MAX_SAS_CHUNK >= 5
    case 5 :
    {
      searchINMEM_k<T,noStages,noHarms,5><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->cuSrch->d_planeFull, batch->cuSrch->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1, d_cnts );
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 6  and MAX_SAS_CHUNK >= 6
    case 6 :
    {
      searchINMEM_k<T,noStages,noHarms,6><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->cuSrch->d_planeFull, batch->cuSrch->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1, d_cnts );
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 7  and MAX_SAS_CHUNK >= 7
    case 7 :
    {
      searchINMEM_k<T,noStages,noHarms,7><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->cuSrch->d_planeFull, batch->cuSrch->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1, d_cnts );
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 8  and MAX_SAS_CHUNK >= 8
    case 8 :
    {
      searchINMEM_k<T,noStages,noHarms,8><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->cuSrch->d_planeFull, batch->cuSrch->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1, d_cnts );
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 9  and MAX_SAS_CHUNK >= 9
    case 9 :
    {
      searchINMEM_k<T,noStages,noHarms,9><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->cuSrch->d_planeFull, batch->cuSrch->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1, d_cnts );
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 10  and MAX_SAS_CHUNK >= 10
    case 10 :
    {
      searchINMEM_k<T,noStages,noHarms,10><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->cuSrch->d_planeFull, batch->cuSrch->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1, d_cnts );
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 11  and MAX_SAS_CHUNK >= 11
    case 11 :
    {
      searchINMEM_k<T,noStages,noHarms,11><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->cuSrch->d_planeFull, batch->cuSrch->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1, d_cnts );
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 12  and MAX_SAS_CHUNK >= 12
    case 12 :
    {
      searchINMEM_k<T,noStages,noHarms,12><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->cuSrch->d_planeFull, batch->cuSrch->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1, d_cnts );
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 13  and MAX_SAS_CHUNK >= 13
    case 13 :
    {
      searchINMEM_k<T,noStages,noHarms,13><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->cuSrch->d_planeFull, batch->cuSrch->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1, d_cnts );
      break;
    }
#endif

#if MIN_SAS_CHUNK <= 14  and MAX_SAS_CHUNK >= 14
    case 14 :
    {
      searchINMEM_k<T,noStages,noHarms,14><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->cuSrch->d_planeFull, batch->cuSrch->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1, d_cnts );
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

template<typename T >
__host__ void searchINMEM_p(cuFFdotBatch* batch )
{
  const int noStages = batch->cuSrch->noHarmStages;

  switch (noStages)
  {
    case 1 :
    {
      searchINMEM_c<T,1,1>(batch);
      break;
    }
    case 2 :
    {
      searchINMEM_c<T,2,2>(batch);
      break;
    }
    case 3 :
    {
      searchINMEM_c<T,3,4>(batch);
      break;
    }
    case 4 :
    {
      searchINMEM_c<T,4,8>(batch);
      break;
    }
    case 5 :
    {
      searchINMEM_c<T,5,16>(batch);
      break;
    }
    default:
    {
      fprintf(stderr, "ERROR: %s has not been templated for %i stages\n", __FUNCTION__, noStages);
      exit(EXIT_FAILURE);
    }
  }
}

#endif	// WITH_SAS_IM

__host__ void add_and_search_IMMEM(cuFFdotBatch* batch )
{
#ifdef WITH_SAS_IM
  PROF // Profiling - Time previous components  .
  {
    if ( (batch->flags & FLAG_PROF) )
    {
      if ( (*batch->rAraays)[batch->rActive+1][0][0].numrs )
      {
	infoMSG(5,5,"Time previous components");

	// Inmem Sum and Search kernel
	timeEvents( batch->searchInit, batch->searchComp, &batch->compTime[NO_STKS*COMP_GEN_SS],   "Search kernel");
      }
    }
  }

  if ( (*batch->rAraays)[batch->rActive][0][0].numrs )
  {
    infoMSG(2,2,"In-mem sum and search - Iteration %3i.", (*batch->rAraays)[batch->rActive][0][0].iteration);

    FOLD // Synchronisation  .
    {
      if      ( batch->flags & FLAG_SS_INMEM )
      {
	infoMSG(5,5,"Synchronise stream %s on %s.\n", "srchStream", "searchComp");
	CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->srchStream, batch->searchComp, 0),  "Waiting on event searchComp");
      }
      else
      {
	infoMSG(5,5,"Synchronise stream %s on %s.\n", "srchStream", "candCpyComp");
	CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->srchStream, batch->candCpyComp, 0), "Waiting on event candCpyComp");
      }
    }

    PROF // Profiling event  .
    {
      if ( batch->flags & FLAG_PROF )
      {
	infoMSG(5,5,"cudaEventRecord %s in stream %s.\n", "searchInit", "srchStream");
	CUDA_SAFE_CALL(cudaEventRecord(batch->searchInit,  batch->srchStream),        "Recording event: searchInit");
      }
    }

    FOLD // Call the kernel  .
    {
      infoMSG(3,3,"S&S Kernel\n");

      if ( batch->flags & FLAG_POW_HALF  )
      {
#ifdef	WITH_HALF_RESCISION_POWERS
#if 	CUDA_VERSION >= 7050
	searchINMEM_p<half>(batch);
#else	// CUDA_VERSION
	fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
	exit(EXIT_FAILURE);
#endif	// CUDA_VERSION
#else	// WITH_HALF_RESCISION_POWERS
	EXIT_DIRECTIVE("WITH_HALF_RESCISION_POWERS");
#endif	// WITH_HALF_RESCISION_POWERS
      }
      else
      {
#ifdef	WITH_SINGLE_RESCISION_POWERS
	searchINMEM_p<float>(batch);
#else	// WITH_SINGLE_RESCISION_POWERS
	EXIT_DIRECTIVE("WITH_SINGLE_RESCISION_POWERS");
#endif	// WITH_SINGLE_RESCISION_POWERS
      }

      CUDA_SAFE_CALL(cudaGetLastError(), "Calling searchINMEM kernel.");
    }

    FOLD // Synchronisation  .
    {
      infoMSG(5,5,"cudaEventRecord %s in stream %s.\n", "searchComp", "srchStream");
      CUDA_SAFE_CALL(cudaEventRecord(batch->searchComp,  batch->srchStream),"Recording event: searchComp");
    }
  }

#else	// WITH_SAS_IM
  EXIT_DIRECTIVE("WITH_SAS_IM");
#endif	// WITH_SAS_IM
}
