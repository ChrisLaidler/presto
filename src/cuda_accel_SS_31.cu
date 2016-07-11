#include "cuda_accel_SS.h"

#define SS31_X           16                    // X Thread Block
#define SS31_Y           8                     // Y Thread Block
#define SS31BS           (SS31_X*SS31_Y)


/** Sum and Search - loop down - column max - multi-step - step outer .
 *
 * The first thread (tid 0) loops down the first "good" column of the plane and sums and searches
 * It writes its results to the 0 spot in the results array
 *
 * @param width       The width
 * @param d_cands
 * @param texs
 * @param powersArr
 */
template<typename T, int64_t FLAGS, const int noStages, const int noHarms, const int cunkSize, const int noSteps>
__global__ void add_and_searchCU31(const uint width, candPZs* d_cands, const int oStride, vHarmList powersArr)
{
  const int bidx  = threadIdx.y * SS31_X  +  threadIdx.x;           ///< Block index
  const int tid   = blockIdx.x  * SS31BS  +  bidx;                  ///< Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    const int zeroHeight  = HEIGHT_STAGE[0];

    int             inds      [noHarms];
    candPZs         candLists [noStages][noSteps];
    float           powers    [noSteps][cunkSize];                  ///< registers to hold values to increase mem cache hits
    T*              array     [noHarms];                            ///< A pointer array

    FOLD // Set the values of the pointer array
    {
      for ( int i = 0; i < noHarms; i++)
      {
	array[i] = (T*)powersArr[i];
      }
    }

    FOLD // Prep - Initialise the x indices & set candidates to 0  .
    {
      // Calculate the x indices or create a pointer offset by the correct amount
      for ( int harm = 0; harm < noHarms; harm++ )                	// loop over harmonic  .
      {
	//// NOTE: the indexing below assume each plane starts on a multiple of noHarms
	int   ix        = lround_t( tid*FRAC_STAGE[harm] ) + PSTART_STAGE[harm] ;
	inds[harm]      = ix;
      }

      FOLD  // Set the local and return candidate powers to zero  .
      {
	int xStride = noSteps*oStride ;

	for ( int stage = 0; stage < noStages; stage++ )
	{
	  for ( int step = 0; step < noSteps; step++)               // Loop over steps  .
	  {
	    candLists[stage][step].value = 0 ;
	    d_cands[stage*gridDim.y*xStride + blockIdx.y*xStride + (step*ALEN + tid) ].value = 0;
	  }
	}
      }
    }

    FOLD // Sum & Search - Ignore contaminated ends tid to starts at correct spot  .
    {
      short   lDepth  = ceilf(zeroHeight/(float)gridDim.y);
      short   y0      = lDepth*blockIdx.y;
      short   y1      = MIN(y0+lDepth, zeroHeight);

      for( short y = y0; y < y1 ; y += cunkSize )                   // loop over chunks  .
      {
	FOLD // Initialise powers for each section column to 0  .
	{
	  for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )          // Loop over powers  .
	  {
	    for ( int step = 0; step < noSteps; step++)             // Loop over steps  .
	    {
	      powers[step][yPlus]       = 0 ;
	    }
	  }
	}

	FOLD // Loop over stages, sum and search  .
	{
	  for ( int stage = 0 ; stage < noStages; stage++)          // Loop over stages  .
	  {
	    short start = STAGE[stage][0] ;
	    short end   = STAGE[stage][1] ;

	    FOLD // Create a section of summed powers one for each step  .
	    {
	      for ( int harm = start; harm <= end; harm++ )         // Loop over harmonics (batch) in this stage  .
	      {
		//float*  t     = powersArr[harm];
		int     ix1   = inds[harm] ;
		int     ix2   = ix1;
		short   iyP   = -1;
		float   pow[noSteps];

		for( short yPlus = 0; yPlus < cunkSize; yPlus++ )   // Loop over the chunk  .
		{
		  short trm     = y + yPlus ;                       ///< True Y index in plane
		  short iy1     = YINDS[ (zeroHeight+INDS_BUFF)*harm + trm ];
		  //  OR
		  //int iy1     = roundf( (HEIGHT_STAGE[harm]-1.0)*trm/(float)(zeroHeight-1.0) ) ;

		  int iy2;

		  if ( (iyP != iy1) )                               // Only read power if it is not the same as the previous  .
		  {
		    for ( int step = 0; step < noSteps; step++)     // Loop over steps  .
		    {
		      FOLD // Calculate index  .
		      {
			if        ( FLAGS & FLAG_ITLV_ROW )
			{
			  ix2     = ix1 + step    * STRIDE_STAGE[harm] ;
			  iy2     = iy1 * noSteps * STRIDE_STAGE[harm];
			}
			else
			{
			  iy2     = ( iy1 + step * HEIGHT_STAGE[harm] ) * STRIDE_STAGE[harm] ;
			}
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
		    for ( short step = 0; step < noSteps; step++)   // Loop over steps  .
		    {
		      powers[step][yPlus] += pow[step];
		    }
		  }
		}
	      }
	    }

	    FOLD // Search set of powers  .
	    {
	      for ( short step = 0; step < noSteps; step++)         // Loop over steps  .
	      {
		float pow;
		float maxP = POWERCUT_STAGE[stage];
		short maxI;

		for( short yPlus = 0; yPlus < cunkSize ; yPlus++ )  // Loop over section  .
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

      for ( int step = 0; step < noSteps; step++)             	    // Loop over steps  .
      {
	for ( int stage = 0 ; stage < noStages; stage++)      	    // Loop over stages  .
	{
	  if  ( candLists[stage][step].value > POWERCUT_STAGE[stage] )
	  {
	    // Write to DRAM
	    d_cands[stage*gridDim.y*xStride + blockIdx.y*xStride + (step*ALEN + tid) ] = candLists[stage][step];
	  }
	}
      }
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

  switch (noSteps)
  {
    case 1:
    {
      add_and_searchCU31< T, FLAGS,noStages,noHarms,cunkSize,1><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers );
      break;
    }
    case 2:
    {
      add_and_searchCU31< T, FLAGS,noStages,noHarms,cunkSize,2><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers );
      break;
    }
    case 3:
    {
      add_and_searchCU31< T, FLAGS,noStages,noHarms,cunkSize,3><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers );
      break;
    }
    case 4:
    {
      add_and_searchCU31< T, FLAGS,noStages,noHarms,cunkSize,4><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers );
      break;
    }
    case 5:
    {
      add_and_searchCU31< T, FLAGS,noStages,noHarms,cunkSize,5><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers );
      break;
    }
    case 6:
    {
      add_and_searchCU31< T, FLAGS,noStages,noHarms,cunkSize,6><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers );
      break;
    }
    case 7:
    {
      add_and_searchCU31< T, FLAGS,noStages,noHarms,cunkSize,7><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers );
      break;
    }
    case 8:
    {
      add_and_searchCU31< T, FLAGS,noStages,noHarms,cunkSize,8><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers );
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
    else
    {
      add_and_searchCU31_p<T, 0>                (dimGrid, dimBlock, stream, batch);
    }
  }
}

__host__ void add_and_searchCU31( cudaStream_t stream, cuFFdotBatch* batch )
{
  dim3 dimBlock, dimGrid;

  dimBlock.x  = SS31_X;
  dimBlock.y  = SS31_Y;

  float bw    = SS31BS;
  float ww    = batch->accelLen / ( bw );

  dimGrid.x   = ceil(ww);
  dimGrid.y   = batch->ssSlices;

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
