/** @file cuda_accel_SS_30.cu
 *  @brief The implementation of the dummy "optimal" sum and search kernel
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
 *  [2017-04-17]
 *     Removed unused test kernels
 *
 */

#include "cuda_accel_SS.h"

#define SS00_X           16                   // X Thread Block
#define SS00_Y           8                    // Y Thread Block
#define SS00BS           (SS00_X*SS00_Y)

#ifdef WITH_SAS_00	// Loop down column - Zero candidates and read all needed needed memory  .

/** Sum and Search memory access only stage order - loop down column  -  Read only needed memory - sliced  .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSegments
 */
template<typename T >
__global__ void sum_and_searchCU00_k(const uint width, candPZs* d_cands, int oStride, vHarmList powersArr, const int noHarms, const int noStages, const int noSegments )
{
  const int bidx  = threadIdx.y * SS00_X  +  threadIdx.x;	/// Block index
  const int tid   = blockIdx.x  * SS00BS  +  bidx;		/// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    FOLD  // Set the local and return candidate powers to zero
    {
      int xStride = noSegments*oStride ;

      for ( int stage = 0; stage < noStages; stage++ )
      {
	for ( int sIdx = 0; sIdx < noSegments; sIdx++)		// Loop over segments
	{
	  d_cands[stage*gridDim.y*xStride + blockIdx.y*xStride + sIdx*ALEN + tid].value = 0;
	}
      }
    }

    T* array[MAX_HARM_NO];					///< A pointer array

    // Set the values of the pointer array
    for ( int harm = 0; harm < noHarms; harm++)
    {
      array[harm] = (T*)powersArr[harm] + PSTART_STAGE[harm] + tid ;
    }

    for ( int harm = 0; harm < noHarms ; harm++)		// Loop over planes
    {
      const int maxW	= ceilf(width * FRAC_STAGE[harm]);
      const int stride	= STRIDE_STAGE[harm];

      if ( tid < maxW )
      {
	uint nHeight	= HEIGHT_STAGE[harm] * noSegments;
	float tSum	= 0;
	int   lDepth	= ceilf(nHeight/(float)gridDim.y);
	int   y0	= lDepth*blockIdx.y;
	int   y1	= MIN(y0+lDepth, nHeight);

	for ( int y = y0; y < y1; y++ )
	{
	  int idx  = (y) * stride;

	  FOLD // Read  .
	  {
	    tSum += getPowerAsFloat(array[harm], idx );
	  }
	}

	if ( tSum < 0 )	// This should never be the case but needed so the compiler doesn't optimise out the sum
	{
	  printf("sum_and_searchCU00_k tSum < 0 tid: %04i  Sum: %9.5f ???\n", tid, tSum);
	}
      }
    }
  }
}

__host__ void sum_and_searchCU00_f(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuCgPlan* plan )
{
  const int   noStages  = log2((double)plan->noGenHarms) + 1 ;
  vHarmList  powers;

  for (int i = 0; i < plan->noGenHarms; i++)
  {
    int sIdx        = plan->cuSrch->sIdx[i]; // Stage order
    powers.val[i]   = plan->planes[sIdx].d_planePowr;
  }

  if      ( plan->flags & FLAG_POW_HALF         )
  {
#ifdef	WITH_HALF_RESCISION_POWERS
#if 	CUDART_VERSION >= 7050
    sum_and_searchCU00_k< half>        <<<dimGrid,  dimBlock, 0, stream >>>(plan->accelLen, (candPZs*)plan->d_outData1, plan->strideOut, powers, plan->noGenHarms, noStages, plan->noSegments  );
#else	// CUDART_VERSION
    fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
    exit(EXIT_FAILURE);
#endif	// CUDART_VERSION
#else	// WITH_HALF_RESCISION_POWERS
    EXIT_DIRECTIVE("WITH_HALF_RESCISION_POWERS");
#endif	// WITH_HALF_RESCISION_POWERS
  }
  else if ( plan->flags & FLAG_CUFFT_CB_POW )
  {
#ifdef	WITH_SINGLE_RESCISION_POWERS
    sum_and_searchCU00_k< float>       <<<dimGrid,  dimBlock, 0, stream >>>(plan->accelLen, (candPZs*)plan->d_outData1, plan->strideOut, powers, plan->noGenHarms, noStages, plan->noSegments  );
#else	// WITH_SINGLE_RESCISION_POWERS
    EXIT_DIRECTIVE("WITH_SINGLE_RESCISION_POWERS");
#endif	// WITH_SINGLE_RESCISION_POWERS
  }
  else
  {
#ifdef	WITH_COMPLEX_POWERS
    sum_and_searchCU00_k< float2>  <<<dimGrid,  dimBlock, 0, stream >>>(plan->accelLen, (candPZs*)plan->d_outData1, plan->strideOut, powers, plan->noGenHarms, noStages, plan->noSegments  );
#else	// WITH_COMPLEX_POWERS
    EXIT_DIRECTIVE("WITH_COMPLEX_POWERS");
#endif	// WITH_COMPLEX_POWERS
  }
}

#endif // WITH_SAS_00

__host__ void sum_and_searchCU00(cudaStream_t stream, cuCgPlan* plan )
{
  dim3 dimBlock, dimGrid;

  dimBlock.x  = SS00_X;
  dimBlock.y  = SS00_Y;

  float bw    = SS00BS ;
  float ww    = plan->accelLen / ( bw );

  dimGrid.x   = ceil(ww);
  dimGrid.y   = plan->ssSlices;

  if ( 0 )
  {
    // Dummy
  }
#ifdef WITH_SAS_00 // Stage order  .
  else if ( 1 )
  {
    sum_and_searchCU00_f(dimGrid,dimBlock,stream, plan );
  }
#endif
  else
  {
    fprintf(stderr, "ERROR: Code has not been compiled with Sum & Search \"optimal\" kernel." );
    exit(EXIT_FAILURE);
  }

}

