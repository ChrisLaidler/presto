#include "cuda_accel_MU.h"

/** Kernel to copy powers from complex plane to in-mem plane  .
 *
 * One thread per column
 *
 */
template<typename T>
__global__ void cpyPowers_ker( T* dst, size_t  dpitch, T*  src, size_t  spitch, size_t  width, size_t  height)
{
  int ix = blockIdx.x * CPY_WIDTH + threadIdx.x ;

  for ( int iy = 0 ; iy < height; iy++)
  {
    if ( ix < width && iy < height)
    {
      dst[iy*dpitch + ix] = src[iy*spitch +ix];
    }
  }
}

/** Kernel to copy powers from complex plane to in-mem plane  .
 *
 * One thread per column
 */
template<typename T>
__global__ void cpyCmplx_ker( T* dst, size_t  dpitch, float2* src, size_t  spitch, size_t  width, size_t  height)
{
  int ix = blockIdx.x * CPY_WIDTH + threadIdx.x ;

  const int buffLen = 4;

  float buff[buffLen];

  if ( ix < width )
  {
    int iy;

    FOLD // All iterations with no height check
    {
      for ( iy = 0 ; iy < height - buffLen ; iy+=buffLen)
      {
	for ( int by = 0 ; by < buffLen; by++)
	{
	  int gy = iy + by;

	  buff[by]          = getPowerAsFloat(src, gy*spitch + ix);
	}

	for ( int by = 0 ; by < buffLen; by++)
	{
	  int gy = iy + by;

	  set(dst, gy*dpitch + ix, buff[by]);
	}
      }
    }

    FOLD // One last iteration with height checks
    {
      for ( int by = 0 ; by < buffLen; by++)
      {
	int gy = iy + by;

	if ( gy < height)
	{
	  buff[by]          = getPowerAsFloat(src, gy*spitch + ix);
	}
      }

      for ( int by = 0 ; by < buffLen; by++)
      {
	int gy = iy + by;

	if ( gy < height)
	{
	  set(dst, gy*dpitch + ix, buff[by]);
	}
      }
    }
  }
}

/** Function to call the kernel to copy powers from powers plane to in-mem plane  .
 */
template<typename T>
void cpyPowers( T* dst, size_t  dpitch, T* src, size_t  spitch, size_t  width, size_t  height, cudaStream_t  stream)
{
  dim3 dimBlock, dimGrid;

  dimBlock.x  = CPY_WIDTH;
  dimBlock.y  = 1 ;

  float ww    = width  / (float)dimBlock.x ;

  dimGrid.x   = ceil(ww);
  dimGrid.y   = 1 ;

  cpyPowers_ker<T><<<dimGrid,  dimBlock, 0, stream >>>(dst, dpitch, src, spitch, width, height);
}

/** Function to call the kernel to copy powers from powers plane to in-mem plane  .
 */
template<typename T>
void cpyCmplx( T* dst, size_t  dpitch, float2* src, size_t  spitch, size_t  width, size_t  height, cudaStream_t  stream)
{
  dim3 dimBlock, dimGrid;

  dimBlock.x  = CPY_WIDTH;
  dimBlock.y  = 1 ;

  float ww    = width  / (float)dimBlock.x ;

  dimGrid.x   = ceil(ww);
  dimGrid.y   = 1 ;

  cpyCmplx_ker<T><<<dimGrid,  dimBlock, 0, stream >>>(dst, dpitch, src, spitch, width, height);
}

/** Copy results of iFFT from powers plane to the inmem plane using 2D async memory copy
 *
 * This is done using one appropriately strided 2d memory copy for each segment of a stack
 *
 */
template<typename Tin, typename Tout>
void copyIFFTtoPln( cuCgPlan* plan, cuFfdotStack* cStack)
{
  Tout*   dst;
  Tin*    src;
  size_t  dpitch;
  size_t  spitch;
  size_t  width;
  size_t  height;

  int inSz  = 1;
  int outSz = 1;

  inSz  = sizeof(Tin);
  outSz = sizeof(Tout);

  dpitch  = plan->cuSrch->inmemStride * outSz;
  height  = cStack->height;
  spitch  = cStack->stridePower * inSz;

  // Error check
  if (cStack->noInStack > 1 )
  {
    fprintf(stderr,"ERROR: %s cannot handle stacks with more than one plane.\n", __FUNCTION__);
    exit(EXIT_FAILURE);
  }

  if ( plan->flags & FLAG_CUFFT_CB_INMEM )
  {
    // Copying was done by the callback directly
    infoMSG(5,5,"break - Copy done by callback");
    return;
  }

  for ( int sIdx = 0; sIdx < plan->noSegments; sIdx++ )
  {
    rVals* rVal	= &(*plan->rAraays)[plan->rActive][sIdx][0];

    if ( rVal->numrs )
    {
      width	= rVal->numrs;					// Width is dependent on the number of good values
      MINN( width, plan->cuSrch->inmemStride - rVal->segment * plan->accelLen -1 );	// Clamp to plane

      // Check
      size_t  end = rVal->segment * plan->accelLen + width ;
      if ( end >= plan->cuSrch->inmemStride )
      {
	fprintf(stderr,"ERROR: Data exceeds plane.\n");
	exit(EXIT_FAILURE);
      }

      width	*= outSz;
      dst	= ((Tout*)plan->cuSrch->d_planeFull)    + rVal->segment * plan->accelLen;

      if      ( plan->flags & FLAG_ITLV_ROW )
      {
	src	= ((Tin*)cStack->d_planePowr)  + cStack->stridePower*sIdx + cStack->harmInf->plnStart;
	spitch	= cStack->stridePower*plan->noSegments*inSz;
      }
#ifdef WITH_ITLV_PLN
      else
      {
	src	= ((Tin*)cStack->d_planePowr)  + cStack->stridePower*height*sIdx + cStack->harmInf->plnStart;
      }
#else
      else
      {
	fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
	exit(EXIT_FAILURE);
      }
#endif

      CUDA_SAFE_CALL(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, plan->srchStream ), "Calling cudaMemcpy2DAsync after IFFT.");
    }
  }
}

/** Copy results of the iFFT from powers plane to the inmem plane using a kernel  .
 *
 */
void cmplxToPln( cuCgPlan* plan, cuFfdotStack* cStack)
{
  float2*       src;

  size_t        dpitch;
  size_t        spitch;
  size_t        width;
  size_t        height;

  dpitch  = plan->cuSrch->inmemStride;
  width   = plan->accelLen;
  height  = cStack->height;
  spitch  = cStack->strideCmplx;

  // Error check
  if (cStack->noInStack > 1 )
  {
    fprintf(stderr,"ERROR: %s cannot handle stacks with more than one plane.\n", __FUNCTION__);
    exit(EXIT_FAILURE);
  }

  if ( plan->flags & FLAG_CUFFT_CB_INMEM )
  {
    // Copying was done by the callback directly
    return;
  }

  for ( int sIdx = 0; sIdx < plan->noSegments; sIdx++ )
  {
    rVals* rVal = &(*plan->rAraays)[plan->rActive][sIdx][0];

    if ( rVal->numrs ) // Valid segment
    {
      FOLD // Calculate striding info
      {
	// Source data location
	if ( plan->flags & FLAG_ITLV_ROW )
	{
	  src     = ((float2*)cStack->d_planePowr)  + cStack->strideCmplx*sIdx + cStack->harmInf->plnStart;
	  spitch  = cStack->strideCmplx*plan->noSegments;
	}
#ifdef WITH_ITLV_PLN
	else
	{
	  src     = ((float2*)cStack->d_planePowr)  + cStack->strideCmplx*height*sIdx + cStack->harmInf->plnStart;
	}
#else
	else
	{
	  fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
	  exit(EXIT_FAILURE);
	}
#endif
      }

      if ( plan->flags & FLAG_POW_HALF )
      {
#ifdef	WITH_HALF_RESCISION_POWERS
#if	CUDART_VERSION >= 7050
	// Each segment has its own start location in the inmem plane
	half *dst = ((half*)plan->cuSrch->d_planeFull)        + rVal->segment * plan->accelLen;

	// Call kernel
	cpyCmplx<half>(dst, dpitch, src, spitch,  width,  height, plan->srchStream );
#else	// CUDART_VERSION
	fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
	exit(EXIT_FAILURE);
#endif	// CUDART_VERSION
#else	// WITH_HALF_RESCISION_POWERS
	EXIT_DIRECTIVE("WITH_HALF_RESCISION_POWERS");
#endif	// WITH_HALF_RESCISION_POWERS
      }
      else
      {
#ifdef	WITH_SINGLE_RESCISION_POWERS
	// Each segment has its own start location in the inmem plane
	float *dst  = ((float*)plan->cuSrch->d_planeFull)        + rVal->segment * plan->accelLen;

	// Call kernel
	cpyCmplx<float>(dst, dpitch, src, spitch,  width,  height, plan->srchStream );

#else	// WITH_SINGLE_RESCISION_POWERS
	EXIT_DIRECTIVE("WITH_SINGLE_RESCISION_POWERS");
#endif	// WITH_SINGLE_RESCISION_POWERS
      }
    }
  }
}

/** Copy the complex plane to the in-memory plane  .
 *
 */
void cg_copyToInMemPln(cuCgPlan* plan)
{
  if ( plan->flags & FLAG_SS_INMEM )
  {
    if ( plan->flags & FLAG_CUFFT_CB_INMEM )
    {
      // Copying was done by the callback directly
      return;
    }

    PROF // Profiling  .
    {
      if ( (plan->flags & FLAG_PROF) )
      {
        if ( (*plan->rAraays)[plan->rActive+1][0][0].numrs )
        {
  	infoMSG(5,5,"Time previous components");

  	for (int stack = 0; stack < plan->noStacks; stack++)
  	{
  	  cuFfdotStack* cStack = &plan->stacks[stack];
  	  timeEvents( cStack->ifftMemInit, cStack->ifftMemComp, &plan->compTime[NO_STKS*COMP_GEN_D2D + stack ],  "Copy to full plane");
  	}
        }
      }
    }

    if ( (*plan->rAraays)[plan->rActive][0][0].numrs )
    {

      infoMSG(2,2,"Copy powers to in-mem plane - Iteration %3i.", (*plan->rAraays)[plan->rActive][0][0].iteration);

      cuFfdotStack* cStack = plan->stacks;

      PROF // Profiling  .
      {
	NV_RANGE_PUSH("CPY2IM");
      }

      // Error check
      if (plan->noStacks > 1 )
      {
	fprintf(stderr,"ERROR: %s cannot handle a family with more than one plane.\n", __FUNCTION__);
	exit(EXIT_FAILURE);
      }

      FOLD // Copy back data  .
      {
	FOLD // Synchronisation  .
	{
	  infoMSG(5,5,"Synchronise stream %s on %s.\n", "srchStream", "ifftComp");

	  CUDA_SAFE_CALL(cudaStreamWaitEvent(plan->srchStream, cStack->ifftComp,    0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
	}

	PROF // Profiling  .
	{
	  if ( plan->flags & FLAG_PROF )
	  {
	    infoMSG(5,5,"Event %s in %s.\n", "ifftMemInit", "srchStream");

	    CUDA_SAFE_CALL(cudaEventRecord(cStack->ifftMemInit, plan->srchStream),"Recording event: ifftMemInit");
	  }
	}

	FOLD // Copy memory on the device  .
	{
	  if ( plan->flags & FLAG_CUFFT_CB_POW )
	  {
	    infoMSG(4,4,"2D async D2D memory copy");

	    // Copy memory using a 2D async memory copy
	    if ( plan->flags & FLAG_POW_HALF )
	    {
#ifdef	WITH_HALF_RESCISION_POWERS
#if 	CUDART_VERSION >= 7050
	      copyIFFTtoPln<half,half>( plan, cStack );
#else
	      fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
	      exit(EXIT_FAILURE);
#endif
#else	// WITH_HALF_RESCISION_POWERS
	      EXIT_DIRECTIVE("WITH_HALF_RESCISION_POWERS");
#endif	// WITH_HALF_RESCISION_POWERS
	    }
	    else
	    {
#ifdef	WITH_SINGLE_RESCISION_POWERS
	      copyIFFTtoPln<float, float>( plan, cStack );
#else	// WITH_SINGLE_RESCISION_POWERS
	      EXIT_DIRECTIVE("WITH_SINGLE_RESCISION_POWERS");
#endif	// WITH_SINGLE_RESCISION_POWERS
	    }
	  }
	  else
	  {
	    infoMSG(4,4,"Kernel memory copy\n");

	    // Use kernel to copy powers from powers plane to the inmem plane
	    cmplxToPln( plan, cStack );
	  }

	  CUDA_SAFE_CALL(cudaGetLastError(), "At IFFT - cg_copyToInMemPln");
	}

	FOLD // Synchronisation  .
	{
	  infoMSG(5,5,"Event %s in %s.\n", "ifftMemComp", "srchStream");

	  CUDA_SAFE_CALL(cudaEventRecord(cStack->ifftMemComp, plan->srchStream),"Recording event: ifftMemComp");
	}
      }

      PROF // Profiling  .
      {
	NV_RANGE_POP("CPY2IM");
      }
    }
  }
}
