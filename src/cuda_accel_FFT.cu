/** @file cuda_accel_FFT.cu
 *  @brief Some utility functions to perform FFT related tasks (predominantly the iFFT)
 *
 *  @author Chris Laidler
 *  @bug No known bugs.
 *
 *
 *  Change Log
 *
 *  [2.02.17] [2020-02-03]
 *    Created this file - A major refactor for release
 *
 */

#include "cuda_utils.h"
#include "cuda_accel_utils.h"

#if	CUDART_VERSION >= 6050

#ifdef	WITH_POW_POST_CALLBACK	// Post Callbacks (power)  .

//================================ Constants for Callback bit packing  =======================================\\

#define	CB_STRT_WIDTH		0
#define	CB_WITH_WIDTH		4
#define	CB_MASK_WIDTH		0xF

#define	CB_STRT_NUMP		4
#define	CB_WITH_NUMP		4
#define	CB_MASK_NUMP		0xF0

#define	CB_STRT_START		8
#define	CB_WITH_START		12
#define	CB_MASK_START		0xFFF00

#define	CB_STRT_HEIGHT		20
#define	CB_WITH_HEIGHT		12
#define	CB_MASK_HEIGHT		0xFFF00000

// Second 32 bits

#define	CB_STRT_ALENG		0
#define	CB_WITH_ALENG		16
#define	CB_MASK_ALENG		0xFFFF

#define	CB_STRT_NO_SEG		16
#define	CB_WITH_NO_SEG		16
#define	CB_MASK_NO_SEG		0xFFFF0000

//========================================== Functions  ====================================================\\

/** CUFFT callback kernel to calculate and store float powers after the FFT  .
 */
template<typename T>
__device__ void CB_powerToPowerPlane( void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  // Write result (offsets are the same)
  T notthing;
  ((T*)dataOut)[offset] = getPower(element, notthing);
}

/** CUFFT callback kernel to calculate and store float powers after the FFT  .
 */
template<typename T>
__device__ void CB_powerToPowerPlane_clip( void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  uint lo;
  uint hi;
  asm("mov.b64 {%0, %1}, %2 ; " : "=r"(lo), "=r"(hi) : "l"(callerInfo));
  const uint width  = 1<<bfe(lo, CB_STRT_WIDTH, CB_WITH_WIDTH);
  const uint start  = bfe(lo, CB_STRT_START, CB_WITH_START );
  const uint aleng  = bfe(hi, CB_STRT_ALENG, CB_WITH_ALENG );
  const uint col    = offset & (width-1);

  if ( col >= start && col <= (start+aleng) )
  {
    // Write result (offsets are the same)
    T notthing;
    ((T*)dataOut)[offset] = getPower(element, notthing);
  }
}

/** CUFFT callback kernel to calculate and store half powers in the in-memory plane, after the FFT  .
 *  CallerInfo is passed as the address of the first element of the first segment in the inmem plane
 *  Assumes row interleaved data
 *
 */
template<typename T, int width, int no>
__device__ void CB_powerToInMemPlane_pow2( void *dataOut, const size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  uint lo;
  uint hi;
  asm("mov.b64 {%0, %1}, %2 ; " : "=r"(lo), "=r"(hi) : "l"(callerInfo));
  const uint start  = bfe(lo, CB_STRT_START, CB_WITH_START );
  const uint no_seg = bfe(hi, CB_STRT_NO_SEG, CB_WITH_NO_SEG );
  const uint aleng  = bfe(hi, CB_STRT_ALENG, CB_WITH_ALENG );
  const uint stride = no_seg*width*no;

  uint col    = offset & (width-1);
  uint rrr    = offset / width;

  uint pln    = rrr & (no-1);
  uint row    = rrr / (no);

  if ( col >= start && col <= (start+aleng) )
  {
    // Calculate and store power
    T notthing;
    ((T*)dataOut)[row*stride + pln*aleng + col] = getPower(element, notthing);
  }
}

/** CUFFT callback kernel to calculate and store half powers in the in-memory plane, after the FFT  .
 *  CallerInfo is passed as the address of the first element of the first segment in the inmem plane
 *  Assumes row interleaved data
 *
 */
template<typename T>
__device__ void CB_powerToInMemPlane( void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  uint lo;
  uint hi;
  asm("mov.b64 {%0, %1}, %2 ; " : "=r"(lo), "=r"(hi) : "l"(callerInfo));

  const uint width  = 1<<bfe(lo, CB_STRT_WIDTH, CB_WITH_WIDTH);
  const uint no	    = bfe(lo, CB_STRT_NUMP,  CB_WITH_NUMP  );

  if      ( width == 8192 && no == 8 )
  {
    CB_powerToInMemPlane_pow2<T, 8192, 8>(dataOut, offset, element, callerInfo, sharedPtr );
  }
  else if ( width == 4096 && no == 8 )
  {
    CB_powerToInMemPlane_pow2<T, 4096, 8>(dataOut, offset, element, callerInfo, sharedPtr );
  }
  else if ( width == 8192 && no == 4 )
  {
    CB_powerToInMemPlane_pow2<T, 8192, 4>(dataOut, offset, element, callerInfo, sharedPtr );
  }
  else if ( width == 4096 && no == 4 )
  {
    CB_powerToInMemPlane_pow2<T, 4096, 4>(dataOut, offset, element, callerInfo, sharedPtr );
  }
  else
  {
    const uint start  = bfe(lo, CB_STRT_START, CB_WITH_START );
    const uint no_seg = bfe(hi, CB_STRT_NO_SEG, CB_WITH_NO_SEG );
    const uint aleng  = bfe(hi, CB_STRT_ALENG, CB_WITH_ALENG );
    const uint stride = no_seg*width*no;

    uint col    = offset & (width-1);
    uint rrr    = offset / width;

    uint pln    = rrr % (no);
    uint row    = rrr / (no);

    if ( col >= start && col <= (start+aleng) )
    {
      // Calculate and store power
      T nothing;
      ((T*)dataOut)[row*stride + pln*aleng + col] = getPower(element, nothing);
    }
  }
}

#ifdef	WITH_SINGLE_RESCISION_POWERS
__device__ cufftCallbackStoreC d_powerRow_f         = CB_powerToPowerPlane<float>;
__device__ cufftCallbackStoreC d_powerRow_clip_f    = CB_powerToPowerPlane_clip<float>;
__device__ cufftCallbackStoreC d_inmemRow_f         = CB_powerToInMemPlane<float>;
__device__ cufftCallbackStoreC d_inmemRow_08_04_f   = CB_powerToInMemPlane_pow2<float, 8192, 4>;	// Common case so template
__device__ cufftCallbackStoreC d_inmemRow_08_08_f   = CB_powerToInMemPlane_pow2<float, 8192, 8>;	// Common case so template
__device__ cufftCallbackStoreC d_inmemRow_04_04_f   = CB_powerToInMemPlane_pow2<float, 4096, 4>;	// Common case so template
__device__ cufftCallbackStoreC d_inmemRow_04_08_f   = CB_powerToInMemPlane_pow2<float, 4096, 8>;	// Common case so template
#endif	// WITH_SINGLE_RESCISION_POWERS

#if	CUDART_VERSION >= 7050 && defined(WITH_HALF_RESCISION_POWERS)	// Half precision CUFFT power call back  .
__device__ cufftCallbackStoreC d_powerRow_h         = CB_powerToPowerPlane<half>;
__device__ cufftCallbackStoreC d_powerRow_clip_h    = CB_powerToPowerPlane_clip<half>;
__device__ cufftCallbackStoreC d_inmemRow_h         = CB_powerToInMemPlane<half>;
__device__ cufftCallbackStoreC d_inmemRow_08_04_h   = CB_powerToInMemPlane_pow2<half, 8192, 4>;		// Common case so template
__device__ cufftCallbackStoreC d_inmemRow_08_08_h   = CB_powerToInMemPlane_pow2<half, 8192, 8>;		// Common case so template
__device__ cufftCallbackStoreC d_inmemRow_04_04_h   = CB_powerToInMemPlane_pow2<half, 4096, 4>;		// Common case so template
__device__ cufftCallbackStoreC d_inmemRow_04_08_h   = CB_powerToInMemPlane_pow2<half, 4096, 8>;		// Common case so template
#endif	// CUDART_VERSION >= 7050 & WITH_HALF_RESCISION_POWERS


#endif	// WITH_POW_POST_CALLBACK

#endif	//CUDART_VERSION


/** Load the CUFFT callbacks  .
 */
acc_err copy_CuFFT_store_CBs(cuCgPlan* plan, cuFfdotStack* cStack)
{
  acc_err ret = ACC_ERR_NONE;
  PROF // Profiling  .
  {
    NV_RANGE_PUSH("CUFFT callbacks");
  }

  if ( plan->flags & FLAG_CUFFT_CB_OUT )
  {
#ifdef WITH_POW_POST_CALLBACK
    infoMSG(5,5,"Set out CB function.");

    if ( plan->flags & FLAG_POW_HALF )
    {
#ifdef	WITH_HALF_RESCISION_POWERS
#if 	CUDART_VERSION >= 7050
      if ( plan->flags & FLAG_ITLV_ROW )		// Row interleaved
      {
	if ( plan->flags & FLAG_CUFFT_CB_INMEM )	// Store powers to inmem plane
	{
	  if      (cStack->width == 8192 &&  plan->noSegments == 4)	// Special common case
	  {
	    CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &cStack->h_stCallbackPtr, d_inmemRow_08_04_h, sizeof(cufftCallbackStoreC), 0, cudaMemcpyDeviceToHost, cStack->initStream),  "Getting constant memory address.");
	  }
	  else if (cStack->width == 8192 &&  plan->noSegments == 8)	// Special common case
	  {
	    CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &cStack->h_stCallbackPtr, d_inmemRow_08_08_h, sizeof(cufftCallbackStoreC), 0, cudaMemcpyDeviceToHost, cStack->initStream),  "Getting constant memory address.");
	  }
	  else if (cStack->width == 4096 &&  plan->noSegments == 4)	// Special common case
	  {
	    CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &cStack->h_stCallbackPtr, d_inmemRow_04_04_h, sizeof(cufftCallbackStoreC), 0, cudaMemcpyDeviceToHost, cStack->initStream),  "Getting constant memory address.");
	  }
	  else if (cStack->width == 4096 &&  plan->noSegments == 8)	// Special common case
	  {
	    CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &cStack->h_stCallbackPtr, d_inmemRow_04_08_h, sizeof(cufftCallbackStoreC), 0, cudaMemcpyDeviceToHost, cStack->initStream),  "Getting constant memory address.");
	  }
	  else								// Generic
	  {
	    CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &cStack->h_stCallbackPtr, d_inmemRow_h, sizeof(cufftCallbackStoreC), 0, cudaMemcpyDeviceToHost, cStack->initStream),  "Getting constant memory address.");
	  }
	}
	else if ( plan->flags & FLAG_CUFFT_CB_POW )	// Store powers to powers plane
	{
	  CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &cStack->h_stCallbackPtr, d_powerRow_h, sizeof(cufftCallbackStoreC), 0, cudaMemcpyDeviceToHost, cStack->initStream),  "Getting constant memory address.");
	}
      }
      else						// Row interleaved
      {
	SAFE_CALL(ACC_ERR_DEPRICATED, "ERROR: Plane interleaving has been depricated.");
      }
#else	// CUDART_VERSION
      fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
      exit(EXIT_FAILURE);
#endif	//CUDART_VERSION
#else	// WITH_HALF_RESCISION_POWERS
    EXIT_DIRECTIVE("WITH_HALF_RESCISION_POWERS");
#endif	// WITH_HALF_RESCISION_POWERS
    }
    else // Single precision powers
    {
#ifdef	WITH_SINGLE_RESCISION_POWERS
      if ( plan->flags & FLAG_ITLV_ROW )		// Row interleaved
      {
	if ( plan->flags & FLAG_CUFFT_CB_INMEM )	// Store powers to in-mem plane
	{
	  if      (cStack->width == 8192 &&  plan->noSegments == 4)	// Special common case
	  {
	    CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &cStack->h_stCallbackPtr, d_inmemRow_08_04_f, sizeof(cufftCallbackStoreC), 0, cudaMemcpyDeviceToHost, cStack->initStream),  "Getting constant memory address.");
	  }
	  else if (cStack->width == 8192 &&  plan->noSegments == 8)	// Special common case
	  {
	    CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &cStack->h_stCallbackPtr, d_inmemRow_08_08_f, sizeof(cufftCallbackStoreC), 0, cudaMemcpyDeviceToHost, cStack->initStream),  "Getting constant memory address.");
	  }
	  else if (cStack->width == 4096 &&  plan->noSegments == 4)	// Special common case
	  {
	    CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &cStack->h_stCallbackPtr, d_inmemRow_04_04_f, sizeof(cufftCallbackStoreC), 0, cudaMemcpyDeviceToHost, cStack->initStream),  "Getting constant memory address.");
	  }
	  else if (cStack->width == 4096 &&  plan->noSegments == 8)	// Special common case
	  {
	    CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &cStack->h_stCallbackPtr, d_inmemRow_04_08_f, sizeof(cufftCallbackStoreC), 0, cudaMemcpyDeviceToHost, cStack->initStream),  "Getting constant memory address.");
	  }
	  else								// Generic
	  {
	    CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &cStack->h_stCallbackPtr, d_inmemRow_f, sizeof(cufftCallbackStoreC), 0, cudaMemcpyDeviceToHost, cStack->initStream),  "Getting constant memory address.");
	  }
	}
	else if ( plan->flags & FLAG_CUFFT_CB_POW )	// Store powers to powers plane
	{
	  CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &cStack->h_stCallbackPtr, d_powerRow_f, sizeof(cufftCallbackStoreC), 0, cudaMemcpyDeviceToHost, cStack->initStream),  "Getting constant memory address.");
	}
      }
      else						// Plane interleaved
      {
	SAFE_CALL(ACC_ERR_DEPRICATED, "ERROR: Plane interleaving has been depricated.");
      }
#else	// WITH_SINGLE_RESCISION_POWERS
    EXIT_DIRECTIVE("WITH_SINGLE_RESCISION_POWERS");
#endif	// WITH_SINGLE_RESCISION_POWERS
    }
#else	// WITH_POW_POST_CALLBACK
    EXIT_DIRECTIVE("WITH_POW_POST_CALLBACK");
#endif	// WITH_POW_POST_CALLBACK
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP("CUFFT callbacks");
  }

  return ret;
}

/** Set CUFFT store FFT callback  .
 *
 */
acc_err set_CuFFT_store_CBs(cuCgPlan* plan, cuFfdotStack* cStack)
{
  acc_err ret = ACC_ERR_NONE;

  if ( plan->flags & FLAG_CUFFT_CB_OUT )
  {
    infoMSG(5,5,"Set CB powers output\n");

#if 	CUDART_VERSION >= 6050

#ifdef	WITH_POW_POST_CALLBACK	// Post Callbacks (power)  .

    ulong bits = 0;

    SAFE_CALL(add_to_bits(&bits, log2(cStack->width), CB_STRT_WIDTH, CB_MASK_WIDTH),			"ERROR: Palne with too large to be bit encoded for FFT callback.");

    SAFE_CALL(add_to_bits(&bits, plan->noSegments, CB_STRT_NUMP, CB_MASK_NUMP),				"ERROR: Number of segements too large to be bit encoded for FFT callback.");

    SAFE_CALL(add_to_bits(&bits, cStack->harmInf->plnStart, CB_STRT_START, CB_MASK_START),		"ERROR: Start offset too large to be bit encoded for FFT callback.");

    SAFE_CALL(add_to_bits(&bits, cStack->harmInf->requirdWidth, CB_STRT_ALENG+32, ((ulong)CB_MASK_ALENG)<<32), 	"ERROR: segment size too large to be bit encoded for FFT callback.")

    if ( plan->flags & FLAG_CUFFT_CB_INMEM )
    {
      uint width = plan->noSegments*cStack->width;
      if ( plan->cuSrch->inmemStride % width ) SAFE_CALL(ACC_ERR_SIZE,					"ERROR: In-memory stride not divisible by correct width.");

      uint no_seg = plan->cuSrch->inmemStride/width;
      SAFE_CALL(add_to_bits(&bits, no_seg, CB_STRT_NO_SEG+32, ((ulong)CB_MASK_NO_SEG)<<32),		"ERROR: Search too long to encode callback details. try changing 'IN_MEM_POWERS' or try with a wider plane width or more segments.");
    }

    infoMSG(7,7,"Set CB pointer mask %p %p \n", bits, bits>>32);

    if ( plan->flags & FLAG_DOUBLE )
    {
#if	CUDART_VERSION >= 9000
      // This is a hack!
      CUFFT_SAFE_CALL(cufftXtClearCallback(cStack->plnPlan, CUFFT_CB_ST_COMPLEX_DOUBLE), "Error clearing CUFFT store callback.");
#endif // CUDART_VERSION
      CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&cStack->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX_DOUBLE, (void**)&bits ),"Error assigning CUFFT store callback.");
    }
    else
    {
#if	CUDART_VERSION >= 9000
      // This is a hack!
      CUFFT_SAFE_CALL(cufftXtClearCallback(cStack->plnPlan, CUFFT_CB_ST_COMPLEX), "Error clearing CUFFT store callback.");
#endif	// CUDART_VERSION
      CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&cStack->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&bits ),"Error assigning CUFFT store callback.");
    }
#else 	// WITH_POW_POST_CALLBACK
    EXIT_DIRECTIVE("WITH_POW_POST_CALLBACK");
#endif

#else	// CUDART_VERSION
    fprintf(stderr,"ERROR: CUFFT callbacks can only be used with CUDA 6.5 or later!\n");
    exit(EXIT_FAILURE);
#endif	// CUDART_VERSION
  }

  return ret;
}

void time_iFFT_Stack(cuCgPlan* plan)
{
  PROF // Profiling - Time previous components  .
  {
    if ( (plan->flags & FLAG_PROF) )
    {
      if ( (*plan->rAraays)[plan->rActive+1][0][0].numrs )
      {
	infoMSG(5,5,"Time previous components");

	for (int ss = 0; ss < plan->noStacks; ss++)
	{
	  cuFfdotStack* cStack = &plan->stacks[ss];
	  timeEvents( cStack->ifftInit, cStack->ifftComp, &plan->compTime[NO_STKS*COMP_GEN_IFFT + cStack->stackIdx ],  "Stack iFFT");
	}
      }
    }
  }
}

/** A simple function to call the plane CUFFT plan  .
 *
 * This is a separate function so one can be called by a omp critical and another not
 */
static void callPlaneCUFTT(cuCgPlan* plan, cuFfdotStack* cStack)
{
  PROF // Profiling  .
  {
    if ( plan->flags & FLAG_PROF )
    {
      infoMSG(5,5,"Event %s in %s.\n", "ifftInit", "fftPStream");
      CUDA_SAFE_CALL(cudaEventRecord(cStack->ifftInit, cStack->fftPStream),"Recording event: ifftInit");
    }
  }

  FOLD // Call the FFT  .
  {
    infoMSG(5,5,"Call CUFFT kernel");

    void* src;
    void* dst;

    src = get_pointer(plan, cStack, COMPLEX, 0, 0, 0, 0); //getCBwriteLocation(plan, cStack, segment);

    if ( (plan->flags & FLAG_SS_INMEM) && (plan->flags & FLAG_CUFFT_CB_INMEM) )
    {
      dst = get_pointer(plan, cStack, IM_PLN,  0, 0, 0, 0);
    }
    else
    {
      dst = get_pointer(plan, cStack, POWERS,  0, 0, 0, 0);
    }

    if ( (plan->flags & FLAG_SS_INMEM) && (plan->flags & FLAG_CUFFT_CB_INMEM) )
    {
      CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with fftPStream.");

      infoMSG(7,7,"CUFFT src    %p", src);		// DBG REM
      infoMSG(7,7,"CUFFT dst    %p   %i ", dst);	// DBG REM

      if ( plan->flags & FLAG_DOUBLE )
	CUFFT_SAFE_CALL(cufftExecZ2Z(cStack->plnPlan, (cufftDoubleComplex *) src, (cufftDoubleComplex *) dst, CUFFT_INVERSE),"Error executing CUFFT plan.");
      else
	CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) src, (cufftComplex *) dst, CUFFT_INVERSE),"Error executing CUFFT plan.");
    }
    else
    {
    CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with fftPStream.");

    infoMSG(7,7,"CUFFT src    %p", src);
    infoMSG(7,7,"CUFFT dst    %p", dst);

    if ( plan->flags & FLAG_DOUBLE )
      CUFFT_SAFE_CALL(cufftExecZ2Z(cStack->plnPlan, (cufftDoubleComplex *) src, (cufftDoubleComplex *) dst, CUFFT_INVERSE),"Error executing CUFFT plan.");
    else
      CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) src, (cufftComplex *) dst, CUFFT_INVERSE),"Error executing CUFFT plan.");
    }
  }

  FOLD // Synchronisation  .
  {
    infoMSG(5,5,"Event %s in %s.\n", "ifftComp", "fftPStream");
    CUDA_SAFE_CALL(cudaEventRecord(cStack->ifftComp, cStack->fftPStream),"Recording event: ifftInit");

    // If using power calculate call back with the inmem plane
    if ( plan->flags & FLAG_CUFFT_CB_INMEM )
    {
#if CUDART_VERSION >= 6050
      infoMSG(5,5,"Event %s in %s.\n", "ifftMemComp", "fftPStream");
      CUDA_SAFE_CALL(cudaEventRecord(cStack->ifftMemComp, cStack->fftPStream),"Recording event: ifftInit");
#endif
    }
  }
}

/**  iFFT a specific stack  .
 *
 * @param plan
 * @param cStack
 * @param pStack
 */
void iFFT_Stack(cuCgPlan* plan, cuFfdotStack* cStack, cuFfdotStack* pStack)
{
  infoMSG(3,3,"iFFT Stack %i\n", cStack->stackIdx);

  FOLD // Synchronisation  .
  {
    infoMSG(5,5,"Synchronise stream %s on %s.\n", "fftPStream", "multComp (stack and CG plan)");
    infoMSG(5,5,"Synchronise stream %s on %s.\n", "fftPStream", "ifftComp");

    // Wait for multiplication to finish
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack->multComp,      0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, plan->multComp,        0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

    // Wait for previous iFFT to finish
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack->ifftComp,      0), "Waiting for GPU to be ready to copy data to device.");
    if ( plan->flags & FLAG_SS_INMEM  )
    {
      infoMSG(5,5,"Synchronise stream %s on %s.\n", "fftPStream", "ifftMemComp");
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack->ifftMemComp, 0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
    }

    // Wait for previous search to finish
    infoMSG(5,5,"Synchronise stream %s on %s.\n", "fftPStream", "searchComp");
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, plan->searchComp,      0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

    if ( (plan->retType & CU_STR_PLN) && (plan->flags & FLAG_CUFFT_CB_OUT) ) // This has been deprecated! TODO check if I can remove this
    {
      infoMSG(5,5,"Synchronise stream %s on %s.\n", "fftPStream", "candCpyComp");
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, plan->candCpyComp,  0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
    }

    if ( plan->flags & FLAG_SYNCH )
    {
      // Wait for all the multiplications to complete
      infoMSG(5,5,"Synchronise stream %s on %s.\n", "fftPStream", "multComp (neighbours)");
      for (int synchIdx = 0; synchIdx < plan->noStacks; synchIdx++)
      {
	cuFfdotStack* cStack2 = &plan->stacks[synchIdx];
	CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack2->multComp, 0), "Waiting for event multComp.");
      }

      // Wait for the previous fft to complete
      if ( pStack != NULL )
      {
	infoMSG(5,5,"Synchronise stream %s on %s.\n", "fftPStream", "ifftComp (previous stack synch)");
	CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, pStack->ifftComp, 0), "Waiting for event ifftComp.");
      }
    }

  }

  FOLD // Call the inverse CUFFT  .
  {
    infoMSG(4,4,"Call the iFFT\n");

    if ( cStack->flags & CU_FFT_SEP_PLN )
    {
      callPlaneCUFTT(plan, cStack);
    }
    else
    {
#pragma omp critical
      callPlaneCUFTT(plan, cStack);
    }
  }

  FOLD // Plot  .
  {
#ifdef CBL
    if ( plan->flags & FLAG_DPG_PLT_POWERS )
    {
      FOLD // Synchronisation  .
      {
	infoMSG(4,4,"Blocking synchronisation on %s", "ifftMemComp" );

	CUDA_SAFE_CALL(cudaEventSynchronize(cStack->ifftMemComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
      }

      FOLD // Get data  .
      {
	float* outVals = (float*)malloc(plan->cmlxDataSize);
	void*  tmpRow  = malloc(plan->inptDataSize);
	ulong sz  = 0;

	for ( int plainNo = 0; plainNo < cStack->noInStack; plainNo++ )
	{
	  cuHarmInfo* cHInfo	= &cStack->harmInf[plainNo];		// The current harmonic we are working on
	  void*       tmpRow	= malloc(plan->inptDataSize);
	  cuFFdot*    plane	= &cStack->planes[plainNo];		// The current plane

	  int harm = cStack->startIdx+plainNo;

	  for ( int sIdx = 0; sIdx < plan->noSegments; sIdx ++)		// Loop over segments
	  {
	    rVals* rVal = &(((*plan->rAraays)[plan->rActive])[sIdx][harm]);

	    if ( rVal->numdata )
	    {
	      char tName[1024];
	      sprintf(tName, "/home/chris/accel/Powers_setp_%05i_h_%02i.csv", rVal->segment, harm );
	      FILE *f2 = fopen(tName, "w");

	      fprintf(f2,"Harm plane\n");
	      fprintf(f2,"centR: %.23f\n", (rVal->drhi-rVal->drlo)/2.0);
	      fprintf(f2,"centZ: %.23f\n", abs(cHInfo->zEnd-cHInfo->zStart)/2.0);
	      fprintf(f2,"rSize: %.23f\n", (rVal->drhi-rVal->drlo));
	      fprintf(f2,"zSize: %.23f\n", abs(cHInfo->zEnd-cHInfo->zStart));
	      fprintf(f2,"noZ: %.i\n",     (int)cHInfo->noZ);
	      fprintf(f2,"noR: %.i\n",     (int)rVal->numrs);
	      fprintf(f2,"Harms: %i\n",    1);

	      fprintf(f2,"Type: power\n");
	      fprintf(f2,"Layout: Harmonics\n");

	      fprintf(f2,"Harm %i", 1);

	      for ( int i = 0; i < rVal->numrs; i++)
	      {
		double r = rVal->drlo + i / (double)plan->conf->noResPerBin;
		fprintf(f2,"\t%.17e", r );
	      }
	      fprintf(f2,"\n");

	      // Copy pain from GPU
	      for( int y = 0; y < cHInfo->noZ; y++ )
	      {
		void *powers;
		int offset;
		int elsz;

		FOLD // Get the row as floats
		{
		  if      ( plan->flags & FLAG_ITLV_ROW )
		  {
		    offset = (y*plan->noSegments + sIdx)*cStack->strideCmplx   + cHInfo->plnStart ;
		  }
#ifdef WITH_ITLV_PLN
		  else
		  {
		    offset  = (y + sIdx*cHInfo->noZ)*cStack->strideCmplx   + cHInfo->plnStart ;
		  }
#else
		  else
		  {
		    fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
		    exit(EXIT_FAILURE);
		  }
#endif

		  if      ( plan->flags & FLAG_POW_HALF )
		  {
#ifdef	WITH_HALF_RESCISION_POWERS
#if 	CUDART_VERSION >= 7050   // Half precision getter and setter  .
		    powers =  &((half*)      plane->d_planePowr)[offset];
		    elsz   = sizeof(half);
		    CUDA_SAFE_CALL(cudaMemcpy(tmpRow, powers, (rVal->numrs)*elsz,   cudaMemcpyDeviceToHost), "Failed to copy input data from device.");

		    for ( int i = 0; i < rVal->numrs; i++)
		    {
		      outVals[i] = half2float(((ushort*)tmpRow)[i]);
		    }
#else	// CUDART_VERSION
		    fprintf(stderr, "ERROR: Half precision can only be used with CUDA 7.5 or later! Reverting to single precision!\n");
		    exit(EXIT_FAILURE);
#endif	// CUDART_VERSION
#else	// WITH_HALF_RESCISION_POWERS
		    EXIT_DIRECTIVE("WITH_HALF_RESCISION_POWERS");
#endif	// WITH_HALF_RESCISION_POWERS
		  }
		  else if ( plan->flags & FLAG_CUFFT_CB_POW )
		  {
#ifdef	WITH_SINGLE_RESCISION_POWERS
		    powers =  &((float*)     plane->d_planePowr)[offset];
		    elsz   = sizeof(float);
		    CUDA_SAFE_CALL(cudaMemcpy(outVals, powers, (rVal->numrs)*elsz,   cudaMemcpyDeviceToHost), "Failed to copy input data from device.");

#else	// WITH_SINGLE_RESCISION_POWERS
		    EXIT_DIRECTIVE("WITH_SINGLE_RESCISION_POWERS");
#endif	// WITH_SINGLE_RESCISION_POWERS
		  }
		  else
		  {
#ifdef	WITH_COMPLEX_POWERS
		    fcomplexcu *cmplxData;

		    powers =  &((fcomplexcu*) plane->d_planePowr)[offset];
		    elsz   = sizeof(cmplxData);
		    CUDA_SAFE_CALL(cudaMemcpy(tmpRow, powers, (rVal->numrs)*elsz,   cudaMemcpyDeviceToHost), "Failed to copy input data from device.");

		    for ( int i = 0; i < rVal->numrs; i++)
		    {
		      outVals[i] = POWERC(((fcomplexcu*)tmpRow)[i]);
		    }

#else	// WITH_COMPLEX_POWERS
		    EXIT_DIRECTIVE("WITH_COMPLEX_POWERS");
#endif	// WITH_COMPLEX_POWERS
		  }
		}

		FOLD // Write line to csv  .
		{
		  double z = cHInfo->zStart + (cHInfo->zEnd-cHInfo->zStart)/(double)(cHInfo->noZ-1)*y;
		  if (cHInfo->noZ == 1 )
		    z = 0;
		  fprintf(f2,"%.17e", z);

		  for ( int i = 0; i < rVal->numrs; i++)
		  {
		    fprintf(f2,"\t%.17e", outVals[i] );
		  }
		  fprintf(f2,"\n");
		}
	      }

	      fclose(f2);

	      FOLD // Make image  .
	      {
		infoMSG(4,4,"Image %s\n", tName );

		PROF // Profiling  .
		{
		  NV_RANGE_PUSH("Image");
		}

		char cmd[1024];
		sprintf(cmd,"python $PRESTO/python/plt_ffd.py %s  -r 2 -s 50 -a 0.9  > /dev/null 2>&1", tName);
		infoMSG(6,6,"%s", cmd);
		int ret = system(cmd);
		if ( ret )
		{
		  fprintf(stderr,"ERROR: Problem running potting python script.");
		}

		PROF // Profiling  .
		{
		  NV_RANGE_POP("Image");
		}
	      }

	      sz += cStack->strideCmplx;
	    }
	  }
	}
      }
    }
#endif
  }
}

/** iFFT all stack of a CG plan  .
 *
 *
 */
void cg_iFFT(cuCgPlan* plan)
{
  PROF // Profiling - Time previous components  .
  {
    time_iFFT_Stack(plan);
  }

  if ( (*plan->rAraays)[plan->rActive][0][0].numrs )
  {
    infoMSG(2,2,"iFFT CG plan - Iteration %3i.", (*plan->rAraays)[plan->rActive][0][0].iteration);

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("IFFT");
    }

    cuFfdotStack* pStack = NULL;  // Previous stack

    for (int ss = 0; ss < plan->noStacks; ss++)
    {
      int stkIdx;
      cuFfdotStack* cStack;

      FOLD // Chose stack to use  .
      {
	if ( plan->flags & FLAG_STK_UP )
	  stkIdx = plan->noStacks - 1 - ss;
	else
	  stkIdx = ss;

	cStack = &plan->stacks[stkIdx];
      }

      FOLD // IFFT  .
      {
	iFFT_Stack(plan, cStack, pStack);
      }

      pStack = cStack;
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP("IFFT");
    }
  }
}

