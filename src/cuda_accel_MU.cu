/** @file cuda_accel_MU.cu
 *  @brief Functions to manage plane multiplication and FFT tasks
 *
 *  This contains the various functions that control plane multiplication and FFT tasks
 *  These include:
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
 *  [0.0.01] [2017-01-29 08:20]
 *    Added static function to call CUFFT plan for plane, this allows identical calls from non critical and non critical blocks
 *    Made some other functions static
 *
 */

#include "cuda_accel_MU.h"

//========================================== Functions  ====================================================\\

/** Multiplication kernel - One plane at a time  .
 * Each thread reads one input value and loops down over the kernels
 *
 * This should only be called by multiplyBatch, to check active iteration and time events
 */
static void multiplyPlane(cuFFdotBatch* batch)
{
  dim3 dimGrid, dimBlock;

  cuFfdotStack* pStack = NULL;  // Previous stack

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  for (int stack = 0; stack < batch->noStacks; stack++)              // Loop through Stacks
  {
    cuFfdotStack* cStack = &batch->stacks[stack];

    FOLD // Synchronisation  .
    {
      // This iteration
      infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "inpFFTinitComp");
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->inpFFTinitComp,0),       "Waiting for GPU to be ready to copy data to device.");  // Need input data

      // CFF output callback has its own data so can start once FFT is complete
      infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "ifftComp");
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->ifftComp, 0),      "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

      if ( (batch->retType & CU_STR_PLN) && !(batch->flags & FLAG_CUFFT_CB_OUT) )
      {
	infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "candCpyComp");
	CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, batch->candCpyComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // Multiplication will change the plane
      }

      if ( batch->flags & FLAG_SYNCH )
      {
	// Wait for all the input FFT's to complete
	infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "inpFFTinitComp (stacks)");
	for (int ss = 0; ss< batch->noStacks; ss++)
	{
	  cuFfdotStack* cStack2 = &batch->stacks[ss];
	  CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack2->inpFFTinitComp, 0),  "Waiting for event inpFFTinitComp");
	}

	// Wait for the previous multiplication to complete
	if ( pStack != NULL )
	{
	  infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "multComp (previous)");
	  CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, pStack->multComp, 0),  "Waiting for event inpFFTinitComp");
	}
      }
    }

    PROF // Profiling  .
    {
      if ( batch->flags & FLAG_PROF )
      {
	infoMSG(5,5,"Event %s in %s.\n", "multInit", "multStream");
	CUDA_SAFE_CALL(cudaEventRecord(cStack->multInit, cStack->multStream),"Recording event: multInit");
      }
    }

    FOLD // call kernel(s)  .
    {
      mult11(cStack->multStream, batch, cStack);
    }

    FOLD // Synchronisation  .
    {
      infoMSG(5,5,"Event %s in %s.\n", "multComp", "multStream");
      CUDA_SAFE_CALL(cudaEventRecord(cStack->multComp, cStack->multStream),"Recording event: multComp");
    }

    pStack = cStack;
  }

}

/** Multiply a specific stack using one of the multiplication 2 or 0 kernels  .
 *
 * This should only be called by multiplyBatch, to check active iteration and time events
 *
 * @param batch
 * @param cStack
 * @param pStack
 */
static void multiplyStack(cuFFdotBatch* batch, cuFfdotStack* cStack, cuFfdotStack* pStack = NULL)
{
  infoMSG(3,3,"Multiply stack %i \n", cStack->stackIdx);

  FOLD // Synchronisation  .
  {
    infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "inpFFTinitComp");
    infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "ifftComp");

    // This iteration
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->inpFFTinitComp,    0),   "Waiting for GPU to be ready to copy data to device.");  // Need input data

    // iFFT has its own data so can start once iFFT is complete
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->ifftComp,    0),   "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

    if ( (batch->retType & CU_STR_PLN) && !(batch->flags & FLAG_CUFFT_CB_OUT) )
    {
      infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "candCpyComp");

      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, batch->candCpyComp,  0), "Waiting for GPU to be ready to copy data to device.");  // Multiplication will change the plane
    }

    if ( batch->flags & FLAG_SYNCH )
    {
      // Wait for all the input FFT's to complete
      for (int synchIdx = 0; synchIdx < batch->noStacks; synchIdx++)
      {
	infoMSG(5,5,"Synchronise stream %s on %s - stack %i.\n", "multStream", "inpFFTinitComp", synchIdx);

	cuFfdotStack* cStack2 = &batch->stacks[synchIdx];
	CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack2->inpFFTinitComp, 0), "Stream wait on event inpFFTinitComp.");
      }

      // Wait for iFFT to finish - In-mem search - I found that GPU compute interferes with D2D copy so wait for it to finish
      infoMSG(5,5,"Synchronise stream %s on %s.\n", "inptStream", "ifftMemComp");
      cudaStreamWaitEvent(cStack->inptStream, batch->stacks->ifftMemComp, 0);

      // Wait for the previous multiplication to complete
      if ( pStack != NULL )
      {
	infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "multComp (previous)");

	CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, pStack->multComp, 0), "Stream wait on event inpFFTinitComp.");
      }
    }
  }

  PROF // Profiling  .
  {
    if ( batch->flags & FLAG_PROF )
    {
      infoMSG(5,5,"Event %s in %s.\n", "multInit", "multStream");
      CUDA_SAFE_CALL(cudaEventRecord(cStack->multInit, cStack->multStream),"Recording event: multInit");
    }
  }

  FOLD // Call kernel(s) .
  {
    if      ( cStack->flags & FLAG_MUL_00 )
    {
      infoMSG(4,4,"Kernel call mult00\n");
      mult00(cStack->multStream, batch, cStack);
    }
    else if ( cStack->flags & FLAG_MUL_21 )
    {
      infoMSG(4,4,"Kernel call mult21\n");
      mult21(cStack->multStream, batch, cStack);
    }
    else if ( cStack->flags & FLAG_MUL_22 )
    {
      infoMSG(4,4,"Kernel call mult22\n");
      mult22(cStack->multStream, batch, cStack);
    }
    else if ( cStack->flags & FLAG_MUL_23 )
    {
      infoMSG(4,4,"Kernel call mult23\n");
      mult23(cStack->multStream, batch, cStack);
    }
    else
    {
      fprintf(stderr,"ERROR: No valid stack multiplication specified. Line %i in %s.\n", __LINE__, __FILE__);
      exit(EXIT_FAILURE);
    }

    // Run message
    CUDA_SAFE_CALL(cudaGetLastError(), "At multiplication kernel launch.");
  }

  FOLD // Synchronisation  .
  {
    infoMSG(5,5,"Event %s in %s.\n", "multComp", "multStream");
    CUDA_SAFE_CALL(cudaEventRecord(cStack->multComp, cStack->multStream),"Recording event: multInit");
  }
}

/** Call all the multiplication kernels for batch  .
 *
 * @param batch
 */
void multiplyBatch(cuFFdotBatch* batch)
{
  PROF // Profiling - Time previous components  .
  {
    if ( (batch->flags & FLAG_PROF) )
    {
      if ( (*batch->rAraays)[batch->rActive+1][0][0].numrs )
      {
	infoMSG(5,5,"Time previous components");

	// Time batch multiply
	timeEvents( batch->multInit, batch->multComp, &batch->compTime[NO_STKS*COMP_GEN_MULT], "Batch multiplication");

	// Stack multiply
	for (int stack = 0; stack < batch->noStacks; stack++)
	{
	  cuFfdotStack* cStack = &batch->stacks[stack];

	  timeEvents( cStack->multInit, cStack->multComp, &batch->compTime[NO_STKS*COMP_GEN_MULT + stack ],  "Stack multiplication");
	}
      }
    }
  }

  if ( (*batch->rAraays)[batch->rActive][0][0].numrs )
  {
    infoMSG(2,2,"Multiply Batch - Iteration %3i.", (*batch->rAraays)[batch->rActive][0][0].iteration);

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Multiply");
    }

    if      ( batch->flags & FLAG_MUL_BATCH )  // Do the multiplications on an entire family   .
    {
      FOLD // Synchronisation  .
      {
	// Synchronise input data preparation for all stacks
	infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "inpFFTinitComp (stacks)");
	infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "ifftComp (stacks)");
	for (int synchIdx = 0; synchIdx < batch->noStacks; synchIdx++)
	{
	  cuFfdotStack* cStack = &batch->stacks[synchIdx];

	  CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, cStack->inpFFTinitComp, 0),     "Waiting for input data to be FFT'ed.");    // Need input data

	  // iFFT has its own data so can start once iFFT is complete
	  CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, cStack->ifftComp, 0),     "Waiting for iFFT.");  // This will overwrite the plane so search must be compete
	}

	if ( !(batch->flags & FLAG_CUFFT_CB_OUT) )
	{
	  // Have to wait for search to finish reading data
	  infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "searchComp");
	  CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, batch->searchComp, 0),    "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
	}

	if ( (batch->retType & CU_STR_PLN) && !(batch->flags & FLAG_CUFFT_CB_OUT) )
	{
	  infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "candCpyComp");
	  CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, batch->candCpyComp, 0),   "Waiting for GPU to be ready to copy data to device.");   // Multiplication will change the plane
	}
      }

      FOLD // Call kernel  .
      {
	PROF // Profiling  .
	{
	  if ( batch->flags & FLAG_PROF )
	  {
	    infoMSG(5,5,"Event %s in %s.\n", "multInit", "multStream");
	    CUDA_SAFE_CALL(cudaEventRecord(batch->multInit, batch->multStream),"Recording event: multInit");
	  }
	}

	mult31(batch->multStream, batch);

	// Run message
	CUDA_SAFE_CALL(cudaGetLastError(), "At kernel launch");
      }

      FOLD // Synchronisation  .
      {
	infoMSG(5,5,"Event %s in %s.\n", "multComp", "multStream");
	CUDA_SAFE_CALL(cudaEventRecord(batch->multComp, batch->multStream),"Recording event: multComp");
      }
    }
    else if ( batch->flags & FLAG_MUL_STK   )  // Do the multiplications one stack  at a time  .
    {
      cuFfdotStack* pStack = NULL;  // Previous stack

      for (int ss = 0; ss < batch->noStacks; ss++)
      {
	int stkIdx;
	cuFfdotStack* cStack;

	FOLD // Chose stack to use  .
	{
	  if ( batch->flags & FLAG_STK_UP )
	    stkIdx = batch->noStacks - 1 - ss;
	  else
	    stkIdx = ss;

	  cStack = &batch->stacks[stkIdx];
	}

	FOLD // Multiply  .
	{
	  if ( batch->flags & FLAG_MUL_CB )
	  {
#ifdef WITH_MUL_PRE_CALLBACK
	    // Just synchronise, the iFFT will do the multiplication once the multComp event has been recorded
	    infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "inpFFTinitComp");
	    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack->inpFFTinitComp,0),   "Waiting for GPU to be ready to copy data to device.");

	    infoMSG(5,5,"Event %s in %s.\n", "multComp", "fftPStream");
	    CUDA_SAFE_CALL(cudaEventRecord(cStack->multComp, cStack->fftPStream),         "Recording event: multComp");
#else
	    fprintf(stderr, "ERROR: Not compiled with multiplication through CUFFT callbacks enabled. \n");
	    exit(EXIT_FAILURE);
#endif
	  }
	  else
	  {
	    multiplyStack(batch, cStack, pStack);
	  }
	}

	FOLD // IFFT if integrated convolution  .
	{
	  if ( batch->flags & FLAG_CONV )
	  {
	    IFFTStack(batch, cStack, pStack);
	  }
	}

	pStack = cStack;
      }
    }
    else if ( batch->flags & FLAG_MUL_PLN   )  // Do the multiplications one plane  at a time  .
    {
      multiplyPlane(batch);
    }
    else
    {
      fprintf(stderr, "ERROR: multiplyBatch not templated for this type of multiplication.\n");
      exit(EXIT_FAILURE);
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP("Multiply");
    }
  }
}

/** A simple function to call the plane CUFFT plan  .
 *
 * This is a separate function so one can be called by a omp critical and another not
 */
static void callPlaneCUFTT(cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  PROF // Profiling  .
  {
    if ( batch->flags & FLAG_PROF )
    {
      infoMSG(5,5,"Event %s in %s.\n", "ifftInit", "fftPStream");
      CUDA_SAFE_CALL(cudaEventRecord(cStack->ifftInit, cStack->fftPStream),"Recording event: ifftInit");
    }
  }

  FOLD // Call the FFT  .
  {
    infoMSG(5,5,"Call CUFFT kernel");

    void* dst = getCBwriteLocation(batch, cStack);

    CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with multStream.");

    infoMSG(7,7,"CUFFT input  %p", cStack->d_planeMult);

    if ( batch->flags & FLAG_DOUBLE )
      CUFFT_SAFE_CALL(cufftExecZ2Z(cStack->plnPlan, (cufftDoubleComplex *) cStack->d_planeMult, (cufftDoubleComplex *) dst, CUFFT_INVERSE),"Error executing CUFFT plan.");
    else
      CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) cStack->d_planeMult, (cufftComplex *) dst, CUFFT_INVERSE),"Error executing CUFFT plan.");
  }

  FOLD // Synchronisation  .
  {
    infoMSG(5,5,"Event %s in %s.\n", "ifftComp", "fftPStream");
    CUDA_SAFE_CALL(cudaEventRecord(cStack->ifftComp, cStack->fftPStream),"Recording event: ifftInit");

    // If using power calculate call back with the inmem plane
    if ( batch->flags & FLAG_CUFFT_CB_INMEM )
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
 * @param batch
 * @param cStack
 * @param pStack
 */
void IFFTStack(cuFFdotBatch* batch, cuFfdotStack* cStack, cuFfdotStack* pStack)
{
  infoMSG(3,3,"iFFT Stack %i\n", cStack->stackIdx);

  PROF // Profiling - Time previous components  .
  {
    if ( (batch->flags & FLAG_PROF) )
    {
      if ( (*batch->rAraays)[batch->rActive+1][0][0].numrs )
      {
	infoMSG(5,5,"Time previous components");

	timeEvents( cStack->ifftInit, cStack->ifftComp, &batch->compTime[NO_STKS*COMP_GEN_IFFT + cStack->stackIdx ],  "Stack iFFT");
      }
    }
  }

  FOLD // Synchronisation  .
  {
    infoMSG(5,5,"Synchronise stream %s on %s.\n", "fftPStream", "multComp (stack and batch)");
    infoMSG(5,5,"Synchronise stream %s on %s.\n", "fftPStream", "ifftComp");

    // Wait for multiplication to finish
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack->multComp,      0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, batch->multComp,       0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

    // Wait for previous iFFT to finish
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack->ifftComp,      0), "Waiting for GPU to be ready to copy data to device.");
    if ( batch->flags & FLAG_SS_INMEM  )
    {
      infoMSG(5,5,"Synchronise stream %s on %s.\n", "fftPStream", "ifftMemComp");
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack->ifftMemComp, 0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
    }

    // Wait for previous search to finish
    infoMSG(5,5,"Synchronise stream %s on %s.\n", "fftPStream", "searchComp");
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, batch->searchComp,     0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

    if ( (batch->retType & CU_STR_PLN) && (batch->flags & FLAG_CUFFT_CB_OUT) ) // This has been deprecated!
    {
      infoMSG(5,5,"Synchronise stream %s on %s.\n", "fftPStream", "candCpyComp");
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, batch->candCpyComp,  0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
    }

    if ( batch->flags & FLAG_SYNCH )
    {
      // Wait for all the multiplications to complete
      infoMSG(5,5,"Synchronise stream %s on %s.\n", "fftPStream", "multComp (neighbours)");
      for (int synchIdx = 0; synchIdx < batch->noStacks; synchIdx++)
      {
	cuFfdotStack* cStack2 = &batch->stacks[synchIdx];
	CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack2->multComp, 0), "Waiting for event ifftComp.");
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
      callPlaneCUFTT(batch, cStack);
    }
    else
    {
#pragma omp critical
      callPlaneCUFTT(batch, cStack);
    }
  }
}

/** iFFT all stack of a batch  .
 *
 * If using the FLAG_CONV flag no iFFT is done as this should have been done by the multiplication
 *
 */
void IFFTBatch(cuFFdotBatch* batch)
{
  if ( (*batch->rAraays)[batch->rActive][0][0].numrs )
  {
    if ( !( (batch->flags & FLAG_CONV) && (batch->flags & FLAG_MUL_STK) ) )
    {
      infoMSG(2,2,"iFFT Batch - Iteration %3i.", (*batch->rAraays)[batch->rActive][0][0].iteration);

      PROF // Profiling  .
      {
	NV_RANGE_PUSH("IFFT");
      }

      cuFfdotStack* pStack = NULL;  // Previous stack

      for (int ss = 0; ss < batch->noStacks; ss++)
      {
	int stkIdx;
	cuFfdotStack* cStack;

	FOLD // Chose stack to use  .
	{
	  if ( batch->flags & FLAG_STK_UP )
	    stkIdx = batch->noStacks - 1 - ss;
	  else
	    stkIdx = ss;

	  cStack = &batch->stacks[stkIdx];
	}

	FOLD // IFFT  .
	{
	  IFFTStack(batch, cStack, pStack);
	}

	pStack = cStack;
      }

      PROF // Profiling  .
      {
	NV_RANGE_POP("IFFT");
      }
    }
  }
}

/** Multiply and iFFT the complex f-∂f plane  .
 * This assumes the input data is ready and on the device
 * This creates a complex f-∂f plane
 */
void convolveBatch(cuFFdotBatch* batch)
{
  // Multiply
  multiplyBatch(batch);

  // IFFT
  IFFTBatch(batch);
}


