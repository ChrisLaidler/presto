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
	infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "inpFFTinitComp");

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
      NV_RANGE_POP();
    }
  }
}

/** A simple function to call the plane CUFFT plan  .
 *
 * This is a seperate function so one can be called by a omp critical and another not
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

  FOLD // Set the load and store FFT callback if necessary  .
  {
    setCB(batch, cStack);
  }

  FOLD // Call the FFT  .
  {
    infoMSG(5,5,"CUFFT kernel");

    void* dst = getCBwriteLocation(batch, cStack);

    CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with multStream.");

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
#if CUDA_VERSION >= 6050
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
    infoMSG(4,4,"Call the inverse CUFFT\n");

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

  FOLD // Plot  .
  {
#ifdef CBL
    if ( batch->flags & FLAG_DPG_PLT_POWERS )
    {
      FOLD // Synchronisation  .
      {
	infoMSG(4,4,"blocking synchronisation on %s", "ifftMemComp" );

	CUDA_SAFE_CALL(cudaEventSynchronize(cStack->ifftMemComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
      }

      FOLD // Get data  .
      {
	float* outVals = (float*)malloc(batch->plnDataSize);
	void*  tmpRow  = malloc(batch->inpDataSize);
	ulong sz  = 0;

	for ( int plainNo = 0; plainNo < cStack->noInStack; plainNo++ )
	{
	  cuHarmInfo* cHInfo	= &cStack->harmInf[plainNo];		// The current harmonic we are working on
	  void*       tmpRow	= malloc(batch->inpDataSize);
	  cuFFdot*    plan	= &cStack->planes[plainNo];		// The current plane

	  int harm = cStack->startIdx+plainNo;

	  for ( int step = 0; step < batch->noSteps; step ++)		// Loop over steps
	  {
	    rVals* rVal = &(((*batch->rAraays)[batch->rActive])[step][harm]);

	    if ( rVal->numdata )
	    {
	      char tName[1024];
	      sprintf(tName,"/home/chris/accel/Powers_setp_%05i_h_%02i.csv", rVal->step, harm );
	      FILE *f2 = fopen(tName, "w");

	      fprintf(f2,"%i",harm);

	      for ( int i = 0; i < rVal->numrs; i++)
	      {
		double r = rVal->drlo + i / (double)batch->cuSrch->sSpec->noResPerBin;
		fprintf(f2,"\t%.6f",r);
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
		  if      ( batch->flags & FLAG_ITLV_ROW )
		  {
		    //offset = (y*trdBatch->noSteps + step)*cStack->strideCmplx   + cHInfo->halfWidth * 2 ;
		    offset = (y*batch->noSteps + step)*cStack->strideCmplx   + cHInfo->kerStart ;
		  }
#ifdef WITH_ITLV_PLN
		  else
		  {
		    //offset  = (y + step*cHInfo->height)*cStack->strideCmplx   + cHInfo->halfWidth * 2 ;
		    offset  = (y + step*cHInfo->noZ)*cStack->strideCmplx   + cHInfo->kerStart ;
		  }
#else
		  else
		  {
		    fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
		    exit(EXIT_FAILURE);
		  }
#endif

		  if      ( batch->flags & FLAG_POW_HALF )
		  {
#ifdef	WITH_HALF_RESCISION_POWERS
#if 	CUDA_VERSION >= 7050   // Half precision getter and setter  .
		    powers =  &((half*)      plan->d_planePowr)[offset];
		    elsz   = sizeof(half);
		    CUDA_SAFE_CALL(cudaMemcpy(tmpRow, powers, (rVal->numrs)*elsz,   cudaMemcpyDeviceToHost), "Failed to copy input data from device.");

		    for ( int i = 0; i < rVal->numrs; i++)
		    {
		      outVals[i] = half2float(((ushort*)tmpRow)[i]);
		    }
#else	// CUDA_VERSION
		    fprintf(stderr, "ERROR: Half precision can only be used with CUDA 7.5 or later! Reverting to single precision!\n");
		    exit(EXIT_FAILURE);
#endif	// CUDA_VERSION
#else	// WITH_HALF_RESCISION_POWERS
		    EXIT_DIRECTIVE("WITH_HALF_RESCISION_POWERS");
#endif	// WITH_HALF_RESCISION_POWERS
		  }
		  else if ( batch->flags & FLAG_CUFFT_CB_POW )
		  {
#ifdef	WITH_SINGLE_RESCISION_POWERS
		    powers =  &((float*)     plan->d_planePowr)[offset];
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

		    powers =  &((fcomplexcu*) plan->d_planePowr)[offset];
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
		  fprintf(f2,"%.15f",z);

		  for ( int i = 0; i < rVal->numrs; i++)
		  {
		    fprintf(f2,"\t%.20f", outVals[i] );
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
		sprintf(cmd,"python ~/bin/bin/plt_ffd.py %s 2.5 > /dev/null 2>&1", tName);
		system(cmd);

		PROF // Profiling  .
		{
		  NV_RANGE_POP();
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
	NV_RANGE_POP();
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


