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
#include "cuda_accel_FFT.h"

//========================================== Functions  ====================================================\\

/** Multiplication kernel - One plane at a time  .
 * Each thread reads one input value and loops down over the kernels
 *
 * This should only be called by cg_multiply, to check active iteration and time events
 */
static void multiply_Plane(cuCgPlan* plan)
{
  dim3 dimGrid, dimBlock;

  cuFfdotStack* pStack = NULL;  // Previous stack

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  for (int stack = 0; stack < plan->noStacks; stack++)              // Loop through Stacks
  {
    cuFfdotStack* cStack = &plan->stacks[stack];

    FOLD // Synchronisation  .
    {
      // This iteration
      infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "inpFFTinitComp");
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->inpFFTinitComp,0),       "Waiting for GPU to be ready to copy data to device.");  // Need input data

      // CFF output callback has its own data so can start once FFT is complete
      infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "ifftComp");
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->ifftComp, 0),      "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

      if ( (plan->retType & CU_STR_PLN) && !(plan->flags & FLAG_CUFFT_CB_OUT) )
      {
	infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "candCpyComp");
	CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, plan->candCpyComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // Multiplication will change the plane
      }

      if ( plan->flags & FLAG_SYNCH )
      {
	// Wait for all the input FFT's to complete
	infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "inpFFTinitComp (stacks)");
	for (int ss = 0; ss< plan->noStacks; ss++)
	{
	  cuFfdotStack* cStack2 = &plan->stacks[ss];
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
      if ( plan->flags & FLAG_PROF )
      {
	infoMSG(5,5,"Event %s in %s.\n", "multInit", "multStream");
	CUDA_SAFE_CALL(cudaEventRecord(cStack->multInit, cStack->multStream),"Recording event: multInit");
      }
    }

    FOLD // call kernel(s)  .
    {
      mult11(cStack->multStream, plan, cStack);
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
 * @param plan
 * @param cStack
 * @param pStack
 */
static void multiply_Stack(cuCgPlan* plan, cuFfdotStack* cStack, cuFfdotStack* pStack = NULL)
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

    if ( (plan->retType & CU_STR_PLN) && !(plan->flags & FLAG_CUFFT_CB_OUT) )
    {
      infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "candCpyComp");

      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, plan->candCpyComp,  0), "Waiting for GPU to be ready to copy data to device.");  // Multiplication will change the plane
    }

    if ( plan->flags & FLAG_SYNCH )
    {
      // Wait for all the input FFT's to complete
      for (int synchIdx = 0; synchIdx < plan->noStacks; synchIdx++)
      {
	infoMSG(5,5,"Synchronise stream %s on %s - stack %i.\n", "multStream", "inpFFTinitComp", synchIdx);

	cuFfdotStack* cStack2 = &plan->stacks[synchIdx];
	CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack2->inpFFTinitComp, 0), "Stream wait on event inpFFTinitComp.");
      }

      // Wait for iFFT to finish - In-mem search - I found that GPU compute interferes with D2D copy so wait for it to finish
      infoMSG(5,5,"Synchronise stream %s on %s.\n", "inptStream", "ifftMemComp");
      cudaStreamWaitEvent(cStack->inptStream, plan->stacks->ifftMemComp, 0);

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
    if ( plan->flags & FLAG_PROF )
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
      mult00(cStack->multStream, plan, cStack);
    }
    else if ( cStack->flags & FLAG_MUL_21 )
    {
      infoMSG(4,4,"Kernel call mult21\n");
      mult21(cStack->multStream, plan, cStack);
    }
    else if ( cStack->flags & FLAG_MUL_22 )
    {
      infoMSG(4,4,"Kernel call mult22\n");
      mult22(cStack->multStream, plan, cStack);
    }
    else if ( cStack->flags & FLAG_MUL_23 )
    {
      infoMSG(4,4,"Kernel call mult23\n");
      mult23(cStack->multStream, plan, cStack);
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

void time_multiply_CgPlan(cuCgPlan* plan)
{
  PROF // Profiling - Time previous components  .
  {
    if ( (plan->flags & FLAG_PROF) )
    {
      if ( (*plan->rAraays)[plan->rActive+1][0][0].numrs )
      {
	infoMSG(5,5,"Time previous components");

	if ( plan->flags & FLAG_MUL_BATCH ) // Time batch multiplication
	{
	  timeEvents(plan->multInit, plan->multComp, &plan->compTime[NO_STKS*COMP_GEN_MULT], "Batch multiplication");
	}

	if ( ( plan->flags & FLAG_MUL_PLN ) || ( plan->flags & FLAG_MUL_STK ) ) // Time stack multiplication
	{
	  for (int stack = 0; stack < plan->noStacks; stack++)
	  {
	    cuFfdotStack* cStack = &plan->stacks[stack];

	    timeEvents( cStack->multInit, cStack->multComp, &plan->compTime[NO_STKS*COMP_GEN_MULT + stack ],  "Stack multiplication");
	  }
	}
      }
    }
  }
}

/** Call all the convolution kernels for CG plan  .
 *
 * @param plan
 */
void cg_multiply(cuCgPlan* plan)
{
  PROF // Profiling - Time previous components  .
  {
    time_multiply_CgPlan(plan);
  }

  if ( (*plan->rAraays)[plan->rActive][0][0].numrs )
  {
    infoMSG(2,2,"Multiply CG pan - Iteration %3i.", (*plan->rAraays)[plan->rActive][0][0].iteration);

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Multiply");
    }

    if      ( plan->flags & FLAG_MUL_BATCH )  // Do the multiplications on an entire family   .
    {
      FOLD // Synchronisation  .
      {
	// Synchronise input data preparation for all stacks
	infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "inpFFTinitComp (stacks)");
	infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "ifftComp (stacks)");
	for (int synchIdx = 0; synchIdx < plan->noStacks; synchIdx++)
	{
	  cuFfdotStack* cStack = &plan->stacks[synchIdx];

	  CUDA_SAFE_CALL(cudaStreamWaitEvent(plan->multStream, cStack->inpFFTinitComp, 0),     "Waiting for input data to be FFT'ed.");    // Need input data

	  // iFFT has its own data so can start once iFFT is complete
	  CUDA_SAFE_CALL(cudaStreamWaitEvent(plan->multStream, cStack->ifftComp, 0),     "Waiting for iFFT.");  // This will overwrite the plane so search must be compete
	}

	if ( !(plan->flags & FLAG_CUFFT_CB_OUT) )
	{
	  // Have to wait for search to finish reading data
	  infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "searchComp");
	  CUDA_SAFE_CALL(cudaStreamWaitEvent(plan->multStream, plan->searchComp, 0),    "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
	}

	if ( (plan->retType & CU_STR_PLN) && !(plan->flags & FLAG_CUFFT_CB_OUT) )
	{
	  infoMSG(5,5,"Synchronise stream %s on %s.\n", "multStream", "candCpyComp");
	  CUDA_SAFE_CALL(cudaStreamWaitEvent(plan->multStream, plan->candCpyComp, 0),   "Waiting for GPU to be ready to copy data to device.");   // Multiplication will change the plane
	}
      }

      FOLD // Call kernel  .
      {
	PROF // Profiling  .
	{
	  if ( plan->flags & FLAG_PROF )
	  {
	    infoMSG(5,5,"Event %s in %s.\n", "multInit", "multStream");
	    CUDA_SAFE_CALL(cudaEventRecord(plan->multInit, plan->multStream),"Recording event: multInit");
	  }
	}

	mult31(plan->multStream, plan);

	// Run message
	CUDA_SAFE_CALL(cudaGetLastError(), "At kernel launch");
      }

      FOLD // Synchronisation  .
      {
	infoMSG(5,5,"Event %s in %s.\n", "multComp", "multStream");
	CUDA_SAFE_CALL(cudaEventRecord(plan->multComp, plan->multStream),"Recording event: multComp");
      }
    }
    else if ( plan->flags & FLAG_MUL_STK   )  // Do the multiplications one stack  at a time  .
    {
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

	FOLD // Multiply  .
	{
	  if ( plan->flags & FLAG_MUL_CB )
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
	    multiply_Stack(plan, cStack, pStack);
	  }
	}

	pStack = cStack;
      }
    }
    else if ( plan->flags & FLAG_MUL_PLN   )  // Do the multiplications one plane  at a time  .
    {
      multiply_Plane(plan);
    }
    else
    {
      fprintf(stderr, "ERROR: multiplyCgPlan not templated for this type of multiplication.\n");
      exit(EXIT_FAILURE);
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP("Multiply");
    }
  }
}

/** Multiply and iFFT the complex f-∂f plane  .
 * This assumes the input data is ready and on the device
 * This creates a complex f-∂f plane
 */
void cg_convolve(cuCgPlan* plan)
{
  // Multiply
  cg_multiply(plan);

  // IFFT
  cg_iFFT(plan);
}


