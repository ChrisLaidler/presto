#include "cuda_accel_MU.h"

//========================================== Functions  ====================================================\\

/** Multiply all the planes of a batch using the mutl11 kernel  .
 *
 * @param batch
 */
void multiplyPlane(cuFFdotBatch* batch)
{
  dim3 dimGrid, dimBlock;

  cuFfdotStack* pStack = NULL;  // Previous stack

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  for (int stack = 0; stack < batch->noStacks; stack++)              // Loop through Stacks
  {
    cuFfdotStack* cStack = &batch->stacks[stack];
    void*       d_planeData;    // The complex f-∂f plane data
    fcomplexcu* d_iData;        // The complex input array

    FOLD // Synchronisation  .
    {
      // This iteration
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->inpFFTinitComp,0),       "Waiting for GPU to be ready to copy data to device.");  // Need input data

      // CFF output callback has its own data so can start once FFT is complete
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->ifftComp, 0),      "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

      if ( (batch->retType & CU_STR_PLN) && !(batch->flags & FLAG_CUFFT_CB_OUT) )
      {
	CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, batch->candCpyComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // Multiplication will change the plane
      }

      if ( batch->flags & FLAG_SYNCH )
      {
	// Wait for all the input FFT's to complete
	for (int ss = 0; ss< batch->noStacks; ss++)
	{
	  cuFfdotStack* cStack2 = &batch->stacks[ss];
	  cudaStreamWaitEvent(cStack->multStream, cStack2->inpFFTinitComp, 0);
	}

	// Wait for the previous multiplication to complete
	if ( pStack != NULL )
	  cudaStreamWaitEvent(cStack->multStream, pStack->multComp, 0);
      }
    }

    PROF // Profiling  .
    {
      if ( batch->flags & FLAG_PROF )
      {
	CUDA_SAFE_CALL(cudaEventRecord(cStack->multInit, cStack->multStream),"Recording event: multInit");
      }
    }

    FOLD // call kernel(s)  .
    {
      for (int plane = 0; plane < cStack->noInStack; plane++)         // Loop through planes in stack
      {
	cuHarmInfo* cHInfo    = &cStack->harmInf[plane];              // The current harmonic we are working on
	cuFFdot*    cPlane    = &cStack->planes[plane];               // The current f-∂f plane

	dimGrid.x = ceil(cHInfo->width / (float) ( CNV_DIMX * CNV_DIMY ));
	dimGrid.y = 1;

	for (int step = 0; step < batch->noSteps; step++)             // Loop through Steps
	{
	  d_iData         = cPlane->d_iData + cStack->strideCmplx * step;

	  if      ( batch->flags & FLAG_ITLV_ROW )
	  {
	    fprintf(stderr,"ERROR: Cannot do single plane multiplications with row-interleaved multi step stacks.\n");
	    exit(EXIT_FAILURE);
	  }
#ifdef WITH_ITLV_PLN
	  else
	  {
	    // Shift by plane height
	    if ( batch->flags & FLAG_DOUBLE )
	      d_planeData   = (double2*)cPlane->d_planeMult + step * cHInfo->noZ * cStack->strideCmplx;
	    else
	      d_planeData   = (float2*) cPlane->d_planeMult + step * cHInfo->noZ * cStack->strideCmplx;
	  }
#else
	  else
	  {
	    fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
	    exit(EXIT_FAILURE);
	  }
#endif

	  // Texture memory in multiplication is now deprecated
	  //if ( batch->flag & FLAG_TEX_MUL )
	  //  mult12<<<dimGrid, dimBlock, 0, cStack->multStream>>>(d_planeData, cHInfo->width, cStack->strideCmplx, cHInfo->height, d_iData, cPlane->kernel->kerDatTex);
	  //else
	  mult11<<<dimGrid, dimBlock, 0, cStack->multStream>>>((fcomplexcu*)d_planeData, cHInfo->width, cStack->strideCmplx, cHInfo->noZ, d_iData, (fcomplexcu*)cPlane->kernel->d_kerData);

	  // Run message
	  CUDA_SAFE_CALL(cudaGetLastError(), "At multiplication kernel launch");
	}
      }
    }

    FOLD // Synchronisation  .
    {
      cudaEventRecord(cStack->multComp, cStack->multStream);
    }

    pStack = cStack;
  }

}

/** Multiply a specific stack using one of the multiplication 2 or 0 kernels  .
 *
 * @param batch
 * @param cStack
 * @param pStack
 */
void multiplyStack(cuFFdotBatch* batch, cuFfdotStack* cStack, cuFfdotStack* pStack = NULL)
{
  infoMSG(3,4,"Multiply stack\n");

  FOLD // Synchronisation  .
  {
    infoMSG(3,5,"Pre synchronisation\n");

    // This iteration
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->inpFFTinitComp,    0),   "Waiting for GPU to be ready to copy data to device.");  // Need input data

    // iFFT has its own data so can start once iFFT is complete
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->ifftComp,    0),   "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

    if ( (batch->retType & CU_STR_PLN) && !(batch->flags & FLAG_CUFFT_CB_OUT) )
    {
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, batch->candCpyComp,  0), "Waiting for GPU to be ready to copy data to device.");  // Multiplication will change the plane
    }

    if ( batch->flags & FLAG_SYNCH )
    {
      // Wait for all the input FFT's to complete
      for (int synchIdx = 0; synchIdx < batch->noStacks; synchIdx++)
      {
	cuFfdotStack* cStack2 = &batch->stacks[synchIdx];
	cudaStreamWaitEvent(cStack->multStream, cStack2->inpFFTinitComp, 0);
      }

      // Wait for the previous multiplication to complete
      if ( pStack != NULL )
	cudaStreamWaitEvent(cStack->multStream, pStack->multComp, 0);
    }
  }

  PROF // Profiling  .
  {
    if ( batch->flags & FLAG_PROF )
    {
      CUDA_SAFE_CALL(cudaEventRecord(cStack->multInit, cStack->multStream),"Recording event: multInit");
    }
  }

  FOLD // Call kernel(s) .
  {
    infoMSG(3,5,"Kernel call\n");

    if      ( cStack->flags & FLAG_MUL_00 )
    {
      mult00(cStack->multStream, batch, cStack);
    }
    else if ( cStack->flags & FLAG_MUL_21 )
    {
      mult21(cStack->multStream, batch, cStack);
    }
    else if ( cStack->flags & FLAG_MUL_22 )
    {
      mult22(cStack->multStream, batch, cStack);
    }
    else if ( cStack->flags & FLAG_MUL_23 )
    {
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
    infoMSG(3,5,"Post synchronisation\n");

    cudaEventRecord(cStack->multComp, cStack->multStream);
  }
}

/** Call all the multiplication kernels for batch  .
 *
 * @param batch
 */
void multiplyBatch(cuFFdotBatch* batch)
{
  PROF // Profiling  .
  {
    if ( (batch->flags & FLAG_PROF) )
    {
      if ( batch->rActive < batch->noRArryas-1 )
      {
	if ( (*batch->rAraays)[batch->rActive+1][0][0].numrs )
	{
	  timeEvents( batch->multInit, batch->multComp, &batch->compTime[NO_STKS*TIME_CMP_MULT], "Batch multiplication");

	  for (int stack = 0; stack < batch->noStacks; stack++)
	  {
	    cuFfdotStack* cStack = &batch->stacks[stack];

	    timeEvents( cStack->multInit, cStack->multComp, &batch->compTime[NO_STKS*TIME_CMP_MULT + stack ],  "Stack multiplication");
	    timeEvents( cStack->ifftInit, cStack->ifftComp, &batch->compTime[NO_STKS*TIME_CMP_IFFT + stack ],  "Stack iFFT");
	  }
	}
      }
      else
      {
	fprintf(stderr,"ERROR: previous of the active step is out of bounds.");
      }
    }
  }

  if ( (*batch->rAraays)[batch->rActive][0][0].numrs )
  {
    infoMSG(1,2,"Multiply\n");

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Multiply");
    }

    if      ( batch->flags & FLAG_MUL_BATCH )  // Do the multiplications one family at a time  .
    {
      FOLD // Synchronisation  .
      {
	// Synchronise input data preparation for all stacks
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
	  CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, batch->searchComp, 0),    "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
	}

	if ( (batch->retType & CU_STR_PLN) && !(batch->flags & FLAG_CUFFT_CB_OUT) )
	{
	  CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, batch->candCpyComp, 0),   "Waiting for GPU to be ready to copy data to device.");   // Multiplication will change the plane
	}
      }

      FOLD // Call kernel  .
      {
	PROF // Profiling  .
	{
	  if ( batch->flags & FLAG_PROF )
	  {
	    CUDA_SAFE_CALL(cudaEventRecord(batch->multInit, batch->multStream),"Recording event: multInit");
	  }
	}

	mult31(batch->multStream, batch);

	// Run message
	CUDA_SAFE_CALL(cudaGetLastError(), "At kernel launch");
      }

      FOLD // Synchronisation  .
      {
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

	infoMSG(3,3,"Stack %i\n", stkIdx);

	FOLD // Multiply  .
	{
	  if ( batch->flags & FLAG_MUL_CB )
	  {
	    // Just synchronise, the iFFT will do the multiplication once the multComp event has been recorded
	    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack->inpFFTinitComp,0),   "Waiting for GPU to be ready to copy data to device.");
	    CUDA_SAFE_CALL(cudaEventRecord(cStack->multComp, cStack->fftPStream),         "Recording event: multComp");
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

/**  iFFT a specific stack  .
 *
 * @param batch
 * @param cStack
 * @param pStack
 */
void IFFTStack(cuFFdotBatch* batch, cuFfdotStack* cStack, cuFfdotStack* pStack)
{
  infoMSG(3,4,"iFFT Stack\n");

  FOLD // Synchronisation  .
  {
    infoMSG(3,5,"pre synchronisation\n");

    // Wait for multiplication to finish
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack->multComp,      0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, batch->multComp,       0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

    // Wait for previous iFFT to finish
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack->ifftComp,      0), "Waiting for GPU to be ready to copy data to device.");
    if ( batch->flags & FLAG_SS_INMEM  )
    {
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack->ifftMemComp, 0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
    }

    // Wait for previous search to finish
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, batch->searchComp,     0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

    if ( (batch->retType & CU_STR_PLN) && (batch->flags & FLAG_CUFFT_CB_OUT) ) // This has been deprecated!
    {
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, batch->candCpyComp,  0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
    }

    if ( batch->flags & FLAG_SYNCH )
    {
      // Wait for all the multiplications to complete
      for (int synchIdx = 0; synchIdx < batch->noStacks; synchIdx++)
      {
	cuFfdotStack* cStack2 = &batch->stacks[synchIdx];
	cudaStreamWaitEvent(cStack->fftPStream, cStack2->multComp, 0);
      }

      // Wait for the previous fft to complete
      if ( pStack != NULL )
	cudaStreamWaitEvent(cStack->fftPStream, pStack->ifftComp, 0);
    }

  }

  FOLD // Call the inverse CUFFT  .
  {
    infoMSG(3,5,"Call the inverse CUFFT\n");

    if ( cStack->flags & CU_FFT_SEP )
    {
      //#pragma omp critical
      {
	PROF // Profiling  .
	{
	  if ( batch->flags & FLAG_PROF )
	  {
	    cudaEventRecord(cStack->ifftInit, cStack->fftPStream);
	  }
	}

	FOLD // Set the load and store FFT callback if necessary  .
	{
	  setCB(batch, cStack);
	}

	FOLD // Call the FFT  .
	{
	  void* dst = getCBwriteLocation(batch, cStack);

	  CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with multStream.");

	  if ( batch->flags & FLAG_DOUBLE )
	    CUFFT_SAFE_CALL(cufftExecZ2Z(cStack->plnPlan, (cufftDoubleComplex *) cStack->d_planeMult, (cufftDoubleComplex *) dst, CUFFT_INVERSE),"Error executing CUFFT plan.");
	  else
	    CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) cStack->d_planeMult, (cufftComplex *) dst, CUFFT_INVERSE),"Error executing CUFFT plan.");
	}
      }
    }
    else
    {
#pragma omp critical
      {
	PROF // Profiling  .
	{
	  if ( batch->flags & FLAG_PROF )
	  {
	    cudaEventRecord(cStack->ifftInit, cStack->fftPStream);
	  }
	}

	FOLD // Set the load and store FFT callback if necessary  .
	{
	  setCB(batch, cStack);
	}

	FOLD // Call the FFT  .
	{
	  void* dst = getCBwriteLocation(batch, cStack);

	  CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with multStream.");

	  if ( batch->flags & FLAG_DOUBLE )
	    CUFFT_SAFE_CALL(cufftExecZ2Z(cStack->plnPlan, (cufftDoubleComplex *) cStack->d_planeMult, (cufftDoubleComplex *) dst, CUFFT_INVERSE),"Error executing CUFFT plan.");
	  else
	    CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) cStack->d_planeMult, (cufftComplex *) dst, CUFFT_INVERSE),"Error executing CUFFT plan.");
	}
      }
    }
  }

  FOLD // Synchronisation  .
  {
    infoMSG(3,5,"post synchronisation\n");

    cudaEventRecord(cStack->ifftComp, cStack->fftPStream);

    // If using power calculate call back with the inmem plane
    if ( batch->flags & FLAG_CUFFT_CB_INMEM )
    {
#if CUDA_VERSION >= 6050
      cudaEventRecord(cStack->ifftMemComp, cStack->fftPStream);
    }
#endif
  }

  FOLD // Plot  .
  {
#ifdef CBL
    if ( batch->flags & FLAG_DPG_PLT_POWERS )
    {
      FOLD // Synchronisation  .
      {
	CUDA_SAFE_CALL(cudaEventSynchronize(cStack->ifftMemComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
      }

      FOLD // Get data  .
      {
	float* outVals = (float*)malloc(batch->plnDataSize);
	void*  tmpRow  = malloc(batch->inpDataSize);
	ulong sz  = 0;

	for ( int plainNo = 0; plainNo < cStack->noInStack; plainNo++ )
	{
	  cuHarmInfo* cHInfo    = &cStack->harmInf[plainNo];		// The current harmonic we are working on
	  void*       tmpRow      = malloc(batch->inpDataSize);
	  cuFFdot*    plan      = &cStack->planes[plainNo];		// The current plane

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
		fcomplexcu *cmplxData;
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
#if CUDA_VERSION >= 7050   // Half precision getter and setter  .
		    powers =  &((half*)      plan->d_planePowr)[offset];
		    elsz   = sizeof(half);
		    CUDA_SAFE_CALL(cudaMemcpy(tmpRow, powers, (rVal->numrs)*elsz,   cudaMemcpyDeviceToHost), "Failed to copy input data from device.");

		    for ( int i = 0; i < rVal->numrs; i++)
		    {
		      outVals[i] = half2float(((ushort*)tmpRow)[i]);
		    }
#else
		    fprintf(stderr, "ERROR: Half precision can only be used with CUDA 7.5 or later! Reverting to single precision!\n");
		    exit(EXIT_FAILURE);
#endif
		  }
		  else if ( batch->flags & FLAG_CUFFT_CB_POW )
		  {
		    powers =  &((float*)     plan->d_planePowr)[offset];
		    elsz   = sizeof(float);
		    CUDA_SAFE_CALL(cudaMemcpy(outVals, powers, (rVal->numrs)*elsz,   cudaMemcpyDeviceToHost), "Failed to copy input data from device.");
		  }
		  else
		  {
		    powers =  &((fcomplexcu*) plan->d_planePowr)[offset];
		    elsz   = sizeof(cmplxData);
		    CUDA_SAFE_CALL(cudaMemcpy(tmpRow, powers, (rVal->numrs)*elsz,   cudaMemcpyDeviceToHost), "Failed to copy input data from device.");

		    for ( int i = 0; i < rVal->numrs; i++)
		    {
		      outVals[i] = POWERC(((fcomplexcu*)tmpRow)[i]);
		    }
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
		infoMSG(4,4,"Image\n");

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
      infoMSG(1,2,"Inverse FFT Batch\n");

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

	infoMSG(3,3,"Stack %i\n", stkIdx);

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


