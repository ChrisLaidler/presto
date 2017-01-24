#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"
#include "cuda_accel_IN.h"

int    cuMedianBuffSz = -1;             ///< The size of the sub sections to use in the CUDA median selection algorithm - If <= 0 an automatic value is used

void CPU_Norm_Spread(cuFFdotBatch* batch, fcomplexcu* fft)
{
  NV_RANGE_PUSH("CPU_Norm_Spread");

  int harm = 0;

  FOLD // Normalise, spread and copy raw input fft data to pinned memory  .
  {
    for (int stack = 0; stack < batch->noStacks; stack++)
    {
      cuFfdotStack* cStack = &batch->stacks[stack];

      int sz = 0;
      struct timeval start, end;  // Timing variables
      int noRespPerBin = batch->cuSrch->sSpec->noResPerBin;

      if ( batch->flags & FLAG_TIME ) // Timing  .
      {
        gettimeofday(&start, NULL);
      }

      for (int si = 0; si < cStack->noInStack; si++)
      {
        for (int step = 0; step < batch->noSteps; step++)
        {
          rVals* rVal = &(*batch->rAraays)[batch->rActive][step][harm];

          if ( rVal->numdata )
          {
            if ( batch->cuSrch->sSpec->normType == 0 )	// Block median normalisation  .
            {
              int start = rVal->lobin < 0 ? -rVal->lobin : 0 ;
              int end   = rVal->lobin + rVal->numdata >= batch->cuSrch->SrchSz->searchRHigh ? rVal->lobin + rVal->numdata - batch->cuSrch->SrchSz->searchRHigh : rVal->numdata ;

              if ( rVal->norm == 0.0 )
              {
                FOLD // Calculate and store powers  .
                {
                  NV_RANGE_PUSH("Powers");
                  for (int ii = 0; ii < rVal->numdata; ii++)
                  {
                    if ( rVal->lobin+ii < 0 || rVal->lobin+ii  >= batch->cuSrch->SrchSz->searchRHigh ) // Zero Pad
                    {
                      batch->h_normPowers[ii] = 0;
                    }
                    else
                    {
                      batch->h_normPowers[ii] = POWERCU(fft[rVal->lobin+ii].r, fft[rVal->lobin+ii].i);
                    }
                  }
                  NV_RANGE_POP();
                }

                FOLD // Calculate normalisation factor from median  .
                {
                  NV_RANGE_PUSH("Median");
                  if ( batch->flags & CU_NORM_EQUIV )
                  {
                    rVal->norm = 1.0 / sqrt(median(batch->h_normPowers, (rVal->numdata)) / log(2.0));        /// NOTE: This is the same method as CPU version
                  }
                  else
                  {
                    rVal->norm = 1.0 / sqrt(median(&batch->h_normPowers[start], (end-start)) / log(2.0));    /// NOTE: This is a slightly better method (in my opinion)
                  }
                  NV_RANGE_POP();
                }
              }

              FOLD // Normalise and spread  .
              {
                NV_RANGE_PUSH("Write");
                for (int ii = 0; ( ii < rVal->numdata ) && ( (ii*noRespPerBin) < cStack->strideCmplx ); ii++)
                {
                  if ( rVal->lobin+ii < 0  || rVal->lobin+ii  >= batch->cuSrch->SrchSz->searchRHigh )  // Zero Pad
                  {
                    cStack->h_iBuffer[sz + ii * noRespPerBin].r = 0;
                    cStack->h_iBuffer[sz + ii * noRespPerBin].i = 0;
                  }
                  else
                  {
                    if ( ii * noRespPerBin > cStack->strideCmplx )
                    {
                      fprintf(stderr, "ERROR: nice_numdata is greater that width.\n");
                      exit(EXIT_FAILURE);
                    }

                    cStack->h_iBuffer[sz + ii * noRespPerBin].r = fft[rVal->lobin + ii].r * rVal->norm;
                    cStack->h_iBuffer[sz + ii * noRespPerBin].i = fft[rVal->lobin + ii].i * rVal->norm;
                  }
                }
                NV_RANGE_POP();
              }
            }
            else					// or double-tophat normalisation
            {
              int nice_numdata = cu_next2_to_n(rVal->numdata);  // for FFTs

              if ( nice_numdata > cStack->width )
              {
                fprintf(stderr, "ERROR: nice_numdata is greater that width.\n");
                //exit(EXIT_FAILURE);
              }

              // Do the actual copy
              //memcpy(batch->h_powers, &fft[lobin], numdata * sizeof(fcomplexcu) );

              //  new-style running double-tophat local-power normalization
              float *loc_powers;

              //powers = gen_fvect(nice_numdata);
              for (int ii = 0; ii< nice_numdata; ii++)
              {
                batch->h_normPowers[ii] = POWERCU(fft[rVal->lobin+ii].r, fft[rVal->lobin+ii].i);
              }
              loc_powers = corr_loc_pow(batch->h_normPowers, nice_numdata);

              //memcpy(&batch->h_iBuffer[sz], &fft[lobin], nice_numdata * sizeof(fcomplexcu) );

              for (int ii = 0; ii < rVal->numdata; ii++)
              {
                float norm = invsqrt(loc_powers[ii]);

                batch->h_iBuffer[sz + ii * noRespPerBin].r = fft[rVal->lobin+ ii].r* norm;
                batch->h_iBuffer[sz + ii * noRespPerBin].i = fft[rVal->lobin+ ii].i* norm;
              }

              vect_free(loc_powers);  // I hate doing this!!!
            }
          }

          sz += cStack->strideCmplx;
        }
        harm++;
      }

      if ( batch->flags & FLAG_TIME ) // Timing  .
      {
        gettimeofday(&end, NULL);

        float v1 =  ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec))*1e-3  ;
        batch->compTime[batch->noStacks*TIME_CMP_NRM + stack ] += v1;
      }
    }
  }

  NV_RANGE_POP();
}

/** Calculate the r bin values for this batch of steps and store them in planes->rInput
 *
 * This calculates r-low and halfwidth
 *
 * @param batch the batch to work with
 * @param searchRLow an array of the step r-low values
 * @param searchRHi an array of the step r-high values
 */
void setGenRVals(cuFFdotBatch* batch)
{
  infoMSG(2,2,"Set Stack R-Vals\n");
  NV_RANGE_PUSH("Set R-Valst");

  int       hibin;
  int       binoffset;  // The extra bins to add onto the start of the data
  double    drlo, drhi;

  int lobin;      /// The first bin to copy from the the input fft ( serachR scaled - halfwidth )
  int numdata;    /// The number of input fft points to read
  int numrs;      /// The number of good bins in the plane ( expanded units )
  int noResPerBin;
  for (int harm = 0; harm < batch->noGenHarms; harm++)
  {
    cuHarmInfo* cHInfo		= &batch->hInfos[harm];                             	// The current harmonic we are working on
    noResPerBin			= cHInfo->noResPerBin;
    binoffset			= cHInfo->kerStart / noResPerBin;		// This aligns all the planes so the all the "usable" parts start at the same offset in the stack

    for (int step = 0; step < batch->noSteps; step++)
    {
      rVals* rVal		= &(*batch->rAraays)[batch->rActive][step][harm];
      rVals* rValFund		= &(*batch->rAraays)[batch->rActive][step][0];

      if ( rValFund->drlo == rValFund->drhi )
      {
        rVal->drlo		= 0;
        rVal->lobin		= 0;
        rVal->numrs		= 0;
        rVal->numdata		= 0;
        rVal->expBin		= 0;
        rVal->step		= -1; // Invalid step!
      }
      else
      {
        drlo			= cu_calc_required_r(cHInfo->harmFrac, rValFund->drlo, noResPerBin);
        drhi			= cu_calc_required_r(cHInfo->harmFrac, rValFund->drhi, noResPerBin);

        lobin			= (int) floor(drlo) - binoffset;
        hibin			= (int) ceil(drhi)  + binoffset;

        if ( batch->flags & CU_NORM_GPU )
        {
          // GPU normalisation now relies on all input for a stack being of the same length
          numdata		= ceil(cHInfo->width / (float)noResPerBin); // Thus may use much more input data than is strictly necessary but thats OK!
        }
        else
        {
          // CPU normalisation can normalise differing length data so use the correct lengths
          numdata		= hibin - lobin + 1;
        }

        //numrs			= (int) ((ceil(drhi) - floor(drlo)) * noResPerBin + DBLCORRECT) + 1;
        numrs			= (int) ((ceil(drhi) - floor(drlo)) * noResPerBin);	// DBG This is a test, I found it gave erros with r-res that was greater than 2
        if ( harm == 0 )
          numrs			= batch->accelLen;
        else if ( numrs % noResPerBin )
          numrs			= (numrs / noResPerBin + 1) * noResPerBin;

        rVal->drlo		= drlo;
        rVal->drhi		= drhi;
        rVal->lobin		= lobin;
        rVal->numrs		= numrs;
        rVal->numdata		= numdata;
        rVal->expBin		= (lobin+binoffset)*noResPerBin;

        int noEls		= numrs + 2*cHInfo->kerStart;

        if  ( noEls > cHInfo->width )
        {
          fprintf(stderr, "ERROR: Number of elements in step greater than width of the plane! harm: %i\n", harm);
          exit(EXIT_FAILURE);
        }
      }
    }
  }

  NV_RANGE_POP();
}

/** Calculate the r bin values for this batch of steps and store them in planes->rInput
 *
 * This calculates r-low and halfwidth
 *
 * @param batch the batch to work with
 * @param searchRLow an array of the step r-low values
 * @param searchRHi an array of the step r-high values
 */
void setSearchRVals(cuFFdotBatch* batch, double searchRLow, long len)
{
  infoMSG(2,2,"Set Stack R-Vals\n");

  FOLD // Set the r values for this step  .
  {
    for (int harm = 0; harm < batch->noGenHarms; harm++)
    {
      for (int step = 0; step < batch->noSteps; step++)
      {
        rVals* rVal           = &(*batch->rAraays)[batch->rActive][step][harm];

        if ( (step != 0) || (len == 0) )
        {
          rVal->drlo          = 0;
          rVal->lobin         = 0;
          rVal->numrs         = 0;
          rVal->numdata       = 0;
          rVal->expBin        = 0;
          rVal->step          = -1; // Invalid step!
        }
        else
        {
          rVal->drlo          = searchRLow;
          rVal->lobin         = 0;
          rVal->numrs         = len;
          rVal->numdata       = 0;
          rVal->expBin        = 0;
        }
      }
    }
  }
}

/** Initialise input data for a f-∂f plane(s)  ready for convolution  .
 * This:
 *  Normalises the chunk of input data
 *  Spreads it (interbinning)
 *  FFT it ready for convolution
 *
 * @param batch the batch to work with
 */
void prepInputCPU(cuFFdotBatch* batch )
{
  setGenRVals(batch);

  // Timing
  if ( batch->flags & FLAG_TIME )
  {
    if ( (*batch->rAraays)[batch->rActive+1][0][0].numrs )
    {
      // GPU Normalisation
      if ( batch->flags & CU_NORM_GPU )
      {
        for (int stack = 0; stack < batch->noStacks; stack++)
        {
          cuFfdotStack* cStack = &batch->stacks[stack];

          timeEvents( cStack->normInit, cStack->normComp, &batch->compTime[NO_STKS*TIME_CMP_NRM + stack ],    "Stack input normalisation");
        }
      }

      // Input FFT
      if ( !(batch->flags & CU_INPT_FFT_CPU) )
      {
        for (int stack = 0; stack < batch->noStacks; stack++)
        {
          cuFfdotStack* cStack = &batch->stacks[stack];

          timeEvents( cStack->inpFFTinit, cStack->inpFFTinitComp, &batch->compTime[NO_STKS*TIME_CMP_FFT + stack ],    "Stack input FFT");
        }
      }

      // Copying Data to device
      timeEvents( batch->iDataCpyInit, batch->iDataCpyComp, &batch->compTime[NO_STKS*TIME_CMP_H2D],   "Copy to device");
    }
  }

  if ( (*batch->rAraays)[batch->rActive][0][0].numrs ) // This is real data ie this isn't just a call to finish off asynchronous work
  {
    infoMSG(1,2,"CPU prep input\n");

    NV_RANGE_PUSH("CPU prep input");

    fcomplexcu* fft = (fcomplexcu*)batch->cuSrch->sSpec->fftInf.fft;

    if ( !(batch->flags & CU_NORM_GPU)  ) // Copy chunks of FFT data and normalise and spread using the CPU  .
    {
      // Timing variables  .
      struct timeval start, end;

      infoMSG(2,3,"CPU normalisation\n");

      FOLD // Zero pinned host memory  .
      {
        NV_RANGE_PUSH("Zero");
        memset(batch->h_iBuffer, 0, batch->inpDataSize);
        NV_RANGE_POP();
      }

      FOLD // CPU Normalise  .
      {
        CPU_Norm_Spread(batch, fft);
      }

      FOLD // FFT
      {
        if ( batch->flags & CU_INPT_FFT_CPU ) // CPU FFT  .
        {
          infoMSG(2,3,"CPU FFT Input\n");

  #pragma omp critical
          FOLD
          {
            for (int stack = 0; stack < batch->noStacks; stack++)
            {
              cuFfdotStack* cStack = &batch->stacks[stack];

              if ( batch->flags & FLAG_TIME ) // Timing  .
              {
                gettimeofday(&start, NULL);
              }

              NV_RANGE_PUSH("CPU FFT");
              fftwf_execute_dft(cStack->inpPlanFFTW, (fftwf_complex*)cStack->h_iBuffer, (fftwf_complex*)cStack->h_iBuffer);
              NV_RANGE_POP();

              if ( batch->flags & FLAG_TIME ) // Timing  .
              {
                gettimeofday(&end, NULL);

                float v1 =  ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec))*1e-3  ;
                batch->compTime[NO_STKS*TIME_CMP_FFT + stack ] += v1;
              }
            }
          }
        }
      }
    }

    FOLD  // Copy data to pinned memory  .
    {
      FOLD // Synchronisation [ blocking ]  .
      {
        infoMSG(3,4,"pre synchronisation [blocking] iDataCpyComp\n");

        NV_RANGE_PUSH("EventSynch");
        CUDA_SAFE_CALL(cudaGetLastError(), "Before Synchronising");
        CUDA_SAFE_CALL(cudaEventSynchronize(batch->iDataCpyComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
        NV_RANGE_POP();
      }

      if ( !(batch->flags & CU_NORM_GPU)  )	// Copy CPU prepped data to the pagelocked input data
      {
        NV_RANGE_PUSH("memcpy");
        memcpy(batch->h_iData, batch->h_iBuffer, batch->inpDataSize );
        NV_RANGE_POP();
      }
      else					// Copy chunks of FFT data and normalise and spread using the GPU  .
      {
        infoMSG(2,3,"CPU prep input\n");

        FOLD // Zero pinned host memory  .
        {
          NV_RANGE_PUSH("Zero");
          memset(batch->h_iData, 0, batch->inpDataSize);
          NV_RANGE_POP();
        }

        infoMSG(3,4,"Write to pinned\n");

        FOLD // Write input data segments to contiguous page locked memory  .
        {
          infoMSG(3,5,"Write fft data segments to contiguous page locked memory\n");

          int harm  = 0;
          int sz    = 0;

          for ( int stack = 0; stack< batch->noStacks; stack++)  // Loop over stack
          {
            cuFfdotStack* cStack = &batch->stacks[stack];

            for ( int plane = 0; plane < cStack->noInStack; plane++)
            {
              for (int step = 0; step < batch->noSteps; step++)
              {
                rVals* rVal = &(*batch->rAraays)[batch->rActive][step][harm];

                if ( rVal->numdata )
                {
                  int start = 0;
                  if ( rVal->lobin < 0 )
                    start = -rVal->lobin;

                  // Do the actual copy
                  memcpy(&batch->h_iData[sz+start], &fft[rVal->lobin+start], (rVal->numdata-start) * sizeof(fcomplexcu));
                }
                sz += cStack->strideCmplx;
              }
              harm++;
            }
          }
        }
      }
    }

    NV_RANGE_POP();
  }
}

void copyInputToDevice(cuFFdotBatch* batch)
{
  if ( (*batch->rAraays)[batch->rActive][0][0].numrs ) // This is real data ie this isn't just a call to finish off asynchronous work
  {
    FOLD // Synchronisation  .
    {
      FOLD // Previous
      {
        // Wait for previous per-stack multiplications to finish
        for (int ss = 0; ss < batch->noStacks; ss++)
        {
          cuFfdotStack* cStack = &batch->stacks[ss];
          CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, cStack->multComp, 0), "Waiting for GPU to be ready to copy data to device.\n");
        }

        // Wait for batch multiplications to finish
        CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, batch->multComp, 0), "Waiting for GPU to be ready to copy data to device.\n");
      }

      // Note don't have to wait for GPU input work as it is done in the same stream

      if ( batch->flags & FLAG_TIME ) // Timing  .
      {
        cudaEventRecord(batch->iDataCpyInit, batch->inpStream);
      }
    }

    FOLD // Copy pinned memory to device  .
    {
      infoMSG(2,3,"Copy to device\n");

      CUDA_SAFE_CALL(cudaMemcpyAsync(batch->d_iData, batch->h_iData, batch->inpDataSize, cudaMemcpyHostToDevice, batch->inpStream), "Failed to copy input data to device");
      CUDA_SAFE_CALL(cudaGetLastError(), "Copying input data to the device.");
    }

    FOLD // Synchronisation  .
    {
      cudaEventRecord(batch->iDataCpyComp,  batch->inpStream);

      if ( !(batch->flags & CU_NORM_GPU)  )
      {
        // Data has been normalised by CPU
        cudaEventRecord(batch->normComp,      batch->inpStream);
      }

      if ( batch->flags & CU_INPT_FFT_CPU )
      {
        // Data has been FFT'ed by CPU
        for (int ss = 0; ss < batch->noStacks; ss++)
        {
          cuFfdotStack* cStack = &batch->stacks[ss];
          cudaEventRecord(cStack->inpFFTinitComp, batch->inpStream);
        }
      }
    }
  }
}

void prepInputGPU(cuFFdotBatch* batch)
{
  if ( (*batch->rAraays)[batch->rActive][0][0].numrs )
  {
    infoMSG(1,2,"GPU prep input\n");
    NV_RANGE_PUSH("GPU prep input");

    FOLD // Normalise and spread on GPU  .
    {
      if ( batch->flags & CU_NORM_GPU )
      {
        infoMSG(3,4,"Normalise on device\n");

        cuFfdotStack* pStack = NULL;  // Previous stack

        for ( int stack = 0; stack < batch->noStacks; stack++)  // Loop over stacks  .
        {
          infoMSG(3,5,"Stack %i\n", stack);

          cuFfdotStack* cStack = &batch->stacks[stack];

          FOLD // Synchronisation  .
          {
            // This iteration
            CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->inptStream, batch->iDataCpyComp, 0), "Waiting for GPU to be ready to copy data to device\n");

            if ( batch->flags & FLAG_SYNCH )
            {
              // Wait for previous FFT to complete
              if ( pStack != NULL )
                cudaStreamWaitEvent(cStack->inptStream, pStack->normComp, 0);
            }

            if ( batch->flags & FLAG_TIME ) // Timing  .
            {
              cudaEventRecord(cStack->normInit, cStack->inptStream);
            }
          }

          FOLD // Call the kernel to normalise and spread the input data  .
          {
            normAndSpread(cStack->inptStream, batch, stack );
          }

          FOLD // Synchronisation  .
          {
            cudaEventRecord(cStack->normComp, cStack->inptStream);
          }

          pStack = cStack;
        }

        if ( batch->flags & FLAG_SYNCH ) // Wait for the last stack to complete normalisation  .
        {
          cuFfdotStack* lStack = &batch->stacks[batch->noStacks -1];
          CUDA_SAFE_CALL(cudaStreamWaitEvent(lStack->inptStream, lStack->normComp, 0), "Waiting for event normComp");
          CUDA_SAFE_CALL(cudaEventRecord(batch->normComp, lStack->inptStream), "Recording for event inptStream");
        }
      }
    }

    FOLD // FFT the input on the GPU  .
    {
      if ( !(batch->flags & CU_INPT_FFT_CPU) )
      {
        infoMSG(2,3,"GPU FFT\n");

        cuFfdotStack* pStack = NULL;  // Previous stack

        for (int stackIdx = 0; stackIdx < batch->noStacks; stackIdx++)
        {
          infoMSG(3,4,"Stack %i\n", stackIdx);

          cuFfdotStack* cStack = &batch->stacks[stackIdx];

          CUDA_SAFE_CALL(cudaGetLastError(), "Before input fft.");

          FOLD // Synchronisation  .
          {
            CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftIStream, cStack->normComp,     0), "Waiting for event stack normComp");
            CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftIStream, batch->normComp,      0), "Waiting for event batch normComp");
            CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftIStream, batch->iDataCpyComp,  0), "Waiting for event iDataCpyComp");

            if ( batch->flags & FLAG_SYNCH )
            {
              // Wait for the search to complete before FFT'ing the next set of input
              cudaStreamWaitEvent(cStack->fftIStream, batch->searchComp, 0);

              // Wait for all normalisation to be completed
              cudaStreamWaitEvent(cStack->fftIStream, batch->normComp, 0);

              // Wait for previous FFT to complete
              if ( pStack != NULL )
                cudaStreamWaitEvent(cStack->fftIStream, pStack->inpFFTinitComp, 0);

              // Wait for all GPU normalisations to complete
              for (int stack2Idx = 0; stack2Idx < batch->noStacks; stack2Idx++)
              {
                cuFfdotStack* stack2 = &batch->stacks[stackIdx];
                cudaStreamWaitEvent(cStack->fftIStream, stack2->normComp, 0);
              }
            }
          }

          FOLD // Do the FFT on the GPU  .
          {
#pragma omp critical
            FOLD // Kernel
            {
              if ( batch->flags & FLAG_TIME ) // Timing  .
              {
                cudaEventRecord(cStack->inpFFTinit, cStack->fftIStream);
              }

              CUFFT_SAFE_CALL(cufftSetStream(cStack->inpPlan, cStack->fftIStream),"Failed associating a CUFFT plan with FFT input stream\n");
              CUFFT_SAFE_CALL(cufftExecC2C(cStack->inpPlan, (cufftComplex *) cStack->d_iData, (cufftComplex *) cStack->d_iData, CUFFT_FORWARD),"Failed to execute input CUFFT plan.");

              CUDA_SAFE_CALL(cudaGetLastError(), "FFT'ing the input data.");
            }
          }

          FOLD // Synchronisation  .
          {
            cudaEventRecord(cStack->inpFFTinitComp, cStack->fftIStream);
          }

          pStack = cStack;
        }
      }
    }

    NV_RANGE_POP();
  }
}

/** Initialise input data for a f-∂f plane(s)  ready for convolution  .
 * This:
 *  Normalises the chunk of input data
 *  Spreads it (interbinning)
 *  FFT it ready for convolution
 *
 * @param batch the batch to work with
 */
void prepInput(cuFFdotBatch* batch)
{
  prepInputCPU(batch);
  copyInputToDevice(batch);
  prepInputGPU(batch);
}
