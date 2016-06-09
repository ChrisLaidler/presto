#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"
#include "cuda_accel_IN.h"

int    cuMedianBuffSz = -1;             ///< The size of the sub sections to use in the CUDA median selection algorithm - If <= 0 an automatic value is used

void CPU_Norm_Spread(cuFFdotBatch* batch, int norm_type, fcomplexcu* fft)
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
            if ( norm_type== 0 )  // Normal normalise  .
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
                for (int ii = 0; ( ii < rVal->numdata ) && ( (ii*ACCEL_NUMBETWEEN) < cStack->strideCmplx ); ii++)
                {
                  if ( rVal->lobin+ii < 0  || rVal->lobin+ii  >= batch->cuSrch->SrchSz->searchRHigh )  // Zero Pad
                  {
                    cStack->h_iData[sz + ii * ACCEL_NUMBETWEEN].r = 0;
                    cStack->h_iData[sz + ii * ACCEL_NUMBETWEEN].i = 0;
                  }
                  else
                  {
                    if ( ii * ACCEL_NUMBETWEEN > cStack->strideCmplx )
                    {
                      fprintf(stderr, "ERROR: nice_numdata is greater that width.\n");
                      exit(EXIT_FAILURE);
                    }

                    cStack->h_iData[sz + ii * ACCEL_NUMBETWEEN].r = fft[rVal->lobin + ii].r * rVal->norm;
                    cStack->h_iData[sz + ii * ACCEL_NUMBETWEEN].i = fft[rVal->lobin + ii].i * rVal->norm;
                  }
                }
                NV_RANGE_POP();
              }
            }
            else                  // or double-tophat normalisation
            {
              int nice_numdata = next2_to_n_cu(rVal->numdata);  // for FFTs

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

              //memcpy(&batch->h_iData[sz], &fft[lobin], nice_numdata * sizeof(fcomplexcu) );

              for (int ii = 0; ii < rVal->numdata; ii++)
              {
                float norm = invsqrt(loc_powers[ii]);

                batch->h_iData[sz + ii * ACCEL_NUMBETWEEN].r = fft[rVal->lobin+ ii].r* norm;
                batch->h_iData[sz + ii * ACCEL_NUMBETWEEN].i = fft[rVal->lobin+ ii].i* norm;
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
        batch->normTime[stack] += v1;
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
void setGenRVals(cuFFdotBatch* batch, double* searchRLow, double* searchRHi)
{
  infoMSG(2,2,"Set Stack R-Vals\n");

  int       hibin;
  int       binoffset;  // The extra bins to add onto the start of the data
  double    drlo, drhi;

  int lobin;      /// The first bin to copy from the the input fft ( serachR scaled - halfwidth )
  int numdata;    /// The number of input fft points to read
  int numrs;      /// The number of good bins in the plane ( expanded units )

  for (int harm = 0; harm < batch->noGenHarms; harm++)
  {
    cuHarmInfo* cHInfo      = &batch->hInfos[harm];                             // The current harmonic we are working on
    binoffset               = batch->hInfos[harm].kerStart / ACCEL_NUMBETWEEN;  // This aligns all the planes so the all the "usable" parts start at the same offset in the stack

    for (int step = 0; step < batch->noSteps; step++)
    {
      rVals* rVal           = &(*batch->rAraays)[batch->rActive][step][harm];

      if ( searchRLow[step] == searchRHi[step] )
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
        drlo                = calc_required_r_gpu(cHInfo->harmFrac, searchRLow[step]);
        drhi                = calc_required_r_gpu(cHInfo->harmFrac, searchRHi[step] );

        lobin               = (int) floor(drlo) - binoffset;
        hibin               = (int) ceil(drhi)  + binoffset;

        numdata             = hibin - lobin + 1;
        numrs               = (int) ((ceil(drhi) - floor(drlo)) * ACCEL_RDR + DBLCORRECT) + 1;

        if ( harm == 0 )
          numrs             = batch->accelLen;
        else if ( numrs % ACCEL_RDR )
          numrs             = (numrs / ACCEL_RDR + 1) * ACCEL_RDR;

        rVal->drlo          = drlo;
        rVal->lobin         = lobin;
        rVal->numrs         = numrs;
        rVal->numdata       = numdata;
        rVal->expBin        = (lobin+binoffset)*ACCEL_RDR;

        int noEls           = numrs + 2*binoffset*ACCEL_RDR;

        if  ( noEls > cHInfo->width )
        {
          fprintf(stderr, "ERROR: Number of elements in step greater than width of the plane! harm: %i\n", harm);
          exit(EXIT_FAILURE);
        }
      }
    }
  }
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
 * @param norm_type   The type of normalisation to perform
 */
void initInput(cuFFdotBatch* batch, int norm_type )
{
  // Timing
  if ( batch->flags & FLAG_TIME )
  {
    if ( (*batch->rAraays)[batch->rActive+1][0][0].numrs )
    {
      // GPU Normalisation
      if ( !(batch->flags & CU_NORM_CPU) )
      {
        for (int stack = 0; stack < batch->noStacks; stack++)
        {
          cuFfdotStack* cStack = &batch->stacks[stack];

          timeEvents( cStack->normInit, cStack->normComp, &batch->normTime[stack],    "Stack input normalisation");
        }
      }

      // Input FFT
      if ( !(batch->flags & CU_INPT_FFT_CPU) )
      {
        for (int stack = 0; stack < batch->noStacks; stack++)
        {
          cuFfdotStack* cStack = &batch->stacks[stack];

          timeEvents( cStack->inpFFTinit, cStack->prepComp, &batch->InpFFTTime[stack],    "Stack input FFT");
        }
      }

      // Copying Data to device
      timeEvents( batch->iDataCpyInit, batch->iDataCpyComp, &batch->copyH2DTime[0],   "Copy to device");
    }
  }

  if ( (*batch->rAraays)[batch->rActive][0][0].numrs ) // This is real data ie this isn't just a call to finish off asynchronous work
  {
    infoMSG(1,2,"Input\n");

    NV_RANGE_PUSH("Input");

    fcomplexcu* fft = (fcomplexcu*)batch->cuSrch->sSpec->fftInf.fft;

    FOLD  // Normalise and spread and copy to device memory  .
    {
      if ( batch->flags & CU_NORM_CPU  ) // Copy chunks of FFT data and normalise and spread using the CPU  .
      {
        // Timing variables  .
        struct timeval start, end;

        infoMSG(2,3,"CPU normalisation\n");

        FOLD // Blocking synchronisation, Make sure the previous thread has complete reading from page locked memory
        {
          infoMSG(3,4,"pre synchronisation [blocking] iDataCpyComp\n");

          NV_RANGE_PUSH("EventSynch");
          CUDA_SAFE_CALL(cudaGetLastError(), "Before Synchronising");
          CUDA_SAFE_CALL(cudaEventSynchronize(batch->iDataCpyComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
          NV_RANGE_POP();
        }

        FOLD // Zero pinned host memory  .
        {
          NV_RANGE_PUSH("Zero");
          memset(batch->h_iData, 0, batch->inpDataSize);
          NV_RANGE_POP();
        }

        CPU_Norm_Spread(batch, norm_type, fft);

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
              fftwf_execute_dft(cStack->inpPlanFFTW, (fftwf_complex*)cStack->h_iData, (fftwf_complex*)cStack->h_iData);
              NV_RANGE_POP();

              if ( batch->flags & FLAG_TIME ) // Timing  .
              {
                gettimeofday(&end, NULL);

                float v1 =  ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec))*1e-3  ;
                batch->InpFFTTime[stack] += v1;
              }

            }
          }
        }

        FOLD // Synchronisation  .
        {
          // Wait for per stack multiplications to finish
          for (int ss = 0; ss < batch->noStacks; ss++)
          {
            cuFfdotStack* cStack = &batch->stacks[ss];
            CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, cStack->multComp, 0), "Waiting for GPU to be ready to copy data to device.\n");
          }

          // Wait for batch multiplications to finish
          CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, batch->multComp, 0), "Waiting for GPU to be ready to copy data to device.\n");

          if ( batch->flags & FLAG_TIME ) // Timing  .
          {
            cudaEventRecord(batch->iDataCpyInit, batch->inpStream);
          }

        }

        FOLD // Copy pinned memory to device  .
        {
          infoMSG(2,3,"Copy to device\n");

          CUDA_SAFE_CALL(cudaMemcpyAsync(batch->d_iData, batch->h_iData, batch->inpDataSize, cudaMemcpyHostToDevice, batch->inpStream), "Failed to copy input data to device");
          CUDA_SAFE_CALL(cudaGetLastError(), "Preparing the input data.");
        }

        FOLD // Synchronisation  .
        {
          cudaEventRecord(batch->normComp,      batch->inpStream);
          cudaEventRecord(batch->iDataCpyComp,  batch->inpStream);

          if ( batch->flags & CU_INPT_FFT_CPU )
          {
            for (int ss = 0; ss < batch->noStacks; ss++)
            {
              cuFfdotStack* cStack = &batch->stacks[ss];
              cudaEventRecord(cStack->prepComp, batch->inpStream);
            }
          }
        }
      }
      else                               // Copy chunks of FFT data and normalise and spread using the GPU  .
      {
        infoMSG(2,3,"GPU normalisation\n");

        FOLD // Synchronisation  .
        {
          infoMSG(3,4,"pre synchronisation [blocking] iDataCpyComp\n");

          // Make sure the previous thread has complete reading from page locked memory
          NV_RANGE_PUSH("EventSynch");
          CUDA_SAFE_CALL(cudaEventSynchronize(batch->iDataCpyComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
          NV_RANGE_POP();
        }

        FOLD // Zero pinned host memory  .
        {
          infoMSG(3,4,"Zero pinned memory\n");

          NV_RANGE_PUSH("Zero");
          memset(batch->h_iData, 0, batch->inpDataSize);
          NV_RANGE_POP();
        }

        FOLD // Copy fft data to device  .
        {
          infoMSG(3,4,"Copy data\n");

          FOLD // Write fft data segments to contiguous page locked memory  .
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

          FOLD // Synchronisation  .
          {
            infoMSG(3,5,"Synchronisation\n");

            // Wait for per stack multiplications to finish
            for (int ss = 0; ss< batch->noStacks; ss++)
            {
              cuFfdotStack* cStack = &batch->stacks[ss];
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, cStack->multComp, 0), "Waiting for GPU to be ready to copy data to device\n");
            }

            // Wait for batch multiplication to finish
            CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, batch->multComp, 0), "Waiting for GPU to be ready to copy data to device\n");

            if ( batch->flags & FLAG_TIME ) // Timing  .
            {
              cudaEventRecord(batch->iDataCpyInit, batch->inpStream);\
            }
          }

          FOLD // Copy to device  .
          {
            infoMSG(3,5,"Copy to device\n");

            CUDA_SAFE_CALL(cudaMemcpyAsync(batch->d_iData, batch->h_iData, batch->inpDataSize, cudaMemcpyHostToDevice, batch->inpStream), "Failed to copy data to device");

            // Synchronisation
            cudaEventRecord(batch->iDataCpyComp, batch->inpStream);

            CUDA_SAFE_CALL(cudaGetLastError(), "Copying a section of input FTD data to the device.");
          }
        }

        FOLD // Normalise and spread on GPU  .
        {
          infoMSG(3,4,"Normalise on device\n");

          cuFfdotStack* pStack = NULL;  // Previous stack

          for ( int stack = 0; stack < batch->noStacks; stack++)  // Loop over stacks  .
          {
            infoMSG(3,5,"Stack %i\n", stack);

            cuFfdotStack* cStack = &batch->stacks[stack];

            FOLD // Synchronisation  .
            {
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
    }

    FOLD  // FFT the input on the GPU data  .
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
            CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftIStream, cStack->normComp,     0), "Waiting for event normComp");
            CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftIStream, batch->normComp,      0), "Waiting for event normComp");
            CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftIStream, batch->iDataCpyComp,  0), "Waiting for event iDataCpyComp");

            if ( batch->flags & FLAG_SYNCH )
            {
              // Wait for the search to complete before FFT'ing the next set of input
              cudaStreamWaitEvent(cStack->fftIStream, batch->searchComp, 0);

              // Wait for all normalisation to be completed
              cudaStreamWaitEvent(cStack->fftIStream, batch->normComp, 0);

              // Wait for previous FFT to complete
              if ( pStack != NULL )
                cudaStreamWaitEvent(cStack->fftIStream, pStack->prepComp, 0);

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
            cudaEventRecord(cStack->prepComp, cStack->fftIStream);
          }

          pStack = cStack;
        }
      }
    }

    NV_RANGE_POP();
  }
}
