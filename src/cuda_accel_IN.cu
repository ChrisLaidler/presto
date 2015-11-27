#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"
#include "cuda_accel_IN.h"

int    cuMedianBuffSz = -1;

void CPU_Norm_Spread(cuFFdotBatch* batch, int norm_type, fcomplexcu* fft)
{
  nvtxRangePush("CPU_Norm_Spread");

  int harm = 0;

  FOLD // Normalise, spread and copy raw input fft data to pinned memory  .
  {
    for (int stack = 0; stack < batch->noStacks; stack++)
    {
      cuFfdotStack* cStack = &batch->stacks[stack];

      int sz = 0;

#ifdef TIMING // Timing  .
      struct timeval start, end;
      gettimeofday(&start, NULL);
#endif

      for (int si = 0; si < cStack->noInStack; si++)
      {
        for (int step = 0; step < batch->noSteps; step++)
        {
          rVals* rVal = &batch->rValues[step][harm];

          if ( rVal->numdata )
          {
            if ( norm_type== 0 )  // Normal normalise  .
            {
              double norm;    /// The normalising factor
              int start = rVal->lobin < 0 ? -rVal->lobin : 0 ;
              int end   = rVal->lobin + rVal->numdata >= batch->SrchSz->searchRHigh ? rVal->lobin + rVal->numdata - batch->SrchSz->searchRHigh : rVal->numdata ;

              FOLD // Calculate and store powers  .
              {
                nvtxRangePush("Powers");
                for (int ii = 0; ii < rVal->numdata; ii++)
                {
                  if ( rVal->lobin+ii < 0 || rVal->lobin+ii  >= batch->SrchSz->searchRHigh ) // Zero Pad
                  {
                    batch->normPowers[ii] = 0;
                  }
                  else
                  {
                    batch->normPowers[ii] = POWERCU(fft[rVal->lobin+ii].r, fft[rVal->lobin+ii].i);
                  }
                }
                nvtxRangePop();
              }

              FOLD // Calculate normalisation factor from median  .
              {
                nvtxRangePush("Median");
                if ( batch->flags & CU_NORM_EQUIV )
                {
                  norm = 1.0 / sqrt(median(batch->normPowers, (rVal->numdata)) / log(2.0));        /// NOTE: This is the same method as CPU version
                }
                else
                {
                  norm = 1.0 / sqrt(median(&batch->normPowers[start], (end-start)) / log(2.0));    /// NOTE: This is a slightly better method (in my opinion)
                }
                nvtxRangePop();
              }

              FOLD // Normalise and spread  .
              {
                nvtxRangePush("Write");
                for (int ii = 0; ( ii < rVal->numdata ) && ( (ii*ACCEL_NUMBETWEEN) < cStack->strideCmplx ); ii++)
                {
                  if ( rVal->lobin+ii < 0  || rVal->lobin+ii  >= batch->SrchSz->searchRHigh )  // Zero Pad
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

                    cStack->h_iData[sz + ii * ACCEL_NUMBETWEEN].r = fft[rVal->lobin + ii].r * norm;
                    cStack->h_iData[sz + ii * ACCEL_NUMBETWEEN].i = fft[rVal->lobin + ii].i * norm;
                  }
                }
                nvtxRangePop();
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
                batch->normPowers[ii] = POWERCU(fft[rVal->lobin+ii].r, fft[rVal->lobin+ii].i);
              }
              loc_powers = corr_loc_pow(batch->normPowers, nice_numdata);

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

#ifdef TIMING // Timing  .
      gettimeofday(&end, NULL);

      float v1 =  ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec))*1e-3  ;
      batch->normTime[stack] += v1;
#endif
    }
  }

  nvtxRangePop();
}

/** Calculate the r bin values for this batch of steps and store them in planes->rInput
 *
 * This calculates r-low and halfwidth
 *
 * @param batch the batch to work with
 * @param searchRLow an array of the step r-low values
 * @param searchRHi an array of the step r-high values
 */
void setStackRVals(cuFFdotBatch* batch, double* searchRLow, double* searchRHi)
{
#ifdef STPMSG
  printf("\tSet Stack R-Vals\n");
#endif

  int       hibin;
  int       binoffset;  // The extra bins to add onto the start of the data
  double    drlo, drhi;

  int lobin;      /// The first bin to copy from the the input fft ( serachR scaled - halfwidth )
  int numdata;    /// The number of input fft points to read
  int numrs;      /// The number of good bins in the plane ( expanded units )

  for (int harm = 0; harm < batch->noHarms; harm++)
  {
    cuHarmInfo* cHInfo      = &batch->hInfos[harm];       // The current harmonic we are working on
    binoffset               = batch->planes[harm].kerStart / ACCEL_NUMBETWEEN; // This aligns all the planes so the all the "usable" parts start at the same offset in the stack

    for (int step = 0; step < batch->noSteps; step++)
    {
      rVals* rVal           = &batch->rValues[step][harm];

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
#ifdef TIMING // Timing variables  .
  struct timeval start, end;
#endif

  if ( batch->rValues[0][0].numrs ) // This is real data ie this isn't just a call to finish off asynchronous work
  {
    nvtxRangePush("Input");

    fcomplexcu* fft = (fcomplexcu*)batch->sInf->sSpec->fftInf.fft;

#ifdef STPMSG
    printf("\tInput\n");
#endif

    FOLD  // Normalise and spread and copy to device memory  .
    {
      if ( batch->flags & CU_NORM_CPU  ) // Copy chunks of FFT data and normalise and spread using the CPU  .
      {
#ifdef STPMSG
        printf("\t\tCPU normalisation\n");
#endif

        FOLD // Blocking synchronisation, Make sure the previous thread has complete reading from page locked memory
        {
          nvtxRangePush("EventSynch");
          CUDA_SAFE_CALL(cudaGetLastError(), "Before Synchronising");
          CUDA_SAFE_CALL(cudaEventSynchronize(batch->iDataCpyComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
          nvtxRangePop();
        }

        FOLD // Zero pinned host memory  .
        {
          nvtxRangePush("Zero");
          memset(batch->h_iData, 0, batch->inpDataSize);
          nvtxRangePop();
        }

        CPU_Norm_Spread(batch, norm_type, fft);

        if ( batch->flags & CU_INPT_FFT_CPU ) // CPU FFT  .
        {
#ifdef STPMSG
          printf("\t\tCPU FFT Input\n");
#endif

#pragma omp critical
          FOLD
          {
            for (int stack = 0; stack < batch->noStacks; stack++)
            {
              cuFfdotStack* cStack = &batch->stacks[stack];

#ifdef TIMING // Timing  .
              gettimeofday(&start, NULL);
#endif

              nvtxRangePush("CPU FFT");
              fftwf_execute_dft(cStack->inpPlanFFTW, (fftwf_complex*)cStack->h_iData, (fftwf_complex*)cStack->h_iData);
              nvtxRangePop();

#ifdef TIMING // Timing  .
              gettimeofday(&end, NULL);

              float v1 =  ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec))*1e-3  ;
              //printf("Input FFT stack %02i  %15.2f \n", stack, v1);
              batch->InpFFTTime[stack] += v1;
#endif

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

#ifdef TIMING
          cudaEventRecord(batch->iDataCpyInit, batch->inpStream);
#endif

        }

        FOLD // Copy pinned memory to device  .
        {
#ifdef STPMSG
          printf("\t\tCopy to device\n");
#endif
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
      else                              // Copy chunks of FFT data and normalise and spread using the GPU  .
      {
#ifdef STPMSG
        printf("\t\tGPU normalisation\n");
#endif

        FOLD // Synchronisation  .
        {
          // Make sure the previous thread has complete reading from page locked memory
          nvtxRangePush("EventSynch");
          CUDA_SAFE_CALL(cudaEventSynchronize(batch->iDataCpyComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
          nvtxRangePop();
        }

        FOLD // Zero pinned host memory  .
        {
          nvtxRangePush("Zero");
          memset(batch->h_iData, 0, batch->inpDataSize);
          nvtxRangePop();
        }

        FOLD // Copy fft data to device  .
        {
          FOLD // Write fft data segments to contiguous page locked memory  .
          {
            int harm  = 0;
            int sz    = 0;

            for ( int stack = 0; stack< batch->noStacks; stack++)  // Loop over stack
            {
              cuFfdotStack* cStack = &batch->stacks[stack];

              for ( int plane = 0; plane < cStack->noInStack; plane++)
              {
                for (int step = 0; step < batch->noSteps; step++)
                {
                  rVals* rVal = &batch->rValues[step][harm];

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
            // Wait for per stack multiplications to finish
            for (int ss = 0; ss< batch->noStacks; ss++)
            {
              cuFfdotStack* cStack = &batch->stacks[ss];
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, cStack->multComp, 0), "Waiting for GPU to be ready to copy data to device\n");
            }

            // Wait for batch multiplication to finish
            CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, batch->multComp, 0), "Waiting for GPU to be ready to copy data to device\n");

#ifdef TIMING  // Timing  .
            cudaEventRecord(batch->iDataCpyInit, batch->inpStream);
#endif
          }

          FOLD // Copy to device  .
          {
#ifdef STPMSG
            printf("\t\tCopy to device\n");
#endif
            CUDA_SAFE_CALL(cudaMemcpyAsync(batch->d_iData, batch->h_iData, batch->inpDataSize, cudaMemcpyHostToDevice, batch->inpStream), "Failed to copy data to device");

            // Synchronisation
            cudaEventRecord(batch->iDataCpyComp, batch->inpStream);

            CUDA_SAFE_CALL(cudaGetLastError(), "Copying a section of input FTD data to the device.");
          }
        }

        FOLD // Normalise and spread on GPU  .
        {
#ifdef STPMSG
          printf("\t\tNormalise on device\n");
#endif

#ifdef SYNCHRONOUS
          cuFfdotStack* pStack = NULL;  // Previous stack
#endif

          for ( int stack = 0; stack < batch->noStacks; stack++)  // Loop over stacks  .
          {
            cuFfdotStack* cStack = &batch->stacks[stack];

            FOLD // Synchronisation  .
            {
              CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->inptStream, batch->iDataCpyComp, 0), "Waiting for GPU to be ready to copy data to device\n");

#ifdef SYNCHRONOUS
              // Wait for previous FFT to complete
              if ( pStack != NULL )
                cudaStreamWaitEvent(cStack->inptStream, pStack->normComp, 0);
#endif

#ifdef TIMING
              cudaEventRecord(cStack->normInit, cStack->inptStream);
#endif
            }

            FOLD // Call the kernel to normalise and spread the input data  .
            {
              normAndSpread_f(cStack->inptStream, batch, stack );
            }

            FOLD // Synchronisation  .
            {
              cudaEventRecord(cStack->normComp, cStack->inptStream);

#ifdef SYNCHRONOUS
              pStack = cStack;
#endif
            }
          }

#ifdef SYNCHRONOUS // Wait for the last stack to complete normalisation  .
          cuFfdotStack* lStack = &batch->stacks[batch->noStacks -1];
          cudaStreamWaitEvent(lStack->inptStream, lStack->normComp, 0);
          cudaEventRecord(batch->normComp, lStack->inptStream);
#endif
        }
      }
    }

    FOLD  // FFT the input on the GPU data  .
    {
      if ( !(batch->flags & CU_INPT_FFT_CPU) )
      {
#ifdef STPMSG
        printf("\t\tGPU FFT\n");
#endif

#ifdef SYNCHRONOUS
        cuFfdotStack* pStack = NULL;  // Previous stack
#endif

        for (int stackIdx = 0; stackIdx < batch->noStacks; stackIdx++)
        {
          cuFfdotStack* cStack = &batch->stacks[stackIdx];

          CUDA_SAFE_CALL(cudaGetLastError(), "Before input fft.");

          FOLD // Synchronisation  .
          {
            cudaStreamWaitEvent(cStack->fftIStream, cStack->normComp,     0);
            cudaStreamWaitEvent(cStack->fftIStream, batch->normComp,      0);
            cudaStreamWaitEvent(cStack->fftIStream, batch->iDataCpyComp,  0);

#ifdef SYNCHRONOUS
            // Wait for the search to complete before FFT'ing the next set of input
            cudaStreamWaitEvent(cStack->fftIStream, batch->searchComp, 0);

            // Wait for previous FFT to complete
            if ( pStack != NULL )
              cudaStreamWaitEvent(cStack->fftIStream, pStack->prepComp, 0);

            // Wait for all GPU normalisations to complete
            for (int stack2Idx = 0; stack2Idx < batch->noStacks; stack2Idx++)
            {
              cuFfdotStack* stack2 = &batch->stacks[stackIdx];
              cudaStreamWaitEvent(cStack->fftIStream, stack2->normComp, 0);
            }
#endif
          }

          FOLD // Do the FFT on the GPU  .
          {
#pragma omp critical
            FOLD // Kernel
            {
#ifdef TIMING // Event .
              cudaEventRecord(cStack->inpFFTinit, cStack->fftIStream);
#endif

              CUFFT_SAFE_CALL(cufftSetStream(cStack->inpPlan, cStack->fftIStream),"Failed associating a CUFFT plan with FFT input stream\n");
              CUFFT_SAFE_CALL(cufftExecC2C(cStack->inpPlan, (cufftComplex *) cStack->d_iData, (cufftComplex *) cStack->d_iData, CUFFT_FORWARD),"Failed to execute input CUFFT plan.");

              CUDA_SAFE_CALL(cudaGetLastError(), "FFT'ing the input data.");
            }
          }

          FOLD // Synchronisation  .
          {
            cudaEventRecord(cStack->prepComp, cStack->fftIStream);

#ifdef SYNCHRONOUS
            pStack = cStack;
#endif

          }
        }
      }
    }

    nvtxRangePop();
  }
}
