#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"
#include "cuda_accel_IN.h"



void CPU_Norm_Spread(cuFFdotBatch* batch, double* searchRLow, double* searchRHi, int norm_type, fcomplexcu* fft)
{
  nvtxRangePush("CPU_Norm_Spread");

  int harm = 0;

  FOLD // Copy raw input fft data to device  .
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
        cuHarmInfo* cHInfo  = &batch->hInfos[harm];      // The current harmonic we are working on

        for (int step = 0; step < batch->noSteps; step++)
        {
          if ( !(searchRLow[step] == 0 &&  searchRHi[step] == 0) )
          {
            rVals* rVal = &((*batch->rInput)[step][harm]);

            printf("stack %02i si %02i step %02i  cStack->width %04uli   rVal->numdata %04li \n", stack, si,  step, cStack->width, rVal->numdata);

            if ( norm_type== 0 )  // Normal normalise  .
            {
              double norm;    /// The normalising factor

              //nvtxRangePush("Powers");
              for (int ii = 0; ii < rVal->numdata; ii++)
              {
                if ( rVal->lobin+ii < 0 || rVal->lobin+ii  >= batch->SrchSz->searchRHigh ) // Zero Pad
                {
                  batch->h_powers[ii] = 0;
                }
                else
                {
                  batch->h_powers[ii] = POWERCU(fft[rVal->lobin+ii].r, fft[rVal->lobin+ii].i);
                }
              }
              //nvtxRangePop();

              if ( DBG_INP01 )
              {
                float* data = (float*)&fft[rVal->lobin];
                int gx;
                printf("\nGPU Input Data RAW FFTs [ Half width: %i  lowbin: %i  drlo: %.2f ] \n", cHInfo->halfWidth, rVal->lobin, rVal->drlo);

                for ( gx = 0; gx < 10; gx++)
                  printf("%.4f ",((float*)data)[gx]);
                printf("\n");
              }

              //nvtxRangePush("Median");
              norm = 1.0 / sqrt(median(batch->h_powers, (rVal->numdata))/ log(2.0));                       /// NOTE: This is the same method as CPU version
              //norm = 1.0 / sqrt(median(&plains->h_powers[start], (rVal->numdata-start))/ log(2.0));       /// NOTE: This is a slightly better method (in my opinion)
              //nvtxRangePop();

              // Normalise and spread
              //nvtxRangePush("Write");
              for (int ii = 0; ( ii < rVal->numdata ) && ( (ii*ACCEL_NUMBETWEEN) < cStack->inpStride ); ii++)
              {
                if ( rVal->lobin+ii < 0  || rVal->lobin+ii  >= batch->SrchSz->searchRHigh )  // Zero Pad
                {
                  cStack->h_iData[sz + ii * ACCEL_NUMBETWEEN].r = 0;
                  cStack->h_iData[sz + ii * ACCEL_NUMBETWEEN].i = 0;
                }
                else
                {
                  if ( ii * ACCEL_NUMBETWEEN > cStack->inpStride )
                  {
                    fprintf(stderr, "ERROR: nice_numdata is greater that width.\n");
                    exit(EXIT_FAILURE);
                  }

                  cStack->h_iData[sz + ii * ACCEL_NUMBETWEEN].r = fft[rVal->lobin + ii].r * norm;
                  cStack->h_iData[sz + ii * ACCEL_NUMBETWEEN].i = fft[rVal->lobin + ii].i * norm;
                }
              }
              //nvtxRangePop();
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
                batch->h_powers[ii] = POWERCU(fft[rVal->lobin+ii].r, fft[rVal->lobin+ii].i);
              }
              loc_powers = corr_loc_pow(batch->h_powers, nice_numdata);

              //memcpy(&batch->h_iData[sz], &fft[lobin], nice_numdata * sizeof(fcomplexcu) );

              for (int ii = 0; ii < rVal->numdata; ii++)
              {
                float norm = invsqrt(loc_powers[ii]);

                batch->h_iData[sz + ii * ACCEL_NUMBETWEEN].r = fft[rVal->lobin+ ii].r* norm;
                batch->h_iData[sz + ii * ACCEL_NUMBETWEEN].i = fft[rVal->lobin+ ii].i* norm;
              }

              vect_free(loc_powers);  // I hate doing this!!!
            }

            // I tested doing the FFT's on the CPU and its drastically faster doing it on the GPU, and can often be done synchronously -- Chris L
            //nvtxRangePush("CPU FFT");
            //COMPLEXFFT((fcomplex *)&batch->h_iData[sz], numdata*ACCEL_NUMBETWEEN, -1);
            //nvtxRangePop();
          }

          sz += cStack->inpStride;
        }
        harm++;
      }

#ifdef TIMING // Timing  .
      gettimeofday(&end, NULL);

      float v1 =  ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec))*1e-3  ;
      batch->InpNorm[stack] += v1;
#endif
    }
  }

  nvtxRangePop();
}

/** Calculate the r bin values for this batch of steps and store them in plains->rInput
 *
 * This calculates r-low and halfwidth
 *
 * @param batch the batch to work with
 * @param searchRLow an array of the step r-low values
 * @param searchRHi an array of the step r-high values
 */
void setStackRVals(cuFFdotBatch* batch, double* searchRLow, double* searchRHi)
{
  //printf("setStackRVals\n");

  int       hibin, binoffset;
  double    drlo, drhi;

  int lobin;      /// The first bin to copy from the the input fft ( serachR scaled - halfwidth )
  int numdata;    /// The number of input fft points to read
  int numrs;      /// The number of good bins in the plain ( expanded units )

  for (int harm = 0; harm < batch->noHarms; harm++)
  {
    cuHarmInfo* cHInfo    = &batch->hInfos[harm];       // The current harmonic we are working on
    binoffset             = cHInfo->halfWidth;          //

    for (int step = 0; step < batch->noSteps; step++)
    {
      rVals* rVal         = &((*batch->rInput)[step][harm]);

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
    }
  }
}

/** Initialise input data for a f-∂f plain(s)  ready for convolution  .
 * This:
 *  Normalises the chunk of input data
 *  Spreads it (interbinning)
 *  FFT it ready for convolution
 *
 * @param plains      The plains
 * @param searchRLow  The index of the low  R bin (1 value for each step)
 * @param searchRHi   The index of the high R bin (1 value for each step)
 * @param norm_type   The type of normalisation to perform
 * @param fft         The fft
 */
void initInput(cuFFdotBatch* batch, double* searchRLow, double* searchRHi, int norm_type, fcomplexcu* fft)
{
  dim3 dimBlock, dimGrid;


#ifdef TIMING // Timing  .
  struct timeval start, end;

  if ( batch->haveSearchResults )
  {
    // Make sure the previous thread has complete reading from page locked memory
    CUDA_SAFE_CALL(cudaEventSynchronize(batch->iDataCpyComp), "ERROR: Synchronising before writing input data to page locked host memory.");

    float time;         // Time in ms of the thing
    cudaError_t ret;    // Return status of cudaEventElapsedTime

    //cudaError_t stxcef = cudaEventQuery( batch->iDataCpyComp );

    FOLD // Copy input data  .
    {
      ret = cudaEventElapsedTime(&time, batch->iDataCpyInit, batch->iDataCpyComp);

      if ( ret == cudaErrorNotReady )
      {
        //printf("Not ready\n");
      }
      else
      {
        //printf("    ready\n");
#pragma omp atomic
        batch->copyH2DTime[0] += time;
      }
    }

    FOLD // Input FFT timing  .
    {
      if ( !(batch->flag & CU_INPT_CPU_FFT) )
      {
        for (int ss = 0; ss < batch->noStacks; ss++)
        {
          cuFfdotStack* cStack = &batch->stacks[ss];

          ret = cudaEventElapsedTime(&time, cStack->inpFFTinit, cStack->prepComp);
          if ( ret == cudaErrorNotReady )
          {
            //printf("Not ready\n");
          }
          else
          {
            //printf("    ready\n");
#pragma omp atomic
            batch->InpFFTTime[ss] += time;

            //if ( ss == 0 )
            //  printf("\nFFT: %f ms\n",time);
          }
        }
      }
    }

  }
#endif

  if ( searchRLow[0] < searchRHi[0] ) // This is real data ie this isn't just a call to finish off asynchronous work
  {
    nvtxRangePush("Input");
#ifdef STPMSG
    printf("\tInput\n");
#endif

    // Calculate R values
    setStackRVals(batch, searchRLow, searchRHi );

    FOLD  // Normalise and spread and copy to device memory  .
    {
      if      ( batch->flag & CU_INPT_SINGLE_G  )
      {
        // Copy chunks of FFT data and normalise and spread using the GPU

        FOLD // Synchronisation  .
        {
          // Make sure the previous thread has complete reading from page locked memory
          CUDA_SAFE_CALL(cudaEventSynchronize(batch->iDataCpyComp), "ERROR: copying data to device");
        }

        FOLD // Zero host memory  .
        {
          nvtxRangePush("Zero");
          memset(batch->h_iData, 0, batch->inpDataSize*batch->noSteps);
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
              int idxc  = 0;

              printf("-------\n");

              for ( int plain = 0; plain < cStack->noInStack; plain++)
              {
                //cuFFdot* cPlain         = &batch->plains[harm];     //

                for (int step = 0; step < batch->noSteps; step++)
                {
                  rVals* rVal = &((*batch->rInput)[step][harm]);

                  FOLD // DEBUG  .
                  {
                    for (int ii = 0; ii < rVal->numdata; ii++)
                    {
                      if ( rVal->lobin+ii < 0 || rVal->lobin+ii  >= batch->SrchSz->searchRHigh ) // Zero Pad
                      {
                        batch->h_powers[ii] = 0;
                      }
                      else
                      {
                        batch->h_powers[ii] = POWERCU(fft[rVal->lobin+ii].r, fft[rVal->lobin+ii].i);
                      }
                    }

                    float mm = median(batch->h_powers, (rVal->numdata));
                    printf("%02i  median: %.6f \n",idxc,  mm);
                    idxc++;
                  }

                  int start = 0;
                  if ( rVal->lobin < 0 )
                    start = -rVal->lobin;

                  // Do the actual copy
                  memcpy(&batch->h_iData[sz+start], &fft[rVal->lobin+start], (rVal->numdata-start) * sizeof(fcomplexcu));

                  sz += cStack->inpStride;
                }
                harm++;
              }
            }
          }

          FOLD // Synchronisation  .
          {
            // Wait for per stack convolutions to finish
            for (int ss = 0; ss< batch->noStacks; ss++)
            {
              cuFfdotStack* cStack = &batch->stacks[ss];
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, cStack->convComp, 0), "ERROR: waiting for GPU to be ready to copy data to device\n");
            }

            // Wait for batch convolution to finish
            CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, batch->convComp, 0), "ERROR: waiting for GPU to be ready to copy data to device\n");

#ifdef TIMING  // Timing  .
            cudaEventRecord(batch->iDataCpyInit, batch->inpStream);
#endif
          }

          FOLD // Copy to device  .
          {
#ifdef STPMSG
            printf("\t\tCopy to device\n");
#endif
            CUDA_SAFE_CALL(cudaMemcpyAsync(batch->d_iData, batch->h_iData, batch->inpDataSize*batch->noSteps, cudaMemcpyHostToDevice, batch->inpStream), "Failed to copy data to device");

            // Synchronisation
            cudaEventRecord(batch->iDataCpyComp, batch->inpStream);

            CUDA_SAFE_CALL(cudaGetLastError(), "Copying a section of input FTD data to the device.");
          }
        }

        FOLD // Normalise and spread on GPU  .
        {
          for ( int stack = 0; stack < batch->noStacks; stack++)  // Loop over stack
          {
            cuFfdotStack* cStack = &batch->stacks[stack];

            FOLD // Synchronisation  .
            {
              CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->inpStream, batch->iDataCpyComp, 0), "ERROR: waiting for GPU to be ready to copy data to device\n");

#ifdef TIMING
              cudaEventRecord(cStack->normInit, cStack->inpStream);
#endif
            }

            FOLD // Call the kernel to normalise and spread the input data  .
            {
              normAndSpread_f(cStack->inpStream, batch, stack );
            }

            FOLD // Synchronisation  .
            {
              cudaEventRecord(cStack->normComp, cStack->inpStream);
            }

            cudaDeviceSynchronize();
          }

          batch->flag &= ~CU_INPT_CPU_FFT;
        }
      }
      else if ( batch->flag & CU_INPT_HOST      )
      {
        // Copy chunks of FFT data and normalise and spread using the GPU

        if ( batch->noSteps > 1 ) // TODO: multi step
        {
          fprintf(stderr,"ERROR: CU_INPT_HOST has not been set up for multi-step.");
          exit(EXIT_FAILURE);
        }

        // Make sure the previous thread has complete reading from page locked memory
        CUDA_SAFE_CALL(cudaEventSynchronize(batch->iDataCpyComp), "ERROR: copying data to device");
        //nvtxRangePush("Zero");
        //memset(plains->h_iData, 0, plains->inpDataSize);
        CUDA_SAFE_CALL(cudaMemsetAsync(batch->d_iData, 0, batch->inpDataSize*batch->noSteps, batch->inpStream),"Initialising input data to 0");
        //nvtxRangePop();

        FOLD // Copy fft data to device  .
        {
          int harm = 0;
          int sz = 0;

          int step = 0; // TODO mylti-step

          // Write fft data segments to contiguous page locked memory
          for (int ss = 0; ss< batch->noStacks; ss++)
          {
            cuFfdotStack* cStack = &batch->stacks[ss];

            // Synchronisation
            CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, cStack->convComp, 0), "ERROR: waiting for GPU to be ready to copy data to device\n");

            for (int si = 0; si< cStack->noInStack; si++)
            {
              cuHarmInfo* cHInfo  = &batch->hInfos[harm];  // The current harmonic we are working on
              cuFFdot*    cPlain  = &batch->plains[harm];  //
              rVals*      rVal    = &((*batch->rInput)[step][harm]);

              /*
              drlo = calc_required_r_gpu(cHInfo->harmFrac, searchRLow[0]);
              drhi = calc_required_r_gpu(cHInfo->harmFrac, searchRHi[0]);

              binoffset = cHInfo->halfWidth;
              lobin     = (int) floor(drlo) - binoffset;
              hibin     = (int)  ceil(drhi) + binoffset;
              numdata   = hibin - lobin + 1;

              numrs     = (int) ((ceil(drhi) - floor(drlo)) * ACCEL_RDR + DBLCORRECT) + 1;
              if (harm == 0)
              {
                numrs = batch->accelLen;
              }
              else if (numrs % ACCEL_RDR)
                numrs = (numrs / ACCEL_RDR + 1) * ACCEL_RDR;
              int numtocopy = cHInfo->width - 2 * cHInfo->halfWidth * ACCEL_NUMBETWEEN;
              if (numrs < numtocopy)
                numtocopy = numrs;
               */

              //lengths.val[harm]       = rVal->numdata;
              //d_iDataList.val[harm]   = cPlain->d_iData;
              //widths.val[harm]        = cStack->width;

              int start = 0;

              if ( (rVal->lobin - batch->SrchSz->rLow)  < 0 )
              {
                // This should be unnecessary as rLow can be < 0 and h_iData is zero padded
                start = -(rVal->lobin - batch->SrchSz->rLow);
                CUDA_SAFE_CALL(cudaMemsetAsync(cPlain->d_iData, 0, start*sizeof(fcomplexcu), batch->inpStream),"Initialising input data to 0");
              }

              // Copy section to device
              CUDA_SAFE_CALL(cudaMemcpyAsync(&cPlain->d_iData[start], &batch->h_iData[rVal->lobin-batch->SrchSz->rLow+start], (rVal->numdata-start)*sizeof(fcomplexcu), cudaMemcpyHostToDevice, batch->inpStream), "Failed to copy data to device");

              sz += cStack->inpStride;

              if ( DBG_INP01 ) // Print debug info
              {
                printf("\nCPU Input Data RAW FFTs [ Half width: %i  lowbin: %i  drlo: %.2f ] \n", cHInfo->halfWidth, rVal->lobin, rVal->drlo);

                //printfData<<<1,1,0,batch->inpStream>>>((float*)cPlain->d_iData,10,1, cStack->inpStride);
                CUDA_SAFE_CALL(cudaStreamSynchronize(batch->inpStream),"");
              }

              harm++;
            }
          }

          // Synchronisation
          //cudaEventRecord(plains->iDataCpyComp, batch->inpStream);

          CUDA_SAFE_CALL(cudaGetLastError(), "Copying a section of input FTD data to the device.");
        }

        FOLD // Normalise and spread  .
        {
          // Blocks of 1024 threads ( the maximum number of threads per block )
          dimBlock.x = NAS_DIMX;
          dimBlock.y = NAS_DIMY;
          dimBlock.z = 1;

          // One block per harmonic, thus we can sort input powers in Shared memory
          dimGrid.x = batch->noHarms;
          dimGrid.y = 1;

          // Call the kernel to normalise and spread the input data
          //normAndSpreadBlks<<<dimGrid, dimBlock, (lengths.val[0]+1)*sizeof(float), batch->inpStream>>>(d_iDataList, lengths, widths);

          // Synchronisation
          cudaEventRecord(batch->normComp, batch->inpStream);

          CUDA_SAFE_CALL(cudaGetLastError(), "Calling the normalisation and spreading kernel.");
        }
      }
      else if ( batch->flag & CU_INPT_DEVICE    )
      {
        // Copy chunks of FFT data and normalise and spread using the GPU

        if ( batch->noSteps > 1 ) // TODO: multi step  .
        {
          fprintf(stderr,"ERROR: CU_INPT_DEVICE has not been set up for multi-step.");
          exit(EXIT_FAILURE);
        }

        // Make sure the previous thread has complete reading from page locked memory
        //CUDA_SAFE_CALL(cudaEventSynchronize(plains->iDataCpyComp), "ERROR: copying data to device");
        //nvtxRangePush("Zero");
        //memset(plains->h_iData, 0, plains->inpDataSize);
        //nvtxRangePop();

        FOLD // Setup parameters  .
        {
          //int harm  = 0;
          //int step  = 0; // TODO multi-step
          //int sz    = 0;

          for (int ss = 0; ss< batch->noStacks; ss++)
          {
            cuFfdotStack* cStack = &batch->stacks[ss];

            for (int si = 0; si< cStack->noInStack; si++)
            {
              //cuHarmInfo* cHInfo  = &batch->hInfos[harm];  // The current harmonic we are working on
              //cuFFdot*    cPlain  = &batch->plains[harm];     //
              //rVals*      rVal    = &((*batch->rInput)[step][harm]);

              /*
              drlo = calc_required_r_gpu(cHInfo->harmFrac, searchRLow[0]);
              drhi = calc_required_r_gpu(cHInfo->harmFrac, searchRHi[0]);

              binoffset = cHInfo->halfWidth;
              lobin = (int) floor(drlo) - binoffset;
              hibin = (int)  ceil(drhi) + binoffset;
              numdata = hibin - lobin + 1;

              numrs = (int) ((ceil(drhi) - floor(drlo)) * ACCEL_RDR + DBLCORRECT) + 1;
              if (harm == 0)
              {
                numrs = batch->accelLen;
              }
              else if (numrs % ACCEL_RDR)
                numrs = (numrs / ACCEL_RDR + 1) * ACCEL_RDR;
              int numtocopy = cHInfo->width - 2 * cHInfo->halfWidth * ACCEL_NUMBETWEEN;
              if (numrs < numtocopy)
                numtocopy = numrs;
               */

              //lengths.val[harm]     = rVal->numdata;
              //d_iDataList.val[harm] = cPlain->d_iData;
              //widths.val[harm]      = cStack->width;
              //if ( rVal->lobin-batch->SrchSz->rLow < 0 )
              {
                // NOTE could use an offset parameter here
                //printf("ERROR: Input data index out of bounds.\n");
                //exit(EXIT_FAILURE);
              }
              //d_fftList.val[harm]   = &batch->d_iData[rVal->lobin-batch->SrchSz->rLow];

              //sz += cStack->inpStride;

              //harm++;
            }
          }
        }

        FOLD // Normalise and spread  .
        {
          // Blocks of 1024 threads ( the maximum number of threads per block )
          dimBlock.x = NAS_DIMX;
          dimBlock.y = NAS_DIMY;
          dimBlock.z = 1;

          // One block per harmonic, thus we can sort input powers in Shared memory
          dimGrid.x = batch->noHarms;
          dimGrid.y = 1;

          // Synchronisation
          for (int ss = 0; ss< batch->noStacks; ss++)
          {
            cuFfdotStack* cStack = &batch->stacks[ss];
            CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, cStack->convComp, 0), "ERROR: waiting for GPU to be ready to copy data to device\n");
          }

          // Call the kernel to normalise and spread the input data
          //normAndSpreadBlksDevice<<<dimGrid, dimBlock, (lengths.val[0]+1)*sizeof(float), batch->inpStream>>>(d_fftList, d_iDataList, lengths, widths);

          // Synchronisation
          cudaEventRecord(batch->normComp, batch->inpStream);

          CUDA_SAFE_CALL(cudaGetLastError(), "Calling the normalisation and spreading kernel.");
        }
      }
      else if ( batch->flag & CU_INPT_SINGLE_C  )
      {
        // Copy chunks of FFT data and normalise and spread using the CPU

        // Make sure the previous thread has complete reading from page locked memory
        CUDA_SAFE_CALL(cudaEventSynchronize(batch->iDataCpyComp), "ERROR: Synchronising before writing input data to page locked host memory.");

        nvtxRangePush("Zero");
        memset(batch->h_iData, 0, batch->inpDataSize*batch->noSteps);
        nvtxRangePop();

        CPU_Norm_Spread(batch, searchRLow, searchRHi, norm_type, fft);

        FOLD // CPU FFT  .
        {
          if ( batch->flag & CU_INPT_CPU_FFT )
          {
#ifdef STPMSG
          printf("\t\tCPU FFT Input\n");
#endif

#pragma omp critical
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
        }

        FOLD // Synchronisation  .
        {
          // Wait for per stack convolutions to finish
          for (int ss = 0; ss< batch->noStacks; ss++)
          {
            cuFfdotStack* cStack = &batch->stacks[ss];
            CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, cStack->convComp, 0), "ERROR: waiting for GPU to be ready to copy data to device\n");
          }

          // Wait for batch convolution to finish
          CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, batch->convComp, 0), "ERROR: waiting for GPU to be ready to copy data to device\n");

#ifdef TIMING
          cudaEventRecord(batch->iDataCpyInit, batch->inpStream);
#endif

        }

        FOLD // Copy to device  .
        {
#ifdef STPMSG
          printf("\t\tCopy to device\n");
#endif
          CUDA_SAFE_CALL(cudaMemcpyAsync(batch->d_iData, batch->h_iData, batch->inpDataSize*batch->noSteps, cudaMemcpyHostToDevice, batch->inpStream), "Failed to copy input data to device");
          CUDA_SAFE_CALL(cudaGetLastError(), "Error preparing the input data.");
        }

        FOLD // Synchronisation  .
        {
          cudaEventRecord(batch->normComp, batch->inpStream);
          cudaEventRecord(batch->iDataCpyComp, batch->inpStream);

          if ( batch->flag & CU_INPT_CPU_FFT )
          {
            for (int ss = 0; ss < batch->noStacks; ss++)
            {
              cuFfdotStack* cStack = &batch->stacks[ss];
              cudaEventRecord(cStack->prepComp, batch->inpStream);
            }
          }
        }
      }
    }

    if ( DBG_INP03 ) // Print debug info  .
    {
      for (int ss = 0; ss< batch->noHarms && true; ss++)
      {
        cuFFdot* cPlain     = &batch->plains[ss];
        printf("\nGPU Input Data pre FFT h:%i   f: %f\n",ss,cPlain->harmInf->harmFrac);
        //printfData<<<1,1,0,0>>>((float*)cPlain->d_iData,10,1, cPlain->harmInf->inpStride);
        CUDA_SAFE_CALL(cudaStreamSynchronize(0),"");
        for (int ss = 0; ss< batch->noStacks; ss++)
        {
          cuFfdotStack* cStack = &batch->stacks[ss];
          CUDA_SAFE_CALL(cudaStreamSynchronize(cStack->fftIStream),"");
        }
      }
    }

    FOLD // fft the input on the GPU data  .
    {
      if ( !(batch->flag & CU_INPT_CPU_FFT) )
      {
#ifdef STPMSG
          printf("\t\tGPU FFT\n");
#endif

#ifdef SYNCHRONOUS
        cuFfdotStack* pStack = NULL;
#endif

        for (int ss = 0; ss < batch->noStacks; ss++)
        {
          cuFfdotStack* cStack = &batch->stacks[ss];

          CUDA_SAFE_CALL(cudaGetLastError(), "Error before input fft.");

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
#endif
          }

          FOLD // Do the FFT  .
          {
#pragma omp critical
            {

#ifdef TIMING
                cudaEventRecord(cStack->inpFFTinit, cStack->fftIStream);
#endif

              CUFFT_SAFE_CALL(cufftSetStream(cStack->inpPlan, cStack->fftIStream),"Failed associating a CUFFT plan with FFT input stream\n");
              CUFFT_SAFE_CALL(cufftExecC2C(cStack->inpPlan, (cufftComplex *) cStack->d_iData, (cufftComplex *) cStack->d_iData, CUFFT_FORWARD),"Failed to execute input CUFFT plan.");

              CUDA_SAFE_CALL(cudaGetLastError(), "Error FFT'ing the input data.");

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
        CUDA_SAFE_CALL(cudaGetLastError(), "Error FFT'ing the input data.");
      }
    }

    if ( DBG_INP04 ) // Print debug info  .
    {
      for (int ss = 0; ss< batch->noHarms && true; ss++)
      {
        cuFFdot* cPlain     = &batch->plains[ss];
        printf("\nGPU Input Data post FFT h:%i   f: %f\n",ss,cPlain->harmInf->harmFrac);
        //printfData<<<1,1,0,0>>>((float*)cPlain->d_iData,10,1, cPlain->harmInf->inpStride);
        CUDA_SAFE_CALL(cudaStreamSynchronize(0),"");
        for (int ss = 0; ss< batch->noStacks; ss++)
        {
          cuFfdotStack* cStack = &batch->stacks[ss];
          CUDA_SAFE_CALL(cudaStreamSynchronize(cStack->fftIStream),"");
        }
      }
    }

    batch->haveInput = 1;

    nvtxRangePop();
  }
}
