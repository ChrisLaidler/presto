#include "cuda_accel_CV.h"

__device__ cufftCallbackLoadC d_loadCallbackPtr     = CB_ConvolveInput;
__device__ cufftCallbackStoreC d_storeCallbackPtr   = CB_PowerOut;

__device__ cufftComplex CB_ConvolveInput( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr)
{
  stackInfo *inf  = (stackInfo*)callerInfo;

  int fIdx        = inf->famIdx;
  int noSteps     = inf->noSteps;
  int noPlains    = inf->noPlains;
  int stackStrd   = STRIDE_FAM_ORDER[fIdx];
  int width       = WIDTH_FAM_ORDER[fIdx];

  int strd        = stackStrd * noSteps ;                 /// Stride taking into acount steps)
  int gRow        = offset / strd;                        /// Row (ignoring steps)
  int col         = offset % stackStrd;                   /// 2D column
  int top         = 0;                                    /// The top of the plain
  int pHeight     = 0;
  int pln         = 0;

  for ( int i = 0; i < noPlains; i++ )
  {
    top += HEIGHT_FAM_ORDER[fIdx+i];

    if ( gRow >= top )
    {
      pln         = i+1;
      pHeight     = top;
    }
  }

  int row         = offset / stackStrd - pHeight*noSteps;
  int pIdx        = fIdx + pln;
  int plnHeight   = HEIGHT_FAM_ORDER[pIdx];
  int step;

  if ( inf->flag & FLAG_ITLV_PLN )
  {
    step = row / plnHeight;
    row  = row % plnHeight;
  }
  else
  {
    step  = row % noSteps;
    row   = row / noSteps;
  }

  cufftComplex ker = ((cufftComplex*)(KERNEL_FAM_ORDER[pIdx]))[row*stackStrd + col];      //
  cufftComplex inp = ((cufftComplex*)inf->d_iData)[(pln*noSteps+step)*stackStrd + col];   //

  // Do the convolution
  cufftComplex out;
  out.x = ( inp.x * ker.x + inp.y * ker.y ) / (float)width;
  out.y = ( inp.y * ker.x - inp.x * ker.y ) / (float)width;

  return out;
}

__device__ void CB_PowerOut( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  // Calculate power
  float power = element.x*element.x + element.y*element.y ;

  // Write result (offsets are the same)
  ((float*)callerInfo)[offset] = power;
}

void copyCUFFT_LD_CB(cuFFdotBatch* batch)
{
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_ldCallbackPtr, d_loadCallbackPtr,  sizeof(cufftCallbackLoadC)),   "");
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_storeCallbackPtr, sizeof(cufftCallbackStoreC)),  "");
}

/** Convolve and inverse FFT the complex f-∂f plain using FFT callback
 * @param batch
 */
void convolveBatchCUFFT(cuFFdotBatch* batch )
{
#ifdef SYNCHRONOUS
  cuFfdotStack* pStack = NULL;  // Previous stack
#endif

  // Convolve this entire stack in one block
  for (int ss = 0; ss< batch->noStacks; ss++)
  {
    cuFfdotStack* cStack = &batch->stacks[ss];

    FOLD // Synchronisation  .
    {
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack->prepComp,0),   "Waiting for GPU to be ready to copy data to device.");  // Need input data
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, batch->searchComp, 0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plain so search must be compete

#ifdef SYNCHRONOUS
      // Wait for all the input FFT's to complete
      for (int ss = 0; ss< batch->noStacks; ss++)
      {
        cuFfdotStack* cStack2 = &batch->stacks[ss];
        cudaStreamWaitEvent(cStack->fftPStream, cStack2->prepComp, 0);
      }

      // Wait for the previous convolution to complete
      if ( pStack != NULL )
        cudaStreamWaitEvent(cStack->fftPStream, pStack->plnComp, 0);
#endif
    }

    FOLD // Do the FFT  .
    {
#pragma omp critical
      FOLD
      {
        FOLD // Timing  .
        {
#ifdef TIMING
          cudaEventRecord(cStack->invFFTinit, cStack->fftPStream);
#endif
        }

        FOLD // Set store FFT callback  .
        {
          if ( batch->flag & FLAG_CNV_CB_OUT )
          {
            CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&cStack->d_plainPowers ),"");
          }
        }

        FOLD // Set load FFT callback  .
        {
          CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_ldCallbackPtr, CUFFT_CB_LD_COMPLEX, (void**)&cStack->d_sInf ),"");
        }

        CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with cnvlStream.");
        CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) cStack->d_plainData, (cufftComplex *) cStack->d_plainData, CUFFT_INVERSE),"Error executing CUFFT plan.");
      }
    }

    FOLD // Synchronisation  .
    {
      cudaEventRecord(cStack->plnComp, cStack->fftPStream);

#ifdef SYNCHRONOUS
      pStack = cStack;
#endif
    }
  }
}

/** Convolve and inverse FFT the complex f-∂f plain
 * This assumes the input data is ready and on the device
 * This creates a complex f-∂f plain
 */
void convolveBatch(cuFFdotBatch* batch)
{
  //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  if ( batch->haveInput )
  {
    nvtxRangePush("Convolve & FFT");
#ifdef STPMSG
    printf("\tConvolve & FFT\n");
#endif

    dim3 dimBlock, dimGrid;

    if ( batch->flag & FLAG_CNV_CB_IN )  	// Do the convolution using a CUFFT callback  .
    {
#ifdef STPMSG
      printf("\t\tConvolve with CUFFT\n");
#endif
      convolveBatchCUFFT( batch );
    }
    else                                    // Do the convolution and FFT separately  .
    {
      FOLD // Convolve  .
      {
#ifdef STPMSG
        printf("\t\tConvolve\n");
#endif

        // In my testing I found convolving each plain separately works fastest so it is the "default"
        if      ( batch->flag & FLAG_CNV_BATCH ) 	// Do the convolutions one family at a time  .
        {
          FOLD // Synchronisation  .
          {
            for (int ss = 0; ss < batch->noStacks; ss++) // Synchronise input data preparation for all stacks
            {
              cuFfdotStack* cStack = &batch->stacks[ss];
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->convStream, cStack->prepComp,0),     "Waiting for GPU to be ready to copy data to device.");    // Need input data
            }

            CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->convStream, batch->searchComp, 0),      "Waiting for GPU to be ready to copy data to device.");   // This will overwrite the f-fdot plain so search must be compete
          }

          FOLD // Call kernel  .
          {
#ifdef TIMING // Timing event  .
            CUDA_SAFE_CALL(cudaEventRecord(batch->convInit, batch->convStream),"Recording event: convInit");
#endif

            convolveffdot50_f(batch->convStream, batch);

            // Run message
            CUDA_SAFE_CALL(cudaGetLastError(), "Error at kernel launch");
          }

          FOLD // Synchronisation  .
          {
            CUDA_SAFE_CALL(cudaEventRecord(batch->convComp, batch->convStream),"Recording event: convComp");
          }
        }
        else if ( batch->flag & FLAG_CNV_STK ) 	  // Do the convolutions one stack  at a time  .
        {
#ifdef SYNCHRONOUS
          cuFfdotStack* pStack = NULL;  // Previous stack
#endif

          // Convolve this entire stack in one block
          for (int ss = 0; ss < batch->noStacks; ss++)
          {
            cuFfdotStack* cStack = &batch->stacks[ss];

            FOLD // Synchronisation  .
            {
              CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->cnvlStream, cStack->prepComp,0),    "Waiting for GPU to be ready to copy data to device.");  // Need input data
              CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->cnvlStream, batch->searchComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plain so search must be compete

#ifdef SYNCHRONOUS
              // Wait for all the input FFT's to complete
              for (int ss = 0; ss< batch->noStacks; ss++)
              {
                cuFfdotStack* cStack2 = &batch->stacks[ss];
                cudaStreamWaitEvent(cStack->cnvlStream, cStack2->prepComp, 0);
              }

              // Wait for the previous convolution to complete
              if ( pStack != NULL )
                cudaStreamWaitEvent(cStack->cnvlStream, pStack->convComp, 0);
#endif
            }

            FOLD // Timing event  .
            {
#ifdef TIMING
              CUDA_SAFE_CALL(cudaEventRecord(cStack->convInit, cStack->cnvlStream),"Recording event: convInit");
#endif
            }

            FOLD // Call kernel(s)  .
            {
              if      ( batch->flag & FLAG_CNV_00 )
              {
                convolveffdot00_f(cStack->cnvlStream, batch, ss);
              }
              else if ( batch->flag & FLAG_CNV_10 )
              {
                convolveffdot10_f(cStack->cnvlStream, batch, ss);
              }
              else if ( batch->flag & FLAG_CNV_41 )
              {
                convolveffdot41_f(cStack->cnvlStream, batch, ss);
              }
              else if ( batch->flag & FLAG_CNV_42 )
              {
                convolveffdot42_f(cStack->cnvlStream, batch, ss);
              }
              else if ( batch->flag & FLAG_CNV_43 )
              {
                convolveffdot43_f(cStack->cnvlStream, batch, ss);
              }
              else
              {
                fprintf(stderr,"ERROR: No valid convolve specified. Line %i in %s.\n", __LINE__, __FILE__);
                exit(EXIT_FAILURE);
              }

              // Run message
              CUDA_SAFE_CALL(cudaGetLastError(), "Error at kernel launch (convolveffdot7)");
            }

            FOLD // Synchronisation  .
            {
              cudaEventRecord(cStack->convComp, cStack->cnvlStream);

#ifdef SYNCHRONOUS
              pStack = cStack;
#endif
            }
          }
        }
        else if ( batch->flag & FLAG_CNV_PLN ) 	  // Do the convolutions one plain  at a time  .
        {
#ifdef SYNCHRONOUS
          cuFfdotStack* pStack = NULL;  // Previous stack
#endif

          dimBlock.x = CNV_DIMX;
          dimBlock.y = CNV_DIMY;

          //for (int ss = plains->noStacks-1; ss >= 0; ss-- )
          for (int stack = 0; stack < batch->noStacks; stack++)              // Loop through Stacks
          {
            cuFfdotStack* cStack = &batch->stacks[stack];
            fcomplexcu* d_plainData;    // The complex f-∂f plain data
            fcomplexcu* d_iData;        // The complex input array

            FOLD // Synchronisation  .
            {
              CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->cnvlStream, cStack->prepComp,0),   "Waiting for GPU to be ready to copy data to device.");  // Need input data
              CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->cnvlStream, batch->searchComp, 0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plain so search must be compete

#ifdef SYNCHRONOUS
              // Wait for all the input FFT's to complete
              for (int ss = 0; ss< batch->noStacks; ss++)
              {
                cuFfdotStack* cStack2 = &batch->stacks[ss];
                cudaStreamWaitEvent(cStack->cnvlStream, cStack2->prepComp, 0);
              }

              // Wait for the previous convolution to complete
              if ( pStack != NULL )
                cudaStreamWaitEvent(cStack->cnvlStream, pStack->convComp, 0);
#endif
            }

            FOLD // Timing event  .
            {
#ifdef TIMING
              CUDA_SAFE_CALL(cudaEventRecord(cStack->convInit, cStack->cnvlStream),"Recording event: convInit");
#endif
            }

            FOLD // call kernel(s)  .
            {
              for (int plain = 0; plain < cStack->noInStack; plain++)         // Loop through plains in stack
              {
                cuHarmInfo* cHInfo    = &cStack->harmInf[plain];       // The current harmonic we are working on
                cuFFdot*    cPlain    = &cStack->plains[plain];        // The current f-∂f plain

                dimGrid.x = ceil(cHInfo->width / (float) ( CNV_DIMX * CNV_DIMY ));
                dimGrid.y = 1;

                for (int step = 0; step < batch->noSteps; step++)       // Loop through Steps
                {
                  d_iData         = cPlain->d_iData + cStack->strideCmplx * step;

                  if      ( batch->flag & FLAG_ITLV_ROW )
                  {
                    fprintf(stderr,"ERROR: Cannot do single plain convolutions with row-interleaved multi step stacks.\n");
                    exit(EXIT_FAILURE);
                  }
                  else if ( batch->flag & FLAG_ITLV_PLN )
                    d_plainData   = cPlain->d_plainData + step * cHInfo->height * cStack->strideCmplx;   // Shift by plain height
                  else
                    d_plainData   = cPlain->d_plainData;  // If nothing is specified just use plain data

                  if ( batch->flag & FLAG_CNV_TEX )
                    convolveffdot36<<<dimGrid, dimBlock, 0, cStack->cnvlStream>>>(d_plainData, cHInfo->width, cStack->strideCmplx, cHInfo->height, d_iData, cPlain->kernel->kerDatTex);
                  else
                    convolveffdot31<<<dimGrid, dimBlock, 0, cStack->cnvlStream>>>(d_plainData, cHInfo->width, cStack->strideCmplx, cHInfo->height, d_iData, cPlain->kernel->d_kerData);

                  // Run message
                  CUDA_SAFE_CALL(cudaGetLastError(), "Error at convolution kernel launch");
                }
              }
            }

            FOLD // Synchronisation  .
            {
              cudaEventRecord(cStack->convComp, cStack->cnvlStream);

#ifdef SYNCHRONOUS
              pStack = cStack;
#endif
            }
          }

        }
        else
        {
          fprintf(stderr, "ERROR: convolveBatch not templated for this type of convolution.\n");
        }
      }

      FOLD // Inverse FFT the f-∂f plain  .
      {

#ifdef STPMSG
        printf("\t\tInverse FFT\n");
#endif

#ifdef SYNCHRONOUS
        cuFfdotStack* pStack = NULL;  // Previous stack
#endif

        // Copy fft data to device
        //for (int ss = plains->noStacks-1; ss >= 0; ss-- )
        for (int ss = 0; ss< batch->noStacks; ss++)
        {
          cuFfdotStack* cStack = &batch->stacks[ss];

          FOLD // Synchronisation  .
          {
            cudaStreamWaitEvent(cStack->fftPStream, cStack->convComp, 0);
            cudaStreamWaitEvent(cStack->fftPStream, batch->convComp,  0);

#ifdef SYNCHRONOUS
            // Wait for all the convolutions to complete
            for (int ss = 0; ss< batch->noStacks; ss++)
            {
              cuFfdotStack* cStack2 = &batch->stacks[ss];
              cudaStreamWaitEvent(cStack->fftPStream, cStack2->convComp, 0);
            }

            // Wait for the previous fft to complete
            if ( pStack != NULL )
              cudaStreamWaitEvent(cStack->fftPStream, pStack->plnComp, 0);
#endif
          }

          FOLD // Call the inverse CUFFT  .
          {
#pragma omp critical
            {
              FOLD // Timing  .
              {
#ifdef TIMING
                cudaEventRecord(cStack->invFFTinit, cStack->fftPStream);
#endif
              }

              if ( batch->flag & FLAG_CNV_CB_OUT ) // Set the CUFFT callback to calculate and store powers  .
              {
                CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&cStack->d_plainPowers ),"");
              }

              CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with cnvlStream.");
              CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) cStack->d_plainData, (cufftComplex *) cStack->d_plainData, CUFFT_INVERSE),"Error executing CUFFT plan.");

              FOLD // Synchronisation  .
              {
                cudaEventRecord(cStack->plnComp, cStack->fftPStream);

#ifdef SYNCHRONOUS
                pStack = cStack;
#endif
              }
            }
          }
        }
      }
    }

    batch->haveInput    = 0;
    batch->haveConvData = 1;

    nvtxRangePop();
  }

  // Set the r-values and width for the next iteration when we will be doing the actual Add and Search
  cycleRlists(batch);
}

