#include "cuda_accel_MU.h"

__device__ cufftCallbackLoadC d_loadCallbackPtr     = CB_MultiplyInput;
__device__ cufftCallbackStoreC d_storeCallbackPtr   = CB_PowerOut;

__device__ cufftComplex CB_MultiplyInput( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr)
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

  // Do the multiplication
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

/** Multiply and inverse FFT the complex f-∂f plain using FFT callback
 * @param batch
 */
void multiplyBatchCUFFT(cuFFdotBatch* batch )
{
#ifdef SYNCHRONOUS
  cuFfdotStack* pStack = NULL;  // Previous stack
#endif

  // Multiply this entire stack in one block
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

      // Wait for the previous multiplication to complete
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
          if ( batch->flag & FLAG_MUL_CB_OUT )
          {
            CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&cStack->d_plainPowers ),"");
          }
        }

        FOLD // Set load FFT callback  .
        {
          CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_ldCallbackPtr, CUFFT_CB_LD_COMPLEX, (void**)&cStack->d_sInf ),"");
        }

        CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with multStream.");
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

/** Multiply and inverse FFT the complex f-∂f plain
 * This assumes the input data is ready and on the device
 * This creates a complex f-∂f plain
 */
void multiplyBatch(cuFFdotBatch* batch)
{
  //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  if ( batch->haveInput )
  {
    nvtxRangePush("Multiply & FFT");
#ifdef STPMSG
    printf("\tMultiply & FFT\n");
#endif

    dim3 dimBlock, dimGrid;

    if ( batch->flag & FLAG_MUL_CB_IN )  		// Do the multiplication using a CUFFT callback  .
    {
#ifdef STPMSG
      printf("\t\tMultiply with CUFFT\n");
#endif
      multiplyBatchCUFFT( batch );
    }
    else                                    // Do the multiplication and FFT separately  .
    {
      FOLD // Multiply  .
      {
#ifdef STPMSG
        printf("\t\tMultiply\n");
#endif

        // In my testing I found multiplying each plain separately works fastest so it is the "default"
        if      ( batch->flag & FLAG_MUL_BATCH ) 	// Do the multiplications one family at a time  .
        {
          FOLD // Synchronisation  .
          {
            for (int ss = 0; ss < batch->noStacks; ss++) // Synchronise input data preparation for all stacks
            {
              cuFfdotStack* cStack = &batch->stacks[ss];
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, cStack->prepComp,0),     "Waiting for GPU to be ready to copy data to device.");    // Need input data
            }

            CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, batch->searchComp, 0),      "Waiting for GPU to be ready to copy data to device.");   // This will overwrite the f-fdot plain so search must be compete
          }

          FOLD // Call kernel  .
          {
#ifdef TIMING // Timing event  .
            CUDA_SAFE_CALL(cudaEventRecord(batch->multInit, batch->multStream),"Recording event: multInit");
#endif

            mult50_f(batch->multStream, batch);

            // Run message
            CUDA_SAFE_CALL(cudaGetLastError(), "Error at kernel launch");
          }

          FOLD // Synchronisation  .
          {
            CUDA_SAFE_CALL(cudaEventRecord(batch->multComp, batch->multStream),"Recording event: multComp");
          }
        }
        else if ( batch->flag & FLAG_MUL_STK ) 	  // Do the multiplications one stack  at a time  .
        {
#ifdef SYNCHRONOUS
          cuFfdotStack* pStack = NULL;  // Previous stack
#endif

          // Multiply this entire stack in one block
          for (int ss = 0; ss < batch->noStacks; ss++)
          {
            cuFfdotStack* cStack = &batch->stacks[ss];

            FOLD // Synchronisation  .
            {
              CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->prepComp,0),    "Waiting for GPU to be ready to copy data to device.");  // Need input data
              CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, batch->searchComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plain so search must be compete

#ifdef SYNCHRONOUS
              // Wait for all the input FFT's to complete
              for (int ss = 0; ss< batch->noStacks; ss++)
              {
                cuFfdotStack* cStack2 = &batch->stacks[ss];
                cudaStreamWaitEvent(cStack->multStream, cStack2->prepComp, 0);
              }

              // Wait for the previous multiplication to complete
              if ( pStack != NULL )
                cudaStreamWaitEvent(cStack->multStream, pStack->multComp, 0);
#endif
            }

            FOLD // Timing event  .
            {
#ifdef TIMING
              CUDA_SAFE_CALL(cudaEventRecord(cStack->multInit, cStack->multStream),"Recording event: multInit");
#endif
            }

            FOLD // Call kernel(s)  .
            {
              if      ( batch->flag & FLAG_MUL_00 )
              {
                mult00_f(cStack->multStream, batch, ss);
              }
              else if ( batch->flag & FLAG_MUL_10 )
              {
                mult10_f(cStack->multStream, batch, ss);
              }
              else if ( batch->flag & FLAG_MUL_20 )
              {
                mult20_f(cStack->multStream, batch, ss);
              }
              else if ( batch->flag & FLAG_MUL_30 )
              {
                mult30_f(cStack->multStream, batch, ss);
              }
              /*
              else if ( batch->flag & FLAG_MUL_41 )
              {
                mult41_f(cStack->multStream, batch, ss);
              }
              else if ( batch->flag & FLAG_MUL_42 )
              {
                mult42_f(cStack->multStream, batch, ss);
              }
              else if ( batch->flag & FLAG_MUL_43 )
              {
                mult43_f(cStack->multStream, batch, ss);
              }
              */
              else
              {
                fprintf(stderr,"ERROR: No valid multiply specified. Line %i in %s.\n", __LINE__, __FILE__);
                exit(EXIT_FAILURE);
              }

              // Run message
              CUDA_SAFE_CALL(cudaGetLastError(), "Error at kernel launch (mult7)");
            }

            FOLD // Synchronisation  .
            {
              cudaEventRecord(cStack->multComp, cStack->multStream);

#ifdef SYNCHRONOUS
              pStack = cStack;
#endif
            }
          }
        }
        else if ( batch->flag & FLAG_MUL_PLN ) 	  // Do the multiplications one plain  at a time  .
        {
          mult40(batch);
        }
        else
        {
          fprintf(stderr, "ERROR: multiplyBatch not templated for this type of multiplication.\n");
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
            cudaStreamWaitEvent(cStack->fftPStream, cStack->multComp, 0);
            cudaStreamWaitEvent(cStack->fftPStream, batch->multComp,  0);

#ifdef SYNCHRONOUS
            // Wait for all the multiplications to complete
            for (int ss = 0; ss< batch->noStacks; ss++)
            {
              cuFfdotStack* cStack2 = &batch->stacks[ss];
              cudaStreamWaitEvent(cStack->fftPStream, cStack2->multComp, 0);
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

              if ( batch->flag & FLAG_MUL_CB_OUT ) // Set the CUFFT callback to calculate and store powers  .
              {
                CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&cStack->d_plainPowers ),"");
              }

              CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with multStream.");
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

