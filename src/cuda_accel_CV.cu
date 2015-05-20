#include "cuda_accel_CV.h"

__device__ cufftCallbackLoadC d_loadCallbackPtr     = CB_ConvolveInput;
__device__ cufftCallbackStoreC d_storeCallbackPtr   = CB_PowerOut;

__device__ __constant__ int           HEIGHT_FAM_ORDER[MAX_HARM_NO];        ///< Plain height in stage order
__device__ __constant__ int           STRIDE_FAM_ORDER[MAX_HARM_NO];        ///< Plain stride in stage order
__device__ __constant__ fcomplexcu*   KERNEL_FAM_ORDER[MAX_HARM_NO];        ///< Kernel pointer in stage order

__device__ cufftComplex CB_ConvolveInput( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr)
{

  fftCnvlvInfo *inf = (fftCnvlvInfo*)callerInfo;

  const int strd = inf->stride * inf->noSteps;

  size_t grow = offset / strd;
  size_t col  = offset % inf->stride;
  size_t step = ( offset % strd ) / inf->stride ;
  size_t pln  = 0;

  for ( int i = 0; i < inf->noPlains; i++ )
  {
    if ( grow >= inf->top[i] )
    {
      pln = i;
    }
  }

  size_t row  = grow - inf->top[pln];

  cufftComplex ker = ((cufftComplex*)inf->d_kernel[pln])[row*inf->stride + col ];
  cufftComplex inp = ((cufftComplex*)inf->d_idata[pln])[step*inf->stride + col ];

  cufftComplex out;
  out.x = ( inp.x * ker.x + inp.y * ker.y ) / inf->width;
  out.y = ( inp.y * ker.x - inp.x * ker.y ) / inf->width;

  return out;

  //return ((cufftComplex*)dataIn)[offset];
}

__device__ void CB_PowerOut( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  float power = element.x*element.x + element.y*element.y ;
  ((float*)callerInfo)[offset] = power;
}

void copyCUFFT_LD_CB(cuFFdotBatch* batch)
{
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_ldCallbackPtr,  d_loadCallbackPtr,  sizeof(cufftCallbackLoadC)),   "");
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_storeCallbackPtr, sizeof(cufftCallbackStoreC)),  "");
}

int setConstVals_Fam_Order( cuFFdotBatch* batch )
{
  FOLD // Set other constant values
  {
    void *dcoeffs;

    int           height[MAX_HARM_NO];
    int           stride[MAX_HARM_NO];
    fcomplexcu*   kerPnt[MAX_HARM_NO];
    for (int i = 0; i < batch->noHarms; i++)
    {
      height[i] = batch->hInfos[i].height;
      stride[i] = batch->hInfos[i].width;
      kerPnt[i] = batch->kernels[i].d_kerData;

      if (batch->hInfos[i].width != batch->hInfos[i].inpStride )
      {
        fprintf(stderr,"ERROR: Width is not the same as stride, using width this may case errors in the convolution.\n");
      }
    }

    for (int i = batch->noHarms; i < MAX_HARM_NO; i++) // Zero the rest
    {
      height[i] = 0;
      stride[i] = 0;
    }

    cudaGetSymbolAddress((void **)&dcoeffs, HEIGHT_FAM_ORDER);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &height, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, STRIDE_FAM_ORDER);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &stride, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, KERNEL_FAM_ORDER);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &kerPnt, MAX_HARM_NO * sizeof(fcomplexcu*), cudaMemcpyHostToDevice),      "Copying stages to device");
  }

  CUDA_SAFE_CALL(cudaGetLastError(), "Error preparing the constant memory values for the convolutions.");

  return 1;
}

/** Convolve and inverse FFT the complex f-∂f plain using FFT callback
 * @param plains
 */
void convolveBatchCUFFT(cuFFdotBatch* batch )
{
  // Convolve this entire stack in one block
  for (int ss = 0; ss< batch->noStacks; ss++)
  {
    cuFfdotStack* cStack = &batch->stacks[ss];

    // Synchronisation
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack->prepComp, 0),    "Waiting for GPU to be ready to copy data to device.");  // Need input data
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, batch->searchComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plain so search must be compete

    // Do the FFT
#pragma omp critical
    FOLD
    {
      if ( batch->flag & FLAG_CNV_CB_OUT )
      {
        //CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&h_storeCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&cStack->d_plainPowers ),"");
      }

      CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with cnvlStream.");
      CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) cStack->d_plainData, (cufftComplex *) cStack->d_plainData, CUFFT_INVERSE),"Error executing CUFFT plan.");
    }

    // Synchronise
    cudaEventRecord(cStack->plnComp, cStack->fftPStream);
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
        if      ( batch->flag & FLAG_CNV_BATCH ) // Do the convolutions one family at a time  .
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
        else if ( batch->flag & FLAG_CNV_STK ) // Do the convolutions one stack  at a time  .
        {
#ifdef SYNCHRONOUS
          cuFfdotStack* pStack = NULL;
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
                fprintf(stderr,"ERROR: No valid convolve specifyed. Line %i in %s.\n", __LINE__, __FILE__);
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
        else if ( batch->flag & FLAG_CNV_PLN ) // Do the convolutions one plain  at a time  .
        {
#ifdef SYNCHRONOUS
          cuFfdotStack* pStack = NULL;
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
        cuFfdotStack* pStack = NULL;
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
                //cufftCallbackLoadC hostCopyOfCallbackPtr;
                //CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &hostCopyOfCallbackPtr, d_storeCallbackPtr, sizeof(hostCopyOfCallbackPtr)),  "");
                //CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&hostCopyOfCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&cStack->d_plainPowers ),"");

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


