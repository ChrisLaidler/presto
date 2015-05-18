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
      if ( batch->flag & FLAG_CUFFTCB_OUT )
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

    if ( batch->flag & FLAG_CUFFTCB_INP )  	// Do the convolution using a CUFFT callback  .
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

        dimBlock.x = CNV_DIMX;   // in my experience 16 is almost always best (half warp)
        dimBlock.y = CNV_DIMY;   // in my experience 16 is almost always best (half warp)

        // In my testing I found convolving each plain separately works fastest so it is the "default"
        if      ( batch->flag & FLAG_CNV_FAM ) // Do the convolutions one family at a time  .
        {
          dimGrid.x = ceil(batch->hInfos[0].width / (float) ( CNV_DIMX * CNV_DIMY ));
          dimGrid.y = 1;

          FOLD // Synchronisation  .
          {
            for (int ss = 0; ss < batch->noStacks; ss++) // Synchronise input data preparation for all stacks
            {
              cuFfdotStack* cStack = &batch->stacks[ss];
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->convStream, cStack->prepComp,0),     "Waiting for GPU to be ready to copy data to device.");    // Need input data
            }

            CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->convStream, batch->searchComp, 0),      "Waiting for GPU to be ready to copy data to device.");   // This will overwrite the f-fdot plain so search must be compete
          }

          FOLD // Timing event  .
          {
#ifdef TIMING
          CUDA_SAFE_CALL(cudaEventRecord(batch->convInit, batch->convStream),"Recording event: convInit");
#endif
          }

          FOLD // call kernel  .
          {
            convolveffdot5_f(dimGrid, dimBlock, 0, batch->convStream, batch);

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

            iHarmList hlist;
            cHarmList plainsDat;
            cHarmList kerDat;
            iHarmList zUp;
            iHarmList zDn;

            for (int i = 0; i < cStack->noInStack; i++)     // Loop over plains to determine where they start
            {
              hlist.val[i]      =  cStack->harmInf[i].height;
              plainsDat.val[i]  =  cStack->plains[i].d_plainData;
              kerDat.val[i]     =  cStack->kernels[i].d_kerData;

              zUp.val[i]        =  cStack->zUp[i];
              zDn.val[i]        =  cStack->zDn[i];
            }

            dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY ));
            dimGrid.y = 1;

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
              if ( batch->flag & FLAG_RAND_2 )
              {
                convolveffdot00_f(dimGrid, dimBlock, 0, cStack->cnvlStream, batch, ss);
              }
              else
              {
                if ( batch->flag & FLAG_CNV_OVLP )
                {
                  // NOTE: convolveffdot41 seams faster and has been adapted for multi-step
                  convolveffdot71_f(dimGrid, dimBlock, 0, cStack->cnvlStream, cStack->d_kerData, cStack->d_iData, plainsDat, cStack->width, cStack->inpStride, hlist, cStack->height, cStack->kerDatTex, zUp, zDn, batch->noSteps, cStack->noInStack, batch->flag );
                  //convolveffdot72_f(dimGrid, dimBlock, 0, cStack->cnvlStream, batch, ss);
                }
                else
                {
                  if( batch->flag & FLAG_RAND_1 )
                  {
                    convolveffdot43_f(dimGrid, dimBlock, 0, cStack->cnvlStream, batch, ss);
                  }
                  else
                  {
                    convolveffdot41_f(dimGrid, dimBlock, 0, cStack->cnvlStream, cStack->d_kerData, cStack->d_iData, cStack->d_plainData, cStack->width, cStack->inpStride, hlist, cStack->height, kerDat, cStack->kerDatTex, batch->noSteps, cStack->noInStack, batch->flag );
                  }
                }
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
          // NOTE: The use of FLAG_CNV_1KER in this section will be handled because we are using the "kernels" pointers to the complex data
#ifdef SYNCHRONOUS
      cuFfdotStack* pStack = NULL;
#endif

          //for (int ss = plains->noStacks-1; ss >= 0; ss-- )
          for (int ss = 0; ss< batch->noStacks; ss++)              // Loop through Stacks
          {
            cuFfdotStack* cStack = &batch->stacks[ss];
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
              for (int si = 0; si< cStack->noInStack; si++)         // Loop through plains in stack
              {
                cuHarmInfo* cHInfo    = &cStack->harmInf[si];       // The current harmonic we are working on
                cuFFdot*    cPlain    = &cStack->plains[si];        // The current f-∂f plain

                dimGrid.x = ceil(cHInfo->width / (float) ( CNV_DIMX * CNV_DIMY ));
                dimGrid.y = 1;

                for (int sti = 0; sti < batch->noSteps; sti++)       // Loop through Steps
                {
                  d_iData       = cPlain->d_iData + cHInfo->inpStride * sti;

                  if      ( batch->flag & FLAG_STP_ROW )
                  {
                    fprintf(stderr,"ERROR: Cannot do single plain convolutions with row interleave multi step stacks.\n");
                    exit(EXIT_FAILURE);
                  }
                  else if ( batch->flag & FLAG_STP_PLN )
                    d_plainData = cPlain->d_plainData + sti * cHInfo->height * cHInfo->inpStride;   // Shift by plain height
                  else if ( batch->flag & FLAG_STP_STK )
                    d_plainData = cPlain->d_plainData + sti * cStack->height * cHInfo->inpStride;   // Shift by stack height
                  else
                    d_plainData   = cPlain->d_plainData;  // If nothing is specified just use plain data

                  if ( batch->flag & FLAG_CNV_TEX )
                    convolveffdot36<<<dimGrid, dimBlock, 0, cStack->cnvlStream>>>(d_plainData, cHInfo->width, cHInfo->inpStride, cHInfo->height, d_iData, cPlain->kernel->kerDatTex);
                  else
                    convolveffdot31<<<dimGrid, dimBlock, 0, cStack->cnvlStream>>>(d_plainData, cHInfo->width, cHInfo->inpStride, cHInfo->height, d_iData, cPlain->kernel->d_kerData);

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

        if ( DBG_PLN01 ) // Print debug info  .
        {
          for (int ss = 0; ss < batch->noStacks; ss++)
          {
            cuFfdotStack* cStack = &batch->stacks[ss];
            CUDA_SAFE_CALL(cudaStreamSynchronize(cStack->cnvlStream),"");
          }

          for (int ss = 0; ss < batch->noHarms; ss++) // Print
          {
            cuFFdot* cPlain     = &batch->plains[batch->pIdx[ss]];
            printf("\nGPU Convolved h:%i   f: %f\n",ss,cPlain->harmInf->harmFrac);
            printData_cu(batch, batch->flag, batch->pIdx[ss], 10, 1);
            CUDA_SAFE_CALL(cudaStreamSynchronize(0),"");
          }
        }
      }

      FOLD // Inverse FFT the  f-∂f plain  .
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

              if ( batch->flag & FLAG_CUFFTCB_OUT ) // Set the CUFFT callback to calculate and store powers  .
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


