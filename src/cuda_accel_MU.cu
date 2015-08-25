#include <cuda_fp16.h>

#include "cuda_accel_MU.h"

//====================================== Constant variables  ===============================================\\

__device__ cufftCallbackLoadC  d_loadCallbackPtr    = CB_MultiplyInput;
__device__ cufftCallbackStoreC d_storePow_f         = CB_PowerOut_f;
__device__ cufftCallbackStoreC d_storePow_h         = CB_PowerOut_h;
__device__ cufftCallbackStoreC d_storeInmemRow      = CB_PowerOutInmem_ROW;
__device__ cufftCallbackStoreC d_storeInmemPln      = CB_PowerOutInmem_PLN;


//======================================= Global variables  ================================================\\


//========================================== Functions  ====================================================\\

__device__ cufftComplex CB_MultiplyInput( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr)
{
  stackInfo *inf  = (stackInfo*)callerInfo;

  int fIdx        = inf->famIdx;
  int noSteps     = inf->noSteps;
  int noPlains    = inf->noPlains;
  int stackStrd   = STRIDE_HARM[fIdx];
  int width       = WIDTH_HARM[fIdx];

  int strd        = stackStrd * noSteps ;                 /// Stride taking into acount steps)
  int gRow        = offset / strd;                        /// Row (ignoring steps)
  int col         = offset % stackStrd;                   /// 2D column
  int top         = 0;                                    /// The top of the plain
  int pHeight     = 0;
  int pln         = 0;

  for ( int i = 0; i < noPlains; i++ )
  {
    top += HEIGHT_HARM[fIdx+i];

    if ( gRow >= top )
    {
      pln         = i+1;
      pHeight     = top;
    }
  }

  int row         = offset / stackStrd - pHeight*noSteps;
  int pIdx        = fIdx + pln;
  int plnHeight   = HEIGHT_HARM[pIdx];
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

  cufftComplex ker = ((cufftComplex*)(KERNEL_HARM[pIdx]))[row*stackStrd + col];      //
  cufftComplex inp = ((cufftComplex*)inf->d_iData)[(pln*noSteps+step)*stackStrd + col];   //

  // Do the multiplication
  cufftComplex out;
  out.x = ( inp.x * ker.x + inp.y * ker.y ) / (float)width;
  out.y = ( inp.y * ker.x - inp.x * ker.y ) / (float)width;

  return out;
}

__device__ void CB_PowerOut_f( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  // Calculate power
  float power = element.x*element.x + element.y*element.y ;

  // Write result (offsets are the same)
  ((float*)callerInfo)[offset] = power;
}

__device__ void CB_PowerOut_h( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  // Calculate power
  float power = element.x*element.x + element.y*element.y ;

  // Write result (offsets are the same)
  ((half*)callerInfo)[offset] = __float2half(power);
}

__device__ void CB_PowerOutInmem_ROW( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  const int hw  = 56 * 2  ; // HWIDTH_STAGE[0];
  const int al  = 3870 ; //ALEN ;
  const int ns  = 4 ; // NO_STEPS
  int row   = offset  / ( INMEM_FFT_WIDTH * ns ) ;
  int col   = offset  % INMEM_FFT_WIDTH;
  int step  = ( offset % ( INMEM_FFT_WIDTH * ns ) ) / INMEM_FFT_WIDTH;


  col      -= hw;

  //if ( col >= 0 && col < al )
  {

    // Calculate power
    float power = element.x*element.x + element.y*element.y ;
    //half  power = __float2half(element.x*element.x + element.y*element.y) ;

    // Write result (offsets are the same)
    //int plnOff = /*row * PLN_STRIDE*/ + step*al + col;
    //PLN_START[plnOff] = power;
    //PLN_START[offset] = power;
    //((float*)callerInfo)[plnOff] = power;
    //((float*)callerInfo)[offset] = power;
    ((half*)callerInfo)[offset] = __float2half(power);
    //((half*)callerInfo)[offset] = power;

    //  if ( offset == 162735 )
    //  {
    //    printf("\n");
    //
    //    printf("PLN_START:  %p \n", PLN_START);
    //    printf("PLN_STRIDE: %i \n", PLN_STRIDE);
    //    printf("NO_STEPS:   %i \n", NO_STEPS);
    //    printf("step0:      %i \n", step0);
    //
    //    printf("row:        %i \n", row);
    //    printf("col:        %i \n", col);
    //    printf("step:       %i \n", step);
    //  }
  }
}

__device__ void CB_PowerOutInmem_PLN( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  //  int step0 = (int)callerInfo; // I know this isn't right but its faster than accessing the pointer =)
  //  int row   = offset  / INMEM_FFT_WIDTH;
  //  int step  = row /  HEIGHT_STAGE[0];
  //  row       = row %  HEIGHT_STAGE[0];  // Assumes plain interleaved!
  //  int col   = offset % INMEM_FFT_WIDTH;
  //int plnOff = row * PLN_STRIDE + step0 + step + col;

  // Calculate power
  float power = element.x*element.x + element.y*element.y ;

  // Write result
  //PLN_START[plnOff] = power;
  //((float*)callerInfo)[offset] = power;
  ((half*)callerInfo)[offset] = __float2half(power);
}

void copyCUFFT_LD_CB(cuFFdotBatch* batch)
{
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_ldCallbackPtr, d_loadCallbackPtr,  sizeof(cufftCallbackLoadC)),   "");

  //  if ( batch->flag & FLAG_SS_INMEM  )
  //  {
  //    if      ( batch->flag & FLAG_ITLV_ROW )
  //      CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_storeInmemRow, sizeof(cufftCallbackStoreC)),  "");
  //    else if ( batch->flag & FLAG_ITLV_PLN )
  //      CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_storeInmemPln, sizeof(cufftCallbackStoreC)),  "");
  //    else
  //    {
  //      fprintf(stderr,"ERROR: invalid memory lay out. Line %i in %s\n", __LINE__, __FILE__);
  //    }
  //  }
  //  else
  //  {
  //    CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_storeCallbackPtr, sizeof(cufftCallbackStoreC)),  "");
  //  }

  if (  (batch->flag & FLAG_SS_INMEM) && ( batch->flag & FLAG_HALF) )
  {
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_storePow_h, sizeof(cufftCallbackStoreC)),  "");
  }
  else
  {
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_storePow_f, sizeof(cufftCallbackStoreC)),  "");
  }

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

      if ( batch->retType & CU_STR_PLN )
      {
        CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, batch->candCpyComp, 0), "Waiting for GPU to be ready to copy data to device.");  // Multiplication will change the plain
      }

#ifdef SYNCHRONOUS
      // Wait for all the input FFT's to complete
      for (int ss = 0; ss < batch->noStacks; ss++)
      {
        cuFfdotStack* cStack2 = &batch->stacks[ss];
        cudaStreamWaitEvent(cStack->fftPStream, cStack2->prepComp, 0);
      }

      // Wait for the previous multiplication to complete
      if ( pStack != NULL )
        cudaStreamWaitEvent(cStack->fftPStream, pStack->ifftComp, 0);
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
          cudaEventRecord(cStack->ifftInit, cStack->fftPStream);
#endif
        }

        FOLD // Set store FFT callback  .
        {
          if ( batch->flag & FLAG_CUFFT_CB_OUT )
          {
            if ( batch->flag & FLAG_SS_INMEM  )
            {
              rVals* rVal;
              rVal = &((*batch->rSearch)[0][0]);

              printf("\nRval: %i  adressL %p  \n", rVal->step, &rVal->step );

              CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)rVal->step ),"");
            }
            else
            {
              CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&cStack->d_plainPowers ),"");
            }
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
      cudaEventRecord(cStack->ifftComp, cStack->fftPStream);

#ifdef SYNCHRONOUS
      pStack = cStack;
#endif
    }
  }
}

__global__ void cpyPowers_ker( float*  dst, size_t  dpitch, const float *  src, size_t  spitch, size_t  width, size_t  height)
{
  int ix = blockIdx.x * 16 + threadIdx.x ;
  //int iy = blockIdx.y * 16 + threadIdx.y ;

  for ( int iy = 0 ; iy < height; iy++)
  {
    if ( ix < width && iy < height)
    {
      dst[iy*dpitch + ix] = src[iy*spitch +ix];
    }
  }
}

void cpyPowers( float* __restrict__ dst, size_t  dpitch, const float* __restrict__ src, size_t  spitch, size_t  width, size_t  height, cudaStream_t  stream)
{
  dim3 dimBlock, dimGrid;

  dimBlock.x  = 512;
  dimBlock.y  = 1 ; //16;

  float ww    = width  / (float)dimBlock.x ;
  float hh    = height / (float)dimBlock.y ;

  dimGrid.x   = ceil(ww);
  dimGrid.y   = 1 ; // ceil(hh);

  cpyPowers_ker<<<dimGrid,  dimBlock, 0, stream >>>(dst, dpitch, src, spitch, width, height);
}

template<typename T>
void copyIFFTtoPln( cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  rVals* rVal;

  T* dst;
  T* src;
  size_t  dpitch;
  size_t  spitch;
  size_t  width;
  size_t  height;

  int powSz;

  powSz = sizeof(T);

  dpitch  = batch->sInf->mInf->inmemStride * powSz;
  width   = batch->accelLen * powSz;
  height  = cStack->height;
  spitch  = cStack->strideFloat * powSz;

  for ( int step = 0; step < batch->noSteps; step++ )
  {
    rVal = &((*batch->rInput)[step][0]);

    //if ( rVal->step >= 0 && rVal->step < batch->sInf->SrchSz->noSteps )
    if ( rVal->numrs )
    {
      dst     = ((T*)batch->d_plainFull)    + rVal->step * batch->accelLen;

      if      ( batch->flag & FLAG_ITLV_ROW )
      {
        src     = ((T*)cStack->d_plainPowers)  + cStack->strideFloat*step + batch->hInfos->halfWidth * ACCEL_NUMBETWEEN;
        spitch  = cStack->strideFloat*batch->noSteps*powSz;
      }
      else if ( batch->flag & FLAG_ITLV_PLN )
      {
        src     = ((T*)cStack->d_plainPowers)  + cStack->strideFloat*height*step + batch->hInfos->halfWidth * ACCEL_NUMBETWEEN ;
      }
      else
      {
        fprintf(stderr,"ERROR: Invalid interleaving, on line %i in %s.", __LINE__, __FILE__);
      }

      //CUDA_SAFE_CALL(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, cStack->fftPStream ),"Error calling cudaMemcpy2DAsync after IFFT.");

      CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->strmSearch, cStack->ifftComp, 0), "");
      CUDA_SAFE_CALL(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, batch->strmSearch ),"Error calling cudaMemcpy2DAsync after IFFT.");

      FOLD // Synchronisation  .
      {
        cudaEventRecord(cStack->ifftMemComp, batch->strmSearch);
      }
    }
    else
    {
      TMP
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

    if ( batch->flag & FLAG_CUFFT_CB_IN )   // Do the multiplication using a CUFFT callback  .
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
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, cStack->prepComp,0),      "Waiting for GPU to be ready to copy data to device.");    // Need input data

              if ( (batch->flag & FLAG_CUFFT_CB_OUT) )
              {
                // CFF output callback has its own data so can start once FFT is complete
                CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, cStack->ifftComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plain so search must be compete
              }
            }

            if ( !(batch->flag & FLAG_CUFFT_CB_OUT) )
            {
              // Have to wait for search to finish reading data
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, batch->searchComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plain so search must be compete
            }

            if ( (batch->retType & CU_STR_PLN) && !(batch->flag & FLAG_CUFFT_CB_OUT) )
            {
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, batch->candCpyComp, 0),   "Waiting for GPU to be ready to copy data to device.");   // Multiplication will change the plain
            }
          }

          FOLD // Call kernel  .
          {
#ifdef TIMING // Timing event  .
            CUDA_SAFE_CALL(cudaEventRecord(batch->multInit, batch->multStream),"Recording event: multInit");
#endif

            mult30_f(batch->multStream, batch);

            // Run message
            CUDA_SAFE_CALL(cudaGetLastError(), "Error at kernel launch");
          }

          FOLD // Synchronisation  .
          {
            CUDA_SAFE_CALL(cudaEventRecord(batch->multComp, batch->multStream),"Recording event: multComp");
          }
        }
        else if ( batch->flag & FLAG_MUL_STK   )  // Do the multiplications one stack  at a time  .
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
              CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->prepComp,  0),  "Waiting for GPU to be ready to copy data to device.");  // Need input data

              if ( (batch->flag & FLAG_CUFFT_CB_OUT) )
              {
                // CFF output callback has its own data so can start once FFT is complete
                CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->ifftComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plain so search must be compete
              }
              else
              {
                // Have to wait for search to finish reading data
                CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, batch->searchComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plain so search must be compete
              }

              if ( (batch->retType & CU_STR_PLN) && !(batch->flag & FLAG_CUFFT_CB_OUT) )
              {
                CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, batch->candCpyComp, 0), "Waiting for GPU to be ready to copy data to device.");  // Multiplication will change the plain
              }

#ifdef SYNCHRONOUS
              // Wait for all the input FFT's to complete
              for (int ss = 0; ss < batch->noStacks; ss++)
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
              if      ( cStack->flag & FLAG_MUL_00 )
              {
                mult00(cStack->multStream, batch, ss);
              }
              else if ( cStack->flag & FLAG_MUL_21 )
              {
                mult21_f(cStack->multStream, batch, ss);
              }
              else if ( cStack->flag & FLAG_MUL_22 )
              {
                mult22_f(cStack->multStream, batch, ss);
              }
              else if ( cStack->flag & FLAG_MUL_23 )
              {
                mult23_f(cStack->multStream, batch, ss);
              }
              else
              {
                fprintf(stderr,"ERROR: No valid stack multiplication specified. Line %i in %s.\n", __LINE__, __FILE__);
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
          mult10(batch);
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
        for (int ss = 0; ss < batch->noStacks; ss++)
        {
          cuFfdotStack* cStack = &batch->stacks[ss];

#ifdef STPMSG
          printf("\t\t\tStack %i\n",ss);
#endif

          FOLD // Synchronisation  .
          {
#ifdef STPMSG
            printf("\t\t\t\tSynchronisation\n");
#endif
            cudaStreamWaitEvent(cStack->fftPStream, cStack->multComp, 0);
            cudaStreamWaitEvent(cStack->fftPStream, batch->multComp,  0);

            if ( (batch->retType & CU_STR_PLN) && (batch->flag & FLAG_CUFFT_CB_OUT) )
            {
              CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, batch->candCpyComp, 0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plain so search must be compete
            }

            if ( batch->flag & FLAG_SS_INMEM  )
            {
              CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack->ifftMemComp, 0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plain so search must be compete
            }

#ifdef SYNCHRONOUS
            // Wait for all the multiplications to complete
            for (int ss = 0; ss< batch->noStacks; ss++)
            {
              cuFfdotStack* cStack2 = &batch->stacks[ss];
              cudaStreamWaitEvent(cStack->fftPStream, cStack2->multComp, 0);
            }

            // Wait for the previous fft to complete
            if ( pStack != NULL )
              cudaStreamWaitEvent(cStack->fftPStream, pStack->ifftComp, 0);
#endif
          }

          FOLD // Call the inverse CUFFT  .
          {
            //#pragma omp critical
            {
#ifdef STPMSG
              printf("\t\t\t\tCall the inverse CUFFT\n");
#endif
              FOLD // Timing  .
              {
#ifdef TIMING
                cudaEventRecord(cStack->ifftInit, cStack->fftPStream);
#endif
              }

              rVals* rVal;
              rVal = &((*batch->rInput)[0][0]);

              FOLD // Set store FFT callback  .
              {
                if ( batch->flag & FLAG_CUFFT_CB_OUT )
                {
                  //                  if ( batch->flag & FLAG_SS_INMEM  )
                  //                  {
                  //                    CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&cStack->d_plainPowers ),"");
                  //                  }
                  //                  else
                  {
                    CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&cStack->d_plainPowers ),"");
                  }
                }
              }

              FOLD // Call the FFT  .
              {
                CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with multStream.");
                CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) cStack->d_plainData, (cufftComplex *) cStack->d_plainData, CUFFT_INVERSE),"Error executing CUFFT plan.");
              }

              FOLD // Synchronisation  .
              {
                cudaEventRecord(cStack->ifftComp, cStack->fftPStream);
              }

              FOLD // Copy data to device plain  .
              {
                if ( batch->flag & FLAG_SS_INMEM  )
                {
                  FOLD // Copy memory on the device  .
                  {
                    if ( batch->flag & FLAG_HALF )
                    {
                      copyIFFTtoPln<half>( batch, cStack );
                    }
                    else
                    {
                      copyIFFTtoPln<float>( batch, cStack );
                    }

                    CUDA_SAFE_CALL(cudaGetLastError(), "Error at IFFT - cudaMemcpy2DAsync");
                  }
                }

              }

#ifdef SYNCHRONOUS
              pStack = cStack;
#endif
            }
          }

#ifdef STPMSG
          printf("\t\t\tDone\n",ss);
#endif
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

