#include "cuda_accel_MU.h"

//====================================== Constant variables  ===============================================\\

#if CUDA_VERSION >= 6050
__device__ cufftCallbackLoadC  d_loadCallbackPtr    = CB_MultiplyInput;
__device__ cufftCallbackStoreC d_storePow_f         = CB_PowerOut_f;
#if CUDA_VERSION >= 7050
__device__ cufftCallbackStoreC d_storePow_h         = CB_PowerOut_h;
#endif
//__device__ cufftCallbackStoreC d_storeInmemRow      = CB_PowerOutInmem_ROW;
//__device__ cufftCallbackStoreC d_storeInmemPln      = CB_PowerOutInmem_PLN;
#endif

//======================================= Global variables  ================================================\\


//========================================== Functions  ====================================================\\

#if CUDA_VERSION >= 6050

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

#if CUDA_VERSION >= 7050
__device__ void CB_PowerOut_h( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  // Calculate power
  float power = element.x*element.x + element.y*element.y ;

  // Write result (offsets are the same)
  ((half*)callerInfo)[offset] = __float2half(power);
}
#endif

//__device__ void CB_PowerOutInmem_ROW( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
//{
//  //  const int hw  = HWIDTH_STAGE[0];
//  //  const int al  = ALEN ;
//  //  const int ns  = NO_STEPS;
//  //  int row   = offset  / ( INMEM_FFT_WIDTH * ns ) ;
//  //  int col   = offset  % INMEM_FFT_WIDTH;
//  //  int step  = ( offset % ( INMEM_FFT_WIDTH * ns ) ) / INMEM_FFT_WIDTH;
//
//
//  //  col      -= hw;
//
//  //if ( col >= 0 && col < al )
//  {
//
//    // Calculate power
//    float power = element.x*element.x + element.y*element.y ;
//    //half  power = __float2half(element.x*element.x + element.y*element.y) ;
//
//    // Write result (offsets are the same)
//    //int plnOff = /*row * PLN_STRIDE*/ + step*al + col;
//    //PLN_START[plnOff] = power;
//    //PLN_START[offset] = power;
//    //((float*)callerInfo)[plnOff] = power;
//    //((float*)callerInfo)[offset] = power;
//    ((half*)callerInfo)[offset] = __float2half(power);
//    //((half*)callerInfo)[offset] = power;
//
//    //  if ( offset == 162735 )
//    //  {
//    //    printf("\n");
//    //
//    //    printf("PLN_START:  %p \n", PLN_START);
//    //    printf("PLN_STRIDE: %i \n", PLN_STRIDE);
//    //    printf("NO_STEPS:   %i \n", NO_STEPS);
//    //    printf("step0:      %i \n", step0);
//    //
//    //    printf("row:        %i \n", row);
//    //    printf("col:        %i \n", col);
//    //    printf("step:       %i \n", step);
//    //  }
//  }
//}
//
//__device__ void CB_PowerOutInmem_PLN( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
//{
//  //  int step0 = (int)callerInfo; // I know this isn't right but its faster than accessing the pointer =)
//  //  int row   = offset  / INMEM_FFT_WIDTH;
//  //  int step  = row /  HEIGHT_STAGE[0];
//  //  row       = row %  HEIGHT_STAGE[0];  // Assumes plain interleaved!
//  //  int col   = offset % INMEM_FFT_WIDTH;
//  //int plnOff = row * PLN_STRIDE + step0 + step + col;
//
//  // Calculate power
//  float power = element.x*element.x + element.y*element.y ;
//
//  // Write result
//  //PLN_START[plnOff] = power;
//  //((float*)callerInfo)[offset] = power;
//  ((half*)callerInfo)[offset] = __float2half(power);
//}

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
#if CUDA_VERSION >= 7050
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_storePow_h, sizeof(cufftCallbackStoreC)),  "");
#else
    fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
    exit(EXIT_FAILURE);
#endif
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
  for (int ss = 0; ss < batch->noStacks; ss++)
  {
    int sIdx;

    if ( batch->flag & FLAG_STK_UP )
      sIdx = batch->noStacks - 1 - ss;
    else
      sIdx = ss;

    cuFfdotStack* cStack = &batch->stacks[sIdx];

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
      for (int synchIdx = 0; synchIdx < batch->noStacks; synchIdx++)
      {
        cuFfdotStack* cStack2 = &batch->stacks[synchIdx];
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
#if CUDA_VERSION >= 6050
            if ( batch->flag & FLAG_SS_INMEM  )
            {
              //rVals* rVal;
              //rVal = &((*batch->rSearch)[0][0]);
              rVals* rVal = &batch->rArrays[1][0][0];

              printf("\nRval: %i  adressL %p  \n", rVal->step, &rVal->step );

              CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)rVal->step ),"");
            }
            else
            {
              CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&cStack->d_plainPowers ),"");
            }
#else
            fprintf(stderr,"ERROR: CUFFT callbacks can only be used with CUDA 6.5 or later!\n");
            exit(EXIT_FAILURE);
#endif

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

template<typename T>
__global__ void cpyPowers_ker( T*  dst, size_t  dpitch, T*  src, size_t  spitch, size_t  width, size_t  height)
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

template<typename T>
void cpyPowers( T* __restrict__ dst, size_t  dpitch, T* __restrict__ src, size_t  spitch, size_t  width, size_t  height, cudaStream_t  stream)
{
  dim3 dimBlock, dimGrid;

  dimBlock.x  = 512;
  dimBlock.y  = 1 ; //16;

  float ww    = width  / (float)dimBlock.x ;
  float hh    = height / (float)dimBlock.y ;

  dimGrid.x   = ceil(ww);
  dimGrid.y   = 1 ; // ceil(hh);

  cpyPowers_ker<T><<<dimGrid,  dimBlock, 0, stream >>>(dst, dpitch, src, spitch, width, height);
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

  int powSz = 1;

  powSz = sizeof(T);

  dpitch  = batch->sInf->mInf->inmemStride * powSz;
  width   = batch->accelLen * powSz;
  height  = cStack->height;
  spitch  = cStack->strideFloat * powSz;

  for ( int step = 0; step < batch->noSteps; step++ )
  {
    //rVal = &((*batch->rInput)[step][0]);
    //rVal = &((*batch->rConvld)[step][0]);
    rVals* rVal = &batch->rArrays[2][0][0];

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

      //cpyPowers<T>(dst, dpitch, src, spitch,  width,  height, batch->strmSearch );

      //CUDA_SAFE_CALL(cudaMemcpyAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, batch->strmSearch ),"Error calling cudaMemcpy2DAsync after IFFT.");
      //CUDA_SAFE_CALL(cudaMemcpyAsync(cStack->d_plainPowers, batch->d_iData, batch->accelLen*10, cudaMemcpyDeviceToDevice, batch->strmSearch), "Failed to copy input data to device");

    }

    FOLD // Synchronisation  .
    {
      cudaEventRecord(cStack->ifftMemComp, batch->strmSearch);
    }
  }
}

#endif

void multiplyBatch(cuFFdotBatch* batch, int rIdx)
{
  if ( batch->rArrays[rIdx][0][0].numrs )
  {
    nvtxRangePush("Multiply");
#ifdef STPMSG
    printf("\tMultiply & FFT\n");
#endif

    if ( batch->flag & FLAG_CUFFT_CB_IN )   // Do the multiplication using a CUFFT callback  .
    {
#ifdef STPMSG
      printf("\t\tMultiply with CUFFT\n");
#endif

#if CUDA_VERSION >= 6050
      multiplyBatchCUFFT( batch );
#else
      fprintf(stderr,"ERROR: CUFFT callbacks can only be used with CUDA 6.5 or later!\n");
      exit(EXIT_FAILURE);
#endif
    }
    else                                    // Do the multiplication and FFT separately  .
    {
      FOLD // Multiply  .
      {
#ifdef STPMSG
        printf("\t\tMultiply\n");
#endif

        // In my testing I found multiplying each plain separately works fastest so it is the "default"
        if      ( batch->flag & FLAG_MUL_BATCH )  // Do the multiplications one family at a time  .
        {
          FOLD // Synchronisation  .
          {
            // Synchronise input data preparation for all stacks
            for (int synchIdx = 0; synchIdx < batch->noStacks; synchIdx++)
            {
              cuFfdotStack* cStack = &batch->stacks[synchIdx];

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
            int sIdx;

            if ( batch->flag & FLAG_STK_UP )
              sIdx = batch->noStacks - 1 - ss;
            else
              sIdx = ss;

            cuFfdotStack* cStack = &batch->stacks[sIdx];

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
              for (int synchIdx = 0; synchIdx < batch->noStacks; synchIdx++)
              {
                cuFfdotStack* cStack2 = &batch->stacks[synchIdx];
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
                mult00(cStack->multStream, batch, sIdx);
              }
              else if ( cStack->flag & FLAG_MUL_21 )
              {
                mult21_f(cStack->multStream, batch, sIdx);
              }
              else if ( cStack->flag & FLAG_MUL_22 )
              {
                mult22_f(cStack->multStream, batch, sIdx);
              }
              else if ( cStack->flag & FLAG_MUL_23 )
              {
                mult23_f(cStack->multStream, batch, sIdx);
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
        else if ( batch->flag & FLAG_MUL_PLN )    // Do the multiplications one plain  at a time  .
        {
          mult10(batch);
        }
        else
        {
          fprintf(stderr, "ERROR: multiplyBatch not templated for this type of multiplication.\n");
        }
      }
    }

    nvtxRangePop();
  }
}

void IFFTBatch(cuFFdotBatch* batch, int rIdx)
{
  if ( batch->rArrays[rIdx][0][0].numrs ) // Inverse FFT the batch  .
  {
    nvtxRangePush("IFFT");

#ifdef STPMSG
    printf("\t\tInverse FFT\n");
#endif

#ifdef SYNCHRONOUS
    cuFfdotStack* pStack = NULL;  // Previous stack
#endif

    for (int ss = 0; ss < batch->noStacks; ss++)
    {
      int sIdx;

      if ( batch->flag & FLAG_STK_UP )
        sIdx = batch->noStacks - 1 - ss;
      else
        sIdx = ss;

      cuFfdotStack* cStack = &batch->stacks[sIdx];

#ifdef STPMSG
      printf("\t\t\tStack %i\n", sIdx);
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
        for (int synchIdx = 0; synchIdx < batch->noStacks; synchIdx++)
        {
          cuFfdotStack* cStack2 = &batch->stacks[synchIdx];
          cudaStreamWaitEvent(cStack->fftPStream, cStack2->multComp, 0);
        }

        // Wait for the previous fft to complete
        if ( pStack != NULL )
          cudaStreamWaitEvent(cStack->fftPStream, pStack->ifftComp, 0);
#endif
      }

      FOLD // Call the inverse CUFFT  .
      {
#pragma omp critical
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

          rVals* rVal = &batch->rArrays[rIdx][0][0];

          FOLD // Set store FFT callback  .
          {
            if ( batch->flag & FLAG_CUFFT_CB_OUT )
            {
#if CUDA_VERSION >= 6050
              CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&cStack->d_plainPowers ),"");
#else
              fprintf(stderr,"ERROR: CUFFT callbacks can only be used with CUDA 6.5 or later!\n");
              exit(EXIT_FAILURE);
#endif
            }
          }

          FOLD // Call the FFT  .
          {
            CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with multStream.");
            CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) cStack->d_plainData, (cufftComplex *) cStack->d_plainData, CUFFT_INVERSE),"Error executing CUFFT plan.");
          }
#ifdef SYNCHRONOUS
          pStack = cStack;
#endif
        }
      }

      FOLD // Synchronisation  .
      {
        cudaEventRecord(cStack->ifftComp, cStack->fftPStream);
      }

#ifdef STPMSG
      printf("\t\t\tDone\n", sIdx);
#endif
    }

    nvtxRangePop();
  }
}

void copyToInMemPln(cuFFdotBatch* batch, int rIdx)
{
  if ( batch->rArrays[rIdx][0][0].numrs )
  {
    if ( batch->flag & FLAG_SS_INMEM )
    {
      // Copy back data  (out of order)  .
      for (int ss = 0; ss < batch->noStacks; ss++)
      {
        int sIdx;

        if ( batch->flag & FLAG_STK_UP )
          sIdx = batch->noStacks - 1 - ss;
        else
          sIdx = ss;

        cuFfdotStack* cStack = &batch->stacks[sIdx];

        FOLD // Copy memory on the device  .
        {
          if ( batch->flag & FLAG_HALF )
          {
#if CUDA_VERSION >= 7050
            copyIFFTtoPln<half>( batch, cStack );
#else
            fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
            exit(EXIT_FAILURE);
#endif
          }
          else
          {
            copyIFFTtoPln<float>( batch, cStack );
          }

          CUDA_SAFE_CALL(cudaGetLastError(), "Error at IFFT - cudaMemcpy2DAsync");
        }
      }
    }
  }
}

/** Multiply and inverse FFT the complex f-∂f plain  .
 * This assumes the input data is ready and on the device
 * This creates a complex f-∂f plain
 */
void convolveBatch(cuFFdotBatch* batch, int rIdx)
{
  //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  // Multiply
  if ( batch->rArrays[rIdx][0][0].numrs )
  {
    nvtxRangePush("Multiply");
#ifdef STPMSG
    printf("\tMultiply & FFT\n");
#endif

    if ( batch->flag & FLAG_CUFFT_CB_IN )   // Do the multiplication using a CUFFT callback  .
    {
#ifdef STPMSG
      printf("\t\tMultiply with CUFFT\n");
#endif

#if CUDA_VERSION >= 6050
      multiplyBatchCUFFT( batch );
#else
      fprintf(stderr,"ERROR: CUFFT callbacks can only be used with CUDA 6.5 or later!\n");
      exit(EXIT_FAILURE);
#endif
    }
    else                                    // Do the multiplication and FFT separately  .
    {
      FOLD // Multiply  .
      {
#ifdef STPMSG
        printf("\t\tMultiply\n");
#endif

        // In my testing I found multiplying each plain separately works fastest so it is the "default"
        if      ( batch->flag & FLAG_MUL_BATCH )  // Do the multiplications one family at a time  .
        {
          FOLD // Synchronisation  .
          {
            // Synchronise input data preparation for all stacks
            for (int synchIdx = 0; synchIdx < batch->noStacks; synchIdx++)
            {
              cuFfdotStack* cStack = &batch->stacks[synchIdx];

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
            int sIdx;

            if ( batch->flag & FLAG_STK_UP )
              sIdx = batch->noStacks - 1 - ss;
            else
              sIdx = ss;

            cuFfdotStack* cStack = &batch->stacks[sIdx];

            FOLD // Multiply  .
            {
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
                for (int synchIdx = 0; synchIdx < batch->noStacks; synchIdx++)
                {
                  cuFfdotStack* cStack2 = &batch->stacks[synchIdx];
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
                  mult00(cStack->multStream, batch, sIdx);
                }
                else if ( cStack->flag & FLAG_MUL_21 )
                {
                  mult21_f(cStack->multStream, batch, sIdx);
                }
                else if ( cStack->flag & FLAG_MUL_22 )
                {
                  mult22_f(cStack->multStream, batch, sIdx);
                }
                else if ( cStack->flag & FLAG_MUL_23 )
                {
                  mult23_f(cStack->multStream, batch, sIdx);
                }
                else
                {
                  fprintf(stderr,"ERROR: No valid stack multiplication specified. Line %i in %s.\n", __LINE__, __FILE__);
                  exit(EXIT_FAILURE);
                }

                // Run message
                CUDA_SAFE_CALL(cudaGetLastError(), "Error at kernel launch multiplication");
              }

              FOLD // Synchronisation  .
              {
                cudaEventRecord(cStack->multComp, cStack->multStream);

#ifdef SYNCHRONOUS
                pStack = cStack;
#endif
              }
            }

            FOLD // IFFT  .
            {
              if ( batch->flag & FLAG_CONV )
              {
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
                  for (int synchIdx = 0; synchIdx < batch->noStacks; synchIdx++)
                  {
                    cuFfdotStack* cStack2 = &batch->stacks[synchIdx];
                    cudaStreamWaitEvent(cStack->fftPStream, cStack2->multComp, 0);
                  }

                  // Wait for the previous fft to complete
                  if ( pStack != NULL )
                    cudaStreamWaitEvent(cStack->fftPStream, pStack->ifftComp, 0);
#endif
                }

                FOLD // Call the inverse CUFFT  .
                {
#pragma omp critical
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

                    rVals* rVal = &batch->rArrays[rIdx][0][0];

                    FOLD // Set store FFT callback  .
                    {
                      if ( batch->flag & FLAG_CUFFT_CB_OUT )
                      {
#if CUDA_VERSION >= 6050
                        CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&cStack->d_plainPowers ),"");
#else
                        fprintf(stderr,"ERROR: CUFFT callbacks can only be used with CUDA 6.5 or later!\n");
                        exit(EXIT_FAILURE);
#endif
                      }
                    }

                    FOLD // Call the FFT  .
                    {
                      CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with multStream.");
                      CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) cStack->d_plainData, (cufftComplex *) cStack->d_plainData, CUFFT_INVERSE),"Error executing CUFFT plan.");
                    }
#ifdef SYNCHRONOUS
                    pStack = cStack;
#endif
                  }
                }

                FOLD // Synchronisation  .
                {
                  cudaEventRecord(cStack->ifftComp, cStack->fftPStream);
                }
              }
            }
          }
        }
        else if ( batch->flag & FLAG_MUL_PLN )    // Do the multiplications one plain  at a time  .
        {
          mult10(batch);
        }
        else
        {
          fprintf(stderr, "ERROR: multiplyBatch not templated for this type of multiplication.\n");
        }
      }
    }

    nvtxRangePop();
  }

  //  if ( batch->flag & FLAG_SS_INMEM )
  //  {
  //    if ( batch->state & HAVE_PLN ) // Copy back data  (out of order)  .
  //    {
  //      for (int ss = 0; ss < batch->noStacks; ss++)
  //      {
  //        int sIdx;
  //
  //        if ( batch->flag & FLAG_STK_UP )
  //          sIdx = batch->noStacks - 1 - ss;
  //        else
  //          sIdx = ss;
  //
  //        cuFfdotStack* cStack = &batch->stacks[sIdx];
  //
  //        FOLD // Copy memory on the device  .
  //        {
  //          if ( batch->flag & FLAG_HALF )
  //          {
  //#if CUDA_VERSION >= 7050
  //            copyIFFTtoPln<half>( batch, cStack );
  //#else
  //            fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
  //            exit(EXIT_FAILURE);
  //#endif
  //
  //          }
  //          else
  //          {
  //            copyIFFTtoPln<float>( batch, cStack );
  //          }
  //
  //          CUDA_SAFE_CALL(cudaGetLastError(), "Error at IFFT - cudaMemcpy2DAsync");
  //        }
  //      }
  //      batch->state &= ~HAVE_PLN;
  //    }
  //  }

  // IFFT  .
  if ( batch->rArrays[rIdx][0][0].numrs )
  {
    if ( !( (batch->flag & FLAG_CONV) && (batch->flag & FLAG_MUL_STK) ) )
    {
      nvtxRangePush("IFFT");

#ifdef STPMSG
      printf("\t\tInverse FFT\n");
#endif

#ifdef SYNCHRONOUS
      cuFfdotStack* pStack = NULL;  // Previous stack
#endif

      for (int ss = 0; ss < batch->noStacks; ss++)
      {
        int sIdx;

        if ( batch->flag & FLAG_STK_UP )
          sIdx = batch->noStacks - 1 - ss;
        else
          sIdx = ss;

        cuFfdotStack* cStack = &batch->stacks[sIdx];

#ifdef STPMSG
        printf("\t\t\tStack %i\n", sIdx);
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
          for (int synchIdx = 0; synchIdx < batch->noStacks; synchIdx++)
          {
            cuFfdotStack* cStack2 = &batch->stacks[synchIdx];
            cudaStreamWaitEvent(cStack->fftPStream, cStack2->multComp, 0);
          }

          // Wait for the previous fft to complete
          if ( pStack != NULL )
            cudaStreamWaitEvent(cStack->fftPStream, pStack->ifftComp, 0);
#endif
        }

        FOLD // Call the inverse CUFFT  .
        {
#pragma omp critical
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

            rVals* rVal = &batch->rArrays[rIdx][0][0];

            FOLD // Set store FFT callback  .
            {
              if ( batch->flag & FLAG_CUFFT_CB_OUT )
              {
#if CUDA_VERSION >= 6050
                CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&cStack->d_plainPowers ),"");
#else
                fprintf(stderr,"ERROR: CUFFT callbacks can only be used with CUDA 6.5 or later!\n");
                exit(EXIT_FAILURE);
#endif
              }
            }

            FOLD // Call the FFT  .
            {
              CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with multStream.");
              CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) cStack->d_plainData, (cufftComplex *) cStack->d_plainData, CUFFT_INVERSE),"Error executing CUFFT plan.");
            }
#ifdef SYNCHRONOUS
            pStack = cStack;
#endif
          }
        }

        FOLD // Synchronisation  .
        {
          cudaEventRecord(cStack->ifftComp, cStack->fftPStream);
        }

#ifdef STPMSG
        printf("\t\t\tDone\n", sIdx);
#endif
      }

      nvtxRangePop();
    }
  }
}

