#include "cuda_accel_MU.h"

#define CPY_WIDTH 512  // 256  512 768

//====================================== Constant variables  ===============================================\\

#if CUDA_VERSION >= 6050
__device__ cufftCallbackLoadC  d_loadCallbackPtr    = CB_MultiplyInput;
__device__ cufftCallbackStoreC d_storePow_f         = CB_PowerOut_f;
#if __CUDACC_VER__ >= 70500
__device__ cufftCallbackStoreC d_storePow_h         = CB_PowerOut_h;
#endif
#endif

//======================================= Global variables  ================================================\\


//========================================== Functions  ====================================================\\

#if CUDA_VERSION >= 6050        // CUFFT callbacks only implemented in CUDA 6.5  .

/** CUFFT callback kernel to multiply the complex f-∂f before the FFT  .
 */
__device__ cufftComplex CB_MultiplyInput( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr)
{
  stackInfo *inf  = (stackInfo*)callerInfo;

  int fIdx        = inf->famIdx;
  int noSteps     = inf->noSteps;
  int noPlanes    = inf->noPlanes;
  int stackStrd   = STRIDE_HARM[fIdx];
  int width       = WIDTH_HARM[fIdx];

  int strd        = stackStrd * noSteps ;                 /// Stride taking into acount steps)
  int gRow        = offset / strd;                        /// Row (ignoring steps)
  int col         = offset % stackStrd;                   /// 2D column
  int top         = 0;                                    /// The top of the plane
  int pHeight     = 0;
  int pln         = 0;

  for ( int i = 0; i < noPlanes; i++ )
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

  if ( inf->flag & FLAG_ITLV_ROW )
  {
    step  = row % noSteps;
    row   = row / noSteps;
  }
  else
  {
    step = row / plnHeight;
    row  = row % plnHeight;
  }

  cufftComplex ker = ((cufftComplex*)(KERNEL_HARM[pIdx]))[row*stackStrd + col];           //
  cufftComplex inp = ((cufftComplex*)inf->d_iData)[(pln*noSteps+step)*stackStrd + col];   //

  // Do the multiplication
  cufftComplex out;
  out.x = ( inp.x * ker.x + inp.y * ker.y ) / (float)width;
  out.y = ( inp.y * ker.x - inp.x * ker.y ) / (float)width;

  return out;
}

/** CUFFT callback kernel to calculate and store float powers after the FFT  .
 */
__device__ void CB_PowerOut_f( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  // Calculate power
  float power = element.x*element.x + element.y*element.y ;

  // Write result (offsets are the same)
  ((float*)callerInfo)[offset] = power;
}

#if __CUDACC_VER__ >= 70500 // Half precision CUFFT power call back

/** CUFFT callback kernel to calculate and store half powers after the FFT  .
 */
__device__ void CB_PowerOut_h( void *dataIn, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  // Calculate power
  float power = element.x*element.x + element.y*element.y ;

  // Write result (offsets are the same)
  ((half*)callerInfo)[offset] = __float2half(power);
}

#endif  // CUDA_VERSION >= 7050

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
//  //  row       = row %  HEIGHT_STAGE[0];  // Assumes plane interleaved!
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

/** Load the CUFFT store callback  .
 */
void copyCUFFT_LD_CB(cuFFdotBatch* batch)
{
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_ldCallbackPtr, d_loadCallbackPtr,  sizeof(cufftCallbackLoadC)),   "Getting constant memory address.");

  if ( batch->flag & FLAG_HALF )
  {
#if __CUDACC_VER__ >= 70500
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_storePow_h, sizeof(cufftCallbackStoreC)),  "Getting constant memory address.");
#else
    fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
    exit(EXIT_FAILURE);
#endif
  }
  else
  {
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_storePow_f, sizeof(cufftCallbackStoreC)),  "Getting constant memory address.");
  }

}

/** Multiply and inverse FFT the complex f-∂f plane using FFT callback  .
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
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, batch->searchComp, 0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

      if ( batch->retType & CU_STR_PLN )
      {
        CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, batch->candCpyComp, 0), "Waiting for GPU to be ready to copy data to device.");  // Multiplication will change the plane
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
            CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&cStack->d_planePowr ),"Error assigning CUFFT store callback.");
  #else
            fprintf(stderr,"ERROR: CUFFT callbacks can only be used with CUDA 6.5 or later!\n");
            exit(EXIT_FAILURE);
  #endif
          }
        }

        FOLD // Set load FFT callback  .
        {
          CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_ldCallbackPtr, CUFFT_CB_LD_COMPLEX, (void**)&cStack->d_sInf ),"Error assigning CUFFT load callback.");
        }

        CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with multStream.");
        CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) cStack->d_planeMult, (cufftComplex *) cStack->d_planeMult, CUFFT_INVERSE),"Error executing CUFFT plan.");
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

#endif  // CUDA_VERSION >= 6050

/** Kernel to copy powers from complex plane to inmeme plane  .
 *
 */
template<typename T>
__global__ void cpyPowers_ker( T* dst, size_t  dpitch, T*  src, size_t  spitch, size_t  width, size_t  height)
{
  int ix = blockIdx.x * CPY_WIDTH + threadIdx.x ;
  //int iy = blockIdx.y * 16 + threadIdx.y ;

  for ( int iy = 0 ; iy < height; iy++)
  {
    if ( ix < width && iy < height)
    {
      dst[iy*dpitch + ix] = src[iy*spitch +ix];
    }
  }
}

/** Kernel to copy powers from complex plane to inmeme plane  .
 *
 */
__global__ void cpyCmplx_ker( float* dst, size_t  dpitch, fcomplexcu* src, size_t  spitch, size_t  width, size_t  height)
{
  int ix = blockIdx.x * CPY_WIDTH + threadIdx.x ;
  //int iy = blockIdx.y * 16 + threadIdx.y ;

  const int buffLen = 4;

  float buff[buffLen];

  if ( ix < width )
  {
    //for ( int iy = 0 ; iy < height; iy++)
    int iy;

    for ( iy = 0 ; iy < height - buffLen ; iy+=buffLen)
    {
      for ( int by = 0 ; by < buffLen; by++)
      {
        int gy = iy + by;

        //if ( gy < height)
        {
          fcomplexcu cmplx  = src[gy*spitch + ix];
          buff[by]          = cmplx.i*cmplx.i + cmplx.r*cmplx.r;
        }
      }

      for ( int by = 0 ; by < buffLen; by++)
      {
        int gy = iy + by;

        //if ( gy < height)
        {
          dst[gy*dpitch + ix] = buff[by];
        }
      }
    }

    for ( int by = 0 ; by < buffLen; by++)
    {
      int gy = iy + by;

      if ( gy < height)
      {
        fcomplexcu cmplx  = src[gy*spitch + ix];
        buff[by]          = cmplx.i*cmplx.i + cmplx.r*cmplx.r;
      }
    }

    for ( int by = 0 ; by < buffLen; by++)
    {
      int gy = iy + by;

      if ( gy < height)
      {
        dst[gy*dpitch + ix] = buff[by];
      }
    }
  }
}

/** Function to call the kernel to copy powers from powers plane to inmeme plane  .
 */
template<typename T>
void cpyPowers( T* dst, size_t  dpitch, T* src, size_t  spitch, size_t  width, size_t  height, cudaStream_t  stream)
{
  dim3 dimBlock, dimGrid;

  dimBlock.x  = CPY_WIDTH;
  dimBlock.y  = 1 ;

  float ww    = width  / (float)dimBlock.x ;

  dimGrid.x   = ceil(ww);
  dimGrid.y   = 1 ;

  cpyPowers_ker<T><<<dimGrid,  dimBlock, 0, stream >>>(dst, dpitch, src, spitch, width, height);
}

/** Function to call the kernel to copy powers from powers plane to inmeme plane  .
 */
void cpyCmplx( float* dst, size_t  dpitch, fcomplexcu* src, size_t  spitch, size_t  width, size_t  height, cudaStream_t  stream)
{
  dim3 dimBlock, dimGrid;

  dimBlock.x  = CPY_WIDTH;
  dimBlock.y  = 1 ;

  float ww    = width  / (float)dimBlock.x ;

  dimGrid.x   = ceil(ww);
  dimGrid.y   = 1 ;

  cpyCmplx_ker<<<dimGrid,  dimBlock, 0, stream >>>(dst, dpitch, src, spitch, width, height);
}

template<typename Tin, typename Tout>
void copyIFFTtoPln( cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  Tout*   dst;
  Tin*    src;
  size_t  dpitch;
  size_t  spitch;
  size_t  width;
  size_t  height;

  int inSz  = 1;
  int outSz = 1;

  inSz  = sizeof(Tin);
  outSz = sizeof(Tout);

  dpitch  = batch->sInf->mInf->inmemStride * outSz;
  width   = batch->accelLen * outSz;
  height  = cStack->height;
  spitch  = cStack->stridePower * inSz;

  for ( int step = 0; step < batch->noSteps; step++ )
  {
    rVals* rVal = &batch->rValues[step][0];

    if ( rVal->numrs )
    {
      dst     = ((Tout*)batch->d_planeFull)    + rVal->step * batch->accelLen;

      if      ( batch->flag & FLAG_ITLV_ROW )
      {
        src     = ((Tin*)cStack->d_planePowr)  + cStack->stridePower*step + batch->hInfos->halfWidth * ACCEL_NUMBETWEEN;
        spitch  = cStack->stridePower*batch->noSteps*inSz;
      }
      else
      {
        src     = ((Tin*)cStack->d_planePowr)  + cStack->stridePower*height*step + batch->hInfos->halfWidth * ACCEL_NUMBETWEEN ;
      }

      CUDA_SAFE_CALL(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, batch->srchStream ),"Calling cudaMemcpy2DAsync after IFFT.");
    }
  }
}

void cmplxToPln( cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  float*        dst;
  fcomplexcu*   src;

  size_t        dpitch;
  size_t        spitch;
  size_t        width;
  size_t        height;

  dpitch  = batch->sInf->mInf->inmemStride;
  width   = batch->accelLen;
  height  = cStack->height;
  spitch  = cStack->strideCmplx;

  for ( int step = 0; step < batch->noSteps; step++ )
  {
    rVals* rVal = &batch->rValues[step][0];

    if ( rVal->numrs )
    {
      if ( batch->flag & FLAG_HALF )
      {
        fprintf(stderr, "ERROR: Cannot use non CUFFT callbacks with half presison.\n");
        exit(EXIT_FAILURE);
      }
      dst       = ((float*)batch->d_planeFull)        + rVal->step * batch->accelLen;

      if ( batch->flag & FLAG_ITLV_ROW )
      {
        src     = ((fcomplexcu*)cStack->d_planePowr)  + cStack->strideCmplx*step + batch->hInfos->halfWidth * ACCEL_NUMBETWEEN;
        spitch  = cStack->strideCmplx*batch->noSteps;
      }
      else
      {
        src     = ((fcomplexcu*)cStack->d_planePowr)  + cStack->strideCmplx*height*step + batch->hInfos->halfWidth * ACCEL_NUMBETWEEN ;
      }

      cpyCmplx(dst, dpitch, src, spitch,  width,  height, batch->srchStream );
    }
  }
}

void multStack(cuFFdotBatch* batch, cuFfdotStack* cStack, int sIdx, cuFfdotStack* pStack = NULL)
{
  FOLD // Synchronisation  .
  {
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->prepComp,    0), "Waiting for GPU to be ready to copy data to device.");  // Need input data

    // CFF output callback has its own data so can start once FFT is complete
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->ifftComp,    0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

    if ( !(batch->flag & FLAG_CUFFT_CB_OUT) )
    {
      // Have to wait for search to finish reading data
      //CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, batch->searchComp,   0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
      //CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->ifftMemComp, 0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the complex plane so search must be compete
    }

    if ( (batch->retType & CU_STR_PLN) && !(batch->flag & FLAG_CUFFT_CB_OUT) )
    {
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, batch->candCpyComp,  0), "Waiting for GPU to be ready to copy data to device.");  // Multiplication will change the plane
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
    CUDA_SAFE_CALL(cudaGetLastError(), "At kernel launch (mult7)");
  }

  FOLD // Synchronisation  .
  {
    cudaEventRecord(cStack->multComp, cStack->multStream);
  }
}

void multiplyBatch(cuFFdotBatch* batch)
{
  if ( batch->rValues[0][0].numrs )
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

        // In my testing I found multiplying each plane separately works fastest so it is the "default"
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
                CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, cStack->ifftComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
              }
            }

            if ( !(batch->flag & FLAG_CUFFT_CB_OUT) )
            {
              // Have to wait for search to finish reading data
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, batch->searchComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

            }

            if ( (batch->retType & CU_STR_PLN) && !(batch->flag & FLAG_CUFFT_CB_OUT) )
            {
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, batch->candCpyComp, 0),   "Waiting for GPU to be ready to copy data to device.");   // Multiplication will change the plane
            }
          }

          FOLD // Call kernel  .
          {
#ifdef TIMING // Timing event  .
            CUDA_SAFE_CALL(cudaEventRecord(batch->multInit, batch->multStream),"Recording event: multInit");
#endif

            mult30_f(batch->multStream, batch);

            // Run message
            CUDA_SAFE_CALL(cudaGetLastError(), "At kernel launch");
          }

          FOLD // Synchronisation  .
          {
            CUDA_SAFE_CALL(cudaEventRecord(batch->multComp, batch->multStream),"Recording event: multComp");
          }
        }
        else if ( batch->flag & FLAG_MUL_STK   )  // Do the multiplications one stack  at a time  .
        {
          cuFfdotStack* pStack = NULL;  // Previous stack

          // Multiply this entire stack in one block
          for (int ss = 0; ss < batch->noStacks; ss++)
          {
            int sIdx;

            if ( batch->flag & FLAG_STK_UP )
              sIdx = batch->noStacks - 1 - ss;
            else
              sIdx = ss;

            cuFfdotStack* cStack = &batch->stacks[sIdx];

            multStack(batch, cStack, sIdx, pStack);

            pStack = cStack;
          }
        }
        else if ( batch->flag & FLAG_MUL_PLN )    // Do the multiplications one plane  at a time  .
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

void IFFTStack(cuFFdotBatch* batch, cuFfdotStack* cStack, cuFfdotStack* pStack = NULL)
{
  FOLD // Synchronisation  .
  {
#ifdef STPMSG
    printf("\t\t\t\tSynchronisation\n");
#endif

    // Wait for multiplication to finish
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack->multComp,      0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, batch->multComp,       0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

    // Wait for previous search to finish
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, batch->searchComp,     0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

    if ( (batch->retType & CU_STR_PLN) && (batch->flag & FLAG_CUFFT_CB_OUT) )
    {
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, batch->candCpyComp,  0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
    }

    if ( batch->flag & FLAG_SS_INMEM  )
    {
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack->ifftMemComp, 0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
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

      FOLD // Set store FFT callback  .
      {
        if ( batch->flag & FLAG_CUFFT_CB_OUT )
        {
#if CUDA_VERSION >= 6050
          CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&cStack->d_planePowr ),"Error assigning CUFFT store callback.");
#else
          fprintf(stderr,"ERROR: CUFFT callbacks can only be used with CUDA 6.5 or later!\n");
          exit(EXIT_FAILURE);
#endif
        }
      }

      FOLD // Call the FFT  .
      {
        CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with multStream.");
        CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) cStack->d_planeMult, (cufftComplex *) cStack->d_planePowr, CUFFT_INVERSE),"Error executing CUFFT plan.");
      }
    }
  }

  FOLD // Synchronisation  .
  {
    cudaEventRecord(cStack->ifftComp, cStack->fftPStream);
  }
}

void IFFTBatch(cuFFdotBatch* batch)
{
  if ( batch->rValues[0][0].numrs ) // Inverse FFT the batch  .
  {
    nvtxRangePush("IFFT");

#ifdef STPMSG
    printf("\t\tInverse FFT\n");
#endif

    cuFfdotStack* pStack = NULL;  // Previous stack

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

      IFFTStack(batch, cStack, pStack);

      pStack = cStack;

#ifdef STPMSG
      printf("\t\t\tDone\n", sIdx);
#endif
    }

    nvtxRangePop();
  }
}

void copyToInMemPln(cuFFdotBatch* batch)
{
  if ( batch->rValues[0][0].numrs )
  {
    if ( batch->flag & FLAG_SS_INMEM )
    {
      // Copy back data
      for (int ss = 0; ss < batch->noStacks; ss++) // Note: This is probally unnessecary as there should be only 1 stack for FLAG_SS_INMEM
      {
        int sIdx;

        if ( batch->flag & FLAG_STK_UP )
          sIdx = batch->noStacks - 1 - ss;
        else
          sIdx = ss;

        cuFfdotStack* cStack = &batch->stacks[sIdx];

        FOLD // Synchronisation  .
        {
          CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->srchStream, cStack->ifftComp,    0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
        }

        FOLD // Copy memory on the device  .
        {
          if ( batch->flag & FLAG_CUFFT_CB_OUT )
          {
            if ( batch->flag & FLAG_HALF )
            {
#if __CUDACC_VER__ >= 70500
              copyIFFTtoPln<half,half>( batch, cStack );
#else
              fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
              exit(EXIT_FAILURE);
#endif
            }
            else
            {
              copyIFFTtoPln<float, float>( batch, cStack );
            }
          }
          else
          {
            cmplxToPln( batch, cStack );
          }

          CUDA_SAFE_CALL(cudaGetLastError(), "At IFFT - copyToInMemPln");
        }

        FOLD // Synchronisation  .
        {
          CUDA_SAFE_CALL(cudaEventRecord(cStack->ifftMemComp, batch->srchStream),"Recording event: ifftMemComp");
        }
      }
    }
  }
}

/** Multiply and inverse FFT the complex f-∂f plane  .
 * This assumes the input data is ready and on the device
 * This creates a complex f-∂f plane
 */
void convolveBatch(cuFFdotBatch* batch)
{
  // Multiply
  if ( batch->rValues[0][0].numrs )
  {
    nvtxRangePush("Multiply");
#ifdef STPMSG
    printf("\tMultiply & FFT\n");
#endif

    if ( batch->flag & FLAG_CUFFT_CB_IN )   // Do the multiplication using a CUFFT callback (in my testing this is VERY slow!)  .
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

        // In my testing I found multiplying each plane separately works fastest so it is the "default"
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
                CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, cStack->ifftComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
              }
            }

            if ( !(batch->flag & FLAG_CUFFT_CB_OUT) )
            {
              // Have to wait for search to finish reading data
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, batch->searchComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
            }

            if ( (batch->retType & CU_STR_PLN) && !(batch->flag & FLAG_CUFFT_CB_OUT) )
            {
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, batch->candCpyComp, 0),   "Waiting for GPU to be ready to copy data to device.");   // Multiplication will change the plane
            }
          }

          FOLD // Call kernel  .
          {
#ifdef TIMING // Timing event  .
            CUDA_SAFE_CALL(cudaEventRecord(batch->multInit, batch->multStream),"Recording event: multInit");
#endif

            mult30_f(batch->multStream, batch);

            // Run message
            CUDA_SAFE_CALL(cudaGetLastError(), "At kernel launch");
          }

          FOLD // Synchronisation  .
          {
            CUDA_SAFE_CALL(cudaEventRecord(batch->multComp, batch->multStream),"Recording event: multComp");
          }
        }
        else if ( batch->flag & FLAG_MUL_STK   )  // Do the multiplications one stack  at a time  .
        {
          cuFfdotStack* pStack = NULL;  // Previous stack

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
              multStack(batch, cStack, sIdx, pStack);
            }

            FOLD // IFFT  .
            {
              if ( batch->flag & FLAG_CONV )
              {
                IFFTStack(batch, cStack, pStack);
              }
            }

            pStack = cStack;
          }
        }
        else if ( batch->flag & FLAG_MUL_PLN )    // Do the multiplications one plane  at a time  .
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

  // IFFT  .
  if ( batch->rValues[0][0].numrs )
  {
    if ( !( (batch->flag & FLAG_CONV) && (batch->flag & FLAG_MUL_STK) ) )
    {
      nvtxRangePush("IFFT");

#ifdef STPMSG
      printf("\t\tInverse FFT\n");
#endif

      cuFfdotStack* pStack = NULL;  // Previous stack

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

        FOLD // IFFT  .
        {
          IFFTStack(batch, cStack, pStack);
        }

        pStack = cStack;

#ifdef STPMSG
        printf("\t\t\tDone\n", sIdx);
#endif
      }

      nvtxRangePop();
    }
  }
}

