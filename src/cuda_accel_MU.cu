#include "cuda_accel_MU.h"

#define CPY_WIDTH 512  // 256  512 768

//====================================== Constant variables  ===============================================\\

#if CUDA_VERSION >= 6050
__device__ cufftCallbackLoadC  d_loadCallbackPtr    = CB_MultiplyInput;
__device__ cufftCallbackStoreC d_storePow_f         = CB_PowerOut_f;
__device__ cufftCallbackStoreC d_inmemRow_f         = CB_InmemOutRow_f;
__device__ cufftCallbackStoreC d_inmemPln_f         = CB_InmemOutPln_f;
#if CUDA_VERSION >= 7050
__device__ cufftCallbackStoreC d_storePow_h         = CB_PowerOut_h;
__device__ cufftCallbackStoreC d_inmemRow_h         = CB_InmemOutRow_h;
__device__ cufftCallbackStoreC d_inmemPln_h         = CB_InmemOutPln_h;
#endif
#endif

//======================================= Global variables  ================================================\\


//========================================== Functions  ====================================================\\

__device__ int calcInMemIdx_ROW( size_t offset )
{
  const int hw  = HWIDTH_STAGE[0];
  const int st  = STRIDE_STAGE[0];
  const int al  = ALEN ;
  int col       = ( offset % st ) - hw * ACCEL_NUMBETWEEN ;

  if ( col < 0 || col >= al )
    return -1;

  const int ns  = NO_STEPS;

  int row       =   offset  / ( st * ns ) ;
  int step      = ( offset  % ( st * ns ) ) / st;

  size_t plnOff = row * PLN_STRIDE + step * al + col;

  return plnOff;
}

__device__ int calcInMemIdx_PLN( size_t offset )
{
  const int hw  = HWIDTH_STAGE[0];
  const int st  = STRIDE_STAGE[0];
  const int al  = ALEN ;
  int col       = ( offset % st ) - hw * ACCEL_NUMBETWEEN ;

  if ( col < 0 || col >= al )
    return -1;

  const int ht  = HEIGHT_STAGE[0];

  int row       = offset  /   st;
  int step      = row     /   ht;
  row           = row     %   ht;  // Plane interleaved!

  size_t plnOff = row * PLN_STRIDE + step * al + col;

  return plnOff;
}


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

  if ( inf->flags & FLAG_ITLV_ROW )
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
__device__ void CB_PowerOut_f( void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  // Calculate power
  float power = element.x*element.x + element.y*element.y ;

  // Write result (offsets are the same)
  ((float*)dataOut)[offset] = power;
}

/** CUFFT callback kernel to calculate and store float powers in the in-memory plane, after the FFT  .
 *  CallerInfo is passed as the address of the first element of the first step in the inmem plane
 *
 */
__device__ void CB_InmemOutRow_f( void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  int plnOff = calcInMemIdx_ROW(offset);

  if ( plnOff == -1 )
  {
    // This element is in the contaminated ends
    return;
  }

  // Calculate power
  float power = element.x*element.x + element.y*element.y ;

  // Write result (offsets are the same)
  ((float*)dataOut)[ plnOff ] = power;
}

/** CUFFT callback kernel to calculate and store float powers in the in-memory plane, after the FFT  .
 *  CallerInfo is passed as the address of the first element of the first step in the inmem plane
 *
 */
__device__ void CB_InmemOutPln_f( void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  int plnOff = calcInMemIdx_PLN(offset);

  if ( plnOff == -1 )
  {
    // This element is in the contaminated ends
    return;
  }

  // Calculate power
  float power = element.x*element.x + element.y*element.y ;

  // Write result (offsets are the same)
  ((float*)dataOut)[ plnOff ] = power;
}


#if CUDA_VERSION >= 7050 // Half precision CUFFT power call back

/** CUFFT callback kernel to calculate and store half powers after the FFT  .
 */
__device__ void CB_PowerOut_h( void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  // Calculate power
  float power = element.x*element.x + element.y*element.y ;

  // Write result (offsets are the same)
  ((half*)dataOut)[offset] = __float2half(power);
}

/** CUFFT callback kernel to calculate and store half powers in the in-memory plane, after the FFT  .
 *  CallerInfo is passed as the address of the first element of the first step in the inmem plane
 *  Assumes row interleaved data
 *
 */
__device__ void CB_InmemOutRow_h( void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  int plnOff = calcInMemIdx_ROW(offset);

  if ( plnOff == -1 )
  {
    // This element is in the contaminated ends
    return;
  }

  // Calculate power
  float power = element.x*element.x + element.y*element.y ;

  // Write result (offsets are the same)
  ((half*)dataOut)[plnOff] = __float2half(power);
}

/** CUFFT callback kernel to calculate and store half powers in the in-memory plane, after the FFT  .
 *  CallerInfo is passed as the address of the first element of the first step in the inmem plane
 *  Assumes plane interleaved data
 *
 */
__device__ void CB_InmemOutPln_h( void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  int plnOff = calcInMemIdx_PLN(offset);

  if ( plnOff == -1 )
  {
    // This element is in the contaminated ends
    return;
  }

  // Calculate power
  float power = element.x*element.x + element.y*element.y ;

  // Write result (offsets are the same)
  ((half*)dataOut)[plnOff] = __float2half(power);
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
  if ( batch->flags & FLAG_CUFFT_CB_IN )
  {
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_ldCallbackPtr, d_loadCallbackPtr,  sizeof(cufftCallbackLoadC)),   "Getting constant memory address.");
  }

  if ( batch->flags & FLAG_CUFFT_CB_OUT )
  {
    if ( batch->flags & FLAG_HALF )
    {
#if CUDA_VERSION >= 7050
      if ( batch->flags & FLAG_CUFFT_CB_INMEM )
      {
        // Store powers to inmem plane
        if ( batch->flags & FLAG_ITLV_ROW )    // Row interleaved
        {
          CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_inmemRow_h, sizeof(cufftCallbackStoreC)),  "Getting constant memory address.");
        }
        else                                  // Plane interleaved
        {
          CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_inmemPln_h, sizeof(cufftCallbackStoreC)),  "Getting constant memory address.");
        }
      }
      else
      {
        // Calculate powers and write to powers half precision plane
        CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_storePow_h, sizeof(cufftCallbackStoreC)),  "Getting constant memory address.");
      }
#else
      fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
      exit(EXIT_FAILURE);
#endif
    }
    else
    {
      if ( batch->flags & FLAG_CUFFT_CB_INMEM )
      {
        // Store powers to inmem plane
        if ( batch->flags & FLAG_ITLV_ROW )    // Row interleaved
        {
          CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_inmemRow_f, sizeof(cufftCallbackStoreC)),  "Getting constant memory address.");
        }
        else                                  // Plane interleaved
        {
          CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_inmemPln_f, sizeof(cufftCallbackStoreC)),  "Getting constant memory address.");
        }
      }
      else
      {
        // Calculate powers and write to powers half single plane
        CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_storePow_f, sizeof(cufftCallbackStoreC)),  "Getting constant memory address.");
      }
    }
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

    if ( batch->flags & FLAG_STK_UP )
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

        // Set store FFT callback
        setCB(batch, cStack);

        FOLD // Set load FFT callback  .
        {
          CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_ldCallbackPtr, CUFFT_CB_LD_COMPLEX, (void**)&cStack->d_sInf ),"Error assigning CUFFT load callback.");
        }

        FOLD // Call the FFT  .
        {
          void* dst = getCBwriteLocation(batch, cStack);

          CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with multStream.");
          CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) cStack->d_planeMult, (cufftComplex *) dst, CUFFT_INVERSE),"Error executing CUFFT plan.");
        }
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

void* getCBwriteLocation(cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  void* dst = cStack->d_planePowr;

  if ( batch->flags &    FLAG_CUFFT_CB_INMEM )
  {
#if CUDA_VERSION >= 6050
    rVals* rVal   = &batch->rValues[0][0];

    if ( batch->flags &  FLAG_HALF )
    {
#if CUDA_VERSION >= 7050
      dst    = ((half*)batch->d_planeFull)    + rVal->step * batch->accelLen; // A pointer to the location of the first step in the inmeme plane
#else
      fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
      exit(EXIT_FAILURE);
#endif
    }
    else
    {
      dst    = ((float*)batch->d_planeFull)    + rVal->step * batch->accelLen; // A pointer to the location of the first step in the inmeme plane
    }
#else
    fprintf(stderr,"ERROR: CUFFT callbacks can only be used with CUDA 6.5 or later!\n");
    exit(EXIT_FAILURE);
#endif
  }

  return dst;
}

void setCB(cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  if ( batch->flags & FLAG_CUFFT_CB_OUT )
  {

#if CUDA_VERSION >= 6050

    void* dst;

    if ( batch->flags &    FLAG_CUFFT_CB_INMEM )
    {
      rVals* rVal   = &batch->rValues[0][0];

      if ( batch->flags &  FLAG_HALF )
      {
#if CUDA_VERSION >= 7050
        dst    = ((half*)batch->d_planeFull)    + rVal->step * batch->accelLen; // A pointer to the location of the first step in the inmeme plane
#else
        fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
        exit(EXIT_FAILURE);
#endif
      }
      else
      {
        dst    = ((float*)batch->d_planeFull)    + rVal->step * batch->accelLen; // A pointer to the location of the first step in the inmeme plane
      }
    }
    else
    {
      dst = cStack->d_planePowr;

			// Testing passing values in the actual pointer
      //uint width  = cStack->strideCmplx ;
      //uint skip   = cStack->kerStart ;
      //uint pass   = (width << 16) | skip ;
      //dst = (void*)pass;
    }

    CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&dst ),"Error assigning CUFFT store callback.");
#else
    fprintf(stderr,"ERROR: CUFFT callbacks can only be used with CUDA 6.5 or later!\n");
    exit(EXIT_FAILURE);
#endif
  }
}

/** Kernel to copy powers from complex plane to inmeme plane  .
 *
 */
template<typename T>
__global__ void cpyPowers_ker( T* dst, size_t  dpitch, T*  src, size_t  spitch, size_t  width, size_t  height)
{
  int ix = blockIdx.x * CPY_WIDTH + threadIdx.x ;

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
template<typename T>
__global__ void cpyCmplx_ker( T* dst, size_t  dpitch, fcomplexcu* src, size_t  spitch, size_t  width, size_t  height)
{
  int ix = blockIdx.x * CPY_WIDTH + threadIdx.x ;

  const int buffLen = 4;

  float buff[buffLen];

  if ( ix < width )
  {
    int iy;

    FOLD // All iterations with no height check
    {
      for ( iy = 0 ; iy < height - buffLen ; iy+=buffLen)
      {
        for ( int by = 0 ; by < buffLen; by++)
        {
          int gy = iy + by;

          buff[by]          = getPower(src, gy*spitch + ix);
        }

        for ( int by = 0 ; by < buffLen; by++)
        {
          int gy = iy + by;

          set(dst, gy*dpitch + ix, buff[by]);
        }
      }
    }

    FOLD // One last iteration with height checks
    {
      for ( int by = 0 ; by < buffLen; by++)
      {
        int gy = iy + by;

        if ( gy < height)
        {
          buff[by]          = getPower(src, gy*spitch + ix);
        }
      }

      for ( int by = 0 ; by < buffLen; by++)
      {
        int gy = iy + by;

        if ( gy < height)
        {
          set(dst, gy*dpitch + ix, buff[by]);
        }
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
template<typename T>
void cpyCmplx( T* dst, size_t  dpitch, fcomplexcu* src, size_t  spitch, size_t  width, size_t  height, cudaStream_t  stream)
{
  dim3 dimBlock, dimGrid;

  dimBlock.x  = CPY_WIDTH;
  dimBlock.y  = 1 ;

  float ww    = width  / (float)dimBlock.x ;

  dimGrid.x   = ceil(ww);
  dimGrid.y   = 1 ;

  cpyCmplx_ker<T><<<dimGrid,  dimBlock, 0, stream >>>(dst, dpitch, src, spitch, width, height);
}

/** Copy results of iFFT from powers plane to the inmem plane using 2D async memory copy
 *
 * This is done using one appropriately strided 2d memory copy for each step of a stack
 *
 */
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

  dpitch  = batch->sInf->pInf->inmemStride * outSz;
  width   = batch->accelLen * outSz;
  height  = cStack->height;
  spitch  = cStack->stridePower * inSz;

  // Error check
  if (cStack->noInStack > 1 )
  {
    fprintf(stderr,"ERROR: %s cannot handle stacks with more than one plane.\n", __FUNCTION__);
    exit(EXIT_FAILURE);
  }

  if ( batch->flags & FLAG_CUFFT_CB_INMEM )
  {
    // Copying was done by the callback directly
    return;
  }

  for ( int step = 0; step < batch->noSteps; step++ )
  {
    rVals* rVal = &batch->rValues[step][0];

    if ( rVal->numrs )
    {
      dst     = ((Tout*)batch->d_planeFull)    + rVal->step * batch->accelLen;

      if      ( batch->flags & FLAG_ITLV_ROW )
      {
        src     = ((Tin*)cStack->d_planePowr)  + cStack->stridePower*step + cStack->kerStart;
        spitch  = cStack->stridePower*batch->noSteps*inSz;
      }
      else
      {
        src     = ((Tin*)cStack->d_planePowr)  + cStack->stridePower*height*step + cStack->kerStart;
      }

      CUDA_SAFE_CALL(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, batch->srchStream ),"Calling cudaMemcpy2DAsync after IFFT.");
    }
  }
}

/** Copy results of the iFFT from powers plane to the inmem plane using a kernel  .
 *
 */
void cmplxToPln( cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  fcomplexcu*   src;

  size_t        dpitch;
  size_t        spitch;
  size_t        width;
  size_t        height;

  dpitch  = batch->sInf->pInf->inmemStride;
  width   = batch->accelLen;
  height  = cStack->height;
  spitch  = cStack->strideCmplx;

  // Error check
  if (cStack->noInStack > 1 )
  {
    fprintf(stderr,"ERROR: %s cannot handle stacks with more than one plane.\n", __FUNCTION__);
    exit(EXIT_FAILURE);
  }

  if ( batch->flags & FLAG_CUFFT_CB_INMEM )
  {
    // Copying was done by the callback directly
    return;
  }

  for ( int step = 0; step < batch->noSteps; step++ )
  {
    rVals* rVal = &batch->rValues[step][0];

    if ( rVal->numrs ) // Valid step
    {
      FOLD // Calculate striding info
      {
        // Source data location
        if ( batch->flags & FLAG_ITLV_ROW )
        {
          src     = ((fcomplexcu*)cStack->d_planePowr)  + cStack->strideCmplx*step + cStack->kerStart;
          spitch  = cStack->strideCmplx*batch->noSteps;
        }
        else
        {
          src     = ((fcomplexcu*)cStack->d_planePowr)  + cStack->strideCmplx*height*step + cStack->kerStart;
        }
      }

      if ( batch->flags & FLAG_HALF )
      {
#if CUDA_VERSION >= 7050
        // Each Step has its own start location in the inmem plane
        half *dst = ((half*)batch->d_planeFull)        + rVal->step * batch->accelLen;

        // Call kernel
        cpyCmplx<half>(dst, dpitch, src, spitch,  width,  height, batch->srchStream );
#else
        fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
        exit(EXIT_FAILURE);
#endif
      }
      else
      {
        // Each Step has its own start location in the inmem plane
        float *dst  = ((float*)batch->d_planeFull)        + rVal->step * batch->accelLen;

        // Call kernel
        cpyCmplx<float>(dst, dpitch, src, spitch,  width,  height, batch->srchStream );
      }

    }
  }
}

void multStack(cuFFdotBatch* batch, cuFfdotStack* cStack, int sIdx, cuFfdotStack* pStack = NULL)
{
  FOLD // Synchronisation  .
  {
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->prepComp,    0), "Waiting for GPU to be ready to copy data to device.");  // Need input data

    // iFFT has its own data so can start once iFFT is complete
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->multStream, cStack->ifftComp,    0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

    if ( (batch->retType & CU_STR_PLN) && !(batch->flags & FLAG_CUFFT_CB_OUT) )
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
    if      ( cStack->flags & FLAG_MUL_00 )
    {
      mult00(cStack->multStream, batch, sIdx);
    }
    else if ( cStack->flags & FLAG_MUL_21 )
    {
      mult21_f(cStack->multStream, batch, sIdx);
    }
    else if ( cStack->flags & FLAG_MUL_22 )
    {
      mult22_f(cStack->multStream, batch, sIdx);
    }
    else if ( cStack->flags & FLAG_MUL_23 )
    {
      mult23_f(cStack->multStream, batch, sIdx);
    }
    else
    {
      fprintf(stderr,"ERROR: No valid stack multiplication specified. Line %i in %s.\n", __LINE__, __FILE__);
      exit(EXIT_FAILURE);
    }

    // Run message
    CUDA_SAFE_CALL(cudaGetLastError(), "At multiplication kernel launch.");
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

    if ( batch->flags & FLAG_CUFFT_CB_IN )   // Do the multiplication using a CUFFT callback  .
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
        if      ( batch->flags & FLAG_MUL_BATCH )  // Do the multiplications one family at a time  .
        {
          FOLD // Synchronisation  .
          {
            // Synchronise input data preparation for all stacks
            for (int synchIdx = 0; synchIdx < batch->noStacks; synchIdx++)
            {
              cuFfdotStack* cStack = &batch->stacks[synchIdx];

              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, cStack->prepComp, 0),     "Waiting for GPU to be ready to copy data to device.");    // Need input data

              // iFFT has its own data so can start once iFFT is complete
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, cStack->ifftComp, 0),     "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
            }

            if ( !(batch->flags & FLAG_CUFFT_CB_OUT) )
            {
              // Have to wait for search to finish reading data
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, batch->searchComp, 0),    "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
            }

            if ( (batch->retType & CU_STR_PLN) && !(batch->flags & FLAG_CUFFT_CB_OUT) )
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
        else if ( batch->flags & FLAG_MUL_STK   )  // Do the multiplications one stack  at a time  .
        {
          cuFfdotStack* pStack = NULL;  // Previous stack

          // Multiply this entire stack in one block
          for (int ss = 0; ss < batch->noStacks; ss++)
          {
            int sIdx;

            if ( batch->flags & FLAG_STK_UP )
              sIdx = batch->noStacks - 1 - ss;
            else
              sIdx = ss;

            cuFfdotStack* cStack = &batch->stacks[sIdx];

            multStack(batch, cStack, sIdx, pStack);

            pStack = cStack;
          }
        }
        else if ( batch->flags & FLAG_MUL_PLN )    // Do the multiplications one plane  at a time  .
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

    // Wait for previous iFFT to finish
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack->ifftComp,      0), "Waiting for GPU to be ready to copy data to device.");
    if ( batch->flags & FLAG_SS_INMEM  )
    {
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, cStack->ifftMemComp, 0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
    }

    // Wait for previous search to finish
    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, batch->searchComp,     0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete

    if ( (batch->retType & CU_STR_PLN) && (batch->flags & FLAG_CUFFT_CB_OUT) )
    {
      CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftPStream, batch->candCpyComp,  0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
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

      setCB(batch, cStack);

      FOLD // Call the FFT  .
      {
        void* dst = getCBwriteLocation(batch, cStack);

        CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->fftPStream),  "Error associating a CUFFT plan with multStream.");
        CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) cStack->d_planeMult, (cufftComplex *) dst, CUFFT_INVERSE),"Error executing CUFFT plan.");
      }
    }
  }

  FOLD // Synchronisation  .
  {
    cudaEventRecord(cStack->ifftComp, cStack->fftPStream);

    // If using power calculate call back with the inmem plane
    if ( batch->flags & FLAG_CUFFT_CB_INMEM )
    {
#if CUDA_VERSION >= 6050
      cudaEventRecord(cStack->ifftMemComp, cStack->fftPStream);
    }
#endif
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

      if ( batch->flags & FLAG_STK_UP )
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

/**
 *
 */
void copyToInMemPln(cuFFdotBatch* batch)
{
  if ( batch->rValues[0][0].numrs )
  {
    if ( batch->flags & FLAG_SS_INMEM )
    {
      if ( batch->flags & FLAG_CUFFT_CB_INMEM )
      {
        // Copying was done by the callback directly
        return;
      }

      // Error check
      if (batch->noStacks > 1 )
      {
        fprintf(stderr,"ERROR: %s cannot handle a family with more than one plane.\n", __FUNCTION__);
        exit(EXIT_FAILURE);
      }

      FOLD // Copy back data  .
      {
        cuFfdotStack* cStack = &batch->stacks[0];

        FOLD // Synchronisation  .
        {
          CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->srchStream, cStack->ifftComp,    0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
        }

        FOLD // Copy memory on the device  .
        {
          if ( batch->flags & FLAG_CUFFT_CB_POW )
          {
            // Copy memory using a 2D async memory copy
            if ( batch->flags & FLAG_HALF )
            {
#if CUDA_VERSION >= 7050
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
            // Use kernel to copy powers from powers plane to the inmem plane
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

    if ( batch->flags & FLAG_CUFFT_CB_IN )   // Do the multiplication using a CUFFT callback (in my testing this is VERY slow!)  .
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
        if      ( batch->flags & FLAG_MUL_BATCH )  // Do the multiplications one family at a time  .
        {
          FOLD // Synchronisation  .
          {
            // Synchronise input data preparation for all stacks
            for (int synchIdx = 0; synchIdx < batch->noStacks; synchIdx++)
            {
              cuFfdotStack* cStack = &batch->stacks[synchIdx];

              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, cStack->prepComp,0),   "Waiting for GPU to be ready to copy data to device.");    // Need input data

              // iFFT has its own data so can start once iFFT is complete
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, cStack->ifftComp, 0),  "Waiting for GPU to be ready to copy data to device.");
            }

            if ( !(batch->flags & FLAG_CUFFT_CB_OUT) )
            {
              // Have to wait for search to finish reading data
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, batch->searchComp, 0),  "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
            }

            if ( (batch->retType & CU_STR_PLN) && !(batch->flags & FLAG_CUFFT_CB_OUT) )
            {
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->multStream, batch->candCpyComp, 0), "Waiting for GPU to be ready to copy data to device.");   // Multiplication will change the plane
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
        else if ( batch->flags & FLAG_MUL_STK   )  // Do the multiplications one stack  at a time  .
        {
          cuFfdotStack* pStack = NULL;  // Previous stack

          // Multiply this entire stack in one block
          for (int ss = 0; ss < batch->noStacks; ss++)
          {
            int sIdx;

            if ( batch->flags & FLAG_STK_UP )
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
              if ( batch->flags & FLAG_CONV )
              {
                IFFTStack(batch, cStack, pStack);
              }
            }

            pStack = cStack;
          }
        }
        else if ( batch->flags & FLAG_MUL_PLN )    // Do the multiplications one plane  at a time  .
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
    if ( !( (batch->flags & FLAG_CONV) && (batch->flags & FLAG_MUL_STK) ) )
    {
      nvtxRangePush("IFFT");

#ifdef STPMSG
      printf("\t\tInverse FFT\n");
#endif

      cuFfdotStack* pStack = NULL;  // Previous stack

      for (int ss = 0; ss < batch->noStacks; ss++)
      {
        int sIdx;

        if ( batch->flags & FLAG_STK_UP )
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

