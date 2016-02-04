#include "cuda_accel_MU.h"


/** Kernel to copy powers from complex plane to in-mem plane  .
 *
 * One thread per column
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

/** Kernel to copy powers from complex plane to in-mem plane  .
 *
 * One thread per column
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

/** Function to call the kernel to copy powers from powers plane to in-mem plane  .
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

/** Function to call the kernel to copy powers from powers plane to in-mem plane  .
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

  dpitch  = batch->sInf->inmemStride * outSz;
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
    rVals* rVal = &(*batch->rAraays)[batch->rActive][step][0];

    if ( rVal->numrs )
    {
      dst     = ((Tout*)batch->sInf->d_planeFull)    + rVal->step * batch->accelLen;

      if      ( batch->flags & FLAG_ITLV_ROW )
      {
        src     = ((Tin*)cStack->d_planePowr)  + cStack->stridePower*step + cStack->harmInf->kerStart;
        spitch  = cStack->stridePower*batch->noSteps*inSz;
      }
      else
      {
        src     = ((Tin*)cStack->d_planePowr)  + cStack->stridePower*height*step + cStack->harmInf->kerStart;
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

  dpitch  = batch->sInf->inmemStride;
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
    rVals* rVal = &(*batch->rAraays)[batch->rActive][step][0];

    if ( rVal->numrs ) // Valid step
    {
      FOLD // Calculate striding info
      {
        // Source data location
        if ( batch->flags & FLAG_ITLV_ROW )
        {
          src     = ((fcomplexcu*)cStack->d_planePowr)  + cStack->strideCmplx*step + cStack->harmInf->kerStart;
          spitch  = cStack->strideCmplx*batch->noSteps;
        }
        else
        {
          src     = ((fcomplexcu*)cStack->d_planePowr)  + cStack->strideCmplx*height*step + cStack->harmInf->kerStart;
        }
      }

      if ( batch->flags & FLAG_HALF )
      {
#if CUDA_VERSION >= 7050
        // Each Step has its own start location in the inmem plane
        half *dst = ((half*)batch->sInf->d_planeFull)        + rVal->step * batch->accelLen;

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
        float *dst  = ((float*)batch->sInf->d_planeFull)        + rVal->step * batch->accelLen;

        // Call kernel
        cpyCmplx<float>(dst, dpitch, src, spitch,  width,  height, batch->srchStream );
      }

    }
  }
}

/** Copy the complex plane to the in-memory plane  .
 *
 */
void copyToInMemPln(cuFFdotBatch* batch)
{
  // Timing
  if ( (batch->flags & FLAG_TIME) )
  {
    if ( (*batch->rAraays)[batch->rActive+1][0][0].numrs )
    {
      for (int stack = 0; stack < batch->noStacks; stack++)
      {
        cuFfdotStack* cStack = &batch->stacks[stack];
        timeEvents( cStack->ifftMemInit, cStack->ifftMemComp, &batch->copyToPlnTime[stack],  "Copy to full plane");
      }
    }
  }

  if ( (*batch->rAraays)[batch->rActive][0][0].numrs )
  {
    if ( batch->flags & FLAG_SS_INMEM )
    {
      infoMSG(2,2,"Copy to in-mem plane\n");

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
          infoMSG(3,4,"pre synchronisation\n");

          CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->srchStream, cStack->ifftComp,    0), "Waiting for GPU to be ready to copy data to device.");  // This will overwrite the plane so search must be compete
        }

        FOLD // Timing event  .
        {
          if ( batch->flags & FLAG_TIME )
          {
            CUDA_SAFE_CALL(cudaEventRecord(cStack->ifftMemInit, batch->srchStream),"Recording event: multInit");
          }
        }

        FOLD // Copy memory on the device  .
        {
          if ( batch->flags & FLAG_CUFFT_CB_POW )
          {
            infoMSG(3,4,"2D async memory copy\n");

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
            infoMSG(3,4,"kernel memory copy\n");

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

      FOLD // Blocking if synchronises  .
      {
        if ( batch->flags & FLAG_SYNCH )
        {
          infoMSG(3,4,"post synchronisation [blocking] ifftMemComp\n");

          cuFfdotStack* cStack = &batch->stacks[0];

          nvtxRangePush("EventSynch");
          CUDA_SAFE_CALL(cudaEventSynchronize(cStack->ifftMemComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
          nvtxRangePop();
        }
      }
    }
  }
}