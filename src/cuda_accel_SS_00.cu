#include "cuda_accel_SS.h"

#define SS00_X           16                   // X Thread Block
#define SS00_Y           8                    // Y Thread Block
#define SS00BS           (SS00_X*SS00_Y)

#ifdef WITH_SAS_00	// loop down column  - Read memory  .

/** Sum and Search memory access only family order - loop down column  .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSteps
 */
template<typename T, int noBatch >
__global__ void add_and_searchCU00_k(const uint width, candPZs* d_cands, int oStride, vHarmList powersArr, const int noHarms, const int noStages, const int noSteps )
{
  const int bidx  = threadIdx.y * SS00_X  +  threadIdx.x;   /// Block index
  const int tid   = blockIdx.x  * SS00BS  +  bidx;          /// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    T* array[MAX_HARM_NO];                                          ///< A pointer array

    // Set the values of the pointer array
    for ( int i = 0; i < noHarms; i++)
    {
      array[i] = (T*)powersArr[i];
    }

    FOLD  // Set the local and return candidate powers to zero  .
    {
      int xStride = noSteps*oStride ;

      for ( int stage = 0; stage < noStages; stage++ )
      {
        for ( int step = 0; step < noSteps; step++)               // Loop over steps
        {
          d_cands[stage*gridDim.y*xStride + blockIdx.y*xStride + step*ALEN + tid].value = 0;
        }
      }
    }

    for ( int harm = 0; harm < noHarms ; harm++)                // Loop over planes  .
    {
      int maxW          = ceilf(width * FRAC_HARM[harm]);

      if ( tid < maxW )
      {
        //float*  t       = powersArr[harm];
        float tSum      = 0;
        uint  nHeight   = HEIGHT_HARM[harm] * noSteps;
        int   stride    = STRIDE_HARM[harm];
        int   lDepth    = ceilf(nHeight/(float)gridDim.y);
        int   y0        = lDepth*blockIdx.y;
        int   y1        = MIN(y0+lDepth, nHeight);

        FOLD // Read data from planes  .
        {
          for ( int y = y0; y < y1; y++ )
          {
            int idx     = (y) * stride;

            FOLD // Read  .
            {
              tSum += getPower(array[harm], tid + idx );
            }
          }
        }

        if ( tSum < 0 )	// This should never be the case but needed so the compiler doesn't optimise out the sum
        {
          printf("add_and_searchCU00_k tSum < 0 tid: %04i ???\n", tid);
        }
      }
    }
  }
}

__host__ void add_and_searchCU00_f(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int   noStages  = log2((double)batch->noGenHarms) + 1 ;
  vHarmList  powers;

  for (int i = 0; i < batch->noGenHarms; i++)
  {
    int idx         = i; // Family order
    powers.val[i]   = batch->planes[idx].d_planePowr;
  }

  if      ( batch->flags & FLAG_POW_HALF         )
  {
#if CUDA_VERSION >= 7050
    add_and_searchCU00_k< half, 0>        <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, batch->noGenHarms, noStages, batch->noSteps  );
#else
    fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
    exit(EXIT_FAILURE);
#endif
  }
  else if ( batch->flags & FLAG_CUFFT_CB_POW )
  {
    add_and_searchCU00_k< float, 0>       <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, batch->noGenHarms, noStages, batch->noSteps  );
  }
  else
  {
    add_and_searchCU00_k< fcomplexcu, 0>  <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, batch->noGenHarms, noStages, batch->noSteps  );
  }
}

#endif // WITH_SAS_00

#ifdef WITH_SAS_01	// Loop down column  -  Read only needed memory  .

/** Sum and Search memory access only stage order - loop down column  -  Read only needed memory  .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSteps
 */
template<typename T, int noBatch >
__global__ void add_and_searchCU01_k(const uint width, candPZs* d_cands, int oStride, vHarmList powersArr, const int noHarms, const int noStages, const int noSteps )
{
  const int bidx  = threadIdx.y * SS00_X  +  threadIdx.x;	/// Block index
  const int tid   = blockIdx.x  * SS00BS  +  bidx;		/// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    T* array[MAX_HARM_NO];					///< A pointer array

    // Set the values of the pointer array
    for ( int i = 0; i < noHarms; i++)
    {
      array[i] = (T*)powersArr[i];
    }

    FOLD  // Set the local and return candidate powers to zero
    {
      int xStride = noSteps*oStride ;

      for ( int stage = 0; stage < noStages; stage++ )
      {
        for ( int step = 0; step < noSteps; step++)		// Loop over steps
        {
          d_cands[stage*gridDim.y*xStride + blockIdx.y*xStride + step*ALEN + tid].value = 0;
        }
      }
    }

    for ( int harm = 0; harm < noHarms ; harm++)		// Loop over planes
    {
      int maxW      = ceilf(width * FRAC_STAGE[harm]);
      int stride    = STRIDE_STAGE[harm];

      if ( tid < maxW )
      {
        uint nHeight  = HEIGHT_STAGE[harm] * noSteps;
        float tSum    = 0;
        int   lDepth    = ceilf(nHeight/(float)gridDim.y);
        int   y0        = lDepth*blockIdx.y;
        int   y1        = MIN(y0+lDepth, nHeight);

        for ( int y = y0; y < y1; y++ )
        {
          int idx  = (y) * stride;

          FOLD // Read  .
          {
            tSum += getPower(array[harm], tid + idx );
          }
        }

        if ( tSum < 0 )	// This should never be the case but needed so the compiler doesn't optimise out the sum
        {
          printf("add_and_searchCU01_k");
        }
      }
    }
  }
}

__host__ void add_and_searchCU01_f(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int   noStages  = log2((double)batch->noGenHarms) + 1 ;
  vHarmList  powers;

  for (int i = 0; i < batch->noGenHarms; i++)
  {
    int sIdx        = batch->cuSrch->sIdx[i]; // Stage order
    powers.val[i]   = batch->planes[sIdx].d_planePowr;
  }

  if      ( batch->flags & FLAG_POW_HALF         )
  {
#if CUDA_VERSION >= 7050
    add_and_searchCU01_k< half, 0>        <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, batch->noGenHarms, noStages, batch->noSteps  );
#else
    fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
    exit(EXIT_FAILURE);
#endif
  }
  else if ( batch->flags & FLAG_CUFFT_CB_POW )
  {
    add_and_searchCU01_k< float, 0>       <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, batch->noGenHarms, noStages, batch->noSteps  );
  }
  else
  {
    add_and_searchCU01_k< fcomplexcu, 0>  <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, batch->noGenHarms, noStages, batch->noSteps  );
  }
}

#endif // WITH_SAS_01

#ifdef WITH_SAS_02	// Loop down column  -  Read extra memory  .

/** Sum and Search memory access only stage order - loop down column  -  Read extra memory  .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSteps
 */
template<typename T, int noBatch >
__global__ void add_and_searchCU02_k(const uint width, candPZs* d_cands, int oStride, vHarmList powersArr, const int noHarms, const int noStages, const int noSteps )
{
  const int bidx  = threadIdx.y * SS00_X  +  threadIdx.x;	/// Block index
  const int tid   = blockIdx.x  * SS00BS  +  bidx;		/// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    T* array = (T*)powersArr[0];				///< A pointer array

    FOLD  // Set the local and return candidate powers to zero  .
    {
      int xStride = noSteps*oStride ;

      for ( int stage = 0; stage < noStages; stage++ )
      {
        for ( int step = 0; step < noSteps; step++)		// Loop over steps
        {
          d_cands[stage*gridDim.y*xStride + blockIdx.y*xStride + step*ALEN + tid].value = 0;
        }
      }
    }

    for ( int harm = 0; harm < noHarms ; harm++)		// Loop over planes  .
    {
      int maxW      = ceilf(width * FRAC_STAGE[0]);
      int stride    = STRIDE_STAGE[0];

      if ( tid < maxW )
      {
        uint nHeight = HEIGHT_STAGE[0] * noSteps;
        float tSum = 0;
        int   lDepth    = ceilf(nHeight/(float)gridDim.y);
        int   y0        = lDepth*blockIdx.y;
        int   y1        = MIN(y0+lDepth, nHeight);

        FOLD // Read data from planes  .
        {
          for ( int y = y0; y < y1; y++ )
          {
            int idx  = (y) * stride ;

            FOLD // Read  .
            {
              tSum += getPower(array, tid + idx );
            }
          }
        }

        if ( tSum < 0 )	// This should never be the case but needed so the compiler doesn't optimise out the sum
        {
          printf("add_and_searchCU02_k");
        }
      }
    }
  }
}

__host__ void add_and_searchCU02_f(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int   noStages  = log2((double)batch->noGenHarms) + 1 ;
  vHarmList   powers;

  for (int i = 0; i < batch->noGenHarms; i++)
  {
    int sIdx        = batch->cuSrch->sIdx[i]; // Stage order
    powers.val[i]   = batch->planes[sIdx].d_planePowr;
  }

  if      ( batch->flags & FLAG_POW_HALF         )
  {
#if CUDA_VERSION >= 7050
    add_and_searchCU02_k< half, 0>        <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, batch->noGenHarms, noStages, batch->noSteps  );
#else
    fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
    exit(EXIT_FAILURE);
#endif
  }
  else if ( batch->flags & FLAG_CUFFT_CB_POW )
  {
    add_and_searchCU02_k< float, 0>       <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, batch->noGenHarms, noStages, batch->noSteps  );
  }
  else
  {
    add_and_searchCU02_k< fcomplexcu, 0>  <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_outData1, batch->strideOut, powers, batch->noGenHarms, noStages, batch->noSteps  );
  }

}

#endif // WITH_SAS_02

__host__ void add_and_searchCU00(cudaStream_t stream, cuFFdotBatch* batch )
{
  dim3 dimBlock, dimGrid;

  dimBlock.x  = SS00_X;
  dimBlock.y  = SS00_Y;

  float bw    = SS00BS ;
  float ww    = batch->accelLen / ( bw );

  dimGrid.x   = ceil(ww);
  dimGrid.y   = batch->ssSlices;

  if ( 0 )
  {
    // Dummy
  }
#ifdef WITH_SAS_01 // Stage order  .
  else
  {
    add_and_searchCU01_f(dimGrid,dimBlock,stream, batch );
  }
#endif
#ifdef WITH_SAS_00
  else if ( 1 )
  {
    add_and_searchCU00_f(dimGrid,dimBlock,stream, batch );
  }
#endif
  else
  {
    fprintf(stderr, "ERROR: Code has not been compiled with Sum & Search \"optimal\" kernel." );
    exit(EXIT_FAILURE);
  }

}

