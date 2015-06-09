#include "cuda_accel_SS.h"

#define SS33_X           16                    // X Thread Block
#define SS33_Y           8                     // Y Thread Block

/** Sum and Search - loop down - column max - multi-step - step outer .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSteps
 */
template<uint FLAGS, int noBatch >
__global__ void add_and_searchCU00_k(const uint width, accelcandBasic* d_cands, fsHarmList powersArr, cHarmList cmplxArr, const int noHarms, const int noStages, const int noSteps )
{
  const int bidx  = threadIdx.y * SS33_X          +  threadIdx.x;   /// Block index
  const int tid   = blockIdx.x  * (SS33_Y*SS33_X) +  bidx;          /// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    int oStride    = STRIDE_HARM[0];
    FOLD  // Set the local and return candidate powers to zero  .
    {
      for ( int stage = 0; stage < noStages; stage++ )
      {
        for ( int step = 0; step < noSteps; step++)               // Loop over steps
        {
          d_cands[step*noStages*oStride + stage*oStride + tid ].sigma = 0;
        }
      }
    }

    float batch[noBatch];
    for ( int harm = 0; harm < noHarms ; harm++)  // Loop over plains  .
    {
      int maxW      = ceilf(width * FRAC_HARM[harm]);
      int stride    = STRIDE_HARM[harm];

      if ( tid < maxW )
      {
        uint nHeight = HEIGHT_HARM[harm] * noSteps;

        FOLD // Read data from plains  .
        {
          //for ( int y = 0; y < nHeight; y++ )
          for ( int yBase = 0; yBase < nHeight; yBase += noBatch )
          {
            for ( int yPlus = 0; yPlus < noBatch; yPlus++ )
            {
              int idx  = (yBase+yPlus) * stride;

              FOLD // Read  .
              {
                if      ( FLAGS & FLAG_MUL_CB_OUT )
                {
                  float cmpf            = powersArr[harm][ tid + idx ];
                  batch[yPlus]          = cmpf;
                }
                else
                {
                  fcomplexcu cmpc       = cmplxArr[harm][ tid + idx ];
                  batch[yPlus]          = cmpc.r * cmpc.r + cmpc.i * cmpc.i ;
                }
              }
            }
            for ( int yPlus = 0; yPlus < noBatch; yPlus++ )
            {
              if (yPlus + yBase < nHeight )
              {
                if ( batch[yPlus] < 0 )
                {
                  printf("SS\n");
                }
              }
            }
          }
        }
      }
    }
  }
}

/** Sum and Search - loop down - column max - multi-step - step outer .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSteps
 */
template<uint FLAGS, int noBatch >
__global__ void add_and_searchCU01_k(const uint width, accelcandBasic* d_cands, fsHarmList powersArr, cHarmList cmplxArr, const int noHarms, const int noStages, const int noSteps )
{
  const int bidx  = threadIdx.y * SS33_X          +  threadIdx.x;   /// Block index
  const int tid   = blockIdx.x  * (SS33_Y*SS33_X) +  bidx;          /// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    int oStride    = STRIDE_STAGE[0];
    FOLD  // Set the local and return candidate powers to zero
    {
      for ( int stage = 0; stage < noStages; stage++ )
      {
        for ( int step = 0; step < noSteps; step++)               // Loop over steps
        {
          d_cands[step*noStages*oStride + stage*oStride + tid ].sigma = 0;
        }
      }
    }

    float batch[noBatch];
    for ( int harm = 0; harm < noHarms ; harm++)  // Loop over plains
    {
      int maxW      = ceilf(width * FRAC_STAGE[harm]);
      int stride    = STRIDE_STAGE[harm];

      if ( tid < maxW )
      {
        uint nHeight = HEIGHT_STAGE[harm] * noSteps;

        FOLD // Read data from plains  .
        {
          //for ( int y = 0; y < nHeight; y++ )
          for ( int yBase = 0; yBase < nHeight; yBase+=noBatch )
          {
            for ( int yPlus = 0; yPlus < noBatch; yPlus++ )
            {
              int idx  = (yBase+yPlus) * stride;

              FOLD // Read  .
              {
                if      ( FLAGS & FLAG_MUL_CB_OUT )
                {
                  float cmpf            = powersArr[harm][ tid + idx ];
                  batch[yPlus]          = cmpf;
                }
                else
                {
                  fcomplexcu cmpc       = cmplxArr[harm][ tid + idx ];
                  batch[yPlus]          = cmpc.r * cmpc.r + cmpc.i * cmpc.i ;
                }
              }
            }

            for ( int yPlus = 0; yPlus < noBatch; yPlus++ )
            {
              if (yPlus + yBase < nHeight )
              {
                if ( batch[yPlus] < 0 )
                {
                  printf("SS\n");
                }
              }
            }
          }
        }
      }
    }
  }
}

/** Sum and Search - loop down - column max - multi-step - step outer .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSteps
 */
template<uint FLAGS, int noBatch >
__global__ void add_and_searchCU02_k(const uint width, accelcandBasic* d_cands, fsHarmList powersArr, cHarmList cmplxArr, const int noHarms, const int noStages, const int noSteps )
{
  const int bidx  = threadIdx.y * SS33_X          +  threadIdx.x;   /// Block index
  const int tid   = blockIdx.x  * (SS33_Y*SS33_X) +  bidx;          /// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    int oStride    = STRIDE_STAGE[0];

    FOLD  // Set the local and return candidate powers to zero  .
    {
      for ( int stage = 0; stage < noStages; stage++ )
      {
        for ( int step = 0; step < noSteps; step++)               // Loop over steps
        {
          d_cands[step*noStages*oStride + stage*oStride + tid ].sigma = 0;
        }
      }
    }

    float batch[noBatch];
    for ( int harm = 0; harm < noHarms ; harm++)  // Loop over plains  .
    {
      int maxW      = ceilf(width * FRAC_STAGE[0]);
      int stride    = STRIDE_STAGE[0];

      if ( tid < maxW )
      {
        uint nHeight = HEIGHT_STAGE[0] * noSteps;

        FOLD // Read data from plains  .
        {
          for ( int yBase = 0; yBase < nHeight; yBase += noBatch )
          {
            for ( int yPlus = 0; yPlus < noBatch; yPlus++ )
            {
              int idx  = (yBase+yPlus) * stride ;

              FOLD // Read  .
              {
                if      ( FLAGS & FLAG_MUL_CB_OUT )
                {
                  float cmpf            = powersArr[0][ tid + idx ];
                  batch[yPlus]          = cmpf;
                }
                else
                {
                  fcomplexcu cmpc       = cmplxArr[0][ tid + idx ];
                  batch[yPlus]          = cmpc.r * cmpc.r + cmpc.i * cmpc.i ;
                }
              }
            }

            for ( int yPlus = 0; yPlus < noBatch; yPlus++ )
            {
              if (yPlus + yBase < nHeight )
              {
                if ( batch[yPlus] < 0 )
                {
                  printf("SS\n");
                }
              }
            }
          }
        }
      }
    }
  }
}

__host__ void add_and_searchCU00(cudaStream_t stream, cuFFdotBatch* batch )
{
  dim3 dimBlock, dimGrid;
  const uint FLAGS = batch->flag;
  const int noStages = log2((double)batch->noHarms) + 1 ;

  dimBlock.x  = SS33_X;
  dimBlock.y  = SS33_Y;

  float bw    = SS33_X * SS33_Y;
  float ww    = batch->accelLen / ( bw );

  dimGrid.x   = ceil(ww);
  dimGrid.y   = 1;

  fsHarmList powers;
  cHarmList   cmplx;

  if ( 1 )  // Stage order
  {
    for (int i = 0; i < batch->noHarms; i++)
    {
      int idx         = batch->stageIdx[i]; // Stage order
      powers.val[i]   = batch->plains[idx].d_plainPowers;
      cmplx.val[i]    = batch->plains[idx].d_plainData;
    }

    if        ( FLAGS & FLAG_MUL_CB_OUT )
    {
      add_and_searchCU02_k<FLAG_MUL_CB_OUT, 12> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
    }
    else
    {
      add_and_searchCU02_k< 0, 12 > <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
    }
  }
  else
  {

    for (int i = 0; i < batch->noHarms; i++)
    {
      int idx = i;
      powers.val[i]   = batch->plains[idx].d_plainPowers;
      cmplx.val[i]    = batch->plains[idx].d_plainData;
    }

    if        ( FLAGS & FLAG_MUL_CB_OUT )
    {
      add_and_searchCU00_k<FLAG_MUL_CB_OUT, 12> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
    }
    else
    {
      add_and_searchCU00_k< 0, 12 > <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
    }
  }

}

