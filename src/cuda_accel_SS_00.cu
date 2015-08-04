#include "cuda_accel_SS.h"

#define SS00_X           16                   // X Thread Block
#define SS00_Y           8                    // Y Thread Block
#define SS00BS           (SS00_X*SS00_Y)

/** Sum and Search memory access only family order - loop down column  .
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
  const int bidx  = threadIdx.y * SS00_X  +  threadIdx.x;   /// Block index
  const int tid   = blockIdx.x  * SS00BS  +  bidx;          /// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    FOLD  // Set the local and return candidate powers to zero  .
    {
      int oStride    = STRIDE_HARM[0];

      for ( int stage = 0; stage < noStages; stage++ )
      {
        for ( int step = 0; step < noSteps; step++)               // Loop over steps
        {
          d_cands[blockIdx.y*noSteps*noStages*oStride + step*noStages*oStride + stage*oStride + tid ].sigma = 0;
        }
      }
    }

    for ( int harm = 0; harm < noHarms ; harm++)                // Loop over plains  .
    {
      int maxW          = ceilf(width * FRAC_HARM[harm]);

      if ( tid < maxW )
      {
        float*  t       = powersArr[harm];
        float tSum      = 0;
        uint  nHeight   = HEIGHT_HARM[harm] * noSteps;
        int   stride    = STRIDE_HARM[harm];
        int   lDepth    = ceilf(nHeight/(float)gridDim.y);
        int   y0        = lDepth*blockIdx.y;
        int   y1        = MIN(y0+lDepth, nHeight);

        FOLD // Read data from plains  .
        {
          for ( int y = y0; y < y1; y++ )
          {
            int idx     = (y) * stride;

            FOLD // Read  .
            {
              if      ( FLAGS & FLAG_MUL_CB_OUT )
              {
                tSum                  += t[ tid + idx ];
              }
              else
              {
                fcomplexcu cmpc       = cmplxArr[harm][ tid + idx ];
                tSum                  += cmpc.r * cmpc.r + cmpc.i * cmpc.i ;
              }
            }
          }
        }

        if (tSum < 0 )
        {
          printf("add_and_searchCU00_k");
        }
      }
    }
  }
}

/** Sum and Search memory access only stage order - loop down column  -  Read only needed memory  .
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
  const int bidx  = threadIdx.y * SS00_X  +  threadIdx.x;   /// Block index
  const int tid   = blockIdx.x  * SS00BS  +  bidx;          /// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    FOLD  // Set the local and return candidate powers to zero
    {
      int oStride    = STRIDE_STAGE[0];

      for ( int stage = 0; stage < noStages; stage++ )
      {
        for ( int step = 0; step < noSteps; step++)               // Loop over steps
        {
          d_cands[blockIdx.y*noSteps*noStages*oStride + step*noStages*oStride + stage*oStride + tid ].sigma = 0;
        }
      }
    }

    for ( int harm = 0; harm < noHarms ; harm++)  // Loop over plains
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
            if      ( FLAGS & FLAG_MUL_CB_OUT )
            {
              float cmpf            = powersArr[harm][ tid + idx ];
              tSum                  += cmpf;
            }
            else
            {
              fcomplexcu cmpc       = cmplxArr[harm][ tid + idx ];
              tSum                  += cmpc.r * cmpc.r + cmpc.i * cmpc.i ;
            }
          }
        }

        if ( tSum < 0 )
        {
          printf("add_and_searchCU01_k");
        }
      }
    }
  }
}

/** Sum and Search memory access only stage order - loop down column  -  Read extra memory  .
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
  const int bidx  = threadIdx.y * SS00_X  +  threadIdx.x;   /// Block index
  const int tid   = blockIdx.x  * SS00BS  +  bidx;          /// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    FOLD  // Set the local and return candidate powers to zero  .
    {
      int oStride    = STRIDE_STAGE[0];

      for ( int stage = 0; stage < noStages; stage++ )
      {
        for ( int step = 0; step < noSteps; step++)             // Loop over steps
        {
          d_cands[blockIdx.y*noSteps*noStages*oStride + step*noStages*oStride + stage*oStride + tid ].sigma = 0;
        }
      }
    }

    for ( int harm = 0; harm < noHarms ; harm++)                // Loop over plains  .
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

        FOLD // Read data from plains  .
        {
          for ( int y = y0; y < y1; y++ )
          {
            int idx  = (y) * stride ;

            FOLD // Read  .
            {
              if      ( FLAGS & FLAG_MUL_CB_OUT )
              {
                float cmpf            = powersArr[0][ tid + idx ];
                tSum                  += cmpf;
              }
              else
              {
                fcomplexcu cmpc       = cmplxArr[0][ tid + idx ];
                tSum                  += cmpc.r * cmpc.r + cmpc.i * cmpc.i ;
              }
            }
          }
        }

        if ( tSum < 0 )
        {
          printf("add_and_searchCU02_k");
        }
      }
    }
  }
}

template<uint FLAGS>
__host__ void add_and_searchCU00_c(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int   noStages  = log2((double)batch->noHarms) + 1 ;
  fsHarmList  powers;
  cHarmList   cmplx;

  for (int i = 0; i < batch->noHarms; i++)
  {
    int idx = i;
    powers.val[i]   = batch->plains[idx].d_plainPowers;
    cmplx.val[i]    = batch->plains[idx].d_plainData;
  }

  //  switch (globalInt01)
  //  {
  //    case 1:
  //    {
  //      add_and_searchCU00_k<FLAGS, 1> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    case 2:
  //    {
  //      add_and_searchCU00_k<FLAGS,2> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  ////    case 3:
  ////    {
  ////      add_and_searchCU00_k<FLAGS,3> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  ////      break;
  ////    }
  //    case 4:
  //    {
  //      add_and_searchCU00_k<FLAGS,4> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    case 5:
  //    {
  //      add_and_searchCU00_k<FLAGS,5> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  ////    case 6:
  ////    {
  ////      add_and_searchCU00_k<FLAGS,6> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  ////      break;
  ////    }
  //    case 7:
  //    {
  //      add_and_searchCU00_k<FLAGS,7> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    case 8:
  //    {
  //      add_and_searchCU00_k<FLAGS,8> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  ////    case 9:
  ////    {
  ////      add_and_searchCU00_k<FLAGS,9> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  ////      break;
  ////    }
  //    case 10:
  //    {
  //      add_and_searchCU00_k<FLAGS,10> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    case 12:
  //    {
  //      add_and_searchCU00_k<FLAGS,12> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    case 14:
  //    {
  //      add_and_searchCU00_k<FLAGS,14> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    case 16:
  //    {
  //      add_and_searchCU00_k<FLAGS,16> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    case 18:
  //    {
  //      add_and_searchCU00_k<FLAGS,18> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    case 20:
  //    {
  //      add_and_searchCU00_k<FLAGS,20> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    case 24:
  //    {
  //      add_and_searchCU00_k<FLAGS,24> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    default:
  //      fprintf(stderr, "ERROR: %s has not been templated for %i chunk size.\n", __FUNCTION__, globalInt01);
  //      exit(EXIT_FAILURE);
  //  }

  add_and_searchCU00_k<FLAGS,0> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
}

template<uint FLAGS>
__host__ void add_and_searchCU01_c(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int   noStages  = log2((double)batch->noHarms) + 1 ;
  fsHarmList  powers;
  cHarmList   cmplx;

  for (int i = 0; i < batch->noHarms; i++)
  {
    int idx         = batch->stageIdx[i]; // Stage order
    powers.val[i]   = batch->plains[idx].d_plainPowers;
    cmplx.val[i]    = batch->plains[idx].d_plainData;
  }

  add_and_searchCU01_k<FLAGS,0> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
}

template<uint FLAGS>
__host__ void add_and_searchCU02_c(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int   noStages  = log2((double)batch->noHarms) + 1 ;
  fsHarmList  powers;
  cHarmList   cmplx;

  for (int i = 0; i < batch->noHarms; i++)
  {
    int idx         = batch->stageIdx[i]; // Stage order
    powers.val[i]   = batch->plains[idx].d_plainPowers;
    cmplx.val[i]    = batch->plains[idx].d_plainData;
  }

  //  switch (globalInt01)
  //  {
  //    case 1:
  //    {
  //      add_and_searchCU01_k<FLAGS, 1> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    case 2:
  //    {
  //      add_and_searchCU01_k<FLAGS,2> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    //    case 3:
  //    //    {
  //    //      add_and_searchCU01_k<FLAGS,3> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //    //      break;
  //    //    }
  //    case 4:
  //    {
  //      add_and_searchCU01_k<FLAGS,4> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    //    case 5:
  //    //    {
  //    //      add_and_searchCU01_k<FLAGS,5> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //    //      break;
  //    //    }
  //    case 6:
  //    {
  //      add_and_searchCU01_k<FLAGS,6> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    //    case 7:
  //    //    {
  //    //      add_and_searchCU01_k<FLAGS,7> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //    //      break;
  //    //    }
  //    case 8:
  //    {
  //      add_and_searchCU01_k<FLAGS,8> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    //    case 9:
  //    //    {
  //    //      add_and_searchCU01_k<FLAGS,9> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //    //      break;
  //    //    }
  //    case 10:
  //    {
  //      add_and_searchCU01_k<FLAGS,10> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    case 12:
  //    {
  //      add_and_searchCU01_k<FLAGS,12> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    case 14:
  //    {
  //      add_and_searchCU01_k<FLAGS,14> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    case 16:
  //    {
  //      add_and_searchCU01_k<FLAGS,16> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    case 18:
  //    {
  //      add_and_searchCU01_k<FLAGS,18> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    case 20:
  //    {
  //      add_and_searchCU01_k<FLAGS,20> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    case 24:
  //    {
  //      add_and_searchCU01_k<FLAGS,24> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
  //      break;
  //    }
  //    default:
  //      fprintf(stderr, "ERROR: %s has not been templated for %i chunk size.\n", __FUNCTION__, globalInt01);
  //      exit(EXIT_FAILURE);
  //  }

  add_and_searchCU02_k<FLAGS,0> <<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, powers, cmplx, batch->noHarms, noStages, batch->noSteps  );
}

__host__ void add_and_searchCU00(cudaStream_t stream, cuFFdotBatch* batch )
{
  dim3 dimBlock, dimGrid;

  const uint FLAGS    = batch->flag;

  dimBlock.x  = SS00_X;
  dimBlock.y  = SS00_Y;

  float bw    = SS00BS ;
  float ww    = batch->accelLen / ( bw );

  dimGrid.x   = ceil(ww);
  dimGrid.y   = batch->noSSSlices;

  if ( 0 )  // Stage order  .
  {
    if        ( FLAGS & FLAG_MUL_CB_OUT )
    {
      add_and_searchCU01_c<FLAG_MUL_CB_OUT>(dimGrid,dimBlock,stream, batch );
    }
    else
    {
      add_and_searchCU01_c<0>(dimGrid,dimBlock,stream, batch );
    }
  }
  else
  {
    if        ( FLAGS & FLAG_MUL_CB_OUT )
    {
      add_and_searchCU00_c<FLAG_MUL_CB_OUT>(dimGrid,dimBlock,stream, batch );
    }
    else
    {
      add_and_searchCU00_c<0>(dimGrid,dimBlock,stream, batch );
    }
  }

}

