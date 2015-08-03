#include "cuda_accel_SS.h"

#define SS32_X           16                    // X Thread Block
#define SS32_Y           8                     // Y Thread Block
#define SS32BS           (SS32_X*SS32_Y)


template<int noStages, int noSteps>
__device__ __forceinline__ int idxSS(int tid, int stage, int step)
{
  return stage * noSteps * SS32BS + SS32BS * step + tid ;
}

/** Sum and Search - loop down - column max - multi-step - step outer .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSteps
 */
template<uint FLAGS, const int noStages, const int noHarms, const int cunkSize, const int noSteps>
__global__ void add_and_searchCU32_k(const uint width, __restrict__ candPZs* d_cands, tHarmList texs, fsHarmList powersArr, cHarmList cmplxArr )
{
  const int tid   = threadIdx.y * SS32_X  +  threadIdx.x;   /// Block index
  const int gid   = blockIdx.x  * SS32BS  +  tid;           /// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    const int zeroHeight  = HEIGHT_STAGE[0];
    const int oStride     = STRIDE_STAGE[0];                            /// The stride of the output data

    int                 inds      [noHarms];
    float               candPow   [noStages];
    int                 candZ     [noStages];
    float               powers    [cunkSize];                           /// registers to hold values to increase mem cache hits

    FOLD // Prep - Initialise the x indices & set candidates to 0 .
    {
      FOLD // Calculate the x indices or create a pointer offset by the correct amount  .
      {
        //#pragma unroll
        for ( int harm = 0; harm < noHarms; harm++ )                /// loop over harmonic  .
        {
          // NOTE: the indexing below assume each plain starts on a multiple of noHarms
          int   ix    = roundf( gid*FRAC_STAGE[harm] ) + HWIDTH_STAGE[harm] ;
          inds[harm]  = ix;
        }
      }

      FOLD  // Set the local and return candidate powers to zero  .
      {
        //#pragma unroll
        for ( int stage = 0; stage < noStages; stage++ )
        {
          //#pragma unroll
          for ( int step = 0; step < noSteps; step++)               // Loop over steps
          {
            d_cands[step*noStages*oStride + stage*oStride + gid ].value = 0 ;
          }
        }
      }
    }

    FOLD // Sum & Search - Ignore contaminated ends tid to starts at correct spot  .
    {
      for ( int step = 0; step < noSteps; step++)         // Loop over steps  .
      {
        FOLD // Initialise candidate to zero
        {
          for ( int stage = 0; stage < noStages; stage++ )
          {
            candPow [stage]        = 0 ;
          }
        }

        for( int y = 0; y < zeroHeight ; y += cunkSize )              // loop over chunks .
        {
          FOLD // Initialise powers for each section column to 0  .
          {
            for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )          // Loop over powers .
            {
              powers[yPlus] = 0;
            }
          }

          FOLD // Loop over stages, sum and search  .
          {
            //#pragma unroll
            for ( int stage = 0 ; stage < noStages; stage++)          // Loop over stages  .
            {
              int start = STAGE[stage][0] ;
              int end   = STAGE[stage][1] ;

              FOLD // Create a section of summed powers one for 1 step and all plains in stage  .
              {
                for ( int harm = start; harm <= end; harm++ )           // Loop over harmonics (batch) in this stage  .
                {
                  int     ix1     = inds[harm] ;
                  int     ix2     = ix1;
                  int     iyP     = -1;
                  float   pow     = 0;

                  for( int yPlus = 0; yPlus < cunkSize; yPlus++ )       // Loop over the chunk  .
                  {
                    int trm     = y + yPlus ;                           ///< True Y index in plain

                    int iy1     = YINDS[ zeroHeight*harm + trm ];
                    //  OR
                    //int iy1     = roundf( (HEIGHT_STAGE[harm]-1.0)*trm/(float)(zeroHeight-1.0) ) ;
                    // OR
                    //int iy1     = ( h1 * trm + zh2 ) / zh1;

                    int iy2;

                    if ( iyP != iy1 ) // Only read power if it is not the same as the previous
                    {
                      FOLD // Calculate index  .
                      {
                        if        ( FLAGS & FLAG_ITLV_PLN )
                        {
                          iy2 = ( iy1 + step * HEIGHT_STAGE[harm] ) * STRIDE_STAGE[harm] ;
                        }
                        else
                        {
                          ix2 = ix1 + step    * STRIDE_STAGE[harm] ;
                          iy2 = iy1 * noSteps * STRIDE_STAGE[harm];
                        }
                      }

                      FOLD // Read powers  .
                      {
                        if      ( FLAGS & FLAG_MUL_CB_OUT )
                        {
                          float cmpf            = powersArr[harm][ iy2 + ix2 ];
                          pow                   = cmpf;
                        }
                        else
                        {
                          fcomplexcu cmpc       = cmplxArr[harm][ iy2 + ix2 ];
                          pow                   = cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                        }
                      }

                      iyP = iy1;
                    }

                    FOLD // // Accumulate powers  .
                    {
                      powers[yPlus] += pow;
                    }
                  }
                }
              }

              FOLD // Search set of powers  .
              {
                for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )     // Loop over section  .
                {
                  if  (  powers[yPlus] > POWERCUT_STAGE[stage] )
                  {
                    if ( powers[yPlus] > candPow [stage] )
                    {
                      if ( y + yPlus < zeroHeight )
                      {
                        candPow [stage]  = powers[yPlus];
                        candZ   [stage]  = y+yPlus;
                      }
                    }
                  }
                }
              }
            }
          }
        }

        FOLD // Write results back to DRAM and calculate sigma if needed  .
        {
          //#pragma unroll
          for ( int stage = 0 ; stage < noStages; stage++)      // Loop over stages
          {
            if  ( candPow [stage] >  POWERCUT_STAGE[stage] )
            {
              candPZs tt;
              tt.value = candPow [stage];
              tt.z     = candZ   [stage];

              // Write to DRAM
              d_cands[step*noStages*oStride + stage*oStride + gid] = tt;
            }
          }
        }
      }
    }
  }
}

template<uint FLAGS, int noStages, const int noHarms, const int cunkSize>
__host__ void add_and_searchCU32_q(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int noSteps = batch->noSteps ;

  for ( int step = 0; step < noSteps; step++)
  {
    long long firstBin  = (*batch->rConvld)[step][0].expBin ;

    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->stageIdx[i];

      long long binb      = (*batch->rConvld)[step][idx].expBin ;

      if ( firstBin * h_FRAC_STAGE[i] != binb )
      {
        fprintf(stderr,"ERROR, in function %s, R values are not properly aligned! Each step should start on a multiple of (2 x No Harms).\n", __FUNCTION__ );
        fprintf(stderr,"%f != %f.\n", firstBin * h_FRAC_STAGE[i], (float)binb );
        exit(EXIT_FAILURE);
      }
    }
  }

  tHarmList   texs;
  fsHarmList powers;
  cHarmList   cmplx;

  for (int i = 0; i < noHarms; i++)
  {
    int idx         = batch->stageIdx[i];
    texs.val[i]     = batch->plains[idx].datTex;
    powers.val[i]   = batch->plains[idx].d_plainPowers;
    cmplx.val[i]    = batch->plains[idx].d_plainData;
  }

  switch (noSteps)
  {
    case 1:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,1><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 2:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,2><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 3:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,3><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 4:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,4>, cudaFuncCachePreferL1);
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,4><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 5:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,5><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 6:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,6><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 7:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,7><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 8:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,8><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    default:
      fprintf(stderr, "ERROR: add_and_searchCU311 has not been templated for %i steps\n", noSteps);
      exit(EXIT_FAILURE);
  }
}

template<uint FLAGS, int noStages, const int noHarms>
__host__ void add_and_searchCU32_c(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  switch (globalInt01)
  {
    case 1:
    {
      add_and_searchCU32_q<FLAGS,noStages,noHarms,1>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 2:
    {
      add_and_searchCU32_q<FLAGS,noStages,noHarms,2>(dimGrid, dimBlock, stream, batch);
      break;
    }
    //    case 3:
    //    {
    //      add_and_searchCU32_q<FLAGS,noStages,noHarms,3>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    case 4:
    {
      add_and_searchCU32_q<FLAGS,noStages,noHarms,4>(dimGrid, dimBlock, stream, batch);
      break;
    }
    //    case 5:
    //    {
    //      add_and_searchCU32_q<FLAGS,noStages,noHarms,5>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    case 6:
    {
      add_and_searchCU32_q<FLAGS,noStages,noHarms,6>(dimGrid, dimBlock, stream, batch);
      break;
    }
    //    case 7:
    //    {
    //      add_and_searchCU32_q<FLAGS,noStages,noHarms,7>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    case 8:
    {
      add_and_searchCU32_q<FLAGS,noStages,noHarms,8>(dimGrid, dimBlock, stream, batch);
      break;
    }
    //    case 9:
    //    {
    //      add_and_searchCU32_q<FLAGS,noStages,noHarms,9>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    case 10:
    {
      add_and_searchCU32_q<FLAGS,noStages,noHarms,10>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 12:
    {
      add_and_searchCU32_q<FLAGS,noStages,noHarms,12>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 14:
    {
      add_and_searchCU32_q<FLAGS,noStages,noHarms,14>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 16:
    {
      add_and_searchCU32_q<FLAGS,noStages,noHarms,16>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 18:
    {
      add_and_searchCU32_q<FLAGS,noStages,noHarms,18>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 20:
    {
      add_and_searchCU32_q<FLAGS,noStages,noHarms,20>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 24:
    {
      add_and_searchCU32_q<FLAGS,noStages,noHarms,24>(dimGrid, dimBlock, stream, batch);
      break;
    }
    default:
      fprintf(stderr, "ERROR: %s has not been templated for %i chunk size.\n", __FUNCTION__, globalInt01);
      exit(EXIT_FAILURE);
  }

}

template<uint FLAGS >
__host__ void add_and_searchCU32_p(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int noStages = batch->noHarmStages;

  switch (noStages)
  {
    //    case 1:
    //    {
    //      add_and_searchCU32_c<FLAGS,1,1>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    //    case 2:
    //    {
    //      add_and_searchCU32_c<FLAGS,2,2>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    //    case 3:
    //    {
    //      add_and_searchCU32_c<FLAGS,3,4>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    //    case 4:
    //    {
    //      add_and_searchCU32_c<FLAGS,4,8>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    case 5:
    {
      add_and_searchCU32_c<FLAGS,5,16>(dimGrid, dimBlock, stream, batch);
      break;
    }
    default:
      fprintf(stderr, "ERROR: %s has not been templated for %i stages\n", __FUNCTION__, noStages);
      exit(EXIT_FAILURE);
  }
}

__host__ void add_and_searchCU32(cudaStream_t stream, cuFFdotBatch* batch )
{
  const uint FLAGS = batch->flag;
  dim3 dimBlock, dimGrid;

  dimBlock.x  = SS32_X;
  dimBlock.y  = SS32_Y;

  float bw    = SS32BS;
  float ww    = batch->accelLen / ( bw );

  dimGrid.x   = ceil(ww);
  dimGrid.y   = 1;

  if        ( FLAGS & FLAG_MUL_CB_OUT )
  {
    if      ( FLAGS & FLAG_ITLV_ROW )
      add_and_searchCU32_p<FLAG_MUL_CB_OUT | FLAG_ITLV_ROW> (dimGrid, dimBlock, stream, batch);
    else if ( FLAGS & FLAG_ITLV_PLN )
      add_and_searchCU32_p<FLAG_MUL_CB_OUT | FLAG_ITLV_PLN>  (dimGrid, dimBlock, stream, batch);
    else
    {
      fprintf(stderr, "ERROR: %s has not been templated for flag combination. \n", __FUNCTION__ );
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    if      ( FLAGS & FLAG_ITLV_ROW )
      add_and_searchCU32_p<FLAG_ITLV_ROW> (dimGrid, dimBlock, stream, batch);
    else if ( FLAGS & FLAG_ITLV_PLN )
      add_and_searchCU32_p<FLAG_ITLV_PLN> (dimGrid, dimBlock, stream, batch);
    else
    {
      fprintf(stderr, "ERROR: %s has not been templated for flag combination. \n", __FUNCTION__ );
      exit(EXIT_FAILURE);
    }
  }
}

