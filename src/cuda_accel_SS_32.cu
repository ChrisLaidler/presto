#include "cuda_accel_SS.h"

#define SS33_X           16                    // X Thread Block
#define SS33_Y           8                     // Y Thread Block
#define SS33BS           (SS33_X*SS33_Y)


template<int noStages, int noSteps>
__device__ __forceinline__ int idxSS(int tid, int stage, int step)
{
  return stage * noSteps * SS33BS + SS33BS * step + tid ;
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
__global__ void add_and_searchCU32_k(const uint width, __restrict__ candPZ* d_cands, tHarmList texs, fsHarmList powersArr, cHarmList cmplxArr )
{
  const int tid   = threadIdx.y * SS33_X          +  threadIdx.x;   /// Block index
  const int gid   = blockIdx.x  * (SS33_Y*SS33_X) +  tid;           /// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    const int zeroHeight  = HEIGHT_STAGE[0];
    const int oStride     = STRIDE_STAGE[0];                            /// The stride of the output data

    //const int zh1         = zeroHeight-1;
    //const int zh2         = zeroHeight-2;

    int                 inds      [noHarms];
    float               candPow   [noStages];
    int                 candZ     [noStages];
    float               powers    [cunkSize];                           /// registers to hold values to increase mem cache hits
    int                 stride    [noHarms];

    FOLD // Prep - Initialise the x indices & set candidates to 0 .
    {
      FOLD // Calculate the x indices or create a pointer offset by the correct amount  .
      {
#pragma unroll
        for ( int harm = 0; harm < noHarms; harm++ )                /// loop over harmonic  .
        {
          // NOTE: the indexing below assume each plain starts on a multiple of noHarms
          int   ix    = roundf( gid*FRAC_STAGE[harm] ) + HWIDTH_STAGE[harm] ;
          inds[harm]  = ix;

          if ( FLAGS & FLAG_ITLV_ROW )
          {
            stride[harm] = noSteps*STRIDE_STAGE[harm] ;
          }
          else
          {
            stride[harm] = STRIDE_STAGE[harm] ;
          }
        }
      }

      FOLD  // Set the local and return candidate powers to zero  .
      {
#pragma unroll
        for ( int stage = 0; stage < noStages; stage++ )
        {
#pragma unroll
          for ( int step = 0; step < noSteps; step++)               // Loop over steps
          {
            d_cands[step*noStages*oStride + stage*oStride + gid ].value = 0 ;
          }
        }
      }
    }

    FOLD // Sum & Search - Ignore contaminated ends tid to starts at correct spot  .
    {
#pragma unroll
      for ( int step = 0; step < noSteps; step++)         // Loop over steps  .
      {
        FOLD // Initialise candidate to zero
        {
#pragma unroll
          for ( int stage = 0; stage < noStages; stage++ )
          {
            candPow [stage]        = 0 ;
          }
        }

        for( int y = 0; y < zeroHeight ; y += cunkSize )              // loop over chunks .
        {
          FOLD // Initialise powers for each section column to 0  .
          {
#pragma unroll
            for ( int step = 0; step < noSteps; step++)                 // Loop over steps .
            {
#pragma unroll
              for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )          // Loop over powers .
              {
                powers[yPlus] = 0;
              }
            }
          }

          FOLD // Loop over stages, sum and search  .
          {
#pragma unroll
            for ( int stage = 0 ; stage < noStages; stage++)          // Loop over stages  .
            {
              int start = STAGE[stage][0] ;
              int end   = STAGE[stage][1] ;

              FOLD // Create a section of summed powers one for 1 step and all plains in stage  .
              {
#pragma unroll
                for ( int harm = start; harm <= end; harm++ )           // Loop over harmonics (batch) in this stage  .
                {
                  int ix1       = inds[harm] ;
                  int ix2       = ix1;
                  //int h1      = HEIGHT_STAGE[harm]-1;

#pragma unroll
                  for( int yPlus = 0; yPlus < cunkSize; yPlus++ )       // Loop over the chunk  .
                  {
                    int trm     = y + yPlus ;                           ///< True Y index in plain

                    int iy1     = YINDS[ zeroHeight*harm + trm ];
                    //  OR
                    //int iy1     = roundf( (HEIGHT_STAGE[harm]-1.0)*trm/(float)(zeroHeight-1.0) ) ;
                    // OR
                    //int iy1     = ( h1 * trm + zh2 ) / zh1;

                    int iy2;

                    if        ( FLAGS & FLAG_ITLV_PLN )
                    {
                      iy2 = ( iy1 + step * HEIGHT_STAGE[harm] ) * stride[harm];
                    }
                    else
                    {
                      ix2 = ix1 + step * STRIDE_STAGE[harm] ;
                      iy2 = iy1*stride[harm];
                    }

                    if        ( FLAGS & FLAG_SAS_TEX )
                    {
                      if      ( FLAGS & FLAG_MUL_CB_OUT )
                      {
                        const float cmpf      = tex2D < float > (texs.val[harm], ix2+0.5f, iy2+0.5f ); // + 0.5 YES + 0.5 I REALLY wish someone had documented that one, 2 days of debugging to find that!!!!!!
                        powers[yPlus]   += cmpf;
                      }
                      else
                      {
                        const float r         = tex2D < float > (texs.val[harm], ix2*2+0.5f, iy2+0.5f ); // + 0.5 YES + 0.5 I REALLY wish someone had documented that one, 2 days of debugging to find that!!!!!!
                        const float i         = tex2D < float > (texs.val[harm], ix2*2+1.5f, iy2+0.5f ); // + 0.5 YES + 0.5 I REALLY wish someone had documented that one, 2 days of debugging to find that!!!!!!
                        powers[yPlus]   += r*r+i*i;
                      }
                    }
                    else
                    {
                      if      ( FLAGS & FLAG_MUL_CB_OUT )
                      {
                        float cmpf            = powersArr[harm][ iy2 + ix2 ];
                        powers[yPlus]         += cmpf;

//                        if ( gid == 1022 && stage == 0 && step == 0 ) // TMP
//                        {
//                          if ( candPow [stage] < cmpf )
//                            printf("%03i %15.3f  %15.3f  %04i %04i New best! \n", trm, cmpf, candPow [stage], iy2, ix2 );
//                          else
//                            printf("%03i %15.3f  %15.3f  %04i %04i \n",trm, cmpf, candPow [stage], iy2, ix2 );
//                        }
                      }
                      else
                      {
                        fcomplexcu cmpc       = cmplxArr[harm][ iy2 + ix2 ];
                        powers[yPlus]  += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                      }
                    }
                  }
                }
              }

              FOLD // Search set of powers  .
              {
#pragma unroll
                for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )     // Loop over section  .
                {
//                  if ( gid == 1022 && stage == 0 && step == 0 ) // TMP
//                  {
//                    printf("  %03i %12.3f  %15.3f  %15.3f \n", yPlus, powers[yPlus], candPow [stage], POWERCUT_STAGE[stage] );
//                  }

                  if  (  powers[yPlus] > POWERCUT_STAGE[stage] )
                  {
                    if ( powers[yPlus] > candPow [stage] )
                    {
                      if ( y + yPlus < zeroHeight )
                      {
//                        if ( gid == 1022 && stage == 0 && step == 0 ) // TMP
//                        {
//                          printf("-- New Max --\n");
//                        }
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
#pragma unroll
          for ( int stage = 0 ; stage < noStages; stage++)      // Loop over stages
          {
            if  ( candPow [stage] >  POWERCUT_STAGE[stage] )
            {
              candPZ tt;
              tt.value = candPow [stage];
              tt.z     = candZ   [stage];

//              if ( gid == 0 && stage == 0 && step == 1 ) // TMP
//              {
//                printf(" GPU MAX %f z: %i at %i \n", tt.value, tt.z, step*noStages*oStride + stage*oStride + gid );
//              }

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
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,1><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 2:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,2><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 3:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,3><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 4:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,4>, cudaFuncCachePreferL1);
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,4><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 5:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,5><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 6:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,6><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 7:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,7><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 8:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,8><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    default:
      fprintf(stderr, "ERROR: add_and_searchCU311 has not been templated for %i steps\n", noSteps);
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
    //      add_and_searchCU32_q<FLAGS,1,1>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    //    case 2:
    //    {
    //      add_and_searchCU32_q<FLAGS,2,2>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    //    case 3:
    //    {
    //      add_and_searchCU32_q<FLAGS,3,4>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    //    case 4:
    //    {
    //      add_and_searchCU32_q<FLAGS,4,8>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    case 5:
    {
      add_and_searchCU32_q<FLAGS,5,16,8>(dimGrid, dimBlock, stream, batch);
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

  dimBlock.x  = SS33_X;
  dimBlock.y  = SS33_Y;

  float bw    = SS33_X * SS33_Y;
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

