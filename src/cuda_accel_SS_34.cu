#include "cuda_accel_SS.h"

#define SS34_X           8                     // X Thread Block
#define SS34_Y           8                     // Y Thread Block
#define SS34BS           (SS34_X*SS34_Y)

/** Sum and Search - loop down - column max - multi-step - step outer .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSteps
 */
template<uint FLAGS, const int noStages, const int noHarms, const int cunkSize, const int noSteps>
__global__ void add_and_searchCU34_k(const uint width, __restrict__ candPZ* d_cands, tHarmList texs, fsHarmList powersArr, cHarmList cmplxArr )
{
  const int tid   = threadIdx.y * SS34_X  +  threadIdx.x;   /// Block index
  const int gid   = blockIdx.x  * SS34BS  +  tid;           /// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    const int zeroHeight  = HEIGHT_STAGE[0];
    const int oStride     = STRIDE_STAGE[0];                /// The stride of the output data

    int       inds      [noHarms];
    int       stride    [noHarms];

    // Candidates
    //float     candPow   [noStages];
    //int       candZ     [noStages];
    float     candPow   [noStages][noSteps];
    int       candZ     [noStages][noSteps];

    // Powers
    float     powers    [noHarms][noSteps];
    int       idxP      [noHarms];

    FOLD // Prep - Initialise the x indices & set candidates to 0 .
    {
      FOLD // Calculate the x indices or create a pointer offset by the correct amount  .
      {
        for ( int harm = 0; harm < noHarms; harm++ )                // loop over harmonic  .
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
        for ( int stage = 0; stage < noStages; stage++ )
        {
          for ( int step = 0; step < noSteps; step++)                 // Loop over steps
          {
            candPow [stage][step] = POWERCUT_STAGE[stage];
            d_cands[step*noStages*oStride + stage*oStride + gid ].value   = 0 ;
          }
        }
      }
    }

    FOLD // Sum & Search - Ignore contaminated ends tid to starts at correct spot  .
    {

      //for ( int step = 0; step < noSteps; step++)                   // Loop over steps  .
      {

        FOLD // Reset max candidate  .
        {
          for ( int harm = 0; harm < noHarms; harm++ )                // loop over harmonic  .
          {
            idxP[harm] = -1;
          }
        }

        for( int y = 0; y < zeroHeight ; y++ )                        // loop over plain .
        {
          float pow[noSteps];

          for ( int step = 0; step < noSteps; step++)                 // Loop over steps  .
          {
            pow[step] = 0;
          }

          for ( int stage = 0 ; stage < noStages; stage++)            // Loop over stages  .
          {
            const int start = STAGE[stage][0] ;
            const int end   = STAGE[stage][1] ;

            FOLD // Calculate summed power for this stage  .
            {
              for ( int harm = start; harm <= end; harm++ )           // Loop over harmonics (batch) in this stage  .
              {
                int ix1       = inds[harm] ;
                int ix2       = ix1;

                //int iy0       = YINDS[ zeroHeight*harm + y-1  ];    // TODO: fix this
                int iy1       = YINDS[ zeroHeight*harm + y    ];
                int iy2;

                if ( idxP[harm] != iy1 )  // Read power from main memory  .
                {
                  for ( int step = 0; step < noSteps; step++)         // Loop over steps  .
                  {
                    FOLD // Calculate index  .
                    {
                      if        ( FLAGS & FLAG_ITLV_PLN )
                      {
                        iy2                   = ( iy1 + step * HEIGHT_STAGE[harm] ) * stride[harm];
                      }
                      else
                      {
                        ix2                   = ix1 + step * STRIDE_STAGE[harm] ;
                        iy2                   = iy1 * stride[harm];
                      }
                    }

                    FOLD // Accumulate powers  .
                    {
                      if      ( FLAGS & FLAG_MUL_CB_OUT )
                      {
                        //float cmpf            = powersArr[harm][ iy2 + ix2 ];
                        //powers[harm][step]    = cmpf;

                        //powers[harm][step]    = powersArr[harm][ iy2 + ix2 ];
                      }
                      else
                      {
                        fcomplexcu cmpc       = cmplxArr[harm][ iy2 + ix2 ];
                        powers[harm][step]    = cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                      }
                    }
                  }

                  idxP[harm] = iy1;
                }

                for ( int step = 0; step < noSteps; step++)           // Loop over steps  .
                {
                  pow[step] += powers[harm][step];
                }
              }
            }

//            FOLD // Update maximum  .
//            {
//              for ( int step = 0; step < noSteps; step++)             // Loop over steps  .
//              {
//                if ( pow[step] > candPow [stage][step] )
//                {
//                  // This is our new max!
//                  candPow [stage][step]   = pow[step];
//                  candZ   [stage][step]   = y;
//                }
//              }
//            }
          }
        }

        FOLD // Write results back to DRAM and calculate sigma if needed  .
        {
          for ( int step = 0; step < noSteps; step++)                 // Loop over steps  .
          {
            for ( int stage = 0 ; stage < noStages; stage++)          // Loop over stages
            {
              if  ( candPow [stage][step] > POWERCUT_STAGE[stage] )
              {
                candPZ tt;

                tt.value = candPow [stage][step];
                tt.z     = candZ   [stage][step];

                // Write to DRAM
                d_cands[step*noStages*oStride + stage*oStride + gid] = tt;
              }
            }
          }
        }

      }
    }
  }
}

template<uint FLAGS, int noStages, const int noHarms, const int cunkSize>
__host__ void add_and_searchCU34_q(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
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
  fsHarmList  powers;
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
      add_and_searchCU34_k<FLAGS,noStages,noHarms,cunkSize,1><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 2:
    {
      add_and_searchCU34_k<FLAGS,noStages,noHarms,cunkSize,2><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 3:
    {
      add_and_searchCU34_k<FLAGS,noStages,noHarms,cunkSize,3><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 4:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU34_k<FLAGS,noStages,noHarms,cunkSize,4>, cudaFuncCachePreferL1);
      add_and_searchCU34_k<FLAGS,noStages,noHarms,cunkSize,4><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 5:
    {
      add_and_searchCU34_k<FLAGS,noStages,noHarms,cunkSize,5><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 6:
    {
      add_and_searchCU34_k<FLAGS,noStages,noHarms,cunkSize,6><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 7:
    {
      add_and_searchCU34_k<FLAGS,noStages,noHarms,cunkSize,7><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 8:
    {
      add_and_searchCU34_k<FLAGS,noStages,noHarms,cunkSize,8><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    default:
      fprintf(stderr, "ERROR: add_and_searchCU311 has not been templated for %i steps\n", noSteps);
      exit(EXIT_FAILURE);
  }
}

template<uint FLAGS, int noStages, const int noHarms>
__host__ void add_and_searchCU34_c(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
//  switch (globalInt01)
//  {
//    case 1:
//    {
//      add_and_searchCU34_q<FLAGS,noStages,noHarms,1>(dimGrid, dimBlock, stream, batch);
//      break;
//    }
//    case 2:
//    {
//      add_and_searchCU34_q<FLAGS,noStages,noHarms,2>(dimGrid, dimBlock, stream, batch);
//      break;
//    }
//    //    case 3:
//    //    {
//    //      add_and_searchCU34_q<FLAGS,noStages,noHarms,3>(dimGrid, dimBlock, stream, batch);
//    //      break;
//    //    }
//    case 4:
//    {
//      add_and_searchCU34_q<FLAGS,noStages,noHarms,4>(dimGrid, dimBlock, stream, batch);
//      break;
//    }
//    //    case 5:
//    //    {
//    //      add_and_searchCU34_q<FLAGS,noStages,noHarms,5>(dimGrid, dimBlock, stream, batch);
//    //      break;
//    //    }
//    case 6:
//    {
//      add_and_searchCU34_q<FLAGS,noStages,noHarms,6>(dimGrid, dimBlock, stream, batch);
//      break;
//    }
//    //    case 7:
//    //    {
//    //      add_and_searchCU34_q<FLAGS,noStages,noHarms,7>(dimGrid, dimBlock, stream, batch);
//    //      break;
//    //    }
//    case 8:
//    {
//      add_and_searchCU34_q<FLAGS,noStages,noHarms,8>(dimGrid, dimBlock, stream, batch);
//      break;
//    }
//    //    case 9:
//    //    {
//    //      add_and_searchCU34_q<FLAGS,noStages,noHarms,9>(dimGrid, dimBlock, stream, batch);
//    //      break;
//    //    }
//    case 10:
//    {
//      add_and_searchCU34_q<FLAGS,noStages,noHarms,10>(dimGrid, dimBlock, stream, batch);
//      break;
//    }
//    case 12:
//    {
//      add_and_searchCU34_q<FLAGS,noStages,noHarms,12>(dimGrid, dimBlock, stream, batch);
//      break;
//    }
//    case 14:
//    {
//      add_and_searchCU34_q<FLAGS,noStages,noHarms,14>(dimGrid, dimBlock, stream, batch);
//      break;
//    }
//    case 16:
//    {
//      add_and_searchCU34_q<FLAGS,noStages,noHarms,16>(dimGrid, dimBlock, stream, batch);
//      break;
//    }
//    case 18:
//    {
//      add_and_searchCU34_q<FLAGS,noStages,noHarms,18>(dimGrid, dimBlock, stream, batch);
//      break;
//    }
//    case 20:
//    {
//      add_and_searchCU34_q<FLAGS,noStages,noHarms,20>(dimGrid, dimBlock, stream, batch);
//      break;
//    }
//    case 24:
//    {
//      add_and_searchCU34_q<FLAGS,noStages,noHarms,24>(dimGrid, dimBlock, stream, batch);
//      break;
//    }
//    default:
//      fprintf(stderr, "ERROR: %s has not been templated for %i chunk size.\n", __FUNCTION__, globalInt01);
//      exit(EXIT_FAILURE);
//  }

  add_and_searchCU34_q<FLAGS,noStages,noHarms,0>(dimGrid, dimBlock, stream, batch);
}

template<uint FLAGS >
__host__ void add_and_searchCU34_p(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int noStages = batch->noHarmStages;

  switch (noStages)
  {
    //    case 1:
    //    {
    //      add_and_searchCU34_c<FLAGS,1,1>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    //    case 2:
    //    {
    //      add_and_searchCU34_c<FLAGS,2,2>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    //    case 3:
    //    {
    //      add_and_searchCU34_c<FLAGS,3,4>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    //    case 4:
    //    {
    //      add_and_searchCU34_c<FLAGS,4,8>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    case 5:
    {
      add_and_searchCU34_c<FLAGS,5,16>(dimGrid, dimBlock, stream, batch);
      break;
    }
    default:
      fprintf(stderr, "ERROR: %s has not been templated for %i stages\n", __FUNCTION__, noStages);
      exit(EXIT_FAILURE);
  }
}

__host__ void add_and_searchCU34(cudaStream_t stream, cuFFdotBatch* batch )
{
  const uint FLAGS = batch->flag;
  dim3 dimBlock, dimGrid;

  dimBlock.x  = SS34_X ;
  dimBlock.y  = SS34_Y ;

  float bw    = SS34BS ;
  float ww    = batch->accelLen / ( bw );

  dimGrid.x   = ceil(ww);
  dimGrid.y   = 1;

  if        ( FLAGS & FLAG_MUL_CB_OUT )
  {
    if      ( FLAGS & FLAG_ITLV_ROW )
      add_and_searchCU34_p<FLAG_MUL_CB_OUT | FLAG_ITLV_ROW> (dimGrid, dimBlock, stream, batch);
    else if ( FLAGS & FLAG_ITLV_PLN )
      add_and_searchCU34_p<FLAG_MUL_CB_OUT | FLAG_ITLV_PLN>  (dimGrid, dimBlock, stream, batch);
    else
    {
      fprintf(stderr, "ERROR: %s has not been templated for flag combination. \n", __FUNCTION__ );
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    if      ( FLAGS & FLAG_ITLV_ROW )
      add_and_searchCU34_p<FLAG_ITLV_ROW> (dimGrid, dimBlock, stream, batch);
    else if ( FLAGS & FLAG_ITLV_PLN )
      add_and_searchCU34_p<FLAG_ITLV_PLN> (dimGrid, dimBlock, stream, batch);
    else
    {
      fprintf(stderr, "ERROR: %s has not been templated for flag combination. \n", __FUNCTION__ );
      exit(EXIT_FAILURE);
    }
  }
}

