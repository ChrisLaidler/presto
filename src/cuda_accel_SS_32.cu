#include "cuda_accel_SS.h"

#define SS32_X           16                    // X Thread Block
#define SS32_Y           8                     // Y Thread Block

/** Sum and Search - loop down - column max - multi-step - step outer .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSteps
 */
template<uint FLAGS, const int noStages, const int noHarms, const int cunkSize, const int noSteps, typename stpType>
__global__ void add_and_searchCU32(const uint width, accelcandBasic* d_cands, stpType rBin, fsHarmList powersArr, cHarmList cmplxArr )
{
  const int bidx  = threadIdx.y * SS32_X         +  threadIdx.x;   /// Block index
  const int tid   = blockIdx.x  * (SS32_Y*SS32_X) +  bidx;          /// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    const int zeroHeight  = HEIGHT[0];
    const int oStride     = STRIDE[0];                            /// The stride of the output data

    int             inds      [noHarms];
    accelcandBasic  candLists [noStages];
    float           powers    [cunkSize];                         /// registers to hold values to increase mem cache hits
    int             stride    [noHarms];

    FOLD // Calculate the x indices or create a pointer offset by the correct amount  .
    {
      for ( int harm = 0; harm < noHarms; harm++ )               // loop over harmonic  .
      {
        if ( FLAGS & FLAG_ITLV_ROW )
        {
          stride[harm] = noSteps*STRIDE[harm] ;
        }
        else
        {
          stride[harm] = STRIDE[harm] ;
        }
      }
    }

    FOLD // Set the local and return candidate powers to zero  .
    {
      for ( int step = 0; step < noSteps; step++)                 // Loop over steps
      {
        for ( int stage = 0; stage < noStages; stage++ )
        {
          candLists[stage].sigma = 0 ;
          d_cands[step*noStages*oStride + stage*oStride + tid ].sigma = 0;
        }
      }
    }

    for ( int step = 0; step < noSteps; step++)                 // Loop over steps  .
    {
      FOLD // Prep - Initialise the x indices for this step  .
      {
        // Calculate the x indices or create a pointer offset by the correct amount
        for ( int harm = 0; harm < noHarms; harm++ )                // loop over harmonic  .
        {
          float fx    = (rBin.val[step] + tid)*FRAC[harm] - rBin.val[harm*noSteps + step]  + HWIDTH[harm] ;
          int   ix    = roundf(fx) ;

          if ( FLAGS & FLAG_ITLV_ROW )
          {
            ix += step*STRIDE[harm] ;
          }

          inds[harm] = ix;
        }
      }
    }

    FOLD // Get the best candidate for each stage  .
    {
      FOLD // Sum & Search - Ignore contaminated ends tid to starts at correct spot  .
      {
        for( int y = 0; y < zeroHeight ; y += cunkSize )        // loop over chunks .
        {
          for ( int step = 0; step < noSteps; step++)                 // Loop over steps  .
          {
            Fout // Prep - Initialise the x indices for this step  .
            {
              // Calculate the x indices or create a pointer offset by the correct amount
              for ( int harm = 0; harm < noHarms; harm++ )                // loop over harmonic  .
              {
                float fx    = (rBin.val[step] + tid)*FRAC[harm] - rBin.val[harm*noSteps + step]  + HWIDTH[harm] ;
                int   ix    = roundf(fx) ;

                if ( FLAGS & FLAG_ITLV_ROW )
                {
                  ix += step*STRIDE[harm] ;
                }

                inds[harm] = ix;
              }
            }

            FOLD // Initialise powers for each section column to 0
            {
              for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )      // Loop over powers .
              {
                powers[yPlus] = 0;
              }
            }

            FOLD // Loop over stages, sum and search  .
            {
              for ( int stage = 0 ; stage < noStages; stage++)    // Loop over stages .
              {
                int start = STAGE[stage][0] ;
                int end   = STAGE[stage][1] ;

                // Create a section of summed powers one for each step
                for ( int harm = start; harm <= end; harm++ )     // Loop over harmonics (batch) in this stage  .
                {
                  //float hf = (HEIGHT[harm]-1.0f) / (float)(zeroHeight-1.0f) ;

                  for( int yPlus = 0; yPlus < cunkSize; yPlus++ ) // Loop over the chunk  .
                  {
                    int trm     = y + yPlus ;                     ///< True Y index in plain

                    int iy1     = YINDS[ zeroHeight*harm + trm ];
                    //  OR
                    //int iy1     = roundf( hf*trm ) ;

                    int iy2     = iy1*stride[harm];

                    int ix      = inds[harm] ;

                    if        ( FLAGS & FLAG_ITLV_PLN )
                    {
                      iy2 = iy1 + step * HEIGHT[harm];                // stride step by plain
                    }

                    if      ( FLAGS & FLAG_MUL_CB_OUT )
                    {
                      float cmpf            = powersArr[harm][ ix + iy2 ];
                      powers[yPlus]         += cmpf;
                    }
                    else
                    {
                      fcomplexcu cmpc       = cmplxArr[harm][ ix + iy2 ];
                      powers[yPlus]         += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                    }

                  }
                }

                for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )  // Loop over section
                {
                  if  (  powers[yPlus] > POWERCUT[stage] && powers[yPlus] > candLists[stage].sigma )
                  {
                    if ( y + yPlus < zeroHeight )
                    {
                      // This is our new max!
                      candLists[stage].sigma  = powers[yPlus];
                      candLists[stage].z      = y+yPlus;
                    }
                  }
                }
              }
            }
          }

        }


        FOLD // Write results back to DRAM and calculate sigma if needed  .
        {
          for ( int step = 0; step < noSteps; step++)                 // Loop over steps  .
          {
            for ( int stage = 0 ; stage < noStages; stage++ )         // Loop over stages  .
            {
              if  ( candLists[stage].sigma >  POWERCUT[stage] )
              {
                // Write to DRAM
                d_cands[step*noStages*oStride + stage*oStride + tid] = candLists[stage];
              }
            }
          }
        }
      }

    }
  }
}

template<uint FLAGS, int noStages, const int noHarms, const int cunkSize, int noSteps>
__host__ void add_and_searchCU32_s(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  fsHarmList powers;
  cHarmList   cmplx;

  for (int i = 0; i < noHarms; i++)
  {
    int idx         = batch->pIdx[i];
    powers.val[i]   = batch->plains[idx].d_plainPowers;
    cmplx.val[i]    = batch->plains[idx].d_plainData;
  }

  int tmp = 0 ;

  if      ( noHarms*noSteps <= 4   )
  {
    long04 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->pIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    //cudaFuncSetCacheConfig(add_and_searchCU32<FLAGS,noStages,noHarms,cunkSize,noSteps,long08>, cudaFuncCachePreferL1);
    add_and_searchCU32<FLAGS,noStages,noHarms,cunkSize,noSteps,long04><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, rBin, powers, cmplx );
  }
  else if ( noHarms*noSteps <= 8   )
  {
    long08 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->pIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    //cudaFuncSetCacheConfig(add_and_searchCU32<FLAGS,noStages,noHarms,cunkSize,noSteps,long08>, cudaFuncCachePreferL1);
    add_and_searchCU32<FLAGS,noStages,noHarms,cunkSize,noSteps,long08><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, rBin, powers, cmplx );
  }
  else if ( noHarms*noSteps <= 16  )
  {
    long16 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->pIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    //cudaFuncSetCacheConfig(add_and_searchCU32<FLAGS,noStages,noHarms,cunkSize,noSteps,long16>, cudaFuncCachePreferL1);
    add_and_searchCU32<FLAGS,noStages,noHarms,cunkSize,noSteps,long16><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData,rBin, powers, cmplx );
  }
  else if ( noHarms*noSteps <= 32  )
  {
    long32 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->pIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
      //printf("drlo: %.3f \n", (*batch->rConvld)[0][idx].drlo );
    }
    //cudaFuncSetCacheConfig(add_and_searchCU32<FLAGS,noStages,noHarms,cunkSize,noSteps,long32>, cudaFuncCachePreferL1);
    add_and_searchCU32<FLAGS,noStages,noHarms,cunkSize,noSteps,long32><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData,rBin, powers, cmplx );
  }
  else if ( noHarms*noSteps <= 64  )
  {
    long64 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->pIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    //cudaFuncSetCacheConfig(add_and_searchCU32<FLAGS,noStages,noHarms,cunkSize,noSteps,long64>, cudaFuncCachePreferL1);
    add_and_searchCU32<FLAGS,noStages,noHarms,cunkSize,noSteps,long64><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData,rBin, powers, cmplx );
  }
  /*
  else if ( noHarms*noSteps <= 128 )
  {
    long128 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->pIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    //cudaFuncSetCacheConfig(add_and_searchCU32<FLAGS,noStages,noHarms,cunkSize,noSteps,long128>, cudaFuncCachePreferL1);
    add_and_searchCU32<FLAGS,noStages,noHarms,cunkSize,noSteps,long128><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData,rBin, powers, cmplx );
  }
  */
  else
  {
    fprintf(stderr,"ERROR: %s has not been set up to work with %i elements.",__FUNCTION__, noHarms*noSteps);
  }
}

template<uint FLAGS, int noStages, const int noHarms, const int cunkSize>
__host__ void add_and_searchCU32_q(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int noSteps = batch->noSteps ;

  switch ( noSteps )
  {
    /*
    case 1:
    {
      add_and_searchCU32_s<FLAGS,noStages,noHarms,cunkSize,1>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 2:
    {
      add_and_searchCU32_s<FLAGS,noStages,noHarms,cunkSize,2>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 3:
    {
      add_and_searchCU32_s<FLAGS,noStages,noHarms,cunkSize,3>(dimGrid, dimBlock, stream, batch);
      break;
    }
    */
    case 4:
    {
      add_and_searchCU32_s<FLAGS,noStages,noHarms,cunkSize,4>(dimGrid, dimBlock, stream, batch);
      break;
    }
    /*
    case 5:
    {
      add_and_searchCU32_s<FLAGS,noStages,noHarms,cunkSize,5>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 6:
    {
      add_and_searchCU32_s<FLAGS,noStages,noHarms,cunkSize,6>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 7:
    {
      add_and_searchCU32_s<FLAGS,noStages,noHarms,cunkSize,7>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 8:
    {
      add_and_searchCU32_s<FLAGS,noStages,noHarms,cunkSize,8>(dimGrid, dimBlock, stream, batch);
      break;
    }
    */
    default:
      fprintf(stderr, "ERROR: add_and_searchCU32 has not been templated for %i steps\n", noSteps);
      exit(EXIT_FAILURE);
  }
}

template<uint FLAGS >
__host__ void add_and_searchCU32_p(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int noStages = batch->noHarmStages;

  switch ( noStages )
  {
    /*
    case 1:
    {
      add_and_searchCU32_q<FLAGS,1,1,4>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 2:
    {
      add_and_searchCU32_q<FLAGS,2,2,8>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 3:
    {
      add_and_searchCU32_q<FLAGS,3,4,8>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 4:
    {
      add_and_searchCU32_q<FLAGS,4,8,8>(dimGrid, dimBlock, stream, batch);
      break;
    }
    */
    case 5:
    {
      switch ( globalInt01 )
      {
        case 1:
        {
          add_and_searchCU32_q<FLAGS,5,16,1>(dimGrid, dimBlock, stream, batch);
          break;
        }
        case 2:
        {
          add_and_searchCU32_q<FLAGS,5,16,2>(dimGrid, dimBlock, stream, batch);
          break;
        }
        case 3:
        {
          add_and_searchCU32_q<FLAGS,5,16,3>(dimGrid, dimBlock, stream, batch);
          break;
        }
        case 4:
        {
          add_and_searchCU32_q<FLAGS,5,16,4>(dimGrid, dimBlock, stream, batch);
          break;
        }
        case 5:
        {
          add_and_searchCU32_q<FLAGS,5,16,5>(dimGrid, dimBlock, stream, batch);
          break;
        }
        case 6:
        {
          add_and_searchCU32_q<FLAGS,5,16,6>(dimGrid, dimBlock, stream, batch);
          break;
        }
        case 7:
        {
          add_and_searchCU32_q<FLAGS,5,16,7>(dimGrid, dimBlock, stream, batch);
          break;
        }
        case 8:
        {
          add_and_searchCU32_q<FLAGS,5,16,8>(dimGrid, dimBlock, stream, batch);
          break;
        }
        case 9:
        {
          add_and_searchCU32_q<FLAGS,5,16,9>(dimGrid, dimBlock, stream, batch);
          break;
        }
        case 10:
        {
          add_and_searchCU32_q<FLAGS,5,16,10>(dimGrid, dimBlock, stream, batch);
          break;
        }
        case 11:
        {
          add_and_searchCU32_q<FLAGS,5,16,11>(dimGrid, dimBlock, stream, batch);
          break;
        }
        case 12:
        {
          add_and_searchCU32_q<FLAGS,5,16,12>(dimGrid, dimBlock, stream, batch);
          break;
        }
        case 14:
        {
          add_and_searchCU32_q<FLAGS,5,16,14>(dimGrid, dimBlock, stream, batch);
          break;
        }
        case 16:
        {
          add_and_searchCU32_q<FLAGS,5,16,16>(dimGrid, dimBlock, stream, batch);
          break;
        }
        case 20:
        {
          add_and_searchCU32_q<FLAGS,5,16,20>(dimGrid, dimBlock, stream, batch);
          break;
        }
        case 24:
        {
          add_and_searchCU32_q<FLAGS,5,16,24>(dimGrid, dimBlock, stream, batch);
          break;
        }
        default:
          fprintf(stderr, "ERROR: %s has not been templated for globalInt01 %i\n", __FUNCTION__, globalInt01);
          exit(EXIT_FAILURE);
      }
      break;
    }
    default:
      fprintf(stderr, "ERROR: %s has not been templated for %i stages\n", __FUNCTION__, noStages);
      exit(EXIT_FAILURE);
  }
}

__host__ void add_and_searchCU32_f(cudaStream_t stream, cuFFdotBatch* batch )
{
  const uint FLAGS = batch->flag;
  dim3 dimBlock, dimGrid;

  dimBlock.x  = SS32_X;
  dimBlock.y  = SS32_Y;

  float bw    = SS32_X * SS32_Y;
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
    /*
    if      ( FLAGS & FLAG_ITLV_ROW )
      add_and_searchCU32_p<FLAG_ITLV_ROW> (dimGrid, dimBlock, stream, batch);
    else if ( FLAGS & FLAG_ITLV_PLN )
      add_and_searchCU32_p<FLAG_ITLV_PLN> (dimGrid, dimBlock, stream, batch);
    else
    */
    {
      fprintf(stderr, "ERROR: %s has not been templated for flag combination. \n", __FUNCTION__ );
      exit(EXIT_FAILURE);
    }
  }
}
