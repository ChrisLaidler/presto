#include "cuda_accel_SS.h"

#define SS31_X           16                    // X Thread Block
#define SS31_Y           8                     // Y Thread Block
#define SS31BS           (SS31_X*SS31_Y)

/** Sum and Search - loop down - column max - multi-step - step outer .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSteps
 */
template<uint FLAGS, const int noStages, const int noHarms, const int cunkSize, const int noSteps>
__global__ void add_and_searchCU31(const uint width, candPZs* d_cands, tHarmList texs, fsHarmList powersArr, cHarmList cmplxArr )
{
  const int bidx  = threadIdx.y * SS31_X  +  threadIdx.x;     /// Block index
  const int tid   = blockIdx.x  * SS31BS  +  bidx;            /// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    const int zeroHeight  = HEIGHT_STAGE[0];
    const int oStride     = STRIDE_STAGE[0];                    /// The stride of the output data

    int             inds      [noHarms];
    candPZs         candLists [noStages][noSteps];
    float           powers    [noSteps][cunkSize];              /// registers to hold values to increase mem cache hits

    FOLD // Prep - Initialise the x indices & set candidates to 0 .
    {
      // Calculate the x indices or create a pointer offset by the correct amount
      for ( int harm = 0; harm < noHarms; harm++ )                	// loop over harmonic  .
      {
        //// NOTE: the indexing below assume each plain starts on a multiple of noHarms
        int   ix        = roundf( tid*FRAC_STAGE[harm] ) + HWIDTH_STAGE[harm] ;
        inds[harm]      = ix;
      }

      FOLD  // Set the local and return candidate powers to zero
      {
        for ( int stage = 0; stage < noStages; stage++ )
        {
          for ( int step = 0; step < noSteps; step++)               // Loop over steps
          {
            candLists[stage][step].value = 0 ;
            d_cands[step*noStages*oStride + stage*oStride + tid ].value = 0;
          }
        }
      }
    }

    FOLD // Sum & Search - Ignore contaminated ends tid to starts at correct spot  .
    {
      for( int y = 0; y < zeroHeight ; y += cunkSize )              // loop over chunks .
      {
        FOLD // Initialise powers for each section column to 0  .
        {
          for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )          // Loop over powers .
          {
            for ( int step = 0; step < noSteps; step++)             // Loop over steps .
            {
              powers[step][yPlus]       = 0 ;
            }
          }
        }

        FOLD // Loop over stages, sum and search  .
        {
          for ( int stage = 0 ; stage < noStages; stage++)          // Loop over stages .
          {
            int start = STAGE[stage][0] ;
            int end   = STAGE[stage][1] ;

            FOLD // Create a section of summed powers one for each step  .
            {
              for ( int harm = start; harm <= end; harm++ )         // Loop over harmonics (batch) in this stage  .
              {
                int ix1       = inds[harm] ;
                int ix2       = ix1;
                float* t      = powersArr[harm];
                int   iyP     = -1;
                float pow[noSteps];

                for( int yPlus = 0; yPlus < cunkSize; yPlus++ )     // Loop over the chunk  .
                {
                  int trm     = y + yPlus ;                         ///< True Y index in plain
                  int iy1     = YINDS[ zeroHeight*harm + trm ];
                  //  OR
                  //int iy1     = roundf( (HEIGHT_STAGE[harm]-1.0)*trm/(float)(zeroHeight-1.0) ) ;

                  int iy2;

                  if ( iyP != iy1 ) // Only read power if it is not the same as the previous
                  {
                    for ( int step = 0; step < noSteps; step++)     // Loop over steps  .
                    {
                      FOLD // Calculate index  .
                      {
                        if        ( FLAGS & FLAG_ITLV_PLN )
                        {
                          iy2 = ( iy1 + step * HEIGHT_STAGE[harm] ) * STRIDE_STAGE[harm] ;
                        }
                        else
                        {
                          ix2 = ix1 + step * STRIDE_STAGE[harm] ;
                          iy2 = iy1 * noSteps * STRIDE_STAGE[harm];
                        }
                      }

                      FOLD // Read powers  .
                      {
                        if      ( FLAGS & FLAG_MUL_CB_OUT )
                        {
                          //pow[step]             = powersArr[harm][ iy2 + ix2 ];
                          pow[step]             = t[ iy2 + ix2 ];
                        }
                        else
                        {
                          fcomplexcu cmpc       = cmplxArr[harm][ iy2 + ix2 ];
                          pow[step]             = cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                        }
                      }
                    }

                    iyP = iy1;
                  }

                  FOLD // // Accumulate powers  .
                  {
                    for ( int step = 0; step < noSteps; step++)     // Loop over steps  .
                    {
                      powers[step][yPlus] += pow[step];
                    }
                  }
                }
              }
            }

            FOLD // Search set of powers  .
            {
              for ( int step = 0; step < noSteps; step++)           // Loop over steps
              {
                float pow;
                float maxP = POWERCUT_STAGE[stage];
                short maxI;

                for( short yPlus = 0; yPlus < cunkSize ; yPlus++ )    // Loop over section
                {
                  pow = powers[step][yPlus];

                  if  ( pow > maxP )
                  {
                    short idx = y + yPlus;

                    if ( idx < zeroHeight )
                    {
                      maxP = pow;
                      maxI = idx;
                    }
                  }
                }

                if  (  maxP > POWERCUT_STAGE[stage] )
                {
                  if ( maxP > candLists[stage][step].value )
                  {
                    // This is our new max!
                    candLists[stage][step].value  = maxP;
                    candLists[stage][step].z      = maxI;
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
      for ( int step = 0; step < noSteps; step++)             // Loop over steps
      {
        for ( int stage = 0 ; stage < noStages; stage++)      // Loop over stages
        {
          if  ( candLists[stage][step].value > POWERCUT_STAGE[stage] )
          {
            // Write to DRAM
            d_cands[step*noStages*oStride + stage*oStride + tid] = candLists[stage][step];
          }
        }
      }
    }
  }
}

template<uint FLAGS, int noStages, const int noHarms, const int cunkSize>
__host__ void add_and_searchCU31_q(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int noSteps = batch->noSteps ;

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
      //add_and_searchCU31_s<FLAGS,noStages,noHarms,cunkSize,1>(dimGrid, dimBlock, stream, batch);
      add_and_searchCU31<FLAGS,noStages,noHarms,cunkSize,1><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 2:
    {
      //add_and_searchCU31_s<FLAGS,noStages,noHarms,cunkSize,2>(dimGrid, dimBlock, stream, batch);
      add_and_searchCU31<FLAGS,noStages,noHarms,cunkSize,2><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 3:
    {
      //add_and_searchCU31_s<FLAGS,noStages,noHarms,cunkSize,3>(dimGrid, dimBlock, stream, batch);
      add_and_searchCU31<FLAGS,noStages,noHarms,cunkSize,3><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 4:
    {
      //add_and_searchCU31_s<FLAGS,noStages,noHarms,cunkSize,4>(dimGrid, dimBlock, stream, batch);
      add_and_searchCU31<FLAGS,noStages,noHarms,cunkSize,4><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 5:
    {
      //add_and_searchCU31_s<FLAGS,noStages,noHarms,cunkSize,5>(dimGrid, dimBlock, stream, batch);
      add_and_searchCU31<FLAGS,noStages,noHarms,cunkSize,5><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 6:
    {
      //add_and_searchCU31_s<FLAGS,noStages,noHarms,cunkSize,6>(dimGrid, dimBlock, stream, batch);
      add_and_searchCU31<FLAGS,noStages,noHarms,cunkSize,6><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 7:
    {
      //add_and_searchCU31_s<FLAGS,noStages,noHarms,cunkSize,7>(dimGrid, dimBlock, stream, batch);
      add_and_searchCU31<FLAGS,noStages,noHarms,cunkSize,7><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 8:
    {
      //add_and_searchCU31_s<FLAGS,noStages,noHarms,cunkSize,8>(dimGrid, dimBlock, stream, batch);
      add_and_searchCU31<FLAGS,noStages,noHarms,cunkSize,8><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZs*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    default:
      fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for %i steps\n", noSteps);
      exit(EXIT_FAILURE);
  }
}

template<uint FLAGS, int noStages, const int noHarms>
__host__ void add_and_searchCU31_c(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  switch (globalInt01)
  {
    case 1:
    {
      add_and_searchCU31_q<FLAGS,noStages,noHarms,1>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 2:
    {
      add_and_searchCU31_q<FLAGS,noStages,noHarms,2>(dimGrid, dimBlock, stream, batch);
      break;
    }
    //    case 3:
    //    {
    //      add_and_searchCU31_q<FLAGS,noStages,noHarms,3>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    case 4:
    {
      add_and_searchCU31_q<FLAGS,noStages,noHarms,4>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 5:
    {
      add_and_searchCU31_q<FLAGS,noStages,noHarms,5>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 6:
    {
      add_and_searchCU31_q<FLAGS,noStages,noHarms,6>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 7:
    {
      add_and_searchCU31_q<FLAGS,noStages,noHarms,7>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 8:
    {
      add_and_searchCU31_q<FLAGS,noStages,noHarms,8>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 9:
    {
      add_and_searchCU31_q<FLAGS,noStages,noHarms,9>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 10:
    {
      add_and_searchCU31_q<FLAGS,noStages,noHarms,10>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 12:
    {
      add_and_searchCU31_q<FLAGS,noStages,noHarms,12>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 14:
    {
      add_and_searchCU31_q<FLAGS,noStages,noHarms,14>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 16:
    {
      add_and_searchCU31_q<FLAGS,noStages,noHarms,16>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 18:
    {
      add_and_searchCU31_q<FLAGS,noStages,noHarms,18>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 20:
    {
      add_and_searchCU31_q<FLAGS,noStages,noHarms,20>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 24:
    {
      add_and_searchCU31_q<FLAGS,noStages,noHarms,24>(dimGrid, dimBlock, stream, batch);
      break;
    }
    default:
      fprintf(stderr, "ERROR: %s has not been templated for %i chunk size.\n", __FUNCTION__, globalInt01);
      exit(EXIT_FAILURE);
  }

}

template<uint FLAGS >
__host__ void add_and_searchCU31_p(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int noStages = batch->noHarmStages;

  switch (noStages)
  {
    //    case 1:
    //    {
    //      add_and_searchCU31_c<FLAGS,1,1>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    //    case 2:
    //    {
    //      add_and_searchCU31_c<FLAGS,2,2>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    //    case 3:
    //    {
    //      add_and_searchCU31_c<FLAGS,3,4>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    //    case 4:
    //    {
    //      add_and_searchCU31_c<FLAGS,4,8>(dimGrid, dimBlock, stream, batch);
    //      break;
    //    }
    case 5:
    {
      add_and_searchCU31_c<FLAGS,5,16>(dimGrid, dimBlock, stream, batch);
      break;
    }
    default:
      fprintf(stderr, "ERROR: %s has not been templated for %i stages\n", __FUNCTION__, noStages);
      exit(EXIT_FAILURE);
  }
}

__host__ void add_and_searchCU31( cudaStream_t stream, cuFFdotBatch* batch )
{
  const uint FLAGS = batch->flag;
  dim3 dimBlock, dimGrid;

  dimBlock.x  = SS31_X;
  dimBlock.y  = SS31_Y;

  float bw    = SS31BS;
  float ww    = batch->accelLen / ( bw );

  dimGrid.x   = ceil(ww);
  dimGrid.y   = 1;

  FOLD // Call flag template  .
  {
    if        ( FLAGS & FLAG_MUL_CB_OUT )
    {
      if      ( FLAGS & FLAG_ITLV_ROW )
        add_and_searchCU31_p<FLAG_MUL_CB_OUT | FLAG_ITLV_ROW> (dimGrid, dimBlock, stream, batch);
      else if ( FLAGS & FLAG_ITLV_PLN )
        add_and_searchCU31_p<FLAG_MUL_CB_OUT | FLAG_ITLV_PLN>  (dimGrid, dimBlock, stream, batch);
      else
      {
        fprintf(stderr, "ERROR: %s has not been templated for flag combination. \n", __FUNCTION__ );
        exit(EXIT_FAILURE);
      }
    }
    else
    {
      if      ( FLAGS & FLAG_ITLV_ROW )
        add_and_searchCU31_p<FLAG_ITLV_ROW> (dimGrid, dimBlock, stream, batch);
      else if ( FLAGS & FLAG_ITLV_PLN )
        add_and_searchCU31_p<FLAG_ITLV_PLN> (dimGrid, dimBlock, stream, batch);
      else
      {
        fprintf(stderr, "ERROR: %s has not been templated for flag combination. \n", __FUNCTION__ );
        exit(EXIT_FAILURE);
      }
    }
  }
}
