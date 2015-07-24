#include "cuda_accel_SS.h"

/** Sum and Search - loop down - column max - multi-step - step outer .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSteps
 */
template<uint FLAGS, const int noStages, const int noHarms, const int cunkSize, const int noSteps/*, typename stpType*/>
__global__ void add_and_searchCU31(const uint width, candPZ* d_cands, /*stpType rBin,*/ tHarmList texs, fsHarmList powersArr, cHarmList cmplxArr )
{
  const int bidx  = threadIdx.y * SS3_X         +  threadIdx.x;   /// Block index
  const int tid   = blockIdx.x  * (SS3_Y*SS3_X) +  bidx;          /// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    const int zeroHeight  = HEIGHT_STAGE[0];
    const int oStride     = STRIDE_STAGE[0];                            /// The stride of the output data

    //int             inds      [noSteps][noHarms];
    int             inds      [noHarms];
    candPZ          candLists [noStages][noSteps];
    float           powers    [noSteps][cunkSize];                /// registers to hold values to increase mem cache hits
    int             stride    [noHarms];

    FOLD // Prep - Initialise the x indices & set candidates to 0 .
    {
      // Calculate the x indices or create a pointer offset by the correct amount
      //#pragma unroll
      for ( int harm = 0; harm < noHarms; harm++ )                /// loop over harmonic  .
      {
        if ( FLAGS & FLAG_ITLV_ROW )
        {
          stride[harm] = noSteps*STRIDE_STAGE[harm] ;
        }
        else
        {
          stride[harm] = STRIDE_STAGE[harm] ;
        }

        //// NOTE: the indexing below assume each plain starts on a multiple of noHarms
        int   ix      = roundf( tid*FRAC_STAGE[harm] ) + HWIDTH_STAGE[harm] ;
        inds[harm]    = ix;


        //#pragma unroll
//                for ( int step = 0; step < noSteps; step++)               /// Loop over steps
//                {
//                  float fx    = (rBin.val[step] + tid)*FRAC_STAGE[harm] - rBin.val[harm*noSteps + step]  + HWIDTH_STAGE[harm] ;
//                  int   ix    = roundf(fx) ;
//
//                  if ( FLAGS & FLAG_ITLV_ROW )
//                  {
//                    ix += step*STRIDE_STAGE[harm] ;
//                  }
//
//                  inds[step][harm] = ix;
//                }
      }

      FOLD  // Set the local and return candidate powers to zero
      {
        //#pragma unroll
        for ( int stage = 0; stage < noStages; stage++ )
        {
          //#pragma unroll
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
        // Initialise powers for each section column to 0
        //#pragma unroll
        for ( int step = 0; step < noSteps; step++)                 // Loop over steps .
        {
          //#pragma unroll
          for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )          // Loop over powers .
          {
            powers[step][yPlus] = 0;
          }
        }

        FOLD // Loop over stages, sum and search  .
        {
          //#pragma unroll
          for ( int stage = 0 ; stage < noStages; stage++)          // Loop over stages .
          {
            int start = STAGE[stage][0] ;
            int end   = STAGE[stage][1] ;

            // Create a section of summed powers one for each step
            //#pragma unroll
            for ( int harm = start; harm <= end; harm++ )           // Loop over harmonics (batch) in this stage  .
            {
              int ix1 = inds[harm] ;
              int ix2 = ix1;

              //#pragma unroll
              for( int yPlus = 0; yPlus < cunkSize; yPlus++ )       // Loop over the chunk  .
              {
                int trm     = y + yPlus ;                           ///< True Y index in plain

                int iy1     = YINDS[ zeroHeight*harm + trm ];
                //  OR
                //int iy1     = roundf( (HEIGHT_STAGE[harm]-1.0)*trm/(float)(zeroHeight-1.0) ) ;

                int iy2;

                //#pragma unroll
                for ( int step = 0; step < noSteps; step++)         // Loop over steps  .
                {
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
                      powers[step][yPlus]   += cmpf;
                    }
                    else
                    {
                      const float r         = tex2D < float > (texs.val[harm], ix2*2+0.5f, iy2+0.5f ); // + 0.5 YES + 0.5 I REALLY wish someone had documented that one, 2 days of debugging to find that!!!!!!
                      const float i         = tex2D < float > (texs.val[harm], ix2*2+1.5f, iy2+0.5f ); // + 0.5 YES + 0.5 I REALLY wish someone had documented that one, 2 days of debugging to find that!!!!!!
                      powers[step][yPlus]   += r*r+i*i;
                    }
                  }
                  else
                  {
                    if      ( FLAGS & FLAG_MUL_CB_OUT )
                    {
                      float cmpf            = powersArr[harm][ iy2 + ix2 ];
                      powers[step][yPlus]  += cmpf;
                    }
                    else
                    {
                      fcomplexcu cmpc       = cmplxArr[harm][ iy2 + ix2 ];
                      powers[step][yPlus]  += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                    }
                  }

                }
              }
            }

            // Search set of powers
            //#pragma unroll
            for ( int step = 0; step < noSteps; step++)           // Loop over steps
            {
              //#pragma unroll
              for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )     // Loop over section
              {
                if  (  powers[step][yPlus] > POWERCUT_STAGE[stage] )
                {
                  if ( powers[step][yPlus] > candLists[stage][step].value )
                  {
                    if ( y + yPlus < zeroHeight )
                    {
                      // This is our new max!
                      candLists[stage][step].value  = powers[step][yPlus];
                      candLists[stage][step].z      = y+yPlus;
                    }
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
      for ( int step = 0; step < noSteps; step++)             // Loop over steps
      {
        //#pragma unroll
        for ( int stage = 0 ; stage < noStages; stage++)      // Loop over stages
        {
          if  ( candLists[stage][step].value >  POWERCUT_STAGE[stage] )
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
      add_and_searchCU31<FLAGS,noStages,noHarms,cunkSize,1><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 2:
    {
      //add_and_searchCU31_s<FLAGS,noStages,noHarms,cunkSize,2>(dimGrid, dimBlock, stream, batch);
      add_and_searchCU31<FLAGS,noStages,noHarms,cunkSize,2><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 3:
    {
      //add_and_searchCU31_s<FLAGS,noStages,noHarms,cunkSize,3>(dimGrid, dimBlock, stream, batch);
      add_and_searchCU31<FLAGS,noStages,noHarms,cunkSize,3><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 4:
    {
      //add_and_searchCU31_s<FLAGS,noStages,noHarms,cunkSize,4>(dimGrid, dimBlock, stream, batch);
      add_and_searchCU31<FLAGS,noStages,noHarms,cunkSize,4><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 5:
    {
      //add_and_searchCU31_s<FLAGS,noStages,noHarms,cunkSize,5>(dimGrid, dimBlock, stream, batch);
      add_and_searchCU31<FLAGS,noStages,noHarms,cunkSize,5><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 6:
    {
      //add_and_searchCU31_s<FLAGS,noStages,noHarms,cunkSize,6>(dimGrid, dimBlock, stream, batch);
      add_and_searchCU31<FLAGS,noStages,noHarms,cunkSize,6><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 7:
    {
      //add_and_searchCU31_s<FLAGS,noStages,noHarms,cunkSize,7>(dimGrid, dimBlock, stream, batch);
      add_and_searchCU31<FLAGS,noStages,noHarms,cunkSize,7><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 8:
    {
      //add_and_searchCU31_s<FLAGS,noStages,noHarms,cunkSize,8>(dimGrid, dimBlock, stream, batch);
      add_and_searchCU31<FLAGS,noStages,noHarms,cunkSize,8><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (candPZ*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    default:
      fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for %i steps\n", noSteps);
      exit(EXIT_FAILURE);
  }
}

template<uint FLAGS >
__host__ void add_and_searchCU31_p(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int noStages = batch->noHarmStages;

  switch (noStages)
  {
    case 1:
    {
      add_and_searchCU31_q<FLAGS,1,1,4>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 2:
    {
      add_and_searchCU31_q<FLAGS,2,2,8>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 3:
    {
      add_and_searchCU31_q<FLAGS,3,4,8>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 4:
    {
      add_and_searchCU31_q<FLAGS,4,8,8>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 5:
    {
      add_and_searchCU31_q<FLAGS,5,16,8>(dimGrid, dimBlock, stream, batch);
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

  dimBlock.x  = SS3_X;
  dimBlock.y  = SS3_Y;

  float bw    = SS3_X * SS3_Y;
  float ww    = batch->accelLen / ( bw );

  dimGrid.x   = ceil(ww);
  dimGrid.y   = 1;

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
