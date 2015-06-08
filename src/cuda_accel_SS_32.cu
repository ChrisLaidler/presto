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
template<uint FLAGS, const int noStages, const int noHarms, const int cunkSize, const int noSteps>
__global__ void add_and_searchCU32_k(const uint width, accelcandBasic* d_cands, tHarmList texs, fsHarmList powersArr, cHarmList cmplxArr )
{
  const int bidx  = threadIdx.y * SS33_X          +  threadIdx.x;   /// Block index
  const int tid   = blockIdx.x  * (SS33_Y*SS33_X) +  bidx;          /// Global thread id (ie column) 0 is the first 'good' column

  __shared__ float smBlock[cunkSize][SS33_Y*SS33_X];

  if ( tid < width )
  {
    const int zeroHeight  = HEIGHT_STAGE[0];
    const int oStride     = STRIDE_STAGE[0];                            /// The stride of the output data

    //const int zh1         = zeroHeight-1;
    //const int zh2         = zeroHeight-2;

    //int             inds      [noHarms];
    //accelcandBasic  candLists [noStages][noSteps];
    //float           powers    [noSteps][cunkSize];                /// registers to hold values to increase mem cache hits
    int             stride    [noHarms];
    int             x0        [noHarms];
    int             x1        [noHarms];

    FOLD // Prep - Initialise the x indices & set candidates to 0 .
    {
      // Calculate the x indices or create a pointer offset by the correct amount
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

        //for ( int step = 0; step < noSteps; step++)               /// Loop over steps
        {
          // NOTE: the indexing below assume each plain starts on a multiple of noHarms
          int   ix    = roundf( tid*FRAC_STAGE[harm] ) + HWIDTH_STAGE[harm] ;

//          if ( FLAGS & FLAG_ITLV_ROW )
//          {
//            ix += step*STRIDE_STAGE[harm] ;
//          }

          //inds[harm] = ix;
        }

        x0[harm] = roundf( blockIdx.x  * (SS33_Y*SS33_X)*FRAC_STAGE[harm] ) + HWIDTH_STAGE[harm] ;
        x1[harm] = roundf( blockIdx.x  * (  (blockIdx.x+1)  * (SS33_Y*SS33_X)-1)*FRAC_STAGE[harm] ) + HWIDTH_STAGE[harm] - x0[harm] ;
      }

      FOLD  // Set the local and return candidate powers to zero
      {
        for ( int stage = 0; stage < noStages; stage++ )
        {
          for ( int step = 0; step < noSteps; step++)               // Loop over steps
          {
            //candLists[stage][step].sigma = 0 ;
            d_cands[step*noStages*oStride + stage*oStride + tid ].sigma = 0;
          }
        }
      }
    }

    FOLD // Sum & Search - Ignore contaminated ends tid to starts at correct spot  .
    {
      for( int y = 0; y < zeroHeight ; y += cunkSize )              // loop over chunks .
      {
//        FOLD // Initialise powers for each section column to 0  .
//        {
//          for ( int step = 0; step < noSteps; step++)                 // Loop over steps .
//          {
//            for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )          // Loop over powers .
//            {
//              powers[step][yPlus] = 0;
//            }
//          }
//        }

        FOLD // Loop over stages, sum and search  .
        {
          for ( int stage = 0 ; stage < noStages; stage++)          // Loop over stages .
          {
            int start = STAGE[stage][0] ;
            int end   = STAGE[stage][1] ;

            // Create a section of summed powers one for each step
            for ( int harm = start; harm <= end; harm++ )           // Loop over harmonics (batch) in this stage  .
            {
              //int ix        = inds[harm] ;
              //int ix2       = ix;
              //int h1      = HEIGHT_STAGE[harm]-1;

              int ix        = x0[harm] + bidx ;
              int ix2       = ix ;

              if ( bidx < x1[harm] )
              {

                int y0 = YINDS[ zeroHeight*harm + y ];
                int y1 = YINDS[ zeroHeight*harm + y + cunkSize ];

                //for( int yPlus = 0; yPlus < cunkSize; yPlus++ )       // Loop over the chunk  .
                for( int yPlus = y0; yPlus < y1; yPlus++ )       // Loop over the chunk  .
                {
                  //int trm     = y + yPlus ;                           ///< True Y index in plain

                  //int iy1     = YINDS[ zeroHeight*harm + trm ];
                  //  OR
                  //int iy1     = roundf( (HEIGHT_STAGE[harm]-1.0)*trm/(float)(zeroHeight-1.0) ) ;
                  // OR
                  //int iy1     = ( h1 * trm + zh2 ) / zh1;

                  int iy1     = yPlus ;
                  int iy2     = iy1 * stride[harm] ;

                  for ( int step = 0; step < noSteps; step++)         // Loop over steps  .
                  {
                    if        ( FLAGS & FLAG_ITLV_PLN )
                    {
                      iy2 = iy1 + step * HEIGHT_STAGE[harm];                // stride step by plain
                    }
                    else
                    {
                      ix2 = ix + step * STRIDE_STAGE[harm] ;
                    }

                    FOLD
                    {
                      if      ( FLAGS & FLAG_MUL_CB_OUT )
                      {
                        float cmpf            = powersArr[harm][ iy2 + ix2 ];
                        //powers[step][yPlus]  += cmpf;

                        if ( cmpf < 0 ) // TMP
                          printf("SS33");
                        //atomicAdd(&smBlock[yPlus][tid], cmpf);
                      }
                      else
                      {
                        //fcomplexcu cmpc       = cmplxArr[harm][ iy2 + ix2 ];
                        //float cmpf            = cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                        //powers[step][yPlus]  += cmpc.r * cmpc.r + cmpc.i * cmpc.i;

//                        if ( cmpf < 0 ) // TMP
//                          printf("SS33");
                      }
                    }
                  }
                }
              }
            }
            /*
            // Search set of powers
            for ( int step = 0; step < noSteps; step++)           // Loop over steps
            {
              for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )     // Loop over section
              {
                if  (  powers[step][yPlus] > POWERCUT_STAGE[stage] )
                {
                  if ( powers[step][yPlus] > candLists[stage][step].sigma )
                  {
                    if ( y + yPlus < zeroHeight )
                    {
                      // This is our new max!
                      candLists[stage][step].sigma  = powers[step][yPlus];
                      candLists[stage][step].z      = y+yPlus;
                    }
                  }
                }
              }
            }
            */
          }
        }
      }
    }
/*
    FOLD // Write results back to DRAM and calculate sigma if needed  .
    {
      for ( int step = 0; step < noSteps; step++)             // Loop over steps
      {
        for ( int stage = 0 ; stage < noStages; stage++)      // Loop over stages
        {
          if  ( candLists[stage][step].sigma >  POWERCUT_STAGE[stage] )
          {
            if ( (FLAGS & FLAG_SIG_GPU) && FALSE)             // Calculate the actual sigma value on the GPU
            {
              const int numharm                 = ( 1 << stage );
              // Calculate sigma value
              long long numtrials               = NUMINDEP_STAGE[stage];
              candLists[stage][step].sigma      = (float)candidate_sigma_cu(candLists[stage][step].sigma, numharm, numtrials);
            }

            // Write to DRAM
            d_cands[step*noStages*oStride + stage*oStride + tid] = candLists[stage][step];
          }
        }
      }
    }
*/
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
      int idx =  batch->pIdx[i];

      long long binb      = (*batch->rConvld)[step][idx].expBin ;

      if ( firstBin * batch->hInfos[idx].harmFrac != binb )
      {
        fprintf(stderr,"ERROR, in function %s, R values are not properly aligned! Each step should start on a multiple of (2 x No Harms).\n", __FUNCTION__ );
        exit(EXIT_FAILURE);
      }
    }
  }

  tHarmList   texs;
  fsHarmList powers;
  cHarmList   cmplx;

  for (int i = 0; i < noHarms; i++)
  {
    int idx         = batch->pIdx[i];
    texs.val[i]     = batch->plains[idx].datTex;
    powers.val[i]   = batch->plains[idx].d_plainPowers;
    cmplx.val[i]    = batch->plains[idx].d_plainData;
  }

  switch (noSteps)
  {
    case 1:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,1><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 2:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,2><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 3:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,3><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 4:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,4><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 5:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,5><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 6:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,6><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 7:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,7><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, texs, powers, cmplx );
      break;
    }
    case 8:
    {
      add_and_searchCU32_k<FLAGS,noStages,noHarms,cunkSize,8><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, texs, powers, cmplx );
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

