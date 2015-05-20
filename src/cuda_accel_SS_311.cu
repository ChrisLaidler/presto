#include "cuda_accel_SS.h"

/** Sum and Search - loop down - column max - multi-step - step outer .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSteps
 */
#if TEMPLATE_SEARCH == 1
template<uint FLAGS, typename sType, int noStages, typename stpType, int noSteps>
__global__ void add_and_searchCU311(cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base, sType pd, stpType rLows )
#else
template<uint FLAGS, typename sType, int noStages, typename stpType>
__global__ void add_and_searchCU311(cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base, sType pd, stpType rLows, int noSteps )
#endif
{
  /*
  const int bidx  = threadIdx.y * SS3_X         +  threadIdx.x;
  const int tid   = blockIdx.x  * (SS3_Y*SS3_X) +  bidx;

  const int width = searchList.widths.val[0];

  if ( tid < width )
  {
    const int noHarms   = ( 1 << (noStages-1) ) ;
    const int CHUNKSZ   = 17 ; // (noStages)*2;      // The number of powers to batch calculate together, *2 is a "random choice it would be this or noHarms

    //register float power;
    float powers[CHUNKSZ];            // registers to hold values to increase mem cache hits

    const int zeroHeight = searchList.heights.val[0] ;

    int nStride[noHarms];

#if TEMPLATE_SEARCH == 1
    accelcandBasic candLists[noStages];
    //register float maxP[noStages];
    //int z[noStages];

    // One of the two variables below should get optimised out depending on FLAG_STP_ROW or FLAG_STP_PLN
    int inds[noHarms];
    fcomplexcu* pData[noHarms];
    //float powers[CHUNKSZ];         // registers to hold values to increase mem cache hits
#else
    accelcandBasic candLists[noStages];
    //float maxP[noStages];
    //int z[noStages];

    // One of the two variables below should get optimised out depending on FLAG_STP_ROW or FLAG_STP_PLN
    int inds[noHarms];
    fcomplexcu* pData[noHarms];
    //float powers[CHUNKSZ];         // registers to hold values to increase mem cache hits
#endif

//#if TEMPLATE_SEARCH == 1
//#pragma unroll
//#endif
    for ( int step = 0; step < noSteps; step++)     // Loop over steps
    {
      int start   = 0;
      int end     = 0;
      int iy;
      int y;

      FOLD // Prep - Initialise the x indices & set candidates to 0 .
      {
        int ix;

        // Calculate the x indices or create a pointer offset by the correct amount
#pragma unroll
        for ( int harm = 0; harm < noHarms; harm++ )      // loop over harmonic
        {
          float rLow    = rLows.arry[step] * searchList.frac.val[harm];
          float diff    = rLow - (int)rLow;
          float idxS    = 0.5f + diff*ACCEL_RDR ;

          ix = (int)( tid * searchList.frac.val[harm] + idxS ) + searchList.ffdBuffre.val[harm];
          if     (FLAGS & FLAG_PLN_TEX)  // Calculate x index
          {
            inds[harm]      = ix;
          }
          else                           // Create a pointer list that is offset by the correct amount
          {
            inds[harm]      = ix;

            if        ( FLAGS & FLAG_STP_ROW )
            {
              pData[harm]   = &searchList.datas.val[harm][ ix + searchList.strides.val[harm]*step ] ;
            }
            else if   ( FLAGS & FLAG_STP_PLN )
            {
              pData[harm]   = &searchList.datas.val[harm][ ix + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
            }
          }

          // Change the stride for this harmonic
          if     ( !( FLAGS & FLAG_PLN_TEX ) && ( FLAGS & FLAG_STP_ROW ) )
          {
            //searchList.strides.val[harm] *= noSteps;
            nStride[harm] = searchList.strides.val[harm] * noSteps;
          }
        }

        // Set the local and return candidate powers to zero
        FOLD
        {
#pragma unroll
          for ( int stage = 0; stage < noStages; stage++ )
          {
            candLists[stage].sigma    = POWERCUT[stage];
            //maxP[stage]               = POWERCUT[stage];

            if ( FLAGS & CU_OUTP_SINGLE )
            {
              d_cands[step*noStages*width + stage*width + tid ].sigma = 0;
            }
          }
        }
      }

      FOLD // Sum & Search .
      {
        for( y = 0; y < zeroHeight ; y+=CHUNKSZ ) // loop over chunks  .
        {
          FOLD // Initialise powers for each section column to 0  .
          {
#pragma unroll
            for( int yPlus = 0; yPlus < CHUNKSZ; yPlus++ )                // Loop over the chunk
            {
              powers[yPlus] = 0;
            }
          }

          // Loop over stages, sum and search
#pragma unroll
          for ( int stage = 0 ; stage < noStages; stage++)        // Loop over stages .
          {
            if      ( stage == 0 )
            {
              start = 0;
              end = 1;
            }
            else if ( stage == 1 )
            {
              start = 1;
              end = 2;
            }
            else if ( stage == 2 )
            {
              start = 2;
              end = 4;
            }
            else if ( stage == 3 )
            {
              start = 4;
              end = 8;
            }
            else if ( stage == 4 )
            {
              start = 8;
              end = 16;
            }

            // Create a section of summed powers one for each step

#pragma unroll
            for ( int harm = start; harm < end; harm++ )          // Loop over harmonics (batch) in this stage
            {

#pragma unroll
              for( int yPlus = 0; yPlus < CHUNKSZ; yPlus++ )                // Loop over the chunk
              {
                int trm       = y + yPlus ;
                iy            = YINDS[ searchList.yInds.val[harm] + trm ] ;

                if     (FLAGS & FLAG_PLN_TEX)
                {
                  // Calculate y indice
                  if      ( FLAGS & FLAG_STP_ROW )
                  {
                    iy  = ( iy * noSteps + step );
                  }
                  else if ( FLAGS & FLAG_STP_PLN )
                  {
                    iy  = ( iy + searchList.heights.val[harm]*step ) ;
                  }

                  const float2 cmpf      = tex2D < float2 > (searchList.texs.val[harm], inds[harm], iy);
                  //power                 += cmpf.x * cmpf.x + cmpf.y * cmpf.y;
                  powers[yPlus]         += cmpf.x * cmpf.x + cmpf.y * cmpf.y;
                }
                else
                {
                  fcomplexcu cmpc;
                  if        ( FLAGS & FLAG_STP_ROW )
                  {
                    //cmpc = searchList.datas.val[harm][ inds[harm]  + searchList.strides.val[harm]*noSteps*iy + searchList.strides.val[harm]*step ] ;
                    cmpc = pData[harm][nStride[harm]*iy] ; // Note stride has been set depending on multi-step type
                  }
                  else if   ( FLAGS & FLAG_STP_PLN )
                  {
                    cmpc = searchList.datas.val[harm][ inds[harm]  + searchList.strides.val[harm]*iy + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
                  }
                  //power           += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                  powers[yPlus]   += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                }
              }
            }

            //#pragma unroll
            for( int yPlus = 0; yPlus < CHUNKSZ; yPlus++ )                // Loop over the chunk
            {
              if ( powers[yPlus] > candLists[stage].sigma )
              {
                if ( yPlus + y < zeroHeight)
                {
                  // This is our new max!
                  candLists[stage].sigma  = powers[yPlus];
                  candLists[stage].z      = y + yPlus;
                }
              }
            }

          }
        }
      }

      // Write results back to DRAM and calculate sigma if needed
      if      ( FLAGS & CU_OUTP_DEVICE   )
      {
        //#pragma unroll
        for ( int stage = 0 ; stage < noStages; stage++)
        {
          const short numharm = 1 << stage;

//#if TEMPLATE_SEARCH == 1
//#pragma unroll
//#endif
//          for ( int step = 0; step < noSteps; step++)         // Loop over steps
          {

            if  ( candLists[stage].sigma >  POWERCUT[stage] )
            {
              //float rLow    = rLows.arry[step] * searchList.frac.val[harm];
              //float diff    = rLow - (int)rLow;
              //float idxS    = 0.5  + diff*ACCEL_RDR ;

              int idx =  (int)(( rLows.arry[step] + tid * (double) ACCEL_DR ) / (double)numharm ) - base ;
              if ( idx >= 0 )
              {
                long long numtrials             = NUMINDEP[stage];
                candLists[stage].numharm  = numharm;
                //candLists[stage][step].z      = ( candLists[stage].z*(double) ACCEL_DZ - searchList.zMax.val[0]  )  / (double)numharm ;
                candLists[stage].sigma    = (float)candidate_sigma_cu(candLists[stage].sigma, numharm, numtrials);

                FOLD // Atomic write to global list
                {
                  volatile bool done = false;
                  while (!done)
                  {
                    volatile int prev = atomicCAS(&d_sem[idx], UINT_MAX, tid );
                    if ( prev == UINT_MAX )
                    {
                      if ( candLists[stage].sigma > d_cands[idx].sigma )
                      {
                        d_cands[idx]              = candLists[stage];
                      }
                      d_sem[idx]                  = UINT_MAX;
                      done = true;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else if ( FLAGS & CU_OUTP_SINGLE )
      {
//#if TEMPLATE_SEARCH == 1
//#pragma unroll
//#endif
//        for ( int step = 0; step < noSteps; step++)             // Loop over steps
        {
#pragma unroll
          for ( int stage = 0 ; stage < noStages; stage++)      // Loop over stages
          {

            if  ( candLists[stage].sigma >  POWERCUT[stage] )
            {
              const short numharm                 = ( 1 << stage );
              candLists[stage].numharm      = numharm;

              if ( FLAGS & FLAG_SIG_GPU && FALSE)
              {
                // Calculate sigma value
                long long numtrials               = NUMINDEP[stage];
                candLists[stage].sigma      = (float)candidate_sigma_cu(candLists[stage].sigma, numharm, numtrials);
              }

              // Write to DRAM
              d_cands[step*noStages*width + stage*width + tid] = candLists[stage];
            }
          }
        }
      }
    }
  }
*/
}

template<uint FLAGS, const int noStages, const int noHarms, const int cunkSize, const int noSteps, typename stpType>
__global__ void add_and_searchCU3111(const uint width, accelcandBasic* d_cands, stpType rBin, tHarmList texs, fsHarmList powersArr, cHarmList cmplxArr )
{
  const int bidx  = threadIdx.y * SS3_X         +  threadIdx.x;   /// Block index
  const int tid   = blockIdx.x  * (SS3_Y*SS3_X) +  bidx;          /// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    const int zeroHeight  = HEIGHT[0];
    const int oStride     = STRIDE[0];                            /// The stride of the output data

    int             inds      [noSteps][noHarms];
    accelcandBasic  candLists [noStages][noSteps];
    float           powers    [noSteps][cunkSize];                /// registers to hold values to increase mem cache hits
    int             stride    [noHarms];

    FOLD // Prep - Initialise the x indices & set candidates to 0 .
    {
      // Calculate the x indices or create a pointer offset by the correct amount
//#pragma unroll
      for ( int harm = 0; harm < noHarms; harm++ )                /// loop over harmonic  .
      {
        if ( FLAGS & FLAG_STP_ROW )
        {
          stride[harm] = noSteps*STRIDE[harm] ;
        }
        else
        {
          stride[harm] = STRIDE[harm] ;
        }

//#pragma unroll
        for ( int step = 0; step < noSteps; step++)               /// Loop over steps
        {
          float fx    = (rBin.val[step] + tid)*FRAC[harm] - rBin.val[harm*noSteps + step]  + HWIDTH[harm] ;
          int   ix    = round(fx) ;

          if ( FLAGS & FLAG_STP_ROW )
          {
            ix += step*STRIDE[harm] ;
          }

          inds[step][harm] = ix;
        }
      }

      FOLD  // Set the local and return candidate powers to zero
      {
//#pragma unroll
        for ( int stage = 0; stage < noStages; stage++ )
        {
//#pragma unroll
          for ( int step = 0; step < noSteps; step++)               // Loop over steps
          {
            candLists[stage][step].sigma = 0 ;
            d_cands[step*noStages*oStride + stage*oStride + tid ].sigma = 0;
          }
        }
      }
    }

    FOLD // Sum & Search - Ignore contaminated ends tid to starts at correct spot  .
    {
      for( int y = 0; y < zeroHeight ; y += cunkSize )               // loop over chunks .
      {
        // Initialise powers for each section column to 0
        //#pragma unroll
        for ( int step = 0; step < noSteps; step++)                 // Loop over steps .
        {
          //#pragma unroll
          for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )           // Loop over powers .
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
              //#pragma unroll
              for( int yPlus = 0; yPlus < cunkSize; yPlus++ )       // Loop over the chunk  .
              {
                int trm     = y + yPlus ;                           ///< True Y index in plain

                int iy1     = YINDS[ zeroHeight*harm + trm ];
                //  OR
                //int iy1     = round( (HEIGHT[harm]-1.0)*trm/(float)(zeroHeight-1.0) ) ;

                int iy2     = iy1*stride[harm];

                //#pragma unroll
                for ( int step = 0; step < noSteps; step++)         // Loop over steps  .
                {
                  int ix = inds[step][harm] ;

                  if        ( FLAGS & FLAG_STP_PLN )
                  {
                    iy2 = iy1 + step * HEIGHT[harm];                // stride step by plain
                  }

                  if        ( FLAGS & FLAG_PLN_TEX )
                  {
                    if      ( FLAGS & FLAG_CUFFTCB_OUT )
                    {
                      const float cmpf      = tex2D < float > (texs.val[harm], ix+0.5f, iy2+0.5f ); // + 0.5 YES + 0.5 I REALLY wish someone had documented that one, 2 days of debugging to find that!!!!!!
                      powers[step][yPlus]   += cmpf;
                    }
                    else
                    {
                      const float r         = tex2D < float > (texs.val[harm], ix*2+0.5f, iy2+0.5f ); // + 0.5 YES + 0.5 I REALLY wish someone had documented that one, 2 days of debugging to find that!!!!!!
                      const float i         = tex2D < float > (texs.val[harm], ix*2+1.5f, iy2+0.5f ); // + 0.5 YES + 0.5 I REALLY wish someone had documented that one, 2 days of debugging to find that!!!!!!
                      powers[step][yPlus]   += r*r+i*i;
                    }
                  }
                  else
                  {
                    if      ( FLAGS & FLAG_CUFFTCB_OUT )
                    {
                      float cmpf            = powersArr[harm][ ix + iy2 ];
                      powers[step][yPlus]  += cmpf;
                    }
                    else
                    {
                      fcomplexcu cmpc       = cmplxArr[harm][ ix + iy2 ];
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
                if  (  powers[step][yPlus] > POWERCUT[stage] )
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
          if  ( candLists[stage][step].sigma >  POWERCUT[stage] )
          {
            // This can be calculated from stage
            //const short numharm                 = ( 1 << stage );
            //candLists[stage][step].numharm      = numharm;

            if ( (FLAGS & FLAG_SIG_GPU) && FALSE)             // Calculate the actual sigma value on the GPU
            {
              const int numharm                 = ( 1 << stage );
              // Calculate sigma value
              long long numtrials               = NUMINDEP[stage];
              candLists[stage][step].sigma      = (float)candidate_sigma_cu(candLists[stage][step].sigma, numharm, numtrials);
            }

            // Write to DRAM
            d_cands[step*noStages*oStride + stage*oStride + tid] = candLists[stage][step];
          }
        }
      }
    }
  }
}

template<uint FLAGS, int noStages, const int noHarms, const int cunkSize, int noSteps>
__host__ void add_and_searchCU311_s(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  /*
  int heights[noHarms];
  int strides[noHarms];
  float  frac[noHarms];
  int  hWidth[noHarms];
  //__global__ long   rBin[noHarms][noSteps];

  int i = 0;
  for (int i = 0; i < noHarms; i++)
  {
    int idx =  batch->pIdx[i];

    heights[i]              = batch->hInfos[idx].height;
    strides[i]              = batch->hInfos[idx].inpStride;
    frac[i]                 = batch->hInfos[idx].harmFrac;
    hWidth[i]               = batch->hInfos[idx].halfWidth*ACCEL_NUMBETWEEN;

    for ( int step = 0; step < noSteps; step++)
    {
      rBin[i][step]         = (*batch->rConvld)[step][idx].expBin ;
    }
  }
  */

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
    //cudaFuncSetCacheConfig(add_and_searchCU3111<FLAGS,noStages,noHarms,cunkSize,noSteps,long08>, cudaFuncCachePreferL1);
    add_and_searchCU3111<FLAGS,noStages,noHarms,cunkSize,noSteps,long04><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, rBin, texs, powers, cmplx );
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
    //cudaFuncSetCacheConfig(add_and_searchCU3111<FLAGS,noStages,noHarms,cunkSize,noSteps,long08>, cudaFuncCachePreferL1);
    add_and_searchCU3111<FLAGS,noStages,noHarms,cunkSize,noSteps,long08><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, rBin, texs, powers, cmplx );
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
    //cudaFuncSetCacheConfig(add_and_searchCU3111<FLAGS,noStages,noHarms,cunkSize,noSteps,long16>, cudaFuncCachePreferL1);
    add_and_searchCU3111<FLAGS,noStages,noHarms,cunkSize,noSteps,long16><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData,rBin, texs, powers, cmplx );
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
    //cudaFuncSetCacheConfig(add_and_searchCU3111<FLAGS,noStages,noHarms,cunkSize,noSteps,long32>, cudaFuncCachePreferL1);
    add_and_searchCU3111<FLAGS,noStages,noHarms,cunkSize,noSteps,long32><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData,rBin, texs, powers, cmplx );
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
    //cudaFuncSetCacheConfig(add_and_searchCU3111<FLAGS,noStages,noHarms,cunkSize,noSteps,long64>, cudaFuncCachePreferL1);
    add_and_searchCU3111<FLAGS,noStages,noHarms,cunkSize,noSteps,long64><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData,rBin, texs, powers, cmplx );
  }
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
    //cudaFuncSetCacheConfig(add_and_searchCU3111<FLAGS,noStages,noHarms,cunkSize,noSteps,long128>, cudaFuncCachePreferL1);
    add_and_searchCU3111<FLAGS,noStages,noHarms,cunkSize,noSteps,long128><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData,rBin, texs, powers, cmplx );
  }
  else
  {
    fprintf(stderr,"ERROR: %s has not been set up to work with %i elements.",__FUNCTION__, noHarms*noSteps);
  }
}

template<uint FLAGS, int noStages, const int noHarms, const int cunkSize>
__host__ void add_and_searchCU311_q(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int noSteps = batch->noSteps ;

  switch (noSteps)
  {
    case 1:
    {
      add_and_searchCU311_s<FLAGS,noStages,noHarms,cunkSize,1>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 2:
    {
      add_and_searchCU311_s<FLAGS,noStages,noHarms,cunkSize,2>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 3:
    {
      add_and_searchCU311_s<FLAGS,noStages,noHarms,cunkSize,3>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 4:
    {
      add_and_searchCU311_s<FLAGS,noStages,noHarms,cunkSize,4>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 5:
    {
      add_and_searchCU311_s<FLAGS,noStages,noHarms,cunkSize,5>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 6:
    {
      add_and_searchCU311_s<FLAGS,noStages,noHarms,cunkSize,6>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 7:
    {
      add_and_searchCU311_s<FLAGS,noStages,noHarms,cunkSize,7>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 8:
    {
      add_and_searchCU311_s<FLAGS,noStages,noHarms,cunkSize,7>(dimGrid, dimBlock, stream, batch);
      break;
    }
    default:
      fprintf(stderr, "ERROR: add_and_searchCU311 has not been templated for %i steps\n", noSteps);
      exit(EXIT_FAILURE);
  }
}

template<uint FLAGS >
__host__ void add_and_searchCU311_p(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int noStages = batch->noHarmStages;

  switch (noStages)
  {
    case 1:
    {
      add_and_searchCU311_q<FLAGS,1,1,4>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 2:
    {
      add_and_searchCU311_q<FLAGS,2,2,8>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 3:
    {
      add_and_searchCU311_q<FLAGS,3,4,8>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 4:
    {
      add_and_searchCU311_q<FLAGS,4,8,8>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 5:
    {
      add_and_searchCU311_q<FLAGS,5,16,8>(dimGrid, dimBlock, stream, batch);
      break;
    }
    default:
      fprintf(stderr, "ERROR: %s has not been templated for %i stages\n", __FUNCTION__, noStages);
      exit(EXIT_FAILURE);
  }
}

__host__ void add_and_searchCU311_f(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const uint FLAGS = batch->flag;

    {
      if        ( FLAGS & FLAG_CUFFTCB_OUT )
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU311_p<FLAG_CUFFTCB_OUT | FLAG_STP_ROW> (dimGrid, dimBlock, stream, batch);
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU311_p<FLAG_CUFFTCB_OUT | FLAG_STP_PLN>  (dimGrid, dimBlock, stream, batch);
        else
        {
          fprintf(stderr, "ERROR: %s has not been templated for flag combination. \n", __FUNCTION__ );
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU311_p<FLAG_STP_ROW> (dimGrid, dimBlock, stream, batch);
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU311_p<FLAG_STP_PLN> (dimGrid, dimBlock, stream, batch);
        else
        {
          fprintf(stderr, "ERROR: %s has not been templated for flag combination. \n", __FUNCTION__ );
          exit(EXIT_FAILURE);
        }
      }
    }
  }
