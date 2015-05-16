#include "cuda_accel_SS.h"

/** Sum and Search - loop down - column max - multi-step - shared memory .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSteps
 */
#if TEMPLATE_SEARCH == 1
template<uint FLAGS, typename sType, int noStages, typename stpType, int noSteps>
__global__ void add_and_searchCU32(cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base, sType pd, stpType rLows )
#else
template<uint FLAGS, typename sType, int noStages, typename stpType>
__global__ void add_and_searchCU32(cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base, sType pd, stpType rLows, int noSteps )
#endif
{
  /*
  const int bid   = threadIdx.y * SS3_X         +  threadIdx.x;
  const int tid   = blockIdx.x  * (SS3_Y*SS3_X) +  bid;
  const int width = searchList.widths.val[0];

  if ( tid < width )
  {
    const int noHarms   = ( 1 << (noStages-1) ) ;
    const int hlfHarms  = noHarms / 2.0 ;
    const int CHUNKSZ   = hlfHarms ;

    accelcandBasic candLists[noStages];

    // One of the two variables below should get optimised out depending on FLAG_STP_ROW or FLAG_STP_PLN
    int inds[noHarms];
    //fcomplexcu* pData[noHarms];
    float powers[CHUNKSZ];         // registers to hold values to increase mem cache hits

    __shared__ float smPowers[hlfHarms][hlfHarms][SS3_Y*SS3_X];  //

    int start   = 0;
    int end     = 0;
    int iy;
    int ix;
    int y;
    const int zeroHeight = searchList.heights.val[0] ;

    for ( int step = 0; step < noSteps; step++)     // Loop over steps
    {
      FOLD // Prep - Initialise the x indices & set candidates to 0 .
      {
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
          }

        }

        // Set the local and return candidate powers to zero
        FOLD
        {
          //#if TEMPLATE_SEARCH == 1
          //#pragma unroll
          //#endif
          //for ( int step = 0; step < noSteps; step++)   // Loop over steps
          {
#pragma unroll
            for ( int stage = 0; stage < noStages; stage++ )
            {
              candLists[stage].sigma = 0;

              if ( FLAGS & CU_OUTP_SINGLE )
              {
                d_cands[step*noStages*width + stage*width + tid ].sigma = 0;
              }
            }
          }
        }
      }

      FOLD // Sum & Search
      {
        FOLD  // Loop over blocks of set length .
        {
          for( y = 0; y < searchList.heights.val[0] ; y += nPowers )  // loop over chunks .
          {
            // Loop over stages, sum and search
#pragma unroll
            for ( int stage = 0 ; stage < noStages; stage++)          // Loop over stages .
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

              FOLD // Read summed powers into shared memory
              {
#pragma unroll
                for ( int harm = start; harm < end; harm++ )            // Loop over harmonics (batch) in this stage
                {
                  int hi = harm - start;

                  int startY, endY;

                  startY        = YINDS[ searchList.yInds.val[harm] + y ];
                  endY          = YINDS[ searchList.yInds.val[harm] + y + CHUNKSZ - 1 ];
                  int yDist     = endY -  startY ;

                  //for (int yy = startY ; yy <= endY; yy++ )
                  for (int yd = 0 ; yd < yDist; yd++ )
                  {
                    if     (FLAGS & FLAG_PLN_TEX)
                    {
                      // Calculate y indice
                      if      ( FLAGS & FLAG_STP_ROW )
                      {
                        iy  = ( yy * noSteps + step );
                      }
                      else if ( FLAGS & FLAG_STP_PLN )
                      {
                        iy  = ( yy + searchList.heights.val[harm]*step ) ;
                      }

                      const float2 cmpf       = tex2D < float2 > (searchList.texs.val[harm], inds[harm], iy);
                      powers[yy-startY]     += cmpf.x * cmpf.x + cmpf.y * cmpf.y;
                    }
                    else
                    {
                      fcomplexcu cmpc;
                      if        ( FLAGS & FLAG_STP_ROW )
                      {
                        cmpc = searchList.datas.val[harm][ inds[harm]  + searchList.strides.val[harm]*((yd+startY)*noSteps + step) ] ;
                        //cmpc = pData[harm][searchList.strides.val[harm]*noSteps*yy] ; // Note stride has been set depending on multi-step type
                      }
                      else if   ( FLAGS & FLAG_STP_PLN )
                      {
                        //cmpc = searchList.datas.val[harm][ inds[harm]  + searchList.strides.val[harm]*yy + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
                      }

                      //if      ( stage == 0 )  // Fundamental Harmonic
                      {
                        powers[yd]               = cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                      }
                      //else                    // Other Harmonics
                      {
                        //smPowers[hi][yd][bid]    = cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                      }
                    }
                  }
                }
              }

              if ( stage != 0 ) // Create summed powers for this stage
              {
                for ( int harm = start; harm < end; harm++ )          // Loop over harmonics (batch) in this stage
                {
                  int startY        = YINDS[ searchList.yInds.val[harm] + y ];

                  for( int yPlus = 0; yPlus < CHUNKSZ; yPlus++ )      // Loop over the chunk
                  {
                    int trm = y + yPlus ;

                    if ( trm < zeroHeight )
                    {
                      iy            = YINDS[ searchList.yInds.val[harm] + trm ];

                      int sy = iy - startY;

                      if ( sy >= 0 && sy < hlfHarms && harm-start < hlfHarms  && bid < SS3_Y*SS3_X )
                      {
                        //printf("yPlus %i harm: %i   sy: %i   bid: %i  \n",yPlus, harm-start, sy, bid );

                        //powers[yPlus] += smPowers[harm-start][sy][bid];
                      }
                      else
                      {
                        //printf("Error %i\n",tid);
                        //printf("Error: yPlus %i harm: %i   sy: %i   bid: %i  \n",yPlus, harm-start, sy, bid );
                      }
                    }
                    else
                    {
                      //printf("Error\n");
                    }
                  }
                }
              }

              // Search set of powers
              for( int i = 0; i < CHUNKSZ ; i++ )                     // Loop over section
              {
                if  (  powers[i] > POWERCUT[stage] )
                {
                  if ( powers[i] > candLists[stage].sigma )
                  {
                    if ( y + i < zeroHeight )
                    {
                      // This is our new max!
                      candLists[stage].sigma  = powers[i];
                      candLists[stage].z      = y+i;
                    }
                  }
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
