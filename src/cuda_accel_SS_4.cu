#include "cuda_accel_SS.h"

/** Sum and Search - loop down - column max - use blocks .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base
 */
template<int noStages, int canMethoud>
__global__ void add_and_searchCU4(cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base)
{
  const int x   = threadIdx.x;
  const int y   = threadIdx.y;
  const int gx  = blockIdx.x*SS4_X + x;

  if ( gx < searchList.widths.val[0] )
  {
    int batches = searchList.heights.val[0] / (float)( SS4_Y );

    const int noHarms = (1 << (noStages - 1));
    int inds[noHarms];
    int start, end;

    float powerThread[noStages];
    int   z[noStages];

    for ( int stage = 0; stage < noStages; stage++ )
    {
      powerThread[stage]  = 0;
      z[stage]            = 0;
    }

    // Initialise the x indices of this thread
    inds[0] = gx + searchList.ffdBuffre.val[0];

    // Calculate the x indices
#pragma unroll
    for ( int i = 1; i < noHarms; i++ )
    {
      //inds[i]     = (int)(gx*searchList.frac.val[i]+searchList.idxSum.val[i]) + searchList.ffdBuffre.val[i];
    }

    for ( int b = 0;  b < batches;  b++)  // Loop over blocks
    {
      float blockPower = 0;
      int by = b*SS4_Y + y;

#pragma unroll
      for ( int stage = 0; stage < noStages; stage++ ) // Loop over harmonic stages
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

        // Sum set of powers
#pragma unroll
        for ( int harm = start; harm < end; harm++ ) // Loop over sub harmonics
        {
          if  ( (canMethoud & FLAG_PLN_TEX ) )
          {
            const float2 cmpf = tex2D < float2 > (searchList.texs.val[harm], inds[harm], YINDS[searchList.yInds.val[harm]+by]);
            blockPower += cmpf.x * cmpf.x + cmpf.y * cmpf.y;
          }
          else
          {
            const fcomplexcu cmpc  = searchList.datas.val[harm][YINDS[searchList.yInds.val[harm]+by]*searchList.strides.val[harm]+inds[harm]];
            blockPower += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
          }
        }

        if  (  blockPower >  POWERCUT[stage] )
        {
          if ( blockPower > powerThread[stage] )
          {
            powerThread[stage]  = blockPower;
            z[stage]            = b;
          }
        }
      }
    }

#pragma unroll
    for ( int stage = 0; stage < 1; stage++ ) // Loop over harmonic stages
    {
      accelcandBasic can;
      long long numtrials         = NUMINDEP[stage];
      const short numharm = 1 << stage;

      if  ( powerThread[stage] >  POWERCUT[stage] )
      {
        if ( canMethoud & CU_OUTP_SINGLE )
        {
          can.numharm = numharm;
          can.sigma   = powerThread[0];
          can.z       = z[0];
          if ( canMethoud & FLAG_SIG_GPU )
          {
            // Calculate sigma value
            can.sigma   = (float)candidate_sigma_cu(powerThread[0], numharm, numtrials);
          }

          // Write to DRAM
          d_cands[ searchList.widths.val[0]*stage*y +  stage*searchList.widths.val[0] + gx ] = can;
        }
      }
    }
  }

  /*

  __shared__ float s_powers[noStages][SS4_Y][SS4_X];
  __shared__ uint  s_z[noStages][SS4_Y][SS4_X];
  __shared__ int sum[noStages];

  if (x < noStages && y == 0)
  {
    sum[x] = 0;
  }

  // __syncthreads();

  // Write all results to shard memory
  for ( int s = 0 ; s <  noStages; s++)
  {
    if (powerThread[s] > 0 )
    {
      s_powers[s][y][x]  = powerThread[s];
      s_z[s][y][x]       = z[s] ; // *SS4_Y+y;
      atomicAdd(&sum[s], 1);
    }
  }

  __syncthreads();

  // Write back to DRAM
  if ( y < noStages && sum[y] > 0 )
  {
    z[0] = 0;
    powerThread[0] = 0;
    int stage = y;

    for ( int by = 0 ; by < SS4_Y; by++ )
    {
      if( s_powers[stage][by][x] > powerThread[0] )
      {
        powerThread[0]  = s_powers[stage][by][x];
        z[0]            = s_z[stage][by][x]*SS4_Y + by;
      }
    }

    if  ( powerThread[0] >  POWERCUT[stage] )
    {
      accelcandBasic can;
      long long numtrials         = NUMINDEP[stage];
      const short numharm = 1 << stage;

      // Write results back to DRAM and calculate sigma if needed
      if      ( canMethoud & CU_OUTP_DEVICE   )
      {
        int idx =  (int)(( searchList.rLow.val[0] + gx * (double) ACCEL_DR ) / (double)numharm ) - base ;
        if ( idx >= 0 )
        {
          can.numharm = numharm;
          can.sigma   = (float)candidate_sigma_cu(powerThread[0], numharm, numtrials);
          can.z       = ( z[0]*(float) ACCEL_DZ - searchList.zMax.val[0]  )  / (float)numharm ;

          FOLD // Atomic write to global list
          {
            volatile bool done = false;
            while (!done)
            {
              volatile int prev = atomicCAS(&d_sem[idx], UINT_MAX, gx );
              if ( prev == UINT_MAX )
              {
                if ( can.sigma > d_cands[idx].sigma )
                {
                  d_cands[idx]   = can;
                }
                d_sem[idx]      = UINT_MAX;
                done            = true;
              }
            }
          }
        }
      }
      else if ( canMethoud & CU_OUTP_SINGLE )
      {
        can.numharm = numharm;
        can.sigma   = powerThread[0];
        can.z       = z[0];
        if ( canMethoud & FLAG_SIG_GPU )
        {
          // Calculate sigma value
          can.sigma   = (float)candidate_sigma_cu(powerThread[0], numharm, numtrials);
        }

        // Write to DRAM
        d_cands[gx*noStages + stage] = can;
      }
    }
  }
   */
}
