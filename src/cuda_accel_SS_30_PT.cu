#include "cuda_accel_SS.h"

#define SS3T_X           16                    // X Thread Block
#define SS3T_Y           8                     // Y Thread Block
#define SS3TBS           (SS3T_X*SS3T_Y)

#define CHUNKSZ         6

template<uint FLAGS, int noStages, const int noHarms, int noSteps, typename stpType>
__global__ void add_and_searchCU3_PT(const uint width, accelcandBasic* d_cands, stpType rBin, tHarmList texs)
{
  const int bidx  = threadIdx.y * SS3T_X  +  threadIdx.x;   /// Block index
  const int tid   = blockIdx.x  * SS3TBS  +  bidx;          /// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    const int zeroHeight  = HEIGHT_STAGE[0];
    const int oStride     = STRIDE_STAGE[0];                          /// The stride of the output data

    float           inds      [noSteps][noHarms];
    accelcandBasic  candLists [noStages][noSteps];
    float           powers    [noSteps][CHUNKSZ];

    FOLD // Prep - Initialise the x indices & set candidates to 0 .
    {
      // Calculate the x indices or create a pointer offset by the correct amount
//#pragma unroll
      for ( int harm = 0; harm < noHarms; harm++ )              // loop over harmonic  .
      {
//#pragma unroll
        for ( int step = 0; step < noSteps; step++)             // Loop over steps
        {
          float fx    = (rBin.val[step] + tid)*FRAC_STAGE[harm] - rBin.val[harm*noSteps + step]  + HWIDTH_STAGE[harm];

          if        ( FLAGS & FLAG_ITLV_ROW )
          {
            fx += step*STRIDE_STAGE[harm] ;
          }
          inds[step][harm]      = fx+0.5f; // Add 0.5 to make centre of pixel
        }
      }

      FOLD  // Set the local and return candidate powers to zero
      {
//#pragma unroll
        for ( int stage = 0; stage < noStages; stage++ )
        {
//#pragma unroll
          for ( int step = 0; step < noSteps; step++)           // Loop over steps
          {
            // Set the local  candidate
            candLists[stage][step].sigma = 0 ;

            // Set the return candidate
            d_cands[step*noStages*oStride + stage*oStride + tid ].sigma = 0;
          }
        }
      }
    }

    FOLD // Sum & Search (ignore contaminated ends tid to starts at correct spot  .
    {
      for( int y = 0; y < zeroHeight ; y += CHUNKSZ )               // loop over chunks .
      {

        // Initialise powers for each section column to 0
//#pragma unroll
        for ( int step = 0; step < noSteps; step++)             // Loop over steps .
        {
//#pragma unroll
          for( int i = 0; i < CHUNKSZ ; i++ )                   // Loop over powers .
          {
            powers[step][i] = 0;
          }
        }

        FOLD // Loop over stages, sum and search  .
        {
          //#pragma unroll
          for ( int stage = 0 ; stage < noStages; stage++)        // Loop over stages .
          {
            int start = STAGE[stage][0];
            int end   = STAGE[stage][1];

            // Create a section of summed powers one for each step
            //#pragma unroll
            for ( int harm = start; harm <= end; harm++ )         // Loop over harmonics (batch) in this stage  .
            {
              //#pragma unroll
              for( int yPlus = 0; yPlus < CHUNKSZ; yPlus++ )      // Loop over the chunk  .
              {
                int trm     = y + yPlus ;                         ///< True Y index in plain

                float fy1   = (HEIGHT_STAGE[harm]-1.0)*trm/(float)(zeroHeight-1.0) + 0.5f;
                float fy2   = fy1;

                //#pragma unroll
                for ( int step = 0; step < noSteps; step++)        // Loop over steps  .
                {
                  if        ( FLAGS & FLAG_ITLV_PLN )
                  {
                    fy2 = fy1 + step * HEIGHT_STAGE[harm];  // stride step by plain
                  }

                  float fx              = inds[step][harm] ;

                  const float cmpf      = tex2D < float > (texs.val[harm], fx, fy2 ); // + 0.5 YES + 0.5 I REALLY wish someone had documented that one, 2 days of debugging to find that!!!!!!

                  powers[step][yPlus]   += cmpf;
                }
              }
            }

            // Search set of powers
            //#pragma unroll
            FOLD
            {
              for ( int step = 0; step < noSteps; step++)           // Loop over steps
              {
                for( int yPlus = 0; yPlus < CHUNKSZ ; yPlus++ )     // Loop over section
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
          if  ( candLists[stage][step].sigma >  POWERCUT_STAGE[stage] )
          {
            // Write to DRAM
            d_cands[step*noStages*oStride + stage*oStride + tid] = candLists[stage][step];
          }
        }
      }
    }
  }
}

template<uint FLAGS, int noStages, const int noHarms,  int noSteps>
__host__ void add_and_searchCU3_PT_d(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  tHarmList   texs;
  for (int i = 0; i < noHarms; i++)
  {
    int idx         = batch->stageIdx[i];
    texs.val[i]     = batch->plains[idx].datTex;
  }

  if      (noHarms*noSteps <= 8 )
  {
    long08 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->stageIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    add_and_searchCU3_PT<FLAGS,noStages,noHarms,noSteps,long08><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, rBin, texs );
  }
  else if (noHarms*noSteps <= 16 )
  {
    long16 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->stageIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    add_and_searchCU3_PT<FLAGS,noStages,noHarms,noSteps,long16><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, rBin, texs );
  }
  else if (noHarms*noSteps <= 32 )
  {
    long32 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->stageIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    add_and_searchCU3_PT<FLAGS,noStages,noHarms,noSteps,long32><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, rBin, texs );
  }
  else if (noHarms*noSteps <= 64 )
  {
    long64 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->stageIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    add_and_searchCU3_PT<FLAGS,noStages,noHarms,noSteps,long64><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, rBin, texs );
  }
  else if (noHarms*noSteps <= 128 )
  {
    long128 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->stageIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    add_and_searchCU3_PT<FLAGS,noStages,noHarms,noSteps,long128><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, rBin, texs );
  }
  else
  {
    fprintf(stderr,"ERROR: %s has not been set up to work with %i elements.",__FUNCTION__, noHarms*noSteps);
  }
}

template<uint FLAGS, int noStages, const int noHarms>
__host__ void add_and_searchCU3_PT_s(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int noSteps = batch->noSteps ;

  switch (noSteps)
  {
    case 1:
    {
      add_and_searchCU3_PT_d<FLAGS,noStages,noHarms,1>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 2:
    {
      add_and_searchCU3_PT_d<FLAGS,noStages,noHarms,2>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 3:
    {
      add_and_searchCU3_PT_d<FLAGS,noStages,noHarms,3>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 4:
    {
      add_and_searchCU3_PT_d<FLAGS,noStages,noHarms,4>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 5:
    {
      add_and_searchCU3_PT_d<FLAGS,noStages,noHarms,5>(dimGrid, dimBlock, stream, batch);
      break;
    }
    default:
      fprintf(stderr, "ERROR: add_and_searchCU311 has not been templated for %i steps\n", noSteps);
      exit(EXIT_FAILURE);
  }
}

template<uint FLAGS >
__host__ void add_and_searchCU3_PT_p(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int noStages = batch->noHarmStages;

  switch (noStages)
  {
    case 1:
    {
      add_and_searchCU3_PT_s<FLAGS,1,1>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 2:
    {
      add_and_searchCU3_PT_s<FLAGS,2,2>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 3:
    {
      add_and_searchCU3_PT_s<FLAGS,3,4>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 4:
    {
      add_and_searchCU3_PT_s<FLAGS,4,8>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 5:
    {
      add_and_searchCU3_PT_s<FLAGS,5,16>(dimGrid, dimBlock, stream, batch);
      break;
    }
    default:
      fprintf(stderr, "ERROR: %s has not been templated for %i stages\n", __FUNCTION__, noStages);
      exit(EXIT_FAILURE);
  }
}

__host__ void add_and_searchCU3_PT_f(cudaStream_t stream, cuFFdotBatch* batch )
{
  const uint FLAGS = batch->flag;
  dim3 dimBlock, dimGrid;

  dimBlock.x  = SS3T_X;
  dimBlock.y  = SS3T_Y;

  float bw    = SS3TBS ;
  float ww    = batch->accelLen / ( bw );

  dimGrid.x   = ceil(ww);
  dimGrid.y   = 1;

  if        ( (FLAGS & FLAG_CUFFT_CB_OUT) && (FLAGS & FLAG_SAS_TEX) && (FLAGS & FLAG_TEX_INTERP) )
  {
    if      ( FLAGS & FLAG_ITLV_ROW )
      add_and_searchCU3_PT_p<FLAG_ITLV_ROW> (dimGrid, dimBlock, stream, batch);
    else if ( FLAGS & FLAG_ITLV_PLN )
      add_and_searchCU3_PT_p<FLAG_ITLV_PLN> (dimGrid, dimBlock, stream, batch);
    else
    {
      fprintf(stderr, "ERROR: %s has not been templated for flag combination. \n", __FUNCTION__ );
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    fprintf(stderr, "ERROR: %s has not been templated for flag combination. \n", __FUNCTION__ );
    exit(EXIT_FAILURE);
  }
}


