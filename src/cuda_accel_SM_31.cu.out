#include "cuda_accel_SS.h"

#define SM31_X           16                    // X Thread Block
#define SM31_Y           8                     // Y Thread Block
#define SM31BS           (SM31_X*SM31_Y)

#define CHUNKSZ         6

/** Sum and Search - loop down - column max - multi-step  .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base
 * @param noSteps
 */
template<uint FLAGS, int noStages, typename stpType, int noSteps>
__global__ void add_and_maxCU31(cuSearchList searchList, float* d_cands, uint* d_sem, int base, stpType rLows )
{
  const int bidx  = threadIdx.y * SM31_X  +  threadIdx.x;   /// Block index
  const int tid   = blockIdx.x  * SM31BS  +  bidx;          /// Global thread id (ie column)
  const int width = searchList.widths.val[0];                     /// The width of usable data

  if ( tid < width )
  {
    const int noHarms     = ( 1 << (noStages-1) ) ;
    const int zeroHeight  = searchList.heights.val[0];
    const int oStride     = searchList.strides.val[0];

    float       candLists[noSteps];
    int         inds[noSteps][noHarms];
    fcomplexcu* pData[noSteps][noHarms];
    float       powers[noSteps][CHUNKSZ];               // registers to hold values to increase mem cache hits

    int iy, ix;                                         ///< Global indices scaled to sub-batch
    int y;

    FOLD // Prep - Initialise the x indices & set candidates to 0  .
    {
      // Calculate the x indices or create a pointer offset by the correct amount
//#pragma unroll
      for ( int harm = 0; harm < noHarms; harm++ )      // loop over harmonic  .
      {
//#pragma unroll
        for ( int step = 0; step < noSteps; step++)     // Loop over steps  .
        {
          int drlo = (int) ( ACCEL_RDR * rLows.arry[step] * searchList.frac.val[harm] + 0.5 ) * ACCEL_DR ;
          float srlo = (int) ( ACCEL_RDR * ( rLows.arry[step] + tid * ACCEL_DR ) * searchList.frac.val[harm] + 0.5 ) * ACCEL_DR ;

          ix = (srlo - drlo) * ACCEL_RDR + searchList.ffdBuffre.val[harm] ;

          //float rLow    = rLows.arry[step] * searchList.frac.val[harm];
          //float diff    = rLow - (int)rLow;
          //float idxS    = 0.5f + diff*ACCEL_RDR ;
          //ix = (int)( tid * searchList.frac.val[harm] + idxS ) + searchList.ffdBuffre.val[harm];

          if     (FLAGS & FLAG_SAS_TEX)  // Calculate x index
          {
            inds[step][harm]      = ix;
          }
          else                           // Create a pointer list that is offset by the correct amount
          {
            inds[step][harm]      = ix;

            if        ( FLAGS & FLAG_ITLV_ROW )
            {
              if      ( FLAGS & FLAG_CUFFT_CB_OUT )
              {
                //pPowr[step][harm]   = &searchList.powers.val[harm][ ix + searchList.strides.val[harm]*step ] ;
              }
              else
              {
                pData[step][harm]   = &searchList.datas.val[harm][  ix + searchList.strides.val[harm]*step ] ;
              }
            }
            else if   ( FLAGS & FLAG_ITLV_PLN )
            {
              if      ( FLAGS & FLAG_CUFFT_CB_OUT )
              {
                //pPowr[step][harm]   = &searchList.powers.val[harm][ ix + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
              }
              else
              {
                pData[step][harm]   = &searchList.datas.val[harm][  ix + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
              }
            }
          }
        }

        // Change the stride for this harmonic
        if     ( FLAGS & FLAG_SAS_TEX )
        {
        }
        else
        {
          if        ( FLAGS & FLAG_ITLV_ROW )
          {
            if ( FLAGS & FLAG_CUFFT_CB_OUT )
            {
              //searchList.strides.val[harm] *= noSteps;
            }
            else
            {
              searchList.strides.val[harm] *= noSteps;
            }
          }
        }
      }

      FOLD // Set the stored   .
      {
//#pragma unroll
        for ( int step = 0; step < noSteps; step++)   // Loop over steps
        {
          candLists[step] = 0 ;
        }
      }
    }

    FOLD // Sum & Max  .
    {
      FOLD  // Loop over blocks of set length .
      {
        for( y = 0; y < zeroHeight ; y += CHUNKSZ )               // loop over chunks .
        {
          int start   = 0;
          int end     = 0;

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

          // Loop over stages, sum and search
//#pragma unroll
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
//#pragma unroll
            for ( int harm = start; harm < end; harm++ )          // Loop over harmonics (batch) in this stage
            {

//#pragma unroll
              for( int yPlus = 0; yPlus < CHUNKSZ; yPlus++ )      // Loop over the chunk
              {
                int trm = y + yPlus ;

                iy            = YINDS[ searchList.yInds.val[harm] + trm ];

//#pragma unroll
                for ( int step = 0; step < noSteps; step++)       // Loop over steps
                {
                  if     (FLAGS & FLAG_SAS_TEX)
                  {
                    // Calculate y indice
                    if      ( FLAGS & FLAG_ITLV_ROW )
                    {
                      iy  = ( iy * noSteps + step );
                    }
                    else if ( FLAGS & FLAG_ITLV_PLN )
                    {
                      iy  = ( iy + searchList.heights.val[harm]*step ) ;
                    }

                    const float2 cmpf         = tex2D < float2 > (searchList.texs.val[harm], inds[step][harm], iy);
                    powers[step][yPlus]      += cmpf.x * cmpf.x + cmpf.y * cmpf.y;
                  }
                  else
                  {
                    if ( FLAGS & FLAG_CUFFT_CB_OUT )
                    {
                      float power;
                      if        ( FLAGS & FLAG_ITLV_ROW )
                      {
                        power = searchList.powers.val[harm][ (inds[step][harm]  + searchList.strides.val[harm]*noSteps*iy + searchList.strides.val[harm]*step) ] ;
                        //power = pPowr[step][harm][searchList.strides.val[harm]*iy] ; // Note stride has been set depending on multi-step type
                      }
                      else if   ( FLAGS & FLAG_ITLV_PLN )
                      {
                        power = searchList.powers.val[harm][ inds[step][harm]  + searchList.strides.val[harm]*iy + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
                      }

                      //if ( (y+yPlus) == (searchList.heights.val[0]-1)/2 )
                      if ( (y+yPlus) == 0 )
                      {
                        powers[step][yPlus]        += power;
                      }
                      //powers[step][yPlus]        += power;
                    }
                    else
                    {
                      fcomplexcu cmpc;
                      if        ( FLAGS & FLAG_ITLV_ROW )
                      {
                        //cmpc = searchList.datas.val[harm][ inds[step][harm]  + searchList.strides.val[harm]*noSteps*iy + searchList.strides.val[harm]*step ] ;
                        cmpc = pData[step][harm][searchList.strides.val[harm]*iy] ; // Note stride has been set depending on multi-step type
                      }
                      else if   ( FLAGS & FLAG_ITLV_PLN )
                      {
                        cmpc = searchList.datas.val[harm][ inds[step][harm]  + searchList.strides.val[harm]*iy + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
                      }

                      if ( (y+yPlus) == (zeroHeight-1)/2.0 )
                      //if ( (y+yPlus) == 0 )
                      {
                        powers[step][yPlus]        += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                      }
                    }
                  }
                }
              }
            }
          }

          // Get max
          for ( int step = 0; step < noSteps; step++)             // Loop over steps
          {
            //#pragma unroll
            for( int i = 0; i < CHUNKSZ ; i++ )                   // Loop over section
            {
              if ( powers[step][i] > candLists[step] )
              {
                if ( y + i < zeroHeight )
                {
                  // This is our new max!
                  candLists[step]  = powers[step][i];
                }
              }
            }
          }
        }
      }
    }

    FOLD // Write results  .
    {
//#pragma unroll
      for ( int step = 0; step < noSteps; step++)             // Loop over steps
      {
        // Write to DRAM
        d_cands[step*oStride + tid] = candLists[step];
      }
    }
  }
}

template<uint FLAGS, uint noStages>
__host__ void add_and_maxCU31_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream,cuSearchList searchList, float* d_cands, uint* d_sem, int base,  float* rLows, int noSteps)
{
  switch (noSteps)
  {
    case 1:
    {
      //cudaFuncSetCacheConfig(add_and_maxCU31<FLAGS,sType,noStages,f01,1>, cudaFuncCachePreferL1);
      f01 caseArr;
      for (int i = 0; i < noSteps; i++)
        caseArr.arry[i] = rLows[i];
      add_and_maxCU31<FLAGS, noStages,f01,1><<<dimGrid,  dimBlock, i1, multStream >>>(searchList, d_cands, d_sem, base, caseArr);
      break;
    }
    case 2:
    {
      //cudaFuncSetCacheConfig(add_and_maxCU31<FLAGS,sType,noStages,f02,2>, cudaFuncCachePreferL1);
      f02 caseArr;
      for (int i = 0; i < noSteps; i++)
        caseArr.arry[i] = rLows[i];
      add_and_maxCU31<FLAGS, noStages,f02,2><<<dimGrid,  dimBlock, i1, multStream >>>(searchList, d_cands, d_sem, base, caseArr);
      break;
    }
    case 3:
    {
      //cudaFuncSetCacheConfig(add_and_maxCU31<FLAGS,sType,noStages,f03,3>, cudaFuncCachePreferL1);
      f03 caseArr;
      for (int i = 0; i < noSteps; i++)
        caseArr.arry[i] = rLows[i];
      add_and_maxCU31<FLAGS, noStages,f03,3><<<dimGrid,  dimBlock, i1, multStream >>>(searchList, d_cands, d_sem, base, caseArr);
      break;
    }
    case 4:
    {
      //cudaFuncSetCacheConfig(add_and_maxCU31<FLAGS,sType,noStages,f04,4>, cudaFuncCachePreferL1);
      f04 caseArr;
      for (int i = 0; i < noSteps; i++)
        caseArr.arry[i] = rLows[i];
      add_and_maxCU31<FLAGS, noStages,f04,4><<<dimGrid,  dimBlock, i1, multStream >>>(searchList, d_cands, d_sem, base, caseArr);
      break;
    }
    case 5:
    {
      //cudaFuncSetCacheConfig(add_and_maxCU31<FLAGS,sType,noStages,f05,5>, cudaFuncCachePreferL1);
      f05 caseArr;
      for (int i = 0; i < noSteps; i++)
        caseArr.arry[i] = rLows[i];
      add_and_maxCU31<FLAGS, noStages,f05,5><<<dimGrid,  dimBlock, i1, multStream >>>(searchList, d_cands, d_sem, base, caseArr);
      break;
    }
    case 6:
    {
      //cudaFuncSetCacheConfig(add_and_maxCU31<FLAGS,sType,noStages,f06,6>, cudaFuncCachePreferL1);
      f06 caseArr;
      for (int i = 0; i < noSteps; i++)
        caseArr.arry[i] = rLows[i];
      add_and_maxCU31<FLAGS, noStages,f06,6><<<dimGrid,  dimBlock, i1, multStream >>>(searchList, d_cands, d_sem, base, caseArr);
      break;
    }
    case 7:
    {
      //cudaFuncSetCacheConfig(add_and_maxCU31<FLAGS,sType,noStages,f07,7>, cudaFuncCachePreferL1);
      f07 caseArr;
      for (int i = 0; i < noSteps; i++)
        caseArr.arry[i] = rLows[i];
      add_and_maxCU31<FLAGS, noStages,f07,7><<<dimGrid,  dimBlock, i1, multStream >>>(searchList, d_cands, d_sem, base, caseArr);
      break;
    }
    case 8:
    {
      //cudaFuncSetCacheConfig(add_and_maxCU31<FLAGS,sType,noStages,f08,8>, cudaFuncCachePreferL1);
      f08 caseArr;
      for (int i = 0; i < noSteps; i++)
        caseArr.arry[i] = rLows[i];
      add_and_maxCU31<FLAGS, noStages,f08,8><<<dimGrid,  dimBlock, i1, multStream >>>(searchList, d_cands, d_sem, base, caseArr);
      break;
    }
    default:
      fprintf(stderr, "ERROR: add_and_maxCU31 has not been templated for %i steps\n", noSteps);
      exit(EXIT_FAILURE);
  }
}

template<uint FLAGS >
__host__ void add_and_maxCU31_p(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream,cuSearchList searchList, float* d_cands, uint* d_sem, int base, float* rLows, int noSteps, const uint noStages )
{
  switch (noStages)
  {
    case 1:
    {
      add_and_maxCU31_s<FLAGS, 1> (dimGrid, dimBlock, i1, multStream, searchList, d_cands, d_sem, base, rLows, noSteps );
      break;
    }
    case 2:
    {
      add_and_maxCU31_s<FLAGS, 2> (dimGrid, dimBlock, i1, multStream, searchList, d_cands, d_sem, base, rLows, noSteps );
      break;
    }
    case 3:
    {
      add_and_maxCU31_s<FLAGS, 3> (dimGrid, dimBlock, i1, multStream, searchList, d_cands, d_sem, base, rLows, noSteps );
      break;
    }
    case 4:
    {
      add_and_maxCU31_s<FLAGS, 4> (dimGrid, dimBlock, i1, multStream, searchList, d_cands, d_sem, base, rLows, noSteps );
      break;
    }
    case 5:
    {
      add_and_maxCU31_s<FLAGS, 5> (dimGrid, dimBlock, i1, multStream, searchList, d_cands, d_sem, base, rLows, noSteps );
      break;
    }
    default:
      fprintf(stderr, "ERROR: add_and_maxCU31 has not been templated for %i stages\n", noStages);
      exit(EXIT_FAILURE);
  }
}

__host__ void add_and_maxCU31_f(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream,cuSearchList searchList, float* d_cands, uint* d_sem, int base, float* rLows, int noSteps, const uint noStages, uint FLAGS )
{
  if        ( FLAGS & FLAG_CUFFT_CB_OUT )
  {
    if      ( FLAGS & FLAG_ITLV_ROW )
      add_and_maxCU31_p<FLAG_CUFFT_CB_OUT | FLAG_ITLV_ROW> (dimGrid, dimBlock, i1, multStream, searchList, d_cands, d_sem, base, rLows, noSteps, noStages );
    else if ( FLAGS & FLAG_ITLV_PLN )
      add_and_maxCU31_p<FLAG_CUFFT_CB_OUT | FLAG_ITLV_PLN> (dimGrid, dimBlock, i1, multStream, searchList, d_cands, d_sem, base, rLows, noSteps, noStages );
    else
    {
      fprintf(stderr, "ERROR: add_and_maxCU31 has not been templated for flag combination. \n" );
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    if      ( FLAGS & FLAG_ITLV_ROW )
      add_and_maxCU31_p< FLAG_ITLV_ROW> (dimGrid, dimBlock, i1, multStream, searchList, d_cands, d_sem, base, rLows, noSteps, noStages );
    else if ( FLAGS & FLAG_ITLV_PLN )
      add_and_maxCU31_p< FLAG_ITLV_PLN> (dimGrid, dimBlock, i1, multStream, searchList, d_cands, d_sem, base, rLows, noSteps, noStages );
    else
    {
      fprintf(stderr, "ERROR: add_and_maxCU31 has not been templated for flag combination. \n" );
      exit(EXIT_FAILURE);
    }
  }
}
