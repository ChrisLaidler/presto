#include "cuda_accel_SS.h"


/** Sum and Search - loop down - column max - multi-step .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSteps
 */
#if TEMPLATE_SEARCH == 1
template<uint FLAGS, /*typename sType,*/ int noStages, typename stpType, int noSteps>
__global__ void add_and_searchCU31(cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base/*, sType pd*/, stpType rLows )
#else
template<uint FLAGS, int noStages, typename stpType>
__global__ void add_and_searchCU31(cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base/*, sType pd*/, stpType rLows, int noSteps )
#endif
{
  const int bidx  = threadIdx.y * SS3_X         +  threadIdx.x;   /// Block index
  const int tid   = blockIdx.x  * (SS3_Y*SS3_X) +  bidx;          /// Global thread id (ie column)
  const int width = searchList.widths.val[0];                     /// The width of usable data



  if ( tid < width )
  {
    const int noHarms     = ( 1 << (noStages-1) ) ;
    const int zeroHeight  = searchList.heights.val[0];
    const int oStride     = searchList.strides.val[0];          /// The stride of the output data
    int iy, ix;                                                 /// Global indices scaled to sub-batch
    int y;

#if TEMPLATE_SEARCH == 1
    accelcandBasic candLists[noStages][noSteps];
    int         inds[noSteps][noHarms];
    fcomplexcu* pData[noSteps][noHarms];
    float       powers[noSteps][CHUNKSZ];                       // registers to hold values to increase mem cache hits
#else
    accelcandBasic candLists[noStages][MAX_STEPS];
    int         inds[MAX_STEPS][noHarms];
    fcomplexcu* pData[MAX_STEPS][noHarms];
    float       powers[MAX_STEPS][CHUNKSZ];                     // registers to hold values to increase mem cache hits
#endif

    FOLD // Prep - Initialise the x indices & set candidates to 0 .
    {
      // Calculate the x indices or create a pointer offset by the correct amount
#pragma unroll
      for ( int harm = 0; harm < noHarms; harm++ )              // loop over harmonic  .
      {
#if TEMPLATE_SEARCH == 1
#pragma unroll
#endif
        for ( int step = 0; step < noSteps; step++)             // Loop over steps
        {
          int drlo =   (int) ( ACCEL_RDR * rLows.arry[step] * searchList.frac.val[harm] + 0.5 ) * ACCEL_DR ;
          float srlo = (int) ( ACCEL_RDR * ( rLows.arry[step] + tid * ACCEL_DR ) * searchList.frac.val[harm] + 0.5 ) * ACCEL_DR ;

          ix = (srlo - drlo) * ACCEL_RDR + searchList.ffdBuffre.val[harm] ;

          //float rLow    = rLows.arry[step] * searchList.frac.val[harm];
          //float diff    = rLow - (int)rLow;
          //float idxS    = 0.5f + diff*ACCEL_RDR ;
          //ix = (int)( tid * searchList.frac.val[harm] + idxS ) + searchList.ffdBuffre.val[harm];

          if     (FLAGS & FLAG_PLN_TEX)  // Calculate x index
          {
            inds[step][harm]      = ix;
          }
          else                           // Create a pointer list that is offset by the correct amount
          {
            inds[step][harm]      = ix;

            if        ( FLAGS & FLAG_STP_ROW )
            {
              if      ( FLAGS & FLAG_CUFFTCB_OUT )
              {
                //pPowr[step][harm]   = &searchList.powers.val[harm][ ix + searchList.strides.val[harm]*step ] ;
              }
              else
              {
                pData[step][harm]   = &searchList.datas.val[harm][  ix + searchList.strides.val[harm]*step ] ;
              }
            }
            else if   ( FLAGS & FLAG_STP_PLN )
            {
              if      ( FLAGS & FLAG_CUFFTCB_OUT )
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
        if     ( FLAGS & FLAG_PLN_TEX )
        {
        }
        else
        {
          if        ( FLAGS & FLAG_STP_ROW )
          {
            if ( FLAGS & FLAG_CUFFTCB_OUT )
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

      FOLD  // Set the local and return candidate powers to zero
      {
#pragma unroll
        for ( int stage = 0; stage < noStages; stage++ )
        {
#if TEMPLATE_SEARCH == 1
#pragma unroll
#endif
          for ( int step = 0; step < noSteps; step++)           // Loop over steps
          {
            candLists[stage][step].sigma = 0 ;

            if ( FLAGS & CU_OUTP_SINGLE )
            {
              d_cands[step*noStages*oStride + stage*oStride + tid ].sigma = 0;
            }
          }
        }
      }
    }

    FOLD // Sum & Search (ignore contaminated ends tid o starts at correct spot
    {
      for( y = 0; y < zeroHeight ; y += CHUNKSZ )               // loop over chunks .
      {
        int start   = 0;
        int end     = 0;

        // Initialise powers for each section column to 0
#if TEMPLATE_SEARCH == 1
#pragma unroll
#endif
        for ( int step = 0; step < noSteps; step++)             // Loop over steps .
        {
#pragma unroll
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
            for( int yPlus = 0; yPlus < CHUNKSZ; yPlus++ )      // Loop over the chunk  .
            {
              int trm   = y + yPlus ;                           /// True Y index in plain
              iy        = YINDS[ searchList.yInds.val[harm] + trm ];

#if TEMPLATE_SEARCH == 1
              #pragma unroll
#endif
              for ( int step = 0; step < noSteps; step++)        // Loop over steps  .
              {
                if     (FLAGS & FLAG_PLN_TEX)
                {
                  if ( FLAGS & FLAG_CUFFTCB_OUT )
                  {
                    // TODO: NB: use powers and texture memory to interpolate values
                  }
                  else
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

                    const float2 cmpf         = tex2D < float2 > (searchList.texs.val[harm], inds[step][harm], iy);
                    powers[step][yPlus]      += cmpf.x * cmpf.x + cmpf.y * cmpf.y;
                  }
                }
                else
                {
                  if ( FLAGS & FLAG_CUFFTCB_OUT )
                  {
                    float power;
                    if        ( FLAGS & FLAG_STP_ROW )
                    {
                      power = searchList.powers.val[harm][ (inds[step][harm]  + searchList.strides.val[harm]*noSteps*iy + searchList.strides.val[harm]*step) ] ;
                      //power = pPowr[step][harm][searchList.strides.val[harm]*iy] ; // Note stride has been set depending on multi-step type
                    }
                    else if   ( FLAGS & FLAG_STP_PLN )
                    {
                      power = searchList.powers.val[harm][ inds[step][harm]  + searchList.strides.val[harm]*iy + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
                    }
                    powers[step][yPlus]        += power;
                  }
                  else
                  {
                    fcomplexcu cmpc;
                    if        ( FLAGS & FLAG_STP_ROW )
                    {
                      //cmpc = searchList.datas.val[harm][ inds[step][harm]  + searchList.strides.val[harm]*noSteps*iy + searchList.strides.val[harm]*step ] ;
                      cmpc = pData[step][harm][searchList.strides.val[harm]*iy] ; // Note stride has been set depending on multi-step type
                    }
                    else if   ( FLAGS & FLAG_STP_PLN )
                    {
                      cmpc = searchList.datas.val[harm][ inds[step][harm]  + searchList.strides.val[harm]*iy + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
                    }

                    powers[step][yPlus]        += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                  }
                }
              }
            }
          }

          // Search set of powers
#if TEMPLATE_SEARCH == 1
          //#pragma unroll
#endif
          for ( int step = 0; step < noSteps; step++)           // Loop over steps
          {
            //#pragma unroll
            for( int yPlus = 0; yPlus < CHUNKSZ ; yPlus++ )     // Loop over section
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

    // Write results back to DRAM and calculate sigma if needed
    if      ( FLAGS & CU_OUTP_DEVICE && 0)
    {
//#pragma unroll
      for ( int stage = 0 ; stage < noStages; stage++)
      {
        const short numharm = 1 << stage;

#if TEMPLATE_SEARCH == 1
#pragma unroll
#endif
        for ( int step = 0; step < noSteps; step++)         // Loop over steps
        {
          if  ( candLists[stage][step].sigma >  POWERCUT[stage] )
          {
            //float rLow    = rLows.arry[step] * searchList.frac.val[harm];
            //float diff    = rLow - (int)rLow;
            //float idxS    = 0.5  + diff*ACCEL_RDR ;

            int idx =  (int)(( rLows.arry[step] + tid * (double) ACCEL_DR ) / (double)numharm ) - base ;
            if ( idx >= 0 )
            {
              long long numtrials             = NUMINDEP[stage];
              candLists[stage][step].numharm  = numharm;
              //candLists[stage][step].z      = ( candLists[stage].z*(double) ACCEL_DZ - searchList.zMax.val[0]  )  / (double)numharm ;
              candLists[stage][step].sigma    = (float)candidate_sigma_cu(candLists[stage][step].sigma, numharm, numtrials);

              FOLD // Atomic write to global list
              {
                volatile bool done = false;
                while (!done)
                {
                  volatile int prev = atomicCAS(&d_sem[idx], UINT_MAX, tid );
                  if ( prev == UINT_MAX )
                  {
                    if ( candLists[stage][step].sigma > d_cands[idx].sigma )
                    {
                      d_cands[idx]              = candLists[stage][step];
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
#if TEMPLATE_SEARCH == 1
#pragma unroll
#endif
      for ( int step = 0; step < noSteps; step++)             // Loop over steps
      {
#pragma unroll
        for ( int stage = 0 ; stage < noStages; stage++)      // Loop over stages
        {
          if  ( candLists[stage][step].sigma >  POWERCUT[stage] )
          {
            // This can be calculated from stage
            //const short numharm                 = ( 1 << stage );
            //candLists[stage][step].numharm      = numharm;

            if ( (FLAGS & FLAG_SAS_SIG) && FALSE)             // Calculate the actual sigma value on the GPU
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


template<uint FLAGS, /*typename sType,*/ uint noStages>
__host__ void add_and_searchCU31_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream,cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base, /*sType pd,*/ float* rLows, int noSteps)
{
#if TEMPLATE_SEARCH == 1
  switch (noSteps)
  {
    case 1:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU31<FLAGS,sType,noStages,f01,1>, cudaFuncCachePreferL1);
      f01 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_searchCU31<FLAGS,/*sType,*/noStages,f01,1><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 2:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU31<FLAGS,sType,noStages,f02,2>, cudaFuncCachePreferL1);
      f02 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_searchCU31<FLAGS,/*sType,*/noStages,f02,2><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 3:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU31<FLAGS,sType,noStages,f03,3>, cudaFuncCachePreferL1);
      f03 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_searchCU31<FLAGS,/*sType,*/noStages,f03,3><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 4:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU31<FLAGS,sType,noStages,f04,4>, cudaFuncCachePreferL1);
      f04 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_searchCU31<FLAGS,/*sType,*/noStages,f04,4><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 5:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU31<FLAGS,sType,noStages,f05,5>, cudaFuncCachePreferL1);
      f05 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_searchCU31<FLAGS,/*sType,*/noStages,f05,5><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 6:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU31<FLAGS,sType,noStages,f06,6>, cudaFuncCachePreferL1);
      f06 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_searchCU31<FLAGS,/*sType,*/noStages,f06,6><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 7:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU31<FLAGS,sType,noStages,f07,7>, cudaFuncCachePreferL1);
      f07 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_searchCU31<FLAGS,/*sType,*/noStages,f07,7><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 8:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU31<FLAGS,sType,noStages,f08,8>, cudaFuncCachePreferL1);
      f08 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_searchCU31<FLAGS,/*sType,*/noStages,f08,8><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    default:
      fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for %i steps\n", noSteps);
      exit(EXIT_FAILURE);
  }
#else
  //cudaFuncSetCacheConfig(add_and_searchCU31<FLAGS,sType,noStages,fMax>, cudaFuncCachePreferL1);
  fMax tmpArr;
  for (int i = 0; i < noSteps; i++)
    tmpArr.arry[i] = rLows[i];

  add_and_searchCU31<FLAGS,/*sType,*/noStages,fMax> <<<dimGrid, dimBlock, i1, cnvlStream>>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr, noSteps);
#endif
}

template<uint FLAGS >
__host__ void add_and_searchCU31_p(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream,cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base, float* rLows, int noSteps, const uint noStages )
{
  switch (noStages)
  {
    case 1:
    {
      add_and_searchCU31_s<FLAGS,/*sch1,*/1> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, /*tmpArr,*/ rLows, noSteps );
      break;
    }
    case 2:
    {
      add_and_searchCU31_s<FLAGS,/*sch2,*/2> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, /*tmpArr,*/ rLows, noSteps );
      break;
    }
    case 3:
    {
      add_and_searchCU31_s<FLAGS,/*sch4,*/3> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, /*tmpArr,*/ rLows, noSteps );
      break;
    }
    case 4:
    {
      add_and_searchCU31_s<FLAGS,/*sch8,*/4> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, /*tmpArr,*/ rLows, noSteps );
      break;
    }
    case 5:
    {
      add_and_searchCU31_s<FLAGS,/*sch16,*/5> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, /*tmpArr,*/ rLows, noSteps );
      break;
    }
    default:
      fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for %i stages\n", noStages);
      exit(EXIT_FAILURE);
  }
}

__host__ void add_and_searchCU31_f(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream,cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base, float* rLows, int noSteps, const uint noStages, uint FLAGS )
{
  if        ( FLAGS & FLAG_CUFFTCB_OUT )
  {
    if      ( FLAGS & FLAG_STP_ROW )
      add_and_searchCU31_p<FLAG_CUFFTCB_OUT | CU_OUTP_SINGLE | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, rLows, noSteps, noStages );
    else if ( FLAGS & FLAG_STP_PLN )
      add_and_searchCU31_p<FLAG_CUFFTCB_OUT | CU_OUTP_SINGLE | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, rLows, noSteps, noStages );
    //else if ( FLAGS & FLAG_STP_STK )
    //  add_and_searchCU31_p<FLAG_CUFFTCB_OUT | CU_OUTP_SINGLE | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, pd, rLows, noSteps, noStages );
    else
    {
      fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n" );
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    if      ( FLAGS & FLAG_STP_ROW )
      add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, rLows, noSteps, noStages );
    else if ( FLAGS & FLAG_STP_PLN )
      add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, rLows, noSteps, noStages );
    //else if ( FLAGS & FLAG_STP_STK )
    //  add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, pd, rLows, noSteps, noStages );
    else
    {
      fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n" );
      exit(EXIT_FAILURE);
    }
  }

// Uncomenting this block will make compile time VERY long! I mean days!
/*
  if( FLAGS & CU_OUTP_DEVICE )
  {
    if( FLAGS & FLAG_PLN_TEX )
    {
      if ( FLAGS & FLAG_SAS_SIG )
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination.\n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_PLN_TEX | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_PLN_TEX | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_PLN_TEX | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
    }
    else
    {
      if ( FLAGS & FLAG_SAS_SIG )
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_SAS_SIG | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_SAS_SIG | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_SAS_SIG | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
    }
  }
  else if ( (FLAGS & CU_OUTP_SINGLE) || (FLAGS & CU_OUTP_HOST) )
  {
    if( FLAGS & FLAG_PLN_TEX )
    {
      if ( FLAGS & FLAG_SAS_SIG )
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination.  \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
    }
    else
    {
      if ( FLAGS & FLAG_SAS_SIG )
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination.  \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
    }
  }
  else if (  FLAGS & CU_OUTP_SINGLE )
  {
    if( FLAGS & FLAG_PLN_TEX )
    {
      if ( FLAGS & FLAG_SAS_SIG )
      {
        if      (FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_STK )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination.  \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      (FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_STK )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
    }
    else
    {
      if ( FLAGS & FLAG_SAS_SIG )
      {
        if      (FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_STK )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination.  \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      (FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_STK )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
    }
  }
*/
}
