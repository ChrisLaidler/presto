#include "cuda_accel_SS.h"

#define SSIM_X           16                    // X Thread Block
#define SSIM_Y           16                    // Y Thread Block
#define SSIMBS           (SSIM_X*SSIM_Y)
#define MAX_BLKS         256





template<typename T, const int noStages, const int noHarms, const int cunkSize>
__global__ void searchINMEM_k(T* __restrict__ read, int iStride, int cStride, int firstBin, int start, int end, candPZs* d_cands)
{
  const int bidx        = threadIdx.y * SSIM_X  +  threadIdx.x;     /// Block index
  const int tid         = blockIdx.x  * SSIMBS  +  bidx;            /// Global thread id (ie column) 0 is the first 'good' column
  const int zeroHeight  = HEIGHT_STAGE[0];

  int             inds      [noHarms];
  //T*              ads[noHarms];
  candPZs         candLists [noStages];
  float           powers    [cunkSize];                             /// registers to hold values to increase mem cache hits

  int            idx   = start + tid ;
  int            len   = end - start;

  if ( tid < len )
  {
    //    int tmpRe[noHarms];
    //    for ( int harm = 0; harm < noHarms; harm++ )         // Loop over harmonics (batch) in this stage  .
    //    {
    //      tmpRe[harm]=0;
    //    }


    FOLD  // Set the local and return candidate powers to zero  .
    {
      for ( int stage = 0; stage < noStages; stage++ )
      {
        candLists[stage].value = 0 ;
        //d_cands[blockIdx.y*noStages*cStride + stage*cStride + tid ].value = 0;
        d_cands[stage*gridDim.y*cStride + blockIdx.y*cStride + tid].value = 0;

      }
    }

    FOLD // Prep - Initialise the x indices  .
    {
      FOLD 	// Calculate the x indices or create a pointer offset by the correct amount  .
      {
        for ( int harm = 0; harm < noHarms; harm++ )         // Loop over harmonics (batch) in this stage  .
        {
          //int   ix        = roundf( idx*FRAC_STAGE[harm] ) - firstBin;
          int   ix        = floorf( idx*FRAC_STAGE[harm] ) - firstBin;
          inds[harm]      = ix;
        }
      }
    }

    FOLD // Sum & Search - Ignore contaminated ends tid to starts at correct spot  .
    {
      int   lDepth        = ceilf(zeroHeight/(float)gridDim.y);
      int   y0            = lDepth*blockIdx.y;
      int   y1            = MIN(y0+lDepth, zeroHeight);
      long  yIndsChnksz   = zeroHeight+INDS_BUFF;



      //      int   iyP[noHarms];
      //      float   pow[noHarms];
      //      for ( int harm = 0 ; harm < noHarms; harm++)
      //      {
      //        iyP[harm] = -1;
      //      }

      for( int y = y0; y < y1 ; y += cunkSize )              // loop over chunks  .
      {
        FOLD // Initialise chunk of powers to zero .
        {
          for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )          // Loop over powers  .
            powers[yPlus] = 0;
        }

        FOLD // Loop over other stages, sum and search  .
        {
          for ( int stage = 0 ; stage < noStages; stage++)          // Loop over stages  .
          {
            int start = STAGE[stage][0] ;
            int end   = STAGE[stage][1] ;

            if ( inds[end] ) // Valid stage
            {
              FOLD // Create a section of summed powers one for each step  .
              {
                for ( int harm = start; harm <= end; harm++ )         // Loop over harmonics (batch) in this stage  .
                {
                  int     ix1   = inds[harm] ;
                  //T* sub = read + inds[harm] ;

                  long iyP       = -1;
                  float pow     = 0 ;

                  //                  int     sy0 = YINDS[ (zeroHeight+INDS_BUFF)*harm + y ];
                  //                  int     sy1 = YINDS[ (zeroHeight+INDS_BUFF)*harm + y + cunkSize - 1 ];
                  //                  float   vals[cunkSize];
                  //
                  //                  for( int yPlus = sy0; yPlus <= sy1; yPlus++ )     // Loop over the chunk  .
                  //                  {
                  //                    vals[yPlus-sy0] = get(read, yPlus*iStride + ix1 );
                  //                  }

                  for( int yPlus = 0; yPlus < cunkSize; yPlus++ )     // Loop over the chunk  .
                  {
                    long trm     = y + yPlus ;                         ///< True Y index in plane


                    //if( trm < zeroHeight )
                    {
                      long iy1     = YINDS[ yIndsChnksz*harm + trm ];

                      //if ( iyP[harm] != iy1 ) // Only read power if it is not the same as the previous  .
                      if ( iyP != iy1 ) // Only read power if it is not the same as the previous  .
                      {
                        //pow[harm] = get(read, iy1*iStride + ix1 );
                        //iyP[harm] = iy1;
                        pow = get(read, iy1*iStride + ix1 );
                        iyP = iy1;
                      }
                      FOLD // // Accumulate powers  .
                      {
                        powers[yPlus] += pow;
                      }
                    }
                  }
                }
              }

              FOLD // Search set of powers  .
              {
                float pow;
                float maxP = POWERCUT_STAGE[stage];
                int maxI;

                for( int yPlus = 0; yPlus < cunkSize ; yPlus++ )    // Loop over section  .
                {
                  pow = powers[yPlus];

                  if  ( pow > maxP )
                  {
                    int idx = y + yPlus;

                    if ( idx < y1 )
                    {
                      maxP = pow;
                      maxI = idx;
                    }
                  }
                }

                if  (  maxP > POWERCUT_STAGE[stage] )
                {
                  if ( maxP > candLists[stage].value )
                  {
                    // This is our new max!
                    candLists[stage].value  = maxP;
                    candLists[stage].z      = maxI;
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
      for ( int stage = 0 ; stage < noStages; stage++)      // Loop over stages  .
      {
        if  ( candLists[stage].value > POWERCUT_STAGE[stage] )
        {
          // Write to DRAM
          //d_cands[blockIdx.y*noStages*cStride + stage*cStride + tid ] = candLists[stage];
          d_cands[stage*gridDim.y*cStride + blockIdx.y*cStride + tid] = candLists[stage];
        }
      }
    }

  }

  //  if( tid == 0 )
  //  {
  //    printf("\n");
  //  }
}

template<typename T, const int noStages, const int noHarms>
__host__ void searchINMEM_c(cuFFdotBatch* batch )
{
  dim3 dimBlock, dimGrid;

  rVals* rVal = &batch->rValues[0][0];

  FOLD // TMP  .
  {
    double lastBin_d  = batch->sInf->sSpec->fftInf.rlo*ACCEL_RDR + batch->SrchSz->noSteps * batch->accelLen ;
    double maxUint    = std::numeric_limits<uint>::max();
    if ( maxUint <= lastBin_d )
    {
      fprintf(stderr, "ERROR: There is not enough precision in uint in %s in %s.\n", __FUNCTION__, __FILE__ );
      exit(EXIT_FAILURE);
    }
  }

  uint firstBin = batch->SrchSz->searchRLow * ACCEL_RDR ;
  uint start    = rVal->drlo * ACCEL_RDR ;
  uint end      = start + rVal->numrs;
  uint noBins   = end - start;

  dimBlock.x    = SSIM_X;
  dimBlock.y    = SSIM_Y;

  dimGrid.y     = batch->ssSlices;
  dimGrid.x     = ceil(noBins / (float) SSIMBS );

  switch ( batch->ssChunk )
  {
    case 1 :
    {
      searchINMEM_k<T,noStages,noHarms,1><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->d_planeFull, batch->sInf->mInf->inmemStride, batch->strideRes, firstBin, start, end, (candPZs*)batch->d_retData1 );
      break;
    }
    case 2 :
    {
      searchINMEM_k<T,noStages,noHarms,2><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->d_planeFull, batch->sInf->mInf->inmemStride, batch->strideRes, firstBin, start, end, (candPZs*)batch->d_retData1 );
      break;
    }
    case 3 :
    {
      searchINMEM_k<T,noStages,noHarms,3><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->d_planeFull, batch->sInf->mInf->inmemStride, batch->strideRes, firstBin, start, end, (candPZs*)batch->d_retData1 );
      break;
    }
    case 4 :
    {
      searchINMEM_k<T,noStages,noHarms,4><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->d_planeFull, batch->sInf->mInf->inmemStride, batch->strideRes, firstBin, start, end, (candPZs*)batch->d_retData1 );
      break;
    }
    case 5 :
    {
      searchINMEM_k<T,noStages,noHarms,5><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->d_planeFull, batch->sInf->mInf->inmemStride, batch->strideRes, firstBin, start, end, (candPZs*)batch->d_retData1 );
      break;
    }
    case 6 :
    {
      searchINMEM_k<T,noStages,noHarms,6><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->d_planeFull, batch->sInf->mInf->inmemStride, batch->strideRes, firstBin, start, end, (candPZs*)batch->d_retData1 );
      break;
    }
    case 7 :
    {
      searchINMEM_k<T,noStages,noHarms,7><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->d_planeFull, batch->sInf->mInf->inmemStride, batch->strideRes, firstBin, start, end, (candPZs*)batch->d_retData1 );
      break;
    }
    case 8 :
    {
      searchINMEM_k<T,noStages,noHarms,8><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->d_planeFull, batch->sInf->mInf->inmemStride, batch->strideRes, firstBin, start, end, (candPZs*)batch->d_retData1 );
      break;
    }
    case 9 :
    {
      searchINMEM_k<T,noStages,noHarms,9><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->d_planeFull, batch->sInf->mInf->inmemStride, batch->strideRes, firstBin, start, end, (candPZs*)batch->d_retData1 );
      break;
    }
    case 10:
    {
      searchINMEM_k<T,noStages,noHarms,10><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->d_planeFull, batch->sInf->mInf->inmemStride, batch->strideRes, firstBin, start, end, (candPZs*)batch->d_retData1 );
      break;
    }
    case 12:
    {
      searchINMEM_k<T,noStages,noHarms,12><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->d_planeFull, batch->sInf->mInf->inmemStride, batch->strideRes, firstBin, start, end, (candPZs*)batch->d_retData1 );
      break;
    }
    case 14:
    {
      searchINMEM_k<T,noStages,noHarms,14><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->d_planeFull, batch->sInf->mInf->inmemStride, batch->strideRes, firstBin, start, end, (candPZs*)batch->d_retData1 );
      break;
    }
    case 20:
    {
      searchINMEM_k<T,noStages,noHarms,20><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->d_planeFull, batch->sInf->mInf->inmemStride, batch->strideRes, firstBin, start, end, (candPZs*)batch->d_retData1 );
      break;
    }
    case 25:
    {
      searchINMEM_k<T,noStages,noHarms,25><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->d_planeFull, batch->sInf->mInf->inmemStride, batch->strideRes, firstBin, start, end, (candPZs*)batch->d_retData1 );
      break;
    }
    default:
      fprintf(stderr, "ERROR: %s has not been templated for %i chunk size.\n", __FUNCTION__, batch->ssChunk);
      exit(EXIT_FAILURE);
  }
}

template<typename T >
__host__ void searchINMEM_p(cuFFdotBatch* batch )
{
  const int noStages = batch->sInf->noHarmStages;

  switch (noStages)
  {
    case 1 :
    {
      searchINMEM_c<T,1,1>(batch);
      break;
    }
    case 2 :
    {
      searchINMEM_c<T,2,2>(batch);
      break;
    }
    case 3 :
    {
      searchINMEM_c<T,3,4>(batch);
      break;
    }
    case 4 :
    {
      searchINMEM_c<T,4,8>(batch);
      break;
    }
    case 5 :
    {
      searchINMEM_c<T,5,16>(batch);
      break;
    }
    default:
      fprintf(stderr, "ERROR: %s has not been templated for %i stages\n", __FUNCTION__, noStages);
      exit(EXIT_FAILURE);
  }
}

__host__ void add_and_search_IMMEM(cuFFdotBatch* batch )
{
  if ( batch->rValues[0][0].numrs )
  {
    FOLD // Synchronisation  .
    {
      if      ( batch->flag & FLAG_SS_INMEM )
        cudaStreamWaitEvent(batch->srchStream, batch->searchComp, 0);
      else
        cudaStreamWaitEvent(batch->srchStream, batch->candCpyComp, 0);
    }

    FOLD // Timing event  .
    {
#ifdef TIMING // Timing event
      CUDA_SAFE_CALL(cudaEventRecord(batch->searchInit,  batch->srchStream),"Recording event: searchInit");
#endif
    }

    FOLD // Call the kernel  .
    {
      if ( batch->flag & FLAG_HALF  )
      {
#if __CUDACC_VER__ >= 70500
        searchINMEM_p<half>(batch);
#else
        fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
        exit(EXIT_FAILURE);
#endif

      }
      else
      {
        searchINMEM_p<float>(batch);
      }
    }

    FOLD // Synchronisation  .
    {
      CUDA_SAFE_CALL(cudaEventRecord(batch->searchComp,  batch->srchStream),"Recording event: searchComp");
    }
  }
}

/*
template<typename T, int chunkSz>
__global__ void search(T* read, uint iStride, uint firstBin, uint start, uint end, candPZs* d_cands, float maxPow )
{
  const int bidx    = threadIdx.y * SSIM_X  +  threadIdx.x;     /// Block index
  const int tid     = blockIdx.x  * SSIMBS  +  bidx;            /// Global thread id (ie column) 0 is the first 'good' column
  const int cStride = SSIMBS * MAX_BLKS;

  candPZs cand;
  cand.value = 0;

  if( tid < end - start)
  {
    int idx     = start + tid - firstBin;

    int zeroHeight  = HEIGHT_STAGE[0];
    short   lDepth  = ceilf(zeroHeight/(float)gridDim.y);
    short   y0      = lDepth*blockIdx.y;
    short   y1      = MIN(y0+lDepth, zeroHeight);
    float maxPowF   = maxPow;
    int   max       = 0;

    //    if ( start + tid == 20146 ) // TMP
    //    {
    //      printf("\n");
    //    }

    for( short y = y0; y < y1 ; y++)
    {
      float baseVal    = get(read, y*iStride + idx);    //read[ y  *iStride + idx ];

      //      if ( start + tid == 20146 ) // TMP
      //      {
      //        printf("%03i %10.5f\n", y, baseVal);
      //      }

      if ( baseVal > maxPowF )
      {
        maxPowF   = baseVal;
        max       = y;
      }
    }

    //    if ( start + tid == 20146 ) // TMP
    //    {
    //      printf("maxPowF: %10.5f\n\n", maxPowF);
    //    }

    FOLD // Write results  .
    {
      //candPZs cand;
      if ( maxPowF > maxPow )
      {
        cand.value  = maxPowF;
        cand.z      = max;
      }
      //      else
      //      {
      //        cand.value  = 0;
      //        //        cand.z      = -56;
      //      }

      //      if ( start + tid == 20146 ) // TMP
      //      {
      //        printf("cand: %10.5f\n\n", cand.value);
      //      }
    }
  }
  d_cands[ blockIdx.y*cStride + tid ] = cand;
}

template<typename T, int chunkSz>
__global__ void addSplit_k(T* read, uint iStride, T* write, uint oStride, uint firstBin, uint start, uint end, candPZs* d_cands, float maxPow )
{
  const int bidx    = threadIdx.y * SSIM_X  +  threadIdx.x;     /// Block index
  const int tid     = blockIdx.x  * SSIMBS  +  bidx;            /// Global thread id (ie column) 0 is the first 'good' column
  const int cStride = SSIMBS * MAX_BLKS;

  candPZs cand;
  cand.value = 0;

  if( tid < end - start)
  {
    int idx     = start + tid;
    int halfX   = round(idx/2.0);

    idx     -= firstBin;
    halfX   -= firstBin;

    if ( halfX > 0 )
    {
      int zeroHeight  = HEIGHT_STAGE[0];
      short   lDepth  = ceilf(zeroHeight/(float)gridDim.y);
      short   y0      = lDepth*blockIdx.y;
      short   y1      = MIN(y0+lDepth, zeroHeight);
      float maxPowF   = maxPow;
      int   max       = 0;

      //      if ( start + tid == 20146 ) // TMP
      //      {
      //        printf("\n");
      //      }

      for( short y = y0; y < y1 ; y++)
      {
        short iy1 = YINDS[ (zeroHeight+INDS_BUFF) + y ];

        float baseVal    = get(read, y  *iStride + idx);    //read[ y  *iStride + idx ];
        float halfVal    = get(read, iy1*iStride + halfX);  // read[ iy1*iStride + halfX ];

        //        if ( start + tid == 20146 ) // TMP
        //        {
        //          printf("%03i %10.5f + %10.5f = %10.5f   YINDS: %i \n", y, baseVal, halfVal, baseVal+halfVal, iy1 );
        //        }

        baseVal     += halfVal;

        //write[ y*oStride + idx ] = baseVal;
        set(write, y*oStride + idx, baseVal);

        if ( baseVal > maxPowF)
        {
          maxPowF   = baseVal;
          max       = y;
        }
      }

      //      if ( start + tid == 20146 ) // TMP
      //      {
      //        printf("maxPowF %10.5f\n", maxPowF);
      //      }

      FOLD // Write results  .
      {
        //candPZs cand;
        if ( maxPowF > maxPow )
        {
          cand.value  = maxPowF;
          cand.z      = max;
        }
        //        else
        //        {
        //          cand.value  = 0;
        //        }


      }
    }
  }

  d_cands[ blockIdx.y*cStride + tid ] = cand;
}
 */

__host__ void addSplit(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch, uint firstBin, uint start, uint end, int stage)
{
  //  FOLD // Timing event  .
  //  {
  //#ifdef TIMING // Timing event
  //    CUDA_SAFE_CALL(cudaEventRecord(batch->searchInit,  batch->strmSearch),"Recording event: searchInit");
  //#endif
  //  }
  //
  //  switch ( batch->ssChunk )
  //  {
  //    case 1 :
  //    {
  //      addSplit_k<1><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
  //      break;
  //    }
  //    case 2 :
  //    {
  //      addSplit_k<2><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
  //      break;
  //    }
  //    case 3 :
  //    {
  //      addSplit_k<3><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
  //      break;
  //    }
  //    case 4 :
  //    {
  //      addSplit_k<4><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
  //      break;
  //    }
  //    case 5 :
  //    {
  //      addSplit_k<5><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
  //      break;
  //    }
  //    case 6 :
  //    {
  //      addSplit_k<6><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
  //      break;
  //    }
  //    case 7 :
  //    {
  //      addSplit_k<7><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
  //      break;
  //    }
  //    case 8 :
  //    {
  //      addSplit_k<8><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
  //      break;
  //    }
  //    case 9 :
  //    {
  //      addSplit_k<9><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
  //      break;
  //    }
  //    case 10:
  //    {
  //      addSplit_k<10><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
  //      break;
  //    }
  //    case 12:
  //    {
  //      addSplit_k<12><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
  //      break;
  //    }
  //    default:
  //    {
  //      addSplit_k<2><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
  //    }
  //  }
  //
  //  FOLD // Synchronisation  .
  //  {
  //    CUDA_SAFE_CALL(cudaEventRecord(batch->searchComp,  batch->strmSearch),"Recording event: searchComp");
  //  }

}

//
//__host__ void processResults( cuFFdotBatch* batch, uint end, uint start, int stage, uint cStride )
//{
//  if ( end - start  > 0 )
//  {
//    FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host
//    {
//      nvtxRangePush("EventSynch");
//      CUDA_SAFE_CALL(cudaEventSynchronize(batch->candCpyComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
//      nvtxRangePop();
//    }
//
//#ifdef TIMING // Timing  .
//    float time;         // Time in ms of the thing
//    cudaError_t ret;    // Return status of cudaEventElapsedTime
//
//    FOLD // Search Timing  .
//    {
//      ret = cudaEventElapsedTime(&time, batch->searchInit, batch->searchComp);
//
//      if ( ret == cudaErrorNotReady )
//      {
//        //printf("Not ready\n");
//      }
//      else
//      {
//        //printf("    ready\n");
//#pragma omp atomic
//        batch->searchTime[0] += time;
//      }
//
//      CUDA_SAFE_CALL(cudaGetLastError(), "Search Timing");
//    }
//
//
//    FOLD // Copy D2H Timing  .
//    {
//      ret = cudaEventElapsedTime(&time, batch->candCpyInit, batch->candCpyComp);
//
//      if ( ret == cudaErrorNotReady )
//      {
//        //printf("Not ready\n");
//      }
//      else
//      {
//        //printf("    ready\n");
//#pragma omp atomic
//        batch->copyD2HTime[0] += time;
//      }
//
//      CUDA_SAFE_CALL(cudaGetLastError(), "Copy D2H Timing");
//    }
//
//    struct timeval startT, endT;
//    gettimeofday(&startT, NULL);
//#endif
//
//    nvtxRangePush("CPU Process results");
//
//    double poww, sig;
//    double rr, zz;
//    int numharm = (1<<stage);
//
//    FOLD // Critical section to handle candidates  .
//    {
//      uint idx;
//      uint x0, x1;
//      uint y0, y1;
//
//      y0 = 0;
//      y1 = batch->ssSlices;
//
//      x0 = 0;
//      x1 = end - start ;
//
//      float cutoff = batch->sInf->powerCut[stage];
//
//      for ( uint y = y0; y < y1; y++ )
//      {
//        for ( uint x = x0; x < x1; x++ )
//        {
//          poww      = 0;
//          sig       = 0;
//          zz        = 0;
//
//          idx       = y*cStride + x ;
//
//          FOLD
//          {
//            candPZs candM         = ((candPZs*)batch->h_retData)[idx];
//            if ( candM.value > cutoff )
//            {
//              sig                 = candM.value;
//              poww                = candM.value;
//              zz                  = candM.z;
//            }
//          }
//
//
//
//          if ( poww > 0 )
//          {
//            rr      = (start + x) *  ACCEL_DR ;
//
//            //            FOLD // TMP
//            //            {
//            //              uint bin   = start + x - batch->sInf->sSpec->fftInf.rlo*ACCEL_RDR;
//            //              uint stp   = bin / (float) batch->accelLen ;
//            //
//            //              if ( stp == 4 )
//            //              {
//            //                printf("%i\t%i\t%.4f\t%.5f\n", stage, x, rr, poww);
//            //              }
//            //            }
//
//            procesCanidate(batch, rr, zz, poww, sig, stage, numharm ) ;
//          }
//        }
//      }
//    }
//
//#ifdef TIMING // Timing  .
//    gettimeofday(&endT, NULL);
//    float v1 =  ((endT.tv_sec - startT.tv_sec) * 1e6 + (endT.tv_usec - startT.tv_usec))*1e-3  ;
//    batch->resultTime[0] += v1;
//#endif
//
//    nvtxRangePop();
//
//    printf("Stage %i got %4i cands \n",stage, batch->noResults );
//  }
//}

/*
__host__ void add_and_search_IMMEM_all(cuFFdotBatch* batch )
{
  dim3 dimBlock, dimGrid;

  dimBlock.x  = SSIM_X;
  dimBlock.y  = SSIM_Y;

  dimGrid.y   = batch->ssSlices;

  float bw    = SSIMBS;

  double lastBin_d  = batch->sInf->sSpec->fftInf.rlo*ACCEL_RDR + batch->SrchSz->noSteps * batch->accelLen ;
  double maxUint    = std::numeric_limits<uint>::max();

  if ( maxUint <= lastBin_d )
  {
    fprintf(stderr, "ERROR: There is not enough precision in uint in %s in %s.\n", __FUNCTION__, __FILE__ );
    exit(EXIT_FAILURE);
  }

  FOLD // Do synchronisations  .
  {
    //cudaStreamWaitEvent(batch->strmSearch, batch->stacks->ifftComp,  0);

    for (int ss = 0; ss< batch->noStacks; ss++)
    {
      cuFfdotStack* cStack = &batch->stacks[ss];

      FOLD // Synchronisation  .
      {
        //cudaEventRecord(cStack->ifftMemComp, batch->strmSearch);
        cudaStreamWaitEvent(batch->strmSearch, cStack->ifftMemComp,  0);
      }
    }
  }

  uint bSz        = MAX_BLKS;

  uint firstBin   = batch->sInf->sSpec->fftInf.rlo*ACCEL_RDR ;
  uint width      = batch->SrchSz->noSteps * batch->accelLen;
  uint lastBin    = firstBin + width;
  uint start;
  uint end;
  uint sWidth;
  float cndMemSz  = 0;
  float bffMemSz  = 0;
  float*  buffer  = NULL;

  uint powSz;

  if ( batch->retType & CU_HALF )
  {
    powSz = sizeof(half);
  }
  else
  {
    powSz = sizeof(float);
  }

  FOLD // Allocate device memory  .
  {
    cndMemSz  = bSz*bw*sizeof(candPZs)*batch->ssSlices;
    bffMemSz  = bSz*bw*batch->hInfos->height*powSz;

    cudaFreeNull(batch->d_retData);
    CUDA_SAFE_CALL(cudaMalloc(&buffer, bffMemSz ),   "Failed to allocate device memory for kernel stack.");

    cudaFreeNull(batch->d_retData);
    CUDA_SAFE_CALL(cudaMalloc(&batch->d_retData,  cndMemSz ),   "Failed to allocate device memory for kernel stack.");

    cudaFreeHostNull(batch->h_retData);
    CUDA_SAFE_CALL(cudaMallocHost(&batch->h_retData,  cndMemSz ),   "Failed to allocate device memory for kernel stack.");
  }

  for ( int stage = 0; stage < batch->sInf->noHarmStages; stage++ )
  {
    end         = lastBin;
    sWidth      = floor(end/2.0);
    int lastBlk = 0;
    int pStart  = 0;
    int pEnd    = 0;

    while ( (end > firstBin) )
    {
      sWidth        = ceil(end/2.0);
      float maxBlk  = floor(sWidth/bw);

      FOLD // Call SS kernel  .
      {
        FOLD // Timing event  .
        {
#ifdef TIMING // Timing event
          CUDA_SAFE_CALL(cudaEventRecord(batch->searchInit,  batch->strmSearch),"Recording event: searchInit");
#endif
        }

        if ( !lastBlk )
        {
          dimGrid.x     = MIN(maxBlk,bSz);

          sWidth        = bw * dimGrid.x ;
          start         = MAX(firstBin, end - sWidth);
        }
        else
        {
          sWidth        = end - firstBin;
          maxBlk        = ceil(sWidth/bw);
          dimGrid.x     = MIN(maxBlk,bSz);
          start         = firstBin;
        }

        if ( stage == 0 )
        {

          if ( batch->retType & CU_HALF )
          {
            search<half,0><<<dimGrid,  dimBlock, 0, batch->strmSearch >>>((half*)batch->d_planeFull, batch->sInf->mInf->inmemStride, firstBin, start, end, (candPZs*)batch->d_retData, batch->sInf->powerCut[stage] );
          }
          else
          {
            search<float,0><<<dimGrid,  dimBlock, 0, batch->strmSearch >>>((float*)batch->d_planeFull, batch->sInf->mInf->inmemStride, firstBin, start, end, (candPZs*)batch->d_retData, batch->sInf->powerCut[stage]);
          }
        }
        else
        {
          if ( !lastBlk )
          {
            if ( batch->retType & CU_HALF )
            {
              addSplit_k<half,0><<<dimGrid,  dimBlock, 0, batch->strmSearch >>>((half*)batch->d_planeFull, batch->sInf->mInf->inmemStride, (half*)batch->d_planeFull, batch->sInf->mInf->inmemStride, firstBin, start, end, (candPZs*)batch->d_retData, batch->sInf->powerCut[stage] );
            }
            else
            {
              addSplit_k<float,0><<<dimGrid,  dimBlock, 0, batch->strmSearch >>>((float*)batch->d_planeFull, batch->sInf->mInf->inmemStride, (float*)batch->d_planeFull, batch->sInf->mInf->inmemStride, firstBin, start, end, (candPZs*)batch->d_retData, batch->sInf->powerCut[stage]);
            }
          }
          else
          {
            if ( batch->retType & CU_HALF )
            {
              addSplit_k<half,0><<<dimGrid,  dimBlock, 0, batch->strmSearch >>>((half*)buffer, bSz*bw, (half*)batch->d_planeFull, batch->sInf->mInf->inmemStride, firstBin, start, end, (candPZs*)batch->d_retData, batch->sInf->powerCut[stage] );
            }
            else
            {
              addSplit_k<float,0><<<dimGrid,  dimBlock, 0, batch->strmSearch >>>((float*)buffer, bSz*bw, (float*)batch->d_planeFull, batch->sInf->mInf->inmemStride, firstBin, start, end, (candPZs*)batch->d_retData, batch->sInf->powerCut[stage]);
            }
          }

          //          if ( !lastBlk )
          //          {
          //            dimGrid.x     = MIN(maxBlk,bSz);
          //
          //            sWidth        = bw * dimGrid.x ;
          //            start         = MAX(firstBin, end - sWidth);
          //
          //            if ( batch->retType & CU_HALF )
          //            {
          //              addSplit_k<half,0><<<dimGrid,  dimBlock, 0, batch->strmSearch >>>((half*)batch->d_candidates, batch->sInf->mInf->inmemStride, (half*)batch->d_candidates, batch->sInf->mInf->inmemStride, firstBin, start, end, (candPZs*)batch->d_retData, batch->sInf->powerCut[stage] );
          //            }
          //            else
          //            {
          //              addSplit_k<float,0><<<dimGrid,  dimBlock, 0, batch->strmSearch >>>((float*)batch->d_candidates, batch->sInf->mInf->inmemStride, (float*)batch->d_candidates, batch->sInf->mInf->inmemStride, firstBin, start, end, (candPZs*)batch->d_retData, batch->sInf->powerCut[stage]);
          //            }
          //          }
          //          else
          //          {
          //            sWidth        = end - firstBin;
          //            maxBlk        = ceil(sWidth/bw);
          //            dimGrid.x     = MIN(maxBlk,bSz);
          //            start         = firstBin;
          //
          //            if ( batch->retType & CU_HALF )
          //            {
          //              addSplit_k<half,0><<<dimGrid,  dimBlock, 0, batch->strmSearch >>>((half*)buffer, bSz*bw, (half*)batch->d_candidates, batch->sInf->mInf->inmemStride, firstBin, start, end, (candPZs*)batch->d_retData, batch->sInf->powerCut[stage] );
          //            }
          //            else
          //            {
          //              addSplit_k<float,0><<<dimGrid,  dimBlock, 0, batch->strmSearch >>>((float*)buffer, bSz*bw, (float*)batch->d_candidates, batch->sInf->mInf->inmemStride, firstBin, start, end, (candPZs*)batch->d_retData, batch->sInf->powerCut[stage]);
          //            }
          //          }
        }

        FOLD // Synchronisation  .
        {
          CUDA_SAFE_CALL(cudaEventRecord(batch->searchComp,  batch->strmSearch),"Recording event: searchComp");
        }
      }

      FOLD // Process results  .
      {
        processResults( batch, pEnd, pStart, stage, bSz*bw );
      }

#ifdef TIMING // Timing event  .
      CUDA_SAFE_CALL(cudaEventRecord(batch->candCpyInit,  batch->strmSearch),"Recording event: candCpyInit");
#endif
      CUDA_SAFE_CALL(cudaMemcpyAsync(batch->h_retData, batch->d_retData, cndMemSz, cudaMemcpyDeviceToHost, batch->strmSearch), "Failed to copy results back");
      CUDA_SAFE_CALL(cudaEventRecord(batch->candCpyComp, batch->strmSearch),"Recording event: readComp");
      CUDA_SAFE_CALL(cudaGetLastError(), "Copying results back from device.");

      if ( maxBlk < bSz )
      {
        lastBlk = 1;
      }

      //sWidth        = end - firstBin;

#ifdef TIMING // Timing event  .

      float time;         // Time in ms of the thing
      cudaError_t ret;    // Return status of cudaEventElapsedTime

      FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host
      {
        nvtxRangePush("EventSynch");
        CUDA_SAFE_CALL(cudaEventSynchronize(batch->candCpyComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
        nvtxRangePop();
      }

      ret = cudaEventElapsedTime(&time, batch->searchInit, batch->searchComp);
#pragma omp atomic
      batch->searchTime[0] += time;

      ret = cudaEventElapsedTime(&time, batch->candCpyInit, batch->candCpyComp);
#pragma omp atomic
      batch->copyD2HTime[0] += time;

#endif

      pEnd          = end;
      pStart        = start;

      end           = start;
    }

    FOLD // Copy end over  .
    {
      if ( stage < batch->sInf->noHarmStages-1 )
      {
        float* dst;
        float* src;
        size_t  dpitch;
        size_t  spitch;
        size_t  width;
        size_t  height;

        dpitch  = bSz*bw * powSz;
        width   = bSz*bw * powSz;
        height  = batch->hInfos->height;
        spitch  = batch->sInf->mInf->inmemStride*powSz;
        dst     = buffer;
        src     = (float*)batch->d_planeFull;

        CUDA_SAFE_CALL(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, batch->strmSearch ),"Calling cudaMemcpy2DAsync after IFFT.");
      }
    }

    FOLD // Process the last set of results  .
    {
      processResults( batch, pEnd, pStart, stage, bSz*bw );
    }

    printf("Stage %i got %4i cands \n",stage, batch->noResults );
  }

  nvtxRangePush("EventSynch");
  CUDA_SAFE_CALL(cudaEventSynchronize(batch->candCpyComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
  nvtxRangePop();
}
 */
