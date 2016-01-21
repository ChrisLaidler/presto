#include "cuda_accel_SS.h"

#define SSIM_X           16                    // X Thread Block
#define SSIM_Y           16                    // Y Thread Block
#define SSIMBS           (SSIM_X*SSIM_Y)
#define MAX_BLKS         256


template<typename T, const int noStages, const int noHarms, const int cunkSize>
//__global__ void searchINMEM_k(T* __restrict__ read, int iStride, int cStride, int firstBin, int start, int end, candPZs* d_cands)
__global__ void searchINMEM_k(T* read, int iStride, int cStride, int firstBin, int start, int end, candPZs* d_cands)
{
  const int bidx        = threadIdx.y * SSIM_X  +  threadIdx.x;     /// Block index
  const int tid         = blockIdx.x  * SSIMBS  +  bidx;            /// Global thread id (ie column) 0 is the first 'good' column
  const int zeroHeight  = HEIGHT_STAGE[0];

  int             inds      [noHarms];
  candPZs         candLists [noStages];
  float           powers    [cunkSize];                             /// registers to hold values to increase mem cache hits

  int            idx   = start + tid ;
  int            len   = end - start;

  if ( tid < len )
  {
    FOLD  // Set the local and return candidate powers to zero  .
    {
      for ( int stage = 0; stage < noStages; stage++ )
      {
        candLists[stage].value = 0 ;
        d_cands[stage*gridDim.y*cStride + blockIdx.y*cStride + tid].value = 0;
      }
    }

    FOLD // Prep - Initialise the x indices  .
    {
      FOLD 	// Calculate the x indices or create a pointer offset by the correct amount  .
      {
        for ( int harm = 0; harm < noHarms; harm++ )         // Loop over harmonics (batch) in this stage  .
        {
          int   ix        = roundf( idx*FRAC_STAGE[harm] ) - firstBin;
          //int   ix        = floorf( idx*FRAC_STAGE[harm] ) - firstBin;
          inds[harm]      = ix;
        }
      }
    }

    FOLD // Sum & Search - Ignore contaminated ends tid to starts at correct spot  .
    {
      int   lDepth        = ceilf(zeroHeight/(float)gridDim.y);
      int   y0            = lDepth*blockIdx.y;
      int   y1            = MIN(y0+lDepth, zeroHeight);
      int   yIndsChnksz   = zeroHeight+INDS_BUFF;

      for( int y = y0; y < y1 ; y += cunkSize )                     // loop over chunks  .
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

            //if ( inds[end] >= 0 ) // Valid stage
            {
              FOLD // Create a section of summed powers one for each step  .
              {
                for ( int harm = start; harm <= end; harm++ )         // Loop over harmonics (batch) in this stage  .
                {
                  int     ix1   = inds[harm] ;
                  //T* sub = read + inds[harm] ;

                  if ( ix1 >= 0 ) // Valid stage
                  {
                    int   iyP       = -1;                               // The previous y-index used
                    float pow       = 0 ;

                    for( int yPlus = 0; yPlus < cunkSize; yPlus++ )     // Loop over the chunk  .
                    {
                      int yPln     = y + yPlus ;                         ///< True Y index in plane

                      // Don't check yPln against zeroHeight, YINDS contains a buffer at the end, only do the check later
                      int iy1     = YINDS[ yIndsChnksz*harm + yPln ];

                      if ( iyP != iy1 ) // Only read power if it is not the same as the previous  .
                      {
                        int izz = iy1*iStride + ix1 ;

//#ifdef DEBUG
//                        if ( izz >= zeroHeight * iStride || izz < 0  )
//                        {
//                          printf("ERROR: thread %i is assessing out of bounds memory in %s!\n", tid, __FUNCTION__ );
//                        }
//                        else
//#endif
                          //if ( harm < 5 )
                        pow = get(read, izz );

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
          d_cands[stage*gridDim.y*cStride + blockIdx.y*cStride + tid] = candLists[stage];
        }
      }
    }

  }
}

template<typename T, const int noStages, const int noHarms>
__host__ void searchINMEM_c(cuFFdotBatch* batch )
{
  dim3 dimBlock, dimGrid;

  rVals* rVal = &(*batch->rAraays)[batch->rActive][0][0];

  FOLD // Check if we can use the specific data types in the kernel  .
  {
    double lastBin_d  = batch->sInf->sSpec->fftInf.rlo*ACCEL_RDR + batch->sInf->SrchSz->noSteps * batch->accelLen ;
    double maxUint    = std::numeric_limits<uint>::max();
    if ( maxUint <= lastBin_d )
    {
      fprintf(stderr, "ERROR: There is not enough precision in uint in %s in %s.\n", __FUNCTION__, __FILE__ );
      exit(EXIT_FAILURE);
    }

    lastBin_d  = batch->sInf->inmemStride * batch->hInfos->height;
    double maxInt    = std::numeric_limits<int>::max();
    if ( maxInt <= lastBin_d )
    {
      fprintf(stderr, "ERROR: There is not enough precision in int in %s in %s.\n", __FUNCTION__, __FILE__ );
      exit(EXIT_FAILURE);
    }

  }

  uint firstBin = batch->sInf->SrchSz->searchRLow * ACCEL_RDR ;
  uint start    = rVal->drlo * ACCEL_RDR ;
  uint end      = start + rVal->numrs;
  uint noBins   = end - start;

  dimBlock.x    = SSIM_X;
  dimBlock.y    = SSIM_Y;

  dimGrid.y     = batch->ssSlices;
  dimGrid.x     = ceil(noBins / (float) SSIMBS );

  switch ( batch->ssChunk )
  {
//    case 1 :
//    {
//      searchINMEM_k<T,noStages,noHarms,1><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->sInf->d_planeFull, batch->sInf->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1 );
//      break;
//    }
//    case 2 :
//    {
//      searchINMEM_k<T,noStages,noHarms,2><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->sInf->d_planeFull, batch->sInf->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1 );
//      break;
//    }
//    case 3 :
//    {
//      searchINMEM_k<T,noStages,noHarms,3><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->sInf->d_planeFull, batch->sInf->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1 );
//      break;
//    }
//    case 4 :
//    {
//      searchINMEM_k<T,noStages,noHarms,4><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->sInf->d_planeFull, batch->sInf->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1 );
//      break;
//    }
//    case 5 :
//    {
//      searchINMEM_k<T,noStages,noHarms,5><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->sInf->d_planeFull, batch->sInf->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1 );
//      break;
//    }
//    case 6 :
//    {
//      searchINMEM_k<T,noStages,noHarms,6><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->sInf->d_planeFull, batch->sInf->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1 );
//      break;
//    }
//    case 7 :
//    {
//      searchINMEM_k<T,noStages,noHarms,7><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->sInf->d_planeFull, batch->sInf->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1 );
//      break;
//    }
//    case 8 :
//    {
//      searchINMEM_k<T,noStages,noHarms,8><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->sInf->d_planeFull, batch->sInf->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1 );
//      break;
//    }
    case 9 :
    {
      searchINMEM_k<T,noStages,noHarms,9><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->sInf->d_planeFull, batch->sInf->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1 );
      break;
    }
//    case 10:
//    {
//      searchINMEM_k<T,noStages,noHarms,10><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->sInf->d_planeFull, batch->sInf->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1 );
//      break;
//    }
//    case 12:
//    {
//      searchINMEM_k<T,noStages,noHarms,12><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->sInf->d_planeFull, batch->sInf->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1 );
//      break;
//    }
//    case 14:
//    {
//      searchINMEM_k<T,noStages,noHarms,14><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->sInf->d_planeFull, batch->sInf->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1 );
//      break;
//    }
//    case 20:
//    {
//      searchINMEM_k<T,noStages,noHarms,20><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->sInf->d_planeFull, batch->sInf->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1 );
//      break;
//    }
//    case 25:
//    {
//      searchINMEM_k<T,noStages,noHarms,25><<<dimGrid,  dimBlock, 0, batch->srchStream >>>((T*)batch->sInf->d_planeFull, batch->sInf->inmemStride, batch->strideOut, firstBin, start, end, (candPZs*)batch->d_outData1 );
//      break;
//    }
    default:
    {
      fprintf(stderr, "ERROR: %s has not been templated for %i chunk size.\n", __FUNCTION__, batch->ssChunk);
      exit(EXIT_FAILURE);
    }
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
  FOLD // Timing  .
  {
    if ( batch->flags & FLAG_TIME )
    {
      infoMSG(3,3,"Timing\n");

      if ( (*batch->rAraays)[batch->rActive+1][0][0].numrs )
      {
        // Inmem Sum and Search kernel
        timeEvents( batch->searchInit, batch->searchComp, &batch->searchTime[0],   "Search kernel");
      }
    }
  }

  if ( (*batch->rAraays)[batch->rActive][0][0].numrs )
  {
    infoMSG(1,2,"In-mem sum and search\n");

    FOLD // Synchronisation  .
    {
      infoMSG(3,3,"pre synchronisation\n");

      if      ( batch->flags & FLAG_SS_INMEM )
      {
        CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->srchStream, batch->searchComp, 0),  "Waiting on event searchComp");
      }
      else
      {
        CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->srchStream, batch->candCpyComp, 0), "Waiting on event candCpyComp");
      }
    }

    FOLD // Timing event  .
    {
      if ( batch->flags & FLAG_TIME ) // Timing event
      {
        CUDA_SAFE_CALL(cudaEventRecord(batch->searchInit,  batch->srchStream),        "Recording event: searchInit");
      }
    }

    FOLD // Call the kernel  .
    {
      infoMSG(3,3,"S&S Kernel\n");

      if ( batch->flags & FLAG_HALF  )
      {
#if CUDA_VERSION >= 7050
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

      CUDA_SAFE_CALL(cudaGetLastError(), "Calling searchINMEM kernel.");
    }

    FOLD // Synchronisation  .
    {
      infoMSG(3,3,"post synchronisation\n");

      CUDA_SAFE_CALL(cudaEventRecord(batch->searchComp,  batch->srchStream),"Recording event: searchComp");

#ifdef DEBUG // This is just a hack, I'm not sure why this is necessary but it appears it is
      if ( batch->flags & FLAG_SYNCH ) // TMP
      {
        infoMSG(4,4,"DEBUG synchronisation blocking.\n");

        CUDA_SAFE_CALL(cudaEventSynchronize(batch->searchComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
      }
#endif
    }
  }
}
