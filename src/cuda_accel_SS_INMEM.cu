#include "cuda_accel_SS.h"

#define SSIM_X           16                    // X Thread Block
#define SSIM_Y           16                    // Y Thread Block
#define SSIMBS           (SSIM_X*SSIM_Y)

template<int chunkSz>
__global__ void addSplit_k(uint firstBin, uint start, uint end, candPZs* d_cands, int stage )
{
  const int bidx  = threadIdx.y * SSIM_X  +  threadIdx.x;     /// Block index
  const int tid   = blockIdx.x  * SSIMBS  +  bidx;            /// Global thread id (ie column) 0 is the first 'good' column

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
    float maxPow    = POWERCUT_STAGE[stage];
    int   max       = 0;

    for( short y = y0; y < y1 ; y++)
    {
      short iy1 = YINDS[ zeroHeight + y ];

      float read    = PLN_START[ iy1*PLN_STRIDE + halfX ];
      float base    = PLN_START[ y*PLN_STRIDE + idx ];
      base    += read;
      PLN_START[ y*PLN_STRIDE + idx ] = base;

      if ( base > maxPow)
      {
        maxPow  = base;
        max     = y;
      }
    }

    FOLD // Write results  .
    {
      candPZs cand;
      if ( maxPow > POWERCUT_STAGE[stage] )
      {
        cand.value  = maxPow;
        cand.z      = max;
      }
      else
      {
        cand.value  = 0;
      }

      d_cands[ blockIdx.y*PLN_STRIDE + idx ] = cand;
    }
  }
}

__host__ void addSplit(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch, uint firstBin, uint start, uint end, int stage)
{
  switch ( batch->ssChunk )
  {
    case 1 :
    {
      addSplit_k<1><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
      break;
    }
    case 2 :
    {
      addSplit_k<2><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
      break;
    }
    case 3 :
    {
      addSplit_k<3><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
      break;
    }
    case 4 :
    {
      addSplit_k<4><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
      break;
    }
    case 5 :
    {
      addSplit_k<5><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
      break;
    }
    case 6 :
    {
      addSplit_k<6><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
      break;
    }
    case 7 :
    {
      addSplit_k<7><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
      break;
    }
    case 8 :
    {
      addSplit_k<8><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
      break;
    }
    case 9 :
    {
      addSplit_k<9><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
      break;
    }
    case 10:
    {
      addSplit_k<10><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
      break;
    }
    case 12:
    {
      addSplit_k<12><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
      break;
    }
    default:
    {
      addSplit_k<2><<<dimGrid,  dimBlock, 0, stream >>>(firstBin, start, end, (candPZs*)batch->d_retData, stage);
    }
  }
}

void add_and_search_IMMEM(cuFFdotBatch* batch )
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

  uint firstBin  = batch->sInf->sSpec->fftInf.rlo*ACCEL_RDR ;
  uint width     = batch->SrchSz->noSteps * batch->accelLen;
  uint lastBin   = firstBin + width;
  uint start;
  uint end;
  uint sWidth;

  FOLD // TMP
  {
    cudaFreeNull(batch->d_retData);
    CUDA_SAFE_CALL(cudaMalloc(&batch->d_retData,  batch->sInf->mInf->inmemStride*sizeof(candPZs)*batch->ssSlices ),   "Failed to allocate device memory for kernel stack.");

    cudaFreeHostNull(batch->h_retData);
    CUDA_SAFE_CALL(cudaMallocHost(&batch->h_retData,  batch->sInf->mInf->inmemStride*sizeof(candPZs)*batch->ssSlices ),   "Failed to allocate device memory for kernel stack.");
  }

  for ( int stage = 0; stage < batch->sInf->noHarmStages; stage++ )
  {
    end         = lastBin;
    sWidth      = floor(end/2.0);

    while ( (end > firstBin) && (sWidth > bw * 10) )
    {
      sWidth      = floor(end/2.0);
      dimGrid.x   = floor(sWidth/bw);
      sWidth      = bw * dimGrid.x ;
      start       = MAX(firstBin, end - sWidth);
      sWidth      = end - start;

      addSplit(dimGrid, dimBlock, batch->strmSearch, batch, firstBin, start, end, stage );

#ifdef TIMING // Timing event  .
    CUDA_SAFE_CALL(cudaEventRecord(batch->candCpyInit,  batch->strmSearch),"Recording event: candCpyInit");
#endif
      CUDA_SAFE_CALL(cudaMemcpyAsync(batch->h_retData, batch->d_retData, sWidth*batch->ssSlices*sizeof(candPZs), cudaMemcpyDeviceToHost, batch->strmSearch), "Failed to copy results back");
      CUDA_SAFE_CALL(cudaEventRecord(batch->candCpyComp, batch->strmSearch),"Recording event: readComp");
      CUDA_SAFE_CALL(cudaGetLastError(), "Copying results back from device.");

      end         = start;
    }
  }

  nvtxRangePush("EventSynch");
  CUDA_SAFE_CALL(cudaEventSynchronize(batch->candCpyComp), "ERROR: copying result from device to host.");
  nvtxRangePop();
}
