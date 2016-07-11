#include "cuda_accel_IN.h"
#include "cuda_median.h"


/** Kernel to calculate median of the powers of complex values and spread and normalise the complex using the calculated median value
 *
 * @param data    The data, initially the input is in the first half, results are written to the same location spread by 2
 * @param lens    The lengths of the individual input sections
 */
template<int BS_MAX>
__global__ void normAndSpread_k(fcomplexcu* data, int stride, int lenth, int noRespPerBin)
{
  const int bid = blockIdx.y  * gridDim.x  + blockIdx.x;        /// Block ID (flat index)
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;       /// Thread ID in block (flat index)
  const int bSz = blockDim.x  * blockDim.y;                     /// Block size

  int width = lenth;
  if (width)
  {
    __shared__ float sData[BS_MAX];

    float   medianValue;
    float   factor;

    int batches = ceilf( width / (float) bSz );

    // Stride input data
    data += stride*bid;

    if ( width <= BS_MAX )
    {
      FOLD // Calculate and store powers in shared memory  .
      {
        for ( int batch = 0; batch < batches; batch++)
        {
          int idx = batch*bSz+tid;

          if ( idx < width )
          {
            fcomplexcu val  = data[idx];
            sData[idx]      = val.r*val.r+val.i*val.i;
          }
        }
      }

      medianValue = cuMedianOne(sData, width);
    }
    else // Use device memory for powers
    {
      // The output data is spread by two
      // This the input is only in the first half and we can use the last half to store the powers

      float* powers = (float*)(&data[stride/2]); // Stride should always be a power of 2

      FOLD // Calculate and store powers in device memory  .
      {
        for ( int batch = 0; batch < batches; batch++)
        {
          fcomplexcu val            = data[batch*bSz+tid];
          powers[batch*bSz  + tid]  = val.r*val.r+val.i*val.i;
        }
      }

      medianValue = cuMedianBySection<BS_MAX>(powers, sData, width);
    }

    // Calculate normalisation factor
    factor = 1.0 / sqrt( medianValue / LN2 );

    batches = ceil( stride / (float) bSz );

    // Write, spread and normalised
    for ( int batch = batches-1; batch >= 0; batch--)
    {
      int idx	= batch*bSz+tid;
      int expIdx = idx * noRespPerBin;

      // Read all values into registers
      fcomplexcu val = data[idx];

      __syncthreads(); // Needed to ensure all values are in refisters before writing data

      if ( expIdx < stride)
      {
        // Set the value to normalised complex number spread by 2
        if ( idx < width )
        {
          val.i *= factor;
          val.r *= factor;
        }
        else
        {
          val.i = 0;
          val.r = 0;
        }
        data[expIdx]     = val;

        // Set every second value to 0
        for (int i = 1; i < noRespPerBin; i++ )
        {
          val.i = 0;
          val.r = 0;
          data[expIdx+i]   = val;
        }
      }
    }
  }
}

/** A function in the template tree used to call the CUDA median normalisation kernel
 *
 * This function determines the buffer width and
* calls the function in the template tree
 *
 */
void normAndSpread_w(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t stream, cuFfdotStack* cStack, int numData )
{
  int bufWidth;

  if ( cuMedianBuffSz > 0 )
  {
    bufWidth            = cuMedianBuffSz;
  }
  else
  {
    if ( cStack->width > 8192 )
      bufWidth          = 2048;			// Multi bitonic sort with a buffer size of 2048 (2048 found to be close to optimal most of the time)
    else
      bufWidth          = numData; 		// Use the actual number of input elements (1 bitonic sort)
  }

  bufWidth              = MAX( (numData)/32.0f, bufWidth );
  bufWidth              = MAX( 128,  bufWidth );
  bufWidth              = MIN( 8192, bufWidth );

  // TODO: Profile this with noResPerBin templated

  switch ( bufWidth )
  {
    case 128   :
    {
      normAndSpread_k<128><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, cStack->width, numData, cStack->harmInf->noResPerBin );
      break;
    }
    case 256   :
    {
      normAndSpread_k<256><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, cStack->width, numData, cStack->harmInf->noResPerBin );
      break;
    }
    case 512   :
    {
      normAndSpread_k<512><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, cStack->width, numData, cStack->harmInf->noResPerBin );
      break;
    }
    case 1024  :
    {
      normAndSpread_k<1024><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, cStack->width, numData, cStack->harmInf->noResPerBin );
      break;
    }
    case 2048  :
    {
      normAndSpread_k<2048><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, cStack->width, numData, cStack->harmInf->noResPerBin );
      break;
    }
    case 4096  :
    {
      normAndSpread_k<4096><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, cStack->width, numData, cStack->harmInf->noResPerBin );
      break;
    }
    case 8192  :
    {
      normAndSpread_k<8192><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, cStack->width, numData, cStack->harmInf->noResPerBin );
      break;
    }
    default    :
    {
      fprintf(stderr, "ERROR: %s has not been templated for sorting with %i elements.\n", __FUNCTION__, bufWidth );
      exit(EXIT_FAILURE);
    }
  }
}

/** A function in the template tree used to call the CUDA median normalisation kernel
 *
 * This function determines the data width and
 * calls the function in the template tree
 *
 */
__host__ void normAndSpread(cudaStream_t stream, cuFFdotBatch* batch, uint stack )
{
  dim3 dimBlock, dimGrid;
  cuFfdotStack* cStack = &batch->stacks[stack];

  // Blocks of 1024 threads ( the maximum number of threads per block )
  dimBlock.x = NAS_DIMX;
  dimBlock.y = NAS_DIMY;
  dimBlock.z = 1;

  // One block per harmonic, thus we can sort input powers in Shared memory
  dimGrid.x = cStack->noInStack * batch->noSteps;
  dimGrid.y = 1;

  if ( batch->cuSrch->sSpec->normType != 0 )
  {
    fprintf(stderr, "ERROR: GPU normalisation can only pefrom oldstyle block median normalisation of the step input.\n");
    exit(EXIT_FAILURE);
  }

  rVals* rVal 	= &(*batch->rAraays)[batch->rActive][0][cStack->startIdx];
  int numData	=  rVal->numdata;

  normAndSpread_w(dimGrid, dimBlock, 0, stream, cStack, numData );

  CUDA_SAFE_CALL(cudaGetLastError(), "Calling the normalisation and spreading kernel.");
}
