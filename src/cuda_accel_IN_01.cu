#include "cuda_accel_IN.h"
#include "cuda_sort.h"




template<int batches>
__device__ void scaleAndSpread(float2* data, int stride, int noRespPerBin, const float factor, const int noEls)
{
  //const int bid = blockIdx.y  * gridDim.x  + blockIdx.x;	/// Block ID (flat index)
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;	/// Thread ID in block (flat index)
  const int bSz = NAS_NTRD;					/// Block size

  for ( int batch = batches-1; batch >= 0; batch--)
  {
    int idx	= batch*bSz+tid;
    int expIdx	= idx * noRespPerBin;

    // Read all values into registers
    float2 val = data[idx];

    __syncthreads(); // Needed to ensure all values are in registers before writing data

    if ( expIdx < stride )
    {
      // Set the value to normalised complex number spread by 2
      if ( idx < noEls )
      {
	val.y *= factor;
	val.x *= factor;
      }
      else
      {
	val.y = 0;
	val.x = 0;
      }
      data[expIdx] = val;

      // Set every second value to 0
      for (int i = 1; i < noRespPerBin; i++ )
      {
	val.y = 0;
	val.x = 0;
	data[expIdx+i]   = val;
      }
    }
  }
}

/** Kernel to calculate median of the powers of complex values and spread and normalise the complex using the calculated median value
 *
 * @param data    The data, initially the input is in the first half, results are written to the same location spread by 2
 * @param lens    The lengths of the individual input sections
 */
template<int noEls>
__global__ void normAndSpread_SM(float2* data, int stride, int noRespPerBin)
{
  const int bid = blockIdx.y  * gridDim.x  + blockIdx.x;	/// Block ID (flat index)
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;	/// Thread ID in block (flat index)
  const int bSz = NAS_NTRD;					/// Block size

  const int batches 	= ( noEls + (NAS_NTRD - 1 ) ) / (NAS_NTRD) ; // int round up done at compile time ;)
  float     factor      = 1.0f;

  // Stride input data
  data += stride*bid;

  FOLD  //  Calculate the normalisation factor from the median .
  {
    int os = noEls/2-1;

    float median = 1.0f;

    __shared__ float smData[noEls];				/// SM to store powers

    FOLD // Read values into SM  .
    {
      for ( int batch = 0; batch < batches; batch++)
      {
	int idx = batch*bSz+tid;

	if ( idx < noEls )
	{
	  float2 val      = data[idx];
	  smData[idx]     = val.x*val.x+val.y*val.y;
	}
      }
      __syncthreads();	// Writing values to SM
    }

    // Sort
    bitonicSort_SM<float, noEls>(smData);
    median = smData[os];

    // Calculate normalisation factor
    factor = 1.0 / sqrtf( median / (float)LN2 );
  }

  scaleAndSpread<batches>(data, stride, noRespPerBin, factor, noEls);
}

/** Kernel to calculate median of the powers of complex values and spread and normalise the complex using the calculated median value
 *
 * @param data    The data, initially the input is in the first half, results are written to the same location spread by 2
 * @param lens    The lengths of the individual input sections
 */
template<int noEls>
__global__ void normAndSpread_SM_MIN(float2* data, int stride, int noRespPerBin)
{
  const int bid = blockIdx.y  * gridDim.x  + blockIdx.x;	/// Block ID (flat index)
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;	/// Thread ID in block (flat index)
  const int bSz = NAS_NTRD;					/// Block size

  const int batches 	= ( noEls + (NAS_NTRD - 1 ) ) / (NAS_NTRD) ;
  float     factor      = 1.0f;

  // Stride input data
  data += stride*bid;

  FOLD  //  Calculate the normalisation factor from the median .
  {
    float  powers[batches];					/// Registers to store powers
    int    os		= noEls/2-1;
    float  median 	= 1.0f;

    FOLD // Read values into registers  .
    {
      for ( int batch = 0; batch < batches; batch++)
      {
	int idx = batch*bSz+tid;

	if ( idx < noEls )
	{
	  float2 val      = data[idx];
	  powers[batch]   = val.x*val.x+val.y*val.y;
	}
      }
    }

    // Sort
    bitonicSort_reg<float, noEls, batches>(powers);
    median = getValue<float, batches>(powers, os);

    // Calculate normalisation factor
    factor = 1.0 / sqrtf( median / (float)LN2 );
  }

  scaleAndSpread<batches>(data, stride, noRespPerBin, factor, noEls);
}

/** Kernel to calculate median of the powers of complex values and spread and normalise the complex using the calculated median value
 *
 * @param data    The data, initially the input is in the first half, results are written to the same location spread by 2
 * @param lens    The lengths of the individual input sections
 */
#ifdef WITH_NORM_GPU_OS
template<int noEls>
__global__ void normAndSpread_OS(float2* data, int stride, int noRespPerBin)
{
  const int bid = blockIdx.y  * gridDim.x  + blockIdx.x;	/// Block ID (flat index)
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;	/// Thread ID in block (flat index)
  const int bSz = NAS_NTRD;					/// Block size

  const int batches 	= ( noEls + (NAS_NTRD - 1 ) ) / (NAS_NTRD) ;
  float     factor      = 1.0f;

  // Stride input data
  data += stride*bid;

  FOLD  //  Calculate the normalisation factor from the median .
  {
    float  powers[batches];					/// Registers to store powers
    int    os		= noEls/2-1;
    float  median 	= 1.0f;

    FOLD // Read values into registers  .
    {
      for ( int batch = 0; batch < batches; batch++)
      {
	int idx = batch*bSz+tid;

	if ( idx < noEls )
	{
	  float2 val      = data[idx];
	  powers[batch]   = val.x*val.x+val.y*val.y;
	}
      }
    }

    // Get median
    median = cuOrderStatPow2_radix<noEls>(powers, os, 0);

    // Calculate normalisation factor
    factor = 1.0 / sqrtf( median / (float)LN2 );
  }

  scaleAndSpread<batches>(data, stride, noRespPerBin, factor, noEls);
}
#endif

/** A function in the template tree used to call the CUDA median normalisation kernel
 *
 * This function determines the buffer width and
 * calls the function in the template tree
 *
 */
template<int noEls>
void normAndSpread_m(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t stream, cuFfdotStack* cStack )
{
  FOLD // Call flag template  .
  {
    const int64_t FLAGS = cStack->flags;

    if      ( FLAGS & CU_NORM_GPU_SM     )
    {
#ifdef WITH_NORM_GPU
      normAndSpread_SM<noEls><<< dimGrid,  dimBlock, 0, stream >>>((float2*)cStack->d_iData, cStack->width, cStack->harmInf->noResPerBin );
#else
      fprintf(stderr, "ERROR: %s disabled at compile time. Function %s in %s.\n", "GPU normalisation", __FUNCTION__, __FILE__);
      exit(EXIT_FAILURE);
#endif
    }
    else if ( FLAGS & CU_NORM_GPU_SM_MIN )
    {
#ifdef WITH_NORM_GPU
      if ( noEls <= 1024 )
      {
	normAndSpread_SM<noEls><<< dimGrid,  dimBlock, 0, stream >>>((float2*)cStack->d_iData, cStack->width, cStack->harmInf->noResPerBin );
      }
      else
      {
	normAndSpread_SM_MIN<noEls><<< dimGrid,  dimBlock, 0, stream >>>((float2*)cStack->d_iData, cStack->width, cStack->harmInf->noResPerBin );
      }
#else
      fprintf(stderr, "ERROR: %s disabled at compile time. Function %s in %s.\n", "GPU normalisation", __FUNCTION__, __FILE__);
      exit(EXIT_FAILURE);
#endif
    }
    else if ( FLAGS & CU_NORM_GPU_OS     )
    {
#ifdef WITH_NORM_GPU_OS
      if ( noEls <= 1024 )
      {
	normAndSpread_SM<noEls><<< dimGrid,  dimBlock, 0, stream >>>((float2*)cStack->d_iData, cStack->width, cStack->harmInf->noResPerBin );
      }
      else
      {
	normAndSpread_OS<noEls><<< dimGrid,  dimBlock, 0, stream >>>((float2*)cStack->d_iData, cStack->width, cStack->harmInf->noResPerBin );
      }
#else
      fprintf(stderr, "ERROR: %s disabled at compile time. Function %s in %s.\n", "GPU normalisation by radix", __FUNCTION__, __FILE__);
      exit(EXIT_FAILURE);
#endif
    }
    else
    {
      fprintf(stderr,"ERROR: No valid GPU normalisation specified. Function %s in %s.\n", __FUNCTION__, __FILE__);
      exit(EXIT_FAILURE);
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
  switch ( numData )
  {
    case 64    :
    {
      normAndSpread_m<64   >(dimGrid, dimBlock, 0, stream, cStack );
      break;
    }
    case 128   :
    {
      normAndSpread_m<128 >(dimGrid, dimBlock, 0, stream, cStack );
      break;
    }
    case 256   :
    {
      normAndSpread_m<256 >(dimGrid, dimBlock, 0, stream, cStack );
      break;
    }
    case 512   :
    {
      normAndSpread_m<512 >(dimGrid, dimBlock, 0, stream, cStack );
      break;
    }
    case 1024  :
    {
      normAndSpread_m<1024>(dimGrid, dimBlock, 0, stream, cStack );
      break;
    }
    case 2048  :
    {
      normAndSpread_m<2048>(dimGrid, dimBlock, 0, stream, cStack );
      break;
    }
    case 4096  :
    {
      normAndSpread_m<4096>(dimGrid, dimBlock, 0, stream, cStack );
      break;
    }
    case 8192  :
    {
      normAndSpread_m<8192>(dimGrid, dimBlock, 0, stream, cStack );
      break;
    }
    default    :
    {
      fprintf(stderr, "ERROR: %s has not been templated for sorting with %i elements. Try CPU input normalisation.\n", __FUNCTION__, numData );
      exit(EXIT_FAILURE);
      break;
    }
  }
}

/** A function in the template tree used to call the CUDA median normalisation kernel
 *
 * This function determines the data width and
 * calls the function in the template tree
 *
 */
__host__ void normAndSpread(cudaStream_t stream, cuCgPlan* plan, uint stack )
{
  dim3 dimBlock, dimGrid;
  cuFfdotStack* cStack = &plan->stacks[stack];

  // Blocks of 1024 threads ( the maximum number of threads per block )
  dimBlock.x = NAS_DIMX;
  dimBlock.y = NAS_DIMY;
  dimBlock.z = 1;

  // One block per plane of the stack, thus we can sort input powers in Shared memory
  dimGrid.x = cStack->noInStack * plan->noSegments;
  dimGrid.y = 1;

  if ( plan->conf->normType != 0 )
  {
    fprintf(stderr, "ERROR: GPU normalisation can only perform old-style block median normalisation of the segment input.\n");
    exit(EXIT_FAILURE);
  }

  rVals* rVal 	= &(*plan->rAraays)[plan->rActive][0][cStack->startIdx];
  int numData	= rVal->numdata;		// NB: This assumes all segments have the same numdata

  // Call the templated kernel chain
  normAndSpread_w(dimGrid, dimBlock, 0, stream, cStack, numData );

  CUDA_SAFE_CALL(cudaGetLastError(), "Calling the normalisation and spreading kernel.");
}
