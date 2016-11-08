#include "cuda_accel_IN.h"
#include "cuda_sort.h"


/** Kernel to calculate median of the powers of complex values and spread and normalise the complex using the calculated median value
 *
 * @param data    The data, initially the input is in the first half, results are written to the same location spread by 2
 * @param lens    The lengths of the individual input sections
 */
template<int noEls>
__global__ void normAndSpread_k(fcomplexcu* data, int stride, int noRespPerBin)
{
  const int bid = blockIdx.y  * gridDim.x  + blockIdx.x;        /// Block ID (flat index)
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;       /// Thread ID in block (flat index)
  const int bSz = blockDim.x  * blockDim.y;                     /// Block size

  const int batches 	= ( noEls + (NAS_DIMX*NAS_DIMY - 1 ) ) / ( NAS_DIMX*NAS_DIMY) ;

  float   factor;

  // Stride input data
  data += stride*bid;

  FOLD  //  .
  {
    FOLD // Calculate and store powers in shared memory  .
    {
      int os = noEls/2-1;

      float  powers[batches];				/// Registers to store powers

      float median_orderstat_radix = 1;
      float median_orderstat_sort = 1;
      float median_Sort = 1;
      float median_Sort_mult = 1;
      float median = 1;

//      FOLD // Order stat - sort  .
//      {
//	for ( int batch = 0; batch < batches; batch++)
//	{
//	  int idx = batch*bSz+tid;
//
//	  if ( idx < noEls )
//	  {
//	    fcomplexcu val  = data[idx];
//	    powers[batch]   = val.r*val.r+val.i*val.i;
//	  }
//	}
//	median_orderstat_sort = cuOrderStatPow2_sort<float, noEls, batches>(powers, os);
//
//	__syncthreads();
//
//	median = median_orderstat_sort;
//      }

//      FOLD // Order stat radix  .
//      {
//	for ( int batch = 0; batch < batches; batch++)
//	{
//	  int idx = batch*bSz+tid;
//
//	  if ( idx < noEls )
//	  {
//	    fcomplexcu val  = data[idx];
//	    powers[batch]   = val.r*val.r+val.i*val.i;
//	  }
//	}
//	median_orderstat_radix = cuOrderStatPow2_radix<noEls>(powers, os, 0);
//
//	__syncthreads();
//
//	median = median_orderstat_radix;
//      }

      FOLD // Sort - SM  .
      {
	__shared__ float smData[noEls];
	for ( int batch = 0; batch < batches; batch++)
	{
	  int idx = batch*bSz+tid;

	  if ( idx < noEls )
	  {
	    fcomplexcu val  = data[idx];
	    smData[idx]     = val.r*val.r+val.i*val.i;
	  }
	}
	__syncthreads();

	bitonicSort_mem<float, noEls>(smData);

	__syncthreads();

	median_Sort = smData[os];

	__syncthreads();

	median = median_Sort;
      }

//      FOLD // Sort SM 1024  .
//      {
//	for ( int batch = 0; batch < batches; batch++)
//	{
//	  int idx = batch*bSz+tid;
//
//	  if ( idx < noEls )
//	  {
//	    fcomplexcu val  = data[idx];
//	    powers[batch]   = val.r*val.r+val.i*val.i;
//	  }
//	}
//	bitonicSort_reg<float, noEls, batches>(powers);
//
//	__syncthreads();
//
//	median_Sort = getValue<float, batches>(powers, os);
//
//	__syncthreads();
//
//	median = median_Sort;
//      }




      // Calculate normalisation factor
      factor = 1.0 / sqrtf( median / (float)LN2 );
    }
  }

  // Write, spread and normalised
  for ( int batch = batches-1; batch >= 0; batch--)
  {
    int idx	 = batch*bSz+tid;
    int expIdx = idx * noRespPerBin;

    // Read all values into registers
    fcomplexcu val = data[idx];

    __syncthreads(); // Needed to ensure all values are in registers before writing data

    if ( expIdx < stride)
    {
      // Set the value to normalised complex number spread by 2
      if ( idx < noEls )
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
    case 64   :
    {
      normAndSpread_k<64  ><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, cStack->width, cStack->harmInf->noResPerBin );
      break;
    }
    case 128   :
    {
      normAndSpread_k<128 ><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, cStack->width, cStack->harmInf->noResPerBin );
      break;
    }
    case 256   :
    {
      normAndSpread_k<256 ><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, cStack->width, cStack->harmInf->noResPerBin );
      break;
    }
    case 512   :
    {
      normAndSpread_k<512 ><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, cStack->width, cStack->harmInf->noResPerBin );
      break;
    }
    case 1024  :
    {
      normAndSpread_k<1024><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, cStack->width, cStack->harmInf->noResPerBin );
      break;
    }
    case 2048  :
    {
      normAndSpread_k<2048><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, cStack->width, cStack->harmInf->noResPerBin );
      break;
    }
    case 4096  :
    {
      normAndSpread_k<4096><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, cStack->width, cStack->harmInf->noResPerBin );
      break;
    }
    case 8192  :
    {
      normAndSpread_k<8192><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, cStack->width, cStack->harmInf->noResPerBin );
      break;
    }
    default    :
    {
      fprintf(stderr, "ERROR: %s has not been templated for sorting with %i elements.\n", __FUNCTION__, numData );
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
    fprintf(stderr, "ERROR: GPU normalisation can only perform old-style block median normalisation of the step input.\n");
    exit(EXIT_FAILURE);
  }

  rVals* rVal 	= &(*batch->rAraays)[batch->rActive][0][cStack->startIdx];
  int numData	=  rVal->numdata;

  normAndSpread_w(dimGrid, dimBlock, 0, stream, cStack, numData );

  CUDA_SAFE_CALL(cudaGetLastError(), "Calling the normalisation and spreading kernel.");
}
