#define FOLD  if(1)
#define Fout  if(0)

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>


#include <curand.h>
#include <curand_kernel.h>


#include "cuda_math.h"
#include "cuda_sort.h"

//#define SORT_DBG

/** Compare and swap two values (if they are in the wrong order)  .
 *
 * @param valA The first value
 * @param valB The second value
 * @param dir direction to sort data ( 1 -> smallest to largest AND -1 -> largest to smallest )
 */
__device__ inline void Comparator(float &valA, float &valB, uint dir)
{
  if ((valA > valB) == dir)
  {
    register float t;
    //swap(*(int*)&valA, *(int*)&valB );
    t     = valA;
    valA  = valB;
    valB  = t;
  }
}

/** XOR swap two integer values  .
 *
 * @param a integer a
 * @param b integer b
 */
__device__ inline void swap(int & a, int & b)
{
  a = a ^ b;
  b = a ^ b;
  a = a ^ b;
}

/** Butterfly shuffle swap  .
 *
 * @param x
 * @param mask
 * @param dir
 * @return
 */
template <typename T>
__device__ inline T shflSwap(const T x, int mask, int dir)
{
  T y = __shfl_xor(x, mask);
  return ((x < y) == dir) ? y : x;
}

/** Per warp, bitonic sort of 2^x elements in registers up to 32 elements using shuffle operation  .
 *
 * This function sorts the values in a warp using a bitonic sort
 * Values are passed in as a parameter and passed back.
 * Because it is intra warp no synchronisation is required.
 *
 * Each stage results in a partially sorted list in the same direction
 *
 * NOTE: I tried inlining these and the increased register use decreased performance
 *
 * @param val		The per thread element to be sorted
 * @param laneId	The lane of the thread
 * @return		The sorted values each thread returns the relevant sorted value
 */
template <typename T, const int noSorted, const int noEls>
__device__ /*inline*/ T bitonicSort3x_warp(T val, const int laneId)
{
  if ( noSorted < 2  && noEls >= 2 )  // 2   .
  {
    val = shflSwap<T>(val, 0x01, bfe(laneId, 0));
  }

  if ( noSorted < 4  && noEls >= 4 )  // 4   .
  {
    val = shflSwap<T>(val, 0x03, bfe(laneId, 1));

    val = shflSwap<T>(val, 0x01, bfe(laneId, 0));
  }

  if ( noSorted < 8  && noEls >= 8 )  // 8   .
  {
    val = shflSwap<T>(val, 0x07, bfe(laneId, 2));

    val = shflSwap<T>(val, 0x02, bfe(laneId, 1));
    val = shflSwap<T>(val, 0x01, bfe(laneId, 0));
  }

  if ( noSorted < 16 && noEls >= 16 ) // 16  .
  {
    val = shflSwap<T>(val, 0x0F, bfe(laneId, 3));

    val = shflSwap<T>(val, 0x04, bfe(laneId, 2));
    val = shflSwap<T>(val, 0x02, bfe(laneId, 1));
    val = shflSwap<T>(val, 0x01, bfe(laneId, 0));
  }

  if ( noSorted < 32 && noEls >= 32 ) // 32  .
  {
    val = shflSwap<T>(val, 0x1F, bfe(laneId, 4));

    val = shflSwap<T>(val, 0x08, bfe(laneId, 3));
    val = shflSwap<T>(val, 0x04, bfe(laneId, 2));
    val = shflSwap<T>(val, 0x02, bfe(laneId, 1));
    val = shflSwap<T>(val, 0x01, bfe(laneId, 0));
  }

  return val;
}

/** Block, Bitonic sort of power of two up to  256 elements in per thread arrays, using shuffle operation, no memory writes  .
 *
 * @param val		The per thread element to be sorted
 * @param laneId	The lane of the thread
 * @return		The noSorted values are stored val
 */
template <typename T, const int noSorted, const int noSort, const int NoArr>
__device__ /*inline*/ void bitonicSort3x_warp_regs(T* val, const int laneId)
{
  if ( noSort <= 32 )
  {
    // Sort each section less than 32
    for ( int i = 0; i < NoArr; i++)
    {
      val[i] = bitonicSort3x_warp<T, noSorted, noSort>(val[i], laneId);
    }
  }
  else
  {
    // Sort each section of 32
    for ( int i = 0; i < NoArr; i++)
    {
      val[i] = bitonicSort3x_warp<T, noSorted, 32>(val[i], laneId);
    }
  }

  if ( noSorted < 64  && noSort >= 64 )  // 64  .
  {
    FOLD // Bitonic sort  .
    {
      for ( int i = 0; i < NoArr/2; i++)
      {
	const int p1 = 0;

	FOLD // p loop
	{
	  int i0 = (i+1)*2-1-p1;	// 1
	  int i1 = i*2+p1;		// 0

	  T v0 = __shfl_xor(val[i1], 0x1F);
	  T v1 = __shfl_xor(val[i0], 0x1F);

	  val[i0] = val[i0] <  v0 ? v0 : val[i0];
	  val[i1] = val[i1] >= v1 ? v1 : val[i1];
	}
      }
    }

    FOLD // Bitonic Merge  .
    {
      for ( int i = 0; i < NoArr; i++)
      {
	val[i] = shflSwap<T>(val[i], 0xF0, bfe(laneId, 4));
	val[i] = shflSwap<T>(val[i], 0x08, bfe(laneId, 3));
	val[i] = shflSwap<T>(val[i], 0x04, bfe(laneId, 2));
	val[i] = shflSwap<T>(val[i], 0x02, bfe(laneId, 1));
	val[i] = shflSwap<T>(val[i], 0x01, bfe(laneId, 0));
      }
    }
  }

  if ( noSorted < 128 && noSort >= 128 ) // 128 .
  {
    FOLD // Bitonic sort  .
    {
      for ( int i = 0; i < NoArr/4; i++)
      {
	for ( int p1 = 0; p1 < 2; p1++)
	{
	  int i0 = (i+1)*4-1-p1;	// 3 2
	  int i1 = i*4+p1;		// 0 1

	  T v0 = __shfl_xor(val[i1], 0x1F);
	  T v1 = __shfl_xor(val[i0], 0x1F);

	  val[i0] = val[i0] < v0 ? v0 : val[i0];
	  val[i1] = val[i1] < v1 ? val[i1] : v1;
	}
      }
    }

    FOLD // Bitonic Merge  .
    {
      for ( int i = 0; i < NoArr/2; i++)
      {
	int i0 = i*2+0;
	int i1 = i*2+1;

	T v0=val[i0];
	T v1=val[i1];

	val[i0] = v0 < v1 ? v0 : v1;
	val[i1] = v0 < v1 ? v1 : v0;
      }

      for ( int i = 0; i < NoArr; i++)
      {
	val[i] = shflSwap<T>(val[i], 0xF0, bfe(laneId, 4));
	val[i] = shflSwap<T>(val[i], 0x08, bfe(laneId, 3));
	val[i] = shflSwap<T>(val[i], 0x04, bfe(laneId, 2));
	val[i] = shflSwap<T>(val[i], 0x02, bfe(laneId, 1));
	val[i] = shflSwap<T>(val[i], 0x01, bfe(laneId, 0));
      }
    }
  }

  if ( noSorted < 256 && noSort >= 256 ) // 256 .
  {
    FOLD // Bitonic sort  .
    {
      for ( int i = 0; i < NoArr/8; i++)
      {
	for ( int p1 = 0; p1 < 4; p1++)
	{
	  int i0 = (i+1)*8-1-p1;	// 7 6 5 4
	  int i1 = i*8+p1;		// 0 1 2 3

	  T v0 = __shfl_xor(val[i1], 0x1F);
	  T v1 = __shfl_xor(val[i0], 0x1F);

	  val[i0] = val[i0] < v0 ? v0 : val[i0];
	  val[i1] = val[i1] < v1 ? val[i1] : v1;
	}
      }
    }

    FOLD // Bitonic Merge  .
    {
      for ( int i = 0; i < NoArr/4; i++)
      {
	for ( int j = 0; j < 2; j++)
	{
	  int i0 = i*4+j+0; // 0 1
	  int i1 = i*4+j+2; // 2 3

	  T v0=val[i0];
	  T v1=val[i1];

	  val[i0] = v0 < v1 ? v0 : v1;
	  val[i1] = v0 < v1 ? v1 : v0;
	}
      }

      for ( int i = 0; i < NoArr/2; i++)
      {
	int i0 = i*2+0;
	int i1 = i*2+1;

	T v0=val[i0];
	T v1=val[i1];

	val[i0] = v0 < v1 ? v0 : v1;
	val[i1] = v0 < v1 ? v1 : v0;
      }

      for ( int i = 0; i < NoArr; i++)
      {
	val[i] = shflSwap<T>(val[i], 0xF0, bfe(laneId, 4));
	val[i] = shflSwap<T>(val[i], 0x08, bfe(laneId, 3));
	val[i] = shflSwap<T>(val[i], 0x04, bfe(laneId, 2));
	val[i] = shflSwap<T>(val[i], 0x02, bfe(laneId, 1));
	val[i] = shflSwap<T>(val[i], 0x01, bfe(laneId, 0));
      }
    }
  }
}

/** Block, Bitonic sort of power of two up to 8192 elements in per thread arrays (register) using as much SM as elements  .
 *
 * NB: This contains synchronisation and must be called by all threads, only the correct number of elements will be sorted
 *
 * This is an in-place bitonic sort.
 * This is very fast for small numbers of items, ie; when they can all fit in shared memory, ie < ~12K
 *
 * It has a constant performance of \f$ O\left(n\ \log^2 n \right)\f$ where n is the number of items to be sorted.
 * It requires the sort to be performed by only one block, as it requires synchronisation.
 * But this allows for the use of SM
 *
 * Each thread counts for to items in the array, as each thread performs comparisons between to elements.
 * Generally there is ~48.0 KBytes of shared memory, thus could sort up to 12288 items. However there is a
 * maximum of 1024 thread per block, thus if there are more that 2048 threads each thread must do multiple comparisons at
 * each step. These are refereed to as batches.
 *
 * @param data A pointer to an shared memory array containing elements to be sorted.
 * @param arrayLength The number of elements in the array
 * @param trdId the index of the calling thread (1 thread for 2 items in data)
 * @param noThread The number of thread that are sorting this data
 * @param dir direction to sort data ( 1 -> smallest to largest AND -1 -> largest to smallest )
 */
template <typename T, const int noSorted, const int noSort, const int NoArr>
__device__ void bitonicSort3x_regs_SM(T *data, T *val)
{
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;       /// Thread ID in block (flat index)
  const int wId = tid / 32;					/// Warp ID
  const int laneId = tid % 32;  				/// Lane ID

  if ( noSort <= 32 )
  {
    if ( wId == 0 )
    {
      // Sort each section less than 32
      for ( int a = 0; a < NoArr; a++)
      {
	val[a] = bitonicSort3x_warp<T, noSorted, noSort>(val[a], laneId);
      }

      FOLD // Write results to SM  .
      {
	for ( int a = 0; a < NoArr; a++)
	{
	  int idx = wId*32*NoArr + a*32 + laneId;
	  data[idx] = val[a];
	}
      }
    }
  }
  else
  {
    const int noInWarp = NoArr*32;
    const bool warpActive = ( wId < (noSort+noInWarp-1)/noInWarp );
    int bPos = bfind(noInWarp);

    FOLD // Full sort sections that fit in 1 warp using registers and no synchronisation  .
    {
      if( warpActive )
      {
	bitonicSort3x_warp_regs<T, noSorted, noSort, NoArr>(val, laneId);

	FOLD // Store values in SM  .
	{
#pragma unroll
	  for ( int a = 0; a < NoArr; a++)
	  {
	    int idx = wId*32*NoArr + a*32 + laneId;
	    data[idx] = val[a];
	  }
	}
      }
    }

    FOLD // Sort sections that are larger than 1 warp using SM  .
    {
      for ( int pow2 = noInWarp*2; pow2 <= noSort; pow2*=2, bPos++ )
      {
	// NB We assume all values have been written to SM and synched

	if ( pow2 > noSorted && pow2 <= noSort )
	{
	  FOLD // Bitonic sort across warps  .
	  {
	    // Compare values from SM
	    __syncthreads(); // SM Writes
	    if ( warpActive )
	    {
#pragma unroll
	      for ( int a = 0; a < NoArr; a++)
	      {
		int idx = wId*32*NoArr + a*32 + laneId;
		int bit = bfe(idx, bPos);
		int otherIdx = idx ^ (pow2-1);
		T otherV =  data[otherIdx];
		val[a] = val[a] < otherV == !bit ? val[a] : otherV;
	      }
	    }

	    // Write back to SM
	    __syncthreads(); // SM reads complete
	    if ( warpActive )
	    {
#pragma unroll
	      for ( int a = 0; a < NoArr; a++)
	      {
		int idx = wId*32*NoArr + a*32 + laneId;
		data[idx] = val[a];
	      }
	    }
	  }

	  FOLD // Bitonic Merge  .
	  {
	    FOLD // Merge sections larger than 1 warp using SM  .
	    {
	      for ( int mLen = pow2/2, bPos2 = bPos-1; mLen > noInWarp; mLen/=2, bPos2-- )
	      {
		// Compare values from SM
		__syncthreads(); // SM Writes
		if ( warpActive )
		{
#pragma unroll
		  for ( int a = 0; a < NoArr; a++)
		  {
		    int idx = wId*32*NoArr + a*32 + laneId;
		    int bit = bfe(idx, bPos2);
		    int otherIdx = idx ^ (mLen/2);

		    T otherV =  data[otherIdx];
		    val[a] = val[a] < otherV == !bit ? val[a] : otherV;
		  }
		}

		// Write back to SM
		__syncthreads(); // SM reads
		if ( warpActive )
		{
#pragma unroll
		  for ( int a = 0; a < NoArr; a++)
		  {
		    int idx = wId*32*NoArr + a*32+laneId;
		    data[idx] = val[a];
		  }
		}
	      }
	    }

	    FOLD // Merge sections in 1 warp  .
	    {
	      if ( warpActive )
	      {
		FOLD // Sections > 32 ( in warp )
		{
#pragma unroll
		  for ( int sLen = NoArr; sLen > 1; sLen/=2  )
		  {
#pragma unroll
		    for ( int i = 0; i < NoArr/sLen; i++)
		    {
#pragma unroll
		      for ( int j = 0; j < sLen/2; j++)
		      {
			int i0 = i*sLen+j+0;
			int i1 = i*sLen+j+sLen/2;

			T v0=val[i0];
			T v1=val[i1];

			val[i0] = v0 < v1 ? v0 : v1;
			val[i1] = v0 < v1 ? v1 : v0;
		      }
		    }
		  }
		}

		FOLD // Sections <= 32  .
		{
#pragma unroll
		  for ( int i = 0; i < NoArr; i++)
		  {
		    val[i] = shflSwap<T>(val[i], 0xF0, bfe(laneId, 4));
		    val[i] = shflSwap<T>(val[i], 0x08, bfe(laneId, 3));
		    val[i] = shflSwap<T>(val[i], 0x04, bfe(laneId, 2));
		    val[i] = shflSwap<T>(val[i], 0x02, bfe(laneId, 1));
		    val[i] = shflSwap<T>(val[i], 0x01, bfe(laneId, 0));
		  }
		}

		FOLD // Store values in SM  .
		{
#pragma unroll
		  for ( int a = 0; a < NoArr; a++)
		  {
		    int idx = wId*32*NoArr + a*32 + laneId;
		    data[idx] = val[a];
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  __syncthreads();  // Ensure all data is sorted before we return
}



/** Block, Bitonic sort of power of two up to 8192 elements in per thread arrays (register) using only 1024 SM elements .
 *
 * What a fucking beauty! Sort values in registers with only 1024 SM elements!
 *
 * NB: This contains synchronisation and must be called by all threads, only the correct number of elements will be sorted
 *
 * This is an in-place bitonic sort.
 * This is very fast for small numbers of items, ie; when they can all fit in shared memory, ie < ~12K
 *
 * It has a constant performance of \f$ O\left(n\ \log^2 n \right)\f$ where n is the number of items to be sorted.
 * It requires the sort to be performed by only one block, as it requires synchronisation.
 * But this allows for the use of SM
 *
 * Each thread counts for to items in the array, as each thread performs comparisons between to elements.
 * Generally there is ~48.0 KBytes of shared memory, thus could sort up to 12288 items. However there is a
 * maximum of 1024 thread per block, thus if there are more that 2048 threads each thread must do multiple comparisons at
 * each step. These are refereed to as batches.
 *
 * @param data A pointer to an shared memory array containing elements to be sorted.
 * @param arrayLength The number of elements in the array
 * @param trdId the index of the calling thread (1 thread for 2 items in data)
 * @param noThread The number of thread that are sorting this data
 * @param dir direction to sort data ( 1 -> smallest to largest AND -1 -> largest to smallest )
 */
template <typename T, const int noSorted, const int noSort, const int NoArr>
__device__ void bitonicSort3x_regs_1024(T *val)
{
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;       /// Thread ID in block (flat index)
  const int wId = tid / 32;					/// Warp ID
  const int laneId = tid % 32;					/// Lane ID 	// TODO: get from PTX register

  __shared__ T data[1024];					/// Can do exchanges with a buffer of only 1024 elements (1 per thread)

  const int noInWarp = NoArr*32;
  const bool warpActive = ( wId < (noSort+noInWarp-1)/noInWarp ) ;

  FOLD // Full sort sections that fit in 1 warp using registers and no synchronisation  .
  {
    if ( warpActive )
    {
      bitonicSort3x_warp_regs<T, noSorted, noSort, NoArr>(val, laneId);
    }
  }

  FOLD // Sort sections across warps using SM to exchange values (requires synchronisation) .
  {
#pragma unroll
    for ( int pow2 = noInWarp*2, wPos=2, bPos=0; pow2 <= noSort; pow2*=2, bPos++, wPos*=2 ) // Loop over sections to sort
    {
      // NB We assume all values have been written to SM and synched

      if ( pow2 > noSorted && pow2 <= noSort )
      {
	FOLD // Bitonic sort across warps  .
	{
	  __syncthreads(); // SM Writes

	  int bit = bfe(wId, bPos);
	  int writeIdx = wId*32 + laneId;
	  int readIdx  = (wId ^ (wPos-1)) * 32 + laneId ^ 31;
#pragma unroll
	  for ( int a = 0; a < NoArr; a++)
	  {
	    int aIdx;

	    if ( bit )
	      aIdx = (NoArr-1) ^ a;
	    else
	      aIdx = a;

	    // Each warp write one block of 32 values to SM
	    data[writeIdx] = val[aIdx];

	    __syncthreads(); // SM Writes

	    // Read comparison value from SM
	    T otherV = data[readIdx];

	    // Do the comparison and save
	    val[aIdx] = val[aIdx] < otherV == !bit ? val[aIdx] : otherV;
	  }
	}

	FOLD // Bitonic Merge  .
	{
	  FOLD // Merge sections larger than 1 warp using SM  .
	  {
#pragma unroll
	    for ( int mLen = pow2/2, bPos2 = bPos-1, wPos2 = wPos/4; mLen > noInWarp; mLen/=2, bPos2--, wPos2/=2 )
	    {
	      __syncthreads(); // SM Writes

	      int bit = bfe(wId, bPos2);
	      int writIdx = wId*32 + laneId;
	      int readIdx = ( wId ^ (wPos2)) * 32 + laneId;

#pragma unroll
	      for ( int a = 0; a < NoArr; a++)
	      {
		// Each warp write one block of 32 values to SM
		data[writIdx] = val[a];

		__syncthreads(); // SM Writes

		// Read comparison value from SM
		T otherV = data[readIdx];

		// Do the comparison and save
		val[a] = val[a] < otherV == !bit ? val[a] : otherV;
	      }
	    }
	  }

	  FOLD // Merge sections in 1 warp  .
	  {
	    FOLD // Sections > 32 ( in warp )
	    {
#pragma unroll
	      for ( int sLen = NoArr; sLen > 1; sLen/=2  )
	      {
#pragma unroll
		for ( int i = 0; i < NoArr/sLen; i++)
		{
#pragma unroll
		  for ( int j = 0; j < sLen/2; j++)
		  {
		    int i0 = i*sLen+j+0;
		    int i1 = i*sLen+j+sLen/2;

		    T v0=val[i0];
		    T v1=val[i1];

		    val[i0] = v0 < v1 ? v0 : v1;
		    val[i1] = v0 < v1 ? v1 : v0;
		  }
		}
	      }
	    }

	    FOLD // Sections <= 32  .
	    {
#pragma unroll
	      for ( int i = 0; i < NoArr; i++)
	      {
		val[i] = shflSwap<T>(val[i], 0xF0, bfe(laneId, 4));
		val[i] = shflSwap<T>(val[i], 0x08, bfe(laneId, 3));
		val[i] = shflSwap<T>(val[i], 0x04, bfe(laneId, 2));
		val[i] = shflSwap<T>(val[i], 0x02, bfe(laneId, 1));
		val[i] = shflSwap<T>(val[i], 0x01, bfe(laneId, 0));
	      }
	    }
	  }
	}
      }
    }
  }
}

/**
 *
 * NB: Assumes 8 element per thread
 */
template <typename T, int noVals, int sLen >
__device__ inline float boundsReduce1(int* offset, T *val, T *array, float *lftBound, float *rhtBound, const int tid, const int wId, const int laneId, const int printVals)
{
  const int hLen 	= sLen / 2;
  const int noTotMels	= noVals / sLen ;	// 64
  const int noMels 	= noTotMels / 32 ;	// 2
  const int NoArr 	= noVals / 1024 ;	// 4

  //int warpActive = 0;							///< Weather or not a warp is active
  float lft, rht;

  __shared__ int	belowlft;					///< SM for communication
  __shared__ int	belowRht;					///< SM for communication

  FOLD // Combine sections .
  {
    bitonicSort3x_warp_regs<float, hLen, sLen, NoArr>(val, laneId);
  }

  FOLD // Write warp medians  .
  {
    if ( laneId == hLen-1 )
    {
      for ( int m=0; m < noMels; m++)
      {
	array[wId * noMels + m ] = val[m*sLen/32];
      }
    }
  }

  FOLD // Sort median values  .
  {
    float mVals;

    __syncthreads(); // SM Writes

    FOLD // Load into registers  .
    {
      if ( wId < noMels )
      {
	int idx = wId*32 + laneId;
	mVals = array[idx];
      }
    }

    // Sort median values
    bitonicSort3x_regs_SM<float, 0, noTotMels, 1>(array, &mVals);
  }

  FOLD // Count around vals  .
  {
    int pOffset = MIN( *offset / sLen, noTotMels-2);			// Offset in sections

    int iix = 0;
    while ( iix++ < noTotMels )
    {
      lft = array[pOffset];
      rht = array[pOffset+1];

      if      ( *lftBound != NAN && rht < *lftBound )		// Check if section median is below current bound
      {
	pOffset++;
	if ( pOffset >= noTotMels - 1 )				// Use all bottoms
	{
	  if ( *lftBound == NAN )				// No bound
	  {
	    *lftBound = rht;
	  }
	  else							// Have bound
	  {
	    *lftBound = MAX(rht, *lftBound);
	  }
	  lft = rht;						// Use the last median as the pivot (lft is the pivot)

	  break;
	}
      }
      else if ( *rhtBound != NAN && lft > *rhtBound )		// Check if section median is above current bound
      {
	pOffset--;
	if ( pOffset < 0 )					// Use all tops
	{
	  if ( *rhtBound == NAN )				// No bound
	  {
	    *rhtBound = lft;
	  }
	  else							// Have bound
	  {
	    *rhtBound = MIN(lft, *rhtBound);
	  }
	  lft = *lftBound;					// Use the existing left bound as the pivot (lft is the pivot)

	  break;
	}
      }
      else							// Count actual values
      {
	if ( tid == 0 )						// Clear counts in SM
	{
	  belowlft = 0;
	  belowRht = 0;
	}
	__syncthreads(); 					// SM clears

	int cntL = 0;						// Warp specific counts
	int cntR = 0;

	for ( int a = 0; a < NoArr; a++)			// Count each threads values
	{
	  uint bitsL = __ballot((val[a] <= lft));
	  uint bitsR = __ballot((val[a] <= rht));
	  cntL += __popc(bitsL);
	  cntR += __popc(bitsR);
	}

	if ( laneId  ==0 )					// Atomic sums  .
	{
	  atomicAdd(&belowlft, cntL);
	  atomicAdd(&belowRht, cntR);
	}

	__syncthreads(); 					// Atomic SM sums  .

#ifdef SORT_DBG // DBG - Bounds New line  .
	if ( tid == 0 && printVals )
	{
	  printf("%2i %4i pOffset = %3i/%3i  Bound [%9.6f - %9.6f]  Below %4i < %9.6f -  %4i < %9.6f -  looking for %i  \n", bid, noVals, pOffset+1, noTotMels, *lftBound, *rhtBound, belowlft, lft, belowRht, rht, *offset);
	}
#endif

	if 	  ( belowlft == *offset+1 )			// Hit the actual orders stat
	{
	  return lft;
	}
	else if ( belowRht == *offset+1 )			// Hit the actual orders stat
	{
	  return rht;
	}
	else if ( belowlft <= *offset && belowRht > *offset )	// The order statistic is in these bounds
	{
	  if ( *lftBound == NAN )				// No bound
	  {
	    *lftBound = lft;
	  }
	  else							// Have bound
	  {
	    *lftBound = MAX(lft, *lftBound);
	  }

	  if ( *rhtBound == NAN )				// No bound
	  {
	    *rhtBound = rht;
	  }
	  else							// Have bound
	  {
	    *rhtBound = MIN(rht, *rhtBound);
	  }

	  break;
	}
	else if ( belowlft >  *offset )				// Shift bounds left
	{
	  pOffset--;
	  if ( pOffset < 0 )					// Quit bounds  .
	  {
	    if ( *rhtBound == NAN )				// No bound
	    {
	      *rhtBound = lft;
	    }
	    else						// Have bound
	    {
	      *rhtBound = MIN(lft, *rhtBound);
	    }
	    lft = *lftBound;					// Set the pivot

	    break;
	  }
	}
	else if ( belowRht <= *offset )				// Shift bounds right
	{
	  pOffset++;
	  if ( pOffset >= noTotMels - 1 )			// Quit bounds  .
	  {
	    if ( *lftBound == NAN )				// No bound
	    {
	      *lftBound = rht;
	    }
	    else						// Have bound
	    {
	      *lftBound = MAX(rht, *lftBound);
	    }
	    lft = rht;						// Set the pivot

	    break;
	  }
	}
      }
    }

    *offset -= (pOffset+1) * (sLen/2) ;
  }

  FOLD // Write values back to array  .
  {
    for ( int a = 0; a < noMels; a++)
    {
      float wVal = __shfl(val[a*2], 31);

      if ( wVal <= lft )
      {
	val[a] = val[a*2+1];
      }
      else
      {
	val[a] = val[a*2];
      }
    }
  }

  return NAN;
}

/** Iterative order statistic, up to 8192 elements in per thread arrays (register) using only 1024 SM elements .
 *
 * @param array
 * @param offset
 * @return
 */
template <typename T, int noEls>
__device__ float cuOrderStatPow2_radix_local(int offset, int noArrays, T *val, const int printVals)
{
#ifdef SORT_DBG
  const int bid = blockIdx.y  * gridDim.x  + blockIdx.x;        	/// Block ID (flat index)
#endif
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;       	/// Thread ID in block (flat index)
  const int wId = tid / 32;						/// Warp ID
  const int laneId = tid % 32;						/// Lane ID 	// TODO: get from PTX register
  const int noWarps = (noEls + noArrays-1) / noArrays ;			/// Number of warps in the thread block

  __shared__ int	belowlft;					///< SM for communication
  __shared__ int	belowRht;					///< SM for communication

  __shared__ T	 	array[1024];					///< SM buffer to exchange data

  const int sLen = 64 ;							///< The length to sort sections into when size is larger than 1024
  const int hLen = 32 ;							///<

  int pOffset;								///<

  T lftBound = NAN;
  T rhtBound = NAN;

  const int sortLen = 512;						///< Determine where to just sort

  if ( noEls <= sortLen)						// Just sort small sections  .
  {
    // Sort values in SM
    //bitonicSort3X_regs<T, 0, noEls>(array, *val);			// Sort using old method
    bitonicSort3x_regs_SM<float, 0, noEls, 1>(array, val);		// Sort using SM buffer
    return array[offset];
  }

#ifdef SORT_DBG // DBG New line  .
  if ( printVals && (tid == 0) && (bid==0) ) // DBG
  {
    printf("\n");
  }
#endif

  FOLD // Initial sort each section of 32 elements ( 1 warp )  .
  {
    __syncthreads(); // SM reads

    if ( wId < noWarps )
    {
      // Sort each section less than 32
      for ( int a = 0; a < noArrays; a++)
      {
	val[a] = bitonicSort3x_warp<float, 0, hLen>(val[a], laneId);
      }
    }
  }

  if ( noEls >= 8192 )
  {
    const int noVals 	= 8192;			// 8192

    float os = boundsReduce1<T, noVals, sLen>(&offset, val, array, &lftBound, &rhtBound, tid, wId, laneId, printVals);
    if ( !isnan(os) )
    {
      return os;
    }

//    const int noTotMels = noVals / sLen ;	// 128
//    const int noMels 	= noTotMels / 32 ;	// 4
//    const int NoArr 	= noVals / 1024 ;	// 8
//    float lft, rht;
//
//    FOLD // Combine sections .
//    {
//      bitonicSort3x_warp_regs<float, hLen, sLen, NoArr>(val, laneId);
//    }
//
//    FOLD // Write warp medians  .
//    {
//      if ( laneId == hLen-1 )
//      {
//	for ( int m=0; m < noMels; m++)
//	{
//	  array[wId * noMels + m ] = val[m*sLen/32];
//	}
//      }
//    }
//
//    FOLD // Sort median values  .
//    {
//      float mVals;
//
//      __syncthreads(); // SM Writes
//
//      FOLD // Load into registers  .
//      {
//	if ( wId < noMels )
//	{
//	  int idx = wId*32 + laneId;
//	  mVals = array[idx];
//	}
//      }
//
//      // Sort median values
//      bitonicSort3x_regs_SM<float, 0, noTotMels, 1>(array, &mVals);
//    }
//
//    FOLD // Count around vals  .
//    {
//      pOffset = MIN( offset / sLen, noTotMels-2);			// Offset in sections
//
//      int iix = 0;
//      while ( iix++ < noTotMels )
//      {
//	lft = array[pOffset];
//	rht = array[pOffset+1];
//
//	if      ( lftBound != NAN && rht < lftBound )			// Check if section median is below current bound
//	{
//	  pOffset++;
//	  if ( pOffset >= noTotMels - 1 )				// Use all bottoms
//	  {
//	    if ( lftBound == NAN )					// No bound
//	    {
//	      lftBound = rht;
//	    }
//	    else							// Have bound
//	    {
//	      lftBound = MAX(rht, lftBound);
//	    }
//	    lft = rht;							// Use the last median as the pivot (lft is the pivot)
//
//	    break;
//	  }
//	}
//	else if ( rhtBound != NAN && lft > rhtBound )			// Check if section median is above current bound
//	{
//	  pOffset--;
//	  if ( pOffset < 0 )						// Use all tops
//	  {
//	    if ( rhtBound == NAN )					// No bound
//	    {
//	      rhtBound = lft;
//	    }
//	    else							// Have bound
//	    {
//	      rhtBound = MIN(lft, rhtBound);
//	    }
//	    lft = lftBound;						// Use the existing left bound as the pivot (lft is the pivot)
//
//	    break;
//	  }
//	}
//	else								// Count actual values
//	{
//	  if ( tid == 0 )						// Clear counts in SM
//	  {
//	    belowlft = 0;
//	    belowRht = 0;
//	  }
//	  __syncthreads(); 						// SM clears
//
//	  int cntL = 0;							// Warp specific counts
//	  int cntR = 0;
//
//	  for ( int a = 0; a < NoArr; a++)				// Count each threads values
//	  {
//	    uint bitsL = __ballot((val[a] <= lft));
//	    uint bitsR = __ballot((val[a] <= rht));
//	    cntL += __popc(bitsL);
//	    cntR += __popc(bitsR);
//	  }
//
//	  if ( laneId  ==0 )						// Atomic sums  .
//	  {
//	    atomicAdd(&belowlft, cntL);
//	    atomicAdd(&belowRht, cntR);
//	  }
//
//	  __syncthreads(); 						// Atomic SM sums  .
//
//#ifdef SORT_DBG // DBG - Bounds New line  .
//	  if ( tid == 0 && printVals )
//	  {
//	    printf("%2i %4i pOffset = %3i/%3i  Bound [%9.6f - %9.6f]  Below %4i < %9.6f -  %4i < %9.6f -  looking for %i  \n", bid, noVals, pOffset+1, noTotMels, lftBound, rhtBound, belowlft, lft, belowRht, rht, offset);
//	  }
//#endif
//
//	  if 	  ( belowlft == offset+1 )				// Hit the actual orders stat
//	  {
//	    return lft;
//	  }
//	  else if ( belowRht == offset+1 )				// Hit the actual orders stat
//	  {
//	    return rht;
//	  }
//	  else if ( belowlft <= offset && belowRht > offset )		// The order statistic is in these bounds
//	  {
//	    if ( lftBound == NAN )					// No bound
//	    {
//	      lftBound = lft;
//	    }
//	    else							// Have bound
//	    {
//	      lftBound = MAX(lft, lftBound);
//	    }
//
//	    if ( rhtBound == NAN )					// No bound
//	    {
//	      rhtBound = rht;
//	    }
//	    else							// Have bound
//	    {
//	      rhtBound = MIN(rht, rhtBound);
//	    }
//
//	    break;
//	  }
//	  else if ( belowlft >  offset )				// Shift bounds left
//	  {
//	    pOffset--;
//	    if ( pOffset < 0 )						// Quit bounds  .
//	    {
//	      if ( rhtBound == NAN )					// No bound
//	      {
//		rhtBound = lft;
//	      }
//	      else							// Have bound
//	      {
//		rhtBound = MIN(lft, rhtBound);
//	      }
//	      lft = lftBound;						// Set the pivot
//
//	      break;
//	    }
//	  }
//	  else if ( belowRht <= offset )				// Shift bounds right
//	  {
//	    pOffset++;
//	    if ( pOffset >= noTotMels - 1 )				// Quit bounds  .
//	    {
//	      if ( lftBound == NAN )					// No bound
//	      {
//		lftBound = rht;
//	      }
//	      else							// Have bound
//	      {
//		lftBound = MAX(rht, lftBound);
//	      }
//	      lft = rht;						// Set the pivot
//
//	      break;
//	    }
//	  }
//	}
//      }
//
//      offset -= (pOffset+1) * (sLen/2) ;
//    }
//
//    FOLD // Write values back to array  .
//    {
//      for ( int a = 0; a < noMels; a++)
//      {
//	float wVal = __shfl(val[a*2], 31);
//
//	if ( wVal <= lft )
//	{
//	  val[a] = val[a*2+1];
//	}
//	else
//	{
//	  val[a] = val[a*2];
//	}
//      }
//    }

  }

  if ( noEls >= 4096 )
  {
    const int noVals 	= 4096;			// 4096

    float os = boundsReduce1<T, noVals, sLen>(&offset, val, array, &lftBound, &rhtBound, tid, wId, laneId, printVals);
    if ( !isnan(os) )
    {
      return os;
    }

//    const int noTotMels = noVals / sLen ;	// 64
//    const int noMels 	= noTotMels / 32 ;	// 2
//    const int NoArr 	= noVals / 1024 ;	// 4
//    float lft, rht;
//
//    FOLD // Combine sections .
//    {
//      bitonicSort3x_warp_regs<float, hLen, sLen, NoArr>(val, laneId);
//    }
//
//    FOLD // Write warp medians  .
//    {
//      if ( laneId == hLen-1 )
//      {
//	for ( int m=0; m < noMels; m++)
//	{
//	  array[wId * noMels + m ] = val[m*sLen/32];
//	}
//      }
//    }
//
//    FOLD // Sort median values  .
//    {
//      float mVals;
//
//      __syncthreads(); // SM Writes
//
//      FOLD // Load into registers  .
//      {
//	if ( wId < noMels )
//	{
//	  int idx = wId*32 + laneId;
//	  mVals = array[idx];
//	}
//      }
//
//      // Sort median values
//      bitonicSort3x_regs_SM<float, 0, noTotMels, 1>(array, &mVals);
//    }
//
//    FOLD // Count around vals  .
//    {
//      pOffset = MIN( offset / sLen, noTotMels-2);			// Offset in sections
//
//      int iix = 0;
//      while ( iix++ < noTotMels )
//      {
//	lft = array[pOffset];
//	rht = array[pOffset+1];
//
//	if      ( lftBound != NAN && rht < lftBound )			// Check if section median is below current bound
//	{
//	  pOffset++;
//	  if ( pOffset >= noTotMels - 1 )				// Use all bottoms
//	  {
//	    if ( lftBound == NAN )					// No bound
//	    {
//	      lftBound = rht;
//	    }
//	    else							// Have bound
//	    {
//	      lftBound = MAX(rht, lftBound);
//	    }
//	    lft = rht;							// Use the last median as the pivot (lft is the pivot)
//
//	    break;
//	  }
//	}
//	else if ( rhtBound != NAN && lft > rhtBound )			// Check if section median is above current bound
//	{
//	  pOffset--;
//	  if ( pOffset < 0 )						// Use all tops
//	  {
//	    if ( rhtBound == NAN )					// No bound
//	    {
//	      rhtBound = lft;
//	    }
//	    else							// Have bound
//	    {
//	      rhtBound = MIN(lft, rhtBound);
//	    }
//	    lft = lftBound;						// Use the existing left bound as the pivot (lft is the pivot)
//
//	    break;
//	  }
//	}
//	else								// Count actual values
//	{
//	  if ( tid == 0 )						// Clear counts in SM
//	  {
//	    belowlft = 0;
//	    belowRht = 0;
//	  }
//	  __syncthreads(); 						// SM clears
//
//	  int cntL = 0;							// Warp specific counts
//	  int cntR = 0;
//
//	  for ( int a = 0; a < NoArr; a++)				// Count each threads values
//	  {
//	    uint bitsL = __ballot((val[a] <= lft));
//	    uint bitsR = __ballot((val[a] <= rht));
//	    cntL += __popc(bitsL);
//	    cntR += __popc(bitsR);
//	  }
//
//	  if ( laneId  ==0 )						// Atomic sums  .
//	  {
//	    atomicAdd(&belowlft, cntL);
//	    atomicAdd(&belowRht, cntR);
//	  }
//
//	  __syncthreads(); 						// Atomic SM sums  .
//
//#ifdef SORT_DBG // DBG - Bounds New line  .
//	  if ( tid == 0 && printVals )
//	  {
//	    printf("%2i %4i pOffset = %3i/%3i  Bound [%9.6f - %9.6f]  Below %4i < %9.6f -  %4i < %9.6f -  looking for %i  \n", bid, noVals, pOffset+1, noTotMels, lftBound, rhtBound, belowlft, lft, belowRht, rht, offset);
//	  }
//#endif
//
//	  if 	  ( belowlft == offset+1 )				// Hit the actual orders stat
//	  {
//	    return lft;
//	  }
//	  else if ( belowRht == offset+1 )				// Hit the actual orders stat
//	  {
//	    return rht;
//	  }
//	  else if ( belowlft <= offset && belowRht > offset )		// The order statistic is in these bounds
//	  {
//	    if ( lftBound == NAN )					// No bound
//	    {
//	      lftBound = lft;
//	    }
//	    else							// Have bound
//	    {
//	      lftBound = MAX(lft, lftBound);
//	    }
//
//	    if ( rhtBound == NAN )					// No bound
//	    {
//	      rhtBound = rht;
//	    }
//	    else							// Have bound
//	    {
//	      rhtBound = MIN(rht, rhtBound);
//	    }
//
//	    break;
//	  }
//	  else if ( belowlft >  offset )				// Shift bounds left
//	  {
//	    pOffset--;
//	    if ( pOffset < 0 )						// Quit bounds  .
//	    {
//	      if ( rhtBound == NAN )					// No bound
//	      {
//		rhtBound = lft;
//	      }
//	      else							// Have bound
//	      {
//		rhtBound = MIN(lft, rhtBound);
//	      }
//	      lft = lftBound;						// Set the pivot
//
//	      break;
//	    }
//	  }
//	  else if ( belowRht <= offset )				// Shift bounds right
//	  {
//	    pOffset++;
//	    if ( pOffset >= noTotMels - 1 )				// Quit bounds  .
//	    {
//	      if ( lftBound == NAN )					// No bound
//	      {
//		lftBound = rht;
//	      }
//	      else							// Have bound
//	      {
//		lftBound = MAX(rht, lftBound);
//	      }
//	      lft = rht;						// Set the pivot
//
//	      break;
//	    }
//	  }
//	}
//      }
//
//      offset -= (pOffset+1) * (sLen/2) ;
//    }
//
//    FOLD // Write values back to array  .
//    {
//      for ( int a = 0; a < noMels; a++)
//      {
//	float wVal = __shfl(val[a*2], 31);
//
//	if ( wVal <= lft )
//	{
//	  val[a] = val[a*2+1];
//	}
//	else
//	{
//	  val[a] = val[a*2];
//	}
//      }
//    }
//

  }

  if ( noEls >= 2048 )
  {
    const int noVals 	= 2048;			// 2048

    float os = boundsReduce1<T, noVals, sLen>(&offset, val, array, &lftBound, &rhtBound, tid, wId, laneId, printVals);
    if ( !isnan(os) )
    {
      return os;
    }

//    const int noTotMels = noVals / sLen ;	// 32
//    const int noMels 	= noTotMels / 32 ;	// 1
//    const int NoArr 	= noVals / 1024 ;	// 2
//    float lft, rht;
//
//    FOLD // Combine sections .
//    {
//      bitonicSort3x_warp_regs<float, hLen, sLen, NoArr>(val, laneId);
//    }
//
//    FOLD // Write warp medians  .
//    {
//      if ( laneId == hLen-1 )
//      {
//	for ( int m=0; m < noMels; m++)
//	{
//	  array[wId * noMels + m ] = val[m*sLen/32];
//	}
//      }
//    }
//
//    FOLD // Sort median values  .
//    {
//      float mVals;
//
//      __syncthreads(); // SM Writes
//
//      FOLD // Load into registers  .
//      {
//	if ( wId < noMels )
//	{
//	  int idx = wId*32 + laneId;
//	  mVals = array[idx];
//	}
//      }
//
//      // Sort median values
//      bitonicSort3x_regs_SM<float, 0, noTotMels, 1>(array, &mVals);
//    }
//
//    FOLD // Count around vals  .
//    {
//      pOffset = MIN( offset / sLen, noTotMels-2);			// Offset in sections
//
//      int iix = 0;
//      while ( iix++ < noTotMels )
//      {
//	lft = array[pOffset];
//	rht = array[pOffset+1];
//
//	if      ( lftBound != NAN && rht < lftBound )			// Check if section median is below current bound
//	{
//	  pOffset++;
//	  if ( pOffset >= noTotMels - 1 )				// Use all bottoms
//	  {
//	    if ( lftBound == NAN )					// No bound
//	    {
//	      lftBound = rht;
//	    }
//	    else							// Have bound
//	    {
//	      lftBound = MAX(rht, lftBound);
//	    }
//	    lft = rht;							// Use the last median as the pivot (lft is the pivot)
//
//	    break;
//	  }
//	}
//	else if ( rhtBound != NAN && lft > rhtBound )			// Check if section median is above current bound
//	{
//	  pOffset--;
//	  if ( pOffset < 0 )						// Use all tops
//	  {
//	    if ( rhtBound == NAN )					// No bound
//	    {
//	      rhtBound = lft;
//	    }
//	    else							// Have bound
//	    {
//	      rhtBound = MIN(lft, rhtBound);
//	    }
//	    lft = lftBound;						// Use the existing left bound as the pivot (lft is the pivot)
//
//	    break;
//	  }
//	}
//	else								// Count actual values
//	{
//	  if ( tid == 0 )						// Clear counts in SM
//	  {
//	    belowlft = 0;
//	    belowRht = 0;
//	  }
//	  __syncthreads(); 						// SM clears
//
//	  int cntL = 0;							// Warp specific counts
//	  int cntR = 0;
//
//	  for ( int a = 0; a < NoArr; a++)				// Count each threads values
//	  {
//	    uint bitsL = __ballot((val[a] <= lft));
//	    uint bitsR = __ballot((val[a] <= rht));
//	    cntL += __popc(bitsL);
//	    cntR += __popc(bitsR);
//	  }
//
//	  if ( laneId  ==0 )						// Atomic sums  .
//	  {
//	    atomicAdd(&belowlft, cntL);
//	    atomicAdd(&belowRht, cntR);
//	  }
//
//	  __syncthreads(); 						// Atomic SM sums  .
//
//#ifdef SORT_DBG // DBG - Bounds New line  .
//	  if ( tid == 0 && printVals )
//	  {
//	    printf("%2i %4i pOffset = %3i/%3i  Bound [%9.6f - %9.6f]  Below %4i < %9.6f -  %4i < %9.6f -  looking for %i  \n", bid, noVals, pOffset+1, noTotMels, lftBound, rhtBound, belowlft, lft, belowRht, rht, offset);
//	  }
//#endif
//
//	  if 	  ( belowlft == offset+1 )				// Hit the actual orders stat
//	  {
//	    return lft;
//	  }
//	  else if ( belowRht == offset+1 )				// Hit the actual orders stat
//	  {
//	    return rht;
//	  }
//	  else if ( belowlft <= offset && belowRht > offset )		// The order statistic is in these bounds
//	  {
//	    if ( lftBound == NAN )					// No bound
//	    {
//	      lftBound = lft;
//	    }
//	    else							// Have bound
//	    {
//	      lftBound = MAX(lft, lftBound);
//	    }
//
//	    if ( rhtBound == NAN )					// No bound
//	    {
//	      rhtBound = rht;
//	    }
//	    else							// Have bound
//	    {
//	      rhtBound = MIN(rht, rhtBound);
//	    }
//
//	    break;
//	  }
//	  else if ( belowlft >  offset )				// Shift bounds left
//	  {
//	    pOffset--;
//	    if ( pOffset < 0 )						// Quit bounds  .
//	    {
//	      if ( rhtBound == NAN )					// No bound
//	      {
//		rhtBound = lft;
//	      }
//	      else							// Have bound
//	      {
//		rhtBound = MIN(lft, rhtBound);
//	      }
//	      lft = lftBound;						// Set the pivot
//
//	      break;
//	    }
//	  }
//	  else if ( belowRht <= offset )				// Shift bounds right
//	  {
//	    pOffset++;
//	    if ( pOffset >= noTotMels - 1 )				// Quit bounds  .
//	    {
//	      if ( lftBound == NAN )					// No bound
//	      {
//		lftBound = rht;
//	      }
//	      else							// Have bound
//	      {
//		lftBound = MAX(rht, lftBound);
//	      }
//	      lft = rht;						// Set the pivot
//
//	      break;
//	    }
//	  }
//	}
//      }
//
//      offset -= (pOffset+1) * (sLen/2) ;
//    }
//
//    FOLD // Write values back to array  .
//    {
//      for ( int a = 0; a < noMels; a++)
//      {
//	float wVal = __shfl(val[a*2], 31);
//
//	if ( wVal <= lft )
//	{
//	  val[a] = val[a*2+1];
//	}
//	else
//	{
//	  val[a] = val[a*2];
//	}
//      }
//    }
//
  }

  ///////// Swap over to sorts of 32 elements \\\\\\\\\\\

  if ( noEls >= 1024 )
  {
    const int noVals		= 1024;
    const int noTotMels		= noVals / 32 ;				// The number of median elements
    const bool warpActive	= ( wId < noTotMels );			// Is the current warp active
    float lft, rht;							// Pivots

    FOLD // Sort and get val  .
    {
      if ( sortLen == noVals )
      {
	FOLD // Sort remaining values  .
	{
	  bitonicSort3x_regs_SM<float, hLen, noVals, 1>(array, val);
	}

	return array[offset];
      }
    }

    FOLD // Write warp medians  .
    {
      __syncthreads(); // SM reads

      if ( warpActive && laneId == 15 )
      {
	array[wId] = val[0];
      }
    }

    FOLD // Sort median values 32  .
    {
      __syncthreads(); // SM Writes

      if ( wId == 0 )
      {
	float mVals;

	FOLD // Load into registers  .
	{
	  if ( tid < noTotMels )
	  {
	    mVals = array[tid];
	  }
	}

	// Sort median values
	mVals = bitonicSort3x_warp<float, 0, noTotMels>(mVals, laneId);

	FOLD // Store in SM  .
	{
	  if ( tid < noTotMels )
	  {
	    array[tid] = mVals;
	  }
	}
      }

      __syncthreads(); // SM Writes of sorted median elements
    }

    FOLD // Count around vals  .
    {
      pOffset = MIN( offset / 32, noTotMels-2);

      int iix = 0;
      while ( iix++ < noTotMels )
      {
	lft = array[pOffset];
	rht = array[pOffset+1];

	if      ( lftBound != NAN && rht < lftBound )			// Check if section median is below current bound
	{
	  pOffset++;
	  if ( pOffset >= noTotMels - 1 )				// Use all bottoms
	  {
	    if ( lftBound == NAN )					// No bound
	    {
	      lftBound = rht;
	    }
	    else							// Have bound
	    {
	      lftBound = MAX(rht, lftBound);
	    }
	    lft = rht;							// Use the last median as the pivot (lft is the pivot)

	    break;
	  }
	}
	else if ( rhtBound != NAN && lft > rhtBound )			// Check if section median is above current bound
	{
	  pOffset--;
	  if ( pOffset < 0 )						// Use all tops
	  {
	    if ( rhtBound == NAN )					// No bound
	    {
	      rhtBound = lft;
	    }
	    else							// Have bound
	    {
	      rhtBound = MIN(lft, rhtBound);
	    }
	    lft = lftBound;						// Use the existing left bound as the pivot (lft is the pivot)

	    break;
	  }
	}
	else								// Count actual values
	{
	  if ( tid == 0 )						// Clear counts in SM
	  {
	    belowlft = 0;
	    belowRht = 0;
	  }
	  __syncthreads(); 						// SM clears

	  int cntL = 0;							// Warp specific counts
	  int cntR = 0;

	  if ( tid < noVals )						// Count each threads values  .
	  {
	    uint bitsL = __ballot((val[0] <= lft));
	    uint bitsR = __ballot((val[0] <= rht));
	    cntL += __popc(bitsL);
	    cntR += __popc(bitsR);
	  }

	  if ( laneId  ==0 )						// Atomic sums  .
	  {
	    atomicAdd(&belowlft, cntL);
	    atomicAdd(&belowRht, cntR);
	  }

	  __syncthreads(); 						// Atomic SM sums  .

#ifdef SORT_DBG // DBG - Bounds New line  .
	  if ( tid == 0 && printVals )
	  {
	    printf("%2i %4i pOffset = %3i/%3i  Bound [%9.6f - %9.6f]  Below %4i < %9.6f -  %4i < %9.6f -  looking for %i  \n", bid, noVals, pOffset+1, noTotMels, lftBound, rhtBound, belowlft, lft, belowRht, rht, offset);
	  }
#endif

	  if 	  ( belowlft == offset+1 )				// Hit the actual orders stat
	  {
	    return lft;
	  }
	  else if ( belowRht == offset+1 )				// Hit the actual orders stat
	  {
	    return rht;
	  }
	  else if ( belowlft <= offset && belowRht > offset )		// The order statistic is in these bounds
	  {
	    if ( lftBound == NAN )					// No bound
	    {
	      lftBound = lft;
	    }
	    else							// Have bound
	    {
	      lftBound = MAX(lft, lftBound);
	    }

	    if ( rhtBound == NAN )					// No bound
	    {
	      rhtBound = rht;
	    }
	    else							// Have bound
	    {
	      rhtBound = MIN(rht, rhtBound);
	    }

	    break;
	  }
	  else if ( belowlft >  offset )				// Shift bounds left
	  {
	    pOffset--;
	    if ( pOffset < 0 )						// Quit bounds  .
	    {
	      if ( rhtBound == NAN )					// No bound
	      {
		rhtBound = lft;
	      }
	      else							// Have bound
	      {
		rhtBound = MIN(lft, rhtBound);
	      }
	      lft = lftBound;						// Set the pivot

	      break;
	    }
	  }
	  else if ( belowRht <= offset )				// Shift bounds right
	  {
	    pOffset++;
	    if ( pOffset >= noTotMels - 1 )				// Quit bounds  .
	    {
	      if ( lftBound == NAN )					// No bound
	      {
		lftBound = rht;
	      }
	      else							// Have bound
	      {
		lftBound = MAX(rht, lftBound);
	      }
	      lft = rht;						// Set the pivot

	      break;
	    }
	  }
	}
      }

      offset -= (pOffset+1) * (16) ;
    }

    FOLD // Store values in arrays  .
    {
      FOLD // Store in SM  .
      {
	if ( warpActive )
	{
	  float wVal	= __shfl(val[0], 15);				// Median of this warp
	  int aIdx 	= wId*16 + (laneId & 15) ;			// Thread specific index

	  if ( wVal <= lft )						// Store bottom half  .
	  {
	    if (laneId > 15 )
	    {
	      array[aIdx] = val[0];
	    }
	  }
	  else								// Store top half  .
	  {
	    if (laneId <= 15 )
	    {
	      array[aIdx] = val[0];
	    }
	  }
	}
      }

      __syncthreads(); // SM Writes

      FOLD // Read into register arrays  .
      {
	if ( tid < noVals/2 )
	{
	  val[0] = array[tid];

	  bitonicSort3x_warp_regs<float, hLen/2, hLen, 1>(val, laneId);
	}
      }
    }
  }

  if ( noEls >= 512  )
  {
    const int noVals		= 512;
    const int noTotMels		= noVals/32 ;				// The number of median elements
    const bool warpActive	= ( wId < noTotMels );			// Is the current warp active
    float lft, rht;							// Pivots

    FOLD // Sort and get val  .
    {
      if ( sortLen == noVals )
      {
	FOLD // Sort remaining values  .
	{
	  bitonicSort3x_regs_SM<float, hLen, noVals, 1>(array, val);
	}

	return array[offset];
      }
    }

    FOLD // Write warp medians  .
    {
      __syncthreads(); // SM reads

      if ( warpActive && laneId == 15 )
      {
	array[wId] = val[0];
      }
    }

    FOLD // Sort median values 32  .
    {
      __syncthreads(); // SM Writes

      if ( wId == 0 )
      {
	float mVals;

	FOLD // Load into registers  .
	{
	  if ( tid < noTotMels )
	  {
	    mVals = array[tid];
	  }
	}

	// Sort median values
	mVals = bitonicSort3x_warp<float, 0, noTotMels>(mVals, laneId);

	FOLD // Store in SM  .
	{
	  if ( tid < noTotMels )
	  {
	    array[tid] = mVals;
	  }
	}
      }

      __syncthreads(); // SM Writes of sorted median elements
    }

    FOLD // Count around vals  .
    {
      pOffset = MIN( offset / 32, noTotMels-2);

      int iix = 0;
      while ( iix++ < noTotMels )
      {
	lft = array[pOffset];
	rht = array[pOffset+1];

	if      ( lftBound != NAN && rht < lftBound )			// Check if section median is below current bound
	{
	  pOffset++;
	  if ( pOffset >= noTotMels - 1 )				// Use all bottoms
	  {
	    if ( lftBound == NAN )					// No bound
	    {
	      lftBound = rht;
	    }
	    else							// Have bound
	    {
	      lftBound = MAX(rht, lftBound);
	    }
	    lft = rht;							// Use the last median as the pivot (lft is the pivot)

	    break;
	  }
	}
	else if ( rhtBound != NAN && lft > rhtBound )			// Check if section median is above current bound
	{
	  pOffset--;
	  if ( pOffset < 0 )						// Use all tops
	  {
	    if ( rhtBound == NAN )					// No bound
	    {
	      rhtBound = lft;
	    }
	    else							// Have bound
	    {
	      rhtBound = MIN(lft, rhtBound);
	    }
	    lft = lftBound;						// Use the existing left bound as the pivot (lft is the pivot)

	    break;
	  }
	}
	else								// Count actual values
	{
	  if ( tid == 0 )						// Clear counts in SM
	  {
	    belowlft = 0;
	    belowRht = 0;
	  }
	  __syncthreads(); 						// SM clears

	  int cntL = 0;							// Warp specific counts
	  int cntR = 0;

	  if ( tid < noVals )						// Count each threads values  .
	  {
	    uint bitsL = __ballot((val[0] <= lft));
	    uint bitsR = __ballot((val[0] <= rht));
	    cntL += __popc(bitsL);
	    cntR += __popc(bitsR);
	  }

	  if ( laneId  ==0 )						// Atomic sums  .
	  {
	    atomicAdd(&belowlft, cntL);
	    atomicAdd(&belowRht, cntR);
	  }

	  __syncthreads(); 						// Atomic SM sums  .

#ifdef SORT_DBG // DBG - Bounds New line  .
	  if ( tid == 0 && printVals )
	  {
	    printf("%2i %4i pOffset = %3i/%3i  Bound [%9.6f - %9.6f]  Below %4i < %9.6f -  %4i < %9.6f -  looking for %i  \n", bid, noVals, pOffset+1, noTotMels, lftBound, rhtBound, belowlft, lft, belowRht, rht, offset);
	  }
#endif

	  if 	  ( belowlft == offset+1 )				// Hit the actual orders stat
	  {
	    return lft;
	  }
	  else if ( belowRht == offset+1 )				// Hit the actual orders stat
	  {
	    return rht;
	  }
	  else if ( belowlft <= offset && belowRht > offset )		// The order statistic is in these bounds
	  {
	    if ( lftBound == NAN )					// No bound
	    {
	      lftBound = lft;
	    }
	    else							// Have bound
	    {
	      lftBound = MAX(lft, lftBound);
	    }

	    if ( rhtBound == NAN )					// No bound
	    {
	      rhtBound = rht;
	    }
	    else							// Have bound
	    {
	      rhtBound = MIN(rht, rhtBound);
	    }

	    break;
	  }
	  else if ( belowlft >  offset )				// Shift bounds left
	  {
	    pOffset--;
	    if ( pOffset < 0 )						// Quit bounds  .
	    {
	      if ( rhtBound == NAN )					// No bound
	      {
		rhtBound = lft;
	      }
	      else							// Have bound
	      {
		rhtBound = MIN(lft, rhtBound);
	      }
	      lft = lftBound;						// Set the pivot

	      break;
	    }
	  }
	  else if ( belowRht <= offset )				// Shift bounds right
	  {
	    pOffset++;
	    if ( pOffset >= noTotMels - 1 )				// Quit bounds  .
	    {
	      if ( lftBound == NAN )					// No bound
	      {
		lftBound = rht;
	      }
	      else							// Have bound
	      {
		lftBound = MAX(rht, lftBound);
	      }
	      lft = rht;						// Set the pivot

	      break;
	    }
	  }
	}
      }

      offset -= (pOffset+1) * (16) ;
    }

    FOLD // Store values in arrays  .
    {
      FOLD // Store in SM  .
      {
	if ( warpActive )
	{
	  float wVal	= __shfl(val[0], 15);				// Median of this warp
	  int aIdx 	= wId*16 + (laneId & 15) ;			// Thread specific index

	  if ( wVal <= lft )						// Store bottom half  .
	  {
	    if (laneId > 15 )
	    {
	      array[aIdx] = val[0];
	    }
	  }
	  else								// Store top half  .
	  {
	    if (laneId <= 15 )
	    {
	      array[aIdx] = val[0];
	    }
	  }
	}
      }
      __syncthreads(); // SM Writes

      FOLD // Read into register arrays  .
      {
	if ( tid < noVals/2 )
	{
	  val[0] = array[tid];
	  bitonicSort3x_warp_regs<float, hLen/2, hLen, 1>(val, laneId);
	}
      }
    }
  }

  if ( noEls >= 256  )
  {
    const int noVals		= 256;
    const int noTotMels		= noVals/32 ;				// The number of median elements
    const bool warpActive	= ( wId < noTotMels );			// Is the current warp active
    float lft, rht;							// Pivots

    FOLD // Sort and get val  .
    {
      if ( sortLen == noVals )
      {
	FOLD // Sort remaining values  .
	{
	  bitonicSort3x_regs_SM<float, hLen, noVals, 1>(array, val);
	}

	return array[offset];
      }
    }

    FOLD // Write warp medians  .
    {
      __syncthreads(); // SM reads

      if ( warpActive && laneId == 15 )
      {
	array[wId] = val[0];
      }
    }

    FOLD // Sort median values 32  .
    {
      __syncthreads(); // SM Writes

      if ( wId == 0 )
      {
	float mVals;

	FOLD // Load into registers  .
	{
	  if ( tid < noTotMels )
	  {
	    mVals = array[tid];
	  }
	}

	// Sort median values
	mVals = bitonicSort3x_warp<float, 0, noTotMels>(mVals, laneId);

	FOLD // Store in SM  .
	{
	  if ( tid < noTotMels )
	  {
	    array[tid] = mVals;
	  }
	}
      }

      __syncthreads(); // SM Writes of sorted median elements
    }

    FOLD // Count around vals  .
    {
      pOffset = MIN( offset / 32, noTotMels-2);

      int iix = 0;
      while ( iix++ < noTotMels )
      {
	lft = array[pOffset];
	rht = array[pOffset+1];

	if      ( lftBound != NAN && rht < lftBound )			// Check if section median is below current bound
	{
	  pOffset++;
	  if ( pOffset >= noTotMels - 1 )				// Use all bottoms
	  {
	    if ( lftBound == NAN )					// No bound
	    {
	      lftBound = rht;
	    }
	    else							// Have bound
	    {
	      lftBound = MAX(rht, lftBound);
	    }
	    lft = rht;							// Use the last median as the pivot (lft is the pivot)

	    break;
	  }
	}
	else if ( rhtBound != NAN && lft > rhtBound )			// Check if section median is above current bound
	{
	  pOffset--;
	  if ( pOffset < 0 )						// Use all tops
	  {
	    if ( rhtBound == NAN )					// No bound
	    {
	      rhtBound = lft;
	    }
	    else							// Have bound
	    {
	      rhtBound = MIN(lft, rhtBound);
	    }
	    lft = lftBound;						// Use the existing left bound as the pivot (lft is the pivot)

	    break;
	  }
	}
	else								// Count actual values
	{
	  if ( tid == 0 )						// Clear counts in SM
	  {
	    belowlft = 0;
	    belowRht = 0;
	  }
	  __syncthreads(); 						// SM clears

	  int cntL = 0;							// Warp specific counts
	  int cntR = 0;

	  if ( tid < noVals )						// Count each threads values  .
	  {
	    uint bitsL = __ballot((val[0] <= lft));
	    uint bitsR = __ballot((val[0] <= rht));
	    cntL += __popc(bitsL);
	    cntR += __popc(bitsR);
	  }

	  if ( laneId  ==0 )						// Atomic sums  .
	  {
	    atomicAdd(&belowlft, cntL);
	    atomicAdd(&belowRht, cntR);
	  }

	  __syncthreads(); 						// Atomic SM sums  .

#ifdef SORT_DBG // DBG - Bounds New line  .
	  if ( tid == 0 && printVals )
	  {
	    printf("%2i %4i pOffset = %3i/%3i  Bound [%9.6f - %9.6f]  Below %4i < %9.6f -  %4i < %9.6f -  looking for %i  \n", bid, noVals, pOffset+1, noTotMels, lftBound, rhtBound, belowlft, lft, belowRht, rht, offset);
	  }
#endif

	  if 	  ( belowlft == offset+1 )				// Hit the actual orders stat
	  {
	    return lft;
	  }
	  else if ( belowRht == offset+1 )				// Hit the actual orders stat
	  {
	    return rht;
	  }
	  else if ( belowlft <= offset && belowRht > offset )		// The order statistic is in these bounds
	  {
	    if ( lftBound == NAN )					// No bound
	    {
	      lftBound = lft;
	    }
	    else							// Have bound
	    {
	      lftBound = MAX(lft, lftBound);
	    }

	    if ( rhtBound == NAN )					// No bound
	    {
	      rhtBound = rht;
	    }
	    else							// Have bound
	    {
	      rhtBound = MIN(rht, rhtBound);
	    }

	    break;
	  }
	  else if ( belowlft >  offset )				// Shift bounds left
	  {
	    pOffset--;
	    if ( pOffset < 0 )						// Quit bounds  .
	    {
	      if ( rhtBound == NAN )					// No bound
	      {
		rhtBound = lft;
	      }
	      else							// Have bound
	      {
		rhtBound = MIN(lft, rhtBound);
	      }
	      lft = lftBound;						// Set the pivot

	      break;
	    }
	  }
	  else if ( belowRht <= offset )				// Shift bounds right
	  {
	    pOffset++;
	    if ( pOffset >= noTotMels - 1 )				// Quit bounds  .
	    {
	      if ( lftBound == NAN )					// No bound
	      {
		lftBound = rht;
	      }
	      else							// Have bound
	      {
		lftBound = MAX(rht, lftBound);
	      }
	      lft = rht;						// Set the pivot

	      break;
	    }
	  }
	}
      }

      offset -= (pOffset+1) * (16) ;
    }

    FOLD // Store values in arrays  .
    {
      FOLD // Store in SM  .
      {
	if ( warpActive )
	{
	  float wVal	= __shfl(val[0], 15);				// Median of this warp
	  int aIdx 	= wId*16 + (laneId & 15) ;			// Thread specific index

	  if ( wVal <= lft )						// Store bottom half  .
	  {
	    if (laneId > 15 )
	    {
	      array[aIdx] = val[0];
	    }
	  }
	  else								// Store top half  .
	  {
	    if (laneId <= 15 )
	    {
	      array[aIdx] = val[0];
	    }
	  }
	}
      }
      __syncthreads(); // SM Writes

      FOLD // Read into register arrays  .
      {
	if ( tid < noVals/2 )
	{
	  val[0] = array[tid];
	  bitonicSort3x_warp_regs<float, hLen/2, hLen, 1>(val, laneId);
	}
      }
    }
  }

  if ( noEls >= 128  )
  {
    const int noVals = 128;

    FOLD // Sort remaining values  .
    {
      bitonicSort3x_regs_SM<float, hLen, noVals, 1>(array, val);
    }

    return array[offset];
  }

  return NAN; // This should not be reached
}

/**
 *
 * NB: Assumes 8 element per thread
 */
template <typename T, int hLen, int sLen, int noVals, int noTotMels>
__device__ inline float boundsReduce2(int* offset, T *val, T* array, float *lftBound, float * rhtBound, const int tid, const int wId, const int laneId, const int printVals)
{
  int warpActive = 0;							///< Weather or not a warp is active
  float lft, rht;

  __shared__ int	belowlft;					///< SM for communication
  __shared__ int	belowRht;					///< SM for communication

#ifdef SORT_DBG // DBG New line  .
  if ( tid == 0 )
  {
    printf( "Enter %4i %i %i - %i \n", noVals, sLen, hLen, noTotMels );
  }
  __syncthreads();
#endif

  FOLD // Set active warps  .
  {
    if ( wId < noTotMels )
    {
      warpActive = 1;
    }
  }

  FOLD // Combine sections .
  {
    if ( warpActive )
    {
      bitonicSort3x_warp_regs<T, hLen, sLen, 8>(val, laneId);
    }
  }

  FOLD // Write warp medians  .
  {
    if ( warpActive && laneId == 31 )					// Load into SM  .
    {
      array[wId] = val[3];
    }
  }

  FOLD // Sort median values  .
  {

    float mVals;

    __syncthreads(); // SM Writes

    if ( wId == 0 )
    {
      FOLD // Load into registers  .
      {
	if ( laneId < noTotMels )
	{
	  mVals = array[laneId];
	}
      }

      // Sort median values
      mVals = bitonicSort3x_warp<T, 0, noTotMels>(mVals, laneId);

      FOLD // Save into SM  .
      {
	if ( laneId < noTotMels )
	{
	  array[laneId] = mVals;
	}
      }
    }

    __syncthreads(); // SM Writes

  }

  FOLD // Count around vals  .
  {
    int pOffset = MIN( *offset / sLen, noTotMels-2);			// Offset in sections

    int iix = 0;
    while ( iix++ < noTotMels )
    {
      lft = array[pOffset];
      rht = array[pOffset+1];

      if      ( *lftBound != NAN && rht < *lftBound )			// Check if section median is below current bound
      {
	pOffset++;
	if ( pOffset >= noTotMels - 1 )					// Use all bottoms
	{
	  if ( *lftBound == NAN )					// No bound
	  {
	    *lftBound = rht;
	  }
	  else								// Have bound
	  {
	    *lftBound = MAX(rht, *lftBound);
	  }
	  lft = rht;							// Use the last median as the pivot (lft is the pivot)

	  break;
	}
      }
      else if ( *rhtBound != NAN && lft > *rhtBound )			// Check if section median is above current bound
      {
	pOffset--;
	if ( pOffset < 0 )						// Use all tops
	{
	  if ( *rhtBound == NAN )					// No bound
	  {
	    *rhtBound = lft;
	  }
	  else								// Have bound
	  {
	    *rhtBound = MIN(lft, *rhtBound);
	  }
	  lft = *lftBound;						// Use the existing left bound as the pivot (lft is the pivot)

	  break;
	}
      }
      else								// Count actual values
      {
	if ( tid == 0 )							// Clear counts in SM
	{
	  belowlft = 0;
	  belowRht = 0;
	}

	__syncthreads(); 						// SM clears

	if (warpActive)							// Count values
	{
	  int cntL = 0;							// Warp specific counts
	  int cntR = 0;

	  for ( int a = 0; a < 8; a++)					// Count each threads values
	  {
	    uint bitsL = __ballot((val[a] <= lft));
	    uint bitsR = __ballot((val[a] <= rht));
	    cntL += __popc(bitsL);
	    cntR += __popc(bitsR);
	  }

	  if ( laneId  == 0 )						// Atomic sums  .
	  {
	    atomicAdd(&belowlft, cntL);
	    atomicAdd(&belowRht, cntR);
	  }
	}

	__syncthreads(); 						// Atomic SM sums  .

#ifdef SORT_DBG // DBG - Bounds New line  .
	if ( tid == 0 && printVals )
	{
	  const int bid = blockIdx.y  * gridDim.x  + blockIdx.x;        /// Block ID (flat index)
	  printf("%2i %4i pOffset = %3i/%3i  Bound [%9.6f - %9.6f]  Below %4i < %9.6f -  %4i < %9.6f -  looking for %i  \n", bid, noVals, pOffset+1, noTotMels, *lftBound, *rhtBound, belowlft, lft, belowRht, rht, *offset);
	}
#endif

	if 	( belowlft == *offset+1 )				// Hit the actual orders stat
	{
	  return lft;
	}
	else if ( belowRht == *offset+1 )				// Hit the actual orders stat
	{
	  return rht;
	}
	else if ( belowlft <= *offset && belowRht > *offset )		// The order statistic is in these bounds
	{
	  if ( *lftBound == NAN )					// No bound
	  {
	    *lftBound = lft;
	  }
	  else								// Have bound
	  {
	    *lftBound = MAX(lft, *lftBound);
	  }

	  if ( *rhtBound == NAN )					// No bound
	  {
	    *rhtBound = rht;
	  }
	  else								// Have bound
	  {
	    *rhtBound = MIN(rht, *rhtBound);
	  }

	  break;
	}
	else if ( belowlft >  *offset )					// Shift bounds left
	{
	  pOffset--;
	  if ( pOffset < 0 )						// Quit bounds  .
	  {
	    if ( *rhtBound == NAN )					// No bound
	    {
	      *rhtBound = lft;
	    }
	    else							// Have bound
	    {
	      *rhtBound = MIN(lft, *rhtBound);
	    }
	    lft = *lftBound;						// Set the pivot

	    break;
	  }
	}
	else if ( belowRht <= *offset )					// Shift bounds right
	{
	  pOffset++;
	  if ( pOffset >= noTotMels - 1 )				// Quit bounds  .
	  {
	    if ( *lftBound == NAN )					// No bound
	    {
	      *lftBound = rht;
	    }
	    else							// Have bound
	    {
	      *lftBound = MAX(rht, *lftBound);
	    }
	    lft = rht;							// Set the pivot

	    break;
	  }
	}
      }
    }

    *offset -= (pOffset+1) * (hLen) ;
  }

  FOLD // Write values back to array  .
  {
    float wVal = __shfl(val[3], 31);

    for ( int a = 0; a < 4; a++)
    {
      if ( warpActive && wId >= noTotMels/2 )
      {
	int keep = 0;
	int excl = 0;

	if ( wVal <= lft )
	{
	  // Keep second half so
	  keep = 4;
	}
	else
	{
	  // Exclude second half so
	  excl = 4;
	}

	// Move second half to first half
	//for ( int a = 0; a < 4; a++)
	{
	  int idx = (wId-noTotMels/2)*32 + laneId;
	  array[idx] = val[keep+a];
	}
      }

      __syncthreads(); 							// SM writes  .

      if ( warpActive && wId < noTotMels/2 )
      {
	int keep = 0;
	int excl = 0;

	if ( wVal <= lft )
	{
	  // Keep second half so
	  keep = 4;
	}
	else
	{
	  // Exclude second half so
	  excl = 4;
	}

	// Move second half to first half
	//for ( int a = 0; a < 4; a++)
	{
	  int idx = (wId)*32 + laneId;
	  val[excl+a] = array[idx];
	}
      }
    }

    __syncthreads(); 							// SM reads (this may be unnecessary)  .
  }

#ifdef SORT_DBG // DBG New line  .
  if ( tid == 0 )
  {
    printf( "Exit  %4i %i %i \n\n", noVals, sLen, hLen);
  }
#endif

  return NAN;
}

/** Iterative order statistic, one value per thread maximum of 8192 values  .
 *
 * @param array
 * @param offset
 * @return
 */
template <typename T, int noEls>
__device__ float cuOrderStatPow2_radix_local_warps(int offset, T *val, const int printVals)
{
  const int bid = blockIdx.y  * gridDim.x  + blockIdx.x;		/// Block ID (flat index)
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;		/// Thread ID in block (flat index)
  const int wId = tid / 32;						/// Warp ID
  const int laneId = tid % 32;						/// Lane ID

  __shared__ T	 	array[1024];					///< SM buffer to exchange data

  const int hLen = 128 ;						///<
  const int sLen = hLen*2 ;						///< The length to sort sections into when size is larger than 1024
  const int noWarps = (noEls ) / 256 ;					/// Number of warps in the thread block

  T lftBound = NAN;
  T rhtBound = NAN;

  const int sortLen = 128;						///< Determine where to just sort

  if ( noEls <= sortLen)						// Just sort small sections  .
  {
    // Sort values in SM
    //bitonicSort3X_regs<T, 0, noEls>(array, *val);			// Sort using old method
    bitonicSort3x_regs_SM<float, 0, noEls, 1>(array, val);		// Sort using new SM buffer
    return array[offset];
  }

#ifdef SORT_DBG
  if ( printVals && (tid == 0) && (bid==0) )
  {
    printf("\n");
  }
#endif

  FOLD // Initial sort each section of hLen elements ( 1 warp )  .
  {
    __syncthreads(); // SM reads

    if ( wId < noWarps )
    {
      bitonicSort3x_warp_regs<T, 0, hLen, 8>(val, laneId);
      //bitonicSort3x_warp_regs<T, 0, 32, 8>(val, laneId);
    }
  }

  if ( noEls >= 8192 )
  {
    const int noVals 	= 8192;
    const int noTotMels = noVals / sLen ;				// 32 - The number of sections and thus median values

    float os = boundsReduce2<T, hLen, sLen, noVals, noTotMels>(&offset, val, array, &lftBound, &rhtBound, tid, wId, laneId, printVals);
    if ( !isnan(os) )
    {
      return os;
    }
  }

  if ( noEls >= 4096 )
  {
    const int noVals 	= 4096;
    const int noTotMels = noVals / sLen ;				// 16 - The number of sections and thus median values

    float os = boundsReduce2<T, hLen, sLen, noVals, noTotMels>(&offset, val, array, &lftBound, &rhtBound, tid, wId, laneId, printVals);
    if ( !isnan(os) )
    {
      return os;
    }
  }

  if ( noEls >= 2048 )
  {
    const int noVals 	= 2048;
    const int noTotMels		= noVals / sLen ;			// The number of median elements

    float os = boundsReduce2<T, hLen, sLen, noVals, noTotMels>(&offset, val, array, &lftBound, &rhtBound, tid, wId, laneId, printVals);
    if ( !isnan(os) )
    {
      return os;
    }
  }

  if ( noEls >= 1024 )
  {
    const int noVals		= 1024;
    const int noTotMels		= noVals / sLen ;			// The number of median elements

    FOLD // Sort and get val  .
    {
      if ( sortLen == noVals )
      {
	FOLD // Sort remaining values  .
	{
	  bitonicSort3x_regs_SM<float, hLen, noVals, 8>(array, val);
	}

	return array[offset];
      }
    }

    float os = boundsReduce2<T, hLen, sLen, noVals, noTotMels>(&offset, val, array, &lftBound, &rhtBound, tid, wId, laneId, printVals);
    if ( !isnan(os) )
    {
      return os;
    }
  }

  if ( noEls >= 512  )
  {
    const int noVals		= 512;
    const int noTotMels		= noVals / sLen ;			// The number of median elements

    FOLD // Sort and get val  .
    {
      if ( sortLen == noVals )
      {
	FOLD // Sort remaining values  .
	{
	  bitonicSort3x_regs_SM<float, hLen, noVals, 8>(array, val);
	}

	return getValue<float, 8>(val, offset );
      }
    }

    float os = boundsReduce2<T, hLen, sLen, noVals, noTotMels>(&offset, val, array, &lftBound, &rhtBound, tid, wId, laneId, printVals);
    if ( !isnan(os) )
    {
      return os;
    }
  }

  if ( noEls >= 256  )
  {
    const int noVals		= 256;
    const int noTotMels		= noVals / sLen ;			// The number of median elements

    FOLD // Sort and get val  .
    {
      if ( sortLen == noVals )
      {
	FOLD // Sort remaining values  .
	{
	  bitonicSort3x_regs_SM<float, hLen, noVals, 8>(array, val);
	}

	return array[offset];
      }
    }

    float os = boundsReduce2<T, hLen, sLen, noVals, noTotMels>(&offset, val, array, &lftBound, &rhtBound, tid, wId, laneId, printVals);
    if ( !isnan(os) )
    {
      return os;
    }
  }

  if ( noEls >= 128  )
  {
    const int noVals = 128;

    FOLD // Sort remaining values  .
    {
      bitonicSort3x_regs_SM<float, hLen, noVals, 8>(array, val);
    }

    return array[offset];
  }

  return NAN; // This should not be reached
}


////  Public sort functions  \\\\\\

template <typename T, int noEls>
__device__ void bitonicSort_mem(T *data)
{
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;       	/// Thread ID in block (flat index)
  const int wId = tid / 32;						/// Warp ID
  const int laneId = tid % 32;						/// Lane ID

//  if ( noEls <= 1024 ) // Can have each thread handle one value
//  {
//    T val = 0 ;
//
//    if ( tid < noEls ) // Read values into register
//    {
//      val = data[tid];
//    }
//    __syncthreads(); 							// SM reads
//
//    //bitonicSort3X_regs<T, 0, noEls>(data, val);			// Old method
//    bitonicSort3x_regs_SM<T, 0, noEls, 1>(data, &val);			// New method
//  }
//  else
  {
    const int NoArr = (noEls + 1023) / 1024.0 ;				// int round up done at compile time ;)

    T val[NoArr]; // Registers

    FOLD // Read values into registers
    {
#pragma unroll
      for ( int a = 0; a < NoArr; a++)
      {
	int idx = wId*32*NoArr + a*32 + laneId;

	if ( idx < noEls )
	{
	  val[a] = data[idx];
	}
      }
      __syncthreads(); 							// SM reads
    }

    bitonicSort3x_regs_SM<T, 0, noEls, NoArr>(data, val);
  }
}

/** Block, Bitonic sort of power of two up to 8192 elements in per thread arrays (register) using as much SM as elements  .
 *
 * NB: This contains synchronisation and must be called by all threads, only the correct number of elements will be sorted
 *
 *
 * @param data A pointer to an shared memory array containing elements to be sorted.
 */
template <typename T, const int noSort>
__device__ void bitonicSort_SM(T *data )
{
  const int tid 	= threadIdx.y * blockDim.x + threadIdx.x;       /// Thread ID in block (flat index)
  const int wId 	= tid / 32;					/// Warp ID
  const int laneId 	= tid % 32;  					/// Lane ID
  const int NoArr 	= (noSort + 1023) / 1024.0 ;			// int round up done at compile time ;)
  T val[NoArr];

  FOLD // Read values into registers  .
  {
#pragma unroll
    for ( int a = 0; a < NoArr; a++)
    {
	int idx = wId*32*NoArr + a*32 + laneId;

	if ( idx < noSort )
	{
	  val[a] = data[idx];
	}
    }
  }

  if ( noSort <= 32 )							// Sort by calling sort function
  {
    if ( wId == 0 )
    {
      // Sort each section less than 32
#pragma unroll
      for ( int a = 0; a < NoArr; a++)
      {
	val[a] = bitonicSort3x_warp<T, 0, noSort>(val[a], laneId);
      }

      FOLD // Write results to SM  .
      {
#pragma unroll
	for ( int a = 0; a < NoArr; a++)
	{
	  int idx = wId*32*NoArr + a*32 + laneId;
	  data[idx] = val[a];
	}
      }
    }
  }
  else									// Sort with bitonic sort
  {
    //bitonicSort3x_regs_SM<T, 0, noSort, NoArr>(data, val);		// NOTE: Could use the function but I found that slower than manually including the code

    const int noInWarp 		= NoArr*32;
    const bool warpActive 	= ( wId < (noSort+noInWarp-1)/noInWarp );
    int bPos 			= bfind(noInWarp);

    FOLD // Full sort sections that fit in 1 warp using registers and no synchronisation  .
    {
      if ( warpActive )
      {
	bitonicSort3x_warp_regs<T, 0, noSort, NoArr>(val, laneId);

	FOLD // Store values in SM  .
	{
#pragma unroll
	  for ( int a = 0; a < NoArr; a++)
	  {
	    int idx = wId*32*NoArr + a*32 + laneId;
	    data[idx] = val[a];
	  }
	}
      }
    }

    FOLD // Sort sections that are larger than 1 warp using SM some synchronisation  .
    {
      for ( int pow2 = noInWarp*2; pow2 <= noSort; pow2*=2, bPos++ )
      {
	// NB We assume all values have been written to SM

	FOLD // Bitonic sort across warps  .
	{
	  FOLD // Get values from SM and compare  .
	  {
	    __syncthreads();						// SM Writes
	    if ( warpActive )
	    {
#pragma unroll
	      for ( int a = 0; a < NoArr; a++)
	      {
		int idx = wId*32*NoArr + a*32 + laneId;
		int bit = bfe(idx, bPos);
		int otherIdx = idx ^ (pow2-1);
		T otherV =  data[otherIdx];
		val[a] = val[a] < otherV == !bit ? val[a] : otherV;
	      }
	    }
	  }

	  FOLD // Write back to SM  .
	  {
	    __syncthreads();						// SM reads complete
	    if ( warpActive )
	    {
#pragma unroll
	      for ( int a = 0; a < NoArr; a++)
	      {
		int idx = wId*32*NoArr + a*32 + laneId;
		data[idx] = val[a];
	      }
	    }
	  }
	}

	FOLD // Bitonic Merge  .
	{
	  FOLD // Merge sections larger than 1 warp using SM  .
	  {
	    for ( int mLen = pow2/2, bPos2 = bPos-1; mLen > noInWarp; mLen/=2, bPos2-- )
	    {
	      // Compare values from SM
	      __syncthreads(); // SM Writes
	      if ( warpActive )
	      {
#pragma unroll
		for ( int a = 0; a < NoArr; a++)
		{
		  int idx = wId*32*NoArr + a*32 + laneId;
		  int bit = bfe(idx, bPos2);
		  int otherIdx = idx ^ (mLen/2);

		  T otherV =  data[otherIdx];
		  val[a] = val[a] < otherV == !bit ? val[a] : otherV;
		}
	      }

	      // Write back to SM
	      __syncthreads(); // SM reads
	      if ( warpActive )
	      {
#pragma unroll
		for ( int a = 0; a < NoArr; a++)
		{
		  int idx = wId*32*NoArr + a*32+laneId;
		  data[idx] = val[a];
		}
	      }
	    }
	  }

	  FOLD // Merge sections in 1 warp  .
	  {
	    if ( warpActive )
	    {
	      FOLD // Sections > 32 ( in warp )
	      {
#pragma unroll
		for ( int sLen = NoArr; sLen > 1; sLen/=2  )
		{
#pragma unroll
		  for ( int i = 0; i < NoArr/sLen; i++)
		  {
#pragma unroll
		    for ( int j = 0; j < sLen/2; j++)
		    {
		      int i0 = i*sLen+j+0;
		      int i1 = i*sLen+j+sLen/2;

		      T v0=val[i0];
		      T v1=val[i1];

		      val[i0] = v0 < v1 ? v0 : v1;
		      val[i1] = v0 < v1 ? v1 : v0;
		    }
		  }
		}
	      }

	      FOLD // Sections <= 32  .
	      {
#pragma unroll
		for ( int i = 0; i < NoArr; i++)
		{
		  val[i] = shflSwap<T>(val[i], 0xF0, bfe(laneId, 4));
		  val[i] = shflSwap<T>(val[i], 0x08, bfe(laneId, 3));
		  val[i] = shflSwap<T>(val[i], 0x04, bfe(laneId, 2));
		  val[i] = shflSwap<T>(val[i], 0x02, bfe(laneId, 1));
		  val[i] = shflSwap<T>(val[i], 0x01, bfe(laneId, 0));
		}
	      }

	      FOLD // Store values in SM  .
	      {
#pragma unroll
		for ( int a = 0; a < NoArr; a++)
		{
		  int idx = wId*32*NoArr + a*32 + laneId;
		  data[idx] = val[a];
		}
	      }
	    }
	  }
	}
      }
    }
  }

  __syncthreads();  // Ensure all data is written to SM before we return
}

template <typename T, int noEls, int noArr>
__device__ void bitonicSort_reg(T *val)
{
  //bitonicSort3x_regs_SM<T, 0, noEls, noArr>(val);
  bitonicSort3x_regs_1024<T, 0, noEls, noArr>(val);
}


////   Old Non kepler code   \\\\\\

///** In-place Bitonic sort a float array.
// *
// * This is an in-place bitonic sort.
// * This is very fast for small numbers of items, ie; when they can all fit in shared memory, ie < ~12K
// *
// * It has a constant performance of \f$ O\left(n\ \log^2 n \right)\f$ where n is the number of items to be noSorted.
// * It requires the sort to be performed by only one block, as it requires synchronisation.
// * But this allows for the use of SM
// *
// * Each thread counts for two items in the array, as each thread performs comparisons between to elements.
// * Generally there is ~48.0 KBytes of shared memory, thus could sort up to 12288 items. However there is a
// * maximum of 1024 thread per block, thus if there are more that 2048 threads each thread must do multiple comparisons at
// * each step. These are refereed to as batches.
// *
// * @param data A pointer to an shared memory array containing elements to be noSorted.
// * @param arrayLength The number of elements in the array
// * @param trdId the index of the calling thread (1 thread for 2 items in data)
// * @param noThread The number of thread that are sorting this data
// * @param dir direction to sort data ( 1 -> smallest to largest AND -1 -> largest to smallest )
// */
//__device__ void bitonicSort(float *data, const uint arrayLength, const uint trdId, const uint noThread, const int dir )
//{
//  const uint noBatch = ceilf(arrayLength / 2.0f / noThread);     // Number of comparisons each thread must do
//  uint idx;                               // The index including batch adjustment
//  const uint max = arrayLength * 2;       // The maximum distance a thread could compare
//  uint bIdx;                              // The thread position in the block
//  uint hSz = 1;                           // half block size
//  uint pos1, pos2, blk;                   // index of points to be compared
//  uint len;                               // The distance between items to swap
//  uint bach;                              // The batch we are processing
//  uint shift = 32;                        // Amount to bitshift by to calculate remainders
//  uint shift2;
//  uint hsl1;
//
//  // Incrementally sort blocks of 2 then 4 then 8 ... items
//  for (uint size = 2; size < max; size <<= 1, shift--)
//  {
//    hSz = (size >> 1);
//    hsl1 = hSz - 1;
//
//    __syncthreads();
//
//    // Bitonic sort, two Bitonic noSorted list into Bitonic list
//    for (bach = 0; bach < noBatch; bach++)
//    {
//      idx = (trdId + bach * noThread);
//
//      //bIdx = hSz - 1 - idx % hSz;
//      //bIdx = hsl1 - (idx << shift) >> shift;  // My method
//      bIdx = hsl1 - idx & (hSz - 1);// x mod y == x & (y-1), where y is 2^n.
//
//      blk = idx / hSz;
//
//      len = size - 1 - bIdx * 2;
//      pos1 = blk * size + bIdx;
//      pos2 = pos1 + len;
//
//      if (pos2 < arrayLength)
//	Comparator(data[pos1], data[pos2], dir);
//    }
//
//    // Bitonic Merge
//    for (len = (hSz >>= 1), shift2 = shift + 1; len > 0; len >>= 1, shift2++)
//    {
//      hSz = (len << 1);
//
//      __syncthreads();
//      for (bach = 0; bach < noBatch; bach++)
//      {
//	idx = (trdId + bach * noThread);
//
//	//bIdx  = idx % len;
//	//bIdx = (idx << shift2) >> shift2;
//	bIdx = idx & (len - 1);// x mod y == x & (y-1), where y is 2^n.
//
//	blk = idx / len;
//
//	pos1 = blk * hSz + bIdx;
//	pos2 = pos1 + len;
//
//	if (pos2 < arrayLength)
//	  Comparator(data[pos1], data[pos2], dir);
//      }
//    }
//  }
//
//  __syncthreads();  // Ensure all data is noSorted before we return
//}
//
///** In-place Bitonic sort a float array.
// *
// * This is an in-place bitonic sort.
// * This is very fast for small numbers of items, ie; when they can all fit in shared memory, ie < ~12K
// *
// * It has a constant performance of \f$ O\left(n\ \log^2 n \right)\f$ where n is the number of items to be sorted.
// * It requires the sort to be performed by only one block, as it requires synchronisation.
// * But this allows for the use of SM
// *
// * Each thread counts for two items in the array, as each thread performs comparisons between to elements.
// * Generally there is ~48.0 KBytes of shared memory, thus could sort up to 12288 items. However there is a
// * maximum of 1024 thread per block, thus if there are more that 2048 threads each thread must do multiple comparisons at
// * each step. These are refereed to as batches.
// *
// * @param data A pointer to an shared memory array containing elements to be sorted.
// * @param arrayLength The number of elements in the array
// * @param trdId the index of the calling thread (1 thread for 2 items in data)
// * @param noThread The number of thread that are sorting this data
// * @param dir direction to sort data ( 1 -> smallest to largest AND -1 -> largest to smallest )
// *
// */
//__device__ void bitonicSort1Warp(float *data, const uint arrayLength, const uint trdId, const uint noThread, const int dir )
//{
//  const uint noBatch = ceilf(arrayLength / 2.0f / noThread);     // Number of comparisons each thread must do
//  uint idx;                               // The index including batch adjustment
//  const uint max = arrayLength * 2;       // The maximum distance a thread could compare
//  uint bIdx;                              // The thread position in the block
//  uint hSz = 1;                           // half block size
//  uint pos1, pos2, blk;                   // index of points to be compared
//  uint len;                               // The distance between items to swap
//  uint bach;                              // The batch we are processing
//  uint shift = 32;                        // Amount to bitshift by to calculate remainders
//  uint shift2;
//  uint hsl1;
//
//  // Incrementally sort blocks of 2 then 4 then 8 ... items
//  for (uint size = 2; size < max; size <<= 1, shift--)
//  {
//    hSz = (size >> 1);
//    hsl1 = hSz - 1;
//
//    // Bitonic sort, two Bitonic noSorted list into Bitonic list
//    for (bach = 0; bach < noBatch; bach++)
//    {
//      idx = (trdId + bach * noThread);
//
//      //bIdx = hSz - 1 - idx % hSz;
//      //bIdx = hsl1 - (idx << shift) >> shift;  // My method
//      bIdx = hsl1 - idx & (hSz - 1);// x mod y == x & (y-1), where y is 2^n.
//
//      blk = idx / hSz;
//
//      len = size - 1 - bIdx * 2;
//      pos1 = blk * size + bIdx;
//      pos2 = pos1 + len;
//
//      if (pos2 < arrayLength)
//	Comparator(data[pos1], data[pos2], dir);
//    }
//
//    // Bitonic Merge
//    for (len = (hSz >>= 1), shift2 = shift + 1; len > 0; len >>= 1, shift2++)
//    {
//      hSz = (len << 1);
//
//      for (bach = 0; bach < noBatch; bach++)
//      {
//	idx = (trdId + bach * noThread);
//
//	//bIdx  = idx % len;
//	//bIdx = (idx << shift2) >> shift2;
//	bIdx = idx & (len - 1);// x mod y == x & (y-1), where y is 2^n.
//
//	blk = idx / len;
//
//	pos1 = blk * hSz + bIdx;
//	pos2 = pos1 + len;
//
//	if (pos2 < arrayLength)
//	  Comparator(data[pos1], data[pos2], dir);
//      }
//    }
//  }
//}


////  Public order statistic \\\\\\


/**
 *
 * @param val
 * @param os
 * @return
 */
template <typename T, int noEls, int noArr>
__device__ T cuOrderStatPow2_sort(T *val, int offset)
{
  bitonicSort3x_regs_1024<T, 0, noEls, noArr>(val);
  return getValue<T, noArr>(val, offset);
}

template <typename T, int noEls>
__device__ void cuOrderStatPow2_sort_SM(T *data, int os)
{
  //const int NoArr = (noEls + 1023) / 1024.0 ; // int round up ;)

  //bitonicSort3x_regs_SM<T, 0, noEls, NoArr>(data, val);
  //return data[os];

  //bitonicSort_mem<T, noEls>(data);
  //return data[os];

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;       	/// Thread ID in block (flat index)
  const int wId = tid / 32;						/// Warp ID
  const int laneId = tid % 32;						/// Lane ID
  const int NoArr = (noEls + 1023) / 1024.0 ;				// Int round up done at compile time ;)

  T val[NoArr]; // Registers

  FOLD // Read values into registers
  {
#pragma unroll
    for ( int a = 0; a < NoArr; a++)
    {
      int idx = wId*32*NoArr + a*32 + laneId;

      if ( idx < noEls )
      {
	val[a] = data[idx];
      }
    }
    __syncthreads(); 							// SM reads
  }

  bitonicSort3x_regs_SM<T, 0, noEls, NoArr>(data, val);

  //return data[os];
}

/** Generic order statistic function
 *
 * @param val
 * @param offset
 * @return
 */
template< int noEls >
__device__ float cuOrderStatPow2_radix(float *val , int offset, int printVals)
{
  const int NoArr = (noEls + 1023) / 1024.0 ; // int round up ;)

  return cuOrderStatPow2_radix_local<float, noEls>(offset, NoArr, val, printVals);
  //return cuOrderStatPow2_radix_local_warps<float, noEls>(offset, val, printVals);
}


template __device__ void bitonicSort_mem<float, 2   >(float *data);
template __device__ void bitonicSort_mem<float, 4   >(float *data);
template __device__ void bitonicSort_mem<float, 8   >(float *data);
template __device__ void bitonicSort_mem<float, 16  >(float *data);
template __device__ void bitonicSort_mem<float, 32  >(float *data);
template __device__ void bitonicSort_mem<float, 64  >(float *data);
template __device__ void bitonicSort_mem<float, 128 >(float *data);
template __device__ void bitonicSort_mem<float, 256 >(float *data);
template __device__ void bitonicSort_mem<float, 512 >(float *data);
template __device__ void bitonicSort_mem<float, 1024>(float *data);
template __device__ void bitonicSort_mem<float, 2048>(float *data);
template __device__ void bitonicSort_mem<float, 4096>(float *data);
template __device__ void bitonicSort_mem<float, 8192>(float *data);

template __device__ void bitonicSort_reg<float, 2,    1>(float *val);
template __device__ void bitonicSort_reg<float, 4,    1>(float *val);
template __device__ void bitonicSort_reg<float, 8,    1>(float *val);
template __device__ void bitonicSort_reg<float, 16,   1>(float *val);
template __device__ void bitonicSort_reg<float, 32,   1>(float *val);
template __device__ void bitonicSort_reg<float, 64,   1>(float *val);
template __device__ void bitonicSort_reg<float, 128,  1>(float *val);
template __device__ void bitonicSort_reg<float, 256,  1>(float *val);
template __device__ void bitonicSort_reg<float, 512,  1>(float *val);
template __device__ void bitonicSort_reg<float, 1024, 1>(float *val);
template __device__ void bitonicSort_reg<float, 2048, 2>(float *val);
template __device__ void bitonicSort_reg<float, 4096, 4>(float *val);
template __device__ void bitonicSort_reg<float, 8192, 8>(float *val);

template __device__ void bitonicSort_SM<float, 2   >(float *data);
template __device__ void bitonicSort_SM<float, 4   >(float *data);
template __device__ void bitonicSort_SM<float, 8   >(float *data);
template __device__ void bitonicSort_SM<float, 16  >(float *data);
template __device__ void bitonicSort_SM<float, 32  >(float *data);
template __device__ void bitonicSort_SM<float, 64  >(float *data);
template __device__ void bitonicSort_SM<float, 128 >(float *data);
template __device__ void bitonicSort_SM<float, 256 >(float *data);
template __device__ void bitonicSort_SM<float, 512 >(float *data);
template __device__ void bitonicSort_SM<float, 1024>(float *data);
template __device__ void bitonicSort_SM<float, 2048>(float *data);
template __device__ void bitonicSort_SM<float, 4096>(float *data);
template __device__ void bitonicSort_SM<float, 8192>(float *data);


template __device__ float cuOrderStatPow2_sort<float, 2,    1>(float *val, int os);
template __device__ float cuOrderStatPow2_sort<float, 4,    1>(float *val, int os);
template __device__ float cuOrderStatPow2_sort<float, 8,    1>(float *val, int os);
template __device__ float cuOrderStatPow2_sort<float, 16,   1>(float *val, int os);
template __device__ float cuOrderStatPow2_sort<float, 32,   1>(float *val, int os);
template __device__ float cuOrderStatPow2_sort<float, 64,   1>(float *val, int os);
template __device__ float cuOrderStatPow2_sort<float, 128,  1>(float *val, int os);
template __device__ float cuOrderStatPow2_sort<float, 256,  1>(float *val, int os);
template __device__ float cuOrderStatPow2_sort<float, 512,  1>(float *val, int os);
template __device__ float cuOrderStatPow2_sort<float, 1024, 1>(float *val, int os);
template __device__ float cuOrderStatPow2_sort<float, 2048, 2>(float *val, int os);
template __device__ float cuOrderStatPow2_sort<float, 4096, 4>(float *val, int os);
template __device__ float cuOrderStatPow2_sort<float, 8192, 8>(float *val, int os);

template __device__ void cuOrderStatPow2_sort_SM<float, 2   >(float *data, int os);
template __device__ void cuOrderStatPow2_sort_SM<float, 4   >(float *data, int os);
template __device__ void cuOrderStatPow2_sort_SM<float, 8   >(float *data, int os);
template __device__ void cuOrderStatPow2_sort_SM<float, 16  >(float *data, int os);
template __device__ void cuOrderStatPow2_sort_SM<float, 32  >(float *data, int os);
template __device__ void cuOrderStatPow2_sort_SM<float, 64  >(float *data, int os);
template __device__ void cuOrderStatPow2_sort_SM<float, 128 >(float *data, int os);
template __device__ void cuOrderStatPow2_sort_SM<float, 256 >(float *data, int os);
template __device__ void cuOrderStatPow2_sort_SM<float, 512 >(float *data, int os);
template __device__ void cuOrderStatPow2_sort_SM<float, 1024>(float *data, int os);
template __device__ void cuOrderStatPow2_sort_SM<float, 2048>(float *data, int os);
template __device__ void cuOrderStatPow2_sort_SM<float, 4096>(float *data, int os);
template __device__ void cuOrderStatPow2_sort_SM<float, 8192>(float *data, int os);

template __device__ float cuOrderStatPow2_radix<2   >(float *val, int os, int printVals);
template __device__ float cuOrderStatPow2_radix<4   >(float *val, int os, int printVals);
template __device__ float cuOrderStatPow2_radix<8   >(float *val, int os, int printVals);
template __device__ float cuOrderStatPow2_radix<16  >(float *val, int os, int printVals);
template __device__ float cuOrderStatPow2_radix<32  >(float *val, int os, int printVals);
template __device__ float cuOrderStatPow2_radix<64  >(float *val, int os, int printVals);
template __device__ float cuOrderStatPow2_radix<128 >(float *val, int os, int printVals);
template __device__ float cuOrderStatPow2_radix<256 >(float *val, int os, int printVals);
template __device__ float cuOrderStatPow2_radix<512 >(float *val, int os, int printVals);
template __device__ float cuOrderStatPow2_radix<1024>(float *val, int os, int printVals);
template __device__ float cuOrderStatPow2_radix<2048>(float *val, int os, int printVals);
template __device__ float cuOrderStatPow2_radix<4096>(float *val, int os, int printVals);
template __device__ float cuOrderStatPow2_radix<8192>(float *val, int os, int printVals);
