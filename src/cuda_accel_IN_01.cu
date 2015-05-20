#include "cuda_accel_IN.h"

/** XOR swap two integer values
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

/** Compare and swap two values (if they are in the wrong order).
 *
 * @param valA The first value
 * @param valB The second value
 * @param dir the desired order 1 = increasing
 */
__device__ inline void Comparator(float &valA, float &valB, uint dir)
{
  if ((valA > valB) == dir)
  {
    register float t;
    //swap(*(int*)&valA, *(int*)&valB );
    t = valA;
    valA = valB;
    valB = t;
  }
}

/** In-place Bitonic sort a float array.
 * @param data A pointer to an shared memory array containing elements to be sorted.
 * @param arrayLength The number of elements in the array
 * @param trdId the index of the calling thread (1 thread for 2 items in data)
 * @param noThread The number of thread that are sorting this data
 * @param dir direction to sort data ( 1 -> smallest to largest AND -1 -> largest to smallest )
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
 */
__device__ void bitonicSort(float *data, const uint arrayLength, const uint trdId, const uint noThread, const int dir )
{
  const uint noBatch = ceilf(arrayLength / 2.0f / noThread);     // Number of comparisons each thread must do
  uint idx;                               // The index including batch adjustment
  const uint max = arrayLength * 2;       // The maximum distance a thread could compare
  uint bIdx;                              // The thread position in the block
  uint hSz = 1;                           // half block size
  uint pos1, pos2, blk;                   // index of points to be compared
  uint len;                               // The distance between items to swap
  uint bach;                              // The batch we are processing
  uint shift = 32;                        // Amount to bitshift by to calculate remainders
  uint shift2;
  uint hsl1;

  // Incrementally sort blocks of 2 then 4 then 8 ... items
  for (uint size = 2; size < max; size <<= 1, shift--)
  {
    hSz = (size >> 1);
    hsl1 = hSz - 1;

    __syncthreads();

    // Bitonic sort, two Bitonic sorted list into Bitonic list
    for (bach = 0; bach < noBatch; bach++)
    {
      idx = (trdId + bach * noThread);

      //bIdx = hSz - 1 - idx % hSz;
      //bIdx = hsl1 - (idx << shift) >> shift;  // My method
      bIdx = hsl1 - idx & (hSz - 1);// x mod y == x & (y-1), where y is 2^n.

      blk = idx / hSz;

      len = size - 1 - bIdx * 2;
      pos1 = blk * size + bIdx;
      pos2 = pos1 + len;

      if (pos2 < arrayLength)
        Comparator(data[pos1], data[pos2], dir);
    }

    // Bitonic Merge
    for (len = (hSz >>= 1), shift2 = shift + 1; len > 0; len >>= 1, shift2++)
    {
      hSz = (len << 1);

      __syncthreads();
      for (bach = 0; bach < noBatch; bach++)
      {
        idx = (trdId + bach * noThread);

        //bIdx  = idx % len;
        //bIdx = (idx << shift2) >> shift2;
        bIdx = idx & (len - 1);// x mod y == x & (y-1), where y is 2^n.

        blk = idx / len;

        pos1 = blk * hSz + bIdx;
        pos2 = pos1 + len;

        if (pos2 < arrayLength)
          Comparator(data[pos1], data[pos2], dir);
      }
    }
  }

  __syncthreads();  // Ensure all data is sorted before we return
}

/** Calculate the median of an array of float values  .
 *
 * This sorts the actual array so the values will be reordered
 * This uses a bitonicSort which is very fast if the array is in SM
 * This means that there
 *
 * @param array array of floats to search, this will be reordered should be in SM
 * @param arrayLength the number of floats in the array
 * @param dir the direction to sort the array 1 = increasing
 * @return the median value
 */
__device__ float cuMedianOne(float *array, uint arrayLength)
{
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;       /// Block ID - flat index
  const int bSz = blockDim.x  * blockDim.y;                     /// Block size

  __shared__ float  medianValue;

  FOLD // Sort  .
  {
    __syncthreads();

    bitonicSort(array, arrayLength, tid, bSz, 1);
  }

  FOLD // Calculate the median  .
  {
    if ( tid == 0 )
    {
      int idx = arrayLength / 2.0f;

      if ((arrayLength & 1))      // odd
      {
        medianValue = array[idx];
      }
      else                        //even
      {
        // mean
        //medianValue = (smBuffer[idx-1] + smBuffer[idx])/2.0f;

        // lower
        medianValue = array[idx - 1];

        // upper
        //medianValue = smBuffer[idx];
      }
    }
  }

  __syncthreads();

  return medianValue;
}

/** Calculate the median of up to 16*bufferSz float values  .
 *
 * This Sorts sections of the array, and then find the median by extracting and combining the
 * centre chunk(s) of these and sorting that. To find the median.
 *
 * Note this reorders the original array
 *
 * @param data the value to find the median of
 * @param buffer to do the sorting in this is bufferSz long and should be in SM
 * @param arrayLength the length of the data array
 * @return The median of data
 */
template< int bufferSz>
__device__ float cuMedianBySection(float *data, float *buffer, uint arrayLength)
{
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;       /// Block ID - flat index
  const int bSz = blockDim.x  * blockDim.y;                     /// Block size

  const float secs  = 32;

  __shared__ float lower[16];
  __shared__ float upper[16];

  __shared__ float  medianValue;

  __shared__ float  maxLower;
  __shared__ float  minUpper;
  __shared__ int    locLower;
  __shared__ int    locUpper;

  uint noSections   = ceilf(arrayLength/(float)bufferSz);
  uint noBatches    = ceilf(bufferSz/(float)(bSz) );

  uint subLen       = bufferSz * 2 / secs;
  uint len          = 0;
  uint before       = 0;

  FOLD // Sort each section and write back to device memory  .
  {
    for ( int sec = 0; sec < noSections; sec++)
    {
      int sStart      = MIN(bufferSz*sec, arrayLength);
      int sEnd        = MIN(bufferSz*(sec+1)-1, arrayLength);
      int sLen        = sEnd - sStart;

      int mStart      = MAX(0,sLen/2.0f - bufferSz/secs);
      int mEnd        = MIN(sLen,mStart+subLen);

      FOLD // Load section into shared memory  .
      {
        for ( int batch = 0; batch < noBatches; batch++)
        {
          int start = sec*bufferSz+batch*bSz;
          int idx   = start + tid;

          if ( idx < arrayLength )
          {
            buffer[batch*bSz+tid] = data[idx];
          }
        }
      }

      FOLD // Sort  .
      {
        __syncthreads();

        int width = MIN(arrayLength-sec*bufferSz, bufferSz);

        bitonicSort(buffer, width, tid, bSz, 1);
      }

      FOLD // Write section from shared memory main memory  .
      {
        __syncthreads();

        for ( int batch = 0; batch < noBatches; batch++)
        {
          int start   = sec*bufferSz+batch*bSz;
          int gmIdx   = start + tid;
          int smIdx   = batch*bSz+tid;

          if ( smIdx >= mStart && smIdx < mEnd )
          {
            data[gmIdx] = buffer[smIdx];
          }
        }
      }
    }
  }

  FOLD // Get median from sections  .
  {
    noBatches  = ceilf(subLen/(float)(bSz) );

    FOLD // Load the middle of each section into shared memory  .
    {
      __syncthreads();

      for ( int sec = 0; sec < noSections; sec++)
      {
        int sStart      = MIN(bufferSz*sec, arrayLength);
        int sEnd        = MIN(bufferSz*(sec+1)-1, arrayLength);
        int sLen        = sEnd - sStart;

        int mStart      = MAX(0,sLen/2.0f - bufferSz/secs);
        int mEnd        = MIN(sLen,mStart+subLen);
        int mLen        = mEnd - mStart;

        int startGM     = bufferSz*sec + mStart;

        for ( int batch = 0; batch < noBatches; batch++)
        {
          int idx = startGM + batch*bSz + tid;

          if ( idx < arrayLength )
          {
            buffer[len + batch*bSz + tid] = data[idx];
          }
        }

        len             += mLen;
        before          += mStart;
      }
    }

    FOLD // Read start values from SM  .
    {
      __syncthreads();

      if ( tid < noSections )
      {
        lower[tid]      = buffer[subLen*tid];

        int idx         = MIN(len,subLen*(tid+1));
        upper[tid]      = buffer[idx-1];
      }
    }

    FOLD // Sort the collection of mid sections in SM  .
    {
      __syncthreads();

      bitonicSort(buffer, len, tid, bSz, 1);
    }

    FOLD // Find the bounding vales  .
    {
      __syncthreads();

      if ( tid == 0 )
      {
        maxLower = lower[0];
        minUpper = upper[0];
        locLower = 0;
        locUpper = len;

        for ( int sec = 0; sec < noSections; sec++)
        {
          if ( lower[sec] > maxLower )
            maxLower = lower[sec];

          if ( upper[sec] < minUpper )
            minUpper = upper[sec];
        }
      }
    }

    FOLD // Find the location of the bounding vales  .
    {
      __syncthreads();

      noBatches  = ceilf(len/(float)(bSz) );

      for ( int batch = 0; batch < noBatches; batch++)
      {
        int idx = batch*bSz + tid;

        if ( idx < len )
        {
          if ( buffer[idx] == maxLower )
          {
            atomicMax(&locLower, idx);
          }

          if ( buffer[idx] == minUpper )
          {
            atomicMin(&locUpper, idx);
          }
        }
      }
    }

    FOLD // Find the index of the median in the buffer  .
    {
      __syncthreads();

      if (tid == 0 )
      {
        int GMIdx   = arrayLength / 2.0f;
        int SMIdx   = GMIdx - before;

        // The true median will fall between the bounding values
        if ( (SMIdx >= locUpper) || (SMIdx <= locLower) )
        {
          printf("ERROR: In function %s, median not in mid section, median value will be incorrect!\n", __FUNCTION__ );
        }

        FOLD // Median selection  .
        {
          if ( (arrayLength & 1) )    // odd
          {
            medianValue = buffer[SMIdx];
          }
          else                        //even
          {
            // mean
            //medianValue = ( smBuffer[SMIdx-1] + smBuffer[mIdx] ) / 2.0f;

            // lower
            medianValue = buffer[SMIdx-1];

            // upper
            //medianValue = buffer[SMIdx];
          }
        }
      }
    }
  }

  __syncthreads();

  return medianValue;
}

template< int stride, typename stpType>
__global__ void normAndSpread(fcomplexcu* data, stpType lens)
{
  const int bid = blockIdx.y  * gridDim.x  + blockIdx.x;        /// Block ID (flat index)
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;       /// Thread ID in block (flat index)
  const int bSz = blockDim.x  * blockDim.y;                     /// Block size

  int width = lens.val[bid];
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
            fcomplexcu val = data[idx];
            sData[idx] = val.r*val.r+val.i*val.i;
          }
        }
      }

      medianValue = cuMedianOne(sData, width);
    }
    else
    {
      float* powers = (float*)&data[width];

      FOLD // Calculate and store powers in device memory  .
      {
        for ( int batch = 0; batch < batches; batch++)
        {
          fcomplexcu val = data[batch*bSz+tid];
          powers[batch*bSz+tid] = val.r*val.r+val.i*val.i;
        }
      }

      medianValue = cuMedianBySection<BS_MAX>(powers, sData, width);
    }

    // Calculate normalisation factor
    factor = 1.0 / sqrt( medianValue / LN2 );

    //  if ( tid == 0 )
    //  {
    //    float sec = width / (float)BS_MAX ;
    //    printf("%02i  batches: %4.2f %3.2f section  median %.6f  factor: %20.20f \n", bid, width / (float) bSz, sec, medianValue, factor );
    //  }

    batches = ceil( stride / (float) bSz );

    // Write spread by 2 and normalise
    for ( int batch = batches-1; batch >= 0; batch--)
    {
      // Read all values into registers
      fcomplexcu val = data[batch*bSz+tid];
      __syncthreads();

      int idx = batch*bSz+tid;

      if ( (idx)*2 < stride)
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
        data[(idx)*2]     = val;

        // Set every second value to 0
        val.i = 0;
        val.r = 0;
        data[(idx)*2+1]   = val;
      }
    }
  }
}

template<int width>
__host__ void normAndSpread_w(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t stream, cuFFdotBatch* batch, uint stack )
{
  int harm    = 0;
  int stp     = 0;
  for ( int i = 0; i < stack; i++)
  {
    harm += batch->stacks[i].noInStack;
  }

  cuFfdotStack* cStack = &batch->stacks[stack];

  int noInput = cStack->noInStack * batch->noSteps ;

  if      ( noInput <= 1   )
  {
    int01 iLen;
    for (int si = 0; si < cStack->noInStack; si++)
    {
      for (int step = 0; step < batch->noSteps; step++)
      {
        rVals* rVal = &((*batch->rInput)[step][harm]);

        if ( stp < noInput )
          iLen.val[stp] = rVal->numdata ;

        stp++;
      }
      harm++;
    }

    normAndSpread<width, int01><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, iLen);
  }
  else if ( noInput <= 2   )
  {
    int02 iLen;
    for (int si = 0; si < cStack->noInStack; si++)
    {
      for (int step = 0; step < batch->noSteps; step++)
      {
        rVals* rVal = &((*batch->rInput)[step][harm]);

        iLen.val[stp] = rVal->numdata ;
        stp++;
      }
      harm++;
    }

    normAndSpread<width, int02><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, iLen);
  }
  else if ( noInput <= 4   )
  {
    int04 iLen;
    for (int si = 0; si < cStack->noInStack; si++)
    {
      for (int step = 0; step < batch->noSteps; step++)
      {
        rVals* rVal = &((*batch->rInput)[step][harm]);

        iLen.val[stp] = rVal->numdata ;
        stp++;
      }
      harm++;
    }

    normAndSpread<width, int04><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, iLen);
  }
  else if ( noInput <= 8   )
  {
    int08 iLen;
    for (int si = 0; si < cStack->noInStack; si++)
    {
      for (int step = 0; step < batch->noSteps; step++)
      {
        rVals* rVal = &((*batch->rInput)[step][harm]);

        iLen.val[stp] = rVal->numdata ;
        stp++;
      }
      harm++;
    }

    normAndSpread<width, int08><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, iLen);
  }
  else if ( noInput <= 16  )
  {
    int16 iLen;
    for (int si = 0; si < cStack->noInStack; si++)
    {
      for (int step = 0; step < batch->noSteps; step++)
      {
        rVals* rVal = &((*batch->rInput)[step][harm]);

        iLen.val[stp] = rVal->numdata ;
        stp++;
      }
      harm++;
    }

    normAndSpread<width, int16><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, iLen);
  }
  else if ( noInput <= 32  )
  {
    int32 iLen;
    for (int si = 0; si < cStack->noInStack; si++)
    {
      for (int step = 0; step < batch->noSteps; step++)
      {
        rVals* rVal = &((*batch->rInput)[step][harm]);

        iLen.val[stp] = rVal->numdata ;
        stp++;
      }
      harm++;
    }

    normAndSpread<width, int32><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, iLen);
  }
  else if ( noInput <= 64  )
  {
    int64 iLen;
    for (int si = 0; si < cStack->noInStack; si++)
    {
      for (int step = 0; step < batch->noSteps; step++)
      {
        rVals* rVal = &((*batch->rInput)[step][harm]);

        iLen.val[stp] = rVal->numdata ;
        stp++;
      }
      harm++;
    }

    normAndSpread<width, int64><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, iLen);
  }
  else if ( noInput <= 128 )
  {
    int128 iLen;
    for (int si = 0; si < cStack->noInStack; si++)
    {
      for (int step = 0; step < batch->noSteps; step++)
      {
        rVals* rVal = &((*batch->rInput)[step][harm]);

        iLen.val[stp] = rVal->numdata ;
        stp++;
      }
      harm++;
    }

    normAndSpread<width, int128><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, iLen);
  }
  else
  {
    fprintf(stderr,"ERROR: %s has not been set up to work with %i input sections.",__FUNCTION__, noInput);
  }
}

__host__ void normAndSpread_f(cudaStream_t inpStream, cuFFdotBatch* batch, uint stack )
{
  dim3 dimBlock, dimGrid;
  int i1 = 0;
  cuFfdotStack* cStack = &batch->stacks[stack];

  // Blocks of 1024 threads ( the maximum number of threads per block )
  dimBlock.x = NAS_DIMX;
  dimBlock.y = NAS_DIMY;
  dimBlock.z = 1;

  // One block per harmonic, thus we can sort input powers in Shared memory
  dimGrid.x = cStack->noInStack * batch->noSteps;
  dimGrid.y = 1;

  switch (cStack->width)
  {
    case 128   :
    {
      normAndSpread_w<128>(dimGrid, dimBlock, i1, inpStream, batch, stack);
      break;
    }
    case 256   :
    {
      normAndSpread_w<256>(dimGrid, dimBlock, i1, inpStream, batch, stack);
      break;
    }
    case 512   :
    {
      normAndSpread_w<512>(dimGrid, dimBlock, i1, inpStream, batch, stack);
      break;
    }
    case 1024  :
    {
      normAndSpread_w<1024>(dimGrid, dimBlock, i1, inpStream, batch, stack);
      break;
    }
    case 2048  :
    {
      normAndSpread_w<2048>(dimGrid, dimBlock, i1, inpStream, batch, stack);
      break;
    }
    case 4096  :
    {
      normAndSpread_w<4096>(dimGrid, dimBlock, i1, inpStream, batch, stack);
      break;
    }
    case 8192  :
    {
      normAndSpread_w<8192>(dimGrid, dimBlock, i1, inpStream, batch, stack);
      break;
    }
    case 16384 :
    {
      normAndSpread_w<16384>(dimGrid, dimBlock, i1, inpStream, batch, stack);
      break;
    }
    case 32768 :
    {
      normAndSpread_w<32768>(dimGrid, dimBlock, i1, inpStream, batch, stack);
      break;
    }
    default    :
    {
      fprintf(stderr, "ERROR: %s has not been templated for %lu steps\n", __FUNCTION__, cStack->width);
      exit(EXIT_FAILURE);
    }
  }

  CUDA_SAFE_CALL(cudaGetLastError(), "Calling the normalisation and spreading kernel.");
}
