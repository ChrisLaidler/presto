#include "cuda_accel_IN.h"

__device__ inline void swap(int & a, int & b)
{
  a = a ^ b;
  b = a ^ b;
  a = a ^ b;
}

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


/** in-place bitonic sort float array in shared memory  .
 * @param data A pointer to an shared memory array containing elements to be sorted.
 * @param arrayLength The number of elements in the array
 * @param trdId the index of the calling thread (1 thread for 2 items in data)
 * @param noThread The number of thread that are sorting this data
 * @param dir direction to sort data ( 1 -> smallest to largest AND -1 -> largest to smallest )
 *
 * This is an in-place bitonic sort.
 * This is very fast for small numbers of items, ie; when they can all fit in shared memory, or generally are less that 1K or 2K
 *
 * It has a constant performance of \f$ O\left(n\ \log^2 n \right)\f$ where n is the number of items to be sorted.
 * It only works on shared memory as it requires synchronisation.
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

/** Calculate the median of float values  .
 *
 * @param data
 * @param arrayLength
 * @param output
 * @param noSections
 * @param median
 * @param dir
 * @return
 */
template< int bufferSz>
__device__ float cuMedian(float *smBuffer, uint arrayLength, int dir)
{
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;       /// Block ID - flat index
  const int bSz = blockDim.x  * blockDim.y;                     /// Block size

  __shared__ float  medianValue;

  uint noBatches    = ceilf(bufferSz/(float)(bSz) );

  FOLD // Sort  .
  {
    __syncthreads();

    bitonicSort(smBuffer, arrayLength, tid, bSz, dir);
  }

  FOLD // Calculate the median  .
  {
    if ( tid == 0 )
    {
      int idx = arrayLength / 2.0f;

      //const int bid = blockIdx.y  * gridDim.x  + blockIdx.x;        /// Block ID (flat index)
      //printf("%02i: idx %i -> %f  \n", bid, idx, arrayLength / 2.0);

      if ((arrayLength & 1))      // odd
      {
        medianValue = smBuffer[idx];
      }
      else                        //even
      {
        // mean
        //medianValue = (smBuffer[idx-1] + smBuffer[idx])/2.0f;

        // lower
        medianValue = smBuffer[idx - 1];

        // upper
        //medianValue = smBuffer[idx];
      }
    }
  }

  __syncthreads();

  return medianValue;
}

/** Calculate the median of float values  .
 *
 * @param data
 * @param arrayLength
 * @param output
 * @param noSections
 * @param median
 * @param dir
 * @return
 */
template< int bufferSz>
__device__ float cuMedianBySection(float *data, float *smBuffer, uint arrayLength, int dir)
{
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;       /// Block ID - flat index
  const int bSz = blockDim.x  * blockDim.y;                     /// Block size

  // DEBUG //TMP
  const int bid = blockIdx.y  * gridDim.x  + blockIdx.x;        /// Block ID (flat index)

  __shared__ float lower[32];
  __shared__ float upper[32];

  __shared__ float  medianValue;

  __shared__ float  maxLower;
  __shared__ float  minUpper;
  __shared__ int    locLower;
  __shared__ int    locUpper;

  uint noSections   = ceilf(arrayLength/(float)bufferSz);
  uint noBatches    = ceilf(bufferSz/(float)(bSz) );

  const float secs  = 16;

  uint subLen       = bufferSz * 2 / secs;
  uint subStart     = bufferSz / 2 - bufferSz/secs;
  //uint subEnd       = bufferSz / 2 + bufferSz/secs;
  uint len          = noSections * subLen;

  FOLD // Sort each section and write back to device memory  .
  {
    for ( int sec = 0; sec < noSections; sec++)
    {
      FOLD // Load section into shared memory  .
      {
        for ( int batch = 0; batch < noBatches; batch++)
        {
          int start = sec*bufferSz+batch*bSz;
          int idx   = start + tid;

          if ( idx < arrayLength )
          {
            smBuffer[batch*bSz+tid] = data[idx];
          }
        }
      }

      FOLD // Sort  .
      {
        __syncthreads();

        int width = MIN(arrayLength-sec*bufferSz, bufferSz);

        if ( tid == 0 )
        {
          if ( arrayLength-sec*bufferSz < bufferSz )
          {
            printf("%02i  sec %02i  Width %i  \n", bid, sec, width );
          }
        }

        bitonicSort(smBuffer, width, tid, bSz, dir);
      }

      FOLD // Write section from shared memory main memory  .
      {
        __syncthreads();

        for ( int batch = 0; batch < noBatches; batch++)
        {
          int start = sec*bufferSz+batch*bSz;
          int idx   = start + tid;

          if ( idx < arrayLength )
          {
            data[idx] = smBuffer[batch*bSz+tid];
          }
        }
      }
    }
  }

  FOLD // Get median from sections  .
  {
    noBatches  = ceilf(subLen/(float)(bSz) );

    if ( tid == 0 )
    {
      printf("%02i  noBatches: %.2f  sections %.2f  subStart %ui  subLen: %ui \n", bid, subLen/(float)(bSz), arrayLength/(float)bufferSz, subStart, subLen );
    }

    FOLD // Load section into shared memory  .
    {
      __syncthreads();

      for ( int sec = 0; sec < noSections; sec++)
      {
        int startGM     = bufferSz*sec + subStart;
        int startSM     = subLen*sec;

        for ( int batch = 0; batch < noBatches; batch++)
        {
          int idx = startGM + batch*bSz + tid;
          if ( idx < arrayLength )
          {
            smBuffer[startSM + batch*bSz + tid] = data[idx];
          }
          else
          {
            printf("%02i: BAD\n", bid);
          }
        }
      }
    }

    FOLD // Read start values from SM  .
    {
      __syncthreads();

      if ( tid < noSections )
      {
        lower[tid]      = smBuffer[subLen*tid];
        upper[tid]      = smBuffer[subLen*(tid+1)-1];
      }
    }

    FOLD // Sort the mid section in SM  .
    {
      __syncthreads();

      bitonicSort(smBuffer, len, tid, bSz, dir);
    }

    FOLD // Get the largest of the lower values  .
    {
      __syncthreads();

      if ( tid == 0 )
      {
        maxLower = lower[0];
        minUpper = lower[0];
        locLower = 0;
        locUpper = len;

        for ( int sec = 0; sec < noSections; sec++)
        {
          if ( lower[sec] > maxLower )
            maxLower = lower[sec];

          if ( upper[sec] < minUpper )
            minUpper = upper[sec];
        }

        if ( tid == 0 )
        {
          printf("%02i maxLower %.6i  minUpper %.6f", bid, maxLower, minUpper);
        }
      }
    }

    FOLD // Find the last location of the largest of the lower values  .
    {
      __syncthreads();

      noBatches  = ceilf(len/(float)(bSz) );
      for ( int batch = 0; batch < noBatches; batch++)
      {
        int idx = batch*bSz + tid;

        if ( idx < len )
        {
          if ( smBuffer[idx] == maxLower )
          {
            printf("%02i tid %02i  maxLower = %i \n", bid, tid, idx );
            atomicMax(&locLower, idx);
          }

          if ( smBuffer[idx] == minUpper )
          {
            printf("%02i tid %02i  locUpper = %i \n", bid, tid, idx );
            atomicMin(&locUpper, idx);
          }
        }
      }

      __syncthreads(); // TMP
      if ( tid == 0 )
      {
        printf("%02i locLower %i  locUpper %i", bid, locLower, locUpper);
      }
    }

    FOLD //  .
    {
      __syncthreads();

      if (tid == 0 )
      {
        int before  = noSections*subStart + locLower;
        //int after   = noSections*(bufferSz-subEnd) + (len-locUpper);
        int mid     = len / 2.0f;

        if ( locLower > mid || locUpper < mid )
        {
          printf("ERROR: Section length to short!\n");
        }

        int idx = arrayLength / 2.0f;
        int medianInSecton  = idx - before;

        medianValue = smBuffer[locLower+medianInSecton];

        if ((arrayLength & 1))      // odd
        {
          medianValue = smBuffer[locLower+medianInSecton];
        }
        else                        //even
        {
          // mean
          //medianValue = ( smBuffer[locLower+medianInSecton-1] + smBuffer[locLower+medianInSecton] ) / 2.0f;

          // lower
          medianValue = smBuffer[locLower+medianInSecton-1];

          // upper
          //medianValue = smBuffer[locLower+medianInSecton];
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
  const int bSz = blockDim.x*blockDim.y;                        /// Block size

  __shared__ float sData[BS_MAX];
  float medianValue;
  float factor;

  int width = lens.val[bid];

  int batches = ceil( width / (float) bSz );

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

    medianValue = cuMedian<BS_MAX>(sData, width, 1);

    if ( tid == 0 )
    {
      printf("%02i  batches: %.2f 1 section  median %.6f \n", bid, width / (float) bSz, medianValue );
    }
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

    medianValue = cuMedianBySection<BS_MAX>(powers, sData, width, 1);

    if ( tid == 0 )
    {
      printf("%02i  batches: %.2f 1 section  median %.3f \n", bid, width / (float) bSz, medianValue );
    }
  }

  factor = 1.0 / sqrt(medianValue / LN2 );

  for ( int batch = batches-1; batch >= 0; batch--)
  {
    // Read all values into registers
    fcomplexcu val = data[batch*bSz+tid];
    __syncthreads();

    // Set the value to normalised complex number spread by 2
    val.i *= factor;
    val.r *= factor;
    data[(batch*bSz+tid)*2]     = val;

    // Set every second value to 0
    val.i = 0;
    val.r = 0;
    data[(batch*bSz+tid)*2+1]   = val;
  }

}

//template<uint FLAGS, int noStages, const int noHarms, const int cunkSize, int noSteps>
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
  noInput = 1 ;

  if      ( noInput <= 1   )
  {
    int01 iLen;
    for (int si = 0; si < cStack->noInStack; si++)
    {
      for (int step = 0; step < batch->noSteps; step++)
      {
        //if ( !(searchRLow[step] == 0 &&  searchRHi[step] == 0) )
        {
          rVals* rVal = &((*batch->rInput)[step][harm]);

          printf("stack %02i si %02i step %02i  cStack->width %04i   rVal->numdata %04i \n", stack, si,  step, cStack->width, rVal->numdata);

          if (stp < noInput ) // TMP
            iLen.val[stp] = rVal->numdata ;
        }
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
        //if ( !(searchRLow[step] == 0 &&  searchRHi[step] == 0) )
        {
          rVals* rVal = &((*batch->rInput)[step][harm]);

          printf("stack %02i si %02i step %02i  cStack->width %04i   rVal->numdata %04i \n", stack, si,  step, cStack->width, rVal->numdata);

          iLen.val[stp] = rVal->numdata ;
        }
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
        //if ( !(searchRLow[step] == 0 &&  searchRHi[step] == 0) )
        {
          rVals* rVal = &((*batch->rInput)[step][harm]);

          printf("stack %02i si %02i step %02i  cStack->width %04i   rVal->numdata %04i \n", stack, si,  step, cStack->width, rVal->numdata);

          iLen.val[stp] = rVal->numdata ;
        }
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
        //if ( !(searchRLow[step] == 0 &&  searchRHi[step] == 0) )
        {
          rVals* rVal = &((*batch->rInput)[step][harm]);

          printf("stack %02i si %02i step %02i  cStack->width %04i   rVal->numdata %04i \n", stack, si,  step, cStack->width, rVal->numdata);

          iLen.val[stp] = rVal->numdata ;
        }
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
        {
          rVals* rVal = &((*batch->rInput)[step][harm]);

          printf("stack %02i si %02i step %02i  cStack->width %04i   rVal->numdata %04i \n", stack, si,  step, cStack->width, rVal->numdata);

          iLen.val[stp] = rVal->numdata ;
        }
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
        {
          rVals* rVal = &((*batch->rInput)[step][harm]);

          printf("stack %02i si %02i step %02i  cStack->width %04i   rVal->numdata %04li \n", stack, si,  step, cStack->width, rVal->numdata);

          iLen.val[stp] = rVal->numdata ;
        }
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
        {
          rVals* rVal = &((*batch->rInput)[step][harm]);

          printf("stack %02i si %02i step %02i  cStack->width %04i   rVal->numdata %04i \n", stack, si,  step, cStack->width, rVal->numdata);

          iLen.val[stp] = rVal->numdata ;
        }
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
        {
          rVals* rVal = &((*batch->rInput)[step][harm]);

          printf("stack %02i si %02i step %02i  cStack->width %04i   rVal->numdata %04i \n", stack, si,  step, cStack->width, rVal->numdata);

          iLen.val[stp] = rVal->numdata ;
        }
        stp++;
      }
      harm++;
    }

    normAndSpread<width, int128><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, iLen);
  }
  else
  {
    fprintf(stderr,"ERROR: %s has not been set up to work with %i elements.",__FUNCTION__, noInput);
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

  dimGrid.x = 1;

  printf("\n\n======================  normAndSpread_f  ======================\n");

  switch (cStack->width)
  {
    case 512:
    {
      normAndSpread_w<512>(dimGrid, dimBlock, i1, inpStream, batch, stack);
      break;
    }
    case 1024:
    {
      normAndSpread_w<1024>(dimGrid, dimBlock, i1, inpStream, batch, stack);
      break;
    }
    case 2048:
    {
      normAndSpread_w<2048>(dimGrid, dimBlock, i1, inpStream, batch, stack);
      break;
    }
    case 4096:
    {
      normAndSpread_w<4096>(dimGrid, dimBlock, i1, inpStream, batch, stack);
      break;
    }
    case 8192:
    {
      normAndSpread_w<8192>(dimGrid, dimBlock, i1, inpStream, batch, stack);
      break;
    }
    case 16384:
    {
      normAndSpread_w<16384>(dimGrid, dimBlock, i1, inpStream, batch, stack);
      break;
    }
    case 32768:
    {
      normAndSpread_w<32768>(dimGrid, dimBlock, i1, inpStream, batch, stack);
      break;
    }
    default:
    {
      fprintf(stderr, "ERROR: %s has not been templated for %lu steps\n", __FUNCTION__, cStack->width);
      exit(EXIT_FAILURE);
    }
  }

  CUDA_SAFE_CALL(cudaGetLastError(), "Calling the normalisation and spreading kernel.");
}
