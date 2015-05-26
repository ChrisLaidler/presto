#include "cuda_accel_IN.h"

__device__ inline float median(float* data, int arrayLength, int eType = 0 )
{
  float medianValue;

  int idx = arrayLength/2.0;

  if ( (arrayLength & 1) )    // odd  .
  {
    medianValue = data[idx];
  }
  else                        //even  .
  {
    if ( eType == -1 )        // lower  .
    {
      medianValue = data[idx-1];
    }
    else if ( eType == 1 )    // upper  .
    {
      medianValue = data[idx];
    }
    else                      // mean  .
    {
      medianValue = ( data[idx-1] + data[idx] ) / 2.0f;
    }
  }

  return medianValue;
}

__device__ inline int midpoint(int imin, int imax)
{
  return (imin + imax) / 2.0 ;
}

__device__ int binSearch(const float* data, float key, int arrayLength)
{
  const int bid = blockIdx.y  * gridDim.x  + blockIdx.x;        /// Block ID (flat index) // TMP
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;       /// Block ID - flat index

  int imax = arrayLength-1;
  int imin = 0;

//  int ite = 0;

  while ( imax >= imin )
  {
    int imid = midpoint(imin, imax) ;

//    if ( bid == 0 && tid == 0 )
//      printf("%02i imin %05i  imin %05i imax %05i   %22.6f  %22.6f  %22.6f    key %22.6f  \n", ite, imin, imid, imax, data[imin], data[imid], data[imax],  key);

    if      ( data[imid] == key )
      return imid;
    else if ( data[imid] <  key )
      imin = imid + 1;
    else
      imax = imid - 1;
  }

//  if ( bid == 0 && tid == 0 )
//    printf("Returning %02i  \n", imin);

  return imin;
}

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
__device__ void bitonicSort1Warp(float *data, const uint arrayLength, const uint trdId, const uint noThread, const int dir )
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
template< int bufferSz >
__device__ float cuMedianBySection(float *data, float *buffer, uint arrayLength)
{
  const int bid = blockIdx.y  * gridDim.x  + blockIdx.x;        /// Block ID (flat index) // TMP

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;       /// Block ID - flat index
  const int bSz = blockDim.x  * blockDim.y;                     /// Block size

  const int maxSub = 32;

  __shared__ float  lower[maxSub];
  __shared__ float  upper[maxSub];
  __shared__ float  medianA[maxSub];

  __shared__ float  medianValue;

  __shared__ float  maxLower;
  __shared__ float  minUpper;
  __shared__ int    locLower;
  __shared__ int    locUpper;
  __shared__ bool   found;
  __shared__ int    belowN;

  int noSections        = ceilf( arrayLength/(float)bufferSz );
  int noBatches         = ceilf( bufferSz/(float)(bSz) );

  if ( noSections <= maxSub )
  {
    //if ( tid < bufferSz)
    {
      int midLen          = floorf( bufferSz/(float)noSections );   // The length of middle sections
      float hSz           = ceilf( midLen/2.0f );                   // Half the length of the middle sections
      int ite             = 0;                                      // Iteration
      float rMedian;                                                // The number of medians to consider
      int len;                                                      // Only really needed by tid 0
      int before;                                                   // Only really needed by tid 0
      float pPos;

      float lowerVal;
      float upperVal;

      if ( tid == 0 )
      {
        found = false;
      }

      FOLD // Sort each section and write back to device memory  .
      {
        for ( int sec = 0; sec < noSections; sec++ )
        {
          int sStart      = MIN(bufferSz*sec,     arrayLength);
          int sEnd        = MIN(bufferSz*(sec+1), arrayLength);
          int sLen        = sEnd - sStart;

          int mMid        = sLen / 2.0 ;

          FOLD // Load section into shared memory  .
          {
            __syncthreads();

            for ( int batch = 0; batch < noBatches; batch++)
            {
              int dataIdx   = sec*bufferSz + batch*bSz + tid ;
              int bufferIdx = batch*bSz + tid ;

              if ( dataIdx < arrayLength && bufferIdx < sLen )
              {
                buffer[bufferIdx] = data[dataIdx];
              }
            }
          }

          FOLD // Sort  .
          {
            __syncthreads();

            bitonicSort(buffer, sLen, tid, bSz, 1);
          }

          FOLD // Write section from shared memory main memory  .
          {
            __syncthreads();

            for ( int batch = 0; batch < noBatches; batch++)
            {
              int dataIdx   = sec*bufferSz + batch*bSz + tid ;
              int bufferIdx = batch*bSz + tid ;

              if ( dataIdx < arrayLength && bufferIdx < sLen )
              {
                float val       = buffer[bufferIdx];
                data[dataIdx]   = val;

                if ( bufferIdx == mMid )
                {
                  medianA[sec]  = val;
                }
              }
            }
          }
        }
      }

      FOLD // Calculate the median of the median  .
      {
        FOLD // Sort the medians  .
        {
          __syncthreads();

          if ( tid < noSections )
          {
            bitonicSort1Warp(medianA, noSections, tid, bSz, 1);
          }
        }

        FOLD // Calculate the median of medians  .
        {
          __syncthreads();

          if ( tid == 0 )
          {
            medianValue           = median(medianA, noSections, 0 );
            pPos                  = (noSections-1)/2.0;
            lowerVal              = medianA[0];
            upperVal              = medianA[noSections-1];
          }
        }
      }

      FOLD // Get median from sections  .
      {
        __syncthreads();

        while ( !found )
        {
          FOLD // Initialise values  .
          {
            len         = 0;
            before      = 0;
            rMedian     = medianValue;
          }

          FOLD // Load the middle of each section into shared memory and save lower and upper values  .
          {
            __syncthreads();

            for ( int sec = 0; sec < noSections; sec++)
            {
              int sStart      = MIN(bufferSz*sec,     arrayLength);
              int sEnd        = MIN(bufferSz*(sec+1), arrayLength);
              int sLen        = sEnd - sStart;

              FOLD // Find out how many points below the pivot  .
              {
                __syncthreads();

                if ( tid == 0 )
                {
                  belowN      = binSearch(&data[sec*bufferSz], rMedian, sLen);
                }
              }

              __syncthreads();

              int mStart      = MAX(0,    belowN - hSz );
              int mEnd        = MIN(sLen, belowN + hSz );
              int mLen        = mEnd    - mStart;
              int dataStart   = sStart  + mStart;

              for ( int batch = 0; batch < noBatches; batch++ )
              {
                int bIdx      = batch*bSz + tid ;

                if ( bIdx < mLen )
                {
                  int dataIdx   = dataStart + bIdx ;
                  int bufferIdx = len       + bIdx ;

                  float val         = data[dataIdx];
                  buffer[bufferIdx] = val;

                  FOLD  // Set the max and min for this section  .
                  {
                    // Lower
                    if ( bIdx == 0 )
                    {
                      if ( mStart > 0 )
                        lower[sec] = val;
                      else
                        lower[sec] = -1;
                    }

                    // Upper
                    if ( bIdx == mLen - 1 )
                    {
                      if ( mEnd < sLen )
                        upper[sec] = val;
                      else
                        upper[sec] = -1;
                    }
                  }
                }
              }

              len             += mLen;
              before          += mStart;
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
                if ( lower[sec] >= 0 )
                {
                  if ( lower[sec] > maxLower || maxLower == -1 )
                    maxLower = lower[sec];
                }

                if ( upper[sec] >= 0 )
                {
                  if ( upper[sec] < minUpper || minUpper == -1 )
                    minUpper = upper[sec];
                }
              }
            }
          }

          FOLD // Find the location of the bounding vales  .
          {
            __syncthreads();

            int noBatchesMid  = ceilf( len/(float)(bSz) );

            for ( int batch = 0; batch < noBatchesMid; batch++)
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

            Fout // TMP Write section values .
            {
              if ( (ite > 0) && (tid == 0) && (bid == 14) )
              {
                __syncthreads();

                for ( int sec = 0; sec < noSections; sec++)
                {
                  int sStart      = MIN(bufferSz*sec,     arrayLength);
                  int sEnd        = MIN(bufferSz*(sec+1), arrayLength);
                  int sLen        = sEnd - sStart;

                  printf("%02i sec\t%02i\tsmid\t%04i\t%22.6f\t%22.6f\t\n", bid,  sec, (int)(sLen/2.0f),  lower[sec], upper[sec] ) ;
                }
              }
            }
          }

          FOLD // Find the index of the median in the buffer  .
          {
            __syncthreads();

            if ( tid == 0 )
            {
              int GMIdx   = arrayLength / 2.0f;
              int SMIdx   = GMIdx - before;

              if      ( (SMIdx <= 0) 	|| (SMIdx <= locLower) )
              {
                medianValue = buffer[locLower];
                //upperVal    = buffer[locLower];
                //medianValue = (upperVal + lowerVal )/2.0;
              }
              else if ( (SMIdx >= len) || (SMIdx >= locUpper) )
              {
                medianValue = buffer[locUpper];
                //lowerVal    = buffer[locUpper];
                //medianValue = (upperVal + lowerVal )/2.0;
              }
              else
              {
                found = true;
              }

              Fout  // TMP  Write lots of output .
              {
                if ( !found )
                {
                  if  ( bid == 12 )
                  {
                    int bid = blockIdx.y  * gridDim.x  + blockIdx.x;        /// Block ID (flat index)
                    printf("\n");
                    printf(" %02i\t%22.6f\t%22.6f\t%22.6f \n", ite,lowerVal, medianValue, upperVal );
                    printf("ERROR: In function %s, In block %02i median not in mid section, median value will be incorrect!  buff %04i | no sec %02i  |  locLower %04i  SMIdx %04i  locUpper %04i \n", __FUNCTION__,  bid, bufferSz, noSections, locLower, SMIdx, locUpper );
                    printf(" ite %02i old rMedian %.6f  new rMedian %.6f \n", ite, rMedian, medianValue );
                    printf(" rMedian %.6f   maxLower %.6f   minUpper %.6f   arrayLength %04i   medIdx %04i    before %04i   len %04i  \n", medianValue, maxLower, minUpper, (int)arrayLength, (int)(arrayLength/2.0), before, len ) ;
                    printf(" %22.6f  %22.6f   %04i  %04i \n", maxLower, minUpper, locLower, locUpper ) ;

                    FOLD // Write section values .
                    {
                      for ( int sec = 0; sec < noSections; sec++)
                      {
                        int sStart      = MIN(bufferSz*sec,     arrayLength);
                        int sEnd        = MIN(bufferSz*(sec+1), arrayLength);
                        int sLen        = sEnd - sStart;

                        printf("sec\t%02i\tsmid\t%04i\t%22.6f\t%22.6f\t\n", sec, (int)(sLen/2.0f),  lower[sec], upper[sec] ) ;
                      }
                    }

                    FOLD // TMP  write medians  .
                    {
                      printf("%02i medians:", noSections );
                      for ( int i = 0; i < noSections; i++ )
                      {
                        printf("\t%.6f", medianA[i] );
                      }
                      printf("\n");
                    }
                  }
                }
              }

              if ( ite > 40 )
              {
                printf("\nERROR: in %s. Block %02i iterated %i times and did not find the correct median.\n", __FUNCTION__, bid, ite+1 );
                found = true;
                SMIdx = MAX(0,SMIdx);
                SMIdx = MIN(len-1,SMIdx);
              }

              if (found)
              {
                FOLD // Median selection  .
                {
                  if ( (arrayLength & 1) )    // odd   .
                  {
                    medianValue = buffer[SMIdx];
                  }
                  else                        // even  .
                  {
                    // mean
                    //medianValue = ( smBuffer[SMIdx-1] + smBuffer[mIdx] ) / 2.0f;

                    // lower
                    medianValue = buffer[SMIdx-1];

                    // upper
                    //medianValue = buffer[SMIdx];
                  }
                }

                //if  ( bid == 12 )
                //  printf("FOUND\n %02i\t%22.6f\t%22.6f\t%22.6f \n", ite, lowerVal, medianValue, upperVal );

                if ( ite > 20 )
                  printf("\nFound median in block %02i after %02i iterations.\n", bid, ite );
              }
            }
          }

          ite++;
          __syncthreads();
        }
      }
    }
    __syncthreads();
  }
  else
  {
    if( tid == 0 )
    {
      printf("\nERROR: in %s number of sections (%.2f) is larger than the max number compiled with (%i).\n", __FUNCTION__, arrayLength/(float)bufferSz, maxSub );
    }
    return 0;
  }

  return medianValue;
}

template< int stride, int BS_MAX, typename stpType>
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
            fcomplexcu val  = data[idx];
            sData[idx]      = val.r*val.r+val.i*val.i;
          }
        }
      }

      medianValue = cuMedianOne(sData, width);
    }
    else
    {
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

//    if ( tid == 0 )
//    {
//      float sec = width / (float)BS_MAX ;
//      printf("%02i  batches: %4.2f %4.2f section  median %.6f  factor: %10.10f \n", bid, width / (float) bSz, sec, medianValue, factor );
//    }

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

__global__ void normAndSpread_writeOnly(fcomplexcu* data, int width)
{

  const int bid = blockIdx.y  * gridDim.x  + blockIdx.x;        /// Block ID (flat index)
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;       /// Thread ID in block (flat index)

  const int idx = bid*(blockDim.x * blockDim.y) + tid;                     /// Block size

  if (idx < width )
  {
    fcomplexcu val  = data[idx];
    val.r *= 1;
    val.i *= 1;
    data[idx]       = val;
  }

}

template<int width, int buffer>
__host__ void normAndSpread_b(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t stream, cuFFdotBatch* batch, uint stack )
{
  int harm    = 0;
  int stp     = 0;

  cuFfdotStack* cStack = &batch->stacks[stack];
  harm = cStack->startIdx;

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

    normAndSpread<width, buffer, int01><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, iLen);
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

    normAndSpread<width, buffer, int02><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, iLen);
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

    normAndSpread<width, buffer, int04><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, iLen);
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

    normAndSpread<width, buffer, int08><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, iLen);
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

    normAndSpread<width, buffer, int16><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, iLen);
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

    normAndSpread<width, buffer, int32><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, iLen);
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

    normAndSpread<width, buffer, int64><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, iLen);
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

    normAndSpread<width, buffer, int128><<< dimGrid,  dimBlock, 0, stream >>>(cStack->d_iData, iLen);
  }
  else
  {
    fprintf(stderr,"ERROR: %s has not been set up to work with %i input sections.",__FUNCTION__, noInput);
  }
}

template<int width>
__host__ void normAndSpread_w(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t stream, cuFFdotBatch* batch, uint stack )
{
  cuFfdotStack* cStack  = &batch->stacks[stack];
  int bufWidth          = MIN( globalFloat01, (cStack->width/2.0f)/globalFloat02 );

  bufWidth              = MAX( (cStack->width/2.0f)/32.0, bufWidth );
  bufWidth              = MAX( 256, bufWidth );

  switch ( bufWidth )
  {
    case 256   :
    {
      normAndSpread_b<width,256>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 512   :
    {
      normAndSpread_b<width,512>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 1024  :
    {
      normAndSpread_b<width,1024>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 2048  :
    {
      normAndSpread_b<width,2048>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 4096  :
    {
      normAndSpread_b<width,4096>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 8192  :
    {
      normAndSpread_b<width,8192>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    default    :
    {
      fprintf(stderr, "ERROR: %s has not been templated for sorting with %i elements.\n", __FUNCTION__, bufWidth );
      exit(EXIT_FAILURE);
    }
  }
}

__host__ void normAndSpread_f2(cudaStream_t stream, cuFFdotBatch* batch, uint stack )
{
  dim3 dimBlock, dimGrid;
  cuFfdotStack* cStack = &batch->stacks[stack];

  // Blocks of 1024 threads ( the maximum number of threads per block )
  dimBlock.x = NAS_DIMX;
  dimBlock.y = NAS_DIMY;
  dimBlock.z = 1;

  // One block per harmonic, thus we can sort input powers in Shared memory
  dimGrid.x = ceil(cStack->width*cStack->height*batch->noSteps / float(dimBlock.x*dimBlock.y*dimBlock.z));
  dimGrid.y = 1;

  normAndSpread_writeOnly<<<dimGrid, dimBlock, 0, stream>>>(cStack->d_iData, cStack->width*cStack->height*batch->noSteps );
}

__host__ void normAndSpread_f(cudaStream_t stream, cuFFdotBatch* batch, uint stack )
{
  //normAndSpread_f2(stream, batch, stack );
  //return;

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
      normAndSpread_w<128>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 256   :
    {
      normAndSpread_w<256>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 512   :
    {
      normAndSpread_w<512>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 1024  :
    {
      normAndSpread_w<1024>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 2048  :
    {
      normAndSpread_w<2048>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 4096  :
    {
      normAndSpread_w<4096>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 8192  :
    {
      normAndSpread_w<8192>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 16384 :
    {
      normAndSpread_w<16384>(dimGrid, dimBlock, i1, stream, batch, stack);
      break;
    }
    case 32768 :
    {
      normAndSpread_w<32768>(dimGrid, dimBlock, i1, stream, batch, stack);
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
