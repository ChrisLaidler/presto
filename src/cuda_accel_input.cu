#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"




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

__device__ void bitonicSort(float *sData, uint arrayLength, int dir)
{
  const uint noBatch = ceilf(arrayLength / 2.0 / blockDim.x);    // Number of comparisons each thread must do
  uint idx;
  const uint max = arrayLength * 2;
  uint bIdx;// The thread position in the block
  uint hSz = 1;// half block size
  uint pos1, pos2, blk;
  uint len;// The distance between items to swap
  uint bach;// The batch we are processing

  for (uint size = 2; size < max; size <<= 1)
  {
    hSz = (size >> 1);

    __syncthreads();

    for (bach = 0; bach < noBatch; bach++)
    {
      idx = (threadIdx.x + bach * blockDim.x);

      bIdx = hSz - 1 - idx % hSz;
      blk = idx / hSz;

      len = size - 1 - bIdx * 2;
      pos1 = blk * size + bIdx;
      pos2 = pos1 + len;

      if (pos2 < arrayLength)
        Comparator(sData[pos1], sData[pos2], dir);
    }

    for (len = (hSz >>= 1); len > 0; len >>= 1)
    {
      hSz = (len << 1);

      __syncthreads();
      for (bach = 0; bach < noBatch; bach++)
      {
        idx = (threadIdx.x + bach * blockDim.x);

        bIdx = idx % len;
        blk = idx / len;

        pos1 = blk * hSz + bIdx;
        pos2 = pos1 + len;

        if (pos2 < arrayLength)
          Comparator(sData[pos1], sData[pos2], dir);
      }
    }
  }

  __syncthreads();
}

__device__ uint binarySearch(volatile float* sDataGlob, uint Start, uint End, float value)
{
  uint lower = Start;
  uint upper = End;
  uint mid = (End - Start) / 2;

  // continue searching while [imin,imax] is not empty
  while (upper > lower)
  {
    float ll = sDataGlob[lower];
    float uu = sDataGlob[upper];

    // calculate the midpoint for roughly equal partition
    mid = (upper + lower) / 2;
    float mm = sDataGlob[mid];

    if (mm < value)
      lower = mid + 1;
    else
      upper = mid - 1;
  }
  float ll = sDataGlob[lower];
  float uu = sDataGlob[upper];
  float mm = sDataGlob[mid];
  return lower;
}

/** in-place bitonic sort float array in shared memory
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
__device__ void bitonicSort(float *data, const uint arrayLength, const uint trdId, const uint noThread, const int dir = 1)
{
  const uint noBatch = ceilf(arrayLength / 2.0 / noThread);     // Number of comparisons each thread must do
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

__global__ void normAndSpreadBlks(cHarmList data, iHarmList len, iHarmList widths)
{
  int blkId = blockIdx.y  * NAS_DIMY  + blockIdx.x;        // The flat ID of this block in the grid
  int trdId = threadIdx.y * NAS_DIMX  + threadIdx.x;       // The flat ID of this thread in the block

  fcomplexcu* inp   = data.val[blkId];
  uint arrayLength  = len.val[blkId];

  extern __shared__ float s[];
  float* sData  = &s[0];              // Shared memory to do the median calculation
  float* factor = &s[arrayLength+1];  // Shared memory factor for normalisation

  bool D2 = false;

  // There are a maximum of 1024 threads in a block
  // We can store 12288 floats in shared memory, thus each thread could do 12 batches of work
  // Number of threads = arrayLength/2.0 -> each thread handles 2 values at a time
  uint noBatch = ceil((float)arrayLength / (float)( NAS_NTRD * 2 ) );// Number of comparisons each thread must do
  uint idx;
  float2 bob;

  float2 hld[BS_MAX / NAS_NTRD];      // A temporary store to hole values read
  float2 *ii = (float2*) &inp[0].r;   // The data in memory

  // Load data into shared memory
  if (D2)
  {
    for (int i = 0; i < noBatch * 2; i++)
    {
      idx = trdId + NAS_NTRD * i;
      if (idx < arrayLength )
      {
        bob = ii[idx * ACCEL_NUMBETWEEN];
        sData[idx] = bob.x * bob.x + bob.y * bob.y;
      }
    }
    __syncthreads();  // Make sure we have read all the memory before we starts sorting
  }
  else
  {
    // read the data from memory, store in temporary 'hld' and store powers in shared memory
    for (int i = 0; i < noBatch * 2; i++)
    {
      idx = trdId + NAS_NTRD * i;
      if (idx < arrayLength )
      {
        bob = ii[idx];

        // Keep values
        hld[i] = bob;

        // Store powers in shared memory
        sData[idx] = bob.x * bob.x + bob.y * bob.y;
      }
    }

    // Now set the entire block of input on the device memory to 0
    {
      // Note the latency of this write will mostly be absorbed by the sort
      // It could be done with the final write back but this works as fats

      __syncthreads(); // Make sure we have read from memory before we zero it and make sure shared memory is full

      uint width = widths.val[blkId];
      uint noBatch2 = ceilf(width * 2 / NAS_NTRD);// Number of memset's each thread must do
      float *data = (float*) inp;

      for (int i = 0; i < noBatch2; i++)
      {
        idx = trdId + NAS_NTRD * i;
        if (idx < width * 2)
        {
          data[idx] = 0;
        }
      }
    }
  }

  bitonicSort(sData, arrayLength, trdId, NAS_NTRD);

  // Calculate the normalisation factor
  if ( trdId == 0 )
  {
    idx = arrayLength / 2.0;
    float medainl = -1;

    if ((arrayLength & 1))   // odd
    {
      medainl = sData[idx];
    }
    else                        //even
    {
      // mean
      //medainl = (sData[idx-1] + sData[idx])/2.0;

      // lower
      medainl = sData[idx - 1];

      // upper
      //medainl = sData[idx];
    }
    *factor = 1.0 / sqrt(medainl / log(2.0));
  }
  __syncthreads();  // Make sure all threads can see factor

  // Normalise complex numbers and write back with spread
  if (D2)
  {
    /*
    for (int i = 0; i < noBatch * 2; i++)
    {
      idx = trdId + NAS_NTRD * i;
      if (idx < arrayLength)
      {
        ii[idx * ACCEL_NUMBETWEEN].x *= *factor;
        ii[idx * ACCEL_NUMBETWEEN].y *= *factor;
      }
    }
     */
  }
  else
  {
    for (int i = 0; i < noBatch * 2; i++)
    {
      idx = trdId + NAS_NTRD * i;
      if (idx < arrayLength)
      {
        // Normalise
        hld[i].x *= *factor;
        hld[i].y *= *factor;
        //hld[0].x *= factor;
        //hld[0].y *= factor;

        // Write back to memory with spread
        ii[idx * ACCEL_NUMBETWEEN] = hld[i];
      }
    }
  }

  /*
   //uint ix = blockIdx.x * blockDim.x + threadIdx.x;
   arrayLength = widths[blkId];

   uint noBatch = ceilf(arrayLength*2 / NAS_NTRD);    // Number of comparisons each thread must do
   for (int i = 0; i < noBatch ; i++)
   {
   idx = trdId + NAS_NTRD * i;
   float n = 0;

   // This assumes ACCEL_NUMBETWEEN = 2!
   if ( idx & 2 )
   {
   n = 0;
   }
   else
   {
   if (idx & 1)
   {
   n =

   }
   }
   else


   bIdx = idx & (len-1) ;             // x mod y == x & (y-1), where y is 2^n.



   //uint mx1 = arrayLength / 2;
   if (ix < arrayLength)
   {
   if (ix < arrayLength)
   {
   dataOut[ix * ACCEL_NUMBETWEEN].r = data[ix].r * (*median);
   dataOut[ix * ACCEL_NUMBETWEEN].i = data[ix].i * (*median);
   }
   else
   {
   dataOut[ix * ACCEL_NUMBETWEEN].r = 0;
   dataOut[ix * ACCEL_NUMBETWEEN].i = 0;
   }
   dataOut[ix * ACCEL_NUMBETWEEN + 1].r = 0;
   dataOut[ix * ACCEL_NUMBETWEEN + 1].i = 0;
   }
   }
   */

  /*
   if (true)
   {
   float *ii = &inp[0].r;
   // Load data into shared memory
   for (int i = 0; i < noBatch * 4; i++)
   {
   idx = trdId + NAS_NTRD * i;
   if (idx < arrayLength * 2)
   {
   //inp[ idx ].r *= factor;
   //inp[ idx ].i *= factor;
   ii[idx] *= factor;
   }
   }
   }
   else
   {
   // Load data into shared memory
   for (int i = 0; i < noBatch * 2; i++)
   {
   idx = trdId + NAS_NTRD * i;
   if (idx < arrayLength)
   {
   inp[idx].r *= factor;
   inp[idx].i *= factor;
   //ii[idx] *= factor ;
   }
   }
   }
   */
}

__global__ void normAndSpreadBlksDevice(cHarmList readData, cHarmList writeData, iHarmList len, iHarmList widths)
{
  int blkId = blockIdx.y  * gridDim.x  + blockIdx.x;        // The flat ID of this block in the grid
  int trdId = threadIdx.y * blockDim.x + threadIdx.x;       // The flat ID of this thread in the block

  fcomplexcu* inp     = readData.val[blkId];
  //fcomplexcu* output  = readData.val[blkId];
  uint arrayLength    = len.val[blkId];

  // Set up shared memory
  extern __shared__ float s[];
  float* sData        = &s[0];              // Shared memory to do the median calculation
  float* factor       = &s[arrayLength+1];  // Shared memory factor for normalisation

  // There are a maximum of 1024 threads in a block
  // We can store 12288 floats in shared memory, thus each thread could do 12 batches of work
  // Number of threads = arrayLength/2.0 -> each thread handles 2 values at a time
  uint noBatch = ceil((float)arrayLength / (float)( NAS_NTRD * 2 ) );// Number of comparisons each thread must do
  uint idx;
  float2 bob;

  float2 hld[ BS_MAX / NAS_NTRD ];      // A temporary store to hole values read
  float2 *ii = (float2*) &inp[0].r;     // The data in memory

  // read the data from memory, store in temporary 'hld' and store powers in shared memory
  for (int i = 0; i < noBatch * 2; i++)
  {
    idx = trdId + NAS_NTRD * i;
    if (idx < arrayLength )
    {
      bob = ii[idx];

      // Keep values
      hld[i] = bob;

      // Store powers in shared memory
      sData[idx] = bob.x * bob.x + bob.y * bob.y;
    }
  }

  // Now set the entire block of device memory to 0
  {
    // Note the latency of this write will mostly be absorbed by the sort
    // It could be done with the final write back but this works as fats
    uint width = widths.val[blkId];
    uint noBatch2 = ceilf(width * 2 / NAS_NTRD);// Number of memset's each thread must do
    float *data = (float*) writeData.val[0];

    for (int i = 0; i < noBatch2; i++)
    {
      idx = trdId + NAS_NTRD * i;
      if (idx < width * 2)
      {
        data[idx] = 0;
      }
    }
  }

  bitonicSort(sData, arrayLength, trdId, NAS_NTRD);

  // Calculate the normalisation factor
  if (trdId == 0)
  {
    idx = arrayLength / 2.0;
    float medainl = -1;

    if ((arrayLength & 1))   // odd
    {
      medainl = sData[idx];
    }
    else                        //even
    {
      // mean
      //medainl = (sData[idx-1] + sData[idx])/2.0;

      // lower
      medainl = sData[idx - 1];

      // upper
      //medainl = sData[idx];
    }
    *factor = 1.0 / sqrt(medainl / log(2.0));
  }
  __syncthreads();  // Make sure all threads can see factor

  // Normalise complex numbers and write back with spread
  for (int i = 0; i < noBatch * 2; i++)
  {
    idx = trdId + NAS_NTRD * i;
    if (idx < arrayLength)
    {
      // Normalise
      hld[i].x *= *factor;
      hld[i].y *= *factor;

      // Write back to memory with spread
      ii[idx * ACCEL_NUMBETWEEN] = hld[i];
    }
  }
}

__global__ void median1Block(const float *data, uint arrayLength, float *median, uint noBatch)
{

  //const int trdId = threadIdx.y * blockDim.x + threadIdx.x; // The flat ID of this thread in the block
  //int noThread = blockDim.x * blockDim.y;                 // The number of threads in a block
  //const int noThread = 1024;


  __shared__ float sData[BS_MAX];                   // Shared memory to do the calculation

  // There are a maximum of 1024 threads in a block
  // We can store 12288 floats in shared memory, thus each thread could to 12 batches of work
  // Number of threads = arrayLength/2.0 -> each thread handles 2 values at a time

  //uint noBatch = ceilf(arrayLength / 2.0 / blockDim.x);    // Number of comparisons each thread must do
  //   uint x;
  uint idx;
  int dir = 1;

  //idx = threadIdx.x;
  //data[idx] = noBatch;

  // Load data into shared memory
  for (int i = 0; i < noBatch * 2; i++)
  {
    idx = threadIdx.x + blockDim.x * i;

    if (idx < arrayLength)
    {
      sData[idx] = data[idx];
    }
  }

  //__syncthreads();

  //bitonicSort(sData, arrayLength, 1);

  uint max = arrayLength * 2;
  uint bIdx;// The thread position in the block
  uint hSz = 1;// half block size
  uint pos1, pos2, blk;
  uint len;// The distance between items to swap
  //uint bach;                      // The batch we are processing

  for (uint size = 2; size < max; size <<= 1)
  {
    hSz = (size >> 1);

    __syncthreads();
    for (int bach = 0; bach < noBatch; bach++)
    {
      idx = (threadIdx.x + bach * blockDim.x);

      bIdx = hSz - 1 - idx % hSz;
      blk = idx / hSz;

      len = size - 1 - bIdx * 2;
      pos1 = blk * size + bIdx;
      pos2 = pos1 + len;

      if (pos2 < arrayLength)
        Comparator(sData[pos1], sData[pos2], dir);
    }

    for (len = (hSz >>= 1); len > 0; len >>= 1)
    {
      hSz = (len << 1);
      __syncthreads();

      for (int bach = 0; bach < noBatch; bach++)
      {
        idx = (threadIdx.x + bach * blockDim.x);

        bIdx = idx % len;
        blk = idx / len;

        pos1 = blk * hSz + bIdx;
        pos2 = pos1 + len;

        if (pos2 < arrayLength)
          Comparator(sData[pos1], sData[pos2], dir);
      }
    }
  }

  __syncthreads();

  if (threadIdx.x == 0)
  {

    idx = arrayLength / 2.0;
    double medainl = -1;

    if ((arrayLength & 1))   // odd
    {
      medainl = sData[idx];
    }
    else                        //even
    {
      // mean
      //medainl = (sData[idx - 1] + sData[idx]) / 2.0;

      // lower
      medainl = sData[idx-1];

      // upper
      //medainl = sData[idx];
    }

    medainl = 1.0 / sqrt(medainl / log(2.0));
    *median = medainl;
  }
}

__global__ void sortNBlock(float *data, uint arrayLength, float *output, int dir = 1)
{
  __shared__ float sData[BS_MAX];                           // Shared memory to do the calculation

  //float noB = arrayLength / 2.0 / (blockDim.x * gridDim.x);
  uint noBatch = ceilf(arrayLength / 2.0 / (blockDim.x * gridDim.x));// The number of comparisons each thread must perform at each step
  uint bachLen = noBatch * blockDim.x * 2;// Number of keys sorted by a thread block (each thread counts for two numbers)

  uint gIdx;// Global data index
  uint bIdx;// Thread block index
  uint bblen = bachLen;

  // Load data into shared memory
  for (int i = 0; i < noBatch * 2; i++)
  {
    bIdx = threadIdx.x + blockDim.x * i;
    gIdx = bIdx + blockIdx.x * bachLen;

    if (gIdx < arrayLength)
      sData[bIdx] = data[gIdx];
  }

  // Set bachLen for last block
  if ((gridDim.x - 1) == blockIdx.x)
    bachLen = arrayLength - bachLen * blockIdx.x;

  bitonicSort(sData, bachLen, dir);

  // Load data back into main memory
  for (int i = 0; i < noBatch * 2; i++)
  {
    bIdx = threadIdx.x + blockDim.x * i;
    gIdx = bIdx + blockIdx.x * bblen;

    if (gIdx < arrayLength)
      data[gIdx] = sData[bIdx];
  }

  if (threadIdx.x == 0)
  {
    float max = sData[bachLen - 1];
    float min = sData[0];
    float median = sData[(bachLen - 1) / 2];

    output[blockIdx.x] = median;
    output += gridDim.x;
    output[blockIdx.x] = min;
    output += gridDim.x;
    output[blockIdx.x] = max;

    //printf("Block %03i found %14.3f\n                %14.3f\n                %14.3f\n", blockIdx.x, max, min, median);
  }

}

__global__ void selectMedianCands(float *data, uint arrayLength, float *output, int dir = 1)
{
  __shared__ uint dist;
  __shared__ uint lower;
  __shared__ uint upper;

  __shared__ float sData[BS_MAX];             // Shared memory to do the calculation

  //float noB = arrayLength / 2.0 / (blockDim.x * gridDim.x);
  uint noBatch = ceilf(arrayLength / 2.0 / (blockDim.x * gridDim.x));// The number of comparisons each thread must perform at each step
  uint bachLen = noBatch * blockDim.x * 2;// Number of keys sorted by a thread block (each thread counts for two numbers)
  uint bblen = bachLen;

  uint gIdx;
  uint bIdx;

  //data += blockIdx.x*bachLen ;

  // Load data into shared memory
  for (int i = 0; i < noBatch * 2; i++)
  {
    bIdx = threadIdx.x + blockDim.x * i;
    gIdx = blockIdx.x * bblen + bIdx;

    if (gIdx < arrayLength)
      sData[bIdx] = data[gIdx];
  }

  if ((gridDim.x - 1) == blockIdx.x)
    bachLen = arrayLength - bachLen * blockIdx.x;

  __syncthreads();

  float max;
  float min;

  if (threadIdx.x == 0)
  {
    max = output[0];
    min = output[0];
    for (int i = 1; i < gridDim.x; i++)
    {
      if (output[i] < min)
        min = output[i];

      if (output[i] > max)
        max = output[i];

    }
    lower = binarySearch(sData, 0, bachLen - 1, min);
    upper = binarySearch(sData, 0, bachLen - 1, max);
    dist = upper - lower;

    //printf("Block %02i is %04i  min %10.2f  max %10.2f  %05i %05i\n", blockIdx.x, dist, min, max, lower, upper);

    output += gridDim.x * 3;// Skip previous values (median min max)

    // Number of items in this list
    output[blockIdx.x] = dist;
    output += gridDim.x;

    // Number of items below the list
    output[blockIdx.x] = lower;
    output += gridDim.x;

    // Number of items above the list
    output[blockIdx.x] = upper;
  }

  __syncthreads();

  // Load data into memory
  for (int i = 0; i < noBatch * 2; i++)
  {
    bIdx = threadIdx.x + blockDim.x * i;
    gIdx = blockIdx.x * bblen + bIdx;

    if (gIdx < arrayLength && bIdx >= lower && bIdx <= upper)
      data[gIdx - lower] = sData[bIdx];
  }
}

__global__ void medFromMedians(float *data, uint arrayLength, float *output, int noSections, float *median, int dir = 1)
{
  __shared__ uint dist[100];
  __shared__ uint lower[100];
  //__shared__ uint upper[100];

  __shared__ float sData[BS_MAX];// Shared memory to do the calculation

  __shared__ uint length;

  uint noBatch = ceilf(arrayLength / 2.0 / (blockDim.x * noSections));// The number of comparisons each thread must perform at each step
  uint bachLen = noBatch * blockDim.x * 2;// Number of keys sorted by a thread block (each thread counts for two numbers)
  //uint bblen = bachLen;

  //uint noBatch  = 1;
  //uint bachLen  = arrayLength / blockDim.x*2 / noSections;         // Number of keys sorted by a thread block (each thread counts for two numbers)

  uint gIdx;
  uint bIdx;

  output += noSections * 3;// skip forward to output

  if (threadIdx.x == 0)
  {
    dist[0] = 0;
    int noPoints = 0;
    for (int i = 0; i < noSections; i++)
    {

      dist[i] = noPoints;
      noPoints += output[i];

      //printf("Block %02i is %i\n", i, output[i] );

      // Lower
      lower[i] = bachLen * i + output[noSections + i];

      //output +=  noSections;
      //upper[i] = bachLen * i + output[noSections * 2 + i];
    }
    length = noPoints;

    if (noPoints >= BS_MAX)
      printf("ERROR: error in CUDA finding meadian, number of points wont fit in Shared Memeor!\n");
  }

  __syncthreads();

  //float bb = length / blockDim.x / 2.0;
  noBatch = ceilf(length / blockDim.x / 2.0);

  //data += blockIdx.x*bachLen ;

  // Load data into shared memory
  for (int i = 0; i < noBatch * 2; i++)
  {
    bIdx = threadIdx.x + blockDim.x * i;
    gIdx = blockIdx.x * bachLen + bIdx;

    int read = -1;
    int write = -1;

    for (int i = 0; i < noSections; i++)
    {
      if (gIdx >= dist[i] && gIdx < dist[i + 1])
      {
        read = lower[i] + gIdx - dist[i];
        write = gIdx;
      }
    }

    if (read > 0)
    {
      sData[write] = data[read];
    }
  }

  __syncthreads();

  bitonicSort(sData, length, dir);

  __syncthreads();

  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    int below = 0;
    for (int i = 0; i < noSections; i++)
    {
      below += output[noSections + i];
    }

    int medianPosGlobal = (arrayLength) / 2;

    int medianPos = (arrayLength) / 2 - below;

    float medainl;

    if ((medianPosGlobal & 1))   // odd
    {
      medainl = sData[medianPos];
    }
    else                        //even
    {
      // mean
      medainl = (sData[medianPos - 1] + sData[medianPos]) / 2.0;

      // lower
      //medainl = sData[idx-1];

      // upper
      //medainl = sData[idx];
    }
    *median = 1.0 / sqrt(medainl / log(2.0));
    //printf("Median is normalization factor is %15.10f  median:%f\n",*median, medainl );
  }
}

__global__ void normAndSpread(fcomplexcu *data, uint arrayLength, fcomplexcu *dataOut, uint maxSpread)
{
  __shared__ float sData[BS_MAX];                           // Shared memory to do the calculation
  __shared__ float median;// Shared memory to do the calculation

  //float noB = arrayLength / 2.0 / (blockDim.x * gridDim.x);
  const uint noBatch = ceilf(arrayLength / 2.0 / (blockDim.x * gridDim.x));// The number of comparisons each thread must perform at each step
  const uint bachLen = noBatch * blockDim.x * 2;// Number of keys sorted by a thread block (each thread counts for two numbers)

  uint gIdx;// Global data index
  uint bIdx;// Thread block index
  //uint bblen = bachLen;

  const uint mx1 = maxSpread / 2;

  // Load data into shared memory
  for (int i = 0; i < noBatch * 2; i++)
  {
    bIdx = threadIdx.x + blockDim.x * i;
    gIdx = bIdx + blockIdx.x * bachLen;

    if (gIdx < arrayLength)
    {
      sData[bIdx] = data[bIdx].r * data[bIdx].r + data[bIdx].i * data[bIdx].i;

      //float r = data[bIdx].r;
      //float i = data[bIdx].i;
      //sData[bIdx] = r*r + i*i ;
    }
  }

  // Sort in shared memory
  bitonicSort(sData, arrayLength, 1);

  // Find median in sorted data
  if (threadIdx.x == 0)
  {
    bIdx = arrayLength / 2.0;
    float medainl = -1;

    if ((arrayLength & 1))   // odd
    {
      medainl = sData[bIdx];
    }
    else                        //even
    {
      // mean
      medainl = (sData[bIdx - 1] + sData[bIdx]) / 2.0;

      // lower
      //medainl = sData[idx-1];

      // upper
      //medainl = sData[idx];
    }

    median = 1.0 / sqrt(medainl / logf(2.0));
  }

  __syncthreads();

  // Copy back to main memory
  for (int i = 0; i < noBatch * 2; i++)
  {
    bIdx = threadIdx.x + blockDim.x * i;
    gIdx = bIdx + blockIdx.x * bachLen;
    if (gIdx < mx1)
    {
      if (gIdx < arrayLength)
      {
        dataOut[gIdx * ACCEL_NUMBETWEEN].r = data[gIdx].r * median;
        dataOut[gIdx * ACCEL_NUMBETWEEN].i = data[gIdx].i * median;
      }
      else
      {
        dataOut[gIdx * ACCEL_NUMBETWEEN].r = 0;
        dataOut[gIdx * ACCEL_NUMBETWEEN].i = 0;
      }
      dataOut[gIdx * ACCEL_NUMBETWEEN + 1].r = 0;
      dataOut[gIdx * ACCEL_NUMBETWEEN + 1].i = 0;
    }
  }
}

__global__ void calculatePowers(fcomplexcu *data, float* powers, uint arrayLength)
{
  uint ix = blockIdx.x * blockDim.x+ threadIdx.x;
  if (ix < arrayLength)
  {
    float r = data[ix].r;
    float i = data[ix].i;
    powers[ix] = r * r + i * i;
  }
}

__global__ void devideAndSpreadFFT(fcomplexcu *data, uint arrayLength, fcomplexcu *dataOut, uint maxSpread, float *median)
{
  uint ix = blockIdx.x * blockDim.x + threadIdx.x;
  uint mx1 = maxSpread / 2;
  if (ix < mx1)
  {
    if (ix < arrayLength)
    {
      dataOut[ix * ACCEL_NUMBETWEEN].r = data[ix].r * (*median);
      dataOut[ix * ACCEL_NUMBETWEEN].i = data[ix].i * (*median);
    }
    else
    {
      dataOut[ix * ACCEL_NUMBETWEEN].r = 0;
      dataOut[ix * ACCEL_NUMBETWEEN].i = 0;
    }
    dataOut[ix * ACCEL_NUMBETWEEN + 1].r = 0;
    dataOut[ix * ACCEL_NUMBETWEEN + 1].i = 0;
  }
}

__global__ void chopAndpower(fcomplexcu *ffdot, uint width, uint strideFfdot, uint height, float *ffdotPowers, uint stridePowers, uint chopBefore, uint length)
{
  uint pix = blockIdx.x * blockDim.x + threadIdx.x;
  uint piy = blockIdx.y * blockDim.y + threadIdx.y;

  if (pix < length && piy < height)
  {
    uint fidx = piy * strideFfdot + pix + chopBefore;
    uint pidx = piy * stridePowers + pix;

    fcomplexcu cmp = ffdot[fidx];

    ffdotPowers[pidx] = cmp.r * cmp.r + cmp.i * cmp.i;

    //    ffdotPowers[pidx] = 0;
  }
}

__global__ void sumPlains(float* fund, int fWidth, int fStride, int fHeight, float* sub, int sWidth, int sStride, int sHeight, float frac, float fRlow, float fZlow, float sRlow, float sZlow)
{
  //__shared__ int indsX[ACCEL_USELEN];
  //__shared__ int indsY[ACCEL_USELEN];

  int ix = (blockIdx.x * blockDim.x + threadIdx.x);
  int iy = (blockIdx.y * blockDim.y + threadIdx.y);

  //int idx = iy * fStride + iy;

  //int bidx;
  int thredsinBlock = (blockDim.x * blockDim.y);
  int batches = ceilf(fWidth / (float) thredsinBlock);

  /*
   for ( int i = 0; i < batches; i++ )
   {
   bidx = i*thredsinBlock + threadIdx.y*blockDim.x + threadIdx.x;

   if ( bidx < fWidth )
   {
   int rr = fRlow + bidx * ACCEL_DR;
   int subr = calc_required_r_gpu(frac, rr);
   indsX[bidx] = index_from_r(subr, sRlow);
   }

   if ( bidx < fHeight )
   {
   int zz = fZlow + bidx * ACCEL_DZ;
   int subz  = calc_required_z(frac, zz);
   indsY[bidx] = index_from_z(subz, sZlow);
   }
   }
   __syncthreads();
   */

  if (ix < fWidth && iy < fHeight)
  {
    int rr = fRlow + ix * ACCEL_DR;
    int subr = calc_required_r_gpu(frac, rr);
    int isx = index_from_r(subr, sRlow);

    //int zz    = fZlow + (fHeight-1-iy) * ACCEL_DZ;
    int zz = fZlow + iy * ACCEL_DZ;
    int subz = calc_required_z(frac, zz);
    int isy = index_from_z(subz, sZlow);

    fund[iy * fStride + ix] += sub[isy * sStride + isx];
  }
}



float cuGetMedian(float *data, uint len)
{
  dim3 dimBlock, dimGrid;
  float* dArrayA = NULL;
  cudaError_t result;

  //cudaMalloc ( ( void ** ) &dArrayA, (maxZ*2+1) * fftlen * sizeof ( float )*2 );
  CUDA_SAFE_CALL(cudaMalloc((void ** ) &dArrayA, (len+ 1)* sizeof(float)), "Failed to allocate device memory for.");
  //__cuSafeCall    (cudaMalloc((void ** ) &dArrayA, (len+ 1)* sizeof(float)), __FILE__, __LINE__, "Failed to allocate device memory for." ) ;
  CUDA_SAFE_CALL(cudaMemcpy(dArrayA, data, len* sizeof(float), cudaMemcpyHostToDevice), "Failed to copy data to device");

  if (len< 49152/ sizeof(float))
  {
    //uint blockSz = 5;
    uint blockSz = BS_DIM;
    //blockSz = 5;

    if (len/ 2.0< blockSz)
      dimBlock.x = ceil(len/ 2.0);
    else
      dimBlock.x = blockSz;

    dimGrid.x = 1;

    uint noBatch = ceilf(len/ 2.0/ dimBlock.x);    // Number of comparisons each thread must do

    //printf ( "Calling kernel %i %i %i (%i %i %i)\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z );
    median1Block<<<dimGrid, dimBlock>>>(dArrayA, len, &dArrayA[len], noBatch);

    // Run message
    {
      result = cudaGetLastError();  // This determines whether the kernel was launched

      if (result== cudaSuccess)
      {
        //printf ( "Running kernel ..." );
      }
      else
      {
        fprintf(stderr, "ERROR: Error at kernel launch %s\n", cudaGetErrorString(result));
        exit(EXIT_FAILURE);
      }
    }

    {
      result = cudaDeviceSynchronize();  // This will return when the kernel computation is complete, remember asynchronous execution
      // Complete message;

      if (result== cudaSuccess)
      {
        //printf ( " Complete.\n" );
      }
      else
        fprintf(stderr, "\nERROR: Error after kernel launch %s\n", cudaGetErrorString(result));
    }
    float result;
    CUDA_SAFE_CALL(cudaMemcpy(&result, &dArrayA[len], sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy data back from device");

    return result;
  }
  return 0;
}

void CPU_Norm_Spread(cuFFdotBatch* batch, double* searchRLow, double* searchRHi, int norm_type, fcomplexcu* fft)
{
  nvtxRangePush("CPU_Norm_Spread");

  int harm = 0;

  FOLD // Copy raw input fft data to device  .
  {
    for (int stack = 0; stack < batch->noStacks; stack++)
    {
      cuFfdotStack* cStack = &batch->stacks[stack];
      int sz = 0;
      for (int si = 0; si < cStack->noInStack; si++)
      {
        cuHarmInfo* cHInfo  = &batch->hInfos[harm];      // The current harmonic we are working on

        for (int step = 0; step < batch->noSteps; step++)
        {
          if ( !(searchRLow[step] == 0 &&  searchRHi[step] == 0) )
          {
            rVals* rVal = &((*batch->rInput)[step][harm]);

            if ( norm_type== 0 )  // Normal normalise  .
            {
              double norm;    /// The normalising factor

              //nvtxRangePush("Powers");
              for (int ii = 0; ii < rVal->numdata; ii++)
              {
                if ( rVal->lobin+ii < 0 || rVal->lobin+ii  >= batch->SrchSz->searchRHigh ) // Zero Pad
                {
                  batch->h_powers[ii] = 0;
                }
                else
                {
                  batch->h_powers[ii] = POWERCU(fft[rVal->lobin+ii].r, fft[rVal->lobin+ii].i);
                }
              }
              //nvtxRangePop();

              if ( DBG_INP01 )
              {
                float* data = (float*)&fft[rVal->lobin];
                int gx;
                printf("\nGPU Input Data RAW FFTs [ Half width: %i  lowbin: %i  drlo: %.2f ] \n", cHInfo->halfWidth, rVal->lobin, rVal->drlo);

                for ( gx = 0; gx < 10; gx++)
                  printf("%.4f ",((float*)data)[gx]);
                printf("\n");
              }

              //nvtxRangePush("Median");
              norm = 1.0 / sqrt(median(batch->h_powers, (rVal->numdata))/ log(2.0));                       /// NOTE: This is the same method as CPU version
              //norm = 1.0 / sqrt(median(&plains->h_powers[start], (rVal->numdata-start))/ log(2.0));       /// NOTE: This is a slightly better method (in my opinion)
              //nvtxRangePop();

              // Normalise and spread
              //nvtxRangePush("Write");
              for (int ii = 0; ii < rVal->numdata && ii * ACCEL_NUMBETWEEN < cStack->inpStride; ii++)
              {
                if ( rVal->lobin+ii < 0  || rVal->lobin+ii  >= batch->SrchSz->searchRHigh )  // Zero Pad
                {
                  cStack->h_iData[sz + ii * ACCEL_NUMBETWEEN].r = 0;
                  cStack->h_iData[sz + ii * ACCEL_NUMBETWEEN].i = 0;
                }
                else
                {
                  if (ii * ACCEL_NUMBETWEEN > cStack->inpStride)
                  {
                    fprintf(stderr, "ERROR: nice_numdata is greater that width.\n");
                    exit(EXIT_FAILURE);
                  }

                  cStack->h_iData[sz + ii * ACCEL_NUMBETWEEN].r = fft[rVal->lobin+ ii].r * norm;
                  cStack->h_iData[sz + ii * ACCEL_NUMBETWEEN].i = fft[rVal->lobin+ ii].i * norm;
                }
              }
              //nvtxRangePop();
            }
            else                  // or double-tophat normalisation
            {
              int nice_numdata = next2_to_n_cu(rVal->numdata);  // for FFTs

              if ( nice_numdata > cStack->width )
              {
                fprintf(stderr, "ERROR: nice_numdata is greater that width.\n");
                //exit(EXIT_FAILURE);
              }

              // Do the actual copy
              //memcpy(batch->h_powers, &fft[lobin], numdata * sizeof(fcomplexcu) );

              //  new-style running double-tophat local-power normalization
              float *loc_powers;

              //powers = gen_fvect(nice_numdata);
              for (int ii = 0; ii< nice_numdata; ii++)
              {
                batch->h_powers[ii] = POWERCU(fft[rVal->lobin+ii].r, fft[rVal->lobin+ii].i);
              }
              loc_powers = corr_loc_pow(batch->h_powers, nice_numdata);

              //memcpy(&batch->h_iData[sz], &fft[lobin], nice_numdata * sizeof(fcomplexcu) );

              for (int ii = 0; ii < rVal->numdata; ii++)
              {
                float norm = invsqrt(loc_powers[ii]);

                batch->h_iData[sz + ii * ACCEL_NUMBETWEEN].r = fft[rVal->lobin+ ii].r* norm;
                batch->h_iData[sz + ii * ACCEL_NUMBETWEEN].i = fft[rVal->lobin+ ii].i* norm;
              }

              vect_free(loc_powers);  // I hate doing this!!!
            }

            // I tested doing the FFT's on the CPU and its drastically faster doing it on the GPU, and can often be done synchronously -- Chris L
            //nvtxRangePush("CPU FFT");
            //COMPLEXFFT((fcomplex *)&plains->h_iData[sz], numdata*ACCEL_NUMBETWEEN, -1);
            //nvtxRangePop();
          }

          sz += cStack->inpStride;
        }
        harm++;
      }
    }
  }

  nvtxRangePop();
}

/** Calculate the r bin values for this batch of steps and store them in plains->rInput
 *
 * This calculates r-low and halfwidth
 *
 * @param batch the batch to work with
 * @param searchRLow an array of the step r-low values
 * @param searchRHi an array of the step r-high values
 */
void setStackRVals(cuFFdotBatch* batch, double* searchRLow, double* searchRHi)
{
  //printf("setStackRVals\n");

  int       hibin, binoffset;
  double    drlo, drhi;

  int lobin;      /// The first bin to copy from the the input fft ( serachR scaled - halfwidth )
  int numdata;    /// The number of input fft points to read
  int numrs;      /// The number of good bins in the plain ( expanded units )

  //printf("                      |                       |                        |                       |                      \n" );

  for (int harm = 0; harm < batch->noHarms; harm++)
  {
    cuHarmInfo* cHInfo    = &batch->hInfos[harm];      // The current harmonic we are working on
    binoffset             = cHInfo->halfWidth;          //

    for (int step = 0; step < batch->noSteps; step++)
    {
      rVals* rVal         = &((*batch->rInput)[step][harm]);

      drlo                = calc_required_r_gpu(cHInfo->harmFrac, searchRLow[step]);
      drhi                = calc_required_r_gpu(cHInfo->harmFrac, searchRHi[step] );

      lobin               = (int) floor(drlo) - binoffset;
      hibin               = (int) ceil(drhi)  + binoffset;

      numdata             = hibin - lobin + 1;
      numrs               = (int) ((ceil(drhi) - floor(drlo)) * ACCEL_RDR + DBLCORRECT) + 1;

      if ( harm == 0 )
        numrs = batch->accelLen;
      else if ( numrs % ACCEL_RDR )
        numrs = (numrs / ACCEL_RDR + 1) * ACCEL_RDR;

      rVal->drlo          = drlo;
      rVal->lobin         = lobin;
      rVal->numrs         = numrs;
      rVal->numdata       = numdata;
      rVal->expBin        = (lobin+binoffset)*ACCEL_RDR;

      if( step == 0 ) // This is only for debug purposes
      {
        double ExBin      = searchRLow[step]*((float)ACCEL_RDR)*cHInfo->harmFrac ;
        double ExBinR     = floor( ExBin / 2.0) * 2.0 ;
        double BsBinR     = ExBinR / 2.0 ;

        //printf("searchR: %11.2f  |   drlo: %11.2f   |   ExBin: %11.2f   |   ExBinR: %9.0f   |   BsBinR: %11lli\n", searchRLow[step]*cHInfo->harmFrac, drlo, ExBin, ExBinR, rVal->expBin   );
      }
    }
  }
  //printf("                      |                       |                        |                       |                      \n" );
  //TMP
}

/** Initialise input data for a f-∂f plain(s)  ready for convolution  .
 * This:
 *  Normalises the chunk of input data
 *  Spreads it (interbinning)
 *  FFT it ready for convolution
 *
 * @param plains      The plains
 * @param searchRLow  The index of the low  R bin (1 value for each step)
 * @param searchRHi   The index of the high R bin (1 value for each step)
 * @param norm_type   The type of normalisation to perform
 * @param fft         The fft
 */
void initInput(cuFFdotBatch* batch, double* searchRLow, double* searchRHi, int norm_type, fcomplexcu* fft)
{
  iHarmList lengths;
  iHarmList widths;
  cHarmList d_iDataList;
  cHarmList d_fftList;
  dim3 dimBlock, dimGrid;


#ifdef TIMING // Timing

  if ( batch->haveSearchResults )
  {
    // Make sure the previous thread has complete reading from page locked memory
    CUDA_SAFE_CALL(cudaEventSynchronize(batch->iDataCpyComp), "ERROR: Synchronising before writing input data to page locked host memory.");

    float time;         // Time in ms of the thing
    cudaError_t ret;    // Return status of cudaEventElapsedTime

    //cudaError_t stxcef = cudaEventQuery( batch->iDataCpyComp );

    FOLD // Copy input data  .
    {
      ret = cudaEventElapsedTime(&time, batch->iDataCpyInit, batch->iDataCpyComp);

      if ( ret == cudaErrorNotReady )
      {
        //printf("Not ready\n");
      }
      else
      {
        //printf("    ready\n");
#pragma omp atomic
        batch->copyH2DTime[0] += time;
      }
    }

    FOLD // Input FFT timing  .
    {
      for (int ss = 0; ss < batch->noStacks; ss++)
      {
        cuFfdotStack* cStack = &batch->stacks[ss];

        ret = cudaEventElapsedTime(&time, cStack->inpFFTinit, cStack->prepComp);
        if ( ret == cudaErrorNotReady )
        {
          //printf("Not ready\n");
        }
        else
        {
          //printf("    ready\n");
#pragma omp atomic
          batch->InpFFTTime[ss] += time;
        }
      }
    }

  }
#endif


  if ( searchRLow[0] < searchRHi[0] ) // This is real data ie this isn't just a call to finish off asynchronous work
  {
    nvtxRangePush("Input");
    //printf("Input\n");

    setStackRVals(batch, searchRLow, searchRHi );

    FOLD  // Normalise and spread and copy to device memory  .
    {
      if      ( batch->flag & CU_INPT_SINGLE_G  )
      {
        // Copy chunks of FFT data and normalise and spread using the GPU

        if ( batch->noSteps > 1 ) // TODO: multi step
        {
          fprintf(stderr,"ERROR: CU_INPT_SINGLE_G has not been set up for multi-step.");
          exit(EXIT_FAILURE);
        }

        // Make sure the previous thread has complete reading from page locked memory
        CUDA_SAFE_CALL(cudaEventSynchronize(batch->iDataCpyComp), "ERROR: copying data to device");


        nvtxRangePush("Zero");
        memset(batch->h_iData, 0, batch->inpDataSize*batch->noSteps);
        nvtxRangePop();

        FOLD // Copy fft data to device
        {
          for (int step = 0; step < batch->noSteps; step++)
          {
            int harm = 0;
            int sz = 0;

            // Write fft data segments to contiguous page locked memory
            for (int stack = 0; stack< batch->noStacks; stack++)
            {
              cuFfdotStack* cStack = &batch->stacks[stack];

              for (int si = 0; si< cStack->noInStack; si++)
              {
                cuHarmInfo* cHInfo = &batch->hInfos[harm];  // The current harmonic we are working on
                cuFFdot* cPlain = &batch->plains[harm];     //

                rVals* rVal = &((*batch->rInput)[step][harm]);

                lengths.val[harm]       = rVal->numdata;
                d_iDataList.val[harm]   = cPlain->d_iData;
                widths.val[harm]        = cStack->width;

                int start = 0;
                if ( rVal->lobin < 0 )
                  start = -rVal->lobin;

                // Do the actual copy
                memcpy(&batch->h_iData[sz+start], &fft[rVal->lobin+start], (rVal->numdata-start)* sizeof(fcomplexcu));

                sz += cStack->inpStride;

                harm++;
              }
            }

            // Synchronisation
            for (int stack = 0; stack < batch->noStacks; stack++)
            {
              cuFfdotStack* cStack = &batch->stacks[stack];
              CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, cStack->convComp, 0), "ERROR: waiting for GPU to be ready to copy data to device\n");
            }

            // Copy to device
            CUDA_SAFE_CALL(cudaMemcpyAsync(batch->d_iData, batch->h_iData, batch->inpDataSize, cudaMemcpyHostToDevice, batch->inpStream), "Failed to copy data to device");

            // Synchronisation
            cudaEventRecord(batch->iDataCpyComp, batch->inpStream);

            CUDA_SAFE_CALL(cudaGetLastError(), "Copying a section of input FTD data to the device.");
          }
        }

        FOLD // Normalise and spread
        {
          // Blocks of 1024 threads ( the maximum number of threads per block )
          dimBlock.x = NAS_DIMX;
          dimBlock.y = NAS_DIMY;
          dimBlock.z = 1;

          // One block per harmonic, thus we can sort input powers in Shared memory
          dimGrid.x = batch->noHarms;
          dimGrid.y = 1;

          // Synchronisation
          CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, batch->iDataCpyComp, 0), "ERROR: waiting for GPU to be ready to copy data to device\n");

          // Call the kernel to normalise and spread the input data
          normAndSpreadBlks<<<dimGrid, dimBlock, (lengths.val[0]+1)*sizeof(float), batch->inpStream>>>(d_iDataList, lengths, widths);

          // Synchronisation
          cudaEventRecord(batch->normComp, batch->inpStream);

          CUDA_SAFE_CALL(cudaGetLastError(), "Calling the normalisation and spreading kernel.");
        }
      }
      else if ( batch->flag & CU_INPT_HOST      )
      {
        // Copy chunks of FFT data and normalise and spread using the GPU

        if ( batch->noSteps > 1 ) // TODO: multi step
        {
          fprintf(stderr,"ERROR: CU_INPT_HOST has not been set up for multi-step.");
          exit(EXIT_FAILURE);
        }

        // Make sure the previous thread has complete reading from page locked memory
        CUDA_SAFE_CALL(cudaEventSynchronize(batch->iDataCpyComp), "ERROR: copying data to device");
        //nvtxRangePush("Zero");
        //memset(plains->h_iData, 0, plains->inpDataSize);
        CUDA_SAFE_CALL(cudaMemsetAsync(batch->d_iData, 0, batch->inpDataSize*batch->noSteps, batch->inpStream),"Initialising input data to 0");
        //nvtxRangePop();

        FOLD // Copy fft data to device
        {
          int harm = 0;
          int sz = 0;

          int step = 0; // TODO mylti-step

          // Write fft data segments to contiguous page locked memory
          for (int ss = 0; ss< batch->noStacks; ss++)
          {
            cuFfdotStack* cStack = &batch->stacks[ss];

            // Synchronisation
            CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, cStack->convComp, 0), "ERROR: waiting for GPU to be ready to copy data to device\n");

            for (int si = 0; si< cStack->noInStack; si++)
            {
              cuHarmInfo* cHInfo  = &batch->hInfos[harm];  // The current harmonic we are working on
              cuFFdot*    cPlain  = &batch->plains[harm];  //
              rVals*      rVal    = &((*batch->rInput)[step][harm]);

              /*
              drlo = calc_required_r_gpu(cHInfo->harmFrac, searchRLow[0]);
              drhi = calc_required_r_gpu(cHInfo->harmFrac, searchRHi[0]);

              binoffset = cHInfo->halfWidth;
              lobin     = (int) floor(drlo) - binoffset;
              hibin     = (int)  ceil(drhi) + binoffset;
              numdata   = hibin - lobin + 1;

              numrs     = (int) ((ceil(drhi) - floor(drlo)) * ACCEL_RDR + DBLCORRECT) + 1;
              if (harm == 0)
              {
                numrs = batch->accelLen;
              }
              else if (numrs % ACCEL_RDR)
                numrs = (numrs / ACCEL_RDR + 1) * ACCEL_RDR;
              int numtocopy = cHInfo->width - 2 * cHInfo->halfWidth * ACCEL_NUMBETWEEN;
              if (numrs < numtocopy)
                numtocopy = numrs;
               */

              lengths.val[harm]       = rVal->numdata;
              d_iDataList.val[harm]   = cPlain->d_iData;
              widths.val[harm]        = cStack->width;

              int start = 0;

              if ( (rVal->lobin - batch->SrchSz->rLow)  < 0 )
              {
                // This should be unnecessary as rLow can be < 0 and h_iData is zero padded
                start = -(rVal->lobin - batch->SrchSz->rLow);
                CUDA_SAFE_CALL(cudaMemsetAsync(cPlain->d_iData, 0, start*sizeof(fcomplexcu), batch->inpStream),"Initialising input data to 0");
              }

              // Copy section to device
              CUDA_SAFE_CALL(cudaMemcpyAsync(&cPlain->d_iData[start], &batch->h_iData[rVal->lobin-batch->SrchSz->rLow+start], (rVal->numdata-start)*sizeof(fcomplexcu), cudaMemcpyHostToDevice, batch->inpStream), "Failed to copy data to device");

              sz += cStack->inpStride;

              if ( DBG_INP01 ) // Print debug info
              {
                printf("\nCPU Input Data RAW FFTs [ Half width: %i  lowbin: %i  drlo: %.2f ] \n", cHInfo->halfWidth, rVal->lobin, rVal->drlo);

                //printfData<<<1,1,0,batch->inpStream>>>((float*)cPlain->d_iData,10,1, cStack->inpStride);
                CUDA_SAFE_CALL(cudaStreamSynchronize(batch->inpStream),"");
              }

              harm++;
            }
          }

          // Synchronisation
          //cudaEventRecord(plains->iDataCpyComp, batch->inpStream);

          CUDA_SAFE_CALL(cudaGetLastError(), "Copying a section of input FTD data to the device.");
        }

        FOLD // Normalise and spread
        {
          // Blocks of 1024 threads ( the maximum number of threads per block )
          dimBlock.x = NAS_DIMX;
          dimBlock.y = NAS_DIMY;
          dimBlock.z = 1;

          // One block per harmonic, thus we can sort input powers in Shared memory
          dimGrid.x = batch->noHarms;
          dimGrid.y = 1;

          // Call the kernel to normalise and spread the input data
          normAndSpreadBlks<<<dimGrid, dimBlock, (lengths.val[0]+1)*sizeof(float), batch->inpStream>>>(d_iDataList, lengths, widths);

          // Synchronisation
          cudaEventRecord(batch->normComp, batch->inpStream);

          CUDA_SAFE_CALL(cudaGetLastError(), "Calling the normalisation and spreading kernel.");
        }
      }
      else if ( batch->flag & CU_INPT_DEVICE    )
      {
        // Copy chunks of FFT data and normalise and spread using the GPU

        if ( batch->noSteps > 1 ) // TODO: multi step  .
        {
          fprintf(stderr,"ERROR: CU_INPT_DEVICE has not been set up for multi-step.");
          exit(EXIT_FAILURE);
        }

        // Make sure the previous thread has complete reading from page locked memory
        //CUDA_SAFE_CALL(cudaEventSynchronize(plains->iDataCpyComp), "ERROR: copying data to device");
        //nvtxRangePush("Zero");
        //memset(plains->h_iData, 0, plains->inpDataSize);
        //nvtxRangePop();

        FOLD // Setup parameters
        {
          int harm  = 0;
          int step  = 0; // TODO multistep
          int sz    = 0;

          for (int ss = 0; ss< batch->noStacks; ss++)
          {
            cuFfdotStack* cStack = &batch->stacks[ss];

            for (int si = 0; si< cStack->noInStack; si++)
            {
              cuHarmInfo* cHInfo  = &batch->hInfos[harm];  // The current harmonic we are working on
              cuFFdot*    cPlain  = &batch->plains[harm];     //
              rVals*      rVal    = &((*batch->rInput)[step][harm]);

              /*
              drlo = calc_required_r_gpu(cHInfo->harmFrac, searchRLow[0]);
              drhi = calc_required_r_gpu(cHInfo->harmFrac, searchRHi[0]);

              binoffset = cHInfo->halfWidth;
              lobin = (int) floor(drlo) - binoffset;
              hibin = (int)  ceil(drhi) + binoffset;
              numdata = hibin - lobin + 1;

              numrs = (int) ((ceil(drhi) - floor(drlo)) * ACCEL_RDR + DBLCORRECT) + 1;
              if (harm == 0)
              {
                numrs = batch->accelLen;
              }
              else if (numrs % ACCEL_RDR)
                numrs = (numrs / ACCEL_RDR + 1) * ACCEL_RDR;
              int numtocopy = cHInfo->width - 2 * cHInfo->halfWidth * ACCEL_NUMBETWEEN;
              if (numrs < numtocopy)
                numtocopy = numrs;
               */

              lengths.val[harm]     = rVal->numdata;
              d_iDataList.val[harm] = cPlain->d_iData;
              widths.val[harm]      = cStack->width;
              if ( rVal->lobin-batch->SrchSz->rLow < 0 )
              {
                // NOTE could use an offset parameter here
                printf("ERROR: Input data index out of bounds.\n");
                exit(EXIT_FAILURE);
              }
              d_fftList.val[harm]   = &batch->d_iData[rVal->lobin-batch->SrchSz->rLow];

              sz += cStack->inpStride;

              harm++;
            }
          }
        }

        FOLD // Normalise and spread
        {
          // Blocks of 1024 threads ( the maximum number of threads per block )
          dimBlock.x = NAS_DIMX;
          dimBlock.y = NAS_DIMY;
          dimBlock.z = 1;

          // One block per harmonic, thus we can sort input powers in Shared memory
          dimGrid.x = batch->noHarms;
          dimGrid.y = 1;

          // Synchronisation
          for (int ss = 0; ss< batch->noStacks; ss++)
          {
            cuFfdotStack* cStack = &batch->stacks[ss];
            CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, cStack->convComp, 0), "ERROR: waiting for GPU to be ready to copy data to device\n");
          }

          // Call the kernel to normalise and spread the input data
          normAndSpreadBlksDevice<<<dimGrid, dimBlock, (lengths.val[0]+1)*sizeof(float), batch->inpStream>>>(d_fftList, d_iDataList, lengths, widths);

          // Synchronisation
          cudaEventRecord(batch->normComp, batch->inpStream);

          CUDA_SAFE_CALL(cudaGetLastError(), "Calling the normalisation and spreading kernel.");
        }
      }
      else if ( batch->flag & CU_INPT_SINGLE_C  )
      {
        // Copy chunks of FFT data and normalise and spread using the CPU

        // Make sure the previous thread has complete reading from page locked memory
        CUDA_SAFE_CALL(cudaEventSynchronize(batch->iDataCpyComp), "ERROR: Synchronising before writing input data to page locked host memory.");

        nvtxRangePush("Zero");
        memset(batch->h_iData, 0, batch->inpDataSize*batch->noSteps);
        nvtxRangePop();

        CPU_Norm_Spread(batch, searchRLow, searchRHi, norm_type, fft);

        FOLD // Synchronisation
        {
          for (int ss = 0; ss< batch->noStacks; ss++)
          {
            cuFfdotStack* cStack = &batch->stacks[ss];
            CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, cStack->convComp, 0), "ERROR: waiting for GPU to be ready to copy data to device\n");
          }

#ifdef TIMING
          cudaEventRecord(batch->iDataCpyInit, batch->inpStream);
#endif

        }

        // Copy to device
        CUDA_SAFE_CALL(cudaMemcpyAsync(batch->d_iData, batch->h_iData, batch->inpDataSize*batch->noSteps, cudaMemcpyHostToDevice, batch->inpStream), "Failed to copy input data to device");

        // Synchronisation
        cudaEventRecord(batch->iDataCpyComp, batch->inpStream);
        cudaEventRecord(batch->normComp, batch->inpStream);

        CUDA_SAFE_CALL(cudaGetLastError(), "Error preparing the input data.");
      }
    }

    if ( DBG_INP03 ) // Print debug info  .
    {
      for (int ss = 0; ss< batch->noHarms && true; ss++)
      {
        cuFFdot* cPlain     = &batch->plains[ss];
        printf("\nGPU Input Data pre FFT h:%i   f: %f\n",ss,cPlain->harmInf->harmFrac);
        //printfData<<<1,1,0,0>>>((float*)cPlain->d_iData,10,1, cPlain->harmInf->inpStride);
        CUDA_SAFE_CALL(cudaStreamSynchronize(0),"");
        for (int ss = 0; ss< batch->noStacks; ss++)
        {
          cuFfdotStack* cStack = &batch->stacks[ss];
          CUDA_SAFE_CALL(cudaStreamSynchronize(cStack->fftIStream),"");
        }
      }
    }

    FOLD // fft the input data  .
    {
      // I tested doing the FFT's on the CPU and its way to slow! so go GPU!
      // TODO: I could make this a flag

#ifdef SYNCHRONOUS
      cuFfdotStack* pStack = NULL;
#endif

      for (int ss = 0; ss< batch->noStacks; ss++)
      {
        cuFfdotStack* cStack = &batch->stacks[ss];

        CUDA_SAFE_CALL(cudaGetLastError(), "Error before input fft.");

        FOLD // Synchronisation  .
        {
          cudaStreamWaitEvent(cStack->fftIStream, batch->normComp, 0);

#ifdef SYNCHRONOUS
          // Wait for the search to complete before FFT'ing the next set of input
          cudaStreamWaitEvent(cStack->fftIStream, batch->searchComp, 0);

          // Wait for previous FFT to complete
          if ( pStack != NULL )
            cudaStreamWaitEvent(cStack->fftIStream, pStack->prepComp, 0);
#endif
        }

        FOLD // Timing  .
        {
#ifdef TIMING
          cudaEventRecord(cStack->inpFFTinit, cStack->fftIStream);
#endif
        }

        FOLD // Do the FFT  .
        {
#pragma omp critical
          {
            CUFFT_SAFE_CALL(cufftSetStream(cStack->inpPlan, cStack->fftIStream),"Failed associating a CUFFT plan with FFT input stream\n");
            CUFFT_SAFE_CALL(cufftExecC2C(cStack->inpPlan, (cufftComplex *) cStack->d_iData, (cufftComplex *) cStack->d_iData, CUFFT_FORWARD),"Failed to execute input CUFFT plan.");

            CUDA_SAFE_CALL(cudaGetLastError(), "Error FFT'ing the input data.");
          }
        }

        FOLD // Synchronisation  .
        {
          cudaEventRecord(cStack->prepComp, cStack->fftIStream);

#ifdef SYNCHRONOUS
          pStack = cStack;
#endif
        }
      }

      CUDA_SAFE_CALL(cudaGetLastError(), "Error FFT'ing the input data.");
    }

    if ( DBG_INP04 ) // Print debug info  .
    {
      for (int ss = 0; ss< batch->noHarms && true; ss++)
      {
        cuFFdot* cPlain     = &batch->plains[ss];
        printf("\nGPU Input Data post FFT h:%i   f: %f\n",ss,cPlain->harmInf->harmFrac);
        //printfData<<<1,1,0,0>>>((float*)cPlain->d_iData,10,1, cPlain->harmInf->inpStride);
        CUDA_SAFE_CALL(cudaStreamSynchronize(0),"");
        for (int ss = 0; ss< batch->noStacks; ss++)
        {
          cuFfdotStack* cStack = &batch->stacks[ss];
          CUDA_SAFE_CALL(cudaStreamSynchronize(cStack->fftIStream),"");
        }
      }
    }

    batch->haveInput = 1;

    nvtxRangePop();
  }
}
