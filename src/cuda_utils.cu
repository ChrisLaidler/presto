/*
 * cuda_utils.cu
 *
 *      Author: claidler Laidler
 *      e-mail: chris.laidler@gmail.com
 *
 *      This contains a number of basic functions for use with CUDA applications
 */

#include "cuda_utils.h"

#if _WIN32
#include <windows.h>
size_t getFreeRamCU()
{
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return status.ullTotalPhys;
}
#elif __linux
#include <sys/sysinfo.h>
/** Get the amount of free RAM in bytes
 *
 */
size_t getFreeRamCU()
{
  long pages = sysconf(_SC_PHYS_PAGES);
  long freePages = sysconf(_SC_AVPHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);

  //return freePages*page_size;

  struct sysinfo sys_info;
  if(sysinfo(&sys_info) != 0)
  {
    fprintf(stderr, "ERROR: Reading memory info.");
    return 0;
  }
  else
  {
    return (sys_info.freeram + sys_info.bufferram )* sys_info.mem_unit ;
  }
}
#else
size_t getFreeRamCU()
{
  fprintf(stderr, "ERROR: getFreeRam not enabled on this system.");
}
#endif

// 32-bit floating-point add, multiply, multiply-add Operations per Clock Cycle per Multiprocessor
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions__throughput-native-arithmetic-instructions
static SMVal opsF32_A_M_MAD_perMPperCC[] =
{
    { 0x10, 8 },      // Tesla   Generation (SM 1.0) G80   class
    { 0x11, 8 },      // Tesla   Generation (SM 1.1) G8x   class
    { 0x12, 8 },      // Tesla   Generation (SM 1.2) G9x   class
    { 0x13, 8 },      // Tesla   Generation (SM 1.3) GT200 class
    { 0x20, 32 },     // Fermi   Generation (SM 2.0) GF100 class
    { 0x21, 48 },     // Fermi   Generation (SM 2.1) GF10x class

    { 0x30, 192 },    // Kepler  Generation (SM 3.0) GK10x class
    { 0x32, 192 },    // Kepler  Generation (SM 3.2) GK10x class
    { 0x35, 192 },    // Kepler  Generation (SM 3.5) GK11x class
    { 0x37, 192 },    // Kepler  Generation (SM 3.7) GK11x class

    { 0x50, 128 },    // Maxwell Generation (SM 5.0) GM10x class
    { 0x52, 128 },    // Maxwell Generation (SM 5.2) GM10x class
    { 0x53, 128 },    // Maxwell Generation (SM 5.3) GM10x class
    { -1, -1 }
};

// 64-bit floating-point add, multiply, multiply-add Operations per Clock Cycle per Multiprocessor
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions__throughput-native-arithmetic-instructions
/*
static SMVal opsF64_A_M_MAD_perMPperCC[] =
{
    { 0x10, 0 },      // Tesla  Generation  (SM 1.0) G80   class
    { 0x11, 0 },      // Tesla  Generation  (SM 1.1) G8x   class
    { 0x12, 0 },      // Tesla  Generation  (SM 1.2) G9x   class
    { 0x13, 1 },      // Tesla  Generation  (SM 1.3) GT200 class
    { 0x20, 16 },     // Fermi  Generation  (SM 2.0) GF100 class
    { 0x21, 4 },      // Fermi  Generation  (SM 2.1) GF10x class
    { 0x30, 8 },      // Kepler Generation  (SM 3.0) GK10x class
    { 0x35, 64 },     // Kepler Generation  (SM 3.5) GK11x class
    { 0x50, 1 },      // Maxwell Generation (SM 5.0) GM10x class
    { -1, -1 }
};
 */

// Defined number of cores for SM of specific compute versions ( Taken from CUDA 6.5 Samples )
static SMVal nGpuArchCoresPerSM[] =
{
    { 0x10, 8 },      // Tesla   Generation (SM 1.0) G80   class
    { 0x11, 8 },      // Tesla   Generation (SM 1.1) G8x   class
    { 0x12, 8 },      // Tesla   Generation (SM 1.2) G9x   class
    { 0x13, 8 },      // Tesla   Generation (SM 1.3) GT200 class
    { 0x20, 32 },     // Fermi   Generation (SM 2.0) GF100 class
    { 0x21, 48 },     // Fermi   Generation (SM 2.1) GF10x class

    { 0x30, 192 },    // Kepler  Generation (SM 3.0) GK10x class
    { 0x32, 192 },    // Kepler  Generation (SM 3.2) GK10x class
    { 0x35, 192 },    // Kepler  Generation (SM 3.5) GK11x class
    { 0x37, 192 },    // Kepler  Generation (SM 3.7) GK11x class

    { 0x50, 128 },    // Maxwell Generation (SM 5.0) GM10x class
    { 0x52, 128 },    // Maxwell Generation (SM 5.2) GM10x class
    { 0x53, 128 },    // Maxwell Generation (SM 5.3) GM10x class
    { -1, -1 }
};


__global__ void clock_block(long long int clock_count)
{
  long long int start_clock = clock64();
  long long int clock_offset = 0;

    while (clock_offset < clock_count)
    {
        clock_offset = clock64() - start_clock;
    }

    if( clock_offset < 0 )
    {
      printf("This is a dummy string so that this wont get optimised out");
    }
}

__host__ void streamSleep(cudaStream_t stream, long long int clock_count )
{
  dim3 dimBlock, dimGrid;

  // Blocks of 1024 threads ( the maximum number of threads per block )
  dimBlock.x = 1;
  dimBlock.y = 1;
  dimBlock.z = 1;

  // One block per harmonic, thus we can sort input powers in Shared memory
  dimGrid.x = 1;
  dimGrid.y = 1;

  clock_block<<< dimGrid,  dimBlock, 0, stream >>>(clock_count);

  CUDA_SAFE_CALL(cudaGetLastError(), "Calling the clock_block kernel.");
}

/** Convert CUDA half values to floats
 *
 * @param h	The half precision floating point value to be converts
 * @return	The single precision floating point value
 */
float half2float(const ushort h)
{
  unsigned int sign     = ((h >> 15) & 1);
  unsigned int exponent = ((h >> 10) & 0x1f);
  unsigned int mantissa = ((h & 0x3ff) << 13);

  if (exponent == 0x1f)     // NaN or Inf
  {
    mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
    exponent = 0xff;
  }
  else if (!exponent)       // Denorm or Zero
  {
    if (mantissa)
    {
      unsigned int msb;
      exponent = 0x71;
      do
      {
	msb = (mantissa & 0x400000);
	mantissa <<= 1;  /* normalize */
	--exponent;
      }
      while (!msb);

      mantissa &= 0x7fffff;  /* 1.mantissa is implicit */
    }
  }
  else
  {
    exponent += 0x70;
  }

  uint res = ((sign << 31) | (exponent << 23) | mantissa);
  return  *((float*)(&res));
}

void debugMessage ( const char* format, ... )
{
#ifdef DEBUG
  if ( detect_gdb_tree() )
  {
    //printf("in GDB\n");
    va_list ap;
    va_start ( ap, format );
    vprintf ( format, ap );      // Write the line
    va_end ( ap );

    //std::cout.flush();
  }
  else
  {
    //printf("NOT in GDB\n");
    printf ( MAGENTA );

    va_list ap;
    va_start ( ap, format );
    vprintf ( format, ap );      // Write the line
    va_end ( ap );

    printf ( RESET );
    //std::cout.flush();
  }
#endif
}

void errMsg ( const char* format, ... )
{
  va_list ap;
  va_start ( ap, format );
  vfprintf (stderr, format, ap );
  va_end ( ap );
}

int detect_gdb_tree(void)
{
  int gdb;
  {
    int rc = 0;
    FILE *fd = fopen("/tmp", "r");

    if (fileno(fd) >= 5)
    {
      rc = 1;
    }

    fclose(fd);
    gdb = rc;
  }

  return gdb;
}

void __cufftSafeCall(cufftResult cudaStat, const char *file, const int line, const char *errorMsg)
{
  if (cudaStat != CUFFT_SUCCESS)
  {
    fprintf(stderr, "CUFFT ERROR: %s [ %s at line %d in file %s ]\n", errorMsg, _cudaGetErrorEnum(cudaStat), line, file);
    exit(EXIT_FAILURE);
  }
}

void __cuSafeCall(cudaError_t cudaStat, const char *file, const int line, const char *errorMsg)
{
  if (cudaStat != cudaSuccess)
  {
    fprintf(stderr, "CUDA ERROR: %s [ %s at line %d in file %s ]\n", errorMsg, cudaGetErrorString(cudaStat), line, file);
    exit(EXIT_FAILURE);
  }
}

void __exit_directive(const char *file, const int line, const char *flag)
{
  fprintf(stderr, "ERROR: This code has not bee compiled with the \"%s\" preprocessor directive. Line: %d In: %s.\n\tIf you have enabled this you may need a full recompile ie. make cudaclean; make \n", flag,line, file);
  exit(EXIT_FAILURE);
}

int getGPUCount()
{
  int deviceCount;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount), "Failed to get device count using cudaGetDeviceCount");
  return deviceCount;
}

inline int getValFromSMVer(int major, int minor, SMVal* vals)
{
  int index = 0;

  while (vals[index].SM != -1)
  {
    //int thisSM = ((major << 4) + minor);
    //int testSM = vals[index].SM;

    if (vals[index].SM == ((major << 4) + minor))
      return vals[index].value;

    index++;
  }

  // If we get here we didn't find the value in the array
  return -1;
}



void listDevices()
{
  cudaDeviceProp deviceProp;
  int currentDevvice, deviceCount;
  size_t free, total;

  CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount), "Failed to get device count using cudaGetDeviceCount");
  if ( deviceCount == 0 )
  {
    printf("Could not detect any CUDA capable devices.\n");
    return;
  }

  int driverVersion = 0, runtimeVersion = 0;

  CUDA_SAFE_CALL( cudaDriverGetVersion (&driverVersion),  "Failed to get driver version using cudaDriverGetVersion");
  CUDA_SAFE_CALL( cudaRuntimeGetVersion(&runtimeVersion), "Failed to get run time version using cudaRuntimeGetVersion");
  printf("\n  CUDA Driver Version    %d.%d \n", driverVersion  / 1000, (driverVersion  % 100) / 10);
  printf("  Runtime Version        %d.%d \n",   runtimeVersion / 1000, (runtimeVersion % 100) / 10);

  printf("\nListing %i device(s):\n",deviceCount);

  for (int device = 0; device < deviceCount; device++)
  {
    CUDA_SAFE_CALL( cudaSetDevice ( device ), "Failed to set device using cudaSetDevice");

    // Check if the the current device is 'device'
    CUDA_SAFE_CALL( cudaGetDevice(&currentDevvice), "Failed to get device using cudaGetDevice" );
    if ( currentDevvice != device)
    {
      fprintf(stderr, "ERROR: Device not set.\n");
      exit(EXIT_FAILURE);
    }

    CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, device), "Failed to get device properties device using cudaGetDeviceProperties");

    printf("\nDevice %d: \"%s\"\n", device, deviceProp.name);
    printf("  CUDA Capability Major.Minor version number:    %d.%d\n",
        deviceProp.major, deviceProp.minor);

    char msg[256];
    printf(
        "  (%2d) Multiprocessors x (%3d) CUDA Cores/MP:    %d CUDA Cores\n",
        deviceProp.multiProcessorCount,
        getValFromSMVer(deviceProp.major, deviceProp.minor, nGpuArchCoresPerSM),
        getValFromSMVer(deviceProp.major, deviceProp.minor, nGpuArchCoresPerSM)
        * deviceProp.multiProcessorCount);
    printf(
        "  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n",
        deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
    double maxGflops = deviceProp.multiProcessorCount*
        getValFromSMVer(deviceProp.major, deviceProp.minor, opsF32_A_M_MAD_perMPperCC)*
        (2*deviceProp.clockRate * 1e-6f); // Add brackets to stop overflow errors

    printf("  Max Gigaflops (FMAD's):                        %.1f Gflops\n",
        maxGflops);

    // This is supported in CUDA 5.0 (runtime API device properties)
    sprintf(msg,
        "  Total amount of global memory:                 %.1f Gibibyte (%llu bytes)\n",
        (float) deviceProp.totalGlobalMem / 1073741824.0f,
        (unsigned long long) deviceProp.totalGlobalMem);
    printf("%s", msg);

    CUDA_SAFE_CALL(cudaMemGetInfo ( &free, &total ), "Getting Device memory information");
    sprintf(msg,
        "   of which %6.2f%% is currently free:           %.1f Gibibyte (%llu bytes) free\n",
        free / (float)total * 100.0f, (float) free / 1073741824.0f,
        (unsigned long long) free);
    printf("%s", msg);

    printf("  Memory Clock rate:                             %.0f Mhz\n",
        deviceProp.memoryClockRate * 1e-3f);
  }

  printf("\n");
}

int getMemAlignment()
{
  size_t stride;
  float* rnd;

  CUDA_SAFE_CALL(cudaMallocPitch(&rnd,    &stride, sizeof(float), 1),   "Failed to allocate device memory for getMemAlignment.");
  CUDA_SAFE_CALL(cudaFree(rnd),                                         "Failed to free device memory for getMemAlignment.");

  return stride;
}

int getStride(int noEls, int elSz, int blockSz)
{
  int     noBlocks = ceil(noEls*elSz/(float)blockSz);
  float   elStride = noBlocks * blockSz / (float)elSz;

  float rem = elStride - (int)elStride;
  if ( rem != 0 )
  {
    fprintf(stderr, "ERROR: Memory not aligned to the size of stride. Pleas contact Chris Laidler.\n");
    exit(EXIT_FAILURE);
  }

  return elStride;
}

const char* _cudaGetErrorEnum(cufftResult error)
{
  switch (error)
  {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";

    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "CUFFT_INCOMPLETE_PARAMETER_LIST";

    case CUFFT_INVALID_DEVICE:
      return "CUFFT_INVALID_DEVICE";

    case CUFFT_PARSE_ERROR:
      return "CUFFT_PARSE_ERROR";

    case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE";

#if CUDA_VERSION >= 6050
    case CUFFT_LICENSE_ERROR:
      return "CUFFT_LICENSE_ERROR";

    case CUFFT_NOT_IMPLEMENTED:
      return "CUFFT_NOT_IMPLEMENTED";
#endif

  }

  return "<unknown>";
}

/**
 * @brief printf a message for info logging
 *
 * @param lev     The info level of this message
 * @param indent  The indentation level pof this message
 * @param format  C string that contains a format string that follows the same specifications as format in <a href="http://www.cplusplus.com/printf">printf</a>
 * @return void
 **/
void infoMSG ( int lev, int indent, const char* format, ... )
{
  if ( lev <= msgLevel )
  {
    char buffer[1024];
    char *msg = buffer;

    va_list ap;
    va_start ( ap, format );
    vsprintf ( buffer, format, ap );      // Write the line
    va_end ( ap );

    while ( *msg == 10 )
    {
      printf("\n");
      msg++;
    }

    printf("Info %02i ", lev);

    for ( int i = 0; i < indent;  i++ )
      printf("  ");

    printf("%s", msg);

    if ( msg[strlen(msg)-1] != 10 )
    {
      printf("\n");
    }

    fflush(stdout);
    fflush(stderr);
  }
}

void queryEvents( cudaEvent_t   evmt, const char* msg )
{
  cudaError_t ret;

  ret = cudaEventQuery(evmt);

  if ( ret == cudaSuccess )
  {
    infoMSG(6,6,"Event Query %s: Done or not called.\n", msg);
  }
  else if ( ret == cudaErrorNotReady )
  {
    infoMSG(6,6,"Event Query %s: Not finished.\n", msg);
  }
  else
  {
    infoMSG(6,6,"Event Query %s: Unknown.. %s ", msg, cudaGetErrorString(ret) );
  }
}

void timeEvents( cudaEvent_t   start, cudaEvent_t   end, long long* timeSum, const char* msg )
{
  // Check for previous errors
  CUDA_SAFE_CALL(cudaGetLastError(), "Entering timing");

  float time;         // Time in ms of the thing
  cudaError_t ret;    // Return status of cudaEventElapsedTime

  ret = cudaEventQuery(end);
  if ( ret == cudaErrorNotReady )
  {
    // This is not ideal!
    infoMSG(6,6,"timeEvents, end event not complete, Blocking");

    cudaError_t res = cudaGetLastError(); // Resets the error to cudaSuccess
    char msg2[1024];

    PROF // Profiling  .
    {
      sprintf(msg2, "Timing block [%s]", msg);
      NV_RANGE_PUSH(msg2);
    }

    sprintf(msg2, "At a timing blocking synchronisation \"%s\"", msg);
    CUDA_SAFE_CALL(cudaEventSynchronize(end), msg2 );

    PROF // Profiling  .
    {
      NV_RANGE_POP(); // msg2
    }
  }

  // Do the actual timing
  ret = cudaEventElapsedTime(&time, start, end);

  if      ( ret == cudaErrorInvalidResourceHandle )
  {
    // This is OK the event just hasn't been called yet
    // This shouldn't happen if the checks were done correctly!

    cudaError_t res = cudaGetLastError(); // Resets the error to cudaSuccess
    infoMSG(6,6,"Event not created yet?");
  }
  else if ( ret == cudaErrorNotReady )
  {
    infoMSG(6,6,"Event no ready!\n");
  }
  else if ( ret == cudaSuccess )
  {
#pragma omp atomic
    (*timeSum) += time*1e3; // Convert to Microsecond

    infoMSG(6,6,"Event: \"%s\"  Time: %.3f ms \n", msg, time);
  }
  else
  {
    char msg2[1024];
    sprintf(msg2, "Timing %s", msg);

    fprintf(stderr, "CUDA ERROR: %s [ %s ]\n", msg2, cudaGetErrorString(ret));
    exit(EXIT_FAILURE);
  }
}

void printContext()
{
  int currentDevvice;
  CUcontext pctx;
  cuCtxGetCurrent ( &pctx );
  CUDA_SAFE_CALL(cudaGetDevice(&currentDevvice), "Failed to get device using cudaGetDevice");

  int trd = 0;
#ifdef	WITHOMP
  trd = omp_get_thread_num();
#endif	// WITHOMP

  printf("Thread %02i  currentDevvice: %i Context %p \n", trd, currentDevvice, pctx);
}

int setDevice(int device)
{
  int dev;

  CUDA_SAFE_CALL(cudaGetDevice(&dev), "Failed to get device using cudaGetDevice");

  if ( dev != device )
  {
    CUDA_SAFE_CALL(cudaSetDevice(device), "Failed to set device using cudaSetDevice");
    CUDA_SAFE_CALL(cudaGetDevice(&dev), "Failed to get device using cudaGetDevice");
    if ( dev != device )
    {
      fprintf(stderr, "ERROR: CUDA Device not set.\n");
      exit(EXIT_FAILURE);
    }
  }

  return dev;
}
