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
unsigned long getFreeRamCU()
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
unsigned long getFreeRamCU()
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
    { 0x32, 192},     // Kepler  Generation (SM 3.2) GK10x class
    { 0x35, 192 },    // Kepler  Generation (SM 3.5) GK11x class
    { 0x37, 192},     // Kepler  Generation (SM 3.7) GK21x class
    { 0x50, 128 },    // Maxwell Generation (SM 5.0) GM10x class
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
    { 0x10,  8 }, // Tesla   Generation (SM 1.0) G80   class
    { 0x11,  8 }, // Tesla   Generation (SM 1.1) G8x   class
    { 0x12,  8 }, // Tesla   Generation (SM 1.2) G9x   class
    { 0x13,  8 }, // Tesla   Generation (SM 1.3) GT200 class
    { 0x20, 32 }, // Fermi   Generation (SM 2.0) GF100 class
    { 0x21, 48 }, // Fermi   Generation (SM 2.1) GF10x class
    { 0x30, 192}, // Kepler  Generation (SM 3.0) GK10x class
    { 0x32, 192}, // Kepler  Generation (SM 3.2) GK10x class
    { 0x35, 192}, // Kepler  Generation (SM 3.5) GK11x class
    { 0x37, 192}, // Kepler  Generation (SM 3.7) GK21x class
    { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
    {   -1, -1 }
};

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

void initGPUs(gpuSpecs* gSpec)
{
  int currentDevvice, deviceCount;
  size_t free, total;
  char txt[1024];

  CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount), "Failed to get device count using cudaGetDeviceCount");

  for (int dIdx = 0; dIdx < gSpec->noDevices; dIdx++)
  {
    int device = gSpec->devId[dIdx];

    CUDA_SAFE_CALL( cudaSetDevice ( device ), "Failed to set device using cudaSetDevice");

    // Check if the the current device is 'device'
    CUDA_SAFE_CALL( cudaGetDevice(&currentDevvice), "Failed to get device using cudaGetDevice" );
    if ( currentDevvice != device)
    {
      fprintf(stderr, "ERROR: Device not set.\n");
      exit(EXIT_FAILURE);
    }

    FOLD // call something to initialise the device
    {
      //sprintf(txt,"Init device %02i", device );
      //nvtxRangePush(txt);

      CUDA_SAFE_CALL(cudaMemGetInfo ( &free, &total ), "Getting Device memory information");

      cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

      //nvtxRangePop();
    }
  }
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

int getStrie(int noEls, int elSz, int blockSz)
{
  int     noBlocks = ceil(noEls*elSz/(float)blockSz);
  float   elStride = noBlocks * blockSz / (float)elSz;

  float rem = elStride - (int)elStride;
  if ( rem != 0 )
    fprintf(stderr, "ERROR: Memory not aligned to the size of stride.\n");

  return elStride;
}

