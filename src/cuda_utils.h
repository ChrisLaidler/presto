/*
 * cuda_utils.h
 *
 *      Author: claidler Laidler 
 *      e-mail: chris.laidler@gmail.com
 *      
 *      This contains a number of basic functions for use with CUDA applications
 */

#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#include <stdio.h>
#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cuda_accel.h"


#define   POWERCU(r,i)  ((r)*(r) + (i)*(i))   /// The sum of powers of two number
#define   POWERC(c)     POWERCU(c.r, c.i)     /// The sum of powers of a complex number

// Free a pointer and set value to zero
#define freeNull(pointer) { if (pointer) free ( pointer ); pointer = NULL; }
#define cudaFreeNull(pointer) { if (pointer) CUDA_SAFE_CALL(cudaFree(pointer), "Failed to free device memory."); pointer = NULL; }
#define cudaFreeHostNull(pointer) { if (pointer) CUDA_SAFE_CALL(cudaFreeHost(pointer), "Failed to free host memory."); pointer = NULL; }

// cuFFT API errors
#ifdef _CUFFT_H_
static const char *_cudaGetErrorEnum(cufftResult error)
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

    case CUFFT_LICENSE_ERROR:
      return "CUFFT_LICENSE_ERROR";

    case CUFFT_NOT_IMPLEMENTED:
      return "CUFFT_NOT_IMPLEMENTED";
  }

  return "<unknown>";
}
#endif

// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM)
typedef struct
{
    int SM; // 0xMm (hexadecimal notation), M = SM Major version, and m = SM minor version  ie 0x12 (18) is compute 1.2
    int value;
} SMVal;


//====================================== Inline functions ================================================\\


//==================================== Function Prototypes ===============================================\\

inline int getValFromSMVer(int major, int minor, SMVal* vals);

/**
 * @brief get free ram in bytes
 *
 * @return number of bytes of free RAM
 **/
ExternC unsigned long getFreeRamCU();

ExternC int  ffdotPln(float* powers, fcomplex* fft, int loR, int noBins, int noHarms, double centR, double centZ, double rSZ, double zSZ, int noR, int noZ, int halfwidth, float* fac);
ExternC void rz_interp_cu(fcomplex* fft, int loR, int noR, double centR, double centZ, int halfwidth);
ExternC void opt_candPlns(accelcand* cand, accelobs* obs, int nn, cuOptCand* pln);
ExternC void opt_candSwrm(accelcand* cand, accelobs* obs, int nn, cuOptCand* pln);

ExternC void __cuSafeCall(cudaError_t cudaStat,    const char *file, const int line, const char *errorMsg);
ExternC void __cufftSafeCall(cufftResult cudaStat, const char *file, const int line, const char *errorMsg);

/** Get the number of CUDA capable GPUS's
 */
ExternC int getGPUCount();

/** Print a nice list of CUDA capable device(s) with some details
 */
ExternC void listDevices();

ExternC int getMemAlignment();
ExternC int getStrie(int noEls, int elSz, int blockSz);

#endif /* CUDA_UTILS_H_ */
