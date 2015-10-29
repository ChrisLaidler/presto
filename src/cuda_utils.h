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


#define POWERCU(r,i)  ((r)*(r) + (i)*(i))     /// The sum of the powers of two number
#define POWERC(c)     POWERCU(c.r, c.i)       /// The sum of the powers of a complex number

#define BLACK     "\033[22;30m"
#define GREEN     "\033[22;31m"
#define MAGENTA   "\033[22;35m"
#define RESET     "\033[0m"

// Free a pointer and set value to zero
#define freeNull(pointer) { if (pointer) free ( pointer ); pointer = NULL; }
#define cudaFreeNull(pointer) { if (pointer) CUDA_SAFE_CALL(cudaFree(pointer), "Failed to free device memory."); pointer = NULL; }
#define cudaFreeHostNull(pointer) { if (pointer) CUDA_SAFE_CALL(cudaFreeHost(pointer), "Failed to free host memory."); pointer = NULL; }

// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM)
typedef struct
{
    int SM; // 0xMm (hexadecimal notation), M = SM Major version, and m = SM minor version  ie 0x12 (18) is compute 1.2
    int value;
} SMVal;


//====================================== Inline functions ================================================//

const char* _cudaGetErrorEnum(cufftResult error);

//==================================== Function Prototypes ===============================================//

inline int getValFromSMVer(int major, int minor, SMVal* vals);

/**
 * @brief printf a message iff compiled in debug mode
 *
 * @param format C string that contains a format string that follows the same specifications as format in <a href="http://www.cplusplus.com/printf">printf</a>
 * @return void
 **/
void debugMessage ( const char* format, ... );

void errMsg ( const char* format, ... );

int detect_gdb_tree(void);



/**
 * @brief get free ram in bytes
 *
 * @return number of bytes of free RAM
 **/
ExternC unsigned long getFreeRamCU();

ExternC int  ffdotPln(float* powers, fcomplex* fft, int loR, int noBins, int noHarms, double centR, double centZ, double rSZ, double zSZ, int noR, int noZ, int halfwidth, float* fac);
ExternC void rz_interp_cu(fcomplex* fft, int loR, int noR, double centR, double centZ, int halfwidth);
ExternC void opt_candPlns(accelcand* cand, cuSearch* srch, accelobs* obs, int nn, cuOptCand* pln);
ExternC void opt_candSwrm(accelcand* cand, accelobs* obs, int nn, cuOptCand* pln);

ExternC void __cuSafeCall(cudaError_t cudaStat,    const char *file, const int line, const char *errorMsg);
ExternC void __cufftSafeCall(cufftResult cudaStat, const char *file, const int line, const char *errorMsg);

/** Get the number of CUDA capable GPUS's
 */
ExternC int getGPUCount();

ExternC void initGPUs(gpuSpecs* gSpec);

/** Print a nice list of CUDA capable device(s) with some details
 */
ExternC void listDevices();

/** Get GPU memory alignment in bytes  .
 *
 */
ExternC int getMemAlignment();

/** Get the stride (in number of elements) given a number of elements and the "block" size  .
 */
ExternC int getStrie(int noEls, int elSz, int blockSz);

#endif /* CUDA_UTILS_H_ */
