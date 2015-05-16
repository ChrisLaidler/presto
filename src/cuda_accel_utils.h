#ifndef CUDA_ACCEL_UTILS_INCLUDED
#define CUDA_ACCEL_UTILS_INCLUDED

#include <stdio.h>
#include <cuda.h>
#include <cufft.h>
//#include <cufftXt.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cuda_accel.h"
#include "cuda_utils.h"

#ifdef CBL
#include "array.h"
#include "arrayDsp.h"
#endif

extern "C"
{
#define __float128 long double
#include "accel.h"
}

#pragma once

#define   TEMPLATE_CONVOLVE 1
#define   TEMPLATE_SEARCH   1

#define   MAXMUL        (1e3)                 /// Not sure what this is?

#define   SEMFREE       2147483647            /// The value to indicate that a semaphore is free

//#define   CU_MAX_BLOCK_SIZE   1024
#define   CU_MAX_BLOCK_SIZE   2048            //  2K FFT's
//#define   CU_MAX_BLOCK_SIZE   4096          //  4K FFT's
//#define   CU_MAX_BLOCK_SIZE   8192          //  8K FFT's
//#define   CU_MAX_BLOCK_SIZE   16384         // 16K FFT's
//#define   CU_MAX_BLOCK_SIZE   32768         // 32K FFT's

//#define BS_MAX          (CU_MAX_BLOCK_SIZE/2) // BITonic sort max number of elements
#define BS_MAX          8192                  // BITonic sort max number of elements

//#define ACCEL_USELEN 7470     // This works up to zmax=300 to use 8K FFTs
//#define ACCEL_USELEN 62500    // 64K   up to zmax=1200
//#define ACCEL_USELEN 30000    // 32K   up to zmax=1200
//#define ACCEL_USELEN 13500    // 16K   up to zmax=1200
//#define ACCEL_USELEN 5300     // 8K    up to zmax=1200
//#define ACCEL_USELEN 6000     // 8K    up to zmax=900
//#define ACCEL_USELEN 6990     // 8K    up to zmax=500
//#define ACCEL_USELEN 1200     // 4K    up to zmax=1200

#define CHUNKSZ         8                     // Added by run time script

#define SS_X            16                    // X Thread Block
#define SS_Y            16                    // Y Thread Block
#define SS_X_TILES      3                     // X No. tiles covered by 1 Thread block
#define SS_Y_TILES      3                     // Y No. tiles covered by 1 Thread block
#define SS_X_NUM        (SS_X*SS_X_TILES)     // No X elements covered by 1 Thread Block
#define SS_Y_NUM        (SS_Y*SS_Y_TILES)     // No Y elements covered by 1 Thread Block
#define SS_Z_MAX_BUF    0                     // No points to consider for local maxima calculation  ( 8 = 16/2 )
#define SS_R_MAX_BUF    0                     // No points to consider for local maxima calculation  ( 8 = 16/2 )
#define SS_X_OVERLAP    0                     // X Overlap between Thread Blocks
#define SS_Y_OVERLAP    0                     // Y Overlap between Thread Blocks


#define SS3_X           16                    // X Thread Block
#define SS3_Y           4                     // Y Thread Block
//#define SS3_NPOWERS     21                    // Added by auto-tune script

#define SS4_X           32                    // X Thread Block
#define SS4_Y           16                    // Y Thread Block
#define SS4_NB          1                     // Added by auto-tune script

#define CNV_DIMX        16                    // X Thread Block
#define CNV_DIMY        4                     // Y Thread Block
#define CNV_WORK        12                    // The number of values to load into SM

/** Details of the Normalise and spread kernel
 * Should be as large as possible usually 32x32
 * NOTE: If compiled in debug mode it may be necessary to drop NAS_DIMY to 16
 *       else you may get Error "too many resources requested for launch"
 */
#define NAS_DIMX        32                    // Normalise and spread X dimension
#define NAS_DIMY        32                    // Normalise and spread Y dimension
#define NAS_NTRD        (NAS_DIMX*NAS_DIMY)   // Normalise and spread thread per block

#define MAX_CANDS_PER_BLOCK 6000000

#define BLOCKSIZE     16
#define BLOCK1DSIZE   BLOCKSIZE*BLOCKSIZE


#define BS_DIM        1024    // compute 3.x +
//#define BS_DIM        576   // compute 2.x


#define POWERR(r,i) (r)*(r)+(i)*(i)


typedef struct iList
{
    int val[MAX_IN_STACK];
} iList;

typedef struct fList
{
    float val[MAX_IN_STACK];
} fList;

//---------- LONG -------- \\

typedef struct long04
{
    long val[4];
} long04;

typedef struct long08
{
    long val[8];
} long08;

typedef struct long16
{
    long val[16];
} long16;

typedef struct long32
{
    long val[32];
} long32;

typedef struct long64
{
    long val[64];
} long64;

typedef struct long128
{
    long val[128];
} long128;


//---------- INT -------- \\

typedef struct int01
{
    int val[1];
} int01;

typedef struct int02
{
    int val[2];
} int02;

typedef struct int04
{
    int val[4];
} int04;

typedef struct int08
{
    int val[8];
} int08;

typedef struct int16
{
    int val[16];
} int16;

typedef struct int32
{
    int val[32];
} int32;

typedef struct int64
{
    int val[64];
} int64;

typedef struct int128
{
    int val[128];
} int128;


//---------- FLOAT -------- \\

typedef struct float08
{
    float val[8];
} float08;

typedef struct float16
{
    float val[16];
} float16;

typedef struct float32
{
    float val[32];
} float32;

typedef struct float64
{
    float val[64];
} float64;

typedef struct float128
{
    float val[128];
} float128;

//------------- Arrays that can be passed to kernels -------------------\\

typedef struct iHarmList
{
    int val[MAX_HARM_NO];
    __host__ __device__ inline int operator [](const int idx) { return val[idx]; }
} iHarmList;

typedef struct fHarmList
{
    float val[MAX_HARM_NO];
    __host__ __device__ inline float operator [](const int idx) { return val[idx]; }
} fHarmList;

typedef struct fsHarmList
{
    float* __restrict__ val[MAX_HARM_NO];
    __host__ __device__ inline float* __restrict__ operator [](const int idx) { return val[idx]; }
} fsHarmList;

typedef struct dHarmList
{
    double val[MAX_HARM_NO];
    __host__ __device__ inline double operator [](const int idx) { return val[idx]; }
} dHarmList;

typedef struct cHarmList
{
    fcomplexcu* __restrict__ val[MAX_HARM_NO];
    __host__ __device__ inline fcomplexcu* operator [](const int idx) { return val[idx]; }
} cHarmList;

typedef struct tHarmList
{
    cudaTextureObject_t val[MAX_HARM_NO];
    __host__ __device__ inline cudaTextureObject_t operator [](const int idx) { return val[idx]; }
} tHarmList;


extern long long time1;       /// Global variable used for timing
extern long long time2;       /// Global variable used for timing
extern long long time3;       /// Global variable used for timing
extern long long time4;       /// Global variable used for timing


typedef struct cuSearchList
{
    tHarmList   texs;           ///
    cHarmList   datas;          ///
    fsHarmList  powers;         ///
    iHarmList   yInds;          ///
    fHarmList   frac;           ///
    iHarmList   heights;        ///
    iHarmList   widths;         /// The width of usable r values in each plain
    iHarmList   strides;        ///
    iHarmList   ffdBuffre;      ///
    iHarmList   zMax;           ///
    iHarmList   rLow;           ///
} cuSearchList;

typedef struct f01
{
     float arry[1];
} f01;

typedef struct f02
{
     float arry[2];
} f02;

typedef struct f03
{
     float arry[3];
} f03;

typedef struct f04
{
     float arry[4];
} f04;

typedef struct f05
{
     float arry[5];
} f05;

typedef struct f06
{
     float arry[6];
} f06;

typedef struct f07
{
     float arry[7];
} f07;

typedef struct f08
{
     float arry[8];
} f08;

typedef struct fMax
{
     float arry[MAX_STEPS];
} fMax;


//-------------------------  Prototypes  -------------------------------\\


/* Calculate the 'r' you need for subharmonic  */
/* harm_fract = harmnum / numharm if the       */
/* 'r' at the fundamental harmonic is 'rfull'. */
__host__ __device__ static double calc_required_r_gpu(double harm_fract, double rfull)
{
  return (int) ( ((double)ACCEL_RDR) * rfull * harm_fract + 0.5) * ((double)ACCEL_DR);
}

/* Return an index for a Fourier Freq given an array that */
/* has stepsize ACCEL_DR and low freq 'lor'.              */
__host__ __device__ inline float index_from_r(float r, float lor)
{
  return /* (int) */((r - lor) * (float)ACCEL_RDR /* + 1e-6 */);
}

/* Calculate the 'z' you need for subharmonic  */
/* harm_fract = harmnum / numharm if the       */
/* 'z' at the fundamental harmonic is 'zfull'. */
__host__ __device__ static inline int calc_required_z(float harm_fract, float zfull)
{
  return round(ACCEL_RDZ * zfull * harm_fract) * ACCEL_DZ;
}

/** Return the index for a given z value of a F-Fplain
 *  Assume a stepsize of ACCEL_DZ
 *
 *  Return an index for a Fourier Fdot given an array that
 *  has stepsize ACCEL_DZ and low freq 'lor'.
 *
 * @param z
 * @param loz the low freq 'lor' of the plain
 * @return
 */
__host__ __device__ static inline int index_from_z(float z, float loz)
{
  return (int) ((z - loz) * ACCEL_RDZ + 1e-6);
}

__global__ void print_YINDS(int no);

double _GammaP (double n, double x);
double _GammaQ (double n, double x);



/////////////////////////////////////// Utility prototypes \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

/** Cycle the arrays of r-values  .
 *
 * @param batch
 */
void cycleRlists(cuFFdotBatch* batch);

/** Return the first value of 2^n >= x
 */
__host__ __device__ long long next2_to_n_cu(long long x);

/** Select the GPU to use
 * @param device The device to use
 * @param print if set to 1 will print the name and details of the device
 * @return The SMX version of the decide
 */
ExternC int selectDevice(int device, int print);

ExternC void printCands(const char* fileName, GSList *candsCPU);

ExternC void printContext();

ExternC void setContext(cuFFdotBatch* stkList) ;

ExternC void testzm();

/** Write CUFFT call backs to device
 */
void copyCUFFT_LD_CB(cuFFdotBatch* batch);

/** Create the stacks to do the
 *
 * @param numharmstages
 * @param zmax
 * @param obs
 * @return
 */
cuHarmInfo* createStacks(int numharmstages, int zmax, accelobs* obs);

ExternC int ffdot_planeCU2(cuFFdotBatch* plains, double searchRLow, double searchRHi, int norm_type, int search, fcomplexcu* fft, accelobs * obs, GSList** cands);

void printData_cu(cuFFdotBatch* stkLst, const int FLAGS, int harmonic, int nX = 10, int nY = 5, int sX = 0, int sY = 0);

/** Initialise the pointers of the stacks data structures of a batch  .
 *
 * This assumes the various memory blocks of the batch have been created
 *
 * @param batch
 */
ExternC void setStkPointers(cuFFdotBatch* batch);

/** Initialise the pointers of the plains data structures of a batch  .
 *
 * This assumes the stack pointers have already been setup
 *
 * @param batch
 */
ExternC void setPlainPointers(cuFFdotBatch* batch);

/** Initialise the pointers of the stacks and plains data structures of a batch  .
 *
 * This assumes the various memory blocks of the batch have been created
 *
 * @param batch
 */
ExternC void setBatchPointers(cuFFdotBatch* batch);

/** Print a integer in binary
 *
 * @param val The value to print
 */
void printBitString(uint val);



/////////////////////////////////////// Kernel prototypes \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

/** Create one GPU kernel. One kernel the size of the largest plain  .
 *
 * @param kernel
 * @return
 */
int createStackKernel(cuFfdotStack* cStack);

/** Create GPU kernels. One for each plain of the stack  .
 *
 * @param kernel
 * @return
 */
int createStackKernels(cuFfdotStack* cStack);

int init_harms(cuHarmInfo* hInf, int noHarms, accelobs *obs);



///////////////////////////////////////// Init prototypes \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

float cuGetMedian(float *data, uint len);

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
void initInput(cuFFdotBatch* batch, double* searchRLow, double* searchRHi, int norm_type, fcomplexcu* fft);



////////////////////////////////////// Convolution Prototypes \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

int setConstVals_Fam_Order( cuFFdotBatch* batch );

/** Convolve and inverse FFT the complex f-∂f plain using FFT callback
 * @param plains
 */
void convolveBatchCUFFT(cuFFdotBatch* batch );

/** Convolve and inverse FFT the complex f-∂f plain
 * This assumes the input data is ready and on the device
 * This creates a complex f-∂f plain
 */
void convolveBatch(cuFFdotBatch* batch);




//////////////////////////////////// Sum and search Prototypes \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

int setConstVals( cuFFdotBatch* stkLst, int numharmstages, float *powcut, long long *numindep );

__host__ __device__ double incdf (double p, double q );

void sumAndSearch(cuFFdotBatch* plains, long long* numindep, GSList** cands);

/** A function to call a kernel to harmonicall sum a plan and retunr the max of each column
 *
 */
void sumAndMax(cuFFdotBatch* plains, long long *numindep, float* powers);

#endif // CUDA_ACCEL_UTILS_INCLUDED
