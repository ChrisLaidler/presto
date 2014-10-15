#ifndef CUDA_ACCEL_UTILS_INCLUDED
#define CUDA_ACCEL_UTILS_INCLUDED

#include <stdio.h>
#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cuda_accel.h"
#include "cuda_utils.h"

extern "C"
{
#define __float128 long double
#include "accel.h"
}

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

#define BS_MAX          (CU_MAX_BLOCK_SIZE/2) // BITonic sort max number of elements
#define BS_MAX          8192                  // BITonic sort max number of elements

//#define ACCEL_USELEN 7470     // This works up to zmax=300 to use 8K FFTs
//#define ACCEL_USELEN 62500    // 64K   up to zmax=1200
//#define ACCEL_USELEN 30000    // 32K   up to zmax=1200
//#define ACCEL_USELEN 13500    // 16K   up to zmax=1200
//#define ACCEL_USELEN 5300     // 8K    up to zmax=1200
//#define ACCEL_USELEN 6000     // 8K    up to zmax=900
//#define ACCEL_USELEN 6990     // 8K    up to zmax=500
//#define ACCEL_USELEN 1200     // 4K    up to zmax=1200

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
#define SS3_Y           8                     // Y Thread Block  
#define SS3_NPOWERS     21                    // Added by auto-tune script

#define SS4_X           32                    // X Thread Block
#define SS4_Y           16                    // Y Thread Block
#define SS4_NB          1                     // Added by auto-tune script

#define CNV_DIMX        16                    // X Thread Block
#define CNV_DIMY        8                     // Y Thread Block 
#define CNV_WORK        4

/** Details o the Normalise and spread kernel
 * Should be as large as possible usually 32x32
 * NOTE: If compiled in debug mode it may be necessary to drop NAS_DIMY to 16
 *       else you may get Error "too many resources requested for launch"
 */ 
#define NAS_DIMX        32                    // Normalise and spread X dimension 
#define NAS_DIMY        16                    // Normalise and spread Y dimension
#define NAS_NTRD        (NAS_DIMX*NAS_DIMY)   // Normalise and spread thread per block

//#define SS              3

#define MAX_CANDS_PER_BLOCK 6000000

#define BLOCKSIZE     16
#define BLOCK1DSIZE   BLOCKSIZE*BLOCKSIZE


#define BS_DIM        1024    // compute 3.x +
//#define BS_DIM        576   // compute 2.x

typedef struct fcmplxB
{
    fcomplexcu dat[CNV_WORK];
} fcmplxB ;

typedef struct iList
{
    int val[MAX_IN_STACK];
} iList;

typedef struct fList
{
    float val[MAX_IN_STACK];
} fList;

extern long long time1;       /// Global variable used for timing
extern long long time2;       /// Global variable used for timing
extern long long time3;       /// Global variable used for timing
extern long long time4;       /// Global variable used for timing


typedef struct cuFfdot
{
    cudaStream_t stream;        // CUDA stream
    fcomplexcu* dInpFFT;        // The number of complex numbers in each kernel (2 floats)
    ulong inputLen;

    float* dPowers;             // The number of complex numbers in each kernel (2 floats)
    ulong powersLen;

    fcomplexcu* dSpreadFFT;     // The number of complex numbers in each kernel (2 floats)
    ulong spreadLen;

    int inds;                   // The offset of the y offset in constant memory
    int ffdBuffre;              // The offset of the y offset in constant memory
    float harmFraction;         // The offset of the y offset in constant memory

    fcomplexcu* ffdot;          // The f-∂f plain this
    ulong ffdotWidth;           // The width of the f-∂f plan
    ulong ffdotStride;          // The stride of the f-∂f plain
    ulong ffdotHeight;          // The height of this f-∂f plain

    float* ffdotPowers;
    float* ffdotMedData;
    ulong ffPowWidth;
    ulong ffPowStride;
    ulong ffPowHeight;

    int planMany;               // cufft plan of length width and height
    int plan1D;                 // cufft plan of length width and height

    cand* canidates;            /// A list of the candidates

    //size_t stride;      // The x stride in bytes
    //size_t height;      // The number if rows (Z's)
    //float zmax;         // The maximum (and min) z
    //float dz;           // The distance between z elements
    //float harmFrac;     // The harmonic fraction
    //int planMany;       // cufft plan of length width and height
    //int plan1D;         // cufft plan of length width and height
    //int halfWidth;      // The kernel half width
    //fcomplexcu* data;   // The kernels themselves
    //unsigned short *rinds; /* Table of indices for Fourier Freqs */
} cuFfdot;

typedef struct cuFfdotStackStr
{
    int noInStack;              /// The number of plains in this stack
    int startR[MAX_IN_STACK];   /// The heights of the individual plains
    float zMax[MAX_IN_STACK];   /// The heights of the individual plains

} cuFfdotStackStr;

typedef struct cuStackHarms
{
    int noInStack;
    size_t width;                /// The number of complex numbers in each kernel (2 floats)
    size_t stride;               /// The x stride in complex numbers
    size_t height;               /// The number if rows (Z's)

    size_t start[MAX_IN_STACK];  /// The heights of the individual plains
    size_t zmax[MAX_IN_STACK];   /// The heights of the individual plains
    float frac[MAX_IN_STACK];    /// The heights of the individual plains
} cuStackHarms;

typedef struct cuSearchList
{
    tHarmList texs;           ///
    cHarmList datas;          ///
    iHarmList yInds;          ///
    fHarmList frac;           ///
    iHarmList heights;        ///
    iHarmList widths;         ///
    iHarmList strides;        ///
    iHarmList ffdBuffre;      ///
    iHarmList zMax;           ///
    iHarmList fullRLow;       ///
    iHarmList rLow;           ///

    fHarmList idxSum;

    double     searchRLow;      /// The value of the r bin to start the search at
} cuSearchList;

typedef struct cuSearchItem
{
    fCplxTex tex;         ///
    fcomplexcu* data;     ///
    int yInd;             ///
    float frac;           ///
    int height;           ///
    int width;            ///
    int stride;           ///
    int ffdBuffre;        ///
    int zMax;             ///

    //int fullRLow;       ///
    //int rLow;           ///

    //float idxSum;

    //double     searchRLow;      /// The value of the r bin to start the search at
} cuSearchItem;


//typedef  cuSearchItem[2]  schb;
//typedef  cuSearchItem[4]  sch4;
//typedef  cuSearchItem[8]  sch8;
//typedef  cuSearchItem[16] sch16;

typedef struct sch1
{
    cuSearchItem arry[1];
} sch1;

typedef struct sch2
{
    cuSearchItem arry[2];
} sch2;

typedef struct sch4
{
    cuSearchItem arry[4];
} sch4;

typedef struct sch8
{
    cuSearchItem arry[8];
} sch8;

typedef struct sch16
{
    cuSearchItem arry[16];
} sch16;

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

typedef struct cuFfdot10
{
    cuFfdot arr[MAX_HARM_NO];
} cuFfdot10;

typedef struct primaryInf
{
  fcomplexcu* data;
  float fRlow;
  float fZlow;
  int width;
  int stride;
  int height;
  int ffdBuffre;
} primaryInf;


//__device__ __constant__ int        YINDS[MAX_YINDS];
//__device__ __constant__ float      POWERCUT[MAX_HARM_NO];
//__device__ __constant__ long long  NUMINDEP[MAX_HARM_NO];


__device__ volatile int g_canSem          = SEMFREE;
__device__ int g_canCount                 = 0;
__device__ int g_canCount_aut             = 0;
__device__ volatile int can_count_total   = 0;
__device__ int can_count2                 = 0;
__device__ uint g_max                     = 0;

cuHarmInfo* createsubharminfos(int numharmstages, int zmax, accelobs* obs);

/** Create the stacks to do the
 *
 * @param numharmstages
 * @param zmax
 * @param obs
 * @return
 */
cuHarmInfo* createStacks(int numharmstages, int zmax, accelobs* obs);


/* Calculate the 'r' you need for subharmonic  */
/* harm_fract = harmnum / numharm if the       */
/* 'r' at the fundamental harmonic is 'rfull'. */
__host__ __device__ static /*inline*/ double calc_required_r_gpu(double harm_fract, double rfull)
{
  return (int) ( ((double)ACCEL_RDR) * rfull * harm_fract + 0.5) * ((double)ACCEL_DR);
}

/* Return an index for a Fourier Freq given an array that */
/* has stepsize ACCEL_DR and low freq 'lor'.              */
__host__ __device__ /*static*/inline float index_from_r(float r, float lor)
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
 * @param z
 * @param loz the low freq 'lor' of the plain
 * @return
 */
__host__ __device__ static inline int index_from_z(float z, float loz)
/* Return an index for a Fourier Fdot given an array that */
/* has stepsize ACCEL_DZ and low freq 'lor'.              */
{
  return (int) ((z - loz) * ACCEL_RDZ + 1e-6);
}

//template<uint FLAGS, typename sType, int noStages, typename stpType>
__global__ void print_YINDS(int no);

/** Return the first value of 2^n >= x
 */
//__host__ __device__ long long next2_to_n(long long x);

//#ifdef _cplusplus
//extern "C"
//#endif
fcomplexcu* prepFFTdata(fcomplexcu *data, uint len, uint len2, cuFfdot* ffdotPlain);

float cuGetMedian(float *data, uint len);

//cuSubharminfo* createsubharminfos(int numharmstages, int zmax);
cuHarmInfo* createsubharminfos(int numharmstages, int zmax, accelobs* obs);

//int init_harms(cuSubharminfo* hInf, int noHarms);
//int init_harms(cuSubharminfo* hInf, int noHarms, accelobs * obs);
int init_harms(cuHarmInfo* hInf, int noHarms, accelobs *obs);



float* ffdot_planeCU(int harm, double searchRLow, double fullrhi, cuHarmInfo* hInf, int norm_type, fcomplexcu* fft, cuFfdot* ffdotPlain);
//float* ffdot_planeCU2(cuStackList* plains, double searchRLow, float fullrhi, int norm_type, int search, fcomplexcu* fft);

ExternC int ffdot_planeCU2(cuStackList* plains, double searchRLow, double searchRHi, int norm_type, int search, fcomplexcu* fft, accelobs * obs, GSList** cands);
ExternC int ffdot_planeCU3(cuStackList* plains, double* searchRLow, double* searchRHi, int norm_type, int search, fcomplexcu* fft, accelobs * obs, GSList** cands);

int add_ffdot_planeCU(int harm, cuHarmInfo* hInf, cuFfdot* fund, cuFfdot* ffdotPlain, double searchRLow);

int add_ffdot_planeCU2(cuFfdot* fund, int firstSub, int noSubs, cuHarmInfo* hInf, double searchRLow);


int add_and_search(cuFfdot* plains, int stages, cuHarmInfo* hInf, double searchRLow, int copyBack, int search);

/*
template<int noStages, int canMethoud>
__global__ void add_and_searchCU3(cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base);
*/

int drawPlainPowers(cuFfdot* ffdotPlain, char* name);
int drawPlainCmlx(cuFfdot* ffdotPlain, char* name);
int drawPlainCmplx(fcomplexcu* ffdotPlain, char* name, int stride, int height);

double _GammaP (double n, double x);
double _GammaQ (double n, double x);

void printData_cu(cuStackList* stkLst, const int FLAGS, int harmonic, int nX = 10, int nY = 5, int sX = 0, int sY = 0);


//int setYINDS( cuStackList* stkLst, int numharmstages );
int setConstVals( cuStackList* stkLst, int numharmstages, float *powcut, long long *numindep );


///////////////////////////////////////////// Convolution Prototypes \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

void convolveStack(cuStackList* plains, accelobs * obs, GSList** cands);

/*
__global__ void convolveffdot(fcomplexcu *ffdot, const int width, const int stride, const int height, const fcomplexcu *data, const fcomplexcu *kernels);

__global__ void convolveffdot2(fcomplexcu *ffdot, uint width, uint stride, uint height, fcomplexcu *data, fcomplexcu *kernels, uint number);

__global__ void convolveffdot3(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, const fcomplexcu *kernels);

__global__ void convolveffdot35(fcomplexcu *ffdot, uint width, uint stride, const uint height, const fcomplexcu *data, const fcomplexcu *kernels);

__global__ void convolveffdot37(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, const fcomplexcu *kernels);

__global__ void convolveffdot36(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, fCplxTex kerTex);

template <int noBatch>
__global__ void convolveffdot38(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, const fcomplexcu *kernels);

template<uint FLAGS, uint no>
__global__ void convolveffdot4(const fcomplexcu *kernels, const fcomplexcu *datas, fcomplexcu *ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, fCplxTex kerTex);

__host__ void convolveffdot41_f(dim3 dimBlock, dim3 dimGrid, int i1, cudaStream_t cnvlStream, const fcomplexcu *kernels, const fcomplexcu *datas, fcomplexcu *ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, fCplxTex kerTex, uint noSteps, uint noPlns, uint FLAGS );

template<uint FLAGS>
__global__ void convolveffdot5(const fcomplexcu *kernels, const fcomplexcu *datas, fcomplexcu *ffdot, iHarmList widths, iHarmList strides, iHarmList heights, uint no, uint hh, uint noSteps, uint step );

template<uint FLAGS, uint no>
__global__ void convolveffdot6(const fcomplexcu *kernels, const fcomplexcu *datas, cHarmList ffdot, const int width, const uint stride, iHarmList heights, const uint hh, fCplxTex kerTex, iHarmList zUp, iHarmList zDn );

__host__ void convolveffdot7_f(dim3 dimBlock, dim3 dimGrid, int i1, cudaStream_t cnvlStream, const fcomplexcu *kernel, const fcomplexcu *datas, cHarmList ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, fCplxTex kerTex, iHarmList zUp, iHarmList zDn, const uint noSteps, const uint noPlns, uint FLAGS );
*/

//////////////////////////////////////////////////////////// \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


//__host__
__host__ __device__ double incdf (double p, double q );
//__device__ double incdf (double p, double q );

/** Select the GPU to use
 * @param device The device to use
 * @param print if set to 1 will print the name and details of the device
 * @return The SMX version of the decide
 */
ExternC int selectDevice(int device, int print);

ExternC void printCands(char* fileName, GSList *candsCPU);

void testVL();

ExternC void printContext();

ExternC void setContext(cuStackList* stkList) ;

ExternC void testzm();


#endif // CUDA_ACCEL_UTILS_INCLUDED
