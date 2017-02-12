/**   cuda_accel_utils.h
 *
 * This file contains all the common defines and specific to cuda
 *
 */

#ifndef CUDA_ACCEL_UTILS_INCLUDED
#define CUDA_ACCEL_UTILS_INCLUDED

#include <stdio.h>
#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>

#ifdef WITHOMP
#include <omp.h>
#endif

#include "cuda_accel.h"
#include "cuda_utils.h"
#include "candTree.h"
#include "cuda_math.h"


#ifdef CBL
#include "array.h"
#include "arrayDsp.h"
#include "log.h"
#endif

extern "C"
{
#define __float128 long double
#include "accel.h"
}

//#pragma once

#define CNV_DIMX        16                    // X Thread Block
#define CNV_DIMY        8                     // Y Thread Block

/**  Details of the Normalise and spread kernel  .
 * Should be as large as possible usually 32x32
 * NOTE: If compiled in debug mode it may be necessary to drop NAS_DIMY to 16
 *       else you may get Error "too many resources requested for launch"
 */
#define NAS_DIMX        32                    // Normalise and spread X dimension
#define NAS_DIMY        32                    // Normalise and spread Y dimension
#define NAS_NTRD        (NAS_DIMX*NAS_DIMY)   // Normalise and spread thread per block

#define MAX_CANDS_PER_BLOCK 6000000

#define BLOCKSIZE       16
#define BLOCK1DSIZE     BLOCKSIZE*BLOCKSIZE

#define BS_DIM          1024    // compute 3.x +
//#define BS_DIM          576   // compute 2.x

#define MAX_SAS_BLKS	1024

typedef enum {
  IM_TOP,
  IM_BOT,
} ImPlane;



typedef struct iList
{
    int val[MAX_IN_STACK];
} iList;

typedef struct fList
{
    float val[MAX_IN_STACK];
} fList;

//---------- LONG -------- \\

typedef struct long01
{
    long val[1];
} long01;

typedef struct long02
{
    long val[2];
} long02;

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

typedef struct long96
{
    long val[96];
} long96;

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

//-------- POINTER ------- \\

typedef struct ptr01
{
    void* val[1];
} ptr01;

typedef struct ptr02
{
    void* val[2];
} ptr02;

typedef struct ptr04
{
    void* val[4];
} ptr04;

typedef struct ptr08
{
    void* val[8];
} ptr08;

typedef struct ptr16
{
    void* val[16];
} ptr16;

typedef struct ptr32
{
    void* val[32];
} ptr32;

typedef struct ptr64
{
    void* val[64];
} ptr64;

typedef struct ptr128
{
    void* val[128];
} ptr128;

//------------- Arrays that can be passed to kernels -------------------\\

typedef struct vHarmList
{
    void* __restrict__ val[MAX_HARM_NO];
    __host__ __device__ inline void* operator [](const int idx) { return val[idx]; }
} vHarmList;

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
    __host__ __device__ inline float* operator [](const int idx) { return val[idx]; }
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

//-------- POINTER ------- \\

typedef struct cuSearchList
{
    tHarmList   texs;           ///
    cHarmList   datas;          ///
    fsHarmList  powers;         ///
    iHarmList   yInds;          ///
    fHarmList   frac;           ///
    iHarmList   heights;        ///
    iHarmList   widths;         /// The width of usable r values in each plane
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


//======================================= Constant memory =================================================\\

//-------------------  Details in Family order  ------------------------\\

extern __device__ __constant__ int	HEIGHT_HARM[MAX_HARM_NO];		///< Plane  heights   in family
extern __device__ __constant__ int	STRIDE_HARM[MAX_HARM_NO];		///< Plane  strides   in family
extern __device__ __constant__ int	WIDTH_HARM[MAX_HARM_NO];		///< Plane  strides   in family
extern __device__ __constant__ void*	KERNEL_HARM[MAX_HARM_NO];		///< Kernel pointers  in family
extern __device__ __constant__ int	KERNEL_OFF_HARM[MAX_HARM_NO];		///< The offset of the first row of each plane in thier respective kernels

//--------------------  Details in stage order  ------------------------\\

extern __device__ __constant__ float	POWERCUT_STAGE[MAX_HARM_NO];		///<
extern __device__ __constant__ float	NUMINDEP_STAGE[MAX_HARM_NO];		///<
extern __device__ __constant__ int	HEIGHT_STAGE[MAX_HARM_NO];		///< Plane heights in stage order
extern __device__ __constant__ int	STRIDE_STAGE[MAX_HARM_NO];		///< Plane strides in stage order
extern __device__ __constant__ int	PSTART_STAGE[MAX_HARM_NO];		///< Plane half width in stage order

//-------------------  In-mem constant values  -------------------------\\

extern __device__ __constant__ void*	PLN_START;				///< A pointer to the start of the in-mem plane
extern __device__ __constant__ uint	PLN_STRIDE;				///< The strided in units of the in-mem plane
extern __device__ __constant__ int	NO_STEPS;				///< The number of steps used in the search  -  NB: this is specific to the batch not the search, but its only used in the inmem search!
extern __device__ __constant__ int	ALEN;					///< CUDA copy of the accelLen used in the search

//-------------------  Other constant values  --------------------------\\

extern __device__ __constant__ stackInfo	STACKS[64];			///< Stack infos
extern __device__ __constant__ int		YINDS[MAX_YINDS];		///< Z Indices in int

extern __device__ __constant__ int		STK_STRD[MAX_STACKS];		///< Stride of the stacks
extern __device__ __constant__ char		STK_INP[MAX_STACKS][4069];	///< input details


//======================================= Constant Values =================================================\\

const int   stageOrder[16]        =  { 0, 8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15};
const float HARM_FRAC_FAM[16]     =  { 1.0f, 0.9375f, 0.875f, 0.8125f, 0.75f, 0.6875f, 0.625f, 0.5625f, 0.5f, 0.4375f, 0.375f, 0.3125f, 0.25f, 0.1875f, 0.125f, 0.0625f } ;
const short STAGE_CPU[5][2]       =  { {0,0}, {1,1}, {2,3}, {4,7}, {8,15} } ;
const short CHUNKSZE_CPU[5]       =  { 4, 8, 8, 8, 8 } ;

const float HARM_FRAC_STAGE[16]   =  { 1.0000f, 0.5000f, 0.7500f, 0.2500f, 0.8750f, 0.6250f, 0.3750f, 0.1250f, 0.9375f, 0.8125f, 0.6875f, 0.5625f, 0.4375f, 0.3125f, 0.1875f, 0.0625f } ;
//const float h_FRAC_STAGE[16]      =  { 1.0000f, 0.5000f, 0.2500f, 0.7500f, 0.1250f, 0.3750f, 0.6250f, 0.8750f, 0.0625f, 0.1875f, 0.3125f, 0.4375f, 0.5625f, 0.6875f, 0.8125f, 0.9375f } ;


//========================================= Global vals ===================================================\\

extern long long time1;       /// Global variable used for timing
extern long long time2;       /// Global variable used for timing
extern long long time3;       /// Global variable used for timing
extern long long time4;       /// Global variable used for timing

extern int    globalInt01;
extern int    globalInt02;
extern int    globalInt03;
extern int    globalInt04;
extern int    globalInt05;

extern float  globalFloat01;
extern float  globalFloat02;
extern float  globalFloat03;
extern float  globalFloat04;
extern float  globalFloat05;

//extern int    optpln01;
//extern int    optpln02;
//extern int    optpln03;
//extern int    optpln04;
//extern int    optpln05;
//extern int    optpln06;
//
//extern float  downScale;
//
//extern float  optSz01;
//extern float  optSz02;
//extern float  optSz04;
//extern float  optSz08;
//extern float  optSz16;

//extern int    pltOpt;
//extern int    skpOpt;

//====================================== Inline functions ================================================\\

/** Return the first value of 2^n >= x
 */
__host__ __device__ static inline long long cu_next2_to_n(long long x)
{
  long long i = 1;

  while (i < x)
    i <<= 1;

  return i;
}

template<typename T>
__host__ __device__ static inline int cu_z_resp_halfwidth_low(T z)
{
  int m;

  z = fabs_t(z);

  m = (long) (z * ((T)0.00089 * z + (T)0.3131) + NUMFINTBINS);
  m = (m < NUMFINTBINS) ? NUMFINTBINS : m;

  // Prevent the equation from blowing up in large z cases

  if (z > (T)100 && m > (T)0.6 * z)
    m = (T)0.6 * z;

  return m;
}

template<typename T>
__host__ __device__ static inline int cu_z_resp_halfwidth_high(T z)
{
  int m;

  z = fabs_t(z);

  m = (long) (z * ((T)0.002057 * z + (T)0.0377) + NUMFINTBINS * 3);
  m += ((NUMLOCPOWAVG >> 1) + DELTAAVGBINS);

  /* Prevent the equation from blowing up in large z cases */

  if (z > (T)100 && m > (T)1.2 * z)
    m = (T)1.2 * z;

  return m;
}

template<typename T>
__host__ __device__ static inline int cu_z_resp_halfwidth(T z, presto_interp_acc accuracy)
{
  if ( accuracy == LOWACC )
    return cu_z_resp_halfwidth_low<T>(z);
  else
    return cu_z_resp_halfwidth_high<T>(z);
}

/* Calculate the 'r' you need for subharmonic  */
/* harm_fract = harmnum / numharm if the       */
/* 'r' at the fundamental harmonic is 'rfull'. */
__host__ __device__ static inline double cu_calc_required_r(double harm_fract, double rfull, int noResPerBin)
{
  return (int) ( ((double)noResPerBin) * rfull * harm_fract + 0.5) / ((double)noResPerBin);
}

/* Return an index for a Fourier Freq given an array that */
/* has stepsize ACCEL_DR and low freq 'lor'.              */
__host__ __device__ static inline float cu_index_from_r(float r, float lor)
{
  return /* (int) */((r - lor) * (float)ACCEL_RDR /* + 1e-6 */);
}

///* Calculate the 'z' you need for subharmonic  */
///* harm_fract = harmnum / numharm if the       */
///* 'z' at the fundamental harmonic is 'zfull'. */
//__host__ __device__ static inline int calc_required_z(float harm_fract, float zfull, float dz)
//{
//  return ( round( zfull * harm_fract / dz ) * dz );
//}

/* Calculate the 'z' you need for subharmonic  */
template<typename T>
__host__ __device__ static inline T cu_calc_required_z(T harm_fract, T zfull, T dz)
{
  return ( round_t( zfull * harm_fract / dz ) * dz );
}


/** Calculate the index for a given z value of a f-∂f plane  .
 *
 *  Return an index offset for a Fourier Fdot given the step size and
 *  start location
 *
 * @param z
 * @param loz the low freq 'lor' of the plane
 * @return
 */
template<typename T>
__host__ __device__ static inline int cu_index_from_z(T z, T zStart, T dz)
{
  //return (int) (fabsf(z - zStart) / dz + 1e-6);
  return (lround_t(fabs_t(z - zStart) / dz));
}

/* The fft length needed to properly process a subharmonic */
template<typename T>
__host__ __device__ static inline int cu_calc_fftlen(T harm_fract, T max_zfull, uint accelLen, presto_interp_acc accuracy, int noResPerBin, T dz)
{
  int bins_needed, end_effects;

  bins_needed = accelLen * harm_fract + 2;
  float subZ = cu_calc_required_z<T>(harm_fract, max_zfull, dz);
  int halfwidth = cu_z_resp_halfwidth(subZ, accuracy );
  end_effects = 2 * noResPerBin * halfwidth ;
  return cu_next2_to_n(bins_needed + end_effects);
}


//////////////////////////////////////// Getter & setters \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

__device__ inline float get(float* __restrict__ adress, int offset)
{
  return adress[offset];
}

__device__ inline float getLong(float* __restrict__ adress, unsigned long long offset)
{
  return adress[offset];
}

__device__ inline void set(float* adress, uint offset, float value)
{
  adress[offset] = value;
}

__device__ inline float getPower(float* adress, uint offset)
{
  return adress[offset];
}

__device__ inline fcomplexcu get(fcomplexcu* __restrict__ adress, int offset)
{
  return adress[offset];
}

__device__ inline fcomplexcu getLong(fcomplexcu* __restrict__ adress, unsigned long long offset)
{
  return adress[offset];
}



__device__ inline void set(fcomplexcu* adress, uint offset, fcomplexcu value)
{
  adress[offset] = value;
}

__device__ inline float getPower(fcomplexcu* adress, uint offset)
{
  return POWERC(adress[offset]);
}

#if CUDA_VERSION >= 7050   // Half precision getter and setter  .

#ifdef __CUDACC__

__device__ inline float get(half* __restrict__ adress, int offset)
{
  return __half2float(adress[offset]);
}

__device__ inline float getLong(half* __restrict__ adress, unsigned long long offset)
{
  return __half2float(adress[offset]);
}

__device__ inline void set(half* adress, uint offset, float value)
{
  adress[offset] = __float2half(value);
}

__device__ inline float getPower(half* adress, uint offset)
{
  return __half2float(adress[offset]);
}
#endif

#endif  // CUDA_VERSION >= 7050


//===================================== Function Prototypes ===============================================\\

/////////////////////////////////////// Utility prototypes \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

float half2float(const ushort h);

/** Set up the threading  .
 *
 */
void intSrchThrd(cuSearch* srch);

/** Set the active batch  .
 *
 */
void setActiveBatch(cuFFdotBatch* batch, int rIdx = 0);

/** Cycle the arrays of r-values  .
 *
 * @param batch
 */
void cycleRlists(cuFFdotBatch* batch);

/** Cycle the arrays of r-values  .
 *
 * @param batch
 */
void cycleOutput(cuFFdotBatch* batch);

/** Select the GPU to use  .
 * @param device The device to use
 * @param print if set to 1 will print the name and details of the device
 * @return The SMX version of the decide
 */
int selectDevice(int device, int print);

int cu_calc_fftlen(double harm_fract, int max_zfull, uint accelLen, presto_interp_acc accuracy, int noResPerBin, float dz); 

void printContext();

/** Write CUFFT call backs to device  .
 */
void copyCUFFT_LD_CB(cuFFdotBatch* batch);

/** Create the stacks to do the  .
 *
 * @param numharmstages
 * @param zmax
 * @param obs
 * @return
 */
cuHarmInfo* createStacks(int numharmstages, int zmax, accelobs* obs);

//int ffdot_planeCU2(cuFFdotBatch* planes, double searchRLow, double searchRHi, int search, fcomplexcu* fft, accelobs * obs, GSList** cands);

/** Initialise the pointers of the stacks data structures of a batch  .
 *
 * This assumes the various memory blocks of the batch have been created
 *
 * @param batch
 */
void setStkPointers(cuFFdotBatch* batch);

/** Initialise the pointers of the planes data structures of a batch  .
 *
 * This assumes the stack pointers have already been setup
 *
 * @param batch
 */
void setPlanePointers(cuFFdotBatch* batch);

/** Initialise the pointers of the stacks and planes data structures of a batch  .
 *
 * This assumes the various memory blocks of the batch have been created
 *
 * @param batch
 */
void setBatchPointers(cuFFdotBatch* batch);

void setKernelPointers(cuFFdotBatch* batch);

/** Print a integer in binary  .
 *
 * @param val The value to print
 */
void printBitString(uint val);

/**
 *
 * @param kernel  .
 */
void freeKernel(cuFFdotBatch* kernel);



/////////////////////////////////////// Kernel prototypes \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

void createBatchKernels(cuFFdotBatch* batch, void* buffer);

/** Create one GPU kernel. One kernel the size of the largest plane  .
 *
 * @param kernel
 * @return
 */
int createStackKernel(cuFfdotStack* cStack);

/** Calculate the step size from a width if the width is < 100 it is skate to be the closest power of two  .
 *
 * @param width
 * @param zmax
 * @return
 */
uint calcAccellen(float width, float zmax, presto_interp_acc accuracy, int noResPerBin);

/** Calculate the optimal step size from a desired plane width
 *
 * This calculates the optimal step size from a desired plane width.
 * This applies the rule that the second plane should fall in the second stack (if there is more than one plane in the family).
 * If you want to ignore this rule just set noHarms to one.
 *
 * If hamrDevis is true, the step size will be scaled down until it is divisible by the number of harmonics, as is need by the SS10 Kernel.
 *
 * If the width is greater than 100 it it is assumed the user is manually specifying the step size.
 * This should only be used for DEBUG purposes, if hamrDevis is set to true the "manual" step size WILL be scaled back.
 *
 * @param width		The width, if < 100 plane width in k, if not assumed to be the actual step size.
 * @param zmax		The zmax of the plane (should be divisible by zRes)
 * @param noHarms	The number of harmonics being summed
 * @param accuracy	LOWACC or HIGHACC - Determines the size and shape of the response function
 * @param noResPerBin	The number of r per FT bin, must be an integer, defaults to 2 (interbinning)
 * @param zRes		The z resolution (defaults to 2)
 * @param hamrDevis	Weather to make the step size divisible by the number of harmonics summed, as is need by the GPU SS10 kernel
 * @return 		The step optimal size
 */
uint calcAccellen(float width, float zmax, int noHarms, presto_interp_acc accuracy, int noResPerBin, float zRes, bool hamrDevis);


///////////////////////////////////////// Init prototypes \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

float cuGetMedian(float *data, uint len);

void setGenRVals(cuFFdotBatch* batch);

void setSearchRVals(cuFFdotBatch* batch, double searchRLow, long len);

void prepInputCPU(cuFFdotBatch* batch);

void copyInputToDevice(cuFFdotBatch* batch);

void prepInputGPU(cuFFdotBatch* batch);

/** Initialise input data for a f-∂f plane(s)  ready for multiplication  .
 * This:
 *  Normalises the chunk of input data
 *  Spreads it (interbinning)
 *  FFT it ready for multiplication
 *
 * @param planes      The planes
 * @param searchRLow  The index of the low  R bin (1 value for each step)
 * @param searchRHi   The index of the high R bin (1 value for each step)
 * @param fft         The fft
 */
void prepInput(cuFFdotBatch* batch);



////////////////////////////////////// Multiplication Prototypes \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

int setConstVals_Fam_Order( cuFFdotBatch* batch );

int setStackVals( cuFFdotBatch* batch );

/** Multiply and inverse FFT the complex f-∂f plane using FFT callback  .
 * @param planes
 */
void multiplyBatchCUFFT(cuFFdotBatch* batch );

/** Multiply the complex f-∂f plane  .
 * This assumes the input data is ready and on the device
 * This writes to the complex f-∂f plane
 *
 * If FLAG_CONV flag is set and doing stack multiplications, the iFFT will be called directly after the multiplication for each stack
 */
void multiplyBatch(cuFFdotBatch* batch );

/**  iFFT a specific stack  .
 *
 * @param batch
 * @param cStack
 * @param pStack
 */
void IFFTStack(cuFFdotBatch* batch, cuFfdotStack* cStack, cuFfdotStack* pStack = NULL);

/**  iFFT all stack of a batch  .
 *
 * If using the FLAG_CONV flag no iFFT is done as this should have been done by the multiplication
 *
 * This assumes the input data is ready and on the device
 * This creates a complex f-∂f plane
 */
void IFFTBatch(cuFFdotBatch* batch );

void copyToInMemPln(cuFFdotBatch* batch );

/** Multiply and inverse FFT the complex f-∂f plane  .
 * This assumes the input data is ready and on the device
 * This creates a complex f-∂f plane
 */
void convolveBatch(cuFFdotBatch* batch );



//////////////////////////////////// Sum and search Prototypes \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

int setConstVals( cuFFdotBatch* stkLst );

void processBatchResults(cuFFdotBatch* batch );

void getResults(cuFFdotBatch* batch );

void sumAndSearch(cuFFdotBatch* batch );

void sumAndSearchOrr(cuFFdotBatch* batch);

/** A function to call a kernel to harmonically sum a plan and return the max of each column  .
 *
 */
void sumAndMax(cuFFdotBatch* planes, long long *numindep, float* powers);



//////////////////////////////////////// Optimisation \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

candTree* opt_cont(candTree* oTree, cuOptCand* pln, container* cont, fftInfo* fft, int nn = 0 );

void opt_genResponse(cuRespPln* pln, cudaStream_t stream);

void freeHarmInput(cuHarmInput* inp);



//////////////////////////////////////////// Stats \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

__host__ __device__ double incdf (double p, double q );
double candidate_sigma_cu(double poww, int numharm, long long numindep);


//////////////////////////////////////// Some other stuff \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\



////////////////////////////////// defines for templated functions \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

template int cu_z_resp_halfwidth_low<float >(float  z);
template int cu_z_resp_halfwidth_low<double>(double z);

template int cu_z_resp_halfwidth_high<float >(float  z);
template int cu_z_resp_halfwidth_high<double>(double z);

template int cu_z_resp_halfwidth<float >(float  z, presto_interp_acc accuracy);
template int cu_z_resp_halfwidth<double>(double z, presto_interp_acc accuracy);


template float  cu_calc_required_z<float >(float  harm_fract, float  zfull, float  dz);
template double cu_calc_required_z<double>(double harm_fract, double zfull, double dz);

template int cu_index_from_z<float >(float  z, float  zStart, float  dz);
template int cu_index_from_z<double>(double z, double zStart, double dz);

template int cu_calc_fftlen<float >(float  harm_fract, float  max_zfull, uint accelLen, presto_interp_acc accuracy, int noResPerBin, float  dz);
template int cu_calc_fftlen<double>(double harm_fract, double max_zfull, uint accelLen, presto_interp_acc accuracy, int noResPerBin, double dz);

#endif // CUDA_ACCEL_UTILS_INCLUDED

