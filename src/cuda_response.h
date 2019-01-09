#ifndef CUTIL_CORR_H
#define CUTIL_CORR_H

#include "cuda_accel.h"
#include "cuda_accel_utils.h"

#define Z_LIM_F		1e-5			// Got this through testing - this is the Z value below which inaccuracy creep in for double
#define Z_LIM_D		1e-6			// I have found that beyond this errors start to creep in for long data sets

#define R_LIM_F		1e-5			// Got this through testing - this is the Z value below which inaccuracy creep in for double
#define R_LIM_D		1e-9			// Got this through testing - this is the Z value below which inaccuracy creep in for double

// Boundary constant estimate term 1 - AKA Fourier interpolation
#define E0_LIM_F	1e-20f			//
#define E0_LIM_D	1e-20			//
#define E0_LIM_Q	1e-20L			//

// Boundary liner estimate - a2.r (2nd coefficient)
#define E1R_LIM_F	0.009f			// Bound: ~9e-3  Error: ~3e-6
#define E1R_LIM_D	0.00006			// Bound: ~6e-5  Error: ~1e-12
#define E1R_LIM_Q	0.00002L		//

// Boundary quadratic estimate - a2.i (2nd coefficient)
#define E1I_LIM_F	0.035f			// Bound: ~3.5e-2  Error: ~6e-6
#define E1I_LIM_D	0.0014			// Bound: ~1.4e-3  Error: ~1e-11
#define E1I_LIM_Q	0.0009L			//

// Boundary for both liner and quadratic estimate - a2 (3rd coefficient)
#define E2_LIM_F	0.06f			// For a2*z*z, bound: ~6e-2 error : ~4e-14 - Bounds of a2: Real bound: ~9e-2 err: ~1e-4 - Imag bound ~0.05 err: ~1e-4
#define E2_LIM_D	0.004			// For a2*z*z, bound: ~4e-3 error : ~7e-18 - Bounds of a2: Real bound: ~7e-3 err: ~3e-9 - Imag bound ~1e-3 err: ~3e-8
#define E2_LIM_Q	0.004L			//

#define SINCLIM		1e-5			// x value below which to call sinc(x) 1

#define FRES_DOUBLE	256			// DEPRICATED: When to use double precision calculations for phase calculation in evaluating fresnel integrals // was 2e2


#define FREESLIM1	1.600781059358212171622054418655453316130105033155		// Sqrt ( 2.5625 ) - I have found the maximum error for float, of the two methods crosses at ~1.11 while 1.6 works well for float and double
//#define FREESLIM2	36974.0
//#define FREESLIM2	262144.0

#define FRESLIM2_F	5000000.0		// This value marks the point when there Fresnel amplitude is less than single precision to differentiate from 0.5 - This assumes sqMod4 is used
#define FRESLIM2_D	5000000000.0		// This value marks the point when there Fresnel amplitude is less than double precision to differentiate from 0.5 - This assumes sqMod4 is used


#define FRES_SINGLE_PHASE	64		// Start of single precision phase adjustments. Good values: 1024 - 2048 - 4096 - This is going to go down to 64!

#define RESP_SINGLE_PHASE	32768		// Transition from Single to Double Precision phase adjustments. Good values - 2^13 8192 - 2^14 16384 - 2^15 32768 - 2^16 65536 - 2^17 131072 - 2^18 262144 - 2^19 524288 - 2^21 2097152


#ifndef FOLD
#define FOLD if(1)
#endif

#ifdef		OPT_LOC_32

#define		OPT_MAX_LOC_HARMS	32
typedef		int32  optLocInt_t;
typedef		long32 optLocLong_t;
typedef		int32  optLocFloat_t;

#else

#define		OPT_MAX_LOC_HARMS	16
typedef		int16  optLocInt_t;
typedef		long16 optLocLong_t;
typedef		int16  optLocFloat_t;

#endif

//#ifdef DEBUG
//
//#define DEVIS 1
//#define REPS 1
//
//#else

// DBG testing parameters
#define DEVIS 5
#define REPS 20

//#define DEVIS 1
//#define REPS 1

#define OPS_P_REP (DEVIS*REPS)

struct kerStruct
{
  double	devGOps;
  int		reps;
  float		repScale;

  int		iList[3];
  float		fList[3];
  double	dList[3];

  dim3		dimBlock;
  dim3		dimGrid;

  void*		h_buffer1;
  void*		d_buffer1;

  void*		h_buffer2;
  void*		d_buffer2;

  void*		h_buffer3;
  void*		d_buffer3;
};

//#endif

/////////////////////////////////

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val);
#endif

/////////////////////////////////

__host__ __device__ inline float getE0lim(float nothing)
{
  return E0_LIM_F;
}

__host__ __device__ inline double getE0lim(double nothing)
{
  return E0_LIM_D;
}

__host__            inline long double getE0lim(long double nothing)
{
  return E0_LIM_Q;
}

__host__ __device__ inline float getE1iLim(float nothing)
{
  return E1I_LIM_F;
}

__host__ __device__ inline double getE1iLim(double nothing)
{
  return E1I_LIM_D;
}

__host__            inline long double getE1iLim(long double nothing)
{
  return E1I_LIM_Q;
}

__host__ __device__ inline float getE1rLim(float nothing)
{
  return E1R_LIM_F;
}

__host__ __device__ inline double getE1rLim(double nothing)
{
  return E1R_LIM_D;
}

__host__            inline long double getE1rLim(long double nothing)
{
  return E1R_LIM_Q;
}

__host__ __device__ inline float getE2lim(float nothing)
{
  return E2_LIM_F;
}

__host__ __device__ inline double getE2lim(double nothing)
{
  return E2_LIM_D;
}

__host__            inline long double getE2lim(long double nothing)
{
  return E2_LIM_Q;
}

__host__ __device__ float getZlim(float r);
__host__ __device__ double getZlim(double r);
__host__            long double getZlim(long double r);

/////////////////////////////////

template<typename T, typename idxT>
__host__ __device__ void fresnl(idxT x, T* cc, T* ss);

/////////////////////////////////

template<typename T>
__host__ __device__ void calc_coefficient(T offset, T z, T* real, T* imag);

__host__ __device__ double2 calc_coefficient(double offset, double z);

__host__ __device__ float2  calc_coefficient(float  offset, float  z);

template<typename T>
__host__ __device__ void calc_coefficient_bin(long bin, double r, T z,  T* real, T* imag);

/////////////////////////////////

template<typename T>
__host__ __device__ void calc_coefficient_r(T offset, T* real, T* imag);

template<typename T, bool phaseCheck>
__host__ __device__ void calc_coefficient_z(T offset, T z, T* real, T* imag);

template<typename T, bool phaseCheck>
__host__ __device__ inline void calc_coefficient_z(T Qk, T dr, T z, T sq2overAbsZ, T overSq2AbsZ, int sighnZ, T* real, T* imag);

template<typename T>
__host__ __device__ void calc_coefficient_a(T offset, T z, T* resReal, T* resImag);

/////////////////////////////////

template<typename T, typename dataT>
__host__ __device__ void rz_convolution_cu(const dataT* inputData, long loR, long noBins, double r, T z, int kern_half_width, T* real, T* imag);

template<int noColumns>
__host__ __device__ void rz_convolution_sfl(float2* inputData, const long loR, const long inStride, const double r, const float z, const int kern_half_width, float2* outData, const int colWidth, const int ic, const int cIdx);

template<typename T>
__global__ void ffdotPlnByShfl_ker(void* powers, float2* fft, int noHarms, int harmWidth, double firstR, double firstZ, double zSZ, double rSZ, int noOffsets, int noR, int noZ, int colWidth, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw, uint flags, int noColumns);

template<typename T, int noBlk>
__global__ void ffdotPlnByBlk_ker3(void* powers, float2* fft, int noHarms, int harmWidth, double firstR, double firstZ, double zSZ, double rSZ, int blkDimX, int noR, int noZ, int blkWidth, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw, uint flags);

/////////////////////////////////

template<typename T>
__global__ void k_finterpin(kerStruct inf);			// DBG - prototype

template<typename T, typename T2>
__global__ void k_finterpEval(T* input, T2* output );		// DBG - prototype

template<typename T>
__global__ void k_fresnlin(kerStruct inf);			// DBG - prototype

template<typename T>
__global__ void k_responsein(kerStruct inf);			// DBG - prototype

template<typename T, typename T2>
__global__ void k_fresnEval(T* input, T2* output );		// DBG - prototype

//__global__ void k_fresnEval_f(float* input, float2* output);	// DBG - prototype
//__global__ void k_fresnEval_d(double* input, double2* output );	// DBG - prototype

#endif // CUTIL_CORR_H
