#ifndef CUTIL_CORR_H
#define CUTIL_CORR_H

#define Z_LIM_F		1e-5			// Got this through testing - this is the Z value below which inaccuracy creep in for double
#define Z_LIM_D		1e-6			// I have found that beyond this errors start to creep in for long data sets

#define R_LIM_F		1e-5			// Got this through testing - this is the Z value below which inaccuracy creep in for double
#define R_LIM_D		1e-9			// Got this through testing - this is the Z value below which inaccuracy creep in for double

#define E0_LIM_F	1e-20f			//
#define E0_LIM_D	1e-20			//
#define E0_LIM_Q	1e-20L			//

// Liner term
#define E1R_LIM_F	0.03f			//
#define E1R_LIM_D	0.0001			//
#define E1R_LIM_Q	0.00002L		//

// Quadratic term
#define E1I_LIM_F	0.005f			// Rough error intersection
#define E1I_LIM_D	0.0009			// This could be 0.002 but the noise error is less than the estimation error
#define E1I_LIM_Q	0.0009L			// 0.00002L - 0.00006L

// Both liner and quadratic terms
#define E2_LIM_F	0.1f			//
#define E2_LIM_D	0.01			//
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

__host__ long double getZlim(long double r);

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

template<typename T>
__host__ __device__ void calc_coefficient_a(T offset, T z, T* resReal, T* resImag);

/////////////////////////////////

template<typename T, typename outT>
__host__ __device__ void gen_response_cu(double r, T z, int kern_half_width, outT* out);


template<typename T, typename dataT>
__host__ __device__ void rz_convolution_cu(dataT* inputData, long loR, long noBins, double r, T z, int kern_half_width, T* real, T* imag);

template<typename T, typename dataT>
__host__            void rz_convolution_cu_debg(dataT* inputData, long loR, long noBins, double r, T z, int kern_half_width, T* real, T* imag);

template<typename T, typename dataT>
__host__ __device__ void rz_single_mult_cu(dataT* inputData, long loR, long noBins, double r, T z, int kern_half_width, T* real, T* imag, int i);

template<typename T, typename dataIn, typename dataOut>
__host__ __device__ void rz_convolution_cu(dataIn* inputData, long loR, long inStride, double r, T z, int kern_half_width, dataOut* outData, int blkWidth, int noBlk);

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
