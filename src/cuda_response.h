#ifndef CUTIL_CORR_H
#define CUTIL_CORR_H

#include "cuda_accel_utils.h"

//#define FILIM_F		1e-10		// Testing
//#define FILIM_D		1e-10		// Testing

#define FILIM_F		1e-5			// Got this through testing - this is the Z value below which inaccuracy creep in for double
#define FILIM_D		1e-6			// I have found that beyond this errors start to creep in for long data sets

#define SINCLIM		1e-5			// x value below which to call sinc(x) 1
//#define DLIM		0.0			// 0.4

#define FRES_DOUBLE	2e2			// When to use double precision calculations for phase calculation in evaluating fresnel integrals



#ifndef FOLD
#define FOLD if(1)
#endif

/** Shared device function to get halfwidth for optimisation planes
 *
 * Note this could be templated for accuracy
 *
 * @param z	The z (acceleration) for the relevant halfwidth
 * @param def	If a halfwidth has been supplied this is its value, multiple value could be given here
 * @return	The half width for the given z
 */
template<typename T>
__host__ __device__ inline int getHw(float z, int val)
{
  int halfW;

  if      ( val == LOWACC  )
  {
    halfW	= cu_z_resp_halfwidth_low<T>(z);
  }
  else if ( val == HIGHACC )
  {
    halfW	= cu_z_resp_halfwidth_high<T>(z);
  }
  else
  {
    halfW	= val;
  }

  return halfW;
}

template<typename T, typename idxT>
__host__ __device__ void fresnl(idxT x, T* ss, T* cc);

template<typename T, uint flags>
__host__ __device__ void calc_z_response(long iPart, T Qk, T z, T sq2overAbsZ, T PIoverZ, T overSq2AbsZ, int sighnZ, T* real, T* imag);

template<typename T>
__host__ __device__ void calc_r_response(T dist, T sinsin, T sincos, T* real, T* imag);

template<typename T>
__host__ __device__ void calc_response_bin(long bin, double r, T z,  T* real, T* imag);

template<typename T>
__host__ __device__ void calc_response_off(T offset, T z, T* real, T* imag);

__host__ __device__ double2 calc_response_off(double offset, double z);

__host__ __device__ float2  calc_response_off(float  offset, float  z);

template<typename T, typename outT>
__host__ __device__ void gen_response_cu(double r, T z, int kern_half_width, outT* out);

template<typename T, typename dataT>
__host__ __device__ void rz_convolution_cu(dataT* inputData, long loR, long noBins, double r, T z, int kern_half_width, T* real, T* imag);

template<typename T, typename dataT>
__host__ void rz_convolution_cu_debg(dataT* inputData, long loR, long noBins, double r, T z, int kern_half_width, T* real, T* imag);

template<typename T, typename dataT>
__host__ __device__ void rz_single_mult_cu(dataT* inputData, long loR, long noBins, double r, T z, int kern_half_width, T* real, T* imag, int i);

template<typename T, typename dataIn, typename dataOut>
__host__ __device__ void rz_convolution_cu(dataIn* inputData, long loR, long inStride, double r, T z, int kern_half_width, dataOut* outData, int blkWidth, int noBlk);

template<typename T, typename dataT>
__host__ __device__ void rz_convolution_cu_inc(dataT* inputData, long loR, long noBins, double r, T z, int kern_half_width, T* real, T* imag);



#endif // CUTIL_CORR_H
