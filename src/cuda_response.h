#ifndef CUTIL_CORR_H
#define CUTIL_CORR_H


#define FILIM_F		1e-6
#define FILIM_D		1e-14
#define SINCLIM		1e-6
#define DLIM		0.0               // 0.4


#ifndef FOLD
#define FOLD if(1)
#endif

template<typename T>
__host__ __device__ void fresnl(T xxa, T* ss, T* cc);

template<typename T, uint flags>
__host__ __device__ void calc_z_response(T Qk, T z, T sq2overAbsZ, T PIoverZ, T overSq2AbsZ, int sighnZ, T* real, T* imag);

template<typename T>
__host__ __device__ void calc_r_response(T dist, T sinsin, T sincos, T* real, T* imag);

template<typename T>
__host__ __device__ void calc_response_bin(long bin, double r, float z,  T* real, T* imag);

template<typename T>
__host__ __device__ void calc_response_off(float offset, float z,  T* real, T* imag);

template<typename T, typename dataT>
__host__ __device__ void rz_interp_cu(dataT* inputData, long loR, long noBins, double r, float z, int kern_half_width, T* real, T* imag);

template<typename T, typename outT>
__host__ __device__ void gen_response_cu(double r, float z, int kern_half_width, outT* out);

#endif // CUTIL_CORR_H
