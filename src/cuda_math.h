#ifndef CUTIL_MATH_H
#define CUTIL_MATH_H

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

#ifndef SQRT2
#define SQRT2		1.4142135623730950488016887242096980785696718753769
#endif

#ifndef DBLCORRECT
#define DBLCORRECT	1e-14
#endif

#ifndef PI
#define PI		3.1415926535897932384626433832795028841971693993751
#endif

#ifndef TWOPI
#define TWOPI		6.2831853071795864769252867665590057683943387987502
#endif

#ifndef SQRTTWOPI
#define SQRTTWOPI	2.506628274631000502415765284811045253006986740609
#endif

#ifndef DEGTORAD
#define DEGTORAD	0.017453292519943295769236907684886127134428718885417
#endif

#ifndef RADTODEG
#define RADTODEG	57.29577951308232087679815481410517033240547246656
#endif

#ifndef PIBYTWO
#define PIBYTWO		1.5707963267948966192313216916397514420985846996876
#endif

#define POWERCU(r,i)  ((r)*(r) + (i)*(i))     /// The sum of the powers of two number
#define POWERC(c)     POWERCU(c.r, c.i)       /// The sum of the powers of a complex number
#define POWERF(f)     POWERCU(f.x, f.y)       /// The sum of the powers of a complex number

#ifndef MAX
#define	MAX(a, b)	( ((a)<(b))?(b):(a) )
#endif

// Compare and set
#ifndef MAXX
#define MAXX(a, b)  ( a=(a)<(b)?b:a )
#endif

#ifndef MIN
#define	MIN(a, b)	( ((a)>(b))?(b):(a) )
#endif

// Compare pare and set
#ifndef MINN
#define MINN(a, b)  (a=(a)>(b)?(b):(a) )
#endif

__device__ __forceinline__ unsigned int    __lidd() { unsigned int laneid; asm volatile ("mov.u32 %0, %laneid;" : "=r"(laneid)); return laneid; }
//__device__ __forceinline__ unsigned int __warpid() { unsigned int warpid; asm volatile ("mov.u32 %0, %warpid;" : "=r"(warpid)); return warpid; }
//__device__ __forceinline__ unsigned int   __smid() { unsigned int smid;   asm volatile ("mov.u32 %0, %smid;"   : "=r"(smid));   return smid;   }
//__device__ __forceinline__ unsigned int __gridid() { unsigned int gridid; asm volatile ("mov.u32 %0, %gridid;" : "=r"(gridid)); return gridid; }
//__device__ __forceinline__ unsigned int __tIdBlock() { unsigned int tid; asm volatile ("mad.lo.u32 %ntid.x, %tid.y, %tid.x, %0;" : "=r"(tid)); return tid; }


__host__ __device__ inline float  cos_t(float  x)
{
  return cosf(x);
}
__host__ __device__ inline double cos_t(double x)
{
  return cos(x);
}

__host__ __device__ inline float  sin_t(float  x)
{
  return sinf(x);
}
__host__ __device__ inline double sin_t(double x)
{
  return sin(x);
}

__host__ __device__ inline void sincos_t(float  x, float*  s, float*  c )
{
  sincosf(x, s, c);
}
__host__ __device__ inline void sincos_t(double x, double* s, double* c )
{
  sincos(x, s, c);
}

__host__ __device__ inline float  sqrt_t(float  x)
{
  return sqrtf(x);
}
__host__ __device__ inline double sqrt_t(double x)
{
  return sqrt(x);
}

__host__ __device__ inline float  modf_t ( float  x, float   *y )
{
  return modff(x, y);
}
__host__ __device__ inline double modf_t ( double  x, double *y )
{
  return modf(x, y);
}

__host__ __device__ inline float  fabs_t(float  x )
{
  return fabsf(x);
}
__host__ __device__ inline double fabs_t(double x )
{
  return fabs(x);
}

__host__ __device__ inline float  ceil_t(float  x )
{
  return ceilf(x);
}
__host__ __device__ inline double ceil_t(double x )
{
  return ceil(x);
}

__host__ __device__ inline float  floor_t(float  x )
{
  return floorf(x);
}
__host__ __device__ inline double floor_t(double x )
{
  return floor(x);
}

__host__ __device__ inline float  fmod_t ( float  x, float   y )
{
  return fmodf(x, y);
}
__host__ __device__ inline double fmod_t ( double  x, double y )
{
  return fmod(x, y);
}

__host__ __device__ inline float  round_t ( float  x )
{
  //return rintf(x);
  return roundf(x);
}
__host__ __device__ inline double round_t ( double  x )
{
  //return rint(x);
  return round(x);
}

__host__ __device__ inline int lround_t ( float  x )
{
  //return lrintf(x);
  return lroundf(x);
}
__host__ __device__ inline int lround_t ( double  x )
{
  //return lrint(x);
  return lround(x);
}

#if __CUDA_ARCH__ >= 200
__device__ static float atomicMax(float* address, float val)
{
  int* address_as_i = (int*) address;
  int old = *address_as_i, assumed;
  do
  {
    assumed = old;
    old = atomicCAS(address_as_i, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ static float atomicMin(float* address, float val)
{
  int* address_as_i = (int*) address;
  int old = *address_as_i, assumed;
  do
  {
    assumed = old;
    old = atomicCAS(address_as_i, assumed, __float_as_int(::fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}
#endif

/** Extract k-th bit from i
 *
 */
__device__ inline int bfe(int i, int k)
{
  int ret;
  asm("bfe.u32 %0, %1, %2, 1;" : "=r"(ret) : "r"(i), "r"(k));
  return ret;
}

/** Find most significant non-sign bit from i.
 *
 */
__device__ inline int bfind(int i)
{
  int ret;
  asm("bfind.s32 %0, %1;" : "=r"(ret) : "r"(i));
  return ret;
}

#endif
