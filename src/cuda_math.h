#ifndef CUTIL_MATH_H
#define CUTIL_MATH_H

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
//#include <stdint.h>
//#include <tgmath.h>

//// Constants

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

#ifndef SQRTTWOOVERPI
#define SQRTTWOOVERPI	0.7978845608028653558798921198687637369517172623298
#endif

#ifndef SQRTPIOVERTWO
#define SQRTPIOVERTWO	1.253314137315500251207882642405522626503493370304969158314
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

/// Basic Function

#define POWERCU(r,i)	((r)*(r) + (i)*(i))	/// The sum of the powers of two number
#define POWERC(c)	POWERCU(c.r, c.i)	/// The sum of the powers of a complex number
#define POWERF(f)	POWERCU(f.x, f.y)	/// The sum of the powers of a complex number

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
__device__ __forceinline__ unsigned int  __warpid() { unsigned int warpid; asm volatile ("mov.u32 %0, %warpid;" : "=r"(warpid)); return warpid; }
//__device__ __forceinline__ unsigned int   __smid() { unsigned int smid;   asm volatile ("mov.u32 %0, %smid;"   : "=r"(smid));   return smid;   }
//__device__ __forceinline__ unsigned int __gridid() { unsigned int gridid; asm volatile ("mov.u32 %0, %gridid;" : "=r"(gridid)); return gridid; }
//__device__ __forceinline__ unsigned int __tIdBlock() { unsigned int tid; asm volatile ("mad.lo.u32 %ntid.x, %tid.y, %tid.x, %0;" : "=r"(tid)); return tid; }

#ifdef  __CUDA_ARCH__

__device__ inline float  rcp_t(float  x )
{
  return __frcp_rd(x);
}
__device__ inline double rcp_t(double x )
{
  return __drcp_rd(x);
}

#endif

__host__ __device__ inline float  cos_t(float  x)
{
  return cosf(x);
}
__host__ __device__ inline double cos_t(double x)
{
  return cos(x);
}
__host__            inline long double cos_t(long double x)
{
  return cosl(x);
}

__host__ __device__ inline float  sin_t(float  x)
{
  return sinf(x);
}
__host__ __device__ inline double sin_t(double x)
{
  return sin(x);
}
__host__            inline long double sin_t(long double x)
{
  return sinl(x);
}

__host__ __device__ inline float  tan_t(float  x)
{
  return tanf(x);
}
__host__ __device__ inline double tan_t(double x)
{
  return tan(x);
}
__host__            inline long double tan_t(long double x)
{
  return tanl(x);
}

__host__ __device__ inline void sincos_t(float  x, float*  s, float*  c )
{
  sincosf(x, s, c);
}
__host__ __device__ inline void sincos_t(double x, double* s, double* c )
{
  sincos(x, s, c);
}
__host__            inline void sincos_t(long double x, long double* s, long double* c )
{
  sincosl(x, s, c);
}

__host__ __device__ inline void sincospi_t(float  x, float*  s, float*  c )
{
#ifdef  __CUDA_ARCH__
  sincospif(x, s, c);
#else
  sincosf(x*(float)PI, s, c);
#endif
}
__host__ __device__ inline void sincospi_t(double x, double* s, double* c )
{
#ifdef  __CUDA_ARCH__
  sincospi(x, s, c);
#else
  sincos(x*(double)PI, s, c);
#endif
}

__host__ __device__ inline float  sqrt_t(float  x)
{
  return sqrtf(x);
}
__host__ __device__ inline double sqrt_t(double x)
{
  return sqrt(x);
}
__host__            inline long double sqrt_t(long double x)
{
  return sqrtl(x);
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
__host__            inline long double fabs_t(long double x )
{
  return fabsl(x);
}

__host__ __device__ inline float  pow_t(float x, float y )
{
  return powf(x, y);
}
__host__ __device__ inline double pow_t(double x, double y )
{
  return pow(x, y);
}
__host__            inline long double pow_t(long double x, long double y )
{
  return powl(x, y);
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
  //return llrintf(x);
  //return llrintf(x);	// roundf() maps to an 8-instruction sequence on the device, whereas rintf() maps to a single instruction
}
__host__ __device__ inline double round_t ( double x )
{
  //return rint(x);
  return round(x);
  //return llrint(x);
  //return llrint(x);	// roundf() maps to an 8-instruction sequence on the device, whereas rintf() maps to a single instruction
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

__host__ __device__ inline float sqMod4( float x)
{
#ifdef  __CUDA_ARCH__

  asm("{" 									// use braces to limit scope
      " .reg .u32 r1;"								// temp reg,
      " .reg .f32 f1, f2;"							// temp reg t1,
      " and.b32 		r1,	%1,	2139095040	;"		// Exponent bits
      " shr.b32 		r1,	r1,	23		;"		// Shift to relevant spot
      " sub.s32 		r1,	151,	r1		;"		// r1 = 24 - ( exp - 127 ) -  Remove base - shift 24 bits for mantissa length
      " shr.b32 		f2,	%1,	r1		;"		// f2 = x  >> sft
      " shl.b32 		f1,	f2,	r1		;"		// f2 = f2 << sft
      " sub.f32 		f2,	%1,	f1		;"		// b = x - a
      " fma.rn.f32		f1,	2.0,	f1,	f2	;"		// a = 2*a+b
      " mul.f32			%0,	f1,	f2		;"		// a = a*b
      "}"
      : "=f"(x)
      : "f"(x)	);

  return x;

#else
  int nptr, sft, rep;
  float a, b;
  float man	= frexpf ( x, &nptr );

  sft		= 25 - nptr;
  rep		= ((*((int*)&man) >> sft ) << sft) ;
  a		= *((float*)&rep);
  b		= man-a;

  return 	scalbnf( (2*a+b)*b, nptr*2 );
#endif
}

__host__ __device__ inline double sqMod4( double x)
{
#ifdef  __CUDA_ARCH__

  asm("{" 									// use braces to limit scope
      " .reg .u32 r1;"								// temp reg,
      " .reg .f64 f1, f2;"							// temp reg t1,
      " .reg .u32 hi, lo;"							// temp reg,
      " mov.b64 		{hi, lo}, %1			;"
      " and.b32 		r1,	lo,	2146435072	;"		// Exponent bits
      " shr.b32 		r1,	r1,	20		;"		// Shift to relevant spot
      " sub.s32 		r1,	1076,	r1		;"		// r1 = 53 - ( exp - 1023 ) -  Remove base - shift 53 bits for mantissa length
      " shr.b64 		f2,	%1,	r1		;"		// f2 = x  >> sft
      " shl.b64 		f1,	f2,	r1		;"		// f2 = f2 << sft
      " sub.f64 		f2,	%1,	f1		;"		// b = x - a
      " fma.rn.f64		f1,	2.0,	f1,	f2	;"		// a = 2*a+b
      " mul.f64			%0,	f1,	f2		;"		// a = a*b
      "}"
      : "=d"(x)
      : "d"(x)	);

  return x;
#else
  int nptr, sft;
  double a, b;
  double man	= frexp ( x, &nptr );
  int64_t rep;

  sft		= 54 - nptr;
  rep		= ((*((int64_t*)&man) >> sft ) << sft) ;
  a		= *((double*)&rep);
  b		= man-a;

  return 	scalbn( (2*a+b)*b, nptr*2 );
#endif
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
