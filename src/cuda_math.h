#ifndef CUTIL_MATH_H
#define CUTIL_MATH_H


#ifndef SQRT2
#define SQRT2         1.4142135623730950488016887242096980785696718753769
#endif
#ifndef DBLCORRECT
#define DBLCORRECT    1e-14
#endif
#ifndef PI
#define PI            3.1415926535897932384626433832795028841971693993751
#endif
#ifndef TWOPI
#define TWOPI         6.2831853071795864769252867665590057683943387987502
#endif
#ifndef DEGTORAD
#define DEGTORAD      0.017453292519943295769236907684886127134428718885417
#endif
#ifndef RADTODEG
#define RADTODEG      57.29577951308232087679815481410517033240547246656
#endif
#ifndef PIBYTWO
#define PIBYTWO       1.5707963267948966192313216916397514420985846996876
#endif

#define POWERCU(r,i)  ((r)*(r) + (i)*(i))     /// The sum of the powers of two number
#define POWERC(c)     POWERCU(c.r, c.i)       /// The sum of the powers of a complex number
#define POWERF(f)     POWERCU(f.x, f.y)       /// The sum of the powers of a complex number

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

__host__ __device__ inline float  fmod_t ( float  x, float   y )
{
  return fmodf(x, y);
}
__host__ __device__ inline double fmod_t ( double  x, double y )
{
  return fmod(x, y);
}


#endif