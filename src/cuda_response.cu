#include <iostream>
#include <stdio.h>

#include "cuda_math.h"
#include "cuda_response.h"


__host__ __device__ inline float getFIlim(float nothing)
{
  return FILIM_F;
}

__host__ __device__ inline double getFIlim(double nothing)
{
  return FILIM_D;
}

/** Fresnel integral  .
 *
 * DESCRIPTION:
 *
 * Evaluates the Fresnel integrals
 *
 *           x
 *           -
 *          | |
 * C(x) =   |   cos(pi/2 t**2) dt,
 *        | |
 *         -
 *          0
 *
 *           x
 *           -
 *          | |
 * S(x) =   |   sin(pi/2 t**2) dt.
 *        | |
 *         -
 *          0
 *
 *
 * The integrals are evaluated by a power series for x < 1.
 * For x >= 1 auxiliary functions f(x) and g(x) are employed
 * such that
 *
 * C(x) = 0.5 + f(x) sin( pi/2 x**2 ) - g(x) cos( pi/2 x**2 )
 * S(x) = 0.5 - f(x) cos( pi/2 x**2 ) - g(x) sin( pi/2 x**2 )
 *
 *
 *
 * ACCURACY:
 *
 *  Relative error.
 *
 * Arithmetic  function   domain     # trials      peak         rms
 *   IEEE       S(x)      0, 10       10000       2.0e-15     3.2e-16
 *   IEEE       C(x)      0, 10       10000       1.8e-15     3.3e-16
 *   DEC        S(x)      0, 10        6000       2.2e-16     3.9e-17
 *   DEC        C(x)      0, 10        5000       2.3e-16     3.9e-17
 *
 *   This function is adapted from:
 *   Cephes Math Library Release 2.8:  June, 2000
 *   Copyright 1984, 1987, 1989, 2000 by Stephen L. Moshier
 *
 * @param xxa Value to evaluate the Fresnel integral at
 * @param ss  The result S(xxa)
 * @param cc  The result C(xxa)
 */
template<typename T>
__host__ __device__ void fresnl(T xxa, T* ss, T* cc)
{
  T f, g, c, s, t, u;
  T x, x2;

  x       = fabs_t(xxa);
  x2      = x * x;

  if      ( x2 < 2.5625   )     // Small so use a polynomial approximation  .
  {
    t     = x2 * x2;

    T t01 = t;
    T t02 = t01*t;
    T t03 = t02*t;
    T t04 = t03*t;
    T t05 = t04*t;
    T t06 = t05*t;
    T sn  = (T)3.18016297876567817986e11 + (T)-4.42979518059697779103e10*t01 + (T)2.54890880573376359104e9*t02  + (T)-6.29741486205862506537e7*t03  + (T)7.08840045257738576863e5 *t04 - (T)2.99181919401019853726e3  *t05;
    T sd  = (T)6.07366389490084639049e11 + (T) 2.24411795645340920940e10*t01 + (T)4.19320245898111231129e8*t02  + (T) 5.17343888770096400730e6*t03  + (T)4.55847810806532581675e4 *t04 + (T)2.81376268889994315696e2  *t05 + t06 ;
    T cn  = (T)9.99999999999999998822e-1 + (T)-2.05525900955013891793e-1*t01 + (T)1.88843319396703850064e-2*t02 + (T)-6.45191435683965050962e-4*t03 + (T)9.50428062829859605134e-6*t04 - (T)4.98843114573573548651e-8 *t05;
    T cd  = (T)1.00000000000000000118e0  + (T) 4.12142090722199792936e-2*t01 + (T)8.68029542941784300606e-4*t02 + (T) 1.22262789024179030997e-5*t03 + (T)1.25001862479598821474e-7*t04 + (T)9.15439215774657478799e-10*t05 + (T)3.99982968972495980367e-12*t06 ;

    *ss   = x * x2 * sn / sd;
    *cc   = x * cn / cd;
  }
  else if ( x  > (T)36974.0  )  // Asymptotic behaviour  .
  {
    *cc   = 0.5;
    *ss   = 0.5;
  }
  else                          // Auxiliary functions for large argument  .
  {
    x2    = x * x;
    t     = (T)PI * x2;
    u     = 1.0 / (t * t);
    t     = 1.0 / t;

    T u01 = u;
    T u02 = u01*u;
    T u03 = u02*u;
    T u04 = u03*u;
    T u05 = u04*u;
    T u06 = u05*u;
    T u07 = u06*u;
    T u08 = u07*u;
    T u09 = u08*u;
    T u10 = u09*u;
    T u11 = u10*u;
    T fn  = (T)3.76329711269987889006e-20 + (T)1.34283276233062758925e-16*u01 + (T)1.72010743268161828879e-13*u02 + (T)1.02304514164907233465e-10*u03 + (T)3.05568983790257605827e-8 *u04 + (T)4.63613749287867322088e-6*u05 + (T)3.45017939782574027900e-4*u06 + (T)1.15220955073585758835e-2*u07 + (T)1.43407919780758885261e-1*u08 + (T)4.21543555043677546506e-1*u09;
    T fd  = (T)1.25443237090011264384e-20 + (T)4.52001434074129701496e-17*u01 + (T)5.88754533621578410010e-14*u02 + (T)3.60140029589371370404e-11*u03 + (T)1.12699224763999035261e-8 *u04 + (T)1.84627567348930545870e-6*u05 + (T)1.55934409164153020873e-4*u06 + (T)6.44051526508858611005e-3*u07 + (T)1.16888925859191382142e-1*u08 + (T)7.51586398353378947175e-1*u09 + u10;
    T gn  = (T)1.86958710162783235106e-22 + (T)8.36354435630677421531e-19*u01 + (T)1.37555460633261799868e-15*u02 + (T)1.08268041139020870318e-12*u03 + (T)4.45344415861750144738e-10*u04 + (T)9.82852443688422223854e-8*u05 + (T)1.15138826111884280931e-5*u06 + (T)6.84079380915393090172e-4*u07 + (T)1.87648584092575249293e-2*u08 + (T)1.97102833525523411709e-1*u09 + (T)5.04442073643383265887e-1*u10 ;
    T gd  = (T)1.86958710162783236342e-22 + (T)8.39158816283118707363e-19*u01 + (T)1.38796531259578871258e-15*u02 + (T)1.10273215066240270757e-12*u03 + (T)4.60680728146520428211e-10*u04 + (T)1.04314589657571990585e-7*u05 + (T)1.27545075667729118702e-5*u06 + (T)8.14679107184306179049e-4*u07 + (T)2.53603741420338795122e-2*u08 + (T)3.37748989120019970451e-1*u09 + (T)1.47495759925128324529e0 *u10 + u11 ;

    f     = 1.0 - u * fn / fd;
    g     =       t * gn / gd;

    t     = (T)PIBYTWO * x2;
    sincos_t(t, &s, &c);
    t     = (T)PI * x;

    *cc   = (T)0.5 + (f * s - g * c) / t;
    *ss   = (T)0.5 - (f * c + g * s) / t;
  }

  if ( xxa < 0.0 )              // Swap as function is antisymmetric  .
  {
    *cc   = -*cc;
    *ss   = -*ss;
  }
}

template<typename T, uint flags>
__host__ __device__ void calc_z_response(T Qk, T z, T sq2overAbsZ, T PIoverZ, T overSq2AbsZ, int sighnZ, T* real, T* imag)
{
  /* This is evaluating Eq (39) in:
   * Ransom, Scott M., Stephen S. Eikenberry, and John Middleditch. "Fourier techniques for very long astrophysical time-series analysis." The Astronomical Journal 124.3 (2002): 1788.
   *
   * Where: qₖ  is the variable Qk and is: (the distance from the centre frequency) - ṙ/2
   *        ṙ   is the variable z
   *
   *        The rest of the variables are values that do not change with k
   */

  T SZk, CZk, SYk, CYk;
  T sin, cos;

  T xx = PIoverZ * Qk * Qk ;
  T Yk = sq2overAbsZ * Qk ;
  T Zk = sq2overAbsZ * ( Qk + z) ;

  sincos_t(xx, &sin, &cos);

  fresnl<T>(Yk, &SYk, &CYk);
  fresnl<T>(Zk, &SZk, &CZk);

  T Sk =  SZk - SYk ;
  T Ck =  CYk - CZk ;

  Ck *= sighnZ;

  if ( flags )
  {
    // This is the version I get by doing the math
    *real = overSq2AbsZ * (Sk*cos + Ck*sin) ;
    *imag = overSq2AbsZ * (Sk*sin - Ck*cos) ;
  }
  else
  {
    // This is the version in accelsearch
    *real = overSq2AbsZ * (Sk*sin - Ck*cos) ;
    *imag = overSq2AbsZ * (Sk*cos + Ck*sin) ;
  }
}

template<typename T>
__host__ __device__ void calc_r_response(T dist, T sinsin, T sincos, T* real, T* imag)
{
  /* This is evaluating Eq (30) in:
   * Ransom, Scott M., Stephen S. Eikenberry, and John Middleditch. "Fourier techniques for very long astrophysical time-series analysis." The Astronomical Journal 124.3 (2002): 1788.
   *
   */

  if ( dist > -SINCLIM && dist < SINCLIM )
  {
    // Correct for division by zero ie: sinc(0) = 1
    *real = (T)1.0;
    *imag = (T)0.0;
  }
  else
  {
    *real = sincos / dist ;
    *imag = sinsin / dist ;
  }
}

template<typename T>
__host__ __device__ void calc_response_bin(long bin, double r, float z,  T* real, T* imag)
{
  calc_response_off<T>( r-bin, z, real, imag );
}

template<typename T>
__host__ __device__ void calc_response_off(float offset, float z,  T* real, T* imag)
{
  if ( z < FILIM_F && z > -FILIM_F )
  {
    // Do Fourier interpolation
    T dist = (T)PI*(-offset);
    T sin, cos;

    sincos_t(dist, &sin, &cos);

    calc_r_response(dist, sin*sin, sin*cos, real, imag);
  }
  else
  {
    int signZ       = (z < 0.0) ? -1 : 1;
    T Qk            = (-offset) - z / (T)2.0;			// Adjust for acceleration
    T absZ          = fabs_t(z);
    T sqrtAbsZ      = sqrt_t(absZ);
    T sq2overAbsZ   = SQRT2 / sqrtAbsZ;
    T PIoverZ       = PI / z;
    T overSq2AbsZ   = 1.0 / SQRT2 / sqrtAbsZ ;

    calc_z_response<T,0>(Qk, z, sq2overAbsZ, PIoverZ, overSq2AbsZ, signZ, real, imag);
  }
}

template<typename T, typename dataT>
__host__ __device__ void rz_interp_cu(dataT* inputData, long loR, long noBins, double r, float z, int kern_half_width, T* real, T* imag)
{
  *real = (T)0.0;
  *imag = (T)0.0;

  dataT   inp;							// The input data, this is a complex number stored as, float2 or double2
  double  fracfreq;						// Fractional part of r   - double precision
  double  dintfreq;						// Integer part of r      - double precision
  long    start;						// The first bin to use
  float   offset;						// The distance from the centre frequency (r)
  int     numkern;						// The actual number of kernel values to use
  T 	  resReal 	= 0;					// Response value - real
  T 	  resImag 	= 0;					// Response value - imaginary

  FOLD // Calculate the reference bin (closes integer bin to r)  .
  {
    fracfreq	= modf_t(r, &dintfreq);				// This is always double precision because - r needs to be r
    start	= dintfreq + 1 - kern_half_width ;

    if ( fracfreq > 0.5 ) // Adjust to closest bin
    {
      fracfreq -= 1.0 ;
      dintfreq++;
    }

    offset = ( r - start);					// This is rc-k for the first bin
  }

  FOLD // Clamp values to usable bounds  .
  {
    numkern 		= 2 * kern_half_width;

    if ( start < 0 )
    {
      numkern += start;						// Decrease number of kernel values
      offset  -= start;						// offset and start are negative so subtract
      start    = 0;
    }

    start -= loR;

    if ( start + numkern >= noBins )
    {
      numkern = noBins - start;
    }
  }

  if ( z < FILIM_F && z > -FILIM_F )				// Do a Fourier interpolation
  {
    T dist = (T)PI*offset;
    T sin, cos;
    T sinsin, sincos;

    // Do all the trig calculations for the constants
    sincos_t(dist, &sin, &cos);
    sinsin = sin * sin;
    sincos = sin * cos;

    for ( int i = 0 ; i < numkern; i++, dist-=(T)PI )		// Loop over the kernel elements  .
    {
      FOLD //  Read the input value  .
      {
        inp	= inputData[start+i];
      }

      FOLD // Calculate response  .
      {
	calc_r_response(dist, sinsin,  sincos, &resReal, &resImag);
      }

      FOLD //  Do the multiplication and sum  accumulate  .
      {
	*real	+= resReal * inp.x - resImag*inp.y;
	*imag	+= resReal * inp.y + resImag*inp.x;
      }
    }
  }
  else								// Use a correlation kernel  .
  {
    // Calculate all the constants
    int signZ       = (z < 0.0) ? -1 : 1;
    T absZ          = fabs_t(z);
    T sqrtAbsZ      = sqrt_t(absZ);
    T sq2overAbsZ   = SQRT2 / sqrtAbsZ;
    T PIoverZ       = PI / z;
    T overSq2AbsZ   = 1.0 / SQRT2 / sqrtAbsZ ;
    T Qk            = offset - z / (T)2.0;			// Just for acceleration

    for ( int i = 0 ; i < numkern; i++, Qk-- )			// Loop over the kernel elements
    {
      FOLD //  Read the input value  .
      {
        inp	= inputData[start+i];
      }

      FOLD // Calculate response  .
      {
	calc_z_response<T,0>(Qk, z, sq2overAbsZ, PIoverZ, overSq2AbsZ, signZ, real, imag);
      }

      FOLD //  Do the multiplication and sum  accumulate  .
      {
	*real	+= resReal * inp.x - resImag*inp.y;
	*imag	+= resReal * inp.y + resImag*inp.x;
      }
    }
  }
}


template<typename T, typename outT>
__host__ __device__ void gen_response_cu(double r, float z, int kern_half_width, outT* out)
{
  outT*   resp;							// The input data, this is a complex number stored as, float2 or double2
  double  fracfreq;						// Fractional part of r   - double precision
  double  dintfreq;						// Integer part of r      - double precision
  long    start = 0;
  float   offset;						// The distance from the centre frequency (r)
  int     numkern;						// The actual number of kernel values to use

  FOLD // Calculate the reference bin (closes integer bin to r)  .
  {
    fracfreq	= modf_t(r, &dintfreq);				// This is always double precision because - r needs to be r
    start	= dintfreq - kern_half_width ;

    if ( fracfreq > 0.5 )					// Adjust to closest bin
    {
      fracfreq -= 1.0 ;
      dintfreq++;
    }

    offset = ( r - start );					// This is rc-k for the first bin
  }

  FOLD // Clamp values to usable bounds  .
  {
    numkern 		= 2 * kern_half_width;
  }

  if ( z < FILIM_F && z > -FILIM_F )				// Do a Fourier interpolation  .
  {
    T dist = (T)PI*offset;
    T sin, cos;
    T sinsin, sincos;

    // Do all the trig calculations for the constants
    sincos_t(dist, &sin, &cos);
    sinsin = sin * sin;
    sincos = sin * cos;

    for ( int i = 0 ; i < numkern; i++, dist-=(T)PI )		// Loop over the kernel elements  .
    {
      //  Get the address of the output value  .
      resp	= &out[start+i];

      // Calculate response
      calc_r_response<T>(dist, sinsin,  sincos, &resp->x, &resp->y);

      //T sighn = pow(-1,i);
      //printf("%04i response: %19.16f %19.16f  r: %15.10f  c: %15.10f s: %15.10f sinc: %15.10f\n", i, resp->x, resp->y, dist, sighn*sin, sighn*cos, sighn*sin/dist );
    }
  }
  else								// Use a correlation kernel  .
  {
    // Calculate all the constants
    int signZ       = (z < 0.0) ? -1 : 1;
    T absZ          = fabs_t(z);
    T sqrtAbsZ      = sqrt_t(absZ);
    T sq2overAbsZ   = SQRT2 / sqrtAbsZ;
    T PIoverZ       = PI / z;
    T overSq2AbsZ   = 1.0 / SQRT2 / sqrtAbsZ ;
    T Qk            = offset - z / (T)2.0;			// Just for acceleration

    for ( int i = 0 ; i < numkern; i++, Qk-- )			// Loop over the kernel elements
    {
      //  Get the address of the output value  .
      resp	= &out[start+i];

      // Calculate response
      calc_z_response<T,0>(Qk, z, sq2overAbsZ, PIoverZ, overSq2AbsZ, signZ, &resp->x, &resp->y);
    }
  }
}


template void fresnl<float> (float  xxa, float*  ss, float*  cc);
template void fresnl<double>(double xxa, double* ss, double* cc);

template void calc_z_response<float,  0>(float  Qk, float  z, float  sq2overAbsZ, float  PIoverZ, float  overSq2AbsZ, int sighnZ, float*  real, float*  imag);
template void calc_z_response<double, 0>(double Qk, double z, double sq2overAbsZ, double PIoverZ, double overSq2AbsZ, int sighnZ, double* real, double* imag);
template void calc_z_response<float,  1>(float  Qk, float  z, float  sq2overAbsZ, float  PIoverZ, float  overSq2AbsZ, int sighnZ, float*  real, float*  imag);
template void calc_z_response<double, 1>(double Qk, double z, double sq2overAbsZ, double PIoverZ, double overSq2AbsZ, int sighnZ, double* real, double* imag);

template void calc_response_bin<float> (long bin, double r, float z,  float*  real, float*  imag);
template void calc_response_bin<double>(long bin, double r, float z,  double* real, double* imag);

template void calc_response_off<float> (float offset, float z,  float*  real, float*  imag);
template void calc_response_off<double>(float offset, float z,  double* real, double* imag);

template void rz_interp_cu<float,  float2> (float2*  inputData, long loR, long noBins, double r, float z, int kern_half_width, float*  real, float*  imag);
template void rz_interp_cu<float,  double2>(double2* inputData, long loR, long noBins, double r, float z, int kern_half_width, float*  real, float*  imag);
template void rz_interp_cu<double, float2> (float2*  inputData, long loR, long noBins, double r, float z, int kern_half_width, double* real, double* imag);
template void rz_interp_cu<double, double2>(double2* inputData, long loR, long noBins, double r, float z, int kern_half_width, double* real, double* imag);

template void gen_response_cu<double, double2>(double r, float z, int kern_half_width, double2* out);
template void gen_response_cu<float,  float2> (double r, float z, int kern_half_width, float2*  out);
