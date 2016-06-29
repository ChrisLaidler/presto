#include <iostream>
#include <stdio.h>

#include "cuda_math.h"
#include "cuda_response.h"
#include "cuda_accel.h"

#define FREESLIM1	1.600781059358212171622054418655453316130105033155		// Sqrt ( 2.5625 )
#define FREESLIM2	36974.0

template<typename T>
__host__ __device__ int z_resp_halfwidth_cu(T z)
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
__host__ __device__ int z_resp_halfwidth_cu_high(T z)
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

/** get the limits of using Fourier Interpolation when using floating point precision  .
 *
 *  Inlining and "templated" function
 *
 * @param nothing This is just a dummy factor for "templating" to floating point
 * @return
 */
__host__ __device__ inline float getFIlim(float nothing)
{
  return FILIM_F;
}

/** get the limits of using Fourier Interpolation when using double point precision  .
 *
 *  Inlining and "templated" function
 *
 * @param nothing This is just a dummy factor for "templating" to floating point
 * @return
 */
__host__ __device__ inline double getFIlim(double nothing)
{
  return FILIM_D;
}


__host__ __device__ inline void sinecos_fres(float x, float x2, float* sin, float* cos)
{
  float trigT = x*x;
  if ( trigT < 1e4 && trigT >-1e4 )
  {
    // Single Precision
    trigT 	= fmod_t(trigT, 4.0f);
  }
  else
  {
    // Double Precision
    trigT 	= fmod_t((double)x*(double)x, 4.0);
  }
  trigT 	= trigT*(float)PIBYTWO;
  sincos_t(trigT, sin, cos);

  //  double sinD, cosD;
  //  sincos_t((double)x*(double)x*(double)PIBYTWO, &sinD, &cosD);
  //  *sin = sinD;
  //  *cos = cosD;
}

__host__ __device__ inline void sinecos_fres(double x, double x2, double* sin, double* cos)
{
  //  double trigT 	= fmod_t(x2, 4.0);
  //  trigT 	= trigT*(double)PIBYTWO;
  //  sincos_t(trigT, sin, cos);

  sincos_t(x2*(double)PIBYTWO, sin, cos);
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
template<typename T, typename idxT>
__host__ __device__ void fresnl(idxT x, T* ss, T* cc)
{
  T f, g, c, s, t, u;
  T 	absX;				// Absolute value of x

  absX       = fabs_t(x);		//

  if      ( absX < (T)FREESLIM1   )	// Small so use a polynomial approximation  .
  {
    T x2	= absX * absX;
    t		= x2 * x2;

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

    *ss   = absX * x2 * sn / sd;
    *cc   = absX * cn / cd;
  }
  else if ( absX > (T)FREESLIM2  )	// Asymptotic behaviour  .
  {
    *cc   = (T)0.5;
    *ss   = (T)0.5;
  }
  else					// Auxiliary functions for large argument  .
  {
    T x2	= absX * absX;		// x * x ( Standard precision value of x squared )

    t		= (T)PI * x2;
    u		= (T)1.0 / (t * t);
    t		= (T)1.0 / t;

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

    f     = (T)1.0 - u * fn / fd;
    g     =          t * gn / gd;

    // Templated for double phase calculations for large x
    sinecos_fres(x, x2, &s, &c);

    t     = (T)PI * absX;

    *cc   = (T)0.5 + (f * s - g * c) / t;
    *ss   = (T)0.5 - (f * c + g * s) / t;

  }

  if ( x < (idxT)0.0 )				// Swap as function is antisymmetric  .
  {
    *cc   = -*cc;
    *ss   = -*ss;
  }
}

__host__ __device__ inline void sinecos_resp(float Qk, float z, float PIoverZ, float* sin, float* cos)
{
  float x_float = Qk * Qk / z ;

  if ( x_float < 1e4 && x_float > -1e4 )
  {
    // single is enough precision
    x_float = fmod_t(x_float, 2.0f);
  }
  else
  {
    // Have to use double
    double x_double	= (double)Qk * (double)Qk / (double)z ;
    x_float	= fmod_t(x_double, 2.0);
  }

  x_float *= (float)PI;
  sincos_t(x_float, sin, cos);
  
  //double sinD, cosD;
  //sincos_t((double)PIoverZ*(double)Qk*(double)Qk, &sinD, &cosD);
  //*sin = sinD;
  //*cos = cosD;
}

__host__ __device__ inline void sinecos_resp(double Qk, double z, double PIoverZ, double* sin, double* cos)
{
  double  xx	= fmod_t(Qk*Qk/z, 2.0);
  xx		*= (double)PI;
  sincos_t(xx, sin, cos);

  //sincos_t(PIoverZ*Qk*Qk, sin, cos);
}

/** Calculate the correlation response value  (z != 0)  .
 *
 * If you want a set of response values at a point or an Fourier interpolation value of FFT data see rz_response_cu and rz_convolution_cu
 *
 * This function calculates the applicable Fourier interpolation response value at a specific distance from a point, at a given z.
 *
 * It uses evaluating Eq (39) in:
 * Ransom, Scott M., Stephen S. Eikenberry, and John Middleditch. "Fourier techniques for very long astrophysical time-series analysis." The Astronomical Journal 124.3 (2002): 1788.
 *
 * Where: qₖ  is the variable Qk and is: ((the distance from the centre frequency) - ṙ/2)
 *        z   is the variable ṙ
 *
 * This function takes as parameters, previously calculated values.
 * These values are the values specific to the value of z only
 * They need only be calculated one per response value and are used as parameters
 * Making this potently efficient
 *
 * This function is templated for precision can can be calculated at single or double precision
 *
 * If called as single precision
 * This function will introduce some double procession math in the basic trigonometric calculations
 * These will calculate phase related information at large (>200) values of Qk (trig calculations still done at single precision)
 * Similar double precision phase calculations are done in the evaluation of the Fresnel integral
 * This increases the range and accuracy of the single precision value at the minimum required amount double calculations
 *
 *
 * Precedence for double:
 * 1:	sq2overAbsZ
 *
 * @param Qk			(rc - k ) - z/2  ( The distance of the response from the reference point, scaled for z)
 * @param z			|z| ( The absolute value of z ) [ The assumption is that z != 0 ]
 * @param sq2overAbsZ		The square root of ( 2 / PI / |z| )
 * @param PIoverZ		PI/z
 * @param overSq2AbsZ		1/sqrt(|z|)
 * @param sighnZ		z/|z| The sign of z (1 or -1)
 * @param real			A pointer to the real part of the response value
 * @param imag			A pointer to the real part of the response value
 */
template<typename T, uint flags>
__host__ __device__ void calc_z_response(T Qk, T z, T sq2overAbsZ, T PIoverZ, T overSq2AbsZ, int sighnZ, T* real, T* imag)
{
  T sin, cos;
  T Yk,Zk;
  T SZk, CZk, SYk, CYk;
  T Sk, Ck;

  // Trig calculations templated for large Qk so phase value is calculated as a double if needed
  sinecos_resp(Qk, z, PIoverZ, &sin, &cos);

  FOLD // Fresnel stuff  .
  {
    Yk = sq2overAbsZ * Qk ;
    if ( Yk > (T)FRES_DOUBLE || Yk < -(T)FRES_DOUBLE )
    {
      // Yk and will be squared so they need to be double if they are large
      double Ykd = (double)sq2overAbsZ *   (double)Qk;
      fresnl<T, double>(Ykd, &SYk, &CYk);
    }
    else
    {
      fresnl<T, T>(Yk, &SYk, &CYk);
    }

    Zk = sq2overAbsZ * ( Qk + z) ;
    if ( Zk > (T)FRES_DOUBLE || Zk < -(T)FRES_DOUBLE )
    {
      // Zk and will be squared so they need to be double if they are large
      double Zkd = (double)sq2overAbsZ * ( (double)Qk + z ) ;
      fresnl<T, double>(Zkd, &SZk, &CZk);
    }
    else
    {
      fresnl<T, T>(Zk, &SZk, &CZk);
    }

    Sk =  ( SZk - SYk );  				// Can be float
    Ck =  ( CYk - CZk ) * sighnZ ;			// Can be float
  }

#if CORRECT_MULT
  // This is the "correct" version
  *real =  overSq2AbsZ * ( Sk * sin - Ck * cos ) ;
  *imag = -overSq2AbsZ * ( Sk * cos + Ck * sin ) ;
#else
  if ( flags )
  {
    // This is the "correct" version
    *real =  overSq2AbsZ * ( Sk * sin - Ck * cos ) ;
    *imag = -overSq2AbsZ * ( Sk * cos + Ck * sin ) ;
  }
  else
  {
    // This is the version accelsearch uses, ( added for comparison )
    *real = overSq2AbsZ * ( Sk * sin - Ck * cos ) ;
    *imag = overSq2AbsZ * ( Sk * cos + Ck * sin ) ;
  }
#endif
}

/** Calculate a single Fourier interpolation response value at a distance from a point  .
 *
 * If you want a set of response values at a point or an Fourier interpolation value of FFT data see rz_response_cu and rz_convolution_cu
 *
 * This function calculates the applicable Fourier interpolation response value at a specific distance from a point.
 *
 * This function takes as parameters, previously calculated values.
 * These are the trigonometric values that are phase related to the reference point in unit steps in r from the point
 * These values are pre-divided by PI
 * and need only be scaled by distance
 *
 * This can be efficiently used to calculate a set of response values for a reference point
 *

 *
 * @param dist		The "distance" in bins of the reference r-value and a "true" location, measured in bins
 * @param sinsinPI	Phase value, sin*sin/PI, signs of trig values irrelevant
 * @param sincosPI	Phase value, sin*cos/PI, signs of trig values irrelevant
 * @param real
 * @param imag
 */
template<typename T>
__host__ __device__ void calc_r_response(T dist, T sinsinPI, T sincosPI, T* real, T* imag)
{
  if ( dist > -getFIlim(dist) && dist < getFIlim(dist) )
  {
    // Correct for division by zero ie: sinc(0) = 1
    *real = (T)1.0;
    *imag = (T)0.0;
  }
  else
  {
#if CORRECT_MULT
    // This is the "correct" version
    *real =  sincosPI / dist ;
    *imag = -sinsinPI / dist ;
#else
    // This is the version accelsearch uses, ( added for comparison )
    *real =  sincosPI / dist ;
    *imag =  sinsinPI / dist ;
#endif
  }
}

/** Calculate a response value at specific bin for a given reference r  .
 *
 * This function calculates the applicable response value at a specific distance from a point.
 * These are used in the correlation to correct FFT values at a given z value and distance in r.
 * If z is close to zero, the Fourier interpolation response is given else the correlation response is returned
 *
 *
 * The distance is, the distance a FFT bin is in from the reference point measured in bin's.
 * where bins with values below the reference point are negative, and points above positive.
 *
 * This function calculates all the "generic" values that are independent of distance for a specific z
 * This is inefficient when requiring all response values for a point
 * In those cases use rz_convolution_cu or rz_response_cu
 *
 * @param bin		The Fourier bin to be multiplied with the response value
 * @param z		The value of fdot, measured in (FFT) bins
 * @param real		Pointer to the real response
 * @param imag		Pointer to the imaginary response
 */
template<typename T>
__host__ __device__ void calc_response_bin(long bin, double r, T z,  T* real, T* imag)
{
  calc_response_off<T>( r-bin, z, real, imag );
}

/** Calculate a response value at a given distance, in r  .
 *
 * If you want a set of response values at a point or an correlation of FFT data see rz_response_cu and rz_convolution_cu
 *
 * This function calculates the applicable response value at a specific distance from a point.
 * These are used in the correlation to correct FFT values at a given z value and distance in r.
 * Where the distance is the distance a FFT bin is in from the reference point measured in bin's.
 * where bins with values below the reference point are negative, and points above positive.
 *
 * This function calculates all the "generic" values that are independent of distance for a specific z
 * This is inefficient when requiring all response values for a point
 * In those cases use rz_convolution_cu or rz_response_cu
 *
 * @param offset	The distance of the (real) r value from the f-fdot position, negative below the location. Measured in (FFT) bins.
 * @param z		The value of fdot, measured in (FFT) bins
 * @param real		Pointer to the real response
 * @param imag		Pointer to the imaginary response
 */
template<typename T>
__host__ __device__ void calc_response_off(T offset, T z, T* real, T* imag)
{
  if ( z < getFIlim(z) && z > -getFIlim(z) )			// Do a Fourier interpolation  .
  {
    double  fracfreq;						// Fractional part of r   - double precision
    double  dintfreq;						// Integer part of r      - double precision

    fracfreq	= modf_t(offset, &dintfreq);			// This is always double precision because - r needs to be r

    // Do Fourier interpolation
    T dist = -offset;
    T sin, cos;

    // This is done at standard precision phase values as its a single value
    sincos_t((T)PI*fracfreq, &sin, &cos);

    calc_r_response(dist, sin*sin/(T)PI, sin*cos/(T)PI, real, imag);
  }
  else
  {
    // Calculate all the constants
    int signZ       = (z < (T)0.0) ? -1 : 1;
    T absZ          = fabs_t(z);
    T sqrtAbsZ      = sqrt_t(absZ);
    T sq2overAbsZ   = (T)SQRT2 / sqrtAbsZ;
    T PIoverZ       = (T)PI / z;
    T overSq2AbsZ   = (T)1.0 / (T)SQRT2 / sqrtAbsZ ;
    T Qk            = (-offset) - z / (T)2.0;			// Adjust for acceleration

    calc_z_response<T,0>(Qk, z, sq2overAbsZ, PIoverZ, overSq2AbsZ, signZ, real, imag);
  }
}

// I found that calling the function above with pointers somtimes gave errors
// So the two functions below pass actual values
__host__ __device__ double2 calc_response_off(double offset, double z)
{
  double2 resp;
  calc_response_off<double>(offset, z, &resp.x, &resp.y);

  return resp;
}

__host__ __device__ float2 calc_response_off(float offset, float z)
{
  float2 resp;
  calc_response_off<float>(offset, z, &resp.x, &resp.y);

  return resp;
}

/** Calculate a set of response values for a give f-fdot value
 *
 * @param r			The desired fractional frequency in bins
 * @param z
 * @param kern_half_width
 * @param out
 */
template<typename T, typename outT>
__host__ __device__ void rz_response_cu(double r, T z, int kern_half_width, outT* out)
{
  outT*   resp;							// The input data, this is a complex number stored as, float2 or double2
  double  fracfreq;						// Fractional part of r   - double precision
  double  dintfreq;						// Integer part of r      - double precision
  long    start = 0;
  T	  offset;						// The distance from the centre frequency (r)
  int     numkern;						// The actual number of kernel values to use

  FOLD // Calculate the reference bin (closes integer bin to r)  .
  {
    fracfreq	= modf_t(r, &dintfreq);				// This is always double precision because - r needs to be r
    start	= dintfreq + 1 - kern_half_width ;

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

  if ( z < getFIlim(z) && z > -getFIlim(z) )			// Do a Fourier interpolation  .
  {
    T dist = offset;

    T sin, cos;
    T sinsinPI, sincosPI;

    // Do all the trig calculations for the constants, can drop PI*dintfreq (signs work out)
    sincos_t((T)PI*fracfreq, &sin, &cos);			// Highest precision using (T)PI*fracfreq
    sinsinPI = sin * sin / (T)PI;
    sincosPI = sin * cos / (T)PI;

    for ( int i = 0 ; i < numkern; i++, dist-- )		// Loop over the kernel elements  .
    {
      //  Get the address of the output value  .
      resp	= &out[start+i];

      // Calculate response
      calc_r_response<T>(dist, sinsinPI,  sincosPI, &resp->x, &resp->y);
    }
  }
  else								// Use a correlation kernel  .
  {
    // Calculate all the constants
    int signZ       = (z < (T)0.0) ? -1 : 1;
    T absZ          = fabs_t(z);
    T sqrtAbsZ      = sqrt_t(absZ);
    T sq2overAbsZ   = (T)SQRT2 / sqrtAbsZ;
    T PIoverZ       = (T)PI / z;
    T overSq2AbsZ   = (T)1.0 / (T)SQRT2 / sqrtAbsZ ;
    T Qk            = offset - z / (T)2.0;			// Adjust for acceleration

    for ( int i = 0 ; i < numkern; i++, Qk-- )			// Loop over the kernel elements
    {
      //  Get the address of the response value  .
      resp	= &out[start+i];

      // Calculate response
      calc_z_response<T,0>(Qk, z, sq2overAbsZ, PIoverZ, overSq2AbsZ, signZ, &resp->x, &resp->y);
    }
  }
}


/** calculate a single point in the f-fdot plain from FFT values
 *
 * This calculation is done by direct application of a convolution
 * It convolves a number of local bins from the FFT with the relevant
 * Response value.
 *
 * It is templated for the precision of the calculation and the input data.
 * Single precision calculations use minimal double point calculation to increase accuracy
 *
 * This is done fairly efficiently, reusing some constants
 *
 * @param inputData
 * @param loR
 * @param noBins
 * @param r
 * @param z
 * @param kern_half_width
 * @param real
 * @param imag
 */
template<typename T, typename dataT>
__host__ __device__ void rz_convolution_cu(dataT* inputData, long loR, long noBins, double r, T z, int kern_half_width, T* real, T* imag)
{
  *real = (T)0.0;
  *imag = (T)0.0;

  dataT   inp;							// The input data, this is a complex number stored as, float2 or double2
  double  fracfreq;						// Fractional part of r   - double precision
  double  dintfreq;						// Integer part of r      - double precision
  long    start;						// The first bin to use
  T       offset;						// The distance from the centre frequency (r) - NOTE: This could be double, float can get ~5 decimal places for lengths of < 999
  int     numkern;						// The actual number of kernel values to use
  T 	  resReal 	= 0;					// Response value - real
  T 	  resImag 	= 0;					// Response value - imaginary

  FOLD 								// Calculate the reference bin (closes integer bin to r)  .
  {
    fracfreq	= modf_t(r, &dintfreq);				// This is always double precision because - r needs to be r
    start	= dintfreq + 1 - kern_half_width ;

    if ( fracfreq > 0.5 ) // Adjust to closest bin
    {
      fracfreq -= 1.0 ;
      dintfreq++;
    }
  }

  FOLD 								// Clamp values to usable bounds  .
  {
    numkern 		= 2 * kern_half_width;

    if ( start < 0 )
    {
      numkern += start;						// Decrease number of kernel values
      start    = 0;
    }

    offset = (r - start);					// This is rc-k for the first bin
  }

  FOLD 								// Adjust for FFT
  {
    // Adjust to FFT
    if ( start >= loR )
    {
      start -= loR;						// Adjust for accessing the input FFT
    }
    else
    {
      // Start is below beginning of available data so start at available data
      numkern -= loR - start;
      offset = ( r - loR);					// This is rc-k for the first bin
      start = 0;
    }

    if ( start + numkern >= noBins )
    {
      numkern = noBins - start;
    }
  }

  if ( z < getFIlim(resReal) && z > -getFIlim(resReal) )	// Do a Fourier interpolation  .
  {
    T dist = offset;
    T sin, cos;
    T sinsinPI, sincosPI;

    // Do all the trig calculations for the constants, can drop PI*dintfreq (signs work out)
    sincos_t((T)PI*fracfreq, &sin, &cos);			// Highest precision using (T)PI*fracfreq
    sinsinPI = sin * sin / (T)PI;
    sincosPI = sin * cos / (T)PI;

    for ( int i = 0 ; i < numkern; i++, dist-- )		// Loop over the kernel elements  .
    {
      FOLD //  Read the input value  .
      {
	inp	= inputData[start+i];
      }

      FOLD 							// Calculate response  .
      {
	calc_r_response<T>(dist, sinsinPI,  sincosPI, &resReal, &resImag);
      }

      FOLD 							//  Do the multiplication and sum  accumulate  .
      {
#if CORRECT_MULT
	// This is the "correct" version
	*real += (resReal * inp.x - resImag * inp.y);
	*imag += (resReal * inp.y + resImag * inp.x);
#else
	// This is the version accelsearch uses, ( added for comparison )
	*real += (resReal * inp.x + resImag * inp.y);
	*imag += (resImag * inp.x - resReal * inp.y);
#endif
      }
    }
  }
  else								// Use a correlation kernel  .
  {
    // Calculate all the constants
    int signZ       = (z < (T)0.0) ? -1 : 1;
    T absZ          = fabs_t(z);
    T sqrtAbsZ      = sqrt_t(absZ);
    T sq2overAbsZ   = (T)SQRT2 / sqrtAbsZ;
    T PIoverZ       = (T)PI / z;
    T overSq2AbsZ   = (T)1.0 / (T)SQRT2 / sqrtAbsZ ;
    T Qk            = offset - z / (T)2.0;			// Adjust for acceleration

    for ( int i = 0 ; i < numkern; i++ , Qk-- )			// Loop over the kernel elements
    {

      FOLD 							//  Read the input value  .
      {
	inp	= inputData[start+i];
      }

      FOLD 							// Calculate response  .
      {
	calc_z_response<T,0>(Qk, z, sq2overAbsZ, PIoverZ, overSq2AbsZ, signZ, &resReal, &resImag);
      }

      FOLD 							//  Do the multiplication and sum  accumulate  .
      {
#if CORRECT_MULT
	// This is the "correct" version
	*real += (resReal * inp.x - resImag * inp.y);
	*imag += (resReal * inp.y + resImag * inp.x);
#else
	// This is the version accelsearch uses, ( added for comparison )
	*real += (resReal * inp.x + resImag * inp.y);
	*imag += (resImag * inp.x - resReal * inp.y);
#endif
      }
    }
  }
}


template<typename T, typename dataT>
__host__ __device__ void rz_single_mult_cu(dataT* inputData, long loR, long noBins, double r, T z, int kern_half_width, T* real, T* imag, int i)
{
  *real = (T)0.0;
  *imag = (T)0.0;

  dataT   inp;                                                  // The input data, this is a complex number stored as, float2 or double2
  double  fracfreq;                                             // Fractional part of r   - double precision
  double  dintfreq;                                             // Integer part of r      - double precision
  long    start;                                                // The first bin to use
  T       offset;                                               // The distance from the centre frequency (r) - NOTE: This could be double, float can get ~5 decimal places for lengths of < 999
  T       resReal       = 0;                                    // Response value - real
  T       resImag       = 0;                                    // Response value - imaginary

  FOLD                                                          // Calculate the reference bin (closes integer bin to r)  .
  {
    fracfreq    = modf_t(r, &dintfreq);                         // This is always double precision because - r needs to be r
    start       = dintfreq + 1 - kern_half_width + i ;

    if ( fracfreq > 0.5 ) // Adjust to closest bin
    {
      fracfreq -= 1.0 ;
      dintfreq++;
    }
  }

  FOLD                                                          // Clamp values to usable bounds  .
  {
    offset = ( r - start );                                  // This is rc-k for the first bin
  }

  FOLD                                                          // Adjust for FFT
  {
    // Adjust to FFT
    {
      start -= loR;                                             // Adjust for accessing the input FFT
    }
    if ( start < 0 )
      return;
    if ( start >= noBins )
      return;
  }

  if ( z < getFIlim(resReal) && z > -getFIlim(resReal) )        // Do a Fourier interpolation  .
  {
    T dist = offset;

    T sin, cos;
    T sinsinPI, sincosPI;

    // Do all the trig calculations for the constants, can drop PI*dintfreq (signs work out)
    sincos_t((T)PI*fracfreq, &sin, &cos);                       // Highest precision using (T)PI*fracfreq
    sinsinPI = sin * sin / (T)PI;
    sincosPI = sin * cos / (T)PI;


    FOLD //  Read the input value  .
    {
      inp     = inputData[start];
    }

    FOLD                                                      // Calculate response  .
    {
      calc_r_response<T>(dist, sinsinPI,  sincosPI, &resReal, &resImag);
    }

    FOLD                                                      //  Do the multiplication and sum  accumulate  .
    {
#if CORRECT_MULT
      // This is the "correct" version
      *real += (resReal * inp.x - resImag * inp.y);
      *imag += (resReal * inp.y + resImag * inp.x);
#else
      // This is the version accelsearch uses, ( added for comparison )
      *real += (resReal * inp.x + resImag * inp.y);
      *imag += (resImag * inp.x - resReal * inp.y);
#endif
    }
  }
  else                                                          // Use a correlation kernel  .
  {
    // Calculate all the constants
    int signZ       = (z < (T)0.0) ? -1 : 1;
    T absZ          = fabs_t(z);
    T sqrtAbsZ      = sqrt_t(absZ);
    T sq2overAbsZ   = (T)SQRT2 / sqrtAbsZ;
    T PIoverZ       = (T)PI / z;
    T overSq2AbsZ   = (T)1.0 / (T)SQRT2 / sqrtAbsZ ;
    T Qk            = offset - z / (T)2.0;                      // Adjust for acceleration

    FOLD                                                      //  Read the input value  .
    {
      inp     = inputData[start];
    }

    FOLD                                                      // Calculate response  .
    {
      calc_z_response<T,0>(Qk, z, sq2overAbsZ, PIoverZ, overSq2AbsZ, signZ, &resReal, &resImag);
    }

    FOLD                                                      //  Do the multiplication and sum  accumulate  .
    {
#if CORRECT_MULT
      // This is the "correct" version
      *real += (resReal * inp.x - resImag * inp.y);
      *imag += (resReal * inp.y + resImag * inp.x);
#else
      // This is the version accelsearch uses, ( added for comparison )
      *real += (resReal * inp.x + resImag * inp.y);
      *imag += (resImag * inp.x - resReal * inp.y);
#endif
    }
  }
}

/**  Uses the correlation method to do a Fourier interpolation at a number integer spaced (r) points in the f-fdot plane.
 *
 * It does the correlations manually. (i.e. no FFTs)
 * The kernels can be reused for the same value of z and fraction of r
 * Thus each thread calculates each kernel value once and uses it to calculate the value of
 * a number of integer spaced points in the r direction
 *
 * @param inputData           A pointer to the beginning of the input data
 * @param outData             A pointer to the location of the output complex numbers, this is a thread dependent array of length noBlk
 * @param loR                 The R value of the first bin in the input data
 * @param r                   The R value of the first point to do the interpolation at
 * @param z                   The Z value of the to do the interpolation at
 * @param blkWidth            The width of the blocks in bins
 * @param kern_half_width     The half width of the points to use in the interpolation
 */
template<typename T, typename dataIn, typename dataOut, int noBlk>
__host__ __device__ void rz_convolution_cu(dataIn* inputData, long loR, long inStride, double r, T z, int kern_half_width, dataOut* outData, int blkWidth)
{
  for ( int blk = 0; blk < noBlk; blk++ )
  {
    outData[blk].x = 0;
    outData[blk].y = 0;
  }

  dataIn  inp;							// The input data, this is a complex number stored as, float2 or double2
  double  fracfreq;						// Fractional part of r   - double precision
  double  dintfreq;						// Integer part of r      - double precision
  long    start;						// The first bin to use
  T       offset;						// The distance from the centre frequency (r) - NOTE: This could be double, float can get ~5 decimal places for lengths of < 999
  int     numkern;						// The actual number of kernel values to use
  T 	  resReal 	= (T)0.0;				// Response value - real
  T 	  resImag 	= (T)0.0;				// Response value - imaginary

  FOLD 								// Calculate the reference bin (closes integer bin to r)  .
  {
    fracfreq	= modf_t(r, &dintfreq);				// This is always double precision because - r needs to be r
    start	= dintfreq + 1 - kern_half_width ;		// TODO check this +1????

    if ( fracfreq > 0.5 ) // Adjust to closest bin
    {
      fracfreq -= 1.0 ;
      dintfreq++;
    }
  }

  FOLD 								// Clamp values to usable bounds  .
  {
    numkern 		= 2 * kern_half_width;

    offset = ( r - start);					// This is rc-k for the first bin
  }

  FOLD 								// Adjust for FFT
  {
    // Adjust to FFT
    start -= loR;						// Adjust for accessing the input FFT
  }

  if ( z < getFIlim(resReal) && z > -getFIlim(resReal) )	// Do a Fourier interpolation  .
  {
    T dist = offset;

    T sin, cos;
    T sinsinPI, sincosPI;

    // Do all the trig calculations for the constants, can drop PI*dintfreq (signs work out)
    sincos_t((T)PI*fracfreq, &sin, &cos);			// Highest precision using (T)PI*fracfreq
    sinsinPI = sin * sin / (T)PI;
    sincosPI = sin * cos / (T)PI;

    for ( int i = 0 ; i < numkern; i++, dist-- )		// Loop over the kernel elements  .
    {
      FOLD 							// Calculate response  .
      {
	calc_r_response<T>(dist, sinsinPI,  sincosPI, &resReal, &resImag);
      }

      // Use the kernel value on each input value with the same fractional part
      for ( int blk = 0; blk < noBlk; blk++ )
      {
	FOLD // Clamp values to usable bounds  .
	{
	  int idx = start+i+blk*blkWidth;

	  if ( idx >= 0 && idx < inStride )
	  {
	    FOLD //  Read the input value  .
	    {
	      inp             = inputData[idx];
	    }

	    FOLD //  Do the multiplication  .
	    {
#if CORRECT_MULT
	      // This is the "correct" version
	      outData[blk].x += (resReal * inp.x - resImag * inp.y);
	      outData[blk].y += (resReal * inp.y + resImag * inp.x);
#else
	      // This is the version accelsearch uses, ( added for comparison )
	      outData[blk].x += (resReal * inp.x + resImag * inp.y);
	      outData[blk].y += (resImag * inp.x - resReal * inp.y);
#endif
	    }
	  }
	}
      }
    }
  }
  else								// Use a correlation kernel  .
  {
    // Calculate all the constants
    int signZ       = (z < (T)0.0) ? -1 : 1;
    T absZ          = fabs_t(z);
    T sqrtAbsZ      = sqrt_t(absZ);
    T sq2overAbsZ   = (T)SQRT2 / sqrtAbsZ;
    T PIoverZ       = (T)PI / z;
    T overSq2AbsZ   = (T)1.0 / (T)SQRT2 / sqrtAbsZ ;
    T Qk            = offset - z / (T)2.0;			// Adjust for acceleration

    for ( int i = 0 ; i < numkern; i++, Qk-- )			// Loop over the kernel elements
    {

      FOLD 							// Calculate response  .
      {
	calc_z_response<T,0>(Qk, z, sq2overAbsZ, PIoverZ, overSq2AbsZ, signZ, &resReal, &resImag);
      }

      // Use the kernel value on each input value with the same fractional part
      for ( int blk = 0; blk < noBlk; blk++ )
      {
	FOLD // Clamp values to usable bounds  .
	{
	  int idx = start+i+blk*blkWidth;

	  if ( idx >= 0 && idx < inStride )
	  {
	    FOLD //  Read the input value  .
	    {
	      inp             = inputData[idx];
	    }

	    FOLD //  Do the multiplication  .
	    {
#if CORRECT_MULT
	      // This is the "correct" version
	      outData[blk].x += (resReal * inp.x - resImag * inp.y);
	      outData[blk].y += (resReal * inp.y + resImag * inp.x);
#else
	      // This is the version accelsearch uses, ( added for comparison )
	      outData[blk].x += (resReal * inp.x + resImag * inp.y);
	      outData[blk].y += (resImag * inp.x - resReal * inp.y);
#endif
	    }
	  }
	}
      }
    }
  }
}

template<typename T, typename dataT>
__host__ __device__ void rz_convolution_cu_inc(dataT* inputData, long loR, long noBins, double r, T z, int kern_half_width, T* real, T* imag)
{
  *real = (T)0.0;
  *imag = (T)0.0;

  dataT   inp;							// The input data, this is a complex number stored as, float2 or double2
  double  fracfreq;						// Fractional part of r   - double precision
  double  dintfreq;						// Integer part of r      - double precision

  T 	  resReal 	= (T)0.0;				// Response value - real
  T 	  resImag 	= (T)0.0;				// Response value - imaginary

  T 	  respRealSum 	= (T)0.0;				//
  T 	  respImagSum 	= (T)0.0;				//

  T 	  inRealSum 	= (T)0.0;				//
  T 	  inImagSum 	= (T)0.0;				//

  T 	  mathRealSum 	= (T)0.0;				//
  T 	  mathImagSum 	= (T)0.0;				//

  T 	  accelRealSum 	= (T)0.0;				//
  T 	  accelImagSum 	= (T)0.0;				//

  FOLD // Calculate the reference bin (closes integer bin to r)  .
  {
    fracfreq	= modf_t(r, &dintfreq);				// This is always double precision because - r needs to be r

    if ( fracfreq > 0.5 )					// Adjust to closest bin  .
    {
      fracfreq -= 1.0 ;
      dintfreq++;
    }
  }

  if ( z < getFIlim(z) && z > -getFIlim(z) )			// Do a Fourier interpolation  .
  {
    T dist;
    T sin, cos;
    T sinsinPI, sincosPI;

    // Do all the trig calculations for the constants
    sincos_t((T)PI*fracfreq, &sin, &cos);
    sinsinPI = sin * sin / (T)PI;
    sincosPI = sin * cos / (T)PI;


    for ( int i = 0 ; i <= kern_half_width; i++ )		// Loop over the kernel elements  .
    {
      for ( int sn = -1; sn < 2; sn += 2 )
      {
	long k = dintfreq - i*sn;

	dist = r - k;

	FOLD //  Read the input value  .
	{
	  inp	= inputData[ k - loR ];

	  inRealSum += inp.x;
	  inImagSum += inp.y;
	}

	FOLD // Calculate response  .
	{
	  calc_r_response<T>(dist, sinsinPI,  sincosPI, &resReal, &resImag);

	  accelRealSum += resReal * inp.x - resImag*inp.y;
	  accelImagSum += resReal * inp.y + resImag*inp.x;

	  respRealSum += resReal;
	  respImagSum += resImag;
	}

	FOLD //  Do the multiplication and sum  accumulate  .
	{
	  *real	+= resReal * inp.x - resImag*inp.y;
	  *imag	+= resReal * inp.y + resImag*inp.x;
	}

      }
    }
  }
  else								// Use a correlation kernel  .
  {
    // Calculate all the constants
    int signZ       = (z < (T)0.0) ? -1 : 1;
    T absZ          = fabs_t(z);
    T sqrtAbsZ      = sqrt_t(absZ);
    T sq2overAbsZ   = (T)SQRT2 / sqrtAbsZ;
    T PIoverZ       = (T)PI / z;
    T overSq2AbsZ   = (T)1.0 / (T)SQRT2 / sqrtAbsZ ;

    //for ( int i = 0 ; i <= kern_half_width; i++ )		// Loop over the kernel elements
    for ( int i = 0 ; i <= kern_half_width*2; i++ )		// Loop over the kernel elements
    {
      //for ( int sn = -1; sn < 2; sn += 2 )
      {
	//long k = dintfreq - i*sn;
	long k = dintfreq + 1 - kern_half_width + i;

	T Qk            = (r-k) - z / (T)2.0;			// Adjust for acceleration

	FOLD //  Read the input value  .
	{
	  inp	= inputData[ k - loR ];

	  inRealSum += inp.x;
	  inImagSum += inp.y;
	}

	FOLD // Calculate response  .
	{
	  calc_z_response<T,0>(Qk, z, sq2overAbsZ, PIoverZ, overSq2AbsZ, signZ, &resReal, &resImag);

	  accelRealSum += resReal * inp.x - resImag * inp.y;
	  accelImagSum += resReal * inp.y + resImag * inp.x;

	  respRealSum += resReal;
	  respImagSum += resImag;

	  calc_z_response<T,0>(Qk, z, sq2overAbsZ, PIoverZ, overSq2AbsZ, signZ, &resReal, &resImag);

	  mathRealSum += resReal * inp.x - resImag * inp.y;
	  mathImagSum += resReal * inp.y + resImag * inp.x;

	}

	FOLD //  Do the multiplication and sum  accumulate  .
	{
	  *real	+= resReal * inp.x - resImag*inp.y;
	  *imag	+= resReal * inp.y + resImag*inp.x;
	}
      }
    }
  }
}



template int z_resp_halfwidth_cu<float >(float  z);
template int z_resp_halfwidth_cu<double>(double z);

template int z_resp_halfwidth_cu_high<float >(float  z);
template int z_resp_halfwidth_cu_high<double>(double z);

template void fresnl<float,  float>  (float  xxa, float*  ss, float*  cc);
template void fresnl<float,  double> (double xxa, float*  ss, float*  cc);
template void fresnl<double, double> (double xxa, double* ss, double* cc);

template void calc_z_response<float,  0>(float  Qk, float  z, float  sq2overAbsZ, float  PIoverZ, float  overSq2AbsZ, int sighnZ, float*  real, float*  imag);
template void calc_z_response<double, 0>(double Qk, double z, double sq2overAbsZ, double PIoverZ, double overSq2AbsZ, int sighnZ, double* real, double* imag);
template void calc_z_response<float,  1>(float  Qk, float  z, float  sq2overAbsZ, float  PIoverZ, float  overSq2AbsZ, int sighnZ, float*  real, float*  imag);
template void calc_z_response<double, 1>(double Qk, double z, double sq2overAbsZ, double PIoverZ, double overSq2AbsZ, int sighnZ, double* real, double* imag);

template void calc_r_response<float >(float  dist, float  sinsinPI, float  sincosPI, float*  real, float*  imag);
template void calc_r_response<double>(double dist, double sinsinPI, double sincosPI, double* real, double* imag);

template void calc_response_bin<float> (long bin, double r, float  z,  float*  real, float*  imag);
template void calc_response_bin<double>(long bin, double r, double z,  double* real, double* imag);

template void calc_response_off<float> (float  offset, float  z,  float*  real, float*  imag);
template void calc_response_off<double>(double offset, double z,  double* real, double* imag);

template void rz_convolution_cu<float,  float2> (float2*  inputData, long loR, long noBins, double r, float  z, int kern_half_width, float*  real, float*  imag);
template void rz_convolution_cu<float,  double2>(double2* inputData, long loR, long noBins, double r, float  z, int kern_half_width, float*  real, float*  imag);
template void rz_convolution_cu<double, float2> (float2*  inputData, long loR, long noBins, double r, double z, int kern_half_width, double* real, double* imag);
template void rz_convolution_cu<double, double2>(double2* inputData, long loR, long noBins, double r, double z, int kern_half_width, double* real, double* imag);


template void rz_single_mult_cu<float,  float2> (float2*  inputData, long loR, long noBins, double r, float  z, int kern_half_width, float*  real, float*  imag, int i);
template void rz_single_mult_cu<float,  double2>(double2* inputData, long loR, long noBins, double r, float  z, int kern_half_width, float*  real, float*  imag, int i);
template void rz_single_mult_cu<double, float2> (float2*  inputData, long loR, long noBins, double r, double z, int kern_half_width, double* real, double* imag, int i);
template void rz_single_mult_cu<double, double2>(double2* inputData, long loR, long noBins, double r, double z, int kern_half_width, double* real, double* imag, int i);

//template void rz_convolution_cu_inc<float,  float2> (float2*  inputData, long loR, long noBins, double r, float  z, int kern_half_width, float*  real, float*  imag);
//template void rz_convolution_cu_inc<float,  double2>(double2* inputData, long loR, long noBins, double r, float  z, int kern_half_width, float*  real, float*  imag);
//template void rz_convolution_cu_inc<double, float2> (float2*  inputData, long loR, long noBins, double r, double z, int kern_half_width, double* real, double* imag);
//template void rz_convolution_cu_inc<double, double2>(double2* inputData, long loR, long noBins, double r, double z, int kern_half_width, double* real, double* imag);

template void rz_response_cu<double, double2>(double r, double z, int kern_half_width, double2* out);
template void rz_response_cu<float,  float2> (double r, float  z, int kern_half_width, float2*  out);

template void rz_convolution_cu<float,  float2, float2, 2> (float2* inputData, long loR, long inStride, double r, float  z, int kern_half_width, float2* outData, int blkWidth);
template void rz_convolution_cu<float,  float2, float2, 3> (float2* inputData, long loR, long inStride, double r, float  z, int kern_half_width, float2* outData, int blkWidth);
template void rz_convolution_cu<float,  float2, float2, 4> (float2* inputData, long loR, long inStride, double r, float  z, int kern_half_width, float2* outData, int blkWidth);
template void rz_convolution_cu<float,  float2, float2, 5> (float2* inputData, long loR, long inStride, double r, float  z, int kern_half_width, float2* outData, int blkWidth);
template void rz_convolution_cu<float,  float2, float2, 6> (float2* inputData, long loR, long inStride, double r, float  z, int kern_half_width, float2* outData, int blkWidth);
template void rz_convolution_cu<float,  float2, float2, 7> (float2* inputData, long loR, long inStride, double r, float  z, int kern_half_width, float2* outData, int blkWidth);
template void rz_convolution_cu<float,  float2, float2, 8> (float2* inputData, long loR, long inStride, double r, float  z, int kern_half_width, float2* outData, int blkWidth);
template void rz_convolution_cu<float,  float2, float2, 9> (float2* inputData, long loR, long inStride, double r, float  z, int kern_half_width, float2* outData, int blkWidth);
template void rz_convolution_cu<float,  float2, float2, 10>(float2* inputData, long loR, long inStride, double r, float  z, int kern_half_width, float2* outData, int blkWidth);
//template void rz_convolution_cu<float,  float2, float2, 11>(float2* inputData, long loR, long inStride, double r, float  z, int kern_half_width, float2* outData, int blkWidth);
//template void rz_convolution_cu<float,  float2, float2, 12>(float2* inputData, long loR, long inStride, double r, float  z, int kern_half_width, float2* outData, int blkWidth);
//template void rz_convolution_cu<float,  float2, float2, 13>(float2* inputData, long loR, long inStride, double r, float  z, int kern_half_width, float2* outData, int blkWidth);
//template void rz_convolution_cu<float,  float2, float2, 14>(float2* inputData, long loR, long inStride, double r, float  z, int kern_half_width, float2* outData, int blkWidth);
//template void rz_convolution_cu<float,  float2, float2, 15>(float2* inputData, long loR, long inStride, double r, float  z, int kern_half_width, float2* outData, int blkWidth);


template void rz_convolution_cu<double, float2, float2, 2> (float2* inputData, long loR, long inStride, double r, double z, int kern_half_width, float2* outData, int blkWidth);
template void rz_convolution_cu<double, float2, float2, 3> (float2* inputData, long loR, long inStride, double r, double z, int kern_half_width, float2* outData, int blkWidth);
template void rz_convolution_cu<double, float2, float2, 4> (float2* inputData, long loR, long inStride, double r, double z, int kern_half_width, float2* outData, int blkWidth);
template void rz_convolution_cu<double, float2, float2, 5> (float2* inputData, long loR, long inStride, double r, double z, int kern_half_width, float2* outData, int blkWidth);
template void rz_convolution_cu<double, float2, float2, 6> (float2* inputData, long loR, long inStride, double r, double z, int kern_half_width, float2* outData, int blkWidth);
template void rz_convolution_cu<double, float2, float2, 7> (float2* inputData, long loR, long inStride, double r, double z, int kern_half_width, float2* outData, int blkWidth);
template void rz_convolution_cu<double, float2, float2, 8> (float2* inputData, long loR, long inStride, double r, double z, int kern_half_width, float2* outData, int blkWidth);
template void rz_convolution_cu<double, float2, float2, 9> (float2* inputData, long loR, long inStride, double r, double z, int kern_half_width, float2* outData, int blkWidth);
template void rz_convolution_cu<double, float2, float2, 10>(float2* inputData, long loR, long inStride, double r, double z, int kern_half_width, float2* outData, int blkWidth);
//template void rz_convolution_cu<double, float2, float2, 11>(float2* inputData, long loR, long inStride, double r, double z, int kern_half_width, float2* outData, int blkWidth);
//template void rz_convolution_cu<double, float2, float2, 12>(float2* inputData, long loR, long inStride, double r, double z, int kern_half_width, float2* outData, int blkWidth);
//template void rz_convolution_cu<double, float2, float2, 13>(float2* inputData, long loR, long inStride, double r, double z, int kern_half_width, float2* outData, int blkWidth);
//template void rz_convolution_cu<double, float2, float2, 14>(float2* inputData, long loR, long inStride, double r, double z, int kern_half_width, float2* outData, int blkWidth);
//template void rz_convolution_cu<double, float2, float2, 15>(float2* inputData, long loR, long inStride, double r, double z, int kern_half_width, float2* outData, int blkWidth);
