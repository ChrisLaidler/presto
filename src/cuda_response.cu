/** @file cuda_response.cu
 *  @brief Utility functions and kernels to calculate response filter coefficients and perfrom correlations
 *
 *  @author Chris Laidler
 *  @bug No known bugs.
 *
 *  This includes fresnel integrals
 *
 *  Change Log
 *
 *  2017-10-20
 *    Start the change log - I know this is a bit late =/
 *    Had a big refactor of all the functions here
 *    Full work up of error and optimisation of speed - See my thesis for full details
 *    In brief:
 *       Max error in "generic" single coefficients of ~1e-5 from offset > 0.04 - Note the error can be > 2e-5 for offset > 1000
 *       Max error in "generic" double coefficients of ~1e-12 from 0.0002 < offset < 0.04 - Error then drops to 2e-15 at offset = 10 then increases to 3e-14 at offset = 1000
 *
 *
 */

#include <iostream>
#include <stdio.h>

#include "cuda_math.h"
#include "cuda_response.h"


/** Point beyond which to return 0.5, the asymptotic value  .
 *
 *  At some point the Fresnel amplitude is smaller than the error
 *
 * @param nothing   This is just a dummy factor for "templating" to floating point
 * @return          The float specific boundary
 */
__host__ __device__ inline float fresLim2(float nothing)
{
  // From testing this value is ~5e6
  return FRESLIM2_F;
}

/** Point beyond which to return 0.5, the asymptotic value  .
 *
 *  At some point the Fresnel amplitude is smaller than the error
 *
 * @param nothing   This is just a dummy factor for "templating" to floating point
 * @return          The double specific boundary
 */
__host__ __device__ inline double fresLim2(double nothing)
{
  // From testing this value is ~5e9
  return FRESLIM2_D;
}

/** Get the limit below which, Fourier Interpolation, returns the actual bin  .
 *
 *  Inlined and templated function, so that it quickly returns a float specific constant value
 *
 * @param nothing   This is just a dummy factor for "templating" to floating point
 * @return          The float specific boundary
 */
__host__ __device__ inline float getRlim(float nothing)
{
  return R_LIM_F;
}

/** Get the limit below which, Fourier Interpolation, returns the actual bin  .
 *
 *  Inlined and templated function, so that it quickly returns a double specific constant value
 *
 * @param nothing   This is just a dummy factor for "templating" to floating point
 * @return          The double specific boundary
 */
__host__ __device__ inline double getRlim(double nothing)
{
  return R_LIM_D;
}

/** Get the limit below which, to do a Fourier Interpolation rather than calculate a the acceleration coefficient  .
 *
 *  Inlined and templated function, so that it quickly returns a float specific constant value
 *
 *  The bound I found by inspection is: 1e-5 + r*r/3e4 + |r|*1e3
 *  This is a quadratic value below which I have found the error in the float coefficient to be greater than float interpolation value
 *
 * @param r         The offset of the bin from the point
 * @return          The quadratic float specific boundary
 */
__host__ __device__ /*inline*/ float getZlim(float r)
{
  return 0.028f + 0.0325f * pow_t(fabs_t(r), 1.25f );
}

/** Get the limit below which, to do a Fourier Interpolation rather than calculate a the acceleration coefficient  .
 *
 *  Inlined and templated function, so that it quickly returns a float specific constant value
 *
 *  The bound I found by inspection is: 1e-5 + r*r/3e4 + |r|*1e3
 *  This is a quadratic value below which I have found the error in the float coefficient to be greater than float interpolation value
 *
 * @param r         The offset of the bin from the point
 * @return          The quadratic float specific boundary
 */
__host__ __device__ /*inline*/ double getZlim(double r)
{
  return 0.00015f + 0.0002256f * pow_t(fabs_t((float)r), 1.25f );
}

/** Get the limit below which, to do a Fourier Interpolation rather than calculate a the acceleration coefficient  .
 *
 *  Inlined and templated function, so that it quickly returns a float specific constant value
 *
 *  The bound I found by inspection is: 1e-5 + r*r/3e4 + |r|*1e3
 *  This is a quadratic value below which I have found the error in the float coefficient to be greater than float interpolation value
 *
 * @param r         The offset of the bin from the point
 * @return          The quadratic float specific boundary
 */
__host__ long double getZlim(long double r)
{
  return 0.0002L + 0.00024L * pow_t(fabs_t(r), 1.5L) ;
}

__host__ __device__ inline void fres_sinecos_phase(float x, float x2, float* sin, float* cos)
{
  float trigT;

  if      ( x2 <= FRES_SINGLE_PHASE  )				// Single Precision - no phase calcs - 1024 - 2048 - 4096
  {
    // This is the simples and most common case.
    // this accounts for the vast majority of the the coefficients between the high accurate bounds
    // In these cases     the trig calculation (assuming float has 6 significant digits)
    trigT 	= x2;
  }
  else								// Single Precision - phase adjust - 2^13 8192 - 2^14 16384 - 2^15 32768
  {
    // My modulus of the square
    trigT	= sqMod4(x);
  }

//  TESTING:							// Double Precision - phase adjust
//  {
//    // This general happens when z is close to zero, a good approximation to the boundary condition (x2 == 1e4) is:
//    // |z| < 2e-4 x (offset)^2
//    // This double precision float modulus can be very computationally expensive and it is thus worth avoiding if possible
//    trigT 	= fmod_t((double)x*(double)x, 4.0);
//  }

  // The actual trigonometric calculation - Using intrinsic function faster but less accurate.
  trigT 	= trigT*(float)PIBYTWO;
  sincos_t(trigT, sin, cos);

  // TESTING: Below is slower than the above trig, as there is no intrinsic single precision sincospif and improves performance only slightly
  //trigT 	= trigT/2.0f;
  //sincospi_t(trigT, sin, cos);
}

__host__ __device__ inline void fres_sinecos_phase(double x, double x2, double* sin, double* cos)
{
  double trigT;

  if      ( x2 <= FRES_SINGLE_PHASE  )				// Double Precision - no phase calcs
  {
    // Strait double implementation
    trigT 	= x2;
  }
  else								// Phase calculation on the x^2 term
  {
    // Do a DP phase correction
    trigT 	= sqMod4(x);
  }

  // Use sincospi it is faster and more accurate, only faster because there is no intrinsic __sincos for double precision
  sincospi_t(trigT/2.0, sin, cos);

  // TESTING: Most basic double
  //sincos_t(x2*(double)PIBYTWO, sin, cos);
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
 * @param x   Value to evaluate the Fresnel integral at
 * @param ss  The result S(x)
 * @param cc  The result C(x)
 */
template<typename T, typename idxT>
__host__ __device__ void fresnl(idxT x, T* cc, T* ss)
{
  T f, g, c, s, t, u;
  T absX;					// Absolute value of x
  absX       = fabs_t(x);			// Use templated absolute CUDA function

  /**
   *  In our case x = sqrt(2/|z|) * (-offset - z / 2.0 )
   *
   *  TODO: This function could be templated for accuracy 1-9 determining the number of elements of the polynomials that are used -
   */
  if      ( absX < (T)FREESLIM1  )		// Small so use a polynomial approximation  .
  {
    /*
       This method only gets used about 5 % of the time
       From actual run-tests I found that this cases can run in as little as ~55 clock cycles in single precision and ~1655 clock cycles in double precision on Maxwell Generation

       Op-Count
        * 31
        / 2
        + 21
    */
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
  else if ( absX > fresLim2((T)0.0)  )		// Asymptotic behaviour  .
  {
    // From testing I found the point where the error is greater than the Fresnel amplitude, at that point return the asymptotic value
    *cc   = (T)0.5;
    *ss   = (T)0.5;
  }
  else						// Auxiliary functions for large argument  .
  {
    /*
       This method gets used more than 95 % of the time and is computationally more intensive, with ~115 basic flops as well as trig and fabs

       From actual run-tests I found that this cases can run in as little as ~212-482 clock cycles in single precision and ~3960 clock cycles in double precision on Maxwell Generation
       If the trig calculation is done strait up using the __sincosf intrinsic and no phase correction is done this can run in as little as ~157 clock cycles - this is not suggested as this negatively effects accuracy.
       If single precision phase correction is done, this can run in ~212, cock cycles, if the phase correction is done in double precision ~482 clock cycles are used, almost doubling the run time! (900+ clock cycles if trig is done in double precision)

       Op-Count
        * 59 + 1
        / 6
        + 43
        - 3
        fmod
        sincos
    */

    T x2	= absX * absX;			// x * x ( Standard precision value of x squared )

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

    // This trig calculation is templated for T precision phase calculations for large x
    // If T precision phase calculations are used this can almost T the run time of this function!
    // Even in the single precision case this accounts for ~0.25 of computation time
    fres_sinecos_phase((T)x, x2, &s, &c);

    // TESTING: Double for comparison of accuracy
    //double ts, tc;
    //sincospi_t((double)x*(double)x/2.0, &ts, &tc);
    //s = ts; c = tc;

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

////////////////////  Coefficient - Fourier interpolation

/** Calculate a single Fourier interpolation coefficient at a distance from a point  .
 *
 * If you want a set of coefficients at a point or an Fourier interpolation value of FFT data see rz_coefficients and rz_convolution_cu
 *
 * This function calculates the applicable Fourier interpolation coefficient at a specific distance from a point.
 *
 * This function takes as parameters, previously calculated values.
 * These are the trigonometric values that are phase related to the reference point in unit steps in r from the point
 * These values are pre-divided by PI
 * and need only be scaled by distance
 *
 * This can be efficiently used to calculate a set of coefficients for a reference point
 *
 *
 * @param dist		The "distance" in bins of the reference r-value and a "true" location, measured in bins
 * @param sinsinPI	Phase value, sin*sin/PI, signs of trig values irrelevant
 * @param sincosPI	Phase value, sin*cos/PI, signs of trig values irrelevant
 * @param real		A pointer to the real part of the Fourier value
 * @param imag		A pointer to the real part of the Fourier value
 */
template<typename T>
__host__ __device__ inline void calc_coefficient_r(T offset, T sinsinbyPI, T sincosbyPI, T* real, T* imag)
{
  if ( fabs_t(offset) < getRlim(sinsinbyPI) )			// Check for close to actual bin values - fabs_t 1 op
  {
    // Correct for division by zero ie: sinc(0) = 1
    *real = (T)1.0;
    *imag = (T)0.0;
  }
  else
  {
    *real =  sincosbyPI / offset ;
    *imag = -sinsinbyPI / offset ;
  }
}

/** Calculate Fourier interpolation value at a given distance, in r  .
 *
 * If you want a set of coefficients at a point or an correlation of FFT data see rz_coefficients and rz_convolution_cu
 *
 * This function calculates the applicable Fourier interpolation components at a specific distance from a point.
 * Where the distance is the distance a FFT bin is in from the reference point measured in bin's.
 * where bins with values below the reference point are positive, and points above negative.
 *
 * This function calculates all the "generic" values that are independent of distance for a specific z
 * This is inefficient when requiring all coefficients for a point
 * In those cases use rz_convolution_cu or rz_coefficients
 *
 * @param offset	The distance of the (real) r value from the f-fdot position, negative below the location. Measured in (FFT) bins.
 * @param real		Pointer to the real coefficient
 * @param imag		Pointer to the imaginary coefficient
 */
template<typename T>
__host__ __device__ inline void calc_coefficient_r(T offset, T* resReal, T* resImag)
{
  // Do Fourier interpolation
  T sin, cos;

  /** NOTE: Single precision accuracy
   *   I tested using  sincospif(x)  vs  __sincosf(x*PI)
   *   I found a significant change in accuracy
   *   __sincosf(x*PI):
   *   This faster intrinsic has an error that tails off at around 1e-7,
   *   this is a decreasing error relative to the amplitude of the of values
   *   sincospif():
   *   Has an error that drops off at the same rate as the amplitude,
   *   thus has a roughly constant error of approximately 6 decimal places.
   */
  sincospi_t(offset, &sin, &cos);		// Slightly slower but constant error relative to amplitude
  //sincos_t(offset*(T)PI, &sin, &cos);		// Slightly faster but relative accuracy drops with offset

  calc_coefficient_r(offset, sin*sin/(T)PI, sin*cos/(T)PI, resReal, resImag);
}

////////////////////  Coefficient - Acceleration

__host__ __device__ inline void resp_sinecos_phase(float Qk, float dr, float z, float* sin, float* cos)
{
  //float x_float = fabs_t(z);

  float x_float = dr * dr / z - dr + z/4.0f ;
  if      ( fabs_t(x_float) < RESP_SINGLE_PHASE )
  {
    // Here the final trig term is still be below some power of two term, leaving desired precision for trig
    // Pre-calculated the value because its needed for the check ;)
  }
  else
  {
    // Calculate phase in double precision
    // This should not happen, double precision math that will probably give results worse than a R search
    // This only happens if z is very small, ie very close to zero acceleration, or if the offset is lager than the high accuracy kernel length, which is a bit long at zero actually.
    double x_double	= (double)Qk * (double)Qk / (double)z ;
    x_float		= fmod_t(x_double, 2.0);
  }

  // This could have been done but then have to change the bound multiples crazy factors
  x_float *= (float)PI;

  // The actual trigonometric calculation
  // TODO: Test accuracy using sincospi_t vs sincos_t(PI*)
  sincos_t(x_float, sin, cos);

  // NOTE: Below is slower than the above tig, as there is no intrinsic single precision sincospif func, true sincospif accuracy was not noticed in my application
  //sincospi_t(x_float, sin, cos);

  // NOTE: Tested double trig - Does not improve accuracy
  //double dSin, dCos;
  //sincospi_t((double)Qk * (double)Qk / (double)z, &dSin, &dCos );
  //*sin = dSin;
  //*cos = dCos;
}

__host__ __device__ inline void resp_sinecos_phase(double Qk, double dr, double z, double* sin, double* cos)
{
  double  xx	= (dr * dr / z - dr + z/4.0);

  // Double precision phase clipping (unnecessary as it is trig is done in double)
  //xx		= fmod_t(xx, 2.0);
  
  //xx		*= (double)PI;
  //sincos_t(xx, sin, cos);

  // Use sincospi it is faster and more accurate, only faster because there is no intrinsic __sincos for double precision
  sincospi_t(xx, sin, cos);
}

__host__ __device__ inline void resp_sinecos(float Qk, float dr, float z, float* sin, float* cos)
{
  //sincos_t(Qk * Qk / z * (float)PI, sin, cos);			// High error in places
  //sincos_t((dr*dr-z*dr+z*z/4.0) / z * (float)PI, sin, cos);
  sincos_t((dr * dr / z - dr + z/4.0f )* (float)PI, sin, cos);		// Reduce max error but with slightly higher general error
}

__host__ __device__ inline void resp_sinecos(double Qk, double dr, double z, double* sin, double* cos)
{
  //sincospi_t(Qk*Qk/z, sin, cos);				// High error in places
  sincospi_t((dr * dr / z - dr + z/4.0), sin, cos);		// Reduce max error but with slightly higher general error
}

/** Calculate the correlation coefficient  (z != 0)  .
 *
 * If you want a set of coefficients at a point or an Fourier interpolation value of FFT data see rz_coefficients and rz_convolution_cu
 *
 * This function calculates the applicable Fourier interpolation coefficient at a specific distance from a point, at a given z.
 *
 * It uses evaluating Eq (39) in:
 * Ransom, Scott M., Stephen S. Eikenberry, and John Middleditch. "Fourier techniques for very long astrophysical time-series analysis." The Astronomical Journal 124.3 (2002): 1788.
 *
 * Where: qₖ  is the variable Qk and is: ((the distance from the centre frequency) - ṙ/2)
 *        z   is the variable ṙ
 *
 * This function takes as parameters, previously calculated values.
 * These values are the values specific to the value of z only
 * They need only be calculated one per coefficient and are used as parameters
 * Making this more efficient
 *
 * This function is templated for precision can can be calculated at single or double precision
 *
 * If called as single precision
 * This function may introduce some double procession math to calculate some phase information
 * Similar double precision phase calculations are done in the evaluation of the Fresnel integral
 * This increases the range and accuracy of the single precision value at the minimum required amount double calculations
 *
 *
 * @param Qk			(rc - k ) - z/2  ( The distance of the coefficient from the reference point, scaled for z)
 * @param z			|z| ( The absolute value of z ) [ The assumption is that z != 0 ]
 * @param sq2overAbsZ		The square root of ( 2 / PI / |z| )
 * @param overSq2AbsZ		1/sqrt(|z|)
 * @param sighnZ		z/|z| The sign of z (1 or -1)
 * @param real			A pointer to the real part of the coefficient
 * @param imag			A pointer to the real part of the coefficient
 */
template<typename T, bool phaseCheck>
__host__ __device__ inline void calc_coefficient_z(T Qk, T dr, T z, T sq2overAbsZ, T overSq2AbsZ, int sighnZ, T* real, T* imag)
{
  T sin, cos;
  T Yk,Zk;
  T SZk, CZk, SYk, CYk;
  T Sk, Ck;

  if ( phaseCheck )						// This check should be evaluated at compile time
  {
    // Trig calculations templated for large Qk so phase value is calculated as a double if needed
    // Double will generally happens at very low z an approximation when |z| < 1.02e-4 x (offset)^2
    resp_sinecos_phase(Qk, dr, z, &sin, &cos);
  }
  else
  {
    resp_sinecos(Qk, dr, z, &sin, &cos);
  }

  FOLD // Fresnel calculations  .
  {
    Yk = sq2overAbsZ * Qk;
    fresnl<T, T>(Yk, &CYk, &SYk);

    Zk = sq2overAbsZ * ( Qk + z) ;
    fresnl<T, T>(Zk, &CZk, &SZk);

    Sk =  ( SZk - SYk );
    Ck =  ( CYk - CZk ) * sighnZ ;
  }

  // This is the "correct" version
  *real =  overSq2AbsZ * ( Sk * sin - Ck * cos ) ;
  *imag = -overSq2AbsZ * ( Sk * cos + Ck * sin ) ;

//  // TESTING: Double
//  *real =  rsqrt(2.0*fabs((double)z)) * ( (double)Sk * (double)sin - (double)Ck * (double)cos ) ;
//  *imag = -rsqrt(2.0*fabs((double)z)) * ( (double)Sk * (double)cos + (double)Ck * (double)sin ) ;
}

template<typename T, bool phaseCheck>
__host__ __device__ inline void calc_coefficient_z(T offset, T z, T* resReal, T* resImag)
{
  // Calculate all the "constants"
  // 6 basic fops and fabs and sqrt
  int signZ		= (z < (T)0.0) ? -1 : 1;
  T absZ		= fabs_t(z);
  T sqrtAbsZ		= sqrt_t(absZ);
  T sq2overAbsZ		= (T)SQRT2 / sqrtAbsZ;
  T overSq2AbsZ		= (T)1.0 / (T)SQRT2 / sqrtAbsZ ;
  T Qk			= offset - z / (T)2.0;			// Adjust for acceleration

  calc_coefficient_z<T, phaseCheck>(Qk, offset, z, sq2overAbsZ, overSq2AbsZ, signZ, resReal, resImag);
}

////////////////////  Coefficient - Approximation

/** Calculate Fourier interpolation value at a given distance, in r  .
 *
 * If you want a set of coefficients at a point or an correlation of FFT data see rz_coefficients and rz_convolution_cu
 *
 * This function calculates the applicable Fourier interpolation components at a specific distance from a point.
 * Where the distance is the distance a FFT bin is in from the reference point measured in bin's.
 * where bins with values below the reference point are positive, and points above negative.
 *
 * This function calculates all the "generic" values that are independent of distance for a specific z
 * This is inefficient when requiring all coefficients for a point
 * In those cases use rz_convolution_cu or rz_coefficients
 *
 * @param offset	The distance of the (real) r value from the f-fdot position, negative below the location. Measured in (FFT) bins.
 * @param real		Pointer to the real coefficient
 * @param imag		Pointer to the imaginary coefficient
 */
template<typename T>
__host__ __device__ inline void calc_coefficient_a(T offset, T z, T piR, T sinPiR, T cosPiR, T* resReal, T* resImag)
{
  T r1_abs = fabs_t(offset);
  T r1 = offset;
  T r2 = r1 * r1 ;
  T r3 = r2 * r1 ;
  T term;

  // T0 (constant) ie: Fourier interpolation
  T a0_r	= +cosPiR*sinPiR/piR;
  T a0_i	= -sinPiR*sinPiR/piR;
  if ( r1_abs < getE0lim(r1) )								// Check for close to actual bin values
  {
    a0_r = (T)1.0;
    a0_i = (T)0.0;
  }
  
  // T1 (linear) coefficient
  term		= (cosPiR-sinPiR/piR)/r2/(T)PI/(T)2.0 ;
  T a1_r	= -sinPiR * term;
  T a1_i	= -cosPiR * term;
  if ( r1_abs < getE1rLim(r1) )
  {
    a1_r	= (T)1.64493406512755329404 * r1 ;					// Liner interpolate crossing at 0
  }
  if ( r1_abs < getE1iLim(r1) )
  {
    a1_i	= (T)0.523598775598298873067 - (T)3.10062372500642122663 * r2 ;		// Quadratic interpolate intercept at Pi/6
  }

  // T2 (quadratic) coefficient
  term		= (T)0.25/(T)PI/r3*((T)3.0/(T)PI/r1*(-sinPiR/piR+cosPiR)+sinPiR);
  T a2_r	= +cosPiR*term;
  T a2_i	= -sinPiR*term;
  if ( r1_abs < getE2lim(r1) )
  {
    a2_r	= -(T)0.164493406684822643659 + (T)0.927685388563495792822 * r2 ;	// Quadratic interpolate intercept at Pi^2/60
    a2_i	=  (T)0.516751848464762894412 * r1 ;					// Liner interpolate crossing at 0
  }

  // T3 (cubic) coefficient
  // I had a look at this, I got close, but not quite there - future work?
  //T a3_r	= 0 ; //
  //T a3_i	= 0 ; // +cosPiR/(T)8.0/PI/r/r/r/r*((T)6.0/piR*((T)2.499983/piR*(-sinPiR/piR+cosPiR)+sinPiR)-cosPiR - sinPiR*sinPiR/cosPiR/(T)125.0*r/fabs_t(r) ) ;

  *resReal	= a0_r + a1_r*z + a2_r*z*z  ; //+ a3_r*z*z*z ;
  *resImag	= a0_i + a1_i*z + a2_i*z*z  ; //+ a3_i*z*z*z ;
}

/** Calculate Fourier interpolation value at a given distance, in r  .
 *
 * If you want a set of coefficients at a point or an correlation of FFT data see rz_coefficients and rz_convolution_cu
 *
 * This function calculates the applicable Fourier interpolation components at a specific distance from a point.
 * Where the distance is the distance a FFT bin is in from the reference point measured in bin's.
 * where bins with values below the reference point are positive, and points above negative.
 *
 * This function calculates all the "generic" values that are independent of distance for a specific z
 * This is inefficient when requiring all coefficients for a point
 * In those cases use rz_convolution_cu or rz_coefficients
 *
 * @param offset	The distance of the (real) r value from the f-fdot position, negative below the location. Measured in (FFT) bins.
 * @param real		Pointer to the real coefficient
 * @param imag		Pointer to the imaginary coefficient
 */
template<typename T>
__host__ __device__ inline void calc_coefficient_a(T offset, T z, T* resReal, T* resImag)
{
  T piR, sinPiR, cosPiR;

  // Basic terms
  piR = (T)PI*offset;
  sincospi_t(offset, &sinPiR, &cosPiR);		// Slightly slower but constant error relative to amplitude
  //sincos_t(piR, &sinPiR, &cosPiR);		// Slightly faster but relative accuracy drops with offset

  return calc_coefficient_a<T>(offset, z, piR, sinPiR, cosPiR, resReal, resImag);
}

////////////////////  Coefficient - Generic

/** Calculate a coefficient at a given distance, in r  .
 *
 * If you want a set of coefficients at a point or an correlation of FFT data see rz_coefficients and rz_convolution_cu
 *
 * This function calculates the applicable coefficient at a specific distance from a point.
 * These are used in the correlation to correct FFT values at a given z value and distance in r.
 * Where the distance is the distance a FFT bin is in from the reference point measured in bin's.
 * where bins with values below the reference point are positive, and points above negative.
 *
 * This function calculates all the "generic" values that are independent of distance for a specific z
 * This is inefficient when requiring all coefficients for a point
 * In those cases use rz_convolution_cu or rz_coefficients
 *
 * @param offset	The distance of the (real) r value from the f-fdot position, negative below the location. Measured in (FFT) bins.
 * @param z		The value of fdot, measured in (FFT) bins
 * @param real		Pointer to the real coefficient
 * @param imag		Pointer to the imaginary coefficient
 */
template<typename T>
__host__ __device__ void calc_coefficient(T offset, T z, T* resReal, T* resImag)
{
  T abs_z = fabs_t(z);

  if ( abs_z > getZlim(offset) )				// Calculate raw coefficient .
  {
    calc_coefficient_z<T, false>(offset, z, resReal, resImag);
  }
  else								// Calculate approximation coefficient  .
  {
    calc_coefficient_a<T>(offset, z, resReal, resImag);
  }
}

__host__ __device__ double2 calc_coefficient(double offset, double z)
{
  double2 resp;
  calc_coefficient<double>(offset, z, &resp.x, &resp.y);

  return resp;
}

__host__ __device__ float2  calc_coefficient(float  offset, float  z)
{
  float2 resp;
  calc_coefficient<float>(offset, z, &resp.x, &resp.y);

  return resp;
}

/** Calculate a coefficient at specific bin for a given reference r  .
 *
 * This function calculates the applicable coefficient at a specific distance from a point.
 * These are used in the correlation to correct FFT values at a given z value and distance in r.
 * If z is close to zero, the Fourier interpolation coefficient is given else the correlation coefficient is returned
 *
 *
 * The distance is, the distance a FFT bin is in from the reference point measured in bin's.
 * where bins with values below the reference point are negative, and points above positive.
 *
 * This function calculates all the "generic" values that are independent of distance for a specific z
 * This is inefficient when requiring all coefficients for a point
 * In those cases use rz_convolution_cu or rz_coefficients
 *
 * @param bin		The Fourier bin to be multiplied with the coefficient
 * @param z		The value of fdot, measured in (FFT) bins
 * @param real		Pointer to the real coefficient
 * @param imag		Pointer to the imaginary coefficient
 */
template<typename T>
__host__ __device__ void calc_coefficient_bin(long bin, double r, T z,  T* real, T* imag)
{
  calc_coefficient<T>( r-bin, z, real, imag );
}

////////////////////  Generate an array of coefficients

/** Calculate a set of coefficients for a give f-fdot value
 *
 * @param r			The desired fractional frequency in bins
 * @param z
 * @param kern_half_width
 * @param out
 */
template<typename T, typename outT>
__host__ __device__ void rz_coefficients(double r, T z, int kern_half_width, outT* out)
{
  outT*   resp;							// The input data, this is a complex number stored as, float2 or double2
  long    dintfreq;						// Integer part of r      - double precision
  long    start = 0;
  T	  offset;						// The distance from the centre frequency (r)
  int     numkern;						// The actual number of kernel values to use

  FOLD // Calculate the reference bin (closes integer bin to r)  .
  {
    dintfreq	= r;
    start	= dintfreq + 1 - kern_half_width ;
    offset 	= ( r - start );				// This is rc-k for the first bin
  }

  FOLD // Clamp values to usable bounds  .
  {
    numkern 	= 2 * kern_half_width;
  }

  FOLD // Calculate coefficients  .
  {
    // Calculate all the constants
    int signZ		= (z < (T)0.0) ? -1 : 1;
    T absZ		= fabs_t(z);
    T sqrtAbsZ		= sqrt_t(absZ);
    T sq2overAbsZ	= (T)SQRT2 / sqrtAbsZ;
    T overSq2AbsZ	= (T)1.0 / (T)SQRT2 / sqrtAbsZ ;
    T Qk		= offset - z / (T)2.0;			// Adjust for acceleration

    for ( int i = 0 ; i < numkern; i++, Qk--, offset-- )	// Loop over the kernel elements  .
    {
      //  Get the address of the coefficient  .
      resp	= &out[start+i];

      FOLD // Calculate coefficient  .
      {
	if ( absZ > getZlim(offset) )				// Calculate raw coefficients .
	{
	  calc_coefficient_z<T, false>(Qk, offset, z, sq2overAbsZ, overSq2AbsZ, signZ, &resp->x, &resp->y);
	}
	else							// Calculate approximation coefficients  .
	{
	  calc_coefficient_a<T>(offset, z, &resp->x, &resp->y);
	}
      }
    }
  }
}

////////////////////  Convolution

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
__host__ __device__ void rz_convolution_cu(const dataT* inputData, long loR, long noBins, double r, T z, int kern_half_width, T* real, T* imag)
{
  *real = (T)0.0;
  *imag = (T)0.0;

  dataT   inp;							// The input data, this is a complex number stored as, float2 or double2
  long    dintfreq;						// Integer part of r      - double precision
  long    start;						// The first bin to use
  T       offset;						// The distance from the centre frequency (r) - NOTE: This could be double, float can get ~5 decimal places for lengths of < 999
  int     numkern;						// The actual number of kernel values to use
  T 	  resReal 	= 0;					// Response value - real
  T 	  resImag 	= 0;					// Response value - imaginary

  FOLD // Calculate the reference bin (closes integer bin to r)  .
  {
    dintfreq	= r;						// This type cast will always be the floor - unless R is negative =/
    start	= dintfreq + 1 - kern_half_width ;
    offset 	= (r - start);					// This is rc-k for the first bin
    numkern 	= 2 * kern_half_width;
  }

  FOLD // Adjust to input Data
  {
    // Adjust to start of input Data
    if ( start >= loR )
    {
      start	-= loR;						// Adjust for accessing the input FFT
    }
    else
    {
      // Start is below beginning of available data so start at available data
      numkern	-= loR - start;
      offset	= ( r - loR);					// This is rc-k for the first bin
      start	= 0;
    }

    // Adjust to end of input Data
    if ( start + numkern >= noBins )
    {
      numkern = noBins - start;
    }
  }

  FOLD // Main loop - Read input, calculate coefficients, multiply and sum results  .
  {
    // Calculate all the constants
    int signZ       = (z < (T)0.0) ? -1 : 1;
    T absZ          = fabs_t(z);
    T sqrtAbsZ      = sqrt_t(absZ);
    T sq2overAbsZ   = (T)SQRT2 / sqrtAbsZ;
    T overSq2AbsZ   = (T)1.0 / (T)SQRT2 / sqrtAbsZ ;
    T Qk            = offset - z / (T)2.0;			// Adjust for acceleration

    for ( int i = 0 ; i < numkern; i++ , Qk--, offset-- )	// Loop over the kernel elements
    {
      FOLD 							//  Read the input value  .
      {
	inp	= inputData[start+i];
      }

      FOLD 							// Calculate coefficients  .
      {
	if ( absZ > getZlim(offset) )				// Calculate raw coefficients .
	{
	  calc_coefficient_z<T, false>(Qk, offset, z, sq2overAbsZ, overSq2AbsZ, signZ, &resReal, &resImag);
	}
	else							// Calculate approximation coefficients  .
	{
	  calc_coefficient_a<T>(offset, z, &resReal, &resImag);
	}
      }

      FOLD 							//  Do the multiplication and sum  accumulate  .
      {
	*real += (resReal * inp.x - resImag * inp.y);
	*imag += (resReal * inp.y + resImag * inp.x);
      }
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
template<typename T, typename dataIn, typename dataOut>
__host__ __device__ void rz_convolution_cu(const dataIn* inputData, long loR, long inStride, double r, T z, int kern_half_width, dataOut* outData, int blkWidth, int noBlk)
{
  for ( int blk = 0; blk < noBlk; blk++ )
  {
    outData[blk].x = 0;
    outData[blk].y = 0;
  }

  dataIn  inp;							// The input data, this is a complex number stored as, float2 or double2
  long    dintfreq;						// Integer part of r      - double precision
  long    start;						// The first bin to use
  T       offset;						// The distance from the centre frequency (r) - NOTE: This could be double, float can get ~5 decimal places for lengths of < 999
  int     numkern;						// The actual number of kernel values to use
  T       resReal 	= (T)0.0;				// Response value - real
  T       resImag 	= (T)0.0;				// Response value - imaginary

  FOLD 								// Calculate the reference bin (closes integer bin to r)  .
  {
    dintfreq	= r;						// TODO: Check this when r is < 0 ?????
    start	= dintfreq + 1 - kern_half_width ;
  }

  FOLD 								// Clamp values to usable bounds  .
  {
    numkern	= 2 * kern_half_width;
    offset	= ( r - start);					// This is rc-k for the first bin
  }

  FOLD 								// Adjust for FFT  .
  {
    // Adjust to FFT
    start -= loR;						// Adjust for accessing the input FFT
  }

  FOLD // Main loop - Read input, calculate coefficients, multiply and sum results  .
  {
    // Calculate all the constants
    int signZ		= (z < (T)0.0) ? -1 : 1;
    T absZ		= fabs_t(z);
    T sqrtAbsZ		= sqrt_t(absZ);
    T sq2overAbsZ	= (T)SQRT2 / sqrtAbsZ;
    T overSq2AbsZ	= (T)1.0 / (T)SQRT2 / sqrtAbsZ ;
    T Qk		= offset - z / (T)2.0;			// Adjust for acceleration

    for ( int i = 0 ; i < numkern; i++, Qk--, offset--)		// Loop over the kernel elements
    {

      FOLD 							// Calculate coefficient  .
      {
	if ( fabs_t(z) > getZlim(offset) )			// Calculate raw coefficients .
	{
	  calc_coefficient_z<T, false>(Qk, offset, z, sq2overAbsZ, overSq2AbsZ, signZ, &resReal, &resImag);
	}
	else							// Calculate approximation coefficients  .
	{
	  calc_coefficient_a<T>(offset, z, &resReal, &resImag);
	}
      }

      // Use the coefficient on each input value with the same fractional part

      for ( int blk = 0; blk < noBlk; blk++ )
      {
	FOLD // Clamp values to usable bounds  .
	{
	  int idx = start+i+blk*blkWidth;

	  if ( idx >= 0 && idx < inStride )
	  {
	    FOLD //  Read the input value  .
	    {
	      inp	= inputData[idx];
	    }

	    FOLD //  Do the multiplication  .
	    {
	      outData[blk].x += (resReal * inp.x - resImag * inp.y);
	      outData[blk].y += (resReal * inp.y + resImag * inp.x);
	    }
	  }
	}
      }
    }
  }
}


//#ifdef WITH_OPT_BLK_SHF

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
template<int noColumns>
__host__ __device__ void rz_convolution_sfl(float2* inputData, const long loR, const long inStride, const double r, const float z, const int kern_half_width, float2* outData, const int colWidth, const int ic, const int cIdx)
{
  long    dintfreq;						// Integer part of r      - double precision
  long    start;						// The first bin to use
  float   offset;						// The distance from the centre frequency (r) - NOTE: This could be double, float can get ~5 decimal places for lengths of < 999
  int     numkern;						// The actual number of kernel values to use
  float   resReal;						// Response value - real
  float   resImag;						// Response value - imaginary

  FOLD 								// Calculate the reference bin (closes integer bin to r)  .
  {
    dintfreq	= r;						// TODO: Check this when r is < 0 ?????
    start	= dintfreq + 1 - kern_half_width ;
  }

  FOLD 								// Clamp values to usable bounds  .
  {
    numkern	= 2 * kern_half_width;
    offset	= ( r - cIdx - start);				// This is rc-k for the first bin
  }

  FOLD 								// Adjust for FFT  .
  {
    // Adjust to FFT
    start -= loR;						// Adjust for accessing the input FFT
  }

  FOLD // Zero the output
  {
    outData->x = 0.0f;
    outData->y = 0.0f;
  }

  inputData = &inputData[start+(cIdx)*colWidth];

  FOLD // Main loop - Read input, calculate coefficients, multiply and sum results  .
  {
    // Calculate all the constants
    int signZ		= (z < (float)0.0) ? -1 : 1;
    float absZ		= fabs_t(z);
    float sqrtAbsZ	= sqrt_t(absZ);
    float sq2overAbsZ	= (float)SQRT2 / sqrtAbsZ;
    float overSq2AbsZ	= (float)1.0 / (float)SQRT2 / sqrtAbsZ ;
    float Qk		= offset - z / (float)2.0;		// Adjust for acceleration

    for ( int i = 0 ; i < numkern; i+=noColumns, Qk-=noColumns, offset-=noColumns)		// Loop over the kernel elements
    {
      FOLD 							// Calculate coefficient  .
      {
	//calc_coefficient<float>(offset, z, &resReal, &resImag);
	if ( fabs_t(z) > getZlim(offset) )			// Calculate raw coefficients .
	{
	  calc_coefficient_z<float, false>(Qk, offset, z, sq2overAbsZ, overSq2AbsZ, signZ, &resReal, &resImag);
	}
	else							// Calculate approximation coefficients  .
	{
	  calc_coefficient_a<float>(offset, z, &resReal, &resImag);
	}
      }

      FOLD 							//  Do the multiplication and sum  accumulate  .
      {
	for( int idx = 0; idx < noColumns; idx++)
	{
	  // TODO: May have to do an end condition check here?

	  // Read input - These reads are generally coalesced
	  // I have found they are highly cached, so much so that no manual caching or sharing with shuffle is needed!
	  //float2 inp = inputData[start + i + idx + (cIdx)*colWidth];
	  float2 inp = inputData[i + idx];

#ifdef  __CUDA_ARCH__
	  float resCRea_c = __shfl(resReal, idx, noColumns );
	  float resImag_c = __shfl(resImag, idx, noColumns );
	  outData->x += (resCRea_c * inp.x - resImag_c * inp.y);
	  outData->y += (resCRea_c * inp.y + resImag_c * inp.x);
#else
	  //	  float resCRea_c;
	  //	  float resImag_c;
	  //
	  //	  FOLD 							// Calculate coefficient  .
	  //	  {
	  //	    int adjust = cIdx-idx; // TODO: this needs to be checked, sigh change?
	  //	    if ( fabs_t(z) > getZlim(offset) )			// Calculate raw coefficients .
	  //	    {
	  //	      calc_coefficient_z<float, false>(Qk+adjust, offset+adjust, z, sq2overAbsZ, overSq2AbsZ, signZ, &resCRea_c, &resImag_c);
	  //	    }
	  //	    else							// Calculate approximation coefficients  .
	  //	    {
	  //	      calc_coefficient_a<float>(offset+adjust, z, &resCRea_c, &resImag_c);
	  //	    }
	  //	  }
//	  outData->x += (resCRea_c * inp.x - resImag_c * inp.y);
//	  outData->y += (resCRea_c * inp.y + resImag_c * inp.x);
#endif

	}
      }
    }
  }
}

////////////////////  DBG

// DBG - Testing function
template<typename T>
__global__ void k_fresnlin(kerStruct inf)
{
  T v[DEVIS][3];

  T start	= inf.fList[0];
  T end		= inf.fList[1];
  T off2	= (end-start)/(float)inf.reps/(float)REPS;

  for ( int i = 0 ; i < inf.reps; ++i)
  {
    float off	= start + i/(float)inf.reps*(end-start);

    for ( int ii = 0; ii < REPS; ++ii )
    {
      off += off2;
#pragma unroll
      for ( int x = 0; x < DEVIS; x++ )
      {
	fresnl<T, T>(off + (T)0.00000001*x, &v[x][0], &v[x][1]);
	v[x][2] += v[x][0] + v[x][1];
      }
    }
  }

  if ( threadIdx.y * blockDim.x + threadIdx.x > 1024 )
  {
#pragma unroll
    for ( int x = 0; x < DEVIS; x++ )
    {
      printf("%f %f %f - ", v[x][0], v[x][1], v[x][2] );
    }
    printf("\n");
  }
}

template<typename T, typename T2>
__global__ void k_fresnEval(T* input, T2* output )
{
  const int bid = blockIdx.y  * gridDim.x  + blockIdx.x;		/// Block ID (flat index)
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;		/// Thread ID in block (flat index)
  const int gid = bid * blockDim.x* blockDim.y + tid;			/// Grid ID (flat index)

  T v = input[gid];
  T2 res;

  fresnl<T, T>(v, &res.x, &res.y);

  output[gid] = res;
}

//__global__ void k_fresnEval_d(double* input, double2* output )
//{
//  const int bid = blockIdx.y  * gridDim.x  + blockIdx.x;        	/// Block ID (flat index)
//  const int tid = threadIdx.y * blockDim.x + threadIdx.x;       	/// Thread ID in block (flat index)
//  const int gid = bid * blockDim.x* blockDim.y + tid;			/// Grid ID (flat index)
//
//  double v = input[gid];
//  double2 res;
//
//  fresnl<double, double>(v, &res.x, &res.y);
//
//  output[gid] = res;
//}

// DBG - Testing function
template<typename T>
__global__ void k_finterpin(kerStruct inf)
{
  T v[DEVIS][3];

  T start	= inf.fList[0];
  T end		= inf.fList[1];
  T off2	= (end-start)/(float)inf.reps/(float)REPS;

  for ( int i = 0 ; i < inf.reps; ++i)
  {
    float off	= start + i/(float)inf.reps*(end-start);

    for ( int ii = 0; ii < REPS; ++ii )
    {
      off += off2;
#pragma unroll
      for ( int x = 0; x < DEVIS; x++ )
      {
	calc_coefficient_r<T>( off + (T)0.00000001*x, &v[x][0], &v[x][1] );
	v[x][2] += v[x][0] + v[x][1];
      }
    }
  }

  if ( threadIdx.y * blockDim.x + threadIdx.x > 1024 )
  {
#pragma unroll
    for ( int x = 0; x < DEVIS; x++ )
    {
      printf("%f %f %f - ", v[x][0], v[x][1], v[x][2] );
    }
    printf("\n");
  }
}

template<typename T, typename T2>
__global__ void k_finterpEval(T* input, T2* output )
{
  const int bid = blockIdx.y  * gridDim.x  + blockIdx.x;        	/// Block ID (flat index)
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;       	/// Thread ID in block (flat index)
  const int gid = bid * blockDim.x* blockDim.y + tid;			/// Grid ID (flat index)

  T v = input[gid];
  T2 res;

  calc_coefficient_r<T>(v, &res.x, &res.y);

  output[gid] = res;
}

// DBG - Testing function
template<typename T>
__global__ void k_responsein(kerStruct inf)
{
  T v[DEVIS][3];

  // Standard case - single precision no phase
  T off	= inf.fList[0];
  T z	= inf.fList[1];

  for ( int i = 0 ; i < inf.reps; ++i)
  {
    for ( int ii = 0; ii < REPS; ++ii )
    {
#pragma unroll
      for ( int x = 0; x < DEVIS; x++ )
      {
	calc_coefficient<T>(off+ (T)0.00000001*x, z, &v[x][0], &v[x][1]) ;
	v[x][2] += v[x][0] + v[x][1];
      }
    }
  }

  if ( threadIdx.y * blockDim.x + threadIdx.x > 1024 )
  {
#pragma unroll
    for ( int x = 0; x < DEVIS; x++ )
    {
      printf("%f %f %f - ", v[x][0], v[x][1], v[x][2] );
    }
    printf("\n");
  }
}

// DBG Templates
template __global__ void k_fresnlin<float>(kerStruct inf);
template __global__ void k_fresnlin<double>(kerStruct inf);
template __global__ void k_responsein<float>(kerStruct inf);
template __global__ void k_responsein<double>(kerStruct inf);

template __global__ void k_fresnEval<float , float2 >(float* input,  float2* output  );
template __global__ void k_fresnEval<double, double2>(double* input, double2* output );


template __global__ void k_finterpEval<float , float2 >(float*  input, float2*  output );
template __global__ void k_finterpEval<double, double2>(double* input, double2* output );
template __global__ void k_finterpin<float >(kerStruct inf);
template __global__ void k_finterpin<double>(kerStruct inf);

//////////////////// Templates

template void fresnl<float,  float>  (float  xxa, float*  cc, float*  ss);
template void fresnl<float,  double> (double xxa, float*  cc, float*  ss);
template void fresnl<double, double> (double xxa, double* cc, double* ss);

////////////////////

template void calc_coefficient<float> (float  offset, float  z,  float*  real, float*  imag);
template void calc_coefficient<double>(double offset, double z,  double* real, double* imag);

template void calc_coefficient_bin<float> (long bin, double r, float  z,  float*  real, float*  imag);
template void calc_coefficient_bin<double>(long bin, double r, double z,  double* real, double* imag);


template void calc_coefficient_r<float >(float  dist, float  sinsinPI, float  sincosPI, float*  real, float*  imag);
template void calc_coefficient_r<double>(double dist, double sinsinPI, double sincosPI, double* real, double* imag);

template void calc_coefficient_r<float >(float  offset, float*  real, float*  imag);
template void calc_coefficient_r<double>(double offset, double* real, double* imag);


//template void calc_coefficient_z<float,  true >(float  Qk, float  dr, float  z, float  sq2overAbsZ, float  overSq2AbsZ, int sighnZ, float*  real, float*  imag);
//template void calc_coefficient_z<float,  false>(float  Qk, float  dr, float  z, float  sq2overAbsZ, float  overSq2AbsZ, int sighnZ, float*  real, float*  imag);
//template void calc_coefficient_z<double, true >(double Qk, double dr, double z, double sq2overAbsZ, double overSq2AbsZ, int sighnZ, double* real, double* imag);
//template void calc_coefficient_z<double, false>(double Qk, double dr, double z, double sq2overAbsZ, double overSq2AbsZ, int sighnZ, double* real, double* imag);

template void calc_coefficient_z<float,  true >(float  offset, float  z, float*  real, float*  imag);
template void calc_coefficient_z<float,  false>(float  offset, float  z, float*  real, float*  imag);
template void calc_coefficient_z<double, true >(double offset, double z, double* real, double* imag);
template void calc_coefficient_z<double, false>(double offset, double z, double* real, double* imag);

template inline void calc_coefficient_z<float,  false>(float Qk, float dr, float z, float sq2overAbsZ, float overSq2AbsZ, int sighnZ, float* real, float* imag);


template void calc_coefficient_a<float >(float  offset, float  z, float*  real, float*  imag);
template void calc_coefficient_a<double>(double offset, double z, double* real, double* imag);


template void rz_coefficients<double, double2>(double r, double z, int kern_half_width, double2* out);
template void rz_coefficients<float,  float2> (double r, float  z, int kern_half_width, float2*  out);

////////////////////


//#ifdef WITH_OPT_BLK_SHF
template void rz_convolution_sfl<1 >(float2* inputData, const long loR, const long inStride, const double r, const float z, const int kern_half_width, float2* outData, const int colWidth, const int ic, const int cIdx);
template void rz_convolution_sfl<2 >(float2* inputData, const long loR, const long inStride, const double r, const float z, const int kern_half_width, float2* outData, const int colWidth, const int ic, const int cIdx);
template void rz_convolution_sfl<4 >(float2* inputData, const long loR, const long inStride, const double r, const float z, const int kern_half_width, float2* outData, const int colWidth, const int ic, const int cIdx);
template void rz_convolution_sfl<8 >(float2* inputData, const long loR, const long inStride, const double r, const float z, const int kern_half_width, float2* outData, const int colWidth, const int ic, const int cIdx);
template void rz_convolution_sfl<16>(float2* inputData, const long loR, const long inStride, const double r, const float z, const int kern_half_width, float2* outData, const int colWidth, const int ic, const int cIdx);
template void rz_convolution_sfl<32>(float2* inputData, const long loR, const long inStride, const double r, const float z, const int kern_half_width, float2* outData, const int colWidth, const int ic, const int cIdx);
//#endif

template void rz_convolution_cu<float,  float2> (const float2*  inputData, long loR, long noBins, double r, float  z, int kern_half_width, float*  real, float*  imag);
//template void rz_convolution_cu<float,  double2>(const double2* inputData, long loR, long noBins, double r, float  z, int kern_half_width, float*  real, float*  imag);
template void rz_convolution_cu<double, float2> (const float2*  inputData, long loR, long noBins, double r, double z, int kern_half_width, double* real, double* imag);
//template void rz_convolution_cu<double, double2>(const double2* inputData, long loR, long noBins, double r, double z, int kern_half_width, double* real, double* imag);



template void rz_convolution_cu<float,  float2, float2> (const float2* inputData, long loR, long inStride, double r, float  z, int kern_half_width, float2* outData, int blkWidth, int noBlk);
template void rz_convolution_cu<double, float2, float2> (const float2* inputData, long loR, long inStride, double r, double z, int kern_half_width, float2* outData, int blkWidth, int noBlk);


////////////////////
