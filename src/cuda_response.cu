#include <iostream>
#include <stdio.h>

#include "cuda_math.h"
#include "cuda_response.h"
#include "cuda_accel.h"

#define FREESLIM1	1.600781059358212171622054418655453316130105033155		// Sqrt ( 2.5625 )
#define FREESLIM2	36974.0

__host__ __device__ inline float getFIlim(float nothing)
{
  return FILIM_F;
}

__host__ __device__ inline double getFIlim(double nothing)
{
  return FILIM_D;
}

__host__ __device__ inline void sinecos_fres(double x, float x2, float* sin, float* cos)
{
  double trigT 	= fmod_t(x * x, 4.0);
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
  T 	absX;
  idxT 	absXi;

  absXi      = fabs_t(x);
  absX       = absXi;

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
    *cc   = 0.5;
    *ss   = 0.5;
  }
  else					// Auxiliary functions for large argument  .
  {
    // x2 Could get really big so use double precision
    T x2	= absX * absX;

    t		= (T)PI * x2;
    u		= 1.0 / (t * t);
    t		= 1.0 / t;

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

    sinecos_fres(absXi, x2, &s, &c);
    //t     = (T)PIBYTWO * x2;
    //sincos_t(t, &s, &c);

    t     = (T)PI * absX;

    *cc   = (T)0.5 + (f * s - g * c) / t;
    *ss   = (T)0.5 - (f * c + g * s) / t;

  }

  if ( x < 0.0 )              // Swap as function is antisymmetric  .
  {
    *cc   = -*cc;
    *ss   = -*ss;
  }
}

__host__ __device__ inline void sinecos_resp(float Qk, float z, float PIoverZ, float* sin, float* cos)
{
  double x	= (double)Qk * (double)Qk / (double)z ;
  float  xx	= fmod_t(x, 2.0);
  xx		*= (float)PI;
  sincos_t(xx, sin, cos);

//  double sinD, cosD;
//  sincos_t((double)PIoverZ*(double)Qk*(double)Qk, &sinD, &cosD);
//  *sin = sinD;
//  *cos = cosD;
}

__host__ __device__ inline void sinecos_resp(double Qk, double z, double PIoverZ, double* sin, double* cos)
{
  double  xx	= fmod_t(Qk*Qk/z, 2.0);
  xx		*= (double)PI;
  sincos_t(xx, sin, cos);

  //sincos_t(PIoverZ*Qk*Qk, sin, cos);
}

/**
 *
 * Precedence for double:
 * 1:	sq2overAbsZ
 *
 * @param Qk
 * @param z
 * @param sq2overAbsZ
 * @param PIoverZ
 * @param overSq2AbsZ
 * @param sighnZ
 * @param real
 * @param imag
 */
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

  T sin, cos;
  sinecos_resp(Qk, z, PIoverZ, &sin, &cos);

  T Yk,Zk;
  T SZk, CZk, SYk, CYk;
  Yk = sq2overAbsZ * Qk ;
  Zk = sq2overAbsZ * ( Qk + z) ;

  if ( Yk > 200 || Yk < -200 )
  {
    // Yk and Zk will be sqared so they need to be double if they are large

    double Ykd, Zkd;

    Ykd = (double)sq2overAbsZ *   (double)Qk;
    Zkd = (double)sq2overAbsZ * ( (double)Qk + z ) ;

    fresnl<T, double>(Ykd, &SYk, &CYk);
    fresnl<T, double>(Zkd, &SZk, &CZk);
  }
  else
  {
    fresnl<T, T>(Yk, &SYk, &CYk);
    fresnl<T, T>(Zk, &SZk, &CZk);
  }

  T Sk =  ( SZk - SYk );  				// Can be float
  T Ck =  ( CYk - CZk ) * sighnZ ;			// Can be float
  //Ck *= sighnZ;

#if CORRECT_MULT
  // This is the version I get by doing the math
  //*real = overSq2AbsZ * (Sk*cos + Ck*sin) ;
  //*imag = overSq2AbsZ * (Sk*sin - Ck*cos) ;

  //// This is the corrected version ( math * -i )
  *real =  overSq2AbsZ * ( Sk * sin - Ck * cos ) ;
  *imag = -overSq2AbsZ * ( Sk * cos + Ck * sin ) ;
#else
  if ( flags )
  {
    //// This is the version I get by doing the math
    //*real = overSq2AbsZ * (Sk*cos + Ck*sin) ;
    //*imag = overSq2AbsZ * (Sk*sin - Ck*cos) ;

    //// This is the corrected version ( math * -i )
    *real =  overSq2AbsZ * (Sk*sin - Ck*cos) ;
    *imag = -overSq2AbsZ * (Sk*cos + Ck*sin) ;
  }
  else
  {
    // This is the version in accelsearch
    *real = overSq2AbsZ * (Sk*sin - Ck*cos) ;
    *imag = overSq2AbsZ * (Sk*cos + Ck*sin) ;
  }
#endif
}

template<typename T>
__host__ __device__ void calc_r_response(T dist, T sinsinPI, T sincosPI, T* real, T* imag)
{
  /* This is evaluating Eq (30) in:
   * Ransom, Scott M., Stephen S. Eikenberry, and John Middleditch. "Fourier techniques for very long astrophysical time-series analysis." The Astronomical Journal 124.3 (2002): 1788.
   *
   */

  //if ( dist > -SINCLIM && dist < SINCLIM )
  if ( dist > -getFIlim(dist) && dist < getFIlim(dist) )
  {
    // Correct for division by zero ie: sinc(0) = 1
    *real = (T)1.0;
    *imag = (T)0.0;
  }
  else
  {
    *real = sincosPI / dist ;
    *imag = -sinsinPI / dist ;
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
    T dist = -offset;
    T sin, cos;

    sincos_t(dist, &sin, &cos);					// TODO : this could be a bit more accurate if we split offset into fractional part

    calc_r_response(dist, sin*sin/(T)PI, sin*cos/(T)PI, real, imag);
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

    calc_z_response<T,1>(Qk, z, sq2overAbsZ, PIoverZ, overSq2AbsZ, signZ, real, imag);
  }
}

template<typename T, typename dataT>
__host__ __device__ void rz_interp_cu(dataT* inputData, long loR, long noBins, double r, float z, int kern_half_width, T* real, T* imag)
{
  *real = (T)0.0;
  *imag = (T)0.0;

  double real_D = 0 ;
  double imag_D = 0 ;

  double resReal_D = 0 ;
  double resImag_D = 0 ;

  dataT   inp;							// The input data, this is a complex number stored as, float2 or double2
  double  fracfreq;						// Fractional part of r   - double precision
  double  dintfreq;						// Integer part of r      - double precision
  long    start;						// The first bin to use
  double  offset;						// The distance from the centre frequency (r) - NOTE: This could be double, float can get ~5 decimal places for lengths of < 999
  int     numkern;						// The actual number of kernel values to use
  T 	  resReal 	= 0;					// Response value - real
  T 	  resImag 	= 0;					// Response value - imaginary

  // TMP
  T 	  respRealSum 	= 0;					//
  T 	  respImagSum 	= 0;					//

  T 	  inRealSum 	= 0;					//
  T 	  inImagSum 	= 0;					//

  T 	  mathRealSum 	= 0;					//
  T 	  mathImagSum 	= 0;					//

  T 	  accelRealSum 	= 0;					//
  T 	  accelImagSum 	= 0;					//

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

    if ( start < 0 )
    {
      numkern += start;						// Decrease number of kernel values
      start    = 0;
    }

    offset = ( r - start);					// This is rc-k for the first bin
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

    // Do all the trig calculations for the constants
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
	calc_r_response(dist, sinsinPI,  sincosPI, &resReal, &resImag);

	accelRealSum += resReal * inp.x - resImag*inp.y;
	accelImagSum += resReal * inp.y + resImag*inp.x;

	respRealSum += resReal;
	respImagSum += resImag;
      }

      FOLD 							//  Do the multiplication and sum  accumulate  .
      {
	*real	+= resReal * inp.x - resImag*inp.y;
	*imag	+= resReal * inp.y + resImag*inp.x;
      }

//      if ( (start+i) >= 97 && (start+i) <= 103 )
//      {
//	printf("%10.4f\t%li\t", fabs_t(r - (start+i)), (start+i));
//	printf("%8.3f\t%8.3f\t%8.3f\t", inRealSum, inImagSum, sqrt(POWERCU(inRealSum, inImagSum)));
//	printf("%8.3f\t%8.3f\t%8.3f\t", accelRealSum, accelImagSum, sqrt(POWERCU(accelRealSum, accelImagSum)));
//	printf("%8.3f\t%8.3f\t%8.3f\t", mathRealSum, mathImagSum, sqrt(POWERCU(mathRealSum, mathImagSum)));
//	printf("%8.3f\t%8.3f\t%8.3f\t", respRealSum, respImagSum, sqrt(POWERCU(respRealSum, respImagSum)));
//	printf("\n");
//      }

    }
  }
  else								// Use a correlation kernel  .
  {
    // Calculate all the constants
    int signZ       = (z < 0.0) ? -1 : 1;
    T absZ          = fabs_t(z);
    //double absZ          = fabs_t((double)z);
    T sqrtAbsZ      = sqrt_t(absZ);
    //double sqrtAbsZ      = sqrt_t((double)absZ);
    T sq2overAbsZ   = (T)SQRT2 / sqrtAbsZ;
    //double sq2overAbsZ   = (double)SQRT2 / sqrtAbsZ;
    T PIoverZ       = (T)PI / z;
    //double PIoverZ       = (double)PI / z;
    T overSq2AbsZ   = (T)1.0 / (T)SQRT2 / sqrtAbsZ ;
    //double overSq2AbsZ   = (double)1.0 / (double)SQRT2 / sqrtAbsZ ;
    T Qk            = offset - z / (T)2.0;			// Just for acceleration
    //double Qk       = offset - z / (double)2.0;			// Just for acceleration

    for ( int i = 0 ; i < numkern; i++, Qk-- )			// Loop over the kernel elements
    {

      FOLD 							//  Read the input value  .
      {
        inp	= inputData[start+i];
      }

      FOLD 							// Calculate response  .
      {
	calc_z_response<T,0>(Qk, z, sq2overAbsZ, PIoverZ, overSq2AbsZ, signZ, &resReal, &resImag);
	//calc_z_response<double,0>(Qk, z, sq2overAbsZ, PIoverZ, overSq2AbsZ, signZ, &resReal_D, &resImag_D);

	//double dr, di;
	//calc_z_response<double,0>(Qk, z, sq2overAbsZ, PIoverZ, overSq2AbsZ, signZ, &dr, &di);
	//resReal = dr;
	//resImag = di;

//	double SZk, CZk, SYk, CYk;
//	double sin, cos;
//
//	double Yk = sq2overAbsZ * Qk ;
//	double Zk = sq2overAbsZ * ( Qk + z) ;
//
//	float bace;
//	float frac;
//	frac	= modf_t(Qk, &bace);				// This is always double precision because - r needs to be r
//	//double xx = PIoverZ * (-2*bace*frac + frac*frac);
//
//	double xx = PIoverZ * Qk * Qk;
//	sincos_t(xx, &sin, &cos);
//
//	//fresnl<T>(Yk, &SYk, &CYk);
//	//fresnl<T>(Zk, &SZk, &CZk);
//	fresnl<double>(Yk, &SYk, &CYk);
//	fresnl<double>(Zk, &SZk, &CZk);
//
//	//T Sk =  SYk - SZk;
//	//T Ck =  CZk - CYk;
//	T Sk =  SZk - SYk;
//	T Ck =  CYk - CZk;
//
//	Ck *= signZ;
//
//	//// This is the corrected version ( math * -i )
//	resReal =  overSq2AbsZ * (Sk*sin - Ck*cos) ;
//	resImag = -overSq2AbsZ * (Sk*cos + Ck*sin) ;
      }

      FOLD 							//  Do the multiplication and sum  accumulate  .
      {
	*real	+= resReal * inp.x - resImag*inp.y;
	*imag	+= resReal * inp.y + resImag*inp.x;

//	*real	+= resReal_D * inp.x - resImag_D*inp.y;
//	*imag	+= resReal_D * inp.y + resImag_D*inp.x;
      }
    }
  }
//  *real = real_D;
//  *imag = imag_D;
}

template<typename T, typename outT>
__host__ __device__ void gen_response_cu(double r, float z, int kern_half_width, outT* out)
{
  outT*   resp;							// The input data, this is a complex number stored as, float2 or double2
  double  fracfreq;						// Fractional part of r   - double precision
  double  dintfreq;						// Integer part of r      - double precision
  long    start = 0;
  double  offset;						// The distance from the centre frequency (r)
  int     numkern;						// The actual number of kernel values to use
  T	  dummy;

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

  //if ( z < FILIM_F && z > -FILIM_F )				// Do a Fourier interpolation  .
  if ( z < getFIlim(dummy) && z > -getFIlim(dummy) )		// Do a Fourier interpolation  .
  {
    T dist = offset;
    T sin, cos;
    T sinsinPI, sincosPI;

    // Do all the trig calculations for the constants
    sincos_t(fracfreq*(T)PI, &sin, &cos);
    sinsinPI = sin * sin / (T)PI ;
    sincosPI = sin * cos / (T)PI ;

    for ( int i = 0 ; i < numkern; i++, dist-- )		// Loop over the kernel elements  .
    {
      //  Get the address of the output value  .
      resp	= &out[start+i];

      // Calculate response
      calc_r_response<T>(dist, sinsinPI,  sincosPI, &resp->x, &resp->y);

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


template<typename T, typename dataT>
__host__ __device__ void rz_interp_cu_inc(dataT* inputData, long loR, long noBins, double r, float z, int kern_half_width, T* real, T* imag)
{
  *real = (T)0.0;
  *imag = (T)0.0;

  dataT   inp;							// The input data, this is a complex number stored as, float2 or double2
  double  fracfreq;						// Fractional part of r   - double precision
  double  dintfreq;						// Integer part of r      - double precision
  double  offset;						// The distance from the centre frequency (r)

  T 	  resReal 	= 0;					// Response value - real
  T 	  resImag 	= 0;					// Response value - imaginary

  T 	  respRealSum 	= 0;					//
  T 	  respImagSum 	= 0;					//

  T 	  inRealSum 	= 0;					//
  T 	  inImagSum 	= 0;					//

  T 	  mathRealSum 	= 0;					//
  T 	  mathImagSum 	= 0;					//

  T 	  accelRealSum 	= 0;					//
  T 	  accelImagSum 	= 0;					//

  FOLD // Calculate the reference bin (closes integer bin to r)  .
  {
    fracfreq	= modf_t(r, &dintfreq);				// This is always double precision because - r needs to be r

    if ( fracfreq > 0.5 )					// Adjust to closest bin  .
    {
      fracfreq -= 1.0 ;
      dintfreq++;
    }

    offset = ( r - dintfreq);					// This is rc-k for the first bin
  }

  if ( z < FILIM_F && z > -FILIM_F )				// Do a Fourier interpolation  .
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
	  calc_r_response(dist, sinsinPI,  sincosPI, &resReal, &resImag);

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

	printf("%10.4f\t%li\t", fabs_t(r - k), k);
	printf("%8.3f\t%8.3f\t%8.3f\t", inRealSum, inImagSum, sqrt(POWERCU(inRealSum, inImagSum)));
	printf("%8.3f\t%8.3f\t%8.3f\t", accelRealSum, accelImagSum, sqrt(POWERCU(accelRealSum, accelImagSum)));
	printf("%8.3f\t%8.3f\t%8.3f\t", mathRealSum, mathImagSum, sqrt(POWERCU(mathRealSum, mathImagSum)));
	printf("%8.3f\t%8.3f\t%8.3f\t", respRealSum, respImagSum, sqrt(POWERCU(respRealSum, respImagSum)));
	printf("\n");

	if ( i == 0 )
	  break;
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

    for ( int i = 0 ; i <= kern_half_width; i++ )		// Loop over the kernel elements
    {
      for ( int sn = -1; sn < 2; sn += 2 )
      {
	long k = dintfreq - i*sn;
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

	  accelRealSum += resReal * inp.x - resImag*inp.y;
	  accelImagSum += resReal * inp.y + resImag*inp.x;

	  respRealSum += resReal;
	  respImagSum += resImag;

	  calc_z_response<T,1>(Qk, z, sq2overAbsZ, PIoverZ, overSq2AbsZ, signZ, &resReal, &resImag);

	  mathRealSum += resReal * inp.x - resImag*inp.y;
	  mathImagSum += resReal * inp.y + resImag*inp.x;

	}

	FOLD //  Do the multiplication and sum  accumulate  .
	{
	  *real	+= resReal * inp.x - resImag*inp.y;
	  *imag	+= resReal * inp.y + resImag*inp.x;
	}

	printf("%10.4f\t%li\t", fabs_t(r - k), k);
	printf("%8.3f\t%8.3f\t%8.3f\t", inRealSum, inImagSum, sqrt(POWERCU(inRealSum, inImagSum)));
	printf("%8.3f\t%8.3f\t%8.3f\t", accelRealSum, accelImagSum, sqrt(POWERCU(accelRealSum, accelImagSum)));
	printf("%8.3f\t%8.3f\t%8.3f\t", mathRealSum, mathImagSum, sqrt(POWERCU(mathRealSum, mathImagSum)));
	printf("%8.3f\t%8.3f\t%8.3f\t", respRealSum, respImagSum, sqrt(POWERCU(respRealSum, respImagSum)));
	printf("\n");

	if ( i == 0 )
	  break;
      }
    }
  }
}

template void fresnl<float,  float>  (float  xxa, float*  ss, float*  cc);
template void fresnl<float,  double> (double xxa, float*  ss, float*  cc);
template void fresnl<double, double> (double xxa, double* ss, double* cc);

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

template void rz_interp_cu_inc<float,  float2> (float2*  inputData, long loR, long noBins, double r, float z, int kern_half_width, float*  real, float*  imag);
template void rz_interp_cu_inc<float,  double2>(double2* inputData, long loR, long noBins, double r, float z, int kern_half_width, float*  real, float*  imag);
template void rz_interp_cu_inc<double, float2> (float2*  inputData, long loR, long noBins, double r, float z, int kern_half_width, double* real, double* imag);
template void rz_interp_cu_inc<double, double2>(double2* inputData, long loR, long noBins, double r, float z, int kern_half_width, double* real, double* imag);


template void gen_response_cu<double, double2>(double r, float z, int kern_half_width, double2* out);
template void gen_response_cu<float,  float2> (double r, float z, int kern_half_width, float2*  out);
