
#include <curand.h>
#include <math.h>             // log
#include <curand_kernel.h>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/binomial.hpp>

#include "cuda_math_ext.h"
#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"
#include "cuda_response.h"

const double EPS    = std::numeric_limits<double>::epsilon();
const double FPMIN  = std::numeric_limits<double>::min()/EPS;

static const int ngau = 18;
const double y[18] = {0.0021695375159141994,0.011413521097787704,0.027972308950302116,0.051727015600492421,0.082502225484340941,0.12007019910960293, 0.16415283300752470, 0.21442376986779355, 0.27051082840644336, 0.33199876341447887, 0.39843234186401943, 0.46931971407375483, 0.54413605556657973, 0.62232745288031077, 0.70331500465597174, 0.78649910768313447, 0.87126389619061517, 0.95698180152629142  };
const double w[18] = {0.0055657196642445571,0.012915947284065419,0.020181515297735382,0.027298621498568734,0.034213810770299537,0.040875750923643261,0.047235083490265582,0.053244713977759692,0.058860144245324798,0.064039797355015485,0.068745323835736408,0.072941885005653087,0.076598410645870640,0.079687828912071670,0.082187266704339706,0.084078218979661945,0.085346685739338721,0.085983275670394821 };


template<int n>
void cdfgam_d(double x, double *p, double* q)
{
  if      ( n == 1  )
  {
    *q = exp(-x);
  }
  else if ( n == 2  )
  {
    *q = exp(-x)*( x + 1.0 );
  }
  else if ( n == 4  )
  {
    *q = exp(-x)*( x*(x*(x/6.0 + 0.5) + 1.0 ) + 1.0 );
  }
  else if ( n == 8  )
  {
    *q = exp(-x)*( x*(x*(x*(x*(x*(x*(x/5040.0 + 1.0/720.0 ) + 1.0/120.0 ) + 1.0/24.0 ) + 1.0/6.0 ) + 0.5 ) + 1.0 ) + 1.0 );
  }
  else if ( n == 16 )
  {
    *q = exp(-x)*( x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x*(x/1.307674368e12 +  1.0/8.71782912e10 ) \
        + 1.0/6227020800.0 )+ 1.0/479001600.0 ) \
        + 1.0/39916800.0 )+ 1.0/3628800.0 )     \
        + 1.0/362880.0 ) + 1.0/40320.0 )        \
        + 1.0/5040.0 ) + 1.0/720.0 ) + 1.0/120.0 ) + 1.0/24.0 ) + 1.0/6.0 ) + 0.5 ) + 1.0 )  + 1.0 );
  }
  else
  {
    *q = 1.0 + x ;
    double numerator    = x;
    double denominator  = 1.0;

#pragma unroll
    for ( int i = 2 ; i < n ; i ++ )
    {
      denominator *= i;
      numerator   *= x;
      *q += numerator/denominator;
    }
  }
  *p = 1-*q;
}

double gammln(double xx)
{
  double x,tmp,ser;
  static double cof[6]= { 76.18009173, -86.50532033, 24.01409822, -1.231739516, 0.120858003e-2, -0.536382e-5 };
  int j;

  x       =   xx - 1.0;
  tmp     =   x + 5.5;
  tmp     -=  (x+0.5)*log(tmp);
  ser     =   1.0;

  for ( j=0; j<=5;  j++ )
  {
    x     += 1.0;
    ser   += cof[j]/x;
  }

  return -tmp + log(2.50662827465*ser);
}

/** Inverse normal CDF - ie calculate σ from p and/or q
 * We include p and q because if p is close to 1 or -1 , q can hold more precision
 */
__host__ __device__ double incdf (double p, double q )
{
  double a[] = {              \
      -3.969683028665376e+01, \
      2.209460984245205e+02,  \
      -2.759285104469687e+02, \
      1.383577518672690e+02,  \
      -3.066479806614716e+01, \
      2.506628277459239e+00   };

  double b[] = {              \
      -5.447609879822406e+01, \
      1.615858368580409e+02,  \
      -1.556989798598866e+02, \
      6.680131188771972e+01,  \
      -1.328068155288572e+01  };

  double c[] = {              \
      -7.784894002430293e-03, \
      -3.223964580411365e-01, \
      -2.400758277161838e+00, \
      -2.549732539343734e+00, \
      4.374664141464968e+00, \
      2.938163982698783e+00 };

  double d[] = {            \
      7.784695709041462e-03, \
      3.224671290700398e-01, \
      2.445134137142996e+00, \
      3.754408661907416e+00 };

  double l, ll, x, e, u;
  double sighn = 1.0;

  // More precision in q so use it
  if ( p > 0.99 || p < -0.99 )
  {
    if ( q < 1.0 )
    {
      sighn = -1.0;
      double hold = p;
      p = q;
      q = hold;
    }
  }

  // Make an initial estimate for x
  // The algorithm taken from: http://home.online.no/~pjacklam/notes/invnorm/#The_algorithm
  if ( 0.02425 <= p && p <= 0.97575 )
  {
    l    =  p - 0.5;
    ll   = l*l;
    x    = (((((a[0]*ll+a[1])*ll+a[2])*ll+a[3])*ll+a[4])*ll+a[5])*l / (((((b[0]*ll+b[1])*ll+b[2])*ll+b[3])*ll+b[4])*ll+1.0);
  }
  else
  {
    if ( p == 0 )
      return NAN;	// I found sigma bottoms out at ~ -35 0 NAN marks an error

    if ( 0.02425 > p )
    {
      l = sqrt(-2.0*log(p));
    }
    else if ( 0.97575 < p )
    {
      l = sqrt(-2.0*log( 1.0 - p ));
    }
    x = (((((c[0]*l+c[1])*l+c[2])*l+c[3])*l+c[4])*l+c[5]) / ((((d[0]*l+d[1])*l+d[2])*l+d[3])*l+1.0);

    if ( 0.97575 < p )
    {
      x *= -1.0;
    }
  }

  // Now do a Newton Raphson recursion to refine the answer.
  // Using erfc and exp to calculate  f(x) = Φ(x)-p  and  f'(x) = Φ'(x)
  double f = 0.5 * erfc(-x/1.414213562373095048801688724209) - p ;
  double xOld = x;
  for ( int i = 0; i < 10 ; i++ ) // Note: only doing 10 recursions this could be pushed up
  {
    u = 0.398942*exp(-x*x/2.0);
    x = x - f / u ;

    f = 0.5 * erfc(-x/1.414213562373095048801688724209) - p;
    e = f / p;

    if ( fabs(e) < 1e-15 || ( x == xOld ) )
      break ;

    xOld = x;
  }

  return sighn*x;
}

//Incomplete gamma by quadrature. Returns P .a; x/ or Q.a; x/, when psig is 1 or 0,
//respectively. User should not call directly.
double gammpapprox(double a, double x, int psig)
{
  double  xu,t,sum,ans;
  double  a1      = a-1.0;
  double  lna1    = log(a1);
  double  sqrta1  = sqrt(a1);
  double  gln     = gammln(a);

  //Set how far to integrate into the tail:
  if (x > a1)
    xu = MAX(a1 + 11.5*sqrta1, x + 6.0*sqrta1);
  else
    xu = MAX(0.,MIN(a1 - 7.5*sqrta1, x - 5.0*sqrta1));

  sum = 0;

  for ( int j=0; j < ngau; j++) // Gauss-Legendre
  {
    t = x + (xu-x)*y[j];
    sum += w[j]*exp(-(t-a1)+a1*(log(t)-lna1));
  }
  ans = sum*(xu-x)*exp(a1*(lna1-1.)-gln);
  return (psig?(ans>0.0? 1.0-ans:-ans):(ans>=0.0? ans:1.0+ans));
}

double gser(const double a, const double x)
{
  //Returns the incomplete gamma function P .a; x/ evaluated by its series representation.
  //Also sets ln .a/ as gln. User should not call directly.
  double sum,del,ap, gln;

  gln=gammln(a);
  ap=a;
  del=sum=1.0/a;
  for (;;)
  {
    ++ap;
    del *= x/ap;
    sum += del;
    if (fabs(del) < fabs(sum)*EPS)
    {
      return sum*exp(-x+a*log(x)-gln);
    }
  }
}

double gcf(const double a, const double x)
{
  //Returns the incomplete gamma function Q.a; x/ evaluated by its continued fraction rep-
  //resentation. Also sets ln .a/ as gln. User should not call directly.
  int i;
  double an,b,c,d,del,h;

  double gln  = gammln(a);
  b           = x+1.0-a;
  //Set up for evaluating continued fraction
  c           = 1.0/FPMIN;
  //by modified Lentz’s method (5.2)
  d           = 1.0/b;
  //with b0 D 0.
  h           = d;

  for (i=1;;i++)
  {
    //Iterate to convergence.
    an = -i*(i-a);
    b += 2.0;
    d=an*d+b;
    if (fabs(d) < FPMIN)
      d=FPMIN;
    c=b+an/c;
    if (fabs(c) < FPMIN)
      c=FPMIN;
    d=1.0/d;
    del=d*c;
    h *= del;
    if (fabs(del-1.0) <= EPS)
      break;
  }

  return exp(-x+a*log(x)-gln)*h;
  //Put factors in front.
}

//Returns the incomplete gamma function P .a; x/.
double gammp(const double a, const double x)
{
  if (x < 0.0 || a <= 0.0)
  {
    throw("bad args in gammp");
  }
  if (x == 0.0)
  {
    return 0.0;
  }
  else if ((int)a >= 100 )                      // Quadrature  .
  {
    return gammpapprox(a,x,1);
  }
  else if (x < a+1.0)                           // Use the series representation  .
  {
    return gser(a,x);
  }
  else                                          // Use the continued fraction representation  .
  {
    return 1.0-gcf(a,x);
  }

}

double gammq(const double a, const double x)
{
  //Returns the incomplete gamma function Q.a; x/ Á 1 P .a; x/.
  if (x < 0.0 || a <= 0.0)
    throw("bad args in gammq");
  if (x == 0.0)
    return 1.0;
  else if ((int)a >= 100)         // Quadrature.
    return gammpapprox(a,x,0);
  else if (x < a+1.0)             // Use the series representation.
    return 1.0-gser(a,x);
  else                            // Use the continued fraction representation.
    return gcf(a,x);
}

double logIGamma_i(int s, double x )
{
  //double x = 1.592432984e8 ;
  //int s = 10;

  double num = pow(x,0) ;
  double den = 1;

  double sum = num/den;
  double trm;

  for( int k = 1; k <= s-1; k++ )
  {
    num   = pow(x,k) ;
    den  *= k;

    trm   = num/den;

    sum  += trm;

    printf("%03i  trm %6e   sum: %6e \n", k, trm, sum );
  }

  double t1 = lgamma((double)s) ;
  double t2 = -x ;
  double t3 = log(sum) ;
  return t1 + t2 + t3 ;
}

double logQChi2_i(int s, double x )
{
  double sum = 0 ;
  double num;
  double den;
  double trm;
  double sum0;

  for( int k = s-1; k >= 0 ; k-- )
  {
    sum0  = sum;
    num   = pow(x,k) ;
    den   = boost::math::factorial<double>(k);
    trm   = num/den;
    sum  += trm;

    if ( sum-sum0 == 0 )
      break;
  }

  double t2 = -x ;
  double t3 = log(sum) ;
  return t2 + t3 ;
}

void calcNQ(double qOrr, long long n, double* p, double* q)
{
  double qq  = 0;
  double pp  = 1;

  double trueV = 1-pow((1-qOrr),n);

  if ( trueV > 0.95 )
  {
    *q = (double)( (long double)1.0 -  pow((long double)(1.0-qOrr),(long double)n) );
    *p = (double)(        (long double)pow((long double)(1.0-qOrr),(long double)n) );
    return;
  }

  FOLD // Else do a series expansion  .
  {
    double  dff ;
    double term   = 1;
    long long k   = 0;
    double  sum0  = qq;
    double  coef  = 1;
    double  fact  = 1;

    qq = 0;

    do
    {
      sum0 = qq;
      coef *= ( n - (k) );
      k++;
      fact *= k;
      double bcoef = coef / fact ;

      double t1   = pow(-qOrr,k);

      if( t1 == 0 )
      {
        if ( k > 1 )
        {
          *p = pp ;
          *q = qq ;
          return;
        }
        else
        {
          *p = 1 - n * qOrr;
          *q =     n * qOrr;
          return;
        }
      }

      term = bcoef*t1;
      qq  -= term;
      pp  += term;
      dff  = fabs(sum0-qq);

//      if ( trueV > 0.5 )
//	printf("calcNQ %03i sum: %.4e  term: %.6e   dff: %.3e\n", k-1, pp, term, dff );
    }
    while ( dff > 0 && k < n && k <= 20 );

    *p = pp ;
    *q = qq ;
  }
}

/**  Calculate the STD normal distribution sigma value of a the sum of a normalised powers  .
 *
 * This is thread safe!
 *
 * @param poww		The incoherent sum of numharm powers
 * @param numharm	The number of powers summed
 * @param numindep	The total number of powers the given powers were sampled from
 * @return
 */
double candidate_sigma_cu(double poww, int numharm, long long numindep)
{
  double logQ, gpu_p, gpu_q, sigc ;

  int     k       = numharm * 2.0 ;     // Each harm is 2 powers
  double  gamP    = poww * 2.0 ;        // A just for normalisation of powers
  int     n       = numharm;

  if ( poww > 100 )
  {
    if      ( n == 1  )
    {
      logQ = -poww;
    }
    else if ( n == 2  )
    {
      logQ = -poww+log( poww + 1.0 );
    }
    else if ( n == 4  )
    {
      logQ = -poww + log( poww*(poww*(poww/6.0 + 0.5) + 1.0 ) + 1.0 );
    }
    else if ( n == 8  )
    {
      logQ = -poww + log( poww*(poww*(poww*(poww*(poww*(poww*(poww/5040.0 + 1.0/720.0 ) + 1.0/120.0 ) + 1.0/24.0 ) + 1.0/6.0 ) + 0.5 ) + 1.0 ) + 1.0 );
    }
    else if ( n == 16 )
    {
      logQ = -poww + log( poww*(poww*(poww*(poww*(poww*(poww*(poww*(poww*(poww*(poww*(poww*(poww*(poww*(poww*(poww/1.307674368e12 +  1.0/8.71782912e10 ) \
          + 1.0/6227020800.0 )+ 1.0/479001600.0 ) \
          + 1.0/39916800.0 )+ 1.0/3628800.0 ) \
          + 1.0/362880.0 ) + 1.0/40320.0 ) \
          + 1.0/5040.0 ) + 1.0/720.0 ) + 1.0/120.0 ) + 1.0/24.0 ) + 1.0/6.0 ) + 0.5 ) + 1.0 )  + 1.0 );
    }
    else
    {
      logQ = logQChi2_i(k / 2.0, gamP / 2.0 ) ;
    }

    // Correct q for number of trials
    logQ    += log( (double)numindep );

    double l = sqrt(-2.0*logQ);
    sigc     = l - ( 2.515517 + l * (0.802853 + l * 0.010328) ) / ( 1.0 + l * (1.432788 + l * (0.189269 + l * 0.001308)) ) ;

    return sigc;
  }
  else
  {
    if      ( numharm == 1 )
      cdfgam_d<1>(poww, &gpu_p, &gpu_q );
    else if ( numharm == 2 )
      cdfgam_d<2>(poww, &gpu_p, &gpu_q );
    else if ( numharm == 4 )
      cdfgam_d<4>(poww, &gpu_p, &gpu_q );
    else if ( numharm == 8 )
      cdfgam_d<8>(poww, &gpu_p, &gpu_q );
    else if ( numharm == 16)
      cdfgam_d<16>(poww, &gpu_p, &gpu_q );
    else
    {
      gpu_p = boost::math::gamma_p<double>(k / 2.0, gamP / 2.0 ) ;
      gpu_q = boost::math::gamma_q<double>(k / 2.0, gamP / 2.0 ) ;
    }

    // Correct q for number of trials
    calcNQ(gpu_q, numindep, &gpu_p, &gpu_q);

    sigc = incdf(gpu_p, gpu_q);

    return sigc;
  }
}
