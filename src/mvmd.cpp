/*
 ============================================================================
 Name        : MVMD.cu
 Author      : Chris Laidler
 Version     :
 Copyright   :
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <math.h>

#include <boost/math/special_functions/bessel.hpp>

#include "makedata.h"

const double pi = boost::math::constants::pi<double>();

double MVMD(double k, double phase, double a, double* bess)
{

  if ( k == 0.0 )
  {
    return cos(2*pi*phase) + 1.0;
  }

  double bess_l;
  if (bess == NULL)
  {
    // calculate Bessel
    bess_l = boost::math::cyl_bessel_i<double, double>(0.0, k);
  }
  else
  {
    if ( *bess <= 0 )
    {
      // calculate new Bessel
      *bess = boost::math::cyl_bessel_i<double, double>(0.0, k);
    }
    bess_l = *bess;
  }

  double cosTerm      = cos(2*pi*phase);
  double expTerm      = exp(-k);

  double numerator    = exp(k*cosTerm) - expTerm;

  if ( isinf(numerator) )
  {
    fprintf(stderr, "ERROR: numerator in MVMD is infinity.\n");
    exit(EXIT_FAILURE);
  }

  double denominator  = bess_l - expTerm;

  if ( isinf(denominator) )
  {
    fprintf(stderr, "ERROR: denominator in MVMD is infinity.\n");
    exit(EXIT_FAILURE);
  }

  if ( denominator == 0 )
  {
    if ( numerator == 0 )
      return 0;

    fprintf(stderr, "ERROR: denominator in MVMD is 0.\n");
    exit(EXIT_FAILURE);
  }

  double val = a * numerator / denominator ;

  return val;
}

double calcFWHM(double k)
{
  double t1 = cosh(k);

  if ( isinf(t1) )
  {
    fprintf(stderr, "ERROR: %s cannot handle a k value of %.6f.\n", __FUNCTION__, k);
    exit(EXIT_FAILURE);
  }

  double t2 = log(t1)/k;
  double t3 = acos(t2);
  double FWHM_ss = acos( log(cosh(k))/k ) / pi;

  return FWHM_ss;
}

double calcFWHM_binSearch(double k)
{
  double bess     = 0;
  double amp      = MVMD(k, 1.0, 1, &bess);
  double phs      = 0.75;
  double first    = 0.5;
  double last     = 1.0;
  double hAmp;
  double height;
  double pHeight  = 1;

  int    ite      = 0;

  while ( true )
  {
    phs      = ( first + last ) / 2.0 ;
    hAmp     = MVMD(k, phs, 1, &bess);
    height   = hAmp / amp ;

    if ( pHeight == height )
    {
      return 2.0*(1.0-phs);
    }
    else
    {
      pHeight  = height;

      //printf("%i\t%25.20lf\t%25.20lf\n",ite, 2.0*(1.0-phs), height );

      if ( height == 0.5 )
      {
        return 2.0*(1.0-phs);
      }
      else if ( height < 0.5 )
      {
        first = phs;
      }
      else
      {
        last = phs;
      }
      ite++;
    }
  }
}

double calcK(double fwhm)
{
  double first    = 0.0;
  double last     = 700.0;
  double k;

  double FWHM     = calcFWHM(last);
  if ( fwhm <= FWHM )
  {
    fprintf(stderr, "ERROR: FWHM is below the limit %.5f function %s can calculate.\n", FWHM, __FUNCTION__ );
    exit(EXIT_FAILURE);
  }

  double prev     = 0;
  int   ite       = 0;

  if ( fwhm >= 0.5 )
    return 0;

  while ( true )
  {
    k     = ( first + last ) / 2.0 ;
    FWHM  = calcFWHM(k);

    //printf("%i\t%25.20lf\t%25.20lf\n",ite, k, FWHM );

    if ( FWHM == prev )
    {
      return k;
    }
    else if ( FWHM > fwhm )
    {
      first = k;
    }
    else
    {
      last = k;
    }
    prev = FWHM;
  }
}

//int main(void)
//{
//  double noVals   = 400;
//  double FWHM;
//  int idx=0;
//
//  double FWHM2    = calcFWHM( 710 );
//  //double k        = calcK(0.02);
//  //k               = calcK(0.0142);
//
//  double bess[100];
//  double k[100];
//  double sum[100];
//  double prevh[100];
//
//  printf("%6s\t", "Per");
//
//  for ( FWHM = 0.5; FWHM >= 0.014; FWHM -= 0.025 )
//  {
//    bess[idx]   = 0;
//    sum[idx]    = 0;
//    prevh[idx]  = -1;
//    k[idx]      = calcK(FWHM);
//    printf("%.10f\t", FWHM );
//    idx++;
//  }
//
//  printf("\n");
//
//  double width = 1.0 / noVals ;
//
//  for ( int i = 0; i <= noVals; i++)
//  {
//    double pre          = i / noVals;
//    printf("%6.4f\t", pre );
//    int idx=0;
//
//    for ( FWHM = 0.5; FWHM > 0.02; FWHM -= 0.025 )
//    {
//      double val          = MVMD(k[idx], pre+0.5, 1, &bess[idx]);
//      printf("%.10f\t", val );
//
//      if ( prevh[idx] != -1 )
//      {
//        double hh = ( prevh[idx] + val ) / 2.0 * width ;
//        //hh = width ;
//        sum[idx] += hh ;
//      }
//
//      prevh[idx] = val;
//      idx++;
//    }
//
//    printf("\n");
//  }
//
//  printf("%6s\t", "");
//  idx = 0;
//  for ( FWHM = 0.5; FWHM > 0.02; FWHM -= 0.025 )
//  {
//    printf("%.10f\t", sum[idx] );
//    idx++;
//  }
//  printf("\n");
//
//  return 0;
//}
