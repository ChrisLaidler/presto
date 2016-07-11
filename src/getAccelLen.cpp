#include <stdio.h>      /* printf, fgets */
#include <stdlib.h>     /* atof */
#include <math.h>       /* sin */


extern "C"
{
#define __float128 long double
#include "accel.h"
}

//#include "cuda_accel.h"
//#include "cuda_utils.h"
#include "cuda_accel_utils.h"
//#include "cuda_accel_IN.h"

int main(int argc, char *argv[])
{
  bool usage = 0 ;

  uint accelLen = 0;

  if ( argc != 4 )
    usage = true;

  if (usage )
  {
    printf("\nUsage:\n  getAccelLen zmax width harms\n\nWher 10 <= zmanx <= 1200 and width is approximate width in 1000's ( ie. a width of 8 will give plains of width 8192 ) \n\n");
    return(1);
  }

  float zmax    = atof(argv[1]);
  float width   = atof(argv[2]);
  int   noHarms = atof(argv[3]);

  if ( zmax < 10 || zmax > 1200 )
  {
    fprintf(stderr,"ERROR: invalid zmax.\n");
    usage = true;
  }

  int idx = round(log2(width*1000.0));

  if ( idx < 10 || idx > 15 )
  {
    fprintf(stderr,"ERROR: invalid width\n");
    usage = true;
  }



  presto_interp_acc  accuracy = LOWACC;

  if ( noHarms > 1 )
  {

    // Working with a family of planes

    int   oAccelLen1, oAccelLen2;

    // This adjustment makes sure no more than half the harmonics are in the largest stack (reduce waisted work - gives a 0.01 - 0.12 speed increase )
    oAccelLen1  = calcAccellen(width,     zmax, accuracy, 2);
    oAccelLen2  = calcAccellen(width/2.0, zmax/2.0, accuracy, 2);

    if ( width > 100 )
    {
      // The user specified the exact width they want to use for accellen
      accelLen  = oAccelLen1;
    }
    else
    {
      // Use double the accellen of the half plane
      accelLen  = MIN(oAccelLen2*2, oAccelLen1);
    }

    if ( width < 100 ) // Check  .
    {
      float fWidth    = floor(cu_calc_fftlen<double>(1, zmax, accelLen, accuracy, 2, 2.0)/1000.0);

      float ss        = cu_calc_fftlen<double>(1, zmax, accelLen, accuracy, 2, 2.0) ;
      float l2        = log2( ss );

      if      ( l2 == 10 )
        fWidth = 1 ;
      else if ( l2 == 11 )
        fWidth = 2 ;
      else if ( l2 == 12 )
        fWidth = 4 ;
      else if ( l2 == 13 )
        fWidth = 8 ;
      else if ( l2 == 14 )
        fWidth = 16 ;
      else if ( l2 == 15 )
        fWidth = 32 ;
      else if ( l2 == 16 )
        fWidth = 64 ;

      if ( fWidth != width )
      {
        fprintf(stderr,"ERROR: Width calculation did not give the desired value.\n");
        exit(EXIT_FAILURE);
      }
    }
  }
  else
  {
    // Just a single plane
    accelLen = calcAccellen(width, zmax, accuracy, 2);
  }

  printf("%u\n", accelLen);

//  {
//    float szzz = pow(2 , idx );
//
//    float accelLen = 0;
//
//    float halfwidth =  cu_z_resp_halfwidth<double>(zmax);
//
//    accelLen    = floor(szzz - 2 - 2 * ACCEL_NUMBETWEEN * halfwidth);
//    if (accelLen < 100 )
//    {
//      printf("Got a accelLen of: %.0f but thats too small.\n", accelLen);
//      return 3;
//    }
//
//    printf("%.0f\n", accelLen);
//    return 0;
//  }

  return 2;
}
