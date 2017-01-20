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
  bool usage		= 0 ;
  float zmax		= 0;
  float width		= 0;
  int   noHarms		= 16;
  uint accelLen 	= 0;
  float zRes		= 2;
  int noResPerBin	= 2;


  if ( argc != 4 )
    usage = true;
  else
  {
    zmax    = atof(argv[1]);
    width   = atof(argv[2]);
    noHarms = atof(argv[3]);

    if ( zmax < 0 || zmax > 1200 )
    {
      fprintf(stderr,"ERROR: invalid zmax.\n");
      usage = true;
    }

    zmax    = cu_calc_required_z<double>(1.0, zmax, zRes);

    int idx = round(log2(width*1000.0));

    if ( idx < 10 || idx > 15 )
    {
      fprintf(stderr,"ERROR: invalid width\n");
      usage = true;
    }
  }

  if (usage )
  {
    printf("\nUsage:\n  getAccelLen zmax width harms\n\nWher: 0 <= z-max <= 1200 and width is approximate width in 1000's ( ie. a width of 8 will give plains of width 8192 ) \n\n");
    return(1);
  }
  else
  {
    //accelLen = calcAccellen(width, zmax, noHarms, LOWACC, noResPerBin, zRes);
  }

  uint accelLenBasic = calcAccellen(width, zmax, LOWACC, noResPerBin);
  uint accelLenHalf  = calcAccellen(width, zmax, noHarms, LOWACC, noResPerBin, zRes);
  uint accelLenDivs  = floor(accelLenHalf/float(noHarms*noResPerBin))*(noHarms*noResPerBin);

  printf("%u	%u	%u\n", accelLenBasic, accelLenHalf, accelLenDivs );

  return 2;
}
