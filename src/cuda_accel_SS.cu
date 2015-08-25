#include "cuda_accel_SS.h"


/*
#include <cub/cub.cuh>
#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>

#include "cuda_accel_utils.h"
#include "cuda_utils.h"

extern "C"
{
#define __float128 long double
#include "accel.h"
}
 */

#include <cufft.h>
#include <algorithm>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>

#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"
#include "cuda_accel_SS.h"

//======================================= Constant memory =================================================\\

__device__ __constant__ int       YINDS[MAX_YINDS];
__device__ __constant__ float     POWERCUT_STAGE[MAX_HARM_NO];
__device__ __constant__ float     NUMINDEP_STAGE[MAX_HARM_NO];

__device__ __constant__ int       HEIGHT_STAGE[MAX_HARM_NO];         ///< Plain heights in stage order
__device__ __constant__ int       STRIDE_STAGE[MAX_HARM_NO];         ///< Plain strides in stage order
__device__ __constant__ int       HWIDTH_STAGE[MAX_HARM_NO];         ///< Plain half width in stage order

__device__ __constant__ float*    PLN_START;
__device__ __constant__ uint      PLN_STRIDE;
__device__ __constant__ int       NO_STEPS;
__device__ __constant__ int       ALEN;

//====================================== Constant variables  ===============================================\\

__device__ const float FRAC_STAGE[16]     =  { 1.0000f, 0.5000f, 0.7500f, 0.2500f, 0.8750f, 0.6250f, 0.3750f, 0.1250f, 0.9375f, 0.8125f, 0.6875f, 0.5625f, 0.4375f, 0.3125f, 0.1875f, 0.0625f } ;

//__device__ const float FRAC_STAGE[16]     =  { 1.0000f, 0.5000f, 0.2500f, 0.7500f, 0.1250f, 0.3750f, 0.6250f, 0.8750f, 0.0625f, 0.1875f, 0.3125f, 0.4375f, 0.5625f, 0.6875f, 0.8125f, 0.9375f } ;

__device__ const float FRAC_HARM[16]      =  { 1.0f, 0.9375f, 0.875f, 0.8125f, 0.75f, 0.6875f, 0.625f, 0.5625f, 0.5f, 0.4375f, 0.375f, 0.3125f, 0.25f, 0.1875f, 0.125f, 0.0625f } ;
__device__ const short STAGE[5][2]        =  { {0,0}, {1,1}, {2,3}, {4,7}, {8,15} } ;
__device__ const short CHUNKSZE[5]        =  { 4, 8, 8, 8, 8 } ;

//======================================= Global variables  ================================================\\


//========================================== Functions  ====================================================\\

/** Return x such that 2**x = n
 *
 * @param n
 * @return
 */
__host__ __device__ inline int twon_to_index(int n)
{
  int x = 0;

  while (n > 1)
  {
    n >>= 1;
    x++;
  }
  return x;
}

template<uint FLAGS>
__device__ inline int getY(int plainY, const int noSteps,  const int step, const int plainHeight = 0 )
{
  // Calculate y indice from interleave method
  if      ( FLAGS & FLAG_ITLV_ROW )
  {
    return plainY * noSteps + step;
  }
  else if ( FLAGS & FLAG_ITLV_PLN )
  {
    return plainY + plainHeight*step;
  }
  /*
  else if ( FLAGS & FLAG_ITLV_STK )
  {
    return plainY + stackHeight*step;
  }
   */
  else
    return 0;
}

template<uint FLAGS>
__device__ inline float getPower(const int ix, const int iy, cudaTextureObject_t tex, fcomplexcu* base, const int stride)
{
  if  ( (FLAGS & FLAG_SAS_TEX ) )
  {
    const float2 cmpf = tex2D < float2 > (tex, ix, iy);
    return (cmpf.x * cmpf.x + cmpf.y * cmpf.y);
  }
  else
  {
    const fcomplexcu cmpc  = base[iy*stride+ix];
    return (cmpc.r * cmpc.r + cmpc.i * cmpc.i);
  }
}

//double gammp(const Doub a, const Doub x)
//{
//  //Returns the incomplete gamma function P .a; x/.
//  if (x < 0.0 || a <= 0.0)
//    throw("bad args in gammp");
//  if (x == 0.0)
//    return 0.0;
//  else if ((Int)a >= ASWITCH)
//    return gammpapprox(a,x,1);    //  Quadrature.
//  else if (x < a+1.0)
//    return gser(a,x);             //  Use the series representation.
//  else
//    return 1.0-gcf(a,x);          //  Use the continued fraction representation.
//}

inline float half2float(const ushort h)
{
  unsigned int sign = ((h >> 15) & 1);
  unsigned int exponent = ((h >> 10) & 0x1f);
  unsigned int mantissa = ((h & 0x3ff) << 13);

  if (exponent == 0x1f)   	// NaN or Inf
  {
    mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
    exponent = 0xff;
  }
  else if (!exponent)       // Denorm or Zero
  {
    if (mantissa)
    {
      unsigned int msb;
      exponent = 0x71;
      do
      {
        msb = (mantissa & 0x400000);
        mantissa <<= 1;  /* normalize */
        --exponent;
      }
      while (!msb);

      mantissa &= 0x7fffff;  /* 1.mantissa is implicit */
    }
  }
  else
  {
    exponent += 0x70;
  }

  uint res = ((sign << 31) | (exponent << 23) | mantissa);
  return  *((float*)(&res));
}

/** Calculate the CDF of a gamma distribution
 */
__host__ __device__ void cdfgam_d(double x, int n, double *p, double* q)
{
  if ( x <= 0 )
  {
    *p = 0;
    *q = 1;
    return;
  }

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

    for ( int i = 2 ; i < n ; i ++ )
    {
      denominator *= i;
      numerator   *= x;
      *q += numerator/denominator;
    }
  }

  *p = 1-*q;
}

/** Calculate the CDF of a gamma distribution
 */
template<int n>
__host__ __device__ void cdfgam_d(double x, double *p, double* q)
{
  if ( x <= 0 )
  {
    *p = 0;
    *q = 1;
    return;
  }

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
      return 0;

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

/** Calculate a sigma value
 */
__host__ __device__ double candidate_sigma_cu(double poww, int numharm, long long numindep)
{
  int n = numharm;
  double gpu_p, gpu_q, sigc ;

  if ( poww > 100)
  {
    cdfgam_d(poww, n*2, &gpu_p, &gpu_q );

    double logQ;
    if      ( n == 1 )
    {
      logQ = -poww;
    }
    else if ( n == 2 )
    {
      logQ = -poww+log( poww + 1.0 );
    }
    else if ( n == 4 )
    {
      logQ = -poww + log( poww*(poww*(poww/6.0 + 0.5) + 1.0 ) + 1.0 );
    }
    else if ( n == 8 )
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

    logQ += log( (double)numindep );

    double l = sqrt(-2.0*logQ);

    double sigc = l - ( 2.515517 + l * (0.802853 + l * 0.010328) ) / ( 1.0 + l * (1.432788 + l * (0.189269 + l * 0.001308)) ) ;

    return sigc;
  }
  else
  {
    if(numharm==1)
      cdfgam_d<1>(poww, &gpu_p, &gpu_q );
    else if(numharm==2)
      cdfgam_d<2>(poww, &gpu_p, &gpu_q );
    else if(numharm==4)
      cdfgam_d<4>(poww, &gpu_p, &gpu_q );
    else if(numharm==8)
      cdfgam_d<8>(poww, &gpu_p, &gpu_q );
    else if(numharm==16)
      cdfgam_d<16>(poww, &gpu_p, &gpu_q );
    else
    {
      cdfgam_d(poww, numharm*2, &gpu_p, &gpu_q );
    }

    if (gpu_p == 1.0)
      gpu_q *= numindep;
    else
    {
      double lq = log(gpu_q * numindep);
      double q2 = exp(lq);

      double pp = pow((1.0-gpu_q),1.0/(double)numindep);
      double qq = 1 - pp;
      sigc = incdf(pp, qq);

      gpu_q = 1.0 - pow(gpu_p, (double)numindep);
    }
    gpu_p = 1.0 - gpu_q;

    sigc = incdf(gpu_p, gpu_q);

    return sigc;
  }
}

/** Main loop down call
 *
 * This will asses and call the correct templated kernel
 *
 * @param dimGrid
 * @param dimBlock
 * @param stream
 * @param batch
 */
__host__ void add_and_searchCU3(cudaStream_t stream, cuFFdotBatch* batch )
{
  const uint FLAGS = batch->flag ;

  if            ( (FLAGS & FLAG_CUFFT_CB_OUT) && (FLAGS & FLAG_SAS_TEX) && (FLAGS & FLAG_TEX_INTERP) )
  {
    fprintf(stderr,"ERROR: Invalid sum and search kernel. Line %i in %s\n", __LINE__, __FILE__ );
    exit(EXIT_FAILURE);
    //add_and_searchCU3_PT_f(stream, batch );
  }
  else
  {
    if      ( FLAGS & FLAG_SS_00 )
    {
      add_and_searchCU00(stream, batch );
    }
    else if ( FLAGS & FLAG_SS_10 )
    {
      add_and_searchCU31(stream, batch );
    }
    //    else if ( FLAGS & FLAG_SS_20 )
    //    {
    //      add_and_searchCU32(stream, batch );
    //    }
    //    else if ( FLAGS & FLAG_SS_30 )
    //    {
    //      add_and_searchCU33(stream, batch );
    //    }
    else
    {
      fprintf(stderr,"ERROR: Invalid sum and search kernel.\n");
      exit(EXIT_FAILURE);
    }
  }
}

int setConstVals( cuFFdotBatch* batch, int numharmstages, float *powcut, long long *numindep )
{
  int noHarms         = batch->sInf->noHarms;
  void *dcoeffs;

  FOLD // Calculate Y coefficients and copy to constant memory
  {
    if ( ((batch->hInfos->height + INDS_BUFF) * noHarms) > MAX_YINDS)
    {
      printf("ERROR! YINDS to small!");
    }

    freeNull(batch->sInf->yInds);
    batch->sInf->yInds    = (int*) malloc( (batch->hInfos->height + INDS_BUFF) * noHarms * sizeof(int));
    int *indsY            = batch->sInf->yInds;
    int bace              = 0;

    batch->hInfos->yInds  = 0;

    int zmax = batch->hInfos->zmax ;

    for (int ii = 0; ii < noHarms; ii++)
    {
      if ( ii == 0 )
      {
        for (int j = 0; j < batch->hInfos->height; j++)
        {
          indsY[bace + j] = j;
        }
      }
      else
      {
        float harmFrac  = HARM_FRAC_STAGE[ii];
        int sZmax;

        if ( batch->flag & FLAG_SS_INMEM )
        {
          sZmax = zmax;
        }
        else
        {
          int pidx  = batch->stageIdx[ii];
          sZmax = batch->hInfos[pidx].zmax;
          //calc_required_z(harmFrac, zmax);
        }

        for (int j = 0; j < batch->hInfos->height; j++)
        {
          int zz    = -zmax + j* ACCEL_DZ;
          int subz  = calc_required_z( harmFrac, zz );
          int zind  = index_from_z( subz, -sZmax );

          indsY[bace + j] = zind;
        }
      }

      if ( ii < batch->noHarms)
      {
        batch->hInfos[ii].yInds = bace;
      }

      bace += batch->hInfos->height;

      // Buffer with last value
      for (int j = 0; j < INDS_BUFF; j++)
      {
        indsY[bace + j] = indsY[bace + j-1];
      }

      bace += INDS_BUFF;
    }

    cudaGetSymbolAddress((void **)&dcoeffs, YINDS);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, indsY, bace*sizeof(int), cudaMemcpyHostToDevice),                      "Copying Y indices to device");
  }

  if ( powcut )
  {
    cudaGetSymbolAddress((void **)&dcoeffs, POWERCUT_STAGE);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, powcut, numharmstages * sizeof(float), cudaMemcpyHostToDevice),      "Copying power cutoff to device");
  }
  else
  {
    float pw[5];
    for ( int i = 0; i < 5; i++)
    {
      pw[i] = 0;
    }
    cudaGetSymbolAddress((void **)&dcoeffs, POWERCUT_STAGE);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &pw, 5 * sizeof(float), cudaMemcpyHostToDevice),         "Copying power cutoff to device");
  }

  if (numindep)
  {
    cudaGetSymbolAddress((void **)&dcoeffs, NUMINDEP_STAGE);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, numindep, numharmstages * sizeof(long long), cudaMemcpyHostToDevice),  "Copying stages to device");
  }
  else
  {
    long long numi[5];
    for ( int i = 0; i < 5; i++)
    {
      numi[i] = 0;
    }
    cudaGetSymbolAddress((void **)&dcoeffs, NUMINDEP_STAGE);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &numi, 5 * sizeof(long long), cudaMemcpyHostToDevice),      "Copying stages to device");

  }

  FOLD // Some other values  .
  {
    cudaMemcpyToSymbol(NO_STEPS, &(batch->noSteps), sizeof(int));
    cudaMemcpyToSymbol(ALEN, &(batch->accelLen), sizeof(int));

    if ( batch->flag & FLAG_SS_INMEM  )
    {
      cudaMemcpyToSymbol(PLN_START, &(batch->d_plainFull), sizeof(float*));
      cudaMemcpyToSymbol(PLN_STRIDE, &batch->sInf->mInf->inmemStride, sizeof(int));
    }
  }

  FOLD // Set other stage specific values  .
  {
    int height[MAX_HARM_NO];
    int stride[MAX_HARM_NO];
    int hwidth[MAX_HARM_NO];

    FOLD // Set values  .
    {
      for (int i = 0; i < batch->noHarms; i++)
      {
        int pidx  = batch->stageIdx[i];
        height[i] = batch->hInfos[pidx].height;
        stride[i] = batch->hInfos[pidx].width;
        hwidth[i] = batch->hInfos[pidx].halfWidth*ACCEL_NUMBETWEEN;
      }

      FOLD // The rest  .
      {
        int zeroHeight  = batch->hInfos->height;
        int zeroZMax    = batch->hInfos->zmax;

        for (int i = batch->noHarms; i < MAX_HARM_NO; i++)
        {
          float harmFrac  = HARM_FRAC_FAM[i];
          int zmax        = calc_required_z(harmFrac, zeroZMax);
          height[i]       = (zmax / ACCEL_DZ) * 2 + 1;
          stride[i]       = calc_fftlen3(harmFrac, zmax, batch->accelLen);
          hwidth[i]       = -1;
        }
      }
    }


    cudaGetSymbolAddress((void **)&dcoeffs, HEIGHT_STAGE);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &height, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, STRIDE_STAGE);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &stride, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, HWIDTH_STAGE);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &hwidth, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice),      "Copying stages to device");
  }

  CUDA_SAFE_CALL(cudaGetLastError(), "Error Preparing the constant memory.");

  return 1;
}

void SSKer(cuFFdotBatch* batch)
{
  if ( batch->haveConvData )
  {
    nvtxRangePush("Add & Search");

    FOLD // Do synchronisations  .
    {
      for (int ss = 0; ss < batch->noStacks; ss++)
      {
        cuFfdotStack* cStack = &batch->stacks[ss];

        if ( batch->flag & FLAG_SS_INMEM )
        {
          cudaStreamWaitEvent(batch->strmSearch, cStack->ifftMemComp, 0);
        }
        else
        {
          cudaStreamWaitEvent(batch->strmSearch, cStack->ifftComp, 0);
        }
      }
    }

    FOLD // Timing event  .
    {
#ifdef TIMING // Timing event
      CUDA_SAFE_CALL(cudaEventRecord(batch->searchInit,  batch->strmSearch),"Recording event: searchInit");
#endif
    }

#ifdef STPMSG
    printf("\t\tSum & search kernel\n");
#endif

    FOLD // Call the SS kernel  .
    {
      if ( batch->retType & CU_POWERZ_S )
      {
        if      ( batch->flag & FLAG_SS_STG )
        {
          add_and_searchCU3(batch->strmSearch, batch );
        }
        else if ( batch->flag & FLAG_SS_INMEM )
        {
          add_and_search_IMMEM(batch);
        }
        else
        {
          fprintf(stderr,"ERROR: function %s is not setup to handle this type of search.\n",__FUNCTION__);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        fprintf(stderr,"ERROR: function %s is not setup to handle this type of return data for GPU accel search\n",__FUNCTION__);
        exit(EXIT_FAILURE);
      }
      CUDA_SAFE_CALL(cudaGetLastError(), "Error at SSKer kernel launch");
    }

    FOLD // Synchronisation  .
    {
      CUDA_SAFE_CALL(cudaEventRecord(batch->searchComp,  batch->strmSearch),"Recording event: searchComp");
    }

    nvtxRangePop();
  }
}

int procesCanidate(cuFFdotBatch* batch, double rr, double zz, double poww, double sig, int stage, int numharm )
{
  if ( rr < batch->SrchSz->searchRHigh )
  {
    rr    /=  (double)numharm ;
    zz    =   ( zz * ACCEL_DZ - batch->hInfos[0].zmax ) / (double)numharm ;

    if      ( batch->cndType & CU_STR_LST     )
    {
      //*cands = insert_new_accelcand(*cands, poww, sig, numharm, rr, zz, &added);
    }
    else if ( batch->cndType & CU_STR_ARR     )
    {
      if ( !(batch->flag & FLAG_SIG_GPU) ) // Do the sigma calculation  .
      {
        sig     = candidate_sigma(poww, numharm, batch->sInf->numindep[stage]);
      }

      double  rDiff = rr - batch->SrchSz->searchRLow ;
      long    grIdx;   /// The index of the candidate in the global list

      if ( batch->flag & FLAG_STORE_EXP )
      {
        grIdx = floor(rDiff*ACCEL_RDR);
      }
      else
      {
        grIdx = floor(rDiff);
      }

      if ( grIdx >= 0 && grIdx < batch->SrchSz->noOutpR )  // Valid index  .
      {
        //batch->noResults++;

        if ( batch->flag & FLAG_STORE_ALL )               // Store all stages  .
        {
          grIdx += stage * (batch->SrchSz->noOutpR);      // Stride by size
        }

        if ( batch->cndType & CU_CANDFULL )
        {
          {
            cand* candidate = &((cand*)batch->h_candidates)[grIdx];

            // this sigma is greater than the current sigma for this r value
            if ( candidate->sig < sig )
            {
              if ( candidate->sig == 0 )
                batch->noResults++;

              candidate->sig      = sig;
              candidate->power    = poww;
              candidate->numharm  = numharm;
              candidate->r        = rr;
              candidate->z        = zz;
            }
          }
        }
        else
        {
          fprintf(stderr,"ERROR: function %s requires storing full candidates.\n",__FUNCTION__);
          exit(EXIT_FAILURE);
        }
      }
    }
    else if ( batch->cndType & CU_STR_QUAD    )
    {
      gridQuadTree<double, float>* qt = (gridQuadTree<double, float>*)(batch->h_candidates) ;

      quadPoint<double, float> voxel;
      voxel.position.x  = rr;
      voxel.position.y  = zz;
      voxel.value       = poww;

      qt->insertDynamic(voxel);

      quadNode<double, float>* head = qt->getHead();

      //                        if ( head->noEls > 4000 && poww > 20 )
      //                        {
      //                          int harmNo = 0;
      //
      //                          for ( int i = 1; i < 16; i++ )
      //                          {
      //                            vector2<double>    position;
      //
      //                            position.x = rr * h_FRAC_STAGE[i];
      //                            position.y = zz * h_FRAC_STAGE[i];
      //
      //                            float val;
      //                            int good = qt->get(position, &val);
      //
      //                            if ( good )
      //                            {
      //                              harmNo++;
      //                              //printf("%5.3f  ( %6.3f,  %6.3f)  %6.3f \n", h_FRAC_STAGE[i], position.x, position.y, val );
      //                            }
      //
      //                            int tmp = 0;
      //                          }
      //
      //                          if ( harmNo > 0 )
      //                          {
      //                            harmNo = 0;
      //
      //                            printf("----------------- \n");
      //                            for ( int i = 0; i < 16; i++ )
      //                            {
      //                              vector2<double>    position;
      //
      //                              position.x = rr * h_FRAC_STAGE[i];
      //                              position.y = zz * h_FRAC_STAGE[i];
      //
      //                              float val;
      //                              int good = qt->get(position, &val);
      //
      //                              if ( good )
      //                              {
      //                                harmNo++;
      //                                printf("%5.3f  ( %7.3f, %8.3f)  %6.3f \n", h_FRAC_STAGE[i], position.x, position.y, val );
      //                              }
      //
      //                              int tmp = 0;
      //                            }
      //                            printf("----------------- \n");
      //                          }
      //                        }
    }
    else
    {
      fprintf(stderr,"ERROR: function %s requires cand\n",__FUNCTION__);
      exit(EXIT_FAILURE);
    }
  }
}

void processSearchResults(cuFFdotBatch* batch)
{
  if ( batch->haveSearchResults )
  {
#ifdef STPMSG
    printf("\t\tProcess previous results\n");
#endif

    FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host
    {
#ifdef STPMSG
      printf("\t\t\tEvent Synchronise\n");
#endif

      nvtxRangePush("EventSynch");
      CUDA_SAFE_CALL(cudaEventSynchronize(batch->candCpyComp), "ERROR: copying result from device to host.");
      nvtxRangePop();
    }

#ifdef TIMING // Timing  .
    struct timeval start, end;
    gettimeofday(&start, NULL);
#endif

    nvtxRangePush("CPU Process results");

    double poww, sig;
    double rr, zz;
    int numharm;
    int noStages = batch->sInf->noHarmStages;

    FOLD // ADD candidates to global list  .
    {
#ifdef STPMSG
      printf("\t\t\tAdd To List\n");
#endif

#pragma omp critical
      FOLD // Critical section to handle candidates  .
      {
        int noSteps = batch->noSteps;
        int oStride = batch->strideRes;
        int idx;
        int x0, x1;
        int y0, y1;
        rVals* rVal;
        double minR;

        for ( int step = 0; step < noSteps; step++) // Loop over steps  .
        {
#ifdef SYNCHRONOUS
          rVal = &((*batch->rConvld)[step][0]);
#else
          rVal = &((*batch->rSearch)[step][0]);
#endif

          if ( batch->retType & CU_STR_PLN    )
          {
            // For plains have to consider halfwidth
            x0 = batch->hInfos->halfWidth*ACCEL_NUMBETWEEN;
            x1 = x0 + batch->accelLen ;

            y0 = 0;
            y1 = batch->hInfos->height;
          }
          else
          {
            // Other searches write results starting at the beginning
            x0 = 0;
            x1 = rVal->numrs;

            y0 = 0;
            y1 = batch->ssSlices;
          }

          if ( rVal->numrs )
          {
            if ( batch->retType & CU_STR_PLN    )
            {
              minR  = rVal->drlo - batch->hInfos->halfWidth ;
            }
            else
            {
              minR  = rVal->drlo;
            }

            for ( int stage = 0; stage < noStages; stage++ )
            {
              numharm = (1<<stage);
              float cutoff = batch->sInf->powerCut[stage];

              for ( int y = y0; y < y1; y++ )
              {
                for ( int x = x0; x < x1; x++ )
                {
                  poww      = 0;
                  sig       = 0;
                  zz        = 0;

                  if ( batch->retType & CU_STR_PLN    )
                  {
                    if      ( batch->flag & FLAG_ITLV_ROW )
                    {
                      idx = oStride*noSteps*y + oStride*step + x ;
                    }
                    else if ( batch->flag & FLAG_ITLV_PLN )
                    {
                      idx = step*oStride*y1 + oStride*y + x ;
                    }
                  }
                  else if (batch->flag & FLAG_SS_INMEM  )
                  {
                    idx = y*noStages*oStride + stage*oStride + x ;
                  }
                  else
                  {
                    idx = y*noSteps*noStages*oStride + step*noStages*oStride + stage*oStride + x ;
                  }

                  if      ( batch->retType & CU_CANDMIN  	  )
                  {
                    candMin candM         = ((candMin*)batch->h_retData)[idx];
                    if ( candM.power > poww )
                    {
                      sig                 = candM.power;
                      poww                = candM.power;
                      zz                  = candM.z;
                    }
                  }
                  else if ( batch->retType & CU_POWERZ_S   	)
                  {
                    candPZs candM         = ((candPZs*)batch->h_retData)[idx];

                    //                      int bin = minR * ACCEL_RDR + x ;
                    //                      if ( stage == 0 && bin == 20146 )
                    //                      {
                    //                        printf("%03i %10.5f\n", y, candM.value);
                    //                      }

                    if ( candM.value > poww )
                    {
                      sig                 = candM.value;
                      poww                = candM.value;
                      zz                  = candM.z;
                    }
                  }
                  else if ( batch->retType & CU_CANDBASC 		)
                  {
                    accelcandBasic candB  = ((accelcandBasic*)batch->h_retData)[idx];
                    if ( candB.sigma > poww )
                    {
                      poww                = candB.sigma;
                      sig                 = candB.sigma;
                      zz                  = candB.z;
                    }
                  }
                  else if ( batch->retType & CU_FLOAT    	  )
                  {
                    float val  = ((float*)batch->h_retData)[idx];

                    //                      int bin = minR * ACCEL_RDR + x ;
                    //                      if ( bin == 20146 )
                    //                      {
                    //                        printf("%03i %10.5f\n", y, val);
                    //                      }

                    if ( val > cutoff )
                    {
                      poww                = val;
                      sig                 = val;
                      zz                  = y;
                    }

                    poww = 0; // TMP
                  }
                  else if ( batch->retType & CU_HALF        )
                  {
                    float val  = half2float( ((ushort*)batch->h_retData)[idx] );

                    //                      if ( rVal->step == 0 && x == 1833 )
                    //                      {
                    //                        printf("%03i %10.5f\n", y, val);
                    //                      }

                    if ( val > cutoff )
                    {
                      poww                  = val;
                      sig                   = val;
                      zz                    = y;
                    }

                    poww = 0; // TMP

                  }
                  else
                  {
                    fprintf(stderr,"ERROR: function %s requires accelcandBasic\n",__FUNCTION__);
                    exit(EXIT_FAILURE);
                  }

                  if ( poww > 0 )
                  {
                    rr      = minR + x *  ACCEL_DR ;

                    //                      if ( rVal->step == 5 ) // TMP
                    //                      {
                    //                        int bin = minR * ACCEL_RDR + x ;
                    //
                    //                        printf("%i\t%i\t%.4f\t%.5f\n", stage, bin, rr, poww);
                    //                      }

                    //int in = batch->noResults;

                    procesCanidate(batch, rr, zz, poww, sig, stage, numharm ) ;

                    //int out = batch->noResults;

                    //                        if (in != out )
                    //                          cnts[stage]++;

                  }
                }
              }
            }

            //                printf("%03i %3i %4i %3i %3i %3i %3i %3i \n", rVal->step, inCnt, batch->noResults, cnts[0], cnts[1], cnts[2], cnts[3], cnts[4] ); // TMP
          }
        }
      }

#ifdef STPMSG
      printf("\t\t\tDone\n");
#endif

    }

    nvtxRangePop();

#ifdef TIMING // Timing  .
    gettimeofday(&end, NULL);
    float v1 =  ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec))*1e-3  ;
    batch->resultTime[0] += v1;
#endif

    FOLD // Synchronisation  .
    {
      CUDA_SAFE_CALL(cudaEventRecord(batch->processComp, batch->strmSearch),"Recording event: searchComp");
    }

    batch->haveSearchResults = 0;
  }
}

void getResults(cuFFdotBatch* batch)
{
  if ( batch->haveConvData )
  {
    FOLD // Do synchronisations  .
    {
      cudaStreamWaitEvent(batch->strmSearch, batch->searchComp,  0);
      cudaStreamWaitEvent(batch->strmSearch, batch->processComp, 0);
    }

#ifdef TIMING // Timing event  .
    CUDA_SAFE_CALL(cudaEventRecord(batch->candCpyInit,  batch->strmSearch),"Recording event: candCpyInit");
#endif

    FOLD // Copy relevant data back  .
    {
#ifdef STPMSG
      printf("\t\tCopy results from device to host\n");
#endif

      if      ( batch->retType & CU_STR_PLN )
      {
        if ( batch->flag & FLAG_CUFFT_CB_OUT )
          CUDA_SAFE_CALL(cudaMemcpyAsync(batch->h_retData, batch->d_plainPowers, batch->pwrDataSize, cudaMemcpyDeviceToHost, batch->strmSearch), "Failed to copy results back");
        else
          CUDA_SAFE_CALL(cudaMemcpyAsync(batch->h_retData, batch->d_plainData, batch->plnDataSize, cudaMemcpyDeviceToHost, batch->strmSearch), "Failed to copy results back");
      }
      else
      {
        CUDA_SAFE_CALL(cudaMemcpyAsync(batch->h_retData, batch->d_retData, batch->retDataSize, cudaMemcpyDeviceToHost, batch->strmSearch), "Failed to copy results back");
      }

      batch->haveConvData        = 0;
      batch->haveSearchResults   = 1;
    }

    CUDA_SAFE_CALL(cudaEventRecord(batch->candCpyComp, batch->strmSearch),"Recording event: readComp");
    CUDA_SAFE_CALL(cudaGetLastError(), "Copying results back from device.");
  }
}

void sumAndSearch(cuFFdotBatch* batch)
{
  if ( (batch->haveSearchResults || batch->haveConvData) ) // previous plain has data data so sum and search
  {
#ifdef STPMSG
    printf("\tSum & Search\n");
#endif

    FOLD // Sum and search the IFFT'd data  .
    {
      if      ( batch->retType & CU_STR_PLN )
      {
        // Nothing!
      }
      else if ( batch->flag & FLAG_SS_INMEM )
      {
        // NOTHING
      }
      else if ( batch->flag & FLAG_SS_CPU )
      {
        // NOTHING
      }
      else
      {
        SSKer(batch);
      }
    }

#ifdef SYNCHRONOUS

    FOLD // Copy results from device to host  .
    {
      if  ( batch->flag & FLAG_SS_INMEM )
      {
        // Nothing
      }
      else
      {
        getResults(batch);
      }
    }

    FOLD // Process previous results  .
    {
      if  ( batch->flag & FLAG_SS_INMEM )
      {
        // Nothing
      }
      else
      {
        processSearchResults(batch);
      }
    }

#else

    FOLD // Process previous results  .
    {
      if  ( batch->flag & FLAG_SS_INMEM )
      {
        // Nothing
      }
      else
      {
        processSearchResults(batch);
      }
    }

    FOLD // Copy results from device to host  .
    {
      if  ( batch->flag & FLAG_SS_INMEM )
      {
        // Nothing
      }
      else
      {
        getResults(batch);
      }
    }

#endif
  }

#ifdef TIMING // Timing  .

#ifndef SYNCHRONOUS
  if ( batch->haveSearchResults )
#endif
  {
    float time;         // Time in ms of the thing
    cudaError_t ret;    // Return status of cudaEventElapsedTime

    FOLD // Convolution timing  .
    {
      if ( !(batch->flag & FLAG_CUFFT_CB_IN) )
      {
        // Did the convolution by separate kernel

        if ( batch->flag & FLAG_MUL_BATCH )   // Convolution was done on the entire batch  .
        {
          FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host
          {
            nvtxRangePush("EventSynch");
            CUDA_SAFE_CALL(cudaEventSynchronize(batch->multComp), "ERROR: copying result from device to host.");
            nvtxRangePop();
          }

          ret = cudaEventElapsedTime(&time, batch->multInit, batch->multComp);
          if ( ret == cudaErrorNotReady )
          {
            //printf("Not ready\n");
          }
          else
          {
            //printf("    ready\n");
#pragma omp atomic
            batch->multTime[0] += time;
          }
        }
        else                                // Convolution was on a per stack basis  .
        {
          for (int ss = 0; ss < batch->noStacks; ss++)              // Loop through Stacks
          {
            cuFfdotStack* cStack = &batch->stacks[ss];

            FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host
            {
              nvtxRangePush("EventSynch");
              CUDA_SAFE_CALL(cudaEventSynchronize(cStack->multComp), "ERROR: copying result from device to host.");
              nvtxRangePop();
            }

            ret = cudaEventElapsedTime(&time, cStack->multInit, cStack->multComp);
            if ( ret == cudaErrorNotReady )
            {
              //printf("Not ready\n");
            }
            else
            {
              //printf("    ready\n");
#pragma omp atomic
              batch->multTime[ss] += time;
            }
          }
        }
      }

      CUDA_SAFE_CALL(cudaGetLastError(), "Convolution timing  .");
    }

    FOLD // Inverse FFT timing  .
    {
      for (int ss = 0; ss < batch->noStacks; ss++)
      {
        cuFfdotStack* cStack = &batch->stacks[ss];

        cudaError_t e1 = cudaEventQuery(cStack->ifftInit);
        cudaError_t e2 = cudaEventQuery(cStack->ifftComp);

        if ( ss == -1 )
        {
          printf("\n");
          if ( e1 == cudaSuccess )
          {
            printf(" e1 Good\n");
          }
          else
          {
            printf(" e1 Bad\n");
          }

          if ( e2 == cudaSuccess )
          {
            printf(" e2 Good\n");
          }
          else
          {
            printf(" e2 Bad\n");
          }
        }

        FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host
        {
          nvtxRangePush("EventSynch");
          CUDA_SAFE_CALL(cudaEventSynchronize(cStack->ifftComp), "ERROR: copying result from device to host.");
          nvtxRangePop();
        }

        ret = cudaEventElapsedTime(&time, cStack->ifftInit, cStack->ifftComp);
        if ( ret == cudaErrorNotReady )
        {
          //printf("Not ready\n");
        }
        else
        {
          //printf("    ready\n");
#pragma omp atomic
          batch->InvFFTTime[ss] += time;

          //if ( ss == 0 )
          //  printf("\nInvFFT: %f ms\n",time);
        }
      }
      CUDA_SAFE_CALL(cudaGetLastError(), "Inverse FFT timing");
    }

    FOLD // Search Timing  .
    {
      if ( !(batch->flag & FLAG_SS_CPU) && !(batch->flag & FLAG_SS_INMEM ) )
      {
        FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host
        {
          nvtxRangePush("EventSynch");
          CUDA_SAFE_CALL(cudaEventSynchronize(batch->searchComp), "ERROR: copying result from device to host.");
          nvtxRangePop();
        }

        ret = cudaEventElapsedTime(&time, batch->searchInit, batch->searchComp);

        if ( ret == cudaErrorNotReady )
        {
          //printf("Not ready\n");
        }
        else
        {
          //printf("    ready\n");
#pragma omp atomic
          batch->searchTime[0] += time;
        }

        CUDA_SAFE_CALL(cudaGetLastError(), "Search Timing");
      }
    }

    FOLD // Copy D2H  .
    {
      if ( !(batch->flag & FLAG_SS_INMEM ) )
      {
        FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host
        {
          nvtxRangePush("EventSynch");
          CUDA_SAFE_CALL(cudaEventSynchronize(batch->candCpyComp), "ERROR: copying result from device to host.");
          nvtxRangePop();
        }

        ret = cudaEventElapsedTime(&time, batch->candCpyInit, batch->candCpyComp);

        if ( ret == cudaErrorNotReady )
        {
          //printf("Not ready\n");
        }
        else
        {
          //printf("    ready\n");
#pragma omp atomic
          batch->copyD2HTime[0] += time;
        }

        CUDA_SAFE_CALL(cudaGetLastError(), "Copy D2H Timing");
      }
    }
  }
#endif
}

void sumAndMax(cuFFdotBatch* batch)
{
  //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  dim3 dimBlock, dimGrid;

  nvtxRangePush("Add & Max");

  if ( batch->haveSearchResults || batch->haveConvData ) // previous plain has data data so sum and search  .
  {
    int noStages = log(batch->noHarms)/log(2) + 1;

    FOLD // Do synchronisations  .
    {
      for (int ss = 0; ss< batch->noStacks; ss++)
      {
        cuFfdotStack* cStack = &batch->stacks[ss];

        cudaStreamWaitEvent(batch->strmSearch, cStack->ifftComp, 0);
      }
    }

    if ( batch->haveConvData ) // We have a convolved plain so call Sum & search  kernel .
    {
      FOLD // Call the main sum & search kernel
      {
        //        dimBlock.x  = SS3_X;
        //        dimBlock.y  = SS3_Y;
        //
        //        float bw    = SS3_X * SS3_Y;
        //        //float ww    = batch->batch[0].ffdotPowWidth[0] / ( bw );
        //        float ww    = batch->accelLen / ( bw );
        //
        //        dimGrid.x   = ceil(ww);
        //        dimGrid.y   = 1;
        //
        //        //add_and_maxCU31_f(dimGrid, dimBlock, 0, batch->strmSearch, searchList, (float*)batch->d_retData, batch->d_candSem, 0, pd, &batch->batch->rLow[0], batch->noSteps, batch->noHarmStages, batch->flag );
        //
        //        // Run message
        //        CUDA_SAFE_CALL(cudaGetLastError(), "Error at add_and_searchCU31 kernel launch");
        //
        //        CUDA_SAFE_CALL(cudaEventRecord(batch->searchComp,  batch->strmSearch),"Recording event: searchComp");
      }
    }

    if ( batch->haveSearchResults ) // Process previous results  .
    {
      FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host
      {
        nvtxRangePush("EventSynch");
        CUDA_SAFE_CALL(cudaEventSynchronize(batch->candCpyComp), "ERROR: copying result from device to host.");
        nvtxRangePop();
      }

      nvtxRangePush("CPU Process results");

      for ( int step = 0; step < batch->noSteps; step++ )
      {
        rVals* rVal = &((*batch->rInput)[step][0]);

        //int gIdx = batch->plains[0].searchRlowPrev[step] ;
        int gIdx = rVal->drlo;

        if ( batch->flag & FLAG_STORE_EXP )
          gIdx =  ( rVal->drlo ) * ACCEL_RDR ;

        float* gWrite = (float*)batch->h_candidates + gIdx;
        float* pRead = (float*)(batch->h_retData) + batch->hInfos->width*step;

        memcpy(gWrite, pRead, batch->accelLen*sizeof(float));
      }

      nvtxRangePop();

      // Do some Synchronisation
      CUDA_SAFE_CALL(cudaEventRecord(batch->processComp, batch->strmSearch),"Recording event: searchComp");

      batch->haveSearchResults = 0;
    }

    FOLD // Copy results from device to host  .
    {
      if ( batch->haveConvData )
      {
        cudaStreamWaitEvent(batch->strmSearch, batch->searchComp,  0);
        cudaStreamWaitEvent(batch->strmSearch, batch->processComp, 0);

        CUDA_SAFE_CALL(cudaMemcpyAsync(batch->h_retData, batch->d_retData, batch->retDataSize, cudaMemcpyDeviceToHost, batch->strmSearch), "Failed to copy results back");

        CUDA_SAFE_CALL(cudaEventRecord(batch->candCpyComp, batch->strmSearch),"Recording event: readComp");
        CUDA_SAFE_CALL(cudaGetLastError(), "Copying results back from device.");

        batch->haveConvData        = 0;
        batch->haveSearchResults   = 1;
      }
    }
  }

  nvtxRangePop();
}

void inMem(cuFFdotBatch* batch)
{
  long long noX = batch->accelLen * batch->SrchSz->noSteps ;
  int       noY = batch->hInfos->height;
  float*    pln = (float*)batch->h_candidates;

  //for ( int stage = 0; stage < batch->noHarmStages; stage++ )
  for ( int stage = 0; stage < 5 ; stage++ )
  {
    omp_set_num_threads(8);

#pragma omp parallel
    {
      int tid = omp_get_thread_num();

      printf("inMem tid %02i \n", tid);

#pragma omp for
      for ( int iy = 0; iy < noY; iy++ )
      {
        int y1 = iy       * noX   ;
        int y2 = (iy*0.5) * noX   ;

        for ( int ix = noX -1; ix >= 0; ix-- )
        {
          int idx1 = y1 +  ix ;
          int idx2 = y2 +  (ix*0.5) ;

          pln[idx1] += pln[idx2];
        }
      }
    }
  }
}

void inmemSumAndSearch(cuSearch* cuSrch)
{
  cuFFdotBatch* master  = &cuSrch->mInf->kernels[0];   // The first kernel created holds global variables
  uint startBin         = master->SrchSz->searchRLow * ACCEL_RDR;
  uint endBin           = startBin + cuSrch->SrchSz->noSteps * master->accelLen;

#ifndef DEBUG   // Parallel if we are not in debug mode  .
  omp_set_num_threads(cuSrch->mInf->noBatches);
#pragma omp parallel
#endif
  FOLD  //                              ---===== Main Loop =====---  .
  {
    int tid = omp_get_thread_num();
    cuFFdotBatch* batch = &cuSrch->mInf->batches[tid];

    setDevice(batch) ;

    uint firstBin = 0;
    uint len      = 0;

    while ( endBin > startBin )
    {
#pragma omp critical
      FOLD // Calculate the step  .
      {
        firstBin    = startBin;
        len         = MIN(batch->strideRes, endBin - firstBin) ;
        startBin   += len;
      }

      rVals* rVal   = &((*batch->rInput)[0][0]);
      rVal->drlo    = firstBin * ACCEL_DR;
      rVal->numrs   = len;

      FOLD //SS
      {
        add_and_search_IMMEM(batch);
      }

      FOLD // Process results
      {
        processSearchResults(batch);
      }

      FOLD //COPY
      {
        getResults(batch);
      }

      FOLD // Cycle r values
      {
        rVals*** rvals    = batch->rSearch;
        batch->rSearch = batch->rInput;
        batch->rInput  = rvals;
      }
    }

    FOLD // Process results
    {
      processSearchResults(batch);
    }
  }
}
