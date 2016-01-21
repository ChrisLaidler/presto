#include "cuda_accel_SS.h"

#include <semaphore.h>

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

__device__ __constant__ int       YINDS[MAX_YINDS];                   ///< The harmonic related Y index for each plane
__device__ __constant__ float     POWERCUT_STAGE[MAX_HARM_NO];        ///<
__device__ __constant__ float     NUMINDEP_STAGE[MAX_HARM_NO];        ///<

__device__ __constant__ int       HEIGHT_STAGE[MAX_HARM_NO];          ///< Plane heights in stage order
__device__ __constant__ int       STRIDE_STAGE[MAX_HARM_NO];          ///< Plane strides in stage order
__device__ __constant__ int       PSTART_STAGE[MAX_HARM_NO];          ///< Start offset of good points in a plane, stage order

__device__ __constant__ void*     PLN_START;                          ///< A pointer to the start of the in-mem plane
__device__ __constant__ uint      PLN_STRIDE;                         ///< The strided in units of the in-mem plane
__device__ __constant__ int       NO_STEPS;                           ///< The number of steps used in the search  -  NB: this is specific to the batch not the search, but its only used in the inmem search!
__device__ __constant__ int       ALEN;                               ///< CUDA copy of the accelLen used in the search

//====================================== Constant variables  ===============================================\\

__device__ const float FRAC_STAGE[16]     =  { 1.0000f, 0.5000f, 0.7500f, 0.2500f, 0.8750f, 0.6250f, 0.3750f, 0.1250f, 0.9375f, 0.8125f, 0.6875f, 0.5625f, 0.4375f, 0.3125f, 0.1875f, 0.0625f } ;

//__device__ const float FRAC_STAGE[16]     =  { 1.0000f, 0.5000f, 0.2500f, 0.7500f, 0.1250f, 0.3750f, 0.6250f, 0.8750f, 0.0625f, 0.1875f, 0.3125f, 0.4375f, 0.5625f, 0.6875f, 0.8125f, 0.9375f } ;

__device__ const float FRAC_HARM[16]      =  { 1.0f, 0.9375f, 0.875f, 0.8125f, 0.75f, 0.6875f, 0.625f, 0.5625f, 0.5f, 0.4375f, 0.375f, 0.3125f, 0.25f, 0.1875f, 0.125f, 0.0625f } ;
__device__ const short STAGE[5][2]        =  { {0,0}, {1,1}, {2,3}, {4,7}, {8,15} } ;
__device__ const short CHUNKSZE[5]        =  { 4, 8, 8, 8, 8 } ;

//======================================= Global variables  ================================================\\

//int    inMemSrchSz = 32768; // Deprecated

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

template<int64_t FLAGS>
__device__ inline int getY(int planeY, const int noSteps,  const int step, const int planeHeight = 0 )
{
  // Calculate y indice from interleave method
  if      ( FLAGS & FLAG_ITLV_ROW )
  {
    return planeY * noSteps + step;
  }
  else
  {
    return planeY + planeHeight*step;
  }
}

template<int64_t FLAGS>
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
    if      (numharm==1)
      cdfgam_d<1>(poww, &gpu_p, &gpu_q );
    else if (numharm==2)
      cdfgam_d<2>(poww, &gpu_p, &gpu_q );
    else if (numharm==4)
      cdfgam_d<4>(poww, &gpu_p, &gpu_q );
    else if (numharm==8)
      cdfgam_d<8>(poww, &gpu_p, &gpu_q );
    else if (numharm==16)
      cdfgam_d<16>(poww, &gpu_p, &gpu_q );

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
  const int64_t FLAGS = batch->flags ;

  if            ( (FLAGS & FLAG_CUFFT_CB_POW) && (FLAGS & FLAG_SAS_TEX) && (FLAGS & FLAG_TEX_INTERP) )
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
    //		Depricated
    //
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
  void *dcoeffs;

  FOLD // Calculate Y coefficients and copy to constant memory  .
  {
    int noHarms         = batch->sInf->noSrchHarms;

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

        if ( batch->flags & FLAG_SS_INMEM )
        {
          sZmax = zmax;
        }
        else
        {
          int sIdx  = batch->sInf->sIdx[ii];
          sZmax = batch->hInfos[sIdx].zmax;
        }

        for (int j = 0; j < batch->hInfos->height; j++)
        {
          int zz    = -zmax + j* ACCEL_DZ;
          int subz  = calc_required_z( harmFrac, zz );
          int zind  = index_from_z( subz, -sZmax );

          indsY[bace + j] = zind;
        }
      }

      if ( ii < batch->noSrchHarms)
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
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, indsY, bace*sizeof(int), cudaMemcpyHostToDevice, batch->stacks->initStream),                      "Copying Y indices to device");
  }

  FOLD // copy power cutoff values  .
  {
    if ( powcut )
    {
      cudaGetSymbolAddress((void **)&dcoeffs, POWERCUT_STAGE);
      CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, powcut, numharmstages * sizeof(float), cudaMemcpyHostToDevice, batch->stacks->initStream),      "Copying power cutoff to device");
    }
    else
    {
      float pw[5];
      for ( int i = 0; i < 5; i++)
      {
        pw[i] = 0;
      }
      cudaGetSymbolAddress((void **)&dcoeffs, POWERCUT_STAGE);
      CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, &pw, 5 * sizeof(float), cudaMemcpyHostToDevice, batch->stacks->initStream),         "Copying power cutoff to device");
    }
  }

  FOLD // number of independent values  .
  {
    if (numindep)
    {
      cudaGetSymbolAddress((void **)&dcoeffs, NUMINDEP_STAGE);
      CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, numindep, numharmstages * sizeof(long long), cudaMemcpyHostToDevice, batch->stacks->initStream),  "Copying stages to device");
    }
    else
    {
      long long numi[5];
      for ( int i = 0; i < 5; i++)
      {
        numi[i] = 0;
      }
      cudaGetSymbolAddress((void **)&dcoeffs, NUMINDEP_STAGE);
      CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, &numi, 5 * sizeof(long long), cudaMemcpyHostToDevice, batch->stacks->initStream),      "Copying stages to device");

    }
  }

  FOLD // Some other values  .
  {
    cudaGetSymbolAddress((void **)&dcoeffs, NO_STEPS);
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs,  &(batch->noSteps),  sizeof(int), cudaMemcpyHostToDevice, batch->stacks->initStream),  "Copying number of steps");

    cudaGetSymbolAddress((void **)&dcoeffs, ALEN);
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs,  &(batch->accelLen), sizeof(int), cudaMemcpyHostToDevice, batch->stacks->initStream),  "Copying accelLen");
  }

  FOLD // In-mem plane details  .
  {
    if ( batch->flags & FLAG_SS_INMEM  )
    {
      cudaGetSymbolAddress((void **)&dcoeffs, PLN_START);
      CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, &(batch->sInf->d_planeFull),  sizeof(void*),  cudaMemcpyHostToDevice, batch->stacks->initStream),  "Copying accelLen");

      cudaGetSymbolAddress((void **)&dcoeffs, PLN_STRIDE);
      CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, &(batch->sInf->inmemStride),  sizeof(uint),   cudaMemcpyHostToDevice, batch->stacks->initStream),  "Copying accelLen");
    }
  }

  FOLD // Set other stage specific values  .
  {
    int height[MAX_HARM_NO];
    int stride[MAX_HARM_NO];
    int pStart[MAX_HARM_NO];

    FOLD // Set values  .
    {
      for (int i = 0; i < batch->noGenHarms; i++)
      {
        int sIdx  = batch->sInf->sIdx[i];
        height[i] = batch->hInfos[sIdx].height;
        stride[i] = batch->hInfos[sIdx].width;
        pStart[i] = batch->hInfos[sIdx].kerStart;
      }

      FOLD // The rest  .
      {
        int zeroZMax    = batch->hInfos->zmax;

        presto_interp_acc accuracy = LOWACC;
        if ( batch->flags & FLAG_KER_HIGH )
          accuracy = HIGHACC;

        for (int i = batch->noGenHarms; i < MAX_HARM_NO; i++)
        {
          float harmFrac  = HARM_FRAC_FAM[i];
          int zmax        = calc_required_z(harmFrac, zeroZMax);
          height[i]       = (zmax / ACCEL_DZ) * 2 + 1;
          stride[i]       = calc_fftlen3(harmFrac, zmax, batch->accelLen, accuracy);
          pStart[i]       = -1;
        }
      }
    }

    cudaGetSymbolAddress((void **)&dcoeffs, HEIGHT_STAGE);
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, &height, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice, batch->stacks->initStream),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, STRIDE_STAGE);
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, &stride, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice, batch->stacks->initStream),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, PSTART_STAGE);
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, &pStart, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice, batch->stacks->initStream),      "Copying stages to device");
  }

  CUDA_SAFE_CALL(cudaGetLastError(), "Preparing the constant memory.");

  return (1);
}

void SSKer(cuFFdotBatch* batch)
{
  infoMSG(2,3,"Sum & Search\n");

  nvtxRangePush("S&S Ker");

  FOLD // Do synchronisations  .
  {
    infoMSG(3,4,"pre synchronisations\n");

    for (int ss = 0; ss < batch->noStacks; ss++)
    {
      cuFfdotStack* cStack = &batch->stacks[ss];

      if ( batch->flags & FLAG_SS_INMEM )
      {
        cudaStreamWaitEvent(batch->srchStream, cStack->ifftMemComp,   0);
      }
      else
      {
        cudaStreamWaitEvent(batch->srchStream, cStack->ifftComp,      0);
      }
    }
  }

  FOLD // Timing event  .
  {
    if ( batch->flags & FLAG_TIME ) // Timing event
    {
      CUDA_SAFE_CALL(cudaEventRecord(batch->searchInit,  batch->srchStream),"Recording event: searchInit");
    }
  }



  FOLD // Call the SS kernel  .
  {
    infoMSG(3,4,"kernel\n");

    if ( batch->retType & CU_POWERZ_S )
    {
      if      ( batch->flags & FLAG_SS_STG )
      {
        add_and_searchCU3(batch->srchStream, batch );
      }
      else if ( batch->flags & FLAG_SS_INMEM )
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
    CUDA_SAFE_CALL(cudaGetLastError(), "At SSKer kernel launch");
  }

  FOLD // Synchronisation  .
  {
    infoMSG(3,4,"post synchronisations\n");

    CUDA_SAFE_CALL(cudaEventRecord(batch->searchComp,  batch->srchStream),"Recording event: searchComp");
  }

  nvtxRangePop();
}

/** Process an individual candidate  .
 *
 */
int procesCanidate(resultData* res, double rr, double zz, double poww, double sig, int stage, int numharm)
{
  // Adjust r and z for the number of harmonics
  rr    /=  (double)numharm ;
  zz    =   ( zz * ACCEL_DZ - res->zMax ) / (double)numharm ;

  if ( rr < res->SrchSz->searchRHigh )
  {
    if ( !(res->flags & FLAG_SIG_GPU) ) // Do the sigma calculation  .
    {
      sig     = candidate_sigma_cl(poww, numharm, res->numindep[stage]);
    }

    if      ( res->cndType & CU_STR_LST     )
    {
      GSList *candsGPU  = (GSList*)res->cndData;
      int     added     = 0;

      if ( res->threasdInfo )
      {
        // Thread safe
        pthread_mutex_lock(&res->threasdInfo->candAdd_mutex);
        res->cndData        = insert_new_accelcand(candsGPU, poww, sig, numharm, rr, zz, &added );
        (*res->noResults)++;
        pthread_mutex_unlock(&res->threasdInfo->candAdd_mutex);
      }
      else
      {
        res->cndData        = insert_new_accelcand(candsGPU, poww, sig, numharm, rr, zz, &added );
        (*res->noResults)++;
      }
    }
    else if ( res->cndType & CU_STR_ARR     )
    {
      double  rDiff = rr - res->SrchSz->searchRLow ;
      long    grIdx;   /// The index of the candidate in the global list

      if ( res->flags & FLAG_STORE_EXP )
      {
        grIdx = floor(rDiff*ACCEL_RDR);
      }
      else
      {
        grIdx = floor(rDiff);
      }

      if ( grIdx >= 0 && grIdx < res->SrchSz->noOutpR )  // Valid index  .
      {
        if ( res->flags & FLAG_STORE_ALL )               // Store all stages  .
        {
          grIdx += stage * (res->SrchSz->noOutpR);      // Stride by size
        }

        if ( res->cndType & CU_CANDFULL )
        {
          cand* candidate = &((cand*)res->cndData)[grIdx];

          // this sigma is greater than the current sigma for this r value
          if ( candidate->sig < sig )
          {
            if ( res->threasdInfo )
            {
              pthread_mutex_lock(&res->threasdInfo->candAdd_mutex);
              if ( candidate->sig < sig )
              {
                if ( candidate->sig == 0 )
                  (*res->noResults)++;

                candidate->sig      = sig;
                candidate->power    = poww;
                candidate->numharm  = numharm;
                candidate->r        = rr;
                candidate->z        = zz;
              }
              pthread_mutex_unlock(&res->threasdInfo->candAdd_mutex);
            }
            else
            {
              if ( candidate->sig == 0 )
                (*res->noResults)++;

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
    else if ( res->cndType & CU_STR_QUAD    )
    {
      candTree* qt = (candTree*)res->cndData;

      cand* candidate     = new cand;

      candidate->sig      = sig;
      candidate->power    = poww;
      candidate->numharm  = numharm;
      candidate->r        = rr;
      candidate->z        = zz;

      (*res->noResults)++;

      qt->insert(candidate);
    }
    else
    {
      fprintf(stderr,"ERROR: Function %s unknown candidate storage type.\n", __FUNCTION__);
      exit(EXIT_FAILURE);
    }
  }

  return (0);
}

/** Process the results of the search this is usually run in a separate CPU thread  .
 *
 * This function is meant to be the entry of a separate thread
 *
 */
void* processSearchResults(void* ptr)
{
  resultData* res = (resultData*)ptr;
  struct timeval start, end;      // Timing variables

  if ( res->flags & FLAG_TIME ) 		// Timing  .
  {
    gettimeofday(&start, NULL);
  }

  double poww, sig;
  double rr, zz;
  int numharm;
  int idx;

  for ( int stage = 0; stage < res->noStages; stage++ )
  {
    numharm       = (1<<stage);
    float cutoff  = res->powerCut[stage];

    for ( int y = res->y0; y < res->y1; y++ )
    {
      for ( int x = res->x0; x < res->x1; x++ )
      {
        poww      = 0;
        sig       = 0;
        zz        = 0;

        idx = stage*res->xStride*res->yStride + y*res->xStride + x ;

        if      ( res->retType & CU_CANDMIN     )
        {
          candMin candM         = ((candMin*)res->retData)[idx];

          if ( candM.power > poww )
          {
            sig                 = candM.power;
            poww                = candM.power;
            zz                  = candM.z;
          }
        }
        else if ( res->retType & CU_POWERZ_S    )
        {
          candPZs candM         = ((candPZs*)res->retData)[idx];

          if ( candM.value > poww )
          {
            sig                 = candM.value;
            poww                = candM.value;
            zz                  = candM.z;
          }
        }
        else if ( res->retType & CU_CANDBASC    )
        {
          accelcandBasic candB  = ((accelcandBasic*)res->retData)[idx];

          if ( candB.sigma > poww )
          {
            poww                = candB.sigma;
            sig                 = candB.sigma;
            zz                  = candB.z;
          }
        }
        else if ( res->retType & CU_FLOAT       )
        {
          float val  = ((float*)res->retData)[idx];

          if ( val > cutoff )
          {
            poww                = val;
            sig                 = val;
            zz                  = y;
          }
        }
        else if ( res->retType & CU_HALF        )
        {
          float val  = half2float( ((ushort*)res->retData)[idx] );

          if ( val > cutoff )
          {
            poww                  = val;
            sig                   = val;
            zz                    = y;
          }
        }
        else
        {
          fprintf(stderr,"ERROR: function %s requires accelcandBasic\n",__FUNCTION__);
          if ( res->threasdInfo )
          {
            sem_trywait(&res->threasdInfo->running_threads);
          }
          exit(EXIT_FAILURE);
        }

        if ( poww > 0 )
        {
          // This value is above the threshold
          rr      = res->rLow + x * ACCEL_DR ;
          procesCanidate(res, rr, zz, poww, sig, stage, numharm ) ;
        }
      }
    }
  }

  if ( res->flags & FLAG_TIME ) // Timing  .
  {
    gettimeofday(&end, NULL);
    float time =  ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec))*1e-3  ;

    if ( res->threasdInfo )
    {
      pthread_mutex_lock(&res->threasdInfo->candAdd_mutex);
      res->resultTime[0] += time;
      pthread_mutex_unlock(&res->threasdInfo->candAdd_mutex);
    }
    else
    {
      res->resultTime[0] += time;
    }
  }

  // Decrease the count number of running threads
  if ( res->threasdInfo )
  {
    sem_trywait(&res->threasdInfo->running_threads);
  }

  FOLD // Free memory
  {
    if ( res->flags & FLAG_THREAD )
      free (res->retData);
    free (res);
  }

  return (NULL);
}

/** Process the search results for the batch  .
 * This usually spawns a separate CPU thread to do the sigma calculations
 */
void processSearchResults(cuFFdotBatch* batch)
{
  if ( (*batch->rAraays)[batch->rActive][0][0].numrs )
  {
    struct timeval start, end;          // Timing variables
    resultData* thrdDat;

    infoMSG(1,2,"Process previous results\n");

    nvtxRangePush("CPU Process results");

    if ( batch->flags & FLAG_TIME )    // Timing  .
    {
      gettimeofday(&start, NULL);
    }

    FOLD // Allocate temporary memory to copy results back to  .
    {
      nvtxRangePush("malloc");

      thrdDat = new resultData;     // A data structure to hold info for the thread processing the results
      memset(thrdDat, 0, sizeof(resultData) );

      if ( batch->flags & FLAG_THREAD )
      {
        thrdDat->retData = (void*)malloc(batch->retDataSize);
      }

      nvtxRangePop();
    }

    FOLD // Initialise data structure  .
    {
      rVals* rVal = &(*batch->rAraays)[batch->rActive][0][0];

      infoMSG(3,3,"Initialise data structure\n");

      thrdDat->SrchSz       = batch->sInf->SrchSz;
      thrdDat->cndData      = batch->sInf->h_candidates;
      thrdDat->cndType      = batch->cndType;
      thrdDat->noStages     = batch->sInf->noHarmStages;
      thrdDat->numindep     = batch->sInf->numindep;
      thrdDat->powerCut     = batch->sInf->powerCut;
      thrdDat->rLow         = rVal->drlo;
      thrdDat->retType      = batch->retType;
      thrdDat->threasdInfo  = batch->sInf->threasdInfo;
      thrdDat->flags         = batch->flags;
      thrdDat->zMax         = batch->hInfos->zmax;
      thrdDat->resultTime   = batch->resultTime;
      thrdDat->noResults    = &batch->noResults;

      thrdDat->x0           = 0;
      thrdDat->x1           = 0;
      thrdDat->y0           = 0;
      thrdDat->y1           = batch->ssSlices;

      thrdDat->xStride      = batch->strideOut;
      thrdDat->yStride      = batch->ssSlices;

      if ( !(batch->flags & FLAG_THREAD) && (batch->flags & FLAG_SYNCH ) )
      {
        thrdDat->threasdInfo = NULL;
      }

      if ( !(batch->flags & FLAG_SS_INMEM) )
      {
        // Multi-step

        thrdDat->xStride    *= batch->noSteps;

        for ( int step = 0; step < batch->noSteps; step++) // Loop over steps  .
        {
          rVals* rVal       = &(*batch->rAraays)[batch->rActive][step][0];
          thrdDat->x1       += rVal->numrs;                 // These should all be Acelllen but there may be the case of the last step!
        }
      }
      else
      {
        // NB: In-mem has only one step
        thrdDat->x1         = rVal->numrs;
      }

      if ( thrdDat->x1 > thrdDat->xStride )
      {
        fprintf(stderr,"ERROR: Number of elements of greater than stride. In function %s  \n",__FUNCTION__);
        exit(EXIT_FAILURE);
      }
    }

    FOLD // Timing 0  .
    {
      if ( batch->flags & FLAG_TIME )
      {
        gettimeofday(&end, NULL);
        float time = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec))*1e-3  ;
        int idx = MIN(2, batch->noStacks-1);

        pthread_mutex_lock(&batch->sInf->threasdInfo->candAdd_mutex);
        batch->resultTime[idx] += time;
        pthread_mutex_unlock(&batch->sInf->threasdInfo->candAdd_mutex);
      }
    }

    FOLD // Copy data from device  .
    {
      FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host  .
      {
        infoMSG(3,4,"pre synchronisation [blocking] candCpyComp\n");

        nvtxRangePush("EventSynch");
        CUDA_SAFE_CALL(cudaEventSynchronize(batch->candCpyComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
        nvtxRangePop();
      }

      FOLD // Timing  .
      {
        if ( batch->flags & FLAG_TIME )
        {
          gettimeofday(&start, NULL);
        }
      }

      FOLD // Copy data  .
      {
        infoMSG(3,3,"copy to temporary memory\n");

        nvtxRangePush("memcpy");

        void *gpuOutput;

        if ( !(batch->flags & FLAG_SYNCH) && (batch->flags & FLAG_SS_INMEM) )
        {
          gpuOutput = batch->h_outData2;
        }
        else
        {
          gpuOutput = batch->h_outData1;
        }

        if ( batch->flags & FLAG_THREAD )
        {
          memcpy(thrdDat->retData, gpuOutput, batch->retDataSize);

          FOLD // Synchronisation  .
          {
            infoMSG(3,4,"synchronise\n");

            // This will allow kernels to run while the CPU continues
            CUDA_SAFE_CALL(cudaEventRecord(batch->processComp, batch->srchStream),"Recording event: processComp");
          }
        }
        else
        {
          thrdDat->retData = gpuOutput;
        }

        nvtxRangePop();
      }

      FOLD // Timing 1  .
      {
        if ( batch->flags & FLAG_TIME )
        {
          gettimeofday(&end, NULL);
          float time =  ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec))*1e-3  ;
          int idx = MIN(1, batch->noStacks-1);

          pthread_mutex_lock(&batch->sInf->threasdInfo->candAdd_mutex);
          batch->resultTime[idx] += time;
          pthread_mutex_unlock(&batch->sInf->threasdInfo->candAdd_mutex);
        }
      }
    }

    FOLD // ADD candidates to global list potently in a separate thread  .
    {
      if ( thrdDat->threasdInfo )// Increase the count number of running threads, processSearchResults will decrease it when its finished
        sem_post(&batch->sInf->threasdInfo->running_threads);

      if ( batch->flags & FLAG_SYNCH )
      {
        nvtxRangePush("Thread");
      }

      if ( batch->flags & FLAG_THREAD ) 	// Create thread  .
      {
        infoMSG(3,4,"create thread\n");

        pthread_t thread;
        int  iret1 = pthread_create( &thread, NULL, processSearchResults, (void*) thrdDat);

        if (iret1)
        {
          fprintf(stderr,"Error - pthread_create() return code: %d\n", iret1);
          exit(EXIT_FAILURE);
        }

        if ( batch->flags & FLAG_SYNCH )
        {
          void *status;
          if ( pthread_join(thread, &status) )
          {
            fprintf(stderr,"ERROR: Failed to join results thread.\n");
            exit(EXIT_FAILURE);
          }
        }
      }
      else                              	// Just call the function  .
      {
        infoMSG(3,4,"non thread\n");

        processSearchResults( (void*) thrdDat );

        if ( !(batch->flags & FLAG_THREAD) )
        {
          // Not using threading so using original memory location

          FOLD // Synchronisation  .
          {
            infoMSG(3,4,"synchronise\n");

            // This will allow kernels to run while the CPU continues
            CUDA_SAFE_CALL(cudaEventRecord(batch->processComp, batch->srchStream),"Recording event: processComp");
          }
        }
      }

      if ( batch->flags & FLAG_SYNCH )
      {
        nvtxRangePop();
      }
    }

    nvtxRangePop();
  }
}

void getResults(cuFFdotBatch* batch)
{
  // Timing
  if ( batch->flags & FLAG_TIME )
  {
    if ( (*batch->rAraays)[batch->rActive+1][0][0].numrs )
    {
      // Sum & Search kernel
      timeEvents( batch->candCpyInit, batch->candCpyComp, &batch->copyD2HTime[0],   "Copy device to host");
    }
  }

  if ( (*batch->rAraays)[batch->rActive][0][0].numrs )
  {
    infoMSG(1,2,"Copy results from device to host\n");

    FOLD // Synchronisations  .
    {
      infoMSG(3,3,"pre synchronise\n");

      CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->resStream, batch->searchComp,  0),"Waiting on event searchComp");
      CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->resStream, batch->processComp, 0),"Waiting on event processComp");
    }

    FOLD // Timing event  .
    {
      if ( batch->flags & FLAG_TIME )
      {
        CUDA_SAFE_CALL(cudaEventRecord(batch->candCpyInit,  batch->srchStream),"Recording event: candCpyInit");
      }
    }

    FOLD // Copy relevant data back  .
    {
      infoMSG(3,3,"Async memcpy\n");

      if      ( batch->retType & CU_STR_PLN )
      {
        CUDA_SAFE_CALL(cudaMemcpyAsync(batch->h_outData1, batch->d_planePowr, batch->pwrDataSize, cudaMemcpyDeviceToHost, batch->resStream), "Failed to copy results back");
      }
      else
      {
        CUDA_SAFE_CALL(cudaMemcpyAsync(batch->h_outData1, batch->d_outData1,  batch->retDataSize, cudaMemcpyDeviceToHost, batch->resStream), "Failed to copy results back");
      }

      CUDA_SAFE_CALL(cudaGetLastError(), "Copying results back from device.");
    }

    FOLD // Synchronisations  .
    {
      infoMSG(3,3,"post synchronise\n");

      CUDA_SAFE_CALL(cudaEventRecord(batch->candCpyComp, batch->resStream),"Recording event: readComp");
    }

    CUDA_SAFE_CALL(cudaGetLastError(), "Leaving getResults.");
  }
}

void sumAndSearch(cuFFdotBatch* batch)        // Function to call to SS and process data in normal steps  .
{
  // Timing
  if ( batch->flags & FLAG_TIME )
  {
    if ( (*batch->rAraays)[batch->rActive+1][0][0].numrs )
    {
      // Sum & Search kernel
      timeEvents( batch->searchInit, batch->searchComp, &batch->searchTime[0],   "Sum & Search");
    }
  }

  // Sum and search the IFFT'd data  .
  if ( (*batch->rAraays)[batch->rActive][0][0].numrs )
  {
    infoMSG(1,2,"Sum & Search\n");

    if      ( batch->retType 	& CU_STR_PLN 	  )
    {
      // Nothing!
    }
    else if ( batch->flags    & FLAG_SS_INMEM )
    {
      // NOTHING
    }
    else if ( batch->flags    & FLAG_SS_CPU   )
    {
      // NOTHING
    }
    else
    {
      SSKer(batch);
    }
  }
}

void sumAndSearchOrr(cuFFdotBatch* batch)     // Function to call to SS and process data in normal steps  .
{
  FOLD // Sum and search the IFFT'd data  .
  {
    infoMSG(2,1,"Sum & Search\n");

    if      ( batch->retType & CU_STR_PLN )
    {
      // Nothing!
    }
    else if ( batch->flags & FLAG_SS_INMEM )
    {
      // NOTHING
    }
    else if ( batch->flags & FLAG_SS_CPU )
    {
      // NOTHING
    }
    else
    {
      SSKer(batch);
    }
  }

  if ( batch->flags & FLAG_SYNCH )
  {
    FOLD // Copy results from device to host  .
    {
      if  ( batch->flags & FLAG_SS_INMEM )
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
      if  ( batch->flags & FLAG_SS_INMEM )
      {
        // Nothing
      }
      else
      {
        processSearchResults(batch);
      }
    }
  }
  else
  {
    FOLD // Process previous results  .
    {
      if  ( batch->flags & FLAG_SS_INMEM )
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
      if  ( batch->flags & FLAG_SS_INMEM )
      {
        // Nothing
      }
      else
      {
        getResults(batch);
      }
    }
  }
}

void sumAndMax(cuFFdotBatch* batch)
{
  // TODO write this
}

void inMem(cuFFdotBatch* batch)
{
  long long noX = batch->accelLen * batch->sInf->SrchSz->noSteps ;
  int       noY = batch->hInfos->height;
  float*    pln = (float*)batch->sInf->h_candidates;

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

void inmemSS(cuFFdotBatch* batch, double drlo, int len)
{
  infoMSG(1,2,"Inmem Search\n");

  setActiveBatch(batch, 0);
  setSearchRVals(batch, drlo, len);

//  if ( msgLevel >= 3 )
//  {
//    for ( int i = 0 ; i < batch->noRArryas; i++ )
//    {
//      rVals* rVal = &(*batch->rAraays)[i][0][0];
//
//      printf("%i  step: %03i  r-low: %8.1f  numrs: %06ld\n", i, rVal->step, rVal->drlo, rVal->numrs );
//    }
//  }

  if ( batch->flags & FLAG_SYNCH )
  {
    add_and_search_IMMEM(batch);

    getResults(batch);

    processSearchResults(batch);
  }
  else
  {
    setActiveBatch(batch, 0);
    add_and_search_IMMEM(batch);

    setActiveBatch(batch, 1);
    processSearchResults(batch);

    setActiveBatch(batch, 0);
    getResults(batch);
  }
  
  // Cycle r values
  cycleRlists(batch);
  setActiveBatch(batch, 1);

  // Cycle candidate output
  cycleOutput(batch);
}

void inmemSumAndSearch(cuSearch* cuSrch)
{
  infoMSG(1,2,"Inmem Sum And Search\n");

  cuFFdotBatch* master  = &cuSrch->pInf->kernels[0];   // The first kernel created holds global variables
  uint startBin         = cuSrch->SrchSz->searchRLow * ACCEL_RDR;
  uint endBin           = startBin + cuSrch->SrchSz->noSteps * master->accelLen;
  float totaBinsl       = endBin - startBin ;
  int iteration         = 0;
  uint currentBin       = startBin;

  nvtxRangePush("Inmem Search");

  FOLD // Set all r-values to zero  .
  {
    for ( int bIdx = 0; bIdx < cuSrch->pInf->noBatches; bIdx++ )
    {
      cuFFdotBatch* batch = &cuSrch->pInf->batches[bIdx];

      for ( int rIdx = 0; rIdx < batch->noRArryas; rIdx++ )
      {
        for ( int step = 0; step < batch->noSteps; step++ )
        {
          for ( int harm = 0; harm < batch->noGenHarms; harm++ )
          {
            rVals* rVal = &(*batch->rAraays)[rIdx][step][harm];
            memset(rVal, 0, sizeof(rVals) );

            rVal->step = -1;
          }
        }
      }
    }
  }

#ifndef DEBUG   // Parallel if we are not in debug mode  .

  if ( cuSrch->sSpec->flags & FLAG_SYNCH )
  {
    omp_set_num_threads(1);
  }
  else
  {
    omp_set_num_threads(cuSrch->pInf->noBatches);
  }

#pragma omp parallel
#endif
  FOLD  //                              ---===== Main Loop =====---  .
  {
    int tid = omp_get_thread_num();
    cuFFdotBatch* batch = &cuSrch->pInf->batches[tid];

    setDevice(batch->device) ;

    uint firstBin = 0;
    uint len      = 0;

    while ( currentBin < endBin )
    {
#pragma omp critical
      FOLD // Calculate the step  .
      {
        FOLD  // Synchronous behaviour  .
        {
#ifndef  DEBUG
          if ( cuSrch->sSpec->flags & FLAG_SYNCH )
#endif
          {
            // If running in synchronous mode use multiple batches, just synchronously
            tid     = iteration % cuSrch->pInf->noBatches ;
            batch   = &cuSrch->pInf->batches[tid];
            setDevice(batch->device) ;
          }
        }

        iteration++;

        int step    = (currentBin-startBin)/batch->strideOut;
        firstBin    = currentBin;
        len         = MIN(batch->strideOut, endBin - firstBin) ;
        currentBin += len;
        rVals* rVal = &(*batch->rAraays)[0][0][0];
        rVal->step  = step;

        if ( msgLevel >= 1 )
        {
          int tot  = (endBin)/batch->strideOut;

          infoMSG(1,1,"\nStep %4i of %4i thread %02i processing %02i steps on GPU %i\n", step+1, tot, tid, 1, batch->device );
        }
      }

      inmemSS(batch, firstBin * ACCEL_DR, len);

#pragma omp critical
      FOLD // Output  .
      {
        if ( msgLevel == 0  )
        {
          int noTrd;
          sem_getvalue(&master->sInf->threasdInfo->running_threads, &noTrd );
          printf("\rSearching  in-mem GPU plane. %5.1f%% ( %3i Active CPU threads processing found candidates)  ", (totaBinsl-endBin+currentBin)/totaBinsl*100.0, noTrd );
          fflush(stdout);
        }
        else
        {

        }
      }

    }

    for ( int step= 0 ; step < batch->noRArryas; step++ )
    {
      inmemSS(batch, 0, 0);
    }
  }

  printf("\rSearching  in-mem GPU plane. %5.1f%%                                                                                    \n\n", 100.0 );

  FOLD // Wait for all processing threads to terminate
  {
    waitForThreads(&master->sInf->threasdInfo->running_threads, "Waiting for CPU thread(s) to finish processing returned from the GPU.", 200 );
  }

  nvtxRangePop();
}
