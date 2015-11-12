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
__device__ __constant__ int       HWIDTH_STAGE[MAX_HARM_NO];          ///< Plane half width in stage order

__device__ __constant__ void*     PLN_START;                          ///< A pointer to the start of the inmeme plane
__device__ __constant__ uint      PLN_STRIDE;                         ///< The strided in units of the inmeme plane
__device__ __constant__ int       NO_STEPS;                           ///< The number of steps used in the search
__device__ __constant__ int       ALEN;                               ///< CUDA copy of the accelLen used in the search

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
//    else
//    {
//      cdfgam_d(poww, numharm*2, &gpu_p, &gpu_q );
//    }

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
    cudaMemcpyToSymbol(NO_STEPS,  &(batch->noSteps),  sizeof(int) );
    cudaMemcpyToSymbol(ALEN,      &(batch->accelLen), sizeof(int) );

    if ( batch->flag & FLAG_SS_INMEM  )
    {
      cudaMemcpyToSymbol(PLN_START,   &(batch->d_planeFull),            sizeof(void*)  );
      cudaMemcpyToSymbol(PLN_STRIDE,  &batch->sInf->pInf->inmemStride,  sizeof(int)     );
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

  CUDA_SAFE_CALL(cudaGetLastError(), "Preparing the constant memory.");

  return 1;
}

void SSKer(cuFFdotBatch* batch)
{
  nvtxRangePush("S&S Ker");

  FOLD // Do synchronisations  .
  {
    for (int ss = 0; ss < batch->noStacks; ss++)
    {
      cuFfdotStack* cStack = &batch->stacks[ss];

      if ( batch->flag & FLAG_SS_INMEM )
      {
        cudaStreamWaitEvent(batch->srchStream, cStack->ifftMemComp, 0);
      }
      else
      {
        cudaStreamWaitEvent(batch->srchStream, cStack->ifftComp, 0);
      }
    }
  }

  FOLD // Timing event  .
  {
#ifdef TIMING // Timing event
    CUDA_SAFE_CALL(cudaEventRecord(batch->searchInit,  batch->srchStream),"Recording event: searchInit");
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
        add_and_searchCU3(batch->srchStream, batch );
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
    CUDA_SAFE_CALL(cudaGetLastError(), "At SSKer kernel launch");
  }

  FOLD // Synchronisation  .
  {
    CUDA_SAFE_CALL(cudaEventRecord(batch->searchComp,  batch->srchStream),"Recording event: searchComp");
  }

  nvtxRangePop();
}

/** Process an individual candidate  .
 *
 */
int procesCanidate(resultData* res, double rr, double zz, double poww, double sig, int stage, int numharm, FILE* myfile = NULL)
{
  // Adjust r and z for the number of harmonics
  rr    /=  (double)numharm ;
  zz    =   ( zz * ACCEL_DZ - res->zMax ) / (double)numharm ;

  if ( rr < res->SrchSz->searchRHigh )
  {
    if      ( res->cndType & CU_STR_LST     )
    {
      //*cands = insert_new_accelcand(*cands, poww, sig, numharm, rr, zz, &added);
    }
    else if ( res->cndType & CU_STR_ARR     )
    {
      if ( !(res->flag & FLAG_SIG_GPU) ) // Do the sigma calculation  .
      {
        sig     = candidate_sigma_cl(poww, numharm, res->numindep[stage]);
      }

      double  rDiff = rr - res->SrchSz->searchRLow ;
      long    grIdx;   /// The index of the candidate in the global list

      if ( res->flag & FLAG_STORE_EXP )
      {
        grIdx = floor(rDiff*ACCEL_RDR);
      }
      else
      {
        grIdx = floor(rDiff);
      }

      if ( grIdx >= 0 && grIdx < res->SrchSz->noOutpR )  // Valid index  .
      {
        if ( res->flag & FLAG_STORE_ALL )               // Store all stages  .
        {
          grIdx += stage * (res->SrchSz->noOutpR);      // Stride by size
        }

        if ( res->cndType & CU_CANDFULL )
        {
          cand* candidate = &((cand*)res->cndData)[grIdx];

          // this sigma is greater than the current sigma for this r value
          if ( candidate->sig < sig )
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

      if ( !(res->flag & FLAG_SIG_GPU) ) // Do the sigma calculation  .
      {
        sig     = candidate_sigma_cl(poww, numharm, res->numindep[stage]);
      }

      cand* candidate     = new cand;

      candidate->sig      = sig;
      candidate->power    = poww;
      candidate->numharm  = numharm;
      candidate->r        = rr;
      candidate->z        = zz;

      if ( myfile != NULL )
      {
        fprintf ( myfile, "%.15f %.15f %i %.15f %.15f \n", sig, poww, numharm, rr, zz );
      }

      (*res->noResults)++;

      qt->insert(candidate);
    }
    else
    {
      fprintf(stderr,"ERROR: Function %s unknown candidate storage type.\n", __FUNCTION__);
      exit(EXIT_FAILURE);
    }
  }

  return 0;
}

/** Process the results of the search this is usually run in a separate CPU thread  .
 *
 * This function is meant to be the entry of a separate thread
 *
 */
void* processSearchResults(void* ptr)
{
  resultData* res = (resultData*)ptr;

  //// Decrease the count number of running threads
  //sem_post(&res->threasdInfo->running_threads);

//  FILE * myfile;                                    // TMPS
//  myfile = fopen ( "/home/chris/src.cvs", "a+" );   // TMPS
//  fseek(myfile, 0, SEEK_END);                       // TMPS

#ifdef TIMING // Timing  .
  struct timeval start, end;
  gettimeofday(&start, NULL);
#endif

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
          sem_trywait(&res->threasdInfo->running_threads);
          exit(EXIT_FAILURE);
        }

        if ( poww > 0 )
        {
          // This value is was above the threshold
          rr      = res->rLow + x * ACCEL_DR ;
          //procesCanidate(res, rr, zz, poww, sig, stage, numharm, myfile ) ;
          procesCanidate(res, rr, zz, poww, sig, stage, numharm ) ;
        }
      }
    }
  }

#ifdef TIMING // Timing  .
  pthread_mutex_lock(&res->threasdInfo->candAdd_mutex);
  gettimeofday(&end, NULL);
  float v1 =  ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec))*1e-3  ;
  res->resultTime[0] += v1;
  pthread_mutex_unlock(&res->threasdInfo->candAdd_mutex);
#endif

  // Decrease the count number of running threads
  sem_trywait(&res->threasdInfo->running_threads);

  FOLD // Free memory
  {
    free (res->retData);
    free (res);
  }

  //fclose(myfile);                                   // TMPS

  return NULL;
}

/** Process the search results for the batch  .
 * This usually spawns a separate CPU thread to do the sigma calculations
 */
void processSearchResults(cuFFdotBatch* batch)
{
  if ( batch->rValues[0][0].numrs )
  {
#ifdef STPMSG
    printf("\t\tProcess previous results\n");
#endif

    FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host  .
    {
#ifdef STPMSG
      printf("\t\t\tEvent Synchronise\n");
#endif

      nvtxRangePush("EventSynch");
      CUDA_SAFE_CALL(cudaEventSynchronize(batch->candCpyComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
      nvtxRangePop();
    }

    nvtxRangePush("CPU Process results");

    FOLD // ADD candidates to global list  .
    {
#ifdef STPMSG
      printf("\t\t\tAdd To List\n");
#endif

      rVals* rVal = &batch->rValues[0][0];

      resultData* thrdDat = new resultData;     // A data structure to hold info for the thread processing the results
      memset(thrdDat, 0, sizeof(resultData) );

      // Allocate temporary memory to copy results back to
      nvtxRangePush("malloc");
      thrdDat->retData = (void*)malloc(batch->retDataSize);
      nvtxRangePop();

      // Copy data
      nvtxRangePush("memcpy");
      if ( batch->flag & FLAG_SS_INMEM )
        memcpy(thrdDat->retData, batch->h_retData2, batch->retDataSize);
      else
        memcpy(thrdDat->retData, batch->h_retData1, batch->retDataSize);
      nvtxRangePop();

      FOLD // Synchronisation  .
      {
        // This will allow kernels to run while the CPU continues
        CUDA_SAFE_CALL(cudaEventRecord(batch->processComp, batch->srchStream),"Recording event: searchComp");
      }

      thrdDat->SrchSz       = batch->SrchSz;
      thrdDat->cndData      = batch->h_candidates;
      thrdDat->cndType      = batch->cndType;
      thrdDat->noStages     = batch->sInf->noHarmStages;
      thrdDat->numindep     = batch->sInf->numindep;
      thrdDat->powerCut     = batch->sInf->powerCut;
      thrdDat->rLow         = rVal->drlo;
      thrdDat->retType      = batch->retType;
      thrdDat->threasdInfo  = batch->sInf->threasdInfo;
      thrdDat->flag         = batch->flag;
      thrdDat->zMax         = batch->hInfos->zmax;
      thrdDat->resultTime   = batch->resultTime;
      thrdDat->noResults    = &batch->noResults;

      thrdDat->x0           = 0;
      thrdDat->x1           = 0;
      thrdDat->y0           = 0;
      thrdDat->y1           = batch->ssSlices;

      thrdDat->xStride      = batch->strideRes;
      thrdDat->yStride      = batch->ssSlices;

      if ( !(batch->flag & FLAG_SS_INMEM) )
      {
        thrdDat->xStride   *= batch->noSteps;
      }

      for ( int step = 0; step < batch->noSteps; step++) // Loop over steps  .
      {
        rVals* rVal         = &batch->rValues[step][0];
        thrdDat->x1        += rVal->numrs;
      }

      if ( thrdDat->x1 > thrdDat->xStride )
      {
        fprintf(stderr,"ERROR: Number of elements of greater than stride. In function %s  \n",__FUNCTION__);
        exit(EXIT_FAILURE);
      }

      // Increase the count number of running threads
      sem_post(&batch->sInf->threasdInfo->running_threads);

#ifndef SYNCHRONOUS
      if ( batch->flag & FLAG_THREAD ) 	// Create thread  .
      {
        pthread_t thread;
        int  iret1 = pthread_create( &thread, NULL, processSearchResults, (void*) thrdDat);

        if (iret1)
        {
          fprintf(stderr,"Error - pthread_create() return code: %d\n", iret1);
          exit(EXIT_FAILURE);
        }
      }
      else                              // Just call the function  .
#endif
      {
        processSearchResults( (void*) thrdDat );
      }
    }

    nvtxRangePop();
  }
}

void getResults(cuFFdotBatch* batch)
{
  if ( batch->rValues[0][0].numrs )
  {
    FOLD // Synchronisations  .
    {
      cudaStreamWaitEvent(batch->resStream, batch->searchComp,  0);
      cudaStreamWaitEvent(batch->resStream, batch->processComp, 0);
    }

#ifdef TIMING // Timing event  .
    CUDA_SAFE_CALL(cudaEventRecord(batch->candCpyInit,  batch->srchStream),"Recording event: candCpyInit");
#endif

    FOLD // Copy relevant data back  .
    {
#ifdef STPMSG
      printf("\t\tCopy results from device to host\n");
#endif

      if      ( batch->retType & CU_STR_PLN )
      {
        CUDA_SAFE_CALL(cudaMemcpyAsync(batch->h_retData1, batch->d_planePowr, batch->pwrDataSize, cudaMemcpyDeviceToHost, batch->resStream), "Failed to copy results back");
      }
      else
      {
        CUDA_SAFE_CALL(cudaMemcpyAsync(batch->h_retData1, batch->d_retData1, batch->retDataSize, cudaMemcpyDeviceToHost, batch->resStream), "Failed to copy results back");
      }

      CUDA_SAFE_CALL(cudaGetLastError(), "Copying results back from device.");
    }

    FOLD // Synchronisations  .
    {
      CUDA_SAFE_CALL(cudaEventRecord(batch->candCpyComp, batch->resStream),"Recording event: readComp");
    }
  }
}

void sumAndSearch(cuFFdotBatch* batch)        // Function to call to SS and process data in normal steps  .
{
  // Sum and search the IFFT'd data  .
  if ( batch->rValues[0][0].numrs )
  {
#ifdef STPMSG
    printf("\tSum & Search\n");
#endif

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
}

void sumAndSearchOrr(cuFFdotBatch* batch)     // Function to call to SS and process data in normal steps  .
{
  FOLD // Sum and search the IFFT'd data  .
  {
#ifdef STPMSG
    printf("\tSum & Search\n");
#endif

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

void sumAndMax(cuFFdotBatch* batch)
{
//  //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
//
//  dim3 dimBlock, dimGrid;
//
//  nvtxRangePush("Add & Max");
//
//  if ( (batch->state & HAVE_SS) || (batch->state & HAVE_MULT) ) // previous plane has data data so sum and search  .
//  {
//    int noStages = log(batch->noHarms)/log(2) + 1;
//
//    FOLD // Do synchronisations  .
//    {
//      for (int ss = 0; ss< batch->noStacks; ss++)
//      {
//        cuFfdotStack* cStack = &batch->stacks[ss];
//
//        cudaStreamWaitEvent(batch->strmSearch, cStack->ifftComp, 0);
//      }
//    }
//
//    if ( batch->state & HAVE_MULT ) // We have a convolved plane so call Sum & search  kernel .
//    {
//      FOLD // Call the main sum & search kernel
//      {
//        //        dimBlock.x  = SS3_X;
//        //        dimBlock.y  = SS3_Y;
//        //
//        //        float bw    = SS3_X * SS3_Y;
//        //        //float ww    = batch->batch[0].ffdotPowWidth[0] / ( bw );
//        //        float ww    = batch->accelLen / ( bw );
//        //
//        //        dimGrid.x   = ceil(ww);
//        //        dimGrid.y   = 1;
//        //
//        //        //add_and_maxCU31_f(dimGrid, dimBlock, 0, batch->strmSearch, searchList, (float*)batch->d_retData, batch->d_candSem, 0, pd, &batch->batch->rLow[0], batch->noSteps, batch->noHarmStages, batch->flag );
//        //
//        //        // Run message
//        //        CUDA_SAFE_CALL(cudaGetLastError(), "At add_and_searchCU31 kernel launch");
//        //
//        //        CUDA_SAFE_CALL(cudaEventRecord(batch->searchComp,  batch->strmSearch),"Recording event: searchComp");
//      }
//    }
//
//    if ( (batch->state & HAVE_SS) ) // Process previous results  .
//    {
//      FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host
//      {
//        nvtxRangePush("EventSynch");
//        CUDA_SAFE_CALL(cudaEventSynchronize(batch->candCpyComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
//        nvtxRangePop();
//      }
//
//      nvtxRangePush("CPU Process results");
//
//      for ( int step = 0; step < batch->noSteps; step++ )
//      {
//        //rVals* rVal = &((*batch->rInput)[step][0]);
//        rVals* rVal = &batch->rArrays[3][step][0];
//
//        //int gIdx = batch->planes[0].searchRlowPrev[step] ;
//        int gIdx = rVal->drlo;
//
//        if ( batch->flag & FLAG_STORE_EXP )
//          gIdx =  ( rVal->drlo ) * ACCEL_RDR ;
//
//        float* gWrite = (float*)batch->h_candidates + gIdx;
//        float* pRead = (float*)(batch->h_retData) + batch->hInfos->width*step;
//
//        memcpy(gWrite, pRead, batch->accelLen*sizeof(float));
//      }
//
//      nvtxRangePop();
//
//      // Do some Synchronisation
//      CUDA_SAFE_CALL(cudaEventRecord(batch->processComp, batch->strmSearch),"Recording event: searchComp");
//
//      batch->state &= ~HAVE_SS;
//    }
//
//    FOLD // Copy results from device to host  .
//    {
//      if ( (batch->state & HAVE_MULT) )
//      {
//        cudaStreamWaitEvent(batch->strmSearch, batch->searchComp,  0);
//        cudaStreamWaitEvent(batch->strmSearch, batch->processComp, 0);
//
//        CUDA_SAFE_CALL(cudaMemcpyAsync(batch->h_retData, batch->d_retData, batch->retDataSize, cudaMemcpyDeviceToHost, batch->strmSearch), "Failed to copy results back");
//
//        CUDA_SAFE_CALL(cudaEventRecord(batch->candCpyComp, batch->strmSearch),"Recording event: readComp");
//        CUDA_SAFE_CALL(cudaGetLastError(), "Copying results back from device.");
//
//        batch->state &= ~HAVE_MULT;
//        batch->state |=  HAVE_SS;
//      }
//    }
//  }
//
//  nvtxRangePop();
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

void inmemSS(cuFFdotBatch* batch, double drlo, int len)
{
  setActiveBatch(batch, 0);

  FOLD // Set the r values for this step  .
  {
    rVals* rVal   = &batch->rValues[0][0];
    rVal->drlo    = drlo;
    rVal->numrs   = len;
  }

#ifdef SYNCHRONOUS

    add_and_search_IMMEM(batch);

    getResults(batch);

    processSearchResults(batch);

#else

    setActiveBatch(batch, 0);
    add_and_search_IMMEM(batch);

    setActiveBatch(batch, 1);
    processSearchResults(batch);

    setActiveBatch(batch, 0);
    getResults(batch);

#endif

  // Cycle r values
  cycleRlists(batch);

  // Cycle r output
  cycleOutput(batch);
}

void inmemSumAndSearch(cuSearch* cuSrch)
{
  cuFFdotBatch* master  = &cuSrch->pInf->kernels[0];   // The first kernel created holds global variables
  uint startBin         = master->SrchSz->searchRLow * ACCEL_RDR;
  uint endBin           = startBin + cuSrch->SrchSz->noSteps * master->accelLen;
  float totaBinsl       = endBin - startBin ;

#ifndef DEBUG   // Parallel if we are not in debug mode  .
  omp_set_num_threads(cuSrch->pInf->noBatches);
#pragma omp parallel
#endif
  FOLD  //                              ---===== Main Loop =====---  .
  {
    int tid = omp_get_thread_num();
    cuFFdotBatch* batch = &cuSrch->pInf->batches[tid];

    setDevice(batch->device) ;

    uint firstBin = 0;
    uint len      = 0;

    FOLD // Set all r-values to zero  .
    {
      for ( int rIdx = 0; rIdx < batch->noRArryas; rIdx++ )
      {
        for ( int step = 0; step < batch->noSteps; step++ )
        {
          for ( int harm = 0; harm < batch->noHarms; harm++ )
          {
            rVals* rVal = &batch->rArrays[rIdx][step][harm];
            memset(rVal, 0, sizeof(rVals) );
          }
        }
      }
    }

    while ( endBin > startBin )
    {

#pragma omp critical
      FOLD // Calculate the step  .
      {
        firstBin    = startBin;
        len         = MIN(batch->strideRes, endBin - firstBin) ;
        startBin   += len;
      }

      inmemSS(batch, firstBin * ACCEL_DR, len);

#pragma omp critical
      FOLD // Output
      {
        int noTrd;
        sem_getvalue(&master->sInf->threasdInfo->running_threads, &noTrd );
        printf("\rSearching  in-mem GPU plane. %5.1f%% ( %3i Active CPU threads processing found candidates)  ", (totaBinsl-endBin+startBin)/totaBinsl*100.0, noTrd );
        fflush(stdout);
      }

    }

    for ( int step= 0 ; step < batch->noRArryas; step++ )
    {
      inmemSS(batch, 0, 0);
    }
  }

  printf("\rSearching  in-mem GPU plane. %5.1f%%                                                                                    \n\n", 100.0 );

  //printf("Searching Done\n");

  FOLD // Wait for all processing threads to terminate
  {
    waitForThreads(&master->sInf->threasdInfo->running_threads, "Waiting for CPU thread(s) to finish processing returned from the GPU.", 200 );
  }
}
