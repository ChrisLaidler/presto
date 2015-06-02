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

__device__ __constant__ int        YINDS[MAX_YINDS];
//__device__ __constant__ float      YINDS_F[MAX_YINDS];
__device__ __constant__ float      POWERCUT[MAX_HARM_NO];
__device__ __constant__ float      NUMINDEP[MAX_HARM_NO];

__device__ __constant__ int        HEIGHT[MAX_HARM_NO];         ///< Plain heights in stage order
__device__ __constant__ int        STRIDE[MAX_HARM_NO];         ///< Plain strides in stage order
__device__ __constant__ int        HWIDTH[MAX_HARM_NO];         ///< Plain half width in stage order

//====================================== Constant variables  ===============================================\\

__device__ const float FRAC[16]      =  {1.0f, 0.5f, 0.25f, 0.75f, 0.125f, 0.375f, 0.625f, 0.875f, 0.0625f, 0.1875f, 0.3125f, 0.4375f, 0.5625f, 0.6875f, 0.8125f, 0.9375f } ;
__device__ const int   STAGE[5][2]   =  { {0,0}, {1,1}, {2,3}, {4,7}, {8,15} } ;
__device__ const int   CHUNKSZE[5]   =  { 4, 8, 8, 8, 8 } ;


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

/** Calculate the CDF of a gamma distribution
 */
template<int n>
__host__ __device__ void cdfgam_d(double x, double *p, double* q)
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
  for ( int i = 0; i < 5 ; i++ ) // Note: only doing 5 recursions this could be pushed up
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
  if ( poww > 100)
  {
    /*
    double c[] = { \
        -7.784894002430293e-03, \
        -3.223964580411365e-01, \
        -2.400758277161838e+00, \
        -2.549732539343734e+00, \
        4.374664141464968e+00,  \
        2.938163982698783e+00 };

    double d[] = { \
        7.784695709041462e-03, \
        3.224671290700398e-01, \
        2.445134137142996e+00, \
        3.754408661907416e+00 };
*/
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

    //logP = log(1-exp(logQ));

    logQ += log( (double)numindep );

    double l = sqrt(-2.0*logQ);

    //double x = -1.0 * (((((c[1]*l+c[2])*l+c[3])*l+c[4])*l+c[5])*l+c[6]) / ((((d[1]*l+d[2])*l+d[3])*l+d[4])*l+1.0);
    double x = l - ( 2.515517 + l * (0.802853 + l * 0.010328) ) / ( 1.0 + l * (1.432788 + l * (0.189269 + l * 0.001308)) ) ;

    //return logQ;
    return x;
  }
  else
  {
    double gpu_p, gpu_q, sigc ;

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

    if (gpu_p == 1.0)
      gpu_q *= numindep;
    else
    {
      gpu_q = 1.0 - pow(gpu_p, (double)numindep);
      //pp = pow((1.0-gpu_q),1.0/(double)numindep);
    }
    gpu_p = 1.0 - gpu_q;

    sigc = incdf(gpu_p, gpu_q);

    //return gpu_q;
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
__host__ void add_and_searchCU3(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const uint FLAGS = batch->flag ;

  if            ( (FLAGS & FLAG_MUL_CB_OUT) && (FLAGS & FLAG_SAS_TEX) && (FLAGS & FLAG_TEX_INTERP) )
  {
    add_and_searchCU3_PT_f(dimGrid, dimBlock, stream, batch );
  }
  else
  {
    add_and_searchCU311_f(dimGrid, dimBlock, stream, batch );
  }
}

int setConstVals( cuFFdotBatch* batch, int numharmstages, float *powcut, long long *numindep )
{
  int noHarms         = (1 << (numharmstages - 1) );
  void *dcoeffs;

  FOLD // Calculate Y coefficients and copy to constant memory
  {
    if (batch->hInfos[0].height * (noHarms /*-1*/ ) > MAX_YINDS)
    {
      printf("ERROR! YINDS to small!");
    }
    int *indsY    = (int*) malloc(batch->hInfos[0].height * noHarms * sizeof(int));
    int bace      = 0;
    batch->hInfos[0].yInds = 0;
    for (int ii = 0; ii< batch->noHarms; ii++)
    {
      if ( ii == 0 )
      {
        for (int j = 0; j< batch->hInfos[0].height; j++)
        {
          indsY[bace + j] = j;
        }
      }
      else
      {
        int idx = batch->pIdx[ii];

        for (int j = 0; j< batch->hInfos[0].height; j++)
        {
          int zz    = -batch->hInfos[0].zmax + j* ACCEL_DZ;
          int subz  = calc_required_z(batch->hInfos[idx].harmFrac, zz);
          int zind  = index_from_z(subz, -batch->hInfos[idx].zmax);
          if (zind < 0 || zind >= batch->hInfos[idx].height)
          {
            printf("ERROR! YINDS Wrong!");
          }
          indsY[bace + j] = zind;
        }
      }
      batch->hInfos[ii].yInds = bace;
      bace += batch->hInfos[0].height;
    }

    cudaGetSymbolAddress((void **)&dcoeffs, YINDS);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, indsY, bace*sizeof(int), cudaMemcpyHostToDevice),                      "Copying Y indices to device");
  }

  if ( powcut )
  {
    cudaGetSymbolAddress((void **)&dcoeffs, POWERCUT);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, powcut, numharmstages * sizeof(float), cudaMemcpyHostToDevice),      "Copying power cutoff to device");
  }
  else
  {
    float pw[numharmstages];
    for ( int i = 0; i < numharmstages; i++)
    {
      pw[i] = 0;
    }
    cudaGetSymbolAddress((void **)&dcoeffs, POWERCUT);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &pw, numharmstages * sizeof(float), cudaMemcpyHostToDevice),         "Copying power cutoff to device");
  }

  if (numindep)
  {
    cudaGetSymbolAddress((void **)&dcoeffs, NUMINDEP);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, numindep, numharmstages * sizeof(long long), cudaMemcpyHostToDevice),  "Copying stages to device");
  }
  else
  {
    long long numi[numharmstages];
    for ( int i = 0; i < numharmstages; i++)
    {
      numi[i] = 0;
    }
    cudaGetSymbolAddress((void **)&dcoeffs, NUMINDEP);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &numi, numharmstages * sizeof(long long), cudaMemcpyHostToDevice),      "Copying stages to device");

  }

  FOLD // Set other constant values
  {
    int height[MAX_HARM_NO];
    int stride[MAX_HARM_NO];
    int hwidth[MAX_HARM_NO];

    for (int i = 0; i < batch->noHarms; i++)
    {
      int pidx = batch->pIdx[i];

      height[i] = batch->hInfos[pidx].height;
      stride[i] = batch->hInfos[pidx].width;
      hwidth[i] = batch->hInfos[pidx].halfWidth*ACCEL_NUMBETWEEN;
    }

    for (int i = batch->noHarms; i < MAX_HARM_NO; i++) // Zero the rest
    {
      height[i] = 0;
      stride[i] = 0;
      hwidth[i] = 0;
    }

    cudaGetSymbolAddress((void **)&dcoeffs, HEIGHT);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &height, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, STRIDE);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &stride, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, HWIDTH);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &hwidth, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice),      "Copying stages to device");
  }

  CUDA_SAFE_CALL(cudaGetLastError(), "Error Preparing the constant memory.");

  return 1;
}

void SSKer(cuFFdotBatch* batch, long long *numindep)
{
  if ( batch->haveConvData )
  {
#ifdef STPMSG
    printf("\t\tSum & search kernel\n");
#endif
    dim3 dimBlock, dimGrid;

    int noStages = log(batch->noHarms)/log(2) + 1;

    FOLD // Timing event  .
    {
#ifdef TIMING // Timing event
      CUDA_SAFE_CALL(cudaEventRecord(batch->searchInit,  batch->strmSearch),"Recording event: searchInit");
#endif
    }

    FOLD // Call the SS kernel  .
    {
      dimBlock.x  = SS3_X;
      dimBlock.y  = SS3_Y;

      float bw    = SS3_X * SS3_Y;
      float ww    = batch->accelLen / ( bw );

      dimGrid.x   = ceil(ww);
      dimGrid.y   = 1;

      if ( batch->retType & CU_SMALCAND )
      {
        //add_and_searchCU31_f(dimGrid, dimBlock, 0, batch->strmSearch, searchList, (accelcandBasic*)batch->d_retData, batch->d_candSem, 0, pd, &batch->batch->rLow[0], batch->noSteps, batch->noHarmStages, batch->flag );
        //add_and_searchCU311_f(dimGrid, dimBlock, batch->strmSearch, batch );
        //if ( (batch->flag&FLAG_MUL_CB_OUT) && (batch->flag&FLAG_SAS_TEX) )
        {
          add_and_searchCU3(dimGrid, dimBlock, batch->strmSearch, batch );
        }
      }
      else
      {
        fprintf(stderr,"ERROR: function %s is not setup to handle this type of return data for GPU accel search\n",__FUNCTION__);
        exit(EXIT_FAILURE);
      }
      CUDA_SAFE_CALL(cudaGetLastError(), "Error at add_and_searchCU31 kernel launch");
    }

    FOLD // Synchronisation  .
    {
      CUDA_SAFE_CALL(cudaEventRecord(batch->searchComp,  batch->strmSearch),"Recording event: searchComp");
    }
  }
}

void processSearchResults(cuFFdotBatch* batch, long long *numindep)
{
  if ( batch->haveSearchResults )
  {
    // A blocking synchronisation to ensure results are ready to be proceeded by the host
    CUDA_SAFE_CALL(cudaEventSynchronize(batch->candCpyComp), "ERROR: copying result from device to host.");

    int noStages = log(batch->noHarms)/log(2) + 1;

#ifdef TIMING // Timing  .
    struct timeval start, end;
    gettimeofday(&start, NULL);
#endif

    nvtxRangePush("CPU Process results");

#ifdef STPMSG
    printf("\t\tProcess previous results\n");
#endif

    batch->noResults = 0;

    double poww, sig;
    double rr, zz;
    int numharm;
    poww = 0;

    FOLD  // Loop over results and find candidates  .
    {
      //float* powers = NULL;
      //powers = (float*)malloc(batch->noSteps*noStages*batch->accelLen);
      //memset(powers, 0, batch->noSteps*noStages*batch->accelLen);

      // TODO: We could loop over values and remove non local maxima to reuse sigma calculations

      Fout // Calculate sigma vales from power  .
      {
        // We don't want to do the expensive sigma calculations in the critical block ...
        if ( !(batch->flag & FLAG_SIG_GPU) )
        {
          nvtxRangePush("CPU sigma calculations");
#ifdef STPMSG
          printf("\t\t\tCalculate sigma\n");
#endif

          if ( batch->retType & CU_SMALCAND )
          {
            //powers = (float*)malloc(batch->noSteps*noStages*batch->accelLen);
          }

          for ( int step = 0; step < batch->noSteps; step++)         // Loop over steps  .
          {
            for ( int stage = 0; stage < noStages; stage++ )
            {
              for ( int x = 0; x < batch->accelLen; x++ )
              {
                int idx   = step*noStages*batch->hInfos->width + stage*batch->hInfos->width + x ;

                if ( batch->retType & CU_SMALCAND )
                {
                  accelcandBasic candB  = ((accelcandBasic*)batch->h_retData)[idx] ;
                  poww                  = candB.sigma ;

                  if ( poww > 0 )
                  {
                    //powers[step*noStages*batch->accelLen + stage*batch->accelLen + x ] = poww ;

                    numharm     = (1<<stage);
                    candB.sigma = candidate_sigma(poww, numharm, numindep[stage]);
                  }
                }
                else if ( batch->retType & CU_FULLCAND )
                {
                  cand candd  = ((cand*)batch->h_retData)[idx] ;
                  poww        = candd.power;

                  if ( poww > 0 )
                  {
                    numharm   = (1<<stage);
                    candd.sig = candidate_sigma(poww, numharm, numindep[stage]);
                  }
                }
                else
                {
                  fprintf(stderr,"ERROR: function %s requires accelcandBasic or cand\n",__FUNCTION__);
                  exit(1);
                }
              }
            }
          }

          nvtxRangePop();
        }
      }

      FOLD // ADD candidates to global list  .
      {

#ifdef STPMSG
        printf("\t\t\tAdd To List\n");
#endif

#pragma omp critical
        {
          for ( int step = 0; step < batch->noSteps; step++) // Loop over steps  .
          {
            rVals* rVal = &((*batch->rSearch)[step][0]);

            for ( int stage = 0; stage < noStages; stage++ )
            {
              for ( int x = 0; x < batch->accelLen; x++ )
              {
                int idx   = step*noStages*batch->hInfos->width + stage*batch->hInfos->width + x ;

                if ( batch->retType & CU_SMALCAND )
                {
                  accelcandBasic candB  = ((accelcandBasic*)batch->h_retData)[idx] ;
                  sig                   = candB.sigma ;

                  if ( sig > 0 )
                  {
                    batch->noResults++;
                    numharm = (1<<stage);

                    if ( !(batch->flag & FLAG_SIG_GPU) )
                    {
                      sig   = candidate_sigma(sig, numharm, numindep[stage]);
                      poww  = candB.sigma;
                    }
                    else
                    {
                      poww  = candB.sigma;
                    }

                    if ( !(batch->flag & FLAG_SIG_GPU) )
                    {
                      //poww    = powers[step*noStages*batch->accelLen + stage*batch->accelLen + x ];
                    }
                    else
                    {
                      //poww = candB.sigma;
                    }
                    rr      = rVal->drlo + x *  ACCEL_DR ;

                    if ( rr < batch->SrchSz->searchRHigh )
                    {
                      rr    /=  (double)numharm ;
                      zz    =   ( candB.z * ACCEL_DZ - batch->hInfos[0].zmax )              / (double)numharm ;

                      if      ( batch->flag & CU_CAND_LST )
                      {
                        //*cands = insert_new_accelcand(*cands, poww, sig, numharm, rr, zz, &added);
                      }
                      else if ( batch->flag & CU_CAND_ARR )
                      {
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

                        if ( grIdx >= 0 && grIdx < batch->SrchSz->noOutpR )  // Valid index
                        {
                          batch->noResults++;

                          if ( batch->flag & FLAG_STORE_ALL )               // Store all stages
                          {
                            grIdx += stage * (batch->SrchSz->noOutpR);      // Stride by size
                          }

                          if ( batch->cndType == CU_FULLCAND )
                          {
                            //#pragma omp critical
                            {
                              cand* candidate = &((cand*)batch->h_candidates)[grIdx];

                              // this sigma is greater than the current sigma for this r value
                              if ( candidate->sig < sig )
                              {
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
                            exit(1);
                          }
                        }
                      }
                      else
                      {
                        fprintf(stderr,"ERROR: function %s requires cand\n",__FUNCTION__);
                        exit(1);
                      }
                    }
                  }
                }
                else
                {
                  fprintf(stderr,"ERROR: function %s requires accelcandBasic\n",__FUNCTION__);
                  exit(1);
                }
              }
            }
          }
        }

#ifdef STPMSG
        printf("\t\t\tDone\n");
#endif
        //if ( !(batch->flag & FLAG_SIG_GPU) && (batch->retType & CU_SMALCAND) )
        //  free(powers);
      }

      //free(powers);
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

#ifdef STPMSG
    printf("\t\tCopy results from device to host\n");
#endif

    cudaStreamWaitEvent(batch->strmSearch, batch->searchComp,  0);
    cudaStreamWaitEvent(batch->strmSearch, batch->processComp, 0);

#ifdef TIMING // Timing event  .
    CUDA_SAFE_CALL(cudaEventRecord(batch->candCpyInit,  batch->strmSearch),"Recording event: candCpyInit");
#endif

    CUDA_SAFE_CALL(cudaMemcpyAsync(batch->h_retData, batch->d_retData, batch->retDataSize*batch->noSteps, cudaMemcpyDeviceToHost, batch->strmSearch), "Failed to copy results back");

    CUDA_SAFE_CALL(cudaEventRecord(batch->candCpyComp, batch->strmSearch),"Recording event: readComp");
    CUDA_SAFE_CALL(cudaGetLastError(), "Copying results back from device.");

    batch->haveConvData        = 0;
    batch->haveSearchResults   = 1;
  }
}

void sumAndSearch(cuFFdotBatch* batch, long long *numindep)
{
  //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  if ( batch->haveSearchResults || batch->haveConvData ) // previous plain has data data so sum and search
  {
    nvtxRangePush("Add & Search");
#ifdef STPMSG
    printf("\tSum & Search\n");
#endif

    int noStages = log(batch->noHarms)/log(2) + 1;

    FOLD // Do synchronisations  .
    {
      for (int ss = 0; ss < batch->noStacks; ss++)
      {
        cuFfdotStack* cStack = &batch->stacks[ss];

        cudaStreamWaitEvent(batch->strmSearch, cStack->plnComp, 0);
      }
    }

    FOLD // Call the main sum & search kernel  .
    {
      SSKer(batch, numindep);
    }


#ifdef SYNCHRONOUS
    FOLD // Copy results from device to host  .
    {
      getResults(batch);
    }

    FOLD // Process previous results  .
    {
      processSearchResults(batch, numindep);
    }
#else
    FOLD // Process previous results  .
    {
      processSearchResults(batch, numindep);
    }

    FOLD // Copy results from device to host  .
    {
      getResults(batch);
    }
#endif

    nvtxRangePop();
  }

#ifdef TIMING // Timing  .

#ifndef SYNCHRONOUS
  if ( batch->haveSearchResults )
#endif
  {
    // A blocking synchronisation to ensure results are ready to be proceeded by the host
    CUDA_SAFE_CALL(cudaEventSynchronize(batch->candCpyComp), "ERROR: copying result from device to host.");

    float time;         // Time in ms of the thing
    cudaError_t ret;    // Return status of cudaEventElapsedTime

    FOLD // Convolution timing  .
    {
      if ( !(batch->flag & FLAG_MUL_CB_IN) )
      {
        // Did the convolution by separate kernel

        if ( batch->flag & FLAG_MUL_BATCH )   // Convolution was done on the entire batch  .
        {
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
    }

    FOLD // Inverse FFT timing  .
    {
      for (int ss = 0; ss < batch->noStacks; ss++)
      {
        cuFfdotStack* cStack = &batch->stacks[ss];

        cudaError_t e1 = cudaEventQuery(cStack->invFFTinit);
        cudaError_t e2 = cudaEventQuery(cStack->plnComp);

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

        ret = cudaEventElapsedTime(&time, cStack->invFFTinit, cStack->plnComp);
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
    }

    FOLD // Search Timing  .
    {
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
    }

    FOLD // Copy D2H  .
    {
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
    }
  }
#endif
}

void sumAndMax(cuFFdotBatch* batch, long long *numindep, float* powers)
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

        cudaStreamWaitEvent(batch->strmSearch, cStack->plnComp, 0);
      }
    }

    if ( batch->haveConvData ) // We have a convolved plain so call Sum & search  kernel .
    {
      FOLD // Call the main sum & search kernel
      {
        dimBlock.x  = SS3_X;
        dimBlock.y  = SS3_Y;

        float bw    = SS3_X * SS3_Y;
        //float ww    = batch->batch[0].ffdotPowWidth[0] / ( bw );
        float ww    = batch->accelLen / ( bw );

        dimGrid.x   = ceil(ww);
        dimGrid.y   = 1;

        //add_and_maxCU31_f(dimGrid, dimBlock, 0, batch->strmSearch, searchList, (float*)batch->d_retData, batch->d_candSem, 0, pd, &batch->batch->rLow[0], batch->noSteps, batch->noHarmStages, batch->flag );

        // Run message
        CUDA_SAFE_CALL(cudaGetLastError(), "Error at add_and_searchCU31 kernel launch");

        CUDA_SAFE_CALL(cudaEventRecord(batch->searchComp,  batch->strmSearch),"Recording event: searchComp");
      }
    }

    if ( batch->haveSearchResults ) // Process previous results  .
    {
      // A blocking synchronisation to ensure results are ready to be proceeded by the host
      CUDA_SAFE_CALL(cudaEventSynchronize(batch->candCpyComp), "ERROR: copying result from device to host.");

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

        CUDA_SAFE_CALL(cudaMemcpyAsync(batch->h_retData, batch->d_retData, batch->retDataSize*batch->noSteps, cudaMemcpyDeviceToHost, batch->strmSearch), "Failed to copy results back");

        CUDA_SAFE_CALL(cudaEventRecord(batch->candCpyComp, batch->strmSearch),"Recording event: readComp");
        CUDA_SAFE_CALL(cudaGetLastError(), "Copying results back from device.");

        batch->haveConvData        = 0;
        batch->haveSearchResults   = 1;
      }
    }
  }

  nvtxRangePop();
}
