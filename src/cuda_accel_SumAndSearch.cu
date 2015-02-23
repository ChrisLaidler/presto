/**
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation version 3.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

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

__device__ __constant__ int        YINDS[MAX_YINDS];
__device__ __constant__ float      YINDS_F[MAX_YINDS];
__device__ __constant__ float      POWERCUT[MAX_HARM_NO];
__device__ __constant__ float      NUMINDEP[MAX_HARM_NO];

__device__ __constant__ int        HEIGHT[MAX_HARM_NO];
__device__ __constant__ int        STRIDE[MAX_HARM_NO];
__device__ __constant__ int        HWIDTH[MAX_HARM_NO];
//__device__ __constant__ long       BINNO[MAX_HARM_NO][MAX_STEPS];


//========================================= Constants ====================================================\\

__device__ const float FRAC[16]      =  {1, 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875,0.625,0.1875,0.3125,0.4375,0.5625,0.6875,0.8125,0.9375};
__device__ const int   STAGE[5][2]   =  { {0,0}, {1,1}, {2,3}, {4,7}, {8,15} };


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
  if      ( FLAGS & FLAG_STP_ROW )
  {
    return plainY * noSteps + step;
  }
  else if ( FLAGS & FLAG_STP_PLN )
  {
    return plainY + plainHeight*step;
  }
  /*
  else if ( FLAGS & FLAG_STP_STK )
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
  if  ( (FLAGS & FLAG_PLN_TEX ) )
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
  else if ( n == 4 	)
  {
    *q = exp(-x)*( x*(x*(x/6.0 + 0.5) + 1.0 ) + 1.0 );
  }
  else if ( n == 8 	)
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
      double tmp = p;
      p = q;
      q = tmp;
    }
  }

  // Make an initial estimate for x
  // The algorithm taken from: http://home.online.no/~pjacklam/notes/invnorm/#The_algorithm
  if ( 0.02425 <= p && p <= 0.97575 )
  {
    l    =  p - 0.5;
    ll   = l*l;
    x    = (((((a[1]*ll+a[2])*ll+a[3])*ll+a[4])*ll+a[5])*ll+a[6])*l / (((((b[1]*ll+b[2])*ll+b[3])*ll+b[4])*ll+b[5])*ll+1.0);
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
    x = (((((c[1]*l+c[2])*l+c[3])*l+c[4])*l+c[5])*l+c[6]) / ((((d[1]*l+d[2])*l+d[3])*l+d[4])*l+1.0);

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

    double logQ, logP;
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
    double gpu_p, gpu_q, sigc, pp ;

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

__global__ void print_YINDS2(int no)
{
  const int bidx  = threadIdx.y * SS3_X       +   threadIdx.x;
  const int tid   = blockIdx.x  * (SS3_Y*SS3_X) + bidx;

  if ( tid == 0 )
  {
    printf("%p\n", YINDS );

    for(int i = 0 ; i < no; i ++)
    {
      printf("%03i: %-5i  %i \n", i, YINDS[i], sizeof(int)*8 );
    }
    printf("\n");
  }
}


template<uint FLAGS, int noStages, const int noHarms, int noSteps, typename stpType>
__global__ void add_and_searchCU3_PT(const uint width, accelcandBasic* d_cands, stpType rBin, tHarmList texs)
{
  //  HEIGHT[MAX_HARM_NO];
  //  STRIDE[MAX_HARM_NO];
  //  HWIDTH[MAX_HARM_NO];

  const int bidx  = threadIdx.y * SS3_X         +  threadIdx.x;   /// Block index
  const int tid   = blockIdx.x  * (SS3_Y*SS3_X) +  bidx;          /// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    const int nPowers     = 8 ; // (noStages)*2;                /// NB this is a configurable value, The number of powers to batch calculate together, *2 is a "random choice it would be this or noHarms
    const int zeroHeight  = HEIGHT[0];
    const int oStride     = STRIDE[0];                          /// The stride of the output data

    //int iy, ix;                                                 /// Global indices scaled to sub-batch
    //int y;

    float           inds      [noSteps][noHarms];
    accelcandBasic  candLists [noStages][noSteps];
    float           powers    [noSteps][nPowers];

    FOLD // Prep - Initialise the x indices & set candidates to 0 .
    {
      // Calculate the x indices or create a pointer offset by the correct amount
//#pragma unroll
      for ( int harm = 0; harm < noHarms; harm++ )              // loop over harmonic  .
      {
//#pragma unroll
        for ( int step = 0; step < noSteps; step++)             // Loop over steps
        {
          float fx    = (rBin.val[step] + tid)*FRAC[harm] - rBin.val[harm*noSteps + step]  + HWIDTH[harm];

          if        ( FLAGS & FLAG_STP_ROW )
          {
            fx += step*STRIDE[harm] ;
          }

          if(tid == 99 && step == 0 )
          {
            float fy = 0;

            const float cmpf      = tex2D < float > (texs[harm], fx, fy);


            printf("harm: %02i  fx: %8.2f  zeroHeight %i   cmpf: %f\n",harm, fx, zeroHeight, cmpf);
          }

          inds[step][harm]      = fx;
        }
      }

      FOLD  // Set the local and return candidate powers to zero
      {
//#pragma unroll
        for ( int stage = 0; stage < noStages; stage++ )
        {
//#pragma unroll
          for ( int step = 0; step < noSteps; step++)           // Loop over steps
          {
            // Set the local  candidate
            candLists[stage][step].sigma = 0 ;

            // Set the return candidate
            d_cands[step*noStages*oStride + stage*oStride + tid ].sigma = 0;
          }
        }
      }
    }

    FOLD // Sum & Search (ignore contaminated ends tid o starts at correct spot
    {
      for( int y = 0; y < zeroHeight ; y += nPowers )               // loop over chunks .
      {
        int start   = 0;
        int end     = 0;

        /*
        if ( tid == 99 )
        {
          printf("Y %i  noStages: %02i \n",y, noStages);
        }
        */

        // Initialise powers for each section column to 0
//#pragma unroll
        for ( int step = 0; step < noSteps; step++)             // Loop over steps .
        {
//#pragma unroll
          for( int i = 0; i < nPowers ; i++ )                   // Loop over powers .
          {
            powers[step][i] = 0;
          }
        }

        FOLD // Loop over stages, sum and search
        {
          //#pragma unroll
          for ( int stage = 0 ; stage < noStages; stage++)        // Loop over stages .
          {

            if ( tid == 99 )
            {
            start = STAGE[stage][0];
            end   = STAGE[stage][1];
              printf("Y %i  %02i  %02i \n",y, start, end);
            }

            /*
            // Create a section of summed powers one for each step
            //#pragma unroll
            for ( int harm = start; harm <= end; harm++ )         // Loop over harmonics (batch) in this stage
            {
              //#pragma unroll
              for( int yPlus = 0; yPlus < nPowers; yPlus++ )      // Loop over the chunk  .
              {
                int trm   = y + yPlus ;                           ///< True Y index in plain
                float iy  = zeroHeight*trm/HEIGHT[harm];

                //#pragma unroll
                for ( int step = 0; step < noSteps; step++)        // Loop over steps  .
                {
                  if        ( FLAGS & FLAG_STP_PLN )
                  {
                    iy += step*STRIDE[harm] ;
                  }

                  //const float cmpf      = tex2D < float > (texs[harm], inds[step][harm], iy);
                  //powers[step][yPlus]   += cmpf;

                  if ( tid == 99 && step == 0 && trm == 27 )
                  {
                    const float cmpf      = tex2D < float > (texs[harm], inds[step][harm], iy);
                    //powers[step][yPlus]   += cmpf;

                    float tl = tex2D < float > (texs[harm],  floor(inds[step][harm]) , ceil(iy)  );
                    float tr = tex2D < float > (texs[harm],  ceil(inds[step][harm])  , ceil(iy)  );
                    float bl = tex2D < float > (texs[harm],  floor(inds[step][harm]) , floor(iy) );
                    float br = tex2D < float > (texs[harm],  ceil(inds[step][harm])  , floor(iy) );

                    printf("-- %03i --\n",trm);
                    printf("%15.3f |                 | %15.3f\n",tl,tr);
                    printf("               | %15.3f | \n",cmpf);
                    printf("%15.3f |                 | %15.3f\n",bl,br);
                  }


                }
              }
            }

            */


            // Search set of powers
            //#pragma unroll
            FOUT
            {
              for ( int step = 0; step < noSteps; step++)           // Loop over steps
              {
                //#pragma unroll
                for( int yPlus = 0; yPlus < nPowers ; yPlus++ )     // Loop over section
                {
                  if  (  powers[step][yPlus] > POWERCUT[stage] )
                  {
                    if ( powers[step][yPlus] > candLists[stage][step].sigma )
                    {
                      if ( y + yPlus < zeroHeight )
                      {
                        // This is our new max!
                        candLists[stage][step].sigma  = powers[step][yPlus];
                        candLists[stage][step].z      = y+yPlus;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    FOUT // Write results back to DRAM and calculate sigma if needed
    {
//#pragma unroll
      for ( int step = 0; step < noSteps; step++)             // Loop over steps
      {
//#pragma unroll
        for ( int stage = 0 ; stage < noStages; stage++)      // Loop over stages
        {
          if  ( candLists[stage][step].sigma >  POWERCUT[stage] )
          {
            // Write to DRAM
            d_cands[step*noStages*oStride + stage*oStride + tid] = candLists[stage][step];
          }
        }
      }
    }
  }
}

template<uint FLAGS, int noStages, const int noHarms,  int noSteps>
__host__ void add_and_searchCU3_PT_d(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  tHarmList   texs;
  for (int i = 0; i < noHarms; i++)
  {
    int idx         =  batch->pIdx[i];
    texs.val[i]     = batch->plains[idx].datTex;
  }

  if      (noHarms*noSteps <= 8 )
  {
    long08 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->pIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    add_and_searchCU3_PT<FLAGS,noStages,noHarms,noSteps,long08><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, rBin, texs );
  }
  else if (noHarms*noSteps <= 16 )
  {
    long16 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->pIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    add_and_searchCU3_PT<FLAGS,noStages,noHarms,noSteps,long16><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, rBin, texs );
  }
  else if (noHarms*noSteps <= 32 )
  {
    long32 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->pIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    add_and_searchCU3_PT<FLAGS,noStages,noHarms,noSteps,long32><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, rBin, texs );
  }
  else if (noHarms*noSteps <= 64 )
  {
    long64 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->pIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    add_and_searchCU3_PT<FLAGS,noStages,noHarms,noSteps,long64><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, rBin, texs );
  }
  else if (noHarms*noSteps <= 128 )
  {
    long128 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->pIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    add_and_searchCU3_PT<FLAGS,noStages,noHarms,noSteps,long128><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, rBin, texs );
  }
  else
  {
    fprintf(stderr,"ERROR: %s has not been set up to work with %i elements.",__FUNCTION__, noHarms*noSteps);
  }
}

template<uint FLAGS, int noStages, const int noHarms>
__host__ void add_and_searchCU3_PT_s(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int noSteps = batch->noSteps ;

  switch (noSteps)
  {
    case 1:
    {
      //add_and_searchCU3_PT_d<FLAGS,noStages,noHarms,1>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 2:
    {
      //add_and_searchCU3_PT_d<FLAGS,noStages,noHarms,2>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 3:
    {
      //add_and_searchCU3_PT_d<FLAGS,noStages,noHarms,3>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 4:
    {
      add_and_searchCU3_PT_d<FLAGS,noStages,noHarms,4>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 5:
    {
      //add_and_searchCU3_PT_d<FLAGS,noStages,noHarms,5>(dimGrid, dimBlock, stream, batch);
      break;
    }
    default:
      fprintf(stderr, "ERROR: add_and_searchCU311 has not been templated for %i steps\n", noSteps);
      exit(EXIT_FAILURE);
  }
}

template<uint FLAGS >
__host__ void add_and_searchCU3_PT_p(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int noStages = batch->noHarmStages;

  switch (noStages)
  {
    case 1:
    {
      //add_and_searchCU3_PT_s<FLAGS,1,1>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 2:
    {
      //add_and_searchCU3_PT_s<FLAGS,2,2>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 3:
    {
      //add_and_searchCU3_PT_s<FLAGS,3,4>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 4:
    {
      add_and_searchCU3_PT_s<FLAGS,4,8>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 5:
    {
      //add_and_searchCU3_PT_s<FLAGS,5,16>(dimGrid, dimBlock, stream, batch);
      break;
    }
    default:
      fprintf(stderr, "ERROR: %s has not been templated for %i stages\n", __FUNCTION__, noStages);
      exit(EXIT_FAILURE);
  }
}

__host__ void add_and_searchCU3_PT_f(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const uint FLAGS = batch->flag ;

  if        ( (FLAGS & FLAG_CUFFTCB_OUT) &&  (FLAGS & FLAG_PLN_TEX) )
  {
    if      ( FLAGS & FLAG_STP_ROW )
      add_and_searchCU3_PT_p<FLAG_CUFFTCB_OUT | FLAG_PLN_TEX | CU_OUTP_SINGLE | FLAG_STP_ROW> (dimGrid, dimBlock, stream, batch);
    else if ( FLAGS & FLAG_STP_PLN )
      add_and_searchCU3_PT_p<FLAG_CUFFTCB_OUT | FLAG_PLN_TEX | CU_OUTP_SINGLE | FLAG_STP_PLN> (dimGrid, dimBlock, stream, batch);
    else
    {
      fprintf(stderr, "ERROR: %s has not been templated for flag combination. \n", __FUNCTION__ );
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    fprintf(stderr, "ERROR: %s has not been templated for flag combination. \n", __FUNCTION__ );
    exit(EXIT_FAILURE);
  }
}


template<uint FLAGS, int noStages, const int noHarms, int noSteps, typename stpType>
__global__ void add_and_searchCU3111(const uint width, accelcandBasic* d_cands, stpType rBin, fMax rLows, tHarmList texs, fsHarmList powers, cHarmList cmplx  )
{
  //  HEIGHT[MAX_HARM_NO];
  //  STRIDE[MAX_HARM_NO];
  //  HWIDTH[MAX_HARM_NO];

  const int bidx  = threadIdx.y * SS3_X         +  threadIdx.x;   /// Block index
  const int tid   = blockIdx.x  * (SS3_Y*SS3_X) +  bidx;          /// Global thread id (ie column) 0 is the first 'good' column

  if ( tid < width )
  {
    const int nPowers     = 8 ; // (noStages)*2;                /// NB this is a configurable value, The number of powers to batch calculate together, *2 is a "random choice it would be this or noHarms
    const int zeroHeight  = HEIGHT[0];
    const int oStride     = STRIDE[0];                          /// The stride of the output data

    int iy, ix;                                                 /// Global indices scaled to sub-batch
    int y;

    int             inds      [noSteps][noHarms];
    accelcandBasic  candLists [noStages][noSteps];
    fcomplexcu*     pData     [noSteps][noHarms];
    //float*          pPowr     [noSteps][noHarms];
    float           powers    [noSteps][nPowers];                       // registers to hold values to increase mem cache hits

    FOLD // Prep - Initialise the x indices & set candidates to 0 .
    {
      // Calculate the x indices or create a pointer offset by the correct amount
//#pragma unroll
      for ( int harm = 0; harm < noHarms; harm++ )              // loop over harmonic  .
      {
//#pragma unroll
        for ( int step = 0; step < noSteps; step++)             // Loop over steps
        {
          float fx    = (rBin.val[step] + tid)*FRAC[harm] - rBin.val[harm*noSteps + step]  + HWIDTH[harm] ;
          int   ix    = round(fx) ;

          if ( tid == 99 && step == 0 )
          {
            printf("harm %03i  %.3f   fx: %6.3f  ix: %5i  hWidth %04i  rscaled  %012.3f  rbase %06i \n", harm, FRAC[harm], fx, ix, HWIDTH[harm],  (rBin.val[step] + tid)*FRAC[harm], rBin.val[harm*noSteps + step]  );
          }

          int drlo    = (int) ( ACCEL_RDR * rLows.arry[step] * FRAC[harm] + 0.5 ) * ACCEL_DR ;
          float srlo  = (int) ( ACCEL_RDR * ( rLows.arry[step] + tid * ACCEL_DR ) * FRAC[harm] + 0.5 ) * ACCEL_DR ;
          float fx2   = (srlo - drlo ) * ACCEL_RDR ;
          int ix2     = (srlo - drlo) * ACCEL_RDR + HWIDTH[harm] ;

          if ( ix != ix2 )
          {
            printf("tid: %04i  step:: %02i   harm: %02i   ix error! ix: %05i  ix2: %05i   fx: %8.3f   fx2: %8.3f   drlo: %07i  srlo: %8.2f  \n", tid, step, harm, ix, ix2, fx,  fx2,  drlo, srlo);
          }

          if     (FLAGS & FLAG_PLN_TEX)  // Calculate x index
          {
            inds[step][harm]      = ix;
          }
          else                           // Create a pointer list that is offset by the correct amount
          {
            inds[step][harm]      = ix;

            if        ( FLAGS & FLAG_STP_ROW )
            {
              if      ( FLAGS & FLAG_CUFFTCB_OUT )
              {
                //pPowr[step][harm]   = &powers[harm][ ix + STRIDE[harm]*step ] ;
              }
              else
              {
                pData[step][harm]   = &cmplx[harm][  ix + STRIDE[harm]*step ] ;
              }
            }
            else if   ( FLAGS & FLAG_STP_PLN )
            {
              if      ( FLAGS & FLAG_CUFFTCB_OUT )
              {
                //pPowr[step][harm]   = &powers[harm][ ix + STRIDE[harm]*step*HEIGHT[harm] ] ;
              }
              else
              {
                pData[step][harm]   = &cmplx[harm][  ix + STRIDE[harm]*step*HEIGHT[harm] ] ;
              }
            }
          }
        }

        // Change the stride for this harmonic
        if     ( FLAGS & FLAG_PLN_TEX )
        {
        }
        else
        {
          if        ( FLAGS & FLAG_STP_ROW )
          {
            if ( FLAGS & FLAG_CUFFTCB_OUT )
            {
              //STRIDE[harm] *= noSteps;
            }
            else
            {
              //STRIDE[harm] *= noSteps;
            }
          }
        }
      }

      FOLD  // Set the local and return candidate powers to zero
      {
//#pragma unroll
        for ( int stage = 0; stage < noStages; stage++ )
        {
//#pragma unroll
          for ( int step = 0; step < noSteps; step++)           // Loop over steps
          {
            candLists[stage][step].sigma = 0 ;

            if ( FLAGS & CU_OUTP_SINGLE )
            {
              d_cands[step*noStages*oStride + stage*oStride + tid ].sigma = 0;
            }
          }
        }
      }
    }

    FOLD // Sum & Search (ignore contaminated ends tid o starts at correct spot
    {
      for( y = 0; y < zeroHeight ; y += nPowers ) 							// loop over chunks .
      {
        int start   = 0;
        int end     = 0;

        // Initialise powers for each section column to 0
//#pragma unroll
        for ( int step = 0; step < noSteps; step++)       	    // Loop over steps .
        {
//#pragma unroll
          for( int i = 0; i < nPowers ; i++ )                   // Loop over powers .
          {
            powers[step][i] = 0;
          }
        }

        // Loop over stages, sum and search
        //#pragma unroll
        for ( int stage = 0 ; stage < noStages; stage++)        // Loop over stages .
        {
          /*
          if      ( stage == 0 )
          {
            start = 0;
            end = 1;
          }
          else if ( stage == 1 )
          {
            start = 1;
            end = 2;
          }
          else if ( stage == 2 )
          {
            start = 2;
            end = 4;
          }
          else if ( stage == 3 )
          {
            start = 4;
            end = 8;
          }
          else if ( stage == 4 )
          {
            start = 8;
            end = 16;
          }
*/
          start = STAGE[stage][0];
          end   = STAGE[stage][1];

          // Create a section of summed powers one for each step
          //#pragma unroll
          for ( int harm = start; harm <= end; harm++ )         // Loop over harmonics (batch) in this stage
          {
            //#pragma unroll
            for( int yPlus = 0; yPlus < nPowers; yPlus++ )      // Loop over the chunk  .
            {
              int trm		= y + yPlus ;														///< True Y index in plain
              iy        = YINDS[ zeroHeight*harm + trm ];

//#pragma unroll
              for ( int step = 0; step < noSteps; step++)        // Loop over steps  .
              {
                if     (FLAGS & FLAG_PLN_TEX)
                {
                  if ( FLAGS & FLAG_CUFFTCB_OUT )
                  {
                    // TODO: NB: use powers and texture memory to interpolate values, this requires a float value for y will have to calculate
                  }
                  else
                  {
                    // Calculate y indice
                    if      ( FLAGS & FLAG_STP_ROW )
                    {
                      iy  = ( iy * noSteps + step );
                    }
                    else if ( FLAGS & FLAG_STP_PLN )
                    {
                      iy  = ( iy + HEIGHT[harm]*step ) ;
                    }

                    const float2 cmpf         = tex2D < float2 > (texs[harm], inds[step][harm], iy);
                    powers[step][yPlus]      += cmpf.x * cmpf.x + cmpf.y * cmpf.y;
                  }
                }
                else
                {
                  if ( FLAGS & FLAG_CUFFTCB_OUT )
                  {
                    float power;
                    if        ( FLAGS & FLAG_STP_ROW )
                    {
                      power = powers[harm][ (inds[step][harm]  + STRIDE[harm]*noSteps*iy + STRIDE[harm]*step) ] ;
                      //power = pPowr[step][harm][STRIDE[harm]*iy] ; // Note stride has been set depending on multi-step type
                    }
                    else if   ( FLAGS & FLAG_STP_PLN )
                    {
                      power = powers[harm][ inds[step][harm]  + STRIDE[harm]*iy + STRIDE[harm]*step*HEIGHT[harm] ] ;
                    }
                    powers[step][yPlus]        += power;
                  }
                  else
                  {
                    fcomplexcu cmpc;
                    if        ( FLAGS & FLAG_STP_ROW )
                    {
                      //cmpc = cmplx[harm][ inds[step][harm]  + STRIDE[harm]*noSteps*iy + STRIDE[harm]*step ] ;
                      cmpc = pData[step][harm][STRIDE[harm]*iy] ; // Note stride has been set depending on multi-step type
                    }
                    else if   ( FLAGS & FLAG_STP_PLN )
                    {
                      cmpc = cmplx[harm][ inds[step][harm]  + STRIDE[harm]*iy + STRIDE[harm]*step*HEIGHT[harm] ] ;
                    }

                    powers[step][yPlus]        += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                  }
                }
              }
            }
          }

          // Search set of powers
//#pragma unroll
          for ( int step = 0; step < noSteps; step++)         	// Loop over steps
          {
//#pragma unroll
            for( int yPlus = 0; yPlus < nPowers ; yPlus++ )     // Loop over section
            {
              if  (  powers[step][yPlus] > POWERCUT[stage] )
              {
                if ( powers[step][yPlus] > candLists[stage][step].sigma )
                {
                  if ( y + yPlus < zeroHeight )
                  {
                    // This is our new max!
                    candLists[stage][step].sigma  = powers[step][yPlus];
                    candLists[stage][step].z      = y+yPlus;
                  }
                }
              }
            }
          }
        }
      }
    }

    // Write results back to DRAM and calculate sigma if needed
    if      ( FLAGS & CU_OUTP_DEVICE && 0)
    {
      /*
//#pragma unroll
      for ( int stage = 0 ; stage < noStages; stage++)
      {
        const short numharm = 1 << stage;

#if TEMPLATE_SEARCH == 1
#pragma unroll
#endif
        for ( int step = 0; step < noSteps; step++)         // Loop over steps
        {
          if  ( candLists[stage][step].sigma >  POWERCUT[stage] )
          {
            //float rLow    = rLows.arry[step] * searchList.frac.val[harm];
            //float diff    = rLow - (int)rLow;
            //float idxS    = 0.5  + diff*ACCEL_RDR ;

            int idx =  (int)(( rLows.arry[step] + tid * (double) ACCEL_DR ) / (double)numharm ) - base ;
            if ( idx >= 0 )
            {
              long long numtrials             = NUMINDEP[stage];
              candLists[stage][step].numharm  = numharm;
              //candLists[stage][step].z      = ( candLists[stage].z*(double) ACCEL_DZ - searchList.zMax.val[0]  )  / (double)numharm ;
              candLists[stage][step].sigma    = (float)candidate_sigma_cu(candLists[stage][step].sigma, numharm, numtrials);

              FOLD // Atomic write to global list
              {
                volatile bool done = false;
                while (!done)
                {
                  volatile int prev = atomicCAS(&d_sem[idx], UINT_MAX, tid );
                  if ( prev == UINT_MAX )
                  {
                    if ( candLists[stage][step].sigma > d_cands[idx].sigma )
                    {
                      d_cands[idx]              = candLists[stage][step];
                    }
                    d_sem[idx]                  = UINT_MAX;
                    done = true;
                  }
                }
              }
            }
          }
        }
      }
      */
    }
    else if ( FLAGS & CU_OUTP_SINGLE )
    {
//#pragma unroll
      for ( int step = 0; step < noSteps; step++)             // Loop over steps
      {
//#pragma unroll
        for ( int stage = 0 ; stage < noStages; stage++)      // Loop over stages
        {
          if  ( candLists[stage][step].sigma >  POWERCUT[stage] )
          {
	          // This can be calculated from stage
            //const short numharm                 = ( 1 << stage );
            //candLists[stage][step].numharm      = numharm;

            if ( (FLAGS & FLAG_SAS_SIG) && FALSE)  						// Calculate the actual sigma value on the GPU
            {
              const int numharm                 = ( 1 << stage );
              // Calculate sigma value
              long long numtrials               = NUMINDEP[stage];
              candLists[stage][step].sigma      = (float)candidate_sigma_cu(candLists[stage][step].sigma, numharm, numtrials);
            }

            // Write to DRAM
            d_cands[step*noStages*oStride + stage*oStride + tid] = candLists[stage][step];
          }
        }
      }
    }
  }
}

template<uint FLAGS, int noStages, const int noHarms,  int noSteps>
__host__ void add_and_searchCU311_s(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  /*
  int heights[noHarms];
  int strides[noHarms];
  float  frac[noHarms];
  int  hWidth[noHarms];
  //__global__ long   rBin[noHarms][noSteps];

  int i = 0;
  for (int i = 0; i < noHarms; i++)
  {
    int idx =  batch->pIdx[i];

    heights[i]              = batch->hInfos[idx].height;
    strides[i]              = batch->hInfos[idx].inpStride;
    frac[i]                 = batch->hInfos[idx].harmFrac;
    hWidth[i]               = batch->hInfos[idx].halfWidth*ACCEL_NUMBETWEEN;

    for ( int step = 0; step < noSteps; step++)
    {
      rBin[i][step]         = (*batch->rConvld)[step][idx].expBin ;
    }
  }
  */

  fMax rLow;
  for ( int step = 0; step < noSteps; step++)
  {
    rLow.arry[step]=(*batch->rConvld)[step][0].drlo ;
  }

  tHarmList texs;
  fsHarmList powers;
  cHarmList cmplx;

  for (int i = 0; i < noHarms; i++)
  {
    int idx =  batch->pIdx[i];
    texs.val[i]     = batch->plains[idx].datTex;
    powers.val[i]   = batch->plains[idx].d_plainPowers;
    cmplx.val[i]    = batch->plains[idx].d_plainData;
  }


  if      (noHarms*noSteps <= 8 )
  {
    long08 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->pIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    add_and_searchCU3111<FLAGS,noStages,noHarms,noSteps,long08><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData, rBin, rLow, texs, powers, cmplx );
  }
  else if (noHarms*noSteps <= 16 )
  {
    long16 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->pIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    add_and_searchCU3111<FLAGS,noStages,noHarms,noSteps,long16><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData,rBin, rLow, texs, powers, cmplx );
  }
  else if (noHarms*noSteps <= 32 )
  {
    long32 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->pIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
      //printf("drlo: %.3f \n", (*batch->rConvld)[0][idx].drlo );
    }
    add_and_searchCU3111<FLAGS,noStages,noHarms,noSteps,long32><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData,rBin, rLow, texs, powers, cmplx );
  }
  else if (noHarms*noSteps <= 64 )
  {
    long64 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->pIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    add_and_searchCU3111<FLAGS,noStages,noHarms,noSteps,long64><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData,rBin, rLow, texs, powers, cmplx );
  }
  else if (noHarms*noSteps <= 128 )
  {
    long128 rBin;
    for (int i = 0; i < noHarms; i++)
    {
      int idx =  batch->pIdx[i];
      for ( int step = 0; step < noSteps; step++)
      {
        rBin.val[i*noSteps + step]  = (*batch->rConvld)[step][idx].expBin ;
      }
    }
    add_and_searchCU3111<FLAGS,noStages,noHarms,noSteps,long128><<<dimGrid,  dimBlock, 0, stream >>>(batch->accelLen, (accelcandBasic*)batch->d_retData,rBin, rLow, texs, powers, cmplx );
  }
  else
  {
    fprintf(stderr,"ERROR: %s has not been set up to work with %i elements.",__FUNCTION__, noHarms*noSteps);
  }
}

template<uint FLAGS, int noStages, const int noHarms>
__host__ void add_and_searchCU311_q(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int noSteps = batch->noSteps ;

  switch (noSteps)
  {
    case 1:
    {
      //add_and_searchCU311_s<FLAGS,noStages,noHarms,1>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 2:
    {
      //add_and_searchCU311_s<FLAGS,noStages,noHarms,2>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 3:
    {
      //add_and_searchCU311_s<FLAGS,noStages,noHarms,3>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 4:
    {
      add_and_searchCU311_s<FLAGS,noStages,noHarms,4>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 5:
    {
      //add_and_searchCU311_s<FLAGS,noStages,noHarms,5>(dimGrid, dimBlock, stream, batch);
      break;
    }
    default:
      fprintf(stderr, "ERROR: add_and_searchCU311 has not been templated for %i steps\n", noSteps);
      exit(EXIT_FAILURE);
  }
}

template<uint FLAGS >
__host__ void add_and_searchCU311_p(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const int noStages = batch->noHarmStages;

  switch (noStages)
  {
    case 1:
    {
      //add_and_searchCU311_q<FLAGS,1,1>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 2:
    {
      //add_and_searchCU311_q<FLAGS,2,2>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 3:
    {
      //add_and_searchCU311_q<FLAGS,3,4>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 4:
    {
      add_and_searchCU311_q<FLAGS,4,8>(dimGrid, dimBlock, stream, batch);
      break;
    }
    case 5:
    {
      //add_and_searchCU311_q<FLAGS,5,16>(dimGrid, dimBlock, stream, batch);
      break;
    }
    default:
      fprintf(stderr, "ERROR: %s has not been templated for %i stages\n", __FUNCTION__, noStages);
      exit(EXIT_FAILURE);
  }
}

__host__ void add_and_searchCU311_f(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream, cuFFdotBatch* batch )
{
  const uint FLAGS = batch->flag ;

  if        ( FLAGS & FLAG_CUFFTCB_OUT )
  {
    if      ( FLAGS & FLAG_STP_ROW )
      add_and_searchCU311_p<FLAG_CUFFTCB_OUT | CU_OUTP_SINGLE | FLAG_STP_ROW> (dimGrid, dimBlock, stream, batch);
    else if ( FLAGS & FLAG_STP_PLN )
      add_and_searchCU311_p<FLAG_CUFFTCB_OUT | CU_OUTP_SINGLE | FLAG_STP_PLN>  (dimGrid, dimBlock, stream, batch);
    //else if ( FLAGS & FLAG_STP_STK )
    //  add_and_searchCU311_p<FLAG_CUFFTCB_OUT | CU_OUTP_SINGLE | FLAG_STP_STK> (dimGrid, dimBlock, stream, batch);
    else
    {
      fprintf(stderr, "ERROR: %s has not been templated for flag combination. \n", __FUNCTION__ );
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    if      ( FLAGS & FLAG_STP_ROW )
      add_and_searchCU311_p<CU_OUTP_SINGLE | FLAG_STP_ROW> (dimGrid, dimBlock, stream, batch);
    else if ( FLAGS & FLAG_STP_PLN )
      add_and_searchCU311_p<CU_OUTP_SINGLE | FLAG_STP_PLN> (dimGrid, dimBlock, stream, batch);
    //else if ( FLAGS & FLAG_STP_STK )
    //  add_and_searchCU311_p<CU_OUTP_SINGLE | FLAG_STP_STK> (dimGrid, dimBlock, stream, batch);
    else
    {
      fprintf(stderr, "ERROR: %s has not been templated for flag combination. \n", __FUNCTION__ );
      exit(EXIT_FAILURE);
    }
  }
}


/** Sum and Search - loop down - column max - multi-step .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSteps
 */
#if TEMPLATE_SEARCH == 1
template<uint FLAGS, /*typename sType,*/ int noStages, typename stpType, int noSteps>
__global__ void add_and_searchCU31(cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base/*, sType pd*/, stpType rLows )
#else
template<uint FLAGS, /*typename sType,*/ int noStages, typename stpType>


__global__ void add_and_searchCU31(cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base/*, sType pd*/, stpType rLows, int noSteps )
#endif
{
  const int bidx  = threadIdx.y * SS3_X         +  threadIdx.x;   /// Block index
  const int tid   = blockIdx.x  * (SS3_Y*SS3_X) +  bidx;          /// Global thread id (ie column)
  const int width = searchList.widths.val[0];                     /// The width of usable data



  if ( tid < width )
  {
    const int noHarms     = ( 1 << (noStages-1) ) ;
    const int nPowers     = 8 ; // (noStages)*2;      					// NB this is a configurable value, The number of powers to batch calculate together, *2 is a "random choice it would be this or noHarms
    const int zeroHeight  = searchList.heights.val[0];
    const int oStride     = searchList.strides.val[0];          /// The stride of the output data
    int iy, ix;                                                 /// Global indices scaled to sub-batch
    int y;

#if TEMPLATE_SEARCH == 1
    accelcandBasic candLists[noStages][noSteps];
    int         inds[noSteps][noHarms];
    fcomplexcu* pData[noSteps][noHarms];
    float       powers[noSteps][nPowers];           						// registers to hold values to increase mem cache hits
#else
    accelcandBasic candLists[noStages][MAX_STEPS];
    int         inds[MAX_STEPS][noHarms];
    fcomplexcu* pData[MAX_STEPS][noHarms];
    float       powers[MAX_STEPS][nPowers];         						// registers to hold values to increase mem cache hits
#endif

    FOLD // Prep - Initialise the x indices & set candidates to 0 .
    {
      // Calculate the x indices or create a pointer offset by the correct amount
#pragma unroll
      for ( int harm = 0; harm < noHarms; harm++ )      				// loop over harmonic  .
      {
#if TEMPLATE_SEARCH == 1
#pragma unroll
#endif
        for ( int step = 0; step < noSteps; step++)     				// Loop over steps
        {
          int drlo =   (int) ( ACCEL_RDR * rLows.arry[step] * searchList.frac.val[harm] + 0.5 ) * ACCEL_DR ;
          float srlo = (int) ( ACCEL_RDR * ( rLows.arry[step] + tid * ACCEL_DR ) * searchList.frac.val[harm] + 0.5 ) * ACCEL_DR ;

          ix = (srlo - drlo) * ACCEL_RDR + searchList.ffdBuffre.val[harm] ;

          //float rLow    = rLows.arry[step] * searchList.frac.val[harm];
          //float diff    = rLow - (int)rLow;
          //float idxS    = 0.5f + diff*ACCEL_RDR ;
          //ix = (int)( tid * searchList.frac.val[harm] + idxS ) + searchList.ffdBuffre.val[harm];

          if     (FLAGS & FLAG_PLN_TEX)  // Calculate x index
          {
            inds[step][harm]      = ix;
          }
          else                           // Create a pointer list that is offset by the correct amount
          {
            inds[step][harm]      = ix;

            if        ( FLAGS & FLAG_STP_ROW )
            {
              if      ( FLAGS & FLAG_CUFFTCB_OUT )
              {
                //pPowr[step][harm]   = &searchList.powers.val[harm][ ix + searchList.strides.val[harm]*step ] ;
              }
              else
              {
                pData[step][harm]   = &searchList.datas.val[harm][  ix + searchList.strides.val[harm]*step ] ;
              }
            }
            else if   ( FLAGS & FLAG_STP_PLN )
            {
              if      ( FLAGS & FLAG_CUFFTCB_OUT )
              {
                //pPowr[step][harm]   = &searchList.powers.val[harm][ ix + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
              }
              else
              {
                pData[step][harm]   = &searchList.datas.val[harm][  ix + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
              }
            }
          }
        }

        // Change the stride for this harmonic
        if     ( FLAGS & FLAG_PLN_TEX )
        {
        }
        else
        {
          if        ( FLAGS & FLAG_STP_ROW )
          {
            if ( FLAGS & FLAG_CUFFTCB_OUT )
            {
              //searchList.strides.val[harm] *= noSteps;
            }
            else
            {
              searchList.strides.val[harm] *= noSteps;
            }
          }
        }
      }

      FOLD  // Set the local and return candidate powers to zero
      {
#pragma unroll
        for ( int stage = 0; stage < noStages; stage++ )
        {
#if TEMPLATE_SEARCH == 1
#pragma unroll
#endif
          for ( int step = 0; step < noSteps; step++)   				// Loop over steps
          {
            candLists[stage][step].sigma = 0 ;

            if ( FLAGS & CU_OUTP_SINGLE )
            {
              d_cands[step*noStages*oStride + stage*oStride + tid ].sigma = 0;
            }
          }
        }
      }
    }

    FOLD // Sum & Search (ignore contaminated ends tid o starts at correct spot
    {
      for( y = 0; y < zeroHeight ; y += nPowers ) 							// loop over chunks .
      {
        int start   = 0;
        int end     = 0;

        // Initialise powers for each section column to 0
#if TEMPLATE_SEARCH == 1
#pragma unroll
#endif
        for ( int step = 0; step < noSteps; step++)       	    // Loop over steps .
        {
#pragma unroll
          for( int i = 0; i < nPowers ; i++ )                   // Loop over powers .
          {
            powers[step][i] = 0;
          }
        }

        // Loop over stages, sum and search
        //#pragma unroll
        for ( int stage = 0 ; stage < noStages; stage++)        // Loop over stages .
        {
          if      ( stage == 0 )
          {
            start = 0;
            end = 1;
          }
          else if ( stage == 1 )
          {
            start = 1;
            end = 2;
          }
          else if ( stage == 2 )
          {
            start = 2;
            end = 4;
          }
          else if ( stage == 3 )
          {
            start = 4;
            end = 8;
          }
          else if ( stage == 4 )
          {
            start = 8;
            end = 16;
          }

          // Create a section of summed powers one for each step
          //#pragma unroll
          for ( int harm = start; harm < end; harm++ )          // Loop over harmonics (batch) in this stage
          {
            //#pragma unroll
            for( int yPlus = 0; yPlus < nPowers; yPlus++ )      // Loop over the chunk  .
            {
              int trm		= y + yPlus ;														/// True Y index in plain
              iy        = YINDS[ searchList.yInds.val[harm] + trm ];

#if TEMPLATE_SEARCH == 1
              #pragma unroll
#endif
              for ( int step = 0; step < noSteps; step++)        // Loop over steps  .
              {
                if     (FLAGS & FLAG_PLN_TEX)
                {
                  if ( FLAGS & FLAG_CUFFTCB_OUT )
                  {
                    // TODO: NB: use powers and texture memory to interpolate values
                  }
                  else
                  {
                    // Calculate y indice
                    if      ( FLAGS & FLAG_STP_ROW )
                    {
                      iy  = ( iy * noSteps + step );
                    }
                    else if ( FLAGS & FLAG_STP_PLN )
                    {
                      iy  = ( iy + searchList.heights.val[harm]*step ) ;
                    }

                    const float2 cmpf         = tex2D < float2 > (searchList.texs.val[harm], inds[step][harm], iy);
                    powers[step][yPlus]      += cmpf.x * cmpf.x + cmpf.y * cmpf.y;
                  }
                }
                else
                {
                  if ( FLAGS & FLAG_CUFFTCB_OUT )
                  {
                    float power;
                    if        ( FLAGS & FLAG_STP_ROW )
                    {
                      power = searchList.powers.val[harm][ (inds[step][harm]  + searchList.strides.val[harm]*noSteps*iy + searchList.strides.val[harm]*step) ] ;
                      //power = pPowr[step][harm][searchList.strides.val[harm]*iy] ; // Note stride has been set depending on multi-step type
                    }
                    else if   ( FLAGS & FLAG_STP_PLN )
                    {
                      power = searchList.powers.val[harm][ inds[step][harm]  + searchList.strides.val[harm]*iy + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
                    }
                    powers[step][yPlus]        += power;
                  }
                  else
                  {
                    fcomplexcu cmpc;
                    if        ( FLAGS & FLAG_STP_ROW )
                    {
                      //cmpc = searchList.datas.val[harm][ inds[step][harm]  + searchList.strides.val[harm]*noSteps*iy + searchList.strides.val[harm]*step ] ;
                      cmpc = pData[step][harm][searchList.strides.val[harm]*iy] ; // Note stride has been set depending on multi-step type
                    }
                    else if   ( FLAGS & FLAG_STP_PLN )
                    {
                      cmpc = searchList.datas.val[harm][ inds[step][harm]  + searchList.strides.val[harm]*iy + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
                    }

                    powers[step][yPlus]        += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                  }
                }
              }
            }
          }

          // Search set of powers
#if TEMPLATE_SEARCH == 1
          //#pragma unroll
#endif
          for ( int step = 0; step < noSteps; step++)         	// Loop over steps
          {
            //#pragma unroll
            for( int yPlus = 0; yPlus < nPowers ; yPlus++ )     // Loop over section
            {
              if  (  powers[step][yPlus] > POWERCUT[stage] )
              {
                if ( powers[step][yPlus] > candLists[stage][step].sigma )
                {
                  if ( y + yPlus < zeroHeight )
                  {
                    // This is our new max!
                    candLists[stage][step].sigma  = powers[step][yPlus];
                    candLists[stage][step].z      = y+yPlus;
                  }
                }
              }
            }
          }
        }
      }
    }

    // Write results back to DRAM and calculate sigma if needed
    if      ( FLAGS & CU_OUTP_DEVICE && 0)
    {
//#pragma unroll
      for ( int stage = 0 ; stage < noStages; stage++)
      {
        const short numharm = 1 << stage;

#if TEMPLATE_SEARCH == 1
#pragma unroll
#endif
        for ( int step = 0; step < noSteps; step++)         // Loop over steps
        {
          if  ( candLists[stage][step].sigma >  POWERCUT[stage] )
          {
            //float rLow    = rLows.arry[step] * searchList.frac.val[harm];
            //float diff    = rLow - (int)rLow;
            //float idxS    = 0.5  + diff*ACCEL_RDR ;

            int idx =  (int)(( rLows.arry[step] + tid * (double) ACCEL_DR ) / (double)numharm ) - base ;
            if ( idx >= 0 )
            {
              long long numtrials             = NUMINDEP[stage];
              candLists[stage][step].numharm  = numharm;
              //candLists[stage][step].z      = ( candLists[stage].z*(double) ACCEL_DZ - searchList.zMax.val[0]  )  / (double)numharm ;
              candLists[stage][step].sigma    = (float)candidate_sigma_cu(candLists[stage][step].sigma, numharm, numtrials);

              FOLD // Atomic write to global list
              {
                volatile bool done = false;
                while (!done)
                {
                  volatile int prev = atomicCAS(&d_sem[idx], UINT_MAX, tid );
                  if ( prev == UINT_MAX )
                  {
                    if ( candLists[stage][step].sigma > d_cands[idx].sigma )
                    {
                      d_cands[idx]              = candLists[stage][step];
                    }
                    d_sem[idx]                  = UINT_MAX;
                    done = true;
                  }
                }
              }
            }
          }
        }
      }
    }
    else if ( FLAGS & CU_OUTP_SINGLE )
    {
#if TEMPLATE_SEARCH == 1
#pragma unroll
#endif
      for ( int step = 0; step < noSteps; step++)             // Loop over steps
      {
#pragma unroll
        for ( int stage = 0 ; stage < noStages; stage++)      // Loop over stages
        {
          if  ( candLists[stage][step].sigma >  POWERCUT[stage] )
          {
	          // This can be calculated from stage
            //const short numharm                 = ( 1 << stage );
            //candLists[stage][step].numharm      = numharm;

            if ( (FLAGS & FLAG_SAS_SIG) && FALSE)  						// Calculate the actual sigma value on the GPU
            {
              const int numharm                 = ( 1 << stage );
              // Calculate sigma value
              long long numtrials               = NUMINDEP[stage];
              candLists[stage][step].sigma      = (float)candidate_sigma_cu(candLists[stage][step].sigma, numharm, numtrials);
            }

            // Write to DRAM
            d_cands[step*noStages*oStride + stage*oStride + tid] = candLists[stage][step];
          }
        }
      }
    }
  }
}

/** Sum and Search - loop down - column max - multi-step - step outer .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSteps
 */
#if TEMPLATE_SEARCH == 1
template<uint FLAGS, typename sType, int noStages, typename stpType, int noSteps>
__global__ void add_and_searchCU311(cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base, sType pd, stpType rLows )
#else
template<uint FLAGS, typename sType, int noStages, typename stpType>
__global__ void add_and_searchCU311(cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base, sType pd, stpType rLows, int noSteps )
#endif
{
  /*
  const int bidx  = threadIdx.y * SS3_X         +  threadIdx.x;
  const int tid   = blockIdx.x  * (SS3_Y*SS3_X) +  bidx;

  const int width = searchList.widths.val[0];

  if ( tid < width )
  {
    const int noHarms   = ( 1 << (noStages-1) ) ;
    const int nPowers   = 17 ; // (noStages)*2;      // The number of powers to batch calculate together, *2 is a "random choice it would be this or noHarms

    //register float power;
    float powers[nPowers];            // registers to hold values to increase mem cache hits

    const int zeroHeight = searchList.heights.val[0] ;

    int nStride[noHarms];

#if TEMPLATE_SEARCH == 1
    accelcandBasic candLists[noStages];
    //register float maxP[noStages];
    //int z[noStages];

    // One of the two variables below should get optimised out depending on FLAG_STP_ROW or FLAG_STP_PLN
    int inds[noHarms];
    fcomplexcu* pData[noHarms];
    //float powers[nPowers];         // registers to hold values to increase mem cache hits
#else
    accelcandBasic candLists[noStages];
    //float maxP[noStages];
    //int z[noStages];

    // One of the two variables below should get optimised out depending on FLAG_STP_ROW or FLAG_STP_PLN
    int inds[noHarms];
    fcomplexcu* pData[noHarms];
    //float powers[nPowers];         // registers to hold values to increase mem cache hits
#endif

//#if TEMPLATE_SEARCH == 1
//#pragma unroll
//#endif
    for ( int step = 0; step < noSteps; step++)     // Loop over steps
    {
      int start   = 0;
      int end     = 0;
      int iy;
      int y;

      FOLD // Prep - Initialise the x indices & set candidates to 0 .
      {
        int ix;

        // Calculate the x indices or create a pointer offset by the correct amount
#pragma unroll
        for ( int harm = 0; harm < noHarms; harm++ )      // loop over harmonic
        {
          float rLow    = rLows.arry[step] * searchList.frac.val[harm];
          float diff    = rLow - (int)rLow;
          float idxS    = 0.5f + diff*ACCEL_RDR ;

          ix = (int)( tid * searchList.frac.val[harm] + idxS ) + searchList.ffdBuffre.val[harm];
          if     (FLAGS & FLAG_PLN_TEX)  // Calculate x index
          {
            inds[harm]      = ix;
          }
          else                           // Create a pointer list that is offset by the correct amount
          {
            inds[harm]      = ix;

            if        ( FLAGS & FLAG_STP_ROW )
            {
              pData[harm]   = &searchList.datas.val[harm][ ix + searchList.strides.val[harm]*step ] ;
            }
            else if   ( FLAGS & FLAG_STP_PLN )
            {
              pData[harm]   = &searchList.datas.val[harm][ ix + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
            }
          }

          // Change the stride for this harmonic
          if     ( !( FLAGS & FLAG_PLN_TEX ) && ( FLAGS & FLAG_STP_ROW ) )
          {
            //searchList.strides.val[harm] *= noSteps;
            nStride[harm] = searchList.strides.val[harm] * noSteps;
          }
        }

        // Set the local and return candidate powers to zero
        FOLD
        {
#pragma unroll
          for ( int stage = 0; stage < noStages; stage++ )
          {
            candLists[stage].sigma    = POWERCUT[stage];
            //maxP[stage]               = POWERCUT[stage];

            if ( FLAGS & CU_OUTP_SINGLE )
            {
              d_cands[step*noStages*width + stage*width + tid ].sigma = 0;
            }
          }
        }
      }

      FOLD // Sum & Search .
      {
        for( y = 0; y < zeroHeight ; y+=nPowers ) // loop over chunks  .
        {
          FOLD // Initialise powers for each section column to 0  .
          {
#pragma unroll
            for( int yPlus = 0; yPlus < nPowers; yPlus++ )                // Loop over the chunk
            {
              powers[yPlus] = 0;
            }
          }

          // Loop over stages, sum and search
#pragma unroll
          for ( int stage = 0 ; stage < noStages; stage++)        // Loop over stages .
          {
            if      ( stage == 0 )
            {
              start = 0;
              end = 1;
            }
            else if ( stage == 1 )
            {
              start = 1;
              end = 2;
            }
            else if ( stage == 2 )
            {
              start = 2;
              end = 4;
            }
            else if ( stage == 3 )
            {
              start = 4;
              end = 8;
            }
            else if ( stage == 4 )
            {
              start = 8;
              end = 16;
            }

            // Create a section of summed powers one for each step

#pragma unroll
            for ( int harm = start; harm < end; harm++ )          // Loop over harmonics (batch) in this stage
            {

#pragma unroll
              for( int yPlus = 0; yPlus < nPowers; yPlus++ )                // Loop over the chunk
              {
                int trm       = y + yPlus ;
                iy            = YINDS[ searchList.yInds.val[harm] + trm ] ;

                if     (FLAGS & FLAG_PLN_TEX)
                {
                  // Calculate y indice
                  if      ( FLAGS & FLAG_STP_ROW )
                  {
                    iy  = ( iy * noSteps + step );
                  }
                  else if ( FLAGS & FLAG_STP_PLN )
                  {
                    iy  = ( iy + searchList.heights.val[harm]*step ) ;
                  }

                  const float2 cmpf      = tex2D < float2 > (searchList.texs.val[harm], inds[harm], iy);
                  //power                 += cmpf.x * cmpf.x + cmpf.y * cmpf.y;
                  powers[yPlus]         += cmpf.x * cmpf.x + cmpf.y * cmpf.y;
                }
                else
                {
                  fcomplexcu cmpc;
                  if        ( FLAGS & FLAG_STP_ROW )
                  {
                    //cmpc = searchList.datas.val[harm][ inds[harm]  + searchList.strides.val[harm]*noSteps*iy + searchList.strides.val[harm]*step ] ;
                    cmpc = pData[harm][nStride[harm]*iy] ; // Note stride has been set depending on multi-step type
                  }
                  else if   ( FLAGS & FLAG_STP_PLN )
                  {
                    cmpc = searchList.datas.val[harm][ inds[harm]  + searchList.strides.val[harm]*iy + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
                  }
                  //power           += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                  powers[yPlus]   += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                }
              }
            }

            //#pragma unroll
            for( int yPlus = 0; yPlus < nPowers; yPlus++ )                // Loop over the chunk
            {
              if ( powers[yPlus] > candLists[stage].sigma )
              {
                if ( yPlus + y < zeroHeight)
                {
                  // This is our new max!
                  candLists[stage].sigma  = powers[yPlus];
                  candLists[stage].z      = y + yPlus;
                }
              }
            }

          }
        }
      }

      // Write results back to DRAM and calculate sigma if needed
      if      ( FLAGS & CU_OUTP_DEVICE   )
      {
        //#pragma unroll
        for ( int stage = 0 ; stage < noStages; stage++)
        {
          const short numharm = 1 << stage;

//#if TEMPLATE_SEARCH == 1
//#pragma unroll
//#endif
//          for ( int step = 0; step < noSteps; step++)         // Loop over steps
          {

            if  ( candLists[stage].sigma >  POWERCUT[stage] )
            {
              //float rLow    = rLows.arry[step] * searchList.frac.val[harm];
              //float diff    = rLow - (int)rLow;
              //float idxS    = 0.5  + diff*ACCEL_RDR ;

              int idx =  (int)(( rLows.arry[step] + tid * (double) ACCEL_DR ) / (double)numharm ) - base ;
              if ( idx >= 0 )
              {
                long long numtrials             = NUMINDEP[stage];
                candLists[stage].numharm  = numharm;
                //candLists[stage][step].z      = ( candLists[stage].z*(double) ACCEL_DZ - searchList.zMax.val[0]  )  / (double)numharm ;
                candLists[stage].sigma    = (float)candidate_sigma_cu(candLists[stage].sigma, numharm, numtrials);

                FOLD // Atomic write to global list
                {
                  volatile bool done = false;
                  while (!done)
                  {
                    volatile int prev = atomicCAS(&d_sem[idx], UINT_MAX, tid );
                    if ( prev == UINT_MAX )
                    {
                      if ( candLists[stage].sigma > d_cands[idx].sigma )
                      {
                        d_cands[idx]              = candLists[stage];
                      }
                      d_sem[idx]                  = UINT_MAX;
                      done = true;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else if ( FLAGS & CU_OUTP_SINGLE )
      {
//#if TEMPLATE_SEARCH == 1
//#pragma unroll
//#endif
//        for ( int step = 0; step < noSteps; step++)             // Loop over steps
        {
#pragma unroll
          for ( int stage = 0 ; stage < noStages; stage++)      // Loop over stages
          {

            if  ( candLists[stage].sigma >  POWERCUT[stage] )
            {
              const short numharm                 = ( 1 << stage );
              candLists[stage].numharm      = numharm;

              if ( FLAGS & FLAG_SAS_SIG && FALSE)
              {
                // Calculate sigma value
                long long numtrials               = NUMINDEP[stage];
                candLists[stage].sigma      = (float)candidate_sigma_cu(candLists[stage].sigma, numharm, numtrials);
              }

              // Write to DRAM
              d_cands[step*noStages*width + stage*width + tid] = candLists[stage];
            }
          }
        }
      }
    }
  }
*/
}

/** Sum and Search - loop down - column max - multi-step - shared memory .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSteps
 */
#if TEMPLATE_SEARCH == 1
template<uint FLAGS, typename sType, int noStages, typename stpType, int noSteps>
__global__ void add_and_searchCU32(cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base, sType pd, stpType rLows )
#else
template<uint FLAGS, typename sType, int noStages, typename stpType>
__global__ void add_and_searchCU32(cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base, sType pd, stpType rLows, int noSteps )
#endif
{
  /*
  const int bid   = threadIdx.y * SS3_X         +  threadIdx.x;
  const int tid   = blockIdx.x  * (SS3_Y*SS3_X) +  bid;
  const int width = searchList.widths.val[0];

  if ( tid < width )
  {
    const int noHarms   = ( 1 << (noStages-1) ) ;
    const int hlfHarms  = noHarms / 2.0 ;
    const int nPowers   = hlfHarms ;

    accelcandBasic candLists[noStages];

    // One of the two variables below should get optimised out depending on FLAG_STP_ROW or FLAG_STP_PLN
    int inds[noHarms];
    //fcomplexcu* pData[noHarms];
    float powers[nPowers];         // registers to hold values to increase mem cache hits

    __shared__ float smPowers[hlfHarms][hlfHarms][SS3_Y*SS3_X];  //

    int start   = 0;
    int end     = 0;
    int iy;
    int ix;
    int y;
    const int zeroHeight = searchList.heights.val[0] ;

    for ( int step = 0; step < noSteps; step++)     // Loop over steps
    {
      FOLD // Prep - Initialise the x indices & set candidates to 0 .
      {
        // Calculate the x indices or create a pointer offset by the correct amount
#pragma unroll
        for ( int harm = 0; harm < noHarms; harm++ )      // loop over harmonic
        {
          float rLow    = rLows.arry[step] * searchList.frac.val[harm];
          float diff    = rLow - (int)rLow;
          float idxS    = 0.5f + diff*ACCEL_RDR ;

          ix = (int)( tid * searchList.frac.val[harm] + idxS ) + searchList.ffdBuffre.val[harm];
          if     (FLAGS & FLAG_PLN_TEX)  // Calculate x index
          {
            inds[harm]      = ix;
          }
          else                           // Create a pointer list that is offset by the correct amount
          {
            inds[harm]      = ix;
          }

        }

        // Set the local and return candidate powers to zero
        FOLD
        {
          //#if TEMPLATE_SEARCH == 1
          //#pragma unroll
          //#endif
          //for ( int step = 0; step < noSteps; step++)   // Loop over steps
          {
#pragma unroll
            for ( int stage = 0; stage < noStages; stage++ )
            {
              candLists[stage].sigma = 0;

              if ( FLAGS & CU_OUTP_SINGLE )
              {
                d_cands[step*noStages*width + stage*width + tid ].sigma = 0;
              }
            }
          }
        }
      }

      FOLD // Sum & Search
      {
        FOLD  // Loop over blocks of set length .
        {
          for( y = 0; y < searchList.heights.val[0] ; y += nPowers )  // loop over chunks .
          {
            // Loop over stages, sum and search
#pragma unroll
            for ( int stage = 0 ; stage < noStages; stage++)          // Loop over stages .
            {
              if      ( stage == 0 )
              {
                start = 0;
                end = 1;
              }
              else if ( stage == 1 )
              {
                start = 1;
                end = 2;
              }
              else if ( stage == 2 )
              {
                start = 2;
                end = 4;
              }
              else if ( stage == 3 )
              {
                start = 4;
                end = 8;
              }
              else if ( stage == 4 )
              {
                start = 8;
                end = 16;
              }

              FOLD // Read summed powers into shared memory
              {
#pragma unroll
                for ( int harm = start; harm < end; harm++ )            // Loop over harmonics (batch) in this stage
                {
                  int hi = harm - start;

                  int startY, endY;

                  startY        = YINDS[ searchList.yInds.val[harm] + y ];
                  endY          = YINDS[ searchList.yInds.val[harm] + y + nPowers - 1 ];
                  int yDist     = endY -  startY ;

                  //for (int yy = startY ; yy <= endY; yy++ )
                  for (int yd = 0 ; yd < yDist; yd++ )
                  {
                    if     (FLAGS & FLAG_PLN_TEX)
                    {
                      // Calculate y indice
                      if      ( FLAGS & FLAG_STP_ROW )
                      {
                        iy  = ( yy * noSteps + step );
                      }
                      else if ( FLAGS & FLAG_STP_PLN )
                      {
                        iy  = ( yy + searchList.heights.val[harm]*step ) ;
                      }

                      const float2 cmpf       = tex2D < float2 > (searchList.texs.val[harm], inds[harm], iy);
                      powers[yy-startY]     += cmpf.x * cmpf.x + cmpf.y * cmpf.y;
                    }
                    else
                    {
                      fcomplexcu cmpc;
                      if        ( FLAGS & FLAG_STP_ROW )
                      {
                        cmpc = searchList.datas.val[harm][ inds[harm]  + searchList.strides.val[harm]*((yd+startY)*noSteps + step) ] ;
                        //cmpc = pData[harm][searchList.strides.val[harm]*noSteps*yy] ; // Note stride has been set depending on multi-step type
                      }
                      else if   ( FLAGS & FLAG_STP_PLN )
                      {
                        //cmpc = searchList.datas.val[harm][ inds[harm]  + searchList.strides.val[harm]*yy + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
                      }

                      //if      ( stage == 0 )  // Fundamental Harmonic
                      {
                        powers[yd]               = cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                      }
                      //else                    // Other Harmonics
                      {
                        //smPowers[hi][yd][bid]    = cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                      }
                    }
                  }
                }
              }

              if ( stage != 0 ) // Create summed powers for this stage
              {
                for ( int harm = start; harm < end; harm++ )          // Loop over harmonics (batch) in this stage
                {
                  int startY        = YINDS[ searchList.yInds.val[harm] + y ];

                  for( int yPlus = 0; yPlus < nPowers; yPlus++ )      // Loop over the chunk
                  {
                    int trm = y + yPlus ;

                    if ( trm < zeroHeight )
                    {
                      iy            = YINDS[ searchList.yInds.val[harm] + trm ];

                      int sy = iy - startY;

                      if ( sy >= 0 && sy < hlfHarms && harm-start < hlfHarms  && bid < SS3_Y*SS3_X )
                      {
                        //printf("yPlus %i harm: %i   sy: %i   bid: %i  \n",yPlus, harm-start, sy, bid );

                        //powers[yPlus] += smPowers[harm-start][sy][bid];
                      }
                      else
                      {
                        //printf("Error %i\n",tid);
                        //printf("Error: yPlus %i harm: %i   sy: %i   bid: %i  \n",yPlus, harm-start, sy, bid );
                      }
                    }
                    else
                    {
                      //printf("Error\n");
                    }
                  }
                }
              }

              // Search set of powers
              for( int i = 0; i < nPowers ; i++ )                     // Loop over section
              {
                if  (  powers[i] > POWERCUT[stage] )
                {
                  if ( powers[i] > candLists[stage].sigma )
                  {
                    if ( y + i < zeroHeight )
                    {
                      // This is our new max!
                      candLists[stage].sigma  = powers[i];
                      candLists[stage].z      = y+i;
                    }
                  }
                }
              }

            }
          }
        }
      }

      // Write results back to DRAM and calculate sigma if needed
      if      ( FLAGS & CU_OUTP_DEVICE   )
      {
        //#pragma unroll
        for ( int stage = 0 ; stage < noStages; stage++)
        {
          const short numharm = 1 << stage;

          //#if TEMPLATE_SEARCH == 1
          //#pragma unroll
          //#endif
          //          for ( int step = 0; step < noSteps; step++)         // Loop over steps
          {

            if  ( candLists[stage].sigma >  POWERCUT[stage] )
            {
              //float rLow    = rLows.arry[step] * searchList.frac.val[harm];
              //float diff    = rLow - (int)rLow;
              //float idxS    = 0.5  + diff*ACCEL_RDR ;

              int idx =  (int)(( rLows.arry[step] + tid * (double) ACCEL_DR ) / (double)numharm ) - base ;
              if ( idx >= 0 )
              {
                long long numtrials             = NUMINDEP[stage];
                candLists[stage].numharm  = numharm;
                //candLists[stage][step].z      = ( candLists[stage].z*(double) ACCEL_DZ - searchList.zMax.val[0]  )  / (double)numharm ;
                candLists[stage].sigma    = (float)candidate_sigma_cu(candLists[stage].sigma, numharm, numtrials);

                FOLD // Atomic write to global list
                {
                  volatile bool done = false;
                  while (!done)
                  {
                    volatile int prev = atomicCAS(&d_sem[idx], UINT_MAX, tid );
                    if ( prev == UINT_MAX )
                    {
                      if ( candLists[stage].sigma > d_cands[idx].sigma )
                      {
                        d_cands[idx]              = candLists[stage];
                      }
                      d_sem[idx]                  = UINT_MAX;
                      done = true;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else if ( FLAGS & CU_OUTP_SINGLE )
      {
        //#if TEMPLATE_SEARCH == 1
        //#pragma unroll
        //#endif
        //        for ( int step = 0; step < noSteps; step++)             // Loop over steps
        {
#pragma unroll
          for ( int stage = 0 ; stage < noStages; stage++)      // Loop over stages
          {

            if  ( candLists[stage].sigma >  POWERCUT[stage] )
            {
              const short numharm                 = ( 1 << stage );
              candLists[stage].numharm      = numharm;

              if ( FLAGS & FLAG_SAS_SIG && FALSE)
              {
                // Calculate sigma value
                long long numtrials               = NUMINDEP[stage];
                candLists[stage].sigma      = (float)candidate_sigma_cu(candLists[stage].sigma, numharm, numtrials);
              }

              // Write to DRAM
              d_cands[step*noStages*width + stage*width + tid] = candLists[stage];
            }
          }
        }
      }
    }
  }
  */
}

template<uint FLAGS, /*typename sType,*/ uint noStages>
__host__ void add_and_searchCU31_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream,cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base, /*sType pd,*/ float* rLows, int noSteps)
{
#if TEMPLATE_SEARCH == 1
  switch (noSteps)
  {
    case 1:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU31<FLAGS,sType,noStages,f01,1>, cudaFuncCachePreferL1);
      f01 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_searchCU31<FLAGS,/*sType,*/noStages,f01,1><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 2:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU31<FLAGS,sType,noStages,f02,2>, cudaFuncCachePreferL1);
      f02 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_searchCU31<FLAGS,/*sType,*/noStages,f02,2><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 3:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU31<FLAGS,sType,noStages,f03,3>, cudaFuncCachePreferL1);
      f03 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_searchCU31<FLAGS,/*sType,*/noStages,f03,3><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 4:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU31<FLAGS,sType,noStages,f04,4>, cudaFuncCachePreferL1);
      f04 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_searchCU31<FLAGS,/*sType,*/noStages,f04,4><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 5:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU31<FLAGS,sType,noStages,f05,5>, cudaFuncCachePreferL1);
      f05 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_searchCU31<FLAGS,/*sType,*/noStages,f05,5><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 6:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU31<FLAGS,sType,noStages,f06,6>, cudaFuncCachePreferL1);
      f06 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_searchCU31<FLAGS,/*sType,*/noStages,f06,6><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 7:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU31<FLAGS,sType,noStages,f07,7>, cudaFuncCachePreferL1);
      f07 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_searchCU31<FLAGS,/*sType,*/noStages,f07,7><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 8:
    {
      //cudaFuncSetCacheConfig(add_and_searchCU31<FLAGS,sType,noStages,f08,8>, cudaFuncCachePreferL1);
      f08 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_searchCU31<FLAGS,/*sType,*/noStages,f08,8><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    default:
      fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for %i steps\n", noSteps);
      exit(EXIT_FAILURE);
  }
#else
  //cudaFuncSetCacheConfig(add_and_searchCU31<FLAGS,sType,noStages,fMax>, cudaFuncCachePreferL1);
  fMax tmpArr;
  for (int i = 0; i < noSteps; i++)
    tmpArr.arry[i] = rLows[i];

  add_and_searchCU31<FLAGS,/*sType,*/noStages,fMax> <<<dimGrid, dimBlock, i1, cnvlStream>>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr, noSteps);
#endif
}

template<uint FLAGS >
__host__ void add_and_searchCU31_p(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream,cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base, float* rLows, int noSteps, const uint noStages )
{
  switch (noStages)
  {
    case 1:
    {
      add_and_searchCU31_s<FLAGS,/*sch1,*/1> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, /*tmpArr,*/ rLows, noSteps );
      break;
    }
    case 2:
    {
      add_and_searchCU31_s<FLAGS,/*sch2,*/2> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, /*tmpArr,*/ rLows, noSteps );
      break;
    }
    case 3:
    {
      add_and_searchCU31_s<FLAGS,/*sch4,*/3> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, /*tmpArr,*/ rLows, noSteps );
      break;
    }
    case 4:
    {
      add_and_searchCU31_s<FLAGS,/*sch8,*/4> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, /*tmpArr,*/ rLows, noSteps );
      break;
    }
    case 5:
    {
      add_and_searchCU31_s<FLAGS,/*sch16,*/5> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, /*tmpArr,*/ rLows, noSteps );
      break;
    }
    default:
      fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for %i stages\n", noStages);
      exit(EXIT_FAILURE);
  }
}

__host__ void add_and_searchCU31_f(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream,cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base, float* rLows, int noSteps, const uint noStages, uint FLAGS )
{
  if        ( FLAGS & FLAG_CUFFTCB_OUT )
  {
    if      ( FLAGS & FLAG_STP_ROW )
      add_and_searchCU31_p<FLAG_CUFFTCB_OUT | CU_OUTP_SINGLE | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, rLows, noSteps, noStages );
    else if ( FLAGS & FLAG_STP_PLN )
      add_and_searchCU31_p<FLAG_CUFFTCB_OUT | CU_OUTP_SINGLE | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, rLows, noSteps, noStages );
    //else if ( FLAGS & FLAG_STP_STK )
    //  add_and_searchCU31_p<FLAG_CUFFTCB_OUT | CU_OUTP_SINGLE | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, pd, rLows, noSteps, noStages );
    else
    {
      fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n" );
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    if      ( FLAGS & FLAG_STP_ROW )
      add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, rLows, noSteps, noStages );
    else if ( FLAGS & FLAG_STP_PLN )
      add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, rLows, noSteps, noStages );
    //else if ( FLAGS & FLAG_STP_STK )
    //  add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, pd, rLows, noSteps, noStages );
    else
    {
      fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n" );
      exit(EXIT_FAILURE);
    }
  }

// Uncomenting this block will make compile time VERY long! I mean days!
/*
  if( FLAGS & CU_OUTP_DEVICE )
  {
    if( FLAGS & FLAG_PLN_TEX )
    {
      if ( FLAGS & FLAG_SAS_SIG )
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination.\n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_PLN_TEX | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_PLN_TEX | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_PLN_TEX | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
    }
    else
    {
      if ( FLAGS & FLAG_SAS_SIG )
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_SAS_SIG | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_SAS_SIG | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_SAS_SIG | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_OUTP_DEVICE | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
    }
  }
  else if ( (FLAGS & CU_OUTP_SINGLE) || (FLAGS & CU_OUTP_HOST) )
  {
    if( FLAGS & FLAG_PLN_TEX )
    {
      if ( FLAGS & FLAG_SAS_SIG )
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination.  \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
    }
    else
    {
      if ( FLAGS & FLAG_SAS_SIG )
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination.  \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
    }
  }
  else if (  FLAGS & CU_OUTP_SINGLE )
  {
    if( FLAGS & FLAG_PLN_TEX )
    {
      if ( FLAGS & FLAG_SAS_SIG )
      {
        if      (FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_STK )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination.  \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      (FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_STK )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_PLN_TEX | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
    }
    else
    {
      if ( FLAGS & FLAG_SAS_SIG )
      {
        if      (FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_STK )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination.  \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      (FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_STK )
          add_and_searchCU31_p<CU_OUTP_SINGLE | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
    }
  }
*/
}

/** Sum and Search - loop down - column max - multi-step  .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_OUTP_DEVICE
 * @param noSteps
 */
#if TEMPLATE_SEARCH == 1
template<uint FLAGS, int noStages, typename stpType, int noSteps>
__global__ void add_and_maxCU31(cuSearchList searchList, float* d_cands, uint* d_sem, int base/*, sType pd*/, stpType rLows )
#else
template<uint FLAGS, /*typename sType,*/ int noStages, typename stpType>
__global__ void add_and_maxCU31(cuSearchList searchList, float* d_cands, uint* d_sem, int base/*, sType pd*/, stpType rLows, int noSteps )
#endif
{
  const int bidx  = threadIdx.y * SS3_X         +  threadIdx.x;   /// Block index
  const int tid   = blockIdx.x  * (SS3_Y*SS3_X) +  bidx;          /// Global thread id (ie column)
  const int width = searchList.widths.val[0];                     /// The width of usable data

  if ( tid < width )
  {
    const int noHarms     = ( 1 << (noStages-1) ) ;
    const int nPowers     = 8 ; // (noStages)*2;      	// The number of powers to batch calculate together, *2 is a "random choice it would be this or noHarms
    const int zeroHeight  = searchList.heights.val[0];
    const int oStride     = searchList.strides.val[0];

#if TEMPLATE_SEARCH == 1
    float       candLists[noSteps];
    int         inds[noSteps][noHarms];
    fcomplexcu* pData[noSteps][noHarms];
    float       powers[noSteps][nPowers];           		// registers to hold values to increase mem cache hits
#else
    float       candLists[MAX_STEPS];
    int         inds[MAX_STEPS][noHarms];
    fcomplexcu* pData[MAX_STEPS][noHarms];
    float       powers[MAX_STEPS][nPowers];         		// registers to hold values to increase mem cache hits
#endif

    int iy, ix;             														/// Global indiciec scaled to sub-batch
    int y;

    FOLD // Prep - Initialise the x indices & set candidates to 0  .
    {
      // Calculate the x indices or create a pointer offset by the correct amount
//#pragma unroll
      for ( int harm = 0; harm < noHarms; harm++ )      // loop over harmonic  .
      {
#if TEMPLATE_SEARCH == 1
#pragma unroll
#endif
        for ( int step = 0; step < noSteps; step++)     // Loop over steps  .
        {
          int drlo = (int) ( ACCEL_RDR * rLows.arry[step] * searchList.frac.val[harm] + 0.5 ) * ACCEL_DR ;
          float srlo = (int) ( ACCEL_RDR * ( rLows.arry[step] + tid * ACCEL_DR ) * searchList.frac.val[harm] + 0.5 ) * ACCEL_DR ;

          ix = (srlo - drlo) * ACCEL_RDR + searchList.ffdBuffre.val[harm] ;

          //float rLow    = rLows.arry[step] * searchList.frac.val[harm];
          //float diff    = rLow - (int)rLow;
          //float idxS    = 0.5f + diff*ACCEL_RDR ;
          //ix = (int)( tid * searchList.frac.val[harm] + idxS ) + searchList.ffdBuffre.val[harm];

          if     (FLAGS & FLAG_PLN_TEX)  // Calculate x index
          {
            inds[step][harm]      = ix;
          }
          else                           // Create a pointer list that is offset by the correct amount
          {
            inds[step][harm]      = ix;

            if        ( FLAGS & FLAG_STP_ROW )
            {
              if      ( FLAGS & FLAG_CUFFTCB_OUT )
              {
                //pPowr[step][harm]   = &searchList.powers.val[harm][ ix + searchList.strides.val[harm]*step ] ;
              }
              else
              {
                pData[step][harm]   = &searchList.datas.val[harm][  ix + searchList.strides.val[harm]*step ] ;
              }
            }
            else if   ( FLAGS & FLAG_STP_PLN )
            {
              if      ( FLAGS & FLAG_CUFFTCB_OUT )
              {
                //pPowr[step][harm]   = &searchList.powers.val[harm][ ix + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
              }
              else
              {
                pData[step][harm]   = &searchList.datas.val[harm][  ix + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
              }
            }
          }
        }

        // Change the stride for this harmonic
        if     ( FLAGS & FLAG_PLN_TEX )
        {
        }
        else
        {
          if        ( FLAGS & FLAG_STP_ROW )
          {
            if ( FLAGS & FLAG_CUFFTCB_OUT )
            {
              //searchList.strides.val[harm] *= noSteps;
            }
            else
            {
              searchList.strides.val[harm] *= noSteps;
            }
          }
        }
      }

      FOLD // Set the stored   .
      {
#if TEMPLATE_SEARCH == 1
#pragma unroll
#endif
        for ( int step = 0; step < noSteps; step++)   // Loop over steps
        {
          candLists[step] = 0 ;
        }
      }
    }

    FOLD // Sum & Max  .
    {
      FOLD  // Loop over blocks of set length .
      {
        for( y = 0; y < zeroHeight ; y += nPowers ) 							// loop over chunks .
        {
          int start   = 0;
          int end     = 0;

          // Initialise powers for each section column to 0
#if TEMPLATE_SEARCH == 1
#pragma unroll
#endif
          for ( int step = 0; step < noSteps; step++)             // Loop over steps .
          {
#pragma unroll
            for( int i = 0; i < nPowers ; i++ )                   // Loop over powers .
            {
              powers[step][i] = 0;
            }
          }

          // Loop over stages, sum and search
//#pragma unroll
          for ( int stage = 0 ; stage < noStages; stage++)       	// Loop over stages .
          {
            if      ( stage == 0 )
            {
              start = 0;
              end = 1;
            }
            else if ( stage == 1 )
            {
              start = 1;
              end = 2;
            }
            else if ( stage == 2 )
            {
              start = 2;
              end = 4;
            }
            else if ( stage == 3 )
            {
              start = 4;
              end = 8;
            }
            else if ( stage == 4 )
            {
              start = 8;
              end = 16;
            }

            // Create a section of summed powers one for each step
//#pragma unroll
            for ( int harm = start; harm < end; harm++ )         	// Loop over harmonics (batch) in this stage
            {

//#pragma unroll
              for( int yPlus = 0; yPlus < nPowers; yPlus++ )      // Loop over the chunk
              {
                int trm = y + yPlus ;

                iy            = YINDS[ searchList.yInds.val[harm] + trm ];

#if TEMPLATE_SEARCH == 1
//#pragma unroll
#endif
                for ( int step = 0; step < noSteps; step++)      	// Loop over steps
                {
                  if     (FLAGS & FLAG_PLN_TEX)
                  {
                    // Calculate y indice
                    if      ( FLAGS & FLAG_STP_ROW )
                    {
                      iy  = ( iy * noSteps + step );
                    }
                    else if ( FLAGS & FLAG_STP_PLN )
                    {
                      iy  = ( iy + searchList.heights.val[harm]*step ) ;
                    }

                    const float2 cmpf         = tex2D < float2 > (searchList.texs.val[harm], inds[step][harm], iy);
                    powers[step][yPlus]      += cmpf.x * cmpf.x + cmpf.y * cmpf.y;
                  }
                  else
                  {
                    if ( FLAGS & FLAG_CUFFTCB_OUT )
                    {
                      float power;
                      if        ( FLAGS & FLAG_STP_ROW )
                      {
                        power = searchList.powers.val[harm][ (inds[step][harm]  + searchList.strides.val[harm]*noSteps*iy + searchList.strides.val[harm]*step) ] ;
                        //power = pPowr[step][harm][searchList.strides.val[harm]*iy] ; // Note stride has been set depending on multi-step type
                      }
                      else if   ( FLAGS & FLAG_STP_PLN )
                      {
                        power = searchList.powers.val[harm][ inds[step][harm]  + searchList.strides.val[harm]*iy + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
                      }

                      //if ( (y+yPlus) == (searchList.heights.val[0]-1)/2 )
                      if ( (y+yPlus) == 0 )
                      {
                        powers[step][yPlus]        += power;
                      }
                      //powers[step][yPlus]        += power;
                    }
                    else
                    {
                      fcomplexcu cmpc;
                      if        ( FLAGS & FLAG_STP_ROW )
                      {
                        //cmpc = searchList.datas.val[harm][ inds[step][harm]  + searchList.strides.val[harm]*noSteps*iy + searchList.strides.val[harm]*step ] ;
                        cmpc = pData[step][harm][searchList.strides.val[harm]*iy] ; // Note stride has been set depending on multi-step type
                      }
                      else if   ( FLAGS & FLAG_STP_PLN )
                      {
                        cmpc = searchList.datas.val[harm][ inds[step][harm]  + searchList.strides.val[harm]*iy + searchList.strides.val[harm]*step*searchList.heights.val[harm] ] ;
                      }

                      if ( (y+yPlus) == (zeroHeight-1)/2.0 )
                      //if ( (y+yPlus) == 0 )
                      {
                        powers[step][yPlus]        += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                      }
                    }
                  }
                }
              }
            }
          }

          // Get max
          for ( int step = 0; step < noSteps; step++)             // Loop over steps
          {
            //#pragma unroll
            for( int i = 0; i < nPowers ; i++ )                   // Loop over section
            {
              if ( powers[step][i] > candLists[step] )
              {
                if ( y + i < zeroHeight )
                {
                  // This is our new max!
                  candLists[step]  = powers[step][i];
                }
              }
            }
          }
        }
      }
    }

    if ( FLAGS & CU_OUTP_SINGLE )
    {
#if TEMPLATE_SEARCH == 1
#pragma unroll
#endif
      for ( int step = 0; step < noSteps; step++)             // Loop over steps
      {
        // Write to DRAM
        d_cands[step*oStride + tid] = candLists[step];
      }
    }
  }
}

template<uint FLAGS, /*typename sType,*/ uint noStages>
__host__ void add_and_maxCU31_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream,cuSearchList searchList, float* d_cands, uint* d_sem, int base,  float* rLows, int noSteps)
{
#if TEMPLATE_SEARCH == 1
  switch (noSteps)
  {
    case 1:
    {
      //cudaFuncSetCacheConfig(add_and_maxCU31<FLAGS,sType,noStages,f01,1>, cudaFuncCachePreferL1);
      f01 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_maxCU31<FLAGS,/*sType,*/noStages,f01,1><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 2:
    {
      //cudaFuncSetCacheConfig(add_and_maxCU31<FLAGS,sType,noStages,f02,2>, cudaFuncCachePreferL1);
      f02 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_maxCU31<FLAGS,/*sType,*/noStages,f02,2><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 3:
    {
      //cudaFuncSetCacheConfig(add_and_maxCU31<FLAGS,sType,noStages,f03,3>, cudaFuncCachePreferL1);
      f03 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_maxCU31<FLAGS,/*sType,*/noStages,f03,3><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 4:
    {
      //cudaFuncSetCacheConfig(add_and_maxCU31<FLAGS,sType,noStages,f04,4>, cudaFuncCachePreferL1);
      f04 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_maxCU31<FLAGS,/*sType,*/noStages,f04,4><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 5:
    {
      //cudaFuncSetCacheConfig(add_and_maxCU31<FLAGS,sType,noStages,f05,5>, cudaFuncCachePreferL1);
      f05 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_maxCU31<FLAGS,/*sType,*/noStages,f05,5><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 6:
    {
      //cudaFuncSetCacheConfig(add_and_maxCU31<FLAGS,sType,noStages,f06,6>, cudaFuncCachePreferL1);
      f06 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_maxCU31<FLAGS,/*sType,*/noStages,f06,6><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 7:
    {
      //cudaFuncSetCacheConfig(add_and_maxCU31<FLAGS,sType,noStages,f07,7>, cudaFuncCachePreferL1);
      f07 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_maxCU31<FLAGS,/*sType,*/noStages,f07,7><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    case 8:
    {
      //cudaFuncSetCacheConfig(add_and_maxCU31<FLAGS,sType,noStages,f08,8>, cudaFuncCachePreferL1);
      f08 tmpArr;
      for (int i = 0; i < noSteps; i++)
        tmpArr.arry[i] = rLows[i];
      add_and_maxCU31<FLAGS,/*sType,*/noStages,f08,8><<<dimGrid,  dimBlock, i1, cnvlStream >>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr);
      break;
    }
    default:
      fprintf(stderr, "ERROR: add_and_maxCU31 has not been templated for %i steps\n", noSteps);
      exit(EXIT_FAILURE);
  }
#else
  //cudaFuncSetCacheConfig(add_and_maxCU31<FLAGS,sType,noStages,fMax>, cudaFuncCachePreferL1);
  fMax tmpArr;
  for (int i = 0; i < noSteps; i++)
    tmpArr.arry[i] = rLows[i];

  add_and_maxCU31<FLAGS,/*sType,*/noStages,fMax> <<<dimGrid, dimBlock, i1, cnvlStream>>>(searchList, d_cands, d_sem, base, /*pd,*/ tmpArr,noSteps);
#endif
}

template<uint FLAGS >
__host__ void add_and_maxCU31_p(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream,cuSearchList searchList, float* d_cands, uint* d_sem, int base, float* rLows, int noSteps, const uint noStages )
{
  switch (noStages)
  {
    case 1:
    {
      add_and_maxCU31_s<FLAGS,/*sch1,*/1> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, /*tmpArr,*/ rLows, noSteps );
      break;
    }
    case 2:
    {
      add_and_maxCU31_s<FLAGS,/*sch2,*/2> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, /*tmpArr,*/ rLows, noSteps );
      break;
    }
    case 3:
    {
      add_and_maxCU31_s<FLAGS,/*sch4,*/3> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, /*tmpArr,*/ rLows, noSteps );
      break;
    }
    case 4:
    {
      add_and_maxCU31_s<FLAGS,/*sch8,*/4> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, /*tmpArr,*/ rLows, noSteps );
      break;
    }
    case 5:
    {
      add_and_maxCU31_s<FLAGS,/*sch16,*/5> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, /*tmpArr,*/ rLows, noSteps );
      break;
    }
    default:
      fprintf(stderr, "ERROR: add_and_maxCU31 has not been templated for %i stages\n", noStages);
      exit(EXIT_FAILURE);
  }
}

__host__ void add_and_maxCU31_f(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream,cuSearchList searchList, float* d_cands, uint* d_sem, int base, float* rLows, int noSteps, const uint noStages, uint FLAGS )
{
  if        ( FLAGS & FLAG_CUFFTCB_OUT )
  {
    if      ( FLAGS & FLAG_STP_ROW )
      add_and_maxCU31_p<FLAG_CUFFTCB_OUT | CU_OUTP_SINGLE | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, rLows, noSteps, noStages );
    //else if ( FLAGS & FLAG_STP_PLN )
    //  add_and_maxCU31_p<FLAG_CUFFTCB_OUT | CU_OUTP_SINGLE | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, pd, rLows, noSteps, noStages );
    //else if ( FLAGS & FLAG_STP_STK )
    //  add_and_maxCU31_p<FLAG_CUFFTCB_OUT | CU_OUTP_SINGLE | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, pd, rLows, noSteps, noStages );
    else
    {
      fprintf(stderr, "ERROR: add_and_maxCU31 has not been templated for flag combination. \n" );
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    if      ( FLAGS & FLAG_STP_ROW )
      add_and_maxCU31_p<CU_OUTP_SINGLE | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, rLows, noSteps, noStages );
    //else if ( FLAGS & FLAG_STP_PLN )
    //  add_and_maxCU31_p<CU_OUTP_SINGLE | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, pd, rLows, noSteps, noStages );
    //else if ( FLAGS & FLAG_STP_STK )
    //  add_and_maxCU31_p<CU_OUTP_SINGLE | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, pd, rLows, noSteps, noStages );
    else
    {
      fprintf(stderr, "ERROR: add_and_maxCU31 has not been templated for flag combination. \n" );
      exit(EXIT_FAILURE);
    }
  }
}

/** Sum and Search - loop down - column max - use blocks .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base
 */
template<int noStages, int canMethoud>
__global__ void add_and_searchCU4(cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base)
{
  const int x   = threadIdx.x;
  const int y   = threadIdx.y;
  const int gx  = blockIdx.x*SS4_X + x;

  if ( gx < searchList.widths.val[0] )
  {
    int batches = searchList.heights.val[0] / (float)( SS4_Y );

    const int noHarms = (1 << (noStages - 1));
    int inds[noHarms];
    int start, end;

    float powerThread[noStages];
    int   z[noStages];

    for ( int stage = 0; stage < noStages; stage++ )
    {
      powerThread[stage]  = 0;
      z[stage]            = 0;
    }

    // Initialise the x indices of this thread
    inds[0] = gx + searchList.ffdBuffre.val[0];

    // Calculate the x indices
#pragma unroll
    for ( int i = 1; i < noHarms; i++ )
    {
      //inds[i]     = (int)(gx*searchList.frac.val[i]+searchList.idxSum.val[i]) + searchList.ffdBuffre.val[i];
    }

    for ( int b = 0;  b < batches;  b++)  // Loop over blocks
    {
      float blockPower = 0;
      int by = b*SS4_Y + y;

#pragma unroll
      for ( int stage = 0; stage < noStages; stage++ ) // Loop over harmonic stages
      {
        if      ( stage == 0 )
        {
          start = 0;
          end = 1;
        }
        else if ( stage == 1 )
        {
          start = 1;
          end = 2;
        }
        else if ( stage == 2 )
        {
          start = 2;
          end = 4;
        }
        else if ( stage == 3 )
        {
          start = 4;
          end = 8;
        }
        else if ( stage == 4 )
        {
          start = 8;
          end = 16;
        }

        // Sum set of powers
#pragma unroll
        for ( int harm = start; harm < end; harm++ ) // Loop over sub harmonics
        {
          if  ( (canMethoud & FLAG_PLN_TEX ) )
          {
            const float2 cmpf = tex2D < float2 > (searchList.texs.val[harm], inds[harm], YINDS[searchList.yInds.val[harm]+by]);
            blockPower += cmpf.x * cmpf.x + cmpf.y * cmpf.y;
          }
          else
          {
            const fcomplexcu cmpc  = searchList.datas.val[harm][YINDS[searchList.yInds.val[harm]+by]*searchList.strides.val[harm]+inds[harm]];
            blockPower += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
          }
        }

        if  (  blockPower >  POWERCUT[stage] )
        {
          if ( blockPower > powerThread[stage] )
          {
            powerThread[stage]  = blockPower;
            z[stage]            = b;
          }
        }
      }
    }

#pragma unroll
    for ( int stage = 0; stage < 1; stage++ ) // Loop over harmonic stages
    {
      accelcandBasic can;
      long long numtrials         = NUMINDEP[stage];
      const short numharm = 1 << stage;

      if  ( powerThread[stage] >  POWERCUT[stage] )
      {
        if ( canMethoud & CU_OUTP_SINGLE )
        {
          can.numharm = numharm;
          can.sigma   = powerThread[0];
          can.z       = z[0];
          if ( canMethoud & FLAG_SAS_SIG )
          {
            // Calculate sigma value
            can.sigma   = (float)candidate_sigma_cu(powerThread[0], numharm, numtrials);
          }

          // Write to DRAM
          d_cands[ searchList.widths.val[0]*stage*y +  stage*searchList.widths.val[0] + gx ] = can;
        }
      }
    }
  }

  /*

  __shared__ float s_powers[noStages][SS4_Y][SS4_X];
  __shared__ uint  s_z[noStages][SS4_Y][SS4_X];
  __shared__ int sum[noStages];

  if (x < noStages && y == 0)
  {
    sum[x] = 0;
  }

  // __syncthreads();

  // Write all results to shard memory
  for ( int s = 0 ; s <  noStages; s++)
  {
    if (powerThread[s] > 0 )
    {
      s_powers[s][y][x]  = powerThread[s];
      s_z[s][y][x]       = z[s] ; // *SS4_Y+y;
      atomicAdd(&sum[s], 1);
    }
  }

  __syncthreads();

  // Write back to DRAM
  if ( y < noStages && sum[y] > 0 )
  {
    z[0] = 0;
    powerThread[0] = 0;
    int stage = y;

    for ( int by = 0 ; by < SS4_Y; by++ )
    {
      if( s_powers[stage][by][x] > powerThread[0] )
      {
        powerThread[0]  = s_powers[stage][by][x];
        z[0]            = s_z[stage][by][x]*SS4_Y + by;
      }
    }

    if  ( powerThread[0] >  POWERCUT[stage] )
    {
      accelcandBasic can;
      long long numtrials         = NUMINDEP[stage];
      const short numharm = 1 << stage;

      // Write results back to DRAM and calculate sigma if needed
      if      ( canMethoud & CU_OUTP_DEVICE   )
      {
        int idx =  (int)(( searchList.rLow.val[0] + gx * (double) ACCEL_DR ) / (double)numharm ) - base ;
        if ( idx >= 0 )
        {
          can.numharm = numharm;
          can.sigma   = (float)candidate_sigma_cu(powerThread[0], numharm, numtrials);
          can.z       = ( z[0]*(float) ACCEL_DZ - searchList.zMax.val[0]  )  / (float)numharm ;

          FOLD // Atomic write to global list
          {
            volatile bool done = false;
            while (!done)
            {
              volatile int prev = atomicCAS(&d_sem[idx], UINT_MAX, gx );
              if ( prev == UINT_MAX )
              {
                if ( can.sigma > d_cands[idx].sigma )
                {
                  d_cands[idx]   = can;
                }
                d_sem[idx]      = UINT_MAX;
                done            = true;
              }
            }
          }
        }
      }
      else if ( canMethoud & CU_OUTP_SINGLE )
      {
        can.numharm = numharm;
        can.sigma   = powerThread[0];
        can.z       = z[0];
        if ( canMethoud & FLAG_SAS_SIG )
        {
          // Calculate sigma value
          can.sigma   = (float)candidate_sigma_cu(powerThread[0], numharm, numtrials);
        }

        // Write to DRAM
        d_cands[gx*noStages + stage] = can;
      }
    }
  }
   */
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
        for (int j = 0; j< batch->hInfos[0].height; j++)
        {
          int zz    = -batch->hInfos[0].zmax+ j* ACCEL_DZ;
          int subz  = calc_required_z(batch->hInfos[ii].harmFrac, zz);
          int zind  = index_from_z(subz, -batch->hInfos[ii].zmax);
          if (zind< 0|| zind>= batch->hInfos[ii].height)
          {
            int Err = 0;
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
}

void sumAndSearch(cuFFdotBatch* batch, long long *numindep, GSList** cands)
{
  //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  //printf("Sum & Search\n");

  dim3 dimBlock, dimGrid; 

  nvtxRangePush("Add & Search");

  if ( batch->haveSearchResults || batch->haveConvData ) // previous plain has data data so sum and search
  {
    int noStages = log(batch->noHarms)/log(2) + 1;
    int harmtosum;
    cuSearchList searchList;      // The list of details of all the individual batch
    //cuSearchItem* pd;
    float *rLows;
    //pd = (cuSearchItem*)malloc(batch->noHarms * sizeof(cuSearchItem));

    rLows = (float*)malloc(batch->noSteps * sizeof(float));

    FOLD // Do synchronisations  .
    {
      for (int ss = 0; ss< batch->noStacks; ss++)
      {
        cuFfdotStack* cStack = &batch->stacks[ss];

        cudaStreamWaitEvent(batch->strmSearch, cStack->plnComp, 0);
      }
    }

    if ( batch->haveConvData ) // Sum & search  .
    {
      FOLD // Create search list to pass to kernel  .
      {
        int i = 0;
        for (int stage = 0; stage < noStages; stage++)
        {
          harmtosum = 1 << stage;

          for (int harm = 1; harm <= harmtosum; harm += 2)
          {
            float fract = 1-harm/ float(harmtosum);
            int idx = round(fract* batch->noHarms);
            if ( fract == 1 )
              idx = 0;

            searchList.texs.val[i]      = batch->plains[idx].datTex;
            searchList.datas.val[i]     = batch->plains[idx].d_plainData;
            searchList.powers.val[i]    = batch->plains[idx].d_plainPowers;
            searchList.frac.val[i]      = batch->hInfos[idx].harmFrac;
            searchList.yInds.val[i]     = batch->hInfos[idx].yInds;
            searchList.heights.val[i]   = batch->hInfos[idx].height;
            searchList.strides.val[i]   = batch->hInfos[idx].inpStride;
            searchList.ffdBuffre.val[i] = batch->hInfos[idx].halfWidth*ACCEL_NUMBETWEEN;
            searchList.zMax.val[i]      = batch->hInfos[idx].zmax;

            searchList.widths.val[i]    = (*batch->rSearch)[0][idx].numrs;
            searchList.rLow.val[i]      = (*batch->rSearch)[0][idx].drlo; // batch->batch[idx].rLow[0];

            i++;
          }
        }
      }

      FOLD // Call the main sum & search kernel  .
      {
        /*
        if      (  batch->flag & CU_OUTP_DEVICE )
        {
          dimBlock.x = SS3_X;
          dimBlock.y = SS3_Y;

          float bw = SS3_X * SS3_Y;
          float ww = batch->batch[0].ffdotPowWidth / ( bw );

          dimGrid.x = ceil(ww);
          dimGrid.y = 1;

          if( batch->flag & FLAG_PLN_TEX )
          {
            if      ( noStages == 1 )
              add_and_searchCU3<1,CU_OUTP_DEVICE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
            else if ( noStages == 2 )
              add_and_searchCU3<2,CU_OUTP_DEVICE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
            else if ( noStages == 3 )
              add_and_searchCU3<3,CU_OUTP_DEVICE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
            else if ( noStages == 4 )
              add_and_searchCU3<4,CU_OUTP_DEVICE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
            else if ( noStages == 5 )
              add_and_searchCU3<5,CU_OUTP_DEVICE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
          }
          else
          {
            if      ( noStages == 1 )
              add_and_searchCU3<1,CU_OUTP_DEVICE><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
            else if ( noStages == 2 )
              add_and_searchCU3<2,CU_OUTP_DEVICE><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
            else if ( noStages == 3 )
              add_and_searchCU3<3,CU_OUTP_DEVICE><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
            else if ( noStages == 4 )
              add_and_searchCU3<4,CU_OUTP_DEVICE><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
            else if ( noStages == 5 )
              add_and_searchCU3<5,CU_OUTP_DEVICE><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );

          }
          batch->haveCData = 0;
        }
        else if ( (batch->flag & CU_OUTP_SINGLE) || (batch->flag & CU_OUTP_HOST) )
        {
          dimBlock.x = SS3_X;
          dimBlock.y = SS3_Y;

          float bw = SS3_X * SS3_Y;
          float ww = batch->batch[0].ffdotPowWidth / ( bw );

          dimGrid.x = ceil(ww);
          dimGrid.y = 1;

          if( batch->flag & FLAG_PLN_TEX )
          {
            if ( batch->flag & FLAG_SAS_SIG )
            {
              if      ( noStages == 1 )
                add_and_searchCU3<1,CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 2 )
                add_and_searchCU3<2,CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 3 )
                add_and_searchCU3<3,CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 4 )
                add_and_searchCU3<4,CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 5 )
                add_and_searchCU3<5,CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
            }
            else
            {
              if      ( noStages == 1 )
                add_and_searchCU3<1,CU_OUTP_SINGLE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 2 )
                add_and_searchCU3<2,CU_OUTP_SINGLE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 3 )
                add_and_searchCU3<3,CU_OUTP_SINGLE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 4 )
                add_and_searchCU3<4,CU_OUTP_SINGLE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 5 )
                add_and_searchCU3<5,CU_OUTP_SINGLE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
            }
          }
          else
          {
            if ( batch->flag & FLAG_SAS_SIG )
            {
              if      ( noStages == 1 )
                add_and_searchCU3<1,CU_OUTP_SINGLE | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 2 )
                add_and_searchCU3<2,CU_OUTP_SINGLE | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 3 )
                add_and_searchCU3<3,CU_OUTP_SINGLE | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 4 )
                add_and_searchCU3<4,CU_OUTP_SINGLE | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 5 )
                add_and_searchCU3<5,CU_OUTP_SINGLE | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
            }
            else
            {
              if      ( noStages == 1 )
                add_and_searchCU3<1,CU_OUTP_SINGLE><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 2 )
                add_and_searchCU3<2,CU_OUTP_SINGLE><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 3 )
                add_and_searchCU3<3,CU_OUTP_SINGLE><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 4 )
                add_and_searchCU3<4,CU_OUTP_SINGLE><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 5 )
                add_and_searchCU3<5,CU_OUTP_SINGLE><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
            }
          }

                  }
        else if (  batch->flag & CU_OUTP_SINGLE )
        {
          dimBlock.x = SS3_X;
          dimBlock.y = SS3_Y;

          float bw = SS3_X * SS3_Y;
          float ww = batch->batch[0].ffdotPowWidth / ( bw );

          dimGrid.x = ceil(ww);
          dimGrid.y = 1;

          if ( batch->flag & FLAG_PLN_TEX )
          {
            if ( batch->flag & FLAG_SAS_SIG )
            {
              if      ( noStages == 1 )
                add_and_searchCU3<1,CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 2 )
                add_and_searchCU3<2,CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 3 )
                add_and_searchCU3<3,CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 4 )
                add_and_searchCU3<4,CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 5 )
                add_and_searchCU3<5,CU_OUTP_SINGLE | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
            }
            else
            {
              if      ( noStages == 1 )
                add_and_searchCU3<1,CU_OUTP_SINGLE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 2 )
                add_and_searchCU3<2,CU_OUTP_SINGLE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 3 )
                add_and_searchCU3<3,CU_OUTP_SINGLE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 4 )
                add_and_searchCU3<4,CU_OUTP_SINGLE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 5 )
                add_and_searchCU3<5,CU_OUTP_SINGLE><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
            }
          }
          else
          {
            if ( batch->flag & FLAG_SAS_SIG )
            {
              if      ( noStages == 1 )
                add_and_searchCU3<1,CU_OUTP_SINGLE | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 2 )
                add_and_searchCU3<2,CU_OUTP_SINGLE | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 3 )
                add_and_searchCU3<3,CU_OUTP_SINGLE | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 4 )
                add_and_searchCU3<4,CU_OUTP_SINGLE | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 5 )
                add_and_searchCU3<5,CU_OUTP_SINGLE | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
            }
            else
            {
              if      ( noStages == 1 )
                add_and_searchCU3<1,CU_OUTP_SINGLE><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 2 )
                add_and_searchCU3<2,CU_OUTP_SINGLE><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 3 )
                add_and_searchCU3<3,CU_OUTP_SINGLE><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 4 )
                add_and_searchCU3<4,CU_OUTP_SINGLE><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
              else if ( noStages == 5 )
                add_and_searchCU3<5,CU_OUTP_SINGLE><<<dimGrid, dimBlock, 0, batch->strmSearch>>>(searchList, batch->d_retData, batch->d_candSem, batch->rLow );
            }
          }
        }
         */

        if ( (batch->flag & CU_OUTP_SINGLE) || (batch->flag & CU_OUTP_HOST) ) // Call the templated kernel  .
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
            if ( (batch->flag&FLAG_CUFFTCB_OUT) && (batch->flag&FLAG_PLN_TEX) )
            {
              add_and_searchCU3_PT_f(dimGrid, dimBlock, batch->strmSearch, batch );
            }
          }
          else
          {
            fprintf(stderr,"ERROR: function %s is not setup to handle this type of return data for GPU accel search\n",__FUNCTION__);
            exit(EXIT_FAILURE);
          }
        }
        else
        {
          fprintf(stderr,"ERROR: function %s is not setup to handle this type of output for GPU accel search\n",__FUNCTION__);
          exit(EXIT_FAILURE);
        }

        CUDA_SAFE_CALL(cudaGetLastError(), "Error at add_and_searchCU31 kernel launch");

        // Event
        CUDA_SAFE_CALL(cudaEventRecord(batch->searchComp,  batch->strmSearch),"Recording event: searchComp");
      }
    }

    if ( batch->haveSearchResults ) // Process previous results  .
    {
      if ( batch->flag & CU_OUTP_SINGLE )
      {
        // A blocking synchronisation to ensure results are ready to be proceeded by the host
        CUDA_SAFE_CALL(cudaEventSynchronize(batch->candCpyComp), "ERROR: copying result from device to host.");

        nvtxRangePush("CPU Process results");

        batch->noResults=0;

        double poww, sig, sigx, sigc;
        double rr, zz;
        int added = 0;
        int numharm;
        poww = 0;

        FOLD
        {
#pragma omp critical
          for ( int step = 0; step < batch->noSteps; step++)         // Loop over steps
          {
            for ( int stage = 0; stage < noStages; stage++ )
            {
              //for ( int x = 0; x < batch->plains->numrs[step]; x++ )
              for ( int x = 0; x < batch->accelLen; x++ )
              {
                int idx   = step*noStages*batch->hInfos->width + stage*batch->hInfos->width + x ;

                if ( batch->retType & CU_SMALCAND )
                {
                  accelcandBasic candB  = ((accelcandBasic*)batch->h_retData)[idx] ;
                  poww                  = candB.sigma ;

                  if ( poww > 0 ) 
                  {
                    batch->noResults++;

                    numharm   = (1<<stage);

                    if ( batch->flag & FLAG_SAS_SIG )
                      sig     = poww;
                    else
                      sig     = candidate_sigma(poww, numharm, numindep[stage]);

                    rr = ( batch->plains[0].searchRlowPrev[step] + x *  ACCEL_DR );

                    if ( rr > batch->SrchSz->searchRHigh )
                    {
                      int tmp = 0;
                    }
                    
                    if ( rr < batch->SrchSz->searchRHigh )
                    {
                      rr /= (double)numharm ;
                      zz = ( candB.z * ACCEL_DZ - batch->hInfos[0].zmax )              / (double)numharm ;

                      if ( rr >= 400000 )
                      {
                        //printf("GPU r %.2f z %.1f \n", rr, zz);
                      }

                      if      ( batch->flag & CU_CAND_LST )
                      {
                        added = 0;
                        //*cands = insert_new_accelcand(*cands, poww, sig, numharm, rr, zz, &added);
                      }
                      else if ( batch->flag & CU_CAND_ARR )
                      {
                        double rDiff = rr - batch->SrchSz->searchRLow ;
                        long grIdx;   /// The index of the candidate in the global list

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

                          if ( batch->flag & FLAG_STORE_ALL )								// Store all stages
                          {
                            grIdx += stage * (batch->SrchSz->noOutpR); // Stride by size
                          }

                          if ( batch->cndType == CU_FULLCAND )
                          {
                            cand* candidate = &((cand*)batch->h_candidates)[grIdx];

                            // this sigma is greater that the current sigma for this r value
                            if ( candidate->sig < sig )
                            {
                              candidate->sig      = sig;
                              candidate->power    = poww;
                              candidate->numharm  = numharm;
                              candidate->r        = rr;
                              candidate->z        = zz;
                              added = 1;
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

        nvtxRangePop();

        // Do some Synchronisation
        CUDA_SAFE_CALL(cudaEventRecord(batch->processComp, batch->strmSearch),"Recording event: searchComp");

        batch->haveSearchResults = 0;
      }
    }

    // Copy results from device to host
    if ( (batch->flag & CU_OUTP_SINGLE) || (batch->flag & CU_OUTP_HOST) )
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

void sumAndMax(cuFFdotBatch* batch, long long *numindep, float* powers)
{
  //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  dim3 dimBlock, dimGrid;

  nvtxRangePush("Add & Max");

  if ( batch->haveSearchResults || batch->haveConvData ) // previous plain has data data so sum and search  .
  {
    int noStages = log(batch->noHarms)/log(2) + 1;
    int harmtosum;
    cuSearchList searchList;      // The list of details of all the individual batch
    //cuSearchItem* pd;
    float *rLows;
    //pd = (cuSearchItem*)malloc(batch->noHarms * sizeof(cuSearchItem));
    rLows = (float*)malloc(batch->noSteps * sizeof(float));

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
      FOLD // Create search list
      {
        int i = 0;
        for (int stage = 0; stage < noStages; stage++)
        {
          harmtosum = 1 << stage;

          for (int harm = 1; harm <= harmtosum; harm += 2)
          {
            //printf("Stage  %i harm %i \n", stage, harm);

            float fract = 1-harm/ float(harmtosum);
            int idx = round(fract* batch->noHarms);
            if ( fract == 1 )
              idx = 0;

            searchList.texs.val[i]      = batch->plains[idx].datTex;
            searchList.datas.val[i]     = batch->plains[idx].d_plainData;
            searchList.powers.val[i]    = batch->plains[idx].d_plainPowers;
            searchList.frac.val[i]      = batch->hInfos[idx].harmFrac;
            searchList.yInds.val[i]     = batch->hInfos[idx].yInds;
            searchList.heights.val[i]   = batch->hInfos[idx].height;
            //searchList.widths.val[i]    = batch->batch[idx].ffdotPowWidth[0];
            searchList.strides.val[i]   = batch->hInfos[idx].inpStride;
            searchList.ffdBuffre.val[i] = batch->hInfos[idx].halfWidth*ACCEL_NUMBETWEEN;
            searchList.zMax.val[i]      = batch->hInfos[idx].zmax;
            //searchList.rLow.val[i]      = batch->batch[idx].rLow[0];

            searchList.widths.val[i]    = (*batch->rSearch)[0][idx].numrs;
            searchList.rLow.val[i]      = (*batch->rSearch)[0][idx].drlo; // batch->batch[idx].rLow[0];

            i++;
          }
        }
      }

      FOLD // Call the main sum & search kernel
      {
        if ( (batch->flag & CU_OUTP_SINGLE) || (batch->flag & CU_OUTP_HOST) ) // Call the templated kernel
        {
          dimBlock.x  = SS3_X;
          dimBlock.y  = SS3_Y;

          float bw    = SS3_X * SS3_Y;
          //float ww    = batch->batch[0].ffdotPowWidth[0] / ( bw );
          float ww    = batch->accelLen / ( bw );

          dimGrid.x   = ceil(ww);
          dimGrid.y   = 1;

          //add_and_maxCU31_f(dimGrid, dimBlock, 0, batch->strmSearch, searchList, (float*)batch->d_retData, batch->d_candSem, 0, pd, &batch->batch->rLow[0], batch->noSteps, batch->noHarmStages, batch->flag );
        }

        // Run message
        CUDA_SAFE_CALL(cudaGetLastError(), "Error at add_and_searchCU31 kernel launch");

        CUDA_SAFE_CALL(cudaEventRecord(batch->searchComp,  batch->strmSearch),"Recording event: searchComp");
      }
    }

    if ( batch->haveSearchResults ) // Process previous results  .
    {
      if ( batch->flag & CU_OUTP_SINGLE )
      {
        // A blocking synchronisation to ensure results are ready to be proceeded by the host
        CUDA_SAFE_CALL(cudaEventSynchronize(batch->candCpyComp), "ERROR: copying result from device to host.");

        nvtxRangePush("CPU Process results");

        for ( int step = 0; step < batch->noSteps; step++ )
        {
          int gIdx = batch->plains[0].searchRlowPrev[step] ;

          if ( batch->flag & FLAG_STORE_EXP )
            gIdx =  ( batch->plains[0].searchRlowPrev[step] ) * ACCEL_RDR ;

          float* gWrite = (float*)batch->h_candidates + gIdx;
          float* pRead = (float*)(batch->h_retData) + batch->hInfos->width*step;

          //memcpy(gWrite, pRead, batch->plains->numrs[0]*sizeof(float));
          memcpy(gWrite, pRead, batch->accelLen*sizeof(float));
        }
        nvtxRangePop();

        // Do some Synchronisation
        CUDA_SAFE_CALL(cudaEventRecord(batch->processComp, batch->strmSearch),"Recording event: searchComp");

        batch->haveSearchResults = 0;
      }
    }

    // Copy results from device to host
    if ( (batch->flag & CU_OUTP_SINGLE) || (batch->flag & CU_OUTP_HOST) )
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
