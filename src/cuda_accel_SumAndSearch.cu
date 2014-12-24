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

extern "C"
{
//#define __float128 long double
//#include "accel.h"
}

#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"

__device__ __constant__ int        YINDS[MAX_YINDS];
__device__ __constant__ float      POWERCUT[MAX_HARM_NO];
__device__ __constant__ long long  NUMINDEP[MAX_HARM_NO];

/** Return x such that 2**x = n
 *
 * @param n
 * @return
 */
inline int twon_to_index(int n)
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
__device__ inline int getY(int plainY, const int noSteps,  const int step, const int plainHeight = 0 /*, const int stackHeight = 0*/ )
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

template<int n>
__host__ __device__ void cdfgam_d(double x, double *p, double* q)
{
  if      ( n == 1 )
  {
    *q = exp(-x);
  }
  else if ( n == 2 )
  {
    *q = exp(-x)*( x + 1.0 );
  }
  else if ( n == 4 )
  {
    *q = exp(-x)*( x*(x*(x/6.0 + 0.5) + 1.0 ) + 1.0 );
  }
  else if ( n == 8 )
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

/*
__device__ void search(const float powers[SS_Y_NUM][SS_X_NUM], int stage, cand* d_candList, cand* candListsm, int *count, float rlo, float zlo)
{
  int sx;     // Shared memory x location
  int sy;     // Shared memory y location

  // Specialize BlockScan for a 1D block of 128 threads on type int
  //typedef cub::BlockScan<int, SS_X, cub::BLOCK_SCAN_RAKING, SS_Y> BlockScan;

  // Allocate shared memory for BlockScan
  //__shared__ typename BlockScan::TempStorage temp_storage;

  //const uint tid = threadIdx.y * SS_X + threadIdx.x;

  const float powcut = POWERCUT[stage];
  float mypwer;

  //__shared__ uint ff;

  __shared__ unsigned long long int sem;
  //  __shared__ uint ff2;
  //__shared__ int mutex;
  //__shared__ int cnts[SS_X*SS_Y];

  //if (tid == 0)
  {
    //*count = 0;
    //ff2 = 0;
    //sem = 0;
  }

  //int start = 0;
  //int spin = 0;
  //cand lCands[SS_Y_TILES*SS_X_TILES];

  __syncthreads();

  // Read data from fundamental into shared memory
#ifndef DEBUG
#pragma unroll
#endif
  for (int y = 0; y < SS_Y_TILES; y++)
  {
    sy = y * SS_Y + threadIdx.y;

#ifndef DEBUG
#pragma unroll
#endif
    for (int x = 0; x < SS_X_TILES; x++)
    {
      sx = x * SS_X + threadIdx.x;
      if (sx > SS_X_OVERLAP && sy > SS_Y_OVERLAP && sx < SS_X_NUM - SS_X_OVERLAP && sy < SS_Y_NUM - SS_Y_OVERLAP)
      {
        mypwer = powers[sy][sx];
        int lmax = true;

        if (mypwer > powcut)
        {
          //if (lmax == true)
          {
            //float power =
            //float pow = powers[y][x];
            //float sig = candidate_sigma(pow, numharm, numindep);
            //float rr = (ffdot->rlo + jj * (double) ACCEL_DR) / (double) numharm;
            //float zz = (ffdot->zlo + ii * (double) ACCEL_DZ) / (double) numharm;
            //printf("Found a canidate! Power: %.3f  sig: %.4f \n",pow, sig);
            //          ff++;

            //while ( atomicCAS( &mutex, 0, 1 ) );

            //ff2++;
            //mutex = 0;
            //atomicExch( &mutex, 0 );

            //cnts[tid]++;
            //volatile
            //volatile
            //__syncthreads();
            //while (atomicCAS((unsigned long long int*) &sem, 0, tid + 1) != 0)
            //{
            //}

            //printf("Assuire sem %i \n", tid );

            //acquire_semaphore(&g_canSem);
            // candList[ff2].power =  mypwer;
            //candList[ff2].r = x;
            //candList[ff2].z = y;

            //__syncthreads();

            if (1)
            {
              if (0)
              {
                volatile bool done = false;

                //atomicAdd(&g_canCount, 1);
                // Global synch and add
                int cnt = 0;
                while (!done)
                {
                  volatile int prev = atomicCAS((unsigned long long int*) &g_canSem, SEMFREE, 1);
                  if (prev == SEMFREE )  // || cnt > 5000)
                  {
                    if (g_canCount < MAX_CANDS_PER_BLOCK)
                    {
                      d_candList[g_canCount].power = mypwer;
                      d_candList[g_canCount].r = (rlo + sx * (double) ACCEL_DR);
                      d_candList[g_canCount].z = (zlo + sy * (double) ACCEL_DZ);
                      d_candList[g_canCount].numharm = stage;
                      g_canCount++;
                    }
                    else
                    {
                      printf("/rCUDA Candidate list to short, not recording candidate!  lost: %i candidates.\n", g_canCount-MAX_CANDS_PER_BLOCK);
                      //g_canCount++;
                    }
                    //printf("Found a canidate! Power: %.3f  %i  \n", mypwer, g_canCount);
                    //ff2++;
                    done = true;
                    g_canSem = SEMFREE;
                  }
                  //cnt++;
                  //__syncthreads();
                }
              }
              else
              {
                //int cnt = 0;
                // Shared synch and add
                //while (!done && cnt < 5000)
                {
                  //volatile int prev = atomicCAS((unsigned long long int*) &sem, 0, 1);
                  //prev = 0;
                  //if (prev == 0)
                  //if ( (*count) < SS_X_TILES*SS_Y_TILES*5 )
                  if ( true )
                  {
                    atomicAdd((unsigned long long int*) &g_canCount, 1);

                    candListsm[*count].power = mypwer;
                    candListsm[*count].r = (rlo + sx * (double) ACCEL_DR);
                    candListsm[*count].z = (zlo + sy * (double) ACCEL_DZ);
                    candListsm[*count].numharm = stage;
                    (*count)++;

                    //ff2++;
                    //done = true;
                    //sem = 0;
                  }
                  //else
                  {
                    //printf("Well this is strange?\n");
                  }
                  //cnt++;
                  //__syncthreads();
                }
              }
            }
            else
            {
              //atomicAdd((unsigned long long int*) &g_canSem, 1);
              //atomicAdd((unsigned long long int*) &sem, 1);
            }

            //__syncthreads();
            //atomicAdd(&ff, 1);
          }
          //else
          {
            //powers[sy][sx] = 0;
          }
        }
        else
        {
          //powers[sy][sx] = 0;
        }
      }
    }
  }
  //if ( ff )
  //printf("Found %i canidates!\n", ff);

  //__syncthreads();

  // Collectively compute the block-wide exclusive prefix sum
  //BlockScan(temp_storage).ExclusiveSum(count, start);

  //__syncthreads();
  //__threadfence();

  //if (threadIdx.x == (SS_X-1) && threadIdx.y == (SS_Y-1) && start > 0 )
  //  printf("Found %i canidates!\n", start);

   //if (tid == (SS_X) * (SS_Y) && *count > 0)
   //printf("Found %i  %i  %i  canidates!\n", *count, (int) sem, spin);

  //for( int i = 0; i < count; i++ )
  {
    //candList[start+i] = lCands[i];
  }
}
*/

/*
template<int stage>
__device__ static inline void searcht(const float powers_t[SS_Y_TILES][SS_X_TILES], cand* d_candList, cand* candListsm, int *count, double rlo, float zlo, int tlx, int tly)
{
  int sx;     // Shared memory x location
  int sy;     // Shared memory y location

  // Specialize BlockScan for a 1D block of 128 threads on type int
  //typedef cub::BlockScan<int, SS_X, cub::BLOCK_SCAN_RAKING, SS_Y> BlockScan;

  // Allocate shared memory for BlockScan
  //__shared__ typename BlockScan::TempStorage temp_storage;

  const uint tid = threadIdx.y * SS_X + threadIdx.x;

  const float powcut = POWERCUT[stage];
  float mypwer;

  //__shared__ uint ff;

  __shared__ unsigned long long int sem;
  //  __shared__ uint ff2;
  //__shared__ int mutex;
  //__shared__ int cnts[SS_X*SS_Y];

  //if (tid == 0)
  {
    //*count = 0;
    //ff2 = 0;
    //sem = 0;
  }

  //int start = 0;
  //int spin = 0;
  //cand lCands[SS_Y_TILES*SS_X_TILES];

  //__syncthreads();

  // Read data from fundamental into shared memory
#ifndef DEBUG
#pragma unroll
#endif
  for (int y = 0; y < SS_Y_TILES; y++)
  {
    //if ( piy >= 0 && piy < searchList.height.val[0] )
    {
#ifndef DEBUG
#pragma unroll
#endif
      for (int x = 0; x < SS_X_TILES; x++)
      {

        //if (sx > SS_X_OVERLAP && sy > SS_Y_OVERLAP && sx < SS_X_NUM - SS_X_OVERLAP && sy < SS_Y_NUM - SS_Y_OVERLAP)

        //if ( pix >= 0 && pix < searchList.widths.val[0] )
        {
          //mypwer = powers[sy][sx];
          mypwer = powers_t[y][x];
          //int lmax = true;

          if (mypwer > powcut)
          {


            //if (lmax == true)
            {
              //float power =
              //float pow = powers[y][x];
              //float sig = candidate_sigma(pow, numharm, numindep);
              //float rr = (ffdot->rlo + jj * (double) ACCEL_DR) / (double) numharm;
              //float zz = (ffdot->zlo + ii * (double) ACCEL_DZ) / (double) numharm;
              //printf("Found a canidate! Power: %.3f  sig: %.4f \n",pow, sig);
              //          ff++;

              //while ( atomicCAS( &mutex, 0, 1 ) );

              //ff2++;
              //mutex = 0;
              //atomicExch( &mutex, 0 );

              //cnts[tid]++;
              //volatile
              //volatile
              //__syncthreads();
              //while (atomicCAS((unsigned long long int*) &sem, 0, tid + 1) != 0)
              //{
              //}

              //printf("Assuire sem %i \n", tid );

              //acquire_semaphore(&g_canSem);
              // candList[ff2].power =  mypwer;
              //candList[ff2].r = x;
              //candList[ff2].z = y;

              //__syncthreads();

              if (1)
              {
                if (0)
                {
                  volatile bool done = false;

                  //atomicAdd(&g_canCount, 1);
                  // Global synch and add
                  int cnt = 0;
                  while (!done)
                  {
                    volatile int prev = atomicCAS((unsigned long long int*) &g_canSem, SEMFREE, 1);
                    if (prev == SEMFREE )  // || cnt > 5000)
                    {
                      if (g_canCount < MAX_CANDS_PER_BLOCK)
                      {
                        int numharm = 1 << stage;
                        d_candList[g_canCount].power = mypwer;
                        //(ffdot->rlo + jj * (double) ACCEL_DR) / (double) numharm;
                        //zz = plains->h_candidates[i].z;
                        //(ffdot->zlo + ii * (double) ACCEL_DZ) / (double) numharm;

                        d_candList[g_canCount].r = (rlo + sx * (float) ACCEL_DR ) / (float) (numharm);
                        d_candList[g_canCount].z = (zlo + sy * (float) ACCEL_DZ ) / (float) (numharm);
                        d_candList[g_canCount].numharm = numharm;
                        g_canCount++;
                      }
                      else
                      {
                        printf("/rCUDA Candidate list to short, not recording candidate!  lost: %i candidates.\n", g_canCount-MAX_CANDS_PER_BLOCK);
                        //g_canCount++;
                      }
                      //printf("Found a canidate! Power: %.3f  %i  \n", mypwer, g_canCount);
                      //ff2++;
                      done = true;
                      g_canSem = SEMFREE;
                    }
                    //cnt++;
                    //__syncthreads();
                  }
                }
                else
                {
                  //int cnt = 0;
                  // Shared synch and add
                  //while (!done && cnt < 5000)
                  {
                    //volatile int prev = atomicCAS((unsigned long long int*) &sem, 0, 1);
                    //prev = 0;
                    //if (prev == 0)
                    //if ( (*count) < SS_X_TILES*SS_Y_TILES*5 )
                    if ( true )
                    {
                      //atomicAdd(s_count, 1);

                      sy = y * SS_Y + threadIdx.y;
                      sx = x * SS_X + threadIdx.x;

                      int piy = tly + sy;
                      int pix = tlx + sx;

                      int numharm = 1 << stage;
                      candListsm[*count].power = mypwer;
                      candListsm[*count].r = (ACCEL_RDR * (rlo + pix * (float) ACCEL_DR) / (float)numharm + 0.5) * ACCEL_DR;
                      //(rlo + pix * (float) ACCEL_DR) / (float) (numharm);
                      //return (int) (ACCEL_RDR * (rlo + pix * (float) ACCEL_DR) * harm_fract + 0.5) * ACCEL_DR;
                      candListsm[*count].z = (zlo + piy * (float) ACCEL_DZ) / (float) (numharm);
                      candListsm[*count].numharm = numharm;

                      (*count)++;

                      //printf("Found a candidate! Power: %.3f  \n", mypwer);

                      //ff2++;
                      //done = true;
                      //sem = 0;
                    }
                    //else
                    {
                      //printf("Well this is strange?\n");
                    }
                    //cnt++;
                    //__syncthreads();
                  }
                }
              }
              else
              {
                //atomicAdd((unsigned long long int*) &g_canSem, 1);
                //atomicAdd((unsigned long long int*) &sem, 1);
              }

              //__syncthreads();
              //atomicAdd(&ff, 1);
            }
            //else
            {
              //powers[sy][sx] = 0;
            }
          }
          else
          {
            //powers[sy][sx] = 0;
          }

        }
      }
    }
  }
  //if ( ff )
  //printf("Found %i canidates!\n", ff);

  //__syncthreads();

  // Collectively compute the block-wide exclusive prefix sum
  //BlockScan(temp_storage).ExclusiveSum(count, start);

  //__syncthreads();
  //__threadfence();

  //if (threadIdx.x == (SS_X-1) && threadIdx.y == (SS_Y-1) && start > 0 )
  //  printf("Found %i canidates!\n", start);

  //for( int i = 0; i < count; i++ )
  {
    //candList[start+i] = lCands[i];
  }
}
*/

/*
__device__ void sumPlainsSm(float powers[SS_Y_NUM][SS_X_NUM], cuFfdot10 others, int stage, float fRlow, float fZlow, int tlx, int tly)
{
  int tId = threadIdx.y * SS_X + threadIdx.x;       // Index in thread block

  const int threadsInB = SS_X * SS_Y;//
  int batch = ceilf(SS_X_NUM * SS_Y_NUM / (float) (threadsInB));

  int pix;// X location in powers data
  int piy;// Y location in powers data

  int sx;// X location in Shared memory
  int sy;// Y location in Shared memory

  int cmplxX;// X location in the complex data
  int cmplxY;// Y location in the complex data

  float lPwer = 0;// The power calculated
  fcomplexcu* cplxRow;// Row in complex data
  float frac;

  int start, end;

  if (stage > 0)
  {
    start = (1 << (stage - 1));
    end = (1 << stage) - 1;
  }

  //__syncthreads(); // Not necessary

  __shared__ int xInds[16][SS_X_NUM];
  int noBaches = ceilf(SS_X_NUM * start / (float) threadsInB);

  // Calculate xInicies and store in shared memory
  for (int b = 0; b < noBaches; b++)
  {
    int pos = b * threadsInB + tId;
    int harmListIdx = floorf(pos / (float) SS_X_NUM);
    int harm = harmListIdx + start;
    if (harm <= end)
    {
      frac = others.arr[harm].harmFraction;
      sx = pos - harmListIdx * SS_X_NUM;

      if (sx < SS_X_NUM)
      {
        pix = tlx + sx;
        float rr = fRlow + pix * ACCEL_DR;
        float subr = calc_required_r_gpu(frac, rr);
        int sRlow = (int) floorf(calc_required_r_gpu(frac, fRlow));
        int isx = index_from_r(subr, sRlow);
        xInds[harmListIdx][sx] = isx;
      }
    }
  }

  __syncthreads();

  for (int harm = start; harm <= end; harm++)
  {
    int harmListIdx = harm - start;
    int* yInds = &YINDS[others.arr[harm].inds];

    // Read data from fundamental into shared memory
#ifndef DEBUG
#pragma unroll
#endif
    for (int y = 0; y < SS_Y_TILES; y++)
    {
      sy = y * SS_Y + threadIdx.y;
      piy = tly + sy;

      if (piy >= 0 && piy < others.arr[0].ffdotHeight)
      {
        cmplxY = yInds[piy];                        // Y inex lookup from const memory
        cmplxY *= others.arr[harm].ffdotStride;// Stride down
        cplxRow = &others.arr[harm].ffdot[cmplxY + others.arr[harm].ffdBuffre];

#ifndef DEBUG
#pragma unroll
#endif
        for (int x = 0; x < SS_X_TILES; x++)
        {
          sx = x * SS_X + threadIdx.x;
          pix = tlx + sx;                           // Note I should chop off this value as it is "out of bounds" but for the moment I am including it.
          cmplxX = pix;
          cmplxX = xInds[harmListIdx][sx];
          //cmplxX += others.arr[harm].ffdBuffre;
          //cmplxX++;

          //int CStar = others.arr[harm].ffdBuffre;
          //int CEnd  = others.arr[harm].ffdotWidth - others.arr[harm].ffdBuffre;
          if (cmplxX >= 0 && cmplxX < others.arr[harm].ffdotWidth - 2*others.arr[harm].ffdBuffre)
          {
            //if ( cmplxX >= CStar && cmplxX < CEnd )
            {
              fcomplexcu cmp = cplxRow[cmplxX];
              lPwer = cmp.r * cmp.r + cmp.i * cmp.i;
              powers[sy][sx] += lPwer;
              //atomicAdd(&powers[sy][sx], lPwer);
            }
            //else
            {
              //printf("cmplxX out of bounds\n");
            }
          }
          else
          {
            //powers[sy][sx] = 0; // TMP
          }
        }
      }
      else
      {
        for (int x = 0; x < SS_X_TILES; x++)
        {
          //powers[sy][sx] = 0; // TMP
        }
      }
    }
  }

  __syncthreads();
}
*/

template<int stage>
__device__ static inline  void sumPlainsSmTex(float powers_t[SS_Y_TILES][SS_X_TILES], /*uint xInds[8][SS_X_NUM],*/const cuSearchList searchList, /*const primaryInf* pInf, const int stage,*/ int tlx, int tly, cand* candListsm, int *count)
{
  //int tId = threadIdx.y * SS_X + threadIdx.x;       // Index in thread block

  //const int threadsInB = SS_X * SS_Y;                       //
  //int batch = ceilf(SS_X_NUM * SS_Y_NUM / (float) (threadsInB));

  int pix;// X location in powers data
  int piy;// Y location in powers data

  int sx;// X location in Shared memory
  int sy;// Y location in Shared memory

  //int cmplxX;                                       // X location in the complex data
  //uint cmplxY;                                       // Y location in the complex data

  const float powcut = POWERCUT[stage];

  float lPwer = 0;// The power calculated
  //fcomplexcu* cplxRow;                              // Row in complex data
  //float frac2;

  //int start, end;

  //if (stage > 0)
  //{
  //    start = (1 << (stage - 1));
  //end = (1 << stage) - 1;
  //}

  //const int end = start + searchList.cnt.val[stage];

  //__syncthreads(); // Not nessesary

  //int start, end;

  //if ( stage > 0 )
  //{
  const int start = (1 << (stage - 1));
  const int end = (1 << stage) - 1;

  //const int tId = threadIdx.y * SS_X + threadIdx.x;       // Index in thread block
  //const int blkId = blockIdx.y*gridDim.x+blockIdx.x;

  //}

  /*
   __syncthreads(); //make sure no one is still using xInds

   //int noBaches = ceilf(SS_X_NUM * searchList.cnt.val[stage] / (float) (SS_X * SS_Y) );
   const int noBaches = ceilf(SS_X_NUM * (start) / (float) (SS_X * SS_Y));

   // Calculate xInicies and store in shared memory
   for (int b = 0; b < noBaches; b++)
   {
   int pos = b * (SS_X * SS_Y) + tId;
   int harmListIdx = floorf(pos / (float) SS_X_NUM);
   int harm = harmListIdx + start;
   if (harm <= end)
   {
   frac2 = searchList.frac.val[harm];
   sx = pos - harmListIdx * SS_X_NUM;

   if (sx >= 0 && sx < SS_X_NUM)
   {
   pix = tlx + sx;
   float rr = pInf->fRlow + pix * ACCEL_DR;
   float subr = calc_required_r_gpu(frac2, rr);
   int sRlow = (int) floorf(calc_required_r_gpu(frac2, pInf->fRlow));
   int isx = index_from_r(subr, sRlow);
   xInds[harmListIdx][sx] = isx;
   }
   }
   }
   __syncthreads(); // make sure xInds is complete
   */

  int xInds2[SS_X_TILES];
  //int yInds2[SS_Y_TILES];

#ifndef DEBUG
#pragma unroll
#endif
  for (int harm = start; harm <= end; harm++)
  {
    const int* yInds       = &YINDS[searchList.yInds.val[harm]];
    const fcomplexcu* data    = searchList.datas.val[harm];
    const float frac2         = searchList.frac.val[harm];

#ifndef DEBUG
#pragma unroll
#endif
    for (int x = 0; x < SS_X_TILES; x++)
    {
      sx = x * SS_X + threadIdx.x;
      if (sx >= 0 && sx < SS_X_NUM)
      {
        pix         = tlx + sx;
        //double rr    = searchList.searchRLow + pix * ACCEL_DR;
        //rr = searchList.rLow.val[0] + pix * ACCEL_DR;
        //double subr  = calc_required_r_gpu(frac2, rr);
        //subr = rr ;//* frac2   ; // ( (ACCEL_RDR) * rr * frac2 + 0.5) * (ACCEL_DR) ;
        //int isx     = index_from_r(subr, sRlow);
        //xInds2[x]   = isx ;
        //xInds2[x]   += searchList.ffdBuffre.val[harm];
        //xInds2[x]   = ((((( 2.0 * (searchList.searchRLow + pix * ACCEL_DR) * frac2 + 0.5) * ACCEL_DR) - sRlow) * 2.0 + 1e-3 ));
        //xInds2[x] = subr;
        //xInds2[x]   = (int)(pix*frac2+searchList.idxSum.val[harm]) + searchList.ffdBuffre.val[harm];
      }
    }

    /* // Block calculating tiles
     const int harmListIdx = harm - start;
     #pragma unroll SS_Y_TILES
     for (int y = 0; y < SS_Y_TILES; y++)
     {
     sy = y * SS_Y + threadIdx.y;
     piy = tly + sy;
     if (piy >= 0 && piy < pInf->height)
     {
     int zz = pInf->fZlow + piy * ACCEL_DZ;
     int subz = calc_required_z(frac2, zz);
     int zind = index_from_z(subz, searchList.zMax.val[harm]);
     yInds2[y] = zind;
     }
     }
     */

    // Read data from fundamental into shared memory
#ifndef DEBUG
#pragma unroll
#endif
    for (int y = 0; y < SS_Y_TILES; y++)
    {
      sy = y * SS_Y + threadIdx.y;
      piy = tly + sy;

      if (piy >= 0 && piy < searchList.heights.val[0])
      {
        //int cmplxY = yInds[piy];                        // Y index lookup from constant memory
        //cmplxY = yInds2[y];

        //if ( tId == 0 && blkId == 0 )
        //  printf("piy: %-2i  cmplxY: %-2i  %.2f\n", piy, cmplxY, frac2);

        /*
         int zz = pInf->fZlow + piy * ACCEL_DZ;
         int subz = calc_required_z(frac2, zz);
         int zind = index_from_z(subz, searchList.zMax.val[harm]);
         cmplxY = zind;
         */

        //uint add = piy + searchList.yInds.val[harm];
        //int std = searchList.strides.val[harm];
        //if (searchList.yInds.val[harm] + piy > MAX_YINDS )
        //{
        //  int xxvv = searchList.yInds.val[harm] + piy;
        //  printf("yind to large!\n", xxvv);
        //}
        //int sz2 = cmplxY + 0 /*searchList.ffdBuffre.val[harm]*/ ;
        //if (sz2 > searchList.strides.val[harm]*searchList.heights.val[harm] )
        //  printf("sz2 %i to large!\n", sz2);

        const fcomplexcu* cplxRow = &(data[yInds[piy]*searchList.strides.val[harm]]);
        const int cmplxY = yInds[piy]; //*searchList.strides.val[harm];// Stride down

        //--&searchList.datas.val[harm][cmplxY + searchList.ffdBuffre.val[harm]];      //+ searchList.ffdBuffre.val[harm] ];

        //printf("add: %i std: %i \n", add, std);

#ifndef DEBUG
#pragma unroll
#endif
        for (int x = 0; x < SS_X_TILES; x++)
        {
          sx = x * SS_X + threadIdx.x;
          pix = tlx + sx;                           // Note I should chop off this value as it is "out of bounds" but for the moment I am including it.
          //cmplxX = pix;
          //const int cmplxX = xInds[harmListIdx][sx];
          const int cmplxX = xInds2[x];

          if (pix >= 0 && pix < searchList.widths.val[0] /* && cmplxX < searchList.strides.val[harm] */ )
          {
            //kker = tex2D < float2 > (kerTex, tid, y);
            //fcomplexcu cmp = searchList.texs[harm] [cmplxX][cmplxY];

            //const float2 cmp = tex2D < float2 > (searchList.texs.val[harm], cmplxX, cmplxY);
            //lPwer = cmp.x * cmp.x + cmp.y * cmp.y;

            //fcomplexcu cmp = searchList.datas.val[harm][ cmplxY*searchList.strides.val[harm] + cmplxX ];
            //const fcomplexcu cmp {0,0}; // = cplxRow[cmplxX];

            fcomplexcu cmp = cplxRow[cmplxX];
            lPwer = cmp.r * cmp.r + cmp.i * cmp.i;

            /*
            if ( sy == 0 && sx ==0 && blkId == 0 )
              printf("Stage %i - [%.2f]\n", stage, frac2);
            if ( sy == 0 && sx < 10 && blkId == 0 )
              printf("%.4f ", lPwer);
            if ( sy == 0 && sx == 9 && blkId == 0 )
              printf("\n");
             */

            powers_t[y][x] += lPwer;
            //powers_t[y][x] = xInds2[x];

            if ( harm == end )
            {
              if ( powers_t[y][x] >= powcut  )
              {
                int numharm = 1 << stage;
                candListsm[*count].power = powers_t[y][x];
                candListsm[*count].r = ( searchList.rLow.val[0] + pix * (double)ACCEL_DR ) / (double)numharm ;
                candListsm[*count].z = ( piy*(double) ACCEL_DZ - searchList.zMax.val[0]  ) / (double)numharm ;
                candListsm[*count].numharm = numharm;

                (*count)++;

                //printf("Found a candidate! Power: %.3f  \n", powers_t[y][x] );
              }

              //powcut
              //printf("pow: %f \n", powers_t[y][x] );
            }

            //if ( powers_t[y][x] >= powcut )
            //  printf("We found value pow %f\n", powers_t[y][x]);


            //atomicAdd(&powers[sy][sx], lPwer);
          }
          //else
          {
            //powers[sy][sx] = 0; // TMP
          }
        }
      }
      //else
      {
        //for (int x = 0; x < SS_X_TILES; x++)
        {
          //powers[sy][sx] = 0; // TMP
        }
      }
    }
  }

  // */

  //__syncthreads();
}

/*
__global__ void add_and_searchCU(cuFfdot10 others, cand* d_cands, int noStages, float fRlow, float fZlow, int copyBack, int searchP)
{
  __shared__ float powers[SS_Y_NUM][SS_X_NUM];

  // Temporary memory for candidates
  cand candList[SS_X_TILES*SS_Y_TILES*5];
  int candCount = 0;

  int tlx = blockIdx.x * (SS_X_NUM - 2 * SS_X_OVERLAP) - SS_X_OVERLAP;// Top leaf of the block covered by this block
  int tly = blockIdx.y * (SS_Y_NUM - 2 * SS_Y_OVERLAP) - SS_Y_OVERLAP;// Top leaf of the block covered by this block

  int threadsInB = SS_X * SS_Y;//
  int batch = ceilf(SS_X_NUM * SS_Y_NUM / (float) (threadsInB));

  int pix;// X location in powers data
  int piy;// Y location in powers data

  int sx;// X location in Shared memory
  int sy;// Y location in Shared memory

  int cmplxX;// X location in the complex data
  //int cmplxY;                                       // Y location in the complex data

  float lPwer = 0;// The power calculated
  fcomplexcu* cplxRow;// Row in complex data

  // Read data from fundamental into shared memory
  if ( true )
  {
#ifndef DEBUG
#pragma unroll
#endif
    for (int y = 0; y < SS_Y_TILES; y++)
    {
      sy = y * SS_Y + threadIdx.y;
      piy = tly + sy;

      if (piy >= 0 && piy < others.arr[0].ffdotHeight)
      {
        piy *= (others.arr[0].ffdotStride);         // Stride down
        cplxRow = &others.arr[0].ffdot[piy];

#ifndef DEBUG
#pragma unroll
#endif
        for (int x = 0; x < SS_X_TILES; x++)
        {
          sx = x * SS_X + threadIdx.x;
          pix = tlx + sx;                           // Note I should chop off this value as it is "out of bounds" but for the moment I am including it.
          cmplxX = pix + others.arr[0].ffdBuffre;   // Shift the address by ffdBuffre
          lPwer = 0;

          if (cmplxX >= 0 && cmplxX < others.arr[0].ffdotWidth)
          {
            fcomplexcu cmp = cplxRow[cmplxX];
            lPwer = cmp.r * cmp.r + cmp.i * cmp.i;
          }
          powers[sy][sx] = lPwer;


        }
      }
      else
      {
        // Set to zero
        for (int x = 0; x < SS_X_TILES; x++)
        {
          sx = x * SS_X + threadIdx.x;
          powers[sy][sx] = 0;
        }
      }
    }
  }

  if (searchP)
  {
    search(powers, 0, d_cands, candList, &candCount, fRlow, fZlow);
  }

  if ( true )
  {
    for (int stage = 1; stage < noStages; stage++)
    {
      sumPlainsSm(powers, others, stage, fRlow, fZlow, tlx, tly);

      if (searchP)
      {
        search(powers, stage, d_cands, candList, &candCount, fRlow, fZlow);
      }
    }
  }


  //others.arr[0].ffdotPowers[tId] += 1;
  //__threadfence();
  //if (  threadIdx.y == 0 && threadIdx.x ==0 && blockIdx.x == 0 && blockIdx.y == 0 && g_canCount != 0 )
  //  printf("Found %i \n",g_canCount);
}
*/

/** Sum and Search - loop down - find column max
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base
 */
template<int noStages, int FLAGS>
__global__ void add_and_searchCU3(cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base)
{
  const int bidx  = threadIdx.y * SS3_X   +   threadIdx.x;
  const int tid   = blockIdx.x  * SS3_Y*SS3_X + bidx;

  if ( tid < searchList.widths.val[0] )
  {
    const int noHarms = (1 << (noStages - 1));
    int inds[noHarms];
    accelcandBasic candLists[noStages];

    int start   = 0;
    int end     = 0;

    FOLD // Prep
    {
      // Initialise the x indices of this thread
      inds[0] = tid + searchList.ffdBuffre.val[0];

      // Calculate the x indices
#pragma unroll
      for ( int i = 1; i < noHarms; i++ )
      {
        //inds[i]     = (int)(tid*searchList.frac.val[i]+searchList.idxSum.val[i]) + searchList.ffdBuffre.val[i];
      }

      // Set the local and return candidate powers to zero
#pragma unroll
      for ( int i = 0; i < noStages; i++ )
      {
        candLists[i].sigma = 0;

        if ( FLAGS & CU_CAND_SINGLE_G )
          d_cands[tid*noStages+i].sigma = 0;
      }
    }

    FOLD // Prep
    {
      if  ( noStages >= 1 )
      {
        const int nPowers = (noStages)*2;   // The number of powers to batch calculate together, *2 is a "random choice it would be this or noHarms
        float powers[nPowers];              // registers to hold values to increase mem cache hits

        int y;
        for( y = 0; y < searchList.heights.val[0]-nPowers ; y+= nPowers )
        {

#pragma unroll
          for( int i = 0; i < nPowers ; i++ )
          {
            powers[i] = 0;
          }

          // Loop over stages, sum and search
#pragma unroll
          for ( int stage = 0 ; stage < noStages; stage++)
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
            for ( int harm = start; harm < end; harm++ )
            {

#pragma unroll
              for( int i = 0; i < nPowers; i++ )
              {
                if  ( (FLAGS & FLAG_PLN_TEX ) )
                {
                  const float2 cmpf = tex2D < float2 > (searchList.texs.val[harm], inds[harm], YINDS[searchList.yInds.val[harm]+y+i]);
                  powers[i] += cmpf.x * cmpf.x + cmpf.y * cmpf.y;
                }
                else
                {
                  const fcomplexcu cmpc  = searchList.datas.val[harm][YINDS[searchList.yInds.val[harm]+y+i]*searchList.strides.val[harm]+inds[harm]];
                  powers[i] += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                }
              }

            }

            // Search set of powers
#pragma unroll
            for( int i = 0; i < nPowers ; i++ )
            {
              if  (  powers[i] >  POWERCUT[stage] )
              {
                //if ( lPwer > candLists[stage].sigma )
                if ( powers[i] > candLists[stage].sigma )
                {
                  candLists[stage].sigma  = powers[i];
                  candLists[stage].z      = y;
                }
              }
            }
          }
        }

        // one last loop with the if statement included
        {
          start   = 0;
          end     = 0;

#pragma unroll
          for( int i = 0; i < nPowers ; i++ )
          {
            powers[i] = 0;
          }

#pragma unroll
          for ( int stage = 0 ; stage < noStages; stage++)
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

#pragma unroll
            for ( int harm = start; harm < end; harm++ )
            {

#pragma unroll
              for( int i = 0; i < nPowers; i++ )
              {
                if ( i + y < searchList.heights.val[0])
                {
                  if  ( ( FLAGS & FLAG_PLN_TEX ) )
                  {
                    const float2 cmpf = tex2D < float2 > (searchList.texs.val[harm], inds[harm], YINDS[searchList.yInds.val[harm]+y+i]);
                    powers[i] += cmpf.x * cmpf.x + cmpf.y * cmpf.y;
                  }
                  else
                  {
                    const fcomplexcu cmpc  = searchList.datas.val[harm][YINDS[searchList.yInds.val[harm]+y+i]*searchList.strides.val[harm]+inds[harm]];
                    powers[i] += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                  }
                }
              }

            }

#pragma unroll
            for( int i = 0; i < nPowers ; i++ )
            {
              if  (  powers[i] >  POWERCUT[stage] )
              {
                //if ( lPwer > candLists[stage].sigma )
                if ( powers[i] > candLists[stage].sigma )
                {
                  candLists[stage].sigma  = powers[i];
                  candLists[stage].z      = y;
                }
              }
            }
          }
        }
      }
      else
      {
        float lPwer = 0;
        int y;
        for( y = 0; y < searchList.heights.val[0] ; y++ )
        {
          lPwer = 0;

#pragma unroll
          for ( int stage = 0 ; stage < noStages; stage++)
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

#pragma unroll
            for ( int harm = start; harm < end; harm++ )
            {

              //if ( noStages == -1 )
              {
                //cmpc  = searchList.datas.val[0][y*searchList.strides.val[0]+inds[0]];
                //lPwer += cmpc.r * cmpc.r + cmpc.i * cmpc.i;

                //cmpf = tex2D < float2 > (searchList.texs.val[harm], inds[harm], y);
                //lPwer += cmpf.x * cmpf.x + cmpf.y * cmpf.y;

                //const fcomplexcu cmpc  = searchList.datas.val[0][y*searchList.strides.val[0]+searchList.ffdBuffre.val[0] + tid];
                //const fcomplexcu cmpc =  baseaDDS[0][y*searchList.strides.val[0] + searchList.ffdBuffre.val[0] + tid ];
                //lPwer += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
              }
              /*
            else if ( searchList.frac.val[harm] > 0.7  )
            {
              cmpc  = searchList.datas.val[harm][YINDS[searchList.yInds.val[harm]+y]*searchList.strides.val[harm]+inds[harm]];
              lPwer += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
            }
               */
              //else //if ( harm == 1 )
              //else
              {


                //if ( FLAGS & FLAG_PLN_TEX )
                {
                  //const float2 cmpf = tex2D < float2 > (searchList.texs.val[harm], inds[harm], YINDS[searchList.yInds.val[harm]+y]);
                  //lPwer += cmpf.x * cmpf.x + cmpf.y * cmpf.y;
                }
                //else
                {
                  const fcomplexcu cmpc  = searchList.datas.val[harm][YINDS[searchList.yInds.val[harm]+y]*searchList.strides.val[harm]+inds[harm]];
                  //const fcomplexcu cmpc  = searchList.datas.val[0][YINDS[searchList.yInds.val[0]+y]*searchList.strides.val[0]+inds[0]];
                  //const fcomplexcu cmpc =  baseaDDS[harm][YINDS[searchList.yInds.val[harm]+y]*searchList.strides.val[harm]+inds[harm]];
                  //lPwer += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                  lPwer += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
                }


                //int cmplxY = YINDS[searchList.yInds.val[harm]+y] ;

                /*
              if ( harm == 0)
                cmplxY =  y ;
                else
                  cmplxY =  YINDS[searchList.yInds.val[harm]+y] ;
                 */

                //const float2 cmpf = tex2D < float2 > (searchList.texs.val[harm], inds[harm], YINDS[searchList.yInds.val[harm]+y]);
                //const float2 cmpf = tex2D < float2 > (searchList.texs.val[0], inds[0], YINDS[searchList.yInds.val[0]+y]);
                //lPwer += cmpf.x * cmpf.x + cmpf.y * cmpf.y;
              }

              //lPwer += cmpc.r * cmpc.r + cmpc.i * cmpc.i;
            }

            if  ( lPwer >  POWERCUT[stage] )
            {
              if ( lPwer > candLists[stage].sigma )
              {
                candLists[stage].sigma  = lPwer;
                candLists[stage].z      = y;
              }
            }
          }
        }
      }
    }

    // Write results back to DRAM and calculate sigma if needed
    if      ( FLAGS & CU_CAND_DEVICE   )
    {
#pragma unroll
      for ( int stage = 0 ; stage < noStages; stage++)
      {
        const short numharm = 1 << stage;

        if  ( candLists[stage].sigma >  POWERCUT[stage] )
        {
          int idx =  (int)(( searchList.rLow.val[0] + tid * (double) ACCEL_DR ) / (double)numharm ) - base ;
          if ( idx >= 0 )
          {
            long long numtrials         = NUMINDEP[stage];
            candLists[stage].numharm    = numharm;
            //candLists[stage].z          = ( candLists[stage].z*(double) ACCEL_DZ - searchList.zMax.val[0]  )  / (double)numharm ;
            candLists[stage].sigma      = (float)candidate_sigma_cu(candLists[stage].sigma, numharm, numtrials);

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
    else if ( FLAGS & CU_CAND_SINGLE_G )
    {
#pragma unroll
      for ( int stage = 0 ; stage < noStages; stage++)
      {
        //if ( candLists[stage].sigma > 0 )
        if  ( candLists[stage].sigma >  POWERCUT[stage] )
        {
          const short numharm = ( 1 << stage );
          candLists[stage].numharm      = numharm;

          if ( FLAGS & FLAG_SAS_SIG )
          {
            // Calculate sigma value
            long long numtrials         = NUMINDEP[stage];
            candLists[stage].sigma      = (float)candidate_sigma_cu(candLists[stage].sigma, numharm, numtrials);
          }

          // Write to DRAM
          d_cands[tid*noStages + stage] = candLists[stage];
        }
      }
    }
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

/** Sum and Search - loop down - column max - multi-step .
 *
 * @param searchList
 * @param d_cands
 * @param d_sem
 * @param base          Used in CU_CAND_DEVICE
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
  const int bidx  = threadIdx.y * SS3_X         +  threadIdx.x;
  const int tid   = blockIdx.x  * (SS3_Y*SS3_X) +  bidx;

  const int width = searchList.widths.val[0];

  if ( tid < width )
  {
    const int noHarms     = ( 1 << (noStages-1) ) ;
    const int nPowers     = 8 ; // (noStages)*2;      // The number of powers to batch calculate together, *2 is a "random choice it would be this or noHarms
    const int zeroHeight  = searchList.heights.val[0] ;

#if TEMPLATE_SEARCH == 1
    accelcandBasic candLists[noStages][noSteps];

    int         inds[noSteps][noHarms];
    fcomplexcu* pData[noSteps][noHarms];
    //float*      pPowr[noSteps][noHarms];
    float       powers[noSteps][nPowers];         // registers to hold values to increase mem cache hits
#else
    accelcandBasic candLists[noStages][MAX_STEPS];

    int         inds[MAX_STEPS][noHarms];
    fcomplexcu* pData[MAX_STEPS][noHarms];
    //float*      pPowr[MAX_STEPS][noHarms];
    float       powers[MAX_STEPS][nPowers];         // registers to hold values to increase mem cache hits
#endif

    int start   = 0;
    int end     = 0;
    int iy, ix;
    int y;

    FOLD // Prep - Initialise the x indices & set candidates to 0 .
    {
      // Calculate the x indices or create a pointer offset by the correct amount
#pragma unroll
      for ( int harm = 0; harm < noHarms; harm++ )      // loop over harmonic
      {
#if TEMPLATE_SEARCH == 1
#pragma unroll
#endif
        for ( int step = 0; step < noSteps; step++)     // Loop over steps
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
              if      ( FLAGS & FLAG_FFT_OUT )
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
              if      ( FLAGS & FLAG_FFT_OUT )
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
            if ( FLAGS & FLAG_FFT_OUT )
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

      // Set the local and return candidate powers to zero
      FOLD
      {
#pragma unroll
        for ( int stage = 0; stage < noStages; stage++ )
        {
#if TEMPLATE_SEARCH == 1
#pragma unroll
#endif
          for ( int step = 0; step < noSteps; step++)   // Loop over steps

          {
            candLists[stage][step ].sigma = 0 ;

            if ( FLAGS & CU_CAND_SINGLE_G )
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
        //for( y = 0; y < searchList.heights.val[0] - nPowers ; y += nPowers ) // loop over chunks .
        for( y = 0; y < searchList.heights.val[0] ; y += nPowers ) // loop over chunks .
        {
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
            for ( int harm = start; harm < end; harm++ )          // Loop over harmonics (plains) in this stage
            {

//#pragma unroll
              for( int yPlus = 0; yPlus < nPowers; yPlus++ )                // Loop over the chunk
              {
                int trm = y + yPlus ;

                iy            = YINDS[ searchList.yInds.val[harm] + trm ];


#if TEMPLATE_SEARCH == 1
//#pragma unroll
#endif
                for ( int step = 0; step < noSteps; step++)         // Loop over steps
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
                    if ( FLAGS & FLAG_FFT_OUT )
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
            for ( int step = 0; step < noSteps; step++)         	    // Loop over steps
            {
              //#pragma unroll
              for( int i = 0; i < nPowers ; i++ )                     // Loop over section
              {
                if  (  powers[step][i] > POWERCUT[stage] )
                {
                  if ( powers[step][i] > candLists[stage][step].sigma )
                  {
                    if ( y + i < zeroHeight )
                    {
                      // This is our new max!
                      candLists[stage][step].sigma  = powers[step][i];
                      candLists[stage][step].z      = y+i;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    // Write results back to DRAM and calculate sigma if needed
    if      ( FLAGS & CU_CAND_DEVICE   )
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
    else if ( FLAGS & CU_CAND_SINGLE_G )
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
            //const short numharm                 = ( 1 << stage );
            //candLists[stage][step].numharm      = numharm;

            if ( FLAGS & FLAG_SAS_SIG && FALSE)
            {
              const short numharm               = ( 1 << stage );

              // Calculate sigma value
              long long numtrials               = NUMINDEP[stage];
              candLists[stage][step].sigma      = (float)candidate_sigma_cu(candLists[stage][step].sigma, numharm, numtrials);
            }

            // Write to DRAM
            d_cands[step*noStages*width + stage*width + tid] = candLists[stage][step];
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
 * @param base          Used in CU_CAND_DEVICE
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

            if ( FLAGS & CU_CAND_SINGLE_G )
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
            for ( int harm = start; harm < end; harm++ )          // Loop over harmonics (plains) in this stage
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
      if      ( FLAGS & CU_CAND_DEVICE   )
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
      else if ( FLAGS & CU_CAND_SINGLE_G )
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
 * @param base          Used in CU_CAND_DEVICE
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

              if ( FLAGS & CU_CAND_SINGLE_G )
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
                for ( int harm = start; harm < end; harm++ )            // Loop over harmonics (plains) in this stage
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
                for ( int harm = start; harm < end; harm++ )          // Loop over harmonics (plains) in this stage
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
      if      ( FLAGS & CU_CAND_DEVICE   )
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
      else if ( FLAGS & CU_CAND_SINGLE_G )
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
__host__ void add_and_searchCU31_p(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream,cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base, cuSearchItem* pd, float* rLows, int noSteps, const uint noStages )
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

__host__ void add_and_searchCU31_f(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream,cuSearchList searchList, accelcandBasic* d_cands, uint* d_sem, int base, cuSearchItem* pd, float* rLows, int noSteps, const uint noStages, uint FLAGS )
{
  if        ( FLAGS & FLAG_FFT_OUT )
  {
    if      ( FLAGS & FLAG_STP_ROW )
      add_and_searchCU31_p<FLAG_FFT_OUT | CU_CAND_SINGLE_G | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, pd, rLows, noSteps, noStages );
    //else if ( FLAGS & FLAG_STP_PLN )
    //  add_and_searchCU31_p<FLAG_FFT_OUT | CU_CAND_SINGLE_G | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, pd, rLows, noSteps, noStages );
    //else if ( FLAGS & FLAG_STP_STK )
    //  add_and_searchCU31_p<FLAG_FFT_OUT | CU_CAND_SINGLE_G | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, pd, rLows, noSteps, noStages );
    else
    {
      fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n" );
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    if      ( FLAGS & FLAG_STP_ROW )
      add_and_searchCU31_p<CU_CAND_SINGLE_G | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, pd, rLows, noSteps, noStages );
    //else if ( FLAGS & FLAG_STP_PLN )
    //  add_and_searchCU31_p<CU_CAND_SINGLE_G | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, pd, rLows, noSteps, noStages );
    //else if ( FLAGS & FLAG_STP_STK )
    //  add_and_searchCU31_p<CU_CAND_SINGLE_G | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, pd, rLows, noSteps, noStages );
    else
    {
      fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n" );
      exit(EXIT_FAILURE);
    }
  }

// Uncomenting this block will make compile time VERY long! I mean days!
/*
  if( FLAGS & CU_CAND_DEVICE )
  {
    if( FLAGS & FLAG_PLN_TEX )
    {
      if ( FLAGS & FLAG_SAS_SIG )
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_CAND_DEVICE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_CAND_DEVICE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_CAND_DEVICE | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination.\n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_CAND_DEVICE | FLAG_PLN_TEX | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_CAND_DEVICE | FLAG_PLN_TEX | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_CAND_DEVICE | FLAG_PLN_TEX | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
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
          add_and_searchCU31_p<CU_CAND_DEVICE | FLAG_SAS_SIG | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_CAND_DEVICE | FLAG_SAS_SIG | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_CAND_DEVICE | FLAG_SAS_SIG | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_CAND_DEVICE | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_CAND_DEVICE | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_CAND_DEVICE | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
    }
  }
  else if ( (FLAGS & CU_CAND_SINGLE_G) || (FLAGS & CU_CAND_HOST) )
  {
    if( FLAGS & FLAG_PLN_TEX )
    {
      if ( FLAGS & FLAG_SAS_SIG )
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_CAND_SINGLE_G | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_CAND_SINGLE_G | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_CAND_SINGLE_G | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination.  \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_CAND_SINGLE_G | FLAG_PLN_TEX | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_CAND_SINGLE_G | FLAG_PLN_TEX | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_CAND_SINGLE_G | FLAG_PLN_TEX | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
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
          add_and_searchCU31_p<CU_CAND_SINGLE_G | FLAG_SAS_SIG | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_CAND_SINGLE_G | FLAG_SAS_SIG | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_CAND_SINGLE_G | FLAG_SAS_SIG | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination.  \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      ( FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_CAND_SINGLE_G | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_CAND_SINGLE_G | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        //else if ( FLAGS & FLAG_STP_STK )
        //  add_and_searchCU31_p<CU_CAND_SINGLE_G | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination. \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
    }
  }
  else if (  FLAGS & CU_CAND_SINGLE_C )
  {
    if( FLAGS & FLAG_PLN_TEX )
    {
      if ( FLAGS & FLAG_SAS_SIG )
      {
        if      (FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_CAND_SINGLE_C | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_CAND_SINGLE_C | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_STK )
          add_and_searchCU31_p<CU_CAND_SINGLE_C | FLAG_PLN_TEX | FLAG_SAS_SIG | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination.  \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      (FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_CAND_SINGLE_C | FLAG_PLN_TEX | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_CAND_SINGLE_C | FLAG_PLN_TEX | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_STK )
          add_and_searchCU31_p<CU_CAND_SINGLE_C | FLAG_PLN_TEX | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
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
          add_and_searchCU31_p<CU_CAND_SINGLE_C | FLAG_SAS_SIG | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_CAND_SINGLE_C | FLAG_SAS_SIG | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_STK )
          add_and_searchCU31_p<CU_CAND_SINGLE_C | FLAG_SAS_SIG | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else
        {
          fprintf(stderr, "ERROR: add_and_searchCU31 has not been templated for flag combination.  \n", noPlns);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        if      (FLAGS & FLAG_STP_ROW )
          add_and_searchCU31_p<CU_CAND_SINGLE_C | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_PLN )
          add_and_searchCU31_p<CU_CAND_SINGLE_C | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
        else if ( FLAGS & FLAG_STP_STK )
          add_and_searchCU31_p<CU_CAND_SINGLE_C | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, searchList, d_cands, d_sem, base, noSteps, noPlns );
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
        if ( canMethoud & CU_CAND_SINGLE_G )
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
      if      ( canMethoud & CU_CAND_DEVICE   )
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
      else if ( canMethoud & CU_CAND_SINGLE_G )
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

int setConstVals( cuStackList* stkLst, int numharmstages, float *powcut, long long *numindep )
{
  int noHarms         = (1 << (numharmstages - 1) );

  int szx = sizeof(int)*8;

  if (stkLst->hInfos[0].height* (noHarms /*-1*/ ) > MAX_YINDS)
  {
    printf("ERROR! YINDS to small!");
  }
  int *indsY    = (int*) malloc(stkLst->hInfos[0].height * noHarms * sizeof(int));
  int bace      = 0;
  stkLst->hInfos[0].yInds = 0;
  for (int ii = 0; ii< stkLst->noHarms; ii++)
  {
    if ( ii == 0 )
    {
      for (int j = 0; j< stkLst->hInfos[0].height; j++)
      {
        indsY[bace + j] = j;
      }
    }
    else
    {
      for (int j = 0; j< stkLst->hInfos[0].height; j++)
      {
        int zz    = -stkLst->hInfos[0].zmax+ j* ACCEL_DZ;
        int subz  = calc_required_z(stkLst->hInfos[ii].harmFrac, zz);
        int zind  = index_from_z(subz, -stkLst->hInfos[ii].zmax);
        if (zind< 0|| zind>= stkLst->hInfos[ii].height)
        {
          int Err = 0;
          printf("ERROR! YINDS Wrong!");
        }
        indsY[bace + j] = zind;
      }
    }
    stkLst->hInfos[ii].yInds = bace;
    bace += stkLst->hInfos[0].height;
  }

  void *dcoeffs;

  cudaGetSymbolAddress((void **)&dcoeffs, YINDS);
  CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, indsY, bace*sizeof(int), cudaMemcpyHostToDevice),                      "Copying Y indices to device");

  cudaGetSymbolAddress((void **)&dcoeffs, POWERCUT);
  CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, powcut, numharmstages * sizeof(float), cudaMemcpyHostToDevice),        "Copying power cutoff to device");

  cudaGetSymbolAddress((void **)&dcoeffs, NUMINDEP);
  CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, numindep, numharmstages * sizeof(long long), cudaMemcpyHostToDevice),  "Copying stages to device");

  //CUDA_SAFE_CALL(cudaMemcpyToSymbol(YINDS,    indsY,         bace * sizeof(int), cudaMemcpyHostToDevice), "Failed to copy Y indices to device memory.");
  //cudaMemcpyToSymbol(POWERCUT, powcut,    numharmstages * sizeof(float));
  //cudaMemcpyToSymbol(NUMINDEP, numindep,  numharmstages * sizeof(long long));

  //for(int i = 0 ; i < 400; i ++)
  {
    //printf("%03i:  %-5i  %i \n", i, indsY[i], sizeof(int)*8 );
  }

  //CUDA_SAFE_CALL(cudaDeviceSynchronize(),"");

  //print_YINDS<<<1,1>>>(400);

  //CUDA_SAFE_CALL(cudaDeviceSynchronize(),"");

  CUDA_SAFE_CALL(cudaGetLastError(), "Error Preparing the constant memory.");
}

void sumAndSearch(cuStackList* plains, accelobs* obs, GSList** cands)
{
  //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  dim3 dimBlock, dimGrid;

  nvtxRangePush("Add & Search");

  if ( plains->haveSData || plains->haveCData ) // previous plain has data data so sum and search
  {
    int noStages = log(plains->noHarms)/log(2) + 1;
    int harmtosum;
    cuSearchList searchList;      // The list of details of all the individual plains
    cuSearchItem* pd;
    float *rLows;
    pd = (cuSearchItem*)malloc(plains->noHarms * sizeof(cuSearchItem));
    rLows = (float*)malloc(plains->noSteps * sizeof(float));

    FOLD // Do synchronisations
    {
      for (int ss = 0; ss< plains->noStacks; ss++)
      {
        cuFfdotStack* cStack = &plains->stacks[ss];

        cudaStreamWaitEvent(plains->strmSearch, cStack->plnComp, 0);
      }
    }

    if ( plains->haveCData ) // Sum & search
    {
      FOLD // Create search list
      {
        //printf("\n");

        searchList.searchRLow = plains->searchRLow;
        int i = 0;
        for (int stage = 0; stage < noStages; stage++)
        {
          harmtosum = 1 << stage;

          for (int harm = 1; harm <= harmtosum; harm += 2)
          {
            //printf("Stage  %i harm %i \n", stage, harm);

            float fract = 1-harm/ float(harmtosum);
            int idx = round(fract* plains->noHarms);
            if ( fract == 1 )
              idx = 0;

            searchList.texs.val[i]      = plains->plains[idx].datTex;
            searchList.datas.val[i]     = plains->plains[idx].d_plainData;
            searchList.powers.val[i]    = plains->plains[idx].d_powers;
            searchList.frac.val[i]      = plains->hInfos[idx].harmFrac;
            searchList.yInds.val[i]     = plains->hInfos[idx].yInds;
            searchList.heights.val[i]   = plains->hInfos[idx].height;
            searchList.widths.val[i]    = plains->plains[idx].ffdotPowWidth[0];
            searchList.strides.val[i]   = plains->hInfos[idx].inpStride;
            searchList.ffdBuffre.val[i] = plains->hInfos[idx].halfWidth*ACCEL_NUMBETWEEN;
            searchList.zMax.val[i]      = plains->hInfos[idx].zmax;
            //searchList.fullRLow.val[i]  = plains->plains[idx].searchRlowPrev[0];// .fullRLow[0];
            searchList.rLow.val[i]      = plains->plains[idx].rLow[0];

            /*
            pd[i].tex                   = plains->plains[idx].datTex;
            pd[i].data                  = plains->plains[idx].d_plainData;
            pd[i].frac                  = plains->hInfos[idx].harmFrac;
            pd[i].yInd                  = plains->hInfos[idx].yInds;
            pd[i].height                = plains->hInfos[idx].height;
            pd[i].width                 = plains->plains[idx].ffdotPowWidth[0];
            pd[i].stride                = plains->hInfos[idx].inpStride;
            pd[i].ffdBuffre             = plains->hInfos[idx].halfWidth*ACCEL_NUMBETWEEN;
            pd[i].zMax                  = plains->hInfos[idx].zmax;


            // Values for creating x indices
            //double diff =  plains->plains[idx].rLow[0] * plains->hInfos[idx].harmFrac - floor(plains );
            double diff =  plains->plains[idx].searchRlowPrev[0] * plains->hInfos[idx].harmFrac - floor( plains->plains[idx].searchRlowPrev[0] * plains->hInfos[idx].harmFrac );
            searchList.idxSum.val[i]    = 0.5 + diff*ACCEL_RDR ;

            for ( int step = 0; step < plains->noSteps; step++)         // Loop over steps
            {
              float dd      = plains->plains[idx].rLow[step] - plains->plains[0].rLow[step] * plains->hInfos[idx].harmFrac;

              double p1     = plains->plains[0].rLow[step] * plains->hInfos[idx].harmFrac ;
              double p2     = floor(plains->plains[idx].rLow[step]) ;
              double diff   = plains->plains[0].rLow[step] * plains->hInfos[idx].harmFrac - floor(plains->plains[idx].rLow[step]);
              float idxS    = 0.5 + diff*ACCEL_RDR ;
              //printf("Step %02i    diff: %12.2f    idx: %12.2f \n", step, diff, idxS);
            }*/

            i++;
          }
        }
      }

      FOLD // Call the main sum & search kernel
      {
        /*
        if      (  plains->flag & CU_CAND_DEVICE )
        {
          dimBlock.x = SS3_X;
          dimBlock.y = SS3_Y;

          float bw = SS3_X * SS3_Y;
          float ww = plains->plains[0].ffdotPowWidth / ( bw );

          dimGrid.x = ceil(ww);
          dimGrid.y = 1;

          if( plains->flag & FLAG_PLN_TEX )
          {
            if      ( noStages == 1 )
              add_and_searchCU3<1,CU_CAND_DEVICE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
            else if ( noStages == 2 )
              add_and_searchCU3<2,CU_CAND_DEVICE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
            else if ( noStages == 3 )
              add_and_searchCU3<3,CU_CAND_DEVICE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
            else if ( noStages == 4 )
              add_and_searchCU3<4,CU_CAND_DEVICE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
            else if ( noStages == 5 )
              add_and_searchCU3<5,CU_CAND_DEVICE | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
          }
          else
          {
            if      ( noStages == 1 )
              add_and_searchCU3<1,CU_CAND_DEVICE><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
            else if ( noStages == 2 )
              add_and_searchCU3<2,CU_CAND_DEVICE><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
            else if ( noStages == 3 )
              add_and_searchCU3<3,CU_CAND_DEVICE><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
            else if ( noStages == 4 )
              add_and_searchCU3<4,CU_CAND_DEVICE><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
            else if ( noStages == 5 )
              add_and_searchCU3<5,CU_CAND_DEVICE><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );

          }
          plains->haveCData = 0;
        }
        else if ( (plains->flag & CU_CAND_SINGLE_G) || (plains->flag & CU_CAND_HOST) )
        {
          dimBlock.x = SS3_X;
          dimBlock.y = SS3_Y;

          float bw = SS3_X * SS3_Y;
          float ww = plains->plains[0].ffdotPowWidth / ( bw );

          dimGrid.x = ceil(ww);
          dimGrid.y = 1;

          if( plains->flag & FLAG_PLN_TEX )
          {
            if ( plains->flag & FLAG_SAS_SIG )
            {
              if      ( noStages == 1 )
                add_and_searchCU3<1,CU_CAND_SINGLE_G | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 2 )
                add_and_searchCU3<2,CU_CAND_SINGLE_G | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 3 )
                add_and_searchCU3<3,CU_CAND_SINGLE_G | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 4 )
                add_and_searchCU3<4,CU_CAND_SINGLE_G | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 5 )
                add_and_searchCU3<5,CU_CAND_SINGLE_G | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
            }
            else
            {
              if      ( noStages == 1 )
                add_and_searchCU3<1,CU_CAND_SINGLE_G | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 2 )
                add_and_searchCU3<2,CU_CAND_SINGLE_G | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 3 )
                add_and_searchCU3<3,CU_CAND_SINGLE_G | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 4 )
                add_and_searchCU3<4,CU_CAND_SINGLE_G | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 5 )
                add_and_searchCU3<5,CU_CAND_SINGLE_G | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
            }
          }
          else
          {
            if ( plains->flag & FLAG_SAS_SIG )
            {
              if      ( noStages == 1 )
                add_and_searchCU3<1,CU_CAND_SINGLE_G | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 2 )
                add_and_searchCU3<2,CU_CAND_SINGLE_G | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 3 )
                add_and_searchCU3<3,CU_CAND_SINGLE_G | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 4 )
                add_and_searchCU3<4,CU_CAND_SINGLE_G | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 5 )
                add_and_searchCU3<5,CU_CAND_SINGLE_G | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
            }
            else
            {
              if      ( noStages == 1 )
                add_and_searchCU3<1,CU_CAND_SINGLE_G><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 2 )
                add_and_searchCU3<2,CU_CAND_SINGLE_G><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 3 )
                add_and_searchCU3<3,CU_CAND_SINGLE_G><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 4 )
                add_and_searchCU3<4,CU_CAND_SINGLE_G><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 5 )
                add_and_searchCU3<5,CU_CAND_SINGLE_G><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
            }
          }

                  }
        else if (  plains->flag & CU_CAND_SINGLE_C )
        {
          dimBlock.x = SS3_X;
          dimBlock.y = SS3_Y;

          float bw = SS3_X * SS3_Y;
          float ww = plains->plains[0].ffdotPowWidth / ( bw );

          dimGrid.x = ceil(ww);
          dimGrid.y = 1;

          if ( plains->flag & FLAG_PLN_TEX )
          {
            if ( plains->flag & FLAG_SAS_SIG )
            {
              if      ( noStages == 1 )
                add_and_searchCU3<1,CU_CAND_SINGLE_C | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 2 )
                add_and_searchCU3<2,CU_CAND_SINGLE_C | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 3 )
                add_and_searchCU3<3,CU_CAND_SINGLE_C | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 4 )
                add_and_searchCU3<4,CU_CAND_SINGLE_C | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 5 )
                add_and_searchCU3<5,CU_CAND_SINGLE_C | FLAG_SAS_SIG | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
            }
            else
            {
              if      ( noStages == 1 )
                add_and_searchCU3<1,CU_CAND_SINGLE_C | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 2 )
                add_and_searchCU3<2,CU_CAND_SINGLE_C | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 3 )
                add_and_searchCU3<3,CU_CAND_SINGLE_C | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 4 )
                add_and_searchCU3<4,CU_CAND_SINGLE_C | FLAG_PLN_TEX><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 5 )
                add_and_searchCU3<5,CU_CAND_SINGLE_C><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
            }
          }
          else
          {
            if ( plains->flag & FLAG_SAS_SIG )
            {
              if      ( noStages == 1 )
                add_and_searchCU3<1,CU_CAND_SINGLE_C | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 2 )
                add_and_searchCU3<2,CU_CAND_SINGLE_C | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 3 )
                add_and_searchCU3<3,CU_CAND_SINGLE_C | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 4 )
                add_and_searchCU3<4,CU_CAND_SINGLE_C | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 5 )
                add_and_searchCU3<5,CU_CAND_SINGLE_C | FLAG_SAS_SIG><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
            }
            else
            {
              if      ( noStages == 1 )
                add_and_searchCU3<1,CU_CAND_SINGLE_C><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 2 )
                add_and_searchCU3<2,CU_CAND_SINGLE_C><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 3 )
                add_and_searchCU3<3,CU_CAND_SINGLE_C><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 4 )
                add_and_searchCU3<4,CU_CAND_SINGLE_C><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
              else if ( noStages == 5 )
                add_and_searchCU3<5,CU_CAND_SINGLE_C><<<dimGrid, dimBlock, 0, plains->strmSearch>>>(searchList, plains->d_bCands, plains->d_candSem, plains->rLow );
            }
          }
        }
         */

        if ( (plains->flag & CU_CAND_SINGLE_G) || (plains->flag & CU_CAND_HOST) ) // Call the templated kernel
        {
          dimBlock.x  = SS3_X;
          dimBlock.y  = SS3_Y;

          float bw    = SS3_X * SS3_Y;
          float ww    = plains->plains[0].ffdotPowWidth[0] / ( bw );

          dimGrid.x   = ceil(ww);
          dimGrid.y   = 1;

          add_and_searchCU31_f(dimGrid, dimBlock, 0, plains->strmSearch, searchList, plains->d_bCands, plains->d_candSem, 0, pd, &plains->plains->rLow[0], plains->noSteps, plains->noHarmStages, plains->flag );
        }

        // Run message
        CUDA_SAFE_CALL(cudaGetLastError(), "Error at add_and_searchCU31 kernel launch");

        CUDA_SAFE_CALL(cudaEventRecord(plains->searchComp,  plains->strmSearch),"Recording event: searchComp");
      }
    }

    if ( plains->haveSData ) // Process previous results
    {
      if ( plains->flag & CU_CAND_SINGLE_G )
      {
        // A blocking synchronisation to ensure results are ready to be proceeded by the host
        CUDA_SAFE_CALL(cudaEventSynchronize(plains->candCpyComp), "ERROR: copying result from device to host.");

        nvtxRangePush("CPU Process results");

        plains->noResults=0;

        long long numindep;

        double poww, sig, sigx, sigc, diff;
        double gpu_p, gpu_q;
        double rr, zz;
        int added = 0;
        int numharm;
        poww = 0;

        double diffRL = plains->plains[0].searchRlow - plains->plains[0].rLow;

        if      ( ( plains->flag & CU_CAND_SINGLE_C ) == CU_CAND_SINGLE_C ) 	  // Process previous results
        {
#pragma omp critical
          for ( int x = 0; x < plains->accelLen; x++ )
          {
            for ( int i = 0; i < noStages; i++ )
            {
              for ( int step = 0; step < plains->mxSteps; step++)         // Loop over steps
              {
                int idx   = x*noStages + i ;
                poww      = plains->h_bCands[idx].sigma;

                if ( poww > 0 )
                {
                  plains->noResults++;

                  numharm   = plains->h_bCands[idx].numharm;
                  numindep  = obs->numindep[twon_to_index(numharm)];

                  if ( plains->flag & FLAG_SAS_SIG )
                    sig     = poww;
                  else
                    sig     = candidate_sigma(poww, numharm, numindep);

                  rr = ( plains->plains[0].searchRlowPrev[step] + x *  ACCEL_DR )        / (double)numharm ;
                  zz = ( plains->h_bCands[idx].z * ACCEL_DZ - plains->hInfos[0].zmax )   / (double)numharm ;

                  FOLD
                  {
                    added = 0;
                    //cands = insert_new_accelcand2(cands, poww, sig, numharm, rr, zz, &added);
                    *cands = insert_new_accelcand(*cands, poww, sig, numharm, rr, zz, &added);
                  }
                }

                if (added && !obs->dat_input)
                {
                  //fprintf(obs->workfile, "%12.2f [ %12.5f %12.5f ]  %3d  %14.4f  %14.4f  %10.4f  GPU\n", poww, sig, sigx, numharm, rr, rr / obs->T, zz);
                }
              }
            }
          }
        }
        else if ( ( plains->flag & CU_CAND_SINGLE_G ) == CU_CAND_SINGLE_G )     // Process previous results
        {
#pragma omp critical
          //for ( int x = 0; x < ACCEL_USELEN; x++ )
          for ( int step = 0; step < plains->mxSteps; step++)         // Loop over steps
          {
            for ( int stage = 0; stage < plains->noHarmStages; stage++ )
            {
              numharm   = 1 << stage;

              for ( int x = 0; x < plains->accelLen; x++ )
              {
                int idx   = step*plains->noHarmStages*plains->accelLen +   stage*plains->accelLen + x;
                poww      = plains->h_bCands[idx].sigma;

                if ( numharm  != plains->h_bCands[idx].numharm )
                {
                  int TMP = 0;
                  /// TODO this can be calculated from the stage we don't need to store it!
                }

                if ( poww > 0 )
                {
                  // We have a candidate
                  rr = ( plains->plains[0].searchRlowPrev[step] + x *  ACCEL_DR )        / (double)numharm ;
                  zz = ( plains->h_bCands[idx].z * ACCEL_DZ - plains->hInfos[0].zmax )   / (double)numharm ;

                  added = 0;
                  long grIdx = floor( rr - plains->rLow );     /// The index of the candidate in the global list
                  if ( grIdx >= 0 )
                  {
                    plains->noResults++;
                    numindep  = obs->numindep[twon_to_index(numharm)];

                    // Calculate sigma of detection
                    if ( plains->flag & FLAG_SAS_SIG )
                      sig     = poww;
                    else
                      sig     = candidate_sigma(poww, numharm, numindep);

                    if ( plains->h_candidates[grIdx].sig < sig )
                    {
                      // this sigma is greater that the current sigma for this r value
                      if ( plains->h_candidates[grIdx].sig < sig )
                      {
                        plains->h_candidates[grIdx].sig      = sig;
                        plains->h_candidates[grIdx].power    = poww;
                        plains->h_candidates[grIdx].numharm  = numharm;
                        plains->h_candidates[grIdx].r        = rr;
                        plains->h_candidates[grIdx].z        = zz;
                        added = 1;
                      }
                    }
                  }

                  if (added && !obs->dat_input)
                  {
                    //fprintf(obs->workfile, "%12.2f  %12.5f  %3d  %14.4f  %14.4f  %10.4f  GPU\n", poww, sig, numharm, rr, rr / obs->T, zz);
                  }
                }
              }
            }
          }
        }

        nvtxRangePop();

        // Do some Synchronisation
        CUDA_SAFE_CALL(cudaEventRecord(plains->processComp, plains->strmSearch),"Recording event: searchComp");

        plains->haveSData = 0;
      }
    }

    // Copy results from device to host
    if ( plains->flag & CU_CAND_SINGLE_G || plains->flag & CU_CAND_HOST )
    {
      if ( plains->haveCData )
      {
        cudaStreamWaitEvent(plains->strmSearch, plains->searchComp,  0);
        cudaStreamWaitEvent(plains->strmSearch, plains->processComp, 0);

        //CUDA_SAFE_CALL(cudaMemcpyAsync(plains->h_bCands, plains->d_bCands, ACCEL_USELEN*noStages*sizeof(accelcandBasic), cudaMemcpyDeviceToHost, plains->strmSearch), "Failed to copy results back");
        CUDA_SAFE_CALL(cudaMemcpyAsync(plains->h_bCands, plains->d_bCands, plains->accelLen*plains->noHarmStages*plains->noSteps*sizeof(accelcandBasic), cudaMemcpyDeviceToHost, plains->strmSearch), "Failed to copy results back");

        CUDA_SAFE_CALL(cudaEventRecord(plains->candCpyComp, plains->strmSearch),"Recording event: readComp");
        CUDA_SAFE_CALL(cudaGetLastError(), "COPY");

        plains->haveCData = 0;
        plains->haveSData = 1;
      }
    }
  }
  nvtxRangePop();
}

