/** @file cuda_cand_OPT.cu
 *  @brief Utility functions and kernels for GPU optimisation
 *
 *
 *  @author Chris Laidler
 *  @bug No known bugs.
 *
 *  Change Log
 *
 *  [0.0.01] []
 *    Beginning of change log
 *    Working version un-numbed
 *
 *  [0.0.02] [2017-02-16]
 *    Separated candidate and optimisation CPU threading
 *
 */

#include <curand.h>
#include <math.h>		// log
#include <curand_kernel.h>
#include <stdlib.h>
#include <stdint.h>		// uint64_t

#include "cuda_math_ext.h"
#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"
#include "candTree.h"
#include "cuda_response.h"
#include "cuda_cand_OPT.h"


#define SCALE_AUT       (1000000000)

#define OPT_INP_BUF   25

extern "C"
{
#define __float128 long double
#include "accel.h"
}

#define		NM_BEST		0
#define		NM_MIDL		1
#define		NM_WRST		2

#define		WITH_

#define SWAP_PTR(p1, p2) do { initCand* tmp = p1; p1 = p2; p2 = tmp; } while (0)

/** Shared device function to get halfwidth for optimisation planes
 *
 * Note this could be templated for accuracy
 *
 * @param z	The z (acceleration) for the relevant halfwidth
 * @param def	If a halfwidth has been supplied this is its value, multiple value could be given here
 * @return	The half width for the given z
 */
__device__ inline int getHw(float z, int def)
{
  int halfW;
  if ( def )
  {
    halfW	= def;
  }
  else
  {
    halfW	= cu_z_resp_halfwidth_high<float>(z); // NB: In original accelsearch this is (z+4) I'm not sure why?
  }

  return halfW;
}

#ifdef WITH_OPT_BLK_NRM

/** Plane generation, blocked, point per ff point
 *
 * @param pln
 * @param stream
 */
template<typename T, int noBlk>
__global__ void ffdotPlnByBlk_ker1(float* powers, float2* data, int noHarms, int halfwidth, double firstR, double firstZ, double zSZ, double rSZ, int blkDimX, int noR, int noZ, int blkWidth, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw, uint flags)
{
  const int bx = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if ( bx < blkDimX && iy < noZ)
  {
    double	r	= firstR + bx/(double)(noR-1) * rSZ ;
    double	z	= firstZ - iy/(double)(noZ-1) * zSZ ;
    if (noZ == 1)
      z = firstZ;

    float2      ans[noBlk];
    int halfW;

    int width = (noR - 1 - bx)/blkDimX+1;

    FOLD
    {
      for( int hamr = 1; hamr <= noHarms; hamr++ )           // Loop over harmonics
      {
	int	hIdx	= hamr-1;

	FOLD // Determine half width
	{
	  halfW = getHw(z*hamr, hw.val[hIdx]);
	}

	// Set complex values to 0 for this harmonic
	for( int blk = 0; blk < noBlk; blk++ )
	{
	  ans[blk].x = 0;
	  ans[blk].y = 0;
	}

	FOLD // Calculate complex value, using direct application of the convolution
	{
	  rz_convolution_cu<T, float2, float2>(&data[iStride*(hIdx)], loR.val[hIdx], iStride, r*hamr, z*hamr, halfW, ans, blkWidth*hamr, width);
	}

	// Calculate power for the harmonic
	for( int blk = 0; blk < noBlk; blk++ )
	{
	  int ix = blk*blkDimX + bx;
	  if ( ix < noR )
	  {
	    if ( flags & (uint)(FLAG_HAMRS ) )
	    {
	      // Write per harming values
	      if ( flags & (uint)(FLAG_CMPLX) )
	      {
		((float2*)powers)[iy*oStride*noHarms + (ix)*noHarms + hIdx ] = ans[blk];
	      }
	      else
		powers[iy*oStride*noHarms + (ix)*noHarms + hIdx ] = POWERF(ans[blk]);
	    }
	    else
	    {
	      // Accumulate harmonic to total sum
	      powers[iy*oStride + ix] += POWERF(ans[blk]);
	    }
	  }
	}
      }
    }
  }
}

#endif

#ifdef WITH_OPT_BLK_EXP

/** Generate respocen values
 *
 *  TODO: Fix this
 *
 * @param pln
 * @param stream
 */
__global__ void opt_genResponse_ker(cuRespPln pln)
{
  const int ix  = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy  = blockIdx.y * blockDim.y + threadIdx.y;

  const int firstBin = ix / pln.noRpnts;
  const int is  = ix % pln.noRpnts;

  if ( iy < pln.noZ && ix < pln.noR )
  {
    const double frac =  is / (double)pln.noRpnts ;

    double     zVal   = pln.zMax - (double)iy*pln.dZ ;
    double     offSet = -pln.halfWidth - frac  +  firstBin ;

    double2 response = calc_response_off(offSet, zVal);

    // Write values to memory
    pln.d_pln[iy*pln.oStride + ix ].x = (float)response.x;
    pln.d_pln[iy*pln.oStride + ix ].y = (float)response.y;
  }
}

/** Not yet implemented
 *
 * @param pln
 * @param stream
 */
template<typename T, int noBlk>
__global__ void ffdotPlnByBlk_ker2(float2* powers, float2* data, cuRespPln pln, int noHarms, int halfwidth, int zIdxTop, int rIdxLft, double zSZ, int noR, int noZ, int blkWidth, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw, uint flags)
{
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  const int is = tx % pln.noRpnts ;		///< Block Number
  const int ix = tx / pln.noRpnts ;
  const int iy = ty;

  if ( iy < noZ )
  {
    int plnCent = (pln.noZ)/2 ;

    int zIdx = plnCent - zIdxTop - iy ;

    float2	kerVal;
    float2	outVal;
    int 	halfW;

    FOLD
    {
      for( int hIdx = 0; hIdx < noHarms; hIdx++ )           // Loop over harmonics
      {
	int 	hrm 	= hIdx+1;

	int	zhIdx;		///< The harmonic specific Z index in the response
	int	shIdx;		///< The harmonic specific response step index
	int 	rhIdx;		///< The harmonic specific

	zhIdx 	= plnCent - zIdx * hrm;
	shIdx 	= is * hrm;					// Multiply we need the
	rhIdx 	= rIdxLft * hrm + shIdx / pln.noRpnts;		// Need the int part
	shIdx 	= shIdx % pln.noRpnts ;				// Adjust the remainder

	double 	zh = -1;

	FOLD // Determine half width
	{
	  halfW = getHw(zIdx * hrm * pln.dZ, hw.val[hIdx]);
	}

	if ( ix < halfW*2 )
	{
	  double off	= (ix - halfW);

	  int	hRidx 	= pln.halfWidth + (ix - halfW);

	  int khIdx = (hRidx) * pln.noRpnts + shIdx ;

	  if ( zhIdx  >= 0 && zhIdx < pln.noZ && hRidx >= 0 && hRidx < pln.halfWidth*2 )
	  {
	    kerVal = pln.d_pln[zhIdx*pln.oStride + khIdx ];
	  }
	  else
	  {
	    kerVal = calc_response_off((float)off, (float)zh);
	  }

	  int start = rhIdx - halfW - loR.val[hIdx]  ;

	  // Calculate power for the harmonic
	  for( int blk = 0; blk < noBlk; blk++ )
	  {
	    float2 inp = data[iStride*hIdx + start + blk*hrm + ix ];

#if CORRECT_MULT
	    // This is the "correct" version
	    outVal.x = (kerVal.x * inp.x - kerVal.y * inp.y);
	    outVal.y = (kerVal.x * inp.y + kerVal.y * inp.x);
#else
	    // This is the version accelsearch uses, ( added for comparison )
	    outVal.x = (kerVal.x * inp.x + kerVal.y * inp.y);
	    outVal.y = (kerVal.y * inp.x - kerVal.x * inp.y);
#endif

	    // if ( ix == 0 )
	    {
	      atomicAdd(&(powers[iy*oStride*noHarms + (is + blk*pln.noRpnts)*noHarms + hIdx].x), (float)(outVal.x));
	      atomicAdd(&(powers[iy*oStride*noHarms + (is + blk*pln.noRpnts)*noHarms + hIdx].y), (float)(outVal.y));
	    }
	  }
	}
      }
    }
  }
}

#endif

#ifdef WITH_OPT_BLK_HRM

/** Plane generation, blocked, point per ff point per harmonic
 *
 * @param pln
 * @param stream
 */
template<typename T, int noBlk>
__global__ void ffdotPlnByBlk_ker3(float* powers, float2* fft, int noHarms, int harmWidth, double firstR, double firstZ, double zSZ, double rSZ, int blkDimX, int noR, int noZ, int blkWidth, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw, uint flags)
{
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  const int	hIdx	= tx / harmWidth;
  const int	bx	= tx % harmWidth;
  const int	iy	= ty;

  if ( bx < blkDimX && iy < noZ)
  {
    double	r	= firstR + bx/(double)(noR-1) * rSZ ;
    double	z	= firstZ - iy/(double)(noZ-1) * zSZ ;
    if (noZ == 1)
      z = firstZ;

    float2      ans[noBlk];
    int halfW;

    int width = (noR - 1 - bx)/blkDimX+1;

    FOLD
    {
      int hrm = hIdx+1;

      FOLD // Determine half width
      {
	halfW = getHw(z*hrm, hw.val[hIdx]);
      }

      // Set complex values to 0 for this harmonic
      for( int blk = 0; blk < width; blk++ )
      {
	ans[blk].x = 0;
	ans[blk].y = 0;
      }

      FOLD // Calculate complex value, using direct application of the convolution
      {
	rz_convolution_cu<T, float2, float2>(&fft[iStride*hIdx], loR.val[hIdx], iStride, r*hrm, z*hrm, halfW, ans, blkWidth*hrm, width);
      }
    }

    FOLD // Write values back to memory
    {
      for( int blk = 0; blk < width; blk++ )
      {
	int ix = blk*blkDimX + bx;
	if ( ix < noR )
	{
	  if ( flags & (uint)(FLAG_HAMRS ) )
	  {
	    // Write per harming values
	    if ( flags & (uint)(FLAG_CMPLX) )
	    {
	      ((float2*)powers)[iy*oStride*noHarms + ix*noHarms + hIdx ] = ans[blk];
	    }
	    else
	      powers[iy*oStride*noHarms + ix*noHarms + hIdx ] = POWERF(ans[blk]);
	  }
	  else
	  {
	    // Accumulate harmonic to total sum
	    // This has a thread per harmonics so have to use atomic add
	    atomicAdd(&(powers[iy*oStride + ix]), POWERF(ans[blk]));
	  }
	}
      }
    }
  }
}
#endif

#ifdef WITH_OPT_PLN_NRM

/** Plane generation, points, thread per ff point
 *
 * @param pln
 * @param stream
 */
template<typename T>
__global__ void ffdotPln_ker1(float* powers, float2* data, int noHarms, int halfwidth, double firstR, double firstZ, double rSZ, double zSZ, int noR, int noZ, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw, uint flags)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if ( ix < noR && iy < noZ)
  {
    int halfW;
    double r            = firstR + ix/(double)(noR-1) * rSZ ;
    double z            = firstZ - iy/(double)(noZ-1) * zSZ ;
    if (noZ == 1)
      z = firstZ;

    T total_power  = 0;
    //T h_power  = 0;

    for( int hIdx = 0; hIdx < noHarms; hIdx++ )
    {
      T real = 0;
      T imag = 0;

      int hrm = hIdx+1;

      FOLD // Determine half width
      {
	halfW = getHw(z*hrm, hw.val[hIdx]);
      }

      rz_convolution_cu<T, float2>(&data[iStride*hIdx], loR.val[hIdx], iStride, r*hrm, z*hrm, halfW, &real, &imag);

      if ( flags & (uint)(FLAG_HAMRS ) )
      {
	// Write per harming values
	if ( flags & (uint)(FLAG_CMPLX) )
	{
	  float2 val;
	  val.x = real;
	  val.y = imag;
	  ((float2*)powers)[iy*oStride*noHarms + ix*noHarms + hIdx ] = val ;
	}
	else
	{
	  powers[iy*oStride*noHarms + ix*noHarms + hIdx ] = POWERCU(real, imag);
	}
      }
      else
      {
	// Accumulate harmonic to total sum
	total_power	+= POWERCU(real, imag);
      }
    }

    if ( total_power )
    {
      // Write incoherent sum of powers
      powers[iy*oStride + ix] = total_power;
    }
  }
}
#endif

#ifdef WITH_OPT_PLN_EXP

/** Plane generation, points, thread per convolution opperation
 *
 * This is slown and should proballu not be used
 *
 * @param pln
 * @param stream
 */
template<typename T>
__global__ void ffdotPln_ker2(float2* powers, float2* data, int noHarms, int halfwidth, double firstR, double firstZ, double rSZ, double zSZ, int noR, int noZ, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw, uint flags)
{
  const int 	tIdx	= threadIdx.y * blockDim.x + threadIdx.x;
  const int 	wIdx	= tIdx / 32;
  const int 	lIdx	= tIdx % 32;

  const int	noStps	= halfwidth / blockDim.x * blockDim.y ;
  const int 	off	= blockIdx.x * (blockDim.x * blockDim.y) + tIdx;

  const int	ix	= blockIdx.y ;

  if ( ( ix < noR ) && ( off < halfwidth*2 ) )
  {
    double	r	= firstR + ix/(double)(noR-1) * rSZ ;

    for( int harm = 1; harm <= noHarms; harm++ )
    {
      int hIdx = harm-1;

      long	idx	= (int)(r*harm) - halfwidth / 2 + off ;
      double	distD	= (r*harm) - idx ;
      long	distI	= (int)distD;
      int 	inpR	= idx - loR.val[hIdx];
      float2	inp;

      if ( inpR >= 0 && inpR < iStride )
      {
	for ( int iy = 0; iy < noZ; iy++ )
	{
	  int halfW;
	  double	z	= firstZ - iy/(double)(noZ-1) * zSZ ;
	  if (noZ == 1)
	    z = firstZ;

	  FOLD // Determine half width
	  {
	    halfW = getHw(z*harm, hw.val[hIdx]);
	  }

	  T real = 0;
	  T imag = 0;

	  if ( distI < halfW && distI > -halfW )
	  {
	    if ( inpR >= 0)
	    {
	      inp = data[iStride*hIdx + (int)(inpR)];
	      inpR = -1;	// Mark it as read
	    }

	    FOLD // Do the convolution  .
	    {
	      T resReal = 0;
	      T resImag = 0;

	      calc_response_off<T>(distD, z*harm, &resReal, &resImag);

	      FOLD 							//  Do the multiplication and sum  accumulate  .
	      {
		real = (resReal * inp.x - resImag * inp.y);
		imag = (resReal * inp.y + resImag * inp.x);
	      }
	    }

	    FOLD // Write results back to memory  .
	    {
	      /*
	       * I know using global atomics seams like a bad idea but in my testing it worked best, still bad but better than SM
	       * or even warp reduction first, crazy I know
	       */

	      atomicAdd(&(powers[iy*oStride*noHarms + ix*noHarms + hIdx].x), (float)(real));
	      atomicAdd(&(powers[iy*oStride*noHarms + ix*noHarms + hIdx].y), (float)(imag));
	    }
	  }
	}
      }
    }
  }
}
#endif

#ifdef WITH_OPT_PLN_HRM

/** Plane generation, points, thread per ff point per harmonic
 *
 * This is fast and best of the blocked kernels
 *
 * @param pln
 * @param stream
 */
template<typename T>
__global__ void ffdotPln_ker3(float* powers, float2* fft, int noHarms, int harmWidth, double firstR, double firstZ, double rSZ, double zSZ, int noR, int noZ, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw, uint flags)
{
  const int tx		= blockIdx.x * blockDim.x + threadIdx.x;
  const int ty		= blockIdx.y * blockDim.y + threadIdx.y;

  const int hIdx	= tx / harmWidth ;
  const int ix		= tx % harmWidth ;
  const int iy		= ty;

  if ( ix < noR && iy < noZ)
  {
    int halfW;
    double r            = firstR + ix/(double)(noR-1) * rSZ ;
    double z            = firstZ - iy/(double)(noZ-1) * zSZ ;
    if (noZ == 1)
      z = firstZ;

    T real = 0;
    T imag = 0;

    const int hrm = hIdx+1;

    FOLD // Determine half width
    {
      halfW = getHw(z*hrm, hw.val[hIdx]);
    }

    rz_convolution_cu<T, float2>(&fft[iStride*hIdx], loR.val[hIdx], iStride, r*hrm, z*hrm, halfW, &real, &imag);

    if ( flags & (uint)(FLAG_HAMRS ) )
    {
      // Write per harming values
      if ( flags & (uint)(FLAG_CMPLX) )
      {
	float2 val;
	val.x = real;
	val.y = imag;
	((float2*)powers)[iy*oStride*noHarms + ix*noHarms + hIdx ] = val ;
      }
      else
      {
	powers[iy*oStride*noHarms + ix*noHarms + hIdx ] = POWERCU(real, imag);
      }
    }
    else
    {
      // Accumulate harmonic to total sum
      atomicAdd(&(powers[iy*oStride + ix]), POWERCU(real, imag) );
    }
  }
}
#endif

#ifdef WITH_OPT_PLN_SHR
#ifdef CBL

/** This function is under development
 *
 * @param pln
 * @param stream
 */
template<typename T, int noHarms>
__global__ void ffdotPlnSM_ker(float* powers, float2* data, int halfwidth, double firstR, double firstZ, double rSZ, double zSZ, int noR, int noZ, int iStride, int oStride, int smLen, optLocInt_t loR, optLocInt_t hw, uint flags)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  const int	blkSz	= blockDim.x * blockDim.y;
  const int	tid	= blockDim.x * threadIdx.y + threadIdx.x;
  //const int	bid 	= blockIdx.y * gridDim.x + blockIdx.x;

  extern __shared__ float2 smmm[];

  //__shared__ unsigned int sSum;

  __syncthreads();

  //  if ( tid == 0 )
  //    sSum = 0;

  __syncthreads();

  float2* sm = smmm;

  int halfW;
  double r            = firstR + ix/(double)(noR-1) * rSZ ;
  double z            = firstZ - iy/(double)(noZ-1) * zSZ ;
  if (noZ == 1)
    z = firstZ;

  T total_power  = 0;
  T real = (T)0;
  T imag = (T)0;

  //T real_O = (T)0;
  //T imag_O = (T)0;

  int 	width;
  long	first;
  long	last;
  int	noStp;
  int	bOff;

  //double bwidth = (blockDim.x) / (double)(noR-1) * rSZ ;

  int buff = 1;

  double fP = firstR + (blockIdx.x * blockDim.x) /(double)(noR-1) * rSZ ;
  double lP = firstR + MIN(noR, ((blockIdx.x+1) * blockDim.x - 1 ) ) /(double)(noR-1) * rSZ ;

  //  if ( ix < noR && iy < noZ)
  //    //if( total_power != 0 )
  //  {
  //    //      if ( blockIdx.y == 0 )
  //    powers[iy*oStride + ix] = 0;
  //    //      else
  //    //	powers[iy*oStride + ix] = 172 ;
  //  }

  //int nno = 0;

  int bIdx = 0;
  for( int i = 1; i <= noHarms; i++ )
  {
    sm = &smmm[bIdx];

    FOLD // Calc vlas
    {
      halfW	= hw.val[i-1];
      //first	= MAX(loR.val[i-1], floor_t( (firstR + blockIdx.x * bwidth )*i ));
      //double fR = (fP)*i;
      //first	= MAX(loR.val[i-1], floor_t(fR) - halfW - buff );
      //first	= MAX(loR.val[i-1], floor_t(fP*i) - halfW - buff );
      first	= floor(fP*i) - halfW - buff ;
      last	= ceil(lP*i)  + halfW + buff ;
      //first	= floor(fR) - halfW ;
      //width	= halfW*2 + ceil_t(bwidth*i) + buff*2 ;
      //width	= halfW*2 + rSZ*i + 5;
      width 	= last - first;
      bOff	= first - loR.val[i-1];
      noStp	= ceilf( width / (float)blkSz );
      //nno	+= width;
      bIdx	+= width;
    }

    FOLD // // Load input into SM  .
    {
      //      if ( width > smLen )
      //      {
      //	printf(" width > smLen  %i > %i   tid %i  \n", width, smLen, tid );
      //      }

      //      if ( ix == 16 && iy == 16 )
      //      {
      //	printf("h: %2i  smLen: %4i  width: %4i  halfW: %4i  bwidth: %8.4f  first: %7i  loR: %7i  bOff: %3i  len: %3i r: %10.4f fr: %10.4f\n", i, smLen, width, halfW, bwidth*i, first, loR.val[i-1], bOff, bOff + width, r*i, fR );
      //      }

      __syncthreads();

      for ( int stp = 0; stp < noStp ; stp++)
      {
	int odd = stp*blkSz + tid;
	if ( odd < width /* && odd < smLen */ )
	{
	  int o2 = bOff + odd;
	  //if ( o2 < iStride )
	  {
	    //	      int tmp = 0;
	    //	    }
	    //	    else
	    //	    {
	    //	    if ( bid == 0 && i == 16 )
	    //	    {
	    //	      printf("tid: %i odd: %i \n",tid,odd);
	    //	    }

	    sm[odd] = data[(i-1)*iStride + o2 ];

	    //atomicInc(&sSum, 1000000 );
	  }
	}
      }

      //	noStp	= ceil_t(iStride / (float)blkSz);
      //	for ( int stp = 0; stp < noStp ; stp++)
      //	{
      //	  int odd = stp*blkSz + tid;
      //
      //	  if ( odd < iStride )
      //	    sm[odd] = fft[(i-1)*iStride + odd ];
      //	}

      //      if ( ix == 20 )
      //      {
      //	printf(" %03i %2i %8li %4i %4i \n", iy, i, first, width, halfW );
      //      }

      __syncthreads(); // Make sure data is written before doing the convolutions

      //      if ( ix < noR && iy < noZ)
      //      {
      //	__syncthreads(); // Make sure data is written before doing the convolutions
      //	rz_convolution_cu<T, float2>(sm, first, width, r*i, z*i, halfW, &real, &imag);
      //	total_power     += POWERCU(real, imag);
      //      }
      //
      //      __syncthreads(); // Make sure data is written before doing the convolutions

    }
  }

  __syncthreads(); // Make sure data is written before doing the convolutions


  if ( ix < noR && iy < noZ)
  {

    bIdx = 0;
    //#pragma unroll
    for( int i = 1; i <= noHarms; i++ )
    {
      sm = &smmm[bIdx];

      FOLD // Calc vlas
      {
	halfW	= hw.val[i-1];
	//first	= MAX(loR.val[i-1], floor_t( (firstR + blockIdx.x * bwidth )*i ));
	//double fR = (fP)*i;
	//first	= MAX(loR.val[i-1], floor_t(fR) - halfW - buff );
	//first	= MAX(loR.val[i-1], floor_t(fP*i) - halfW - buff );
	first	= floor(fP*i) - halfW - buff ;
	last	= ceil(lP*i)  + halfW + buff ;
	//first	= floor(fR) - halfW ;
	//width	= halfW*2 + ceil_t(bwidth*i) + buff*2 ;
	//width	= halfW*2 + rSZ*i + 5;
	width 	= last - first;
	bOff	= first - loR.val[i-1];
	noStp	= ceilf( width / (float)blkSz );
	//nno	+= width;
	bIdx	+= width;
      }

      //    if ( i != 8 )
      //      continue;

      //sm = &smmm[(i-1)*smLen];

      //      __syncthreads(); // Make sure data is written before doing the convolutions
      //
      //      FOLD // Zero
      //      {
      //	noStp	= ceil_t( smLen / (float)blkSz );
      //	float2 zz;
      //	zz.x = 0;
      //	zz.y = 0;
      //
      //	for ( int stp = 0; stp < noStp ; stp++)
      //	{
      //	  int odd = stp*blkSz + tid;
      //	  if ( odd < smLen /* && odd < smLen */ )
      //	  {
      //	    sm[odd] = zz;
      //	  }
      //	}
      //      }
      //
      //__syncthreads(); // Make sure data is written before doing the convolutions
      //__threadfence_block();
      //__threadfence();

      //    if ( ix >= noR || iy >= noZ)
      //      continue;

      //    real = (T)0.0;
      //    imag = (T)0.0;

      //__syncblocks_atomic();

      //    if ( sSum != nno )
      //    {
      //      printf("Bad2 h: %2i  tid: %3i  %5i %5i\n", i, tid, sSum, nno);
      //    }

      //halfW	= cu_z_resp_halfwidth_high<float>(z*i);

      rz_convolution_cu<T, float2>(sm, first, width, r*i, z*i, halfW, &real, &imag);

      //rz_convolution_cu<T, float2>(&fft[iStride*(i-1)], loR.val[i-1], iStride, r*i, z*i, halfW, &real, &imag);
      //rz_convolution_cu<T, float2>(&fft[iStride*(i-1)+bOff], first, width, r*i, z*i, halfW, &real, &imag);

      //      for ( int ic = 0; ic < width; ic++)
      //      {
      //	real += sm[ic].x;
      //	imag += sm[ic].y;
      //      }

      //      rz_convolution_cu<T, float2>(&fft[iStride*(i-1)+bOff], first, width, r*i, z*i, halfW, &real_O, &imag_O);
      //      if ( real != real_O || imag != imag_O )
      //      {
      //	int tmp = 0;
      //      }

      __syncthreads(); // Make sure data is written before doing the convolutions

      total_power     += POWERCU(real, imag);
    }

    //    if ( ix < noR && iy < noZ)
    //    {}
    //    else
    //    {
    //      real = (T)0.0;
    //      imag = (T)0.0;
    //    }

    //__syncthreads(); // Make sure has all been read before writing

    //__syncthreads(); // Make sure has all been read before writing

    //if ( ix < noR && iy < noZ)
    //if( total_power != 0 )
    //{
    //      if ( blockIdx.y == 0 )
    powers[iy*oStride + ix] = total_power;
    //      else
    //	powers[iy*oStride + ix] = 172 ;
  }
}

#endif

#endif

/** Create the pre-calculated response plane
 *
 * @param pln
 * @param stream
 */
void opt_genResponse(cuRespPln* pln, cudaStream_t stream)
{
#ifdef WITH_OPT_BLK_EXP
  infoMSG(5, 5, "Generating optimisation response function values.\n" );

  dim3 dimBlock, dimGrid;

  dimBlock.x = 16;
  dimBlock.y = 16;
  dimBlock.z = 1;

  cudaDeviceSynchronize();
  CUDA_SAFE_CALL(cudaGetLastError(), "Calling the ffdot_ker kernel.");

  infoMSG(6, 6, "1 Synch.\n" );

  // One block per harmonic, thus we can sort input powers in Shared memory
  dimGrid.x = ceil(pln->noR / (float)dimBlock.x);
  dimGrid.y = ceil(pln->noZ / (float)dimBlock.y);

  opt_genResponse_ker<<<dimGrid, dimBlock, 0, stream >>>(*pln);

  cudaDeviceSynchronize();
  CUDA_SAFE_CALL(cudaGetLastError(), "Calling the ffdot_ker kernel.");

#else
  fprintf(stderr, "ERROR: Not compiled with response using block optimising kernel.\n");
  exit(EXIT_FAILURE);
#endif
}


/** Get a nice text representation of the current plane kernel name
 *
 * @param pln	The plane to check options for
 * @param name	A text pointer to put the name into
 * @return
 */
ACC_ERR_CODE getKerName(cuOptCand* pln, char* name)
{
  ACC_ERR_CODE err = ACC_ERR_NONE;

  if      ( pln->flags & FLAG_OPT_BLK_NRM )
    sprintf(name,"%s","BLK_NRM" );
  else if ( pln->flags & FLAG_OPT_BLK_EXP )
    sprintf(name,"%s","BLK_EXP" );
  else if ( pln->flags & FLAG_OPT_BLK_HRM )
    sprintf(name,"%s","BLK_HRM" );
  else if ( pln->flags & FLAG_OPT_PTS_NRM )
    sprintf(name,"%s","PTS_NRM" );
  else if ( pln->flags & FLAG_OPT_PTS_EXP )
    sprintf(name,"%s","PTS_EXP" );
  else if ( pln->flags & FLAG_OPT_PTS_HRM )
    sprintf(name,"%s","PTS_HRM" );
  else if ( pln->flags &= FLAG_OPT_PTS_SHR)
    sprintf(name,"%s","PTS_SHR" );
  else
    sprintf(name,"%s","UNKNOWN" );

  return err;
}

/** Check if the plane, with current settings, requires new input
 *
 * This does not load the actual input
 * This check the input in the input data structure of the plane
 *
 * @param pln     The plane to check, current settings ( centZ, centR, zSize, rSize, etc.) used
 * @param fft     The FFT data that will make up the input
 * @param newInp  Set to 1 if new input is needed
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE ffdotPln_chkInput( cuOptCand* pln, fftInfo* fft, int* newInp)
{
  PROF // Profiling  .
  {
    NV_RANGE_PUSH("Harm INP");
  }

  infoMSG(4,4,"Check if plane needs new input.\n");

  ACC_ERR_CODE	err	= ACC_ERR_NONE;
  double	maxZ	= (pln->centZ + pln->zSize/2.0);
  double	minZ	= (pln->centZ - pln->zSize/2.0);
  double	maxR	= (pln->centR + pln->rSize/2.0);
  double	minR	= (pln->centR - pln->rSize/2.0);

  if ( !newInp )
  {
    err += ACC_ERR_NULL;
    return err;
  }
  else
  {
    // initinilse to zero
    *newInp = 0;
  }

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering ffdotPln.");

  pln->halfWidth	= cu_z_resp_halfwidth_high<double>(MAX(fabs(maxZ*pln->noHarms), fabs(minZ*pln->noHarms)) + 4 );				// NOTE this include the + 4 of original accelsearch this is not the end of the world as this is just the check

  int	datStart;		// The start index of the input data
  int	datEnd;			// The end   index of the input data

  *newInp		= 0;	// Flag whether new input is needed

  if ( pln->noHarms != pln->input->noHarms )
  {
    infoMSG(6,6,"New = True - Harms dont match.\n");
    *newInp = 1;
  }

  // Determine if new input is needed
  for( int h = 0; (h < pln->noHarms) && !(*newInp) ; h++ )
  {
    datStart        = floor( minR*(h+1) - pln->halfWidth );
    datEnd          = ceil(  maxR*(h+1) + pln->halfWidth );

    if ( datStart > fft->lastBin || datEnd <= fft->firstBin )
    {
      if ( h == 0 )
      {
	fprintf(stderr, "ERROR: Trying to optimise a candidate beyond scope of the FFT?");
	*newInp = 0;
	err += ACC_ERR_OUTOFBOUNDS;
	break;
      }
      pln->noHarms = h; // use previous harmonic
      infoMSG(6,6,"Max harms %2i - Bounds.\n", h);
      break;
    }

    if ( datStart < pln->input->loR[h] )
    {
      infoMSG(6,6,"New = True - Input harm %2i.\n", h);
      *newInp = 1;
    }
    else if ( pln->input->loR[h] + pln->input->stride < datEnd )
    {
      infoMSG(6,6,"New = True - Input harm %2i.\n", h);
      *newInp = 1;
    }
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP(); // Harm INP
  }

  return err;
}

/** Copy relevant input from FFT to data structure normalising as needed
 *
 *  Note this contains a blocking synchronisation to make sure the pinned host memory is free
 *
 * @param pln     The plane to check
 * @param fft     The FFT data that will make up the input
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE ffdotPln_prepInput( cuOptCand* pln, fftInfo* fft )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;
  confSpecsOpt*  conf		= pln->conf;
  int		off;			// Offset
  double	maxZ		= (pln->centZ + pln->zSize/2.0);
  double	minZ		= (pln->centZ - pln->zSize/2.0);
  double	maxR		= (pln->centR + pln->rSize/2.0);
  double	minR		= (pln->centR - pln->rSize/2.0);

  double	rSpread		= ceil((maxR+OPT_INP_BUF)*pln->noHarms  + pln->halfWidth) - floor((minR-OPT_INP_BUF)*pln->noHarms - pln->halfWidth);
  int		inpStride	= getStride(rSpread, sizeof(cufftComplex), pln->gInf->alignment);

  int		datStart;		// The start index of the input data
  int		datEnd;			// The end   index of the input data

  PROF	// Profiling  .
  {
    NV_RANGE_PUSH("prep Input");
  }

  // Initialise values to 0
  for( int h = 0; h < OPT_MAX_LOC_HARMS; h++)
  {
    pln->hw[h] = 0;
  }

  FOLD // Calculate normalisation factor  .
  {
    infoMSG(5,5,"New Input required - Doing all harms\n");

    pln->input->stride  = inpStride;
    pln->input->noHarms = pln->noHarms;

    if ( pln->input->stride*pln->noHarms*sizeof(cufftComplex) > pln->input->size )
    {
      fprintf(stderr, "ERROR: In function %s, cuOptCand not created with large enough input buffer.\n", __FUNCTION__);
      //fprintf(stderr, "maxZ: %.3f  minZ: %f  minR: %.1f maxR: %.1f  rSpread: %.1f  half width: %i  Harms: %i   \n", maxZ, minZ, minR, maxR, rSpread, pln->halfWidth, pln->noHarms );
      exit (EXIT_FAILURE);
    }

    FOLD // Calculate normalisation factor  .
    {
      PROF // Profiling  .
      {
	NV_RANGE_PUSH("Calc Norm factor");
      }

      for ( int i = 1; i <= pln->noHarms; i++ )
      {
	char mthd[20];	// Normalisation tetx
	if      ( conf->flags & FLAG_OPT_NRM_LOCAVE   )
	{
	  pln->input->norm[i-1]  = get_localpower3d(fft->data, fft->noBins, (pln->centR-fft->firstBin)*i, pln->centZ*i, 0.0);
	  sprintf(mthd,"2D Avle");
	}
	else if ( conf->flags & FLAG_OPT_NRM_MEDIAN1D )
	{
	  pln->input->norm[i-1]  = get_scaleFactorZ(fft->data, fft->noBins, (pln->centR-fft->firstBin)*i, pln->centZ*i, 0.0);
	  sprintf(mthd,"1D median");
	}
	else if ( conf->flags & FLAG_OPT_NRM_MEDIAN2D )
	{
	  fprintf(stderr,"ERROR: 2D median normalisation has not been written yet.\n");
	  sprintf(mthd,"2D median");
	  exit(EXIT_FAILURE);
	}
	else
	{
	  // No normalisation this is plausible but not recommended
	  pln->input->norm[i-1] = 1;
	  sprintf(mthd,"None");
	}
	infoMSG(6,6,"Harm %2i %s normalisation factor: %6.4f\n", i, mthd, pln->input->norm[i-1]);
      }

      PROF // Profiling  .
      {
	NV_RANGE_POP(); // Calc Norm factor
      }
    }
  }

  FOLD // A blocking synchronisation to make sure we can write to host memory  .
  {
    infoMSG(4,4,"Blocking synchronisation on %s", "inpCmp" );

    CUDA_SAFE_CALL(cudaEventSynchronize(pln->inpCmp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
  }

  // Calculate values for harmonics     and   normalise input and write data to host memory
  for( int h = 0; h < pln->noHarms; h++)
  {
    datStart            = floor( minR*(h+1) - pln->halfWidth );
    datEnd              = ceil(  maxR*(h+1) + pln->halfWidth );

    pln->hw[h]          = cu_z_resp_halfwidth<double>(MAX(fabs(maxZ*(h+1)), fabs(minZ*(h+1))), HIGHACC);

    if ( pln->hw[h] > pln->halfWidth )
    {
      fprintf(stderr, "ERROR: Harmonic half-width is greater than plain maximum.\n");
      pln->hw[h] = pln->halfWidth;
    }

    FOLD // Normalise input and Write data to host memory  .
    {
      int startV = MIN( ((datStart + datEnd - pln->input->stride ) / 2.0), datStart ); //Start value if the data is centred

      pln->input->loR[h]     = startV;
      double factor   = sqrt(pln->input->norm[h]);		// Correctly normalise input by the sqrt of the local power

      for ( int i = 0; i < pln->input->stride; i++ )		// Normalise input  .
      {
	off = startV - fft->firstBin + i;

	if ( off >= 0 && off < fft->noBins )
	{
	  pln->input->h_inp[h*pln->input->stride + i].r = fft->data[off].r / factor ;
	  pln->input->h_inp[h*pln->input->stride + i].i = fft->data[off].i / factor ;
	}
	else
	{
	  pln->input->h_inp[h*pln->input->stride + i].r = 0;
	  pln->input->h_inp[h*pln->input->stride + i].i = 0;
	}
      }
    }
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP(); // Harm INP
  }

  return err;
}

/** Copy pre-prepared memory from pinned hsot memory to device memory
 *
 * This assumes that the input data has been written to the pinned host memory
 *
 * @param pln     The plane to check
 * @param fft     The FFT data that will make up the input
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE ffdotPln_cpyInput( cuOptCand* pln, fftInfo* fft )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  infoMSG(4,4,"1D async memory copy H2D");

  CUDA_SAFE_CALL(cudaMemcpyAsync(pln->input->d_inp, pln->input->h_inp, pln->input->stride*pln->noHarms*sizeof(fcomplexcu), cudaMemcpyHostToDevice, pln->stream), "Copying optimisation input to the device");
  CUDA_SAFE_CALL(cudaEventRecord(pln->inpCmp, pln->stream),"Recording event: inpCmp");

  return err;
}

/** Check the configuration of how the plane section is going to be generated
 *
 * Note the configuration flags are used to set the optimiser flags
 *
 * @param pln	  optimiser
 * @param fft	  FFT data structure
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE ffdotPln_prep( cuOptCand* pln, fftInfo* fft )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;
  confSpecsOpt*	conf		= pln->conf;
  cuRespPln* 	rpln 		= pln->responsePln;

  pln->blkCnt	= 1;
  pln->blkWidth	= 1;
  pln->blkDimX	= pln->noR;

  infoMSG(4,4,"Prep plane creation. \n");

  infoMSG(5,5,"ff section, Centred on (%.6f, %.6f) with %2i harmonics.\n", pln->centR, pln->centZ, pln->noHarms );

  FOLD // Determine optimisation kernels  .
  {
    if ( conf->flags & FLAG_OPT_BLK ) // Use the block kernel  .
    {
      /*	NOTE:	Chris Laidler	22/06/2016
       *
       * The per harmonic blocked kernel is fastest in my testing
       */

      err += remOptFlag(pln, FLAG_OPT_KER_ALL);
      err += setOptFlag(pln, (conf->flags & FLAG_OPT_BLK) );

      // New method finer granularity
      if ( pln->flags & FLAG_OPT_BLK_EXP )
      {
	infoMSG(5,5,"Prep plane creation. \n");
	if ( !rpln )
	{
	  fprintf(stderr, "ERROR, Optimising with NULL response plane, reverting to standard block method.\n");
	  remOptFlag(pln, FLAG_OPT_BLK);
	  err += ACC_ERR_UNINIT;

#ifdef 	WITH_OPT_BLK_HRM
	  setOptFlag(pln, FLAG_OPT_BLK_HRM );
#elif	defined(WITH_OPT_BLK_NRM)
	  setOptFlag(pln, FLAG_OPT_BLK_NRM );
#endif
	}
	else
	{
	  // NOTE: I think this pre-calculated response value kernel has been removed?

	  pln->blkWidth	= 1;
	  pln->blkDimX	= rpln->noRpnts;
	  pln->blkCnt	= ceil(pln->rSize);

	  pln->noR	= pln->blkDimX * pln->blkCnt;
	  pln->rSize	= (pln->noR-1)/(double)rpln->noRpnts;
	  pln->lftIdx	= round( pln->centR - pln->rSize/2.0 ) ;
	  pln->centR	= pln->lftIdx + pln->rSize/2.0;

	  pln->noZ	= pln->noR;
	  pln->zSize	= (pln->noZ-1)*rpln->dZ;
	  pln->topZidx	= round( (pln->centZ + pln->zSize/2.0 )/rpln->dZ );
	  double top	= pln->topZidx*rpln->dZ;
	  pln->topZidx	= rpln->noZ/2-pln->topZidx;
	  pln->centZ	= top - pln->zSize/2.0;
	}
      }

      if ( pln->rSize <= 1.0 )
      {
	pln->blkWidth		= 1;
	pln->blkCnt		= 1;

	if ( pln->flags & FLAG_RES_FAST )
	  pln->blkDimX	= ceil( pln->noR / (double)conf->blkDivisor ) * conf->blkDivisor ;
	else
	  pln->blkDimX	= pln->noR;

	pln->noR		= pln->blkDimX;
      }
      else
      {
	infoMSG(6,6,"Orr  #R: %3i  Sz: %9.6f  Res: %9.6f \n", pln->noR, pln->rSize, pln->rSize/(double)(pln->noR-1) );

	if      ( pln->flags & FLAG_RES_CLOSE )
	{
	  // TODO: Check noR on fermi cards, the increased registers may justify using larger blocks widths
	  do
	  {
	    pln->blkWidth++;
	    pln->blkDimX	= ceil( pln->blkWidth * (pln->noR-1) / pln->rSize );
	    MINN(pln->blkDimX, pln->noR );
	    pln->blkCnt	= ceil( ( pln->rSize + 1 / (double)pln->blkDimX ) / pln->blkWidth );
	    // Can't have blocks wider than 16 - Thread block limit
	  }
	  while ( pln->blkCnt > 16 ); // TODO: Make block count a hash define

	  if ( pln->blkCnt == 1 )
	  {
	    pln->blkDimX	= pln->noR;
	  }
	  else
	  {
	    pln->noR		= ceil( pln->rSize / (double)(pln->blkWidth) * (pln->blkDimX) ) + 1 ;
	    pln->rSize		= (pln->noR-1)*(pln->blkWidth)/double(pln->blkDimX);
	  }
	}
	else if ( pln->flags & FLAG_RES_FAST  )
	{
	  // This method generally has a same or higher resolution
	  // The final width may be slightly smaller (by one resolution)
	  // The block widths are set to nice divisible numbers making the kernel a bit faster

	  pln->blkWidth		= ceil(pln->rSize / 16.0 );
	  double rPerBlock	= pln->noR / ( pln->rSize / (double)pln->blkWidth );
	  pln->blkDimX		= ceil(rPerBlock/(double)conf->blkDivisor)*conf->blkDivisor;
	  pln->blkCnt		= ceil( ( pln->rSize ) / pln->blkWidth );

	  // Check if we should increase plane width
	  if( rPerBlock < (double)conf->blkDivisor*0.80 )
	  {
	    // NOTE: Could look for higher divisors ie 3/2
	    pln->blkCnt		= ceil(pln->noR/(double)conf->blkDivisor);
	    pln->blkDimX	= conf->blkDivisor;
	    pln->blkWidth	= floor(pln->rSize/(double)pln->blkCnt);
	  }

	  pln->noR		= ceil( pln->rSize / (double)(pln->blkWidth) * (pln->blkDimX) ) + 1; // May as well get close but above
	  pln->noR		= ceil( pln->noR / (double)conf->blkDivisor ) * conf->blkDivisor ;
	  if( pln->noR > pln->blkCnt * pln->blkDimX )
	    pln->noR		= pln->blkCnt * pln->blkDimX;
	  pln->rSize		= (pln->noR-1)*(pln->blkWidth)/double(pln->blkDimX);
	}
	else
	{
	  // This will do the convolution exactly as is
	  // NOTE: If the resolution is a "good" value there is still the possibility to do it with blocks
	  // That would require some form of prime factorisation of numerator and denominator (I think), this could still be implemented

	  pln->blkDimX		= pln->noR;
	  pln->blkWidth		= 1;
	  pln->blkCnt		= 1;
	}

	infoMSG(6,6,"New  #R: %3i  Sz: %9.6f  Res: %9.6f  - Blk Width: %2i  -  No Blks: %.2f  -  Blk DimX: %2i \n", pln->noR, pln->rSize, pln->rSize/(double)(pln->noR-1), pln->blkWidth, pln->noR / (double)pln->blkDimX, pln->blkDimX );
      }

#ifdef 	WITH_OPT_PLN_HRM
      if ( pln->blkCnt == 1)
      {
	infoMSG(6,6,"Only one block, so going to use points kernel.\n");

	// In my testing a single block is faster with the points kernel
	err += remOptFlag(pln, FLAG_OPT_KER_ALL );
	err += setOptFlag(pln, FLAG_OPT_PTS_HRM );
      }
#endif
    }
    else
    {
      /*	NOTE:	Chris Laidler	22/06/2016
       *
       * I found 16 testing on a 750ti, running in synchronous mode.
       * This could probably be tested on more cards but I expect similar results
       * This relates to a optPlnDim of 16, I found anything less than 20 shows
       * significant speed up using the finer granularity kernel.
       */

      char kerName[20];

      remOptFlag(pln, FLAG_OPT_KER_ALL);
      setOptFlag(pln, (conf->flags & FLAG_OPT_PTS) );

      if ( !(pln->flags&FLAG_OPT_PTS) )
      {
#ifdef 	WITH_OPT_PLN_HRM
	setOptFlag(pln, FLAG_OPT_PTS_HRM );
#elif	defined(WITH_OPT_PLN_NRM)
	setOptFlag(pln, FLAG_OPT_PTS_NRM );
#elif	defined(WITH_OPT_PLN_EXP)
	setOptFlag(pln, FLAG_OPT_PTS_EXP );
#endif

	getKerName(pln, kerName);
	infoMSG(6,6,"Auto select points kernel %s.\n", kerName);
      }

      if ( pln->flags & FLAG_OPT_PTS_EXP )
      {
	if ( !(pln->flags&FLAG_CMPLX) || !(pln->flags&FLAG_HAMRS) )
	{
	  infoMSG(7,7,"Bad settings for expanded points kernel switching to Complex harmonics.\n" );

	  // DBG put back
	  //fprintf(stderr, "WARNING: Expanded kernel requires using complex values for each harmonic.\n");
	  pln->flags |= FLAG_CMPLX;
	  pln->flags |= FLAG_HAMRS;
	}
      }

      getKerName(pln, kerName);
      infoMSG(6,6,"Points Kernel %s\n", kerName );
    }

    if ( !(pln->flags & FLAG_HAMRS) && (pln->flags & FLAG_CMPLX) )
    {
      fprintf(stderr, "WARNING: Can't return sum of complex numbers, changing to incoherent sum of powers.\n");
      pln->flags &= ~(FLAG_CMPLX);
    }

    // All kernels use the same output stride
    pln->outStride    = pln->noR;
  }

  if ( pln->noR > pln->maxNoR )
  {
    fprintf(stderr, "ERROR: Plane number of R greater than the initialised maximum.\n");
    err += ACC_ERR_OUTOFBOUNDS;
  }
  if ( pln->noZ > pln->maxNoZ )
  {
    fprintf(stderr, "ERROR: Plane number of R greater than the initialised maximum.\n");
    err += ACC_ERR_OUTOFBOUNDS;
  }

  infoMSG(5,5,"Size (%.6f x %.6f) Points (%i x %i) %i  Resolution: %.7f r  %.7f z.\n", pln->rSize,pln->zSize, pln->noR, pln->noZ, pln->noR*pln->noZ, pln->rSize/double(pln->noR-1), pln->zSize/double(pln->noZ-1) );

  return err;
}

/** Make sure the input is for the current plane settings is ready in device memory
 *
 * This checks if new memory is needed
 * Normalises it and copies it to the device
 *
 * @param pln	  optimiser
 * @param fft	  FFT data structure
 * @param newInp  Set to 1 if new input is needed
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE ffdotPln_input( cuOptCand* pln, fftInfo* fft, int* newInp )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  // Check input
  int newInp_l;
  err += ffdotPln_chkInput( pln, fft, &newInp_l );

  if ( newInp_l ) // Copy input data to the device  .
  {
    err += ffdotPln_prepInput( pln, fft );

    err += ffdotPln_cpyInput( pln, fft );
  }

  if ( newInp )
    *newInp = newInp_l;

  return err;
}

/** Call the kernel to create the plane
 *
 * This assumes the settings for the plane have been checked - ffdotPln_prep
 * and the correct input is on the device -  ffdotPln_input
 *
 * @param pln	  The plane to generate
 * @param fft	  FFT data structure
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
template<typename T>
ACC_ERR_CODE ffdotPln_ker( cuOptCand* pln, fftInfo* fft)
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;
  confSpecsOpt*	conf		= pln->conf;
  cuRespPln* 	rpln 		= pln->responsePln;

  int		maxHW 		= 0;	// The maximum possible halfwidth of the elements being tested

  optLocInt_t	rOff;			// Row offset
  optLocInt_t	hw;			// The halfwidth for each harmonic
  optLocFloat_t	norm;			// Normalisation factor for each harmonic

  infoMSG(5,5,"Height: %5.4f z - Width: %5.4f \n", pln->zSize, pln->rSize );

  // Calculate bounds on potently newly scaled plane
  double maxZ		= (pln->centZ + pln->zSize/2.0);
  double minR		= (pln->centR - pln->rSize/2.0);

  // Initialise values to 0
  for( int h = 0; h < OPT_MAX_LOC_HARMS; h++)
  {
    rOff.val[h]		= pln->input->loR[h];
    hw.val[h]		= pln->hw[h];
    norm.val[h]		= sqrt(pln->input->norm[h]);             // Correctly normalised by the sqrt of the local power

    MAXX(maxHW, hw.val[h]);
  }

  // Halfwidth stuff
  if ( (conf->flags & FLAG_OPT_DYN_HW) || (pln->zSize >= 2) )
  {
    infoMSG(5,5,"Using dynamic half Width");
    for( int h = 0; h < pln->noHarms; h++)
    {
      hw.val[h] = 0;
    }
    maxHW = pln->halfWidth;
  }
  else
  {
    infoMSG(5,5,"Using constant half Width of %i", maxHW);
  }

  FOLD // Check output size  .
  {
    // No points
    pln->resSz = pln->outStride*pln->noZ;

    // Size of type
    if ( pln->flags & FLAG_CMPLX )
    {
      pln->resSz *= sizeof(float2);
      infoMSG(7,7,"Return complex values\n");
    }
    else
    {
      pln->resSz *= sizeof(float);
      infoMSG(7,7,"Return powers\n");
    }

    // ( FLAG_OPT_BLK_EXP | FLAG_OPT_PTS_EXP | FLAG_OPT_PTS_NRM ) )
    if ( pln->flags & FLAG_HAMRS )
    {
      pln->resSz *= pln->noHarms;			// One point per harmonic
      infoMSG(7,7,"Return individual harmonics\n");
    }
    else
    {
      infoMSG(7,7,"Return incoherent sum\n");
    }

    if ( pln->resSz > pln->outSz )
    {
      fprintf(stderr, "ERROR: Optimisation plane larger than allocated memory.\n");
      exit(EXIT_FAILURE);
    }
  }

  FOLD // Call kernel  .
  {
    dim3 dimBlock, dimGrid;

    if ( conf->flags & FLAG_SYNCH )
      CUDA_SAFE_CALL(cudaEventRecord(pln->compInit, pln->stream),"Recording event: compInit");

    uint flags =  pln->flags & ( FLAG_HAMRS | FLAG_CMPLX );

    if ( pln->flags &  FLAG_OPT_BLK )			// Use block kernel
    {
      infoMSG(4,4,"Block kernel [ No threads %i  Width %i no Blocks %i]", (int)pln->blkDimX, pln->blkWidth, pln->blkCnt);

      if      ( pln->flags & FLAG_OPT_BLK_NRM )		// Use block kernel
      {
#ifdef WITH_OPT_BLK_NRM

	infoMSG(5,5,"Block kernel 1 - Standard");

	// Thread blocks
	dimBlock.x = MIN(16, pln->blkDimX);
	dimBlock.y = MIN(16, pln->noZ);
	dimBlock.z = 1;

	// One block per harmonic, thus we can sort input powers in Shared memory
	dimGrid.x = ceil(pln->blkDimX/(float)dimBlock.x);
	dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

	CUDA_SAFE_CALL(cudaEventRecord(pln->tInit1, pln->stream),"Recording event: tInit1");

	infoMSG(6,6,"Blk %i x %i", dimBlock.x, dimBlock.y);
	infoMSG(6,6,"Grd %i x %i", dimGrid.x, dimGrid.y);

	// Call the kernel to normalise and spread the input data
	switch (pln->blkCnt)
	{
	  case 1:
	    ffdotPlnByBlk_ker1<T,1> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 2:
	    ffdotPlnByBlk_ker1<T,2> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 3:
	    ffdotPlnByBlk_ker1<T,3> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 4:
	    ffdotPlnByBlk_ker1<T,4> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 5:
	    ffdotPlnByBlk_ker1<T,5> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 6:
	    ffdotPlnByBlk_ker1<T,6> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 7:
	    ffdotPlnByBlk_ker1<T,7> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 8:
	    ffdotPlnByBlk_ker1<T,8> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 9:
	    ffdotPlnByBlk_ker1<T,9> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 10:
	    ffdotPlnByBlk_ker1<T,10><<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 11:
	    ffdotPlnByBlk_ker1<T,11><<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 12:
	    ffdotPlnByBlk_ker1<T,12><<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 13:
	    ffdotPlnByBlk_ker1<T,13><<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 14:
	    ffdotPlnByBlk_ker1<T,14><<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 15:
	    ffdotPlnByBlk_ker1<T,15><<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 16:
	    ffdotPlnByBlk_ker1<T,16><<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  default:
	  {
	    fprintf(stderr, "ERROR: %s has not been templated for %i blocks.\n", __FUNCTION__, pln->blkCnt );
	    exit(EXIT_FAILURE);
	  }
	}

	CUDA_SAFE_CALL(cudaEventRecord(pln->tComp1, pln->stream),"Recording event: tComp1");
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_BLK_NRM.\n");
	exit(EXIT_FAILURE);
#endif
      }
      else if ( pln->flags & FLAG_OPT_BLK_EXP )
      {
#ifdef WITH_OPT_BLK_EXP

	infoMSG(5,5,"Block kernel 2 - Expanded");

	dimBlock.x = 16;
	dimBlock.y = 16;

	CUDA_SAFE_CALL(cudaEventRecord(pln->tInit2, pln->stream),"Recording event: tInit2");

	cudaMemsetAsync ( pln->d_out, 0, pln->resSz, pln->stream );
	CUDA_SAFE_CALL(cudaGetLastError(), "Zeroing the output memory");

	maxHW *=2 ;

	// One block per harmonic, thus we can sort input powers in Shared memory
	dimGrid.x = ceil(maxHW*rpln->noRpnts/(float)dimBlock.x);
	dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

	// Call the kernel to normalise and spread the input data
	switch (pln->blkCnt)
	{
	  case 2:
	    ffdotPlnByBlk_ker2<T, 2> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 3:
	    ffdotPlnByBlk_ker2<T, 3> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 4:
	    ffdotPlnByBlk_ker2<T, 4> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 5:
	    ffdotPlnByBlk_ker2<T, 5> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 6:
	    ffdotPlnByBlk_ker2<T, 6> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 7:
	    ffdotPlnByBlk_ker2<T, 7> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 8:
	    ffdotPlnByBlk_ker2<T, 8> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 9:
	    ffdotPlnByBlk_ker2<T, 9> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 10:
	    ffdotPlnByBlk_ker2<T,10> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 11:
	    ffdotPlnByBlk_ker2<T,11> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 12:
	    ffdotPlnByBlk_ker2<T,12> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 13:
	    ffdotPlnByBlk_ker2<T,13> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 14:
	    ffdotPlnByBlk_ker2<T,14> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 15:
	    ffdotPlnByBlk_ker2<T,15> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 16:
	    ffdotPlnByBlk_ker2<T,16> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  default:
	  {
	    fprintf(stderr, "ERROR: %s has not been templated for %i blocks.\n", __FUNCTION__, pln->blkCnt );
	    exit(EXIT_FAILURE);
	  }
	}

	CUDA_SAFE_CALL(cudaEventRecord(pln->tComp2, pln->stream),"Recording event: tComp1");
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_BLK_EXP.\n");
	exit(EXIT_FAILURE);
#endif
      }
      else if ( pln->flags & FLAG_OPT_BLK_HRM )
      {
#ifdef WITH_OPT_BLK_HRM
	infoMSG(5,5,"Block kernel 3 - Harms");

	dimBlock.x = MIN(16, pln->blkDimX);
	dimBlock.y = MIN(16, pln->noZ);

	CUDA_SAFE_CALL(cudaEventRecord(pln->tInit2, pln->stream),"Recording event: tInit2");

	int noX = ceil(pln->blkDimX / (float)dimBlock.x);
	int harmWidth = noX*dimBlock.x;

	cudaMemsetAsync ( pln->d_out, 0, pln->resSz, pln->stream );
	CUDA_SAFE_CALL(cudaGetLastError(), "Zeroing the output memory");

	// One block per harmonic, thus we can sort input powers in shared memory
	dimGrid.x = noX * pln->noHarms ;
	dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

	// Call the kernel to normalise and spread the input data
	switch (pln->blkCnt)
	{
	  case 1:
	    // NOTE: in this case I find the points kernel to be a bit faster (~5%)
	    ffdotPlnByBlk_ker3<T, 1> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 2:
	    ffdotPlnByBlk_ker3<T, 2> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 3:
	    ffdotPlnByBlk_ker3<T, 3> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 4:
	    ffdotPlnByBlk_ker3<T, 4> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 5:
	    ffdotPlnByBlk_ker3<T, 5> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 6:
	    ffdotPlnByBlk_ker3<T, 6> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 7:
	    ffdotPlnByBlk_ker3<T, 7> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 8:
	    ffdotPlnByBlk_ker3<T, 8> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 9:
	    ffdotPlnByBlk_ker3<T, 9> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 10:
	    ffdotPlnByBlk_ker3<T,10> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 11:
	    ffdotPlnByBlk_ker3<T,11> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 12:
	    ffdotPlnByBlk_ker3<T,12> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 13:
	    ffdotPlnByBlk_ker3<T,13> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 14:
	    ffdotPlnByBlk_ker3<T,14> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 15:
	    ffdotPlnByBlk_ker3<T,15> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  case 16:
	    ffdotPlnByBlk_ker3<T,16> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw, flags);
	    break;
	  default:
	  {
	    fprintf(stderr, "ERROR: %s has not been templated for %i blocks.\n", __FUNCTION__, pln->blkCnt );
	    exit(EXIT_FAILURE);
	  }
	}

	CUDA_SAFE_CALL(cudaEventRecord(pln->tComp2, pln->stream),"Recording event: tComp1");
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_BLK_HRM.\n");
	exit(EXIT_FAILURE);
#endif
      }
      else
      {
	fprintf(stderr, "ERROR: No block optimisation specified.\n");
	exit(EXIT_FAILURE);
      }
    }
    else                  // Use normal kernel
    {
      infoMSG(4,4,"Grid kernel");

      dimBlock.x = 16;
      dimBlock.y = 16;
      dimBlock.z = 1;

      maxHW = ceil(maxHW*2/(float)dimBlock.x)*dimBlock.x;

      if      ( pln->flags &  FLAG_OPT_PTS_SHR ) // Shared mem  .
      {
#ifdef WITH_OPT_PLN_SHR
#ifdef CBL
	float smSz = 0 ;

	for( int h = 0; h < pln->noHarms; h++)
	{
	  smSz += ceil(hw.val[h]*2 + pln->rSize*(h+1) + 4 );
	}

	if ( smSz < 6144*0.9 ) // ~% of SM	10: 4915
	{

	  infoMSG(5,5,"Flat kernel 5 - SM \n");

	  // One block per harmonic, thus we can sort input powers in Shared memory
	  dimGrid.x = ceil(pln->noR/(float)dimBlock.x);
	  dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

	  //int noTB = dimGrid.x * dimGrid.y ;

	  // Call the kernel to normalise and spread the input data
	  switch (pln->noHarms)
	  {
	    case 1:
	      ffdotPlnSM_ker<T,1 ><<<dimGrid, dimBlock, smSz*sizeof(float2), pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->input->stride, pln->outStride, smSz, rOff, hw, flags);
	      break;
	    case 2:
	      ffdotPlnSM_ker<T,2 ><<<dimGrid, dimBlock, smSz*sizeof(float2), pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->input->stride, pln->outStride, smSz, rOff, hw, flags);
	      break;
	    case 4:
	      ffdotPlnSM_ker<T,4 ><<<dimGrid, dimBlock, smSz*sizeof(float2), pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->input->stride, pln->outStride, smSz, rOff, hw, flags);
	      break;
	    case 8:
	      ffdotPlnSM_ker<T,8 ><<<dimGrid, dimBlock, smSz*sizeof(float2), pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->input->stride, pln->outStride, smSz, rOff, hw, flags);
	      break;
	    case 16:
	      ffdotPlnSM_ker<T,16><<<dimGrid, dimBlock, smSz*sizeof(float2), pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->input->stride, pln->outStride, smSz, rOff, hw, flags);
	      break;
	  }
	}
#endif
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_PLN_SHR.\n");
	exit(EXIT_FAILURE);
#endif
      }
      else if ( pln->flags &  FLAG_OPT_PTS_NRM ) // Thread point  .
      {
#ifdef WITH_OPT_PLN_NRM
	infoMSG(5,5,"Flat kernel 1 - Standard\n");

	CUDA_SAFE_CALL(cudaEventRecord(pln->tInit1, pln->stream),"Recording event: tInit1");

	// One block per harmonic, thus we can sort input powers in Shared memory
	dimGrid.x = ceil(pln->noR/(float)dimBlock.x);
	dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

	// Call the kernel create a section of the f-fdot plane
	ffdotPln_ker1<T><<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->input->stride, pln->outStride, rOff, norm, hw, flags );

	CUDA_SAFE_CALL(cudaEventRecord(pln->tComp1, pln->stream),"Recording event: tComp1");
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_PLN_NRM.\n");
	exit(EXIT_FAILURE);
#endif
      }
      else if ( pln->flags &  FLAG_OPT_PTS_EXP ) // Thread response pos  .
      {
#ifdef WITH_OPT_PLN_EXP
	infoMSG(5,5,"Flat kernel 2 - Expanded\n");

	if ( !(pln->flags&FLAG_CMPLX) )
	{
	  fprintf(stderr, "ERROR: Per point plane kernel can not sum powers.");
	  exit(EXIT_FAILURE);
	}

	CUDA_SAFE_CALL(cudaEventRecord(pln->tInit2, pln->stream),"Recording event: tInit2");

	cudaMemsetAsync ( pln->d_out, 0, pln->resSz, pln->stream );
	CUDA_SAFE_CALL(cudaGetLastError(), "Zeroing the output memory");

	// One block per harmonic, thus we can sort input powers in Shared memory
	dimGrid.x = ceil(maxHW/((float)dimBlock.x*dimBlock.y) );
	dimGrid.y = ceil(pln->noZ /* * pln->noR */ ); // REM - super seeded
	dimGrid.y = pln->noR ;

	// Call the kernel create a section of the f-fdot plane
	ffdotPln_ker2<T><<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, maxHW, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->input->stride, pln->outStride, rOff, norm, hw, flags);

	CUDA_SAFE_CALL(cudaEventRecord(pln->tComp2, pln->stream),"Recording event: tComp2");
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_PLN_EXP.\n");
	exit(EXIT_FAILURE);
#endif
      }
      else if ( pln->flags &  FLAG_OPT_PTS_HRM ) // Thread point of harmonic  .
      {
#ifdef WITH_OPT_PLN_HRM
	infoMSG(5,5,"Flat kernel 3 - Harmonics\n");

	int noX = ceil(pln->noR / (float)dimBlock.x);
	int harmWidth = noX*dimBlock.x;

	CUDA_SAFE_CALL(cudaEventRecord(pln->tInit3, pln->stream),"Recording event: tInit3");

	cudaMemsetAsync ( pln->d_out, 0, pln->resSz, pln->stream );
	CUDA_SAFE_CALL(cudaGetLastError(), "Zeroing the output memory");

	// One block per harmonic, thus we can sort input powers in Shared memory
	dimGrid.x = noX * pln->noHarms ;
	dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

	// Call the kernel create a section of the f-fdot plane
	ffdotPln_ker3<T><<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->input->stride, pln->outStride, rOff, norm, hw, flags);

	CUDA_SAFE_CALL(cudaEventRecord(pln->tComp3, pln->stream),"Recording event: tComp3");
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_PLN_HRM.\n");
	exit(EXIT_FAILURE);
#endif
      }
      else
      {
	fprintf(stderr, "ERROR: No optimisation plane kernel specified.\n");
	exit(EXIT_FAILURE);
      }
    }

    CUDA_SAFE_CALL(cudaGetLastError(), "Calling the ffdot_ker kernel.");

    if ( conf->flags & FLAG_SYNCH )
      CUDA_SAFE_CALL(cudaEventRecord(pln->compCmp, pln->stream), "Recording event: compCmp");
  }

  return err;
}

/**
 *  This only calls the asynchronous copy
 *
 *  To make sure the points are in host memory call ffdotPln_ensurePln
 *
 * @param pln
 * @param fft
 * @return
 */
ACC_ERR_CODE ffdotPln_cpyResultsD2H( cuOptCand* pln, fftInfo* fft )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;
  confSpecsOpt*	conf		= pln->conf;
  cuRespPln* 	rpln 		= pln->responsePln;

  FOLD // Copy data back to host  .
  {
    infoMSG(4,4,"1D async memory copy D2H");

    CUDA_SAFE_CALL(cudaMemcpyAsync(pln->h_out, pln->d_out, pln->resSz, cudaMemcpyDeviceToHost, pln->stream), "Copying optimisation results back from the device.");
    CUDA_SAFE_CALL(cudaEventRecord(pln->outCmp, pln->stream),"Recording event: outCmp");
  }

  return err;
}

/** Ensure the values are in host memory
 *
 * This assumes the kernels has been called and the asynchronous memory copy has been called
 *
 * Block on memory copy and make sure the points have been written to host memory
 *
 * @param pln
 * @param fft
 * @return
 */
ACC_ERR_CODE ffdotPln_ensurePln( cuOptCand* pln, fftInfo* fft )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  FOLD // Wait  .
  {
    FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host  .
    {
      infoMSG(4,4,"Blocking synchronisation on %s", "outCmp" );

      PROF // Profiling  .
      {
	NV_RANGE_PUSH("EventSynch");
      }

      CUDA_SAFE_CALL(cudaEventSynchronize(pln->outCmp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");

      PROF // Profiling  .
      {
	NV_RANGE_POP(); // EventSynch
      }
    }
  }

  Fout // Calc Powers  .
  {
    //if ( pln->flags & ( FLAG_OPT_BLK_EXP | FLAG_OPT_PTS_EXP | FLAG_OPT_PTS_NRM ) )
    //if ( pln->flags & FLAG_HAMRS )
    {
      infoMSG(5,5,"Converting per harmonic powers to summed powers");

      PROF // Profiling  .
      {
	NV_RANGE_PUSH("Calc Powers");
      }

      int noHarms = pln->noHarms;

      // Complex harmonic output
      for (int indy = 0; indy < pln->noZ; indy++ )
      {
	for (int indx = 0; indx < pln->noR ; indx++ )
	{
	  float yy2 = 0;
	  for (int i = 0; i < pln->noHarms ; i++ )
	  {
	    float2 p1 = ((float2*)pln->h_out)[ indy*pln->outStride*noHarms + indx*noHarms + i ];
	    yy2 += POWERF(p1);
	  }

	  //((float*)pln->h_out)[ indy*pln->outStride + indx ] = yy2;

	  //atomicAdd(&(powers[iy*oStride*noHarms + ix*noHarms + hrm].x), (float)(real));
	  //atomicAdd(&(powers[iy*oStride*noHarms + ix*noHarms + hrm].y), (float)(imag));
	}
      }

      PROF // Profiling  .
      {
	NV_RANGE_POP(); // Calc Powers
      }
    }
  }

  return err;
}

/**  Plot a ff plane
 *
 * This assumes the plane has already been created
 *
 * @param pln
 * @param dir	Directory to place in figure in
 * @param name	File name excluding extension
 * @return
 */
ACC_ERR_CODE ffdotPln_plotPln( cuOptCand* pln, const char* dir, const char* name )
{
  infoMSG(4,4,"Plot ffdot plane section.\n");

  ACC_ERR_CODE	err		= ACC_ERR_NONE;
  char tName[1024];
  sprintf(tName,"%s/%s.csv", dir, name);
  FILE *f2 = fopen(tName, "w");

  FOLD // Write CSV  .
  {
    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Write CVS");
    }

    infoMSG(5,5,"Write CVS\n");

    // Add number of harmonics summed as the first line
    fprintf(f2,"%i", pln->noHarms);

    infoMSG(8,8,"Harms %i sz: %i x %i \n", pln->noHarms, pln->noZ, pln->noR );

    // Print R values
    for (int indx = 0; indx < pln->noR ; indx++ )
    {
      double r = pln->centR - pln->rSize/2.0 + indx/(double)(pln->noR-1) * (pln->rSize) ;
      fprintf(f2,"\t%.6f",r);
    }
    fprintf(f2,"\n");

    for (int indy = 0; indy < pln->noZ; indy++ )
    {
      // Print Z value
      double z = pln->centZ + pln->zSize/2.0 - indy/(double)(pln->noZ-1) * (pln->zSize) ;
      if ( pln->noZ == 1 )
	z = pln->centZ;
      fprintf(f2,"%.15f",z);

      // Print power
      for (int indx = 0; indx < pln->noR ; indx++ )
      {
	float yy2 = 0;
	if ( pln->flags & FLAG_HAMRS )
	{
	  for ( int hIdx = 0; hIdx < pln->noHarms; hIdx++)
	  {
	    if ( pln->flags & FLAG_CMPLX )
	      yy2 +=  POWERF(((float2*)pln->h_out)[indy*pln->outStride*pln->noHarms + indx*pln->noHarms + hIdx]);
	    else
	      yy2 +=  ((float*)pln->h_out)[indy*pln->outStride*pln->noHarms + indx*pln->noHarms + hIdx];
	  }
	}
	else
	{
	  yy2 +=  ((float*)pln->h_out)[indy*pln->outStride + indx];
	}
	fprintf(f2,"\t%.20f",yy2);
      }
      fprintf(f2,"\n");
    }
    fclose(f2);

    PROF // Profiling  .
    {
      NV_RANGE_POP(); // Write CVS
    }
  }

  FOLD // Make image  .
  {
    infoMSG(5,5,"Image\n");

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Image");
    }

    char cmd[1024];
    sprintf(cmd,"python $PRESTO/python/plt_ffd.py %s > /dev/null 2>&1", tName);
    system(cmd);

    PROF // Profiling  .
    {
      NV_RANGE_POP(); // Image
    }
  }

  return err;
}

template<typename T>
ACC_ERR_CODE ffdotPln( cuOptCand* pln, fftInfo* fft, int* newInp = NULL )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  err += ffdotPln_prep( pln,  fft );

  err += ffdotPln_input( pln, fft, newInp );

  err += ffdotPln_ker<T>( pln, fft );

  err += ffdotPln_cpyResultsD2H( pln, fft );

  err += ffdotPln_ensurePln( pln, fft );

  return err;
}

void optemiseTree(candTree* tree, cuOptCand* oPlnPln)
{
  container* cont = tree->getLargest();

  while (cont)
  {
    cont = cont->smaller;
  }
}

int addPlnToTree(candTree* tree, cuOptCand* pln)
{
  PROF // Profiling  .
  {
    NV_RANGE_PUSH("addPlnToTree");
  }

  FOLD // Get new max  .
  {
    int ggr = 0;

    for (int indy = 0; indy < pln->noZ; indy++ )
    {
      for (int indx = 0; indx < pln->noR ; indx++ )
      {
	float yy2 = ((float*)pln->h_out)[indy*pln->outStride+indx];
	{
	  initCand* canidate = new initCand;

	  canidate->numharm = pln->noHarms;
	  canidate->power   = yy2;
	  canidate->r       = pln->centR - pln->rSize/2.0 + indx/(double)(pln->noR-1) * (pln->rSize) ;
	  canidate->z       = pln->centZ + pln->zSize/2.0 - indy/(double)(pln->noZ-1) * (pln->zSize) ;
	  canidate->sig     = yy2;
	  if ( pln->noZ == 1 )
	    canidate->z = pln->centZ;
	  if ( pln->noR == 1 )
	    canidate->r = pln->centR;

	  ggr++;

	  tree->insert(canidate, 0.2 );
	}
      }
    }
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP(); // addPlnToTree
  }

  return 0;
}

candTree* opt_cont(candTree* oTree, cuOptCand* pln, container* cont, fftInfo* fft, int nn)
{
  //  PROF // Profiling  .
  //  {
  //    NV_RANGE_PUSH("opt_cont");
  //  }
  //
  //  confSpecsGen*  sSpec   = pln->cuSrch->sSpec;
  //  initCand* iCand 	= (initCand*)cont->data;

  //
  //  optInitCandLocPlns(iCand, pln, nn );
  //
  //  accelcand* cand = new accelcand;
  //  memset(cand, 0, sizeof(accelcand));
  //
  //  int lrep      = 0;
  //  int noP       = 30;
  //  float snoop   = 0.3;
  //  float sz;
  //  float v1, v2;
  //
  //  const int mxRep = 10;
  //
  //  initCand* canidate = (initCand*)cont->data;
  //
  //  candTree* thisOpt = new candTree;
  //
  //  if ( canidate->numharm == 1  )
  //    sz = conf->optPlnSiz[0];
  //  if ( canidate->numharm == 2  )
  //    sz = conf->optPlnSiz[1];
  //  if ( canidate->numharm == 4  )
  //    sz = conf->optPlnSiz[2];
  //  if ( canidate->numharm == 8  )
  //    sz = conf->optPlnSiz[3];
  //  if ( canidate->numharm == 16 )
  //    sz = conf->optPlnSiz[4];
  //
  //  //int numindep        = (obs->rhi - obs->rlo ) * (obs->zhi +1 ) * (ACCEL_DZ / 6.95) / pln->noHarms ;
  //
  //  //printf("\n%03i  r: %15.6f   z: %12.6f \n", nn, cand->r, cand->z);
  //
  //  pln->halfWidth = 0;
  //
  //  int plt = 0;
  //
  //  if ( optpln01 > 0 )
  //  {
  //    noP               = optpln01 ;
  //    lrep              = 0;
  //    canidate->power   = 0;     // Set initial power to zero
  //    do
  //    {
  //      generatePln<float>(canidate, fft, pln, noP, sz, plt++, nn );
  //
  //      container* optC =  oTree->getLargest(canidate, 1);
  //
  //      if ( optC )
  //      {
  //        // This has feature has already been optimised!
  //        cont->flag |= REMOVE_CONTAINER;
  //        NV_RANGE_POP();
  //        return thisOpt;
  //      }
  //
  //      //addPlnToTree(thisOpt, pln);
  //
  //      v1 = fabs(( pln->centR - canidate->r )/(pln->rSize/2.0));
  //      v2 = fabs(( pln->centZ - canidate->z )/(pln->zSize/2.0));
  //
  //      if ( ++lrep > mxRep )
  //      {
  //        break;
  //      }
  //    }
  //    while ( v1 > snoop || v2 > snoop );
  //    sz /= downScale;
  //  }
  //
  //  if ( optpln02 > 0 )
  //  {
  //    noP               = optpln02 ;
  //    lrep              = 0;
  //    canidate->power   = 0;     // Set initial power to zero
  //    do
  //    {
  //      generatePln<float>(canidate, fft, pln, noP, sz, plt++, nn );
  //
  //      container* optC =  oTree->getLargest(canidate, 1);
  //
  //      if ( optC )
  //      {
  //        // This has feature has already been optimised!
  //        cont->flag |= REMOVE_CONTAINER;
  //        NV_RANGE_POP();
  //        return thisOpt;
  //      }
  //
  //      //addPlnToTree(thisOpt, pln);
  //
  //      v1 = fabs(( pln->centR - canidate->r )/(pln->rSize/2.0));
  //      v2 = fabs(( pln->centZ - canidate->z )/(pln->zSize/2.0));
  //
  //      if ( ++lrep > mxRep )
  //      {
  //        break;
  //      }
  //    }
  //    while ( v1 > snoop || v2 > snoop );
  //    sz /= downScale;
  //  }
  //
  //  if ( optpln03 > 0 )
  //  {
  //    noP               = optpln03 ;
  //    lrep              = 0;
  //    canidate->power   = 0;     // Set initial power to zero
  //    do
  //    {
  //      generatePln<float>(canidate, fft, pln, noP, sz, plt++, nn );
  //
  //      container* optC =  oTree->getLargest(canidate, 1);
  //
  //      if ( optC )
  //      {
  //        // This has feature has already been optimised!
  //        cont->flag |= REMOVE_CONTAINER;
  //        NV_RANGE_POP();
  //        return thisOpt;
  //      }
  //
  //      //addPlnToTree(thisOpt, pln);
  //
  //      v1 = fabs(( pln->centR - canidate->r )/(pln->rSize/2.0));
  //      v2 = fabs(( pln->centZ - canidate->z )/(pln->zSize/2.0));
  //
  //      if ( ++lrep > mxRep )
  //      {
  //        break;
  //      }
  //    }
  //    while ( v1 > snoop || v2 > snoop );
  //    sz /= downScale*2;
  //  }
  //
  //  if ( optpln04 > 0 )
  //  {
  //    noP               = optpln04 ;
  //    lrep              = 0;
  //    canidate->power   = 0;     // Set initial power to zero
  //    do
  //    {
  //      generatePln<float>(canidate, fft, pln, noP, sz, plt++, nn );
  //
  //      container* optC =  oTree->getLargest(canidate, 1);
  //
  //      if ( optC )
  //      {
  //        // This has feature has already been optimised!
  //        cont->flag |= REMOVE_CONTAINER;
  //        NV_RANGE_POP();
  //        return thisOpt;
  //      }
  //
  //      //addPlnToTree(thisOpt, pln);
  //
  //      v1 = fabs(( pln->centR - canidate->r )/(pln->rSize/2.0));
  //      v2 = fabs(( pln->centZ - canidate->z )/(pln->zSize/2.0));
  //
  //      if ( ++lrep > mxRep )
  //      {
  //        break;
  //      }
  //    }
  //    while ( v1 > snoop || v2 > snoop );
  //    sz /= downScale*2;
  //  }
  //
  //  if ( optpln05 > 0 )
  //  {
  //    noP               = optpln05 ;
  //    lrep              = 0;
  //    canidate->power   = 0;     // Set initial power to zero
  //    do
  //    {
  //      generatePln<float>(canidate, fft, pln, noP, sz, plt++, nn );
  //
  //      container* optC =  oTree->getLargest(canidate, 1);
  //
  //      if ( optC )
  //      {
  //        // This has feature has already been optimised!
  //        cont->flag |= REMOVE_CONTAINER;
  //        NV_RANGE_POP();
  //        return thisOpt;
  //      }
  //
  //      //addPlnToTree(thisOpt, pln);
  //
  //      v1 = fabs(( pln->centR - canidate->r )/(pln->rSize/2.0));
  //      v2 = fabs(( pln->centZ - canidate->z )/(pln->zSize/2.0));
  //
  //      if ( ++lrep > mxRep )
  //      {
  //        break;
  //      }
  //    }
  //    while ( v1 > snoop || v2 > snoop );
  //    sz /= downScale*2;
  //  }
  //
  //  if ( optpln06 > 0 )
  //  {
  //    noP               = optpln06 ;
  //    lrep              = 0;
  //    canidate->power   = 0;     // Set initial power to zero
  //    do
  //    {
  //      generatePln<double>(canidate, fft, pln, noP, sz, plt++, nn );
  //
  //      container* optC =  oTree->getLargest(canidate, 1);
  //
  //      if ( optC )
  //      {
  //        // This has feature has already been optimised!
  //        cont->flag |= REMOVE_CONTAINER;
  //        NV_RANGE_POP();
  //        return thisOpt;
  //      }
  //
  //      //addPlnToTree(thisOpt, pln);
  //
  //      v1 = fabs(( pln->centR - canidate->r )/(pln->rSize/2.0));
  //      v2 = fabs(( pln->centZ - canidate->z )/(pln->zSize/2.0));
  //
  //      if ( ++lrep > mxRep )
  //      {
  //        break;
  //      }
  //    }
  //    while ( v1 > snoop || v2 > snoop );
  //    sz /= downScale*2;
  //  }
  //
  //  cont->flag |= OPTIMISED_CONTAINER;
  //
  //  NV_RANGE_POP();
  //  return thisOpt;
  return NULL;
}

template<typename T>
int optInitCandPosPln(initCand* cand, cuOptCand* pln, int noP, double scale, int plt = -1, int nn = 0, int lv = 0 )
{
  int newInput = 0;

  fftInfo*	fft	= pln->cuSrch->fft;
  confSpecsOpt*	conf	= pln->conf;

  FOLD // Generate plain points  .
  {
    pln->noZ		= noP;
    pln->noR		= noP;
    pln->rSize		= scale;
    pln->zSize		= scale*conf->zScale;
    double rRes		= pln->rSize / (double)(noP-1);
    double zRes		= pln->zSize / (double)(noP-1);

    if ( noP % 2 )
    {
      // Odd
      pln->centR	= cand->r;
      pln->centZ	= cand->z;
    }
    else
    {
      // Even
      pln->centR	= cand->r + rRes/2.0;
      pln->centZ	= cand->z - zRes/2.0;
    }

    ffdotPln<T>(pln, fft, &newInput);
    if ( newInput ) // Create the section of ff plane  .
    {
      // New input was used so don't maintain the old max
      cand->power	= 0;
    }
  }

  FOLD // Get new max  .
  {
    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Get Max");
    }

    for (int indy = 0; indy < pln->noZ; indy++ )
    {
      for (int indx = 0; indx < pln->noR ; indx++ )
      {
	float yy2 = 0;
	if ( pln->flags & FLAG_HAMRS )
	{
	  for ( int hIdx = 0; hIdx < pln->noHarms; hIdx++)
	  {
	    if ( pln->flags & FLAG_CMPLX )
	      yy2 +=  POWERF(((float2*)pln->h_out)[indy*pln->outStride*pln->noHarms + indx*pln->noHarms + hIdx]);
	    else
	      yy2 +=  ((float*)pln->h_out)[indy*pln->outStride*pln->noHarms + indx*pln->noHarms + hIdx];
	  }
	}
	else
	{
	  yy2 +=  ((float*)pln->h_out)[indy*pln->outStride + indx];
	}

	if ( yy2 > cand->power )
	{
	  cand->power	= yy2;
	  cand->r	= pln->centR - pln->rSize/2.0 + indx/(double)(pln->noR-1) * (pln->rSize) ;
	  cand->z	= pln->centZ + pln->zSize/2.0 - indy/(double)(pln->noZ-1) * (pln->zSize) ;
	  if ( pln->noZ	== 1 )
	    cand->z = pln->centZ;
	  if ( pln->noR	== 1 )
	    cand->r = pln->centR;
	}
      }
    }

    infoMSG(4,4,"Max Power %8.3f at (%.6f %.6f)\n", cand->power, cand->r, cand->z);

    PROF // Profiling  .
    {
      NV_RANGE_POP(); // Get Max
    }
  }

  FOLD // Write CVS & plot output  .
  {
#ifdef CBL
    if ( conf->flags & FLAG_DPG_PLT_OPT ) // Write CVS & plot output  .
    {
      // TODO: Check if we can get the dir name and then this can be added into standard accelsearch
      char tName[1024];
      sprintf(tName,"Cand_%05i_Rep_%02i_Lv_%i_h%02i.csv", nn, plt, lv, cand->numharm );

      ffdotPln_plotPln( pln, "/home/chris/accel/", tName );
    }
    //    if ( conf->flags & FLAG_DPG_PLT_OPT ) // Write CVS & plot output  .
    //    {
    //      infoMSG(4,4,"Write CVS\n");
    //
    //      char tName[1024];
    //      sprintf(tName,"/home/chris/accel/Cand_%05i_Rep_%02i_Lv_%i_h%02i.csv", nn, plt, lv, cand->numharm );
    //      FILE *f2 = fopen(tName, "w");
    //
    //      FOLD // Write CSV
    //      {
    //
    //	PROF // Profiling  .
    //	{
    //	  NV_RANGE_PUSH("Write CVS");
    //	}
    //
    //	// Add number of harmonics summed as the first line
    //	fprintf(f2,"%i",pln->noHarms);
    //
    //	// Print R values
    //	for (int indx = 0; indx < pln->noR ; indx++ )
    //	{
    //	  double r = pln->centR - pln->rSize/2.0 + indx/(double)(pln->noR-1) * (pln->rSize) ;
    //	  if ( pln->noR == 1 )
    //	    r = pln->centR;
    //	  fprintf(f2,"\t%.6f",r);
    //	}
    //	fprintf(f2,"\n");
    //
    //	for (int indy = 0; indy < pln->noZ; indy++ )
    //	{
    //	  // Print Z value
    //	  double z = pln->centZ + pln->zSize/2.0 - indy/(double)(pln->noZ-1) * (pln->zSize) ;
    //	  if ( pln->noZ == 1 )
    //	    z = pln->centZ;
    //	  fprintf(f2,"%.15f",z);
    //
    //	  // Print power
    //	  for (int indx = 0; indx < pln->noR ; indx++ )
    //	  {
    //	    float yy2 = ((float*)pln->h_out)[indy*pln->outStride+indx];
    //	    fprintf(f2,"\t%.20f",yy2);
    //	  }
    //	  fprintf(f2,"\n");
    //	}
    //	fclose(f2);
    //
    //	PROF // Profiling  .
    //	{
    //	  NV_RANGE_POP(); // Write CVS
    //	}
    //      }
    //
    //      FOLD // Make image  .
    //      {
    //	infoMSG(4,4,"Image\n");
    //
    //	PROF // Profiling  .
    //	{
    //	  NV_RANGE_PUSH("Image");
    //	}
    //
    //	char cmd[1024];
    //	sprintf(cmd,"python ~/bin/bin/plt_ffd.py %s > /dev/null 2>&1", tName);
    //	system(cmd);
    //
    //	PROF // Profiling  .
    //	{
    //	  NV_RANGE_POP(); // Image
    //	}
    //      }
    //    }
#endif
  }

  return newInput;
}

template<typename T>
T pow(initCand* cand, cuHarmInput* inp)
{
  int halfW;
  double r            = cand->r;
  double z            = cand->z;

  T total_power  = 0;
  T real = 0;
  T imag = 0;

  for( int i = 1; i <= cand->numharm; i++ )
  {
    // Determine half width - high precision
    halfW = cu_z_resp_halfwidth_high<float>(z*i);

    rz_convolution_cu<T, float2>(&((float2*)inp->h_inp)[(i-1)*inp->stride], inp->loR[i-1], inp->stride, r*i, z*i, halfW, &real, &imag);

    total_power     += POWERCU(real, imag);
  }

  cand->power =  total_power;

  return total_power;
}

ACC_ERR_CODE prepInput(initCand* cand, cuOptCand* pln, double sz, int *newInp)
{
  ACC_ERR_CODE	err	= ACC_ERR_NONE;
  fftInfo*	fft	= pln->cuSrch->fft;

  FOLD // Large points  .
  {
    pln->noHarms	= cand->numharm;
    pln->centR          = cand->r;
    pln->centZ          = cand->z;
    pln->rSize          = sz;
    pln->zSize          = sz*pln->conf->zScale;
  }

  // Check the input
  err += ffdotPln_chkInput( pln, fft, newInp);

  return err;
}

// Simplex method
template<typename T>
int optInitCandPosSim(initCand* cand, cuHarmInput* inp, double rSize = 1.0, double zSize = 1.0, int plt = 0, int nn = 0, int lv = 0 )
{
  infoMSG(3,3,"Simplex refine position - lvl %i  size %f by %f \n", lv+1, rSize, zSize);

  // These are the Nelder–Mead parameter values
  double reflect	= 1.0;
  double expand		= 2.0;
  double contract	= 0.4;
  double shrink		= 0.3;

  initCand  cnds[3];
  initCand* olst[3];

  initCand  centroid    = *cand;
  initCand  reflection  = *cand;
  initCand  expansion   = *cand;
  initCand  contraction = *cand;

  cnds[0] = *cand;
  cnds[1] = *cand;
  cnds[2] = *cand;

  pow<T>(&cnds[0], inp);
  double inpPow = cnds[0].power;

  cnds[1].r += rSize;
  pow<T>(&cnds[1], inp);

  cnds[2].z += zSize;
  pow<T>(&cnds[2], inp);

  olst[NM_BEST] = &cnds[0];
  olst[NM_MIDL] = &cnds[1];
  olst[NM_WRST] = &cnds[2];

  int ite = 0;
  double rtol;			///< Ratio of low to high

  infoMSG(4,4,"Start - Power: %8.3f at (%.6f %.6f)", cnds[0].power, cnds[0].r, cnds[0].z);

  while (1)
  {
    FOLD // Order
    {
      if (olst[NM_WRST]->power > olst[NM_MIDL]->power )
	SWAP_PTR(olst[NM_WRST], olst[NM_MIDL]);

      if (olst[NM_MIDL]->power > olst[NM_BEST]->power )
      {
	SWAP_PTR(olst[NM_MIDL], olst[NM_BEST]);

	if (olst[NM_WRST]->power > olst[NM_MIDL]->power )
	  SWAP_PTR(olst[NM_WRST], olst[NM_MIDL]);
      }
    }

    FOLD // Centroid  .
    {
      centroid.r = ( olst[NM_BEST]->r + olst[NM_MIDL]->r ) / 2.0  ;
      centroid.z = ( olst[NM_BEST]->z + olst[NM_MIDL]->z ) / 2.0  ;
      //pow<T>(&centroid, inp);
    }

    ite++;

    rtol = 2.0 * fabs(olst[NM_BEST]->power - olst[NM_WRST]->power) / (fabs(olst[NM_BEST]->power) + fabs(olst[NM_MIDL]->power) + 1.0e-15) ;

    if (rtol < 1.0e-7 )  // Within error so leave  .
    {
      break;
    }

    if ( ite == 100 )
    {
      break;
    }

    FOLD // Reflection  .
    {
      reflection.r = centroid.r + reflect*(centroid.r - olst[NM_WRST]->r ) ;
      reflection.z = centroid.z + reflect*(centroid.z - olst[NM_WRST]->z ) ;
      pow<T>(&reflection, inp);

      if ( olst[NM_BEST]->power <= reflection.power && reflection.power < olst[NM_MIDL]->power )
      {
	*olst[NM_WRST] = reflection;
	continue;
      }
    }

    FOLD // Expansion  .
    {
      if ( reflection.power > olst[NM_BEST]->power )
      {
	expansion.r = centroid.r + expand*(reflection.r - centroid.r ) ;
	expansion.z = centroid.z + expand*(reflection.z - centroid.z ) ;
	pow<T>(&expansion, inp);

	if (expansion.power > reflection.power)
	{
	  *olst[NM_WRST] = expansion;
	}
	else
	{
	  *olst[NM_WRST] = reflection;
	}
	continue;
      }
    }

    FOLD // Contraction  .
    {
      contraction.r = centroid.r + contract*(olst[NM_WRST]->r - centroid.r) ;
      contraction.z = centroid.z + contract*(olst[NM_WRST]->z - centroid.z) ;
      pow<T>(&contraction, inp);

      if ( contraction.power > olst[NM_WRST]->power )
      {
	*olst[NM_WRST] = contraction;
	continue;
      }
    }

    FOLD // Shrink  .
    {
      olst[NM_MIDL]->r = olst[NM_BEST]->r + shrink*(olst[NM_MIDL]->r - olst[NM_BEST]->r);
      olst[NM_MIDL]->z = olst[NM_BEST]->z + shrink*(olst[NM_MIDL]->z - olst[NM_BEST]->z);
      pow<T>(olst[NM_MIDL], inp);

      olst[NM_WRST]->r = olst[NM_BEST]->r + shrink*(olst[NM_WRST]->r - olst[NM_BEST]->r);
      olst[NM_WRST]->z = olst[NM_BEST]->z + shrink*(olst[NM_WRST]->z - olst[NM_BEST]->z);
      pow<T>(olst[NM_WRST], inp);
    }
  }

  double dist = sqrt( (cand->r-olst[NM_BEST]->r)*(cand->r-olst[NM_BEST]->r) + (cand->z-olst[NM_BEST]->z)*(cand->z-olst[NM_BEST]->z) );
  double powInc  = olst[NM_BEST]->power - inpPow;

  cand->r = olst[NM_BEST]->r;
  cand->z = olst[NM_BEST]->z;
  cand->power = olst[NM_BEST]->power;

  infoMSG(4,4,"End   - Power: %8.3f at (%.6f %.6f) %3i iterations moved %9.7f  power inc: %9.7f", cand->power, cand->r, cand->z, ite, dist, powInc);

  return 1;
}

cuHarmInput* duplicateHost(cuHarmInput* orr)
{
  if ( orr )
  {
    size_t sz = MIN(orr->size, orr->noHarms * orr->stride * sizeof(fcomplexcu) * 1.1);

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Opt derivs");
    }

    cuHarmInput* res = (cuHarmInput*)malloc(sizeof(cuHarmInput));

    memcpy(res, orr, sizeof(cuHarmInput));
    res->d_inp = NULL;
    res->h_inp = (fcomplexcu*)malloc(sz);

    memcpy(res->h_inp, orr->h_inp, res->noHarms * res->stride * sizeof(fcomplexcu));

    PROF // Profiling  .
    {
      NV_RANGE_POP(); //Opt derivs
    }

    return res;
  }
  else
  {
    return NULL;
  }
}

/** Initiate a optimisation plane
 * If oPln has not been pre initialised and is NULL it will create a new data structure.
 * If oPln has been pre initialised the device ID and Idx are used!
 *
 */
cuOptCand* initOptimiser(cuSearch* sSrch, cuOptCand* oPln, int devLstId )
{
  //confSpecsGen* sSpec = sSrch->genConf;
  confSpecsOpt*	conf	= sSrch->conf->opt;

  infoMSG(5,5,"Initialising optimiser.\n");

  FOLD // Get the possibly pre-initialised optimisation plane  .
  {
    if ( !oPln )
    {
      infoMSG(5,5,"Allocating optimisation plane.\n");

      oPln = (cuOptCand*)malloc(sizeof(cuOptCand));
      memset(oPln,0,sizeof(cuOptCand));

      if ( devLstId < MAX_GPUS )
      {
	oPln->gInf = &sSrch->gSpec->devInfo[devLstId];
      }
      else
      {
	fprintf(stderr, "ERROR: Device list index is greater that the list length, in function: %s.\n", __FUNCTION__);
	exit(EXIT_FAILURE);
      }
    }
    else
    {
      infoMSG(5,5,"Checking existing optimisation plane.\n");

      if ( oPln->gInf != &sSrch->gSpec->devInfo[devLstId] )
      {
	bool found = false;

	for ( int lIdx = 0; lIdx < MAX_GPUS; lIdx++ )
	{
	  if ( sSrch->gSpec->devInfo[lIdx].devid == oPln->gInf->devid )
	  {
	    devLstId 	= lIdx;
	    found 	= true;
	    break;
	  }
	}

	if (!found)
	{
	  if (devLstId < MAX_GPUS )
	  {
	    oPln->gInf = &sSrch->gSpec->devInfo[devLstId];
	  }
	  else
	  {
	    fprintf(stderr, "ERROR: Device list index is greater that the list length, in function: %s.\n", __FUNCTION__);
	    exit(EXIT_FAILURE);
	  }
	}
      }
    }
  }

  FOLD // Create all stuff  .
  {
    setDevice(oPln->gInf->devid) ;

    int		maxSz		= 1;					///< The max plane width in points
    int		maxWidth	= 1;					///< The max width (area) the plane can cover
    float	zMaxMax		= 1;					///< Max Z-Max this plane should be able to handle

    FOLD // Determine the largest zMaxMax  .
    {
      zMaxMax	= MAX(sSrch->sSpec->zMax+50, sSrch->sSpec->zMax*2);
      zMaxMax	= MAX(zMaxMax, 60 * sSrch->noSrchHarms );
      zMaxMax	= MAX(zMaxMax, sSrch->sSpec->zMax * 34 + 50 );  		// TODO: This may be a bit high!
    }

    FOLD // Determine max plane size  .
    {
      for ( int i=0; i < sSrch->noHarmStages; i++ )
      {
	MAXX(maxWidth, conf->optPlnSiz[i] );
      }
      for ( int i=0; i < NO_OPT_LEVS; i++ )
      {
	MAXX(maxSz, conf->optPlnDim[i]);
      }
      //#ifdef WITH_OPT_BLK_EXP // TODO: Check if this needs to go back if WITH_OPT_BLK_EXP is used
      //      MAXX(maxSz, maxWidth * conf->optResolution);
      //#endif
      oPln->maxNoR	= maxSz*1.15;					// The maximum number of r points we can handle (the extra is to cater for block kernels slight auto increase)
      oPln->maxNoZ 	= maxSz;					// The maximum number of z points we can handle
    }

    oPln->cuSrch	= sSrch;					// Set the pointer t the search specifications
    oPln->maxHalfWidth	= cu_z_resp_halfwidth<double>( zMaxMax, HIGHACC );	// The halfwidth of the largest plane we think we may handle
    oPln->conf		= conf;						// Should this rather be a duplicate?
    oPln->flags		= oPln->conf->flags;				// Individual flags allows separate configuration

    FOLD // Create streams  .
    {
      infoMSG(5,6,"Create streams.\n");

      CUDA_SAFE_CALL(cudaStreamCreate(&oPln->stream),"Creating stream for candidate optimisation.");

      PROF // Profiling, name stream  .
      {
	char nmStr[1024];
	sprintf(nmStr,"Optimisation Stream %02i", oPln->pIdx);
	NV_NAME_STREAM(oPln->stream, nmStr);
      }
    }

    FOLD // Create events  .
    {
      if ( oPln->flags & FLAG_PROF )
      {
	infoMSG(5,5,"Create Events.\n");

	CUDA_SAFE_CALL(cudaEventCreate(&oPln->inpInit),     "Creating input event inpInit." );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->inpCmp),      "Creating input event inpCmp."  );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->compInit),    "Creating input event compInit.");
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->compCmp),     "Creating input event compCmp." );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->outInit),     "Creating input event outInit." );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->outCmp),      "Creating input event outCmp."  );

	CUDA_SAFE_CALL(cudaEventCreate(&oPln->tInit1),      "Creating input event tInit1."  );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->tComp1),      "Creating input event tComp1."  );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->tInit2),      "Creating input event tInit2."  );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->tComp2),      "Creating input event tComp2."  );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->tInit3),      "Creating input event tInit3."  );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->tComp3),      "Creating input event tComp3."  );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->tInit4),      "Creating input event tInit4."  );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->tComp4),      "Creating input event tComp4."  );
      }
      else
      {
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->inpInit,	cudaEventDisableTiming),	"Creating input event inpInit." );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->inpCmp,	cudaEventDisableTiming),	"Creating input event inpCmp."  );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->compInit,	cudaEventDisableTiming),	"Creating input event compInit.");
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->compCmp,	cudaEventDisableTiming),	"Creating input event compCmp." );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->outInit,	cudaEventDisableTiming),	"Creating input event outInit." );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->outCmp,	cudaEventDisableTiming),	"Creating input event outCmp."  );

	CUDA_SAFE_CALL(cudaEventCreate(&oPln->tInit1, cudaEventDisableTiming),      "Creating input event tInit1."  );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->tComp1, cudaEventDisableTiming),      "Creating input event tComp1."  );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->tInit2, cudaEventDisableTiming),      "Creating input event tInit2."  );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->tComp2, cudaEventDisableTiming),      "Creating input event tComp2."  );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->tInit3, cudaEventDisableTiming),      "Creating input event tInit3."  );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->tComp3, cudaEventDisableTiming),      "Creating input event tComp3."  );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->tInit4, cudaEventDisableTiming),      "Creating input event tInit4."  );
	CUDA_SAFE_CALL(cudaEventCreate(&oPln->tComp4, cudaEventDisableTiming),      "Creating input event tComp4."  );
      }
    }

    FOLD // Allocate device memory  .
    {
      infoMSG(5,6,"Allocate device memory.\n");

      size_t freeMem, totalMem;
      int maxHarm = 1;

      oPln->input	= (cuHarmInput*)malloc(sizeof(cuHarmInput));
      memset(oPln->input, 0, sizeof(cuHarmInput));

      oPln->outSz	= (oPln->maxNoR * oPln->maxNoZ ) * sizeof(float2);	// This allows the possibility of returning complex value for the base plane

      // Now adjust for harmonic returns
      maxHarm		= MAX(conf->optMinLocHarms, sSrch->noSrchHarms );
      oPln->outSz	*= maxHarm;

      oPln->input->size	= (maxWidth*10 + 2*oPln->maxHalfWidth) * sSrch->noSrchHarms * sizeof(cufftComplex)*2; // The noR is oversized to allow for moves of the plane without getting new input

      CUDA_SAFE_CALL(cudaMemGetInfo ( &freeMem, &totalMem ), "Getting Device memory information");
#ifdef MAX_GPU_MEM
      long  Diff = totalMem - MAX_GPU_MEM;
      if( Diff > 0 )
      {
	freeMem  -= Diff;
	totalMem -= Diff;
      }
#endif

      if ( (oPln->input->size + oPln->outSz) > freeMem )
      {
	printf("Not enough GPU memory to create any more stacks.\n");
	free(oPln);
	return NULL;
      }
      else
      {
	infoMSG(6,6,"Input %.2f MB output %.2f MB.\n", oPln->input->size*1e-6, oPln->outSz*1e-6 );

	// Allocate device memory
	CUDA_SAFE_CALL(cudaMalloc(&oPln->d_out,  oPln->outSz),   "Failed to allocate device memory for kernel stack.");
	CUDA_SAFE_CALL(cudaMalloc(&oPln->input->d_inp,  oPln->input->size),   "Failed to allocate device memory for kernel stack.");

	// Allocate host memory
	CUDA_SAFE_CALL(cudaMallocHost(&oPln->h_out,  oPln->outSz), "Failed to allocate device memory for kernel stack.");
	CUDA_SAFE_CALL(cudaMallocHost(&oPln->input->h_inp,  oPln->input->size), "Failed to allocate device memory for kernel stack.");
      }
    }
  }

  return oPln;
}

/** Create multiplication kernel and allocate memory for planes on all devices  .
 *
 * Create the kernels on the first device and then copy it to all others
 *
 * @param sSrch     A pointer to the search structure
 *
 * @return
 */
void initOptimisers(cuSearch* sSrch )
{
  size_t free, total;                           ///< GPU memory

  infoMSG(4,4,"Initialise all optimisers.\n");

  sSrch->oInf = new cuOptInfo;
  memset(sSrch->oInf, 0, sizeof(cuOptInfo));

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering initOptimisers.");

  double halfWidth = cu_z_resp_halfwidth<double>(sSrch->sSpec->zMax+10, HIGHACC)+10;	// Candidate may be on the z-max border so buffer a bit

  cuOptCand*	devOpts[MAX_GPUS];

  FOLD // Create the primary stack on each device, this contains the kernel  .
  {
    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Init Optimisers");
    }

    // Determine the number of optimisers to make
    sSrch->oInf->noOpts = 0;
    for ( int dev = 0 ; dev < sSrch->gSpec->noDevices; dev++ ) // Loop over devices  .
    {
      if ( sSrch->gSpec->noDevOpt[dev] <= 0 )
      {
	// Use the default of 4
	sSrch->gSpec->noDevOpt[dev] = 4;

	infoMSG(5,5,"Using the default %i optimisers per GPU.\n", sSrch->gSpec->noDevOpt[dev]);
      }
      sSrch->oInf->noOpts += sSrch->gSpec->noDevOpt[dev];
    }

    infoMSG(5,5,"Initialising %i optimisers on %i devices.\n", sSrch->oInf->noOpts, sSrch->gSpec->noDevices);

    // Initialise the individual optimisers
    sSrch->oInf->opts = (cuOptCand*)malloc(sSrch->oInf->noOpts*sizeof(cuOptCand));
    memset(sSrch->oInf->opts, 0, sSrch->oInf->noOpts*sizeof(cuOptCand));

    int idx = 0;
    for ( int dev = 0 ; dev < sSrch->gSpec->noDevices; dev++ ) // Loop over devices  .
    {
      for ( int oo = 0 ; oo < sSrch->gSpec->noDevOpt[dev]; oo++ )
      {
	// Setup some basic info
	sSrch->oInf->opts[idx].pIdx	= idx;
	sSrch->oInf->opts[idx].gInf	= &sSrch->gSpec->devInfo[dev];

	initOptimiser(sSrch, &sSrch->oInf->opts[idx], dev );

	// Initialise device
	if ( oo == 0 )
	{
	  devOpts[dev] = &sSrch->oInf->opts[idx];
	}

	idx++;
      }
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP(); // Init Optimisers
    }
  }

  // Note I found the response plane method to be slower or just equivalent
  Fout // Setup response plane  .
  {
    // Set up planes
    int sz = sSrch->gSpec->noDevices*sizeof(cuRespPln); 	// The size in bytes if the plane
    sSrch->oInf->responsePlanes =  (cuRespPln*)malloc(sz);
    memset(sSrch->oInf->responsePlanes, 0, sz);
    for ( int dev = 0 ; dev < sSrch->gSpec->noDevices; dev++ ) 	// Loop over devices  .
    {
      gpuInf* gInf     	= &sSrch->gSpec->devInfo[dev];
      int device	= gInf->devid;
      cuRespPln* resp	= &sSrch->oInf->responsePlanes[dev];

      FOLD // See if we can use the cuda device and whether it may be possible to do GPU in-mem search .
      {
	infoMSG(5,6,"access device %i\n", device);

	PROF // Profiling  .
	{
	  NV_RANGE_PUSH("Get Device");
	}

	if ( device >= getGPUCount() )
	{
	  fprintf(stderr, "ERROR: There is no CUDA device %i.\n", device);
	  continue;
	}
	int currentDevvice;
	CUDA_SAFE_CALL(cudaSetDevice(device), "Failed to set device using cudaSetDevice");
	CUDA_SAFE_CALL(cudaGetDevice(&currentDevvice), "Failed to get device using cudaGetDevice");
	if (currentDevvice != device)
	{
	  fprintf(stderr, "ERROR: CUDA Device not set.\n");
	  continue;
	}
	else
	{
	  CUDA_SAFE_CALL(cudaMemGetInfo ( &free, &total ), "Getting Device memory information");
#ifdef MAX_GPU_MEM
	  long  Diff = total - MAX_GPU_MEM;
	  if( Diff > 0 )
	  {
	    free-= Diff;
	    total-=Diff;
	  }
#endif
	}

	PROF // Profiling  .
	{
	  NV_RANGE_POP(); // Get Device
	}
      }

      FOLD // Calculate the size of a response function plane  .
      {
	resp->zMax	= (ceil(sSrch->sSpec->zMax/sSrch->noSrchHarms)+20)*sSrch->noSrchHarms ;
	resp->dZ 	= sSrch->conf->opt->zScale / (double)sSrch->conf->opt->optResolution;
	resp->noRpnts	= sSrch->conf->opt->optResolution;
	resp->noZ	= resp->zMax * 2 / resp->dZ + 1 ;
	resp->halfWidth = halfWidth;
	resp->noR	= sSrch->conf->opt->optResolution*halfWidth*2 ;
	resp->oStride 	= getStride( resp->noR, sizeof(float2), sSrch->gSpec->devInfo[dev].alignment);
	resp->size	= resp->oStride * resp->noZ * sizeof(float2);
      }

      if ( resp->size < free*0.95 )
      {
	printf("Allocating optimisation response function plane %.2f MB\n", resp->size/1e6 );

	infoMSG(5, 5, "Allocating optimisation response function plane %.2f MB\n", resp->size/1e6 );

	CUDA_SAFE_CALL(cudaMalloc(&resp->d_pln,  resp->size), "Failed to allocate device memory optimisation response plane.");
	CUDA_SAFE_CALL(cudaMemsetAsync(resp->d_pln, 0, resp->size, devOpts[dev]->stream), "Failed to initiate optimisation response plane to zero");

	// This kernel isn't really necessary anymore
	//opt_genResponse(resp, devOpts[dev]->stream);

	for ( int optN = 0; optN < sSrch->oInf->noOpts; optN++ )
	{
	  cuOptCand* oCnd = &sSrch->oInf->opts[optN];

	  if ( oCnd->gInf->devid == devOpts[dev]->gInf->devid )
	  {
	    oCnd->responsePln = resp;
	  }
	}
      }
      else
      {
	fprintf(stderr,"WARNING: Not enough free GPU memory to use a response plane for optimisation. Pln needs %.2f GB there is %.2f GB. \n", resp->size/1e9, free/1e9 );
	memset(resp, 0, sizeof(cuRespPln) );
      }
    }
  }

}

cuSearch* initCuOpt(cuSearch* srch)
{
  //if ( !srch )
  //  srch = initSearchInf(sSpec, gSpec, srch);

  PROF // Profiling  .
  {
    NV_RANGE_PUSH("Init CUDA optimisers");
  }

  if ( !srch->oInf )
  {
    initOptimisers( srch );
  }
  else
  {
    // TODO: Do a whole bunch of checks here!
    fprintf(stderr, "ERROR: %s has not been set up to handle a pre-initialised memory info data structure.\n", __FUNCTION__);
    exit(EXIT_FAILURE);
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP();	// Init CUDA optimisers
  }

  return srch;
}

void freeHarmInput(cuHarmInput* inp)
{
  if ( inp )
  {
    cudaFreeNull(inp->d_inp);
    freeNull(inp->h_inp);
    freeNull(inp);
  }
}

/** Optimise derivatives of a candidate  .
 *
 */
void* optCandDerivs(accelcand* cand, cuSearch* srch )
{
  int ii;
  struct timeval start, end;    // Profiling variables

  fftInfo*	fft	= srch->fft;
  confSpecsOpt*	conf	= srch->conf->opt;
  searchSpecs*	sSpec	= srch->sSpec;

  FOLD // Update fundamental values to the optimised ones  .
  {
    infoMSG(5,5,"DERIVS\n");

    float	maxSig		= 0;
    int		bestH		= 0;
    float	bestP		= 0;
    double  	sig		= 0; // can be a float
    long long	numindep;
    float	candHPower	= 0;
    int		noStages	= 0;
    int 	kern_half_width;
    double	locpow;
    double	real;
    double	imag;
    double	power;
    int		maxHarms  	= MAX(cand->numharm, conf->optMinRepHarms) ;

    PROF // Profiling  .
    {
      if ( !(!(conf->flags & FLAG_SYNCH) && (conf->flags & FLAG_OPT_THREAD)) )
      {
	NV_RANGE_PUSH("DERIVS");
      }

      if ( conf->flags & FLAG_PROF )
      {
	gettimeofday(&start, NULL);
      }
    }

    cand->power   = 0;

    // Set up candidate
    cand->pows    = gen_dvect(maxHarms);
    cand->hirs    = gen_dvect(maxHarms);
    cand->hizs    = gen_dvect(maxHarms);
    cand->derivs  = (rderivs *)   malloc(sizeof(rderivs)  * maxHarms  );

    // Initialise values
    for( ii=0; ii < maxHarms; ii++ )
    {
      cand->hirs[ii]  = cand->r*(ii+1);
      cand->hizs[ii]  = cand->z*(ii+1);
    }

    for( ii = 1; ii <= maxHarms; ii++ )			// Calculate derivatives, powers and sigma for all harmonics  .
    {
      if      ( conf->flags & FLAG_OPT_NRM_LOCAVE   )
      {
	locpow = get_localpower3d(fft->data, fft->noBins, cand->r*ii, cand->z*ii, 0.0);
      }
      else if ( conf->flags & FLAG_OPT_NRM_MEDIAN1D )
      {
	locpow = get_scaleFactorZ(fft->data, fft->noBins, cand->r*ii, cand->z*ii, 0.0);
      }
      else if ( conf->flags & FLAG_OPT_NRM_MEDIAN2D )
      {
	fprintf(stderr,"ERROR: 2D median normalisation has not been written yet.\n");
	exit(EXIT_FAILURE);
      }
      else
      {
	// No normalisation this is plausible but not recommended

	// TODO: This should error if it is out of bounds?
	locpow = 1;
      }

      if ( locpow )
      {
	kern_half_width   = cu_z_resp_halfwidth<double>(fabs(cand->z*ii), HIGHACC);

	rz_convolution_cu<double, float2>((float2*)fft->data, fft->firstBin, fft->noBins, cand->r*ii, cand->z*ii, kern_half_width, &real, &imag);

	// Normalised power
	power = POWERCU(real, imag) / locpow ;

	cand->pows[ii-1] = power;

	get_derivs3d(fft->data, fft->noBins, cand->r*ii, cand->z*ii, 0.0, locpow, &cand->derivs[ii-1] );

	cand->power	+= power;
	int numz 	= round(srch->conf->gen->zMax / srch->conf->gen->zRes) * 2 + 1;
	if ( numz == 1 )
	{
	  numindep	= (sSpec->searchRHigh - sSpec->searchRLow) / (double)(ii) ;
	}
	else
	{
	  numindep	= (sSpec->searchRHigh - sSpec->searchRLow) * (numz + 1) * ( srch->conf->gen->zRes / 6.95 ) / (double)(ii);
	}

	sig		= candidate_sigma_cu(cand->power, (ii), numindep );

	infoMSG(6,6,"Harm %2i  local power %6.3f, normalised power %8.3f,   sigma %5.2f \n", ii, locpow, power, sig );

	if ( sig > maxSig || ii == 1 )
	{
	  maxSig        = sig;
	  bestP         = cand->power;
	  bestH         = (ii);
	}

	if ( ii == cand->numharm )
	{
	  candHPower    = cand->power;

	  if ( !(conf->flags & FLAG_OPT_BEST) )
	  {
	    break;
	  }
	}
      }
    }

    // Final values
    if ( bestP && (conf->flags & FLAG_OPT_BEST) && ( maxSig > 0.001 ) )
    {
      cand->numharm	= bestH;
      cand->sigma	= maxSig;
      cand->power	= bestP;

      infoMSG(4,4,"Cand best val Sigma: %5.2f Power: %6.4f  %i harmonics summed.", maxSig, bestP, bestH);
    }
    else
    {
      cand->power	= candHPower;
      noStages		= log2((double)cand->numharm);
      numindep		= srch->numindep[noStages];
      cand->sigma	= candidate_sigma_cu(candHPower, cand->numharm, numindep);

      infoMSG(4,4,"Cand harm val Sigma: %5.2f Power: %6.4f  %i harmonics summed.", cand->sigma, cand->power, cand->numharm);
    }

    PROF // Profiling  .
    {
      if ( !(!(conf->flags & FLAG_SYNCH) && (conf->flags & FLAG_OPT_THREAD)) )
      {
	NV_RANGE_POP(); // DERIVS
      }

      if ( conf->flags & FLAG_PROF )
      {
	gettimeofday(&end, NULL);
	float v1 =  (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);

	// Thread (pthread) safe add to timing value
	pthread_mutex_lock(&srch->threasdInfo->candAdd_mutex);
	srch->timings[COMP_OPT_DERIVS] += v1;
	pthread_mutex_unlock(&srch->threasdInfo->candAdd_mutex);
      }
    }
  }

  return (NULL);
}

/** CPU process results
 *
 * This function is meant to be the entry of a separate thread
 *
 */
void* cpuProcess(void* ptr)
{
  candSrch*	res	= (candSrch*)ptr;
  cuSearch*	srch	= res->cuSrch;

  struct timeval start, end;    // Profiling variables

  accelcand*    cand	= res->cand;
  confSpecsOpt*	conf	= srch->conf->opt;

  if ( conf->flags & FLAG_OPT_NM_REFINE )
  {
    PROF // Profiling  .
    {
      if ( !(!(conf->flags & FLAG_SYNCH) && (conf->flags & FLAG_OPT_THREAD)) )
      {
	NV_RANGE_PUSH("NM_REFINE");
      }

      if ( conf->flags & FLAG_PROF )
      {
	gettimeofday(&start, NULL);
      }
    }

    initCand iCand;
    iCand.numharm	= cand->numharm;
    iCand.power		= cand->power;
    iCand.r		= cand->r;
    iCand.z		= cand->z;

    // Run the NM
    optInitCandPosSim<double>(&iCand,  res->input, 0.0005, 0.0005*conf->optPlnScale );

    cand->r		= iCand.r;
    cand->z		= iCand.z;
    cand->power		= iCand.power;

    // Free thread specific input memory
    freeHarmInput(res->input);
    res->input = NULL;

    PROF // Profiling  .
    {
      if ( !(!(conf->flags & FLAG_SYNCH) && (conf->flags & FLAG_OPT_THREAD)) )
      {
	NV_RANGE_POP(); // NM_REFINE
      }

      if ( conf->flags & FLAG_PROF )
      {
	gettimeofday(&end, NULL);
	float v1 =  (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);

	// Thread (pthread) safe add to timing value
	pthread_mutex_lock(&res->cuSrch->threasdInfo->candAdd_mutex);
	srch->timings[COMP_OPT_REFINE_2] += v1;
	pthread_mutex_unlock(&res->cuSrch->threasdInfo->candAdd_mutex);
      }
    }
  }

  optCandDerivs(cand, srch);

  // Decrease the count number of running threads
  sem_trywait(&srch->threasdInfo->running_threads);

  free(res);

  return (NULL);
}

/** Optimise derivatives of a candidate Using the CPU  .
 * This usually spawns a separate CPU thread to do the sigma calculations
 */
void processCandDerivs(accelcand* cand, cuSearch* srch, cuHarmInput* inp = NULL, int candNo = -1)
{
  infoMSG(2,2,"Calc Cand Derivatives.\n");

  candSrch*     thrdDat  = new candSrch;
  memset(thrdDat, 0, sizeof(candSrch));

  confSpecsOpt*	conf	= srch->conf->opt;

  thrdDat->cand   = cand;
  thrdDat->cuSrch = srch;
  thrdDat->candNo = candNo;

  if ( conf->flags & FLAG_OPT_NM_REFINE )
  {
    // Make a copy of the input data for the thread to use
    thrdDat->input = duplicateHost(inp);
  }

  PROF // Profiling  .
  {
    NV_RANGE_PUSH("Post Thread");
  }

  // Increase the count number of running threads
  sem_post(&srch->threasdInfo->running_threads);

  if ( !(conf->flags & FLAG_SYNCH) && (conf->flags & FLAG_OPT_THREAD) )  // Create thread  .
  {
    pthread_t thread;
    int  iret1 = pthread_create( &thread, NULL, cpuProcess, (void*) thrdDat);

    if (iret1)	// Check return status
    {
      fprintf(stderr,"Error - pthread_create() return code: %d\n", iret1);
      exit(EXIT_FAILURE);
    }
  }
  else                              // Just call the function  .
  {
    cpuProcess( (void*) thrdDat );
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP(); // Post Thread
  }

  infoMSG(2,2,"Done");
}

/** This is the main function called by external elements  .
 *
 * @param cand		The canidate to refine
 * @param pln		The plane data structure to use for the GPU position refinement
 * @param candNo	The index of the candidate being optimised
 */
void optInitCandLocPlns(initCand* cand, cuOptCand* pln, int candNo )
{
  infoMSG(2,2,"Refine location by plain\n");

  PROF // Profiling  .
  {
    NV_RANGE_PUSH("Plns");
  }

  confSpecsOpt*	conf	= pln->conf;

  // Number of harmonics to check, I think this could go up to 32!
  int maxHarms	= MAX(cand->numharm, conf->optMinLocHarms);

  // Setup GPU plane
  pln->centR	= cand->r ;
  pln->centZ	= cand->z ;
  pln->noHarms	= maxHarms ;

  FOLD // Get best candidate location using iterative GPU planes  .
  {
    int depth;
    int noP;
    int rep	= 0;
    int lrep	= 0;
    bool doub	= false;
    const int	mxRep		= 10;
    const float moveBound	= 0.67;
    const float outBound	= 0.9;
    double sz;
    float posR, posZ;

    if ( pln->noHarms == 1  )
      sz = conf->optPlnSiz[0];
    if ( pln->noHarms == 2  )
      sz = conf->optPlnSiz[1];
    if ( pln->noHarms == 4  )
      sz = conf->optPlnSiz[2];
    if ( pln->noHarms == 8  )
      sz = conf->optPlnSiz[3];
    if ( pln->noHarms == 16 )
      sz = conf->optPlnSiz[4];

    pln->halfWidth 	= 0;
    cand->power	= 0;					// Set initial power to zero

    for ( int lvl = 0; lvl < NO_OPT_LEVS; lvl++ )
    {
      noP		= conf->optPlnDim[lvl] ;	// Set in the defaults text file

      lrep		= 0;
      depth		= 1;

      if ( noP )					// Check if there are points in this plane ie. are we optimising position at this level  .
      {
	if ( ( lvl == NO_OPT_LEVS-1 ) || (sz < 0.002) /*|| ( (sz < 0.06) && (abs(pln->centZ) < 0.05) )*/ )	// Potently force double precision
	{
	  // If last plane is not 0, it will be done with double precision
	  if (!doub)
	    cand->power = 0;

	  doub = true;
	}

	while ( (depth > 0) && (lrep < mxRep) )		// Recursively make planes at this scale  .
	{
	  if ( doub )
	  {
	    infoMSG(3,3,"Generate double precision plane - lvl %i  depth: %i  iteration %2i\n", lvl+1, depth, lrep);

	    // Double precision
	    optInitCandPosPln<double>(cand, pln, noP, sz,  rep++, candNo, lvl + 1 );
	  }
	  else
	  {
	    infoMSG(3,3,"Generate single precision plane - lvl %i  depth: %i  iteration %2i\n", lvl+1, depth, lrep);

	    // Standard single precision
	    optInitCandPosPln<float>(cand, pln, noP, sz,  rep++, candNo, lvl + 1 );
	  }

	  posR = fabs(( pln->centR - cand->r )/(pln->rSize/2.0));
	  posZ = fabs(( pln->centZ - cand->z )/(pln->zSize/2.0));

	  double rRes = pln->rSize/(double)(pln->noR-1) ;

	  if ( posR > moveBound || posZ > moveBound )
	  {
	    if ( ( (posR > outBound) || (posZ > outBound) ) && ( depth < lvl+1) )
	    {
	      // Zoom out by half
	      sz *= conf->optPlnScale / 2.0 ;
	      depth++;
	      infoMSG(5,5,"Zoom out");
	    }
	    else
	    {
	      // we'r just going to move the plane
	      infoMSG(5,5,"Move plain");
	    }
	  }
	  else
	  {
	    // Break condition
	    if ( rRes < 1e-5 )
	    {
	      infoMSG(5,5,"Break size is small enough\n");
	      break;
	    }

	    // Zoom in
	    sz /= conf->optPlnScale;
	    depth--;
	    infoMSG(5,5,"Zoom in\n");
	    if ( sz < 2.0*rRes )
	      sz = rRes*2.0;
	  }

	  ++lrep;
	}
      }
      else
      {
	if ( doub )
	  infoMSG(3,3,"Skip plane lvl %i (double precision)", lvl+1);
	else
	  infoMSG(3,3,"Skip plane lvl %i (single precision)", lvl+1);
      }
    }
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP(); // Plns
  }
}

/** This is the main function called by external elements  .
 *
 * @param cand
 * @param pln
 * @param nn
 */
void opt_accelcand(accelcand* cand, cuOptCand* pln, int candNo)
{
  confSpecsOpt*  conf	= pln->conf;

  PROF // Profiling  .
  {
    char Txt[1024];
    sprintf(Txt, "Opt Cand %03i", candNo);

    NV_RANGE_PUSH(Txt);
  }

  initCand iCand;				// plane refining uses an initial candidate data structure
  iCand.r 		= cand->r;
  iCand.z 		= cand->z;
  iCand.power		= cand->power;
  iCand.numharm 	= cand->numharm;

  FOLD // Refine position in ff space  .
  {
    struct timeval start, end;    // Profiling variables

    PROF // Profiling  .
    {
	NV_RANGE_PUSH("Refine pos");

	if ( conf->flags & FLAG_PROF )
	{
	  gettimeofday(&start, NULL);
	}
    }

    if      ( conf->flags & FLAG_OPT_NM )
    {
      prepInput(&iCand, pln, 15);
      optInitCandPosSim<double>(&iCand, pln->input, 0.5, 0.5*conf->optPlnScale);
    }
    else if ( conf->flags & FLAG_OPT_SWARM )
    {
      fprintf(stderr,"ERROR: Particle swarm optimisation has been removed.\n");
      exit(EXIT_FAILURE);
    }
    else // Default use planes
    {
      optInitCandLocPlns(&iCand, pln, candNo);
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP();	// Refine pos

      if ( conf->flags & FLAG_PROF )
      {
	gettimeofday(&end, NULL);
	float v1 =  (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec) ;

	// Thread (omp) safe add to timing value
#pragma omp atomic
	pln->cuSrch->timings[COMP_OPT_REFINE_1] += v1;
      }
    }
  }

  // Update the details of the final candidate from the updated initial candidate
  cand->r 		= iCand.r;
  cand->z 		= iCand.z;
  cand->power		= iCand.power;
  cand->numharm 	= iCand.numharm;

  FOLD // Optimise derivatives  .
  {
    prepInput(&iCand, pln, 15);
    processCandDerivs(cand, pln->cuSrch, pln->input,  candNo);
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP(); // Txt
  }
}

int optList(GSList *listptr, cuSearch* cuSrch)
{
  struct timeval start, end;

  TIME //  Timing  .
  {
      NV_RANGE_PUSH("GPU Kernels");
  }

  int numcands 	= g_slist_length(listptr);

  int ii	= 0;
  int comp	= 0;

#if	!defined(DEBUG) && defined(WITHOMP)   // Parallel if we are not in debug mode  .
  if ( cuSrch->conf->opt->flags & FLAG_SYNCH )
  {
    omp_set_num_threads(1);
  }
  else
  {
    omp_set_num_threads(cuSrch->oInf->noOpts);
  }
#pragma omp parallel
#endif	// !DEBUG && WITHOMP
  FOLD  	// Main GPU loop  .
  {
    accelcand *candGPU;

    int tid         = 0;
    int ti          = 0; // tread specific index
#ifdef	WITHOMP
    omp_get_thread_num();
#endif	// WITHOMP

    cuOptCand* oPlnPln = &(cuSrch->oInf->opts[tid]);

    setDevice(oPlnPln->gInf->devid) ;

    // Make sure all initialisation and other stuff on the device is complete
    CUDA_SAFE_CALL(cudaDeviceSynchronize(), "Synchronising device before candidate generation");

    while (listptr)  // Main Loop  .
    {
#pragma omp critical
      FOLD  // Synchronous behaviour  .
      {
#ifndef  DEBUG
	if ( cuSrch->conf->opt->flags & FLAG_SYNCH )
#endif
	{
	  tid 		= ii % cuSrch->oInf->noOpts ;
	  oPlnPln 	= &(cuSrch->oInf->opts[tid]);
	  setDevice(oPlnPln->gInf->devid);
	}

	FOLD // Calculate candidate  .
	{
	  if ( listptr )
	  {
	    candGPU	= (accelcand *) (listptr->data);
	    listptr	= listptr->next;
	    ii++;
	    ti = ii;
#ifdef CBL
	    FOLD // TMP: This can get removed
	    {
	      candGPU->init_power    = candGPU->power;
	      candGPU->init_sigma    = candGPU->sigma;
	      candGPU->init_numharm  = candGPU->numharm;
	      candGPU->init_r        = candGPU->r;
	      candGPU->init_z        = candGPU->z;
	    }
#endif
	  }
	  else
	  {
	    candGPU = NULL;
	  }
	}
      }

      if ( candGPU ) // Optimise  .
      {
	infoMSG(2,2,"\nOptimising initial candidate %i/%i, Power: %.3f  Sigma %.2f  Harm %i at (%.3f %.3f)\n", ti, numcands, candGPU->power, candGPU->sigma, candGPU->numharm, candGPU->r, candGPU->z );

	opt_accelcand(candGPU, oPlnPln, ti);

#pragma omp atomic
	comp++;

	if ( msgLevel == 0 )
	{
	  printf("\rGPU optimisation %5.1f%% complete   ", comp / (float)numcands * 100.0f );
	  fflush(stdout);
	}
      }
    }
  }

  printf("\rGPU optimisation %5.1f%% complete                      \n", 100.0f );

  TIME //  Timing  .
  {
    NV_RANGE_POP(); // GPU Kernels
    gettimeofday(&start, NULL);
  }

  // Wait for CPU derivative threads to finish
  waitForThreads(&cuSrch->threasdInfo->running_threads, "Waiting for CPU threads to complete.", 200 );

  TIME //  Timing  .
  {
    gettimeofday(&end, NULL);
    cuSrch->timings[TIME_OPT_WAIT] += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
  }

  return 0;
}


template ACC_ERR_CODE ffdotPln<float >( cuOptCand* pln, fftInfo* fft, int* newInput );
template ACC_ERR_CODE ffdotPln<double>( cuOptCand* pln, fftInfo* fft, int* newInput );

template ACC_ERR_CODE ffdotPln_ker<float >( cuOptCand* pln, fftInfo* fft );
template ACC_ERR_CODE ffdotPln_ker<double>( cuOptCand* pln, fftInfo* fft );
