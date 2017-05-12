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

#ifdef WITH_OPT_BLK1

template<typename T, int noBlk>
__global__ void ffdotPlnByBlk_ker(float* powers, float2* data, int noHarms, int halfwidth, double firstR, double firstZ, double zSZ, int noR, int noZ, int blkWidth, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if ( ix < noR && iy < noZ)
  {
    double r            = firstR + ix*blkWidth/(double)(noR) ;
    double z            = firstZ - iy/(double)(noZ-1) * zSZ ;
    if (noZ == 1)
      z = 0;

    float       total_power[noBlk];
    float2      ans[noBlk];
    int halfW;

    for( int blk = 0; blk < noBlk; blk++ )
    {
      total_power[blk] = 0;
    }

    FOLD
    {
      for( int i = 1; i <= noHarms; i++ )           // Loop over harmonics
      {
	double absz         = fabs(z*i);

	FOLD // Determine half width
	{
	  if ( hw.val[i-1] )
	  {
	    halfW	= hw.val[i-1];
	  }
	  else
	  {
	    halfW       = cu_z_resp_halfwidth_high<float>(z*i); // NB this was (z*i+4) I'm not sure why?
	  }
	}

	// Set complex values to 0 for this harmonic
	for( int blk = 0; blk < noBlk; blk++ )
	{
	  ans[blk].x = 0;
	  ans[blk].y = 0;
	}

	FOLD // Calculate complex value, using direct application of the convolution
	{
	  rz_convolution_cu<T, float2, float2, noBlk>(&data[iStride*(i-1)], loR.val[i-1], iStride, r*i, z*i, halfW, ans, blkWidth*i);
	}

	// Calculate power for the harmonic
	for( int blk = 0; blk < noBlk; blk++ )
	{
	  total_power[blk] += POWERF(ans[blk]);
	}
      }
    }

    // Write values back to
    for( int blk = 0; blk < noBlk; blk++ )
    {
      powers[iy*oStride + blk*noR + ix] = total_power[blk];
    }
  }
}

#endif

#ifdef WITH_OPT_BLK2

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

template<typename T, int noBlk>
__global__ void ffdotPlnByBlk_ker2(float2* powers, float2* data, cuRespPln pln, int noHarms, int halfwidth, int zIdxTop, int rIdxLft, double zSZ, int noR, int noZ, int blkWidth, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw)
{
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  const int is = tx % pln.noRpnts ;		///<
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
      for( int i = 0; i < noHarms; i++ )           // Loop over harmonics
      {
	int 	hrm 	= i+1;

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
	  if ( hw.val[i] )
	  {
	    halfW	= hw.val[i];
	  }
	  else
	  {
	    zh	= zIdx * hrm * pln.dZ;
	    halfW       = cu_z_resp_halfwidth_high<float>(zh); // NB this was (z*i+4) I'm not sure why?
	  }
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

	  int start = rhIdx - halfW - loR.val[i]  ;

	  // Calculate power for the harmonic
	  for( int blk = 0; blk < noBlk; blk++ )
	  {
	    float2 inp = data[iStride*i + start + blk*hrm + ix ];

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
	      atomicAdd(&(powers[iy*oStride*noHarms + (is + blk*pln.noRpnts)*noHarms + i].x), (float)(outVal.x));
	      atomicAdd(&(powers[iy*oStride*noHarms + (is + blk*pln.noRpnts)*noHarms + i].y), (float)(outVal.y));
	    }
	  }
	}
      }
    }
  }
}

#endif

#ifdef WITH_OPT_BLK3
template<typename T, int noBlk>
__global__ void ffdotPlnByBlk_ker3(float* powers, float2* fft, int noHarms, int harmWidth, double firstR, double firstZ, double zSZ, int noR, int noZ, int blkWidth, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw)
{
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  const int i  = tx / harmWidth;
  const int ix = tx % harmWidth;
  const int iy = ty;

  if ( ix < noR && iy < noZ)
  {
    double r            = firstR + ix*blkWidth/(double)(noR) ;
    double z            = firstZ - iy/(double)(noZ-1) * zSZ ;
    if (noZ == 1)
      z = 0;

    float2      ans[noBlk];
    int halfW;

    FOLD
    {
      //for( int i = 0; i < noHarms; i++ )           // Loop over harmonics
      {
	int hrm = i+1;

	double absz         = fabs(z*hrm);

	FOLD // Determine half width
	{
	  if ( hw.val[i] )
	  {
	    halfW	= hw.val[i];
	  }
	  else
	  {
	    halfW       = cu_z_resp_halfwidth_high<float>(z*hrm); // NB this was (z*hrm+4) I'm not sure why?
	  }
	}

	// Set complex values to 0 for this harmonic
	for( int blk = 0; blk < noBlk; blk++ )
	{
	  ans[blk].x = 0;
	  ans[blk].y = 0;
	}

	FOLD // Calculate complex value, using direct application of the convolution
	{
	  rz_convolution_cu<T, float2, float2, noBlk>(&fft[iStride*i], loR.val[i], iStride, r*hrm, z*hrm, halfW, ans, blkWidth*hrm);
	}
      }
    }

    // Write values back to
    for( int blk = 0; blk < noBlk; blk++ )
    {
      float power = POWERF(ans[blk]);
      atomicAdd(&(powers[iy*oStride + blk*noR + ix]), power);
    }
  }
}
#endif


#ifdef WITH_OPT_PLN1
template<typename T>
__global__ void ffdotPln_ker(float* powers, float2* data, int noHarms, int halfwidth, double firstR, double firstZ, double rSZ, double zSZ, int noR, int noZ, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if ( ix < noR && iy < noZ)
  {
    int halfW;
    double r            = firstR + ix/(double)(noR-1) * rSZ ;
    double z            = firstZ - iy/(double)(noZ-1) * zSZ ;
    if (noZ == 1)
      z = 0;

    T total_power  = 0;
    T real = 0;
    T imag = 0;

    for( int i = 0; i < noHarms; i++ )
    {
      int hrm = i+1;
      FOLD // Determine half width
      {
	if ( hw.val[i] )
	  halfW	= hw.val[i];
	else
	  halfW	= cu_z_resp_halfwidth_high<float>(z*hrm);
      }

      rz_convolution_cu<T, float2>(&data[iStride*i], loR.val[i], iStride, r*hrm, z*hrm, halfW, &real, &imag);

      total_power     += POWERCU(real, imag);
    }

    powers[iy*oStride + ix] = total_power;
  }
}
#endif

#ifdef WITH_OPT_PLN2
template<typename T>
__global__ void ffdotPln_ker2(float2* powers, float2* data, int noHarms, int halfwidth, double firstR, double firstZ, double rSZ, double zSZ, int noR, int noZ, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw)
{
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  int off = tx % halfwidth ;
  int hrm = tx / halfwidth ;

  const int ix = ty % noR;
  const int iy = ty / noR;

  if ( ix < noR && iy < noZ)
  {
    int halfW;
    double r            = firstR + ix/(double)(noR-1) * rSZ ;
    double z            = firstZ - iy/(double)(noZ-1) * zSZ ;
    if (noZ == 1)
      z = 0;

    T real = 0;
    T imag = 0;

//    FOLD
//    {
//      FOLD // Determine half width
//      {
//        if ( hw.val[hrm] )
//          halfW = hw.val[hrm];
//        else
//          halfW = cu_z_resp_halfwidth_high<float>(z*(hrm+1));
//      }
//
//      if (off < halfW*2 )
//      {
//        rz_single_mult_cu<T, float2>(&fft[iStride*hrm], loR.val[hrm], iStride, r*(hrm+1), z*(hrm+1), halfW, &real, &imag, off);
//
//        atomicAdd(&(powers[iy*oStride*noHarms + ix*noHarms+hrm].x), (float)(real));
//        atomicAdd(&(powers[iy*oStride*noHarms + ix*noHarms+hrm].y), (float)(imag));
//      }
//    }

    for( int i = 1; i <= noHarms; i++ )
    {
      hrm = i-1;

      FOLD // Determine half width
      {
	if ( hw.val[hrm] )
	  halfW = hw.val[hrm];
	else
	  halfW = cu_z_resp_halfwidth_high<float>(z*i);
      }

      if (off < halfW*2 )
      {
	rz_single_mult_cu<T, float2>(&data[iStride*hrm], loR.val[hrm], iStride, r*i, z*i, halfW, &real, &imag, off);

	atomicAdd(&(powers[iy*oStride*noHarms + ix*noHarms + hrm].x), (float)(real));
	atomicAdd(&(powers[iy*oStride*noHarms + ix*noHarms + hrm].y), (float)(imag));
      }
    }
  }
}
#endif

#ifdef WITH_OPT_PLN3
template<typename T>
__global__ void ffdotPln_ker3(float* powers, float2* fft, int noHarms, int harmWidth, double firstR, double firstZ, double rSZ, double zSZ, int noR, int noZ, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw)
{
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  const int i  = tx / harmWidth ;
  const int ix = tx % harmWidth ;
  const int iy = ty;

  if ( ix < noR && iy < noZ)
  {
    int halfW;
    double r            = firstR + ix/(double)(noR-1) * rSZ ;
    double z            = firstZ - iy/(double)(noZ-1) * zSZ ;
    if (noZ == 1)
      z = 0;

    T hrm_power  = 0;			///< The power of a single point for the harmonic
    T real = 0;
    T imag = 0;

    const int hrm = i+1;
    FOLD // Determine half width
    {
      if ( hw.val[i] )
	halfW	= hw.val[i];
      else
	halfW	= cu_z_resp_halfwidth_high<float>(z*hrm);
    }

    rz_convolution_cu<T, float2>(&fft[iStride*i], loR.val[i], iStride, r*hrm, z*hrm, halfW, &real, &imag);

    hrm_power     = POWERCU(real, imag);

    atomicAdd(&(powers[iy*oStride + ix]), hrm_power);
  }
}
#endif

#ifdef WITH_OPT_PLN4
#ifdef CBL

// This function is under development, for some strange reason synchthreads is not working
template<typename T, int noHarms>
__global__ void ffdotPlnSM_ker(float* powers, float2* data, int halfwidth, double firstR, double firstZ, double rSZ, double zSZ, int noR, int noZ, int iStride, int oStride, int smLen, optLocInt_t loR, optLocInt_t hw)
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
    z = 0;

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

void opt_genResponse(cuRespPln* pln, cudaStream_t stream)
{
#ifdef WITH_OPT_BLK2
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

int chKpn( cuOptCand* pln, fftInfo* fft )
{
  PROF // Profiling  .
  {
    NV_RANGE_PUSH("Harm INP");
  }

  confSpecsOpt*  conf	= pln->conf;

  double	maxZ	= (pln->centZ + pln->zSize/2.0);
  double	minZ	= (pln->centZ - pln->zSize/2.0);
  double	maxR	= (pln->centR + pln->rSize/2.0);
  double	minR	= (pln->centR - pln->rSize/2.0);

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering ffdotPln.");

  pln->halfWidth	= cu_z_resp_halfwidth_high<double>(MAX(fabs(maxZ*pln->noHarms), fabs(minZ*pln->noHarms)) + 4 );
  double rSpread	= ceil((maxR+OPT_INP_BUF)*pln->noHarms  + pln->halfWidth) - floor((minR-OPT_INP_BUF)*pln->noHarms - pln->halfWidth);
  int	inpStride	= getStride(rSpread, sizeof(cufftComplex), pln->gInf->alignment);

  int	datStart;		// The start index of the input data
  int	datEnd;			// The end   index of the input data
  int	off;			// Offset
  int	newInp		= 0;	// Flag whether new input is needed

  if ( pln->noHarms != pln->input->noHarms )
  {
    newInp = 1;
  }

  // Determine if new input is needed
  for( int h = 0; (h < pln->noHarms) /* && !newInp */ ; h++ )
  {
    datStart        = floor( minR*(h+1) - pln->halfWidth );
    datEnd          = ceil(  maxR*(h+1) + pln->halfWidth );

    if ( datStart > fft->noBins || datEnd <= fft->firstBin )
    {
      if ( h == 0 )
      {
	fprintf(stderr, "ERROR: Trying to optimise a candidate beyond scope of the FFT?");
	return 0;
      }
      pln->noHarms = h; // use previous harmonic
      break;
    }

    if ( datStart < pln->input->loR[h] )
    {
      newInp = 1;
    }
    else if ( pln->input->loR[h] + pln->input->stride < datEnd )
    {
      newInp = 1;
    }
  }

  // Initialise values to 0
  for( int h = 0; h < OPT_MAX_LOC_HARMS; h++)
  {
    pln->hw[h] = 0;
  }

  if ( newInp ) // Calculate normalisation factor  .
  {
    infoMSG(4,4,"New Input\n");

    pln->input->stride = inpStride;
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
	if      ( conf->flags & FLAG_OPT_NRM_LOCAVE   )
	{
	  pln->input->norm[i-1]  = get_localpower3d(fft->data, fft->noBins, (pln->centR-fft->firstBin)*i, pln->centZ*i, 0.0);
	}
	else if ( conf->flags & FLAG_OPT_NRM_MEDIAN1D )
	{
	  pln->input->norm[i-1]  = get_scaleFactorZ(fft->data, fft->noBins, (pln->centR-fft->firstBin)*i, pln->centZ*i, 0.0);
	}
	else if ( conf->flags & FLAG_OPT_NRM_MEDIAN2D )
	{
	  fprintf(stderr,"ERROR: 2D median normalisation has not been written yet.\n");
	  exit(EXIT_FAILURE);
	}
	else
	{
	  // No normalisation this is plausible but not recommended
	  pln->input->norm[i-1] = 1;
	}
	infoMSG(6,6,"Harm %2i Normalisation %6.4f\n", i, pln->input->norm[i-1]);
      }

      PROF // Profiling  .
      {
	NV_RANGE_POP(); // Calc Norm factor
      }
    }
  }

  if ( newInp ) // A blocking synchronisation to make sure we can write to host memory  .
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

    if ( newInp ) // Normalise input and Write data to host memory  .
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

  return newInp;
}

/**
 *
 * Note the configuration flags are used to set the optimiser flags
 *
 * @param pln	optimiser
 * @param fft	FFT data structure
 * @return
 */
int ffdotPln_prep( cuOptCand* pln, fftInfo* fft )
{
  confSpecsOpt*	conf		= pln->conf;
  cuRespPln* 	rpln 		= pln->responsePln;

  pln->blkCnt = 1;
  pln->blkWidth = 0;

  infoMSG(4,4,"Generating ff section, Centred on (%.6f, %.6f)\n", pln->centR, pln->centZ );

  FOLD // Determine optimisation kernels  .
  {
    if ( pln->rSize > 1.5 &&  (conf->flags & FLAG_OPT_BLK) ) // Use the block kernel  .
    {
      /*	NOTE:	Chris Laidler	22/06/2016
       *
       * The per harmonic blocked kernel is fastest in my testing
       */

      remOptFlag(pln, FLAG_OPT_KER_ALL);
      setOptFlag(pln, (conf->flags & FLAG_OPT_BLK) );

      // New method finer granularity
      if ( pln->flags & FLAG_OPT_BLK_EXP )
      {
	if ( !rpln )
	{
	  fprintf(stderr, "ERROR, optimising with NULL response plane, reverting to standard block method.\n");
	  remOptFlag(pln, FLAG_OPT_BLK);

#ifdef 	WITH_OPT_BLK3
	  setOptFlag(pln, FLAG_OPT_BLK_HRM );
#elif	defined(WITH_OPT_BLK1)
	  setOptFlag(pln, FLAG_OPT_BLK_NRM );
#endif
	}
	else
	{
	  // NOTE: I think this pre-calculated resposen value kernel has been removed?

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

      if ( pln->flags & ( FLAG_OPT_BLK_NRM | FLAG_OPT_BLK_HRM ) )
      {
	// TODO: Check noR on fermi cards, the increased registers may justify using larger blocks widths
	do
	{
	  pln->blkWidth++;
	  pln->blkDimX	= pln->blkWidth / ( pln->rSize / pln->noR );
	  //blkDimX	= MIN(ceil(blkDimX),32);                      // The max of 32 is not strictly necessary
	  pln->blkCnt	= ceil(pln->rSize / (float)pln->blkWidth );
	}
	while ( pln->blkCnt > 10 ); // TODO: Make this 10 a hash define

	pln->rSize	= pln->blkCnt*pln->blkWidth - pln->blkWidth/(double)pln->blkDimX;
	pln->noR	= pln->blkDimX * pln->blkCnt;
      }
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

      remOptFlag(pln, FLAG_OPT_KER_ALL);
      setOptFlag(pln, (conf->flags & FLAG_OPT_PTS) );

      if ( !(pln->flags&FLAG_OPT_PTS) )
      {
#ifdef 	WITH_OPT_PLN3
	setOptFlag(pln, FLAG_OPT_PTS_HRM );
#elif	defined(WITH_OPT_PLN1)
	setOptFlag(pln, FLAG_OPT_PTS_NRM );
#elif	defined(WITH_OPT_PLN2)
	setOptFlag(pln, FLAG_OPT_PTS_EXP );
#endif
      }
    }

    // All kernels use the same output stride
    pln->outStride    = pln->noR;
  }
}

int ffdotPln_input( cuOptCand* pln, fftInfo* fft )
{
  // Check input
  int newInp = chKpn( pln, fft );

  if ( newInp ) // Copy input data to the device  .
  {
    infoMSG(4,4,"1D async memory copy H2D");

    CUDA_SAFE_CALL(cudaMemcpyAsync(pln->input->d_inp, pln->input->h_inp, pln->input->stride*pln->noHarms*sizeof(fcomplexcu), cudaMemcpyHostToDevice, pln->stream), "Copying optimisation input to the device");
    CUDA_SAFE_CALL(cudaEventRecord(pln->inpCmp, pln->stream),"Recording event: inpCmp");
  }

  return newInp;
}

template<typename T>
int ffdotPln_ker( cuOptCand* pln, fftInfo* fft)
{
  confSpecsOpt*	conf		= pln->conf;
  cuRespPln* 	rpln 		= pln->responsePln;

  int		maxHW 		= 0;	// The maximum possible halfwidth of the elements being tested

  optLocInt_t	rOff;			// Row offset
  optLocInt_t	hw;			// The halfwidth for each harmonic
  optLocFloat_t	norm;			// Normalisation factor for each harmonic
  //size_t	pln->resSz		= 0;	//

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
    // One float per point
    pln->resSz = pln->outStride*pln->noZ*sizeof(float) ;
    if ( pln->flags & ( FLAG_OPT_BLK_EXP | FLAG_OPT_PTS_EXP ) )
      pln->resSz = pln->outStride*pln->noZ*pln->noHarms*sizeof(fcomplex);	// One point per harmonic

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

    if ( pln->flags &  FLAG_OPT_BLK )			// Use block kernel
    {
      infoMSG(4,4,"Block kernel [ No threads %i  Width %i no Blocks %i]", (int)pln->blkDimX, pln->blkWidth, pln->blkCnt);

      if      ( pln->flags &  FLAG_OPT_BLK_NRM )		// Use block kernel
      {
#ifdef WITH_OPT_BLK1

	infoMSG(5,5,"Block kernel 1 - Standard");

	// Thread blocks
	dimBlock.x = pln->blkDimX;
	dimBlock.y = 16;
	dimBlock.z = 1;

	// One block per harmonic, thus we can sort input powers in Shared memory
	dimGrid.x = 1;
	dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

	CUDA_SAFE_CALL(cudaEventRecord(pln->tInit1, pln->stream),"Recording event: tInit1");

	// Call the kernel to normalise and spread the input data
	switch (pln->blkCnt)
	{
	  case 2:
	    ffdotPlnByBlk_ker<T,2> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 3:
	    ffdotPlnByBlk_ker<T,3> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 4:
	    ffdotPlnByBlk_ker<T,4> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 5:
	    ffdotPlnByBlk_ker<T,5> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 6:
	    ffdotPlnByBlk_ker<T,6> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 7:
	    ffdotPlnByBlk_ker<T,7> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 8:
	    ffdotPlnByBlk_ker<T,8> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 9:
	    ffdotPlnByBlk_ker<T,9> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 10:
	    ffdotPlnByBlk_ker<T,10><<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  default:
	  {
	    fprintf(stderr, "ERROR: %s has not been templated for %i blocks.\n", __FUNCTION__, pln->blkCnt );
	    exit(EXIT_FAILURE);
	  }
	}

	CUDA_SAFE_CALL(cudaEventRecord(pln->tComp1, pln->stream),"Recording event: tComp1");
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_BLK1.\n");
	exit(EXIT_FAILURE);
#endif
      }
      else if ( pln->flags &  FLAG_OPT_BLK_EXP )
      {
#ifdef WITH_OPT_BLK2

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
	    ffdotPlnByBlk_ker2<T, 2> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 3:
	    ffdotPlnByBlk_ker2<T, 3> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 4:
	    ffdotPlnByBlk_ker2<T, 4> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 5:
	    ffdotPlnByBlk_ker2<T, 5> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 6:
	    ffdotPlnByBlk_ker2<T, 6> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 7:
	    ffdotPlnByBlk_ker2<T, 7> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 8:
	    ffdotPlnByBlk_ker2<T, 8> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 9:
	    ffdotPlnByBlk_ker2<T, 9> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 10:
	    ffdotPlnByBlk_ker2<T,10> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 11:
	    ffdotPlnByBlk_ker2<T,11> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 12:
	    ffdotPlnByBlk_ker2<T,12> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 13:
	    ffdotPlnByBlk_ker2<T,13> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 14:
	    ffdotPlnByBlk_ker2<T,14> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 15:
	    ffdotPlnByBlk_ker2<T,15> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 16:
	    ffdotPlnByBlk_ker2<T,16> <<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, *rpln, pln->noHarms, pln->halfWidth, pln->topZidx, pln->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  default:
	  {
	    fprintf(stderr, "ERROR: %s has not been templated for %i blocks.\n", __FUNCTION__, pln->blkCnt );
	    exit(EXIT_FAILURE);
	  }
	}

	CUDA_SAFE_CALL(cudaEventRecord(pln->tComp2, pln->stream),"Recording event: tComp1");
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_BLK2.\n");
	exit(EXIT_FAILURE);
#endif
      }
      else if ( pln->flags & FLAG_OPT_BLK_HRM )
      {
#ifdef WITH_OPT_BLK3
	infoMSG(5,5,"Block kernel 3 - Harms");

	dimBlock.x = 16;
	dimBlock.y = 16;

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
	  case 2:
	    ffdotPlnByBlk_ker3<T, 2> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 3:
	    ffdotPlnByBlk_ker3<T, 3> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 4:
	    ffdotPlnByBlk_ker3<T, 4> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 5:
	    ffdotPlnByBlk_ker3<T, 5> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 6:
	    ffdotPlnByBlk_ker3<T, 6> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 7:
	    ffdotPlnByBlk_ker3<T, 7> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 8:
	    ffdotPlnByBlk_ker3<T, 8> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 9:
	    ffdotPlnByBlk_ker3<T, 9> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  case 10:
	    ffdotPlnByBlk_ker3<T,10> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, pln->input->stride, pln->outStride, rOff, norm, hw);
	    break;
	  default:
	  {
	    fprintf(stderr, "ERROR: %s has not been templated for %i blocks.\n", __FUNCTION__, pln->blkCnt );
	    exit(EXIT_FAILURE);
	  }
	}

	CUDA_SAFE_CALL(cudaEventRecord(pln->tComp2, pln->stream),"Recording event: tComp1");
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_BLK3.\n");
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
#ifdef WITH_OPT_PLN4
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
	      ffdotPlnSM_ker<T,1 ><<<dimGrid, dimBlock, smSz*sizeof(float2), pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->input->stride, pln->outStride, smSz, rOff, hw);
	      break;
	    case 2:
	      ffdotPlnSM_ker<T,2 ><<<dimGrid, dimBlock, smSz*sizeof(float2), pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->input->stride, pln->outStride, smSz, rOff, hw);
	      break;
	    case 4:
	      ffdotPlnSM_ker<T,4 ><<<dimGrid, dimBlock, smSz*sizeof(float2), pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->input->stride, pln->outStride, smSz, rOff, hw);
	      break;
	    case 8:
	      ffdotPlnSM_ker<T,8 ><<<dimGrid, dimBlock, smSz*sizeof(float2), pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->input->stride, pln->outStride, smSz, rOff, hw);
	      break;
	    case 16:
	      ffdotPlnSM_ker<T,16><<<dimGrid, dimBlock, smSz*sizeof(float2), pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->input->stride, pln->outStride, smSz, rOff, hw);
	      break;
	  }
	  //ffdotPlnSM_ker<T><<<dimGrid, dimBlock, smSz*sizeof(float2)*pln->noHarms*1.2, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->input->stride, pln->outStride, smSz, rOff, hw);
	}
#endif
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_PLN4.\n");
	exit(EXIT_FAILURE);
#endif
      }
      else if ( pln->flags &  FLAG_OPT_PTS_NRM ) // Thread point  .
      {
#ifdef WITH_OPT_PLN1
	infoMSG(5,5,"Flat kernel 1 - Standard\n");

	CUDA_SAFE_CALL(cudaEventRecord(pln->tInit1, pln->stream),"Recording event: tInit1");

	// One block per harmonic, thus we can sort input powers in Shared memory
	dimGrid.x = ceil(pln->noR/(float)dimBlock.x);
	dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

	// Call the kernel create a section of the f-fdot plane
	ffdotPln_ker<T><<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->input->stride, pln->outStride, rOff, norm, hw);

	CUDA_SAFE_CALL(cudaEventRecord(pln->tComp1, pln->stream),"Recording event: tComp1");
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_PLN1.\n");
	exit(EXIT_FAILURE);
#endif
      }
      else if ( pln->flags &  FLAG_OPT_PTS_EXP ) // Thread response pos  .
      {
#ifdef WITH_OPT_PLN2
	infoMSG(5,5,"Flat kernel 2 - Expanded\n");

	CUDA_SAFE_CALL(cudaEventRecord(pln->tInit2, pln->stream),"Recording event: tInit2");

	cudaMemsetAsync ( pln->d_out, 0, pln->resSz, pln->stream );
	//cudaMemset ( pln->d_out, 0, pln->resSz );
	CUDA_SAFE_CALL(cudaGetLastError(), "Zeroing the output memory");

	// One block per harmonic, thus we can sort input powers in Shared memory
	//dimGrid.x = ceil(maxHW*pln->noHarms/(float)dimBlock.x);
	dimGrid.x = ceil(maxHW/(float)dimBlock.x);
	dimGrid.y = ceil(pln->noZ*pln->noR/(float)dimBlock.y);

	// Call the kernel create a section of the f-fdot plane
	ffdotPln_ker2<T><<<dimGrid, dimBlock, 0, pln->stream >>>((float2*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, maxHW, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->input->stride, pln->outStride, rOff, norm, hw);

	CUDA_SAFE_CALL(cudaEventRecord(pln->tComp2, pln->stream),"Recording event: tComp2");
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_PLN2.\n");
	exit(EXIT_FAILURE);
#endif
      }
      else if ( pln->flags &  FLAG_OPT_PTS_HRM ) // Thread point of harmonic  .
      {
#ifdef WITH_OPT_PLN3
	infoMSG(5,5,"Flat kernel 3 - Harmonics\n");

	int noX = ceil(pln->noR / (float)dimBlock.x);
	int harmWidth = noX*dimBlock.x;

	CUDA_SAFE_CALL(cudaEventRecord(pln->tInit3, pln->stream),"Recording event: tInit3");

	cudaMemsetAsync ( pln->d_out, 0, pln->resSz, pln->stream );
	CUDA_SAFE_CALL(cudaGetLastError(), "Zeroing the output memory");

	// One block per harmonic, thus we can sort input powers in Shared memory
	//dimGrid.x = ceil(maxHW*pln->noHarms/(float)dimBlock.x);
	dimGrid.x = noX * pln->noHarms ;
	dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

	// Call the kernel create a section of the f-fdot plane
	ffdotPln_ker3<T><<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->input->stride, pln->outStride, rOff, norm, hw);

	CUDA_SAFE_CALL(cudaEventRecord(pln->tComp3, pln->stream),"Recording event: tComp3");
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_PLN3.\n");
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

}

int ffdotPln_get( cuOptCand* pln, fftInfo* fft )
{
  confSpecsOpt*	conf		= pln->conf;
  cuRespPln* 	rpln 		= pln->responsePln;

  FOLD // Copy data back to host  .
  {
    infoMSG(4,4,"1D async memory copy D2H");

    CUDA_SAFE_CALL(cudaMemcpyAsync(pln->h_out, pln->d_out, pln->resSz, cudaMemcpyDeviceToHost, pln->stream), "Copying optimisation results back from the device.");
    CUDA_SAFE_CALL(cudaEventRecord(pln->outCmp, pln->stream),"Recording event: outCmp");
  }
}

int ffdotPln_process( cuOptCand* pln, fftInfo* fft )
{
  confSpecsOpt*	conf		= pln->conf;
  cuRespPln* 	rpln 		= pln->responsePln;

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

  FOLD // Calc Powers  .
  {
    if ( pln->flags & ( FLAG_OPT_BLK_EXP | FLAG_OPT_PTS_EXP ) )
    {
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
	}
      }

      PROF // Profiling  .
      {
	NV_RANGE_POP(); // Calc Powers
      }
    }
  }
}

template<typename T>
int ffdotPln( cuOptCand* pln, fftInfo* fft )
{
  ffdotPln_prep( pln,  fft );

  int newInp = ffdotPln_input( pln, fft );

  ffdotPln_ker<T>( pln, fft );

  ffdotPln_get( pln, fft );

  ffdotPln_process( pln, fft );

  return newInp;
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
	    canidate->z = 0;

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

    if ( ffdotPln<T>(pln, fft) ) // Create the section of ff plane  .
    {
      // New input was used so don't maintain the old max
      cand->power	= 0;
      newInput		= 1;
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
	float yy2 = ((float*)pln->h_out)[indy*pln->outStride+indx];
	if ( yy2 > cand->power )
	{
	  cand->power	= yy2;
	  cand->r	= pln->centR - pln->rSize/2.0 + indx/(double)(pln->noR-1) * (pln->rSize) ;
	  cand->z	= pln->centZ + pln->zSize/2.0 - indy/(double)(pln->noZ-1) * (pln->zSize) ;
	  if ( pln->noZ	== 1 )
	    cand->z = 0;
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
      infoMSG(4,4,"Write CVS\n");

      char tName[1024];
      sprintf(tName,"/home/chris/accel/Cand_%05i_Rep_%02i_Lv_%i_h%02i.csv", nn, plt, lv, cand->numharm );
      FILE *f2 = fopen(tName, "w");

      FOLD // Write CSV
      {

	PROF // Profiling  .
	{
	  NV_RANGE_PUSH("Write CVS");
	}

	// Add number of hmonics summed as the first line
	fprintf(f2,"%i",pln->noHarms);

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
	    z = 0;
	  fprintf(f2,"%.15f",z);
	  
	  // Print power
	  for (int indx = 0; indx < pln->noR ; indx++ )
	  {
	    float yy2 = ((float*)pln->h_out)[indy*pln->outStride+indx];
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
	infoMSG(4,4,"Image\n");

	PROF // Profiling  .
	{
	  NV_RANGE_PUSH("Image");
	}

	char cmd[1024];
	sprintf(cmd,"python ~/bin/bin/plt_ffd.py %s > /dev/null 2>&1", tName);
	system(cmd);

	PROF // Profiling  .
	{
	  NV_RANGE_POP(); // Image
	}
      }
    }
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

int prepInput(initCand* cand, cuOptCand* pln, double sz)
{
  fftInfo*      fft     = pln->cuSrch->fft;

  FOLD // Large points  .
  {
    pln->noHarms	= cand->numharm;
    pln->centR          = cand->r;
    pln->centZ          = cand->z;
    pln->rSize          = sz;
    pln->zSize          = sz*pln->conf->zScale;
  }

  // Check the input
  int newInp = chKpn( pln, fft );

  return newInp;
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
cuOptCand* initOptCand(cuSearch* sSrch, cuOptCand* oPln, int devLstId )
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

    int maxSz = 0;
    int maxWidth = 0;
    float zMax;

    FOLD // Determine the largest zMax  .
    {
      zMax	= MAX(sSrch->sSpec->zMax+50, sSrch->sSpec->zMax*2);
      zMax	= MAX(zMax, 60 * sSrch->noSrchHarms );
      zMax	= MAX(zMax, sSrch->sSpec->zMax * 34 + 50 );  		// TODO: This may be a bit high!
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
#ifdef WITH_OPT_BLK2
      MAXX(maxSz, maxWidth * conf->optResolution);
#endif
      oPln->maxNoR	= maxSz*1.15;					// The maximum number of r points we can handle
      oPln->maxNoZ 	= maxSz;					// The maximum number of z points we can handle
    }

    oPln->cuSrch	= sSrch;					// Set the pointer t the search specifications
    oPln->maxHalfWidth	= cu_z_resp_halfwidth<double>( zMax, HIGHACC );	// The halfwidth of the largest plane we think we may handle
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
      oPln->outSz	= (oPln->maxNoR * oPln->maxNoZ ) * sizeof(float);

#ifdef	WITH_OPT_PLN2
      maxHarm		= MAX(conf->optMinLocHarms, sSrch->noSrchHarms );
      oPln->outSz	= (oPln->maxNoR * maxHarm * oPln->maxNoZ ) * sizeof(cufftComplex);
#endif	// WITH_OPT_PLN2

#ifdef	WITH_OPT_PLN3
      oPln->input->size	= (maxWidth*10 + 2*oPln->maxHalfWidth) * sSrch->noSrchHarms * sizeof(cufftComplex)*2; // The noR is oversized to allow for moves of the plane without getting new input
#endif	// WITH_OPT_PLN3

      CUDA_SAFE_CALL(cudaMemGetInfo ( &freeMem, &totalMem ), "Getting Device memory information");
#ifdef MAX_GPU_MEM
      long  Diff = totalMem - MAX_GPU_MEM;
      if( Diff > 0 )
      {
	freeMem-= Diff;
	totalMem-=Diff;
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

	initOptCand(sSrch, &sSrch->oInf->opts[idx], dev );

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


template int ffdotPln<float >( cuOptCand* pln, fftInfo* fft );
template int ffdotPln<double>( cuOptCand* pln, fftInfo* fft );

template int ffdotPln_ker<float >( cuOptCand* pln, fftInfo* fft );
template int ffdotPln_ker<double>( cuOptCand* pln, fftInfo* fft );
