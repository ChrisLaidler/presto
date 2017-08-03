/** @file cuda_accel_PLN.cu
 *  @brief Utility functions and kernels to generate sections of ff plane
 *
 *  @author Chris Laidler
 *  @bug No known bugs.
 *
 *  Change Log
 *
 *  2017-04-24
 *    Create this file
 *    Moved some functions from optimisation to here
 *    Refactor a bunch of stuff to here
 *
 */


#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_math.h"
#include "candTree.h"
#include "cuda_response.h"
#include "cuda_accel_PLN.h"
#include "cuda_accel_utils.h"



#ifdef WITH_OPT_BLK_NRM

/** Plane generation, blocked, point per ff point
 *
 * @param pln
 * @param stream
 */
template<typename T, int noBlk>
__global__ void ffdotPlnByBlk_ker1(float* powers, float2* data, int noHarms, double firstR, double firstZ, double zSZ, double rSZ, int blkDimX, int noR, int noZ, int blkWidth, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw, uint flags)
{
  const int bx = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if ( bx < blkDimX && iy < noZ )
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
      for( int hamr = 1; hamr <= noHarms; hamr++ )		// Loop over harmonics
      {
	int	hIdx	= hamr-1;

	FOLD // Determine half width
	{
	  halfW = getHw<T>(z*hamr, hw.val[hIdx]);
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
		((float2*)powers)[iy*oStride + (ix)*noHarms + hIdx ] = ans[blk];
	      }
	      else
		((float*)powers)[iy*oStride + (ix)*noHarms + hIdx ] = POWERF(ans[blk]);
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

#ifdef WITH_OPT_BLK_RSP

/** Generate response values
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
__global__ void ffdotPlnByBlk_ker2(float2* powers, float2* data, cuRespPln pln, int noHarms, int zIdxTop, int rIdxLft, double zSZ, int noR, int noZ, int blkWidth, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw, uint flags)
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
	  halfW = getHw<T>(zIdx * hrm * pln.dZ, hw.val[hIdx]);
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
	      atomicAdd(&(powers[iy*oStride + (is + blk*pln.noRpnts)*noHarms + hIdx].x), (float)(outVal.x));
	      atomicAdd(&(powers[iy*oStride + (is + blk*pln.noRpnts)*noHarms + hIdx].y), (float)(outVal.y));
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
	halfW = getHw<T>(z*hrm, hw.val[hIdx]);
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
	      ((float2*)powers)[iy*oStride + ix*noHarms + hIdx ] = ans[blk];
	    }
	    else
	      powers[iy*oStride + ix*noHarms + hIdx ] = POWERF(ans[blk]);
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

#ifdef WITH_OPT_PTS_NRM

/** Plane generation, points, thread per ff point
 *
 * @param pln
 * @param stream
 */
template<typename T>
__global__ void ffdotPln_ker1(float* powers, float2* data, int noHarms, double firstR, double firstZ, double rSZ, double zSZ, int noR, int noZ, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw, uint flags)
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
	halfW = getHw<T>(z*hrm, hw.val[hIdx]);
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
	  ((float2*)powers)[iy*oStride + ix*noHarms + hIdx ] = val ;
	}
	else
	{
	  powers[iy*oStride + ix*noHarms + hIdx ] = POWERCU(real, imag);
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

#ifdef WITH_OPT_PTS_EXP

/** Plane generation, points, thread per convolution operation
 *
 * This is slow and should probably not be used
 *
 * @param pln
 * @param stream
 */
template<typename T>
__global__ void ffdotPln_ker2(float2* powers, float2* data, int noHarms, int halfwidth, double firstR, double firstZ, double rSZ, double zSZ, int noR, int noZ, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw, uint flags)
{
  const int 	tIdx	= threadIdx.y * blockDim.x + threadIdx.x;
  //const int 	wIdx	= tIdx / 32;
  //const int 	lIdx	= tIdx % 32;

  //const int	noStps	= halfwidth / blockDim.x * blockDim.y ;
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
	    halfW = getHw<T>(z*harm, hw.val[hIdx]);
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

	      atomicAdd(&(powers[iy*oStride + ix*noHarms + hIdx].x), (float)(real));
	      atomicAdd(&(powers[iy*oStride + ix*noHarms + hIdx].y), (float)(imag));
	    }
	  }
	}
      }
    }
  }
}
#endif

#ifdef WITH_OPT_PTS_HRM

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
      halfW = getHw<T>(z*hrm, hw.val[hIdx]);
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
	((float2*)powers)[iy*oStride + ix*noHarms + hIdx ] = val ;
      }
      else
      {
	powers[iy*oStride + ix*noHarms + hIdx ] = POWERCU(real, imag);
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

#ifdef WITH_OPT_PTS_SHR
#ifdef CBL

/** This function is under development
 *
 * @param pln
 * @param stream
 */
template<typename T, int noHarms>
__global__ void ffdotPlnSM_ker(float* powers, float2* data, double firstR, double firstZ, double rSZ, double zSZ, int noR, int noZ, int iStride, int oStride, int smLen, optLocInt_t loR, optLocInt_t hw, uint flags)
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


/** Get a nice text representation of the current plane kernel name
 *
 * @param pln     The plane to check options for
 * @param name    A text pointer to put the name into
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE getKerName(cuPlnGen* plnGen, char* name)
{
  ACC_ERR_CODE err = ACC_ERR_NONE;

  if      ( plnGen->flags & FLAG_OPT_BLK_NRM )
    sprintf(name,"%s","BLK_NRM" );
  else if ( plnGen->flags & FLAG_OPT_BLK_EXP )
    sprintf(name,"%s","BLK_EXP" );
  else if ( plnGen->flags & FLAG_OPT_BLK_HRM )
    sprintf(name,"%s","BLK_HRM" );
  else if ( plnGen->flags & FLAG_OPT_BLK_RSP )
      sprintf(name,"%s","BLK_RSP" );
  else if ( plnGen->flags & FLAG_OPT_PTS_NRM )
    sprintf(name,"%s","PTS_NRM" );
  else if ( plnGen->flags & FLAG_OPT_PTS_EXP )
    sprintf(name,"%s","PTS_EXP" );
  else if ( plnGen->flags & FLAG_OPT_PTS_HRM )
    sprintf(name,"%s","PTS_HRM" );
  else if ( plnGen->flags & FLAG_OPT_PTS_SHR)
    sprintf(name,"%s","PTS_SHR" );
  else if ( plnGen->flags & FLAG_OPT_PTS_RSP)
      sprintf(name,"%s","PTS_RSP" );
  else
    sprintf(name,"%s","UNKNOWN" );

  return err;
}

/** Set plane type settings from flags
 *
 * @param plnGen  The plane to read the flags from
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE setPlnGenTypeFromFlags( cuPlnGen* plnGen )
{
  ACC_ERR_CODE	 err		= ACC_ERR_NONE;

  if ( plnGen && plnGen->pln )
  {
    plnGen->pln->type = CU_NONE;

    if ( plnGen->flags & FLAG_CMPLX )
    {
      plnGen->pln->type += CU_CMPLXF;
    }
    else
    {
      plnGen->pln->type += CU_FLOAT;
    }

    if ( plnGen->flags & FLAG_HAMRS )
    {
      plnGen->pln->type += CU_STR_HARMONICS;
    }
    else
    {
      plnGen->pln->type += CU_STR_INCOHERENT_SUM;
    }
  }
  else
  {
    err += ACC_ERR_NULL;
  }

  return err;
}

/** Zero the device memory
 *
 * Not this assumes resSz has been set
 *
 * @param plnGen
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE zeroPln( cuPlnGen* plnGen )
{
  ACC_ERR_CODE	 err		= ACC_ERR_NONE;

  if ( plnGen && plnGen->pln && plnGen->pln->d_data )
  {
    cudaMemsetAsync ( plnGen->pln->d_data, 0, plnGen->pln->resSz, plnGen->stream );
    CUDA_SAFE_CALL(cudaGetLastError(), "Zeroing the output memory");
  }
  else
  {
    err += ACC_ERR_NULL;
  }

  return err;
}

/** Create the pre-calculated response plane
 *
 * @param pln
 * @param stream
 */
void opt_genResponse(cuRespPln* pln, cudaStream_t stream)
{
#ifdef WITH_OPT_BLK_RSP
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

/** Calculate the number of convolution operations needed to generate the plane
 *
 * This uses the current settings (size and half-width), thus assumes prep_Opt(...) has been called.
 *
 * This function returns values for each harmonic
 *
 * @param plnGen	The plane to read the flags from
 * @param cOps		A pointer to an array of minimum length of the number of harmonics, the results will be written to this array
 * @return		ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE ffdotPln_cOps_harms( cuPlnGen* plnGen, unsigned long long* cOps)
{
  ACC_ERR_CODE	 err		= ACC_ERR_NONE;
  cuRzHarmPlane* pln 		= plnGen->pln;

  if ( !cOps )
    return ACC_ERR_NULL;

  // Check input
  if ( plnGen->accu == 0 )
  {
    err += ACC_ERR_UNINIT;
  }
  else
  {
    for ( int hIdx = 0; hIdx < pln->noHarms; hIdx++ )
    {
      cOps[hIdx] = 0;
      for ( int z = 0; z < pln->noZ; z++ )
      {
	double zv	= pln->centZ + pln->zSize/2.0 - pln->zSize*(z+1)/(double)pln->noZ;
	int halfW;

	if ( plnGen->hw[hIdx] <= 0 )
	{
	  // In this case the hw value is the accuracy, so calculate halfwidth
	  halfW		= cu_z_resp_halfwidth<double>( zv*(hIdx+1), (presto_interp_acc)plnGen->hw[hIdx] );
	}
	else
	{
	  // halfwidth was previously calculated
	  halfW		= plnGen->hw[hIdx];
	}

	cOps[hIdx] += halfW * 2 * ( pln->noR ) ;
      }
    }
  }

  return err;
}

/** Calculate the number of convolution operations needed to generate the plane
 *
 * This uses the current settings (size and half-width), thus assumes prep_Opt(...) has been called.
 *
 * @param plnGen	The plane to read the flags from
 * @param cOps		A pointer to a value where the result will be written to
 * @return		ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE ffdotPln_cOps( cuPlnGen* plnGen, unsigned long long* cOps)
{
  ACC_ERR_CODE	 err		= ACC_ERR_NONE;
  cuRzHarmPlane* pln 		= plnGen->pln;

  if ( !cOps )
    return ACC_ERR_NULL;

  // Check input
  if ( plnGen->accu == 0 )
  {
    err += ACC_ERR_UNINIT;
  }
  else
  {
    unsigned long long cOps_hrm[32];
    *cOps = 0;

    err += ffdotPln_cOps_harms( plnGen, cOps_hrm);
    ERROR_MSG(err, "ERROR: Preparing plane.");

    for ( int hIdx = 0; hIdx < pln->noHarms; hIdx++ )
    {
      *cOps += cOps_hrm[hIdx];
    }
  }

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
ACC_ERR_CODE ffdotPln_ker( cuPlnGen* plnGen )
{
  ACC_ERR_CODE	 err	= ACC_ERR_NONE;
  confSpecsOpt*	 conf	= plnGen->conf;
  cuRespPln* 	 rpln 	= plnGen->responsePln;
  cuRzHarmPlane* pln 	= plnGen->pln;
  cuHarmInput*	 input	= plnGen->input;

  // Data structures to pass to the kernels
  optLocInt_t	rOff;			// Row offset
  optLocInt_t	hw;			// The halfwidth for each harmonic
  optLocFloat_t	norm;			// Normalisation factor for each harmonic

  infoMSG(4,4,"Calling CUDA kernel to generate plane.\n" );

  // Calculate bounds on potently newly scaled plane
  double maxZ		= (pln->centZ + pln->zSize/2.0);
  double minR		= (pln->centR - pln->rSize/2.0);

  if (!pln->zSize || !pln->zSize)
  {
    err += ACC_ERR_UNINIT;
  }

  // Initialise values to 0
  for( int h = 0; h < OPT_MAX_LOC_HARMS; h++)
  {
    rOff.val[h]		= input->loR[h];
    hw.val[h]		= plnGen->hw[h];
    norm.val[h]		= sqrt(input->norm[h]);			// Correctly normalised by the sqrt of the local power

    if ( h < plnGen->pln->noHarms && plnGen->hw[h] == 0 )
    {
      err += ACC_ERR_UNINIT;
    }
  }

  if ( ERROR_MSG(err, "ERROR: Generating f-fdot plane section.") )
  {
    return err;
  }

  err += setPlnGenTypeFromFlags(plnGen);

  FOLD // Call kernel  .
  {
    dim3 dimBlock, dimGrid;

    if ( conf->flags & FLAG_SYNCH )
    {
      CUDA_SAFE_CALL(cudaEventRecord(plnGen->compInit, plnGen->stream),"Recording event: compInit");
    }

    // These are the only flags specific to the kernel
    uint flags =  plnGen->flags & ( FLAG_HAMRS | FLAG_CMPLX );

    if ( plnGen->flags &  FLAG_OPT_BLK )			// Use block kernel
    {
      infoMSG(4,4,"Block kernel [ No threads %i  Width %i no Blocks %i]\n", (int)pln->blkDimX, pln->blkWidth, pln->blkCnt);

      if      ( plnGen->flags & FLAG_OPT_BLK_NRM )		// Use block kernel
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

	zeroPln(plnGen);

	infoMSG(6,6,"Blk %i x %i", dimBlock.x, dimBlock.y);
	infoMSG(6,6,"Grd %i x %i", dimGrid.x, dimGrid.y);

	// Call the kernel to normalise and spread the input data
	switch (pln->blkCnt)
	{
	  case 1:
	    ffdotPlnByBlk_ker1<T,1> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 2:
	    ffdotPlnByBlk_ker1<T,2> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 3:
	    ffdotPlnByBlk_ker1<T,3> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 4:
	    ffdotPlnByBlk_ker1<T,4> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 5:
	    ffdotPlnByBlk_ker1<T,5> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 6:
	    ffdotPlnByBlk_ker1<T,6> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 7:
	    ffdotPlnByBlk_ker1<T,7> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 8:
	    ffdotPlnByBlk_ker1<T,8> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 9:
	    ffdotPlnByBlk_ker1<T,9> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 10:
	    ffdotPlnByBlk_ker1<T,10><<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 11:
	    ffdotPlnByBlk_ker1<T,11><<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 12:
	    ffdotPlnByBlk_ker1<T,12><<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 13:
	    ffdotPlnByBlk_ker1<T,13><<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 14:
	    ffdotPlnByBlk_ker1<T,14><<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 15:
	    ffdotPlnByBlk_ker1<T,15><<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 16:
	    ffdotPlnByBlk_ker1<T,16><<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  default:
	  {
	    fprintf(stderr, "ERROR: %s has not been templated for %i blocks.\n", __FUNCTION__, pln->blkCnt );
	    exit(EXIT_FAILURE);
	  }
	}
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_BLK_NRM.\n");
	exit(EXIT_FAILURE);
#endif
      }
      else if ( plnGen->flags & FLAG_OPT_BLK_RSP )
      {
#ifdef WITH_OPT_BLK_RSP

	infoMSG(5,5,"Block kernel 2 - Expanded");

	dimBlock.x = 16;
	dimBlock.y = 16;

	zeroPln(plnGen);

	// One block per harmonic, thus we can sort input powers in Shared memory
	dimGrid.x = ceil(plnGen->maxHalfWidth*2*rpln->noRpnts/(float)dimBlock.x);
	dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

	// Call the kernel to normalise and spread the input data
	switch (pln->blkCnt)
	{
	  case 2:
	    ffdotPlnByBlk_ker2<T, 2> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float2*)pln->d_data, (float2*)input->d_inp, *rpln, pln->noHarms, plnGen->topZidx, plnGen->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 3:
	    ffdotPlnByBlk_ker2<T, 3> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float2*)pln->d_data, (float2*)input->d_inp, *rpln, pln->noHarms, plnGen->topZidx, plnGen->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 4:
	    ffdotPlnByBlk_ker2<T, 4> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float2*)pln->d_data, (float2*)input->d_inp, *rpln, pln->noHarms, plnGen->topZidx, plnGen->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 5:
	    ffdotPlnByBlk_ker2<T, 5> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float2*)pln->d_data, (float2*)input->d_inp, *rpln, pln->noHarms, plnGen->topZidx, plnGen->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 6:
	    ffdotPlnByBlk_ker2<T, 6> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float2*)pln->d_data, (float2*)input->d_inp, *rpln, pln->noHarms, plnGen->topZidx, plnGen->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 7:
	    ffdotPlnByBlk_ker2<T, 7> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float2*)pln->d_data, (float2*)input->d_inp, *rpln, pln->noHarms, plnGen->topZidx, plnGen->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 8:
	    ffdotPlnByBlk_ker2<T, 8> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float2*)pln->d_data, (float2*)input->d_inp, *rpln, pln->noHarms, plnGen->topZidx, plnGen->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 9:
	    ffdotPlnByBlk_ker2<T, 9> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float2*)pln->d_data, (float2*)input->d_inp, *rpln, pln->noHarms, plnGen->topZidx, plnGen->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 10:
	    ffdotPlnByBlk_ker2<T,10> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float2*)pln->d_data, (float2*)input->d_inp, *rpln, pln->noHarms, plnGen->topZidx, plnGen->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 11:
	    ffdotPlnByBlk_ker2<T,11> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float2*)pln->d_data, (float2*)input->d_inp, *rpln, pln->noHarms, plnGen->topZidx, plnGen->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 12:
	    ffdotPlnByBlk_ker2<T,12> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float2*)pln->d_data, (float2*)input->d_inp, *rpln, pln->noHarms, plnGen->topZidx, plnGen->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 13:
	    ffdotPlnByBlk_ker2<T,13> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float2*)pln->d_data, (float2*)input->d_inp, *rpln, pln->noHarms, plnGen->topZidx, plnGen->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 14:
	    ffdotPlnByBlk_ker2<T,14> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float2*)pln->d_data, (float2*)input->d_inp, *rpln, pln->noHarms, plnGen->topZidx, plnGen->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 15:
	    ffdotPlnByBlk_ker2<T,15> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float2*)pln->d_data, (float2*)input->d_inp, *rpln, pln->noHarms, plnGen->topZidx, plnGen->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 16:
	    ffdotPlnByBlk_ker2<T,16> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float2*)pln->d_data, (float2*)input->d_inp, *rpln, pln->noHarms, plnGen->topZidx, plnGen->lftIdx, pln->zSize, pln->blkDimX, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  default:
	  {
	    fprintf(stderr, "ERROR: %s has not been templated for %i blocks.\n", __FUNCTION__, pln->blkCnt );
	    exit(EXIT_FAILURE);
	  }
	}
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_BLK_RSP.\n");
	exit(EXIT_FAILURE);
#endif
      }
      else if ( plnGen->flags & FLAG_OPT_BLK_HRM )
      {
#ifdef WITH_OPT_BLK_HRM
	infoMSG(5,5,"Block kernel 3 - Harms");

	dimBlock.x = MIN(16, pln->blkDimX);
	dimBlock.y = MIN(16, pln->noZ);

	int noX = ceil(pln->blkDimX / (float)dimBlock.x);
	int harmWidth = noX*dimBlock.x;

	zeroPln(plnGen);

	// One block per harmonic, thus we can sort input powers in shared memory
	dimGrid.x = noX * pln->noHarms ;
	dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

	// Call the kernel to normalise and spread the input data
	switch (pln->blkCnt)
	{
	  case 1:
	    // NOTE: in this case I find the points kernel to be a bit faster (~5%)
	    ffdotPlnByBlk_ker3<T, 1> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 2:
	    ffdotPlnByBlk_ker3<T, 2> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 3:
	    ffdotPlnByBlk_ker3<T, 3> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 4:
	    ffdotPlnByBlk_ker3<T, 4> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 5:
	    ffdotPlnByBlk_ker3<T, 5> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 6:
	    ffdotPlnByBlk_ker3<T, 6> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 7:
	    ffdotPlnByBlk_ker3<T, 7> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 8:
	    ffdotPlnByBlk_ker3<T, 8> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 9:
	    ffdotPlnByBlk_ker3<T, 9> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 10:
	    ffdotPlnByBlk_ker3<T,10> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 11:
	    ffdotPlnByBlk_ker3<T,11> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 12:
	    ffdotPlnByBlk_ker3<T,12> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 13:
	    ffdotPlnByBlk_ker3<T,13> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 14:
	    ffdotPlnByBlk_ker3<T,14> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 15:
	    ffdotPlnByBlk_ker3<T,15> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  case 16:
	    ffdotPlnByBlk_ker3<T,16> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
	  default:
	  {
	    fprintf(stderr, "ERROR: %s has not been templated for %i blocks.\n", __FUNCTION__, pln->blkCnt );
	    exit(EXIT_FAILURE);
	  }
	}
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
    else						// Use normal kernel
    {
      infoMSG(4,4,"Grid kernel");

      infoMSG(4,4,"Grid kernel\n");

      dimBlock.x = 16;
      dimBlock.y = 16;
      dimBlock.z = 1;

      if      ( plnGen->flags &  FLAG_OPT_PTS_SHR ) // Shared mem  .
      {
#ifdef WITH_OPT_PTS_SHR
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
	      ffdotPlnSM_ker<T,1 ><<<dimGrid, dimBlock, smSz*sizeof(float2), plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, input->stride, pln->zStride, smSz, rOff, hw, flags);
	      break;
	    case 2:
	      ffdotPlnSM_ker<T,2 ><<<dimGrid, dimBlock, smSz*sizeof(float2), plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, input->stride, pln->zStride, smSz, rOff, hw, flags);
	      break;
	    case 4:
	      ffdotPlnSM_ker<T,4 ><<<dimGrid, dimBlock, smSz*sizeof(float2), plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, input->stride, pln->zStride, smSz, rOff, hw, flags);
	      break;
	    case 8:
	      ffdotPlnSM_ker<T,8 ><<<dimGrid, dimBlock, smSz*sizeof(float2), plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, input->stride, pln->zStride, smSz, rOff, hw, flags);
	      break;
	    case 16:
	      ffdotPlnSM_ker<T,16><<<dimGrid, dimBlock, smSz*sizeof(float2), plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, input->stride, pln->zStride, smSz, rOff, hw, flags);
	      break;
	  }
	}
#endif
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_PTS_SHR.\n");
	exit(EXIT_FAILURE);
#endif
      }
      else if ( plnGen->flags &  FLAG_OPT_PTS_NRM ) // Thread point  .
      {
#ifdef WITH_OPT_PTS_NRM
	infoMSG(5,5,"Flat kernel 1 - Standard\n");

	// One block per harmonic, thus we can sort input powers in Shared memory
	dimGrid.x = ceil(pln->noR/(float)dimBlock.x);
	dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

	// Call the kernel create a section of the f-fdot plane
	ffdotPln_ker1<T><<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, input->stride, pln->zStride, rOff, norm, hw, flags );

#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_PTS_NRM.\n");
	exit(EXIT_FAILURE);
#endif
      }
      else if ( plnGen->flags &  FLAG_OPT_PTS_EXP ) // Thread response pos  .
      {
#ifdef WITH_OPT_PTS_EXP
	infoMSG(5,5,"Flat kernel 2 - Expanded\n");

	if ( !(plnGen->flags&FLAG_CMPLX) )
	{
	  fprintf(stderr, "ERROR: Per point plane kernel can not sum powers.");
	  exit(EXIT_FAILURE);
	}

	zeroPln(plnGen);

	// One block per harmonic, thus we can sort input powers in Shared memory
	int respWidth = ceil(plnGen->maxHalfWidth*2/(float)dimBlock.x)*dimBlock.x;
	dimGrid.x = ceil(respWidth/((float)dimBlock.x*dimBlock.y) );
	dimGrid.y = ceil(pln->noZ /* * pln->noR */ ); // REM - super seeded
	dimGrid.y = pln->noR ;

	// Call the kernel create a section of the f-fdot plane
	ffdotPln_ker2<T><<<dimGrid, dimBlock, 0, plnGen->stream >>>((float2*)pln->d_data, (float2*)input->d_inp, pln->noHarms, respWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, input->stride, pln->zStride, rOff, norm, hw, flags);

#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_PTS_EXP.\n");
	exit(EXIT_FAILURE);
#endif
      }
      else if ( plnGen->flags &  FLAG_OPT_PTS_HRM ) // Thread point of harmonic  .
      {
#ifdef WITH_OPT_PTS_HRM
	infoMSG(5,5,"Flat kernel 3 - Harmonics\n");

	int noX = ceil(pln->noR / (float)dimBlock.x);
	int harmWidth = noX*dimBlock.x;

	zeroPln(plnGen);

	// One block per harmonic, thus we can sort input powers in Shared memory
	dimGrid.x = noX * pln->noHarms ;
	dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

	// Call the kernel create a section of the f-fdot plane
	ffdotPln_ker3<T><<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, input->stride, pln->zStride, rOff, norm, hw, flags);

#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_PTS_HRM.\n");
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
    {
      CUDA_SAFE_CALL(cudaEventRecord(plnGen->compCmp, plnGen->stream), "Recording event: compCmp");
    }
  }

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
ACC_ERR_CODE chkInput_ffdotPln( cuPlnGen* plnGen, fftInfo* fft, int* newInp)
{
  return chkInput_pln(plnGen->input, plnGen->pln, fft, newInp);
}

/** Copy relevant input from FFT to data structure normalising as needed
 *
 *  Note this contains a blocking synchronisation to make sure the pinned host memory is free
 *
 * @param plnGen  The plane to check
 * @param fft     The FFT data that will make up the input
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE prepInput_ffdotPln( cuPlnGen* plnGen, fftInfo* fft )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  double rSize = MAX(plnGen->pln->rSize, 20);
  double zSize = MAX(plnGen->pln->zSize, 20*plnGen->conf->zScale);

  err += loadHostHarmInput(plnGen->input, fft, plnGen->pln->centR, plnGen->pln->centZ, rSize, zSize, plnGen->pln->noHarms, plnGen->flags, &plnGen->inpCmp );
  ERROR_MSG(err, "ERROR: Loading input values.");

  return err;
}

/** Set the per harmonic half width using plane accuracy
 *
 * @param plnGen  The plane to check
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE setHalfWidth_ffdotPln( cuPlnGen* plnGen )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  // Check input
  if ( plnGen->accu == 0 )
  {
    err += ACC_ERR_UNINIT;
  }
  else
  {
    if ( plnGen->accu == LOWACC )
    {
      infoMSG(4,4,"Half width: standard accuracy");
    }
    else
    {
      infoMSG(4,4,"Half width: high accuracy");
    }

    // Initialise values to 0
    for( int hIdx = 0; hIdx < OPT_MAX_LOC_HARMS; hIdx++)
    {
      plnGen->hw[hIdx] = 0;
    }
    plnGen->maxHalfWidth = 0;

    double 	maxZ	= (plnGen->pln->centZ + plnGen->pln->zSize/2.0);
    double	minZ	= (plnGen->pln->centZ - plnGen->pln->zSize/2.0);
    double	lrgstZ	= MAX(fabs(maxZ), fabs(minZ));

    for( int hIdx = 0; hIdx < plnGen->pln->noHarms; hIdx++)
    {
      // TODO: Check OPT
      plnGen->hw[hIdx]	= cu_z_resp_halfwidth<double>(lrgstZ*(hIdx+1), plnGen->accu );
      MAXX(plnGen->maxHalfWidth, plnGen->hw[hIdx]);

      // Reset the halfwidth back to what its meant to be back
      if ( (plnGen->flags & FLAG_OPT_DYN_HW) || (plnGen->pln->zSize*(hIdx+1) >= 2) )
      {
	plnGen->hw[hIdx] = plnGen->accu;
      }
    }
  }

  return err;
}

/** Copy pre-prepared memory from pinned host memory to device memory
 *
 * This assumes that the input data has been written to the pinned host memory
 *
 * @param pln     The plane to check
 * @param fft     The FFT data that will make up the input
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE cpyInput_ffdotPln( cuPlnGen* plnGen, fftInfo* fft )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  infoMSG(4,4,"1D async memory copy H2D");

  CUDA_SAFE_CALL(cudaMemcpyAsync(plnGen->input->d_inp, plnGen->input->h_inp, plnGen->input->stride*plnGen->input->noHarms*sizeof(fcomplexcu), cudaMemcpyHostToDevice, plnGen->stream), "Copying optimisation input to the device");
  CUDA_SAFE_CALL(cudaEventRecord(plnGen->inpCmp, plnGen->stream),"Recording event: inpCmp");

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
ACC_ERR_CODE prep_Opt( cuPlnGen* plnGen, fftInfo* fft )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;
  confSpecsOpt*	conf		= plnGen->conf;

  infoMSG(4,4,"Prep optimiser.\n");

  FOLD // Determine optimisation kernels  .
  {
    // Initialise to single column
    plnGen->pln->blkCnt		= 1;
    plnGen->pln->blkWidth	= 1;
    plnGen->pln->blkDimX	= plnGen->pln->noR;

    if ( !(plnGen->flags & FLAG_OPT_KER_ALL) )
    {
      // Get the kernel from the options, this probably shouldn't happen
      err += setOptFlag(plnGen, (conf->flags & FLAG_OPT_KER_ALL) );
    }

    if ( plnGen->flags & FLAG_OPT_BLK ) // Use the block kernel  .
    {
      err += ffdotPln_calcCols( plnGen->pln, plnGen->flags, conf->blkDivisor);

#ifdef 	WITH_OPT_PTS_HRM
      if ( plnGen->pln->blkCnt == 1)
      {
	infoMSG(6,6,"Only one block, so going to use points kernel.\n");

	// In my testing a single block is faster with the points kernel
	err += remOptFlag(plnGen, FLAG_OPT_KER_ALL );
	err += setOptFlag(plnGen, FLAG_OPT_PTS_HRM );
      }
      else
#endif
    else
    {
      if ( !(plnGen->flags&FLAG_OPT_PTS) )
      {
	// No points kernel in generator flags, so get the "default" from configuration
	remOptFlag(plnGen, FLAG_OPT_KER_ALL);
	setOptFlag(plnGen, (conf->flags & FLAG_OPT_PTS) );
      }

      if ( !(plnGen->flags&FLAG_OPT_PTS) )
      {
#ifdef 	WITH_OPT_PTS_HRM
	setOptFlag(plnGen, FLAG_OPT_PTS_HRM );
#elif	defined(WITH_OPT_PTS_NRM)
	setOptFlag(plnGen, FLAG_OPT_PTS_NRM );
#elif	defined(WITH_OPT_PTS_EXP)
	setOptFlag(plnGen, FLAG_OPT_PTS_EXP );
#else
	fprintf(stderr,"ERROR: Not compiled with any per point block creation kernels.")
	err += ACC_ERR_COMPILED;
#endif

	char kerName[20];
	getKerName(plnGen, kerName);
	infoMSG(6,6,"Auto select points kernel %s.\n", kerName);
      }

      // Sanity check
      if ( plnGen->flags & FLAG_OPT_PTS_EXP )
      {
	if ( !(plnGen->flags&FLAG_CMPLX) || !(plnGen->flags&FLAG_HAMRS) )
	{
	  infoMSG(7,7,"Bad settings for expanded points kernel switching to Complex harmonics.\n" );

	  // DBG Put message back after testing
	  //fprintf(stderr, "WARNING: Expanded kernel requires using complex values for each harmonic.\n");
	  plnGen->flags |= FLAG_CMPLX;
	  plnGen->flags |= FLAG_HAMRS;
	}
      }

      getKerName(plnGen, kerName);
      infoMSG(6,6,"Points Kernel %s\n", kerName );
    }

    if ( !(plnGen->flags & FLAG_HAMRS) && (plnGen->flags & FLAG_CMPLX) )
    {
      fprintf(stderr, "WARNING: Can't return sum of complex numbers, changing to incoherent sum of powers.\n");
      plnGen->flags &= ~(FLAG_CMPLX);
    }

    err += setPlnGenTypeFromFlags(plnGen);

    err += stridePln(plnGen->pln, plnGen->gInf);

    // Now snap the grid to the centre
    //err += snapPlane(opt->pln); // TODO: This is bad, need to snap to the candidate
  }

  err += setHalfWidth_ffdotPln( plnGen );

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
ACC_ERR_CODE ffdotPln_cpyResultsD2H( cuPlnGen* plnGen, fftInfo* fft )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  FOLD // Copy data back to host  .
  {
    infoMSG(4,4,"1D async memory copy D2H");

    CUDA_SAFE_CALL(cudaMemcpyAsync(plnGen->pln->h_data, plnGen->pln->d_data, plnGen->pln->resSz, cudaMemcpyDeviceToHost, plnGen->stream), "Copying optimisation results back from the device.");
    CUDA_SAFE_CALL(cudaEventRecord(plnGen->outCmp, plnGen->stream),"Recording event: outCmp");
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
ACC_ERR_CODE ffdotPln_ensurePln( cuPlnGen* plnGen, fftInfo* fft )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host  .
  {
    infoMSG(4,4,"Blocking synchronisation on %s", "outCmp" );

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("EventSynch");
    }

    CUDA_SAFE_CALL(cudaEventSynchronize(plnGen->outCmp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");

    PROF // Profiling  .
    {
      NV_RANGE_POP("EventSynch");
    }

  }

  return err;
}

/** Calculate the section of ffdot plane using the GPU and put the results in host memory
 *
 * This is the function to use if you want to create a section of ff plane.
 *
 * This assumes that plnGen has been initialise and the relevant flags set
 * If the FLAG_RES_CLOSE or FLAG_RES_FAST are set the size and resolution of the plane may be changed slightly
 *
 * @param plnGen
 * @param fft
 * @param newInp
 * @return		ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
template<typename T>
ACC_ERR_CODE ffdotPln( cuPlnGen* plnGen, fftInfo* fft, int* newInp )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  infoMSG(4,4,"Generate plane ff section, Centred on (%.6f, %.6f) with %2i harmonics.\n", plnGen->pln->centR, plnGen->pln->centZ, plnGen->pln->noHarms );

  err += prep_Opt( plnGen,  fft );
  if (ERROR_MSG(err, "ERROR: Preparing plane."))
    return err;

  err += input_plnGen( plnGen, fft, newInp );
  if (ERROR_MSG(err, "ERROR: Getting input for the plane."))
    return err;

  err += ffdotPln_ker<T>( plnGen );
  if (ERROR_MSG(err, "ERROR: Running the kernel."))
    return err;

  err += ffdotPln_cpyResultsD2H( plnGen, fft );
  if (ERROR_MSG(err, "ERROR: Copying the results."))
    return err;

  err += ffdotPln_ensurePln( plnGen, fft );
  if (ERROR_MSG(err, "ERROR: Waiting."))
    return err;

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
ACC_ERR_CODE input_plnGen( cuPlnGen* plnGen, fftInfo* fft, int* newInp )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  // Check input
  int newInp_l;
  err += chkInput_ffdotPln( plnGen, fft, &newInp_l );

  if ( newInp_l ) // Copy input data to the device  .
  {
    err += prepInput_ffdotPln( plnGen, fft );

    err += cpyInput_ffdotPln( plnGen, fft );
  }

  if ( newInp )
    *newInp = newInp_l;

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
ACC_ERR_CODE ffdotPln_calcCols( cuRzHarmPlane* pln, int64_t flags, int colDivisor)
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  pln->blkCnt	= 1;
  pln->blkWidth	= 1;
  pln->blkDimX	= pln->noR;

  infoMSG(4,4,"Calculate plane column sizes. \n");

  if (colDivisor < 1)
  {
    infoMSG(6,6,"ERROR: Invalid divisor %i.\n", colDivisor);
    colDivisor = 1;
    err += ACC_ERR_INVLD_CONFIG;
  }

  FOLD // Determine optimisation kernels  .
  {
    if ( pln->rSize <= 1.0 )
    {
      pln->blkWidth	= 1;
      pln->blkCnt	= 1;

      if ( flags & FLAG_RES_FAST  )
	pln->blkDimX	= ceil( pln->noR / (double)colDivisor ) * colDivisor ;
      else
	pln->blkDimX	= pln->noR;

      pln->noR		= pln->blkDimX;
    }
    else
    {
      infoMSG(6,6,"Orr  #R: %3i  Sz: %9.6f  Res: %9.6f \n", pln->noR, pln->rSize, pln->rSize/(double)(pln->noR-1) );

      if      ( flags & FLAG_RES_CLOSE )
      {
	// This method tries to create a block structure that is close to the orrigional
	// The size will always be same or larger than that specifyed
	// And the resolution will be the same of finer than that specifyed
	
	// TODO: Check noR on fermi cards, the increased registers may justify using larger blocks widths
	do
	{
	  pln->blkWidth++;
	  pln->blkDimX		= ceil( pln->blkWidth * (pln->noR-1) / pln->rSize );
	  MINN(pln->blkDimX, pln->noR );
	  pln->blkCnt		= ceil( ( pln->rSize + 1 / (double)pln->blkDimX ) / pln->blkWidth );
	  // Can't have blocks wider than 16 - Thread block limit
	}
	while ( pln->blkCnt > 16 ); // TODO: Make block count a hash define

	if ( pln->blkCnt == 1 )
	{
	  pln->blkDimX		= pln->noR;
	}
	else
	{
	  pln->noR		= ceil( pln->rSize / (double)(pln->blkWidth) * (pln->blkDimX) ) + 1 ;
	  pln->rSize		= (pln->noR-1)*(pln->blkWidth)/double(pln->blkDimX);
	}
      }
      else if ( flags & FLAG_RES_FAST  )
      {
	// This method attempts to choose the parameters so as to be computationally fats
	// This speed is obtained at the "cost" of the size and resolution of the plane section created.
	// Generally the resolution will be higher than the original
	// The final width may be slightly smaller (by one resolution)
	// The block widths are set to be nicely divisible numbers, this can make the kernel a bit faster

	// Get initial best values
	pln->blkWidth		= ceil(pln->rSize / 16.0 );				// Max column width in Fourier bins
	double rPerBlock	= pln->noR / ( pln->rSize / (double)pln->blkWidth );	// Calculate the number of threads per column
	pln->blkDimX		= ceil(rPerBlock/(double)colDivisor)*colDivisor;	// Make the column width divisible (this can speed up processing)
	pln->blkCnt		= ceil( ( pln->rSize ) / pln->blkWidth );		//

	// Check if we should increase column width
	if( rPerBlock < (double)colDivisor*0.80 )
	{
	  // NOTE: Could look for higher divisors ie 3/2
	  pln->blkCnt		= ceil(pln->noR/(double)colDivisor);
	  pln->blkDimX		= colDivisor;
	  pln->blkWidth		= floor(pln->rSize/(double)pln->blkCnt);
	}

	pln->noR		= ceil( pln->rSize / (double)(pln->blkWidth) * (pln->blkDimX) ) + 1; // May as well get close but above
	pln->noR		= ceil( pln->noR / (double)colDivisor ) * colDivisor ;	// Make the column width divisible (this can speed up processing)
	if ( pln->noR > pln->blkCnt * pln->blkDimX )
	  pln->noR		= pln->blkCnt * pln->blkDimX;				// This is the reduction that reduces the size of the final plane to one resolution point less than the "desired" width
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

      infoMSG(6,6,"New  #R: %3i  Sz: %9.6f  Res: %9.6f  - Col Width: %2i  -  No cols: %.2f  -  col DimX: %2i \n", pln->noR, pln->rSize, pln->rSize/(double)(pln->noR-1), pln->blkWidth, pln->noR / (double)pln->blkDimX, pln->blkDimX );
    }

    // All kernels use the same output stride - These values can be changed later to suite a specific GPU memory alignment and data type
    pln->zStride		= pln->noR;
  }

  infoMSG(5,5,"Size (%.6f x %.6f) Points (%i x %i) %i  Resolution: %.7f r  %.7f z.\n", pln->rSize,pln->zSize, pln->noR, pln->noZ, pln->noR*pln->noZ, pln->rSize/double(pln->noR-1), pln->zSize/double(pln->noZ-1) );

  return err;
}

/** Check if the plane, with current settings, requires new input
 *
 * This does not load the actual input
 * This check the input in the input data structure of the plane
 *
 * @param pln		The plane to check, current settings ( centZ, centR, zSize, rSize, etc.) used
 * @param fft		The FFT data that will make up the input
 * @param newInp	Set to 1 if new input is needed
 * @return		ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE chkInput_pln(cuHarmInput* input, cuRzHarmPlane* pln, fftInfo* fft, int* newInp)
{
  return  chkInput(input, pln->centR, pln->centZ, pln->rSize, pln->zSize, pln->noHarms, newInp);
}

/** Set the stride values of plane memory
 *
 * @param pln
 * @param elSize
 * @param gInf
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE stridePln(cuRzHarmPlane* pln, gpuInf* gInf)
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  infoMSG(6,6,"Aligning plane to device memory.\n" );

  size_t	zStride	= 0;
  size_t	hStride	= 0;
  size_t	elSz	= 0;

  if      ( pln->type == CU_CMPLXF )
  {
    infoMSG(7,7,"Output: complex\n" );
    elSz = sizeof(float2);
  }
  else if ( pln->type == CU_FLOAT  )
  {
    infoMSG(7,7,"Output: powers\n" );
    elSz = sizeof(float);
  }
  else
  {
    infoMSG(6,6,"ERROR: Plane type has not been initialised.\n" );
    err += ACC_ERR_UNINIT;
  }

  if      ( pln->type == CU_STR_HARMONICS )
  {
    infoMSG(7,7,"Output: harmonics\n" );
    hStride = pln->noHarms;
  }
  else if ( pln->type == CU_STR_INCOHERENT_SUM )
  {
    infoMSG(7,7,"Output: incoherent sum.\n" );
    hStride = 1;
  }
  else
  {
    infoMSG(6,6,"ERROR: Plane type has not been initialised.\n" );
    err += ACC_ERR_UNINIT;
  }
  zStride = getStride(pln->noR*hStride, elSz, gInf->alignment);

  if ( zStride * pln->noZ * elSz < pln->size )
  {
    pln->zStride	= zStride;
  }
  else if ( pln->noR * pln->noZ * hStride * elSz  < pln->size )
  {
    fprintf(stderr, "ERROR: Plane size exceeds allocated memory!\n");

    err += ACC_ERR_MEM;
    pln->zStride	= 0;
  }
  else
  {
    // Well we just can can't have nicely aligned memory
    infoMSG(6,6,"Aligning plane to device memory would overflow the memory so no alignment.\n" );

    pln->zStride	= pln->noR;
  }

  // Set the size of the used part of memory
  pln->resSz = pln->zStride*pln->noZ*elSz;

  infoMSG(7,7,"Output size %.2f MB.\n", pln->resSz*1e-6 );

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
ACC_ERR_CODE ffdotPln_plotPln( cuRzHarmPlane* pln, const char* dir, const char* name )
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
	int noStrHarms;

	if      ( pln->type == CU_STR_HARMONICS )
	  noStrHarms = pln->noHarms;
	else if ( pln->type == CU_STR_INCOHERENT_SUM )
	  noStrHarms = 1;
	else
	{
	  infoMSG(6,6,"ERROR: Plane type has not been initialised.\n" );
	  err += ACC_ERR_UNINIT;
	  break;
	}

	for ( int hIdx = 0; hIdx < noStrHarms; hIdx++)
	{
	  if      ( pln->type == CU_CMPLXF )
	    yy2 +=  POWERF(((float2*)pln->h_data)[indy*pln->zStride + indx*noStrHarms + hIdx]);
	  else if ( pln->type == CU_FLOAT )
	    yy2 +=  ((float*)pln->h_data)[indy*pln->zStride + indx*noStrHarms + hIdx];
	  else
	  {
	    infoMSG(6,6,"ERROR: Plane type has not been initialised.\n" );
	    err += ACC_ERR_DATA_TYPE;
	    break;
	  }
	}

	fprintf(f2,"\t%.20f",yy2);
      }
      fprintf(f2,"\n");
    }
    fclose(f2);

    PROF // Profiling  .
    {
      NV_RANGE_POP("Write CVS");
    }
  }

  if ( !err ) // Make image  .
  {
    infoMSG(5,5,"Image %s\n", tName);

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Image");
    }

    char cmd[1024];
    sprintf(cmd,"python $PRESTO/python/plt_ffd.py %s > /dev/null 2>&1", tName);
    int ret = system(cmd);
    if ( ret )
    {
      fprintf(stderr,"ERROR: Problem running potting python script.");
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP("Image");
    }
  }

  return err;
}

ACC_ERR_CODE addPlnToTree(candTree* tree, cuRzHarmPlane* pln)
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

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
	float yy2 = 0;
	int noStrHarms;
	if      ( pln->type == CU_STR_HARMONICS )
	  noStrHarms = pln->noHarms;
	else if ( pln->type == CU_STR_INCOHERENT_SUM )
	  noStrHarms = 1;
	else
	{
	  infoMSG(6,6,"ERROR: Plane type has not been initialised.\n" );
	  err += ACC_ERR_UNINIT;
	  break;
	}

	for ( int hIdx = 0; hIdx < noStrHarms; hIdx++)
	{
	  if      ( pln->type == CU_CMPLXF )
	    yy2 +=  POWERF(((float2*)pln->h_data)[indy*pln->zStride + indx*noStrHarms + hIdx]);
	  else if ( pln->type == CU_FLOAT )
	    yy2 +=  ((float*)pln->h_data)[indy*pln->zStride + indx*noStrHarms + hIdx];
	  else
	  {
	    infoMSG(6,6,"ERROR: Plane type has not been initialised.\n" );
	    err += ACC_ERR_DATA_TYPE;
	    break;
	  }
	}

	FOLD // Create candidate and add to tree
	{
	  initCand* canidate = new initCand;

	  canidate->numharm = pln->noHarms;
	  canidate->power   = yy2;
	  canidate->r       = pln->centR - pln->rSize/2.0 + indx/(double)(pln->noR-1) * (pln->rSize) ;
	  canidate->z       = pln->centZ + pln->zSize/2.0 - indy/(double)(pln->noZ-1) * (pln->zSize) ;
	  canidate->sig     = yy2;
	  if ( pln->noZ == 1 )
	    canidate->z     = pln->centZ;
	  if ( pln->noR == 1 )
	    canidate->r     = pln->centR;

	  ggr++;

	  tree->insert(canidate, 0.2 );
	}
      }
    }
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP("addPlnToTree");
  }

  return err;
}

/** Initialise a plane, allocating matched host and device memory for the plane
 *
 * @param memSize
 * @return		ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
cuRzHarmPlane* initPln( size_t memSize )
{
  size_t freeMem, totalMem;

  infoMSG(4,4,"Creating new harmonic plane\n");

  cuRzHarmPlane* pln	= (cuRzHarmPlane*)malloc(sizeof(cuRzHarmPlane));
  memset(pln, 0, sizeof(cuRzHarmPlane));

  CUDA_SAFE_CALL(cudaMemGetInfo ( &freeMem, &totalMem ), "Getting Device memory information");
#ifdef MAX_GPU_MEM
  long  Diff = totalMem - MAX_GPU_MEM;
  if( Diff > 0 )
  {
	freeMem  -= Diff;
	totalMem -= Diff;
  }
#endif

  if ( memSize > freeMem )
  {
    printf("Not enough GPU memory to create any more stacks.\n");
    return NULL;
  }
  else
  {
    infoMSG(6,6,"Memory size %.2f MB.\n", memSize*1e-6 );

    // Allocate device memory
    CUDA_SAFE_CALL(cudaMalloc(&pln->d_data, memSize), "Failed to allocate device memory for kernel stack.");

    // Allocate host memory
    CUDA_SAFE_CALL(cudaMallocHost(&pln->h_data, memSize), "Failed to allocate device memory for kernel stack.");

    pln->size = memSize;

    // Set default data type (complex values for all harmonics ie. most information possible)
    pln->type += CU_CMPLXF;
    pln->type += CU_STR_HARMONICS;
  }

  return pln;
}

/** Free all memory related to a cuRzHarmPlane
 *
 * @param pln	The pointer of the plane to free
 * @return	ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE freePln(cuRzHarmPlane* pln)
{
  infoMSG(3,3,"Free plane data structure.\n");

  if ( pln )
  {
    FOLD // Free pinned memory
    {
      infoMSG(4,4,"Free pinned memory\n");

      cudaFreeHostNull(pln->h_data);
    }

    FOLD // Free device memory
    {
      infoMSG(4,4,"Free device memory\n");

      // Using separate output so free both
      cudaFreeNull(pln->d_data);
    }

    freeNull(pln);
  }
  else
  {
    return ACC_ERR_NULL;
  }

  return ACC_ERR_NONE;
}

cuPlnGen* initPlnGen(int maxHarms, float zMax, confSpecsOpt* conf, gpuInf* gInf)
{
  infoMSG(3,3,"Initialise a GPU rz plane generator.\n");

  cuPlnGen* plnGen = (cuPlnGen*)malloc(sizeof(cuPlnGen));
  memset(plnGen, 0, sizeof(cuPlnGen));

  if ( conf == NULL )
  {
    infoMSG(4,4,"No configuration specified getting default configuration.\n");
    confSpecs* confAll = getConfig();
    conf = confAll->opt;
  }
  if ( gInf == NULL )
  {
    infoMSG(4,4,"No GPU specified.\n");
    gInf = getGPU(NULL);
  }
  if (!gInf)
  {
    infoMSG(4,4,"ERROR: invalid GPU.\n");
    return NULL;
  }

  plnGen->conf		= conf;					// Should this rather be a duplicate?
  plnGen->flags		= conf->flags;				// Individual flags allows separate configuration
  plnGen->gInf		= gInf;
  plnGen->accu		= HIGHACC;				// Default to high accuracy

  setDevice(plnGen->gInf->devid);

  FOLD // Create streams  .
  {
    infoMSG(5,6,"Create streams.\n");

    CUDA_SAFE_CALL(cudaStreamCreate(&plnGen->stream),"Creating stream for candidate optimisation.");
  }

  FOLD // Create events  .
  {
    if ( plnGen->flags & FLAG_PROF )
    {
      infoMSG(5,5,"Create Events.\n");

      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->inpInit),     "Creating input event inpInit." );
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->inpCmp),      "Creating input event inpCmp."  );
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->compInit),    "Creating input event compInit.");
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->compCmp),     "Creating input event compCmp." );
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->outInit),     "Creating input event outInit." );
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->outCmp),      "Creating input event outCmp."  );
    }
    else
    {
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->inpInit,	cudaEventDisableTiming),	"Creating input event inpInit." );
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->inpCmp,	cudaEventDisableTiming),	"Creating input event inpCmp."  );
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->compInit,	cudaEventDisableTiming),	"Creating input event compInit.");
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->compCmp,	cudaEventDisableTiming),	"Creating input event compCmp." );
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->outInit,	cudaEventDisableTiming),	"Creating input event outInit." );
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->outCmp,	cudaEventDisableTiming),	"Creating input event outCmp."  );
    }
  }

  FOLD // Allocate device memory  .
  {
    infoMSG(5,6,"Allocate device memory.\n");

    int	maxDim		= 1;					///< The max plane width in points
    int	maxWidth	= 1;					///< The max width (area) the plane can cover for all harmonics
    float	zMaxMax		= 1;					///< Max Z-Max this plane should be able to handle

    int	maxNoR		= 1;					///<
    int	maxNoZ		= 1;					///<

    // Number of harmonics to check, I think this could go up to 32!

    FOLD // Determine max plane size  .
    {
      for ( int i=0; i < MAX_NO_STAGES; i++ )
      {
	MAXX(maxWidth, (conf->optPlnSiz[i]) );
      }

      for ( int i=0; i < NO_OPT_LEVS; i++ )
      {
	MAXX(maxDim, conf->optPlnDim[i]);
      }

      //#ifdef WITH_OPT_BLK_RSP // TODO: Check if this needs to go back if WITH_OPT_BLK_RSP is used
      //      MAXX(maxSz, maxWidth * conf->optResolution);
      //#endif

      FOLD // Determine the largest zMaxMax  .
      {
	zMaxMax	= MAX(zMax+50, zMax*2);
	zMaxMax	= MAX(zMaxMax, (zMax+maxDim)*maxHarms);				// This should be enough
	zMaxMax	= MAX(zMaxMax, 60 * maxHarms );
	//zMaxMax	= MAX(zMaxMax, sSrch->sSpec->zMax * 34 + 50 );  	// TODO: This is 34th harmonic of the fundamental plane. This may be a bit high!
      }

      maxNoR		= maxDim*1.5;					// The maximum number of r points, in the generated plane. The extra is to cater for block kernels which can auto increase
      maxNoZ 		= maxDim;					// The maximum number of z points, in the generated plane
    }

    // Allocate input memory
    plnGen->input	= initHarmInput(maxWidth*10, zMaxMax, maxHarms, plnGen->gInf);

    FOLD // Create plane and set its settings  .
    {
      size_t plnSz	= (maxNoR * maxNoZ * maxHarms ) * sizeof(float2);	// This allows the possibility of returning complex value for the base plane
      plnGen->pln	= initPln( plnSz );
    }
  }

  return plnGen;
}

ACC_ERR_CODE freePlnGen(cuPlnGen* plnGen)
{
  ACC_ERR_CODE err	= ACC_ERR_NONE;

  err += freePln(plnGen->pln);

  err += freeHarmInput(plnGen->input);

  return err;
}

ACC_ERR_CODE snapPlane(cuRzHarmPlane* pln)
{
  return centerPlane(pln, pln->centR, pln->centZ, true );
}

ACC_ERR_CODE centerPlane(cuRzHarmPlane* pln, double r, double z, bool snap )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  if ( pln->rSize && pln->zSize && pln->noR && pln->noZ )
  {
    double rRes		= pln->rSize / (double)(pln->noR-1);
    double zRes		= pln->zSize / (double)(pln->noZ-1);

    if (pln->noR == 1 )
      rRes		= pln->rSize;
    if (pln->noZ == 1 )
      zRes		= pln->zSize;

    if ( snap && ( pln->noR % 2 )  )
    {
      // Odd
      pln->centR	= r;
    }
    else
    {
      // Even
      pln->centR	= r + rRes/2.0;
    }

    if ( snap && ( pln->noZ % 2 )  )
    {
      // Odd
      pln->centZ	= z;
    }
    else
    {
      // Even
      pln->centZ	= z - zRes/2.0;
    }
  }
  else
  {
    infoMSG(6,6,"ERROR: Plane location parameters have not been initialised.\n" );
    err += ACC_ERR_UNINIT;
  }

  return err;
}

ACC_ERR_CODE centerPlaneOnCand(cuRzHarmPlane* pln, initCand* cand, bool snap)
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  if ( pln && cand )
  {
    pln->noHarms	= cand->numharm ;
    err += centerPlane(pln, cand->r, cand->z, snap);
  }
  else
  {
    infoMSG(6,6,"ERROR: NULL pointer centring plane.\n" );
    err += ACC_ERR_NULL;
  }

  return err;
}

template ACC_ERR_CODE ffdotPln_ker<float >( cuPlnGen* plnGen );
template ACC_ERR_CODE ffdotPln_ker<double>( cuPlnGen* plnGen );

template ACC_ERR_CODE ffdotPln<float >( cuPlnGen* plnGen, fftInfo* fft, int* newInput );
template ACC_ERR_CODE ffdotPln<double>( cuPlnGen* plnGen, fftInfo* fft, int* newInput );
