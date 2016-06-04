#include <curand.h>
#include <math.h>             // log
#include <curand_kernel.h>

#include "cuda_math_ext.h"
#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"
#include "cuda_response.h"


#define OPT_INP_BUF   10

extern "C"
{
#define __float128 long double
#include "accel.h"
}

#ifdef CBL
template<typename T, int noHarms>
__global__ void ffdotPlnSM_ker(float* powers, float2* fft /*, int noHarms*/, int halfwidth, double firstR, double firstZ, double rSZ, double zSZ, int noR, int noZ, int iStride, int oStride, int smLen, int32 loR, int32 hw)
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

  T total_power  = 0;
  T real = (T)0;
  T imag = (T)0;

  T real_O = (T)0;
  T imag_O = (T)0;

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

	    sm[odd] = fft[(i-1)*iStride + o2 ];

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



      //halfW	= z_resp_halfwidth_cu_high<float>(z*i);




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

template<typename T>
__global__ void ffdotPln_ker(float* powers, float2* fft, int noHarms, int halfwidth, double firstR, double firstZ, double rSZ, double zSZ, int noR, int noZ, int iStride, int oStride, int32 loR, float32 norm, int32 hw)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if ( ix < noR && iy < noZ)
  {
    int halfW;
    double r            = firstR + ix/(double)(noR-1) * rSZ ;
    double z            = firstZ - iy/(double)(noZ-1) * zSZ ;

    T total_power  = 0;
    T real = 0;
    T imag = 0;

    for( int i = 1; i <= noHarms; i++ )
    {
      FOLD // Determine half width
      {
	if ( hw.val[i-1] )
	  halfW	= hw.val[i-1];
	else
	  halfW	= z_resp_halfwidth_cu_high<float>(z*i);
      }

      rz_convolution_cu<T, float2>(&fft[iStride*(i-1)], loR.val[i-1], iStride, r*i, z*i, halfW, &real, &imag);

      total_power     += POWERCU(real, imag);
    }

    powers[iy*oStride + ix] = total_power;
  }
}

template<typename T, int noBlk>
__global__ void ffdotPlnByBlk_ker(float* powers, float2* fft, int noHarms, int halfwidth, double firstR, double firstZ, double zSZ, int noR, int noZ, int blkWidth, int iStride, int oStride, int32 loR, float32 norm, int32 hw)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if ( ix < noR && iy < noZ)
  {
    double r            = firstR + ix*blkWidth/(double)(noR) ;
    double z            = firstZ - iy/(double)(noZ-1) * zSZ ;

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
	    halfW	= hw.val[i-1];
	  else
	    halfW	= z_resp_halfwidth_cu_high<float>(z*i+4);
	}

	// Set complex values to 0 for this harmonic
	for( int blk = 0; blk < noBlk; blk++ )
	{
	  ans[blk].x = 0;
	  ans[blk].y = 0;
	}

	FOLD // Calculate complex value, using direct application of the convolution
	{
	  rz_convolution_cu<T, float2, float2, noBlk>(&fft[iStride*(i-1)], loR.val[i-1], iStride, r*i, z*i, halfW, ans, blkWidth*i);
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

template<typename T>
int ffdotPln( cuOptCand* pln, fftInfo* fft )
{
  searchSpecs*  sSpec   = pln->cuSrch->sSpec;

  bool    blkKer    = 0;  // Weather to use the block kernel
  int     noBlk     = 1;  // The number of blocks each thread will cover
  int     blkWidth  = 0;  // The width of a block
  double  noR       = 0;  // The number of cuda threads in the x dimension of a the blocked kernel

  if ( pln->rSize > 1.5 ) // Use the block kernel  .
  {
    noR             = pln->noR / pln->rSize ;
    double rSize    = ceil(pln->rSize);

    // TODO: Check noR on fermi cards, the increased registers may justify using larger blocks widths

    do
    {
      blkWidth++;
      noR             = blkWidth / ( pln->rSize / pln->noR );
      noR             = MIN(ceil(noR),32);                      // The max of 32 is not strictly necessary
      noBlk           = ceil(pln->rSize / (float)blkWidth );
    }
    while ( noBlk > 10 );

    pln->rSize      = noBlk*blkWidth - blkWidth/noR;
    pln->noR        = noR * noBlk;
    blkKer          = 1;
  }

  double maxZ       = (pln->centZ + pln->zSize/2.0);
  double minZ       = (pln->centZ - pln->zSize/2.0);
  double maxR       = (pln->centR + pln->rSize/2.0);
  double minR       = (pln->centR - pln->rSize/2.0);

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering ffdotPln.");

  pln->halfWidth    = z_resp_halfwidth(MAX(fabs(maxZ*pln->noHarms), fabs(minZ*pln->noHarms)) + 4, HIGHACC);
  double rSpread    = ceil((maxR+OPT_INP_BUF)*pln->noHarms  + pln->halfWidth) - floor((minR-OPT_INP_BUF)*pln->noHarms - pln->halfWidth);
  int    inpStride  = getStrie(rSpread, sizeof(cufftComplex), pln->alignment);
  pln->outStride    = getStrie(pln->noR,  sizeof(float), pln->alignment);

  int     datStart;         // The start index of the input data
  int     datEnd;           // The end   index of the input data
  int32   rOff;             // Row offset
  int32   hw;               // The halfwidth for each harmonic
  float32 norm;             // Normalisation factor for each harmonic
  int     off;              // Offset
  int     newInp = 0;       // Flag whether new input is needed

  // Determine if new input is needed
  for( int h = 0; (h < pln->noHarms) /* && !newInp */ ; h++ )
  {
    datStart        = floor( minR*(h+1) - pln->halfWidth );
    datEnd          = ceil(  maxR*(h+1) + pln->halfWidth );

    if ( datStart > fft->nor || datEnd <= fft->idx )
    {
      if ( h == 0 )
      {
	fprintf(stderr, "ERROR: Trying to optimise a candidate beyond scope of the FFT?");
	return 0;
      }
      pln->noHarms = h; // use previous harmonic
      break;
    }

    if ( datStart < pln->loR[h] )
    {
      newInp = 1;
    }
    else if ( pln->loR[h] + pln->inpStride < datEnd )
    {
      newInp = 1;
    }
  }

  // Initialise values to 0
  for( int h = 0; h < 32; h++)
  {
    rOff.val[h] = 0;
    hw.val[h]   = 0;
  }

  if ( newInp ) // Calculate normalisation factor  .
  {
    infoMSG(3,3,"New Input\n");

    pln->inpStride = inpStride;

    if ( pln->inpStride*pln->noHarms*sizeof(cufftComplex) > pln->inpSz )
    {
      fprintf(stderr, "ERROR: In function %s, cuOptCand not created with large enough input buffer.", __FUNCTION__);
      exit(EXIT_FAILURE);
    }

    FOLD // Calculate normalisation factor  .
    {
      nvtxRangePush("Calc Norm factor");

      for ( int i = 1; i <= pln->noHarms; i++ )
      {
	if ( sSpec->flags & FLAG_OPT_LOCAVE )
	{
	  pln->norm[i-1]  = get_localpower3d(fft->fft, fft->nor, (pln->centR-fft->idx)*i, pln->centZ*i, 0.0);
	}
	else
	{
	  pln->norm[i-1]  = get_scaleFactorZ(fft->fft, fft->nor, (pln->centR-fft->idx)*i, pln->centZ*i, 0.0);
	}
      }

      nvtxRangePop();
    }
  }

  if ( newInp ) // A blocking synchronisation to make sure we can write to host memory  .
  {
    infoMSG(3,4,"pre synchronisation [blocking]\n");

    CUDA_SAFE_CALL(cudaEventSynchronize(pln->inpCmp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
  }

  // Calculate values for harmonics     and   normalise input and write data to host memory
  for( int h = 0; h < pln->noHarms; h++)
  {
    datStart		= floor( minR*(h+1) - pln->halfWidth );
    datEnd		= ceil(  maxR*(h+1) + pln->halfWidth );

    rOff.val[h]		= pln->loR[h];
    hw.val[h]		= z_resp_halfwidth(MAX(fabs(maxZ*(h+1)), fabs(minZ*(h+1))) + 4, HIGHACC);

    if ( hw.val[h] > pln->halfWidth )
    {
      fprintf(stderr, "ERROR: Harmonic half-width is greater than plain maximum.\n");
      hw.val[h] = pln->halfWidth;
    }

    if ( newInp ) // Normalise input and Write data to host memory  .
    {
      int startV = MIN( ((datStart + datEnd - pln->inpStride ) / 2.0), datStart ); //Start value if the data is centred

      rOff.val[h]     = startV;
      pln->loR[h]     = startV;
      double factor   = sqrt(pln->norm[h]);		// Correctly normalised by the sqrt of the local power
      norm.val[h]     = factor;

      for ( int i = 0; i < pln->inpStride; i++ ) // Normalise input  .
      {
	off = rOff.val[h] - fft->idx + i;

	if ( off >= 0 && off < fft->nor )
	{
	  pln->h_inp[h*pln->inpStride + i].r = fft->fft[off].r / factor ;
	  pln->h_inp[h*pln->inpStride + i].i = fft->fft[off].i / factor ;
	}
	else
	{
	  pln->h_inp[h*pln->inpStride + i].r = 0;
	  pln->h_inp[h*pln->inpStride + i].i = 0;
	}
      }
    }
  }

  if ( newInp ) // Copy input data to the device  .
  {
    infoMSG(3,4,"Copy input to device\n");

    CUDA_SAFE_CALL(cudaMemcpyAsync(pln->d_inp, pln->h_inp, pln->inpStride*pln->noHarms*sizeof(fcomplexcu), cudaMemcpyHostToDevice, pln->stream), "Copying optimisation input to the device");
    CUDA_SAFE_CALL(cudaEventRecord(pln->inpCmp, pln->stream),"Recording event: inpCmp");
  }

  FOLD // Call kernel  .
  {
    dim3 dimBlock, dimGrid;

    if ( sSpec->flags & FLAG_SYNCH )
      CUDA_SAFE_CALL(cudaEventRecord(pln->compInit, pln->stream),"Recording event: compInit");

    if ( blkKer )      	  // Use normal kernel
    {
      infoMSG(4,5,"Block kernel [ No threads %i  Width %i no Blocks %i]\n", (int)noR, blkWidth, noBlk);

      if ( (sSpec->flags & FLAG_OPT_DYN_HW) || (pln->zSize >= 2) )
      {
	for( int h = 0; h < pln->noHarms; h++)
	{
	  hw.val[h] 	= 0;
	}
      }

      // Thread blocks
      dimBlock.x = noR;
      dimBlock.y = 16;
      dimBlock.z = 1;

      // One block per harmonic, thus we can sort input powers in Shared memory
      dimGrid.x = 1;
      dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

      // Call the kernel to normalise and spread the input data
      switch (noBlk)
      {
	case 2:
	  ffdotPlnByBlk_ker<T,2> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, noR, pln->noZ, blkWidth, pln->inpStride, pln->outStride, rOff, norm, hw);
	  break;
	case 3:
	  ffdotPlnByBlk_ker<T,3> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, noR, pln->noZ, blkWidth, pln->inpStride, pln->outStride, rOff, norm, hw);
	  break;
	case 4:
	  ffdotPlnByBlk_ker<T,4> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, noR, pln->noZ, blkWidth, pln->inpStride, pln->outStride, rOff, norm, hw);
	  break;
	case 5:
	  ffdotPlnByBlk_ker<T,5> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, noR, pln->noZ, blkWidth, pln->inpStride, pln->outStride, rOff, norm, hw);
	  break;
	case 6:
	  ffdotPlnByBlk_ker<T,6> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, noR, pln->noZ, blkWidth, pln->inpStride, pln->outStride, rOff, norm, hw);
	  break;
	case 7:
	  ffdotPlnByBlk_ker<T,7> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, noR, pln->noZ, blkWidth, pln->inpStride, pln->outStride, rOff, norm, hw);
	  break;
	case 8:
	  ffdotPlnByBlk_ker<T,8> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, noR, pln->noZ, blkWidth, pln->inpStride, pln->outStride, rOff, norm, hw);
	  break;
	case 9:
	  ffdotPlnByBlk_ker<T,9> <<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, noR, pln->noZ, blkWidth, pln->inpStride, pln->outStride, rOff, norm, hw);
	  break;
	case 10:
	  ffdotPlnByBlk_ker<T,10><<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->zSize, noR, pln->noZ, blkWidth, pln->inpStride, pln->outStride, rOff, norm, hw);
	  break;
	default:
	{
	  fprintf(stderr, "ERROR: %s has not been templated for %i blocks.\n", __FUNCTION__, noBlk );
	  exit(EXIT_FAILURE);
	}
      }
    }
    else                  // Use block kernel
    {
      dimBlock.x = 16;
      dimBlock.y = 16;
      dimBlock.z = 1;

//#ifdef CBL
//      float smSz = 0 ;
//
//      //smSz = pln->inpStride ; // TMP test
//      //smSz = ( ceil(hw.val[pln->noHarms-1]*2 + pln->rSize*pln->noHarms) + 10) ;
//      for( int h = 0; h < pln->noHarms; h++)
//      {
//	smSz += ceil(hw.val[h]*2 + pln->rSize*(h+1) + 4 );
//      }
//
//      if ( smSz < 6144*0.9 ) // ~% of SM	10: 4915
//      {
//
//	infoMSG(3,5,"Flat kernel\n");
//
//	// One block per harmonic, thus we can sort input powers in Shared memory
//	dimGrid.x = ceil(pln->noR/(float)dimBlock.x);
//	dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);
//
//	int noTB = dimGrid.x * dimGrid.y ;
//
//	// Call the kernel to normalise and spread the input data
//	switch (pln->noHarms)
//	{
//	  case 1:
//	    ffdotPlnSM_ker<T,1><<<dimGrid, dimBlock, smSz*sizeof(float2), pln->stream >>>((float*)pln->d_out, (float2*)pln->d_inp, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->inpStride, pln->outStride, smSz, rOff, hw);
//	    break;
//	  case 2:
//	    ffdotPlnSM_ker<T,2><<<dimGrid, dimBlock, smSz*sizeof(float2), pln->stream >>>((float*)pln->d_out, (float2*)pln->d_inp, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->inpStride, pln->outStride, smSz, rOff, hw);
//	    break;
//	  case 4:
//	    ffdotPlnSM_ker<T,4><<<dimGrid, dimBlock, smSz*sizeof(float2), pln->stream >>>((float*)pln->d_out, (float2*)pln->d_inp, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->inpStride, pln->outStride, smSz, rOff, hw);
//	    break;
//	  case 8:
//	    ffdotPlnSM_ker<T,8><<<dimGrid, dimBlock, smSz*sizeof(float2), pln->stream >>>((float*)pln->d_out, (float2*)pln->d_inp, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->inpStride, pln->outStride, smSz, rOff, hw);
//	    break;
//	  case 16:
//	    ffdotPlnSM_ker<T,16><<<dimGrid, dimBlock, smSz*sizeof(float2), pln->stream >>>((float*)pln->d_out, (float2*)pln->d_inp, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->inpStride, pln->outStride, smSz, rOff, hw);
//	    break;
//	}
//	//ffdotPlnSM_ker<T><<<dimGrid, dimBlock, smSz*sizeof(float2)*pln->noHarms*1.2, pln->stream >>>((float*)pln->d_out, (float2*)pln->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->inpStride, pln->outStride, smSz, rOff, hw);
//      }
//      else
//#endif
      {
	infoMSG(3,5,"Flat kernel\n");

	if ( (sSpec->flags & FLAG_OPT_DYN_HW) || (pln->zSize >= 2) )
	{
	  for( int h = 0; h < pln->noHarms; h++)
	  {
	    hw.val[h] 	= 0;
	  }
	}

	// One block per harmonic, thus we can sort input powers in Shared memory
	dimGrid.x = ceil(pln->noR/(float)dimBlock.x);
	dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

	// Call the kernel create a section of the f-fdot plane
	ffdotPln_ker<T><<<dimGrid, dimBlock, 0, pln->stream >>>((float*)pln->d_out, (float2*)pln->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->inpStride, pln->outStride, rOff, norm, hw);
      }
    }

    CUDA_SAFE_CALL(cudaGetLastError(), "Calling the ffdot_ker kernel.");

    if ( sSpec->flags & FLAG_SYNCH )
      CUDA_SAFE_CALL(cudaEventRecord(pln->compCmp, pln->stream), "Recording event: compCmp");

  }

  FOLD // Copy data back to host  .
  {
    infoMSG(3,4,"Copy data back\n");

    // TMP
    CUDA_SAFE_CALL(cudaEventSynchronize(pln->compCmp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");


    CUDA_SAFE_CALL(cudaMemcpyAsync(pln->h_out, pln->d_out, pln->outStride*pln->noZ*sizeof(float), cudaMemcpyDeviceToHost, pln->stream), "Copying optimisation results back from the device.");
    CUDA_SAFE_CALL(cudaEventRecord(pln->outCmp, pln->stream),"Recording event: outCmp");
  }

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
  nvtxRangePush("addPlnToTree");

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

	  ggr++;

	  tree->insert(canidate, 0.2 );
	}
      }
    }
  }

  nvtxRangePop();

  return 0;
}

candTree* opt_cont(candTree* oTree, cuOptCand* pln, container* cont, fftInfo* fft, int nn)
{
  //  nvtxRangePush("opt_cont");
  //
  //  searchSpecs*  sSpec   = pln->cuSrch->sSpec;
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
  //    sz = sSpec->optPlnSiz[0];
  //  if ( canidate->numharm == 2  )
  //    sz = sSpec->optPlnSiz[1];
  //  if ( canidate->numharm == 4  )
  //    sz = sSpec->optPlnSiz[2];
  //  if ( canidate->numharm == 8  )
  //    sz = sSpec->optPlnSiz[3];
  //  if ( canidate->numharm == 16 )
  //    sz = sSpec->optPlnSiz[4];
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
  //        nvtxRangePop();
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
  //        nvtxRangePop();
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
  //        nvtxRangePop();
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
  //        nvtxRangePop();
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
  //        nvtxRangePop();
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
  //        nvtxRangePop();
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
  //  nvtxRangePop();
  //  return thisOpt;
  return NULL;
}

template<typename T>
void optInitCandPosPln(initCand* cand, cuOptCand* pln, int noP, double scale, int plt = -1, int nn = 0 )
{
  infoMSG(3,2,"Gen plain\n");

  fftInfo*	fft	= &pln->cuSrch->sSpec->fftInf;

  FOLD // Large points  .
  {
    pln->centR          = cand->r;
    pln->centZ          = cand->z;
    pln->noZ            = noP*2 + 1;
    pln->noR            = noP*2 + 1;
    pln->rSize          = scale;
    //pln->zSize          = scale*4.0;
    pln->zSize          = scale;

    if ( ffdotPln<T>(pln, fft) )
    {
      // New input was used so don't maintain the old max
      cand->power = 0;
    }

  }

  FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host  .
  {
    infoMSG(3,4,"pre synchronisation [blocking]\n");

    nvtxRangePush("EventSynch");
    CUDA_SAFE_CALL(cudaEventSynchronize(pln->outCmp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
    nvtxRangePop();
  }

  FOLD // Write CVS & plot output  .
  {
#ifdef CBL
    searchSpecs*  sSpec	= pln->cuSrch->sSpec;

    if ( sSpec->flags & FLAG_DPG_PLT_OPT ) // Write CVS & plot output  .
    {
      infoMSG(4,4,"Write CVS\n");

      nvtxRangePush("Write CVS");

      char tName[1024];
      sprintf(tName,"/home/chris/accel/Cand_%05i_Rep_%02i_h%02i.csv", nn, plt, cand->numharm );
      FILE *f2 = fopen(tName, "w");

      FOLD // Write CSV
      {
	fprintf(f2,"%i",pln->noHarms);

	for (int indx = 0; indx < pln->noR ; indx++ )
	{
	  double r = pln->centR - pln->rSize/2.0 + indx/(double)(pln->noR-1) * (pln->rSize) ;
	  fprintf(f2,"\t%.6f",r);
	}
	fprintf(f2,"\n");

	for (int indy = 0; indy < pln->noZ; indy++ )
	{
	  double z = pln->centZ + pln->zSize/2.0 - indy/(double)(pln->noZ-1) * (pln->zSize) ;

	  fprintf(f2,"%.6f",z);

	  for (int indx = 0; indx < pln->noR ; indx++ )
	  {
	    float yy2 = ((float*)pln->h_out)[indy*pln->outStride+indx];
	    fprintf(f2,"\t%.15f",yy2);
	  }
	  fprintf(f2,"\n");
	}
	fclose(f2);
      }

      FOLD // Make image  .
      {
	infoMSG(4,4,"Image\n");

	nvtxRangePush("Image");
	char cmd[1024];
	sprintf(cmd,"python ~/bin/bin/plt_ffd.py %s > /dev/null 2>&1", tName);
	system(cmd);
	nvtxRangePop();
      }

      nvtxRangePop();
    }
#endif
  }

  FOLD // Get new max  .
  {
    nvtxRangePush("Get Max");

    for (int indy = 0; indy < pln->noZ; indy++ )
    {
      for (int indx = 0; indx < pln->noR ; indx++ )
      {
	float yy2 = ((float*)pln->h_out)[indy*pln->outStride+indx];
	if ( yy2 > cand->power )
	{
	  cand->power   = yy2;
	  cand->r       = pln->centR - pln->rSize/2.0 + indx/(double)(pln->noR-1) * (pln->rSize) ;
	  cand->z       = pln->centZ + pln->zSize/2.0 - indy/(double)(pln->noZ-1) * (pln->zSize) ;
	}
      }
    }

    infoMSG(4,4,"Max Power %8.3f at (%.4f %.4f)\n", cand->power, cand->r, cand->z);

    nvtxRangePop();
  }
}

/** Optimise derivatives of a candidate this is usually run in a separate CPU thread  .
 *
 * This function is meant to be the entry of a separate thread
 *
 */
void* optCandDerivs(void* ptr)
{
  candSrch*	res	= (candSrch*)ptr;
  cuSearch*	srch	= res->cuSrch;

  int ii;
  struct timeval start, end;    // Timing variables

  accelcand*    cand  = res->cand;
  searchSpecs*  sSpec = srch->sSpec;
  fftInfo*      fft   = &sSpec->fftInf;

  if ( srch->sSpec->flags & FLAG_TIME ) // Timing  .
  {
    gettimeofday(&start, NULL);
  }

  int maxHarms  = MAX(cand->numharm, sSpec->optMinRepHarms) ;

  // Set up candidate
  cand->pows    = gen_dvect(maxHarms);
  cand->hirs    = gen_dvect(maxHarms);
  cand->hizs    = gen_dvect(maxHarms);
  cand->derivs  = (rderivs *)   malloc(sizeof(rderivs)  * maxHarms  );

  // Initialise values
  for( ii=0; ii < maxHarms; ii++ )
  {
    //r_offset[ii]  = 0;
    cand->hirs[ii]  = cand->r*(ii+1);
    cand->hizs[ii]  = cand->z*(ii+1);
  }

  FOLD // Update fundamental values to the optimised ones  .
  {
    float   	maxSig		= 0;
    int     	bestH		= 0;
    float   	bestP		= 0;
    double  	sig		= 0; // can be a float
    int     	numindep;
    float   	candHPower	= 0;
    int     	noStages	= 0;
    int 	kern_half_width;
    double 	locpow;
    double 	real;
    double 	imag;
    double 	power;

    cand->power	= 0;

    for( ii = 1; ii <= maxHarms; ii++ )
    {

      infoMSG(5,5,"Harm %i\n",ii );

      if ( sSpec->flags & FLAG_OPT_LOCAVE )
      {
	locpow = get_localpower3d(fft->fft, fft->nor, cand->r*ii, cand->z*ii, 0.0);
      }
      else
      {
	locpow = get_scaleFactorZ(fft->fft, fft->nor, cand->r*ii, cand->z*ii, 0.0);
      }
      //infoMSG(6,6,"locpow %.5f \n", locpow );

      locpow = 1.0 ; //RODO: Remove this!!!!!!!!!!

      if ( locpow )
      {
	kern_half_width   = z_resp_halfwidth(fabs(cand->z*ii) + 4.0, HIGHACC);

	rz_convolution_cu<double, float2>((float2*)fft->fft, fft->idx, fft->nor, cand->r*ii, cand->z*ii, kern_half_width, &real, &imag);

	power = POWERCU(real, imag) / locpow ;

	cand->pows[ii-1] = power;

	get_derivs3d(fft->fft, fft->nor, cand->r*ii, cand->z*ii, 0.0, locpow, &res->cand->derivs[ii-1] );

	cand->power	+= power;
	numindep	= (sSpec->fftInf.rhi - sSpec->fftInf.rlo ) * (sSpec->zMax+1) * (ACCEL_DZ / 6.95) / (ii) ;

	sig		= candidate_sigma_cu(cand->power, (ii), numindep );

	//infoMSG(6,6,"Power %7.3f  Sig: %6.3f  Sum: Power %7.3f  Sig: %6.3f\n", power, candidate_sigma_cu(power, 1, 1 ), cand->power, sig ); // TMP

	if ( sig > maxSig || ii == 1 )
	{
	  maxSig        = sig;
	  bestP         = cand->power;
	  bestH         = (ii);
	}

	if ( ii == cand->numharm )
	{
	  candHPower    = cand->power;

	  if ( !(srch->sSpec->flags & FLAG_OPT_BEST) )
	  {
	    break;
	  }
	}
      }
    }

    if ( bestP && (srch->sSpec->flags & FLAG_OPT_BEST) && ( maxSig > 0.001 ) )
    {
      cand->numharm	= bestH;
      cand->sigma	= maxSig;
      cand->power	= bestP;

      infoMSG(4,4,"Cand best val Sigma: %5.2f Power: %6.4f\n", maxSig, bestP);
    }
    else
    {
      cand->power	= candHPower;
      noStages		= log2((double)cand->numharm);
      numindep		= srch->numindep[noStages];
      cand->sigma	= candidate_sigma_cu(candHPower, cand->numharm, numindep);

      infoMSG(4,4,"Cand harm val Sigma: %5.2f Power: %6.4f\n", cand->sigma, cand->power);
    }
  }

  if ( srch->sSpec->flags & FLAG_TIME ) // Timing  .
  {
    pthread_mutex_lock(&res->cuSrch->threasdInfo->candAdd_mutex);
    gettimeofday(&end, NULL);
    float v1 =  ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec))*1e-3  ;
    res->cuSrch->pInf->batches->resultTime[0] += v1;
    pthread_mutex_unlock(&res->cuSrch->threasdInfo->candAdd_mutex);
  }

  // Decrease the count number of running threads
  sem_trywait(&srch->threasdInfo->running_threads);

  free(res);

  return (NULL);
}

/** Optimise derivatives of a candidate Using the CPU  .
 * This usually spawns a separate CPU thread to do the sigma calculations
 */
void processCandDerivs(accelcand* cand, cuSearch* srch, int candNo = -1)
{
  infoMSG(2,2,"Calc Cand Derivatives \n");

  candSrch*     thrdDat  = new candSrch;

  thrdDat->cand   = cand;
  thrdDat->cuSrch = srch;
  thrdDat->candNo = candNo;

  nvtxRangePush("Opt derivs");

  // Increase the count number of running threads
  sem_post(&srch->threasdInfo->running_threads);

  if ( !(srch->sSpec->flags & FLAG_SYNCH) && (srch->sSpec->flags & FLAG_THREAD) )  // Create thread  .
  {
    pthread_t thread;
    int  iret1 = pthread_create( &thread, NULL, optCandDerivs, (void*) thrdDat);

    if (iret1)
    {
      fprintf(stderr,"Error - pthread_create() return code: %d\n", iret1);
      exit(EXIT_FAILURE);
    }
  }
  else                              // Just call the function  .
  {
    optCandDerivs( (void*) thrdDat );
  }

  nvtxRangePop();

  infoMSG(2,2,"Done");
}

/** This is the main function called by external elements
 *
 * @param cand
 * @param srch
 * @param obs
 * @param nn
 * @param pln
 */
void optInitCandLocPlns(initCand* cand, cuOptCand* pln, int no )
{
  infoMSG(2,2,"Optimise candidate by plain\n");

  searchSpecs*  sSpec   = pln->cuSrch->sSpec;
  fftInfo*      fft     = &sSpec->fftInf;

  char Txt[1024];
  sprintf(Txt, "Opt Cand %03i", no);
  nvtxRangePush(Txt);

  // Number of harmonics to check, I think this could go up to 32!
  int maxHarms	= MAX(cand->numharm,sSpec->optMinLocHarms);

  // Setup GPU plane
  pln->centR	= cand->r ;
  pln->centZ	= cand->z ;
  pln->noHarms	= maxHarms ;
  for ( int i=1; i <= maxHarms; i++ )
  {
    if ( sSpec->flags & FLAG_OPT_LOCAVE )
    {
      pln->norm[i-1]  = get_localpower3d(fft->fft, fft->nor, (pln->centR-fft->idx)*i, pln->centZ*i, 0.0);
    }
    else
    {
      pln->norm[i-1]  = get_scaleFactorZ(fft->fft, fft->nor, (pln->centR-fft->idx)*i, pln->centZ*i, 0.0);
    }
  }

  FOLD // Get best candidate location using GPU planes  .
  {
    int rep       = 0;
    int lrep      = 0;
    int noP       = 30;
    float snoop   = 0.3;
    float sz;
    float v1, v2;

    const int mxRep = 10;

    if ( cand->numharm == 1  )
      sz = sSpec->optPlnSiz[0];
    if ( cand->numharm == 2  )
      sz = sSpec->optPlnSiz[1];
    if ( cand->numharm == 4  )
      sz = sSpec->optPlnSiz[2];
    if ( cand->numharm == 8  )
      sz = sSpec->optPlnSiz[3];
    if ( cand->numharm == 16 )
      sz = sSpec->optPlnSiz[4];

    pln->halfWidth 	= 0;

    for ( int idx = 0; idx < NO_OPT_LEVS; idx++ )
    {
      if ( sSpec->optPlnDim[idx] > 0 )
      {
	noP		= sSpec->optPlnDim[idx] ;
	lrep		= 0;
	cand->power	= 0;				// Set initial power to zero

	do
	{
	  pln->centR	= cand->r ;
	  pln->centZ	= cand->z ;
	  if ( idx == NO_OPT_LEVS-1 )
	  {
	    // Last if last plane is not 0, it will be done with double precision
	    optInitCandPosPln<double>(cand, pln, noP, sz,  rep++, no );
	  }
	  else
	  {
	    // Standard single precision
	    optInitCandPosPln<float>(cand, pln, noP, sz,  rep++, no );
	  }
	  v1 = fabs(( pln->centR - cand->r )/(pln->rSize/2.0));
	  v2 = fabs(( pln->centZ - cand->z )/(pln->zSize/2.0));

	  if ( ++lrep > mxRep )
	  {
	    break;
	  }
	}
	while ( v1 > snoop || v2 > snoop );
	sz /= sSpec->optPlnScale;
      }
    }
  }

  nvtxRangePop();
}

/** This is the main function called by external elements
 *
 * @param cand
 * @param pln
 * @param nn
 */
void opt_accelcand(accelcand* cand, cuOptCand* pln, int no)
{
  searchSpecs*  sSpec   = pln->cuSrch->sSpec;

  if ( sSpec->flags & FLAG_OPT_SWARM )
  {
    fprintf(stderr,"ERROR: partial swarm has been removed.\n");
    exit(EXIT_FAILURE);
  }
  else
  {
    initCand iCand;
    iCand.r 		= cand->r;
    iCand.z 		= cand->z;
    iCand.power		= cand->power;
    iCand.numharm 	= cand->numharm;

    optInitCandLocPlns(&iCand, pln, no);

    cand->r 		= iCand.r;
    cand->z 		= iCand.z;
    cand->power		= iCand.power;
    cand->numharm 	= iCand.numharm;
  }

  FOLD // Optimise derivatives  .
  {
    processCandDerivs(cand, pln->cuSrch, no);
  }

}

int optList(GSList *listptr, cuSearch* cuSrch)
{
  struct timeval start01, end01;

  nvtxRangePush("GPU Optimisation");
  gettimeofday(&start01, NULL);       // Profiling

  int numcands = g_slist_length(listptr);

  int ii	= 0;
  int comp	= 0;

#ifndef DEBUG   // Parallel if we are not in debug mode  .
  if ( cuSrch->sSpec->flags & FLAG_SYNCH )
  {
    omp_set_num_threads(1);
  }
  else
  {
    omp_set_num_threads(cuSrch->oInf->noOpts);
  }

#pragma omp parallel
#endif
  FOLD  	// Main GPU loop  .
  {
    accelcand *candGPUP;

    int tid         = omp_get_thread_num();
    int ti          = 0; // tread specific index


    cuOptCand* oPlnPln = &(cuSrch->oInf->opts[tid]);
    setDevice(oPlnPln->device) ;

    while (listptr)  // Main Loop  .
    {
#pragma omp critical

      FOLD  // Synchronous behaviour  .
      {
#ifndef  DEBUG
	if ( cuSrch->sSpec->flags & FLAG_SYNCH )
#endif
	{
	  tid 		= ii % cuSrch->oInf->noOpts ;
	  oPlnPln 	= &(cuSrch->oInf->opts[tid]);
	  setDevice(oPlnPln->device) ;
	}

	FOLD // Calculate candidate  .
	{
	  if ( listptr )
	  {
	    candGPUP  = (accelcand *) (listptr->data);
	    listptr   = listptr->next;
	    ii++;
	    ti = ii;
#ifdef CBL
	    FOLD // TMP: This can get removed
	    {
	      candGPUP->init_power    = candGPUP->power;
	      candGPUP->init_sigma    = candGPUP->sigma;
	      candGPUP->init_numharm  = candGPUP->numharm;
	      candGPUP->init_r        = candGPUP->r;
	      candGPUP->init_z        = candGPUP->z;
	    }
#endif
	  }
	  else
	  {
	    candGPUP = NULL;
	  }
	}
      }

      if ( candGPUP ) // Optimise  .
      {
	infoMSG(2,2,"\nOptimising initial candidate %i/%i, Power: %.3f  Sigma %.2f  Harm %i at (%.3f %.3f)\n", ti, numcands, candGPUP->power, candGPUP->sigma, candGPUP->numharm, candGPUP->r, candGPUP->z );

	//	if ( ti == 13 )
	opt_accelcand(candGPUP, oPlnPln, ti);

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

  nvtxRangePop();

  // Wait for CPU derivative threads to finish
  waitForThreads(&cuSrch->threasdInfo->running_threads, "Waiting for CPU threads to complete.", 200 );

  gettimeofday(&end01, NULL);
  cuSrch->timings[TIME_GPU_OPT] += ((end01.tv_sec - start01.tv_sec) * 1e6 + (end01.tv_usec - start01.tv_usec));

  return 0;
}
