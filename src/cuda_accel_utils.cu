/** @file cuda_accel_utils.cu
 *  @brief Utility functions for CUDA accelsearch
 *
 *  This contains the various utility functions for the CUDA accelsearch
 *  These include:
 *    Determining plane - widths and step size and accellen
 *    Generating kernel structures
 *    Generating plane structures
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
 *  [0.0.02] [2017-01-07 10:25]
 *    Fixed bug in determining optimal plane width - half plane using correct z-max
 *
 */

#include <cufft.h>
#include <algorithm>

#include <thrust/sort.h>
#include <thrust/device_vector.h>

extern "C"
{
#define __float128 long double
#include "accel.h"
}

#ifdef USEFFTW
#include <fftw3.h>
#endif

#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"
#include "cuda_accel_IN.h"
#include "cuda_cand_OPT.h"

#ifdef CBL
#include <unistd.h>
#include "log.h"
#endif

#define MAX_GPU_MEM	3400000000					///< This is a TMP 970 memory hack.  REALLY NVIDIA, YOU SUCK!!!

__device__ __constant__ int           HEIGHT_HARM[MAX_HARM_NO];		///< Plane  height  in stage order
__device__ __constant__ int           STRIDE_HARM[MAX_HARM_NO];		///< Plane  stride  in stage order
__device__ __constant__ int           WIDTH_HARM[MAX_HARM_NO];		///< Plane  strides   in family
__device__ __constant__ void*         KERNEL_HARM[MAX_HARM_NO];		///< Kernel pointer in stage order
__device__ __constant__ int           KERNEL_OFF_HARM[MAX_HARM_NO];	///< The offset of the first row of each plane in their respective kernels
__device__ __constant__ stackInfo     STACKS[64];
__device__ __constant__ int           STK_STRD[MAX_STACKS];		///< Stride of the stacks
__device__ __constant__ char          STK_INP[MAX_STACKS][4069];	///< input details


int    globalInt01    = 0;
int    globalInt02    = 0;
int    globalInt03    = 0;
int    globalInt04    = 0;
int    globalInt05    = 0;

float  globalFloat01  = 0;
float  globalFloat02  = 0;
float  globalFloat03  = 0;
float  globalFloat04  = 0;
float  globalFloat05  = 0;

int     useUnopt      = 0;
int     msgLevel      = 0;

double ratioARR[] = {
    3.0 / 2.0,
    5.0 / 2.0,
    2.0 / 3.0,
    4.0 / 3.0,
    5.0 / 3.0,
    3.0 / 4.0,
    5.0 / 4.0,
    2.0 / 5.0,
    3.0 / 5.0,
    4.0 / 5.0,
    5.0 / 6.0,
    2.0 / 7.0,
    3.0 / 7.0,
    4.0 / 7.0,
    3.0 / 8.0,
    5.0 / 8.0,
    2.0 / 9.0,
    3.0 / 10.0,
    2.0 / 11.0,
    3.0 / 11.0,
    2.0 / 13.0,
    3.0 / 13.0,
    2.0 / 15.0
};


///////////////////////// Function prototypes ////////////////////////////////////

__global__ void printfData(float* data, int nX, int nY, int stride, int sX = 0, int sY = 0)
{
  //printf("\n");
  for (int x = 0; x < nX; x++)
  {
    printf("---------");
  }
  printf("\n");
  for (int y = 0; y < nY; y++)
  {
    for (int x = 0; x < nX; x++)
    {
      printf("%8.4f ",data[ (y+sY)*stride + sX+ x ]);
    }
    printf("\n");
  }
  for (int x = 0; x < nX; x++)
  {
    printf("---------");
  }
  printf("\n");
}

void setActiveBatch(cuFFdotBatch* batch, int rIdx)
{
  batch->rActive = rIdx;
}

float half2float(const ushort h)
{
  unsigned int sign     = ((h >> 15) & 1);
  unsigned int exponent = ((h >> 10) & 0x1f);
  unsigned int mantissa = ((h & 0x3ff) << 13);

  if (exponent == 0x1f)     // NaN or Inf
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

/** Calculate an optimal accellen given a width  .
 *
 * @param width the width of the plane usually a power of two
 * @param zmax
 * @return
 * If width is not a power of two it will be rounded up to the nearest power of two
 */
uint optAccellen(float width, int zmax, presto_interp_acc accuracy, int noResPerBin)
{
  double halfwidth	= cu_z_resp_halfwidth<double>(zmax, accuracy); /// The halfwidth of the maximum zmax, to calculate step size
  double pow2		= pow(2 , round(log2(width)) );
  uint oAccelLen	= floor(pow2 - 2 - 2 * halfwidth * noResPerBin );

  return oAccelLen;
}

/** Calculate the step size from a width if the width is < 100 it is skate to be the closest power of two  .
 *
 * @param width
 * @param zmax
 * @return
 */
uint calcAccellen(float width, float zmax, presto_interp_acc accuracy, int noResPerBin)
{
  int accelLen;

  if ( width > 100 )
  {
    accelLen = width;
  }
  else
  {
    accelLen = optAccellen(width*1000.0, zmax, accuracy, noResPerBin) ;
  }
  return accelLen;
}

uint calcAccellen(float width, float zmax, int noHarms, presto_interp_acc accuracy, int noResPerBin, float zRes, bool hamrDevis)
{
  infoMSG(5,5,"Calculating step size\n");

  uint	accelLen, oAccelLen1, oAccelLen2;

  oAccelLen1  = calcAccellen(width, zmax, accuracy, noResPerBin);
  infoMSG(6,6,"Initial optimal step size %i for a fundamental plane of width %.0f with z-max %.1f \n", oAccelLen1, width, zmax);

  if ( width > 100 )				// The user specified the exact width they want to use for accellen  .
  {
    accelLen  = oAccelLen1;
    infoMSG(6,6,"User specified step size %i - using %i \n", width, oAccelLen1);
  }
  else						// Determine accellen by, examining the accellen at the second stack  .
  {
    if ( noHarms > 1 )				// Working with a family of planes
    {
      float halfZ	= cu_calc_required_z<double>(0.5, zmax, zRes);
      oAccelLen2	= calcAccellen(width*0.5, halfZ, accuracy, noResPerBin);
      accelLen		= MIN(oAccelLen2*2, oAccelLen1);

      infoMSG(6,6,"Second optimal step size %i from half plane step size of %i.\n", accelLen, oAccelLen2);
    }
    else
    {
      // Just a single plane
      accelLen		= oAccelLen1;
    }

    FOLD // Check  .
    {
      double ss        = cu_calc_fftlen<double>(1, zmax, accelLen, accuracy, noResPerBin, zRes) ;
      double l2        = log2( ss ) - 10 ;
      double fWidth    = pow(2, l2);

      if ( fWidth != width )
      {
	fprintf(stderr,"ERROR: Width calculation did not give the desired value.\n");
	exit(EXIT_FAILURE);
      }
    }
  }

  FOLD						// Ensure divisibility  .
  {
    float devisNo = 2;				// Divisible by 2, not sure why, its not noResPerBin its 2?

    if ( hamrDevis )				// Adjust to be divisible by number of harmonics  .
    {
      devisNo = noResPerBin*noHarms;
    }
    accelLen = floor( accelLen/devisNo ) * (devisNo);

    infoMSG(6,6,"Divisible %i.\n", accelLen);
  }

  return accelLen;
}

/** Allocate R value array  .
 *
 */
void createRvals(cuFFdotBatch* batch, rVals** rLev1, rVals**** rAraays )
{
  rVals**   rLev2;

  int oSet                = 0;


  (*rLev1)                = (rVals*)malloc(sizeof(rVals)*batch->noSteps*batch->noGenHarms*batch->noRArryas);
  memset((*rLev1), 0, sizeof(rVals)*batch->noSteps*batch->noGenHarms*batch->noRArryas);
  for (int i1 = 0 ; i1 < batch->noSteps*batch->noGenHarms*batch->noRArryas; i1++)
  {
    (*rLev1)[i1].step     = -1; // Invalid step (0 is a valid value!)
  }

  *rAraays                = (rVals***)malloc(batch->noRArryas*sizeof(rVals**));

  for (int rIdx = 0; rIdx < batch->noRArryas; rIdx++)
  {
    rLev2                 = (rVals**)malloc(sizeof(rVals*)*batch->noSteps);
    (*rAraays)[rIdx]      = rLev2;

    for (int step = 0; step < batch->noSteps; step++)
    {
      rLev2[step]         = &((*rLev1)[oSet]);
      oSet               += batch->noGenHarms;
    }
  }
}

void freeRvals(cuFFdotBatch* batch, rVals** rLev1, rVals**** rAraays )
{
  if (*rAraays)
  {
    for (int rIdx = 0; rIdx < batch->noRArryas; rIdx++)
    {
      rVals**   rLev2;

      rLev2 = (*rAraays)[rIdx];

      freeNull(rLev2);
    }

    freeNull(*rAraays);
  }

  freeNull(*rLev1);
}

void createFFTPlans(cuFFdotBatch* kernel)
{
  char msg[1024];

  PROF // Profiling  .
  {
    NV_RANGE_PUSH("FFT plans");
  }

  // Note creating the plans is the most expensive task in the GPU init, I tried doing it in parallel but it was slower
  for (int i = 0; i < kernel->noStacks; i++)
  {
    cuFfdotStack* cStack  = &kernel->stacks[i];

    PROF // Profiling  .
    {
      sprintf(msg,"Stack %i",i);
      NV_RANGE_PUSH(msg);
    }

    FOLD // Input FFT's  .
    {
      int n[]             = {cStack->width};

      int inembed[]       = {cStack->strideCmplx * sizeof(fcomplexcu)};         /// Storage dimensions of the input data in memory
      int istride         = 1;                                                  /// The distance between two successive input elements in the least significant (i.e., innermost) dimension
      int idist           = cStack->strideCmplx;                                /// The distance between the first element of two consecutive signals in a batch of the input data

      int onembed[]       = {cStack->strideCmplx * sizeof(fcomplexcu)};
      int ostride         = 1;
      int odist           = cStack->strideCmplx;

      FOLD // Create the input FFT plan  .
      {
	if ( kernel->flags & CU_INPT_FFT_CPU )
	{
	  PROF // Profiling  .
	  {
	    NV_RANGE_PUSH("FFTW");
	  }

	  cStack->inpPlanFFTW = fftwf_plan_many_dft(1, n, cStack->noInStack*kernel->noSteps, (fftwf_complex*)cStack->h_iData, n, istride, idist, (fftwf_complex*)cStack->h_iData, n, ostride, odist, -1, FFTW_ESTIMATE);

	  PROF // Profiling  .
	  {
	    NV_RANGE_POP(); // FFTW
	  }
	}
	else
	{
	  PROF // Profiling  .
	  {
	    NV_RANGE_PUSH("CUFFT Inp");
	  }

	  CUFFT_SAFE_CALL(cufftPlanMany(&cStack->inpPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, cStack->noInStack*kernel->noSteps), "Creating plan for input data of stack.");

	  PROF // Profiling  .
	  {
	    NV_RANGE_POP(); //CUFFT Inp
	  }
	}
      }
    }

    FOLD // inverse FFT's  .
    {
      if ( kernel->flags & FLAG_DOUBLE )
      {
	int n[]             = {cStack->width};

	int inembed[]       = {cStack->strideCmplx * sizeof(double2)};            /// Storage dimensions of the input data in memory
	int istride         = 1;                                                  /// The distance between two successive input elements in the least significant (i.e., innermost) dimension
	int idist           = cStack->strideCmplx;                                /// The distance between the first element of two consecutive signals in a batch of the input data

	int onembed[]       = {cStack->strideCmplx * sizeof(double2)};
	int ostride         = 1;
	int odist           = cStack->strideCmplx;

	FOLD // Create the stack iFFT plan  .
	{
	  PROF // Profiling  .
	  {
	    NV_RANGE_PUSH("CUFFT Pln");
	  }

	  CUFFT_SAFE_CALL(cufftPlanMany(&cStack->plnPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, cStack->height*kernel->noSteps), "Creating plan for complex data of stack.");

	  PROF // Profiling  .
	  {
	    NV_RANGE_POP(); // CUFFT Pln
	  }
	}
      }
      else
      {
	int n[]             = {cStack->width};

	int inembed[]       = {cStack->strideCmplx * sizeof(fcomplexcu)};         /// Storage dimensions of the input data in memory
	int istride         = 1;                                                  /// The distance between two successive input elements in the least significant (i.e., innermost) dimension
	int idist           = cStack->strideCmplx;                                /// The distance between the first element of two consecutive signals in a batch of the input data

	int onembed[]       = {cStack->strideCmplx * sizeof(fcomplexcu)};
	int ostride         = 1;
	int odist           = cStack->strideCmplx;

	FOLD // Create the stack iFFT plan  .
	{
	  PROF // Profiling  .
	  {
	    NV_RANGE_PUSH("CUFFT Pln");
	  }

	  CUFFT_SAFE_CALL(cufftPlanMany(&cStack->plnPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, cStack->height*kernel->noSteps), "Creating plan for complex data of stack.");

	  PROF // Profiling  .
	  {
	    NV_RANGE_POP(); // CUFFT Pln
	  }
	}

      }
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP(); // msg
    }

    CUDA_SAFE_CALL(cudaGetLastError(), "Creating FFT plans for the stacks.");
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP(); // FFT plans
  }
}

/** Initialise a kernel data structure and values on a given device  .
 *
 * First Initialise kernel data structure (this is just a batch)
 *
 * Next create kernel values
 * If master is NULL this is the first device so calculate the actual kernel values
 * If master != NULL copy the kernel values from the master on another device
 *
 * @param kernel
 * @param master
 * @param numharmstages
 * @param zmax
 * @param fftinf
 * @param device
 * @param noBatches
 * @param noSteps
 * @param width
 * @param powcut
 * @param numindep
 * @param flags
 * @param outType
 * @param outData
 * @return
 */
int initKernel(cuFFdotBatch* kernel, cuFFdotBatch* master, cuSearch*   cuSrch, int devID )
{
  std::cout.flush();

  size_t free, total;                           ///< GPU memory
  int noInStack[MAX_HARM_NO];

  noInStack[0]        = 0;
  size_t kerSize      = 0;                      ///< Total size (in bytes) of all the data
  size_t batchSize    = 0;                      ///< Total size (in bytes) of all the data need by a single family (ie one step) excluding FFT temporary
  size_t fffTotSize   = 0;                      ///< Total size (in bytes) of FFT temporary memory
  size_t planeSize    = 0;                      ///< Total size (in bytes) of memory required independently of batch(es)
  size_t familySz     = 0;                      ///< The size in bytes of memory required for one family including kernel data
  float plnElsSZ      = 0;                      ///< The size of an element of the in-mem ff plane (generally the size of float complex)
  float powElsSZ      = 0;                      ///< The size of an element of the powers plane
  float cmpElsSZ      = 0;                      ///< The size of an element of the kernel and complex plane

  gpuInf* gInf		= &cuSrch->gSpec->devInfo[devID];
  int noBatches		= cuSrch->gSpec->noDevBatches[devID];
  int noSteps		= cuSrch->gSpec->noDevSteps[devID];

  presto_interp_acc  accuracy = LOWACC;

  assert(sizeof(size_t) == 8);			// Check the compiler implementation of size_t is 64 bits

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering initKernel.");

  infoMSG(3,3,"%s device %i\n",__FUNCTION__, gInf->devid);

  PROF // Profiling  .
  {
    char msg[1024];
    sprintf(msg, "Dev %02i", gInf->devid );
    NV_RANGE_PUSH(msg);
  }

  FOLD // See if we can use the cuda device  .
  {
    infoMSG(4,4,"access device %i\n", gInf->devid);

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Get Device");
    }

    if ( gInf->devid >= getGPUCount() )
    {
      fprintf(stderr, "ERROR: There is no CUDA device %i.\n", gInf->devid);
      return (0);
    }
    int currentDevvice;
    CUDA_SAFE_CALL(cudaSetDevice(gInf->devid), "Failed to set device using cudaSetDevice");
    CUDA_SAFE_CALL(cudaGetDevice(&currentDevvice), "Failed to get device using cudaGetDevice");
    if (currentDevvice != gInf->devid)
    {
      fprintf(stderr, "ERROR: CUDA Device not set.\n");
      return (0);
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

  FOLD // See if this device could do a GPU in-mem search  .
  {
    if ( master == NULL ) // For the moment lets try this on only the first card!
    {
      infoMSG(4,4,"Checking if in-mem possible?\n");

      kerSize     = 0;
      batchSize   = 0;
      fffTotSize  = 0;
      planeSize   = 0;

      int    plnStride  = cuSrch->sSpec->ssStepSize;
      int    plnY       = ceil(cuSrch->sSpec->zMax / cuSrch->sSpec->zRes ) + 1 ; // This assumes we are splitting the inmem plane into a top and bottom section (FLAG_Z_SPLIT)

      if ( cuSrch->sSpec->flags & FLAG_POW_HALF )
      {
#if CUDA_VERSION >= 7050
	plnElsSZ = sizeof(half);
#else
	plnElsSZ = sizeof(float);
	fprintf(stderr, "WARNING: Half precision can only be used with CUDA 7.5 or later! Reverting to single precision!\n");
	cuSrch->sSpec->flags &= ~FLAG_POW_HALF;
#endif
      }
      else
      {
	plnElsSZ = sizeof(float);
      }

      if ( cuSrch->sSpec->flags & FLAG_KER_HIGH )
	accuracy = HIGHACC;

      // Calculate stride
      if ( plnStride <= 0 )
      {
	plnStride = 32768; // TODO: I need to check for a good default
      }

      if ( noSteps <= 0 )
      {
	noSteps = MAX_STEPS;
      }

      // Calculate "approximate" plane size
      size_t accelLen   = calcAccellen(cuSrch->sSpec->pWidth, cuSrch->sSpec->zMax, 1, accuracy, cuSrch->sSpec->noResPerBin, cuSrch->sSpec->zRes, false);
      double fftLen	= cu_calc_fftlen<double>(1, cuSrch->sSpec->zMax, accelLen, accuracy, cuSrch->sSpec->noResPerBin, cuSrch->sSpec->zRes);
      int halfWidth	= cu_z_resp_halfwidth<double>(cuSrch->sSpec->zMax, accuracy);
      int noStepsP	= ( cuSrch->sSpec->fftInf.rhi + 2 * halfWidth  - cuSrch->sSpec->fftInf.rlo / (double)cuSrch->noSrchHarms ) * cuSrch->sSpec->noResPerBin / (double)( accelLen ) ; // The number of planes to make
      int nss 		= noStepsP;
      noStepsP		= ceil(noStepsP/(float)noSteps)*noSteps;	// Round up to be divisible by steps
      planeSize		= (noStepsP * accelLen) * plnY * plnElsSZ ;
      infoMSG(6,6,"in-mem plane: %.2f GB - (%i X %i) X %i   noStepsP: %i  noSteps: %i  nss :%i \n", planeSize*1e-9, noStepsP, accelLen, plnY, noStepsP, noSteps, nss);

      // Kernel
      kerSize		+= fftLen * plnY * sizeof(cufftComplex);						// Kernel
      if ( (kernel->flags & FLAG_KER_DOUBFFT) && !(kernel->flags & FLAG_DOUBLE) )
	kerSize        += fftLen * plnY * sizeof(double)*2;							// Kernel

      // Calculate the  "approximate" size of a single 1 step batch
      batchSize		+= fftLen * sizeof(cufftComplex);							// Input
      batchSize		+= plnStride * cuSrch->noHarmStages * cuSrch->sSpec->ssSlices * sizeof(candPZs);	// Output
      batchSize		+= fftLen * plnY * sizeof(cufftComplex);						// Complex plain
      batchSize		+= fftLen * plnY * sizeof(float);							// Powers plain

      infoMSG(6,6,"inpDataSize: %.2f MB - plnDataSize: ~%.2f MB - pwrDataSize: ~%.2f MB - retDataSize: ~%.2f MB \n",
	  ( fftLen * sizeof(cufftComplex) )*1e-6,
	  ( fftLen * plnY * sizeof(cufftComplex) )*1e-6,
	  ( fftLen * plnY * sizeof(float) )*1e-6,
	  ( plnStride * cuSrch->noHarmStages * cuSrch->sSpec->ssSlices * sizeof(candPZs) )*1e-6 );


      fffTotSize        = fftLen * plnY * sizeof(cufftComplex);
      familySz          = kerSize + batchSize + fffTotSize;

      infoMSG(6,6,"Free: %.3f GB  - in-mem plane: %.2f GB - Kernel: ~%.2f MB - batch: ~%.2f MB - fft: ~%.2f MB - full batch: ~%.2f MB \n", free*1e-9, planeSize*1e-9, kerSize*1e-6, batchSize*1e-6, fffTotSize*1e-6, familySz*1e-6 );

      bool possIm = false;		//< In-mem is possible
      bool prefIm = false;		//< Weather automatic configuration prefers in-mem
      bool doIm   = false;		//< Whether to do an in-mem search

      if ( planeSize + familySz < free )
      {
	// We can do a in-mem search
	infoMSG(5,5,"In-mem is possible\n");

	possIm = 1;

	if ( cuSrch->noHarmStages > 2 )
	{
	  // It's probably better to do an in-mem search
	  prefIm = true;
	  infoMSG(5,5,"Prefer in-mem\n");
	}

	if ( !(cuSrch->sSpec->flags & FLAG_SS_ALL) || (cuSrch->sSpec->flags & FLAG_SS_INMEM) )
	{
	  printf("Device %i can do a in-mem GPU search.\n", gInf->devid);
	  printf("  There is %.2f GB free memory.\n  A split f-∂f plane requires ~%.2f GB and the workspace ~%.2f MB.\n", free*1e-9, planeSize*1e-9, familySz*1e-6 );
	}

	if ( (cuSrch->sSpec->flags & FLAG_SS_INMEM) || ( prefIm && !(cuSrch->sSpec->flags & FLAG_SS_ALL)) )
	{
	  doIm = true;
	}
      }
      else
      {
	// Nothing was selected so let the user know info
	printf("Device %i can not do a in-mem GPU search.\n", gInf->devid);
	printf("  There is %.2f GB free memory.\n  The entire f-∂f plane (split) requires %.2f GB and the workspace ~%.2f MB.\n\n", free*1e-9, planeSize*1e-9, familySz*1e-6 );
      }

      if ( doIm )
      {
	cuSrch->noGenHarms        = 1;				// Only one plane so generate the plane with only the fundamental plane of a batch

	if ( cuSrch->gSpec->noDevices > 1 )
	{
	  fprintf(stderr,"  Warning: Reverting to single device search.\n");
	  cuSrch->gSpec->noDevices = 1;
	}

	if ( (planeSize + familySz) < free * 0.5 )
	{
	  if ( cuSrch->sSpec->flags & FLAG_Z_SPLIT )
	    fprintf(stderr,"  WARNING: Opting to split the in-mem plane when you don't need to.\n");
	  else
	    printf("    No need to split the in-mem plane.\n");
	}
	else
	{
	  printf("    Have to split the in-mem plane.\n");
	  cuSrch->sSpec->flags |= FLAG_Z_SPLIT;
	}

	cuSrch->sSpec->flags |= FLAG_SS_INMEM ;	// TODO: Don't edit sSpec directly?
#if CUDA_VERSION >= 6050
	if ( !(cuSrch->sSpec->flags & FLAG_CUFFT_CB_POW) )
	  fprintf(stderr,"  Warning: Doing an in-mem search with no CUFFT callbacks, this is not ideal.\n"); // It should be on by default the user must have disabled it
#else
	fprintf(stderr,"  Warning: Doing an in-mem search with no CUFFT callbacks, this is not ideal. Try upgrading to CUDA 6.5 or later.\n");
	cuSrch->sSpec->flags &= ~FLAG_CUFFT_ALL;
#endif

#if CUDA_VERSION >= 7050
	if ( !(cuSrch->sSpec->flags & FLAG_POW_HALF) )
	  fprintf(stderr,"  Warning: You could be using half precision.\n"); // They should be on by default the user must have disabled them
#else
	fprintf(stderr,"  Warning: You could be using half precision. Try upgrading to CUDA 7.5 or later.\n");
#endif

	FOLD // Set types  .
	{
	  cuSrch->sSpec->retType &= ~CU_TYPE_ALLL;
	  cuSrch->sSpec->retType |= CU_POWERZ_S;

	  cuSrch->sSpec->retType &= ~CU_SRT_ALL;
	  cuSrch->sSpec->retType |= CU_STR_ARR;
	}
      }
      else
      {
	// No In-mem
	infoMSG(5,5,"Not doing in-mem\n");

	if ( cuSrch->sSpec->flags & FLAG_SS_INMEM  )
	{
	  // Warning
	  fprintf(stderr,"ERROR: Requested an in-memory GPU search, this is not possible.\n\tThere is %.2f GB of free memory.\n\tIn-mem (split plane) GPU search would require ~%.2f GB\n\n", free*1e-9, (planeSize + familySz)*1e-9 );

	  printf("Temporary exit search specs\n"); // TMP
	  exit(EXIT_FAILURE); // TMP
	}
	else if ( possIm & !prefIm )
	{
	  printf("  Opting to not do an in-mem search.\n");
	}
	else if ( prefIm )
	{
	  // "Should" do a IM search

	  if ( (cuSrch->sSpec->flags & FLAG_SS_ALL) )
	  {
	    fprintf(stderr,"WARNING: Opting to NOT do a in-mem search when you could!\n");
	  }
	}

	cuSrch->sSpec->flags &= ~FLAG_SS_INMEM ;
	cuSrch->sSpec->flags &= ~FLAG_Z_SPLIT;
      }

      // Sanity check
      if ( !(cuSrch->sSpec->flags & FLAG_SS_ALL) )
      {
	// Default to S&S 1.
	cuSrch->sSpec->flags |= FLAG_SS_10;
	cuSrch->sSpec->flags |= FLAG_RET_STAGES;
      }

      // Reset sizes as they will need to be set to the actual values
      kerSize     = 0;
      batchSize   = 0;
      fffTotSize  = 0;
      planeSize   = 0;

      printf("\n");
    }
  }

  FOLD // Do a sanity check on Flags and CUDA version  .
  {
    // TODO: do a check whether there is enough precision in an int to store the index of the largest point

    if ( master == NULL ) // For the moment lets try this on only the first card!
    {
      infoMSG(4,4,"FLAGS\n");

      // CUFFT callbacks
#if CUDA_VERSION < 6050
      cuSrch->sSpec->flags &= ~FLAG_CUFFT_ALL;
#endif

      if ( (cuSrch->sSpec->flags & FLAG_POW_HALF) && !(cuSrch->sSpec->flags & FLAG_SS_INMEM) && !(cuSrch->sSpec->flags & FLAG_CUFFT_CB_POW) )
      {
#if CUDA_VERSION >= 7050
	fprintf(stderr, "WARNING: Can't use half precision with out of memory search and no CUFFT callbacks. Reverting to single precision!\n");
#endif
	cuSrch->sSpec->flags &= ~FLAG_POW_HALF;
      }

      if ( !(cuSrch->sSpec->flags & FLAG_SS_INMEM) && (cuSrch->sSpec->flags & FLAG_CUFFT_CB_INMEM) )
      {
	fprintf(stderr, "WARNING: Can't use inmem callback with out of memory search. Disabling in-mem callback.\n");
	cuSrch->sSpec->flags &= ~FLAG_CUFFT_CB_INMEM;
      }

      if ( (cuSrch->sSpec->flags & FLAG_CUFFT_CB_POW) && (cuSrch->sSpec->flags & FLAG_CUFFT_CB_INMEM) )
      {
	fprintf(stderr, "WARNING: in-mem CUFFT callback will supersede power callback, I have found power callbacks to be the best.\n");
	cuSrch->sSpec->flags &= ~FLAG_CUFFT_CB_POW;
      }

      if ( (cuSrch->sSpec->flags & FLAG_SS_10) || (cuSrch->sSpec->flags & FLAG_SS_INMEM) )
      {
	cuSrch->sSpec->flags |= FLAG_RET_STAGES;
      }

      if ( !(cuSrch->sSpec->flags & FLAG_SS_INMEM) && (cuSrch->sSpec->flags & FLAG_Z_SPLIT) )
      {
	infoMSG(6,6,"Disabling in-mem plane split\n");

	cuSrch->sSpec->flags &= ~FLAG_Z_SPLIT;
      }

      char typeS[1024];
      sprintf(typeS, "Doing");

      if ( cuSrch->sSpec->flags & FLAG_SS_INMEM )
      {
	sprintf(typeS, "%s a in-memory search", typeS);
	if ( cuSrch->sSpec->flags & FLAG_Z_SPLIT )
	{
	  sprintf(typeS, "%s with a split plane", typeS);
	}
	else
	{
	  sprintf(typeS, "%s with the full plane", typeS);
	}
      }
      else
	sprintf(typeS, "%s an out of memory search", typeS);

      sprintf(typeS, "%s, using", typeS);
      if ( cuSrch->sSpec->flags & FLAG_POW_HALF )
	sprintf(typeS, "%s half", typeS);
      else
	sprintf(typeS, "%s single", typeS);

      sprintf(typeS, "%s precision powers", typeS);
      if ( cuSrch->sSpec->flags & FLAG_CUFFT_CB_POW )
	sprintf(typeS, "%s and CUFFT callbacks to calculate powers.", typeS);
      else if ( cuSrch->sSpec->flags & FLAG_CUFFT_CB_INMEM )
	sprintf(typeS, "%s and CUFFT callbacks to calculate powers and store in the full plane.", typeS);
      else
	sprintf(typeS, "%s and no CUFFT callbacks.", typeS);

      printf("%s\n\n", typeS);
    }

    FOLD // Determine the size of the elements of the planes  .
    {
      // Set power plane size
      if ( cuSrch->sSpec->flags & FLAG_DOUBLE )
      {
	cmpElsSZ = sizeof(double2);
      }
      else
      {
	cmpElsSZ = sizeof(fcomplexcu);
      }

      // Half precision plane
      if ( cuSrch->sSpec->flags & FLAG_POW_HALF )
      {
#if CUDA_VERSION >= 7050
	plnElsSZ = sizeof(half);
#else
	plnElsSZ = sizeof(float);
	fprintf(stderr, "WARNING: Half precision can only be used with CUDA 7.5 or later! Reverting to single precision!\n");
	cuSrch->sSpec->flags &= ~FLAG_POW_HALF;
#endif
      }
      else
      {
	plnElsSZ = sizeof(float);
      }

      // Set power plane size
      if ( cuSrch->sSpec->flags & FLAG_CUFFT_CB_POW )
      {
	powElsSZ = plnElsSZ;
      }
      else
      {
	powElsSZ = sizeof(fcomplexcu);
      }

    }
  }

  FOLD // Allocate and zero some structures  .
  {
    infoMSG(4,4,"Allocate and zero structures\n");

    FOLD // Initialise main pointer to this kernel  .
    {
      memset(kernel, 0, sizeof(cuFFdotBatch));

      if ( master != NULL )  // Copy all pointers and sizes from master. All non global pointers must be overwritten.
      {
	memcpy(kernel,  master,  sizeof(cuFFdotBatch));
	kernel->srchMaster  = 0;
      }
      else
      {
	kernel->flags         = cuSrch->sSpec->flags;
	kernel->srchMaster    = 1;
	kernel->noHarmStages  = cuSrch->noHarmStages;
	kernel->noGenHarms    = cuSrch->noGenHarms;
	kernel->noSrchHarms   = cuSrch->noSrchHarms;
      }
    }

    FOLD // Set the device specific parameters  .
    {
      kernel->cuSrch		= cuSrch;
      kernel->gInf		= gInf;
      kernel->isKernel		= 1;                // This is the device master
    }

    FOLD // Allocate memory  .
    {
      kernel->hInfos        = (cuHarmInfo*) malloc(kernel->noSrchHarms * sizeof(cuHarmInfo));
      kernel->kernels       = (cuKernel*)   malloc(kernel->noGenHarms * sizeof(cuKernel));

      // Zero memory for kernels and harmonics
      memset(kernel->hInfos,  0, kernel->noSrchHarms * sizeof(cuHarmInfo));
      memset(kernel->kernels, 0, kernel->noGenHarms  * sizeof(cuKernel));
    }
  }

  FOLD // Determine step size and how many stacks and how many planes in each stack  .
  {
    if ( master == NULL ) 	// Calculate details for the batch  .
    {
      FOLD // Determine step size  .
      {
	bool devisSS = false;						//< Weather to make the step size divisible for SS10 kernel

	printf("Determining GPU step size and plane width:\n");
	infoMSG(4,4,"Determining step size and width\n");

	FOLD // Get step size  .
	{
	  if ( !(cuSrch->sSpec->flags & FLAG_SS_INMEM)  )
	    devisSS = true;

	  kernel->accelLen = calcAccellen(cuSrch->sSpec->pWidth, cuSrch->sSpec->zMax, kernel->noGenHarms, accuracy, cuSrch->sSpec->noResPerBin, cuSrch->sSpec->zRes, devisSS);
	}

	FOLD // Print kernel accuracy  .
	{
	  printf(" • Using ");

	  if ( cuSrch->sSpec->flags & FLAG_KER_HIGH )
	  {
	    printf("high ");
	  }
	  else
	  {
	    printf("standard ");
	  }
	  printf("accuracy response functions.\n");

	  if ( cuSrch->sSpec->flags & FLAG_KER_MAX )
	    printf(" • Using maximum response function length for entire kernel.\n");
	}

	if ( kernel->accelLen > 100 ) // Print output
	{
	  double ratio	= 1;
	  double fftLen	= cu_calc_fftlen<double>(1, cuSrch->sSpec->zMax, kernel->accelLen, accuracy, cuSrch->sSpec->noResPerBin, cuSrch->sSpec->zRes);
	  double fWidth;
	  int oAccelLen;

	  if ( cuSrch->sSpec->pWidth > 100 ) // User specified step size, check how close to optimal it is
	  {	  
	    double l2	= log2( fftLen ) - 10 ;
	    fWidth	= pow(2, l2);
	    oAccelLen	= calcAccellen(fWidth, cuSrch->sSpec->zMax, kernel->noGenHarms, accuracy, cuSrch->sSpec->noResPerBin, cuSrch->sSpec->zRes, devisSS);
	    ratio	= kernel->accelLen/double(oAccelLen);
	  }

	  printf(" • Using max plane width of %.0f and", fftLen);

	  if ( ratio < 1 )
	  {
	    printf(" a suboptimal step-size of %i. (%.2f%% of optimal) \n",  kernel->accelLen, ratio*100 );
	    printf("   > For a zmax of %.1f using %iK FFTs the optimal step-size is %i.\n", cuSrch->sSpec->zMax, (int)fWidth, oAccelLen);

	    if ( cuSrch->sSpec->pWidth > 100 )
	    {
	      fprintf(stderr,"     WARNING: Using manual width\\step-size is not advised rather set width to one of 2 4 8 16 32.\n");
	    }
	  }
	  else
	  {
	    printf(" a optimal step-size of %i.\n", kernel->accelLen );
	  }
	}
	else
	{
	  fprintf(stderr,"ERROR: With a width of %i, the step-size would be %i and this is too small, try with a wider width or lower z-max.\n", cuSrch->sSpec->pWidth, kernel->accelLen);
	  exit(EXIT_FAILURE);
	}
      }

      FOLD // Set some harmonic related values  .
      {
	int prevWidth       = 0;
	int noStacks        = 0;
	int stackHW         = 0;
	int hIdx, sIdx;
	double hFrac;

	FOLD // Set up basic details of all the harmonics  .
	{
	  infoMSG(4,4,"Determine number of stacks and planes\n");

	  for (int i = kernel->noSrchHarms; i > 0; i--)
	  {
	    cuHarmInfo* hInfs;
	    hFrac               = (i) / (double)kernel->noSrchHarms;
	    hIdx                = kernel->noSrchHarms-i;
	    hInfs               = &kernel->hInfos[hIdx];                              // Harmonic index

	    hInfs->harmFrac     = hFrac;
	    hInfs->zmax         = cu_calc_required_z<double>(hInfs->harmFrac, cuSrch->sSpec->zMax, cuSrch->sSpec->zRes);
	    hInfs->width        = cu_calc_fftlen<double>(hInfs->harmFrac, kernel->hInfos[0].zmax, kernel->accelLen, accuracy, cuSrch->sSpec->noResPerBin, cuSrch->sSpec->zRes);
	    hInfs->halfWidth    = cu_z_resp_halfwidth<double>(hInfs->zmax, accuracy);
	    hInfs->noResPerBin  = cuSrch->sSpec->noResPerBin;

	    if ( kernel->flags & FLAG_Z_SPLIT )
	    {
	      hInfs->zStart	= cu_calc_required_z<double>(1, 0.0, cuSrch->sSpec->zRes);
	      hInfs->zEnd	= cu_calc_required_z<double>(1, hInfs->zmax, cuSrch->sSpec->zRes);
	    }
	    else
	    {
	      // This is the CPU orientation
	      hInfs->zStart	= cu_calc_required_z<double>(1, -hInfs->zmax, cuSrch->sSpec->zRes);
	      hInfs->zEnd	= cu_calc_required_z<double>(1,  hInfs->zmax, cuSrch->sSpec->zRes);
	    }
	    hInfs->noZ       	= round(fabs(hInfs->zEnd - hInfs->zStart) / cuSrch->sSpec->zRes) + 1;

	    if ( prevWidth != hInfs->width )	// Stack creation and checks
	    {
	      infoMSG(5,5,"New stack\n");

	      // We have a new stack
	      noStacks++;

	      if ( hIdx < kernel->noGenHarms )
	      {
		kernel->noStacks = noStacks;
	      }

	      noInStack[noStacks - 1]       = 0;
	      prevWidth                     = hInfs->width;
	      stackHW                       = cu_z_resp_halfwidth<double>(hInfs->zmax, accuracy);

	      // Maximise, centre and align halfwidth
	      int   sWidth                  = (int) ( ceil(kernel->accelLen * hInfs->harmFrac / (double)cuSrch->sSpec->noResPerBin ) * cuSrch->sSpec->noResPerBin ) + 1 ;	// Width of usable data for this plane
	      float centHW                  = (hInfs->width  - sWidth)/2.0/(double)cuSrch->sSpec->noResPerBin;									//
	      float noAlg                   = gInf->alignment / float(sizeof(fcomplex)) / (double)cuSrch->sSpec->noResPerBin ;							// halfWidth will be multiplied by ACCEL_NUMBETWEEN so can divide by it here!
	      float centAlgnHW              = floor(centHW/noAlg) * noAlg ;													// Centre and aligned half width

	      if ( stackHW > centAlgnHW )
	      {
		stackHW                     = floor(centHW);

		infoMSG(6,6,"can not align stack half width GPU value. Using %i \n", stackHW );
	      }
	      else
	      {
		stackHW                     = centAlgnHW;

		infoMSG(6,6,"aligned stack half width for GPU is %i \n", stackHW );
	      }
	    }

	    infoMSG(6,6,"Harm: %2i  frac %5.3f  z-max: %5.1f  z: %7.2f to %7.2f  width: %5i half width %4i \n", i, hFrac, hInfs->zmax, hInfs->zStart, hInfs->zEnd, hInfs->width, hInfs->halfWidth );

	    hInfs->stackNo      = noStacks-1;

	    if ( kernel->flags & FLAG_CENTER )
	    {
	      hInfs->kerStart   = stackHW*cuSrch->sSpec->noResPerBin;
	    }
	    else
	    {
	      hInfs->kerStart   = hInfs->halfWidth*cuSrch->sSpec->noResPerBin;
	    }

	    if ( hIdx < kernel->noGenHarms )
	    {
	      noInStack[noStacks - 1]++;
	    }
	  }
	}

	FOLD // Set up the indexing details of all the harmonics  .
	{

	  infoMSG(4,4,"Indexing harmonics.\n");

	  // Calculate the stage order of the harmonics
	  sIdx = 0;

	  for ( int stage = 0; stage < kernel->noHarmStages; stage++ )
	  {
	    infoMSG(5,5,"Stage %i \n", stage);

	    int harmtosum = 1 << stage;
	    for (int harm = 1; harm <= harmtosum; harm += 2, sIdx++)
	    {
	      hFrac       = harm/float(harmtosum);
	      hIdx        = hFrac == 1 ? 0 : round(hFrac*kernel->noSrchHarms);

	      kernel->hInfos[hIdx].stageIndex	= sIdx;
	      cuSrch->sIdx[sIdx]		= hIdx;

	      infoMSG(6,6,"Fraction: %5.3f ( %i/%i ), Harmonic idx %2i, Stage idx %2i \n", hFrac, harm, harmtosum, hIdx, sIdx );
	    }
	  }
	}
      }
    }
    else			// Copy details from the master batch  .
    {
      // Copy memory from kernels and harmonics
      memcpy(kernel->hInfos,  master->hInfos,  kernel->noSrchHarms * sizeof(cuHarmInfo));
      memcpy(kernel->kernels, master->kernels, kernel->noGenHarms  * sizeof(cuKernel));
    }
  }

  FOLD // Allocate all the memory for the stack data structures  .
  {
    infoMSG(4,4,"Allocate memory for stacks\n");

    long long neede = kernel->noStacks * sizeof(cuFfdotStack) + kernel->noSrchHarms * sizeof(cuHarmInfo) + kernel->noGenHarms * sizeof(cuKernel);

    if ( neede > getFreeRamCU() )
    {
      fprintf(stderr, "ERROR: Not enough host memory for search.\n");
    }
    else
    {
      // Set up stacks
      kernel->stacks = (cuFfdotStack*) malloc(kernel->noStacks* sizeof(cuFfdotStack));

      if ( master == NULL )
      {
	memset(kernel->stacks, 0, kernel->noStacks * sizeof(cuFfdotStack));
      }
      else
      {
	memcpy(kernel->stacks, master->stacks, kernel->noStacks * sizeof(cuFfdotStack));

	FOLD // Zero some of the relevant values in the stack
	{
	  for (int i = 0; i < kernel->noStacks; i++)           // Loop through Stacks  .
	  {
	    cuFfdotStack* cStack  = &kernel->stacks[i];

	    cStack->plnPlan       = 0;
	    cStack->inpPlan       = 0;
	  }
	}
      }
    }
  }

  FOLD // Set up the basic details of all the stacks and calculate the stride  .
  {
    FOLD // Set up the basic details of all the stacks  .
    {
      if ( master == NULL )
      {
	infoMSG(4,4,"Stack details\n");

	int prev                = 0;
	for (int i = 0; i < kernel->noStacks; i++)           // Loop through Stacks  .
	{
	  cuFfdotStack* cStack  = &kernel->stacks[i];
	  cStack->height        = 0;
	  cStack->noInStack     = noInStack[i];
	  cStack->startIdx      = prev;
	  cStack->harmInf       = &kernel->hInfos[cStack->startIdx];
	  cStack->kernels       = &kernel->kernels[cStack->startIdx];
	  cStack->width         = cStack->harmInf->width;
	  cStack->kerHeigth     = cStack->harmInf->noZ;
	  cStack->flags         = kernel->flags;               // Used to create the kernel, will be over written later

	  for (int j = 0; j < cStack->noInStack; j++)
	  {
	    cStack->startZ[j]   = cStack->height;
	    cStack->height     += cStack->harmInf[j].noZ;
	  }

	  prev                 += cStack->noInStack;
	}
      }
    }

    FOLD // Calculate the stride and data thus data size of the stacks  .
    {
      // This is device specific so done on each card

      infoMSG(4,4,"Stride details\n");

      kernel->inpDataSize     = 0;
      kernel->kerDataSize     = 0;
      kernel->plnDataSize     = 0;
      kernel->pwrDataSize     = 0;

      for (int i = 0; i < kernel->noStacks; i++)          // Loop through Stacks  .
      {
	cuFfdotStack* cStack  = &kernel->stacks[i];

	FOLD // Compute size of
	{
	  // Compute stride  .
	  cStack->strideCmplx =   getStrie(cStack->width, cmpElsSZ, gInf->alignment);
	  cStack->stridePower =   getStrie(cStack->width, powElsSZ, gInf->alignment);

	  kernel->inpDataSize +=  cStack->strideCmplx * cStack->noInStack * sizeof(cufftComplex);
	  kernel->kerDataSize +=  cStack->strideCmplx * cStack->kerHeigth * cmpElsSZ;
	  kernel->plnDataSize +=  cStack->strideCmplx * cStack->height    * cmpElsSZ;

	  if ( !(kernel->flags & FLAG_CUFFT_CB_INMEM) )
	    kernel->pwrDataSize +=  cStack->stridePower * cStack->height  * powElsSZ;
	}
      }
    }
  }

  FOLD // Batch specific streams  .
  {
    PROF // Profiling  .
    {
      NV_RANGE_PUSH("streams");
    }

    infoMSG(4,4,"Batch streams\n");

    char strBuff[1024];

    if ( kernel->flags & FLAG_SYNCH )
    {
      cuFfdotStack* fStack = &kernel->stacks[0];

      CUDA_SAFE_CALL(cudaStreamCreate(&fStack->initStream),"Creating CUDA stream for initialisation");

      sprintf(strBuff,"%i.0.0.0 Initialisation", kernel->gInf->devid );
      NV_NAME_STREAM(fStack->initStream, strBuff);
      //printf("cudaStreamCreate: %s\n", strBuff);

      for (int i = 0; i < kernel->noStacks; i++)
      {
	cuFfdotStack* cStack = &kernel->stacks[i];

	cStack->initStream = fStack->initStream;
      }
    }
    else
    {
      for (int i = 0; i < kernel->noStacks; i++)
      {
	cuFfdotStack* cStack = &kernel->stacks[i];

	CUDA_SAFE_CALL(cudaStreamCreate(&cStack->initStream),"Creating CUDA stream for initialisation");

	sprintf(strBuff,"%i.0.0.%i Initialisation", kernel->gInf->devid, i);
	NV_NAME_STREAM(cStack->initStream, strBuff);
	//printf("cudaStreamCreate: %s\n", strBuff);
      }
    }

    if ( !(kernel->flags & CU_FFT_SEP) )
    {
      if ( !(kernel->flags & CU_INPT_FFT_CPU) )
      {
	for (int i = 0; i < kernel->noStacks; i++)
	{
	  cuFfdotStack* cStack = &kernel->stacks[i];

	  CUDA_SAFE_CALL(cudaStreamCreate(&cStack->fftIStream),"Creating CUDA stream for fft's");
	  sprintf(strBuff,"%i.0.2.%i FFT Input Dev", kernel->gInf->devid, i);
	  NV_NAME_STREAM(cStack->fftIStream, strBuff);
	  //printf("cudaStreamCreate: %s\n", strBuff);
	}
      }

      for (int i = 0; i < kernel->noStacks; i++)
      {
	cuFfdotStack* cStack = &kernel->stacks[i];

	CUDA_SAFE_CALL(cudaStreamCreate(&cStack->fftPStream),"Creating CUDA stream for fft's");
	sprintf(strBuff,"%i.0.4.%i FFT Plane Dev", kernel->gInf->devid, i);
	NV_NAME_STREAM(cStack->fftPStream, strBuff);
	//printf("cudaStreamCreate: %s\n", strBuff);
      }
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP(); // streams
    }
  }

  FOLD // Allocate device memory for all the kernels data  .
  {
    PROF // Profiling  .
    {
      NV_RANGE_PUSH("kernel malloc");
    }

    infoMSG(4,4,"Allocate device memory for all the kernels data\n");

    if ( kernel->kerDataSize > free )
    {
      fprintf(stderr, "ERROR: Not enough device memory for GPU multiplication kernels. There is only %.2f MB free and you need %.2f MB \n", free / 1048576.0, kernel->kerDataSize / 1048576.0 );
      freeKernel(kernel);
      return (0);
    }
    else
    {
      CUDA_SAFE_CALL(cudaMalloc((void**)&kernel->d_kerData, kernel->kerDataSize), "Failed to allocate device memory for kernel stack.");
      CUDA_SAFE_CALL(cudaGetLastError(), "Allocation of device memory for kernel?.\n");
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP(); // kernel malloc
    }
  }

  FOLD // Multiplication kernels  .
  {
    FOLD // Set the sizes values of the harmonics and kernels and pointers to kernel data  .
    {
      setKernelPointers(kernel);
    }

    if ( master == NULL )     // Create the kernels  .
    {
      infoMSG(4,4,"Initialise the multiplication kernels.\n");

      FOLD // Check contamination of the largest stack  .
      {
	float contamination = (kernel->hInfos->halfWidth*2*cuSrch->sSpec->noResPerBin)/(float)kernel->hInfos->width*100 ;
	if ( contamination > 25 )
	{
	  fprintf(stderr, "WARNING: Contamination is high, consider increasing width with the -width flag.\n");
	}
      }

      FOLD // Print details on the stacks  .
      {
	printf("\n");

	int hh      = 1;
	for (int i = 0; i < kernel->noStacks; i++)
	{
	  cuFfdotStack* cStack = &kernel->stacks[i];

	  float contamination = (cStack->harmInf->halfWidth*2*cuSrch->sSpec->noResPerBin)/(float)cStack->harmInf->width*100 ;
	  float padding       = (1-(kernel->accelLen*cStack->harmInf->harmFrac + cStack->harmInf->halfWidth*2*cuSrch->sSpec->noResPerBin ) / cStack->harmInf->width)*100.0 ;

	  printf("  ■ Stack %i has %02i f-∂f plane(s). width: %5li  stride: %5li  Height: %6li  Memory size: %7.1f MB \n", i+1, cStack->noInStack, cStack->width, cStack->strideCmplx, cStack->height, cStack->height*cStack->strideCmplx*sizeof(fcomplex)/1024.0/1024.0);

	  printf("    ► Created kernel %i  Size: %7.1f MB  Height %4lu   Contamination: %5.2f %%  Padding: %5.2f %%\n", i+1, cStack->harmInf->noZ*cStack->strideCmplx*sizeof(fcomplex)/1024.0/1024.0, cStack->harmInf->noZ, contamination, padding);

	  for (int j = 0; j < cStack->noInStack; j++)
	  {
	    printf("      • Harmonic %02i  Fraction: %5.3f   Z-Max: %6.1f   Half Width: %4i  Start offset: %4i \n", hh, cStack->harmInf[j].harmFrac, cStack->harmInf[j].zmax, cStack->harmInf[j].halfWidth, cStack->harmInf[j].kerStart / cuSrch->sSpec->noResPerBin  );
	    hh++;
	  }
	}
      }

      printf("\nGenerating GPU multiplication kernels using device %i (%s).\n\n", kernel->gInf->devid, kernel->gInf->name);

      createBatchKernels(kernel, NULL);
    }
  }

  FOLD // Decide how to handle input and output and allocate required memory  .
  {
    infoMSG(4,4,"Input and output.\n");

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("data");
    }

    printf("\nInitializing GPU %i (%s)\n", kernel->gInf->devid, cuSrch->gSpec->devInfo[devID].name );

    if ( master != NULL )     // Copy the kernels  .
    {
      infoMSG(4,4,"Copy multiplication kernels\n");

      // TODO: Check this works in this location
      printf("• Copying multiplication kernels from device %i.\n", master->gInf->devid);
      CUDA_SAFE_CALL(cudaMemcpyPeerAsync(kernel->d_kerData, kernel->gInf->devid, master->d_kerData, master->gInf->devid, master->kerDataSize, master->stacks->initStream ), "Copying multiplication kernels between devices.");
    }

    printf("• Examining GPU memory of device %2i:\n", kernel->gInf->devid);

    ulong freeRam;          /// The amount if free host memory
    int retSZ     = 0;      /// The size in byte of the returned data
    int candSZ    = 0;      /// The size in byte of the candidates
    int retY      = 0;      /// The number of candidates return per family (one step)
    ulong hostC   = 0;      /// The size in bytes of device memory used for candidates

    FOLD // Calculate the search size in bins  .
    {
      if ( master == NULL )
      {
	int minR              = floor ( cuSrch->sSpec->fftInf.rlo /(double) kernel->noSrchHarms - kernel->hInfos->halfWidth );
	int maxR              = ceil  ( cuSrch->sSpec->fftInf.rhi  + kernel->hInfos->halfWidth );

	searchScale* SrchSz   = new searchScale;
	cuSrch->SrchSz          = SrchSz;
	memset(SrchSz, 0, sizeof(searchScale));

	SrchSz->searchRLow    = cuSrch->sSpec->fftInf.rlo / (double)kernel->noSrchHarms;
	SrchSz->searchRHigh   = cuSrch->sSpec->fftInf.rhi;
	SrchSz->rLow          = minR;
	SrchSz->rHigh         = maxR;
	SrchSz->noInpR        = maxR - minR  ;  /// The number of input data points

	// Determine the number of steps
	if ( kernel->flags & FLAG_SS_INMEM   )
	{
	  SrchSz->noSteps     = ( SrchSz->searchRHigh - SrchSz->searchRLow ) * cuSrch->sSpec->noResPerBin / (float)( kernel->accelLen ) ; // The number of planes to make
	}
	else
	{
	  SrchSz->noSteps       = ( cuSrch->sSpec->fftInf.rhi - cuSrch->sSpec->fftInf.rlo ) * cuSrch->sSpec->noResPerBin / (float)( kernel->accelLen ) ; // The number of planes to make
	}

	// Determin the number of canidate 'r' values
	if ( kernel->flags  & FLAG_STORE_EXP )
	{
	  SrchSz->noOutpR     = ceil( (SrchSz->searchRHigh - SrchSz->searchRLow)*cuSrch->sSpec->noResPerBin );
	}
	else
	{
	  SrchSz->noOutpR     = ceil(SrchSz->searchRHigh - SrchSz->searchRLow);
	}

	if ( (kernel->flags & FLAG_STORE_ALL) && !( kernel->flags  & FLAG_RET_STAGES) )
	{
	  printf("   Storing all results implies returning all results so adding FLAG_RET_STAGES to flags!\n");
	  kernel->flags  |= FLAG_RET_STAGES;
	}
      }
    }

    FOLD // Chunks and Slices  .
    {
      FOLD // Multiplication defaults are set per batch  .
      {
	kernel->mulSlices         = cuSrch->sSpec->mulSlices;
	kernel->mulChunk          = cuSrch->sSpec->mulChunk;

	FOLD // Set stack multiplication slices
	{
	  for (int i = 0; i < kernel->noStacks; i++)
	  {
	    cuFfdotStack* cStack  = &kernel->stacks[i];
	    cStack->mulSlices     = cuSrch->sSpec->mulSlices;
	    cStack->mulChunk      = cuSrch->sSpec->mulChunk;
	  }
	}
      }

      FOLD // Sum  & search  .
      {
	kernel->ssChunk           = cuSrch->sSpec->ssChunk;
	kernel->ssSlices          = cuSrch->sSpec->ssSlices;

	if ( kernel->ssSlices <= 0 )
	{
	  if      ( kernel->stacks->width <= 1024 )
	  {
	    kernel->ssSlices      = 8 ;
	  }
	  else if ( kernel->stacks->width <= 2048 )
	  {
	    kernel->ssSlices      = 4 ;
	  }
	  else if ( kernel->stacks->width <= 4096 )
	  {
	    kernel->ssSlices      = 2 ;
	  }
	  else
	  {
	    kernel->ssSlices      = 1 ;
	  }

	}
	kernel->ssSlices          = MIN(kernel->ssSlices, ceil(kernel->hInfos->noZ/20.0) );
      }
    }

    FOLD // Calculate candidate type  .
    {
      if ( master == NULL )   // There is only one list of candidates per search so only do this once!
      {
	kernel->cndType         = cuSrch->sSpec->cndType;

	if      ( !(kernel->cndType & CU_TYPE_ALLL) )
	{
	  fprintf(stderr,"Warning: No candidate data type specified in %s. Setting to default.\n",__FUNCTION__);
	  kernel->cndType = CU_CANDFULL;
	}

	if      (kernel->cndType & CU_CMPLXF   )
	{
	  candSZ = sizeof(fcomplexcu);
	}
	else if (kernel->cndType & CU_INT      )
	{
	  candSZ = sizeof(int);
	}
	else if (kernel->cndType & CU_FLOAT    )
	{
	  candSZ = sizeof(float);
	}
	else if (kernel->cndType & CU_POWERZ_S )
	{
	  candSZ = sizeof(candPZs);
	}
	else if (kernel->cndType & CU_POWERZ_I )
	{
	  candSZ = sizeof(candPZi);
	}
	else if (kernel->cndType & CU_CANDMIN  )
	{
	  candSZ = sizeof(candMin);
	}
	else if (kernel->cndType & CU_CANDSMAL )
	{
	  candSZ = sizeof(candSml);
	}
	else if (kernel->cndType & CU_CANDBASC )
	{
	  candSZ = sizeof(accelcandBasic);
	}
	else if (kernel->cndType & CU_CANDFULL )  // This should be the default
	{
	  candSZ = sizeof(initCand);
	}
	else
	{
	  fprintf(stderr,"ERROR: No output type specified in %s setting to default.\n", __FUNCTION__);
	  kernel->cndType |= CU_CANDFULL;
	  candSZ = sizeof(initCand);
	}

	if      ( !(kernel->cndType & CU_SRT_ALL   ) ) // Set defaults  .
	{
	  fprintf(stderr,"Warning: No candidate storage type specified in %s. Setting to default.\n",__FUNCTION__);
	  kernel->cndType = CU_STR_ARR   ;
	}
      }
    }

    FOLD // Calculate return type, size and data structure  .
    {
      kernel->retType       = cuSrch->sSpec->retType;

      if      (kernel->retType & CU_STR_PLN   )
      {
	if (  (kernel->flags & FLAG_CUFFT_CB_POW) && ( !( (kernel->retType & CU_HALF) || (kernel->retType & CU_FLOAT)))   )
	{
	  fprintf(stderr,"WARNING: Returning plane and CUFFT output requires float return type.\n");
	  kernel->retType &= ~CU_TYPE_ALLL;
	  kernel->retType |= CU_FLOAT;
	}

	if ( !(kernel->flags & FLAG_CUFFT_CB_POW) && !(kernel->retType & CU_CMPLXF) )
	{
	  fprintf(stderr,"WARNING: Returning plane requires complex float return type.\n");
	  kernel->retType &= ~CU_TYPE_ALLL;
	  kernel->retType |= CU_CMPLXF;
	}

	if ( kernel->flags & FLAG_SIG_GPU )
	{
	  fprintf(stderr,"WARNING: Cannot do GPU sigma calculations when returning plane data.\n");
	  kernel->flags &= ~FLAG_SIG_GPU;
	}
      }

      if      (kernel->retType & CU_CMPLXF    )
      {
	retSZ = sizeof(fcomplexcu);
      }
      else if (kernel->retType & CU_INT       )
      {
	retSZ = sizeof(int);
      }
      else if (kernel->retType & CU_HALF      )
      {
#if CUDA_VERSION >= 7050
	retSZ = sizeof(half);
#else
	fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
	exit(EXIT_FAILURE);
#endif
      }
      else if (kernel->retType & CU_FLOAT     )
      {
	retSZ = sizeof(float);
      }
      else if (kernel->retType & CU_DOUBLE    )
      {
	retSZ = sizeof(double);
      }
      else if (kernel->retType & CU_POWERZ_S  )
      {
	retSZ = sizeof(candPZs);
      }
      else if (kernel->retType & CU_POWERZ_I  )
      {
	retSZ = sizeof(candPZi);
      }
      else if (kernel->retType & CU_CANDMIN   )
      {
	retSZ = sizeof(candMin);
      }
      else if (kernel->retType & CU_CANDSMAL  )
      {
	retSZ = sizeof(candSml);
      }
      else if (kernel->retType & CU_CANDBASC  )
      {
	retSZ = sizeof(accelcandBasic);
      }
      else if (kernel->retType & CU_CANDFULL  )
      {
	retSZ = sizeof(initCand);
      }
      else
      {
	fprintf(stderr,"ERROR: No output type specified in %s\n",__FUNCTION__);
	kernel->retType &= ~CU_TYPE_ALLL ;
	kernel->retType |=  CU_POWERZ_S ;
	retSZ = sizeof(candPZs);
      }

      FOLD // Sum and search slices  .
      {
	if      ( kernel->retType & CU_STR_PLN )
	{
	  // Each stage returns a plane the size of the fundamental
	  retY = kernel->hInfos->noZ;
	}
	else
	{
	  retY = kernel->ssSlices;
	}
      }

      FOLD // Return data structure  .
      {
	if      ( kernel->flags & FLAG_SS_INMEM )
	{
	  // NOTE: The in-mem sum and search does not need the search step size to be divisible by number of harmonics

	  if ( cuSrch->sSpec->ssStepSize <= 100 )
	  {
	    if ( cuSrch->sSpec->ssStepSize > 0 )
	      fprintf(stderr, "WARNING: In-mem plane search stride too small try 0 or something larger that 100 say 16384 or 32768.\n");

	    kernel->strideOut = 32768; // TODO: I need to check for a good default
	  }
	  else
	  {
	    kernel->strideOut = cuSrch->sSpec->ssStepSize;
	  }
	}
	else if ( (kernel->retType & CU_STR_ARR) || (kernel->retType & CU_STR_LST) || (kernel->retType & CU_STR_QUAD) )
	{
	  kernel->strideOut = kernel->accelLen;
	}
	else if (  kernel->retType & CU_STR_PLN  )
	{
	  if      ( kernel->retType & CU_FLOAT  )
	  {
	    kernel->strideOut = kernel->stacks->stridePower ;
	  }
	  else if ( kernel->retType & CU_HALF   )
	  {
	    kernel->strideOut = kernel->stacks->stridePower ;
	  }
	  else if ( kernel->retType & CU_CMPLXF )
	  {
	    kernel->strideOut = kernel->stacks->strideCmplx ;
	  }
	  else
	  {
	    fprintf(stderr,"ERROR: CUDA return type not compatible with returning plane.\n");
	    exit(EXIT_FAILURE);
	  }
	}
	else
	{
	  fprintf(stderr,"ERROR: CUDA return structure not specified.\n");
	  exit(EXIT_FAILURE);
	}
      }

      // Calculate return data size for one step
      kernel->retDataSize   = retY*kernel->strideOut*retSZ;

      if ( kernel->flags & FLAG_RET_STAGES )
	kernel->retDataSize *= kernel->noHarmStages;

      infoMSG(6,6,"retSZ: %i  alignment: %i  strideOut: %i  retDataSize: %i \n", retSZ, kernel->gInf->alignment, kernel->strideOut, kernel->retDataSize);
    }

    FOLD // Calculate batch size and number of steps and batches on this device  .
    {
      PROF // Profiling  .
      {
	NV_RANGE_PUSH("Calc steps");
      }

      CUDA_SAFE_CALL(cudaMemGetInfo ( &free, &total ), "Getting Device memory information"); // TODO: This call may not be necessary we could calculate this from previous values
#ifdef MAX_GPU_MEM
      long  Diff = total - MAX_GPU_MEM;
      if( Diff > 0 )
      {
	free-= Diff;
	total-=Diff;
      }
#endif
      freeRam = getFreeRamCU();

      printf("   There is a total of %.2f GiB of device memory, %.2f GiB is free. There is %.2f GiB free host memory.\n",total / 1073741824.0, (free ) / 1073741824.0, freeRam / 1073741824.0 );

      FOLD // Calculate size of various memory's'  .
      {
	batchSize             = kernel->inpDataSize + kernel->plnDataSize + kernel->pwrDataSize + kernel->retDataSize;  // This is currently the size of one step
	fffTotSize            = kernel->inpDataSize + kernel->plnDataSize;                                              // FFT data treated separately because there will be only one set per device

	infoMSG(5,5,"inpDataSize: %.2f GB - plnDataSize: ~%.2f MB - pwrDataSize: ~%.2f MB - retDataSize: ~%.2f MB \n", kernel->inpDataSize*1e-6, kernel->plnDataSize*1e-6, kernel->pwrDataSize*1e-6, kernel->retDataSize*1e-6 ); //  free*1e-9, planeSize*1e-9, kerSize*1e-6, batchSize*1e-6, fffTotSize*1e-6, singleBatchSz*1e-6 );

	if ( kernel->flags & FLAG_SS_INMEM  ) // Size of memory for plane full ff plane  .
	{
	  size_t noStepsP       = ceil(cuSrch->SrchSz->noSteps / (float)noSteps) * noSteps;
	  size_t nX             = noStepsP * kernel->accelLen;
	  size_t nY             = kernel->hInfos->noZ;
	  planeSize	        = nX * nY * plnElsSZ;
	}
      }

      kerSize                   = kernel->kerDataSize;
      familySz                  = kerSize + batchSize + fffTotSize;

      infoMSG(5,5,"Free: %.3fGB  - in-mem plane: %.2f GB - Kernel: ~%.2f MB - batch: ~%.2f MB - fft: ~%.2f MB - full batch: ~%.2f MB  - %i \n", free*1e-9, planeSize*1e-9, kerSize*1e-6, batchSize*1e-6, fffTotSize*1e-6, familySz*1e-6, batchSize );

      FOLD // Calculate how many batches and steps to do  .
      {
	FOLD // No steps possible for given number of batches
	{
	  float possSteps[MAX_BATCHES];
	  bool trySomething = 0;

	  // Calculate the maximum number of steps for each possible number if steps
	  for ( int i = 0; i < MAX_BATCHES; i++)
	  {
	    // The number of
	    if ( kernel->flags & CU_FFT_SEP )
	    {
	      possSteps[i] = ( free - planeSize ) / (double) ( (fffTotSize + batchSize) * (i+1) ) ;
	    }
	    else
	    {
	      possSteps[i] = ( free - planeSize ) / (double) (  fffTotSize + batchSize  * (i+1) ) ;  // (fffTotSize * possSteps) for the CUFFT memory for FFT'ing the plane(s) and (totSize * noThreads * possSteps) for each thread(s) plan(s)
	    }
	  }

	  noBatches       = cuSrch->gSpec->noDevBatches[devID];
	  noSteps         = cuSrch->gSpec->noDevSteps[devID];

	  if ( noBatches == 0 )
	  {
	    if ( noSteps == 0 )
	    {
	      // We have free range to do what we want!
	      trySomething = 1;
	    }
	    else
	    {
	      int maxBatches = 0;

	      // Determine the maximum number batches for the given steps
	      for ( int i = 0; i < MAX_BATCHES; i++)
	      {
		if ( possSteps[i] > noSteps )
		  maxBatches = i+1;
		else
		  break;
	      }

	      if      ( maxBatches > 3 )
	      {
		printf("     Requested %i steps per batch, could do up to %i batches, using 3.\n", noSteps, maxBatches);
		// Lets just do 3 batches, more than that doesn't really help often
		noBatches         = 3;
		kernel->noSteps   = noSteps;
	      }
	      else if ( maxBatches > 2 )
	      {
		printf("     Requested %i steps per batch, can do 2 batches.\n", noSteps);
		// Lets do 2 batches
		noBatches         = 2;
		kernel->noSteps   = noSteps;
	      }
	      else if ( maxBatches > 1 )
	      {
		// Lets just do 2 batches
		printf("     Requested %i steps per batch, can only do 1 batch.\n", noSteps);
		if ( noSteps >= 4 )
		  printf("       WARNING: Requested %i steps per batch, can only do 1 batch, perhaps consider using fewer steps.\n", noSteps );
		noBatches         = 1;
		kernel->noSteps   = noSteps;
	      }
	      else
	      {
		printf("       ERROR: Can't have 1 batch with the requested %i steps.\n", noSteps);
		// Well we can't do one one batch with the desired steps
		// Auto scale!
		trySomething = 1;
	      }
	    }
	  }
	  else
	  {
	    printf("     Requested %i batches.\n", noBatches);

	    if ( noSteps == 0 )
	    {
	      if ( possSteps[noBatches-1] >= 1 )
	      {
		// do as many steps as possible!
		kernel->noSteps   = floor(possSteps[noBatches-1]);
		MINN(kernel->noSteps, MAX_STEPS );
		printf("     With %i batches, can do %i steps.\n", noBatches, kernel->noSteps);
		if ( noBatches >= 3 && kernel->noSteps < 3 )
		{
		  printf("       WARNING: %i steps is quite low, perhaps consider using fewer batches.\n", kernel->noSteps);
		}
	      }
	      else
	      {
		printf("       ERROR: It is not possible to have %i batches with at least one step each on this device.\n", noBatches );
		trySomething = 1;
	      }
	    }
	    else
	    {
	      if ( possSteps[noBatches-1] >= noSteps )
	      {
		printf("     Requested %i steps per batch on this device.\n", noSteps);

		// We can do what we asked for!
		kernel->noSteps   = noSteps;
	      }
	      else
	      {
		printf("     ERROR: Can't have %i batches with %i steps on this device.\n", noBatches, noSteps);
		trySomething = 1;
	      }
	    }
	  }

	  if ( trySomething )
	  {
	    printf("     Determining a combination of batches and steps.\n");
	    if      ( possSteps[2] >= 4 )
	    {
	      noBatches         = 3;
	      kernel->noSteps   = floor(possSteps[noBatches-1]);
	      printf("       Can have %0.1f steps with %i batches.\n", possSteps[noBatches-1], noBatches );
	    }
	    else if ( possSteps[1] >= 2 )
	    {
	      // Lets do 2 batches and scale steps
	      noBatches         = 2;
	      kernel->noSteps   = floor(possSteps[noBatches-1]);
	      printf("       Can have %0.1f steps with %i batches.\n", possSteps[noBatches-1], noBatches );
	    }
	    else if ( possSteps[0] >  1 )
	    {
	      // Lets do 2 batches and scale steps
	      noBatches         = 1;
	      kernel->noSteps   = floor(possSteps[noBatches-1]);
	      printf("       Can only have %0.1f steps with %i batch.\n", possSteps[noBatches-1], noBatches );
	    }
	    else
	    {
	      // Well we can't really do anything!
	      noBatches = 0;
	      kernel->noSteps = 0;
	      printf("       ERROR: Can only have %0.1f steps with %i batch.\n", possSteps[0], noBatches );
	    }
	    MINN(kernel->noSteps, MAX_STEPS );
	  }

	  if ( kernel->noSteps > MAX_STEPS )
	  {
	    kernel->noSteps = MAX_STEPS;
	    printf("      Trying to use more steps that the maximum number (%i) this code is compiled with.\n", kernel->noSteps );
	  }

	  if ( noBatches <= 0 || kernel->noSteps <= 0 )
	  {
	    fprintf(stderr, "ERROR: Insufficient memory to make make any planes. One step would require %.2fGiB of device memory.\n", ( fffTotSize + batchSize )/1073741824.0 );

	    freeKernel(kernel);
	    return (0);
	  }

#ifdef CBL        // DBG TMP remove this
	  if ( (cuSrch->gSpec->noDevSteps[devID] && ( kernel->noSteps != cuSrch->gSpec->noDevSteps[devID]) ) || (cuSrch->gSpec->noDevBatches[devID] && ( noBatches != cuSrch->gSpec->noDevBatches[devID]) )  )
	  {
	    fprintf(stderr, "ERROR: Dropping out cos we can't have the requested steps and batches.\n");
	    freeKernel(kernel);
	    return (0);
	  }
#endif
	}

	// Recalculate planeSize (with new step count)
	if ( kernel->flags & FLAG_SS_INMEM  ) // Size of memory for plane full ff plane  .
	{
	  size_t noStepsP       = ceil(cuSrch->SrchSz->noSteps / (float)kernel->noSteps) * kernel->noSteps;
	  size_t nX             = noStepsP * kernel->accelLen;
	  size_t nY             = kernel->hInfos->noZ;
	  planeSize             = nX * nY * plnElsSZ;

	  infoMSG(6,6,"in-mem plane: %.2f GB - %i X %i   noStepsP: %i  noSteps: %i  nss :%i \n", planeSize*1e-9, nX, nY, noStepsP, noSteps,cuSrch->SrchSz->noSteps );
	}


	char  cufftType[1024];
	if ( kernel->flags & CU_FFT_SEP )
	{
	  // one CUFFT plan per batch
	  fffTotSize *= noBatches;
	  sprintf(cufftType, "( separate plans for each batch )");
	}
	else
	{
	  sprintf(cufftType, "( single plan for all batches )");
	}

	float  totUsed = ( kernel->kerDataSize + planeSize + ( fffTotSize + batchSize * noBatches ) * kernel->noSteps ) ;

	printf("     Processing %i steps with each of the %i batch(s)\n", kernel->noSteps, noBatches );

	printf("    -----------------------------------------------\n" );
	printf("    Kernels        use: %5.2f GiB of device memory.\n", (kernel->kerDataSize) / 1073741824.0 );
	printf("    CUFFT         uses: %5.2f GiB of device memory, %s\n", (fffTotSize*kernel->noSteps) / 1073741824.0, cufftType );
	if ( planeSize )
	{
	  printf("    In-mem plane  uses: %5.2f GiB of device memory.", (planeSize) / 1073741824.0 );

	  if ( kernel->flags & FLAG_POW_HALF )
	  {
	    printf(" ( using half precision )\n");
	  }
	  else
	  {
	    printf("\n");
	  }
	}
	printf("    Each batch    uses: %5.2f GiB of device memory.\n", (batchSize*kernel->noSteps) / 1073741824.0 );
	printf("                 Using: %5.2f GiB of %.2f [%.2f%%] of GPU memory for search.\n",  totUsed / 1073741824.0, total / 1073741824.0, totUsed / (float)total * 100.0f );
      }

      PROF // Profiling  .
      {
	NV_RANGE_POP();
      }
    }

    FOLD // Scale data sizes by number of steps  .
    {
      kernel->inpDataSize *= kernel->noSteps;
      kernel->plnDataSize *= kernel->noSteps;
      kernel->pwrDataSize *= kernel->noSteps;
      if ( !(kernel->flags & FLAG_SS_INMEM)  )
	kernel->retDataSize *= kernel->noSteps;       // In-mem search stage does not use steps
    }

    float fullCSize     = cuSrch->SrchSz->noOutpR * candSZ;               /// The full size of all candidate data

    if ( kernel->flags  & FLAG_STORE_ALL )
      fullCSize *= kernel->noHarmStages; // Store  candidates for all stages

    FOLD // DO a sanity check on flags  .
    {
      FOLD // How to handle input  .
      {
	if ( (kernel->flags & CU_INPT_FFT_CPU) && (kernel->flags & CU_NORM_GPU) )
	{
	  fprintf(stderr, "WARNING: Using CPU FFT of the input data necessitate doing the normalisation on CPU.\n");
	  kernel->flags &= ~CU_NORM_GPU;
	}
      }

      FOLD // Set the stack flags  .
      {
	for (int i = 0; i < kernel->noStacks; i++)
	{
	  cuFfdotStack* cStack  = &kernel->stacks[i];
	  cStack->flags         = kernel->flags;
	}
      }
    }

    FOLD // Batch independent device memory  .
    {
      if ( kernel->flags & FLAG_SS_INMEM  )
      {
	PROF // Profiling  .
	{
	  NV_RANGE_PUSH("in-mem alloc");
	}

	size_t noStepsP =  ceil(cuSrch->SrchSz->noSteps / (float)kernel->noSteps) * kernel->noSteps ;
	size_t nX       = noStepsP * kernel->accelLen;
	size_t nY       = kernel->hInfos->noZ;
	size_t stride;

	CUDA_SAFE_CALL(cudaMallocPitch(&cuSrch->d_planeFull,    &stride, plnElsSZ*nX, nY),   "Failed to allocate strided memory for in-memory plane.");
	CUDA_SAFE_CALL(cudaMemsetAsync(cuSrch->d_planeFull, 0, stride*nY, kernel->stacks->initStream),"Failed to initiate in-memory plane to zero");

	// Round down to units
	cuSrch->inmemStride = ceil(stride / (double)plnElsSZ); // NOTE: this will most lightly be a int

	PROF // Profiling  .
	{
	  NV_RANGE_POP();
	}
      }

    }

    FOLD // Allocate global (device independent) host memory  .
    {
      // One set of global set of "candidates" for all devices
      if ( master == NULL )
      {
	PROF // Profiling  .
	{
	  NV_RANGE_PUSH("host alloc");
	}

	if 	( kernel->cndType & CU_STR_ARR	)
	{
	  if ( cuSrch->sSpec->outData == NULL   )
	  {
	    // Have to allocate the array!

	    freeRam  = getFreeRamCU();
	    if ( fullCSize < freeRam*0.9 )
	    {
	      // Same host candidates for all devices
	      // This can use a lot of memory for long searches!
	      cuSrch->h_candidates = malloc( fullCSize );
	      memset(cuSrch->h_candidates, 0, fullCSize );
	      hostC += fullCSize;
	    }
	    else
	    {
	      fprintf(stderr, "ERROR: Not enough host memory for candidate list array. Need %.2fGiB there is %.2fGiB.\n", fullCSize / 1073741824.0, freeRam / 1073741824.0 );
	      fprintf(stderr, "       Try set -fhi to a lower value. ie: numharm*1000. ( or buy more RAM, or close Chrome ;)\n");
	      fprintf(stderr, "       Will continue trying to use a dynamic list.\n");

	      // Candidate type
	      kernel->cndType &= ~CU_TYPE_ALLL ;
	      kernel->cndType &= ~CU_SRT_ALL   ;

	      kernel->cndType |= CU_CANDFULL   ;
	      kernel->cndType |= CU_STR_LST    ;
	    }
	  }
	  else
	  {
	    // This memory has already been allocated
	    cuSrch->h_candidates = cuSrch->sSpec->outData;
	    memset(cuSrch->h_candidates, 0, fullCSize ); // NOTE: this may error if the preallocated memory int karge enough!
	  }
	}
	else if ( kernel->cndType & CU_STR_QUAD	)
	{
	  if ( cuSrch->sSpec->outData == NULL )
	  {
	    candTree* qt = new candTree;
	    cuSrch->h_candidates = qt;
	  }
	  else
	  {
	    cuSrch->h_candidates = cuSrch->sSpec->outData;
	  }
	}
	else if ( kernel->cndType & CU_STR_LST	)
	{
	  // Nothing here
	}
	else if ( kernel->cndType & CU_STR_PLN  )
	{
	  fprintf(stderr,"WARNING: The case of candidate planes has not been implemented!\n");

	  // This memory has already been allocated
	  cuSrch->h_candidates = cuSrch->sSpec->outData;
	}

	PROF // Profiling  .
	{
	  NV_RANGE_POP();
	}
      }
    }

    if ( hostC )
    {
      printf("    Input and candidates use and additional:\n");
      if ( hostC )
	printf("                        %5.2f GiB of host   memory\n", hostC / 1073741824.0 );
    }
    printf("    -----------------------------------------------\n" );

    CUDA_SAFE_CALL(cudaGetLastError(), "Failed to create memory for candidate list or input data.");

    printf("  Done\n");

    PROF // Profiling  .
    {
      NV_RANGE_POP(); // data
    }
  }

  FOLD // Create FFT plans, ( 1 - set per device )  .
  {
    PROF // Profiling  .
    {
      NV_RANGE_PUSH("FFT plans");
    }

    if ( ( kernel->flags & CU_INPT_FFT_CPU ) && master == NULL )
    {
      PROF // Profiling  .
      {
	NV_RANGE_PUSH("read_wisdom");
      }

      read_wisdom();

      PROF // Profiling  .
      {
	NV_RANGE_POP(); // read_wisdom
      }
    }

    if ( !(kernel->flags & CU_FFT_SEP) )
    {
      infoMSG(4,4,"Create FFT plans\n");

      createFFTPlans(kernel);
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP(); // FFT plans
    }
  }

  FOLD // Create texture memory from kernels  .
  {
    if ( kernel->flags & FLAG_TEX_MUL )
    {
      infoMSG(4,4,"Create texture memory\n");

      PROF // Profiling  .
      {
	NV_RANGE_PUSH("text mem");
      }

      cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);

      CUDA_SAFE_CALL(cudaGetLastError(), "Creating texture from kernel data.");

      for (int i = 0; i < kernel->noStacks; i++)           // Loop through Stacks
      {
	cuFfdotStack* cStack = &kernel->stacks[i];

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0]            = cudaAddressModeClamp;
	texDesc.addressMode[1]            = cudaAddressModeClamp;
	texDesc.filterMode                = cudaFilterModePoint;
	texDesc.readMode                  = cudaReadModeElementType;
	texDesc.normalizedCoords          = 0;

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType                   = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.desc          = channelDesc;
	resDesc.res.pitch2D.devPtr        = cStack->kernels->d_kerData;
	resDesc.res.pitch2D.width         = cStack->harmInf->width;
	resDesc.res.pitch2D.pitchInBytes  = cStack->strideCmplx * sizeof(fcomplex);
	resDesc.res.pitch2D.height        = cStack->harmInf->noZ;

	CUDA_SAFE_CALL(cudaCreateTextureObject(&cStack->kerDatTex, &resDesc, &texDesc, NULL), "Creating texture from kernel data.");

	CUDA_SAFE_CALL(cudaGetLastError(), "Creating texture from the stack of kernel data.");

	// Create the actual texture object
	for (int j = 0; j< cStack->noInStack; j++)        // Loop through planes in stack
	{
	  cuKernel* cKer = &cStack->kernels[j];

	  resDesc.res.pitch2D.devPtr        = cKer->d_kerData;
	  resDesc.res.pitch2D.height        = cKer->harmInf->noZ;
	  resDesc.res.pitch2D.width         = cKer->harmInf->width;
	  resDesc.res.pitch2D.pitchInBytes  = cStack->strideCmplx * sizeof(fcomplex);

	  CUDA_SAFE_CALL(cudaCreateTextureObject(&cKer->kerDatTex, &resDesc, &texDesc, NULL), "Creating texture from kernel data.");
	  CUDA_SAFE_CALL(cudaGetLastError(), "Creating texture from kernel data.");
	}
      }

      PROF // Profiling  .
      {
	NV_RANGE_POP();
      }
    }
  }

  FOLD // Set constant memory values  .
  {
    infoMSG(4,4,"Set constant memory values\n");

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("const mem");
    }

    setConstVals( kernel );

    setConstVals_Fam_Order( kernel );                            // Constant values for multiply

    FOLD // Set CUFFT load callback details  .
    {
      if( kernel->flags & FLAG_MUL_CB )
      {
	setStackVals( kernel );
      }
    }

    FOLD // CUFFT store callbacks  .
    {
      if ( !(kernel->flags & CU_FFT_SEP) )
      {
#if CUDA_VERSION >= 6050        // CUFFT callbacks only implemented in CUDA 6.5
	copyCUFFT_LD_CB(kernel);
#endif
      }
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP();
    }
  }

  printf("Done initializing GPU %i.\n", kernel->gInf->devid);

  std::cout.flush();

  PROF // Profiling  .
  {
    NV_RANGE_POP(); // msg
  }

  return noBatches;
}

void freeKernelGPUmem(cuFFdotBatch* kernrl)
{
  cudaFreeNull(kernrl->d_kerData);

  CUDA_SAFE_CALL(cudaGetLastError(), "Freeing device memory for kernel.\n");
}

/** Free kernel data structure  .
 *
 * @param kernel
 * @param master
 */
void freeKernel(cuFFdotBatch* kernrl)
{
  freeKernelGPUmem(kernrl);

  freeNull(kernrl->stacks);
  freeNull(kernrl->hInfos);
  freeNull(kernrl->kernels);
}

/** Initialise the pointers of the planes data structures of a batch  .
 *
 * This assumes the stack pointers have already been setup
 *
 * @param batch
 */
void setPlanePointers(cuFFdotBatch* batch)
{
  infoMSG(4,5,"setPlanePointers\n");

  for (int i = 0; i < batch->noStacks; i++)
  {
    infoMSG(4,6,"stack %i\n", i);

    // Set stack pointers
    cuFfdotStack* cStack  = &batch->stacks[i];

    for (int j = 0; j < cStack->noInStack; j++)
    {
      infoMSG(4,7,"plane %i\n", i);

      cuFFdot* cPlane           = &cStack->planes[j];

      if ( batch->flags & FLAG_DOUBLE )
	cPlane->d_planeMult       = &((double2*)cStack->d_planeMult)[ cStack->startZ[j] * batch->noSteps * cStack->strideCmplx ];
      else
	cPlane->d_planeMult       = &((float2*)cStack->d_planeMult) [ cStack->startZ[j] * batch->noSteps * cStack->strideCmplx ];

      if (cStack->d_planePowr)
      {
	if ( batch->flags & FLAG_POW_HALF )
	{
#if CUDA_VERSION >= 7050
	  cPlane->d_planePowr   = &((half*)         cStack->d_planePowr)[ cStack->startZ[j] * batch->noSteps * cStack->stridePower ];
#else
	  fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
	  exit(EXIT_FAILURE);
#endif
	}
	else
	{
	  if ( batch->flags & FLAG_CUFFT_CB_POW )
	    cPlane->d_planePowr = &((float*)      cStack->d_planePowr)[ cStack->startZ[j] * batch->noSteps * cStack->stridePower ];
	  else
	    cPlane->d_planePowr = &((fcomplexcu*) cStack->d_planePowr)[ cStack->startZ[j] * batch->noSteps * cStack->stridePower ];
	}
      }

      cPlane->d_iData           = &cStack->d_iData[cStack->strideCmplx*j*batch->noSteps];
      cPlane->harmInf           = &cStack->harmInf[j];
      cPlane->kernel            = &cStack->kernels[j];
    }
  }
}

/** Initialise the pointers of the stacks data structures of a batch  .
 *
 * This assumes the various memory blocks of the batch have been created
 *
 * @param batch
 */
void setStkPointers(cuFFdotBatch* batch)
{
  infoMSG(4,5,"setStkPointers\n");

  size_t cmplStart  = 0;
  size_t pwrStart   = 0;
  size_t idSiz      = 0;            /// The size in bytes of input data for one stack
  int harm          = 0;            /// The harmonic index of the first plane the the stack

  for (int i = 0; i < batch->noStacks; i++) // Set the various pointers of the stacks  .
  {
    infoMSG(4,6,"stack %i\n", i);

    cuFfdotStack* cStack  = &batch->stacks[i];

    cStack->d_iData       = &batch->d_iData[idSiz];
    cStack->h_iData       = &batch->h_iData[idSiz];
    cStack->planes        = &batch->planes[harm];
    cStack->kernels       = &batch->kernels[harm];
    if ( batch->h_iBuffer )
      cStack->h_iBuffer   = &batch->h_iBuffer[idSiz];

    if ( batch->flags & FLAG_DOUBLE )
      cStack->d_planeMult   = &((double2*)batch->d_planeMult)[cmplStart];
    else
      cStack->d_planeMult   = &((float2*)batch->d_planeMult) [cmplStart];

    if (batch->d_planePowr)
    {
      if ( batch->flags & FLAG_POW_HALF )
      {
#if CUDA_VERSION >= 7050
	cStack->d_planePowr     = &((half*)       batch->d_planePowr)[ pwrStart ];
#else
	fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
	exit(EXIT_FAILURE);
#endif
      }
      else
      {
	if ( batch->flags & FLAG_CUFFT_CB_POW )
	  cStack->d_planePowr   = &((float*)      batch->d_planePowr)[ pwrStart ];
	else
	  cStack->d_planePowr   = &((fcomplexcu*) batch->d_planePowr)[ pwrStart ];
      }
    }

    // Increment the various values used for offset
    harm                 += cStack->noInStack;
    idSiz                += batch->noSteps  * cStack->strideCmplx * cStack->noInStack;
    cmplStart            += cStack->height  * cStack->strideCmplx * batch->noSteps ;
    pwrStart             += cStack->height  * cStack->stridePower * batch->noSteps ;
  }
}

/** Initialise the pointers of the stacks and planes data structures of a batch  .
 *
 * This assumes the various memory blocks of the batch have been created
 *
 * @param batch
 */
void setBatchPointers(cuFFdotBatch* batch)
{
  // First initialise the various pointers of the stacks
  setStkPointers(batch);

  // Now initialise the various pointers of the planes
  setPlanePointers(batch);
}

void setKernelPointers(cuFFdotBatch* batch)
{
  infoMSG(4,4,"Set the sizes values of the harmonics and kernels and pointers to kernel data\n");

  size_t kerSiz = 0;
  void *d_kerData;

  for (int i = 0; i < batch->noStacks; i++)
  {
    cuFfdotStack* cStack		= &batch->stacks[i];

    // Set the stack pointer
    if ( batch->flags & FLAG_DOUBLE )
      d_kerData		= &((double2*)batch->d_kerData)[kerSiz];
    else
      d_kerData		= &((fcomplexcu*)batch->d_kerData)[kerSiz];

    // Set the indevidaul kernel ifrtmation paramiters
    for (int j = 0; j < cStack->noInStack; j++)
    {
      // Point the plane kernel data to the correct position in the "main" kernel
      cStack->kernels[j].kreOff	= cu_index_from_z<double>(cStack->harmInf[j].zStart, cStack->harmInf->zStart, batch->cuSrch->sSpec->zRes);
      cStack->kernels[j].stride	= cStack->strideCmplx;

      if ( batch->flags & FLAG_DOUBLE )
	cStack->kernels[j].d_kerData	= &((double2*)d_kerData)[cStack->strideCmplx*cStack->kernels[j].kreOff];
      else
	cStack->kernels[j].d_kerData	= &((fcomplexcu*)d_kerData)[cStack->strideCmplx*cStack->kernels[j].kreOff];

      cStack->kernels[j].harmInf	= &cStack->harmInf[j];
    }
    kerSiz				+= cStack->strideCmplx * cStack->kerHeigth;
  }
}

/** Initialise a batch using details from the device kernel  .
 *
 * @param batch
 * @param kernel
 * @param no
 * @param of
 * @return
 */
int initBatch(cuFFdotBatch* batch, cuFFdotBatch* kernel, int no, int of)
{
  PROF // Profiling  .
  {
    char msg[1024];
    sprintf(msg,"%i of %i", no, of);
    NV_RANGE_PUSH(msg); // # of #
  }

  char strBuff[1024];
  size_t free, total;

  infoMSG(3,3,"%s device %i, batch %i of %i \n",__FUNCTION__, kernel->gInf->devid, no+1, of);

  FOLD // See if we can use the cuda device  .
  {
    infoMSG(4,4,"Device %i\n", kernel->gInf->devid);

    setDevice(kernel->gInf->devid) ;

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

  FOLD // Copy details from kernel and allocate stacks .
  {
    infoMSG(4,4,"Copy kernel\n");

    // Copy the basic batch parameters from the kernel
    memcpy(batch, kernel, sizeof(cuFFdotBatch));

    batch->srchMaster   = 0;
    batch->isKernel     = 0;

    infoMSG(4,4,"Create and copy stacks\n");

    // Allocate memory for the stacks
    batch->stacks = (cuFfdotStack*) malloc(batch->noStacks * sizeof(cuFfdotStack));

    // Copy the actual stacks
    memcpy(batch->stacks, kernel->stacks, batch->noStacks  * sizeof(cuFfdotStack));
  }

  FOLD // Set the batch specific flags  .
  {
    infoMSG(4,4,"Set flags\n");

    FOLD // Multiplication flags  .
    {
      for ( int i = 0; i < batch->noStacks; i++ )	// Multiplication is generally stack specific so loop through stacks  .
      {
	cuFfdotStack* cStack  = &batch->stacks[i];

	FOLD // Multiplication kernel  .
	{
	  if ( !(cStack->flags & FLAG_MUL_ALL) )	// Default to multiplication  .
	  {
	    int64_t mFlag = 0;

	    // In my testing I found multiplying each plane separately works fastest so it is the "default"
	    int noInp =  cStack->noInStack * kernel->noSteps ;

	    if ( batch->gInf->capability > 3.0 )
	    {
	      // Lots of registers per thread so 2.1 is good
	      mFlag |= FLAG_MUL_21;
	    }
	    else
	    {
	      // We require fewer registers per thread, so use Multiplication kernel 2.1
	      if ( noInp <= 20 )
	      {
		// TODO: Check small, looks like some times 22 may be faster.
		mFlag |= FLAG_MUL_21;
	      }
	      else
	      {
		if ( kernel->noSteps <= 4 )
		{
		  // very few steps so 2.2 not always the best option
		  if ( kernel->hInfos->zmax > 100 )
		  {
		    // This only really holds for 16 harmonics summed with 3 or 4 steps
		    // In my testing it is generally true for zmax greater than 100
		    mFlag |= FLAG_MUL_23;
		  }
		  else
		  {
		    // Here 22 is usually better
		    mFlag |= FLAG_MUL_22;
		  }
		}
		else
		{
		  // Enough steps to justify Multiplication kernel 2.1
		  mFlag |= FLAG_MUL_22;
		}
	      }
	    }

	    // Set the stack and batch flag
	    cStack->flags |= mFlag;
	    batch->flags  |= mFlag;
	  }
	}

	FOLD // Slices  .
	{
	  if ( cStack->mulSlices <= 0 )
	  {
	    // Multiplication slices not specified so use logical values

	    if      ( cStack->width <= 256  )
	    {
	      cStack->mulSlices = 10;
	    }
	    else if ( cStack->width <= 512  )
	    {
	      cStack->mulSlices = 8;
	    }
	    else if ( cStack->width <= 1024 )
	    {
	      cStack->mulSlices = 6;
	    }
	    else if ( cStack->width <= 2048 )
	    {
	      cStack->mulSlices = 4;
	    }
	    else if ( cStack->width <= 4096 )
	    {
	      cStack->mulSlices = 2;
	    }
	    else
	    {
	      // TODO: check with a card with many SM's
	      cStack->mulSlices = 1;
	    }
	  }

	  // Clamp to size of kernel (ie height of the largest plane)
	  MINN(cStack->mulSlices, cStack->kerHeigth/2.0);
	  MAXX(cStack->mulSlices, 1);

	  infoMSG(5,5,"stack %i  mulSlices %2i \n",i, cStack->mulSlices);

	  if ( i == 0 && batch->mulSlices == 0 )
	  {
	    batch->mulSlices = cStack->mulSlices;
	  }
	}

	FOLD // Chunk size  .
	{
	  if ( cStack->mulChunk <= 0 )
	  {
	    cStack->mulChunk = 4;
	  }

	  // Clamp to size of kernel (ie height of the largest plane)
	  cStack->mulChunk = MIN( cStack->mulChunk, ceil(cStack->kerHeigth/2.0) );

	  if ( i == 0 )
	    batch->mulChunk = cStack->mulChunk;
	}
      }

    }

    FOLD // Sum and search flags  .
    {
      if ( !(batch->flags & FLAG_SS_ALL ) )   // Default to multiplication  .
      {
	batch->flags |= FLAG_SS_10;
      }

      if ( batch->ssChunk <= 0 )
      {
	if ( batch->flags & FLAG_SS_INMEM )
	{
	  // With the inmem search only only one "step" is done at a time
	  // Tested with a 750 Ti, 770 and 970, 6 was the best chunk size on the Maxwell card
	  // The Kepler cards could go to 8
	  // This was tested by Chris L - 12/07/2016
	  batch->ssChunk = 6;
	}
	else
	{
	  // Using standard sum and search kernel

	  if ( batch->gInf->capability <= 3.2 )	// Kepler  .
	  {
	    // Kepler cards have fewer registers so this limit chunk size
	    // I did some testing and found a roughly liner relationship between
	    // Optimal chink size and number of steps with the standard search kernel on Kepler cards.
	    // This was tested on a GTX 770 by Chris L - 12/07/2016
	    batch->ssChunk = round(9.5 - 0.75*batch->noSteps );
	  }
	  else					// Maxwell  .
	  {
	    // More register

	    // Well this looks crazy, but from my testing this gives a pretty good approximation to the optimal cunk size for a range of
	    // steps, harmonics and widths.  Importantly this works well for many steps (~8) and 16 harmonics.
	    // Diagrams and numbers can be produced on request if you don't believe me ;-)
	    // This was tested on a GTX 970 by Chris L - 12/07/2016
	    batch->ssChunk = round(-0.256 * batch->noSteps*batch->noSteps + 2.4 * batch->noSteps + 2.8 );
	  }
	}

	// Clamp to valid bounds
	batch->ssChunk = MAX(MIN(floor(batch->ssChunk), 9),1);
      }
    }
  }

  FOLD // Create FFT plans  .
  {
    if ( kernel->flags & CU_FFT_SEP )
    {
      infoMSG(4,4,"Create FFT plans\n");

      createFFTPlans(batch);

      FOLD // Set CUFFT callbacks
      {
#if CUDA_VERSION >= 6050        // CUFFT callbacks only implemented in CUDA 6.5
	copyCUFFT_LD_CB(batch);
#endif
      }
    }
  }

  FOLD // Allocate all device and host memory for the batch  .
  {
    infoMSG(4,4,"Allocate memory for the batch\n");

    FOLD // Allocate host input memory  .
    {
      PROF // Profiling  .
      {
	NV_RANGE_PUSH("Host");
      }

      // Allocate page-locked host memory for input data  .
      infoMSG(5,5,"Allocate page-locked for input data.  %p  %i Bytes \n", &batch->h_iData, batch->inpDataSize);

      CUDA_SAFE_CALL(cudaMallocHost(&batch->h_iData, batch->inpDataSize ), "Failed to create page-locked host memory plane input data." );

      // Allocate buffer for CPU to work on input data
      batch->h_iBuffer = (fcomplexcu*)malloc(batch->inpDataSize);

      if ( !(batch->flags & CU_NORM_GPU) )
      {
	infoMSG(5,5,"Allocate host memory for normalisation powers.\n");

	// Allocate CPU memory for normalisation
	batch->h_normPowers = (float*) malloc(batch->hInfos->width * sizeof(float));
      }

      PROF // Profiling  .
      {
	NV_RANGE_POP(); // Host
      }
    }

    FOLD // Allocate R value lists  .
    {
      infoMSG(5,5,"Allocate R value lists.\n");

      batch->noRArryas        = 7; // This is just a convenient value

      createRvals(batch, &batch->rArr1, &batch->rArraysPlane);
      batch->rAraays = &batch->rArraysPlane;

      if ( batch->flags & FLAG_SEPRVAL )
	createRvals(batch, &batch->rArr2, &batch->rArraysSrch);
    }

    FOLD // Allocate device Memory for Planes, Stacks & Input data (steps)  .
    {
      PROF // Profiling  .
      {
	NV_RANGE_PUSH("device");
      }

      size_t req = batch->inpDataSize + batch->plnDataSize + batch->pwrDataSize;

      if ( req > free ) // Not enough memory =(
      {
	printf("Not enough GPU memory to create any more batches.\n");
	return (0);
      }
      else
      {
	if ( batch->inpDataSize )
	{
	  infoMSG(5,5,"Allocate device memory for input.\n");

	  CUDA_SAFE_CALL(cudaMalloc((void** )&batch->d_iData,       batch->inpDataSize ), "Failed to allocate device memory for batch input.");
	  free -= batch->inpDataSize;
	}

	if ( batch->plnDataSize )
	{
	  infoMSG(5,5,"Allocate device memory for complex plane.\n");

	  CUDA_SAFE_CALL(cudaMalloc((void** )&batch->d_planeMult,   batch->plnDataSize ), "Failed to allocate device memory for batch complex plane.");
	  free -= batch->plnDataSize;
	}

	if ( batch->pwrDataSize )
	{
	  infoMSG(5,5,"Allocate device memory for powers plane.\n");

	  CUDA_SAFE_CALL(cudaMalloc((void** )&batch->d_planePowr,   batch->pwrDataSize ), "Failed to allocate device memory for batch powers plane.");
	  free -= batch->pwrDataSize;
	}

      }

      PROF // Profiling  .
      {
	NV_RANGE_POP(); // device
      }
    }

    FOLD // Allocate device & page-locked host memory for return data  .
    {
      PROF // Profiling  .
      {
	NV_RANGE_PUSH("Host");
      }

      FOLD // Allocate device memory  .
      {
	if ( kernel->retDataSize && !(kernel->retType & CU_STR_PLN) )
	{
	  if ( batch->retDataSize > free )
	  {
	    // Not enough memory =(
	    printf("Not enough GPU memory for return data.\n");
	    return (0);
	  }
	  else
	  {
	    infoMSG(5,5,"Allocate device memory for return values.\n");

	    CUDA_SAFE_CALL(cudaMalloc((void** ) &batch->d_outData1, batch->retDataSize ), "Failed to allocate device memory for return values.");
	    free -= batch->retDataSize;

	    if ( batch->flags & FLAG_SS_INMEM )
	    {
	      if ( batch->retDataSize > batch->plnDataSize )
	      {
		infoMSG(4,5,"Complex plane is smaller than return data -> FLAG_SEPSRCH\n");

		batch->flags |= FLAG_SEPSRCH;
	      }

	      if ( batch->flags & FLAG_SEPSRCH )
	      {
		infoMSG(5,5,"Allocate device memory for second return values.\n");

		// Create a separate output space
		CUDA_SAFE_CALL(cudaMalloc((void** ) &batch->d_outData2, batch->retDataSize ), "Failed to allocate device memory for return values.");
		free -= batch->retDataSize;
	      }
	      else
	      {
		batch->d_outData2 = batch->d_planeMult;
	      }
	    }
	  }
	}
      }

      FOLD // Allocate page-locked host memory to copy the candidates back to  .
      {
	if ( kernel->retDataSize )
	{
	  infoMSG(5,5,"Allocate page-locked for candidates.\n");

	  CUDA_SAFE_CALL(cudaMallocHost(&batch->h_outData1, kernel->retDataSize), "Failed to create page-locked host memory plane for return data.");
	  memset(batch->h_outData1, 0, kernel->retDataSize );

	  if ( kernel->flags & FLAG_SS_INMEM )
	  {
	    infoMSG(5,5,"Allocate page-locked for candidates.\n");

	    CUDA_SAFE_CALL(cudaMallocHost(&batch->h_outData2, kernel->retDataSize), "Failed to create page-locked host memory plane for return data.");
	    memset(batch->h_outData2, 0, kernel->retDataSize );
	  }
	}
      }

      PROF // Profiling  .
      {
	NV_RANGE_POP(); // Host
      }
    }

    FOLD // Create the planes structures  .
    {
      if ( batch->noGenHarms* sizeof(cuFFdot) > getFreeRamCU() )
      {
	fprintf(stderr, "ERROR: Not enough host memory for search.\n");
	return 0;
      }
      else
      {
	batch->planes = (cuFFdot*) malloc(batch->noGenHarms* sizeof(cuFFdot));
	memset(batch->planes, 0, batch->noGenHarms* sizeof(cuFFdot));
      }
    }

    PROF // Create timing arrays  .
    {
      if ( batch->flags & FLAG_PROF )
      {
	if ( kernel->compTime )
	{
	  // The float array was already created and populated with values for kernel creation
	  batch->compTime = kernel->compTime;
	  kernel->compTime = NULL;
	}
	else
	{
	  int sz = batch->noStacks*sizeof(float)*(TIME_CMP_END+1) ;

	  batch->compTime       = (float*)malloc(sz);
	  memset(batch->compTime,    0, sz);
	}
      }
    }
  }

  FOLD // Setup the pointers for the stacks and planes of this batch  .
  {
    infoMSG(4,4,"Setup the pointers\n");

    setBatchPointers(batch);
  }

  FOLD // Set up the batch streams and events  .
  {
    infoMSG(4,4,"Set up the batch streams and events\n");

    FOLD // Create Streams  .
    {
      FOLD // Input streams  .
      {
	// Batch input ( Always needed, for copying input to device )
	CUDA_SAFE_CALL(cudaStreamCreate(&batch->inpStream),"Creating input stream for batch.");
	sprintf(strBuff,"%i.%i.1.0 Batch Input", batch->gInf->devid, no);
	NV_NAME_STREAM(batch->inpStream, strBuff);
	//printf("cudaStreamCreate: %s\n", strBuff);

	// Stack input
	if ( (batch->flags & CU_NORM_GPU)  )
	{
	  for (int i = 0; i < batch->noStacks; i++)
	  {
	    cuFfdotStack* cStack  = &batch->stacks[i];

	    CUDA_SAFE_CALL(cudaStreamCreate(&cStack->inptStream), "Creating input data multStream for stack");
	    sprintf(strBuff,"%i.%i.1.%i Stack Input", batch->gInf->devid, no, i);
	    NV_NAME_STREAM(cStack->inptStream, strBuff);
	    //printf("cudaStreamCreate: %s\n", strBuff);
	  }
	}
      }

      FOLD // Input FFT streams  .
      {
	for (int i = 0; i < kernel->noStacks; i++)
	{
	  cuFfdotStack* cStack = &batch->stacks[i];

	  if ( kernel->flags & CU_FFT_SEP )       // Create stream  .
	  {
	    if ( !(kernel->flags & CU_INPT_FFT_CPU) )
	    {
	      CUDA_SAFE_CALL(cudaStreamCreate(&cStack->fftIStream),"Creating CUDA stream for input fft's");

	      //              if ( (batch->flags & CU_NORM_GPU)  )
	      //              {
	      //                cStack->fftIStream = cStack->inptStream;
	      //              }
	      //              else
	      {
		sprintf(strBuff,"%i.%i.2.%i Inp FFT", batch->gInf->devid, no, i);
		NV_NAME_STREAM(cStack->fftIStream, strBuff);
		//printf("cudaStreamCreate: %s\n", strBuff);
	      }
	    }
	  }
	  else                                    // Copy stream of the kernel  .
	  {
	    cuFfdotStack* kStack  = &kernel->stacks[i];
	    cStack->fftIStream    = kStack->fftIStream;
	  }
	}
      }

      FOLD // Multiply streams  .
      {
	if      ( batch->flags & FLAG_MUL_BATCH )
	{
	  CUDA_SAFE_CALL(cudaStreamCreate(&batch->multStream),"Creating multiplication stream for batch.");
	  sprintf(strBuff,"%i.%i.3.0 Batch Multiply", batch->gInf->devid, no);
	  NV_NAME_STREAM(batch->multStream, strBuff);
	  //printf("cudaStreamCreate: %s\n", strBuff);

	}

	if ( (batch->flags & FLAG_MUL_STK) || (batch->flags & FLAG_MUL_PLN)  )
	{
	  for (int i = 0; i< batch->noStacks; i++)
	  {
	    cuFfdotStack* cStack  = &batch->stacks[i];

	    //            if ( !(kernel->flags & CU_INPT_FFT_CPU) &&  (batch->flags & CU_FFT_SEP) )
	    //            {
	    //              cStack->multStream = cStack->fftIStream;
	    //              sprintf(strBuff,"%i.%i.3.%i Stack", batch->gInf->devid, no, i);
	    //              NV_NAME_STREAM(cStack->multStream, strBuff);
	    //            }
	    //            else
	    {
	      CUDA_SAFE_CALL(cudaStreamCreate(&cStack->multStream), "Creating multStream for stack");
	      sprintf(strBuff,"%i.%i.3.%i Stack Multiply", batch->gInf->devid, no, i);
	      NV_NAME_STREAM(cStack->multStream, strBuff);
	      //printf("cudaStreamCreate: %s\n", strBuff);
	    }
	  }
	}
      }

      FOLD // Inverse FFT streams  .
      {
	for (int i = 0; i < kernel->noStacks; i++)
	{
	  cuFfdotStack* cStack = &batch->stacks[i];

	  if ( batch->flags & CU_FFT_SEP )           // Create stream
	  {
	    //            if ( (batch->flags & FLAG_MUL_STK) || (batch->flags & FLAG_MUL_PLN)  )
	    //            {
	    //              cStack->fftPStream = cStack->multStream;
	    //            }
	    //            else
	    {
	      CUDA_SAFE_CALL(cudaStreamCreate(&cStack->fftPStream),"Creating CUDA stream for fft's");
	      sprintf(strBuff,"%i.%i.4.%i Stack iFFT", batch->gInf->devid, no, i);
	      NV_NAME_STREAM(cStack->fftPStream, strBuff);
	      //printf("cudaStreamCreate: %s\n", strBuff);
	    }
	  }
	  else                                        // Copy stream of the kernel
	  {
	    cuFfdotStack* kStack  = &kernel->stacks[i];
	    cStack->fftPStream    = kStack->fftPStream;
	  }
	}
      }

      FOLD // Search stream  .
      {
	CUDA_SAFE_CALL(cudaStreamCreate(&batch->srchStream), "Creating strmSearch for batch.");
	sprintf(strBuff,"%i.%i.5.0 Batch Search", batch->gInf->devid, no);
	NV_NAME_STREAM(batch->srchStream, strBuff);
	//printf("cudaStreamCreate: %s\n", strBuff);
      }

      FOLD // Result stream  .
      {
	// Batch output ( Always needed, for copying results from device )
	CUDA_SAFE_CALL(cudaStreamCreate(&batch->resStream), "Creating strmSearch for batch.");
	sprintf(strBuff,"%i.%i.6.0 Batch result", batch->gInf->devid, no);
	NV_NAME_STREAM(batch->resStream, strBuff);
	//printf("cudaStreamCreate: %s\n", strBuff);
      }

      CUDA_SAFE_CALL(cudaGetLastError(), "Creating streams for the batch.");
    }

    FOLD // Create Events  .
    {
      FOLD // Create batch events  .
      {
	if ( batch->flags & FLAG_PROF )
	{
	  CUDA_SAFE_CALL(cudaEventCreate(&batch->iDataCpyComp), "Creating input event iDataCpyComp.");
	  CUDA_SAFE_CALL(cudaEventCreate(&batch->candCpyComp),  "Creating input event candCpyComp.");
	  CUDA_SAFE_CALL(cudaEventCreate(&batch->normComp),     "Creating input event normComp.");
	  CUDA_SAFE_CALL(cudaEventCreate(&batch->multComp),     "Creating input event multComp.");
	  CUDA_SAFE_CALL(cudaEventCreate(&batch->searchComp),   "Creating input event searchComp.");
	  CUDA_SAFE_CALL(cudaEventCreate(&batch->processComp),  "Creating input event processComp.");

	  CUDA_SAFE_CALL(cudaEventCreate(&batch->iDataCpyInit), "Creating input event iDataCpyInit.");
	  CUDA_SAFE_CALL(cudaEventCreate(&batch->candCpyInit),  "Creating input event candCpyInit.");
	  CUDA_SAFE_CALL(cudaEventCreate(&batch->multInit),     "Creating input event multInit.");
	  CUDA_SAFE_CALL(cudaEventCreate(&batch->searchInit),   "Creating input event searchInit.");
	}
	else
	{
	  CUDA_SAFE_CALL(cudaEventCreateWithFlags(&batch->iDataCpyComp,   cudaEventDisableTiming ), "Creating input event iDataCpyComp.");
	  CUDA_SAFE_CALL(cudaEventCreateWithFlags(&batch->candCpyComp,    cudaEventDisableTiming ), "Creating input event candCpyComp.");
	  CUDA_SAFE_CALL(cudaEventCreateWithFlags(&batch->normComp,       cudaEventDisableTiming ), "Creating input event normComp.");
	  CUDA_SAFE_CALL(cudaEventCreateWithFlags(&batch->multComp,       cudaEventDisableTiming ), "Creating input event searchComp.");
	  CUDA_SAFE_CALL(cudaEventCreateWithFlags(&batch->searchComp,     cudaEventDisableTiming ), "Creating input event searchComp.");
	  CUDA_SAFE_CALL(cudaEventCreateWithFlags(&batch->processComp,    cudaEventDisableTiming ), "Creating input event processComp.");
	}
      }

      FOLD // Create stack events  .
      {
	for (int i = 0; i< batch->noStacks; i++)
	{
	  cuFfdotStack* cStack  = &batch->stacks[i];

	  if ( batch->flags & FLAG_PROF )
	  {
	    // in  events (with timing)
	    CUDA_SAFE_CALL(cudaEventCreate(&cStack->normInit),    	"Creating input normalisation event");
	    CUDA_SAFE_CALL(cudaEventCreate(&cStack->inpFFTinit),  	"Creating input FFT initialisation event");
	    CUDA_SAFE_CALL(cudaEventCreate(&cStack->multInit),    	"Creating multiplication initialisation event");
	    CUDA_SAFE_CALL(cudaEventCreate(&cStack->ifftInit), 	  	"Creating inverse FFT initialisation event");
	    CUDA_SAFE_CALL(cudaEventCreate(&cStack->ifftMemInit), 	"Creating inverse FFT copy initialisation event");

	    // out events (with timing)
	    CUDA_SAFE_CALL(cudaEventCreate(&cStack->normComp),		"Creating input normalisation event");
	    CUDA_SAFE_CALL(cudaEventCreate(&cStack->inpFFTinitComp),	"Creating input data preparation complete event");
	    CUDA_SAFE_CALL(cudaEventCreate(&cStack->multComp), 		"Creating multiplication complete event");
	    CUDA_SAFE_CALL(cudaEventCreate(&cStack->ifftComp),    	"Creating IFFT complete event");
	    CUDA_SAFE_CALL(cudaEventCreate(&cStack->ifftMemComp), 	"Creating IFFT memory copy complete event");
	  }
	  else
	  {
	    // out events (without timing)
	    CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->normComp,    	cudaEventDisableTiming), "Creating input data preparation complete event");
	    CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->inpFFTinitComp,    cudaEventDisableTiming), "Creating input data preparation complete event");
	    CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->multComp,    	cudaEventDisableTiming), "Creating multiplication complete event");
	    CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->ifftComp,    	cudaEventDisableTiming), "Creating IFFT complete event");
	    CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->ifftMemComp, 	cudaEventDisableTiming), "Creating IFFT memory copy complete event");
	  }
	}
      }

      CUDA_SAFE_CALL(cudaGetLastError(), "Creating events for the batch.");
    }

    //CUDA_SAFE_CALL(cudaGetLastError(), "Creating streams and events for the batch.");
  }

  FOLD // Create textures for the f-∂f planes  .
  {
    if ( (batch->flags & FLAG_TEX_INTERP) && !( (batch->flags & FLAG_CUFFT_CB_POW) && (batch->flags & FLAG_SAS_TEX) ) )
    {
      fprintf(stderr, "ERROR: Cannot use texture memory interpolation without CUFFT callback to write powers. NOT using texture memory interpolation\n");
      batch->flags &= ~FLAG_TEX_INTERP;
    }

    if ( batch->flags & FLAG_SAS_TEX )
    {
      infoMSG(4,4,"Create textures\n");

      cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

      struct cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.addressMode[0]    = cudaAddressModeClamp;
      texDesc.addressMode[1]    = cudaAddressModeClamp;
      texDesc.readMode          = cudaReadModeElementType;
      texDesc.normalizedCoords  = 0;

      if ( batch->flags & FLAG_TEX_INTERP )
      {
	texDesc.filterMode        = cudaFilterModeLinear;   /// Liner interpolation
      }
      else
      {
	texDesc.filterMode        = cudaFilterModePoint;
      }

      for (int i = 0; i< batch->noStacks; i++)
      {
	cuFfdotStack* cStack = &batch->stacks[i];

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType           = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.desc  = channelDesc;

	for (int j = 0; j< cStack->noInStack; j++)
	{
	  cuFFdot* cPlane = &cStack->planes[j];

	  if ( batch->flags & FLAG_CUFFT_CB_POW ) // float input
	  {
	    if      ( batch->flags & FLAG_ITLV_ROW )
	    {
	      resDesc.res.pitch2D.height          = cPlane->harmInf->noZ;
	      resDesc.res.pitch2D.width           = cPlane->harmInf->width * batch->noSteps;
	      resDesc.res.pitch2D.pitchInBytes    = cStack->harmInf->width * batch->noSteps * sizeof(float);
	      resDesc.res.pitch2D.devPtr          = cPlane->d_planePowr;
	    }
#ifdef WITH_ITLV_PLN
	    else
	    {
	      resDesc.res.pitch2D.height          = cPlane->harmInf->noZ * batch->noSteps ;
	      resDesc.res.pitch2D.width           = cPlane->harmInf->width;
	      resDesc.res.pitch2D.pitchInBytes    = cStack->harmInf->width * sizeof(float);
	      resDesc.res.pitch2D.devPtr          = cPlane->d_planePowr;
	    }
#else
	    else
	    {
	      fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
	      exit(EXIT_FAILURE);
	    }
#endif
	  }
	  else // Implies complex numbers
	  {
	    if      ( batch->flags & FLAG_ITLV_ROW )
	    {
	      resDesc.res.pitch2D.height          = cPlane->harmInf->noZ;
	      resDesc.res.pitch2D.width           = cPlane->harmInf->width * batch->noSteps * 2;
	      resDesc.res.pitch2D.pitchInBytes    = cStack->harmInf->width * batch->noSteps * 2 * sizeof(float);
	      resDesc.res.pitch2D.devPtr          = cPlane->d_planePowr;
	    }
#ifdef WITH_ITLV_PLN
	    else
	    {
	      resDesc.res.pitch2D.height          = cPlane->harmInf->noZ * batch->noSteps ;
	      resDesc.res.pitch2D.width           = cPlane->harmInf->width * 2;
	      resDesc.res.pitch2D.pitchInBytes    = cStack->harmInf->width * 2 * sizeof(float);
	      resDesc.res.pitch2D.devPtr          = cPlane->d_planePowr;
	    }
#else
	    else
	    {
	      fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
	      exit(EXIT_FAILURE);
	    }
#endif
	  }

	  CUDA_SAFE_CALL(cudaCreateTextureObject(&cPlane->datTex, &resDesc, &texDesc, NULL), "Creating texture from the plane data.");
	}
      }
      CUDA_SAFE_CALL(cudaGetLastError(), "Creating textures from the plane data.");
    }
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP(); // # of #
  }

  return (batch->noSteps);
}

/** Free batch data structure  .
 *
 * @param batch
 */
void freeBatchGPUmem(cuFFdotBatch* batch)
{
  CUDA_SAFE_CALL(cudaGetLastError(), "Entering freeBatchGPUmem.");

  setDevice(batch->gInf->devid) ;

  FOLD // Free host memory
  {
    infoMSG(2,2,"Free host memory\n");

    freeNull(batch->h_normPowers);
  }

  FOLD // Free pinned memory
  {
    infoMSG(2,2,"Free pinned memory\n");

    cudaFreeHostNull(batch->h_iData);
    freeNull(batch->h_iBuffer);
    cudaFreeHostNull(batch->h_outData1);
  }

  FOLD // Free device memory
  {
    infoMSG(2,2,"Free device memory\n");

    FOLD // Free the output memory  .
    {
      if ( batch->d_outData1 == batch->d_planeMult )
      {
	// d_outData1 is re using d_planeMult so don't free
	batch->d_outData1 = NULL;
      }
      else if ( batch->d_outData2 == batch->d_planeMult )
      {
	// d_outData2 is re using d_planeMult so don't free
	batch->d_outData2 = NULL;
      }

      if ( batch->d_outData1 == batch->d_outData2 )
      {
	// They are the same so only free one
	cudaFreeNull(batch->d_outData1);
	batch->d_outData2 = NULL;
      }
      else
      {
	// Using separate output so free both
	cudaFreeNull(batch->d_outData1);
	cudaFreeNull(batch->d_outData2);
      }
    }

    // Free the input and planes
    cudaFreeNull(batch->d_iData);
    cudaFreeNull(batch->d_planeMult );
    cudaFreeNull(batch->d_planePowr );

    // Free the rval arrays used during generation and search stages
    freeRvals(batch, &batch->rArr1, &batch->rArraysPlane);
    if ( batch->flags & FLAG_SEPRVAL )
      freeRvals(batch, &batch->rArr2, &batch->rArraysSrch);
  }

  FOLD // Free textures for the f-∂f planes  .
  {
    if ( batch->flags & FLAG_SAS_TEX )
    {
      infoMSG(2,2,"Free textures\n");

      for (int i = 0; i < batch->noStacks; i++)
      {
	cuFfdotStack* cStack = &batch->stacks[i];

	for (int j = 0; j< cStack->noInStack; j++)
	{
	  cuFFdot* cPlane = &cStack->planes[j];

	  if ( cPlane->datTex )
	  {
	    CUDA_SAFE_CALL(cudaDestroyTextureObject(cPlane->datTex), "Creating texture from the plane data.");
	    cPlane->datTex = (fCplxTex)0;
	  }
	}
      }
      CUDA_SAFE_CALL(cudaGetLastError(), "Creating textures from the plane data.");
    }
  }

  CUDA_SAFE_CALL(cudaGetLastError(), "Exiting freeBatchGPUmem.");
}

/** Free batch data structure  .
 *
 * @param batch
 */
void freeBatch(cuFFdotBatch* batch)
{
  freeBatchGPUmem(batch);

  FOLD // Free host memory
  {
    freeNull(batch->stacks);
    freeNull(batch->planes);

    PROF // Profiling
    {
      if ( batch->flags & FLAG_PROF )
      {
	freeNull(batch->compTime);
      }
    }
  }

}

/** Initiate a optimisation plane
 * If oPln has not been pre initialised and is NULL it will create a new data structure.
 * If oPln has been pre initialised the device ID and Idx are used!
 *
 */
cuOptCand* initOptCand(cuSearch* sSrch, cuOptCand* oPln = NULL, int devLstId = 0 )
{
  searchSpecs* sSpec = sSrch->sSpec;

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

    FOLD // Determin the largest zMax  .
    {
      zMax	= MAX(sSpec->zMax+50, sSpec->zMax*2);
      zMax	= MAX(zMax, 60 * sSrch->noSrchHarms );
      zMax	= MAX(zMax, sSpec->zMax * 34 + 50 );  		// TODO: This may be a bit high!
    }

    FOLD // Determine max plane size  .
    {
      for ( int i=0; i < sSpec->noHarmStages; i++ )
      {
	MAXX(maxWidth, sSpec->optPlnSiz[i] );
      }
      for ( int i=0; i < NO_OPT_LEVS; i++ )
      {
	MAXX(maxSz, sSpec->optPlnDim[i]);
      }
#ifdef WITH_OPT_BLK2
      MAXX(maxSz, maxWidth * sSpec->optResolution);
#endif
      oPln->maxNoR	= maxSz*1.15;					// The maximum number of r points we can handle
      oPln->maxNoZ 	= maxSz;					// The maximum number of z points we can handle
    }

    oPln->cuSrch	= sSrch;					// Set the pointer t the search specifications
    oPln->maxHalfWidth  = cu_z_resp_halfwidth<double>( zMax, HIGHACC );	// The halfwidth of the largest plane we think we may handle

    FOLD // Create streams  .
    {
      infoMSG(5,6,"Create streams.\n");

      CUDA_SAFE_CALL(cudaStreamCreate(&oPln->stream),"Creating stream for candidate optimisation.");
      char nmStr[1024];
      sprintf(nmStr,"Optimisation Stream %02i", oPln->pIdx);
      NV_NAME_STREAM(oPln->stream, nmStr);
    }

    FOLD // Create events  .
    {
      infoMSG(5,6,"Create Events.\n");

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

    FOLD // Allocate device memory  .
    {
      infoMSG(5,6,"Allocate device memory.\n");

      size_t freeMem, totalMem;

      oPln->input = (cuHarmInput*)malloc(sizeof(cuHarmInput));

      oPln->outSz   	= (oPln->maxNoR * oPln->maxNoZ ) * sizeof(float);
#ifdef WITH_OPT_BLK2
      int maxHarm = MAX(sSpec->optMinLocHarms, sSrch->noSrchHarms );
      oPln->outSz   	= (oPln->maxNoR * maxHarm * oPln->maxNoZ ) * sizeof(cufftComplex);
#endif
#ifdef WITH_OPT_PLN2
      int maxHarm = MAX(sSpec->optMinLocHarms, sSrch->noSrchHarms );
      oPln->outSz   	= (oPln->maxNoR * maxHarm * oPln->maxNoZ ) * sizeof(cufftComplex);
#endif
      oPln->input->size	= (maxWidth*10 + 2*oPln->maxHalfWidth) * sSrch->noSrchHarms * sizeof(cufftComplex)*2; // The noR is oversized to allo for moves of the plane withought getting new input

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
	infoMSG(5,6,"Input %.1f MB output %.1f MB.\n", oPln->input->size/1e6, oPln->outSz/1e6 );

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

int setStackInfo(cuFFdotBatch* batch, stackInfo* h_inf, int offset)
{
  infoMSG(4,4,"setStackInfo\n" );

  stackInfo* dcoeffs;
  cudaGetSymbolAddress((void **)&dcoeffs, STACKS );

  for (int i = 0; i < batch->noStacks; i++)
  {
    infoMSG(4,5,"stack %i\n",i);

    cuFfdotStack* cStack  = &batch->stacks[i];
    stackInfo*    cInf    = &h_inf[i];

    cInf->noSteps         = batch->noSteps;
    cInf->noPlanes        = cStack->noInStack;
    cInf->famIdx          = cStack->startIdx;
    cInf->flags           = batch->flags;

    cInf->d_iData         = cStack->d_iData;
    cInf->d_planeData     = cStack->d_planeMult;
    cInf->d_planePowers   = cStack->d_planePowr;

    // Set the pointer to constant memory
    cStack->stkIdx        = offset+i;
    cStack->d_sInf        = dcoeffs + offset+i ;
  }

  return batch->noStacks;
}

int setConstVals_Fam_Order( cuFFdotBatch* batch )
{
  FOLD // Set other constant values
  {
    void *dcoeffs;

    int		height[MAX_HARM_NO];
    int		stride[MAX_HARM_NO];
    int		width[MAX_HARM_NO];
    void*	kerPnt[MAX_HARM_NO];
    int		ker_off[MAX_HARM_NO];

    FOLD // Set values  .
    {
      for (int i = 0; i < batch->noGenHarms; i++)
      {
	cuFfdotStack* cStack  = &batch->stacks[ batch->hInfos[i].stackNo];

	height[i]	= batch->hInfos[i].noZ;
	stride[i]	= cStack->strideCmplx;
	width[i]	= batch->hInfos[i].width;
	kerPnt[i]	= batch->kernels[i].d_kerData;
	ker_off[i]	= batch->kernels[i].kreOff;

	if ( (i>=batch->noGenHarms) &&  (batch->hInfos[i].width != cStack->strideCmplx) )
	{
	  fprintf(stderr,"ERROR: Width is not the same as stride, using width this may case errors in the multiplication.\n");
	}
      }

      // Rest
      for (int i = batch->noGenHarms; i < MAX_HARM_NO; i++)
      {
	height[i]	= 0;
	stride[i]	= 0;
	width[i]	= 0;
	kerPnt[i]	= 0;
	ker_off[i]	= 0;
      }
    }

    cudaGetSymbolAddress((void **)&dcoeffs, HEIGHT_HARM);
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, &height, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice, batch->stacks->initStream),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, STRIDE_HARM);
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, &stride, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice, batch->stacks->initStream),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, WIDTH_HARM);
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, &width,  MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice, batch->stacks->initStream),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, KERNEL_HARM);
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, &kerPnt, MAX_HARM_NO * sizeof(void*), cudaMemcpyHostToDevice, batch->stacks->initStream),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, KERNEL_OFF_HARM);
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, &ker_off, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice, batch->stacks->initStream),      "Copying stages to device");
  }

  CUDA_SAFE_CALL(cudaGetLastError(), "Preparing the constant memory values for the multiplications.");

  return 1;
}

int setStackVals( cuFFdotBatch* batch )
{
  stackInfo* dcoeffs;

  if ( batch->noStacks > MAX_STACKS )
  {
    fprintf(stderr, "ERROR: Too many stacks in family in function %s in %s.", __FUNCTION__, __FILE__);
    exit(EXIT_FAILURE);
  }

  //if ( batch->isKernel )
  {
    int         l_STK_STRD[MAX_STACKS];
    char        l_STK_INP[MAX_STACKS][4069];

    for (int i = 0; i < batch->noStacks; i++)
    {
      cuFfdotStack* cStack  = &batch->stacks[i];

      l_STK_STRD[i] = cStack->strideCmplx;

      int         off     = 0;
      char        inpIdx  = 0;

      // Create the actual texture object
      for (int j = 0; j < cStack->noInStack; j++)        // Loop through planes in stack
      {
	cuHarmInfo*  hInf = &cStack->harmInf[j];

	// Create the actual texture object
	for (int k = 0; k < batch->noSteps; k++)        // Loop through planes in stack
	{
	  for ( int h = 0; h < hInf->noZ; h++ )
	  {
	    l_STK_INP[i][off++] = inpIdx;
	  }
	  inpIdx++;
	}
      }
    }

    cudaGetSymbolAddress((void **)&dcoeffs, STK_STRD );
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, l_STK_STRD, sizeof(l_STK_STRD), cudaMemcpyHostToDevice, batch->stacks->initStream),              "Copying stack info to device");

    cudaGetSymbolAddress((void **)&dcoeffs, STK_INP );
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, l_STK_INP, sizeof(l_STK_INP), cudaMemcpyHostToDevice, batch->stacks->initStream),      "Copying stack info to device");
  }

  return 1;
}

/** Copy host stack info to the device constant memory
 *
 * NOTE: The device should already be set!
 *
 * @param h_inf
 * @param noStacks
 * @return
 */
int setConstStkInfo(stackInfo* h_inf, int noStacks,  cudaStream_t stream)
{
  infoMSG(4,4,"set ConstStkInfo(%i)\n", noStacks );

  void *dcoeffs;

  // TODO: Do a test to see if  we are on the correct device

  cudaGetSymbolAddress((void **)&dcoeffs, STACKS);
  CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, h_inf, noStacks * sizeof(stackInfo), cudaMemcpyHostToDevice, stream),      "Copying stack info to device");

  return 1;
}

void drawPlaneCmplx(fcomplexcu* ffdotPlane, char* name, int stride, int height)
{
  float *h_fArr = (float*) malloc(stride * height * sizeof(fcomplexcu));
  //float DestS   = ffdotPlane->ffPowWidth*sizeof(float);
  //float SourceS = ffdotPlane->ffPowStride;
  CUDA_SAFE_CALL(cudaMemcpy2D(h_fArr, stride * sizeof(fcomplexcu), ffdotPlane, stride * sizeof(fcomplexcu), stride * sizeof(fcomplexcu), height, cudaMemcpyDeviceToHost), "Failed to copy data from device to host");

  //draw2DArray(name, h_fArr, stride*2, height);
  free(h_fArr);
}

void timeSynch(cuFFdotBatch* batch)
{
  if ( (*batch->rAraays)[batch->rActive][0][0].numrs )
  {
    PROF // Profiling
    {
      if ( batch->flags & FLAG_PROF )
      {
	infoMSG(1,2,"Profiling\n");

	float time;         // Time in ms of the thing
	cudaError_t ret;    // Return status of cudaEventElapsedTime

	FOLD // Norm Timing  .
	{
	  if ( (batch->flags & CU_NORM_GPU) )
	  {
	    for (int stack = 0; stack < batch->noStacks; stack++)
	    {
	      cuFfdotStack* cStack = &batch->stacks[stack];

	      ret = cudaEventElapsedTime(&time, cStack->normInit, cStack->normComp);

	      if ( ret != cudaErrorNotReady )
	      {
#pragma omp atomic
		batch->compTime[batch->noStacks*TIME_CMP_NRM + stack ] += time;
	      }
	    }

	    CUDA_SAFE_CALL(cudaGetLastError(), "Norm Timing");
	  }
	}

	FOLD // Input FFT timing  .
	{
	  if ( !(batch->flags & CU_INPT_FFT_CPU) )
	  {
	    for (int stack = 0; stack < batch->noStacks; stack++)
	    {
	      cuFfdotStack* cStack = &batch->stacks[stack];

	      ret = cudaEventElapsedTime(&time, cStack->inpFFTinit, cStack->inpFFTinitComp);

	      if ( ret != cudaErrorNotReady )
	      {
#pragma omp atomic
		batch->compTime[batch->noStacks*TIME_CMP_FFT + stack ] += time;
	      }
	    }

	    CUDA_SAFE_CALL(cudaGetLastError(), "Input FFT timing");
	  }
	}

	FOLD // Copy input data  .
	{
	  ret = cudaEventElapsedTime(&time, batch->iDataCpyInit, batch->iDataCpyComp);

	  if ( ret != cudaErrorNotReady )
	  {
#pragma omp atomic
	    batch->compTime[batch->noStacks*TIME_CMP_H2D] += time;
	  }

	  CUDA_SAFE_CALL(cudaGetLastError(), "Copy input timing");
	}

	FOLD // Multiplication timing  .
	{
	  if ( !(batch->flags & FLAG_MUL_CB) )
	  {
	    // Did the convolution by separate kernel

	    if ( batch->flags & FLAG_MUL_BATCH )   	// Convolution was done on the entire batch  .
	    {
	      ret = cudaEventElapsedTime(&time, batch->multInit, batch->multComp);

	      if ( ret != cudaErrorNotReady )
	      {
#pragma omp atomic
		batch->compTime[NO_STKS*TIME_CMP_MULT] += time;
	      }
	    }
	    else                                    // Convolution was on a per stack basis  .
	    {
	      for (int stack = 0; stack < batch->noStacks; stack++)              // Loop through Stacks
	      {
		cuFfdotStack* cStack = &batch->stacks[stack];

		ret = cudaEventElapsedTime(&time, cStack->multInit, cStack->multComp);

		if ( ret != cudaErrorNotReady )
		{
#pragma omp atomic
		  batch->compTime[NO_STKS*TIME_CMP_MULT + stack ] += time;
		}
	      }
	    }

	    CUDA_SAFE_CALL(cudaGetLastError(), "Multiplication timing");
	  }
	}

	FOLD // Inverse FFT timing  .
	{
	  for (int stack = 0; stack < batch->noStacks; stack++)
	  {
	    cuFfdotStack* cStack = &batch->stacks[stack];

	    ret = cudaEventElapsedTime(&time, cStack->ifftInit, cStack->ifftComp);
	    if ( ret != cudaErrorNotReady )
	    {
#pragma omp atomic
	      batch->compTime[NO_STKS*TIME_CMP_IFFT + stack ] += time;
	    }
	  }

	  CUDA_SAFE_CALL(cudaGetLastError(), "Inverse FFT timing");
	}

	FOLD // Copy to in-mem plane timing  .
	{
	  if ( batch->flags & FLAG_SS_INMEM )
	  {
	    for (int stack = 0; stack < batch->noStacks; stack++)
	    {
	      cuFfdotStack* cStack = &batch->stacks[stack];

	      ret = cudaEventElapsedTime(&time, cStack->ifftMemInit, cStack->ifftMemComp);
	      if ( ret != cudaErrorNotReady )
	      {
#pragma omp atomic
		batch->compTime[NO_STKS*TIME_CMP_D2D + stack ] += time;
	      }
	    }

	    CUDA_SAFE_CALL(cudaGetLastError(), "Copy to in-mem plane timing");
	  }
	}

	FOLD // Search Timing  .
	{
	  if ( !(batch->flags & FLAG_SS_CPU) && !(batch->flags & FLAG_SS_INMEM ) )
	  {
	    ret = cudaEventElapsedTime(&time, batch->searchInit, batch->searchComp);

	    if ( ret != cudaErrorNotReady )
	    {
#pragma omp atomic
	      batch->compTime[NO_STKS*TIME_CMP_SS] += time;
	    }

	    CUDA_SAFE_CALL(cudaGetLastError(), "Search Timing");
	  }
	}

	FOLD // Copy D2H  .
	{
	  if ( !(batch->flags & FLAG_SS_INMEM ) )
	  {
	    ret = cudaEventElapsedTime(&time, batch->candCpyInit, batch->candCpyComp);

	    if ( ret != cudaErrorNotReady )
	    {
#pragma omp atomic
	      batch->compTime[NO_STKS*TIME_CMP_D2H] += time;
	    }

	    CUDA_SAFE_CALL(cudaGetLastError(), "Copy D2H Timing");
	  }
	}
      }
    }
  }
}

/** Cycle the arrays of r-values  .
 *
 * @param batch
 */
void cycleRlists(cuFFdotBatch* batch)
{
  infoMSG(2,2,"Cycle R lists\n");

  rVals** hold = (*batch->rAraays)[batch->noRArryas-1];
  for ( int i = batch->noRArryas-1; i > 0; i-- )
  {
    (*batch->rAraays)[i] =  (*batch->rAraays)[i - 1];
  }
  (*batch->rAraays)[0] = hold;

  //  if ( msgLevel >= 3 )
  //  {
  //    for ( int i = 0 ; i < batch->noRArryas; i++ )
  //    {
  //      rVals* rVal = &(*batch->rAraays)[i][0][0];
  //
  //      printf("%i  step: %03i  r-low: %8.1f  numrs: %06ld\n", i, rVal->step, rVal->drlo, rVal->numrs );
  //    }
  //  }
}

/** Cycle the arrays of r-values  .
 *
 * @param batch
 */
void CycleBackRlists(cuFFdotBatch* batch)
{
  infoMSG(2,1,"CycleBackRlists\n");

  rVals** hold = (*batch->rAraays)[0];
  for ( int i = 0; i < batch->noRArryas-1; i++ )
  {
    (*batch->rAraays)[i] =  (*batch->rAraays)[i + 1];
  }

  (*batch->rAraays)[batch->noRArryas-1] = hold;
}

void cycleOutput(cuFFdotBatch* batch)
{
  infoMSG(2,2,"Cycle output\n");

  void* d_hold = batch->d_outData1;
  void* h_hold = batch->h_outData1;

  batch->d_outData1 = batch->d_outData2;
  batch->h_outData1 = batch->h_outData2;

  batch->d_outData2 = d_hold;
  batch->h_outData2 = h_hold;
}

void search_ffdot_batch_CU(cuFFdotBatch* batch)
{
  infoMSG(1,1,"search_ffdot_batch_CU\n");

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering search_ffdot_batch_CU.");

  // Setup input
  setActiveBatch(batch, 0);
  prepInput(batch);

  if ( batch->flags & FLAG_SYNCH )
  {
    multiplyBatch(batch);

    IFFTBatch(batch);

    if  ( batch->flags & FLAG_SS_INMEM )
    {
      copyToInMemPln(batch);
    }
    else
    {
      sumAndSearch(batch);

      getResults(batch);

      processSearchResults(batch);
    }
  }
  else
  {
    if  ( batch->flags & FLAG_SS_INMEM )
    {
      setActiveBatch(batch, 0);
      multiplyBatch(batch);

      setActiveBatch(batch, 0);
      IFFTBatch(batch);

      setActiveBatch(batch, 0);
      copyToInMemPln(batch);
    }
    else
    {
      // Sum and Search
      setActiveBatch(batch, 1);
      sumAndSearch(batch);

      // Results
      setActiveBatch(batch, 2);
      processSearchResults(batch);

      // Copy
      setActiveBatch(batch, 1);
      getResults(batch);

      setActiveBatch(batch, 0);
      convolveBatch(batch);
    }


  }

  // Change R-values
  cycleRlists(batch);
  setActiveBatch(batch, 1);
}

void finish_Search(cuFFdotBatch* batch)
{
  infoMSG(1,1,"Finish search\n");

  FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host
  {
    infoMSG(3,4,"pre synchronisation [blocking] ifftMemComp - stack\n");

    for (int ss = 0; ss < batch->noStacks; ss++)
    {
      infoMSG(4,5,"Stack %i\n", ss);

      PROF // Profiling  .
      {
	NV_RANGE_PUSH("EventSynch");
      }

      cuFfdotStack* cStack = &batch->stacks[ss];
      CUDA_SAFE_CALL(cudaEventSynchronize(cStack->ifftMemComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");

      PROF // Profiling  .
      {
	NV_RANGE_POP(); // EventSynch
      }
    }

    infoMSG(3,4,"pre synchronisation [blocking] processComp\n");

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("EventSynch");
    }

    CUDA_SAFE_CALL(cudaEventSynchronize(batch->processComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");

    PROF // Profiling  .
    {
      NV_RANGE_POP(); // EventSynch
    }
  }
}

void max_ffdot_planeCU(cuFFdotBatch* batch, double* searchRLow, double* searchRHi, fcomplexcu* fft, long long* numindep, float* powers)
{
  CUDA_SAFE_CALL(cudaGetLastError(), "Entering ffdot_planeCU2.");

  FOLD // Initialise input data  .
  {
    //setActiveBatch(batch, 0);
    //initInput(batch);
  }

  if ( batch->flags & FLAG_SYNCH )
  {

    FOLD // Multiply & inverse FFT  .
    {
      convolveBatch(batch);
    }

    FOLD // Sum & Max
    {
      //sumAndMax(batch, numindep, powers);
    }

  }
  else
  {

    FOLD // Sum & Max
    {
      //sumAndMax(batch, numindep, powers);
    }

    FOLD // Multiply & inverse FFT  .
    {
      convolveBatch(batch);
    }

  }

}

int selectDevice(int device, int print)
{
  cudaDeviceProp deviceProp;
  int currentDevvice, deviceCount;  //, device = 0;

  CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount), "Failed to get device count using cudaGetDeviceCount");
  //printf("There are %i CUDA capable devices available.");
  if (device>= deviceCount)
  {
    if (deviceCount== 0)
    {
      fprintf(stderr, "ERROR: Could not detect any CUDA capable devices!\n");
      exit(EXIT_FAILURE);
    }
    fprintf(stderr, "ERROR: Attempting to select device %i when I detect only %i devices, using device 0 instead!\n", device, deviceCount);
    device = 0;
  }

  CUDA_SAFE_CALL(cudaSetDevice(device), "Failed to set device using cudaSetDevice");
  CUDA_SAFE_CALL(cudaDeviceReset(), "Failed to set device using : cudaDeviceReset");
  CUDA_SAFE_CALL(cudaGetLastError(), "At start of everything?.\n");
  CUDA_SAFE_CALL(cudaGetDevice(&currentDevvice), "Failed to get device using cudaGetDevice");
  if (currentDevvice!= device)
  {
    fprintf(stderr, "ERROR: CUDA Device not set.\n");
    exit(EXIT_FAILURE);
  }

  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, currentDevvice), "Failed to get device properties device using cudaGetDeviceProperties");

  if (print)
    printf("\nRunning on device %d: \"%s\"  which has CUDA Capability  %d.%d\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);

  return ((deviceProp.major<< 4)+ deviceProp.minor);
}

void printCands(const char* fileName, GSList *cands, double T)
{
  if ( cands == NULL  )
    return;

  GSList *inp_list = cands ;

  FILE * myfile;                    /// The file being written to
  myfile = fopen ( fileName, "w" );

  if ( myfile == NULL )
    fprintf ( stderr, "ERROR: Unable to open log file %s\n", fileName );
  else
  {
    fprintf(myfile, "%4s\t%14s\t%10s\t%14s\t%13s\t%9s\t%7s\t%2s \n", "#", "t", "f", "z", "fd", "sig", "power", "harm" );
    int i = 0;

    while ( inp_list->next )
    {
      fprintf(myfile, "%4i\t%14.5f\t%10.6f\t%14.2f\t%13.10f\t%9.4f\t%7.2f\t%2i\n", i+1, ((accelcand *) (inp_list->data))->r, ((accelcand *) (inp_list->data))->r / T, ((accelcand *) (inp_list->data))->z,((accelcand *) (inp_list->data))->z/T/T, ((accelcand *) (inp_list->data))->sigma, ((accelcand *) (inp_list->data))->power, ((accelcand *) (inp_list->data))->numharm );
      inp_list = inp_list->next;
      i++;
    }
    fclose ( myfile );
  }
}

void printContext()
{
  int currentDevvice;
  CUcontext pctx;
  cuCtxGetCurrent ( &pctx );
  CUDA_SAFE_CALL(cudaGetDevice(&currentDevvice), "Failed to get device using cudaGetDevice");

  int trd;
#ifdef WITHOMP
  trd = omp_get_thread_num();
#else
  trd = 0;
#endif

  printf("Thread %02i  currentDevvice: %i Context %p \n", trd, currentDevvice, pctx);
}

int setDevice(int device)
{
  int dev;

  CUDA_SAFE_CALL(cudaGetDevice(&dev), "Failed to get device using cudaGetDevice");

  if ( dev != device )
  {
    CUDA_SAFE_CALL(cudaSetDevice(device), "Failed to set device using cudaSetDevice");
    CUDA_SAFE_CALL(cudaGetDevice(&dev), "Failed to get device using cudaGetDevice");
    if ( dev != device )
    {
      fprintf(stderr, "ERROR: CUDA Device not set.\n");
      exit(EXIT_FAILURE);
    }
  }

  return dev;
}

gpuSpecs gSpec(int devID = -1 )
{
  gpuSpecs gSpec;
  memset(&gSpec, 0 , sizeof(gpuSpecs));

  if (devID < 0 )
  {
    gSpec.noDevices      = getGPUCount();

    for ( int i = 0; i < gSpec.noDevices; i++)
      gSpec.devId[i]        = i;
  }
  else
  {
    gSpec.noDevices      = 1;
    gSpec.devId[0]       = devID;
  }

  // Set default
  for ( int i = 0; i < gSpec.noDevices; i++)
  {
    gSpec.noDevBatches[i] = 0;
    gSpec.noDevSteps[i]   = 0;
  }

  return gSpec;
}

/**  Read the GPU details from clig command line  .
 *
 * @param cmd     clig struct
 * @param bInf    A pointer to the accel info struct to fill
 */
gpuSpecs readGPUcmd(Cmdline *cmd)
{
  gpuSpecs gpul;
  memset(&gpul, 0 , sizeof(gpuSpecs));

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering readGPUcmd.");

  if ( cmd->gpuP ) // Determine the index and number of devices
  {
    if ( cmd->gpuC == 0 )  // NB: Note using gpuC == 0 requires a change in accelsearch_cmd.c every time clig is run!!!! [ usually line 32 should be "  /* gpuC = */ 0," ]
    {
      // Make a list of all devices
      gpul.noDevices   = getGPUCount();
      for ( int dev = 0 ; dev < gpul.noDevices; dev++ )
	gpul.devId[dev] = dev;
    }
    else
    {
      // User specified devices(s)
      gpul.noDevices   = cmd->gpuC;
      for ( int dev = 0 ; dev < gpul.noDevices; dev++ )
	gpul.devId[dev] = cmd->gpu[dev];
    }
  }

  for ( int dev = 0 ; dev < gpul.noDevices; dev++ ) // Loop over devices  .
  {
    if ( dev >= cmd->nbatchC )
      gpul.noDevBatches[dev] = cmd->nbatch[cmd->nbatchC-1];
    else
      gpul.noDevBatches[dev] = cmd->nbatch[dev];

    if ( dev >= cmd->nstepsC )
      gpul.noDevSteps[dev] = cmd->nsteps[cmd->nbatchC-1];
    else
      gpul.noDevSteps[dev] = cmd->nsteps[dev];

    if ( dev >= cmd->numoptC )
      gpul.noDevOpt[dev] = cmd->numopt[cmd->nbatchC-1];
    else
      gpul.noDevOpt[dev] = cmd->numopt[dev];

  }

  return gpul;
}

bool strCom(const char* str1, const char* str2)
{
  if ( strncmp(str1,str2, strlen(str2) ) == 0 )
    return 1;
  else
    return 0;
}

bool singleFlag ( int64_t*  flags, const char* str1, const char* str2, int64_t flagVal, const char* onVal, const char* offVal, int lineno, const char* fName )
{
  if      ( strCom("1", str2 ) || strCom(onVal, str2 ) )
  {
    (*flags) |=  flagVal;
    return true;
  }
  else if ( strCom("0", str2 ) || strCom(offVal, str2 ) )
  {
    (*flags) &= ~flagVal;
  }
  else if ( strCom(str2, "#" ) || strCom("", str2 )  )
  {
    // Blank do nothing
  }
  else
  {
    fprintf(stderr, "ERROR: Found unknown value \"%s\" for flag \"%s\" on line %i of %s.\n", str2, str1, lineno, fName);
  }
  return false;
}

/** Read accel search details from the text file
 *
 * @param sSpec
 */
void readAccelDefalts(searchSpecs *sSpec)
{
  int64_t*  flags = &(sSpec->flags);
  FILE *file;
  char fName[1024];
  sprintf(fName, "%s/lib/GPU_defaults.txt", getenv("PRESTO"));

  if ( file = fopen(fName, "r") )  // Read candidates from previous search  .
  {
    printf("Reading GPU search settings from %s\n",fName);

    char* line;
    char  line2[1024];
    int   lineno = 0;

    char str1[1024];
    char str2[1024];

    char *rest;

    while (fgets(line2, sizeof(line2), file))
    {
      lineno++;

      line = line2;

      // Strip proceeding white space
      while ( *line <= 32 &&  *line != 10 )
	line++;

      // Set to only be the word
      int flagLen = 0;
      char* flagEnd = line;
      while ( *flagEnd != ' ' && *flagEnd != 0 && *flagEnd != 10 )
      {
	flagLen++;
	flagEnd++;
      }

      int ll = strlen(line);

      str2[0] = 0;
      int pRead = sscanf(line, "%s %s", str1, str2 );
      if ( str2[0] == '#' )
	str2[0] = 0;

      if ( strCom(str1, "#" ) || ( ll == 1 ) )                  // Comment line
      {
	continue;
      }

      else if ( strCom(str1, "DUMMY" ) )                        // Dummy parameter
      {
	continue;
      }

      else if ( strCom("FLAG_SEPSRCH", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_SEPSRCH, "", "0", lineno, fName );
      }

      else if ( strCom("R_RESOLUTION", str1 ) )
      {
	int no1;
	int read1 = sscanf(line, "%s %i %s", str1, &no1, str2 );
	if ( read1 >= 2 )
	{
	  if ( no1 >= 1 && no1 <= 16 )
	  {
	    sSpec->noResPerBin = no1;
	  }
	  else
	  {
	    fprintf(stderr,"WARNING: Invalid optimisation resolution, it should range between 1 and 16 \n");
	  }
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	}
      }

      else if ( strCom("Z_RESOLUTIOM", str1 ) )
      {
	float no1;
	int read1 = sscanf(line, "%s %f %s", str1, &no1, str2 );
	if ( read1 >= 2 )
	{
	  if ( no1 > 0 && no1 <= 16 )
	  {
	    sSpec->zRes = no1;
	  }
	  else
	  {
	    fprintf(stderr,"WARNING: Invalid optimisation resolution, it should range between 1 and 16 \n");
	  }
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	}
      }

      else if ( strCom("FLAG_Z_SPLIT", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_Z_SPLIT, "", "0", lineno, fName );
      }

      else if ( strCom("INTERLEAVE", str1 ) ||  strCom("IL", str1 ) )   // Interleaving
      {
	singleFlag ( flags, str1, str2, FLAG_ITLV_ROW, "ROW", "PLN", lineno, fName );
      }

      else if ( strCom("RESPONSE", str1 ) )                     // Response shape
      {
	singleFlag ( flags, str1, str2, FLAG_KER_HIGH, "HIGH", "STD", lineno, fName );
      }

      else if ( strCom("FLAG_KER_HIGH", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_KER_HIGH, "", "0", lineno, fName );
      }

      else if ( strCom("FLAG_KER_MAX", str1 ) )                 // Kernel
      {
	singleFlag ( flags, str1, str2, FLAG_KER_MAX, "", "0", lineno, fName );
      }

      else if ( strCom("CENTER_RESPONSE", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_CENTER, "", "off", lineno, fName );
      }

      else if ( strCom("RESPONSE_PRECISION", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_KER_DOUBGEN, "DOUBLE", "SINGLE", lineno, fName );
      }

      else if ( strCom("KER_FFT_PRECISION", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_KER_DOUBFFT, "DOUBLE", "SINGLE", lineno, fName );
      }

      else if ( strCom("INP_NORM",	str1 ) )
      {
	(*flags) &= ~CU_NORM_GPU;	// Clear values

	if      ( strCom("CPU",  str2 ) || strCom("AA",  str2 ) )
	{
	  // CPU is no value clear is sufficient
	}
	else if ( strCom("GPU_SM", str2 ) || strCom("GPU", str2 ) )
	{
	  (*flags) |= CU_NORM_GPU_SM;
	}
	else if ( strCom("GPU_SM_MIN", str2 ) || strCom("GPU_SM2", str2 ))
	{
	  (*flags) |= CU_NORM_GPU_SM_MIN;
	}
	else if ( strCom("GPU_OS", str2 ) )
	{
	  (*flags) |= CU_NORM_GPU_OS;
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value \"%s\" for flag \"%s\" on line %i of %s.\n", str2, str1, lineno, fName);
	}
      }

      else if ( strCom("INP_FFT", str1 ) )
      {
	if ( singleFlag ( flags, str1, str2, CU_INPT_FFT_CPU, "CPU", "GPU", lineno, fName ) )
	{
	  // IF we are doing CPU FFT's we need to do CPU normalisation
	  (*flags) &= ~CU_NORM_GPU;
	}
      }

      else if ( strCom("MUL_KER", str1 ) )
      {
	if      ( strCom("00", str2 ) )
	{
	  (*flags) &= ~FLAG_MUL_ALL;
	  (*flags) |=  FLAG_MUL_00;
	}
	else if ( strCom("11", str2 ) )
	{
	  (*flags) &= ~FLAG_MUL_ALL;
	  (*flags) |=  FLAG_MUL_11;
	}
	else if ( strCom("21", str2 ) )
	{
	  (*flags) &= ~FLAG_MUL_ALL;
	  (*flags) |=  FLAG_MUL_21;
	}
	else if ( strCom("22", str2 ) )
	{
	  (*flags) &= ~FLAG_MUL_ALL;
	  (*flags) |=  FLAG_MUL_22;
	}
	else if ( strCom("23", str2 ) )
	{
	  (*flags) &= ~FLAG_MUL_ALL;
	  (*flags) |=  FLAG_MUL_23;
	}
	else if ( strCom("30", str2 ) )
	{
	  (*flags) &= ~FLAG_MUL_ALL;
	  (*flags) |=  FLAG_MUL_30;
	}
	else if ( strCom("CB", str2 ) )
	{
#if CUDA_VERSION >= 6050
	  (*flags) &= ~FLAG_MUL_ALL;
	  (*flags) |=  FLAG_MUL_CB;
#else
	  line[flagLen] = 0;
	  fprintf(stderr, "WARNING: Use of CUDA callbacks requires CUDA 6.5 or greater.  (FLAG: %s line %i in %s)\n", line, lineno, fName);
#endif
	}
	else if ( strCom(str2, "A"  ) )
	{
	  (*flags) &= ~FLAG_MUL_ALL;
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value \"%s\" for flag \"%s\" on line %i of %s.\n", str2, str1, lineno, fName);
	}
      }

      else if ( strCom("MUL_TEXTURE", str1 ) )
      {
	fprintf(stderr, "WARNING: The flag %s has been deprecated.\n", str1);
      }

      else if ( strCom("MUL_SLICES", str1 ) )
      {
	if ( strCom(str2, "A"   ) )
	{
	  sSpec->mulSlices = 0;
	}
	else
	{
	  int no;
	  int read1 = sscanf(str2, "%i", &no  );
	  if ( read1 == 1 )
	  {
	    sSpec->mulSlices = no;
	  }
	  else
	  {
	    line[flagLen] = 0;
	    fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	  }
	}
      }

      else if ( strCom("MUL_CHUNK", str1 ) )
      {
	if ( strCom(str2, "A"   ) )
	{
	  sSpec->mulChunk = 0;
	}
	else
	{
	  int no;
	  int read1 = sscanf(str2, "%i", &no  );
	  if ( read1 == 1 )
	  {
	    sSpec->mulChunk = no;
	  }
	  else
	  {
	    line[flagLen] = 0;
	    fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	  }
	}
      }

      else if ( strCom("CONVOLVE", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_CONV, "SEP", "CONT", lineno, fName );
      }

      else if ( strCom("STACK", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_STK_UP, "UP", "DN", lineno, fName );
      }

      else if ( strCom("CUFFT_PLAN", str1 ) )
      {
	singleFlag ( flags, str1, str2, CU_FFT_SEP, "SEPARATE", "SINGLE", lineno, fName );
      }

      else if ( strCom("STD_POWERS", str1 ) )
      {
	if      ( strCom("CB", str2 ) )
	{
#if CUDA_VERSION >= 6050
	  (*flags) |=     FLAG_CUFFT_CB_POW;
#else
	  line[flagLen] = 0;
	  fprintf(stderr,"WARNING: Use of CUDA callbacks requires CUDA 6.5 or greater.  (FLAG: %s line %i in %s)\n", line, lineno, fName);
#endif
	}
	else if ( strCom("SS", str2 ) )
	{
	  (*flags) &= ~FLAG_CUFFT_CB_POW;
	}
	else if ( strCom(str2, "#" ) || strCom("", str2 ) || strCom(str2, "A" ) )
	{
	  // Blank do nothing
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value \"%s\" for flag \"%s\" on line %i of %s.\n", str2, str1, lineno, fName);
	}
      }

      else if ( strCom("IN_MEM_POWERS", str1 ) )
      {
	if      ( strCom("CB", str2 ) )
	{
#if CUDA_VERSION >= 6050
	  (*flags) |=     FLAG_CUFFT_CB_INMEM;
#else
	  line[flagLen] = 0;
	  fprintf(stderr,"WARNING: Use of CUDA callbacks requires CUDA 6.5 or greater.  (FLAG: %s %s line %i in %s)\n", str1, str2, lineno, fName);
#endif
	}
	else if ( strCom("MEM_CPY", str2 ) || strCom("", str2 ))
	{
#if CUDA_VERSION >= 6050
	  (*flags) &=    ~FLAG_CUFFT_CB_INMEM;
	  (*flags) |=     FLAG_CUFFT_CB_POW;
#else
	  line[flagLen] = 0;
	  fprintf(stderr,"WARNING: Use of CUDA callbacks requires CUDA 6.5 or greater.  (FLAG: %s %s line %i in %s)\n", str1, str2, lineno, fName);
#endif
	}
	else if ( strCom("KERNEL", str2 ) )
	{
	  (*flags) &=    ~FLAG_CUFFT_CB_INMEM;
	  (*flags) &=    ~FLAG_CUFFT_CB_POW;
	}
	else if ( strCom(str2, "#" ) || strCom("", str2 ) || strCom(str2, "A" )  )
	{
	  // Blank do nothing
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value \"%s\" for flag \"%s\" on line %i of %s.\n", str2, str1, lineno, fName);
	}
      }

      else if ( strCom("FLAG_NO_CB", str1 ) )
      {
	(*flags) &= ~FLAG_CUFFT_ALL;
      }

      else if ( strCom("POWER_PRECISION", str1 ) )
      {
	if      ( strCom("HALF", str2 ) )
	{
#if CUDA_VERSION >= 7050
	  (*flags) |=  FLAG_POW_HALF;
#else
	  (*flags) &= ~FLAG_POW_HALF;

	  line[flagLen] = 0;
	  fprintf(stderr,"WARNING: Half precision can only be used with CUDA 7.5 or later! Reverting to single precision. (FLAG: %s line %i in %s)\n", line, lineno, fName);
#endif
	}
	else if ( strCom("SINGLE", str2 ) )
	{
	  (*flags) &= ~FLAG_POW_HALF;
	}
	else if ( strCom("DOUBLE", str2 ) )
	{
	  fprintf(stderr,"ERROR: Cannot sore in-mem plane as double! Defaulting to float.\n");
	  (*flags) &= ~FLAG_POW_HALF;
	}
	else if ( strCom(str2, "#" ) || strCom("", str2 ) || strCom(str2, "A" )  )
	{
	  // Blank do nothing
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value \"%s\" for flag \"%s\" on line %i of %s.\n", str2, str1, lineno, fName);
	}
      }

      else if ( strCom("SS_KER", str1 ) )
      {
	if      ( strCom("00",  str2 ) )
	{
	  (*flags) &= ~FLAG_SS_ALL;
	  (*flags) |= FLAG_SS_00;
	  (*flags) |= FLAG_RET_STAGES;
	}
	else if ( strCom("CPU", str2 ) )
	{
	  fprintf(stderr, "ERROR: CPU Sum and search is no longer supported.\n\n");
	  continue;

	  (*flags) &= ~FLAG_SS_ALL;
	  (*flags) |= FLAG_SS_CPU;

	  // CPU Significance
	  (*flags) &= ~FLAG_SIG_GPU;

	  sSpec->retType &= ~CU_SRT_ALL   ;
	  sSpec->retType |= CU_STR_PLN    ;

	  if ( (*flags) & FLAG_CUFFT_CB_POW )
	  {
	    sSpec->retType &= ~CU_TYPE_ALLL   ;
	    sSpec->retType |= CU_FLOAT        ;
	  }
	  else
	  {
	    sSpec->retType &= ~CU_TYPE_ALLL   ;
	    sSpec->retType |= CU_CMPLXF       ;
	  }
	}
	else if ( strCom("10",  str2 ) )
	{
	  (*flags) &= ~FLAG_SS_ALL;
	  (*flags) |= FLAG_SS_10;
	  (*flags) |= FLAG_RET_STAGES;
	}
	else if ( strCom("INMEM", str2 ) || strCom("IM", str2 ) )
	{
	  (*flags) |= FLAG_SS_INMEM;
	}
	else if ( strCom(str2, "A"  ) )
	{
	  (*flags) &= ~FLAG_SS_ALL;
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value \"%s\" for flag \"%s\" on line %i of %s.\n", str2, str1, lineno, fName);
	}
      }

      else if ( strCom("FLAG_SAS_TEX", str1 ) )
      {
	(*flags) |= FLAG_SAS_TEX;
      }

      else if ( strCom("FLAG_TEX_INTERP", str1 ) )
      {
	(*flags) |= FLAG_SAS_TEX;
	(*flags) |= FLAG_TEX_INTERP;
      }

      else if ( strCom("SIGNIFICANCE", str1 ) )
      {
	fprintf(stderr, "WARNING: The flag %s has been deprecated.\n", str1);
	//singleFlag ( flags, str1, str2, FLAG_SIG_GPU, "GPU", "CPU", lineno, fName );
      }

      else if ( strCom("RESULTS", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_THREAD, "THREAD", "SEQ", lineno, fName );
      }

      else if ( strCom("SS_SLICES", str1 ) )
      {
	if ( strCom(str2, "A"   ) )
	{
	  sSpec->ssSlices = 0;
	}
	else
	{
	  int no;
	  int read1 = sscanf(str2, "%i", &no  );
	  if ( read1 == 1 )
	  {
	    sSpec->ssSlices = no;
	  }
	  else
	  {
	    line[flagLen] = 0;
	    fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	  }
	}
      }

      else if ( strCom("SS_CHUNK", str1 ) )
      {
	if ( strCom(str2, "A"   ) )
	{
	  sSpec->ssChunk = 0;
	}
	else
	{
	  int no;
	  int read1 = sscanf(str2, "%i", &no  );
	  if ( read1 == 1 )
	  {
	    sSpec->ssChunk = no;
	  }
	  else
	  {
	    line[flagLen] = 0;
	    fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	  }
	}
      }

      else if ( strCom("SS_INMEM_SZ", str1 ) )
      {
	if ( strCom(str2, "A"   ) )
	{
	  sSpec->ssStepSize = 0;
	}
	else
	{
	  int no;
	  int read1 = sscanf(str2, "%i", &no  );
	  if ( read1 == 1 )
	  {
	    sSpec->ssStepSize = no;
	  }
	  else
	  {
	    line[flagLen] = 0;
	    fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	  }
	}
      }

      else if ( strCom("CAND_STORAGE", str1 ) )
      {
	if      ( strCom("ARR", str2 ) || strCom("", str2 ) )
	{
	  // Return type
	  sSpec->retType &= ~CU_TYPE_ALLL ;
	  sSpec->retType &= ~CU_SRT_ALL   ;

	  sSpec->retType |= CU_POWERZ_S   ;
	  sSpec->retType |= CU_STR_ARR    ;

	  // Candidate type
	  sSpec->cndType &= ~CU_TYPE_ALLL ;
	  sSpec->cndType &= ~CU_SRT_ALL   ;

	  sSpec->cndType |= CU_CANDFULL   ;
	  sSpec->cndType |= CU_STR_ARR    ;
	}
	else if ( strCom("LST", str2 ) )
	{
	  // Return type
	  sSpec->retType &= ~CU_TYPE_ALLL ;
	  sSpec->retType &= ~CU_SRT_ALL   ;

	  sSpec->retType |= CU_POWERZ_S   ;
	  sSpec->retType |= CU_STR_ARR    ;

	  // Candidate type
	  sSpec->cndType &= ~CU_TYPE_ALLL ;
	  sSpec->cndType &= ~CU_SRT_ALL   ;

	  sSpec->cndType |= CU_CANDFULL   ;
	  sSpec->cndType |= CU_STR_LST    ;
	}
	else if ( strCom("QUAD", str2 ) )
	{
	  fprintf(stderr, "ERROR: Quadtree storage not yet implemented. Doing nothing!\n");
	  continue;

	  // Candidate type
	  sSpec->cndType &= ~CU_TYPE_ALLL ;
	  sSpec->cndType &= ~CU_SRT_ALL   ;

	  sSpec->cndType |= CU_POWERZ_S   ;
	  sSpec->cndType |= CU_STR_QUAD   ;
	}
	else if ( strCom(str2, "#" ) || strCom("", str2 )  )
	{
	  // Blank do nothing
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value \"%s\" for flag \"%s\" on line %i of %s.\n", str2, str1, lineno, fName);
	}
      }

      else if ( strCom("RETURN", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_RET_STAGES, "STAGES", "FINAL", lineno, fName );
      }

      else if ( strCom("FLAG_RET_ARR", str1 ) )
      {
	sSpec->retType &= ~CU_SRT_ALL   ;
	sSpec->retType |= CU_STR_ARR    ;
      }
      else if ( strCom("FLAG_RET_PLN", str1 ) )
      {
	sSpec->retType &= ~CU_SRT_ALL   ;
	sSpec->retType |= CU_STR_PLN    ;
      }

      else if ( strCom("FLAG_STORE_ALL", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_STORE_ALL, "", "0", lineno, fName );
      }

      else if ( strCom("FLAG_STORE_EXP", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_STORE_EXP, "", "0", lineno, fName );
      }

      else if ( strCom("OPT_METHOUD", str1 ) )
      {
	if      ( strCom("PLANE", str2 ) )
	{
	  (*flags) &= ~FLAG_OPT_ALL;
	}
	else if ( strCom("SWARM", str2 ) )
	{
	  (*flags) &= ~FLAG_OPT_ALL;
	  (*flags) |= FLAG_OPT_SWARM;
	}
	else if ( strCom("NM", str2 ) )
	{
	  (*flags) &= ~FLAG_OPT_ALL;
	  (*flags) |= FLAG_OPT_NM;
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value \"%s\" for flag \"%s\" on line %i of %s.\n", str2, str1, lineno, fName);
	}
      }

      else if ( strCom("OPT_Z_RATIO", str1 ) )
      {
	float no1;
	int read1 = sscanf(line, "%s %f %s", str1, &no1, str2 );
	if ( read1 >= 2 )
	{
	  if ( no1 >= 0 && no1 <= 100 )
	  {
	    sSpec->zScale = no1;
	  }
	  else
	  {
	    fprintf(stderr,"WARNING: Invalid optimisation scale, it should range between 0 and 100 \n");
	  }
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	}
      }

      else if ( strCom("OPT_R_RES", str1 ) )
      {
	int no1;
	int read1 = sscanf(line, "%s %i %s", str1, &no1, str2 );
	if ( read1 >= 2 )
	{
	  if ( no1 >= 1 && no1 <= 128 )
	  {
	    sSpec->optResolution = no1;
	  }
	  else
	  {
	    fprintf(stderr,"WARNING: Invalid optimisation resolution, it should range between 1 and 128 \n");
	  }
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	}
      }

      else if ( strCom("OPT_NORM", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_OPT_LOCAVE, "LOCAVE", "MEDIAN", lineno, fName );
      }

      else if ( strCom("FLAG_OPT_BEST", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_OPT_BEST, "", "0", lineno, fName );
      }

      else if ( strCom("OPT_MIN_LOC_HARMS", str1 ) )
      {
	int no;
	int read1 = sscanf(str2, "%i", &no  );
	if ( read1 == 1 )
	{
	  if ( no >= 1 && no <= OPT_MAX_LOC_HARMS )
	  {
	    sSpec->optMinLocHarms = no;
	  }
	  else
	  {
	    fprintf(stderr,"WARNING: Invalid value, %s should range between 1 and %i \n", str1, OPT_MAX_LOC_HARMS);
	  }
	}
	else
	{
	  line[flagLen] = 0;
	  fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	}
      }

      else if ( strCom("OPT_MIN_REP_HARMS", str1 ) )
      {
	int no;
	int read1 = sscanf(str2, "%i", &no  );
	if ( read1 == 1 )
	{
	  sSpec->optMinRepHarms = no;
	}
	else
	{
	  line[flagLen] = 0;
	  fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	}
      }

      else if ( strCom("optPlnScale", str1 ) )
      {
	float no;
	int read1 = sscanf(str2, "%f", &no  );
	if ( read1 == 1 )
	{
	  sSpec->optPlnScale = no;
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	}
      }

      else if ( strCom("FLAG_OPT_DYN_HW", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_OPT_DYN_HW, "", "0", lineno, fName );
      }

      else if ( strCom("OPT_NELDER_MEAD_REFINE", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_OPT_NM_REFINE, "", "0", lineno, fName );
      }

      else if ( strCom("optPlnSiz", str1 ) )
      {
	int no1;
	int no2;
	int read1 = sscanf(line, "%s %i %i", str1, &no1, &no2 );
	if ( read1 == 3 )
	{
	  if    ( no1 == 1 )
	  {
	    sSpec->optPlnSiz[0] = no2;
	  }
	  else if ( no1 == 2 )
	  {
	    sSpec->optPlnSiz[1] = no2;
	  }
	  else if ( no1 == 4 )
	  {
	    sSpec->optPlnSiz[2] = no2;
	  }
	  else if ( no1 == 8 )
	  {
	    sSpec->optPlnSiz[3] = no2;
	  }
	  else if ( no1 == 16 )
	  {
	    sSpec->optPlnSiz[4] = no2;
	  }
	  else
	  {
	    fprintf(stderr, "WARNING: expecting optplnSiz 01, optplnSiz 02, optplnSiz 04, optplnSiz 08 or optplnSiz 16 \n");
	  }
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	}
      }

      else if ( strCom("optPlnDim", str1 ) )
      {
	int no1;
	int no2;
	int read1 = sscanf(line, "%s %i %i", str1, &no1, &no2 );
	if ( read1 == 3 )
	{
	  if ( no1 >= 1 && no1 <= NO_OPT_LEVS )
	  {
	    sSpec->optPlnDim[no1-1] = no2;
	  }
	  else
	  {
	    fprintf(stderr,"WARNING: Invalid optimisation plane number %i numbers should range between 1 and %i \n", no1, NO_OPT_LEVS);
	  }
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	}
      }

      else if ( strCom(line, "FLAG_OPT_DYN_HW" ) )
      {
	(*flags) |= FLAG_OPT_DYN_HW;
      }


      else if ( strCom("FLAG_DBG_SYNCH", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_SYNCH, "", "0", lineno, fName );
      }

      else if ( strCom("FLAG_DBG_PROFILING", str1 ) )
      {
#ifdef PROFILING
	singleFlag ( flags, str1, str2, FLAG_PROF, "", "0", lineno, fName );
#else
	fprintf(stderr, "ERROR: Found %s on line %i of %s, the program has not been compile with profiling enabled. Check the #define in cuda_accel.h.\n", str1, lineno, fName);
	exit(EXIT_FAILURE); // TMP - Added to mark an error for thesis timing
#endif
      }

      else if ( strCom("FLAG_DPG_PLT_OPT", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_DPG_PLT_OPT, "", "0", lineno, fName );
      }

      else if ( strCom("FLAG_DPG_PLT_POWERS", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_DPG_PLT_POWERS, "", "0", lineno, fName );
      }

      else if ( strCom("FLAG_DPG_UNOPT", str1 ) )
      {
	useUnopt    = 1;
      }

      else if ( strCom("FLAG_DBG_SKIP_OPT", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_DPG_SKP_OPT, "", "0", lineno, fName );
      }

      else if ( strCom("FLAG_DPG_PRNT_CAND", str1 ) )
      {
	singleFlag ( flags, str1, str2, FLAG_DPG_PRNT_CAND, "", "0", lineno, fName );
      }

      else if ( strCom("DBG_LEV", str1 ) )
      {
	int no;
	int read1 = sscanf(str2, "%i", &no  );
	if ( read1 == 1 )
	{
	  msgLevel = no;
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	}
      }

      else if ( strCom(line, "cuMedianBuffSz" ) )             // The size of the sub sections to use in the cuda median selection algorithm
      {
	rest = &line[ strlen("cuMedianBuffSz")+1];
	cuMedianBuffSz = atoi(rest);
      }

      else if ( strCom(line, "globalFloat01" ) )
      {
	rest = &line[ strlen("globalFloat01")+1];
	globalFloat01 = atof(rest);
      }
      else if ( strCom(line, "globalFloat02" ) )
      {
	rest = &line[ strlen("globalFloat02")+1];
	globalFloat02 = atof(rest);
      }
      else if ( strCom(line, "globalFloat03" ) )
      {
	rest = &line[ strlen("globalFloat03")+1];
	globalFloat03 = atof(rest);
      }
      else if ( strCom(line, "globalFloat04" ) )
      {
	rest = &line[ strlen("globalFloat04")+1];
	globalFloat04 = atof(rest);
      }
      else if ( strCom(line, "globalFloat05" ) )
      {
	rest = &line[ strlen("globalFloat05")+1];
	globalFloat05 = atof(rest);
      }

      else if ( strCom(line, "globalInt01" ) )
      {
	rest = &line[ strlen("globalInt01")+1];
	globalInt01 = atoi(rest);
      }
      else if ( strCom(line, "globalInt02" ) )
      {
	rest = &line[ strlen("globalInt02")+1];
	globalInt02 = atoi(rest);
      }
      else if ( strCom(line, "globalInt03" ) )
      {
	rest = &line[ strlen("globalInt03")+1];
	globalInt03 = atoi(rest);
      }
      else if ( strCom(line, "globalInt04" ) )
      {
	rest = &line[ strlen("globalInt04")+1];
	globalInt04 = atoi(rest);
      }
      else if ( strCom(line, "globalInt05" ) )
      {
	rest = &line[ strlen("globalInt05")+1];
	globalInt05 = atoi(rest);
      }

      else
      {
	line[flagLen] = 0;
	fprintf(stderr, "ERROR: Found unknown flag \"%s\" on line %i of %s.\n", line, lineno, fName);
	exit(EXIT_FAILURE); // TMP - Added to mark an error for thesis timing
      }
    }

    fclose (file);
  }
  else
  {
    printf("Unable to read GPU accel settings from %s\n", fName);
    exit(EXIT_FAILURE); // TMP - Added to mark an error for thesis timing
  }
}

searchSpecs readSrchSpecs(Cmdline *cmd, accelobs* obs)
{
  searchSpecs sSpec;
  memset(&sSpec, 0, sizeof(sSpec));

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering readSrchSpecs.");

  FOLD // Defaults for accel search  .
  {
    sSpec.flags		|= FLAG_KER_DOUBGEN ;	// Generate the kernels using double precision math (still stored as floats though)
    sSpec.flags		|= FLAG_ITLV_ROW    ;
    sSpec.flags         |= FLAG_CENTER      ;   // Centre and align the usable part of the planes

#ifndef DEBUG
    sSpec.flags		|= FLAG_THREAD      ; 	// Multithreading really slows down debug so only turn it on by default for release mode, NOTE: This can be over ridden in the defaults file
#endif

#if CUDA_VERSION >= 6050
    sSpec.flags		|= FLAG_CUFFT_CB_POW; 	// CUFFT callback to calculate powers, very efficient so on by default
#endif

#if CUDA_VERSION >= 7050
    sSpec.flags		|= FLAG_POW_HALF;
#endif

    if ( obs->inmem )
    {
      sSpec.flags	|= FLAG_SS_INMEM;
    }

    sSpec.flags         |= FLAG_RET_STAGES;

    sSpec.cndType	|= CU_CANDFULL;  	// Candidate data type - CU_CANDFULL this should be the default as it has all the needed data
    sSpec.cndType	|= CU_STR_ARR;  	// Candidate storage structure - CU_STR_ARR    is generally the fastest

    sSpec.retType	|= CU_POWERZ_S;  	// Return type
    sSpec.retType	|= CU_STR_ARR;  	// Candidate storage structure

    sSpec.fftInf.fft	= obs->fft;
    sSpec.fftInf.noBins	= obs->numbins;
    sSpec.fftInf.rlo	= obs->rlo;
    sSpec.fftInf.rhi	= obs->rhi;

    sSpec.normType	= obs->norm_type;

    sSpec.noResPerBin	= 2;
    sSpec.zRes		= 2;
    sSpec.noHarmStages	= obs->numharmstages;
    sSpec.zMax		= cmd->zmax;
    sSpec.sigma		= cmd->sigma;
    sSpec.pWidth	= cmd->width;

    sSpec.optPlnDim[0]	= 40;
    sSpec.optPlnDim[1]	= 20;
    sSpec.optPlnDim[2]	= 20;
    sSpec.optPlnDim[3]	= 10;
    sSpec.optPlnDim[4]	= 0;
    sSpec.optPlnDim[5]	= 5;

    sSpec.optPlnSiz[0]	= 8;
    sSpec.optPlnSiz[1]	= 7;
    sSpec.optPlnSiz[2]	= 6;
    sSpec.optPlnSiz[3]	= 5;
    sSpec.optPlnSiz[4]	= 4;

    sSpec.optPlnScale	 = 10;
    sSpec.optMinLocHarms = 1;
    sSpec.optMinRepHarms = 1;

    // Default: Auto chose best!
    sSpec.mulSlices	= 0 ;
    sSpec.mulChunk      = 0 ;
    sSpec.ssSlices	= 0 ;
    sSpec.ssChunk       = 0 ;
  }

  // Now read the
  readAccelDefalts(&sSpec);

  sSpec.zMax = cu_calc_required_z<double>(1, fabs(sSpec.zMax), sSpec.zRes);

  if ( sSpec.flags & (FLAG_SS_10 /*| FLAG_SS_20 | FLAG_SS_30 */ ) )
  {
    // Round the first bin to a multiple of the number of harmonics this is needed in the s&s kernel
    sSpec.fftInf.rlo  = floor(obs->rlo/(float)cmd->numharm)*cmd->numharm;
  }

  return sSpec;
}

/** Create multiplication kernel and allocate memory for planes on all devices  .
 *
 * Create the kernels on the first device and then copy it to all others
 *
 * @param sSrch     A pointer to the search structure
 *
 * @return
 */
void initPlanes(cuSearch* sSrch )
{
  infoMSG(2,1,"Create all planes.\n");

  sSrch->pInf = new cuPlnInfo;
  memset(sSrch->pInf, 0, sizeof(cuPlnInfo));

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering initCuAccel.");

  FOLD // Create the primary stack on each device, this contains the kernel  .
  {
    // Wait for cuda context to complete
    compltCudaContext(sSrch->gSpec);

    infoMSG(2,2,"Create the primary stack/kernel on each device\n");

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Init Kernels");
    }

    sSrch->pInf->kernels = (cuFFdotBatch*)malloc(sSrch->gSpec->noDevices*sizeof(cuFFdotBatch));

    int added;
    cuFFdotBatch* master = NULL;

    for ( int dev = 0 ; dev < sSrch->gSpec->noDevices; dev++ ) // Loop over devices  .
    {
      added = initKernel(&sSrch->pInf->kernels[sSrch->pInf->noDevices], master, sSrch, dev );

      if ( added > 0 )
      {
	infoMSG(5,5,"%s - initKernel returned %i batches.\n", __FUNCTION__, added);

	if ( !master ) // This was the first batch so it is the master
	{
	  master = &sSrch->pInf->kernels[0];
	}

	sSrch->gSpec->noDevBatches[dev] = added;
	sSrch->pInf->noBatches += added;
	sSrch->pInf->noDevices++;
      }
      else
      {
	sSrch->gSpec->noDevBatches[dev] = 0;
	fprintf(stderr, "ERROR: failed to set up a kernel on device %i, trying to continue... \n", sSrch->gSpec->devId[dev]);
      }
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP(); // Init Kernels
    }

    if ( sSrch->pInf->noDevices <= 0 ) // Check if we got any devices  .
    {
      fprintf(stderr, "ERROR: Failed to set up a kernel on any device. Try -lsgpu to see what devices there are.\n");
      exit (EXIT_FAILURE);
    }

  }

  FOLD // Create planes for calculations  .
  {
    infoMSG(2,2,"Create planes\n");

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Init Batches");
    }

    sSrch->pInf->noSteps       = 0;
    sSrch->pInf->batches       = (cuFFdotBatch*)malloc(sSrch->pInf->noBatches*sizeof(cuFFdotBatch));
    sSrch->pInf->devNoStacks   = (int*)malloc(sSrch->gSpec->noDevices*sizeof(int));
    sSrch->pInf->h_stackInfo   = (stackInfo**)malloc(sSrch->gSpec->noDevices*sizeof(stackInfo*));

    memset(sSrch->pInf->devNoStacks,0,sSrch->gSpec->noDevices*sizeof(int));

    int bNo = 0;
    int ker = 0;

    for ( int dev = 0 ; dev < sSrch->gSpec->noDevices; dev++ ) // Loop over devices  .
    {
      int noSteps = 0;
      if ( sSrch->gSpec->noDevBatches[dev] > 0 )
      {
	int firstBatch = bNo;

	infoMSG(5,5,"%s - device %i should have %i batches.\n", __FUNCTION__, dev, sSrch->gSpec->noDevBatches[dev]);

	for ( int batch = 0 ; batch < sSrch->gSpec->noDevBatches[dev]; batch++ )
	{
	  infoMSG(3,3,"Initialise batch %02i\n", bNo );

	  infoMSG(5,5,"%s - dev: %i - batch: %i - noBatches %i   \n", __FUNCTION__, sSrch->gSpec->noDevBatches[dev], batch, sSrch->gSpec->noDevBatches[dev] );

	  noSteps = initBatch(&sSrch->pInf->batches[bNo], &sSrch->pInf->kernels[ker], batch, sSrch->gSpec->noDevBatches[dev]-1);

	  if ( noSteps == 0 )
	  {
	    if ( batch == 0 )
	    {
	      fprintf(stderr, "ERROR: Failed to create at least one batch on device %i.\n", sSrch->pInf->kernels[dev].gInf->devid );
	    }
	    break;
	  }
	  else
	  {
	    infoMSG(5,5,"%s - initBatch initialised %i steps in batch %i.\n", __FUNCTION__, noSteps, batch);

	    sSrch->pInf->noSteps           += noSteps;
	    sSrch->pInf->devNoStacks[dev]  += sSrch->pInf->batches[bNo].noStacks;
	    bNo++;
	  }
	}

	int noStacks = sSrch->pInf->devNoStacks[dev] ;
	if ( noStacks )
	{
	  infoMSG(3,3,"Initialise constant memory for stacks\n" );

	  sSrch->pInf->h_stackInfo[dev] = (stackInfo*)malloc(noStacks*sizeof(stackInfo));
	  int idx = 0;

	  // Set the values of the host data structures
	  for (int batch = firstBatch; batch < bNo; batch++)
	  {
	    idx += setStackInfo(&sSrch->pInf->batches[batch], sSrch->pInf->h_stackInfo[dev], idx);
	  }

	  if ( idx != noStacks )
	  {
	    fprintf (stderr,"ERROR: in %s line %i, The number of stacks on device do not match.\n.",__FILE__, __LINE__);
	  }
	  else
	  {
	    setConstStkInfo(sSrch->pInf->h_stackInfo[dev], idx, sSrch->pInf->batches->stacks->initStream);
	  }
	}

	ker++;
      }
    }

    if ( bNo != sSrch->pInf->noBatches )
    {
      fprintf(stderr, "WARNING: Number of batches created does not match the number anticipated.\n");
      sSrch->pInf->noBatches = bNo;
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP(); // Init Batches
    }
  }
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
  sSrch->oInf->optResolution = sSrch->sSpec->optResolution;

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
	sSrch->oInf->opts[idx].pIdx     = idx;
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
	resp->dZ 	= sSrch->sSpec->zScale / (double)sSrch->sSpec->optResolution;
	resp->noRpnts	= sSrch->oInf->optResolution;
	resp->noZ	= resp->zMax * 2 / resp->dZ + 1 ;
	resp->halfWidth = halfWidth;
	resp->noR	= sSrch->oInf->optResolution*halfWidth*2 ;
	resp->oStride 	= getStrie( resp->noR, sizeof(float2), sSrch->gSpec->devInfo[dev].alignment);
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

void freeAccelGPUMem(cuPlnInfo* aInf)
{
  infoMSG(2,0,"FreeAccelGPUMem\n");

  PROF // Profiling  .
  {
    NV_RANGE_PUSH("Free GPU Mem");
  }

  FOLD // Free planes  .
  {
    for ( int batch = 0 ; batch < aInf->noBatches; batch++ )  // Batches
    {
      infoMSG(2,1,"freeBatchGPUmem %i\n", batch);

      freeBatchGPUmem(&aInf->batches[batch]);
    }
  }

  FOLD // Free kernels  .
  {
    for ( int dev = 0 ; dev < aInf->noDevices; dev++)         // Loop over devices
    {
      infoMSG(2,1,"freeKernelGPUmem device: %i\n", dev);

      freeKernelGPUmem(&aInf->kernels[dev]);
    }
  }

  PROF // Profiling  .
    {
      NV_RANGE_POP(); // Free GPU Mem
    }
}

void freeCuAccel(cuPlnInfo* mInf)
{
  if ( mInf )
  {
    FOLD // Free planes  .
    {
      for ( int batch = 0 ; batch < mInf->noBatches; batch++ )  // Batches
      {
	freeBatch(&mInf->batches[batch]);
      }
    }

    FOLD // Free kernels  .
    {
      for ( int dev = 0 ; dev < mInf->noDevices; dev++)  // Loop over devices
      {
	freeKernel(&mInf->kernels[dev] );
      }
    }

    freeNull(mInf->batches);
    freeNull(mInf->kernels);

    //    for ( int i = 0; i < MAX_GPUS; i++ )
    //      freeNull(mInf->name[i]);

    freeNull(mInf->devNoStacks);

    FOLD // Stack infos  .
    {
      for ( int dev = 0 ; dev < mInf->noDevices; dev++)  // Loop over devices
      {
	freeNull(mInf->h_stackInfo[dev]);
      }

      freeNull(mInf->h_stackInfo);
    }
  }
}

void intSrchThrd(cuSearch* srch)
{
  resThrds* tInf = srch->threasdInfo;

  if ( !tInf )
  {
    tInf     = new resThrds;
    memset(tInf, 0, sizeof(resThrds));
  }

  if (pthread_mutex_init(&tInf->candAdd_mutex, NULL))
  {
    printf("Unable to initialise a mutex.\n");
    exit(EXIT_FAILURE);
  }

  if (sem_init(&tInf->running_threads, 0, 0))
  {
    printf("Could not initialise a semaphore\n");
    exit(EXIT_FAILURE);
  }

  srch->threasdInfo = tInf;
}

cuSearch* initSearchInf(searchSpecs* sSpec, gpuSpecs* gSpec, cuSearch* srch)
{
  infoMSG(2,1,"Initialise search data structure\n");

  PROF // Profiling  .
  {
    NV_RANGE_PUSH("init Search inf");
  }

  bool same   = true;

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering initCuSearch.");

  if ( srch )				// Check if the search values have been pre-initialised  .
  {
    if ( srch->noHarmStages != sSpec->noHarmStages )
    {
      same = false;
      // ERROR recreate everything
    }

    if ( same && srch->pInf )
    {
      if ( srch->pInf->kernels->hInfos->zmax != sSpec->zMax )
      {
	same = false;
	// Have to recreate
      }

      presto_interp_acc accuracy = LOWACC;
      if ( sSpec->flags & FLAG_KER_HIGH )
	accuracy = HIGHACC;

      if ( srch->pInf->kernels->accelLen != optAccellen(sSpec->pWidth,sSpec->zMax, accuracy, sSpec->noResPerBin ) )
      {
	same = false;
	// Have to recreate
      }

      if ( !same )
      {
	fprintf(stderr,"ERROR: Call to %s with differing GPU search parameters. Will have to allocate new GPU memory and kernels.\n      NB: Not freeing the old memory!", __FUNCTION__);
      }
      else
      {
	// NB Assuming the GPU specks are all the same
      }
    }
  }

  if ( !srch || !same )			// Create a new search data structure  .
  {
    infoMSG(2,2,"Create a new search data structure\n");

    if ( !srch )
    {
      srch = new cuSearch;
    }
    memset(srch, 0, sizeof(cuSearch));

    srch->noHarmStages    = sSpec->noHarmStages;
    srch->noGenHarms      = ( 1<<(srch->noHarmStages-1) );
    srch->noSrchHarms     = ( 1<<(srch->noHarmStages-1) );
    srch->sIdx            = (int*)malloc(srch->noGenHarms * sizeof(int));
    srch->powerCut        = (float*)malloc(srch->noHarmStages * sizeof(float));
    srch->numindep        = (long long*)malloc(srch->noHarmStages * sizeof(long long));
  }
  else
  {
    infoMSG(2,2,"Using the existing search data structure\n");
  }

  srch->sSpec             = sSpec;
  srch->gSpec             = gSpec;

  FOLD // Calculate power cutoff and number of independent values  .
  {
    infoMSG(3,2,"Calculate power cutoff and number of independent values\n");

    // Calculate appropriate z-max
    int numz = round(sSpec->zMax / sSpec->zRes) * 2 + 1;
    float adjust = 0;

    FOLD // Calculate power cutoff and number of independent values  .
    {
      for (int ii = 0; ii < srch->noHarmStages; ii++)
      {
	if ( sSpec->zMax == 1 )
	{
	  srch->numindep[ii]  = (sSpec->fftInf.rhi - sSpec->fftInf.rlo) / (double)(1<<ii) ;
	}
	else
	{
	  srch->numindep[ii]  = (sSpec->fftInf.rhi - sSpec->fftInf.rlo) * (numz + 1) * ( sSpec->zRes / 6.95 ) / (double)(1<<ii);
	}


	// Power cutoff
	srch->powerCut[ii]    = power_for_sigma(sSpec->sigma, (1<<ii), srch->numindep[ii]);


	FOLD // Adjust for some lack in precision, if using half precision
	{
	  if ( sSpec->flags & FLAG_POW_HALF )
	  {
	    float noP = log10( srch->powerCut[ii] );
	    float dp = pow(10, floor(noP)-4 );  		// "Last" significant value

	    adjust = -dp*(1<<ii);			// Subtract one significant "value" for each harmonic
	    srch->powerCut[ii] += adjust;
	  }
	}

	infoMSG(6,6,"Stage %i numindep %12lli  threshold power %9.7f  adjusted %9.7f  \n", ii, srch->numindep[ii], srch->powerCut[ii], adjust);
      }
    }
  }

  FOLD // Set up the CPU threading  .
  {
    infoMSG(3,2,"Set up the CPU threading\n");

    intSrchThrd(srch);
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP();	// init Search inf
  }

  return srch;
}

cuSearch* initCuKernels(searchSpecs* sSpec, gpuSpecs* gSpec, cuSearch* srch)
{
  infoMSG(1,0,"Initialise CU search data structures\n");

  if ( !srch )			// Create basic data structure
  {
    srch = initSearchInf(sSpec, gSpec, srch);
  }

  if ( !srch->pInf )		// Populate data structure and create correlation kernels
  {
    initPlanes( srch );		// This initialises the plane info
  }
  else
  {
    // TODO: Do a whole bunch of checks here!
    fprintf(stderr, "ERROR: %s has not been set up to handle a pre-initialised memory info data structure.\n", __FUNCTION__);
    exit(EXIT_FAILURE);
  }

  return srch;
}

cuSearch* initCuOpt(searchSpecs* sSpec, gpuSpecs* gSpec, cuSearch* srch)
{
  if ( !srch )
    srch = initSearchInf(sSpec, gSpec, srch);

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
    NV_RANGE_POP();	// Init CUDA optemisers
  }

  return srch;
}

void freeCuSearch(cuSearch* srch)
{
  if (srch)
  {
    if ( srch->pInf )
      freeCuAccel(srch->pInf);

    freeNull(srch->sIdx);
    freeNull(srch->powerCut);
    freeNull(srch->numindep);

    freeNull(srch)
  }
}

void accelMax(cuSearch* srch)
{
  /*
  bool newKer = false;


  if ( aInf == NULL )
  {
    newKer = true;
    aInf = oneDevice(-1, fftinf, numharmstages, zMax, 8, 2, 4, CU_CAND_ARR | FLAG_STORE_EXP, CU_FLOAT, CU_FLOAT, (void*)powers );
  }

  master = &srch->mInf->kernels[0];
   */

  cuFFdotBatch* master   = NULL;    // The first kernel stack created
  master = srch->pInf->kernels;

#ifdef WITHOMP
  omp_set_num_threads(srch->pInf->noBatches);
#endif

  int ss = 0;
  int maxxx = ( srch->sSpec->fftInf.rhi - srch->sSpec->fftInf.rlo ) * srch->sSpec->noResPerBin / (float)( master->accelLen ) ; /// The number of planes we can work with

  if ( maxxx < 0 )
    maxxx = 0;

  int firstStep = 0;

#ifndef DEBUG
#pragma omp parallel
#endif
  FOLD
  {
#ifdef WITHOMP
    int tid = omp_get_thread_num();
#else
    int tid = 0;
#endif

    cuFFdotBatch* trdBatch = &srch->pInf->batches[tid];

    double*  startrs = (double*)malloc(sizeof(double)*trdBatch->noSteps);
    double*  lastrs  = (double*)malloc(sizeof(double)*trdBatch->noSteps);
    size_t rest = trdBatch->noSteps;

    setDevice(trdBatch->gInf->devid) ;

    while ( ss < maxxx )
    {
#pragma omp critical
      {
	firstStep = ss;
	ss       += trdBatch->noSteps;
	printf("\r   Step %07i of %-i %7.2f%%      \r", firstStep, maxxx,  firstStep/(float)maxxx*100);
	std::cout.flush();
      }

      if ( firstStep >= maxxx )
      {
	break;
      }

      for ( int step = 0; step < trdBatch->noSteps ; step ++)
      {
	if ( step < rest )
	{
	  startrs[step] = srch->sSpec->fftInf.rlo   + (firstStep+step) * ( master->accelLen / (double)srch->sSpec->noResPerBin );
	  lastrs[step]  = startrs[step] + master->accelLen  / (double)srch->sSpec->noResPerBin - 1 / (double)srch->sSpec->noResPerBin ;
	}
	else
	{
	  startrs[step] = 0 ;
	  lastrs[step]  = 0 ;
	}
      }
      //max_ffdot_planeCU(trdBatch, startrs, lastrs, 1, (fcomplexcu*)fftinf->fft, numindep, powers );
    }

    for ( int step = 0; step < trdBatch->noSteps ; step ++)
    {
      startrs[step] = 0;
      lastrs[step]  = 0;
    }

    // Finish searching the planes, this is required because of the out of order asynchronous calls
    for ( int pln = 0 ; pln < 2; pln++ )
    {
      //max_ffdot_planeCU(trdBatch, startrs, lastrs, 1,(fcomplexcu*)fftinf->fft, numindep, powers );

      //trdBatch->mxSteps = rest;
    }
    printf("\n");
  }

  /*
  printf("Free planes \n");

  FOLD // Free planes
  {
    for ( int pln = 0 ; pln < nPlanes; pln++ )  // Batches
    {
      freeBatch(planesj[pln]);
    }
  }

  printf("Free kernels \n");

  FOLD // Free kernels
  {
    for ( int dev = 0 ; dev < noKers; dev++)  // Loop over devices
    {
      freeHarmonics(&kernels[dev], master, (void*)powers );
    }
  }
   */

#ifndef DEBUG
  //printCands("GPU_Cands.csv", candsGPU);
#endif
}

void plotPlanes(cuFFdotBatch* batch)
{
  //#ifdef CBL
  //  printf("\n Creating data sets...\n");
  //
  //  nDarray<2, float>gpuCmplx [batch->noSteps][batch->noHarms];
  //  nDarray<2, float>gpuPowers[batch->noSteps][batch->noHarms];
  //  for ( int si = 0; si < batch->noSteps ; si ++)
  //  {
  //    for (int harm = 0; harm < batch->noGenHarms; harm++)
  //    {
  //      cuHarmInfo *hinf  = &batch[0].hInfos[harm];
  //
  //      gpuCmplx[si][harm].addDim(hinf->width*2, 0, hinf->width);
  //      gpuCmplx[si][harm].addDim(hinf->height, -hinf->zmax, hinf->zmax);
  //      gpuCmplx[si][harm].allocate();
  //
  //      gpuPowers[si][harm].addDim(hinf->width, 0, hinf->width);
  //      gpuPowers[si][harm].addDim(hinf->height, -hinf->zmax, hinf->zmax);
  //      gpuPowers[si][harm].allocate();
  //    }
  //  }
  //
  //  for ( int step = 0; step < batch->noSteps ; step ++)
  //  {
  //    for ( int stack = 0 ; stack < batch->noStacks; stack++ )
  //    {
  //      for (int harm = 0; harm < batch->noGenHarms; harm++)
  //      {
  //        cuHarmInfo   *cHInfo  = &batch->hInfos[harm];
  //        cuFfdotStack *cStack  = &batch->stacks[cHInfo->stackNo];
  //        rVals* rVal           = &batch->rArrays[batch->rActive][step][harm];
  //
  //        for( int y = 0; y < cHInfo->height; y++ )
  //        {
  //
  //          fcomplexcu *cmplxData;
  //          float *powers;
  //
  //          if ( batch->flag & FLAG_ITLV_ROW )
  //          {
  //            cmplxData = &batch->d_planeMult[  (y*batch->noSteps + step)*cStack->strideCmplx ];
  //            powers    = &batch->d_planePowr[ ((y*batch->noSteps + step)*cStack->strideFloat + cHInfo->halfWidth * 2 ) ];
  //          }
  //          else
  //          {
  //            cmplxData = &batch->d_planeMult[  (y + step*cHInfo->height)*cStack->strideCmplx ];
  //            powers    = &batch->d_planePowr[ ((y + step*cHInfo->height)*cStack->strideFloat  + cHInfo->halfWidth * 2 ) ];
  //          }
  //
  //          cmplxData += cHInfo->halfWidth*2;
  //          //CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[step][harm].getP(0,y), cmplxData, (cHInfo->width-2*2*cHInfo->halfWidth)*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
  //          //CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[step][harm].getP(0,y), cmplxData, (cPlane->numrs[step])*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
  //          CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[step][harm].getP(0,y), cmplxData, (rVal->numrs)*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
  //          if ( batch->flag & FLAG_CUFFT_CB_OUT )
  //          {
  //            //CUDA_SAFE_CALL(cudaMemcpyAsync(gpuPowers[step][harm].getP(0,y), powers, (cPlane->numrs[step])*sizeof(float),   cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
  //            CUDA_SAFE_CALL(cudaMemcpyAsync(gpuPowers[step][harm].getP(0,y), powers, (rVal->numrs)*sizeof(float),   cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
  //            /*
  //            for( int jj = 0; jj < plan->numrs[step]; jj++)
  //            {
  //              float *add = gpuPowers[step][harm].getP(jj*2+1,y);
  //              gpuPowers[step][harm].setPoint<ARRAY_SET>(add, 0);
  //            }
  //             */
  //          }
  //        }
  //      }
  //    }
  //  }
  //#else
  //  fprintf(stderr,"ERROR: Not compiled with debug libraries.\n");
  //#endif
}

void printBitString( int64_t val )
{
  printf("Value %015ld : ", val );

  for ( int i = 0; i < 64; i++)
  {
    if( val & ( 1ULL << (63-i) ) )
      printf("1");
    else
      printf("0");
  }
  printf("\n");
}

void printCommandLine(int argc, char *argv[])
{
  printf("Command:\t");

  for ( int i =0; i < argc; i ++ )
  {
    printf("%s ",argv[i]);
  }
  printf("\n");
}

void writeLogEntry(const char* fname, accelobs* obs, cuSearch* cuSrch, long long prepTime, long long cpuKerTime, long long cupTime, long long gpuKerTime, long long gpuTime, long long optTime, long long cpuOptTime, long long gpuOptTime)
{
#ifdef CBL
  searchSpecs* sSpec;         ///< Specifications of the search
  cuPlnInfo* mInf;            ///< The allocated Device and host memory and data structures to create planes including the kernels
  cuFFdotBatch* batch;

  sSpec         = cuSrch->sSpec;
  mInf          = cuSrch->pInf;

  if ( !cuSrch || !sSpec || !mInf  )
    return;

  batch         = cuSrch->pInf->batches;
  double noRR   = sSpec->fftInf.rhi - sSpec->fftInf.rlo;

  char hostname[1024];
  gethostname(hostname, 1024);

  Logger* cvsLog = new Logger((char*)fname, 1);
  cvsLog->setCsvDeliminator('\t');

  // Get the current time
  time_t rawtime;
  tm* ptm;
  time(&rawtime);
  ptm = gmtime(&rawtime);

  FOLD // Basics  .
  {
    cvsLog->csvWrite("Width",     "#", "%4i",     sSpec->pWidth);
    cvsLog->csvWrite("Stride",    "#", "%5i",     batch->stacks->strideCmplx);
    cvsLog->csvWrite("A-Len",     "#", "%5i",     batch->accelLen);

    cvsLog->csvWrite("Z max",     "#", "%03i",    sSpec->zMax);

    cvsLog->csvWrite("Devices",   "#", "%2i",     mInf->noDevices);
    cvsLog->csvWrite("GPU",       "#", "%2i",     batch->gInf->devid);

    cvsLog->csvWrite("Har",       "#", "%2li",    cuSrch->noGenHarms);
    cvsLog->csvWrite("Plns",      "#", "%2i",     batch->stacks->noInStack);

    cvsLog->csvWrite("Obs N",     "#", "%7.3f",   obs->N  * 1e-6);
    cvsLog->csvWrite("R bins",    "#", "%7.3f",   noRR    * 1e-6);

    cvsLog->csvWrite("Batches",   "#", "%2i",     mInf->noBatches);

    cvsLog->csvWrite("Steps",     "#", "%2i",     batch->noSteps);

    cvsLog->csvWrite("MU Slices", "#", "%2i",     batch->mulSlices);
    cvsLog->csvWrite("MU Chunk",  "#", "%2i",     batch->mulChunk);

    cvsLog->csvWrite("SS Slices", "#", "%2i",     batch->ssSlices);
    cvsLog->csvWrite("SS Chunk",  "#", "%2i",     batch->ssChunk);

    cvsLog->csvWrite("Sigma",     "#", "%4.2f",   sSpec->sigma);
    cvsLog->csvWrite("Time", "-", "%04i/%02i/%02i %02i:%02i:%02i", 1900 + ptm->tm_year, ptm->tm_mon, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec);
    cvsLog->csvWrite("hostname",  "s", "%s",      hostname);
  }

  FOLD // Flags  .
  {
    if ( batch->flags & FLAG_ITLV_ROW )
      cvsLog->csvWrite("IL",      "flg", "ROW");
    else
      cvsLog->csvWrite("IL",      "flg", "PLN");

    if ( batch->flags & CU_NORM_GPU_SM )
      cvsLog->csvWrite("NORM",    "flg", "GPU_SM");
    if ( batch->flags & CU_NORM_GPU_OS )
      cvsLog->csvWrite("NORM",    "flg", "GPU_OS");
    if ( batch->flags & CU_NORM_GPU )
      cvsLog->csvWrite("NORM",    "flg", "GPU");
    else
      cvsLog->csvWrite("NORM",    "flg", "CPU");

    if ( batch->flags & CU_INPT_FFT_CPU )
      cvsLog->csvWrite("Inp FFT", "flg", "CPU");
    else
      cvsLog->csvWrite("Inp FFT", "flg", "GPU");

    if      ( batch->flags & FLAG_MUL_00 )
      cvsLog->csvWrite("MUL",    "flg", "00");
    else if ( batch->flags & FLAG_MUL_11 )
      cvsLog->csvWrite("MUL",    "flg", "11");
    else if ( batch->flags & FLAG_MUL_21 )
      cvsLog->csvWrite("MUL",    "flg", "21");
    else if ( batch->flags & FLAG_MUL_22 )
      cvsLog->csvWrite("MUL",    "flg", "22");
    else if ( batch->flags & FLAG_MUL_23 )
      cvsLog->csvWrite("MUL",    "flg", "23");
    else if ( batch->flags & FLAG_MUL_30 )
      cvsLog->csvWrite("MUL",    "flg", "30");
    else if ( batch->flags & FLAG_MUL_CB )
      cvsLog->csvWrite("MUL",    "flg", "CB");
    else
      cvsLog->csvWrite("MUL",    "flg", "?");

    if      ( batch->flags & FLAG_SS_00  )
      cvsLog->csvWrite("SS",    "flg", "00");
    else if ( batch->flags & FLAG_SS_10  )
      cvsLog->csvWrite("SS",    "flg", "10");
    //    else if ( batch->flag & FLAG_SS_20  )
    //      cvsLog->csvWrite("SS",    "flg", "20");
    //    else if ( batch->flag & FLAG_SS_30  )
    //      cvsLog->csvWrite("SS",    "flg", "30");
    else if ( batch->flags & FLAG_SS_INMEM )
      cvsLog->csvWrite("SS",    "flg", "In-Mem");
    else if ( batch->flags & FLAG_SS_CPU )
      cvsLog->csvWrite("SS",    "flg", "CPU");
    else
      cvsLog->csvWrite("SS",    "flg", "?");

    cvsLog->csvWrite("in-mem ss",  "#", "%i", batch->strideOut );


    cvsLog->csvWrite("CB POW",    "flg", "%i", (bool)(batch->flags & FLAG_CUFFT_CB_POW));
    cvsLog->csvWrite("CB INMEM",  "flg", "%i", (bool)(batch->flags & FLAG_CUFFT_CB_INMEM));

    cvsLog->csvWrite("MUL_TEX",   "flg", "%i", (bool)(batch->flags & FLAG_TEX_MUL));
    cvsLog->csvWrite("SAS_TEX",   "flg", "%i", (bool)(batch->flags & FLAG_SAS_TEX));
    cvsLog->csvWrite("INTERP",    "flg", "%i", (bool)(batch->flags & FLAG_TEX_INTERP));
    if ( batch->flags & FLAG_SIG_GPU )
      cvsLog->csvWrite("SIG",    "flg", "GPU");
    else
      cvsLog->csvWrite("SIG",    "flg", "CPU");

    FOLD // Return details  .
    {
      if      ( batch->retType & CU_STR_ARR   )
	cvsLog->csvWrite("RET",  "strct", "ARR");
      else if ( batch->retType & CU_STR_LST  	)
	cvsLog->csvWrite("RET",  "strct", "LST");
      else if ( batch->retType & CU_STR_QUAD  )
	cvsLog->csvWrite("RET",  "strct", "QUAD");
      else
	cvsLog->csvWrite("RET",  "strct", "?");

      if      ( batch->retType & CU_POWERZ_S  )
	cvsLog->csvWrite("RET",  "type", "POWERZ_S");
      else if ( batch->retType & CU_POWERZ_I  )
	cvsLog->csvWrite("RET",  "type", "CU_POWERZ_I");
      else if ( batch->retType & CU_FLOAT  	  )
	cvsLog->csvWrite("RET",  "type", "FLOAT");
      else if ( batch->retType & CU_CANDFULL  )
	cvsLog->csvWrite("RET",  "type", "CU_CANDFULL");
      else
	cvsLog->csvWrite("RET",  "type", "?");
    }

    FOLD // Candidate storage  .
    {
      if      ( batch->cndType & CU_STR_ARR   )
	cvsLog->csvWrite("CAND",  "strct", "ARR");
      else if ( batch->cndType & CU_STR_LST  	)
	cvsLog->csvWrite("CAND",  "strct", "LST");
      else if ( batch->cndType & CU_STR_QUAD  )
	cvsLog->csvWrite("CAND",  "strct", "QUAD");
      else
	cvsLog->csvWrite("CAND",  "strct", "?");

      if      ( batch->cndType & CU_POWERZ_S  )
	cvsLog->csvWrite("CAND",  "type", "POWERZ_S");
      else if ( batch->cndType & CU_POWERZ_I  )
	cvsLog->csvWrite("CAND",  "type", "CU_POWERZ_I");
      else if ( batch->cndType & CU_FLOAT  	  )
	cvsLog->csvWrite("CAND",  "type", "FLOAT");
      else if ( batch->cndType & CU_CANDFULL  )
	cvsLog->csvWrite("CAND",  "type", "CU_CANDFULL");
      else
	cvsLog->csvWrite("CAND",  "type", "?");
    }

    cvsLog->csvWrite("RET_ALL",     "flg", "%i", (bool)(batch->flags & FLAG_RET_STAGES));
    cvsLog->csvWrite("STR_ALL",     "flg", "%i", (bool)(batch->flags & FLAG_STORE_ALL));
    cvsLog->csvWrite("STR_EXP",     "flg", "%i", (bool)(batch->flags & FLAG_STORE_EXP));

    if      ( batch->cndType & FLAG_KER_HIGH  )
      cvsLog->csvWrite("KER_HW",  "type", "HIGH");
    else
      cvsLog->csvWrite("KER_HW",  "type", "STD");

    cvsLog->csvWrite("KER_MAX",     "flg", "%i", (bool)(batch->flags & FLAG_KER_MAX) );
    cvsLog->csvWrite("KER_CENT",    "flg", "%i", (bool)(batch->flags & FLAG_CENTER)  );
  }

  FOLD // Timing  .
  {
    cvsLog->csvWrite("Prep",      "s", "%9.4f",   prepTime    * 1e-6);
    cvsLog->csvWrite("CPU ker",   "s", "%9.4f",   cpuKerTime  * 1e-6);
    cvsLog->csvWrite("CPU Srch",  "s", "%9.4f",   cupTime     * 1e-6);
    cvsLog->csvWrite("GPU ker",   "s", "%9.4f",   gpuKerTime  * 1e-6);
    cvsLog->csvWrite("GPU Srch",  "s", "%9.4f",   gpuTime     * 1e-6);
    cvsLog->csvWrite("Opt",       "s", "%9.4f",   optTime     * 1e-6);
    cvsLog->csvWrite("CPU Opt",   "s", "%9.4f",   cpuOptTime  * 1e-6);
    cvsLog->csvWrite("GPU Opt",   "s", "%9.4f",   gpuOptTime  * 1e-6);
  }

  PROF // Profiling
  {
    if ( batch->flags & FLAG_PROF )
    {
      float	bsums[TIME_CMP_END];
      for ( int i = 0; i < TIME_CMP_END; i++)
	bsums[i] = 0;

      for ( int comp=0; comp < TIME_CMP_END; comp++)
      {
	for (int batch = 0; batch < cuSrch->pInf->noBatches; batch++)
	{
	  for (int stack = 0; stack < cuSrch->pInf->batches[batch].noStacks; stack++)
	  {
	    cuFFdotBatch* batches = &cuSrch->pInf->batches[batch];

	    bsums[comp] += batches->compTime[comp*batches->noStacks+stack];
	  }
	}
      }

      cvsLog->csvWrite("Response",    "ms", "%12.6f", bsums[TIME_CMP_RESP]);
      cvsLog->csvWrite("kerFFT",      "ms", "%12.6f", bsums[TIME_CMP_KERFFT]);
      cvsLog->csvWrite("copyH2D",     "ms", "%12.6f", bsums[TIME_CMP_H2D]);
      cvsLog->csvWrite("InpNorm",     "ms", "%12.6f", bsums[TIME_CMP_NRM]);
      cvsLog->csvWrite("InpFFT",      "ms", "%12.6f", bsums[TIME_CMP_FFT]);
      cvsLog->csvWrite("Mult",        "ms", "%12.6f", bsums[TIME_CMP_MULT]);
      cvsLog->csvWrite("InvFFT",      "ms", "%12.6f", bsums[TIME_CMP_IFFT]);
      cvsLog->csvWrite("plnCpy",      "ms", "%12.6f", bsums[TIME_CMP_D2D]);
      cvsLog->csvWrite("Sum & Srch",  "ms", "%12.6f", bsums[TIME_CMP_SS]);
      cvsLog->csvWrite("Result",      "ms", "%12.6f", bsums[TIME_CMP_STR]);
      cvsLog->csvWrite("copyD2H",     "ms", "%12.6f", bsums[TIME_CMP_D2H]);
      cvsLog->csvWrite("Refine",      "ms", "%12.6f", bsums[TIME_CMP_REFINE]);
      cvsLog->csvWrite("Derivs",      "ms", "%12.6f", bsums[TIME_CMP_DERIVS]);
    }
  }

  cvsLog->csvEndLine();
#endif
}

GSList* getCanidates(cuFFdotBatch* batch, GSList *cands )
{
  //  gridQuadTree<double, float>* qt = (gridQuadTree<double, float>*)(batch->h_candidates) ;
  //  quadNode<double, float>* head = qt->getHead();
  //
  //  qt->update();
  //
  //  printf("GPU search found %li unique values in tree.\n", head->noEls );

  return cands;
}

int hilClimb(candTree* tree, double tooclose = 5)
{
  container* cont = tree->getSmallest();
  //double tooclose = 5;

  while ( cont )
  {
    container* largest = tree->getLargest(cont, tooclose);
    if ( *largest > *cont )
    {
      tree->markForRemoval(cont);
    }
    cont = cont->larger;
  }

  uint rem = tree->removeMarked();
  printf("hilClimb  Removed %6i - %6i remain \n", rem, tree->noVals() );

  return rem;
}

int eliminate_harmonics(candTree* tree, double tooclose = 1.5)
{
  infoMSG(1,2,"Eliminate harmonics");

  int maxharm = 16;
  int numremoved = 0;

  initCand* tempCand = new initCand;
  container* next;
  container* close;
  container* serch;

  container* lst = tree->getLargest();

  while ( lst )
  {
    initCand* candidate = (initCand*)lst->data;

    tempCand->power    = candidate->power;
    tempCand->numharm  = candidate->numharm;
    tempCand->r        = candidate->r;
    tempCand->z        = candidate->z;
    tempCand->sig      = candidate->sig;

    // Remove harmonics down
    for (double ii = 1; ii <= maxharm; ii++)
    {
      FOLD // Remove down candidates  .
      {
	tempCand->r  = candidate->r / ii;
	tempCand->z  = candidate->z / ii;
	serch       = contFromCand(tempCand);
	close       =  tree->getAll(serch, tooclose);

	while (close)
	{
	  next = close->smaller;

	  if ( *close != *lst )
	  {
	    tree->remove(close);
	    numremoved++;
	  }

	  close = next;
	}
      }

      FOLD // Remove down up  .
      {
	tempCand->r  = candidate->r * ii;
	tempCand->z  = candidate->z * ii;
	serch       = contFromCand(tempCand);
	close       =  tree->getAll(serch, tooclose/**sqrt(ii)*/);

	while (close)
	{
	  next = close->smaller;

	  if ( *close != *lst )
	  {
	    tree->remove(close);
	    numremoved++;
	  }

	  close = next;
	}
      }
    }

    for (int ii = 1; ii < 23; ii++)
    {
      tempCand->r  = candidate->r * ratioARR[ii];
      tempCand->z  = candidate->z * ratioARR[ii];
      serch       = contFromCand(tempCand);
      close       =  tree->getAll(serch, tooclose);

      while (close)
      {
	next = close->smaller;

	if ( *close != *lst )
	{
	  tree->remove(close);
	  numremoved++;
	}

	close = next;
      }
    }

    lst = lst->smaller;
  }

  printf("Harmonics Removed %6i - %6i remain \n", numremoved, tree->noVals() );

  return (numremoved);
}

//GSList *testTest(cuFFdotBatch* batch, GSList *candsGPU)
//{
//  candTree optemised;
//
//  candTree trees[batch->noHarmStages];
//
//  candTree* qt =(candTree*)batch->h_candidates;
//
//  hilClimb(qt, 5);
//  eliminate_harmonics(qt);
//
//  cuOptCand* oPlnPln;
//  oPlnPln   = initOptPln(batch->sInf->sSpec);
//
//  container* cont = qt->getLargest();
//
//  int i = 0;
//
//  while ( cont )
//  {
//    i++;
//    printf("\n");
//    //if ( i == 12 )
//    {
//      cand*   candidate = (cand*)cont->data;
//      cont->flag &= ~OPTIMISED_CONTAINER;
//
//      printf("Candidate %03i  harm: %2i   pow: %9.3f   r: %9.4f  z: %7.4f\n",i, candidate->numharm, candidate->power, candidate->r, candidate->z );
//
//      //
//      //    numharm   = candidate->numharm;
//      //    sig       = candidate->sig;
//      //    rr        = candidate->r;
//      //    zz        = candidate->z;
//      //    poww      = candidate->power;
//      //
//      //    candsGPU  = insert_new_accelcand(candsGPU, poww, sig, numharm, rr, zz, &added );
//
//      //accelcand *cand = new accelcand;
//      //memset(cand, 0, sizeof(accelcand));
//      //cand->power   = candidate->power;
//      //cand->r       = candidate->r;
//      //cand->sigma   = candidate->sig;
//      //cand->z       = candidate->z;
//      //cand->numharm = candidate->numharm;
//
//      //accelcand* cand = create_accelcand(candidate->power, candidate->sig, candidate->numharm, candidate->r, candidate->z);
//
//      //candsGPU = insert_accelcand(candsGPU, cand),
//
//      int stg = log2((float)candidate->numharm);
//      candTree* ret = opt_cont(&trees[stg], oPlnPln, cont, &batch->sInf->sSpec->fftInf, i);
//
//      trees[stg].add(ret);
//
//      delete(ret);
//
//      if ( cont->flag & OPTIMISED_CONTAINER )
//      {
//        candidate->sig = candidate_sigma_cu(candidate->power, candidate->numharm,  batch->sInf->numindep[stg] );
//        container* cont = optemised.insert(candidate, 0.1);
//
//        if ( cont )
//        {
//          printf("          %03i  harm: %2i   pow: %9.3f   r: %9.4f  z: %7.4f\n",i, candidate->numharm, candidate->power, candidate->r, candidate->z );
//        }
//        else
//        {
//          printf("          NO\n");
//        }
//      }
//      else
//      {
//        printf("          Already Done\n");
//      }
//    }
//
//    cont = cont->smaller;
//  }
//
//  printf("Optimisation Removed %6i - %6i remain \n", qt->noVals() - optemised.noVals(), optemised.noVals() );
//
//  eliminate_harmonics(&optemised);
//
//  return candsGPU;
//}

//
//uint getOffset(int height, int stride, int harmNo, int stepNo, int rowNo, void* data = NULL)
//{
//  offset    = (rowNo + stepNo*height)*stride + cHInfo->halfWidth * 2
//}
//
//void* getPowerRow(cuFFdotBatch* batch, int harmNo, int stepNo, int rowNo, void* data = NULL)
//{
//  int stackNo = batch->hInfos[harmNo].stackNo;
//
//  cuFfdotStack* cStack    = &batch->stacks[stackNo];
//  cuFFdot*      plan      = &batch->planes[harmNo];
//  cuHarmInfo*   cHInfo    = &batch->hInfos[harmNo];      // The current harmonic we are working on
//
//  if ( data == NULL )
//    data = batch->d_planePowr;
//
//  void* plnData;
//  int   offset = 0;
//
//  if      ( batch->flag & FLAG_ITLV_ROW )
//  {
//    offset    = (rowNo*batch->noSteps + stepNo)*cStack->stridePower + cHInfo->halfWidth * 2 ;
//    //powers    = &((float*)plan->d_planePowr)[offset];
//  }
//  else
//  {
//    offset    = (rowNo + stepNo*cHInfo->height)*cStack->stridePower + cHInfo->halfWidth * 2
//    //powers    = &((float*)plan->d_planePowr)[offset];
//  }
//
//  if      ( batch->flag & FLAG_POW_HALF )
//  {
//    retrun &((half*)      plan->d_planePowr)[offset];
//  }
//  else if ( batch->flag & FLAG_CUFFT_CB_POW )
//  {
//    retrun &((float*)     plan->d_planePowr)[offset];
//  }
//  else if ( batch->flag & FLAG_CUFFT_CB_POW )
//  {
//    retrun &((cmplxData*) plan->d_planePowr)[offset];
//  }
//
//}
//
//// Copy data from device  .
//int getPowers(cuFFdotBatch* batch, float* dst)
//{
//  ulong sz  = 0;
//  harm      = 0;
//
//  void* out;
//
//  if      ( batch->flag & FLAG_POW_HALF )         // half output
//  {
//    out = malloc( batch->pwrDataSize * 2 );
//  }
//  else if ( batch->flag & FLAG_CUFFT_CB_POW ) // float output
//  {
//    out = malloc( batch->pwrDataSize );
//  }
//  else
//  {
//    out = malloc( batch->pwrDataSize / 2.0 ); // fcomplexcu output
//  }
//
//  // Read data from device
//  CUDA_SAFE_CALL(cudaMemcpyAsync(out, batch->d_planePowr, batch->pwrDataSize,   cudaMemcpyDeviceToHost, batch->resStream), "Failed to copy input data from device.");
//
//  // Write data to page locked memory
//  for ( int stackNo = 0; stackNo < batch->noStacks; stackNo++ )
//  {
//    cuFfdotStack* cStack = &batch->stacks[stackNo];
//
//    for ( int plainNo = 0; plainNo < cStack->noInStack; plainNo++ )
//    {
//      cuHarmInfo* cHInfo    = &batch->hInfos[harm];      // The current harmonic we are working on
//      cuFFdot*    plan      = &cStack->planes[plainNo];          // The current plane
//
//      for ( int stepNo = 0; stepNo < batch->noSteps; stepNo ++) // Loop over steps
//      {
//        rVals* rVal = &((batch->rValues)[stepNo][harm]);
//
//        if ( rVal->numdata )
//        {
//          //// Copy input data from GPU
//          //fcomplexcu *data = &batch->d_iData[sz];
//          //CUDA_SAFE_CALL(cudaMemcpyAsync(out, data, cStack->strideCmplx*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftIStream), "Failed to copy input data from device.");
//
//          CUDA_SAFE_CALL(cudaMemcpyAsync(out, plan->d_planePowr, (rVal->numrs)*sizeof(float),   cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
//
//          // Copy pain from GPU
//          for( int y = 0; y < cHInfo->height; y++ )
//          {
//            fcomplexcu *cmplxData;
//            float *powers;
//
//            if      ( batch->flag & FLAG_ITLV_ROW )
//            {
//              cmplxData = &plan->d_planeMult[(y*batch->noSteps + stepNo)*cStack->strideCmplx   + cHInfo->halfWidth * 2 ];
//              powers    = &((float*)plan->d_planePowr)[(y*batch->noSteps + stepNo)*cStack->stridePower + cHInfo->halfWidth * 2 ];
//            }
//            else
//            {
//              cmplxData = &plan->d_planeMult[(y + stepNo*cHInfo->height)*cStack->strideCmplx   + cHInfo->halfWidth * 2 ];
//              powers    = &((float*)plan->d_planePowr)[(y + stepNo*cHInfo->height)*cStack->stridePower + cHInfo->halfWidth * 2 ];
//            }
//
//            if      ( batch->flag & FLAG_CUFFT_CB_OUT )
//            {
//              //CUDA_SAFE_CALL(cudaMemcpyAsync(gpuPowers[step][harm].getP(0,y), powers, (plan->numrs[step])*sizeof(float),   cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
//              CUDA_SAFE_CALL(cudaMemcpyAsync(gpuPowers[stepNo][harm].getP(0,y), powers, (rVal->numrs)*sizeof(float),   cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
//              /*
//                                   for( int jj = 0; jj < plan->numrs[step]; jj++)
//                                   {
//                                     float *add = gpuPowers[step][harm].getP(jj*2+1,y);
//                                     gpuPowers[step][harm].setPoint<ARRAY_SET>(add, 0);
//                                   }
//               */
//            }
//            else
//            {
//              //cmplxData += cHInfo->halfWidth*ACCEL_RDR;
//              //CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[step][harm].getP(0,y), cmplxData, (plan->numrs[step])*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
//              CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[stepNo][harm].getP(0,y), cmplxData, (rVal->numrs)*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
//            }
//          }
//        }
//
//        sz += cStack->strideCmplx;
//      }
//      harm++;
//    }
//
//    // New events for Synchronisation (this event will override the previous event)
//    cudaEventRecord(cStack->prepComp, cStack->fftIStream);
//    cudaEventRecord(cStack->ifftComp,  cStack->fftPStream);
//  }
//
//  free(out);
//}

/**  Wait for CPU threads to complete  .
 *
 */
int waitForThreads(sem_t* running_threads, const char* msg, int sleepMS )
{
  infoMSG(1,2,"Wait for CPU threads to complete\n");

  int noTrd;
  sem_getvalue(running_threads, &noTrd );

  if (noTrd)
  {
    char waitMsg[1024];
    int ite = 0;

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Wait on CPU threads");
    }

    while ( noTrd > 0 )
    {
      PROF // Profiling  .
      {
	NV_RANGE_PUSH("Sleep");
      }

      ite++;

      if ( noTrd >= 1 && !(ite % 10) )
      {
	sprintf(waitMsg,"%s  %3i thread still active.", msg, noTrd);

	FOLD  // Spinner  .
	{
	  if      (ite == 1 )
	    printf("\r%s⌜   ", waitMsg);
	  if      (ite == 2 )
	    printf("\r%s⌝   ", waitMsg);
	  if      (ite == 3 )
	    printf("\r%s⌟   ", waitMsg);
	  if      (ite == 4 )
	  {
	    printf("\r%s⌞   ", waitMsg);
	    ite = 0;
	  }
	  fflush(stdout);
	}
      }

      usleep(sleepMS);
      sem_getvalue(running_threads, &noTrd );

      PROF // Profiling  .
      {
	NV_RANGE_POP(); // Sleep
      }
    }

    if (ite >= 10 )
      printf("\n\n");

    PROF // Profiling  .
    {
      NV_RANGE_POP(); // Wait on CPU threads
    }

    return (ite);
  }

  return (0);
}

void* contextInitTrd(void* ptr)
{
  //long long* contextInit = (long long*)malloc(sizeof(long long));
  //*contextInit = 0;

  struct timeval start, end;
  gpuSpecs* gSpec = (gpuSpecs*)ptr;

  TIME // Start the timer  .
  {
    NV_RANGE_PUSH("Context");

    gettimeofday(&start, NULL);
  }

  initGPUs(gSpec);

  TIME // End the timer  .
  {
    NV_RANGE_POP(); // Context

    gettimeofday(&end, NULL);
    gSpec->nctxTime += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));
  }

  pthread_exit(&gSpec->nctxTime);

  return (NULL);
}

long long initCudaContext(gpuSpecs* gSpec)
{
  if (gSpec)
  {
    infoMSG(4, 4, "Creating context pthread for CUDA context initialisation.\n");

    int iret1 = pthread_create( &gSpec->cntxThread, NULL, contextInitTrd, (void*) gSpec);

    if ( iret1 )
    {
      struct timeval start, end;

      fprintf(stderr,"ERROR: Failed to initialise context tread. pthread_create() return code: %d.\n", iret1);
      gSpec->cntxThread = 0;

      TIME // Start the timer  .
      {
	gettimeofday(&start, NULL);

	NV_RANGE_PUSH("Context");
      }

      printf("Initializing CUDA context's\n");
      initGPUs(gSpec);

      TIME // End the timer  .
      {
	NV_RANGE_POP(); // Context

	gettimeofday(&end, NULL);
	gSpec->nctxTime += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));
      }
    }
  }

  return 0;
}

long long compltCudaContext(gpuSpecs* gSpec)
{
  if ( gSpec)
  {
    if ( gSpec->cntxThread )
    {
      infoMSG(4, 4, "Wait on CUDA context thread\n");

      PROF // Profiling  .
      {
	NV_RANGE_PUSH("Wait on context thread");
      }

      printf("Waiting for CUDA context initialisation complete ...");
      fflush(stdout);

      void *status;
      struct timespec ts;
      if (clock_gettime(CLOCK_REALTIME, &ts) == -1)
      {
	fprintf(stderr,"ERROR: Failed to get time.\n");
      }
      ts.tv_sec += 10;

      int rr = pthread_timedjoin_np(gSpec->cntxThread, &status, &ts);
      if ( rr )
      {
	fprintf(stderr,"ERROR: Failed to join context thread.\n");
	if ( pthread_kill(gSpec->cntxThread, SIGALRM) )
	{
	  fprintf(stderr,"ERROR: Failed to kill context thread.\n");
	}

	for ( int i = 0; i < gSpec->noDevices; i++)
	{
	  CUDA_SAFE_CALL(cudaSetDevice(gSpec->devId[i]), "ERROR in cudaSetDevice");
	  CUDA_SAFE_CALL(cudaDeviceReset(), "Error in device reset.");
	}

	exit(EXIT_FAILURE);
      }

      printf("\r                                                          ");
      fflush(stdout);

      gSpec->cntxThread = 0;

      infoMSG(4, 4, "Done\n");

      PROF // Profiling  .
      {
	NV_RANGE_POP(); // Wait on context thread
      }
    }

    return gSpec->nctxTime;
  }
  else
  {
    fprintf(stderr,"ERROR: Called %s with NULL pointer.\n", __FUNCTION__ );
  }

  return 0;
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

void setInMemPlane(cuSearch* cuSrch, ImPlane planePos)
{
  bool ker = false;
  double zStart, zEnd;

  if ( !(cuSrch->sSpec->flags & FLAG_Z_SPLIT) )
  {
    fprintf(stderr,"ERROR: Trying to set inmem plane wehrn not using plane split?\n");
    exit(EXIT_FAILURE);
  }

  // for the moment there shpuld only be one kernel!
  // Note we could split the inmem plane adross two devices!
  cuFFdotBatch* kernel	= &cuSrch->pInf->kernels[0];
  cuFFdotBatch* batch	= &cuSrch->pInf->batches[0];

  // Calculate the start and end z values
  for (int i = kernel->noSrchHarms; i > 0; i--)
  {
    cuHarmInfo* hInfs;
    int hIdx            = kernel->noSrchHarms-i;
    hInfs               = &kernel->hInfos[hIdx];                              // Harmonic index

    if ( planePos == IM_TOP )
    {
      zStart		= cu_calc_required_z<double>(1, 0.0, cuSrch->sSpec->zRes);
      zEnd		= cu_calc_required_z<double>(1, hInfs->zmax, cuSrch->sSpec->zRes);
    }
    else
    {
      zStart		= cu_calc_required_z<double>(1, 0.0, cuSrch->sSpec->zRes);
      zEnd		= cu_calc_required_z<double>(1, -hInfs->zmax, cuSrch->sSpec->zRes);
    }

    if ( hInfs->zStart != zStart || hInfs->zEnd != zEnd )
    {
      ker = true;
    }
    hInfs->zStart	= zStart;
    hInfs->zEnd		= zEnd;
    hInfs->noZ       	= round(fabs(hInfs->zEnd - hInfs->zStart) / cuSrch->sSpec->zRes) + 1;
  }

  FOLD // Set the sizes values of the harmonics and kernels and pointers to kernel data  .
  {
    setKernelPointers(kernel);
  }

  FOLD // Generate kernel values if neede  .
  {
    if ( ker )
    {
      printf("\nGenerating GPU multiplication kernels using device %i (%s).\n", kernel->gInf->devid, kernel->gInf->name);

      createBatchKernels(kernel, batch->d_planeMult);
    }
  }

  setConstVals( kernel );
  setConstVals_Fam_Order( kernel );                            // Constant values for multiply
}

void genPlane(cuSearch* cuSrch, char* msg)
{
  infoMSG(2,2,"Candidate generation\n");

  struct timeval start, end;
  double startr			= 0;
  int maxxx;
  cuFFdotBatch* master		= &cuSrch->pInf->kernels[0];   // The first kernel created holds global variables
  int   ss			= 0;
  int iteration 		= 0;

  TIME // Basic timing  .
  {
    NV_RANGE_PUSH("Pln Gen");
    gettimeofday(&start, NULL);
  }

  FOLD // Set the bounds of the search  .
  {
    // Search bounds
    maxxx	= MAX(cuSrch->SrchSz->noSteps, 0);

    if ( master->flags & FLAG_SS_INMEM  )
    {
      startr  = cuSrch->SrchSz->searchRLow ; // ie ( rlo / no harms)
    }
    else
    {
      startr  = cuSrch->sSpec->fftInf.rlo;
    }

    if ( msgLevel == 0 )
    {
      print_percent_complete(startr - startr, cuSrch->sSpec->fftInf.rhi - startr, msg, 1);
    }
    else
    {
      fflush(stdout);
      fflush(stderr);
      infoMSG(1,0,"\nGPU loop will process %i steps\n", maxxx);
    }
  }

#ifndef DEBUG 	// Parallel if we are not in debug mode  .
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

  FOLD  //					---===== Main Loop =====---  .
  {
    // These are all thread specific variables
    int tid = omp_get_thread_num();
    cuFFdotBatch* batch		= &cuSrch->pInf->batches[tid];				///< Thread specific batch to process
    int		rest		= batch->noSteps;
    int		firstStep	= 0;							///< Thread specific value for the first step the batch is processing
    int		step;

    // Set the device this thread will be using
    setDevice(batch->gInf->devid) ;

    FOLD // Set the r arrays to zero  .
    {
      rVals* rVal = (*batch->rAraays)[0][0];
      memset(rVal, 0, sizeof(rVals)*batch->noSteps);
    }

    while ( ss < maxxx )  //			---===== Main Loop =====---  .
    {
      FOLD // Calculate the step(s) to handle  .
      {
#pragma omp critical		// Calculate the step(s) this batch is processing  .

	FOLD
	{
	  FOLD  // Synchronous behaviour  .
	  {
#ifndef  DEBUG
	    if ( cuSrch->sSpec->flags & FLAG_SYNCH )
#endif
	    {
	      // If running in synchronous mode use multiple batches, just synchronously
	      tid = iteration % cuSrch->pInf->noBatches ;
	      batch = &cuSrch->pInf->batches[tid];
	      setDevice(batch->gInf->devid) ;			// Switch over to applicable device
	    }
	  }

	  iteration++;

	  firstStep = ss;
	  ss       += batch->noSteps;
	  cuSrch->noSteps++;

	  infoMSG(1,1,"\nStep %4i of %4i thread %02i processing %02i steps on GPU %i\n", firstStep+1, maxxx, tid, batch->noSteps, batch->gInf->devid );
	}

	if ( firstStep >= maxxx )
	  break;

	if ( firstStep + (int)batch->noSteps >= maxxx ) // End case (there is some overflow)  .
	{
	  // TODO: There are a number of steps we don't need to run see if we can use 'setplanePointers(trdBatch)'
	  // To see if we can do less work on the last step
	  rest = maxxx - firstStep;
	}
      }

      FOLD // Set start r-vals for all steps in this batch  .
      {
	for ( step = 0; step < (int)batch->noSteps ; step++ )
	{
	  rVals* rVal = &(*batch->rAraays)[0][step][0];

	  if ( step < rest )
	  {
	    rVal->drlo		= startr + (firstStep+step) * ( batch->accelLen / (double)cuSrch->sSpec->noResPerBin );
	    rVal->drhi		= rVal->drlo + ( batch->accelLen - 1 ) / (double)cuSrch->sSpec->noResPerBin;

	    for ( int harm = 0; harm < batch->noGenHarms; harm++)
	    {
	      rVal		= &(*batch->rAraays)[0][step][harm];
	      rVal->step	= firstStep + step;
	      rVal->norm	= 0.0;
	    }
	  }
	  else
	  {
	    rVal->drlo		= 0;
	    rVal->drhi		= 0;
	  }
	}
      }

      FOLD // Call the CUDA search  .
      {
	search_ffdot_batch_CU(batch);
      }

      FOLD // Print message  .
      {
	if ( msgLevel == 0  )
	{
	  if      ( master->flags & FLAG_SS_INMEM     )
	  {
	    printf("\r%s  %5.1f%%", msg, firstStep/(float)maxxx*100.0);
	  }
	  else
	  {
	    int noTrd;
	    sem_getvalue(&master->cuSrch->threasdInfo->running_threads, &noTrd );
	    printf("\r%s  %5.1f%% ( %3i Active CPU threads processing initial candidates)  ", msg, firstStep/(float)maxxx*100.0, noTrd);
	  }

	  fflush(stdout);
	}
      }
    }

    FOLD  // Finish off CUDA search  .
    {
      infoMSG(1,0,"\nFinish off search.\n");

      // Set r values to 0 so as to not process details  .
      for ( step = 0; step < (int)batch->noSteps ; step++)
      {
	rVals* rVal	= &(*batch->rAraays)[0][step][0];
	rVal->drlo	= 0;
	rVal->drhi	= 0;
      }

      // Finish searching the planes, this is required because of the out of order asynchronous calls
      for ( rest = 0 ; rest < batch->noRArryas; rest++ )
      {
	FOLD // Set the r arrays to zero  .
	{
	  rVals* rVal = (*batch->rAraays)[0][0];
	  memset(rVal, 0, sizeof(rVals)*batch->noSteps);
	}

	search_ffdot_batch_CU(batch);
      }

      // Wait for asynchronous execution to complete
      finish_Search(batch);
    }
  }

  printf("\r%s. %5.1f%%                                                                                         \n", msg, 100.0);

  FOLD // Wait for CPU threads to complete  .
  {
    waitForThreads(&master->cuSrch->threasdInfo->running_threads, "Waiting for CPU thread(s) to finish processing returned from the GPU ", 200 );
  }

  TIME // Basic timing  .
  {
    NV_RANGE_POP(); // Pln Gen

    gettimeofday(&end, NULL);
    cuSrch->timings[TIME_GPU_PLN] += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));
  }


}

cuSearch* searchGPU(cuSearch* cuSrch, gpuSpecs* gSpec, searchSpecs* sSpec)
{
  struct timeval start, end;
  struct timeval start01, end01;
  struct timeval start02, end02;
  cuFFdotBatch* master;
  long noCands = 0;

  // Wait for the context thread to complete, NOTE: cuSrch might not be initialised at this point?
  long long contextTime = compltCudaContext(gSpec);

  TIME // Basic timing  .
  {
    NV_RANGE_PUSH("GPU Srch");
    cuSrch->timings[TIME_CONTEXT] = contextTime;
  }

#ifdef NVVP // Start profiler
  cudaProfilerStart();              // Start profiling, only really necessary for debug and profiling, surprise surprise
#endif

  printf("\n*************************************************************************************************\n                         Doing GPU Search \n*************************************************************************************************\n");

  char srcTyp[1024];

  TIME // Basic timing of device setup and kernel creation  .
  {
    gettimeofday(&start, NULL);
  }

  FOLD // Init GPU kernels and planes  .
  {
    TIME // Basic timing of device setup and kernel creation  .
    {
      NV_RANGE_PUSH("GPU Initialise");
    }

    cuSrch    = initCuKernels(sSpec, gSpec, cuSrch);
    master    = &cuSrch->pInf->kernels[0];   // The first kernel created holds global variables

    TIME // Basic timing of device setup and kernel creation  .
    {
      NV_RANGE_POP();	// GPU Initialise

      gettimeofday(&end, NULL);
      cuSrch->timings[TIME_GPU_INIT] += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));
    }

    if ( master->flags & FLAG_SYNCH )
      fprintf(stderr, "WARNING: Running synchronous search, this will slow things down and should only be used for debug and testing.\n");
  }

  FOLD // Candidate generation  .
  {
    TIME // Basic timing  .
    {
      NV_RANGE_PUSH("Cand Gen");
      gettimeofday(&start01, NULL);
    }

    printf("\nRunning GPU search of %lli steps with %i simultaneous families of f-∂f planes spread across %i device(s).\n\n", cuSrch->SrchSz->noSteps, cuSrch->pInf->noSteps, cuSrch->pInf->noDevices );

    if      ( master->flags & FLAG_SS_INMEM     )	// In-mem search  .
    {
      if ( master->flags & FLAG_Z_SPLIT )		// Z-Split  .
      {
	setInMemPlane(cuSrch, IM_TOP);
	sprintf(srcTyp, "Generating top half in-mem GPU plane");
	genPlane(cuSrch, srcTyp);
	inmemSumAndSearch(cuSrch);

	setInMemPlane(cuSrch, IM_BOT);
	sprintf(srcTyp, "Generating top half in-mem GPU plane");
	genPlane(cuSrch, srcTyp);
	inmemSumAndSearch(cuSrch);
      }
      else						// Entire plane at once  .
      {
	sprintf(srcTyp, "Generating full in-mem GPU plane");
	genPlane(cuSrch, srcTyp);
	inmemSumAndSearch(cuSrch);
      }
    }
    else						// Standard search  .
    {
      sprintf(srcTyp, "GPU search");
      genPlane(cuSrch, srcTyp);
    }

    TIME // Basic timing  .
    {
      gettimeofday(&end01, NULL); // This is not used???
    }

    TIME // Basic timing  .
    {
      NV_RANGE_PUSH("GPU Cand");
      gettimeofday(&start02, NULL);
    }

    FOLD // Process candidates  .
    {
      infoMSG(1,1,"\nProcess candidates\n");

      if      ( master->cndType & CU_STR_ARR    ) // Copying candidates from array to list for optimisation  .
      {
	if ( !(master->flags & FLAG_DPG_SKP_OPT) )
	{
	  printf("\nCopying initial candidates from array to list for optimisation.\n");

	  PROF // Profiling  .
	  {
	    NV_RANGE_PUSH("Add to list");
	  }

	  int     cdx;
	  double  poww, sig;
	  double  rr, zz;
	  int     added = 0;
	  int     numharm;
	  initCand*   candidate = (initCand*)cuSrch->h_candidates;
	  poww    = 0;

	  for (cdx = 0; cdx < (int)cuSrch->SrchSz->noOutpR; cdx++)  // Loop
	  {
	    poww        = candidate[cdx].power;

	    if ( poww > 0 )
	    {
	      numharm   = candidate[cdx].numharm;
	      sig       = candidate[cdx].sig;
	      rr        = candidate[cdx].r;
	      zz        = candidate[cdx].z;

	      cuSrch->cands  = insert_new_accelcand(cuSrch->cands, poww, sig, numharm, rr, zz, &added );

	      noCands++;
	    }
	  }

	  PROF // Profiling  .
	  {
	    NV_RANGE_POP(); // Add to list
	  }
	}
      }
      else if ( master->cndType & CU_STR_LST    )
      {
	cuSrch->cands  = (GSList*)cuSrch->h_candidates;

	int bIdx;
	for ( bIdx = 0; bIdx < cuSrch->pInf->noBatches; bIdx++ )
	{
	  noCands += cuSrch->pInf->batches[bIdx].noResults;
	}

	if ( cuSrch->cands )
	{
	  if ( cuSrch->cands->data == NULL )
	  {
	    // No real candidates found!
	    cuSrch->cands = NULL;
	  }
	}
      }
      else if ( master->cndType & CU_STR_QUAD   ) // Copying candidates from array to list for optimisation  .
      {
	// TODO: write the code!

	fprintf(stderr, "ERROR: Quad-tree candidates has not yet been finalised for optimisation!\n");
	exit(EXIT_FAILURE);

	//candsGPU = testTest(master, candsGPU);
      }
      else
      {
	fprintf(stderr, "ERROR: Bad candidate storage method?\n");
	exit(EXIT_FAILURE);
      }
    }

    TIME // Basic timing  .
    {
      NV_RANGE_POP(); // GPU Cand
      NV_RANGE_POP(); // Cand Gen
      gettimeofday(&end02, NULL);
      cuSrch->timings[TIME_CND] += ((end02.tv_sec - start02.tv_sec) * 1e6 + (end02.tv_usec - start02.tv_usec));
      cuSrch->timings[TIME_GPU_CND_GEN] += ((end02.tv_sec - start01.tv_sec) * 1e6 + (end02.tv_usec - start01.tv_usec));
    }
  }

  FOLD // Free GPU memory  .
  {
    freeAccelGPUMem(cuSrch->pInf);
  }

  printf("\nGPU found %li initial candidates of which %i are unique", noCands, g_slist_length(cuSrch->cands));

  TIME // Basic timing  .
  {
    NV_RANGE_POP(); // GPU Srch
    gettimeofday(&end, NULL);

    cuSrch->timings[TIME_GPU_SRCHALL] += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));

    printf(", it Took %.4f ms", cuSrch->timings[TIME_GPU_SRCHALL]/1000.0);
  }

  printf(".\n");



  return cuSrch;
}
