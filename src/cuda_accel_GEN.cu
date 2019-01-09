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
 *  [0.0.02] [2017-01-31 18:50]
 *    Fixed more bugs in accel len calculation
 *    Caged the way profiling and timing happens, introduced the PROF macro
 *    Changed GPUDefaylts text values
 *    New better ordering for asynchronous & profiling standard search (faster overlap GPU and CPU)
 *    Added many more debug messages in initialisation routines
 *    Fixed bug in iFFT stream creation
 *
 *  [0.0.03] []
 *    Added a new fag to allow separate treatment of input and plane FFT's (separate vs single)
 *    Caged createFFTPlans to allow creating the FFT plans for input and plane separately
 *    Reorder stream creation in initKernel
 *    Synchronous runs now default to one batch and separate FFT's
 *    Added ZBOUND_NORM flag to specify bound to swap over to CPU input normalisation
 *    Added ZBOUND_INP_FFT flag to specify bound to swap over to CPU FFT's for input
 *    Added 3 generic debug flags ( FLAG_DPG_TEST_1, FLAG_DPG_TEST_2, FLAG_DPG_TEST_3 )
 *
 *  [0.0.04] [2017-02-01]
 *    Fixed a bug in the ordering of the process results component in - standard, synchronous mode
 *    Re-ordered things so sum & search slices uses output stride, this means in-mem now uses the correct auto slices for sum and search
 *
 *  [0.0.05] [2017-02-01]
 *    Converted candidate processing to use a circular buffer of results in pinned memory
 *    Added a function to zero r-array, it preserves pointer to pinned host memory
 *
 *  [0.0.03] [2017-02-05]
 *    Reorder in-mem async to slightly faster (3 way)
 *
 *  [0.0.03] [2017-02-10]
 *    Multi batch async fixed finishing off search
 *
 *  [0.0.03] [2017-02-16]
 *    Separated candidate and optimisation CPU threading
 *
 *  [0.0.03] [2017-02-24]
 *     Added preprocessor directives for steps and chunks
 *
 *  [0.0.03] [2017-03-04]
 *     Work on automatic step, batch and chunk selection
 *
 *  [0.0.03] [2017-03-09]
 *     Added slicing exit for testing
 *
 *  [0.0.03] [2017-03-25]
 *  Improved multiplication chunk handling
 *  Added temporary output of chunks and step size
 *  Clamp SAS chunks to SAS slice width
 *
 *  [2017-03-30]
 *  	Fix in-mem plane size estimation to be more accurate
 *  	Added function to calculate in-mem plane size
 *  	Re worked the search size data structure and the manner number of steps is calculated
 *  	Converted some debug messages sizes from GiB to GB and MiB to MB
 *  	Added separate candidate array resolution - Deprecating FLAG_STORE_EXP
 *
 *  [2017-04-17]
 *  	Fixed clipping of multiplication chunks back to max slice height
 *
 *  [2017-04-24]
 *  	Reworked calculating the y-index and added the setPlaneBounds function
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
#include "cuda_accel_GEN.h"
#include "cuda_accel_IN.h"
#include "cuda_accel_MU.h"
#include "cuda_cand_OPT.h"

__device__ __constant__ int		HEIGHT_HARM[MAX_HARM_NO];		///< Plane  height  in stage order
__device__ __constant__ int		STRIDE_HARM[MAX_HARM_NO];		///< Plane  stride  in stage order
__device__ __constant__ int		WIDTH_HARM[MAX_HARM_NO];		///< Plane  strides   in family
__device__ __constant__ void*		KERNEL_HARM[MAX_HARM_NO];		///< Kernel pointer in stage order
__device__ __constant__ int		KERNEL_OFF_HARM[MAX_HARM_NO];		///< The offset of the first row of each plane in their respective kernels
__device__ __constant__ stackInfo	STACKS[64];
__device__ __constant__ int		STK_STRD[MAX_STACKS];			///< Stride of the stacks
__device__ __constant__ char		STK_INP[MAX_STACKS][4069];		///< input details


void setActiveIteration(cuFFdotBatch* batch, int rIdx)
{
  if ( rIdx < 0  )
  {
    // Flip it to positive so it can index, in terms if iterations negatives make more sense
    // I was told in undergrad to be a defensive programmer ;)
    rIdx = -rIdx;
  }

  if ( rIdx >= batch->noRArryas )
  {
    fprintf(stderr,"ERROR: Index larger than ring buffer.\n");
    exit(EXIT_FAILURE);
  }

  batch->rActive = rIdx;
}

///////////////////////// IM-Stuff /////////////////////////////////////

/** Calculate the width of the in-memory plane
 *
 * @param minLen	The minimum length of the plane
 * @param genStride	The stride of steps of the generation stage
 * @param searchStride	The stride of steps of the search stage
 * @return
 */
size_t calcImWidth(size_t minLen, size_t genStride, size_t searchStride)
{
  size_t genX		= ceil( minLen / (double)genStride )    * genStride;				// Generation size
  size_t srchX		= ceil( minLen / (double)searchStride ) * searchStride;				// Max search size

  return genX;

  // Not necessary
  //return MAX(genX,srchX);;
}

/** Set the plane bounds for all harmonics
 *
 * Set the start stop and number of z values of each plane
 *
 * @param sSpec		Search Specifications
 * @param hInfs		Pointer to an array harmonic infos
 * @param noHarms	The number of harmonics in the array
 * @param planePos	What type of plane to index
 */
void setPlaneBounds(confSpecsGen* conf, cuHarmInfo* hInfs, int noHarms, ImPlane planePos)
{
  // Calculate the start and end z values
  for (int i = 0; i < noHarms; i++)
  {
    cuHarmInfo* hInf	= &hInfs[i];

    if      ( planePos == IM_FULL )
    {
      hInf->zStart	= cu_calc_required_z<double>(1, -hInf->zmax, conf->zRes);
      hInf->zEnd	= cu_calc_required_z<double>(1,  hInf->zmax, conf->zRes);
    }
    else if ( planePos == IM_TOP )
    {
      hInf->zStart	= cu_calc_required_z<double>(1,  0.0,        conf->zRes);
      hInf->zEnd	= cu_calc_required_z<double>(1,  hInf->zmax, conf->zRes);
    }
    else if ( planePos == IM_BOT )
    {
      hInf->zStart	= cu_calc_required_z<double>(1,  0.0,        conf->zRes);
      hInf->zEnd	= cu_calc_required_z<double>(1, -hInf->zmax, conf->zRes);
    }
    else
    {
      fprintf(stderr, "ERROR: invalid in-memory plane.\n" );
      exit(EXIT_FAILURE);
    }
    hInf->noZ       	= round(fabs(hInf->zEnd - hInf->zStart) / conf->zRes) + 1;

    infoMSG(6,6,"Harm: %2i  z: %7.2f to %7.2f  noZ %4i \n", i, hInf->zStart, hInf->zEnd, hInf->noZ );
  }
}

void setInMemPlane(cuSearch* cuSrch, ImPlane planePos)
{
  // for the moment there should only be one kernel!
  // Note we could split the inmem plane across two devices!
  cuFFdotBatch* kernel	= &cuSrch->pInf->kernels[0];
  cuFFdotBatch* batch	= &cuSrch->pInf->batches[0];

  if ( !(batch->flags & FLAG_Z_SPLIT) )
  {
    fprintf(stderr,"ERROR: Trying to set inmem plane when not using plane split?\n");
    exit(EXIT_FAILURE);
  }

  FOLD // Set the plane bounds  .
  {
    setPlaneBounds(cuSrch->conf->gen, kernel->hInfos, kernel->noSrchHarms, planePos  );
  }

  FOLD // Set the sizes values of the harmonics and kernels and pointers to kernel data  .
  {
    setKernelPointers(kernel);
  }

  FOLD // Generate kernel values if needed  .
  {
    printf("\nGenerating GPU multiplication kernels using device %i (%s).\n", kernel->gInf->devid, kernel->gInf->name);
    createBatchKernels(kernel, batch->d_planeMult);
  }

  setConstVals( kernel );					//
  setConstVals_Fam_Order( kernel );				// Constant values for multiply
}

bool checkIMposable(searchSpecs* sSpec, confSpecsGen* conf, gpuInf* gInf)
{
  size_t free, total;                           ///< GPU memory

  size_t kerSize      = 0;                      ///< Total size (in bytes) of all the data
  size_t batchSize    = 0;                      ///< Total size (in bytes) of all the data need by a single family (ie one step) excluding FFT temporary
  size_t fffTotSize   = 0;                      ///< Total size (in bytes) of FFT temporary memory
  size_t planeSize    = 0;                      ///< Total size (in bytes) of memory required independently of batch(es)
  float kerElsSZ      = 0;                      ///< The size of an element of the kernel
  float plnElsSZ      = 0;                      ///< The size of an element of the full in-mem ff plane

  long long imStride		= 0;

  presto_interp_acc  accuracy = LOWACC;

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
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP("Get Device");
    }
  }

  FOLD // Initialise some values  .
  {
    infoMSG(4,4,"Calculate some initial values.\n");

    FOLD // Determine the size of the elements of the planes  .
    {
      // Kernel element size
      if ( (conf->flags & FLAG_KER_DOUBFFT) || (conf->flags & FLAG_DOUBLE) )
      {
	kerElsSZ = sizeof(dcomplexcu);
      }
      else
      {
	kerElsSZ = sizeof(fcomplexcu);
      }

      // Half precision plane
      if ( conf->flags & FLAG_POW_HALF )
      {
#if CUDART_VERSION >= 7050
	plnElsSZ = sizeof(half);
	infoMSG(7,7,"in-mem - half precision powers \n");
#else
	plnElsSZ = sizeof(float);
	fprintf(stderr, "WARNING: Half precision can only be used with CUDA 7.5 or later! Reverting to single precision!\n");
	conf->flags &= ~FLAG_POW_HALF;
	infoMSG(7,7,"in-mem - single precision powers \n");
#endif
      }
      else
      {
	plnElsSZ = sizeof(float);
	infoMSG(7,7,"in-mem - single precision powers \n");
      }
    }

    FOLD // Kernel accuracy  .
    {
      if ( conf->flags & FLAG_KER_HIGH )
      {
	infoMSG(7,7,"High accuracy kernels\n");
	accuracy = HIGHACC;
      }
      else
      {
	accuracy = LOWACC;
	infoMSG(7,7,"Low accuracy kernels\n");
      }
    }

    FOLD // IM step size  Note this is the actual value used later on  .
    {
      if ( conf->ssStepSize <= 100 )
      {
	if ( conf->ssStepSize > 0 )
	  fprintf(stderr, "WARNING: In-mem plane search stride too small, try auto ( 0 ) or something larger than 100 say 16384 or 32768.\n");

	imStride = 32768; // TODO: I need to check for a good default

	infoMSG(7,7,"In-mem search step size automatically set to %i.\n", imStride);
      }
      else
      {
	imStride = conf->ssStepSize;
      }
    }
  }

  FOLD // See if this device could do a GPU in-mem search  .
  {
    infoMSG(4,4,"Checking if in-mem possible?\n");

    //int noSteps;
    int slices;

    // Initialise some variables used to calculate sizes
    kerSize     = 0;
    batchSize   = 0;
    fffTotSize  = 0;
    planeSize   = 0;

    int	plnY		= ceil(conf->zMax / conf->zRes ) + 1 ;	// This assumes we are splitting the inmem plane into a top and bottom section (FLAG_Z_SPLIT)
    size_t 	accelLen;		///< Size of steps
    float	pow2width;		///< width of the planes
    int 	halfWidth;		///< Kernel halfwidth
    float	memGeuss;
    int 	noGenHarms	= 1;	// This is what the in-mem search uses and is used below to calculate predicted values, this should get over written later

    FOLD // Calculate memory sizes  .
    {
      infoMSG(5,5,"Calculating memory guess.\n" );

      FOLD // Set some defaults  .
      {
	FOLD // Number of search slices  .
	{
	  slices = conf->ssSlices ? conf->ssSlices : 1 ;
	}

	FOLD // Plane width  .
	{
	  accelLen		= calcAccellen(conf->planeWidth, conf->zMax, noGenHarms, accuracy, conf->noResPerBin, conf->zRes, false); // Note: noGenHarms is 1
	  pow2width		= cu_calc_fftlen<double>(1, conf->zMax, accelLen, accuracy, conf->noResPerBin, conf->zRes);
	  halfWidth		= cu_z_resp_halfwidth<double>(conf->zMax, accuracy);
	}

	FOLD // Set the size of the search  .
	{
	  //	    if ( !cuSrch->sSpec )
	  //	    {
	  //	      cuSrch->sSpec = new searchSpecs;
	  //	    }
	  //	    memset(cuSrch->sSpec, 0, sizeof(searchSpecs));

	  setSrchSize(sSpec, halfWidth, noGenHarms); // Note: noGenHarms is 1
	}
      }

      FOLD // Kernel  .
      {
	kerSize		= pow2width * plnY * kerElsSZ;							// Kernel
	infoMSG(7,7,"split plane kernel size: Total: %.2f MB \n", kerSize*1e-6);
      }

      FOLD // Calculate "approximate" in-memory plane size  .
      {
	size_t imWidth		= calcImWidth(sSpec->noSearchR*conf->noResPerBin, accelLen,  imStride );
	planeSize		= imWidth * plnY * plnElsSZ;

	infoMSG(7,7,"split plane in-mem plane: %.2f GB - %i  ( %i x %i ) points at %i Bytes. \n", planeSize*1e-9, imWidth, plnY, plnElsSZ);
      }

      FOLD // Calculate the  "approximate" size of a single 1 step batch  .
      {
	float batchInp	= pow2width * sizeof(cufftComplex) * slices;					// Input
	float batchOut	= imStride  * sSpec->noHarmStages * sizeof(candPZs);				// Output
	float batchCpx	= pow2width * plnY * sizeof(cufftComplex);					// Complex plain
	float batchPow	= pow2width * plnY * plnElsSZ;							// Powers plain
	fffTotSize	= pow2width * plnY * sizeof(cufftComplex);					// FFT plan memory

	batchSize	= batchInp + batchOut + batchCpx + batchPow + fffTotSize ;

	infoMSG(7,7,"Batch sizes: Total: %.2f MB, Input: %.2f MB,  Complex: ~%.2f MB, FFT: ~%.2f MB, Powers: ~%.2f MB, Return: ~%.2f MB \n",
	    batchSize*1e-6,
	    batchInp*1e-6,
	    batchCpx*1e-6,
	    fffTotSize*1e-6,
	    batchPow*1e-6,
	    batchOut*1e-6 );
      }

      memGeuss = kerSize + batchSize + planeSize;

      infoMSG(6,6,"Free: %.3f GB  - Guess: %.3f GB - in-mem plane: %.2f GB - Kernel: ~%.2f MB - batch: ~%.2f MB \n",
	  free*1e-9,
	  memGeuss*1e-9,
	  planeSize*1e-9,
	  kerSize*1e-6,
	  batchSize*1e-6);
    }

    if ( memGeuss < free )
    {
      // We can do a in-mem search
      infoMSG(5,5,"In-mem is possible\n");

      return true;
    }
    else
    {
      // We can do a in-mem search
      infoMSG(5,5,"In-mem not possible\n");

      return false;
    }
  }
}

////////////////////////// Initialise  /////////////////////////////////


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
  float kerElsSZ      = 0;                      ///< The size of an element of the kernel
  float plnElsSZ      = 0;                      ///< The size of an element of the full in-mem ff plane
  float cmpElsSZ      = 0;                      ///< The size of an element of the kernel and complex plane
  float powElsSZ      = 0;                      ///< The size of an element of the powers plane

  gpuInf*	gInf		= &cuSrch->gSpec->devInfo[devID];
  int		noBatches	= cuSrch->gSpec->noDevBatches[devID];

  confSpecsGen*	conf		= cuSrch->conf->gen;

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

      int driverVersion = 0;
      int runtimeVersion = 0;
      CUDA_SAFE_CALL( cudaDriverGetVersion (&driverVersion),  "Failed to get driver version using cudaDriverGetVersion");
      CUDA_SAFE_CALL( cudaRuntimeGetVersion(&runtimeVersion), "Failed to get run time version using cudaRuntimeGetVersion");
      printf("  CUDA Runtime Version: %02d.%02d \n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
      printf("  CUDA Driver Version:  %02d.%02d \n", driverVersion / 1000, (driverVersion % 100) / 10);
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP("Get Device");
    }
  }

  FOLD // Allocate and zero some data structures  .
  {
    infoMSG(4,4,"Allocate and zero structures\n");

    FOLD // Initialise main pointer to this kernel  .
    {
      memset(kernel, 0, sizeof(cuFFdotBatch));

      if ( master != NULL )  // Copy all pointers and sizes from master. All non global pointers must be overwritten.
      {
	memcpy(kernel,  master,  sizeof(cuFFdotBatch));
	kernel->srchMaster	= 0;
      }
      else
      {
	kernel->flags		= conf->flags;
	kernel->srchMaster	= 1;
      }
    }

    FOLD // Set the device specific parameters  .
    {
      kernel->cuSrch		= cuSrch;
      kernel->gInf		= gInf;
      kernel->isKernel		= 1;                // This is the device master
    }
  }

  FOLD // Initialise some values  .
  {
    if ( master == NULL )
    {
      infoMSG(4,4,"Calculate some initial values.\n");

      FOLD // Determine the size of the elements of the planes  .
      {
	// Kernel element size
	if ( (kernel->flags & FLAG_KER_DOUBFFT) || (kernel->flags & FLAG_DOUBLE) )
	{
	  kerElsSZ = sizeof(dcomplexcu);
	}
	else
	{
	  kerElsSZ = sizeof(fcomplexcu);
	}

	// Set complex plane size - NOTE: Double hasn't been enabled yet
	if ( kernel->flags & FLAG_DOUBLE )
	{
	  cmpElsSZ = sizeof(dcomplexcu);
	}
	else
	{
	  cmpElsSZ = sizeof(fcomplexcu);
	}

	// Half precision plane
	if ( kernel->flags & FLAG_POW_HALF )
	{
#if CUDART_VERSION >= 7050
	  plnElsSZ = sizeof(half);
	  infoMSG(7,7,"in-mem - half precision powers \n");
#else
	  plnElsSZ = sizeof(float);
	  fprintf(stderr, "WARNING: Half precision can only be used with CUDA 7.5 or later! Reverting to single precision!\n");
	  kernel->flags &= ~FLAG_POW_HALF;
	  infoMSG(7,7,"in-mem - single precision powers \n");
#endif
	}
	else
	{
	  plnElsSZ = sizeof(float);
	  infoMSG(7,7,"in-mem - single precision powers \n");
	}

	// Set power plane size
	if ( kernel->flags & FLAG_CUFFT_CB_POW )
	{
	  // This should be the default
	  powElsSZ = plnElsSZ;
	}
	else
	{
	  powElsSZ = sizeof(fcomplexcu);
	}
      }

      FOLD // Kernel accuracy  .
      {
	if ( kernel->flags & FLAG_KER_HIGH )
	{
	  infoMSG(7,7,"High accuracy kernels\n");
	  accuracy = HIGHACC;
	}
	else
	{
	  accuracy = LOWACC;
	  infoMSG(7,7,"Low accuracy kernels\n");
	}
      }

      FOLD // IM step size  Note this is the actual value used later on  .
      {
	if ( conf->ssStepSize <= 100 )
	{
	  if ( conf->ssStepSize > 0 )
	    fprintf(stderr, "WARNING: In-mem plane search stride too small, try auto ( 0 ) or something larger than 100 say 16384 or 32768.\n");

	  kernel->strideOut = 32768; // TODO: I need to check for a good default

	  infoMSG(7,7,"In-mem search step size automatically set to %i.\n", kernel->strideOut);
	}
	else
	{
	  kernel->strideOut = conf->ssStepSize;
	}
      }
    }
  }

  FOLD // See if this device could do a GPU in-mem search  .
  {
    if ( master == NULL ) // For the moment lets try this on only the first card!
    {
      infoMSG(4,4,"Checking if in-mem possible?\n");

      int noSteps;
      int slices;

      // Initialise some variables used to calculate sizes
      kerSize     = 0;
      batchSize   = 0;
      fffTotSize  = 0;
      planeSize   = 0;

      int	plnY       = ceil(conf->zMax / conf->zRes ) + 1 ;	// This assumes we are splitting the inmem plane into a top and bottom section (FLAG_Z_SPLIT)
      size_t 	accelLen;		///< Size of steps
      float	pow2width;		///< width of the planes
      int 	halfWidth;		///< Kernel halfwidth
      float	memGeuss;

      FOLD // Calculate memory sizes  .
      {
	infoMSG(5,5,"Calculating memory guess.\n" );

	FOLD // Set some defaults  .
	{
	  kernel->noGenHarms = 1; // This is what the in-mem search uses and is used below to calculate predicted values, this should get over written later

	  FOLD // Number of steps  .
	  {
	    // To allow in-memory lets test with the minimum possible
	    noSteps = cuSrch->gSpec->noDevSteps[devID] ? cuSrch->gSpec->noDevSteps[devID] : MIN_STEPS;

	    // Clip to max and min compiled with
	    MAXX(noSteps, MIN_STEPS);
	    MINN(noSteps, MAX_STEPS);
	  }

	  FOLD // Number of search slices  .
	  {
	    slices = conf->ssSlices ? conf->ssSlices : 1 ;
	  }

	  FOLD // Plane width  .
	  {
	    accelLen		= calcAccellen(conf->planeWidth, conf->zMax, kernel->noGenHarms, accuracy, conf->noResPerBin, conf->zRes, false); // Note: noGenHarms is 1
	    pow2width		= cu_calc_fftlen<double>(1, conf->zMax, accelLen, accuracy, conf->noResPerBin, conf->zRes);
	    halfWidth		= cu_z_resp_halfwidth<double>(conf->zMax, accuracy);
	  }

	  FOLD // Set the size of the search  .
	  {
//	    if ( !cuSrch->sSpec )
//	    {
//	      cuSrch->sSpec = new searchSpecs;
//	    }
//	    memset(cuSrch->sSpec, 0, sizeof(searchSpecs));

	    setSrchSize(cuSrch->sSpec, halfWidth, kernel->noGenHarms); // Note: noGenHarms is 1
	  }
	}

	FOLD // Kernel  .
	{
	  kerSize		= pow2width * plnY * kerElsSZ;							// Kernel
	  infoMSG(7,7,"split plane kernel size: Total: %.2f MB \n", kerSize*1e-6);
	}

	FOLD // Calculate "approximate" in-memory plane size  .
	{
	  size_t imWidth	= calcImWidth(cuSrch->sSpec->noSearchR*conf->noResPerBin, accelLen,  kernel->strideOut );
	  planeSize		= imWidth * plnY * plnElsSZ;

	  infoMSG(7,7,"split plane in-mem plane: %.2f GB - %i  ( %i x %i ) points at %i Bytes. \n", planeSize*1e-9, imWidth*plnY, imWidth, plnY, plnElsSZ);
	}

	FOLD // Calculate the  "approximate" size of a single 1 step batch  .
	{
	  float batchInp	= pow2width * sizeof(cufftComplex) * slices;					// Input
	  float batchOut	= kernel->strideOut * cuSrch->noHarmStages * sizeof(candPZs);			// Output
	  float batchCpx	= pow2width * plnY * sizeof(cufftComplex);					// Complex plain
	  float batchPow	= pow2width * plnY * plnElsSZ;							// Powers plain
	  fffTotSize		= pow2width * plnY * sizeof(cufftComplex);					// FFT plan memory

	  batchSize		= batchInp + batchOut + batchCpx + batchPow + fffTotSize ;

	  infoMSG(7,7,"Batch sizes: Total: %.2f MB, Input: %.2f MB,  Complex: ~%.2f MB, FFT: ~%.2f MB, Powers: ~%.2f MB, Return: ~%.2f MB \n",
	      batchSize*1e-6,
	      batchInp*1e-6,
	      batchCpx*1e-6,
	      fffTotSize*1e-6,
	      batchPow*1e-6,
	      batchOut*1e-6 );
	}

	memGeuss = kerSize + batchSize + planeSize;

	infoMSG(6,6,"Free: %.3f GB  - Guess: %.3f GB - in-mem plane: %.2f GB - Kernel: ~%.2f MB - batch: ~%.2f MB \n",
	    free*1e-9,
	    memGeuss*1e-9,
	    planeSize*1e-9,
	    kerSize*1e-6,
	    batchSize*1e-6);
      }

      bool possIm = false;		//< In-mem is possible
      bool prefIm = false;		//< Weather automatic configuration prefers in-mem
      bool doIm   = false;		//< Whether to do an in-mem search

      if ( memGeuss < free )
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

	if ( !(kernel->flags & FLAG_SS_ALL) || (kernel->flags & FLAG_SS_INMEM) )
	{
	  printf("Device %i can do a in-mem GPU search.\n", gInf->devid);
	  printf("  There is %.2f GB free memory.\n  A split f-∂f plane requires ~%.2f GB and the workspace ~%.2f MB.\n", free*1e-9, planeSize*1e-9, familySz*1e-6 );
	}

	if ( (kernel->flags & FLAG_SS_INMEM) || ( prefIm && !(kernel->flags & FLAG_SS_ALL)) )
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

	if ( memGeuss < (free * 0.5) )
	{
	  if ( kernel->flags & FLAG_Z_SPLIT )
	    fprintf(stderr,"  WARNING: Opting to split the in-mem plane when you don't need to.\n");
	  else
	    printf("    No need to split the in-mem plane.\n");
	}
	else
	{
	  printf("    Have to split the in-mem plane.\n");
	  kernel->flags |= FLAG_Z_SPLIT;
	}

	kernel->flags |= FLAG_SS_INMEM ;

#if CUDART_VERSION >= 6050
	if ( !(kernel->flags & FLAG_CUFFT_CB_POW) )
	  fprintf(stderr,"  Warning: Doing an in-mem search with no CUFFT callbacks, this is not ideal.\n"); // It should be on by default the user must have disabled it
#else
	fprintf(stderr,"  Warning: Doing an in-mem search with no CUFFT callbacks, this is not ideal. Try upgrading to CUDA 6.5 or later.\n");
	kernel->flags &= ~FLAG_CUFFT_ALL;
#endif

#if CUDART_VERSION >= 7050
	if ( !(kernel->flags & FLAG_POW_HALF) )
	  fprintf(stderr,"  Warning: You could be using half precision.\n"); // They should be on by default the user must have disabled them
#else
	fprintf(stderr,"  Warning: You could be using half precision. Try upgrading to CUDA 7.5 or later.\n");
#endif

	FOLD // Set types  .
	{
	  // NOTE: Change the original configuration
	  conf->retType &= ~CU_TYPE_ALLL;
	  conf->retType |= CU_POWERZ_S;

	  conf->retType &= ~CU_SRT_ALL;
	  conf->retType |= CU_STR_ARR;
	}
      }
      else
      {
	// No In-mem
	infoMSG(5,5,"Not doing in-mem\n");

	if ( kernel->flags & FLAG_SS_INMEM  )
	{
	  // Warning
	  fprintf(stderr,"ERROR: Requested an in-memory GPU search, this is not possible.\n\tThere is %.2f GB of free memory.\n\tIn-mem (split plane) GPU search would require ~%.2f GB\n\n", free*1e-9, (planeSize + familySz)*1e-9 );
	}
	else if ( possIm & !prefIm )
	{
	  printf("  Opting to not do an in-mem search.\n");
	}
	else if ( prefIm )
	{
	  // "Should" do a IM search

	  if ( (kernel->flags & FLAG_SS_ALL) )
	  {
	    fprintf(stderr,"WARNING: Opting to NOT do a in-mem search when you could!\n");
	  }
	}

	kernel->flags &= ~FLAG_SS_INMEM ;
	kernel->flags &= ~FLAG_Z_SPLIT;
      }

      // Sanity check
      if ( !(kernel->flags & FLAG_SS_ALL) )
      {
	// Default to S&S 3.1
	kernel->flags |= FLAG_SS_31;
	kernel->flags |= FLAG_STAGES;
      }

      // Reset sizes as they will need to be set to the actual values
      kerSize     = 0;
      batchSize   = 0;
      fffTotSize  = 0;
      planeSize   = 0;

      printf("\n");
    }

    FOLD // Set kernel values from cuSrch values  .
    {
      kernel->noHarmStages	= cuSrch->noHarmStages;
      kernel->noGenHarms	= cuSrch->noGenHarms;
      kernel->noSrchHarms	= cuSrch->noSrchHarms;
      kernel->conf		= duplicate(conf);
    }
  }

  FOLD // Do a global sanity check on Flags and CUDA version  .
  {
    // TODO: Do a check whether there is enough precision in an int to store the index of the largest point

    if ( master == NULL ) // For the moment lets try this on only the first card!
    {
      infoMSG(4,4,"Sanity check on some settings.\n");

      // CUFFT callbacks
#if CUDART_VERSION < 6050
      kernel->flags &= ~FLAG_CUFFT_ALL;
#endif

      if ( (kernel->flags & FLAG_POW_HALF) && !(kernel->flags & FLAG_SS_INMEM) && !(kernel->flags & FLAG_CUFFT_CB_POW) )
      {
#if CUDART_VERSION >= 7050
	fprintf(stderr, "WARNING: Can't use half precision with out of memory search and no CUFFT callbacks. Reverting to single precision!\n");
#endif
	kernel->flags &= ~FLAG_POW_HALF;
      }

      if ( !(kernel->flags & FLAG_SS_INMEM) && (kernel->flags & FLAG_CUFFT_CB_INMEM) )
      {
	fprintf(stderr, "WARNING: Can't use inmem callback with out of memory search. Disabling in-mem callback.\n");
	kernel->flags &= ~FLAG_CUFFT_CB_INMEM;
      }

      if ( (kernel->flags & FLAG_CUFFT_CB_POW) && (kernel->flags & FLAG_CUFFT_CB_INMEM) )
      {
	fprintf(stderr, "WARNING: in-mem CUFFT callback will supersede power callback, I have found power callbacks to be the best.\n");
	kernel->flags &= ~FLAG_CUFFT_CB_POW;
      }

      if ( (kernel->flags & FLAG_SS_31) || (kernel->flags & FLAG_SS_INMEM) )
      {
	kernel->flags |= FLAG_STAGES;
      }

      if ( !(kernel->flags & FLAG_SS_INMEM) && (kernel->flags & FLAG_Z_SPLIT) )
      {
	infoMSG(5,5,"Disabling in-mem plane split\n");

	kernel->flags &= ~FLAG_Z_SPLIT;
      }

      if ( !(kernel->flags & FLAG_CAND_THREAD) && (kernel->flags & FLAG_CAND_MEM_PRE) )
      {
	infoMSG(5,5,"Disable separate candidate memory (sequential candidates).\n");

	kernel->flags &= ~FLAG_CAND_MEM_PRE;
      }

      FOLD // Print some user output  .
      {
	char typeS[1024];
	sprintf(typeS, "Doing");

	if ( kernel->flags & FLAG_SS_INMEM )
	{
	  sprintf(typeS, "%s a in-memory search", typeS);
	  if ( kernel->flags & FLAG_Z_SPLIT )
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
	if ( kernel->flags & FLAG_POW_HALF )
	  sprintf(typeS, "%s half", typeS);
	else
	  sprintf(typeS, "%s single", typeS);

	sprintf(typeS, "%s precision powers", typeS);
	if ( kernel->flags & FLAG_CUFFT_CB_POW )
	  sprintf(typeS, "%s and CUFFT callbacks to calculate powers.", typeS);
	else if ( kernel->flags & FLAG_CUFFT_CB_INMEM )
	  sprintf(typeS, "%s and CUFFT callbacks to calculate powers and store in the full plane.", typeS);
	else
	  sprintf(typeS, "%s and no CUFFT callbacks.", typeS);

	printf("%s\n\n", typeS);
      }
    }
  }

  FOLD // Determine step size and how many stacks and how many planes in each stack  .
  {
    FOLD // Allocate memory  .
    {
      kernel->hInfos		= (cuHarmInfo*) malloc(kernel->noSrchHarms * sizeof(cuHarmInfo));
      kernel->kernels		= (cuKernel*)   malloc(kernel->noGenHarms  * sizeof(cuKernel));

      // Zero memory for kernels and harmonics
      memset(kernel->hInfos,  0, kernel->noSrchHarms * sizeof(cuHarmInfo));
      memset(kernel->kernels, 0, kernel->noGenHarms  * sizeof(cuKernel));
    }

    if ( master == NULL ) 	// Calculate details for the batch  .
    {
      FOLD // Determine step size  .
      {
	bool devisSS = false;						//< Weather to make the step size divisible for SS10 kernel

	printf("Determining GPU step size and plane width:\n");
	infoMSG(4,4,"Determining step size and width\n");

	FOLD // Get step size  .
	{
	  if ( !(kernel->flags & FLAG_SS_INMEM) || (kernel->flags & CU_NORM_GPU ) )
	  {
	    // Standard Sum & search kernel (SS31) requires divisible
	    // GPU normalisation requires all steps to have the same width, which requires divisibility
	    devisSS = true;
	  }

	  kernel->accelLen = calcAccellen(conf->planeWidth, conf->zMax, kernel->noGenHarms, accuracy, conf->noResPerBin, conf->zRes, devisSS);
	}

	FOLD // Print kernel accuracy  .
	{
	  printf(" • Using ");

	  if ( accuracy == HIGHACC )
	  {
	    printf("high ");
	  }
	  else
	  {
	    printf("standard ");
	  }
	  printf("accuracy response functions.\n");

	  if ( kernel->flags & FLAG_KER_MAX )
	    printf(" • Using maximum response function length for entire kernel.\n");
	}

	if ( kernel->accelLen > 100 ) // Print output  .
	{
	  double ratio	= 1;
	  double fftLen	= cu_calc_fftlen<double>(1, conf->zMax, kernel->accelLen, accuracy, conf->noResPerBin, conf->zRes);
	  double fWidth;
	  int oAccelLen;

	  if ( conf->planeWidth > 100 ) // User specified step size, check how close to optimal it is  .
	  {
	    double l2	= log2( fftLen ) - 10 ;
	    fWidth	= pow(2, l2);
	    oAccelLen	= calcAccellen(fWidth, conf->zMax, kernel->noGenHarms, accuracy, conf->noResPerBin, conf->zRes, devisSS);
	    ratio	= kernel->accelLen/double(oAccelLen);
	  }

	  printf(" • Using max plane width of %.0f and", fftLen);

	  if ( ratio < 1 )
	  {
	    printf(" a suboptimal step-size of %i. (%.2f%% of optimal) \n",  kernel->accelLen, ratio*100 );
	    printf("   > For a zmax of %.1f using %.0f K FFTs the optimal step-size is %i.\n", conf->zMax, fWidth, oAccelLen);

	    if ( conf->planeWidth > 100 )
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
	  fprintf(stderr,"ERROR: With a width of %i, the step-size would be %i and this is too small, try with a wider width or lower z-max.\n", conf->planeWidth, kernel->accelLen);
	  exit(EXIT_FAILURE);
	}
      }

      FOLD // Set some harmonic related values  .
      {
	int prevWidth		= 0;
	int noStacks		= 0;
	int stackHW		= 0;
	int hIdx, sIdx;
	double hFrac;

	FOLD // Set up basic details of all the harmonics  .
	{
	  infoMSG(4,4,"Determine number of stacks and planes\n");

	  for (int i = kernel->noSrchHarms; i > 0; i--)
	  {
	    cuHarmInfo* hInfs;
	    hFrac		= (i) / (double)kernel->noSrchHarms;
	    hIdx		= kernel->noSrchHarms-i;
	    hInfs		= &kernel->hInfos[hIdx];                              // Harmonic index

	    hInfs->harmFrac	= hFrac;
	    hInfs->zmax		= cu_calc_required_z<double>(hInfs->harmFrac, conf->zMax, conf->zRes);
	    hInfs->width	= cu_calc_fftlen<double>(hInfs->harmFrac, kernel->hInfos[0].zmax, kernel->accelLen, accuracy, conf->noResPerBin, conf->zRes);
	    hInfs->halfWidth	= cu_z_resp_halfwidth<double>(hInfs->zmax, accuracy);
	    hInfs->noResPerBin	= conf->noResPerBin;

	    if ( prevWidth != hInfs->width )	// Stack creation and checks
	    {
	      infoMSG(5,5,"New stack\n");

	      // We have a new stack
	      noStacks++;

	      if ( hIdx < kernel->noGenHarms )
	      {
		kernel->noStacks	= noStacks;
	      }

	      noInStack[noStacks - 1]	= 0;
	      prevWidth			= hInfs->width;
	      stackHW			= cu_z_resp_halfwidth<double>(hInfs->zmax, accuracy);

	      // Maximise, centre and align halfwidth
	      int   sWidth	= (int) ( ceil(kernel->accelLen * hInfs->harmFrac / (double)conf->noResPerBin ) * conf->noResPerBin ) + 1 ;		// Width of usable data for this plane
	      float centHW	= (hInfs->width  - sWidth)/2.0/(double)conf->noResPerBin;								//
	      float noAlg	= gInf->alignment / float(sizeof(fcomplex)) / (double)conf->noResPerBin ;						// halfWidth will be multiplied by ACCEL_NUMBETWEEN so can divide by it here!
	      float centAlgnHW	= floor(centHW/noAlg) * noAlg ;												// Centre and aligned half width

	      if ( stackHW > centAlgnHW )
	      {
		stackHW		= floor(centHW);

		infoMSG(6,6,"can not align stack half width GPU value. Using %i \n", stackHW );
	      }
	      else
	      {
		stackHW		= centAlgnHW;

		infoMSG(6,6,"aligned stack half width for GPU is %i \n", stackHW );
	      }
	    }

	    infoMSG(6,6,"Harm: %2i  frac %5.3f  z-max: %5.1f  z: %7.2f to %7.2f  width: %5i half width %4i \n", i, hFrac, hInfs->zmax, hInfs->zStart, hInfs->zEnd, hInfs->width, hInfs->halfWidth );

	    hInfs->stackNo	= noStacks-1;

	    if ( kernel->flags & FLAG_CENTER )
	    {
	      hInfs->kerStart	= stackHW*conf->noResPerBin;
	    }
	    else
	    {
	      hInfs->kerStart	= hInfs->halfWidth*conf->noResPerBin;
	    }

	    if ( hIdx < kernel->noGenHarms )
	    {
	      noInStack[noStacks - 1]++;
	    }
	  }
	}

	FOLD // Set the plane bounds  .
	{
	  if ( kernel->flags & FLAG_Z_SPLIT )
	  {
	    setPlaneBounds(conf, kernel->hInfos, kernel->noSrchHarms, IM_TOP  );
	  }
	  else
	  {
	    setPlaneBounds(conf, kernel->hInfos, kernel->noSrchHarms, IM_FULL );
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
	      cuSrch->sIdx[sIdx]		= hIdx; // TODO: Move this

	      infoMSG(6,6,"Fraction: %5.3f ( %2i/%2i ), Harmonic idx %2i, Stage idx %2i \n", hFrac, harm, harmtosum, hIdx, sIdx );
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
	  cStack->stackIdx	= i;
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
	  cStack->strideCmplx =   getStride(cStack->width, cmpElsSZ, gInf->alignment);
	  cStack->stridePower =   getStride(cStack->width, powElsSZ, gInf->alignment);

	  kernel->inpDataSize +=  cStack->strideCmplx * cStack->noInStack * sizeof(cufftComplex);
	  kernel->kerDataSize +=  cStack->strideCmplx * cStack->kerHeigth * cmpElsSZ;
	  kernel->plnDataSize +=  cStack->strideCmplx * cStack->height    * cmpElsSZ;

	  if ( !(kernel->flags & FLAG_CUFFT_CB_INMEM) )
	    kernel->pwrDataSize +=  cStack->stridePower * cStack->height  * powElsSZ;
	}
      }
    }
  }

  FOLD // Batch initialisation streams  .
  {
    PROF // Profiling  .
    {
      NV_RANGE_PUSH("init streams");
    }

    infoMSG(4,4,"Batch initialisation streams\n");

    char strBuff[1024];

    if ( kernel->flags & FLAG_SYNCH )	// Only one stream  .
    {
      cuFfdotStack* fStack = &kernel->stacks[0];

      CUDA_SAFE_CALL(cudaStreamCreate(&fStack->initStream),"Creating CUDA stream for initialisation");

      PROF // Profiling, name stream  .
      {
	sprintf(strBuff,"%i.0.0.0 Initialisation", kernel->gInf->devid );
	NV_NAME_STREAM(fStack->initStream, strBuff);
      }

      for (int i = 0; i < kernel->noStacks; i++)
      {
	cuFfdotStack* cStack	= &kernel->stacks[i];
	cStack->initStream	= fStack->initStream;
      }
    }
    else				// Separate streams  .
    {
      for (int i = 0; i < kernel->noStacks; i++)
      {
	cuFfdotStack* cStack = &kernel->stacks[i];

	CUDA_SAFE_CALL(cudaStreamCreate(&cStack->initStream),"Creating CUDA stream for initialisation");

	PROF // Profiling, name stream  .
	{
	  sprintf(strBuff,"%i.0.0.%i Initialisation", kernel->gInf->devid, i);
	  NV_NAME_STREAM(cStack->initStream, strBuff);
	}
      }
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP("streams");
    }
  }

  FOLD // Allocate device memory for all the multiplication kernels data  .
  {
    PROF // Profiling  .
    {
      NV_RANGE_PUSH("kernel malloc");
    }

    infoMSG(4,4,"Allocate device memory for all the kernels data %.2f MB.\n", kernel->kerDataSize * 1e-6 );

    if ( kernel->kerDataSize > free )
    {
      fprintf(stderr, "ERROR: Not enough device memory for GPU multiplication kernels. There is only %.2f MB free and you need %.2f MB \n", free * 1e-6, kernel->kerDataSize * 1e-6 );
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
      NV_RANGE_POP("kernel malloc");
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
	float contamination = (kernel->hInfos->halfWidth*2*conf->noResPerBin)/(float)kernel->hInfos->width*100 ;
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

	  float contamination = (cStack->harmInf->halfWidth*2*conf->noResPerBin)/(float)cStack->harmInf->width*100 ;
	  float padding       = (1-(kernel->accelLen*cStack->harmInf->harmFrac + cStack->harmInf->halfWidth*2*conf->noResPerBin ) / (float)cStack->harmInf->width)*100.0 ;

	  printf("  ■ Stack %i has %02i f-∂f plane(s). width: %5li  stride: %5li  Height: %6li  Memory size: %7.1f MB \n", i+1, cStack->noInStack, cStack->width, cStack->strideCmplx, cStack->height, cStack->height*cStack->strideCmplx*sizeof(fcomplex)*1e-6);

	  printf("    ► Created kernel %i  Size: %7.1f MB  Height %4lu   Contamination: %5.2f %%  Padding: %5.2f %%\n", i+1, cStack->harmInf->noZ*cStack->strideCmplx*sizeof(fcomplex)*1e-6, cStack->harmInf->noZ, contamination, padding);

	  for (int j = 0; j < cStack->noInStack; j++)
	  {
	    printf("      • Harmonic %02i  Fraction: %5.3f   Z-Max: %6.1f   Half Width: %4i  Start offset: %4i \n", hh, cStack->harmInf[j].harmFrac, cStack->harmInf[j].zmax, cStack->harmInf[j].halfWidth, cStack->harmInf[j].kerStart / conf->noResPerBin  );
	    hh++;
	  }
	}
      }

      if ( !( (kernel->flags & FLAG_SS_INMEM) && (kernel->flags & FLAG_Z_SPLIT) ) )
      {
	// In-mem kernels are created separately (top and bottom)
	printf("\nGenerating GPU multiplication kernels using device %i (%s).\n\n", kernel->gInf->devid, kernel->gInf->name);
	createBatchKernels(kernel, NULL);
      }
    }
  }

  FOLD // Decide how to handle input and output and allocate required memory  .
  {
    infoMSG(4,4,"Input and output.\n");

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("data");
    }

    printf("\nInitializing GPU %i (%s)\n", kernel->gInf->devid, gInf->name );

    if ( master != NULL )	// Copy the kernels  .
    {
      infoMSG(4,4,"Copy multiplication kernels\n");

      // TODO: Check this works in this location
      printf("• Copying multiplication kernels from device %i.\n", master->gInf->devid);
      CUDA_SAFE_CALL(cudaMemcpyPeerAsync(kernel->d_kerData, kernel->gInf->devid, master->d_kerData, master->gInf->devid, master->kerDataSize, master->stacks->initStream ), "Copying multiplication kernels between devices.");
    }

    ulong freeRam;		/// The amount if free host memory
    int retSZ     = 0;		/// The size in byte of the returned data
    int candSZ    = 0;		/// The size in byte of the candidates
    int retY      = 0;		/// The number of candidates return per family (one step)
    ulong hostC   = 0;		/// The size in bytes of device memory used for candidates

    FOLD // Check defaults and auto selection on CPU input FFT's  .
    {
      if ( conf->inputNormzBound >= 0 )
      {
	if ( conf->zMax >= conf->inputNormzBound )
	{
	  infoMSG(5,5,"Auto selecting CPU input normalisation.\n");
	  kernel->flags &= ~CU_NORM_GPU;
	}
	else
	{
#ifdef WITH_NORM_GPU
	  infoMSG(5,5,"Auto selecting GPU input normalisation.\n");
	  kernel->flags |= CU_NORM_GPU_SM;
#endif
	}
      }

      if ( conf->inputFFFTzBound >= 0 )
      {
	if ( conf->zMax >= conf->inputFFFTzBound )
	{
	  infoMSG(5,5,"Auto selecting CPU input FFT and normalisation.\n");
	  kernel->flags |= CU_INPT_FFT_CPU;
	  kernel->flags &= ~CU_NORM_GPU;
	}
	else
	{
	  infoMSG(5,5,"Auto selecting GPU input FFT's.\n");
	  kernel->flags &= ~CU_INPT_FFT_CPU;
	}
      }

      FOLD // Output  .
      {
	char  inpType[1024];
	sprintf(inpType, "• Using ");
	if ( kernel->flags & CU_NORM_GPU )
	  sprintf(inpType, "%s%s", inpType, "GPU ");
	else
	  sprintf(inpType, "%s%s", inpType, "CPU ");
	sprintf(inpType, "%s%s", inpType, "normalisationa and ");
	if ( kernel->flags & CU_INPT_FFT_CPU )
	  sprintf(inpType, "%s%s", inpType, "CPU ");
	else
	  sprintf(inpType, "%s%s", inpType, "GPU ");
	sprintf(inpType, "%s%s", inpType, "FFT's for input.\n");

	printf("%s",inpType);
      }
    }

    printf("• Examining GPU memory of device %2i:\n", kernel->gInf->devid);

    FOLD // Calculate the search size in bins  .
    {
      if ( master == NULL )
      {
	int alignemen = 1;

	if ( kernel->flags & FLAG_SS_31 )
	  alignemen = kernel->noGenHarms;

	setSrchSize(cuSrch->sSpec, kernel->hInfos->halfWidth, kernel->noGenHarms, alignemen);

	if ( (kernel->flags & FLAG_STORE_ALL) && !( kernel->flags  & FLAG_STAGES) )
	{
	  printf("   Storing all results implies returning all results so adding FLAG_STAGES to flags!\n");
	  kernel->flags  |= FLAG_STAGES;
	}
      }
    }

    FOLD // Calculate candidate type  .
    {
      if ( master == NULL )   // There is only one list of candidates per search so only do this once!
      {
	kernel->cndType         = conf->cndType;

	if      ( !(kernel->cndType & CU_TYPE_ALLL) )
	{
	  fprintf(stderr,"Warning: No candidate data type specified in %s. Setting to default.\n",__FUNCTION__);
	  kernel->cndType = CU_CANDFULL;
	}

	if      (kernel->cndType & CU_CANDFULL )		// Default
	{
	  candSZ = sizeof(initCand);
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
	else if (kernel->cndType & CU_CMPLXF   )
	{
	  candSZ = sizeof(fcomplexcu);
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
     kernel->retType       = conf->retType;

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
      }

      if      (kernel->retType & CU_POWERZ_S  )		// Default
      {
	infoMSG(7, 7, "SAS will return candPZs ( a float and short)");
	retSZ = sizeof(candPZs);	// I found that this auto aligns to 8 bytes, which is good for alignment bad(ish) for size
      }
      else if (kernel->retType & CU_POWERH_S  )
      {
	infoMSG(7, 7, "SAS will return candHs ( a half and short)");
	retSZ = sizeof(candHs);
      }
      else if (kernel->retType & CU_CMPLXF    )
      {
	retSZ = sizeof(fcomplexcu);
      }
      else if (kernel->retType & CU_INT       )
      {
	retSZ = sizeof(int);
      }
      else if (kernel->retType & CU_HALF      )
      {
#if CUDART_VERSION >= 7050
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

      FOLD // Return data structure  .
      {
	if      (  kernel->flags & FLAG_SS_INMEM )
	{
	  // NOTE: The in-mem sum and search does not need the search step size to be divisible by number of harmonics
	  // StrideOut has already been set to cuSrch->sSpec->ssStepSize
	}
	else
	{
	  if      ( (kernel->retType & CU_STR_ARR) || (kernel->retType & CU_STR_LST) || (kernel->retType & CU_STR_QUAD) )
	  {
	    // Standard search so generation and search are the same step size
	    kernel->strideOut = kernel->accelLen;
	  }
	  else if (  kernel->retType & CU_STR_PLN  )
	  {
	    // This isn't really used anymore

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

	  FOLD // Make sure the stride is aligned  .
	  {
	    int iStride = kernel->strideOut;
	    kernel->strideOut = getStride(kernel->strideOut, retSZ, gInf->alignment);

	    infoMSG(6,6,"Return size %i, elements: %i   initial stride: %i   aligned stride1: %i", retSZ, kernel->accelLen, iStride, kernel->strideOut );
	  }
	}
      }

      FOLD // Chunks and Slices  .
      {
	FOLD // Multiplication defaults are set per batch  .
	{
	  kernel->mulSlices		= conf->mulSlices;
	  kernel->mulChunk		= conf->mulChunk;

	  FOLD // Set stack multiplication slices  .
	  {
	    for (int i = 0; i < kernel->noStacks; i++)
	    {
	      cuFfdotStack* cStack	= &kernel->stacks[i];
	      cStack->mulSlices		= conf->mulSlices;
	      cStack->mulChunk		= conf->mulChunk;
	    }
	  }
	}

	FOLD // Sum & search  .
	{
	  kernel->ssChunk		= conf->ssChunk;
	  kernel->ssSlices		= conf->ssSlices;
	  kernel->ssColumn		= conf->ssColumn;

	  if ( kernel->ssSlices <= 0 )
	  {
	    size_t ssWidth		= kernel->strideOut;

	    if      ( ssWidth <= 1024 )
	    {
	      kernel->ssSlices		= 8 ;
	    }
	    else if ( ssWidth <= 2048 )
	    {
	      kernel->ssSlices		= 4 ;
	    }
	    else if ( ssWidth <= 4096 )
	    {
	      kernel->ssSlices		= 2 ;
	    }
	    else
	    {
	      kernel->ssSlices		= 1 ;
	    }
	  }
	  kernel->ssSlices		= MIN(kernel->ssSlices, ceil(kernel->hInfos->noZ/20.0) );		// TODO make this 20 a configurable parameter

	  infoMSG(5,5,"Sum & Search slices set to %i ", kernel->ssSlices);
	}
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

      // Calculate return data size for one step
      kernel->cndDataSize   = retY*kernel->strideOut*retSZ;

      if ( kernel->flags & FLAG_STAGES )
	kernel->cndDataSize *= kernel->noHarmStages;

      infoMSG(6,6,"retSZ: %i  alignment: %i  strideOut: %i  cndDataSize: ~%.2f MB\n", retSZ, kernel->gInf->alignment, kernel->strideOut, kernel->cndDataSize*1e-6);
    }

    FOLD // Calculate batch size and number of steps and batches on this device  .
    {
      infoMSG(4,4,"No Steps and batches.\n");

      PROF // Profiling  .
      {
	NV_RANGE_PUSH("Calc steps");
      }

      CUDA_SAFE_CALL(cudaMemGetInfo ( &free, &total ), "Getting Device memory information"); // TODO: This call may not be necessary we could calculate this from previous values

      freeRam = getFreeRamCU();

      printf("   There is a total of %.2f GB of device memory, %.2f GB is free. There is %.2f GB free host memory.\n",total*1e-9, (free)*1e-9, freeRam*1e-9 );

      FOLD // Calculate size of various memories for a single step batch  .
      {
	batchSize		= kernel->inpDataSize + kernel->plnDataSize + kernel->pwrDataSize + kernel->cndDataSize;  // This is currently the size of one step
	fffTotSize		= kernel->inpDataSize + kernel->plnDataSize;                                              // FFT data treated separately because there may be only one set per device
	kerSize			= kernel->kerDataSize;
	familySz		= kerSize + batchSize + fffTotSize;

	infoMSG(5,5,"batch: %.3f MB - inpDataSize: %.2f MB - plnDataSize: ~%.2f MB - pwrDataSize: ~%.2f MB - cndDataSize: ~%.2f MB \n",
	    batchSize*1e-6,
	    kernel->inpDataSize*1e-6,
	    kernel->plnDataSize*1e-6,
	    kernel->pwrDataSize*1e-6,
	    kernel->cndDataSize*1e-6 );
      }

      infoMSG(6,6,"Free: %.3f GB  - in-mem plane: %.2f GB - Kernel: %.2f MB - batch: %.2f MB - fft: %.2f MB - full batch: %.2f MB  - %i \n", free*1e-9, planeSize*1e-9, kerSize*1e-6, batchSize*1e-6, fffTotSize*1e-6, familySz*1e-6, batchSize );

      FOLD // Calculate how many batches and steps to do  .
      {
	FOLD // No steps possible for given number of batches
	{
	  float	possSteps[MAX_BATCHES];
	  bool	trySomething = 0;
	  int	targetSteps = 0;
	  int	noSteps;

	  // Reset # steps and batches, steps at least was changed previously
	  noBatches		= cuSrch->gSpec->noDevBatches[devID];
	  noSteps		= cuSrch->gSpec->noDevSteps[devID];

	  FOLD // Check synchronisation  .
	  {
	    if ( kernel->flags & FLAG_SYNCH  )		// Synchronous behaviour  .
	    {
	      if ( noBatches == 0 )			// NOTE: This can be overridden by forcing the number of batches
	      {
		printf("     Synchronous run so auto selecting 1 batch using separate FFT behaviour.\n");
		noBatches = 1;
		kernel->flags |= CU_FFT_SEP_ALL;	// NOTE: There is now way to over ride this (if batches are set, but it I believe it can only be faster esp with only one batch.
	      }
	    }
	  }

	  // Calculate the maximum number of steps for each possible number of batches
	  for ( int noBatchsTest = 0; noBatchsTest < MAX_BATCHES; noBatchsTest++)
	  {
	    // Initialise to zero
	    possSteps[noBatchsTest] = 0;

	    for ( int noStepsTest = MAX_STEPS; noStepsTest >= MIN_STEPS; noStepsTest--)
	    {
	      if ( kernel->flags & FLAG_SS_INMEM  ) // Size of memory for plane full fft plane  .
	      {
		size_t imWidth		= calcImWidth(cuSrch->sSpec->noSearchR*conf->noResPerBin, kernel->accelLen*noStepsTest,  kernel->strideOut);
		planeSize		= imWidth * kernel->hInfos->noZ * plnElsSZ;
	      }

	      if ( kernel->flags & CU_FFT_SEP_PLN )
	      {
		if ( ( free ) > kerSize + planeSize + ( (fffTotSize + batchSize) * (noBatchsTest+1) * noStepsTest ) )
		{
		  // Culate fractional value
		  possSteps[noBatchsTest] = ( free - kerSize - planeSize ) / (double)( (fffTotSize + batchSize) * (noBatchsTest+1) );;
		  break;
		}
	      }
	      else
	      {
		if ( ( free ) > kerSize + planeSize + (fffTotSize * noStepsTest) + ( batchSize  * (noBatchsTest+1) * noStepsTest ) )
		{
		  // Culate fractional value
		  possSteps[noBatchsTest] = ( free - kerSize - planeSize ) / (double)( fffTotSize + (batchSize * (noBatchsTest+1)) );
		  break;
		}
	      }
	    }

	    infoMSG(6,6,"For %2i batches can have %.1f steps.  In-mem plane size %.4f GB.\n", noBatchsTest+1, possSteps[noBatchsTest], planeSize*1e-9);

	    infoMSG(7,7,"ker: %.3f MB - FFT: %.3f MB - batch: %.3f MB - inpDataSize: %.2f MB - plnDataSize: ~%.2f MB - pwrDataSize: ~%.2f MB - cndDataSize: ~%.2f MB \n",
		kerSize*1e-6,
		fffTotSize*possSteps[noBatchsTest]*1e-6,
		batchSize*possSteps[noBatchsTest]*1e-6,
		kernel->inpDataSize*possSteps[noBatchsTest]*1e-6,
		kernel->plnDataSize*possSteps[noBatchsTest]*1e-6,
		kernel->pwrDataSize*possSteps[noBatchsTest]*1e-6,
		kernel->cndDataSize*possSteps[noBatchsTest]*1e-6 );
	  }

	  FOLD  // Set target steps  .
	  {
	    if ( gInf->capability > 3.2 )	// Maxwell  .
	    {
	      targetSteps = 8;
	    }
	    else				// Kepler  .
	    {
	      targetSteps = 6;
	    }
	  }

	  if ( noBatches == 0 )
	  {
	    if ( noSteps == 0 )
	    {
	      // We have free range to do what we want!
	      trySomething = 1;

	      infoMSG(5,5,"Automatic steps and batches.\n");
	    }
	    else
	    {
	      int maxBatches = 0;

	      // Determine the maximum number batches for the given steps
	      for ( int i = 0; i < MAX_BATCHES; i++)
	      {
		if ( possSteps[i] >= MAX(noSteps,MIN_STEPS) )
		  maxBatches = i+1;
		else
		  break;
	      }

	      if      ( maxBatches >= 3 )
	      {
		printf("     Requested %i steps per batch, could do up to %i batches, using 3.\n", noSteps, maxBatches);
		// Lets just do 3 batches, more than that doesn't really help often
		noBatches         = 3;
		kernel->noSteps   = MAX(noSteps,MIN_STEPS);
	      }
	      else if ( maxBatches >= 2 )
	      {
		printf("     Requested %i steps per batch, can do 2 batches.\n", noSteps);
		// Lets do 2 batches
		noBatches         = 2;
		kernel->noSteps   = MAX(noSteps,MIN_STEPS);
	      }
	      else if ( maxBatches >= 1 )
	      {
		// Lets just do 2 batches
		printf("     Requested %i steps per batch, can only do 1 batch.\n", noSteps);
		if ( noSteps >= 4 )
		  printf("       WARNING: Requested %i steps per batch, can only do 1 batch, perhaps consider using fewer steps.\n", noSteps );
		noBatches         = 1;
		kernel->noSteps   = MAX(noSteps,MIN_STEPS);
	      }
	      else
	      {
		printf("       ERROR: Can't even have 1 batch with the requested %i steps.\n", noSteps);
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
	      if ( possSteps[noBatches-1] >= MAX(1,MIN_STEPS) )
	      {
		FOLD
		{
		  // As many steps as possible!
		  kernel->noSteps   = floor(possSteps[noBatches-1]);

		  // Clip to target steps
		  MINN(kernel->noSteps, targetSteps);
		}

		if ( kernel->noSteps < MIN_STEPS )
		{
		  fprintf(stderr, "ERROR: Maximum number of steps (%i) possible is less than the compiled minimum (%i).\n", kernel->noSteps, MIN_STEPS);
		  exit(EXIT_FAILURE);
		}

		printf("     With %i batches, can do %.1f steps, using %i steps.\n", noBatches, possSteps[noBatches-1], kernel->noSteps);

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
	      if ( possSteps[noBatches-1] >= MAX(noSteps, MIN_STEPS) )
	      {
		printf("     Requested %i steps per batch on this device.\n", noSteps);

		// We can do what we asked for!
		kernel->noSteps   = MAX(noSteps, MIN_STEPS);

		if ( noSteps < MIN_STEPS )
		  printf("     Requested steps below the compile minimum of %i, using %i.\n", MIN_STEPS, kernel->noSteps);
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

	    // I have found for larger number of harmonics summed the number of steps has the biggest effect so optimise no steps first

	    // First see if can get optimal number of steps with any batches
	    for (int noB = 3; noB >= 1; noB--)
	    {
	      if      ( possSteps[noB-1] >= MAX(targetSteps, MIN_STEPS) )
	      {
		noBatches	= noB;
		kernel->noSteps	= floor(possSteps[noBatches-1]);
		printf("       Can have %.1f steps with %i batches.\n", possSteps[noBatches-1], noBatches );
		break;
	      }
	    }

	    // If couldn't get optimal number of steps see what else we can do
	    if ( !kernel->noSteps )
	    {
	      if      ( possSteps[2] >= MAX(4, MIN_STEPS) )
	      {
		noBatches	= 3;
		kernel->noSteps	= floor(possSteps[noBatches-1]);
		printf("       Can have %.1f steps with %i batches.\n", possSteps[noBatches-1], noBatches );
	      }
	      else if ( possSteps[1] >= MAX(2, MIN_STEPS) )
	      {
		// Lets do 2 batches and scale steps
		noBatches	= 2;
		kernel->noSteps	= floor(possSteps[noBatches-1]);
		printf("       Can have %.1f steps with %i batches.\n", possSteps[noBatches-1], noBatches );
	      }
	      else if ( possSteps[0] >= MAX(1, MIN_STEPS) )
	      {
		// Lets do 1 batches and scale steps
		noBatches	= 1;
		kernel->noSteps	= floor(possSteps[noBatches-1]);
		printf("       Can only have %.1f steps with %i batch.\n", possSteps[noBatches-1], noBatches );
	      }
	      else
	      {
		// Well we can't really do anything!
		noBatches	= 0;
		kernel->noSteps	= 0;

		if ( possSteps[0] > 0 && possSteps[0] < MIN_STEPS )
		{
		  fprintf(stderr, "ERROR: Can have %.1f steps with %i batch, BUT compiled with min of %i steps.\n", possSteps[0], 1, MIN_STEPS );
		}
		else
		{
		  printf("       ERROR: Can only have %.1f steps with %i batch.\n", possSteps[0], 1 );
		}
	      }
	    }

	    if ( kernel->noSteps > targetSteps )
	    {
	      printf("       Scaling steps down to target of %i.\n", targetSteps );
	      kernel->noSteps = targetSteps;
	    }
	  }

	  // Clip to compiled bounds
	  if ( kernel->noSteps > MAX_STEPS )
	  {
	    kernel->noSteps = MAX_STEPS;
	    printf("      Trying to use more steps that the maximum number (%i) this code is compiled with.\n", kernel->noSteps );
	  }
	  if ( kernel->noSteps < MIN_STEPS )
	  {
	    kernel->noSteps = MIN_STEPS;
	    printf("      Trying to use less steps that the maximum number (%i) this code is compiled with.\n", kernel->noSteps );
	  }

	  if ( noBatches <= 0 || kernel->noSteps <= 0 )
	  {
	    fprintf(stderr, "ERROR: Insufficient memory to make make any planes. One step would require %.2f GB of device memory.\n", ( fffTotSize + batchSize )*1e-9 );

	    freeKernel(kernel);
	    return (0);
	  }

	  // Final sanity check
	  if ( possSteps[noBatches-1] < kernel->noSteps )
	  {
	    fprintf(stderr, "ERROR: Unable to process %i steps with %i batches.\n", kernel->noSteps, noBatches );

	    freeKernel(kernel);
	    return (0);
	  }

	}

	// Final calculation of planeSize (with new step count)
	if ( kernel->flags & FLAG_SS_INMEM  ) // Size of memory for plane full FF plane  .
	{
	  size_t imWidth	= calcImWidth(cuSrch->sSpec->noSearchR*conf->noResPerBin, kernel->accelLen*kernel->noSteps, kernel->strideOut);
	  planeSize		= imWidth * kernel->hInfos->noZ * plnElsSZ;

	  infoMSG(7,7,"In-mem plane: %.2f GB - %i  ( %i x %i ) points at %i Bytes. \n", planeSize*1e-9, imWidth, kernel->hInfos->noZ, plnElsSZ);
	}

	char  cufftType[1024];
	if ( kernel->flags & CU_FFT_SEP_PLN )
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
	printf("    Kernels        use: %5.2f GB of device memory.\n", (kernel->kerDataSize) * 1e-9 );
	printf("    CUFFT         uses: %5.2f GB of device memory, %s\n", (fffTotSize*kernel->noSteps) * 1e-9, cufftType );
	if ( planeSize )
	{
	  printf("    In-mem plane  uses: %5.2f GB of device memory.", (planeSize) * 1e-9 );

	  if ( kernel->flags & FLAG_POW_HALF )
	  {
	    printf(" ( using half precision )\n");
	  }
	  else
	  {
	    printf("\n");
	  }
	}
	printf("    Each batch    uses: %5.2f GB of device memory.\n", (batchSize*kernel->noSteps) * 1e-9 );
	printf("                 Using: %5.2f GB of %.2f [%.2f%%] of GPU memory for search.\n",  totUsed * 1e-9, total * 1e-9, totUsed / (float)total * 100.0f );
      }

      PROF // Profiling  .
      {
	NV_RANGE_POP("Calc steps");
      }
    }

    FOLD // Scale data sizes by number of steps  .
    {
      kernel->inpDataSize *= kernel->noSteps;
      kernel->plnDataSize *= kernel->noSteps;
      kernel->pwrDataSize *= kernel->noSteps;
      if ( !(kernel->flags & FLAG_SS_INMEM)  )
	kernel->cndDataSize *= kernel->noSteps;					// In-mem search stage does not use steps
      kernel->retDataSize = kernel->cndDataSize + MAX_SAS_BLKS*sizeof(int);	// Add a bit extra to store return data

      // TODO: Perhaps we should make sure all these sizes are strided?

      // Update size to take into account steps
      batchSize		= kernel->inpDataSize + kernel->plnDataSize + kernel->pwrDataSize + kernel->retDataSize;  // This is currently the size of one step
      fffTotSize	= kernel->inpDataSize + kernel->plnDataSize;                                              // FFT data treated separately because there may be only one set per device
      kerSize		= kernel->kerDataSize;
      familySz		= kerSize + batchSize + fffTotSize;
    }

    // Calculate the stride and size of the candidate array
    cuSrch->candStride	= ceil(cuSrch->sSpec->noSearchR * conf->candRRes);
    float fullCSize     = cuSrch->candStride * candSZ;			// The full size of all candidate data

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

    FOLD // Batch independent device memory (ie in-memory plane) .
    {
      if ( kernel->flags & FLAG_SS_INMEM  )
      {
	PROF // Profiling  .
	{
	  NV_RANGE_PUSH("in-mem alloc");
	}

	size_t stride;
	size_t imWidth		= calcImWidth(cuSrch->sSpec->noSearchR*conf->noResPerBin, kernel->accelLen*kernel->noSteps, kernel->strideOut);
	planeSize		= imWidth * kernel->hInfos->noZ * plnElsSZ;

	infoMSG(7,7,"In-mem plane: %.2f GB - %i  ( %i x %i ) points at %i Bytes. \n", planeSize*1e-9, imWidth, kernel->hInfos->noZ, plnElsSZ);

	CUDA_SAFE_CALL(cudaMallocPitch(&cuSrch->d_planeFull,    &stride, plnElsSZ*imWidth, kernel->hInfos->noZ),   "Failed to allocate strided memory for in-memory plane.");
	CUDA_SAFE_CALL(cudaMemsetAsync(cuSrch->d_planeFull, 0, stride*kernel->hInfos->noZ, kernel->stacks->initStream),"Failed to initiate in-memory plane to zero");

	free -= stride*kernel->hInfos->noZ;
	infoMSG(7,7,"In-mem plane: %.2f GB free: free %.3f MB\n", stride*kernel->hInfos->noZ*1e-9, free*1e-6);

	FOLD // Round down to units  .
	{
	  cuSrch->inmemStride = ceil(stride / (double)plnElsSZ);
	  if ( cuSrch->inmemStride != stride / (double)plnElsSZ )
	  {
	    fprintf(stderr, "ERROR: Stride of in-memory plane is not divisabe by elements size. Pleas contact Chris Laidler.");
	    exit(EXIT_FAILURE);
	  }
	}

	infoMSG(7,7,"ker: %.3f MB - FFT: %.3f MB - batch: %.3f MB - inpDataSize: %.2f MB - plnDataSize: ~%.2f MB - pwrDataSize: ~%.2f MB - cndDataSize: ~%.2f MB \n",
	    kerSize*1e-6,
	    fffTotSize*1e-6,
	    batchSize*1e-6,
	    kernel->inpDataSize*1e-6,
	    kernel->plnDataSize*1e-6,
	    kernel->pwrDataSize*1e-6,
	    kernel->retDataSize*1e-6 );

	PROF // Profiling  .
	{
	  NV_RANGE_POP("in-mem alloc");
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
	  NV_RANGE_PUSH("host str alloc");
	}

	if 	( kernel->cndType & CU_STR_ARR	)
	{
	  if ( conf->outData == NULL   )
	  {
	    // Have to allocate the array!

	    freeRam  = getFreeRamCU();
	    if ( fullCSize < freeRam*0.9 )
	    {
	      infoMSG(5,5,"Allocate host memory for candidate array. (%.2f MB)\n", fullCSize*1e-6 );

	      // Same host candidates for all devices
	      // This can use a lot of memory for long searches!
	      cuSrch->h_candidates = malloc( fullCSize );

	      PROF // Profiling  .
	      {
		NV_RANGE_PUSH("memset");
	      }

	      memset(cuSrch->h_candidates, 0, fullCSize );

	      PROF // Profiling  .
	      {
		NV_RANGE_POP("memset");
	      }

	      hostC += fullCSize;
	    }
	    else
	    {
	      fprintf(stderr, "ERROR: Not enough host memory for candidate list array. Need %.2f GB there is %.2f GB.\n", fullCSize * 1e-9, freeRam * 1e-9 );
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
	    cuSrch->h_candidates = conf->outData;
	    memset(cuSrch->h_candidates, 0, fullCSize ); // NOTE: this may error if the preallocated memory int large enough!
	  }
	}
	else if ( kernel->cndType & CU_STR_QUAD	)
	{
	  if ( conf->outData == NULL )
	  {
	    infoMSG(5,5,"Creating quadtree for candidates.\n" );

	    candTree* qt = new candTree;
	    cuSrch->h_candidates = qt;
	  }
	  else
	  {
	    cuSrch->h_candidates = conf->outData;
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
	  cuSrch->h_candidates = conf->outData;
	}

	PROF // Profiling  .
	{
	  NV_RANGE_POP("host alloc");
	}
      }
    }

    if ( hostC )
    {
      printf("    Input and candidates use an additional:\n");
      if ( hostC )
	printf("                        %5.2f GB of host   memory\n", hostC * 1e-9 );
    }
    printf("    -----------------------------------------------\n" );

    CUDA_SAFE_CALL(cudaGetLastError(), "Failed to create memory for candidate list or input data.");

    PROF // Profiling  .
    {
      NV_RANGE_POP("data");
    }
  }

  FOLD // Set up FFT's  .
  {
    FOLD // Batch FFT streams  .
    {
      PROF // Profiling  .
      {
	NV_RANGE_PUSH("FFT streams");
      }

      infoMSG(4,4,"Batch FFT streams\n");

      char strBuff[1024];

      if ( !(kernel->flags & CU_INPT_FFT_CPU) && !(kernel->flags & CU_FFT_SEP_INP) )
      {
	for (int i = 0; i < kernel->noStacks; i++)
	{
	  cuFfdotStack* cStack = &kernel->stacks[i];

	  CUDA_SAFE_CALL(cudaStreamCreate(&cStack->fftIStream),"Creating CUDA stream for fft's");

	  PROF // Profiling, name stream  .
	  {
	    sprintf(strBuff,"%i.0.2.%i FFT Input Dev", kernel->gInf->devid, i);
	    NV_NAME_STREAM(cStack->fftIStream, strBuff);
	  }
	}
      }

      if ( !(kernel->flags & CU_FFT_SEP_PLN) )
      {
	for (int i = 0; i < kernel->noStacks; i++)
	{
	  cuFfdotStack* cStack = &kernel->stacks[i];

	  CUDA_SAFE_CALL(cudaStreamCreate(&cStack->fftPStream),"Creating CUDA stream for fft's");

	  PROF // Profiling, name stream  .
	  {
	    sprintf(strBuff,"%i.0.4.%i FFT Plane Dev", kernel->gInf->devid, i);
	    NV_NAME_STREAM(cStack->fftPStream, strBuff);
	  }
	}
      }

      PROF // Profiling  .
      {
	NV_RANGE_POP("streams");
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

#ifdef USEFFTW
	PROF // Profiling  .
	{
	  NV_RANGE_PUSH("read_wisdom");
	}

	read_wisdom();

	PROF // Profiling  .
	{
	  NV_RANGE_POP("read_wisdom");
	}
#else
	fprintf(stderr,"ERROR: GPU need fftw");
	exit(EXIT_FAILURE);
#endif


      }

      if ( kernel->flags & CU_FFT_SEP_ALL  )
      {
	infoMSG(4,4,"Create \"Global\" FFT plans.\n");
      }

      if ( !(kernel->flags & CU_FFT_SEP_INP) && !(kernel->flags & CU_FFT_SEP_PLN) )
      {
	createFFTPlans(kernel, FFT_BOTH);
      }
      else if ( !(kernel->flags & CU_FFT_SEP_INP) )
      {
	createFFTPlans(kernel, FFT_INPUT);
      }
      else if ( !(kernel->flags & CU_FFT_SEP_PLN) )
      {
	createFFTPlans(kernel, FFT_PLANE);
      }

      PROF // Profiling  .
      {
	NV_RANGE_POP("FFT plans");
      }
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
	NV_RANGE_POP("text mem");
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

    setConstVals( kernel );					//
    setConstVals_Fam_Order( kernel );				// Constant values for multiply

    FOLD // Set CUFFT load callback details  .
    {
      if( kernel->flags & FLAG_MUL_CB )
      {
	setStackVals( kernel );
      }
    }

    FOLD // CUFFT store callbacks  .
    {
      if ( !(kernel->flags & CU_FFT_SEP_PLN) )
      {
#if CUDART_VERSION >= 6050                                      // CUFFT callbacks only implemented in CUDA 6.5
	copyCUFFT_CBs(kernel);
#endif
      }
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP("const mem");
    }
  }

  printf("Done initializing GPU %i.\n", kernel->gInf->devid);

  std::cout.flush();

  PROF // Profiling  .
  {
    NV_RANGE_POP('msg');
  }

  return noBatches;
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
  char msg[1024];

  PROF // Profiling  .
  {
    sprintf(msg,"%i of %i", no+1, of+1);
    NV_RANGE_PUSH(msg); // # of #
  }

  char strBuff[1024];
  size_t free, total;

  infoMSG(3,3,"\n%s - Device %i, batch %i of %i  (%p)\n",__FUNCTION__, kernel->gInf->devid, no+1, of+1, batch);

  FOLD // See if we can use the cuda device  .
  {
    infoMSG(4,4,"Set device to %i.\n", kernel->gInf->devid);

    setDevice(kernel->gInf->devid) ;

    CUDA_SAFE_CALL(cudaMemGetInfo ( &free, &total ), "Getting Device memory information");
  }

  FOLD // Copy details from kernel and allocate stacks .
  {
    infoMSG(4,4,"Copy kernel data struct.\n");

    // Copy the basic batch parameters from the kernel
    memcpy(batch, kernel, sizeof(cuFFdotBatch));

    batch->srchMaster   = 0;
    batch->isKernel     = 0;

    infoMSG(4,4,"Create and copy stack data structs.\n");

    // Allocate memory for the stacks
    batch->stacks = (cuFfdotStack*) malloc(batch->noStacks * sizeof(cuFfdotStack));

    // Copy the actual stacks
    memcpy(batch->stacks, kernel->stacks, batch->noStacks  * sizeof(cuFfdotStack));
  }

  FOLD // Set the batch specific flags  .
  {
    infoMSG(4,4,"Set batch specific flags\n");

    FOLD // Multiplication flags  .
    {
      for ( int i = 0; i < batch->noStacks; i++ )	// Multiplication is generally stack specific so loop through stacks  .
      {
	cuFfdotStack* cStack  = &batch->stacks[i];

	FOLD // Multiplication kernel  .
	{
	  if ( !(cStack->flags & FLAG_MUL_ALL) )	// Default to multiplication  .
	  {
	    infoMSG(5,5,"No multiplication kernel specified, auto select (Good).");
	    int64_t mFlag = 0;

	    // In my testing I found multiplying each plane separately works fastest so it is the "default"
	    int noInp =  cStack->noInStack * kernel->noSteps ;

	    if ( batch->gInf->capability > 3.0 )
	    {
	      // Lots of registers per thread so 2.1 is good
	      infoMSG(5,5,"Compute capability %.1f > 3.0. Easy, use multiplication kernel 2.1\n", batch->gInf->capability);
#ifdef WITH_MUL_21
	      mFlag |= FLAG_MUL_21;
#else	// WITH_MUL_21
	      fprintf(stderr, "ERROR: Not compiled with Mult 21 kernel pleas manually specify multiplication kernel.");
	      exit(EXIT_FAILURE);
#endif	// WITH_MUL_21
	    }
	    else
	    {
	      infoMSG(5,5,"Compute caperbility %.1f <= 3.0. (device has a smaller number registers)\n", batch->gInf->capability);

#if defined(WITH_MUL_22) && defined(WITH_MUL_22)

	      // Require fewer registers per thread, so use Multiplication kernel 2.1
	      if ( noInp <= 20 )
	      {
		infoMSG(5,5,"# input for stack %i is %i, this is <= 20 so use mult 2.1 \n", i, noInp);

		// TODO: Check small, looks like some times MUL_22 may be faster.
		mFlag |= FLAG_MUL_21;
	      }
	      else
	      {
		infoMSG(5,5,"# input for stack %i is %i, this is > 20\n", i, noInp);

		if ( kernel->noSteps <= 4 )
		{
		  infoMSG(5,5,"steps (%i) < 4\n", kernel->noSteps );

		  // very few steps so 2.2 not always the best option
		  if ( kernel->hInfos->zmax > 100 )  // TODO: this should use stack height rather than total zmax
		  {
		    infoMSG(5,5,"zmax > 100 use mult 2.3.\n");

		    // This only really holds for 16 harmonics summed with 3 or 4 steps
		    // In my testing it is generally true for zmax greater than 100
		    mFlag |= FLAG_MUL_23;
		  }
		  else
		  {
		    infoMSG(5,5,"zmax <= 100 use mult 2.2.\n");

		    // Here 22 is usually better
		    mFlag |= FLAG_MUL_22;
		  }
		}
		else
		{
		  infoMSG(5,5,"Plenty steps so use mult 2.2 \n");

		  // Enough steps to justify Multiplication kernel 2.2
		  mFlag |= FLAG_MUL_22;
		}
	      }
#elif defined(WITH_MUL_22)
	      fprintf(stderr, "WARNNG: Not compiled with Mult 23 so using Mult 22 kernel.");
	      infoMSG(5,5,"# only compiled with mult 2.2 \n", i, noInp);
	      mFlag |= FLAG_MUL_22;
#elif defined(WITH_MUL_23)
	      fprintf(stderr, "WARNNG: Not compiled with Mult 22 so using Mult 23 kernel.");
	      infoMSG(5,5,"# only compiled with mult 2.3 \n", i, noInp);
	      mFlag |= FLAG_MUL_23;
#else
	      fprintf(stderr, "ERROR: Not compiled with Mult 22 or 23 kernels pleas manually specify multiplication kernel.");
	      exit(EXIT_FAILURE);
#endif
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
	    cStack->mulChunk = 12;	// TODO: Profile this parameter
	  }

	  // Clamp chunk length to slice length
	  MINN(cStack->mulChunk, ceilf(cStack->kerHeigth/(float)cStack->mulSlices) );

	  // Clamp to compilation bounds
	  MINN(cStack->mulChunk, MAX_MUL_CHUNK);
	  MAXX(cStack->mulChunk, MIN_MUL_CHUNK);

	  if ( i == 0 )
	    batch->mulChunk = cStack->mulChunk;

	  infoMSG(5,5,"stack %i  mulChunk %2i \n",i, cStack->mulChunk);
	}
      }
    }

    FOLD // Sum and search flags  .
    {

      if ( !(batch->flags & FLAG_SS_ALL ) )   // Default to multiplication  .
      {
	batch->flags |= FLAG_SS_31;
      }

      if ( batch->ssChunk <= 0 )
      {
	if ( batch->flags & FLAG_SS_INMEM )
	{
	  // With the inmem search only one big "step" in the search phase
	  if ( batch->gInf->capability < 5.0 )	// Kepler and older .
	  {
	    // NOTE: These values were computed from testing with a GTX 770 - These could be made an auto tune
	    int lookup[5] = { 5, 5, 5, 6, 6 };
	    batch->ssChunk = lookup[batch->noHarmStages-1];

#ifdef WITH_SAS_COUNT
	    // I found in this case just maximise chunk size 	// TODO: Recheck this
	    batch->ssChunk = MIN(12, MAX_SAS_CHUNK);
#endif
	  }
	  else					// Maxwell and newer .
	  {
	    // NOTE: These values were computed from testing with a GTX 970 - These could be made an auto tune
	    int lookup[5] = { 7, 10, 8, 8, 6 };
	    batch->ssChunk = lookup[batch->noHarmStages-1];
	  }
	}
	else
	{
	  // Using standard sum and search kernel

	  if ( batch->gInf->capability < 5.0 )	// Kepler and older .
	  {
	    // Kepler cards have fewer registers so this limit chunk size
	    // NOTE: These values were computed from testing with a GTX 770 - These could be made an auto tune
	    int lookup[5][12] = {	{12, 8,  4,  5, 4, 3, 2, 2, 1, 1, 1, 1},
					{11, 12, 8,  5, 3, 2, 1, 1, 4, 3, 3, 4},
					{12, 12, 10, 8, 7, 6, 5, 4, 4, 3, 3, 3},
					{12, 12, 10, 8, 7, 6, 4, 4, 3, 3, 2, 2},
					{12, 11, 9,  8, 6, 6, 4, 3, 3, 2, 2, 2} };
	    batch->ssChunk = lookup[batch->noHarmStages-1][batch->noSteps-1];
	  }
	  else					// Maxwell and newer .
	  {
	    // More register
	    // NOTE: These values were computed from testing with a GTX 970 - These could be made an auto tune
	    int lookup[5][12] = {	{12, 8,  6, 7, 5, 6, 4, 3, 2, 2, 1, 1},
					{12, 10, 8, 7, 5, 4, 1, 1, 6, 6, 5, 5},
					{10, 12, 6, 9, 6, 6, 5, 4, 6, 3, 5, 5},
					{12, 9,  9, 9, 6, 6, 4, 3, 3, 6, 2, 4},
					{10, 12, 9, 8, 7, 5, 4, 3, 2, 2, 5, 5} };
	    batch->ssChunk = lookup[batch->noHarmStages-1][batch->noSteps-1];
	  }
	}
      }

      if ( batch->ssColumn <= 0 )
      {
	// TODO: Profile this
	batch->ssColumn = 8;
      }

      FOLD // Clamps
      {
	// Clamp S&S chunks to slice height
	batch->ssChunk = MINN(batch->ssChunk, ceil(kernel->hInfos->noZ/(float)batch->ssSlices) );

	// Clamp S&S chunks to valid bounds
	MINN(batch->ssChunk, MAX_SAS_CHUNK);
	MAXX(batch->ssChunk, MIN_SAS_CHUNK);

	MINN(batch->ssColumn, MAX_SAS_COLUMN);
	MAXX(batch->ssColumn, MIN_SAS_COLUMN);
      }
    }
  }

  FOLD // Create FFT plans  .
  {
    if ( kernel->flags & CU_FFT_SEP_ALL  )
    {
      infoMSG(4,4,"Create batch FFT plans.\n");
    }

    if ( (kernel->flags & CU_FFT_SEP_INP) && (kernel->flags & CU_FFT_SEP_PLN) )
    {
      createFFTPlans(batch, FFT_BOTH);
    }
    else if ( kernel->flags & CU_FFT_SEP_INP )
    {
      createFFTPlans(batch, FFT_INPUT);
    }
    else if ( kernel->flags & CU_FFT_SEP_PLN )
    {
      createFFTPlans(batch, FFT_PLANE);
    }

    if ( kernel->flags & CU_FFT_SEP_PLN )               // Set CUFFT callbacks
    {
#if CUDART_VERSION >= 6050                              // CUFFT callbacks only implemented in CUDA 6.5
      copyCUFFT_CBs(batch);
#endif
    }

    // Set the CUFFT load and store callback if necessary  .
    for (int i = 0; i < batch->noStacks; i++)           // Loop through Stacks
    {
	cuFfdotStack* cStack = &batch->stacks[i];
	setCB(batch, cStack);
    }
  }

  FOLD // Allocate all device and host memory for the batch  .
  {
    infoMSG(4,4,"Allocate memory for the batch\n");

    FOLD // Standard host memory allocation  .
    {
      PROF // Profiling  .
      {
	NV_RANGE_PUSH("malloc");
      }

      FOLD // Allocate R value lists  .
      {
	infoMSG(5,5,"Allocate R value lists.\n");

	batch->noRArryas        = batch->conf->ringLength;

	createRvals(batch, &batch->rArr1, &batch->rArraysPlane);
	batch->rAraays = &batch->rArraysPlane;
      }

      FOLD // Create the planes data structures  .
      {
	if ( batch->noGenHarms* sizeof(cuFFdot) > getFreeRamCU() )
	{
	  fprintf(stderr, "ERROR: Not enough host memory for search.\n");
	  return 0;
	}
	else
	{
	  infoMSG(5,5,"Allocate planes data structures.\n");

	  batch->planes = (cuFFdot*) malloc(batch->noGenHarms* sizeof(cuFFdot));
	  memset(batch->planes, 0, batch->noGenHarms* sizeof(cuFFdot));
	}
      }

      FOLD // Allocate host input memory  .
      {
	// Allocate buffer for CPU to work on input data
	batch->h_iBuffer = (fcomplexcu*)malloc(batch->inpDataSize);
	memset(batch->h_iBuffer, 0, batch->inpDataSize);

	if ( !(batch->flags & CU_NORM_GPU) )
	{
	  infoMSG(5,5,"Allocate memory for normalisation powers. (%.2f MB)\n", batch->hInfos->width * sizeof(float)*1e-6 );

	  // Allocate CPU memory for normalisation
	  batch->h_normPowers = (float*) malloc(batch->hInfos->width * sizeof(float));
	}
      }

      PROF // Create timing arrays  .
      {
	if ( batch->flags & FLAG_PROF )
	{
	  int sz = batch->noStacks*sizeof(long long)*(COMP_GEN_MAX) ;

	  infoMSG(5,5,"Allocate timing array. (%.2f MB)\n", sz*1e-6 );

	  batch->compTime       = (long long*)malloc(sz);
	  memset(batch->compTime,    0, sz);
	}
      }

      PROF // Profiling  .
      {
	NV_RANGE_POP("malloc");
      }
    }

    FOLD // Allocate device Memory for Planes, Stacks & Input data (steps)  .
    {
      PROF // Profiling  .
      {
	NV_RANGE_PUSH("malloc device");
      }

      size_t req = batch->inpDataSize + batch->plnDataSize + batch->pwrDataSize + kernel->retDataSize;

      if ( req > free ) // Not enough memory =(
      {
	printf("Not enough GPU memory to create any more batches. %.3f MB required %.3f MB free.\n", req*1e-6, free*1e-6);
	return (0);
      }
      else
      {
	if ( batch->inpDataSize )
	{
	  infoMSG(5,5,"Allocate device memory for input. (%.2f MB)\n", batch->inpDataSize*1e-6);

	  CUDA_SAFE_CALL(cudaMalloc((void** )&batch->d_iData,       batch->inpDataSize ), "Failed to allocate device memory for batch input.");
	  free -= batch->inpDataSize;
	}

	if ( batch->plnDataSize )
	{
	  infoMSG(5,5,"Allocate device memory for complex plane. (%.2f MB)\n", batch->plnDataSize*1e-6);

	  CUDA_SAFE_CALL(cudaMalloc((void** )&batch->d_planeMult,   batch->plnDataSize ), "Failed to allocate device memory for batch complex plane.");
	  free -= batch->plnDataSize;
	  infoMSG(7,7,"complex plane: %p", batch->d_planeMult);
	}

	if ( batch->pwrDataSize )
	{
	  infoMSG(5,5,"Allocate device memory for powers plane. (%.2f MB)\n", batch->pwrDataSize*1e-6);

	  CUDA_SAFE_CALL(cudaMalloc((void** )&batch->d_planePowr,   batch->pwrDataSize ), "Failed to allocate device memory for batch powers plane.");
	  free -= batch->pwrDataSize;
	  infoMSG(7,7,"powers plane:  %p", batch->d_planePowr);
	}

	if ( kernel->retDataSize && !(kernel->retType & CU_STR_PLN) )
	{
	  infoMSG(5,5,"Allocate device memory for return values. (%.2f MB)\n", batch->retDataSize*1e-6);

	  CUDA_SAFE_CALL(cudaMalloc((void** ) &batch->d_outData1, batch->retDataSize ), "Failed to allocate device memory for return values.");
	  CUDA_SAFE_CALL(cudaMemsetAsync(batch->d_outData1, 0, batch->retDataSize, kernel->stacks->initStream),"Failed to initiate return data to zero");
	  free -= batch->retDataSize;

	  if ( batch->flags & FLAG_SS_INMEM )
	  {
	    // NOTE: Most of the time could use complex plane for both sets of return data.

	    if ( batch->retDataSize > batch->plnDataSize )
	    {
	      infoMSG(5,5,"Complex plane is smaller than return data -> FLAG_SEPSRCH\n");

	      batch->flags |= FLAG_SEPSRCH;
	    }

	    if ( batch->flags & FLAG_SEPSRCH )
	    {
	      infoMSG(5,5,"Allocate device memory for second return values. (%.2f MB)\n", batch->retDataSize*1e-6);

	      // Create a separate output space
	      CUDA_SAFE_CALL(cudaMalloc((void** ) &batch->d_outData2, batch->retDataSize ), "Failed to allocate device memory for return values.");
	      CUDA_SAFE_CALL(cudaMemsetAsync(batch->d_outData2, 0, batch->retDataSize, kernel->stacks->initStream),"Failed to initiate return data to zero");
	      free -= batch->retDataSize;
	    }
	    else
	    {
	      infoMSG(5,5,"Using complex plane for second return values. (%.2f MB of %.2f MB)\n", batch->retDataSize*1e-6, batch->plnDataSize*1e-6 );

	      batch->d_outData2 = batch->d_planeMult;
	    }
	  }
	}
      }

      PROF // Profiling  .
      {
	NV_RANGE_POP("malloc device");
      }
    }

    FOLD // Allocate page-locked host memory for return data  .
    {
      PROF // Profiling  .
      {
	NV_RANGE_PUSH("Malloc Host");
      }

      if ( batch->inpDataSize )
      {
	infoMSG(5,5,"Allocate page-locked for input data. (%.2f MB)\n", batch->inpDataSize*1e-6 );

	CUDA_SAFE_CALL(cudaMallocHost(&batch->h_iData, batch->inpDataSize ), "Failed to create page-locked host memory plane input data." );
      }

      if ( kernel->retDataSize ) // Allocate page-locked host memory to copy the candidates back to  .
      {
	infoMSG(5,5,"Allocate page-locked for candidates. (%.2f MB) \n", kernel->retDataSize*batch->noRArryas*1e-6);

	for (int i = 0 ; i < batch->noRArryas; i++)
	{
	  rVals* rVal = &(((*batch->rAraays)[i])[0][0]);

	  CUDA_SAFE_CALL(cudaMallocHost(&rVal->h_outData, kernel->retDataSize), "Failed to create page-locked host memory plane for return data.");
	  //memset(rVal->h_outData, 0, kernel->retDataSize ); // Not necessary ?
	}
      }

      PROF // Profiling  .
      {
	NV_RANGE_POP("Host");
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
    infoMSG(4,4,"Set up the batch streams and events.\n");

    FOLD // Create Streams  .
    {
      FOLD // Input streams  .
      {
	infoMSG(5,5,"Create input stream for batch.\n");

	// Batch input ( Always needed, for copying input to device )
	CUDA_SAFE_CALL(cudaStreamCreate(&batch->inpStream),"Creating input stream for batch.");

	PROF // Profiling name streams  .
	{
	  sprintf(strBuff,"%i.%i.1.0 Batch Input", batch->gInf->devid, no);
	  NV_NAME_STREAM(batch->inpStream, strBuff);
	}

	// Stack input
	if ( (batch->flags & CU_NORM_GPU)  )
	{
	  infoMSG(5,5,"Create input stream for stacks to normalise with.\n");

	  for (int i = 0; i < batch->noStacks; i++)
	  {
	    cuFfdotStack* cStack  = &batch->stacks[i];

	    CUDA_SAFE_CALL(cudaStreamCreate(&cStack->inptStream), "Creating input data multStream for stack");

	    PROF 				// Profiling, name stream  .
	    {
	      sprintf(strBuff,"%i.%i.1.%i Stack Input", batch->gInf->devid, no, i);
	      NV_NAME_STREAM(cStack->inptStream, strBuff);
	    }
	  }
	}
      }

      FOLD // Input FFT streams  .
      {
	if ( !(kernel->flags & CU_INPT_FFT_CPU)  )		// Using CUFFT for input  .
	{
	  for (int i = 0; i < kernel->noStacks; i++)
	  {
	    cuFfdotStack* cStack = &batch->stacks[i];

	    if ( kernel->flags & CU_FFT_SEP_INP )		// Create stream  .
	    {
	      infoMSG(5,5,"Create stream for input FFT, stack %i.\n", i);

	      CUDA_SAFE_CALL(cudaStreamCreate(&cStack->fftIStream),"Creating CUDA stream for input fft's");

	      PROF 						// Profiling name streams  .
	      {
		sprintf(strBuff,"%i.%i.2.%i Inp FFT", batch->gInf->devid, no, i);
		NV_NAME_STREAM(cStack->fftIStream, strBuff);
	      }
	    }
	    else						// Copy stream of the kernel  .
	    {
	      infoMSG(5,5,"Using global input FFT stream for stack %i.\n", i);

	      cuFfdotStack* kStack  = &kernel->stacks[i];	// Kernel stack, has the "global" stream
	      cStack->fftIStream    = kStack->fftIStream;
	    }
	  }
	}
      }



      FOLD // Multiply streams  .
      {
	if      ( batch->flags & FLAG_MUL_BATCH )
	{
	  infoMSG(5,5,"Create batch stream for multiplication.\n");

	  CUDA_SAFE_CALL(cudaStreamCreate(&batch->multStream),"Creating multiplication stream for batch.");

	  PROF 					// Profiling name streams  .
	  {
	    sprintf(strBuff,"%i.%i.3.0 Batch Multiply", batch->gInf->devid, no);
	    NV_NAME_STREAM(batch->multStream, strBuff);
	  }
	}

	if ( (batch->flags & FLAG_MUL_STK) || (batch->flags & FLAG_MUL_PLN)  )
	{
	  infoMSG(5,5,"Create streams for stack multiplication.\n");
	  for (int i = 0; i< batch->noStacks; i++)
	  {
	    cuFfdotStack* cStack  = &batch->stacks[i];
	    CUDA_SAFE_CALL(cudaStreamCreate(&cStack->multStream), "Creating multStream for stack");

	    PROF 				// Profiling name streams  .
	    {
	      sprintf(strBuff,"%i.%i.3.%i Stack Multiply", batch->gInf->devid, no, i);
	      NV_NAME_STREAM(cStack->multStream, strBuff);
	    }
	  }
	}
      }

      FOLD // Inverse FFT streams  .
      {
	for (int i = 0; i < kernel->noStacks; i++)
	{
	  cuFfdotStack* cStack = &batch->stacks[i];

	  if ( batch->flags & CU_FFT_SEP_PLN )	// Create stream
	  {
	    infoMSG(5,5,"Create streams for stack iFFT, stack %i.\n",i);
	    CUDA_SAFE_CALL(cudaStreamCreate(&cStack->fftPStream), "Creating fftPStream for stack");

	    PROF 				// Profiling name streams  .
	    {
	      sprintf(strBuff,"%i.%i.4.%i Stack iFFT", batch->gInf->devid, no, i);
	      NV_NAME_STREAM(cStack->fftPStream, strBuff);
	    }
	  }
	  else					// Copy stream of the kernel
	  {
	    infoMSG(5,5,"Using global iFFT stream for stack %i.\n", i);

	    cuFfdotStack* kStack  = &kernel->stacks[i];
	    cStack->fftPStream    = kStack->fftPStream;
	  }
	}
      }

      FOLD // Search stream  .
      {
	infoMSG(5,5,"Create stream for batch search.\n");

	CUDA_SAFE_CALL(cudaStreamCreate(&batch->srchStream), "Creating strmSearch for batch.");

	PROF 				// Profiling name streams  .
	{
	  sprintf(strBuff,"%i.%i.5.0 Batch Search", batch->gInf->devid, no);
	  NV_NAME_STREAM(batch->srchStream, strBuff);
	}
      }

      FOLD // Result stream  .
      {
	infoMSG(5,5,"Create stream top copy results back drom device.\n");

	// Batch output ( Always needed, for copying results from device )
	CUDA_SAFE_CALL(cudaStreamCreate(&batch->resStream), "Creating resStream for batch.");

	PROF 				// Profiling name streams  .
	{
	  sprintf(strBuff,"%i.%i.6.0 Batch result", batch->gInf->devid, no);
	  NV_NAME_STREAM(batch->resStream, strBuff);
	}

      }

      CUDA_SAFE_CALL(cudaGetLastError(), "Creating streams for the batch.");
    }

    FOLD // Create Events  .
    {
      FOLD // Create batch events  .
      {
	if ( batch->flags & FLAG_PROF )
	{
	  infoMSG(4,4,"Create batch events with timing enabled.\n");

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
	  infoMSG(4,4,"Create batch events with timing disabled.\n");

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
	    infoMSG(4,4,"Create stack %i events with timing enabled.\n", i);

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
	    infoMSG(4,4,"Create stack %i events with timing disabled.\n", i);

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

//  FOLD // Create textures for the f-∂f planes  .
//  {
//    if ( (batch->flags & FLAG_TEX_INTERP) && !( (batch->flags & FLAG_CUFFT_CB_POW) && (batch->flags & FLAG_SAS_TEX) ) )
//    {
//      fprintf(stderr, "ERROR: Cannot use texture memory interpolation without CUFFT callback to write powers. NOT using texture memory interpolation.\n");
//      batch->flags &= ~FLAG_TEX_INTERP;
//    }
//
//    if ( batch->flags & FLAG_SAS_TEX ) // This is depricated, but could be woth revisiting   .
//    {
//      infoMSG(4,4,"Create textures\n");
//
//      cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
//
//      struct cudaTextureDesc texDesc;
//      memset(&texDesc, 0, sizeof(texDesc));
//      texDesc.addressMode[0]    = cudaAddressModeClamp;
//      texDesc.addressMode[1]    = cudaAddressModeClamp;
//      texDesc.readMode          = cudaReadModeElementType;
//      texDesc.normalizedCoords  = 0;
//
//      if ( batch->flags & FLAG_TEX_INTERP )
//      {
//	texDesc.filterMode        = cudaFilterModeLinear;   /// Liner interpolation
//      }
//      else
//      {
//	texDesc.filterMode        = cudaFilterModePoint;
//      }
//
//      for (int i = 0; i< batch->noStacks; i++)
//      {
//	cuFfdotStack* cStack = &batch->stacks[i];
//
//	cudaResourceDesc resDesc;
//	memset(&resDesc, 0, sizeof(resDesc));
//	resDesc.resType           = cudaResourceTypePitch2D;
//	resDesc.res.pitch2D.desc  = channelDesc;
//
//	for (int j = 0; j< cStack->noInStack; j++)
//	{
//	  cuFFdot* cPlane = &cStack->planes[j];
//
//	  if ( batch->flags & FLAG_CUFFT_CB_POW ) // float input
//	  {
//	    if      ( batch->flags & FLAG_ITLV_ROW )
//	    {
//	      resDesc.res.pitch2D.height          = cPlane->harmInf->noZ;
//	      resDesc.res.pitch2D.width           = cPlane->harmInf->width * batch->noSteps;
//	      resDesc.res.pitch2D.pitchInBytes    = cStack->harmInf->width * batch->noSteps * sizeof(float);
//	      resDesc.res.pitch2D.devPtr          = cPlane->d_planePowr;
//	    }
//#ifdef WITH_ITLV_PLN
//	    else
//	    {
//	      resDesc.res.pitch2D.height          = cPlane->harmInf->noZ * batch->noSteps ;
//	      resDesc.res.pitch2D.width           = cPlane->harmInf->width;
//	      resDesc.res.pitch2D.pitchInBytes    = cStack->harmInf->width * sizeof(float);
//	      resDesc.res.pitch2D.devPtr          = cPlane->d_planePowr;
//	    }
//#else
//	    else
//	    {
//	      fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
//	      exit(EXIT_FAILURE);
//	    }
//#endif
//	  }
//	  else // Implies complex numbers
//	  {
//	    if      ( batch->flags & FLAG_ITLV_ROW )
//	    {
//	      resDesc.res.pitch2D.height          = cPlane->harmInf->noZ;
//	      resDesc.res.pitch2D.width           = cPlane->harmInf->width * batch->noSteps * 2;
//	      resDesc.res.pitch2D.pitchInBytes    = cStack->harmInf->width * batch->noSteps * 2 * sizeof(float);
//	      resDesc.res.pitch2D.devPtr          = cPlane->d_planePowr;
//	    }
//#ifdef WITH_ITLV_PLN
//	    else
//	    {
//	      resDesc.res.pitch2D.height          = cPlane->harmInf->noZ * batch->noSteps ;
//	      resDesc.res.pitch2D.width           = cPlane->harmInf->width * 2;
//	      resDesc.res.pitch2D.pitchInBytes    = cStack->harmInf->width * 2 * sizeof(float);
//	      resDesc.res.pitch2D.devPtr          = cPlane->d_planePowr;
//	    }
//#else
//	    else
//	    {
//	      fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
//	      exit(EXIT_FAILURE);
//	    }
//#endif
//	  }
//
//	  CUDA_SAFE_CALL(cudaCreateTextureObject(&cPlane->datTex, &resDesc, &texDesc, NULL), "Creating texture from the plane data.");
//	}
//      }
//      CUDA_SAFE_CALL(cudaGetLastError(), "Creating textures from the plane data.");
//    }
//  }

  PROF // Profiling  .
  {
    NV_RANGE_POP(msg);
  }

  return (batch->noSteps);
}

/** Create the FFT plans for a batch
 *
 * @param batch		The batch
 * @param type		The Type of plans (CUFFT or FFTW)
 */
void createFFTPlans(cuFFdotBatch* batch, presto_fft_type type)
{
  char msg[1024];

  PROF // Profiling  .
  {
    NV_RANGE_PUSH("FFT plans");
  }

  // Note creating the plans is the most expensive task in the GPU init, I tried doing it in parallel but it was slower
  for (int i = 0; i < batch->noStacks; i++)
  {
    cuFfdotStack* cStack  = &batch->stacks[i];

    PROF // Profiling  .
    {
      sprintf(msg,"Stack %i",i);
      NV_RANGE_PUSH(msg);
    }

    if ( (type == FFT_INPUT) || (type == FFT_BOTH) ) // Input FFT's  .
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
	if ( batch->flags & CU_INPT_FFT_CPU )
	{
	  infoMSG(5,5,"Creating single precision FFTW plan for input FFT.\n");

	  PROF // Profiling  .
	  {
	    NV_RANGE_PUSH("FFTW");
	  }

	  cStack->inpPlanFFTW = fftwf_plan_many_dft(1, n, cStack->noInStack*batch->noSteps, (fftwf_complex*)cStack->h_iData, n, istride, idist, (fftwf_complex*)cStack->h_iData, n, ostride, odist, -1, FFTW_ESTIMATE);

	  PROF // Profiling  .
	  {
	    NV_RANGE_POP("FFTW");
	  }
	}
	else
	{
	  infoMSG(5,5,"Creating Single CUFFT plan for input FFT.\n");

	  PROF // Profiling  .
	  {
	    NV_RANGE_PUSH("CUFFT Inp");
	  }

	  CUFFT_SAFE_CALL(cufftPlanMany(&cStack->inpPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, cStack->noInStack*batch->noSteps), "Creating plan for input data of stack.");

	  PROF // Profiling  .
	  {
	    NV_RANGE_POP("CUFFT Inp");
	  }
	}
      }
    }

    if ( (type == FFT_PLANE) || (type == FFT_BOTH) ) // inverse FFT's  .
    {
      if ( batch->flags & FLAG_DOUBLE )
      {
	infoMSG(5,5,"Creating double precision CUFFT plan for iFFT\n");

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

	  CUFFT_SAFE_CALL(cufftPlanMany(&cStack->plnPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, cStack->height*batch->noSteps), "Creating plan for complex data of stack.");

	  PROF // Profiling  .
	  {
	    NV_RANGE_POP("CUFFT Pln");
	  }
	}
      }
      else
      {
	infoMSG(5,5,"Creating single precision CUFFT plan for iFFT\n");

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

	  CUFFT_SAFE_CALL(cufftPlanMany(&cStack->plnPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, cStack->height*batch->noSteps), "Creating plan for complex data of stack.");

	  PROF // Profiling  .
	  {
	    NV_RANGE_POP("CUFFT Pln");
	  }
	}
      }
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP(msg);
    }

    CUDA_SAFE_CALL(cudaGetLastError(), "Creating FFT plans for the stacks.");
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP("FFT plans");
  }
}


/** Allocate R value array  .
 *
 */
void createRvals(cuFFdotBatch* batch, rVals** rLev1, rVals**** rAraays )
{
  rVals**   rLev2;

  int oSet		= 0;

  int no		= batch->noSteps*batch->noGenHarms*batch->noRArryas;
  int sz		= sizeof(rVals)*no;
  (*rLev1)		= (rVals*)malloc(sz);
  memset((*rLev1), 0, sz);
  for (int i1 = 0 ; i1 < no; i1++)
  {
    (*rLev1)[i1].step	= -1; // Invalid step (0 is a valid value!)
  }

  *rAraays		= (rVals***)malloc(batch->noRArryas*sizeof(rVals**));

  for (int rIdx = 0; rIdx < batch->noRArryas; rIdx++)
  {
    rLev2		= (rVals**)malloc(sizeof(rVals*)*batch->noSteps);
    (*rAraays)[rIdx]	= rLev2;

    for (int step = 0; step < batch->noSteps; step++)
    {
      rLev2[step]	= &((*rLev1)[oSet]);
      oSet		+= batch->noGenHarms;
    }
  }
}

/** Set the stack specific values from the harmonic information
 *
 * @param batch
 * @param h_inf
 * @param offset
 * @return
 */
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
    cStack->stkConstIdx   = offset+i;
    cStack->d_sInf        = dcoeffs + offset+i ;
  }

  return batch->noStacks;
}

/** Create multiplication kernel and allocate memory for planes on all devices  .
 *
 * Create the kernels on the first device and then copy it to all others
 *
 * @param sSrch     A pointer to the search structure
 *
 * @return
 */
void createGenKernels(cuSearch* cuSrch )
{
  infoMSG(2,1,"Create all planes.\n");

  if ( cuSrch->pInf )
  {
    infoMSG(5,5,"Planes data already initialised? Freeing\n");

    // This shouldn't be the case?
    freeAccelGPUMem(cuSrch->pInf);
    cuSrch->pInf = NULL;
  }

  cuSrch->pInf = new cuPlnInfo;
  memset(cuSrch->pInf, 0, sizeof(cuPlnInfo));

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering initCandGeneration.");

  FOLD // Create the primary batch on each device, this contains the kernel  .
  {
    // Wait for cuda context to complete
    compltCudaContext(cuSrch->gSpec);

    infoMSG(2,2,"Create the primary stack/kernel on each device\n");

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Init Kernels");
    }

    cuSrch->pInf->kernels = (cuFFdotBatch*)malloc(cuSrch->gSpec->noDevices*sizeof(cuFFdotBatch));
    memset(cuSrch->pInf->kernels, 0, cuSrch->gSpec->noDevices*sizeof(cuFFdotBatch));

    int added;
    cuFFdotBatch* master = NULL;

    for ( int dev = 0 ; dev < cuSrch->gSpec->noDevices; dev++ ) // Loop over devices  .
    {
      added = initKernel(&cuSrch->pInf->kernels[cuSrch->pInf->noDevices], master, cuSrch, dev );

      if ( added > 0 )
      {
	infoMSG(5,5,"%s - initKernel returned %i batches.\n", __FUNCTION__, added);

	if ( !master ) // This was the first batch so it is the master
	{
	  master = &cuSrch->pInf->kernels[0];
	}

	cuSrch->gSpec->noDevBatches[dev] = added;
	cuSrch->pInf->noBatches += added;
	cuSrch->pInf->noDevices++;
      }
      else
      {
	cuSrch->gSpec->noDevBatches[dev] = 0;
	fprintf(stderr, "ERROR: failed to set up a kernel on device %i, trying to continue... \n", cuSrch->gSpec->devId[dev]);
      }
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP("Init Kernels");
    }

    if ( cuSrch->pInf->noDevices <= 0 ) // Check if we got any devices  .
    {
      fprintf(stderr, "ERROR: Failed to set up a kernel on any device. Try -lsgpu to see what devices there are.\n");
      exit (EXIT_FAILURE);
    }

  }

  FOLD // Create planes for calculations  .
  {
    infoMSG(2,2,"Create Batches\n");

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Init Batches");
    }

    cuSrch->pInf->noSteps       = 0;
    cuSrch->pInf->batches       = (cuFFdotBatch*)malloc(cuSrch->pInf->noBatches*sizeof(cuFFdotBatch));
    cuSrch->pInf->devNoStacks   = (int*)malloc(cuSrch->gSpec->noDevices*sizeof(int));
    cuSrch->pInf->h_stackInfo   = (stackInfo**)malloc(cuSrch->gSpec->noDevices*sizeof(stackInfo*));

    memset(cuSrch->pInf->batches, 0, cuSrch->pInf->noBatches*sizeof(cuFFdotBatch));
    memset(cuSrch->pInf->devNoStacks,0,cuSrch->gSpec->noDevices*sizeof(int));

    int bNo = 0;
    int ker = 0;

    for ( int dev = 0 ; dev < cuSrch->gSpec->noDevices; dev++ ) // Loop over devices  .
    {
      int noSteps = 0;
      if ( cuSrch->gSpec->noDevBatches[dev] > 0 )
      {
	int firstBatch = bNo;

	infoMSG(5,5,"%s - device %i should have %i batches.\n", __FUNCTION__, dev, cuSrch->gSpec->noDevBatches[dev]);

	for ( int batch = 0 ; batch < cuSrch->gSpec->noDevBatches[dev]; batch++ )
	{
	  infoMSG(3,3,"Initialise batch %02i\n", bNo );

	  infoMSG(5,5,"%s - dev: %i - batch: %i - noBatches %i   \n", __FUNCTION__, cuSrch->gSpec->noDevBatches[dev], batch, cuSrch->gSpec->noDevBatches[dev] );

	  noSteps = initBatch(&cuSrch->pInf->batches[bNo], &cuSrch->pInf->kernels[ker], batch, cuSrch->gSpec->noDevBatches[dev]-1);

	  if ( noSteps == 0 )
	  {
	    if ( batch == 0 )
	    {
	      fprintf(stderr, "ERROR: Failed to create at least one batch on device %i.\n", cuSrch->pInf->kernels[dev].gInf->devid );
	    }
	    break;
	  }
	  else
	  {
	    infoMSG(3,3,"Successfully initialised %i steps in batch %i.\n", noSteps, batch+1);

	    cuSrch->pInf->noSteps           += noSteps;
	    cuSrch->pInf->devNoStacks[dev]  += cuSrch->pInf->batches[bNo].noStacks;
	    bNo++;
	  }
	}

	int noStacks = cuSrch->pInf->devNoStacks[dev] ;
	if ( noStacks )
	{
	  infoMSG(3,3,"\nInitialise constant memory for stacks\n" );

	  cuSrch->pInf->h_stackInfo[dev] = (stackInfo*)malloc(noStacks*sizeof(stackInfo));
	  int idx = 0;

	  // Set the values of the host data structures
	  for (int batch = firstBatch; batch < bNo; batch++)
	  {
	    idx += setStackInfo(&cuSrch->pInf->batches[batch], cuSrch->pInf->h_stackInfo[dev], idx);
	  }

	  if ( idx != noStacks )
	  {
	    fprintf (stderr,"ERROR: in %s line %i, The number of stacks on device do not match.\n.",__FILE__, __LINE__);
	  }
	  else
	  {
	    setConstStkInfo(cuSrch->pInf->h_stackInfo[dev], idx, cuSrch->pInf->batches->stacks->initStream);
	  }
	}

	ker++;
      }
    }

    if ( bNo != cuSrch->pInf->noBatches )
    {
      fprintf(stderr, "WARNING: Number of batches created does not match the number anticipated.\n");
      cuSrch->pInf->noBatches = bNo;
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP("Init Batches");
    }
  }

  infoMSG(2,2,"\n");
}

void initCandGeneration(cuSearch* cuSrch )
{
  struct timeval start, end;

  TIME // Basic timing of device setup and kernel creation  .
  {
      gettimeofday(&start, NULL);
      NV_RANGE_PUSH("Cand Gen Initialise");
  }

  bool createNew = false;
  if ( cuSrch->pInf )
  {
    for ( int i=0;i < cuSrch->pInf->noBatches; i++ )
    {
      if ( !compare( cuSrch->pInf->batches[i].conf, cuSrch->conf->gen ) )
      {
	infoMSG(5,5,"Planes data exists and differ, Freeing\n");
	createNew = true;
      }
      else
      {
	infoMSG(5,5,"Planes data exists and are the same (I hope!)\n");
      }
    }

    if ( createNew )
    {
      freeAccelGPUMem(cuSrch->pInf);
      cuSrch->pInf = NULL;
    }
  }
  else
  {
    createNew = true;
  }

  if ( createNew )
  {
    createGenKernels( cuSrch );
  }

  TIME // Basic timing of device setup and kernel creation  .
  {
    NV_RANGE_POP("GPU Initialise");

    gettimeofday(&end, NULL);
    cuSrch->timings[TIME_GPU_INIT] += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
  }


}

///////////////////////// Free  /////////////////////////////////////////

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

/** Free batch data structure  .
 *
 * @param batch
 */
void freeBatchGPUmem(cuFFdotBatch* batch)
{
  CUDA_SAFE_CALL(cudaGetLastError(), "Entering freeBatchGPUmem.");

  setDevice(batch->gInf->devid) ;

  infoMSG(2,2,"freeBatchGPUmem %i\n", batch);

  FOLD // Free host memory
  {
    infoMSG(3,3,"Free host memory\n");

    freeNull(batch->h_normPowers);
  }

  FOLD // Free pinned memory
  {
    infoMSG(3,3,"Free pinned memory\n");

    cudaFreeHostNull(batch->h_iData);
    freeNull(batch->h_iBuffer);			// This could be cudaFreeHostNull, is this memory ever allocated "normally"
    //cudaFreeHostNull(batch->h_outData1);
  }

  FOLD // Free device memory
  {
    infoMSG(3,3,"Free device memory\n");

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
  }

//  FOLD // Free textures for the f-∂f planes  .
//  {
//    if ( batch->flags & FLAG_SAS_TEX )
//    {
//      infoMSG(2,2,"Free textures\n");
//
//      for (int i = 0; i < batch->noStacks; i++)
//      {
//	cuFfdotStack* cStack = &batch->stacks[i];
//
//	for (int j = 0; j< cStack->noInStack; j++)
//	{
//	  cuFFdot* cPlane = &cStack->planes[j];
//
//	  if ( cPlane->datTex )
//	  {
//	    CUDA_SAFE_CALL(cudaDestroyTextureObject(cPlane->datTex), "Creating texture from the plane data.");
//	    cPlane->datTex = (fCplxTex)0;
//	  }
//	}
//      }
//      CUDA_SAFE_CALL(cudaGetLastError(), "Creating textures from the plane data.");
//    }
//  }

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

/** Free the memory allocated to store the R-values of a batch
 *
 * @param batch		The batch
 * @param rLev1
 * @param rAraays
 */
void freeRvals(cuFFdotBatch* batch, rVals** rLev1, rVals**** rAraays )
{
  infoMSG(3,3,"Free r-araays\n");

  // TODO: Check this function again

  if (*rAraays)
  {
    for (int i = 0; i < batch->noRArryas; i++)
    {
      rVals* rVal = &(((*batch->rAraays)[i])[0][0]);
      cudaFreeHostNull( rVal->h_outData );

      rVals**   rLev2;

      rLev2 = (*rAraays)[i];

      freeNull(rLev2);
    }

    freeNull(*rAraays);
  }

  if (*rLev1)
  {
    freeNull(*rLev1);
  }
}

void freeAccelGPUMem(cuPlnInfo* aInf)
{
  infoMSG(1,1,"Free all batch Mem\n");

  PROF // Profiling  .
  {
    NV_RANGE_PUSH("Free GPU Mem");
  }

  FOLD // Free planes  .
  {
    for ( int batch = 0 ; batch < aInf->noBatches; batch++ )  // Batches
    {
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
    NV_RANGE_POP("Free GPU Mem");
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

////////////////////////// Clear /////////////////////////////////


/** Clear all the values of a r-Value data struct
 *
 * @param rVal	A pointer to the struct to clear
 */
void clearRval( rVals* rVal)
{
  // NB this will not clear the pointer to host pinned memory

  if ( rVal->outBusy)
  {
    // This is actually OK,
    infoMSG(5,5,"Clearing a busy step %i %i (%p)", rVal->iteration, rVal->step, &rVal->outBusy );
  }

  rVal->drlo		= 0;
  rVal->drhi		= 0;
  rVal->lobin		= 0;
  rVal->numrs		= 0;
  rVal->numdata		= 0;
  rVal->expBin		= 0;
  rVal->norm		= 0;

  rVal->step		= -1; // Invalid step!
  rVal->iteration	= -1;
}

/** Clear all the r-Value data structs of a batch
 *
 * @param batch	A pointer to the batch who's r-Value data structs are to be cleared
 */
void clearRvals(cuFFdotBatch* batch)
{
  infoMSG(6,6,"Clearing array of r step information.");

  for ( int i = 0; i < batch->noSteps*batch->noGenHarms*batch->noRArryas; i++ )
  {
    clearRval(&batch->rArr1[i]);
  }
}

////////////////////////// Set pointers of data structures ////////////////////////////

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
    infoMSG(6,6,"stack %i\n", i);

    // Set stack pointers
    cuFfdotStack* cStack  = &batch->stacks[i];

    for (int j = 0; j < cStack->noInStack; j++)
    {
      infoMSG(6,7,"plane %i\n", j);

      cuFFdot* cPlane           = &cStack->planes[j];

      if ( batch->flags & FLAG_DOUBLE )
	cPlane->d_planeMult       = &((double2*)cStack->d_planeMult)[ cStack->startZ[j] * batch->noSteps * cStack->strideCmplx ];
      else
	cPlane->d_planeMult       = &((float2*)cStack->d_planeMult) [ cStack->startZ[j] * batch->noSteps * cStack->strideCmplx ];

      if (cStack->d_planePowr)
      {
	if ( batch->flags & FLAG_POW_HALF )
	{
#if CUDART_VERSION >= 7050
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
#if CUDART_VERSION >= 7050
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

/** Set the pointer to the various sections of the multiplication kernel  .
 *
 * @param batch
 */
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
      d_kerData		= &((dcomplexcu*)batch->d_kerData)[kerSiz];
    else
      d_kerData		= &((fcomplexcu*)batch->d_kerData)[kerSiz];

    // Set the individual kernel information parameters
    for (int j = 0; j < cStack->noInStack; j++)
    {
      // Point the plane kernel data to the correct position in the "main" kernel
      cStack->kernels[j].kreOff		= cu_index_from_z<double>(cStack->harmInf[j].zStart, cStack->harmInf->zStart, batch->conf->zRes);
      cStack->kernels[j].stride		= cStack->strideCmplx;

      if ( batch->flags & FLAG_DOUBLE )
	cStack->kernels[j].d_kerData	= &((dcomplexcu*)d_kerData)[cStack->strideCmplx*cStack->kernels[j].kreOff];
      else
	cStack->kernels[j].d_kerData	= &((fcomplexcu*)d_kerData)[cStack->strideCmplx*cStack->kernels[j].kreOff];

      cStack->kernels[j].harmInf	= &cStack->harmInf[j];
    }
    kerSiz				+= cStack->strideCmplx * cStack->kerHeigth;
  }
}


////////////////////////   GPU Constant Memory   /////////////////////////////////////


/** Set the GPU constant memory values specific to the batches
 *
 * @param batch
 * @return
 */
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
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, &kerPnt, MAX_HARM_NO * sizeof(void*), cudaMemcpyHostToDevice, batch->stacks->initStream),     "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, KERNEL_OFF_HARM);
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, &ker_off, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice, batch->stacks->initStream),      "Copying stages to device");
  }

  CUDA_SAFE_CALL(cudaGetLastError(), "Preparing the constant memory values for the multiplications.");

  return 1;
}

/** Set the GPU constant memory data specific to each stack  .
 *
 * A pointer to stack specific constant memory is passed to CUFFT callbacks
 * Here is where this constant memory is populated
 *
 * @param batch
 * @return
 */
int setStackVals( cuFFdotBatch* batch )
{
#ifdef WITH_MUL_PRE_CALLBACK
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
      for (int j = 0; j < cStack->noInStack; j++)	// Loop through planes in stack
      {
	cuHarmInfo*  hInf = &cStack->harmInf[j];

	// Create the actual texture object
	for (int k = 0; k < batch->noSteps; k++)	// Loop through planes in stack
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
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, l_STK_STRD, sizeof(l_STK_STRD), cudaMemcpyHostToDevice, batch->stacks->initStream),		"Copying stack info to device");

    cudaGetSymbolAddress((void **)&dcoeffs, STK_INP );
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, l_STK_INP, sizeof(l_STK_INP), cudaMemcpyHostToDevice, batch->stacks->initStream),		"Copying stack info to device");
  }

  return 1;
#else
  return 0;
#endif
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
  infoMSG(5,5,"set ConstStkInfo to %i\n", noStacks );

  void *dcoeffs;

  // TODO: Do a test to see if  we are on the correct device

  cudaGetSymbolAddress((void **)&dcoeffs, STACKS);
  CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, h_inf, noStacks * sizeof(stackInfo), cudaMemcpyHostToDevice, stream),      "Copying stack info to device");

  return 1;
}



///////////////////////////  R-Value arrays  ////////////////////////////

/** Cycle the arrays of r-values  .
 *
 * @param batch
 */
void cycleRlists(cuFFdotBatch* batch)
{
  infoMSG(4,4,"Cycle R lists\n");

  rVals** hold = (*batch->rAraays)[batch->noRArryas-1];
  for ( int i = batch->noRArryas-1; i > 0; i-- )
  {
    (*batch->rAraays)[i] =  (*batch->rAraays)[i - 1];
  }
  (*batch->rAraays)[0] = hold;
}

/** Cycle the arrays of r-values  .
 *
 * @param batch
 */
void CycleBackRlists(cuFFdotBatch* batch)
{
  infoMSG(4,4,"CycleBackRlists\n");

  rVals** hold = (*batch->rAraays)[0];
  for ( int i = 0; i < batch->noRArryas-1; i++ )
  {
    (*batch->rAraays)[i] =  (*batch->rAraays)[i + 1];
  }

  (*batch->rAraays)[batch->noRArryas-1] = hold;
}

void cycleOutput(cuFFdotBatch* batch)
{
  infoMSG(4,4,"Cycle output\n");

  void* d_hold		= batch->d_outData1;
  batch->d_outData1	= batch->d_outData2;
  batch->d_outData2	= d_hold;
}



///////////////////////// Searches ////////////////////////////////////

/** One iteration of the candidate generation stage
 *
 * @param batch
 */
void search_ffdot_batch_CU(cuFFdotBatch* batch)
{
  CUDA_SAFE_CALL(cudaGetLastError(), "Entering search_ffdot_batch_CU.");

  PROF
  {
    setActiveIteration(batch, 0);
    rVals* rVal = &((*batch->rAraays)[batch->rActive][0][0]);
    infoMSG(1,1,"\nIteration %4i - Start step %4i,  processing %02i steps on GPU %i  Start bin: %9.2f \n", rVal->iteration,rVal->step, batch->noSteps, batch->gInf->devid, rVal->drlo );
  }

  if ( batch->flags & FLAG_SYNCH )
  {
    if  ( batch->flags & FLAG_SS_INMEM )
    {
      setActiveIteration(batch, 1);
      multiplyBatch(batch);

      setActiveIteration(batch, 1);
      IFFTBatch(batch);

      setActiveIteration(batch, 1);
      copyToInMemPln(batch);

      // Setup input
      setActiveIteration(batch, 0);
      prepInput(batch);
    }
    else
    {
      if (batch->cuSrch->pInf->noBatches > 1 ) // This is true synchronise behaviour, but that is over kill  .
      {
	// Setup input
	setActiveIteration(batch, 0);
	prepInput(batch);

	setActiveIteration(batch, 0);
	multiplyBatch(batch);

	setActiveIteration(batch, 0);
	IFFTBatch(batch);

	setActiveIteration(batch, 0);
	sumAndSearch(batch);

	setActiveIteration(batch, 0);
	getResults(batch);

	setActiveIteration(batch, 0);
	processBatchResults(batch);
      }
      else					// This overlaps CPU and GPU but each runs its stuff synchronise, good enough for timing and a bit faster
      {
	setActiveIteration(batch, 2);		// This will block on getResults, so it must be 1 more than that to allow CUDA kernels to run
	processBatchResults(batch);

	setActiveIteration(batch, 1);
	sumAndSearch(batch);

	setActiveIteration(batch, 1);
	getResults(batch);

	// Setup input
	setActiveIteration(batch, 0);
	prepInput(batch);

	setActiveIteration(batch, 0);
	multiplyBatch(batch);

	setActiveIteration(batch, 0);
	IFFTBatch(batch);
      }
    }
  }
  else
  {

    if  ( batch->flags & FLAG_SS_INMEM )
    {
      setActiveIteration(batch, 0);
      prepInput(batch);
      multiplyBatch(batch);
      IFFTBatch(batch);
      copyToInMemPln(batch);
    }
    else
    {
      if      ( batch->conf->cndProcessDelay == 0)
      {
	// This is synchronous processing of a single iteration

	setActiveIteration(batch, 0);
	prepInput(batch);
	convolveBatch(batch);
	sumAndSearch(batch);
	getResults(batch);
	processBatchResults(batch);
      }
      else if ( batch->conf->cndProcessDelay == 1)
      {
	// This is good and will overlap GPU kernels as well as CPU computation

	setActiveIteration(batch, 0);
	prepInput(batch);
	convolveBatch(batch);
	sumAndSearch(batch);

	setActiveIteration(batch, 1);
	processBatchResults(batch);

	setActiveIteration(batch, 0);
	getResults(batch);
      }
      else if ( batch->conf->cndProcessDelay == 2)
      {
	// I generally find this is the best option especially with smaller z-max

	setActiveIteration(batch, 0);
	prepInput(batch);
	convolveBatch(batch);

	setActiveIteration(batch, 2);
	processBatchResults(batch);

	setActiveIteration(batch, 1);
	getResults(batch);

	setActiveIteration(batch, 0);
	sumAndSearch(batch);
      }
      else if ( batch->conf->cndProcessDelay == 3)
      {
	setActiveIteration(batch, 0);
	prepInput(batch);

	setActiveIteration(batch, 3);
	processBatchResults(batch);

	setActiveIteration(batch, 2);
	getResults(batch);

	setActiveIteration(batch, 1);
	sumAndSearch(batch);

	setActiveIteration(batch, 0);
	convolveBatch(batch);

      }
      else if ( batch->conf->cndProcessDelay == 4)
      {
	setActiveIteration(batch, 0);
	prepInput(batch);

	setActiveIteration(batch, 4);
	processBatchResults(batch);

	setActiveIteration(batch, 3);
	getResults(batch);

	setActiveIteration(batch, 2);
	sumAndSearch(batch);

	setActiveIteration(batch, 1);
	IFFTBatch(batch);

	setActiveIteration(batch, 0);
	multiplyBatch(batch);
      }
      else
      {
	fprintf(stderr, "ERROR: invalid value of cndProcessDelay  AKA  CAND_DELAY");
	exit(EXIT_FAILURE);
      }
    }
  }

  // Change R-values
  cycleRlists(batch);
  setActiveIteration(batch, 1);
}

void finish_Search(cuFFdotBatch* batch)
{
  infoMSG(1,1,"Finish search\n");

  FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host
  {
    for (int stack = 0; stack < batch->noStacks; stack++)
    {
      cuFfdotStack* cStack = &batch->stacks[stack];

      infoMSG(4,4,"Blocking synchronisation on %s stack %i", "ifftMemComp", stack );

      PROF // Profiling  .
      {
	NV_RANGE_PUSH("EventSynch");
      }

      CUDA_SAFE_CALL(cudaEventSynchronize(cStack->ifftMemComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");

      PROF // Profiling  .
      {
	NV_RANGE_POP("EventSynch");
      }
    }

    FOLD
    {
      infoMSG(4,4,"Blocking synchronisation on %s", "processComp" );

      PROF // Profiling  .
      {
	NV_RANGE_PUSH("EventSynch");
      }

      CUDA_SAFE_CALL(cudaEventSynchronize(batch->processComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");

      PROF // Profiling  .
      {
	NV_RANGE_POP("EventSynch");
      }
    }
  }
}

int genPlane(cuSearch* cuSrch, char* msg)
{
  infoMSG(2,2,"\nCandidate generation");

  struct timeval start01, end;
  struct timeval start02;
  double startr		= 0;				/// The first bin to start searching at
  double cuentR		= 0;				/// The start bin of the input FFT to process next
  double noR		= 0;				/// The number of input FFT bins the search covers
  double noSteps	= 0;				/// The number of steps to generate the initial candidates
  cuFFdotBatch* master	= &cuSrch->pInf->kernels[0];	/// The first kernel created holds global variables
  int iteration 	= 0;				/// Actual loop iteration
  int   step		= 0;				/// The next step to be processed (each iteration can handle multiple steps)
  ACC_ERR_CODE ret	= ACC_ERR_NONE;
  confSpecsGen*	conf	= cuSrch->conf->gen;

  TIME // Basic timing  .
  {
    NV_RANGE_PUSH("Pln Gen");
    gettimeofday(&start01, NULL);
  }

  FOLD // Set the bounds of the search  .
  {
    // Search bounds
    startr		= cuSrch->sSpec->searchRLow;
    noR			= cuSrch->sSpec->noSearchR;
    noSteps		= noR * conf->noResPerBin / (double)master->accelLen ;
    cuentR 		= startr;

    fflush(stdout);
    fflush(stderr);
    infoMSG(1,0,"\nGPU loop will process %i steps\n", ceil(noSteps) );
  }

#if !defined(DEBUG) && defined(WITHOMP)   // Parallel if we are not in debug mode  .
  if ( conf->flags & FLAG_SYNCH )
  {
    // NOTE: this uses the search flags not the batch specific flags, but FLAG_SYNCH should be set before initialising the kernels
    infoMSG(4,4,"Throttling to 1 thread");
    omp_set_num_threads(1);
  }
  else
  {
    omp_set_num_threads(cuSrch->pInf->noBatches);
  }

#pragma omp parallel
#endif	// !DEBUG && WITHOMP
  FOLD  //					---===== Main Loop =====---  .
  {
    // These are all thread specific variables
    int tid = 0;
#ifdef	WITHOMP
    tid = omp_get_thread_num();
#endif	// WITHOMP

    cuFFdotBatch* batch		= &cuSrch->pInf->batches[tid];				///< Thread specific batch to process
    int		firstStep	= 0;							///< Thread specific value for the first step the batch is processing
    double	firstR		= 0;							///< Thread specific value for the first input FT bin index being searched
    int		ite		= 0;							///< The iteration the batch is working on (local to each thread)

    // Set the device this thread will be using
    setDevice(batch->gInf->devid) ;

    // Make sure kernel create and all constant memory reads and writes are complete
    CUDA_SAFE_CALL(cudaDeviceSynchronize(), "Synchronising device before candidate generation");

    FOLD // Clear the r array  .
    {
      clearRvals(batch);

#ifndef  DEBUG
      if ( conf->flags & FLAG_SYNCH )
#endif
      {
	// If running in synchronous mode use multiple batches, just synchronously so clear all batches
	for ( int bId = 0; bId < cuSrch->pInf->noBatches; bId++ )
	{
	  batch = &cuSrch->pInf->batches[bId];
	  clearRvals(batch);
	}
      }
    }

    while ( cuentR < cuSrch->sSpec->searchRHigh )  //			---===== Main Loop =====---  .
    {
      FOLD // Calculate the step(s) to handle  .
      {
#pragma omp critical		// Calculate the step(s) this batch is processing  .
	FOLD
	{
	  FOLD  // Synchronous behaviour  .
	  {
#ifndef  DEBUG
	    if ( conf->flags & FLAG_SYNCH )
#endif
	    {
	      // If running in synchronous mode use multiple batches, just synchronously
	      tid = iteration % cuSrch->pInf->noBatches ;
	      batch = &cuSrch->pInf->batches[tid];
	      setDevice(batch->gInf->devid) ;			// Switch over to applicable device
	    }
	  }

	  iteration++;
	  ite 		= iteration;
	  firstStep 	= step;
	  step		+= batch->noSteps;

	  firstR	= cuentR;
	  cuentR	+= batch->noSteps * batch->accelLen / (double)conf->noResPerBin ;
	}

	if ( firstR > cuSrch->sSpec->searchRHigh )
	{
	  break;
	}
      }

      FOLD // Set start r-vals for all steps in this batch  .
      {
	ret = startBatchR (batch, firstR, ite, firstStep ) & (~ACC_ERR_OVERFLOW);
	ERROR_MSG(ret, "ERROR: Setting batch r location");

//	for ( int batchStep = 0; batchStep < (int)batch->noSteps ; batchStep++ )
//	{
//	  rVals* rVal = &(*batch->rAraays)[0][batchStep][0];
//	  clearRval(rVal);
//
//	  // Set the bounds of the fundamental
//	  rVal->drlo		= firstR + batchStep * ( batch->accelLen / (double)conf->noResPerBin );
//	  rVal->drhi		= rVal->drlo + ( batch->accelLen - 1 ) / (double)conf->noResPerBin;
//
//	  if ( rVal->drlo < cuSrch->sSpec->searchRHigh )
//	  {
//	    // Set step and iteration for all harmonics
//	    for ( int harm = 0; harm < batch->noGenHarms; harm++)
//	    {
//	      rVal		= &(*batch->rAraays)[0][batchStep][harm];
//
//	      rVal->step	= firstStep + batchStep;
//	      rVal->iteration   = ite;
//	      rVal->norm	= 0.0;
//	    }
//	  }
//	  else
//	  {
//	    // Not actually a valid step
//	    rVal->drlo		= 0;
//	    rVal->drhi		= 0;
//	  }
//	}
      }

      FOLD // Call the CUDA search  .
      {
	search_ffdot_batch_CU(batch);
      }

      FOLD // Print message  .
      {
	if ( msgLevel == 0  )
	{
	  double per = (cuentR - startr)/noR *100.0;

	  if      ( master->flags & FLAG_SS_INMEM )
	  {
	    printf("\r%s  %5.1f%%", msg, per);
	  }
	  else
	  {
	    int noTrd;
	    sem_getvalue(&master->cuSrch->threasdInfo->running_threads, &noTrd );
	    printf("\r%s  %5.1f%% ( %3i Active CPU threads processing initial candidates)  ", msg, per, noTrd);
	  }

	  fflush(stdout);
	}
      }
    }

    FOLD  // Finish off CUDA search  .
    {
      infoMSG(1,0,"\nFinish off plane.\n");

      // Finish searching the planes, this is required because of the out of order asynchronous calls
      for ( int rest = 0 ; rest < batch->noRArryas; rest++ )
      {
	FOLD // Set the r arrays to zero  .
	{
	  rVals* rVal = (*batch->rAraays)[0][0];
	  clearRval(rVal); // Clear the fundamental
	}

	search_ffdot_batch_CU(batch);
      }

      // Wait for asynchronous execution to complete
      finish_Search(batch);


#ifndef  DEBUG
      if ( conf->flags & FLAG_SYNCH )
#endif
      {
	// If running in synchronous mode use multiple batches, just synchronously so clear all batches
	for ( int bId = 0; bId < cuSrch->pInf->noBatches; bId++ )
	{
	  infoMSG(1,0,"\nFinish off plane (synch batch %i).\n", bId);

	  batch = &cuSrch->pInf->batches[bId];

	  // Finish searching the planes, this is required because of the out of order asynchronous calls
	  for ( int rest = 0 ; rest < batch->noRArryas; rest++ )
	  {
	    FOLD // Set the r arrays to zero  .
	    {
	      rVals* rVal = (*batch->rAraays)[0][0];
	      clearRval(rVal); // Clear the fundamental
	    }

	    search_ffdot_batch_CU(batch);
	  }

	  // Wait for asynchronous execution to complete
	  finish_Search(batch);
	}
      }
    }
  }

  printf("\r%s. %5.1f%%                                                                                         \n", msg, 100.0);

  TIME // Basic timing  .
  {
    gettimeofday(&start02, NULL);
  }

  FOLD // Wait for CPU threads to complete  .
  {
    waitForThreads(&master->cuSrch->threasdInfo->running_threads, "Waiting for CPU thread(s) to finish processing returned from the GPU ", 200 );
  }

  TIME // Basic timing  .
  {
    NV_RANGE_POP("Pln Gen");

    gettimeofday(&end, NULL);
    cuSrch->timings[TIME_GPU_PLN] += (end.tv_sec - start01.tv_sec) * 1e6 + (end.tv_usec - start01.tv_usec);
    cuSrch->timings[TIME_GEN_WAIT] += (end.tv_sec - start02.tv_sec) * 1e6 + (end.tv_usec - start02.tv_usec);
  }

  return 0;
}

/** Generate initial candidates using a pre-initialised search structure  .
 *
 * @param cuSrch
 * @param gSpec
 * @param sSpec
 * @return
 */
GSList* generateCandidatesGPU(cuSearch* cuSrch)
{
  struct timeval start, end;
  struct timeval start01, end01;
  struct timeval start02, end02;
  cuFFdotBatch* master;
  long noCands = 0;

  gpuSpecs*	gSpec	= cuSrch->gSpec;
  confSpecsGen*	conf	= cuSrch->conf->gen;

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
    initCandGeneration(cuSrch);

    master    = &cuSrch->pInf->kernels[0];   // The first kernel created holds global variables

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

    int noGenSteps = ceil( (cuSrch->sSpec->searchRHigh - cuSrch->sSpec->searchRLow/(double)cuSrch->noGenHarms) * conf->noResPerBin / (double)master->accelLen );
    printf("\nRunning GPU search of %i steps with %i simultaneous families of f-∂f planes spread across %i device(s).\n\n", noGenSteps, cuSrch->pInf->noSteps, cuSrch->pInf->noDevices );

    if      ( master->flags & FLAG_SS_INMEM     )	// In-mem search  .
    {
      if ( master->flags & FLAG_Z_SPLIT )		// Z-Split  .
      {
	setInMemPlane(cuSrch, IM_TOP);
	sprintf(srcTyp, "Generating top half in-mem GPU plane");
	genPlane(cuSrch, srcTyp);
	inmemSumAndSearch(cuSrch);

	setInMemPlane(cuSrch, IM_BOT);
	sprintf(srcTyp, "Generating bottom half in-mem GPU plane");
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
      // This includes the search (standard)
    }

    TIME // Basic timing  .
    {
      gettimeofday(&end01, NULL); // This is not used???
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

	  ulong     cdx;
	  double  poww, sig;
	  double  rr, zz;
	  int     added = 0;
	  int     numharm;
	  initCand*   candidate = (initCand*)cuSrch->h_candidates;
	  poww    = 0;
	  ulong max = cuSrch->candStride;

	  if ( master->flags  & FLAG_STORE_ALL )
	    max *= master->noHarmStages; // Store  candidates for all stages

	  for (cdx = 0; cdx < max; cdx++)  // Loop
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
	    NV_RANGE_POP("Add to list");
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
      NV_RANGE_POP("GPU Cand");
      NV_RANGE_POP("Cand Gen");
      gettimeofday(&end02, NULL);
      cuSrch->timings[TIME_GPU_CND_GEN] += (end02.tv_sec - start01.tv_sec) * 1e6 + (end02.tv_usec - start01.tv_usec);
      cuSrch->timings[TIME_CND] += (end02.tv_sec - start02.tv_sec) * 1e6 + (end02.tv_usec - start02.tv_usec);
    }
  }

  FOLD // Free GPU memory  .
  {
    freeAccelGPUMem(cuSrch->pInf);
  }

  printf("\nGPU found %li initial candidates of which %i are unique", noCands, g_slist_length(cuSrch->cands));

  TIME // Basic timing  .
  {
    NV_RANGE_POP("GPU Srch");
    gettimeofday(&end, NULL);

    cuSrch->timings[TIME_GPU_SRCH] += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);

    printf(", it Took %.4f ms", cuSrch->timings[TIME_GPU_SRCH]/1000.0);
  }

  printf(".\n");

  return cuSrch->cands;
}
