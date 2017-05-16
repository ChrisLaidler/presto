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
 *
 *  [2017-05-12]
 *  	Massive refactor. Moved a bunch of stuff to a separate candidate generation stage file
 *	Added the printErrors capability
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
#include "cuda_cand_OPT.h"

#ifdef CBL
#include <unistd.h>
#include "log.h"
#endif

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


void __printErrors( ACC_ERR_CODE value, const char* file, int lineNo, const char* errorMsg)
{
  if (value)
  {
    char msg[1024] = {0};

    if ( errorMsg )
      sprintf(msg, "\n%s", errorMsg );
    else
      sprintf(msg, "ERROR: Unspecified. ( Someone was being lazy... )" );
    sprintf(msg, "\n%s\n  On line: %4i in %s\n  Error codes:\n", msg, lineNo, file );

    if (value & ACC_ERR_NAN )
    {
      value &= (~ACC_ERR_NAN);
      sprintf(msg, "%s     NAN \n", msg);
    }

    if (value & ACC_ERR_NEG )
    {
      value &= (~ACC_ERR_NEG);
      sprintf(msg, "%s     Negative value \n", msg);
    }

    if (value & ACC_ERR_STRIDE )
    {
      value &= (~ACC_ERR_STRIDE);
      sprintf(msg, "%s     Invalid stride \n", msg);
    }

    if (value & ACC_ERR_ALIGHN )
    {
      value &= (~ACC_ERR_ALIGHN);
      sprintf(msg, "%s     Alignment\n", msg);
    }

    if (value & ACC_ERR_OVERFLOW )
    {
      value &= (~ACC_ERR_OVERFLOW);
      sprintf(msg, "%s     Overflow\n", msg );
    }

    if (value & ACC_ERR_OUTOFBOUNDS )
    {
      value &= (~ACC_ERR_OUTOFBOUNDS);
      sprintf(msg, "%s     Out of bounds\n", msg );
    }

    if (value & ACC_ERR_NULL )
    {
      value &= (~ACC_ERR_NULL);
      sprintf(msg, "%s     NULL pointer\n", msg );
    }




    if (value )
    {
      sprintf(msg, "%s     Unknown? %i \n", msg, value);
    }

    fprintf(stderr, "%s\n", msg);
  }
}

void setDebugMsgLevel(int lvl)
{
  msgLevel = lvl;
}

/** Calculate an optimal accellen given a width  .
 *
 * @param width		The width of the plane (usually a power of two) if width < 100 the closes power of 2 to width*1000 will be used ie 8 -> 8024
 * @param zmax		The highest z value being searched for
 * @param accuracy	The accuracy of the kernel
 * @param noResPerBin	The resolution 2 -> interbinning
 * @return
 * If width is not a power of two it will be rounded up to the nearest power of two
 */
uint optAccellen(float width, float zmax, presto_interp_acc accuracy, int noResPerBin)
{
  double halfwidth	= cu_z_resp_halfwidth<double>(zmax, accuracy); /// The halfwidth of the maximum zmax, to calculate step size
  double pow2		= pow(2 , round(log2(width)) );
  uint oAccelLen	= floor(pow2 - 2 - 2 * halfwidth * noResPerBin );	// NOTE: I think the extra ( - 2 ) in here is not needed?

  infoMSG(6,6,"For a width %.0f and z-max %.1f with spacing %i, Halfwidth is %.0f and step size is %u. \n", pow2, zmax, noResPerBin, halfwidth, oAccelLen );
  return oAccelLen;
}

/** Calculate the step size from a width if the width is < 100 it is scaled to be the closest power of two  .
 *
 * @param width		The width of the plane (usually a power of two) if width < 100 the closes power of 2 to width*1000 will be used ie 8 -> 8024
 * @param zmax		The highest z value being searched for
 * @param accuracy	The accuracy of the kernel
 * @param noResPerBin	The resolution 2 -> interbinning
 * @return		The step size
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

/** Calculate the step size from a width if the width is < 100 it is skate to be the closest power of two  .
 *
 * @param width		The width of the plane (usually a power of two) if width < 100 the closes power of 2 to width*1000 will be used ie 8 -> 8024
 * @param zmax		The highest z value being searched for
 * @param noHarms	The number of harmonics being summed ( power of 2 )
 * @param accuracy	The accuracy of the kernel
 * @param noResPerBin	The resolution 2 -> interbinning
 * @param zRes		The resolution of the z values
 * @param hamrDevis	Make sure the width is divisible by the number of harmonics (needed for CUDA sum and search)
 * @return		The step size
 */
uint calcAccellen(float width, float zmax, int noHarms, presto_interp_acc accuracy, int noResPerBin, float zRes, bool hamrDevis)
{
  infoMSG(5,5,"Calculating step size\n");

  uint	accelLen, oAccelLen1, oAccelLen2;

  oAccelLen1  = calcAccellen(width, zmax, accuracy, noResPerBin);
  infoMSG(6,6,"Initial optimal step size %i for a fundamental plane of width %.0f with z-max %.1f \n", oAccelLen1, width, zmax);

  if ( width > 100 )				// The user specified the exact width they want to use for accellen  .
  {
    accelLen  = oAccelLen1;
    infoMSG(6,6,"User specified step size %.0f - using: %i \n", width, oAccelLen1);
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

fftInfo* readFFT(char* fileName)
{
  char name[1024] = {0};
  char* suffix;
  int loc;

  printf("Opening %s \n", fileName);

  suffix = strrchr(fileName, '.');
  if ( suffix == NULL )
    return NULL;
  if (strcmp(++suffix, "fft") == 0)
  {
    fftInfo* fft = new fftInfo;
    memset(fft, 0, sizeof(fftInfo));

    FILE* fftFile	= chkfopen(fileName, "rb");
    long long filelen	= chkfilelen(fftFile, sizeof(fcomplex));
    fft->data		= (fcomplex*)malloc(sizeof(fcomplex) * filelen );
    long long fileRead	= chkfread(fft->data, sizeof(fcomplex), filelen, fftFile);
    fclose(fftFile);

    infodata * idata = new(infodata);
    //--suffix[0]=0;	// Remove suffix
    strncpy(name, fileName, suffix - fileName-1 );
    readinf(idata, name);

    fft->firstBin = 0;
    fft->lastBin = fileRead-1;
    fft->noBins = fileRead;

    fft->N = idata->N;
    fft->dt = idata->dt;
    fft->T = fft->N * fft->dt;

    return fft;
  }

  return NULL;
}

/** Set the search size parameters
 *
 * This calculates the search size parameters from the FFT, number of harmonics being summed, halfwidth and resolution
 *
 */
void setSrchSize(searchSpecs* SrchSz, int halfWidth, int noHarms, int alighnment)
{
  SrchSz->searchRHigh	= ceil  ( SrchSz->specRHigh );
  SrchSz->searchRLow	= floor ( SrchSz->specRLow  / (double)noHarms / (double)alighnment ) * alighnment ;
  SrchSz->noSearchR	= SrchSz->searchRHigh - SrchSz->searchRLow ;		/// Determine the number of candidate 'r' values

  SrchSz->rLow		= SrchSz->searchRLow  - halfWidth ;
  SrchSz->rHigh		= SrchSz->searchRHigh + halfWidth ;
  SrchSz->noInpR	= SrchSz->rHigh - SrchSz->rLow  ;  			/// The number of input data points
}

/**
 *
 * @param device
 * @param print
 * @return
 */
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
    fprintf(myfile, "%4s\t%14s\t%10s\t%14s\t%13s\t%9s\t%7s\t%2s \n", "#", "r", "f", "z", "fd", "sig", "power", "harm" );
    int i = 0;

    while ( inp_list )
    {
      fprintf(myfile, "%4i\t%14.5f\t%10.6f\t%14.2f\t%13.10f\t%9.4f\t%7.2f\t%2i\n", i+1, ((accelcand *) (inp_list->data))->r, ((accelcand *) (inp_list->data))->r / T, ((accelcand *) (inp_list->data))->z,((accelcand *) (inp_list->data))->z/T/T, ((accelcand *) (inp_list->data))->sigma, ((accelcand *) (inp_list->data))->power, ((accelcand *) (inp_list->data))->numharm );
      inp_list = inp_list->next;
      i++;
    }
    fclose ( myfile );
  }
}

void initGPUs(gpuSpecs* gSpec)
{
  int currentDevvice, deviceCount;
  char txt[1024];

  int major           = 0;
  int minor           = 0;

  CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount), "Failed to get device count using cudaGetDeviceCount");

  for (int dIdx = 0; dIdx < gSpec->noDevices; dIdx++)
  {
    int device    = gSpec->devId[dIdx];
    gpuInf* gInf  = &gSpec->devInfo[dIdx];

    CUDA_SAFE_CALL( cudaSetDevice ( device ), "Failed to set device using cudaSetDevice");

    // Check if the the current device is 'device'
    CUDA_SAFE_CALL( cudaGetDevice(&currentDevvice), "Failed to get device using cudaGetDevice" );

    if ( currentDevvice != device)
    {
      fprintf(stderr, "ERROR: Device not set.\n");
    }
    else // call something to initialise the device
    {
      sprintf(txt,"Init device %02i", device );

      PROF // Profiling  .
      {
	NV_RANGE_PUSH(txt);
      }

      cudaDeviceProp deviceProp;
      CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, device), "Failed to get device properties device using cudaGetDeviceProperties");

      major                           = deviceProp.major;
      minor                           = deviceProp.minor;
      gInf->capability                = major + minor/10.0f;
      gInf->alignment                 = getMemAlignment();                  // This action will initialise the CUDA context
      gInf->devid                     = device;
      gInf->name                      = (char*)malloc(256*sizeof(char));

      sprintf(gInf->name, "%s", deviceProp.name );

      // TODO: Profile this
      CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1),"Failed to set cache config"); // cudaFuncCachePreferNone OR cudaFuncCachePreferShared OR cudaFuncCachePreferL1 OR cudaFuncCachePreferEqual

      PROF // Profiling  .
      {
	NV_RANGE_POP(); // txt
      }
    }
  }
}

gpuSpecs* getGpuSpec(int devID, int batch, int steps, int opts)
{
  gpuSpecs* gSpec = new gpuSpecs;
  memset(gSpec, 0, sizeof(gpuSpecs));

  if (devID < 0 )
  {
    gSpec->noDevices		= getGPUCount();

    for ( int i = 0; i < gSpec->noDevices; i++)
      gSpec->devId[i]		= i;
  }
  else
  {
    gSpec->noDevices		= 1;
    gSpec->devId[0]		= devID;

    gSpec->noDevBatches[0]	= batch;
    gSpec->noDevSteps[0]	= steps;
    gSpec->noDevOpt[0]		= opts;
  }

//  // Set default
//  for ( int i = 0; i < gSpec->noDevices; i++)
//  {
//    gSpec->noDevBatches[i]	= 0;
//    gSpec->noDevSteps[i]	= 0;
//    gSpec->noDevOpt[i]		= 0;
//  }



  return gSpec;
}

/**  Read the GPU details from clig command line  .
 *
 * @param cmd     clig struct
 * @param bInf    A pointer to the accel info struct to fill
 */
gpuSpecs* readGPUcmd(Cmdline *cmd)
{
  gpuSpecs* gpul = new gpuSpecs;
  memset(gpul, 0, sizeof(gpuSpecs));

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering readGPUcmd.");

  if ( cmd->gpuP ) // Determine the index and number of devices
  {
    if ( cmd->gpuC == 0 )  // NB: Note using gpuC == 0 requires a change in accelsearch_cmd.c every time clig is run!!!! [ usually line 32 should be "  /* gpuC = */ 0," ]
    {
      // Make a list of all devices
      gpul->noDevices   = getGPUCount();
      for ( int dev = 0 ; dev < gpul->noDevices; dev++ )
	gpul->devId[dev] = dev;
    }
    else
    {
      // User specified devices(s)
      gpul->noDevices		= cmd->gpuC;
      for ( int dev = 0 ; dev < gpul->noDevices; dev++ )
	gpul->devId[dev]	= cmd->gpu[dev];
    }
  }

  for ( int dev = 0 ; dev < gpul->noDevices; dev++ ) // Loop over devices  .
  {
    if ( dev >= cmd->nbatchC )
      gpul->noDevBatches[dev]	= cmd->nbatch[cmd->nbatchC-1];
    else
      gpul->noDevBatches[dev]	= cmd->nbatch[dev];

    if ( dev >= cmd->nstepsC )
      gpul->noDevSteps[dev]	= cmd->nsteps[cmd->nbatchC-1];
    else
      gpul->noDevSteps[dev]	= cmd->nsteps[dev];

    if ( dev >= cmd->numoptC )
      gpul->noDevOpt[dev]	= cmd->numopt[cmd->nbatchC-1];
    else
      gpul->noDevOpt[dev]	= cmd->numopt[dev];

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
void readAccelDefalts(confSpecs *conf)
{
  int64_t*  genFlags = &(conf->gen->flags);
  int64_t*  optFlags = &(conf->opt->flags);

  FILE *file;
  char fName[1024];
  sprintf(fName, "%s/lib/GPU_defaults.txt", getenv("PRESTO"));

  file = fopen(fName, "r");
  if ( file )  // Read candidates from previous search  .
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
	singleFlag ( genFlags, str1, str2, FLAG_SEPSRCH, "", "0", lineno, fName );
      }

      else if ( strCom("R_RESOLUTION", str1 ) )
      {
	int no1;
	int read1 = sscanf(line, "%s %i %s", str1, &no1, str2 );
	if ( read1 >= 2 )
	{
	  if ( no1 >= 1 && no1 <= 16 )
	  {
	    conf->gen->noResPerBin = no1;
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
	    conf->gen->zRes = no1;
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
	singleFlag ( genFlags, str1, str2, FLAG_Z_SPLIT, "", "0", lineno, fName );
      }

      else if ( strCom(line, "RESULTS_RING" ) )			// The size of the per batch results ring buffer
      {
	int no1;
	int read1 = sscanf(line, "%s %i %s", str1, &no1, str2 );
	if ( read1 >= 2 )
	{
	  if ( no1 >= 3 && no1 <= 16 )
	  {
	    conf->gen->ringLength = no1;
	  }
	  else
	  {
	    fprintf(stderr,"WARNING: Invalid ring size (%s), it should range between 3 and 16 \n", str1);
	  }
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	}
      }

      else if ( strCom("INTERLEAVE", str1 ) ||  strCom("IL", str1 ) )   // Interleaving
      {
	singleFlag ( genFlags, str1, str2, FLAG_ITLV_ROW, "ROW", "PLN", lineno, fName );
      }

      else if ( strCom("RESPONSE", str1 ) )                     // Response shape
      {
	singleFlag ( genFlags, str1, str2, FLAG_KER_HIGH, "HIGH", "STD", lineno, fName );
      }

      else if ( strCom("FLAG_KER_HIGH", str1 ) )
      {
	singleFlag ( genFlags, str1, str2, FLAG_KER_HIGH, "", "0", lineno, fName );
      }

      else if ( strCom("FLAG_KER_MAX", str1 ) )                 // Kernel
      {
	singleFlag ( genFlags, str1, str2, FLAG_KER_MAX, "", "0", lineno, fName );
      }

      else if ( strCom("CENTER_RESPONSE", str1 ) )
      {
	singleFlag ( genFlags, str1, str2, FLAG_CENTER, "", "off", lineno, fName );
      }

      else if ( strCom("RESPONSE_PRECISION", str1 ) )
      {
	singleFlag ( genFlags, str1, str2, FLAG_KER_DOUBGEN, "DOUBLE", "SINGLE", lineno, fName );
      }

      else if ( strCom("KER_FFT_PRECISION", str1 ) )
      {
	singleFlag ( genFlags, str1, str2, FLAG_KER_DOUBFFT, "DOUBLE", "SINGLE", lineno, fName );
      }

      else if ( strCom("INP_NORM",	str1 ) )
      {
	(*genFlags) &= ~CU_NORM_GPU;	// Clear values

	if      ( strCom("CPU",  str2 ) || strCom(str2, "A" ) )
	{
	  // CPU is no value clear is sufficient
	}
	else if ( strCom("GPU_SM", str2 ) || strCom("GPU", str2 ) )
	{
	  (*genFlags) |= CU_NORM_GPU_SM;
	}
	else if ( strCom("GPU_SM_MIN", str2 ) || strCom("GPU_SM2", str2 ))
	{
	  (*genFlags) |= CU_NORM_GPU_SM_MIN;
	}
	else if ( strCom("GPU_OS", str2 ) )
	{
	  (*genFlags) |= CU_NORM_GPU_OS;
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value \"%s\" for flag \"%s\" on line %i of %s.\n", str2, str1, lineno, fName);
	}
      }

      else if ( strCom("ZBOUND_NORM", str1 ) )
      {
	float no1;
	int read1 = sscanf(line, "%s %f %s", str1, &no1, str2 );
	if ( no1 < 0 )
	{
	  if ( no1 < -1 )
	  {
	    fprintf(stderr,"WARNING: Invalid bound (%.1f) on CPU normalisation, value must be >= 0.  Ignoring value.\n", no1 );
	  }
	}
	else
	{
	  conf->gen->inputNormzBound = no1;
	}
      }

      else if ( strCom("INP_FFT", str1 ) )
      {
	if      ( strCom(str2, "A") )
	{
	  // Default to GPU FFT's - CPU FFT's may be worth doing if z-max is lager than 50 or 100 depends on the CPU and GPU
	  (*genFlags) &= ~CU_INPT_FFT_CPU;
	}
	else if ( singleFlag ( genFlags, str1, str2, CU_INPT_FFT_CPU, "CPU", "GPU", lineno, fName ) )
	{
	  if ( (*genFlags) & CU_NORM_GPU )  // TMP REM - Added to mark an error for thesis timing
	  {
	    printf("Temporary exit - input FFT / NORM \n");
	    exit(EXIT_FAILURE);
	  }

	  // IF we are doing CPU FFT's we need to do CPU normalisation
	  (*genFlags) &= ~CU_NORM_GPU;
	}
      }

      else if ( strCom("ZBOUND_FFT", str1 ) )
      {
	float no1;
	int read1 = sscanf(line, "%s %f %s", str1, &no1, str2 );
	if ( no1 < 0 )
	{
	  if ( no1 < -1 )
	  {
	    fprintf(stderr,"WARNING: Invalid bound (%.1f) on input FFT, value must be >= 0.  Ignoring value.\n", no1 );
	  }
	}
	else
	{
	  conf->gen->inputFFFTzBound = no1;
	}
      }

      else if ( strCom("MUL_KER", str1 ) )
      {
	if      ( strCom("00", str2 ) )
	{
#if defined(WITH_MUL_00) || defined(WITH_MUL_01) || defined(WITH_MUL_02)
	  (*genFlags) &= ~FLAG_MUL_ALL;
	  (*genFlags) |=  FLAG_MUL_00;
#else
	  line[flagLen] = 0;
	  fprintf(stderr, "WARNING: Not compiled with multiplication 2.3 kernel.  (FLAG: %s line %i in %s)\n", line, lineno, fName);

	  FOLD  // TMP REM - Added to mark an error for thesis timing
	  {
	    printf("Temporary exit - mult Kernel \n");
	    exit(EXIT_FAILURE);
	  }
#endif	// WITH_MUL_00 WITH_MUL_01 WITH_MUL_02
	}
	else if ( strCom("11", str2 ) )
	{
#ifdef WITH_MUL_11
	  (*genFlags) &= ~FLAG_MUL_ALL;
	  (*genFlags) |=  FLAG_MUL_11;
#else	// WITH_MUL_11
	  line[flagLen] = 0;
	  fprintf(stderr, "WARNING: Not compiled with multiplication 1.1 kernel.  (FLAG: %s line %i in %s)\n", line, lineno, fName);

	  FOLD  // TMP REM - Added to mark an error for thesis timing
	  {
	    printf("Temporary exit - mult Kernel \n");
	    exit(EXIT_FAILURE);
	  }
#endif	// WITH_MUL_11
	}
	else if ( strCom("21", str2 ) )
	{
#ifdef WITH_MUL_21
	  (*genFlags) &= ~FLAG_MUL_ALL;
	  (*genFlags) |=  FLAG_MUL_21;
#else	// WITH_MUL_21
	  line[flagLen] = 0;
	  fprintf(stderr, "WARNING: Not compiled with multiplication 2.1 kernel.  (FLAG: %s line %i in %s)\n", line, lineno, fName);

	  FOLD  // TMP REM - Added to mark an error for thesis timing
	  {
	    printf("Temporary exit - mult Kernel \n");
	    exit(EXIT_FAILURE);
	  }
#endif	// WITH_MUL_21
	}
	else if ( strCom("22", str2 ) )
	{
#ifdef WITH_MUL_22
	  (*genFlags) &= ~FLAG_MUL_ALL;
	  (*genFlags) |=  FLAG_MUL_22;
#else	// WITH_MUL_22
	  line[flagLen] = 0;
	  fprintf(stderr, "WARNING: Not compiled with multiplication 2.2 kernel.  (FLAG: %s line %i in %s)\n", line, lineno, fName);

	  FOLD  // TMP REM - Added to mark an error for thesis timing
	  {
	    printf("Temporary exit - mult Kernel \n");
	    exit(EXIT_FAILURE);
	  }
#endif	// WITH_MUL_22
	}
	else if ( strCom("23", str2 ) )
	{
#ifdef WITH_MUL_23
	  (*genFlags) &= ~FLAG_MUL_ALL;
	  (*genFlags) |=  FLAG_MUL_23;
#else	// WITH_MUL_23
	  line[flagLen] = 0;
	  fprintf(stderr, "WARNING: Not compiled with multiplication 2.3 kernel.  (FLAG: %s line %i in %s)\n", line, lineno, fName);

	  FOLD  // TMP REM - Added to mark an error for thesis timing
	  {
	    printf("Temporary exit - mult Kernel \n");
	    exit(EXIT_FAILURE);
	  }
#endif	// WITH_MUL_23
	}
	else if ( strCom("31", str2 ) )
	{
#ifdef WITH_MUL_31
	  (*genFlags) &= ~FLAG_MUL_ALL;
	  (*genFlags) |=  FLAG_MUL_31;
#else	// WITH_MUL_31
	  line[flagLen] = 0;
	  fprintf(stderr, "WARNING: Not compiled with multiplication 3.1 kernel.  (FLAG: %s line %i in %s)\n", line, lineno, fName);

	  FOLD  // TMP REM - Added to mark an error for thesis timing
	  {
	    printf("Temporary exit - mult Kernel \n");
	    exit(EXIT_FAILURE);
	  }
#endif	// WITH_MUL_31
	}
	else if ( strCom("CB", str2 ) )
	{
#if CUDA_VERSION >= 6050

#ifdef	WITH_MUL_PRE_CALLBACK
	  (*genFlags) &= ~FLAG_MUL_ALL;
	  (*genFlags) |=  FLAG_MUL_CB;
#else
	  line[flagLen] = 0;
	  fprintf(stderr, "WARNING: Not compiled with multiplication through CUFFT callbacks enabled.  (FLAG: %s line %i in %s)\n", line, lineno, fName);

	  FOLD  // TMP REM - Added to mark an error for thesis timing
	  {
	    printf("Temporary exit - mult Kernel \n");
	    exit(EXIT_FAILURE);
	  }
#endif
#else
	  line[flagLen] = 0;
	  fprintf(stderr, "WARNING: Use of CUDA callbacks requires CUDA 6.5 or greater.  (FLAG: %s line %i in %s)\n", line, lineno, fName);
#endif
	}
	else if ( strCom(str2, "A"  ) )
	{
	  (*genFlags) &= ~FLAG_MUL_ALL;
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value \"%s\" for flag \"%s\" on line %i of %s.\n", str2, str1, lineno, fName);

	  FOLD  // TMP REM - Added to mark an error for thesis timing
	  {
	    printf("Temporary exit - mult Kernel \n");
	    exit(EXIT_FAILURE);
	  }
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
	  conf->gen->mulSlices = 0;
	}
	else
	{
	  int no;
	  int read1 = sscanf(str2, "%i", &no  );
	  if ( read1 == 1 )
	  {
	    conf->gen->mulSlices = no;
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
	  conf->gen->mulChunk = 0;
	}
	else
	{
	  int no;
	  int read1 = sscanf(str2, "%i", &no  );
	  if ( read1 == 1 )
	  {
	    conf->gen->mulChunk = no;
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
	singleFlag ( genFlags, str1, str2, FLAG_CONV, "SEP", "CONT", lineno, fName );
      }

      else if ( strCom("STACK", str1 ) )
      {
	singleFlag ( genFlags, str1, str2, FLAG_STK_UP, "UP", "DN", lineno, fName );
      }

      else if ( strCom("CUFFT_PLAN_INP", str1 ) )
      {
	singleFlag ( genFlags, str1, str2, CU_FFT_SEP_INP, "SEPARATE", "SINGLE", lineno, fName );
      }

      else if ( strCom("CUFFT_PLAN_PLN", str1 ) )
      {
	singleFlag ( genFlags, str1, str2, CU_FFT_SEP_PLN, "SEPARATE", "SINGLE", lineno, fName );
      }

      else if ( strCom("STD_POWERS", str1 ) )
      {
	if      ( strCom("CB", str2 ) )
	{
#if CUDA_VERSION >= 6050
	  (*genFlags) |=     FLAG_CUFFT_CB_POW;
#else
	  line[flagLen] = 0;
	  fprintf(stderr,"WARNING: Use of CUDA callbacks requires CUDA 6.5 or greater.  (FLAG: %s line %i in %s)\n", line, lineno, fName);
#endif
	}
	else if ( strCom("SS", str2 ) )
	{
	  (*genFlags) &= ~FLAG_CUFFT_CB_POW;
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
	  (*genFlags) |=     FLAG_CUFFT_CB_INMEM;
#else
	  line[flagLen] = 0;
	  fprintf(stderr,"WARNING: Use of CUDA callbacks requires CUDA 6.5 or greater.  (FLAG: %s %s line %i in %s)\n", str1, str2, lineno, fName);
#endif
	}
	else if ( strCom("MEM_CPY", str2 ) || strCom("", str2 ))
	{
#if CUDA_VERSION >= 6050
	  (*genFlags) &=    ~FLAG_CUFFT_CB_INMEM;
	  (*genFlags) |=     FLAG_CUFFT_CB_POW;
#else
	  line[flagLen] = 0;
	  fprintf(stderr,"WARNING: Use of CUDA callbacks requires CUDA 6.5 or greater.  (FLAG: %s %s line %i in %s)\n", str1, str2, lineno, fName);
#endif
	}
	else if ( strCom("KERNEL", str2 ) )
	{
	  (*genFlags) &=    ~FLAG_CUFFT_CB_INMEM;
	  (*genFlags) &=    ~FLAG_CUFFT_CB_POW;
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
	(*genFlags) &= ~FLAG_CUFFT_ALL;
      }

      else if ( strCom("POWER_PRECISION", str1 ) )
      {
	if      ( strCom("HALF",   str2 ) )
	{
#if CUDA_VERSION >= 7050
	  (*genFlags) |=  FLAG_POW_HALF;
#else
	  (*genFlags) &= ~FLAG_POW_HALF;

	  line[flagLen] = 0;
	  fprintf(stderr,"WARNING: Half precision can only be used with CUDA 7.5 or later! Reverting to single precision. (FLAG: %s line %i in %s)\n", line, lineno, fName);
#endif
	}
	else if ( strCom("SINGLE", str2 ) )
	{
	  (*genFlags) &= ~FLAG_POW_HALF;
	}
	else if ( strCom("DOUBLE", str2 ) )
	{
	  fprintf(stderr,"ERROR: Cannot sore in-mem plane as double! Defaulting to float.\n");
	  (*genFlags) &= ~FLAG_POW_HALF;
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
	  (*genFlags) &= ~FLAG_SS_ALL;
	  (*genFlags) |= FLAG_SS_00;
	  (*genFlags) |= FLAG_STAGES;
	}
	else if ( strCom("CPU", str2 ) )
	{
	  fprintf(stderr, "ERROR: CPU Sum and search is no longer supported.\n\n");
	  continue;

	  (*genFlags) &= ~FLAG_SS_ALL;
	  (*genFlags) |= FLAG_SS_CPU;

	  conf->gen->retType &= ~CU_SRT_ALL   ;
	  conf->gen->retType |= CU_STR_PLN    ;

	  if ( (*genFlags) & FLAG_CUFFT_CB_POW )
	  {
	    conf->gen->retType &= ~CU_TYPE_ALLL   ;
	    conf->gen->retType |= CU_FLOAT        ;
	  }
	  else
	  {
	    conf->gen->retType &= ~CU_TYPE_ALLL   ;
	    conf->gen->retType |= CU_CMPLXF       ;
	  }
	}
	else if ( strCom("10",  str2 ) || strCom("31",  str2 ) )
	{
	  (*genFlags) &= ~FLAG_SS_ALL;
	  (*genFlags) |= FLAG_SS_31;
	  (*genFlags) |= FLAG_STAGES;
	}
	else if ( strCom("INMEM", str2 ) || strCom("IM", str2 ) )
	{
	  (*genFlags) |= FLAG_SS_INMEM;
	}
	else if ( strCom(str2, "A"  ) )
	{
	  (*genFlags) &= ~FLAG_SS_ALL;
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value \"%s\" for flag \"%s\" on line %i of %s.\n", str2, str1, lineno, fName);
	}
      }

      else if ( strCom("SS_COUNT", str1 ) )
      {
	singleFlag ( genFlags, str1, str2, FLAG_SS_COUNT, "", "0", lineno, fName );
#ifndef WITH_SAS_COUNT
	fprintf(stderr,"WARNING: Not compiled with Sum & search counting enabled. Config on line %i in %s has no effect.\n", lineno, fName );
#endif
      }

      else if ( strCom("SS_SLICES", str1 ) )
      {
	if ( strCom(str2, "A"   ) )
	{
	  conf->gen->ssSlices = 0;
	}
	else
	{
	  int no;
	  int read1 = sscanf(str2, "%i", &no  );
	  if ( read1 == 1 )
	  {
	    conf->gen->ssSlices = no;
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
	  conf->gen->ssChunk = 0;
	}
	else
	{
	  int no;
	  int read1 = sscanf(str2, "%i", &no  );
	  if ( read1 == 1 )
	  {
	    if ( no <= 0 )		// Auto
	    {
	      conf->gen->ssChunk = 0;
	    }
	    else if ( (no >= MIN_SAS_CHUNK) and (no <= MAX_SAS_CHUNK) )
	    {
	      conf->gen->ssChunk = no;
	    }
	    else
	    {
	      fprintf(stderr, "WARNING: Sum & search chunk size not in compiled bounds (%i - %i). Line %i of %s.\n", MIN_SAS_CHUNK, MAX_SAS_CHUNK, lineno, fName);
	      conf->gen->ssChunk = 0;

	      FOLD  // TMP REM - Added to mark an error for thesis timing
	      {
		printf("Temporary exit - ssChunk \n");
		exit(EXIT_FAILURE);
	      }
	    }
	  }
	  else
	  {
	    line[flagLen] = 0;
	    fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	  }
	}
      }

      else if ( strCom("SS_COLUMN", str1 ) )
      {
	if ( strCom(str2, "A"   ) )
	{
	  conf->gen->ssColumn = 0;
	}
	else
	{
	  int no;
	  int read1 = sscanf(str2, "%i", &no  );
	  if ( read1 == 1 )
	  {
	    if ( no <= 0 )		// Auto
	    {
	      conf->gen->ssColumn = 0;
	    }
	    else if ( (no >= MIN_SAS_COLUMN) and (no <= MAX_SAS_COLUMN) )
	    {
	      conf->gen->ssColumn = no;
	    }
	    else
	    {
	      fprintf(stderr, "WARNING: Sum & search column size not in compiled bounds (%i - %i). Line %i of %s.\n", MIN_SAS_COLUMN, MAX_SAS_COLUMN, lineno, fName);
	      conf->gen->ssColumn = 0;

	      FOLD  // TMP REM - Added to mark an error for thesis timing
	      {
		printf("Temporary exit - ssColumn \n");
		exit(EXIT_FAILURE);
	      }
	    }
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
	  conf->gen->ssStepSize = 0;
	}
	else
	{
	  int no;
	  int read1 = sscanf(str2, "%i", &no  );
	  if ( read1 == 1 )
	  {
	    conf->gen->ssStepSize = no;
	  }
	  else
	  {
	    line[flagLen] = 0;
	    fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	  }
	}
      }

      else if ( strCom("CAND_PROCESS", str1 ) )
      {
	singleFlag ( genFlags, str1, str2, FLAG_CAND_THREAD, "THREAD", "SEQ", lineno, fName );
      }

      else if ( strCom("CAND_MEM", str1 ) )
      {
	singleFlag ( genFlags, str1, str2, FLAG_CAND_MEM_PRE, "PRE", "RING", lineno, fName );
      }

      else if ( strCom("CAND_STORAGE", str1 ) )
      {
	if      ( strCom("ARR", str2 ) || strCom("", str2 ) )
	{
	  // Return type
	  conf->gen->retType &= ~CU_TYPE_ALLL ;
	  conf->gen->retType &= ~CU_SRT_ALL   ;

	  conf->gen->retType |= CU_POWERZ_S   ;
	  conf->gen->retType |= CU_STR_ARR    ;

	  // Candidate type
	  conf->gen->cndType &= ~CU_TYPE_ALLL ;
	  conf->gen->cndType &= ~CU_SRT_ALL   ;

	  conf->gen->cndType |= CU_CANDFULL   ;
	  conf->gen->cndType |= CU_STR_ARR    ;
	}
	else if ( strCom("LST", str2 ) )
	{
	  // Return type
	  conf->gen->retType &= ~CU_TYPE_ALLL ;
	  conf->gen->retType &= ~CU_SRT_ALL   ;

	  conf->gen->retType |= CU_POWERZ_S   ;
	  conf->gen->retType |= CU_STR_ARR    ;

	  // Candidate type
	  conf->gen->cndType &= ~CU_TYPE_ALLL ;
	  conf->gen->cndType &= ~CU_SRT_ALL   ;

	  conf->gen->cndType |= CU_CANDFULL   ;
	  conf->gen->cndType |= CU_STR_LST    ;
	}
	else if ( strCom("QUAD", str2 ) )
	{
	  fprintf(stderr, "ERROR: Quadtree storage not yet implemented. Doing nothing!\n");
	  continue;

	  // Candidate type
	  conf->gen->cndType &= ~CU_TYPE_ALLL ;
	  conf->gen->cndType &= ~CU_SRT_ALL   ;

	  conf->gen->cndType |= CU_POWERZ_S   ;
	  conf->gen->cndType |= CU_STR_QUAD   ;
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

      else if ( strCom("ARR_RES", str1 ) )
      {
	float no1;
	int read1 = sscanf(line, "%s %f %s", str1, &no1, str2 );
	if ( read1 >= 2 )
	{
	  if ( no1 >= 0.1 && no1 <= 1.0 )
	  {
	    conf->gen->candRRes = no1;
	  }
	  else
	  {
	    fprintf(stderr,"WARNING: Invalid candidate array resolution, it should range between 0.1 and 1 \n");
	  }
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	}
      }

      else if ( strCom("RETURN", str1 ) )
      {
	singleFlag ( genFlags, str1, str2, FLAG_STAGES, "STAGES", "FINAL", lineno, fName );
      }

      else if ( strCom("FLAG_RET_ARR", str1 ) )
      {
	conf->gen->retType &= ~CU_SRT_ALL   ;
	conf->gen->retType |= CU_STR_ARR    ;
      }
      else if ( strCom("FLAG_RET_PLN", str1 ) )
      {
	conf->gen->retType &= ~CU_SRT_ALL   ;
	conf->gen->retType |= CU_STR_PLN    ;
      }

      else if ( strCom("FLAG_STORE_ALL", str1 ) )
      {
	singleFlag ( genFlags, str1, str2, FLAG_STORE_ALL, "", "0", lineno, fName );
      }

      //////////////  OPTEMISATION  \\\\\\\\\\\\\\\\\\\\\\\\

      else if ( strCom("OPT_METHOUD", str1 ) )
      {
	if      ( strCom("PLANE", str2 ) )
	{
	  (*optFlags) &= ~FLAG_OPT_ALL;
	}
	else if ( strCom("SWARM", str2 ) )
	{
	  (*optFlags) &= ~FLAG_OPT_ALL;
	  (*optFlags) |= FLAG_OPT_SWARM;
	}
	else if ( strCom("NM", str2 ) )
	{
	  (*optFlags) &= ~FLAG_OPT_ALL;
	  (*optFlags) |= FLAG_OPT_NM;
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
	    conf->opt->zScale = no1;
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
	    conf->opt->optResolution = no1;
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
	singleFlag ( optFlags, str1, str2, FLAG_OPT_NRM_LOCAVE, "LOCAVE", "MEDIAN", lineno, fName );

	if      ( strCom("NONE", str2 ) )
	{
	  (*optFlags) &= ~FLAG_OPT_NRM_ALL;
	}
	else if ( strCom("LOCAVE", str2 ) )
	{
	  (*optFlags) &= ~FLAG_OPT_NRM_ALL;
	  (*optFlags) |= FLAG_OPT_NRM_LOCAVE;
	}
	else if ( strCom("MEDIAN1D", str2 ) )
	{
	  (*optFlags) &= ~FLAG_OPT_NRM_ALL;
	  (*optFlags) |= FLAG_OPT_NRM_MEDIAN1D;
	}
	else if ( strCom("MEDIAN2D", str2 ) )
	{
	  (*optFlags) &= ~FLAG_OPT_NRM_ALL;
	  (*optFlags) |= FLAG_OPT_NRM_MEDIAN2D;
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value \"%s\" for flag \"%s\" on line %i of %s.\n", str2, str1, lineno, fName);
	}
      }

      else if ( strCom("FLAG_OPT_BEST", str1 ) )
      {
	singleFlag ( optFlags, str1, str2, FLAG_OPT_BEST, "", "0", lineno, fName );
      }

      else if ( strCom("OPT_MIN_LOC_HARMS", str1 ) )
      {
	int no;
	int read1 = sscanf(str2, "%i", &no  );
	if ( read1 == 1 )
	{
	  if ( no >= 1 && no <= OPT_MAX_LOC_HARMS )
	  {
	    conf->opt->optMinLocHarms = no;
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
	  conf->opt->optMinRepHarms = no;
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
	  conf->opt->optPlnScale = no;
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", str1, lineno, fName);
	}
      }

      else if ( strCom("FLAG_OPT_DYN_HW", str1 ) )
      {
	singleFlag ( optFlags, str1, str2, FLAG_OPT_DYN_HW, "", "0", lineno, fName );
      }

      else if ( strCom("OPT_NELDER_MEAD_REFINE", str1 ) )
      {
	singleFlag ( optFlags, str1, str2, FLAG_OPT_NM_REFINE, "", "0", lineno, fName );
      }

      else if ( strCom("OPT_PROCESS", str1 ) )
      {
	singleFlag ( optFlags, str1, str2, FLAG_OPT_THREAD, "THREAD", "SEQ", lineno, fName );
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
	    conf->opt->optPlnSiz[0] = no2;
	  }
	  else if ( no1 == 2 )
	  {
	    conf->opt->optPlnSiz[1] = no2;
	  }
	  else if ( no1 == 4 )
	  {
	    conf->opt->optPlnSiz[2] = no2;
	  }
	  else if ( no1 == 8 )
	  {
	    conf->opt->optPlnSiz[3] = no2;
	  }
	  else if ( no1 == 16 )
	  {
	    conf->opt->optPlnSiz[4] = no2;
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
	    conf->opt->optPlnDim[no1-1] = no2;
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
	singleFlag ( optFlags, str1, str2, FLAG_OPT_DYN_HW, "", "0", lineno, fName );
      }

      else if ( strCom("FLAG_DBG_SYNCH", str1 ) )
      {
	singleFlag ( genFlags, str1, str2, FLAG_SYNCH, "", "0", lineno, fName );
	singleFlag ( optFlags, str1, str2, FLAG_SYNCH, "", "0", lineno, fName );
      }

      else if ( strCom("FLAG_DBG_PROFILING", str1 ) )
      {
#ifdef PROFILING
	singleFlag ( genFlags, str1, str2, FLAG_PROF, "", "0", lineno, fName );
	singleFlag ( optFlags, str1, str2, FLAG_PROF, "", "0", lineno, fName );
#else
	fprintf(stderr, "ERROR: Found %s on line %i of %s, the program has not been compile with profiling enabled. Check the #define in cuda_accel.h.\n", str1, lineno, fName);
	exit(EXIT_FAILURE); // TMP REM - Added to mark an error for thesis timing
#endif
      }

      else if ( strCom("FLAG_DPG_PLT_OPT", str1 ) )
      {
	singleFlag ( optFlags, str1, str2, FLAG_DPG_PLT_OPT, "", "0", lineno, fName );
      }

      else if ( strCom("FLAG_DPG_PLT_POWERS", str1 ) )
      {
	singleFlag ( genFlags, str1, str2, FLAG_DPG_PLT_POWERS, "", "0", lineno, fName );
	singleFlag ( optFlags, str1, str2, FLAG_DPG_PLT_POWERS, "", "0", lineno, fName );
      }

      else if ( strCom("FLAG_DPG_UNOPT", str1 ) )
      {
	useUnopt    = 1;
      }

      else if ( strCom("FLAG_DBG_SKIP_OPT", str1 ) )
      {
	singleFlag ( genFlags, str1, str2, FLAG_DPG_SKP_OPT, "", "0", lineno, fName );
	singleFlag ( optFlags, str1, str2, FLAG_DPG_SKP_OPT, "", "0", lineno, fName );
      }

      else if ( strCom("FLAG_DPG_PRNT_CAND", str1 ) )
      {
	singleFlag ( genFlags, str1, str2, FLAG_DPG_PRNT_CAND, "", "0", lineno, fName );
	singleFlag ( optFlags, str1, str2, FLAG_DPG_PRNT_CAND, "", "0", lineno, fName );
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

      else if ( strCom("FLAG_DBG_TEST", str1 ) )
      {
	if      ( strCom(str2, "0") )
	{
	  (*genFlags) &= ~FLAG_DBG_TEST_ALL;
	}
	else if ( strCom(str2, "1") )
	{
	  (*genFlags) |= FLAG_DBG_TEST_1;
	}
	else if ( strCom(str2, "2") )
	{
	  (*genFlags) |= FLAG_DBG_TEST_2;
	}
	else if ( strCom(str2, "3") )
	{
	  (*genFlags) |= FLAG_DBG_TEST_3;
	}
	else
	{
	  fprintf(stderr, "ERROR: Found unknown value \"%s\" for flag \"%s\" on line %i of %s.\n", str2, str1, lineno, fName);
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
	exit(EXIT_FAILURE); // TMP REM - Added to mark an error for thesis timing
      }
    }

    fclose (file);
  }
  else
  {
    printf("Unable to read GPU accel settings from %s\n", fName);
    exit(EXIT_FAILURE); // TMP REM - Added to mark an error for thesis timing
  }
}

searchSpecs* sSpecsFromObs(Cmdline *cmd, accelobs* obs, confSpecs* conf)
{
  searchSpecs* sSpec = new(searchSpecs);
  memset(sSpec, 0, sizeof(searchSpecs));

  sSpec->specRLow	= obs->rlo;
  sSpec->specRHigh	= obs->rhi;

  sSpec->searchRLow	= obs->rlo;
  sSpec->searchRHigh	= obs->rhi;

  sSpec->noHarmStages	= log2((float)cmd->numharm)+1;
  sSpec->noHarms	= cmd->numharm;
  sSpec->zMax		= cmd->zmax;
  sSpec->sigma		= cmd->sigma;

  conf->gen->planeWidth	= cmd->width;
  conf->gen->normType	= obs->norm_type;

  conf->gen->zMax	= cu_calc_required_z<double>(1, fabs(sSpec->zMax), conf->gen->zRes);

//  REM
//  if ( conf->gen->flags & (FLAG_SS_31 /*| FLAG_SS_20 | FLAG_SS_30 */ ) )
//  {
//    // Round the first bin to a multiple of the number of harmonics this is needed in the s&s kernel
//    sSpec->searchRLow	= floor(sSpec->searchRLow/(float)cmd->numharm)*cmd->numharm;
//  }

  return sSpec;
}

fftInfo* fftFromObs(accelobs* obs)
{
  fftInfo* fftInf = new(fftInfo);
  memset(fftInf, 0, sizeof(fftInfo));

  fftInf->data		= obs->fft;				// Pointer to first memory location of the FT values
  fftInf->T		= obs->T;				// Observation duration
  fftInf->dt		= obs->dt;				// Sampling frequency

  fftInf->firstBin	= 0;					// By default the start of the FT in memory
  fftInf->lastBin	= obs->numbins-1;			// The last bin in memory
  fftInf->noBins	= fftInf->lastBin - fftInf->firstBin + 1;// The number of bins read into memory

  return fftInf;
}

confSpecs* defaultConfig()
{
  confSpecs* conf = new(confSpecs);
  memset(conf, 0, sizeof(confSpecs));

  conf->gen = new(confSpecsGen);
  memset(conf->gen, 0, sizeof(confSpecsGen));

  conf->opt = new(confSpecsOpt);
  memset(conf->opt, 0, sizeof(confSpecsOpt));

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering readSrchSpecs.");

  FOLD // Defaults for accel search  .
  {
    conf->gen->flags	|= FLAG_KER_DOUBGEN ;	// Generate the kernels using double precision math (still stored as floats though)
    conf->gen->flags	|= FLAG_ITLV_ROW    ;
    conf->gen->flags	|= FLAG_CENTER      ;	// Centre and align the usable part of the planes
    conf->gen->flags	|= CU_FFT_SEP_INP   ;	// Input is small and separate FFT plans wont take up too much memory

#ifdef WITH_SAS_COUNT
    conf->gen->flags	|= FLAG_SS_COUNT    ;	// Enable counting results in sum & search kernels
#endif

    // NOTE: I found using the strait ring buffer memory is fastest - If the data is very noisy consider using FLAG_CAND_MEM_PRE
#ifndef DEBUG
    conf->gen->flags	|= FLAG_CAND_THREAD ;	// Multithreading really slows down debug so only turn it on by default for release mode, NOTE: This can be over ridden in the defaults file
    conf->opt->flags	|= FLAG_OPT_THREAD  ;	// Do CPU component of optimisation in a separate thread - A very good idea
#endif

#if CUDA_VERSION >= 6050
    conf->gen->flags	|= FLAG_CUFFT_CB_POW;	// CUFFT callback to calculate powers, very efficient so on by default
#endif

#if CUDA_VERSION >= 7050 && defined(WITH_HALF_RESCISION_POWERS)
    conf->gen->flags	|= FLAG_POW_HALF;
#endif

//    if ( obs->inmem )				// Use the command line to select in-mem search, NOTE: this is over ridden by what ever is in the DEFAULTS file (best to comment out this line then!)
//    {
//      sSpec.flags	|= FLAG_SS_INMEM;
//    }

    conf->gen->flags	|= FLAG_STAGES;

    conf->gen->cndType	|= CU_CANDFULL;  	// Candidate data type - CU_CANDFULL this should be the default as it has all the needed data
    conf->gen->cndType	|= CU_STR_ARR;  	// Candidate storage structure - CU_STR_ARR    is generally the fastest

    conf->gen->retType	|= CU_POWERZ_S;  	// Return type
    conf->gen->retType	|= CU_STR_ARR;  	// Candidate storage structure

    conf->gen->noResPerBin	= 2;		// Inter binning
    conf->gen->candRRes		= 0.5;		// 1 Candidate per 2 bins
    conf->gen->zRes		= 2;
    conf->gen->zMax		= 200;
    conf->gen->ringLength	= 7;		// Just a good number
    conf->gen->planeWidth	= 8;		// A good default for newer GPU's

    conf->gen->normType		= 0;
    conf->gen->inputNormzBound	= -1;		// Default to not uses, only used if specified in the defaults file
    conf->gen->inputFFFTzBound	= -1;		// Default to not uses, only used if specified in the defaults file

    conf->gen->ssStepSize	= 32768;	// TODO: Check this, to small may be inefficient too large can make the IM plane to large

    // Default: Auto chose best!
    conf->gen->mulSlices	= 0 ;
    conf->gen->mulChunk		= 0 ;
    conf->gen->ssSlices		= 0 ;
    conf->gen->ssChunk		= 0 ;

    conf->opt->zScale		= 4;
    conf->opt->optResolution	= 16;
    conf->opt->optPlnScale	= 10;
    conf->opt->optMinLocHarms	= 1;
    conf->opt->optMinRepHarms	= 1;


    conf->opt->flags		|= FLAG_OPT_NRM_MEDIAN1D;
    conf->opt->flags		|= FLAG_OPT_BLK_HRM;
    conf->opt->flags		|= FLAG_OPT_PTS_HRM;
    conf->opt->flags		|= FLAG_OPT_NM_REFINE;

    conf->opt->optPlnDim[0]	= 128;
    conf->opt->optPlnDim[1]	= 32;
    conf->opt->optPlnDim[2]	= 16;
    conf->opt->optPlnDim[3]	= 16;
    conf->opt->optPlnDim[4]	= 0;
    conf->opt->optPlnDim[5]	= 0;
    conf->opt->optPlnDim[6]	= 0;

    conf->opt->optPlnSiz[0]	= 16;
    conf->opt->optPlnSiz[1]	= 14;
    conf->opt->optPlnSiz[2]	= 12;
    conf->opt->optPlnSiz[3]	= 10;
    conf->opt->optPlnSiz[4]	= 8;
  }

  return conf;
}

confSpecs* getConfig()
{
  confSpecs* conf = defaultConfig();

  // Now read the
  readAccelDefalts(conf);

  return conf;
}

searchSpecs* getSpec(fftInfo* fft)
{
  searchSpecs* sSpec = new searchSpecs;
  memset(sSpec, 0, sizeof(searchSpecs));

  sSpec->noHarmStages	= 5;
  sSpec->noHarms 	= 16;
  sSpec->sigma		= 2;
  sSpec->zMax		= 200;

  if ( fft )
  {
    if ( fft->N && fft->dt )
    {
      sSpec->specRLow		= 1*fft->N*fft->dt;
      sSpec->specRHigh		= 10000.0*fft->N*fft->dt;
    }
    else
    {
      sSpec->specRLow		= fft->firstBin;
      sSpec->specRHigh		= fft->lastBin;
    }

    sSpec->searchRLow	= sSpec->specRLow;
    sSpec->searchRHigh	= sSpec->specRHigh;
  }

  return sSpec;
}

void intSrchThrd(cuSearch* srch)
{
  resThrds* tInf = srch->threasdInfo;

  if ( !tInf )
  {
    tInf     = new(resThrds);
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

searchSpecs* duplicate(searchSpecs* sSpec)
{
  searchSpecs* dup = new searchSpecs;
  memcpy(dup,sSpec,sizeof(searchSpecs));
  return dup;
}

confSpecsGen* duplicate(confSpecsGen* conf)
{
  confSpecsGen* dup = new confSpecsGen;
  memcpy(dup, conf, sizeof(confSpecsGen));
  return dup;
}

confSpecsOpt* duplicate(confSpecsOpt* conf)
{
  confSpecsOpt* dup = new confSpecsOpt;
  memcpy(dup, conf, sizeof(confSpecsOpt));
  return dup;
}

confSpecs* duplicate(confSpecs* conf)
{
  confSpecs* dup = new confSpecs;
  dup->gen = duplicate(conf->gen);
  dup->opt = duplicate(conf->opt);
  return dup;
}

gpuSpecs* duplicate(gpuSpecs* gSpec)
{
  gpuSpecs* dup = new gpuSpecs;
  memcpy(dup, gSpec, sizeof(gpuSpecs));
  return dup;
}

bool compare(searchSpecs* sSpec1, searchSpecs* sSpec2)
{
  if ( sSpec1 == sSpec2 )
    return true;

  if ( sSpec1->noHarmStages != sSpec2->noHarmStages )
    return false;

  if ( sSpec1->noHarms != sSpec2->noHarms )
    return false;

  if ( sSpec1->searchRHigh != sSpec2->searchRHigh )
    return false;

  if ( sSpec1->searchRLow != sSpec2->searchRLow )
    return false;

  if ( sSpec1->sigma != sSpec2->sigma )
    return false;

  if ( sSpec1->zMax != sSpec2->zMax )
    return false;

  return true;
}

bool compare(confSpecsGen* conf1, confSpecsGen* conf2)
{
  if ( conf1 == conf2 )
    return true;

  if (conf1->candRRes != conf2->candRRes )
    return false;

  if (conf1->inputFFFTzBound != conf2->inputFFFTzBound )
    return false;

  if (conf1->inputNormzBound != conf2->inputNormzBound )
    return false;

  if (conf1->noResPerBin != conf2->noResPerBin )
    return false;

  if (conf1->planeWidth != conf2->planeWidth )
    return false;

  if (conf1->ssStepSize != conf2->ssStepSize )
    return false;

  if (conf1->zMax != conf2->zMax )
    return false;

  if (conf1->zRes != conf2->zRes )
    return false;

  if ( conf1->flags != conf2->flags )
  {
    if ( (conf1->flags & (FLAG_Z_SPLIT) ) != ( conf2->flags & (FLAG_Z_SPLIT) ) )
      return false;

    if ( (conf1->flags & (CU_NORM_GPU) ) != ( conf2->flags & (CU_NORM_GPU) ) )
      return false;

    int64_t mul1 = conf1->flags & (FLAG_MUL_ALL);
    int64_t mul2 = conf2->flags & (FLAG_MUL_ALL);

    if ( (conf1->flags & (FLAG_MUL_CB) ) != ( conf2->flags & (FLAG_MUL_CB) ) )
      return false;

    if ( (conf1->flags & (CU_FFT_SEP_ALL) ) != ( conf2->flags & (CU_FFT_SEP_ALL) ) )
      return false;

    if ( (conf1->flags & (FLAG_CUFFT_ALL) ) != ( conf2->flags & (FLAG_CUFFT_ALL) ) )
      return false;

    if ( (conf1->flags & (FLAG_POW_HALF) ) != ( conf2->flags & (FLAG_POW_HALF) ) )
      return false;

    if ( (conf1->flags & (FLAG_SS_ALL) ) != ( conf2->flags & (FLAG_SS_ALL) ) )
      return false;

    if ( (conf1->flags & (FLAG_SS_ALL) ) != ( conf2->flags & (FLAG_SS_ALL) ) )
      return false;
  }

  return true;
}

bool compare(confSpecsOpt* conf1, confSpecsOpt* conf2)
{
  if ( conf1 == conf2 )
    return true;

  if (conf1->optMinLocHarms != conf2->optMinLocHarms )
    return false;

  if (conf1->optMinRepHarms != conf2->optMinRepHarms )
    return false;

  if (conf1->optPlnScale != conf2->optPlnScale )
    return false;

  if (conf1->optResolution != conf2->optResolution )
    return false;

  if (conf1->zScale != conf2->zScale )
    return false;

  return true;
}

bool compare(fftInfo* fft1, fftInfo* fft2)
{
  if ( fft1 == fft2 )
    return true;

  if (fft1->N != fft2->N )
    return false;

  if (fft1->T != fft2->T )
    return false;

  if (fft1->dt != fft2->dt )
    return false;

  if (fft1->firstBin != fft2->firstBin )
    return false;

  if (fft1->lastBin != fft2->lastBin )
    return false;

  return true;
}

bool compare(gpuSpecs* gSpec1, gpuSpecs* gSpec2)
{
  if ( gSpec1 == gSpec2 )
    return true;

  if (gSpec1->noDevices != gSpec1->noDevices )
    return false;

  for ( int devNo = 0; devNo < gSpec1->noDevices; devNo++ )
  {
    if ( gSpec1->devId[devNo] != gSpec1->devId[devNo] )
      return false;

    if ( gSpec1->noDevBatches[devNo] != gSpec1->noDevBatches[devNo] )
      return false;

    if ( gSpec1->noDevOpt[devNo] != gSpec1->noDevOpt[devNo] )
      return false;

    if ( gSpec1->noDevSteps[devNo] != gSpec1->noDevSteps[devNo] )
      return false;
  }

  return true;
}

bool compare(cuSearch* search, searchSpecs* sSpec, confSpecs* conf, gpuSpecs* gSpec, fftInfo* fftInf)
{
  CUDA_SAFE_CALL(cudaGetLastError(), "Entering initCuSearch.");

  if (!search)
    return false;

  if ( search->noGenHarms != sSpec->noHarms )
    return false;

  if ( search->noHarmStages != sSpec->noHarmStages )
    return false;

  if ( !compare(search->sSpec, sSpec) )
    return false;

  if ( !compare(search->conf->gen, conf->gen) )
    return false;

  if ( !compare(search->conf->opt, conf->opt) )
    return false;

  return true;
}

cuSearch* initSearchInf(searchSpecs* sSpec, confSpecs* conf, gpuSpecs* gSpec, fftInfo* fftInf)
{
  cuSearch* srch = new(cuSearch);
  memset(srch, 0, sizeof(cuSearch));

  srch->noHarmStages		= sSpec->noHarmStages;
  srch->noGenHarms		= ( 1<<(srch->noHarmStages-1) );
  srch->noSrchHarms		= ( 1<<(srch->noHarmStages-1) );

  srch->sIdx			= (int*)malloc(srch->noGenHarms * sizeof(int));
  srch->powerCut		= (float*)malloc(srch->noHarmStages * sizeof(float));
  srch->numindep		= (long long*)malloc(srch->noHarmStages * sizeof(long long));

  srch->sSpec			= sSpec;
  srch->conf			= conf;
  srch->gSpec			= gSpec;
  srch->fft			= fftInf;

  FOLD // Calculate power cutoff and number of independent values  .
  {
    infoMSG(3,2,"Calculate power cutoff and number of independent values\n");

    // Calculate appropriate z-max
    int numz = round(conf->gen->zMax / conf->gen->zRes) * 2 + 1;
    float adjust = 0;

    FOLD // Calculate power cutoff and number of independent values  .
    {
      for (int ii = 0; ii < srch->noHarmStages; ii++)
      {
      if ( numz == 1 )
      {
	srch->numindep[ii]	= (sSpec->searchRHigh - sSpec->searchRLow) / (double)(1<<ii) ;
      }
      else
      {
	srch->numindep[ii]	= (sSpec->searchRHigh - sSpec->searchRLow) * (numz + 1) * ( conf->gen->zRes / 6.95 ) / (double)(1<<ii);
      }

      // Power cutoff
      srch->powerCut[ii]	= power_for_sigma(sSpec->sigma, (1<<ii), srch->numindep[ii]);


      FOLD // Adjust for some lack in precision, if using half precision
      {
	if ( conf->gen->flags & FLAG_POW_HALF )
	{
	  float noP = log10( srch->powerCut[ii] );
	  float dp = pow(10, floor(noP)-4 );  		// "Last" significant value

	  adjust = -dp*(1<<ii);				// Subtract one significant "value" for each harmonic
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

cuSearch* initSearchInfCMD(Cmdline *cmd, accelobs* obs, gpuSpecs* gSpec)
{
  confSpecs*	conf	= getConfig();
  fftInfo*	fft	= fftFromObs(obs);
  searchSpecs*	sSpec	= sSpecsFromObs(cmd, obs, conf);

  return initSearchInf(sSpec, conf, gSpec, fft);
}

int remOptFlag(cuOptCand* pln, int64_t flag)
{
  if ( pln )
    pln->flags &= ~flag;
  else
  {
    fprintf(stderr, "ERROR: Null pointer");
    return ACC_ERR_NULL;
  }
  return ACC_ERR_NONE;
}

int setOptFlag(cuOptCand* pln, int64_t flag)
{
  if ( pln )
    pln->flags |=  flag;
  else
  {
    fprintf(stderr, "ERROR: Null pointer");
    return ACC_ERR_NULL;
  }
  return ACC_ERR_NONE;
}

int remOptFlag(cuOptInfo* oInf, int64_t flag)
{
  if ( !oInf )
  {
    fprintf(stderr, "ERROR: Null pointer");
    return ACC_ERR_NULL;
  }
  if ( !oInf->opts )
  {
    fprintf(stderr, "ERROR: Null pointer");
    return ACC_ERR_NULL;
  }

  int ret = ACC_ERR_NONE;
  for ( int i =0; i < oInf->noOpts; i++ )
  {
    ret |= remOptFlag(&oInf->opts[i], flag);
  }
  return ret;
}

int setOptFlag(cuOptInfo* oInf, int64_t flag)
{
  if ( !oInf )
  {
    fprintf(stderr, "ERROR: Null pointer");
    return ACC_ERR_NULL;
  }
  if ( !oInf->opts )
  {
    fprintf(stderr, "ERROR: Null pointer");
    return ACC_ERR_NULL;
  }

  int ret = ACC_ERR_NONE;
  for ( int i =0; i < oInf->noOpts; i++ )
  {
    ret |= setOptFlag(&oInf->opts[i], flag);
  }
  return ret;
}

int setOptFlag(cuSearch* cuSrch, int64_t flag)
{
  int ret = ACC_ERR_NONE;
  if ( cuSrch )
    ret |= setOptFlag(cuSrch->oInf, flag);
  else
  {
    fprintf(stderr, "ERROR: Null pointer");
    ret |= ACC_ERR_NULL;
  }
  return ret;
}

int remOptFlag(cuSearch* cuSrch, int64_t flag)
{
  int ret = ACC_ERR_NONE;
  if ( cuSrch )
    ret |= remOptFlag(cuSrch->oInf, flag);
  else
  {
    fprintf(stderr, "ERROR: Null pointer");
    ret |= ACC_ERR_NULL;
  }
  return ret;
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

  initCand* tempCand = new(initCand);
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

/**  Wait for CPU threads to complete  .
 *
 */
int waitForThreads(sem_t* running_threads, const char* msg, int sleepMS )
{
  infoMSG(2,2,"Wait for CPU threads to complete\n");

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
    gSpec->nctxTime += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
  }

  pthread_exit(&gSpec->nctxTime);

  return (NULL);
}

long long initCudaContext(gpuSpecs* gSpec)
{
  if (gSpec)
  {
    infoMSG(4, 4, "Creating context pthread for CUDA context initialisation.\n");

    int iret1 = 1;

#ifndef DEBUG
    iret1 = pthread_create( &gSpec->cntxThread, NULL, contextInitTrd, (void*) gSpec);
#endif

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
	gSpec->nctxTime += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
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
