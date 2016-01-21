#include <cufft.h>
#include <algorithm>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>

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

#ifdef CBL
#include <unistd.h>
#include "log.h"
#endif

__device__ __constant__ int           HEIGHT_HARM[MAX_HARM_NO];    ///< Plane  height  in stage order
__device__ __constant__ int           STRIDE_HARM[MAX_HARM_NO];    ///< Plane  stride  in stage order
__device__ __constant__ int           WIDTH_HARM[MAX_HARM_NO];     ///< Plane  strides   in family
__device__ __constant__ fcomplexcu*   KERNEL_HARM[MAX_HARM_NO];    ///< Kernel pointer in stage order
__device__ __constant__ stackInfo     STACKS[64];
__device__ __constant__ int           STK_STRD[4];                 ///< Stride of the stacks
__device__ __constant__ char          STK_INP[4][4069];            ///< input details


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

/** Return the first value of 2^n >= x
 */
__host__ __device__ long long next2_to_n_cu(long long x)
{
  long long i = 1;

  while (i < x)
    i <<= 1;

  return i;
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

/* The fft length needed to properly process a subharmonic */
int calc_fftlen3(double harm_fract, int max_zfull, uint accelLen, presto_interp_acc accuracy)
{
  int bins_needed, end_effects;

  bins_needed = accelLen * harm_fract + 2;
  end_effects = 2 * ACCEL_NUMBETWEEN * z_resp_halfwidth(calc_required_z(harm_fract, max_zfull), accuracy);
  return next2_to_n_cu(bins_needed + end_effects);
}

/** Calculate an optimal accellen given a width  .
 *
 * @param width the width of the plane usually a power of two
 * @param zmax
 * @return
 * If width is not a power of two it will be rounded up to the nearest power of two
 */
uint optAccellen(float width, int zmax, presto_interp_acc accuracy)
{
  float halfwidth       = z_resp_halfwidth(zmax, accuracy); /// The halfwidth of the maximum zmax, to calculate accel len
  float pow2            = pow(2 , round(log2(width)) );
  uint oAccelLen        = floor(pow2 - 2 - 2 * ACCEL_NUMBETWEEN * halfwidth);

  return oAccelLen;
}

/** Calculate the step size from a width if the width is < 100 it is skate to be the closest power of two  .
 *
 * @param width
 * @param zmax
 * @return
 */
uint calcAccellen(float width, float zmax, presto_interp_acc accuracy)
{
  int accelLen;

  if ( width > 100 )
  {
    accelLen = width;
  }
  else
  {
    accelLen = optAccellen(width*1000.0, zmax, accuracy) ;
  }
  return accelLen;
}

/** Allocate R value array
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

  nvtxRangePush("FFT plans");

  // Note creating the plans is the most expensive task in the GPU init, I tried doing it in parallel but it was slower
  for (int i = 0; i < kernel->noStacks; i++)
  {
    cuFfdotStack* cStack  = &kernel->stacks[i];

    FOLD //  .
    {
      sprintf(msg,"Stack %i",i);
      nvtxRangePush(msg);

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
          nvtxRangePush("FFTW");
          cStack->inpPlanFFTW = fftwf_plan_many_dft(1, n, cStack->noInStack*kernel->noSteps, (fftwf_complex*)cStack->h_iData, n, istride, idist, (fftwf_complex*)cStack->h_iData, n, ostride, odist, -1, FFTW_ESTIMATE);
          nvtxRangePop();
        }
        else
        {
          nvtxRangePush("CUFFT Inp");
          CUFFT_SAFE_CALL(cufftPlanMany(&cStack->inpPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, cStack->noInStack*kernel->noSteps), "Creating plan for input data of stack.");
          nvtxRangePop();
        }
      }

      FOLD // Create the stack iFFT plan  .
      {
        nvtxRangePush("CUFFT Pln");
        CUFFT_SAFE_CALL(cufftPlanMany(&cStack->plnPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, cStack->height*kernel->noSteps), "Creating plan for complex data of stack.");
        nvtxRangePop();
      }

      nvtxRangePop();
    }

    CUDA_SAFE_CALL(cudaGetLastError(), "Creating FFT plans for the stacks.");
  }

  nvtxRangePop();
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
int initKernel(cuFFdotBatch* kernel, cuFFdotBatch* master, cuSearch*   sInf, int devID )
{
  std::cout.flush();

  size_t free, total;                           ///< GPU memory
  int noInStack[MAX_HARM_NO];

  //int noSrchHarms     = noGenHarms;

  noInStack[0]        = 0;
  size_t batchSize    = 0;                      ///< Total size (in bytes) of all the data need by a family (ie one step) excluding FFT temporary
  size_t fffTotSize   = 0;                      ///< Total size (in bytes) of FFT temporary memory
  size_t planeSize    = 0;                      ///< Total size (in bytes) of memory required independently of batch(es)
  float plnElsSZ      = 0;                      ///< The size of an element of the in-mem ff plane (generally the size of float complex)
  float powElsSZ      = 0;                      ///< The size of an element of the powers plane

  gpuInf* gInf        = &sInf->gSpec->devInfo[devID];
  int device          = gInf->devid;
  int noBatches       = sInf->gSpec->noDevBatches[devID];
  int noSteps         = sInf->gSpec->noDevSteps[devID];
  int alignment       = gInf->alignment;

  presto_interp_acc  accuracy = LOWACC;

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering initKernel.");

  infoMSG(3,3,"%s device %i\n",__FUNCTION__, device);

  char msg[1024];
  sprintf(msg, "Dev %02i", device );
  nvtxRangePush(msg);

  FOLD // See if we can use the cuda device and whether it may be possible to do GPU in-mem search .
  {
    infoMSG(3,4,"access device %i\n", device);

    nvtxRangePush("Get Device");

    if ( device >= getGPUCount() )
    {
      fprintf(stderr, "ERROR: There is no CUDA device %i.\n", device);
      return (0);
    }
    int currentDevvice;
    CUDA_SAFE_CALL(cudaSetDevice(device), "Failed to set device using cudaSetDevice");
    CUDA_SAFE_CALL(cudaGetDevice(&currentDevvice), "Failed to get device using cudaGetDevice");
    if (currentDevvice != device)
    {
      fprintf(stderr, "ERROR: CUDA Device not set.\n");
      return (0);
    }
    else
    {
      CUDA_SAFE_CALL(cudaMemGetInfo ( &free, &total ), "Getting Device memory information");
    }

    nvtxRangePop();
  }

  FOLD // Now see if this device could do a GPU in-mem search  .
  {
    if ( master == NULL ) // For the moment lets try this on only the first card!
    {
      infoMSG(3,4,"in-mem?\n");

      int noarms        = (1 << (sInf->noHarmStages - 1) );

      double plnX       = ( sInf->sSpec->fftInf.rhi - sInf->sSpec->fftInf.rlo/(double)noarms ) / (double)( ACCEL_DR ) ; // The number of bins
      int    plnY       = calc_required_z(1.0, (float)sInf->sSpec->zMax );

      if ( sInf->sSpec->flags & FLAG_HALF )
      {
#if CUDA_VERSION >= 7050
        plnElsSZ = sizeof(half);
#else
        plnElsSZ = sizeof(float);
        fprintf(stderr, "WARNING: Half precision can only be used with CUDA 7.5 or later! Reverting to single precision!\n");
        sInf->sSpec->flags &= ~FLAG_HALF;
#endif
      }
      else
      {
        plnElsSZ = sizeof(float);
      }

      if ( sInf->sSpec->flags & FLAG_KER_HIGH )
        accuracy = HIGHACC;

      // Calculate "approximate" plane width
      uint accelLen     = calcAccellen(sInf->sSpec->pWidth, sInf->sSpec->zMax, accuracy );
      float fftLen      = calc_fftlen3(1, sInf->sSpec->zMax, accelLen, accuracy );

      double totalSize  = plnX * plnY * plnElsSZ ;
      double appRoxWrk  = plnY * fftLen * ( 4 * 3 + 1) ; // 4 planes * ( input + CUFFT )

      if ( totalSize + appRoxWrk < free )
      {
        if ( !(sInf->sSpec->flags & FLAG_SS_ALL) || (sInf->sSpec->flags & FLAG_SS_INMEM) )
        {
          printf("Device %i can do a in-mem GPU search.\n", device);
          printf("  There is %.2fGB free memory.\n  The entire f-∂f plane requires %.2f GB and the workspace ~%.2f MB.\n\n", free*1e-9, totalSize*1e-9, appRoxWrk*1e-6 );
        }

        if ( (sInf->sSpec->flags & FLAG_SS_ALL) && !(sInf->sSpec->flags & FLAG_SS_INMEM) )
        {
          fprintf(stderr,"WARNING: Opting to NOT do a in-mem search when you could!\n");
        }
        else
        {
          sInf->noGenHarms        = 1;

          if ( sInf->gSpec->noDevices > 1 )
          {
            fprintf(stderr,"  Warning: Reverting to single device search.\n");
            sInf->gSpec->noDevices = 1;
          }

          sInf->sSpec->flags |= FLAG_SS_INMEM ;

#if CUDA_VERSION >= 6050
          if ( !(sInf->sSpec->flags & FLAG_CUFFT_CB_POW) )
            fprintf(stderr,"  Warning: Doing an in-mem search with no CUFFT callbacks, this is not ideal.\n"); // It should be on by default the user must have disabled it
#else
          fprintf(stderr,"  Warning: Doing an in-mem search with no CUFFT callbacks, this is not ideal. Try upgrading to CUDA 6.5 or later.\n");
          sInf->sSpec->flags &= ~FLAG_CUFFT_ALL;
#endif

#if CUDA_VERSION >= 7050
          if ( !(sInf->sSpec->flags & FLAG_HALF) )
            fprintf(stderr,"  Warning: You could be using half precision.\n"); // They should be on by default the user must have disabled them
#else
          fprintf(stderr,"  Warning: You could be using half precision. Try upgrading to CUDA 7.5 or later.\n");
#endif

          FOLD // Set types  .
          {
            sInf->sSpec->retType &= ~CU_TYPE_ALLL;
            sInf->sSpec->retType |= CU_POWERZ_S;

            sInf->sSpec->retType &= ~CU_SRT_ALL;
            sInf->sSpec->retType |= CU_STR_ARR;
          }
        }
      }
      else
      {
        if ( !(sInf->sSpec->flags & FLAG_SS_ALL) || (sInf->sSpec->flags & FLAG_SS_INMEM) )
        {
          printf("Device %i can not do a in-mem GPU search.\n", device);
          printf("  There is %.2fGB free memory.\n  The entire f-∂f plane requires %.2f GB and the workspace ~%.2f MB.\n\n", free*1e-9, totalSize*1e-9, appRoxWrk*1e-6 );
        }

        if ( sInf->sSpec->flags & FLAG_SS_INMEM  )
        {
          fprintf(stderr,"ERROR: Requested an in-memory GPU search, this is not possible\n\tThere is %.2f GB of free memory.\n\tIn-mem GPU search would require ~%.2f GB\n\n", free*1e-9, (totalSize + appRoxWrk)*1e-9 );
        }
        sInf->sSpec->flags &= ~FLAG_SS_INMEM ;
      }

      if ( !(sInf->sSpec->flags & FLAG_SS_ALL) )
      {
        // Default to S&S 1.
        sInf->sSpec->flags |= FLAG_SS_10;
        sInf->sSpec->flags |= FLAG_RET_STAGES;
      }
    }
  }

  FOLD // Do a sanity check on Flags and CUDA version  .
  {
    // TODO: do a check whether there is enough precision in an int to store the index of the largest point

    if ( master == NULL ) // For the moment lets try this on only the first card!
    {
      infoMSG(3,4,"FLAGS\n");

      // CUFFT callbacks
#if CUDA_VERSION < 6050
      sInf->sSpec->flags &= ~FLAG_CUFFT_ALL;
#endif

      if ( (sInf->sSpec->flags & FLAG_HALF) && !(sInf->sSpec->flags & FLAG_SS_INMEM) && !(sInf->sSpec->flags & FLAG_CUFFT_CB_POW) )
      {
#if CUDA_VERSION >= 7050
        fprintf(stderr, "WARNING: Can't use half precision with out of memory search and no CUFFT callbacks. Reverting to single precision!\n");
#endif
        sInf->sSpec->flags &= ~FLAG_HALF;
      }

      if ( !(sInf->sSpec->flags & FLAG_SS_INMEM) && (sInf->sSpec->flags & FLAG_CUFFT_CB_INMEM) )
      {
        fprintf(stderr, "WARNING: Can't use inmem callback with out of memory search. Disabling in-mem callback.\n");
        sInf->sSpec->flags &= ~FLAG_CUFFT_CB_INMEM;
      }

      if ( (sInf->sSpec->flags & FLAG_CUFFT_CB_POW) && (sInf->sSpec->flags & FLAG_CUFFT_CB_INMEM) )
      {
        fprintf(stderr, "WARNING: in-mem CUFFT callback will supersede power callback, I have found power callbacks to be the best.\n");
        sInf->sSpec->flags &= ~FLAG_CUFFT_CB_POW;
      }

      if ( (sInf->sSpec->flags & FLAG_SS_10) || (sInf->sSpec->flags & FLAG_SS_INMEM) )
      {
        sInf->sSpec->flags |= FLAG_RET_STAGES;
      }

      char typeS[1024];
      sprintf(typeS, "Doing");

      if ( sInf->sSpec->flags & FLAG_SS_INMEM )
        sprintf(typeS, "%s a in-memory", typeS);
      else
        sprintf(typeS, "%s an out of memory", typeS);

      sprintf(typeS, "%s search using", typeS);
      if ( sInf->sSpec->flags & FLAG_HALF )
        sprintf(typeS, "%s half", typeS);
      else
        sprintf(typeS, "%s single", typeS);

      sprintf(typeS, "%s precision", typeS);
      if ( sInf->sSpec->flags & FLAG_CUFFT_CB_POW )
        sprintf(typeS, "%s and CUFFT callbacks to calculate powers.", typeS);
      else if ( sInf->sSpec->flags & FLAG_CUFFT_CB_INMEM )
        sprintf(typeS, "%s and CUFFT callbacks to calculate powers and store in the full plane.", typeS);
      else
        sprintf(typeS, "%s and no CUFFT callbacks.", typeS);

      printf("\n%s\n\n", typeS);
    }

    FOLD // Determine the size of the elements of the planes  .
    {
      // Half precision?
      if ( sInf->sSpec->flags & FLAG_HALF )
      {
#if CUDA_VERSION >= 7050
        plnElsSZ = sizeof(half);
#else
        plnElsSZ = sizeof(float);
        fprintf(stderr, "WARNING: Half precision can only be used with CUDA 7.5 or later! Reverting to single precision!\n");
        sInf->sSpec->flags &= ~FLAG_HALF;
#endif
      }
      else
      {
        plnElsSZ = sizeof(float);
      }

      // Set power plane size
      if ( sInf->sSpec->flags & FLAG_CUFFT_CB_POW )
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
    infoMSG(3,4,"Allocate and zero structures\n");

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
        kernel->flags         = sInf->sSpec->flags;
        kernel->srchMaster    = 1;
        kernel->noHarmStages  = sInf->noHarmStages;
        kernel->noGenHarms    = sInf->noGenHarms;
        kernel->noSrchHarms   = sInf->noSrchHarms;
      }
    }

    FOLD // Set the device specific parameters  .
    {
      kernel->sInf          = sInf;
      kernel->device        = device;
      kernel->isKernel      = 1;                // This is the device master
      kernel->capability    = gInf->capability;
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

  FOLD // Determine how many stacks and how many planes in each stack  .
  {
    if ( master == NULL ) 	// Calculate details for the batch  .
    {
      infoMSG(3,4,"Determine number of stacks and planes\n");

      FOLD // Determine accellen and step size  .
      {
        infoMSG(3,5,"Determining step size and width\n");

        printf("Determining GPU step size and plane width:\n");

        if ( kernel->noSrchHarms > 1 )
        {
          // Working with a family of planes

          int   oAccelLen1, oAccelLen2;

          // This adjustment makes sure no more than half the harmonics are in the largest stack (reduce waisted work - gives a 0.01 - 0.12 speed increase )
          oAccelLen1  = calcAccellen(sInf->sSpec->pWidth,     sInf->sSpec->zMax, accuracy);
          oAccelLen2  = calcAccellen(sInf->sSpec->pWidth/2.0, sInf->sSpec->zMax/2.0, accuracy);

          if ( sInf->sSpec->pWidth > 100 )
          {
            // The user specified the exact width they want to use for accellen
            kernel->accelLen  = oAccelLen1;
          }
          else
          {
            // Use double the accellen of the half plane
            kernel->accelLen  = MIN(oAccelLen2*2, oAccelLen1);
          }

          if ( sInf->sSpec->pWidth < 100 ) // Check  .
          {
            float fWidth    = floor(calc_fftlen3(1, sInf->sSpec->zMax, kernel->accelLen, accuracy)/1000.0);

            float ss        = calc_fftlen3(1, sInf->sSpec->zMax, kernel->accelLen, accuracy) ;
            float l2        = log2( ss );

            if      ( l2 == 10 )
              fWidth = 1 ;
            else if ( l2 == 11 )
              fWidth = 2 ;
            else if ( l2 == 12 )
              fWidth = 4 ;
            else if ( l2 == 13 )
              fWidth = 8 ;
            else if ( l2 == 14 )
              fWidth = 16 ;
            else if ( l2 == 15 )
              fWidth = 32 ;
            else if ( l2 == 16 )
              fWidth = 64 ;

            if ( fWidth != sInf->sSpec->pWidth )
            {
              fprintf(stderr,"ERROR: Width calculation did not give the desired value.\n");
              exit(EXIT_FAILURE);
            }
          }
        }
        else
        {
          // Just a single plane
          kernel->accelLen = calcAccellen(sInf->sSpec->pWidth, sInf->sSpec->zMax, accuracy);
        }

        FOLD // Now make sure that accelLen is divisible by (noSrchHarms*ACCEL_RDR) this "rule" is used for indexing in the sum and search kernel
        {
          kernel->accelLen = floor( kernel->accelLen/(float)(kernel->noSrchHarms*ACCEL_RDR) ) * (kernel->noSrchHarms*ACCEL_RDR);

          if ( sInf->sSpec->pWidth > 100 ) // Check  .
          {
            if ( sInf->sSpec->pWidth != kernel->accelLen )
            {
              fprintf(stderr,"ERROR: Using manual step size, value must be divisible by numharm * %i (%i) try %i.\n", ACCEL_RDR, kernel->noSrchHarms*ACCEL_RDR, kernel->accelLen );
              exit(EXIT_FAILURE);
            }
          }
        }

        FOLD // Print kernel accuracy  .
        {
          printf(" • Using ");

          if ( sInf->sSpec->flags & FLAG_KER_HIGH )
          {
            printf("high ");
          }
          else
          {
            printf("standard ");
          }
          printf("accuracy response functions.\n");

          if ( sInf->sSpec->flags & FLAG_KER_MAX )
            printf(" • Using maximum response function length for entire kernel.\n");
        }

        if ( kernel->accelLen > 100 ) // Print output
        {
          float fftLen      = calc_fftlen3(1, sInf->sSpec->zMax, kernel->accelLen, accuracy);
          int   oAccelLen   = optAccellen(fftLen, sInf->sSpec->zMax, accuracy);
          float ratio       = kernel->accelLen/float(oAccelLen);

          printf(" • Using max plane width of %.0f and thus", fftLen);

          if    	( ratio < 0.90 )
          {
            printf(" an non-optimal step-size of %i.\n", kernel->accelLen );
            if ( sInf->sSpec->pWidth > 100 )
            {
              int K              = round(fftLen/1000.0);
              fprintf(stderr,"    WARNING: Using manual width\\step-size is not advised rather set width to one of 2 4 8 46 32.\n    For a zmax of %i using %iK FFTs the optimal step-size is %i.\n", sInf->sSpec->zMax, K, oAccelLen);
            }
          }
          else if ( ratio < 0.95 )
          {
            printf(" an close to optimal step-size of %i.\n", kernel->accelLen );
          }
          else
          {
            printf(" an optimal step-size of %i.\n", kernel->accelLen );
          }
        }
        else
        {
          fprintf(stderr,"ERROR: With a width of %i, the step-size would be %i and this is too small, try with a wider width or lower z-max.\n", sInf->sSpec->pWidth, kernel->accelLen);
          exit(EXIT_FAILURE);
        }
      }

      FOLD // Set some harmonic related values  .
      {
        int prevWidth       = 0;
        int noStacks        = 0;
        int stackHW         = 0;
        int hIdx, sIdx;
        float hFrac;

        FOLD // Set up basic details of all the harmonics  .
        {
        for (int i = kernel->noSrchHarms; i > 0; i--)
        {
          cuHarmInfo* hInfs;
          hFrac               = (i) / (float)kernel->noSrchHarms;
          hIdx                = kernel->noSrchHarms-i;
          hInfs               = &kernel->hInfos[hIdx];                              // Harmonic index

          hInfs->harmFrac     = hFrac;
          hInfs->zmax         = calc_required_z(hInfs->harmFrac, sInf->sSpec->zMax);
          hInfs->height       = (hInfs->zmax / ACCEL_DZ) * 2 + 1;
          hInfs->width        = calc_fftlen3(hInfs->harmFrac, kernel->hInfos[0].zmax, kernel->accelLen, accuracy);
          hInfs->halfWidth    = z_resp_halfwidth(hInfs->zmax, accuracy);

          if ( prevWidth != hInfs->width )
          {
            // We have a new stack
            noStacks++;

            if ( hIdx < kernel->noGenHarms )
            {
              kernel->noStacks = noStacks;
            }

            noInStack[noStacks - 1]       = 0;
            prevWidth                     = hInfs->width;
            stackHW                       = z_resp_halfwidth(hInfs->zmax, accuracy);

            // Maximise, centre and align halfwidth
            int   sWidth                  = (int) ( ceil(kernel->accelLen * hInfs->harmFrac * ACCEL_DR ) * ACCEL_RDR + DBLCORRECT ) + 1 ;     // Width of usable data for this plane
            float centHW                  = (hInfs->width  - sWidth)/2.0/(float)ACCEL_NUMBETWEEN;                                             //
            float noAlg                   = alignment / float(sizeof(fcomplex)) / (float)ACCEL_NUMBETWEEN ;                                   // halfWidth will be multiplied by ACCEL_NUMBETWEEN so can divide by it here!
            float centAlgnHW              = floor(centHW/noAlg) * noAlg ;                                                                     // Centre and aligned half width

            if ( stackHW > centAlgnHW )
            {
              stackHW                     = floor(centHW);
            }
            else
            {
              stackHW                     = centAlgnHW;
            }
          }

          hInfs->stackNo      = noStacks-1;

          if ( kernel->flags & FLAG_CENTER )
          {
            hInfs->kerStart   = stackHW*ACCEL_NUMBETWEEN;
          }
          else
          {
            hInfs->kerStart   = hInfs->halfWidth*ACCEL_NUMBETWEEN;
          }

          if ( hIdx < kernel->noGenHarms )
          {
            noInStack[noStacks - 1]++;
          }
        }
        }

        FOLD // Set up the indexing details of all the harmonics  .
        {
          // Calculate the stage order of the harmonics
          sIdx = 0;

          for ( int stage = 0; stage < kernel->noHarmStages; stage++ )
          {
            int harmtosum = 1 << stage;
            for (int harm = 1; harm <= harmtosum; harm += 2, sIdx++)
            {
              hFrac     = harm/float(harmtosum);
              hIdx      = hFrac == 1 ? 0 : round(hFrac*kernel->noSrchHarms);



              kernel->hInfos[hIdx].stageIndex   = sIdx;
              sInf->sIdx[sIdx]                  = hIdx;
            }
          }
        }
      }
    }
    else                    // Copy details from the master batch  .
    {
      // Copy memory from kernels and harmonics
      memcpy(kernel->hInfos,  master->hInfos,  kernel->noSrchHarms * sizeof(cuHarmInfo));
      memcpy(kernel->kernels, master->kernels, kernel->noGenHarms  * sizeof(cuKernel));
    }
  }

  FOLD // Allocate all the memory for the stack data structures  .
  {
    infoMSG(3,4,"Allocate memory for stacks\n");

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
        infoMSG(3,4,"Stack details\n");

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
          cStack->kerHeigth     = cStack->harmInf->height;
          cStack->flags         = kernel->flags;               // Used to create the kernel, will be over written later

          for (int j = 0; j < cStack->noInStack; j++)
          {
            cStack->startZ[j]   = cStack->height;
            cStack->height     += cStack->harmInf[j].height;
          }

          prev                 += cStack->noInStack;
        }
      }
    }

    FOLD // Calculate the stride and data thus data size of the stacks  .
    {
      // This is device specific so done on each card

      infoMSG(3,4,"Stride details\n");

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
          cStack->strideCmplx =   getStrie(cStack->width, sizeof(cufftComplex), alignment);
          cStack->stridePower =   getStrie(cStack->width, powElsSZ,             alignment);

          kernel->inpDataSize +=  cStack->strideCmplx * cStack->noInStack * sizeof(cufftComplex);
          kernel->kerDataSize +=  cStack->strideCmplx * cStack->kerHeigth * sizeof(cufftComplex);
          kernel->plnDataSize +=  cStack->strideCmplx * cStack->height    * sizeof(cufftComplex);

          if ( !(kernel->flags & FLAG_CUFFT_CB_INMEM) )
            kernel->pwrDataSize +=  cStack->stridePower * cStack->height    * powElsSZ;
        }
      }
    }
  }

  FOLD // Batch specific streams  .
  {
    nvtxRangePush("streams");

    infoMSG(3,4,"Batch streams\n");

    char strBuff[1024];

    if ( kernel->flags & FLAG_SYNCH )
    {
      cuFfdotStack* fStack = &kernel->stacks[0];

      CUDA_SAFE_CALL(cudaStreamCreate(&fStack->initStream),"Creating CUDA stream for initialisation");

      sprintf(strBuff,"%i.0.0.0 Initialisation", device );
      nvtxNameCudaStreamA(fStack->initStream, strBuff);
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

        sprintf(strBuff,"%i.0.0.%i Initialisation", device, i);
        nvtxNameCudaStreamA(cStack->initStream, strBuff);
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
          sprintf(strBuff,"%i.0.2.%i FFT Input Dev", device, i);
          nvtxNameCudaStreamA(cStack->fftIStream, strBuff);
          //printf("cudaStreamCreate: %s\n", strBuff);
        }
      }

      for (int i = 0; i < kernel->noStacks; i++)
      {
        cuFfdotStack* cStack = &kernel->stacks[i];

        CUDA_SAFE_CALL(cudaStreamCreate(&cStack->fftPStream),"Creating CUDA stream for fft's");
        sprintf(strBuff,"%i.0.4.%i FFT Plane Dev", device, i);
        nvtxNameCudaStreamA(cStack->fftPStream, strBuff);
        //printf("cudaStreamCreate: %s\n", strBuff);
      }
    }

    nvtxRangePop();
  }

  FOLD // Allocate device memory for all the kernels data  .
  {
    nvtxRangePush("kernel malloc");

    infoMSG(3,4,"Allocate device memory for all the kernels data\n");

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

    nvtxRangePop();
  }

  FOLD // Set the sizes values of the harmonics and kernels and pointers to kernel data  .
  {
    infoMSG(3,4,"Set the sizes values of the harmonics and kernels and pointers to kernel data\n");

    size_t kerSiz = 0;

    for (int i = 0; i < kernel->noStacks; i++)
    {
      cuFfdotStack* cStack            = &kernel->stacks[i];
      cStack->d_kerData               = &kernel->d_kerData[kerSiz];

      // Set the stride
      for (int j = 0; j< cStack->noInStack; j++)
      {
        // Point the plane kernel data to the correct position in the "main" kernel
        int iDiff                     = cStack->kerHeigth - cStack->harmInf[j].height ;
        float fDiff                   = iDiff / 2.0;
        cStack->kernels[j].d_kerData  = &cStack->d_kerData[cStack->strideCmplx*(int)fDiff];
        cStack->kernels[j].harmInf    = &cStack->harmInf[j];
      }
      kerSiz                          += cStack->strideCmplx * cStack->kerHeigth;
    }

  }

  FOLD // Initialise the multiplication kernels  .
  {
    if ( master == NULL )     // Create the kernels  .
    {
      infoMSG(3,4,"Initialise the multiplication kernels\n");

      // Run message
      CUDA_SAFE_CALL(cudaGetLastError(), "Before creating GPU kernels");

      FOLD // Check contamination of the largest stack  .
      {
        float contamination = (kernel->hInfos->halfWidth*2*ACCEL_NUMBETWEEN)/(float)kernel->hInfos->width*100 ;
        if ( contamination > 25 )
        {
          fprintf(stderr, "WARNING: Contamination is high, consider increasing width with the -width flag.\n");
        }
      }

      printf("\nGenerating GPU multiplication kernels using device %i\n", device);

      FOLD // Calculate the response values  .
      {
        infoMSG(3,5,"Calculate the response values\n");

        nvtxRangePush("Calc response");

        int hh      = 1;
        for (int i = 0; i < kernel->noStacks; i++)
        {
          cuFfdotStack* cStack = &kernel->stacks[i];

          float contamination = (cStack->harmInf->halfWidth*2*ACCEL_NUMBETWEEN)/(float)cStack->harmInf->width*100 ;
          float padding       = (1-(kernel->accelLen*cStack->harmInf->harmFrac + cStack->harmInf->halfWidth*2*ACCEL_NUMBETWEEN ) / cStack->harmInf->width)*100.0 ;

          printf("  ■ Stack %i has %02i f-∂f plane(s). width: %5li  stride: %5li  Height: %6li  Memory size: %7.1f MB \n", i+1, cStack->noInStack, cStack->width, cStack->strideCmplx, cStack->height, cStack->height*cStack->strideCmplx*sizeof(fcomplex)/1024.0/1024.0);

          // Call the CUDA kernels
          // Only need one kernel per stack
          createStackKernel(cStack);

          printf("    ► Created kernel %i  Size: %7.1f MB  Height %4i   Contamination: %5.2f %%  Padding: %5.2f %%\n", i+1, cStack->harmInf->height*cStack->strideCmplx*sizeof(fcomplex)/1024.0/1024.0, cStack->harmInf->zmax, contamination, padding);

          for (int j = 0; j < cStack->noInStack; j++)
          {
            printf("      • Harmonic %02i  Fraction: %5.3f   Z-Max: %4i   Half Width: %4i  Start offset: %4i \n", hh, cStack->harmInf[j].harmFrac, cStack->harmInf[j].zmax, cStack->harmInf[j].halfWidth, cStack->harmInf[j].kerStart / ACCEL_NUMBETWEEN  );
            hh++;
          }
        }

        nvtxRangePop();
      }

      FOLD // FFT the kernels  .
      {
        infoMSG(3,5,"FFT the  response values\n");

        nvtxRangePush("FFT kernels");

        fflush(stdout);
        printf("  FFT'ing the kernels ");
        fflush(stdout);

        for (int i = 0; i < kernel->noStacks; i++)
        {
          infoMSG(4,6,"Stack %i\n",i);

          cuFfdotStack* cStack = &kernel->stacks[i];

          FOLD // Create the plan  .
          {
            infoMSG(4,6,"Create plan\n");

            sprintf(msg,"Plan %i",i);
            nvtxRangePush(msg);

            int n[]             = {cStack->width};
            int inembed[]       = {cStack->strideCmplx* sizeof(fcomplexcu)};
            int istride         = 1;
            int idist           = cStack->strideCmplx;
            int onembed[]       = {cStack->strideCmplx* sizeof(fcomplexcu)};
            int ostride         = 1;
            int odist           = cStack->strideCmplx;
            int height          = cStack->kerHeigth;

            // Normal plans
            CUFFT_SAFE_CALL(cufftPlanMany(&cStack->plnPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, height), "Creating plan for FFT'ing the kernel.");
            CUDA_SAFE_CALL(cudaGetLastError(), "Creating FFT plans for the stacks.");

            nvtxRangePop();
          }

          FOLD // Call the plan  .
          {
            infoMSG(4,6,"Call the plan\n");

            sprintf(msg,"Call %i",i);
            nvtxRangePush(msg);

            CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->initStream),  "Error associating a CUFFT plan with multStream.");
            CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) cStack->d_kerData, (cufftComplex *) cStack->d_kerData, CUFFT_FORWARD), "FFT'ing the kernel data. [cufftExecC2C]");
            CUDA_SAFE_CALL(cudaGetLastError(), "FFT'ing the multiplication kernels.");

            nvtxRangePop();
          }

          FOLD // Destroy the plan  .
          {
            infoMSG(4,6,"Destroy the plan\n");

            sprintf(msg,"Dest %i",i);
            nvtxRangePush(msg);

            CUFFT_SAFE_CALL(cufftDestroy(cStack->plnPlan), "Destroying plan for complex data of stack. [cufftDestroy]");
            CUDA_SAFE_CALL(cudaGetLastError(), "Destroying the plan.");

            nvtxRangePop();
          }

          printf("•");
          fflush(stdout);
        }

        CUDA_SAFE_CALL(cudaGetLastError(), "FFT'ing the multiplication kernels.");

        printf("\n");

        nvtxRangePop();
      }

      printf("Done generating GPU multiplication kernels\n");
    }
    else
    {
      infoMSG(3,4,"Copy multiplication kernels\n");

      // TODO: Check this works in this location

      printf("• Copying multiplication kernels from device %i.\n", master->device);
      //CUDA_SAFE_CALL(cudaMemcpyPeer(kernel->d_kerData, kernel->device, master->d_kerData, master->device, master->kerDataSize ), "Copying multiplication kernels between devices.");
      CUDA_SAFE_CALL(cudaMemcpyPeerAsync(kernel->d_kerData, kernel->device, master->d_kerData, master->device, master->kerDataSize, master->stacks->initStream ), "Copying multiplication kernels between devices.");
    }
  }

  FOLD // Decide how to handle input and output and allocate required memory  .
  {
    infoMSG(3,4,"Input and output.\n");

    nvtxRangePush("data");

    printf("\nInitializing GPU %i (%s)\n", device, sInf->gSpec->devInfo[devID].name );

    printf("• Examining GPU memory of device %2i:\n", kernel->device);

    ulong freeRam;          /// The amount if free host memory
    int retSZ     = 0;      /// The size in byte of the returned data
    int candSZ    = 0;      /// The size in byte of the candidates
    int retY      = 0;      /// The number of candidates return per family (one step)
    ulong hostC   = 0;      /// The size in bytes of device memory used for candidates

    FOLD // Calculate the search size in bins  .
    {
      if ( master == NULL )
      {
        int minR              = floor ( sInf->sSpec->fftInf.rlo /(double) kernel->noSrchHarms - kernel->hInfos->halfWidth );
        int maxR              = ceil  ( sInf->sSpec->fftInf.rhi  + kernel->hInfos->halfWidth );

        searchScale* SrchSz   = new searchScale;
        sInf->SrchSz          = SrchSz;
        memset(SrchSz, 0, sizeof(searchScale));

        SrchSz->searchRLow    = sInf->sSpec->fftInf.rlo / (double)kernel->noSrchHarms;
        SrchSz->searchRHigh   = sInf->sSpec->fftInf.rhi;
        SrchSz->rLow          = minR;
        SrchSz->rHigh         = maxR;
        SrchSz->noInpR        = maxR - minR  ;  /// The number of input data points
        SrchSz->noSteps       = ( sInf->sSpec->fftInf.rhi - sInf->sSpec->fftInf.rlo ) / (float)( kernel->accelLen * ACCEL_DR ) ; // The number of planes to make

        if ( kernel->flags & FLAG_SS_INMEM   )
        {
          SrchSz->noSteps     = ( SrchSz->searchRHigh - SrchSz->searchRLow ) / (float)( kernel->accelLen * ACCEL_DR ) ; // The number of planes to make
        }

        if ( kernel->flags  & FLAG_STORE_EXP )
        {
          SrchSz->noOutpR     = ceil( (SrchSz->searchRHigh - SrchSz->searchRLow)/ACCEL_DR );
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
        kernel->mulSlices         = sInf->sSpec->mulSlices;
        kernel->mulChunk          = sInf->sSpec->mulChunk;

        FOLD // Set stack multiplication slices
        {
          for (int i = 0; i < kernel->noStacks; i++)
          {
            cuFfdotStack* cStack  = &kernel->stacks[i];
            cStack->mulSlices     = sInf->sSpec->mulSlices;
            cStack->mulChunk      = sInf->sSpec->mulChunk;
          }
        }
      }

      FOLD // Sum  & search  .
      {
        kernel->ssChunk           = sInf->sSpec->ssChunk;
        kernel->ssSlices          = sInf->sSpec->ssSlices;

        if ( kernel->ssSlices <= 0 )
        {
          if      ( kernel->stacks->width <= 1024 )
          {
            kernel->ssSlices      = 8 ; // Default value
          }
          else if ( kernel->stacks->width <= 2048 )
          {
            kernel->ssSlices      = 4 ; // Default value
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
        kernel->ssSlices          = MIN(kernel->ssSlices, ceil(kernel->hInfos->height/20.0) );
      }
    }

    FOLD // Calculate candidate type  .
    {
      if ( master == NULL )   // There is only one list of candidates per search so only do this once!
      {
        kernel->cndType         = sInf->sSpec->cndType;

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
          candSZ = sizeof(cand);
        }
        else
        {
          fprintf(stderr,"ERROR: No output type specified in %s setting to default.\n", __FUNCTION__);
          kernel->cndType |= CU_CANDFULL;
          candSZ = sizeof(cand);
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
      kernel->retType       = sInf->sSpec->retType;

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
        retSZ = sizeof(cand);
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
          retY = kernel->hInfos->height;
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
          kernel->strideOut = sInf->sSpec->ssStepSize;
        }
        else if ( (kernel->retType & CU_STR_ARR) )
        {
          //kernel->strideOut = kernel->hInfos->width;  // NOTE: This could be accellen rather than width, but to allow greater flexibility keep it at width. CU_STR_PLN    requires width
          kernel->strideOut = getStrie(kernel->accelLen, retSZ, alignment);
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
    }

    FOLD // Calculate batch size and number of steps and batches on this device  .
    {
      nvtxRangePush("Calc steps");

      CUDA_SAFE_CALL(cudaMemGetInfo ( &free, &total ), "Getting Device memory information"); // TODO: This call may not be necessary we could calculate this from previous values
      freeRam = getFreeRamCU();

      printf("   There is a total of %.2f GiB of device memory of which there is %.2f GiB free and %.2f GiB free host memory.\n",total / 1073741824.0, (free )  / 1073741824.0, freeRam / 1073741824.0 );

      FOLD // Calculate size of various memory's'  .
      {
        batchSize             = kernel->inpDataSize + kernel->plnDataSize + kernel->pwrDataSize + kernel->retDataSize;  // This is currently the size of one step
        fffTotSize            = kernel->inpDataSize + kernel->plnDataSize;                                              // FFT data treated separately because there will be only one set per device

        if ( kernel->flags & FLAG_SS_INMEM  ) // Size of memory for plane full ff plane  .
        {
          uint noStepsP       =  ceil(sInf->SrchSz->noSteps / (float)noSteps) * noSteps;
          uint nX             = noStepsP * kernel->accelLen;
          uint nY             = kernel->hInfos->height;
          planeSize          += nX * nY * plnElsSZ ;
        }
      }

      FOLD // Calculate how many batches and steps to do  .
      {
        float possSteps;
        char cufftType[1024];

        if ( kernel->flags & CU_FFT_SEP )
        {
          possSteps = ( free - planeSize ) / (double) ( (fffTotSize + batchSize) * noBatches ) ;
        }
        else
        {
          possSteps = ( free - planeSize ) / (double) (  fffTotSize + batchSize  * noBatches ) ;  // (fffTotSize * possSteps) for the CUFFT memory for FFT'ing the plane(s) and (totSize * noThreads * possSteps) for each thread(s) plan(s)
        }

        printf("     Requested %i batches on this device.\n", noBatches);
        if ( possSteps > 1 )
        {
          if ( noSteps > floor(possSteps) )
          {
            printf("      Requested %i steps per batch, but with %i batches we can only do %.2f steps per batch. \n", noSteps, noBatches, possSteps );
            noSteps = floor(possSteps);
          }

          if ( floor(possSteps) > noSteps + 1 && (noSteps < MAX_STEPS) )
          {
            printf("       Note: requested %i steps per batch, you could do up to %.2f steps per batch. \n", noSteps, possSteps );
          }

          kernel->noSteps = noSteps;

          if ( kernel->noSteps > MAX_STEPS )
          {
            kernel->noSteps = MAX_STEPS;
            printf("      Trying to use more steps that the maximum number (%i) this code is compiled with.\n", kernel->noSteps );
          }
        }
        else
        {
          printf("      There is not enough memory to crate %i batches with one plane each.\n", noBatches);

          float noSteps1    = ( free ) / (double) ( fffTotSize + batchSize ) ;
          noSteps           = MIN(MAX_STEPS, floor(noSteps1));
          kernel->noSteps   = noSteps;
          noBatches         = 1;

          printf("        Throttling to %i steps in 1 batch.\n", kernel->noSteps);
        }

        if ( noBatches <= 0 || kernel->noSteps <= 0 )
        {
          fprintf(stderr, "ERROR: Insufficient memory to make make any planes on this device. One step would require %.2fGiB of device memory.\n", ( fffTotSize + batchSize )/1073741824.0 );

          // TODO: check flags here!

          freeKernel(kernel);
          return (0);
        }

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

          if ( kernel->flags & FLAG_HALF )
          {
            printf(" (using half precision)\n");
          }
          else
          {
            printf("\n");
          }
        }
        printf("    Each batch    uses: %5.2f GiB of device memory.\n", (batchSize*kernel->noSteps) / 1073741824.0 );
        printf("                 Using: %5.2f GiB of %.2f [%.2f%%] of GPU memory for search.\n",  totUsed / 1073741824.0, total / 1073741824.0, totUsed / (float)total * 100.0f );
      }

      nvtxRangePop();
    }

    FOLD // Scale data sizes by number of steps  .
    {
      kernel->inpDataSize *= kernel->noSteps;
      kernel->plnDataSize *= kernel->noSteps;
      kernel->pwrDataSize *= kernel->noSteps;
      if ( !(kernel->flags & FLAG_SS_INMEM)  )
        kernel->retDataSize *= kernel->noSteps;       // In-mem search stage does not use steps
    }

    float fullCSize     = sInf->SrchSz->noOutpR * candSZ;               /// The full size of all candidate data

    if ( kernel->flags  & FLAG_STORE_ALL )
      fullCSize *= kernel->noHarmStages; // Store  candidates for all stages

    FOLD // DO a sanity check on flags  .
    {
      FOLD // How to handle input  .
      {
        if ( (kernel->flags & CU_INPT_FFT_CPU) && !(kernel->flags & CU_NORM_CPU) )
        {
          fprintf(stderr, "WARNING: Using CPU FFT of the input data necessitate doing the normalisation on CPU.\n");
          kernel->flags |= CU_NORM_CPU;
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
        nvtxRangePush("in-mem alloc");

        uint    noStepsP =  ceil(sInf->SrchSz->noSteps / (float)kernel->noSteps) * kernel->noSteps ;
        uint    nX       = noStepsP * kernel->accelLen;
        uint    nY       = kernel->hInfos->height;
        size_t  stride;

        CUDA_SAFE_CALL(cudaMallocPitch(&sInf->d_planeFull,    &stride, plnElsSZ*nX, nY),   "Failed to allocate device memory for getMemAlignment.");
        CUDA_SAFE_CALL(cudaMemsetAsync(sInf->d_planeFull, 0, stride*nY, kernel->stacks->initStream),"Failed to initiate plane memory to zero");

        sInf->inmemStride = stride / plnElsSZ;

        nvtxRangePop();
      }

    }

    FOLD // Allocate global (device independent) host memory  .
    {
      // One set of global set of "candidates" for all devices
      if ( master == NULL )
      {
        nvtxRangePush("host alloc");

        if      ( kernel->cndType & CU_STR_ARR  )
        {
          if ( sInf->sSpec->outData == NULL   )
          {
            // Have to allocate the array!

            freeRam  = getFreeRamCU();
            if ( fullCSize < freeRam*0.90 )
            {
              // Same host candidates for all devices
              // This can use a lot of memory for long searches!
              sInf->h_candidates = malloc( fullCSize );
              memset(sInf->h_candidates, 0, fullCSize );
              hostC += fullCSize;
            }
            else
            {
              fprintf(stderr, "ERROR: Not enough host memory for candidate list array. Need %.2fGiB there is %.2fGiB.\n", fullCSize / 1073741824.0, freeRam / 1073741824.0 );
              fprintf(stderr, "       Try set -fhi to a lower value. ie: numharm*1000. ( or buy more RAM, or close Chrome ;)\n");
              fprintf(stderr, "       Will continue trying to use a dynamic list.\n");

              kernel->cndType &= ~CU_SRT_ALL ;
              kernel->cndType |= CU_STR_LST ;
            }
          }
          else
          {
            // This memory has already been allocated
            sInf->h_candidates = sInf->sSpec->outData;
            memset(sInf->h_candidates, 0, fullCSize ); // NOTE: this may error if the preallocated memory int karge enough!
          }
        }
        else if ( kernel->cndType & CU_STR_QUAD )
        {
          if ( sInf->sSpec->outData == NULL )
          {
            candTree* qt = new candTree;
            sInf->h_candidates = qt;
          }
          else
          {
            sInf->h_candidates = sInf->sSpec->outData;
          }
        }
        else if ( kernel->cndType & CU_STR_LST  )
        {
          // Nothing really to do here =/
          GSList* lst = g_slist_alloc();
          lst->data = NULL;
          lst->next = NULL;

          sInf->h_candidates    = lst;
        }
        else if ( kernel->cndType & CU_STR_PLN  )
        {
          fprintf(stderr,"WARNING: The case of candidate planes has not been implemented!\n");

          // This memory has already been allocated
          sInf->h_candidates = sInf->sSpec->outData;
        }

        nvtxRangePop();
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

    nvtxRangePop();
  }

  FOLD // Create FFT plans, ( 1 - set per device )  .
  {
    nvtxRangePush("FFT plans");

    if ( ( kernel->flags & CU_INPT_FFT_CPU ) && master == NULL )
    {
      nvtxRangePush("read_wisdom");

      read_wisdom();

      nvtxRangePop();
    }

    if ( !(kernel->flags & CU_FFT_SEP) )
    {
      infoMSG(3,4,"Create FFT plans\n");

      createFFTPlans(kernel);
    }

    nvtxRangePop();
  }

  FOLD // Create texture memory from kernels  .
  {
    if ( kernel->flags & FLAG_TEX_MUL )
    {
      infoMSG(3,4,"Create texture memory\n");

      nvtxRangePush("text mem");

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
        resDesc.res.pitch2D.devPtr        = cStack->d_kerData;
        resDesc.res.pitch2D.width         = cStack->width;
        resDesc.res.pitch2D.pitchInBytes  = cStack->strideCmplx * sizeof(fcomplex);
        resDesc.res.pitch2D.height        = cStack->kerHeigth;

        CUDA_SAFE_CALL(cudaCreateTextureObject(&cStack->kerDatTex, &resDesc, &texDesc, NULL), "Creating texture from kernel data.");

        CUDA_SAFE_CALL(cudaGetLastError(), "Creating texture from the stack of kernel data.");

        // Create the actual texture object
        for (int j = 0; j< cStack->noInStack; j++)        // Loop through planes in stack
        {
          cuKernel* cKer = &cStack->kernels[j];

          resDesc.res.pitch2D.devPtr        = cKer->d_kerData;
          resDesc.res.pitch2D.height        = cKer->harmInf->height;
          resDesc.res.pitch2D.width         = cKer->harmInf->width;
          resDesc.res.pitch2D.pitchInBytes  = cStack->strideCmplx * sizeof(fcomplex);

          CUDA_SAFE_CALL(cudaCreateTextureObject(&cKer->kerDatTex, &resDesc, &texDesc, NULL), "Creating texture from kernel data.");
          CUDA_SAFE_CALL(cudaGetLastError(), "Creating texture from kernel data.");
        }
      }

      nvtxRangePop();
    }
  }

  FOLD // Set constant memory values  .
  {
    infoMSG(3,4,"Set constant memory values\n");

    nvtxRangePush("const mem");

    setConstVals( kernel,  sInf->noHarmStages, sInf->powerCut, sInf->numindep );

    setConstVals_Fam_Order( kernel );                            // Constant values for multiply

    setStackVals( kernel );

    FOLD // // CUFFT callbacks
    {
      if ( !(kernel->flags & CU_FFT_SEP) )
      {
#if CUDA_VERSION >= 6050        // CUFFT callbacks only implemented in CUDA 6.5
        copyCUFFT_LD_CB(kernel);
#endif
      }
    }

    nvtxRangePop();
  }

  printf("Done initializing GPU %i.\n",device);

  std::cout.flush();
  nvtxRangePop();

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

      cPlane->d_planeMult       = &cStack->d_planeMult[ cStack->startZ[j] * batch->noSteps * cStack->strideCmplx ];

      if (cStack->d_planePowr)
      {
        if ( batch->flags & FLAG_HALF )
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
    cStack->d_planeMult   = &batch->d_planeMult[cmplStart];
    if (batch->d_planePowr)
    {
      if ( batch->flags & FLAG_HALF )
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
  sprintf(msg,"%i of %i", no, of);
  nvtxRangePush(msg);

  char strBuff[1024];
  size_t free, total;

  FOLD // See if we can use the cuda device  .
  {
    infoMSG(3,4,"Device %i\n", kernel->device);

    setDevice(kernel->device) ;

    CUDA_SAFE_CALL(cudaMemGetInfo ( &free, &total ), "Getting Device memory information");
  }

  FOLD // Copy details from kernel and allocate stacks .
  {
    infoMSG(3,4,"Copy kernel\n");

    // Copy the basic batch parameters from the kernel
    memcpy(batch, kernel, sizeof(cuFFdotBatch));

    batch->srchMaster   = 0;
    batch->isKernel     = 0;

    infoMSG(3,4,"Create and copy stacks\n");

    // Allocate memory for the stacks
    batch->stacks = (cuFfdotStack*) malloc(batch->noStacks * sizeof(cuFfdotStack));

    // Copy the actual stacks
    memcpy(batch->stacks, kernel->stacks, batch->noStacks  * sizeof(cuFfdotStack));
  }

  FOLD // Set the batch specific flags  .
  {
    infoMSG(3,4,"Set flags\n");

    FOLD // Multiplication flags  .
    {
      for ( int i = 0; i < batch->noStacks; i++ )   // Multiplication is generally stack specific so loop through stacks  .
      {
        cuFfdotStack* cStack  = &batch->stacks[i];

        FOLD // multiplication kernel  .
        {
          if ( !(cStack->flags & FLAG_MUL_ALL ) )   // Default to multiplication  .
          {
            int64_t mFlag = 0;

            // In my testing I found multiplying each plane separately works fastest so it is the "default"
            int noInp =  cStack->noInStack * kernel->noSteps ;

            if ( batch->capability > 3.0 )
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
          cStack->mulSlices = MIN(cStack->mulSlices,cStack->kerHeigth/2.0);
        }

        FOLD // Chunk size  .
        {
          if ( cStack->mulChunk <= 0 )
          {
            cStack->mulChunk = 4;
          }

          // Clamp to size of kernel (ie height of the largest plane)
          cStack->mulChunk = MIN( cStack->mulChunk, ceil(cStack->kerHeigth/2.0) );
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
        //kernel->ssChunk         = 8 ;
        float val = 30.0 / (float) batch->noSteps ;

        batch->ssChunk = MAX(MIN(floor(val), 9),1);
      }
    }
  }

  FOLD // Create FFT plans  .
  {
    if ( kernel->flags & CU_FFT_SEP )
    {
      infoMSG(3,4,"Create FFT plans\n");

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
    infoMSG(3,4,"Allocate memory for the batch\n");

    FOLD // Allocate page-locked host memory for input data  .
    {
      nvtxRangePush("Host");

      CUDA_SAFE_CALL(cudaMallocHost(&batch->h_iData, batch->inpDataSize ), "Failed to create page-locked host memory plane input data." );

      if ( batch->flags & CU_NORM_CPU ) // Allocate memory for normalisation
        batch->h_normPowers = (float*) malloc(batch->hInfos->width * sizeof(float));

      nvtxRangePop();
    }

    FOLD // Allocate R value lists  .
    {
      batch->noRArryas        = 5; // This is just a convenient value

      createRvals(batch, &batch->rArr1, &batch->rArraysPlane);
      batch->rAraays = &batch->rArraysPlane;

      if ( batch->flags & FLAG_SEPRVAL )
        createRvals(batch, &batch->rArr2, &batch->rArraysSrch);

//      rVals*    rLev1;
//      rVals**   rLev2;
//
//      int oSet                = 0;
//      batch->noRArryas        = 5; // This is just a convenient value
//
//      rLev1                   = (rVals*)malloc(sizeof(rVals)*batch->noSteps*batch->noHarms*batch->noRArryas);
//      memset(rLev1, 0, sizeof(rVals)*batch->noSteps*batch->noHarms*batch->noRArryas);
//      for (int i1 = 0 ; i1 < batch->noSteps*batch->noHarms*batch->noRArryas; i1++)
//      {
//        rLev1[i1].step = -1; // Invalid step (0 is a valid value!)
//      }
//
//      (*batch->rAraays)          = (rVals***)malloc(batch->noRArryas*sizeof(rVals**));
//
//      for (int rIdx = 0; rIdx < batch->noRArryas; rIdx++)
//      {
//        rLev2                 = (rVals**)malloc(sizeof(rVals*)*batch->noSteps);
//        (*batch->rAraays)[rIdx]  = rLev2;
//
//        for (int step = 0; step < batch->noSteps; step++)
//        {
//          rLev2[step]         = &rLev1[oSet];
//          oSet               += batch->noHarms;
//        }
//      }
    }

    FOLD // Allocate device Memory for Planes, Stacks & Input data (steps)  .
    {
      nvtxRangePush("device");

      size_t req = batch->inpDataSize + batch->plnDataSize + batch->pwrDataSize;

      if ( req > free ) // Not enough memory =(
      {
        printf("Not enough GPU memory to create any more batches.\n");
        return 0;
      }
      else
      {
        if ( batch->inpDataSize )
        {
          CUDA_SAFE_CALL(cudaMalloc((void** )&batch->d_iData,       batch->inpDataSize ), "Failed to allocate device memory for batch input.");
          free -= batch->inpDataSize;
        }

        if ( batch->plnDataSize )
        {
          CUDA_SAFE_CALL(cudaMalloc((void** )&batch->d_planeMult,   batch->plnDataSize ), "Failed to allocate device memory for batch complex plane.");
          free -= batch->plnDataSize;
        }

        if ( batch->pwrDataSize )
        {
          CUDA_SAFE_CALL(cudaMalloc((void** )&batch->d_planePowr,   batch->pwrDataSize ), "Failed to allocate device memory for batch powers plane.");
          free -= batch->pwrDataSize;
        }
      }

      nvtxRangePop();
    }

    FOLD // Allocate device & page-locked host memory for return data  .
    {
      nvtxRangePush("Host");

      FOLD // Allocate device memory  .
      {
        if ( kernel->retDataSize && !(kernel->retType & CU_STR_PLN) )
        {
          if ( batch->retDataSize > free )
          {
            // Not enough memory =(
            printf("Not enough GPU memory for return data.\n");
            return 0;
          }
          else
          {
            CUDA_SAFE_CALL(cudaMalloc((void** ) &batch->d_outData1, batch->retDataSize ), "Failed to allocate device memory for return values.");
            free -= batch->retDataSize;

            if ( batch->flags & FLAG_SS_INMEM )
            {
              if ( batch->flags & FLAG_SEPSRCH )
              {
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
          CUDA_SAFE_CALL(cudaMallocHost(&batch->h_outData1, kernel->retDataSize), "Failed to create page-locked host memory plane for return data.");
          memset(batch->h_outData1, 0, kernel->retDataSize );

          if ( kernel->flags & FLAG_SS_INMEM )
          {
            CUDA_SAFE_CALL(cudaMallocHost(&batch->h_outData2, kernel->retDataSize), "Failed to create page-locked host memory plane for return data.");
            memset(batch->h_outData2, 0, kernel->retDataSize );
          }
        }
      }

      nvtxRangePop();
    }

    FOLD // Create the planes structures
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

    FOLD // Create timing arrays  .
    {
      if ( batch->flags & FLAG_TIME )
      {
        int sz = batch->noStacks*sizeof(float) ;

        batch->copyH2DTime    = (float*)malloc(sz);
        batch->normTime       = (float*)malloc(sz);
        batch->InpFFTTime     = (float*)malloc(sz);
        batch->multTime       = (float*)malloc(sz);
        batch->InvFFTTime     = (float*)malloc(sz);
        batch->copyToPlnTime  = (float*)malloc(sz);
        batch->searchTime     = (float*)malloc(sz);
        batch->resultTime     = (float*)malloc(sz);
        batch->copyD2HTime    = (float*)malloc(sz);

        memset(batch->copyH2DTime,    0, sz);
        memset(batch->normTime,       0, sz);
        memset(batch->InpFFTTime,     0, sz);
        memset(batch->multTime,       0, sz);
        memset(batch->InvFFTTime,     0, sz);
        memset(batch->copyToPlnTime,  0, sz);
        memset(batch->searchTime,     0, sz);
        memset(batch->resultTime,     0, sz);
        memset(batch->copyD2HTime,    0, sz);
      }
    }
  }

  FOLD // Setup the pointers for the stacks and planes of this batch  .
  {
    infoMSG(3,4,"Setup the pointers\n");

    setBatchPointers(batch);
  }

  FOLD // Set up the batch streams and events  .
  {
    infoMSG(3,4,"Set up the batch streams and events\n");

    FOLD // Create Streams  .
    {
      FOLD // Input streams  .
      {
        // Batch input ( Always needed, for copying input to device )
        CUDA_SAFE_CALL(cudaStreamCreate(&batch->inpStream),"Creating input stream for batch.");
        sprintf(strBuff,"%i.%i.1.0 Batch Input", batch->device, no);
        nvtxNameCudaStreamA(batch->inpStream, strBuff);
        //printf("cudaStreamCreate: %s\n", strBuff);

        // Stack input
        if ( !(batch->flags & CU_NORM_CPU)  )
        {
          for (int i = 0; i < batch->noStacks; i++)
          {
            cuFfdotStack* cStack  = &batch->stacks[i];

            CUDA_SAFE_CALL(cudaStreamCreate(&cStack->inptStream), "Creating input data multStream for stack");
            sprintf(strBuff,"%i.%i.1.%i Stack Input", batch->device, no, i);
            nvtxNameCudaStreamA(cStack->inptStream, strBuff);
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

              sprintf(strBuff,"%i.%i.2.%i Inp FFT", batch->device, no, i);
              nvtxNameCudaStreamA(cStack->fftIStream, strBuff);
              //printf("cudaStreamCreate: %s\n", strBuff);
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
          sprintf(strBuff,"%i.%i.3.0 Batch Multiply", batch->device, no);
          nvtxNameCudaStreamA(batch->multStream, strBuff);
          //printf("cudaStreamCreate: %s\n", strBuff);
        }

        if ( (batch->flags & FLAG_MUL_STK) || (batch->flags & FLAG_MUL_PLN)  )
        {
          for (int i = 0; i< batch->noStacks; i++)
          {
            cuFfdotStack* cStack  = &batch->stacks[i];

            CUDA_SAFE_CALL(cudaStreamCreate(&cStack->multStream), "Creating multStream for stack");
            sprintf(strBuff,"%i.%i.3.%i Stack Multiply", batch->device, no, i);
            nvtxNameCudaStreamA(cStack->multStream, strBuff);
            //printf("cudaStreamCreate: %s\n", strBuff);
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
            CUDA_SAFE_CALL(cudaStreamCreate(&cStack->fftPStream),"Creating CUDA stream for fft's");

            sprintf(strBuff,"%i.%i.4.%i Stack iFFT", batch->device, no, i);
            nvtxNameCudaStreamA(cStack->fftPStream, strBuff);
            //printf("cudaStreamCreate: %s\n", strBuff);
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
        sprintf(strBuff,"%i.%i.5.0 Batch Search", batch->device, no);
        nvtxNameCudaStreamA(batch->srchStream, strBuff);
        //printf("cudaStreamCreate: %s\n", strBuff);
      }

      FOLD // Result stream  .
      {
        // Batch output ( Always needed, for copying results from device )
        CUDA_SAFE_CALL(cudaStreamCreate(&batch->resStream), "Creating strmSearch for batch.");
        sprintf(strBuff,"%i.%i.6.0 Batch result", batch->device, no);
        nvtxNameCudaStreamA(batch->resStream, strBuff);
        //printf("cudaStreamCreate: %s\n", strBuff);
      }

      CUDA_SAFE_CALL(cudaGetLastError(), "Creating streams for the batch.");
    }

    FOLD // Create Events  .
    {
      FOLD // Create batch events  .
      {
        if ( batch->flags & FLAG_TIME )
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

          if ( batch->flags & FLAG_TIME )
          {
            // in  events (with timing)
            CUDA_SAFE_CALL(cudaEventCreate(&cStack->normInit),    "Creating input normalisation event");
            CUDA_SAFE_CALL(cudaEventCreate(&cStack->inpFFTinit),  "Creating input FFT initialisation event");
            CUDA_SAFE_CALL(cudaEventCreate(&cStack->multInit),    "Creating multiplication initialisation event");
            CUDA_SAFE_CALL(cudaEventCreate(&cStack->ifftInit), 	  "Creating inverse FFT initialisation event");
            CUDA_SAFE_CALL(cudaEventCreate(&cStack->ifftMemInit), "Creating inverse FFT copy initialisation event");

            // out events (with timing)
            CUDA_SAFE_CALL(cudaEventCreate(&cStack->normComp),    "Creating input normalisation event");
            CUDA_SAFE_CALL(cudaEventCreate(&cStack->prepComp), 		"Creating input data preparation complete event");
            CUDA_SAFE_CALL(cudaEventCreate(&cStack->multComp), 		"Creating multiplication complete event");
            CUDA_SAFE_CALL(cudaEventCreate(&cStack->ifftComp),    "Creating IFFT complete event");
            CUDA_SAFE_CALL(cudaEventCreate(&cStack->ifftMemComp), "Creating IFFT memory copy complete event");
          }
          else
          {
            // out events (without timing)
            CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->normComp,    cudaEventDisableTiming), "Creating input data preparation complete event");
            CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->prepComp,    cudaEventDisableTiming), "Creating input data preparation complete event");
            CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->multComp,    cudaEventDisableTiming), "Creating multiplication complete event");
            CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->ifftComp,    cudaEventDisableTiming), "Creating IFFT complete event");
            CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->ifftMemComp, cudaEventDisableTiming), "Creating IFFT memory copy complete event");
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
      infoMSG(3,4,"Create textures\n");

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
              resDesc.res.pitch2D.height          = cPlane->harmInf->height;
              resDesc.res.pitch2D.width           = cPlane->harmInf->width * batch->noSteps;
              resDesc.res.pitch2D.pitchInBytes    = cStack->harmInf->width * batch->noSteps * sizeof(float);
              resDesc.res.pitch2D.devPtr          = cPlane->d_planePowr;
            }
            else
            {
              resDesc.res.pitch2D.height          = cPlane->harmInf->height * batch->noSteps ;
              resDesc.res.pitch2D.width           = cPlane->harmInf->width;
              resDesc.res.pitch2D.pitchInBytes    = cStack->harmInf->width * sizeof(float);
              resDesc.res.pitch2D.devPtr          = cPlane->d_planePowr;
            }
          }
          else // Implies complex numbers
          {
            if      ( batch->flags & FLAG_ITLV_ROW )
            {
              resDesc.res.pitch2D.height          = cPlane->harmInf->height;
              resDesc.res.pitch2D.width           = cPlane->harmInf->width * batch->noSteps * 2;
              resDesc.res.pitch2D.pitchInBytes    = cStack->harmInf->width * batch->noSteps * 2 * sizeof(float);
              resDesc.res.pitch2D.devPtr          = cPlane->d_planePowr;
            }
            else
            {
              resDesc.res.pitch2D.height          = cPlane->harmInf->height * batch->noSteps ;
              resDesc.res.pitch2D.width           = cPlane->harmInf->width * 2;
              resDesc.res.pitch2D.pitchInBytes    = cStack->harmInf->width * 2 * sizeof(float);
              resDesc.res.pitch2D.devPtr          = cPlane->d_planePowr;
            }
          }

          CUDA_SAFE_CALL(cudaCreateTextureObject(&cPlane->datTex, &resDesc, &texDesc, NULL), "Creating texture from the plane data.");
        }
      }
      CUDA_SAFE_CALL(cudaGetLastError(), "Creating textures from the plane data.");
    }
  }

  nvtxRangePop();

  return batch->noSteps;
}

/** Free batch data structure  .
 *
 * @param batch
 */
void freeBatchGPUmem(cuFFdotBatch* batch)
{
  CUDA_SAFE_CALL(cudaGetLastError(), "Entering freeBatchGPUmem.");

  setDevice(batch->device) ;

  FOLD // Free host memory
  {
    infoMSG(2,2,"Free host memory\n");

    freeNull(batch->h_normPowers);
  }

  FOLD // Free pinned memory
  {
    infoMSG(2,2,"Free pinned memory\n");

    cudaFreeHostNull(batch->h_iData);
    cudaFreeHostNull(batch->h_outData1);
  }

  FOLD // Free device memory
  {
    infoMSG(2,2,"Free device memory\n");

    // Free the output memory
    if ( batch->d_outData1 == batch->d_planeMult )
    {
      batch->d_outData1 = NULL;
    }
    else if ( batch->d_outData2 == batch->d_planeMult )
    {
      batch->d_outData2 = NULL;
    }

    if ( batch->d_outData1 == batch->d_outData2 )
    {
      cudaFreeNull(batch->d_outData1);
      batch->d_outData2 = NULL;
    }
    else
    {
      cudaFreeNull(batch->d_outData1);
      cudaFreeNull(batch->d_outData2);
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

    if ( batch->flags & FLAG_TIME )
    {
      freeNull(batch->copyH2DTime   );
      freeNull(batch->normTime      );
      freeNull(batch->InpFFTTime    );
      freeNull(batch->multTime      );
      freeNull(batch->InvFFTTime    );
      freeNull(batch->copyToPlnTime );
      freeNull(batch->searchTime    );
      freeNull(batch->resultTime    );
      freeNull(batch->copyD2HTime   );
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

  if ( !oPln )
  {
    oPln = (cuOptCand*)malloc(sizeof(cuOptCand));
    memset(oPln,0,sizeof(cuOptCand));

    if ( devLstId < MAX_GPUS )
    {
      oPln->device = sSrch->gSpec->devId[devLstId];
    }
    else
    {
      fprintf(stderr, "ERROR: Device list index is greater that the list length, in function: %s.\n", __FUNCTION__);
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    if ( oPln->device != sSrch->gSpec->devId[devLstId] )
    {
      bool found = false;

      for ( int lIdx = 0; lIdx < MAX_GPUS; lIdx++ )
      {
        if ( sSrch->gSpec->devId[lIdx] == oPln->device )
        {
          devLstId = lIdx;
          found = true;
          break;
        }
      }

      if (!found)
      {
        if (devLstId < MAX_GPUS )
        {
          oPln->device = sSrch->gSpec->devId[devLstId];
        }
        else
        {
          fprintf(stderr, "ERROR: Device list index is greater that the list length, in function: %s.\n", __FUNCTION__);
          exit(EXIT_FAILURE);
        }

      }
    }
  }

  FOLD // Create stuff  .
  {
    setDevice(oPln->device) ;

    int   noHarms       = (1<<(sSpec->noHarmStages-1));
    float zMax          = MAX(sSpec->zMax+50, sSpec->zMax*2);
    zMax                = MAX(zMax, 60 * noHarms );
    //zMax                = MAX(zMax, sSpec->zMax * 34 + 50 );  // This may be a bit high!

    oPln->maxHalfWidth  = z_resp_halfwidth( zMax, HIGHACC );
    oPln->maxNoR        = 512;
    oPln->maxNoZ        = 512;
    oPln->outSz         = oPln->maxNoR * oPln->maxNoZ ;       // This needs to be multiplied by the size of the output element
    oPln->alignment     = sSrch->gSpec->devInfo[devLstId].alignment; //getMemAlignment();

    // Create streams
    CUDA_SAFE_CALL(cudaStreamCreate(&oPln->stream),"Creating stream for candidate optimisation.");
    char nmStr[1024];
    sprintf(nmStr,"Optimisation Stream %02i", oPln->pIdx);
    nvtxNameCudaStreamA(oPln->stream, nmStr);
    //printf("cudaStreamCreate: %s\n", nmStr);

    // Events
    CUDA_SAFE_CALL(cudaEventCreate(&oPln->inpInit),     "Creating input event inpInit." );
    CUDA_SAFE_CALL(cudaEventCreate(&oPln->inpCmp),      "Creating input event inpCmp."  );
    CUDA_SAFE_CALL(cudaEventCreate(&oPln->compInit),    "Creating input event compInit.");
    CUDA_SAFE_CALL(cudaEventCreate(&oPln->compCmp),     "Creating input event compCmp." );
    CUDA_SAFE_CALL(cudaEventCreate(&oPln->outInit),     "Creating input event outInit." );
    CUDA_SAFE_CALL(cudaEventCreate(&oPln->outCmp),      "Creating input event outCmp."  );

    size_t freeMem, totalMem;

    oPln->outSz        *= sizeof(float);
    oPln->inpSz         = (oPln->maxNoR + 2*oPln->maxHalfWidth)*noHarms*sizeof(cufftComplex)*2;

    CUDA_SAFE_CALL(cudaMemGetInfo ( &freeMem, &totalMem ), "Getting Device memory information");

    if ( (oPln->inpSz + oPln->outSz) > freeMem )
    {
      printf("Not enough GPU memory to create any more stacks.\n");
      free(oPln);
      return NULL;
    }
    else
    {
      // Allocate device memory
      CUDA_SAFE_CALL(cudaMalloc(&oPln->d_out,  oPln->outSz),   "Failed to allocate device memory for kernel stack.");
      CUDA_SAFE_CALL(cudaMalloc(&oPln->d_inp,  oPln->inpSz),   "Failed to allocate device memory for kernel stack.");

      // Allocate host memory
      CUDA_SAFE_CALL(cudaMallocHost(&oPln->h_out,  oPln->outSz), "Failed to allocate device memory for kernel stack.");
      CUDA_SAFE_CALL(cudaMallocHost(&oPln->h_inp,  oPln->inpSz), "Failed to allocate device memory for kernel stack.");
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

    int           height[MAX_HARM_NO];
    int           stride[MAX_HARM_NO];
    int            width[MAX_HARM_NO];
    fcomplexcu*   kerPnt[MAX_HARM_NO];

    FOLD // Set values  .
    {
      for (int i = 0; i < batch->noGenHarms; i++)
      {
        cuFfdotStack* cStack  = &batch->stacks[ batch->hInfos[i].stackNo];

        height[i] = batch->hInfos[i].height;
        stride[i] = cStack->strideCmplx;
        width[i]  = batch->hInfos[i].width;
        kerPnt[i] = batch->kernels[i].d_kerData;

        if ( (i>=batch->noGenHarms) &&  (batch->hInfos[i].width != cStack->strideCmplx) )
        {
          fprintf(stderr,"ERROR: Width is not the same as stride, using width this may case errors in the multiplication.\n");
        }
      }

      // Rest
      for (int i = batch->noGenHarms; i < MAX_HARM_NO; i++)
      {
        height[i] = 0;
        stride[i] = 0;
        width[i]  = 0;
        kerPnt[i] = 0;
      }
    }

    cudaGetSymbolAddress((void **)&dcoeffs, HEIGHT_HARM);
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, &height, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice, batch->stacks->initStream),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, STRIDE_HARM);
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, &stride, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice, batch->stacks->initStream),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, WIDTH_HARM);
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, &width,  MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice, batch->stacks->initStream),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, KERNEL_HARM);
    CUDA_SAFE_CALL(cudaMemcpyAsync(dcoeffs, &kerPnt, MAX_HARM_NO * sizeof(fcomplexcu*), cudaMemcpyHostToDevice, batch->stacks->initStream),      "Copying stages to device");
  }

  CUDA_SAFE_CALL(cudaGetLastError(), "Preparing the constant memory values for the multiplications.");

  return 1;
}

int setStackVals( cuFFdotBatch* batch )
{
  stackInfo* dcoeffs;
  //if ( batch->isKernel )
  {
    int         l_STK_STRD[4];
    char        l_STK_INP[4][4069];

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
          for ( int h = 0; h < hInf->height; h++ )
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
  infoMSG(3,4,"set ConstStkInfo(%i)\n", noStacks );

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
    if ( batch->flags & FLAG_TIME ) // Timing  .
    {
      infoMSG(1,2,"Timing\n");

      float time;         // Time in ms of the thing
      cudaError_t ret;    // Return status of cudaEventElapsedTime

      FOLD // Norm Timing  .
      {
        if ( !(batch->flags & CU_NORM_CPU) )
        {
          for (int ss = 0; ss < batch->noStacks; ss++)
          {
            cuFfdotStack* cStack = &batch->stacks[ss];

            ret = cudaEventElapsedTime(&time, cStack->normInit, cStack->normComp);

            if ( ret != cudaErrorNotReady )
            {
#pragma omp atomic
              batch->normTime[ss] += time;
            }
          }

          CUDA_SAFE_CALL(cudaGetLastError(), "Norm Timing");
        }
      }

      FOLD // Input FFT timing  .
      {
        if ( !(batch->flags & CU_INPT_FFT_CPU) )
        {
          for (int ss = 0; ss < batch->noStacks; ss++)
          {
            cuFfdotStack* cStack = &batch->stacks[ss];

            ret = cudaEventElapsedTime(&time, cStack->inpFFTinit, cStack->prepComp);

            if ( ret != cudaErrorNotReady )
            {
#pragma omp atomic
              batch->InpFFTTime[ss] += time;
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
          batch->copyH2DTime[0] += time;
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
              batch->multTime[0] += time;
            }
          }
          else                                    // Convolution was on a per stack basis  .
          {
            for (int ss = 0; ss < batch->noStacks; ss++)              // Loop through Stacks
            {
              cuFfdotStack* cStack = &batch->stacks[ss];

              ret = cudaEventElapsedTime(&time, cStack->multInit, cStack->multComp);

              if ( ret != cudaErrorNotReady )
              {
#pragma omp atomic
                batch->multTime[ss] += time;
              }
            }
          }

          CUDA_SAFE_CALL(cudaGetLastError(), "Multiplication timing");
        }
      }

      FOLD // Inverse FFT timing  .
      {
        for (int ss = 0; ss < batch->noStacks; ss++)
        {
          cuFfdotStack* cStack = &batch->stacks[ss];

          ret = cudaEventElapsedTime(&time, cStack->ifftInit, cStack->ifftComp);
          if ( ret != cudaErrorNotReady )
          {
#pragma omp atomic
            batch->InvFFTTime[ss] += time;
          }
        }

        CUDA_SAFE_CALL(cudaGetLastError(), "Inverse FFT timing");
      }

      FOLD // Copy to in-mem plane timing  .
      {
        if ( batch->flags & FLAG_SS_INMEM )
        {
          for (int ss = 0; ss < batch->noStacks; ss++)
          {
            cuFfdotStack* cStack = &batch->stacks[ss];

            ret = cudaEventElapsedTime(&time, cStack->ifftMemInit, cStack->ifftMemComp);
            if ( ret != cudaErrorNotReady )
            {
#pragma omp atomic
              batch->copyToPlnTime[ss] += time;
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
            batch->searchTime[0] += time;
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
            batch->copyD2HTime[0] += time;
          }

          CUDA_SAFE_CALL(cudaGetLastError(), "Copy D2H Timing");
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

void search_ffdot_batch_CU(cuFFdotBatch* batch, double* searchRLow, double* searchRHi, int norm_type )
{
  infoMSG(1,1,"search_ffdot_batch_CU\n");

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering search_ffdot_batch_CU.");

  // Calculate R values
  setActiveBatch(batch, 0);
  setGenRVals(batch, searchRLow, searchRHi );

  if ( batch->flags & FLAG_SYNCH )
  {
    initInput(batch, norm_type);

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
    setActiveBatch(batch, 0);
    initInput(batch, norm_type);

    if  ( batch->flags & FLAG_SS_INMEM )
    {
      setActiveBatch(batch, 0);
      multiplyBatch(batch);

      setActiveBatch(batch, 1);
      copyToInMemPln(batch);

      setActiveBatch(batch, 0);
      IFFTBatch(batch);
    }
    else
    {
      setActiveBatch(batch, 1);
      sumAndSearch(batch);

      setActiveBatch(batch, 2);
      processSearchResults(batch);

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

      nvtxRangePush("EventSynch");
      cuFfdotStack* cStack = &batch->stacks[ss];
      CUDA_SAFE_CALL(cudaEventSynchronize(cStack->ifftMemComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
      nvtxRangePop();
    }

    infoMSG(3,4,"pre synchronisation [blocking] processComp\n");

    nvtxRangePush("EventSynch");
    CUDA_SAFE_CALL(cudaEventSynchronize(batch->processComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
    nvtxRangePop();
  }
}

void max_ffdot_planeCU(cuFFdotBatch* batch, double* searchRLow, double* searchRHi, int norm_type, fcomplexcu* fft, long long* numindep, float* powers)
{
  CUDA_SAFE_CALL(cudaGetLastError(), "Entering ffdot_planeCU2.");

  FOLD // Initialise input data  .
  {
    setActiveBatch(batch, 0);
    initInput(batch, norm_type);
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
    fprintf(myfile, "#\tr\tf\tz\tfd\tsig\tpower\tharm \n");
    int i = 0;

    while ( inp_list->next )
    {
      fprintf(myfile, "%i\t%14.5f\t%10.6f\t%14.2f\t%13.10f\t%-7.4f\t%7.2f\t%i \n", i+1, ((accelcand *) (inp_list->data))->r, ((accelcand *) (inp_list->data))->r / T, ((accelcand *) (inp_list->data))->z,((accelcand *) (inp_list->data))->z/T/T, ((accelcand *) (inp_list->data))->sigma, ((accelcand *) (inp_list->data))->power, ((accelcand *) (inp_list->data))->numharm );
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
    gSpec.noDevBatches[i] = 2;
    gSpec.noDevSteps[i]   = 4;
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

      if      ( strCom(line, "FLAG_ITLV_ROW" ) || strCom(line, "INTERLEAVE_ROW" ) ||  strCom(line, "IL_ROW" ) )
      {
        (*flags) |= FLAG_ITLV_ROW;
      }
      else if ( strCom(line, "FLAG_ITLV_PLN" ) || strCom(line, "INTERLEAVE_PLN" ) ||  strCom(line, "IL_PLN" ) )
      {
        (*flags) &= ~FLAG_ITLV_ROW;
      }

      else if ( strCom(line, "FLAG_KER_STD"  ) )
      {
        (*flags) &= ~FLAG_KER_HIGH;
      }
      else if ( strCom(line, "FLAG_KER_HIGH" ) )
      {
        (*flags) |= FLAG_KER_HIGH;
      }
      else if ( strCom(line, "FLAG_KER_MAX"  ) )
      {
        (*flags) |= FLAG_KER_MAX;
      }
      else if ( strCom(line, "FLAG_CENTER"   ) )
      {
        (*flags) |= FLAG_CENTER;
      }

      else if ( strCom(line, "CU_NORM_CPU" ) || strCom(line, "NORM_CPU" ) )
      {
        (*flags) |= CU_NORM_CPU;
      }
      else if ( strCom(line, "CU_NORM_GPU" ) || strCom(line, "NORM_GPU" ) )
      {
        (*flags) &= ~CU_NORM_CPU;
      }

      else if ( strCom(line, "CU_INPT_FFT_CPU" ) || strCom(line, "CPU_FFT" ) || strCom(line, "FFT_CPU" ) )
      {
        (*flags) |= CU_NORM_CPU;
        (*flags) |= CU_INPT_FFT_CPU;
      }
      else if ( strCom(line, "CU_INPT_GPU_FFT" ) || strCom(line, "GPU_FFT" ) || strCom(line, "FFT_GPU" ) )
      {
        (*flags) &= ~CU_INPT_FFT_CPU;
      }

      else if ( strCom(line, "FLAG_MUL_00" ) || strCom(line, "MUL_00" ) )
      {
        (*flags) &= ~FLAG_MUL_ALL;
        (*flags) |=  FLAG_MUL_00;
      }
      else if ( strCom(line, "FLAG_MUL_11" ) || strCom(line, "MUL_11" ) )
      {
        (*flags) &= ~FLAG_MUL_ALL;
        (*flags) |=  FLAG_MUL_11;
      }
      else if ( strCom(line, "FLAG_MUL_21" ) || strCom(line, "MUL_21" ) )
      {
        (*flags) &= ~FLAG_MUL_ALL;
        (*flags) |=  FLAG_MUL_21;
      }
      else if ( strCom(line, "FLAG_MUL_22" ) || strCom(line, "MUL_22" ) )
      {
        (*flags) &= ~FLAG_MUL_ALL;
        (*flags) |=  FLAG_MUL_22;
      }
      else if ( strCom(line, "FLAG_MUL_23" ) || strCom(line, "MUL_23" ) )
      {
        (*flags) &= ~FLAG_MUL_ALL;
        (*flags) |=  FLAG_MUL_23;
      }
      else if ( strCom(line, "FLAG_MUL_30" ) || strCom(line, "MUL_30" ) )
      {
        (*flags) &= ~FLAG_MUL_ALL;
        (*flags) |=  FLAG_MUL_30;
      }
      else if ( strCom(line, "FLAG_MUL_CB" ) || strCom(line, "MUL_CB" ) )
      {
#if CUDA_VERSION >= 6050
        (*flags) &= ~FLAG_MUL_ALL;
        (*flags) |=  FLAG_MUL_CB;
#else
        line[flagLen] = 0;
        fprintf(stderr, "WARNING: Use of CUDA callbacks requires CUDA 6.5 or greater.  (FLAG: %s line %i in %s)\n", line, lineno, fName);
#endif
      }
      else if ( strCom(line, "FLAG_MUL_A"  ) || strCom(line, "MUL_A"  ) )
      {
        (*flags) &= ~FLAG_MUL_ALL;
      } 

      else if ( strCom(line, "FLAG_FFT_SEPERATE"  ) || strCom(line, "FLAG_FFT_SEP"  ) )
      {
        (*flags) |= CU_FFT_SEP;
      }

      else if ( strCom(line, "FLAG_TEX_MUL" ) )
      {
        fprintf(stderr, "WARNING: The flag FLAG_TEX_MUL has been deprecated.\n");
        //(*flags) |= FLAG_TEX_MUL;
      }

      else if ( strCom(line, "MUL_Chunk"  ) || strCom(line, "MUL_CHUNK"  ) )
      {
        char str1[1024];
        char str2[1024];
        int no;
        int read1 = sscanf(line, "%s %i ", str1, &no  );
        int read2 = sscanf(line, "%s %s ", str1, str2 );

        if ( read1 == 2 )
        {
          sSpec->mulChunk = no;
        }
        else if ( strCom(str2, "AA"  ) || strCom(str2, "A"   ) )
        {
          sSpec->mulChunk = 0;
        }
        else
        {
          line[flagLen] = 0;
          fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", line, lineno, fName);
        }
      }
      else if ( strCom(line, "MUL_Slices" ) || strCom(line, "MUL_SLICES" ) )
      {
        char str1[1024];
        char str2[1024];
        int no;
        int read1 = sscanf(line, "%s %i ", str1, &no  );
        int read2 = sscanf(line, "%s %s ", str1, str2 );

        if ( read1 == 2 )
        {
          sSpec->mulSlices = no;
        }
        else if ( strCom(str2, "AA"  ) || strCom(str2, "A"   ) )
        {
          sSpec->mulSlices = 0;
        }
        else
        {
          line[flagLen] = 0;
          fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", line, lineno, fName);
        }
      }

      else if ( strCom(line, "FLAG_CUFFT_CB_POW" ) 		|| strCom(line, "CB_POW"   ) )
      {
#if CUDA_VERSION >= 6050
        (*flags) |= FLAG_CUFFT_CB_POW;
#else
        line[flagLen] = 0;
        fprintf(stderr,"WARNING: Use of CUDA callbacks requires CUDA 6.5 or greater.  (FLAG: %s line %i in %s)\n", line, lineno, fName);
#endif
      }
      else if ( strCom(line, "FLAG_CUFFT_CB_INMEM" )  || strCom(line, "CB_INMEM" ) )
      {
#if CUDA_VERSION >= 6050
        (*flags) |= FLAG_CUFFT_CB_INMEM;
#else
        line[flagLen] = 0;
        fprintf(stderr,"WARNING: Use of CUDA callbacks requires CUDA 6.5 or greater.  (FLAG: %s line %i in %s)\n", line, lineno, fName);
#endif
      }
      else if ( strCom(line, "FLAG_NO_CB" )           || strCom(line, "NO_CB" 	 ) )
      {
        (*flags) &= ~FLAG_CUFFT_ALL;
      }

      else if ( strCom(line, "FLAG_SAS_TEX" ) )
      {
        (*flags) |= FLAG_SAS_TEX;
      }

      else if ( strCom(line, "FLAG_TEX_INTERP" ) )
      {
        (*flags) |= FLAG_SAS_TEX;
        (*flags) |= FLAG_TEX_INTERP;
      }

      else if ( strCom(line, "FLAG_SIG_GPU" ) || strCom(line, "SIG_GPU" ) )
      {
        (*flags) |= FLAG_SIG_GPU;
      }
      else if ( strCom(line, "FLAG_SIG_CPU" ) || strCom(line, "SIG_CPU" ) )
      {
        (*flags) &= ~FLAG_SIG_GPU;
      }

      else if ( strCom(line, "SS_INMEM_SZ" ) )
      {
        rest                = &line[ strlen("inMemSrchSz")+1];
        sSpec->ssStepSize   = atoi(rest);
      }

      else if ( strCom(line, "FLAG_SS_CPU" 	) || strCom(line, "SS_CPU" 	) )
      {
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
      else if ( strCom(line, "FLAG_SS_00"  	) || strCom(line, "SS_00"  	) )
      {
        (*flags) &= ~FLAG_SS_ALL;
        (*flags) |= FLAG_SS_00;
        (*flags) |= FLAG_RET_STAGES;
      }
      else if ( strCom(line, "FLAG_SS_10"  	) || strCom(line, "SS_10"  	) )
      {
        (*flags) &= ~FLAG_SS_ALL;
        (*flags) |= FLAG_SS_10;
        (*flags) |= FLAG_RET_STAGES;
      }
      else if ( strCom(line, "FLAG_SS_INMEM") || strCom(line, "SS_INMEM") )
      {
        (*flags) |= FLAG_SS_INMEM;
      }
      else if ( strCom(line, "FLAG_SS_A"    ) || strCom(line, "SS_A"   	) )
      {
        (*flags) &= ~FLAG_SS_ALL;
      }
      else if ( strCom(line, "FLAG_SS "    	) || strCom(line, "SS "     ) )
      {
        char str1[1024];
        char str2[1024];
        int no;
        sscanf(line, "%s %i ", str1, &no  );
        sscanf(line, "%s %s ", str1, str2 );

        if      ( no == 0 )
        {
          (*flags) &= ~FLAG_SS_ALL;
          (*flags) |= FLAG_SS_00;
          (*flags) |= FLAG_RET_STAGES;
        }
        else if ( no == 1 )
        {
          (*flags) &= ~FLAG_SS_ALL;
          (*flags) |= FLAG_SS_10;
          (*flags) |= FLAG_RET_STAGES;
        }
        else if ( strCom(str2, "AA"  ) || strCom(str2, "A"   ) )
        {
          (*flags) &= ~FLAG_SS_ALL;
        }
        else if ( strCom(line, "CPU" ) || strCom(line, "cpu" ) )
        {
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
        else if ( strCom(line, "INMEM" ) || strCom(line, "inmem" ) )
        {
          (*flags) |= FLAG_SS_INMEM;
        }
        else
        {
          line[flagLen] = 0;
          fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", line, lineno, fName);
        }
      }

      else if ( strCom(line, "SS_Chunk"  ) || strCom(line, "SS_CHUNK"  ) )
      {
        char str1[1024];
        char str2[1024];
        int no;
        int read1 = sscanf(line, "%s %i ", str1, &no  );
        int read2 = sscanf(line, "%s %s ", str1, str2 );

        if ( read1 == 2 )
        {
          sSpec->ssChunk = no;
        }
        else if ( strCom(str2, "AA"  ) || strCom(str2, "A"   ) )
        {
          sSpec->ssChunk = 0;
        }
        else
        {
          line[flagLen] = 0;
          fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", line, lineno, fName);
        }
      }
      else if ( strCom(line, "SS_Slices" ) || strCom(line, "SS_SLICES" ) )
      {
        char str1[1024];
        char str2[1024];
        int no;
        int read1 = sscanf(line, "%s %i ", str1, &no  );
        int read2 = sscanf(line, "%s %s ", str1, str2 );

        if ( read1 == 2 )
        {
          sSpec->ssSlices = no;
        }
        else if ( strCom(str2, "AA"  ) || strCom(str2, "A"   ) )
        {
          sSpec->ssSlices = 0;
        }
        else
        {
          line[flagLen] = 0;
          fprintf(stderr, "ERROR: Found unknown value for %s on line %i of %s.\n", line, lineno, fName);
        }
      }

      else if ( strCom(line, "CU_CAND_ARR"  ) || strCom(line, "CAND_ARR"  ) )
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
      else if ( strCom(line, "CU_CAND_LST"  ) || strCom(line, "CAND_LST"  ) )
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
      else if ( strCom(line, "CU_CAND_QUAD" ) || strCom(line, "CAND_QUAD" ) )
      {
        // Candidate type
        sSpec->cndType &= ~CU_TYPE_ALLL ;
        sSpec->cndType &= ~CU_SRT_ALL   ;

        sSpec->cndType |= CU_POWERZ_S   ;
        sSpec->cndType |= CU_STR_QUAD   ;
      }

      else if ( strCom(line, "FLAG_HALF" 	  ) )
      {
#if CUDA_VERSION >= 7050
        (*flags) |=  FLAG_HALF;
#else
        (*flags) &= ~FLAG_HALF;

        line[flagLen] = 0;
        fprintf(stderr,"WARNING: Half precision can only be used with CUDA 7.5 or later! Reverting to single precision. (FLAG: %s line %i in %s)\n", line, lineno, fName);
#endif
      }
      else if ( strCom(line, "FLAG_SINGLE" 	) )
      {
        (*flags) &= ~FLAG_HALF;
      }
      else if ( strCom(line, "FLAG_DOUBLE"  ) )
      {
        fprintf(stderr,"ERROR: Cannot sore in-mem plane as double! Defaulting to float.\n");
        (*flags) &= ~FLAG_HALF;
      }

      else if ( strCom(line, "FLAG_RET_STAGES" ) )
      {
        (*flags) |= FLAG_RET_STAGES;
      }
      else if ( strCom(line, "FLAG_RETURN_FINAL" ) )
      {
        (*flags) &= ~FLAG_RET_STAGES;
      }

      else if ( strCom(line, "FLAG_RET_ARR" ) )
      {
        sSpec->retType &= ~CU_SRT_ALL   ;
        sSpec->retType |= CU_STR_ARR    ;
      }
      else if ( strCom(line, "FLAG_RET_PLN" ) )
      {
        sSpec->retType &= ~CU_SRT_ALL   ;
        sSpec->retType |= CU_STR_PLN    ;
      }

      else if ( strCom(line, "FLAG_STORE_ALL" ) )
      {
        (*flags) |= FLAG_STORE_ALL;
      }

      else if ( strCom(line, "FLAG_THREAD" ) )
      {
        (*flags) |= FLAG_THREAD;
      }
      else if ( strCom(line, "FLAG_SEQ" ) )
      {
        (*flags) &= ~FLAG_THREAD;
      }

      else if ( strCom(line, "FLAG_STK_UP" ) )
      {
        (*flags) |= FLAG_STK_UP;
      }
      else if ( strCom(line, "FLAG_STK_DOWN" ) )
      {
        (*flags) &= ~FLAG_STK_UP;
      }

      else if ( strCom(line, "FLAG_CONV" ) )
      {
        (*flags) |= FLAG_CONV;
      }
      else if ( strCom(line, "FLAG_SEP" ) )
      {
        (*flags) &= ~FLAG_CONV;
      }

      else if ( strCom(line, "FLAG_STORE_EXP" ) )
      {
        (*flags) |= FLAG_STORE_EXP;
      }

      else if ( strCom(line, "FLAG_RAND_1" ) || strCom(line, "RAND_1" ) )
      {
        (*flags) |= FLAG_RAND_1;
      }


      else if ( strCom(line, "FLAG_DBG_SYNCH" ) )
      {
        (*flags) |= FLAG_SYNCH;
      }
      else if ( strCom(line, "FLAG_DBG_TIMING" ) )
      {
        //(*flags) |= FLAG_SYNCH; // Timing relies on synchronous search
        (*flags) |= FLAG_TIME;
      }

      else if ( strCom(line, "FLAG" ) || strCom(line, "CU_" ) )
      {
        line[flagLen] = 0;
        fprintf(stderr, "ERROR: Found unknown flag %s on line %i of %s.\n", line, lineno, fName);
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

      // Optimisation vars
      else if ( strCom(line, "optpln01" ) )
      {
        rest      = &line[ strlen("optpln01")+1];
        optpln01  = atoi(rest);
      }
      else if ( strCom(line, "optpln02" ) )
      {
        rest      = &line[ strlen("optpln02")+1];
        optpln02  = atoi(rest);
      }
      else if ( strCom(line, "optpln03" ) )
      {
        rest      = &line[ strlen("optpln03")+1];
        optpln03  = atoi(rest);
      }
      else if ( strCom(line, "optpln04" ) )
      {
        rest      = &line[ strlen("optpln04")+1];
        optpln04  = atoi(rest);
      }
      else if ( strCom(line, "optpln05" ) )
      {
        rest      = &line[ strlen("optpln05")+1];
        optpln05  = atoi(rest);
      }
      else if ( strCom(line, "optpln06" ) )
      {
        rest      = &line[ strlen("optpln06")+1];
        optpln06  = atoi(rest);
      }

      else if ( strCom(line, "downScale" ) )
      {
        rest      = &line[ strlen("downScale")+1];
        downScale = atof(rest);
      }

      else if ( strCom(line, "optSz01" ) )
      {
        rest      = &line[ strlen("optSz01")+1];
        optSz01   = atof(rest);
      }
      else if ( strCom(line, "optSz02" ) )
      {
        rest      = &line[ strlen("optSz02")+1];
        optSz02   = atof(rest);
      }
      else if ( strCom(line, "optSz04" ) )
      {
        rest      = &line[ strlen("optSz04")+1];
        optSz04   = atof(rest);
      }
      else if ( strCom(line, "optSz08" ) )
      {
        rest      = &line[ strlen("optSz08")+1];
        optSz08   = atof(rest);
      }
      else if ( strCom(line, "optSz16" ) )
      {
        rest      = &line[ strlen("optSz16")+1];
        optSz16   = atof(rest);
      }

      else if ( strCom(line, "pltOpt"  ) || strCom(line, "PLT_OPT" ) )
      {
        pltOpt    = 1;
      }

      else if ( strCom(line, "UNOPT" ) )
      {
        useUnopt    = 1;
      }

      else if ( strCom(line, "DBG_LEV" ) )
      {
        rest      = &line[ strlen("DBG_LEV")+1];
        msgLevel  = atoi(rest);
      }


      else if ( strCom(line, "skpOpt"  ) || strCom(line, "SKP_OPT" ) || strCom(line, "FLAG_DBG_SKIP_OPT" ) )
      {
        skpOpt  = 1;
      }

      else if ( strCom(line, "#" ) || ll == 1 )
      {
        // Comment line !
      }

      else
      {
        line[flagLen] = 0;
        fprintf(stderr, "ERROR: Found unknown flag \"%s\" on line %i of %s.\n", line, lineno, fName);
      }
    }

    fclose (file);
  }
  else
  {
    printf("Unable to read GPU accel settings from %s\n", fName);
  }
}

searchSpecs readSrchSpecs(Cmdline *cmd, accelobs* obs)
{
  searchSpecs sSpec;
  memset(&sSpec, 0, sizeof(sSpec));

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering readSrchSpecs.");

  // Defaults for accel search
  sSpec.flags         |= FLAG_RET_STAGES  ;
  sSpec.flags         |= FLAG_ITLV_ROW    ;

#ifndef DEBUG
  sSpec.flags         |= FLAG_THREAD      ; 	// Multithreading really slows down debug so only turn it on by default for release mode, NOTE: This can be over ridden in the defaults file
#endif

#if CUDA_VERSION >= 6050
  sSpec.flags         |= FLAG_CUFFT_CB_POW; 	// CUFFT callback to calculate powers, very efficient so on by default
#endif

#if CUDA_VERSION >= 7050
  sSpec.flags         |= FLAG_HALF;
#endif

  if ( obs->inmem )
  {
    sSpec.flags       |= FLAG_SS_INMEM;
  }

  sSpec.cndType       |= CU_CANDFULL    ;   	// Candidate data type - CU_CANDFULL this should be the default as it has all the needed data
  sSpec.cndType       |= CU_STR_ARR     ;   	// Candidate storage structure - CU_STR_ARR    is generally the fastest

  sSpec.retType       |= CU_POWERZ_S    ;   	// Return type
  sSpec.retType       |= CU_STR_ARR     ;   	// Candidate storage structure

  sSpec.fftInf.fft    = obs->fft;
  sSpec.fftInf.nor    = obs->numbins;
  sSpec.fftInf.rlo    = obs->rlo;
  sSpec.fftInf.rhi    = obs->rhi;

  sSpec.noHarmStages  = obs->numharmstages;
  sSpec.zMax          = obs->zhi;
  sSpec.sigma         = cmd->sigma;
  sSpec.pWidth        = cmd->width;

  readAccelDefalts(&sSpec);

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
    infoMSG(2,2,"Create the primary stack/kernel on each device\n");

    nvtxRangePush("Initialise Kernels");

    sSrch->pInf->kernels = (cuFFdotBatch*)malloc(sSrch->gSpec->noDevices*sizeof(cuFFdotBatch));

    int added;
    cuFFdotBatch* master = NULL;

    for ( int dev = 0 ; dev < sSrch->gSpec->noDevices; dev++ ) // Loop over devices  .
    {
      added = initKernel(&sSrch->pInf->kernels[sSrch->pInf->noDevices], master, sSrch, dev );

      if ( added && !master ) // This was the first batch so it is the master
      {
        master = &sSrch->pInf->kernels[0];
      }

      if ( added )
      {
        sSrch->pInf->noBatches += added;
        sSrch->pInf->noDevices++;
      }
      else
      {
        sSrch->gSpec->noDevBatches[dev] = 0;
        fprintf(stderr, "ERROR: failed to set up a kernel on device %i, trying to continue... \n", sSrch->gSpec->devId[dev]);
      }
    }

    nvtxRangePop();

    if ( sSrch->pInf->noDevices <= 0 ) // Check if we got any devices  .
    {
      fprintf(stderr, "ERROR: Failed to set up a kernel on any device. Try -lsgpu to see what devices there are.\n");
      exit (EXIT_FAILURE);
    }

  }

  FOLD // Create planes for calculations  .
  {
    infoMSG(2,2,"Create planes\n");

    nvtxRangePush("Initialise Batches");

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

        for ( int batch = 0 ; batch < sSrch->gSpec->noDevBatches[dev]; batch++ )
        {
          infoMSG(3,3,"Initialise batch %02i\n", bNo );

          noSteps = initBatch(&sSrch->pInf->batches[bNo], &sSrch->pInf->kernels[ker], batch, sSrch->gSpec->noDevBatches[dev]-1);

          if ( noSteps == 0 )
          {
            if ( batch == 0 )
            {
              fprintf(stderr, "ERROR: Failed to create at least one batch on device %i.\n", sSrch->pInf->kernels[dev].device);
            }
            break;
          }
          else
          {
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

    nvtxRangePop();
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
  sSrch->oInf = new cuOptInfo;
  memset(sSrch->oInf, 0, sizeof(cuOptInfo));

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering initOptimisers.");

  FOLD // Create the primary stack on each device, this contains the kernel  .
  {
    nvtxRangePush("Initialise Optimisers");

    sSrch->oInf->noOpts = 0;

    for ( int dev = 0 ; dev < sSrch->gSpec->noDevices; dev++ ) // Loop over devices  .
    {
      if ( sSrch->gSpec->noDevOpt[dev] > 0 )
      {
        sSrch->oInf->noOpts+=sSrch->gSpec->noDevOpt[dev];
      }
    }

    sSrch->oInf->opts = (cuOptCand*)malloc(sSrch->oInf->noOpts*sizeof(cuOptCand));
    memset(sSrch->oInf->opts, 0, sSrch->oInf->noOpts*sizeof(cuOptCand));

    int idx = 0;
    for ( int dev = 0 ; dev < sSrch->gSpec->noDevices; dev++ ) // Loop over devices  .
    {
      for ( int oo = 0 ; oo < sSrch->gSpec->noDevOpt[dev]; oo++ )
      {
        // Setup some basic info
        sSrch->oInf->opts[idx].pIdx     = idx;
        sSrch->oInf->opts[idx].device   = sSrch->gSpec->devId[dev];

        initOptCand(sSrch, &sSrch->oInf->opts[idx], dev );
        idx++;
      }
    }

    nvtxRangePop();
  }
}

void freeAccelGPUMem(cuPlnInfo* aInf)
{
  infoMSG(2,0,"FreeAccelGPUMem\n");

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
  //if ( srch->sSpec->flags & FLAG_THREAD )
  {
    resThrds* tInf = srch->threasdInfo;

    if ( !tInf )
    {
      tInf     = new resThrds;
      memset(tInf, 0, sizeof(cuSearch));
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
}

cuSearch* initSearchInf(searchSpecs* sSpec, gpuSpecs* gSpec, cuSearch* srch)
{
  infoMSG(2,1,"Initialise search data structure\n");

  bool same   = true;

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering initCuSearch.");

  if ( srch ) 	                    // Check if the search values have been pre-initialised  .
  {
    if ( srch->noHarmStages != sSpec->noHarmStages )
    {
      same = false;
      // ERROR recreate everything
    }

    if ( srch->pInf )
    {
      if ( srch->pInf->kernels->hInfos->zmax != sSpec->zMax )
      {
        same = false;
        // Have to recreate
      }

      presto_interp_acc accuracy = LOWACC;
      if ( sSpec->flags & FLAG_KER_HIGH )
        accuracy = HIGHACC;

      if ( srch->pInf->kernels->accelLen != optAccellen(sSpec->pWidth,sSpec->zMax, accuracy) )
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

  if ( !srch || same == false)      // Create a new search data structure  .
  {
    infoMSG(2,2,"Create a new search data structure\n");

    srch = new cuSearch;
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
    if ( sSpec->zMax % ACCEL_DZ )
      sSpec->zMax = (sSpec->zMax / ACCEL_DZ + 1) * ACCEL_DZ;

    int numz = (sSpec->zMax / ACCEL_DZ) * 2 + 1;

    FOLD // Calculate power cutoff and number of independent values  .
    {
      for (int ii = 0; ii < srch->noHarmStages; ii++)
      {
        if ( sSpec->zMax == 1 )
        {
          srch->numindep[ii]  = (sSpec->fftInf.rhi - sSpec->fftInf.rlo) / srch->noGenHarms;
        }
        else
        {
          srch->numindep[ii]  = (sSpec->fftInf.rhi - sSpec->fftInf.rlo) * (numz + 1) * ( ACCEL_DZ / 6.95) / (double)(1<<ii);
        }

        // Power cutoff
        // TODO: Check if using half precision may affect this
        srch->powerCut[ii]  = power_for_sigma(sSpec->sigma, (1<<ii), srch->numindep[ii]);
      }
    }
  }

  FOLD // Set up the CPU threading  .
  {
    infoMSG(3,2,"Set up the CPU threading\n");

    intSrchThrd(srch);
  }

  return srch;
}

cuSearch* initCuKernels(searchSpecs* sSpec, gpuSpecs* gSpec, cuSearch* srch)
{
  infoMSG(1,0,"Initialise CU search data structures\n");

  if ( !srch )
  {
    srch = initSearchInf(sSpec, gSpec, srch);
  }

  if ( !srch->pInf )
  {
    initPlanes( srch ); // This initialises the plane info
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
  int maxxx = ( srch->sSpec->fftInf.rhi - srch->sSpec->fftInf.rlo ) / (float)( master->accelLen * ACCEL_DR ) ; /// The number of planes we can work with

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

    setDevice(trdBatch->device) ;

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
          startrs[step] = srch->sSpec->fftInf.rlo   + (firstStep+step) * ( master->accelLen * ACCEL_DR );
          lastrs[step]  = startrs[step] + master->accelLen * ACCEL_DR - ACCEL_DR;
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

void writeLogEntry(char* fname, accelobs* obs, cuSearch* cuSrch, long long prepTime, long long cpuKerTime, long long cupTime, long long gpuKerTime, long long gpuTime, long long optTime, long long cpuOptTime, long long gpuOptTime)
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

  Logger* cvsLog = new Logger(fname, 1);
  cvsLog->sedCsvDeliminator('\t');

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
    cvsLog->csvWrite("GPU",       "#", "%2i",     batch->device);

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

    if ( batch->flags & CU_NORM_CPU )
      cvsLog->csvWrite("NORM",    "flg", "CPU");
    else
      cvsLog->csvWrite("NORM",    "flg", "GPU");

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

  FOLD // Advanced Timing  .
  {
    float copyH2DT  = 0;
    float InpNorm   = 0;
    float InpFFT    = 0;
    float multT     = 0;
    float InvFFT    = 0;
    float plnCpy    = 0;
    float ss        = 0;
    float resultT   = 0;
    float copyD2HT  = 0;

    if ( batch->flags & FLAG_TIME )
    {
      for (int batch = 0; batch < cuSrch->pInf->noBatches; batch++)
      {
        float l_copyH2DT  = 0;
        float l_InpNorm   = 0;
        float l_InpFFT    = 0;
        float l_multT     = 0;
        float l_InvFFT    = 0;
        float l_plnCpy    = 0;
        float l_ss        = 0;
        float l_resultT   = 0;
        float l_copyD2HT  = 0;

        for (int stack = 0; stack < cuSrch->pInf->batches[batch].noStacks; stack++)
        {
          cuFFdotBatch* batches = &cuSrch->pInf->batches[batch];
          l_copyH2DT  += batches->copyH2DTime[stack];
          l_InpNorm   += batches->normTime[stack];
          l_InpFFT    += batches->InpFFTTime[stack];
          l_multT     += batches->multTime[stack];
          l_InvFFT    += batches->InvFFTTime[stack];
          l_plnCpy    += batches->copyToPlnTime[stack];
          l_ss        += batches->searchTime[stack];
          l_resultT   += batches->resultTime[stack];
          l_copyD2HT  += batches->copyD2HTime[stack];
        }
        copyH2DT  += l_copyH2DT;
        InpNorm   += l_InpNorm;
        InpFFT    += l_InpFFT;
        multT     += l_multT;
        InvFFT    += l_InvFFT;
        plnCpy    += l_plnCpy;
        ss        += l_ss;
        resultT   += l_resultT;
        copyD2HT  += l_copyD2HT;
      }
    }
    cvsLog->csvWrite("copyH2D",     "ms", "%12.6f", copyH2DT);
    cvsLog->csvWrite("InpNorm",     "ms", "%12.6f", InpNorm);
    cvsLog->csvWrite("InpFFT",      "ms", "%12.6f", InpFFT);
    cvsLog->csvWrite("Mult",        "ms", "%12.6f", multT);
    cvsLog->csvWrite("InvFFT",      "ms", "%12.6f", InvFFT);
    cvsLog->csvWrite("plnCpy",      "ms", "%12.6f", plnCpy);
    cvsLog->csvWrite("Sum & Srch",  "ms", "%12.6f", ss);
    cvsLog->csvWrite("result",      "ms", "%12.6f", resultT);
    cvsLog->csvWrite("copyD2H",     "ms", "%12.6f", copyD2HT);
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

  cand* tempCand = new cand;
  container* next;
  container* close;
  container* serch;

  container* lst = tree->getLargest();

  while ( lst )
  {
    cand* candidate = (cand*)lst->data;

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
//        candidate->sig = candidate_sigma_cl(candidate->power, candidate->numharm,  batch->sInf->numindep[stg] );
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
//  if      ( batch->flag & FLAG_HALF )
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
//  if      ( batch->flag & FLAG_HALF )         // half output
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

    nvtxRangePush("Wait on CPU threads");

    while ( noTrd > 0 )
    {
      nvtxRangePush("Sleep");

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

      nvtxRangePop();
    }

    if (ite >= 10 )
      printf("\n\n");

    nvtxRangePop();

    return (ite);
  }

  return (0);
}
