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
  batch->rValues = batch->rArrays[rIdx];
}


/* The fft length needed to properly process a subharmonic */
int calc_fftlen3(double harm_fract, int max_zfull, uint accelLen)
{
  int bins_needed, end_effects;

  bins_needed = accelLen * harm_fract + 2;
  end_effects = 2 * ACCEL_NUMBETWEEN * z_resp_halfwidth(calc_required_z(harm_fract, max_zfull), LOWACC);
  return next2_to_n_cu(bins_needed + end_effects);
}

/** Calculate an optimal accellen given a width  .
 *
 * @param width the width of the plane usually a power of two
 * @param zmax
 * @return
 * If width is not a power of two it will be rounded up to the nearest power of two
 */
uint optAccellen(float width, int zmax)
{
  float halfwidth       = z_resp_halfwidth(zmax, LOWACC); /// The halfwidth of the maximum zmax, to calculate accel len
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
uint calcAccellen(float width, float zmax)
{
  int accelLen;

  if ( width > 100 )
  {
    accelLen = width;
  }
  else
  {
    accelLen = optAccellen(width*1000.0,zmax) ;
  }
  return accelLen;
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
int initKernel(cuFFdotBatch* kernel, cuFFdotBatch* master, cuSearch*   sInf, int device, int noBatches, int noSteps )
{
  nvtxRangePush("initKernel");
  std::cout.flush();

  size_t free, total;             /// GPU memory
  int noInStack[MAX_HARM_NO];
  int noHarms         = (1 << (sInf->noHarmStages - 1) );
  int major           = 0;
  int minor           = 0;
  noInStack[0]        = 0;
  size_t batchSize    = 0;        /// Total size (in bytes) of all the data need by a family (ie one step) excluding FFT temporary
  size_t fffTotSize   = 0;        /// Total size (in bytes) of FFT temporary memory
  size_t planeSize    = 0;        /// Total size (in bytes) of memory required independently of batch(es)
  int flags           = sInf->sSpec->flags;
  int alignment       = 0;
  float plnElsSZ      = 0;

  CUDA_SAFE_CALL(cudaGetLastError(), "Error entering initKernel.");

  FOLD // See if we can use the cuda device  and whether it may be possible to do GPU in-mem search .
  {
    nvtxRangePush("Get Device");

    if ( device >= getGPUCount() )
    {
      fprintf(stderr, "ERROR: There is no CUDA device %i.\n",device);
      return 0;
    }
    int currentDevvice;
    CUDA_SAFE_CALL(cudaSetDevice(device), "ERROR: cudaSetDevice");
    CUDA_SAFE_CALL(cudaGetDevice(&currentDevvice), "Failed to get device using cudaGetDevice");
    if (currentDevvice != device)
    {
      fprintf(stderr, "ERROR: CUDA Device not set.\n");
      return 0;
    }
    else
    {
      cudaDeviceProp deviceProp;
      CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, device), "Failed to get device properties device using cudaGetDeviceProperties");
      CUDA_SAFE_CALL(cudaMemGetInfo ( &free, &total ), "Getting Device memory information");

      major                           = deviceProp.major;
      minor                           = deviceProp.minor;
      float ver                       = major + minor/10.0f;
      kernel->capability              = ver;

      alignment                       = getMemAlignment();

      sInf->mInf->alignment[device]   = alignment;
      sInf->mInf->capability[device]  = ver;
      sInf->mInf->name[device]        = (char*)malloc(256*sizeof(char));
      sprintf(sInf->mInf->name[device], "%s", deviceProp.name );
    }

    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    nvtxRangePop();
  }

  FOLD // Now see if this device could do a GPU in-mem search  .
  {
    if ( master == NULL ) // For the moment lets try this on only the first card!
    {
      double plnX       = ( sInf->sSpec->fftInf.rhi - sInf->sSpec->fftInf.rlo/(double)noHarms ) / (double)( ACCEL_DR ) ; // The number of bins
      int    plnY       = calc_required_z(1.0, (float)sInf->sSpec->zMax );

      if ( flags & FLAG_HALF )
      {
#if __CUDACC_VER__ >= 70500
        plnElsSZ = sizeof(half);
#else
        plnElsSZ = sizeof(float);
        fprintf(stderr, "WARNING: Half precision can only be used with CUDA 7.5 or later! Reverting to single precision!\n");
        flags &= ~FLAG_HALF;
#endif
      }
      else
      {
        plnElsSZ = sizeof(float);
      }

      double totalSize  = plnX * plnY * plnElsSZ ;
      double appRoxWrk  = plnY * INMEM_FFT_WIDTH * ( 4 * 3 + 1) ; // 4 planes * ( input + CUFFT )

      if ( totalSize + appRoxWrk < free )
      {
        if ( !(flags & FLAG_SS_ALL) || (flags & FLAG_SS_INMEM) )
        {
          printf("Device %i can do a in-mem GPU search.\n", device);
          printf("  There is %.2fGB free memory.\n  The entire plane requires %.2f GB and the workspace ~%.2f MB.\n\n", free*1e-9, totalSize*1e-9, appRoxWrk*1e-6 );
        }

        if ( (flags & FLAG_SS_ALL) && !(flags & FLAG_SS_INMEM) )
        {
          fprintf(stderr,"WARNING: Opting to NOT do a in-mem search when we could!\n");
        }
        else
        {
          noHarms               = 1;
          sInf->sSpec->pWidth   = INMEM_FFT_WIDTH / 1000.0 ;

          if ( sInf->gSpec->noDevices > 1 )
          {
            fprintf(stderr,"Warning: Reverting to single device search.\n");
            sInf->gSpec->noDevices = 1;
          }

          flags |= FLAG_SS_INMEM ;

#if CUDA_VERSION >= 6050
          flags |= FLAG_CUFFT_CB_OUT;
#else
          fprintf(stderr,"Warning: Doing an in-mem search with no CUFFT callbacks, this is not ideal. Try upgrading to CUDA 6.5 or later.\n");
          flags &= ~FLAG_CUFFT_CB_OUT;
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
        if ( flags & FLAG_SS_INMEM  )
        {
          fprintf(stderr,"ERROR: Requested an in-memory GPU search, this is not possible\n\tThere is %.2f GB of free memory.\n\tIn-mem GPU search would require ~%.2f GB\n\n", free*1e-9, (totalSize + appRoxWrk)*1e-9 );
        }
        flags &= ~FLAG_SS_INMEM ;
      }
    }
  }

  FOLD // Allocate and zero some structures  .
  {
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
        kernel->flag        = flags;
        kernel->srchMaster  = 1;
      }
    }

    FOLD // Set the device specific parameters  .
    {
      kernel->sInf          = sInf;
      kernel->device        = device;
      kernel->isKernel      = 1;              // This is the device master
    }

    FOLD // Allocate memory  .
    {
      kernel->hInfos        = (cuHarmInfo*) malloc(noHarms * sizeof(cuHarmInfo));
      kernel->kernels       = (cuKernel*)   malloc(noHarms * sizeof(cuKernel));

      // Zero memory for kernels and harmonics
      memset(kernel->hInfos,  0, noHarms * sizeof(cuHarmInfo));
      memset(kernel->kernels, 0, noHarms * sizeof(cuKernel));
    }
  }

  FOLD // First determine how many stacks and how many planes in each stack  .
  {
    if ( master == NULL ) 	// Calculate details for the batch  .
    {
      FOLD // Determine accellen and step size  .
      {
        printf("Determining GPU step size and plane width:\n");

        Fout // TMP!
        {
          int   zmax0 = calc_required_z(1, sInf->sSpec->zMax);

          int   sSz = (noHarms*ACCEL_RDR);
          int   oAccelLen1  = calcAccellen(sInf->sSpec->pWidth,     sInf->sSpec->zMax);
          int   aLen = floor( oAccelLen1/(float)(sSz) ) * (sSz);

          int i = 0;

          //for ( int i = 0; i < 100; i++ )
          while ( aLen - i * sSz > 0 )
          {
            int waist = 0;
            int good = 0;
            int waist2 = 0;
            int waist3 = 0;

            int aLength = aLen - i * sSz;
            int wsth;

            for (int h = noHarms; h > 0; h--)
            {
              float harmFrac    = (h) / (float)noHarms;
              int   zmax        = calc_required_z(harmFrac, sInf->sSpec->zMax);
              int   height      = (zmax / ACCEL_DZ) * 2 + 1;
              int   end_effects = 2 * ACCEL_NUMBETWEEN * z_resp_halfwidth(calc_required_z(harmFrac, zmax0), LOWACC);
              int   pWidth      = harmFrac * aLength + 2 + end_effects ;
              int   sWidth      = calc_fftlen3(harmFrac, zmax0, aLength);

              waist += height * (sWidth - pWidth );
              good  += height * ( pWidth );

              if ( h == noHarms)
                wsth = sWidth;

              if ( wsth == sWidth )
              {
                waist3 += height * (sWidth - pWidth );
              }

              pWidth      = harmFrac * aLength + 2 ;
              waist2 += height * (sWidth - pWidth );
            }

            printf("%i\t%i\t%i\t%i\t%i\t%.4f\n", i, wsth, aLength, waist, good, good/(float)(waist+good));

            i++;
          }

          printf("\n\n");
        }

        if ( noHarms > 1 )
        {
          int   oAccelLen1, oAccelLen2;

          // This adjustment makes sure no more than half the harmonics are in the largest stack (reduce waisted work - gives a 0.01 - 0.12 speed increase )
          oAccelLen1  = calcAccellen(sInf->sSpec->pWidth,     sInf->sSpec->zMax);
          oAccelLen2  = calcAccellen(sInf->sSpec->pWidth/2.0, sInf->sSpec->zMax/2.0);

          if ( sInf->sSpec->pWidth > 100 )
          {
            kernel->accelLen  = oAccelLen1;
          }
          else
          {
            kernel->accelLen  = MIN(oAccelLen2*2, oAccelLen1);
          }

          if ( sInf->sSpec->pWidth < 100 ) // Check  .
          {
            float fWidth    = floor(calc_fftlen3(1, sInf->sSpec->zMax, kernel->accelLen)/1000.0);

            float ss        = calc_fftlen3(1, sInf->sSpec->zMax, kernel->accelLen) ;
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
          kernel->accelLen = calcAccellen(sInf->sSpec->pWidth, sInf->sSpec->zMax);
        }

        // Now make sure that accelLen is divisible by (noHarms*ACCEL_RDR) this "rule" is used for indexing in the sum and search kernel
        //if ( kernel->flag & (FLAG_SS_00 | FLAG_SS_10 | FLAG_SS_20 | FLAG_SS_30) )
        {
          kernel->accelLen = floor( kernel->accelLen/(float)(noHarms*ACCEL_RDR) ) * (noHarms*ACCEL_RDR);

          if ( sInf->sSpec->pWidth > 100 ) // Check  .
          {
            if ( sInf->sSpec->pWidth != kernel->accelLen )
            {
              fprintf(stderr,"ERROR: Using manual step size, value must be divisible by numharm * %i (%i) try %i.\n", ACCEL_RDR, noHarms*ACCEL_RDR, kernel->accelLen );
              exit(EXIT_FAILURE);
            }
          }
        }

        if ( kernel->accelLen > 100 )
        {
          float fftLen      = calc_fftlen3(1, sInf->sSpec->zMax, kernel->accelLen);
          int   oAccelLen   = optAccellen(fftLen, sInf->sSpec->zMax);
          float ratio       = kernel->accelLen/float(oAccelLen);

          printf(" • Using max FFT length of %.0f and thus", fftLen);

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
        int halfWidth       = 0;

        for (int i = noHarms; i > 0; i--)
        {
          int idx                         = noHarms-i;
          kernel->hInfos[idx].harmFrac    = (i) / (float)noHarms;
          kernel->hInfos[idx].zmax        = calc_required_z(kernel->hInfos[idx].harmFrac, sInf->sSpec->zMax);
          kernel->hInfos[idx].height      = (kernel->hInfos[idx].zmax / ACCEL_DZ) * 2 + 1;
          kernel->hInfos[idx].width       = calc_fftlen3(kernel->hInfos[idx].harmFrac, kernel->hInfos[0].zmax, kernel->accelLen);
          kernel->hInfos[idx].stackNo     = noStacks;

          if ( prevWidth != kernel->hInfos[idx].width )
          {
            noStacks++;
            noInStack[noStacks - 1]       = 0;
            prevWidth                     = kernel->hInfos[idx].width;
            halfWidth                     = z_resp_halfwidth(kernel->hInfos[idx].zmax, LOWACC);

            // Maximise and align halfwidth
            int sWidth                    = (int) ( ceil(kernel->accelLen * kernel->hInfos[idx].harmFrac * ACCEL_DR ) * ACCEL_RDR + DBLCORRECT ) + 1 ;
            float hw                      = (kernel->hInfos[idx].width  - sWidth)/2.0/(float)ACCEL_NUMBETWEEN;
            float noAlg                   = alignment / float(sizeof(fcomplex)) / (float)ACCEL_NUMBETWEEN ;     // halfWidth will be multiplied by ACCEL_NUMBETWEEN so can divide by it here!
            float hw4                     = floor(hw/noAlg) * noAlg ;

            if ( halfWidth > hw4)
              halfWidth                   = floor(hw);
            else
              halfWidth                   = hw4;
          }

          if ( kernel->flag & FLAG_KER_ACC )
          {
            kernel->hInfos[idx].halfWidth = halfWidth; // Use maximum halfwidth for all planes in a stack this gives higher accuracy at small Z at no extra cost!
          }
          else
          {
            kernel->hInfos[idx].halfWidth = z_resp_halfwidth(kernel->hInfos[idx].zmax, LOWACC);
          }

          noInStack[noStacks - 1]++;
        }

        kernel->noHarms                   = noHarms;
        kernel->noHarmStages              = log2((float)noHarms)+1;
        kernel->noStacks                  = noStacks;
      }
    }
    else                    // Copy details from the master batch  .
    {
      // Copy memory from kernels and harmonics
      memcpy(kernel->hInfos,  master->hInfos,  noHarms * sizeof(cuHarmInfo));
      memcpy(kernel->kernels, master->kernels, noHarms * sizeof(cuKernel));
    }
  }

  FOLD // Allocate all the memory for the stack data structures  .
  {
    long long neede = kernel->noStacks * sizeof(cuFfdotStack) + noHarms * sizeof(cuHarmInfo) + noHarms * sizeof(cuKernel);

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
        memcpy(kernel->stacks, master->stacks, kernel->noStacks * sizeof(cuFfdotStack));
    }
  }

  FOLD // Set up the basic details of all the harmonics and calculate the stride  .
  {
    if ( master == NULL )
    {
      FOLD // Set up the basic details of all the harmonics  .
      {
        // Calculate the stage order of the harmonics
        int harmtosum;
        int i = 0;

        for ( int stage = 0; stage < kernel->noHarmStages; stage++ )
        {
          harmtosum = 1 << stage;
          for (int harm = 1; harm <= harmtosum; harm += 2, i++)
          {
            float harmFrac                  = harm/float(harmtosum);
            int idx                         = round(harmFrac*noHarms);
            if ( harmFrac == 1 )
              idx = 0;

            kernel->hInfos[idx].stageIndex  = i;
            kernel->stageIdx[i]             = idx;
          }
        }
      }

      FOLD // Set up the basic details of all the harmonics  .
      {
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
          cStack->flag          = kernel->flag;               // Used to create the kernel, will be over written later

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
          cStack->strideFloat =   getStrie(cStack->width, sizeof(float),        alignment);

          kernel->inpDataSize +=  cStack->strideCmplx * cStack->noInStack * sizeof(cufftComplex);
          kernel->kerDataSize +=  cStack->strideCmplx * cStack->kerHeigth * sizeof(cufftComplex);
          kernel->plnDataSize +=  cStack->strideCmplx * cStack->height    * sizeof(cufftComplex);

          if ( kernel->flag & FLAG_CUFFT_CB_OUT )
            kernel->pwrDataSize +=  cStack->strideFloat * cStack->height; // Powers only used with CUFFT callbacks
        }
      }
    }
  }

  FOLD // Allocate device memory for all the kernels data  .
  {
    nvtxRangePush("kernel malloc");

    if ( kernel->kerDataSize > free )
    {
      fprintf(stderr, "ERROR: Not enough device memory for GPU multiplication kernels. There is only %.2f MB free and you need %.2f MB \n", free / 1048576.0, kernel->kerDataSize / 1048576.0 );
      freeKernel(kernel);
      return 0;
    }
    else
    {
      CUDA_SAFE_CALL(cudaMalloc((void**)&kernel->d_kerData, kernel->kerDataSize), "Failed to allocate device memory for kernel stack.");
      CUDA_SAFE_CALL(cudaGetLastError(), "CUDA Error allocation of device memory for kernel?.\n");
    }

    nvtxRangePop();
  }

  FOLD // Set the sizes values of the harmonics and kernels and pointers to kernel data  .
  {
    size_t kerSiz = 0;

    for (int i = 0; i < kernel->noStacks; i++)
    {
      cuFfdotStack* cStack            = &kernel->stacks[i];
      cStack->d_kerData               = &kernel->d_kerData[kerSiz];

      // Set the stride
      for (int j = 0; j< cStack->noInStack; j++)
      {
        cStack->harmInf[j].inpStride  = cStack->strideCmplx;

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
    if ( master == NULL )  	  // Create the kernels  .
    {
      nvtxRangePush("Initialise kernels");

      // Run message
      CUDA_SAFE_CALL(cudaGetLastError(), "Error before creating GPU kernels");

      float contamination = (kernel->hInfos->halfWidth*2*ACCEL_NUMBETWEEN)/(float)kernel->hInfos->width*100 ;

      if ( contamination > 25 )
      {
        fprintf(stderr, "WARNING: Contamination is high, consider increasing width with the -width flag.\n");
      }

      printf("\nGenerating GPU multiplication kernels using device %02i\n", device);

      int hh      = 1;
      for (int i = 0; i < kernel->noStacks; i++)
      {
        cuFfdotStack* cStack = &kernel->stacks[i];

        printf("  ■ Stack %i has %02i f-∂f plane(s). width: %5li  stride: %5li  Height: %6li  Memory size: %7.1f MB \n", i+1, cStack->noInStack, cStack->width, cStack->strideCmplx, cStack->height, cStack->height*cStack->strideCmplx*sizeof(fcomplex)/1024.0/1024.0);

        // Call the CUDA kernels
        // Only need one kernel per stack

        createStackKernel(cStack);
        printf("    ► Created kernel %i  Size: %7.1f MB  Height %4i   Contamination: %5.2f %% \n", i+1, cStack->harmInf->height*cStack->strideCmplx*sizeof(fcomplex)/1024.0/1024.0, cStack->harmInf->zmax, (cStack->harmInf->halfWidth*2*ACCEL_NUMBETWEEN)/(float)cStack->width*100);

        for (int j = 0; j < cStack->noInStack; j++)
        {
          printf("      • Harmonic %02i  Fraction: %5.3f   Z-Max: %4i   Half Width: %4i \n", hh, cStack->harmInf[j].harmFrac, cStack->harmInf[j].zmax, cStack->harmInf[j].halfWidth );
          hh++;
        }
      }

      FOLD // FFT the kernels  .
      {
        cufftHandle plnPlan;

        fflush(stdout);
        printf("  FFT'ing the kernels ");
        fflush(stdout);

        for (int i = 0; i < kernel->noStacks; i++)
        {
          cuFfdotStack* cStack = &kernel->stacks[i];

          FOLD // Create the plan  .
          {
            size_t fftSize      = 0;

            int n[]             = {cStack->width};
            int inembed[]       = {cStack->strideCmplx* sizeof(fcomplexcu)};
            int istride         = 1;
            int idist           = cStack->strideCmplx;
            int onembed[]       = {cStack->strideCmplx* sizeof(fcomplexcu)};
            int ostride         = 1;
            int odist           = cStack->strideCmplx;
            int height          = cStack->kerHeigth;

            CUFFT_SAFE_CALL(cufftCreate(&plnPlan),"Creating plan for complex data of stack. [cufftCreate]");
            CUFFT_SAFE_CALL(cufftMakePlanMany(plnPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, height,    &fftSize), "Creating plan for complex data of stack. [cufftMakePlanMany]");
            CUDA_SAFE_CALL(cudaGetLastError(), "Creating FFT plans for the stacks.");

            fffTotSize          += fftSize;
          }

          FOLD // Call the plan  .
          {
            CUFFT_SAFE_CALL(cufftExecC2C(plnPlan, (cufftComplex *) cStack->d_kerData, (cufftComplex *) cStack->d_kerData, CUFFT_FORWARD), "FFT'ing the kernel data. [cufftExecC2C]");
            CUDA_SAFE_CALL(cudaGetLastError(), "FFT'ing the multiplication kernels.");
          }

          FOLD // Destroy the plan  .
          {
            CUFFT_SAFE_CALL(cufftDestroy(plnPlan), "Destroying plan for complex data of stack. [cufftDestroy]");
            CUDA_SAFE_CALL(cudaGetLastError(), "Destroying the plan.");
          }

          printf("•");
          fflush(stdout);
        }

        CUDA_SAFE_CALL(cudaGetLastError(), "FFT'ing the multiplication kernels.");

        printf("\n");
      }

      printf("Done generating GPU multiplication kernels\n");

      nvtxRangePop();
    }
  }

  FOLD // Decide how to handle input and output and allocate required memory  .
  {
    nvtxRangePush("data");

    printf("\nInitializing GPU %i (%s)\n", device, sInf->mInf->name[device] );

    if ( master != NULL )     // Create the kernels  .
    {
      printf("• Copying multiplication kernels from device %i.\n", master->device);
      CUDA_SAFE_CALL(cudaMemcpyPeer(kernel->d_kerData, kernel->device, master->d_kerData, master->device, master->kerDataSize ), "Copying multiplication kernels between devices.");
    }

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
        double noHarmsReal    = (1<<(sInf->sSpec->noHarmStages-1));

        int minR              = floor ( sInf->sSpec->fftInf.rlo / noHarmsReal - kernel->hInfos->halfWidth );
        int maxR              = ceil  ( sInf->sSpec->fftInf.rhi  + kernel->hInfos->halfWidth );

        searchScale* SrchSz   = new searchScale;
        kernel->SrchSz        = SrchSz;
        sInf->SrchSz          = SrchSz;
        memset(SrchSz, 0, sizeof(searchScale));

        SrchSz->searchRLow    = sInf->sSpec->fftInf.rlo / noHarmsReal;
        SrchSz->searchRHigh   = sInf->sSpec->fftInf.rhi;
        SrchSz->rLow          = minR;
        SrchSz->rHigh         = maxR;
        SrchSz->noInpR        = maxR - minR  ;  /// The number of input data points

        SrchSz->noSteps       = ( sInf->sSpec->fftInf.rhi - sInf->sSpec->fftInf.rlo ) / (float)( kernel->accelLen * ACCEL_DR ) ; // The number of planes to make

        if ( kernel->flag & FLAG_SS_INMEM   )
        {
          SrchSz->noSteps     = ( SrchSz->searchRHigh - SrchSz->searchRLow ) / (float)( kernel->accelLen * ACCEL_DR ) ; // The number of planes to make
        }

        if ( kernel->flag  & FLAG_STORE_EXP )
        {
          SrchSz->noOutpR     = ceil( (SrchSz->searchRHigh - SrchSz->searchRLow)/ACCEL_DR );
        }
        else
        {
          SrchSz->noOutpR     = ceil(SrchSz->searchRHigh - SrchSz->searchRLow);
        }

        if ( (kernel->flag & FLAG_STORE_ALL) && !( kernel->flag  & FLAG_RET_STAGES) )
        {
          printf("   Storing all results implies returning all results so adding FLAG_RET_STAGES to flags!\n");
          kernel->flag  |= FLAG_RET_STAGES;
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
        if (  (kernel->flag & FLAG_CUFFT_CB_OUT) && ( !( (kernel->retType & CU_HALF) || (kernel->retType & CU_FLOAT)))   )
        {
          fprintf(stderr,"WARNING: Returning plane and CUFFT output requires float return type.\n");
          kernel->retType &= ~CU_TYPE_ALLL;
          kernel->retType |= CU_FLOAT;
        }

        if ( !(kernel->flag & FLAG_CUFFT_CB_OUT) && !(kernel->retType & CU_CMPLXF) )
        {
          fprintf(stderr,"WARNING: Returning plane requires complex float return type.\n");
          kernel->retType &= ~CU_TYPE_ALLL;
          kernel->retType |= CU_CMPLXF;
        }

        if ( kernel->flag & FLAG_SIG_GPU )
        {
          fprintf(stderr,"WARNING: Cannot do GPU sigma calculations when returning plane data.\n");
          kernel->flag &= ~FLAG_SIG_GPU;
        }
      }

      if      (kernel->retType & CU_CMPLXF   	)
      {
        retSZ = sizeof(fcomplexcu);
      }
      else if (kernel->retType & CU_INT      	)
      {
        retSZ = sizeof(int);
      }
      else if (kernel->retType & CU_HALF      )
      {
#if __CUDACC_VER__ >= 70500
        retSZ = sizeof(half);
#else
        fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
        exit(EXIT_FAILURE);
#endif
      }
      else if (kernel->retType & CU_FLOAT    	)
      {
        retSZ = sizeof(float);
      }
      else if (kernel->retType & CU_DOUBLE    )
      {
        retSZ = sizeof(double);
      }
      else if (kernel->retType & CU_POWERZ_S 	)
      {
        retSZ = sizeof(candPZs);
      }
      else if (kernel->retType & CU_POWERZ_I 	)
      {
        retSZ = sizeof(candPZi);
      }
      else if (kernel->retType & CU_CANDMIN  	)
      {
        retSZ = sizeof(candMin);
      }
      else if (kernel->retType & CU_CANDSMAL 	)
      {
        retSZ = sizeof(candSml);
      }
      else if (kernel->retType & CU_CANDBASC 	)
      {
        retSZ = sizeof(accelcandBasic);
      }
      else if (kernel->retType & CU_CANDFULL 	)
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
        if      ( kernel->flag & FLAG_SS_INMEM )
        {
          //kernel->strideRes = 4096;
          kernel->strideRes = 8192;
          //kernel->strideRes = 16384;
          //kernel->strideRes = 32768;
        }
        else if ( (kernel->retType & CU_STR_ARR) )
        {
          kernel->strideRes = kernel->hInfos->width;  // NOTE: This could be accellen rather than width, but to allow greater flexibility keep it at width. CU_STR_PLN    requires width
        }
        else if (  kernel->retType & CU_STR_PLN  )
        {
          if      ( kernel->retType & CU_FLOAT  )
          {
            kernel->strideRes = kernel->stacks->strideFloat ;
          }
          else if ( kernel->retType & CU_HALF   )
          {
            kernel->strideRes = kernel->stacks->strideFloat ;
          }
          else if ( kernel->retType & CU_CMPLXF )
          {
            kernel->strideRes = kernel->stacks->strideCmplx ;
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
      kernel->retDataSize   = retY*kernel->strideRes*retSZ;

      if ( kernel->flag & FLAG_RET_STAGES )
        kernel->retDataSize *= sInf->noHarmStages;
    }

    FOLD // Calculate batch size and number of steps and batches on this device  .
    {
      CUDA_SAFE_CALL(cudaMemGetInfo ( &free, &total ), "Getting Device memory information"); // TODO: this call may not be necessary we could calculate this from previous values
      freeRam = getFreeRamCU();

      printf("   There is a total of %.2f GiB of device memory of which there is %.2f GiB free and %.2f GiB free host memory.\n",total / 1073741824.0, (free )  / 1073741824.0, freeRam / 1073741824.0 );

      FOLD // Calculate size of various memory's'  .
      {
        FOLD // Determine the size of the powers plane  .
        {
          if ( kernel->flag & FLAG_HALF )
          {
#if __CUDACC_VER__ >= 70500
            kernel->pwrDataSize *= sizeof(half);
#else
            kernel->pwrDataSize *= sizeof(float);
            fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
#endif
          }
          else
          {
            kernel->pwrDataSize *= sizeof(float);
          }
        }

        batchSize             = kernel->inpDataSize + kernel->plnDataSize + kernel->pwrDataSize + kernel->retDataSize;  // This is currently the size of one step
        fffTotSize            = kernel->inpDataSize + kernel->plnDataSize;                                              // FFT data treated separately cos there will be only one set per device

        if ( flags & FLAG_SS_INMEM  ) // Size of memory for plane  full ff plane.
        {
          uint noStepsP       =  ceil(kernel->SrchSz->noSteps / (float)noSteps) * noSteps;
          uint nX             = noStepsP * kernel->accelLen;
          uint nY             = kernel->hInfos->height;
          planeSize          += nX * nY * plnElsSZ ;
        }

        if ( !(flags & FLAG_CUFFT_CB_OUT) )
        {
          // Need a second plane
          batchSize        += kernel->plnDataSize;
        }
      }

      FOLD // Calculate how many batches and steps to do  .
      {
        float possSteps = ( free - planeSize ) / (double) ( fffTotSize + batchSize * noBatches ) ;  // (fffTotSize * possSteps) for the CUFFT memory for FFT'ing the plane(s) and (totSize * noThreads * possSteps) for each thread(s) plan(s)

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

          printf("        Throttling to %i steps in 1 batch.\n", noSteps);
        }

        if ( noBatches <= 0 || noSteps <= 0 )
        {
          fprintf(stderr, "ERROR: Insufficient memory to make make any planes on this device. One step would require %.2fGiB of device memory.\n", ( fffTotSize + batchSize )/1073741824.0 );

          // TODO: check flags here!

          freeKernel(kernel);
          return 0;
        }
        float  totUsed = ( kernel->kerDataSize + planeSize + ( fffTotSize + batchSize * noBatches ) * kernel->noSteps ) ;

        printf("     Processing %i steps with each of the %i batch(s)\n", noSteps, noBatches );

        printf("    -----------------------------------------------\n" );
        printf("    Kernels      use: %5.2f GiB of device memory.\n", (kernel->kerDataSize) / 1073741824.0 );
        printf("    CUFFT       uses: %5.2f GiB of device memory.\n", (fffTotSize*kernel->noSteps) / 1073741824.0 );
        if ( planeSize )
        {
          printf("    Plane       uses: %5.2f GiB of device memory.", (planeSize) / 1073741824.0 );

          if ( kernel->flag & FLAG_HALF )
          {
            printf(" (using half precision)\n");
          }
          else
          {
            printf("\n");
          }
        }
        printf("    Each batch  uses: %5.2f GiB of device memory.\n", (batchSize*kernel->noSteps) / 1073741824.0 );
        printf("               Using: %5.2f GiB of %.2f [%.2f%%] of GPU memory for search.\n",  totUsed / 1073741824.0, total / 1073741824.0, totUsed / (float)total * 100.0f );
      }
    }

    FOLD // Scale data sizes by number of steps  .
    {
      kernel->inpDataSize *= kernel->noSteps;
      kernel->plnDataSize *= kernel->noSteps;
      kernel->pwrDataSize *= kernel->noSteps;
      if ( !(flags & FLAG_SS_INMEM)  )
        kernel->retDataSize *= kernel->noSteps;       // In-meme search stage does not use steps
    }

    float fullCSize     = kernel->SrchSz->noOutpR * candSZ;               /// The full size of all candidate data

    if ( kernel->flag  & FLAG_STORE_ALL )
      fullCSize *= kernel->noHarmStages; // Store  candidates for all stages

    FOLD // DO a sanity check on flags  .
    {
      FOLD // How to handle input  .
      {
        if ( (kernel->flag & CU_INPT_FFT_CPU) && !(kernel->flag & CU_NORM_CPU) )
        {
          fprintf(stderr, "WARNING: Using CPU FFT of the input data necessitate doing the normalisation on CPU.\n");
          kernel->flag |= CU_NORM_CPU;
        }
      }

      FOLD // Set the stack flags  .
      {
        for (int i = 0; i < kernel->noStacks; i++)
        {
          cuFfdotStack* cStack  = &kernel->stacks[i];
          cStack->flag          = kernel->flag;
        }
      }
    }

    FOLD // Batch independent device memory  .
    {
      if ( flags & FLAG_SS_INMEM  )
      {
        uint noStepsP =  ceil(kernel->SrchSz->noSteps / (float)noSteps) * kernel->noSteps ;
        uint nX       = noStepsP * kernel->accelLen;
        uint nY       = kernel->hInfos->height;
        size_t stride;

        CUDA_SAFE_CALL(cudaMallocPitch(&kernel->d_planeFull,    &stride, plnElsSZ*nX, nY),   "Failed to allocate device memory for getMemAlignment.");
        kernel->sInf->mInf->inmemStride = stride / plnElsSZ;
        CUDA_SAFE_CALL(cudaMemsetAsync(kernel->d_planeFull, 0, stride*nY, 0),"Failed to initiate plane memory to zero");
      }
    }

    FOLD // Allocate global (device independent) host memory  .
    {
      // One set of global set of "candidates" for all devices
      if ( master == NULL )
      {
        if      ( kernel->cndType & CU_STR_ARR  )
        {
          if ( sInf->sSpec->outData == NULL 	)
          {
            // Have to allocate the array!

            freeRam  = getFreeRamCU();
            if ( fullCSize < freeRam*0.90 )
            {
              // Same host candidates for all devices
              // This can use a lot of memory for long searches!
              kernel->h_candidates = malloc( fullCSize );
              memset(kernel->h_candidates, 0, fullCSize );
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
            // TODO: this needs to be freed at some point

            // This memory has already been allocated
            kernel->h_candidates = sInf->sSpec->outData;
            memset(kernel->h_candidates, 0, fullCSize ); // NOTE: this may error if the preallocated memory int karge enough!
          }
        }
        else if ( kernel->cndType & CU_STR_QUAD )
        {
          remove( "/home/chris/src.cvs" ); // TMP

          if ( sInf->sSpec->outData == NULL )
          {
            candTree* qt = new candTree;
            kernel->h_candidates = qt;
          }
          else
          {
            kernel->h_candidates = sInf->sSpec->outData;
          }
        }
        else if ( kernel->cndType & CU_STR_LST  )
        {
          // Nothing really to do here =/
        }
        else if ( kernel->cndType & CU_STR_PLN  )
        {
          fprintf(stderr,"WARNING: The case of candidate planes has not been implemented!\n");

          // This memory has already been allocated
          kernel->h_candidates = sInf->sSpec->outData;
        }
      }
    }

    if ( hostC )
    {
      printf("    Input and candidates use and additional:\n");
      if ( hostC )
        printf("                      %5.2f GiB of host   memory\n", hostC / 1073741824.0 );
    }
    printf("    -----------------------------------------------\n" );

    CUDA_SAFE_CALL(cudaGetLastError(), "Failed to create memory for candidate list or input data.");

    printf("  Done\n");

    nvtxRangePop();
  }

  FOLD // Stack specific events  .
  {
    nvtxRangePush("events");

    if ( noBatches > 1 )
    {
      char strBuff[1024];

      if ( !(kernel->flag & CU_INPT_FFT_CPU) )
      {
        for (int i = 0; i < kernel->noStacks; i++)
        {
          cuFfdotStack* cStack = &kernel->stacks[i];
          CUDA_SAFE_CALL(cudaStreamCreate(&cStack->fftIStream),"Creating CUDA stream for fft's");
          sprintf(strBuff,"%i.0.2.%i FFT Input", device, i);
          nvtxNameCudaStreamA(cStack->fftIStream, strBuff);
          //printf("cudaStreamCreate: %s\n", strBuff);
        }
      }

      for (int i = 0; i < kernel->noStacks; i++)
      {
        cuFfdotStack* cStack = &kernel->stacks[i];
        CUDA_SAFE_CALL(cudaStreamCreate(&cStack->fftPStream),"Creating CUDA stream for fft's");
        sprintf(strBuff,"%i.0.4.%i FFT Plane", device, i);
        nvtxNameCudaStreamA(cStack->fftPStream, strBuff);
        //printf("cudaStreamCreate: %s\n", strBuff);
      }
    }

    nvtxRangePop();
  }

  FOLD // Create texture memory from kernels  .
  {
    nvtxRangePush("text mem");

    if ( kernel->flag & FLAG_TEX_MUL )
    {
      cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);

      CUDA_SAFE_CALL(cudaGetLastError(), "CUDA Error creating texture from kernel data.");

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

        CUDA_SAFE_CALL(cudaCreateTextureObject(&cStack->kerDatTex, &resDesc, &texDesc, NULL), "Error Creating texture from kernel data.");

        CUDA_SAFE_CALL(cudaGetLastError(), "CUDA Error creating texture from the stack of kernel data.");

        // Create the actual texture object
        for (int j = 0; j< cStack->noInStack; j++)        // Loop through planes in stack
        {
          cuKernel* cKer = &cStack->kernels[j];

          resDesc.res.pitch2D.devPtr        = cKer->d_kerData;
          resDesc.res.pitch2D.height        = cKer->harmInf->height;
          resDesc.res.pitch2D.width         = cKer->harmInf->width;
          resDesc.res.pitch2D.pitchInBytes  = cStack->strideCmplx * sizeof(fcomplex);

          CUDA_SAFE_CALL(cudaCreateTextureObject(&cKer->kerDatTex, &resDesc, &texDesc, NULL), "Error Creating texture from kernel data.");
          CUDA_SAFE_CALL(cudaGetLastError(), "CUDA Error creating texture from kernel data.");
        }
      }
    }

    nvtxRangePop();
  }

  FOLD // Set constant memory values  .
  {
    nvtxRangePush("const mem");

    setConstVals( kernel,  sInf->noHarmStages, sInf->powerCut, sInf->numindep );
    setConstVals_Fam_Order( kernel );                            // Constant values for multiply

#if CUDA_VERSION >= 6050        // CUFFT callbacks only implimented in CUDA 6.5
    copyCUFFT_LD_CB(kernel);
#endif

    nvtxRangePop();
  }

  FOLD // Create FFT plans, ( 1 - set per device )  .
  {
    nvtxRangePush("FFT plans");

    if ( ( kernel->flag & CU_INPT_FFT_CPU ) && master == NULL)
    {
      read_wisdom();
    }

    fffTotSize = 0;
    for (int i = 0; i < kernel->noStacks; i++)
    {
      cuFfdotStack* cStack  = &kernel->stacks[i];
      size_t fftSize        = 0;

      FOLD
      {
        int n[]             = {cStack->width};
        int inembed[]       = {cStack->strideCmplx * sizeof(fcomplexcu)};
        int istride         = 1;
        int idist           = cStack->strideCmplx;
        int onembed[]       = {cStack->strideCmplx * sizeof(fcomplexcu)};
        int ostride         = 1;
        int odist           = cStack->strideCmplx;

        cufftCreate(&cStack->plnPlan);
        cufftCreate(&cStack->inpPlan);

        CUFFT_SAFE_CALL(cufftMakePlanMany(cStack->plnPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, cStack->height*kernel->noSteps,    &fftSize), "Creating plan for complex data of stack.");
        fffTotSize += fftSize;

        if (kernel->flag & CU_INPT_FFT_CPU )
        {
          cStack->inpPlanFFTW = fftwf_plan_many_dft(1, n, cStack->noInStack*kernel->noSteps, (fftwf_complex*)cStack->h_iData, n, istride, idist, (fftwf_complex*)cStack->h_iData, n, ostride, odist, -1, FFTW_ESTIMATE);
        }
        else
        {
          CUFFT_SAFE_CALL(cufftMakePlanMany(cStack->inpPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, cStack->noInStack*kernel->noSteps, &fftSize), "Creating plan for input data of stack.");
          fffTotSize += fftSize;
        }

      }
      CUDA_SAFE_CALL(cudaGetLastError(), "Creating FFT plans for the stacks.");
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

  CUDA_SAFE_CALL(cudaGetLastError(), "CUDA Error freeing device memory for kernel.\n");
}

/** Free kernel data structure  .
 *
 * @param kernel
 * @param master
 */
void freeKernel(cuFFdotBatch* kernrl)
{
  freeKernelGPUmem(kernrl);

  if ( kernrl->srchMaster )
  {
    freeNull(kernrl->SrchSz);
    freeNull(kernrl->h_candidates);
  }

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
  for (int i = 0; i < batch->noStacks; i++)
  {
    // Set stack pointers
    cuFfdotStack* cStack  = &batch->stacks[i];

    for (int j = 0; j < cStack->noInStack; j++)
    {
      cuFFdot* cPlane           = &cStack->planes[j];

      cPlane->d_planeMult       = &cStack->d_planeMult[ cStack->startZ[j] * batch->noSteps * cStack->strideCmplx ];
      if (cStack->d_planeIFFT)
        cPlane->d_planeIFFT     = &cStack->d_planeIFFT[ cStack->startZ[j] * batch->noSteps * cStack->strideCmplx ];
      if (cStack->d_planePowr)
        cPlane->d_planePowr     = &cStack->d_planePowr[ cStack->startZ[j] * batch->noSteps * cStack->strideFloat ]; // TODO: Fix
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
  size_t cmplStart  = 0;
  size_t pwrStart   = 0;
  size_t idSiz      = 0;            /// The size in bytes of input data for one stack
  int harm          = 0;            /// The harmonic index of the first plane the the stack

  for (int i = 0; i < batch->noStacks; i++) // Set the various pointers of the stacks  .
  {
    cuFfdotStack* cStack  = &batch->stacks[i];

    cStack->d_iData       = &batch->d_iData[idSiz];
    cStack->h_iData       = &batch->h_iData[idSiz];
    cStack->planes        = &batch->planes[harm];
    cStack->kernels       = &batch->kernels[harm];
    cStack->d_planeMult   = &batch->d_planeMult[cmplStart];
    if (batch->d_planeIFFT)
      cStack->d_planeIFFT = &batch->d_planeIFFT[cmplStart];
    if (batch->d_planePowr)
      cStack->d_planePowr = &batch->d_planePowr[pwrStart]; // TODO: Fix

    // Increment the various values used for offset
    harm                 += cStack->noInStack;
    idSiz                += batch->noSteps  * cStack->strideCmplx * cStack->noInStack;
    cmplStart            += cStack->height  * cStack->strideCmplx * batch->noSteps ;
    pwrStart             += cStack->height  * cStack->strideFloat * batch->noSteps ;
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
  char strBuff[1024];
  size_t free, total;

  FOLD // See if we can use the cuda device  .
  {
    setDevice(kernel) ;

    CUDA_SAFE_CALL(cudaMemGetInfo ( &free, &total ), "Getting Device memory information");
  }

  FOLD // Set up basic slack list parameters from the harmonics  .
  {
    // Copy the basic batch parameters from the kernel
    memcpy(batch, kernel, sizeof(cuFFdotBatch));

    batch->srchMaster   = 0;
    batch->isKernel     = 0;

    // Copy the actual stacks
    batch->stacks = (cuFfdotStack*) malloc(batch->noStacks   * sizeof(cuFfdotStack));
    memcpy(batch->stacks, kernel->stacks, batch->noStacks    * sizeof(cuFfdotStack));
  }

  FOLD // Set the flags  .
  {
    FOLD // multiplication flags  .
    {
      for ( int i = 0; i < batch->noStacks; i++ )
      {
        cuFfdotStack* cStack  = &batch->stacks[i];

        if ( !(cStack->flag & FLAG_MUL_ALL ) )   // Default to multiplication  .
        {
          int noInp =  cStack->noInStack * kernel->noSteps ;

          if ( batch->capability > 3.0 )
          {
            // Lots of registers per thread so 4.2 is good
            cStack->flag |= FLAG_MUL_21;
          }
          else
          {
            // We require fewer registers per thread, so use Multiplication kernel 2.1
            if ( noInp <= 20 )
            {
              // TODO: Check small, looks like some times 22 may be faster.
              cStack->flag |= FLAG_MUL_21;
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
                  cStack->flag |= FLAG_MUL_23;
                }
                else
                {
                  // Here 22 is usually better
                  cStack->flag |= FLAG_MUL_22;
                }
              }
              else
              {
                // Enough steps to justify Multiplication kernel 2.1
                cStack->flag |= FLAG_MUL_22;
              }
            }
          }

          batch->flag |= FLAG_MUL_STK;
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
      if ( !(batch->flag & FLAG_SS_ALL ) )   // Default to multiplication  .
      {
        batch->flag |= FLAG_SS_10;
      }

      if ( batch->ssChunk <= 0 )
      {
        //kernel->ssChunk         = 8 ;
        float val = 30.0 / (float) batch->noSteps ;

        batch->ssChunk = MAX(MIN(floor(val), 9),1);
      }
    }
  }

  FOLD // Allocate all device and host memory for the stacks  .
  {
    FOLD // Allocate page-locked host memory for input data  .
    {
      CUDA_SAFE_CALL(cudaMallocHost(&batch->h_iData, batch->inpDataSize ), "Failed to create page-locked host memory plane input data." );

      if ( batch->flag & CU_NORM_CPU ) // Allocate memory for normalisation
        batch->normPowers = (float*) malloc(batch->hInfos->width * sizeof(float));
    }

    FOLD // Allocate R value lists  .
    {
      rVals*    rLev1;
      rVals**   rLev2;

      int oSet                = 0;
      batch->noRArryas        = 5; // This is just a convenient value

      rLev1                   = (rVals*)malloc(sizeof(rVals)*batch->noSteps*batch->noHarms*batch->noRArryas);
      memset(rLev1, 0, sizeof(rVals)*batch->noSteps*batch->noHarms*batch->noRArryas);

      batch->rArrays          = (rVals***)malloc(batch->noRArryas*sizeof(rVals**));

      for (int rIdx = 0; rIdx < batch->noRArryas; rIdx++)
      {
        rLev2                 = (rVals**)malloc(sizeof(rVals*)*batch->noSteps);
        batch->rArrays[rIdx]  = rLev2;

        for (int step = 0; step < batch->noSteps; step++)
        {
          rLev2[step]         = &rLev1[oSet];
          oSet               += batch->noHarms;
        }
      }
    }

    FOLD // Allocate device Memory for Planes, Stacks & Input data (steps)  .
    {
      size_t req = batch->inpDataSize + batch->plnDataSize + batch->pwrDataSize;

      if ( !(batch->flag & FLAG_CUFFT_CB_OUT) )
      {
        // For second complex plane
        req += batch->plnDataSize;
      }

      if ( req > free )
      {
        // Not enough memory =(

        // NOTE: we could reduce noSteps for this stack, but all batches must be the same size to share the same CFFT plan

        printf("Not enough GPU memory to create any more stacks.\n");
        return 0;
      }
      else
      {
        // Allocate device memory

        CUDA_SAFE_CALL(cudaMalloc((void** )&batch->d_iData,         batch->inpDataSize ), "Failed to allocate device memory for kernel stack.");
        free -= batch->inpDataSize;

        CUDA_SAFE_CALL(cudaMalloc((void** )&batch->d_planeMult,     batch->plnDataSize ), "Failed to allocate device memory for kernel stack.");
        free -= batch->plnDataSize;

        if ( !(batch->flag & FLAG_CUFFT_CB_OUT) ) // Second complex plane
        {
          CUDA_SAFE_CALL(cudaMalloc((void** )&batch->d_planeIFFT,   batch->plnDataSize ), "Failed to allocate device memory for kernel stack.");
          free -= batch->plnDataSize;
        }

        if ( batch->pwrDataSize )
        {
          CUDA_SAFE_CALL(cudaMalloc((void** )&batch->d_planePowr,   batch->pwrDataSize ), "Failed to allocate device memory for kernel stack.");
          free -= batch->pwrDataSize;
        }
      }
    }

    FOLD // Allocate device & page-locked host memory for return data  .
    {
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
            CUDA_SAFE_CALL(cudaMalloc((void** ) &batch->d_retData1, batch->retDataSize ), "Failed to allocate device memory for return values.");
            free -= batch->retDataSize;

            if ( batch->flag & FLAG_SS_INMEM )
              batch->d_retData2 = batch->d_planeMult;
          }
        }
      }

      FOLD // Allocate page-locked host memory to copy the candidates back to  .
      {
        if ( kernel->retDataSize )
        {
          CUDA_SAFE_CALL(cudaMallocHost(&batch->h_retData1, kernel->retDataSize), "Failed to create page-locked host memory plane for return data.");
          memset(batch->h_retData1, 0, kernel->retDataSize );

          if ( kernel->flag & FLAG_SS_INMEM )
          {
            CUDA_SAFE_CALL(cudaMallocHost(&batch->h_retData2, kernel->retDataSize), "Failed to create page-locked host memory plane for return data.");
            memset(batch->h_retData2, 0, kernel->retDataSize );
          }
        }
      }
    }

    // Create the planes structures
    if ( batch->noHarms* sizeof(cuFFdot) > getFreeRamCU() )
    {
      fprintf(stderr, "ERROR: Not enough host memory for search.\n");
      return 0;
    }
    else
    {
      batch->planes = (cuFFdot*) malloc(batch->noHarms* sizeof(cuFFdot));
      memset(batch->planes, 0, batch->noHarms* sizeof(cuFFdot));
    }

    FOLD // Create timing arrays  .
    {
#ifdef TIMING
      batch->copyH2DTime    = (float*)malloc(batch->noStacks*sizeof(float));
      batch->normTime       = (float*)malloc(batch->noStacks*sizeof(float));
      batch->InpFFTTime     = (float*)malloc(batch->noStacks*sizeof(float));
      batch->multTime       = (float*)malloc(batch->noStacks*sizeof(float));
      batch->InvFFTTime     = (float*)malloc(batch->noStacks*sizeof(float));
      batch->copyToPlnTime  = (float*)malloc(batch->noStacks*sizeof(float));
      batch->searchTime     = (float*)malloc(batch->noStacks*sizeof(float));
      batch->resultTime     = (float*)malloc(batch->noStacks*sizeof(float));
      batch->copyD2HTime    = (float*)malloc(batch->noStacks*sizeof(float));

      memset(batch->copyH2DTime,    0,batch->noStacks*sizeof(float));
      memset(batch->normTime,       0,batch->noStacks*sizeof(float));
      memset(batch->InpFFTTime,     0,batch->noStacks*sizeof(float));
      memset(batch->multTime,       0,batch->noStacks*sizeof(float));
      memset(batch->InvFFTTime,     0,batch->noStacks*sizeof(float));
      memset(batch->copyToPlnTime,  0,batch->noStacks*sizeof(float));
      memset(batch->searchTime,     0,batch->noStacks*sizeof(float));
      memset(batch->resultTime,     0,batch->noStacks*sizeof(float));
      memset(batch->copyD2HTime,    0,batch->noStacks*sizeof(float));
#endif
    }
  }

  FOLD // Set up the batch streams and events  .
  {
    FOLD // Create Streams  .
    {
      FOLD // Input streams  .
      {
        // Batch input ( Always needed, for copying input to device )
        CUDA_SAFE_CALL(cudaStreamCreate(&batch->inpStream),"Creating input stream for batch.");
        sprintf(strBuff,"%i.%i.0.0 Batch Input", batch->device, no);
        nvtxNameCudaStreamA(batch->inpStream, strBuff);
        //printf("cudaStreamCreate: %s\n", strBuff);

        // Stack input
        if ( !(batch->flag & CU_NORM_CPU)  )
        {
          for (int i = 0; i < batch->noStacks; i++)
          {
            cuFfdotStack* cStack  = &batch->stacks[i];

            CUDA_SAFE_CALL(cudaStreamCreate(&cStack->inptStream), "Creating input data multStream for stack");
            sprintf(strBuff,"%i.%i.0.%i Stack Input", batch->device, no, i);
            nvtxNameCudaStreamA(cStack->inptStream, strBuff);
            //printf("cudaStreamCreate: %s\n", strBuff);
          }
        }
      }

      FOLD // Input FFT streams  .
      {
        if ( (no == 0) && (of == 0) && !(kernel->flag & CU_INPT_FFT_CPU) )
        {
          //printf("\nRecreating Input FFT streams\n");

          for (int i = 0; i < kernel->noStacks; i++)
          {
            cuFfdotStack* kStack = &kernel->stacks[i];
            //CUDA_SAFE_CALL(cudaStreamDestroy(kStack->fftIStream),"Creating CUDA stream for fft's");

            cuFfdotStack* cStack = &batch->stacks[i];
            CUDA_SAFE_CALL(cudaStreamCreate(&cStack->fftIStream),"Creating CUDA stream for fft's");
            //sprintf(strBuff,"%i FFT Input %i Stack", batch->device, i);
            sprintf(strBuff,"%i.0.2.%i FFT Input", batch->device, i);
            nvtxNameCudaStreamA(cStack->fftIStream, strBuff);
            kStack->fftIStream = cStack->fftIStream;
            //printf("cudaStreamCreate: %s\n", strBuff);
          }
        }
      }

      FOLD // Multiply streams  .
      {
        if      ( batch->flag & FLAG_MUL_BATCH )
        {
          CUDA_SAFE_CALL(cudaStreamCreate(&batch->multStream),"Creating multiplication stream for batch.");
          sprintf(strBuff,"%i.%i.3.0 Batch Multiply", batch->device, no);
          nvtxNameCudaStreamA(batch->multStream, strBuff);
          //printf("cudaStreamCreate: %s\n", strBuff);
        }

        if ( (batch->flag & FLAG_MUL_STK) || (batch->flag & FLAG_MUL_PLN)  )
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
        if ( (no == 0) && (of == 0) )
        {
          //printf("\nRecreating Inverse FFT streams\n");

          for (int i = 0; i < kernel->noStacks; i++)
          {
            cuFfdotStack* kStack = &kernel->stacks[i];
            //CUDA_SAFE_CALL(cudaStreamDestroy(kStack->fftPStream),"Creating CUDA stream for fft's");

            cuFfdotStack* cStack = &batch->stacks[i];
            CUDA_SAFE_CALL(cudaStreamCreate(&cStack->fftPStream),"Creating CUDA stream for fft's");
            //sprintf(strBuff,"%i FFT Plane %i Stack", batch->device, i);
            sprintf(strBuff,"%i.0.4.%i FFT Plane", batch->device, i);
            nvtxNameCudaStreamA(cStack->fftPStream, strBuff);
            kStack->fftPStream = cStack->fftPStream;
            //printf("cudaStreamCreate: %s\n", strBuff);
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
        CUDA_SAFE_CALL(cudaStreamCreate(&batch->resStream), "Creating strmSearch for batch.");
        sprintf(strBuff,"%i.%i.6.0 Batch result", batch->device, no);
        nvtxNameCudaStreamA(batch->resStream, strBuff);
        //printf("cudaStreamCreate: %s\n", strBuff);
      }


    }

    FOLD // Create Events  .
    {
      FOLD // Create batch events  .
      {
#ifdef TIMING
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
#else
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(&batch->iDataCpyComp,   cudaEventDisableTiming ), "Creating input event iDataCpyComp.");
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(&batch->candCpyComp,    cudaEventDisableTiming ), "Creating input event candCpyComp.");
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(&batch->normComp,       cudaEventDisableTiming ), "Creating input event normComp.");
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(&batch->multComp,       cudaEventDisableTiming ), "Creating input event searchComp.");
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(&batch->searchComp,     cudaEventDisableTiming ), "Creating input event searchComp.");
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(&batch->processComp,    cudaEventDisableTiming ), "Creating input event processComp.");
#endif
      }

      FOLD // Create stack events  .
      {
        for (int i = 0; i< batch->noStacks; i++)
        {
          cuFfdotStack* cStack  = &batch->stacks[i];

#ifdef TIMING
          // in  events
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->normInit),    "Creating input normalisation event");
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->inpFFTinit),  "Creating input FFT initialisation event");
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->ifftInit), 	  "Creating inverse FFT initialisation event");
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->multInit), 		"Creating multiplication initialisation event");

          // out events (with timing)
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->normComp),    "Creating input normalisation event");
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->prepComp), 		"Creating input data preparation complete event");
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->multComp), 		"Creating multiplication complete event");
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->ifftComp),    "Creating IFFT complete event");
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->ifftMemComp), "Creating IFFT memory copy complete event");
#else
          // out events (without timing)
          CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->normComp,    cudaEventDisableTiming), "Creating input data preparation complete event");
          CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->prepComp,    cudaEventDisableTiming), "Creating input data preparation complete event");
          CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->multComp,    cudaEventDisableTiming), "Creating multiplication complete event");
          CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->ifftComp,    cudaEventDisableTiming), "Creating IFFT complete event");
          CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->ifftMemComp, cudaEventDisableTiming), "Creating IFFT memory copy complete event");
#endif
        }
      }
    }

    if ( 0 ) // TMP
    {
      for (int i = 0; i< batch->noStacks; i++)
      {
        cuFfdotStack* cStack = &batch->stacks[i];

        cStack->fftIStream = cStack->inptStream;
        cStack->fftPStream = cStack->multStream;
      }
    }

    CUDA_SAFE_CALL(cudaGetLastError(), "Creating streams and events for the batch.");
  }

  FOLD // Setup the pointers for the stacks and planes of this batch  .
  {
    setBatchPointers(batch);
  }

  /*   // Rather use 1 FFT plan per device  .
  FOLD // Create FFT plans  .
  {
    for (int i = 0; i < batch->noStacks; i++)
    {
      cuFfdotStack* cStack  = &batch->stacks[i];
      size_t fftSize        = 0;

      FOLD
      {
        int n[]             = {cStack->width};
        int inembed[]       = {cStack->stride* sizeof(fcomplexcu)};
        int istride         = 1;
        int idist           = cStack->stride;
        int onembed[]       = {cStack->stride* sizeof(fcomplexcu)};
        int ostride         = 1;
        int odist           = cStack->stride;

        cufftCreate(&cStack->plnPlan);
        cufftCreate(&cStack->inpPlan);

        CUFFT_SAFE_CALL(cufftPlanMany    (&cStack->plnPlan, 1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, cStack->height),     "Creating plan for complex data of stack.");
        //CUFFT_SAFE_CALL(cufftMakePlanMany(cStack->plnPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, cStack->height, &fftSize),     "Creating plan for complex data of stack.");
        CUFFT_SAFE_CALL(cufftPlanMany    (&cStack->inpPlan, 1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, cStack->noInStack),  "Creating plan for input data of stack.");
        //CUFFT_SAFE_CALL(cufftMakePlanMany(cStack->inpPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, cStack->noInStack, &fftSize),  "Creating plan for input data of stack.");
      }
      CUDA_SAFE_CALL(cudaGetLastError(), "Creating FFT plans for the stacks.");
    }
  }
   */

  FOLD // Create textures for the f-∂f planes  .
  {
    if ( (batch->flag & FLAG_TEX_INTERP) && !( (batch->flag & FLAG_CUFFT_CB_OUT) && (batch->flag & FLAG_SAS_TEX) ) )
    {
      fprintf(stderr, "ERROR: Cannot use texture memory interpolation without CUFFT callback to write powers. NOT using texture memory interpolation\n");
      batch->flag &= ~FLAG_TEX_INTERP;
    }

    if ( batch->flag & FLAG_SAS_TEX )
    {
      cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

      struct cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.addressMode[0]    = cudaAddressModeClamp;
      texDesc.addressMode[1]    = cudaAddressModeClamp;
      texDesc.readMode          = cudaReadModeElementType;
      texDesc.normalizedCoords  = 0;

      if ( batch->flag & FLAG_TEX_INTERP )
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

          if ( batch->flag & FLAG_CUFFT_CB_OUT ) // float input
          {
            if      ( batch->flag & FLAG_ITLV_ROW )
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
            if      ( batch->flag & FLAG_ITLV_ROW )
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

  return batch->noSteps;
}

/** Free batch data structure  .
 *
 * @param batch
 */
void freeBatchGPUmem(cuFFdotBatch* batch)
{
  CUDA_SAFE_CALL(cudaGetLastError(), "Entering freeBatchGPUmem.");

  setDevice(batch) ;

  FOLD // Free host memory
  {
#ifdef STPMSG
    printf("\t\tfree host memory\n", batch);
#endif

    freeNull(batch->normPowers);
  }

  FOLD // Free pinned memory
  {
#ifdef STPMSG
    printf("\t\tfree pinned memory\n", batch);
#endif
    cudaFreeHostNull(batch->h_iData);
    cudaFreeHostNull(batch->h_retData1);
  }

  FOLD // Free device memory
  {
#ifdef STPMSG
    printf("\t\tfree device memory\n", batch);
#endif

    if ( batch->d_retData1 == batch->d_planeMult )
      batch->d_retData1 = batch->d_retData2 ;

    cudaFreeNull(batch->d_iData);
    cudaFreeNull(batch->d_planeMult );
    cudaFreeNull(batch->d_planeIFFT );
    cudaFreeNull(batch->d_planePowr );
    cudaFreeNull(batch->d_retData1);
    batch->d_retData2 = NULL;
  }

  FOLD // Free textures for the f-∂f planes  .
  {
    if ( batch->flag & FLAG_SAS_TEX )
    {
#ifdef STPMSG
      printf("\t\tfree textures\n", batch);
#endif

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

#ifdef TIMING
    freeNull(batch->copyH2DTime   );
    freeNull(batch->normTime      );
    freeNull(batch->InpFFTTime    );
    freeNull(batch->multTime      );
    freeNull(batch->InvFFTTime    );
    freeNull(batch->copyToPlnTime );
    freeNull(batch->searchTime    );
    freeNull(batch->resultTime    );
    freeNull(batch->copyD2HTime   );
#endif
  }

}

cuOptCand* initOptCand(searchSpecs* sSpec)
{
  cuOptCand* oPln;

  oPln = (cuOptCand*)malloc(sizeof(cuOptCand));
  memset(oPln, 0, sizeof(cuOptCand));

  int       noHarms   = (1<<(sSpec->noHarmStages-1));

  oPln->maxNoR        = 512;
  oPln->maxNoZ        = 512;
  oPln->outSz         = oPln->maxNoR * oPln->maxNoZ ;   // This needs to be multiplied by the size of the output element
  oPln->alignment     = getMemAlignment();
  float zMax          = MAX(sSpec->zMax+50, sSpec->zMax*2);
  zMax                = MAX(zMax, 60 * noHarms );
  zMax                = MAX(zMax, sSpec->zMax * 34 + 50 ); // TMP
  oPln->maxHalfWidth  = z_resp_halfwidth( zMax, HIGHACC );

  // Create streams
  CUDA_SAFE_CALL(cudaStreamCreate(&oPln->stream),"Creating stream for candidate optimisation.");
  //nvtxNameCudaStreamA(oPln->stream, "Optimisation Stream");

  // Events
  CUDA_SAFE_CALL(cudaEventCreate(&oPln->inpInit),     "Creating input event inpInit." );
  CUDA_SAFE_CALL(cudaEventCreate(&oPln->inpCmp),      "Creating input event inpCmp."  );
  CUDA_SAFE_CALL(cudaEventCreate(&oPln->compInit),    "Creating input event compInit.");
  CUDA_SAFE_CALL(cudaEventCreate(&oPln->compCmp),     "Creating input event compCmp." );
  CUDA_SAFE_CALL(cudaEventCreate(&oPln->outInit),     "Creating input event outInit." );
  CUDA_SAFE_CALL(cudaEventCreate(&oPln->outCmp),      "Creating input event outCmp."  );

  return oPln;
}

cuOptCand* initOptPln(searchSpecs* sSpec)
{
  nvtxRangePush("Init plane");
  size_t freeMem, totalMem;

  int       noHarms   = (1<<(sSpec->noHarmStages-1));

  cuOptCand* oPln     = initOptCand(sSpec);

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

  nvtxRangePop();

  return(oPln);
}

cuOptCand* initOptSwrm(searchSpecs* sSpec)
{
  size_t freeMem, totalMem;

  cuOptCand* oPln     = initOptCand(sSpec);

  oPln->outSz         *= sizeof(candOpt);
  oPln->inpSz         = sSpec->fftInf.nor * sizeof(fcomplex) ;

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
    CUDA_SAFE_CALL(cudaMalloc(&oPln->d_inp,  oPln->inpSz),    "Failed to allocate device memory for kernel stack.");
    CUDA_SAFE_CALL(cudaMalloc(&oPln->d_out,  oPln->outSz),    "Failed to allocate device memory for kernel stack.");

    // Allocate host memory
    CUDA_SAFE_CALL(cudaMallocHost(&oPln->h_out, oPln->outSz), "Failed to allocate device memory for kernel stack.");

    // Copy FFT to device memory
    CUDA_SAFE_CALL(cudaMemcpy(oPln->d_inp, sSpec->fftInf.fft, oPln->inpSz, cudaMemcpyHostToDevice),      "Copying FFT to device");
  }

  return(oPln);
}

int setStackInfo(cuFFdotBatch* batch, stackInfo* h_inf, int offset)
{
  stackInfo* dcoeffs;
  cudaGetSymbolAddress((void **)&dcoeffs, STACKS );

  for (int i = 0; i < batch->noStacks; i++)
  {
    cuFfdotStack* cStack  = &batch->stacks[i];
    stackInfo*    cInf    = &h_inf[i];

    cInf->noSteps         = batch->noSteps;
    cInf->noPlanes        = cStack->noInStack;
    cInf->famIdx          = cStack->startIdx;
    cInf->flag            = batch->flag;

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
      for (int i = 0; i < batch->noHarms; i++)
      {
        height[i] = batch->hInfos[i].height;
        stride[i] = batch->hInfos[i].inpStride;
        width[i]  = batch->hInfos[i].width;
        kerPnt[i] = batch->kernels[i].d_kerData;

        if ( (i>=batch->noHarms) &&  (batch->hInfos[i].width != batch->hInfos[i].inpStride) )
        {
          fprintf(stderr,"ERROR: Width is not the same as stride, using width this may case errors in the multiplication.\n");
        }
      }

      // Rest
      for (int i = batch->noHarms; i < MAX_HARM_NO; i++)
      {
        height[i] = 0;
        stride[i] = 0;
        width[i]  = 0;
        kerPnt[i] = 0;
      }
    }

    cudaGetSymbolAddress((void **)&dcoeffs, HEIGHT_HARM);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &height, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, STRIDE_HARM);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &stride, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, WIDTH_HARM);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &width,  MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, KERNEL_HARM);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &kerPnt, MAX_HARM_NO * sizeof(fcomplexcu*), cudaMemcpyHostToDevice),      "Copying stages to device");
  }

  CUDA_SAFE_CALL(cudaGetLastError(), "Error preparing the constant memory values for the multiplications.");

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
int setConstStkInfo(stackInfo* h_inf, int noStacks)
{
  void *dcoeffs;

  // TODO: Do a test to see if  we are on the correct device

  cudaGetSymbolAddress((void **)&dcoeffs, STACKS);
  CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, h_inf, noStacks * sizeof(stackInfo), cudaMemcpyHostToDevice),      "Copying stack info to device");

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

void timeAsynch(cuFFdotBatch* batch)
{
  if ( batch->rValues[0][0].numrs )
  {
#ifdef TIMING // Timing  .

    float time;         // Time in ms of the thing
    cudaError_t ret;    // Return status of cudaEventElapsedTime

    FOLD // Norm Timing  .
    {
      if ( !(batch->flag & CU_NORM_CPU) )
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
      }
    }

    FOLD // Input FFT timing  .
    {
      if ( !(batch->flag & CU_INPT_FFT_CPU) )
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
    }

    FOLD // Convolution timing  .
    {
      if ( !(batch->flag & FLAG_CUFFT_CB_IN) )
      {
        // Did the convolution by separate kernel

        if ( batch->flag & FLAG_MUL_BATCH )   // Convolution was done on the entire batch  .
        {
          ret = cudaEventElapsedTime(&time, batch->multInit, batch->multComp);

          if ( ret != cudaErrorNotReady )
          {
#pragma omp atomic
            batch->multTime[0] += time;
          }
        }
        else                                // Convolution was on a per stack basis  .
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
      }

      CUDA_SAFE_CALL(cudaGetLastError(), "Convolution timing  .");
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

    FOLD // Copy to InMem Plane timing  .
    {
      if ( batch->flag & FLAG_SS_INMEM )
      {
        for (int ss = 0; ss < batch->noStacks; ss++)
        {
          cuFfdotStack* cStack = &batch->stacks[ss];

          ret = cudaEventElapsedTime(&time, cStack->ifftComp, cStack->ifftMemComp);
          if ( ret != cudaErrorNotReady )
          {
#pragma omp atomic
            batch->copyToPlnTime[ss] += time;
          }
        }
        CUDA_SAFE_CALL(cudaGetLastError(), "Copy to InMem Plane timing");
      }
    }

    FOLD // Search Timing  .
    {
      if ( !(batch->flag & FLAG_SS_CPU) && !(batch->flag & FLAG_SS_INMEM ) )
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
      if ( !(batch->flag & FLAG_SS_INMEM ) )
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

#endif
  }

}

/** Cycle the arrays of r-values  .
 *
 * @param batch
 */
void cycleRlists(cuFFdotBatch* batch)
{
#ifdef STPMSG
  printf("\tcycleRlists\n");
#endif

  rVals** hold = batch->rArrays[batch->noRArryas-1];
  for ( int i = batch->noRArryas-1; i > 0; i-- )
  {
    batch->rArrays[i] =  batch->rArrays[i - 1];
  }
  batch->rArrays[0] = hold;
}

void cycleOutput(cuFFdotBatch* batch)
{
  void* d_hold = batch->d_retData1;
  void* h_hold = batch->h_retData1;

  batch->d_retData1 = batch->d_retData2;
  batch->h_retData1 = batch->h_retData2;

  batch->d_retData2 = d_hold;
  batch->h_retData2 = h_hold;
}

void search_ffdot_batch_CU(cuFFdotBatch* batch, double* searchRLow, double* searchRHi, int norm_type )
{
  CUDA_SAFE_CALL(cudaGetLastError(), "Error entering search_ffdot_batch_CU.");

#ifdef STPMSG
  printf("  %s\n", __FUNCTION__);
#endif

  // Calculate R values
  setActiveBatch(batch, 0);
  setStackRVals(batch, searchRLow, searchRHi );

#ifdef SYNCHRONOUS

  initInput(batch, norm_type);

  multiplyBatch(batch);

  IFFTBatch(batch);

  if  ( batch->flag & FLAG_SS_INMEM )
  {
    copyToInMemPln(batch);
  }
  else
  {
    sumAndSearch(batch);

    getResults(batch);

    processSearchResults(batch);
  }

  timeAsynch(batch);

#else

  if ( 0 )
  {
    // This ordering has been deprecated

    setActiveBatch(batch, 2);
    sumAndSearch(batch);

    setActiveBatch(batch, 1);
    convolveBatch(batch);

    setActiveBatch(batch, 3);
    processSearchResults(batch);

    setActiveBatch(batch, 2);
    getResults(batch);

    setActiveBatch(batch, 0);
    initInput(batch, norm_type);
  }
  else
  {
    setActiveBatch(batch, 0);
    initInput(batch, norm_type);

    if  ( batch->flag & FLAG_SS_INMEM )
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
#endif

// Change R-values
cycleRlists(batch);

#ifdef STPMSG
printf("  Done (%s)\n", __FUNCTION__);
#endif
}

void finish_Search(cuFFdotBatch* batch)
{
  FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host
  {
    for (int ss = 0; ss < batch->noStacks; ss++)
    {
      nvtxRangePush("EventSynch");
      cuFfdotStack* cStack = &batch->stacks[ss];
      CUDA_SAFE_CALL(cudaEventSynchronize(cStack->ifftMemComp), "ERROR: cudaEventSynchronize.");
      nvtxRangePop();
    }

    nvtxRangePush("EventSynch");
    CUDA_SAFE_CALL(cudaEventSynchronize(batch->processComp), "ERROR: cudaEventSynchronize.");
    nvtxRangePop();
  }
}

void max_ffdot_planeCU(cuFFdotBatch* batch, double* searchRLow, double* searchRHi, int norm_type, fcomplexcu* fft, long long* numindep, float* powers)
{
  CUDA_SAFE_CALL(cudaGetLastError(), "Error entering ffdot_planeCU2.");

  FOLD // Initialise input data  .
  {
    setActiveBatch(batch, 0);
    initInput(batch, norm_type);
  }

#ifdef SYNCHRONOUS

  FOLD // Multiply & inverse FFT  .
  {
    convolveBatch(batch);
  }

  FOLD // Sum & Max
  {
    //sumAndMax(batch, numindep, powers);
  }

#else

  FOLD // Sum & Max
  {
    //sumAndMax(batch, numindep, powers);
  }

  FOLD // Multiply & inverse FFT  .
  {
    convolveBatch(batch);
  }

#endif

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

  CUDA_SAFE_CALL(cudaSetDevice(device), "ERROR: cudaSetDevice");
  CUDA_SAFE_CALL(cudaDeviceReset(), "ERROR: cudaDeviceReset");
  CUDA_SAFE_CALL(cudaGetLastError(), "CUDA Error At start of everything?.\n");
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

int setDevice(cuFFdotBatch* batch)
{
  int dev;

  CUDA_SAFE_CALL(cudaGetDevice(&dev), "Failed to get device using cudaGetDevice");

  if ( dev != batch->device )
  {
    CUDA_SAFE_CALL(cudaSetDevice(batch->device), "ERROR: cudaSetDevice");
    CUDA_SAFE_CALL(cudaGetDevice(&dev), "Failed to get device using cudaGetDevice");
    if ( dev != batch->device )
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

  CUDA_SAFE_CALL(cudaGetLastError(), "Error entering readGPUcmd.");

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

#ifdef TIMING
    if ( gpul.noDevBatches[dev] > 1 )
    {
      //fprintf(stderr,"WARING: Compiled in timing mode, user requested %i batches but will only process 1 on device %i.\n", gpul.noDevBatches[dev], gpul.devId[dev] );
      //gpul.noDevBatches[dev] = 1;
    }
#endif

    if ( dev >= cmd->nstepsC )
      gpul.noDevSteps[dev] = cmd->nsteps[cmd->nbatchC-1];
    else
      gpul.noDevSteps[dev] = cmd->nsteps[dev];
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

void readAccelDefalts(searchSpecs *sSpec)
{
  uint* flags = &(sSpec->flags);
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
        (*flags) |= FLAG_MUL_00;
      }
      else if ( strCom(line, "FLAG_MUL_10" ) || strCom(line, "MUL_10" ) )
      {
        (*flags) &= ~FLAG_MUL_ALL;
        (*flags) |= FLAG_MUL_10;
      }
      else if ( strCom(line, "FLAG_MUL_21" ) || strCom(line, "MUL_21" ) )
      {
        (*flags) &= ~FLAG_MUL_ALL;
        (*flags) |= FLAG_MUL_21;
      }
      else if ( strCom(line, "FLAG_MUL_22" ) || strCom(line, "MUL_22" ) )
      {
        (*flags) &= ~FLAG_MUL_ALL;
        (*flags) |= FLAG_MUL_22;
      }
      else if ( strCom(line, "FLAG_MUL_23" ) || strCom(line, "MUL_23" ) )
      {
        (*flags) &= ~FLAG_MUL_ALL;
        (*flags) |= FLAG_MUL_23;
      }
      else if ( strCom(line, "FLAG_MUL_30" ) || strCom(line, "MUL_30" ) )
      {
        (*flags) &= ~FLAG_MUL_ALL;
        (*flags) |= FLAG_MUL_30;
      }
      else if ( strCom(line, "FLAG_MUL_A"  ) || strCom(line, "MUL_A"  ) )
      {
        (*flags) &= ~FLAG_MUL_ALL;
      }

      else if ( strCom(line, "FLAG_TEX_MUL" ) )
      {
        (*flags) |= FLAG_TEX_MUL;
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

      else if ( strCom(line, "FLAG_CUFFT_CB_IN" ) || strCom(line, "CB_IN" ) )
      {
#if CUDA_VERSION >= 6050
        (*flags) |= FLAG_CUFFT_CB_IN;
#else
        line[flagLen] = 0;
        fprintf(stderr, "WARNING: Use of CUDA callbacks requires CUDA 6.5 or greater.  (FLAG: %s line %i in %s)\n", line, lineno, fName);
#endif
      }

      else if ( strCom(line, "FLAG_CUFFT_CB_OUT" ) || strCom(line, "CB_OUT" ) )
      {
#if CUDA_VERSION >= 6050
        (*flags) |= FLAG_CUFFT_CB_OUT;
#else
        line[flagLen] = 0;
        fprintf(stderr,"WARNING: Use of CUDA callbacks requires CUDA 6.5 or greater.  (FLAG: %s line %i in %s)\n", line, lineno, fName);
#endif
      }

      else if ( strCom(line, "FLAG_NO_CB" ) || strCom(line, "NO_CB" ) )
      {
        (*flags) &= ~FLAG_CUFFT_CB_IN;
        (*flags) &= ~FLAG_CUFFT_CB_OUT;
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

      else if ( strCom(line, "FLAG_SS_CPU" 	) || strCom(line, "SS_CPU" 	) )
      {
        (*flags) &= ~FLAG_SS_ALL;
        (*flags) |= FLAG_SS_CPU;

        // CPU Significance
        (*flags) &= ~FLAG_SIG_GPU;

        sSpec->retType &= ~CU_SRT_ALL   ;
        sSpec->retType |= CU_STR_PLN    ;

        if ( (*flags) & FLAG_CUFFT_CB_OUT )
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
      }
      else if ( strCom(line, "FLAG_SS_10"  	) || strCom(line, "SS_10"  	) )
      {
        (*flags) &= ~FLAG_SS_ALL;
        (*flags) |= FLAG_SS_10;
      }
      //      else if ( strCom(line, "FLAG_SS_20"  	) || strCom(line, "SS_20"  	) )
      //      {
      //        (*flags) &= ~FLAG_SS_ALL;
      //        (*flags) |= FLAG_SS_20;
      //      }
      //      else if ( strCom(line, "FLAG_SS_30"  	) || strCom(line, "SS_30"  	) )
      //      {
      //        (*flags) &= ~FLAG_SS_ALL;
      //        (*flags) |= FLAG_SS_30;
      //      }
      else if ( strCom(line, "FLAG_SS_INMEM") || strCom(line, "SS_INMEM") )
      {
        (*flags) |= FLAG_SS_INMEM;

#if CUDA_VERSION >= 6050
        (*flags) |= FLAG_CUFFT_CB_OUT;
#else
        (*flags) &= ~FLAG_CUFFT_CB_OUT;
#endif

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
        }
        else if ( no == 1 )
        {
          (*flags) &= ~FLAG_SS_ALL;
          (*flags) |= FLAG_SS_10;
        }
        //        else if ( no == 2 )
        //        {
        //          (*flags) &= ~FLAG_SS_ALL;
        //          (*flags) |= FLAG_SS_20;
        //        }
        //        else if ( no == 3 )
        //        {
        //          (*flags) &= ~FLAG_SS_ALL;
        //          (*flags) |= FLAG_SS_30;
        //        }
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

          if ( (*flags) & FLAG_CUFFT_CB_OUT )
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
#if CUDA_VERSION >= 6050
          (*flags) |= FLAG_CUFFT_CB_OUT;
#else
          (*flags) &= ~FLAG_CUFFT_CB_OUT;
#endif
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
#if __CUDACC_VER__ >= 70500
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
      else if ( strCom(line, "FLAG_RAND_2" ) || strCom(line, "RAND_2" ) )
      {
        (*flags) |= FLAG_RAND_2;
      }

      else if ( strCom(line, "FLAG_KER_ACC" ) )
      {
        (*flags) |= FLAG_KER_ACC;
      }

      else if ( strCom(line, "FLAG" ) || strCom(line, "CU_" ) )
      {
        line[flagLen] = 0;
        fprintf(stderr, "ERROR: Found unknown flag %s on line %i of %s.\n", line, lineno, fName);
      }

      else if ( strCom(line, "cuMedianBuffSz" ) )
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

      else if ( strCom(line, "pltOpt"  ) )
      {
        pltOpt    = 1;
      }
      else if ( strCom(line, "PLT_OPT" ) )
      {
        pltOpt    = 1;
      }

      else if ( strCom(line, "UNOPT" ) )
      {
        useUnopt    = 1;
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

  CUDA_SAFE_CALL(cudaGetLastError(), "Error entering readSrchSpecs.");

  // Defaults for accel search
  sSpec.flags         |= FLAG_RET_STAGES  ;
  sSpec.flags         |= FLAG_ITLV_ROW    ;

#ifndef DEBUG
  sSpec.flags         |= FLAG_THREAD      ; // Multithreading really slows down debug so only turn it on by default for release mode, NOTE: This can be over ridden in the defaults file
#endif

#if CUDA_VERSION >= 6050
  sSpec.flags         |= FLAG_CUFFT_CB_OUT;
#endif

  sSpec.cndType       |= CU_CANDFULL    ;   // Candidate data type - CU_CANDFULL this should be the default as it has all the needed data
  sSpec.cndType       |= CU_STR_ARR     ;   // Candidate storage structure - CU_STR_ARR    is generally the fastest

  sSpec.retType       |= CU_POWERZ_S    ;   // Return type
  sSpec.retType       |= CU_STR_ARR     ;   // Candidate storage structure

  sSpec.fftInf.fft    = obs->fft;
  sSpec.fftInf.nor    = obs->numbins;
  sSpec.fftInf.rlo    = obs->rlo;
  sSpec.fftInf.rhi    = obs->rhi;

  sSpec.noHarmStages  = obs->numharmstages;
  sSpec.zMax          = obs->zhi;
  sSpec.sigma         = cmd->sigma;
  sSpec.pWidth        = cmd->width;

  if ( obs->inmem )
  {
    sSpec.flags |= FLAG_SS_INMEM;

#if CUDA_VERSION >= 6050
    sSpec.flags |= FLAG_CUFFT_CB_OUT;
#else
    sSpec.flags &= ~FLAG_CUFFT_CB_OUT;
#endif
  }

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
 * @param gSpec
 * @param sSpec
 * @param powcut
 * @param numindep
 * @return
 */
void initCuAccel(cuSearch* sSrch )
{
  //cuMemInfo* aInf = new cuMemInfo;
  sSrch->mInf = new cuMemInfo;
  memset(sSrch->mInf, 0, sizeof(cuMemInfo));


  CUDA_SAFE_CALL(cudaGetLastError(), "Error entering initCuAccel.");

  FOLD // Create the primary stack on each device, this contains the kernel  .
  {
    nvtxRangePush("Initialise Kernels");

    sSrch->mInf->kernels = (cuFFdotBatch*)malloc(sSrch->gSpec->noDevices*sizeof(cuFFdotBatch));

    int added;
    cuFFdotBatch* master = NULL;

    for ( int dev = 0 ; dev < sSrch->gSpec->noDevices; dev++ ) // Loop over devices  .
    {
      added = initKernel(&sSrch->mInf->kernels[sSrch->mInf->noDevices], master, sSrch, sSrch->gSpec->devId[dev], sSrch->gSpec->noDevBatches[dev], sSrch->gSpec->noDevSteps[dev] );

      if ( added && !master ) // This was the first batch so it is the master
      {
        master = &sSrch->mInf->kernels[0];
      }

      if ( added )
      {
        sSrch->mInf->noBatches += added;
        sSrch->mInf->noDevices++;
      }
      else
      {
        sSrch->gSpec->noDevBatches[dev] = 0;
        fprintf(stderr, "ERROR: failed to set up a kernel on device %i, trying to continue... \n", sSrch->gSpec->devId[dev]);
      }
    }

    nvtxRangePop();

    if ( sSrch->mInf->noDevices <= 0 ) // Check if we got any devices  .
    {
      fprintf(stderr, "ERROR: Failed to set up a kernel on any device. Try -lsgpu to see what devices there are.\n");
      exit (EXIT_FAILURE);
    }

  }

  FOLD // Create planes for calculations  .
  {
    nvtxRangePush("Initialise Batches");

    sSrch->mInf->noSteps       = 0;
    sSrch->mInf->batches       = (cuFFdotBatch*)malloc(sSrch->mInf->noBatches*sizeof(cuFFdotBatch));
    sSrch->mInf->devNoStacks   = (int*)malloc(sSrch->gSpec->noDevices*sizeof(int));
    sSrch->mInf->h_stackInfo   = (stackInfo**)malloc(sSrch->gSpec->noDevices*sizeof(stackInfo*));

    memset(sSrch->mInf->devNoStacks,0,sSrch->gSpec->noDevices*sizeof(int));

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
          noSteps = initBatch(&sSrch->mInf->batches[bNo], &sSrch->mInf->kernels[ker], batch, sSrch->gSpec->noDevBatches[dev]-1);

          if ( noSteps == 0 )
          {
            if ( batch == 0 )
            {
              fprintf(stderr, "ERROR: Failed to create at least one batch on device %i.\n", sSrch->mInf->kernels[dev].device);
            }
            break;
          }
          else
          {
            sSrch->mInf->noSteps           += noSteps;
            sSrch->mInf->devNoStacks[dev]  += sSrch->mInf->batches[bNo].noStacks;
            bNo++;
          }
        }

        int noStacks = sSrch->mInf->devNoStacks[dev] ;
        if ( noStacks )
        {
          sSrch->mInf->h_stackInfo[dev] = (stackInfo*)malloc(noStacks*sizeof(stackInfo));
          int idx = 0;

          // Set the values of the host data structures
          for (int batch = firstBatch; batch < bNo; batch++)
          {
            idx += setStackInfo(&sSrch->mInf->batches[batch], sSrch->mInf->h_stackInfo[dev], idx);
          }

          if ( idx != noStacks )
          {
            fprintf (stderr,"ERROR: in %s line %i, The number of steps on device do not match.\n.",__FILE__, __LINE__);
          }
          else
          {
            setConstStkInfo(sSrch->mInf->h_stackInfo[dev], idx);
          }
        }

        ker++;
      }
    }

    if ( bNo != sSrch->mInf->noBatches )
    {
      fprintf(stderr, "WARNING: Number of batches created does not match the number anticipated.\n");
      sSrch->mInf->noBatches = bNo;
    }

    nvtxRangePop();
  }
}

void freeAccelGPUMem(cuMemInfo* aInf)
{
#ifdef STPMSG
  printf("freeAccelGPUMem\n");
#endif

  FOLD // Free planes  .
  {
    for ( int batch = 0 ; batch < aInf->noBatches; batch++ )  // Batches
    {
#ifdef STPMSG
      printf("\tfreeBatchGPUmem %i\n", batch);
#endif
      freeBatchGPUmem(&aInf->batches[batch]);
    }
  }

  FOLD // Free kernels  .
  {
    for ( int dev = 0 ; dev < aInf->noDevices; dev++)         // Loop over devices
    {
#ifdef STPMSG
      printf("\tfreeKernelGPUmem %i\n", dev);
#endif
      freeKernelGPUmem(&aInf->kernels[dev]);
    }
  }

#ifdef STPMSG
  printf("Done\n");
#endif
}

void freeCuAccel(cuMemInfo* mInf)
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

    for ( int i = 0; i < MAX_GPUS; i++ )
      freeNull(mInf->name[i]);

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

cuSearch* initCuSearch(searchSpecs* sSpec, gpuSpecs* gSpec, cuSearch* srch)
{
  bool same   = true;

  CUDA_SAFE_CALL(cudaGetLastError(), "Error entering initCuSearch.");

  if ( srch )
  {
    if ( srch->noHarmStages != sSpec->noHarmStages )
    {
      same = false;
      // ERROR recreate everything
    }

    if ( srch->mInf )
    {
      if ( srch->mInf->kernels->hInfos->zmax != sSpec->zMax )
      {
        same = false;
        // Have to recreate
      }
      if ( srch->mInf->kernels->accelLen != optAccellen(sSpec->pWidth,sSpec->zMax) )
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

  if ( !srch || same == false)
  {
    srch = new cuSearch;
    memset(srch, 0, sizeof(cuSearch));

    srch->noHarmStages    = sSpec->noHarmStages;
    srch->noHarms         = ( 1<<(srch->noHarmStages-1) );

    srch->pIdx            = (int*)malloc(srch->noHarms * sizeof(int));
    srch->powerCut        = (float*)malloc(srch->noHarmStages * sizeof(float));
    srch->numindep        = (long long*)malloc(srch->noHarmStages * sizeof(long long));

    srch->threasdInfo     = new resThrds;
  }

  srch->sSpec             = sSpec;
  srch->gSpec             = gSpec;

  FOLD // Calculate power cutoff and number of independent values  .
  {
    if (sSpec->zMax % ACCEL_DZ)
      sSpec->zMax = (sSpec->zMax / ACCEL_DZ + 1) * ACCEL_DZ;

    int numz = (sSpec->zMax / ACCEL_DZ) * 2 + 1;

    FOLD //
    {
      for (int ii = 0; ii < srch->noHarmStages; ii++)
      {
        if ( sSpec->zMax == 1 )
        {
          srch->numindep[ii]  = (sSpec->fftInf.rhi - sSpec->fftInf.rlo) / srch->noHarms;
        }
        else
        {
          srch->numindep[ii]  = (sSpec->fftInf.rhi - sSpec->fftInf.rlo) * (numz + 1) * ( ACCEL_DZ / 6.95) / (double)(1<<ii);
        }

        srch->powerCut[ii]  = power_for_sigma(sSpec->sigma, (1<<ii), srch->numindep[ii]);
      }
    }
  }

  FOLD // Set up the threading  .
  {
    if (pthread_mutex_init(&srch->threasdInfo->candAdd_mutex, NULL))
    {
      printf("Unable to initialise a mutex.\n");
      exit(EXIT_FAILURE);
    }

    if (sem_init(&srch->threasdInfo->running_threads, 0, 0))
    {
      printf("Could not initialise a semaphore\n");
      exit(EXIT_FAILURE);
    }
    else
    {
      //sem_post(&srch->threasdInfo->running_threads); // Set to 1
      int noTrd;
      sem_getvalue(&srch->threasdInfo->running_threads, &noTrd );
    }


    //    if (pthread_mutex_init(&srch->threasdInfo->running_mutex, NULL))
    //    {
    //      printf("Unable to initialise a mutex.\n");
    //      exit(EXIT_FAILURE);
    //    }
  }

  //  if ( sSpec->cndType & CU_STR_QUAD )
  //  {
  //    // If we are using the quadtree method we can do a 1 stage search
  //    srch->noHarmStages  = 1;
  //    srch->noHarms       = 1;
  //  }

  if ( !srch->mInf )
  {
    //srch->mInf = initCuAccel(gSpec, sSpec, srch->powerCut, srch->numindep );
    initCuAccel( srch );
  }
  else
  {
    // TODO do a whole bunch of checks here!
  }

  return srch;
}

void freeCuSearch(cuSearch* srch)
{
  if (srch)
  {
    if ( srch->mInf )
      freeCuAccel(srch->mInf);

    freeNull(srch->pIdx);
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
  master = srch->mInf->kernels;

#ifdef WITHOMP
  omp_set_num_threads(srch->mInf->noBatches);
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

    cuFFdotBatch* trdBatch = &srch->mInf->batches[tid];

    double*  startrs = (double*)malloc(sizeof(double)*trdBatch->noSteps);
    double*  lastrs  = (double*)malloc(sizeof(double)*trdBatch->noSteps);
    size_t rest = trdBatch->noSteps;

    setDevice(trdBatch) ;

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
#ifdef CBL
  printf("\n Creating data sets...\n");

  nDarray<2, float>gpuCmplx [batch->noSteps][batch->noHarms];
  nDarray<2, float>gpuPowers[batch->noSteps][batch->noHarms];
  for ( int si = 0; si < batch->noSteps ; si ++)
  {
    for (int harm = 0; harm < batch->noHarms; harm++)
    {
      cuHarmInfo *hinf  = &batch[0].hInfos[harm];

      gpuCmplx[si][harm].addDim(hinf->width*2, 0, hinf->width);
      gpuCmplx[si][harm].addDim(hinf->height, -hinf->zmax, hinf->zmax);
      gpuCmplx[si][harm].allocate();

      gpuPowers[si][harm].addDim(hinf->width, 0, hinf->width);
      gpuPowers[si][harm].addDim(hinf->height, -hinf->zmax, hinf->zmax);
      gpuPowers[si][harm].allocate();
    }
  }

  for ( int step = 0; step < batch->noSteps ; step ++)
  {
    for ( int stack = 0 ; stack < batch->noStacks; stack++ )
    {
      for (int harm = 0; harm < batch->noHarms; harm++)
      {
        cuHarmInfo   *cHInfo  = &batch->hInfos[harm];
        cuFfdotStack *cStack  = &batch->stacks[cHInfo->stackNo];
        rVals* rVal           = &batch->rValues[step][harm];

        for( int y = 0; y < cHInfo->height; y++ )
        {

          fcomplexcu *cmplxData;
          float *powers;

          if ( batch->flag & FLAG_ITLV_ROW )
          {
            cmplxData = &batch->d_planeMult[  (y*batch->noSteps + step)*cStack->strideCmplx ];
            powers    = &batch->d_planePowr[ ((y*batch->noSteps + step)*cStack->strideFloat + cHInfo->halfWidth * 2 ) ];
          }
          else
          {
            cmplxData = &batch->d_planeMult[  (y + step*cHInfo->height)*cStack->strideCmplx ];
            powers    = &batch->d_planePowr[ ((y + step*cHInfo->height)*cStack->strideFloat  + cHInfo->halfWidth * 2 ) ];
          }

          cmplxData += cHInfo->halfWidth*2;
          //CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[step][harm].getP(0,y), cmplxData, (cHInfo->width-2*2*cHInfo->halfWidth)*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
          //CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[step][harm].getP(0,y), cmplxData, (cPlane->numrs[step])*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
          CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[step][harm].getP(0,y), cmplxData, (rVal->numrs)*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
          if ( batch->flag & FLAG_CUFFT_CB_OUT )
          {
            //CUDA_SAFE_CALL(cudaMemcpyAsync(gpuPowers[step][harm].getP(0,y), powers, (cPlane->numrs[step])*sizeof(float),   cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
            CUDA_SAFE_CALL(cudaMemcpyAsync(gpuPowers[step][harm].getP(0,y), powers, (rVal->numrs)*sizeof(float),   cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
            /*
            for( int jj = 0; jj < plan->numrs[step]; jj++)
            {
              float *add = gpuPowers[step][harm].getP(jj*2+1,y);
              gpuPowers[step][harm].setPoint<ARRAY_SET>(add, 0);
            }
             */
          }
        }
      }
    }
  }
#else
  fprintf(stderr,"ERROR: Not compiled with debug libraries.\n");
#endif
}

void printBitString(uint val)
{
  printf("Value %015i : ", val);

  for ( int i = 0; i < 32; i++)
  {
    if( val & (1<<(31-i) ) )
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
  searchSpecs* sSpec;  ///< Specifications of the search
  cuMemInfo* mInf;  ///< The allocated Device and host memory and data structures to create planes including the kernels
  cuFFdotBatch* batch;
  sSpec = cuSrch->sSpec;
  mInf = cuSrch->mInf;
  batch = cuSrch->mInf->batches;
  double noRR = sSpec->fftInf.rhi - sSpec->fftInf.rlo;

  char hostname[1024];
  gethostname(hostname, 1024);

  Logger* cvsLog = new Logger(fname, 1);
  cvsLog->sedCsvDeliminator('\t');

  // get the current time
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

    cvsLog->csvWrite("Har",       "#", "%2li",    cuSrch->noHarms);
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
    if ( batch->flag & FLAG_ITLV_ROW )
      cvsLog->csvWrite("IL",      "flg", "ROW");
    else
      cvsLog->csvWrite("IL",      "flg", "PLN");

    if ( batch->flag & CU_NORM_CPU )
      cvsLog->csvWrite("NORM",    "flg", "CPU");
    else
      cvsLog->csvWrite("NORM",    "flg", "GPU");

    if ( batch->flag & CU_INPT_FFT_CPU )
      cvsLog->csvWrite("Inp FFT", "flg", "CPU");
    else
      cvsLog->csvWrite("Inp FFT", "flg", "GPU");

    if      ( batch->flag & FLAG_MUL_00 )
      cvsLog->csvWrite("MUL",    "flg", "00");
    else if ( batch->flag & FLAG_MUL_10 )
      cvsLog->csvWrite("MUL",    "flg", "10");
    else if ( batch->flag & FLAG_MUL_21 )
      cvsLog->csvWrite("MUL",    "flg", "21");
    else if ( batch->flag & FLAG_MUL_22 )
      cvsLog->csvWrite("MUL",    "flg", "22");
    else if ( batch->flag & FLAG_MUL_23 )
      cvsLog->csvWrite("MUL",    "flg", "23");
    else if ( batch->flag & FLAG_MUL_30 )
      cvsLog->csvWrite("MUL",    "flg", "30");
    else
      cvsLog->csvWrite("MUL",    "flg", "?");

    if      ( batch->flag & FLAG_SS_00  )
      cvsLog->csvWrite("SS",    "flg", "00");
    else if ( batch->flag & FLAG_SS_10  )
      cvsLog->csvWrite("SS",    "flg", "10");
    //    else if ( batch->flag & FLAG_SS_20  )
    //      cvsLog->csvWrite("SS",    "flg", "20");
    //    else if ( batch->flag & FLAG_SS_30  )
    //      cvsLog->csvWrite("SS",    "flg", "30");
    else if ( batch->flag & FLAG_SS_CPU )
      cvsLog->csvWrite("SS",    "flg", "CPU");
    else
      cvsLog->csvWrite("SS",    "flg", "?");

    cvsLog->csvWrite("CB IN",     "flg", "%i", (bool)(batch->flag & FLAG_CUFFT_CB_IN));
    cvsLog->csvWrite("CB OUT",    "flg", "%i", (bool)(batch->flag & FLAG_CUFFT_CB_OUT));

    cvsLog->csvWrite("MUL_TEX",   "flg", "%i", (bool)(batch->flag & FLAG_TEX_MUL));
    cvsLog->csvWrite("SAS_TEX",   "flg", "%i", (bool)(batch->flag & FLAG_SAS_TEX));
    cvsLog->csvWrite("INTERP",    "flg", "%i", (bool)(batch->flag & FLAG_TEX_INTERP));
    if ( batch->flag & FLAG_SIG_GPU )
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

    cvsLog->csvWrite("RET_ALL",     "flg", "%i", (bool)(batch->flag & FLAG_RET_STAGES));
    cvsLog->csvWrite("STR_ALL",     "flg", "%i", (bool)(batch->flag & FLAG_STORE_ALL));
    cvsLog->csvWrite("STR_EXP",     "flg", "%i", (bool)(batch->flag & FLAG_STORE_EXP));
    cvsLog->csvWrite("KER_ACC",     "flg", "%i", (bool)(batch->flag & FLAG_KER_ACC));
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
#ifdef TIMING
    for (int batch = 0; batch < cuSrch->mInf->noBatches; batch++)
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

      for (int stack = 0; stack < cuSrch->mInf->batches[batch].noStacks; stack++)
      {
        cuFFdotBatch* batches = &cuSrch->mInf->batches[batch];
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
#endif
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
  int maxharm = 16;
  int numremoved = 0;

  //double tooclose = 1.5;

  cand* tmpCand = new cand;
  container* next;
  container* close;
  container* serch;

  container* lst = tree->getLargest();

  while ( lst )
  {
    cand* candidate = (cand*)lst->data;

    tmpCand->power    = candidate->power;
    tmpCand->numharm  = candidate->numharm;
    tmpCand->r        = candidate->r;
    tmpCand->z        = candidate->z;
    tmpCand->sig      = candidate->sig;

    // Remove harmonics down
    for (double ii = 1; ii <= maxharm; ii++)
    {
      FOLD // Remove down candidates  .
      {
        tmpCand->r  = candidate->r / ii;
        tmpCand->z  = candidate->z / ii;
        serch       = contFromCand(tmpCand);
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
        tmpCand->r  = candidate->r * ii;
        tmpCand->z  = candidate->z * ii;
        serch       = contFromCand(tmpCand);
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
      tmpCand->r  = candidate->r * ratioARR[ii];
      tmpCand->z  = candidate->z * ratioARR[ii];
      serch       = contFromCand(tmpCand);
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

  return numremoved;
}

GSList *testTest(cuFFdotBatch* batch, GSList *candsGPU)
{
  candTree optemised;

  candTree trees[batch->noHarmStages];

  candTree* qt =(candTree*)batch->h_candidates;

  hilClimb(qt, 5);
  eliminate_harmonics(qt);

  cuOptCand* oPlnPln;
  oPlnPln   = initOptPln(batch->sInf->sSpec);

  container* cont = qt->getLargest();

  int i = 0;

  while ( cont )
  {
    i++;
    printf("\n");
    //if ( i == 12 )
    {
      cand*   candidate = (cand*)cont->data;
      cont->flag &= ~OPTIMISED_CONTAINER;

      printf("Candidate %03i  harm: %2i   pow: %9.3f   r: %9.4f  z: %7.4f\n",i, candidate->numharm, candidate->power, candidate->r, candidate->z );

      //
      //    numharm   = candidate->numharm;
      //    sig       = candidate->sig;
      //    rr        = candidate->r;
      //    zz        = candidate->z;
      //    poww      = candidate->power;
      //
      //    candsGPU  = insert_new_accelcand(candsGPU, poww, sig, numharm, rr, zz, &added );

      //accelcand *cand = new accelcand;
      //memset(cand, 0, sizeof(accelcand));
      //cand->power   = candidate->power;
      //cand->r       = candidate->r;
      //cand->sigma   = candidate->sig;
      //cand->z       = candidate->z;
      //cand->numharm = candidate->numharm;

      //accelcand* cand = create_accelcand(candidate->power, candidate->sig, candidate->numharm, candidate->r, candidate->z);

      //candsGPU = insert_accelcand(candsGPU, cand),

      int stg = log2((float)candidate->numharm);
      candTree* ret = opt_cont(&trees[stg], oPlnPln, cont, &batch->sInf->sSpec->fftInf, i);

      trees[stg].add(ret);

      delete(ret);

      if ( cont->flag & OPTIMISED_CONTAINER )
      {
        candidate->sig = candidate_sigma_cl(candidate->power, candidate->numharm,  batch->sInf->numindep[stg] );
        container* cont = optemised.insert(candidate, 0.1);

        if ( cont )
        {
          printf("          %03i  harm: %2i   pow: %9.3f   r: %9.4f  z: %7.4f\n",i, candidate->numharm, candidate->power, candidate->r, candidate->z );
        }
        else
        {
          printf("          NO\n");
        }
      }
      else
      {
        printf("          Already Done\n");
      }
    }

    cont = cont->smaller;
  }

  printf("Optimisation Removed %6i - %6i remain \n", qt->noVals() - optemised.noVals(), optemised.noVals() );

  eliminate_harmonics(&optemised);

  return candsGPU;
}
