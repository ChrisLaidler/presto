// Automated testing SS: 16 16

#include <cufft.h>
#include <algorithm>
#include <omp.h>


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

#ifdef CBL
#include <unistd.h>
#include "log.h"
#endif


__device__ __constant__ int           HEIGHT_FAM_ORDER[MAX_HARM_NO];    ///< Plain  height  in stage order
__device__ __constant__ int           STRIDE_FAM_ORDER[MAX_HARM_NO];    ///< Plain  stride  in stage order
__device__ __constant__ int           WIDTH_FAM_ORDER[MAX_HARM_NO];     ///< Plain  strides   in family
__device__ __constant__ fcomplexcu*   KERNEL_FAM_ORDER[MAX_HARM_NO];    ///< Kernel pointer in stage order
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

/*
void printData_cu(cuFFdotBatch* batch, const int FLAGS, int harmonic, int nX, int nY, int sX, int sY)
{
  //cuFFdot* cPlain       = &batch->plains[harmonic];
  //printfData<<<1,1,0,0>>>((float*)cPlain->d_iData, nX, nY, cPlain->harmInf->inpStride, sX, sY);
}
*/

/* The fft length needed to properly process a subharmonic */
static int calc_fftlen3(double harm_fract, int max_zfull, uint accelLen)
{
  int bins_needed, end_effects;

  bins_needed = accelLen * harm_fract + 2;
  end_effects = 2 * ACCEL_NUMBETWEEN * z_resp_halfwidth(calc_required_z(harm_fract, max_zfull), LOWACC);
  return next2_to_n_cu(bins_needed + end_effects);
}

/** Calculate an optimal accellen given a width  .
 *
 * @param width the width of the plain usually a power of two
 * @param zmax
 * @return
 * If width is not a power of two it will be rounded up to the nearest power of two
 */
uint optAccellen(float width, int zmax)
{
  float halfwidth       = z_resp_halfwidth(zmax, LOWACC); /// The halfwidth of the maximum zmax, to calculate accel len
  float pow2            = pow(2 , round(log2(width)) );
  uint oAccelLen        = floor(pow2  - 2 - 2 * ACCEL_NUMBETWEEN * halfwidth);

  return oAccelLen;
}

/** Calculate the step size from a width if the width is < 100 it is skate to be the closest power of two  .
 *
 * @param width
 * @param zmax
 * @return
 */
uint calcAccellen(int width, int zmax)
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
int initKernel(cuFFdotBatch* kernel, cuFFdotBatch* master, int numharmstages, int zmax, fftInfo* fftinf, int device, int noBatches, int noSteps, int width, float*  powcut, long long*  numindep, int flags = 0, int outType = CU_FULLCAND, void* outData = NULL)
{
  nvtxRangePush("initKernel");

  size_t free, total;             /// GPU memory
  int noInStack[MAX_HARM_NO];
  int noHarms         = (1 << (numharmstages - 1) );
  int prevWidth       = 0;
  int noStacks        = 0;
  int major           = 0;
  int minor           = 0;
  noInStack[0]        = 0;
  size_t totSize      = 0;        /// Total size (in bytes) of all the data need by a family (ie one step) excluding FFT temporary
  size_t fffTotSize   = 0;        /// Total size (in bytes) of FFT temporary memory

  CUDA_SAFE_CALL(cudaGetLastError(), "Error entering initKernel.");

  FOLD // See if we can use the cuda device  .
  {
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
      return(0);
    }
    else
    {
      cudaDeviceProp deviceProp;
      CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, device), "Failed to get device properties device using cudaGetDeviceProperties");
      printf("\nInitializing GPU %i (%s)\n",device,deviceProp.name);

      major = deviceProp.major;
      minor = deviceProp.minor;
    }

    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  }

  FOLD // First determine how many stacks and how many plains in each stack  .
  {
    // Allocate and zero
    memset(kernel, 0, sizeof(cuFFdotBatch));

    if (master != NULL )  // Copy all pointers and sizes from master. All non global pointers must be overwritten.
      memcpy(kernel,  master,  sizeof(cuFFdotBatch));

    // Allocate memory
    kernel->hInfos  = (cuHarmInfo*) malloc(noHarms * sizeof(cuHarmInfo));
    kernel->kernels = (cuKernel*)   malloc(noHarms * sizeof(cuKernel));

    if ( master == NULL ) 	// Calculate details for the batch  .
    {
      // Zero memory for kernels and harmonics
      memset(kernel->hInfos,  0, noHarms * sizeof(cuHarmInfo));
      memset(kernel->kernels, 0, noHarms * sizeof(cuKernel));

      FOLD // Determine accellen and step size  .
      {
        kernel->accelLen = calcAccellen(width,zmax);

        if ( kernel->accelLen < 100 )
        {
          fprintf(stderr,"ERROR: With a width of %i, the step-size would be %i and this is too small, try with a wider width or lower z-max.\n", width, kernel->accelLen);
          return(1);
        }
        else
        {
          float fftLen      = calc_fftlen3(1, zmax, kernel->accelLen);
          int   oAccelLen   = optAccellen(fftLen, zmax);
          float ratio       = kernel->accelLen/float(oAccelLen);

          printf("• Using max FFT length of %.0f and thus ", fftLen);

          if ( ratio < 0.95 )
          {
            printf(" an non-optimal step-size of %i.\n", kernel->accelLen);
            if ( width > 100 )
            {
              int K              = round(fftLen/1000.0);
              fprintf(stderr,"    WARNING: Using manual width\\step-size is not advised rather set width to one of 2 4 8 46 32.\n    For a zmax of %i using %iK FFTs the optimal step-size is %i.\n", zmax, K, oAccelLen);
            }
          }
          else
          {
            printf(" an optimal step-size of %i.\n", kernel->accelLen);
          }
        }
      }

      // Set some harmonic related values
      for (int i = noHarms; i > 0; i--)
      {
        int idx = noHarms-i;
        kernel->hInfos[idx].harmFrac    = (i) / (double)noHarms;
        kernel->hInfos[idx].zmax        = calc_required_z(kernel->hInfos[idx].harmFrac, zmax);
        kernel->hInfos[idx].height      = (kernel->hInfos[idx].zmax / ACCEL_DZ) * 2 + 1;
        kernel->hInfos[idx].halfWidth   = z_resp_halfwidth(kernel->hInfos[idx].zmax, LOWACC);
        kernel->hInfos[idx].width       = calc_fftlen3(kernel->hInfos[idx].harmFrac, kernel->hInfos[0].zmax, kernel->accelLen);
        kernel->hInfos[idx].stackNo     = noStacks;

        if ( prevWidth != kernel->hInfos[idx].width )
        {
          noStacks++;
          noInStack[noStacks - 1]      = 0;
          prevWidth                    = kernel->hInfos[idx].width;
        }

        noInStack[noStacks - 1]++;
      }

      kernel->noHarms                   = noHarms;
      kernel->noHarmStages              = numharmstages;
      kernel->noStacks                  = noStacks;
    }
    else                    // Copy details from the master batch  .
    {
      // Zero memory for kernels and harmonics
      memcpy(kernel->hInfos,  master->hInfos,  noHarms * sizeof(cuHarmInfo));
      memcpy(kernel->kernels, master->kernels, noHarms * sizeof(cuKernel));
    }

    // Set some parameters
    kernel->device  = device;
    cuCtxGetCurrent ( &kernel->pctx );
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
      //Set up stacks
      kernel->stacks = (cuFfdotStack*) malloc(kernel->noStacks* sizeof(cuFfdotStack));

      if ( master == NULL )
        memset(kernel->stacks, 0, kernel->noStacks * sizeof(cuFfdotStack));
      else
        memcpy(kernel->stacks, master->stacks, kernel->noStacks * sizeof(cuFfdotStack));
    }
  }

  FOLD // Set up the basic details of all the harmonics and Calculate the stride  .
  {
    if ( master == NULL )
    {
      FOLD // Set up the basic details of all the harmonics  .
      {
        // Calculate the stage order of the harmonics
        int harmtosum;
        int i = 0;
        for (int stage = 0; stage < numharmstages; stage++)
        {
          harmtosum = 1 << stage;
          for (int harm = 1; harm <= harmtosum; harm += 2, i++)
          {
            float harmFrac                  = 1-harm/ float(harmtosum);
            int idx                         = round(harmFrac*noHarms);
            kernel->hInfos[idx].stageOrder  = i;
            kernel->pIdx[i]                 = idx;
          }
        }

        kernel->flag = flags;

        // Multi-step data layout method  .
        if ( !(kernel->flag & FLAG_ITLV_ALL ) )
        {
          kernel->flag |= FLAG_ITLV_ROW ;          //  FLAG_ITLV_ROW   or    FLAG_ITLV_PLN
        }

        kernel->cndType = outType;
      }

      FOLD // Calculate the stride of all the stacks (by allocating temporary memory)  .
      {
        int prev                = 0;
        kernel->inpDataSize     = 0;
        kernel->kerDataSize     = 0;
        kernel->plnDataSize     = 0;
        kernel->pwrDataSize     = 0;

        for (int i = 0; i< kernel->noStacks; i++)           // Loop through Stacks  .
        {
          cuFfdotStack* cStack  = &kernel->stacks[i];
          cStack->height        = 0;
          cStack->noInStack     = noInStack[i];
          cStack->startIdx      = prev;
          cStack->harmInf       = &kernel->hInfos[cStack->startIdx];
          cStack->kernels       = &kernel->kernels[cStack->startIdx];
          cStack->width         = cStack->harmInf->width;
          cStack->kerHeigth     = cStack->harmInf->height;

          for (int j = 0; j < cStack->noInStack; j++)
          {
            cStack->startZ[j]   = cStack->height;
            cStack->height     += cStack->harmInf[j].height;
            cStack->zUp[j]      = (cStack->kerHeigth - cStack->harmInf[j].height) / 2.0 ;
          }

          for (int j = 0; j < cStack->noInStack; j++)
          {
            cStack->zDn[j]      = ( cStack->kerHeigth ) - cStack->zUp[cStack->noInStack - 1 - j ];
          }


          FOLD // Allocate temporary device memory to asses input stride  .
          {
            CUDA_SAFE_CALL(cudaMallocPitch(&cStack->d_kerData, &cStack->strideCmplx, cStack->width * sizeof(cufftComplex), cStack->height), "Failed to allocate device memory for kernel stack.");
            CUDA_SAFE_CALL(cudaGetLastError(), "Allocating GPU memory to asses kernel stride.");

            kernel->inpDataSize     += cStack->strideCmplx * cStack->noInStack;           // At this point stride is still in bytes
            kernel->kerDataSize     += cStack->strideCmplx * cStack->kerHeigth;           // At this point stride is still in bytes

            CUDA_SAFE_CALL(cudaFree(cStack->d_kerData), "Failed to free CUDA memory.");
            CUDA_SAFE_CALL(cudaGetLastError(), "Freeing GPU memory.");
          }

          FOLD // Allocate temporary device memory to asses plain data stride  .
          {
            kernel->plnDataSize     += cStack->strideCmplx * cStack->height;              // At this point stride is still in bytes

            if ( kernel->flag & FLAG_CNV_CB_OUT )
            {
              CUDA_SAFE_CALL(cudaMallocPitch(&cStack->d_plainPowers, &cStack->stridePwrs, cStack->width * sizeof(float), cStack->kerHeigth), "Failed to allocate device memory for kernel stack.");
              CUDA_SAFE_CALL(cudaGetLastError(), "Allocating GPU memory to asses plain stride.");

              CUDA_SAFE_CALL(cudaFree(cStack->d_plainPowers), "Failed to free CUDA memory.");
              CUDA_SAFE_CALL(cudaGetLastError(), "Freeing GPU memory.");

              kernel->pwrDataSize    += cStack->stridePwrs * cStack->height;           // At this point stride is still in bytes
              cStack->stridePwrs     /= sizeof(float);
            }
            cStack->strideCmplx       /= sizeof(cufftComplex);                         // Set stride to number of complex numbers rather that bytes

          }
          prev                      += cStack->noInStack;
        }
      }
    }
    else
    {
      // Set up the pointers of each stack
      for (int i = 0; i< kernel->noStacks; i++)
      {
        cuFfdotStack* cStack              = &kernel->stacks[i];
        cStack->kernels                   = &kernel->kernels[cStack->startIdx];
        cStack->harmInf                   = &kernel->hInfos[cStack->startIdx];
      }
    }
  }

  FOLD // Allocate device memory for all the kernels data  .
  {
    CUDA_SAFE_CALL(cudaMemGetInfo ( &free, &total ), "Getting Device memory information");

    if ( kernel->kerDataSize > free )
    {
      fprintf(stderr, "ERROR: Not enough device memory for GPU convolution kernels. There is only %.2f MB free and you need %.2f MB \n", free / 1048576.0, kernel->kerDataSize / 1048576.0 );
      exit(EXIT_FAILURE);
    }
    else
    {
      CUDA_SAFE_CALL(cudaMalloc((void **)&kernel->d_kerData, kernel->kerDataSize), "Failed to allocate device memory for kernel stack.");
      CUDA_SAFE_CALL(cudaGetLastError(), "CUDA Error allocation of device memory for kernel?.\n");
    }
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

        // Point the plain kernel data to the correct position in the "main" kernel
        int iDiff                     = cStack->kerHeigth - cStack->harmInf[j].height ;
        float fDiff                   = iDiff / 2.0;
        cStack->kernels[j].d_kerData  = &cStack->d_kerData[cStack->strideCmplx*(int)fDiff];
        cStack->kernels[j].harmInf    = &cStack->harmInf[j];
      }
      kerSiz                          += cStack->strideCmplx * cStack->kerHeigth;
    }
  }

  FOLD // Initialise the convolution kernels  .
  {
    if ( master == NULL )  	  // Create the kernels  .
    {
      // Run message
      CUDA_SAFE_CALL(cudaGetLastError(), "Error before creating GPU kernels");

      printf("• Generating GPU convolution kernels\n");

      int hh = 1;
      for (int i = 0; i < kernel->noStacks; i++)
      {
        cuFfdotStack* cStack = &kernel->stacks[i];

        printf("    Stack %i has %02i f-∂f plain(s) with Width: %5li,  Stride %5li,  Total Height: %6li,   Memory size: %7.1f MB \n", i, cStack->noInStack, cStack->width, cStack->strideCmplx, cStack->height, cStack->height*cStack->strideCmplx*sizeof(fcomplex)/1024.0/1024.0);

        // call the CUDA kernels
        // Only need one kernel per stack
        createStackKernel(cStack);

        for (int j = 0; j< cStack->noInStack; j++)
        {
          printf("     Harmonic %2i  Fraction: %5.3f   Z-Max: %4i   Half Width: %4i  ", hh, cStack->harmInf[j].harmFrac, cStack->harmInf[j].zmax, cStack->harmInf[j].halfWidth );
          if ( j == 0 )
          {
            printf("Convolution kernel created: %7.1f MB \n", cStack->harmInf[j].height*cStack->strideCmplx*sizeof(fcomplex)/1024.0/1024.0);
          }
          else
          {
            printf("\n");
          }
          hh++;
        }
      }

      FOLD // FFT the kernels  .
      {
        printf("   FFT'ing the kernels ");
        cufftHandle plnPlan;

        //printf("noStacks %i\n", kernel->noStacks);

        for (int i = 0; i < kernel->noStacks; i++)
        {
          cuFfdotStack* cStack = &kernel->stacks[i];

          FOLD // Create the plan  .
          {
            size_t fftSize        = 0;

            int n[]             = {cStack->width};
            int inembed[]       = {cStack->strideCmplx* sizeof(fcomplexcu)};
            int istride         = 1;
            int idist           = cStack->strideCmplx;
            int onembed[]       = {cStack->strideCmplx* sizeof(fcomplexcu)};
            int ostride         = 1;
            int odist           = cStack->strideCmplx;
            int height;

            height = cStack->kerHeigth;

            //printf("cufftCreate %i\n", i);
            cufftCreate(&plnPlan);
            //printf("cufftCreate Done %i\n", i);

            CUFFT_SAFE_CALL(cufftMakePlanMany(plnPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, height,    &fftSize), "Creating plan for complex data of stack.");
            fffTotSize += fftSize;

            CUDA_SAFE_CALL(cudaGetLastError(), "Creating FFT plans for the stacks.");
          }

          FOLD // Call the plan  .
          {
            //printf("cufftExecC2C %i\n", i);
            CUFFT_SAFE_CALL(cufftExecC2C(plnPlan, (cufftComplex *) cStack->d_kerData, (cufftComplex *) cStack->d_kerData, CUFFT_FORWARD),"FFT'ing the kernel data");
            //printf("cufftExecC2C Done %i\n", i);

            printf(".");
            std::cout.flush();

            CUDA_SAFE_CALL(cudaGetLastError(), "FFT'ing the convolution kernels.");
          }

          FOLD // Destroy the plan  .
          {
            //printf("cufftDestroy %i\n", i);
            CUFFT_SAFE_CALL(cufftDestroy(plnPlan), "Destroying plan for complex data of stack.");
            //printf("cufftDestroy Done %i\n", i);

            CUDA_SAFE_CALL(cudaGetLastError(), "Destroying the plan.");
          }
        }

        CUDA_SAFE_CALL(cudaGetLastError(), "FFT'ing the convolution kernels.");
        printf("\n");
      }

      printf("  Done generating GPU convolution kernels.\n");
    }
    else                      // Copy kernels from master device
    {
      printf("• Copying convolution kernels from device %i.\n", master->device);
      CUDA_SAFE_CALL(cudaMemcpyPeer(kernel->d_kerData, kernel->device, master->d_kerData, master->device, master->kerDataSize ), "Copying convolution kernels between devices.");
    }
  }

  FOLD // Decide how to handle input and output and allocate required memory  .
  {
    printf("• Examining GPU memory of device %2i:\n", kernel->device);

    ulong freeRam;          /// The amount if free host memory
    int retSZ   = 0;        /// The size in byte of the returned data
    int candSZ  = 0;        /// The size in byte of the candidates
    int noRets;             /// The number of candidates return per family (one step)
    ulong hostC    = 0;     /// The size in bytes of device memory used for candidates

    if ( master == NULL )   // Calculate the search size in bins  .
    {
      int minR              = floor ( fftinf->rlo / (double)noHarms - kernel->hInfos[0].halfWidth );
      int maxR              = ceil  ( fftinf->rhi  + kernel->hInfos[0].halfWidth );

      searchScale* SrchSz   = new searchScale;
      kernel->SrchSz        = SrchSz;

      SrchSz->searchRLow    = fftinf->rlo / (double)noHarms;
      SrchSz->searchRHigh   = fftinf->rhi;
      SrchSz->rLow          = minR;
      SrchSz->rHigh         = maxR;
      SrchSz->noInpR        = maxR - minR  ;  /// The number of input data points

      if ( kernel->flag  & FLAG_STORE_EXP )
      {
        SrchSz->noOutpR     = ceil( (SrchSz->searchRHigh - SrchSz->searchRLow)/ACCEL_DR );
      }
      else
      {
        SrchSz->noOutpR     = ceil(SrchSz->searchRHigh - SrchSz->searchRLow);
      }

      if ( (kernel->flag & FLAG_STORE_ALL) && !( kernel->flag  & FLAG_RETURN_ALL) )
      {
        printf("   Storing all results implies returning all results so adding FLAG_RETURN_ALL to flags!\n");
        kernel->flag  |= FLAG_RETURN_ALL;
      }
    }

    FOLD // Calculate candidate type  .
    {
      kernel->retType = kernel->cndType;

      if      (kernel->cndType == CU_NONE     )
      {
        fprintf(stderr,"Warning: No output type specified in %s setting to full candidate info.\n",__FUNCTION__);
        kernel->cndType = CU_FULLCAND;
      }
      if      (kernel->cndType == CU_CMPLXF   )
      {
        candSZ = sizeof(fcomplexcu);
      }
      else if (kernel->cndType == CU_INT      )
      {
        candSZ = sizeof(int);
      }
      else if (kernel->cndType == CU_FLOAT    )
      {
        candSZ = sizeof(float);
      }
      else if (kernel->cndType == CU_POWERZ   )
      {
        candSZ = sizeof(accelcand2);
      }
      else if (kernel->cndType == CU_SMALCAND )
      {
        candSZ = sizeof(accelcandBasic);
      }
      else if (kernel->cndType == CU_FULLCAND || (kernel->cndType == CU_GSList) )
      {
        candSZ = sizeof(cand);
        kernel->retType = CU_SMALCAND;
      }
      else
      {
        fprintf(stderr,"ERROR: No output type specified in %s setting to full candidate info.\n",__FUNCTION__);
        kernel->cndType = CU_FULLCAND;
        candSZ = sizeof(cand);
        kernel->retType = CU_SMALCAND;
      }
    }

    FOLD // Calculate candidate return type and size  .
    {
      if      (kernel->retType == CU_CMPLXF   )
      {
        retSZ = sizeof(fcomplexcu);
      }
      else if (kernel->retType == CU_INT      )
      {
        retSZ = sizeof(int);
      }
      else if (kernel->retType == CU_FLOAT    )
      {
        retSZ = sizeof(float);
      }
      else if (kernel->retType == CU_POWERZ   )
      {
        retSZ = sizeof(accelcand2);
      }
      else if (kernel->retType == CU_SMALCAND )
      {
        retSZ = sizeof(accelcandBasic);
      }
      else if (kernel->retType == CU_FULLCAND )
      {
        retSZ = sizeof(cand);
      }
      else
      {
        fprintf(stderr,"ERROR: No output type specified in %s\n",__FUNCTION__);
      }

      noRets                = kernel->hInfos[0].width;  // NOTE: This could be accellen rather than width, but to allow greater flexibility keep it at width

      if ( kernel->flag & FLAG_RETURN_ALL )
        noRets *= numharmstages;

      kernel->retDataSize   = noRets*retSZ;
    }

    FOLD // Calculate batch size and number of steps and batches on this device  .
    {
      CUDA_SAFE_CALL(cudaMemGetInfo ( &free, &total ), "Getting Device memory information");
      freeRam = getFreeRamCU();
      printf("   There is a total of %.2f GiB of device memory of which there is %.2f GiB free and %.2f GiB free host memory.\n",total / 1073741824.0, (free )  / 1073741824.0, freeRam / 1073741824.0 );

      totSize              += kernel->plnDataSize + kernel->pwrDataSize + kernel->inpDataSize + kernel->retDataSize;
      fffTotSize            = kernel->plnDataSize + kernel->inpDataSize;

      float noKers2 = ( free ) / (double) ( fffTotSize + totSize * noBatches ) ;  // (fffTotSize * noKers2) for the CUFFT memory for FFT'ing the plain(s) and (totSize * noThreads * noKers2) for each thread(s) plan(s)

      printf("     Requested %i batches with on this device.\n", noBatches);
      if ( noKers2 > 1 )
      {
        if ( noSteps > floor(noKers2) )
        {
          printf("      Requested %i steps per batch, but with %i batches we can only do %.2f steps per batch. \n",noSteps, noBatches, noKers2 );
          noSteps = floor(noKers2);
        }

        if ( floor(noKers2) > noSteps + 1 && (noSteps < MAX_STEPS) )
          printf("       Note: requested %i steps per batch, you could do up to %.2f steps per batch. \n",noSteps, noKers2 );

        kernel->noSteps = noSteps;

        if ( kernel->noSteps > MAX_STEPS )
        {
          kernel->noSteps = MAX_STEPS;
          printf("      Trying to use more steps that the maximum number (%li) this code is compiled with.\n", kernel->noSteps );
        }
      }
      else
      {
        // TODO: check if we can do more than one step or set number of batches??
        float noKers3 = ( free ) / (double) ( fffTotSize + totSize ) ;
        noSteps = MIN(MAX_STEPS, floor(noKers3));

        printf("      There is not enough memory to crate %i batches with one plain each.\n", noBatches);
        printf("        Throttling to %.0f steps in 1 batch.\n", noKers3);
        kernel->noSteps = noSteps;
        noBatches = 1;
      }

      if ( noBatches <= 0 || noSteps <= 0 )
      {
        fprintf(stderr, "ERROR: Insufficient memory to make make any plains on this device.\n");
        CUDA_SAFE_CALL(cudaFree(kernel->d_kerData), "Failed to free device memory for kernel stack.");
        return 0;
      }
      printf("     Processing %i steps with each of the %i batch(s)\n", noSteps, noBatches );

      printf("    -----------------------------------------------\n" );
      printf("    Kernels      use: %5.2f GiB of device memory.\n", (kernel->kerDataSize) / 1073741824.0 );
      printf("    CUFFT       uses: %5.2f GiB of device memory.\n", (fffTotSize*kernel->noSteps) / 1073741824.0 );
      printf("    Each batch  uses: %5.2f GiB of device memory.\n", (totSize*kernel->noSteps) / 1073741824.0 );
      printf("               Using: %5.2f GiB of %.2f [%.2f%%] of GPU memory for search.\n", (kernel->kerDataSize + ( fffTotSize + totSize * noBatches )*kernel->noSteps ) / 1073741824.0, total / 1073741824.0, (kernel->kerDataSize + ( fffTotSize + totSize * noBatches )*kernel->noSteps ) / (float)total * 100.0f );
    }

    float fullRSize     = kernel->SrchSz->noOutpR * retSZ;                /// The full size of all data returned
    float fullCSize     = kernel->SrchSz->noOutpR * candSZ;               /// The full size of all candidate data

    if ( kernel->flag  & FLAG_RETURN_ALL )
      fullRSize *= numharmstages; // Store  candidates for all stages

    if ( kernel->flag  & FLAG_STORE_ALL )
      fullCSize *= numharmstages; // Store  candidates for all stages

    FOLD // DO a sanity check on flags  .
    {
      FOLD // How to handle input  .
      {
        if ( !( kernel->flag & CU_NORM_ALL ) )
          kernel->flag    |= CU_NORM_CPU;    // Prepare input data using CPU - Generally bets option, as CPU is "idle"

        if ( (kernel->flag & CU_INPT_CPU_FFT) & !(kernel->flag & CU_NORM_CPU))
        {
          fprintf(stderr, "WARNING: Using CPU FFT of the input data necessitate doing the normalisation on CPU.\n");
          kernel->flag &= ~CU_NORM_ALL;
          kernel->flag |= CU_NORM_CPU;
        }
      }

      FOLD // How to handle output  .
      {
        if ( !( kernel->flag & CU_CAND_ALL ) )
          kernel->flag    |= CU_CAND_ARR;    // Prepare input data using CPU - Generally bets option, as CPU is "idle"
      }

      FOLD // Convolution flags  .
      {
        if ( !(kernel->flag & FLAG_CNV_ALL ) )   // Default to convolution  .
        {
          float ver =  major + minor/10.0f;
          int noInp =  kernel->stacks->noInStack * kernel->noSteps ;

          if ( ver > 3.0 )
          {
            // Lots of registers per thread so 4.2 is good
            kernel->flag |= FLAG_CNV_42;
          }
          else
          {
            // We have less registers per thread
            if ( noInp <= 20 )
            {
              kernel->flag |= FLAG_CNV_42;
            }
            else
            {
              if ( kernel->noSteps <= 4 )
                kernel->flag |= FLAG_CNV_43;
              else
                kernel->flag |= FLAG_CNV_41;
            }
          }
        }
      }
    }

    FOLD // Allocate global (device independent) host memory  .
    {
      // One set of global set of "candidates" for all devices
      if( kernel->flag | CU_CAND_ARR )
      {
        if ( master == NULL )
        {
          if ( outData == NULL )
          {
            freeRam  = getFreeRamCU();
            if ( fullCSize < freeRam*0.98 )
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

              kernel->flag &= ~CU_CAND_ARR;
              kernel->flag |= CU_CAND_LST;
            }
          }
          else
          {
            // This memory has already been allocated
            kernel->h_candidates = outData;
            memset(kernel->h_candidates, 0, fullCSize );
          }
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
  }

  FOLD // Stack specific events  .
  {
    char tmpStr[1024];

    for (int i = 0; i< kernel->noStacks; i++)
    {
      cuFfdotStack* cStack = &kernel->stacks[i];
      CUDA_SAFE_CALL(cudaStreamCreate(&cStack->fftIStream),"Creating CUDA stream for fft's");
      sprintf(tmpStr,"%i FFT Input %i Stack", device, i);
      nvtxNameCudaStreamA(cStack->fftIStream, tmpStr);
    }

    for (int i = 0; i< kernel->noStacks; i++)
    {
      cuFfdotStack* cStack = &kernel->stacks[i];
      CUDA_SAFE_CALL(cudaStreamCreate(&cStack->fftPStream),"Creating CUDA stream for fft's");
      sprintf(tmpStr,"%i FFT Plain %i Stack", device, i);
      nvtxNameCudaStreamA(cStack->fftPStream, tmpStr);
    }
  }

  FOLD // Create texture memory from kernels  .
  {
    if ( kernel->flag & FLAG_CNV_TEX )
    {
      cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);

      CUDA_SAFE_CALL(cudaGetLastError(), "CUDA Error creating texture from kernel data.");

      for (int i = 0; i< kernel->noStacks; i++)           // Loop through Stacks
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
        for (int j = 0; j< cStack->noInStack; j++)        // Loop through plains in stack
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
  }

  FOLD // Set constant memory values  .
  {
    setConstVals( kernel,  numharmstages, powcut, numindep );
    setConstVals_Fam_Order( kernel );                            // Constant values for convolve
    copyCUFFT_LD_CB(kernel);
  }

  FOLD // Create FFT plans, ( 1 - set per device )  .
  {
    if ( ( kernel->flag & CU_INPT_CPU_FFT ) && master == NULL)
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

        if (kernel->flag & CU_INPT_CPU_FFT )
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
  }

  printf("Done initialising GPU %i.\n",device);
  nvtxRangePop();

  return noBatches;
}

/** Free kernel data structure  .
 *
 * @param kernel
 * @param master
 */
void freeKernel(cuFFdotBatch* kernrl, cuFFdotBatch* master)
{
  FOLD // Allocate device memory for all the kernels data  .
  {
    CUDA_SAFE_CALL(cudaFree(kernrl->d_kerData), "Failed to allocate device memory for kernel stack.");
    CUDA_SAFE_CALL(cudaGetLastError(), "CUDA Error allocation of device memory for kernel?.\n");
  }

  FOLD // Create texture memory from kernels  .
  {
    if ( kernrl->flag & FLAG_CNV_TEX )
    {
      cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);

      CUDA_SAFE_CALL(cudaGetLastError(), "CUDA Error creating texture from kernel data.");

      for (int i = 0; i< kernrl->noStacks; i++)           // Loop through Stacks
      {
        cuFfdotStack* cStack = &kernrl->stacks[i];

        cudaDestroyTextureObject(cStack->kerDatTex);
        CUDA_SAFE_CALL(cudaGetLastError(), "CUDA Error creating texture from the stack of kernel data.");

        // Create the actual texture object
        for (int j = 0; j< cStack->noInStack; j++)        // Loop through plains in stack
        {
          cuKernel* cKer = &cStack->kernels[j];

          cudaDestroyTextureObject(cKer->kerDatTex);
          CUDA_SAFE_CALL(cudaGetLastError(), "CUDA Error creating texture from kernel data.");
        }
      }
    }
  }

  FOLD // Decide how to handle input and output and allocate required memory  .
  {
    if ( master == kernrl )
      free(kernrl->SrchSz);

    FOLD // Allocate global (device independent) host memory
    {
      // One set of global set of "candidates" for all devices
      if ( master == kernrl )
      {
        if( kernrl->flag | CU_CAND_ARR )
        {
          if ( kernrl->h_candidates )
          {
            free(kernrl->h_candidates);
          }
        }
      }
    }

    CUDA_SAFE_CALL(cudaGetLastError(), "Failed to create memory for candidate list or input data.");
  }

  FOLD // Create CUFFT plans, ( 1 - set per device )  .
  {
    for (int i = 0; i < kernrl->noStacks; i++)
    {
      cuFfdotStack* cStack  = &kernrl->stacks[i];
      CUFFT_SAFE_CALL(cufftDestroy(cStack->plnPlan), "Destroying plan for complex data of stack.");
      CUFFT_SAFE_CALL(cufftDestroy(cStack->inpPlan), "Destroying plan for complex data of stack.");
      CUDA_SAFE_CALL(cudaGetLastError(), "Creating FFT plans for the stacks.");
    }
  }

  FOLD // Allocate all the memory for the stack data structures  .
  {
    free(kernrl->stacks);
  }
}

/** Initialise the pointers of the plains data structures of a batch  .
 *
 * This assumes the stack pointers have already been setup
 *
 * @param batch
 */
void setPlainPointers(cuFFdotBatch* batch)
{
  for (int i = 0; i < batch->noStacks; i++)
  {
    // Set stack pointers
    cuFfdotStack* cStack  = &batch->stacks[i];

    for (int j = 0; j < cStack->noInStack; j++)
    {
      cuFFdot* cPlain           = &cStack->plains[j];

      cPlain->d_plainData       = &cStack->d_plainData[   cStack->startZ[j] * batch->noSteps * cStack->strideCmplx ];
      cPlain->d_plainPowers     = &cStack->d_plainPowers[ cStack->startZ[j] * batch->noSteps * cStack->stridePwrs ];
      cPlain->d_iData           = &cStack->d_iData[cStack->strideCmplx*j*batch->noSteps];
      cPlain->harmInf           = &cStack->harmInf[j];
      cPlain->kernel            = &cStack->kernels[j];
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
  int harm          = 0;            /// The harmonic index of the first plain the the stack

  for (int i = 0; i < batch->noStacks; i++) // Set the various pointers of the stacks  .
  {
    cuFfdotStack* cStack  = &batch->stacks[i];

    cStack->d_iData       = &batch->d_iData[idSiz];
    cStack->h_iData       = &batch->h_iData[idSiz];
    cStack->plains        = &batch->plains[harm];
    cStack->kernels       = &batch->kernels[harm];
    cStack->d_plainData   = &batch->d_plainData[cmplStart];
    cStack->d_plainPowers = &batch->d_plainPowers[pwrStart];

    // Increment the various values used for offset
    harm                 += cStack->noInStack;
    idSiz                += batch->noSteps * cStack->strideCmplx * cStack->noInStack;
    cmplStart            += cStack->height  * cStack->strideCmplx * batch->noSteps ;
    pwrStart             += cStack->height  * cStack->stridePwrs * batch->noSteps ;
  }
}

/** Initialise the pointers of the stacks and plains data structures of a batch  .
 *
 * This assumes the various memory blocks of the batch have been created
 *
 * @param batch
 */
void setBatchPointers(cuFFdotBatch* batch)
{
  // First initialise the various pointers of the stacks
  setStkPointers(batch);

  // Now initialise the various pointers of the plains
  setPlainPointers(batch);
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
  char tmpStr[1024];
  size_t free, total;

  FOLD // See if we can use the cuda device  .
  {
    int currentDevvice;
    CUDA_SAFE_CALL(cudaSetDevice(kernel->device), "ERROR: cudaSetDevice");
    CUDA_SAFE_CALL(cudaGetDevice(&currentDevvice), "Failed to get device using cudaGetDevice");
    if (currentDevvice != kernel->device)
    {
      fprintf(stderr, "ERROR: CUDA Device not set.\n");
      return 0;
    }
  }

  FOLD // Set up basic slack list parameters from the harmonics  .
  {
    // Copy the basic batch parameters
    memcpy(batch, kernel, sizeof(cuFFdotBatch));

    // Copy the actual stacks
    batch->stacks = (cuFfdotStack*) malloc(batch->noStacks  * sizeof(cuFfdotStack));
    memcpy(batch->stacks, kernel->stacks, batch->noStacks    * sizeof(cuFfdotStack));
  }

  FOLD // Allocate all device and host memory for the stacks  .
  {
    FOLD // Allocate page-locked host memory for input data
    {
      CUDA_SAFE_CALL(cudaMallocHost((void**) &batch->h_iData, batch->inpDataSize*batch->noSteps ), "Failed to create page-locked host memory plain input data." );

      if ( batch->flag & CU_NORM_CPU ) // Allocate memory for normalisation
        batch->h_powers = (float*) malloc(batch->hInfos[0].width * sizeof(float));
    }

    FOLD  // Allocate R value lists  .
    {
      rVals*    l;
      rVals**   ll;
      int oSet;

      l  = (rVals*)malloc(sizeof(rVals)*batch->noSteps*batch->noHarms*3);
      oSet = 0;

      ll = (rVals**)malloc(sizeof(rVals*)*batch->noSteps);
      for(int step = 0; step < batch->noSteps; step++)
      {
        ll[step] = &l[oSet];
        oSet+= batch->noHarms;
      }
      batch->rInput  = (rVals***)malloc(sizeof(rVals**));
      *batch->rInput = ll;

      ll = (rVals**)malloc(sizeof(rVals*)*batch->noSteps);
      for(int step = 0; step < batch->noSteps; step++)
      {
        ll[step] = &l[oSet];
        oSet+= batch->noHarms;
      }
      batch->rSearch  = (rVals***)malloc(sizeof(rVals**));
      *batch->rSearch = ll;

      ll = (rVals**)malloc(sizeof(rVals*)*batch->noSteps);
      for(int step = 0; step < batch->noSteps; step++)
      {
        ll[step] = &l[oSet];
        oSet+= batch->noHarms;
      }
      batch->rConvld  = (rVals***)malloc(sizeof(rVals**));
      *batch->rConvld = ll;
    }

    FOLD // Allocate device Memory for Plain Stack & input data (steps)  .
    {
      CUDA_SAFE_CALL(cudaMemGetInfo ( &free, &total ), "Getting Device memory information");

      if ( (batch->inpDataSize + batch->plnDataSize + batch->pwrDataSize ) * batch->noSteps > free )
      {
        // Not enough memory =(

        // NOTE: we could reduce noSteps for this stack, but all batches must be the same size to share the same CFFT plan

        printf("Not enough GPU memory to create any more stacks.\n");
        return 0;
      }
      else
      {
        // Allocate device memory
        CUDA_SAFE_CALL(cudaMalloc((void** )&batch->d_iData,          batch->inpDataSize*batch->noSteps ), "Failed to allocate device memory for kernel stack.");
        CUDA_SAFE_CALL(cudaMalloc((void** )&batch->d_plainData,      batch->plnDataSize*batch->noSteps ), "Failed to allocate device memory for kernel stack.");

        if ( batch->flag & FLAG_CNV_CB_OUT )
        {
          CUDA_SAFE_CALL(cudaMalloc((void** )&batch->d_plainPowers,  batch->pwrDataSize*batch->noSteps ), "Failed to allocate device memory for kernel stack.");
          //batch->d_plainPowers = (float*)batch->d_plainData; // We can just re-use the plain data <- UMMMMMMMMM? No we can't!!
        }
      }
    }

    FOLD // Allocate device & page-locked host memory for candidate  data  .
    {
      CUDA_SAFE_CALL(cudaMemGetInfo ( &free, &total ), "Getting Device memory information");

      FOLD // Allocate device memory  .
      {
        if ( batch->retDataSize*batch->noSteps > free )
        {
          // Not enough memory =(
          printf("Not enough GPU memory to create stacks.\n");
          return 0;
        }
        else
        {
          CUDA_SAFE_CALL(cudaMalloc((void** ) &batch->d_retData, batch->retDataSize*batch->noSteps ), "Failed to allocate device memory for return values.");
        }
      }

      FOLD // Allocate page-locked host memory to copy the candidates back to  .
      {
        CUDA_SAFE_CALL(cudaMallocHost((void**) &batch->h_retData, batch->retDataSize*batch->noSteps),"");
        memset(batch->h_retData, 0, batch->retDataSize*batch->noSteps );
      }
    }

    // Create the plains structures
    if ( batch->noHarms* sizeof(cuFFdot) > getFreeRamCU() )
    {
      fprintf(stderr, "ERROR: Not enough host memory for search.\n");
      return 0;
    }
    else
    {
      batch->plains = (cuFFdot*) malloc(batch->noHarms* sizeof(cuFFdot));
      memset(batch->plains, 0, batch->noHarms* sizeof(cuFFdot));
    }

    FOLD // Create timing arrays
    {
#ifdef TIMING
      batch->copyH2DTime  = (float*)malloc(batch->noStacks*sizeof(float));
      batch->normTime      = (float*)malloc(batch->noStacks*sizeof(float));
      batch->InpFFTTime   = (float*)malloc(batch->noStacks*sizeof(float));
      batch->convTime     = (float*)malloc(batch->noStacks*sizeof(float));
      batch->InvFFTTime   = (float*)malloc(batch->noStacks*sizeof(float));
      batch->searchTime   = (float*)malloc(batch->noStacks*sizeof(float));
      batch->resultTime   = (float*)malloc(batch->noStacks*sizeof(float));
      batch->copyD2HTime  = (float*)malloc(batch->noStacks*sizeof(float));

      memset(batch->copyH2DTime,  0,batch->noStacks*sizeof(float));
      memset(batch->normTime,      0,batch->noStacks*sizeof(float));
      memset(batch->InpFFTTime,   0,batch->noStacks*sizeof(float));
      memset(batch->convTime,     0,batch->noStacks*sizeof(float));
      memset(batch->InvFFTTime,   0,batch->noStacks*sizeof(float));
      memset(batch->searchTime,   0,batch->noStacks*sizeof(float));
      memset(batch->resultTime,   0,batch->noStacks*sizeof(float));
      memset(batch->copyD2HTime,  0,batch->noStacks*sizeof(float));
#endif
    }
  }

  FOLD // Set up the batch streams and events  .
  {
    FOLD // Create Streams  .
    {
      FOLD // Input streams  .
      {
        CUDA_SAFE_CALL(cudaStreamCreate(&batch->inpStream),"Creating input stream for batch.");
        sprintf(tmpStr,"%i.%i.0.0 batch input", batch->device, no);
        nvtxNameCudaStreamA(batch->inpStream, tmpStr);

        for (int i = 0; i< batch->noStacks; i++)
        {
          cuFfdotStack* cStack  = &batch->stacks[i];

          CUDA_SAFE_CALL(cudaStreamCreate(&cStack->inpStream), "Creating input data cnvlStream for stack");
          sprintf(tmpStr,"%i.%i.0.%i Stack Input", batch->device, no, i);
          nvtxNameCudaStreamA(cStack->inpStream, tmpStr);
        }
      }

      FOLD // Convolve streams  .
      {
        CUDA_SAFE_CALL(cudaStreamCreate(&batch->convStream),"Creating convolution stream for batch.");
        sprintf(tmpStr,"%i.%i.0.0 batch convolve", batch->device, no);
        nvtxNameCudaStreamA(batch->convStream, tmpStr);

        for (int i = 0; i< batch->noStacks; i++)
        {
          cuFfdotStack* cStack  = &batch->stacks[i];

          CUDA_SAFE_CALL(cudaStreamCreate(&cStack->cnvlStream), "Creating cnvlStream for stack");
          sprintf(tmpStr,"%i.%i.1.%i Stack Convolve", batch->device, no, i);
          nvtxNameCudaStreamA(cStack->cnvlStream, tmpStr);
        }
      }

      // Search stream
      CUDA_SAFE_CALL(cudaStreamCreate(&batch->strmSearch), "Creating strmSearch for batch.");
      sprintf(tmpStr,"%i.%i.2.0 batch search", batch->device, no);
      nvtxNameCudaStreamA(batch->strmSearch, tmpStr);
    }

    FOLD // Create Events  .
    {
      FOLD // Create batch events  .
      {
#ifdef TIMING
        CUDA_SAFE_CALL(cudaEventCreate(&batch->iDataCpyComp), "Creating input event iDataCpyComp.");
        CUDA_SAFE_CALL(cudaEventCreate(&batch->candCpyComp),  "Creating input event candCpyComp.");
        CUDA_SAFE_CALL(cudaEventCreate(&batch->normComp),     "Creating input event normComp.");
        CUDA_SAFE_CALL(cudaEventCreate(&batch->convComp),     "Creating input event convComp.");
        CUDA_SAFE_CALL(cudaEventCreate(&batch->searchComp),   "Creating input event searchComp.");
        CUDA_SAFE_CALL(cudaEventCreate(&batch->processComp),  "Creating input event processComp.");

        CUDA_SAFE_CALL(cudaEventCreate(&batch->iDataCpyInit), "Creating input event iDataCpyInit.");
        CUDA_SAFE_CALL(cudaEventCreate(&batch->candCpyInit),  "Creating input event candCpyInit.");
        CUDA_SAFE_CALL(cudaEventCreate(&batch->convInit),     "Creating input event convInit.");
        CUDA_SAFE_CALL(cudaEventCreate(&batch->searchInit),   "Creating input event searchInit.");

        //cudaEventRecord(batch->iDataCpyInit);
        //cudaEventRecord(batch->iDataCpyComp);
#else
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(&batch->iDataCpyComp,   cudaEventDisableTiming ), "Creating input event iDataCpyComp.");
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(&batch->candCpyComp,    cudaEventDisableTiming ), "Creating input event candCpyComp.");
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(&batch->normComp,       cudaEventDisableTiming ), "Creating input event normComp.");
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(&batch->convComp,       cudaEventDisableTiming ), "Creating input event searchComp.");
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
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->invFFTinit), 	"Creating inverse FFT initialisation event");
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->convInit), 		"Creating convolution initialisation event");

          // out events (with timing)
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->normComp),    "Creating input normalisation event");
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->prepComp), 		"Creating input data preparation complete event");
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->convComp), 		"Creating convolution complete event");
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->plnComp),    	"Creating convolution complete event");
#else
          // out events (without timing)
          CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->normComp, cudaEventDisableTiming), "Creating input data preparation complete event");
          CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->prepComp, cudaEventDisableTiming), "Creating input data preparation complete event");
          CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->convComp, cudaEventDisableTiming), "Creating convolution complete event");
          CUDA_SAFE_CALL(cudaEventCreateWithFlags(&cStack->plnComp,  cudaEventDisableTiming), "Creating complex plain creation complete event");
#endif
        }
      }
    }

    if ( 0 )
    {
      for (int i = 0; i< batch->noStacks; i++)
      {
        cuFfdotStack* cStack = &batch->stacks[i];

        cStack->fftIStream = cStack->inpStream;
        cStack->fftPStream = cStack->cnvlStream;
      }
    }

    CUDA_SAFE_CALL(cudaGetLastError(), "Creating streams and events for the batch.");
  }

  FOLD // Setup the pointers for the stacks and plains of this batch  .
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

  FOLD // Create textures for the f-∂f plains  .
  {
    if ( (batch->flag&FLAG_TEX_INTERP) && !( (batch->flag&FLAG_CNV_CB_OUT) && (batch->flag&FLAG_SAS_TEX) ) )
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
          cuFFdot* cPlain = &cStack->plains[j];

          if ( batch->flag & FLAG_CNV_CB_OUT ) // float input
          {
            if      ( batch->flag & FLAG_ITLV_ROW )
            {
              resDesc.res.pitch2D.height          = cPlain->harmInf->height;
              resDesc.res.pitch2D.width           = cPlain->harmInf->width * batch->noSteps;
              resDesc.res.pitch2D.pitchInBytes    = cStack->harmInf->width * batch->noSteps * sizeof(float);
              resDesc.res.pitch2D.devPtr          = cPlain->d_plainPowers;
            }
            else if ( batch->flag & FLAG_ITLV_PLN )
            {
              resDesc.res.pitch2D.height          = cPlain->harmInf->height * batch->noSteps ;
              resDesc.res.pitch2D.width           = cPlain->harmInf->width;
              resDesc.res.pitch2D.pitchInBytes    = cStack->harmInf->width * sizeof(float);
              resDesc.res.pitch2D.devPtr          = cPlain->d_plainPowers;
            }
            else
            {
              // Error
            }
          }
          else // Implies complex numbers
          {
            if      ( batch->flag & FLAG_ITLV_ROW )
            {
              resDesc.res.pitch2D.height          = cPlain->harmInf->height;
              resDesc.res.pitch2D.width           = cPlain->harmInf->width * batch->noSteps * 2;
              resDesc.res.pitch2D.pitchInBytes    = cStack->harmInf->width * batch->noSteps * 2 * sizeof(float);
              resDesc.res.pitch2D.devPtr          = cPlain->d_plainPowers;
            }
            else if ( batch->flag & FLAG_ITLV_PLN )
            {
              resDesc.res.pitch2D.height          = cPlain->harmInf->height * batch->noSteps ;
              resDesc.res.pitch2D.width           = cPlain->harmInf->width * 2;
              resDesc.res.pitch2D.pitchInBytes    = cStack->harmInf->width * 2 * sizeof(float);
              resDesc.res.pitch2D.devPtr          = cPlain->d_plainPowers;
            }
            else
            {
              // Error
            }
          }

          CUDA_SAFE_CALL(cudaCreateTextureObject(&cPlain->datTex, &resDesc, &texDesc, NULL), "Creating texture from the plain data.");
        }
      }
      CUDA_SAFE_CALL(cudaGetLastError(), "Creating textures from the plain data.");

    }
  }

  return batch->noSteps;
}

int setStackInfo(cuFFdotBatch* batch, stackInfo* h_inf, int offset)
{
  stackInfo* dcoeffs;
  cudaGetSymbolAddress((void **)&dcoeffs, STACKS );

  for (int i = 0; i < batch->noStacks; i++)
  {
    cuFfdotStack* cStack  = &batch->stacks[offset+i];
    stackInfo*    cInf    = &h_inf[offset+i];

    cInf->noSteps         = batch->noSteps;
    cInf->noPlains        = cStack->noInStack;
    cInf->famIdx          = cStack->startIdx;
    cInf->flag            = batch->flag;

    cInf->d_iData         = cStack->d_iData;
    cInf->d_plainData     = cStack->d_plainData;
    cInf->d_plainPowers   = cStack->d_plainPowers;

    // Set the pointer to constant memory
    cStack->stkIdx        = offset+i;

    //cStack->d_sInf        = &STACKS[offset+i];

    //cudaGetSymbolAddress((void **)&dcoeffs, (&STACKS)+offset+i );
    //cudaGetSymbolAddress((void **)&dcoeffs, STACKS );
    cStack->d_sInf        = dcoeffs + offset+i ;

    //cStack->d_sInf        = ((stackInfo*)dcoeffs) + offset+i ;
    int tmp = 0;
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
    for (int i = 0; i < batch->noHarms; i++)
    {
      height[i] = batch->hInfos[i].height;
      stride[i] = batch->hInfos[i].inpStride;
      width[i]  = batch->hInfos[i].width;
      kerPnt[i] = batch->kernels[i].d_kerData;

      if (batch->hInfos[i].width != batch->hInfos[i].inpStride )
      {
        fprintf(stderr,"ERROR: Width is not the same as stride, using width this may case errors in the convolution.\n");
      }
    }

    for (int i = batch->noHarms; i < MAX_HARM_NO; i++) // Zero the rest
    {
      height[i] = 0;
      stride[i] = 0;
    }

    cudaGetSymbolAddress((void **)&dcoeffs, HEIGHT_FAM_ORDER);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &height, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, STRIDE_FAM_ORDER);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &stride, MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, WIDTH_FAM_ORDER);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &width,  MAX_HARM_NO * sizeof(int), cudaMemcpyHostToDevice),      "Copying stages to device");

    cudaGetSymbolAddress((void **)&dcoeffs, KERNEL_FAM_ORDER);
    CUDA_SAFE_CALL(cudaMemcpy(dcoeffs, &kerPnt, MAX_HARM_NO * sizeof(fcomplexcu*), cudaMemcpyHostToDevice),      "Copying stages to device");
  }

  CUDA_SAFE_CALL(cudaGetLastError(), "Error preparing the constant memory values for the convolutions.");

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

/** Free batch data structure  .
 *
 * @param batch
 */
void freeBatch(cuFFdotBatch* batch)
{
  FOLD // Allocate all device and host memory for the stacks  .
  {
    FOLD // Allocate page-locked host memory for input data
    {
      CUDA_SAFE_CALL(cudaFreeHost(batch->h_iData ), "Failed to create page-locked host memory plain input data." );

      if ( batch->flag & CU_NORM_CPU ) // Allocate memory for normalisation
        free(batch->h_powers);
    }

    FOLD // Allocate device Memory for Plain Stack & input data (steps)  .
    {
      // Allocate device memory
      CUDA_SAFE_CALL(cudaFree(batch->d_iData ), "Failed to allocate device memory for kernel stack.");
      CUDA_SAFE_CALL(cudaFree(batch->d_plainData ), "Failed to allocate device memory for kernel stack.");

      if ( batch->flag & FLAG_CNV_CB_OUT )
      {
        CUDA_SAFE_CALL(cudaFree(batch->d_plainPowers), "Failed to allocate device memory for kernel stack.");
      }
    }

    FOLD // Allocate device & page-locked host memory for candidate  data  .
    {
      CUDA_SAFE_CALL(cudaFree(batch->d_retData     ), "Failed to allocate device memory for return values.");
      CUDA_SAFE_CALL(cudaFreeHost(batch->h_retData ),"");
    }

    // Create the plains structures
    free(batch->plains);

    FOLD // Create timing arrays  .
    {
#ifdef TIMING
      free(batch->copyH2DTime);
      free(batch->InpFFTTime);
      free(batch->convTime);
      free(batch->InvFFTTime);
      free(batch->searchTime);
      free(batch->copyD2HTime);
#endif
    }
  }

  FOLD // Create textures for the f-∂f plains  .
  {
    if ( batch->flag & FLAG_SAS_TEX )
    {

      for (int i = 0; i< batch->noStacks; i++)
      {
        cuFfdotStack* cStack = &batch->stacks[i];

        for (int j = 0; j< cStack->noInStack; j++)
        {
          cuFFdot* cPlain = &cStack->plains[j];

          CUDA_SAFE_CALL(cudaDestroyTextureObject(cPlain->datTex), "Creating texture from the plain data.");
        }
      }
      CUDA_SAFE_CALL(cudaGetLastError(), "Creating textures from the plain data.");
    }
  }

  //free(batch);
}

void drawPlainCmplx(fcomplexcu* ffdotPlain, char* name, int stride, int height)
{
  float *tmpp = (float*) malloc(stride * height * sizeof(fcomplexcu));
  //float DestS   = ffdotPlain->ffPowWidth*sizeof(float);
  //float SourceS = ffdotPlain->ffPowStride;
  CUDA_SAFE_CALL(cudaMemcpy2D(tmpp, stride * sizeof(fcomplexcu), ffdotPlain, stride * sizeof(fcomplexcu), stride * sizeof(fcomplexcu), height, cudaMemcpyDeviceToHost), "Failed to copy data from device to host");

  //draw2DArray(name, tmpp, stride*2, height);
  free(tmpp);
}

/** Cycle the arrays of r-values  .
 *
 * @param batch
 */
void cycleRlists(cuFFdotBatch* batch)
{
  rVals*** tmp    = batch->rSearch;

  batch->rSearch = batch->rConvld;
  batch->rConvld = batch->rInput;
  batch->rInput  = tmp;
}

void search_ffdot_batch_CU(cuFFdotBatch* batch, double* searchRLow, double* searchRHi, int norm_type, int search, fcomplexcu* fft, long long* numindep, GSList** cands)
{
  CUDA_SAFE_CALL(cudaGetLastError(), "Error entering search_ffdot_batch_CU.");

#ifdef STPMSG
  printf("  search_ffdot_batch_CU\n");
#endif

  FOLD // Initialise input data  .
  {
    initInput(batch, searchRLow, searchRHi, norm_type, fft);
  }

#ifdef SYNCHRONOUS

  FOLD // Convolve & inverse FFT  .
  {
    convolveBatch(batch);
  }

  FOLD // Sum & Search  .
  {
    sumAndSearch(batch, numindep, cands);
  }

#else

  FOLD // Sum & Search  .
  {
    sumAndSearch(batch, numindep, cands);
  }

  FOLD // Convolve & inverse FFT  .
  {
    convolveBatch(batch);
  }

#endif

#ifdef STPMSG
  printf("  Done (search_ffdot_batch_CU)\n");
#endif
}

void max_ffdot_planeCU(cuFFdotBatch* batch, double* searchRLow, double* searchRHi, int norm_type, fcomplexcu* fft, long long* numindep, float* powers)
{
  CUDA_SAFE_CALL(cudaGetLastError(), "Error entering ffdot_planeCU2.");

  FOLD // Initialise input data  .
  {
    initInput(batch, searchRLow, searchRHi, norm_type, fft);
  }

#ifdef SYNCHRONOUS

  FOLD // Convolve & inverse FFT  .
  {
    convolveBatch(batch);
  }

  FOLD // Sum & Max
  {
    sumAndMax(batch, numindep, powers);
  }

#else

  FOLD // Sum & Max
  {
    sumAndMax(batch, numindep, powers);
  }

  FOLD // Convolve & inverse FFT  .
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

void printCands(const char* fileName, GSList *cands)
{
  if ( cands == NULL  )
    return;

  GSList *tmp_list = cands ;

  FILE * myfile;                    /// The file being written to
  myfile = fopen ( fileName, "w" );

  if ( myfile == NULL )
    fprintf ( stderr, "ERROR: Unable to open log file %s\n", fileName );
  else
  {
    fprintf(myfile, "# ; r ; z ; sig ; power ; harm \n");
    int i = 0;

    while ( tmp_list->next )
    {
      fprintf(myfile, "%i ; %14.5f ; %14.2f ; %-7.4f ; %7.2f ; %i \n", i, ((accelcand *) (tmp_list->data))->r, ((accelcand *) (tmp_list->data))->z, ((accelcand *) (tmp_list->data))->sigma, ((accelcand *) (tmp_list->data))->power, ((accelcand *) (tmp_list->data))->numharm );
      tmp_list = tmp_list->next;
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

void setContext(cuFFdotBatch* batch)
{
  int dev;
  //printf("Setting device to %i \n", batch->device);
  CUDA_SAFE_CALL(cudaSetDevice(batch->device), "ERROR: cudaSetDevice");
  CUDA_SAFE_CALL(cudaGetDevice(&dev), "Failed to get device using cudaGetDevice");
  if ( dev != batch->device )
  {
    fprintf(stderr, "ERROR: CUDA Device not set.\n");
    exit(EXIT_FAILURE);
  }

  /*
  CUcontext pctx;
  cuCtxGetCurrent ( &pctx );
  if(pctx !=  stkList->pctx )
  {
    CUresult res = cuCtxSetCurrent(batch->pctx);
  }
   */

  //CUcontext pctx;
  //cuCtxGetCurrent ( &pctx );
  //printf("Thread %02i  Context %p \n", omp_get_thread_num(), pctx);
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
    if ( cmd->gpuC == 0 )  // NB: Note using gpuC == 0 requires a change in accelsearch_cmd every time clig is run!!!!
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

void readAccelDefalts(uint *flags)
{
  FILE *file;
  char fName[1024];
  sprintf(fName, "%s/lib/GPU_defaults.txt", getenv("PRESTO"));

  if ( file = fopen(fName, "r") )  // Read candidates from previous search  .
  {
    printf("\nReading GPU settings from %s\n",fName);

    char line[1024];
    int lineno = 0;

    char *rest;

    while (fgets(line, sizeof(line), file))
    {
      lineno++;

      if      ( strCom(line, "FLAG_ITLV_ROW" ) || strCom(line, "INTERLEAVE_ROW" ) ||  strCom(line, "IL_ROW" ) )
      {
        (*flags) &= ~FLAG_ITLV_ALL;
        (*flags) |= FLAG_ITLV_ROW;
      }
      else if ( strCom(line, "FLAG_ITLV_PLN" ) || strCom(line, "INTERLEAVE_PLN" ) || strCom(line, "INTERLEAVE_PLAIN" ) || strCom(line, "IL_PLN" ) )
      {
        (*flags) &= ~FLAG_ITLV_ALL;
        (*flags) |= FLAG_ITLV_PLN;
      }

      else if ( strCom(line, "CU_NORM_CPU" ) || strCom(line, "NORM_CPU" ) )
      {
        (*flags) &= ~CU_NORM_ALL;
        (*flags) |= CU_NORM_CPU;
      }
      else if ( strCom(line, "CU_NORM_GPU" ) || strCom(line, "NORM_GPU" ) )
      {
        (*flags) &= ~CU_NORM_ALL;
        (*flags) |= CU_NORM_GPU;
      }

      else if ( strCom(line, "CU_INPT_CPU_FFT" ) || strCom(line, "CPU_FFT") || strCom(line, "FFT_CPU" ) )
      {
        (*flags) &= ~CU_NORM_ALL;
        (*flags) |= CU_NORM_CPU;
        (*flags) |= CU_INPT_CPU_FFT;
      }
      else if ( strCom(line, "CU_INPT_GPU_FFT" ) || strCom(line, "GPU_FFT" ) || strCom(line, "FFT_GPU" ) )
      {
        (*flags) &= ~CU_INPT_CPU_FFT;
      }

      else if ( strCom(line, "FLAG_CNV_00" ) || strCom(line, "CV_00" ) )
      {
        (*flags) &= ~FLAG_CNV_ALL;
        (*flags) |= FLAG_CNV_00;
      }
      else if ( strCom(line, "FLAG_CNV_10" ) || strCom(line, "CV_10" ) )
      {
        (*flags) &= ~FLAG_CNV_ALL;
        (*flags) |= FLAG_CNV_10;
      }
      else if ( strCom(line, "FLAG_CNV_30" ) || strCom(line, "CV_30" ) )
      {
        (*flags) &= ~FLAG_CNV_ALL;
        (*flags) |= FLAG_CNV_30;
      }
      else if ( strCom(line, "FLAG_CNV_41" ) || strCom(line, "CV_41" ) )
      {
        (*flags) &= ~FLAG_CNV_ALL;
        (*flags) |= FLAG_CNV_41;
      }
      else if ( strCom(line, "FLAG_CNV_42" ) || strCom(line, "CV_42" ) )
      {
        (*flags) &= ~FLAG_CNV_ALL;
        (*flags) |= FLAG_CNV_42;
      }
      else if ( strCom(line, "FLAG_CNV_43" ) || strCom(line, "CV_43" ) )
      {
        (*flags) &= ~FLAG_CNV_ALL;
        (*flags) |= FLAG_CNV_43;
      }
      else if ( strCom(line, "FLAG_CNV_50" ) || strCom(line, "CV_50" ) )
      {
        (*flags) &= ~FLAG_CNV_ALL;
        (*flags) |= FLAG_CNV_50;
      }

      else if ( strCom(line, "FLAG_CNV_TEX" ) )
      {
        (*flags) |= FLAG_CNV_TEX;
      }

      else if ( strCom(line, "FLAG_CNV_CB_IN" ) )
      {
        (*flags) |= FLAG_CNV_CB_IN;
      }

      else if ( strCom(line, "FLAG_CNV_CB_OUT" ) )
      {
        (*flags) |= FLAG_CNV_CB_OUT;
      }

      else if ( strCom(line, "FLAG_NO_CB" ) )
      {
        (*flags) &= ~FLAG_CNV_CB_IN;
        (*flags) &= ~FLAG_CNV_CB_OUT;
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

      else if ( strCom(line, "CU_CAND_ARR"  ) || strCom(line, "CAND_ARR"  ) )
      {
        (*flags) &= ~CU_CAND_ALL;
        (*flags) |= CU_CAND_ARR;
      }
      else if ( strCom(line, "CU_CAND_LST"  ) || strCom(line, "CAND_LST"  ) )
      {
        (*flags) &= ~CU_CAND_ALL;
        (*flags) |= CU_CAND_LST;
      }
      else if ( strCom(line, "CU_CAND_QUAD" ) || strCom(line, "CAND_QUAD" ) )
      {
        (*flags) &= ~CU_CAND_ALL;
        (*flags) |= CU_CAND_QUAD;
      }

      else if ( strCom(line, "FLAG_RETURN_ALL" ) )
      {
        (*flags) |= FLAG_RETURN_ALL;
      }
      else if ( strCom(line, "FLAG_RETURN_FINAL" ) )
      {
        (*flags) &= ~FLAG_RETURN_ALL;
      }

      else if ( strCom(line, "FLAG_STORE_ALL" ) )
      {
        (*flags) |= FLAG_STORE_ALL;
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
      else if ( strCom(line, "FLAG_RAND_3" ) || strCom(line, "RAND_3" ) )
      {
        (*flags) |= FLAG_RAND_3;
      }
      else if ( strCom(line, "FLAG_RAND_4" ) || strCom(line, "RAND_4" ) )
      {
        (*flags) |= FLAG_RAND_4;
      }

      else if ( strCom(line, "FLAG" ) || strCom(line, "CU_" ) )
      {
        int ll = strlen(line);
        line[ll-1] = 0;
        fprintf(stderr, "ERROR: Found unknown flag %s on line %i of %s.\n",line, lineno, fName);
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
    }

    fclose (file);
  }
  else
  {
    printf("Unable to read GPU accel settings from %s\n",fName);
  }
}

searchSpecs readSrchSpecs(Cmdline *cmd, accelobs* obs)
{
  searchSpecs sSpec;
  memset(&sSpec, 0, sizeof(sSpec));

  CUDA_SAFE_CALL(cudaGetLastError(), "Error entering readSrchSpecs.");

  // Defaults for accel search
  sSpec.flags         |= FLAG_RETURN_ALL ;
  sSpec.flags         |= CU_CAND_ARR ;
  sSpec.flags         |= FLAG_ITLV_ROW ;  //   FLAG_ITLV_ROW    FLAG_ITLV_PLN
  //sSpec.flags         |= FLAG_SAS_TEX ;
  //sSpec.flags         |= FLAG_TEX_INTERP ;
  //sSpec.flags         |= FLAG_CNV_CB_OUT ;

  readAccelDefalts(&sSpec.flags);

  sSpec.outType       = CU_FULLCAND ;

  sSpec.fftInf.fft    = obs->fft;
  sSpec.fftInf.nor    = obs->N;
  sSpec.fftInf.rlo    = obs->rlo;
  sSpec.fftInf.rhi    = obs->rhi;

  sSpec.noHarmStages  = obs->numharmstages;
  sSpec.zMax          = obs->zhi;
  sSpec.sigma         = cmd->sigma;
  sSpec.pWidth        = cmd->width;

  return sSpec;
}

/** Create convolution kernel and allocate memory for plains on all devices  .
 *
 * Create the kernels on the first device and then copy it to all others
 *
 * @param gSpec
 * @param sSpec
 * @param powcut
 * @param numindep
 * @return
 */
cuMemInfo* initCuAccel(gpuSpecs* gSpec, searchSpecs*  sSpec, float* powcut, long long* numindep)
{
  cuMemInfo* aInf = new cuMemInfo;
  memset(aInf, 0, sizeof(cuMemInfo));

  CUDA_SAFE_CALL(cudaGetLastError(), "Error entering initCuAccel.");

  FOLD // Create a kernel on each device  .
  {
    nvtxRangePush("Initialise Kernels");

    aInf->kernels = (cuFFdotBatch*)malloc(gSpec->noDevices*sizeof(cuFFdotBatch));

    int added;
    cuFFdotBatch* master = NULL;

    for ( int dev = 0 ; dev < gSpec->noDevices; dev++ ) // Loop over devices  .
    {
      added = initKernel(&aInf->kernels[aInf->noDevices], master, sSpec->noHarmStages, sSpec->zMax, &sSpec->fftInf, gSpec->devId[dev], gSpec->noDevBatches[dev], gSpec->noDevSteps[dev], sSpec->pWidth, powcut, numindep, sSpec->flags, sSpec->outType, sSpec->outData );

      if ( added && !master ) // This was the first batch so it is the master
      {
        master = &aInf->kernels[0];
      }

      if ( added )
      {
        aInf->noBatches += added;
        aInf->noDevices++;
      }
      else
      {
        gSpec->noDevBatches[dev] = 0;
        fprintf(stderr, "ERROR: failed to set up a kernel on device %i, trying to continue... \n", gSpec->devId[dev]);
      }
    }

    nvtxRangePop();

    if ( aInf->noDevices <= 0 ) // Check if we got ant devices
    {
      fprintf(stderr, "ERROR: Failed to set up a kernel on any device. Try -lsgpu to see what devices there are.\n");
      exit (EXIT_FAILURE);
    }


  }

  FOLD // Create plains for calculations  .
  {
    nvtxRangePush("Initialise Batches");

    aInf->noSteps       = 0;
    aInf->batches       = (cuFFdotBatch*)malloc(aInf->noBatches*sizeof(cuFFdotBatch));
    aInf->devNoStacks   = (int*)malloc(gSpec->noDevices*sizeof(int));
    aInf->h_stackInfo   = (stackInfo**)malloc(gSpec->noDevices*sizeof(stackInfo*));

    memset(aInf->devNoStacks,0,gSpec->noDevices*sizeof(int));

    int bNo = 0;
    int ker = 0;

    for ( int dev = 0 ; dev < gSpec->noDevices; dev++ ) // Loop over devices  .
    {
      int noSteps = 0;
      if ( gSpec->noDevBatches[dev] > 0 )
      {
        int firstBatch = bNo;

        for ( int batch = 0 ; batch < gSpec->noDevBatches[dev]; batch++ )
        {
          noSteps = initBatch(&aInf->batches[bNo], &aInf->kernels[ker], batch, gSpec->noDevBatches[dev]-1);

          if ( noSteps == 0 )
          {
            if ( batch == 0 )
            {
              fprintf(stderr, "ERROR: Failed to create at least one batch on device %i.\n", aInf->kernels[dev].device);
            }
            break;
          }
          else
          {
            aInf->noSteps           += noSteps;
            aInf->devNoStacks[dev]  += aInf->batches[bNo].noStacks;
            bNo++;
          }
        }

        int noStacks = aInf->devNoStacks[dev] ;
        if ( noStacks )
        {
          aInf->h_stackInfo[dev] = (stackInfo*)malloc(noStacks*sizeof(stackInfo));
          int idx = 0;

          // Set the values of the host data structures
          for (int batch = firstBatch; batch < bNo; batch++)
          {
            idx += setStackInfo(&aInf->batches[batch], aInf->h_stackInfo[dev], idx);
          }

          if ( idx != noStacks )
          {
            fprintf (stderr,"ERROR: in %s line %i, The number of steps on device do not match\n.",__FILE__, __LINE__);
          }
          else
          {
            setConstStkInfo(aInf->h_stackInfo[dev], idx);
          }
        }

        ker++;
      }
    }

    nvtxRangePop();
  }

  return aInf;
}

cuSearch* initCuSearch(searchSpecs* sSpec, gpuSpecs* gSpec, cuSearch* srch)
{
  bool same   = true;

  CUDA_SAFE_CALL(cudaGetLastError(), "Error entering initCuSearch.");

  if( srch )
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

    srch->noHarmStages  = sSpec->noHarmStages;
    srch->noHarms       = ( 1<<(srch->noHarmStages-1) );

    srch->pIdx          = (int*)malloc(srch->noHarms * sizeof(int));
    srch->powerCut      = (float*)malloc(srch->noHarmStages * sizeof(float));
    srch->numindep      = (long long*)malloc(srch->noHarmStages * sizeof(long long));
  }

  srch->sSpec         = sSpec;
  srch->gSpec         = gSpec;

  FOLD // Calculate power cutoff and number of independent values  .
  {
    if (sSpec->zMax % ACCEL_DZ)
      sSpec->zMax = (sSpec->zMax / ACCEL_DZ + 1) * ACCEL_DZ;
    int numz = (sSpec->zMax / ACCEL_DZ) * 2 + 1;
    for (int ii = 0; ii < srch->noHarmStages; ii++)
    {
      if (sSpec->zMax == 1)
        srch->numindep[ii] = (sSpec->fftInf.rhi - sSpec->fftInf.rlo) / srch->noHarms;
      else
      {
        srch->numindep[ii]  = (sSpec->fftInf.rhi - sSpec->fftInf.rlo) * (numz + 1) * (ACCEL_DZ / 6.95) / srch->noHarms;
        srch->powerCut[ii]  = power_for_sigma(sSpec->sigma, srch->noHarms, srch->numindep[ii]);
      }
    }
  }

  if ( !srch->mInf )
  {
    srch->mInf = initCuAccel(gSpec, sSpec, srch->powerCut, srch->numindep );
  }
  else
  {
    // TODO do a whole bunch of checks here!
  }

  return srch;
}

void freeCuAccel(cuMemInfo* aInf)
{
  FOLD // Free plains  .
  {
    for ( int batch = 0 ; batch < aInf->noBatches; batch++ )  // Batches
    {
      freeBatch(&aInf->batches[batch]);
    }
  }

  FOLD // Free kernels  .
  {
    for ( int dev = 0 ; dev < aInf->noDevices; dev++)  // Loop over devices
    {
      freeKernel(&aInf->kernels[dev], &aInf->kernels[0] );
    }
  }

  FOLD // Stack infos  .
  {
    for ( int dev = 0 ; dev < aInf->noDevices; dev++)  // Loop over devices
    {
      free(aInf->h_stackInfo[dev]);
    }
  }

  free(aInf->batches);
  aInf->batches = NULL;
  free(aInf->kernels);
  aInf->kernels = NULL;

  free(aInf->h_stackInfo);
  aInf->h_stackInfo = NULL;
  free(aInf->devNoStacks);
  aInf->devNoStacks = NULL;

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
  int maxxx = ( srch->sSpec->fftInf.rhi - srch->sSpec->fftInf.rlo ) / (float)( master->accelLen * ACCEL_DR ) ; /// The number of plains we can work with

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

    setContext(trdBatch) ;

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

    // Finish searching the plains, this is required because of the out of order asynchronous calls
    for ( int pln = 0 ; pln < 2; pln++ )
    {
      //max_ffdot_planeCU(trdBatch, startrs, lastrs, 1,(fcomplexcu*)fftinf->fft, numindep, powers );

      //trdBatch->mxSteps = rest;
    }
    printf("\n");
  }

  /*
  printf("Free plains \n");

  FOLD // Free plains
  {
    for ( int pln = 0 ; pln < nPlains; pln++ )  // Batches
    {
      freeBatch(plainsj[pln]);
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

void plotPlains(cuFFdotBatch* batch)
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
        //cuFFdot*      cPlain  = &batch->plains[harm];
        rVals*        rVal    = &((*batch->rInput)[step][harm]);

        for( int y = 0; y < cHInfo->height; y++ )
        {

          fcomplexcu *cmplxData;
          float *powers;

          if ( batch->flag & FLAG_ITLV_ROW )
          {
            cmplxData = &batch->d_plainData[(y*batch->noSteps + step)*cStack->strideCmplx ];
            powers    = &batch->d_plainPowers[((y*batch->noSteps + step)*cStack->stridePwrs + cHInfo->halfWidth * 2 ) ];
          }
          else if ( batch->flag & FLAG_ITLV_PLN )
          {
            cmplxData = &batch->d_plainData[   (y + step*cHInfo->height)*cStack->strideCmplx ];
            powers    = &batch->d_plainPowers[((y + step*cHInfo->height)*cStack->stridePwrs  + cHInfo->halfWidth * 2 ) ];
          }

          cmplxData += cHInfo->halfWidth*2;
          //CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[step][harm].getP(0,y), cmplxData, (cHInfo->width-2*2*cHInfo->halfWidth)*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
          //CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[step][harm].getP(0,y), cmplxData, (cPlain->numrs[step])*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
          CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[step][harm].getP(0,y), cmplxData, (rVal->numrs)*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
          if ( batch->flag & FLAG_CNV_CB_OUT )
          {
            //CUDA_SAFE_CALL(cudaMemcpyAsync(gpuPowers[step][harm].getP(0,y), powers, (cPlain->numrs[step])*sizeof(float),   cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
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

/*
void generatePlain(fftInfo fft, long long loBin, long long hiBin, int zMax, int noHarms)
{
  int width = hiBin - loBin;
  int gpuC = 0;
  int dev  = 0;
  int nplainsC = 5;
  int nplains[10];
  int nsteps[10];
  int gpu[10];
  nplains[0]=2;
  nsteps[0]=2;
  int numharmstages = twon_to_index(noHarms);
  noHarms = 1 << numharmstages ;
  long long numindep[numharmstages];
  float powc[numharmstages];

  int flags;

  gpu[0] = 1;

  cuFFdotBatch* kernels;             // List of stacks with the kernels, one for each device being used
  cuFFdotBatch* master   = NULL;     // The first kernel stack created
  int nPlains           = 0;        // The number of plains
  int noKers            = 0;        // Real number of kernels/devices being used

  fftInfo fftinf;
  fftinf.fft    = fft;
  fftinf.nor    = centerBin + width*2;
  fftinf.rlow   = centerBin - width;
  fftinf.rhi    = centerBin + width;

  int ww =  twon_to_index(width);

  int numz      = (zMax / ACCEL_DZ) * 2 + 1;

  for (int ii = 0; ii < numharmstages; ii++) // Calculate numindep
  {
    powc[ii] = 0;

    if (numz == 1)
      numindep[ii] = (fftinf.rhi - fftinf.rlow) / (1<<ii);
    else
    {
      numindep[ii] = (fftinf.rhi - fftinf.rlow) * (numz + 1) * (ACCEL_DZ / 6.95) / (1<<ii);
    }
  }

  flags = FLAG_CNV_CB_OUT ;

  gpu[0] = 0;
  kernels = new cuFFdotBatch;
  int added = initHarmonics(kernels, master, numharmstages, zMax, fftinf, 0, 1, ww, 1, powc, numindep, flags, CU_FLOAT, CU_FLOAT, NULL );
  cuFFdotBatch* batch = initStkList(kernels, 0, 0);

  cuFFdotBatch* trdStack = batch;
  double*  startrs = (double*)malloc(sizeof(double)*trdStack->noSteps);
  double*  lastrs  = (double*)malloc(sizeof(double)*trdStack->noSteps);

  startrs[0] = (centerBin - width) * noHarms ;
  lastrs[0] = startrs[0] + master->accelLen * ACCEL_DR - ACCEL_DR;

  max_ffdot_planeCU(trdStack, startrs, lastrs, 1, (fcomplexcu*)fftinf.fft, numindep, NULL);

  nvtxRangePop();
}
 */

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

void writeLogEntry(char* fname, accelobs *obs, cuSearch* cuSrch, long long prepTime, long long cupTime, long long gpuTime, long long optTime )
{
  //#ifdef CBL

  searchSpecs*  sSpec;              ///< Specifications of the search
  //gpuSpecs*     gSpec;              ///< Specifications of the GPU's to use

  cuMemInfo*    mInf;               ///< The allocated Device and host memory and data structures to create plains including the kernels

  cuFFdotBatch* batch;

  sSpec   = cuSrch->sSpec;
  //gSpec   = cuSrch->gSpec;
  mInf    = cuSrch->mInf;
  batch   = cuSrch->mInf->batches;

  char hostname[1024];
  gethostname(hostname, 1024);

  double noRR = sSpec->fftInf.rhi - sSpec->fftInf.rlo ;

  // get the current time
  time_t rawtime;
  tm * ptm;
  time ( &rawtime );
  ptm = gmtime ( &rawtime );

  Logger* cvsLog = new Logger(fname, 1);
  cvsLog->sedCsvDeliminator('\t');

  cvsLog->csvWrite("Obs N","#","%7.3f",obs->N*1e-6);
  cvsLog->csvWrite("R bins","#","%7.3f",noRR*1e-6);
  cvsLog->csvWrite("Harms","#","%2li",cuSrch->noHarms);
  cvsLog->csvWrite("Z max","#","%03i",sSpec->zMax);
  cvsLog->csvWrite("sigma","#","%4.2f",sSpec->sigma);
  cvsLog->csvWrite("ACCEL_LEN","#","%5i",ACCEL_USELEN);
  cvsLog->csvWrite("Width","#","%4i",sSpec->pWidth);
  cvsLog->csvWrite("Step Sz","#","%5i",batch->accelLen);
  cvsLog->csvWrite("Stride","#","%5i",batch->stacks->strideCmplx);
  cvsLog->csvWrite("Steps","#","%2i",batch->noSteps);
  cvsLog->csvWrite("Batches","#","%2i",mInf->noBatches);
  cvsLog->csvWrite("Devices","#","%2i",mInf->noDevices);

  cvsLog->csvWrite("Prep","s","%9.4f", prepTime * 1e-6);
  cvsLog->csvWrite("CPU", "s","%9.4f", cupTime * 1e-6);
  cvsLog->csvWrite("GPU", "s","%9.4f", gpuTime * 1e-6);
  cvsLog->csvWrite("Opt", "s","%9.4f", optTime * 1e-6);


#ifdef TIMING  // Advanced timing massage  .

  float copyH2DT  = 0;
  float InpNorm   = 0;
  float InpFFT    = 0;
  float convT     = 0;
  float InvFFT    = 0;
  float ss        = 0;
  float resultT   = 0;
  float copyD2HT  = 0;

  for (int batch = 0; batch < cuSrch->mInf->noBatches; batch++)
  {
    float l_copyH2DT  = 0;
    float l_InpNorm   = 0;
    float l_InpFFT    = 0;
    float l_convT     = 0;
    float l_InvFFT    = 0;
    float l_ss        = 0;
    float l_resultT   = 0;
    float l_copyD2HT  = 0;

    for (int stack = 0; stack < cuSrch->mInf->batches[batch].noStacks; stack++)
    {
      cuFFdotBatch*   batches = &cuSrch->mInf->batches[batch];

      l_copyH2DT  += batches->copyH2DTime[stack];
      l_InpNorm   += batches->normTime[stack];
      l_InpFFT    += batches->InpFFTTime[stack];
      l_convT     += batches->convTime[stack];
      l_InvFFT    += batches->InvFFTTime[stack];
      l_ss        += batches->searchTime[stack];
      l_resultT   += batches->resultTime[stack];
      l_copyD2HT  += batches->copyD2HTime[stack];
    }

    copyH2DT  += l_copyH2DT;
    InpNorm   += l_InpNorm;
    InpFFT    += l_InpFFT;
    convT     += l_convT;
    InvFFT    += l_InvFFT;
    ss        += l_ss;
    resultT   += l_resultT;
    copyD2HT  += l_copyD2HT;
  }

  cvsLog->csvWrite("copyH2D","ms","%12.6f", copyH2DT);
  cvsLog->csvWrite("InpNorm","ms","%12.6f", InpNorm);
  cvsLog->csvWrite("InpFFT","ms","%12.6f", InpFFT);
  cvsLog->csvWrite("Conv","ms","%12.6f", convT);
  cvsLog->csvWrite("InvFFT","ms","%12.6f", InvFFT);
  cvsLog->csvWrite("Sum & Srch","ms","%12.6f", ss);
  cvsLog->csvWrite("result","ms","%12.6f", resultT);
  cvsLog->csvWrite("copyD2H","ms","%12.6f", copyD2HT);

#endif

#ifdef TIMING
  cvsLog->csvWrite("TIMING","s","1");
#else
  cvsLog->csvWrite("TIMING","s","0");
#endif

#ifdef SYNCHRONOUS
  cvsLog->csvWrite("SYNC","s","1");
#else
  cvsLog->csvWrite("SYNC","s","0");
#endif

#ifdef DEBUG
  cvsLog->csvWrite("DEBUG","s","1");
#else
  cvsLog->csvWrite("DEBUG","s","0");
#endif

  cvsLog->csvWrite("Time","-","%04i/%02i/%02i %02i:%02i:%02i", 1900 + ptm->tm_year, ptm->tm_mon, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec );
  cvsLog->csvWrite("hostname","s","%s",hostname);

  FOLD  // Flags  .
  {
    uint flags = batch->flag;

    char flagStr[1024];

    strClear(flagStr);
    if ( flags & FLAG_ITLV_ROW )
      sprintf(flagStr, "%s", "ROW");
    if ( flags & FLAG_ITLV_PLN )
      sprintf(flagStr, "%s", "PLN");
    cvsLog->csvWrite("ITLV","flag","%s", flagStr);


    strClear(flagStr);
    if ( flags & CU_NORM_CPU )
      sprintf(flagStr, "%s", "CPU");
    if ( flags & CU_NORM_GPU )
      sprintf(flagStr, "%s", "GPU");
    cvsLog->csvWrite("NORM","flag","%s", flagStr);


    strClear(flagStr);
    if ( flags & CU_INPT_CPU_FFT )
      sprintf(flagStr, "%s", "CPU");
    else
      sprintf(flagStr, "%s", "GPU");
    cvsLog->csvWrite("INP FFT","flag","%s", flagStr);

    strClear(flagStr);
    if ( flags & FLAG_CNV_00 )
      sprintf(flagStr, "%s", "CV_00");
    if ( flags & FLAG_CNV_10 )
      sprintf(flagStr, "%s", "CV_10");
    if ( flags & FLAG_CNV_30 )
      sprintf(flagStr, "%s", "CV_30");
    if ( flags & FLAG_CNV_41 )
      sprintf(flagStr, "%s", "CV_41");
    if ( flags & FLAG_CNV_42 )
      sprintf(flagStr, "%s", "CV_42");
    if ( flags & FLAG_CNV_43 )
      sprintf(flagStr, "%s", "CV_43");
    if ( flags & FLAG_CNV_50 )
      sprintf(flagStr, "%s", "CV_50");
    cvsLog->csvWrite("CNV","flag","%s", flagStr);

    strClear(flagStr);
    if ( flags & FLAG_CNV_TEX )
      sprintf(flagStr, "%s", "1");
    else
      sprintf(flagStr, "%s", "0");
    cvsLog->csvWrite("CV_TEX","flag","%s", flagStr);

    strClear(flagStr);
    if ( flags & FLAG_CNV_CB_IN )
      sprintf(flagStr, "%s", "1");
    else
      sprintf(flagStr, "%s", "0");
    cvsLog->csvWrite("CB_IN","flag","%s", flagStr);

    strClear(flagStr);
    if ( flags & FLAG_CNV_CB_OUT )
      sprintf(flagStr, "%s", "1");
    else
      sprintf(flagStr, "%s", "0");
    cvsLog->csvWrite("CB_OUT","flag","%s", flagStr);

    strClear(flagStr);
    if ( flags & FLAG_SAS_TEX )
      sprintf(flagStr, "%s", "1");
    else
      sprintf(flagStr, "%s", "0");
    cvsLog->csvWrite("SAS_TEX","flag","%s", flagStr);

    strClear(flagStr);
    if ( flags & FLAG_TEX_INTERP )
      sprintf(flagStr, "%s", "1");
    else
      sprintf(flagStr, "%s", "0");
    cvsLog->csvWrite("SAS_INT","flag","%s", flagStr);

    strClear(flagStr);
    if ( flags & FLAG_SIG_GPU )
      sprintf(flagStr, "%s", "GPU");
    else
      sprintf(flagStr, "%s", "CPU");
    cvsLog->csvWrite("SIG","flag","%s", flagStr);

    strClear(flagStr);
    if ( flags & CU_CAND_ARR )
      sprintf(flagStr, "%s", "ARR");
    if ( flags & CU_CAND_LST )
      sprintf(flagStr, "%s", "LST");
    if ( flags & CU_CAND_QUAD )
      sprintf(flagStr, "%s", "QUAD");
    cvsLog->csvWrite("CAND","flag","%s", flagStr);

    strClear(flagStr);
    if ( flags & FLAG_RETURN_ALL )
      sprintf(flagStr, "%s", "ALL");
    else
      sprintf(flagStr, "%s", "FINAL");
    cvsLog->csvWrite("RET","flag","%s", flagStr);

    if ( flags & FLAG_STORE_ALL )
      sprintf(flagStr, "%s", "ALL");
    else
      sprintf(flagStr, "%s", "BST");
    cvsLog->csvWrite("STR","flag","%s", flagStr);

    if ( flags & FLAG_STORE_EXP )
      sprintf(flagStr, "%s", "EXP");
    else
      sprintf(flagStr, "%s", "CMP");
    cvsLog->csvWrite("STR","flag","%s", flagStr);


    strClear(flagStr);
    if ( flags & FLAG_RAND_1 )
      sprintf(flagStr, "%s", "1");
    cvsLog->csvWrite("RND1","flag","%s", flagStr);

    strClear(flagStr);
    if ( flags & FLAG_RAND_2 )
      sprintf(flagStr, "%s", "1");
    cvsLog->csvWrite("RND2","flag","%s", flagStr);

    strClear(flagStr);
    if ( flags & FLAG_RAND_3 )
      sprintf(flagStr, "%s", "1");
    cvsLog->csvWrite("RND3","flag","%s", flagStr);

    strClear(flagStr);
    if ( flags & FLAG_RAND_4 )
      sprintf(flagStr, "%s", "1");
    cvsLog->csvWrite("RND4","flag","%s", flagStr);
  }

  cvsLog->csvEndLine();

  //#endif
}
