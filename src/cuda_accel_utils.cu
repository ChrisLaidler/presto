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

#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"

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

void printData_cu(cuFFdotBatch* batch, const int FLAGS, int harmonic, int nX, int nY, int sX, int sY)
{
  cuFFdot* cPlain       = &batch->plains[harmonic];

  printfData<<<1,1,0,0>>>((float*)cPlain->d_iData, nX, nY, cPlain->harmInf->inpStride, sX, sY);
}

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
  noInStack[0]        = 0;
  size_t totSize      = 0;        /// Total size (in bytes) of all the data need by a family (ie one step) excluding FFT temporary
  size_t fffTotSize   = 0;        /// Total size (in bytes) of FFT temporary memory

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
            kernel->hInfos[idx].stageOrder   = i;
            kernel->pIdx[i]                  = idx;
          }
        }

        kernel->flag = flags;

        FOLD // How to handle input and output
        {
          // NOTE:  Generally CU_INPT_SINGLE_C and CU_OUTP_SINGLE are the best options and SINGLE cases generally use less memory as well

          if ( !( kernel->flag & CU_INPT_ALL ) )
            kernel->flag    |= CU_INPT_SINGLE_C;    // Prepare input data using CPU - Generally bets option, as CPU is "idle"

          if ( !( flags & CU_OUTP_ALL) )
            kernel->flag    |= CU_OUTP_SINGLE;      // Only get candidates from the current plain - This seams to be best in most cases
        }

        // Multi-step data layout method  .
        if ( !(kernel->flag & FLAG_STP_ALL ) )
        {
          kernel->flag |= FLAG_STP_ROW ;          //  FLAG_STP_ROW   or    FLAG_STP_PLN
        }

        FOLD // Convolution flags  .
        {
          //batch->flag |= FLAG_CNV_TEX;         // Use texture memory to access the kernel for convolution - May give advantage on pre-Fermi generation which we don't really care about
          kernel->flag |= FLAG_CNV_1KER;          // Create a minimal kernel (exploit overlap in stacks)  This should always be the case

          if ( !(kernel->flag & FLAG_CNV_ALL ) )   // Default to convolution
          {
            kernel->flag |= FLAG_CNV_STK;         //  FLAG_CNV_PLN   or   FLAG_CNV_STK   or   FLAG_CNV_FAM
          }
        }


        kernel->cndType = outType;
      }

      FOLD // Calculate the stride of all the stacks (by allocating temporary memory)  .
      {
        int prev               = 0;
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

          for (int j = 0; j < cStack->noInStack; j++)
          {
            cStack->startZ[j]   =  cStack->height;
            cStack->height     += cStack->harmInf[j].height;
            cStack->zUp[j]      =  (cStack->harmInf[0].height - cStack->harmInf[j].height) / 2.0 ;
          }

          for (int j = 0; j < cStack->noInStack; j++)
          {
            cStack->zDn[j]      = ( cStack->harmInf[0].height ) - cStack->zUp[cStack->noInStack - 1 - j ];
          }


          FOLD // Allocate temporary device memory to asses input stride  .
          {
            CUDA_SAFE_CALL(cudaMallocPitch(&cStack->d_kerData, &cStack->inpStride, cStack->width * sizeof(cufftComplex), cStack->harmInf[0].height), "Failed to allocate device memory for kernel stack.");
            CUDA_SAFE_CALL(cudaGetLastError(), "Allocating GPU memory to asses kernel stride.");

            kernel->inpDataSize     += cStack->inpStride * cStack->noInStack;          // At this point stride is still in bytes

            if ( kernel->flag & FLAG_CNV_1KER )
              kernel->kerDataSize   += cStack->inpStride * cStack->harmInf[0].height;  // At this point stride is still in bytes
            else
              kernel->kerDataSize   += cStack->inpStride * cStack->height;             // At this point stride is still in bytes

            CUDA_SAFE_CALL(cudaFree(cStack->d_kerData), "Failed to free CUDA memory.");
            CUDA_SAFE_CALL(cudaGetLastError(), "Freeing GPU memory.");
          }

          FOLD // Allocate temporary device memory to asses plain data stride  .
          {
            kernel->plnDataSize     += cStack->inpStride * cStack->height;            // At this point stride is still in bytes

            if ( kernel->flag & FLAG_CUFFTCB_OUT )
            {
              CUDA_SAFE_CALL(cudaMallocPitch(&cStack->d_plainPowers, &cStack->pwrStride, cStack->width * sizeof(float), cStack->harmInf[0].height), "Failed to allocate device memory for kernel stack.");
              CUDA_SAFE_CALL(cudaGetLastError(), "Allocating GPU memory to asses plain stride.");

              CUDA_SAFE_CALL(cudaFree(cStack->d_plainPowers), "Failed to free CUDA memory.");
              CUDA_SAFE_CALL(cudaGetLastError(), "Freeing GPU memory.");

              kernel->pwrDataSize    += cStack->pwrStride * cStack->height;           // At this point stride is still in bytes
              cStack->pwrStride     /= sizeof(float);
            }
            cStack->inpStride       /= sizeof(cufftComplex);                         // Set stride to number of complex numbers rather that bytes

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
      return (0);
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

    for (int i = 0; i< kernel->noStacks; i++)
    {
      cuFfdotStack* cStack              = &kernel->stacks[i];
      cStack->d_kerData                 = &kernel->d_kerData[kerSiz];

      // Set the stride
      for (int j = 0; j< cStack->noInStack; j++)
      {
        cStack->harmInf[j].inpStride    = cStack->inpStride;
        if ( kernel->flag & FLAG_CNV_1KER )
        {
          // Point the plain kernel data to the correct position in the "main" kernel
          int iDiff                     = cStack->harmInf[0].height - cStack->harmInf[j].height ;
          float fDiff                   = iDiff / 2.0;
          cStack->kernels[j].d_kerData  = &cStack->d_kerData[cStack->inpStride*(int)fDiff];
        }
        else
          cStack->kernels[j].d_kerData  = &cStack->d_kerData[cStack->startZ[j]*cStack->inpStride];

        cStack->kernels[j].harmInf      = &cStack->harmInf[j];
      }

      if ( kernel->flag & FLAG_CNV_1KER )
        kerSiz                          += cStack->inpStride * cStack->harmInf->height;
      else
        kerSiz                          += cStack->inpStride * cStack->height;
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

        printf("    Stack %i has %02i f-∂f plain(s) with Width: %5li,  Stride %5li,  Total Height: %6li,   Memory size: %7.1f MB \n", i, cStack->noInStack, cStack->width, cStack->inpStride, cStack->height, cStack->height*cStack->inpStride*sizeof(fcomplex)/1024.0/1024.0);

        // call the CUDA kernels
        if ( kernel->flag & FLAG_CNV_1KER )
        {
          // Only need one kernel per stack
          createStackKernel(cStack);
        }
        else
        {
          createStackKernels(cStack);
        }

        for (int j = 0; j< cStack->noInStack; j++)
        {
          printf("     Harmonic %2i  Fraction: %5.3f   Z-Max: %4i   Half Width: %4i  ", hh, cStack->harmInf[j].harmFrac, cStack->harmInf[j].zmax, cStack->harmInf[j].halfWidth );
          if ( kernel->flag & FLAG_CNV_1KER )
            if ( j == 0 )
              printf("Convolution kernel created: %7.1f MB \n", cStack->harmInf[j].height*cStack->inpStride*sizeof(fcomplex)/1024.0/1024.0);
            else
              printf("\n");
          else
            printf("Convolution kernel created: %7.1f MB \n", cStack->harmInf[j].height*cStack->inpStride*sizeof(fcomplex)/1024.0/1024.0);
          hh++;
        }
      }

      if ( DBG_KER01 )  // Print debug info  .
      {
        for (int i = 0; i< kernel->noHarms; i++)
        {
          printf("\nKernel pre FFT %i\n", i);
          //printfData<<<1,1,0,0>>>((float*)batch->kernels[i].d_kerData,10,5,batch->hInfos[i].stride*2);
          printData_cu(kernel, kernel->flag, i);
        }
      }

      FOLD // FFT the kernels  .
      {
        printf("   FFT'ing the kernels ");
        cufftHandle plnPlan;
        for (int i = 0; i < kernel->noStacks; i++)
        {
          cuFfdotStack* cStack = &kernel->stacks[i];

          FOLD // Create the plan
          {
            size_t fftSize        = 0;

            int n[]             = {cStack->width};
            int inembed[]       = {cStack->inpStride* sizeof(fcomplexcu)};
            int istride         = 1;
            int idist           = cStack->inpStride;
            int onembed[]       = {cStack->inpStride* sizeof(fcomplexcu)};
            int ostride         = 1;
            int odist           = cStack->inpStride;
            int height;

            if ( kernel->flag & FLAG_CNV_1KER )
              height = cStack->harmInf->height;
            else
              height = cStack->height;

            cufftCreate(&plnPlan);

            CUFFT_SAFE_CALL(cufftMakePlanMany(plnPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, height,    &fftSize), "Creating plan for complex data of stack.");
            fffTotSize += fftSize;

            CUDA_SAFE_CALL(cudaGetLastError(), "Creating FFT plans for the stacks.");
          }

          // Call the plan
          CUFFT_SAFE_CALL(cufftExecC2C(plnPlan, (cufftComplex *) cStack->d_kerData, (cufftComplex *) cStack->d_kerData, CUFFT_FORWARD),"FFT'ing the kernel data");
          printf(".");
          std::cout.flush();

          // Destroy the plan
          CUFFT_SAFE_CALL(cufftDestroy(plnPlan), "Destroying plan for complex data of stack.");
        }
        CUDA_SAFE_CALL(cudaGetLastError(), "FFT'ing the convolution kernels.");
        printf("\n");
      }

      printf("  Done generating GPU convolution kernels.\n");

      if ( DBG_KER02 )  	    // Print debug info  .
      {
        for (int i = 0; i< kernel->noHarms; i++)
        {
          printf("\nKernel post FFT %i\n", i);
          //printfData<<<1,1,0,0>>>((float*)batch->kernels[batch->pIdx[i]].d_kerData,10,5,batch->hInfos[batch->pIdx[i]].stride*2);
          printData_cu(kernel, kernel->flag, kernel->pIdx[i]);
          CUDA_SAFE_CALL(cudaStreamSynchronize(0),"Printing debug info");
        }
      }

      if ( DBG_PRNTKER02 ) 	  // Draw the kernel  .
      {
        /*
        char fname[1024];
        for (int i = 0; i< batch->noHarms; i++)
        {
          sprintf(fname, "./ker_%02i_GPU.png",i);
          drawPlainCmplx(batch->kernels[batch->pIdx[i]].d_kerData, fname, batch->hInfos[batch->pIdx[i]].inpStride, batch->hInfos[batch->pIdx[i]].height );
          CUDA_SAFE_CALL(cudaStreamSynchronize(0),"Printing debug info");
        }
         */
      }
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
    ulong deviceC  = 0;     /// The size in bytes of device memory used for candidates
    ulong hostC    = 0;     /// The size in bytes of device memory used for candidates

    if ( master == NULL )   // Calculate the search size in bins  .
    {
      int minR              = floor ( fftinf->rlo / (double)noHarms - kernel->hInfos[0].halfWidth );
      int maxR              = ceil  ( fftinf->rhi  + kernel->hInfos[0].halfWidth );

      searchScale* SrchSz   = new searchScale;
      kernel->SrchSz         = SrchSz;

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

      if      (kernel->cndType == CU_NONE   )
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

    FOLD // calculate batch size and number of steps and batches on this device  .
    {
      CUDA_SAFE_CALL(cudaMemGetInfo ( &free, &total ), "Getting Device memory information");
      freeRam = getFreeRamCU();
      printf("   There is a total of %.2f GiB of device memory of which there is %.2f GiB free and %.2f GiB free host memory.\n",total / 1073741824.0, (free )  / 1073741824.0, freeRam / 1073741824.0 );

      totSize              += kernel->plnDataSize + kernel->pwrDataSize + kernel->inpDataSize + kernel->retDataSize;
      fffTotSize            = kernel->plnDataSize + kernel->inpDataSize;

      float noKers2 = ( free ) / (double) ( fffTotSize + totSize * noBatches ) ;  // (fffTotSize * noKers2) for the CUFFT memory for FFT'ing the plain(s) and (totSize * noThreads * noKers2) for each thread(s) plan(s)

      printf("     Requested %i batches on this device.\n", noBatches);
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
        printf("      There is not be enough memory to do %i batches, throttling to 1 step per batch.\n", noBatches);
        kernel->noSteps = 1;                  // Default we have to do at least one step at a time
      }

      if ( noBatches <= 0 || noSteps <= 0 )
      {
        fprintf(stderr, "ERROR: Insufficient memory to make make any plains on this device.\n");
        CUDA_SAFE_CALL(cudaFree(kernel->d_kerData), "Failed to free device memory for kernel stack.");
        return 0;
      }
      printf("     Processing %i steps with each of the %i batch(s)\n", noSteps, noBatches );
      //batch->mxSteps = batch->noSteps;

      printf("    Kernels      use: %5.2f GiB of device memory.\n", (kernel->kerDataSize) / 1073741824.0 );
      printf("    CUFFT        use: %5.2f GiB of device memory.\n", (fffTotSize*kernel->noSteps) / 1073741824.0 );
      printf("    Each batch  uses: %5.2f GiB of device memory.\n", (totSize*kernel->noSteps) / 1073741824.0 );
      printf("               Using: %5.2f GiB of %.2f [%.2f%%] device memory for plains.\n", (kernel->kerDataSize + ( fffTotSize + totSize * noBatches )*kernel->noSteps ) / 1073741824.0, total / 1073741824.0, (kernel->kerDataSize + ( fffTotSize + totSize * noBatches )*kernel->noSteps ) / (float)total * 100.0f );
    }

    float fullISize     = kernel->SrchSz->noInpR  * sizeof(fcomplexcu);   /// The full size of relevant input data
    float fullRSize     = kernel->SrchSz->noOutpR * retSZ;                /// The full size of all data returned
    float fullCSize     = kernel->SrchSz->noOutpR * candSZ;               /// The full size of all candidate data
    float fullSem       = kernel->SrchSz->noOutpR * sizeof(uint);         /// size of semaphores

    if ( kernel->flag  & FLAG_RETURN_ALL )
      fullRSize *= numharmstages; // Store  candidates for all stages

    if ( kernel->flag  & FLAG_STORE_ALL )
      fullCSize *= numharmstages; // Store  candidates for all stages

    FOLD // Do sanity checks for input and output and adjust "down" if necessary  .
    {
      float remainigGPU   = free - fffTotSize*kernel->noSteps - totSize*kernel->noSteps*noBatches ;
      float remainingRAM  = freeRam;

      if ( kernel->flag & CU_INPT_DEVICE 	)
      {
        if ( fullISize > remainigGPU*0.98 )
        {
          fprintf(stderr, "WARNING: Requested to store all input data on device but there is insufficient space so changing to page locked memory instead.\n");
          kernel->flag ^= CU_INPT_DEVICE;
          kernel->flag |= CU_INPT_HOST;
        }
        else
        {
          // We can get all points on the device
          remainigGPU -= fullISize ;
        }
      }

      if ( kernel->flag & CU_INPT_HOST 	  )
      {
        if (fullISize > remainingRAM*0.98 )
        {
          fprintf(stderr, "WARNING: Requested to store all input data in page locked host memory but there is insufficient space, so changing to working on single stack at a time.\n");
          kernel->flag ^= CU_INPT_HOST;
          kernel->flag |= CU_INPT_SINGLE_C;
        }
        else
        {
          // We can get all points in ram
          remainingRAM -= fullISize ;
        }
      }

      if ( kernel->flag & CU_OUTP_DEVICE   )
      {
        if ( fullCSize > remainigGPU *0.98 )
        {
          if( master == NULL)
          {
            fprintf(stderr, "WARNING: Requested to store all candidates on device but there is insufficient space so changing to page locked memory instead.\n");
            kernel->flag ^= CU_OUTP_DEVICE;
            kernel->flag |= CU_OUTP_HOST;
          }
          else
          {
            fprintf(stderr, "ERROR: GPU %i has insufficient free memory to store all candidates on device.\n");
            return 0;
          }
        }
        else
        {
          remainigGPU -= fullRSize ;
        }
      }

      if ( ( kernel->flag & CU_OUTP_HOST 	) || ( kernel->flag & CU_OUTP_DEVICE ) )
      {
        if( master == NULL)
        {
          if ( fullCSize > remainingRAM *0.98 )
          {
            fprintf(stderr, "WARNING: Requested to store all candidates in page locked host memory but there is insufficient space, so changing to working on single stack at a time.\n");
            if ( kernel->flag & CU_OUTP_DEVICE  )
              fprintf(stderr, "         This is strange you appear to have enough GPU memory for to store all candidates but not enough host RAM.\n");
            kernel->flag ^= CU_OUTP_HOST;
            kernel->flag ^= CU_OUTP_DEVICE;
            kernel->flag |= CU_OUTP_SINGLE;
          }
          else
          {
            remainingRAM -= fullRSize ;
          }
        }
      }
    }

    FOLD // ALLOCATE device specific memory  .
    {
      if      ( kernel->flag & CU_INPT_DEVICE )
      {
        // Create and copy raw fft data to the device
        CUDA_SAFE_CALL(cudaMalloc((void** )&kernel->d_iData, fullISize), "Failed to allocate device memory for input raw FFT data.");
        CUDA_SAFE_CALL(cudaMemcpy(kernel->d_iData, &fftinf->fft[kernel->SrchSz->rLow], fullISize, cudaMemcpyHostToDevice), "Failed to copy raw FFT data to device.");
        deviceC += fullISize;
      }
      else if ( kernel->flag & CU_INPT_HOST   )
      {
        if ( master == NULL )
        {
          // Create page locked host memory and copy raw fft data - for the entire input data
          CUDA_SAFE_CALL(cudaMallocHost((void**) &kernel->h_iData, fullISize), "Failed to create page-locked host memory for entire input data." );
          deviceC+=fullISize;

          int start = 0;   /// Number of bins to zero pad the beginning
          if ( kernel->SrchSz->rLow < 0 ) // Zero pad if necessary
          {
            start = -kernel->SrchSz->rLow;
            memset(kernel->h_iData, 0, start*sizeof(fcomplex) );
          }

          // Copy input data to pagelocked memory
          memcpy(&kernel->h_iData[start], &fftinf->fft[kernel->SrchSz->rLow+start], (fullISize-start)*sizeof(fcomplex));
          hostC += fullISize;
        }
      }
      else if ( kernel->flag & CU_INPT_SINGLE )
      {
        // Nothing, each batch has its own input data already
      }
      else
      {
        fprintf(stderr, "ERROR: Undecided how to handle input data!\n");
        return 0;
      }

      if      ( kernel->flag & CU_OUTP_DEVICE )
      {
        // Create a candidate list
        CUDA_SAFE_CALL(cudaMalloc((void** )&kernel->d_retData, fullRSize), "Failed to allocate device memory for candidate list stack.");
        CUDA_SAFE_CALL(cudaMemset((void*)kernel->d_retData, 0, fullRSize ), "Failed to initialise  candidate list.");
        deviceC += fullRSize;

        // Create a semaphore list
        CUDA_SAFE_CALL(cudaMalloc((void** )&kernel->d_candSem, fullSem ), "Failed to allocate device memory for candidate semaphore list.");
        CUDA_SAFE_CALL(cudaMemset((void*)kernel->d_candSem, UINT_MAX, fullSem ), "Failed to initialise  semaphore list.");
        deviceC += fullSem;
      }
      else if ( ( kernel->flag & CU_OUTP_HOST ) || ( kernel->flag & CU_OUTP_DEVICE ) )
      {
        if ( master == NULL )
        {
          CUDA_SAFE_CALL(cudaMallocHost((void**) &kernel->h_retData, fullRSize), "Failed to create page-locked host memory for entire candidate list." );
          memset(kernel->h_retData, 0, fullRSize);
          hostC += fullRSize;
        }
      }
      else if ( kernel->flag & CU_OUTP_SINGLE )
      {
        // Nothing, each batch has its own return data already
      }
      else
      {
        fprintf(stderr, "ERROR: Undecided how to handle input data!");
        return 0;
      }
    }

    FOLD // Allocate global (device independent) host memory
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

              kernel->flag ^= CU_CAND_ARR;
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

    if ( deviceC + hostC )
    {
      printf("    Input and candidates use and additional:\n");
      if ( deviceC )
        printf("                      %5.2f GiB of device memory\n", deviceC / 1073741824.0 );
      if ( hostC )
        printf("                      %5.2f GiB of host   memory\n", hostC / 1073741824.0 );
    }

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
        resDesc.res.pitch2D.pitchInBytes  = cStack->inpStride * sizeof(fcomplex);

        if ( kernel->flag & FLAG_CNV_1KER )
          resDesc.res.pitch2D.height      = cStack->harmInf->height;
        else
          resDesc.res.pitch2D.height      = cStack->height;

        CUDA_SAFE_CALL(cudaCreateTextureObject(&cStack->kerDatTex, &resDesc, &texDesc, NULL), "Error Creating texture from kernel data.");

        CUDA_SAFE_CALL(cudaGetLastError(), "CUDA Error creating texture from the stack of kernel data.");

        // Create the actual texture object
        for (int j = 0; j< cStack->noInStack; j++)        // Loop through plains in stack
        {
          cuKernel* cKer = &cStack->kernels[j];

          resDesc.res.pitch2D.devPtr        = cKer->d_kerData;
          resDesc.res.pitch2D.height        = cKer->harmInf->height;
          resDesc.res.pitch2D.width         = cKer->harmInf->width;
          resDesc.res.pitch2D.pitchInBytes  = cStack->inpStride * sizeof(fcomplex);

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

  FOLD // Create CUFFT plans, ( 1 - set per device )  .
  {
    fffTotSize = 0;
    for (int i = 0; i < kernel->noStacks; i++)
    {
      cuFfdotStack* cStack  = &kernel->stacks[i];
      size_t fftSize        = 0;

      FOLD
      {
        int n[]             = {cStack->width};
        int inembed[]       = {cStack->inpStride* sizeof(fcomplexcu)};
        int istride         = 1;
        int idist           = cStack->inpStride;
        int onembed[]       = {cStack->inpStride* sizeof(fcomplexcu)};
        int ostride         = 1;
        int odist           = cStack->inpStride;

        cufftCreate(&cStack->plnPlan);
        cufftCreate(&cStack->inpPlan);

        CUFFT_SAFE_CALL(cufftMakePlanMany(cStack->plnPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, cStack->height*kernel->noSteps,    &fftSize), "Creating plan for complex data of stack.");
        fffTotSize += fftSize;

        CUFFT_SAFE_CALL(cufftMakePlanMany(cStack->inpPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, cStack->noInStack*kernel->noSteps, &fftSize), "Creating plan for input data of stack.");
        fffTotSize += fftSize;
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

    FOLD // ALLOCATE device specific memory  .
    {
      if      ( kernrl->flag & CU_INPT_DEVICE )
      {
        // Create and copy raw fft data to the device
        CUDA_SAFE_CALL(cudaFree(kernrl->d_iData), "Failed to allocate device memory for input raw FFT data.");
      }
      else if ( kernrl->flag & CU_INPT_HOST   )
      {
        if ( master == kernrl )
        {
          // Create page locked host memory and copy raw fft data - for the entire input data
          CUDA_SAFE_CALL(cudaFreeHost(kernrl->h_iData), "Failed to create page-locked host memory for entire input data." );
        }
      }

      if      ( kernrl->flag & CU_OUTP_DEVICE )
      {
        // Create a candidate list
        CUDA_SAFE_CALL(cudaFree(kernrl->d_retData), "Failed to allocate device memory for candidate list stack.");

        // Create a semaphore list
        CUDA_SAFE_CALL(cudaFree(kernrl->d_candSem), "Failed to allocate device memory for candidate semaphore list.");
      }
      else if ( ( kernrl->flag & CU_OUTP_HOST ) || ( kernrl->flag & CU_OUTP_DEVICE ) )
      {
        if ( master == kernrl )
        {
          CUDA_SAFE_CALL(cudaFreeHost(kernrl->h_retData), "Failed to create page-locked host memory for entire candidate list." );
        }
      }
    }

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

      if ( (batch->flag & FLAG_STP_ROW) || (batch->flag & FLAG_STP_PLN) )
      {
        cPlain->d_plainData     = &cStack->d_plainData[   cStack->startZ[j] * batch->noSteps * cStack->inpStride ];
        cPlain->d_plainPowers   = &cStack->d_plainPowers[ cStack->startZ[j] * batch->noSteps * cStack->pwrStride ];
      }
      else // Note this works for 1 step or FLAG_STP_STK
      {
        cPlain->d_plainData     = &cStack->d_plainData[   cStack->startZ[j] * cStack->inpStride];
        cPlain->d_plainPowers   = &cStack->d_plainPowers[ cStack->startZ[j] * cStack->pwrStride];
      }

      cPlain->d_iData           = &cStack->d_iData[cStack->inpStride*j*batch->noSteps];
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
    idSiz                += batch->noSteps * cStack->inpStride * cStack->noInStack;
    cmplStart            += cStack->height  * cStack->inpStride * batch->noSteps ;
    pwrStart             += cStack->height  * cStack->pwrStride * batch->noSteps ;
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

  //cuFFdotBatch* batch = new cuFFdotBatch;

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
    // Allocate page-locked host memory for input data
    if ( batch->flag & CU_INPT_SINGLE ) // TODO: Do a memory check here, ie is the enough
    {
      CUDA_SAFE_CALL(cudaMallocHost((void**) &batch->h_iData, batch->inpDataSize*batch->noSteps ), "Failed to create page-locked host memory plain input data." );

      if ( batch->flag & CU_INPT_SINGLE_C ) // Allocate memory for normalisation
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

        if ( batch->flag & FLAG_CUFFTCB_OUT )
        {
          CUDA_SAFE_CALL(cudaMalloc((void** )&batch->d_plainPowers,  batch->pwrDataSize*batch->noSteps ), "Failed to allocate device memory for kernel stack.");
          //batch->d_plainPowers = (float*)batch->d_plainData; // We can just re-use the plain data <- UMMMMMMMMM? No we can't!!
        }
      }
    }

    FOLD // Allocate device & page-locked host memory for candidate  data  .
    {
      if ( batch->flag & CU_OUTP_SINGLE )
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
      batch->InpFFTTime   = (float*)malloc(batch->noStacks*sizeof(float));
      batch->convTime     = (float*)malloc(batch->noStacks*sizeof(float));
      batch->InvFFTTime   = (float*)malloc(batch->noStacks*sizeof(float));
      batch->searchTime   = (float*)malloc(batch->noStacks*sizeof(float));
      batch->copyD2HTime  = (float*)malloc(batch->noStacks*sizeof(float));

      memset(batch->copyH2DTime,  0,batch->noStacks*sizeof(float));
      memset(batch->InpFFTTime,   0,batch->noStacks*sizeof(float));
      memset(batch->convTime,     0,batch->noStacks*sizeof(float));
      memset(batch->InvFFTTime,   0,batch->noStacks*sizeof(float));
      memset(batch->searchTime,   0,batch->noStacks*sizeof(float));
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
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->inpFFTinit),  "Creating input FFT initialisation event");
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->invFFTinit), 	"Creating inverse FFT initialisation event");
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->convInit), 		"Creating convolution initialisation event");

          // out events (with timing)
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->prepComp), 		"Creating input data preparation complete event");
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->convComp), 		"Creating convolution complete event");
          CUDA_SAFE_CALL(cudaEventCreate(&cStack->plnComp),    	"Creating convolution complete event");
#else
          // out events (without timing)
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
    if ( (batch->flag&FLAG_TEX_INTERP) && !( (batch->flag&FLAG_CUFFTCB_OUT) && (batch->flag&FLAG_PLN_TEX) ) )
    {
      fprintf(stderr, "ERROR: Cannot use texture memory interpolation without CUFFT callback to write powers. NOT using texture memory interpolation\n");
      batch->flag ^= FLAG_TEX_INTERP;
    }

    if ( batch->flag & FLAG_PLN_TEX )
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

          if ( batch->flag & FLAG_CUFFTCB_OUT ) // float input
          {
            if      ( batch->flag & FLAG_STP_ROW )
            {
              resDesc.res.pitch2D.height          = cPlain->harmInf->height;
              resDesc.res.pitch2D.width           = cPlain->harmInf->width * batch->noSteps;
              resDesc.res.pitch2D.pitchInBytes    = cStack->harmInf->width * batch->noSteps * sizeof(float);
              resDesc.res.pitch2D.devPtr          = cPlain->d_plainPowers;
            }
            else if ( batch->flag & FLAG_STP_PLN )
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
            if      ( batch->flag & FLAG_STP_ROW )
            {
              resDesc.res.pitch2D.height          = cPlain->harmInf->height;
              resDesc.res.pitch2D.width           = cPlain->harmInf->width * batch->noSteps * 2;
              resDesc.res.pitch2D.pitchInBytes    = cStack->harmInf->width * batch->noSteps * 2* sizeof(float);
              resDesc.res.pitch2D.devPtr          = cPlain->d_plainPowers;
            }
            else if ( batch->flag & FLAG_STP_PLN )
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

  FOUT // Set up CUFFT call back stuff  .
  {
    if ( (batch->flag & FLAG_CUFFTCB_INP) || (batch->flag & FLAG_CUFFTCB_OUT) )
    {
      if ( batch->flag & FLAG_CUFFTCB_INP )
      {
        for (int i = 0; i < batch->noStacks; i++)
        {
          cuFfdotStack* cStack  = &batch->stacks[i];
          CUDA_SAFE_CALL(cudaMalloc((void **)&cStack->d_cinf, sizeof(fftCnvlvInfo)),"Malloc Device memory for CUFFT call-back structure");

          size_t heights = 0;

          fftCnvlvInfo h_inf;

          h_inf.noSteps         = batch->noSteps;
          h_inf.stride          = cStack->inpStride;
          h_inf.width           = cStack->width;
          h_inf.noPlains        = cStack->noInStack;
          //h_inf.d_plainPowers   = cStack->d_plainPowers;

          for (int i = 0; i < cStack->noInStack; i++)     // Loop over plains to determine where they start
          {
            h_inf.d_idata[i]    = cStack->plains[i].d_iData;
            h_inf.d_kernel[i]   = cStack->kernels[i].d_kerData;
            h_inf.heights[i]    = cStack->harmInf[i].height;
            h_inf.top[i]        = heights;
            heights            += cStack->harmInf[i].height;
          }

          for (int i = cStack->noInStack; i < MAX_STKSZ; i++ )
          {
            h_inf.heights[i]    = cStack->harmInf[i].height;
            printf("top %02i: %6li\n", i, heights);
          }

          // Copy host memory to device
          CUDA_SAFE_CALL(cudaMemcpy(cStack->d_cinf, &h_inf, sizeof(fftCnvlvInfo), cudaMemcpyHostToDevice),"Copy to device");
        }
      }

      //copyCUFFT_LD_CB();
    }
  }

  return batch->noSteps;
}

/** Free batch data structure  .
 *
 * @param batch
 */
void freeBatch(cuFFdotBatch* batch)
{
  FOLD // Allocate all device and host memory for the stacks  .
  {
    // Allocate page-locked host memory for input data
    if ( batch->flag & CU_INPT_SINGLE ) // TODO: Do a memory check here, ie is the enough
    {
      CUDA_SAFE_CALL(cudaFreeHost(batch->h_iData ), "Failed to create page-locked host memory plain input data." );

      if ( batch->flag & CU_INPT_SINGLE_C ) // Allocate memory for normalisation
        free(batch->h_powers);
    }

    FOLD // Allocate device Memory for Plain Stack & input data (steps)  .
    {
      // Allocate device memory
      CUDA_SAFE_CALL(cudaFree(batch->d_iData ), "Failed to allocate device memory for kernel stack.");
      CUDA_SAFE_CALL(cudaFree(batch->d_plainData ), "Failed to allocate device memory for kernel stack.");

      if ( batch->flag & FLAG_CUFFTCB_OUT )
      {
        CUDA_SAFE_CALL(cudaFree(batch->d_plainPowers), "Failed to allocate device memory for kernel stack.");
      }
    }

    FOLD // Allocate device & page-locked host memory for candidate  data  .
    {
      if ( batch->flag & CU_OUTP_SINGLE )
      {
        CUDA_SAFE_CALL(cudaFree(batch->d_retData     ), "Failed to allocate device memory for return values.");
        CUDA_SAFE_CALL(cudaFreeHost(batch->h_retData ),"");
      }
    }

    // Create the plains structures
    free(batch->plains);

    FOLD // Create timing arrays
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
    if ( batch->flag & FLAG_PLN_TEX )
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

  FOLD // Set up CUFFT call back stuff
  {
    if ( (batch->flag & FLAG_CUFFTCB_INP) || (batch->flag & FLAG_CUFFTCB_OUT) )
    {
      if ( batch->flag & FLAG_CUFFTCB_INP )
      {
        for (int i = 0; i < batch->noStacks; i++)
        {
          cuFfdotStack* cStack  = &batch->stacks[i];
          CUDA_SAFE_CALL(cudaFree(cStack->d_cinf),"Malloc Device memory for CUFFT call-back structure");
        }
      }
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

  //printf("\nGeneral search\n");

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

void testzm()
{
  cufftHandle plan;
  CUFFT_SAFE_CALL(cufftCreate(&plan),"Failed associating a CUFFT plan with FFT input stream\n");
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
    fprintf(stderr,"WARING: Compiled in timing mode, user requested %i batches but will only process 1 on device %i.\n", gpul.noDevBatches[dev], gpul.devId[dev] );
    gpul.noDevBatches[dev] = 1;
  }
#endif

    if ( dev >= cmd->nstepsC )
      gpul.noDevSteps[dev] = cmd->nsteps[cmd->nbatchC-1];
    else
      gpul.noDevSteps[dev] = cmd->nsteps[dev];
  }

  return gpul;
}

int readAccelDefalts(int *flags)
{
  FILE *file;
  char fName[1024];
  sprintf(fName, "%s/lib/GPU_defaults.txt", getenv("PRESTO"));

  if ( file = fopen(fName, "r") )  // Read candidates from previous search  .
  {
    printf("\nReading GPU settings from %s\n",fName);

    char line[1024];

    while (fgets(line, sizeof(line), file))
    {

      if      ( strncmp(line,"CU_INPT_DEVICE", 		14) == 0 )
      {
        (*flags) ^= CU_INPT_ALL;
        (*flags) |= CU_INPT_DEVICE;
        printf(" CU_INPT_DEVICE \n");
      }
      else if ( strncmp(line,"CU_INPT_HOST", 	  	12) == 0 )
      {
        (*flags) ^= CU_INPT_ALL;
        (*flags) |= CU_INPT_HOST;
        printf(" CU_INPT_HOST \n");
      }
      else if ( strncmp(line,"CU_INPT_SINGLE_C", 	16) == 0 )
      {
        (*flags) ^= CU_INPT_ALL;
        (*flags) |= CU_INPT_SINGLE_C;
      }
      else if ( strncmp(line,"CU_INPT_SINGLE_G",  16) == 0 )
      {
        (*flags) ^= CU_INPT_ALL;
        (*flags) |= CU_INPT_SINGLE_G;
        printf(" CU_INPT_SINGLE_G \n");
      }

      else if ( strncmp(line,"CU_OUTP_DEVICE", 		14) == 0 )
      {
        (*flags) ^= CU_OUTP_ALL;
        (*flags) |= CU_OUTP_DEVICE;
        printf(" CU_OUTP_DEVICE \n");
      }
      else if ( strncmp(line,"CU_OUTP_HOST", 	  	12) == 0 )
      {
        (*flags) ^= CU_OUTP_ALL;
        (*flags) |= CU_OUTP_HOST;
        printf(" CU_OUTP_HOST \n");
      }
      else if ( strncmp(line,"CU_OUTP_SINGLE", 	  12) == 0 )
      {
        (*flags) ^= CU_OUTP_ALL;
        (*flags) |= CU_OUTP_SINGLE;
        printf(" CU_OUTP_SINGLE \n");
      }

      else if ( strncmp(line,"FLAG_SAS_SIG", 	    12) == 0 )
      {
        (*flags) |= FLAG_SAS_SIG;
        printf(" FLAG_SAS_SIG \n");
      }

      else if ( strncmp(line,"CU_CAND_LST",       11) == 0 )
      {
        (*flags) |= CU_CAND_LST;
        printf(" CU_CAND_LST \n");
      }
      else if ( strncmp(line,"CU_CAND_ARR",       11) == 0 )
      {
        (*flags) |= CU_CAND_ARR;
        printf(" CU_CAND_ARR \n");
      }

      else if ( strncmp(line,"FLAG_CNV_PLN", 	    12) == 0 )
      {
        (*flags) ^= FLAG_CNV_ALL;
        (*flags) |= FLAG_CNV_PLN;
        printf(" FLAG_CNV_PLN \n");
      }
      else if ( strncmp(line,"FLAG_CNV_STK", 	    12) == 0 )
      {
        (*flags) ^= FLAG_CNV_ALL;
        (*flags) |= FLAG_CNV_STK;
        printf(" FLAG_CNV_STK \n");
      }
      else if ( strncmp(line,"FLAG_CNV_FAM", 	    12) == 0 )
      {
        (*flags) ^= FLAG_CNV_ALL;
        (*flags) |= FLAG_CNV_FAM;
        printf(" FLAG_CNV_FAM \n");
      }

      else if ( strncmp(line,"FLAG_STP_ROW",      12) == 0 )
      {
        (*flags) ^= FLAG_STP_ALL;
        (*flags) |= FLAG_STP_ROW;
        printf(" FLAG_STP_ROW \n");
      }
      else if ( strncmp(line,"FLAG_STP_PLN",      12) == 0 )
      {
        (*flags) ^= FLAG_STP_ALL;
        (*flags) |= FLAG_STP_PLN;
        printf(" FLAG_STP_PLN \n");
      }
      else if ( strncmp(line,"FLAG_STP_STK",      12) == 0 )
      {
        (*flags) ^= FLAG_STP_ALL;
        (*flags) |= FLAG_STP_STK;
        printf(" FLAG_STP_STK \n");
      }

      else if ( strncmp(line,"FLAG_CUFFTCB_INP",  16) == 0 )
      {
        (*flags) |= FLAG_CUFFTCB_INP;
        printf(" FLAG_CUFFTCB_INP \n");
      }
      else if ( strncmp(line,"FLAG_CUFFTCB_OUT",  16) == 0 )
      {
        (*flags) |= FLAG_CUFFTCB_OUT;
        printf(" FLAG_CUFFTCB_OUT \n");
      }

      else if ( strncmp(line,"FLAG_CNV_TEX",      12) == 0 )
      {
        (*flags) |= FLAG_CNV_TEX;
        printf(" FLAG_CNV_TEX \n");
      }
      else if ( strncmp(line,"FLAG_PLN_TEX",      12) == 0 )
      {
        (*flags) |= FLAG_CNV_TEX;
        printf(" FLAG_PLN_TEX \n");
      }
      else if ( strncmp(line,"FLAG_TEX_INTERP",   15) == 0 )
      {
        (*flags) |= FLAG_TEX_INTERP;
        printf(" FLAG_RETURN_ALL \n");
      }

      else if ( strncmp(line,"FLAG_RETURN_ALL",   15) == 0 )
      {
        (*flags) |= FLAG_RETURN_ALL;
        printf(" FLAG_RETURN_ALL \n");
      }
      else if ( strncmp(line,"FLAG_STORE_ALL",    14) == 0 )
      {
        (*flags) |= FLAG_STORE_ALL;
        printf(" FLAG_STORE_ALL \n");
      }
      else if ( strncmp(line,"FLAG_STORE_EXP",    14) == 0 )
      {
        (*flags) |= FLAG_STORE_EXP;
        printf(" FLAG_STORE_EXP \n");
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

  // Defaults for accel search
  sSpec.flags         |= FLAG_RETURN_ALL ;
  sSpec.flags         |= CU_CAND_ARR ;
  sSpec.flags         |= FLAG_STP_ROW ;  //   FLAG_STP_ROW    FLAG_STP_PLN
  //sSpec.flags         |= FLAG_PLN_TEX ;
  //sSpec.flags         |= FLAG_TEX_INTERP ;
  sSpec.flags         |= FLAG_CUFFTCB_OUT ;

  // Convolution kernel
  sSpec.flags         |= FLAG_CNV_FAM ;  //  FLAG_CNV_FAM   FLAG_CNV_STK   FLAG_CNV_PLN

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
  }

  FOLD // Create plains for calculations  .
  {
    nvtxRangePush("Init Batches");

    aInf->noSteps = 0;

    aInf->batches = (cuFFdotBatch*)malloc(aInf->noBatches*sizeof(cuFFdotBatch));

    int bNo = 0;
    int ker = 0;

    for ( int dev = 0 ; dev < gSpec->noDevices; dev++ ) // Loop over devices  .
    {
      int noSteps = 0;
      if ( gSpec->noDevBatches[dev] > 0 )
      {
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
            aInf->noSteps += noSteps;
            bNo++;
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
        fprintf(stderr,"ERROR: Call to %s with differing GPU search paramiters. Will have to allocate new GPU memory and kernels.\n      NB: Not freeing the old memory!", __FUNCTION__);
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

  FOLD // Calculate power cutoff and number of independent values
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

int freeCuAccel(cuMemInfo* aInf)
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

  free(aInf->batches);
  aInf->batches = NULL;
  free(aInf->kernels);
  aInf->kernels = NULL;
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
        cuFFdot*      cPlain  = &batch->plains[harm];
        rVals*        rVal    = &((*batch->rInput)[step][harm]);

        for( int y = 0; y < cHInfo->height; y++ )
        {

          fcomplexcu *cmplxData;
          float *powers;

          if ( batch->flag & FLAG_STP_ROW )
          {
            cmplxData = &batch->d_plainData[(y*batch->noSteps + step)*cStack->inpStride ];
            powers    = &batch->d_plainPowers[((y*batch->noSteps + step)*cStack->pwrStride + cHInfo->halfWidth * 2 ) ];
          }
          else if ( batch->flag & FLAG_STP_PLN )
          {
            cmplxData = &batch->d_plainData[   (y + step*cHInfo->height)*cStack->inpStride ];
            powers    = &batch->d_plainPowers[((y + step*cHInfo->height)*cStack->pwrStride  + cHInfo->halfWidth * 2 ) ];
          }

          cmplxData += cHInfo->halfWidth*2;
          //CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[step][harm].getP(0,y), cmplxData, (cHInfo->width-2*2*cHInfo->halfWidth)*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
          //CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[step][harm].getP(0,y), cmplxData, (cPlain->numrs[step])*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
          CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[step][harm].getP(0,y), cmplxData, (rVal->numrs)*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
          if ( batch->flag & FLAG_CUFFTCB_OUT )
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

  flags = FLAG_CUFFTCB_OUT ;

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

