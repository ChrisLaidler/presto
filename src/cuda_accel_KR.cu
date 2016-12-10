#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"
#include "cuda_response.h"

#define KR_DIM_X	16
#define KR_DIM_Y	16

template<typename readT, typename writeT>
__global__ void typeChangeKer(readT* read, writeT* write, size_t stride, size_t height)
{

  const int bidx = threadIdx.y * KR_DIM_X + threadIdx.x;          /// Block ID - flat index
  const int tid  = blockIdx.x  * KR_DIM_X * KR_DIM_Y + bidx;      /// Global thread ID - flat index ie column index of stack
  size_t offset;

  if ( tid < stride )  // Valid thread  .
  {
    read  += tid;
    write += tid;

    for ( int h = 0; h < height; h++ )
    {
      offset = h * stride;

      write[offset] = read[offset];
    }
  }
}

/** Create the convolution kernel for a f-∂f plane  .
 *
 *  This is "copied" from gen_z_response in respocen.c
 *
 * @param response
 * @param maxZ
 * @param fftlen
 */
template<typename genT, typename storeT>
__global__ void init_kernels(storeT* response, double zStart, double zEnd, int noZ, int width, int half_width, int rSteps)
{
  int cx, cy;							/// The x and y index of this thread in the array
  int rx = -1;							/// The x index of the value in the kernel

  // Calculate the 2D index of this thread
  cx = blockDim.x * blockIdx.x + threadIdx.x;			/// use BLOCKSIZE rather (its constant)
  cy = blockDim.y * blockIdx.y + threadIdx.y;			/// use BLOCKSIZE rather (its constant)

  if ( cy < noZ && cx < width )
  {
    // In bound

    // Calculate the z value for the row
    genT z;
    if ( noZ == 0 )
      z = zStart + (zEnd-zStart)/(genT)(noZ-1)*cy;		/// The Fourier Frequency derivative
    else
      z = zStart;

    if      ( half_width == 0  )				// Standard low accuracy half width
    {
      half_width    = cu_z_resp_halfwidth_low<genT>(z);
    }
    else if ( half_width == 1  )				// Use high accuracy kernels
    {
      half_width    = cu_z_resp_halfwidth_high<genT>(z);
    }
    else							// Only used for debug purposes
    {
      // Use the actual halfwidth value for all rows

      //int hw2       = MAX(0.6*z, 16*1);
      //half_width    = MIN( half_width, hw2 ) ;
      //half_width    = cu_z_resp_halfwidth((double) z);
      //half_width    = cu_z_resp_halfwidth((double) z);
    }

    int noResp	= half_width * rSteps;				// The number of response variables per side
    genT offset	= 0;						// The distance of the response value from 0 (negative to the leaf)

    // Calculate the kernel index for this thread (centred on zero inverted and wrapped)
    if		( cx < noResp )
    {
      // Beginning of array ( left half of responce values mirrored about zero)
      offset = -1 * cx / (genT)rSteps;
      rx = 1;
    }
    else if	(cx >= width - noResp )
    {
      // End of array ( right half of responce values mirrored about zero)
      offset = ( width - cx ) / (genT)rSteps;
      rx = 1;
    }

    // The complex response
    genT real = 0.0;
    genT imag = 0.0;

    FOLD // Calculate the response value  .
    {
      if (rx != -1)
      {
	calc_response_off<genT> (offset, z, &real, &imag);
      }
    }

    response[cy * width + cx].x = real;
    response[cy * width + cx].y = imag;
  }
}

/** Create one GPU kernel. One kernel the size of the largest plane  .
 *
 * @param kernel
 * @return
 */
int createStackKernel(cuFfdotStack* cStack)
{
  dim3 dimBlock, dimGrid;

  dimBlock.x          = KR_DIM_X;  // in my experience 16 is almost always best (half warp)
  dimBlock.y          = KR_DIM_Y;  // in my experience 16 is almost always best (half warp)

  // Set up grid
  dimGrid.x = ceil(  cStack->width     / ( float ) dimBlock.x );
  dimGrid.y = ceil ( cStack->kerHeigth / ( float ) dimBlock.y );

  int halfWidth;

  if ( cStack->flags & FLAG_KER_MAX )
  {
    // Use one halfwidth for the entire kernel
    halfWidth = cStack->harmInf->kerStart / 2.0;
  }
  else
  {
    if ( cStack->flags & FLAG_KER_HIGH )
    {
      // high accuracy
       halfWidth = 1;
    }
    else
    {
      // Standard "low" accuracy
      halfWidth = 0;
    }
  }

  FOLD // Call the CUDA kernels  .
  {
    if      ( (cStack->flags & FLAG_KER_DOUBFFT) || (cStack->flags & FLAG_DOUBLE) )
    {
      init_kernels<double, double2><<<dimGrid, dimBlock, 0, cStack->initStream>>>((double2*)cStack->kernels->d_kerData, cStack->harmInf->zStart, cStack->harmInf->zEnd, cStack->harmInf->noZ, cStack->width, halfWidth, cStack->harmInf->noResPerBin);
    }
    else if ( cStack->flags & FLAG_KER_DOUBGEN )
    {
      init_kernels<double, float2><<<dimGrid, dimBlock, 0, cStack->initStream>>>((float2*)cStack->kernels->d_kerData, cStack->harmInf->zStart, cStack->harmInf->zEnd, cStack->harmInf->noZ, cStack->width, halfWidth, cStack->harmInf->noResPerBin);
    }
    else
    {
      init_kernels<float, float2><<<dimGrid, dimBlock, 0, cStack->initStream>>>((float2*)cStack->kernels->d_kerData, cStack->harmInf->zStart, cStack->harmInf->zEnd, cStack->harmInf->noZ, cStack->width, halfWidth, cStack->harmInf->noResPerBin);
    }

    // Run message
    CUDA_SAFE_CALL(cudaGetLastError(), "At kernel launch");
  }

  return 0;
}


int copyKerDoubleToFloat(cuKernel* doubleKer, cuKernel* floatKer, cudaStream_t stream) //   cuFfdotStack* cStack, float* d_orrKer)
{
  dim3 dimBlock, dimGrid;

  dimBlock.x     = KR_DIM_X;  // in my experience 16 is almost always best (half warp)
  dimBlock.y     = KR_DIM_Y;  // in my experience 16 is almost always best (half warp)

  size_t width   = doubleKer->stride * 2 ; // Stride is in complex valuses

  // Set up grid
  dimGrid.x = ceil(  width / ( float ) ( dimBlock.x * dimBlock.y ) );
  dimGrid.y = 1;

  typeChangeKer<double, float><<<dimGrid, dimBlock, 0, stream>>>((double*)doubleKer->d_kerData, (float*)floatKer->d_kerData, width, doubleKer->harmInf->noZ );

  return 0;
}

void createBatchKernels(cuFFdotBatch* batch, void* buffer)
{
  cuKernel doubleKres[MAX_STACKS];
  char msg[1024];

  infoMSG(4,4,"Initialise the multiplication kernels.\n");

  // Run message
  CUDA_SAFE_CALL(cudaGetLastError(), "Before creating GPU kernels");

  FOLD // Allocate temporary memory for kernel wanting double precision FFT's  .
  {
    for (int i = 0; i < MAX_STACKS; i++)
    {
      doubleKres[i].d_kerData = NULL;
    }

    if ( (batch->flags & FLAG_KER_DOUBFFT) && !(batch->flags & FLAG_DOUBLE) )
    {
      for (int i = 0; i < batch->noStacks; i++)
      {
	infoMSG(4,6,"Stack %i\n",i);

	cuFfdotStack* cStack = &batch->stacks[i];

	size_t kerSz = cStack->kernels->stride * cStack->kernels->harmInf->noZ * sizeof(double2);

	memcpy(&doubleKres[i], cStack->kernels, sizeof(cuKernel));

	CUDA_SAFE_CALL(cudaMalloc((void**)&doubleKres[i].d_kerData, kerSz), "Failed to allocate temporary device memory for kernel stack."); // This is temporary double memory it will be freed at the end of this function
      }
    }
  }

  FOLD // Calculate the response values  .
  {
    infoMSG(4,5,"Calculate the response values\n");

    NV_RANGE_PUSH("Calc stack response");

    for (int i = 0; i < batch->noStacks; i++)
    {
      cuFfdotStack* cStack = &batch->stacks[i];

      // Call the CUDA kernels
      createStackKernel(cStack);
    }

    NV_RANGE_POP();
  }

  FOLD // FFT the kernels  .
  {
    infoMSG(4,5,"FFT the  response values\n");

    NV_RANGE_PUSH("FFT kernels");

    for (int i = 0; i < batch->noStacks; i++)
    {
      infoMSG(4,6,"Stack %i\n",i);

      cuFfdotStack* cStack = &batch->stacks[i];

      if ( (batch->flags & FLAG_KER_DOUBFFT) || (batch->flags & FLAG_DOUBLE) )
      {
	FOLD // Create the plan  .
	{
	  infoMSG(4,6,"Create plan\n");

	  sprintf(msg,"Plan %i",i);
	  NV_RANGE_PUSH(msg);

	  int n[]             = {cStack->width};
	  int inembed[]       = {cStack->strideCmplx* sizeof(double2)};
	  int istride         = 1;
	  int idist           = cStack->strideCmplx;
	  int onembed[]       = {cStack->strideCmplx* sizeof(double2)};
	  int ostride         = 1;
	  int odist           = cStack->strideCmplx;
	  int height          = cStack->kerHeigth;

	  // Normal plans
	  CUFFT_SAFE_CALL(cufftPlanMany(&cStack->plnPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, height), "Creating plan for FFT'ing the kernel.");
	  CUDA_SAFE_CALL(cudaGetLastError(), "Creating FFT plans for the stacks.");

	  NV_RANGE_POP();
	}

	FOLD // Call the plan  .
	{
	  infoMSG(4,6,"Call the plan\n");

	  sprintf(msg,"Call %i",i);
	  NV_RANGE_PUSH(msg);

	  CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->initStream),  "Error associating a CUFFT plan with multStream.");
	  CUFFT_SAFE_CALL(cufftExecZ2Z(cStack->plnPlan, (cufftDoubleComplex *)doubleKres[i].d_kerData, (cufftDoubleComplex *) doubleKres[i].d_kerData, CUFFT_FORWARD), "FFT'ing the kernel data. [cufftExecC2C]");
	  CUDA_SAFE_CALL(cudaGetLastError(), "FFT'ing the multiplication kernels.");

	  NV_RANGE_POP();
	}

	FOLD // Destroy the plan  .
	{
	  infoMSG(4,6,"Destroy the plan\n");

	  sprintf(msg,"Dest %i",i);
	  NV_RANGE_PUSH(msg);

	  CUFFT_SAFE_CALL(cufftDestroy(cStack->plnPlan), "Destroying plan for complex data of stack. [cufftDestroy]");
	  CUDA_SAFE_CALL(cudaGetLastError(), "Destroying the plan.");

	  NV_RANGE_POP();
	}
      }
      else
      {
	FOLD // Create the plan  .
	{
	  infoMSG(4,6,"Create plan\n");

	  sprintf(msg,"Plan %i",i);
	  NV_RANGE_PUSH(msg);
	  size_t workSize;

	  int n[]             = {cStack->width};
	  int inembed[]       = {cStack->strideCmplx* sizeof(fcomplexcu)};
	  int istride         = 1;
	  int idist           = cStack->strideCmplx;
	  int onembed[]       = {cStack->strideCmplx* sizeof(fcomplexcu)};
	  int ostride         = 1;
	  int odist           = cStack->strideCmplx;
	  int height          = cStack->kerHeigth;

          // Normal plans
	  if (buffer)
	  {
	    // use pre allocated memory
	    CUFFT_SAFE_CALL( cufftCreate(&cStack->plnPlan), "cufftCreate");
	    CUFFT_SAFE_CALL( cufftSetAutoAllocation(cStack->plnPlan, 0), "cufftSetAutoAllocation");
	    CUFFT_SAFE_CALL( cufftSetWorkArea(cStack->plnPlan, buffer), "cufftSetWorkArea" ); // Assighn pre allocated memory
	    CUFFT_SAFE_CALL( cufftMakePlanMany(cStack->plnPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, height, &workSize), "cufftMakePlanMany" );
	  }
	  else
	  {
	    CUFFT_SAFE_CALL(cufftPlanMany(&cStack->plnPlan,  1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, height), "Creating plan for FFT'ing the kernel.");
	    CUDA_SAFE_CALL(cudaGetLastError(), "Creating FFT plans for the stacks.");
	  }
	  
	  NV_RANGE_POP();
	}

	FOLD // Call the plan  .
	{
	  infoMSG(4,6,"Call the plan\n");

	  sprintf(msg,"Call %i",i);
	  NV_RANGE_PUSH(msg);

	  CUFFT_SAFE_CALL(cufftSetStream(cStack->plnPlan, cStack->initStream),  "Error associating a CUFFT plan with multStream.");
	  CUFFT_SAFE_CALL(cufftExecC2C(cStack->plnPlan, (cufftComplex *) cStack->kernels->d_kerData, (cufftComplex *) cStack->kernels->d_kerData, CUFFT_FORWARD), "FFT'ing the kernel data. [cufftExecC2C]");
	  CUDA_SAFE_CALL(cudaGetLastError(), "FFT'ing the multiplication kernels.");

	  NV_RANGE_POP();
	}

	FOLD // Destroy the plan  .
	{
	  if (!buffer)
	  {
	    infoMSG(4,6,"Destroy the plan\n");

	    sprintf(msg,"Dest %i",i);
	    NV_RANGE_PUSH(msg);

	    CUFFT_SAFE_CALL(cufftDestroy(cStack->plnPlan), "Destroying plan for complex data of stack. [cufftDestroy]");
	    CUDA_SAFE_CALL(cudaGetLastError(), "Destroying the plan.");

	    NV_RANGE_POP();
	  }
	}
      }
    }

    CUDA_SAFE_CALL(cudaGetLastError(), "FFT'ing the multiplication kernels.");

    NV_RANGE_POP();
  }

  FOLD // Copy double FFT'ed data back to the float kernel  .
  {
    if ( (batch->flags & FLAG_KER_DOUBFFT) && !(batch->flags & FLAG_DOUBLE) )
    {
      for (int i = 0; i < batch->noStacks; i++)
      {
	cuFfdotStack* cStack = &batch->stacks[i];

	copyKerDoubleToFloat( &doubleKres[i], cStack->kernels, cStack->initStream );
      }
    }

    FOLD // Free temporary memory for kernel  .
    {
      if ( (batch->flags & FLAG_KER_DOUBFFT) && !(batch->flags & FLAG_DOUBLE) )
      {
	for (int i = 0; i < batch->noStacks; i++)
	{
	  cudaFreeNull( doubleKres[i].d_kerData );			// Free the temporary double data
	}
      }
    }
  }


}
