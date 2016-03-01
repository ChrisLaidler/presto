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
__global__ void init_kernels(storeT* response, int maxZ, int width, int half_width,  float zSteps, float rSteps)
{
  int cx, cy;							/// The x and y index of this thread in the array
  int rx = -1;							/// The x index of the value in the kernel

  // Calculate the 2D index of this thread
  cx = blockDim.x * blockIdx.x + threadIdx.x;			/// use BLOCKSIZE rather (its constant)
  cy = blockDim.y * blockIdx.y + threadIdx.y;			/// use BLOCKSIZE rather (its constant)

  float z = -maxZ + cy * (float)1.0/zSteps;			/// The Fourier Frequency derivative

  if ( z < -maxZ || z > maxZ || cx >= width || cx < 0 )
  {
    // Out of bounds
    return;
  }
  
  if      ( half_width == 0  )
  {
    half_width    = z_resp_halfwidth_cu<float>(z);
  }
  else if ( half_width == 1  )  // Use high accuracy kernels
  {
    half_width    = z_resp_halfwidth_cu_high<float>(z);
  }
  else
  {
     // Use the actual halfwidth value for all rows
  	
    //int hw2       = MAX(0.6*z, 16*1);
    //half_width    = MIN( half_width, hw2 ) ;
    //half_width    = z_resp_halfwidth_cu((double) z);
    //half_width    = z_resp_halfwidth_cu((double) z);
  }

  int noResp      = half_width / rSteps;		// The number of response variables per side
  float offset;						// The distance of the response value from 0 (negative to the leaf)

  // Calculate the kernel index for this thread (centred on zero and wrapped)
  if		( cx < noResp )
  {
    offset = cx * rSteps;
    rx = 1;
  }
  else if	(cx >= width - noResp )
  {
    offset = ( cx - width ) * rSteps;			// This is the negative side of the response function
    rx = 1;
  }

  // the complex response
  genT real = 0.0;
  genT imag = 0.0;

  FOLD // Calculate the response value  .
  {
    if (rx != -1)
    {
      calc_response_off<genT> ((genT)offset, (genT)z, &real, &imag);
    }
  }

  response[cy * width + cx].x = real;
  response[cy * width + cx].y = imag;
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
      init_kernels<double, double2><<<dimGrid, dimBlock, 0, cStack->initStream>>>((double2*)cStack->d_kerData, cStack->harmInf->zmax, cStack->width,  halfWidth, ACCEL_RDZ, ACCEL_DR);
    }
    else if ( cStack->flags & FLAG_KER_DOUBGEN )
    {
      init_kernels<double, float2><<<dimGrid, dimBlock, 0, cStack->initStream>>>((float2*)cStack->d_kerData, cStack->harmInf->zmax, cStack->width,  halfWidth, ACCEL_RDZ, ACCEL_DR);
    }
    else
    {
      init_kernels<float, float2><<<dimGrid, dimBlock, 0, cStack->initStream>>>((float2*)cStack->d_kerData, cStack->harmInf->zmax, cStack->width,  halfWidth, ACCEL_RDZ, ACCEL_DR);
    }

    // Run message
    CUDA_SAFE_CALL(cudaGetLastError(), "At kernel launch");
  }

  return 0;
}


int copyKerDoubleToFloat(cuFfdotStack* cStack, float* d_orrKer)
{
  dim3 dimBlock, dimGrid;

  dimBlock.x     = KR_DIM_X;  // in my experience 16 is almost always best (half warp)
  dimBlock.y     = KR_DIM_Y;  // in my experience 16 is almost always best (half warp)

  size_t width   = cStack->strideCmplx * 2 ;

  // Set up grid
  dimGrid.x = ceil(  width / ( float ) ( BLOCKSIZE * BLOCKSIZE ) );
  dimGrid.y = 1;

  typeChangeKer<double, float><<<dimGrid, dimBlock, 0, cStack->initStream>>>((double*)cStack->d_kerData, d_orrKer, width,  cStack->kerHeigth );

  return 0;
}
