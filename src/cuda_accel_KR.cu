#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"
#include "cuda_response.h"


__device__ int z_resp_halfwidth(double z)
{
  int m;

  z = fabs(z);

  m = (long) (z * (0.00089 * z + 0.3131) + NUMFINTBINS);
  m = (m < NUMFINTBINS) ? NUMFINTBINS : m;

  // Prevent the equation from blowing up in large z cases

  if (z > 100 && m > 0.6 * z)
    m = 0.6 * z;

  return m;
}

__device__ int z_resp_halfwidth_high(double z)
{
  int m;

  z = fabs(z);

  m = (long) (z * (0.002057 * z + 0.0377) + NUMFINTBINS * 3);
  m += ((NUMLOCPOWAVG >> 1) + DELTAAVGBINS);

  /* Prevent the equation from blowing up in large z cases */

  if (z > 100 && m > 1.2 * z)
     m = 1.2 * z;

  return m;
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
  int cx, cy;                                   /// The x and y index of this thread in the array
  int rx = -1;                                  /// The x index of the value in the kernel

  // Calculate the 2D index of this thread
  cx = blockDim.x * blockIdx.x + threadIdx.x;   /// use BLOCKSIZE rather (its constant)
  cy = blockDim.y * blockIdx.y + threadIdx.y;   /// use BLOCKSIZE rather (its constant)

  float z = -maxZ + cy * 1.0/zSteps;            /// The Fourier Frequency derivative

  if ( z < -maxZ || z > maxZ || cx >= width || cx < 0 )
  {
    // Out of bounds
    return;
  }
  
  if      ( half_width == 0  )
  {
    half_width    = z_resp_halfwidth((double) z);
  }
  else if ( half_width == 1  )  // Use high accuracy kernels
  {
    half_width    = z_resp_halfwidth_high((double) z);
  }
  else
  {
     // Use the actual halfwidth value for all rows
  	
    //int hw2       = MAX(0.6*z, 16*1);
    //half_width    = MIN( half_width, hw2 ) ;
    //half_width    = z_resp_halfwidth((double) z);
    //half_width    = z_resp_halfwidth((double) z);
  }

  int noResp      = half_width / rSteps;		// The number of responce variables per side
  float offset;						// The distance of the responce value from 0 (negitive to the leaft)

  // Calculate the kernel index for this thread (centred on zero and wrapped)
  if ( cx < noResp )
  {
    offset = cx * rSteps;
    rx = 1;
  }
  else if  (cx >= width - noResp )
  {
    offset = ( cx - width ) * rSteps; // This is the negitive side of the responce function
    rx = 1;
  }

  // the complex response
  genT real = 0.0;
  genT imag = 0.0;

  FOLD // Calculate the response value  .
  {
    if (rx != -1)
    {
      calc_response_off<genT> ((float)offset, (float)z, &real, &imag);
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

  dimBlock.x          = BLOCKSIZE;  // in my experience 16 is almost always best (half warp)
  dimBlock.y          = BLOCKSIZE;  // in my experience 16 is almost always best (half warp)

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
    if      ( cStack->flags & FLAG_KER_DOUBGEN )
    {
      init_kernels<double, float2><<<dimGrid, dimBlock, 0, cStack->initStream>>>((float2*)cStack->d_kerData, cStack->harmInf->zmax, cStack->width,  halfWidth, ACCEL_RDZ, ACCEL_DR);
    }
    else if ( cStack->flags & FLAG_KER_DOUBFFT )
    {
      init_kernels<double, double2><<<dimGrid, dimBlock, 0, cStack->initStream>>>((double2*)cStack->d_kerData, cStack->harmInf->zmax, cStack->width,  halfWidth, ACCEL_RDZ, ACCEL_DR);
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
