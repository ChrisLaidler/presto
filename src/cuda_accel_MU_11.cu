/** @file cuda_accel_MU_11.cu
 *  @brief The implementation of the plain multiplication kernel v1
 *
 *  @author Chris Laidler
 *  @bug No known bugs.
 *
 *  Change Log
 *
 *  []
 *    Beginning of change log
 *    Working version un-numbed
 *
 *  [2017-04-16]
 *     Added host function for mult11
 *
 */

#include "cuda_accel_MU.h"

#ifdef WITH_MUL_11

/** Convolution kernel - One thread per r location (input FFT)
 * Each thread reads one input value and loops down over the kernels
 */
__global__ void mult11(fcomplexcu *ffdot, uint width, uint stride, uint height, const fcomplexcu *data, const fcomplexcu *kernels)
{
  const int bidx  = threadIdx.y * CNV_DIMX + threadIdx.x;
  const int tid   = blockIdx.x  * CNV_DIMX * CNV_DIMY     + bidx;

  if (tid < width)    // Clip
  {
    fcomplexcu ker;   // item from kernel
    int idx = 0;      // flat index

    const float inpReal = data[tid].r / (float) width;
    const float inpImag = data[tid].i / (float) width;

    // Stride the input and output
    kernels += tid;
    ffdot   += tid;

    //#pragma unroll
    for (int y = 0; y < height; y++)
    {
      idx = y * stride;

      ker = kernels[idx];

#if CORRECT_MULT
      ffdot[idx].r = (inpReal * ker.r - inpImag * ker.i);
      ffdot[idx].i = (inpImag * ker.r + inpReal * ker.i);
#else
      ffdot[idx].r = (inpReal * ker.r + inpImag * ker.i);
      ffdot[idx].i = (inpImag * ker.r - inpReal * ker.i);
#endif
    }
  }
}

#endif

__host__  void mult11(cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack)
{
#ifdef WITH_MUL_11
  dim3 dimGrid, dimBlock;

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  void*       d_planeData;    // The complex f-∂f plane data
  fcomplexcu* d_iData;        // The complex input array

  for (int plane = 0; plane < cStack->noInStack; plane++)         // Loop through planes in stack
  {
    cuHarmInfo* cHInfo    = &cStack->harmInf[plane];              // The current harmonic we are working on
    cuFFdot*    cPlane    = &cStack->planes[plane];               // The current f-∂f plane

    dimGrid.x = ceil(cHInfo->width / (float) ( CNV_DIMX * CNV_DIMY ));
    dimGrid.y = 1;

    for (int step = 0; step < batch->noSteps; step++)             // Loop through Steps
    {
      d_iData         = cPlane->d_iData + cStack->strideCmplx * step;

      if      ( batch->flags & FLAG_ITLV_ROW )
      {
	fprintf(stderr,"ERROR: Cannot do single plane multiplications with row-interleaved multi step stacks.\n");
	exit(EXIT_FAILURE);
      }
#ifdef WITH_ITLV_PLN
      else
      {
	// Shift by plane height
	if ( batch->flags & FLAG_DOUBLE )
	  d_planeData   = (double2*)cPlane->d_planeMult + step * cHInfo->noZ * cStack->strideCmplx;
	else
	  d_planeData   = (float2*) cPlane->d_planeMult + step * cHInfo->noZ * cStack->strideCmplx;
      }
#else
      else
      {
	fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
	exit(EXIT_FAILURE);
      }
#endif

      // Texture memory in multiplication is now deprecated
      //if ( batch->flag & FLAG_TEX_MUL )
      //  mult12<<<dimGrid, dimBlock, 0, multStream>>>(d_planeData, cHInfo->width, cStack->strideCmplx, cHInfo->height, d_iData, cPlane->kernel->kerDatTex);
      //else
      mult11<<<dimGrid, dimBlock, 0, multStream>>>((fcomplexcu*)d_planeData, cHInfo->width, cStack->strideCmplx, cHInfo->noZ, d_iData, (fcomplexcu*)cPlane->kernel->d_kerData);

      // Run message
      CUDA_SAFE_CALL(cudaGetLastError(), "At multiplication kernel launch");
    }
  }
#else
  EXIT_DIRECTIVE("WITH_MUL_11");
#endif
}
