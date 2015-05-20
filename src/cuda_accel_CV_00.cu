#include "cuda_accel_CV.h"

/** Kernel for testing best possible performance - Just write to ffdot plain - 1 thread per complex value
 *
 * @param kernels
 * @param inpData
 * @param ffdot
 * @param width
 * @param height
 * @param stride
 * @param noSteps
 * @param kerHeight
 */
__global__ void convolveffdot00_k(const fcomplexcu* __restrict__ kernels, const fcomplexcu* __restrict__ inpData, fcomplexcu* __restrict__ ffdot, const int width, const int height, const int stride, const int noSteps, const int noPlns, int kerHeight )
{
  const int ix = blockIdx.x * CNV_DIMX + threadIdx.x;
  const int iy = blockIdx.y * CNV_DIMY + threadIdx.y;

  fcomplexcu ker;                                 /// kernel data
  uint nHeight = height * noSteps;

  ker.i = 0;
  ker.r = 0;

  if (ix < width && iy < nHeight)
  {
    // Calculate flat index
    const int idx = iy * stride + ix;

    ffdot[idx] = ker;
  }
}

__global__ void convolveffdot01_k(const __restrict__ fcomplexcu* kernels, const __restrict__ fcomplexcu* inpData, __restrict__ fcomplexcu* ffdot, const int width, const int height, const int stride, const int noSteps, const int noPlns, int kerHeight )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;          /// Block ID - flat index
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;      /// Global thread ID - flat index ie column index of stack

  fcomplexcu ker;                                 /// kernel data

  if ( tid < width )  // Valid thread  .
  {
    int idx;                                      /// flat index of output plain

    FOLD  // Stride, kernel, input data & output data  .
    {
      kernels += tid;
      ffdot   += tid;
      inpData += tid;
    }

    FOLD // Read input data  .
    {
      for (int step = 0; step < noSteps; step++)
      {
        for (int pln = 0; pln < noPlns; pln++)                // Loop through the plains  .
        {
          fcomplexcu ipd        = inpData[ (int)(pln*noSteps*stride + step*stride) ];

          if ( ipd.r < 0 && ipd.r > 0 )  // TMP
          {
            printf("ipd \n");
          }
        }
      }
    }

    FOLD // Read kernel  .
    {
      for (int k = 0; k < kerHeight; k++ )
      {
        idx   = k * stride;
        ker   = kernels[idx];

        if ( ker.r < 0 && ker.r > 0 )  // TMP
        {
          printf("ker \n");
        }
      }
    }

    ker.i = 0;
    ker.r = 0;

    uint nHeight = height * noSteps;

    FOLD // Write data to plains  .
    {
      for (int y = 0; y < nHeight; y++ )
      {
        idx  = y * stride;

        FOLD // Write  .
        {
          ffdot[idx] = ker;
        }
      }
    }
  }
}

/** Kernel for testing best possible performance - Just write to ffdot plain - Each thread loops down over column  .
 *
 * @param kernels
 * @param inpData
 * @param ffdot
 * @param width
 * @param height
 * @param stride
 * @param noSteps
 * @param kerHeight
 */
__host__  void convolveffdot00_f(cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack)
{
  dim3 dimGrid, dimBlock;

  cuFfdotStack* cStack = &batch->stacks[stack];

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  if (0)
  {
    dimGrid.x = ceil(cStack->width                    / (float) ( CNV_DIMX ));
    dimGrid.y = ceil(cStack->height*batch->noSteps    / (float) ( CNV_DIMX ));

    convolveffdot00_k<<<dimGrid, dimBlock, 0, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->height, cStack->strideCmplx, batch->noSteps, cStack->noInStack, cStack->kerHeigth);
  }
  else
  {
    dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY ));
    dimGrid.y = 1;

    convolveffdot01_k<<<dimGrid, dimBlock, 0, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->height, cStack->strideCmplx, batch->noSteps, cStack->noInStack, cStack->kerHeigth);
  }

}
