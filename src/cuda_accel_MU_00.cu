/** @file cuda_accel_MU_00.cu
 *  @brief The implementation of the non functional optimal multiplication kernel
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
 *  [0.0.01] [2017-02-24]
 *     Added preprocessor directives for steps and chunks
 *
 */
 
 #include "cuda_accel_MU.h"

#ifdef WITH_MUL_00

/** Kernel for testing best possible performance - Just write to ffdot plane - 1 thread per complex value  .
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
__global__ void mult00_k(const __restrict__ float2* kernels, const __restrict__ float2*  inpData, __restrict__ float2* ffdot, const int width, const int height, const int stride, const int noSteps, const int noPlns, int kerHeight )
{
  const int ix = blockIdx.x * CNV_DIMX + threadIdx.x;
  const int iy = blockIdx.y * CNV_DIMY + threadIdx.y;

  float2 ker;							/// kernel data
  uint nHeight = height * noSteps;

  ker.y = 0;
  ker.x = 0;

  if (ix < width && iy < nHeight)
  {
    // Calculate flat index
    const int idx = iy * stride + ix;

    ffdot[idx] = ker;
  }
}

#endif	// WITH_MUL_00

#ifdef WITH_MUL_01	// - Read input, read kernel, write to ffdot plane - 1 thread per column  .

/** Kernel for testing best possible performance - Read input, read kernel, write to ffdot plane - 1 thread per column  .
 *
 */
__global__ void mult01_k(const __restrict__ float2* kernels, const __restrict__ float2* inpData, __restrict__ float2* ffdot, const int width, const int height, const int stride, const int noSteps, const int noPlns, int kerHeight )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;	/// Block ID - flat index
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;	/// Global thread ID - flat index ie column index of stack

  float2 ker;							/// kernel data

  if ( tid < width )  // Valid thread  .
  {
    int idx;							/// flat index of output plane

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
        for (int pln = 0; pln < noPlns; pln++)			// Loop through the planes  .
        {
          float2 ipd = inpData[ (int)(pln*noSteps*stride + step*stride) ];

          if ( ipd.x < 0 && ipd.x > 0 )				// Required so as to not optimise out  .
          {
            printf("mult01_k ipd < 0????   tid: %04i  %9.5f %9.5f\n", tid, ipd.x, ipd.y );
          }
        }
      }
    }

    FOLD // Read kernel  .
    {
      int   lDepth  = ceilf(kerHeight/(float)gridDim.y);
      int   y0      = lDepth*blockIdx.y;
      int   y1      = MIN(y0+lDepth, kerHeight);

      for (int kerY = y0; kerY < y1; kerY++ )
      {
        idx   = kerY * stride;
        ker   = kernels[idx];

        if ( ker.x < 0 && ker.x > 0 )				// Required so as to not optimise out  .
        {
          printf("mult01_k ker < 0????   tid: %04i  %9.5f %9.5f\n", tid, ker.x, ker.y );
        }
      }
    }

    FOLD // Write data to planes  .
    {
      int   nHeight = height * noSteps;
      int   lDepth  = ceilf(nHeight/(float)gridDim.y);
      int   y0      = lDepth*blockIdx.y;
      int   y1      = MIN(y0+lDepth, nHeight);

      ker.y         = 0;
      ker.x         = 0;

      for (int y = y0; y < y1; y++ )
      {
        idx         = y * stride;

        FOLD // Write  .
        {
          ffdot[idx] = ker;
        }
      }
    }
  }
}
#endif	// WITH_MUL_01

#ifdef WITH_MUL_02	// Read input, read kernel, write to ffdot plane - 1 thread per column  - templated for steps  .

/** Multiplication kernel - Multiply a stack with a kernel - multi-step - Loop ( Pln - Y - step )  .
 * Each thread loops down a column of the plane
 * Reads the input and multiplies it with the kernel and writes result to plane
 */
template<int64_t FLAGS, int noSteps>
__global__ void mult02_k(const float2* __restrict__ kernels, const float2* __restrict__ inpData, float2* __restrict__ ffdot, const int width, const int stride, int noPlns, const int firstPlane )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;	/// Block ID - flat index
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;	/// Global thread ID - flat index ie column index of stack

  if ( tid < width )  // Valid thread  .
  {
    int idx;							/// flat index of output plane
    int pHeight = 0;						/// Height of previous data in the stack
    float2 ker;							/// kernel data

    FOLD  // Stride, kernel, input data & output data  .
    {
      kernels += tid;
      ffdot   += tid;
      inpData += tid;
    }

    __restrict__ float2 inpDat[noSteps];			// Set of input data for this thread/column

    for (int pln = 0; pln < noPlns; pln++)			// Loop through the planes  .
    {
      const int plnStrd       = pln*stride*noSteps;
      const int plnHeight     = HEIGHT_HARM[firstPlane + pln];
      const int kerYOffset    = (HEIGHT_HARM[firstPlane] - plnHeight)/2;
#ifdef WITH_ITLV_PLN
      const int ns2           = plnHeight * stride;
#endif

      FOLD // Read input data for this plane
      {
        for (int step = 0; step < noSteps; step++)
        {
          float2 inp         = inpData[ (int)(plnStrd + step*stride) ];
          inp.x             /= (float) width;
          inp.y             /= (float) width;
          inpDat[step]      = inp;
        }
      }

      for (int planeY = 0; planeY < plnHeight; planeY++)	// Loop over the individual plane  .
      {
        FOLD // Read the kernel value  .
        {
          ker   = kernels[(kerYOffset+planeY)*stride];
        }

        int off1;

        FOLD // Calculate partial offset  .
        {
          if      ( FLAGS & FLAG_ITLV_ROW )
          {
            off1  = pHeight + planeY*noSteps*stride;
          }
#ifdef WITH_ITLV_PLN
          else
          {
            off1  = pHeight + planeY*stride;
          }
#endif
        }

        for ( int step = 0; step < noSteps; ++step )		// Loop over steps .
        {
          FOLD // Calculate indices  .
          {
            if      ( FLAGS & FLAG_ITLV_ROW )
            {
              idx  = off1 + step * stride;
            }
#ifdef WITH_ITLV_PLN
            else
            {
              idx  = off1 + step * ns2;
            }
#endif
          }

          float2 kv;
          FOLD // Multiply  .
          {
            kv.x = (inpDat[step].x * ker.x + inpDat[step].y * ker.y);
            kv.y = (inpDat[step].y * ker.x - inpDat[step].x * ker.y);
          }

          ffdot[idx]  = kv;
        }
      }

      pHeight += plnHeight * noSteps * stride;
    }
  }
}

template<int64_t FLAGS>
__host__  void mult02_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  int offset            = cStack->startIdx;

  switch (batch->noSteps)
  {
#if MIN_STEPS <= 1  and MAX_STEPS >= 1
    case 1:
    {
      mult02_k<FLAGS,1><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 2  and MAX_STEPS >= 2
    case 2:
    {
      mult02_k<FLAGS,2><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 3  and MAX_STEPS >= 3
    case 3:
    {
      mult02_k<FLAGS,3><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 4  and MAX_STEPS >= 4
    case 4:
    {
      mult02_k<FLAGS,4><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 5  and MAX_STEPS >= 5
    case 5:
    {
      mult02_k<FLAGS,5><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 6  and MAX_STEPS >= 6
    case 6:
    {
      mult02_k<FLAGS,6><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 7  and MAX_STEPS >= 7
    case 7:
    {
      mult02_k<FLAGS,7><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 8  and MAX_STEPS >= 8
    case 8:
    {
      mult02_k<FLAGS,8><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 9  and MAX_STEPS >= 9
    case 9:
    {
      mult02_k<FLAGS,9><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 10 and MAX_STEPS >= 10
    case 10:
    {
      mult02_k<FLAGS,10><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 11 and MAX_STEPS >= 11
    case 11:
    {
      mult02_k<FLAGS,11><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 12 and MAX_STEPS >= 12
    case 12:
    {
      mult02_k<FLAGS,12><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

    default:
    {
      if      ( batch->noSteps < MIN_STEPS )
	fprintf(stderr, "ERROR: In %s, # steps (%i) less than the compiled minimum %i.\n", __FUNCTION__, batch->noSteps, MIN_STEPS );
      else if ( batch->noSteps > MAX_SAS_CHUNK )
	fprintf(stderr, "ERROR: In %s, # steps (%i) greater than the compiled maximum %i.\n", __FUNCTION__, batch->noSteps, MIN_STEPS );
      else
	fprintf(stderr, "ERROR: %s has not been templated for %i steps.\n", __FUNCTION__, batch->noSteps);

      exit(EXIT_FAILURE);
    }
  }
}

__host__  void mult02_f(cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  dim3 dimGrid, dimBlock;

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY ));
  dimGrid.y = 1;

  if      ( batch->flags & FLAG_ITLV_ROW )
    mult02_s<FLAG_ITLV_ROW>(dimGrid, dimBlock, 0, multStream, batch, cStack);
#ifdef WITH_ITLV_PLN
  else
    mult02_s<0>(dimGrid, dimBlock, 0, multStream, batch, cStack);
#else
  else
  {
    fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
    exit(EXIT_FAILURE);
  }
#endif
}

#endif	// WITH_MUL_02

/** Kernel for testing best possible performance - Just write to ffdot plane - Each thread loops down over column  .
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
__host__  void mult00(cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  dim3 dimGrid, dimBlock;

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  if ( 0 )
  {
    // Dummy
  }
#ifdef WITH_MUL_00
  else if ( 1 )
    dimGrid.x = ceil(cStack->width                    / (float) ( CNV_DIMX ));
    dimGrid.y = ceil(cStack->height*batch->noSteps    / (float) ( CNV_DIMX ));

    mult00_k<<<dimGrid, dimBlock, 0, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->height, cStack->strideCmplx, batch->noSteps, cStack->noInStack, cStack->kerHeigth);
  }
#endif
#ifdef WITH_MUL_01
  else if ( 1 )
  {
    dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY ));
    dimGrid.y = cStack->mulSlices;

    mult01_k<<<dimGrid, dimBlock, 0, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeMult, cStack->width, cStack->height, cStack->strideCmplx, batch->noSteps, cStack->noInStack, cStack->kerHeigth);
  }
#endif
  else
  {
    fprintf(stderr, "ERROR: Code has not been compiled with Multiplication \"optimal\" kernel." );
    exit(EXIT_FAILURE);
  }

}
