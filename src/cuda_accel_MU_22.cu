/** @file cuda_accel_MU_22.cu
 *  @brief The implementation of the stack multiplication kernel v2
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

#ifdef WITH_MUL_22

/** Multiplication kernel - Multiply a stack with a kernel - multi-step - Loop ( Pln - Y - step )  .
 * Each thread loops down a column of the plane
 * Reads the input and multiplies it with the kernel and writes result to plane
 */
template<int64_t FLAGS, int noSteps>
__global__ void mult22_k(const __restrict__ fcomplexcu*  kernels, const __restrict__ fcomplexcu*  inpData, __restrict__ fcomplexcu* ffdot, const int width, const int stride, int noPlns, const int firstPlane )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;          /// Block ID - flat index
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;      /// Global thread ID - flat index ie column index of stack

  if ( tid < width )  // Valid thread  .
  {
    int idx;                                      /// flat index of output plane
    int pHeight = 0;                              /// Height of previous data in the stack
    fcomplexcu ker;                               /// kernel data

    FOLD  // Stride, kernel, input data & output data  .
    {
      kernels += tid;
      ffdot   += tid;
      inpData += tid;
    }

    __restrict__ fcomplexcu inpDat[noSteps];                  // Set of input data for this thread/column

    for (int pln = 0; pln < noPlns; pln++)                    // Loop through the planes  .
    {
      const int plnStrd       = pln*stride*noSteps;
      const int plnHeight     = HEIGHT_HARM[firstPlane + pln];
      const int kerYOffset    = KERNEL_OFF_HARM[firstPlane + pln];
#ifdef WITH_ITLV_PLN
      const int ns2           = plnHeight * stride;
#endif

      FOLD // Read input data for this plane  .
      {
        for (int step = 0; step < noSteps; step++)
        {
          fcomplexcu inp      = inpData[ (int)(plnStrd + step*stride) ];
          inp.r               /= (float) width;
          inp.i               /= (float) width;
          inpDat[step]        = inp;
        }
      }

      short   lDepth  = ceilf(plnHeight/(float)gridDim.y);
      short   y0      = lDepth*blockIdx.y;
      short   y1      = MIN(y0+lDepth, plnHeight);

      for (int planeY = y0; planeY < y1; planeY++)      // Loop over the individual plane  .
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

        for ( int step = 0; step < noSteps; ++step )          // Loop over steps .
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

          FOLD // Multiply  .
          {
            fcomplexcu ipd = inpDat[step];
            fcomplexcu out;

#if CORRECT_MULT
              // This is the "correct" version
              out.r = (ipd.r * ker.r - ipd.i * ker.i);
              out.i = (ipd.r * ker.i + ipd.i * ker.r);
#else
              // This is the version accelsearch uses, ( added for comparison )
              out.r = (ipd.r * ker.r + ipd.i * ker.i);
              out.i = (ipd.i * ker.r - ipd.r * ker.i);
#endif
            ffdot[idx] = out;
          }
        }
      }

      pHeight += plnHeight * noSteps * stride;
    }
  }
}

template<int64_t FLAGS>
__host__  void mult22_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  int offset            = cStack->startIdx;

  switch (batch->noSteps)
  {
#if MIN_STEPS <= 1  and MAX_STEPS >= 1
    case 1:
    {
      mult22_k<FLAGS,1><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 2  and MAX_STEPS >= 2
    case 2:
    {
      mult22_k<FLAGS,2><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 3  and MAX_STEPS >= 3
    case 3:
    {
      mult22_k<FLAGS,3><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 4  and MAX_STEPS >= 4
    case 4:
    {
      mult22_k<FLAGS,4><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 5  and MAX_STEPS >= 5
    case 5:
    {
      mult22_k<FLAGS,5><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 6  and MAX_STEPS >= 6
    case 6:
    {
      mult22_k<FLAGS,6><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 7  and MAX_STEPS >= 7
    case 7:
    {
      mult22_k<FLAGS,7><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 8  and MAX_STEPS >= 8
    case 8:
    {
      mult22_k<FLAGS,8><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 9  and MAX_STEPS >= 9
    case 9:
    {
      mult22_k<FLAGS,9><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 10 and MAX_STEPS >= 10
    case 10:
    {
      mult22_k<FLAGS,10><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 11 and MAX_STEPS >= 11
    case 11:
    {
      mult22_k<FLAGS,11><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_STEPS <= 12 and MAX_STEPS >= 12
    case 12:
    {
      mult22_k<FLAGS,12><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

    default:
    {
      if      ( batch->noSteps < MIN_STEPS )
	fprintf(stderr, "ERROR: In %s, # steps (%i) less than the compiled minimum %i.\n", __FUNCTION__, batch->noSteps, MIN_STEPS );
      else if ( batch->noSteps > MAX_STEPS )
	fprintf(stderr, "ERROR: In %s, # steps (%i) greater than the compiled maximum %i.\n", __FUNCTION__, batch->noSteps, MIN_STEPS );
      else
	fprintf(stderr, "ERROR: %s has not been templated for %i steps.\n", __FUNCTION__, batch->noSteps);

      exit(EXIT_FAILURE);
    }
  }
}

#endif	// WITH_MUL_22

__host__  void mult22(cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack)
{
#ifdef WITH_MUL_22

  dim3 dimGrid, dimBlock;

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY ));
  dimGrid.y = cStack->mulSlices;

  if      ( batch->flags & FLAG_ITLV_ROW )
    mult22_s<FLAG_ITLV_ROW>(dimGrid, dimBlock, 0, multStream, batch, cStack);
#ifdef WITH_ITLV_PLN
  else
    mult22_s<0>(dimGrid, dimBlock, 0, multStream, batch, cStack);
#else
    else
    {
      fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
      exit(EXIT_FAILURE);
    }
#endif

#else
  EXIT_DIRECTIVE("WITH_MUL_22");
#endif
}
