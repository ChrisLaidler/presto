/** @file cuda_accel_MU_23.cu
 *  @brief The implementation of the stack multiplication kernel v3
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
 *	Added preprocessor directives for steps and chunks
 *
 *  [0.0.01] [2017-03-25]
 *	Added templating for multiplication chunks
 *
 *  [2017-03-10]
 *	Fixed bug in templating
 *
 */
 
 #include "cuda_accel_MU.h"

/** Multiplication kernel - All multiplications for a stack - Uses registers to store sections of the kernel - Loop ( chunk (read ker) - plan - Y - step ) .
 * Each thread loops down a column of the plane
 * Reads the input and multiples it with the kernel and writes result to plane
 */
/*
template<int64_t FLAGS, int noSteps, int noPlns>
__global__ void mult23_k(const __restrict__ fcomplexcu* kernels, const __restrict__ fcomplexcu* inpData, __restrict__ fcomplexcu* ffdot, const int width, const int stride, const int firstPlane )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;          /// Block ID - flat index
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;      /// Global thread ID - flat index ie column index of stack

  const int cv_chunkSZ = 12;

  if ( tid < width )  // Valid thread  .
  {
    const int kerHeight = HEIGHT_HARM[firstPlane];       // The size of the kernel

    short   lDepth      = ceilf(kerHeight/(float)gridDim.y);
    short   y0          = lDepth*blockIdx.y;
    short   y1          = MIN(y0+lDepth, kerHeight);

    FOLD  // Stride, kernel, input data & output data  .
    {
      kernels += tid;
      ffdot   += tid;
      inpData += tid;
    }

    //int noChunks = ceilf(kerHeight / (float)cv_chunkSZ);
    //for ( int chunk = 0; chunk < noChunks; chunk++ )

    for ( int y = y0; y < y1; y+=cv_chunkSZ )
    {
      //const int c0  = chunk*cv_chunkSZ;
      //const int c1  = MIN(cv_chunkSZ,kerHeight-c0);

      const int c0  = y;
      const int c1  = MIN(cv_chunkSZ,kerHeight-y);
      int pHeight   = 0;

      register fcomplexcu k0   = kernels[(c0+0 )*stride];
      register fcomplexcu k1   = kernels[(c0+1 )*stride];
      register fcomplexcu k2   = kernels[(c0+2 )*stride];
      register fcomplexcu k3   = kernels[(c0+3 )*stride];
      register fcomplexcu k4   = kernels[(c0+4 )*stride];
      register fcomplexcu k5   = kernels[(c0+5 )*stride];
      register fcomplexcu k6   = kernels[(c0+6 )*stride];
      register fcomplexcu k7   = kernels[(c0+7 )*stride];
      register fcomplexcu k8   = kernels[(c0+8 )*stride];
      register fcomplexcu k9   = kernels[(c0+9 )*stride];
      register fcomplexcu k10  = kernels[(c0+10)*stride];
      register fcomplexcu k11  = kernels[(c0+11)*stride];

      for (int pln = 0; pln < noPlns; pln++)                  // Loop through the planes of the stack  .
      {
        const int plnHeight     = HEIGHT_HARM[firstPlane + pln];
        const int kerYOffset    = KERNEL_OFF_HARM[firstPlane + pln];

        const int p0            = MAX(c0 - kerYOffset,0);
        const int p1            = MIN(c0 + c1 - kerYOffset, plnHeight);

        const int kerAddd       = MAX(0, kerYOffset - c0);

#ifdef WITH_ITLV_PLN
        const int ns2           = plnHeight * stride;
#endif

        __restrict__ fcomplexcu inpDat[noSteps];              // Set of input data for this thread/column
        FOLD // Read all input data  .
        {
          // NOTE: I tested reading the input for planes and steps (2 loops above) but that was slower, here uses less registers as well.

          for (int step = 0; step < noSteps; step++)
          {
            fcomplexcu ipd        = inpData[ (int)(pln*noSteps*stride + step*stride) ];
            ipd.r                 /= (float) width;
            ipd.i                 /= (float) width;
            inpDat[step]     = ipd;
          }
        }

        for (int planeY = p0; planeY < p1; planeY++)          // Loop over the individual plane  .
        {
          int y = planeY - p0 + kerAddd;
          fcomplexcu ker;

          FOLD // Read the kernel value  .
          {
            switch(y)
            {
              case 0	:
              {
                ker = k0;
                break;
              }
              case 1	:
              {
                ker = k1;
                break;
              }
              case 2	:
              {
                ker = k2;
                break;
              }
              case 3	:
              {
                ker = k3;
                break;
              }
              case 4	:
              {
                ker = k4;
                break;
              }
              case 5	:
              {
                ker = k5;
                break;
              }
              case 6	:
              {
                ker = k6;
                break;
              }
              case 7	:
              {
                ker = k7;
                break;
              }
              case 8	:
              {
                ker = k8;
                break;
              }
              case 9	:
              {
                ker = k9;
                break;
              }
              case 10	:
              {
                ker = k10;
                break;
              }
              case 11	:
              {
                ker = k11;
                break;
              }
            }

          }

          int offsetPart1;
          FOLD // Calculate partial offset  .
          {
            if      ( FLAGS & FLAG_ITLV_ROW )
            {
              offsetPart1  = pHeight + planeY*noSteps*stride;
            }
#ifdef WITH_ITLV_PLN
            else
            {
              offsetPart1  = pHeight + planeY*stride;
            }
#endif
          }

          for ( int step = 0; step < noSteps; step++ )        // Loop over steps  .
          {
            int idx;

            FOLD // Calculate offset  .
            {
              if      ( FLAGS & FLAG_ITLV_ROW )
              {
                idx  = offsetPart1 + step * stride;
              }
#ifdef WITH_ITLV_PLN
              else
              {
                idx  = offsetPart1 + step * ns2;
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

        pHeight += plnHeight * noSteps * stride;              // Set striding value for next plane
      }
    }
  }
}
*/

/** Multiplication kernel - All multiplications for a stack - Uses registers to store sections of the kernel - Loop ( chunk (read ker) - plan - Y - step ) .
 * Each thread loops down a column of the plane
 * Reads the input and multiples it with the kernel and writes result to plane
 */
template<int64_t FLAGS, int noSteps, int noPlns, const int cunkSize>
__global__ void mult23_k(const __restrict__ fcomplexcu* kernels, const __restrict__ fcomplexcu* inpData, __restrict__ fcomplexcu* ffdot, const int width, const int stride, const int firstPlane )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;	/// Block ID - flat index
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;	/// Global thread ID - flat index ie column index of stack

  if ( tid < width )  // Valid thread  .
  {
    const int kerHeight = HEIGHT_HARM[firstPlane];		// The size of the kernel

    short   lDepth      = ceilf(kerHeight/(float)gridDim.y);
    short   y0          = lDepth*blockIdx.y;
    short   y1          = MIN(y0+lDepth, kerHeight);

    FOLD  // Stride, kernel, input data & output data  .
    {
      kernels += tid;
      ffdot   += tid;
      inpData += tid;
    }

    for ( int y = y0; y < y1; y+= cunkSize )
    {
      const int c0  = y;
      const int c1  = MIN(cunkSize, kerHeight-y);
      int pHeight   = 0;

      fcomplexcu kerChunk[cunkSize];

      for ( int i = 0; i < cunkSize; i++)
      {
	kerChunk[i]   = kernels[(c0+i )*stride];
      }

      for (int pln = 0; pln < noPlns; pln++)			// Loop through the planes of the stack  .
      {
        const int plnHeight     = HEIGHT_HARM[firstPlane + pln];
        const int kerYOffset    = KERNEL_OFF_HARM[firstPlane + pln];

        const int p0            = MAX(c0 - kerYOffset,0);
        const int p1            = MIN(c0 + c1 - kerYOffset, plnHeight);

        const int kerAddd       = MAX(0, kerYOffset - c0);

#ifdef WITH_ITLV_PLN
        const int ns2           = plnHeight * stride;
#endif

        __restrict__ fcomplexcu inpDat[noSteps];		// Set of input data for this thread/column
        FOLD // Read all input data  .
        {
          // NOTE: I tested reading the input for planes and steps (2 loops above) but that was slower, here uses less registers as well.

          for (int step = 0; step < noSteps; step++)
          {
            fcomplexcu ipd         = inpData[ (int)(pln*noSteps*stride + step*stride) ];
            ipd.r                 /= (float) width;
            ipd.i                 /= (float) width;
            inpDat[step]           = ipd;
          }
        }

        for (int planeY = p0; planeY < p1; planeY++)		// Loop over the individual plane  .
        {
          int y = planeY - p0 + kerAddd;
          fcomplexcu ker = kerChunk[y];

          int offsetPart1;
          FOLD // Calculate partial offset  .
          {
            if      ( FLAGS & FLAG_ITLV_ROW )
            {
              offsetPart1  = pHeight + planeY*noSteps*stride;
            }
#ifdef WITH_ITLV_PLN
            else
            {
              offsetPart1  = pHeight + planeY*stride;
            }
#endif
          }

          for ( int step = 0; step < noSteps; step++ )		// Loop over steps  .
          {
            int idx;

            FOLD // Calculate offset  .
            {
              if      ( FLAGS & FLAG_ITLV_ROW )
              {
                idx  = offsetPart1 + step * stride;
              }
#ifdef WITH_ITLV_PLN
              else
              {
                idx  = offsetPart1 + step * ns2;
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

        pHeight += plnHeight * noSteps * stride;              // Set striding value for next plane
      }
    }
  }
}

template<int64_t FLAGS, int noSteps, int noPlns>
__host__  void mult23_c(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  int offset            = cStack->startIdx;

   switch (cStack->mulChunk)
   {
#if MIN_MUL_CHUNK <= 1  and MAX_MUL_CHUNK >= 1
    case 1 :
    {
      mult23_k<FLAGS,noSteps,noPlns,1><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 2  and MAX_MUL_CHUNK >= 2
    case 2 :
    {
      mult23_k<FLAGS,noSteps,noPlns,2><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 3  and MAX_MUL_CHUNK >= 3
    case 3 :
    {
      mult23_k<FLAGS,noSteps,noPlns,3><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 4  and MAX_MUL_CHUNK >= 4
    case 4 :
    {
      mult23_k<FLAGS,noSteps,noPlns,4><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 5  and MAX_MUL_CHUNK >= 5
    case 5 :
    {
      mult23_k<FLAGS,noSteps,noPlns,5><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 6  and MAX_MUL_CHUNK >= 6
    case 6 :
    {
      mult23_k<FLAGS,noSteps,noPlns,6><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 7  and MAX_MUL_CHUNK >= 7
    case 7 :
    {
      mult23_k<FLAGS,noSteps,noPlns,7><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 8  and MAX_MUL_CHUNK >= 8
    case 8 :
    {
      mult23_k<FLAGS,noSteps,noPlns,8><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 9  and MAX_MUL_CHUNK >= 9
    case 9 :
    {
      mult23_k<FLAGS,noSteps,noPlns,9><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 10  and MAX_MUL_CHUNK >= 10
    case 10 :
    {
      mult23_k<FLAGS,noSteps,noPlns,10><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 11  and MAX_MUL_CHUNK >= 11
    case 11 :
    {
      mult23_k<FLAGS,noSteps,noPlns,11><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

#if MIN_MUL_CHUNK <= 12  and MAX_MUL_CHUNK >= 12
    case 12 :
    {
      mult23_k<FLAGS,noSteps,noPlns,12><<<dimGrid, dimBlock, i1, multStream>>>((fcomplexcu*)cStack->kernels->d_kerData , cStack->d_iData, (fcomplexcu*)cStack->d_planeMult, cStack->width, cStack->strideCmplx, offset);
      break;
    }
#endif

    default:
    {
      if ( cStack->mulChunk < MIN_MUL_CHUNK )
	fprintf(stderr, "ERROR: In %s, chunk size (%i) less than the compiled minimum %i.\n", __FUNCTION__, cStack->mulChunk, MIN_MUL_CHUNK );
      else if ( batch->ssChunk > MAX_MUL_CHUNK )
	fprintf(stderr, "ERROR: In %s, chunk size (%i) greater than the compiled maximum %i.\n", __FUNCTION__, cStack->mulChunk, MAX_MUL_CHUNK );
      else
	fprintf(stderr, "ERROR: %s has not been templated for %i chunk size.\n", __FUNCTION__, cStack->mulChunk);

      exit(EXIT_FAILURE);
    }

   }
}

template<int64_t FLAGS, int noSteps>
__host__  void mult23_p(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack)
{
   switch (cStack->noInStack)
  {
    case 1	:
    {
      mult23_c<FLAGS,noSteps,1>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
    case 2	:
    {
      mult23_c<FLAGS,noSteps,2>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
    case 3	:
    {
      mult23_c<FLAGS,noSteps,3>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
    case 4	:
    {
      mult23_c<FLAGS,noSteps,4>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
    case 5	:
    {
      mult23_c<FLAGS,noSteps,5>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
    case 6	:
    {
      mult23_c<FLAGS,noSteps,6>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
    case 7	:
    {
      mult23_c<FLAGS,noSteps,7>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
    case 8	:
    {
      mult23_c<FLAGS,noSteps,8>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
    case 9	:
    {
      mult23_c<FLAGS,noSteps,9>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
    default	:
    {
      fprintf(stderr, "ERROR: mult23 has not been templated for %i planes in a stack.\n", cStack->noInStack);
      exit(EXIT_FAILURE);
    }
  }
}

template<int64_t FLAGS>
__host__  void mult23_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  switch (batch->noSteps)
  {
#if MIN_STEPS <= 1  and MAX_STEPS >= 1
    case 1:
    {
      mult23_p<FLAGS,1>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 2  and MAX_STEPS >= 2
    case 2:
    {
      mult23_p<FLAGS,2>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 3  and MAX_STEPS >= 3
    case 3:
    {
      mult23_p<FLAGS,3>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 4  and MAX_STEPS >= 4
    case 4:
    {
      mult23_p<FLAGS,4>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 5  and MAX_STEPS >= 5
    case 5:
    {
      mult23_p<FLAGS,5>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 6  and MAX_STEPS >= 6
    case 6:
    {
      mult23_p<FLAGS,6>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 7  and MAX_STEPS >= 7
    case 7:
    {
      mult23_p<FLAGS,7>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 8  and MAX_STEPS >= 8
    case 8:
    {
      mult23_p<FLAGS,8>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 9  and MAX_STEPS >= 9
    case 9:
    {
      mult23_p<FLAGS,9>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 10 and MAX_STEPS >= 10
    case 10:
    {
      mult23_p<FLAGS,10>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 11 and MAX_STEPS >= 11
    case 11:
    {
      mult23_p<FLAGS,11>(dimGrid, dimBlock, i1, multStream, batch, cStack);
      break;
    }
#endif

#if MIN_STEPS <= 12 and MAX_STEPS >= 12
    case 12:
    {
      mult23_p<FLAGS,12>(dimGrid, dimBlock, i1, multStream, batch, cStack);
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

__host__  void mult23(cudaStream_t multStream, cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  dim3 dimGrid, dimBlock;

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY )) ;
  dimGrid.y = cStack->mulSlices ;

  if      ( batch->flags & FLAG_ITLV_ROW )
    mult23_s<FLAG_ITLV_ROW>(dimGrid, dimBlock, 0, multStream, batch, cStack);
#ifdef WITH_ITLV_PLN
  else
    mult23_s<0>(dimGrid, dimBlock, 0, multStream, batch, cStack);
#else
    else
    {
      fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
      exit(EXIT_FAILURE);
    }
#endif
}
