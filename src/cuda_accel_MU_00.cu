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
 *     Added preprocessor directives for segments and chunks
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
 * @param noSegments
 * @param kerHeight
 */
__global__ void mult00_k(const __restrict__ float2* kernels, const __restrict__ float2*  inpData, __restrict__ float2* ffdot, const int width, const int height, const int stride, const int noSegments, const int noPlns, int kerHeight )
{
  const int ix = blockIdx.x * CNV_DIMX + threadIdx.x;
  const int iy = blockIdx.y * CNV_DIMY + threadIdx.y;

  float2 ker;							/// kernel data
  uint nHeight = height * noSegments;

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
__global__ void mult01_k(const __restrict__ float2* kernels, const __restrict__ float2* inpData, __restrict__ float2* ffdot, const int width, const int height, const int stride, const int noSegments, const int noPlns, const int kerHeight )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;	/// Block ID - flat index
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;	/// Global thread ID - flat index ie column index of stack

  const float2 ker = {0.0f,0.0f};				/// kernel data

  if ( tid < width )  // Valid thread  .
  {
    float ss      = 0;

    FOLD // Stride, kernel, input data & output data  .
    {
      kernels += tid;
      ffdot   += tid;
      inpData += tid;
    }

    FOLD // Write data to planes  .
    {
      const int   nHeight = height * noSegments;
      const int   lDepth  = ceilf(nHeight/(float)gridDim.y);
      const int   y0      = lDepth*blockIdx.y;
      const int   y1      = MIN(y0+lDepth, nHeight);

      for (int y = y0; y < y1; y++ )
      {
	ffdot[y*stride] = ker;										//
//      I Tested these to try and get closer to the theorticla badnwithd
//	asm("st.wb.global.v2.f32 [%0], {%1, %2};" :: "l"(ffdot+y*stride), "f"(ker.x), "f"(ker.y) );	// write-back					92.5 %	65.30
//	asm("st.cg.global.v2.f32 [%0], {%1, %2};" :: "l"(ffdot+y*stride), "f"(ker.x), "f"(ker.y) );	// global level (cache in L2 and below, not L1)	92.4 %	65.21
//	asm("st.cs.global.v2.f32 [%0], {%1, %2};" :: "l"(ffdot+y*stride), "f"(ker.x), "f"(ker.y) );	// streaming					92.5 %	65.83
//	asm("st.wt.global.v2.f32 [%0], {%1, %2};" :: "l"(ffdot+y*stride), "f"(ker.x), "f"(ker.y) );	// write-through				92.5 %	65.95
      }
    }

    FOLD // Read input data  .
    {
      for (int inpY = 0; inpY < noSegments*noPlns; inpY++)
      {
	// Each slice reads all input
	float2 ipd = inpData[ inpY*stride ];
	ss   += ipd.x + ipd.y;
      }

    }

    FOLD // Read kernel  .
    {
      int   lDepth  = ceilf(kerHeight/(float)gridDim.y);
      int   y0      = lDepth*blockIdx.y;
      int   y1      = MIN(y0+lDepth, kerHeight);

      for (int kerY = y0; kerY < y1; kerY++ )
      {
	float2 krd = kernels[ kerY * stride ];
	ss   += krd.x + krd.y;
      }
    }

    if ( tid >= 1e5  ) // Here to stop optimising out of reads
    {
      printf("mult01_k   tid: %04i  %9.5f \n", tid, ss );
    }

    /**

    On my GTX 770 I get:
    Stack  Height    mul_GB            secs         BW GB/s              %
	0     634     0.259           0.216          205.96          91.82
	1     195     0.041           0.036          198.52          88.51
	2      32     0.004           0.005          129.41          57.70
	3       7     0.000           0.002           37.44          16.69

    GTX 970
    Stack  Height    mul_GB            secs         BW GB/s              %
       0     634     0.343           0.304          145.41          64.83
       1     195     0.054           0.050          139.44          62.16
       2      32     0.005           0.006          105.68          47.11
       3       7     0.001           0.002           36.12          16.11

    Stack  Height    mul_GB            secs         BW GB/s              %
       0     634     0.343           0.302          146.81          65.45
       1     195     0.054           0.049          143.37          63.92
       2      32     0.005           0.006          104.00          46.37
       3       7     0.001           0.002           36.14          16.11

     **/


  }
}
#endif	// WITH_MUL_01

#ifdef WITH_MUL_02	// Read input, read kernel, write to ffdot plane - 1 thread per column  - templated for segments  .

/** Multiplication kernel - Multiply a stack with a kernel - multi-segment - Loop ( Pln - Y - segment )  .
 * Each thread loops down a column of the plane
 * Reads the input and multiplies it with the kernel and writes result to plane
 */
template<int64_t FLAGS, int noSegments>
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

    __restrict__ float2 inpDat[noSegments];			// Set of input data for this thread/column

    for (int pln = 0; pln < noPlns; pln++)			// Loop through the planes  .
    {
      const int plnStrd       = pln*stride*noSegments;
      const int plnHeight     = HEIGHT_HARM[firstPlane + pln];
      const int kerYOffset    = (HEIGHT_HARM[firstPlane] - plnHeight)/2;

      FOLD // Read input data for this plane
      {
        for (int sIdx = 0; sIdx < noSegments; sIdx++)
        {
          float2 inp         = inpData[ (int)(plnStrd + sIdx*stride) ];
          inp.x             /= (float) width;
          inp.y             /= (float) width;
          inpDat[sIdx]      = inp;
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
          off1  = pHeight + planeY*noSegments*stride;
        }

        for ( int sIdx = 0; sIdx < noSegments; ++sIdx )		// Loop over segments .
        {
          FOLD // Calculate indices  .
          {
            idx  = off1 + sIdx * stride;
          }

          float2 kv;
          FOLD // Multiply  .
          {
            kv.x = 0;
            kv.y = 0;
          }

          ffdot[idx]  = kv;
        }
      }

      pHeight += plnHeight * noSegments * stride;
    }
  }
}

template<int64_t FLAGS>
__host__  void mult02_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t multStream, cuCgPlan* plan, cuFfdotStack* cStack)
{
  int offset            = cStack->startIdx;

  switch (plan->noSegments)
  {
#if MIN_SEGMENTS <= 1  and MAX_SEGMENTS >= 1
    case 1:
    {
      mult02_k<FLAGS,1><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_SEGMENTS <= 2  and MAX_SEGMENTS >= 2
    case 2:
    {
      mult02_k<FLAGS,2><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_SEGMENTS <= 3  and MAX_SEGMENTS >= 3
    case 3:
    {
      mult02_k<FLAGS,3><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_SEGMENTS <= 4  and MAX_SEGMENTS >= 4
    case 4:
    {
      mult02_k<FLAGS,4><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_SEGMENTS <= 5  and MAX_SEGMENTS >= 5
    case 5:
    {
      mult02_k<FLAGS,5><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_SEGMENTS <= 6  and MAX_SEGMENTS >= 6
    case 6:
    {
      mult02_k<FLAGS,6><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_SEGMENTS <= 7  and MAX_SEGMENTS >= 7
    case 7:
    {
      mult02_k<FLAGS,7><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_SEGMENTS <= 8  and MAX_SEGMENTS >= 8
    case 8:
    {
      mult02_k<FLAGS,8><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_SEGMENTS <= 9  and MAX_SEGMENTS >= 9
    case 9:
    {
      mult02_k<FLAGS,9><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_SEGMENTS <= 10 and MAX_SEGMENTS >= 10
    case 10:
    {
      mult02_k<FLAGS,10><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_SEGMENTS <= 11 and MAX_SEGMENTS >= 11
    case 11:
    {
      mult02_k<FLAGS,11><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

#if MIN_SEGMENTS <= 12 and MAX_SEGMENTS >= 12
    case 12:
    {
      mult02_k<FLAGS,12><<<dimGrid, dimBlock, i1, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->strideCmplx, cStack->noInStack, offset);
      break;
    }
#endif

    default:
    {
      if      ( plan->noSegments < MIN_SEGMENTS )
	fprintf(stderr, "ERROR: In %s, # segments (%i) less than the compiled minimum %i.\n", __FUNCTION__, plan->noSegments, MIN_SEGMENTS );
      else if ( plan->noSegments > MAX_SEGMENTS )
	fprintf(stderr, "ERROR: In %s, # segments (%i) greater than the compiled maximum %i.\n", __FUNCTION__, plan->noSegments, MAX_SEGMENTS );
      else
	fprintf(stderr, "ERROR: %s has not been templated for %i segments.\n", __FUNCTION__, plan->noSegments);

      exit(EXIT_FAILURE);
    }
  }
}

__host__  void mult02_f(cudaStream_t multStream, cuCgPlan* plan, cuFfdotStack* cStack)
{
  dim3 dimGrid, dimBlock;

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY ));
  dimGrid.y = 1;

  mult02_s<0>(dimGrid, dimBlock, 0, multStream, plan, cStack);
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
 * @param noSegments
 * @param kerHeight
 */
__host__  void mult00(cudaStream_t multStream, cuCgPlan* plan, cuFfdotStack* cStack)
{
  dim3 dimGrid, dimBlock;

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  if ( 0 )
  {
    // Dummy for hash defines to play nicely
  }
#ifdef WITH_MUL_00
  else if ( 1 )
  {
    dimGrid.x = ceil(cStack->width                      / (float) ( CNV_DIMX ));
    dimGrid.y = ceil(cStack->height*plan->noSegments    / (float) ( CNV_DIMY ));

    mult00_k<<<dimGrid, dimBlock, 0, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->height, cStack->strideCmplx, plan->noSegments, cStack->noInStack, cStack->kerHeigth);
  }
#endif
#ifdef WITH_MUL_01
  else if ( 1 )
  {
    // Use slicing
    dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY ));
    dimGrid.y = cStack->mulSlices;

    mult01_k<<<dimGrid, dimBlock, 0, multStream>>>((float2*)cStack->kernels->d_kerData, (float2*)cStack->d_iData, (float2*)cStack->d_planeCplx, cStack->width, cStack->height, cStack->strideCmplx, plan->noSegments, cStack->noInStack, cStack->kerHeigth);
  }
#endif
#ifdef WITH_MUL_02
  else if ( 1 )
  {
    mult02_f(multStream, plan, cStack);
  }
#endif
  else
  {
    fprintf(stderr, "ERROR: Code has not been compiled with Multiplication \"optimal\" kernel." );
    exit(EXIT_FAILURE);
  }

}
