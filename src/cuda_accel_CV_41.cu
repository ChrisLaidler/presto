#include "cuda_accel_CV.h"

/** Convolution kernel - Convolve a stack with a kernel - multi-step  .
 * Each thread loops down a column of the plain
 * Reads the input and convolves it with the kernel and writes result to plain
 */
#if TEMPLATE_CONVOLVE == 1
template<int FLAGS, int noPlns, int noSteps>
__global__ void convolveffdot41(const fcomplexcu* __restrict__ kernels, const fcomplexcu* __restrict__ inpData, fcomplexcu* __restrict__ ffdot, const int width, const int stride, iHarmList heights, const int stackHeight, cHarmList kerDat, fCplxTex kerTex )
#else
template<int FLAGS, int noPlns>
__global__ void convolveffdot41(const fcomplexcu* __restrict__ kernels, const fcomplexcu* __restrict__ inpData, fcomplexcu* __restrict__ ffdot, const int width, const int stride, iHarmList heights, const int stackHeight, cHarmList kerDat, fCplxTex kerTex, const int noSteps )
#endif
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;          /// Block ID - flat index
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;      /// Global thread ID - flat index ie column index of stack

  if ( tid < width )  // Valid thread  .
  {
    int plainY = 0;                               /// y index in the plain
    int idx;                                      /// flat index of output plain
    int pHeight = 0;                              /// Height of previous data in the stack
    fcomplexcu ker;                               /// kernel data
    int newStride = noSteps * stride ;            /// New stride based on type of multi-step

#if TEMPLATE_CONVOLVE == 1
    //fcomplexcu inpDat[noPlns*noSteps];    /// set of input data for this thread/column
#else
    //fcomplexcu inpDat[noPlns*MAX_STEPS];  /// set of input data for this thread/column
#endif

    FOLD  // Set relevant x pointers to correct column in kernel, input data & output data  .
    {
      kernels += tid;
      ffdot   += tid;
      inpData += tid;
      for (int plnNo = 0; plnNo < noPlns; plnNo++)      // Loop through the plains  .
      {
        kerDat.val[plnNo] += tid;
      }
    }

    /*
    FOLD // Read the input data into registers and normalise by width  .
    {
#if TEMPLATE_CONVOLVE == 1
#pragma unroll
#endif
      for (int n = 0; n < noPlns*noSteps; n++)
      {
        inpDat[n]           = inpData[ (int)(n * stride) ] ; // Stride for each step
        inpDat[n].r        /= (float) width ;
        inpDat[n].i        /= (float) width ;
      }
    }
    */

#ifndef DEBUG
//#pragma unroll
#endif
    for (int pln = 0; pln < noPlns; pln++)                // Loop through the plains  .
    {
      __restrict__ fcomplexcu inpDat2[noSteps];           /// set of input data for this thread/column
      //__restrict__ fcomplexcu inpDat2[MAX_STEPS];       /// set of input data for this thread/column
      for (int step = 0; step < noSteps; step++)
      {
        inpDat2[step]           = inpData[ (int)(pln * stride*noSteps + step*stride) ] ; // Stride for each step
        inpDat2[step].r        /= (float) width ;
        inpDat2[step].i        /= (float) width ;
      }

      const int datTerm   = pln*noSteps;                  /// Plain dependent term for addressing input data
      const int plainStrt = pHeight*noSteps*stride;       /// The stack index of the start of the plain
      //__restrict__ fcomplexcu* kern    = kerDat.val[plnNo];
      //if ( FLAGS & FLAG_STP_PLN )
      const int ns2           = heights.val[pln] * stride ;

      for (plainY = 0; plainY < heights.val[pln]; plainY++)   // Loop over the individual plain  .
      {
        //int stackY = plainY + pHeight;            /// Base of stack Y

        int stackY = plainY * stride;
        FOLD // Read the kernel value  .
        {
          if ( FLAGS & FLAG_CNV_TEX )
          {
            ker   = *(fcomplexcu*)& tex2D < float2 > (kerTex, tid, plainY + pHeight);
          }
          else
          {
            ker   = kerDat.val[pln][stackY];
            //ker   = *kerDat.val[plnNo];
            //kern += stride;
          }
        }

        FOLD // Calculate stride values that rely on plain and/or y only  .
        {
          if      ( FLAGS & FLAG_STP_ROW )
          {
            stackY    = (plainY + pHeight) * newStride ;
          }
          else if ( FLAGS & FLAG_STP_PLN )
          {
            stackY    += plainStrt ; // Note adding in plainStrt
          }
        }

        //if ( FLAGS & FLAG_STP_PLN )
        //  ffdot += stackY;

#if TEMPLATE_CONVOLVE == 1
#pragma unroll
#endif
        for ( int step = 0; step < noSteps; ++step )            // Loop over steps .
        {
          FOLD // Calculate indices  .
          {
            if      ( FLAGS & FLAG_STP_ROW )
            {
              idx  = stackY + step * stride;
              //idx  += stackY + step * stride;
            }
            else if ( FLAGS & FLAG_STP_PLN )
            {
              idx  = stackY + step * ns2;
            }
            /*
            else if ( FLAGS & FLAG_STP_STK )
            {
              idx  = ( stackY + stackHeight*step) * stride ;
            }
            */
          }

          //const int ox = datTerm + step;    /// Flat index in input data

          // Convolve
          //ffdot[idx].r = (inpDat[ox].r * ker.r + inpDat[ox].i * ker.i);
          //ffdot[idx].i = (inpDat[ox].i * ker.r - inpDat[ox].r * ker.i);
          ffdot[idx].r = (inpDat2[step].r * ker.r + inpDat2[step].i * ker.i);
          ffdot[idx].i = (inpDat2[step].i * ker.r - inpDat2[step].r * ker.i);

          //ffdot->r = (inpDat[ox].r * kern->r + inpDat[ox].i * kern->i);
          //ffdot->i = (inpDat[ox].i * kern->r - inpDat[ox].r * kern->i);

          //ffdot->r = (inpDat2[step].r * kern->r + inpDat2[step].i * kern->i);
          //ffdot->i = (inpDat2[step].i * kern->r - inpDat2[step].r * kern->i);

          //if      ( FLAGS & FLAG_STP_ROW )
          //  ffdot += stride;
        }

        //kern += stride;
      }
      pHeight += heights.val[pln];  // Add to previous height
      //if ( FLAGS & FLAG_STP_PLN )
      //  ffdot += heights.val[plnNo]*noSteps*stride;
    }
  }
}

template<int FLAGS, int noPlns >
__host__ void convolveffdot41_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, const fcomplexcu *kernels, const fcomplexcu *datas, fcomplexcu *ffdot, const int width, const int stride, iHarmList heights, const int stackHeight, cHarmList kerDat, fCplxTex kerTex, const int noSteps )
{
#if TEMPLATE_CONVOLVE == 1
  switch (noSteps)
  {
  case 1:
    convolveffdot41<FLAGS,noPlns,1><<<dimGrid,  dimBlock, i1, cnvlStream >>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
    break;
  case 2:
    convolveffdot41<FLAGS,noPlns,2> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
      break;
  case 3:
    convolveffdot41<FLAGS,noPlns,3> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
      break;
  case 4:
    convolveffdot41<FLAGS,noPlns,4> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
      break;
  case 5:
    convolveffdot41<FLAGS,noPlns,5> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
      break;
  case 6:
    convolveffdot41<FLAGS,noPlns,6> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
      break;
  case 7:
    convolveffdot41<FLAGS,noPlns,7> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
      break;
  case 8:
    convolveffdot41<FLAGS,noPlns,8> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex  );
      break;
  default:
    fprintf(stderr, "ERROR: convolveffdot41 has not been templated for %i steps\n", noSteps);
    exit(EXIT_FAILURE);
  }

#else
  convolveffdot41<FLAGS,noPlns> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps);
#endif
}

template<int FLAGS >
__host__ void convolveffdot41_p(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, const fcomplexcu *kernels, const fcomplexcu *datas, fcomplexcu *ffdot, const int width, const int stride, iHarmList heights, const int stackHeight, cHarmList kerDat, fCplxTex kerTex, int noSteps, int noPlns )
{
  switch (noPlns)
  {
    case 1:
      convolveffdot41_s<FLAGS,1> (dimGrid, dimBlock, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps );
      break;
    case 2:
      convolveffdot41_s<FLAGS,2> (dimGrid, dimBlock, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps );
      break;
    case 3:
      convolveffdot41_s<FLAGS,3> (dimGrid, dimBlock, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps );
      break;
    case 4:
      convolveffdot41_s<FLAGS,4> (dimGrid, dimBlock, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps );
      break;
    case 5:
      convolveffdot41_s<FLAGS,5> (dimGrid, dimBlock, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps );
      break;
    case 6:
      convolveffdot41_s<FLAGS,6> (dimGrid, dimBlock, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps );
      break;
    case 7:
      convolveffdot41_s<FLAGS,7> (dimGrid, dimBlock, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps );
      break;
    case 8:
      convolveffdot41_s<FLAGS,8> (dimGrid, dimBlock, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps );
      break;
    case 9:
      convolveffdot41_s<FLAGS,9> (dimGrid, dimBlock, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps );
      break;
    default:
      fprintf(stderr, "ERROR: convolveffdot41 has not been templated for %i plains\n", noPlns);
      exit(EXIT_FAILURE);
  }
}

__host__ void convolveffdot41_f(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, const fcomplexcu *kernels, const fcomplexcu *datas, fcomplexcu *ffdot, const int width, const int stride, iHarmList heights, const int stackHeight, cHarmList kerDat, fCplxTex kerTex, int noSteps, int noPlns, int FLAGS )
{
  if ( FLAGS & FLAG_CNV_TEX )
  {
    /*
    if      (FLAGS & FLAG_STP_ROW )
      convolveffdot41_p <FLAG_CNV_TEX | FLAG_STP_ROW> (dimGrid, dimBlock, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps, noPlns );
    else if ( FLAGS & FLAG_STP_PLN )
      convolveffdot41_p <FLAG_CNV_TEX | FLAG_STP_PLN> (dimGrid, dimBlock, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps, noPlns );
    //else if ( FLAGS & FLAG_STP_STK )
    //  convolveffdot41_p <FLAG_CNV_TEX | FLAG_STP_STK> (dimGrid, dimBlock, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps, noPlns );
    else
    {
      fprintf(stderr, "ERROR: convolveffdot41 has not been templated for \n", noPlns);
      exit(EXIT_FAILURE);
    }
    */
    fprintf(stderr, "ERROR: convolveffdot41 has not been setup to use texture memory and multi-step\n", noPlns);
    exit(EXIT_FAILURE);
  }
  else
  {
      if      ( FLAGS & FLAG_STP_ROW )
        convolveffdot41_p< FLAG_STP_ROW > (dimGrid, dimBlock, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps, noPlns );
      else if ( FLAGS & FLAG_STP_PLN )
        convolveffdot41_p< FLAG_STP_PLN > (dimGrid, dimBlock, i1, cnvlStream, kernels,  datas, ffdot, width, stride,  heights,  stackHeight, kerDat, kerTex, noSteps, noPlns );
      else
      {
        fprintf(stderr, "ERROR: convolveffdot41 has not been templated for %i plains\n", noPlns);
        exit(EXIT_FAILURE);
      }
  }
}
