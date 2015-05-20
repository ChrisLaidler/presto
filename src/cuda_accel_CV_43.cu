#include "cuda_accel_CV.h"

/** Convolution kernel - Convolve a stack with a kernel - multi-step - uses shared memory to store sections of the kernel  .
 * Each thread loops down a column of the plain
 * Reads the input and convolves it with the kernel and writes result to plain
 */
template<int FLAGS, int noSteps, int noPlns>
__global__ void convolveffdot43_k(const __restrict__ fcomplexcu* kernels, const __restrict__ fcomplexcu* inpData, __restrict__ fcomplexcu* ffdot, const int width, const int stride, const int firstPlain )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;          /// Block ID - flat index
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;      /// Global thread ID - flat index ie column index of stack

  const int cv_chunkSZ = 12;

  if ( tid < width )  // Valid thread  .
  {
    //int idx;                                      /// flat index of output plain
    //int pHeight = 0;                              /// Height of previous data in the stack
    //fcomplexcu ker;                               /// kernel data

    const int kerHeight = HEIGHT_FAM_ORDER[firstPlain];       // The size of the kernel
    const int bStride   = CNV_DIMX*CNV_DIMY;

    //__shared__ fcomplexcu ker_sm[CNV_DIMX*CNV_DIMY*cv_chunkSZ];
    //fcomplexcu ker_smP[cv_chunkSZ];
    //fcomplexcu* ker_smP = ker_sm;

    FOLD  // Stride, kernel, input data & output data  .
    {
      kernels += tid;
      ffdot   += tid;
      inpData += tid;
      //ker_smP  = &ker_sm[bidx];
    }

    FOLD // TMP - zero values  .
    {
      for( int c = 0; c < cv_chunkSZ; c++ )
      {
        //ker_smP[c].r = 0;
        //ker_smP[c].i = 0;
      }
    }

    ///*__restrict__*/ fcomplexcu inpDat[noSteps][noPlns];          // Set of input data for this thread/column
    /*__restrict__*/ fcomplexcu inpDat[noPlns][noSteps];          // Set of input data for this thread/column

    FOLD // Read input data for this plain  .
    {
      for (int step = 0; step < noSteps; step++)
      {
        for (int pln = 0; pln < noPlns; pln++)                // Loop through the plains  .
        {
          fcomplexcu ipd        = inpData[ (int)(pln*noSteps*stride + step*stride) ];
          ipd.r                 /= (float) width;
          ipd.i                 /= (float) width;
          inpDat[pln][step]     = ipd;
        }
      }
    }

    int noChunks = ceilf(kerHeight / (float)cv_chunkSZ);

    for (int chunk = 0; chunk < noChunks; chunk++ )
    {
      const int c0  = chunk*cv_chunkSZ;
      const int c1  = MIN(cv_chunkSZ,kerHeight-c0);
      int pHeight   = 0;

      /*
      if ( tid == 0 )
      {
        printf("\n");
        printf("chunk %02i  c0: %02i  len %i  \n", chunk, c0, c1);
      }
      */

      fcomplexcu k0   = kernels[(c0+0 )*stride];
      fcomplexcu k1   = kernels[(c0+1 )*stride];
      fcomplexcu k2   = kernels[(c0+2 )*stride];
      fcomplexcu k3   = kernels[(c0+3 )*stride];
      fcomplexcu k4   = kernels[(c0+4 )*stride];
      fcomplexcu k5   = kernels[(c0+5 )*stride];
      fcomplexcu k6   = kernels[(c0+6 )*stride];
      fcomplexcu k7   = kernels[(c0+7 )*stride];
      fcomplexcu k8   = kernels[(c0+8 )*stride];
      fcomplexcu k9   = kernels[(c0+9 )*stride];
      fcomplexcu k10  = kernels[(c0+10)*stride];
      fcomplexcu k11  = kernels[(c0+11)*stride];

      FOLD // Load kernel data into SM  .
      {
        for( int c = 0; c < c1; c++ )
        {
          //ker_sm[c*bStride + bidx] = kernels[(c0+c)*stride];
          //ker_smP[c] = kernels[(c0+c)*stride];

          //if(ker_smP[c].r < 0 && ker_smP[c].r > 0 )
          {
            //printf("hi");
            //int tmp = 0;
          }
        }
      }

//#pragma unroll
      for (int pln = 0; pln < noPlns; pln++)                    // Loop through the plains of the stack  .
      {
        const int plnHeight     = HEIGHT_FAM_ORDER[firstPlain + pln];
        const int kerYOffset    = (kerHeight - plnHeight)/2;

        //const int p0            = c0 - kerYOffset;
        const int p0            = MAX(c0 - kerYOffset,0);

        //const int p1            = c0 + c1 - kerYOffset;
        //const int p1            = p0 + c1;
        //const int p1            = MIN(p0 + c1, plnHeight);
        const int p1            = MIN(c0 + c1 - kerYOffset, plnHeight);

        //const int kerAddd       = c1-p1;
        const int kerAddd       = MAX(0, kerYOffset - c0);

        const int ns2           = plnHeight * stride;

//        if ( tid == 0 )
//        {
//          if ( p1 > p0 )
//            printf("pln %02i  kerYOffset: %02i  kerAddd %02i  p0 %03i  p1 %03i \n", pln, kerYOffset, kerAddd, p0, p1 );
//        }

        for (int plainY = p0; plainY < p1; plainY++)            // Loop over the individual plain  .
        {
          int y = plainY - p0 + kerAddd;
          fcomplexcu ker;

          FOLD // Read the kernel value  .
          {
            //ker   = kernels[(kerYOffset+plainY)*stride];
            //ker   = ker_sm[(plainY-p0+kerAddd)*bStride + bidx];
            //ker   = ker_smP[(plainY-p0+kerAddd)];
            //ker   = ker_smP[y+kerAddd];

            //if ( ker.r < 0 && ker.r > 0 )
            {
              //printf("%i %i %f\n", kerAddd, plainY-p0+kerAddd, ker_smP[(plainY-p0+kerAddd)].r );
              //printf("%i %i %f\n", kerAddd, plainY-p0+kerAddd );
            }
            //ker.r = 1/(plainY-p0);
            //ker.i = 1/(plainY-p0);
            //ker.r = 1/(y);
            //ker.i = 1/(y);

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
              offsetPart1  = pHeight + plainY*noSteps*stride;
            }
            else if ( FLAGS & FLAG_ITLV_PLN )
            {
              offsetPart1  = pHeight + plainY*stride;
            }
          }

#pragma unroll
          for ( int step = 0; step < noSteps; step++ )          // Loop over steps .
          {
            int idx;

            FOLD // Calculate offset  .
            {
              if      ( FLAGS & FLAG_ITLV_ROW )
              {
                idx  = offsetPart1 + step * stride;
              }
              else if ( FLAGS & FLAG_ITLV_PLN )
              {
                idx  = offsetPart1 + step * ns2;
              }
            }

            FOLD // Convolve  .
            {
              //ffdot[idx].r = (inpDat[pln][step].r * ker.r + inpDat[pln][step].i * ker.i);
              //ffdot[idx].i = (inpDat[pln][step].i * ker.r - inpDat[pln][step].r * ker.i);

              //fcomplexcu vv;
              //vv.r = (inpDat[pln][step].r * ker.r + inpDat[pln][step].i * ker.i);
              //vv.i = (inpDat[pln][step].i * ker.r - inpDat[pln][step].r * ker.i);
              //ffdot[idx] = vv;

              fcomplexcu ipd = inpDat[pln][step];
              fcomplexcu vv;
              vv.r = (ipd.r * ker.r + ipd.i * ker.i);
              vv.i = (ipd.i * ker.r - ipd.r * ker.i);
              ffdot[idx] = vv;
            }
          }
        }
        pHeight += plnHeight * noSteps * stride;
      }
    }
  }
}

template<int FLAGS, int noSteps>
__host__  void convolveffdot43_p(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack)
{
  int offset = 0;
  for ( int i = 0; i < stack; i++)
  {
    offset += batch->stacks[i].noInStack;
  }
  cuFfdotStack* cStack = &batch->stacks[stack];

  switch (cStack->noInStack)
  {
    case 1	:
    {
      cudaFuncSetCacheConfig(convolveffdot43_k<FLAGS,noSteps,1>, cudaFuncCachePreferL1);
      convolveffdot43_k<FLAGS,noSteps,1><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 2	:
    {
      cudaFuncSetCacheConfig(convolveffdot43_k<FLAGS,noSteps,2>, cudaFuncCachePreferL1);
      convolveffdot43_k<FLAGS,noSteps,2><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 3	:
    {
      cudaFuncSetCacheConfig(convolveffdot43_k<FLAGS,noSteps,3>, cudaFuncCachePreferL1);
      convolveffdot43_k<FLAGS,noSteps,3><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 4	:
    {
      cudaFuncSetCacheConfig(convolveffdot43_k<FLAGS,noSteps,4>, cudaFuncCachePreferL1);
      convolveffdot43_k<FLAGS,noSteps,4><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 5	:
    {
      cudaFuncSetCacheConfig(convolveffdot43_k<FLAGS,noSteps,5>, cudaFuncCachePreferL1);
      convolveffdot43_k<FLAGS,noSteps,5><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 6	:
    {
      cudaFuncSetCacheConfig(convolveffdot43_k<FLAGS,noSteps,6>, cudaFuncCachePreferL1);
      convolveffdot43_k<FLAGS,noSteps,6><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 7	:
    {
      cudaFuncSetCacheConfig(convolveffdot43_k<FLAGS,noSteps,7>, cudaFuncCachePreferL1);
      convolveffdot43_k<FLAGS,noSteps,7><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 8	:
    {
      cudaFuncSetCacheConfig(convolveffdot43_k<FLAGS,noSteps,8>, cudaFuncCachePreferL1);
      convolveffdot43_k<FLAGS,noSteps,8><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    case 9	:
    {
      cudaFuncSetCacheConfig(convolveffdot43_k<FLAGS,noSteps,9>, cudaFuncCachePreferL1);
      convolveffdot43_k<FLAGS,noSteps,9><<<dimGrid, dimBlock, i1, cnvlStream>>>(cStack->d_kerData , cStack->d_iData, cStack->d_plainData, cStack->width, cStack->strideCmplx, offset);
      break;
    }
    default	:
    {
      fprintf(stderr, "ERROR: convolveffdot43 has not been templated for %i plains in a stack.\n",cStack->noInStack);
      exit(EXIT_FAILURE);
    }
  }
}

template<int FLAGS>
__host__  void convolveffdot43_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack)
{
  switch (batch->noSteps)
  {
    case 1	:
    {
      convolveffdot43_p<FLAGS,1>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 2	:
    {
      convolveffdot43_p<FLAGS,2>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 3	:
    {
      convolveffdot43_p<FLAGS,3>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 4	:
    {
      convolveffdot43_p<FLAGS,4>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 5	:
    {
      convolveffdot43_p<FLAGS,5>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 6	:
    {
      convolveffdot43_p<FLAGS,6>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 7	:
    {
      convolveffdot43_p<FLAGS,7>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    case 8	:
    {
      convolveffdot43_p<FLAGS,8>(dimGrid, dimBlock, i1, cnvlStream, batch, stack);
      break;
    }
    default	:
    {
      fprintf(stderr, "ERROR: convolveffdot42 has not been templated for %lu steps\n", batch->noSteps);
      exit(EXIT_FAILURE);
    }
  }
}

__host__  void convolveffdot43_f(cudaStream_t cnvlStream, cuFFdotBatch* batch, uint stack)
{
  dim3 dimGrid, dimBlock;

  cuFfdotStack* cStack = &batch->stacks[stack];

  dimBlock.x = CNV_DIMX;
  dimBlock.y = CNV_DIMY;

  dimGrid.x = ceil(cStack->width / (float) ( CNV_DIMX * CNV_DIMY ));
  dimGrid.y = 1;


  if      ( batch->flag & FLAG_ITLV_ROW )
    convolveffdot43_s<FLAG_ITLV_ROW>(dimGrid, dimBlock, 0, cnvlStream, batch, stack);
  else if ( batch->flag & FLAG_ITLV_PLN )
    convolveffdot43_s<FLAG_ITLV_PLN>(dimGrid, dimBlock, 0, cnvlStream, batch, stack);
  else
  {
    fprintf(stderr, "ERROR: convolveffdot43 has not been templated for layout.\n");
    exit(EXIT_FAILURE);
  }
}
