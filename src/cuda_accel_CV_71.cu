#include "cuda_accel_CV.h"

/** Convolution kernel - Convolve a multi-step stack - using a 1 plain convolution kernel  .
 * Split the stack into overlapping sections and read the shared kernel values once and convolve with all relevant values
 */
#if TEMPLATE_CONVOLVE == 1
template<uint FLAGS, uint noPlns, uint noSteps>

__global__ void convolveffdot71_k(const fcomplexcu *kernel, const fcomplexcu *datas, cHarmList ffdot, const uint width, const uint stride, iHarmList plnHeights, const uint stkHeight, fCplxTex kerTex, iHarmList zUp, iHarmList zDn )
#else
template<uint FLAGS, uint noPlns>
__global__ void convolveffdot71_k(const fcomplexcu *kernel, const fcomplexcu *datas, cHarmList ffdot, const uint width, const uint stride, iHarmList plnHeights, const uint stkHeight, fCplxTex kerTex, iHarmList zUp, iHarmList zDn, uint noSteps )
#endif
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;    /// The thread ID which is the x location in the step

  if ( tid < width )
  {
    int iyTop = 0;                    // Kernel y index of top section
    int iyBot;                        // Kernel y index of bottom section
    int idxTop, idxBot;               // Plain  y index of top & bottom of section
#if TEMPLATE_CONVOLVE == 1
    fcomplexcu dat[noPlns][noSteps];   // set of input data for this thread
#else
    fcomplexcu dat[noPlns][MAX_STEPS]; // set of input data for this thread
#endif

    FOLD // Stride  .
    {
      kernel          += tid;   // Shift kernel pointer
      datas           += tid;   // Shift data

      // Shift the plain data to the correct x offset
#pragma unroll
      for (int n = 0; n < noPlns; n++)
        ffdot.val[n]  += tid;
    }

    FOLD // Read the input data  .
    {
#if TEMPLATE_CONVOLVE == 1
      #pragma unroll
#endif
      for (int pln = 0; pln < noPlns; pln++)
      {
        for ( int step = 0; step < noSteps; step++ )    // Loop over steps .
        {
          dat[pln][step]     = datas[ pln * stride ] ;
          dat[pln][step].r  /= (float) width ;
          dat[pln][step].i  /= (float) width ;
        }
      }
    }

    FOLD  //  .
    {
      fcomplexcu kerTop[1];
      fcomplexcu kerBot[1];

      FOLD // Loop through sections - read kernel value - convolve with plain values  .
      {
#pragma unroll
        for ( int section = 0; section < noPlns - 1; section++ )                    //
        {
          for ( iyTop = zUp.val[section]; iyTop < zUp.val[section+1] ; iyTop++ )    // Loop over the z values for the kernel for this this section .
          {
            iyBot   = iyTop + zDn.val[noPlns-2-section] ;
            idxTop  = iyTop * stride ;
            idxBot  = iyBot * stride ;

            FOLD  // Read the kernel value  .
            {
              if ( FLAGS & FLAG_CNV_TEX )
              {
                kerTop[0]  = *(fcomplexcu*)& tex2D < float2 > (kerTex, tid, iyTop );
                kerBot[0]  = *(fcomplexcu*)& tex2D < float2 > (kerTex, tid, iyBot );
              }
              else
              {
                kerTop[0]  = kernel[idxTop];
                kerBot[0]  = kernel[idxBot];
              }
            }

            // Loop over all plain values and convolve
#pragma unroll
            for ( int plnNo = 0; plnNo < section+1; plnNo++)  // Loop over sub plains .
            {
              int iyTopPln, iyBotPln;

              FOLD  // Calculate y indices for this plain  .
              {
                if      ( FLAGS & FLAG_ITLV_ROW )
                {
                  //iyTopPln  = ( iyTop - zUp.val[plnNo] ) ;
                  //iyBotPln  = ( iyBot - zUp.val[plnNo] ) ;
                  iyTopPln  = ( iyTop - zUp.val[plnNo] ) * noSteps * stride ;
                  iyBotPln  = ( iyBot - zUp.val[plnNo] ) * noSteps * stride ;
                }
                else if ( FLAGS & FLAG_ITLV_PLN )
                {
                  iyTopPln  = ( iyTop - zUp.val[plnNo] );
                  iyBotPln  = ( iyBot - zUp.val[plnNo] );
                }
                /*
              else if ( FLAGS & FLAG_ITLV_STK )
              {
                iyTopPln  = ( iyTop - zUp.val[plnNo] );
                iyBotPln  = ( iyBot - zUp.val[plnNo] );
              }
                 */
              }

#pragma unroll
              for ( int step = 0; step < noSteps; step++ )    // Loop over steps .
              {
                FOLD // Calculate indices  .
                {
                  if      ( FLAGS & FLAG_ITLV_ROW )
                  {
                    //idxTop  = (iyTopPln * noSteps + step) * stride ;
                    //idxBot  = (iyBotPln * noSteps + step) * stride ;
                    idxTop  = iyTopPln + step * stride ;
                    idxBot  = iyBotPln + step * stride ;
                  }
                  else if ( FLAGS & FLAG_ITLV_PLN )
                  {
                    idxTop  = (iyTopPln + plnHeights.val[plnNo]*step) * stride ;
                    idxBot  = (iyBotPln + plnHeights.val[plnNo]*step) * stride ;
                  }
                  /*
              else if ( FLAGS & FLAG_ITLV_STK )
              {
                idxTop  = (iyTopPln + stkHeight*step) * stride ;
                idxBot  = (iyBotPln + stkHeight*step) * stride ;
              }
                   */
                }

                FOLD // Actual convolution  .
                {
                  (ffdot.val[plnNo])[idxTop].r = ( dat[plnNo][step].r * kerTop[0].r + dat[plnNo][step].i * kerTop[0].i );
                  (ffdot.val[plnNo])[idxTop].i = ( dat[plnNo][step].i * kerTop[0].r - dat[plnNo][step].r * kerTop[0].i );

                  (ffdot.val[plnNo])[idxBot].r = ( dat[plnNo][step].r * kerBot[0].r + dat[plnNo][step].i * kerBot[0].i );
                  (ffdot.val[plnNo])[idxBot].i = ( dat[plnNo][step].i * kerBot[0].r - dat[plnNo][step].r * kerBot[0].i );
                }
              }
            }
          }
        }
      }

      FOLD // Loop through the centre block - convolve with chunks plain values  .
      {
        // I tested reading in "chunks" of kernel and then looping but this made no improvement

        for ( iyTop = zUp.val[noPlns-1]; iyTop < zDn.val[0] ; iyTop += 1 )
        {
          FOLD // Read kernel value  .
          {
            idxTop = ( iyTop ) * stride ;

            // Read the kernel value
            if ( FLAGS & FLAG_CNV_TEX )
            {
              kerTop[0]   = *((fcomplexcu*)& tex2D < float2 > (kerTex, tid, ( iyTop ) )) ;
            }
            else
            {
              kerTop[0]   = kernel[idxTop] ;
            }
          }

          // Now convolve the kernel element with the relevant plain elements
#pragma unroll
          for ( int plnNo = 0; plnNo < noPlns; plnNo++)   // Loop over plains
          {
            float cy = ( (iyTop )  - zUp.val[plnNo] ) * noSteps * stride;

#pragma unroll
            for ( int step = 0; step < noSteps; step++)   // Loop over steps
            {
              FOLD // Calculate indices  .
              {
                if ( FLAGS & FLAG_ITLV_ROW )
                {
                  idxTop  = cy + step * stride ;
                }
                else if ( FLAGS & FLAG_ITLV_PLN )
                {
                  idxTop  = (( (iyTop )  - zUp.val[plnNo] ) + plnHeights.val[plnNo]*step) * stride ;
                }
              }

              FOLD // Actual convolution  .
              {
                (ffdot.val[plnNo])[idxTop].r = (dat[plnNo][step].r * kerTop[0].r + dat[plnNo][step].i * kerTop[0].i);
                (ffdot.val[plnNo])[idxTop].i = (dat[plnNo][step].i * kerTop[0].r - dat[plnNo][step].r * kerTop[0].i);
              }
            }
          }
        }
      }
    }
  }
}

template<uint FLAGS, uint noPlns >
__host__ void convolveffdot71_s(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, const fcomplexcu *kernel, const fcomplexcu *datas, cHarmList ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, fCplxTex kerTex, iHarmList zUp, iHarmList zDn, const uint noSteps )
{
#if TEMPLATE_CONVOLVE == 1
  switch (noSteps)
  {
  case 1:
    cudaFuncSetCacheConfig(convolveffdot71_k<FLAGS,noPlns,1>, cudaFuncCachePreferL1);
    convolveffdot71_k<FLAGS,noPlns,1><<<dimGrid,  dimBlock, i1, cnvlStream >>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
    break;
  case 2:
    cudaFuncSetCacheConfig(convolveffdot71_k<FLAGS,noPlns,2>, cudaFuncCachePreferL1);
    convolveffdot71_k<FLAGS,noPlns,2> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
      break;
  case 3:
    cudaFuncSetCacheConfig(convolveffdot71_k<FLAGS,noPlns,3>, cudaFuncCachePreferL1);
    convolveffdot71_k<FLAGS,noPlns,3> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
      break;
  case 4:
    cudaFuncSetCacheConfig(convolveffdot71_k<FLAGS,noPlns,4>, cudaFuncCachePreferL1);
    convolveffdot71_k<FLAGS,noPlns,4> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
      break;
  case 5:
    cudaFuncSetCacheConfig(convolveffdot71_k<FLAGS,noPlns,5>, cudaFuncCachePreferL1);
    convolveffdot71_k<FLAGS,noPlns,5> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
      break;
  case 6:
    cudaFuncSetCacheConfig(convolveffdot71_k<FLAGS,noPlns,6>, cudaFuncCachePreferL1);
    convolveffdot71_k<FLAGS,noPlns,6> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
      break;
  case 7:
    cudaFuncSetCacheConfig(convolveffdot71_k<FLAGS,noPlns,7>, cudaFuncCachePreferL1);
    convolveffdot71_k<FLAGS,noPlns,7> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
      break;
  case 8:
    cudaFuncSetCacheConfig(convolveffdot71_k<FLAGS,noPlns,8>, cudaFuncCachePreferL1);
    convolveffdot71_k<FLAGS,noPlns,8> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
      break;
  //case 9:
  //  convolveffdot71_k<FLAGS,noPlns,9> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
  //    break;
  //case 10:
  //  convolveffdot71_k<FLAGS,noPlns,10> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
   //   break;
  //case 11:
  //  convolveffdot71_k<FLAGS,noPlns,11> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
  //    break;
  //case MAX_STEPS:
  //  convolveffdot71_k<FLAGS,noPlns,MAX_STEPS> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn );
  //    break;
  default:
    fprintf(stderr, "ERROR: convolveffdot7 has not been templated for %i steps\n", noSteps);
    exit(EXIT_FAILURE);
  }
#else
  cudaFuncSetCacheConfig(convolveffdot71_k<FLAGS,noPlns>, cudaFuncCachePreferL1);
  convolveffdot71_k<FLAGS,noPlns> <<<dimGrid,  dimBlock, i1, cnvlStream>>>(kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps);
#endif
}

template<uint FLAGS >
__host__ void convolveffdot71_p(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, const fcomplexcu *kernel, const fcomplexcu *datas, cHarmList ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, fCplxTex kerTex, iHarmList zUp, iHarmList zDn, const uint noSteps, const uint noPlns )
{
  switch (noPlns)
  {
    case 1:
      convolveffdot71_s<FLAGS,1> (dimGrid, dimBlock, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps );
      break;
    case 2:
      convolveffdot71_s<FLAGS,2> (dimGrid, dimBlock, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps );
      break;
    case 3:
      convolveffdot71_s<FLAGS,3> (dimGrid, dimBlock, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps );
      break;
    case 4:
      convolveffdot71_s<FLAGS,4> (dimGrid, dimBlock, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps );
      break;
    case 5:
      convolveffdot71_s<FLAGS,5> (dimGrid, dimBlock, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps );
      break;
    case 6:
      convolveffdot71_s<FLAGS,6> (dimGrid, dimBlock, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps );
      break;
    case 7:
      convolveffdot71_s<FLAGS,7> (dimGrid, dimBlock, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps );
      break;
    case 8:
      convolveffdot71_s<FLAGS,8> (dimGrid, dimBlock, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps );
      break;
    case 9:
      convolveffdot71_s<FLAGS,9> (dimGrid, dimBlock, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps );
      break;
    default:
      fprintf(stderr, "ERROR: convolveffdot7 has not been templated for %i plains\n", noPlns);
      exit(EXIT_FAILURE);
  }
}

__host__ void convolveffdot71_f(dim3 dimGrid, dim3 dimBlock, int i1, cudaStream_t cnvlStream, const fcomplexcu *kernel, const fcomplexcu *datas, cHarmList ffdot, const int width, const uint stride, iHarmList heights, const uint stackHeight, fCplxTex kerTex, iHarmList zUp, iHarmList zDn, const uint noSteps, const uint noPlns, uint FLAGS )
{
  if ( FLAGS & FLAG_CNV_TEX )
  {
    if      ( FLAGS & FLAG_ITLV_ROW )
      convolveffdot71_p<FLAG_CNV_TEX | FLAG_ITLV_ROW> (dimGrid, dimBlock, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps, noPlns );
    else if ( FLAGS & FLAG_ITLV_PLN )
      convolveffdot71_p<FLAG_CNV_TEX | FLAG_ITLV_PLN> (dimGrid, dimBlock, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps, noPlns );
    //else if ( FLAGS & FLAG_ITLV_STK )
    //  convolveffdot71_p<FLAG_CNV_TEX | FLAG_ITLV_STK> (dimGrid, dimBlock, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps, noPlns );
    else
    {
      fprintf(stderr, "ERROR: convolveffdot7 has not been templated for flag combination. \n");
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    if      ( FLAGS & FLAG_ITLV_ROW )
      convolveffdot71_p< FLAG_ITLV_ROW> (dimGrid, dimBlock, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps, noPlns );
    else if ( FLAGS & FLAG_ITLV_PLN )
      convolveffdot71_p< FLAG_ITLV_PLN> (dimGrid, dimBlock, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps, noPlns );
    //else if ( FLAGS & FLAG_ITLV_STK )
    //  convolveffdot71_p< FLAG_ITLV_STK> (dimGrid, dimBlock, i1, cnvlStream, kernel,  datas, ffdot, width, stride,  heights,  stackHeight, kerTex, zUp, zDn, noSteps, noPlns );
    else
    {
      fprintf(stderr, "ERROR: convolveffdot7 has not been templated for flag combination.\n");
      exit(EXIT_FAILURE);
    }
  }
}
