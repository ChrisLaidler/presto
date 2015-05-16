#include "cuda_accel_CV.h"

/** Convolution kernel - Convolve a stack - using a 1 plain convolution kernel  .
 * Split the stack into overlapping sections and read the shared kernel values once and convolve with all relevant values
 * NOTE: does not with multi step stacks
 */
template<uint FLAGS, uint no>
__global__ void convolveffdot6(const fcomplexcu *kernels, const fcomplexcu *datas, cHarmList ffdot, const int width, const uint stride, iHarmList heights, const uint hh, fCplxTex kerTex, iHarmList zUp, iHarmList zDn )
{
  const int bidx = threadIdx.y * CNV_DIMX + threadIdx.x;
  const int tid  = blockIdx.x  * CNV_DIMX * CNV_DIMY + bidx;

  if ( tid < width )
  {
    int iy = 0;             // y index
    int iy2;                // Upper Y value
    int idx1, idx2;         // flat index
    int plnNo;
    int s = 0;

    if ( no == 1 )
    {
      fcomplexcu ker;     // kernel data
      fcomplexcu dat;     // set of input data for this thread

      // Stride
      kernels         += tid;
      ffdot.val[0]    += tid;
      datas           += tid;

      dat = datas[ 0 ] ;
      dat.r /= (float) width ;
      dat.i /= (float) width ;

      for (; iy < heights.val[0]; iy++)     // Loop over the plain
      {
        idx1 = (iy) * stride ;

        if ( FLAGS & FLAG_CNV_TEX )
        {
          ker   = *(fcomplexcu*)& tex2D < float2 > (kerTex, tid, iy);
        }
        else
        {
          ker   = kernels[idx1];
        }

        ffdot.val[0][idx1].r = (dat.r * ker.r + dat.i * ker.i);
        ffdot.val[0][idx1].i = (dat.i * ker.r - dat.r * ker.i);
      }

    }
    else
    {
      const int noSteps = 2;
      fcomplexcu ker1[noSteps];
      fcomplexcu ker2[1];
      fcomplexcu dat[no];     // set of input data for this thread

      // Stride
      kernels += tid;
      datas   += tid;

#pragma unroll
      for (int n = 0; n < no; n++)
      {
        dat[n]          = datas[ /* tid + */ n * stride] ;
        dat[n].r        /= (float) width ;
        dat[n].i        /= (float) width ;

        ffdot.val[n]    += tid;
      }

      // Loop up through steps
#pragma unroll
      for ( int step = 0; step < no - 1; step++ )
      {
        for ( iy = zUp.val[step]; iy < zUp.val[step+1] ; iy++ ) // Loop over the z values for the current sub plans
        {
          //int mx = noSteps;

          //if ( (iy + mx) > zUp.val[step+1] )
            //mx = zUp.val[step+1] - iy;

//#pragma unroll
          //for ( int s = 0; s < mx ; s++ )
          {
            iy2   = iy  + zDn.val[no-2-step] ;
            idx1  = iy  * stride ;
            idx2  = iy2 * stride ;

            // Read the kernel value
            if ( FLAGS & FLAG_CNV_TEX )
            {
              ker1[s]  = *(fcomplexcu*)& tex2D < float2 > (kerTex, tid, iy  );
              ker2[s]  = *(fcomplexcu*)& tex2D < float2 > (kerTex, tid, iy2 );
            }
            else
            {
              ker1[s]  = kernels[idx1];
              ker2[s]  = kernels[idx2];
            }
          }

#pragma unroll
          for ( int dd = 0; dd < step+1; dd++) // Loop over sub plains
          {
            //for ( int s = 0; s < mx ; s++ )
            {
              idx1  = ( iy  - zUp.val[dd] ) * stride ; // TODO: can do the dd offset at the beginning
              idx2  = ( iy2 - zUp.val[dd] ) * stride ; // TODO: can do the dd offset at the beginning

              plnNo = dd;

              (ffdot.val[plnNo])[idx1].r = (dat[plnNo].r * ker1[s].r + dat[plnNo].i * ker1[s].i);
              (ffdot.val[plnNo])[idx1].i = (dat[plnNo].i * ker1[s].r - dat[plnNo].r * ker1[s].i);

              (ffdot.val[plnNo])[idx2].r = (dat[plnNo].r * ker2[s].r + dat[plnNo].i * ker2[s].i);
              (ffdot.val[plnNo])[idx2].i = (dat[plnNo].i * ker2[s].r - dat[plnNo].r * ker2[s].i);
            }
          }
        }
      }

      // Loop through the centre block
      for ( iy = zUp.val[no-1]; iy < zDn.val[0] ; iy += noSteps )
      {
        int mx = noSteps;

        if ( (iy + mx) > zDn.val[0] )
          mx = zDn.val[0] - iy;

        for ( int s = 0; s < mx ; s++ )
        {
          idx1 = ( iy + s ) * stride ;

          // Read the kernel value
          if ( FLAGS & FLAG_CNV_TEX )
          {
            ker1[s]   = *((fcomplexcu*)& tex2D < float2 > (kerTex, tid, ( iy + s ) )) ;
          }
          else
          {
            ker1[s]    = kernels[idx1] ;
          }
        }

#pragma unroll
        for ( int dd = 0; dd < no; dd++) // Loop over sub plains
        {
          for ( int s = 0; s < mx ; s++ )
          {
            idx1 = ( (iy+s) -zUp.val[dd] ) * stride ; // TODO: can do the dd offset at the beginning

            (ffdot.val[dd])[idx1].r = (dat[dd].r * ker1[s].r + dat[dd].i * ker1[s].i);
            (ffdot.val[dd])[idx1].i = (dat[dd].i * ker1[s].r - dat[dd].r * ker1[s].i);
          }
        }
      }
    }
  }
}
