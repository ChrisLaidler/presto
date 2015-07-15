#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"


extern "C"
{
#define __float128 long double
#include "accel.h"
}


__device__ inline double cos_t(double x)
{
  return cos(x);
}
__device__ inline float cos_t(float x)
{
  return cosf(x);
}

__device__ inline double sin_t(double x)
{
  return sin(x);
}
__device__ inline float sin_t(float x)
{
  return sinf(x);
}

__device__ inline double sqrt_t(double x)
{
  return sqrt(x);
}
__device__ inline float sqrt_t(float x)
{
  return sqrtf(x);
}

__device__ inline void sincos_t(double x, double* s, double* c )
{
  sincos(x, s, c);
}
__device__ inline void sincos_t(float x, float* s, float* c )
{
  sincosf(x, s, c);
}


template<typename T>
__device__ void fresnl(T xxa, T* ss, T* cc)
{
  T f, g, c, s, t, u;
  T x, x2;

  x       = fabs(xxa);
  x2      = x * x;

  if      ( x2 < 2.5625   )    	// Small so use a polynomial approximation  .
  {
    t     = x2 * x2;

    T sn  = (T)3.18016297876567817986e11 + ((T)-4.42979518059697779103e10 + ((T)2.54890880573376359104e9  + ((T)-6.29741486205862506537e7  + ( (T)7.08840045257738576863e5  -   (T)2.99181919401019853726e3   * t)*t)*t)*t)*t;
    T sd  = (T)6.07366389490084639049e11 + ( (T)2.24411795645340920940e10 + ((T)4.19320245898111231129e8  + ( (T)5.17343888770096400730e6  + ( (T)4.55847810806532581675e4  + ( (T)2.81376268889994315696e2   + t)*t)*t)*t)*t)*t ;
    T cn  = (T)9.99999999999999998822e-1 + ((T)-2.05525900955013891793e-1 + ((T)1.88843319396703850064e-2 + ((T)-6.45191435683965050962e-4 + ( (T)9.50428062829859605134e-6 -   (T)4.98843114573573548651e-8  * t)*t)*t)*t)*t;
    T cd  = (T)1.00000000000000000118e0  + ( (T)4.12142090722199792936e-2 + ((T)8.68029542941784300606e-4 + ( (T)1.22262789024179030997e-5 + ( (T)1.25001862479598821474e-7 + ( (T)9.15439215774657478799e-10 + (T)3.99982968972495980367e-12*t)*t)*t)*t)*t)*t ;

    *ss   = x * x2 * sn / sd;
    *cc   = x * cn / cd;
  }
  else if ( x  > 36974.0  )     // Asymptotic behaviour  .
  {
    *cc   = 0.5;
    *ss   = 0.5;
  }
  else                          // Auxiliary functions for large argument  .
  {
    x2    = x * x;
    t     = (T)PI * x2;
    u     = 1.0 / (t * t);
    t     = 1.0 / t;

    T fn  = (T)3.76329711269987889006e-20+((T)1.34283276233062758925e-16+((T)1.72010743268161828879e-13+((T)1.02304514164907233465e-10+((T)3.05568983790257605827e-8 +((T)4.63613749287867322088e-6+((T)3.45017939782574027900e-4+((T)1.15220955073585758835e-2+((T)1.43407919780758885261e-1+ (T)4.21543555043677546506e-1*u)*u)*u)*u)*u)*u)*u)*u)*u;
    T fd  = (T)1.25443237090011264384e-20+((T)4.52001434074129701496e-17+((T)5.88754533621578410010e-14+((T)3.60140029589371370404e-11+((T)1.12699224763999035261e-8 +((T)1.84627567348930545870e-6+((T)1.55934409164153020873e-4+((T)6.44051526508858611005e-3+((T)1.16888925859191382142e-1+((T)7.51586398353378947175e-1+u)*u)*u)*u)*u)*u)*u)*u)*u)*u ;
    T gn  = (T)1.86958710162783235106e-22+((T)8.36354435630677421531e-19+((T)1.37555460633261799868e-15+((T)1.08268041139020870318e-12+((T)4.45344415861750144738e-10+((T)9.82852443688422223854e-8+((T)1.15138826111884280931e-5+((T)6.84079380915393090172e-4+((T)1.87648584092575249293e-2+((T)1.97102833525523411709e-1+ (T)5.04442073643383265887e-1*u)*u)*u)*u)*u)*u)*u)*u)*u)*u ;
    T gd  = (T)1.86958710162783236342e-22+((T)8.39158816283118707363e-19+((T)1.38796531259578871258e-15+((T)1.10273215066240270757e-12+((T)4.60680728146520428211e-10+((T)1.04314589657571990585e-7+((T)1.27545075667729118702e-5+((T)8.14679107184306179049e-4+((T)2.53603741420338795122e-2+((T)3.37748989120019970451e-1+((T)1.47495759925128324529e0 +u)*u)*u)*u)*u)*u)*u)*u)*u)*u)*u ;

    f     = 1.0 - u * fn / fd;
    g     =       t * gn / gd;

    t     = (T)PIBYTWO * x2;
    sincos(t, &s, &c);
    t     = (T)PI * x;
    *cc   = 0.5 + (f * s - g * c) / t;
    *ss   = 0.5 - (f * c + g * s) / t;
  }

  if (xxa < 0.0)                // Swap as function is antisymmetric  .
  {
    *cc   = -*cc;
    *ss   = -*ss;
  }
}


/** Generate the complex response value for Fourier f-dot interpolation  .
 *
 * This is based on gen_z_response in responce.c
 *
 * @param rx            The x index of the value in the kernel
 * @param z             The Fourier Frequency derivative (# of bins the signal smears over during the observation)
 * @param absz          Is the absolute value of z
 * @param roffset       Is the offset in Fourier bins for the full response (i.e. At this point, the response would equal 1.0)
 * @param numbetween    Is the number of points to interpolate between each standard FFT bin. (i.e. 'numbetween' = 2 = interbins, this is the standard)
 * @param numkern       Is the number of complex points that the kernel will contain.
 * @param rr            A pointer to the real part of the complex response for rx
 * @param ri            A pointer to the imaginary part of the complex response for rx
 */
template<typename T>
__device__ inline void gen_z_response(int rx, T z,  T absz, T numbetween, int numkern, float* rr, float* ri)
{
  int signz;
  T zd, r, xx, yy, zz, startr, startroffset;
  T fressy, frescy, fressz, frescz, tmprl, tmpim;
  T s, c, pibyz, cons, delta;

  T zT = z;
  T rT = r;

  startr        = 0 - (0.5 * zT);
  startroffset  = (startr < 0) ? 1.0 + modf(startr, &tmprl) : modf(startr, &tmprl);

  if (rx == numkern / 2.0 && startroffset < 1E-3 && absz < 1E-3)
  {
    T nr, ni;

    zz      = zT * zT;
    xx      = startroffset * startroffset;
    nr      = (T)1.0 - (T)0.16449340668482264365 * zz;
    ni      = (T)-0.5235987755982988731 * zT;
    nr      += startroffset * (T)1.6449340668482264365 * zT;
    ni      += startroffset * ((T)PI - (T)0.5167712780049970029 * zz);
    nr      += xx * ((T)-6.579736267392905746 + (T)0.9277056288952613070 * zz);
    ni      += xx * ((T)3.1006276680299820175 * zT);

    *rr     = nr;
    *ri     = ni;
  }
  else
  {
    /* This is evaluating Eq (39) in:
     * Ransom, Scott M., Stephen S. Eikenberry, and John Middleditch. "Fourier techniques for very long astrophysical time-series analysis." The Astronomical Journal 124.3 (2002): 1788.
     *
     * Where: qᵣ  is the variable r and represents the distance from the centre frequency
     *        |ṙ| is the variable z which is ḟ
     */

    signz   = (zT < 0.0) ? -1 : 1;
    zd      = signz * (T)SQRT2 / sqrt(absz);
    zd      = signz * sqrt(2.0 / absz);
    cons    = zd / 2.0;                             // 1 / sqrt(2*r')

    startr  += numkern / (T) (2 * numbetween);
    delta   = -1.0 / numbetween;
    r       = startr + rx * delta;

    pibyz   = (T)PI / zT;
    yy      = rT * zd;
    zz      = yy + zT * zd;
    xx      = pibyz * rT * rT;

    sincos_t(xx, &s, &c);
    fresnl<T>(yy, &fressy, &frescy);
    fresnl<T>(zz, &fressz, &frescz);

    tmprl   = signz * (frescz - frescy);
    tmpim   = fressy - fressz;

    *rr     =  (tmprl * c - tmpim * s) * cons;
    *ri     = -(tmprl * s + tmpim * c) * cons;
  }
}

/* This routine uses the correlation method to do a Fourier        */
/* complex interpolation at a single point in the f-fdot plane.    */
/* It does the correlations manually. (i.e. no FFTs)               */
/* Arguments:                                                      */
/*   'data' is a complex array of the data to be interpolated.     */
/*   'numdata' is the number of complex points (bins) in data.     */
/*   'r' is the Fourier frequency in data that we want to          */
/*      interpolate.  This can (and should) be fractional.         */
/*   'z' is the fdot to use (z=f-dot*T^2 (T is integration time)). */
/*   'kern_half_width' is the half-width of the kernel in bins.    */
/*   'ans' is the complex answer.                                  */
template<typename T>
__device__ fcomplexcu rz_interp_cu(fcomplexcu* data, int loR, int noBins, double r, double z, int kern_half_width)
{
  int numkern, intfreq;
  double  fracfreq;
  double  dintfreq;
  int signz;
  int ii, lodata;
  T absz, zd, q_r, xx, Yr, Zr, startr;
  T fressy, frescy, fressz, frescz;
  T s, c, pibyz, cons, sinc;

  T zT = z;
  T rT = r;

  fcomplexcu inp;
  fcomplexcu ans;

  T tR, tI;     // Response values

  ans.r = 0.0;
  ans.i = 0.0;

  if ( r > 0 )
  {
    // Split 'r' into integer and fractional parts
    fracfreq          = modf(r, &dintfreq); // This has to be double precision
    intfreq           = (int) dintfreq;
    numkern           = 2 * kern_half_width;
    lodata            = intfreq - kern_half_width;

    // Set up values dependent on Z alone
    absz              = fabs(zT);
    startr            = fracfreq - (0.5 * z);
    signz             = (zT < 0.0) ? -1 : 1;
    zd                = signz * (T)SQRT2 / sqrt(absz);
    cons              = zd / 2.0;
    pibyz             = (T)PI / zT;
    startr            += kern_half_width;

    if ( absz < 1E-4 )
    {
      startr = rT - lodata;
    }

    FOLD // Clamp values to usable bounds  .
    {
      if ( lodata < 0 )
      {
        //printf("lodata < 0\n");
        numkern += lodata;
        startr  += lodata;
        lodata  = 0;
      }

      //printf("lodata: %i    loR: %i  \n", lodata, loR);
      lodata -= loR;

      //printf("lodata + numkern: %i  noR: %i \n", lodata + numkern, noBins );
      if ( lodata + numkern >= noBins )
      {
        //        printf("lodata + numkern >= noBins\n");
        //        printf("%i + %i >= %i\n", lodata, numkern, noBins );
        numkern = noBins - lodata;
      }
      //printf("numkern: %i\n", numkern );
    }

    // Loop over positions, calculate response values and do multiplications
    for ( ii = 0, q_r = startr; ii < numkern; q_r--, ii++ )
    {
      FOLD //  Read the input value  .
      {
        inp             = data[lodata+ii];
      }

      FOLD // Calculate response value  .
      {
        if ( absz < 1E-4 ) // Just do a Fourier Interpolation
        {
          xx              = (T)PI*q_r ;

          sincos_t(xx, &s, &c);

          if (q_r == 0.0)
            sinc = 1.0;
          else
            sinc = s / xx;

          tR              = c * sinc;
          tI              = s * sinc;

          //printf("%04i response: %15.10f %15.10f  r: %15.10f  c: %15.10f s: %15.10f sinc: %15.10f\n", ii, tR, tI, q_r, c, s, sinc );
        }
        else
        {
          Yr              = q_r * zd;
          Zr              = Yr + zT * zd;
          xx              = pibyz * q_r * q_r;

          sincos_t(xx, &s, &c);
          fresnl<T>(Yr, &fressy, &frescy);
          fresnl<T>(Zr, &fressz, &frescz);

          T Ster          = fressz - fressy;
          T Cter          = frescy - frescz;
          tR              = cons * (c*Ster + signz*s*Cter);
          tI              = cons * (s*Ster - signz*c*Cter);
        }
      }

      FOLD // Do the multiplication  .
      {
        ans.r           += tR * inp.r - tI*inp.i;
        ans.i           += tR * inp.i + tI*inp.r;
      }

      //printf("%03i %05i Data %12.2f %12.2f  Response: %13.10f %13.10f   %12.2f \n", ii, loR+lodata+ii, inp.r, inp.i, tR, tI, POWERR(ans.r, ans.i) );
    }
  }

  return ans;
}

template<typename T>
__global__ void ffdot_ker(float* powers, fcomplexcu* fft, int noHarms, int halfwidth, double firstR, double firstZ, double rSZ, double zSZ, int noR, int noZ, int iStride, int oStride, int16 loR, float16 norm)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if ( ix < noR && iy < noZ)
  {
    double r = firstR + ix/(double)(noR-1) * rSZ ;
    double z = firstZ - iy/(double)(noZ-1) * zSZ ;
    float total_power = 0;

    for( int i = 1; i <= noHarms; i++ )
    {
      fcomplexcu ans  = rz_interp_cu<T>(&fft[iStride*(i-1)], loR.val[i-1], iStride, r*i, z*i, halfwidth);

      //total_power     += POWERR(ans.r, ans.i)/norm.val[i-1];
      total_power     += POWERR(ans.r, ans.i);

//      if( ix == 0 && iy > 70 )
//      {
//        printf("%03i %.3f\n", iy, POWERR(ans.r, ans.i) );
//      }
    }

    //powers[iy*noR + ix] = total_power;
    powers[iy*oStride + ix] = total_power;
  }
}

int ffdotPln(float* powers, fcomplex* fft, int loR, int noBins, int noHarms, double centR, double centZ, double rSZ, double zSZ, int noR, int noZ, int halfwidth, float* fac)
{
  double log2 = log(2.0);


  double maxZ = (centZ + zSZ/2.0);
  double minZ = (centZ - zSZ/2.0);
  double minR = (centR - rSZ/2.0);
  double maxR = (centR + rSZ/2.0);

  int halfwidth2    = z_resp_halfwidth(MAX(fabs(maxZ*noHarms), fabs(minZ*noHarms))+4, HIGHACC);
  //halfwidth         = MAX(halfwidth,halfwidth2);

  double rSpread    = ceil(maxR*noHarms  + halfwidth) - floor(minR*noHarms - halfwidth);

  size_t iStride, pStride;
  float *cuPowers;
  fcomplexcu *cuInp;
  fcomplexcu *cpuInp;
  double factor;

  CUDA_SAFE_CALL(cudaMallocPitch(&cuPowers,  &pStride, noR     * sizeof(float),             noZ),   "Failed to allocate device memory for kernel stack.");
  CUDA_SAFE_CALL(cudaMallocPitch(&cuInp,     &iStride, rSpread * sizeof(cufftComplex),  noHarms),   "Failed to allocate device memory for kernel stack.");

  int noInp = iStride/sizeof(cufftComplex);
  int noPow = pStride/sizeof(float);

  int16   rOff;
  float16 norm;

  cpuInp = (fcomplexcu*) malloc(iStride*noHarms);

  for( int h = 0; h < 16; h++)
  {
    rOff.val[h] = 0;
  }

  for( int h = 0; h < noHarms; h++)
  {
    rOff.val[h]   = floor( minR*(h+1) - halfwidth );
    //printf("%i  %f   %i\n", (int)floor(minR*(h+1)), minR*(h+1), halfwidth );

    int datStart  = floor( minR*(h+1) - halfwidth );
    int datEnd    = ceil ( maxR*(h+1) + halfwidth );
    int novals    = datEnd - datStart;
    int noPowers, off;
    float medianv;

    FOLD // Calculate normalisation factor  .
    {
      if ( fac == NULL) // Calculate the normalisation factor  .
      {
        float*  normPow = (float*) malloc(noInp*sizeof(float));

        if ( datStart < loR )
        {
          novals    -=  (loR - datStart);
          datStart  =   loR;
        }
        if ( datEnd >= noBins )
        {
          novals    -=  (datEnd - noBins - 1);
          datEnd    =   noBins-1;
        }

        noPowers = 0;
        for ( int i = 0; i < noInp; i++)
        {
          off = rOff.val[h] - loR + i;
          if (off >= 0 && off < noBins )
          {
            normPow[noPowers++] = POWERR(fft[off].r, fft[off].i ) ;
          }
        }

        medianv       = median(normPow, noPowers);
        factor        = sqrt(medianv/log2);
        //printf("  %02i  %8.3f \n", h+1, factor );

        free(normPow);
      }
      else              // Use precalcualted normalisation factor  .
      {
        factor = sqrt(fac[h]);
      }
      norm.val[h] = fac[h];
      //factor = 1.0;
    }

    for ( int i = 0; i < noInp; i++) // Normalise input  .
    {
      off = rOff.val[h] - loR + i;
      if (off >= 0 && off < noBins && i < novals)
      {
        cpuInp[h*noInp + i].r = fft[off].r / factor ;
        cpuInp[h*noInp + i].i = fft[off].i / factor ;
      }
      else
      {
        cpuInp[h*noInp + i].r = 0;
        cpuInp[h*noInp + i].i = 0;
      }
    }
  }

  CUDA_SAFE_CALL(cudaMemcpy(cuInp, cpuInp, iStride*noHarms, cudaMemcpyHostToDevice), "Copying optimisation input to the device");

  FOLD // Call kernel  .
  {
    dim3 dimBlock, dimGrid;

    // Blocks of 1024 threads ( the maximum number of threads per block )
    dimBlock.x = 16;
    dimBlock.y = 16;
    dimBlock.z = 1;

    // One block per harmonic, thus we can sort input powers in Shared memory
    dimGrid.x = ceil(noR/(float)dimBlock.x);
    dimGrid.y = ceil(noZ/(float)dimBlock.y);

    // Call the kernel to normalise and spread the input data
    ffdot_ker<float><<<dimGrid, dimBlock, 0, 0>>>(cuPowers, cuInp, noHarms, halfwidth, minR, maxZ, rSZ, zSZ, noR, noZ, noInp, noPow, rOff, norm);

    CUDA_SAFE_CALL(cudaGetLastError(), "Calling the ffdot_ker kernel.");
  }

  CUDA_SAFE_CALL(cudaMemcpy(powers, cuPowers, pStride*noZ, cudaMemcpyDeviceToHost), "Copying optimisation results back from the device.");

  cudaDeviceSynchronize();          // TMP
  int TMPP = 0;

  FOLD // Write CVS
  {
    char tName[1024];
    sprintf(tName,"/home/chris/accel/lrg_2_GPU.csv");
    FILE *f2 = fopen(tName, "w");

    int indx = 0;
    int indy = 0;

    fprintf(f2,"%i",noHarms);

    for (int indx = 0; indx < noR ; indx++ )
    {
      double r = minR + indx/(double)(noR-1) * (rSZ) ;
      fprintf(f2,"\t%.6f",r);
    }
    fprintf(f2,"\n");

    for (int indy = 0; indy < noZ; indy++ )
    {
      double z = maxZ - indy/(double)(noZ-1) * (zSZ) ;

      fprintf(f2,"%.6f",z);

      for (int indx = 0; indx < noR ; indx++ )
      {
        float yy2 = powers[indy*noPow+indx];
        fprintf(f2,"\t%.6f",yy2);
      }
      fprintf(f2,"\n");
    }
    fclose(f2);

    FOLD // Make image
    {
      printf("Making lrg_GPU.png    \t... ");
      fflush(stdout);
      char cmd[1024];
      sprintf(cmd,"python ~/bin/bin/plt_ffd.py %s", tName);
      system(cmd);
      printf("Done\n");
    }

    int tmp = 0;
  }

  CUDA_SAFE_CALL(cudaFree(cuPowers),    "Failed free device memory for optimisation powers.");
  CUDA_SAFE_CALL(cudaFree(cuInp),       "Failed free device memory for optimisation inputs.");

  return noPow;
}

template<typename T>
void ffdotPln( cuFDotPlain* pln, fftInfo* fft )
{
  double maxZ       = (pln->centZ + pln->zSize/2.0);
  double minZ       = (pln->centZ - pln->zSize/2.0);
  double maxR       = (pln->centR + pln->rSize/2.0);
  double minR       = (pln->centR - pln->rSize/2.0);

  pln->halfWidth    = z_resp_halfwidth(MAX(fabs(maxZ*pln->noHarms), fabs(minZ*pln->noHarms)) + 4, HIGHACC);
  double rSpread    = ceil(maxR*pln->noHarms  + pln->halfWidth) - floor(minR*pln->noHarms - pln->halfWidth);
  pln->inpStride    = getStrie(rSpread, sizeof(cufftComplex), pln->alignment);
  pln->outStride    = getStrie(pln->noR,  sizeof(float), pln->alignment);
  if ( pln->inpStride*pln->noHarms*sizeof(cufftComplex) > pln->inpSz )
  {
    fprintf(stderr, "ERROR: In function %s, cuFDotPlain not created with large enough input buffer.", __FUNCTION__);
    exit(EXIT_FAILURE);
  }

  int16   rOff;
  float16 norm;
  int     off;
  int datStart,  datEnd, noDat;

  for( int h = 0; h < 16; h++)
  {
    rOff.val[h] = 0;
  }

  for( int h = 0; h < pln->noHarms; h++)
  {
    datStart        = floor( minR*(h+1) - pln->halfWidth );
    datEnd          = ceil(  maxR*(h+1) + pln->halfWidth );
    noDat           = datEnd - datStart;
    rOff.val[h]     = datStart;

    double factor   = sqrt(pln->norm[h]);
    norm.val[h]     = factor;

    for ( int i = 0; i < pln->inpStride; i++ ) // Normalise input  .
    {
      off = rOff.val[h] - fft->rlo + i;

      if ( off >= 0 && off < fft->nor /* && i < noDat */ )
      {
        pln->h_inp[h*pln->inpStride + i].r = fft->fft[off].r / factor ;
        pln->h_inp[h*pln->inpStride + i].i = fft->fft[off].i / factor ;
      }
      else
      {
        pln->h_inp[h*pln->inpStride + i].r = 0;
        pln->h_inp[h*pln->inpStride + i].i = 0;
      }
    }
  }

  CUDA_SAFE_CALL(cudaMemcpy(pln->d_inp, pln->h_inp, pln->inpStride*pln->noHarms*sizeof(fcomplexcu), cudaMemcpyHostToDevice), "Copying optimisation input to the device");

  FOLD // Call kernel  .
  {
    dim3 dimBlock, dimGrid;

    // Blocks of 1024 threads ( the maximum number of threads per block )
    dimBlock.x = 16;
    dimBlock.y = 16;
    dimBlock.z = 1;

    // One block per harmonic, thus we can sort input powers in Shared memory
    dimGrid.x = ceil(pln->noR/(float)dimBlock.x);
    dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

    // Call the kernel to normalise and spread the input data
    ffdot_ker<T><<<dimGrid, dimBlock, 0, 0>>>((float*)pln->d_out, pln->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->inpStride, pln->outStride, rOff, norm);

    //ffdot_ker<double><<<dimGrid, dimBlock, 0, 0>>>((float*)pln->d_powers, pln->d_inp, pln->noHarms, pln->halfWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, pln->inpStride, pln->powerStride, rOff, norm);

    CUDA_SAFE_CALL(cudaGetLastError(), "Calling the ffdot_ker kernel.");
  }

  CUDA_SAFE_CALL(cudaMemcpy(pln->h_out, pln->d_out, pln->outStride*pln->noZ*sizeof(float), cudaMemcpyDeviceToHost), "Copying optimisation results back from the device.");

  int TMPP = 0;
}

__global__ void rz_interp_ker(double r, double z, fcomplexcu* fft, int loR, int noBins, int halfwidth, double normFactor)
{
  float total_power   = 0;

  fcomplexcu ans      = rz_interp_cu<float>(fft, loR, noBins, r, z, halfwidth);
  //fcomplexcu ans      = rz_interp_cu<double>(fft, loR, noBins, r, z, halfwidth);
  total_power         += POWERR(ans.r, ans.i)/normFactor;

  //printf("rz_interp_ker r: %.4f  z: %.4f  Power: %.4f  ( %.4f, %.4f )\n", r, z, POWERR(ans.r, ans.i), ans.r, ans.i);
}

void rz_interp_cu(fcomplex* fft, int loR, int noBins, double centR, double centZ, int halfwidth)
{
  FOLD // TMP: CPU equivalent  .
  {
    double total_power = 0.;
    double powargr, powargi;
    fcomplex ans;

    rz_interp((fcomplex*)fft, noBins, centR, centZ, halfwidth, &ans);
  }

  float *cuPowers;
  fcomplexcu *cuInp;
  fcomplexcu *cpuInp;
  int     rOff, lodata;
  double factor;
  double log2 = log(2.0);

  //halfwidth       = z_resp_halfwidth(fabs(centZ), HIGHACC);
  int noInp       = 2*halfwidth;
  lodata          = floor( centR ) - halfwidth ;
  rOff            = lodata - loR ;

  FOLD // Clamp size  .
  {
    if ( lodata < 0 )
    {
      noInp         += lodata;
      rOff          -= lodata;
    }

    if ( rOff + noInp >= noBins )
    {
      fprintf(stderr, "WARNING: attempting to do a f-∂f interpolation beyond the end of the FFT.\n");
      noInp = noBins - rOff;
    }
  }

  FOLD // GPU Memory operations  .
  {
    CUDA_SAFE_CALL(cudaMalloc((void** )&cuInp, noInp * sizeof(cufftComplex) ),   "Failed to allocate device memory for kernel stack.");
    CUDA_SAFE_CALL(cudaMemcpy(cuInp, &fft[rOff], noInp * sizeof(cufftComplex), cudaMemcpyHostToDevice), "Copying convolution kernels between devices.");
  }

  FOLD // Calculate normalisation factor  .
  {
    float*  normPow = (float*) malloc(noInp*sizeof(float));

    for ( int i = 0; i < noInp; i++ )
    {
      normPow[i] = POWERR(fft[rOff+i].r, fft[rOff+i].i ) ;
    }

    float medianv   = median(normPow, noInp);
    factor          = sqrt(medianv/log2);

    free(normPow);
  }

  FOLD // Call kernel  .
  {
    dim3 dimBlock, dimGrid;

    // Blocks of 1024 threads ( the maximum number of threads per block )
    dimBlock.x = 1;
    dimBlock.y = 1;
    dimBlock.z = 1;

    // One block per harmonic, thus we can sort input powers in Shared memory
    dimGrid.x = 1;
    dimGrid.y = 1;

    // Call the kernel to normalise and spread the input data
    rz_interp_ker<<<dimGrid, dimBlock, 0, 0>>>(centR, centZ, cuInp, rOff, noInp, halfwidth, factor);

    cudaDeviceSynchronize();          // TMP
    int TMPP = 0;
  }
}

template<typename T>
void opt_cand(accelcand* cand, fftInfo* fft, cuFDotPlain* pln, int noP, double scale, int plt = -1, int nn = 0 )
{
  FOLD // Large points  .
  {
    pln->centR          = cand->r;
    pln->centZ          = cand->z;
    pln->noZ            = noP*2 + 1;
    pln->noR            = noP*2 + 1;
    pln->rSize          = scale;
    pln->zSize          = scale*4.0;

//          gettimeofday(&start, NULL);       // TMP

    ffdotPln<T>(pln, fft);

//          gettimeofday(&end, NULL);         // TMP
//          timev1 = ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec)); // TMP
//          printf("%.5f\t",timev1);          // TMP
  }

  if (plt >= 0 ) // Write CVS  .
  {
    cudaDeviceSynchronize();          // TMP

    char tName[1024];
    sprintf(tName,"/home/chris/accel/lrg_GPU_%05i_%02i_h%02i.csv", nn, plt, cand->numharm );
    FILE *f2 = fopen(tName, "w");

    fprintf(f2,"%i",pln->noHarms);

    for (int indx = 0; indx < pln->noR ; indx++ )
    {
      double r = pln->centR - pln->rSize/2.0 + indx/(double)(pln->noR-1) * (pln->rSize) ;
      fprintf(f2,"\t%.6f",r);
    }
    fprintf(f2,"\n");

    for (int indy = 0; indy < pln->noZ; indy++ )
    {
      double z = pln->centZ + pln->zSize/2.0 - indy/(double)(pln->noZ-1) * (pln->zSize) ;

      fprintf(f2,"%.6f",z);

      for (int indx = 0; indx < pln->noR ; indx++ )
      {
        float yy2 = ((float*)pln->h_out)[indy*pln->outStride+indx];
        fprintf(f2,"\t%.6f",yy2);
      }
      fprintf(f2,"\n");
    }
    fclose(f2);

    FOLD // Make image
    {
      //printf("Making lrg_GPU.png    \t... ");
      //fflush(stdout);
      char cmd[1024];
      sprintf(cmd,"python ~/bin/bin/plt_ffd.py %s", tName);
      system(cmd);
      //printf("Done\n");
    }

    int tmp = 0;
  }

  FOLD // Get new max  .
  {
    float max = ((float*)pln->h_out)[0];

    for (int indy = 0; indy < pln->noZ; indy++ )
    {
      for (int indx = 0; indx < pln->noR ; indx++ )
      {
        float yy2 = ((float*)pln->h_out)[indy*pln->outStride+indx];
        if ( yy2 > max )
        {
          max = yy2;
          cand->r   = pln->centR - pln->rSize/2.0 + indx/(double)(pln->noR-1) * (pln->rSize) ;
          cand->z   = pln->centZ + pln->zSize/2.0 - indy/(double)(pln->noZ-1) * (pln->zSize) ;
        }
      }
    }
  }

}

void optimize_accelcand_cu(accelcand* cand, accelobs* obs, int nn, cuFDotPlain* pln)
{
  int ii;
  int *r_offset;
  fcomplex **data;
  double r, z;
  int noP;
  float scale;

  struct timeval start, end, start1, end1;
  double timev1, timev2, timev3;

  //printf("\n%4i  optimize_accelcand  harm %2i   r %20.4f   z %7.3f  pow: %8.3f  sigma: %8.3f\n", nn, cand->numharm, cand->r, cand->z, cand->power, cand->sigma );

  cand->pows   = gen_dvect(cand->numharm);
  cand->hirs   = gen_dvect(cand->numharm);
  cand->hizs   = gen_dvect(cand->numharm);
  cand->derivs = (rderivs *)  malloc(sizeof(rderivs) * cand->numharm);

  int numdata   = obs->numbins;

  pln->centR    = cand->r ;
  pln->centZ    = cand->z ;
  pln->noHarms  = cand->numharm ;

  fftInfo fft;
  fft.fft       = obs->fft;
  fft.rlo       = obs->lobin;
  fft.nor       = obs->numbins;
  fft.rhi       = obs->lobin + obs->numbins;

  //printf("%4i  optimize_accelcand  harm %2i   r %20.4f   z %7.3f  pow: %8.3f \n", nn, pln->noHarms, pln->centR, pln->centZ, 0 );

  for ( int i=1; i <= cand->numharm; i++ )
  {
    pln->norm[i-1]  = get_scaleFactorZ(fft.fft, numdata, (fft.rlo+pln->centR)*i-fft.rlo, pln->centZ*i, 0.0);
  }

  if ( obs->use_harmonic_polishing )
  {
    if ( obs->mmap_file || obs->dat_input )
    {
      FOLD // GPU grid
      {
        int rep = 0;
        int noP = 30;
        float sz;

        if ( cand->numharm == 1 )
          sz = 16;
        if ( cand->numharm == 2 )
          sz = 14;
        if ( cand->numharm == 4 )
          sz = 12;
        if ( cand->numharm == 8 )
          sz = 10;
        if ( cand->numharm == 16 )
          sz = 8;

        printf("\n%03i  r: %15.6f   z: %12.6f \n", nn, cand->r, cand->z);

        noP = 40 ;
        opt_cand<float>(cand, &fft, pln, noP, sz,  rep++, nn );
        printf("     r: %15.6f   z: %12.6f  %8.6f\n", nn, cand->r, cand->z, sz);
        sz = (sz/(float)noP)*2;

        noP = 20 ;
        opt_cand<float>(cand, &fft, pln, noP, sz,  rep++, nn );
        printf("     r: %15.6f   z: %12.6f  %8.6f\n", nn, cand->r, cand->z, sz);
        sz = (sz/(float)noP)*2;

        opt_cand<float>(cand, &fft, pln, noP, sz,  rep++, nn );
        printf("     r: %15.6f   z: %12.6f  %8.6f\n", nn, cand->r, cand->z, sz);
        sz = (sz/(float)noP)*2;

        //opt_cand<float>(cand, &fft, pln, noP, sz,  rep++, nn );
        //printf("     r: %15.6f   z: %12.6f  %8.6f\n", nn, cand->r, cand->z, sz);

        int tmp = 0;

        for ( int i = 1; i <= cand->numharm; i++ )
        {
          double rH = (obs->lobin+r)*i-obs->lobin;
          double rZ = z*i;
          double x[2];

          float locpow = get_scaleFactorZ(obs->fft, obs->numbins, rH, rZ, 0.0);
          x[0] = rH;
          x[1] = rZ/4.0;
          //maxdata = data[i-1];
          //cand->pows[i-1] = -power_call_rz(x[0]);
          get_derivs3d(obs->fft, obs->numbins, rH, rZ, 0.0, locpow, &cand->derivs[i-1] );
        }

        for( ii=0; ii < cand->numharm; ii++ )
        {
          cand->hirs[ii]=(r+obs->lobin)*(ii+1);
          cand->hizs[ii]=z*(ii+1);
        }

        FOLD // Update fundamental values to the optemised ones
        {
          cand->power = 0;
          for( ii=0; ii<cand->numharm; ii++ )
          {
            cand->power += cand->derivs[ii].pow/cand->derivs[ii].locpow;;
          }
          cand->r     = r+obs->lobin;
          cand->z     = z;
        }

        //cand->sigma = candidate_sigma(cand->power, cand->numharm, obs->numindep[twon_to_index(cand->numharm)]);
      }
    }
  }
}
