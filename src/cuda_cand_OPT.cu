#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"


extern "C"
{
#define __float128 long double
#include "accel.h"
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

    T sn  = 3.18016297876567817986e11 + (-4.42979518059697779103e10 + (2.54890880573376359104e9  + (-6.29741486205862506537e7  + ( 7.08840045257738576863e5  -   2.99181919401019853726e3   * t)*t)*t)*t)*t;
    T sd  = 6.07366389490084639049e11 + ( 2.24411795645340920940e10 + (4.19320245898111231129e8  + ( 5.17343888770096400730e6  + ( 4.55847810806532581675e4  + ( 2.81376268889994315696e2   + t)*t)*t)*t)*t)*t ;
    T cn  = 9.99999999999999998822e-1 + (-2.05525900955013891793e-1 + (1.88843319396703850064e-2 + (-6.45191435683965050962e-4 + ( 9.50428062829859605134e-6 -   4.98843114573573548651e-8  * t)*t)*t)*t)*t;
    T cd  = 1.00000000000000000118e0  + ( 4.12142090722199792936e-2 + (8.68029542941784300606e-4 + ( 1.22262789024179030997e-5 +  (1.25001862479598821474e-7 + ( 9.15439215774657478799e-10 + 3.99982968972495980367e-12*t)*t)*t)*t)*t)*t ;

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
    t     = PI * x2;
    u     = 1.0 / (t * t);
    t     = 1.0 / t;

    T fn  = 3.76329711269987889006e-20+(1.34283276233062758925e-16+(1.72010743268161828879e-13+(1.02304514164907233465e-10+(3.05568983790257605827e-8 +(4.63613749287867322088e-6+(3.45017939782574027900e-4+(1.15220955073585758835e-2+(1.43407919780758885261e-1+ 4.21543555043677546506e-1*u)*u)*u)*u)*u)*u)*u)*u)*u;
    T fd  = 1.25443237090011264384e-20+(4.52001434074129701496e-17+(5.88754533621578410010e-14+(3.60140029589371370404e-11+(1.12699224763999035261e-8 +(1.84627567348930545870e-6+(1.55934409164153020873e-4+(6.44051526508858611005e-3+(1.16888925859191382142e-1+(7.51586398353378947175e-1+u)*u)*u)*u)*u)*u)*u)*u)*u)*u ;
    T gn  = 1.86958710162783235106e-22+(8.36354435630677421531e-19+(1.37555460633261799868e-15+(1.08268041139020870318e-12+(4.45344415861750144738e-10+(9.82852443688422223854e-8+(1.15138826111884280931e-5+(6.84079380915393090172e-4+(1.87648584092575249293e-2+(1.97102833525523411709e-1+5.04442073643383265887e-1*u)*u)*u)*u)*u)*u)*u)*u)*u)*u ;
    T gd  = 1.86958710162783236342e-22+(8.39158816283118707363e-19+(1.38796531259578871258e-15+(1.10273215066240270757e-12+(4.60680728146520428211e-10+(1.04314589657571990585e-7+(1.27545075667729118702e-5+(8.14679107184306179049e-4+(2.53603741420338795122e-2+(3.37748989120019970451e-1+(1.47495759925128324529e0+u)*u)*u)*u)*u)*u)*u)*u)*u)*u)*u ;

    f     = 1.0 - u * fn / fd;
    g     =       t * gn / gd;

    t     = PIBYTWO * x2;
    sincos(t, &s, &c);
    t     = PI * x;
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

  startr        = 0 - (0.5 * z);
  startroffset  = (startr < 0) ? 1.0 + modf(startr, &tmprl) : modf(startr, &tmprl);

  if (rx == numkern / 2.0 && startroffset < 1E-3 && absz < 1E-3)
  {
    T nr, ni;

    zz      = z * z;
    xx      = startroffset * startroffset;
    nr      = 1.0 - 0.16449340668482264365 * zz;
    ni      = -0.5235987755982988731 * z;
    nr      += startroffset * 1.6449340668482264365 * z;
    ni      += startroffset * (PI - 0.5167712780049970029 * zz);
    nr      += xx * (-6.579736267392905746 + 0.9277056288952613070 * zz);
    ni      += xx * (3.1006276680299820175 * z);

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

    signz   = (z < 0.0) ? -1 : 1;
    zd      = signz * (T) SQRT2 / sqrt(absz);
    zd      = signz * sqrt(2.0 / absz);
    cons    = zd / 2.0;                             // 1 / sqrt(2*r')

    startr  += numkern / (T) (2 * numbetween);
    delta   = -1.0 / numbetween;
    r       = startr + rx * delta;

    pibyz   = PI / z;
    yy      = r * zd;
    zz      = yy + z * zd;
    xx      = pibyz * r * r;
    c       = cos(xx);
    s       = sin(xx);
    fresnl<T>(yy, &fressy, &frescy);
    fresnl<T>(zz, &fressz, &frescz);
    tmprl   = signz * (frescz - frescy);
    tmpim   = fressy - fressz;

    *rr     =  (tmprl * c - tmpim * s) * cons;
    *ri     = -(tmprl * s + tmpim * c) * cons;
  }
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
  T s, c, pibyz, cons;
  fcomplexcu inp;
  fcomplexcu ans;

  ans.r = 0.0;
  ans.i = 0.0;

  // Split 'r' into integer and fractional parts

  fracfreq          = modf(r, &dintfreq);
  intfreq           = (int) dintfreq;
  numkern           = 2 * kern_half_width;
  lodata            = intfreq - kern_half_width;

  // Set up values dependent on Z alone
  absz              = fabs(z);
  startr            = fracfreq - (0.5 * z);
  printf("r %.10f  fracfreq %.10f   z: %.10f  \n",r, fracfreq, z);
  signz             = (z < 0.0) ? -1 : 1;
  zd                = signz * SQRT2 / sqrt(absz);
  cons              = zd / 2.0;
  pibyz             = PI / z;
  startr            += kern_half_width;

  FOLD // Clamp numkern and q_r
  {
    if ( lodata < 0 )
    {
      numkern += lodata;
      startr  += lodata;
      lodata  = 0;
    }

    lodata -= loR;

    printf("lodata + numkern: %i  noR: %i \n", lodata + numkern, noBins );
    if ( lodata + numkern >= noBins )
    {
      numkern = noBins - lodata;
    }
    printf("numkern: %i\n", numkern );
  }

  // Loop over positions and do correlations
  for ( ii = 0, q_r = startr; ii < numkern; q_r--, ii++ )
  {
    //  Read the input value
    inp             = data[lodata+ii];

    // Calculate response values
    Yr              = q_r * zd;
    Zr              = Yr + z * zd;
    xx              = pibyz * q_r * q_r;
    c               = cos_t(xx);
    s               = sin_t(xx);
    fresnl<T>(Yr, &fressy, &frescy);
    fresnl<T>(Zr, &fressz, &frescz);

    T Ster          = fressz - fressy;
    T Cter          = frescy - frescz;
    T tR            = cons * (c*Ster + signz*s*Cter);
    T tI            = cons * (s*Ster - signz*c*Cter);

    // Do the multiplications
    ans.r           += tR * inp.r - tI*inp.i;
    ans.i           += tR * inp.i + tI*inp.r;

    //printf("%03i  q_r %10.5f \n", ii, q_r );
    printf("%03i %05i Data %12.2f %12.2f  Response: %13.10f %13.10f   %12.2f \n", ii, loR+lodata+ii, inp.r, inp.i, tR, tI, POWERR(ans.r, ans.i) );
  }

  return ans;
}

__global__ void ffdot_ker(float* powers, fcomplexcu* fft, int noBin, int noHarms, int halfwidth, float firstR, float firstZ, float rSZ, float zSZ, int noR, int noZ, int iStride, int oStride, int loR)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  float r = firstR + ix/(float)noR * rSZ ;
  float z = firstZ - iy/(float)noZ * zSZ ;

  int firstRBin = floor(firstR-halfwidth);

  float total_power = 0;

  for(int i = 1; i <= noHarms; i++)
  {
    fcomplexcu ans  = rz_interp_cu<float>(fft, loR, noBin, r, z, halfwidth);
    total_power     += POWERR(ans.r, ans.i);
  }
}

void ffdot(float* powers, fcomplex* fft, int noBins, int noHarms, double centR, double centZ, double rSZ, double zSZ, int noR, int noZ)
{
  double log2 = log(2.0);

  printf("%f \n", centR );
  printf("%f \n", centZ );
  printf("%f \n", rSZ );

  float maxZ = centZ + zSZ/2.0;
  float minZ = centZ - zSZ/2.0;
  float minR = centR - rSZ/2.0;
  float maxR = centR + rSZ/2.0;

  int halfwidth = z_resp_halfwidth(MAX(fabs(maxZ*1.2), fabs(minZ*1.2)), HIGHACC);

  size_t rStride, pStride;
  float *cuPowers;
  fcomplexcu *cuInp;
  fcomplexcu *cpuInp;

  //CUDA_SAFE_CALL(cudaMalloc((void **)cuPowers, noPow*sizeof(float)), "Failed to allocate device memory for kernel stack.");
  CUDA_SAFE_CALL(cudaMallocPitch(&cuPowers,  &pStride, noR * sizeof(float),                        noZ),       "Failed to allocate device memory for kernel stack.");
  CUDA_SAFE_CALL(cudaMallocPitch(&cuInp,     &rStride, (noR+2*halfwidth) * sizeof(cufftComplex)*noHarms,  noHarms),   "Failed to allocate device memory for kernel stack.");

  int noInp = rStride/sizeof(cufftComplex);
  int noPow = pStride/sizeof(float);

  float*  normPow = (float*) malloc(noInp*sizeof(float));
  int*    rOff    = (int*)   malloc(noHarms*sizeof(int));

  cpuInp = (fcomplexcu*) malloc(rStride*noHarms);

  for( int h = 0; h < noHarms; h++)
  {
    rOff[h] = floor( minR*(h+1) ) - halfwidth ;

    for ( int i = 0; i < noInp; i++)
    {
      normPow[i] = POWERR(fft[rOff[h]+i].r, fft[rOff[h]+i].i ) ;
    }

    float medianv = median(normPow, noInp);
    double factor = sqrt(medianv/log2);

    for ( int i = 0; i < noInp; i++)
    {
      cpuInp[h*noInp + h].r = fft[rOff[h]+i].r / factor ;
      cpuInp[h*noInp + h].i = fft[rOff[h]+i].i / factor ;
    }
  }

  CUDA_SAFE_CALL(cudaMemcpy(cuInp, cpuInp, rStride*noHarms, cudaMemcpyHostToDevice), "Copying convolution kernels between devices.");

  dim3 dimBlock, dimGrid;

  // Blocks of 1024 threads ( the maximum number of threads per block )
  dimBlock.x = 16;
  dimBlock.y = 16;
  dimBlock.z = 1;

  // One block per harmonic, thus we can sort input powers in Shared memory
  dimGrid.x = ceil(noR/(float)dimBlock.x);
  dimGrid.y = ceil(noZ/(float)dimBlock.y);

  // Call the kernel to normalise and spread the input data
  ffdot_ker<<<dimGrid, dimBlock, 0, 0>>>(cuPowers, cuInp, noBins, noHarms, halfwidth, minR, minZ, rSZ, zSZ, noR, noZ, noInp,noPow, rOff[0]);
}


__global__ void rz_interp_ker(double r, double z, fcomplexcu* fft, int loR, int noBins, int halfwidth, double normFactor)
{
  float total_power   = 0;

  //fcomplexcu ans      = rz_interp_cu<float>(fft, loR, noBins, r, z, halfwidth);
  fcomplexcu ans      = rz_interp_cu<double>(fft, loR, noBins, r, z, halfwidth);
  total_power         += POWERR(ans.r, ans.i)/normFactor;

  printf("rz_interp_ker r: %.4f  z: %.4f  Power: %.4f  ( %.4f, %.4f )\n", r, z, POWERR(ans.r, ans.i), ans.r, ans.i);
}

void rz_interp_cu(fcomplex* fft, int loR, int noBins, double centR, double centZ, int halfwidth)
{
  FOLD // TMP: CPU equivalent
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
  double log2 = log(2.0);

  //int halfwidth = z_resp_halfwidth(fabs(centZ), HIGHACC);
  int noInp       = 2*halfwidth;
  lodata          = floor( centR ) - halfwidth ;
  rOff            = lodata - loR ;

  FOLD // Clamp size
  {
    if ( lodata < 0 )
    {
      noInp         += lodata;
      rOff          -= lodata;
    }

    printf("rOff + noInp: %i  noR: %i \n", rOff + noInp, noBins);
    if ( rOff + noInp >= noBins )
    {
      fprintf(stderr, "WARNING: attempting to do a f-∂f interpolation beyond the end of the FFT.\n");
      noInp = noBins - rOff;
    }
  }

  // Allocate memory
  CUDA_SAFE_CALL(cudaMalloc((void** )&cuInp, noInp * sizeof(cufftComplex) ),   "Failed to allocate device memory for kernel stack.");
  float*  normPow = (float*) malloc(noInp*sizeof(float));

  for ( int i = 0; i < noInp; i++ )
  {
    normPow[i] = POWERR(fft[rOff+i].r, fft[rOff+i].i ) ;
  }

  float medianv   = median(normPow, noInp);
  double factor   = sqrt(medianv/log2);

  CUDA_SAFE_CALL(cudaMemcpy(cuInp, &fft[rOff], noInp * sizeof(cufftComplex), cudaMemcpyHostToDevice), "Copying convolution kernels between devices.");

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
