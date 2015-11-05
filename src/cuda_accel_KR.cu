#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"

/** Fresnel integral  .
 *
 * DESCRIPTION:
 *
 * Evaluates the Fresnel integrals
 *
 *           x
 *           -
 *          | |
 * C(x) =   |   cos(pi/2 t**2) dt,
 *        | |
 *         -
 *          0
 *
 *           x
 *           -
 *          | |
 * S(x) =   |   sin(pi/2 t**2) dt.
 *        | |
 *         -
 *          0
 *
 *
 * The integrals are evaluated by a power series for x < 1.
 * For x >= 1 auxiliary functions f(x) and g(x) are employed
 * such that
 *
 * C(x) = 0.5 + f(x) sin( pi/2 x**2 ) - g(x) cos( pi/2 x**2 )
 * S(x) = 0.5 - f(x) cos( pi/2 x**2 ) - g(x) sin( pi/2 x**2 )
 *
 *
 *
 * ACCURACY:
 *
 *  Relative error.
 *
 * Arithmetic  function   domain     # trials      peak         rms
 *   IEEE       S(x)      0, 10       10000       2.0e-15     3.2e-16
 *   IEEE       C(x)      0, 10       10000       1.8e-15     3.3e-16
 *   DEC        S(x)      0, 10        6000       2.2e-16     3.9e-17
 *   DEC        C(x)      0, 10        5000       2.3e-16     3.9e-17
 *
 *   This function is adapted from:
 *   Cephes Math Library Release 2.8:  June, 2000
 *   Copyright 1984, 1987, 1989, 2000 by Stephen L. Moshier
 *
 * @param xxa Value to evaluate the Fresnel integral at
 * @param ss  The result S(xxa)
 * @param cc  The result C(xxa)
 */
template<typename T>
__device__ void fresnl(T xxa, T* ss, T* cc)
{
  T f, g, c, s, t, u;
  T x, x2;

  x       = fabs(xxa);
  x2      = x * x;

  if      ( x2 < 2.5625   )     // Small so use a polynomial approximation  .
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

__device__ int z_resp_halfwidth(double z)
{
  int m;

  z = fabs(z);

  m = (long) (z * (0.00089 * z + 0.3131) + NUMFINTBINS);
  m = (m < NUMFINTBINS) ? NUMFINTBINS : m;

  // Prevent the equation from blowing up in large z cases

  if (z > 100 && m > 0.6 * z)
    m = 0.6 * z;

  return m;
}

__device__ int z_resp_halfwidth_high(double z)
{
  int m;

  z = fabs(z);

  m = (long) (z * (0.002057 * z + 0.0377) + NUMFINTBINS * 3);
  m += ((NUMLOCPOWAVG >> 1) + DELTAAVGBINS);

  /* Prevent the equation from blowing up in large z cases */

  if (z > 100 && m > 1.2 * z)
     m = 1.2 * z;

  return m;
}


/** Generate a complex response function for Fourier interpolation  .
 *
 * This is a CUDA "copy" of gen_r_response in responce.c
 *
 * @param kx            The x index of the value in the kernel
 * @param roffset       Is the offset in Fourier bins for the full response (i.e. At this point, the response would equal 1.0)
 * @param numbetween    Is the number of points to interpolate between each standard FFT bin. (i.e. 'numbetween' = 2 = interbins, this is the standard)
 * @param numkern       Is the number of complex points that the kernel will contain.
 * @param rr            A pointer to the real part of the complex response for kx
 * @param ri            A pointer to the imaginary part of the complex response for kx
 */
template<typename T>
__device__ inline void gen_r_response(int kx, T roffset, T numbetween, int numkern, float* rr, float* ri)
{
  int ii;
  T tmp, sinc, s, c, alpha, beta, delta, startr, r;

  startr = PI * (numkern / (T) (2 * numbetween));
  delta = -PI / numbetween;
  tmp = sin(0.5 * delta);
  alpha = -2.0 * tmp * tmp;
  beta = sin(delta);

  c = cos(startr);
  s = sin(startr);

  r = startr + kx * delta;

  if (kx == numkern / 2)
  {
    // Correct for divide by zero when the roffset is close to zero
    *rr = 1 - 6.579736267392905746 * (tmp = roffset * roffset);
    *ri = roffset * (PI - 10.335425560099940058 * tmp);
  }
  else
  {
    // TODO: Fix this!
    // I am recursing in the kernel o0
    // I just haven't had the time to calculate this per thread calculation
    // But it is only called once, so not to critical if it is inefficient
    for (ii = 0, r = startr; ii <= kx; ii++, r += delta)
    {
      if (r == 0.0)
        sinc = 1.0;
      else
        sinc = s / r;

      *rr = c * sinc;
      *ri = s * sinc;
      c = alpha * (tmp = c) - beta * s + c;
      s = alpha * s + beta * tmp + s;
    }
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

/** Create the convolution kernel for one f-∂f plane  .
 *
 *  This is "copied" from gen_z_response in respocen.c
 *
 * @param response
 * @param maxZ
 * @param fftlen
 */
__global__ void init_kernels(fcomplexcu* response, int maxZ, int fftlen, int half_width,  float zSteps, float rSteps)
{
  int cx, cy;                                   /// The x and y index of this thread in the array
  int rx = -1;                                  /// The x index of the value in the kernel

  // Calculate the 2D index of this thread
  cx = blockDim.x * blockIdx.x + threadIdx.x;   /// use BLOCKSIZE rather (its constant)
  cy = blockDim.y * blockIdx.y + threadIdx.y;   /// use BLOCKSIZE rather (its constant)

  float z = -maxZ + cy * 1.0/zSteps;            /// The Fourier Frequency derivative

  if ( z < -maxZ || z > maxZ || cx >= fftlen || cx < 0 )
  {
    // Out of bounds
    return;
  }

  // Calculate the response x position from the plane x position
  if ( half_width <= 0 )
  {
    half_width    = z_resp_halfwidth((double) z);
  }
  else
  {
    int hw2       = MAX(0.6*z, 16*1);
    half_width    = MIN( half_width, hw2 ) ;
    half_width    = z_resp_halfwidth((double) z);
  }

  int hw          = rSteps * half_width;
  int numkern     = 2 * hw;                     /// The number of complex points that the kernel row will contain

  // Calculate the kernel index for this thread (centred on zero and wrapped)
  if (cx < hw)
    rx = cx + hw;
  else if (cx >= fftlen - hw)
    rx = cx - (fftlen - hw);

  fcomplexcu resp;                              /// the complex response
  resp.r = 0.0;
  resp.i = 0.0;

  FOLD // Calculate the response value  .
  {
    if (rx != -1)
    {
      float absz = fabs(z);

      if (absz < 1E-4 )    // If z~=0 use the normal Fourier interpolation kernel  .
      {
        gen_r_response<double> (rx, 0.0,     rSteps, numkern, &resp.r, &resp.i);
      }
      else                 // Calculate the complex response value for Fourier f-dot interpolation  .
      {
        gen_z_response<double> (rx, z, absz, rSteps, numkern, &resp.r, &resp.i);
      }
    }
  }

  response[cy * fftlen + cx]  = resp;
}

/** Create the convolution kernel for an entire stack  .
 *
 * @param response
 * @param stack
 * @param fftlen
 * @param stride
 * @param maxh
 * @param maxZa
 * @param startR
 * @param zmax
 */
__global__ void init_kernels_stack(float2* response, const int fftlen, const int stride, const int maxh, const int noPlanes, iList startR, fList zmax)
{
  int cx, cy;                       /// The x and y index of this thread in the array
  int rx = -1;                      /// The x index of the value in the kernel
  int plane = -1;                   /// The f-∂f plane the thread deals with
  float maxZ;                       /// The Z-Max of the plane this thread deals with

  // Calculate the 2D index of this thread
  cx = blockDim.x * blockIdx.x + threadIdx.x;// use BLOCKSIZE rather (its constant)
  cy = blockDim.y * blockIdx.y + threadIdx.y;// use BLOCKSIZE rather (its constant)

  if ( cy >= maxh || cx >= fftlen || cx < 0 )
  {
    // Out of bounds
    return;
  }

  // Calculate which plane in the stack we are working with
  for ( int i = 0; i < noPlanes; i++ )
  {
    if ( cy >= startR.val[i] && cy < startR.val[i + 1] )
    {
      plane = i;
      break;
    }
  }
  maxZ = zmax.val[plane];
  float z = -maxZ + (cy-startR.val[plane]) * ACCEL_DZ; /// The Fourier Frequency derivative

  // Calculate the response x position from the plane x position
  int kern_half_width = z_resp_halfwidth((double) z);
  int hw = ACCEL_NUMBETWEEN * kern_half_width;
  int numkern = 2 * hw;             /// The number of complex points that the kernel row will contain
  if (cx < hw)
    rx = cx + hw;
  else if (cx >= fftlen - hw)
    rx = cx - (fftlen - hw);

  FOLD // Calculate the response value
  {
    float rr = 0;
    float ri = 0;

    if (rx != -1)
    {
      float absz = fabs(z);
      if (absz < 1E-4 )     // If z~=0 use the normal Fourier interpolation kernel
      {
        gen_r_response<double>(rx, 0.0, ACCEL_NUMBETWEEN, numkern, &rr, &ri);
      }
      else                  // Calculate the complex response value for Fourier f-dot interpolation.
      {
        gen_z_response<double>(rx, z, absz, ACCEL_NUMBETWEEN, numkern, &rr, &ri);
      }
    }

    float2 tmp2 = { rr, ri };
    response[cy * fftlen + cx] = tmp2;
    //response[(cy*fftlen+cx)*2]    = rr;
    //response[(cy*fftlen+cx)*2+1]  = ri;
  }
}

/** Create one GPU kernel. One kernel the size of the largest plane  .
 *
 * @param kernel
 * @return
 */
int createStackKernel(cuFfdotStack* cStack)
{
  dim3 dimBlock, dimGrid;

  dimBlock.x          = BLOCKSIZE;  // in my experience 16 is almost always best (half warp)
  dimBlock.y          = BLOCKSIZE;  // in my experience 16 is almost always best (half warp)

  // Set up grid
  dimGrid.x = ceil(  cStack->width     / ( float ) dimBlock.x );
  dimGrid.y = ceil ( cStack->kerHeigth / ( float ) dimBlock.y );

  int halfWidth;

  if ( cStack->flag & FLAG_KER_ACC )
  {
    // Use one halfwidth for the entire kernel
    halfWidth = cStack->harmInf->halfWidth;
  }
  else
  {
    // Columns closer to a z value of 0 will have smaller halfwidths (dynamically calculated)
    halfWidth = 0;
  }


  FOLD // call the CUDA kernels  .
  {
    // Call kernel
    init_kernels<<<dimGrid, dimBlock>>>(cStack->d_kerData, cStack->harmInf->zmax, cStack->width,  halfWidth, ACCEL_RDZ, ACCEL_RDR);

    // Run message
    CUDA_SAFE_CALL(cudaGetLastError(), "At kernel launch");
  }

  return 0;
}

/** Create GPU kernels. One for each plane of the stack  .
 *
 * @param kernel
 * @return
 */
int createStackKernels(cuFfdotStack* cStack)
{
  dim3 dimBlock, dimGrid;

  dimBlock.x          = BLOCKSIZE;  // in my experience 16 is almost always best (half warp)
  dimBlock.y          = BLOCKSIZE;  // in my experience 16 is almost always best (half warp)

  // Set up grid
  dimGrid.x = ceil(  cStack->width  / ( float ) dimBlock.x );
  dimGrid.y = ceil ( cStack->kerHeigth / ( float ) dimBlock.y );

  iList startR;
  fList zmax;
  for (int j = 0; j< cStack->noInStack; j++)
  {
    startR.val[j]     = cStack->startZ[j];
    zmax.val[j]       = cStack->harmInf[j].zmax;
  }
  startR.val[cStack->noInStack] = cStack->height;

  FOLD // call the CUDA kernels
  {
    // Set up grid
    dimGrid.x = ceil(  cStack->width  / ( float ) dimBlock.x );
    dimGrid.y = ceil ( cStack->height / ( float ) dimBlock.y );

    // Call kernel
    init_kernels_stack<<<dimGrid, dimBlock>>>((float2*) cStack->d_kerData, cStack->width, cStack->strideCmplx, cStack->height, cStack->noInStack , startR, zmax);

    // Run message
    CUDA_SAFE_CALL(cudaGetLastError(), "At kernel launch");
  }

  return 0;
}

//int createOptKernel(cuOptKer* cKer)
//{
//  dim3 dimBlock, dimGrid;
//
//    dimBlock.x          = BLOCKSIZE;  // in my experience 16 is almost always best (half warp)
//    dimBlock.y          = BLOCKSIZE;  // in my experience 16 is almost always best (half warp)
//
//    // Set up grid
//    dimGrid.x = ceil(  cKer->width     / ( float ) dimBlock.x );
//    dimGrid.y = ceil ( cKer->height / ( float ) dimBlock.y );
//
//    FOLD // call the CUDA kernels  .
//    {
//      // Call kernel
//      init_kernels<<<dimGrid, dimBlock>>>(cKer->d_kerData, cKer->maxZ, cKer->width, cKer->noZ, cKer->noR );
//
//      // Run message
//      CUDA_SAFE_CALL(cudaGetLastError(), "At kernel launch");
//    }
//
//    return 0;
//}
