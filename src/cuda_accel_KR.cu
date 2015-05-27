#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"


// S(x) for small x
static __device__ double sn[6] =
{ -2.99181919401019853726E3, 7.08840045257738576863E5, -6.29741486205862506537E7, 2.54890880573376359104E9, -4.42979518059697779103E10, 3.18016297876567817986E11,};

static __device__ double sd[6] =
{ 2.81376268889994315696E2, 4.55847810806532581675E4, 5.17343888770096400730E6, 4.19320245898111231129E8, 2.24411795645340920940E10, 6.07366389490084639049E11,};

// C(x) for small x
static __device__ double cn[6] =
{ -4.98843114573573548651E-8, 9.50428062829859605134E-6, -6.45191435683965050962E-4, 1.88843319396703850064E-2, -2.05525900955013891793E-1, 9.99999999999999998822E-1,};

static __device__ double cd[7] =
{ 3.99982968972495980367E-12, 9.15439215774657478799E-10, 1.25001862479598821474E-7, 1.22262789024179030997E-5, 8.68029542941784300606E-4, 4.12142090722199792936E-2, 1.00000000000000000118E0,};

// Auxiliary function f(x)
static __device__ double fn[10] =
{ 4.21543555043677546506E-1, 1.43407919780758885261E-1, 1.15220955073585758835E-2, 3.45017939782574027900E-4, 4.63613749287867322088E-6, 3.05568983790257605827E-8, 1.02304514164907233465E-10, 1.72010743268161828879E-13, 1.34283276233062758925E-16, 3.76329711269987889006E-20,};

static __device__ double fd[10] =
{ 7.51586398353378947175E-1, 1.16888925859191382142E-1, 6.44051526508858611005E-3, 1.55934409164153020873E-4, 1.84627567348930545870E-6, 1.12699224763999035261E-8, 3.60140029589371370404E-11, 5.88754533621578410010E-14, 4.52001434074129701496E-17, 1.25443237090011264384E-20,};

// Auxiliary function g(x)
static __device__ double gn[11] =
{ 5.04442073643383265887E-1, 1.97102833525523411709E-1, 1.87648584092575249293E-2, 6.84079380915393090172E-4, 1.15138826111884280931E-5, 9.82852443688422223854E-8, 4.45344415861750144738E-10, 1.08268041139020870318E-12, 1.37555460633261799868E-15, 8.36354435630677421531E-19, 1.86958710162783235106E-22,};

static __device__ double gd[11] =
{ 1.47495759925128324529E0, 3.37748989120019970451E-1, 2.53603741420338795122E-2, 8.14679107184306179049E-4, 1.27545075667729118702E-5, 1.04314589657571990585E-7, 4.60680728146520428211E-10, 1.10273215066240270757E-12, 1.38796531259578871258E-15, 8.39158816283118707363E-19, 1.86958710162783236342E-22,};


__device__ double polevl(double x, double *p, int N)
{
  double ans;
  int i;
  //double *p;
  //p = coef;

  ans = *p++;
  i = N;

  do
    ans = ans * x + *p++;
  while (--i);

  return (ans);
}

__device__ double p1evl(double x, double *p, int N)
{
  double ans;
  //double *p;
  int i;

  //p = coef;
  ans = x + *p++;
  i = N - 1;

  do
    ans = ans * x + *p++;
  while (--i);

  return (ans);
}

__device__ int fresnl(double xxa, double *ssa, double *cca)
{
  double f, g, cc, ss, c, s, t, u;
  double x, x2;

  x = fabs(xxa);
  x2 = x * x;
  if (x2 < 2.5625) {
    t = x2 * x2;
    ss = x * x2 * polevl(t, sn, 5) / p1evl(t, sd, 6);
    cc = x * polevl(t, cn, 5) / polevl(t, cd, 6);
    goto done;
  }
  if (x > 36974.0) {
    cc = 0.5;
    ss = 0.5;
    goto done;
  }
  /* Auxiliary functions for large argument  */
  x2 = x * x;
  t = PI * x2;
  u = 1.0 / (t * t);
  t = 1.0 / t;
  f = 1.0 - u * polevl(u, fn, 9) / p1evl(u, fd, 10);
  g = t * polevl(u, gn, 10) / p1evl(u, gd, 11);
  t = PIBYTWO * x2;
  c = cos(t);
  s = sin(t);
  t = PI * x;
  cc = 0.5 + (f * s - g * c) / t;
  ss = 0.5 - (f * c + g * s) / t;

  done:

  if (xxa < 0.0) {
    cc = -cc;
    ss = -ss;
  }
  *cca = cc;
  *ssa = ss;
  return (0);
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
__device__ inline void gen_r_response(int kx, double roffset, float numbetween, int numkern, float* rr, float* ri)
{
  int ii;
  double tmp, sinc, s, c, alpha, beta, delta, startr, r;

  startr = PI * (numkern / (double) (2 * numbetween));
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
 * @param kx            The x index of the value in the kernel
 * @param z             The Fourier Frequency derivative (# of bins the signal smears over during the observation)
 * @param absz          Is the absolute value of z
 * @param roffset       Is the offset in Fourier bins for the full response (i.e. At this point, the response would equal 1.0)
 * @param numbetween    Is the number of points to interpolate between each standard FFT bin. (i.e. 'numbetween' = 2 = interbins, this is the standard)
 * @param numkern       Is the number of complex points that the kernel will contain.
 * @param rr            A pointer to the real part of the complex response for kx
 * @param ri            A pointer to the imaginary part of the complex response for kx
 */
__device__ inline void gen_z_response(int rx, float z,  double absz, float numbetween, int numkern, float* rr, float* ri)
{
  int signz;
  double zd, r, xx, yy, zz, startr, startroffset;
  double fressy, frescy, fressz, frescz, tmprl, tmpim;
  double s, c, pibyz, cons, delta;

  startr        = 0 - (0.5 * z);
  startroffset  = (startr < 0) ? 1.0 + modf(startr, &tmprl) : modf(startr, &tmprl);

  if (rx == numkern / 2.0 && startroffset < 1E-3 && absz < 1E-3)
  {
    double nr, ni;

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
    signz   = (z < 0.0) ? -1 : 1;
    zd      = signz * (double) SQRT2 / sqrt(absz);
    zd      = signz * sqrt(2.0 / absz);
    cons    = zd / 2.0;
    pibyz   = PI / z;
    startr  += numkern / (double) (2 * numbetween);
    delta   = -1.0 / numbetween;

    r       = startr + rx * delta;

    yy      = r * zd;
    zz      = yy + z * zd;
    xx      = pibyz * r * r;
    c       = cos(xx);
    s       = sin(xx);
    fresnl(yy, &fressy, &frescy);
    fresnl(zz, &fressz, &frescz);
    tmprl = signz * (frescz - frescy);
    tmpim = fressy - fressz;

    *rr     = (tmprl * c - tmpim * s) * cons;
    *ri     = -(tmprl * s + tmpim * c) * cons;
  }
}

/** Create the convolution kernel for one f-∂f plain  .
 *
 *  This is "copied" from gen_z_response in respocen.c
 *
 * @param response
 * @param maxZ
 * @param fftlen
 * @param frac
 */
__global__ void init_kernels(float* response, int maxZ, int fftlen, float frac)
{
  int cx, cy;                       /// The x and y index of this thread in the array
  int rx = -1;                      /// The x index of the value in the kernel

  // Calculate the 2D index of this thread
  cx = blockDim.x * blockIdx.x + threadIdx.x;// use BLOCKSIZE rather (its constant)
  cy = blockDim.y * blockIdx.y + threadIdx.y;// use BLOCKSIZE rather (its constant)

  float z = -maxZ + cy * ACCEL_DZ;   /// The Fourier Frequency derivative

  if ( z < -maxZ || z > maxZ || cx >= fftlen || cx < 0 )
  {
    // Out of bounds
    return;
  }

  // Calculate the response x position from the plain x position
  int kern_half_width = z_resp_halfwidth((double) z);
  int hw = ACCEL_NUMBETWEEN * kern_half_width;
  int numkern = 2 * hw;           /// The number of complex points that the kernel row will contain
  if (cx < hw)
    rx = cx + hw;
  else if (cx >= fftlen - hw)
    rx = cx - (fftlen - hw);

  FOLD // Calculate the response value
  {
    float rr = 0;               /// The real part of the complex response
    float ri = 0;               /// The imaginary part of the complex response

    if (rx != -1)
    {
      float absz = fabs(z);

      if (absz < 1E-4 )    // If z~=0 use the normal Fourier interpolation kernel
      {
        gen_r_response (rx, 0.0, ACCEL_NUMBETWEEN, numkern, &rr, &ri);
      }
      else                 // Calculate the complex response value for Fourier f-dot interpolation.
      {
        gen_z_response (rx, z, absz, ACCEL_NUMBETWEEN, numkern, &rr, &ri);
      }
    }

    response[(cy * fftlen + cx) * 2    ]  = rr;
    response[(cy * fftlen + cx) * 2 + 1]  = ri;
  }
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
__global__ void init_kernels_stack(float2* response, const int fftlen, const int stride, const int maxh, const int noPlains, iList startR, fList zmax)
{
  int cx, cy;                       /// The x and y index of this thread in the array
  int rx = -1;                      /// The x index of the value in the kernel
  int plain = -1;                   /// The f-∂f plain the thread deals with
  float maxZ;                       /// The Z-Max of the plain this thread deals with

  // Calculate the 2D index of this thread
  cx = blockDim.x * blockIdx.x + threadIdx.x;// use BLOCKSIZE rather (its constant)
  cy = blockDim.y * blockIdx.y + threadIdx.y;// use BLOCKSIZE rather (its constant)

  if ( cy >= maxh || cx >= fftlen || cx < 0 )
  {
    // Out of bounds
    return;
  }

  // Calculate which plain in the stack we are working with
  for ( int i = 0; i < noPlains; i++ )
  {
    if ( cy >= startR.val[i] && cy < startR.val[i + 1] )
    {
      plain = i;
      break;
    }
  }
  maxZ = zmax.val[plain];
  float z = -maxZ + (cy-startR.val[plain]) * ACCEL_DZ; /// The Fourier Frequency derivative

  // Calculate the response x position from the plain x position
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
      double absz = fabs(z);
      if (absz < 1E-4 )     // If z~=0 use the normal Fourier interpolation kernel
      {
        gen_r_response(rx, 0.0, ACCEL_NUMBETWEEN, numkern, &rr, &ri);
      }
      else                  // Calculate the complex response value for Fourier f-dot interpolation.
      {
        gen_z_response (rx, z, absz, ACCEL_NUMBETWEEN, numkern, &rr, &ri);
      }
    }

    float2 tmp2 = { rr, ri };
    response[cy * fftlen + cx] = tmp2;
    //response[(cy*fftlen+cx)*2]    = rr;
    //response[(cy*fftlen+cx)*2+1]  = ri;
  }
}

/** Create one GPU kernel. One kernel the size of the largest plain  .
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
  dimGrid.x = ceil(  cStack->width  / ( float ) dimBlock.x );
  dimGrid.y = ceil ( cStack->kerHeigth / ( float ) dimBlock.y );

  FOLD // call the CUDA kernels  .
  {
    // Call kernel
    init_kernels<<<dimGrid, dimBlock>>>((float*)cStack->d_kerData, cStack->harmInf->zmax, cStack->width, cStack->harmInf->harmFrac);

    // Run message
    CUDA_SAFE_CALL(cudaGetLastError(), "Error at kernel launch");
  }

  return 0;
}

/** Create GPU kernels. One for each plain of the stack  .
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
    CUDA_SAFE_CALL(cudaGetLastError(), "Error at kernel launch");
  }

  return 0;
}
