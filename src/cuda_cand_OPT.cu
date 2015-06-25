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
__device__ fcomplexcu rz_interp_cu(fcomplexcu* data, float r, float z, int kern_half_width, int loR)
{
  fcomplexcu *dataptr;
  fcomplexcu response;

   int ii, numkern, nsum, intfreq, lodata, hidata, loresp, hiresp;
   float fracfreq, dintfreq, tmpd, tmpr;

   fcomplexcu ans;
   float respR;

   ans.r = 0.0;
   ans.i = 0.0;

   /* Check 'r' and return 0.0 + 0.0i if out of bounds.        */
   /* Should this return an error and exit instead?            */

   /*
   if (r > numdata - 1.0 || r < 0.0)
   {
      return ans;
   }
   */

   /* Split 'r' into integer and fractional parts */

   fracfreq   = modf(r, &dintfreq);
   intfreq    = (int) dintfreq;

   /* Return immediately if 'r' is close to an           */
   /* integer frequency and z is very close to zero      */

   if (fabs(z) < 1E-4)
   {
      if (fracfreq < 1E-5)
      {
         ans.r = data[intfreq].r;
         ans.i = data[intfreq].i;
         return ans;
      }
      if ((1.0 - fracfreq) < 1E-5)
      {
         ans.r = data[intfreq + 1].r;
         ans.i = data[intfreq + 1].i;
         return ans;
      }
   }

   /* Generate the response function */

   numkern    = 2 * kern_half_width;
   //response   = gen_z_response(fracfreq, 1, z, numkern);

   {
      int signz;
      float absz, zd, tmp, xx, yy, zz;
      float fresSY, fresCY, fresSZ, fresCZ, C, S;
      float s, c, pibyz, cons, delta;
      float startr;
      float startroffset;

      //printf("gen_z_response( roffset %13.6f,  numbetween %02i,  z %13.8f,  numkern %5i ) \n", roffset, numbetween, z, numkern );

      /* Check that the arguments are OK */

      /*
      if (roffset < 0.0 || roffset >= 1.0)
      {
         printf("\n  roffset = %f (out of bounds) in gen_z_response().\n\n", roffset);
         exit(-1);
      }
      if (numbetween < 1 || numbetween >= 20000)
      {
         printf("\n  numbetween = %d (out of bounds) in gen_z_response().\n\n",
                numbetween);
         exit(-1);
      }
      if (numkern < numbetween)
      {
         printf("\n  numkern = %d (out of bounds) in gen_z_response().\n\n", numkern);
         exit(-1);
      }
      if ((numkern % (2 * numbetween)) != 0)
      {
         printf("\n  numkern %% (2 * numbetween) != 0 in gen_z_response().\n\n");
         exit(-1);
      }
      */

      /* If z~=0 use the normal Fourier interpolation kernel */

      absz = fabs(z);

      /*// TODO fix this
      if (absz < 1E-4)
      {
         response = gen_r_response(roffset, numbetween, numkern);
         return response;
      }
      */

      /* Begin the calculations */

      startr        = fracfreq - ( z / 2.0 );
      startroffset  = (startr < 0) ? 1.0 + modf(startr, &C) : modf(startr, &C);
      signz         = (z < 0.0) ? -1 : 1;
      zd            = signz * SQRT2 / sqrt(absz);
      cons          = zd / 2.0;
      pibyz         = PI / z;
      startr        += numkern / 2.0;
      delta         = -1.0 / 1.0;

      //float

      dataptr = &data[(intfreq-kern_half_width)-loR];

      for (int ii = 0, respR = startr; ii < numkern; ii++, respR += delta) // loop over kernel
      {
         yy   = respR * zd;
         zz   = yy + z * zd;
         xx   = pibyz * respR * respR;
         sincosf(xx,&s,&c);
         fresnl(yy, &fresSY, &fresCY);
         fresnl(zz, &fresSZ, &fresCZ);
         C    = signz * ( fresCZ - fresCY );
         S    =         ( fresSY - fresSZ );

         response.r =  (C * c - S * s) * cons;
         response.i = -(C * s + S * c) * cons;

         ans.r += dataptr[ii].r * response.r + dataptr[ii].i * response.i ;
         ans.i += dataptr[ii].i * response.r - dataptr[ii].r * response.i ;
      }

      /* Correct for divide by zero when the roffset and z is close to zero */

      /*
      if (startroffset < 1E-3 && absz < 1E-3)
      {
         zz = z * z;
         xx = startroffset * startroffset;
         numkernby2 = numkern / 2;
         response[numkernby2].r = 1.0 - 0.16449340668482264365 * zz;
         response[numkernby2].i = -0.5235987755982988731 * z;
         response[numkernby2].r += startroffset * 1.6449340668482264365 * z;
         response[numkernby2].i += startroffset * (PI - 0.5167712780049970029 * zz);
         response[numkernby2].r += xx * (-6.579736267392905746 + 0.9277056288952613070 * zz);
         response[numkernby2].i += xx * (3.1006276680299820175 * z);
      }
      */
   }

   /* Determine the summation boundaries */

   /*
   lodata = intfreq - kern_half_width;
   if (lodata < 0)
   {
      loresp = abs(lodata);
      lodata = 0;
   }
   else
   {
      loresp = 0;
   }
   hidata = intfreq + kern_half_width - 1;
   if (hidata > numdata - 1)
   {
      hiresp = numkern - hidata + numdata - 1;
   }
   else
   {
      hiresp = numkern;
   }
   nsum = hiresp - loresp;

   // Set up our pointers

   dataptr = (float *) (data + lodata);
   respptr = (float *) (response + loresp);

   // Do the summation

   for (ii = 0; ii < nsum; ii++)
   {
      tmpd = *(dataptr++);
      tmpr = *(respptr++);
      ans.r += tmpd * tmpr + (*dataptr) * (*respptr);
      ans.i += (*dataptr) * tmpr - (*respptr) * tmpd;
      dataptr++;
      respptr++;
   }

   vect_free(response);
*/

   return ans;
}

__global__ void ffdot_ker(float* powers, fcomplexcu* fft, int noHarms, int halfwidth, float firstR, float firstZ, float rSZ, float zSZ, int noR, int noZ, int iStride, int oStride, int loR)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  float r = firstR + ix/(float)noR * rSZ ;
  float z = firstZ - iy/(float)noZ * zSZ ;

  int firstRBin = floor(firstR-halfwidth);

  float total_power = 0;

  for(int i = 1; i <= noHarms; i++)
  {
    fcomplexcu ans  = rz_interp_cu(fft, r, z, halfwidth, loR);
    total_power     += POWERR(ans.r, ans.i);
  }
}

void ffdot(float* powers, fcomplex* fft, int noHarms, double centR, double centZ, double rSZ, double zSZ, int noR, int noZ)
{
  double log2 = log(2.0);

  printf("%f \n", centR );
  printf("%f \n", centZ );
  printf("%f \n", rSZ );

  float maxZ = centZ + zSZ/2.0;
  float minZ = centZ - zSZ/2.0;
  float minR = centR - rSZ/2.0;
  float maxR = centR + rSZ/2.0;

  int halfwidth = z_resp_halfwidth(MAX(fabs(maxZ), fabs(minZ)), HIGHACC);

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
  ffdot_ker<<<dimGrid, dimBlock, 0, 0>>>(cuPowers, cuInp, noHarms, halfwidth, minR, minZ, rSZ, zSZ, noR, noZ, noInp,noPow, rOff[0]);


}
