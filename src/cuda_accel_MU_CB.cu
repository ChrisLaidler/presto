/** @file cuda_accel_MU_CB.cu
 *  @brief The implementation of the stack multiplication kernel v3
 *
 *  @author Chris Laidler
 *  @bug No known bugs.
 *
 *
 *  Change Log
 *
 *  [2.02.17] [2020-02-03]
 *    A major refactor for release
 *
 */

#include "cuda_accel_MU.h"

//====================================== Constant variables  ===============================================\\

#if	CUDART_VERSION >= 6050

#ifdef	WITH_MUL_PRE_CALLBACK	// Pre Callbacks (mult)  .

/** CUFFT callback kernel to simply return constant value  .
 */
__device__ cufftComplex CB_RetConst( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr)
{
  // Constant value
  cufftComplex out;
  out.x = 1;
  out.y = 0;

  return out;
}

/** CUFFT callback kernel to simply read data  .
 */
__device__ cufftComplex CB_RetValue( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr)
{
  // Read input
  cufftComplex out = ((cufftComplex*)dataIn)[offset];

  return out;
}

/** CUFFT callback kernel to simply read data  .
 */
__device__ cufftComplex CB_readInp( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr)
{
  //stackInfo *inf  = (stackInfo*)callerInfo;

  //int fIdx        = inf->famIdx;
  //int stackStrd   = STRIDE_HARM[fIdx];

  //int row         = offset / stackStrd;
  //int col         = offset % 16384;

  // Read input
  //cufftComplex inp = ((cufftComplex*)inf->d_iData)[col];
  //cufftComplex inp = ((cufftComplex*)callerInfo)[col];
  cufftComplex ipd = ((cufftComplex*)callerInfo)[-offset];

  // Read kernel
  cufftComplex ker = ((cufftComplex*)dataIn)[offset];

  cufftComplex out;
  out.x = (ipd.x * ker.x + ipd.y * ker.y);
  out.y = (ipd.y * ker.x - ipd.x * ker.y);

  return out;
}

/** CUFFT callback kernel to simply read data  .
 *
 * @param plan
 * @param cStack
 */
__device__ cufftComplex CB_readInp0( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr)
{
  //if ( offset == 0 )
  //  printf("STK_STRD[0] %i \n", STK_STRD[0]);
  int strd        = STK_STRD[0];
  int col         = offset % ( strd*32 );
  //int row         = offset / strd;
  //char  inpIdx    = 0; STK_INP[0][row];

  // Read input
  //cufftComplex ipd = ((cufftComplex*)callerInfo)[strd*inpIdx + col ];
  cufftComplex ipd = ((cufftComplex*)callerInfo)[ col ];

  // Read kernel
  cufftComplex ker = ((cufftComplex*)dataIn)[offset];

  cufftComplex out;
  //  out.x = (ipd.x * ker.x + ipd.y * ker.y);
  //  out.y = (ipd.y * ker.x - ipd.x * ker.y);
  out.x = ipd.x;
  out.y = ker.y;

  //  cufftComplex out;
  //  out.x = 1;
  //  out.y = 0;
  return out;
}

/** CUFFT callback kernel to simply read data  .
 *
 * @param plan
 * @param cStack
 */
__device__ cufftComplex CB_readInp1( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr)
{
  //  int strd        = STK_STRD[1];
  //  int col         = offset % strd;
  //  int row         = offset / strd;
  //  char  inpIdx    = STK_INP[1][row];
  //
  //  // Read input
  //  cufftComplex ipd = ((cufftComplex*)callerInfo)[strd*inpIdx + col ];
  //
  //  // Read kernel
  //  cufftComplex ker = ((cufftComplex*)dataIn)[offset];
  //
  //  cufftComplex out;
  //  out.x = (ipd.x * ker.x + ipd.y * ker.y);
  //  out.y = (ipd.y * ker.x - ipd.x * ker.y);

  cufftComplex out;
  out.x = 1;
  out.y = 0;
  return out;
}

/** CUFFT callback kernel to simply read data  .
 *
 * @param plan
 * @param cStack
 */
__device__ cufftComplex CB_readInp2( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr)
{
  //  int strd        = STK_STRD[2];
  //  int col         = offset % strd;
  //  int row         = offset / strd;
  //  char  inpIdx    = STK_INP[2][row];
  //
  //  // Read input
  //  cufftComplex ipd = ((cufftComplex*)callerInfo)[strd*inpIdx + col ];
  //
  //  // Read kernel
  //  cufftComplex ker = ((cufftComplex*)dataIn)[offset];
  //
  //  cufftComplex out;
  //  out.x = (ipd.x * ker.x + ipd.y * ker.y);
  //  out.y = (ipd.y * ker.x - ipd.x * ker.y);

  cufftComplex out;
  out.x = 1;
  out.y = 0;
  return out;
}

/** CUFFT callback kernel to simply read data  .
 *
 * @param plan
 * @param cStack
 */
__device__ cufftComplex CB_readInp3( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr)
{
  //  int strd        = STK_STRD[3];
  //  int col         = offset % strd;
  //  int row         = offset / strd;
  //  char  inpIdx    = STK_INP[3][row];
  //
  //  // Read input
  //  cufftComplex ipd = ((cufftComplex*)callerInfo)[strd*inpIdx + col ];
  //
  //  // Read kernel
  //  cufftComplex ker = ((cufftComplex*)dataIn)[offset];
  //
  //  cufftComplex out;
  //  out.x = (ipd.x * ker.x + ipd.y * ker.y);
  //  out.y = (ipd.y * ker.x - ipd.x * ker.y);

  cufftComplex out;
  out.x = 1;
  out.y = 0;
  return out;
}

/** CUFFT callback kernel to multiply the complex f-∂f before the FFT  .
 */
__device__ cufftComplex CB_MultiplyInput( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr)
{
  stackInfo *inf  = (stackInfo*)callerInfo;

  int fIdx        = inf->famIdx;
  int noSegments  = inf->noSegments;
  int noPlanes    = inf->noPlanes;
  int stackStrd   = STRIDE_HARM[fIdx];
  int width       = WIDTH_HARM[fIdx];

  int strd        = stackStrd * noSegments ;              /// Stride taking into account segments)
  int gRow        = offset / strd;                        /// Row (ignoring segments)
  int col         = offset % stackStrd;                   /// 2D column
  int top         = 0;                                    /// The top of the plane
  int pHeight     = 0;
  int pln         = 0;

  for ( int i = 0; i < noPlanes; i++ )
  {
    top += HEIGHT_HARM[fIdx+i];

    if ( gRow >= top )
    {
      pln         = i+1;
      pHeight     = top;
    }
  }

  int row         = offset / stackStrd - pHeight*noSegments;
  int pIdx        = fIdx + pln;
  int plnHeight   = HEIGHT_HARM[pIdx];
  int segment;

  if ( inf->flags & FLAG_ITLV_ROW )
  {
    segment	= row % noSegments;
    row		= row / noSegments;
  }
  else
  {
    segment	= row / plnHeight;
    row		= row % plnHeight;
  }

  cufftComplex ker = ((cufftComplex*)(KERNEL_HARM[pIdx]))[row*stackStrd + col];           //
  cufftComplex inp = ((cufftComplex*)inf->d_iData)[(pln*noSegments+segment)*stackStrd + col];   //

  // Do the multiplication
  cufftComplex out;

#if CORRECT_MULT
  out.x = (inp.x * ker.x - inp.y * ker.y) / (float)width;
  out.y = (inp.y * ker.x + inp.x * ker.y) / (float)width;
#else
  out.x = (inp.x * ker.x + inp.y * ker.y) / (float)width;
  out.y = (inp.y * ker.x - inp.x * ker.y) / (float)width;
#endif

  return out;
}

__device__ cufftCallbackLoadC  d_loadConst          = CB_RetConst;
__device__ cufftCallbackLoadC  d_loadRead           = CB_RetValue;
__device__ cufftCallbackLoadC  d_loadInp            = CB_readInp;
__device__ cufftCallbackLoadC  d_loadInp0           = CB_readInp0;
__device__ cufftCallbackLoadC  d_loadInp1           = CB_readInp1;
__device__ cufftCallbackLoadC  d_loadInp2           = CB_readInp2;
__device__ cufftCallbackLoadC  d_loadInp3           = CB_readInp3;
__device__ cufftCallbackLoadC  d_loadCallbackPtr    = CB_MultiplyInput;

#endif	// WITH_MUL_PRE_CALLBACK

#endif	//CUDART_VERSION

/** Load the CUFFT callbacks  .
 */
acc_err copy_CuFFT_load_CBs(cuCgPlan* plan, cuFfdotStack* cStack)
{
  acc_err ret = ACC_ERR_NONE;

  if ( plan->flags & FLAG_MUL_CB )
  {
    PROF // Profiling  .
    {
      NV_RANGE_PUSH("CUFFT load callbacks");
    }

    infoMSG(5,5,"Set inp CB function.");

#ifdef 	WITH_MUL_PRE_CALLBACK	// Pre Callbacks (mult)  .
    CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &cStack->h_ldCallbackPtr, d_loadCallbackPtr,  sizeof(cufftCallbackLoadC), 0, cudaMemcpyDeviceToHost, cStack->initStream ),   "Getting constant memory address.");
#else	// WITH_MUL_PRE_CALLBACK
    EXIT_DIRECTIVE("WITH_MUL_PRE_CALLBACK");
#endif	// WITH_MUL_PRE_CALLBACK

    PROF // Profiling  .
    {
      NV_RANGE_POP("CUFFT load callbacks");
    }
  }

  return ret;
}


/** Set CUFFT store FFT callback  .
 *
 */
acc_err set_CuFFT_load_CBs(cuCgPlan* plan, cuFfdotStack* cStack)
{
  acc_err ret = ACC_ERR_NONE;

  if ( plan->flags & FLAG_MUL_CB )
  {
    infoMSG(5,5,"Set CB input\n");

#ifdef 	WITH_MUL_PRE_CALLBACK
    if ( plan->flags & FLAG_DOUBLE )
    {
#if	CUDART_VERSION >= 9000
      // This is a hack!
      CUFFT_SAFE_CALL(cufftXtClearCallback(cStack->plnPlan, CUFFT_CB_LD_COMPLEX_DOUBLE), "Error clearing CUFFT load callback.");
#endif // CUDART_VERSION
      CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&cStack->h_ldCallbackPtr, CUFFT_CB_LD_COMPLEX_DOUBLE, (void**)&cStack->d_sInf ),"Error assigning CUFFT load callback.");
    }
    else
    {
#if	CUDART_VERSION >= 9000
      // This is a hack!
      CUFFT_SAFE_CALL(cufftXtClearCallback(cStack->plnPlan, CUFFT_CB_LD_COMPLEX), "Error clearing CUFFT load callback.");
#endif	//CUDART_VERSION
      CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&cStack->h_ldCallbackPtr, CUFFT_CB_LD_COMPLEX, (void**)&cStack->d_sInf ),"Error assigning CUFFT load callback.");
    }
#else	// WITH_MUL_PRE_CALLBACK
    fprintf(stderr, "ERROR: Not compiled with multiplication through CUFFT callbacks enabled. \n");
    exit(EXIT_FAILURE);
#endif	// WITH_MUL_PRE_CALLBACK
  }

  return ret;
}

