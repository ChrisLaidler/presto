#include "cuda_accel_MU.h"

//====================================== Constant variables  ===============================================\\

#if	CUDA_VERSION >= 6050

#ifdef	WITH_MUL_PRE_CALLBACK	// Pre Callbacks (mult)  .
__device__ cufftCallbackLoadC  d_loadConst          = CB_RetConst;
__device__ cufftCallbackLoadC  d_loadRead           = CB_RetValue;
__device__ cufftCallbackLoadC  d_loadInp            = CB_readInp;
__device__ cufftCallbackLoadC  d_loadInp0           = CB_readInp0;
__device__ cufftCallbackLoadC  d_loadInp1           = CB_readInp1;
__device__ cufftCallbackLoadC  d_loadInp2           = CB_readInp2;
__device__ cufftCallbackLoadC  d_loadInp3           = CB_readInp3;
__device__ cufftCallbackLoadC  d_loadCallbackPtr    = CB_MultiplyInput;
#endif	// WITH_MUL_PRE_CALLBACK

#ifdef	WITH_POW_POST_CALLBACK	// Post Callbacks (power)  .
__device__ cufftCallbackStoreC d_storePow_f         = CB_PowerOut_f;
__device__ cufftCallbackStoreC d_inmemRow_f         = CB_InmemOutRow_f;
__device__ cufftCallbackStoreC d_inmemPln_f         = CB_InmemOutPln_f;
#if CUDA_VERSION >= 7050
__device__ cufftCallbackStoreC d_storePow_h         = CB_PowerOut_h;
__device__ cufftCallbackStoreC d_inmemRow_h         = CB_InmemOutRow_h;
__device__ cufftCallbackStoreC d_inmemPln_h         = CB_InmemOutPln_h;
#endif
#endif	// WITH_POW_POST_CALLBACK

#endif	// CUDA_VERSION >= 6050

//========================================== Functions  ====================================================\\


__device__ int calcInMemIdx_ROW( size_t offset )
{
  const int hw  = PSTART_STAGE[0];
  const int st  = STRIDE_STAGE[0];
  const int al  = ALEN ;
  int col       = ( offset % st ) - hw * ACCEL_NUMBETWEEN ;

  if ( col < 0 || col >= al )
    return (-1);

  const int ns  = NO_STEPS;

  int row       =   offset  / ( st * ns ) ;
  int step      = ( offset  % ( st * ns ) ) / st;

  size_t plnOff = row * PLN_STRIDE + step * al + col;

  return (plnOff);
}

__device__ int calcInMemIdx_PLN( size_t offset )
{
  const int hw  = PSTART_STAGE[0];
  const int st  = STRIDE_STAGE[0];
  const int al  = ALEN ;
  int col       = ( offset % st ) - hw * ACCEL_NUMBETWEEN ;

  if ( col < 0 || col >= al )
    return (-1);

  const int ht  = HEIGHT_STAGE[0];

  int row       = offset  /   st;
  int step      = row     /   ht;
  row           = row     %   ht;  // Plane interleaved!

  size_t plnOff = row * PLN_STRIDE + step * al + col;

  return (plnOff);
}


#if CUDA_VERSION >= 6050        // CUFFT callbacks only implemented in CUDA 6.5  .

#ifdef 	WITH_MUL_PRE_CALLBACK	// Pre Callbacks (mult)  .

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
 * @param batch
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
 * @param batch
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
 * @param batch
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
 * @param batch
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
  int noSteps     = inf->noSteps;
  int noPlanes    = inf->noPlanes;
  int stackStrd   = STRIDE_HARM[fIdx];
  int width       = WIDTH_HARM[fIdx];

  int strd        = stackStrd * noSteps ;                 /// Stride taking into account steps)
  int gRow        = offset / strd;                        /// Row (ignoring steps)
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

  int row         = offset / stackStrd - pHeight*noSteps;
  int pIdx        = fIdx + pln;
  int plnHeight   = HEIGHT_HARM[pIdx];
  int step;

  if ( inf->flags & FLAG_ITLV_ROW )
  {
    step  = row % noSteps;
    row   = row / noSteps;
  }
  else
  {
    step = row / plnHeight;
    row  = row % plnHeight;
  }

  cufftComplex ker = ((cufftComplex*)(KERNEL_HARM[pIdx]))[row*stackStrd + col];           //
  cufftComplex inp = ((cufftComplex*)inf->d_iData)[(pln*noSteps+step)*stackStrd + col];   //

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

#endif	// WITH_MUL_PRE_CALLBACK

#ifdef WITH_POW_POST_CALLBACK	// Post Callbacks (power)  .

/** CUFFT callback kernel to calculate and store float powers after the FFT  .
 */
__device__ void CB_PowerOut_f( void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  // Calculate power
  float power = element.x*element.x + element.y*element.y ;

  // Write result (offsets are the same)
  ((float*)dataOut)[offset] = power;
}

/** CUFFT callback kernel to calculate and store float powers in the in-memory plane, after the FFT  .
 *  CallerInfo is passed as the address of the first element of the first step in the inmem plane
 *
 */
__device__ void CB_InmemOutRow_f( void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  int plnOff = calcInMemIdx_ROW(offset);

  if ( plnOff == -1 )
  {
    // This element is in the contaminated ends
    return;
  }

  // Calculate power
  float power = element.x*element.x + element.y*element.y ;

  // Write result (offsets are the same)
  ((float*)dataOut)[ plnOff ] = power;
}

/** CUFFT callback kernel to calculate and store float powers in the in-memory plane, after the FFT  .
 *  CallerInfo is passed as the address of the first element of the first step in the inmem plane
 *
 */
__device__ void CB_InmemOutPln_f( void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  int plnOff = calcInMemIdx_PLN(offset);

  if ( plnOff == -1 )
  {
    // This element is in the contaminated ends
    return;
  }

  // Calculate power
  float power = element.x*element.x + element.y*element.y ;

  // Write result (offsets are the same)
  ((float*)dataOut)[ plnOff ] = power;
}

#if CUDA_VERSION >= 7050 // Half precision CUFFT power call back  .

/** CUFFT callback kernel to calculate and store half powers after the FFT  .
 */
__device__ void CB_PowerOut_h( void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  // Calculate power
  float power = element.x*element.x + element.y*element.y;

  // Write result (offsets are the same)
  ((half*)dataOut)[offset] = __float2half(power);
}

/** CUFFT callback kernel to calculate and store half powers in the in-memory plane, after the FFT  .
 *  CallerInfo is passed as the address of the first element of the first step in the inmem plane
 *  Assumes row interleaved data
 *
 */
__device__ void CB_InmemOutRow_h( void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  int plnOff = calcInMemIdx_ROW(offset);

  if ( plnOff == -1 )
  {
    // This element is in the contaminated ends
    return;
  }

  // Calculate power
  float power = element.x*element.x + element.y*element.y ;

  // Write result (offsets are the same)
  ((half*)dataOut)[plnOff] = __float2half(power);
}

/** CUFFT callback kernel to calculate and store half powers in the in-memory plane, after the FFT  .
 *  CallerInfo is passed as the address of the first element of the first step in the inmem plane
 *  Assumes plane interleaved data
 *
 */
__device__ void CB_InmemOutPln_h( void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr)
{
  int plnOff = calcInMemIdx_PLN(offset);

  if ( plnOff == -1 )
  {
    // This element is in the contaminated ends
    return;
  }

  // Calculate power
  float power = element.x*element.x + element.y*element.y ;

  // Write result (offsets are the same)
  ((half*)dataOut)[plnOff] = __float2half(power);
}

#endif  // CUDA_VERSION >= 7050

#endif	// WITH_POW_POST_CALLBACK

/** Load the CUFFT callbacks  .
 */
void copyCUFFT_LD_CB(cuFFdotBatch* batch)
{
  PROF // Profiling  .
  {
    NV_RANGE_PUSH("CUFFT callbacks");
  }

  if ( batch->flags & FLAG_MUL_CB )
  {
    infoMSG(5,5,"Set inp CB function.");

#ifdef 	WITH_MUL_PRE_CALLBACK	// Pre Callbacks (mult)  .
    //CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_ldCallbackPtr, d_loadConst,        sizeof(cufftCallbackLoadC)),   "Getting constant memory address.");
    //CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_ldCallbackPtr, d_loadRead,         sizeof(cufftCallbackLoadC)),   "Getting constant memory address.");
    //CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_ldCallbackPtr, d_loadInp,          sizeof(cufftCallbackLoadC)),   "Getting constant memory address.");

    //CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_ldCallbackPtr0, d_loadInp0,        sizeof(cufftCallbackLoadC)),   "Getting constant memory address.");
    //CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_ldCallbackPtr1, d_loadInp1,        sizeof(cufftCallbackLoadC)),   "Getting constant memory address.");
    //CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_ldCallbackPtr2, d_loadInp2,        sizeof(cufftCallbackLoadC)),   "Getting constant memory address.");
    //CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_ldCallbackPtr3, d_loadInp3,        sizeof(cufftCallbackLoadC)),   "Getting constant memory address.");

    //CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_ldCallbackPtr, d_loadCallbackPtr,  sizeof(cufftCallbackLoadC)),   "Getting constant memory address.");
    CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &batch->h_ldCallbackPtr, d_loadCallbackPtr,  sizeof(cufftCallbackLoadC), 0, cudaMemcpyDeviceToHost, batch->stacks->initStream ),   "Getting constant memory address.");
#else
    fprintf(stderr, "ERROR: Not compiled with multiplication through CUFFT callbacks enabled. \n");
    exit(EXIT_FAILURE);
#endif
  }

  if ( batch->flags & FLAG_CUFFT_CB_OUT )
  {
    infoMSG(5,5,"Set out CB function.");

    if ( batch->flags & FLAG_POW_HALF )
    {
#if CUDA_VERSION >= 7050
      if ( batch->flags & FLAG_CUFFT_CB_INMEM )
      {
	// Store powers to inmem plane
	if ( batch->flags & FLAG_ITLV_ROW )    // Row interleaved
	{
	  //CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_inmemRow_h, sizeof(cufftCallbackStoreC)),  "Getting constant memory address.");
	  CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &batch->h_stCallbackPtr, d_inmemRow_h, sizeof(cufftCallbackStoreC), 0, cudaMemcpyDeviceToHost, batch->stacks->initStream),  "Getting constant memory address.");
	}
#ifdef WITH_ITLV_PLN
	else                                  // Plane interleaved
	{
	  //CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_inmemPln_h, sizeof(cufftCallbackStoreC)),  "Getting constant memory address.");
	  CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &batch->h_stCallbackPtr, d_inmemPln_h, sizeof(cufftCallbackStoreC), 0, cudaMemcpyDeviceToHost, batch->stacks->initStream),  "Getting constant memory address.");
	}
#else
	else
	{
	  fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
	  exit(EXIT_FAILURE);
	}
#endif
      }
      else
      {
	// Calculate powers and write to powers half precision plane
	//CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_storePow_h, sizeof(cufftCallbackStoreC)),     "Getting constant memory address.");
	CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &batch->h_stCallbackPtr, d_storePow_h, sizeof(cufftCallbackStoreC), 0, cudaMemcpyDeviceToHost, batch->stacks->initStream),     "Getting constant memory address.");
      }
#else
      fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
      exit(EXIT_FAILURE);
#endif
    }
    else
    {
      if ( batch->flags & FLAG_CUFFT_CB_INMEM )
      {
	// Store powers to inmem plane
	if ( batch->flags & FLAG_ITLV_ROW )    // Row interleaved
	{
	  //CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_inmemRow_f, sizeof(cufftCallbackStoreC)),  "Getting constant memory address.");
	  CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &batch->h_stCallbackPtr, d_inmemRow_f, sizeof(cufftCallbackStoreC), 0, cudaMemcpyDeviceToHost, batch->stacks->initStream),  "Getting constant memory address.");
	}
#ifdef WITH_ITLV_PLN
	else                                  // Plane interleaved
	{
	  //CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_inmemPln_f, sizeof(cufftCallbackStoreC)),  "Getting constant memory address.");
	  CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &batch->h_stCallbackPtr, d_inmemPln_f, sizeof(cufftCallbackStoreC), 0, cudaMemcpyDeviceToHost, batch->stacks->initStream),  "Getting constant memory address.");
	}
#else
	else
	{
	  fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
	  exit(EXIT_FAILURE);
	}
#endif
      }
      else
      {
	// Calculate powers and write to powers half single plane
	//CUDA_SAFE_CALL(cudaMemcpyFromSymbol( &batch->h_stCallbackPtr, d_storePow_f, sizeof(cufftCallbackStoreC)),     "Getting constant memory address.");
	CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync( &batch->h_stCallbackPtr, d_storePow_f, sizeof(cufftCallbackStoreC), 0, cudaMemcpyDeviceToHost, batch->stacks->initStream),     "Getting constant memory address.");
      }
    }
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP(); // CUFFT callbacks
  }
}

#endif  // CUDA_VERSION >= 6050

/** Get a pointer to the location of the first element of the output of the CUFFT  .
 *
 */
void* getCBwriteLocation(cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  void* dst = cStack->d_planePowr;

  if ( batch->flags &    FLAG_CUFFT_CB_INMEM )
  {
#if CUDA_VERSION >= 6050
    rVals* rVal   = &(*batch->rAraays)[batch->rActive][0][0];

    if ( batch->flags &  FLAG_POW_HALF )
    {
#if CUDA_VERSION >= 7050
      dst    = ((half*)batch->cuSrch->d_planeFull)    + rVal->step * batch->accelLen; // A pointer to the location of the first step in the in-mem plane
#else
      fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
      exit(EXIT_FAILURE);
#endif
    }
    else
    {
      dst    = ((float*)batch->cuSrch->d_planeFull)    + rVal->step * batch->accelLen; // A pointer to the location of the first step in the in-mem plane
    }
#else
    fprintf(stderr,"ERROR: CUFFT callbacks can only be used with CUDA 6.5 or later!\n");
    exit(EXIT_FAILURE);
#endif
  }

  return (dst);
}

/** Set CUFFT store FFT callback  .
 *
 */
void setCB(cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  if ( batch->flags & FLAG_CUFFT_CB_OUT )
  {
    infoMSG(5,5,"Set CB powers output\n");

#if CUDA_VERSION >= 6050

    void* dst;

    if ( batch->flags &    FLAG_CUFFT_CB_INMEM )
    {
      rVals* rVal   = &(*batch->rAraays)[batch->rActive][0][0];

      if ( batch->flags &  FLAG_POW_HALF )
      {
#if CUDA_VERSION >= 7050
	dst    = ((half*)batch->cuSrch->d_planeFull)    + rVal->step * batch->accelLen; // A pointer to the location of the first step in the in-mem plane
#else
	fprintf(stderr,"ERROR: Half precision can only be used with CUDA 7.5 or later!\n");
	exit(EXIT_FAILURE);
#endif
      }
      else
      {
	dst    = ((float*)batch->cuSrch->d_planeFull)    + rVal->step * batch->accelLen; // A pointer to the location of the first step in the in-mem plane
      }
    }
    else
    {
      dst = cStack->d_planePowr;

      // Testing passing values in the actual pointer
      //uint width  = cStack->strideCmplx ;
      //uint skip   = cStack->kerStart ;
      //uint pass   = (width << 16) | skip ;
      //dst = (void*)pass;
    }

    CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_stCallbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&dst ),"Error assigning CUFFT store callback.");
#else
    fprintf(stderr,"ERROR: CUFFT callbacks can only be used with CUDA 6.5 or later!\n");
    exit(EXIT_FAILURE);
#endif
  }

  if ( batch->flags & FLAG_MUL_CB )
  {
    infoMSG(5,5,"Set CB input\n");

#ifdef 	WITH_MUL_PRE_CALLBACK
    CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_ldCallbackPtr, CUFFT_CB_LD_COMPLEX, (void**)&cStack->d_sInf ),"Error assigning CUFFT load callback.");

//    Testing input FFT callback
//
//    FOLD // Set load FFT callback  .
//    {
//      //CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_ldCallbackPtr, CUFFT_CB_LD_COMPLEX, (void**)&cStack->d_sInf ),"Error assigning CUFFT load callback.");
//      //CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_ldCallbackPtr, CUFFT_CB_LD_COMPLEX, (void**)&cStack->d_iData ),"Error assigning CUFFT load callback.");
//      //size_t siz = cStack->height * cStack->width * batch->noSteps;
//      //fcomplexcu* ennd = cStack->d_planeMult + siz ;
//      //CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_ldCallbackPtr, CUFFT_CB_LD_COMPLEX, (void**)&ennd ),"Error assigning CUFFT load callback.");
//
//      if      ( cStack->stkIdx == 0 )
//	CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_ldCallbackPtr0, CUFFT_CB_LD_COMPLEX, (void**)&cStack->d_iData ),"Error assigning CUFFT load callback.");
//      else if ( cStack->stkIdx == 1 )
//	CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_ldCallbackPtr1, CUFFT_CB_LD_COMPLEX, (void**)&cStack->d_iData ),"Error assigning CUFFT load callback.");
//      else if ( cStack->stkIdx == 2 )
//	CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_ldCallbackPtr2, CUFFT_CB_LD_COMPLEX, (void**)&cStack->d_iData ),"Error assigning CUFFT load callback.");
//      else if ( cStack->stkIdx == 3 )
//	CUFFT_SAFE_CALL(cufftXtSetCallback(cStack->plnPlan, (void **)&batch->h_ldCallbackPtr3, CUFFT_CB_LD_COMPLEX, (void**)&cStack->d_iData ),"Error assigning CUFFT load callback.");
//      else
//      {
//	fprintf(stderr,"ERROR: %s bad stkIdx.\n", __FUNCTION__);
//	exit(EXIT_FAILURE);
//      }
//    }
#else
    fprintf(stderr, "ERROR: Not compiled with multiplication through CUFFT callbacks enabled. \n");
    exit(EXIT_FAILURE);
#endif	// WITH_MUL_PRE_CALLBACK
  }
}

