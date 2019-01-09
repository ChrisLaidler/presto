/** @file cuda_accel_PLN.cu
 *  @brief Utility functions and kernels to generate sections of ff plane
 *
 *  @author Chris Laidler
 *  @bug No known bugs.
 *
 *  Change Log
 *
 *  2017-04-24
 *    Create this file
 *    Moved some functions from optimisation to here
 *    Refactor a bunch of stuff to here
 *
 *  2017-10-26
 *    Added the plane - block with shuffle kernels (best block kernel so far)
 */

#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_math.h"
#include "candTree.h"
#include "cuda_response.h"
#include "cuda_accel_PLN.h"
#include "cuda_accel_utils.h"


#ifdef WITH_OPT_PTS_HRM

/** Plane generation, points, thread per ff point per harmonic
 *
 * This is fast and best of the blocked kernels
 *
 * @param pln
 * @param stream
 */
template<typename T>
__global__ void ffdotPln_ker3(float* powers, float2* fft, int noHarms, int harmWidth, double firstR, double firstZ, double rSZ, double zSZ, int noR, int noZ, int iStride, int oStride, optLocInt_t loR, optLocFloat_t norm, optLocInt_t hw, uint flags)
{
  const int tx		= blockIdx.x * blockDim.x + threadIdx.x;
  const int ty		= blockIdx.y * blockDim.y + threadIdx.y;

  const int hIdx	= tx / harmWidth ;
  const int ix		= tx % harmWidth ;
  const int iy		= ty;

  if ( ix < noR && iy < noZ)
  {
    int halfW;
    double r            = firstR + ix/(double)(noR-1) * rSZ ;
    double z            = firstZ - iy/(double)(noZ-1) * zSZ ;
    if (noZ == 1)
      z = firstZ;

    T real = 0;
    T imag = 0;

    const int hrm = hIdx+1;

    FOLD // Determine half width
    {
      halfW = getHw<T>(z*hrm, hw.val[hIdx]);
    }

    rz_convolution_cu<T, float2>(&fft[iStride*hIdx], loR.val[hIdx], iStride, r*hrm, z*hrm, halfW, &real, &imag);

    FOLD // Write values back to memory  .
    {
      if ( flags & (uint)(FLAG_HAMRS ) )
      {
	// Write per harming values
	if ( flags & (uint)(FLAG_CMPLX) )
	{
	  if ( flags & (uint)(FLAG_DOUBLE) )
	  {
	    double2 val;
	    val.x = real;
	    val.y = imag;
	    ((double2*)powers)[iy*oStride + ix*noHarms + hIdx ] = val ;
	  }
	  else
	  {
	    float2 val;
	    val.x = real;
	    val.y = imag;
	    ((float2*)powers)[iy*oStride + ix*noHarms + hIdx ] = val ;
	  }
	}
	else
	{
	  if ( flags & (uint)(FLAG_DOUBLE) )
	  {
	    ((double*)powers)[iy*oStride + ix*noHarms + hIdx ] = POWERCU(real, imag);
	  }
	  else
	  {
	    ((float*)powers)[iy*oStride + ix*noHarms + hIdx ] = POWERCU(real, imag);
	  }
	}
      }
      else
      {
	// Accumulate harmonic to total sum
	if ( flags & (uint)(FLAG_DOUBLE) )
	{
	  atomicAdd(&(((double*)powers)[iy*oStride + ix]), (double)POWERCU(real, imag) );
	}
	else
	{
	  atomicAdd(&(((float*)powers)[iy*oStride + ix]), (float)POWERCU(real, imag) );
	}
      }
    }
  }
}

#endif

template<typename T>
ACC_ERR_CODE ffdotPln_CPU(cuPlnGen* plnGen, cuRzHarmPlane* pln)
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  PROF // Profiling  .
  {
    NV_RANGE_PUSH("Write CVS");
  }

  cuHarmInput*	input = plnGen->input;
  
  infoMSG(8,8,"Harms %i sz: %i x %i \n", pln->noHarms, pln->noZ, pln->noR );

  if ( !err )
  {
    // Print column
    for (int indy = 0; indy < pln->noZ; indy++ )
    {
      // Print Z value
      double z = pln->centZ + pln->zSize/2.0 - indy/(double)(pln->noZ-1) * (pln->zSize) ;
      if ( pln->noZ == 1 )
	z = pln->centZ;

      // Print plane values
      for (int indx = 0; indx < pln->noR ; indx++ )
      {
	double r = pln->centR - pln->rSize/2.0 + indx/(double)(pln->noR-1) * (pln->rSize) ;

	if ( pln->type == CU_STR_INCOHERENT_SUM )
	{
	  double power = 0;
	  for ( int hIdx = 0; hIdx < pln->noHarms; hIdx++)
	  {
	    int halfW = cu_z_resp_halfwidth<double>( z*(hIdx+1), (presto_interp_acc)plnGen->hw[hIdx] );

	    T real, imag;
	    rz_convolution_cu<T, float2>((float2*)&input->h_inp[hIdx*input->stride], input->loR[hIdx], input->stride, r*(hIdx+1), z*(hIdx+1), halfW, &real, &imag);
	    power += POWERCU(real,imag);
	  }
	  if ( pln->type == CU_FLOAT  )
	  {
	    ((float*)pln->h_data)[indy*pln->zStride + indx] = power;
	  }
	  if ( pln->type == CU_DOUBLE  )
	  {
	    ((double*)pln->h_data)[indy*pln->zStride + indx] = power;
	  }
	}
      }
    }
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP("Write CVS");
  }

  return err;
}

/** Get a nice text representation of the current plane kernel name
 *
 * @param pln     The plane to check options for
 * @param name    A text pointer to put the name into
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE getKerName(cuPlnGen* plnGen, char* name)
{
  ACC_ERR_CODE err = ACC_ERR_NONE;

  if      ( plnGen->flags & FLAG_OPT_BLK_HRM )
    sprintf(name,"%s","BLK_HRM" );
  else if ( plnGen->flags & FLAG_OPT_BLK_SFL )
    sprintf(name,"%s","BLK_SHL" );
  else if ( plnGen->flags & FLAG_OPT_PTS_HRM )
    sprintf(name,"%s","PTS_HRM" );
  else
    sprintf(name,"%s","UNKNOWN" );

  return err;
}

/** Set plane type settings from flags
 *
 * @param plnGen  The plane to read the flags from
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE setPlnGenTypeFromFlags( cuPlnGen* plnGen )
{
  ACC_ERR_CODE	 err		= ACC_ERR_NONE;

  if ( plnGen && plnGen->pln )
  {
    plnGen->pln->type = CU_NONE;

    if     ( (plnGen->flags & FLAG_CMPLX) && (plnGen->flags & FLAG_DOUBLE) )
    {
      plnGen->pln->type += CU_CMPLXD;
    }
    else if ( plnGen->flags & FLAG_CMPLX )
    {
      plnGen->pln->type += CU_CMPLXF;
    }
    else if ( plnGen->flags & FLAG_DOUBLE )
    {
      plnGen->pln->type += CU_DOUBLE;
    }
    else
    {
      plnGen->pln->type += CU_FLOAT;
    }

    if ( plnGen->flags & FLAG_HAMRS )
    {
      plnGen->pln->type += CU_STR_HARMONICS;
    }
    else
    {
      plnGen->pln->type += CU_STR_INCOHERENT_SUM;
    }
  }
  else
  {
    err += ACC_ERR_NULL;
  }

  return err;
}

/** Zero the device memory
 *
 * Not this assumes resSz has been set
 *
 * @param plnGen
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE zeroPln( cuPlnGen* plnGen )
{
  ACC_ERR_CODE	 err		= ACC_ERR_NONE;

  if ( plnGen && plnGen->pln && plnGen->pln->d_data )
  {
    if ( plnGen->pln->resSz <= plnGen->pln->size )
    {
      infoMSG(7,7,"Zero plane device memory\n" );

      err += CUDA_ERR_CALL(cudaMemsetAsync ( plnGen->pln->d_data, 0, plnGen->pln->resSz, plnGen->stream ), "Zeroing memory");
      err += CUDA_ERR_CALL(cudaGetLastError(), "Zeroing the output memory.");
    }
    else
    {
      err += ACC_ERR_SIZE;
      ERROR_MSG(err, "ERROR: Size of results of f-fdot plane are greater than allocated memory,  %.2f MB > %.2f MB \n", plnGen->pln->resSz*1e-6, plnGen->pln->size*1e-6 );
    }
  }
  else
  {
    err += ACC_ERR_NULL;
  }

  return err;
}

/** Calculate the number of convolution operations needed to generate the plane
 *
 * This uses the current settings (size and half-width), thus assumes prep_Opt(...) has been called.
 *
 * This function returns values for each harmonic
 *
 * @param plnGen	The plane to read the flags from
 * @param cOps		A pointer to an array of minimum length of the number of harmonics, the results will be written to this array
 * @return		ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE ffdotPln_cOps_harms( cuPlnGen* plnGen, unsigned long long* cOps)
{
  ACC_ERR_CODE	 err		= ACC_ERR_NONE;
  cuRzHarmPlane* pln 		= plnGen->pln;

  if ( !cOps )
    return ACC_ERR_NULL;

  // Check input
  if ( plnGen->accu == 0 )
  {
    err += ACC_ERR_UNINIT;
  }
  else
  {
    for ( int hIdx = 0; hIdx < pln->noHarms; hIdx++ )
    {
      cOps[hIdx] = 0;
      for ( int z = 0; z < pln->noZ; z++ )
      {
	double zv	= pln->centZ + pln->zSize/2.0 - pln->zSize*(z+1)/(double)pln->noZ;
	int halfW;

	if ( plnGen->hw[hIdx] <= 0 )
	{
	  // In this case the hw value is the accuracy, so calculate halfwidth
	  halfW		= cu_z_resp_halfwidth<double>( zv*(hIdx+1), (presto_interp_acc)plnGen->hw[hIdx] );
	}
	else
	{
	  // halfwidth was previously calculated
	  halfW		= plnGen->hw[hIdx];
	}

	cOps[hIdx] += halfW * 2 * ( pln->noR ) ;
      }
    }
  }

  return err;
}

/** Calculate the number of convolution operations needed to generate the plane
 *
 * This uses the current settings (size and half-width), thus assumes prep_Opt(...) has been called.
 *
 * @param plnGen	The plane to read the flags from
 * @param cOps		A pointer to a value where the result will be written to
 * @return		ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE ffdotPln_cOps( cuPlnGen* plnGen, unsigned long long* cOps)
{
  ACC_ERR_CODE	 err		= ACC_ERR_NONE;
  cuRzHarmPlane* pln 		= plnGen->pln;

  if ( !cOps )
    return ACC_ERR_NULL;

  // Check input
  if ( plnGen->accu == 0 )
  {
    err += ACC_ERR_UNINIT;
  }
  else
  {
    unsigned long long cOps_hrm[32];
    *cOps = 0;

    err += ffdotPln_cOps_harms( plnGen, cOps_hrm);
    ERROR_MSG(err, "ERROR: Preparing plane.");

    for ( int hIdx = 0; hIdx < pln->noHarms; hIdx++ )
    {
      *cOps += cOps_hrm[hIdx];
    }
  }

  return err;
}

/** Call the kernel to create the plane
 *
 * This assumes the settings for the plane have been checked - ffdotPln_prep
 * and the correct input is on the device -  ffdotPln_input
 *
 * @param pln	  The plane to generate
 * @param fft	  FFT data structure
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
template<typename T>
ACC_ERR_CODE ffdotPln_ker( cuPlnGen* plnGen )
{
  ACC_ERR_CODE	 err	= ACC_ERR_NONE;
  confSpecsOpt*	 conf	= plnGen->conf;
  //cuRespPln* 	 rpln 	= plnGen->responsePln;
  cuRzHarmPlane* pln 	= plnGen->pln;
  cuHarmInput*	 input	= plnGen->input;

  // Data structures to pass to the kernels
  optLocInt_t	rOff;			// Row offset
  optLocInt_t	hw;			// The halfwidth for each harmonic
  optLocFloat_t	norm;			// Normalisation factor for each harmonic

  infoMSG(4,4,"Calling CUDA kernel to generate plane.\n" );

  // Calculate bounds on potently newly scaled plane
  double maxZ		= (pln->centZ + pln->zSize/2.0);
  double minR		= (pln->centR - pln->rSize/2.0);

  if (!pln->zSize || !pln->zSize)
  {
    err += ACC_ERR_UNINIT;
  }

  // Initialise values to 0
  for( int h = 0; h < OPT_MAX_LOC_HARMS; h++)
  {
    rOff.val[h]		= input->loR[h];
    hw.val[h]		= plnGen->hw[h];
    norm.val[h]		= sqrt(input->norm[h]);			// Correctly normalised by the sqrt of the local power

    if ( h < plnGen->pln->noHarms && plnGen->hw[h] == 0 )
    {
      err += ACC_ERR_UNINIT;
    }
  }

  if ( ERROR_MSG(err, "ERROR: Generating f-fdot plane section.") )
    return err;

  err += setPlnGenTypeFromFlags(plnGen);

  FOLD // Call kernel  .
  {
    dim3 dimBlock, dimGrid;

    if ( conf->flags & FLAG_SYNCH )
    {
      CUDA_SAFE_CALL(cudaEventRecord(plnGen->compInit, plnGen->stream),"Recording event: compInit");
    }

    // These are the only flags specific to the kernel
    uint flags =  plnGen->flags & ( FLAG_HAMRS | FLAG_CMPLX | FLAG_DOUBLE );

    if      ( plnGen->flags & FLAG_OPT_CPU_PLN )	// Use basic block kernel
    {
      err += ffdotPln_CPU<T>(plnGen, plnGen->pln);
    }
    else if ( plnGen->flags & FLAG_OPT_BLK )		// Use block kernel
    {
      infoMSG(4,4,"Block kernel [ No threads %i  Width %i no Blocks %i]\n", (int)pln->blkDimX, pln->blkWidth, pln->blkCnt);

      if      ( plnGen->flags & FLAG_OPT_BLK_HRM )	// Shared coefficients by storing running sums in registers - starts to still at 8 which is a bit low
      {
#ifdef WITH_OPT_BLK_HRM
	infoMSG(5,5,"Block kernel 3 - Harms");

	dimBlock.x = MIN(16, pln->blkDimX);
	dimBlock.y = MIN(16, pln->noZ);

	int noX = ceil(pln->blkDimX / (float)dimBlock.x);
	int harmWidth = noX*dimBlock.x;

	err += zeroPln(plnGen);

	// One block per harmonic, thus we can sort input powers in shared memory
	dimGrid.x = noX * pln->noHarms ;
	dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

	// Call the kernel to normalise and spread the input data
	switch (pln->blkCnt)
	{
#if  MAX_OPT_BLK_NO >= 1
	  case 1:
	    // NOTE: in this case I find the points kernel to be a bit faster (~5%)
	    ffdotPlnByBlk_ker3<T, 1> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
#endif

#if  MAX_OPT_BLK_NO >= 2
	  case 2:
	    ffdotPlnByBlk_ker3<T, 2> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
#endif

#if  MAX_OPT_BLK_NO >= 3
	  case 3:
	    ffdotPlnByBlk_ker3<T, 3> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
#endif

#if  MAX_OPT_BLK_NO >= 4
	  case 4:
	    ffdotPlnByBlk_ker3<T, 4> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
#endif

#if  MAX_OPT_BLK_NO >= 5
	  case 5:
	    ffdotPlnByBlk_ker3<T, 5> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
#endif

#if  MAX_OPT_BLK_NO >= 6
	  case 6:
	    ffdotPlnByBlk_ker3<T, 6> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
#endif

#if  MAX_OPT_BLK_NO >= 7
	  case 7:
	    ffdotPlnByBlk_ker3<T, 7> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
#endif

#if  MAX_OPT_BLK_NO >= 8
	  case 8:
	    ffdotPlnByBlk_ker3<T, 8> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
#endif

#if  MAX_OPT_BLK_NO >= 9
	  case 9:
	    ffdotPlnByBlk_ker3<T, 9> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
#endif

#if  MAX_OPT_BLK_NO >= 10
	  case 10:
	    ffdotPlnByBlk_ker3<T,10> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
#endif

#if  MAX_OPT_BLK_NO >= 11
	  case 11:
	    ffdotPlnByBlk_ker3<T,11> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
#endif

#if  MAX_OPT_BLK_NO >= 12
	  case 12:
	    ffdotPlnByBlk_ker3<T,12> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
#endif

#if  MAX_OPT_BLK_NO >= 13
	  case 13:
	    ffdotPlnByBlk_ker3<T,13> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
#endif

#if  MAX_OPT_BLK_NO >= 14
	  case 14:
	    ffdotPlnByBlk_ker3<T,14> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
#endif

#if  MAX_OPT_BLK_NO >= 15
	  case 15:
	    ffdotPlnByBlk_ker3<T,15> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
#endif

#if  MAX_OPT_BLK_NO >= 16
	  case 16:
	    ffdotPlnByBlk_ker3<T,16> <<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, pln->rSize, pln->blkDimX, pln->noR, pln->noZ, pln->blkWidth, input->stride, pln->zStride, rOff, norm, hw, flags);
	    break;
#endif
	  default:
	  {
	    fprintf(stderr, "ERROR: %s has not been templated for %i blocks.\n", __FUNCTION__, pln->blkCnt );
	    exit(EXIT_FAILURE);
	  }
	}
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_BLK_HRM.\n");
	err += ACC_ERR_COMPILED;
#endif
      }
      else if ( plnGen->flags & FLAG_OPT_BLK_SFL )	// Kepler shuffle commands - Nice and fast!
      {
#ifdef WITH_OPT_BLK_SHF
	infoMSG(5,5,"Block kernel 4 - Shuffle");

	// Zero the plane memory
	err += zeroPln(plnGen);

	int noB = pln->blkCnt * pln->blkWidth ;
	int mnW = 1 ;
	while ( (noB % mnW) || (pln->noR % (noB/mnW)) )
	{
	  mnW++;
	}
	noB /= mnW;
	int dim1 = pln->noR / noB ;			// This is basically blockX

	float2 *inp = (float2*)input->d_inp;
	void* out   = (void*)pln->d_data;

	while ( noB )
	{
	  int widh = MAX_OPT_SFL_NO;

	  while ( widh*mnW > noB )
	  {
	    widh /= 2;
	  }

	  int rSize = widh*mnW;
	  int rDim  = widh*dim1;

	  FOLD // Create a kernel for the slice being handled
	  {
	    dimBlock.x = MAX(MAX_OPT_SFL_NO,16);				// This max ensures all values are in a single warp
	    dimBlock.y = MIN(16, pln->noZ);

	    int noX = ceil(rDim / (float)dimBlock.x);
	    int harmWidth = noX*dimBlock.x;

	    // One block per harmonic, thus we can sort input powers in Shared memory
	    dimGrid.x = noX * pln->noHarms ;
	    dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

	    FOLD // Call the kernel to normalise and spread the input data
	    {
	      double rSizeSl	= (rDim-1)*(mnW)/double(dim1);

	      infoMSG(6,6,"Partial kernel - Width:%2i  Dim: %4i\n", rSize, rDim);

	      ffdotPlnByShfl_ker<T><<<dimGrid, dimBlock, 0, plnGen->stream >>>(out, inp, pln->noHarms, harmWidth, minR, maxZ, pln->zSize, rSizeSl, dim1, rDim, pln->noZ, mnW, input->stride, pln->zStride, rOff, norm, hw, flags, widh);
	    }
	  }

	  FOLD	// Prepare for next slice
	  {
	    noB  -= rSize;		// Decrease the remaining section of the plane to create
	    minR += rSize;		// Shift location of next section of the plane

	    // Stride the output
	    if     ( (flags & FLAG_CMPLX) && (flags & FLAG_DOUBLE) )
	    {
	      out = &(((double2*)out)[rDim*pln->noHarms]);
	    }
	    else if ( flags & FLAG_CMPLX )
	    {
	      out = &(((float2*)out)[rDim*pln->noHarms]);
	    }
	    else if (flags & FLAG_DOUBLE )
	    {
	      out = &(((double*)out)[rDim]);
	    }
	    else
	    {
	      out = &(((float*)out)[rDim]);
	    }
	  }
	}
#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_BLK_HRM.\n");
	err += ACC_ERR_COMPILED;
#endif
      }
      else
      {
	fprintf(stderr, "ERROR: No block optimisation specified.\n");
	err += ACC_ERR_INVLD_CONFIG;
      }
    }
    else						// Use point kernel
    {
      infoMSG(4,4,"Grid kernel\n");

      dimBlock.x = 16;
      dimBlock.y = 16;
      dimBlock.z = 1;

      if      ( plnGen->flags &  FLAG_OPT_PTS_HRM )	// Thread point of harmonic  .
      {
#ifdef WITH_OPT_PTS_HRM
	infoMSG(5,5,"Flat kernel 3 - Harmonics\n");

	int noX = ceil(pln->noR / (float)dimBlock.x);
	int harmWidth = noX*dimBlock.x;

	err += zeroPln(plnGen);

	// One block per harmonic, thus we can sort input powers in Shared memory
	dimGrid.x = noX * pln->noHarms ;
	dimGrid.y = ceil(pln->noZ/(float)dimBlock.y);

	// Call the kernel create a section of the f-fdot plane
	ffdotPln_ker3<T><<<dimGrid, dimBlock, 0, plnGen->stream >>>((float*)pln->d_data, (float2*)input->d_inp, pln->noHarms, harmWidth, minR, maxZ, pln->rSize, pln->zSize, pln->noR, pln->noZ, input->stride, pln->zStride, rOff, norm, hw, flags);

#else
	fprintf(stderr, "ERROR: Not compiled with WITH_OPT_PTS_HRM.\n");
	err += ACC_ERR_COMPILED;
#endif
      }
      else
      {
	fprintf(stderr, "ERROR: No optimisation plane kernel specified.\n");
	err += ACC_ERR_INVLD_CONFIG;
      }
    }

    err += CUDA_ERR_CALL(cudaGetLastError(), "Calling the ffdot_ker kernel.");

    if ( conf->flags & FLAG_SYNCH )
    {
      CUDA_SAFE_CALL(cudaEventRecord(plnGen->compCmp, plnGen->stream), "Recording event: compCmp");
    }
  }

  return err;
}

/** Check if the plane, with current settings, requires new input
 *
 * This does not load the actual input
 * This check the input in the input data structure of the plane
 *
 * @param pln     The plane to check, current settings ( centZ, centR, zSize, rSize, etc.) used
 * @param fft     The FFT data that will make up the input
 * @param newInp  Set to 1 if new input is needed
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE chkInput_ffdotPln( cuPlnGen* plnGen, fftInfo* fft, int* newInp)
{
  return chkInput_pln(plnGen->input, plnGen->pln, fft, newInp);
}

/** Copy relevant input from FFT to data structure normalising as needed
 *
 *  Note this contains a blocking synchronisation to make sure the pinned host memory is free
 *
 * @param plnGen  The plane to check
 * @param fft     The FFT data that will make up the input
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE prepInput_ffdotPln( cuPlnGen* plnGen, fftInfo* fft )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  double rSize = MAX(plnGen->pln->rSize, 20);
  double zSize = MAX(plnGen->pln->zSize, 20*plnGen->conf->zScale);

  err += loadHostHarmInput(plnGen->input, fft, plnGen->pln->centR, plnGen->pln->centZ, rSize, zSize, plnGen->pln->noHarms, plnGen->flags, &plnGen->inpCmp );
  ERROR_MSG(err, "ERROR: Loading input values.");

  return err;
}

/** Set the per harmonic half width using plane accuracy
 *
 * @param plnGen  The plane to check
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE setHalfWidth_ffdotPln( cuPlnGen* plnGen )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  // Check input
  if ( plnGen->accu == 0 )
  {
    err += ACC_ERR_UNINIT;
  }
  else
  {
    if ( plnGen->accu == LOWACC )
    {
      infoMSG(4,4,"Half width: standard accuracy");
    }
    else
    {
      infoMSG(4,4,"Half width: high accuracy");
    }

    // Initialise values to 0
    for( int hIdx = 0; hIdx < OPT_MAX_LOC_HARMS; hIdx++)
    {
      plnGen->hw[hIdx] = 0;
    }
    plnGen->maxHalfWidth = 0;

    double 	maxZ	= (plnGen->pln->centZ + plnGen->pln->zSize/2.0);
    double	minZ	= (plnGen->pln->centZ - plnGen->pln->zSize/2.0);
    double	lrgstZ	= MAX(fabs(maxZ), fabs(minZ));

    for( int hIdx = 0; hIdx < plnGen->pln->noHarms; hIdx++)
    {
      // TODO: Check OPT
      plnGen->hw[hIdx]	= cu_z_resp_halfwidth<double>(lrgstZ*(hIdx+1), plnGen->accu );
      MAXX(plnGen->maxHalfWidth, plnGen->hw[hIdx]);

      // Reset the halfwidth back to what its meant to be back
      if ( (plnGen->flags & FLAG_OPT_DYN_HW) || (plnGen->pln->zSize*(hIdx+1) >= 2) )
      {
	plnGen->hw[hIdx] = plnGen->accu;
      }
    }
  }

  return err;
}

/** Copy pre-prepared memory from pinned host memory to device memory
 *
 * This assumes that the input data has been written to the pinned host memory
 *
 * @param pln     The plane to check
 * @param fft     The FFT data that will make up the input
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE cpyInput_ffdotPln( cuPlnGen* plnGen, fftInfo* fft )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  infoMSG(4,4,"1D async memory copy H2D");

  err += CUDA_ERR_CALL(cudaMemcpyAsync(plnGen->input->d_inp, plnGen->input->h_inp, plnGen->input->stride*plnGen->input->noHarms*sizeof(fcomplexcu), cudaMemcpyHostToDevice, plnGen->stream), "Copying optimisation input to the device");
  err += CUDA_ERR_CALL(cudaEventRecord(plnGen->inpCmp, plnGen->stream),"Recording event: inpCmp");

  err += CUDA_ERR_CALL(cudaGetLastError(), "Copying plane input to device.");

  return err;
}

ACC_ERR_CODE setTypeFlag( cuPlnGen* plnGen, float nothing )
{
  plnGen->flags &= ~FLAG_DOUBLE;
  return ACC_ERR_NONE;
}

ACC_ERR_CODE setTypeFlag( cuPlnGen* plnGen, double nothing )
{
  plnGen->flags |= FLAG_DOUBLE;
  return ACC_ERR_NONE;
}

/** Check the configuration of how the plane section is going to be generated
 *
 * Note the configuration flags are used to set the optimiser flags
 *
 * @param pln	  optimiser
 * @param fft	  FFT data structure
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
template<typename T>
ACC_ERR_CODE prep_Opt( cuPlnGen* plnGen, fftInfo* fft )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;
  confSpecsOpt*	conf		= plnGen->conf;

  infoMSG(4,4,"Prep plain generator.\n");

  FOLD // Determine optimisation kernels  .
  {
    // Initialise to single column
    plnGen->pln->blkCnt		= 1;
    plnGen->pln->blkWidth	= 1;
    plnGen->pln->blkDimX	= plnGen->pln->noR;
    char kerName[20];
    double dup2pad_ratio;
    T Nothing = 0;				// This is just used to specify the type of the templated functions

    // Clear all "local" kernels
    err += remOptFlag(plnGen, FLAG_PLN_ALL );

    err += setTypeFlag(plnGen, Nothing);

    FOLD // Calculate ratio of redundant duplicates vs padding (points) for block kernels
    {
      double padSum = 0;
      double dupSum = 0;

      for ( int harm =1; harm <= plnGen->pln->noHarms ; harm++)
      {
	double rSz = plnGen->pln->rSize*harm;
	double pad = (ceil(rSz)-rSz)/rSz*plnGen->pln->noR ;
	double dup = MAX(0, rSz-1)/rSz*plnGen->pln->noR ;

	padSum += pad;
	dupSum += dup;
      }
      dup2pad_ratio = (dupSum) / (padSum+1) ;		// +1 just to stop devision by zero

      infoMSG(7,7,"Padding to Duplicates of %.0f:%.0f (1:%.2f) .\n", padSum, dupSum, dup2pad_ratio);
    }

    if ( ( plnGen->pln->rSize >= 1 ) && ( (conf->flags & FLAG_OPT_BLK) || !(conf->flags & FLAG_OPT_PTS) ) ) // Use the block kernel
    {

      if ( !(conf->flags & FLAG_OPT_BLK) )	// Auto select block kernel
      {
	// No points kernel so get one
#if	defined(WITH_OPT_BLK_SHF)
	err += setOptFlag(plnGen, FLAG_OPT_BLK_SFL );
#elif	defined(WITH_OPT_BLK_HRM)
	err += setOptFlag(plnGen, FLAG_OPT_BLK_HRM );
#elif	defined(WITH_OPT_BLK_NRM)
	err += setOptFlag(plnGen, FLAG_OPT_BLK_NRM );
#else
	fprintf(stderr,"ERROR: Not compiled with any per point block creation kernels.")
	err += ACC_ERR_COMPILED;
#endif
	getKerName(plnGen, kerName);
	infoMSG(6,6,"Auto select block kernel %s.\n", kerName);
      }
      else
      {
	// Set block kernel from "global" settings
	err += setOptFlag(plnGen, (conf->flags & FLAG_OPT_BLK) );

	getKerName(plnGen, kerName);
	infoMSG(6,6,"Specified block Kernel %s\n", kerName );
      }

      // Set size and resolution
      err += ffdotPln_calcCols( plnGen->pln, plnGen->flags, conf->blkDivisor, conf->blkMax);

      // Set column section flags
      err += setOptFlag(plnGen, (conf->flags & FLAG_RES_ALL) );
    }
    else
    {
      if ( !(conf->flags&FLAG_OPT_PTS) )	// Auto select
      {
	// No points kernel so get one
#if	defined(WITH_OPT_PTS_HRM)
	err += setOptFlag(plnGen, FLAG_OPT_PTS_HRM );
#else
	fprintf(stderr,"ERROR: Not compiled with any thread per point plane creation kernels.")
	err += ACC_ERR_COMPILED;
#endif

	getKerName(plnGen, kerName);
	infoMSG(6,6,"Auto select points kernel %s.\n", kerName);
      }
      else
      {
	// Set block kernel from "global" settings
	err += setOptFlag(plnGen, (conf->flags & FLAG_OPT_PTS) );

	getKerName(plnGen, kerName);
	infoMSG(6,6,"Specified points Kernel %s\n", kerName );
      }
    }

    if ( !(plnGen->flags & FLAG_HAMRS) && (plnGen->flags & FLAG_CMPLX) )
    {
      fprintf(stderr, "WARNING: Can't return sum of complex numbers, changing to incoherent sum of powers.\n");
      plnGen->flags &= ~(FLAG_CMPLX);
    }

    infoMSG(4,4,"Size (%.6f x %.6f) Points (%i x %i) %i  Resolution: %.7f r  %.7f z.\n", plnGen->pln->rSize,plnGen->pln->zSize, plnGen->pln->noR, plnGen->pln->noZ, plnGen->pln->noR*plnGen->pln->noZ, plnGen->pln->rSize/double(plnGen->pln->noR-1), plnGen->pln->zSize/double(plnGen->pln->noZ-1) );

    err += setPlnGenTypeFromFlags(plnGen);

    err += stridePln(plnGen->pln, plnGen->gInf);

    // Now snap the grid to the centre
    //err += snapPlane(opt->pln); // TODO: This is bad, need to snap to the candidate

    int cnt = __builtin_popcount (plnGen->flags & FLAG_OPT_KER_ALL);
    if ( cnt > 1 )
    {
      fprintf(stderr, "WARNING: Invalid configuration, multiple block kernels selected.");
      err += ACC_ERR_INVLD_CONFIG;
    }
  }

  err += setHalfWidth_ffdotPln( plnGen );

  return err;
}

/**
 *  This only calls the asynchronous copy
 *
 *  To make sure the points are in host memory call ffdotPln_ensurePln
 *
 * @param pln
 * @param fft
 * @return
 */
ACC_ERR_CODE ffdotPln_cpyResultsD2H( cuPlnGen* plnGen, fftInfo* fft )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  FOLD // Copy data back to host  .
  {
    infoMSG(4,4,"1D async memory copy D2H");

    CUDA_SAFE_CALL(cudaMemcpyAsync(plnGen->pln->h_data, plnGen->pln->d_data, plnGen->pln->resSz, cudaMemcpyDeviceToHost, plnGen->stream), "Copying optimisation results back from the device.");
    CUDA_SAFE_CALL(cudaEventRecord(plnGen->outCmp, plnGen->stream),"Recording event: outCmp");
  }

  return err;
}

/** Ensure the values are in host memory
 *
 * This assumes the kernels has been called and the asynchronous memory copy has been called
 *
 * Block on memory copy and make sure the points have been written to host memory
 *
 * @param pln
 * @param fft
 * @return
 */
ACC_ERR_CODE ffdotPln_ensurePln( cuPlnGen* plnGen, fftInfo* fft )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  FOLD // A blocking synchronisation to ensure results are ready to be proceeded by the host  .
  {
    infoMSG(4,4,"Blocking synchronisation on %s", "outCmp" );

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("EventSynch");
    }

    CUDA_SAFE_CALL(cudaEventSynchronize(plnGen->outCmp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");

    PROF // Profiling  .
    {
      NV_RANGE_POP("EventSynch");
    }

  }

  return err;
}

/** Calculate the section of ffdot plane using the GPU and put the results in host memory
 *
 * This is the function to use if you want to create a section of ff plane.
 *
 * This assumes that plnGen has been initialise and the relevant flags set
 * If the FLAG_RES_CLOSE or FLAG_RES_FAST are set the size and resolution of the plane may be changed slightly
 *
 * @param plnGen
 * @param fft
 * @param newInp
 * @return		ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
template<typename T>
ACC_ERR_CODE ffdotPln( cuPlnGen* plnGen, fftInfo* fft, int* newInp )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  infoMSG(4,4,"Generate plane ff section, Centred on (%.6f, %.6f) with %2i harmonics.\n", plnGen->pln->centR, plnGen->pln->centZ, plnGen->pln->noHarms );

  err += prep_Opt<T>( plnGen,  fft );
  if ( ERROR_MSG(err, "ERROR: Preparing plane.") )
    return err;

  err += input_plnGen( plnGen, fft, newInp );
  if (ERROR_MSG(err, "ERROR: Getting input for the plane."))
    return err;

  err += ffdotPln_ker<T>( plnGen );
  if (ERROR_MSG(err, "ERROR: Running the kernel."))
    return err;

  err += ffdotPln_cpyResultsD2H( plnGen, fft );
  if (ERROR_MSG(err, "ERROR: Copying the results."))
    return err;

  err += ffdotPln_ensurePln( plnGen, fft );
  if (ERROR_MSG(err, "ERROR: Waiting."))
    return err;

  return err;
}

/** Make sure the input is for the current plane settings is ready in device memory
 *
 * This checks if new memory is needed
 * Normalises it and copies it to the device
 *
 * @param pln	  optimiser
 * @param fft	  FFT data structure
 * @param newInp  Set to 1 if new input is needed
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE input_plnGen( cuPlnGen* plnGen, fftInfo* fft, int* newInp )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  // Check input
  int newInp_l;
  err += chkInput_ffdotPln( plnGen, fft, &newInp_l );

  if ( newInp_l ) // Copy input data to the device  .
  {
    err += prepInput_ffdotPln( plnGen, fft );

    err += cpyInput_ffdotPln( plnGen, fft );
  }

  if ( newInp )
    *newInp = newInp_l;

  return err;
}

/** Check the configuration of how the plane section is going to be generated
 *
 * Note the flags passed in are the "global" configuration flags
 * these used to set the column widths, these are dependent on the flags:
 *   FLAG_RES_CLOSE   or   FLAG_RES_FAST
 *
 * @param pln	  optimiser
 * @param fft	  FFT data structure
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE ffdotPln_calcCols( cuRzHarmPlane* pln, int64_t flags, int colDivisor, int target_noCol)
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  pln->blkCnt	= 1;
  pln->blkWidth	= 1;
  pln->blkDimX	= pln->noR;

  infoMSG(4,4,"Calculate plane column sizes. \n");

  if (colDivisor < 1)
  {
    infoMSG(6,6,"ERROR: Invalid divisor %i.\n", colDivisor);
    colDivisor = 1;
    err += ACC_ERR_INVLD_CONFIG;
  }

  FOLD // Determine optimisation kernels  .
  {
    infoMSG(6,6,"Orr  #R: %3i  Sz: %9.6f  Res: %9.6f \n", pln->noR, pln->rSize, pln->rSize/(double)(pln->noR-1) );

#ifdef WITH_OPT_BLK_SHF
    if ( flags & FLAG_OPT_BLK_SFL  )
    {
      // Use shuffle kernel

      FOLD // Select a good power of two number of columns  .
      {
	int divs = 2;

	if ( pln->rSize > 8 )
	{
	  divs = 4;
	}
	else if ( pln->rSize > 1 )
	{
	  divs = MIN(MAX_OPT_SFL_NO,exp2(floor(log2(pln->rSize))));
	}
	else
	{
	  // Smaller than one so pretty much have to have a width of 1, and log2 will give negative values
	  divs = 1;
	}
	
	pln->blkCnt	= ceil(pln->rSize/(double)divs)*divs;
      }

      // Other settings are now set using the column width
      // The size and dimension may change significantly but the resolution will be similar or higher (same "accuracy")
      // NOTE: The number of points is set to fill the columns, thus ensures that potently higher harmonics can still be calculated exploiting redundant calculations
      // TODO: Check width 1
      pln->blkWidth	= ceil(pln->rSize / (double)pln->blkCnt );			// Max column width in Fourier bins
      double rPerBlock	= pln->noR / ( pln->rSize / (double)pln->blkWidth );		// Calculate the number of threads per column
      pln->blkDimX	= ceil(rPerBlock/(double)colDivisor)*colDivisor;		// Make the column width divisible (this can speed up processing)

      pln->noR		= pln->blkCnt * pln->blkDimX;					// This is necessary for this kernel
      pln->rSize	= (pln->noR-1)*(pln->blkWidth)/double(pln->blkDimX);		//
    }
    else
#endif
    if      ( flags & FLAG_RES_CLOSE )
    {
      // This method tries to create a block structure that is close to the original
      // The size will always be same or larger than that specified
      // And the resolution will be the same of finer than that specified

      // TODO: Check noR on fermi cards, the increased registers may justify using larger blocks widths
      do
      {
	pln->blkWidth++;
	pln->blkDimX		= ceil( pln->blkWidth * (pln->noR-1) / pln->rSize );
	MINN(pln->blkDimX, pln->noR );
	pln->blkCnt		= ceil( ( pln->rSize + 1 / (double)pln->blkDimX ) / pln->blkWidth );
	// Can't have blocks wider than 16 - Thread block limit
      }
      while ( pln->blkCnt > (double)MIN(MAX_OPT_BLK_NO,target_noCol) );

      if ( pln->blkCnt == 1 )
      {
	pln->blkDimX		= pln->noR;
      }
      else
      {
	pln->noR		= ceil( pln->rSize / (double)(pln->blkWidth) * (pln->blkDimX) ) + 1 ;
	pln->rSize		= (pln->noR-1)*(pln->blkWidth)/double(pln->blkDimX);
      }
    }
    else if ( flags & FLAG_RES_FAST  )
    {
      // This method attempts to choose the parameters so as to be computationally fast
      // This speed is obtained at the "cost" of the size and resolution of the plane section created.
      // Generally the resolution will be higher than the original
      // The final width may be slightly smaller (by one resolution point)
      // The block widths are set to be nicely divisible numbers, this can make the kernel a bit faster

      // Get initial best values

      {
	pln->blkWidth		= ceil(pln->rSize / (double)MIN(MAX_OPT_BLK_NO,target_noCol) );	// Max column width in Fourier bins
	double rPerBlock	= pln->noR / ( pln->rSize / (double)pln->blkWidth );	// Calculate the number of threads per column
	pln->blkDimX		= ceil(rPerBlock/(double)colDivisor)*colDivisor;	// Make the column width divisible (this can speed up processing)
	pln->blkCnt		= ceil( ( pln->rSize ) / pln->blkWidth );		// Number of columns

	// Check if we should increase column width
	if( rPerBlock < (double)colDivisor*0.80 )
	{
	  // NOTE: Could look for higher divisors ie 3/2
	  pln->blkCnt		= ceil(pln->noR/(double)colDivisor);
	  pln->blkDimX		= colDivisor;
	  pln->blkWidth		= floor(pln->rSize/(double)pln->blkCnt);
	}

	pln->noR		= ceil( pln->rSize / (double)(pln->blkWidth) * (pln->blkDimX) ) + 1; // May as well get close but above
	pln->noR		= ceil( pln->noR / (double)colDivisor ) * colDivisor ;	// Make the column width divisible (this can speed up processing)
	if ( pln->noR > pln->blkCnt * pln->blkDimX )
	  pln->noR		= pln->blkCnt * pln->blkDimX;				// This is the reduction that reduces the size of the final plane to one resolution point less than the "desired" width
	pln->rSize		= (pln->noR-1)*(pln->blkWidth)/double(pln->blkDimX);
      }
    }
    else
    {
      // This will do the convolution exactly as is
      // NOTE: If the resolution is a "good" value there is still the possibility to do it with blocks - Not yet implemented
      // That would require some form of prime factorisation of numerator and denominator (I think), this could still be implemented

      pln->blkDimX		= pln->noR;
      pln->blkWidth		= 1;
      pln->blkCnt		= 1;
    }

    infoMSG(6,6,"New  #R: %3i  Sz: %9.6f  Res: %9.6f  - Col Width: %2i  -  No cols: %.2f  -  col DimX: %2i \n", pln->noR, pln->rSize, pln->rSize/(double)(pln->noR-1), pln->blkWidth, pln->noR / (double)pln->blkDimX, pln->blkDimX );
  }

  // All kernels use the same output stride - These values can be changed later to suite a specific GPU memory alignment and data type
  pln->zStride		= pln->noR;
  
  return err;
}

/** Check if the plane, with current settings, requires new input
 *
 * This does not load the actual input
 * This check the input in the input data structure of the plane
 *
 * @param pln		The plane to check, current settings ( centZ, centR, zSize, rSize, etc.) used
 * @param fft		The FFT data that will make up the input
 * @param newInp	Set to 1 if new input is needed
 * @return		ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE chkInput_pln(cuHarmInput* input, cuRzHarmPlane* pln, fftInfo* fft, int* newInp)
{
  return  chkInput(input, pln->centR, pln->centZ, pln->rSize, pln->zSize, pln->noHarms, newInp);
}

/** Set the stride values of plane memory
 *
 * @param pln
 * @param elSize
 * @param gInf
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE stridePln(cuRzHarmPlane* pln, gpuInf* gInf)
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  infoMSG(6,6,"Aligning plane to device memory.\n" );

  size_t	zStride	= 0;
  size_t	hStride	= 0;
  size_t	elSz	= 0;

  if      ( pln->type == CU_CMPLXF )
  {
    infoMSG(7,7,"Output: complex - single precision \n" );
    elSz = sizeof(float2);
  }
  else if ( pln->type == CU_CMPLXD  )
  {
    infoMSG(7,7,"Output: complex - double precision \n" );
    elSz = sizeof(double2);
  }
  else if ( pln->type == CU_FLOAT  )
  {
    infoMSG(7,7,"Output: powers - single precision \n" );
    elSz = sizeof(float);
  }
  else if ( pln->type == CU_DOUBLE  )
  {
    infoMSG(7,7,"Output: powers - double precision \n" );
    elSz = sizeof(double);
  }
  else
  {
    infoMSG(6,6,"ERROR: Plane type has not been initialised.\n" );
    err += ACC_ERR_UNINIT;
  }

  if      ( pln->type == CU_STR_HARMONICS )
  {
    infoMSG(7,7,"Output: harmonics\n" );
    hStride = pln->noHarms;
  }
  else if ( pln->type == CU_STR_INCOHERENT_SUM )
  {
    infoMSG(7,7,"Output: incoherent sum.\n" );
    hStride = 1;
  }
  else
  {
    infoMSG(6,6,"ERROR: Plane type has not been initialised.\n" );
    err += ACC_ERR_UNINIT;
  }
  zStride = getStride(pln->noR*hStride, elSz, gInf->alignment);

  if ( zStride * pln->noZ * elSz < pln->size )
  {
    pln->zStride	= zStride;
  }
  else if ( pln->noR * pln->noZ * hStride * elSz  < pln->size )
  {
    fprintf(stderr, "ERROR: Plane size exceeds allocated memory!\n");

    err += ACC_ERR_MEM;
    pln->zStride	= 0;
  }
  else
  {
    // Well we just can can't have nicely aligned memory
    infoMSG(6,6,"Aligning plane to device memory would overflow the memory so no alignment.\n" );

    pln->zStride	= pln->noR;
  }

  // Set the size of the used part of memory
  pln->resSz = pln->zStride*pln->noZ*elSz;

  infoMSG(7,7,"Output size %.2f MB.\n", pln->resSz*1e-6 );

  return err;
}

/** Write plane points plane text file
 *
 * @param pln
 * @param f2
 * @return	ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE ffdotPln_writePlnToFile(cuRzHarmPlane* pln, FILE *f2)
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  PROF // Profiling  .
  {
    NV_RANGE_PUSH("Write CVS");
  }

  infoMSG(5,5,"Write CVS\n");

  // Add number of harmonics summed as the first line
  fprintf(f2,"Harm plane\n");
  fprintf(f2,"centR: %.23f\n", pln->centR);
  fprintf(f2,"centZ: %.23f\n", pln->centZ);
  fprintf(f2,"rSize: %.23f\n", pln->rSize);
  fprintf(f2,"zSize: %.23f\n", pln->zSize);
  fprintf(f2,"noZ: %.i\n",     pln->noZ);
  fprintf(f2,"noR: %.i\n",     pln->noR);
  fprintf(f2,"Harms: %i\n",    pln->noHarms);

  // Print type
  if      ( pln->type == CU_CMPLXF || pln->type == CU_CMPLXD )
    fprintf(f2,"Type: complex\n");
  else if ( pln->type == CU_FLOAT || pln->type == CU_DOUBLE )
    fprintf(f2,"Type: power\n");

  infoMSG(8,8,"Harms %i sz: %i x %i \n", pln->noHarms, pln->noZ, pln->noR );

  int noStrHarms;

  if      ( pln->type == CU_STR_HARMONICS      )
  {
    fprintf(f2,"Layout: Harmonics\n");
    noStrHarms = pln->noHarms;
  }
  else if ( pln->type == CU_STR_INCOHERENT_SUM )
  {
    fprintf(f2,"Layout: Sum\n");
    noStrHarms = 1;
  }
  else
  {
    infoMSG(6,6,"ERROR: Plane type has not been initialised.\n" );
    err += ACC_ERR_UNINIT;
  }

  if ( !err )
  {
    for ( int hIdx = 0; hIdx < noStrHarms; hIdx++)
    {
      FOLD // Print R values  .
      {
	fprintf(f2,"Harm %i", hIdx+1);

	for (int indx = 0; indx < pln->noR ; indx++ )
	{
	  double r = pln->centR - pln->rSize/2.0 + indx/(double)(pln->noR-1) * (pln->rSize) ;
	  fprintf(f2,"\t%.17e", r*(hIdx+1) );
	}
	fprintf(f2,"\n");
      }

      // Print column
      for (int indy = 0; indy < pln->noZ; indy++ )
      {
	// Print Z value
	double z = pln->centZ + pln->zSize/2.0 - indy/(double)(pln->noZ-1) * (pln->zSize) ;
	if ( pln->noZ == 1 )
	  z = pln->centZ;
	fprintf(f2,"%.17e", z*(hIdx+1));

	// Print plane values
	for (int indx = 0; indx < pln->noR ; indx++ )
	{
	  if      ( pln->type == CU_CMPLXF )
	  {
	    float2 val = ((float2*)pln->h_data)[indy*pln->zStride + indx*noStrHarms + hIdx];

	    fprintf(f2,"\t%.17e | %.17e", val.x, val.y);
	  }
	  else if ( pln->type == CU_CMPLXD )
	  {
	    double2 val = ((double2*)pln->h_data)[indy*pln->zStride + indx*noStrHarms + hIdx];

	    fprintf(f2,"\t%.17e | %.17e", val.x, val.y);
	  }
	  else if ( pln->type == CU_FLOAT  )
	  {
	    float val = ((float*)pln->h_data)[indy*pln->zStride + indx*noStrHarms + hIdx];

	    fprintf(f2,"\t%.17e", val);
	  }
	  else if ( pln->type == CU_DOUBLE  )
	  {
	    double val = ((double*)pln->h_data)[indy*pln->zStride + indx*noStrHarms + hIdx];

	    fprintf(f2,"\t%.17e", val);
	  }
	}
	fprintf(f2,"\n");
      }
    }

    fflush(f2);
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP("Write CVS");
  }

  return err;
}

/**  Plot a ff plane using csv and python script  .
 *
 * NB: This assumes the plane has already been created
 *
 * @param pln
 * @param dir	Directory to place in figure in
 * @param name	File name excluding extension
 * @return	ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE ffdotPln_plotPln( cuRzHarmPlane* pln, const char* dir, const char* name,  const char* prams )
{
  infoMSG(4,4,"Plot ffdot plane section.\n");

  ACC_ERR_CODE	err		= ACC_ERR_NONE;
  char tName[1024];
  sprintf(tName,"%s/%s.csv", dir, name);
  FILE *f2 = fopen(tName, "w");

  FOLD // Write CSV  .
  {
    err += ffdotPln_writePlnToFile(pln, f2);
    fclose(f2);
  }

  if ( !err ) // Make image  .
  {
    infoMSG(5,5,"Image %s\n", tName);

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Image");
    }

    if ( prams )
    {
      sprintf(tName, "%s %s", tName, prams);
    }

    char cmd[1024];
    sprintf(cmd,"python $PRESTO/python/plt_ffd.py %s > /dev/null 2>&1", tName);
    infoMSG(6,6,"%s", cmd);
    int ret = system(cmd);
    if ( ret )
    {
      fprintf(stderr,"ERROR: Problem running potting python script.");
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP("Image");
    }
  }

  return err;
}

ACC_ERR_CODE addPlnToTree(candTree* tree, cuRzHarmPlane* pln)
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  PROF // Profiling  .
  {
    NV_RANGE_PUSH("addPlnToTree");
  }

  FOLD // Get new max  .
  {
    int ggr = 0;

    for (int indy = 0; indy < pln->noZ; indy++ )
    {
      for (int indx = 0; indx < pln->noR ; indx++ )
      {
	double yy2 = 0;
	int noStrHarms;
	if      ( pln->type == CU_STR_HARMONICS )
	  noStrHarms = pln->noHarms;
	else if ( pln->type == CU_STR_INCOHERENT_SUM )
	  noStrHarms = 1;
	else
	{
	  infoMSG(6,6,"ERROR: Plane type has not been initialised.\n" );
	  err += ACC_ERR_UNINIT;
	  break;
	}

	for ( int hIdx = 0; hIdx < noStrHarms; hIdx++)
	{
	  if      ( pln->type == CU_CMPLXF )
	    yy2 +=  POWERF(((float2*)pln->h_data)[indy*pln->zStride + indx*noStrHarms + hIdx]);
	  else if ( pln->type == CU_CMPLXD )
	    yy2 +=  POWERF(((double2*)pln->h_data)[indy*pln->zStride + indx*noStrHarms + hIdx]);
	  else if ( pln->type == CU_FLOAT )
	    yy2 +=  ((float*)pln->h_data)[indy*pln->zStride + indx*noStrHarms + hIdx];
	  else if ( pln->type == CU_DOUBLE )
	    yy2 +=  ((double*)pln->h_data)[indy*pln->zStride + indx*noStrHarms + hIdx];
	  else
	  {
	    infoMSG(6,6,"ERROR: Plane type has not been initialised.\n" );
	    err += ACC_ERR_DATA_TYPE;
	    break;
	  }
	}

	FOLD // Create candidate and add to tree
	{
	  initCand* canidate = new initCand;

	  canidate->numharm = pln->noHarms;
	  canidate->power   = yy2;
	  canidate->r       = pln->centR - pln->rSize/2.0 + indx/(double)(pln->noR-1) * (pln->rSize) ;
	  canidate->z       = pln->centZ + pln->zSize/2.0 - indy/(double)(pln->noZ-1) * (pln->zSize) ;
	  canidate->sig     = yy2;
	  if ( pln->noZ == 1 )
	    canidate->z     = pln->centZ;
	  if ( pln->noR == 1 )
	    canidate->r     = pln->centR;

	  ggr++;

	  tree->insert(canidate, 0.2 );
	}
      }
    }
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP("addPlnToTree");
  }

  return err;
}

/** Initialise a plane, allocating matched host and device memory for the plane
 *
 * @param memSize
 * @return		ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
cuRzHarmPlane* initPln( size_t memSize )
{
  size_t freeMem, totalMem;

  infoMSG(4,4,"Creating new harmonic plane\n");

  cuRzHarmPlane* pln	= (cuRzHarmPlane*)malloc(sizeof(cuRzHarmPlane));
  memset(pln, 0, sizeof(cuRzHarmPlane));

  CUDA_SAFE_CALL(cudaMemGetInfo ( &freeMem, &totalMem ), "Getting Device memory information");

  if ( memSize > freeMem )
  {
    printf("Not enough GPU memory to create any more stacks.\n");
    return NULL;
  }
  else
  {
    infoMSG(6,6,"Memory size %.2f MB (Paired).\n", memSize*1e-6 );

    // Allocate device memory
    CUDA_SAFE_CALL(cudaMalloc(&pln->d_data, memSize), "Failed to allocate device memory for kernel stack.");

    // Allocate host memory
    CUDA_SAFE_CALL(cudaMallocHost(&pln->h_data, memSize), "Failed to allocate device memory for kernel stack.");

    pln->size = memSize;

    // Set default data type (complex values for all harmonics ie. most information possible)
    pln->type += CU_CMPLXF;
    pln->type += CU_STR_HARMONICS;
  }

  return pln;
}

/** Initialise a plane, allocating matched host and device memory for the plane
 *
 * @param memSize
 * @return		ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
cuRzHarmPlane* dupPln( cuRzHarmPlane* orrpln )
{
  size_t freeMem, totalMem;
  cuRzHarmPlane* pln = NULL;

  infoMSG(4,4,"Duplicating harmonic plane\n");

  CUDA_SAFE_CALL(cudaMemGetInfo ( &freeMem, &totalMem ), "Getting Device memory information");

  if ( orrpln->size > freeMem )
  {
    printf("Not enough GPU memory to create any more stacks.\n");
    return NULL;
  }
  else
  {
    pln	= (cuRzHarmPlane*)malloc(sizeof(cuRzHarmPlane));
    memcpy(pln, orrpln, sizeof(cuRzHarmPlane));

    infoMSG(6,6,"Memory size %.2f MB (Paired).\n", orrpln->size*1e-6 );

    // Allocate device memory
    CUDA_SAFE_CALL(cudaMalloc(&pln->d_data, orrpln->size), "Failed to allocate device memory for kernel stack.");

    // Allocate host memory
    CUDA_SAFE_CALL(cudaMallocHost(&pln->h_data, orrpln->size), "Failed to allocate device memory for kernel stack.");
  }

  return pln;
}

/** Free all memory related to a cuRzHarmPlane
 *
 * @param pln	The pointer of the plane to free
 * @return	ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE freePln(cuRzHarmPlane* pln)
{
  infoMSG(3,3,"Free plane data structure.\n");

  if ( pln )
  {
    FOLD // Free pinned memory
    {
      infoMSG(4,4,"Free pinned memory\n");

      cudaFreeHostNull(pln->h_data);
    }

    FOLD // Free device memory
    {
      infoMSG(4,4,"Free device memory\n");

      // Using separate output so free both
      cudaFreeNull(pln->d_data);
    }

    freeNull(pln);
  }
  else
  {
    return ACC_ERR_NULL;
  }

  return ACC_ERR_NONE;
}

cuPlnGen* initPlnGen(int maxHarms, float zMax, confSpecsOpt* conf, gpuInf* gInf)
{
  infoMSG(3,3,"Initialise a GPU rz plane generator.\n");

  cuPlnGen* plnGen = (cuPlnGen*)malloc(sizeof(cuPlnGen));
  memset(plnGen, 0, sizeof(cuPlnGen));

  FOLD // Get the GPU info  .
  {
    if ( conf == NULL )
    {
      infoMSG(4,4,"No configuration specified getting default configuration.\n");
      confSpecs* confAll = getConfig();
      conf = confAll->opt;
    }
    if ( gInf == NULL )
    {
      infoMSG(4,4,"No GPU specified.\n");
      gInf = getGPU(NULL);
    }

    if (!gInf)
    {
      infoMSG(4,4,"ERROR: invalid GPU.\n");
      return NULL;
    }
  }

  plnGen->conf		= conf;					// Should this rather be a duplicate?
  plnGen->flags		= conf->flags;				// Individual flags allows separate configuration
  plnGen->gInf		= gInf;
  plnGen->accu		= HIGHACC;				// Default to high accuracy

  infoMSG(5,5,"Set device %i\n", plnGen->gInf->devid);
  setDevice(plnGen->gInf->devid);

  FOLD // Create streams  .
  {
    infoMSG(5,5,"Create streams.\n");

    CUDA_SAFE_CALL(cudaStreamCreate(&plnGen->stream),"Creating stream for candidate optimisation.");
  }

  FOLD // Create events  .
  {
    infoMSG(5,5,"Create Events.\n");

    if ( plnGen->flags & FLAG_PROF )
    {
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->inpInit),		"Creating input event inpInit." );
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->inpCmp),		"Creating input event inpCmp."  );
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->compInit),	"Creating input event compInit.");
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->compCmp),		"Creating input event compCmp." );
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->outInit),		"Creating input event outInit." );
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->outCmp),		"Creating input event outCmp."  );
    }
    else
    {
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->inpInit,	cudaEventDisableTiming),	"Creating input event inpInit." );
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->inpCmp,	cudaEventDisableTiming),	"Creating input event inpCmp."  );
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->compInit,	cudaEventDisableTiming),	"Creating input event compInit.");
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->compCmp,	cudaEventDisableTiming),	"Creating input event compCmp." );
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->outInit,	cudaEventDisableTiming),	"Creating input event outInit." );
      CUDA_SAFE_CALL(cudaEventCreate(&plnGen->outCmp,	cudaEventDisableTiming),	"Creating input event outCmp."  );
    }
  }

  FOLD // Allocate device memory  .
  {
    infoMSG(5,5,"Allocate device memory.\n");

    int		maxDim		= 1;					///< The max plane width in points
    int		maxWidth	= 1;					///< The max width (area) the plane can cover for all harmonics
    float	zMaxMax		= 1;					///< Max Z-Max this plane should be able to handle

    int		maxNoR		= 1;					///<
    int		maxNoZ		= 1;					///<

    // Number of harmonics to check, I think this could go up to 32!

    FOLD // Determine max plane size  .
    {
      for ( int i=0; i < MAX_NO_STAGES; i++ )
      {
	MAXX(maxWidth, (conf->optPlnSiz[i]) );
      }

      for ( int i=0; i < NO_OPT_LEVS; i++ )
      {
	MAXX(maxDim, conf->optPlnDim[i]);
      }

      FOLD // Determine the largest zMaxMax  .
      {
	zMaxMax	= MAX(zMax+50, zMax*2);
	zMaxMax	= MAX(zMaxMax, (zMax+maxDim)*maxHarms);				// This should be enough
	zMaxMax	= MAX(zMaxMax, 60 * maxHarms );
	//zMaxMax	= MAX(zMaxMax, sSrch->sSpec->zMax * 34 + 50 );  	// TODO: This is 34th harmonic of the fundamental plane. This may be a bit high!
      }

      maxNoR		= maxDim*1.5;					// The maximum number of r points, in the generated plane. The extra is to cater for block kernels which can auto increase
      maxNoZ 		= maxDim;					// The maximum number of z points, in the generated plane
    }

    // Allocate input memory
    plnGen->input	= initHarmInput(maxWidth*10, zMaxMax, maxHarms, plnGen->gInf);

    FOLD // Create plane and set its settings  .
    {
      size_t plnSz	= (maxNoR * maxNoZ * maxHarms ) * sizeof(float2);	// This allows the possibility of returning complex value for the base plane
      plnGen->pln	= initPln( plnSz );
    }
  }

  return plnGen;
}

ACC_ERR_CODE freePlnGen(cuPlnGen* plnGen)
{
  ACC_ERR_CODE err	= ACC_ERR_NONE;

  err += freePln(plnGen->pln);

  err += freeHarmInput(plnGen->input);

  return err;
}

ACC_ERR_CODE snapPlane(cuRzHarmPlane* pln)
{
  return centerPlane(pln, pln->centR, pln->centZ, true );
}

ACC_ERR_CODE centerPlane(cuRzHarmPlane* pln, double r, double z, bool snap )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  if ( pln->rSize && pln->zSize && pln->noR && pln->noZ )
  {
    double rRes		= pln->rSize / (double)(pln->noR-1);
    double zRes		= pln->zSize / (double)(pln->noZ-1);

    if (pln->noR == 1 )
      rRes		= pln->rSize;
    if (pln->noZ == 1 )
      zRes		= pln->zSize;

    if ( snap && ( pln->noR % 2 )  )
    {
      // Odd
      pln->centR	= r;
    }
    else
    {
      // Even
      pln->centR	= r + rRes/2.0;
    }

    if ( snap && ( pln->noZ % 2 )  )
    {
      // Odd
      pln->centZ	= z;
    }
    else
    {
      // Even
      pln->centZ	= z - zRes/2.0;
    }
  }
  else
  {
    infoMSG(6,6,"ERROR: Plane location parameters have not been initialised.\n" );
    err += ACC_ERR_UNINIT;
  }

  return err;
}

ACC_ERR_CODE centerPlaneOnCand(cuRzHarmPlane* pln, initCand* cand, bool snap)
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  if ( pln && cand )
  {
    pln->noHarms	= cand->numharm ;
    err += centerPlane(pln, cand->r, cand->z, snap);
  }
  else
  {
    infoMSG(6,6,"ERROR: NULL pointer centring plane.\n" );
    err += ACC_ERR_NULL;
  }

  return err;
}

template ACC_ERR_CODE ffdotPln_ker<float >( cuPlnGen* plnGen );
template ACC_ERR_CODE ffdotPln_ker<double>( cuPlnGen* plnGen );

template ACC_ERR_CODE ffdotPln<float >( cuPlnGen* plnGen, fftInfo* fft, int* newInput );
template ACC_ERR_CODE ffdotPln<double>( cuPlnGen* plnGen, fftInfo* fft, int* newInput );
