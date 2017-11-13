/** @file cuda_cand_OPT.cu
 *  @brief Utility functions and kernels for GPU optimisation
 *
 *
 *  @author Chris Laidler
 *  @bug No known bugs.
 *
 *  Change Log
 *
 *  [0.0.01] []
 *    Beginning of change log
 *    Working version un-numbed
 *
 *  [0.0.02] [2017-02-16]
 *    Separated candidate and optimisation CPU threading
 *
 *  [2017-06-20]
 *    Made HW a parameter in NM optimisation and power calculations.
 *    Made the maximum repetitions of the NM refinement a configurable parameter
 */

#include <curand.h>
#include <math.h>		// log
#include <curand_kernel.h>
#include <stdlib.h>
#include <stdint.h>		// uint64_t

#include "cuda_math_ext.h"
#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"
#include "candTree.h"
#include "cuda_response.h"
#include "cuda_cand_OPT.h"
#include "cuda_accel_PLN.h"


#define SCALE_AUT       (1000000000)

extern "C"
{
#define __float128 long double
#include "accel.h"
}



#define		NM_BEST		0
#define		NM_MIDL		1
#define		NM_WRST		2


#define SWAP_PTR(p1, p2) do { initCand* tmp = p1; p1 = p2; p2 = tmp; } while (0)

#ifdef CBL // DBG - Thesis output
Logger nmLog;
#endif

template<typename T>
T pow(double r, double z, int numharm, cuHarmInput* inp, int hw = HIGHACC)
{
  int halfW;

  T total_power  = 0;
  T real = 0;
  T imag = 0;

  for( int hIdx = 1; hIdx <= numharm; hIdx++ )
  {
    // Determine half width
    halfW = getHw<T>(z*hIdx, hw);

    rz_convolution_cu<T, float2>(&((float2*)inp->h_inp)[(hIdx-1)*inp->stride], inp->loR[hIdx-1], inp->stride, r*hIdx, z*hIdx, halfW, &real, &imag);

    total_power     += POWERCU(real, imag);
  }

  return total_power;
}

template<typename T>
T pow(initCand* cand, cuHarmInput* inp, int hw = HIGHACC)
{
  double total_power = pow<T>(cand->r, cand->z, cand->numharm, inp, hw);

  cand->power =  total_power;

  return total_power;
}

template<typename T>
T pow(accelcand* cand, cuHarmInput* inp, int hw = HIGHACC)
{
  double total_power = pow<T>(cand->r, cand->z, cand->numharm, inp, hw);

  cand->power =  total_power;

  return total_power;
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
ACC_ERR_CODE chkInput_cand( initCand* cand, cuHarmInput* input, fftInfo* fft, double rSize, double zSize, int* newInp )
{
  return  chkInput(input, cand->r, cand->z, rSize, zSize, cand->numharm, newInp);
}

/** Copy relevant input from FFT to data structure normalising as needed
 *
 *  Note this contains a blocking synchronisation to make sure the pinned host memory is free
 *
 * @param pln     The plane to check
 * @param fft     The FFT data that will make up the input
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE prepInput_cand( initCand* cand, cuHarmInput* input, fftInfo* fft, double rSize, double zSize, int64_t flags )
{
  return loadHostHarmInput(input, fft, cand->r, cand->z, rSize, zSize, cand->numharm, flags, NULL );
}

/** Copy relevant input from FFT to data structure normalising as needed
 *
 *  Note this contains a blocking synchronisation to make sure the pinned host memory is free
 *
 * @param pln     The plane to check
 * @param fft     The FFT data that will make up the input
 * @return        ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE prepInput_cand( accelcand* cand, cuHarmInput* input, fftInfo* fft, double rSize, double zSize, int64_t flags )
{
  return loadHostHarmInput(input, fft, cand->r, cand->z, rSize, zSize, cand->numharm, flags, NULL );
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
ACC_ERR_CODE prepInput_cand( initCand* cand, cuHarmInput* input, fftInfo* fft, double rSize, double zSize, int* newInp, int64_t flags )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  // Check input
  int newInp_l;
  err += chkInput_cand( cand, input, fft, rSize, zSize, &newInp_l );

  if ( newInp_l )
  {
    // load normalised data into host memory
    err += prepInput_cand( cand, input, fft, rSize, zSize, flags );
  }

  if ( newInp )
    *newInp = newInp_l;

  return err;
}

candTree* opt_cont(candTree* oTree, cuPlnGen* pln, container* cont, fftInfo* fft, int nn)
{
  //  PROF // Profiling  .
  //  {
  //    NV_RANGE_PUSH("opt_cont");
  //  }
  //
  //  confSpecsGen*  sSpec   = pln->cuSrch->sSpec;
  //  initCand* iCand 	= (initCand*)cont->data;

  //
  //  optInitCandLocPlns(iCand, pln, nn );
  //
  //  accelcand* cand = new accelcand;
  //  memset(cand, 0, sizeof(accelcand));
  //
  //  int lrep      = 0;
  //  int noP       = 30;
  //  float snoop   = 0.3;
  //  float sz;
  //  float v1, v2;
  //
  //  const int mxRep = 10;
  //
  //  initCand* canidate = (initCand*)cont->data;
  //
  //  candTree* thisOpt = new candTree;
  //
  //  if ( canidate->numharm == 1  )
  //    sz = conf->optPlnSiz[0];
  //  if ( canidate->numharm == 2  )
  //    sz = conf->optPlnSiz[1];
  //  if ( canidate->numharm == 4  )
  //    sz = conf->optPlnSiz[2];
  //  if ( canidate->numharm == 8  )
  //    sz = conf->optPlnSiz[3];
  //  if ( canidate->numharm == 16 )
  //    sz = conf->optPlnSiz[4];
  //
  //  //int numindep        = (obs->rhi - obs->rlo ) * (obs->zhi +1 ) * (ACCEL_DZ / 6.95) / pln->noHarms ;
  //
  //  //printf("\n%03i  r: %15.6f   z: %12.6f \n", nn, cand->r, cand->z);
  //
  //  pln->halfWidth = 0;
  //
  //  int plt = 0;
  //
  //  if ( optpln01 > 0 )
  //  {
  //    noP               = optpln01 ;
  //    lrep              = 0;
  //    canidate->power   = 0;     // Set initial power to zero
  //    do
  //    {
  //      generatePln<float>(canidate, fft, pln, noP, sz, plt++, nn );
  //
  //      container* optC =  oTree->getLargest(canidate, 1);
  //
  //      if ( optC )
  //      {
  //        // This has feature has already been optimised!
  //        cont->flag |= REMOVE_CONTAINER;
  //        NV_RANGE_POP();
  //        return thisOpt;
  //      }
  //
  //      //addPlnToTree(thisOpt, pln);
  //
  //      v1 = fabs(( pln->centR - canidate->r )/(pln->rSize/2.0));
  //      v2 = fabs(( pln->centZ - canidate->z )/(pln->zSize/2.0));
  //
  //      if ( ++lrep > mxRep )
  //      {
  //        break;
  //      }
  //    }
  //    while ( v1 > snoop || v2 > snoop );
  //    sz /= downScale;
  //  }
  //
  //  if ( optpln02 > 0 )
  //  {
  //    noP               = optpln02 ;
  //    lrep              = 0;
  //    canidate->power   = 0;     // Set initial power to zero
  //    do
  //    {
  //      generatePln<float>(canidate, fft, pln, noP, sz, plt++, nn );
  //
  //      container* optC =  oTree->getLargest(canidate, 1);
  //
  //      if ( optC )
  //      {
  //        // This has feature has already been optimised!
  //        cont->flag |= REMOVE_CONTAINER;
  //        NV_RANGE_POP();
  //        return thisOpt;
  //      }
  //
  //      //addPlnToTree(thisOpt, pln);
  //
  //      v1 = fabs(( pln->centR - canidate->r )/(pln->rSize/2.0));
  //      v2 = fabs(( pln->centZ - canidate->z )/(pln->zSize/2.0));
  //
  //      if ( ++lrep > mxRep )
  //      {
  //        break;
  //      }
  //    }
  //    while ( v1 > snoop || v2 > snoop );
  //    sz /= downScale;
  //  }
  //
  //  if ( optpln03 > 0 )
  //  {
  //    noP               = optpln03 ;
  //    lrep              = 0;
  //    canidate->power   = 0;     // Set initial power to zero
  //    do
  //    {
  //      generatePln<float>(canidate, fft, pln, noP, sz, plt++, nn );
  //
  //      container* optC =  oTree->getLargest(canidate, 1);
  //
  //      if ( optC )
  //      {
  //        // This has feature has already been optimised!
  //        cont->flag |= REMOVE_CONTAINER;
  //        NV_RANGE_POP();
  //        return thisOpt;
  //      }
  //
  //      //addPlnToTree(thisOpt, pln);
  //
  //      v1 = fabs(( pln->centR - canidate->r )/(pln->rSize/2.0));
  //      v2 = fabs(( pln->centZ - canidate->z )/(pln->zSize/2.0));
  //
  //      if ( ++lrep > mxRep )
  //      {
  //        break;
  //      }
  //    }
  //    while ( v1 > snoop || v2 > snoop );
  //    sz /= downScale*2;
  //  }
  //
  //  if ( optpln04 > 0 )
  //  {
  //    noP               = optpln04 ;
  //    lrep              = 0;
  //    canidate->power   = 0;     // Set initial power to zero
  //    do
  //    {
  //      generatePln<float>(canidate, fft, pln, noP, sz, plt++, nn );
  //
  //      container* optC =  oTree->getLargest(canidate, 1);
  //
  //      if ( optC )
  //      {
  //        // This has feature has already been optimised!
  //        cont->flag |= REMOVE_CONTAINER;
  //        NV_RANGE_POP();
  //        return thisOpt;
  //      }
  //
  //      //addPlnToTree(thisOpt, pln);
  //
  //      v1 = fabs(( pln->centR - canidate->r )/(pln->rSize/2.0));
  //      v2 = fabs(( pln->centZ - canidate->z )/(pln->zSize/2.0));
  //
  //      if ( ++lrep > mxRep )
  //      {
  //        break;
  //      }
  //    }
  //    while ( v1 > snoop || v2 > snoop );
  //    sz /= downScale*2;
  //  }
  //
  //  if ( optpln05 > 0 )
  //  {
  //    noP               = optpln05 ;
  //    lrep              = 0;
  //    canidate->power   = 0;     // Set initial power to zero
  //    do
  //    {
  //      generatePln<float>(canidate, fft, pln, noP, sz, plt++, nn );
  //
  //      container* optC =  oTree->getLargest(canidate, 1);
  //
  //      if ( optC )
  //      {
  //        // This has feature has already been optimised!
  //        cont->flag |= REMOVE_CONTAINER;
  //        NV_RANGE_POP();
  //        return thisOpt;
  //      }
  //
  //      //addPlnToTree(thisOpt, pln);
  //
  //      v1 = fabs(( pln->centR - canidate->r )/(pln->rSize/2.0));
  //      v2 = fabs(( pln->centZ - canidate->z )/(pln->zSize/2.0));
  //
  //      if ( ++lrep > mxRep )
  //      {
  //        break;
  //      }
  //    }
  //    while ( v1 > snoop || v2 > snoop );
  //    sz /= downScale*2;
  //  }
  //
  //  if ( optpln06 > 0 )
  //  {
  //    noP               = optpln06 ;
  //    lrep              = 0;
  //    canidate->power   = 0;     // Set initial power to zero
  //    do
  //    {
  //      generatePln<double>(canidate, fft, pln, noP, sz, plt++, nn );
  //
  //      container* optC =  oTree->getLargest(canidate, 1);
  //
  //      if ( optC )
  //      {
  //        // This has feature has already been optimised!
  //        cont->flag |= REMOVE_CONTAINER;
  //        NV_RANGE_POP();
  //        return thisOpt;
  //      }
  //
  //      //addPlnToTree(thisOpt, pln);
  //
  //      v1 = fabs(( pln->centR - canidate->r )/(pln->rSize/2.0));
  //      v2 = fabs(( pln->centZ - canidate->z )/(pln->zSize/2.0));
  //
  //      if ( ++lrep > mxRep )
  //      {
  //        break;
  //      }
  //    }
  //    while ( v1 > snoop || v2 > snoop );
  //    sz /= downScale*2;
  //  }
  //
  //  cont->flag |= OPTIMISED_CONTAINER;
  //
  //  NV_RANGE_POP();
  //  return thisOpt;
  return NULL;
}

ACC_ERR_CODE pln_max_pnt( cuRzHarmPlane* pln, initCand* cand )
{
  ACC_ERR_CODE err = ACC_ERR_NONE;

  PROF // Profiling  .
  {
    NV_RANGE_PUSH("Get Max");
  }

  int noStrHarms = 0;
  if      ( pln->type == CU_STR_HARMONICS )
    noStrHarms = pln->noHarms;
  else if ( pln->type == CU_STR_INCOHERENT_SUM )
    noStrHarms = 1;
  else
  {
    infoMSG(6,6,"Plane type has not been initialised.\n" );
    err += ACC_ERR_UNINIT;
  }

  pln->maxBound = 0;

  for (int indy = 0; indy < pln->noZ; indy++ )
  {
    for (int indx = 0; indx < pln->noR ; indx++ )
    {
      double yy2 = 0;

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

      FOLD // Track max and min of plane for
      {
	if ( indy == 0 && indx == 0 )
	{
	  pln->maxPower = yy2;
	  pln->minPower = yy2;
	}
	MINN(pln->minPower, yy2);
	MAXX(pln->maxPower, yy2);
      }

      if ( (indx == 0 || indx == pln->noR-1 || indy == 0 || indy == pln->noZ-1 ) && ( yy2 > pln->maxBound) )
      {
	pln->maxBound = yy2 ;
      }

      if ( yy2 > cand->power )
      {
	cand->power	= yy2;
	cand->r	= pln->centR - pln->rSize/2.0 + indx/(double)(pln->noR-1) * (pln->rSize) ;
	cand->z	= pln->centZ + pln->zSize/2.0 - indy/(double)(pln->noZ-1) * (pln->zSize) ;
	if ( pln->noZ	== 1 )
	  cand->z = pln->centZ;
	if ( pln->noR	== 1 )
	  cand->r = pln->centR;
      }
    }
  }

  infoMSG(4,4,"Max Power %8.5f at (%.6f %.6f) - Min power: %8.5f  Range: %4e \n", cand->power, cand->r, cand->z, pln->minPower, pln->maxPower-pln->minPower);

  PROF // Profiling  .
  {
    NV_RANGE_POP("Get Max");
  }

  return err;
}

ACC_ERR_CODE pln_max_wAve( cuRzHarmPlane* pln, initCand* cand, double bound )
{
  ACC_ERR_CODE err = ACC_ERR_NONE;

  PROF // Profiling  .
  {
    NV_RANGE_PUSH("Get Max");
  }

  int noStrHarms = 0;
  if      ( pln->type == CU_STR_HARMONICS )
    noStrHarms = pln->noHarms;
  else if ( pln->type == CU_STR_INCOHERENT_SUM )
    noStrHarms = 1;
  else
  {
    infoMSG(6,6,"Plane type has not been initialised.\n" );
    err += ACC_ERR_UNINIT;
  }

  double zSum = 0;
  double rSum = 0;

  double weight = 0;

  for (int indy = 0; indy < pln->noZ; indy++ )
  {
    double z	= pln->centZ + pln->zSize/2.0 - indy/(double)(pln->noZ-1) * (pln->zSize) ;

    for (int indx = 0; indx < pln->noR ; indx++ )
    {
      double r	= pln->centR - pln->rSize/2.0 + indx/(double)(pln->noR-1) * (pln->rSize) ;
      double yy2 = 0;

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

      if ( yy2 >= bound )
      {
	weight += yy2;

	zSum += z*yy2;
	rSum += r*yy2;
      }
    }
  }

  if ( weight > 0 )
  {
    cand->z = zSum/weight;
    cand->r = rSum/weight;
  }

  infoMSG(4,4,"Max Power %8.5f at (%.6f %.6f) - Min power: %8.5f  Range: %4e \n", cand->power, cand->r, cand->z, pln->minPower, pln->maxPower-pln->minPower);

  PROF // Profiling  .
  {
    NV_RANGE_POP("Get Max");
  }

  return err;
}

/** Refine candidate location using repetitive planes  .
 *
 * @param cand
 * @param opt
 * @param noP
 * @param scale
 * @param plt
 * @param nn
 * @param lv
 * @return
 */
template<typename T>
ACC_ERR_CODE optRefinePosPln(initCand* cand, cuOpt* opt, int noP, double scale, int plt = -1, int nn = 0, int lv = 0 )
{
  ACC_ERR_CODE err = ACC_ERR_NONE;
  int newInput = 0;

  fftInfo*	fft	= opt->cuSrch->fft;
  confSpecsOpt*	conf	= opt->conf;
  cuRzHarmPlane* pln	= opt->plnGen->pln;

  // Number of harmonics to check, I think this could go up to 32!
  int maxHarms	= MAX(cand->numharm, conf->optMinLocHarms);

  FOLD // Generate plain points  .
  {
    pln->noZ		= noP;
    pln->noR		= noP;
    pln->rSize		= scale;
    pln->zSize		= scale*conf->zScale;

    err += centerPlaneOnCand(pln, cand);
    ERROR_MSG(err, "ERROR: Placing ffdot plane.");

    // Override the candidate number of harmonics (this must be done after centring the plane)
    pln->noHarms	= maxHarms;

    err += ffdotPln<T>(opt->plnGen, fft, &newInput);

    if ( newInput ) // Create the section of ff plane  .
    {
      // New input was used so don't maintain the old max, as different normalisation may cause minor differences making the powers incomparable
      cand->power	= 0;
    }
    ERROR_MSG(err, "ERROR: Generating f-fdot plane.");

    PROF // Profiling - Time components  .
    {
      if ( (opt->flags & FLAG_PROF) )
      {
  	infoMSG(5,5,"Time components");

  	// Time batch multiply
  	timeEvents( opt->plnGen->inpInit,  opt->plnGen->inpCmp,  &opt->compTime[COMP_OPT_H2D], "Copy H2D");
  	timeEvents( opt->plnGen->compInit, opt->plnGen->compCmp, &opt->compTime[COMP_OPT_PLN1+lv], "Optimisation plane calculations");
  	timeEvents( opt->plnGen->outInit,  opt->plnGen->outCmp,  &opt->compTime[COMP_OPT_D2H], "Copy D2H");
      }
    }
  }

  double r1, z1, r2, z2;

  FOLD // Get new max  .
  {
    err += pln_max_pnt( pln, cand );

    r1 = cand->r;
    z1 = cand->z;

    if ( scale < 0.7 )
    {
      double bound = pln->minPower + ( pln->maxPower - pln->minPower )*0.9 ;
      MAXX(bound, pln->maxBound);	// Clamp to max edge value to avoid unjustly weighting
      err += pln_max_wAve( pln, cand, bound );
    }

    r2 = cand->r;
    z2 = cand->z;
  }

  FOLD // Write CSV & plot output  .
  {
#ifdef CBL
    if ( conf->flags & FLAG_DPG_PLT_OPT ) // Write CSV & plot output  .
    {
      // TODO: Check if we can get the directory name and then this can be added into standard accelsearch
      char tName[1024];
      sprintf(tName,"Cand_%05i_Rep_%02i_Lv_%i_h%02i", nn, plt, lv, cand->numharm );

      char pName[1024];
      sprintf(pName," --max1 %.17f %.17f --max2 %.17f %.17f ", r1, z1, r2, z2 );

      ffdotPln_plotPln( pln, "/home/chris/accel/", tName, pName );
    }
#endif
  }

  return err;
}

/** Refine candidate location using simplex
 *
 * @param cand
 * @param inp
 * @param rSize
 * @param zSize
 * @param plt
 * @param nn
 * @param lv
 * @return
 */
template<typename T>
ACC_ERR_CODE optInitCandPosSim(initCand* cand, cuHarmInput* inp, double rSize = 1.0, double zSize = 1.0, int hw = HIGHACC, int maxReps = 100, double bound = 1e-7, int nn = 0, int lv = 0, int plt = 0 )
{
  ACC_ERR_CODE err = ACC_ERR_NONE;

  infoMSG(3,3,"Simplex refine position - lvl %i  size %.4e by %.4e \n", lv+1, rSize, zSize);

#ifdef CBL // DBG - Thesis output
  Logger nmPathLog;
  if ( plt)
  {
    char cmpName[1024];
    sprintf(cmpName, "/home/chris/accel/nm_path_%03i.csv", nn );

    nmPathLog.setFile(cmpName);
    nmPathLog.setEcho(false);
    nmPathLog.setCsvLineNums(true);
    nmPathLog.setCsvDeliminator(';');
  }
#endif

  // These are the Nelder–Mead parameter values
  double reflect	= 1.0;
  double expand		= 2.5;
  double contract	= 0.4;
  double shrink		= 0.3;

  initCand  cnds[3];
  initCand* olst[3];

  initCand  centroid    = *cand;
  initCand  reflection  = *cand;
  initCand  expansion   = *cand;
  initCand  contraction = *cand;

  cnds[0] = *cand;
  cnds[1] = *cand;
  cnds[2] = *cand;

  pow<T>(&cnds[0], inp, hw);
  double inpPow = cnds[0].power;

  cnds[1].r += rSize;
  pow<T>(&cnds[1], inp, hw);

  cnds[2].z += zSize;
  pow<T>(&cnds[2], inp, hw);

  olst[NM_BEST] = &cnds[0];
  olst[NM_MIDL] = &cnds[1];
  olst[NM_WRST] = &cnds[2];

  int ite = 0;
  double rtol;			///< Ratio of low to high

  infoMSG(4,4,"Start - Power: %8.3f at (%.6f %.6f)", cnds[0].power, cnds[0].r, cnds[0].z);

  while (1)
  {
    FOLD // Order
    {
      if (olst[NM_WRST]->power > olst[NM_MIDL]->power )
	SWAP_PTR(olst[NM_WRST], olst[NM_MIDL]);

      if (olst[NM_MIDL]->power > olst[NM_BEST]->power )
      {
	SWAP_PTR(olst[NM_MIDL], olst[NM_BEST]);

	if (olst[NM_WRST]->power > olst[NM_MIDL]->power )
	  SWAP_PTR(olst[NM_WRST], olst[NM_MIDL]);
      }
    }

    FOLD // Centroid  .
    {
      centroid.r = ( olst[NM_BEST]->r + olst[NM_MIDL]->r ) / 2.0  ;
      centroid.z = ( olst[NM_BEST]->z + olst[NM_MIDL]->z ) / 2.0  ;
      // Don't calculate the power as it's not used
    }

    ite++;

    rtol = 2.0 * fabs(olst[NM_BEST]->power - olst[NM_WRST]->power) / (fabs(olst[NM_BEST]->power) + fabs(olst[NM_MIDL]->power) + 1.0e-15) ;

#ifdef CBL
    if ( plt)
    {
      nmPathLog.csvWrite("r0",      "%18.15e", olst[NM_BEST]->r);
      nmPathLog.csvWrite("z0",      "%18.15e", olst[NM_BEST]->z);
      nmPathLog.csvWrite("power0",  "%18.15e", olst[NM_BEST]->power);

      nmPathLog.csvWrite("r1",      "%18.15e", olst[NM_MIDL]->r);
      nmPathLog.csvWrite("z1",      "%18.15e", olst[NM_MIDL]->z);
      nmPathLog.csvWrite("power1",  "%18.15e", olst[NM_MIDL]->power);

      nmPathLog.csvWrite("r2",      "%18.15e", olst[NM_WRST]->r);
      nmPathLog.csvWrite("z2",      "%18.15e", olst[NM_WRST]->z);
      nmPathLog.csvWrite("power2",  "%18.15e", olst[NM_WRST]->power);

      nmPathLog.csvWrite("rtol",   "%18.15e", rtol);

      nmPathLog.csvEndLine();
    }
#endif

    if (rtol < bound )  // Within error so leave  .
    {
      break;
    }

    if ( ite >= maxReps )
    {
      break;
    }

    FOLD // Reflection  .
    {
      reflection.r = centroid.r + reflect*(centroid.r - olst[NM_WRST]->r ) ;
      reflection.z = centroid.z + reflect*(centroid.z - olst[NM_WRST]->z ) ;
      pow<T>(&reflection, inp, hw);

      if ( olst[NM_BEST]->power <= reflection.power && reflection.power < olst[NM_MIDL]->power )
      {
	*olst[NM_WRST] = reflection;
	continue;
      }
    }

    FOLD // Expansion  .
    {
      if ( reflection.power > olst[NM_BEST]->power )
      {
	expansion.r = centroid.r + expand*(reflection.r - centroid.r ) ;
	expansion.z = centroid.z + expand*(reflection.z - centroid.z ) ;
	pow<T>(&expansion, inp, hw);

	if (expansion.power > reflection.power)
	{
	  *olst[NM_WRST] = expansion;
	}
	else
	{
	  *olst[NM_WRST] = reflection;
	}
	continue;
      }
    }

    FOLD // Contraction  .
    {
      contraction.r = centroid.r + contract*(olst[NM_WRST]->r - centroid.r) ;
      contraction.z = centroid.z + contract*(olst[NM_WRST]->z - centroid.z) ;
      pow<T>(&contraction, inp, hw);

      if ( contraction.power > olst[NM_WRST]->power )
      {
	*olst[NM_WRST] = contraction;
	continue;
      }
    }

    FOLD // Shrink  .
    {
      olst[NM_MIDL]->r = olst[NM_BEST]->r + shrink*(olst[NM_MIDL]->r - olst[NM_BEST]->r);
      olst[NM_MIDL]->z = olst[NM_BEST]->z + shrink*(olst[NM_MIDL]->z - olst[NM_BEST]->z);
      pow<T>(olst[NM_MIDL], inp, hw);

      olst[NM_WRST]->r = olst[NM_BEST]->r + shrink*(olst[NM_WRST]->r - olst[NM_BEST]->r);
      olst[NM_WRST]->z = olst[NM_BEST]->z + shrink*(olst[NM_WRST]->z - olst[NM_BEST]->z);
      pow<T>(olst[NM_WRST], inp, hw);
    }
  }

  double dist = sqrt( (cand->r-olst[NM_BEST]->r)*(cand->r-olst[NM_BEST]->r) + (cand->z-olst[NM_BEST]->z)*(cand->z-olst[NM_BEST]->z) );
  double powInc  = olst[NM_BEST]->power - inpPow;

  cand->r = olst[NM_BEST]->r;
  cand->z = olst[NM_BEST]->z;
  cand->power = olst[NM_BEST]->power;

  infoMSG(4,4,"End   - Power: %8.3f at (%.6f %.6f) %3i iterations moved %9.7e  power inc: %9.7e", cand->power, cand->r, cand->z, ite, dist, powInc);

#ifdef CBL
  if ( plt)
  {
    nmLog.csvWrite("ite",   "%i",      ite);
    nmLog.csvWrite("dst",   "%9.7e",   dist);
    nmLog.csvWrite("inc",   "%9.7e",   powInc);
    nmLog.csvWrite("power", "%18.15e", cand->power);
    nmLog.csvEndLine();

    nmPathLog.close();
  }
#endif
  return err;
}

/** Refine candidate location using simplex
 *
 * @param cand
 * @param inp
 * @param rSize
 * @param zSize
 * @param plt
 * @param nn
 * @param lv
 * @return
 */
template<typename T>
ACC_ERR_CODE optInitCandPosSim(accelcand* cand, cuHarmInput* inp, double rSize = 1.0, double zSize = 1.0, int hw = HIGHACC, int maxReps = 100, double bound = 1e-7, int nn = 0, int lv = 0, int plt = 0 )
{
  ACC_ERR_CODE err = ACC_ERR_NONE;

  initCand tCand;
  tCand.numharm	= cand->numharm;
  tCand.r	= cand->r;
  tCand.z	= cand->z;
  //tCand.sig	= cand->numharm;

  err += optInitCandPosSim<T>(&tCand, inp, rSize, zSize, hw, maxReps, bound, nn, lv, plt );

  cand->numharm	= tCand.numharm;
  cand->r	= tCand.r;
  cand->z	= tCand.z;
  cand->power	= tCand.power;

  return err;
}

/** Initiate a optimisation plane
 * If oPln has not been pre initialised and is NULL it will create a new data structure.
 * If oPln has been pre initialised the device ID and Idx are used!
 *
 */
cuOpt* initOptimiser(cuSearch* sSrch, cuOpt* opt, gpuInf* gInf )
{
  confSpecsOpt*	conf	= sSrch->conf->opt;

  infoMSG(3,3,"Initialising optimiser.\n");

  int	maxHarms	= MAX(sSrch->noSrchHarms, conf->optMinLocHarms);
  
  if (!opt)
  {
    infoMSG(5,5,"Allocating new optimiser\n");
    opt = new cuOpt;
    memset(opt, 0, sizeof(cuOpt));
  }

  FOLD // Create all sub structures  .
  {
    opt->cuSrch		= sSrch;					// Set the pointer t the search specifications
    opt->conf		= conf;						// Should this rather be a duplicate?
    opt->gInf 		= gInf;

    if      ( conf->flags & FLAG_OPT_NM )
    {
      int zMaxMax = sSrch->sSpec->zMax + 50;
      MAXX(zMaxMax, 50*maxHarms);

      opt->input	= initHarmInput(40, zMaxMax, maxHarms, gInf);
    }
    else if ( conf->flags & FLAG_OPT_SWARM )
    {
      fprintf(stderr,"ERROR: Particle swarm optimisation has been removed.\n");
      exit(EXIT_FAILURE);
    }
    else // Default use planes
    {
      opt->plnGen	= initPlnGen(maxHarms, sSrch->sSpec->zMax, conf, gInf);
      opt->input	= opt->plnGen->input;
    }
  }

  FOLD // Allocate struct specify memory  .
  {
    int sz = sizeof(long long)*(COMP_OPT_MAX) ;
    opt->compTime       = (long long*)malloc(sz);
    memset(opt->compTime,    0, sz);
  }

  return opt;
}

/** Free individual optimiser
 *
 * @param opt	The optimisers
 * @return
 */
ACC_ERR_CODE freeOptimiser(cuOpt* opt)
{
  ACC_ERR_CODE err	= ACC_ERR_NONE;
  confSpecsOpt*	conf	= opt->conf;

  err += freePlnGen(opt->plnGen);

  if ( conf->flags & FLAG_OPT_NM )
    err += freeHarmInput(opt->input);
  else
    opt->input = NULL;

  freeNull(opt->compTime);

  freeNull(opt);

  return err;
}

/** Create multiplication kernel and allocate memory for planes on all devices  .
 *
 * Create the kernels on the first device and then copy it to all others
 *
 * @param sSrch     A pointer to the search structure
 *
 * @return
 */
ACC_ERR_CODE initOptimisers(cuSearch* sSrch )
{
  ACC_ERR_CODE err = ACC_ERR_NONE;

  infoMSG(2,2,"Initialise all optimisers.\n");

  sSrch->oInf = new cuOptInfo;
  memset(sSrch->oInf, 0, sizeof(cuOptInfo));

  CUDA_SAFE_CALL(cudaGetLastError(), "Entering initOptimisers.");

  double halfWidth = cu_z_resp_halfwidth<double>(sSrch->sSpec->zMax+10, HIGHACC)+10;	// Candidate may be on the z-max border so buffer a bit

  FOLD // Create the primary stack on each device, this contains the kernel  .
  {
    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Init Optimisers");
    }

    FOLD // Determine the number of optimisers to make
    {
      sSrch->oInf->noOpts = 0;
      for ( int dev = 0 ; dev < sSrch->gSpec->noDevices; dev++ ) // Loop over devices  .
      {
	if ( sSrch->gSpec->noDevOpt[dev] <= 0 )
	{
	  // Use the default of 4
	  sSrch->gSpec->noDevOpt[dev] = 4;

	  infoMSG(5,5,"Using the default %i optimisers per GPU.\n", sSrch->gSpec->noDevOpt[dev]);
	}
	sSrch->oInf->noOpts += sSrch->gSpec->noDevOpt[dev];
      }
    }

    infoMSG(5,5,"Initialising %i optimisers on %i devices.\n", sSrch->oInf->noOpts, sSrch->gSpec->noDevices);

    // Initialise the individual optimisers
    sSrch->oInf->opts = (cuOpt*)malloc(sSrch->oInf->noOpts*sizeof(cuOpt));
    memset(sSrch->oInf->opts, 0, sSrch->oInf->noOpts*sizeof(cuOpt));

    int idx = 0;
    for ( int dev = 0 ; dev < sSrch->gSpec->noDevices; dev++ ) // Loop over devices  .
    {
      for ( int oo = 0 ; oo < sSrch->gSpec->noDevOpt[dev]; oo++ )
      {
	initOptimiser(sSrch, &sSrch->oInf->opts[idx], &sSrch->gSpec->devInfo[dev] );
	sSrch->oInf->opts[idx].pIdx = idx;

	idx++;
      }
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP("Init Optimisers");
    }
  }
  
  return err;
}

/** Free all the optimisers of a search  .
 *
 * @param sSrch
 * @return
 */
ACC_ERR_CODE freeOptimisers(cuSearch* sSrch )
{
  ACC_ERR_CODE err	= ACC_ERR_NONE;

  infoMSG(4,4,"Freeing all optimisers.\n");

  if ( sSrch->oInf )
  {
    if ( sSrch->oInf->opts )
    {
      int idx = 0;
      for ( int dev = 0 ; dev < sSrch->gSpec->noDevices; dev++ ) // Loop over devices  .
      {
	for ( int oo = 0 ; oo < sSrch->gSpec->noDevOpt[dev]; oo++ )
	{
	  freeOptimiser(&sSrch->oInf->opts[idx] );
	  idx++;
	}
      }

      freeNull(sSrch->oInf->opts);
    }

    freeNull(sSrch->oInf);
  }

  return err;
}

/** Initialise all the optimisers for the entire search
 *
 * @param srch
 * @return
 */
cuSearch* initCuOpt(cuSearch* srch)
{
  PROF // Profiling  .
  {
    NV_RANGE_PUSH("Init CUDA optimisers");
  }

  if ( !srch->oInf )
  {
    initOptimisers( srch );
  }
  else
  {
    // TODO: Do a whole bunch of checks here!
    fprintf(stderr, "ERROR: %s has not been set up to handle a pre-initialised memory info data structure.\n", __FUNCTION__);
    exit(EXIT_FAILURE);
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP("Init CUDA optimisers");
  }

  return srch;
}

/** Optimise derivatives of a candidate  .
 *
 */
void* optCandDerivs(accelcand* cand, cuSearch* srch )
{
  int ii;
  struct timeval start, end;    // Profiling variables

  fftInfo*	fft	= srch->fft;
  confSpecsOpt*	conf	= srch->conf->opt;
  searchSpecs*	sSpec	= srch->sSpec;

  FOLD // Update fundamental values to the optimised ones  .
  {
    infoMSG(5,5,"DERIVS\n");

    float	maxSig		= 0;
    int		bestH		= 0;
    float	bestP		= 0;
    double  	sig		= 0; // can be a float
    long long	numindep;
    float	candHPower	= 0;
    int		noStages	= 0;
    int 	kern_half_width;
    double	locpow;
    double	real;
    double	imag;
    double	power;
    int		maxHarms  	= MAX(cand->numharm, conf->optMinRepHarms) ;

    PROF // Profiling  .
    {
      if ( !(!(conf->flags & FLAG_SYNCH) && (conf->flags & FLAG_OPT_THREAD)) )
      {
	NV_RANGE_PUSH("DERIVS");
      }

      if ( conf->flags & FLAG_PROF )
      {
	gettimeofday(&start, NULL);
      }
    }

    cand->power   = 0;

    // Set up candidate
    cand->pows    = gen_dvect(maxHarms);
    cand->hirs    = gen_dvect(maxHarms);
    cand->hizs    = gen_dvect(maxHarms);
    cand->derivs  = (rderivs *)   malloc(sizeof(rderivs)  * maxHarms  );

    // Initialise values
    for( ii=0; ii < maxHarms; ii++ )
    {
      cand->hirs[ii]  = cand->r*(ii+1);
      cand->hizs[ii]  = cand->z*(ii+1);
    }

    for( ii = 1; ii <= maxHarms; ii++ )			// Calculate derivatives, powers and sigma for all harmonics  .
    {
      if      ( conf->flags & FLAG_OPT_NRM_LOCAVE   )
      {
	locpow = get_localpower3d(fft->data, fft->noBins, cand->r*ii, cand->z*ii, 0.0);
      }
      else if ( conf->flags & FLAG_OPT_NRM_MEDIAN1D )
      {
	locpow = get_scaleFactorZ(fft->data, fft->noBins, cand->r*ii, cand->z*ii, 0.0);
      }
      else if ( conf->flags & FLAG_OPT_NRM_MEDIAN2D )
      {
	fprintf(stderr,"ERROR: 2D median normalisation has not been written yet.\n");
	exit(EXIT_FAILURE);
      }
      else
      {
	// No normalisation this is plausible but not recommended

	// TODO: This should error if it is out of bounds?
	locpow = 1;
      }

      if ( locpow )
      {
	kern_half_width   = cu_z_resp_halfwidth<double>(fabs(cand->z*ii), HIGHACC);

	rz_convolution_cu<double, float2>((float2*)fft->data, fft->firstBin, fft->noBins, cand->r*ii, cand->z*ii, kern_half_width, &real, &imag);

	// Normalised power
	power = POWERCU(real, imag) / locpow ;

	cand->pows[ii-1] = power;

	get_derivs3d(fft->data, fft->noBins, cand->r*ii, cand->z*ii, 0.0, locpow, &cand->derivs[ii-1] );

	cand->power	+= power;
	int numz 	= round(srch->conf->gen->zMax / srch->conf->gen->zRes) * 2 + 1;
	if ( numz == 1 )
	{
	  numindep	= (sSpec->searchRHigh - sSpec->searchRLow) / (double)(ii) ;
	}
	else
	{
	  numindep	= (sSpec->searchRHigh - sSpec->searchRLow) * (numz + 1) * ( srch->conf->gen->zRes / 6.95 ) / (double)(ii);
	}

	sig		= candidate_sigma_cu(cand->power, (ii), numindep );

	infoMSG(6,6,"Harm %2i  local power %6.3f, normalised power: %8.3f,   cumulative sigma: %6.2f  r: %12.2f  z: %8.2f\n", ii, locpow, power, sig, cand->r*ii, cand->z*ii );

	if ( sig > maxSig || ii == 1 )
	{
	  maxSig        = sig;
	  bestP         = cand->power;
	  bestH         = (ii);
	}

	if ( ii == cand->numharm )
	{
	  candHPower    = cand->power;

	  if ( !(conf->flags & FLAG_OPT_BEST) )
	  {
	    break;
	  }
	}
      }
    }

    // Final values
    if ( bestP && (conf->flags & FLAG_OPT_BEST) && ( maxSig > 0.001 ) )
    {
      cand->numharm	= bestH;
      cand->sigma	= maxSig;
      cand->power	= bestP;

      infoMSG(4,4,"Cand best val Sigma: %5.2f Power: %6.4f  %i harmonics summed.", maxSig, bestP, bestH);
    }
    else
    {
      cand->power	= candHPower;
      noStages		= log2((double)cand->numharm);
      numindep		= srch->numindep[noStages];
      cand->sigma	= candidate_sigma_cu(candHPower, cand->numharm, numindep);

      infoMSG(4,4,"Cand harm val Sigma: %5.2f Power: %6.4f  %i harmonics summed.", cand->sigma, cand->power, cand->numharm);
    }

    PROF // Profiling  .
    {
      if ( !(!(conf->flags & FLAG_SYNCH) && (conf->flags & FLAG_OPT_THREAD)) )
      {
	NV_RANGE_POP("DERIVS");
      }

      if ( conf->flags & FLAG_PROF )
      {
	gettimeofday(&end, NULL);
	float v1 =  (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);

	// Thread (pthread) safe add to timing value
	pthread_mutex_lock(&srch->threasdInfo->candAdd_mutex);
	srch->timings[COMP_OPT_DERIVS] += v1;
	pthread_mutex_unlock(&srch->threasdInfo->candAdd_mutex);
      }
    }
  }

  return (NULL);
}

/** CPU process results
 *
 * This function is meant to be the entry of a separate thread
 *
 */
void* cpuProcess(void* ptr)
{
  struct timeval start, end;    // Profiling variables

  ACC_ERR_CODE	err	= ACC_ERR_NONE;
  candSrch*	res	= (candSrch*)ptr;
  cuSearch*	srch	= res->cuSrch;
  accelcand*	cand	= res->cand;
  confSpecsOpt*	conf	= srch->conf->opt;

  // Yes we use two different types of candidates =/
  initCand iCand;
  iCand.numharm		= cand->numharm;
  iCand.power		= cand->power;
  iCand.r		= cand->r;
  iCand.z		= cand->z;

  if ( conf->nelderMeadReps )
  {
    FOLD // Prep input
    {
      double sz = 5;	// This size could be a configurable parameter
      err += prepInput_cand( &iCand, res->input, srch->fft, sz, sz*conf->zScale, NULL, conf->flags );
    }

    if ( !ERROR_MSG(err, "ERROR: Preparing input for fine NM refinement.") )
    {
      PROF // Profiling  .
      {
	if ( !(!(conf->flags & FLAG_SYNCH) && (conf->flags & FLAG_OPT_THREAD)) )
	{
	  NV_RANGE_PUSH("NM_REFINE");
	}

	if ( conf->flags & FLAG_PROF )
	{
	  gettimeofday(&start, NULL);
	}
      }

      // Run the NM
      if ( conf->flags & FLAG_DPG_CAND_PLN )// DBG REM for testing
      {
	optInitCandPosSim<double>(&iCand,  res->input, res->resolution, res->resolution*conf->zScale, HIGHACC, conf->nelderMeadReps, conf->nelderMeadDelta, res->candNo, 0, 1  );
      }
      else
      {
	optInitCandPosSim<double>(&iCand,  res->input, res->resolution, res->resolution*conf->zScale, HIGHACC, conf->nelderMeadReps, conf->nelderMeadDelta, res->candNo, 0, 0  );
      }

      cand->r		= iCand.r;
      cand->z		= iCand.z;
      cand->power	= iCand.power;

      // Free thread specific input memory
      freeHarmInput(res->input);
      res->input = NULL;

      PROF // Profiling  .
      {
	if ( !(!(conf->flags & FLAG_SYNCH) && (conf->flags & FLAG_OPT_THREAD)) )
	{
	  NV_RANGE_POP("NM_REFINE");
	}

	if ( conf->flags & FLAG_PROF )
	{
	  gettimeofday(&end, NULL);
	  float v1 =  (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);

	  // Thread (pthread) safe add to timing value
	  pthread_mutex_lock(&res->cuSrch->threasdInfo->candAdd_mutex);
	  srch->timings[COMP_OPT_REFINE_2] += v1;
	  pthread_mutex_unlock(&res->cuSrch->threasdInfo->candAdd_mutex);
	}
      }
    }
  }

  optCandDerivs(cand, srch);

  // Decrease the count number of running threads
  sem_trywait(&srch->threasdInfo->running_threads);

  free(res);

  return (NULL);
}

/** Optimise derivatives of a candidate Using the CPU  .
 * This usually spawns a separate CPU thread to do the sigma calculations
 */
ACC_ERR_CODE processCandDerivs(accelcand* cand, cuSearch* srch, cuHarmInput* inp = NULL, double res = 0.001, int candNo = -1)
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  infoMSG(2,2,"Calc Cand Derivatives. r: %.6f  z: %.6f  harm: %i  power: %.2f \n", cand->r, cand->z, cand->numharm, cand->power);

  candSrch*     thrdDat  = new candSrch;
  memset(thrdDat, 0, sizeof(candSrch));

  confSpecsOpt*	conf	= srch->conf->opt;

  thrdDat->cand		= cand;
  thrdDat->cuSrch	= srch;
  thrdDat->candNo	= candNo;
  thrdDat->resolution	= res;

  if ( conf->nelderMeadReps )
  {
    // Make a copy of the input data for the thread to use
    thrdDat->input = duplicateHostInput(inp);
  }

  PROF // Profiling  .
  {
    NV_RANGE_PUSH("Post Thread");
  }

  // Increase the count number of running threads
  sem_post(&srch->threasdInfo->running_threads);

  if ( !(conf->flags & FLAG_SYNCH) && (conf->flags & FLAG_OPT_THREAD) )  // Create thread  .
  {
    pthread_t thread;
    int  iret1 = pthread_create( &thread, NULL, cpuProcess, (void*) thrdDat);

    if (iret1)	// Check return status
    {
      fprintf(stderr,"Error - pthread_create() return code: %d\n", iret1);
      exit(EXIT_FAILURE);
    }
  }
  else                              // Just call the function  .
  {
    cpuProcess( (void*) thrdDat );
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP("Post Thread");
  }

  infoMSG(2,2,"Done");

  return err;
}

/** Optimise a candidate location using ffdot planes  .
 *
 * @param cand		The candidate to refine
 * @param pln		The plane data structure to use for the GPU position refinement
 * @param candNo	The index of the candidate being optimised
 */
ACC_ERR_CODE optInitCandLocPlns(initCand* cand, cuOpt* opt, int candNo )
{
  infoMSG(2,2,"Refine location by plain\n");

  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  PROF // Profiling  .
  {
    NV_RANGE_PUSH("Plns");
  }

  confSpecsOpt*	conf	= opt->conf;
  cuRzHarmPlane* pln	= opt->plnGen->pln;

#ifdef CBL // DBG - Thesis output
  char tName[1024];
  sprintf(tName,"/home/chris/accel/OPT_%03i.csv", candNo);
  FILE *f2;
  if ( conf->flags & FLAG_DPG_CAND_PLN ) // Write CSV & plot output  .
  {
    f2 = fopen(tName, "w");
  }
#endif

  FOLD // Get best candidate location using iterative GPU planes  .
  {
    int depth;
    int noP;
    int rep	= 0;
    int lrep	= 0;
    const int	mxRep		= 10;
    const float moveBound	= 0.67;
    const float outBound	= 0.9;
    double sz;
    float posR, posZ;
    double rRes;

    if      ( cand->numharm == 1  )
      sz = conf->optPlnSiz[0];
    else if ( cand->numharm == 2  )
      sz = conf->optPlnSiz[1];
    else if ( cand->numharm == 4  )
      sz = conf->optPlnSiz[2];
    else if ( cand->numharm == 8  )
      sz = conf->optPlnSiz[3];
    else if ( cand->numharm == 16 )
      sz = conf->optPlnSiz[4];

    cand->power		= 0;				// Set initial power to zero

    CU_TYPE precision	= CU_NONE;

    for ( int lvl = 0; lvl < NO_OPT_LEVS; lvl++ )
    {
      noP		= conf->optPlnDim[lvl];		// Set in the defaults text file
      lrep		= 0;
      depth		= 1;

      if ( opt->plnGen->accu != conf->optPlnAccu[lvl])
      {
	opt->plnGen->accu	= conf->optPlnAccu[lvl];
	cand->power		= 0;			// Reset cand power as we are now using a different half-width
      }

      if ( precision != conf->optPlnPrec[lvl])
      {
	precision		= conf->optPlnPrec[lvl];
	cand->power		= 0;			// Reset cand power as we are now using a different half-width
      }

      double rRes = pln->rSize/(double)(pln->noR-1) ;

      if ( noP )					// Check if there are points in this plane ie. are we optimising position at this level  .
      {
	while ( (depth > 0) && (lrep < mxRep) )		// Recursively make planes at this scale  .
	{
	  infoMSG(3,3,"-----------------------------------------------------\n");

	  if      ( precision == CU_DOUBLE )
	  {
	    infoMSG(3,3,"Generate double precision plane - lvl %i  depth: %i  iteration %2i  size: %6.4f  dimension: %4i\n", lvl+1, depth, lrep, sz, noP );

	    // Double precision
	    optRefinePosPln<double>(cand, opt, noP, sz,  rep++, candNo, lvl + 1 );
	  }
	  else if ( precision == CU_FLOAT  )
	  {
	    infoMSG(3,3,"Generate single precision plane - lvl %i  depth: %i  iteration %2i  size: %6.4f  dimension: %4i\n", lvl+1, depth, lrep, sz, noP );

	    // Standard single precision
	    optRefinePosPln<float>(cand, opt, noP, sz,  rep++, candNo, lvl + 1 );
	  }
	  else
	  {
	    fprintf(stderr, "ERROR: Invalid plane precision\n.");
	    err = ACC_ERR_INVLD_CONFIG;
	  }

	  rRes = pln->rSize/(double)(pln->noR-1) ;				// Resolution of current plane
	  posR = fabs(( pln->centR - cand->r )/(pln->rSize/2.0));
	  posZ = fabs(( pln->centZ - cand->z )/(pln->zSize/2.0));

	  // Output Text
	  if ( posR || posZ )
	    infoMSG(4,4,"Plane max at relative offset: %.4f %.4f from centre of plane.\n", posR, posZ );
	  else
	    infoMSG(4,4,"Plane max in same position as current.\n");

	  FOLD // DBG - Thesis output
	  {
#ifdef CBL
	    if ( conf->flags & FLAG_DPG_CAND_PLN ) // Write CSV & plot output  .
	    {
	      fprintf(f2,"Optimisation details\n");
	      fprintf(f2,"Depth: %i\n", depth);
	      fprintf(f2,"Level: %i\n", lvl);
	      fprintf(f2,"Repetition: %i\n", rep);
	      fprintf(f2,"Precision: %i\n", 1);
	      fprintf(f2,"Max_r: %.23f\n", cand->r);
	      fprintf(f2,"Max_z: %.23f\n", cand->z);

	      err += ffdotPln_writePlnToFile(pln, f2);
	      fflush(f2);
	    }
#endif
	  }

	  // Determine whether to move the grid or zoom in
	  if ( posR > moveBound || posZ > moveBound )
	  {
	    if ( ( (posR > outBound) || (posZ > outBound) ) && ( depth < lvl+1) )
	    {
	      // Zoom out by half
	      sz *= conf->optPlnScale / 2.0 ;
	      depth++;
	      infoMSG(5,5,"Zoom out");
	    }
	    else
	    {
	      // we'r just going to move the plane
	      infoMSG(5,5,"Move plain");
	    }
	  }
	  else
	  {
	    // Zoom in
	    sz /= conf->optPlnScale;
	    depth--;
	    infoMSG(5,5,"Zoom in\n");
//	    if ( sz < 2.0*rRes )
//	      sz = rRes*2.0;
	  }

	  double plnRng = pln->maxPower - pln->minPower ;
	  double pre = pln->maxPower / plnRng ;			// 5000 -
	  if ( plnRng > 0 && plnRng < 0.2 )	// Potently force double precision
	  {
	    if ( precision != CU_DOUBLE )
	    {
	      infoMSG(5,5,"Set to double, range %3e.\n", plnRng);
	      cand->power = 0;	// Reset plane maximum value
	      precision = CU_DOUBLE;
	    }
	  }

	  ++lrep;
	}

	// Break condition
	if ( rRes < 1e-5 )
	{
	  infoMSG(5,5,"Stop iterations - resolution is fine enough.\n");
	  break;
	}
      }
      else
      {
	if ( precision == CU_DOUBLE )
	  infoMSG(3,3,"Skip plane lvl %i (double precision)", lvl+1);
	else
	  infoMSG(3,3,"Skip plane lvl %i (single precision)", lvl+1);
      }
    }
  }

#ifdef CBL // DBG - Thesis output
  if ( conf->flags & FLAG_DPG_CAND_PLN ) // Write CSV & plot output  .
  {
    fclose(f2);

//    FOLD // Make image  .
//    {
//      infoMSG(5,5,"Image %s\n", tName);
//
//      PROF // Profiling  .
//      {
//	NV_RANGE_PUSH("Image");
//      }
//
//      char cmd[1024];
//      sprintf(cmd,"python $PRESTO/python/pltOpt.py %s > /dev/null 2>&1", tName);
//      int ret = system(cmd);
//      if ( ret )
//      {
//	fprintf(stderr,"ERROR: Problem running potting python script.");
//      }
//
//      PROF // Profiling  .
//      {
//	NV_RANGE_POP("Image");
//      }
//    }
  }
#endif

  PROF // Profiling  .
  {
    NV_RANGE_POP("Plns");
  }

  return err;
}

initCand dupCand(accelcand* cand)
{
  initCand iCand;				// plane refining uses an initial candidate data structure
  iCand.r 		= cand->r;
  iCand.z 		= cand->z;
  iCand.power		= cand->power;
  iCand.numharm 	= cand->numharm;

  return iCand;
}

/** This is the main function called by external elements  .
 *
 * @param cand
 * @param pln
 * @param nn
 */
ACC_ERR_CODE opt_accelcand(accelcand* cand, cuOpt* opt, int candNo)
{
  ACC_ERR_CODE	err	= ACC_ERR_NONE;

  confSpecsOpt*	conf	= opt->conf;
  char Txt[128];

  PROF // Profiling  .
  {
    sprintf(Txt, "Opt Cand %03i", candNo);
    NV_RANGE_PUSH(Txt);
  }

  initCand iCand = dupCand(cand);		// plane refining uses an initial candidate data structure

//  initCand iCand;				// plane refining uses an initial candidate data structure
//  iCand.r 		= cand->r;
//  iCand.z 		= cand->z;
//  iCand.power		= cand->power;
//  iCand.numharm 	= cand->numharm;

  double resolved	= 0;

  FOLD // Refine position in ff space  .
  {
    struct timeval start, end;    // Profiling variables

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Refine pos");

      if ( conf->flags & FLAG_PROF )
      {
	gettimeofday(&start, NULL);
      }
    }

    if      ( conf->flags & FLAG_OPT_NM    )
    {
      double sz = 15;	// This size could be a configurable parameter
      prepInput_cand( &iCand, opt->input, opt->cuSrch->fft, sz, sz*conf->zScale, NULL, opt->flags );
      optInitCandPosSim<double>(&iCand, opt->input, 0.5, 0.5*conf->zScale, LOWACC, 1000, 1e-7 );

      resolved  = 1e-5 ;
    }
    else if ( conf->flags & FLAG_OPT_SWARM )
    {
      fprintf(stderr,"ERROR: Particle swarm optimisation has been removed.\n");
      exit(EXIT_FAILURE);
    }
    else // Default use planes
    {
      err += optInitCandLocPlns(&iCand, opt, candNo);

      resolved = opt->plnGen->pln->rSize / (opt->plnGen->pln->noR-1) * 0.1;

#ifdef CBL
      if ( opt->flags & FLAG_DPG_CAND_PLN )
      {
	nmLog.csvWrite("res", "%5e",  opt->plnGen->pln->rSize/(opt->plnGen->pln->noR-1));
      }
#endif

    }

    PROF // Profiling  .
    {
      NV_RANGE_POP("Refine pos");

      if ( conf->flags & FLAG_PROF )
      {
	gettimeofday(&end, NULL);
	float v1 =  (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec) ;

	// Thread (omp) safe add to timing value
#pragma omp atomic
	opt->cuSrch->timings[COMP_OPT_REFINE_1] += v1;
      }
    }
  }

  // Update the details of the final candidate from the updated initial candidate
  cand->r 		= iCand.r;
  cand->z 		= iCand.z;
  cand->power		= iCand.power;
  cand->numharm 	= iCand.numharm;

  FOLD // Optimise derivatives  .
  {
    err += processCandDerivs(cand, opt->cuSrch, opt->input, resolved, candNo);
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP(Txt);
  }

  return err;
}

/** Optimise all the candidates in a list
 *
 * @param listptr
 * @param cuSrch
 * @return
 */
int optList(GSList *listptr, cuSearch* cuSrch)
{
  struct timeval start, end;

#ifdef CBL // DBG - Thesis output
  char cmpName[1024];
  Logger cmpLog;
  if ( cuSrch->conf->opt->flags & FLAG_DPG_CAND_PLN )
  {
    sprintf(cmpName,"/home/chris/accel/cand_cmp.csv");
    cmpLog.setFile(cmpName);
    cmpLog.setEcho(true);
    cmpLog.setCsvLineNums(false);
    cmpLog.setCsvDeliminator(',');

    nmLog.setFile("/home/chris/accel/NM_2.csv");
    nmLog.setEcho(false);
    nmLog.setCsvLineNums(false);
    nmLog.setCsvDeliminator(';');
  }
#endif

  TIME //  Timing  .
  {
    NV_RANGE_PUSH("GPU Kernels");
  }

  int numcands	= g_slist_length(listptr);

  int ii	= 0;
  int comp	= 0;

#if	!defined(DEBUG) && defined(WITHOMP)   // Parallel if we are not in debug mode  .
  if ( cuSrch->conf->opt->flags & FLAG_SYNCH )
  {
    omp_set_num_threads(1);
  }
  else
  {
    omp_set_num_threads(cuSrch->oInf->noOpts);
  }
#pragma omp parallel
#endif	// !DEBUG && WITHOMP
  FOLD  	// Main GPU loop  .
  {
    accelcand *candGPU;

    int tid	= 0;
    int ti	= 0; // tread specific index

#ifdef	WITHOMP
    tid = omp_get_thread_num();
#endif	// WITHOMP

    cuOpt* opt = &(cuSrch->oInf->opts[tid]);

    setDevice(opt->gInf->devid) ;

    // Make sure all initialisation and other stuff on the device is complete
    CUDA_SAFE_CALL(cudaDeviceSynchronize(), "Synchronising device before candidate generation");

    while (listptr)  // Main Loop  .
    {
#pragma omp critical
      FOLD  // Synchronous behaviour  .
      {
#ifndef  DEBUG
	if ( cuSrch->conf->opt->flags & FLAG_SYNCH )
#endif
	{
	  tid 		= ii % cuSrch->oInf->noOpts ;
	  opt 		= &(cuSrch->oInf->opts[tid]);
	  setDevice(opt->gInf->devid);
	}

	FOLD // Calculate candidate  .
	{
	  if ( listptr )
	  {
	    candGPU	= (accelcand *) (listptr->data);
	    listptr	= listptr->next;
	    ii++;
	    ti = ii;
#ifdef CBL
	    if ( cuSrch->conf->opt->flags & FLAG_DPG_CAND_PLN ) // TMP: This can get removed
	    {
	      candGPU->init_power    = candGPU->power;
	      candGPU->init_sigma    = candGPU->sigma;
	      candGPU->init_numharm  = candGPU->numharm;
	      candGPU->init_r        = candGPU->r;
	      candGPU->init_z        = candGPU->z;
	    }
#endif
	  }
	  else
	  {
	    candGPU = NULL;
	  }
	}
      }

      if ( candGPU ) // Optimise  .
      {
	infoMSG(2,2,"\nOptimising initial candidate %i/%i, Power: %.3f  Sigma %.2f  Harm %i at (%.3f %.3f)\n", ti, numcands, candGPU->power, candGPU->sigma, candGPU->numharm, candGPU->r, candGPU->z );

	accelcand* candINT = new accelcand;
	accelcand* candCPU = new accelcand;

	*candINT = *candGPU; // TMP Duplicate candidate for comparison later
	*candCPU = *candGPU; // TMP Duplicate candidate for comparison later

	opt_accelcand(candGPU, opt, ti);

#pragma omp atomic
	comp++;

	FOLD // DBG - Compare optimisation results - Thesis output
	{
#ifdef CBL

	  ACC_ERR_CODE	err		= ACC_ERR_NONE;

	  //void rz_interp(fcomplex* data, int numdata, double r, double z, int kern_half_width, fcomplex * ans);

//	  int hw = getHw<double>(candGPU->z, HIGHACC);
//	  fcomplex ans;
//	  fcomplex* data = (fcomplex*)(&opt->input->h_inp[-opt->input->loR[0]]);
//	  long noBins = 70000000;
//	  double real, imag;
//	  candGPU->z = 0.0000001;
//
//	  ans.r = 0.0;
//	  ans.i = 0.0;
//	  real = 0.0;
//	  imag = 0.0;
//
//	  printf("\nCPU conv %.6f\n", candGPU->r);
//	  rz_interp(data, noBins, candGPU->r, candGPU->z, hw, &ans);
//
//	  printf("\nGPU conv %.6f\n", candGPU->r);
//	  rz_convolution_cu_debg<double, float2> ((float2*)data, 0, noBins, candGPU->r, candGPU->z, hw, &real, &imag);

	  bool prnt = false;

	  if ( cuSrch->conf->opt->flags & FLAG_DPG_PRNT_CAND )
	  {
	    cmpLog.csvWrite("idx",	"%5i",    ti);
	    cmpLog.csvWrite("r",	"%18.9f", candCPU->r);
	    cmpLog.csvWrite("z",	"%15.9f", candCPU->z);
	    cmpLog.csvWrite("pow",	"%15.9f", candCPU->power);
	    cmpLog.csvWrite("sig",	"%15.9f", candCPU->sigma);

	    int *r_offset;
	    fcomplex **data;
	    double r, z;

	    double iSig, gSig, cSig;

	    r_offset     = (int*) malloc(sizeof(int)*candCPU->numharm);
	    data         = (fcomplex**) malloc(sizeof(fcomplex*)*candCPU->numharm);

	    candCPU->pows	= gen_dvect(candCPU->numharm*2);
	    candCPU->hirs	= gen_dvect(candCPU->numharm*2);
	    candCPU->hizs	= gen_dvect(candCPU->numharm*2);
	    candCPU->derivs	= (rderivs *)  malloc(sizeof(rderivs) * candCPU->numharm);

	    for( int ii=0; ii< candCPU->numharm; ii++ )
	    {
	      r_offset[ii]   = 0;
	      data[ii]       = opt->cuSrch->fft->data;
	    }

	    FOLD // Original CPU optimisation
	    {
	      NV_RANGE_PUSH("CPU refine");

	      max_rz_arr_harmonics(data,
		  candCPU->numharm,
		  r_offset,
		  opt->cuSrch->fft->noBins,
		  candCPU->r,
		  candCPU->z,
		  &r,
		  &z,
		  candCPU->derivs,
		  candCPU->pows,
		  opt->input->norm);
	      candCPU->r = r;
	      candCPU->z = z;
	      candCPU->power = 0;
	      candCPU->sigma = 0;

	      NV_RANGE_POP("CPU refine");
	    }
	    else // My new NM optimisation  .
	    {
	      double sz = 15;	// This size could be a configurable parameter
	      prepInput_cand( candCPU, opt->input, opt->cuSrch->fft, sz, sz*opt->conf->zScale, opt->flags );

	      optInitCandPosSim<double>(candCPU, opt->input, 0.4,   0.4*opt->conf->zScale, LOWACC,  1000, 1e-7 );

	      optInitCandPosSim<double>(candCPU, opt->input, 0.01, 0.01*opt->conf->zScale, HIGHACC, 100, 1e-10  );
	    }

	    FOLD // Calculate powers  .
	    {
	      pow<double>(candINT, opt->input);
	      pow<double>(candCPU, opt->input);
	      pow<double>(candGPU, opt->input);

	      int noStages	= log2((double)candGPU->numharm);
	      long long numindep	= cuSrch->numindep[noStages];

	      //candINT->sigma	= candidate_sigma_cu(candINT->power, candINT->numharm, numindep);
	      //candGPU->sigma	= candidate_sigma_cu(candGPU->power, candGPU->numharm, numindep);
	      //candCPU->sigma	= candidate_sigma_cu(candCPU->power, candCPU->numharm, numindep);

	      iSig	= candidate_sigma_cu(candINT->power, candINT->numharm, numindep);
	      gSig	= candidate_sigma_cu(candGPU->power, candGPU->numharm, numindep);
	      cSig	= candidate_sigma_cu(candCPU->power, candCPU->numharm, numindep);
	    }

	    double rDist = candCPU->r - candGPU->r ;
	    double zDist = candCPU->z - candGPU->z ;
	    double bDist = sqrt(rDist*rDist + zDist*zDist);			// Euclidean distance between the two point (in bins)
	    double sDist = gSig - cSig ;

	    cmpLog.csvWrite("INIT pow",		"%15.9f", candINT->power);
	    cmpLog.csvWrite("INIT sig",		"%15.9f", iSig);

	    cmpLog.csvWrite("GPU r",		"%18.9f", candGPU->r);
	    cmpLog.csvWrite("GPU z",		"%15.9f", candGPU->z);
	    cmpLog.csvWrite("GPU Pow",		"%15.9f", candGPU->power );
	    cmpLog.csvWrite("GPU sig",		"%15.9f", gSig );

	    cmpLog.csvWrite("CPU r",		"%18.9f", candCPU->r );
	    cmpLog.csvWrite("CPU z",		"%15.9f", candCPU->z );
	    cmpLog.csvWrite("CPU pow",		"%15.9f", candCPU->power );
	    cmpLog.csvWrite("CPU sig",		"%15.9f", cSig );

	    cmpLog.csvWrite("Dist",		"%15.9f", bDist                              );
	    cmpLog.csvWrite("Pow diff",		"%15.9f",   candGPU->power - candCPU->power  );
	    cmpLog.csvWrite("Neg Pow diff",	"%15.9f", -(candGPU->power - candCPU->power) );
	    cmpLog.csvWrite("Dist",		"%15.9f", bDist                              );
	    cmpLog.csvWrite("Sig diff",		"%15.9f",  sDist                             );
	    cmpLog.csvWrite("Neg Sig diff",	"%15.9f", -sDist                             );

	    cmpLog.csvEndLine();

	    if ( sDist < -0.001 || sDist > 0.001 || bDist > 0.01 )
	    {
	      // These are interesting cases so plot them...
	      //prnt = true; // DBG removed for profiling
	    }

	    FOLD
	    {

	      confSpecsOpt*conf		= opt->conf;
	      cuRzHarmPlane* pln	= opt->plnGen->pln;

	      initCand iCand = dupCand(candGPU);

	      char pltNm[1024];
	      char pthNm[1024];
	      sprintf(pltNm, "CND_close_%03i", ti);
	      sprintf(pthNm, "-p /home/chris/accel/nm_path_%03i.csv", ti);
	      //ffdotPln_plotPln( pln, "/home/chris/accel/", pltNm, pthNm);


	      double scale = pln->rSize / (pln->noR-1);

//	      err += optRefinePosPln<double>(&iCand, opt, 128, scale, 1, ti, 0 );
//	      //ffdotPln_plotPln( pln, "/home/chris/accel/", pltNm );
//
//	      err += optRefinePosPln<double>(&iCand, opt, 128, scale/=10.0, 1, ti, 1 );
//	      //ffdotPln_plotPln( pln, "/home/chris/accel/", pltNm );
//
//	      err += optRefinePosPln<double>(&iCand, opt, 128, scale/=10.0, 1, ti, 2 );
//	      //ffdotPln_plotPln( pln, "/home/chris/accel/", pltNm );
//
//	      err += optRefinePosPln<double>(&iCand, opt, 128, scale/=10.0, 1, ti, 3 );
//	      //ffdotPln_plotPln( pln, "/home/chris/accel/", pltNm );
//
//	      err += optRefinePosPln<double>(&iCand, opt, 128, scale/=10.0, 1, ti, 4 );
//	      //ffdotPln_plotPln( pln, "/home/chris/accel/", pltNm );
//
//	      err += optRefinePosPln<double>(&iCand, opt, 128, scale/=10.0, 1, ti, 5 );
//	      //ffdotPln_plotPln( pln, "/home/chris/accel/", pltNm );


	      pln->centR = candGPU->r;
	      pln->centZ = candGPU->z;

	      //pln->rSize / (pln->noR-1) * 0.5;
	      pln->rSize = pln->rSize / (pln->noR-1) * 1.5 ;
	      pln->zSize = pln->rSize*conf->zScale;

	      pln->noZ		= 128;
	      pln->noR		= 128;

	      //opt->plnGen->flags |= FLAG_OPT_CPU_PLN;

	      ffdotPln<double>(opt->plnGen, opt->cuSrch->fft);
	      ffdotPln_plotPln( pln, "/home/chris/accel/", pltNm, pthNm );

	      //opt->plnGen->flags &= ~FLAG_OPT_CPU_PLN;


	    }

//	    FOLD
//	    {
//	      confSpecsOpt*conf		= opt->conf;
//	      cuRzHarmPlane* pln	= opt->plnGen->pln;
//
//	      pln->centR = candGPU->r;
//	      pln->centZ = candGPU->z;
//
//	      //pln->rSize / (pln->noR-1) * 0.5;
//	      pln->rSize = pln->rSize / (pln->noR-1) * 2.0;
//	      pln->zSize = pln->rSize*conf->zScale;
//
//	      pln->noZ		= 128;
//	      pln->noR		= 128;
//
//	      //opt->plnGen->flags |= FLAG_OPT_CPU_PLN;
//
//	      ffdotPln<double>(opt->plnGen, opt->cuSrch->fft);
//
//	      //opt->plnGen->flags &= ~FLAG_OPT_CPU_PLN;
//
//	      char pltNm[1024];
//	      char pthNm[1024];
//	      sprintf(pltNm, "CND_close_%03i", ti);
//	      sprintf(pthNm, "-p /home/chris/accel/nm_path_%03i.csv", ti);
//	      ffdotPln_plotPln( pln, "/home/chris/accel/", pltNm, pthNm);
//	    }
	  }

	  if ( (opt->conf->flags & FLAG_DPG_CAND_PLN) ) // Write CSV & plot output  .
	  {
	    char tName[1024];
	    sprintf(tName, "/home/chris/accel/OPT_%03i.csv", ti);

	    FOLD // Used to mark points on the optimisation plot (plt_opt_cmp) .
	    {
	      FILE *f2 = fopen(tName, "a");

	      fprintf(f2, "init_r: %.23f\n", candINT->r);
	      fprintf(f2, "init_z: %.23f\n", candINT->z);

	      fprintf(f2, "GPU_r: %.23f\n", candGPU->r);
	      fprintf(f2, "GPU_z: %.23f\n", candGPU->z);

	      fprintf(f2, "CPU_r: %.23f\n", candCPU->r);
	      fprintf(f2, "CPU_z: %.23f\n", candCPU->z);

	      fclose(f2);
	    }

	    if ( prnt || (opt->conf->flags & FLAG_DPG_PLT_OPT) ) // Make image  .
	    {
	      infoMSG(5,5,"Image %s\n", tName);

	      PROF // Profiling  .
	      {
		NV_RANGE_PUSH("Image");
	      }

	      char cmd[1024];
	      sprintf(cmd,"python $PRESTO/python/plt_opt_plns.py %s > /dev/null 2>&1", tName);
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
	  }

#endif // CBL
	}

//	if ( msgLevel == 0 )
//	{
//	  printf("\rGPU optimisation %5.1f%% complete   ", comp / (float)numcands * 100.0f );
//	  fflush(stdout);
//	}
      }
    }
  }

#ifdef CBL // DBG - Thesis output
  if ( cuSrch->conf->opt->flags & FLAG_DPG_PRNT_CAND )
  {
    cmpLog.close();

    char cmd[1024];
    sprintf(cmd,"python /home/chris/projects/thesis-plots/plt_opt_cmp.py %s > /dev/null 2>&1", cmpName);
    int ret = system(cmd);
    if ( ret )
    {
	fprintf(stderr,"ERROR: Problem running potting python script.");
    }
  }
#endif

  printf("\rGPU optimisation %5.1f%% complete                      \n", 100.0f );

  TIME //  Timing  .
  {
    NV_RANGE_POP("GPU Kernels");
    gettimeofday(&start, NULL);
  }

  // Wait for CPU derivative threads to finish
  waitForThreads(&cuSrch->threasdInfo->running_threads, "Waiting for CPU threads to complete.", 200 );

  TIME //  Timing  .
  {
    gettimeofday(&end, NULL);
    cuSrch->timings[TIME_OPT_WAIT] += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
  }

#ifdef CBL // DBG - Thesis output
  if ( cuSrch->conf->opt->flags & FLAG_DPG_PRNT_CAND )
  {
  nmLog.close();
  }
#endif

  return 0;
}



