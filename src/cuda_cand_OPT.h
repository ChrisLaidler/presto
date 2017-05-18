/**   cuda_accel_utils.h
 *
 * This file contains the candidate optimisation specific stuff
 *
 */

#ifndef CUDA_CAND_OPT_INCLUDED
#define CUDA_CAND_OPT_INCLUDED


#include "cuda_accel.h"





#define		OPT_LOC_PNT_NO		16

#ifdef		OPT_LOC_32

#define		OPT_MAX_LOC_HARMS	32

typedef int32 optLocInt_t;
typedef int32 optLocFloat_t;

#else

#define		OPT_MAX_LOC_HARMS	16

typedef int16 optLocInt_t;
typedef int16 optLocFloat_t;

#endif


ExternC int  ffdotPln(float* powers, fcomplex* fft, int loR, int noBins, int noHarms, double centR, double centZ, double rSZ, double zSZ, int noR, int noZ, int halfwidth, float* fac);
//ExternC void opt_candPlns(accelcand* cand, cuSearch* srch, accelobs* obs, int nn, cuOptCand* pln);
//ExternC void opt_candSwrm(accelcand* cand, accelobs* obs, int nn, cuOptCand* pln);

ExternC void opt_accelcand(accelcand* cand, cuOptCand* pln, int no);

// TODO: write up description
cuOptCand* initOptimiser(cuSearch* sSrch, cuOptCand* oPln = NULL, int devLstId = 0 );

ACC_ERR_CODE getKerName(cuOptCand* pln, char* name);

template<typename T>
ACC_ERR_CODE ffdotPln( cuOptCand* pln, fftInfo* fft, int* newInp = NULL );

ACC_ERR_CODE ffdotPln_input( cuOptCand* pln, fftInfo* fft, int* newInp = NULL );

ACC_ERR_CODE ffdotPln_chkInput( cuOptCand* pln, fftInfo* fft, int* newInp);

ACC_ERR_CODE ffdotPln_prepInput( cuOptCand* pln, fftInfo* fft );

ACC_ERR_CODE ffdotPln_cpyInput( cuOptCand* pln, fftInfo* fft );

ACC_ERR_CODE ffdotPln_prep( cuOptCand* pln, fftInfo* fft );

template<typename T>
ACC_ERR_CODE ffdotPln_ker( cuOptCand* pln, fftInfo* fft );

ACC_ERR_CODE ffdotPln_cpyResultsD2H( cuOptCand* pln, fftInfo* fft );

ACC_ERR_CODE ffdotPln_ensurePln( cuOptCand* pln, fftInfo* fft );

ACC_ERR_CODE ffdotPln_plotPln( cuOptCand* pln, const char* dir, const char* name );


#endif
