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

// TODO: write up descrition
cuOptCand* initOptCand(cuSearch* sSrch, cuOptCand* oPln = NULL, int devLstId = 0 );


int ffdotPln_prep( cuOptCand* pln, fftInfo* fft );

int ffdotPln_input( cuOptCand* pln, fftInfo* fft );

template<typename T>
int ffdotPln_ker( cuOptCand* pln, fftInfo* fft );

int ffdotPln_get( cuOptCand* pln, fftInfo* fft );

int ffdotPln_process( cuOptCand* pln, fftInfo* fft );

template<typename T>
int ffdotPln( cuOptCand* pln, fftInfo* fft );

#endif
