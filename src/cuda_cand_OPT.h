/**   cuda_accel_utils.h
 *
 * This file contains the candidate optimisation specific stuff
 *
 */

#ifndef CUDA_CAND_OPT_INCLUDED
#define CUDA_CAND_OPT_INCLUDED


#include "cuda_accel.h"


ACC_ERR_CODE opt_accelcand(accelcand* cand, cuOpt* opt, int no = 0 );

ExternC int  ffdotPln(float* powers, fcomplex* fft, int loR, int noBins, int noHarms, double centR, double centZ, double rSZ, double zSZ, int noR, int noZ, int halfwidth, float* fac);

cuOpt* initOptimiser(cuSearch* sSrch, cuOpt* opt, gpuInf* gInf );

ACC_ERR_CODE freeOptimiser(cuOpt* opt);

ACC_ERR_CODE pln_max_pnt( cuRzHarmPlane* pln, initCand* cand );

ACC_ERR_CODE pln_max_wAve( cuRzHarmPlane* pln, initCand* cand, double bound );

#endif
