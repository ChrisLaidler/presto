/**   cuda_accel_utils.h
 *
 * This file contains the candidate optimisation specific stuff
 *
 */

#ifndef CUDA_CAND_OPT_INCLUDED
#define CUDA_CAND_OPT_INCLUDED


#include "cuda_accel.h"


ACC_ERR_CODE opt_accelcand(accelcand* cand, cuPlnGen* plnGen, int no );

ExternC int  ffdotPln(float* powers, fcomplex* fft, int loR, int noBins, int noHarms, double centR, double centZ, double rSZ, double zSZ, int noR, int noZ, int halfwidth, float* fac);

cuOpt* initOptimiser(cuSearch* sSrch, cuOpt* opt, gpuInf* gInf );

ACC_ERR_CODE freeOptimiser(cuOpt* opt);

#endif
