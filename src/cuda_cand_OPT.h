/**   cuda_accel_utils.h
 *
 * This file contains the candidate optimisation specific stuff
 *
 */

#ifndef CUDA_CAND_OPT_INCLUDED
#define CUDA_CAND_OPT_INCLUDED


#include "cuda_accel.h"


acc_err opt_accelcand(accelcand* cand, cuCoPlan* opt, int no = 0 );

ExternC int  ffdotPln(float* powers, fcomplex* fft, int loR, int noBins, int noHarms, double centR, double centZ, double rSZ, double zSZ, int noR, int noZ, int halfwidth, float* fac);

cuCoPlan* initOptimiser(cuSearch* sSrch, cuCoPlan* opt, gpuInf* gInf );

acc_err freeOptimiser(cuCoPlan* opt);

acc_err pln_max_pnt( cuRzHarmPlane* pln, initCand* cand );

acc_err pln_max_wAve( cuRzHarmPlane* pln, initCand* cand, double bound );

#endif
