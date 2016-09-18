/**   cuda_accel_utils.h
 *
 * This file contains the candidate optimisation specific stuff
 *
 */

#ifndef CUDA_CAND_OPT_INCLUDED
#define CUDA_CAND_OPT_INCLUDED


#include "cuda_accel.h"


#define		OPT_KER_PLN_BLK_NRM	BIT(1)
#define		OPT_KER_PLN_BLK_EXP	BIT(2)
#define		OPT_KER_PLN_BLK_3	BIT(3)
#define		OPT_KER_PLN_BLK		( OPT_KER_PLN_BLK_NRM | OPT_KER_PLN_BLK_EXP | OPT_KER_PLN_BLK_3 )
#define		OPT_KER_PLN_PTS_NRM	BIT(5)
#define		OPT_KER_PLN_PTS_EXP	BIT(6)
#define		OPT_KER_PLN_PTS_3	BIT(7)
#define		OPT_KER_PLN_PTS_SHR	BIT(8)


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


#endif
