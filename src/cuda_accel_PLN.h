/*
 * cuda_accel_PLN.h
 *
 *  Created on: 19 May 2017
 *      Author: chris
 */

#include "cuda_accel.h"
//#include "cuda_utils.h"
//#include "cuda_math.h"
//#include "candTree.h"
//#include "cuda_accel_PLN.h"
#include "cuda_accel_utils.h"


// This allows the possibility of creating planes with up to 32 harmonics

#ifdef		OPT_LOC_32

#define		OPT_MAX_LOC_HARMS	32
typedef		int32 optLocInt_t;
typedef		int32 optLocFloat_t;

#else

#define		OPT_MAX_LOC_HARMS	16
typedef		int16 optLocInt_t;
typedef		int16 optLocFloat_t;

#endif

ACC_ERR_CODE ffdotPln_calcCols( cuRzHarmPlane* pln, int64_t flags, int colDivisor = 1);

ACC_ERR_CODE chkInput_pln(cuHarmInput* input, cuRzHarmPlane* pln, fftInfo* fft, int* newInp);

ACC_ERR_CODE stridePln(cuRzHarmPlane* pln, gpuInf* gInf);

ACC_ERR_CODE ffdotPln_writePlnToFile(cuRzHarmPlane* pln, FILE *f2);

ACC_ERR_CODE ffdotPln_writePlnToFileNew(cuRzHarmPlane* pln, FILE *f2);

ACC_ERR_CODE ffdotPln_plotPln( cuRzHarmPlane* pln, const char* dir, const char* name );

cuRzHarmPlane* initPln( size_t memSize );

ACC_ERR_CODE freePln(cuRzHarmPlane* pln);

cuPlnGen* initPlnGen(int maxHarms, float zMax, confSpecsOpt* conf, gpuInf* gInf);

ACC_ERR_CODE freePlnGen(cuPlnGen* plnGen);

ACC_ERR_CODE snapPlane(cuRzHarmPlane* pln);

ACC_ERR_CODE centerPlane(cuRzHarmPlane* pln, double r, double z, bool snap  = false );

ACC_ERR_CODE centerPlaneOnCand(cuRzHarmPlane* pln, initCand* cand, bool snap = false);

ACC_ERR_CODE getKerName(cuPlnGen* plnGen, char* name);

template<typename T>
ACC_ERR_CODE ffdotPln( cuPlnGen* plnGen, fftInfo* fft, int* newInp = NULL );

ACC_ERR_CODE input_plnGen( cuPlnGen* plnGenGen, fftInfo* fft, int* newInp = NULL );

ACC_ERR_CODE chkInput_ffdotPln( cuPlnGen* plnGenGen, fftInfo* fft, int* newInp );

ACC_ERR_CODE prepInput_ffdotPln( cuPlnGen* plnGen, fftInfo* fft );

ACC_ERR_CODE cpyInput_ffdotPln( cuPlnGen* plnGen, fftInfo* fft );

ACC_ERR_CODE prep_Opt( cuPlnGen* plnGen, fftInfo* fft );

template<typename T>
ACC_ERR_CODE ffdotPln_ker( cuPlnGen* plnGen );

ACC_ERR_CODE ffdotPln_cpyResultsD2H( cuPlnGen* plnGen, fftInfo* fft );

ACC_ERR_CODE ffdotPln_ensurePln( cuPlnGen* plnGen, fftInfo* fft );