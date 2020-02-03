/*
 * cuda_accel_PLN.h
 *
 *  Created on: 19 May 2017
 *      Author: chris
 */

#include "cuda_accel.h"
#include "cuda_accel_utils.h"


acc_err ffdotPln_calcCols( cuRzHarmPlane* pln, int64_t flags, int colDivisor = 4, int target_noCol = 16);

acc_err chkInput_pln(cuHarmInput* input, cuRzHarmPlane* pln, fftInfo* fft, int* newInp);

acc_err stridePln(cuRzHarmPlane* pln, gpuInf* gInf);

acc_err ffdotPln_writePlnToFile(cuRzHarmPlane* pln, FILE *f2);

acc_err ffdotPln_plotPln( cuRzHarmPlane* pln, const char* dir, const char* name,  const char* prams = NULL );

cuRzHarmPlane* initPln( size_t memSize );

cuRzHarmPlane* dupPln( cuRzHarmPlane* orrpln );

acc_err freePln(cuRzHarmPlane* pln);

cuPlnGen* initPlnGen(int maxHarms, float zMax, confSpecsCO* conf, gpuInf* gInf);

acc_err freePlnGen(cuPlnGen* plnGen);

acc_err snapPlane(cuRzHarmPlane* pln);

acc_err centerPlane(cuRzHarmPlane* pln, double r, double z, bool snap  = false );

acc_err centerPlaneOnCand(cuRzHarmPlane* pln, initCand* cand, bool snap = false);

acc_err getKerName(cuPlnGen* plnGen, char* name);

template<typename T>
acc_err ffdotPln( cuPlnGen* plnGen, fftInfo* fft, int* newInp = NULL );

acc_err input_plnGen( cuPlnGen* plnGenGen, fftInfo* fft, int* newInp = NULL );

acc_err chkInput_ffdotPln( cuPlnGen* plnGenGen, fftInfo* fft, int* newInp );

acc_err prepInput_ffdotPln( cuPlnGen* plnGen, fftInfo* fft );

acc_err cpyInput_ffdotPln( cuPlnGen* plnGen, fftInfo* fft );

template<typename T>
acc_err prep_plnGenerator( cuPlnGen* plnGen, fftInfo* fft );

acc_err ffdotPln_cOps( cuPlnGen* plnGen, unsigned long long* cOps);

acc_err ffdotPln_cOps_harms( cuPlnGen* plnGen, unsigned long long* cOps);

template<typename T>
acc_err ffdotPln_ker( cuPlnGen* plnGen );

acc_err ffdotPln_cpyResultsD2H( cuPlnGen* plnGen, fftInfo* fft );

acc_err ffdotPln_ensurePln( cuPlnGen* plnGen, fftInfo* fft );
