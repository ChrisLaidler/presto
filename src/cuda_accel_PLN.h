/*
 * cuda_accel_PLN.h
 *
 *  Created on: 19 May 2017
 *      Author: chris
 */

#include "cuda_accel.h"
#include "cuda_accel_utils.h"


ACC_ERR_CODE ffdotPln_calcCols( cuRzHarmPlane* pln, int64_t flags, int colDivisor = 4, int target_noCol = 16);

ACC_ERR_CODE chkInput_pln(cuHarmInput* input, cuRzHarmPlane* pln, fftInfo* fft, int* newInp);

ACC_ERR_CODE stridePln(cuRzHarmPlane* pln, gpuInf* gInf);

ACC_ERR_CODE ffdotPln_writePlnToFile(cuRzHarmPlane* pln, FILE *f2);

ACC_ERR_CODE ffdotPln_plotPln( cuRzHarmPlane* pln, const char* dir, const char* name,  const char* prams = NULL );

cuRzHarmPlane* initPln( size_t memSize );

cuRzHarmPlane* dupPln( cuRzHarmPlane* orrpln );

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

template<typename T>
ACC_ERR_CODE prep_plnGenerator( cuPlnGen* plnGen, fftInfo* fft );

ACC_ERR_CODE ffdotPln_cOps( cuPlnGen* plnGen, unsigned long long* cOps);

ACC_ERR_CODE ffdotPln_cOps_harms( cuPlnGen* plnGen, unsigned long long* cOps);

template<typename T>
ACC_ERR_CODE ffdotPln_ker( cuPlnGen* plnGen );

ACC_ERR_CODE ffdotPln_cpyResultsD2H( cuPlnGen* plnGen, fftInfo* fft );

ACC_ERR_CODE ffdotPln_ensurePln( cuPlnGen* plnGen, fftInfo* fft );
