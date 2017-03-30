/** @file cuda_accel_IN.cu
 *  @brief Functions to manage normalisation and FFT'ing of input data for CUDA accelsearch
 *
 *  This contains the various functions that control and undertake input normalisation and FFT
 *  These include:
 *    Input Normalisation
 *    Input FFT
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
 *  [0.0.02] [2017-01-28 10:25]
 *    Fixed bug in synchronous in-mem runs (added a block on event ifftMemComp)
 *    Added some debug messages on stream synchronisation on events
 *
 *  [0.0.03] [2017-01-29 08:20]
 *    Added static functions to call both CPU and GPU input FFT's, these allow identical calls from non critical and non critical blocks
 *    Added non critical behaviour for CPU FFT calls
 *    Added some debug messages on stream synchronisation on events, yes even more!
 *    made CPU_Norm_Spread static
 *    Fixed bug in timing of CPU input
 *    
 *  [0.0.03] [2017-02-03 ]
 *    Converted to use of clearRval
 *
 *  [0.0.03] [2017-02-18 ]
 *    Different memory management for GPU normalisation
 *
 */


#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"
#include "cuda_accel_IN.h"

int    cuMedianBuffSz = -1;             ///< The size of the sub sections to use in the CUDA median selection algorithm - If <= 0 an automatic value is used

/** Normalise input using median normalisation
 *
 * This s done using a temporary host buffer
 *
 */
static void CPU_Norm_Spread(cuFFdotBatch* batch, fcomplexcu* fft)
{
  infoMSG(3,3,"CPU normalise batch input.");

  PROF // Profiling  .
  {
    NV_RANGE_PUSH("CPU_Norm_Spread");
  }

  int harm = 0;

  FOLD // Normalise, spread and copy raw input fft data to pinned memory  .
  {
    for (int stack = 0; stack < batch->noStacks; stack++)
    {
      cuFfdotStack* cStack = &batch->stacks[stack];

      int sz = 0;
      struct timeval start, end;  // Profiling variables
      int noRespPerBin = batch->cuSrch->sSpec->noResPerBin;

      PROF // Profiling  .
      {
	if ( batch->flags & FLAG_PROF )
	{
	  gettimeofday(&start, NULL);
	}
      }

      for (int si = 0; si < cStack->noInStack; si++)
      {
	for (int step = 0; step < batch->noSteps; step++)
	{
	  rVals* rVal = &(*batch->rAraays)[batch->rActive][step][harm];

	  if ( rVal->numdata )
	  {
	    if ( batch->cuSrch->sSpec->normType == 0 )	// Block median normalisation  .
	    {
	      int startBin = rVal->lobin < 0 ? -rVal->lobin : 0 ;
	      int endBin   = rVal->lobin + rVal->numdata >= batch->cuSrch->sSpec->fftInf.lastBin ? rVal->lobin + rVal->numdata - batch->cuSrch->sSpec->fftInf.lastBin : rVal->numdata ;

	      if ( rVal->norm == 0.0 )
	      {
		FOLD // Calculate and store powers  .
		{
		  PROF // Profiling  .
		  {
		    NV_RANGE_PUSH("Powers");
		  }

		  for (int ii = 0; ii < rVal->numdata; ii++)
		  {
		    if ( rVal->lobin+ii < batch->cuSrch->sSpec->fftInf.firstBin || rVal->lobin+ii  >= batch->cuSrch->sSpec->fftInf.lastBin ) // Zero Pad
		    {
		      batch->h_normPowers[ii] = 0;
		    }
		    else
		    {
		      batch->h_normPowers[ii] = POWERCU(fft[rVal->lobin+ii].r, fft[rVal->lobin+ii].i);
		    }
		  }

		  PROF // Profiling  .
		  {
		    NV_RANGE_POP(); // Powers
		  }
		}

		FOLD // Calculate normalisation factor from median  .
		{
		  PROF // Profiling  .
		  {
		    NV_RANGE_PUSH("Median");
		  }

		  if ( batch->flags & CU_NORM_EQUIV )
		  {
		    rVal->norm = 1.0 / sqrt(median(batch->h_normPowers, (rVal->numdata)) / log(2.0));        /// NOTE: This is the same method as CPU version
		  }
		  else
		  {
		    rVal->norm = 1.0 / sqrt(median(&batch->h_normPowers[startBin], (endBin-startBin)) / log(2.0));    /// NOTE: This is a slightly better method (in my opinion)
		  }

		  PROF // Profiling  .
		  {
		    NV_RANGE_POP(); // Median
		  }
		}
	      }

	      FOLD // Normalise and spread  .
	      {
		PROF // Profiling  .
		{
		  NV_RANGE_PUSH("Write");
		}

		for (int ii = 0; ( ii < rVal->numdata ) && ( (ii*noRespPerBin) < cStack->strideCmplx ); ii++)
		{
		  if ( rVal->lobin+ii < batch->cuSrch->sSpec->fftInf.firstBin || rVal->lobin+ii  >= batch->cuSrch->sSpec->fftInf.lastBin ) // Zero Pad
		  {
		    cStack->h_iBuffer[sz + ii * noRespPerBin].r = 0;
		    cStack->h_iBuffer[sz + ii * noRespPerBin].i = 0;
		  }
		  else
		  {
		    if ( ii * noRespPerBin > cStack->strideCmplx )
		    {
		      fprintf(stderr, "ERROR: nice_numdata is greater that width.\n");
		      exit(EXIT_FAILURE);
		    }

		    cStack->h_iBuffer[sz + ii * noRespPerBin].r = fft[rVal->lobin + ii].r * rVal->norm;
		    cStack->h_iBuffer[sz + ii * noRespPerBin].i = fft[rVal->lobin + ii].i * rVal->norm;
		  }
		}

		PROF // Profiling  .
		{
		  NV_RANGE_POP(); // Write
		}
	      }
	    }
	    else					// or double-tophat normalisation
	    {
	      int nice_numdata = cu_next2_to_n(rVal->numdata);  // for FFTs

	      if ( nice_numdata > cStack->width )
	      {
		fprintf(stderr, "ERROR: nice_numdata is greater that width.\n");
		//exit(EXIT_FAILURE);
	      }

	      // Do the actual copy
	      //memcpy(batch->h_powers, &fft[lobin], numdata * sizeof(fcomplexcu) );

	      //  new-style running double-tophat local-power normalization
	      float *loc_powers;

	      //powers = gen_fvect(nice_numdata);
	      for (int ii = 0; ii< nice_numdata; ii++)
	      {
		batch->h_normPowers[ii] = POWERCU(fft[rVal->lobin+ii].r, fft[rVal->lobin+ii].i);
	      }
	      loc_powers = corr_loc_pow(batch->h_normPowers, nice_numdata);

	      //memcpy(&batch->h_iBuffer[sz], &fft[lobin], nice_numdata * sizeof(fcomplexcu) );

	      for (int ii = 0; ii < rVal->numdata; ii++)
	      {
		float norm = invsqrt(loc_powers[ii]);

		batch->h_iBuffer[sz + ii * noRespPerBin].r = fft[rVal->lobin+ ii].r* norm;
		batch->h_iBuffer[sz + ii * noRespPerBin].i = fft[rVal->lobin+ ii].i* norm;
	      }

	      vect_free(loc_powers);  // I hate doing this!!!
	    }
	  }

	  sz += cStack->strideCmplx;
	}
	harm++;
      }

      PROF // Profiling  .
      {
	if ( batch->flags & FLAG_PROF )
	{
	  gettimeofday(&end, NULL);

	  float v1 =  (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
	  batch->compTime[batch->noStacks*COMP_GEN_NRM + stack ] += v1;
	}
      }
    }
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP(); // CPU_Norm_Spread
  }
}

/** Calculate the r bin values for this batch of steps and store them in planes->rInput
 *
 * This calculates r-low and halfwidth
 *
 * @param batch the batch to work with
 * @param searchRLow an array of the step r-low values
 * @param searchRHi an array of the step r-high values
 */
void setGenRVals(cuFFdotBatch* batch)
{
  infoMSG(4,4,"Set Stack R-Vals");

  PROF // Profiling  .
  {
    NV_RANGE_PUSH("Set R-Vals");
  }

  int	numrsArr[MAX_HARM_NO];
  for ( int i = 0; i< MAX_HARM_NO; i++ )
    numrsArr[i] = 0;

  int       hibin;
  int       binoffset;  // The extra bins to add onto the start of the data
  double    drlo, drhi;

  int lobin;				/// The first bin to copy from the the input fft ( serachR scaled - halfwidth )
  int numdata;				/// The number of input fft points to read
  int numrs;				/// The number of good bins in the plane ( expanded units )
  int noResPerBin;
  for (int harm = 0; harm < batch->noGenHarms; harm++)
  {
    cuHarmInfo* cHInfo		= &batch->hInfos[harm];					// The current harmonic we are working on
    noResPerBin			= cHInfo->noResPerBin;
    binoffset			= cHInfo->kerStart / noResPerBin;			// This aligns all the planes so the all the "usable" parts start at the same offset in the stack

    for (int step = 0; step < batch->noSteps; step++)
    {
      rVals* rVal		= &(*batch->rAraays)[batch->rActive][step][harm];
      rVals* rValFund		= &(*batch->rAraays)[batch->rActive][step][0];

      if ( rValFund->drhi - rValFund->drlo <= 0 )
      {
	clearRval(rVal);
      }
      else
      {
	drlo			= cu_calc_required_r(cHInfo->harmFrac, rValFund->drlo, noResPerBin);
	drhi			= cu_calc_required_r(cHInfo->harmFrac, rValFund->drhi, noResPerBin);

	lobin			= (int) floor(drlo) - binoffset;
	hibin			= (int) ceil(drhi)  + binoffset;

	if ( batch->flags & CU_NORM_GPU )
	{
	  // GPU normalisation now relies on all input for a stack being of the same length
	  numdata		= ceil(cHInfo->width / (float)noResPerBin); // Thus may use much more input data than is strictly necessary but thats OK!
	}
	else
	{
	  // CPU normalisation can normalise differing length data so use the correct lengths
	  numdata		= hibin - lobin + 1;	// NOTE: This + 1 isn't really necessary is it? its just taken from CPU presto
	}

	//numrs			= (int) ((ceil(drhi) - floor(drlo)) * noResPerBin + DBLCORRECT) + 1;
	numrs			= (int) ((ceil(drhi) - floor(drlo)) * noResPerBin);	// DBG This is a test, I found it gave errors with r-res that was greater than 2
	if ( harm == 0 )
	  numrs			= batch->accelLen;
	else if ( numrs % noResPerBin )
	  numrs			= (numrs / noResPerBin + 1) * noResPerBin;

	rVal->drlo		= drlo;
	rVal->drhi		= drhi;
	rVal->lobin		= lobin;
	rVal->numrs		= numrs;
	rVal->numdata		= numdata;
	rVal->expBin		= (lobin+binoffset)*noResPerBin;
	rVal->iteration		= rValFund->iteration;					//// Is it really necessary to do this here?

	int noEls		= numrs + 2*cHInfo->kerStart;

	FOLD		// DBG this can be taken out if it never fails
	{
	  if ( batch->flags & CU_NORM_GPU )
	  {
	    if ( numrsArr[harm] == 0 )
	    {
	      numrsArr[harm] = numdata;
	    }
	    else
	    {
	      if ( numrsArr[harm] != numdata )
	      {
		fprintf(stderr, "ERROR: numdata bad.");
		exit(EXIT_FAILURE);
	      }
	    }
	  }
	}

	if  ( noEls > cHInfo->width )
	{
	  fprintf(stderr, "ERROR: Number of elements in step greater than width of the plane! harm: %i\n", harm);
	  exit(EXIT_FAILURE);
	}
      }
    }
  }

  PROF // Profiling  .
  {
    NV_RANGE_POP(); // Set R-Vals
  }
}

/** Calculate the r bin values for this batch of steps and store them in planes->rInput
 *
 * This calculates r-low and halfwidth
 *
 * @param batch the batch to work with
 * @param searchRLow an array of the step r-low values
 * @param searchRHi an array of the step r-high values
 */
void setSearchRVals(cuFFdotBatch* batch, double searchRLow, long len)
{
  infoMSG(3,3,"Set Stack R-Vals");

  FOLD // Set the r values for this step  .
  {
    for (int harm = 0; harm < batch->noGenHarms; harm++)
    {
      for (int step = 0; step < batch->noSteps; step++)
      {
	rVals* rVal           = &(*batch->rAraays)[batch->rActive][step][harm];

	if ( (step != 0) || (len == 0) )
	{
	  clearRval(rVal);
	}
	else
	{
	  rVal->drlo		= searchRLow;
	  rVal->lobin		= 0;
	  rVal->numrs		= len;
	  rVal->numdata		= 0;
	  rVal->expBin		= 0;
	}
      }
    }
  }
}

/** A simple function to call the input FFTW plan  .
 *
 * This is a seperate function so one can be called by a omp critical and another not
 */
static void callInputFFTW(cuFFdotBatch* batch)
{
  // Profiling variables  .
  struct timeval start, end;

  for (int stack = 0; stack < batch->noStacks; stack++)
  {
    cuFfdotStack* cStack = &batch->stacks[stack];

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("CPU FFT");

      if ( batch->flags & FLAG_PROF )
      {
	gettimeofday(&start, NULL);
      }
    }

    // Do the FFT using the memory buffer
    fftwf_execute_dft(cStack->inpPlanFFTW, (fftwf_complex*)cStack->h_iBuffer, (fftwf_complex*)cStack->h_iBuffer);

    PROF // Profiling  .
    {
      NV_RANGE_POP(); // CPU FFT

      if ( batch->flags & FLAG_PROF )
      {
	gettimeofday(&end, NULL);

	float v1 =  (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
	batch->compTime[NO_STKS*COMP_GEN_FFT + stack ] += v1;
      }
    }
  }
}

/** A simple function to call the input CUFFT plan  .
 *
 * This is a seperate function so one can be called by a omp critical and another not
 */
static void callInputCUFFT(cuFFdotBatch* batch, cuFfdotStack* cStack)
{
  PROF // Profiling  .
  {
    if ( batch->flags & FLAG_PROF )
    {
      infoMSG(5,5,"Event %s in %s.\n", "inpFFTinit", "fftIStream");
      cudaEventRecord(cStack->inpFFTinit, cStack->fftIStream);
    }
  }

  CUFFT_SAFE_CALL(cufftSetStream(cStack->inpPlan, cStack->fftIStream),"Failed associating a CUFFT plan with FFT input stream\n");
  CUFFT_SAFE_CALL(cufftExecC2C(cStack->inpPlan, (cufftComplex *) cStack->d_iData, (cufftComplex *) cStack->d_iData, CUFFT_FORWARD),"Failed to execute input CUFFT plan.");

  CUDA_SAFE_CALL(cudaGetLastError(), "FFT'ing the input data.");

  FOLD // Synchronisation  .
  {
    infoMSG(5,5,"Event %s in %s.\n", "inpFFTinitComp", "fftIStream");
    cudaEventRecord(cStack->inpFFTinitComp, cStack->fftIStream);
  }
}

/** Initialise input data for a f-∂f plane(s)  ready for convolution  .
 * This:
 *  Normalises the chunk of input data
 *  Spreads it (interbinning)
 *  FFT it ready for convolution
 *
 * @param batch the batch to work with
 */
void prepInputCPU(cuFFdotBatch* batch )
{
  // Calculate various values for the step
  setGenRVals(batch);

  if ( (*batch->rAraays)[batch->rActive][0][0].numrs ) // This is real data ie this isn't just a call to finish off asynchronous work
  {
    infoMSG(2,2,"CPU prep input - Iteration %3i.", (*batch->rAraays)[batch->rActive][0][0].iteration);

    CUDA_SAFE_CALL(cudaGetLastError(), "prepInputCPU");

    // Profiling variables  .
    struct timeval start, end, start0, end0;

    fcomplexcu* fft = (fcomplexcu*)batch->cuSrch->sSpec->fftInf.fft;

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("CPU prep input");

      if ( batch->flags & FLAG_PROF )
      {
	gettimeofday(&start0, NULL);
      }
    }

    FOLD // Prepare the temporary host buffer  .
    {
      // NOTE: I use a temporary host buffer, so that the normalisation and FFT can be done before synchronisation

      if ( batch->flags & CU_NORM_GPU  )	// Write input data segments to contiguous page locked memory  .
      {
	infoMSG(4,4,"GPU normalisation - Copy input data to buffer.");

	int harm  = 0;
	int sz    = 0;

	FOLD // Synchronisation [ blocking ]  .
	{
	  infoMSG(4,4,"blocking synchronisation on %s", "iDataCpyComp" );

	  PROF // Profiling  .
	  {
	    NV_RANGE_PUSH("EventSynch iDataCpyComp");
	  }

	  CUDA_SAFE_CALL(cudaEventSynchronize(batch->iDataCpyComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");

	  PROF // Profiling  .
	  {
	    NV_RANGE_POP(); // EventSynch
	  }
	}

	PROF // Profiling  .
	{
	  NV_RANGE_PUSH("cpy inpt FFT");

	  if ( batch->flags & FLAG_PROF )
	  {
	    gettimeofday(&start, NULL);
	  }
	}

	for ( int stack = 0; stack< batch->noStacks; stack++)  // Loop over stack  .
	{
	  cuFfdotStack* cStack = &batch->stacks[stack];

	  for ( int plane = 0; plane < cStack->noInStack; plane++)
	  {
	    for (int step = 0; step < batch->noSteps; step++)
	    {
	      rVals* rVal = &(*batch->rAraays)[batch->rActive][step][harm];

	      if ( rVal->numdata )
	      {
		int startIdx = 0;
		if ( rVal->lobin < 0 )
		{
		  startIdx = -rVal->lobin;		// Offset

		  // Zero the beginning
		  memset(&batch->h_iData[sz], 0, startIdx * sizeof(fcomplexcu));
		}

		// Do the actual copy
		memcpy(&batch->h_iData[sz+startIdx], &fft[rVal->lobin+startIdx], (rVal->numdata-startIdx) * sizeof(fcomplexcu));
	      }
	      sz += cStack->strideCmplx;
	    }
	    harm++;
	  }
	}

	PROF // Profiling  .
	{
	  NV_RANGE_POP(); // cpy inpt FFT

	  if ( batch->flags & FLAG_PROF )
	  {
	    gettimeofday(&end, NULL);
	    float time = (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
	    int idx = MIN(1, batch->noStacks-1);
	    batch->compTime[NO_STKS*COMP_GEN_MEM+idx] += time;
	  }
	}
      }
      else					// Copy chunks of FFT data and normalise and spread using the CPU  .
      {
	FOLD // Zero host memory buffer  .
	{
	  infoMSG(4,4,"Zero tmp buffer");

	  PROF // Profiling  .
	  {
	    NV_RANGE_PUSH("Zero buffer");

	    if ( batch->flags & FLAG_PROF )
	    {
	      gettimeofday(&start, NULL);
	    }
	  }

	  memset(batch->h_iBuffer, 0, batch->inpDataSize);

	  PROF // Profiling  .
	  {
	    NV_RANGE_POP(); // Zero buffer

	    if ( batch->flags & FLAG_PROF )
	    {
	      gettimeofday(&end, NULL);
	      float time = (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
	      int idx = MIN(0, batch->noStacks-1);
	      batch->compTime[NO_STKS*COMP_GEN_MEM+idx] += time;
	    }
	  }
	}

	FOLD // CPU Normalise  .
	{
	  CPU_Norm_Spread(batch, fft);
	}

	FOLD // FFT  .
	{
	  if ( batch->flags & CU_INPT_FFT_CPU ) // CPU FFT  .
	  {
	    infoMSG(3,3,"CPU FFT Input");

	    if ( batch->flags & CU_FFT_SEP_INP )
	    {
	      callInputFFTW(batch);
	    }
	    else
	    {
#pragma omp critical
	      callInputFFTW(batch);
	    }
	  }
	}

	FOLD // Copy CPU prepped data to the pagelocked input data  .
	{
	  infoMSG(3,3,"Copy buffer over to pinned memory");

	  FOLD // Synchronisation [ blocking ]  .
	  {
	    infoMSG(4,4,"blocking synchronisation on %s", "iDataCpyComp" );

	    PROF // Profiling  .
	    {
	      NV_RANGE_PUSH("EventSynch iDataCpyComp");
	    }

	    CUDA_SAFE_CALL(cudaEventSynchronize(batch->iDataCpyComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");

	    PROF // Profiling  .
	    {
	      NV_RANGE_POP(); // EventSynch
	    }
	  }

	  PROF // Profiling  .
	  {
	    NV_RANGE_PUSH("memcpy 1");

	    if ( batch->flags & FLAG_PROF )
	    {
	      gettimeofday(&start, NULL);
	    }
	  }

	  // Copy the buffer over the pinned memory
	  infoMSG(4,4,"1D synch memory copy H2H");
	  memcpy(batch->h_iData, batch->h_iBuffer, batch->inpDataSize );

	  PROF // Profiling  .
	  {
	    NV_RANGE_POP(); // memcpy 1

	    if ( batch->flags & FLAG_PROF )
	    {
	      gettimeofday(&end, NULL);
	      float time = (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
	      int idx = MIN(2, batch->noStacks-1);
	      batch->compTime[NO_STKS*COMP_GEN_MEM+idx] += time;
	    }
	  }
	}
      }
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP(); // CPU prep input

      if ( batch->flags & FLAG_PROF )
      {
	gettimeofday(&end0, NULL);

	float v1 =  (end0.tv_sec - start0.tv_sec) * 1e6 + (end0.tv_usec - start0.tv_usec);
	batch->compTime[batch->noStacks*COMP_GEN_CINP] += v1;
      }
    }
  }
}

void copyInputToDevice(cuFFdotBatch* batch)
{
  PROF // Profiling - Time previous components  .
  {
    if ( (batch->flags & FLAG_PROF) )
    {
      if ( (*batch->rAraays)[batch->rActive+1][0][0].numrs )
      {
	infoMSG(5,5,"Time previous components");

	// Copying Data to device
	timeEvents( batch->iDataCpyInit, batch->iDataCpyComp, &batch->compTime[NO_STKS*COMP_GEN_H2D],   "Copy to device");
      }
    }
  }

  if ( (*batch->rAraays)[batch->rActive][0][0].numrs ) // This is real data ie this isn't just a call to finish off asynchronous work
  {
    infoMSG(2,2,"Copy to device - Iteration %3i.", (*batch->rAraays)[batch->rActive][0][0].iteration);

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Inp Copy");
    }

    FOLD // Synchronisation  .
    {
      FOLD // Previous  .
      {
	infoMSG(5,5,"Synchronise stream %s on %s.\n", "inpStream", "multComp (batch and stacks)");

	// Wait for previous per-stack multiplications to finish
	for (int ss = 0; ss < batch->noStacks; ss++)
	{
	  cuFfdotStack* cStack = &batch->stacks[ss];
	  CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, cStack->multComp, 0), "Waiting for GPU to be ready to copy data to device.\n");
	}

	// Wait for batch multiplications to finish
	CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, batch->multComp, 0), "Waiting for GPU to be ready to copy data to device.\n");
      }

      // NOTE: don't have to wait for GPU input work as it is done in the same stream

      PROF // Profiling  .
      {
	if ( batch->flags & FLAG_PROF )
	{
	  FOLD // HACK  .
	  {
	    // This hack, adds a dummy kernel to delay the "real" kernels so they can be accurately timed with events

	    infoMSG(4,4,"HACK! - Adding GPU delay");

	    infoMSG(5,5,"Synchronise stream %s on %s.\n", "inpStream", "searchComp");
	    infoMSG(5,5,"Synchronise stream %s on %s.\n", "inpStream", "ifftComp");
	    infoMSG(5,5,"Synchronise stream %s on %s.\n", "inpStream", "ifftMemComp");

	    CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, batch->searchComp, 0), 	"Waiting for Search to complete\n");
	    CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, batch->stacks->ifftComp, 0), 	"Waiting for iFFT complete\n");
	    CUDA_SAFE_CALL(cudaStreamWaitEvent(batch->inpStream, batch->stacks->ifftMemComp, 0),"Waiting for Search to complete\n");

	    infoMSG(5,5,"Call Delay kernel.\n");
	    streamSleep(batch->inpStream, 3e5*batch->noStacks );	// The length of the delay is tuned to the number of stacks and is device dependent
	  }

	  infoMSG(5,5,"Event %s in %s.\n", "iDataCpyInit", "inpStream");
	  cudaEventRecord(batch->iDataCpyInit, batch->inpStream);
	}
      }
    }

    FOLD // Copy pinned memory to device  .
    {
      infoMSG(4,4,"1D async memory copy H2D");

      CUDA_SAFE_CALL(cudaMemcpyAsync(batch->d_iData, batch->h_iData, batch->inpDataSize, cudaMemcpyHostToDevice, batch->inpStream), "Failed to copy input data to device");
      CUDA_SAFE_CALL(cudaGetLastError(), "Copying input data to the device.");
    }

    FOLD // Synchronisation  .
    {
      infoMSG(5,5,"Event %s in %s.\n", "iDataCpyComp", "inpStream");
      cudaEventRecord(batch->iDataCpyComp,  batch->inpStream);

      if ( !(batch->flags & CU_NORM_GPU)  )
      {
	// Data has been normalised by CPU
	infoMSG(5,5,"Event %s in %s.\n", "normComp", "inpStream");
	cudaEventRecord(batch->normComp,      batch->inpStream);
      }

      if ( batch->flags & CU_INPT_FFT_CPU )
      {
	// Data has been FFT'ed by CPU
	infoMSG(5,5,"Event %s in %s.\n", "inpFFTinitComp (stacks)", "inpStream");
	for (int ss = 0; ss < batch->noStacks; ss++)
	{
	  cuFfdotStack* cStack = &batch->stacks[ss];
	  cudaEventRecord(cStack->inpFFTinitComp, batch->inpStream);
	}
      }
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP();		// Inp Copy
    }
  }
}

void prepInputGPU(cuFFdotBatch* batch)
{
  PROF // Profiling - Time previous components  .
  {
    if ( (batch->flags & FLAG_PROF) )
    {
      if ( (*batch->rAraays)[batch->rActive+1][0][0].numrs )
      {
	infoMSG(5,5,"Time previous components");

	// GPU Normalisation
	if ( batch->flags & CU_NORM_GPU )
	{
	  for (int stack = 0; stack < batch->noStacks; stack++)
	  {
	    cuFfdotStack* cStack = &batch->stacks[stack];

	    timeEvents( cStack->normInit, cStack->normComp, &batch->compTime[NO_STKS*COMP_GEN_NRM + stack ],    "Stack input normalisation");
	  }
	}

	// Input FFT
	if ( !(batch->flags & CU_INPT_FFT_CPU) )
	{
	  for (int stack = 0; stack < batch->noStacks; stack++)
	  {
	    cuFfdotStack* cStack = &batch->stacks[stack];

	    timeEvents( cStack->inpFFTinit, cStack->inpFFTinitComp, &batch->compTime[NO_STKS*COMP_GEN_FFT + stack ],    "Stack input FFT");
	  }
	}
      }
    }
  }

  if ( (*batch->rAraays)[batch->rActive][0][0].numrs )
  {
    infoMSG(2,2,"GPU prep input - Iteration %3i.", (*batch->rAraays)[batch->rActive][0][0].iteration);

    struct timeval start, end;  // Profiling variables

    PROF // Profiling  .
    {
      NV_RANGE_PUSH("GPU prep input");

      if ( batch->flags & FLAG_PROF )
      {
	gettimeofday(&start, NULL);
      }
    }

    FOLD // Normalise and spread on GPU  .
    {
      if ( batch->flags & CU_NORM_GPU )
      {
	infoMSG(3,4,"Normalise on device\n");

	cuFfdotStack* pStack = NULL;  // Previous stack

	PROF // Profiling  .
	{
	  NV_RANGE_PUSH("Norm");
	}

	for ( int stack = 0; stack < batch->noStacks; stack++)  // Loop over stacks  .
	{
	  infoMSG(3,5,"Stack %i\n", stack);

	  cuFfdotStack* cStack = &batch->stacks[stack];

	  FOLD // Synchronisation  .
	  {
	    // This iteration
	    infoMSG(5,5,"Synchronise stream %s on %s.\n", "inptStream", "iDataCpyComp");
	    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->inptStream, batch->iDataCpyComp, 0), "Waiting for GPU to be ready to copy data to device\n");

	    if ( batch->flags & FLAG_SYNCH )
	    {
	      // Wait for the search to complete before FFT'ing the next set of input
	      infoMSG(5,5,"Synchronise stream %s on %s.\n", "inptStream", "searchComp");
	      cudaStreamWaitEvent(cStack->inptStream, batch->searchComp, 0);

	      // Wait for iFFT to finish - In-mem search - I found that GPU compute interferes with D2D copy so wait for it to finish
	      infoMSG(5,5,"Synchronise stream %s on %s.\n", "inptStream", "ifftMemComp");
	      cudaStreamWaitEvent(cStack->inptStream, batch->stacks->ifftMemComp, 0);

	      // Wait for previous normalisation to complete
	      infoMSG(5,5,"Synchronise stream %s on %s.\n", "inptStream", "normComp (neighbours)");
	      if ( pStack != NULL )
	      {
		cudaStreamWaitEvent(cStack->inptStream, pStack->normComp, 0);
	      }
	    }

	    PROF // Profiling  .
	    {
	      if ( batch->flags & FLAG_PROF )
	      {
		infoMSG(5,5,"Event %s in %s.\n", "normInit", "inptStream");
		cudaEventRecord(cStack->normInit, cStack->inptStream);
	      }
	    }
	  }

	  FOLD // Call the kernel to normalise and spread the input data  .
	  {
	    normAndSpread(cStack->inptStream, batch, stack );
	  }

	  FOLD // Synchronisation  .
	  {
	    infoMSG(5,5,"Event %s in %s.\n", "normComp", "inptStream");
	    cudaEventRecord(cStack->normComp, cStack->inptStream);
	  }

	  pStack = cStack;
	}

	if ( batch->flags & FLAG_SYNCH ) // Wait for the last stack to complete normalisation  .
	{
	  cuFfdotStack* lStack = &batch->stacks[batch->noStacks -1];
	  CUDA_SAFE_CALL(cudaStreamWaitEvent(lStack->inptStream, lStack->normComp, 0), "Waiting for event normComp");
	  CUDA_SAFE_CALL(cudaEventRecord(batch->normComp, lStack->inptStream), "Recording for event inptStream");
	}

	PROF // Profiling  .
	{
	  NV_RANGE_POP(); // Norm
	}
      }
    }

    FOLD // FFT the input on the GPU  .
    {
      if ( !(batch->flags & CU_INPT_FFT_CPU) )
      {
	infoMSG(3,3,"GPU FFT\n");

	cuFfdotStack* pStack = NULL;  // Previous stack

	PROF // Profiling  .
	{
	  NV_RANGE_PUSH("GPU FFT");
	}

	for (int stackIdx = 0; stackIdx < batch->noStacks; stackIdx++)
	{
	  cuFfdotStack* cStack = &batch->stacks[stackIdx];

	  CUDA_SAFE_CALL(cudaGetLastError(), "Before input fft.");

	  FOLD // Synchronisation  .
	  {
	    infoMSG(5,5,"Synchronise stream %s on %s.\n", "inptStream", "normComp (stack and batch)");
    	    infoMSG(5,5,"Synchronise stream %s on %s.\n", "inptStream", "iDataCpyComp");

	    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftIStream, cStack->normComp,     0), "Waiting for event stack normComp");
	    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftIStream, batch->normComp,      0), "Waiting for event batch normComp");
	    CUDA_SAFE_CALL(cudaStreamWaitEvent(cStack->fftIStream, batch->iDataCpyComp,  0), "Waiting for event iDataCpyComp");

	    if ( batch->flags & FLAG_SYNCH )	// Synchronous execution  .
	    {
	      // Wait for the search to complete before FFT'ing the next set of input
	      infoMSG(5,5,"Synchronise stream %s on %s.\n", "fftIStream", "searchComp");
	      cudaStreamWaitEvent(cStack->fftIStream, batch->searchComp, 0);

	      // Wait for iFFT to finish - In-mem search - I found that GPU compute interferes with D2D copy so wait for it to finish
	      infoMSG(5,5,"Synchronise stream %s on %s.\n", "inptStream", "ifftMemComp");
	      cudaStreamWaitEvent(cStack->inptStream, batch->stacks->ifftMemComp, 0);

	      // Wait for previous FFT to complete
	      infoMSG(5,5,"Synchronise stream %s on %s.\n", "fftIStream", "inpFFTinitComp (neighbours)");
	      if ( pStack != NULL )
	      {
		cudaStreamWaitEvent(cStack->fftIStream, pStack->inpFFTinitComp, 0);
	      }

	      // Wait for all GPU normalisations to complete
	      infoMSG(5,5,"Synchronise stream %s on %s.\n", "inptStream", "normComp  (all)");
	      for (int stack2Idx = 0; stack2Idx < batch->noStacks; stack2Idx++)
	      {
		cuFfdotStack* stack2 = &batch->stacks[stackIdx];
		cudaStreamWaitEvent(cStack->fftIStream, stack2->normComp, 0);
	      }
	    }
	  }

	  FOLD // Do the FFT on the GPU  .
	  {
	    if ( batch->flags & CU_FFT_SEP_INP )
	    {
	      callInputCUFFT(batch, cStack);
	    }
	    else
	    {
#pragma omp critical
	      callInputCUFFT(batch, cStack);
	    }
	  }

	  pStack = cStack;
	}

	PROF // Profiling  .
	{
	  NV_RANGE_POP(); // FFT
	}
      }
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP(); // GPU prep input

      if ( batch->flags & FLAG_PROF )
      {
	gettimeofday(&end, NULL);

	float v1 =  (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
	batch->compTime[batch->noStacks*COMP_GEN_GINP] += v1;
      }
    }
  }
}

/** Initialise input data for a f-∂f plane(s)  ready for convolution  .
 * This:
 *  Normalises the chunk of input data
 *  Spreads it (interbinning)
 *  FFT it ready for convolution
 *
 * @param batch the batch to work with
 */
void prepInput(cuFFdotBatch* batch)
{
  CUDA_SAFE_CALL(cudaGetLastError(), "prepInput");

  prepInputCPU(batch);
  copyInputToDevice(batch);
  prepInputGPU(batch);
}
