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
 *  -
 *    Beginning of change log
 *    Working version un-numbed
 *
 *  2017-01-28 10:25
 *    Fixed bug in synchronous in-mem runs (added a block on event ifftMemComp)
 *    Added some debug messages on stream synchronisation on events
 *
 *  2017-01-29 08:20
 *    Added static functions to call both CPU and GPU input FFT's, these allow identical calls from non critical and non critical blocks
 *    Added non critical behaviour for CPU FFT calls
 *    Added some debug messages on stream synchronisation on events, yes even more!
 *    made CPU_Norm_Spread static
 *    Fixed bug in timing of CPU input
 *    
 *  2017-02-03
 *    Converted to use of clearRval
 *
 *  2017-02-18
 *    Different memory management for GPU normalisation
 *
 *  2017-04-24
 *    Moved plane specific input functions here
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
      int noRespPerBin = batch->conf->noResPerBin;

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
	    if ( batch->conf->normType == 0 )	// Block median normalisation  .
	    {
	      int startBin = rVal->lobin < 0 ? -rVal->lobin : 0 ;
	      int endBin   = rVal->lobin + rVal->numdata >= batch->cuSrch->fft->lastBin ? rVal->lobin + rVal->numdata - batch->cuSrch->fft->lastBin : rVal->numdata ;

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
		    if ( rVal->lobin+ii < batch->cuSrch->fft->firstBin || rVal->lobin+ii  >= batch->cuSrch->fft->lastBin ) // Zero Pad
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
		    NV_RANGE_POP("Powers");
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
		    NV_RANGE_POP("Median");
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
		  if ( rVal->lobin+ii < batch->cuSrch->fft->firstBin || rVal->lobin+ii  >= batch->cuSrch->fft->lastBin ) // Zero Pad
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
		  NV_RANGE_POP("Write");
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
    NV_RANGE_POP("CPU_Norm_Spread");
  }
}

/** Set the position of all the steps of a batch
 *
 * This sets the start of the first good bin of the fundamental plane of the first step
 * to the given value
 *
 * @param batch
 * @param firstR
 * @param iteration
 * @param firstStep
 */
ACC_ERR_CODE startBatchR (cuFFdotBatch* batch, double firstR, int iteration, int firstStep )
{
  infoMSG(4,4,"Set batch R-values starting at %9.3f", firstR);

  ACC_ERR_CODE ret = ACC_ERR_NONE;

  if ( batch->flags & FLAG_SS_31 )
  {
    // SAS 3.1 Kernel requires the value to be aligned
    double hold		= firstR;
    double devisNo	= batch->noGenHarms;
    firstR 		= round(firstR/devisNo)*devisNo;

    if ( firstR != hold )
    {
      infoMSG(5,5,"Auto aligning R-Vals");
      ret += ACC_ERR_ALIGHN;
    }
  }

  int validSteps = 0;

  for ( int batchStep = 0; batchStep < (int)batch->noSteps ; batchStep++ )
  {
    rVals* rVal = &(*batch->rAraays)[0][batchStep][0];
    clearRval(rVal);

    // Set the bounds - only the fundamental is needed
    rVal->drlo		= firstR + batchStep*( batch->accelLen / (double)batch->conf->noResPerBin );
    rVal->drhi		= rVal->drlo + ( batch->accelLen - 1 ) / (double)batch->conf->noResPerBin;

    if ( rVal->drlo < batch->cuSrch->sSpec->searchRHigh  )
    {
      validSteps++;

      // Set step and iteration for all harmonics
      for ( int harm = 0; harm < batch->noGenHarms; harm++)
      {
	rVal		= &(*batch->rAraays)[0][batchStep][harm];

	rVal->step	= firstStep + batchStep;
	rVal->iteration	= iteration;
      }
    }
    else
    {
      infoMSG(5,5,"R-Vals too large");

      // Not actually a valid step
      rVal->drlo	= 0;
      rVal->drhi	= 0;
      ret |= ACC_ERR_OVERFLOW;
    }
  }

  if (!validSteps)
    ret |= ACC_ERR_OUTOFBOUNDS;

  return ret;
}

ACC_ERR_CODE centerBatchR (cuFFdotBatch* batch, double r, int iteration, int firstStep )
{
  double stepWidth	= ( batch->accelLen / (double)batch->conf->noResPerBin ) ;
  double low		= r - stepWidth / 2.0 ;
  int ss 		= batch->noSteps / 2 ;
  double firstR		= low - ss * stepWidth ;

  return startBatchR (batch, firstR, iteration, firstStep );
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
    NV_RANGE_POP("Set R-Vals");
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
      NV_RANGE_POP("CPU FFT");

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

/** Check if a given section of input, is sufficient to calculate a range ff values
 *
 * This does not load the actual input
 * This check the input in the input data structure of the plane
 *
 * @param input		Pointer to the harmonic info data structure
 * @param r		The centre z of the area
 * @param z		The centre r of the area
 * @param rSize		The size, in bins of the area
 * @param zSize		The size, in bins of the area
 * @param noHarms	The number of harmonics
 * @param newInp	Set to 1 if new input is needed or 0 if the current input is good
 * @return		ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE chkInput( cuHarmInput* input, double r, double z, double rSize, double zSize, int noHarms, int* newInp)
{
  ACC_ERR_CODE	err	= ACC_ERR_NONE;

//  PROF // Profiling  .
//  {
//    NV_RANGE_PUSH("Harm INP");
//  }

  infoMSG(4,4,"Check if new input is needed.\n");

  if ( newInp && input )
  {
    double	maxZ	= (z + zSize/2.0);
    double	minZ	= (z - zSize/2.0);
    double	maxR	= (r + rSize/2.0);
    double	minR	= (r - rSize/2.0);

    infoMSG(5,5,"Current r [ %.1f - %.1f ] z [ %.2f - %.2f ]  New r [ %.1f - %.1f ] z [ %.2f - %.2f ]\n", input->minR, input->maxR, input->minZ, input->maxZ, minR, maxR, minZ, maxZ );

    // initialise to zero
    *newInp = 0;

    //CUDA_SAFE_CALL(cudaGetLastError(), "Entering ffdotPln.");

    int halfWidth	= cu_z_resp_halfwidth_high<double>( MAX(fabs(maxZ*noHarms), fabs(minZ*noHarms)) );

    int	datStart;		// The start index of the input data
    int	datEnd;			// The end   index of the input data

    *newInp		= 0;	// Flag whether new input is needed

    if ( noHarms > input->noHarms )
    {
      infoMSG(6,6,"New = True - Harmonics greater.\n");
      *newInp = 1;
    }

    // Determine if new input is needed
    for( int h = 0; (h < noHarms) && !(*newInp) ; h++ )
    {
      // Note we use the largest possible halfWidth from the last harmonic
      datStart        = floor( minR*(h+1) - halfWidth );
      datEnd          = ceil(  maxR*(h+1) + halfWidth );

//      if ( datStart > fft->lastBin || datEnd <= fft->firstBin )
//      {
//	if ( h == 0 )
//	{
//	  fprintf(stderr, "ERROR: Harmonic input beyond scope of the FFT?");
//	}
//	else
//	{
//	  // Harmonic is beyond the scope of the FFT
//	  // This is strange because the input thinks is contains that data?
//	  fprintf(stderr, "ERROR: Harmonic input has data beyond the end of the FFT?");
//	}
//
//	*newInp = 1; // Suggest new input so that that the function that gets the data will error
//	err += ACC_ERR_OUTOFBOUNDS;
//	break;
//      }

      if ( datStart < input->loR[h] )
      {
	infoMSG(6,6,"New = True - Input harm %2i  start %i < %i .\n", h+1, datStart, input->loR[h] );
	*newInp = 1;
      }
      else if ( input->loR[h] + input->stride < datEnd )
      {
	infoMSG(6,6,"New = True - Input harm %2i  end %i > %i .\n", h+1, datEnd, input->loR[h] + input->stride );
	*newInp = 1;
      }
    }

    if (!*newInp)
      infoMSG(5,5,"Current input is good.\n");
  }
  else
  {
    infoMSG(6,6,"ERROR: NULL pointer.\n" );
    err += ACC_ERR_NULL;
    *newInp = 0;
  }

//  PROF // Profiling  .
//  {
//    NV_RANGE_POP("Harm INP");
//  }

  return err;
}


 /** Copy relevant input from FFT to data structure normalising as needed
 *
 *  Note this contains a blocking synchronisation to make sure the pinned host memory is free.
 *  If preWrite is set to null no blocking will be done
 *
 * @param input		The input data structure to "fill"
 * @param fft		The FFT data that will make up the input
 * @param r		The Centre r value of the fundamental
 * @param z		The Centre z value of the fundamental
 * @param rSize
 * @param zSize
 * @param noHarms	The number of harmonics to load into the memory
 * @param flags		bit flags used to determine normalisation
 * @param preWrite	A event to block on before writing - if NULL no blocking will be done
 * @return		ACC_ERR_NONE on success or a collection of error values if full or partial failure
 */
ACC_ERR_CODE loadHostHarmInput( cuHarmInput* input, fftInfo* fft, double r, double z, double rSize, double zSize, int noHarms, int64_t flags, cudaEvent_t* preWrite )
{
  ACC_ERR_CODE	err		= ACC_ERR_NONE;

  infoMSG(5,5,"Loading host harmonic memory.\n");

  if ( input && fft )
  {
    int	datStart;		// The start index of the input data
    int	datEnd;			// The end   index of the input data
    double largestZ;

//    PROF	// Profiling  .
//    {
//      NV_RANGE_PUSH("Prep Input");
//    }

    FOLD // Calculate normalisation factor  .
    {
      input->maxZ		= (z + zSize/2.0);
      input->minZ		= (z - zSize/2.0);
      input->maxR		= (r + rSize/2.0);
      input->minR		= (r - rSize/2.0);

      largestZ			= MAX(fabs(input->maxZ*noHarms), fabs(input->minZ*noHarms)) + 4 ; 		// NOTE this include the + 4 of original accelsearch this is not the end of the world as this is just the check

      input->noHarms		= noHarms;
      input->maxHalfWidth	= cu_z_resp_halfwidth_high<double>( largestZ );
      input->stride		= ceil((input->maxR+OPT_INP_BUF)*noHarms  + input->maxHalfWidth) - floor((input->minR-OPT_INP_BUF)*noHarms - input->maxHalfWidth);
      if(input->gInf)
	input->stride		= getStride(input->stride, sizeof(cufftComplex), input->gInf->alignment);

      FOLD // Calculate normalisation factor  .
      {
	PROF // Profiling  .
	{
	  NV_RANGE_PUSH("Calc Norm factor");
	}

	for ( int h = 1; h <= noHarms; h++ )
	{
	  int hIdx = h-1;

	  datStart	= floor( input->minR*(h) - input->maxHalfWidth );
	  datEnd	= ceil(  input->maxR*(h) + input->maxHalfWidth );

	  if ( datStart > fft->lastBin || datEnd <= fft->firstBin )
	  {
	    infoMSG(6,6,"ERROR: Max harms %2i - out of bounds.\n", h);

	    input->noHarms = h; // Use previous harmonic
	    err += ACC_ERR_OUTOFBOUNDS;
	    break;
	  }

	  char mthd[20];	// Normalisation text
	  if      ( flags & FLAG_OPT_NRM_LOCAVE   )
	  {
	    input->norm[hIdx]  = get_localpower3d(fft->data, fft->noBins, (r-fft->firstBin)*h, z*h, 0.0);
	    sprintf(mthd,"2D Avle");
	  }
	  else if ( flags & FLAG_OPT_NRM_MEDIAN1D )
	  {
	    input->norm[hIdx]  = get_scaleFactorZ(fft->data, fft->noBins, (r-fft->firstBin)*h, z*h, 0.0);
	    sprintf(mthd,"1D median");
	  }
	  else if ( flags & FLAG_OPT_NRM_MEDIAN2D )
	  {
	    fprintf(stderr,"ERROR: 2D median normalisation has not been written yet.\n");
	    sprintf(mthd,"2D median");
	    exit(EXIT_FAILURE);
	  }
	  else
	  {
	    // No normalisation this is plausible but not recommended
	    // NOTE: Perhaps this should not be the default?
	    input->norm[hIdx] = 1;
	    sprintf(mthd,"None");
	  }
	  infoMSG(6,6,"Harm %2i %s normalisation factor: %6.4f  r [ %.1f - %.1f ]   z [ %.2f - %.2f ] \n", h, mthd, input->norm[hIdx], input->minR*(h), input->maxR*(h), input->minZ*(h), input->maxZ*(h) );
	}

	PROF // Profiling  .
	{
	  NV_RANGE_POP("Calc Norm factor");
	}
      }
    }

    infoMSG(5,5,"New input, stride %i - covers r [%.1f - %.1f]  z [%.1f - %.1f] at a max HW of %i.\n", input->stride, input->minR, input->maxR, input->minZ, input->maxZ, input->maxHalfWidth );

    if ( input->stride*input->noHarms*sizeof(cufftComplex) > input->size )
    {
	int width1 = ceil((input->maxR+OPT_INP_BUF)*noHarms) - floor((input->minR-OPT_INP_BUF)*noHarms);
	int strd   = input->stride - width1 - 2*input->maxHalfWidth;

	infoMSG(6,6,"ERROR: Stride %i Harms %i = %i points for zmax: %.4f -> hw: %i   + %i + %i\n", input->stride, input->noHarms, input->stride*input->noHarms, largestZ, input->maxHalfWidth, width1, strd );

	fprintf(stderr, "ERROR: In function %s, harmonic input not created with large enough memory buffer, require %.2f MB  have %.2f.\n", __FUNCTION__, input->stride*noHarms*sizeof(cufftComplex)*1e-6, input->size*1e-6 );
	err += ACC_ERR_MEM;
	input->maxHalfWidth	= 0;
	input->maxZ		= 0;
	input->minZ		= 0;
	input->maxR		= 0;
	input->minR		= 0;
	input->noHarms		= 0;
	NV_RANGE_POP("ERROR");
	return err;
    }

    if ( preWrite ) // A blocking synchronisation to make sure we can write to host memory  .
    {
      infoMSG(4,4,"Blocking synchronisation on %s", "preWrite" );

      CUDA_SAFE_CALL(cudaEventSynchronize(*preWrite), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");
    }

    FOLD // Normalise host  .
    {
      int		off;		// Offset

      infoMSG(5,5,"Normalising input in host memory.\n" );

      // Calculate values for harmonics and normalise input
      for( int h = 0; h < noHarms; h++)
      {
	datStart	= floor( input->minR*(h+1) - input->maxHalfWidth );
	datEnd		= ceil(  input->maxR*(h+1) + input->maxHalfWidth );

	FOLD // Normalise input and Write data to host memory  .
	{
	  int startV = MIN( ((datStart + datEnd - input->stride ) / 2.0), datStart ); //Start value if the data is centred

	  input->loR[h]		= startV;
	  double factor		= sqrt(input->norm[h]);		// Correctly normalise input by the sqrt of the local power

	  for ( int i = 0; i < input->stride; i++ )		// Normalise input  .
	  {
	    off = startV - fft->firstBin + i;

	    if ( off >= 0 && off < fft->noBins )
	    {
	      input->h_inp[h*input->stride + i].r = fft->data[off].r / factor ;
	      input->h_inp[h*input->stride + i].i = fft->data[off].i / factor ;
	    }
	    else
	    {
	      input->h_inp[h*input->stride + i].r = 0;
	      input->h_inp[h*input->stride + i].i = 0;
	    }
	  }
	}
      }
    }

//    PROF // Profiling  .
//    {
//      NV_RANGE_POP("Prep Input");
//    }
  }
  else
  {
    infoMSG(6,6,"ERROR: NULL pointer.\n" );
    err += ACC_ERR_NULL;
  }

  return err;
}


cuHarmInput* initHarmInput( int maxWidth, float zMax, int maxHarms, gpuInf* gInf )
{
  size_t maxHalfWidth	= cu_z_resp_halfwidth<double>( zMax, HIGHACC );
  size_t inData	= (maxWidth + OPT_INP_BUF*2)*maxHarms;				// Data is the width of the highest harmonic
  size_t inpSz	= (inData + 2*maxHalfWidth)*maxHarms * sizeof(cufftComplex);

  infoMSG(7,7,"Input stride %i Harms %i points for zmax: %.1f -> hw: %i \n", inData + 2*maxHalfWidth, maxHarms, zMax, maxHalfWidth );

  return initHarmInput( inpSz, gInf );
}

cuHarmInput* initHarmInput( size_t memSize, gpuInf* gInf )
{
  size_t freeMem, totalMem;

  infoMSG(4,4,"New harmonic input.\n");

  cuHarmInput* input	= (cuHarmInput*)malloc(sizeof(cuHarmInput));
  memset(input, 0, sizeof(cuHarmInput));

  CUDA_SAFE_CALL(cudaMemGetInfo ( &freeMem, &totalMem ), "Getting Device memory information");
#ifdef MAX_GPU_MEM
  long  Diff = totalMem - MAX_GPU_MEM;
  if( Diff > 0 )
  {
    freeMem  -= Diff;
    totalMem -= Diff;
  }
#endif

  if ( memSize > freeMem )
  {
    printf("Not enough GPU memory to create any more stacks.\n");
    return NULL;
  }
  else
  {
    infoMSG(6,6,"Memory size %.2f MB.\n", memSize*1e-6 );

    // Allocate device memory
    CUDA_SAFE_CALL(cudaMalloc(&input->d_inp, memSize), "Failed to allocate device memory for kernel stack.");

    // Allocate host memory
    CUDA_SAFE_CALL(cudaMallocHost(&input->h_inp, memSize), "Failed to allocate device memory for kernel stack.");

    input->size = memSize;
    input->gInf = gInf;
  }

  return input;
}

ACC_ERR_CODE freeHarmInput(cuHarmInput* inp)
{
  ACC_ERR_CODE err = ACC_ERR_NONE;
  if ( inp )
  {
    cudaFreeNull(inp->d_inp);
    //cudaFreeHostNull(inp->h_inp);
    freeNull(inp->h_inp);			// This is not cuda free host because the memory isn't always pined.
    freeNull(inp);
  }

  return err;
}

/** Duplicate the host (CPU) data of a harmonic input data structure
 *
 *  This makes a copy of the host memory
 *  The returned data structure is free of GPU information
 *
 * @param orr	The structure to duplicate
 * @return	NULL on error or A pointer to a new harmonic input structure, note this must be freed manually (including host memry).
 */
cuHarmInput* duplicateHostInput(cuHarmInput* orr)
{
  if ( orr )
  {
    PROF // Profiling  .
    {
      NV_RANGE_PUSH("Duplicate Hamr Inp");
    }

    size_t sz = MIN(orr->size, orr->noHarms * orr->stride * sizeof(fcomplexcu) * 2.0 );
    cuHarmInput* res = (cuHarmInput*)malloc(sizeof(cuHarmInput));

    infoMSG(5,5,"Duplicating harmonic input, %.2f MB\n", sz*1e-6);

    memcpy(res, orr, sizeof(cuHarmInput));

    // Clear GPU dependent items
    res->d_inp	= NULL;
    res->gInf	= NULL;

    // Duplicate host memory
    res->h_inp	= (fcomplexcu*)malloc(sz);
    memcpy(res->h_inp, orr->h_inp, res->noHarms * res->stride * sizeof(fcomplexcu));
    res->size	= sz;

    PROF // Profiling  .
    {
      NV_RANGE_POP("Duplicate Hamr Inp");
    }

    return res;
  }
  else
  {
    return NULL;
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

    fcomplexcu* fft = (fcomplexcu*)batch->cuSrch->fft->data;

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
	  infoMSG(4,4,"Blocking synchronisation on %s", "iDataCpyComp" );

	  PROF // Profiling  .
	  {
	    NV_RANGE_PUSH("EventSynch iDataCpyComp");
	  }

	  CUDA_SAFE_CALL(cudaEventSynchronize(batch->iDataCpyComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");

	  PROF // Profiling  .
	  {
	    NV_RANGE_POP("EventSynch");
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
	  NV_RANGE_POP("cpy inpt FFT");

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
	    NV_RANGE_POP("Zero buffer");

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
	    infoMSG(4,4,"Blocking synchronisation on %s", "iDataCpyComp" );

	    PROF // Profiling  .
	    {
	      NV_RANGE_PUSH("EventSynch iDataCpyComp");
	    }

	    CUDA_SAFE_CALL(cudaEventSynchronize(batch->iDataCpyComp), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");

	    PROF // Profiling  .
	    {
	      NV_RANGE_POP("EventSynch");
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
	    NV_RANGE_POP("memcpy 1");

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
      NV_RANGE_POP("CPU prep input");

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
      NV_RANGE_POP("Inp Copy");
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
	  NV_RANGE_POP("Norm");
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
	  NV_RANGE_POP("FFT");
	}
      }
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP("GPU prep input");

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

