#include "cuda_accel_SS.h"

#include <cufft.h>
#include <algorithm>

#include <thrust/sort.h>
#include <thrust/device_vector.h>


#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"
#include "cuda_accel_SS.h"


void add_and_search_CPU(cuFFdotBatch* batch )
{
  infoMSG(2,2,"Sum & Search CPU\n");

  // Profiling  variables
  struct timeval start, end;

  const int noStages    = batch->noHarmStages;
  const int noHarms     = batch->noGenHarms;
  const int noSteps     = batch->noSteps;
  const int64_t FLAGS   = batch->flags;
  const int zeroHeight  = batch->hInfos->noZ;

  float*      pwerPlnF[noHarms];
  fcomplexcu* pwerPlnC[noHarms];

  candPZs     candLists [noStages][noSteps];
  float       pow[noHarms][noSteps];
  short       iyP[noHarms];
  int         inds[noHarms];
  int         sliceSz   = 16;
  int         noSlices  = ceil( zeroHeight / (float)sliceSz );
  int         noCands   = 0;
  initCand*       cnd       = (initCand*)malloc(sizeof(initCand)*noSlices*batch->accelLen*noStages*noSteps);

  FOLD // Sum search data  .
  {
    PROF // Profiling  .
    {
      NV_RANGE_PUSH("CPU Sum & search");

      if ( batch->flags & FLAG_PROF )
	gettimeofday(&start, NULL);
    }

    FOLD // Prep - Initialise the x indices  .
    {
      int bace = 0;
      for ( int harm = 0; harm < noHarms; harm++ )                  // loop over harmonic  .
      {
	int stgIDX = batch->cuSrch->sIdx[harm];

	// TODO: Convert this cos host data has been moved to the r-array
	//pwerPlnF[stgIDX] = &((float*)batch->h_outData1)[bace];
	//pwerPlnC[stgIDX] = &((fcomplexcu*)batch->h_outData1)[bace];

	bace += batch->hInfos[harm].noZ * batch->stacks[batch->hInfos[harm].stackNo].stridePower * noSteps;
      }
    }

    for ( int ix = 0; ix < batch->accelLen; ix++ )
    {
      FOLD // Prep - Initialise the x indices  .
      {
	for ( int harm = 0; harm < noHarms; harm++ )                // loop over harmonic  .
	{
	  int stgIDX        = batch->cuSrch->sIdx[harm];
	  cuHarmInfo* hInf  = &batch->hInfos[stgIDX];

	  //// NOTE: the indexing below assume each plane starts on a multiple of noHarms
	  int   hIdx        = round( ix*hInf->harmFrac ) + hInf->kerStart;
	  inds[harm]        = hIdx;
	}
      }

      FOLD // Set the local and return candidate powers to zero  .
      {
	for ( int stage = 0; stage < noStages; stage++ )
	{
	  for ( int step = 0; step < noSteps; step++)               // Loop over steps  .
	  {
	    candLists[stage][step].value = batch->cuSrch->powerCut[stage] ;
	  }
	}
      }

      FOLD // Set hold values to zero
      {
	for ( int harm = 0; harm < noHarms; harm++ )
	{
	  iyP[harm] = -1;
	}
      }

      FOLD // Sum & Search - Ignore contaminated ends tid to start at correct spot  .
      {
	for( int y = 0, sy = 0; y < zeroHeight; y++, sy++ )         // Loop over the chunk  .
	{
	  float powers[noSteps];
	  for ( int step = 0; step < noSteps; step++)               // Loop over steps  .
	  {
	    powers[step] = 0;
	  }

	  for ( int stage = 0 ; stage < noStages; stage++)          // Loop over stages  .
	  {
	    short start         = STAGE_CPU[stage][0] ;
	    short end           = STAGE_CPU[stage][1] ;

	    for ( int harm = start; harm <= end; harm++ )         	// Loop over harmonics (batch) in this stage  .
	    {
	      int stgIDX            = batch->cuSrch->sIdx[harm];
	      cuFfdotStack* cStack  = &batch->stacks[ batch->hInfos[stgIDX].stackNo ];
	      int     ix1           = inds[harm] ;
	      int     ix2           = ix1;
	      short   iy1           = batch->cuSrch->yInds[ (zeroHeight+INDS_BUFF)*harm + y ];
#ifdef WITH_ITLV_PLN
	      cuHarmInfo* hInf      = &batch->hInfos[stgIDX];
#endif
	      if ( iyP[harm] != iy1 ) // Only read power if it is not the same as the previous  .
	      {
		for ( int step = 0; step < noSteps; step++ )        // Loop over steps  .
		{
		  int iy2;

		  FOLD // Calculate index  .
		  {
		    if        ( FLAGS & FLAG_ITLV_ROW )
		    {
		      ix2 = ix1 + step    * cStack->strideCmplx;
		      iy2 = iy1 * noSteps * cStack->strideCmplx;
		    }
#ifdef WITH_ITLV_PLN
		    else
		    {
		      iy2 = ( iy1 + step * hInf->noZ ) * cStack->strideCmplx ;
		    }
#else
		    else
		    {
		      fprintf(stderr, "ERROR: functionality disabled in %s.\n", __FUNCTION__);
		      exit(EXIT_FAILURE);
		    }
#endif
		  }

		  FOLD // Read powers  .
		  {
		    if      ( FLAGS & FLAG_CUFFT_CB_POW )
		    {
		      pow[harm][step]         = pwerPlnF[harm][ iy2 + ix2 ];
		    }
		    else
		    {
		      fcomplexcu cmpc         = pwerPlnC[harm][ iy2 + ix2 ];
		      pow[harm][step]         = cmpc.r * cmpc.r + cmpc.i * cmpc.i;
		    }
		  }

		}
		iyP[harm] = iy1;
	      }

	      for ( int step = 0; step < noSteps; step++)           // Loop over steps  .
	      {
		powers[step]  += pow[harm][step];
	      }
	    }

	    for ( int step = 0; step < noSteps; step++)             // Loop over steps  .
	    {
	      if ( powers[step] > candLists[stage][step].value )
	      {
		// This is our new max!
		candLists[stage][step].value  = powers[step];
		candLists[stage][step].z      = y;
	      }
	    }
	  }

	  if ( sy > sliceSz || y == zeroHeight - 1 )
	  {
	    FOLD // Add candidates to list  .
	    {
	      for ( int stage = 0 ; stage < noStages; stage++)      // Loop over stages  .
	      {
		for ( int step = 0; step < noSteps; step++)         // Loop over steps  .
		{
		  if ( candLists[stage][step].value > batch->cuSrch->powerCut[stage] )
		  {
		    rVals* rVal = &(*batch->rAraays)[batch->rActive][step][0];

		    int numharm   = (1<<stage);
		    double rr     = rVal->drlo + ix / (double) batch->hInfos->noResPerBin ;

		    //procesCanidate(batch, rr, y, candLists[stage][step].value, 0, stage, numharm );
		    cnd[noCands].numharm  = numharm;
		    cnd[noCands].power    = candLists[stage][step].value;
		    cnd[noCands].r        = rr;
		    cnd[noCands].sig      = 0;
		    cnd[noCands].z        = y;
		    noCands++;
		  }
		}
	      }
	    }

	    FOLD // Set the local and return candidate powers to zero  .
	    {
	      for ( int stage = 0; stage < noStages; stage++ )
	      {
		for ( int step = 0; step < noSteps; step++)         // Loop over steps  .
		{
		  candLists[stage][step].value = 0 ;
		}
	      }
	    }

	    sy = 0;
	  }
	}
      }
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP();

      if ( batch->flags & FLAG_PROF )
      {
	gettimeofday(&end, NULL);
	float v1 =  (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec) ;
	batch->compTime[NO_STKS*COMP_GEN_SS] += v1;
      }


    }
  }

  FOLD // Process candidates  .
  {
    PROF // Profiling  .
    {
      NV_RANGE_PUSH("CPU Process results");

      if ( batch->flags & FLAG_PROF )
	gettimeofday(&start, NULL);
    }

    for ( int c = 0; c < noCands; c++ )
    {
      int stage = log2((float)cnd[c].numharm);
      //procesCanidate(batch, cnd[c].r, cnd[c].z, cnd[c].power, 0, stage, cnd[c].numharm );
    }

    PROF // Profiling  .
    {
      NV_RANGE_POP();

      if ( batch->flags & FLAG_PROF )
      {
	gettimeofday(&end, NULL);
	float v2 =  (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
	batch->compTime[NO_STKS*COMP_GEN_STR] += v2;
      }


    }
  }

  free(cnd);
}
