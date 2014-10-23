#include "array.h"

extern "C"
{
#include "accel.h"
}

/*#undef USEMMAP*/

#ifdef USEMMAP
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

//#ifdef CUDA
#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
#include <cuda_profiler_api.h>

#include <sys/time.h>
#include <time.h>
//#endif


#include "utilstats.h"
#ifdef WITHOMP
#include <omp.h>
#endif

#ifdef USEDMALLOC
#include "dmalloc.h"
#endif

extern float calc_median_powers(fcomplex * amplitudes, int numamps);
//extern void zapbirds(double lobin, double hibin, FILE * fftfile, fcomplex * fft);

static void print_percent_complete(int current, int number, char *what, int reset)
{
   static int newper = 0, oldper = -1;

#ifdef CUDA
   if ( DBG_INP01 || DBG_INP02 || DBG_INP03 || DBG_INP04 || DBG_PLN01 || DBG_PLN02 || DBG_PLN03 || DBG_PLTPLN06 )
     return;
#endif

   if (reset) {
      oldper = -1;
      newper = 0;
   } else {
      newper = (int) (current / (float) (number) * 100.0);
      if (newper < 0)
         newper = 0;
      if (newper > 100)
         newper = 100;
      if (newper > oldper) {
         printf("\rAmount of %s complete = %3d%%", what, newper);
         fflush(stdout);
         oldper = newper;
      }
   }
}

/* Return x such that 2**x = n */
//static
inline int twon_to_index(int n)
{
   int x = 0;

   while (n > 1) {
      n >>= 1;
      x++;
   }
   return x;
}

static inline double calc_required_r(double harm_fract, double rfull)
/* Calculate the 'r' you need for subharmonic  */
/* harm_fract = harmnum / numharm if the       */
/* 'r' at the fundamental harmonic is 'rfull'. */
{
    return (int) (ACCEL_RDR * rfull * harm_fract + 0.5) * ACCEL_DR;
}

ffdotpows *subharm_ffdot_plane_DBG(int numharm, int harmnum,
                               double fullrlo, double fullrhi,
                               subharminfo * shi, accelobs * obs,
                               nDarray<1, float> *input,
                               nDarray<2, float> *complex
                               )
{
   int ii, lobin, hibin, numdata, nice_numdata, nrs, fftlen, binoffset;
   static int numrs_full = 0;
   float powargr, powargi;
   double drlo, drhi, harm_fract;
   ffdotpows *ffdot;
   fcomplex *data, **result;
   presto_datainf datainf;

   if (numrs_full == 0) {
      if (numharm == 1 && harmnum == 1) {
         numrs_full = ACCEL_USELEN;
      } else {
         printf("You must call subharm_ffdot_plane() with numharm=1 and\n");
         printf("harnum=1 before you use other values!  Exiting.\n\n");
         exit(0);
      }
   }
   ffdot = (ffdotpows *) malloc(sizeof(ffdotpows));

   /* Calculate and get the required amplitudes */

   harm_fract = (double) harmnum / (double) numharm;
   drlo = calc_required_r(harm_fract, fullrlo);
   drhi = calc_required_r(harm_fract, fullrhi);
   ffdot->rlo = (int) floor(drlo);
   ffdot->zlo = calc_required_z(harm_fract, obs->zlo);

   /* Initialize the lookup indices */
   if (numharm > 1 && !obs->inmem) {
      double rr, subr;
      for (ii = 0; ii < numrs_full; ii++) {
         rr = fullrlo + ii * ACCEL_DR;
         subr = calc_required_r(harm_fract, rr);
         shi->rinds[ii] = index_from_r(subr, ffdot->rlo);

         if( ii == 100 )
         {
           //printf("harm: %02i  rLow: %10.5f  ix: %i  drlo: %8.3f  srlo: %8.3f \n", int((double) harmnum / (double) numharm *16), fullrlo, (int)shi->rinds[ii], (float)ffdot->rlo, subr );
         }

      }
   }
   ffdot->rinds = shi->rinds;
   ffdot->numrs = (int) ((ceil(drhi) - floor(drlo))
                         * ACCEL_RDR + DBLCORRECT) + 1;
   if (numharm == 1 && harmnum == 1)
   {
      ffdot->numrs = ACCEL_USELEN;
   }
   else
   {
      if (ffdot->numrs % ACCEL_RDR)
      {
         ffdot->numrs = (ffdot->numrs / ACCEL_RDR + 1) * ACCEL_RDR;
      }
   }
   ffdot->numzs = shi->numkern;
   binoffset = shi->kern[0].kern_half_width;
   fftlen = shi->kern[0].fftlen;
   lobin = ffdot->rlo - binoffset;
   hibin = (int) ceil(drhi) + binoffset;
   numdata = hibin - lobin + 1;
   nice_numdata = next2_to_n(numdata);  // for FFTs
   data = get_fourier_amplitudes(lobin, nice_numdata, obs);
   if (!obs->mmap_file && !obs->dat_input && 0)
       printf("This is newly malloc'd!\n");

   // Normalize the Fourier amplitudes

   if (obs->nph > 0.0)
   {
       //  Use freq 0 normalization if requested (i.e. photons)
       double norm = 1.0 / sqrt(obs->nph);
       for (ii = 0; ii < numdata; ii++) {
           data[ii].r *= norm;
           data[ii].i *= norm;
       }
   }
   else if (obs->norm_type == 0)
   {
       //  old-style block median normalization
       float *powers;
       double norm;

       powers = gen_fvect(numdata);
       for (ii = 0; ii < numdata; ii++)
           powers[ii] = POWER(data[ii].r, data[ii].i);
       norm = 1.0 / sqrt(median(powers, numdata)/log(2.0));
       vect_free(powers);
       for (ii = 0; ii < numdata; ii++) {
           data[ii].r *= norm;
           data[ii].i *= norm;
       }
   }
   else
   {
       //  new-style running double-tophat local-power normalization
       float *powers, *loc_powers;

       powers = gen_fvect(nice_numdata);
       for (ii = 0; ii < nice_numdata; ii++) {
           powers[ii] = POWER(data[ii].r, data[ii].i);
       }
       loc_powers = corr_loc_pow(powers, nice_numdata);
       for (ii = 0; ii < numdata; ii++) {
           float norm = invsqrt(loc_powers[ii]);
           data[ii].r *= norm;
           data[ii].i *= norm;
       }
       vect_free(powers);
       vect_free(loc_powers);
   }

   /* Perform the correlations */

   result = gen_cmatrix(ffdot->numzs, ffdot->numrs);
   datainf = RAW;

   corrData* corrd = initCorrData();

   for (ii = 0; ii < ffdot->numzs; ii++)
   {
      nrs = corr_complex2(corrd,
                         data, numdata, datainf,
                         shi->kern[ii].data, fftlen, FFT,
                         result[ii], ffdot->numrs, binoffset,
                         ACCEL_NUMBETWEEN, binoffset, CORR);
      if (datainf == RAW )
      {
        if ( fftlen != input->ax(0)->noEls()/2.0 )
        {
          printf("ERROR: numdata != length on input data!\n");
        }
        memcpy(input->elems, corrd->dataarray, fftlen*2*sizeof(float) );
      }
      memcpy(complex->getP(0,ii), result[ii], nrs*2*sizeof(float) );
      //memcpy(complex->getP(0,ii), result[ii], fftlen*2*sizeof(float) );
      datainf = SAME;
   }

   // Always free data
   vect_free(data);
   clearCorrData(corrd);

   /* Convert the amplitudes to normalized powers */

   ffdot->powers = gen_fmatrix(ffdot->numzs, ffdot->numrs);
   for (ii = 0; ii < (ffdot->numzs * ffdot->numrs); ii++)
      ffdot->powers[0][ii] = POWER(result[0][ii].r, result[0][ii].i);
   vect_free(result[0]);
   vect_free(result);

   /*
   if (numharm == 1 && harmnum == 1 )
   {
       printf("\nstage:\t %i \t harm:\t %02i \t Power:\t %15.7f \t sum:\t %15.7f \n", 1, 1,  ffdot->powers[100][100],  ffdot->powers[100][100] );
   }
   */
   return ffdot;
}

int main(int argc, char *argv[])
{
   int ii;
   double ttim, utim, stim, tott;
   struct tms runtimes;

   accelobs obs;
   infodata idata;
   GSList *candsCPU = NULL;
   GSList *candsGPU = NULL;
   GSList *cands    = NULL;
   Cmdline *cmd;

   // Timing vars
   long long cupTime = 0, gpuTime = 0, optTime = 0;
   struct timeval start, end, timeval;


   /* Prep the timer */

   tott = times(&runtimes) / (double) CLK_TCK;

   /* Call usage() if we have no command line arguments */

   if (argc == 1) {
      Program = argv[0];
      printf("\n");
      usage();
      exit(1);
   }

   /* Parse the command line using the excellent program Clig */

   cmd = parseCmdline(argc, argv);

#ifdef DEBUG
   showOptionValues();
#endif

   printf("\n\n");
   printf("    Fourier-Domain Acceleration Search Routine\n");
   printf("               by Scott M. Ransom\n\n");

   /* Create the accelobs structure */
   create_accelobs(&obs, &idata, cmd, 1);

   /* Zap birdies if requested and if in memory */
   if (cmd->zaplistP && !obs.mmap_file && obs.fft) {
      int numbirds;
      double *bird_lobins, *bird_hibins, hibin;

      /* Read the Standard bird list */
      numbirds = get_birdies(cmd->zaplist, obs.T, cmd->baryv,
                             &bird_lobins, &bird_hibins);

      /* Zap the birdies */
      /*
      printf("Zapping them using a barycentric velocity of %.5gc.\n\n", cmd->baryv);
      hibin = obs.N / 2;
      for (ii = 0; ii < numbirds; ii++) {
         if (bird_lobins[ii] >= hibin)
            break;
         if (bird_hibins[ii] >= hibin)
            bird_hibins[ii] = hibin - 1;
         zapbirds(bird_lobins[ii], bird_hibins[ii], NULL, obs.fft);
      }
      */

      vect_free(bird_lobins);
      vect_free(bird_hibins);
   }

   printf("Searching with up to %d harmonics summed:\n",
          1 << (obs.numharmstages - 1));
   printf("  f = %.1f to %.1f Hz\n", obs.rlo / obs.T, obs.rhi / obs.T);
   printf("  r = %.1f to %.1f Fourier bins\n", obs.rlo, obs.rhi);
   printf("  z = %.1f to %.1f Fourier bins drifted\n\n", obs.zlo, obs.zhi);

   long long badInp  = 0;
   long long badCplx = 0;

   /* Start the main search loop */

   FOLD // the main loop
   {
      double startr = obs.rlo, lastr = 0, nextr = 0;
      ffdotpows *fundamental;
      int noHarms = (1 << (obs.numharmstages - 1));
      candsGPU = NULL;
      int dev;
      cuStackList* kernels;             // List of stacks with the kernels, one for each device being used
      cuStackList* master   = NULL;     // The first kernel stack created
      int nPlains           = 0;        // The number of plains
      int noKers            = 0;        // Real number of kernels/devices being used
      cuStackList* plainsj[10*5];       // List of pointers to each plain
      int noSteps = 0;


      //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

      subharminfo **subharminfs;

#ifdef CUDA
      nvtxRangePush("CPU");
      gettimeofday(&start, NULL); // Note could start the timer after kernel init
#endif

      /* Generate the CPU correlation kernels */

      printf("Generating CPU correlation kernels:\n");
      subharminfs = create_subharminfos(&obs);
      printf("Done generating kernels.\n\n");

      nDarray<2, float> DFF_kernels;

      FOLD // Generate the GPU kernel
      {
        cudaDeviceSynchronize();          // This is only necessary for timing
        gettimeofday(&start, NULL);       // Profiling
        cudaProfilerStart();              // Start profiling, only really necessary debug and profiling, surprise surprise

        if ( cmd->gpuP ) // Determine the index and number of devices
        {
          if ( cmd->gpuC == 0 )  // NB: Note using gpuC == 0 requires a change in accelsearch_cmd every time clig is run!!!!
          {
            // Make a list of all devices
            cmd->gpuC = getGPUCount();
            cmd->gpu = (int*) malloc( cmd->gpuC * sizeof(int) );
            for ( dev = 0 ; dev < cmd->gpuC; dev++ )
              cmd->gpu[dev] = dev;
          }
        }

        FOLD // Create a kernel on each device
        {
          kernels = (cuStackList*)malloc(cmd->gpuC*sizeof(cuStackList));
          int added;

          for ( dev = 0 ; dev < cmd->gpuC; dev++ ) // Loop over devices
          {
            int no;
            int noSteps;
            if ( dev >= cmd->nplainsC )
              no = cmd->nplains[cmd->nplainsC-1];
            else
              no = cmd->nplains[dev];

            if ( dev >= cmd->nplainsC )
              noSteps = cmd->nsteps[cmd->nplainsC-1];
            else
              noSteps = cmd->nsteps[dev];

            // Get width from ACCEL_USELEN
            float halfwidth       = z_resp_halfwidth(obs.zhi, LOWACC); /// The halfwidth of the maximum zmax, to calculate accel len
            float width           = ACCEL_USELEN + 2 + 2 * ACCEL_NUMBETWEEN * halfwidth;
            float width2          = ceil(log2(width));
            float width3          = pow(2, width2);
            float width4          = floor(width3/1000.0);


            added = initHarmonics(&kernels[noKers], master, obs.numharmstages, (int)obs.zhi, &obs, cmd->gpu[dev], noSteps, ACCEL_USELEN/*cmd->width*/, no );
            if ( added && (master == NULL) )
            {
              master = &kernels[0];
            }
            if ( added )
            {
              noKers++;
            }
            else
            {
              printf("Error: failed to set up a kernel on device %i, trying to continue... \n", cmd->gpu[dev]);
            }
          }
        }

        FOLD // Create plains for calculations
        {
          int pln;
          for ( dev = 0 ; dev < noKers; dev++)
          {
            int no;
            if ( dev >= cmd->nplainsC )
              no = cmd->nplains[cmd->nplainsC-1];
            else
              no = cmd->nplains[dev];

            for ( pln = 0 ; pln < no; pln++ )
            {
              plainsj[nPlains] = initPlains(&kernels[dev], pln, no-1);

              if ( plainsj[nPlains] == NULL)
              {
                if (pln == 0 )
                {
                  fprintf(stderr, "ERROR: Failed to create at least one stack for GPU search on device %i.\n", kernels[dev].device);
                  return -1;
                }
                break;
              }
              else
              {
                noSteps += plainsj[nPlains]->noSteps;
                nPlains++;
              }
            }
          }
        }
      }

      FOLD // Test kernels
      {
        int stage, harmtosum, harm;
        ffdotpows *subharmonic;

        printf("\nComparing GPU & CPU kernels    You can expect a MSE ~3.500e-14 from the differences in FFTW & CUFFT.\n");

        for (stage = 0; stage < obs.numharmstages; stage++)
        {
          harmtosum = 1 << stage;
          for (harm = 1; harm <= harmtosum; harm += 2)
          {
            nDarray<2, float> CPU_kernels;
            nDarray<2, float> GPU_kernels;

            float frac = (float)(harm)/(float)harmtosum;
            int idx = noHarms - frac * noHarms;

            cuHarmInfo *hinf  = &kernels[0].hInfos[idx];
            subharminfo *sinf0 = subharminfs[0];
            subharminfo *sinf1 = subharminfs[1];
            subharminfo *sinf = &subharminfs[stage][harm - 1];

            CPU_kernels.addDim(hinf->width*2, 0, hinf->width);
            CPU_kernels.addDim(hinf->height, -hinf->zmax, hinf->zmax);
            CPU_kernels.allocate();

            GPU_kernels.addDim(hinf->width*2, 0, hinf->width);
            GPU_kernels.addDim(hinf->height, -hinf->zmax, hinf->zmax);
            GPU_kernels.allocate();

            // Copy data from device
            CUDA_SAFE_CALL(cudaMemcpy(GPU_kernels.elems, kernels[0].kernels[idx].d_kerData, GPU_kernels.getBuffSize(), cudaMemcpyDeviceToHost), "Failed to kernrl copy data from.");
            //CUDA_SAFE_CALL(cudaDeviceSynchronize(),"Error synchronising");

            kernels[0].kernels[idx].d_kerData;

            int row;
            for (row=0; row < sinf->numkern; row++  )
            {
              memcpy(CPU_kernels.getP(0,row), sinf->kern[row].data, hinf->width*sizeof(fcomplex));
            }

            //printf("   Input: %02i (%.2f)    MSE: %15.10f  μ: %10.5f   σ: %10.5f\n", harz, trdStack->hInfos[harz].harmFrac, MSE, stat.mean, stat.sigma );

            basicStats stat = GPU_kernels.getStats(true);
            double MSE = GPU_kernels.MSE(CPU_kernels);
            double ERR =  MSE / stat.sigma ;
            printf("   Cmplx: %02i (%.2f)  MSE: %.3e    μ: %10.3e    σ: %.3e    MSE/σ: %.2e ", idx, frac, MSE, stat.mean, stat.sigma, ERR );

            if      ( ERR > 1e1   )
              printf("  BAD!    Not even in the same realm.\n");
            else if ( ERR > 1e0   )
              printf("  Bad.  \n" );
            else if ( ERR > 1e-3  )
              printf("  Bad.   But not that bad.\n");
            else if ( ERR > 1e-6 )
              printf("  Close  But not great. \n");
            else if ( ERR > 1e-9 )
              printf("  GOOD  But a bit high.\n");
            else if ( ERR > 1e-15 )
              printf("  GOOD \n"  );
            else if ( ERR > 1e-19 )
              printf("  GOOD  Very good.\n"  );
            else
              printf("  Great \n");
          }
        }
      }

      if ( cmd->gpuP >= 0)
      {
        printf("\nRunning GPU search with %i simultaneous families of f-∂f plains spread across %i device(s).\n\n", noSteps, noKers);

        omp_set_num_threads(nPlains);

        int harmtosum, harm;
        startr = obs.rlo, lastr = 0, nextr = 0;

        print_percent_complete(startr - obs.rlo, obs.highestbin - obs.rlo, "search", 1);

        int ss = 0;
        int maxxx = ( obs.highestbin - obs.rlo ) / (float)( master->accelLen * ACCEL_DR ) ;

        float ns = ( obs.highestbin - obs.rlo ) / (float)( master->accelLen * ACCEL_DR ) ;

        if ( maxxx < 0 )
          maxxx = 0;

//#pragma omp parallel // Note the CPU version is not set up to be thread capable so can't really test multi-threading
        {
          int tid = omp_get_thread_num();
          cuStackList* trdStack = plainsj[tid];

          nDarray<1, float> **cpuInput = new nDarray<1, float>*[trdStack->noSteps];
          nDarray<2, float> **cpuCmplx = new nDarray<2, float>*[trdStack->noSteps];
          nDarray<1, float> **gpuInput = new nDarray<1, float>*[trdStack->noSteps];
          nDarray<2, float> **gpuCmplx = new nDarray<2, float>*[trdStack->noSteps];

          FOLD // Initialise data structures to hold test data for comparisons
          {
            int hh;
            for(hh = 0; hh < trdStack->noSteps; ++hh)
            {
              cpuInput[hh] = new nDarray<1, float>[trdStack->noHarms];
              cpuCmplx[hh] = new nDarray<2, float>[trdStack->noHarms];

              gpuInput[hh] = new nDarray<1, float>[trdStack->noHarms];
              gpuCmplx[hh] = new nDarray<2, float>[trdStack->noHarms];
            }


            int stage, harmtosum, harm, si;
            printf("\n Creating data sets...\n");
            for ( si = 0; si < trdStack->noSteps ; si ++)
            {
              for (stage = 0; stage < obs.numharmstages; stage++)
              {
                harmtosum = 1 << stage;
                for (harm = 1; harm <= harmtosum; harm += 2)
                {
                  float frac = (float)(harm)/(float)harmtosum;
                  int idx = noHarms - frac * noHarms;

                  cuHarmInfo *hinf  = &kernels[0].hInfos[idx];

                  cpuInput[si][idx].addDim(hinf->width*2, 0, hinf->width);
                  cpuInput[si][idx].allocate();

                  cpuCmplx[si][idx].addDim(hinf->width*2, 0, hinf->width);
                  cpuCmplx[si][idx].addDim(hinf->height, -hinf->zmax, hinf->zmax);
                  cpuCmplx[si][idx].allocate();

                  gpuInput[si][idx].addDim(hinf->width*2, 0, hinf->width);
                  gpuInput[si][idx].allocate();

                  gpuCmplx[si][idx].addDim(hinf->width*2, 0, hinf->width);
                  gpuCmplx[si][idx].addDim(hinf->height, -hinf->zmax, hinf->zmax);
                  gpuCmplx[si][idx].allocate();
                }
              }
            }
          }

          double*  startrs = (double*)malloc(sizeof(double)*trdStack->noSteps);
          double*  lastrs  = (double*)malloc(sizeof(double)*trdStack->noSteps);

          setContext( trdStack ) ;

          //setYINDS( trdStack,  obs.numharmstages );

          int firstStep = 0;

          while ( ss < maxxx ) // Loop over the steps
          {
#pragma omp critical
            {
              firstStep = ss;
              ss       += trdStack->noSteps;
            }

            if ( firstStep >= maxxx )
              break;

            int si;
            for ( si = 0; si < trdStack->noSteps ; si ++)
            {
              startrs[si] = obs.rlo + (firstStep+si) * ( master->accelLen * ACCEL_DR );
              lastrs[si]  = startrs[si] + master->accelLen * ACCEL_DR - ACCEL_DR;
            }

            // Call the CUDA stuff
            ffdot_planeCU3(trdStack, startrs, lastrs, obs.norm_type, 1, (fcomplexcu*)obs.fft, &obs, &candsGPU);

            if ( trdStack->flag & CU_CAND_HOST )
              trdStack->h_bCands = &master->h_bCands[master->accelLen*obs.numharmstages*firstStep] ;

            FOLD // Copy data from device  .
            {
              ulong sz  = 0;
              harm      = 0;

              // Write data to page locked memory
              for (int ss = 0; ss < trdStack->noStacks; ss++)
              {
                cuFfdotStack* cStack = &trdStack->stacks[ss];

                // Synchronise
                //cudaStreamWaitEvent(cStack->fftPStream, cStack->plnComp, 0);

                for (int si = 0; si < cStack->noInStack; si++)
                {
                  cuHarmInfo* cHInfo    = &trdStack->hInfos[harm];      // The current harmonic we are working on
                  cuFFdot*    plan      = &cStack->plains[si];          // The curent plain

                  for (int step = 0; step < trdStack->noSteps; step++)
                  {
                    if ( trdStack->hInfos[harm].harmFrac == 0.5 )
                    {
                      int ZZ = 0;
                    }

                    // Copy input data from GPU
                    fcomplexcu *data = &trdStack->d_iData[sz];
                    //cudaStreamWaitEvent(cStack->fftIStream, trdStack->normComp, 0); // just encase we skip the FFT'ing
                    CUDA_SAFE_CALL(cudaMemcpyAsync(gpuInput[step][harm].elems, data, cStack->stride*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftIStream), "Failed to copy input data from device.");


                    for( int y = 0; y < cHInfo->height; y++ )
                    {
                      fcomplexcu *cmplxData;
                      if ( trdStack->flag & FLAG_STP_ROW )
                      {
                        cmplxData = &plan->d_plainData[(y*trdStack->noSteps + step)*cHInfo->stride ];
                      }
                      else if ( trdStack->flag & FLAG_STP_PLN )
                      {
                        cmplxData = &plan->d_plainData[(y + step*cHInfo->height)*cHInfo->stride ];
                      }

                      cmplxData += cHInfo->halfWidth*2;
                      //CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[step][harm].getP(0,y), cmplxData, (cHInfo->width-2*2*cHInfo->halfWidth)*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
                      CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[step][harm].getP(0,y), cmplxData, (plan->numrs[step])*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
                    }

                    sz += cStack->stride;
                  }
                  harm++;
                }
              }
            }

            //CUDA_SAFE_CALL(cudaDeviceSynchronize(),"Error synchronising");

            FOLD // Now do an equivalent CPU search  .
            {
              for ( si = 0; si < trdStack->noSteps ; si ++) // Loop over steps
              {
                startr  = startrs[si];
                lastr   = lastrs[si];

                fundamental = subharm_ffdot_plane_DBG(1, 1, startr, lastr, &subharminfs[0][0], &obs, &cpuInput[si][0], &cpuCmplx[si][0] );

                candsCPU = search_ffdotpows(fundamental, 1, &obs, candsCPU);

                if (obs.numharmstages > 1)    /* Search the subharmonics */
                {
                  int stage, harmtosum, harm;
                  ffdotpows *subharmonic;

                  // Copy the fundamental's ffdot plane to the full in-core one
                  if (obs.inmem)
                  {
                    if (cmd->otheroptP)
                      fund_to_ffdotplane_trans(fundamental, &obs);
                    else
                      fund_to_ffdotplane(fundamental, &obs);
                  }
                  for (stage = 1; stage < obs.numharmstages; stage++)
                  {
                    harmtosum = 1 << stage;

                    for (harm = 1; harm < harmtosum; harm += 2)
                    {
                      float frac = (float)(harm)/(float)harmtosum;
                      int idx = noHarms - frac * noHarms;

                      if (obs.inmem)
                      {
                        if (cmd->otheroptP)
                          inmem_add_ffdotpows_trans(fundamental, &obs, harmtosum, harm);
                        else
                          inmem_add_ffdotpows(fundamental, &obs, harmtosum, harm);
                      }
                      else
                      {
                        subharmonic = subharm_ffdot_plane_DBG(harmtosum, harm, startr, lastr,
                            &subharminfs[stage][harm - 1],
                            &obs, &cpuInput[si][idx], &cpuCmplx[si][idx] );

                        if (cmd->otheroptP)
                          add_ffdotpows_ptrs(fundamental, subharmonic, harmtosum, harm);
                        else
                          add_ffdotpows(fundamental, subharmonic, harmtosum, harm);
                        free_ffdotpows(subharmonic);
                      }
                    }
                    candsCPU = search_ffdotpows(fundamental, harmtosum, &obs, candsCPU);
                  }
                }
                free_ffdotpows(fundamental);

              }
            }

            FOLD // Print MSE  .
            {
              for ( si = 0; si < trdStack->noSteps ; si ++) // Loop over steps
              {
                printf("\n           ---- Step %03i of %03i ----\n",firstStep + si+1, maxxx);
                for ( int harz = 0; harz < trdStack->noHarms; harz++ )
                {
                  basicStats stat = gpuInput[si][harz].getStats(true);
                  double MSE = gpuInput[si][harz].MSE(cpuInput[si][harz]);
                  double ERR =  MSE / stat.sigma ;

                  printf("   Cmplx: %02i (%.2f)  MSE: %.3e    μ: %10.3e    σ: %.3e    MSE/σ: %.2e ", harz, trdStack->hInfos[harz].harmFrac, MSE, stat.mean, stat.sigma, ERR );

                  if ( ERR > 1e-6  )
                    badInp++;

                  if      ( ERR > 1e1   )
                    printf("  BAD!    Not even in the same realm.\n");
                  else if ( ERR > 1e0   )
                    printf("  Bad.  \n" );
                  else if ( ERR > 1e-3  )
                    printf("  Bad.   But not that bad.\n");
                  else if ( ERR > 1e-6 )
                    printf("  Close  But not great. \n");
                  else if ( ERR > 1e-9 )
                    printf("  GOOD  But a bit high.\n");
                  else if ( ERR > 1e-15 )
                    printf("  GOOD \n"  );
                  else if ( ERR > 1e-19 )
                    printf("  GOOD  Very good.\n"  );
                  else
                    printf("  Great \n");

                }
                printf("\n");
                for ( int harz = 0; harz < trdStack->noHarms; harz++ )
                {
                  basicStats stat = gpuCmplx[si][harz].getStats(true);
                  double MSE = gpuCmplx[si][harz].MSE(cpuCmplx[si][harz]);
                  double ERR =  MSE / stat.sigma ;

                  printf("   Cmplx: %02i (%.2f)  MSE: %.3e    μ: %10.3e    σ: %.3e    MSE/σ: %.2e ", harz, trdStack->hInfos[harz].harmFrac, MSE, stat.mean, stat.sigma, ERR );

                  if ( ERR > 1e-6  )
                    badCplx++;

                  if      ( ERR > 1e1   )
                    printf("  BAD!    Not even in the same realm.\n");
                  else if ( ERR > 1e0   )
                    printf("  Bad.  \n" );
                  else if ( ERR > 1e-3  )
                    printf("  Bad.   But not that bad.\n");
                  else if ( ERR > 1e-6 )
                    printf("  Close  But not great. \n");
                  else if ( ERR > 1e-9 )
                    printf("  GOOD  But a bit high.\n");
                  else if ( ERR > 1e-15 )
                    printf("  GOOD \n"  );
                  else if ( ERR > 1e-19 )
                    printf("  GOOD  Very good.\n"  );
                  else
                    printf("  Great \n");
                }
              }
            }

            print_percent_complete(startrs[0] - obs.rlo, obs.highestbin - obs.rlo, "search", 0);
          }

          int si;
          for ( si = 0; si < trdStack->noSteps ; si ++)
          {
            startrs[si] = 0;
            lastrs[si]  = 0;
          }

          // Finish searching the plains, this is required because of the out of order asynchronous calls
          int pln;
          for ( pln = 0 ; pln < 2; pln++ )
          {
            ffdot_planeCU3(trdStack, startrs, lastrs, obs.norm_type, 1, (fcomplexcu*)obs.fft, &obs, &candsGPU);
          }
        }

        printf("Done\n");

        if ( ( master->flag & CU_CAND_SINGLE_C ) == CU_CAND_SINGLE_G )
        {
          nvtxRangePush("Add to list");
          int cdx;
          int len = master->rHigh - master->rLow;
          long long numindep;

          double poww, sig, sigx, sigc, diff;
          double gpu_p, gpu_q;
          double rr, zz;
          int added = 0;
          int numharm;
          poww = 0;


          for (cdx = 0; cdx < len; cdx++)
          {
            poww        = master->h_candidates[cdx].power;

            if ( poww > 0 )
            {
              printf("ADD %i/%i\n", cdx, len);

              numharm   = master->h_candidates[cdx].numharm;
              numindep  = obs.numindep[twon_to_index(numharm)];
              sig       = master->h_candidates[cdx].sig;
              rr        = master->h_candidates[cdx].r;
              zz        = master->h_candidates[cdx].z;
              candsGPU  = insert_new_accelcand(candsGPU, poww, sig, numharm, rr, zz, &added);
            }
          }
          nvtxRangePop();
        }

        if ( master->flag & CU_CAND_DEVICE )
        {
          nvtxRangePush("Add to list");
          int len = master->rHigh - master->rLow;

          master->h_bCands = (accelcandBasic*)malloc(len*sizeof(accelcandBasic));
          CUDA_SAFE_CALL(cudaMemcpy(master->h_bCands, master->d_bCands, len*sizeof(accelcandBasic), cudaMemcpyDeviceToHost), "Failed to copy data to device");

          int cdx;
          long long numindep;

          double poww, sig, sigx, sigc, diff;
          double gpu_p, gpu_q;
          double rr, zz;
          int added = 0;
          int numharm;
          poww = 0;

          for (cdx = 0; cdx < len; cdx++)
          {
            sig        = master->h_bCands[cdx].sigma;

            if ( sig > 0 )
            {
              numharm   = master->h_bCands[cdx].numharm;
              numindep  = obs.numindep[twon_to_index(numharm)];
              rr        = cdx + master->rLow;
              zz        = master->h_bCands[cdx].z;
              candsGPU  = insert_new_accelcand(candsGPU, poww, sig, numharm, rr, zz, &added);
            }
          }
          nvtxRangePop();
        }

        //if ( time1 != 0 )
        //  printf("\n\nCopy %5.3f  Convolve %5.3f    %5.3f X  \n",time1 / 1000.0, time2 / 1000.0, time1/(double)time2 );

        cudaProfilerStop();

        print_percent_complete(obs.highestbin - obs.rlo, obs.highestbin - obs.rlo, "search", 0);
        cudaDeviceSynchronize();
        gettimeofday(&end, NULL);
        gpuTime += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));
        cands = candsGPU;

        printCands("GPU_Cands.csv", candsGPU);
      }

   }

   printf("\n\nDone searching.\n");
   printf("   We got %i bad input values.\n", badInp);
   printf("   We got %i bad complex plains.\n\n", badCplx);

   return (0);

   printf("\n\nDone searching.  Now optimizing each candidate.\n\n");

   if(0) /* Candidate list trimming and optimization */
   {
      int numcands;
      GSList *listptr;
      accelcand *cand;
      fourierprops *props;


#ifdef CUDA
      gettimeofday(&start, NULL);       // Profiling
      nvtxRangePush("CPU");
      gettimeofday(&start, NULL); // Note could start the timer after kernel init
#endif

      numcands = g_slist_length(cands);

      if (numcands) {

         /* Sort the candidates according to the optimized sigmas */

         cands = sort_accelcands(cands);

         /* Eliminate (most of) the harmonically related candidates */
         if ((cmd->numharm > 1) && !(cmd->noharmremoveP))
             eliminate_harmonics(cands, &numcands);

         /* Now optimize each candidate and its harmonics */

         print_percent_complete(0, 0, NULL, 1);
         listptr = cands;
         for (ii = 0; ii < numcands; ii++) {
            print_percent_complete(ii, numcands, "optimization", 0);
            cand = (accelcand *) (listptr->data);
            optimize_accelcand(cand, &obs);
            listptr = listptr->next;
         }
         print_percent_complete(ii, numcands, "optimization", 0);

         /* Calculate the properties of the fundamentals */

         props = (fourierprops *) malloc(sizeof(fourierprops) * numcands);
         listptr = cands;
         for (ii = 0; ii < numcands; ii++) {
            cand = (accelcand *) (listptr->data);
            /* In case the fundamental harmonic is not significant,  */
            /* send the originally determined r and z from the       */
            /* harmonic sum in the search.  Note that the derivs are */
            /* not used for the computations with the fundamental.   */
            calc_props(cand->derivs[0], cand->r, cand->z, 0.0, props + ii);
            /* Override the error estimates based on power */
            props[ii].rerr = (float) (ACCEL_DR) / cand->numharm;
            props[ii].zerr = (float) (ACCEL_DZ) / cand->numharm;
            listptr = listptr->next;
         }

         /* Write the fundamentals to the output text file */

         output_fundamentals(props, cands, &obs, &idata);

         /* Write the harmonics to the output text file */

         output_harmonics(cands, &obs, &idata);

         /* Write the fundamental fourierprops to the cand file */

         obs.workfile = chkfopen(obs.candnm, "wb");
         chkfwrite(props, sizeof(fourierprops), numcands, obs.workfile);
         fclose(obs.workfile);
         free(props);
         printf("\n\n");
      } else {
         printf("No candidates above sigma = %.2f were found.\n\n", obs.sigma);
      }

#ifdef CUDA
      nvtxRangePop();
      gettimeofday(&end, NULL);
      optTime += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));

      printf("\n CPU: %9.06f  GPU: %9.06f [%6.2f x]  Optimization: %9.06f \n", cupTime * 1e-6, gpuTime * 1e-6, cupTime / (double) gpuTime, optTime * 1e-6 );
#endif
   }

   /* Finish up */

   printf("Searched the following approx numbers of independent points:\n");
   printf("  %d harmonic:   %9lld\n", 1, obs.numindep[0]);
   for (ii = 1; ii < obs.numharmstages; ii++)
      printf("  %d harmonics:  %9lld\n", 1 << ii, obs.numindep[ii]);

   printf("\nTiming summary:\n");
   tott = times(&runtimes) / (double) CLK_TCK - tott;
   utim = runtimes.tms_utime / (double) CLK_TCK;
   stim = runtimes.tms_stime / (double) CLK_TCK;
   ttim = utim + stim;
   printf("    CPU time: %.3f sec (User: %.3f sec, System: %.3f sec)\n",
          ttim, utim, stim);
   printf("  Total time: %.3f sec\n\n", tott);

   printf("Final candidates in binary format are in '%s'.\n", obs.candnm);
   printf("Final Candidates in a text format are in '%s'.\n\n", obs.accelnm);

   free_accelobs(&obs);
   g_slist_foreach(cands, free_accelcand, NULL);
   g_slist_free(cands);
   return (0);
}
