#include "accel.h"

/*#undef USEMMAP*/

//#define CUDA // TMP

#ifdef USEMMAP
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

#ifdef CUDA
#include "cuda_accel.h"

#include <sys/time.h>
#include <time.h>
#endif

#ifdef WITHOMP
#include <omp.h>
#endif

#ifdef USEDMALLOC
#include "dmalloc.h"
#endif

//extern float calc_median_powers(fcomplex * amplitudes, int numamps);
extern void zapbirds(double lobin, double hibin, FILE * fftfile, fcomplex * fft);

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


inline int twon_to_index(int n) // TODO: fix this to be called from one place (ie not static in c file)
{
  int x = 0;

  while (n > 1)
  {
    n >>= 1;
    x++;
  }
  return x;
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
   long long prepTime = 0, cupTime = 0, gpuTime = 0, optTime = 0;
   struct timeval start, end, timeval;   

   /* Prep the timer */

   tott = times(&runtimes) / (double) CLK_TCK;

#ifdef CUDA // Profiling  .
   gettimeofday(&start, NULL);       // Profiling
   nvtxRangePush("Prep");
   gettimeofday(&start, NULL);       // Note could start the timer after kernel init
#endif

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

#ifdef CUDA // List GPU's  .
   if (cmd->lsgpuP)
   {
     listDevices();
     return (0);
   }
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
      printf("Zapping them using a barycentric velocity of %.5gc.\n\n", cmd->baryv);
      hibin = obs.N / 2;
      for (ii = 0; ii < numbirds; ii++) {
         if (bird_lobins[ii] >= hibin)
            break;
         if (bird_hibins[ii] >= hibin)
            bird_hibins[ii] = hibin - 1;
         zapbirds(bird_lobins[ii], bird_hibins[ii], NULL, obs.fft);
      }

      vect_free(bird_lobins);
      vect_free(bird_hibins);
   }

   printf("Searching with up to %d harmonics summed:\n",
          1 << (obs.numharmstages - 1));
   printf("  f = %.1f to %.1f Hz\n", obs.rlo / obs.T, obs.rhi / obs.T);
   printf("  r = %.1f to %.1f Fourier bins\n", obs.rlo, obs.rhi);
   printf("  z = %.1f to %.1f Fourier bins drifted\n\n", obs.zlo, obs.zhi);


#ifdef CUDA  // Profiling  .
   nvtxRangePop();
   gettimeofday(&end, NULL);
   prepTime += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));
   cuSearch*     cuSrch;
#endif

   char candsFile[1024];
   sprintf(candsFile,"%s_hs%02i_zmax%06.1f_sig%06.3f.unoptcands", obs.rootfilenm, obs.numharmstages, obs.zhi, obs.sigma );
   FILE *file;
   if ( file = fopen(candsFile, "rb") && 0 ) 	// Read candidates from previous search  .
   {
     printf("\nReading raw candies from \"%s\" previous search.\n", candsFile);
     int numcands;
     fread( &numcands, sizeof(numcands), 1, file );
     int nc = 0;

     //accelcand* newCnd = (accelcand*)malloc(sizeof(accelcand)*numcands);
     //fread( newCnd, sizeof(accelcand), 1, file );

     for (nc = 0; nc < numcands; nc++)
     {
       accelcand* newCnd = (accelcand*)malloc(sizeof(accelcand));
       fread( newCnd, sizeof(accelcand), 1, file );

       cands=insert_accelcand(cands,newCnd);
     }
     fclose(file);
   }
   else                                       // Run Search  .
   {

     /* Start the main search loop */

     FOLD // the main loop  .
     {
       double startr = obs.rlo, lastr = 0, nextr = 0;
       ffdotpows *fundamental;

       if ( cmd->cpuP )
       {
#ifdef CUDA // Profiling  .
         nvtxRangePush("CPU");
         gettimeofday(&start, NULL); // Note could start the timer after kernel init
#endif

         subharminfo **subharminfs;

         /* Generate the correlation kernels */

         printf("Generating correlation kernels:\n");
         subharminfs = create_subharminfos(&obs);
         printf("Done generating kernels.\n\n");
         printf("Starting the search.\n");
         /* Don't use the *.txtcand files on short in-memory searches */
         if (!obs.dat_input) {
           printf("  Working candidates in a test format are in '%s'.\n\n",
               obs.workfilenm);
         }

         printf("\n------------------------\nDoing CPU Search\n------------------------\n");

         print_percent_complete(startr - obs.rlo, obs.highestbin - obs.rlo, "search", 1);

         while (startr + ACCEL_USELEN * ACCEL_DR < obs.highestbin) {
           /* Search the fundamental */
           print_percent_complete(startr - obs.rlo,
               obs.highestbin - obs.rlo, "search", 0);
           nextr = startr + ACCEL_USELEN * ACCEL_DR;
           lastr = nextr - ACCEL_DR;
           fundamental = subharm_ffdot_plane(1, 1, startr, lastr,
               &subharminfs[0][0], &obs);
           candsCPU = search_ffdotpows(fundamental, 1, &obs, candsCPU);

           if (obs.numharmstages > 1) {   /* Search the subharmonics */
             int stage, harmtosum, harm;
             ffdotpows *subharmonic;

             // Copy the fundamental's ffdot plane to the full in-core one
             if (obs.inmem){
               if (cmd->otheroptP)
                 fund_to_ffdotplane_trans(fundamental, &obs);
               else
                 fund_to_ffdotplane(fundamental, &obs);
             }
             for (stage = 1; stage < obs.numharmstages; stage++) {
               harmtosum = 1 << stage;
               for (harm = 1; harm < harmtosum; harm += 2) {
                 if (obs.inmem) {
                   if (cmd->otheroptP)
                     inmem_add_ffdotpows_trans(fundamental, &obs, harmtosum, harm);
                   else
                     inmem_add_ffdotpows(fundamental, &obs, harmtosum, harm);
                 } else {
                   subharmonic = subharm_ffdot_plane(harmtosum, harm, startr, lastr,
                       &subharminfs[stage][harm - 1],
                       &obs);
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
           startr = nextr;
         }
         print_percent_complete(obs.highestbin - obs.rlo,
             obs.highestbin - obs.rlo, "search", 0);

#ifdef CUDA  // Profiling  .
         gettimeofday(&end, NULL);
         cupTime += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));
         printf("  cupTime %f\n", cupTime/1000.0);
         cands = candsCPU;

         nvtxRangePop();

#ifndef DEBUG
         printCands("CPU_Cands.csv", candsCPU);
#endif

         free_subharminfos(&obs, subharminfs);
#endif
       }

#ifdef CUDA   // --=== The cuda Search == --  .
       if ( cmd->gpuP > 0 )
       {
         candsGPU = NULL;

         //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
         //cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

         int noHarms = (1 << (obs.numharmstages - 1));

         FOLD
         {
           //cudaDeviceSynchronize();          // This is only necessary for timing
           gettimeofday(&start, NULL);       // Profiling
           //cudaProfilerStart();              // Start profiling, only really necessary debug and profiling, surprise surprise

           printf("\n------------------------\nDoing GPU Search \n------------------------\n");

           long noCands = 0;

           gpuSpecs      gSpec;
           searchSpecs   sSpec;

           gSpec       = readGPUcmd(cmd);
           sSpec       = readSrchSpecs(cmd, &obs);
           cuSrch      = initCuSearch(&sSpec, &gSpec, NULL);

           cuFFdotBatch* master    = &cuSrch->mInf->kernels[0];   // The first kernel created holds global variables

           printf("\nRunning GPU search with %i simultaneous f-∂f plains spread across %i device(s).\n\n", cuSrch->mInf->noSteps, cuSrch->mInf->noDevices );

           omp_set_num_threads(cuSrch->mInf->noBatches);

           startr = obs.rlo, lastr = 0, nextr = 0;
           int ss = 0;
           int maxxx = ( obs.highestbin - obs.rlo ) / (float)( cuSrch->mInf->kernels[0].accelLen * ACCEL_DR ) ; // The number of plains to make

           if ( maxxx < 0 )
             maxxx = 0;

           print_percent_complete(startr - obs.rlo, obs.highestbin - obs.rlo, "search", 1);

#ifndef DEBUG
#pragma omp parallel
#endif
           {
             int tid = omp_get_thread_num();

             cuFFdotBatch* trdBatch = &cuSrch->mInf->batches[tid];

             double*  startrs = (double*)malloc(sizeof(double)*trdBatch->noSteps);
             double*  lastrs  = (double*)malloc(sizeof(double)*trdBatch->noSteps);
             size_t rest = trdBatch->noSteps;

             setContext(trdBatch) ;

             int firstStep = 0;

             while ( ss < maxxx )
             {
#pragma omp critical
               {
                 firstStep = ss;
                 ss       += trdBatch->noSteps;
               }

               if ( firstStep >= maxxx )
                 break;

               if ( firstStep + trdBatch->noSteps >= maxxx )
               {
                 // TODO: there are a number of families we don't need to run see if we can use 'setPlainPointers(trdBatch)'
                 // To see if we can do less work on the last step
               }

               int si;
               for ( si = 0; si < trdBatch->noSteps ; si ++)
               {
                 if ( si < rest )
                 {
                   startrs[si] = obs.rlo + (firstStep+si) * ( trdBatch->accelLen * ACCEL_DR );
                   lastrs[si]  = startrs[si] + trdBatch->accelLen * ACCEL_DR - ACCEL_DR;
                 }
                 else
                 {
                   startrs[si] = 0 ;
                   lastrs[si]  = 0 ;
                 }
               }
               search_ffdot_planeCU(trdBatch, startrs, lastrs, obs.norm_type, 1, (fcomplexcu*)obs.fft, obs.numindep, &candsGPU);

               if ( trdBatch->flag & CU_OUTP_HOST )
               {
                 // TODO: check the addressing below for new cases ie:FLAG_STORE_EXP FLAG_STORE_ALL
                 // TODO: to a type casting check here!
                 trdBatch->d_candidates = &master->d_candidates[trdBatch->accelLen*obs.numharmstages*firstStep] ;
               }

               print_percent_complete(startrs[0] - obs.rlo, obs.highestbin - obs.rlo, "search", 0);
             }

             int si;
             for ( si = 0; si < trdBatch->noSteps ; si ++)
             {
               startrs[si] = 0;
               lastrs[si]  = 0;
             }

             // Finish searching the plains, this is required because of the out of order asynchronous calls
             int pln;
             for ( pln = 0 ; pln < 2; pln++ )
             {
               search_ffdot_planeCU(trdBatch, startrs, lastrs, obs.norm_type, 1, (fcomplexcu*)obs.fft, obs.numindep, &candsGPU);
             }
           }

           print_percent_complete(obs.highestbin - obs.rlo, obs.highestbin - obs.rlo, "search", 0);

           if ( master->flag & CU_CAND_ARR )
           {
             printf("\nCopying candidates from array to list for optimisation.\n");

             nvtxRangePush("Add to list");
             int cdx;

             long long numindep;

             double poww, sig, sigx, sigc, diff;
             double gpu_p, gpu_q;
             double rr, zz;
             int added = 0;
             int numharm;
             poww = 0;

             cand* candidate = (cand*)master->h_candidates;

             //FILE * pFile;
             //int n;
             //char name [100];
             //sprintf(name,"%s_hs%02i_zmax%06.1f_sig%06.3f_CU_CAND_ARR.csv", obs.rootfilenm, obs.numharmstages, obs.zhi, obs.sigma );

             //pFile = fopen (name,"w");
             //fprintf (pFile, "idx;rr;zz;sig;power;harm\n");

             printf("\n");
             for (cdx = 0; cdx < master->SrchSz->noOutpR; cdx++)  // Loop
             {
               poww        = candidate[cdx].power;

               if ( poww > 0 )
               {
                 numharm   = candidate[cdx].numharm;
                 numindep  = obs.numindep[twon_to_index(numharm)];
                 sig       = candidate[cdx].sig;
                 rr        = candidate[cdx].r;
                 zz        = candidate[cdx].z;

                 candsGPU  = insert_new_accelcand(candsGPU, poww, sig, numharm, rr, zz, &added);
                 //fprintf (pFile, "%i;%.2f;%.2f;%.2f;%.2f;%i\n",cdx,rr, zz,sig,poww,numharm);
                 if (added)
                 {
                   noCands++;
                   //printf("Cand %04i   x %6i  Pow: %8.2f  Sig: %6.2f  harm: %i  r %6.2f  z: %6.2f \n",noCands, cdx, poww, sig, numharm, rr, zz);
                 }

               }
             }
             printf("\n");
             //fclose (pFile);
           }

           /* TODO: fix this section using SrchSz parameters
          if ( master->flag & CU_OUTP_DEVICE )
          {
            nvtxRangePush("Add to list"); 
            int len = master->rHigh - master->rLow;

            CUDA_SAFE_CALL(cudaMemcpy(master->h_retData, master->d_retData, master->retDataSize*maxxx, cudaMemcpyDeviceToHost), "Failed to copy data to device");

            int cdx; 
            long long numindep;

            double poww, sig, sigx, sigc, diff;
            double gpu_p, gpu_q;
            double rr, zz;
            int added = 0;
            int numharm;
            poww = 0;

            if ( master->retType == CU_SMALCAND &&  master->cndType == CU_FULLCAND )
            {
              accelcandBasic* bsk = (accelcandBasic*)master->h_retData;

              for (cdx = 0; cdx < len; cdx++)
              {
                sig        = bsk[cdx].sigma;

                if ( sig > 0 )
                {
                  numharm   = bsk[cdx].numharm;
                  numindep  = obs.numindep[twon_to_index(numharm)];
                  rr        = cdx + master->rLow;
                  zz        = bsk[cdx].z;
                  candsGPU  = insert_new_accelcand(candsGPU, poww, sig, numharm, rr, zz, &added);
                }
              }
            }
            nvtxRangePop();
          }
            */

           //cudaProfilerStop(); // For profiling of only the 'critical' GPU section

           gettimeofday(&end, NULL);
           gpuTime += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));
           printf("  gpuTime %f\n", gpuTime/1000.0);
           cands = candsGPU;
           printf("GPU found %li candidates, %i unique..\n",noCands, g_slist_length(cands) );

#ifdef DEBUG
           char name [100];
           sprintf(name,"%s_hs%02i_zmax%06.1f_sig%06.3f_GPU_Cands.csv", obs.rootfilenm, obs.numharmstages, obs.zhi, obs.sigma );
           printCands(name, candsGPU);
#endif
         }
       }
#else
       fprintf(stderr,"ERROR: Not compiled with CUDA.\n");
#endif
     }

     if ( file = fopen(candsFile, "wb") )
     {
       int numcands = g_slist_length(cands);
       printf("\nWriting %i raw candidates from search to \"%s\".\n",numcands, candsFile);
       fwrite( &numcands, sizeof(numcands), 1, file );

       GSList *tmpLst = cands;
       int nc = 0;
       while (tmpLst)
       {
         accelcand* newCnd = tmpLst->data;

         fwrite( newCnd, sizeof(accelcand), 1, file );
         tmpLst = tmpLst->next;
         nc++;
       }

       fclose(file);
     }
     else
     {
       fprintf(stderr,"ERROR: unable to open \"%s\" to write candidates.\n",candsFile);
     }

     printf("\n\nDone searching.  Now optimizing each candidate.\n\n");
   }

   if (0) // optimization  .
   {                            /* Candidate list trimming and optimization */
      int numcands;
      GSList *listptr;
      accelcand *cand;
      fourierprops *props;


#ifdef CUDA
      gettimeofday(&start, NULL);       // Profiling
      nvtxRangePush("CPU optimisation");
      gettimeofday(&start, NULL);       // Note could start the timer after kernel init
#endif

      numcands = g_slist_length(cands);

      if (numcands) {

         /* Sort the candidates according to the optimized sigmas */

//#ifdef DEBUG
        printCands("Cands_Raw.csv", cands);
//#endif

         cands = sort_accelcands(cands);

         /* Eliminate (most of) the harmonically related candidates */
         if ((cmd->numharm > 1) && !(cmd->noharmremoveP))
         {
             eliminate_harmonics(cands, &numcands);
         }

//#ifdef DEBUG
         printCands("Cands_Thinned.csv", cands);
//#endif

         /* Now optimize each candidate and its harmonics */

         print_percent_complete(0, 0, NULL, 1);
         listptr = cands;
         for (ii = 0; ii < numcands; ii++) {
            //print_percent_complete(ii, numcands, "optimization", 0);
            cand = (accelcand *) (listptr->data);

            //if ( ii == 4 )
            {
              optimize_accelcand(cand, &obs, ii+1);
            }

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
            //if ( ii == 16 )
            {
            calc_props(cand->derivs[0], cand->r, cand->z, 0.0, props + ii);
            /* Override the error estimates based on power */
            props[ii].rerr = (float) (ACCEL_DR) / cand->numharm;
            props[ii].zerr = (float) (ACCEL_DZ) / cand->numharm;
            }
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
#endif
   }

#ifdef CUDA // Timing message
      printf("\n Timing:  Prep: %9.06f  CPU: %9.06f  GPU: %9.06f [%6.2f x]  Optimization: %9.06f \n\n", prepTime * 1e-6, cupTime * 1e-6, gpuTime * 1e-6, cupTime / (double) gpuTime, optTime * 1e-6 );

#ifdef TIMING
      int batch;
      float InpFFT  = 0;
      float convT   = 0;
      float InvFFT  = 0;
      float ss      = 0;
      for (batch = 0; batch < cuSrch->mInf->noBatches; batch++)
      {
        printf("          Input FFT: %9.06f  Convolve %9.06f  InvFFT: %9.06f  Sum&Search: %9.06f\n", cuSrch->mInf->batches[batch].InpFFTTime * 1e-3, cuSrch->mInf->batches[batch].convTime * 1e-3, cuSrch->mInf->batches[batch].InvFFTTime * 1e-3, cuSrch->mInf->batches[batch].searchTime * 1e-3 );
        InpFFT  += cuSrch->mInf->batches[batch].InpFFTTime;
        convT   += cuSrch->mInf->batches[batch].convTime;
        InvFFT  += cuSrch->mInf->batches[batch].InvFFTTime;
        ss      += cuSrch->mInf->batches[batch].searchTime;
      }
      printf("\n          Input FFT: %9.06f  Convolve %9.06f  InvFFT: %9.06f  Sum&Search: %9.06f\n\n", InpFFT * 1e-3, convT * 1e-3, InvFFT * 1e-3, ss * 1e-3 );
#endif

#endif

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

#ifndef DEBUG
#ifdef CUDA
   cudaDeviceReset();
#endif
#endif

   free_accelobs(&obs);
   g_slist_foreach(cands, free_accelcand, NULL);
   g_slist_free(cands);
   return (0);
}
