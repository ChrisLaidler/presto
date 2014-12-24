#include "accel.h"

/*#undef USEMMAP*/

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

extern float calc_median_powers(fcomplex * amplitudes, int numamps);
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

#ifdef CUDA
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


#ifdef CUDA
      nvtxRangePop();
      gettimeofday(&end, NULL);
      prepTime += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));
#endif


   /* Start the main search loop */

   FOLD // the main loop
   {
      double startr = obs.rlo, lastr = 0, nextr = 0;
      ffdotpows *fundamental;

      if ( cmd->cpuP )
      {
#ifdef CUDA
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
#ifdef CUDA
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
      
//#ifdef CUDA
      if ( cmd->gpuP >= 0)
      {
        candsGPU = NULL;
        
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

        int noHarms = (1 << (obs.numharmstages - 1));
        
        FOLD
        {
          //cudaDeviceSynchronize();          // This is only necessary for timing
          gettimeofday(&start, NULL);       // Profiling
          //cudaProfilerStart();              // Start profiling, only really necessary debug and profiling, surprise surprise
          
          printf("\n------------------------\nDoing GPU Search 2\n------------------------\n"); 

          int dev;
          long noCands = 0;
          
          if ( cmd->gpuP ) // Determine the index and number of devices
          { 
            if ( cmd->gpuC == 0 )  // NB: Note using gpuC == 0 requires a chage in accelsearch_cmd every time clig is run!!!!
            {
              // Make a list of all devices
              cmd->gpuC = getGPUCount();
              cmd->gpu = (int*) malloc( cmd->gpuC * sizeof(int) );
              for ( dev = 0 ; dev < cmd->gpuC; dev++ )
                cmd->gpu[dev] = dev;
            }
          }
          
          cuStackList* kernels;             // List of stacks with the kernels, one for each device being used
          cuStackList* master   = NULL;     // The first kernel stack created
          int nPlains           = 0;        // The number of plains
          int noKers            = 0;        // Real number of kernels/devices being used
          
          FOLD // Create a kernel on each device
          {
            nvtxRangePush("Init Kernels");

            kernels = (cuStackList*)malloc(cmd->gpuC*sizeof(cuStackList));        
            int added; 
            
            for ( dev = 0 ; dev < cmd->gpuC; dev++ ) // Loop over devices  .
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

              added = initHarmonics(&kernels[noKers], master, obs.numharmstages, (int)obs.zhi, &obs, cmd->gpu[dev], noSteps, cmd->width, no );
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

            nvtxRangePop();
          } 
          
          cuStackList* plainsj[noKers*5];   // List of pointers to each plain
          int noSteps = 0;
          
          FOLD // Create plains for calculations
          {
            nvtxRangePush("Init Stacks");

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
                plainsj[nPlains] = initStkList(&kernels[dev], pln, no-1);
                
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

            nvtxRangePop();
          }
          
          printf("\nRunning GPU search with %i simultaneous families of f-∂f plains spread across %i device(s).\n\n", noSteps, noKers);
          
          omp_set_num_threads(nPlains);
          
          startr = obs.rlo, lastr = 0, nextr = 0;
          int ss = 0;
          int maxxx = ( obs.highestbin - obs.rlo ) / (float)( master->accelLen * ACCEL_DR ) ;
          
          //float ns = ( obs.highestbin - obs.rlo ) / (float)( master->accelLen * ACCEL_DR ) ;
          
          if ( maxxx < 0 )
            maxxx = 0;
          
          print_percent_complete(startr - obs.rlo, obs.highestbin - obs.rlo, "search", 1);

          #pragma omp parallel
          {
            int tid = omp_get_thread_num();
            
            cuStackList* trdStack = plainsj[tid];
            
            double*  startrs = (double*)malloc(sizeof(double)*trdStack->noSteps);
            double*  lastrs  = (double*)malloc(sizeof(double)*trdStack->noSteps);
            size_t rest = trdStack->noSteps;
            
            setContext(trdStack) ;
            
            int firstStep = 0;
            
            while ( ss < maxxx )
            {
              #pragma omp critical
              {
                firstStep = ss;
                ss       += trdStack->noSteps;
              }
              
              if ( firstStep >= maxxx )
                break;
              
              if ( firstStep + trdStack->noSteps >= maxxx )
              {
                //trdStack->noSteps = maxxx - firstStep;
                //setPlainPointers(trdStack);
                rest = maxxx - firstStep;
              }

              int si;
              for ( si = 0; si < trdStack->noSteps ; si ++)
              {
                if ( si < rest )
                {
                  startrs[si] = obs.rlo + (firstStep+si) * ( master->accelLen * ACCEL_DR );
                  lastrs[si]  = startrs[si] + master->accelLen * ACCEL_DR - ACCEL_DR;
                }
                else
                {
                  startrs[si] = 0 ;
                  lastrs[si]  = 0 ;
                }
              }

              
              ffdot_planeCU3(trdStack, startrs, lastrs, obs.norm_type, 1, obs.fft, &obs, &candsGPU);
              
              if ( trdStack->flag & CU_CAND_HOST )
                trdStack->h_bCands = &master->h_bCands[master->accelLen*obs.numharmstages*firstStep] ;
              
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
              ffdot_planeCU3(trdStack, startrs, lastrs, obs.norm_type, 1, obs.fft, &obs, &candsGPU);
              trdStack->mxSteps = rest;
            }
          }
          
          print_percent_complete(obs.highestbin - obs.rlo, obs.highestbin - obs.rlo, "search", 0);

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
                double sig = master->h_candidates[cdx].sig;
                int biggest = 1;

                int dx;
                for ( dx = cdx - ACCEL_CLOSEST_R ; dx <= cdx + ACCEL_CLOSEST_R; dx++ )
                {
                  if ( dx >= 0 && dx < len )
                  {
                    if ( master->h_candidates[dx].sig > sig )
                    {
                      biggest = 0;
                      break;
                    }
                  }
                }

                if ( biggest )
                {
                  numharm   = master->h_candidates[cdx].numharm;
                  numindep  = obs.numindep[twon_to_index(numharm)];
                  sig       = master->h_candidates[cdx].sig;
                  rr        = master->h_candidates[cdx].r;
                  zz        = master->h_candidates[cdx].z;
                  candsGPU  = insert_new_accelcand(candsGPU, poww, sig, numharm, rr, zz, &added);

                  if (added)
                  {
                    noCands++;
                  }
                }
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
          
          //cudaProfilerStop();

          //cudaDeviceSynchronize();
          gettimeofday(&end, NULL);
          gpuTime += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));
          printf("  gpuTime %f\n", gpuTime/1000.0);
          cands = candsGPU;
          printf("GPU found %i candidates.\n",noCands);
          
#ifndef DEBUG
          printCands("GPU_Cands.csv", candsGPU);
#endif
          //cudaDeviceSynchronize();
        }
      }

//#endif
   }

   printf("\n\nDone searching.  Now optimizing each candidate.\n\n");


   if (1) // optimization
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
#endif
   }

#ifdef CUDA
      printf("\n Timing:  Prep: %9.06f  CPU: %9.06f  GPU: %9.06f [%6.2f x]  Optimization: %9.06f \n\n", prepTime * 1e-6, cupTime * 1e-6, gpuTime * 1e-6, cupTime / (double) gpuTime, optTime * 1e-6 );
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
   cudaDeviceReset();
#endif

   free_accelobs(&obs);
   g_slist_foreach(cands, free_accelcand, NULL);
   g_slist_free(cands);
   return (0);
}
