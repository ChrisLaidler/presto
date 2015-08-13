#include "accel.h"

/*#undef USEMMAP*/

#undef CUDA  // TMP
#define CUDA // TMP

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

int     pltOpt    = 0;

extern void zapbirds(double lobin, double hibin, FILE * fftfile, fcomplex * fft);

static void print_percent_complete(int current, int number, char *what, int reset)
{
  static int newper = 0, oldper = -1;

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
  long long prepTime = 0, cupTime = 0, gpuTime = 0, optTime = 0, cpuOptTime = 0, gpuOptTime = 0;
  struct timeval start, end;
  struct timeval start01, end01;

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
    exit(EXIT_SUCCESS);
  }
#endif

  printf("\n\n");
  printf("    Fourier-Domain Acceleration Search Routine\n");
  printf("               by Scott M. Ransom\n\n");

  /* Create the accelobs structure */
  create_accelobs(&obs, &idata, cmd, 1);

  /* Zap birdies if requested and if in memory */
  if (cmd->zaplistP && !obs.mmap_file && obs.fft)
  {
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

#ifdef CUDA  	  // Profiling  .
  nvtxRangePop();
  gettimeofday(&end, NULL);
  prepTime += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));

  cuSearch*     cuSrch = NULL;
  gpuSpecs      gSpec;
  searchSpecs   sSpec;

  gSpec        = readGPUcmd(cmd);
  sSpec        = readSrchSpecs(cmd, &obs);
#endif

  char fname[1024];
  sprintf(fname,"%s_hs%02i_zmax%06.1f_sig%06.3f", obs.rootfilenm, obs.numharmstages, obs.zhi, obs.sigma );

  char candsFile[1024];
  sprintf(candsFile,"%s.unoptcands", fname );

  FILE *file;
  if ( (file = fopen(candsFile, "rb")) && useUnopt ) 		  // Read candidates from previous search  . // TMP
  {
    int numcands;
    int wst = fread( &numcands, sizeof(numcands), 1, file );
    int nc = 0;

    printf("\nReading %i raw candies from \"%s\" previous search.\n", numcands, candsFile);

    for (nc = 0; nc < numcands; nc++)
    {
      accelcand* newCnd = (accelcand*)malloc(sizeof(accelcand));
      wst = fread( newCnd, sizeof(accelcand), 1, file );

      cands=insert_accelcand(cands,newCnd);
    }
    fclose(file);
  }
  else                                                    // Run Search  .
  {

    /* Start the main search loop */

    FOLD // the CPU and GPU main loop  .
    {
      double startr = obs.rlo, lastr = 0, nextr = 0;
      ffdotpows *fundamental;

      if ( cmd->cpuP ) // CPU search  .
      {
#ifdef CUDA // Profiling  .
        nvtxRangePush("CPU");
        gettimeofday(&start, NULL); // Note could start the timer after kernel init

        printf("\n*************************************************************************************************\n                         Doing CPU Search\n*************************************************************************************************\n");
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

        print_percent_complete(startr - obs.rlo, obs.highestbin - obs.rlo, "search", 1);

        while (startr + ACCEL_USELEN * ACCEL_DR < obs.highestbin)
        {
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
        printf("\n");

        free_subharminfos(&obs, subharminfs);

#ifdef CUDA  // Profiling  .

        // Basic timing
        gettimeofday(&end, NULL);
        cupTime += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));

        cands = candsCPU;
        printf("\nGPU found %i are unique. In %.4f ms\n", g_slist_length(candsCPU), cupTime/1000.0 );

        nvtxRangePop();

        char name [1024];
        sprintf(name,"%s_CPU_01_Cands.csv",fname);
        printCands(name, candsCPU, obs.T);
#endif

      }

#ifdef CUDA   // --=== The GPU Search == --  .
      if ( cmd->gpuP > 0 )
      {
        printf("\n*************************************************************************************************\n                         Doing GPU Search \n*************************************************************************************************\n");

        candsGPU = NULL;

        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
        //cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

        FOLD   // Contains main GPU loop  .
        {
          //cudaDeviceSynchronize();          // This is only necessary for timing
          gettimeofday(&start, NULL);       // Profiling
          //cudaProfilerStart();              // Start profiling, only really necessary for debug and profiling, surprise surprise

          long noCands            = 0;
          cuSrch                  = initCuSearch(&sSpec, &gSpec, NULL);
          cuFFdotBatch* master    = &cuSrch->mInf->kernels[0];   // The first kernel created holds global variables

          FOLD // TMP
          {
            if ( sSpec.fftInf.rhi != obs.highestbin )
            {
              fprintf(stderr,"ERROR: sSpec.fftInf.rhi != obs.highestbin \n");
            }
          }

          startr = sSpec.fftInf.rlo, lastr = 0, nextr = 0;
          int ss = 0;
          int maxxx = ( sSpec.fftInf.rhi - sSpec.fftInf.rlo ) / (float)( cuSrch->mInf->kernels[0].accelLen * ACCEL_DR ) ; // The number of plains to make

          if ( maxxx < 0 )
            maxxx = 0;

#ifndef STPMSG
          print_percent_complete(startr - sSpec.fftInf.rlo, sSpec.fftInf.rhi - sSpec.fftInf.rlo, "search", 1);
#else
          printf("GPU loop will process %i steps\n", maxxx);
#endif

          printf("\nRunning GPU search of %i steps with %i simultaneous families of f-∂f plains spread across %i device(s) .\n\n", maxxx, cuSrch->mInf->noSteps, cuSrch->mInf->noDevices );

#ifndef DEBUG 	// Parallel if we are not in debug mode  .
          omp_set_num_threads(cuSrch->mInf->noBatches);
          #pragma omp parallel
#endif
          FOLD  // Main GPU loop  .
          {
            int tid = omp_get_thread_num();

            cuFFdotBatch* trdBatch = &cuSrch->mInf->batches[tid];

            double*  startrs = (double*)malloc(sizeof(double)*trdBatch->noSteps);
            double*  lastrs  = (double*)malloc(sizeof(double)*trdBatch->noSteps);
            int      rest    = trdBatch->noSteps;

            setDevice(trdBatch) ;

            int firstStep    = 0;
            int step;

            while ( ss < maxxx )  // Main GPU loop  .
            {
#pragma omp critical
              {
                firstStep = ss;
                ss       += trdBatch->noSteps;
                cuSrch->noSteps++;

#ifdef STPMSG
                printf("\nStep %4i of %4i thread %02i processing %02i steps\n", firstStep+1, maxxx, tid, trdBatch->noSteps);
#endif
              }

              if ( firstStep >= maxxx )
                break;

              if ( firstStep + (int)trdBatch->noSteps >= maxxx )
              {
                // TODO: there are a number of families we don't need to run see if we can use 'setPlainPointers(trdBatch)'
                // To see if we can do less work on the last step
                rest = maxxx - firstStep;
              }

              // Set start r-vals for all steps in this batch
              for ( step = 0; step < (int)trdBatch->noSteps ; step ++)
              {
                if ( step < rest )
                {
                  startrs[step] = sSpec.fftInf.rlo + (firstStep+step) * ( trdBatch->accelLen * ACCEL_DR );
                  lastrs[step]  = startrs[step] + trdBatch->accelLen * ACCEL_DR - ACCEL_DR;
                }
                else
                {
                  startrs[step] = 0 ;
                  lastrs[step]  = 0 ;
                }
              }

              FOLD // Call the CUDA search  .
              {
                search_ffdot_batch_CU(trdBatch, startrs, lastrs, obs.norm_type, 1, (fcomplexcu*)sSpec.fftInf.fft, obs.numindep);
              }

#ifndef STPMSG
              print_percent_complete(startrs[0] - sSpec.fftInf.rlo, sSpec.fftInf.rhi - sSpec.fftInf.rlo, "search", 0);
#endif
            }

            FOLD  // Finish off CUDA search  .
            {
              // Set r values to 0 so as to not process details
              for ( step = 0; step < (int)trdBatch->noSteps ; step ++)
              {
                startrs[step] = 0;
                lastrs[step]  = 0;
              }

              // Finish searching the plains, this is required because of the out of order asynchronous calls
              for ( step = 0 ; step < 2; step++ )
              {
                search_ffdot_batch_CU(trdBatch, startrs, lastrs, obs.norm_type, 1, (fcomplexcu*)sSpec.fftInf.fft, obs.numindep);
              }
            }
          }

          print_percent_complete(sSpec.fftInf.rhi - sSpec.fftInf.rlo, sSpec.fftInf.rhi - sSpec.fftInf.rlo, "search", 0);
          printf("\n");

          if ( (master->flag & CU_CAND_ARR) ) // Copying candidates from array to list for optimisation  .
          {
            printf("\nCopying candidates from array to list for optimisation.\n");

            nvtxRangePush("Add to list");
            int cdx;

            double  poww, sig;
            double  rr, zz;
            int     added = 0;
            int     numharm;
            cand*   candidate = (cand*)master->h_candidates;
            poww    = 0;

            FILE * pFile;
            char name [1024];
            sprintf(name,"%s_GPU_ARRAY.csv",fname);
            pFile = fopen (name,"w");
            fprintf (pFile, "idx;rr;f;zz;sig;harm\n");

            for (cdx = 0; cdx < (int)master->SrchSz->noOutpR; cdx++)  // Loop
            {
              poww        = candidate[cdx].power;

              if ( poww > 0 )
              {
                numharm   = candidate[cdx].numharm;
                sig       = candidate[cdx].sig;
                rr        = candidate[cdx].r;
                zz        = candidate[cdx].z;

                candsGPU  = insert_new_accelcand(candsGPU, poww, sig, numharm, rr, zz, &added);
                fprintf (pFile, "%i;%.2f;%.3f;%.2f;%.2f;%i\n",cdx+1,rr, rr/obs.T, zz, sig, numharm);
                if (added)
                {
                  noCands++;
                }
              }
            }
            fclose (pFile);
          }

          // Basic timing
          gettimeofday(&end, NULL);
          gpuTime += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));

          cands = candsGPU;
          printf("\nGPU found %li candidates of which %i are unique. In %.4f ms\n",noCands, g_slist_length(cands), gpuTime/1000.0 );

          // Free GPU memory
          freeAccelGPUMem(cuSrch->mInf);

#ifdef DEBUG
          char name [1024];
          sprintf(name,"%s_GPU_01_Cands.csv",fname);
          printCands(name, candsGPU, obs.T);
#endif
        }
      }

#else
      fprintf(stderr,"ERROR: Not compiled with CUDA.\n");
#endif
    }

    FOLD // Write candidates to unoptcands file  .
    {
      if ( (file = fopen(candsFile, "wb")) )
      {
        int numcands = g_slist_length(cands);
        //printf("\nWriting %i raw candidates from search to \"%s\".\n",numcands, candsFile);
        fwrite( &numcands, sizeof(numcands), 1, file );

        GSList *candLst = cands;
        int nc = 0;
        while (candLst)
        {
          accelcand* newCnd = candLst->data;

          fwrite( newCnd, sizeof(accelcand), 1, file );
          candLst = candLst->next;
          nc++;
        }

        fclose(file);
      }
      else
      {
        fprintf(stderr,"ERROR: unable to open \"%s\" to write candidates.\n",candsFile);
      }
    }
  }

  FOLD  // optimization  .
  {                            /* Candidate list trimming and optimization */

    //printf("Done searching.  Now optimizing each candidate.\n\n");
    printf("\n*************************************************************************************************\n                          Optimizing candidates\n*************************************************************************************************\n");
    printf("\n");

    int numcands;
    GSList *listptr;
    accelcand *cand;
    fourierprops *props;
    numcands = g_slist_length(cands);

#ifdef CUDA
    char timeMsg[1024], dirname[1024], scmd[1024];
    time_t rawtime;
    struct tm* ptm;
    time ( &rawtime );
    ptm = localtime ( &rawtime );
    sprintf ( timeMsg, "%04i%02i%02i%02i%02i%02i", 1900 + ptm->tm_year, ptm->tm_mon + 1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec );

    sprintf(dirname,"/home/chris/accel/Nelder_Mead/%s-pre", timeMsg );
    mkdir(dirname, 0755);

    sprintf(scmd,"mv /home/chris/accel/Nelder_Mead/n* %s/ 2> /dev/null", dirname );
    system(scmd);

    sprintf(scmd,"mv /home/chris/accel/Nelder_Mead/*.png %s/ 2> /dev/null", dirname );
    system(scmd);

    sprintf(scmd,"mv /home/chris/accel/Nelder_Mead/*.csv %s/ 2> /dev/null", dirname );
    system(scmd);

    nvtxRangePush("Optimisation");
    gettimeofday(&start, NULL);       // Note could start the timer after kernel init
#endif

    if (numcands)
    {
      /* Sort the candidates according to the optimized sigmas */

      cands = sort_accelcands(cands);

      char name [1024];
      sprintf(name,"%s_GPU_02_Cands_Sorted.csv",fname);
      printCands(name, cands, obs.T);

      /* Eliminate (most of) the harmonically related candidates */
      if ((cmd->numharm > 1) && !(cmd->noharmremoveP))
      {
        eliminate_harmonics(cands, &numcands);
      }

      // Update the number of candidates
      numcands = g_slist_length(cands);

      sprintf(name,"%s_GPU_04_Cands_Thinned.csv",fname);
      printCands(name, cands, obs.T);

      /* Now optimize each candidate and its harmonics */

      printf("Optimising %i candidates.\n\n", numcands);

      if ( cmd->cpuP ) // CPU optimisation  .
      {
#ifdef CUDA // Profiling  .

        if (cmd->gpuP)
          candsGPU = duplicate_accelcands(cands);

        nvtxRangePush("CPU");
        gettimeofday(&start01, NULL);       // Profiling

#endif

        accelcand *candCPU;
        listptr = cands;
        print_percent_complete(0, 0, NULL, 1);

        for (ii = 0; ii < numcands; ii++)       //       ----==== Main Loop ====----  .
        {
          candCPU   = (accelcand *) (listptr->data);
          optimize_accelcand(candCPU, &obs, ii+1);
          listptr = listptr->next;
          print_percent_complete(ii, numcands, "optimization", 0);
        }

#ifdef CUDA // Profiling  .
        nvtxRangePop();
        gettimeofday(&end01, NULL);
        cpuOptTime += ((end01.tv_sec - start01.tv_sec) * 1e6 + (end01.tv_usec - start01.tv_usec));
#endif

      }

#ifdef CUDA   // --=== The GPU Optimisation == --  .
      if ( cmd->gpuP > 0  )
      {
        if (cmd->cpuP)
          cands = candsGPU; // This is the original unoptimised list

        nvtxRangePush("GPU Opt");
        gettimeofday(&start01, NULL);       // Profiling

        print_percent_complete(0, 0, NULL, 1);
        listptr = cands;
        ii      = 0;

#ifndef DEBUG   // Parallel if we are not in debug mode  .
        omp_set_num_threads(4);

#pragma omp parallel
#endif
        FOLD  // Main GPU loop  .
        {
          accelcand *candGPUP;
          int tid = omp_get_thread_num();

          FOLD  // Set device
          {
            int device = gSpec.devId[0];
            if ( device >= getGPUCount() )
            {
              fprintf(stderr, "ERROR: There is no CUDA device %i.\n",device);
              exit(EXIT_FAILURE);
            }
            int currentDevvice;
            CUDA_SAFE_CALL(cudaSetDevice(device), "ERROR: cudaSetDevice");
            CUDA_SAFE_CALL(cudaGetDevice(&currentDevvice), "Failed to get device using cudaGetDevice");
            if (currentDevvice != device)
            {
              fprintf(stderr, "ERROR: CUDA Device not set.\n");
              exit(EXIT_FAILURE);
            }
          }

          int       ti = 0; // tread specific index

          cuOptCand* oPlnPln;

          oPlnPln   = initOptPln(&sSpec);

          while (listptr)  // Main Loop  .
          {
#pragma omp critical
            FOLD // Calculate candidate  .
            {
              if ( listptr )
              {
                candGPUP  = (accelcand *) (listptr->data);
                listptr   = listptr->next;
                ii++;
                ti = ii;
              }
              else
              {
                candGPUP = NULL;
              }
            }

            if ( candGPUP ) // Optimise  .
            {
              opt_candPlns(candGPUP, &obs, ti, oPlnPln);
              //print_percent_complete(ti, numcands, "optimization", 0);
            }
          }
        }

        nvtxRangePop();
        gettimeofday(&end01, NULL);
        gpuOptTime += ((end01.tv_sec - start01.tv_sec) * 1e6 + (end01.tv_usec - start01.tv_usec));
      }
#endif

      printf("\n\n");

      // Re sort with new sigma values
      cands = sort_accelcands(cands);

      sprintf(name,"%s_GPU_05_Cands_Optemised.csv",fname);
      printCands(name, cands, obs.T);

      /* Eliminate (most of) the harmonically related candidates */
      if ((cmd->numharm > 1) && !(cmd->noharmremoveP))
      {
        eliminate_harmonics(cands, &numcands);
      }

      // Update the number of candidates
      numcands = g_slist_length(cands);

      /* Calculate the properties of the fundamentals */

      props = (fourierprops *) malloc(sizeof(fourierprops) * numcands);
      listptr = cands;
      for (ii = 0; ii < numcands; ii++)
      {
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

      sprintf(name,"%s_GPU_06_Cands_Optemised_cleaned.csv",fname);
      printCands(name, cands, obs.T);

      /* Write the fundamentals to the output text file */

      output_fundamentals(props, cands, &obs, &idata);

      /* Write the harmonics to the output text file */

      output_harmonics(cands, &obs, &idata);

      /* Write the fundamental fourierprops to the cand file */

      obs.workfile = chkfopen(obs.candnm, "wb");
      chkfwrite(props, sizeof(fourierprops), numcands, obs.workfile);
      fclose(obs.workfile);

      free(props);
      printf("\nDone optimizing.\n\n");

    }
    else
    {
      printf("No candidates above sigma = %.2f were found.\n\n", obs.sigma);
    }

#ifdef CUDA
    nvtxRangePop();

    gettimeofday(&end, NULL);
    optTime += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));

    if ( pltOpt > 0 )
    {
      sprintf(dirname,"/home/chris/accel/Nelder_Mead/%s", timeMsg );
      mkdir(dirname, 0755);

      sprintf(scmd,"mv /home/chris/accel/Nelder_Mead/n* %s/     2>/dev/null", dirname );
      system(scmd);

      sprintf(scmd,"mv /home/chris/accel/*.png %s/              2>/dev/null", dirname );
      system(scmd);

      sprintf(scmd,"mv /home/chris/accel/*.csv %s/              2>/dev/null", dirname );
      system(scmd);
    }
#endif
  }

  FOLD // Timing message  .
  {
#ifdef CUDA
    if (cuSrch)
    {
      printf("\n*************************************************************************************************\n                            Timing\n*************************************************************************************************\n");

      printf("\nTiming:  Prep:\t%9.06f\tCPU:\t%9.06f\tGPU:\t%9.06f\t[%6.2f x]\tOptimization:\t%9.06f\n\n", prepTime * 1e-6, cupTime * 1e-6, gpuTime * 1e-6, cupTime / (double) gpuTime, optTime * 1e-6 );

      writeLogEntry("/home/chris/accelsearch_log.csv", &obs, cuSrch, prepTime, cupTime, gpuTime, optTime, cpuOptTime, gpuOptTime );

#ifdef TIMING  // Advanced timing massage  .

      int batch, stack;
      float copyH2DT  = 0;
      float InpNorm   = 0;
      float InpFFT    = 0;
      float multT     = 0;
      float InvFFT    = 0;
      float ss        = 0;
      float resultT   = 0;
      float copyD2HT  = 0;

      printf("\n===========================================================================================================================================\n");
      printf("\nAdvanced timing, all times are in ms\n");

      for (batch = 0; batch < cuSrch->mInf->noBatches; batch++)
      {
        if ( cuSrch->mInf->noBatches > 1 )
          printf("Batch %02i\n",batch);

        cuFFdotBatch*   batches = &cuSrch->mInf->batches[batch];

        float l_copyH2DT  = 0;
        float l_InpNorm   = 0;
        float l_InpFFT    = 0;
        float l_multT     = 0;
        float l_InvFFT    = 0;
        float l_ss        = 0;
        float l_resultT   = 0;
        float l_copyD2HT  = 0;

        FOLD // Heading  .
        {
          printf("\t\t");
          printf("%s\t","Copy H2D");

          if ( batches->flag & CU_NORM_GPU )
            printf("%s\t","Norm GPU");
          else
            printf("%s\t","Norm CPU");

          if ( batches->flag & CU_INPT_CPU_FFT )
            printf("%s\t","Inp FFT CPU");
          else
            printf("%s\t","Inp FFT GPU");

          printf("%s\t","Multiplication");

          printf("%s\t","Inverse FFT");

          printf("%s\t","Sum & Search");

          if ( batches->flag & FLAG_SIG_GPU )
            printf("%s\t","Sigma GPU");
          else
            printf("%s\t","Sigma CPU");

          printf("%s\t","Copy D2H");

          printf("\n");
        }

        for (stack = 0; stack < (int)cuSrch->mInf->batches[batch].noStacks; stack++)
        {
          printf("Stack\t%02i\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\n", stack, batches->copyH2DTime[stack], batches->normTime[stack], batches->InpFFTTime[stack], batches->multTime[stack], batches->InvFFTTime[stack], batches->searchTime[stack], batches->resultTime[stack], batches->copyD2HTime[stack]  );

          l_copyH2DT  += batches->copyH2DTime[stack];
          l_InpNorm   += batches->normTime[stack];
          l_InpFFT    += batches->InpFFTTime[stack];
          l_multT     += batches->multTime[stack];
          l_InvFFT    += batches->InvFFTTime[stack];
          l_ss        += batches->searchTime[stack];
          l_resultT   += batches->resultTime[stack];
          l_copyD2HT  += batches->copyD2HTime[stack];
        }

        if ( cuSrch->mInf->noBatches > 1 )
        {
          printf("\t\t--------------------------------------------------------------------------------------------------------------------------\n");
          printf("\t\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\n", l_copyH2DT, l_InpNorm, l_InpFFT, l_multT, l_InvFFT, l_ss, l_resultT, l_copyD2HT );
        }

        copyH2DT  += l_copyH2DT;
        InpNorm   += l_InpNorm;
        InpFFT    += l_InpFFT;
        multT     += l_multT;
        InvFFT    += l_InvFFT;
        ss        += l_ss;
        resultT   += l_resultT;
        copyD2HT  += l_copyD2HT;
      }
      printf("\t\t--------------------------------------------------------------------------------------------------------------------------\n");
      printf("TotalT \t\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\n", copyH2DT, InpNorm, InpFFT, multT, InvFFT, ss, resultT, copyD2HT );

      printf("\n===========================================================================================================================================\n\n");

#endif
    }
#endif
  }

  /* Finish up */

  printf("Searched the following approx numbers of independent points:\n");
  for (ii = 0; ii < obs.numharmstages; ii++)
    printf("  %2d harmonics:  %9lld\n", 1 << ii, obs.numindep[ii]);

  printf("\nTiming summary:\n");
  tott = times(&runtimes)   / (double) CLK_TCK - tott;
  utim = runtimes.tms_utime / (double) CLK_TCK;
  stim = runtimes.tms_stime / (double) CLK_TCK;
  ttim = utim + stim;
  printf("    CPU time: %.3f sec (User: %.3f sec, System: %.3f sec)\n", ttim, utim, stim);
  printf("  Total time: %.3f sec\n\n", tott);

  printf("Final candidates in binary format are in '%s'.\n",    obs.candnm);
  printf("Final Candidates in a text format are in '%s'.\n\n",  obs.accelnm);

#ifdef CUDA
  freeCuSearch(cuSrch);
  cuProfilerStop(); // TMP
#endif

  free_accelobs(&obs);
  g_slist_foreach(cands, free_accelcand, NULL);
  g_slist_free(cands);
  return (0);
}

