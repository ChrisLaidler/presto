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

#ifdef CUDA
#include <sys/time.h>
#include <time.h>

#ifdef CUDA_PROF
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#endif

#include "cuda_accel.h"
#include "cuda_accel_utils.h"
#include "cuda_response.h"
#endif

#ifdef USEDMALLOC
#include "dmalloc.h"
#endif

#ifdef WITHOMP
#include <omp.h>
#endif

static void print_percent_complete(int current, int number, const char *what, int reset)
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
      printf("\rAmount of %s complete = %3d%%  ", what, newper);
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

  FILE *file;

  // Timing vars
  //long long contextInit = 0, cuSrch->timings[TIME_PREP] = 0, cuSrch->timings[TIME_CPU_KER] = 0, cuSrch->timings[TIME_GPU_KER] = 0, cuSrch->timings[TIME_CPU_SRCH] = 0, cuSrch->timings[TIME_GPU_SRCH] = 0, cuSrch->timings[TIME_CPU_SRCH] = 0, cuSrch->timings[TIME_ALL_OPT] = 0, cuSrch->timings[TIME_CPU_OPT] = 0, cuSrch->timings[TIME_GPU_OPT] = 0;
  long long contextInit = 0;
  struct timeval start, end;
  struct timeval start01, end01;

  /* Prep the timer */

  tott = times(&runtimes) / (double) CLK_TCK;

  /* Call usage() if we have no command line arguments */

  if (argc == 1)
  {
    Program = argv[0];
    printf("\n");
    usage();
    exit(1);
  }

  /* Parse the command line using the excellent program Clig */

  cmd = parseCmdline(argc, argv);

  char name[1024];

#ifdef DEBUG
  showOptionValues();
#endif

#ifdef CUDA

  if (cmd->lsgpuP) // List GPU's  .
  {
    listDevices();
    exit(EXIT_SUCCESS);
  }

  if (!cmd->cpuP && !cmd->gpuP)
  {
    fprintf(stderr, "\nWARNING: GPU or CPU not specified defaulting to GPU\n\n");
    cmd->gpuP = 1;
  }

#else

  if ( !cmd->cpuP )
  {
    // Not compiled with CUDA so we have to do a CPU search
    cmd->cpuP = 1;
  }

#endif

  printf("\n\n");
  printf("    Fourier-Domain Acceleration Search Routine\n");
  printf("               by Scott M. Ransom\n\n");

#ifdef CUDA // CUDA Runtime initialisation  .

  cuSearch*     cuSrch = NULL;
  gpuSpecs      gSpec;
  searchSpecs   sSpec;

  pthread_t     cntxThread = 0;

  // Start the timer
  gettimeofday(&start, NULL);

  if ( cmd->gpuP ) // Initialises CUDA context(s)  .
  {
    printf("      with GPU additions by Chris Laidler\n\n");
    printf("      The GPU version is still under development.\n");
    printf("      If you find any bugs pleas report to:\n");
    printf("            chris.laidler@gmail.com\n\n");

    gSpec         = readGPUcmd(cmd);

    // Initalise coda context
    contextInit += initCudaContext(&gSpec);
  }

  NV_RANGE_PUSH("Prep");
#endif

  /* Create the accelobs structure */
  create_accelobs(&obs, &idata, cmd, 1);

#ifdef CUDA
  sSpec       = readSrchSpecs(cmd, &obs);
  cuSrch      = initSearchInf(&sSpec, &gSpec, cuSrch);
  printf("\n");
#endif

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

  printf("Searching with up to %d harmonics summed:\n", 1 << (obs.numharmstages - 1));
  printf("  f = %.1f to %.1f Hz\n", obs.rlo / obs.T, obs.rhi / obs.T);
  printf("  r = %.1f to %.1f Fourier bins\n", obs.rlo, obs.rhi);
  printf("  z = %.1f to %.1f Fourier bins drifted\n\n", obs.zlo, obs.zhi);

#ifdef CUDA
  char fname[1024];
  sprintf(fname,"%s_hs%02i_zmax%06.1f_sig%06.3f", obs.rootfilenm, obs.numharmstages, obs.zhi, obs.sigma );
  char candsFile[1024];
  sprintf(candsFile,"%s.unoptcands", fname );

  NV_RANGE_POP();
  gettimeofday(&end, NULL);
  cuSrch->timings[TIME_PREP] += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));

#endif

#ifdef CBL
  if ( (file = fopen(candsFile, "rb")) && useUnopt ) 		// DEBUG: Read candidates from previous search  .
  {
    int numcands;
    fread( &numcands, sizeof(numcands), 1, file );
    int nc = 0;

    printf("\nReading %i raw candies from \"%s\" previous search.\n", numcands, candsFile);

    for (nc = 0; nc < numcands; nc++)
    {
      accelcand* newCnd = (accelcand*)malloc(sizeof(accelcand));
      fread( newCnd, sizeof(accelcand), 1, file );

      cands=insert_accelcand(cands,newCnd);
    }
    fclose(file);

    // Wait for the context thread to complete
    cuSrch->timings[TIME_CONTEXT] = compltCudaContext(&gSpec);
  }
  else								// Run Search  .
#endif
  {
    /* Start the main search loop */

    FOLD  // The CPU and GPU main loop  .
    {
      double startr = obs.rlo, lastr = 0, nextr = 0;
      ffdotpows *fundamental;

      if ( cmd->cpuP ) 	          // --=== The CPU Search == --  .
      {
#ifdef CUDA // Profiling  .
        printf("\n*************************************************************************************************\n                         Doing CPU Search\n*************************************************************************************************\n");

        NV_RANGE_PUSH("CPU");
        gettimeofday(&start, NULL);
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

#ifdef CUDA // Basic timing  .
        gettimeofday(&end, NULL);
        cuSrch->timings[TIME_CPU_KER] += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));
#endif

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

        cands = candsCPU;

#ifdef CUDA  // Profiling  .

        FOLD // Basic timing  .
        {
          gettimeofday(&end, NULL);
          cuSrch->timings[TIME_CPU_SRCH] += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));
        }

        printf("\nCPU found %i initial candidates. In %.4f ms\n", g_slist_length(candsCPU), cuSrch->timings[TIME_CPU_SRCH]/1000.0 );

        NV_RANGE_POP();

        if ( sSpec.flags & FLAG_DPG_PRNT_CAND )
        {
          sprintf(name,"%s_CPU_01_Cands.csv", fname);
          printCands(name, candsCPU, obs.T);
        }

#endif

      }

      if ( cmd->gpuP )            // --=== The GPU Search == --  .
      {
#ifdef CUDA

        // Wait for the context thread to complete
        cuSrch->timings[TIME_CONTEXT] = compltCudaContext(&gSpec);

#ifdef NVVP // Start profiler
        cudaProfilerStart();              // Start profiling, only really necessary for debug and profiling, surprise surprise
#endif

        printf("\n*************************************************************************************************\n                         Doing GPU Search \n*************************************************************************************************\n");

        int maxxx;
        cuFFdotBatch* master;
        char srcTyp[1024];

        long  noCands           = 0;
        int   ss                = 0;
        candsGPU                = NULL;

        FOLD // Basic timing  .
        {
          gettimeofday(&start, NULL);
        }

        FOLD // init GPU kernels and planes  .
        {
          cuSrch    = initCuKernels(&sSpec, &gSpec, cuSrch);
          master    = &cuSrch->pInf->kernels[0];   // The first kernel created holds global variables

          // Timing of device setup and kernel creation
          gettimeofday(&end, NULL);
          cuSrch->timings[TIME_GPU_KER] += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));
        }

        FOLD // Set the bounds of the search
        {
          // Search bounds
          startr    = 0, lastr = 0, nextr = 0;
          maxxx     = cuSrch->SrchSz->noSteps;
          if ( master->flags & FLAG_SS_INMEM  )
          {
            startr  = cuSrch->SrchSz->searchRLow ; // ie ( rlo / no harms)
          }
          else
            startr  = sSpec.fftInf.rlo;

          if ( maxxx < 0 )
            maxxx = 0;

          printf("\nRunning GPU search of %i steps with %i simultaneous families of f-∂f planes spread across %i device(s).\n\n", maxxx, cuSrch->pInf->noSteps, cuSrch->pInf->noDevices );

          if ( msgLevel == 0 )
          {
            print_percent_complete(startr - startr, sSpec.fftInf.rhi - startr, "search", 1);
          }
          else
          {
            fflush(stdout);
            fflush(stderr);
            infoMSG(1,0,"\nGPU loop will process %i steps\n", maxxx);
          }

          if      ( master->flags & FLAG_SS_INMEM     )
            sprintf(srcTyp, "Generating in-mem GPU plane");
          else
            sprintf(srcTyp, "GPU search");
        }

        if ( master->flags & FLAG_SYNCH )
          fprintf(stderr, "WARNING: Running synchronous search, this will slow things down and should only be used for debug and testing.\n");

        FOLD //                                 ---===== Main Loop =====---  .
        {
          FOLD // Do the search or inmem plane creation  .
          {
            infoMSG(1,0,"Plane creation.\n");

            if      ( master->flags & FLAG_SS_INMEM     )
              NV_RANGE_PUSH("In-Mem plane");
            else
              NV_RANGE_PUSH("GPU Search");

            int iteration = 0;

#ifndef DEBUG 	// Parallel if we are not in debug mode  .
            if ( cuSrch->sSpec->flags & FLAG_SYNCH )
            {
              omp_set_num_threads(1);
            }
            else
            {
              omp_set_num_threads(cuSrch->pInf->noBatches);
            }

#pragma omp parallel
#endif
            FOLD  //                              ---===== Main Loop =====---  .
            {
              int tid = omp_get_thread_num();

              cuFFdotBatch* batch = &cuSrch->pInf->batches[tid];

              double*  startrs = (double*)malloc(sizeof(double)*batch->noSteps);
              double*  lastrs  = (double*)malloc(sizeof(double)*batch->noSteps);
              int      rest    = batch->noSteps;

              setDevice(batch->device) ;

              int firstStep    = 0;
              int step;

              while ( ss < maxxx )  //            ---===== Main Loop =====---  .
              {
                FOLD // Calculate the step(s) to handle  .
                {

#pragma omp critical
                  FOLD // Calculate the step  .
                  {

                    FOLD  // Synchronous behaviour  .
                    {
#ifndef  DEBUG
                      if ( cuSrch->sSpec->flags & FLAG_SYNCH )
#endif
                      {
                        // If running in synchronous mode use multiple batches, just synchronously
                        tid = iteration % cuSrch->pInf->noBatches ;
                        batch = &cuSrch->pInf->batches[tid];
                        setDevice(batch->device) ;
                      }
                    }

                    iteration++;

                    firstStep = ss;
                    ss       += batch->noSteps;
                    cuSrch->noSteps++;

                    infoMSG(1,1,"\nStep %4i of %4i thread %02i processing %02i steps on GPU %i\n", firstStep+1, maxxx, tid, batch->noSteps, batch->device );
                  }

                  if ( firstStep >= maxxx )
                    break;

                  if ( firstStep + (int)batch->noSteps >= maxxx ) // End case (there is some overflow)  .
                  {
                    // TODO: There are a number of families we don't need to run see if we can use 'setplanePointers(trdBatch)'
                    // To see if we can do less work on the last step
                    rest = maxxx - firstStep;
                  }
                }

                FOLD // Set start r-vals for all steps in this batch  .
                {
                  for ( step = 0; step < (int)batch->noSteps ; step++ )
                  {
                    rVals* rVal = &(*batch->rAraays)[0][step][0];

                    if ( step < rest )
                    {
                      startrs[step]   = startr        + (firstStep+step) * ( batch->accelLen * ACCEL_DR );
                      lastrs[step]    = startrs[step] + batch->accelLen * ACCEL_DR - ACCEL_DR;

                      rVal->drlo      = startrs[step];
                      rVal->drhi      = lastrs[step];

                      int harm;
                      for (harm = 0; harm < batch->noGenHarms; harm++)
                      {
                        rVal          = &(*batch->rAraays)[0][step][harm];
                        rVal->step    = firstStep + step;
                        rVal->norm    = 0.0;
                      }
                    }
                    else
                    {
                      startrs[step]   = 0 ;
                      lastrs[step]    = 0 ;
                    }
                  }
                }

                FOLD // Call the CUDA search  .
                {
                  search_ffdot_batch_CU(batch, startrs, lastrs, obs.norm_type);
                }

                FOLD // Print message  .
                {
                  if ( msgLevel == 0  )
                  {
                    if      ( master->flags & FLAG_SS_INMEM     )
                    {
                      printf("\rGenerating in-mem GPU plane  %5.1f%%", firstStep/(float)maxxx*100.0);
                    }
                    else
                    {
                      int noTrd;
                      sem_getvalue(&master->cuSrch->threasdInfo->running_threads, &noTrd );
                      printf("\rGPU search  %5.1f%% ( %3i Active CPU threads processing initial candidates)  ", firstStep/(float)maxxx*100.0, noTrd);
                    }

                    fflush(stdout);
                  }
                }

              }

              FOLD  // Finish off CUDA search  .
              {
                infoMSG(1,0,"\nFinish off search.\n");

                // Set r values to 0 so as to not process details  .
                for ( step = 0; step < (int)batch->noSteps ; step++)
                {
                  startrs[step] = 0;
                  lastrs[step]  = 0;
                }

                // Finish searching the planes, this is required because of the out of order asynchronous calls
                for ( rest = 0 ; rest < batch->noRArryas; rest++ )
                {
                  FOLD // Set the r arrays to zero  .
                  {
                    rVals* rVal = (*batch->rAraays)[0][0];
                    memset(rVal, 0, sizeof(rVals)*batch->noSteps);
                  }

                  search_ffdot_batch_CU(batch, startrs, lastrs, obs.norm_type);
                }

                // Wait for asynchronous execution to complete
                finish_Search(batch);
              }
            }

            printf("\r%s. %5.1f%%                                                                                         \n", srcTyp, 100.0);

            FOLD // Wait for CPU threads to complete  .
            {
              waitForThreads(&master->cuSrch->threasdInfo->running_threads, "Waiting for CPU thread(s) to finish processing returned from the GPU,", 200 );
            }

            NV_RANGE_POP();

          }

          FOLD // Do in-mem search  .
          {
            if      ( master->flags & FLAG_SS_INMEM     )
            {
              infoMSG(1,1,"\nIn-mem sum & Search\n");

              inmemSumAndSearch(cuSrch);
            }
          }

          FOLD // Process candidates  .
          {
            infoMSG(1,1,"\nProcess candidates\n");

            FOLD // Basic timing  .
            {
              gettimeofday(&start01, NULL);
            }

            if      ( master->cndType & CU_STR_ARR    ) // Copying candidates from array to list for optimisation  .
            {
              printf("\nCopying initial candidates from array to list for optimisation.\n");

              NV_RANGE_PUSH("Add to list");

              int     cdx;
              double  poww, sig;
              double  rr, zz;
              int     added = 0;
              int     numharm;
              initCand*   candidate = (initCand*)cuSrch->h_candidates;
              poww    = 0;

#ifdef DEBUG
              FILE * pFile;
              sprintf(name,"%s_GPU_ARRAY.csv",fname);
              pFile = fopen (name,"w");
              fprintf (pFile, "idx;rr;f;zz;sig;harm\n");
#endif
              for (cdx = 0; cdx < (int)cuSrch->SrchSz->noOutpR; cdx++)  // Loop
              {
                poww        = candidate[cdx].power;

                if ( poww > 0 )
                {
                  numharm   = candidate[cdx].numharm;
                  sig       = candidate[cdx].sig;
                  rr        = candidate[cdx].r;
                  zz        = candidate[cdx].z;

                  candsGPU  = insert_new_accelcand(candsGPU, poww, sig, numharm, rr, zz, &added );

#ifdef DEBUG
                  fprintf (pFile, "%i;%.2f;%.3f;%.2f;%.2f;%i\n",cdx+1,rr, rr/obs.T, zz, sig, numharm );
#endif
                  noCands++;
                }
              }
#ifdef DEBUG
              fclose (pFile);
#endif

              NV_RANGE_POP();
            }
            else if ( master->cndType & CU_STR_LST    )
            {
              candsGPU  = (GSList*)cuSrch->h_candidates;

              int bIdx;
              for ( bIdx = 0; bIdx < cuSrch->pInf->noBatches; bIdx++ )
              {
                noCands += cuSrch->pInf->batches[bIdx].noResults;
              }

              if ( candsGPU )
              {
                if ( candsGPU->data == NULL )
                {
                  // No real candidates found!
                  candsGPU = NULL;
                }
              }
            }
            else if ( master->cndType & CU_STR_QUAD   ) // Copying candidates from array to list for optimisation  .
            {
              // TODO: write the code!

              fprintf(stderr, "ERROR: Quad-tree candidates has not yet been finalised for optimisation!\n");
              exit(EXIT_FAILURE);

              //candsGPU = testTest(master, candsGPU);
            }
            else
            {
              fprintf(stderr, "ERROR: Bad candidate storage method?\n");
              exit(EXIT_FAILURE);
            }

            cands = candsGPU;

            FOLD // Basic timing  .
            {
              gettimeofday(&end01, NULL);
              cuSrch->timings[TIME_CND] += ((end01.tv_sec - start01.tv_sec) * 1e6 + (end01.tv_usec - start01.tv_usec));
            }
          }
        }

        // Free GPU memory
        freeAccelGPUMem(cuSrch->pInf);

        FOLD // Basic timing  .
        {
          gettimeofday(&end, NULL);
          cuSrch->timings[TIME_GPU_SRCH] += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));
        }

        printf("\nGPU found %li initial candidates of which %i are unique. In %.4f ms\n", noCands, g_slist_length(cands), cuSrch->timings[TIME_GPU_SRCH]/1000.0 );

        if ( cuSrch->sSpec->flags & FLAG_DPG_PRNT_CAND )
        {
          char name [1024];
          sprintf(name,"%s_GPU_01_Cands.csv",fname);
          printCands(name, candsGPU, obs.T);
        }

#else
        fprintf(stderr,"ERROR: Requested a GPU search but, not compiled with CUDA. Edit the make file, and rebuild.\n");
        exit(EXIT_FAILURE);
#endif
      }
    }

#ifdef CUDA
    FOLD // Write candidates to unoptcands file  .
    {
      if ( useUnopt )
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
            accelcand* newCnd = (accelcand*)candLst->data;

            fwrite( newCnd, sizeof(accelcand), 1, file );
            candLst = candLst->next;
            nc++;
          }

          fclose(file);
        }
        else
        {
          fprintf(stderr,"ERROR: unable to open \"%s\" to write initial candidates.\n",candsFile);
        }
      }
    }
#endif

  }

  FOLD  // optimization  .
  {                            /* Candidate list trimming and optimization */
    printf("\n*************************************************************************************************\n                          Optimizing initial candidates\n*************************************************************************************************\n");
    printf("\n");

    int numcands;
    GSList *listptr;
    accelcand *cand;
    fourierprops *props;
    numcands = g_slist_length(cands);

#ifdef CUDA  // Timing and debug stuff
    char timeMsg[1024], dirname[1024], scmd[1024];
    time_t rawtime;
    struct tm* ptm;

    NV_RANGE_PUSH("Optimisation");
    gettimeofday(&start, NULL);       // Note could start the timer after kernel init

#ifdef CBL
    if ( sSpec.flags & FLAG_DPG_PLT_OPT )
    {
      time ( &rawtime );
      ptm = localtime ( &rawtime );
      sprintf ( timeMsg, "%04i%02i%02i%02i%02i%02i", 1900 + ptm->tm_year, ptm->tm_mon + 1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec );

      sprintf(dirname,"/home/chris/accel/Nelder_Mead/%s-pre", timeMsg );
      mkdir(dirname, 0755);

      sprintf(scmd,"mv /home/chris/accel/*.png %s/ 2> /dev/null", dirname );
      system(scmd);

      sprintf(scmd,"mv /home/chris/accel/*.csv %s/ 2> /dev/null", dirname );
      system(scmd);
    }
#endif


    if ( sSpec.flags & FLAG_DPG_SKP_OPT )
      numcands = 0;
#endif

    if (numcands)
    {
      /* Sort the candidates according to the optimized sigmas */

      cands = sort_accelcands(cands);

#ifdef CUDA
      if ( sSpec.flags & FLAG_DPG_PRNT_CAND )
      {
	sprintf(name,"%s_GPU_02_Cands_Sorted.csv",fname);
	printCands(name, cands, obs.T);
      }
#endif

      /* Eliminate (most of) the harmonically related candidates */
      if ((cmd->numharm > 1) && !(cmd->noharmremoveP))
      {
	eliminate_harmonics(cands, &numcands);
      }

      // Update the number of candidates
      numcands = g_slist_length(cands);

#ifdef CUDA
      if ( sSpec.flags & FLAG_DPG_PRNT_CAND )
      {
	sprintf(name,"%s_GPU_04_Cands_Thinned.csv",fname);
	printCands(name, cands, obs.T);
      }
#endif

      /* Now optimize each candidate and its harmonics */

      printf("Optimising the remaining %i initial candidates.\n\n", numcands);

      if ( cmd->cpuP ) 	 	// --=== The CPU Optimisation == --  .
      {
#ifdef CUDA // Profiling  .

        // Doing a GPU search as well so duplicate candidates
	if (cmd->gpuP)
	  candsGPU = duplicate_accelcands(cands);

	NV_RANGE_PUSH("CPU Optimisation");
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
	NV_RANGE_POP();
	gettimeofday(&end01, NULL);
	cuSrch->timings[TIME_CPU_OPT] += ((end01.tv_sec - start01.tv_sec) * 1e6 + (end01.tv_usec - start01.tv_usec));
#endif

      }

      if ( cmd->gpuP )        	// --=== The GPU Optimisation == --  .
      {
#ifdef CUDA

        // Initialise optimisation details!
        cuSrch = initCuOpt(&sSpec, &gSpec, cuSrch);

        // Optimise all the candidates
	optList(cands, cuSrch);

#else
	fprintf(stderr,"ERROR: not compiled with CUDA!\n");
#endif
      }

      printf("\n\n");

#ifdef CBL
      Fout // TMP
      {
	Logger slog(stdout);
	slog.setCsvDeliminator('\t');
	listptr 	= cands;
	double T	= obs.T;

	for (ii = 0; ii < numcands; ii++)
	{
	  cand    = (accelcand *) (listptr->data);
	  listptr = listptr->next;

//	  Fout

	  slog.csvWrite("TAG","cnd");

	  slog.csvWrite("int freq","%9.7f",cand->init_r/T);
	  slog.csvWrite("opt freq","%9.7f",cand->r/T);

	  slog.csvWrite("int z   ","%9.6f",cand->init_z);
	  slog.csvWrite("opt z   ","%9.6f",cand->z);

	  slog.csvWrite("int harm","%9i",cand->init_numharm);
	  slog.csvWrite("opt harm","%9i",cand->numharm);

	  slog.csvWrite("int pow ","%9.5f",cand->init_power);
	  slog.csvWrite("opt pow ","%9.5f",cand->power);

	  slog.csvWrite("int sigma","%9.4f",cand->init_sigma);
	  slog.csvWrite("opt sigma","%9.4f",cand->sigma);

	  slog.csvEndLine();


	  //printf("cnd\t%3i\t%14.10f\t%8.3f\t%2i\t%8.6f\t%8.6f\t%14.10f\t%8.3f\t%2i\t%8.6f\t%8.6f\t%8.6f\t%8.6f\n", ii, cand->init_r/T, cand->init_z, cand->init_numharm, cand->init_power, cand->init_sigma, cand->init_r/T, cand->z, cand->numharm, cand->power, cand->sigma /*,pSum, pSum2*/  );
	}

	exit(0); // TMP;
      }
#endif

      // Re sort with new sigma values
      cands = sort_accelcands(cands);

#ifdef CUDA
      if ( sSpec.flags & FLAG_DPG_PRNT_CAND )
      {
	sprintf(name,"%s_GPU_05_Cands_Optemised.csv",fname);
	printCands(name, cands, obs.T);
      }
#endif

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
	{
	  calc_props(cand->derivs[0], cand->r, cand->z, 0.0, props + ii);
	  /* Override the error estimates based on power */
	  props[ii].rerr = (float) (ACCEL_DR) / cand->numharm;
	  props[ii].zerr = (float) (ACCEL_DZ) / cand->numharm;
	}
	listptr = listptr->next;
      }

#ifdef CUDA
      if ( sSpec.flags & FLAG_DPG_PRNT_CAND )
      {
	sprintf(name,"%s_GPU_06_Cands_Optemised_cleaned.csv",fname);
	printCands(name, cands, obs.T);
      }
#endif

      /* Write the fundamentals to the output text file */

#ifdef CUDA
      NV_RANGE_PUSH("Fundamentals");
#endif

      output_fundamentals(props, cands, &obs, &idata);

#ifdef CUDA
      NV_RANGE_POP();
#endif

      /* Write the harmonics to the output text file */

#ifdef CUDA
      NV_RANGE_PUSH("Harmonics");
#endif

      output_harmonics(cands, &obs, &idata);

#ifdef CUDA
      NV_RANGE_POP();
#endif

      /* Write the fundamental fourierprops to the cand file */

      obs.workfile = chkfopen(obs.candnm, "wb");
      chkfwrite(props, sizeof(fourierprops), numcands, obs.workfile);
      fclose(obs.workfile);

      free(props);
      printf("\nDone optimizing.\n\n");

      double N =   obs.N;

#ifdef CBL // Chris: This is some temporary output to generate data for my thesis,
      Fout // TMP
      {
	Fout // Test sigma calculations  .
	{
	  double noVals = 100 ;
	  double base = 0.9998 ;
	  double rest = 1 - base ;

	  double ajustP;
	  double ajustQ;

	  int i;
	  for ( i = 0; i < noVals; i++ )
	  {
	    double p = base + i/noVals*rest ;
	    double q = (noVals-i)/noVals*rest ;

	    calcNQ(q, obs.numindep[0], &ajustP, &ajustQ);

	    printf("%.20lf\t%.20lf\t%.20lf\t%.20lf\n", p, q, ajustP, ajustQ);
	  }
	}

	numcands 	= g_slist_length(cands);
	listptr 	= cands;
	accelcand* cand;

	double T      = obs.T;

	double pSum   = 0;
	double pSum2  = 0;

	for (ii = 0; ii < cuSrch->sSpec->fftInf.nor; ii++)
	{
	  fcomplex bin = cuSrch->sSpec->fftInf.fft[ii];

	  double pow = bin.i*bin.i + bin.r*bin.r;
	  //pow = fabs(bin.i) + fabs(bin.r) ;

	  //pSum  += pow/N*2/sqrt(2);
	  //pSum2 += sqrt(pow)/N*2/sqrt(2);

	  pSum  += pow;
	  pSum2 += sqrt(pow);
	}

	//printf("\n\n pSum %.2f ss %.2f \n", pSum, pSum2);

	printf("\n\nFFT\n");
	printf("SS\t%.5f\n", pSum);
	printf("Sum Norms\t%.5f\n", pSum2);

	printf("\n\n");

	pSum   = 0;
	pSum2  = 0;

	Logger slog(stdout);
	slog.setCsvDeliminator('\t');

	for (ii = 0; ii < numcands; ii++)
	{
	  cand    = (accelcand *) (listptr->data);
	  listptr = listptr->next;

	  Fout
	  {
	    long long baseR = round(cand->init_r);

	    int hn;
	    for ( hn = 1; hn <= cand->numharm; hn++ )
	    {
	      // if ( cuSrch->sSpec->flags & FLAG_DPG_PRNT_CAND )

	      long long idx   = baseR * hn ;
	      float     freq  =  idx / T ;
	      double 	hr = cand->init_r * hn;
	      if ( (idx >= 0) && ( idx <  cuSrch->sSpec->fftInf.nor ) )
	      {
		fcomplex	bin	= cuSrch->sSpec->fftInf.fft[idx - cuSrch->sSpec->fftInf.idx];
		double	norm	= get_scaleFactorZ(cuSrch->sSpec->fftInf.fft, cuSrch->sSpec->fftInf.nor - cuSrch->sSpec->fftInf.idx, idx - cuSrch->sSpec->fftInf.idx, 0, 0.0);

		double	ang1	= atan (bin.r/bin.i) ;
		double	ang2	= atan (bin.i/bin.r) ;

		double	pow	= bin.i*bin.i + bin.r*bin.r;

		double factor = sqrt(norm);
		double pow2   = pow / factor / factor ;
		double sigma  = candidate_sigma_cu(pow2, 1, obs.numindep[0] );

		if ( sigma < 0  )
		{
		  sigma = candidate_sigma_cu(pow2, 1, obs.numindep[0] );
		}

		//printf("%2i\t%.4f\t%lli\t%10.5f\t%9.5f\t%12.5f\t%15.2f\t%.15lf\t%.15lf\n", hn, freq, idx, sigma, pow2, pow, sqrt(pow), ang1, ang2 );

		printf("%2i\t%.4f\t%.4f\n", hn, cand->r * hn / T, cand->z * hn );
		printf("\n");
		double real, imag;
		rz_convolution_cu_inc<double, float2>((float2*)cuSrch->sSpec->fftInf.fft, cuSrch->sSpec->fftInf.idx, cuSrch->sSpec->fftInf.nor, cand->r * hn, cand->z * hn, cand->z * hn * 2 + 4, &real, &imag);

		printf("\n\n\n");

		pSum  += pow;
		pSum2 += sqrt(pow);
	      }

	    }

	  }

	  slog.csvWrite("idx","%i",ii);

	  slog.csvWrite("int freq","%9.7f",cand->init_r/T);
	  slog.csvWrite("opt freq","%9.7f",cand->r/T);

	  slog.csvWrite("int z   ","%9.6f",cand->init_z);
	  slog.csvWrite("opt z   ","%9.6f",cand->z);

	  slog.csvWrite("int harm","%9i",cand->init_numharm);
	  slog.csvWrite("opt harm","%9i",cand->numharm);

	  slog.csvWrite("int pow ","%9.5f",cand->init_power);
	  slog.csvWrite("opt pow ","%9.5f",cand->power);

	  slog.csvWrite("int sigma","%9.4f",cand->init_sigma);
	  slog.csvWrite("opt sigma","%9.4f",cand->sigma);

	  slog.csvEndLine();

	  printf("cnd\t%3i\t%14.10f\t%8.3f\t%2i\t%8.6f\t%8.6f\t%14.10f\t%8.3f\t%2i\t%8.6f\t%8.6f\t%8.6f\t%8.6f\n", ii, cand->init_r/T, cand->init_z, cand->init_numharm, cand->init_power, cand->init_sigma, cand->init_r/T, cand->z, cand->numharm, cand->power, cand->sigma, pSum, pSum2  );
	}
      }
#endif

    }
    else
    {
      printf("No initial candidates above sigma = %.2f were found.\n\n", obs.sigma);
    }

#ifdef CUDA
    NV_RANGE_POP();

    gettimeofday(&end, NULL);
    cuSrch->timings[TIME_ALL_OPT] += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));

#ifdef CBL
    if ( sSpec.flags & FLAG_DPG_PLT_OPT )
    {
      sprintf(dirname,"/home/chris/accel/Nelder_Mead/%s", timeMsg );
      mkdir(dirname, 0755);

      sprintf(scmd,"mv /home/chris/accel/*.png %s/              2>/dev/null", dirname );
      system(scmd);

      sprintf(scmd,"mv /home/chris/accel/*.csv %s/              2>/dev/null", dirname );
      system(scmd);
    }
#endif

#endif

  }

  FOLD 	// Timing message  .
  {
#ifdef CUDA
    if (cuSrch)
    {
      printf("\n*************************************************************************************************\n                            Timing\n*************************************************************************************************\n");

#ifdef CBL
      printf("\n");

      Logger slog(stdout);
      slog.setCsvDeliminator('\t');
      slog.setCsvLineNums(false);

      slog.csvWrite("Timing:","<secs>");
      slog.csvWrite(" Context",  "%9.06f", cuSrch->timings[TIME_CONTEXT]  * 1e-6 );
      slog.csvWrite("  Prep  ",  "%9.06f", cuSrch->timings[TIME_PREP]     * 1e-6 );

      slog.csvWrite(" CPU Ker",  "%9.06f", cuSrch->timings[TIME_CPU_KER]  * 1e-6 );
      slog.csvWrite(" CPU Srch", "%9.06f", cuSrch->timings[TIME_CPU_SRCH] * 1e-6 );
      slog.csvWrite(" CPU Opt",  "%9.06f", cuSrch->timings[TIME_CPU_OPT]  * 1e-6 );

      slog.csvWrite(" GPU ker",  "%9.06f", cuSrch->timings[TIME_GPU_KER]  * 1e-6 );
      slog.csvWrite(" GPU Srch", "%9.06f", cuSrch->timings[TIME_GPU_SRCH] * 1e-6 );
      slog.csvWrite(" GPU Cand", "%9.06f", cuSrch->timings[TIME_CND]      * 1e-6 );
      slog.csvWrite(" GPU opt",  "%9.06f", cuSrch->timings[TIME_GPU_OPT]  * 1e-6 );

      //slog.csvWrite(" Srch All", "%9.06f", cuSrch->timings[TIME_ALL_SRCH] * 1e-6 );
      slog.csvWrite(" Opt All",  "%9.06f", cuSrch->timings[TIME_ALL_OPT]  * 1e-6 );

      slog.csvWrite("    x    ", "%9.06f", ( cuSrch->timings[TIME_CPU_SRCH] + cuSrch->timings[TIME_CPU_OPT] )/ (float)( cuSrch->timings[TIME_GPU_SRCH] + cuSrch->timings[TIME_GPU_OPT] ) );

      slog.csvEndLine();

      printf("\n\n");
#endif

      if ( sSpec.flags & FLAG_TIME )  // Advanced timing massage  .
      {
        int batch, stack;
        float copyH2DT  = 0;
        float InpNorm   = 0;
        float InpFFT    = 0;
        float multT     = 0;
        float InvFFT    = 0;
        float plnCpy    = 0;
        float ss        = 0;
        float resultT   = 0;
        float copyD2HT  = 0;

        printf("\n===========================================================================================================================================\n");
        printf("\nAdvanced timing, all times are in ms\n");

        for (batch = 0; batch < cuSrch->pInf->noBatches; batch++)
        {
          if ( cuSrch->pInf->noBatches > 1 )
            printf("Batch %02i\n",batch);

          cuFFdotBatch*   batches = &cuSrch->pInf->batches[batch];

          float l_copyH2DT  = 0;
          float l_InpNorm   = 0;
          float l_InpFFT    = 0;
          float l_multT     = 0;
          float l_InvFFT    = 0;
          float l_plnCpy    = 0;
          float l_ss        = 0;
          float l_resultT   = 0;
          float l_copyD2HT  = 0;

          FOLD // Heading  .
          {
            printf("\t\t");
            printf("%s\t","Copy H2D");

            if ( batches->flags & CU_NORM_CPU )
              printf("%s\t","Norm CPU");
            else
              printf("%s\t","Norm GPU");

            if ( batches->flags & CU_INPT_FFT_CPU )
              printf("%s\t","Inp FFT CPU");
            else
              printf("%s\t","Inp FFT GPU");

            printf("%s\t","Multiplication");

            printf("%s\t","Inverse FFT");

            printf("%s\t","Sum & Search");

            if ( batches->flags & FLAG_SIG_GPU )
              printf("%s\t","Sigma GPU");
            else
              printf("%s\t","Sigma CPU");

            printf("%s\t","Copy D2H");

            if ( batches->flags & FLAG_SS_INMEM )
            {
              printf("%s\t","Cpy to pln");
            }

            printf("\n");
          }

          for (stack = 0; stack < (int)cuSrch->pInf->batches[batch].noStacks; stack++)
          {
            printf("Stack\t%02i\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f", stack, batches->copyH2DTime[stack], batches->normTime[stack], batches->InpFFTTime[stack], batches->multTime[stack], batches->InvFFTTime[stack], batches->searchTime[stack], batches->resultTime[stack], batches->copyD2HTime[stack]  );
            if ( batches->flags & FLAG_SS_INMEM )
            {
              printf("\t%9.04f", batches->copyToPlnTime[stack]);
            }
            printf("\n");

            l_copyH2DT  += batches->copyH2DTime[stack];
            l_InpNorm   += batches->normTime[stack];
            l_InpFFT    += batches->InpFFTTime[stack];
            l_multT     += batches->multTime[stack];
            l_InvFFT    += batches->InvFFTTime[stack];
            l_plnCpy    += batches->copyToPlnTime[stack];
            l_ss        += batches->searchTime[stack];
            l_resultT   += batches->resultTime[stack];
            l_copyD2HT  += batches->copyD2HTime[stack];
          }

          if ( cuSrch->pInf->noBatches > 1 )
          {
            printf("\t\t---------\t---------\t---------\t---------\t---------\t---------\t---------\t---------");
            if ( batches->flags & FLAG_SS_INMEM )
            {
              printf("\t---------");
            }
            printf("\n");


            printf("\t\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f", l_copyH2DT, l_InpNorm, l_InpFFT, l_multT, l_InvFFT, l_ss, l_resultT, l_copyD2HT );
            if ( batches->flags & FLAG_SS_INMEM )
            {
              printf("\t%9.04f",l_plnCpy);
            }
            printf("\n");
          }

          copyH2DT  += l_copyH2DT;
          InpNorm   += l_InpNorm;
          InpFFT    += l_InpFFT;
          multT     += l_multT;
          InvFFT    += l_InvFFT;
          plnCpy    += l_plnCpy;
          ss        += l_ss;
          resultT   += l_resultT;
          copyD2HT  += l_copyD2HT;
        }
        printf("\t\t---------\t---------\t---------\t---------\t---------\t---------\t---------\t---------");
        if ( cuSrch->pInf->batches->flags & FLAG_SS_INMEM )
        {
          printf("\t---------");
        }
        printf("\n");

        printf("TotalT \t\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f\t%9.04f", copyH2DT, InpNorm, InpFFT, multT, InvFFT, ss, resultT, copyD2HT );
        if ( cuSrch->pInf->batches->flags & FLAG_SS_INMEM )
        {
          printf("\t%9.04f",plnCpy);
        }
        printf("\n");

        printf("\n===========================================================================================================================================\n\n");
      }
    }
#endif
  }

  /* Finish up */

  printf("Searched the following approx numbers of independent points:\n");
  for (ii = 0; ii < obs.numharmstages; ii++)
    printf("  %2d harmonics:  %9lld  cutoff of %5.2f \n", (1 << ii), obs.numindep[ii], obs.powcut[ii]);

  printf("\nTiming summary:\n");

#ifdef CUDA // More timing  .
  printf("     Prep time: %7.3f sec\n",		cuSrch->timings[TIME_PREP]	* 1e-6 );
  if ( cmd->cpuP )
    printf("    CPU search: %7.3f sec\n",	cuSrch->timings[TIME_CPU_SRCH]	* 1e-6 );
  if ( cmd->gpuP )
  {
    printf("  CUDA Context: %7.3f sec\n",	cuSrch->timings[TIME_CONTEXT]	* 1e-6 );
    printf("    GPU search: %7.3f sec\n",	cuSrch->timings[TIME_GPU_SRCH]	* 1e-6 );
  }
  printf("  Optimization: %7.3f sec\n",		cuSrch->timings[TIME_ALL_OPT]	* 1e-6 );
#endif

  tott = times(&runtimes)   / (double) CLK_TCK - tott;
  utim = runtimes.tms_utime / (double) CLK_TCK;
  stim = runtimes.tms_stime / (double) CLK_TCK;
  ttim = utim + stim;
  printf("      CPU time: %.3f sec (User: %.3f sec, System: %.3f sec)\n", ttim, utim, stim);
  printf("    Total time: %.3f sec\n\n", tott);

  printf("Final candidates in binary format are in '%s'.\n",    obs.candnm);
  printf("Final Candidates in a text format are in '%s'.\n\n",  obs.accelnm);

#ifdef CUDA
  freeCuSearch(cuSrch);

  // Debug studff
  for ( int i = 0; i < gSpec.noDevices; i++)
  {
    CUDA_SAFE_CALL(cudaSetDevice(gSpec.devId[i]), "ERROR in cudaSetDevice");
    CUDA_SAFE_CALL(cudaDeviceReset(), "Error in device reset.");
  }

#ifdef NVVP // Stop profiler
  cudaProfilerStop();
#endif
  cudaDeviceReset();
#endif

  free_accelobs(&obs);
  g_slist_foreach(cands, free_accelcand, NULL);
  g_slist_free(cands);

  return (0);
}

