/** @file accelsearch.cpp
 *  @brief The PRESTO frequency domain acceleration search tool
 *
 *  This is the main of the accelsearch application
 *  This version has been modified for GPU searches
 *
 *  The GPU modifications are enclosed in preprocessor directives (#CHDA), enabled in the make file
 *
 *   @author Scott Ransom & Chris Laidler
  *  @bug No known bugs.
 *
 *  Change Log
 *
 *  [2.0.00] []
 *    The GPU branch foreked of presto master on 09/04/2014, Just after the release of PRESTO 2
 *
 *    A lot of development ....
 *
 *  [2.0.01] [2017-01-30]
 *    Beginning of change log (I know its a bit late)
 *    Fixed timing bug allowing timing of file writes in standard  timing
 *
 */


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

#include "cuda_accel.h"
#include "cuda_accel_utils.h"
#endif

#ifdef USEDMALLOC
#include "dmalloc.h"
#endif

#ifdef WITHOMP
#include <omp.h>
#endif



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
  GSList *cands    = NULL;
  Cmdline *cmd;

  FILE *file;

  // Timing vars
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

#ifdef CUDA	// List GPU's & default to GPU

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

  cuSearch*	cuSrch = NULL;
  gpuSpecs*	gSpec = NULL;
  pthread_t	cntxThread = 0;

  TIME // Start the timer  .
  {
    NV_RANGE_PUSH("Prep");
    gettimeofday(&start, NULL);
  }

  if ( cmd->gpuP ) // Initialises CUDA context(s)  .
  {
    printf("      with GPU additions by Chris Laidler\n\n");
    printf("      The GPU version is still under development.\n");
    printf("      If you find any bugs pleas report to:\n");
    printf("            chris.laidler@gmail.com\n\n");

    gSpec	= readGPUcmd(cmd);

    // Initialise CUDA context
    contextInit	+= initCudaContext(gSpec);
  }

#endif

  /* Create the accelobs structure */
  create_accelobs(&obs, &idata, cmd, 1);

#ifdef CUDA	// Initialise CU Search data struct
  cuSrch	= initSearchInfCMD(cmd, &obs, gSpec);
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

#ifdef CUDA // Timing  .
  char fname[1024];

  TIME // Timing  .
  {
    NV_RANGE_POP("Prep");

    gettimeofday(&end, NULL);
    cuSrch->timings[TIME_PREP] += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
  }
#endif

#ifdef CBL  // Unoptcands  .
  char candsFile[1024];
  sprintf(fname,"%s_hs%02i_zmax%06.1f_sig%06.3f", obs.rootfilenm, obs.numharmstages, obs.zhi, obs.sigma );
  sprintf(candsFile,"%s.unoptcands", fname );

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
    long long contextTinme = compltCudaContext(gSpec);

    candsCPU = cands ;		// Just encase we are not doing the candidate generation

    TIME // Timing  .
    {
      cuSrch->timings[TIME_CONTEXT] = contextTinme;
    }
  }
  else								// Run Search  .
#endif
  { // 		-----------------=================== The CPU and GPU main loop ===================----------------------- .
    /* Start the main search loop */

    FOLD // 	-----------------=================== The CPU and GPU main loop ===================----------------------- .
    {
      double startr = obs.rlo, lastr = 0, nextr = 0;
      ffdotpows *fundamental;

      if ( cmd->cpuP )			// --=== The CPU Search == --  .
      {
#ifdef CUDA // Timing  .
	printf("\n*************************************************************************************************\n                         Doing CPU Search\n*************************************************************************************************\n");

	TIME // Timing  .
	{
	  NV_RANGE_PUSH("CPU Srch");
	  NV_RANGE_PUSH("CPU kernel");
	  gettimeofday(&start, NULL);
	}
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

#ifdef CUDA  // Timing  .

	TIME // Timing  .
	{
	  NV_RANGE_POP("CPU kernel");
	  NV_RANGE_PUSH("CPU Cand Gen");

	  gettimeofday(&end, NULL);
	  cuSrch->timings[TIME_CPU_INIT] += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
	}
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

#ifdef CUDA  // Timing  .

	printf("\nCPU found %i initial candidates.", g_slist_length(candsCPU));

	TIME // Timing  .
	{
	  NV_RANGE_POP("CPU Cand Gen");
	  NV_RANGE_POP("CPU Srch");

	  gettimeofday(&end, NULL);
	  cuSrch->timings[TIME_CPU_SRCH] += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
	  cuSrch->timings[TIME_CPU_CND_GEN] = cuSrch->timings[TIME_CPU_SRCH] - cuSrch->timings[TIME_CPU_INIT];

	  printf("In %.4f ms.\n", cuSrch->timings[TIME_CPU_SRCH]/1000.0 );
	}

	printf("\n");

	if ( cuSrch->conf->gen->flags & FLAG_DPG_PRNT_CAND )
	{
	  sprintf(name,"%s_CPU_01_Cands.csv", fname);
	  printCands(name, candsCPU, obs.T);
	}

#endif

      }

      if ( cmd->gpuP )			// --=== The GPU Search == --  .
      {

#ifdef CUDA	// CUDA search

	if ( cmd->cpuP ) // Duplicate the CPU candidates  .
	{
	  candsCPU = duplicate_accelcands(cands);
	}

	cands = generateCandidatesGPU(cuSrch);

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

  FOLD  // Optimisation  .
  {                            /* Candidate list trimming and optimisation */
    printf("\n*************************************************************************************************\n                          Optimizing initial candidates\n*************************************************************************************************\n");
    printf("\n");

    if ( cmd->cpuP && cmd->cpuP ) // Duplicate the CPU candidates  .
    {
      // We did a CPU search we may as well use the CPU candidates
      // TODO: Think about this and put a output message
      cands = candsCPU;
    }

    int numcands;
    GSList *listptr;
    accelcand *cand;
    fourierprops *props;
    numcands = g_slist_length(cands);

#ifdef CUDA	// Timing and debug stuff  .
    char timeMsg[1024], dirname[1024], scmd[1024];
    time_t rawtime;
    struct tm* ptm;

#ifdef CBL
    if ( cuSrch->conf->opt->flags & FLAG_DPG_CAND_PLN )
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

      // Remove empty directories
      rmdir(dirname);
    }
#endif

    TIME // Timing  .
    {  
       NV_RANGE_PUSH("Optimisation All");
       gettimeofday(&start, NULL);       // Note could start the timer after kernel init
    }

    if ( cuSrch->conf->opt->flags & FLAG_DPG_SKP_OPT )
      numcands = 0;
#endif

    if (numcands)
    {
      /* Sort the candidates according to the optimized sigmas */

      cands = sort_accelcands(cands);

#ifdef CUDA
      if ( cuSrch->conf->opt->flags & FLAG_DPG_PRNT_CAND )
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
      if ( cuSrch->conf->opt->flags & FLAG_DPG_PRNT_CAND )
      {
	sprintf(name,"%s_GPU_04_Cands_Thinned.csv",fname);
	printCands(name, cands, obs.T);
      }
#endif

      /* Now optimize each candidate and its harmonics */

      printf("Optimising the remaining %i initial candidates.\n\n", numcands);

      if ( cmd->cpuP ) 	 	// --=== The CPU position refinement == --  .
      {
#ifdef CUDA // Timing  .

	// Doing a GPU search as well so duplicate candidates
	if (cmd->gpuP)
	  cuSrch->cands = duplicate_accelcands(cands);

	TIME // Timing  .
	{
	  NV_RANGE_PUSH("CPU refine");
	  gettimeofday(&start01, NULL);       // Profiling
	}
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

#ifdef CUDA // Timing  .

	TIME // Timing  .
	{
	  NV_RANGE_POP("CPU refine");

	  gettimeofday(&end01, NULL);
	  cuSrch->timings[TIME_CPU_REFINE] += (end01.tv_sec - start01.tv_sec) * 1e6 + (end01.tv_usec - start01.tv_usec);
	}
#endif

      }

      if ( cmd->gpuP )        	// --=== The GPU position refinement == --  .
      {
#ifdef CUDA
	if ( cmd->cpuP )
	{
	  // Get the pre-optimisation candidates
	  cands = duplicate_accelcands(cuSrch->cands);
	}

	TIME // Timing  .
	{
	  NV_RANGE_PUSH("GPU refine");
	  gettimeofday(&start01, NULL);       // Profiling
	}

	// Initialise optimisation details!
	initCuOpt(cuSrch);

	// Optimise all the candidates
	optList(cands, cuSrch);

	TIME // Timing  .
	{
	  NV_RANGE_POP("GPU refine");

	  gettimeofday(&end01, NULL);
	  cuSrch->timings[TIME_GPU_REFINE] += (end01.tv_sec - start01.tv_sec) * 1e6 + (end01.tv_usec - start01.tv_usec);
	}

#else
	fprintf(stderr,"ERROR: not compiled with CUDA!\n");
#endif
      }

      printf("\n\n");

#ifdef CBL
      Fout // TMP - Some info on optimised candidate
      {
	slog.setCsvDeliminator('\t');
	listptr 	= cands;
	double T	= obs.T;

	for (ii = 0; ii < numcands; ii++)
	{
	  cand    = (accelcand *) (listptr->data);
	  listptr = listptr->next;

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

      }
#endif

#ifdef CUDA
      TIMING // Basic timing  .
      {
	NV_RANGE_PUSH("props");
      }
#endif

      // Re sort with new sigma values
      cands = sort_accelcands(cands);

#ifdef CUDA
      if ( cuSrch->conf->opt->flags & FLAG_DPG_PRNT_CAND )
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
      if ( cuSrch->conf->opt->flags & FLAG_DPG_PRNT_CAND )
      {
	sprintf(name,"%s_GPU_06_Cands_Optemised_cleaned.csv",fname);
	printCands(name, cands, obs.T);
      }
#endif

      /* Write the fundamentals to the output text file */

#ifdef CUDA		// Basic timing  .
      TIME // Basic timing  .
      {
	NV_RANGE_POP("props");
	NV_RANGE_PUSH("Write");
	NV_RANGE_PUSH("Fundamentals");

	gettimeofday(&start01, NULL);       // Note could start the timer after kernel init
      }
#endif

      output_fundamentals(props, cands, &obs, &idata);

#ifdef CUDA
      TIME // Basic timing  .
      {
	NV_RANGE_POP("Fundamentals");
      }
#endif

      /* Write the harmonics to the output text file */

#ifdef CUDA
      TIME // Basic timing  .
      {
	NV_RANGE_PUSH("Harmonics");
      }
#endif

      output_harmonics(cands, &obs, &idata);

#ifdef CUDA
      TIME // Basic timing  .
      {
	NV_RANGE_POP("Harmonics");
	NV_RANGE_PUSH("props");
      }
#endif

      /* Write the fundamental fourierprops to the cand file */

      obs.workfile = chkfopen(obs.candnm, "wb");
      chkfwrite(props, sizeof(fourierprops), numcands, obs.workfile);
      fclose(obs.workfile);

#ifdef CUDA 		// Basic timing  .
      TIME // Basic timing  .
      {
	NV_RANGE_POP("props");
	NV_RANGE_POP("Write");

  	gettimeofday(&end01, NULL);
	float v1 =  (end01.tv_sec - start01.tv_sec) * 1e6 + (end01.tv_usec - start01.tv_usec);
	cuSrch->timings[TIME_OPT_FILE_WRITE] += v1;
      }
#endif

      free(props);
      printf("\nDone optimizing.\n\n");

      double N =   obs.N;
    }
    else
    {
      printf("No initial candidates above sigma = %.2f were found.\n\n", obs.sigma);
    }

#ifdef CUDA	// Timing and debug stuff  .

    TIME // Timing  .
    {
      NV_RANGE_POP("Optimisation All");	// Optimisation All

      gettimeofday(&end, NULL);
      cuSrch->timings[TIME_ALL_OPT] += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
    }

#ifdef CBL
    if ( cuSrch->conf->opt->flags & FLAG_DPG_CAND_PLN )
    {
      sprintf(dirname,"/home/chris/accel/Nelder_Mead/%s-pst", timeMsg );
      mkdir(dirname, 0755);

      sprintf(scmd,"mv /home/chris/accel/*.png %s/ 2> /dev/null", dirname );
      system(scmd);

      sprintf(scmd,"mv /home/chris/accel/*.csv %s/ 2> /dev/null", dirname );
      system(scmd);

      // Remove empty directories
      rmdir(dirname);
    }
#endif

#endif

  }

  FOLD 	// Timing message  .
  {
#ifdef CUDA
    if (cuSrch)
    {
#ifdef CBL

      TIME // Timing  .
      {
	printf("\n*************************************************************************************************\n                            Timing\n*************************************************************************************************\n");

	printf("\n");

	Logger slog(stdout);
	slog.setCsvDeliminator('\t');
	slog.setCsvLineNums(false);

	if ( cmd->cpuP )
	  cuSrch->timings[TIME_CPU_OPT] = cuSrch->timings[TIME_ALL_OPT] - cuSrch->timings[TIME_GPU_REFINE];
	if ( cmd->gpuP )
	  cuSrch->timings[TIME_GPU_OPT] = cuSrch->timings[TIME_ALL_OPT] - cuSrch->timings[TIME_CPU_REFINE];

	char pres[] =  "%15.06f";

	slog.csvWrite("Timing:",   "<secs>");

	slog.csvWrite("      Context",  pres, cuSrch->timings[TIME_CONTEXT]		* 1e-6 );
	slog.csvWrite("         Prep",  pres, cuSrch->timings[TIME_PREP]		* 1e-6 );

	slog.csvWrite("     CPU Init",  pres, cuSrch->timings[TIME_CPU_INIT]		* 1e-6 );
	slog.csvWrite("      CPU Gen",  pres, cuSrch->timings[TIME_CPU_CND_GEN]		* 1e-6 );
	slog.csvWrite("      CPU Opt",  pres, cuSrch->timings[TIME_CPU_OPT] 		* 1e-6 );
	slog.csvWrite("     CPU Time",  pres, (cuSrch->timings[TIME_CPU_SRCH] + cuSrch->timings[TIME_CPU_OPT]) * 1e-6 );

	slog.csvWrite("     GPU Init",  pres, cuSrch->timings[TIME_GPU_INIT]		* 1e-6 );
	slog.csvWrite("      GPU Gen",  pres, cuSrch->timings[TIME_GPU_CND_GEN]		* 1e-6 );
	slog.csvWrite("      GPU Opt",  pres, cuSrch->timings[TIME_GPU_OPT]		* 1e-6 );
	slog.csvWrite("     GPU Time",  pres, (cuSrch->timings[TIME_GPU_SRCH] + cuSrch->timings[TIME_GPU_OPT]) * 1e-6 );
	slog.csvWrite("         x   ",  pres, (cuSrch->timings[TIME_CPU_SRCH] + cuSrch->timings[TIME_CPU_OPT])/(float)(cuSrch->timings[TIME_GPU_SRCH] + cuSrch->timings[TIME_GPU_OPT]) );

	slog.csvWrite("     GPU Srch",  pres, cuSrch->timings[TIME_GPU_SRCH]		* 1e-6 );
	slog.csvWrite("      GPU Pln",  pres, cuSrch->timings[TIME_GPU_PLN]		* 1e-6 );
	slog.csvWrite("      GPU S&S",  pres, cuSrch->timings[TIME_GPU_SS]		* 1e-6 );
	slog.csvWrite("     GPU Cand",  pres, cuSrch->timings[TIME_CND]			* 1e-6 );
	slog.csvWrite("     Gen Wait",  pres, cuSrch->timings[TIME_GEN_WAIT]		* 1e-6 );

	slog.csvWrite("     CPU Srch",  pres, cuSrch->timings[TIME_CPU_SRCH]		* 1e-6 );

	slog.csvWrite("      Opt All",  pres, cuSrch->timings[TIME_ALL_OPT]		* 1e-6 );
	slog.csvWrite("     CPU Rfne",  pres, cuSrch->timings[TIME_CPU_REFINE]		* 1e-6 );
	slog.csvWrite("     GPU Rfne",  pres, cuSrch->timings[TIME_GPU_REFINE]		* 1e-6 );
	slog.csvWrite("     Opt Wait",  pres, cuSrch->timings[TIME_OPT_WAIT]		* 1e-6 );
	slog.csvWrite("       Output",  pres, cuSrch->timings[TIME_OPT_FILE_WRITE]	* 1e-6 );
	slog.csvEndLine();

	printf("\n\n");
      }

      PROF // Profiling  .
      {
	if ( cuSrch->conf->opt->flags & FLAG_PROF )  // Advanced timing massage  .
	{
	  char pres[] =  "%15.06f";

	  int batch, stack;
	  float sums[COMP_GEN_END];
	  for ( int i = 0; i < COMP_GEN_END; i++)
	    sums[i] = 0;

	  printf("\n*************************************************************************************************\n                           Profiling\n*************************************************************************************************\n");

	  printf("\nAll times are in seconds\n\n");

	  FOLD // Basic profiling  .
	  {
	    Logger slog(stdout);
	    slog.setCsvDeliminator('\t');
	    slog.setCsvLineNums(false);

	    slog.csvWrite("Comps:",   "<cmps>");
	    slog.csvWrite("         Resp",  pres, cuSrch->timings[COMP_RESP]		* 1e-6 );
	    slog.csvWrite("      Ker FFT",  pres, cuSrch->timings[COMP_KERFFT]		* 1e-6 );
	    slog.csvWrite("     Refine 1",  pres, cuSrch->timings[COMP_OPT_REFINE_1]	* 1e-6 );
	    slog.csvWrite("     Refine 2",  pres, cuSrch->timings[COMP_OPT_REFINE_2]	* 1e-6 );
	    slog.csvWrite("       Derivs",  pres, cuSrch->timings[COMP_OPT_DERIVS]	* 1e-6 );
	    slog.csvEndLine();
	    printf("\n\n");
	  }

	  if ( !useUnopt )
	  {
	    cuFFdotBatch*   batches = &cuSrch->pInf->batches[0];

	    printf("\n --== Batches ==-- \n");

	    FOLD // Heading  .
	    {
	      char heads[COMP_GEN_END][15];

	      sprintf(heads[COMP_GEN_H2D],		"Copy H2D");

	      if      ( batches->flags & CU_NORM_GPU_SM )
		sprintf(heads[COMP_GEN_NRM],		"Norm GPU SM");
	      else if ( batches->flags & CU_NORM_GPU_OS )
		sprintf(heads[COMP_GEN_NRM],		"Norm GPU RDIX");
	      else
		sprintf(heads[COMP_GEN_NRM],		"Norm CPU");

	      sprintf(heads[COMP_GEN_GINP],		"GPU Input");

	      sprintf(heads[COMP_GEN_CINP],		"CPU Input");

	      sprintf(heads[COMP_GEN_MEM],		"Input Mem");

	      if ( batches->flags & CU_INPT_FFT_CPU )
		sprintf(heads[COMP_GEN_FFT],		"Inp FFT CPU");
	      else
		sprintf(heads[COMP_GEN_FFT],		"Inp FFT GPU");

	      sprintf(heads[COMP_GEN_MULT],		"Multiplication");

	      sprintf(heads[COMP_GEN_IFFT],		"Inverse FFT");

	      sprintf(heads[COMP_GEN_D2D],		"Copy D2D");

	      sprintf(heads[COMP_GEN_SS],			"Sum & Search");

	      sprintf(heads[COMP_GEN_D2H],		"Copy D2H");

	      sprintf(heads[COMP_GEN_STR],		"iCand storage");

	      printf("\t\t");
	      for ( int i = 0; i < COMP_GEN_END; i++)
		printf("%15s\t", heads[i] );

	      printf("\n");
	    }

	    for (batch = 0; batch < cuSrch->pInf->noBatches; batch++)
	    {
	      printf("Batch\t%02i\n",batch);

	      batches = &cuSrch->pInf->batches[batch];

	      float	bsums[COMP_GEN_END];
	      for ( int i = 0; i < COMP_GEN_END; i++)
		bsums[i] = 0;

	      for (stack = 0; stack < (int)cuSrch->pInf->batches[batch].noStacks; stack++)
	      {

		printf("Stack\t%02i\t", stack);

		for ( int i = 0; i < COMP_GEN_END; i++)
		{
		  float val = batches->compTime[i*batches->noStacks+stack];
		  val *= 1e-6; // Convert to us
		  printf("%15.06f\t", val );
		  bsums[i] += val;
		}
		printf("\n");
	      }

	      if ( cuSrch->pInf->noBatches > 1 )
	      {
		printf("\t\t");
		for ( int i = 0; i < COMP_GEN_END; i++)
		  printf("%15s\t", "---------" );
		printf("\n");

		printf("\t\t");
		for ( int i = 0; i < COMP_GEN_END; i++)
		{
		  printf("%15.06f\t", bsums[i] );
		}
		printf("\n");
	      }

	      for ( int i = 0; i < COMP_GEN_END; i++)
	      {
		sums[i] += bsums[i];
	      }

	    }
	    printf("\t\t");
	    for ( int i = 0; i < COMP_GEN_END; i++)
	      printf("%15s\t", "---------" );
	    printf("\n");

	    printf("TotalT\t\t");
	    for ( int i = 0; i < COMP_GEN_END; i++)
	    {
	      printf("%15.06f\t", sums[i] );
	    }
	    printf("\n");
	  }

	  printf("\n*************************************************************************************************\n\n");
	}
      }
#endif
    }
#endif
  }

  /* Finish up */

  printf("Searched the following approx numbers of independent points:\n");
  for (ii = 0; ii < obs.numharmstages; ii++)
    printf("  %2d harmonics:  %9lld  Threshold power: %5.2f \n", (1 << ii), obs.numindep[ii], obs.powcut[ii] );

  printf("\nTiming summary:\n");

#ifdef CUDA // More timing  .

  TIME // Timing that does not need the CBL libraries  .
  {
    printf("     Prep time: %7.3f sec\n",	cuSrch->timings[TIME_PREP]		* 1e-6 );
    if ( cmd->cpuP )
      printf("    CPU search: %7.3f sec\n",	cuSrch->timings[TIME_CPU_SRCH]		* 1e-6 );
    if ( cmd->gpuP )
    {
      printf("  CUDA Context: %7.3f sec\n",	cuSrch->timings[TIME_CONTEXT]		* 1e-6 );
      printf("    GPU search: %7.3f sec\n",	cuSrch->timings[TIME_GPU_SRCH]		* 1e-6 );
    }
    printf("  Optimization: %7.3f sec\n",	cuSrch->timings[TIME_ALL_OPT]		* 1e-6 );
  }
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

  // Debug stuff
  for ( int i = 0; i < gSpec->noDevices; i++)
  {
    CUDA_SAFE_CALL(cudaSetDevice(gSpec->devId[i]), "ERROR in cudaSetDevice");
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


