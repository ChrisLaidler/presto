extern "C"
{
#include "accel.h"
}

#ifdef USEMMAP
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

#ifdef CUDA
#include "cuda_accel.h"
#include "cuda_accel_utils.h"

#include "cuda_response.h"
#include "cuda_cand_OPT.h"
#endif

#include <sys/time.h>
#include <time.h>

#ifdef USEDMALLOC
#include "dmalloc.h"
#endif

#ifdef CBL
#include "array.h"
#include "arrayDsp.h"
#include "util.h"
#endif


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
    double norm,
    nDarray<1, float> *input,
    nDarray<2, float> *complex,
    nDarray<2, float> *powers
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
  ffdot->zlo = cu_calc_required_z<double>(harm_fract, obs->zlo, 2.0);

  /* Initialize the lookup indices */
  if (numharm > 1 && !obs->inmem) {
    double rr, subr;
    for (ii = 0; ii < numrs_full; ii++) {
      rr = fullrlo + ii * ACCEL_DR;
      subr = calc_required_r(harm_fract, rr);
      shi->rinds[ii] = cu_index_from_r(subr, ffdot->rlo);
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
  nice_numdata = cu_next2_to_n(numdata);  // for FFTs
  data = get_fourier_amplitudes(lobin, nice_numdata, obs);
  if (!obs->mmap_file && !obs->dat_input && 0)
    printf("This is newly malloc'd!\n");

  // Normalize the Fourier amplitudes

  if ( obs->nph > 0.0 )
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

    if( !norm )
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
      for (ii = 0; ii < numdata; ii++)
      {
        data[ii].r *= norm;
        data[ii].i *= norm;
      }
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
      if ( fftlen != (*input)(0)->len()/2.0 )
      {
        printf("ERROR: numdata != length on input data!\n");
      }
      memcpy(input->elems, corrd->dataarray, fftlen*2*sizeof(float) );
    }
    memcpy(complex->getP(0,ii), result[ii], nrs*2*sizeof(float) );
    datainf = SAME;
  }

  // Always free data
  vect_free(data);
  clearCorrData(corrd);

  /* Convert the amplitudes to normalized powers */

  ffdot->powers = gen_fmatrix(ffdot->numzs, ffdot->numrs);
  for (ii = 0; ii < (ffdot->numzs * ffdot->numrs); ii++)
    ffdot->powers[0][ii] = POWER(result[0][ii].r, result[0][ii].i);

  for (ii = 0; ii < ffdot->numzs; ii++)
  {
    float *tmp = powers->getP(0,ii);

    memcpy(powers->getP(0,ii), ffdot->powers[ii], ffdot->numrs*sizeof(float) );
  }

  vect_free(result[0]);
  vect_free(result);

  return ffdot;
}

/** A function to comparer CPU and GPU candidates
 * It looks for candidates missed by each search
 * and comparer the ration of sigma's of the various detections
 */
void compareCands(GSList *candsCPU, GSList *candsGPU, double T)
{
  printf("\nComapreing GPU and CPU raw candidate lists\n");

  if (  candsCPU == NULL &&  candsGPU == NULL )
  {
    printf("No candidates found, try searching some noisy data\n");

    return;
  }

  if (  candsCPU != NULL &&  candsGPU == NULL )
  {
    GSList *tmp_list = candsCPU;
    int cands = 0;

    while (tmp_list->next)
    {
      cands++;
      tmp_list = tmp_list->next;
    }

    printf("CPU search found %i candidates and GPU found nothing\n",cands);
    printf("Writing candidates to CPU_Cands.csv\n");
    printCands("CPU_Cands.csv", candsCPU, T);

    return;
  }

  if (  candsCPU == NULL &&  candsGPU != NULL )
  {
    GSList *gpuLst = candsGPU;
    int gpuCands = 0;

    while (gpuLst->next)
    {
      gpuCands++;
      gpuLst = gpuLst->next;
    }

    printf("GPU search found %i candidates and CPU found nothing\n",gpuCands);
    printf("Writing candidates to GPU_Cands.csv\n");
    printCands("GPU_Cands.csv", candsCPU, T);

    return;
  }

  GSList *gpuLst = candsGPU;
  GSList *cpuLst = candsCPU;
  int gpuCands = 0;
  int cpuCands = 0;

  while (cpuLst->next)
  {
    cpuCands++;
    cpuLst = cpuLst->next;
  }
  printf("Got: %5i CPU candidates\n",cpuCands);

  while (gpuLst->next)
  {
    gpuCands++;
    gpuLst = gpuLst->next;
  }
  printf("Got: %5i GPU candidates\n",gpuCands);

  printf("\nWriting candidates to CPU_Cands.csv and GPU_Cands.csv\n");
  printCands("CPU_Cands.csv", candsCPU, T);
  printCands("GPU_Cands.csv", candsGPU, T);

  GSList *tmp_list = candsGPU;

  double sigDsit = 0.003 ;
  double ratio   = 0.01 ;

  double d1 = -1, d2 = -1;
  double s1 = 0,  s2 = 0;
  double r1, r2;

  accelcand *cpu1,*cpu2;
  accelcand *cpu1R,*cpu2R;
  accelcand *gcpu,*gpu1,*gpu2;

  int gpuBetterThanCPU  = 0;
  int cpuBetterThanGPU  = 0;
  int gpuMissed         = 0;
  int cpuMissed         = 0;
  int silar             = 0;
  int superseede        = 0;

  FOLD // Loop through CPU results  .
  {
    GSList *cpul = candsCPU;
    while (cpul->next)
    {
      cpu1 = (accelcand*)cpul->data;
      r1 = 10;
      r2 = 10;

      double rr1 = 10;
      double rr2 = 10;
      double rrMin;
      cpu1R = NULL;
      cpu2R = NULL;

      FOLD // Get Neighbouring GPU vals  .
      {
        GSList *gpul = candsGPU;
        gpu1 = NULL;
        while (gpul->next && ((accelcand*)gpul->data)->r < cpu1->r )
        {
          gpu1 = (accelcand*)gpul->data;
          gpul = gpul->next;
        }

        if(gpu1)
        {
          d1 = fabs(cpu1->r - gpu1->r );
          s1 = gpu1->sigma - cpu1->sigma;
          r1 = std::min<double>(r1,fabs(1-cpu1->sigma/gpu1->sigma));
        }
        else
        {
          d1 = ACCEL_CLOSEST_R;
          s1 = 0;
        }

        if(gpul->data)
        {
          gpu2 = (accelcand*)gpul->data;
          d2 = fabs(cpu1->r - gpu2->r );
          s2 = gpu2->sigma - cpu1->sigma;
          r2 = std::min<double>(r2,fabs(1-cpu1->sigma/gpu2->sigma));
        }
        else
        {
          gpu2 = NULL;
          d2 = ACCEL_CLOSEST_R;
          s2 = 0;
        }

        FOLD  // Loop through CPU again to get correct ratios
        {
          GSList *cpuLst2 = candsCPU;

          while (cpuLst2->next)
          {
            cpu2 = (accelcand *)cpuLst2->data;
            if (gpu1)
            {
              if (cpu2->r < gpu1->r + ACCEL_CLOSEST_R && cpu2->r > gpu1->r - ACCEL_CLOSEST_R )
              {
                double nr = fabs(1-cpu2->sigma/gpu1->sigma);
                if (nr < rr1 )
                {
                  cpu1R   = cpu2;
                  rr1     = nr;
                }
              }
            }

            if (gpu2)
            {
              if ( cpu2->r < gpu2->r + ACCEL_CLOSEST_R && cpu2->r > gpu2->r - ACCEL_CLOSEST_R  )
              {
                double nr = fabs(1-cpu2->sigma/gpu2->sigma);
                if (nr < rr2 )
                {
                  cpu2R   = cpu2;
                  rr2     = nr;
                }
              }

              if (cpu2->r > gpu2->r + ACCEL_CLOSEST_R )
                break;
            }

            cpuLst2 = cpuLst2->next;
          }
        }

      }

      bool coverd = false ;
      bool super  = false ;
      if ( d1 < ACCEL_CLOSEST_R )
      {
        coverd = true;
        if ( s1 > -sigDsit)
          super = true;
      }
      if ( d2 < ACCEL_CLOSEST_R )
      {
        coverd = true;
        if ( s2 > -sigDsit)
          super = true;
      }
      double  minr  = std::min(r1,r2);
      rrMin = std::min(rr1,rr2);

      if       ( !coverd )
      {
        printf("\n");

        if(gpu1)
          printf("       below  r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", gpu1->r, gpu1->z, gpu1->numharm, gpu1->sigma, gpu1->power );

        printf("CPU candidate r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f   Not found in GPU candidates.\n", cpu1->r, cpu1->z, cpu1->numharm, cpu1->sigma, cpu1->power );

        if (gpu2)
          printf("       above  r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", gpu2->r, gpu2->z, gpu2->numharm, gpu2->sigma, gpu2->power );

        gpuMissed++;
      }
      else if  ( minr < ratio )
      {
        // Close relation
        silar++;
      }
      else if  ( super )
      {
        // There is a better GPU candidate
        superseede++;
      }
      else
      {
        printf("\n");
        cpuBetterThanGPU++;

        if ( r1 < r2 )
        {
          printf("↓ %5.3f GPU candidate has a lower sigma by %.3f \n", r1, s1);
          printf("         CPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", cpu1->r, cpu1->z, cpu1->numharm, cpu1->sigma, cpu1->power );
          printf("         GPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", gpu1->r, gpu1->z, gpu1->numharm, gpu1->sigma, gpu1->power );
        }
        else
        {
          printf("↓ %5.3f GPU candidate has a lower sigma by %.3f \n", r2, s2);
          printf("         CPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", cpu1->r, cpu1->z, cpu1->numharm, cpu1->sigma, cpu1->power );
          printf("         GPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", gpu2->r, gpu2->z, gpu2->numharm, gpu2->sigma, gpu2->power );
        }
      }

      cpul = cpul->next;
    }
  }

  FOLD // Loop through GPU results  .
  {
    GSList *gpul = candsCPU;
    while (gpul->next)
    {
      gpu1 = (accelcand*)gpul->data;

      GSList *cpul = candsCPU;
      cpu1 = NULL;
      r1 = 10;
      r2 = 10;

      while (cpul->next && ((accelcand*)cpul->data)->r <= gpu1->r )
      {
        cpu1 = (accelcand*)cpul->data;
        if ( cpu1->r > gpu1->r - ACCEL_CLOSEST_R )
          r1 = std::min<double>(r1,fabs(1-cpu1->sigma/gpu1->sigma) );
        cpul = cpul->next;
      }

      GSList *cpul2 = candsCPU;
      while (cpul2->next && ((accelcand*)cpul2->data)->r < gpu1->r + ACCEL_CLOSEST_R )
      {
        cpu2 = (accelcand*)cpul2->data;
        r2 = std::min<double>(r2,fabs(1-cpu2->sigma/gpu1->sigma) );
        cpul2 = cpul2->next;
      }

      double minr = std::min(r1,r2);

      if(cpu1)
        d1 = fabs(cpu1->r - gpu1->r );
      else
        d1 = ACCEL_CLOSEST_R+1 ;

      if (cpul->next)
        cpu2 = (accelcand*)cpul->next->data;
      else
        cpu2 = NULL;

      if(cpu2)
        d2 = fabs(cpu2->r - gpu1->r );
      else
        d2 = ACCEL_CLOSEST_R+1 ;

      if (d1 >=ACCEL_CLOSEST_R && d2 >= ACCEL_CLOSEST_R )
      {
        cpuMissed++;
        printf("\n");

        if(cpu1)
          printf("       below  r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", cpu1->r, cpu1->z, cpu1->numharm, cpu1->sigma, cpu1->power );

        printf("GPU candidate r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f   Not found in CPU candidates.\n", gpu1->r, gpu1->z, gpu1->numharm, gpu1->sigma, gpu1->power );

        if (cpu2)
          printf("       above  r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", cpu2->r, cpu2->z, cpu2->numharm, cpu2->sigma, cpu2->power );
      }
      else if ( minr > ratio )
      {
        gpuBetterThanCPU++;

        printf("\n");
        if ( r1 < r2 )
        {
          printf("  %5.3f GPU candidate is significantly higher that all covered CPU candidates.\n", r1);
          printf("         CPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", cpu1->r, cpu1->z, cpu1->numharm, cpu1->sigma, cpu1->power );
          printf("         GPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", gpu1->r, gpu1->z, gpu1->numharm, gpu1->sigma, gpu1->power );
        }
        else
        {
          printf("  %5.3f GPU candidate is significantly higher that all covered CPU candidates\n", r2);
          printf("         CPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", cpu2->r, cpu2->z, cpu2->numharm, cpu2->sigma, cpu2->power );
          printf("         GPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", gpu1->r, gpu1->z, gpu1->numharm, gpu1->sigma, gpu1->power );
        }
      }
      gpul = gpul->next;
    }
  }

  printf("\nSummary:\n");
  printf("%4i (%05.2f%%) Similar candidates were found\n",silar+superseede,(silar+superseede)/(float)cpuCands*100.0);
  printf("%4i (%05.2f%%) CPU candidates that were 'covered' by the GPU list.\n",superseede,superseede/(float)cpuCands*100.0);
  printf("%4i (%05.2f%%) Missed by GPU  <- These may be due to the way the dynamic list is created. Check CU_CAND_ARR.csv.\n",gpuMissed,gpuMissed/(float)cpuCands*100.0);
  printf("%4i (%05.2f%%) Missed by CPU  <- These may be due to the way the dynamic list is created.\n",cpuMissed,cpuMissed/(float)cpuCands*100.0);
  printf("%4i (%05.2f%%) Where the GPU sigma was significantly better that the CPU.\n",gpuBetterThanCPU, gpuBetterThanCPU/(float)cpuCands*100.0);
  printf("%4i (%05.2f%%) Where the CPU sigma was significantly better that the GPU.\n",cpuBetterThanGPU, cpuBetterThanGPU/(float)cpuCands*100.0);
  printf("\n");
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
  long long contextInit = 0, prepTime = 0, cupTime = 0, gpuTime = 0, optTime = 0;
  struct timeval start, end, timeval;

  /* Prep the timer */

  tott = times(&runtimes) / (double) CLK_TCK;

#ifdef CUDA // Profiling  .
  gettimeofday(&start, NULL);       // Profiling
  NV_RANGE_PUSH("Prep");
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

#ifdef DEBUG
  showOptionValues();
#endif


  cmd = parseCmdline(argc, argv);

#ifdef CUDA
  if (cmd->lsgpuP) // List GPU's  .
  {
    listDevices();
  }
#endif

  printf("\n\n");
  printf("    Fourier-Domain Acceleration Search Routine\n");
  printf("               by Scott M. Ransom\n");
#ifdef CUDA
  printf("              With GPU additions by:\n");
  printf("                 Chris Laidler\n");
#endif
  printf("\n");

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

  printf("Searching with up to %d harmonics summed:\n", 1 << (obs.numharmstages - 1));
  printf("  f = %.1f to %.1f Hz\n", obs.rlo / obs.T, obs.rhi / obs.T);
  printf("  r = %.1f to %.1f Fourier bins\n", obs.rlo, obs.rhi);
  printf("  z = %.1f to %.1f Fourier bins drifted\n\n", obs.zlo, obs.zhi);

#ifdef CUDA     // Profiling  .
  int  iret1 = 1;

  NV_RANGE_POP();
  gettimeofday(&end, NULL);
  prepTime += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);

  cuSearch*     cuSrch = NULL;
  gpuSpecs      gSpec;
  confSpecsGen   sSpec;
  pthread_t     cntxThread = 0;

  gSpec               = readGPUcmd(cmd);
  sSpec               = readSrchSpecs(cmd, &obs);
  sSpec.planeWidth        = ACCEL_USELEN;			// NB: must have same accellen for tests!
  sSpec.flags        |= CU_NORM_EQUIV;
  sSpec.flags        &= ~FLAG_KER_HIGH;
  sSpec.flags        &= ~FLAG_KER_MAX;
  sSpec.flags        &= ~FLAG_CENTER;
  sSpec.flags        |= FLAG_SEPSRCH;
  sSpec.flags        |= FLAG_SYNCH;			// Synchronous

  contextInit        += initCudaContext(&gSpec);
#endif

  char fname[1024];
  sprintf(fname,"%s_hs%02i_zmax%06.1f_sig%06.3f", obs.rootfilenm, obs.numharmstages, obs.zhi, obs.sigma );

  char candsFile[1024];
  sprintf(candsFile,"%s.unoptcands", fname );

  FILE *file;
  if ( (file = fopen(candsFile, "rb")) && useUnopt )			// Read candidates from previous search  . // TMP
  {
    int numcands;
    size_t read;
    read = fread( &numcands, sizeof(numcands), 1, file );
    int nc = 0;

    printf("\nReading %i raw candies from \"%s\" previous search.\n", numcands, candsFile);

    for (nc = 0; nc < numcands; nc++)
    {
      accelcand* newCnd = (accelcand*)malloc(sizeof(accelcand));
      read = fread( newCnd, sizeof(accelcand), 1, file );

      cands=insert_accelcand(cands,newCnd);
    }
    fclose(file);
  }
  else									// Run Search  .
  {
    long long badInp		= 0;
    long long badCplx		= 0;

    long long badCands		= 0;
    long long similarCands	= 0;
    long long sameCands		= 0;

    /* Start the main search loop */

    FOLD 	                                                    //  -- Main Loop --
    {
      ffdotpows *fundamental;
      double startr = obs.rlo, lastr = 0, nextr = 0;
      int maxxx;
      int ss = 0;

      int noHarms   = (1 << (obs.numharmstages - 1));
      candsGPU      = NULL;

      int noEl = 8;
      fcomplex *tempkern;
      char fname[1024];

      cuFFdotBatch* master   = NULL;    // The first kernel stack created
      subharminfo **subharminfs;

      float buckets[7] = {
          1e-2,      // BAD!    Not even in the same realm.
          1e-3,      // Bad.
          1e-4,      // Bad.   But not that bad.
          1e-5,      // Close  But not great.
          1e-6,      // GOOD   But a bit high.
          1e-7,      // GOOD
          1e-8       // GOOD   Very good.
      };

      NV_RANGE_PUSH("CPU");
      gettimeofday(&start, NULL); // Note could start the timer after kernel init

      // Wait for context thread to finish  .
      contextInit += compltCudaContext(&gSpec);

      //cudaDeviceSynchronize();          // This is only necessary for timing
      gettimeofday(&start, NULL);       // Profiling
      //cudaProfilerStart();              // Start profiling, only really necessary debug and profiling, surprise surprise

      FOLD // Kernel stuff  .
      {
	double z 	= 0;

	double realD;
	double imagD;

	float real;
	float imag;

	Fout // TMP testing stuff response  .
	{
	  //calc_response_off<double>(0.5, 0, &realD, &imagD);

	  double r 	= 100.0;
	  double z 	= 1e-7;
	  float hm 	= 10;

	  double2*  gpuker = (double2*)malloc(sizeof(double2*)*hm*2);

	  float real;
	  float imag;

	  double realD;
	  double imagD;

	  printf("\n\n");

	  double param, fractpart, intpart;
	  fractpart = modf (r , &intpart);

	  int 	nofbin = 2;
	  int	sart = r - nofbin / 2;
	  float step = 0.01;

	  int 	hw = sSpec->fft->noBins*2;
	  hw = 10;

	  //rz_interp_cu<double, float2>((float2*)sSpec.fftInf.fft, sSpec.fftInf.idx, sSpec.fftInf.nor, 97.99, 0, hw, &realD, &imagD );

	  //rz_interp_cu<double, float2>((float2*)sSpec.fftInf.fft, sSpec.fftInf.idx, sSpec.fftInf.nor, 98.01, 0, hw, &realD, &imagD );

	  //printf("r\t%s\t%s z %.4f\t%s\t\t\t%s\t%s\n","Fourier Bins", "Correlation", z, "Power", "Fourier Interpolation", "Power");
	  printf("   r\t%s\t", "Fourier Bins");
	  printf("%s z %1.2e Float \t%s\t\t\t", "Correlation", z, "Power");
	  printf("%s z %1.2e Double\t%s\t\t\t", "Correlation", z, "Power");
	  //printf("%s Float \t%s\t\t",  "Fourier Interpolation", "Power");
	  printf("\n");

	  for ( int ix = 0; ix <= nofbin; ix++ )
	  {
	    printf("%.6f\t%.6f\t%.6f\t%.6f\t", (float)(sart+ix), sSpec->fft->data[(int)sart+ix].r, sSpec->fft->data[(int)sart+ix].i, sqrt(POWERCU(sSpec->fft->data[(int)sart+ix].r, sSpec->fft->data[(int)sart+ix].i)) );
	    //printf("%.6f\t%.6f\n", sSpec.fftInf.fft[(int)sart+ix].r, sSpec.fftInf.fft[(int)sart+ix].i);
	    printf("\n");
	  }
	  printf("\n");

	  //printf("%.6f\t%.6f\t%.6f\t", (float)sart, sSpec.fftInf.fft[(int)sart].r, sSpec.fftInf.fft[(int)sart].i);
	  //printf("\t%.6f\t%.6f\n", sSpec.fftInf.fft[(int)sart].r, sSpec.fftInf.fft[(int)sart].i);


	  for ( float off = 0; off < nofbin; off += step )
	  {
	    printf("%.6f",sart + off);

	    rz_convolution_cu<float, float2>((float2*)sSpec->fft->data, sSpec->fft->firstBin, sSpec->fft->noBins, sart + off, z, hw, &real, &imag );
	    printf("\t%.6f\t%.6f\t%.6f\t", real, imag, sqrt(POWERCU(real, imag)) );

	    rz_convolution_cu<double, float2>((float2*)sSpec->fft->data, sSpec->fft->firstBin, sSpec->fft->noBins, sart + off, z, hw, &realD, &imagD );
	    printf("\t%.6f\t%.6f\t%.6f\t", realD, imagD, sqrt(POWERCU(realD, imagD)) );

	    //rz_convolution_cu<float, float2>((float2*)sSpec.fftInf.fft, sSpec.fftInf.idx, sSpec.fftInf.nor, sart + off, 0, hw, &real, &imag );
	    //printf("\t%.6f\t%.6f\t%.6f", real, imag, sqrt(POWERCU(real, imag)));

	    //rz_convolution_cu<double, float2>((float2*)sSpec.fftInf.fft, sSpec.fftInf.idx, sSpec.fftInf.nor, sart + off, 0, hw, &realD, &imagD );
	    //printf("\t%.6f\t%.6f\t%.6f\t", realD, imagD, sqrt(POWERCU(realD, imagD)) );

	    printf("\n");
	  }
	  //printf("%.6f\t%.6f\t%.6f\t", (float)(sart + nofbin), sSpec.fftInf.fft[(int)sart+nofbin].r, sSpec.fftInf.fft[(int)sart+nofbin].i);
	  //printf("\t%.6f\t%.6f\n", sSpec.fftInf.fft[(int)sart+nofbin].r, sSpec.fftInf.fft[(int)sart+nofbin].i);


//	  for ( z = 0.001; z >= -0.001; z -= 0.001 )
//	  {
//	    for ( float off = -sart; off <= sart; off+= step )
//	    {
//	      calc_response_off<float>(off, z, &real, &imag);
//	      printf("%.6f\t%.6f\t%.6f\t", off, real, imag);
//	    }
//	    printf("\n");
//	  }


	}

        FOLD // Generate the GPU kernel  .
        {
          cuSrch        = initCuKernels(&sSpec, &gSpec, NULL);
          master        =  &cuSrch->pInf->kernels[0];   // The first kernel created holds global variables

          if ( master->accelLen != ACCEL_USELEN )
          {
            fprintf(stderr, "ERROR: GPU and CPU step size do not match!\n");
            exit(EXIT_FAILURE);
          }

#if CORRECT_MULT
          {
            fprintf(stderr, "ERROR: Compiled with correct multiplication on.\n");
            //exit(EXIT_FAILURE);
          }
#endif
        }

        FOLD // Convert CPU to inmem .
        {
          if ( master->flags & FLAG_SS_INMEM  )
          {
            // Set the CPU search to be in-mem
            if ( !obs.inmem ) // Force to standard search  .
            {
              long long memuse;
              memuse = sizeof(float) * (obs.highestbin + ACCEL_USELEN) * obs.numbetween * obs.numz;
              printf("Converting to in-mem accelsearch.\n");

              obs.inmem = 1;
              obs.ffdotplane = gen_fvect(memuse / sizeof(float));
            }
          }
          else  // Set the CPU search to be in-mem
          {
            if ( obs.inmem ) // Force to standard search  .
            {
              obs.inmem = 0;
              vect_free(obs.ffdotplane);
              obs.ffdotplane = NULL;
            }
          }
        }

        FOLD // Generate CPU kernel  .
        {
          printf("Generating CPU correlation kernels:\n");
          subharminfs = create_subharminfos(&obs);
          printf("Done generating kernels.\n\n");
        }

        FOLD // Test kernels  .
        {
          int stage, harmtosum, harm;
          ffdotpows *subharmonic;

          printf("\nComparing GPU & CPU kernels    You can expect a RMSE ~2.00e-7 from the differences in FFTW & CUFFT.\n");

          int noStages = master->noHarmStages;

          if ( master->flags & FLAG_SS_INMEM )
            noStages = 1;

          for (stage = 0; stage < noStages; stage++)
          {
            harmtosum = 1 << stage;
            for ( harm = 1; harm <= harmtosum; harm += 2 )
            {
              nDarray<2, float> CPU_kernels;
              nDarray<2, float> GPU_kernels;
              //nDarray<2, float> CPU_diff;

              float frac  = (float)(harm)/(float)harmtosum;
              int idx     = noHarms - frac * noHarms;

              cuHarmInfo  *hinf   = &cuSrch->pInf->kernels[0].hInfos[idx];
              subharminfo *sinf0  = subharminfs[0];
              subharminfo *sinf1  = subharminfs[1];
              subharminfo *sinf   = &subharminfs[stage][harm - 1];

              CPU_kernels.addDim(hinf->width*2, 0, hinf->width);
              CPU_kernels.addDim(hinf->noZ, -hinf->zmax, hinf->zmax);
              CPU_kernels.allocate();

              GPU_kernels.addDim(hinf->width*2, 0, hinf->width);
              GPU_kernels.addDim(hinf->noZ, -hinf->zmax, hinf->zmax);
              GPU_kernels.allocate();

              // Copy data from device
              CUDA_SAFE_CALL(cudaMemcpy(GPU_kernels.elems, cuSrch->pInf->kernels[0].kernels[idx].d_kerData, GPU_kernels.getBuffSize(), cudaMemcpyDeviceToHost), "Failed to kernel copy data from.");
              //CUDA_SAFE_CALL(cudaDeviceSynchronize(), "At a blocking synchronisation. This is probably a error in one of the previous asynchronous CUDA calls.");

              for ( int row=0; row < sinf->numkern; row++  )
              {
                memcpy(CPU_kernels.getP(0,row), sinf->kern[row].data, hinf->width*sizeof(fcomplex));

		// Swap sighn for imag partys of GPU
                for ( int tx = 0 ; tx < hinf->width; tx++ )
                {
                  GPU_kernels.set(GPU_kernels.get(tx*2+1,row)*-1.0, tx*2+1,row);
                }
              }

              //printf("   Input: %02i (%.2f)    RMSE: %15.10f  μ: %10.5f   σ: %10.5f\n", harz, trdStack->hInfos[harz].harmFrac, RMSE, stat.mean, stat.sigma );

              basicStats statG = GPU_kernels.getStats(true);
              basicStats statC = CPU_kernels.getStats(true);

              //CPU_diff = CPU_kernels - GPU_kernels;
              //basicStats statD = CPU_diff.getStats(true);

              Fout // TMP
              {
        	for ( int row=0; row < sinf->numkern; row++  )
        	{
        	  for ( int tx = 0 ; tx < GPU_kernels.ax(0)->len()/2; tx++ )
        	  {

        	    float g_r = GPU_kernels.get(tx*2+0,row);
        	    float g_i = GPU_kernels.get(tx*2+1,row);

        	    float c_r = CPU_kernels.get(tx*2+0,row);
        	    float c_i = CPU_kernels.get(tx*2+1,row);

        	    float d1 = fabs(g_r - c_r);
        	    float d2 = fabs(g_i - c_i);

        	    if (  d1 > 0.01 || d2 > 0.01 )
        	    {
        	      slog.csvWrite("y","%i", row );
        	      slog.csvWrite("x","%i", tx );

        	      slog.csvWrite("real","%.7f", g_r);
		      slog.csvWrite("imag","%.7f", g_i);
		      slog.csvWrite("pow","%11.7f", g_r*g_r + g_i*g_i );

		      slog.csvWrite("real","%.7f", c_r);
		      slog.csvWrite("imag","%.7f", c_i);
		      slog.csvWrite("pow","%11.7f", c_r*c_r + c_i*c_i );

		      slog.csvEndLine();
        	    }
//
//        	    if ( g_r || g_i || c_r || c_i )
//        	    {
//        	      slog.csvWrite("n","%i", tx );
//
//        	      slog.csvWrite("real","%.7f", g_r);
//        	      slog.csvWrite("imag","%.7f", g_i);
//        	      slog.csvWrite("pow","%11.7f", g_r*g_r + g_i*g_i );
//
//        	      slog.csvWrite("real","%.7f", c_r);
//        	      slog.csvWrite("imag","%.7f", c_i);
//        	      slog.csvWrite("pow","%11.7f", c_r*c_r + c_i*c_i );
//
//        	      slog.csvEndLine();
//        	    }
        	  }
        	}
              }

              errorStats eStat = GPU_kernels.diff(CPU_kernels);
              double RMSE = eStat.RMSE;
              double Error =  RMSE / statG.sigma ;
              //printf("   Cmplx: %02i (%.2f)  RMSE: %10.3e    μ: %10.3e    σ: %10.3e    RMSE/σ: %9.2e  Max: %8.2e ", idx, frac, RMSE, statG.mean, statG.sigma, Error, MAX(fabs(statD.min),fabs(statD.max)) );
              //"(%.2f)   μ: %10.3e    σ: %10.3e   | Error  RMSE: %10.3e   Max: %10.3e  RMSE/σ: %9.2e ", idx, frac, stat.mean, stat.sigma, eStat.RMSE, eStat.largestError, Error );
              printf("   Cmplx: %02i (%.2f)   μ: %10.3e    σ: %10.3e   | Error  RMSE: %10.3e   Max: %10.3e  RMSE/σ: %9.2e ", idx, frac, statG.mean, statG.sigma, eStat.RMSE, eStat.largestError, Error );


              if      ( Error > buckets[0] )
                printf("  BAD!    Not even in the same realm.\n");
              else if ( Error > buckets[1] )
                printf("  Bad.  \n" );
              else if ( Error > buckets[2] )
                printf("  Bad.   But not that bad.\n");
              else if ( Error > buckets[3] )
                printf("  Close  But not great. \n");
              else if ( Error > buckets[4] )
                printf("  GOOD  But a bit high.\n");
              else if ( Error > buckets[5] )
                printf("  GOOD \n"  );
              else if ( Error > buckets[6] )
                printf("  GOOD  Very good.\n"  );
              else
                printf("  Great \n");

              FOLD // Print actual values  .
              {
        	if ( Error > buckets[2] )
        	{
        	  int y = master->hInfos[idx].noZ/2.0 - 1 ;

        	  //for ( y = master->hInfos[idx].noZ/2.0 - 2; y <= master->hInfos[idx].noZ/2.0 + 2; y++ )
        	  {
        	    printf("Harm: %02i y: %i \n", idx, y );
        	    printf("CPU: ");
        	    for ( int x = 0; x < 15; x++ )
        	    {
        	      printf(" %11.6f ", CPU_kernels.get(x,y));
        	    }
        	    printf("\n");

        	    printf("GPU: ");
        	    for ( int x = 0; x < 15; x++ )
        	    {
        	      printf(" %11.6f ", GPU_kernels.get(x,y));
        	    }
        	    printf("\n");
        	  }
        	}
              }
            }
          }
        }
      }

      if ( cmd->gpuP >= 0) 	                                  // -- Main Loop --  .
      {
        int  firstStep      = 0;
        bool printDetails   = false;		// Print out stats on all input and planes
        bool plotAllPlanes  = false;		// Plot all planes
        bool printAllValues = true;		// Print out a couple off all the values
        bool plot           = false;		// Draw bad planes
        bool CSV            = false;		//
        bool contPlotAll    = false;		//
        bool contPlotCnd    = false;		//
        cuFFdotBatch* batch = &cuSrch->pInf->batches[0];

        int   harmtosum, harm;
        startr = obs.rlo, lastr = 0, nextr = 0;

        FOLD // Search bounds  .
        {
          startr	= 0,
          lastr		= 0;
          nextr		= 0;
          maxxx		= cuSrch->sSpec->noSearchR / (double)batch->accelLen ;

          if ( maxxx < 0 )
            maxxx = 0;
        }

        FOLD // Output  .
        {
          printf("\nRunning GPU search of %i steps with %i simultaneous families of f-∂f planes spread across %i device(s).\n", maxxx, cuSrch->pInf->noSteps, cuSrch->pInf->noDevices );

          printf("\nWill check all input and planes and report ");
          if (printDetails)
            printf("all results");
          else
            printf("only poor or bad results");
          printf("\n\n");

          print_percent_complete(startr - obs.rlo, obs.highestbin - obs.rlo, "search", 1);
        }

        FOLD                                                  // -- Main Loop --  .
        {
          int           tid         = 0;

          double*       startrs     = (double*)malloc(sizeof(double)*batch->noSteps);
          double*       lastrs      = (double*)malloc(sizeof(double)*batch->noSteps);
          size_t        rest        = batch->noSteps;
          void*         tmpRow      = malloc(batch->inpDataSize);
          int           noCands     = 0;
          int           iteration   = 0;

          nDarray<1, float> **cpuInput  = new nDarray<1, float>*[batch->noSteps];
          nDarray<2, float> **cpuCmplx  = new nDarray<2, float>*[batch->noSteps];
          nDarray<1, float> **gpuInput  = new nDarray<1, float>*[batch->noSteps];
          nDarray<2, float> **gpuCmplx  = new nDarray<2, float>*[batch->noSteps];
          nDarray<2, float> **cpuPowers = new nDarray<2, float>*[batch->noSteps];
          nDarray<2, float> **gpuPowers = new nDarray<2, float>*[batch->noSteps];
          nDarray<2, float>   plotPowers;

          FOLD // Initialise data structures to hold test data for comparisons  .
          {
            for ( int step = 0; step < batch->noSteps ; step ++)
            {
              FOLD  // Create arrays of pointers
              {
                cpuInput[step]  = new nDarray<1, float>[batch->noGenHarms];
                cpuCmplx[step]  = new nDarray<2, float>[batch->noGenHarms];

                gpuInput[step]  = new nDarray<1, float>[batch->noGenHarms];
                gpuCmplx[step]  = new nDarray<2, float>[batch->noGenHarms];

                cpuPowers[step] = new nDarray<2, float>[batch->noGenHarms];
                gpuPowers[step] = new nDarray<2, float>[batch->noGenHarms];
              }

              FOLD  // Set up ND arrays    .
              {
                for (int harm = 0; harm < master->noGenHarms; harm++ )
                {
                  int sIdx              = cuSrch->sIdx[harm];
                  cuHarmInfo *hinf      = &cuSrch->pInf->kernels[0].hInfos[sIdx];

                  cpuInput[step][sIdx].addDim(hinf->width*2, 0, hinf->width);
                  cpuInput[step][sIdx].allocate();

                  gpuInput[step][sIdx].addDim(hinf->width*2, 0, hinf->width);
                  gpuInput[step][sIdx].allocate();


                  cpuCmplx[step][sIdx].addDim(hinf->width*2, 0, hinf->width);
                  cpuCmplx[step][sIdx].addDim(hinf->noZ, -hinf->zmax, hinf->zmax);
                  cpuCmplx[step][sIdx].allocate();


                  gpuCmplx[step][sIdx].addDim(hinf->width*2, 0, hinf->width);
                  gpuCmplx[step][sIdx].addDim(hinf->noZ, -hinf->zmax, hinf->zmax);
                  gpuCmplx[step][sIdx].allocate();

                  cpuPowers[step][sIdx].addDim(hinf->width, 0, hinf->width);
                  cpuPowers[step][sIdx].addDim(hinf->noZ, -hinf->zmax, hinf->zmax);
                  cpuPowers[step][sIdx].allocate();

                  gpuPowers[step][sIdx].addDim(hinf->width, 0, hinf->width);
                  gpuPowers[step][sIdx].addDim(hinf->noZ, -hinf->zmax, hinf->zmax);
                  gpuPowers[step][sIdx].allocate();
                }
              }
            }

            FOLD // initalise plotPowers  .
            {
              plotPowers.addDim(master->accelLen,  0, master->accelLen  );
              plotPowers.addDim(master->hInfos[0].noZ, 0, master->hInfos[0].noZ );
              plotPowers.allocate();
            }
          }

          if ( batch->flags & FLAG_POW_HALF ) // Set border  .
          {
            buckets[1] = 1e-1 ;   // BAD!    Not even in the same realm.
            buckets[1] = 6e-2 ;   // Bad.
            buckets[2] = 3e-2 ;   // Bad.   But not that bad.
            buckets[3] = 1e-2 ;   // Close But not great.
            buckets[4] = 1e-3 ;   // GOOD  But a bit high.
            buckets[5] = 1e-4 ;   // GOOD
            buckets[6] = 1e-5 ;   // GOOD  Very good
          }

          while ( ss < maxxx ) 	                              // -- Main Loop --  .
          {
            FOLD // Calculate the step(s) to handle  .
            {

//#pragma omp critical
              FOLD // Calculate the step  .
              {
                FOLD  // Synchronous behaviour  .
                {
                  // If running in synchronous mode use multiple batches, just synchronously
                  tid     = iteration % cuSrch->pInf->noBatches ;
                  batch   = &cuSrch->pInf->batches[tid];
                  setDevice(batch->gInf->devid) ;
                }

                iteration++;

                firstStep = ss;
                ss       += batch->noSteps;

                infoMSG(1,1,"Step %4i of %4i thread %02i processing %02i steps\n", firstStep+1, maxxx, tid, batch->noSteps);
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
              //trdBatch->rValues = trdBatch->rArrays[0];

              for ( int step = 0; step < (int)batch->noSteps ; step ++)
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
              search_ffdot_batch_CU(batch);

              if      ( master->flags & FLAG_SS_INMEM     )
              {
                int len = batch->accelLen * batch->noSteps ;
                if ( len != batch->strideOut )
                {
                  fprintf(stderr,"ERROR: SS_INMEM_SZ is wrong set it to %i\n", len );
                  exit(EXIT_FAILURE);
                }
                len = batch->strideOut;

                batch->rAraays  = &batch->rArraysSrch;

                inmemSS(batch, startrs[0], len);

                batch->rAraays  = &batch->rArraysPlane;
              }

            }

            FOLD // Copy data from device  .
            {
              infoMSG(3,3,"Copy Data drom device\n");

              ulong sz  = 0;
              harm      = 0;

              // Write data to page locked memory
              for ( int stackNo = 0; stackNo < batch->noStacks; stackNo++ )
              {
                cuFfdotStack* cStack = &batch->stacks[stackNo];

                // Synchronise
                //cudaStreamWaitEvent(cStack->fftPStream, cStack->plnComp, 0);
                for ( int plainNo = 0; plainNo < cStack->noInStack; plainNo++ )
                {
                  cuHarmInfo* cHInfo    = &batch->hInfos[harm];          // The current harmonic we are working on
                  cuFFdot*    plan      = &batch->planes[harm];          // The current plane

                  for ( int step = 0; step < batch->noSteps; step ++)    // Loop over steps
                  {
                    rVals* rVal = &(((*batch->rAraays)[batch->rActive])[step][harm]);

                    if ( rVal->numdata )
                    {
                      // Copy input data from GPU - This is post FFT data so
                      fcomplexcu *data = &batch->d_iData[sz];
                      CUDA_SAFE_CALL(cudaMemcpy(gpuInput[step][harm].elems, data, cStack->strideCmplx*sizeof(fcomplexcu), cudaMemcpyDeviceToHost), "Failed to copy input data from device.");

                      // Copy pain from GPU
                      for( int y = 0; y < cHInfo->noZ; y++ )
                      {
                        fcomplexcu *cmplxData;
                        void *powers;
                        int offset;
                        int elsz;

                        if      ( batch->flags & FLAG_ITLV_ROW )
                        {
                          //offset = (y*trdBatch->noSteps + step)*cStack->strideCmplx   + cHInfo->halfWidth * 2 ;
                          offset = (y*batch->noSteps + step)*cStack->strideCmplx   + cHInfo->kerStart ;
                        }
                        else
                        {
                          //offset  = (y + step*cHInfo->height)*cStack->strideCmplx   + cHInfo->halfWidth * 2 ;
                          offset  = (y + step*cHInfo->noZ)*cStack->strideCmplx   + cHInfo->kerStart ;
                        }

                        if ( !(batch->flags & FLAG_CUFFT_CB_OUT) )
                        {
                          cmplxData = &((fcomplexcu*)plan->d_planeMult)[offset];
                          CUDA_SAFE_CALL(cudaMemcpy(gpuCmplx[step][harm].getP(0,y), cmplxData, (rVal->numrs)*sizeof(fcomplexcu), cudaMemcpyDeviceToHost), "Failed to copy input data from device.");
                        }

                        float* outVals = gpuPowers[step][harm].getP(0,y);

                        if      ( batch->flags & FLAG_POW_HALF )
                        {
#if CUDART_VERSION >= 7050   // Half precision getter and setter  .
                          powers =  &((half*)      plan->d_planePowr)[offset];
                          elsz   = sizeof(half);
                          CUDA_SAFE_CALL(cudaMemcpy(tmpRow, powers, (rVal->numrs)*elsz,   cudaMemcpyDeviceToHost), "Failed to copy input data from device.");

                          for ( int i = 0; i < rVal->numrs; i++)
                          {
                            outVals[i] = half2float(((ushort*)tmpRow)[i]);
                          }
#else
                          fprintf(stderr, "ERROR: Half precision can only be used with CUDA 7.5 or later! Reverting to single precision!\n");
                          exit(EXIT_FAILURE);
#endif
                        }
                        else if ( batch->flags & FLAG_CUFFT_CB_POW )
                        {
                          powers =  &((float*)     plan->d_planePowr)[offset];
                          elsz   = sizeof(float);
                          CUDA_SAFE_CALL(cudaMemcpy(outVals, powers, (rVal->numrs)*elsz,   cudaMemcpyDeviceToHost), "Failed to copy input data from device.");
                        }
                        else
                        {
                          powers =  &((fcomplexcu*) plan->d_planePowr)[offset];
                          elsz   = sizeof(cmplxData);
                          CUDA_SAFE_CALL(cudaMemcpy(tmpRow, powers, (rVal->numrs)*elsz,   cudaMemcpyDeviceToHost), "Failed to copy input data from device.");

                          for ( int i = 0; i < rVal->numrs; i++)
                          {
                            outVals[i] = POWERC(((fcomplexcu*)tmpRow)[i]);
                          }
                        }
                      }
                    }

                    sz += cStack->strideCmplx;
                  }
                  harm++;
                }

                // New events for Synchronisation (this event will override the previous event)
                cudaEventRecord(cStack->inpFFTinitComp, cStack->fftIStream);
                cudaEventRecord(cStack->ifftComp,  cStack->fftPStream);
              }
            }

            FOLD // Now do an equivalent CPU search  .
            {
              infoMSG(3,3,"CPU Search\n");

              for ( int step = 0; (step < batch->noSteps) && ( firstStep+step < maxxx) ; step ++) // Loop over steps
              {
        	infoMSG(4,4,"Srtep %i\n", step);

                double poww, sig, rr, zz;

                rVals* rVal     = &(((*batch->rAraays)[batch->rActive])[step][0]);

                //double startr   = startrs[step];
                double lastr    	= lastrs[step];
                double CPU_startr	= (*batch->rAraays)[batch->rActive][step][0].drlo;
                double CPU_lastr	= (*batch->rAraays)[batch->rActive][step][0].drhi;
                double rLow		= (*batch->rAraays)[batch->rActive][0][0].drlo;

                FOLD // Do CPU search to compare candidates  .
                {
                  int stage, harmtosum, harm;
                  ffdotpows *subharmonic;

                  // Copy the fundamental's ffdot plane to the full in-core one
                  //for (stage = 0; stage < master->noHarmStages; stage++)
                  for (stage = 0; stage < obs.numharmstages; stage++)
                  {
                    infoMSG(4,4,"stage %i\n", stage);

                    harmtosum   = 1 << stage;

                    if ( stage == 0 )
                    {
                      rVal        = &(((*batch->rAraays)[batch->rActive])[step][0]);

                      fundamental = subharm_ffdot_plane_DBG(1, 1, CPU_startr, CPU_lastr, &subharminfs[0][0], &obs, rVal->norm, &cpuInput[step][0], &cpuCmplx[step][0], &cpuPowers[step][0] );

                      if ( obs.inmem )
                      {
                        fund_to_ffdotplane(fundamental, &obs);
                      }
                    }
                    else
                    {
                      for (harm = 1; harm < harmtosum; harm += 2)
                      {
                        float frac  = (float)(harm)/(float)harmtosum;
                        int idx     = noHarms - frac * noHarms;
                        rVal        = &(((*batch->rAraays)[batch->rActive])[step][idx]);

                        if ( obs.inmem ) //
                        {
                          inmem_add_ffdotpows(fundamental, &obs, harmtosum, harm);
                        }
                        else
                        {
                          subharmonic = subharm_ffdot_plane_DBG(harmtosum, harm, startr, lastr, &subharminfs[stage][harm - 1], &obs, rVal->norm, &cpuInput[step][idx], &cpuCmplx[step][idx], &cpuPowers[step][idx] );

                          add_ffdotpows(fundamental, subharmonic, harmtosum, harm);

                          free_ffdotpows(subharmonic);
                        }
                      }
                      //printf("\n");
                    }

                    candsCPU = search_ffdotpows(fundamental, harmtosum, &obs, candsCPU);

                    int numcands = g_slist_length(candsCPU);

                    Fout // Tempt output  .
                    {
                      if ( stage == 0 && step == 0 )
                      {
                        float max = 0;
                        float vall;

                        printf("\n============================ CPU ============================ \n");

                        for ( int y = 0; y < batch->hInfos->noZ; y++ )
                        {
                          vall = fundamental->powers[y][1022];

                          if ( vall > max )
                          {
                            printf("%03i %15.3f  %04i %04i New best! \n", y, vall, 0, 0 );
                            max = vall;
                          }
                          else
                          {
                            printf("%03i %15.3f  %04i %04i \n", y, vall, 0, 0 );
                          }
                        }
                      }
                    }

                    FOLD // Compare candidates  .
                    {
                      infoMSG(5,5,"Compare candidate\n", stage);

                      void *gpuOutput;

                      rVal              = &(((*batch->rAraays)[batch->rActive])[0][0]);
                      gpuOutput 	= rVal->h_outData; // TODO: this needs to be checked

                      FOLD
                      {
                        rVal              = &(((*batch->rAraays)[batch->rActive])[step][0]);
                        uint  x0          = step*rVal->numrs;
                        uint  x1          = x0 + rVal->numrs;
                        uint  y0          = 0;
                        uint  y1          = batch->ssSlices;
                        uint  xStride     = batch->strideOut;
                        uint  yStride     = batch->ssSlices;

                        if ( !(batch->flags & FLAG_SS_INMEM) )
                        {
                          //xStride         = rVal->numrs;
                          xStride         = batch->strideOut * batch->noSteps;
                        }

                        for ( int y = y0; y < y1; y++ )
                        {
                          //int x0 = step*rVal->numrs;
                          //int x1 = (step+1)*rVal->numrs;

                          for ( int x = x0; x < x1; x++ )
                          {
                            //int idx   = stage*trdBatch->strideRes*trdBatch->noSteps*trdBatch->ssSlices + y*trdBatch->strideRes*trdBatch->noSteps + x ;
                            int idx   = stage*xStride*yStride + y*xStride + x ;

                            int iz    = 0;
                            poww      = 0;
                            sig       = 0;
                            zz        = 0;

                            if      ( batch->retType & CU_CANDMIN  )
                            {
                              candMin candM         = ((candMin*)gpuOutput)[idx];
                              sig                   = candM.power;
                              poww                  = candM.power;
                              iz                    = candM.z;
                            }
                            else if ( batch->retType & CU_POWERZ_S )
                            {
                              candPZs candM         = ((candPZs*)gpuOutput)[idx];
                              sig                   = candM.value;
                              poww                  = candM.value;
                              iz                    = candM.z;
                            }
                            else if ( batch->retType & CU_CANDBASC )
                            {
                              accelcandBasic candB  = ((accelcandBasic*)gpuOutput)[idx];
                              poww                  = candB.sigma;
                              sig                   = candB.sigma;
                              iz                    = candB.z;
                            }
                            else
                            {
                              fprintf(stderr,"ERROR: function %s requires accelcandBasic\n",__FUNCTION__);
                              exit(EXIT_FAILURE);
                            }

                            if ( poww > 0 )
                            {
                              if ( isnan(poww) || isinf(poww) )
                              {
                                fprintf(stderr, "CUDA search returned an NAN power.\n");
                              }
                              else
                              {
                                rr      = ( rLow + x *  ACCEL_DR )                  / (double)harmtosum ;
                                //zz      = ( iz * ACCEL_DZ - batch->hInfos[0].noZ ) / (double)harmtosum ;
                                zz	= (batch->hInfos[0].zStart + (batch->hInfos[0].zEnd - batch->hInfos[0].zStart ) * iz / (double)(batch->hInfos[0].noZ-1) ) ;
                                zz	/= (double)harmtosum ;
                                if ( batch->hInfos[0].noZ == 1 )
                                  zz = 0;

                                float cPow;
                                float *row;

                                row   = fundamental->powers[iz];
                                cPow  = row[x-x0];

                                float p1 = poww;
                                float p2 = cPow;

                                float err = fabs(1-p2/p1);

                                if ( err > 0.01 )
                                {
                                  printf("Candidate r: %9.4f z: %7.2f   CPU pow: %7.3f   GPU pow: %7.3f   Err: %7.5f   Harm: %i\n", rr, zz, p2, p1, err, harmtosum );
                                  badCands++;

                                  FOLD  // Manually calculate the power  .
                                  {
                                    double powC = 0;
                                    double powG = 0;
                                    double real, imag;
                                    fcomplex ans;

                                    for ( int ih = 1; ih <= harmtosum; ih++ )
                                    {
                                      double frac = ih / (double)harmtosum;
                                      int harmNN = batch->noSrchHarms - frac*batch->noSrchHarms;

                                      double c_r = rr * ih;
                                      double c_z = zz * ih;
                                      int c_hw = cu_z_resp_halfwidth_low(c_z);

                                      rVals* c_rVal     = &(((*batch->rAraays)[batch->rActive])[step][harmNN]);

                                      // GPU search
                                      rz_convolution_cu<double, float2>((float2*)cuSrch->data.data, cuSrch->data.firstBin, cuSrch->data.noBins, c_r, c_z, c_hw, &real, &imag);
                                      powG += real*real*c_rVal->norm*c_rVal->norm + imag*imag*c_rVal->norm*c_rVal->norm;

                                      // CPU search
                                      rz_interp(&cuSrch->data.data[-cuSrch->data.firstBin], cuSrch->data.noBins, c_r, c_z, c_hw, &ans);
                                      powC += ans.r*ans.r*c_rVal->norm*c_rVal->norm + ans.i*ans.i*c_rVal->norm*c_rVal->norm;
                                    }

                                    printf("Candidate r: %9.4f z: %7.2f  CPU pow: %6.5f %6.5f  GPU pow: %6.5f %6.5f   Err: %8.6f   Err: %8.6f  Harm: %i\n", rr, zz, p2, powC, p1, powG, fabs(1-p2/p1), fabs(1-powG/p1), harmtosum );
                                  }

//                                  if ( fabs(1-powG/p1) > 0.001 )
//                                  {
//                                    int noZ = 20;
//                                    int noR = 20;
//                                    double zSp = 2;
//                                    double rSp = 1;
//
//                                    for ( int iy = 0; iy <= noZ; iy++ )
//                                    {
//                                      double gz = zz - zSp/2.0 + iy/(double)noZ*zSp;
//
//                        	      slog.csvWrite("z","%.6f", gz );
//
//                        	      for ( int ir = 0; ir <= noR; ir++ )
//                        	      {
//                        		double gr = rr - rSp/2.0 + ir/(double)noR*rSp;
//                        		powG = 0;
//
//                        		for ( int ih = 1; ih <= harmtosum; ih++ )
//                        		{
//                        		  double frac = ih / (double)harmtosum;
//                        		  int harmNN = batch->noSrchHarms - frac*batch->noSrchHarms;
//
//                        		  double c_r = gr * ih;
//                        		  double c_z = gz * ih;
//                        		  int c_hw = cu_z_resp_halfwidth_low(c_z);
//
//                        		  //int harmNN = stageOrder[ih-1];
//
//                        		  rVals* c_rVal     = &(((*batch->rAraays)[batch->rActive])[step][harmNN]);
//
//                        		  rz_convolution_cu<double, float2>((float2*)cuSrch->sSpec->fftInf.fft, cuSrch->sSpec->fftInf.firstBin, cuSrch->sSpec->fftInf.noBins, c_r, c_z, c_hw, &real, &imag);
//                        		  powG += real*real*c_rVal->norm*c_rVal->norm + imag*imag*c_rVal->norm*c_rVal->norm;
//                        		}
//
//                        		char tmp[1024];
//                        		sprintf(tmp, "%.6f",gr);
//                        		slog.csvWrite(tmp,"%.6f", powG );
//
//                        	      }
//
//                        	      slog.csvEndLine();
//                                    }
//
//                                    int tmp = 0;
//                                  }
                                }
                                else if ( err > 0.001 )
                                {
                                  similarCands++;
                                }
                                else
                                {
                                  sameCands++;
                                }
                              }
                            }

                          }
                        }
                      }
                    }

                    if ( CSV || ( noCands != numcands && contPlotCnd) ) // Write CVS  .
                    {
                      infoMSG(5,5,"Write CVS\n", stage);

                      double rr, zz;
                      char tName[1024];
                      sprintf(tName,"/home/chris/accel/ffplane_h%02i_%015.4f-%015.4f.csv", harmtosum, rVal->drlo / (double)harmtosum, (rVal->drlo + (batch->accelLen-1)*ACCEL_DR) / (double)harmtosum );
                      FILE *f2 = fopen(tName, "w");

                      fprintf(f2,"%i", harmtosum);

                      FOLD // Print the bin values as column headers
                      {
                        for ( int x = 0; x < batch->accelLen; x++ )
                        {
                          rr      = ( rVal->drlo + x *  ACCEL_DR )            / (double)harmtosum ;
                          fprintf(f2,"\t%.6f",rr);
                        }
                        fprintf(f2,"\n");
                      }

                      for ( int y = 0; y < batch->hInfos->noZ; y++ )
                      {
                        // First column is the r value
                        zz      = ( y * ACCEL_DZ - batch->hInfos[0].zmax )  / (double)harmtosum ;
                        fprintf(f2,"%.6f",zz);

                        // print the powers
                        for ( int x = 0; x < batch->accelLen; x++ )
                        {
                          float yy2 = fundamental->powers[y][x];
                          fprintf(f2,"\t%.6f",yy2);
                        }

                        fprintf(f2,"\n");
                      }
                      fclose(f2);

                      if ( contPlotAll || ( noCands != numcands && contPlotCnd) )
                      {
                        char cmd[1024];
                        sprintf(cmd,"python ~/bin/bin/plt_ffd.py %s", tName);
                        system(cmd);
                      }
                    }

                    noCands = numcands;
                  }
                }

                infoMSG(5,5,"free_ffdotpows\n");

                free_ffdotpows(fundamental);
              }
            }

            FOLD // Print RMSE  .
            {
              infoMSG(3,3,"Print RMSE\n");

              for ( int step = 0; (step < batch->noSteps) && ( firstStep+step < maxxx) ; step ++) // Loop over steps
              {
                bool good;
                bool bad;

                FOLD // Powers  .
                {
                  good = true;
                  bad  = false;

                  if ( printDetails )
                    printf("\n           ---- Step %03i of %03i ----\n",firstStep + step+1, maxxx);

                  for ( int harz = 0; harz < batch->noGenHarms; harz++ )
                  {
                    basicStats stat       = gpuPowers[step][harz].getStats(true);
                    errorStats eStat 	  = gpuPowers[step][harz].diff(cpuPowers[step][harz]);
                    double RMSE           = eStat.RMSE;
                    double ERR            = RMSE / stat.sigma ;
                    rVals* rVal           = &(((*batch->rAraays)[batch->rActive])[step][harz]);
                    cuHarmInfo* cHInfo    = &batch->hInfos[harz];          // The current harmonic we are working on

                    if ( ERR > buckets[4] )
                    {
                      if ( good && !printDetails )
                        printf("\n           ---- Step %03i of %03i ----\n",firstStep + step+1, maxxx);

                      good = false;
                    }
                    if ( ERR > buckets[3]  )
                    {
                      badCplx++;
                      bad   = true;
                    }

                    if ( !good || printDetails )
                    {
                      printf("  Powers: %02i (%.2f)   μ: %10.3e    σ: %10.3e   | Error  RMSE: %10.3e   Max: %10.3e  RMSE/σ: %9.2e ", harz, batch->hInfos[harz].harmFrac, stat.mean, stat.sigma, eStat.RMSE, eStat.largestError, ERR );

                      if      ( ERR > buckets[0]   )
                        printf("  BAD!    Not even in the same realm.\n");
                      else if ( ERR > buckets[1]   )
                        printf("  Bad.  \n" );
                      else if ( ERR > buckets[2]  )
                        printf("  Bad.   But not that bad.\n");
                      else if ( ERR > buckets[3]  )
                        printf("  Close  But not great. \n");
                      else if ( ERR > buckets[4] )
                        printf("  GOOD  But a bit high.\n");
                      else if ( ERR > buckets[5] )
                        printf("  GOOD \n"  );
                      else if ( ERR > buckets[6] )
                        printf("  GOOD  Very good.\n"  );
                      else
                        printf("  Great \n");

                      if ( ( !good && plot ) || plotAllPlanes )
                      {
                        float *powArr     = (float*)plotPowers.getP(0,0);

                        int nX = rVal->numrs;
                        int nY = cHInfo->noZ;

                        // Copy CPU powers
                        for(int y = 0; y < nY; y++ )
                        {
                          memcpy(&powArr[y*nX],gpuPowers[step][harz].getP(0,y), nX*sizeof(float));
                        }
                        sprintf(fname, "/home/chris/fdotplanes/ffdot_S%05i_H%02i_GPU.png", firstStep+step+1, harz+1);
#ifndef DEBUG
                        printf("\r  Plotting %s  ", fname);
                        fflush(stdout);
#endif
                        draw2DArray6(fname, powArr, nX, nY, MAX(800,nX), MAX(800,nY*3) );
#ifndef DEBUG
                        printf("\r                                                                                                     \r");
                        fflush(stdout);
#endif

                        // Copy CPU powers
                        for(int y = 0; y < nY; y++ )
                        {
                          memcpy(&powArr[y*nX],cpuPowers[step][harz].getP(0,y), nX*sizeof(float));
                        }
                        sprintf(fname, "/home/chris/fdotplanes/ffdot_S%05i_H%02i_CPU.png", firstStep+step+1, harz+1);
#ifndef DEBUG
                        printf("\r  Plotting %s  ", fname);
                        fflush(stdout);
#endif
                        draw2DArray6(fname, powArr, nX, nY, MAX(800,nX), MAX(800,nY*3) );
#ifndef DEBUG
                        printf("\r                                                                                                     \r");
                        fflush(stdout);
#endif
                      }
                    }
                  }

                  if ( bad || printAllValues )
                  {
                    for ( int harz = 0; harz < batch->noGenHarms; harz++ )
                    {
                      printf("Harm: %02i\n", harz );
                      printf("CPU: ");
                      for ( int x = 0; x < 10; x++ )
                      {
                        printf(" %11.6f ", cpuPowers[step][harz].get(x,0));
                      }
                      printf("\n");

                      printf("GPU: ");
                      for ( int x = 0; x < 10; x++ )
                      {
                        printf(" %11.6f ", gpuPowers[step][harz].get(x,0));
                      }
                      printf("\n");
                    }
                  }
                }

                if ( !good ) // Only test the rest if there was something wrong with the powers  .
                {
                  FOLD // Input  .
                  {
                    good = true;
                    bad  = false;

                    if (printDetails)
                      printf("\n           ---- Step %03i of %03i ----\n",firstStep + step+1, maxxx);

                    for ( int harz = 0; harz < batch->noGenHarms; harz++ )
                    {
                      basicStats stat = gpuInput[step][harz].getStats(true);
                      errorStats eStat = gpuInput[step][harz].diff(cpuInput[step][harz]);

                      double RMSE = gpuInput[step][harz].RMSE(cpuInput[step][harz]);
                      double ERR  = eStat.RMSE / stat.sigma ;

                      if ( ERR > buckets[4]  )
                      {
                	if ( good && !printDetails )
                	  printf("\n           ---- Step %03i of %03i ----\n", firstStep + step+1, maxxx);

                	good = false;
                      }
                      if ( ERR > buckets[3]   )
                      {
                	badInp++;
                	bad = true;
                      }

                      if ( !good || printDetails )
                      {
                	printf("   Input: %02i (%.2f)   μ: %10.3e    σ: %10.3e   | Error  RMSE: %10.3e   Max: %10.3e  RMSE/σ: %9.2e ", harz, batch->hInfos[harz].harmFrac, stat.mean, stat.sigma, eStat.RMSE, eStat.largestError, ERR );

                	if      ( ERR > buckets[0] )
                	  printf("  BAD!    Not even in the same realm.\n");
                	else if ( ERR > buckets[1] )
                	  printf("  Bad.  \n" );
                	else if ( ERR > buckets[2] )
                	  printf("  Bad.   But not that bad.\n");
                	else if ( ERR > buckets[3] )
                	  printf("  Close  But not great. \n");
                	else if ( ERR > buckets[4] )
                	  printf("  GOOD  But a bit high.\n");
                	else if ( ERR > buckets[5] )
                	  printf("  GOOD \n"  );
                	else if ( ERR > buckets[6] )
                	  printf("  GOOD  Very good.\n"  );
                	else
                	  printf("  Great \n");
                      }

                    }

                    if ( bad || printAllValues )
                    {
                      for ( int harz = 0; harz < batch->noGenHarms; harz++ )
                      {
                	int y = batch->hInfos[harz].noZ - 1;

                	printf("Harm: %02i\n", harz );
                	printf("CPU: ");
                	for ( int x = 0; x < 15; x++ )
                	{
                	  printf(" %11.6f ", cpuInput[step][harz].get(x,y));
                	}
                	printf("\n");

                	printf("GPU: ");
                	for ( int x = 0; x < 15; x++ )
                	{
                	  printf(" %11.6f ", gpuInput[step][harz].get(x,y));
                	}
                	printf("\n");
                      }
                    }
                  }

                  FOLD // Complex values  .
                  {
                    good = true;
                    bad  = false;

                    if ( !(batch->flags & FLAG_CUFFT_CB_OUT) )
                    {
                      if( printDetails )
                	printf("\n           ---- Step %03i of %03i ----\n",firstStep + step+1, maxxx);

                      for ( int harz = 0; harz < batch->noGenHarms; harz++ )
                      {
                	basicStats stat		= gpuCmplx[step][harz].getStats(true);
                        errorStats eStat	= gpuCmplx[step][harz].diff(cpuCmplx[step][harz]);
                        double RMSE		= eStat.RMSE;
                	double ERR		= RMSE / stat.sigma ;
                	rVals* rVal		= &(((*batch->rAraays)[batch->rActive])[step][harz]);
                	cuHarmInfo* cHInfo	= &batch->hInfos[harz];          // The current harmonic we are working on

                	if ( ERR > buckets[4]  )
                	{
                	  if ( good && !printDetails )
                	    printf("\n           ---- Step %03i of %03i ----\n",firstStep + step+1, maxxx);

                	  good = false;
                	}
                	if ( ERR > buckets[3]   )
                	{
                	  bad = true;
                	  badCplx++;
                	}

                	if ( !good || printDetails )
                	{
                	  printf("   Cmplx: %02i (%.2f)   μ: %10.3e    σ: %10.3e   | Error  RMSE: %10.3e   Max: %10.3e  RMSE/σ: %9.2e ", harz, batch->hInfos[harz].harmFrac, stat.mean, stat.sigma, eStat.RMSE, eStat.largestError, ERR );

                	  if      ( ERR > buckets[0]   )
                	    printf("  BAD!    Not even in the same realm.\n");
                	  else if ( ERR > buckets[1]   )
                	    printf("  Bad.  \n" );
                	  else if ( ERR > buckets[2]  )
                	    printf("  Bad.   But not that bad.\n");
                	  else if ( ERR > buckets[3]  )
                	    printf("  Close  But not great. \n");
                	  else if ( ERR > buckets[4] )
                	    printf("  GOOD  But a bit high.\n");
                	  else if ( ERR > buckets[5] )
                	    printf("  GOOD \n"  );
                	  else if ( ERR > buckets[6] )
                	    printf("  GOOD  Very good.\n"  );
                	  else
                	    printf("  Great \n");

                	}

                	if ( ( !good && plot ) || plotAllPlanes )
                	{
                	  //fcomplex* cmplx   = (fcomplex*)gpuCmplx[step][harz].getP(0,0);
                	  float *powArr     = (float*)plotPowers.getP(0,0);

                	  int nX = rVal->numrs;     // gpuPowers[step][harz].ax(0)->noEls() ;
                	  int nY = cHInfo->noZ;  // gpuPowers[step][harz].ax(1)->noEls() ;

                	  //int width         = trdBatch->accelLen;
                	  //int width         = rVal->numrs;


                	  // Calculate GPU powers
                	  for(int y = 0; y < nY; y++ )
                	  {
                	    fcomplex* cmplx   = (fcomplex*)gpuCmplx[step][harz].getP(0,y);
                	    //float *powArr     = (float*)plotPowers.getP(0,y);

                	    for(int x = 0; x < nX; x++ )
                	    {
                	      powArr[y*nX + x] = cmplx[x].i*cmplx[x].i + cmplx[x].r*cmplx[x].r ;
                	      //
                	      //powArr[y*width + x] = cmplx[y*nX + x].i*cmplx[y*nX + x].i + cmplx[y*nX + x].r*cmplx[y*nX + x].r ;
                	    }
                	  }

                	  //sprintf(fname, "/home/chris/fdotplanes/ffdot_S%05i_H%2i_01_GPU.png", firstStep+si, harz);
                	  //drawArr(fname, &gpuPowers[si][harz], HM_G);

                	  sprintf(fname, "/home/chris/fdotplanes/ffdot_S%05i_H%02i_GPU.png", firstStep+step+1, harz+1);
#ifndef DEBUG
                	  printf("\r  Plotting %s  ", fname);
                	  fflush(stdout);
#endif
                	  draw2DArray6(fname, powArr, nX, nY, MAX(800,nX), MAX(800,nY*3) );
#ifndef DEBUG
                	  printf("\r                                                                                                     \r");
                	  fflush(stdout);
#endif

                	  // Copy CPU powers
                	  for(int y = 0; y < nY; y++ )
                	  {
                	    memcpy(&powArr[y*nX],cpuPowers[step][harz].getP(0,y), nX*sizeof(float));
                	  }
                	  sprintf(fname, "/home/chris/fdotplanes/ffdot_S%05i_H%02i_CPU.png", firstStep+step+1, harz+1);
#ifndef DEBUG
                	  printf("\r  Plotting %s  ", fname);
                	  fflush(stdout);
#endif
                	  draw2DArray6(fname, powArr, nX, nY, MAX(800,nX), MAX(800,nY*3) );
#ifndef DEBUG
                	  printf("\r                                                                                                     \r");
                	  fflush(stdout);
#endif


                	  // Copy CPU powers
                	  for(int y = 0; y < nY; y++ )
                	  {
                	    fcomplex* cmplx   = (fcomplex*)gpuCmplx[step][harz].getP(0,y);
                	    float *powArr     = (float*)plotPowers.getP(0,y);

                	    for(int x = 0; x < nX; x++ )
                	    {
                	      powArr[x] -= cmplx[x].i*cmplx[x].i + cmplx[x].r*cmplx[x].r ;
                	    }
                	  }
                	  sprintf(fname, "/home/chris/fdotplanes/ffdot_S%05i_H%02i_RES.png", firstStep+step+1, harz);
#ifndef DEBUG
                	  printf("\r  Plotting %s  ", fname);
                	  fflush(stdout);
#endif
                	  draw2DArray6(fname, powArr, nX, nY, MAX(800,nX), MAX(800,nY*3) );
#ifndef DEBUG
                	  printf("\r                                                                                                     \r");
                	  fflush(stdout);
#endif

                	  //fundamental = subharm_ffdot_plane(1, 1, startr, lastr, &subharminfs[0][0], &obs);
                	  //draw2DArray6(fname, fundamental->powers[0], fundamental->numrs, fundamental->numzs, 4096, 1602);
                	  //cands = search_ffdotpows(fundamental, 1, &obs, cands);
                	}
                      }

                      if ( bad || printAllValues )
                      {
                	for ( int harz = 0; harz < batch->noGenHarms; harz++ )
                	{
                	  printf("Harm: %02i\n", harz );
                	  printf("CPU: ");
                	  for ( int x = 0; x < 15; x++ )
                	  {
                	    printf(" %9.6f ", cpuCmplx[step][harz].get(x,1));
                	  }
                	  printf("\n");

                	  printf("GPU: ");
                	  for ( int x = 0; x < 15; x++ )
                	  {
                	    printf(" %9.6f ", gpuCmplx[step][harz].get(x,1));
                	  }
                	  printf("\n");
                	}
                      }
                    }
                  }
                }

                int tmp = 0;
              }
            }

            print_percent_complete(startrs[0] - obs.rlo, obs.highestbin - obs.rlo, "search", 0);
          }

          FOLD // Finish off CUDA search  .
          {
            // Set r values to 0 so as to not process details
            for ( int step = 0; step < batch->noSteps ; step ++)
            {
              startrs[step] = 0;
              lastrs[step]  = 0;
            }

            // Finish searching the planes, this is required because of the out of order asynchronous calls
            for ( int step = 0 ; step < 2; step++ )
            {
              search_ffdot_batch_CU( batch );
            }
          }

          FOLD // Wait for CPU threads to complete  .
          {
            waitForThreads(&master->cuSrch->threasdInfo->running_threads, "Waiting for CPU thread(s) to finish processing returned from the GPU,", 200 );
          }

          free(tmpRow);
        }

        print_percent_complete(obs.highestbin - obs.rlo, obs.highestbin - obs.rlo, "search", 0);

        printf("\nDone\n");

        if 	( master->cndType & CU_STR_ARR   )  // Write back from the candidate array to list  .
        {
          printf("\nCopying candidates from array to list.\n");

          NV_RANGE_PUSH("Add to list");
          int cdx;

          long long numindep;

          double  poww = 0;
          double  sig, sigx, sigc, diff;
          double  gpu_p, gpu_q;
          double  rr, zz;
          int     added = 0;
          int     numharm;
          initCand*   candidate = (initCand*)cuSrch->h_candidates;
          FILE *  pFile;
          int n;
          char name [1024];
          pFile = fopen ("CU_CAND_ARR.csv","w");
          fprintf (pFile, "idx;rr;zz;sig;harm\n");
	  ulong max = cuSrch->candStride;

	  if ( master->flags  & FLAG_STORE_ALL )
	    max *= master->noHarmStages; // Store  candidates for all stages

          for (cdx = 0; cdx < max; cdx++)  // Loop  .
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
              fprintf (pFile, "%i;%.2f;%.2f;%.2f;%i\n",cdx,rr, zz,sig,numharm);
            }
          }

          fclose (pFile);
          NV_RANGE_POP();
        }
        else if ( master->cndType & CU_STR_LST   )
        {
          candsGPU  = (GSList*)cuSrch->h_candidates;

          if ( candsGPU )
          {
            if ( candsGPU->data == NULL )
            {
              // No real candidates found!
              candsGPU = NULL;
            }
          }
        }
        else if ( master->cndType & CU_STR_QUAD  ) // Copying candidates from array to list for optimisation  .
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

#ifdef NVVP
        cudaProfilerStop();
#endif

        //cudaDeviceSynchronize();
        gettimeofday(&end, NULL);
        gpuTime += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
        cands = candsGPU;
      }
    }

    FOLD //  Print output  .
    {
      printf("\n\nDone searching.\n");
      printf("   We got %7lli bad input values.\n",             badInp  );
      printf("   We got %7lli bad complex planes.\n",           badCplx );
      printf("   Found  %7lli GPU values above %.2f sigma.\n",  badCands+sameCands, obs.sigma);
      printf("          %6.2f%% were similar to CPU values.\n", sameCands/double(badCands+sameCands)*100.0 );
      printf("\n");
    }

    compareCands(candsCPU, candsGPU, obs.T );

    cands = candsCPU; // Only optimise CPU candidates

    printf("\n\nDone searching.  Now optimizing each candidate.\n\n");
  }

  if(1) /* Candidate list trimming and optimization */
  {
    int numcands;
    GSList *listptr;
    accelcand *cand;
    fourierprops *props;

#ifdef CUDA
    char timeMsg[1024], dirname[1024], scmd[1024];
    time_t rawtime;
    struct tm* ptm;
    time ( &rawtime );
    ptm = localtime ( &rawtime );
    sprintf ( timeMsg, "%04i%02i%02i%02i%02i%02i", 1900 + ptm->tm_year, ptm->tm_mon + 1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec );

    sprintf(dirname,"/home/chris/accel/Nelder_Mead/%s-pre", timeMsg );
    mkdir(dirname, 0755);
    sprintf(scmd,"mv /home/chris/accel/Nelder_Mead/n* %s/       2> /dev/null", dirname );
    system(scmd);
    sprintf(scmd,"mv /home/chris/accel/Nelder_Mead/*.png %s/    2> /dev/null", dirname );
    system(scmd);

    sprintf(scmd,"mv /home/chris/accel/*.png %s/                2>/dev/null", dirname );
    system(scmd);

    sprintf(scmd,"mv /home/chris/accel/*.csv %s/                2>/dev/null", dirname );
    system(scmd);

    gettimeofday(&start, NULL);			// Profiling
    NV_RANGE_PUSH("Optimisation");
    gettimeofday(&start, NULL);			// Note could start the timer after kernel init
#endif

    numcands = g_slist_length(cands);

    if (numcands)
    {
      /* Sort the candidates according to the optimized sigmas */

      cands = sort_accelcands(cands);

      char name [1024];
      sprintf(name,"%s_GPU_02_Cands_Sorted.csv", fname);
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

      /* Now optimise each candidate and its harmonics */

      FOLD //       --=== Dual Optimisation ===--  .
      {
#ifdef CUDA // Profiling  .
        NV_RANGE_PUSH("CPU");
        gettimeofday(&start, NULL);		// Note could start the timer after kernel init
#endif

        printf("Optimising %i candidates.\n\n", numcands);

        print_percent_complete(0, 0, NULL, 1);

        listptr = cands;

        //cuOptCand* oPlnPln;

        FOLD // Initialise optimisation details!
        {
          cuSrch = initCuOpt(&sSpec, &gSpec, cuSrch);
        }

        //        FOLD  // Set device  .
        //        {
        //          int device = gSpec.devId[0];
        //          if ( device >= getGPUCount() )
        //          {
        //            fprintf(stderr, "ERROR: There is no CUDA device %i.\n",device);
        //            exit(EXIT_FAILURE);
        //          }
        //          int currentDevvice;
        //          CUDA_SAFE_CALL(cudaSetDevice(device), "Failed to set device using cudaSetDevice");
        //          CUDA_SAFE_CALL(cudaGetDevice(&currentDevvice), "Failed to get device using cudaGetDevice");
        //          if (currentDevvice != device)
        //          {
        //            fprintf(stderr, "ERROR: CUDA Device not set.\n");
        //            exit(EXIT_FAILURE);
        //          }
        //        }

        //oPlnPln   = initOptPln(&sSpec);

        cuPlnGen* oPlnPln = &(cuSrch->oInf->opts[0]);
        setDevice(oPlnPln->gInf->devid);

        accelcand *candCPU;
        accelcand *candGPU;

        struct timeval startL, endL;
        long long cTime, gTime, cSum, gSum;

        cSum = 0;
        gSum = 0;

        int sm = 0;
        int bt = 0;
        int wo = 0;
        int dff = 0;

        for (ii = 0; ii < numcands; ii++)       //       ----==== Main Loop ====----  .
        {
          candGPU   = (accelcand *) (listptr->data);
          candCPU   = duplicate_accelcand(candGPU);

          //if ( ii == 92 )
          {
            //printf("Initial point %3i  r: %13.5f   z: %10.5f   power: %20.6f             sigma: %9.3f \n", ii+1, candCPU->r, candCPU->z, candCPU->power, candCPU->sigma);

            FOLD // CPU optimisation  .
            {
              NV_RANGE_PUSH("CPU opt");
              gettimeofday(&startL, NULL);       // Profiling
              optimize_accelcand(candCPU, &obs, ii+1);
              gettimeofday(&endL, NULL);
              cTime = (endL.tv_sec - startL.tv_sec) * 1e6 + (endL.tv_usec - startL.tv_usec);
              cSum += cTime;
              NV_RANGE_POP();
              //printf("CPU Opt point      r: %13.5f   z: %10.5f   power: %20.6f             sigma: %9.3f \n", candCPU->r, candCPU->z, candCPU->power, candCPU->sigma);
            }

            float sig1 = candCPU->sigma;
            float dist;

            FOLD // GPU optimisation  .
            {
              NV_RANGE_PUSH("Pln opt");
              gettimeofday(&startL, NULL);       // Profiling
              //opt_candPlns(candGPU, cuSrch, &obs, ii+1, oPlnPln);
              opt_accelcand(candGPU, oPlnPln, ii+1);
              gettimeofday(&endL, NULL);
              gTime = (endL.tv_sec - startL.tv_sec) * 1e6 + (endL.tv_usec - startL.tv_usec);
              gSum += gTime;
              NV_RANGE_POP();
              //printf("GPU Pln            r: %13.5f   z: %10.5f   power: %20.6f   %7.3fx  sigma: %9.3f ", candGPU->r, candGPU->z, candGPU->power, cTime/(float)gTime, candGPU->sigma);
              dist = sqrt( (candCPU->r-candGPU->r)*(candCPU->r-candGPU->r) + (candCPU->z-candGPU->z)*(candCPU->z-candGPU->z) );
              //printf(" Dist %10.6f   ", dist );
              if (dist > 0.1 )
                dff++;
              if ( candGPU->power > candCPU->power )
              {
                //printf("better\n");
                bt++;
              }
              else if ( ( candCPU->power / candGPU->power ) < 1.01 )
              {
                //printf("similar\n");
                sm++;
              }
              else
              {
                //printf("worse       -----\n");
                wo++;
              }
            }
          }

          listptr = listptr->next;
          print_percent_complete(ii, numcands, "optimization", 0);
          //printf("\n");
        }

        printf("\n\nSpeedup %.2fx  similar: %i   better: %i   worse: %i    different location: %i\n",cSum/(float)gSum, sm, bt, wo, dff);
      }

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
    }
    else
    {
      printf("No candidates above sigma = %.2f were found.\n\n", obs.sigma);
    }

#ifdef CUDA
    NV_RANGE_POP();
    gettimeofday(&end, NULL);
    optTime += (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);

    //if ( pltOpt > 0 )
    if ( sSpec.flags & FLAG_DPG_PLT_OPT )
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

    printf("\n CPU: %9.06f  GPU: %9.06f [%6.2f x]  Optimisation: %9.06f \n\n", cupTime * 1e-6, gpuTime * 1e-6, cupTime / (double) gpuTime, optTime * 1e-6 );
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

  printf("Final candidates in binary format are in '%s'.\n", obs.candnm);
  printf("Final Candidates in a text format are in '%s'.\n\n", obs.accelnm);

  free_accelobs(&obs);
  g_slist_foreach(cands, free_accelcand, NULL);
  g_slist_free(cands);

  return (0);
}
