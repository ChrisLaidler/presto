/*
extern "C"
{
#include "accel.h"
}

//#ifdef CBL
#include "array.h"
#include "arrayDsp.h"
#include "util.h"
//#endif

#ifdef USEMMAP
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

#ifdef CUDA
#include "cuda_accel.h"
#include "cuda_utils.h"
#include "cuda_accel_utils.h"
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
#include <cuda_profiler_api.h>

#include <sys/time.h>
#include <time.h>
#endif


//#include "utilstats.h"
#ifdef WITHOMP
#include <omp.h>
#endif

#ifdef USEDMALLOC
#include "dmalloc.h"
#endif
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
#include "cuda_accel.h"
#include "cuda_accel_utils.h"
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
#include <cuda_profiler_api.h>

#include <sys/time.h>
#include <time.h>
#endif

#ifdef WITHOMP
#include <omp.h>
#endif

#ifdef USEDMALLOC
#include "dmalloc.h"
#endif

#ifdef CBL
#include "array.h"
#include "arrayDsp.h"
#include "util.h"
#endif

int     pltOpt    = 0;

extern float calc_median_powers(fcomplex * amplitudes, int numamps);
//extern void zapbirds(double lobin, double hibin, FILE * fftfile, fcomplex * fft);

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
  ffdot->zlo = calc_required_z(harm_fract, obs->zlo);

  /* Initialize the lookup indices */
  if (numharm > 1 && !obs->inmem) {
    double rr, subr;
    for (ii = 0; ii < numrs_full; ii++) {
      rr = fullrlo + ii * ACCEL_DR;
      subr = calc_required_r(harm_fract, rr);
      shi->rinds[ii] = index_from_r(subr, ffdot->rlo);
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
          r1 = std::min(r1,fabs(1-cpu1->sigma/gpu1->sigma));
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
          r2 = std::min(r2,fabs(1-cpu1->sigma/gpu2->sigma));
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
        // Close reation
        silar++;
      }
      else if  ( super )
      {
        // There is a better GPU candidate
        superseede++;

        /*
        if ( r1 < r2 )
        {
          if ( rr1 < ratio)
          {
            silar++;
          }
          else
          {
            gpuBetterThanCPU++;

            printf("\n");
            printf("↑ %5.3f GPU candidate has a higher sigma by %.3f \n", rr1, gpu1->sigma - cpu1R->sigma);
            printf("         CPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", cpu1R->r, cpu1R->z, cpu1R->sigma, cpu1R->power );
            printf("         GPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", gpu1->r, gpu1->z, gpu1->sigma, gpu1->power );
          }
        }
        else if (r2 < r1 )
        {
          if ( rr2 < ratio)
          {
            silar++;
          }
          else
          {
            gpuBetterThanCPU++;

            printf("\n");
            printf("↑ %5.3f GPU candidate has a higher sigma by %.3f \n", rr2, gpu2->sigma - cpu2R->sigma);
            printf("         CPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", cpu2R->r, cpu2R->z, cpu2R->sigma, cpu2R->power );
            printf("         GPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", gpu2->r, gpu2->z, gpu2->sigma, gpu2->power );
          }
        }
        else
        {
          printf("I'm not sure what to do here \n");
        }
         */

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
          r1 = std::min(r1,fabs(1-cpu1->sigma/gpu1->sigma) );
        cpul = cpul->next;
      }

      GSList *cpul2 = candsCPU;
      while (cpul2->next && ((accelcand*)cpul2->data)->r < gpu1->r + ACCEL_CLOSEST_R )
      {
        cpu2 = (accelcand*)cpul2->data;
        r2 = std::min(r2,fabs(1-cpu2->sigma/gpu1->sigma) );
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

#ifdef DEBUG
  showOptionValues();
#endif

  cmd = parseCmdline(argc, argv);

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

  if ( obs.inmem ) // Force to standard search  .
  {
    printf("Reverting to standard accelsearch.\n");
    obs.inmem = 0;
    vect_free(obs.ffdotplane);
    obs.ffdotplane = NULL;
  }

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

#ifdef CUDA     // Profiling  .
  nvtxRangePop();
  gettimeofday(&end, NULL);
  prepTime += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));

  cuSearch*     cuSrch;
  gpuSpecs      gSpec;
  searchSpecs   sSpec;

  gSpec         = readGPUcmd(cmd);
  sSpec         = readSrchSpecs(cmd, &obs);
  sSpec.pWidth  = ACCEL_USELEN; // NB: must have same accellen for tests!
#endif

  char fname[1024];
  sprintf(fname,"%s_hs%02i_zmax%06.1f_sig%06.3f", obs.rootfilenm, obs.numharmstages, obs.zhi, obs.sigma );

  char candsFile[1024];
  sprintf(candsFile,"%s.unoptcands", fname );

  FILE *file;
  if ( (file = fopen(candsFile, "rb")) && useUnopt )       // Read candidates from previous search  . // TMP
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
  }
  else                                              // Run Search  .
  {
    long long badInp  = 0;
    long long badCplx = 0;

    long long badCands  = 0;
    long long goodCands = 0;

    /* Start the main search loop */

    FOLD //  -- Main Loop --
    {
      ffdotpows *fundamental;
      double startr = obs.rlo, lastr = 0, nextr = 0;

      int noHarms   = (1 << (obs.numharmstages - 1));
      candsGPU      = NULL;

      cuFFdotBatch* master   = NULL;    // The first kernel stack created
      cuSearch*     cuSrch;

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

      int noEl = 8;
      fcomplex *tempkern;

      nDarray<2, float> DFF_kernels;

      //cudaDeviceSynchronize();          // This is only necessary for timing
      gettimeofday(&start, NULL);       // Profiling
      //cudaProfilerStart();              // Start profiling, only really necessary debug and profiling, surprise surprise

      FOLD // Generate the GPU kernel  .
      {
        cuSrch        = initCuSearch(&sSpec, &gSpec, NULL);

        master        =  &cuSrch->mInf->kernels[0];   // The first kernel created holds global variables

        if ( master->accelLen != ACCEL_USELEN )
        {
          fprintf(stderr, "ERROR: GPU and CPU step size do not match!\n");
          exit(EXIT_FAILURE);
        }
      }

      FOLD // Test kernels  .
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

            float frac  = (float)(harm)/(float)harmtosum;
            int idx     = noHarms - frac * noHarms;

            cuHarmInfo  *hinf   = &cuSrch->mInf->kernels[0].hInfos[idx];
            subharminfo *sinf0  = subharminfs[0];
            subharminfo *sinf1  = subharminfs[1];
            subharminfo *sinf   = &subharminfs[stage][harm - 1];

            CPU_kernels.addDim(hinf->width*2, 0, hinf->width);
            CPU_kernels.addDim(hinf->height, -hinf->zmax, hinf->zmax);
            CPU_kernels.allocate();

            GPU_kernels.addDim(hinf->width*2, 0, hinf->width);
            GPU_kernels.addDim(hinf->height, -hinf->zmax, hinf->zmax);
            GPU_kernels.allocate();

            // Copy data from device
            CUDA_SAFE_CALL(cudaMemcpy(GPU_kernels.elems, cuSrch->mInf->kernels[0].kernels[idx].d_kerData, GPU_kernels.getBuffSize(), cudaMemcpyDeviceToHost), "Failed to kernel copy data from.");
            //CUDA_SAFE_CALL(cudaDeviceSynchronize(),"Error synchronising");

            for ( int row=0; row < sinf->numkern; row++  )
            {
              memcpy(CPU_kernels.getP(0,row), sinf->kern[row].data, hinf->width*sizeof(fcomplex));
            }

            //printf("   Input: %02i (%.2f)    MSE: %15.10f  μ: %10.5f   σ: %10.5f\n", harz, trdStack->hInfos[harz].harmFrac, MSE, stat.mean, stat.sigma );

            basicStats statG = GPU_kernels.getStats(true);
            basicStats statC = CPU_kernels.getStats(true);
            double MSE = GPU_kernels.MSE(CPU_kernels);
            double ERR =  MSE / statG.sigma ;
            printf("   Cmplx: %02i (%.2f)  MSE: %10.3e    μ: %10.3e    σ: %10.3e    MSE/σ: %9.2e ", idx, frac, MSE, statG.mean, statG.sigma, ERR );

            if      ( ERR > 1e1   )
              printf("  BAD!    Not even in the same realm.\n");
            else if ( ERR > 1e0   )
              printf("  Bad.  \n" );
            else if ( ERR > 1e-4  )
              printf("  Bad.   But not that bad.\n");
            else if ( ERR > 1e-6  )
              printf("  Close  But not great. \n");
            else if ( ERR > 1e-10 )
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

      char fname[1024];

      if ( cmd->gpuP >= 0) // -- Main Loop --  .
      {
        int firstStep       = 0;
        bool printDetails   = false;
        bool printBadLines  = false;
        bool CSV            = false;

        printf("\nRunning GPU search with %i simultaneous families of f-∂f plains spread across %i device(s).\n", cuSrch->mInf->noSteps, cuSrch->mInf->noDevices);
        printf("\nWill check all input and plains and report ");
        if (printDetails)
          printf("all results");
        else
          printf("only poor or bad results");
        printf("\n\n");

        omp_set_num_threads(cuSrch->mInf->noBatches);

        int harmtosum, harm;
        startr = obs.rlo, lastr = 0, nextr = 0;

        print_percent_complete(startr - obs.rlo, obs.highestbin - obs.rlo, "search", 1);

        int ss = 0;
        int maxxx = ( obs.highestbin - obs.rlo ) / (float)( master->accelLen * ACCEL_DR ) ;

        float ns = ( obs.highestbin - obs.rlo ) / (float)( master->accelLen * ACCEL_DR ) ;

        if ( maxxx < 0 )
          maxxx = 0;

        nDarray<2, float> plotPowers;
        //cuSrch->mInf->batches[0].hInfos[0].numrs;

        plotPowers.addDim(master->accelLen,  0, master->accelLen  );
        plotPowers.addDim(master->hInfos[0].height, 0, master->hInfos[0].height );
        plotPowers.allocate();

        //#pragma omp parallel // Note the CPU version is not set up to be thread capable so can't really test multi-threading
        FOLD // -- Main Loop --  .
        {
          int tid = 0;  //omp_get_thread_num();
          cuFFdotBatch* trdBatch = &cuSrch->mInf->batches[tid];

          nDarray<1, float> **cpuInput = new nDarray<1, float>*[trdBatch->noSteps];
          nDarray<2, float> **cpuCmplx = new nDarray<2, float>*[trdBatch->noSteps];
          nDarray<1, float> **gpuInput = new nDarray<1, float>*[trdBatch->noSteps];
          nDarray<2, float> **gpuCmplx = new nDarray<2, float>*[trdBatch->noSteps];

          nDarray<2, float> **cpuPowers = new nDarray<2, float>*[trdBatch->noSteps];
          nDarray<2, float> **gpuPowers = new nDarray<2, float>*[trdBatch->noSteps];

          FOLD // Initialise data structures to hold test data for comparisons  .
          {
            for ( int step = 0; step < trdBatch->noSteps ; step ++)
            {
              FOLD  // Create arrays of pointers
              {
                cpuInput[step]  = new nDarray<1, float>[trdBatch->noHarms];
                cpuCmplx[step]  = new nDarray<2, float>[trdBatch->noHarms];

                gpuInput[step]  = new nDarray<1, float>[trdBatch->noHarms];
                gpuCmplx[step]  = new nDarray<2, float>[trdBatch->noHarms];

                cpuPowers[step] = new nDarray<2, float>[trdBatch->noHarms];
                gpuPowers[step] = new nDarray<2, float>[trdBatch->noHarms];
              }

              //for (int stage = 0; stage < obs.numharmstages; stage++) // allocate arrays
              {
                //int harmtosum = 1 << stage;
                //for (int harm = 1; harm <= harmtosum; harm += 2)
                for (int harm = 0; harm < cuSrch->noHarms; harm++ )
                {
                  //float frac = (float)(harm)/(float)harmtosum;
                  //int idx = noHarms - frac * noHarms;
                  int idx = trdBatch->stageIdx[harm];

                  cuHarmInfo *hinf  = &cuSrch->mInf->kernels[0].hInfos[idx];

                  cpuInput[step][idx].addDim(hinf->width*2, 0, hinf->width);
                  cpuInput[step][idx].allocate();

                  cpuCmplx[step][idx].addDim(hinf->width*2, 0, hinf->width);
                  cpuCmplx[step][idx].addDim(hinf->height, -hinf->zmax, hinf->zmax);
                  cpuCmplx[step][idx].allocate();

                  gpuInput[step][idx].addDim(hinf->width*2, 0, hinf->width);
                  gpuInput[step][idx].allocate();

                  gpuCmplx[step][idx].addDim(hinf->width*2, 0, hinf->width);
                  gpuCmplx[step][idx].addDim(hinf->height, -hinf->zmax, hinf->zmax);
                  gpuCmplx[step][idx].allocate();

                  cpuPowers[step][idx].addDim(hinf->width, 0, hinf->width);
                  cpuPowers[step][idx].addDim(hinf->height, -hinf->zmax, hinf->zmax);
                  cpuPowers[step][idx].allocate();

                  gpuPowers[step][idx].addDim(hinf->width, 0, hinf->width);
                  gpuPowers[step][idx].addDim(hinf->height, -hinf->zmax, hinf->zmax);
                  gpuPowers[step][idx].allocate();
                }
              }
            }
          }

          double*  startrs = (double*)malloc(sizeof(double)*trdBatch->noSteps);
          double*  lastrs  = (double*)malloc(sizeof(double)*trdBatch->noSteps);
          size_t   rest    = trdBatch->noSteps;

          setDevice( trdBatch ) ;

          int noCands = 0;

          while ( ss < maxxx ) // -- Main Loop --  .
          {
#pragma omp critical
            FOLD
            {
              firstStep = ss;
              ss       += trdBatch->noSteps;
            }

            if ( firstStep >= maxxx )
              break;

            if ( firstStep + trdBatch->noSteps >= maxxx )
            {
              int tmp = 0;
              rest    = maxxx - firstStep;
            }

            // Set start r-vals for all steps in this batch
            for ( int step = 0; step < trdBatch->noSteps ; step ++)
            {
              if ( step < rest )
              {
                startrs[step] = obs.rlo + (firstStep+step) * ( trdBatch->accelLen * ACCEL_DR );
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
              search_ffdot_batch_CU(trdBatch, startrs, lastrs, obs.norm_type, 1,  (fcomplexcu*)obs.fft, obs.numindep);
            }

            FOLD // Copy data from device  .
            {
              ulong sz  = 0;
              harm      = 0;

              // Write data to page locked memory
              for (int ss = 0; ss < trdBatch->noStacks; ss++)
              {
                cuFfdotStack* cStack = &trdBatch->stacks[ss];

                // Synchronise
                //cudaStreamWaitEvent(cStack->fftPStream, cStack->plnComp, 0);
                for (int si = 0; si < cStack->noInStack; si++)
                {
                  cuHarmInfo* cHInfo    = &trdBatch->hInfos[harm];      // The current harmonic we are working on
                  cuFFdot*    plan      = &cStack->plains[si];          // The current plain

                  for ( int step = 0; step < trdBatch->noSteps; step ++) // Loop over steps
                  {
                    rVals* rVal = &((*trdBatch->rConvld)[step][harm]);

                    if (rVal->numdata)
                    {
                      //int diff = plan->numrs[step] - cHInfo->numrs;
                      //int diff = plan->numrs[step] - cHInfo->numrs;

                      // Copy input data from GPU
                      fcomplexcu *data = &trdBatch->d_iData[sz];
                      CUDA_SAFE_CALL(cudaMemcpyAsync(gpuInput[step][harm].elems, data, cStack->strideCmplx*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftIStream), "Failed to copy input data from device.");

                      // Copy pain from GPU
                      for( int y = 0; y < cHInfo->height; y++ )
                      {
                        fcomplexcu *cmplxData;
                        float *powers;
                        if      ( trdBatch->flag & FLAG_ITLV_ROW )
                        {
                          cmplxData = &plan->d_plainData[(y*trdBatch->noSteps + step)*cStack->strideCmplx   + cHInfo->halfWidth * 2 ];
                          powers    = &plan->d_plainPowers[(y*trdBatch->noSteps + step)*cStack->stridePwrs + cHInfo->halfWidth * 2 ];
                        }
                        else if ( trdBatch->flag & FLAG_ITLV_PLN )
                        {
                          cmplxData = &plan->d_plainData[(y + step*cHInfo->height)*cStack->strideCmplx   + cHInfo->halfWidth * 2 ];
                          powers    = &plan->d_plainPowers[(y + step*cHInfo->height)*cStack->stridePwrs + cHInfo->halfWidth * 2 ];
                        }

                        if      ( trdBatch->flag & FLAG_CUFFT_CB_OUT )
                        {
                          //CUDA_SAFE_CALL(cudaMemcpyAsync(gpuPowers[step][harm].getP(0,y), powers, (plan->numrs[step])*sizeof(float),   cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
                          CUDA_SAFE_CALL(cudaMemcpyAsync(gpuPowers[step][harm].getP(0,y), powers, (rVal->numrs)*sizeof(float),   cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
                          /*
                                               for( int jj = 0; jj < plan->numrs[step]; jj++)
                                               {
                                                 float *add = gpuPowers[step][harm].getP(jj*2+1,y);
                                                 gpuPowers[step][harm].setPoint<ARRAY_SET>(add, 0);
                                               }
                           */
                        }
                        else
                        {
                          //cmplxData += cHInfo->halfWidth*ACCEL_RDR;
                          //CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[step][harm].getP(0,y), cmplxData, (plan->numrs[step])*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
                          CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[step][harm].getP(0,y), cmplxData, (rVal->numrs)*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
                        }
                      }
                    }

                    sz += cStack->strideCmplx;
                  }
                  harm++;
                }

                // New events for Synchronisation (this event will override the previous event)
                cudaEventRecord(cStack->prepComp, cStack->fftIStream);
                cudaEventRecord(cStack->plnComp,  cStack->fftPStream);
              }
            }

            FOLD // Now do an equivalent CPU search  .
            {
              for ( int step = 0; (step < trdBatch->noSteps) && ( firstStep+step < maxxx) ; step ++) // Loop over steps
              {
                double poww, sig, rr, zz;

                startr  = startrs[step];
                lastr   = lastrs[step];

                rVals* rVal;
#ifdef SYNCHRONOUS
                rVal = &((*trdBatch->rConvld)[step][0]);
#else
                rVal = &((*trdBatch->rSearch)[step][0]);
#endif

                FOLD // ????  .
                {
                  int stage, harmtosum, harm;
                  ffdotpows *subharmonic;

                  // Copy the fundamental's ffdot plane to the full in-core one
                  for (stage = 0; stage < obs.numharmstages; stage++)
                  {
                    harmtosum   = 1 << stage;

                    if (stage == 0)
                    {
                      fundamental = subharm_ffdot_plane_DBG(1, 1, startr, lastr, &subharminfs[0][0], &obs, &cpuInput[step][0], &cpuCmplx[step][0], &cpuPowers[step][0] );
                    }
                    else
                    {
                      for (harm = 1; harm < harmtosum; harm += 2)
                      {
                        float frac = (float)(harm)/(float)harmtosum;
                        int idx = noHarms - frac * noHarms;
                        {
                          subharmonic = subharm_ffdot_plane_DBG(harmtosum, harm, startr, lastr, &subharminfs[stage][harm - 1], &obs, &cpuInput[step][idx], &cpuCmplx[step][idx], &cpuPowers[step][idx] );

                          add_ffdotpows(fundamental, subharmonic, harmtosum, harm);

                          free_ffdotpows(subharmonic);
                        }
                      }
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

                        for ( int y = 0; y < trdBatch->hInfos->height; y++ )
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

                        int tmp = 0;
                      }
                    }

                    FOLD // Compare candidates
                    {
                      for ( int x = 0; x < trdBatch->accelLen; x++ )
                      {
                        int idx   = step*obs.numharmstages*trdBatch->hInfos->width + stage*trdBatch->hInfos->width + x ;

                        int iz    = 0;
                        poww      = 0;
                        sig       = 0;
                        zz        = 0;

                        if      ( trdBatch->retType & CU_CANDMIN  )
                        {
                          candMin candM         = ((candMin*)trdBatch->h_retData)[idx];
                          sig                   = candM.power;
                          poww                  = candM.power;
                          iz                    = candM.z;
                        }
                        else if ( trdBatch->retType & CU_POWERZ_S   )
                        {
                          candPZs candM          = ((candPZs*)trdBatch->h_retData)[idx];
                          sig                   = candM.value;
                          poww                  = candM.value;
                          iz                    = candM.z;
                        }
                        else if ( trdBatch->retType & CU_CANDBASC )
                        {
                          accelcandBasic candB  = ((accelcandBasic*)trdBatch->h_retData)[idx];
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
                          rr      = ( rVal->drlo + x *  ACCEL_DR )                            / (double)harmtosum ;
                          zz      = ( iz * ACCEL_DZ - trdBatch->hInfos[0].zmax )              / (double)harmtosum ;

                          float cPow;
                          float *row;

                          row   = fundamental->powers[iz];
                          cPow  = row[x];

                          float p1 = poww;
                          float p2 = cPow;

                          float err = fabs(1-p2/p1);

                          if ( err > 0.001 )
                          {
                            printf("Candidate r: %9.4f z: %5.2f  CPU pow: %6.2f  GPU pow: %6.2f   %8.6f \n", rr, zz, p2, p1, fabs(1-p2/p1) );
                            badCands++;
                          }
                          else
                          {
                            goodCands++;
                          }

                          int tmp = 0 ;
                        }

                      }
                    }

                    if ( CSV ) // Write CVS  .
                    {
                      double rr, zz;
                      char tName[1024];
                      sprintf(tName,"/home/chris/accel/h%02i_%015.4f-%015.4f.csv", harmtosum, rVal->drlo / (double)harmtosum, (rVal->drlo + (trdBatch->accelLen-1)*ACCEL_DR) / (double)harmtosum );
                      FILE *f2 = fopen(tName, "w");

                      fprintf(f2,"%i",harmtosum);

                      for ( int x = 0; x < trdBatch->accelLen; x++ )
                      {
                        rr      = ( rVal->drlo + x *  ACCEL_DR )                            / (double)harmtosum ;
                        fprintf(f2,"\t%.6f",rr);
                      }
                      fprintf(f2,"\n");

                      for ( int y = 0; y < trdBatch->hInfos->height; y++ )
                      {
                        zz      = ( y * ACCEL_DZ - trdBatch->hInfos[0].zmax )              / (double)harmtosum ;
                        fprintf(f2,"%.6f",zz);

                        for ( int x = 0; x < trdBatch->accelLen; x++ )
                        {
                          float yy2 = fundamental->powers[y][x];
                          fprintf(f2,"\t%.6f",yy2);
                        }
                        fprintf(f2,"\n");
                      }
                      fclose(f2);

                      if ( noCands != numcands )
                      {
                        //                      char cmd[1024];
                        //                      sprintf(cmd,"python ~/bin/bin/plt_ffd.py %s", tName);
                        //                      system(cmd);
                      }
                    }

                    noCands = numcands;
                  }
                }
                free_ffdotpows(fundamental);
              }
            }



            FOLD // Print MSE  .
            {
              for ( int step = 0; (step < trdBatch->noSteps) && ( firstStep+step < maxxx) ; step ++) // Loop over steps
              {
                bool good = true;
                bool bad  = false;

                if (printDetails)
                  printf("\n           ---- Step %03i of %03i ----\n",firstStep + step+1, maxxx);

                for ( int harz = 0; harz < trdBatch->noHarms; harz++ )
                {
                  basicStats stat = gpuInput[step][harz].getStats(true);
                  double MSE = gpuInput[step][harz].MSE(cpuInput[step][harz]);
                  double ERR = MSE / stat.sigma ;

                  if ( ERR > 1e-10  )
                  {
                    if ( good && !printDetails )
                      printf("\n           ---- Step %03i of %03i ----\n",firstStep + step+1, maxxx);

                    good = false;
                  }
                  if ( ERR > 1e-6   )
                  {
                    badInp++;
                    bad = true;
                  }

                  if ( !good || printDetails )
                  {
                    printf("   Input: %02i (%.2f)  MSE: %10.3e    μ: %10.3e    σ: %10.3e    MSE/σ: %9.2e ", harz, trdBatch->hInfos[harz].harmFrac, MSE, stat.mean, stat.sigma, ERR );

                    if      ( ERR > 1e1   )
                      printf("  BAD!    Not even in the same realm.\n");
                    else if ( ERR > 1e0   )
                      printf("  Bad.  \n" );
                    else if ( ERR > 1e-4  )
                      printf("  Bad.   But not that bad.\n");
                    else if ( ERR > 1e-6  )
                      printf("  Close  But not great. \n");
                    else if ( ERR > 1e-10 )
                      printf("  GOOD  But a bit high.\n");
                    else if ( ERR > 1e-15 )
                      printf("  GOOD \n"  );
                    else if ( ERR > 1e-19 )
                      printf("  GOOD  Very good.\n"  );
                    else
                      printf("  Great \n");
                  }

                }
                if ( bad )
                {
                  for ( int harz = 0; harz < trdBatch->noHarms; harz++ )
                  {
                    int y = trdBatch->hInfos[harz].height - 1;

                    printf("Harm: %02i\n", harz );
                    printf("CPU: ");
                    for ( int x = 0; x < 15; x++ )
                    {
                      printf(" %9.6f ", cpuInput[step][harz].get(x,y));
                    }
                    printf("\n");

                    printf("GPU: ");
                    for ( int x = 0; x < 15; x++ )
                    {
                      printf(" %9.6f ", gpuInput[step][harz].get(x,y));
                    }
                    printf("\n");
                  }
                }

                good = true;
                bad  = false;

                if ( trdBatch->flag & FLAG_CUFFT_CB_OUT )
                {
                  if ( printDetails )
                    printf("\n           ---- Step %03i of %03i ----\n",firstStep + step+1, maxxx);

                  for ( int harz = 0; harz < trdBatch->noHarms; harz++ )
                  {
                    basicStats stat = gpuPowers[step][harz].getStats(true);
                    double MSE = gpuPowers[step][harz].MSE(cpuPowers[step][harz]);
                    double ERR = MSE / stat.sigma ;

                    if ( ERR > 1e-12 )
                    {
                      if ( good && !printDetails )
                        printf("\n           ---- Step %03i of %03i ----\n",firstStep + step+1, maxxx);

                      good = false;
                    }
                    if ( ERR > 1e-6  )
                    {
                      badCplx++;
                      bad = true;
                    }

                    if ( !good || printDetails )
                    {
                      printf("  Powers: %02i (%.2f)  MSE: %10.3e    μ: %10.3e    σ: %10.3e    MSE/σ: %9.2e ", harz, trdBatch->hInfos[harz].harmFrac, MSE, stat.mean, stat.sigma, ERR );

                      if      ( ERR > 1e1   )
                        printf("  BAD!    Not even in the same realm.\n");
                      else if ( ERR > 1e0   )
                        printf("  Bad.  \n" );
                      else if ( ERR > 1e-4  )
                        printf("  Bad.   But not that bad.\n");
                      else if ( ERR > 1e-6  )
                        printf("  Close  But not great. \n");
                      else if ( ERR > 1e-10 )
                        printf("  GOOD  But a bit high.\n");
                      else if ( ERR > 1e-15 )
                        printf("  GOOD \n"  );
                      else if ( ERR > 1e-19 )
                        printf("  GOOD  Very good.\n"  );
                      else
                        printf("  Great \n");

                      if ( !good && printDetails )
                      {
                        float *powArr     = (float*)plotPowers.getP(0,0);

                        int nX = gpuPowers[step][harz].ax(0)->noEls() ;
                        int nY = gpuPowers[step][harz].ax(1)->noEls() ;

                        int width         = trdBatch->accelLen;

                        // Copy CPU powers
                        for(int y = 0; y < nY; y++ )
                        {
                          memcpy(&powArr[y*width],gpuPowers[step][harz].getP(0,y), width*sizeof(float));
                        }
                        sprintf(fname, "/home/chris/fdotplains/ffdot_S%05i_H%02i_GPU.png", firstStep+step+1, harz);
                        draw2DArray6(fname, powArr, width, nY, width, nY*3);


                        // Copy CPU powers
                        for(int y = 0; y < nY; y++ )
                        {
                          memcpy(&powArr[y*width],cpuPowers[step][harz].getP(0,y), width*sizeof(float));
                        }
                        sprintf(fname, "/home/chris/fdotplains/ffdot_S%05i_H%02i_CPU.png", firstStep+step+1, harz);
                        draw2DArray6(fname, powArr, width, nY, width, nY*3);
                      }
                    }
                  }

                  if ( bad )
                  {
                    for ( int harz = 0; harz < trdBatch->noHarms; harz++ )
                    {
                      printf("Harm: %02i\n", harz );
                      printf("CPU: ");
                      for ( int x = 0; x < 10; x++ )
                      {
                        printf(" %9.6f ", cpuPowers[step][harz].get(x,0));
                      }
                      printf("\n");

                      printf("GPU: ");
                      for ( int x = 0; x < 10; x++ )
                      {
                        printf(" %9.6f ", gpuPowers[step][harz].get(x,0));
                      }
                      printf("\n");
                    }
                  }
                }
                else
                {
                  if( printDetails )
                    printf("\n           ---- Step %03i of %03i ----\n",firstStep + step+1, maxxx);

                  for ( int harz = 0; harz < trdBatch->noHarms; harz++ )
                  {
                    basicStats stat = gpuCmplx[step][harz].getStats(true);
                    double MSE = gpuCmplx[step][harz].MSE(cpuCmplx[step][harz]);
                    double ERR = MSE / stat.sigma ;

                    if ( ERR > 1e-12  )
                    {
                      if ( good && !printDetails )
                        printf("\n           ---- Step %03i of %03i ----\n",firstStep + step+1, maxxx);

                      good = false;
                    }
                    if ( ERR > 1e-6   )
                    {
                      bad = true;
                      badCplx++;
                    }

                    if ( !good || printDetails )
                    {
                      printf("   Cmplx: %02i (%.2f)  MSE: %10.3e    μ: %10.3e    σ: %10.3e    MSE/σ: %9.2e ", harz, trdBatch->hInfos[harz].harmFrac, MSE, stat.mean, stat.sigma, ERR );

                      if      ( ERR > 1e1   )
                        printf("  BAD!    Not even in the same realm.\n");
                      else if ( ERR > 1e0   )
                        printf("  Bad.  \n" );
                      else if ( ERR > 1e-4  )
                        printf("  Bad.   But not that bad.\n");
                      else if ( ERR > 1e-6  )
                        printf("  Close  But not great. \n");
                      else if ( ERR > 1e-10 )
                        printf("  GOOD  But a bit high.\n");
                      else if ( ERR > 1e-15 )
                        printf("  GOOD \n"  );
                      else if ( ERR > 1e-19 )
                        printf("  GOOD  Very good.\n"  );
                      else
                        printf("  Great \n");

                    }

                    if ( !good && 0 )
                    {
                      fcomplex* cmplx   = (fcomplex*)gpuCmplx[step][harz].getP(0,0);
                      float *powArr     = (float*)plotPowers.getP(0,0);

                      int nX = gpuPowers[step][harz].ax(0)->noEls() ;
                      int nY = gpuPowers[step][harz].ax(1)->noEls() ;

                      int width         = trdBatch->accelLen;

                      // Calculate GPU powers
                      for(int y = 0; y < nY; y++ )
                      {
                        for(int x = 0; x < width; x++ )
                        {
                          powArr[y*width + x] = cmplx[y*nX + x].i*cmplx[y*nX + x].i + cmplx[y*nX + x].r*cmplx[y*nX + x].r ;
                        }
                      }

                      //sprintf(fname, "/home/chris/fdotplains/ffdot_S%05i_H%2i_01_GPU.png", firstStep+si, harz);
                      //drawArr(fname, &gpuPowers[si][harz], HM_G);

                      //int tmp = 0;

                      sprintf(fname, "/home/chris/fdotplains/ffdot_S%05i_H%02i_GPU.png", firstStep+step+1, harz);
                      draw2DArray6(fname, powArr, width, nY, width, nY*3);


                      // Copy CPU powers
                      for(int y = 0; y < nY; y++ )
                      {
                        memcpy(&powArr[y*width],cpuPowers[step][harz].getP(0,y), width*sizeof(float));
                      }
                      sprintf(fname, "/home/chris/fdotplains/ffdot_S%05i_H%02i_CPU.png", firstStep+step+1, harz);
                      draw2DArray6(fname, powArr, width, nY, width, nY*3);


                      // Copy CPU powers
                      for(int y = 0; y < nY; y++ )
                      {
                        for(int x = 0; x < width; x++ )
                        {
                          powArr[y*width + x] -= cmplx[y*nX + x].i*cmplx[y*nX + x].i + cmplx[y*nX + x].r*cmplx[y*nX + x].r ;
                        }
                      }
                      sprintf(fname, "/home/chris/fdotplains/ffdot_S%05i_H%02i_RES.png", firstStep+step+1, harz);
                      draw2DArray6(fname, powArr, width, nY, width, nY*3);

                      //fundamental = subharm_ffdot_plane(1, 1, startr, lastr, &subharminfs[0][0], &obs);
                      //draw2DArray6(fname, fundamental->powers[0], fundamental->numrs, fundamental->numzs, 4096, 1602);
                      //cands = search_ffdotpows(fundamental, 1, &obs, cands);
                    }
                  }

                  if ( bad )
                  {
                    for ( int harz = 0; harz < trdBatch->noHarms; harz++ )
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

              int tmp = 0;
            }

            print_percent_complete(startrs[0] - obs.rlo, obs.highestbin - obs.rlo, "search", 0);
          }

          FOLD  // Finish off CUDA search  .
          {
            // Set r values to 0 so as to not process details
            for ( int step = 0; step < trdBatch->noSteps ; step ++)
            {
              startrs[step] = 0;
              lastrs[step]  = 0;
            }

            // Finish searching the plains, this is required because of the out of order asynchronous calls
            for ( int step = 0 ; step < 2; step++ )
            {
              search_ffdot_batch_CU(trdBatch, startrs, lastrs, obs.norm_type, 1, (fcomplexcu*)obs.fft, obs.numindep);
            }
          }
        }

        print_percent_complete(obs.highestbin - obs.rlo, obs.highestbin - obs.rlo, "search", 0);

        printf("\nDone\n");

        if ( master->flag & CU_CAND_ARR    )  // Write back from the candidate array to list  .
        {
          printf("\nCopying candidates from array to list.\n");

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

          FILE * pFile;
          int n;
          char name [1024];
          pFile = fopen ("CU_CAND_ARR.csv","w");
          fprintf (pFile, "idx;rr;zz;sig;harm\n");

          for (cdx = 0; cdx < master->SrchSz->noOutpR; cdx++)  // Loop  .
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
          nvtxRangePop();
        }

        cudaProfilerStop();

        //cudaDeviceSynchronize();
        gettimeofday(&end, NULL);
        gpuTime += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));
        cands = candsGPU;
      }
    }

    printf("\n\nDone searching.\n");
    printf("   We got %7lli bad input values.\n",             badInp  );
    printf("   We got %7lli bad complex plains.\n",           badCplx );
    printf("   Found  %7lli GPU values above %.2f sigma.\n",  badCands+goodCands, obs.sigma);
    printf("          %6.2f%% were similar to CPU values.\n", goodCands/double(badCands+goodCands)*100.0 );
    printf("\n");

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

    gettimeofday(&start, NULL);       // Profiling
    nvtxRangePush("Optimisation");
    gettimeofday(&start, NULL);       // Note could start the timer after kernel init
    //cudaProfilerStart();              // TMP Start profiling, only really necessary for debug and profiling, surprise surprise
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
        nvtxRangePush("CPU");
        gettimeofday(&start, NULL); // Note could start the timer after kernel init
#endif

        printf("Optimising %i candidates.\n\n", numcands);

        print_percent_complete(0, 0, NULL, 1);

        listptr = cands;

        cuOptCand* oPlnPln;

        FOLD  // Set device  .
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

        oPlnPln   = initOptPln(&sSpec);

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
              nvtxRangePush("CPU opt");
              gettimeofday(&startL, NULL);       // Profiling
              optimize_accelcand(candCPU, &obs, ii+1);
              gettimeofday(&endL, NULL);
              cTime = ((endL.tv_sec - startL.tv_sec) * 1e6 + (endL.tv_usec - startL.tv_usec));
              cSum += cTime;
              nvtxRangePop();
              //printf("CPU Opt point      r: %13.5f   z: %10.5f   power: %20.6f             sigma: %9.3f \n", candCPU->r, candCPU->z, candCPU->power, candCPU->sigma);
            }

            float sig1 = candCPU->sigma;
            float dist;

            FOLD // GPU optimisation  .
            {
              nvtxRangePush("Pln opt");
              gettimeofday(&startL, NULL);       // Profiling
              opt_candPlns(candGPU, &obs, ii+1, oPlnPln);
              gettimeofday(&endL, NULL);
              gTime = ((endL.tv_sec - startL.tv_sec) * 1e6 + (endL.tv_usec - startL.tv_usec));
              gSum += gTime;
              nvtxRangePop();
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
