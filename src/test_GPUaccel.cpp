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

extern float calc_median_powers(fcomplex * amplitudes, int numamps);
//extern void zapbirds(double lobin, double hibin, FILE * fftfile, fcomplex * fft);

static void print_percent_complete(int current, int number, const char *what, int reset)
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

/** A function to comper CPU and GPU canidates
 * It looks for canidates missed by each search
 * and compers the ration of sigma's of the various detections
 */
void compareCands(GSList *candsCPU, GSList *candsGPU )
{
  printf("\nComapreing GPU and CPU raw canidate lists\n");

  if (  candsCPU == NULL &&  candsGPU == NULL )
  {
    printf("No canidates found, try searching some noisy data\n");

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

    printf("CPU search found %i canidates and GPU found nothing\n",cands);
    printf("Writing canidates to CPU_Cands.csv\n");
    printCands("CPU_Cands.csv", candsCPU);

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

    printf("GPU search found %i canidates and CPU found nothing\n",gpuCands);
    printf("Writing canidates to GPU_Cands.csv\n");
    printCands("GPU_Cands.csv", candsCPU);

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
  printf("Got: %5i CPU canidates\n",cpuCands);

  while (gpuLst->next)
  {
    gpuCands++;
    gpuLst = gpuLst->next;
  }
  printf("Got: %5i GPU canidates\n",gpuCands);

  printf("\nWriting canidates to CPU_Cands.csv and GPU_Cands.csv\n");
  printCands("CPU_Cands.csv", candsCPU);
  printCands("GPU_Cands.csv", candsGPU);

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
      //printf("\n----------\n");

      cpu1 = (accelcand*)cpul->data;
      r1 = 10;
      r2 = 10;

      double rr1 = 10;
      double rr2 = 10;
      double rrMin;
      cpu1R = NULL;
      cpu2R = NULL;

      FOLD // Get Neibouring GPU vals  .
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
          printf("      below  r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", gpu1->r, gpu1->z, gpu1->numharm, gpu1->sigma, gpu1->power );

        printf("CPU canidate r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f   Not fould in GPU canidates.\n", cpu1->r, cpu1->z, cpu1->numharm, cpu1->sigma, cpu1->power );

        if (gpu2)
          printf("      above  r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", gpu2->r, gpu2->z, gpu2->numharm, gpu2->sigma, gpu2->power );

        gpuMissed++;
      }
      else if  ( minr < ratio )
      {
        // Close reation
        silar++;
      }
      else if  ( super )
      {
        // There is a better GPU canidate
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
            printf("↑ %5.3f GPU canidate has a higher sigma by %.3f \n", rr1, gpu1->sigma - cpu1R->sigma);
            printf("        CPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", cpu1R->r, cpu1R->z, cpu1R->sigma, cpu1R->power );
            printf("        GPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", gpu1->r, gpu1->z, gpu1->sigma, gpu1->power );
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
            printf("↑ %5.3f GPU canidate has a higher sigma by %.3f \n", rr2, gpu2->sigma - cpu2R->sigma);
            printf("        CPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", cpu2R->r, cpu2R->z, cpu2R->sigma, cpu2R->power );
            printf("        GPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", gpu2->r, gpu2->z, gpu2->sigma, gpu2->power );
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
          printf("↓ %5.3f GPU canidate has a lower sigma by %.3f \n", r1, s1);
          printf("        CPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", cpu1->r, cpu1->z, cpu1->numharm, cpu1->sigma, cpu1->power );
          printf("        GPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", gpu1->r, gpu1->z, gpu1->numharm, gpu1->sigma, gpu1->power );
        }
        else
        {
          printf("↓ %5.3f GPU canidate has a lower sigma by %.3f \n", r2, s2);
          printf("        CPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", cpu1->r, cpu1->z, cpu1->numharm, cpu1->sigma, cpu1->power );
          printf("        GPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", gpu2->r, gpu2->z, gpu2->numharm, gpu2->sigma, gpu2->power );
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
          printf("      below  r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", cpu1->r, cpu1->z, cpu1->numharm, cpu1->sigma, cpu1->power );

        printf("GPU canidate r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f   Not fould in CPU canidates.\n", gpu1->r, gpu1->z, gpu1->numharm, gpu1->sigma, gpu1->power );

        if (cpu2)
          printf("      above  r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", cpu2->r, cpu2->z, cpu2->numharm, cpu2->sigma, cpu2->power );
      }
      else if ( minr > ratio )
      {
        gpuBetterThanCPU++;

        printf("\n");
        if ( r1 < r2 )
        {
          printf("  %5.3f GPU canidate is signficantly higher that all coverd CPU canidates.\n", r1);
          printf("        CPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", cpu1->r, cpu1->z, cpu1->numharm, cpu1->sigma, cpu1->power );
          printf("        GPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", gpu1->r, gpu1->z, gpu1->numharm, gpu1->sigma, gpu1->power );
        }
        else
        {
          printf("  %5.3f GPU canidate is signficantly higher that all coverd CPU canidates\n", r2);
          printf("        CPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", cpu2->r, cpu2->z, cpu2->numharm, cpu2->sigma, cpu2->power );
          printf("        GPU: r: %10.1f   z: %6.1f  h: %02i   Sigma: %8.2f   Power: %10.2f\n", gpu1->r, gpu1->z, gpu1->numharm, gpu1->sigma, gpu1->power );
        }
      }
      gpul = gpul->next;
    }
  }

  printf("\nSummary:\n");
  printf("%4i (%4.2f%%) Similar canidates were found\n",silar+superseede,(silar+superseede)/(float)cpuCands*100.0);
  //printf("%4i (%4.2f%%) CPU canidates that were 'coverd' by the GPU list.\n",superseede,superseede/(float)cpuCands*100.0);
  printf("%4i (%4.2f%%) Missed by GPU  <- These may be due to the way the dynamic list is created. Check CU_CAND_ARR.csv.\n",gpuMissed,gpuMissed/(float)cpuCands*100.0);
  printf("%4i (%4.2f%%) Missed by CPU  <- These may be due to the way the dynamic list is created.\n",cpuMissed,cpuMissed/(float)cpuCands*100.0);
  printf("%4i (%4.2f%%) Where the GPU sigma was significantly better that the CPU.\n",gpuBetterThanCPU, gpuBetterThanCPU/(float)cpuCands*100.0);
  printf("%4i (%4.2f%%) Where the CPU sigma was significantly better that the GPU.\n",cpuBetterThanGPU, cpuBetterThanGPU/(float)cpuCands*100.0);
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

  FOLD //  -- Main Loop --
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

    int flags = FLAG_RETURN_ALL | CU_CAND_ARR ; // | FLAG_CUFFTCB_OUT;

    FOLD // Generate the GPU kernel  .
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

      FOLD // Create a kernel on each device  .
      {
        kernels = (cuStackList*)malloc(cmd->gpuC*sizeof(cuStackList));
        int added;

        fftInfo fftinf;
        fftinf.fft    = obs.fft;
        fftinf.nor    = obs.N;
        fftinf.rlow   = obs.rlo;
        fftinf.rhi    = obs.rhi;

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

          added = initHarmonics(&kernels[noKers], master, obs.numharmstages, (int)obs.zhi, fftinf, cmd->gpu[dev], noSteps, ACCEL_USELEN, no, obs.powcut, obs.numindep, flags, CU_FULLCAND, CU_SMALCAND, (void*)candsGPU);

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

      FOLD // Create plains for calculations  .
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

          float frac = (float)(harm)/(float)harmtosum;
          int idx = noHarms - frac * noHarms;

          cuHarmInfo *hinf    = &kernels[0].hInfos[idx];
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
          CUDA_SAFE_CALL(cudaMemcpy(GPU_kernels.elems, kernels[0].kernels[idx].d_kerData, GPU_kernels.getBuffSize(), cudaMemcpyDeviceToHost), "Failed to kernel copy data from.");
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
          printf("   Cmplx: %02i (%.2f)  MSE: %10.3e    μ: %10.3e    σ: %10.3e    MSE/σ: %9.2e ", idx, frac, MSE, stat.mean, stat.sigma, ERR );

          if      ( ERR > 1e1   )
            printf("  BAD!    Not even in the same realm.\n");
          else if ( ERR > 1e0   )
            printf("  Bad.  \n" );
          else if ( ERR > 1e-4  )
            printf("  Bad.   But not that bad.\n");
          else if ( ERR > 1e-6 )
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

    if ( cmd->gpuP >= 0) // -- Main Loop --
    {
      int firstStep       = 0;
      bool printDetails   = false;
      bool printBadLines  = false;

      printf("\nRunning GPU search with %i simultaneous families of f-∂f plains spread across %i device(s).\n", noSteps, noKers);
      printf("\nWill check all input and plains and report ");
      if (printDetails)
        printf("all results");
      else
        printf("only poor or bad results");
      printf("\n\n");

      //omp_set_num_threads(nPlains);

      int harmtosum, harm;
      startr = obs.rlo, lastr = 0, nextr = 0;

      print_percent_complete(startr - obs.rlo, obs.highestbin - obs.rlo, "search", 1);

      int ss = 0;
      int maxxx = ( obs.highestbin - obs.rlo ) / (float)( master->accelLen * ACCEL_DR ) ;

      float ns = ( obs.highestbin - obs.rlo ) / (float)( master->accelLen * ACCEL_DR ) ;

      if ( maxxx < 0 )
        maxxx = 0;

      nDarray<2, float> plotPowers;
      plainsj[0]->hInfos->numrs;
      plotPowers.addDim(plainsj[0]->hInfos->numrs,0,plainsj[0]->hInfos->numrs);
      plotPowers.addDim(plainsj[0]->hInfos->height,0,plainsj[0]->hInfos->height);
      plotPowers.allocate();

      //#pragma omp parallel // Note the CPU version is not set up to be thread capable so can't really test multi-threading
      FOLD // -- Main Loop --
      {
        int tid = 0;  //omp_get_thread_num();
        cuStackList* trdStack = plainsj[tid];

        nDarray<1, float> **cpuInput = new nDarray<1, float>*[trdStack->noSteps];
        nDarray<2, float> **cpuCmplx = new nDarray<2, float>*[trdStack->noSteps];
        nDarray<1, float> **gpuInput = new nDarray<1, float>*[trdStack->noSteps];
        nDarray<2, float> **gpuCmplx = new nDarray<2, float>*[trdStack->noSteps];

        nDarray<2, float> **cpuPowers = new nDarray<2, float>*[trdStack->noSteps];
        nDarray<2, float> **gpuPowers = new nDarray<2, float>*[trdStack->noSteps];

        FOLD // Initialise data structures to hold test data for comparisons  .
        {
          for ( int step = 0; step < trdStack->noSteps ; step ++)
          {
            FOLD  // Create arryas of pointers
            {
              cpuInput[step]  = new nDarray<1, float>[trdStack->noHarms];
              cpuCmplx[step]  = new nDarray<2, float>[trdStack->noHarms];

              gpuInput[step]  = new nDarray<1, float>[trdStack->noHarms];
              gpuCmplx[step]  = new nDarray<2, float>[trdStack->noHarms];

              cpuPowers[step] = new nDarray<2, float>[trdStack->noHarms];
              gpuPowers[step] = new nDarray<2, float>[trdStack->noHarms];
            }

            for (int stage = 0; stage < obs.numharmstages; stage++) // allocate arryas
            {
              int harmtosum = 1 << stage;
              for (int harm = 1; harm <= harmtosum; harm += 2)
              {
                float frac = (float)(harm)/(float)harmtosum;
                int idx = noHarms - frac * noHarms;

                cuHarmInfo *hinf  = &kernels[0].hInfos[idx];

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

        double*  startrs = (double*)malloc(sizeof(double)*trdStack->noSteps);
        double*  lastrs  = (double*)malloc(sizeof(double)*trdStack->noSteps);

        setContext( trdStack ) ;

        while ( ss < maxxx ) // -- Main Loop --
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
            // TODO: some work here
          }

          for ( int si = 0; si < trdStack->noSteps ; si ++)  // Set R searvch values  .
          {
            startrs[si] = obs.rlo + (firstStep+si) * ( master->accelLen * ACCEL_DR );
            lastrs[si]  = startrs[si] + master->accelLen * ACCEL_DR - ACCEL_DR;
          }

          FOLD  // Call the CUDA search  .
          {
            search_ffdot_planeCU(trdStack, startrs, lastrs, obs.norm_type, 1,  (fcomplexcu*)obs.fft, obs.numindep, &candsGPU);

            if ( trdStack->flag & CU_OUTP_HOST )
            {
              // TODO: check the addressing below for new cases ie:FLAG_STORE_EXP FLAG_STORE_ALL
              // TODO: Fix this!
              //trdStack->d_candidates = &master->d_candidates[master->accelLen*obs.numharmstages*firstStep] ;
            }
          }

          FOLD // Now do an equivalent CPU search  .
          {
            for ( int step = 0; (step < trdStack->noSteps) && ( firstStep+step < maxxx) ; step ++) // Loop over steps
            {
              startr  = startrs[step];
              lastr   = lastrs[step];

              fundamental = subharm_ffdot_plane_DBG(1, 1, startr, lastr, &subharminfs[0][0], &obs, &cpuInput[step][0], &cpuCmplx[step][0], &cpuPowers[step][0] );

              candsCPU    = search_ffdotpows(fundamental, 1, &obs, candsCPU);

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
                          &obs, &cpuInput[step][idx], &cpuCmplx[step][idx], &cpuPowers[step][idx] );

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
                  cuFFdot*    plan      = &cStack->plains[si];          // The current plain

                  for ( int step = 0; (step < trdStack->noSteps) && ( firstStep+step < maxxx) ; step ++) // Loop over steps
                  {
                    int diff = plan->numrs[step] - cHInfo->numrs;

                    if ( diff != 0 )
                    {
                      TMP
                    }

                    // Copy input data from GPU
                    fcomplexcu *data = &trdStack->d_iData[sz];
                    CUDA_SAFE_CALL(cudaMemcpyAsync(gpuInput[step][harm].elems, data, cStack->inpStride*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftIStream), "Failed to copy input data from device.");

                    // Copy pain from GPU
                    for( int y = 0; y < cHInfo->height; y++ )
                    {
                      fcomplexcu *cmplxData;
                      float *powers;
                      if ( trdStack->flag & FLAG_STP_ROW )
                      {
                        cmplxData = &plan->d_plainData[(y*trdStack->noSteps + step)*cStack->inpStride   + cHInfo->halfWidth * 2 ];
                        powers    = &plan->d_plainPowers[(y*trdStack->noSteps + step)*cStack->pwrStride + cHInfo->halfWidth * 2 ];
                      }
                      else if ( trdStack->flag & FLAG_STP_PLN )
                      {
                        cmplxData = &plan->d_plainData[(y + step*cHInfo->height)*cStack->inpStride   + cHInfo->halfWidth * 2 ];
                        powers    = &plan->d_plainPowers[(y + step*cHInfo->height)*cStack->pwrStride + cHInfo->halfWidth * 2 ];
                      }

                      if ( trdStack->flag & FLAG_CUFFTCB_OUT )
                      {
                        CUDA_SAFE_CALL(cudaMemcpyAsync(gpuPowers[step][harm].getP(0,y), powers, (plan->numrs[step])*sizeof(float),   cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
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
                        CUDA_SAFE_CALL(cudaMemcpyAsync(gpuCmplx[step][harm].getP(0,y), cmplxData, (plan->numrs[step])*2*sizeof(float), cudaMemcpyDeviceToHost, cStack->fftPStream), "Failed to copy input data from device.");
                      }
                    }

                    sz += cStack->inpStride;
                  }
                  harm++;
                }

                // New events for Synchronsition (this event will ovrride the previous event)
                cudaEventRecord(cStack->prepComp, cStack->fftIStream);
                cudaEventRecord(cStack->plnComp, cStack->fftPStream);
              }
            }
          }

          FOLD // Print MSE  .
          {
            for ( int step = 0; (step < trdStack->noSteps) && ( firstStep+step < maxxx) ; step ++) // Loop over steps
            {
              bool good = true;
              bool bad = false;

              if (printDetails)
                printf("\n           ---- Step %03i of %03i ----\n",firstStep + step+1, maxxx);

              for ( int harz = 0; harz < trdStack->noHarms; harz++ )
              {
                basicStats stat = gpuInput[step][harz].getStats(true);
                double MSE = gpuInput[step][harz].MSE(cpuInput[step][harz]);
                double ERR =  MSE / stat.sigma ;

                if ( ERR > 1e-10  )
                {
                  if ( good && !printDetails )
                    printf("\n           ---- Step %03i of %03i ----\n",firstStep + step+1, maxxx);


                  good = false;
                }
                if ( ERR > 1e-6  )
                {
                  badInp++;
                  bad = true;
                }

                if ( !good || printDetails )
                {
                  printf("   Input: %02i (%.2f)  MSE: %10.3e    μ: %10.3e    σ: %10.3e    MSE/σ: %9.2e ", harz, trdStack->hInfos[harz].harmFrac, MSE, stat.mean, stat.sigma, ERR );

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
              if ( bad  )
              {
                for ( int harz = 0; harz < trdStack->noHarms; harz++ )
                {
                  printf("Harm: %02i\n", harz );
                  printf("CPU: ");
                  for ( int x = 0; x < 15; x++ )
                  {
                    printf(" %9.6f ", cpuInput[step][harz].get(x,0));
                  }
                  printf("\n");

                  printf("GPU: ");
                  for ( int x = 0; x < 15; x++ )
                  {
                    printf(" %9.6f ", gpuInput[step][harz].get(x,0));
                  }
                  printf("\n");
                }
              }

              good = true;
              bad  = false;

              if ( trdStack->flag & FLAG_CUFFTCB_OUT )
              {
                if ( printDetails )
                  printf("\n           ---- Step %03i of %03i ----\n",firstStep + step+1, maxxx);

                for ( int harz = 0; harz < trdStack->noHarms; harz++ )
                {
                  basicStats stat = gpuPowers[step][harz].getStats(true);
                  double MSE = gpuPowers[step][harz].MSE(cpuPowers[step][harz]);
                  double ERR =  MSE / stat.sigma ;

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
                    printf("  Powers: %02i (%.2f)  MSE: %10.3e    μ: %10.3e    σ: %10.3e    MSE/σ: %9.2e ", harz, trdStack->hInfos[harz].harmFrac, MSE, stat.mean, stat.sigma, ERR );

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

                      int width         = trdStack->hInfos[harz].numrs;

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
                  for ( int harz = 0; harz < trdStack->noHarms; harz++ )
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

                for ( int harz = 0; harz < trdStack->noHarms; harz++ )
                {
                  basicStats stat = gpuCmplx[step][harz].getStats(true);
                  double MSE = gpuCmplx[step][harz].MSE(cpuCmplx[step][harz]);
                  double ERR =  MSE / stat.sigma ;

                  if ( ERR > 1e-12  )
                  {
                    if ( good && !printDetails )
                      printf("\n           ---- Step %03i of %03i ----\n",firstStep + step+1, maxxx);

                    good = false;
                  }
                  if ( ERR > 1e-6  )
                  {
                    bad = true;
                    badCplx++;
                  }

                  if ( !good || printDetails )
                  {
                    printf("   Cmplx: %02i (%.2f)  MSE: %10.3e    μ: %10.3e    σ: %10.3e    MSE/σ: %9.2e ", harz, trdStack->hInfos[harz].harmFrac, MSE, stat.mean, stat.sigma, ERR );

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

                    int width         = trdStack->hInfos[harz].numrs;

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
                  for ( int harz = 0; harz < trdStack->noHarms; harz++ )
                  {
                    printf("Harm: %02i\n", harz );
                    printf("CPU: ");
                    for ( int x = 0; x < 15; x++ )
                    {
                      printf(" %9.6f ", cpuCmplx[step][harz].get(x,0));
                    }
                    printf("\n");

                    printf("GPU: ");
                    for ( int x = 0; x < 15; x++ )
                    {
                      printf(" %9.6f ", gpuCmplx[step][harz].get(x,0));
                    }
                    printf("\n");
                  }
                }
              }
            }
          }

          print_percent_complete(startrs[0] - obs.rlo, obs.highestbin - obs.rlo, "search", 0);
        }

        FOLD  // Finish off CUDA search  .
        {
          for ( int si = 0; si < trdStack->noSteps ; si ++)
          {
            startrs[si] = 0;
            lastrs[si]  = 0;
          }

          // Finish searching the plains, this is required because of the out of order asynchronous calls
          for ( int  pln = 0 ; pln < 2; pln++ )
          {
            search_ffdot_planeCU(trdStack, startrs, lastrs, obs.norm_type, 1,  (fcomplexcu*)obs.fft, obs.numindep, &candsGPU);
          }
        }
      }

      print_percent_complete(obs.highestbin - obs.rlo, obs.highestbin - obs.rlo, "search", 0);

      printf("\nDone\n");

      if ( master->flag & CU_CAND_ARR    )  // Write baues from the canidate array to list  .
      {
        printf("\nCopying canidates from array to list.\n");

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
        char name [100];
        pFile = fopen ("CU_CAND_ARR.csv","w");
        fprintf (pFile, "idx;rr;zz;sig;harm\n");

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
            fprintf (pFile, "%i;%.2f;%.2f;%.2f;%i\n",cdx,rr, zz,sig,numharm);
          }
        }

        fclose (pFile);
        nvtxRangePop();
      }

      if ( master->flag & CU_OUTP_DEVICE )  // Write values from device memory to list  .
      {
        nvtxRangePush("Add to list");

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
          CUDA_SAFE_CALL(cudaMemcpy(master->h_retData, master->d_retData, master->SrchSz->noOutpR*sizeof(accelcandBasic), cudaMemcpyDeviceToHost), "Failed to copy data to device");

          accelcandBasic* bsk = (accelcandBasic*)master->h_retData;

          for (cdx = 0; cdx < master->SrchSz->noOutpR; cdx++)
          {
            sig        = bsk[cdx].sigma;

            if ( sig > 0 )
            {
              numharm   = bsk[cdx].numharm;
              numindep  = obs.numindep[twon_to_index(numharm)];

              if ( master->flag  & FLAG_STORE_EXP )
                rr      = master->SrchSz->searchRLow + cdx*ACCEL_DR;
              else
                rr      = master->SrchSz->searchRLow + cdx;

              zz        = bsk[cdx].z;
              candsGPU  = insert_new_accelcand(candsGPU, poww, sig, numharm, rr, zz, &added);
            }
          }
        }
        nvtxRangePop();
      }

      cudaProfilerStop();

      cudaDeviceSynchronize();
      gettimeofday(&end, NULL);
      gpuTime += ((end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec));
      cands = candsGPU;

    }
  }
  printf("\n\nDone searching.\n");
  printf("   We got %lli bad input values.\n", badInp);
  printf("   We got %lli bad complex plains.\n\n", badCplx);
  
  compareCands(candsCPU, candsGPU );

  printf("\n\nDone searching.  Now optimizing each candidate.\n\n");

  if(1) /* Candidate list trimming and optimization */
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