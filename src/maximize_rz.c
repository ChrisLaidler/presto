#include "presto.h"
#include <sys/stat.h>
#include <sys/time.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define ZSCALE 4.0

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

static fcomplex *maxdata;
static int nummaxdata, max_kern_half_width;

extern void amoeba(double p[3][2], double y[], double ftol,
                   double (*funk) (double[]), int *nfunk);

static double power_call_rz(double rz[])
/* f-fdot plane power function */
{
   double powargr, powargi;
   fcomplex ans;

   /* num_funct_calls++; */
   rz_interp(maxdata, nummaxdata, rz[0], rz[1] * ZSCALE, max_kern_half_width, &ans);
   return -POWER(ans.r, ans.i);
}

double max_rz_arr(fcomplex * data, int numdata, double rin, double zin,
                  double *rout, double *zout, rderivs * derivs)
/* Return the Fourier frequency and Fourier f-dot that      */
/* maximizes the power.                                     */
{
   double y[3], x[3][2], step = 0.4;
   float locpow;
   int numeval;

   maxdata = data;
   nummaxdata = numdata;
   locpow = get_localpower3d(data, numdata, rin, zin, 0.0);

   /*  Now prep the maximization at LOWACC for speed */

   /* Use a slightly larger working value for 'z' just incase */
   /* the true value of z is a little larger than z.  This    */
   /* keeps a little more accuracy.                           */

   max_kern_half_width = z_resp_halfwidth(fabs(zin) + 4.0, LOWACC);

   /* Initialize the starting simplex */

   x[0][0] = rin - step;
   x[0][1] = zin / ZSCALE - step;
   x[1][0] = rin - step;
   x[1][1] = zin / ZSCALE + step;
   x[2][0] = rin + step;
   x[2][1] = zin / ZSCALE;

   /* Initialize the starting function values */

   y[0] = power_call_rz(x[0]);
   y[1] = power_call_rz(x[1]);
   y[2] = power_call_rz(x[2]);

   /* Call the solver: */

   numeval = 0;
   amoeba(x, y, 1.0e-7, power_call_rz, &numeval);

   /*  Restart at minimum using HIGHACC to get a better result */

   max_kern_half_width = z_resp_halfwidth(fabs(x[0][1]) + 4.0, HIGHACC);

   /* Re-Initialize some of the starting simplex */

   x[1][0] = x[0][0] + 0.01;
   x[1][1] = x[0][1];
   x[2][0] = x[0][0];
   x[2][1] = x[0][1] + 0.01;

   /* Re-Initialize the starting function values */

   y[0] = power_call_rz(x[0]);
   y[1] = power_call_rz(x[1]);
   y[2] = power_call_rz(x[2]);

   /* Call the solver: */

   numeval = 0;
   amoeba(x, y, 1.0e-10, power_call_rz, &numeval);

   /* The following calculates derivatives at the peak           */

   *rout = x[0][0];
   *zout = x[0][1] * ZSCALE;
   locpow = get_localpower3d(data, numdata, *rout, *zout, 0.0);
   get_derivs3d(data, numdata, *rout, *zout, 0.0, locpow, derivs);
   return -y[0];
}

typedef struct particle
{
    double position[2];
    double value;
    double velocity[2];
    double bestPos[2];
    double bestVal;
} particle;

double max_rz_file(FILE * fftfile, double rin, double zin,
                   double *rout, double *zout, rderivs * derivs)
/* Return the Fourier frequency and Fourier f-dot that      */
/* maximizes the power of the candidate in 'fftfile'.       */
{
   double maxz, maxpow, rin_int, rin_frac;
   int kern_half_width, filedatalen, startbin, extra = 10;
   fcomplex *filedata;

   maxz = fabs(zin) + 4.0;
   rin_frac = modf(rin, &rin_int);
   kern_half_width = z_resp_halfwidth(maxz, HIGHACC);
   filedatalen = 2 * kern_half_width + extra;
   startbin = (int) rin_int - filedatalen / 2;

   filedata = read_fcomplex_file(fftfile, startbin, filedatalen);
   maxpow = max_rz_arr(filedata, filedatalen, rin_frac + filedatalen / 2,
                       zin, rout, zout, derivs);
   *rout += startbin;
   vect_free(filedata);
   return maxpow;
}

static int max_num_harmonics;
static fcomplex** maxdata_harmonics;
static float* maxlocpow;                  // local powers for normalisation
static int* maxr_offset;


static double power_call_rz_harmonics(double rz[])
{
    int i = 2;
    double total_power = 0.;
    double powargr, powargi;
    fcomplex ans;

    for( i=1; i<= max_num_harmonics; i++ )
    //i = 1 ;
    {
       rz_interp(maxdata_harmonics[i-1], nummaxdata, (maxr_offset[i-1]+rz[0])*i-maxr_offset[i-1], rz[1] * ZSCALE * i, max_kern_half_width, &ans);
       //total_power += POWER(ans.r, ans.i)/maxlocpow[i-1];
       total_power += POWER(ans.r, ans.i)/maxlocpow[i-1]/maxlocpow[i-1]; // TODO: to properly normalise this should be divided by the square of the nomalisation factor????
       //if( i == 1 )
       //  printf("Harm %02i  Power: %10.4f  %10.4f %10.4f norm: %.5f\n", i, POWER(ans.r, ans.i), ans.r, ans.i, maxlocpow[i-1] );
    }
    return -total_power;
}

static double power_call_rz_harmonics_noNorm(double rz[])
{
    int i;
    double total_power = 0.;
    double powargr, powargi;
    fcomplex ans;

    for(i=1; i<=max_num_harmonics; i++)
    {
       rz_interp(maxdata_harmonics[i-1], nummaxdata, (maxr_offset[i-1]+rz[0])*i-maxr_offset[i-1], rz[1] * ZSCALE * i, max_kern_half_width, &ans);
       total_power += POWER(ans.r, ans.i);
    }
    return -total_power;
}

/* Return the Fourier frequency and Fourier f-dot that      */
/* maximizes the power.                                     */
void max_rz_arr_harmonics(fcomplex* data[], int num_harmonics, int r_offset[], int numdata, double rin, double zin, double *rout, double *zout, rderivs derivs[], double power[], int nn)
{
   double y[3], x[3][2], step = 0.4;
   float *locpow;
   int numeval;
   int i;
   double bestVal = 0;
   double bestPos[2];
   char dirname[1024];
   double scale         = 10;

   // initialisation
   locpow             = gen_fvect(num_harmonics);
   maxlocpow          = gen_fvect(num_harmonics);
   maxr_offset        = r_offset;
   maxdata_harmonics  = data;
   FILE *stFile;

   // Calculate the max power around each harmonic for normalisation
   // FIXME: z needs to be multiplied by i everywhere
   //printf("Normalisation powers \n");
   for ( i=1; i<=num_harmonics; i++ )
   {
       //locpow[i-1]      = get_localpower3d(data[i-1], numdata, (r_offset[i-1]+rin)*i-r_offset[i-1], zin*i, 0.0);
       //float ff         = get_scaleFactorZ(data[i-1], numdata, (r_offset[i-1]+rin)*i-r_offset[i-1], zin*i, 0.0);
       locpow[i-1]      = get_scaleFactorZ(data[i-1], numdata, (r_offset[i-1]+rin)*i-r_offset[i-1], zin*i, 0.0);
       maxlocpow[i-1]   = locpow[i-1];
       //printf("  %02i  %8.3f \n", i, locpow[i-1] );
   }
   //printf("\n");
   nummaxdata           = numdata;
   max_num_harmonics    = num_harmonics;

   int skp    = 1 ;
   int swrm   = 1 ;

   /*  Now prep the maximization at LOWACC for speed */

   /* Use a slightly larger working value for 'z' just incase */
   /* the true value of z is a little larger than z.  This    */
   /* keeps a little more accuracy.                           */

   max_kern_half_width = z_resp_halfwidth(fabs(zin*num_harmonics) + 4.0, LOWACC);
   //max_kern_half_width = z_resp_halfwidth(fabs(zin*num_harmonics) + 4.0, HIGHACC);

   //TMP
   //rz_interp_cu(maxdata_harmonics[0], r_offset[0], numdata, 99995.112, -56.23, max_kern_half_width);

   double lrgPnt[2][3];
   double smlPnt[2][3];
   double bstGrd[3];
   bstGrd[2] = 0;

   x[0][0] = rin;
   x[0][1] = zin / ZSCALE;

   lrgPnt[0][0] = rin;
   lrgPnt[0][1] = zin;
   lrgPnt[0][2] = -power_call_rz_harmonics(x[0]);
   //printf("Inp Power %7.3f\n", lrgPnt[0][2] );
   printf("%4i  optimize_accelcand  harm %2i   r %10.3f   z %10.3f  pow: %8.3f \n", nn, num_harmonics, rin, zin, lrgPnt[0][2] );

   /* Initialize the starting simplex */

   x[0][0] = rin - step;
   x[0][1] = zin / ZSCALE - step;
   x[1][0] = rin - step;
   x[1][1] = zin / ZSCALE + step;
   x[2][0] = rin + step;
   x[2][1] = zin / ZSCALE;

   //printf("Simplex 01\n");

   /* Initialize the starting function values */

   y[0] = power_call_rz_harmonics(x[0]);
   y[1] = power_call_rz_harmonics(x[1]);
   y[2] = power_call_rz_harmonics(x[2]);

   /* Call the solver: */

   numeval = 0;
   amoeba(x, y, 1.0e-7, power_call_rz_harmonics, &numeval);

   /*  Restart at minimum using HIGHACC to get a better result */

   max_kern_half_width = z_resp_halfwidth(fabs(x[0][1]*ZSCALE*num_harmonics) + 4.0, HIGHACC); //TODO: add the ZSCALE term to PRESTO
   //max_kern_half_width = z_resp_halfwidth(fabs(x[0][1]*ZSCALE*num_harmonics) + 4.0, LOWACC);

   if ( swrm ) // particle swarm  .
   {
     int MaxTrials        = 500;
     int noBatches        = 1;
     int noInBatch        = 32;
     int noCom            = 100;
     double velocityMax   = 1.25;
     double velocity      = velocityMax;

     if ( num_harmonics == 1 )
     {
       noBatches  = 1;
       scale      = 8;
     }
     if ( num_harmonics == 2 )

     {
       noBatches  = 1;
       scale      = 7;
     }
     if ( num_harmonics == 4 )
     {
       noBatches  = 2;
       scale      = 6;
     }
     if ( num_harmonics == 8 )
     {
       noBatches  = 2;
       scale      = 5;
     }
     if ( num_harmonics == 16)
     {
       noBatches  = 3;
       scale      = 4;
     }

     FOLD // TMP
     {
       scale      = 10;
       noInBatch  = 32  ;
       noBatches  = 2  ;
       //scale = 32;
     }

     velocityMax          = scale / 20.0;
     velocity             = velocityMax;

     int noPoints         = noBatches * noInBatch;
     double batchBest[10][3];

     int noTrials         = ceil(MaxTrials/(float)noPoints);
     noTrials             = 40;

     int ixc;
     float noTet          = 1.0 ;
     const int maxErr     = 4  ;

     // set PSO parameters to the ‘standard’ values suggested in [14]:w= 0.729844 and C1 = C2 = 1.49618.
     //[14J Kennedy, M. Clerc, 2006. <http://www.particleswarm.info/Standard_PSO_2006.c>.]

     double a = 0.8;
     double b = 0.4;
     double c = 0.7;

     int idx;
     int btch;
     int ibtch;

     time_t t;
     FILE *file;
     time(&t);
     sprintf(dirname,"/home/chris/accel/Nelder_Mead/swrm_%i", t);

     if ( (file = fopen("/home/chris/accel/swrm_stats.csv", "r")) )
     {
       fclose(file);

       mkdir(dirname, 0755);
       char scmd[1024];
       sprintf(scmd,"mv /home/chris/accel/*.csv %s/",dirname );
       system(scmd);

       int tmp = 0;
     }

     if ( (file = fopen("/home/chris/accel/Nelder_Mead/swrm_000.png", "r")) )
     {
       fclose(file);

       mkdir(dirname, 0755);
       char scmd[1024];
       sprintf(scmd,"mv /home/chris/accel/Nelder_Mead/*.png %s/",dirname );
       system(scmd);

       int tmp = 0;
     }

     if( noTet > 1.0 )
     {
       stFile  = fopen("/home/chris/accel/swrm_stats.csv", "a");

       fprintf(stFile,"TG\tno\tv\tscale\tno\tnoIn\tCom\tx\ty\tval\tN\t");
       for ( ixc = 1; ixc <= maxErr; ixc++ )
       {
         fprintf(stFile,"C%i\tHit%i\t",ixc, ixc);
       }
       fprintf(stFile,"\n");
     }

     // init position
     particle* list = (particle*)malloc(noPoints*sizeof(particle));

     double gbX = 0;
     double gbY = 0;
     double gbV = 0;
     int havebest = 1;

     if ( nn == -1 ) //25
     {
     }
     /*
     else if ( nn == 00 ) //00
     {
       gbX = 0.0000397862707263;
       gbY = -0.0001075617316072;
       gbV = 194341361.6188967823982239;
     }
     else if ( nn == 02 ) //02
     {
       gbX = 1485.0521902093944391;
       gbY = 0.0246405830899129;
       gbV = 216.1036675292863265;
     }
     else if ( nn == 6 ) //06
     {
       gbX = 1272.9083488429641875;
       gbY = 0.0190627483258364;
       gbV = 65.0487000247432263;
     }
     else if( nn == 11 ) //11
     {
       gbX = 1371.0020835782290760;
       gbY = -0.1193450278986411;
       gbV = 57.5486607510589465;
     }
     else if ( nn == 18 ) //18
     {
       gbX = 16331.4150267042041378;
       gbY = 1.6163588023687263;
       gbV = 36.9130638243663753;
     }
     else if ( nn == 19 ) //19
     {
       gbX = 1497.2090951662569296;
       gbY = -8.1768316448560245;
       gbV = 17.7997458681113621;
     }
     else if ( nn == 25 ) //25
     {
       gbX = 963.7680999662366048;
       gbY = -0.9972862923470285;
       gbV = 18.3416480425218467;
     }
     else if ( nn == 54 ) //54
     {
       gbX = 1548.2525432181880660;
       gbY = -45.4097410118802998;
       gbV = 3.2837074206163894;
     }
     */
     else
     {
       havebest = 0;
     }

     srand (time(NULL));

     //for ( a = 0.1; a <= 0.8; a += 0.1 )
     //for ( scale = 6; scale <= 15; scale += 2 )
     //scale = 10 ;
     {
       //printf("s:    %4.2f\n",scale);

       //for ( velocityMax = 0.25; velocityMax <= 1.75; velocityMax += 0.25 )
       //for ( velocityMax = 4; velocityMax <= 20; velocityMax += 2 )
       {
         //printf("  v:  %4.2f\n",velocityMax);

         float sumV[maxErr+1];
         float cntV[maxErr+1];
         float conV[maxErr+1];

         for ( ixc = 0; ixc <= maxErr; ixc++ )
         {
           sumV[ixc]  = 0;
           cntV[ixc]  = 0;
         }

         velocity = velocityMax;

         char timeMsg[1024];
         time_t rawtime;
         struct tm* ptm;
         time ( &rawtime );
         ptm = localtime ( &rawtime );
         sprintf ( timeMsg, "%04i%02i%02i%02i%02i%02i", 1900 + ptm->tm_year, ptm->tm_mon + 1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec );

         sprintf(dirname,"/home/chris/accel/Nelder_Mead/n%04i_noBtch%1i_noInBtch%02i_scale%04.2f_vel%04.2f_vleMx%04.2f_a%04.2f_b%04.2f_c%04.2f_h%02i_%s", nn, noBatches, noInBatch, scale, velocity, velocityMax, a, b, c, num_harmonics, timeMsg );

         if( noTet > 1.0 )
         {
           fprintf(stFile,"%04.2f|%04.2f\t", velocity, scale );
           fprintf(stFile,"%i\t%04.2f\t%04.2f\t", nn, velocity, scale );
           fprintf(stFile,"%i\t", noBatches );
           fprintf(stFile,"%i\t", noInBatch );
           fprintf(stFile,"%i\t", noCom );
         }

         //fprintf(stFile,"%04i\t%i\t%02i\t%i\t%04.2f\t%04.2f\t%04.2f\t%04.2f\t%04.2f\t%04.2f\t%02i\t%s", nn, noBatches, noInBatch, interV, scale, velocity, velocityMax, a, b, c, num_harmonics, timeMsg );

         //for ( testn = 0; testn < noTet ; testn++ ) // Main loop  .
         {
           memset(list,0,sizeof(particle)*noPoints);

           list[0].position[0] = rin;
           list[0].position[1] = zin / ZSCALE;
           list[0].value = -power_call_rz_harmonics(list[0].position);
           list[0].bestVal = list[0].value;
           list[0].bestPos[0] = list[0].position[0];
           list[0].bestPos[1] = list[0].position[1];
           list[0].velocity[0] = -velocity/2.0 + rand()/(float)RAND_MAX*velocity;
           list[0].velocity[1] = (-velocity/2.0 + rand()/(float)RAND_MAX*velocity)*1;

           for (btch = 0; btch < noBatches; btch++ )
           {
             batchBest[btch][0] = rin;
             batchBest[btch][1] = zin / ZSCALE;
             batchBest[btch][2] = list[0].value ;
           }

           int dm = floor(sqrt(noPoints));
           //if(dm%2 == 0 )
           //  dm--;
           double rx, zy;
           double dd = scale / (dm-1) ;
           rx = rin - scale / 2.0;
           rx = rin - scale / 2.0;
           idx = 0;

           for ( zy = zin / ZSCALE - scale / 2.0 ; zy <= zin / ZSCALE + scale  / 2.0 + dd * 0.1 ; zy += dd )
           {
             for ( rx = rin - scale / 2.0 ; rx <= rin + scale / 2.0 + dd * 0.1 ; rx += dd )
             {
               list[idx].position[0] = rx;
               list[idx].position[1] = zy;
               list[idx].value = -power_call_rz_harmonics(list[idx].position);
               list[idx].bestVal = list[idx].value;
               list[idx].bestPos[0] = list[idx].position[0];
               list[idx].bestPos[1] = list[idx].position[1];
               list[idx].velocity[0] = -velocity/2.0 + rand()/(float)RAND_MAX*velocity;
               list[idx].velocity[1] = (-velocity/2.0 + rand()/(float)RAND_MAX*velocity)*1;

               idx++;
             }
           }

           // Random points
           for (; idx < noPoints; idx++)
           {
             list[idx].position[0] = rin - scale/2.0 + rand()/(float)RAND_MAX*scale;
             list[idx].position[1] = zin / ZSCALE - scale/2.0 + rand()/(float)RAND_MAX*scale*1;
             list[idx].value = -power_call_rz_harmonics(list[idx].position);                      // Calculate the inital value
             list[idx].bestVal = list[idx].value;
             list[idx].bestPos[0] = list[idx].position[0];
             list[idx].bestPos[1] = list[idx].position[1];
             list[idx].velocity[0] = -velocity/2.0 + rand()/(float)RAND_MAX*velocity;
             list[idx].velocity[1] = (-velocity/2.0 + rand()/(float)RAND_MAX*velocity)*1;
           }

           FILE *f = fopen("/home/chris/accel/ps.csv", "w");

           bestVal = 0;
           fprintf(f,"%i", noPoints);

           // Update global maximum
           for (btch = 0; btch < noBatches; btch++ )
           {
             for ( ibtch = 0; ibtch < noInBatch; ibtch++)
             {
               idx = btch*noInBatch + ibtch;

               fprintf(f,"\t%.6f\t%.6f\t%.6f",list[idx].position[0],list[idx].position[1]*ZSCALE,list[idx].value);

               if (list[idx].value > batchBest[btch][2] )
               {
                 batchBest[btch][2] = list[idx].value;
                 batchBest[btch][0] = list[idx].position[0];
                 batchBest[btch][1] = list[idx].position[1];
               }
             }
           }
           fprintf(f,"\n");

           int trial;
           double r1, r2, d1, d2, d3, rat, drat;
           double prefL, prefG;

           for ( ixc = 0; ixc <= maxErr; ixc++ )
           {
             conV[ixc] = -1;
           }

           for (trial = 0; trial < noTrials; trial++)  // Main Loop  .
           {
             FILE *f2 = fopen("/home/chris/accel/ps1.csv", "w");
             fprintf(f,"%i", noPoints);

             //printf("\r  SWRM: %03i  \r",trial);
             fflush(stdout);

             if ( ( (trial+1) % noCom) == 0 )
             {
               for (btch = 0; btch < noBatches; btch++ )
               {
                 if ( batchBest[btch][2] > bestVal )
                 {
                   bestVal    = batchBest[btch][2];
                   bestPos[0] = batchBest[btch][0];
                   bestPos[1] = batchBest[btch][1];
                 }
               }
             }

             for (btch = 0; btch < noBatches; btch++ )
             {
               int tmp = 0;

               for ( ibtch = 0; ibtch < noInBatch; ibtch++)
               {
                 idx = btch*noInBatch + ibtch;

                 rat  = list[idx].bestVal/bestVal;

                 r1 = rand()/(float)RAND_MAX;
                 r2 = rand()/(float)RAND_MAX;

                 d1 =  list[idx].bestPos[0] - list[idx].position[0];
                 d2 = gbX - list[idx].position[0];
                 d3 = gbY - list[idx].position[1];
                 double dst = sqrt(d2*d2 + d3*d3) ;

                 if ( fabs(1-rat) < 0.05 )
                 {
                   if(1)
                   {
                     prefL = 1.0;
                     prefG = 1.0;
                   }
                   else
                   {

                     if ( dst > 1 )
                     {
                       prefL = 1.0;
                       prefG = 0.0;
                     }
                     else if ( dst > 0.5 )
                     {
                       prefL = 1.0;
                       prefG = 0.2;
                     }
                     else if ( dst > 0.2 )
                     {
                       prefL = 1.0;
                       prefG = 1.0;
                     }
                   }
                 }
                 else
                 {
                   prefL = 1.0;
                   prefG = 1.0;
                 }

                 list[idx].velocity[0] = a*list[idx].velocity[0] + b*r1*prefL*(list[idx].bestPos[0] - list[idx].position[0]) + c*r2*prefG*(batchBest[btch][0] - list[idx].position[0]);
                 if (list[idx].velocity[0] > velocityMax )
                   list[idx].velocity[0] = velocityMax;
                 if (list[idx].velocity[0] < -velocityMax )
                   list[idx].velocity[0] = -velocityMax;

                 d1 =  list[idx].bestPos[1] - list[idx].position[1];
                 d2 =  batchBest[btch][1] - list[idx].position[1];
                 drat = fabs(0.1/d2);

                 list[idx].velocity[1] = a*list[idx].velocity[1] + b*r1*prefL*(list[idx].bestPos[1] - list[idx].position[1]) + c*r2*prefG*(batchBest[btch][1] - list[idx].position[1]);

                 if (list[idx].velocity[1] > velocityMax )
                   list[idx].velocity[1] = velocityMax;
                 if (list[idx].velocity[1] < -velocityMax )
                   list[idx].velocity[1] = -velocityMax;

                 // Update position
                 list[idx].position[0] += list[idx].velocity[0] ;
                 list[idx].position[1] += list[idx].velocity[1] ;

                 // Calculate the new value
                 list[idx].value = -power_call_rz_harmonics(list[idx].position);

                 // Update local maximum
                 if ( list[idx].value > list[idx].bestVal )
                 {
                   list[idx].bestVal = list[idx].value;
                   list[idx].bestPos[0] = list[idx].position[0];
                   list[idx].bestPos[1] = list[idx].position[1];
                 }

                 if( noTet > 1.0 )
                 {
                   if ( list[idx].value > gbV )
                   {
                     gbV = list[idx].value;
                     gbX = list[idx].position[0];
                     gbY = list[idx].position[1];

                     printf("else if ( nn == %02i ) //%02i\n{\n",nn,nn);
                     printf("  gbX = %.16f;\n",gbX);
                     printf("  gbY = %.16f;\n",gbY);
                     printf("  gbV = %.16f;\n}\n",gbV);

                     havebest = 0;
                   }
                 }

                 // Update global maximum
                 if ( list[idx].value > batchBest[btch][2] )
                 {
                   double bestSTp = list[idx].value - batchBest[btch][2];

                   batchBest[btch][2] = list[idx].value;
                   batchBest[btch][0] = list[idx].position[0];
                   batchBest[btch][1] = list[idx].position[1];

                   int best = 1;
                   int lb;

                   for (lb = 0; lb < noBatches; lb++ )
                   {
                     if ( batchBest[lb][2] > list[idx].value  )
                     {
                       best = 0;
                     }
                   }

                   if ( best && (noTet > 1.0) )
                   {
                     //fprintf(stFile,"\n%i\t%.15f\t%.15f\t%.15f", trial, list[idx].value, dst, bestSTp);
                     //fflush(stFile);
                     //printf("Best at trial %02i\n", trial);

                     for ( ixc = 1; ixc <= maxErr; ixc++ )
                     {
                       double eer = pow(0.1,(double)ixc);

                       if ( dst < eer && conV[ixc] == -1 ) // Set convergence
                       {
                         conV[ixc] = trial;
                         //printf("  Err %.10f\n", eer);
                       }
                     }

                     if ( havebest == 1 && dst < pow(0.1,(double)maxErr) ) // break if we close enough to the true max
                     {
                       trial = noTrials;
                     }
                   }
                 }

                 fprintf(f, "\t%.6f\t%.6f\t%.6f",list[idx].position[0],list[idx].position[1]*ZSCALE,list[idx].value);
                 fprintf(f2,"%.6f\t%.6f\t%.6f\n",list[idx].position[0],list[idx].position[1]*ZSCALE,list[idx].value);
               }
             }

             fprintf(f,"\n");
             fflush(f);
             fclose(f2);
           }

           fclose(f);

           for (btch = 0; btch < noBatches; btch++ )
           {
             if ( batchBest[btch][2] > bestVal )
             {
               bestVal    = batchBest[btch][2];
               bestPos[0] = batchBest[btch][0];
               bestPos[1] = batchBest[btch][1];
             }
           }
           if( noTet > 1.0 ) // Sum up if we found the correct max
           {
             d2 = gbX - bestPos[0];
             d3 = gbY - bestPos[1];

             double dst = sqrt(d2*d2 + d3*d3) ;

             if ( dst < 1e-2 )
             {
               // Found the right maximum

               for ( ixc = 1; ixc <= maxErr; ixc++ )
               {
                 if (conV[ixc] != -1)
                 {
                   cntV[ixc]++;
                   sumV[ixc] += conV[ixc];
                 }
               }
             }
           }
         }

         if( noTet > 1.0 )
         {
           fprintf(stFile,"%.15f\t%.15f\t%.15f\t%0.f\t",bestPos[0], bestPos[1]*ZSCALE, bestVal, noTet );
           for ( ixc = 1; ixc <= maxErr; ixc++ )
           {
             if ( cntV[ixc] > 0 )
               fprintf(stFile,"%5.2f\t%5.3f\t", sumV[ixc]/cntV[ixc], cntV[ixc]/noTet );
             else
               fprintf(stFile,"%5s\t%5.3f\t","-",0.0);
           }
           fprintf(stFile,"\n");
           fflush(stFile);

           printf("  v:  %4.2f   %4.2f\n", velocityMax, cntV[2]/noTet);
         }
       }
     }

     if( noTet > 1.0 )
       fclose(stFile);

     int tmp = 0;
   }

   if ( skp  )  // Large points  .
   {
     float szDiff = scale / 1.9 ;
     float* gpuPows;

     lrgPnt[1][0] = x[0][0];
     lrgPnt[1][1] = x[0][1] * ZSCALE;
     lrgPnt[1][2] = -y[0];

     int no = 40;

     if ( num_harmonics == 1 )
     {
       no = 40;
     }
     if ( num_harmonics == 2 )
     {
       no = 40;
     }
     if ( num_harmonics == 4 )
     {
       no = 50;
       no   = 40;
     }
     if ( num_harmonics == 8 )
     {
       no = 60;
       no   = 40;
     }
     if ( num_harmonics == 16)
     {
       no = 80;
       no   = 40;
     }

     no   = 100;

     double mx    = MAX(fabs(lrgPnt[0][0]-lrgPnt[1][0]), fabs(lrgPnt[0][1]-lrgPnt[1][1])/ZSCALE);
     double res   = MAX(szDiff/(float)no,mx*1.1/(float)no);

     double mx2   = MAX(fabs(lrgPnt[0][0]-bestPos[0]), fabs(lrgPnt[0][1]-bestPos[1]*ZSCALE)/ZSCALE);
     double res2  = MAX(res,(mx2+0.5)/(float)no);

     if ( res2 > res )
     {
       no   += (res2/res*8.0);
       no    = MIN(no, 200);

       res   = mx2*1.02/(float)no ;
     }

     int tp   = no*2+1 ;

     FOLD // GPU grid
     {
       gpuPows  = (float*)malloc( tp*tp*sizeof(float) );
       //ffdot(gpuPows, data[0],  r_offset[0], numdata, num_harmonics, lrgPnt[0][0], lrgPnt[0][1], (tp-1)*res, (tp-1)*res, tp, tp, max_kern_half_width, locpow);
       ffdot(gpuPows, data[0],  r_offset[0], numdata, num_harmonics, rin, zin, (tp-1)*res, (tp-1)*res*ZSCALE, tp, tp, max_kern_half_width, locpow);
     }

     FILE *f1 = fopen("/home/chris/accel/lrg.csv", "w");

     double sx, sy;
     int indx = 0;
     int indy = 0;
     double ff[2];
     fprintf(f1,"%i",num_harmonics);
     for (sx = rin - res*no, indx = 0; indx < no*2+1 ; sx += res, indx++ )
     {
       fprintf(f1,"\t%.6f",sx);
     }
     fprintf(f1,"\n");

     for (sy = zin / ZSCALE - res*no ; indy < no*2+1;  sy += res, indy++ )
     {
       ff[1]=sy;
       fprintf(f1,"%.6f",sy*ZSCALE);

       for (sx = rin - res*no, indx = 0; indx < no*2+1 ; sx += res, indx++ )
       {
         ff[0]=sx;
         double yy = -power_call_rz_harmonics(ff);

         if (yy > bstGrd[2] )
         {
           bstGrd[2] = yy;
           bstGrd[0] = sx;
           bstGrd[1] = sy;
         }

         double yy2 = gpuPows[indy*tp+indx];
         fprintf(f1,"\t%.6f",yy);
       }
       fprintf(f1,"\n");
     }
     fclose(f1);

     system("python ~/bin/bin/plt_lrg.py");

     free(gpuPows);

     int tmp = 0;
   }

   /* Re-Initialize some of the starting simplex */

   x[1][0] = x[0][0] + 0.01;
   x[1][1] = x[0][1];
   x[2][0] = x[0][0];
   x[2][1] = x[0][1] + 0.01;

   //printf("Simplex 02\n");

   /* Re-Initialize the starting function values */

   y[0] = power_call_rz_harmonics(x[0]);
   y[1] = power_call_rz_harmonics(x[1]);
   y[2] = power_call_rz_harmonics(x[2]);

   /* Call the solver: */

   numeval = 0;
   amoeba(x, y, 1.0e-10, power_call_rz_harmonics, &numeval);

   *rout = x[0][0];
   *zout = x[0][1] * ZSCALE;

   if( skp ) // Small points  .
   {
     smlPnt[1][0] = x[0][0];
     smlPnt[1][1] = x[0][1] * ZSCALE;
     smlPnt[1][2] = -y[0];

     double minX, maxX, minY, maxY;

     minX = MIN(lrgPnt[0][0],lrgPnt[1][0]);
     minX = MIN(minX, smlPnt[1][0] );
     minX = MIN(minX, bestPos[0] );
     minX = MIN(minX, bstGrd[0] );

     maxX = MAX(lrgPnt[0][0],lrgPnt[1][0]);
     maxX = MAX(maxX, smlPnt[1][0] );
     maxX = MAX(maxX, bestPos[0] );
     maxX = MAX(maxX, bstGrd[0] );

     minY = MIN(lrgPnt[0][1],lrgPnt[1][1]);
     minY = MIN(minY, smlPnt[1][1] );
     minY = MIN(minY, bestPos[1]*ZSCALE );
     minY = MIN(minY, bstGrd[1]*ZSCALE );

     maxY = MAX(lrgPnt[0][1],lrgPnt[1][1]);
     maxY = MAX(maxY, smlPnt[1][1] );
     maxY = MAX(maxY, bestPos[1]*ZSCALE );
     maxY = MAX(maxY, bstGrd[1]*ZSCALE );

     double rin = minX + ( maxX - minX)/2.0 ;
     double zin = minY + ( maxY - minY)/2.0 ;

     //double rin = bestPos[0] ;
     //double zin = bestPos[1]*ZSCALE ;

     int no       = 30;

     double res   = MAX((maxX-rin)/(double)no,(maxY-zin)/(double)no/ZSCALE);
     res         *= 1.05 ;
     //res          = 1e-1;

     FILE *f = fopen("/home/chris/accel/sml.csv", "w");
     //printf("rin: %.5f   zin %.5f  \n", rin, zin );

     double sx, sy;
     int indx = 0;
     int indy = 0;
     double ff[2];
     fprintf(f,"        ");
     for (sx = rin - res*no, indx = 0; indx < no*2+1 ; sx += res, indx++ )
     {
       fprintf(f,"\t%.6f",sx);
     }
     fprintf(f,"\n");
     for (sy = zin / ZSCALE - res*no; indy < no*2+1; sy += res, indy++ )
     {
       ff[1]=sy;
       fprintf(f,"%.6f",sy*ZSCALE);
       for (sx = rin - res*no, indx = 0; indx < no*2+1 ; sx += res, indx++ )
       {
         ff[0]=sx;
         double yy = -power_call_rz_harmonics(ff);

         fprintf(f,"\t%.6f",yy);

         if (yy > bstGrd[2] )
         {
           bstGrd[2] = yy;
           bstGrd[0] = sx;
           bstGrd[1] = sy;
         }
       }
       fprintf(f,"\n");
     }
     fclose(f);

     FILE *fp = fopen("/home/chris/accel/pnts.csv", "w");
     fprintf(fp,"Entrance\t%.6f\t%.6f\t%.6f\n",lrgPnt[0][0], lrgPnt[0][1],      lrgPnt[0][2] );
     fprintf(fp,"CPU 01  \t%.6f\t%.6f\t%.6f\n",lrgPnt[1][0], lrgPnt[1][1],      lrgPnt[1][2] );
     fprintf(fp,"CPU Ent \t%.6f\t%.6f\t%.6f\n",smlPnt[0][0], smlPnt[0][1],      smlPnt[0][2] );
     fprintf(fp,"CPU FNL \t%.6f\t%.6f\t%.6f\n",smlPnt[1][0], smlPnt[1][1],      smlPnt[1][2] );
     fprintf(fp,"SWARM   \t%.6f\t%.6f\t%.6f\n",bestPos[0],   bestPos[1]*ZSCALE, bestVal );
     fprintf(fp,"GRID    \t%.6f\t%.6f\t%.6f\n",bstGrd[0],    bstGrd[1]*ZSCALE,  bstGrd[2] );
     fclose(fp);


     // Plot combined
     system("python ~/bin/bin/plt.py");
     //system("python ~/bin/bin/pltSwrm2.py");

     double d1 = bstGrd[0] - smlPnt[1][0];
     double d2 = bstGrd[1] - smlPnt[1][1];

     double dst = sqrt(d1*d1 + d2*d2) ;

     if ( bstGrd[2] > smlPnt[1][2] )
     {
       if ( dst < 1e-2 )
       {
         sprintf(dirname,"%s_c1",dirname );
       }
       else
       {
         printf("CPU    (%.6f) missed point with val of %.6f \n", smlPnt[1][2], bstGrd[2] );
         sprintf(dirname,"%s_c0",dirname );

         *rout = bstGrd[0];
         *zout = bstGrd[1]*ZSCALE;
       }
     }
     else
     {
       sprintf(dirname,"%s_c1",dirname );
     }

     d1 = bstGrd[0] - bestPos[0];
     d2 = bstGrd[1] - bestPos[1];
     dst = sqrt(d1*d1 + d2*d2) ;

     if ( bstGrd[2] > bestVal )
     {
       if ( dst < 1e-2 )
       {
         sprintf(dirname,"%s_g1",dirname );
       }
       else
       {
         printf("Swarm  (%.6f) missed point with val of %.6f \n", bestVal, bstGrd[2] );
         sprintf(dirname,"%s_g0",dirname );

         // Plot swarm
         if (swrm)
         {
           //system("python ~/bin/pltSwrm2.py");
         }
       }
     }
     else
     {
       sprintf(dirname,"%s_g1",dirname );
     }

     if (-bestVal < y[0] )
     {
       printf("Better: (%.5f %.5f) %.5f  vs  (%.5f %.5f) %.5f \n", bestPos[0], bestPos[1]*ZSCALE, bestVal, x[0][0], x[0][1]*ZSCALE, -y[0]);
       sprintf(dirname,"%s_Btr",dirname );

       if ( bestVal > bstGrd[2] )
       {
         *rout = bestPos[0];
         *zout = bestPos[1]*ZSCALE;
       }
     }
     else if ( fabs(-bestVal/y[0]-1) < 0.01 )
     {
       printf("Same: (%.5f %.5f) %.10f  vs  (%.5f %.5f) %.10f \n", bestPos[0], bestPos[1]*ZSCALE, bestVal, x[0][0], x[0][1]*ZSCALE, -y[0]);
       sprintf(dirname,"%s_Sme",dirname );
     }
     else
     {
       printf("Worse:  (%.5f %.5f) %.5f  vs  (%.5f %.5f) %.5f \n", bestPos[0], bestPos[1]*ZSCALE, bestVal, x[0][0], x[0][1]*ZSCALE, -y[0]);
       sprintf(dirname,"%s_Wor",dirname );
     }

     int tmp0 = 0;

     if (1)
     {
       mkdir(dirname, 0755);

       char scmd[1024];

       sprintf(scmd,"mv /home/chris/accel/Nelder_Mead/*.png %s/", dirname );
       system(scmd);

       sprintf(scmd,"mv /home/chris/accel/*.csv %s/", dirname );
       system(scmd);

       sprintf(scmd,"cp eliminate_harmonics.log %s/", dirname );
       system(scmd);

       sprintf(scmd,"cp Cands*.csv %s/", dirname );
       system(scmd);
     }

     int tmp2 = 0;

     //printf("exit\n");
     //char d=(char)(7);
     //printf("%c\n",d);
     //exit(1);
   }

   /* The following calculates derivatives at the peak           */

   for (i=1; i<=num_harmonics; i++)
   {
     locpow[i-1] = get_localpower3d(data[i-1], numdata, (r_offset[i-1]+*rout)*i-r_offset[i-1], (*zout)*i, 0.0);
     x[0][0] = (r_offset[i-1]+*rout)*i-r_offset[i-1];
     x[0][1] = *zout/ZSCALE * i;
     maxdata = data[i-1];
     power[i-1] = -power_call_rz(x[0]);
     get_derivs3d(data[i-1], numdata, (r_offset[i-1]+*rout)*i-r_offset[i-1], (*zout)*i, 0.0, locpow[i-1], &(derivs[i-1]));
   }

   //x[0][0] = *rout;
   //x[0][1] = *zout / ZSCALE;
   //printf("Out Power %7.3f\n",-power_call_rz_harmonics(x[0]));

   vect_free(locpow);
   vect_free(maxlocpow);
}

void max_rz_file_harmonics(FILE * fftfile, int num_harmonics,
                             int lobin,
                             double rin, double zin,
                             double *rout, double *zout, rderivs derivs[],
                             double maxpow[], int nn)
/* Return the Fourier frequency and Fourier f-dot that      */
/* maximizes the power of the candidate in 'fftfile'.       */
/* WARNING: not tested */
{
   int i;
   double maxz, rin_int, rin_frac;
   int kern_half_width, filedatalen, extra = 10;
   int* r_offset;
   fcomplex** filedata;

   r_offset = (int*)malloc(sizeof(int)*num_harmonics);
   filedata = (fcomplex**)malloc(sizeof(fcomplex*)*num_harmonics);
   maxz = fabs(zin*num_harmonics) + 4.0;
   kern_half_width = z_resp_halfwidth(maxz, HIGHACC);
   filedatalen = 2 * kern_half_width + extra;

   for (i=1;i<=num_harmonics;i++) {
       rin_frac = modf(rin*i, &rin_int);
       r_offset[i-1] = (int) rin_int - filedatalen / 2 + lobin;
       filedata[i-1] = read_fcomplex_file(fftfile, r_offset[i-1], filedatalen);
   }
   rin_frac = modf(rin, &rin_int);
   max_rz_arr_harmonics(filedata, num_harmonics,
                        r_offset,
                        filedatalen, rin_frac + filedatalen / 2,
                        zin, rout, zout, derivs,
                        maxpow, nn);

   *rout += r_offset[0];
   for (i=1;i<=num_harmonics;i++) {
       vect_free(filedata[i-1]);
   }
   free(r_offset);
   free(filedata);
}
