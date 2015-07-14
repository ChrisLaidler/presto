/* Note:  This version of the Downhill Simplex Algorithm is     */
/*        a modified version on the NR in C 2ed version         */
/*        by Press et el.  Their copyright statement is below.  */
/*        Modification by Scott M. Ransom, 2 March 2001         */
/* (C) Copr. 1986-92 Numerical Recipes Software 3#1y-i.31-.     */

#include <math.h>
#include <stdio.h>

#ifndef SWAP
/* Swaps two variables of undetermined type */
#define SWAP(a,b) tempzz=(a);(a)=(b);(b)=tempzz;
#endif

//static double amotry(double p[3][2], double *y, double *psum, double (*funk) (double[]), int ihi, double fac, FILE *f);
static double amotry(double p[3][2], double *y, double *psum, double (*funk) (double[]), int ihi, double fac);

void amoeba(double p[3][2], double *y, double ftol,
    double (*funk) (double[]), int *nfunk)
{
  int ii, ihi, ilo, inhi;
  double rtol, ysave, ytry, tempzz;
  double psum[2]; // centroid position

  //printf("\namoeba\n");
  //printf("amoeba\n");

  int noEvals = 3;

  *nfunk = 0;
  psum[0] = p[0][0] + p[1][0] + p[2][0];
  psum[1] = p[0][1] + p[1][1] + p[2][1];

  //char tmpstr[1024];
  //sprintf(tmpstr,"/home/chris/accel/amoeba_%.12f.csv",ftol);
  //FILE *f = fopen(tmpstr, "w");

  //fprintf(f,"%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n", p[0][0], p[0][1]*4.0, p[1][0], p[1][1]*4.0, p[2][0], p[2][1]*4.0 );

  for (;;)
  {
    //printf("ITR\n");

    if(1) // Order  .
    {
      ilo = 0;
      ihi = y[0] > y[1] ? (inhi = 1, 0) : (inhi = 0, 1) ;
      for (ii = 0; ii <= 2; ii++)
      {
        if (y[ii] <= y[ilo])
        {
          ilo = ii;
        }
        if (y[ii] > y[ihi])
        {
          inhi = ihi;
          ihi = ii;
        }
        else if (y[ii] > y[inhi] && ii != ihi)
        {
          inhi = ii;
        }
      }
      //fprintf(f,"%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n", p[0][0], p[0][1]*4.0, p[1][0], p[1][1]*4.0, p[2][0], p[2][1]*4.0 );
    }

    if(1) // Break conditions  .
    {
      // Compute the fractional range from highest to lowest and return if satisfactory  .
      rtol = 2.0 * fabs(y[ihi] - y[ilo]) / (fabs(y[ihi]) + fabs(y[ilo]) + 1.0e-15) ;
      if (rtol < ftol)  // Within error so leave  .
      {
        // If returning, put best point and value in slot 0.
        SWAP(y[0], y[ilo])
        SWAP(p[0][0], p[ilo][0])
        SWAP(p[0][1], p[ilo][1])
        break;
      }

      //if (*nfunk >= 5000) // Break condition  .
      if (*nfunk >= 200) // Break condition  .
      {
        /*
            printf("\nWarning:  amoeba() exceeded %d iterations for r=%f  z=%f.\n",
         *nfunk, p[0][0], p[0][1]);
         */
        return;
      }
    }

    if(1) //  Begin a new iteration  .
    {
      *nfunk += 2;

      // First extrapolate by a factor -1 through the face of the simplex across from the high point, i.e., reflect the simplex from the high point.
      //printf("  Reflection\n");
      ytry = amotry(p, y, psum, funk, ihi, -1.0/*, f*/); // Calculate centroid value  .
      noEvals++;

      if (ytry <= y[ilo])
      {
        // Gave a result better than the best point, so try an additional extrapolation by a factor 2.
        //printf("  Expansion\n");
        ytry = amotry(p, y, psum, funk, ihi, 2.0/*, f*/);
        noEvals++;
      }
      else if (ytry >= y[inhi])
      {
        // The reflected point is worse than the second-highest, so look for an intermediate lower point, i.e., do a one-dimensional contraction.
        ysave = y[ihi];
        //printf("  Contraction\n");
        ytry = amotry(p, y, psum, funk, ihi, 0.5/*, f*/);
        noEvals++;

        if (ytry >= ysave)
        {
          // Can’t seem to get rid of that high point.
          //printf("  Shrink 2\n");
          for (ii = 0; ii <= 2; ii++)
          {
            // Better contract around the lowest
            if (ii != ilo)
            {
              // (best) point.
              p[ii][0] = psum[0] = 0.5 * (p[ii][0] + p[ilo][0]);
              p[ii][1] = psum[1] = 0.5 * (p[ii][1] + p[ilo][1]);
              y[ii] = (*funk) (psum);
              noEvals++;
            }
          }
          *nfunk += 2; // Keep track of function evaluations.
          // Recompute psum
          psum[0] = p[0][0] + p[1][0] + p[2][0];
          psum[1] = p[0][1] + p[1][1] + p[2][1];
        }
      }
      else
      {
        // Correct the evaluation count
        --(*nfunk);
      }
      // Go back for the test of doneness and the next
    }
  }

  //fclose(f);

  //printf("amoeba %i evals;\n", noEvals);
}

/** Extrapolates a point through a face
 *
 * Extrapolates by a factor fac through the face of the simplex across from
 * the high point, tries it, and replaces the high point if the new point is better.
 *
 * @param p
 * @param y
 * @param psum
 * @param funk
 * @param ihi
 * @param fac
 * @return
 */
static double amotry(double p[3][2], double *y, double *psum,
    double (*funk) (double[]), int ihi, double fac /*, FILE *f */ )
{
  double fac1, fac2, ytry, ptry[2];

  // Calculate the trial point
  fac1 = 0.5 * (1.0 - fac);
  fac2 = fac1 - fac;
  ptry[0] = psum[0] * fac1 - p[ihi][0] * fac2;
  ptry[1] = psum[1] * fac1 - p[ihi][1] * fac2;

  // Evaluate the function at the trial point
  ytry = (*funk) (ptry);

  if (ytry < y[ihi])
  {
    // If it’s better than the highest, then replace the highest
    y[ihi] = ytry;
    psum[0] += ptry[0] - p[ihi][0];
    p[ihi][0] = ptry[0];
    psum[1] += ptry[1] - p[ihi][1];
    p[ihi][1] = ptry[1];

    //fprintf(f,"%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n", p[0][0], p[0][1]*4.0, p[1][0], p[1][1]*4.0, p[2][0], p[2][1]*4.0 );
  }

  return ytry;
}
