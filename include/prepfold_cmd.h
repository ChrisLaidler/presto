#ifndef __prepfold_cmd__
#define __prepfold_cmd__
/*****
  command line parser interface 
                        -- generated by clig (Version: 1.1.3)

  The command line parser `clig':
  (C) 1995 Harald Kirsch (kir@iitb.fhg.de)
*****/

#ifndef FALSE
#  define FALSE (1==0)
#  define TRUE  (!FALSE)
#endif

typedef struct s_Cmdline {
  /***** -pkmb: Raw data in Parkes Multibeam format */
  char pkmbP;
  /***** -ebpp: Raw data in Effelsberg-Berkeley Pulsar Processor format.  CURRENTLY UNSUPPORTED */
  char ebppP;
  /***** -nobary: Do not barycenter (assume input parameters are topocentric) */
  char nobaryP;
  /***** -DE405: Use the DE405 ephemeris for barycentering instead of DE200 (the default) */
  char de405P;
  /***** -dm: The central DM of the search (cm^-3 pc) */
  char dmP;
  double dm;
  int dmC;
  /***** -nsub: The number of sub-bands to use for the DM search */
  char nsubP;
  int nsub;
  int nsubC;
  /***** -npart: The number of sub-integrations to use for the period search */
  char npartP;
  int npart;
  int npartC;
  /***** -p: The nominative folding period (s) */
  char pP;
  double p;
  int pC;
  /***** -pd: The nominative period derivative (s/s) */
  char pdP;
  double pd;
  int pdC;
  /***** -pdd: The nominative period 2nd derivative (s/s^2) */
  char pddP;
  double pdd;
  int pddC;
  /***** -f: The nominative folding frequency (hz) */
  char fP;
  double f;
  int fC;
  /***** -fd: The nominative frequency derivative (hz/s) */
  char fdP;
  double fd;
  int fdC;
  /***** -fdd: The nominative frequency 2nd derivative (hz/s^2) */
  char fddP;
  double fdd;
  int fddC;
  /***** -phs: Offset phase for the profil */
  char phsP;
  double phs;
  int phsC;
  /***** -start: The folding start time as a fraction of the full obs */
  char startTP;
  double startT;
  int startTC;
  /***** -end: The folding end time as a fraction of the full obs */
  char endTP;
  double endT;
  int endTC;
  /***** -n: The number of bins in the profile.  Defaults to the number of sampling bins which correspond to one folded period */
  char proflenP;
  int proflen;
  int proflenC;
  /***** -psr: Name of pulsar to fold (do not include J or B) */
  char psrnameP;
  char* psrname;
  int psrnameC;
  /***** -rzwcand: The candidate number to fold from 'infile'_rzw.cand */
  char rzwcandP;
  int rzwcand;
  int rzwcandC;
  /***** -rzwfile: Name of the rzw search file to use (include the full name of the file) */
  char rzwfileP;
  char* rzwfile;
  int rzwfileC;
  /***** -bin: Fold a binary pulsar.  Must include all of the following parameters */
  char binaryP;
  /***** -pb: The orbital period (s) */
  char pbP;
  double pb;
  int pbC;
  /***** -x: The projected orbital semi-major axis (lt-sec) */
  char asinicP;
  double asinic;
  int asinicC;
  /***** -e: The orbital eccentricity */
  char eP;
  double e;
  int eC;
  /***** -To: The time of periastron passage (MJD) */
  char ToP;
  double To;
  int ToC;
  /***** -w: Longitude of periastron (deg) */
  char wP;
  double w;
  int wC;
  /***** -wdot: Rate of advance of periastron (deg/yr) */
  char wdotP;
  double wdot;
  int wdotC;
  /***** uninterpreted command line parameters */
  int argc;
  /*@null*/char **argv;
  /***** the whole command line concatenated */
  char *full_cmd_line;
} Cmdline;


extern char *Program;
extern void usage(void);
extern /*@shared*/Cmdline *parseCmdline(int argc, char **argv);

extern void showOptionValues(void);

#endif

