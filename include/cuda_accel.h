#ifndef CUDA_ACCEL_INCLUDED
#define CUDA_ACCEL_INCLUDED

#include <cuda.h>
#include <cufft.h>

#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>

//#define CBL   // REMOVE

#ifdef __cplusplus
#define ExternC extern "C"
#else
#define ExternC
#endif

#ifdef __cplusplus
extern "C"
{
#endif
#define __float128 long double
#include "accel.h"
#ifdef __cplusplus
}
#endif

//=========================================== Defines ====================================================\\

#define   MAX_IN_STACK  10              ///< NOTE: this is 1 to big to handle the init problem
#define   MAX_HARM_NO   16              ///< The maximum number of harmonics handled by a accel search
#define   MAX_YINDS     8000            ///< The maximum number of y indices to store in constant memory
#define   MAX_STEPS     8               ///< The maximum number of steps
#define   MAX_STKSZ     8               ///< The maximum number of plains in a stack
#define   MAX_GPUS      32              ///< The maximum number GPU's

//======================================== Debug Defines  ================================================\\

#define DBG_KER01       0       ///< Convolution kernel
#define DBG_KER02       0       ///< Convolution kernel post fft
#define DBG_PRNTKER02   0       ///< Convolution kernel post fft

#define DBG_INP01       0       ///< RAW FFT
#define DBG_INP02       0       ///< normalised RAW FFT
#define DBG_INP03       0       ///< spread RAW FFT
#define DBG_INP04       0       ///< FFT'd input data

#define DBG_PLN01       0       ///< Input convolved
#define DBG_PLN02       0       ///< Input convolved & FFTD (Complex plain)
#define DBG_PLN03       0       ///< Summed powers (ie The f-∂f plain)

#define DBG_PLTPLN06    0       ///< Input convolved & FFTD (Complex plain)
#define DBG_PLTDETC     0       ///< Detections


//====================================== Bit flag values =================================================\\

#define     CU_INPT_DEVICE      (1<<0)    ///< Put the entire raw input FFT on device memory - No CPU synchronisation required, but normalisation on GPU!
#define     CU_INPT_HOST        (1<<1)    ///< Use host page locked host data for entire raw FFT data - normalisation on GPU but allows asynchronous copies.
#define     CU_INPT_SINGLE_C    (1<<2)    ///< Prepare input data one step at a time, using CPU - normalisation on CPU - Generally bets option, as CPU is "idle"
#define     CU_INPT_SINGLE_G    (1<<3)    ///< Prepare input data one step at a time, using GPU - normalisation on GPU
#define     CU_INPT_SINGLE      (CU_INPT_SINGLE_G|CU_INPT_SINGLE_C)   ///< Prepare input data one step at a time
#define     CU_INPT_ALL         ( CU_INPT_DEVICE | CU_INPT_HOST | CU_INPT_SINGLE_G | CU_INPT_SINGLE_C  )

#define     CU_OUTP_DEVICE      (1<<4)    ///< Write all candidates to the device memory : This requires a lot of device memory
#define     CU_OUTP_HOST        (1<<5)    ///< Write all candidates to page locked host memory : This requires a lot of memory
#define     CU_OUTP_SINGLE      (1<<6)    ///< Only get candidates from the current plain - This seams to be best in most cases
#define     CU_OUTP_ALL         ( CU_OUTP_DEVICE | CU_OUTP_HOST | CU_OUTP_SINGLE )
#define     FLAG_SAS_SIG        (1<<7)   ///< Do sigma calculations on the GPU - Generally this can be don on the CPU while the GPU works

#define     CU_CAND_LST         (1<<8)    ///< Candidates are stored in a list   (usually a dynamic linked list)
#define     CU_CAND_ARR         (1<<9)    ///< Candidates are stored in an array (requires more memory)

#define     FLAG_CNV_1KER       (1<<12)   ///< Use minimal kernel                  - ie Only the kernel of largest plain in each stack
#define     FLAG_CNV_OVLP       (1<<13)   ///< Use the overlap kernel              - I found this slower that the alternative

#define     FLAG_CNV_PLN        (1<<14)   ///< Convolve one plain at a time
#define     FLAG_CNV_STK        (1<<15)   ///< Convolve one stack at a time        - This seams to be best in most cases
#define     FLAG_CNV_FAM        (1<<16)   ///< Convolve one family at a time       - Preferably don't use this!
#define     FLAG_CNV_ALL        ( FLAG_CNV_PLN | FLAG_CNV_STK | FLAG_CNV_FAM )

#define     FLAG_STP_ROW        (1<<17)   ///< Multi-step Row   interleaved        - This seams to be best in most cases
#define     FLAG_STP_PLN        (1<<18)   ///< Multi-step Plain interleaved        -
#define     FLAG_STP_STK        (1<<19)   ///< Multi-step Stack interleaved        - Preferably don't use this!
#define     FLAG_STP_ALL        ( FLAG_STP_ROW | FLAG_STP_PLN | FLAG_STP_STK )

#define     FLAG_CUFFTCB_INP    (1<<21)   ///< Use an input  callback to do the convolution      - I found this to be very slow
#define     FLAG_CUFFTCB_OUT    (1<<22)   ///< Use an output callback to create powers           - This is a similar speed

#define     FLAG_CNV_TEX        (1<<23)   ///< Use texture memory for convolution  - May give advantage on pre-Fermi generation which we don't really care about
#define     FLAG_PLN_TEX        (1<<24)   ///< Use texture memory to access the d-∂d plains during sum and search ( does not imply interpolation method) - May give advantage on pre-Fermi generation which we don't really care about
#define     FLAG_TEX_INTERP     (1<<25)   ///< Use liner interpolation in with texture memory - This requires - FLAG_CUFFTCB_OUT and FLAG_PLN_TEX

#define     FLAG_RETURN_ALL     (1<<26)   ///< Return results for all stages of summing, default is only the final result
#define     FLAG_STORE_ALL      (1<<28)   ///< Store candidates for all stages of summing, default is only the final result
#define     FLAG_STORE_EXP      (1<<29)   ///< Store expanded candidates

// ----------- This is a list of the data types that can be passed or returned

#define     CU_NONE             (0)       ///< Nothing specified
#define     CU_CMPLXF           (1<<1)    ///< Complex float
#define     CU_INT              (1<<2)    ///< INT
#define     CU_FLOAT            (1<<3)    ///< Float
#define     CU_POWERZ           (1<<4)    ///< A value and a z bin         accelcand2
#define     CU_SMALCAND         (1<<5)    ///< A compressed candidate      accelcandBasic
#define     CU_FULLCAND         (1<<6)    ///< Full detailed candidate     cand
#define     CU_GSList           (1<<7)    ///


//========================================== Macros ======================================================\\

///< Defines for safe calling usable in C
#define CUDA_SAFE_CALL(value, errorMsg)     __cuSafeCall   (value, __FILE__, __LINE__, errorMsg )
#define CUFFT_SAFE_CALL(value,  errorMsg)   __cufftSafeCall(value, __FILE__, __LINE__, errorMsg )


//======================================== Type defines ==================================================\\

///< A complex float in device texture memory
typedef cudaTextureObject_t fCplxTex;

///< A complex number data type
typedef struct fcomplexcu
{
    float r, i;
} fcomplexcu;

///< Basic accel search candidate to be used in CUDA kernels
///< Note this may not be the best choice on a GPU as it has a bad size
typedef struct accelcand2
{
  float value;        // This cab be Sigma or summed power
  short z;            // Fourier f-dot of first harmonic
} accelcand2;

///< Basic accel search candidate to be used in CUDA kernels
typedef struct accelcandBasic
{
  float sigma;        // Sigma - adjusted for number of trials, NOTE: at some points this value holds the sum of powers from which the sigma value will be calculated
  short numharm;      // Number of harmonics summed
  short z;            // Fourier f-dot of first harmonic
} accelcandBasic;

///< Accel search candidate (this holds more info and is thus larger than accelcandBasic
typedef struct cand
{
    float   power;
    double  r;
    double  sig;
    float   z;
    int     numharm;
} cand;

/** A data structure to pass to CUFFT call-back load functions
 */
typedef struct fftCnvlvInfo
{
    int     stride;
    int     width;
    int     noSteps;
    int     noPlains;
    float*  d_plainPowers;

    int heights[MAX_STKSZ];
    int top[MAX_STKSZ];
    fcomplexcu* d_idata [MAX_STKSZ];
    fcomplexcu* d_kernel[MAX_STKSZ];

} fftCnvlvInfo;

/** A structure to hold information on a raw fft
 */
typedef struct fftInfo
{
    double      rlo;      ///< The Low bin   (of interest)
    double      rhi;      ///< The high bin  (of interest)
    int         nor;      ///< The number of bins in the FFT
    fcomplex*   fft;      ///< An array of complex numbers (nor long)
} fftInfo;


//------------- Data structures for, plains, stacks, batches etc ----------------\\

/** Details of the number of bins of the full search
 */
typedef struct searchScale
{
    double searchRLow;                ///< The value of the input r bin to start the search at
    double searchRHigh;               ///< The value of the input r bin to end   the search at

    long long rLow;                   ///< The lowest  possible R this search could find, Including halfwidth, thus may be less than 0
    long long rHigh;                  ///< The highest possible R this search could find, Including halfwidth

    unsigned long long noInpR;        ///< The maximum number of r input ( this is essentially  (rHigh - rLow) ) and me be longer than fft length because of halfwidth this requires the FFT to be padded!
    unsigned long long noOutpR;       ///< The maximum number of r bins the fundamental search will produce. This is ( searchRHigh - searchRLow ) / ( candidate resolution ) It may need to be scaled by numharmstages
} searchScale;

/** Details of the number of bins of the full search
 */
typedef struct rVals
{
    double      drlo;     ///< The R value of the first usable bin of the plain
    long long   lobin;    ///< The first bin to copy from the the input fft ( serachR scaled - halfwidth )
    long        numdata;  ///< The number of input fft points to read
    long        numrs;    ///< The number of good bins in the plain ( expanded units )
    long long   expBin;   ///< The index of the expanded bin of the first good value
} rVals;

//typedef rVals[MAX_STEPS][MAX_HARM_NO] rVlalsList;

/** User specified search details
 *
 */
typedef struct searchSpecs
{
//    int     noHarms;                  ///< The number of harmonics in the family                 m
    int     noHarmStages;             ///< The number of stages of harmonic summing              m

    int     zMax;                     ///< The highest z drift of the fundamental
    int     pWidth;                   ///< The desired width of the plains
    float   sigma;                    ///< The cut off sigma
    fftInfo fftInf;                   ///< The details of the input fft - location size and area to search

    int     flags;                    ///< The search flags
    int     normType;                 ///< The type of normalisation to do

    int     outType;                  ///< The type of output                                    m
    void*   outData;                  ///< A pointer to the location to store candidates
} searchSpecs;

/** User specified GPU search details
 */
typedef struct gpuSpecs
{
    int     noDevices;                ///< The number of devices (GPU's to use in the search)
    int     devId[MAX_GPUS];          ///< A list noDevices long of CUDA GPU device id's
    int     noDevBatches[MAX_GPUS];   ///< A list noDevices long of the number of batches on each device
    int     noDevSteps[MAX_GPUS];     ///< A list noDevices long of the number of steps each device wants to use
} gpuSpecs;

/** The size information of a f-∂f plain
 */
typedef struct cuHarmInfo
{
    size_t  height;                   ///< The number if rows (Z's)
    size_t  width;                    ///< The number of complex numbers in each kernel (2 floats)

    int     halfWidth;                ///< The kernel half width         - in input fft units ie needs to be multiply by ACCEL_RDR to get plain units
    //int     numrs;                    ///< The number of usable values   - in plain units NB: This number is actually dynamic this is just an upper bound rather use (numrs from cuFFdot)

    size_t  inpStride;                ///< The x stride in complex numbers

    int     zmax;                     ///< The maximum (and minimum) z
    float   harmFrac;                 ///< The harmonic fraction
    int     stackNo;                  ///< Which Stack is the plain in. (0 indexed at starting at the widest stack)

    int     yInds;                    ///< The offset of the y offset in constant memory
    int     stageOrder;               ///< The index of this harmonic in the staged order
} cuHarmInfo;

/** The complex convolution kernels of a f-∂f plain
 */
typedef struct cuKernel
{
    cuHarmInfo* harmInf;              ///< A pointer to the harmonic information for this kernel
    fcomplexcu* d_kerData;            ///< A pointer to the first kernel element (Width, Stride and height determined by harmInf)
    fCplxTex    kerDatTex;            ///< A texture holding the kernel data
} cuKernel;

/** A f-∂f plain  .
 * This could be a fundamental or harmonic
 * it holds basic information no memory addresses
 */
typedef struct cuFFdot
{
    cuHarmInfo* harmInf;              ///< A pointer to the harmonic information for this plains
    cuKernel*   kernel;               ///< A pointer to the kernel for this plain

    // pointers to device data
    fcomplexcu* d_plainData;          ///< A pointer to the first element of the complex f-∂f plain (Width, Stride and height determined by harmInf)
    float*      d_plainPowers;        ///< A pointer to the powers for this stack
    fcomplexcu* d_iData;              ///< A pointer to the input data for this plain this is a section of the 'raw' complex fft data, that has been Normalised, spread and FFT'd

    // Texture objects
    fCplxTex    datTex;               ///< A texture holding the kernel data
    fCplxTex    powerTex;             ///< A texture of the power data

    //size_t  numInpData[MAX_STEPS];    ///< The number of input elements for this plain                 - (Number of R bins in the 'raw' FFT input, including halfwidth)
    //size_t  numrs[MAX_STEPS];         ///< The number of good bins in the plain in expanded units -        (Number of R bins in the 'raw' FFT input, excluding halfwidth)

    //float   fullRLow[MAX_STEPS];      ///< The low r bin of the input data used ( Note: the 0 index is [floor(rLow) - halfwidth * DR] )
    //float   rLow[MAX_STEPS];          ///< The r value of good bin - input fft units
    //float   searchRlow[MAX_STEPS];    ///< The low r bin of the input data used ( Note: the 0 index is [floor(rLow) - halfwidth * DR] )
    //int     ffdotPowWidth[MAX_STEPS]; ///< The width of the final f-∂f plain  // TODO: Check is this not the same as numrs

    //float searchRlowPrev[MAX_STEPS];  ///< The low r bin of the input data used ( Note: the 0 index is [floor(rLow) - halfwidth * DR] )
} cuFFdot;

/** A stack of f-∂f plains that all have the same FFT width
 */
typedef struct cuFfdotStack
{
    int     noInStack;                ///< The number of plains in this stack
    int     startIdx;                 ///< The 'global' offset of the first element of the stack
    size_t  width;                    ///< The width of  the entire stack, for one step [ in complex numbers! ]
    size_t  height;                   ///< The height of the entire stack, for one step
    size_t  inpStride;                ///< The stride of the block of memory  [ in complex numbers! ]
    size_t  pwrStride;                ///< The stride of the block of memory  [ in complex numbers! ]

    // Sub data structures associated with this stack
    cuHarmInfo* harmInf;              ///< A pointer to all the harmonic info's for this stack
    cuKernel*   kernels;              ///< A pointer to all the kernels for this stack
    cuFFdot*    plains;               ///< A pointer to all the pains for this stack

    int startZ[MAX_IN_STACK];         ///< The y 'start' of the plains in this stack - assuming one step

    // All sub-kernel of the same width are subset of the largest harmonic
    // Thus all plains in a stack can share one kernel these sub kernels are bounded by zDn and zUp
    int zUp[MAX_IN_STACK];            ///< The upper bound  (y or z) of the plains in the single kernel
    int zDn[MAX_IN_STACK];            ///< The lower bound  (y or z) of the plains in the single kernel

    // CUFFT details
    cufftHandle   plnPlan;            ///< A cufft plan to fft the entire stack
    cufftHandle   inpPlan;            ///< A cufft plan to fft the input data for this stack
    fftCnvlvInfo* d_cinf;             ///< Convolve info structure on device

    // pointers to memory
    fcomplexcu* d_kerData;            ///< Kernel data for this stack
    fcomplexcu* d_plainData;          ///< Plain data for this stack
    float*      d_plainPowers;        ///< Powers for this stack
    fcomplexcu* d_iData;              ///< Input data for this stack
    fcomplexcu* h_iData;              ///< Paged locked input data for this stack

    // Texture
    fCplxTex kerDatTex;               ///< A texture holding the kernel data

    // Events
    cudaEvent_t prepComp;             ///< Preparation of the input data complete
    cudaEvent_t convComp;             ///< Convolution complete
    cudaEvent_t plnComp;              ///< Creation (convolution and FFT) of the complex plain complete

    // Streams
    cudaStream_t fftPStream;          ///< CUDA stream for summing and searching the data
    cudaStream_t fftIStream;          ///< CUDA stream for summing and searching the data
    cudaStream_t cnvlStream;          ///< CUDA stream for work on the stack
    cudaStream_t inpStream;           ///< CUDA stream for work on input data for the stack
} cuFfdotStack;

/** A collection of f-∂f plain(s) and all its/their sub harmonics
 * This is a collection of stack(s) that make up a harmonic family of f-∂f plain(s)
 * And the device specific convolution kernels which is just another batch
 */
typedef struct cuFFdotBatch
{
    searchSpecs* sInf;                ///< A pointer to the search info

    size_t noStacks;                  ///< The number of stacks in this batch
    size_t noHarms;                   ///< The number of harmonics in the family                   m
    size_t noSteps;                   ///< The number of slices in the batch
    int noHarmStages;                 ///< The number of stages of harmonic summing                m

    int pIdx[MAX_HARM_NO];            ///< The index of the plains in the Presto harmonic summing order  m

    cuFfdotStack* stacks;             ///< A list of the stacks
    cuHarmInfo*   hInfos;             ///< A list of the harmonic informations
    cuKernel*     kernels;            ///< A list of the kernels
    cuFFdot*      plains;             ///< A list of the plains

    //cHarmList iDataLst;               ///< A list of the input data allocated in memory
    //iHarmList iDataLens;              ///< A list of the input data allocated in memory

    int inpDataSize;                  ///< The size of the input data memory in bytes for one step
    int plnDataSize;                  ///< The size of the complex plain data memory in bytes for one step
    int pwrDataSize;                  ///< The size of the powers  plain data memory in bytes for one step
    int kerDataSize;                  ///< The size of the plain data memory in bytes for one step
    int retDataSize;                  ///< The size of data to return in bytes for one step

    fcomplexcu* d_kerData;            ///< Kernel data for all the stacks
    fcomplexcu* d_plainData;          ///< Plain data for all the stacks
    float*      d_plainPowers;        ///< Powers for all the stack

    int retType;                      ///< The type of output                                    m
    int cndType;                      ///< The type of output                                    m
    void* h_retData;                  ///< The output
    void* d_retData;                  ///< The output
    void* h_candidates;               ///< Host memory for candidates
    void* d_candidates;               ///< Host memory for candidates

    fcomplexcu* h_iData;              ///< Pointer to page locked host memory of Input data for t
    fcomplexcu* d_iData;              ///< Input data for the batch - NB: This could be a contiguous block of sections or all the input data depending on inpMethoud

    int haveInput;                    ///< Weather the the plain has input ready to convolve
    int haveSearchResults;            ///< Weather the the plain has been searched and there is candidate data to process
    int haveConvData;                 ///< Weather the the plain has convolved data ready for searching

    uint* d_candSem;                  ///< Semaphore for writing to device candidate list

    float* h_powers;                  ///< Powers used for running double-tophat local-power normalisation

    uint flag;                        ///< CUDA accel search flags

    searchScale*  SrchSz;             ///< Details on o the size (in bins) of the search         m

    cudaStream_t inpStream;           ///< CUDA stream for work on input data for the batch
    cudaStream_t strmSearch;          ///< CUDA stream for summing and searching the data

    cudaEvent_t iDataCpyComp;         ///< Copying input data to device
    cudaEvent_t candCpyComp;          ///< Finished reading candidates from the device
    cudaEvent_t normComp;             ///< Normalise and spread input data
    cudaEvent_t searchComp;           ///< Sum & Search complete (candidates ready for reading)
    cudaEvent_t processComp;          ///< Process candidates (usually done on CPU)

    int noResults;                    ///< The number of results from the previous search

    uint accelLen;                    ///< The size to step through the input fft

    rVals*** rInput;                  ///< Pointer to a 2D array [step][harmonic] of the base expanded r index
    rVals*** rConvld;                 ///< Pointer to a 2D array [step][harmonic] of the base expanded r index
    rVals*** rSearch;                 ///< Pointer to a 2D array [step][harmonic] of the base expanded r index

    CUcontext pctx;                   ///< Context for the batch
    int device;                       ///< The CUDA device to run on;

} cuFFdotBatch;

/** A struct to keep info on all the kernels and batches to use with cuda accel
 */
typedef struct cuMemInfo
{
    //int*            devId;              ///< A list noDevices long of CUDA GPU device id's
    //int*            noDevBatches;       ///< A list noDevices long of the number of batches on each device
    //int*            noDevSteps;         ///< A list noDevices long of the number of steps each device wants to use

    int             noDevices;          ///< The number of devices (GPU's to use in the search)
    cuFFdotBatch*   kernels;            ///< A list noDevices long of convolution kernels: These hold: basic info, the address of the convolution kernels on the GPU, the CUFFT plan.

    int             noBatches;          ///< The total number of batches there across all devices
    cuFFdotBatch*   batches;            ///< A list noBatches long of convolution kernels: These hold: basic info, the address of the convolution kernels on the GPU, the CUFFT plan.

    int             noSteps;            ///< The total steps there are across all devices
} cuMemInfo;

/** User independent details  .
 */
typedef struct cuSearch
{
    searchSpecs*  sSpec;              ///< Specifications of the search
    gpuSpecs*     gSpec;              ///< Specifications of the GPU's to use

    cuMemInfo*    mInf;               ///< The allocated Device and host memory and data structures to create plains including the kernels

    // Some extra search details
    int           noHarms;            ///< The number of harmonics in the family                 m
    int           noHarmStages;       ///< The number of stages of harmonic summing              m
    int           numZ;               ///< The number of Z values
    searchScale*  SrchSz;             ///< Details on o the size (in bins) of the search         m
    int*          pIdx;               ///< The index of the plains in the Presto harmonic summing order  m
    float*        powerCut;           ///< The power cutoff
    long long*    numindep;           ///< The number of independent trials
} cuSearch;


//===================================== Function prototypes ===============================================\\

/** Read the GPU details from clig command line
 *
 * @param cmd
 * @return A pointer to the accel info struct to fill
 */
ExternC gpuSpecs readGPUcmd(Cmdline *cmd);

/** Read the GPU details from clig command line
 *
 * @param cmd
 * @return A pointer to the accel info struct to fill
 */
ExternC searchSpecs readSrchSpecs(Cmdline *cmd, accelobs* obs);

ExternC cuSearch* initCuSearch(searchSpecs* sSpec, gpuSpecs* gSpec, cuSearch* srch);

/**
 *
 * @param aInf            A pointer to the accelInfo structure to write details to
 * @param fftinf          Details on the fft to 'search'
 * @param numharmstages   The number of harmonic stages [number of harmonics = 2^(numharmstages-1) = (1<<(numharmstages-1)) ]
 * @param zMax            The maximum and minimum drift to cover in a plain
 * @param width           The width of the plains
 * @param powcut
 * @param numindep
 * @param flags           The a combination of the bit flags to determine how to generate the plains
 * @param candType        The data type of the candidates
 * @param retType         The data type returned from the GPU
 * @param out             An address of to store candidates in if it has already been allocated (can be NULL)
 * @return                Returns the number of steps covered. Main details are written to aInf
 */
//ExternC int initCuAccel(cuMemInfo* aInf, fftInfo* fftinf, int numharmstages, int zMax, int width, float*  powcut, long long*  numindep, int flags, int candType, int retType, void* out);

/** Initialise the template structure and kernels for a multi-step batches
 * This is called once per device
 *
 * @param stkLst            The data structure to fill
 * @param master            The master data structure to copy kernel and some settings from, if this is the first call this should be NULL
 * @param numharmstages     The number of harmonic stages
 * @param zmax              The ZMax of the primary harmonic
 * @param fftinf            The address and accel search info
 * @param device            The device to create the kernels on
 * @param noBatches         The desired number of batches to run on this device
 * @param noSteps           The number of steps for each batch to use
 * @param width             The desired width of the primary harmonic in thousands
 * @param powcut            The value above which to return
 * @param numindep          The number of independent trials
 *
 * @return The number batches set up for this should be noBatches. On failure returns 0
 */
//ExternC int initHarmonics(cuFFdotBatch* stkLst, cuFFdotBatch* master, int numharmstages, int zmax, fftInfo fftinf, int device, int noBatches, int noSteps, int width, float*  powcut, long long*  numindep, int flags, int candType, int retType, void* out);

/** Free all host and device memory allocated by initHarmonics(...)
 * If the stkLst is master, this will free any device independat memory
 *
 * @param stkLst            The data structure to free
 * @param master            The master stak list
 * @param out               The candidate output, if none was specified this should be NULL
 */
//ExternC  void freeHarmonics(cuFFdotBatch* stkLst, cuFFdotBatch* master, void* out);

/** Initialise a multi-step batch from the device kernel
 *
 * @param harms             The kernel to base this multi-step batch
 * @param no                The index of this batch
 * @param of                The desired number of batches on this device
 * @return
 */
//ExternC cuFFdotBatch* initBatch(cuFFdotBatch* harms, int no, int of);

/** Free device and host memory allocated by initStkList
 *
 * @param harms             The batch to free
 */
//ExternC void freeBatch(cuFFdotBatch* stkLst);

ExternC void search_ffdot_planeCU(cuFFdotBatch* plains, double* searchRLow, double* searchRHi, int norm_type, int search, fcomplexcu* fft, long long* numindep, GSList** cands);

ExternC void setStkPointers(cuFFdotBatch* stkLst);

ExternC void setPlainPointers(cuFFdotBatch* stkLst);

ExternC void accelMax(fcomplex* fft, long long noBins, long long startBin, long long endBin, short zMax, short numharmstages, float* powers );

#endif // CUDA_ACCEL_INCLUDED
