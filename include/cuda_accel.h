#ifndef CUDA_ACCEL_INCLUDED
#define CUDA_ACCEL_INCLUDED

#include <pthread.h>
#include <semaphore.h>

#include <cuda.h>
#include <cufft.h>
#include <cufftXt.h>

#if CUDA_VERSION >= 7050   // Half precision
#include <cuda_fp16.h>
#endif



#ifdef WITHOMP
#include <omp.h>
#endif

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


#undef  NVVP
//#define NVVP                // Uncomment to allow CUDA profiling

#undef  CUDA_PROF
#define CUDA_PROF

#ifdef CUDA_PROF
#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>
#endif

//=========================================== Defines ====================================================

#define         MAX_IN_STACK            10      ///< NOTE: this is 1 to big to handle the init problem
#define         MAX_STACKS              5       ///< The maximum number stacks in a family of plains
#define         MAX_HARM_NO             16      ///< The maximum number of harmonics handled by a accel search
#define         MAX_NO_STAGES           5       ///< The maximum number of harmonics handled by a accel search
#define         MAX_YINDS               8000    ///< The maximum number of y indices to store in constant memory
#define         INDS_BUFF               20      ///< The maximum number of y indices to store in constant memory
#define         MAX_STEPS               8       ///< The maximum number of steps in a single batch
#define         MAX_BATCHES             5       ///< The maximum number of batches on a single GPU
#define         MAX_GPUS                32      ///< The maximum number GPU's
#define         INMEM_FFT_WIDTH         4096    ///< The size of FFT planes for in-mem GPU search
#define         CORRECT_MULT            1       ///< Generate the kernel values the correct way and do the
#define         NO_OPT_LEVS             6       ///< The number of optimisation planes/steps

//====================================== Bit flag values =================================================

#define     FLAG_ITLV_ROW       (1ULL<<0)       ///< Multi-step Row   interleaved        - This seams to be best in most cases

#define     FLAG_KER_HIGH       (1ULL<<1)       ///< Use increased response function width for higher accuracy at Z close to zero
#define     FLAG_KER_MAX        (1ULL<<2)       ///< Use maximum   response function width for higher accuracy at Z close to zero
#define     FLAG_CENTER         (1ULL<<3)       ///< Centre and align the usable part of the convolution kernel
#define     FLAG_KER_DOUBGEN    (1ULL<<4)       ///< Create kernel with double precision calculations
#define     FLAG_KER_DOUBFFT    (1ULL<<5)       ///< Create kernel with double precision calculations and FFT's

#define     CU_NORM_CPU         (1ULL<<6)       ///< Prepare input data one step at a time, using CPU - normalisation on CPU - Generally bets option, as CPU is "idle"
#define     CU_NORM_EQUIV       (1ULL<<7)       ///< Do the normalisation the CPU way

#define     CU_INPT_FFT_CPU     (1ULL<<8)       ///< Do the FFT on the CPU

#define     CU_FFT_SEP          (1ULL<<9)       ///< Use a separate FFT plan for each batch

#define     FLAG_MUL_00         (1ULL<<10)      ///< Multiply kernel (Base only do memory reads and writes - NB This does not do the actual multiplication)
#define     FLAG_MUL_11         (1ULL<<11)      ///< Multiply kernel - Do the multiplication one plane ant a time
#define     FLAG_MUL_21         (1ULL<<12)      ///< Multiply kernel - read all input - loop over kernel - loop over planes
#define     FLAG_MUL_22         (1ULL<<13)      ///< Multiply kernel - Loop ( Plane - Y )
#define     FLAG_MUL_23         (1ULL<<14)      ///< Multiply kernel - Loop ( chunk (read ker) - plan - Y - step )
#define     FLAG_MUL_30         (1ULL<<15)      ///< Multiply kernel - Do an entire batch in one kernel
#define     FLAG_MUL_CB         (1ULL<<16)      ///< Multiply kernel - Using a CUFFT callback
#define     FLAG_MUL_PLN        ( FLAG_MUL_11 )
#define     FLAG_MUL_STK        ( FLAG_MUL_00 | FLAG_MUL_21 | FLAG_MUL_22 | FLAG_MUL_23 | FLAG_MUL_CB )
#define     FLAG_MUL_BATCH      ( FLAG_MUL_30 )
#define     FLAG_MUL_ALL        ( FLAG_MUL_BATCH | FLAG_MUL_STK | FLAG_MUL_PLN )

#define     FLAG_TEX_MUL        (1ULL<<17)      ///< [ Deprecated ]  Use texture memory for multiplication                              - May give some advantage on pre-Fermi generation which we don't really care about

#define     FLAG_CUFFT_CB_POW   (1ULL<<18)      ///< Use an output callback to create powers, this works in std or in-mem searches       - This is a similar iFFT speed but speeds up SS
#define     FLAG_CUFFT_CB_INMEM (1ULL<<19)      ///< Use the in-mem FFT's to copy values strait back to in-mem plane
#define     FLAG_CUFFT_CB_OUT   ( FLAG_CUFFT_CB_POW | FLAG_CUFFT_CB_INMEM )   /// All output callbacks
#define     FLAG_CUFFT_ALL      ( FLAG_CUFFT_CB_OUT | FLAG_MUL_CB )           /// All callbacks

#define     FLAG_SAS_TEX        (1ULL<<20)      ///< Use texture memory to access the d-∂d planes during sum and search ( does not imply interpolation method) - May give advantage on pre-Fermi generation which we don't really care about
#define     FLAG_TEX_INTERP     (1ULL<<21)      ///< Use liner interpolation in with texture memory - This requires - FLAG_CUFFT_CB_OUT and FLAG_SAS_TEX
#define     FLAG_SIG_GPU        (1ULL<<22)      ///< Do sigma calculations on the GPU - Generally this can be don on the CPU while the GPU works

#define     FLAG_SS_CPU         (1ULL<<23)      ///< Do the sum and searching on the CPU, this is now deprecated cos its so slow!
#define     FLAG_SS_00          (1ULL<<24)      ///< This is a debug kernel used as a comparison, it is close to numerically and optimal but gives the worn values
#define     FLAG_SS_10          (1ULL<<25)      ///< This is the standard sum and search kernel, there were others but they were deprecated
#define     FLAG_SS_INMEM       (1ULL<<26)      ///< Do an in memory GPU search
#define     FLAG_SS_STG         ( FLAG_SS_00  | FLAG_SS_10 /* | FLAG_SS_20 | FLAG_SS_30 */ )
#define     FLAG_SS_KERS        ( FLAG_SS_STG | FLAG_SS_INMEM  )
#define     FLAG_SS_ALL         ( FLAG_SS_CPU | (FLAG_SS_KERS) )

#define     FLAG_DOUBLE         (1ULL<<27)      ///< Use double precision kernels and complex plane and iFFT's
#define     FLAG_HALF           (1ULL<<28)      ///< Use half precision when doing a INMEM search

#define     FLAG_RET_STAGES     (1ULL<<29)      ///< Return results for all stages of summing, default is only the final result
#define     FLAG_STORE_ALL      (1ULL<<30)      ///< Store candidates for all stages of summing, default is only the final result
#define     FLAG_STORE_EXP      (1ULL<<31)      ///< Store expanded candidates

#define     FLAG_THREAD         (1ULL<<32)      ///< Use separate CPU threads to search for candidates in returned data

#define     FLAG_STK_UP         (1ULL<<33)      ///< Process stack in increasing size order
#define     FLAG_CONV           (1ULL<<34)      ///< Multiply and FFT each stack "together"

#define     FLAG_OPT_LOCAVE     (1ULL<<35)      ///< Use local average normalisation instead of median in the optimisation
#define     FLAG_OPT_BEST       (1ULL<<36)      ///< Use local average normalisation instead of median in the optimisation
#define     FLAG_OPT_SWARM      (1ULL<<37)      ///< Use particle swarm to optimise candidate location
#define     FLAG_OPT_DYN_HW     (1ULL<<46)      ///< Use Dynamic half-width in optimisation

#define     FLAG_RAND_1         (1ULL<<38)      ///< Random Flag 1

#define     FLAG_TIME           (1ULL<<39)      ///< Record and report timing for the various steps in the search, this should only be used with FLAG_SYNCH
#define     FLAG_SYNCH          (1ULL<<40)      ///< Run the search in synchronous mode, this is slow and should only be used for testing
#define     FLAG_SEPSRCH        (1ULL<<41)      ///< Create a separate second output location for the search output
#define     FLAG_SEPRVAL        (1ULL<<42)      ///< Create a separate location list
#define     FLAG_DPG_PRNT_CAND  (1ULL<<43)      ///< Print candidates to
#define     FLAG_DPG_PLT_OPT    (1ULL<<44)      ///< Plot optimisation stages
#define     FLAG_DPG_SKP_OPT    (1ULL<<45)      ///< Plot optimisation stages



//================================== data types identifiers ==============================================

// ----------- This is a list of the data types that and storage structures

#define     CU_CMPLXF           (1<<1)          ///< Complex float
#define     CU_INT              (1<<2)          ///< INT
#define     CU_HALF             (1<<3)          ///< 2 byte float
#define     CU_FLOAT            (1<<4)          ///< Float
#define     CU_DOUBLE           (1<<5)          ///< Float
#define     CU_POWERZ_S         (1<<6)          ///< A value and a z bin         candPZs
#define     CU_POWERZ_I         (1<<7)          ///< A value and a z bin         candPZi
#define     CU_CANDMIN          (1<<8)          ///< A compressed candidate      candMin
#define     CU_CANDSMAL         (1<<9)          ///< A compressed candidate      candSml
#define     CU_CANDBASC         (1<<10)         ///< A compressed candidate      accelcandBasic
#define     CU_CANDFULL         (1<<11)         ///< Full detailed candidate     cand
#define     CU_TYPE_ALLL        (CU_CMPLXF | CU_INT | CU_HALF | CU_FLOAT | CU_POWERZ_S | CU_POWERZ_I | CU_CANDMIN | CU_CANDSMAL | CU_CANDBASC | CU_CANDFULL )

#define     CU_STR_ARR          (1<<20)         ///< Candidates are stored in an array (requires more memory)
#define     CU_STR_PLN          (1<<21)
#define     CU_STR_LST          (1<<22)         ///< Candidates are stored in a list  (usually a dynamic linked list)
#define     CU_STR_QUAD         (1<<23)         ///< Candidates are stored in a dynamic quadtree
#define     CU_SRT_ALL          (CU_STR_ARR | CU_STR_PLN | CU_STR_LST | CU_STR_QUAD )

// ----------- This is a list of the data types that and storage structures

#define     HAVE_INPUT          (1<<1)
#define     HAVE_MULT           (1<<2)
#define     HAVE_PLN            (1<<3)          ///< The Plane data is ready to search
#define     HAVE_SS             (1<<4)          ///< The S&S is complete and the data is read to read
#define     HAVE_RES            (1<<5)          ///< The S&S is complete and the data is read to read


//=========================================== enums ======================================================

#define  TIME_CONTEXT   0
#define  TIME_PREP      1
#define  TIME_CPU_KER   2
#define  TIME_GPU_KER   3
#define  TIME_ALL_SRCH  4
#define  TIME_GPU_SRCH  5
#define  TIME_CPU_SRCH  6
#define  TIME_CND       7
#define  TIME_ALL_OPT   8
#define  TIME_GPU_OPT   9
#define  TIME_CPU_OPT   10
#define  TIME_END       11

//========================================== Macros ======================================================

///< Defines for safe calling usable in C
#define CUDA_SAFE_CALL(value, errorMsg)     __cuSafeCall   (value, __FILE__, __LINE__, errorMsg )
#define CUFFT_SAFE_CALL(value,  errorMsg)   __cufftSafeCall(value, __FILE__, __LINE__, errorMsg )


#ifdef  CUDA_PROF
#define NV_RANGE_POP()          nvtxRangePop()
#define NV_RANGE_PUSH(x)        nvtxRangePush(x)
#define NV_NAME_STREAM(x,y)     nvtxNameCudaStreamA(x,y)
#else
#define NV_RANGE_POP(x)
#define NV_RANGE_PUSH(x)
#define NV_NAME_STREAM(x,y)
#endif

//====================================== Global variables ================================================

extern int    useUnopt;                                                         /// Use a saved text list of candidates this is used in development for optimising the optimisation stage
extern int    msgLevel;                                                         /// The level of debug messages to print, 0 -> none  higher results in more messages

//===================================== Struct prototypes ================================================

typedef struct cuSearch cuSearch;
typedef struct resThrds resThrds;

//======================================== Type defines ==================================================

///< A complex float in device texture memory
typedef cudaTextureObject_t fCplxTex;

///< A complex number data type
typedef struct fcomplexcu
{
    float           r;                  ///< Real Component
    float           i;                  ///< Imaginary Component
} fcomplexcu;

///< Basic accel search candidate to be used in CUDA kernels
///< Note this may not be the best choice on a GPU as it has a bad size
typedef struct candPZs
{
    float           value;              ///< This cab be Sigma or summed power
    short           z;                  ///< Fourier f-dot of first harmonic
} candPZs;

///< Basic accel search candidate to be used in CUDA kernels
///< Note this may not be the best choice on a GPU as it has a bad size
typedef struct candPZi
{
    float           value;              ///< This cab be Sigma or summed power
    int             z;                  ///< Fourier f-dot of first harmonic
} candPZi;

///< The most basic accel search candidate to be used in CUDA kernels (numharm can be got from stage)
typedef struct candMin
{
    float           power;              ///< Power
    int             z;                  ///< Fourier f-dot of first harmonic
} candMin;

///< The most basic accel search candidate to be used in CUDA kernels (numharm can be got from stage)
typedef struct candSml
{
    float           sigma;              ///< Sigma - adjusted for number of trials, NOTE: at some points this value holds the sum of powers from which the sigma value will be calculated
    float           power;              ///< Power
    int             z;                  ///< Fourier f-dot of first harmonic
} candSml;

///< Basic accel search candidate to be used in CUDA kernels
typedef struct accelcandBasic
{
    float           sigma;              ///< Sigma - adjusted for number of trials, NOTE: at some points this value holds the sum of powers from which the sigma value will be calculated
    short           numharm;            ///< Number of harmonics summed
    short           z;                  ///< Fourier f-dot of first harmonic
} accelcandBasic;

///< Accel search candidate (this holds more info and is thus larger than accelcandBasic
typedef struct initCand
{
    double          r;                  ///< Real bin index
    double          z;                  ///< First derivative - This could be float?
    double          power;              ///< Power ( r^2 + i^2 )
    float           sig;                ///< Gaussian sigma equivalent
    int             numharm;            ///< Number of numbers summed
} initCand;

/** A data structure to pass to CUFFT call-back load functions
 * This holds relevant info on the stack being FFT'd
 */
typedef struct stackInfo
{
    int             noSteps;            ///<  The Number of steps in the stack
    int             noPlanes;           ///<  The number of planes in the stack
    int             famIdx;             ///<  The stage order of the first plane in the stack
    int64_t         flags;              ///<  CUDA accel search bit flags

    void*           d_planeData;        ///<  Plane data for this stack
    void*           d_planePowers;      ///<  Powers for this stack
    fcomplexcu*     d_iData;            ///<  Input data for this stack
} stackInfo;

/** A structure to hold information on a raw fft
 */
typedef struct fftInfo
{
    double          rlo;                ///< The Low bin   (of interest)
    double          rhi;                ///< The high bin  (of interest)

    long long       idx;                ///< The FFT bin index of the first memory location
    long long       nor;                ///< The number of bins in the memory location

    fcomplex*       fft;                ///< The array of complex numbers (nor long)
} fftInfo;

typedef struct candOpt
{
    float           power;
    double          r;                  /// TODO: Should this be a double?
    double          z;
} candOpt;

/** Details of a GPU  .
 */
typedef struct gpuInf
{
    int     devid;                      ///<
    int     alignment;                  ///<
    float   capability;                 ///<
    char*   name;                       ///<
} gpuInf;

/** Details of a GPU  .
 */
typedef struct runInf
{
    long long           runTimes[TIME_END];             ///<
} runInf;

//------------- Data structures for, planes, stacks, batches etc ----------------

/** The general information of a f-∂f plane
 * NOTE: This is everything that is not specific to a particular plane
 */
typedef struct cuHarmInfo
{
    size_t          height;             ///< The number if rows (Z's)
    size_t          width;              ///< The number of columns, including the contaminated ends (this should always be a power of 2)

    int             halfWidth;          ///< The kernel half width         - in input fft units ie needs to be multiply by ACCEL_RDR to get plane units
    int             kerStart;           ///< The starting point for data in the various planes, this is essentially the largest halfwidth in the stack

    int             zmax;               ///< The maximum (and minimum) z
    float           harmFrac;           ///< The harmonic fraction
    int             stackNo;            ///< Which Stack is this plane in. (0 indexed at starting at the widest stack)

    int             yInds;              ///< The offset of the y offset in constant memory
    int             stageIndex;         ///< The index of this harmonic in the staged order
} cuHarmInfo;

/** The complex multiplication kernels of a f-∂f plane
 */
typedef struct cuKernel
{
    cuHarmInfo*     harmInf;            ///< A pointer to the harmonic information for this kernel
    void*           d_kerData;          ///< A pointer to the first kernel element (Width, Stride and height determined by harmInf)
    fCplxTex        kerDatTex;          ///< A texture holding the kernel data
} cuKernel;

/** A f-∂f plane  .
 * This could be a fundamental or harmonic
 * it holds basic information no memory addresses
 */
typedef struct cuFFdot
{
    cuHarmInfo*     harmInf;            ///< A pointer to the harmonic information for this planes
    cuKernel*       kernel;             ///< A pointer to the kernel for this plane

    // pointers to device data
    void*           d_planeMult;        ///< A pointer to the first element of the complex f-∂f plane (Width, Stride and height determined by harmInf)
    void*           d_planePowr;        ///< A pointer to the powers for this stack
    fcomplexcu*     d_iData;            ///< A pointer to the input data for this plane this is a section of the 'raw' complex fft data, that has been Normalised, spread and FFT'd

    // Texture objects
    fCplxTex        datTex;             ///< A texture holding the kernel data  [ Depreciated ]
    fCplxTex        powerTex;           ///< A texture of the power data        [ Depreciated ]
} cuFFdot;

/** A stack of f-∂f planes that all have the same FFT width
 */
typedef struct cuFfdotStack
{
    ////////////////// Stack parameters \\\\\\\\\\\\\\\\\\

    int             noInStack;          ///< The number of planes in this stack
    int             startIdx;           ///< The family index the first plane of the stack
    size_t          width;              ///< The width of  the entire stack, for one step [ in complex numbers! ]
    size_t          height;             ///< The height of the entire stack, for one step
    size_t          kerHeigth;          ///< The height of the multiplication kernel for this stack (this is equivalent to the height of the largest plane in the stack)

    ////////////////// sub-structures \\\\\\\\\\\\\\\\\\

    // Sub data structures associated with this stack
    cuFFdot*        planes;             ///< A pointer to all the pains for this stack
    cuHarmInfo*     harmInf;            ///< A pointer to all the harmonic info's for this stack
    cuKernel*       kernels;            ///< A pointer to all the kernels for this stack

    ////////////////// Search parameters \\\\\\\\\\\\\\\\\\

    int64_t         flags;              ///< CUDA accel search bit flags

    int             mulSlices;          ///< The number of slices to do multiplication with
    int             mulChunk;           ///< The Sum and search chunk size

    // CUFFT details
    cufftHandle     plnPlan;            ///< A cufft plan to fft the entire stack
    cufftHandle     inpPlan;            ///< A cufft plan to fft the input data for this stack

    // FFTW details
    fftwf_plan      inpPlanFFTW;        ///< A FFTW plan to fft the input data for this stack

    int startZ[MAX_IN_STACK];           ///< The y 'start' of the planes in this stack - assuming one step

    ////////////////// Memory information \\\\\\\\\\\\\\\\\\

    size_t          strideCmplx;        ///< The stride of the block of memory  [ in complex numbers! ]
    size_t          stridePower;        ///< The stride of the powers

    void*           d_kerData;          ///< Kernel data for this stack

    fcomplexcu*     h_iBuffer;          ///< Pointer to host memory to do CPU "work" on the Input data for the batch
    fcomplexcu*     h_iData;            ///< Paged locked input data for this stack
    fcomplexcu*     d_iData;            ///< Device       input data for this stack


    void*           d_planeMult;        ///< Plane of complex data for multiplication
    void*           d_planePowr;        ///< Plane of float data for the search

    stackInfo*      d_sInf;             ///< Stack info structure on the device (usually in constant memory)
    int             stkIdx;             ///< The index of this stack in the constant device memory list of stacks

    ////////////////// Asynchronous CUDA information \\\\\\\\\\\\\\\\\\

    // Streams
    cudaStream_t    initStream;         ///< CUDA stream for work on input data for the stack
    cudaStream_t    inptStream;         ///< CUDA stream for work on input data for the stack
    cudaStream_t    fftIStream;         ///< CUDA stream to CUFFT the input data
    cudaStream_t    multStream;         ///< CUDA stream for work on the stack
    cudaStream_t    fftPStream;         ///< CUDA stream for the inverse CUFFT the plane

    // CUDA Texture
    fCplxTex        kerDatTex;          ///< A texture holding the kernel data

    // CUDA Events
    cudaEvent_t     normComp;           ///< Normalisation of input data
    cudaEvent_t     inpFFTinitComp;     ///< Preparation of the input data complete
    cudaEvent_t     multComp;           ///< Multiplication complete
    cudaEvent_t     ifftComp;           ///< Creation (multiplication and FFT) of the complex plane complete
    cudaEvent_t     ifftMemComp;        ///< IFFT memory copy

    // CUDA TIMING events
    cudaEvent_t     normInit;           ///< Multiplication starting
    cudaEvent_t     inpFFTinit;         ///< Start of the input FFT
    cudaEvent_t     multInit;           ///< Multiplication starting
    cudaEvent_t     ifftInit;           ///< Start of the inverse FFT
    cudaEvent_t     ifftMemInit;        ///< IFFT memory copy start

} cuFfdotStack;

/** Details of the number of bins of the full search
 */
typedef struct searchScale
{
    double          searchRLow;         ///< The value of the input r bin to start the search at
    double          searchRHigh;        ///< The value of the input r bin to end   the search at

    long long       rLow;               ///< The lowest  possible R this search could find, Including halfwidth, thus may be less than 0
    long long       rHigh;              ///< The highest possible R this search could find, Including halfwidth

    unsigned long long noInpR;          ///< The maximum number of r input ( this is essentially  (rHigh - rLow) ) and me be longer than fft length because of halfwidth this requires the FFT to be padded!
    unsigned long long noOutpR;         ///< The maximum number of r bins the fundamental search will produce. This is ( searchRHigh - searchRLow ) / ( candidate resolution ) It may need to be scaled by numharmstages

    long long       noSteps;            ///< The number of steps the FFT is divided into
} searchScale;

/** Details of the section/step of the input FFT
 */
typedef struct rVals
{
    int             step;               ///< The step these r values cover
    double          drlo;               ///< The value of the first usable bin of the plane (the start of the step). Note: this could be a fraction of a bin (Fourier interpolation)
    double          drhi;               ///< The value of the first usable bin of the plane (the start of the step). Note: this could be a fraction of a bin (Fourier interpolation)
    long long       lobin;              ///< The first bin to copy from the the input FFT ( serachR scaled - halfwidth )
    long            numdata;            ///< The number of input FFT points to read
    long            numrs;              ///< The number of good bins in the plane ( expanded units )
    long long       expBin;             ///< The index of the expanded bin of the first good value
    double          norm;               ///< The normalisation factor used to normalise the input
} rVals;

/** User specified search details
 *
 */
typedef struct searchSpecs
{
    int                 noHarmStages;                   ///< The number of stages of harmonic summing

    int                 zMax;                           ///< The highest z drift of the fundamental
    int                 pWidth;                         ///< The desired width of the planes
    int                 ssStepSize;                     ///< The size of the steps to take through the in-memory plane
    float               sigma;                          ///< The cut off sigma
    fftInfo             fftInf;                         ///< The details of the input fft - location size and area to search

    int64_t             flags;                          ///< The search bit flags
    int                 normType;                       ///< The type of normalisation to do

    int                 mulSlices;                      ///< The number of multiplication slices
    int                 ssSlices;                       ///< The number of Sum and search slices

    int                 ssChunk;                        ///< The multiplication chunk size
    int                 mulChunk;                       ///< The Sum and search chunk size

    int                 retType;                        ///< The type of output
    int                 cndType;                        ///< The type of output

    int                 optMinLocHarms;                 ///< The minimum number of harmonics to localise on
    int                 optMinRepHarms;                 ///< The minimum number of harmonics report on

    int                 optPlnSiz[MAX_NO_STAGES];       ///< The size of optimisation planes
    int                 optPlnDim[NO_OPT_LEVS];         ///< The size of optimisation planes
    float               optPlnScale;

    void*               outData;                        ///< A pointer to the location to store candidates
} searchSpecs;

/** User specified GPU search details
 */
typedef struct gpuSpecs
{
    int         noDevices;                      ///< The number of devices (GPU's to use in the search)
    int         devId[MAX_GPUS];                ///< A list noDevices long of CUDA GPU device id's
    int         noDevBatches[MAX_GPUS];         ///< A list noDevices long of the number of batches on each device
    int         noDevSteps[MAX_GPUS];           ///< A list noDevices long of the number of steps each device wants to use
    int         noDevOpt[MAX_GPUS];             ///< A list noDevices long of the number of optimisations each device wants to do
    gpuInf      devInfo[MAX_GPUS];              ///< A list noDevices long of basic information of the GPU

    pthread_t   cntxThread;                     ///< A pthread to initialise the CUDA context in
    long long   nctxTime;                       ///< The amount of time it took to initialise the cuda contexts
} gpuSpecs;

/** A collection of f-∂f plane(s) and all its/their sub harmonics
 * This is a collection of stack(s) that make up a harmonic family of f-∂f plane(s)
 * And the device specific multiplication kernels which is just another batch
 */
typedef struct cuFFdotBatch
{
    cuSearch*       cuSrch;               ///< A pointer to the parent search info

    ////////////////// Batch parameters \\\\\\\\\\\\\\\\\\

    // Batch specific info
    int             noStacks;           ///< The number of stacks in this batch
    int             noHarmStages;       ///< The number of stages of harmonic summing

    int             noGenHarms;         ///< The number of harmonics in the family
    int             noSrchHarms;        ///< The number of harmonics in the family

    int             noSteps;            ///< The number of steps processed by the batch
    uint            noResults;          ///< The number of results from the previous search
    int             device;             ///< The CUDA device to run on
    float           capability;         ///< The CUDA capability of the device
    int             srchMaster;         ///< Weather this is the master batch
    int             isKernel;           ///< Weather this is the master batch

    ////////////////// sub-structures \\\\\\\\\\\\\\\\\\

    // Pointers to sub-structures
    cuFfdotStack*   stacks;             ///< A list of the stacks
    cuFFdot*        planes;             ///< A list of the planes
    cuKernel*       kernels;            ///< A list of the kernels
    cuHarmInfo*     hInfos;             ///< A list of the harmonic information

    ////////////////// Search parameters \\\\\\\\\\\\\\\\\\

    // Bit flags
    int             retType;            ///< The type of output
    int             cndType;            ///< The type of output
    int64_t         flags;              ///< CUDA accel search bit flags

    // Batch specific search parameters
    int             mulSlices;          ///< The number of slices to do multiplication with
    int             ssSlices;           ///< The number of slices to do sum and search with
    int             ssChunk;            ///< The multiplication chunk size
    int             mulChunk;           ///< The Sum and search chunk size

    // Batch independent search parameters
    uint            accelLen;           ///< The size to step through the input fft

    ////////////////// Memory information \\\\\\\\\\\\\\\\\\

    // Data sizes in bytes
    int             inpDataSize;        ///< The size of the input data memory in bytes
    int             retDataSize;        ///< The size of data to return in bytes
    int             plnDataSize;        ///< The size of the complex plane data memory in bytes
    int             pwrDataSize;        ///< The size of the powers  plane data memory in bytes
    int             kerDataSize;        ///< The size of the plane data memory in bytes

    // Stride information (only the results are specific to the batch)

    uint            strideOut;          ///< The stride of the returned candidate data - The stride of one step

    fcomplexcu*     h_iBuffer;          ///< Pointer to host memory to do CPU "work" on the Input data for the batch
    fcomplexcu*     h_iData;            ///< Pointer to page locked host memory of the input data for the batch
    fcomplexcu*     d_iData;            ///< Input data for the batch - NB: This could be a contiguous block of sections or all the input data depending on inpMethoud

    float*          h_normPowers;       ///< A array to store powers for running double-tophat local-power normalisation

    void*           d_kerData;          ///< Kernel data for all the stacks, generally this is only allocated once per device
    void*           d_planeMult;        ///< Plane of complex data for multiplication
    void*           d_planePowr;        ///< Plane of float data for the search

    void*           h_outData1;         ///< The output
    void*           d_outData1;         ///< The output

    void*           h_outData2;         ///< The output
    void*           d_outData2;         ///< The output

    ////////////////// Step information \\\\\\\\\\\\\\\\\\

    // Information on the input for the batch
    char            noRArryas;          ///< The number of r value arrays
    char            rActive;            ///< The index of the r-array we are working on
    rVals****       rAraays;            ///< Pointer to an array of 2D array [step][harmonic] of the base expanded r index

    rVals*          rArr1;
    rVals*          rArr2;

    rVals***        rArraysPlane;       ///< Pointer to an array of 2D array [step][harmonic] of the base expanded r index
    rVals***        rArraysSrch;        ///< Pointer to an array of 2D array [step][harmonic] of the base expanded r index


    ////////////////// Asynchronous CUDA information \\\\\\\\\\\\\\\\\\

    // Streams
    cudaStream_t    inpStream;          ///< CUDA stream for work on input data for the batch
    cudaStream_t    multStream;         ///< CUDA stream for multiplication
    cudaStream_t    srchStream;         ///< CUDA stream for summing and searching the data
    cudaStream_t    resStream;          ///< CUDA stream for

    // TIMING events
    cudaEvent_t     iDataCpyInit;       ///< Copying input data to device
    cudaEvent_t     multInit;           ///< Start of batch multiplication
    cudaEvent_t     searchInit;         ///< Sum & Search start
    cudaEvent_t     candCpyInit;        ///< Finished reading candidates from the device

    // Synchronisation events
    cudaEvent_t     iDataCpyComp;       ///< Copying input data to device
    cudaEvent_t     normComp;           ///< Normalise and spread input data
    cudaEvent_t     multComp;           ///< Sum & Search complete (candidates ready for reading)
    cudaEvent_t     searchComp;         ///< Sum & Search complete (candidates ready for reading)
    cudaEvent_t     candCpyComp;        ///< Finished reading candidates from the device
    cudaEvent_t     processComp;        ///< Process candidates (usually done on CPU)

    // TIMING values
    float*          kerGenTime;         ///< Array of floats from timing one for each stack
    float*          copyH2DTime;        ///< Array of floats from timing one for each stack
    float*          normTime;           ///< Array of floats from timing one for each stack
    float*          InpFFTTime;         ///< Array of floats from timing one for each stack
    float*          multTime;           ///< Array of floats from timing one for each stack
    float*          InvFFTTime;         ///< Array of floats from timing one for each stack
    float*          copyToPlnTime;      ///< Array of floats from timing one for each stack
    float*          searchTime;         ///< Array of floats from timing one for each stack
    float*          resultTime;         ///< Array of floats from timing one for each stack
    float*          copyD2HTime;        ///< Array of floats from timing one for each stack

#if CUDA_VERSION >= 6050
    cufftCallbackLoadC    h_ldCallbackPtr;
    cufftCallbackStoreC   h_stCallbackPtr;

    cufftCallbackLoadC    h_ldCallbackPtr0;
    cufftCallbackLoadC    h_ldCallbackPtr1;
    cufftCallbackLoadC    h_ldCallbackPtr2;
    cufftCallbackLoadC    h_ldCallbackPtr3;
    cufftCallbackLoadC    h_ldCallbackPtr4;
#endif

} cuFFdotBatch;

/** A struct to keep info on all the kernels and batches to use with cuda accelsearch  .
 */
typedef struct cuPlnInfo
{
    int             noDevices;          ///< The number of devices (GPU's to use in the search)
    cuFFdotBatch*   kernels;            ///< A list noDevices long of multiplication kernels: These hold: basic info, the address of the multiplication kernels on the GPU, the CUFFT plan.

    int             noBatches;          ///< The total number of batches there across all devices
    cuFFdotBatch*   batches;            ///< A list noBatches long of multiplication kernels: These hold: basic info, the address of the multiplication kernels on the GPU, the CUFFT plan.

    int             noSteps;            ///< The total steps in all batches - there are across all devices

    int*            devNoStacks;        ///< An array of the number of stacks on each device
    stackInfo**     h_stackInfo;        ///< An array of pointers to host memory for the stack info
} cuPlnInfo;

/** Data structure to hold the GPU information for performing GPU optimisation  .
 *
 */
typedef struct cuOptCand
{
    cuSearch*       cuSrch;             ///< Details of the search

    int             device;             ///< The device ID of the device to use
    int             alignment;          ///< The memory alignment block size in bytes

    double          centR;
    double          centZ;
    double          rSize;
    double          zSize;

    int             maxNoR;
    int             maxNoZ;

    int             pIdx;               ///< The index of this optimiser in the list

    int             noZ;
    int             noR;

    int             halfWidth;

    int             noHarms;

    double          norm[32];
    int             loR[32];
    int             hw[32];

    int             maxHalfWidth;
    int             inpSz;              ///< The size in bytes of device input buffer
    int             outSz;              ///< The size in bytes of device output buffer

    fcomplexcu*     d_inp;
    void*           d_out;

    fcomplexcu*     h_inp;
    void*           h_out;

    int             outStride;
    int             inpStride;

    // Streams
    cudaStream_t    stream;             ///< CUDA stream for work

    // Events
    cudaEvent_t     inpInit;            ///< Copying input data to device
    cudaEvent_t     inpCmp;             ///< Copying input data to device
    cudaEvent_t     compInit;           ///< Copying input data to device
    cudaEvent_t     compCmp;            ///< Copying input data to device
    cudaEvent_t     outInit;            ///< Copying input data to device
    cudaEvent_t     outCmp;             ///< Copying input data to device
} cuOptCand;

/** A struct to keep info on all the kernels and batches to use with cuda accelsearch  .
 */
typedef struct cuOptInfo
{
    int             noOpts;             ///< The total number of optimisations to do across all devices
    cuOptCand*      opts;               ///< A list noBatches long of multiplication kernels: These hold: basic info, the address of the multiplication kernels on the GPU, the CUFFT plan.
} cuOptInfo;

/** Details of the GPU's  .
 */
typedef struct cuGpuInfo
{
    // Details of the GPU's in use
    int             noDevices;          ///< The number of devices (GPU's to use in the search)

    int             devid[MAX_GPUS];
    int             alignment[MAX_GPUS];
    float           capability[MAX_GPUS];
    char*           name[MAX_GPUS];
} cuGpuInfo;

/** User independent details  .
 */
struct cuSearch
{
    searchSpecs*        sSpec;              ///< Specifications of the search
    gpuSpecs*           gSpec;              ///< Specifications of the GPU's to use
    searchScale*        SrchSz;             ///< Details on o the size (in bins) of the search
    resThrds*           threasdInfo;        ///< Information on threads to handle returned candidates.
    cuPlnInfo*          pInf;               ///< The allocated Device and host memory and data structures to create planes including the kernels
    cuOptInfo*          oInf;               ///< Details of optimisations
    runInf*             rInf;                   ///< Information of the actual run

    // Some extra search details
    int                 noHarmStages;       ///< The number of stages of harmonic summing
    int                 noGenHarms;         ///< The number of harmonics in the family
    int                 noSrchHarms;        ///<
    int                 noSteps;            ///< The number of steps to cover the entire input data

    long long           timings[TIME_END];  ///<

    // Search power cutoff values
    int*                sIdx;               ///< The index of the planes in the Presto harmonic summing order
    float*              powerCut;           ///< The power cutoff
    long long*          numindep;           ///< The number of independent trials
    int*                yInds;              ///< The Y indices

    // Search specific memory
    void*               h_candidates;       ///< Host memory for candidates
    void*               d_planeFull;        ///< Device memory for the in-mem f-∂f plane

    unsigned int        inmemStride;        ///< The stride (in units) of the in-memory plane data in device memory
};

/** Information of the P-threads used in the search
 *
 */
struct resThrds
{
    sem_t           running_threads;

    pthread_mutex_t running_mutex;
    pthread_mutex_t candAdd_mutex;

};

/** A data structure to pass to a thread, containing information on search results
 *
 */
typedef struct resultData
{
    cuSearch*           cuSrch;                 ///< Details of the search

    void*               retData;

    uint                retType;
    uint                cndType;
    int64_t             flags;                 ///< CUDA accel search bit flags

    uint                x0;
    uint                x1;

    uint                y0;
    uint                y1;

    uint                xStride;
    uint                yStride;

    uint                zMax;

    double              rLow;

    rVals               rVal;

    uint*               noResults;
    float*              resultTime;

} resultData;

/** This is just a wrapper to be passed to a CPU thread  .
 *
 */
typedef struct candSrch
{
    cuSearch*           cuSrch;                 ///< Details of the search
    accelcand*          cand;                   ///< The candidate to optimise
    cuOptCand*          optPln;                 ///< The plane data used for optimisation
    int                 candNo;                 ///< The 0 based index of this candidate
    double*             norms;                  ///< Normalisation values for each harmonic
} candSrch;


//===================================== Function prototypes ===============================================

/** Read the GPU details from clig command line  .
 *
 * @param cmd
 * @return A pointer to the accel info struct to fill
 */
ExternC gpuSpecs readGPUcmd(Cmdline *cmd);

/** Read the GPU details from clig command line  .
 *
 * @param cmd
 * @return A pointer to the accel info struct to fill
 */
ExternC searchSpecs readSrchSpecs(Cmdline *cmd, accelobs* obs);

ExternC cuSearch* initSearchInf(searchSpecs* sSpec, gpuSpecs* gSpec, cuSearch* srch);

ExternC cuSearch* initCuKernels(searchSpecs* sSpec, gpuSpecs* gSpec, cuSearch* srch);

ExternC cuSearch* initCuOpt(searchSpecs* sSpec, gpuSpecs* gSpec, cuSearch* srch);

ExternC void freeCuSearch(cuSearch* srch);

ExternC void freeAccelGPUMem(cuPlnInfo* mInf);

ExternC cuOptCand* initOptPln(searchSpecs* sSpec);
ExternC cuOptCand* initOptSwrm(searchSpecs* sSpec);


/** Initialise the template structure and kernels for a multi-step batches  .
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

/** Free all host and device memory allocated by initHarmonics(...)  .
 * If the stkLst is master, this will free any device independat memory
 *
 * @param stkLst            The data structure to free
 * @param master            The master stak list
 * @param out               The candidate output, if none was specified this should be NULL
 */
//ExternC  void freeHarmonics(cuFFdotBatch* stkLst, cuFFdotBatch* master, void* out);

/** Initialise a multi-step batch from the device kernel  .
 *
 * @param harms             The kernel to base this multi-step batch
 * @param no                The index of this batch
 * @param of                The desired number of batches on this device
 * @return
 */
//ExternC cuFFdotBatch* initBatch(cuFFdotBatch* harms, int no, int of);

/** Free device and host memory allocated by initStkList  .
 *
 * @param harms             The batch to free
 */
//ExternC void freeBatch(cuFFdotBatch* stkLst);

ExternC void setContext(cuFFdotBatch* batch) ;

ExternC int setDevice(int device);

ExternC void freeBatchGPUmem(cuFFdotBatch* batch);

ExternC void printCands(const char* fileName, GSList *candsCPU, double T);

ExternC void search_ffdot_batch_CU(cuFFdotBatch* planes, double* searchRLow, double* searchRHi, int norm_type );

ExternC void inmemSS(cuFFdotBatch* batch, double drlo, int len);

ExternC void inmemSumAndSearch(cuSearch* cuSrch);

ExternC void finish_Search(cuFFdotBatch* batch);

ExternC void add_and_search_IMMEM(cuFFdotBatch* batch );

ExternC void accelMax(fcomplex* fft, long long noBins, long long startBin, long long endBin, short zMax, short numharmstages, float* powers );

/** Print the flag values in text  .
 *
 * @param flags
 */
ExternC void printFlags(uint flags);

ExternC void printCommandLine(int argc, char *argv[]);

ExternC void writeLogEntry(const char* fname, accelobs* obs, cuSearch* cuSrch, long long prepTime, long long cpuKerTime, long long cupTime, long long gpuKerTime, long long gpuTime, long long optTime, long long cpuOptTime, long long gpuOptTime);

ExternC GSList* getCanidates(cuFFdotBatch* batch, GSList* cands );

ExternC void calcNQ(double qOrr, long long n, double* p, double* q);

ExternC GSList* testTest(cuFFdotBatch* batch, GSList* candsGPU);

ExternC int waitForThreads(sem_t* running_threads, const char* msg, int sleepMS );

ExternC long long initCudaContext(gpuSpecs* gSpec);

ExternC long long compltCudaContext(gpuSpecs* gSpec);

/** Cycle back the values in the array of input data
 *
 * @param batch
 */
ExternC void CycleBackRlists(cuFFdotBatch* batch);

#endif // CUDA_ACCEL_INCLUDED
