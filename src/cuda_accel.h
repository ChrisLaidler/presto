#ifndef CUDA_ACCEL_INCLUDED
#define CUDA_ACCEL_INCLUDED

#include <cuda.h>
#include <cufft.h>
//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>

#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>

#define ExternC
#ifdef __cplusplus
#define ExternC extern "C"
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

#define   FOLD if (1)                   /// A simple marker used for folding blocks of code in NSIGHT

#define   MAX_IN_STACK  10              /// NOTE: this is 1 to big to handle the init problem
#define   MAX_HARM_NO   16              /// The maximum number of harmonics handled by a accel search
#define   MAX_YINDS     16000           /// The maximum number of y indices to store in constant memory
#define   MAX_STEPS     8               /// The maximum number of steps

//======================================== Debug Defines  ================================================\\

#define DBG_KER01       0       /// Convolution kernel
#define DBG_KER02       0       /// Convolution kernel post fft
#define DBG_PRNTKER02   0       /// Convolution kernel post fft

#define DBG_INP01       0       /// RAW FFT
#define DBG_INP02       0       /// normalised RAW FFT
#define DBG_INP03       0       /// spread RAW FFT
#define DBG_INP04       0       /// FFT'd input data

#define DBG_PLN01       0       /// Input convolved
#define DBG_PLN02       0       /// Input convolved & FFTD (Complex plain)
#define DBG_PLN03       0       /// Summed powers (ie The f-∂f plain)

#define DBG_PLTPLN06    0       /// Input convolved & FFTD (Complex plain)
#define DBG_PLTDETC     0       /// Detections


//====================================== Bit flag values =================================================\\

#define     CU_CAND_DEVICE      (1<<0)    /// Write all candidates to the device
#define     CU_CAND_HOST        (1<<1)    /// Write all candidates to the device
#define     CU_CAND_SINGLE_G    (1<<2)    /// Only get candidates from the current plain - This seams to be best in most cases
#define     CU_CAND_SINGLE_C    ((1<<3)| CU_CAND_SINGLE_G)      /// Only get candidates from the current plain
#define     CU_CAND_ALL         ( CU_CAND_DEVICE | CU_CAND_HOST | CU_CAND_SINGLE_G | CU_CAND_SINGLE_C  )

#define     CU_INPT_DEVICE      (1<<5)   /// Put the entire raw input FFT on device memory - No CPU synchronisation required, but normalisation on GPU!
#define     CU_INPT_HOST        (1<<6)   /// Use host locked data for entire raw FFT data -  normalisation on GPU
#define     CU_INPT_SINGLE_G    (1<<7)   /// Prepare input data using GPU - normalisation on GPU
#define     CU_INPT_SINGLE_C    (1<<8)   /// Prepare input data using CPU - Generally bets option, as CPU is "idle"
#define     CU_INPT_ALL         ( CU_INPT_DEVICE | CU_INPT_HOST | CU_INPT_SINGLE_G | CU_INPT_SINGLE_C  )

#define     FLAG_SAS_SIG        (1<<10)   /// Do sigma calculations on the GPU - Generally this can be don on the CPU while the GPU works
#define     FLAG_PLN_TEX        (1<<11)   /// Use texture memory to access the d-∂d plains during sum and search (non interpolation method) - May give advantage on pre-Fermi generation which we don't really care about

#define     FLAG_CNV_TEX        (1<<12)   /// Use texture memory for convolution  - May give advantage on pre-Fermi generation which we don't really care about
#define     FLAG_CNV_1KER       (1<<13)   /// Use minimal kernel                  - ie Only the kernel of largest plain in each stack
#define     FLAG_CNV_OVLP       (1<<14)   /// Use the overlap kernel              - I found this slower that the alternative
#define     FLAG_CNV_PLN        (1<<15)   /// Convolve one plain at a time
#define     FLAG_CNV_STK        (1<<16)   /// Convolve one stack at a time        - This seams to be best in most cases
#define     FLAG_CNV_FAM        (1<<17)   /// Convolve one family at a time       - Preferably don't use this!
#define     FLAG_CNV_ALL        ( FLAG_CNV_PLN | FLAG_CNV_STK | FLAG_CNV_FAM )

#define     FLAG_STP_ROW        (1<<18 )  /// Multi-step Row   interleaved        - This seams to be best in most cases
#define     FLAG_STP_PLN        (1<<19 )  /// Multi-step Plain interleaved
#define     FLAG_STP_STK        (1<<20 )  /// Multi-step Stack interleaved        - Preferably don't use this!
#define     FLAG_STP_ALL        ( FLAG_STP_ROW | FLAG_STP_PLN | FLAG_STP_STK )



//========================================== Macros ======================================================\\

/// Defines for safe calling usable in C
#define CUDA_SAFE_CALL(value, errorMsg)     __cuSafeCall   (value, __FILE__, __LINE__, errorMsg )
#define CUFFT_SAFE_CALL(value,  errorMsg)   __cufftSafeCall(value, __FILE__, __LINE__, errorMsg )


//======================================== Type defines ==================================================\\

/// A complex float in device texture memory
typedef cudaTextureObject_t fCplxTex;

/// A complex number data type
typedef struct fcomplexcu
{
    float r, i;
} fcomplexcu;

/// Basic accel search candidate to be used in CUDA kernels
typedef struct accelcandBasic
{
  float sigma;        // Sigma - adjusted for number of trials, NOTE: at points this value holds the sum of powers from which the sigma value will be calculated
  short numharm;      // Number of harmonics summed
  short z;            // Fourier f-dot of first harmonic
} accelcandBasic;

/// Accel search candidate (this holds more info and is thus larger than accelcandBasic
typedef struct cand
{
    float   power;
    double  r;
    double  sig;
    float   z;
    int     numharm;
} cand;

//------------- Arrays that can be passed to kernels -------------------\\

typedef struct iHarmList
{
    int val[MAX_HARM_NO];
} iHarmList;

typedef struct fHarmList
{
    float val[MAX_HARM_NO];
} fHarmList;

typedef struct dHarmList
{
    double val[MAX_HARM_NO];
} dHarmList;

typedef struct cHarmList
{
    fcomplexcu* __restrict__ val[MAX_HARM_NO];
} cHarmList;

typedef struct tHarmList
{
    cudaTextureObject_t val[MAX_HARM_NO];
} tHarmList;


//------------- Data structures for, plains, stacks etc ----------------\\

typedef struct cuHarmInfo
{
    size_t width;               /// The number of complex numbers in each kernel (2 floats)
    size_t stride;              /// The x stride in complex numbers
    size_t height;              /// The number if rows (Z's)
    int zmax;                   /// The maximum (and minimum) z
    float harmFrac;             /// The harmonic fraction
    int halfWidth;              /// The kernel half width
    int yInds;                  /// The offset of the y offset in constant memory
    int stageOrder;             /// The index of this harmonic in the staged order
} cuHarmInfo;

typedef struct cuKernel
{
    cuHarmInfo* harmInf;        /// A pointer to the harmonic information for this kernel
    fcomplexcu* d_kerData;      /// A pointer to the first kernel element (Width, Stride and height determined by harmInf)
    fCplxTex kerDatTex;         /// A texture holding the kernel data
} cuKernel;

typedef struct cuFFdot
{
    cuHarmInfo* harmInf;              /// A pointer to the harmonic information for this plains
    cuKernel* kernel;                 /// A pointer to the kernel for this plain

    fcomplexcu* d_plainData;          /// A pointer to the first element of the complex f-∂f plain (Width, Stride and height determined by harmInf)
    fCplxTex datTex;                  /// A texture holding the kernel data

    fcomplexcu* d_iData;              /// A pointer to the input data for this plain this is a section of the 'raw' complex fft data, that has been Normalised, spread and FFT'd

    size_t numInpData[MAX_STEPS];     /// The number of input elements for this plain - (Number of R bins in the 'raw' FFT input)
    size_t numrs[MAX_STEPS];          /// The number of input elements for this plain - (Number of R bins in the 'raw' FFT input)
    float fullRLow[MAX_STEPS];        /// The low r bin of the input data used ( Note: the 0 index is [floor(rLow) - halfwidth * DR] )
    float rLow[MAX_STEPS];            /// The low r value of the plain at input fft
    float searchRlow[MAX_STEPS];      /// The low r bin of the input data used ( Note: the 0 index is [floor(rLow) - halfwidth * DR] )
    int ffdotPowWidth[MAX_STEPS];     /// The width of the final f-∂f plain

    float searchRlowPrev[MAX_STEPS];  /// The low r bin of the input data used ( Note: the 0 index is [floor(rLow) - halfwidth * DR] )

} cuFFdot;

typedef struct cuFfdotStack
{
    int noInStack;              /// The number of plains in this stack
    int startIdx;               /// The 'global' offset of the first element of the stack

    cudaStream_t cnvlStream;    /// CUDA stream for work on the stack
    cudaStream_t inpStream;     /// CUDA stream for work on input data for the stack

    size_t width;               /// The width of the block of memory   [ in complex numbers! ]
    size_t stride;              /// The stride of the block of memory  [ in complex numbers! ]
    size_t height;              /// The height of the block of memory for one step
    int startR[MAX_IN_STACK];   /// The heights of the individual plains assuming one step

    int zUp[MAX_IN_STACK];      /// The heights of the individual plains
    int zDn[MAX_IN_STACK];      /// The heights of the individual plains

    cuHarmInfo* harmInf;        /// A pointer to all the harmonic info's for this stack
    cuKernel* kernels;          /// A pointer to all the kernels for this stack
    cuFFdot* plains;            /// A pointer to all the pains for this stack

    cufftHandle plnPlan;        /// A cufft plan to fft the entire stack
    cufftHandle inpPlan;        /// A cufft plan to fft the input data for this stack

    fcomplexcu* d_kerData;      /// Kernel data for this stack
    fcomplexcu* d_plainData;    /// Plain data for this stack
    fcomplexcu* d_iData;        /// Input data for this stack

    fCplxTex kerDatTex;         /// A texture holding the kernel data

    cudaEvent_t prepComp;       /// Preparation of the input data complete
    cudaEvent_t convComp;       /// Convolution complete
    cudaEvent_t plnComp;        /// Creation (convolution and FFT) of the complex plain complete

    cudaStream_t fftPStream;    /// CUDA stream for summing and searching the data
    cudaStream_t fftIStream;    /// CUDA stream for summing and searching the data
} cuFfdotStack;

typedef struct cuStackList
{
    size_t noStacks;              /// The number of stacks in this stack list
    size_t noHarms;               /// The number of harmonics in the entire stack
    size_t noSteps;               /// The number of slices in the stack list
    int noHarmStages;             /// The number of stages of harmonic summing

    int pIdx[MAX_HARM_NO];        /// The index of the plains in the Presto harmonic summing order

    cuFfdotStack* stacks;         /// A list of the stacks
    cuHarmInfo* hInfos;           /// A list of the harmonic informations
    cuKernel* kernels;            /// A list of the kernels
    cuFFdot* plains;              /// A list of the plains

    cHarmList iDataLst;           /// A list of the input data allocated in memory
    iHarmList iDataLens;          /// A list of the input data allocated in memory

    int inpDataSize;              /// The size of the input data memory in bytes for one step
    int plnDataSize;              /// The size of the plain data memory in bytes for one step
    int kerDataSize;              /// The size of the plain data memory in bytes for one step
    int cndDataSize;              /// The size of candidates data memory in bytes for one step

    fcomplexcu* d_kerData;        /// Kernel data for all the stacks
    fcomplexcu* d_plainData;      /// Plain data for all the stacks
    fcomplexcu* d_iData;          /// Input data for all the stacks - NB: This could be a contiguous block of sections or all the input data depending on inpMethoud

    int haveSData;                /// Weather we are starting with search data
    int haveCData;                /// Weather we are starting with search data

    accelcandBasic* h_bCands;     /// A list of basic candidates in host memory
    accelcandBasic* d_bCands;     /// A list of basic candidates in device memory
    cand* h_candidates;           /// Page locked host memory for candidates
    uint* d_candSem;              /// Semaphore for writing to device candidate list

    fcomplexcu* h_iData;          /// Pointer to page locked host memory of Input data for all the stacks
    float* h_powers;              /// Powers used for running double-tophat local-power normalisation

    uint flag;                    /// CUDA accel search flags

    int rLow;                     /// The lowest possible R this search could find
    int rHigh;                    /// The highest possible R this search could find
    double searchRLow;            /// The value of the r bin to start the search at

    cudaStream_t inpStream;       /// CUDA stream for work on input data for the stack
    cudaStream_t strmSearch;      /// CUDA stream for summing and searching the data

    cudaEvent_t iDataCpyComp;     /// Copying input data to device
    cudaEvent_t candCpyComp;      /// Finished reading candidates from the device
    cudaEvent_t normComp;         /// Normalise and spread input data
    cudaEvent_t searchComp;       /// Sum & Search complete (candidates ready for reading)
    cudaEvent_t processComp;      /// Process candidates (usually done on CPU)

    int noResults;                /// The number of results from the previous search

    uint accelLen;                /// The size to step through the input fft

    CUcontext pctx;               /// Context for the stack
    int device;                   /// The CUDA device to run on;

} cuStackList;



//===================================== Function prototypes ===============================================\\


/** Initialise the template structure and kernels for a multi-step stack list
 * This is called once per device
 *
 * @param stkLst            The data structure to fill
 * @param master            The master data structure to copy kernel and some settings from, if this is the first call this should be NULL
 * @param numharmstages     The number of harmonic stages
 * @param zmax              The ZMax of the primary harmonic
 * @param obs               The observation
 * @param device            The device to create the kernels on
 * @param device            The number of steps to use on the device
 * @param width             The desired width of the primary harmonic in thousands
 * @param noThreads         The desired number of thread to run on this device
 * @return
 */
ExternC int initHarmonics(cuStackList* stkLst, cuStackList* master, int numharmstages, int zmax, accelobs* obs, int device, int noSteps, int width, int noThreads );

/** Initialise a multi-step stack list from the device kernel
 *
 * @param harms             The kernel to base this multi-step stack list
 * @param no                The index of this stack
 * @param of                The desired number of stacks on this device
 * @return
 */
ExternC cuStackList* initPlains(cuStackList* harms, int no, int of);

ExternC void sumAndSearch(cuStackList* plains, accelobs* obs, GSList** cands);

#endif // CUDA_ACCEL_INCLUDED
