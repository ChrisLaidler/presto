#ifndef CUDA_ACCEL_INCLUDED
#define CUDA_ACCEL_INCLUDED

#include <pthread.h>
#include <semaphore.h>

#include <cuda.h>
#include <cufft.h>
#include <cufftXt.h>

#if CUDART_VERSION >= 7050   // Half precision
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

#ifdef CBL
#include "log.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif
#include "accel.h"
#ifdef __cplusplus
}
#endif

#include "cuda_compex.h"



/***************************************** Section Enables *****************************************
 *
 *  This block has preprocessor directives to enable and disable a number of feature at compile time.
 *  A number of them are options enabling a number of different method to perform the various components
 *  (usually on the GPU.) A number are for debugging purposes and profiling purposes, such as the "optimal"
 *  kernels, and should generally not be included. Others work better on certain CUDA architectures.
 *
 *  Other options such as MIN_SEGMENTS control the bounds of a number of the parameters.
 *
 *  Including many version of kernels and increasing parameter bounds, will give a wider range of configurability
 *  but will increase compile time and code size, this will affect the time it takes to initialise the CUDA context (NB).
 *  Generally try to minimise the parameters to those that perform optimally on YOURE specific hardware.
 *
 *  I "ship" this with many of the parameters limited the the standard best, you can tweak them if you need.
 *  At the moment there is no auto tune but one may be on the way.
 *
 *  NB: If you change these parameters you should probably MANUALLY do a clean recompile ie ( make cudaclean; make )
 *
 *  If you have any questions pleas feel free to e-mail me: Chris Laidler ( chris.laidler+presto@gmail.com )
 *
 *  Kernels are given names and numbers ie MULT_21.
 *  The name defines the component MULT -> multiplication, the number defines a version, The first digit means:
 *  	0: Optimal ie correct number of operations with junk data and no synchronisation dependencies
 *  	1: Kernel operates on single plane
 *  	2: Kernel operates on a stack
 *  	3: Kernel operates on a family
 *  The second digit gives the version so 21 is the first version that operates on a stack of data.
 *
 */

// Just encase these define have been used elsewhere
//#undef  TIMING
//#undef  PROFILING
//#undef  NVVP
//#undef  CUDA_PROF

// A user can enable or disable GPU functionality with the defines below, if any of them are changed a full recompile is required (including an make clean!)

//     Timing
#define		TIMING  				/// Implement basic timing of sections, very low overhead, generally a good idea

//     Profiling
#define		PROFILING				/// Implement more advanced profiling. This enables timing of individual components and adding CUDA ranges

//	Visual profiler
//#define	NVVP					/// Uncomment to allow CUDA profiling


////////	General
//#define  	WITH_ITLV_PLN				///< Deprecated - Allow plane interleaving of multi-segment plans - I found this slower than row interleaving

#define		MIN_SEGMENTS		1		///< The minimum number of segments in a single CG plan
#define		MAX_SEGMENTS		12		///< The maximum number of segments in a single CG plan

#define		WITH_HALF_RESCISION_POWERS		///< Able to use powers stored as half-precision floats - This is usually the best
//#define 	WITH_SINGLE_RESCISION_POWERS		///< Able to use powers stored as single-precision floats
//#define 	WITH_COMPLEX_POWERS			///< Able to not store powers, rather store complex single-precision floats and calculate power in the SAS kernel - Not recommended

////////	Normalisation
#define 	WITH_NORM_GPU				///< Allow GPU normalisation - Bitonic sort
//#define 	WITH_NORM_GPU_OS			///< Allow GPU normalisation - using a custom novel order statistic algorithm - still needs proper testing


////////	Multiplication
#define		MIN_MUL_CHUNK		1		///< Reducing the SAS Chunk range can reduce compile time and binary size which reduces CUDA context initialisation time, generally multiplication chunks are higher so this value can be high
#define		MAX_MUL_CHUNK		12		///< I generally find lager multiplication chunks (12) do better

// Only one of the mul 0 kernels will get used
//#define  	WITH_MUL_00				///< Compile with test Multiplication kernel - Version 0 - DEBUG ONLY: Just write to ffdot plane - 1 thread per complex value  .
#define 	WITH_MUL_01				///< Compile with test Multiplication kernel - Version 1 - DEBUG ONLY: Read input, read kernel, write to ffdot plane - 1 thread per column  .
//#define 	WITH_MUL_02				///< Compile with test Multiplication kernel - Version 2 - DEBUG ONLY: Read input, read kernel, write to ffdot plane - 1 thread per column  - templated for segments  .

//#define 	WITH_MUL_PRE_CALLBACK			///< Multiplication as CUFFT callbacks - Seams very slow, probably best to disable this!

#define 	WITH_MUL_11				///< Plain multiplication kernel 1 - (slow) - Single plane at a time - generally slow and unnecessary

#define 	WITH_MUL_21				///< Stack multiplication kernel 1 - (fastest)	- This is the preferred method if compute version is > 3.0 - read all input - loop over kernel - loop over planes
#define 	WITH_MUL_22				///< Stack multiplication kernel 2 - (faster)	- Loop ( column, plain - Y )
#define 	WITH_MUL_23				///< Stack multiplication kernel 3 - (fast)	- Loop ( column, chunk (read ker) - plain - Y - segment )

//#define 	WITH_MUL_31				///< Batch multiplication kernel 1 - (slow)	- Do an entire batch in one kernel


////////	Powers
#define 	WITH_POW_POST_CALLBACK			///< Powers to be calculated in CUFFT callbacks - Always a good option


////////	Sum & Search
#define		MIN_SAS_CHUNK		1		///< Reducing the SAS Chunk range can reduce compile time and binary size which reduces CUDA context initialisation time
#define		MAX_SAS_CHUNK		12		///< Use up to 10

#define		MIN_SAS_COLUMN		1		///< Not in use yet - min columns for SAS kernels
#define		MAX_SAS_COLUMN		32		///< Not in use yet - max columns for SAS kernels

#define 	WITH_SAS_00				///< Compile with test SAS kernel - Version 0 - DEBUG ONLY: Memory reads and writes only - sliced

#define		WITH_SAS_31				///< Compile with main SAS kernel - (required) - This is currently the only sum & search kernel for the standard search

#define		WITH_SAS_IM				///< Compile with main in-memory SAS kernel - (required) - This is currently the only sum & search kernel for the standard search


////////	Candidate
#define  	WITH_SAS_COUNT				///< Allow counting of candidates in sum & search kernel - Not advisable on older ( CC < 5 ) cards


////////	Optimisation
#define		MAX_OPT_BLK_NO		16		///< Maximum number of coefficient "reuses" ie blocks in plan by block kernel (less than 16) I found speeds fatten off at some point about 6 or 8 so no point being much bigger
#define		MAX_OPT_SFL_NO		32		///< Maximum number columns in shuffle kernel - power of 2 <= 32

#define		WITH_OPT_BLK_SHF			///< This is usual the best block kernel - Share common coefficients using shuffle block
#define 	WITH_OPT_BLK_HRM			///< This is good as an alternative to the shuffle kernel (can be used on older hardware)

#define 	WITH_OPT_PTS_HRM			///< This is usual the best kernel


/******************************************* Defines ****************************************************/

#define		BIT(x)			(1ULL<<(x))

#define		MAX_IN_STACK		10		///< NOTE: this is 1 to big to handle the init problem
#define		MAX_STACKS		5		///< The maximum number stacks in a family of plains
#define		MAX_HARM_NO		16		///< The maximum number of harmonics handled by a accel search
#define		MAX_NO_STAGES		5		///< The maximum number of harmonics handled by a accel search
#define		MAX_YINDS		8500		///< The maximum number of y indices to store in constant memory - 8500 Works up to ~500
#define		INDS_BUFF		20		///< The buffer at the ends of each pane in the yInds array
#define		MAX_CG_PLANS		5		///< The maximum number of CG plans on a single GPU
#define		MAX_GPUS		32		///< The maximum number GPU's
#define		CORRECT_MULT		1		///< Generate the kernel values the correct way and do the
#define		NO_OPT_LEVS		7		///< The number of optimisation planes/segments
#define 	OPT_INP_BUF		25		///< Buffer sections of the input FT with this many bins
#define		OPT_LOC_PNT_NO		16


/************************************** Bit flag values *************************************************/

//-------------- General --------//	\\ 0 - 9 - BOTH //

#define		FLAG_DOUBLE		BIT(0)		///< Use double precision kernels and complex plane and iFFT's - Not implemented yet
#define		FLAG_ITLV_ROW		BIT(1)		///< Multi-segment Row interleaved- This seams to be best in most cases

#define		FLAG_STAGES		BIT(2)		///< Return results for all stages of summing, default is only the final result
#define		FLAG_HAMRS		BIT(3)		///< Return results for all harmonics

//		NO_VALUE				///< Save powers in single-precision
#define		FLAG_POW_HALF		BIT(4)		///< Save powers in half-precision	- This is recommended
#define		FLAG_CMPLX		BIT(5)		///< Use complex values - Default is to use powers

//------------- Kernels ------//	\\ 10 - 19 - GEN Only //

#define		FLAG_KER_HIGH		BIT(10)		///< Use increased response function width for higher accuracy at Z close to zero
#define		FLAG_KER_MAX		BIT(11)		///< Use maximum response function width for higher accuracy at Z close to zero
#define		FLAG_CENTER		BIT(12)		///< Centre and align the usable part of the convolution kernel
#define		FLAG_KER_DOUBGEN	BIT(13)		///< Create kernel with double precision calculations
#define		FLAG_KER_DOUBFFT	BIT(14)		///< Create kernel with double precision calculations and FFT's

#define		FLAG_STK_UP		BIT(15)		///< Process stack in increasing size order

//------------- Input ---------//	\\ 20 - 24 - GEN Only //

//		NO_VALUE				///< Prepare input data one segment at a time, using CPU - normalisation on CPU - Generally bets option, as CPU is "idle"
#define		CU_NORM_GPU_SM		BIT(20)		///< Prepare input data one segment at a time, using GPU - Sort using SM
#define		CU_NORM_GPU_SM_MIN	BIT(21)		///< Prepare input data one segment at a time, using CPU - Sort at most 1024 SM floats
#define		CU_NORM_GPU_OS		BIT(22)		///< Prepare input data one segment at a time, using CPU - Innovative Order statistic algorithm
#define		CU_NORM_GPU		( CU_NORM_GPU_SM | CU_NORM_GPU_SM_MIN | CU_NORM_GPU_OS )

#define		CU_NORM_EQUIV		BIT(23)		///< Do the normalisation the CPU way
#define		CU_INPT_FFT_CPU		BIT(24)		///< Do the FFT on the CPU


//------------- Multiplication ------//	\\ 25 - 34 - GEN Only //

#define		FLAG_MUL_00		BIT(25)		///< Multiply kernel (Base only do memory reads and writes - NB This does not do the actual multiplication)
#define		FLAG_MUL_11		BIT(26)		///< Multiply kernel - Do the multiplication one plane ant a time
#define		FLAG_MUL_21		BIT(27)		///< Multiply kernel - read all input - loop over kernel - loop over planes
#define		FLAG_MUL_22		BIT(28)		///< Multiply kernel - Loop ( Plane - Y )
#define		FLAG_MUL_23		BIT(29)		///< Multiply kernel - Loop ( chunk (read ker) - plan - Y - segment )
#define		FLAG_MUL_31		BIT(30)		///< Multiply kernel - Do an entire batch in one kernel
#define		FLAG_MUL_CB		BIT(31)		///< Multiply kernel - Using a CUFFT callback
#define		FLAG_MUL_PLN		( FLAG_MUL_11 )
#define		FLAG_MUL_STK		( FLAG_MUL_00 | FLAG_MUL_21 | FLAG_MUL_22 | FLAG_MUL_23 | FLAG_MUL_CB )
#define		FLAG_MUL_BATCH		( FLAG_MUL_31 )
#define		FLAG_MUL_ALL		( FLAG_MUL_BATCH | FLAG_MUL_STK | FLAG_MUL_PLN )

#define		FLAG_TEX_MUL		BIT(34)		///< [ Deprecated ]Use texture memory for multiplication- May give some advantage on pre-Fermi generation which we don't really care about

//------------- FFT -----------//	\\ 35 - 39
#define		CU_FFT_SEP_INP		BIT(35)		///< Use a separate FFT plan for the input of each CG plan
#define		CU_FFT_SEP_PLN		BIT(36)		///< Use a separate FFT plan for the plane of each CG plan
#define		CU_FFT_SEP_ALL		( CU_FFT_SEP_INP | CU_FFT_SEP_PLN ) /// All callbacks

#define		FLAG_CUFFT_CB_POW	BIT(37)		///< Use an output callback to create powers, this works in std or in-mem searches - This is a similar or faster iFFT speed AND faster SAS so good all round!
#define		FLAG_CUFFT_CB_INMEM	BIT(38)		///< Use the in-mem FFT's to copy values strait back to in-mem plane - this is generally slow
#define		FLAG_CUFFT_CB_OUT	( FLAG_CUFFT_CB_POW | FLAG_CUFFT_CB_INMEM ) /// All output callbacks - good option
#define		FLAG_CUFFT_ALL		( FLAG_CUFFT_CB_OUT | FLAG_MUL_CB ) /// All callbacks

//------------- Sum and search ------//	\\ 40 - 49 - GEN Only //

#define		FLAG_SS_CPU		BIT(40)		///< Do the sum and searching on the CPU, this is now deprecated cos its so slow!
#define		FLAG_SS_00		BIT(41)		///< This is a debug kernel used as a comparison, it is close to numerically and optimal but gives the worn values
#define		FLAG_SS_31		BIT(42)		///< This is the standard sum and search kernel, there were others but they were deprecated
#define		FLAG_SS_INMEM		BIT(43)		///< Do an in memory GPU search
#define		FLAG_SS_STG		( FLAG_SS_00| FLAG_SS_31 /* | FLAG_SS_32 /* | FLAG_SS_30 */ )
#define		FLAG_SS_KERS		( FLAG_SS_STG | FLAG_SS_INMEM )
#define		FLAG_SS_ALL		( FLAG_SS_CPU | (FLAG_SS_KERS) )

#define		FLAG_Z_SPLIT		BIT(44)		///< Split the f-fdot plane into top and bottom sections
#define		FLAG_SS_COUNT		BIT(45)		///< Count initial candidates in kernel and write to memory
#define		FLAG_SEPSRCH		BIT(46)		///< Create a separate second output location for the search output - Generally because the complex plane is smaller than return data
#define		FLAG_STORE_ALL		BIT(47)		///< Store candidates for all stages of summing, default is only the final result
#define		FLAG_CAND_THREAD	BIT(48)		///< Use separate CPU threads to search for candidates in returned data
#define		FLAG_CAND_MEM_PRE	BIT(49)		///< Create a thread specific section of temporary memory and copy results to it before spawning the thread - Else just use the pinned memory of the ring buffer

// ------------ Optimisation -------//	\\ 10 - 49 - Optimisation only //

#define		FLAG_OPT_NM		BIT(10)		///< Use particle swarm to optimise candidate location
#define		FLAG_OPT_SWARM		BIT(11)		///< Use particle swarm to optimise candidate location
#define		FLAG_OPT_ALL		( FLAG_OPT_NM | FLAG_OPT_SWARM )

#define		FLAG_OPT_NRM_LOCAVE	BIT(13)		///< Use local average normalisation instead of median in the optimisation
#define		FLAG_OPT_NRM_MEDIAN1D	BIT(14)		///< Use local 1D Median
#define		FLAG_OPT_NRM_MEDIAN2D	BIT(15)		///< Use local 2D Median
#define		FLAG_OPT_NRM_ALL	( FLAG_OPT_NRM_LOCAVE | FLAG_OPT_NRM_MEDIAN1D | FLAG_OPT_NRM_MEDIAN2D )

#define		FLAG_OPT_BEST		BIT(20)		///<
#define		FLAG_OPT_DYN_HW		BIT(21)		///< Use Dynamic half-width in optimisation
#define		FLAG_OPT_THREAD		BIT(22)		///< Use separate CPU threads for CPU component of optimisation

//		NO_VALUE				///< Dimensions will be leaf exactly as they are (this implies slower points kernel)
#define		FLAG_RES_CLOSE		BIT(17)		///< Size and resolution will match or exceed that specified but could run a bit slower than optimal
#define		FLAG_RES_FAST		BIT(18)		///< Size may be slightly smaller but will usually run faster
#define		FLAG_RES_ALL		( FLAG_RES_CLOSE | FLAG_RES_FAST )

#define		FLAG_OPT_CPU_PLN	BIT(24)		///< Use CPU calculation - Testing only

//		NO_VALUE				///< Auto determine ie. use shuffle
#define		FLAG_OPT_BLK_HRM	BIT(27)		///< Share calc common coefficients once use running sum per location - starts to spill at 8 which is a bit low
#define		FLAG_OPT_BLK_SFL	BIT(29)		///< Share coefficients using shuffle - sums held by threads - Fates - limited to power of two's  (or multiples of powers of two)
#define		FLAG_OPT_BLK		( FLAG_OPT_BLK_HRM | FLAG_OPT_BLK_SFL )

//		NO_VALUE				///< Auto determine
#define		FLAG_OPT_PTS_HRM	BIT(32)		///< Standard fastest - Thread per point per harmonic point
#define		FLAG_OPT_PTS		( FLAG_OPT_PTS_HRM )

#define		FLAG_OPT_KER_ALL	( FLAG_OPT_BLK | FLAG_OPT_PTS )

#define		FLAG_PLN_ALL		( FLAG_OPT_KER_ALL | FLAG_RES_ALL )

// ------------ Debug -------------//	\\ 50 - 63  - COMMOM //

#define		FLAG_PROF		BIT(55)		///< Record and report timing for the various components in the search, this should only be used with FLAG_SYNCH
#define		FLAG_SYNCH		BIT(56)		///< Run the search in synchronous mode, this is slow and should only be used for testing
#define		FLAG_DPG_PRNT_CAND	BIT(57)		///< Print candidates to file
#define		FLAG_DPG_SKP_OPT	BIT(58)		///< Skip optimisation stage
#define		FLAG_DPG_PLT_OPT	BIT(59)		///< Plot optimisation stages
#define		FLAG_DPG_PLT_POWERS	BIT(60)		///< Plot powers

#define		FLAG_DPG_CAND_PLN	( FLAG_DPG_PRNT_CAND | FLAG_DPG_PLT_OPT )

#define		FLAG_DBG_TEST_1		BIT(61)		///< Test 1
#define		FLAG_DBG_TEST_2		BIT(62)		///< Test 2
#define		FLAG_DBG_TEST_3		BIT(63)		///< Test 3
#define		FLAG_DBG_TEST_ALL	( FLAG_DBG_TEST_1 | FLAG_DBG_TEST_2 | FLAG_DBG_TEST_3 )

//#define		FLAG_RAND_1		BIT(59)		///< Random Flag 1

/********************************** data types identifiers **********************************************/

// ----------- This is a list of the data types that and storage structures


// ----------- This is ??????

#define     HAVE_INPUT          	BIT(1)
#define     HAVE_MULT           	BIT(2)
#define     HAVE_PLN            	BIT(3)          ///< The Plane data is ready to search
#define     HAVE_SS             	BIT(4)          ///< The S&S is complete and the data is read to read
#define     HAVE_RES            	BIT(5)          ///< The S&S is complete and the data is read to read


/******************************************* enums ******************************************************/

typedef enum					///< CU_TYPE
{
  CU_NONE		=	0,
  CU_CMPLXF		=	BIT(1),		///< Complex float
  CU_INT		=	BIT(2),		///< INT
  CU_HALF		=	BIT(3),		///< 2 byte float
  CU_FLOAT		=	BIT(4),		///< Float
  CU_DOUBLE		=	BIT(5),		///< Double
  CU_POWERZ_S		=	BIT(6),		///< A value and a z bin         candPZs
  CU_POWERZ_I		=	BIT(7),		///< A value and a z bin         candPZi
  CU_CANDMIN		=	BIT(8),		///< A compressed candidate      candMin
  CU_CANDSMAL		=	BIT(9),		///< A compressed candidate      candSml
  CU_CANDBASC		=	BIT(10),	///< A compressed candidate      accelcandBasic
  CU_CANDFULL		=	BIT(11),	///< Full detailed candidate     cand
  CU_POWERH_S		=	BIT(12),	///< A value and a z bin         candHs
  CU_CMPLXD		=	BIT(13),	///< Complex double

  CU_STR_ARR		=	BIT(20),	///< Candidates are stored in an array (requires more memory)
  CU_STR_PLN		=	BIT(21),	///< Plane 2D?
  CU_STR_LST		=	BIT(22),	///< Candidates are stored in a list  (usually a dynamic linked list)
  CU_STR_QUAD		=	BIT(23),	///< Candidates are stored in a dynamic quadtree
  CU_STR_HARMONICS	=	BIT(24),	///< Stored in an expanded planes (1 value per harmonic)
  CU_STR_INCOHERENT_SUM	=	BIT(25)
} CU_TYPE;

#define		CU_TYPE_ALLL	(CU_CMPLXF | CU_INT | CU_HALF | CU_FLOAT | CU_DOUBLE |CU_POWERZ_S | CU_POWERZ_I | CU_POWERH_S | CU_CANDMIN | CU_CANDSMAL | CU_CANDBASC | CU_CANDFULL | CU_POWERH_S | CU_CMPLXD )
#define		CU_SRT_ALL	(CU_STR_ARR | CU_STR_PLN | CU_STR_LST | CU_STR_QUAD )

typedef enum					///< PLN_MEM
{
  NONE			=	0,
  INPUT			=	BIT(1),		///< Input	-
  COMPLEX		=	BIT(2),		///< Complex	- Between multiplication and iFFT
  POWERS		=	BIT(3),		///< Powers	- Between iFFT and SAS
  OUTPUT		=	BIT(4),		///< Output	- After SAS
  IM_PLN		=	BIT(5),		///< in-memory plane
} PLN_MEM;

typedef enum					/// FFT type
{
  FFT_INPUT,					//!< FFT_INPUT
  FFT_PLANE,					//!< FFT_PLANE
  FFT_BOTH  					//!< FFT_BOTH
} presto_fft_type;

typedef enum					///< ACC_ERR_CODE
{
  ACC_ERR_NONE		=	(0),		///< No error
  ACC_ERR_NAN		=	BIT(0),		///<
  ACC_ERR_NEG		=	BIT(1),		///<
  ACC_ERR_STRIDE	=	BIT(2),		///<
  ACC_ERR_ALIGHN	=	BIT(3),		///<
  ACC_ERR_OVERFLOW	=	BIT(4),		///<
  ACC_ERR_OUTOFBOUNDS	=	BIT(5),		///<
  ACC_ERR_NULL		=	BIT(6),		///< Null pointer
  ACC_ERR_INVLD_CONFIG	=	BIT(7),		///< Invalid configuration
  ACC_ERR_UNINIT	=	BIT(8),		///< Uninitialised
  ACC_ERR_MEM		=	BIT(9),		///< Problem with memory
  ACC_ERR_DATA_TYPE	= 	BIT(10),	///< Data type ...
  ACC_ERR_COMPILED	= 	BIT(11),	///< Broken because of compile
  ACC_ERR_DEV		= 	BIT(12),	///< Under development
  ACC_ERR_CU_CALL	= 	BIT(13),	///< CUDA CALL
  ACC_ERR_SIZE		= 	BIT(13),	///< Size
  ACC_ERR_DEPRICATED	= 	BIT(14),	///< Deprecated
  // If you add to this add text description to '__printErrors'
} acc_err;

inline acc_err operator ~(acc_err a)
{
    return static_cast<acc_err>(~static_cast<int>(a));
}

inline acc_err operator |(acc_err a, acc_err b)
{
    return static_cast<acc_err>(static_cast<int>(a) | static_cast<int>(b));
}

inline acc_err& operator |=(acc_err& a, acc_err b)
{
    return a= a | b;
}

inline bool operator ==(acc_err a, acc_err b)
{
    return (a & b) > 0 ? 1 : 0 ;
}

inline acc_err operator &(acc_err a, acc_err b)
{
    return static_cast<acc_err>(static_cast<int>(a) & static_cast<int>(b));
}

inline acc_err& operator &=(acc_err& a, acc_err b)
{
    return a= a & b;
}

inline acc_err& operator +=(acc_err& a, acc_err b)
{
    return a= a | b;
}

inline acc_err& operator -=(acc_err& a, acc_err b)
{
    return a= a & ~b;
}



inline CU_TYPE operator ~(CU_TYPE a)
{
    return static_cast<CU_TYPE>(~static_cast<int>(a));
}

inline CU_TYPE operator |(CU_TYPE a, CU_TYPE b)
{
    return static_cast<CU_TYPE>(static_cast<int>(a) | static_cast<int>(b));
}

inline CU_TYPE& operator |=(CU_TYPE& a, CU_TYPE b)
{
    return a= a | b;
}

inline bool operator ==(CU_TYPE a, CU_TYPE b)
{
    return (a & b) > 0 ? 1 : 0 ;
}

inline CU_TYPE operator &(CU_TYPE a, CU_TYPE b)
{
    return static_cast<CU_TYPE>(static_cast<int>(a) & static_cast<int>(b));
}

inline CU_TYPE& operator &=(CU_TYPE& a, CU_TYPE b)
{
    return a= a & b;
}

inline CU_TYPE& operator +=(CU_TYPE& a, CU_TYPE b)
{
    return a= a | b;
}

inline CU_TYPE& operator -=(CU_TYPE& a, CU_TYPE b)
{
    return a= a & ~b;
}

/******************************************* enums ******************************************************/

#define  TIME_CPU_INIT		0			/// CPU - Initialisation
#define  TIME_GPU_INIT		1			/// GPU - Initialisation
#define  TIME_ALL_SRCH		2			/// CPU & GPU - Initialisation and Candidate Generation
#define  TIME_GPU_SRCH		3			/// GPU - Initialisation & Generation stages & Candidate copy and clear memory
#define  TIME_CPU_SRCH		4			/// CPU - Initialisation & Generation stage
#define  TIME_CPU_CND_GEN	5			/// CPU - Candidate generation stage
#define  TIME_GPU_CND_GEN	6			/// GPU - Candidate generation stage - Includes the time to copy initial candidates [TIME_CND]
#define  TIME_CPU_OPT		7			/// CPU - Candidate optimisation stage
#define  TIME_GPU_OPT		8			/// GPU - Candidate optimisation stage
#define  TIME_ALL_OPT		9			///     - All Optimisation (duplicates, CPU and GPU refine, writing results to file)

#define  TIME_CONTEXT		10			/// Time for CUDA context initialisation
#define  TIME_PREP		11			/// CPU preparation - parse command line, read, (FFT), and normalise input
#define  TIME_GPU_PLN		12			/// GPU - Plane generation & Sum & Search in standard search
#define  TIME_GPU_SS		13			/// GPU - Sum and search (only in in-mem) of full plane (Not including plane creation)
#define  TIME_GPU_REFINE	14			/// GPU - Candidate refine and properties
#define  TIME_CPU_REFINE	15			/// CPU - Candidate refine and properties
#define  TIME_CND		16			/// GPU - Time to copy candidates from GPU data structure to list for optimisation
#define  TIME_GEN_WAIT		17			/// Time waited for CPU threads to complete in generation stage
#define  TIME_OPT_ASYNCH	18			/// Time to run the asynchronous GPU optimisation
#define  TIME_OPT_WAIT		19			/// Time waited for CPU threads to complete in optimisation stage
#define  TIME_OPT_FILE_WRITE	20			///     - Write candidates to file

#define  COMP_RESP		21			/// Initialisation - response function calculations
#define  COMP_KERFFT		22			/// Convolution kernel FFT's
#define  COMP_OPT_REFINE_1	23			/// First round of candidate position refinement - CPU Simplex or GPU planes
#define  COMP_OPT_REFINE_2	24			/// Second round of candidate position refinement - Small scale CPU Simplex usual run is separate CPU thread
#define  COMP_OPT_DERIVS	25			/// Calculate candidate derivatives - run is separate CPU thread

#define  COMP_MAX		30			/// Nothing - A value to indicate the maximum array length


#define  NO_STKS		(plan->noStacks)	/// A value used to index, components values

#define  COMP_GEN_H2D		0			///
#define  COMP_GEN_CINP		1			/// GPU input stuff - Mem copies, Normalisation and FFT
#define  COMP_GEN_GINP		2			/// GPU input stuff - The time for the CPU thread to call all the GPU input stuff this potently included: Normalisation, FFTs, Mem copies
#define  COMP_GEN_NRM		3			///
#define  COMP_GEN_MEM		4			/// Input preparation: Stack0: Zeroing host buffer, Stack1: Copy input FFT to buffer, Stack2: Copy buffer over pinned
#define  COMP_GEN_FFT		5			/// Input FFT
#define  COMP_GEN_MULT		6			///
#define  COMP_GEN_IFFT		7			///
#define  COMP_GEN_D2D		8			///
#define  COMP_GEN_SS		9			///
#define  COMP_GEN_D2H		10			///
#define  COMP_GEN_STR		11			/// Initial candidate storage and sigma calculations - Stack0: Sigma calcs and data saves, Stack1: memcpy, Stack2: Allocate mem and init data struct
#define  COMP_GEN_END		12			/// Nothing - A value to indicate the end of the used variables
#define  COMP_GEN_MAX		20			/// Nothing - A value to indicate the maximum array length

// Optimisation components
#define  COMP_OPT_NRM		0			/// Normalising input
#define  COMP_OPT_H2D		1			/// Copying data to the device
#define  COMP_OPT_PLN1		2			/// Generating plane 1
#define  COMP_OPT_PLN2		3			/// Generating plane 2
#define  COMP_OPT_PLN3		4			/// Generating plane 3
#define  COMP_OPT_PLN4		5			/// Generating plane 4
#define  COMP_OPT_PLN5		6			/// Generating plane 5
#define  COMP_OPT_PLN6		7			/// Generating plane 6
#define  COMP_OPT_PLN7		8			/// Generating plane 7
#define  COMP_OPT_D2H		9			/// Copy results back from device
#define  COMP_OPT_NM1		10			/// First Simplex
#define  COMP_OPT_NM2		11			/// Second refining Simplex
#define  COMP_OPT_DRV		12			/// Calculate derivatives
#define  COMP_OPT_END		13			///
#define  COMP_OPT_MAX		20			/// Nothing - A value to indicate the maximum array length


/****************************************** Macros ******************************************************/

///< Defines for safe calling usable in C
#define CUDA_SAFE_CALL(value, format... )	__cuSafeCall		(value, __FILE__, __LINE__, format )
#define CUFFT_SAFE_CALL(value, format... )	__cufftSafeCall		(value, __FILE__, __LINE__, format )
#define EXIT_DIRECTIVE(flag)			__exit_directive	(	__FILE__, __LINE__, flag   )
#define ERROR_MSG(value, format... )		__printErrors		(value, __FILE__, __LINE__, format )
#define CUDA_ERR_CALL(value, format... )	__cuErrCall		(value, __FILE__, __LINE__, format )
#define SAFE_CALL(value, format... )		if(__printErrors	(value, __FILE__, __LINE__, format ))  exit(EXIT_FAILURE) ;

#ifdef	TIMING
  #define TIME if(1)			//< A macro used to encapsulate timing code, if TIMING is not defined all timing code should be omitted at compile time
#else
  #define TIME if(0)			//< A macro used to encapsulate timing code, if TIMING is not defined all timing code should be omitted at compile time
#endif

#ifdef PROFILING
  // This macro will allow blocks to contain profiling code, if profiling is disabled this will become if(0) and the compiler SHOULD omit it at build time
  #define PROF if(1)
#else
  // Compiler should ship all profiling blocks at compile time
  #define PROF if(0)
#endif


#ifdef NVVP

// Includes for CUDA profiler (nvvp)
#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>
#include <cuda_profiler_api.h>

//extern int cnttt;

// Macro for adding CUDA ranges
#define NV_RANGE_POP(x)		nvtxRangePop(); 		//infoMSG(7,7,"POP,  %2i  %s\n", cnttt, x ); cnttt--;
#define NV_RANGE_PUSH(x)	nvtxRangePush(x); 		//++cnttt; infoMSG(7,7,"PUSH %2i  %s \n", cnttt, x );
#define NV_NAME_STREAM(x,y)	nvtxNameCudaStreamA(x,y)	// 

#else

#define NV_RANGE_POP(x)
#define NV_RANGE_PUSH(x)
#define NV_NAME_STREAM(x,y)

#endif


/************************************** Global variables ************************************************/

extern int    useUnopt;                                                         /// Use a saved text list of candidates this is used in development for optimising the optimisation stage
extern int    msgLevel;                                                         /// The level of debug messages to print, 0 -> none  higher results in more messages

/************************************* Struct prototypes ************************************************/

typedef struct cuSearch cuSearch;
typedef struct resThrds resThrds;

/**************************************** Type defines **************************************************/

///< A complex float in device texture memory
typedef cudaTextureObject_t fCplxTex;

///< Basic accel search candidate to be used in CUDA kernels
///< Note this may not be the best choice on a GPU as it has a bad size
typedef struct candHs
{
    half            value;              ///< This can be summed power or the sigma there of
    short           z;                  ///< Fourier f-dot of first harmonic
} candHs;

///< Basic accel search candidate to be used in CUDA kernels
///< Note this may not be the best choice on a GPU as it has a bad size
typedef struct candPZs
{
    float           value;              ///< This can be summed power or the sigma there of
    short           z;                  ///< Fourier f-dot of first harmonic
} candPZs;

///< Basic accel search candidate to be used in CUDA kernels
///< Note this may not be the best choice on a GPU as it has a bad size
typedef struct candPZi
{
    float           value;              ///< This can be summed power or the sigma there of
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
    int             noSegments;         ///<  The Number of segments in the stack
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
    long long	firstBin;		///< The FFT bin index of the first memory location
    long long	lastBin;		///< The FFT bin index of the last  memory location
    long long	noBins;			///< The number of bins in the memory location

    long long	N;			///< The number of bins in the FT

    double	dt;			///< Data sample length (s)
    double	T;			///< Total observation length

    fcomplex*	data;			///< The array of complex numbers (nor long)
} fftInfo;

typedef struct candOpt
{
    float           power;
    double          r;
    double          z;
} candOpt;

/** Details of a GPU  .
 */
typedef struct gpuInf
{
    int     devid;                      ///<
    int     alignment;                  ///< The alignment of memory, in bytes
    float   capability;                 ///<
    char*   name;                       ///<
} gpuInf;

/** Used in optimisation
 *
 */
typedef struct cuHarmInput
{
  double	maxZ;			///< The largest  fundamental Z value the input can handle
  double	minZ;			///< The smallest fundamental Z value the input can handle
  double	maxR;			///< The largest  fundamental R value the input can handle
  double	minR;			///< The smallest fundamental R value the input can handle
  int 		maxHalfWidth;		///< The maximum halfwidth (at the highest harmonic) this data can handle

  int		noHarms;		///< The current number of harmonics in the data set
  int		stride;			///< The current stride (harmonic) of the input elements, (measured in fcomplexcu, ie. float2 ) - Used for both host and device
  size_t	size;			///< The size, in bytes, of the full input data (the size of h_inp & d_inp)

  fcomplexcu*	h_inp;			///< A pointer to host memory size bytes big
  fcomplexcu*	d_inp;			///< A pointer to device memory size bytes big

  gpuInf*	gInf;			///< This can handle GPU memory so lets have some info on the device

  int		loR[16];		///< The bin index of the first memory location
  double	norm[16];		///< The normalisation factor used - TODO: CHeck if this is for powers or input
} cuHarmInput;

//------------- Data structures for, planes, stacks, plans etc ----------------

/** The general information of a f-∂f plane  .
 * NOTE: This is everything that is not specific to a particular plane
 */
typedef struct cuHarmInfo
{
    size_t		noZ;				///< The number of rows (Z's)
    float		zmax;				///< The maximum (and minimum) z
    double		zStart;				///< The z value of the first "row" in memory
    double		zEnd;				///< The z value of the last "row" in memory
    int			halfWidth;			///< The kernel half-width         - in input dfft units ie needs to be multiply by noResPerBin to get plane units
    int			plnStart;			///< The starting point for data in the various planes, this is essentially the largest halfwidth in the stack plus some potently padding to centre and align values
    int			requirdWidth;			///< The width of the uncontaminated values

    size_t		width;				///< The number of columns, including the contaminated ends (this should always be a power of 2)
    int			noResPerBin;			///< The number of points sampled at

    double		harmFrac;			///< The harmonic fraction
    int			stackNo;			///< Which Stack is this plane in. (0 indexed at starting at the widest stack)

    int			yInds;				///< The offset of the y offset in constant memory
    int			stageIndex;			///< The index of this harmonic in the staged order
} cuHarmInfo;

/** The complex convolution kernels of a f-∂f plane  .
 */
typedef struct cuKernel
{
    cuHarmInfo*		harmInf;			///< A pointer to the harmonic information for this kernel
    int			kreOff;				///< The offset of the first column of the kernel in the stack kernel
    int			stride;				///< The stride of the data in the kernel
    int			type;				///< The data type of the kernel data (float or double)
    void*		d_kerData;			///< A pointer to the first kernel element (Width, Stride and height determined by harmInf)
    fCplxTex		kerDatTex;			///< A texture holding the kernel data
} cuKernel;

/** A f-∂f plane  .
 * This could be a fundamental or harmonic
 * it holds basic information no memory addresses
 */
typedef struct cuFFdot
{
    cuHarmInfo*		harmInf;			///< A pointer to the harmonic information for this planes
    cuKernel*		kernel;				///< A pointer to the kernel for this plane

    // pointers to device data
    void*		d_planeCplx;			///< A pointer to the first element of the complex f-∂f plane (Width, Stride and height determined by harmInf)
    void*		d_planePowr;			///< A pointer to the powers for this stack
    fcomplexcu*		d_iData;			///< A pointer to the input data for this plane this is a section of the 'raw' complex fft data, that has been Normalised, spread and FFT'd

    // Texture objects
    fCplxTex		datTex;				///< A texture holding the kernel data  [ Depreciated ]
    fCplxTex		powerTex;			///< A texture of the power data        [ Depreciated ]
} cuFFdot;

/** A stack of f-∂f planes that all have the same FFT width  .
 */
typedef struct cuFfdotStack
{
    ////////////////// Stack parameters \\\\\\\\\\\\\\\\\\

    int			noInStack;			///< The number of planes in this stack
    int			startIdx;			///< The family index the first plane of the stack
    int			stackIdx;			///< The index of the stack in the batch
    size_t		width;				///< The width of  the entire stack, for one segment [ in complex numbers! ]
    size_t		height;				///< The height of the entire stack, for one segment
    size_t		kerHeigth;			///< The height of the multiplication kernel for this stack (this is equivalent to the height of the largest plane in the stack)

    ////////////////// sub-structures \\\\\\\\\\\\\\\\\\

    // Sub data structures associated with this stack
    cuFFdot*		planes;				///< A pointer to all the pains for this stack
    cuHarmInfo*		harmInf;			///< A pointer to all the harmonic info's for this stack
    cuKernel*		kernels;			///< A pointer to all the convolution kernels for this stack

    ////////////////// Search parameters \\\\\\\\\\\\\\\\\\

    int64_t		flags;				///< CUDA accel search bit flags

    int			mulSlices;			///< The number of slices to do multiplication with
    int			mulChunk;			///< The Sum and search chunk size

    // CUFFT details
    cufftHandle		plnPlan;			///< A cufft plan to fft the entire stack
    cufftHandle		inpPlan;			///< A cufft plan to fft the input data for this stack

    // FFTW details
    fftwf_plan		inpPlanFFTW;			///< A FFTW plan to fft the input data for this stack

    int startZ[MAX_IN_STACK];				///< The y 'start' of the planes in this stack - assuming one segment

    ////////////////// Memory information \\\\\\\\\\\\\\\\\\

    size_t		strideCmplx;			///< The stride of the block of memory  [ in complex numbers! ]
    size_t		stridePower;			///< The stride of the powers

    fcomplexcu*		h_iBuffer;			///< Pointer to host memory to do CPU "work" on the Input data for the CG plan, this allows delaying the input data synchronisation
    fcomplexcu*		h_iData;			///< Paged locked input data for this stack
    fcomplexcu*		d_iData;			///< Device       input data for this stack

    void*		d_planeCplx;			///< Plane of complex data for multiplication
    void*		d_planePowr;			///< Plane of float data for the search

    stackInfo*		d_sInf;				///< Stack info structure on the device (usually in constant memory)
    int			stkConstIdx;			///< The index of this stack in the constant device memory list of stacks

    ////////////////// Asynchronous CUDA information \\\\\\\\\\\\\\\\\\

    // Streams
    cudaStream_t	initStream;			///< CUDA stream for work on input data for the stack
    cudaStream_t	inptStream;			///< CUDA stream for work on input data for the stack
    cudaStream_t	fftIStream;			///< CUDA stream to CUFFT the input data
    cudaStream_t	multStream;			///< CUDA stream for work on the stack
    cudaStream_t	fftPStream;			///< CUDA stream for the inverse CUFFT the plane

    // CUDA Texture
    fCplxTex		kerDatTex;			///< A texture holding the kernel data

    // CUDA Events
    cudaEvent_t		normComp;			///< Normalisation of input data
    cudaEvent_t		inpFFTinitComp;			///< Preparation of the input data complete
    cudaEvent_t		multComp;			///< Multiplication complete
    cudaEvent_t		ifftComp;			///< Creation (multiplication and FFT) of the complex plane complete
    cudaEvent_t		ifftMemComp;			///< IFFT memory copy

    // CUDA Profiling events
    cudaEvent_t		normInit;			///< Multiplication starting
    cudaEvent_t		inpFFTinit;			///< Start of the input FFT
    cudaEvent_t		multInit;			///< Multiplication starting
    cudaEvent_t		ifftInit;			///< Start of the inverse FFT
    cudaEvent_t		ifftMemInit;			///< IFFT memory copy start

} cuFfdotStack;

/** Details of the segments of the input FFT  .
 */
typedef struct rVals
{
    int			segment;			///< The index of the DFT segment these r values cover
    int			iteration;			///< Iteration - in the candidate generation loop
    double		drlo;				///< The value of the first usable bin of the plane (the start of the segment). Note: this could be a fraction of a bin (Fourier interpolation)
    double		drhi;				///< The value of the first usable bin of the plane (the start of the segment). Note: this could be a fraction of a bin (Fourier interpolation)
    long long		lobin;				///< The first bin to copy from the the input FFT ( serachR scaled - halfwidth )
    long		numdata;			///< The number of input FFT points to read
    long		numrs;				///< The number of good values in the plane ( expanded units ) NOTE: This is used to denote an active "section' if this is set to 0 many of the processes wont run on
    long long		expBin;				///< The index of the expanded bin of the first good value
    double		norm;				///< The normalisation factor used to normalise the input - Not always set

    void*		h_outData;			///< A section of pinned host memory to store the search results in
    int 		noBlocks;			///< The number of thread blocks used to search the data (each one returns a count of candidates found)
    bool		outBusy;			///< A flag to show a thread is still using the output memory
} rVals;

/** Details of the number of bins of the full search  .
 */
typedef struct searchSpecs
{
    int 		noHarms;			///< The number of harmonics to sum in the search
    int			noHarmStages;			///< The number of stages of harmonic summing to use in the search
    double		zMax;				///< The highest z drift of the fundamental plane

    float		sigma;				///< The cut off sigma

    double		specRLow;			///< The user specified input r bin to start the search at
    double		specRHigh;			///< The user specified input r bin to end   the search at

    double		searchRLow;			///< The value of the input r bin to start the search at
    double		searchRHigh;			///< The value of the input r bin to end   the search at

    long long		rLow;				///< The lowest  possible R this search could find, Including halfwidth, thus may be less than 0
    long long		rHigh;				///< The highest possible R this search could find, Including halfwidth

    unsigned long long	noInpR;				///< The maximum number of r input ( this is essentially  (rHigh - rLow) ) and me be longer than fft length because of halfwidth this requires the FFT to be padded!
    unsigned long long	noSearchR;			///< The maximum number of FFT bins ( of the input FT ) covered by the search
} searchSpecs;

/** Configuration parameters for the candidate generation stage  .
 *
 */
typedef struct confSpecsCG
{
    int			noResPerBin;			///< The number of response values per bin of the input fft - this allows "over sampling" the standard value is 2 interbinning (has to be an int)

    double		zMax;				///< The highest z drift of the fundamental plane
    double		zRes;				///< The resolution in the z dimension

    float		inputNormzBound;		///< The boundary z-max to swap over to CPU Normalisation	Not used if set < 0 - default is not used
    float		inputFFFTzBound;		///< The boundary z-max to swap over to CPU FFT's		Not used if set < 0 - default is not used

    int			planeWidth;			///< The desired width of the planes
    int			ssSegmentSize;			///< The size of the segments to break the in-memory plane in to for harmonic summing and searching - Bigger is generally better

    int64_t		flags;				///< The search bit flags specified by the user, the actual bit flag used in the search will be different
    int			normType;			///< The type of normalisation to do

    int			mulSlices;			///< The number of multiplication slices
    int			ssSlices;			///< The number of Sum and search slices

    int			ssSliceMin;			///< The minimum width (in z) of a slice of the sum and search kernels
    int			mulSliceMin;			///< The minimum width (in z) of a slice of the convolution kernels

    int			ssChunk;			///< The multiplication chunk size
    int			mulChunk;			///< The Sum and search chunk size

    int			ssColumn;			///< The number of sum and search columns

    int			ringLength;			///< The number of elements in the results ring buffer
    int			cndProcessDelay;		///< The number of elements in the results ring buffer

    int			retType;			///< The type of output
    int			cndType;			///< The type of output
    float		candRRes;			///< The resolution of the candidate array ( measured in input FT bins')

    void*		outData;			///< A pointer to the location to store candidates

} confSpecsGen;

/** Configuration parameters for the candidate optimisation stage  .
 *
 */
typedef struct confSpecsCO
{
    float		zScale;				///< The ratio between spacing in R and Z in the optimisation planes

    int			optResolution;			///< The number of r points per fft bin to use in the preallocated initial position optimisation

    int			optMinLocHarms;			///< The minimum number of harmonics to localise on
    int			optMinRepHarms;			///< The minimum number of harmonics report on

    int 		blkDivisor;			///< Make blocks of points divisible by this - this is related to warp size and should be 4, 8, 16 or 32
    int 		blkMax;				///< The maximum number of columns to use, this can reduce register pressure

    int			nelderMeadReps;			///< The number of final, double precision high accuracy, Nelder-Mead refinements to do - 0 dose no additional optimisation
    double		nelderMeadDelta;		///< The delta to stop the NM search at

    int			optPlnSiz[MAX_NO_STAGES];	///< The size of optimisation planes
    float		optPlnScale;			///< The ratio by which to decrease the optimisation planes (I like 10)

    int			optPlnDim[NO_OPT_LEVS];		///< The size of optimisation planes
    CU_TYPE		optPlnPrec[NO_OPT_LEVS];	///< The precision of the levels
    presto_interp_acc	optPlnAccu[NO_OPT_LEVS];	///< The accuracy precision

    int64_t		flags;				///< The search bit flags specified by the user, the actual bit flag used in the search will be different
} confSpecsOpt;

/** Configuration parameters  .
 *
 */
typedef struct confSpecs
{
    confSpecsCG*	gen;				///< Configuration specifications of the candidate generation
    confSpecsCO*	opt;				///< Configuration specifications of the candidate optimisation
} confSpecs;

/** User specified GPU search details  .
 */
typedef struct gpuSpecs
{
    int         noDevices;                      ///< The number of devices (GPU's to use in the search)
    int         devId[MAX_GPUS];                ///< A list noDevices long of CUDA GPU device id's
    int         noCgPlans[MAX_GPUS];            ///< A list noDevices long of the number of CG plans on each device
    int         noSegments[MAX_GPUS];           ///< A list noDevices long of the number of segments each device wants to use
    int         noCoPlans[MAX_GPUS];             ///< A list noDevices long of the number of CO plans each device wants to do
    gpuInf      devInfo[MAX_GPUS];              ///< A list noDevices long of basic information of the GPU

    pthread_t   cntxThread;                     ///< A pthread to initialise the CUDA context in
    long long   nctxTime;                       ///< The amount of time it took to initialise the cuda contexts
} gpuSpecs;


/** A CG plan data structure holding the data and configuration to process a single iteration/batch
 * of the main CG loops.
 */
typedef struct cuCgPlan
{
    cuSearch*       	cuSrch;           	///< A pointer to the parent search info
    confSpecsCG*	conf;			///< Configuration - NB: This is a duplicate of the search configuration and should not be edited manually if the search configuration is edited the multiplication kernel and plans should be recreated!
    gpuInf*		gInf;			///< GPU information for the plan

    ////////////////// Batch parameters \\\\\\\\\\\\\\\\\\

    // Batch specific info
    int             	noStacks;		///< The number of stacks in this batch
    int             	noHarmStages;		///< The number of stages of harmonic summing

    int             	noGenHarms;		///< The number of harmonics in the family
    int             	noSrchHarms;		///< The number of harmonics in the family

    int             	noSegments;		///< The number of segments of the DFT processed by the plan
    uint            	noResults;		///< The number of results from the previous search
    int             	srchMaster;		///< Weather this is the master CO plan
    int             	isKernel;		///< Weather this is the master CO plan

    ////////////////// sub-structures \\\\\\\\\\\\\\\\\\

    // Pointers to sub-structures
    cuFfdotStack*   	stacks;			///< A list of the stacks
    cuFFdot*        	planes;			///< A list of the planes
    cuKernel*       	kernels;		///< A list of the kernels
    cuHarmInfo*     	hInfos;			///< A list of the harmonic information

    ////////////////// Search parameters \\\\\\\\\\\\\\\\\\

    // Bit flags
    int             	retType;		///< The type of output
    int             	cndType;		///< The type of output
    int64_t         	flags;			///< CUDA accel search bit flags

    // Plan-specific search parameters
    int             	mulSlices;		///< The number of slices to do multiplication with
    int             	ssSlices;		///< The number of slices to do sum and search with
    int             	ssChunk;		///< The Sum and search chunk size
    int             	ssColumn;		///< The Sum and search number of columns
    int             	mulChunk;		///< The Sum and search chunk size

    // Plan-independent search parameters
    size_t		accelLen;		///< The size to step through the input fft to generate the plane (This is segment size)

    // Stride information (only the results are specific to the CO plan)

    // Data sizes in bytes
    size_t		inptDataSize;		///< The size of the input data memory in bytes
    size_t		candDataSize;		///< The size of the candidates - This excludes the extra bit for candidate counts
    size_t		retnDataSize;		///< The size of data to return in bytes - This is candDataSize + a bit extra for returned values
    size_t		cmlxDataSize;		///< The size of the complex plane data memory in bytes
    size_t		powrDataSize;		///< The size of the powers  plane data memory in bytes
    size_t		kernDataSize;		///< The size of the convolution kernel memory in bytes

    // Stride information (only the results are specific to the batch)

    uint            	strideOut;		///< The stride of the returned candidate data - The stride of one segment

    fcomplexcu*     	h_iBuffer;		///< Pointer to host memory to do CPU "work" on the Input data for the CO plan
    fcomplexcu*     	h_iData;		///< Pointer to page locked host memory of the input data for the CO plan
    fcomplexcu*     	d_iData;		///< Input data for the CO plan - NB: This could be a contiguous block of sections or all the input data depending on inpMethoud

    float*          	h_normPowers;		///< A array to store powers for running double-tophat local-power normalisation

    void*           	d_kerData;		///< Kernel data for all the stacks, generally this is only allocated once per device
    void*           	d_planeCplx;		///< Plane of complex data for multiplication
    void*           	d_planePowr;		///< Plane of float data for the search

    void*           	d_outData1;		///< The output
    void*           	d_outData2;		///< The output

    ////////////////// Segment information \\\\\\\\\\\\\\\\\\

    // Information on the input for the CO plan
    char            	noRArryas;		///< The number of r value arrays
    char            	rActive;		///< The index of the r-array we are working on
    rVals****       	rAraays;		///< Pointer to an array of 2D array [segment][harmonic] of the base expanded r index

    rVals*          	rArr1;			///< A pointer to the first value in a full flat list of r arrays used by the CO plan
    rVals*          	rArr2;			///< A pointer to the first value in a full flat list of r arrays used by the CO plan

    rVals***        	rArraysPlane;		///< Pointer to an array of 2D array [segment][harmonic] of the base expanded r index
    rVals***        	rArraysSrch;		///< Pointer to an array of 2D array [segment][harmonic] of the base expanded r index - TODO: I think I can deprecate this now?


    ////////////////// Asynchronous CUDA information \\\\\\\\\\\\\\\\\\

    // Streams
    cudaStream_t	inpStream;		///< CUDA stream for work on input data for the CO plan
    cudaStream_t	multStream;		///< CUDA stream for multiplication
    cudaStream_t	srchStream;		///< CUDA stream for summing and searching the data
    cudaStream_t	resStream;		///< CUDA stream for

    // CUDA Profiling events
    cudaEvent_t		iDataCpyInit;		///< Copying input data to device
    cudaEvent_t		multInit;		///< Start of CO plan multiplication
    cudaEvent_t		searchInit;		///< Sum & Search start
    cudaEvent_t		candCpyInit;		///< Finished reading candidates from the device

    // Synchronisation events
    cudaEvent_t		iDataCpyComp;		///< Copying input data to device
    cudaEvent_t		normComp;		///< Normalise and spread input data
    cudaEvent_t		multComp;		///< Sum & Search complete (candidates ready for reading)
    cudaEvent_t		searchComp;		///< Sum & Search complete (candidates ready for reading)
    cudaEvent_t		candCpyComp;		///< Finished reading candidates from the device
    cudaEvent_t		processComp;		///< Process candidates (usually done on CPU)

    // TIMING values
    long long*		compTime;		///< Array of floats from timing, one float for each stack

#if CUDART_VERSION >= 6050
    cufftCallbackLoadC	h_ldCallbackPtr;
    cufftCallbackStoreC	h_stCallbackPtr;

    cufftCallbackLoadC	h_ldCallbackPtr0;
    cufftCallbackLoadC	h_ldCallbackPtr1;
    cufftCallbackLoadC	h_ldCallbackPtr2;
    cufftCallbackLoadC	h_ldCallbackPtr3;
    cufftCallbackLoadC	h_ldCallbackPtr4;
#endif

} cuCgPlan;

/** A struct to keep info on all the kernels and plans to use in the CG stage of cuda accelsearch  .
 */
typedef struct cuCgInfo
{
    int			noDevices;		///< The number of devices (GPU's to use in the search)
    cuCgPlan*		kernels;		///< A list noDevices long of convolution kernels: These hold: basic info, the address of the convolution kernels on the GPU, the CUFFT plan.

    int			noCgPlans;		///< The total number of CG plans there across all devices
    cuCgPlan*		cgPlans;		///< A list noCgPlans long of CG plans: These hold: basic info, the address of the convolution kernels on the GPU, the CUFFT plan.

    int			noSegments;		///< The total segments in all CG plans - there are across all devices

    int*		devNoStacks;		///< An array of the number of stacks on each device
    stackInfo**		h_stackInfo;		///< An array of pointers to host memory for the stack info
} cuPlnInfo;

/** A structure to hold the details of a GPU plane of response function values - Passed to CUDA kernel  .
 *
 */
typedef struct cuRespPln
{
    float2*		d_pln;
    int			oStride;
    int			noRpnts ;
    int			halfWidth;
    double		zMax;
    double		dZ;
    int			noR;			///<
    int			noZ;			///< The number of elements in the z Direction
    size_t		size;			///< The size in bytes of the response plane
} cuRespPln;

/** Data structure to hold information of a section of f-fdot plane
 *
 */
typedef struct cuRzHarmPlane
{
    double		centR;			///< Centre of the plane
    double		centZ;			///< Centre of the plane
    double		rSize;			///< The width of the r plane
    double		zSize;			///< The width of the z plane

    double 		maxPower;		///< The min summed power of the plane
    double 		minPower;		///< The max summed power of the plane
    double 		maxBound;		///< The maximum summed power along the edge of the plane

    int			noZ;			///< The number of z - The "rows"     of the plane(s)
    int			noR;			///< The number or r - The "columns"  of the plane(s)

    int			noHarms;		///< The number of harmonics in the plane

    int			blkCnt;			///< The number of column(s) the plane cane be broken down into
    int			blkWidth;		///< Width of a single column, measured in R - This must be an integer because of the fractional part of r is the value and is shared across integer spaced values
    int			blkDimX;		///< The number of points in a single column

    int			zStride;		///< Stride of the r "columns"

    CU_TYPE		type;			///< The data type of the data in the plane

    size_t		resSz;			///< The size of the actual plane results (including striding)
    size_t		size;			///< The size in bytes of device output buffer
    void*		d_data;			///< Return data device
    void*		h_data;			///< Return data host
} cuRzHarmPlane;

/** A data structure to hold configurations needed generate a harmonically related section of rz plane
 *
 */
typedef struct cuPlnGen
{
    confSpecsCO*	conf;			///< Global configuration parameters
    gpuInf*		gInf;			///< Information on the GPU being used
    cuRzHarmPlane*	pln;			///< The ffdot plane section
    cuHarmInput*	input;			///< A pointer holding input data (unique to each plane)

    int64_t		flags;			///< CUDA accel configuration bit flags

    presto_interp_acc	accu;			///< Accuracy to create the plane at
    int			hw[32];			///< The halfwidth to use to generate each harmonic
    int			maxHalfWidth;		///< The maximum half-width of all harmonics

    cuRespPln*		responsePln;		///< A device specific plane holding possibly pre calculated response function values
    int			lftIdx;			///< X index of the pre calculated kernel
    int			topZidx;		///< Y index of the pre calculated kernel

    // Streams
    cudaStream_t	stream;			///< CUDA stream for work

    // Events
    cudaEvent_t		inpInit;		///< Start copying input data to device
    cudaEvent_t		inpCmp;			///< End   copying input data to device
    cudaEvent_t		compInit;		///< Start computation of plane
    cudaEvent_t		compCmp;		///< End   computation of plane
    cudaEvent_t		outInit;		///< start copying results from device
    cudaEvent_t		outCmp;			///< End   copying results from device
} cuOptCand;

/** Data structure to hold the GPU information for performing GPU optimisation  .
 *
 */
typedef struct cuCoPlan
{
    cuSearch*		cuSrch;			///< Details of the search
    confSpecsCO*	conf;			///< Global configuration parameters
    gpuInf*		gInf;			///< Information on the GPU being used
    cuPlnGen*		plnGen;			///< A GPU plane generator
    cuHarmInput*	input;			///< A pointer holding input data

    int			pIdx;			///< The index of this optimiser in the list

    int64_t		flags;			///< CUDA accel configuration bit flags

    // TIMING values
    long long*		compTime;		///< Array of floats from timing, one float for each stack

} cuOpt;

/** A struct to keep info on all the kernels and CO plans to use with cuda accelsearch  .
 */
typedef struct cuCoInfo
{
    int			noCoPlans;		///< The total number of optimisations to do across all devices
    cuCoPlan*		coPlans;		///< A list noOpts long of
    cuRespPln*		responsePlanes;		///< A collection of response functions for optimisation, one per GPU
} cuOptInfo;

/** Details of the GPU's  .
 */
typedef struct cuGpuInfo
{
    // Details of the GPU's in use
    int			noDevices;		///< The number of devices (GPU's to use in the search)

    int			devid[MAX_GPUS];
    int			alignment[MAX_GPUS];
    float		capability[MAX_GPUS];
    char*		name[MAX_GPUS];
} cuGpuInfo;

/** User independent details  .
 */
struct cuSearch
{
    searchSpecs*	sSpec;				///< Details on o the size (in bins) of the search
    confSpecs*		conf;				///< Configuration specifications of the candidate generation
    gpuSpecs*		gSpec;				///< Specifications of the GPU's to use
    fftInfo*		fft;				///< The details of the input fft - location size and area to search

    resThrds*		threasdInfo;			///< Information on threads to handle returned candidates.
    cuCgInfo*		pInf;				///< The allocated Device and host memory and data structures to create planes including the kernels
    cuCoInfo*		oInf;				///< Details of optimisations

    //// Some extra search details
    int			noHarmStages;			///< The number of stages of harmonic summing
    int			noGenHarms;			///< The number of harmonics in the family
    int			noSrchHarms;			///< The number of harmonics to search over

    long long		timings[COMP_MAX];		///< Array for timing values (values stored in μs) - These are used for both timing and profiling, they are only filled if TIMING and or PROFILING are defined in cuda_accel.h

    // Search power cutoff values
    int*		sIdx;				///< The index of the planes in the Presto harmonic summing order
    float*		powerCut;			///< The power cutoff
    long long*		numindep;			///< The number of independent trials
    int*		yInds;				///< The Y indices

    // Search specific memory
    void*		h_candidates;			///< Host memory for candidates
    void*		d_planeFull;			///< Device memory for the in-mem f-∂f plane
    GSList*		cands;				///< The candidates from the GPU search

    size_t		inmemStride;			///< The stride (in units) of the in-memory plane data in device memory
    size_t		candStride;			///< The stride (in units) of the host candidate array
};

/** Information of the P-threads used in the search  .
 *  This is used in candidate generation and optimisation stages
 */
struct resThrds
{
    sem_t		running_threads;		///< Semaphore for number running threads

    pthread_mutex_t	running_mutex;			///< Deprecated
    pthread_mutex_t	candAdd_mutex;			///< Mutex to change the global list of candidates

};

/** A data structure to pass to a thread, containing information on search results  .
 *
 */
typedef struct resultData
{
    cuSearch*		cuSrch;			///< Details of the search

    void*		retData;		///< A pointer to the memory the results are stored in (usual pinned host memory)
    bool*		outBusy;		///< A pointer to the flag indicating that the memory has all been read
    int			resSize;		///< The size of the results data

    uint		retType;		///< The way the candidates should be stored
    uint		cndType;		///<
    int64_t		flags;			///< CUDA accel search bit flags

    cudaEvent_t		preBlock;		///< An event to block the thread on before processing the data
    cudaEvent_t		postScan;		///< An CUDA event to create after the data has finished being used
    cudaStream_t	stream;			///< The stream to record the event in

    uint		x0;
    uint		x1;

    uint		y0;
    uint		y1;

    uint		xStride;
    uint		yStride;

    double		zStart;			///< Max Z-value
    double		zEnd;			///< Min Z-value
    uint		noZ;			///< The number of z-values searched

    double		rLow;			///< The input FT bin "index" of the first valid result
    int 		noResPerBin;		///< The number of response values per bin of the input fft - this allows "over sampling" the standard value is 2 interbinning.
    float		candRRes;		///< The resolution of the candidate array ( measured in input FT bins')

    rVals		rVal;

    uint*		noResults;		///< A value to keep tack of the number of candidates found

    long long*		resultTime;
    long long*		blockTime;		///< This can't really get used...
} resultData;

/** This is just a wrapper to be passed to a CPU thread  .
 *
 */
typedef struct candSrch
{
    cuSearch*		cuSrch;			///< Details of the search
    cuHarmInput*	input;			///< Input data for the harmonics
    accelcand*		cand;			///< The candidate to optimise
    int			candNo;			///< The 0 based index of this candidate
    double		resolution;		///< The size to start the final optimisation at
} candSrch;


/****************************************** Functions ****************************************************/



/************************************* Function prototypes ***********************************************/

cuHarmInput* initHarmInput( int maxWidth, float zMax, int maxHarms, gpuInf* gInf );

cuHarmInput* initHarmInput( size_t memSize, gpuInf* gInf );

acc_err chkInput( cuHarmInput* input, double r, double z, double rSize, double zSize, int noHarms, int* newInp);

acc_err loadHostHarmInput( cuHarmInput* input, fftInfo* fft, double r, double z, double rSize, double zSize, int noHarms, int64_t flags = FLAG_OPT_NRM_LOCAVE, cudaEvent_t* preWrite = NULL );

void setDebugMsgLevel(int lvl);

acc_err setCgPlanStartR  (cuCgPlan* plan, double firstR, int firstIteration = 1, int firstSegment = 1 );

acc_err setCgPlanCenterR (cuCgPlan* plan, double firstR, int firstIteration = 1, int firstSegment = 1 );

/** Read the GPU details from clig command line  .
 *
 * @param cmd
 * @return A pointer to the accel info struct to fill
 */
ExternC gpuSpecs* readGPUcmd(Cmdline *cmd);

// TODO - Write these descriptions
gpuSpecs* getGpuSpec(int devID = -1, int plan = 0, int segments = 0, int opts = 0 );

searchSpecs* getSpec(fftInfo* fft);

confSpecs* getConfig();

void initCandGeneration(cuSearch* sSrch );

/** Read the GPU details from clig command line  .
 *
 * @param cmd
 * @return A pointer to the accel info struct to fill
 */
ExternC confSpecsCG readSrchSpecs(Cmdline *cmd, accelobs* obs);

ExternC cuSearch* initSearchInfCMD(Cmdline *cmd, accelobs* obs, gpuSpecs* gSpec);

ExternC cuSearch* initSearchInf(searchSpecs* sSpec, confSpecs* conf, gpuSpecs* gSpec, fftInfo* fftInf);

ExternC cuSearch* initCuKernels(confSpecsCG* sSpec, gpuSpecs* gSpec, cuSearch* srch);

ExternC GSList* generateCandidatesGPU(cuSearch* cuSrch);

ExternC void libTst();

ExternC fftInfo* readFFT(char* fileName);

ExternC cuSearch* initCuOpt(cuSearch* srch);

ExternC acc_err freeOptimisers(cuSearch* sSrch);

ExternC void freeCuSearch(cuSearch* srch);

ExternC void freeAccelGPUMem(cuCgInfo* mInf);

ExternC cuPlnGen* initOptPln(confSpecsCG* sSpec);

ExternC cuPlnGen* initOptSwrm(confSpecsCG* sSpec);


/** Initialise the template structure and kernels for a multi-segment CO plan  .
 * This is called once per device
 *
 * @param stkLst            The data structure to fill
 * @param master            The master data structure to copy kernel and some settings from, if this is the first call this should be NULL
 * @param numharmstages     The number of harmonic stages
 * @param zmax              The ZMax of the primary harmonic
 * @param fftinf            The address and accel search info
 * @param device            The device to create the kernels on
 * @param noBatches         The desired number of CO plans to run on this device
 * @param noSteps           The number of segments for each CO plan to use
 * @param width             The desired width of the primary harmonic in thousands
 * @param powcut            The value above which to return
 * @param numindep          The number of independent trials
 *
 * @return The number CO plans set up for this should be noBatches. On failure returns 0
 */
//ExternC int initHarmonics(cuCgPlan* stkLst, cuCgPlan* master, int numharmstages, int zmax, fftInfo fftinf, int device, int noBatches, int noSteps, int width, float*  powcut, long long*  numindep, int flags, int candType, int retType, void* out);

/** Free all host and device memory allocated by initHarmonics(...)  .
 * If the stkLst is master, this will free any device independat memory
 *
 * @param stkLst            The data structure to free
 * @param master            The master stak list
 * @param out               The candidate output, if none was specified this should be NULL
 */
//ExternC  void freeHarmonics(cuCgPlan* stkLst, cuCgPlan* master, void* out);

/** Initialise a multi-step CO plan from the device kernel  .
 *
 * @param harms             The kernel to base this multi-step CO plan
 * @param no                The index of this CO plan
 * @param of                The desired number of CO plans on this device
 * @return
 */
//ExternC cuCgPlan* initBatch(cuCgPlan* harms, int no, int of);

/** Free device and host memory allocated by initStkList  .
 *
 * @param harms             The CO plan to free
 */
//ExternC void freeBatch(cuCgPlan* stkLst);

ExternC void setContext(cuCgPlan* plan) ;

ExternC int setDevice(int device);

ExternC void freeCgPlanGPUmem(cuCgPlan* plan);

ExternC void printCands(const char* fileName, GSList *candsCPU, double T);

ExternC void run_CG_plan(cuCgPlan* plan);

ExternC acc_err inmemSS(cuCgPlan* plan, double drlo, int len);

ExternC acc_err inmemSumAndSearch(cuSearch* cuSrch);

ExternC void finish_Search(cuCgPlan* plan);

ExternC void cg_sum_and_search_inmem(cuCgPlan* plan );

ExternC void accelMax(fcomplex* fft, long long noBins, long long startBin, long long endBin, short zMax, short numharmstages, float* powers );

/** Print the flag values in text  .
 *
 * @param flags
 */
ExternC void printFlags(uint flags);

ExternC void printCommandLine(int argc, char *argv[]);

ExternC void writeLogEntry(const char* fname, accelobs* obs, cuSearch* cuSrch, long long prepTime, long long cpuKerTime, long long cupTime, long long gpuKerTime, long long gpuTime, long long optTime, long long cpuOptTime, long long gpuOptTime);

ExternC GSList* getCanidates(cuCgPlan* plan, GSList* cands );

ExternC void calcNQ(double qOrr, long long n, double* p, double* q);

ExternC GSList* testTest(cuCgPlan* plan, GSList* candsGPU);

ExternC int waitForThreads(sem_t* running_threads, const char* msg, int sleepMS );

ExternC long long initCudaContext(gpuSpecs* gSpec);

ExternC long long compltCudaContext(gpuSpecs* gSpec);

/** Cycle back the values in the array of input data
 *
 * @param plan
 */
ExternC void CycleBackRlists(cuCgPlan* plan);

ExternC cuSearch* searchGPU(cuSearch* cuSrch, gpuSpecs* gSpec, confSpecsCG* sSpec);

ExternC void clearRvals(cuCgPlan* plan);

ExternC void clearRval( rVals* rVal);

#endif // CUDA_ACCEL_INCLUDED
