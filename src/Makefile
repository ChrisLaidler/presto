#  Makefile for PRESTO
#   by Scott M. Ransom
#      and Chris Laidler

# OS type
OS = Linux
#OS = OSX

# yes / no
WITH_CUDA  = yes
WITHOMP    = yes
WITHCBL    = yes

# CUDA_ARCH needs to be set to the architecture of the device(s) you want to run the search on
# CUDA_ARCH should be a space separated list consisting to numbers ie "30 50"
# Where the numbers are major minor and can be found by running deviceQuery
# You can use multiple values, just uncomment the specific lines below, or add you’re own
CUDA_ARCH  =
#CUDA_ARCH += 20
#CUDA_ARCH += 30
#CUDA_ARCH += 35
#CUDA_ARCH += 50


OS_SIZE    = $(shell uname -m | sed -e "s/x86_64/64/" -e "s/armv7l/32/" -e "s/aarch64/64/")
OS_ARCH    = $(shell uname -m)
ARCH_FLAGS =

BINPATH    = $(PRESTO)/bin
LIBPATH    = $(PRESTO)/lib

PRESTOOBJS_BASE = accel_utils.o amoeba.o atwood.o barycenter.o birdzap.o cand_output.o\
	characteristics.o cldj.o chkio.o corr_prep.o corr_routines.o\
	correlations.o database.o dcdflib.o dispersion.o\
	fastffts.o fftcalls.o fminbr.o fold.o fresnl.o ioinf.o\
	get_candidates.o iomak.o ipmpar.o maximize_r.o maximize_rz.o\
	median.o minifft.o misc_utils.o clipping.o\
	orbint.o output.o read_fft.o responses.o\
	rzinterp.o rzwinterp.o select.o sorter.o swapendian.o\
	transpose.o twopass.o twopass_real_fwd.o\
	twopass_real_inv.o vectors.o multifiles.o mask.o\
	fitsfile.o hget.o hput.o imio.o djcl.o zapping.o

COMPILE_START_TIME := $(shell date +\(%j\*24\*60\*60+%H\*60\*60+%M\*60+%S\))
COMPILE_CURRENT_TIME = `date  +\(%j\*24\*60\*60+%H\*60\*60+%M\*60+%S\)`

# args: Tcurrent = $(1), Tstart = $(2)
define print_timer
@CT=$(1) && \
printf "\nDone builiding $(3) in %.2d:%.2d (min:sec)\n\n" \
`echo "($$CT - $(2))/60" | bc` \
`echo "($$CT - $(2))%60" | bc`
endef

# Take command line flags that override any of these settings
ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif
ifeq ($(ARMv7),1)
	OS_SIZE    = 32
	OS_ARCH    = armv7l
	ARCH_FLAGS = -target-cpu-arch ARM
endif
ifeq ($(aarch64),1)
	OS_SIZE    = 64
	OS_ARCH    = aarch64
	ARCH_FLAGS = -target-cpu-arch ARM
endif

# Linux is the first choice
ifeq ($(OS),Linux)
	LIBSUFFIX = .so
	LIBCMD = -shared
# else assume Darwin (i.e. OSX)
else
	LIBSUFFIX = .dylib
	LIBCMD = -dynamiclib
endif

# How to link with some needed libraries of PGPLOT
X11LINK := $(shell pkg-config --libs x11)
PNGLINK := $(shell pkg-config --libs libpng)

# Include and link information for PGPLOT v5.X (including shared libs!)
# Typically you need to have your PGPLOT_DIR environment variable set
ifndef PGPLOT_DIR
$(error PGPLOT_DIR is not set)
else
PGPLOTINC  = -I$(PGPLOT_DIR)
PGPLOTLINK = -L$(PGPLOT_DIR) -lcpgplot -lpgplot $(X11LINK) $(PNGLINK)
endif

# Include and link information for the FFTW 3.X single-precision library
FFTINC  := $(shell pkg-config --cflags fftw3f)
FFTLINK := $(shell pkg-config --libs fftw3f)

# Include and link information for the GLIB 2.0 library
GLIBINC  := $(shell pkg-config --cflags glib-2.0)
GLIBLINK := $(shell pkg-config --libs glib-2.0)

# Include and link information for CFITSIO
CFITSIOINC  := $(shell pkg-config --cflags cfitsio)
CFITSIOLINK := $(shell pkg-config --libs cfitsio) -lm

# The standard PRESTO libraries to link into executables
PRESTOLINK = $(CFITSIOLINK) -L$(PRESTO)/lib -lpresto$(POSTFIX) $(GLIBLINK) $(FFTLINK)
CUACCELLINK = ""

CC 	= gcc
CPP = g++
FC = gfortran
#FC = g77
NVCC = nvcc


# DEBUG flags make with mode=debug
ifeq ($(mode),release)
  DEBUG=0
endif
ifeq ($(mode),RELEASE)
  DEBUG=0
endif
ifeq ($(mode),r)
  DEBUG=0
endif
ifeq ($(mode),R)
  DEBUG=0
endif

ifeq ($(mode),DEBUG)
  DEBUG=1
endif
ifeq ($(mode),debug)
  DEBUG=1
endif
ifeq ($(mode),d)
  DEBUG=1
endif
ifeq ($(mode),D)
  DEBUG=1
endif

# With CUDA
ifeq ($(WITH_CUDA),yes)

  ifeq ($(CUDA_ARCH),)
  	$(info )
  	$(info Please specify CUDA_ARCH in the makefile if compiling with CUDA  )
  	$(info                CUDA_ARCH is the devivce specific CUDA Capability )
  	$(info )
  	$(error Error: CUDA_ARCH not specifyed)
  endif
endif
#
# Very recent Intel CPUs might see a few percent speed-up using -mavx
BASEFLAGS 			= -I$(PRESTO)/include $(GLIBINC) $(CFITSIOINC) $(PGPLOTINC) $(FFTINC) -DUSEFFTW -DUSEMMAP -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -g -Wall -W -fPIC
#	 -march=native
#       -g -O3 -ffast-math -Wall -W -fPIC  # Debug builds now done with mode=debug

# Warnings
CXXWARN			 	= -Wno-unused-result
# Other Warnings
CXXWARN			  += -Wno-comment -Wno-unused-but-set-variable -Wno-unused-but-set-parameter -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function

CWARN				  = -Wno-implicit-function-declaration

CFLAGS			 	+= $(BASEFLAGS) $(CWARN) $(CXXWARN)
CXXFLAGS			+= $(BASEFLAGS) $(CXXWARN)

CLINKFLAGS 		= $(CFLAGS)
# NOTE:  Be careful of upping the optimization on the
#        FFLAGs.  Certain compilers (i.e. on Intel Macs) will
#        cause errors for the code in least_squares.f
FFLAGS       	= -g -fPIC
FLINKFLAGS   	= $(FFLAGS)
LINKCOMMAND1 	= $(CC) $(LIBCMD) -o
LINKCOMMAND2 	= ar rcs

CFLAGS_DBG = -I/home/chris/projects/libs/build/include

######## CUDA flags

#Change CUDA_ARCH to the correct form
CUDA_ARCH  := $(foreach root, $(CUDA_ARCH), -gencode arch=compute_$(root),code=sm_$(root))
#CUDA_ARCH  := $(foreach root, $(CUDA_ARCH), -arch compute_$(root))

CUDA_PATH  = /usr/local/cuda

CUDALIBS   += -L$(CUDA_PATH)/lib64 -lcudart -lcuda -lnvToolsExt
CUDAINC    += -I$(CUDA_PATH)/include

NVCCFLAGS  =  $(CUDAINC) $(CUDA_ARCH) -I$(PRESTO)/include $(GLIBINC) $(CFITSIOINC) -m${OS_SIZE} -dc ${ARCH_FLAGS}
NVCCFLAGS  += $(FFTINC) -DUSEFFTW -DUSEMMAP -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64
#NVCCFLAGS  += -Xcompiler -Wno-comment
NVCCFLAGS	 += $(foreach root, $(CXXWARN), -Xcompiler $(root))

NVCLFLAGS  = $(CUDAINC) $(CUDA_ARCH) $(CUDALIBS)
NVCLFLAGS  += $(FFTINC) -DUSEFFTW -DUSEMMAP -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64

GPUOUTP		=


$(info)

OPT_FLAGS =

#Settings dependent on release or debug build
ifeq ($(DEBUG),1)
  $(info Compiling in debug mode)

  OPT_FLAGS     += -O0 -ggdb -DDEBUG

  NVCCFLAGS     += -g -G -O0 -DDEBUG
  NVCLFLAGS     += -g -G -O0 -DDEBUG

  DEBUG_TARGET  =debug
  POSTFIX       =_debug
  export DEBUG  =1
  OBJDIR        =build/debug/
else
  $(info Compileing in release mode)

  # CPU optimisations
  OPT_FLAGS		+= -O3 -ffast-math

  # Additional CPU optimisations
  OPT_FLAGS		+= -unsafe-math -funroll-loops -funswitch-loops -fprefetch-loop-arrays
  # could use -mpc32

  # vectorisation output
  # CFLAGS+=-ftree-vectorizer-verbose=2 -fdump-tree-vect-details

  NVCCFLAGS += -O3 -DFLOAT -DFORUM -use_fast_math -prec-div=false -ftz=true -prec-sqrt=false --fmad=true -lineinfo
  NVCLFLAGS += -O3 -DFLOAT -DFORUM -use_fast_math -prec-div=false -ftz=true -prec-sqrt=false --fmad=true -lineinfo

  # Global memory accesses cached in both L1 and L2 (default)
  NVCCFLAGS += -Xptxas -dlcm=ca
  NVCLFLAGS += -Xptxas -dlcm=ca

  # Global memory accesses cached in L2 only - more SM
  #NVCCFLAGS += -Xptxas -dlcm=cg
  #NVCLFLAGS += -Xptxas -dlcm=cg

  DEBUG_TARGET   =release
  POSTFIX        =
  export DEBUG   =0
  OBJDIR         =build/release/
endif

CFLAGS 		+= $(OPT_FLAGS)
CXXFLAGS	+= $(OPT_FLAGS)


$(info )

OMP_FLAGS		=

# With open MP
ifeq ($(WITHOMP),yes)
  $(info • Compileing with OpenMP)

  OMP_FLAGS   += -fopenmp -DWITHOMP

  NVCCFLAGS   += -Xcompiler -fopenmp -DWITHOMP
  NVCLFLAGS   += -Xcompiler -fopenmp -DWITHOMP
endif

CFLAGS				+= $(OMP_FLAGS)
CXXFLAGS			+= $(OMP_FLAGS)
LDFLAGS				+= $(OMP_FLAGS)


ifeq ($(WITHCBL),yes)
  $(info • Compileing with Chris utility libraries)

  # Include and link information for FreeType
  FTCFLG     	 = $(shell freetype-config --cflags)
  FTLINK     	 = $(shell freetype-config --libs)

  LNDARR     	+= -L$(SRC)/libs/build/lib -lndarray$(POSTFIX) -lcblutil$(POSTFIX) -lpngwriter $(PNGLINK) $(FTLINK)
  INDARR     	+= -I$(SRC)/libs/build/include/ $(FTCFLG)

  NVCCFLAGS  	+= -DCBL $(INDARR)
  NVCLFLAGS  	+= -DCBL $(INDARR) -L$(SRC)/libs/build/lib -lndarray$(POSTFIX) -lcblutil$(POSTFIX) -lpngwriter $(PNGLINK) $(FTLINK)

  CFLAGS     	+= -DCBL
  CXXFLAGS   	+= -DCBL $(INDARR) -Wno-unknown-pragmas

  GPUOUTP    	+= $(BINPATH)/test_GPUaccel$(POSTFIX)
else
  LNDARR     =
  INDARR     =
endif


# With CUDA
ifeq ($(WITH_CUDA),yes)

  CUDA_VER    = $(shell $(NVCC) -V | grep release | cut -f 5 -d ' ' | tr -d ,)

  $(info • Compileing with CUDA version $(CUDA_VER))

  NVCC_SRCS  	= $(wildcard *.cu)
  NVCC_OBS   	= $(patsubst %.cu, $(OBJDIR)%.cu.o,  $(NVCC_SRCS)  )

  NVCCFLAGS  	+= -DCUDA -Xcompiler -fPIC
  NVCLFLAGS  	+= -DCUDA -Xcompiler -fPIC

  #NVCCFLAGS		+= -ptx -keep
  #NVCLFLAGS		+= -ptx -keep

  CFLAGS     	+= -DCUDA $(CUDAINC)
  FLAGS      	+= -DCUDA $(CUDAINC)

  CXXFLAGS   	+= -DCUDA $(CFLAGS)	$(CUDAINC)
  LDFLAGS    	+= -DCUDA $(CUDAINC) $(CLINKFLAGS)

  CUACCL     	= -lcuaccel$(POSTFIX)

  GPUOUTP    	+= $(BINPATH)/accelsearch$(POSTFIX) $(BINPATH)/getAccelLen$(POSTFIX) # $(LIBPATH)/libcuaccel$(POSTFIX)$(LIBSUFFIX)

	# Check for CUDA 6.5 for CUFFT callbacks
	ifeq ($(shell echo ${CUDA_VER}\>=6.5 | bc),1)
  	CUDALIBS  	+= -lcufft_static -lculibos
	else
	  CUDALIBS		+= -lcufft
	endif

else
  NVCC        = $(CPP)
  NVCC_OBS    =
  NVCLFLAGS   = $(CLINKFLAGS)
  GPUOUTP     =
endif



$(info )

PRESTOOBJS = $(patsubst %.o, $(OBJDIR)%.o,  $(PRESTOOBJS_BASE)  )

#$(info $$BIN is [${BIN}])
#$(info $$LIBSUFFIX is [${LIBSUFFIX}])

# Add to the search path for the executables
VPATH = ../lib:../bin

# When modifying the CLIG files, the is the location of the clig binary
CLIG = clig

INSTRUMENTOBJS = $(OBJDIR)backend_common.o $(OBJDIR)zerodm.o $(OBJDIR)sigproc_fb.o $(OBJDIR)psrfits.o

# Use old header reading stuff for readfile
READFILEOBJS = $(INSTRUMENTOBJS) $(OBJDIR)multibeam.o $(OBJDIR)bpp.o $(OBJDIR)spigot.o \
	$(OBJDIR)wapp.o $(OBJDIR)wapp_head_parse.o $(OBJDIR)wapp_y.tab.o

PLOT2DOBJS = $(OBJDIR)powerplot.o $(OBJDIR)xyline.o

BINARIES = makedata makeinf mjd2cal realfft quicklook\
	search_bin search_rzw swap_endian prepdata\
	check_parkes_raw bary shiftdata dftfold\
	patchdata readfile toas2dat taperaw\
	accelsearch prepsubband cal2mjd split_parkes_beams\
	dat2sdat sdat2dat downsample rednoise un_sc_td bincand\
	psrorbit window plotbincand prepfold show_pfd\
	rfifind zapbirds explorefft exploredat\
	weight_psrfits fitsdelrow fitsdelcol psrfits_dumparrays
#	dump_spigot_zerolag spigot2filterbank\
#	spigotSband2filterbank GBT350filterbank\

SLA_SRCS	= $(wildcard slalib/*.f)
SLA_SRCS	:= $(filter-out slalib/sla_test.f, $(SLA_SRCS))
SLA_OBS		= $(patsubst %.f, %.o,  $(SLA_SRCS)  )

libpresto	= $(LIBPATH)/libpresto$(POSTFIX)$(LIBSUFFIX)
libcuaccel	= $(LIBPATH)/libcuaccel$(POSTFIX)$(LIBSUFFIX)

all: libpresto slalib binaries $(GPUOUTP)
	$(call print_timer,$(COMPILE_CURRENT_TIME),$(COMPILE_START_TIME),All)

debug:
	$(MAKE) mode=debug all

# Default indentation is K&R style with no-tabs,
# an indentation level of 4 (default), and a line-length of 85
indent:
	indent -kr -nut -l85 *.c
	rm *.c~

test:
	@echo "NVCC_SRCS: $(NVCC_SRCS)"
	@echo NVCC_OBS: $(NVCC_OBS) $(POSTFIX)
	@echo PRESTOOBJS: $(PRESTOOBJS)
	@echo OBJDIR: $(OBJDIR)


prep:
	touch *_cmd.c

libpresto: $(libpresto)
	@ # Make file cant handle blank targets so here is a random comment =/

libcuaccel: $(libcuaccel)
	@ # Make file cant handle blank targets so here is a random comment =/

binaries: $(BINARIES)
	@ # Make file cant handle blank targets so here is a random comment =/

mpi: mpiprepsubband

GPU: $(GPUOUTP)
	@ # $(LIBPATH)/libcuaccel$(POSTFIX)$(LIBSUFFIX) # Temporally removed
	@ # Make file cant handle blank targets so here is a random comment =/

slalib: slalib/sla_test test_sla
	@ # Make file cant handle blank targets so here is a random comment =/

test_GPUaccel: $(BINPATH)/test_GPUaccel$(POSTFIX)
	@ # Make file cant handle blank targets so here is a random comment =/

getAccelLen: $(BINPATH)/getAccelLen$(POSTFIX)
	@ # Make file cant handle blank targets so here is a random comment =/

accelsearch: $(BINPATH)/accelsearch$(POSTFIX)
	@ # Make file cant handle blank targets so here is a random comment =/

###############   Librays   ################

$(LIBPATH)/libsla$(LIBSUFFIX): $(SLA_OBS)
	$(FC) $(LIBCMD) -o $@ -fno-second-underscore $(SLA_OBS)

$(LIBPATH)/libpresto$(POSTFIX)$(LIBSUFFIX): $(PRESTOOBJS)
	@echo ""
	@echo "Linking libpresto$(POSTFIX)"
	$(LINKCOMMAND1) $@ $(PRESTOOBJS) $(FFTLINK)
	$(call print_timer,$(COMPILE_CURRENT_TIME),$(COMPILE_START_TIME),libpresto$(POSTFIX))


$(LIBPATH)/libcuaccel$(POSTFIX)$(LIBSUFFIX): $(NVCC_OBS)
	@echo ""
	@echo "Linking libcuaccel$(POSTFIX)"
	$(NVCC) $(LIBCMD) -o $@ $(NVCC_OBS) $(FFTLINK) $(NVCLFLAGS)
	$(call print_timer,$(COMPILE_CURRENT_TIME),$(COMPILE_START_TIME),libcuaccel$(POSTFIX))

###############   Tests   ################

test_sla: slalib/sla_test
	@echo ""
	@echo "Testing SLALIB:"
	@slalib/sla_test

timetest: timetest
	$(CC) -o $@ timetest.c
	./timetest
	rm -f timetest

makewisdom: makewisdom
	$(CC) $(CLINKFLAGS) -o $@ makewisdom.c $(FFTLINK)
	./makewisdom
	cp fftw_wisdom.txt $(PRESTO)/lib


###############   CLIG    ################
# Rules for CLIG generated files

%_cmd.c : ../clig/%_cmd.cli
	cd ../clig ; $(CLIG) -o $*_cmd -d $<
	mv ../clig/$*_cmd.h ../include/
	mv ../clig/$*_cmd.c .
	cp ../clig/$*.1 ../docs/

###############  binaries   ################

slalib/sla_test: $(LIBPATH)/libsla$(LIBSUFFIX)
	$(FC) -o slalib/sla_test slalib/sla_test.f -fno-second-underscore -L$(LIBPATH) -lsla

mpiprepsubband: mpiprepsubband_cmd.c $(OBJDIR)mpiprepsubband_cmd.o $(OBJDIR)mpiprepsubband_utils.o $(OBJDIR)mpiprepsubband.o $(INSTRUMENTOBJS)
	mpicc $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)mpiprepsubband_cmd.o $(OBJDIR)mpiprepsubband_utils.o $(OBJDIR)mpiprepsubband.o $(INSTRUMENTOBJS) $(PRESTOLINK) -lcfitsio -lm

${BINPATH}/accelsearch$(POSTFIX): $(libpresto) $(NVCC_OBS) $(OBJDIR)accelsearch_cmd.o $(OBJDIR)accel_utils.o $(OBJDIR)accelsearch.o $(OBJDIR)zapping.o
	@echo ""
	@echo "Linking accelsearch$(POSTFIX)"
	$(NVCC) $(NVCLFLAGS) -o $@ $(OBJDIR)accelsearch_cmd.o $(OBJDIR)accel_utils.o $(OBJDIR)accelsearch.o $(OBJDIR)zapping.o $(PRESTOLINK) $(NVCC_OBS) $(GLIBLINK) -lm $(FFTLINK)
	$(call print_timer,$(COMPILE_CURRENT_TIME),$(COMPILE_START_TIME),accelsearch$(POSTFIX))

$(BINPATH)/test_GPUaccel$(POSTFIX): $(libpresto) $(NVCC_OBS) $(OBJDIR)accelsearch_cmd.o $(OBJDIR)accel_utils.o $(OBJDIR)test_GPUaccel.o $(OBJDIR)zapping.o
	@echo ""
	@echo "Linking test_GPUaccel"${POSTFIX}
	$(NVCC) $(NVCLFLAGS) -o $@ $(OBJDIR)accelsearch_cmd.o $(OBJDIR)accel_utils.o $(OBJDIR)test_GPUaccel.o $(OBJDIR)zapping.o $(PRESTOLINK) $(NVCC_OBS) $(GLIBLINK) -lm
	$(call print_timer,$(COMPILE_CURRENT_TIME),$(COMPILE_START_TIME),test_GPUaccel${POSTFIX})


$(BINPATH)/getAccelLen$(POSTFIX): $(OBJDIR)accel_utils.o $(OBJDIR)getAccelLen.o $(NVCC_OBS) $(libpresto)
	@echo ""
	@echo "Linking getAccelLen"${POSTFIX}
	$(NVCC) $(NVCLFLAGS) $(OBJDIR)accel_utils.o -o $@ $(OBJDIR)getAccelLen.o $(PRESTOLINK) $(NVCC_OBS) $(GLIBLINK) -lm
	$(call print_timer,$(COMPILE_CURRENT_TIME),$(COMPILE_START_TIME),getAccelLen${POSTFIX})

check_parkes_raw: $(OBJDIR)check_parkes_raw.o $(OBJDIR)multibeam.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)check_parkes_raw.o $(OBJDIR)multibeam.o $(PRESTOLINK) -lm

downsample: downsample_cmd.c $(OBJDIR)downsample.o $(OBJDIR)downsample_cmd.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)downsample.o $(OBJDIR)downsample_cmd.o $(PRESTOLINK) -lm

split_parkes_beams: $(OBJDIR)split_parkes_beams.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)split_parkes_beams.o

test_multifiles: $(OBJDIR)test_multifiles.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)test_multifiles.o $(PRESTOLINK) -lm

rfifind: rfifind_cmd.c $(OBJDIR)rfifind_cmd.o $(OBJDIR)rfifind.o $(OBJDIR)rfi_utils.o $(OBJDIR)rfifind_plot.o $(OBJDIR)range_parse.o $(INSTRUMENTOBJS) $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@  $(INSTRUMENTOBJS) $(PLOT2DOBJS) $(OBJDIR)rfifind.o $(OBJDIR)rfi_utils.o $(OBJDIR)rfifind_cmd.o $(OBJDIR)rfifind_plot.o $(OBJDIR)range_parse.o $(PRESTOLINK) $(PGPLOTLINK) -lcfitsio -lm

prepdata: prepdata_cmd.c $(OBJDIR)prepdata_cmd.o $(OBJDIR)prepdata.o $(INSTRUMENTOBJS) $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)prepdata.o $(OBJDIR)prepdata_cmd.o $(INSTRUMENTOBJS) $(PRESTOLINK) -lcfitsio -lm

prepsubband: prepsubband_cmd.c $(OBJDIR)prepsubband_cmd.o $(OBJDIR)prepsubband.o $(INSTRUMENTOBJS) $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)prepsubband.o $(OBJDIR)prepsubband_cmd.o $(INSTRUMENTOBJS) $(PRESTOLINK) -lcfitsio -lm

prepfold: prepfold_cmd.c $(OBJDIR)prepfold_cmd.o $(OBJDIR)prepfold.o $(OBJDIR)prepfold_utils.o $(OBJDIR)prepfold_plot.o $(OBJDIR)least_squares.o $(OBJDIR)polycos.o $(OBJDIR)readpar.o $(INSTRUMENTOBJS) $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)prepfold.o $(OBJDIR)prepfold_utils.o $(OBJDIR)prepfold_plot.o $(OBJDIR)prepfold_cmd.o $(OBJDIR)least_squares.o $(OBJDIR)polycos.o $(OBJDIR)readpar.o $(PLOT2DOBJS) $(INSTRUMENTOBJS) $(LAPACKLINK) $(PRESTOLINK) $(PGPLOTLINK) -lcfitsio -lm

dump_spigot_zerolag: $(OBJDIR)dump_spigot_zerolag.o $(OBJDIR)spigot.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)dump_spigot_zerolag.o $(OBJDIR)spigot.o $(PRESTOLINK) -lm

spigot2filterbank: spigot2filterbank_cmd.c $(OBJDIR)spigot2filterbank_cmd.o $(OBJDIR)spigot2filterbank.o $(OBJDIR)spigot.o $(OBJDIR)sigproc_fb.o$(OBJDIR) $(OBJDIR)sla.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)spigot2filterbank.o $(OBJDIR)spigot.o $(OBJDIR)sigproc_fb.o $(OBJDIR)spigot2filterbank_cmd.o $(OBJDIR)sla.o $(PRESTOLINK) -lsla -lm

GBT350filterbank: $(OBJDIR)GBT350filterbank.o $(OBJDIR)spigot.o $(OBJDIR)sigproc_fb.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)GBT350filterbank.o $(OBJDIR)spigot.o $(OBJDIR)sigproc_fb.o $(PRESTOLINK) -lm

spigotSband2filterbank: $(OBJDIR)spigotSband2filterbank.o $(OBJDIR)spigot.o $(OBJDIR)sigproc_fb.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)spigotSband2filterbank.o $(OBJDIR)spigot.o $(OBJDIR)sigproc_fb.o $(PRESTOLINK) -lm

show_pfd: show_pfd_cmd.c $(OBJDIR)show_pfd.o $(OBJDIR)show_pfd_cmd.o $(OBJDIR)prepfold_utils.o $(OBJDIR)prepfold_plot.o $(OBJDIR)least_squares.o $(OBJDIR)range_parse.o $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)show_pfd.o $(OBJDIR)show_pfd_cmd.o $(OBJDIR)prepfold_utils.o $(OBJDIR)prepfold_plot.o $(OBJDIR)least_squares.o $(OBJDIR)range_parse.o $(PLOT2DOBJS) $(LAPACKLINK) $(PRESTOLINK) $(PGPLOTLINK) -lm

makedata: $(OBJDIR)com.o $(OBJDIR)randlib.o $(OBJDIR)mvmd.o $(OBJDIR)makedata.o $(libpresto)
	$(CPP) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)com.o $(OBJDIR)randlib.o $(OBJDIR)mvmd.o $(OBJDIR)makedata.o  $(PRESTOLINK) -lm

makeinf: $(OBJDIR)makeinf.o $(OBJDIR)ioinf.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)makeinf.o $(OBJDIR)ioinf.o $(PRESTOLINK) -lm

mjd2cal: $(OBJDIR)djcl.o $(OBJDIR)mjd2cal.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)djcl.o $(OBJDIR)mjd2cal.o -lm

cal2mjd: $(OBJDIR)cldj.o $(OBJDIR)cal2mjd.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)cldj.o $(OBJDIR)cal2mjd.o -lm

plotbincand: $(OBJDIR)plotbincand.o $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ $(PLOT2DOBJS) $(OBJDIR)plotbincand.o $(PRESTOLINK) $(PGPLOTLINK) -lm

profile: profile_cmd.c $(OBJDIR)profile_cmd.o $(OBJDIR)profile.o $(PLOT2DOBJS) ${POSTFIX}
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ $(PLOT2DOBJS) $(OBJDIR)profile.o $(OBJDIR)profile_cmd.o $(PRESTOLINK) $(PGPLOTLINK) -lm

psrorbit: $(OBJDIR)psrorbit.o $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ $(PLOT2DOBJS) $(OBJDIR)psrorbit.o $(PRESTOLINK) $(PGPLOTLINK) -lm

testbinresp: $(OBJDIR)testbinresp.o $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)testbinresp.o $(PLOT2DOBJS) $(PGPLOTLINK) $(PRESTOLINK) -lm

quicklook: $(OBJDIR)quicklook.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)quicklook.o $(PRESTOLINK) -lm

readfile: readfile_cmd.c $(OBJDIR)readfile_cmd.o $(OBJDIR)readfile.o $(READFILEOBJS) $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)readfile.o $(OBJDIR)readfile_cmd.o $(READFILEOBJS) $(PRESTOLINK) -lcfitsio -lm

realfft: realfft_cmd.c $(OBJDIR)realfft_cmd.o $(OBJDIR)realfft.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)realfft.o $(OBJDIR)realfft_cmd.o $(PRESTOLINK) -lm

rednoise: rednoise_cmd.c $(OBJDIR)rednoise.o $(OBJDIR)rednoise_cmd.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)rednoise.o $(OBJDIR)rednoise_cmd.o $(PRESTOLINK) -lm

search_bin: search_bin_cmd.c $(OBJDIR)search_bin_cmd.o $(OBJDIR)search_bin.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)search_bin.o $(OBJDIR)search_bin_cmd.o $(PRESTOLINK) -lm

search_rzw: search_rzw_cmd.c $(OBJDIR)search_rzw_cmd.o $(OBJDIR)search_rzw.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)search_rzw.o $(OBJDIR)search_rzw_cmd.o $(PRESTOLINK) -lm

taperaw: $(OBJDIR)taperaw.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)taperaw.o

toas2dat: toas2dat_cmd.c $(OBJDIR)toas2dat_cmd.o $(OBJDIR)toas2dat.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)toas2dat.o $(OBJDIR)toas2dat_cmd.o

un_sc_td:
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ un_sc_td.c

swap_endian:
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ swap_endian.c

window: $(OBJDIR)window.o $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ $(PLOT2DOBJS) $(OBJDIR)window.o $(PRESTOLINK) $(PGPLOTLINK) -lm

zapbirds: zapbirds_cmd.c $(OBJDIR)zapbirds_cmd.o $(OBJDIR)zapbirds.o $(OBJDIR)zapping.o $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)zapbirds_cmd.o $(OBJDIR)zapbirds.o $(OBJDIR)zapping.o $(PLOT2DOBJS) $(PRESTOLINK) $(PGPLOTLINK) $(GLIBLINK) -lm

explorefft: $(OBJDIR)explorefft.o $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)explorefft.o $(PLOT2DOBJS) $(PRESTOLINK) $(PGPLOTLINK) -lm

exploredat: $(OBJDIR)exploredat.o $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)exploredat.o $(PLOT2DOBJS) $(PRESTOLINK) $(PGPLOTLINK) -lm

weight_psrfits: $(OBJDIR)weight_psrfits.o $(INSTRUMENTOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)weight_psrfits.o $(INSTRUMENTOBJS) $(PRESTOLINK)

psrfits_dumparrays: $(OBJDIR)psrfits_dumparrays.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)psrfits_dumparrays.o $(CFITSIOLINK)

fitsdelrow: $(OBJDIR)fitsdelrow.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)fitsdelrow.o $(CFITSIOLINK)

fitsdelcol: $(OBJDIR)fitsdelcol.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)fitsdelcol.o $(CFITSIOLINK)

bary: $(OBJDIR)bary.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)bary.o $(PRESTOLINK) -lm

bincand: bincand_cmd.c $(OBJDIR)bincand_cmd.o $(OBJDIR)bincand.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)bincand.o $(OBJDIR)bincand_cmd.o $(PRESTOLINK) -lm

dftfold: dftfold_cmd.c $(OBJDIR)dftfold_cmd.o $(OBJDIR)dftfold.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)dftfold.o $(OBJDIR)dftfold_cmd.o $(PRESTOLINK) -lm

shiftdata: $(OBJDIR)shiftdata.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)shiftdata.o -lm

patchdata: $(OBJDIR)patchdata.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)patchdata.o

dat2sdat: $(OBJDIR)dat2sdat.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)dat2sdat.o $(PRESTOLINK) -lm

sdat2dat: $(OBJDIR)sdat2dat.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)sdat2dat.o $(PRESTOLINK) -lm


###############  objects   ################

$(OBJDIR)%.cu.o: %.cu
	@mkdir -p $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

$(OBJDIR)%.o: %.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -o $@ -c $<

$(OBJDIR)%.o: %.cpp
	@mkdir -p $(OBJDIR)
	$(CPP) $(CXXFLAGS) -std=c++11 -o $@ -c $< $(INDARR)

$(OBJDIR)%.o: %.f
	@mkdir -p $(OBJDIR)
	$(FC) $(FFLAGS) -o $@ -c $<

slalib/%.o: slalib/%.f
	$(FC) $(FFLAGS) -fno-second-underscore -I. -o $@ -c $<

$(OBJDIR)mpiprepsubband_utils.o: mpiprepsubband_utils.c
	@mkdir -p $(OBJDIR)
	mpicc $(CLINKFLAGS) -o $@ -c $<

mpiprepsubband.o: mpiprepsubband.c
	@mkdir -p $(OBJDIR)
	mpicc $(CLINKFLAGS) -o $@ -c $<

$(OBJDIR)test_GPUaccel.o: test_GPUaccel.cpp
	@mkdir -p $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -std=c++11 -o $@ -c $< $(INDARR)

$(OBJDIR)getAccelLen.o: getAccelLen.cpp
	@mkdir -p $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -std=c++11 -o $@ -c $< $(INDARR)

###############  Cleans   ################

cleanALEN:
	# Debug
	rm -f build/*/accel_utils.o
	rm -f build/*/accelsearch.o
	rm -f build/*/test_GPUaccel.o

cleanPROFILE:
	rm -rf build/*/accelsearch*.o
	rm -rf build/*/cuda_accel_IN.cu.o
	rm -rf build/*/cuda_accel_KR.cu.o
	rm -rf build/*/cuda_accel_MU_CB.cu.o
	rm -rf build/*/cuda_accel_MU_IM.cu.o
	rm -rf build/*/cuda_accel_MU.cu.o
	rm -rf build/*/cuda_accel_SS_CPU.cu.o
	rm -rf build/*/cuda_accel_SS_INMEM.cu.o
	rm -rf build/*/cuda_accel_SS.cu.o
	rm -rf build/*/cuda_accel_utils.cu.o
	rm -rf build/*/cuda_cand_OPT.cu.o
	rm -rf build/*/cuda_utils.cu.o

cleanMULT:
	rm -rf build/*/accelsearch*.o
	rm -rf build/*/test_GPUaccel*.o
	rm -rf build/*/cuda_accel_utils.cu.o
	rm -rf build/*/cuda_accel_MU*.o

cleanSAS:
	rm -rf build/*/accelsearch*.o
	rm -rf build/*/test_GPUaccel*.o
	rm -rf build/*/cuda_accel_utils.cu.o
	rm -rf build/*/cuda_accel_SS*.o

cleanGPU: cudaclean
	rm -f ../lib/libpresto*
	rm -f ../lib/libcuaccel*
	rm -f ../bin/test_GPUaccel*
	rm -f ../bin/accelsearch*
	rm -f ../bin/getAccelLen*

cleancubasics:
	rm -f build/*/cuda_accel_IN.cu.o
	rm -f build/*/cuda_accel_KR.cu.o
	rm -f build/*/cuda_accel_SS.cu.o
	rm -f build/*/cuda_accel_utils.cu.o
	rm -f build/*/cuda_cand_OPT.cu.o
	rm -f build/*/cuda_utils.cu.cu.oF
	rm -f build/*/accelsearch.o
	rm -f build/*/test_GPUaccel.o

cudaclean:
	rm -f build/*/*cu.o
	rm -f build/*/cuda*.o
	rm -f build/*/accelsearch.o
	rm -f build/*/accel_utils.o
	rm -f build/*/test_GPUaccel.o

clean:
	rm -rf build/
	rm -f *.o *~ *#
	rm -f slalib/*.o slalib/sla_test

cleaner: clean
	cd ../bin ; rm -f $(BINARIES)
	rm -f $(PRESTO)/lib/libpresto* $(PRESTO)/lib/libsla.* $(PRESTO)/lib/libcuaccel*

distclean: cleaner cleanGPU

squeaky:  cleaner
	rm -f *.dat *.fft *.inf fftw_wisdom.txt
	rm -f core *.win* *.ps *_rzw *.tmp
	cd $(PRESTO)/clig ; rm -f *# *~
	cd $(PRESTO)/docs ; rm -f *# *~
	cd $(PRESTO)/python ; rm -f *# *~ *.o *.pyc *.pyo
	cd $(PRESTO)/include ; rm -f *# *~
