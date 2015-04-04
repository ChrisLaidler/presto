#  Makefile for PRESTO
#   by Scott M. Ransom

# OS type
OS = Linux
#OS = OSX

# yes / no
WITH_CUDA 	= yes
WITHOMP 		= yes
WITHNDARR 	= yes

OS_SIZE    = $(shell uname -m | sed -e "s/x86_64/64/" -e "s/armv7l/32/" -e "s/aarch64/64/")
OS_ARCH    = $(shell uname -m)
ARCH_FLAGS =

BINPATH				 = $(PRESTO)/bin
LIBPATH				 = $(PRESTO)/lib

PRESTOOBJS_BASE = amoeba.o atwood.o barycenter.o birdzap.o cand_output.o\
	characteristics.o cldj.o chkio.o corr_prep.o corr_routines.o\
	correlations.o database.o dcdflib.o dispersion.o\
	fastffts.o fftcalls.o fminbr.o fold.o fresnl.o ioinf.o\
	get_candidates.o iomak.o ipmpar.o maximize_r.o maximize_rz.o\
	median.o minifft.o misc_utils.o clipping.o\
	orbint.o output.o read_fft.o responses.o\
	rzinterp.o rzwinterp.o select.o sorter.o swapendian.o\
	transpose.o twopass.o twopass_real_fwd.o\
	twopass_real_inv.o vectors.o multifiles.o mask.o\
	fitsfile.o hget.o hput.o imio.o djcl.o

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
PGPLOTINC  = -I$(PGPLOT_DIR)
PGPLOTLINK = -L$(PGPLOT_DIR) -lcpgplot -lpgplot $(X11LINK) $(PNGLINK)

# Include and link information for the FFTW 3.X single-precision library
FFTINC := $(shell pkg-config --cflags fftw3f)
FFTLINK := $(shell pkg-config --libs fftw3f)

# Include and link information for the GLIB 2.0 library
GLIBINC := $(shell pkg-config --cflags glib-2.0)
GLIBLINK := $(shell pkg-config --libs glib-2.0)

# Include and link information for CFITSIO
CFITSIOINC := $(shell pkg-config --cflags cfitsio)
CFITSIOLINK := $(shell pkg-config --libs cfitsio) -lm

# The standard PRESTO libraries to link into executables
PRESTOLINK = $(CFITSIOLINK) -L$(PRESTO)/lib -lpresto$(POSTFIX) $(FFTLINK)

CC = gcc
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

# Very recent Intel CPUs might see a few percent speedup using -mavx
CFLAGS = -I$(PRESTO)/include $(GLIBINC) $(CFITSIOINC) $(PGPLOTINC) $(FFTINC) \
	-DUSEFFTW -DUSEMMAP -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 \
	-g -Wall -W -fPIC
#	 -march=native
#       -g -O3 -ffast-math -Wall -W -fPIC  # Debug builds now done with mode=debug
CLINKFLAGS = $(CFLAGS)
# NOTE:  Be careful of upping the optimization on the
#        FFLAGs.  Certain compilers (i.e. on Intel Macs) will
#        cause errors for the code in least_squares.f
FFLAGS = -g -fPIC
FLINKFLAGS = $(FFLAGS)
LINKCOMMAND1 = $(CC) $(LIBCMD) -o
LINKCOMMAND2 = ar rcs

CFLAGS_DBG = -I/home/chris/projects/libs/build/include

######## CUDA flags

ARCH=
#ARCH += -gencode arch=compute_20,code=sm_21
ARCH += -gencode arch=compute_30,code=sm_30
#ARCH += -gencode arch=compute_35,code=sm_35
ARCH += -gencode arch=compute_50,code=sm_50

CUDA_PATH	= /usr/local/cuda

CUDALIBS	= -L$(CUDA_PATH)/lib64 -lcudart -lcuda -lnvToolsExt -lcufft_static -lculibos
CUDAINC	 	= -I$(CUDA_PATH)/include

NVCCFLAGS	= $(CUDAINC) $(ARCH) -I$(PRESTO)/include $(GLIBINC) $(CFITSIOINC) -m${OS_SIZE} -dc ${ARCH_FLAGS}
NVCCFLAGS	+= -DUSEFFTW -DUSEMMAP -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 

NVCLFLAGS	= $(CUDAINC) $(ARCH) $(CUDALIBS)  
NVCLFLAGS	+= -DUSEFFTW -DUSEMMAP -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 

$(info )

#Settings dependent on release or debug build
ifeq ($(DEBUG),1)
  $(info Compiling in debug mode)

  CFLAGS 		+= -O0 -ggdb  -DDEBUG
  LFLAGS 		+= -O0 -ggdb
  
  NVCCFLAGS += -g -G -O0 -DDEBUG
  NVCLFLAGS += -g -G -O0

  DEBUG_TARGET	=debug
  POSTFIX				=_debug
  export DEBUG	=1    
  OBJDIR				=build/debug/
else  
  $(info Compileing in release mode)
  
  # CPU optimisations
  CFLAGS		+=-O3 -ffast-math 

  # Additional CPU optimisations
  CFLAGS		+=-unsafe-math -funroll-loops -funswitch-loops -fprefetch-loop-arrays
  # could use -mpc32  
  
  # vectorisation output
  # CFLAGS+=-ftree-vectorizer-verbose=2 -fdump-tree-vect-details
  
  NVCCFLAGS += -O3 -DFLOAT -DFORUM -use_fast_math -prec-div=false -ftz=true -prec-sqrt=false --fmad=true
#  NVCCFLAGS += -Xptxas -dlcm=ca
  NVCLFLAGS += -O3 -DFLOAT -DFORUM -use_fast_math -prec-div=false -ftz=true -prec-sqrt=false --fmad=true
  
  DEBUG_TARGET	=release
  POSTFIX				=
  export DEBUG	=0
  OBJDIR				=build/release/
endif

$(info )

# With open MP
ifeq ($(WITHOMP),yes)
  $(info • Compileing with OpenMP)
  
  CFLAGS 		+= -fopenmp -DWITHOMP
  FLAGS 		+= -fopenmp -DWITHOMP
  
  CXXFLAGS 		+= -fopenmp -DWITHOMP
  LDFLAGS 		+= -fopenmp -DWITHOMP
  
  FFTLINK 		+=
  NVCCFLAGS 	+= -Xcompiler -fopenmp -DWITHOMP
  NVCLFLAGS		+= -Xcompiler -fopenmp -DWITHOMP
endif

ifeq ($(WITHNDARR),yes)
  $(info • Compileing with Chris Debug libraries)

  # Include and link information for FreeType	
  FTCFLG = $(shell freetype-config --cflags)
  FTLINK = $(shell freetype-config --libs)

  LNDARR			+= -L$(SRC)/libs/build/lib -lndarray$(POSTFIX) -lcblutil$(POSTFIX) -lpngwriter -lpng $(FTLINK)
  INDARR			+= -I$(SRC)/libs/build/include/ $(FTCFLG) 
	
  NVCCFLAGS	    += -DCBL $(INDARR)
  NVCLFLAGS			+= -DCBL $(INDARR) -L$(SRC)/libs/build/lib -lndarray$(POSTFIX) -lcblutil$(POSTFIX) -lpngwriter -lpng $(FTLINK)
  
  CXXFLAGS			+= -DCBL $(INDARR)
else
  LNDARR =
  INDARR =  
endif

# With CUDA
ifeq ($(WITH_CUDA),yes)
  $(info • Compileing with CUDA)

  NVCC_SRCS 		= $(wildcard *.cu)
  NVCC_OBS  		= $(patsubst %.cu, $(OBJDIR)%.cu.o,  $(NVCC_SRCS)  )
  
  NVCCFLAGS		 	+= -DCUDA -Xcompiler -fPIC
  NVCLFLAGS			+= -DCUDA -Xcompiler -fPIC
  
  CFLAGS 				+= -DCUDA $(CUDAINC)
  FLAGS 				+= -DCUDA $(CUDAINC)
  
  CXXFLAGS 			+= -DCUDA $(CFLAGS)	$(CUDAINC)
  LDFLAGS 			+= -DCUDA $(CUDAINC) $(CLINKFLAGS) 

	CUACCL				= -lcuaccel$(POSTFIX)

	GPUOUTP				= $(BINPATH)/accelsearch$(POSTFIX) $(BINPATH)/test_GPUaccel$(POSTFIX) $(LIBPATH)/libcuaccel$(POSTFIX)$(LIBSUFFIX)
	
#  PRESTOLINK 	 	+= $(CUDALIBS)
#  PRESTOLINK 		+= $(GLIBLINK)
#  CC 				= $(NVCC) 
#  CXX				 = $(NVCC)
#  CLINKFLAGS 		= $(NVCLFLAGS)
#  CFLAGS 			= $(NVCCFLAGS)
  
#  CFLAGS 			 += -I/usr/local/cuda/include/ -DCUDA
  
#	PRESTOOBJS 	 	+= $(NVCC_OBS)
#  LINKCOMMAND1  	= $(NVCC) $(LIBCMD) -o
else
  NVCC      		= $(CC)
  NVCC_OBS  		=
  NVCLFLAGS 		= $(CLINKFLAGS)
	GPUOUTP				=
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

libpresto		= $(LIBPATH)/libpresto$(POSTFIX)$(LIBSUFFIX)
libcuaccel	= $(LIBPATH)/libcuaccel$(POSTFIX)$(LIBSUFFIX)
	
all: libpresto slalib binaries $(GPUOUTP)
	@echo ""
	@echo "Done ALL"	
	@echo ""

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
	@ # Make file cant handle blank tarkets so here is a radom comment =/

libcuaccel: $(libcuaccel)
	@ # Make file cant handle blank tarkets so here is a radom comment =/

binaries: $(BINARIES)
	@ # Make file cant handle blank tarkets so here is a radom comment =/

mpi: mpiprepsubband

GPU: $(BINPATH)/accelsearch$(POSTFIX) $(BINPATH)/test_GPUaccel$(POSTFIX) $(LIBPATH)/libcuaccel$(POSTFIX)$(LIBSUFFIX)
	@ # Make file cant handle blank tarkets so here is a radom comment =/

slalib: slalib/sla_test test_sla
	@ # Make file cant handle blank tarkets so here is a radom comment =/

test_GPUaccel: $(BINPATH)/test_GPUaccel$(POSTFIX)
	@ # Make file cant handle blank tarkets so here is a radom comment =/

accelsearch: $(BINPATH)/accelsearch$(POSTFIX)
	@ # Make file cant handle blank tarkets so here is a radom comment =/

###############   Librays   ################
	
$(LIBPATH)/libsla$(LIBSUFFIX): $(SLA_OBS)
	$(FC) $(LIBCMD) -o $@ -fno-second-underscore $(SLA_OBS)

$(LIBPATH)/libpresto$(POSTFIX)$(LIBSUFFIX): $(PRESTOOBJS)
	@echo ""
	@echo "Building libpresto$(POSTFIX)"	
	$(LINKCOMMAND1) $@ $(PRESTOOBJS) $(FFTLINK)
	@echo "Done libpresto$(POSTFIX)"
	@echo ""


$(LIBPATH)/libcuaccel$(POSTFIX)$(LIBSUFFIX): $(NVCC_OBS)
	@echo ""
	@echo "Building libcuaccel$(POSTFIX)"
	$(NVCC) $(LIBCMD) -o $@ $(NVCC_OBS) $(NVCLFLAGS)
	@echo "Done libcuaccel$(POSTFIX)"
	@echo ""
	
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
	@echo "Building accelsearch$(POSTFIX)"	
	$(NVCC) $(NVCLFLAGS) -o $@ $(NVCC_OBS) $(OBJDIR)accelsearch_cmd.o $(OBJDIR)accel_utils.o $(OBJDIR)accelsearch.o $(OBJDIR)zapping.o $(PRESTOLINK) $(GLIBLINK) -lm	
	@echo "Done accelsearch$(POSTFIX)"	
	@echo ""
	
$(BINPATH)/test_GPUaccel$(POSTFIX): $(libpresto) $(NVCC_OBS) $(OBJDIR)accelsearch_cmd.o $(OBJDIR)accel_utils.o $(OBJDIR)test_GPUaccel.o $(OBJDIR)zapping.o
	@echo ""
	@echo "Building test_GPUaccel"${POSTFIX}
	$(NVCC) $(NVCLFLAGS) -o $@ $(NVCC_OBS) $(OBJDIR)accelsearch_cmd.o $(OBJDIR)accel_utils.o $(OBJDIR)test_GPUaccel.o $(OBJDIR)zapping.o  $(PRESTOLINK) $(GLIBLINK) -lm
	@echo "Done test_GPUaccel${POSTFIX}"	
	@echo ""

check_parkes_raw: $(OBJDIR)check_parkes_raw.o $(OBJDIR)multibeam.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)check_parkes_raw.o $(OBJDIR)multibeam.o $(PRESTOLINK) -lm

downsample: downsample_cmd.c $(OBJDIR)downsample.o $(OBJDIR)downsample_cmd.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)downsample.o $(OBJDIR)downsample_cmd.o $(PRESTOLINK) -lm

split_parkes_beams: $(OBJDIR)split_parkes_beams.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)split_parkes_beams.o

test_multifiles: test_multifiles.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ test_multifiles.o $(PRESTOLINK) -lm

rfifind: rfifind_cmd.c rfifind_cmd.o rfifind.o rfi_utils.o rfifind_plot.o range_parse.o $(INSTRUMENTOBJS) $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@  $(INSTRUMENTOBJS) $(PLOT2DOBJS) rfifind.o rfi_utils.o rfifind_cmd.o rfifind_plot.o range_parse.o $(PRESTOLINK) $(PGPLOTLINK) -lcfitsio -lm

prepdata: prepdata_cmd.c $(OBJDIR)prepdata_cmd.o $(OBJDIR)prepdata.o $(INSTRUMENTOBJS) $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)prepdata.o $(OBJDIR)prepdata_cmd.o $(INSTRUMENTOBJS) $(PRESTOLINK) -lcfitsio -lm

prepsubband: prepsubband_cmd.c $(OBJDIR)prepsubband_cmd.o $(OBJDIR)prepsubband.o $(INSTRUMENTOBJS) $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)prepsubband.o $(OBJDIR)prepsubband_cmd.o $(INSTRUMENTOBJS) $(PRESTOLINK) -lcfitsio -lm

prepfold: prepfold_cmd.c $(OBJDIR)prepfold_cmd.o $(OBJDIR)prepfold.o $(OBJDIR)prepfold_utils.o $(OBJDIR)prepfold_plot.o $(OBJDIR)least_squares.o $(OBJDIR)polycos.o $(OBJDIR)readpar.o $(INSTRUMENTOBJS) $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)prepfold.o $(OBJDIR)prepfold_utils.o $(OBJDIR)prepfold_plot.o $(OBJDIR)prepfold_cmd.o $(OBJDIR)least_squares.o $(OBJDIR)polycos.o $(OBJDIR)readpar.o $(PLOT2DOBJS) $(INSTRUMENTOBJS) $(LAPACKLINK) $(PRESTOLINK) $(PGPLOTLINK) -lcfitsio -lm

dump_spigot_zerolag: dump_spigot_zerolag.o spigot.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ dump_spigot_zerolag.o spigot.o $(PRESTOLINK) -lm

spigot2filterbank: spigot2filterbank_cmd.c spigot2filterbank_cmd.o spigot2filterbank.o spigot.o sigproc_fb.o sla.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ spigot2filterbank.o spigot.o sigproc_fb.o spigot2filterbank_cmd.o sla.o $(PRESTOLINK) -lsla -lm

GBT350filterbank: GBT350filterbank.o spigot.o sigproc_fb.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ GBT350filterbank.o spigot.o sigproc_fb.o $(PRESTOLINK) -lm

spigotSband2filterbank: spigotSband2filterbank.o spigot.o sigproc_fb.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ spigotSband2filterbank.o spigot.o sigproc_fb.o $(PRESTOLINK) -lm

show_pfd: show_pfd_cmd.c show_pfd.o show_pfd_cmd.o prepfold_utils.o prepfold_plot.o least_squares.o range_parse.o $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ show_pfd.o show_pfd_cmd.o prepfold_utils.o prepfold_plot.o least_squares.o range_parse.o $(PLOT2DOBJS) $(LAPACKLINK) $(PRESTOLINK) $(PGPLOTLINK) -lm

makedata: $(OBJDIR)com.o $(OBJDIR)randlib.o $(OBJDIR)makedata.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)com.o $(OBJDIR)randlib.o $(OBJDIR)makedata.o $(PRESTOLINK) -lm

makeinf: $(OBJDIR)makeinf.o $(OBJDIR)ioinf.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)makeinf.o $(OBJDIR)ioinf.o $(PRESTOLINK) -lm

mjd2cal: djcl.o mjd2cal.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ djcl.o mjd2cal.o -lm

cal2mjd: cldj.o cal2mjd.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ cldj.o cal2mjd.o -lm

plotbincand: plotbincand.o $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ $(PLOT2DOBJS) plotbincand.o $(PRESTOLINK) $(PGPLOTLINK) -lm

profile: profile_cmd.c profile_cmd.o profile.o $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ $(PLOT2DOBJS) profile.o profile_cmd.o $(PRESTOLINK) $(PGPLOTLINK) -lm

psrorbit: psrorbit.o $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ $(PLOT2DOBJS) psrorbit.o $(PRESTOLINK) $(PGPLOTLINK) -lm

testbinresp: testbinresp.o $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ testbinresp.o $(PLOT2DOBJS) $(PGPLOTLINK) $(PRESTOLINK) -lm

quicklook: quicklook.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ quicklook.o $(PRESTOLINK) -lm

readfile: readfile_cmd.c readfile_cmd.o readfile.o $(READFILEOBJS) $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ readfile.o readfile_cmd.o $(READFILEOBJS) $(PRESTOLINK) -lcfitsio -lm

realfft: realfft_cmd.c $(OBJDIR)realfft_cmd.o $(OBJDIR)realfft.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)realfft.o $(OBJDIR)realfft_cmd.o $(PRESTOLINK) -lm

rednoise: rednoise_cmd.c rednoise.o rednoise_cmd.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ rednoise.o rednoise_cmd.o $(PRESTOLINK) -lm

search_bin: search_bin_cmd.c search_bin_cmd.o search_bin.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ search_bin.o search_bin_cmd.o $(PRESTOLINK) -lm

search_rzw: search_rzw_cmd.c search_rzw_cmd.o search_rzw.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ search_rzw.o search_rzw_cmd.o $(PRESTOLINK) -lm

taperaw: taperaw.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ taperaw.o

toas2dat: toas2dat_cmd.c toas2dat_cmd.o toas2dat.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ toas2dat.o toas2dat_cmd.o

un_sc_td:
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ un_sc_td.c

swap_endian:
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ swap_endian.c

window: window.o $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ $(PLOT2DOBJS) window.o $(PRESTOLINK) $(PGPLOTLINK) -lm

zapbirds: zapbirds_cmd.c zapbirds_cmd.o zapbirds.o zapping.o $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ zapbirds_cmd.o zapbirds.o zapping.o $(PLOT2DOBJS) $(PRESTOLINK) $(PGPLOTLINK) $(GLIBLINK) -lm

explorefft: $(OBJDIR)explorefft.o $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)explorefft.o $(PLOT2DOBJS) $(PRESTOLINK) $(PGPLOTLINK) -lm

exploredat: $(OBJDIR)exploredat.o $(PLOT2DOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ $(OBJDIR)exploredat.o $(PLOT2DOBJS) $(PRESTOLINK) $(PGPLOTLINK) -lm

weight_psrfits: weight_psrfits.o $(INSTRUMENTOBJS) $(libpresto)
	$(FC) $(FLINKFLAGS) -o $(PRESTO)/bin/$@ weight_psrfits.o $(INSTRUMENTOBJS) $(PRESTOLINK)

psrfits_dumparrays: psrfits_dumparrays.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ psrfits_dumparrays.o $(CFITSIOLINK)

fitsdelrow: fitsdelrow.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ fitsdelrow.o $(CFITSIOLINK)

fitsdelcol: fitsdelcol.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ fitsdelcol.o $(CFITSIOLINK)

bary: bary.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ bary.o $(PRESTOLINK) -lm

bincand: bincand_cmd.c bincand_cmd.o bincand.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ bincand.o bincand_cmd.o $(PRESTOLINK) -lm

dftfold: dftfold_cmd.c dftfold_cmd.o dftfold.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ dftfold.o dftfold_cmd.o $(PRESTOLINK) -lm

shiftdata: shiftdata.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ shiftdata.o -lm

patchdata: patchdata.o
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ patchdata.o

dat2sdat: dat2sdat.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ dat2sdat.o $(PRESTOLINK) -lm

sdat2dat: sdat2dat.o $(libpresto)
	$(CC) $(CLINKFLAGS) -o $(PRESTO)/bin/$@ sdat2dat.o $(PRESTOLINK) -lm


###############  objects   ################

$(OBJDIR)%.cu.o: %.cu
	@mkdir -p $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<
	
$(OBJDIR)%.o: %.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -o $@ -c $<

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

###############  Cleans   ################

cleanGPU:
	rm -f ../lib/libpresto*
	rm -f ../lib/libcuaccel*
	rm -f ../bin/test_GPUaccel*
	rm -f ../bin/accelsearch*
	rm -f cuda*.o
	rm -f accelsearch.o 
	rm -f accel_utils.o
	rm -f test_GPUaccel.o

cudaclean:
	rm -f build/debug/cuda*.o
	rm -f build/debug/accelsearch.o 
	rm -f build/debug/accel_utils.o
	rm -f build/debug/test_GPUaccel.o
	rm -f build/release/cuda*.o
	rm -f build/release/accelsearch.o 
	rm -f build/release/accel_utils.o
	rm -f build/release/test_GPUaccel.o

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
