CC=gcc
CFLAGS=-std=c17 -I. -Wall -Wextra -Wpedantic -DCL_TARGET_OPENCL_VERSION=300 -g3 -O3
LIBS=-lOpenCL -lm -lgomp

DEPS=clu_errcheck.h clu_setup.h
SRC=clu_setup.c

# ================================
#  Parameter Flags
# ================================
FLAGS=
SUFFIX=

ifdef N
FLAGS+=-DN=$(N)
SUFFIX:=_N$(N)
endif

ifdef FLOAT
FLAGS+=-DFLOAT
SUFFIX:=$(SUFFIX)_float
endif

ifdef LOCAL_WORKGROUP_SIZE
FLAGS+=-DLOCAL_WORKGROUP_SIZE=$(LOCAL_WORKGROUP_SIZE)
SUFFIX:=$(SUFFIX)_WG$(LOCAL_WORKGROUP_SIZE)
endif

ifdef DETAILED
FLAGS+=-DDETAILED_TIMING
SUFFIX:=$(SUFFIX)_detailed
endif

# ================================
#  Targets
# ================================
all: seq$(SUFFIX) wg$(SUFFIX) multistage$(SUFFIX)


# ------------------------------------------------------------
#  1. Sequential Reduction
# ------------------------------------------------------------
seq$(SUFFIX): sequential_reduction.c $(SRC) $(DEPS)
	$(CC) -o $@ $< $(SRC) $(CFLAGS) $(FLAGS) $(LIBS)


# ------------------------------------------------------------
#  2. Work-group Parallel Reduction
# ------------------------------------------------------------
wg$(SUFFIX): wg_reduction.c $(SRC) $(DEPS)
	$(CC) -o $@ $< $(SRC) $(CFLAGS) $(FLAGS) $(LIBS)


# ------------------------------------------------------------
#  3. Multi-Stage Reduction (arbitrary vector sizes)
# ------------------------------------------------------------
multistage$(SUFFIX): multistage_reduction.c $(SRC) $(DEPS)
	$(CC) -o $@ $< $(SRC) $(CFLAGS) $(FLAGS) $(LIBS)


# ================================
#  Cleanup
# ================================
clean:
	rm -f seq_* wg_* multistage_* \
	      seq wg multistage
