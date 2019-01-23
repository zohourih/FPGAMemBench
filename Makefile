NAME = fpga-stream
HOST_FILE = $(NAME).c
HOST_BINARY = $(NAME)
KERNEL = $(NAME)-kernel
KERNEL_FILE = $(KERNEL).cl
HOST_COMPILER = gcc
HOST_FLAGS = -O3 -Wall -Wextra -lrt -fopenmp

ifeq ($(INTEL_FPGA),1)
	AOC_VERSION = $(shell aoc --version | grep Build | cut -c 9-10)
	LEGACY = $(shell echo $(AOC_VERSION)\<17 | bc)

	INC = $(shell aocl compile-config)
	LIB = $(shell aocl link-config) -lOpenCL
	KERNEL_COMPILER = aoc
	HOST_FLAGS += -DINTEL_FPGA
	ifeq ($(LEGACY),1)
		DASH = --
		SPACE = $(shell echo " ")
	else
		DASH = -
		SPACE = =
	endif
	KERNEL_FLAGS = -v $(DASH)report

	ifeq ($(LEGACY),1)
		KERNEL_FLAGS += -DLEGACY
	endif

	ifdef HOST_ONLY
		KERNEL_BINARY_STD = 
	else
		KERNEL_BINARY_STD = $(KERNEL)-std.aocx
		KERNEL_BINARY_CH = $(KERNEL)-ch.aocx
		KERNEL_BINARY_SERIAL_CH = $(KERNEL)-serial-ch.aocx
	endif

	ifdef EMULATOR
		HOST_FLAGS += -DEMULATOR
		KERNEL_FLAGS += -march=emulator
	endif

	ifdef BOARD
		KERNEL_FLAGS += $(DASH)board$(SPACE)$(BOARD)
	endif

	ifdef FMAX
		KERNEL_FLAGS += $(DASH)fmax$(SPACE)$(FMAX)
		EXTRA_CONFIG := $(EXTRA_CONFIG)_fmax$(FMAX)
	endif

	ifdef SEED
		KERNEL_FLAGS += $(DASH)seed$(SPACE)$(SEED)
		EXTRA_CONFIG := $(EXTRA_CONFIG)_seed$(SEED)
	endif

	ifdef NO_INTER
		HOST_FLAGS += -DNO_INTERLEAVE -Wno-unknown-pragmas
		KERNEL_FLAGS += $(DASH)no-interleaving$(SPACE)default
		EXTRA_CONFIG := $(EXTRA_CONFIG)_nointer
	endif

	ifndef CACHE
		KERNEL_FLAGS += $(DASH)opt-arg$(SPACE)-nocaching
		EXTRA_CONFIG := $(EXTRA_CONFIG)_nocache
	endif
else ifeq ($(AMD),1)
	KERNEL_BINARY_STD =
	OPENCL_DIR = $(AMDAPPSDKROOT)
	INC += -I$(OPENCL_DIR)/include/
	LIB += -L$(OPENCL_DIR)/lib/x86_64/ -lOpenCL
	HOST_FLAGS += -Wno-deprecated-declarations
else ifeq ($(NVIDIA),1)
	KERNEL_BINARY_STD =
	INC += -I$(CUDA_DIR)/include/
	LIB += -L$(CUDA_DIR)/lib64/ -lOpenCL
	HOST_FLAGS += -Wno-deprecated-declarations
endif

ifdef NDR
	HOST_FLAGS += -DNDR
	KERNEL_FLAGS += -DNDR

	ifdef WGS
		HOST_FLAGS += -DWGS=$(WGS)
		KERNEL_FLAGS += -DWGS=$(WGS)
	endif
endif

ifdef VEC
	HOST_FLAGS += -DVEC=$(VEC)
	KERNEL_FLAGS += -DVEC=$(VEC)
endif

ifdef NDR
	KERNEL_CONFIG = NDR_WGS$(WGS)_VEC$(VEC)
else
	KERNEL_CONFIG = SWI_VEC$(VEC)
endif

std: $(HOST_BINARY) $(KERNEL_BINARY_STD)

ch: $(HOST_BINARY) $(KERNEL_BINARY_CH)

serial_ch: $(HOST_BINARY) $(KERNEL_BINARY_SERIAL_CH)

$(HOST_BINARY): $(HOST_FILE)
	$(HOST_COMPILER) $(HOST_FLAGS) $< $(INC) $(LIB) -o $(HOST_BINARY)

%.aocx: KERNEL_BINARY = $(basename $@)_$(KERNEL_CONFIG)$(EXTRA_CONFIG).aocx
%.aocx: %.cl
	ln -sfn $(KERNEL_BINARY) fpga-stream-kernel.aocx
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) $< -o $(KERNEL_BINARY)

clean:
	rm -f $(HOST_BINARY)
	
clean-kernel:
	rm -rf *.aocx *.aoco rm -rf *.aocx *.aoco *_VEC*