NAME = fpga-stream
HOST_FILE = $(NAME).c
HOST_BINARY = $(NAME)
KERNEL = $(NAME)-kernel
HOST_COMPILER = gcc
HOST_FLAGS = -O3 -Wall -Wextra -lrt -fopenmp
SRC_FOLDER = $(shell pwd)

ifeq ($(INTEL_FPGA),1)
	TEMP = $(shell quartus_fit --version | grep "Pro")
	ifeq ($(TEMP),)
		AOC_PRO = 0
	else
		AOC_PRO = 1
	endif
	AOC_VERSION = $(shell aoc --version | grep Build | cut -c 9-10)
	LEGACY = $(shell echo $(AOC_VERSION)\<17 | bc)
	RTL = $(shell echo $(AOC_VERSION)\>=18 | bc)

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
	ifeq ($(RTL),1)
		ifeq ($(AOC_PRO),1)
			SWITCH = -rtl
			EXT = aocr
		else
			SWITCH = -c
			EXT = aoco
		endif
	else
		SWITCH = -c
		EXT = aoco
	endif
	KERNEL_FLAGS = -v $(DASH)report

	ifeq ($(LEGACY),1)
		KERNEL_FLAGS += -DLEGACY
		HOST_FLAGS += -DLEGACY
	endif

	ifdef HOST_ONLY
		KERNEL_BINARY_STD = 
	else
		KERNEL_BINARY_STD = $(KERNEL)-std.aocx
		KERNEL_BINARY_CH = $(KERNEL)-ch.aocx
		KERNEL_BINARY_BLK2D = $(KERNEL)-blk2d.aocx
		KERNEL_BINARY_SCH = $(shell echo "fpga_1 fpga_2")
	endif

	ifdef KERNEL_ONLY
		HOST_BINARY = 
	endif

	ifdef EMULATOR
		HOST_FLAGS += -DEMULATOR
		KERNEL_FLAGS += -march=emulator
	endif

	ifdef BOARD
		KERNEL_FLAGS += $(DASH)board$(SPACE)$(BOARD)
	endif

	ifeq ($(NO_INTER),1)
		HOST_FLAGS += -DNO_INTERLEAVE -Wno-unknown-pragmas
		KERNEL_FLAGS += $(DASH)no-interleaving$(SPACE)default
		EXTRA_CONFIG := $(EXTRA_CONFIG)_nointer
	endif

	ifeq ($(NO_CACHE),1)
		KERNEL_FLAGS += $(DASH)opt-arg$(SPACE)-nocaching
		EXTRA_CONFIG := $(EXTRA_CONFIG)_nocache
	endif

	ifdef TFMAX
		KERNEL_FLAGS += $(DASH)fmax$(SPACE)$(TFMAX)
		EXTRA_CONFIG := $(EXTRA_CONFIG)_tfmax$(TFMAX)
	endif

	ifdef FMAX
		EXTRA_CONFIG := $(EXTRA_CONFIG)_fmax$(FMAX)
	endif

	ifdef SEED
		KERNEL_FLAGS += $(DASH)seed$(SPACE)$(SEED)
		EXTRA_CONFIG := $(EXTRA_CONFIG)_seed$(SEED)
	endif

	ifdef FOLDER
		FOLDER=$(FOLDER)
	else
		FOLDER=.
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

VEC ?= 1
HOST_FLAGS += -DVEC=$(VEC)
KERNEL_FLAGS += -DVEC=$(VEC)

ifeq ($(NDR),1)
	HOST_FLAGS += -DNDR
	KERNEL_FLAGS += -DNDR

	WGS ?= 64
	HOST_FLAGS += -DWGS=$(WGS)
	KERNEL_FLAGS += -DWGS=$(WGS)
	
	KERNEL_CONFIG = NDR_VEC$(VEC)
else
	KERNEL_CONFIG = SWI_VEC$(VEC)
endif

BSIZE ?= 1024

std: HOST_FLAGS += -DSTD -DBSIZE=$(BSIZE)
std: KERNEL_FLAGS += -DBSIZE=$(BSIZE)
std: $(HOST_BINARY) $(KERNEL_BINARY_STD)

ch: HOST_FLAGS += -DCH
ch: $(HOST_BINARY) $(KERNEL_BINARY_CH)

blk2d: HOST_FLAGS += -DBLK2D -DBSIZE=$(BSIZE)
blk2d: KERNEL_FLAGS += -DBSIZE=$(BSIZE)
blk2d: $(HOST_BINARY) $(KERNEL_BINARY_BLK2D)

sch: HOST_FLAGS += -DSCH
sch: $(HOST_BINARY) $(KERNEL_BINARY_SCH)

fpga-stream: $(HOST_FILE)
	$(HOST_COMPILER) $(HOST_FLAGS) $< $(INC) $(LIB) -o $(HOST_BINARY)

%.aocx: KERNEL_BINARY = $(basename $@)_$(KERNEL_CONFIG)$(EXTRA_CONFIG)
#To bypass the auto conversion of "-" to "_" in AOC_VERSION < 17
%.aocx: KERNEL_BINARY_ALTER = $(shell echo -n $(KERNEL_BINARY) | sed 's/fpga-stream-kernel-/fpga_stream_kernel_/' | sed 's/ //g')
%.aocx: %.cl
	mkdir -p $(FOLDER)
	-ln -sfn $(FOLDER)/$(KERNEL_BINARY).aocx $(KERNEL).aocx
	cd $(FOLDER) && \
	rm -rf $(KERNEL_BINARY)* && \
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) $(SRC_FOLDER)/$< -o $(KERNEL_BINARY_ALTER) $(SWITCH) && \
	sh $(SRC_FOLDER)/override_fmax.sh $(KERNEL_BINARY_ALTER) $(FMAX) && \
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) $(KERNEL_BINARY_ALTER).$(EXT) && \
	mv $(KERNEL_BINARY_ALTER) $(KERNEL_BINARY) && \
	mv $(KERNEL_BINARY_ALTER).aocx $(KERNEL_BINARY).aocx && rm -rf $(KERNEL_BINARY_ALTER).aoc* && \
	cd ..

fpga_1: KERNEL_FLAGS += -DFPGA_1
fpga_1: KERNEL_BINARY = $(KERNEL)-sch_$(KERNEL_CONFIG)$(EXTRA_CONFIG)_FPGA_1
fpga_1: KERNEL_BINARY_ALTER = $(shell echo -n $(KERNEL_BINARY) | sed 's/fpga-stream-kernel-/fpga_stream_kernel_/' | sed 's/ //g')
fpga_1: $(KERNEL)-sch.cl
	mkdir -p $(FOLDER)
	-ln -sfn $(FOLDER)/$(KERNEL_BINARY).aocx $(KERNEL)_FPGA_1.aocx
	cd $(FOLDER) && \
	rm -rf $(KERNEL_BINARY)* && \
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) $(SRC_FOLDER)/$< -o $(KERNEL_BINARY_ALTER) $(SWITCH) && \
	sh $(SRC_FOLDER)/override_fmax.sh $(KERNEL_BINARY_ALTER) $(FMAX) && \
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) $(KERNEL_BINARY_ALTER).$(EXT) && \
	mv $(KERNEL_BINARY_ALTER) $(KERNEL_BINARY) && \
	mv $(KERNEL_BINARY_ALTER).aocx $(KERNEL_BINARY).aocx && rm -rf $(KERNEL_BINARY_ALTER).aoc* && \
	cd ..

fpga_2: KERNEL_FLAGS += -DFPGA_2
fpga_2: KERNEL_BINARY = $(KERNEL)-sch_$(KERNEL_CONFIG)$(EXTRA_CONFIG)_FPGA_2
fpga_2: KERNEL_BINARY_ALTER = $(shell echo -n $(KERNEL_BINARY) | sed 's/fpga-stream-kernel-/fpga_stream_kernel_/' | sed 's/ //g')
fpga_2: $(KERNEL)-sch.cl
	mkdir -p $(FOLDER)
	-ln -sfn $(FOLDER)/$(KERNEL_BINARY).aocx $(KERNEL)_FPGA_2.aocx
	cd $(FOLDER) && \
	rm -rf $(KERNEL_BINARY)* && \
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) $(SRC_FOLDER)/$< -o $(KERNEL_BINARY_ALTER) $(SWITCH) && \
	sh $(SRC_FOLDER)/override_fmax.sh $(KERNEL_BINARY_ALTER) $(FMAX) && \
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) $(KERNEL_BINARY_ALTER).$(EXT) && \
	mv $(KERNEL_BINARY_ALTER) $(KERNEL_BINARY) && \
	mv $(KERNEL_BINARY_ALTER).aocx $(KERNEL_BINARY).aocx && rm -rf $(KERNEL_BINARY_ALTER).aoc* && \
	cd ..

clean:
	rm -f $(HOST_BINARY)
	
clean-kernel:
	rm -rf *.aocx *.aoco *aocr *_VEC*