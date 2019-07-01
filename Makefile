NAME = fpga-stream
HOST_FILE = $(NAME).c
HOST_BINARY = $(NAME)
KERNEL = $(NAME)-kernel
HOST_COMPILER = gcc
HOST_FLAGS = -O3 -Wall -Wextra -lrt -fopenmp
SRC_FOLDER = $(shell pwd)

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

	KERNEL_FLAGS = -v -v -v $(DASH)report

	ifeq ($(LEGACY),1)
		KERNEL_FLAGS += -DLEGACY
		HOST_FLAGS += -DLEGACY
	endif

	ifdef HOST_ONLY
		KERNEL_BINARY_STD = 
	else
		KERNEL_BINARY_STD = $(KERNEL)-std.aocx
		KERNEL_BINARY_CHSTD = $(KERNEL)-chstd.aocx
		KERNEL_BINARY_BLK2D = $(KERNEL)-blk2d.aocx
		KERNEL_BINARY_CHBLK2D = $(KERNEL)-chblk2d.aocx
		KERNEL_BINARY_BLK3D = $(KERNEL)-blk3d.aocx
		KERNEL_BINARY_CHBLK3D = $(KERNEL)-chblk3d.aocx
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

BLOCK_X ?= 1024
BLOCK_Y ?= 1024

ifdef BSIZE
	BLOCK_X=$(BSIZE)
	BLOCK_Y=$(BSIZE)
endif

std: HOST_FLAGS += -DSTD -DBLOCK_X=$(BLOCK_X)
std: KERNEL_FLAGS += -DBLOCK_X=$(BLOCK_X)
std: $(HOST_BINARY) $(KERNEL_BINARY_STD)

chstd: HOST_FLAGS += -DCHSTD -DBLOCK_X=$(BLOCK_X)
chstd: KERNEL_FLAGS += -DBLOCK_X=$(BLOCK_X)
chstd: $(HOST_BINARY) $(KERNEL_BINARY_CHSTD)

blk2d: HOST_FLAGS += -DBLK2D -DBLOCK_X=$(BLOCK_X)
blk2d: KERNEL_FLAGS += -DBLOCK_X=$(BLOCK_X)
blk2d: $(HOST_BINARY) $(KERNEL_BINARY_BLK2D)

chblk2d: HOST_FLAGS += -DCHBLK2D -DBLOCK_X=$(BLOCK_X)
chblk2d: KERNEL_FLAGS += -DBLOCK_X=$(BLOCK_X)
chblk2d: $(HOST_BINARY) $(KERNEL_BINARY_CHBLK2D)

blk3d: HOST_FLAGS += -DBLK3D -DBLOCK_X=$(BLOCK_X) -DBLOCK_Y=$(BLOCK_Y)
blk3d: KERNEL_FLAGS += -DBLOCK_X=$(BLOCK_X) -DBLOCK_Y=$(BLOCK_Y)
blk3d: $(HOST_BINARY) $(KERNEL_BINARY_BLK3D)

chblk3d: HOST_FLAGS += -DCHBLK3D -DBLOCK_X=$(BLOCK_X) -DBLOCK_Y=$(BLOCK_Y)
chblk3d: KERNEL_FLAGS += -DBLOCK_X=$(BLOCK_X) -DBLOCK_Y=$(BLOCK_Y)
chblk3d: $(HOST_BINARY) $(KERNEL_BINARY_CHBLK3D)

sch: HOST_FLAGS += -DSCH
sch: $(HOST_BINARY) $(KERNEL_BINARY_SCH)

fpga-stream: $(HOST_FILE)
	$(HOST_COMPILER) $(HOST_FLAGS) $< $(INC) $(LIB) -o $(HOST_BINARY)

%.aocx: KERNEL_BINARY = $(basename $@)_$(KERNEL_CONFIG)$(EXTRA_CONFIG)
%.aocx: %.cl
	mkdir -p $(FOLDER)
	-ln -sfn $(FOLDER)/$(KERNEL_BINARY).aocx $(KERNEL).aocx
	rm -rf $(FOLDER)/$(KERNEL_BINARY)*
	sh $(SRC_FOLDER)/override_fmax.sh $(FOLDER)/$(KERNEL_BINARY) $(FMAX)
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) $(SRC_FOLDER)/$< -o $(FOLDER)/$(KERNEL_BINARY)
	rm -rf $(FOLDER)/$(KERNEL_BINARY).aoco $(FOLDER)/$(KERNEL_BINARY).aocr

fpga_1: KERNEL_FLAGS += -DFPGA_1
fpga_1: KERNEL_BINARY = $(KERNEL)-sch_$(KERNEL_CONFIG)$(EXTRA_CONFIG)_FPGA_1
fpga_1: $(KERNEL)-sch.cl
	mkdir -p $(FOLDER)
	-ln -sfn $(FOLDER)/$(KERNEL_BINARY).aocx $(KERNEL)_FPGA_1.aocx
	rm -rf $(FOLDER)/$(KERNEL_BINARY)*
	sh $(SRC_FOLDER)/override_fmax.sh $(FOLDER)/$(KERNEL_BINARY) $(FMAX)
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) $(SRC_FOLDER)/$< -o $(FOLDER)/$(KERNEL_BINARY)
	rm -rf $(FOLDER)/$(KERNEL_BINARY).aoco $(FOLDER)/$(KERNEL_BINARY).aocr

fpga_2: KERNEL_FLAGS += -DFPGA_2
fpga_2: KERNEL_BINARY = $(KERNEL)-sch_$(KERNEL_CONFIG)$(EXTRA_CONFIG)_FPGA_2
fpga_2: $(KERNEL)-sch.cl
	mkdir -p $(FOLDER)
	-ln -sfn $(FOLDER)/$(KERNEL_BINARY).aocx $(KERNEL)_FPGA_2.aocx
	rm -rf $(FOLDER)/$(KERNEL_BINARY)*
	sh $(SRC_FOLDER)/override_fmax.sh $(FOLDER)/$(KERNEL_BINARY) $(FMAX)
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) $(SRC_FOLDER)/$< -o $(FOLDER)/$(KERNEL_BINARY)
	rm -rf $(FOLDER)/$(KERNEL_BINARY).aoco $(FOLDER)/$(KERNEL_BINARY).aocr

clean:
	rm -f $(HOST_BINARY)
	
clean-kernel:
	rm -rf *.aocx *.aoco *aocr *_VEC*