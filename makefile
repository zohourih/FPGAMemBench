
NAME = fpga-stream
HOST_FILE = $(NAME).c
HOST_BINARY = $(NAME)
KERNEL = $(NAME)-kernel
KERNEL_FILE = $(KERNEL).cl
HOST_COMPILER = gcc
HOST_FLAGS = -O3 -fopenmp -lrt

ifeq ($(INTEL_FPGA),1)
	INC = $(shell aocl compile-config)
	LIB = $(shell aocl link-config) -lOpenCL
	KERNEL_COMPILER = aoc
	HOST_FLAGS += -DINTEL_FPGA
	KERNEL_FLAGS = -v --report

	ifdef HOST_ONLY
		KERNEL_BINARY = 
	else
		KERNEL_BINARY = $(KERNEL).aocx
	endif

	ifdef EMULATOR
		HOST_FLAGS += -DEMULATOR
		KERNEL_FLAGS += -march=emulator
	endif
	
	ifdef NDR
		HOST_FLAGS += -DNDR
		KERNEL_FLAGS += -DNDR
	endif

	ifdef VEC
		HOST_FLAGS += -DVEC=$(VEC)
		KERNEL_FLAGS += -DVEC=$(VEC)
	endif

	ifdef WGS
		HOST_FLAGS += -DWGS=$(WGS)
		KERNEL_FLAGS += -DWGS=$(WGS)
	endif
	
	ifdef NO_INTER
		HOST_FLAGS += -DNO_INTERLEAVE
		KERNEL_FLAGS += --no-interleaving default
	endif

	ifdef BOARD
		KERNEL_FLAGS += --board $(BOARD)
	endif

	ifdef FMAX
		KERNEL_FLAGS += --fmax $(FMAX)
	endif

	ifdef SEED
		KERNEL_FLAGS += --seed $(SEED)
	endif
else ifeq ($(AMD),1)
	KERNEL_BINARY =
	OPENCL_DIR = $(AMDAPPSDKROOT)
	INC += -I$(OPENCL_DIR)/include/
	LIB += -L$(OPENCL_DIR)/lib/x86_64/ -lOpenCL
	HOST_FLAGS += -Wno-deprecated-declarations
endif

all: $(HOST_BINARY) $(KERNEL_BINARY)

$(HOST_BINARY): $(HOST_FILE)
	$(HOST_COMPILER) $(HOST_FLAGS) $< $(INC) $(LIB) -o $(HOST_BINARY)
	
$(KERNEL_BINARY): $(KERNEL_FILE)
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) $< -o $(KERNEL_BINARY)

clean:
	rm -f $(HOST_BINARY)
	
clean-kernel:
	rm -rf $(KERNEL).aocx $(KERNEL).aoco $(KERNEL)