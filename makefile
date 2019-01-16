
NAME = fpga-stream
HOST = $(NAME).c
HOST_BINARY = $(NAME)
KERNEL = $(NAME).cl
HOST_COMPILER = gcc
HOST_FLAGS = -O3

ifneq ($(INTEL_FPGA),)
	INC = $(shell aocl compile-config)
	LIB = $(shell aocl link-config) -lOpenCL
	KERNEL_COMPILER = aoc
	HOST_FLAGS += -DINTEL_FPGA

	ifdef HOST_ONLY
		KERNEL_BINARY = 
	else
		KERNEL_BINARY = $(NAME).aocx
	endif
	
	ifdef NDR
		HOST_FLAGS += -DNDR
		KERNEL_FLAGS += -DNDR
	endif

	ifdef VEC
		HOST_FLAGS += -DVEC=$(VEC)
		KERNEL_FLAGS += -DVEC=$(VEC)
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
else ifneq ($(AMD),)
	KERNEL_BINARY =
	OPENCL_DIR = $(AMDAPPSDKROOT)
	INC += -I$(OPENCL_DIR)/include/
	LIB += -L$(OPENCL_DIR)/lib/x86_64/ -lOpenCL
endif

all: host kernel

host: $(HOST)
	$(HOST_COMPILER) $(HOST_FLAGS) $< $(INC) $(LIB) -o $(HOST_BINARY)
	
kernel: $(KERNEL)
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) $< -o $(KERNEL_BINARY)

clean:
rm -f $(HOST_BINARY)