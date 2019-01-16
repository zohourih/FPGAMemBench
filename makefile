
NAME = fpga-stream
HOST = $(NAME).c
HOST_BINARY = $(NAME)
KERNEL = $(NAME).cl
HOST_COMPILER = gcc
HOST_FLAGS = -O3

ifeq ($(INTEL_FPGA),1)
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
		-Wdeprecated-declarations-DNO_INTERLEAVE
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

$(HOST_BINARY): $(HOST)
	$(HOST_COMPILER) $(HOST_FLAGS) $< $(INC) $(LIB) -o $(HOST_BINARY)
	
$(KERNEL_BINARY): $(KERNEL)
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) $< -o $(KERNEL_BINARY)

clean:
	rm -f $(HOST_BINARY)