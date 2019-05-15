//====================================================================================================================================
// OpenCL-based memory bandwidth benchmark for OpenCL-capable FPGAs
// (c) 2019, Hamid Reza Zohouri @ Tokyo Institute of Technology
//====================================================================================================================================

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <CL/cl.h>

#include "common/util.h"
#include "common/timer.h"

#ifdef NO_INTERLEAVE
	#include "CL/cl_ext.h"
#endif

#ifdef LEGACY
	#define MEM_BANK_1 CL_MEM_BANK_1_ALTERA
	#define MEM_BANK_2 CL_MEM_BANK_2_ALTERA
#else
	#define MEM_BANK_1 CL_CHANNEL_1_INTELFPGA
	#define MEM_BANK_2 CL_CHANNEL_2_INTELFPGA
#endif

#ifdef BLK2D
	#define DIM 2
#else
	#define DIM 1
#endif

// global variables
static cl_context       context;
#if defined(STD) || defined(BLK2D)
static cl_command_queue queue;
#elif defined(CH) || defined(SCH)
static cl_command_queue queue_read, queue_write;
#endif
static cl_device_id*    deviceList;
static cl_int           deviceCount;

static inline void init()
{
	size_t deviceSize;
	cl_int error;
	cl_uint platformCount;
	cl_platform_id* platforms = NULL;
	cl_device_type   deviceType;
	cl_context_properties ctxprop[3];

	display_device_info(&platforms, &platformCount);
	select_device_type(&deviceType);
	validate_selection(platforms, &platformCount, ctxprop, &deviceType);
	
	// create OpenCL context
	context = clCreateContextFromType(ctxprop, deviceType, NULL, NULL, &error);
	if(!context)
	{
		printf("ERROR: clCreateContextFromType(%s) failed with error code: ", (deviceType == CL_DEVICE_TYPE_ACCELERATOR) ? "FPGA" : (deviceType == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU");
		display_error_message(error, stdout);
		exit(-1);
	}

	// get list of devices
	CL_SAFE_CALL( clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceSize) );
	deviceCount = (int) (deviceSize / sizeof(cl_device_id));
	if(deviceCount < 1)
	{
		printf("ERROR: No devices found.\n");
		exit(-1);
	}

	// allocate memory for devices
	deviceList = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
	if(!deviceList)
	{
		printf("ERROR: malloc(deviceList) failed.\n");
		exit(-1);
	}

	CL_SAFE_CALL( clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceSize, deviceList, NULL) );

	// create command queue for the first device
#if defined(STD) || defined(BLK2D)
	queue = clCreateCommandQueue(context, deviceList[0], 0, NULL);
	if(!queue)
	{
		printf("ERROR: clCreateCommandQueue(queue) failed with error code: ");
		display_error_message(error, stdout);
		exit(-1);
	}
#elif CH
	queue_read = clCreateCommandQueue(context, deviceList[0], 0, NULL);
	if(!queue_read)
	{
		printf("ERROR: clCreateCommandQueue(queue_read) failed with error code: ");
		display_error_message(error, stdout);
		exit(-1);
	}

	queue_write = clCreateCommandQueue(context, deviceList[0], 0, NULL);
	if(!queue_write)
	{
		printf("ERROR: clCreateCommandQueue(queue_write) failed with error code: ");
		display_error_message(error, stdout);
		exit(-1);
	}
#elif SCH
	// FPGA_1
	queue_read = clCreateCommandQueue(context, deviceList[0], 0, NULL);
	if(!queue_read)
	{
		printf("ERROR: clCreateCommandQueue(queue_read) failed with error code: ");
		display_error_message(error, stdout);
		exit(-1);
	}

	// FPGA_2
	queue_write = clCreateCommandQueue(context, deviceList[1], 0, NULL);
	if(!queue_write)
	{
		printf("ERROR: clCreateCommandQueue(queue_write) failed with error code: ");
		display_error_message(error, stdout);
		exit(-1);
	}
#endif

	free(platforms); // platforms isn't needed in the main function
}

static inline void usage(char **argv)
{
#ifdef STD
	printf("\nUsage: %s -s <buffer size in MiB> -n <number of iterations> -p <number of padding indexes> -o <number of overlapped indexes> --verbose --verify\n", argv[0]);
#elif BLK2D
	printf("\nUsage: %s -r <row width> -c <column height> -n <number of iterations> -p <number of padding indexes> -hw <halo width> --verbose --verify\n", argv[0]);
#else
	printf("\nUsage: %s -s <buffer size in MiB> -n <number of iterations> -p <number of padding indexes> --verbose --verify\n", argv[0]);
#endif
}

int main(int argc, char **argv)
{
	// input arguments
	int size_MiB = 100; 							// buffer size, default size is 100 MiB
	int iter = 1;									// number of iterations
	int pad = 0;									// padding
	int verbose = 0, verify = 0;
#ifdef STD
	int overlap = 0;
#elif BLK2D
	int halo = 0;
	int rows = 5120;
	int cols = 5120;
#endif

	// timing measurement
	TimeStamp start, end;
	double totalCopyTime = 0, avgCopyTime = 0;
#ifndef SCH
	double totalMacTime = 0, avgMacTime = 0;
#endif

	// for OpenCL errors
	cl_int error = 0;

	int arg = 1;
	while (arg < argc)
	{
	#ifndef BLK2D
		if(strcmp(argv[arg], "-s") == 0)
		{
			size_MiB = atoi(argv[arg + 1]);
			arg += 2;
		}
	#else
		if(strcmp(argv[arg], "-r") == 0)
		{
			rows = atoi(argv[arg + 1]);
			arg += 2;
		}
		if(strcmp(argv[arg], "-c") == 0)
		{
			cols = atoi(argv[arg + 1]);
			arg += 2;
		}
	#endif
		else if (strcmp(argv[arg], "-n") == 0)
		{
			iter = atoi(argv[arg + 1]);
			arg += 2;
		}
		else if (strcmp(argv[arg], "-p") == 0)
		{
			pad = atoi(argv[arg + 1]);
			arg += 2;
		}
	#ifdef STD
		else if (strcmp(argv[arg], "-o") == 0)
		{
			overlap = atoi(argv[arg + 1]);
			arg += 2;
		}
	#elif BLK2D
		else if (strcmp(argv[arg], "-hw") == 0)
		{
			halo = atoi(argv[arg + 1]);
			arg += 2;
		}
	#endif
		else if (strcmp(argv[arg], "--verbose") == 0)
		{
			verbose = 1;
			arg += 1;
		}
		else if (strcmp(argv[arg], "--verify") == 0)
		{
			verify = 1;
			arg += 1;
		}
		else if (strcmp(argv[arg], "-h") == 0 || strcmp(argv[arg], "--help") == 0)
		{
			usage(argv);
			return 0;
		}
		else
		{
			printf("\nInvalid input!");
			usage(argv);
			return -1;
		}
	}

	// set array size based in input buffer size, default is 256k floats (= 100 MiB)
#ifdef BLK2D
	size_MiB = rows * cols * sizeof(float) / (1024 * 1024);
#endif
	long size_B = (long)size_MiB * 1024 * 1024;
	long array_size = size_B / sizeof(float);
	long padded_array_size = array_size + pad;
	long padded_size_Byte = padded_array_size * sizeof(float);
	int  padded_size_MiB = padded_size_Byte / (1024 * 1024);

	// OpenCL initialization
	init();

	// load kernel file and build program
#ifdef INTEL_FPGA
	#ifdef SCH
		size_t kernelFileSizeFPGA1, kernelFileSizeFPGA2;
		char *kernelSourceFPGA1 = read_kernel("fpga-stream-kernel_FPGA_1.aocx", &kernelFileSizeFPGA1);
		cl_program progFPGA1 = clCreateProgramWithBinary(context, 1, &deviceList[0], &kernelFileSizeFPGA1, (const unsigned char**)&kernelSourceFPGA1, NULL, &error);
		if(error != CL_SUCCESS)
		{
			printf("ERROR: clCreateProgramWithBinary(FPGA1) failed with error: ");
			display_error_message(error, stdout);
			return -1;
		}

		char *kernelSourceFPGA2 = read_kernel("fpga-stream-kernel_FPGA_2.aocx", &kernelFileSizeFPGA2);
		cl_program progFPGA2 = clCreateProgramWithBinary(context, 1, &deviceList[1], &kernelFileSizeFPGA2, (const unsigned char**)&kernelSourceFPGA2, NULL, &error);
		if(error != CL_SUCCESS)
		{
			printf("ERROR: clCreateProgramWithBinary(FPGA2) failed with error: ");
			display_error_message(error, stdout);
			return -1;
		}
	#else
		size_t kernelFileSize;
		char *kernelSource = read_kernel("fpga-stream-kernel.aocx", &kernelFileSize);
		cl_program prog = clCreateProgramWithBinary(context, deviceCount, deviceList, &kernelFileSize, (const unsigned char**)&kernelSource, NULL, &error);
		if(error != CL_SUCCESS)
		{
			printf("ERROR: clCreateProgramWithBinary() failed with error: ");
			display_error_message(error, stdout);
			return -1;
		}
	#endif
#else // for CPU/GPUs
	#if defined(STD) || defined(BLK2D)
		size_t kernelFileSize;
		#ifdef STD
			char *kernelSource = read_kernel("fpga-stream-kernel-std.cl", &kernelFileSize);
		#else
			char *kernelSource = read_kernel("fpga-stream-kernel-blk2d.cl", &kernelFileSize);
		#endif
		cl_program prog = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &error);
		if(error != CL_SUCCESS)
		{
			printf("ERROR: clCreateProgramWithSource() failed with error: ");
			display_error_message(error, stdout);
			return -1;
		}
	#else
		printf("Kernel not supported on this device!\n");
		return -1;
	#endif
#endif

	char clOptions[200] = "";

#ifndef INTEL_FPGA
	sprintf(clOptions + strlen(clOptions), "-DVEC=%d -DBSIZE=%d ", VEC, BSIZE);
#endif

#ifdef NDR
	sprintf(clOptions + strlen(clOptions), "-DNDR");
#endif

	// compile kernel file
#ifdef SCH
	clBuildProgram_SAFE(progFPGA1, 1, &deviceList[0], clOptions, NULL, NULL);
	clBuildProgram_SAFE(progFPGA2, 1, &deviceList[1], clOptions, NULL, NULL);
#else
	clBuildProgram_SAFE(prog, deviceCount, deviceList, clOptions, NULL, NULL);
#endif

	// create kernel objects
#if defined(STD) || defined(BLK2D)
	cl_kernel copyKernel, macKernel;

	copyKernel = clCreateKernel(prog, "copy", &error);
	if(error != CL_SUCCESS)
	{
		printf("ERROR: clCreateKernel(copy) failed with error: ");
		display_error_message(error, stdout);
		return -1;
	}

	macKernel = clCreateKernel(prog, "mac", &error);
	if(error != CL_SUCCESS)
	{
		printf("ERROR: clCreateKernel(mac) failed with error: ");
		display_error_message(error, stdout);
		return -1;
	}

	clReleaseProgram(prog);
#elif CH
	cl_kernel copyReadKernel, copyWriteKernel, macReadKernel, macWriteKernel;

	copyReadKernel = clCreateKernel(prog, "copy_read", &error);
	if(error != CL_SUCCESS)
	{
		printf("ERROR: clCreateKernel(copy_read) failed with error: ");
		display_error_message(error, stdout);
		return -1;
	}

	copyWriteKernel = clCreateKernel(prog, "copy_write", &error);
	if(error != CL_SUCCESS)
	{
		printf("ERROR: clCreateKernel(copy_write) failed with error: ");
		display_error_message(error, stdout);
		return -1;
	}

	macReadKernel = clCreateKernel(prog, "mac_read", &error);
	if(error != CL_SUCCESS)
	{
		printf("ERROR: clCreateKernel(mac_write) failed with error: ");
		display_error_message(error, stdout);
		return -1;
	}

	macWriteKernel = clCreateKernel(prog, "mac_write", &error);
	if(error != CL_SUCCESS)
	{
		printf("ERROR: clCreateKernel(mac_write) failed with error: ");
		display_error_message(error, stdout);
		return -1;
	}

	clReleaseProgram(prog);
#elif SCH
	cl_kernel copyReadKernel, copyWriteKernel;

	copyReadKernel = clCreateKernel(progFPGA1, "copy_read", &error);
	if(error != CL_SUCCESS)
	{
		printf("ERROR: clCreateKernel(copy_read) failed with error: ");
		display_error_message(error, stdout);
		return -1;
	}

	copyWriteKernel = clCreateKernel(progFPGA2, "copy_write", &error);
	if(error != CL_SUCCESS)
	{
		printf("ERROR: clCreateKernel(copy_write) failed with error: ");
		display_error_message(error, stdout);
		return -1;
	}

	clReleaseProgram(progFPGA1);
	clReleaseProgram(progFPGA2);
#endif

#ifdef STD
	printf("Kernel type:           Standard\n");
#elif CH
	printf("Kernel type:           Channelized\n");
#elif BLK2D
	printf("Kernel type:           2D overlapped blocking\n");
#elif SCH
	printf("Kernel type:           Nallatech 510T serial channel\n");
#endif

#ifdef NDR
	printf("Kernel model:          NDRange\n");
#else
	printf("Kernel model:          Single Work-item\n");
#endif

#ifdef BLK2D
	printf("Row size:              %d indexes\n", rows);
	printf("Column size:           %d indexes\n", cols);
#endif

	printf("Array size:            %ld indexes\n", array_size);
	printf("Buffer size:           %d MiB\n", size_MiB);
	printf("Total memory usage:    %d MiB\n", 3 * size_MiB);
	
#ifdef NDR
	#if defined(STD) || defined(BLK2D)
	printf("Work-group\Block size: %d\n", BSIZE);
	#else
	printf("Work-group size:       %d\n", WGS);
	#endif
#else
	#if defined(STD) || defined(BLK2D)
	printf("Block size:            %d\n", BSIZE);
	#endif
#endif

	printf("Vector size:           %d\n", VEC);

#ifdef STD
	printf("Padding:               %d\n", pad);
	printf("Overlap:               %d\n\n", overlap);
#elif BLK2D
	printf("Padding:               %d\n", pad);
	printf("Halo width:            %d\n\n", halo);
#else
	printf("Padding:               %d\n\n", pad);
#endif

	// create host buffers
	if (verbose) printf("Creating host buffers...\n");
	float* hostA = alignedMalloc(padded_size_Byte);
	float* hostB = alignedMalloc(padded_size_Byte);
	float* hostC = alignedMalloc(padded_size_Byte);

	// populate host buffers
	if (verbose) printf("Filling host buffers with random data...\n");
	#pragma omp parallel default(none) firstprivate(array_size, pad) shared(hostA, hostB)
	{
		uint seed = omp_get_thread_num();
		#pragma omp for
		for (long i = 0; i < array_size; i++)
		{
			// generate random float numbers between 0 and 1000
			hostA[pad + i] = 1000.0 * (float)rand_r(&seed) / (float)(RAND_MAX);
			hostB[pad + i] = 1000.0 * (float)rand_r(&seed) / (float)(RAND_MAX);
		}
	}

	// create device buffers
	if (verbose) printf("Creating device buffers...\n");
#ifdef NO_INTERLEAVE
	cl_mem deviceA = clCreateBuffer(context, CL_MEM_READ_ONLY  | MEM_BANK_1, padded_size_Byte, NULL, &error);
	if(error != CL_SUCCESS) { printf("ERROR: clCreateBuffer deviceA (size: %d MiB) failed with error: ", padded_size_MiB); display_error_message(error, stdout); return -1;}
	cl_mem deviceB = clCreateBuffer(context, CL_MEM_READ_ONLY  | MEM_BANK_2, padded_size_Byte, NULL, &error);
	if(error != CL_SUCCESS) { printf("ERROR: clCreateBuffer deviceB (size: %d MiB) failed with error: ", padded_size_MiB); display_error_message(error, stdout); return -1;}
	cl_mem deviceC = clCreateBuffer(context, CL_MEM_WRITE_ONLY | MEM_BANK_2, padded_size_Byte, NULL, &error);
	if(error != CL_SUCCESS) { printf("ERROR: clCreateBuffer deviceC (size: %d MiB) failed with error: ", padded_size_MiB); display_error_message(error, stdout); return -1;}
#else
	cl_mem deviceA = clCreateBuffer(context, CL_MEM_READ_ONLY , padded_size_Byte, NULL, &error);
	if(error != CL_SUCCESS) { printf("ERROR: clCreateBuffer deviceA (size: %d MiB) failed with error: ", padded_size_MiB); display_error_message(error, stdout); return -1;}
	cl_mem deviceB = clCreateBuffer(context, CL_MEM_READ_ONLY , padded_size_Byte, NULL, &error);
	if(error != CL_SUCCESS) { printf("ERROR: clCreateBuffer deviceB (size: %d MiB) failed with error: ", padded_size_MiB); display_error_message(error, stdout); return -1;}
	cl_mem deviceC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, padded_size_Byte, NULL, &error);
	if(error != CL_SUCCESS) { printf("ERROR: clCreateBuffer deviceC (size: %d MiB) failed with error: ", padded_size_MiB); display_error_message(error, stdout); return -1;}
#endif

	//write buffers
	if (verbose) printf("Writing data to device...\n");
#if defined(STD) || defined(BLK2D)
	CL_SAFE_CALL(clEnqueueWriteBuffer(queue, deviceA, 1, 0, padded_size_Byte, hostA, 0, 0, 0));
	CL_SAFE_CALL(clEnqueueWriteBuffer(queue, deviceB, 1, 0, padded_size_Byte, hostB, 0, 0, 0));
#elif CH
	CL_SAFE_CALL(clEnqueueWriteBuffer(queue_read, deviceA, 1, 0, padded_size_Byte, hostA, 0, 0, 0));
	CL_SAFE_CALL(clEnqueueWriteBuffer(queue_read, deviceB, 1, 0, padded_size_Byte, hostB, 0, 0, 0));
#elif SCH
	CL_SAFE_CALL(clEnqueueWriteBuffer(queue_read, deviceA, 1, 0, padded_size_Byte, hostA, 0, 0, 0));
#endif

	// constValue random float value between 0 and 1 for MAC operation in kernel
#ifndef SCH
	float constValue = (float)rand() / (float)(RAND_MAX);
#endif

#ifdef STD
	int valid_blk  = BSIZE - overlap;
	int exit_index = (array_size % valid_blk == 0) ? array_size : array_size + valid_blk - (array_size % valid_blk);
	int num_blk = exit_index / valid_blk;

	#ifdef NDR
		int total_index = BSIZE * num_blk;

		// set local and global work size
		size_t localSize[3] = {(size_t)BSIZE, 1, 1};
		size_t globalSize[3] = {(size_t)total_index, 1, 1};

		CL_SAFE_CALL( clSetKernelArg(copyKernel, 0, sizeof(void*   ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 1, sizeof(void*   ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 2, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 3, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 4, sizeof(cl_int  ), (void*) &overlap   ) );

		CL_SAFE_CALL( clSetKernelArg(macKernel , 0, sizeof(void*   ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 1, sizeof(void*   ), (void*) &deviceB   ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 2, sizeof(void*   ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 3, sizeof(cl_float), (void*) &constValue) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 4, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 5, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 6, sizeof(cl_int  ), (void*) &overlap   ) );
	#else
		int loop_exit = BSIZE * num_blk / VEC;

		CL_SAFE_CALL( clSetKernelArg(copyKernel, 0, sizeof(void*   ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 1, sizeof(void*   ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 2, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 3, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 4, sizeof(cl_int  ), (void*) &loop_exit ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 5, sizeof(cl_int  ), (void*) &overlap   ) );

		CL_SAFE_CALL( clSetKernelArg(macKernel , 0, sizeof(void*   ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 1, sizeof(void*   ), (void*) &deviceB   ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 2, sizeof(void*   ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 3, sizeof(cl_float), (void*) &constValue) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 4, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 5, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 6, sizeof(cl_int  ), (void*) &loop_exit ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 7, sizeof(cl_int  ), (void*) &overlap   ) );
	#endif
#elif BLK2D
	int valid_blk  = BSIZE - 2 * halo;
	int exit_col = (cols % valid_blk == 0) ? cols : cols + valid_blk - (cols % valid_blk);
	int num_blk = exit_col / valid_blk;

	#ifdef NDR
		int total_cols = BSIZE * num_blk;

		// set local and global work size
		size_t localSize[3] = {(size_t)BSIZE, 1, 1};
		size_t globalSize[3] = {(size_t)total_cols, (size_t)rows, 1};

		CL_SAFE_CALL( clSetKernelArg(copyKernel, 0, sizeof(void*   ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 1, sizeof(void*   ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 2, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 3, sizeof(cl_int  ), (void*) &rows      ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 4, sizeof(cl_int  ), (void*) &cols      ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 5, sizeof(cl_int  ), (void*) &halo      ) );

		CL_SAFE_CALL( clSetKernelArg(macKernel , 0, sizeof(void*   ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 1, sizeof(void*   ), (void*) &deviceB   ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 2, sizeof(void*   ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 3, sizeof(cl_float), (void*) &constValue) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 4, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 5, sizeof(cl_int  ), (void*) &rows      ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 6, sizeof(cl_int  ), (void*) &cols      ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 7, sizeof(cl_int  ), (void*) &halo      ) );
	#else
		int loop_exit = BSIZE * num_blk * rows / VEC;

		CL_SAFE_CALL( clSetKernelArg(copyKernel, 0, sizeof(void*   ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 1, sizeof(void*   ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 2, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 3, sizeof(cl_int  ), (void*) &rows      ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 4, sizeof(cl_int  ), (void*) &cols      ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 5, sizeof(cl_int  ), (void*) &loop_exit ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 6, sizeof(cl_int  ), (void*) &halo      ) );

		CL_SAFE_CALL( clSetKernelArg(macKernel , 0, sizeof(void*   ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 1, sizeof(void*   ), (void*) &deviceB   ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 2, sizeof(void*   ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 3, sizeof(cl_float), (void*) &constValue) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 4, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 5, sizeof(cl_int  ), (void*) &rows      ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 6, sizeof(cl_int  ), (void*) &cols      ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 7, sizeof(cl_int  ), (void*) &loop_exit ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 8, sizeof(cl_int  ), (void*) &halo      ) );
	#endif
#elif CH
	#ifdef NDR
		// set local and global work size
		size_t localSize[3] = {(size_t)WGS, 1, 1};
		size_t globalSize[3] = {(size_t)(array_size / VEC), 1, 1};

		CL_SAFE_CALL( clSetKernelArg(copyReadKernel , 0, sizeof(void*   ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(copyReadKernel , 1, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(copyWriteKernel, 0, sizeof(void*   ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(copyWriteKernel, 1, sizeof(cl_int  ), (void*) &pad       ) );

		CL_SAFE_CALL( clSetKernelArg(macReadKernel  , 0, sizeof(void*   ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(macReadKernel  , 1, sizeof(void*   ), (void*) &deviceB   ) );
		CL_SAFE_CALL( clSetKernelArg(macReadKernel  , 2, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(macWriteKernel , 0, sizeof(void*   ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(macWriteKernel , 1, sizeof(cl_float), (void*) &constValue) );
		CL_SAFE_CALL( clSetKernelArg(macWriteKernel , 2, sizeof(cl_int  ), (void*) &pad       ) );
	#else
		CL_SAFE_CALL( clSetKernelArg(copyReadKernel , 0, sizeof(void*   ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(copyReadKernel , 1, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(copyReadKernel , 2, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(copyWriteKernel, 0, sizeof(void*   ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(copyWriteKernel, 1, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(copyWriteKernel, 2, sizeof(cl_long ), (void*) &array_size) );

		CL_SAFE_CALL( clSetKernelArg(macReadKernel  , 0, sizeof(void*   ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(macReadKernel  , 1, sizeof(void*   ), (void*) &deviceB   ) );
		CL_SAFE_CALL( clSetKernelArg(macReadKernel  , 2, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(macReadKernel  , 3, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(macWriteKernel , 0, sizeof(void*   ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(macWriteKernel , 1, sizeof(cl_float), (void*) &constValue) );
		CL_SAFE_CALL( clSetKernelArg(macWriteKernel , 2, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(macWriteKernel , 3, sizeof(cl_long ), (void*) &array_size) );
	#endif
#elif SCH
	#ifdef NDR
		CL_SAFE_CALL( clSetKernelArg(copyReadKernel , 0, sizeof(void*   ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(copyReadKernel , 1, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(copyWriteKernel, 0, sizeof(void*   ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(copyWriteKernel, 1, sizeof(cl_int  ), (void*) &pad       ) );
	#else
		CL_SAFE_CALL( clSetKernelArg(copyReadKernel , 0, sizeof(void*   ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(copyReadKernel , 1, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(copyReadKernel , 2, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(copyWriteKernel, 0, sizeof(void*   ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(copyWriteKernel, 1, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(copyWriteKernel, 2, sizeof(cl_long ), (void*) &array_size) );
	#endif
#endif

	// device warm-up
	if (verbose) printf("Device warm-up...\n");
#if defined(STD) || defined(BLK2D)
	#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue, copyKernel, DIM, NULL, globalSize, localSize, 0, 0, NULL) );
	#else
		CL_SAFE_CALL( clEnqueueTask(queue, copyKernel, 0, NULL, NULL) );
	#endif
		clFinish(queue);
#elif defined(CH) || defined(SCH)
	#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue_read , copyReadKernel , 1, NULL, globalSize, localSize, 0, 0, NULL) );
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue_write, copyWriteKernel, 1, NULL, globalSize, localSize, 0, 0, NULL) );
	#else
		CL_SAFE_CALL( clEnqueueTask(queue_read , copyReadKernel , 0, NULL, NULL) );
		CL_SAFE_CALL( clEnqueueTask(queue_write, copyWriteKernel, 0, NULL, NULL) );
	#endif
		clFinish(queue_write);
#endif

	// copy kernel
	if (verify || verbose) printf("Executing \"Copy\" kernel...\n");
	for (int i = 0; i < iter; i++)
	{
		GetTime(start);

#if defined(STD) || defined(BLK2D)
	#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue, copyKernel, DIM, NULL, globalSize, localSize, 0, 0, NULL) );
	#else
		CL_SAFE_CALL( clEnqueueTask(queue, copyKernel, 0, NULL, NULL) );
	#endif
		clFinish(queue);
#elif defined(CH) || defined(SCH)
	#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue_read , copyReadKernel , 1, NULL, globalSize, localSize, 0, 0, NULL) );
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue_write, copyWriteKernel, 1, NULL, globalSize, localSize, 0, 0, NULL) );
	#else
		CL_SAFE_CALL( clEnqueueTask(queue_read , copyReadKernel , 0, NULL, NULL) );
		CL_SAFE_CALL( clEnqueueTask(queue_write, copyWriteKernel, 0, NULL, NULL) );
	#endif
		clFinish(queue_write);
#endif

		GetTime(end);
		totalCopyTime += TimeDiff(start, end);
	}

	// verify copy kernel
	if (verify)
	{
		// read data back to host
		printf("Reading data back from device...\n");
	#if defined(STD) || defined(BLK2D)
		CL_SAFE_CALL(clEnqueueReadBuffer(queue, deviceC, 1, 0, padded_size_Byte, hostC, 0, 0, 0));
		clFinish(queue);
	#elif defined (CH) || defined(SCH)
		CL_SAFE_CALL(clEnqueueReadBuffer(queue_write, deviceC, 1, 0, padded_size_Byte, hostC, 0, 0, 0));
		clFinish(queue_write);
	#endif

		printf("Verifying \"Copy\" kernel: ");
		int success = 1;
		#pragma omp parallel for ordered default(none) firstprivate(array_size, pad, hostA, hostC, verbose) shared(success)
		for (long i = 0; i < array_size; i++)
		{
			if (hostA[pad + i] != hostC[pad + i])
			{
				if (verbose) printf("Mismatch at index %ld: Expected = %0.6f, Obtained = %0.6f\n", i, hostA[pad + i], hostC[pad + i]);
				success = 0;
			}
		}

		if (success)
		{
			printf("SUCCESS!\n");
		}
		else
		{
			printf("FAILURE!\n");
		}
	}

#ifndef SCH
	// MAC kernel
	if (verify || verbose) printf("Executing \"MAC\" kernel...\n");
	for (int i = 0; i < iter; i++)
	{
		GetTime(start);

#if defined(STD) || defined(BLK2D)
	#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue, macKernel, DIM, NULL, globalSize, localSize, 0, 0, NULL) );
	#else
		CL_SAFE_CALL( clEnqueueTask(queue, macKernel, 0, NULL, NULL) );
	#endif
		clFinish(queue);
#elif CH
	#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue_read , macReadKernel , 1, NULL, globalSize, localSize, 0, 0, NULL) );
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue_write, macWriteKernel, 1, NULL, globalSize, localSize, 0, 0, NULL) );
	#else
		CL_SAFE_CALL( clEnqueueTask(queue_read , macReadKernel , 0, NULL, NULL) );
		CL_SAFE_CALL( clEnqueueTask(queue_write, macWriteKernel, 0, NULL, NULL) );
	#endif
		clFinish(queue_write);
#endif

		GetTime(end);
		totalMacTime += TimeDiff(start, end);
	}

	// verify mac kernel
	if (verify)
	{
		// read data back to host
		printf("Reading data back from device...\n");
	#if defined(STD) || defined(BLK2D)
		CL_SAFE_CALL(clEnqueueReadBuffer(queue, deviceC, 1, 0, padded_size_Byte, hostC, 0, 0, 0));
		clFinish(queue);
	#elif CH
		CL_SAFE_CALL(clEnqueueReadBuffer(queue_write, deviceC, 1, 0, padded_size_Byte, hostC, 0, 0, 0));
		clFinish(queue_write);
	#endif

		printf("Verifying \"MAC\" kernel: ");
		int success = 1;
		#pragma omp parallel for ordered default(none) firstprivate(array_size, pad, constValue, verbose, hostA, hostB, hostC) shared(success)
		for (long i = 0; i < array_size; i++)
		{
			float out = constValue * hostA[pad + i] + hostB[pad + i];
			if (fabs(hostC[pad + i] - out) > 0.001)
			{
				if (verbose) printf("Mismatch at index %ld: Expected = %0.6f, Obtained = %0.6f\n", i, out, hostC[pad + i]);
				success = 0;
			}
		}

		if (success)
		{
			printf("SUCCESS!\n");
		}
		else
		{
			printf("FAILURE!\n");
		}
	}
#endif

	if (verify || verbose) printf("\n");
#ifdef STD
	avgCopyTime = totalCopyTime / (double)iter;
	avgMacTime = totalMacTime / (double)iter;
	long totalSize_B = ((num_blk * BSIZE) - (exit_index + overlap - array_size)) * sizeof(float);
	printf("Copy: %.3f GiB/s (%.3f GB/s)\n", (double)(2 * totalSize_B * 1000.0) / (1.0E9 * avgCopyTime), (double)(2 * totalSize_B) / (1.0E6 * avgCopyTime));
	printf("MAC : %.3f GiB/s (%.3f GB/s)\n", (double)(3 * totalSize_B * 1000.0) / (1.0E9 * avgMacTime ), (double)(3 * totalSize_B) / (1.0E6 * avgMacTime ));
#elif BLK2D
	avgCopyTime = totalCopyTime / (double)iter;
	avgMacTime = totalMacTime / (double)iter;
	long totalSize_B = ((num_blk * BSIZE) - (exit_col + 2 * halo - cols)) * rows * sizeof(float);
	printf("Copy: %.3f GiB/s (%.3f GB/s)\n", (double)(2 * totalSize_B * 1000.0) / (1.0E9 * avgCopyTime), (double)(2 * totalSize_B) / (1.0E6 * avgCopyTime));
	printf("MAC : %.3f GiB/s (%.3f GB/s)\n", (double)(3 * totalSize_B * 1000.0) / (1.0E9 * avgMacTime ), (double)(3 * totalSize_B) / (1.0E6 * avgMacTime ));
#elif SCH
	avgCopyTime = totalCopyTime / (double)iter;
	printf("Channel bandwidth: %.3f GiB/s (%.3f GB/s)\n", (double)(size_MiB * 1000.0) / (1024.0 * avgCopyTime), (double)(size_B) / (1.0E6 * avgCopyTime));
	printf("Memory bandwidth : %.3f GiB/s (%.3f GB/s)\n", (double)(2 * size_MiB * 1000.0) / (1024.0 * avgCopyTime), (double)(2 * size_B) / (1.0E6 * avgCopyTime));
#else
	avgCopyTime = totalCopyTime / (double)iter;
	avgMacTime = totalMacTime / (double)iter;
	printf("Copy: %.3f GiB/s (%.3f GB/s)\n", (double)(2 * size_MiB * 1000.0) / (1024.0 * avgCopyTime), (double)(2 * size_B) / (1.0E6 * avgCopyTime));
	printf("MAC : %.3f GiB/s (%.3f GB/s)\n", (double)(3 * size_MiB * 1000.0) / (1024.0 * avgMacTime ), (double)(3 * size_B) / (1.0E6 * avgMacTime ));
#endif

#if defined(STD) || defined(BLK2D)
	clReleaseCommandQueue(queue);
#elif defined(CH) || defined(SCH)
	clReleaseCommandQueue(queue_read);
	clReleaseCommandQueue(queue_write);
#endif
	clReleaseContext(context);
	clReleaseMemObject(deviceA);
	clReleaseMemObject(deviceB);
	clReleaseMemObject(deviceC);

	free(hostA);
	free(hostB);
	free(hostC);
#ifdef SCH
	free(kernelSourceFPGA1);
	free(kernelSourceFPGA2);
#else
	free(kernelSource);
#endif
	free(deviceList);
}