//====================================================================================================================================
// Memory bandwidth benchmark host for OpenCL-capable FPGAs: Standard
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

#define DIM 1

// global variables
static cl_context       context;
#if defined(STD)
static cl_command_queue queue;
#elif defined(CHSTD)
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
#if defined(STD)
	queue = clCreateCommandQueue(context, deviceList[0], 0, NULL);
	if(!queue)
	{
		printf("ERROR: clCreateCommandQueue(queue) failed with error code: ");
		display_error_message(error, stdout);
		exit(-1);
	}
#elif defined(CHSTD)
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

	free(platforms); // platforms isn't needed in the main function
}

static inline void usage(char **argv)
{
	printf("\nUsage: %s -s <buffer size in MiB> -n <number of iterations> -pad <array padding indexes> -hw <halo width> --verbose --verify\n", argv[0]);
}

int main(int argc, char **argv)
{
	// input arguments
	int size_MiB = 100; 							// buffer size, default size is 100 MiB
	int iter = 1;									// number of iterations
	int pad = 0;									// padding
	int verbose = 0, verify = 0;
	int halo = 0;

	// timing measurement
	TimeStamp start, end;
	double totalr1w1Time = 0, avgr1w1Time = 0;
	double totalr2w1Time = 0, avgr2w1Time = 0;

	// for OpenCL errors
	cl_int error = 0;

	int arg = 1;
	while (arg < argc)
	{
		if(strcmp(argv[arg], "-s") == 0)
		{
			size_MiB = atoi(argv[arg + 1]);
			arg += 2;
		}
		else if (strcmp(argv[arg], "-n") == 0)
		{
			iter = atoi(argv[arg + 1]);
			arg += 2;
		}
		else if (strcmp(argv[arg], "-pad") == 0)
		{
			pad = atoi(argv[arg + 1]);
			arg += 2;
		}
		else if (strcmp(argv[arg], "-hw") == 0)
		{
			halo = atoi(argv[arg + 1]);
			arg += 2;
		}
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
	long size_B = (long)size_MiB * 1024 * 1024;
	long array_size = size_B / sizeof(float);
	long padded_array_size = array_size + pad;
	long padded_size_Byte = padded_array_size * sizeof(float);
	int  padded_size_MiB = padded_size_Byte / (1024 * 1024);

	// OpenCL initialization
	init();

	// load kernel file and build program
#ifdef INTEL_FPGA
		size_t kernelFileSize;
		char *kernelSource = read_kernel("fpga-stream-kernel.aocx", &kernelFileSize);
		cl_program prog = clCreateProgramWithBinary(context, deviceCount, deviceList, &kernelFileSize, (const unsigned char**)&kernelSource, NULL, &error);
		if(error != CL_SUCCESS)
		{
			printf("ERROR: clCreateProgramWithBinary() failed with error: ");
			display_error_message(error, stdout);
			return -1;
		}
#else // for CPU/GPUs
	#if defined(STD)
		size_t kernelFileSize;
		char *kernelSource = read_kernel("fpga-stream-kernel-std.cl", &kernelFileSize);

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
	sprintf(clOptions + strlen(clOptions), "-DVEC=%d -DBLOCK_X=%d ", VEC, BLOCK_X);
#endif

#ifdef NDR
	sprintf(clOptions + strlen(clOptions), "-DNDR");
#endif

	// compile kernel file
	clBuildProgram_SAFE(prog, deviceCount, deviceList, clOptions, NULL, NULL);

	// create kernel objects
#if defined(STD)
	cl_kernel r1w1Kernel, r2w1Kernel;

	r1w1Kernel = clCreateKernel(prog, "r1w1", &error);
	if(error != CL_SUCCESS)
	{
		printf("ERROR: clCreateKernel(r1w1) failed with error: ");
		display_error_message(error, stdout);
		return -1;
	}

	r2w1Kernel = clCreateKernel(prog, "r2w1", &error);
	if(error != CL_SUCCESS)
	{
		printf("ERROR: clCreateKernel(r2w1) failed with error: ");
		display_error_message(error, stdout);
		return -1;
	}

	clReleaseProgram(prog);
#elif defined(CHSTD)
	cl_kernel r1w1ReadKernel, r1w1WriteKernel, r2w1ReadKernel, r2w1WriteKernel;

	r1w1ReadKernel = clCreateKernel(prog, "r1w1_read", &error);
	if(error != CL_SUCCESS)
	{
		printf("ERROR: clCreateKernel(r1w1_read) failed with error: ");
		display_error_message(error, stdout);
		return -1;
	}

	r1w1WriteKernel= clCreateKernel(prog, "r1w1_write", &error);
	if(error != CL_SUCCESS)
	{
		printf("ERROR: clCreateKernel(r1w1_write) failed with error: ");
		display_error_message(error, stdout);
		return -1;
	}

	r2w1ReadKernel = clCreateKernel(prog, "r2w1_read", &error);
	if(error != CL_SUCCESS)
	{
		printf("ERROR: clCreateKernel(r2w1_write) failed with error: ");
		display_error_message(error, stdout);
		return -1;
	}

	r2w1WriteKernel = clCreateKernel(prog, "r2w1_write", &error);
	if(error != CL_SUCCESS)
	{
		printf("ERROR: clCreateKernel(r2w1_write) failed with error: ");
		display_error_message(error, stdout);
		return -1;
	}

	clReleaseProgram(prog);
#endif

#ifdef STD
	printf("Kernel type:           Standard\n");
#elif CHSTD
	printf("Kernel type:           Channelized standard\n");
#endif

#ifdef NDR
	printf("Kernel model:          NDRange\n");
#else
	printf("Kernel model:          Single Work-item\n");
#endif

	printf("Array size:            %ld indexes\n", array_size);
	printf("Buffer size:           %d MiB\n", size_MiB);
	printf("Total memory usage:    %d MiB\n", 3 * size_MiB);
	
#ifdef NDR
	printf("Work-group\\Block size: %d\n", BLOCK_X);
#else
	printf("Block size:            %d\n", BLOCK_X);
#endif

	printf("Vector size:           %d\n", VEC);
	printf("Array padding:         %d\n", pad);
	printf("Halo width             %d\n\n", halo);

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
#if defined(STD)
	CL_SAFE_CALL(clEnqueueWriteBuffer(queue, deviceA, 1, 0, padded_size_Byte, hostA, 0, 0, 0));
	CL_SAFE_CALL(clEnqueueWriteBuffer(queue, deviceB, 1, 0, padded_size_Byte, hostB, 0, 0, 0));
#elif defined(CHSTD)
	CL_SAFE_CALL(clEnqueueWriteBuffer(queue_read, deviceA, 1, 0, padded_size_Byte, hostA, 0, 0, 0));
	CL_SAFE_CALL(clEnqueueWriteBuffer(queue_read, deviceB, 1, 0, padded_size_Byte, hostB, 0, 0, 0));
#endif

#ifdef STD
	int valid_blk_x = BLOCK_X - 2 * halo;
	long last_x = (array_size % valid_blk_x == 0) ? array_size : array_size + valid_blk_x - (array_size % valid_blk_x);
	int num_blk_x = last_x / valid_blk_x;

	#ifdef NDR
		long total_index = (long)(BLOCK_X / VEC) * (long)num_blk_x;

		// set local and global work size
		size_t localSize[3] = {(size_t)(BLOCK_X / VEC), 1, 1};
		size_t globalSize[3] = {(size_t)total_index, 1, 1};

		CL_SAFE_CALL( clSetKernelArg(r1w1Kernel , 0, sizeof(cl_mem  ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1Kernel , 1, sizeof(cl_mem  ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1Kernel , 2, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1Kernel , 3, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(r1w1Kernel , 4, sizeof(cl_int  ), (void*) &halo      ) );

		CL_SAFE_CALL( clSetKernelArg(r2w1Kernel , 0, sizeof(cl_mem  ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1Kernel , 1, sizeof(cl_mem  ), (void*) &deviceB   ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1Kernel , 2, sizeof(cl_mem  ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1Kernel , 3, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1Kernel , 4, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(r2w1Kernel , 5, sizeof(cl_int  ), (void*) &halo      ) );
	#else
		long loop_exit = (long)(BLOCK_X / VEC) * (long)num_blk_x;

		CL_SAFE_CALL( clSetKernelArg(r1w1Kernel, 0, sizeof(cl_mem  ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1Kernel, 1, sizeof(cl_mem  ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1Kernel, 2, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1Kernel, 3, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(r1w1Kernel, 4, sizeof(cl_long ), (void*) &loop_exit ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1Kernel, 5, sizeof(cl_int  ), (void*) &halo      ) );

		CL_SAFE_CALL( clSetKernelArg(r2w1Kernel , 0, sizeof(cl_mem  ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1Kernel , 1, sizeof(cl_mem  ), (void*) &deviceB   ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1Kernel , 2, sizeof(cl_mem  ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1Kernel , 3, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1Kernel , 4, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(r2w1Kernel , 5, sizeof(cl_long ), (void*) &loop_exit ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1Kernel , 6, sizeof(cl_int  ), (void*) &halo      ) );
	#endif
#elif CHSTD
	int valid_blk_x = BLOCK_X - 2 * halo;
	long last_x = (array_size % valid_blk_x == 0) ? array_size : array_size + valid_blk_x - (array_size % valid_blk_x);
	int num_blk_x = last_x / valid_blk_x;

	#ifdef NDR
		long total_index = (long)(BLOCK_X / VEC) * (long)num_blk_x;

		// set local and global work size
		size_t localSize[3] = {(size_t)(BLOCK_X / VEC), 1, 1};
		size_t globalSize[3] = {(size_t)total_index, 1, 1};

		CL_SAFE_CALL( clSetKernelArg(r1w1ReadKernel , 0, sizeof(cl_mem  ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1ReadKernel , 1, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1ReadKernel , 2, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(r1w1ReadKernel , 3, sizeof(cl_int  ), (void*) &halo      ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1WriteKernel, 0, sizeof(cl_mem  ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1WriteKernel, 1, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1WriteKernel, 2, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(r1w1WriteKernel, 3, sizeof(cl_int  ), (void*) &halo      ) );

		CL_SAFE_CALL( clSetKernelArg(r2w1ReadKernel , 0, sizeof(cl_mem  ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1ReadKernel , 1, sizeof(cl_mem  ), (void*) &deviceB   ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1ReadKernel , 2, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1ReadKernel , 3, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(r2w1ReadKernel , 4, sizeof(cl_int  ), (void*) &halo      ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1WriteKernel, 0, sizeof(cl_mem  ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1WriteKernel, 1, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1WriteKernel, 2, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(r2w1WriteKernel, 3, sizeof(cl_int  ), (void*) &halo      ) );
	#else
		long loop_exit = (long)(BLOCK_X / VEC) * (long)num_blk_x;

		CL_SAFE_CALL( clSetKernelArg(r1w1ReadKernel , 0, sizeof(cl_mem  ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1ReadKernel , 1, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1ReadKernel , 2, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(r1w1ReadKernel , 3, sizeof(cl_long ), (void*) &loop_exit ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1ReadKernel , 4, sizeof(cl_int  ), (void*) &halo      ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1WriteKernel, 0, sizeof(cl_mem  ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1WriteKernel, 1, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1WriteKernel, 2, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(r1w1WriteKernel, 3, sizeof(cl_long ), (void*) &loop_exit ) );
		CL_SAFE_CALL( clSetKernelArg(r1w1WriteKernel, 4, sizeof(cl_int  ), (void*) &halo      ) );

		CL_SAFE_CALL( clSetKernelArg(r2w1ReadKernel , 0, sizeof(cl_mem  ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1ReadKernel , 1, sizeof(cl_mem  ), (void*) &deviceB   ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1ReadKernel , 2, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1ReadKernel , 3, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(r2w1ReadKernel , 4, sizeof(cl_long ), (void*) &loop_exit ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1ReadKernel , 5, sizeof(cl_int  ), (void*) &halo      ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1WriteKernel, 0, sizeof(cl_mem  ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1WriteKernel, 1, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1WriteKernel, 2, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(r2w1WriteKernel, 3, sizeof(cl_long ), (void*) &loop_exit ) );
		CL_SAFE_CALL( clSetKernelArg(r2w1WriteKernel, 4, sizeof(cl_int  ), (void*) &halo      ) );
	#endif

	// device warm-up
	if (verbose) printf("Device warm-up...\n");
#if defined(STD)
	#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue, r1w1Kernel, DIM, NULL, globalSize, localSize, 0, 0, NULL) );
	#else
		CL_SAFE_CALL( clEnqueueTask(queue, r1w1Kernel, 0, NULL, NULL) );
	#endif
		clFinish(queue);
#elif defined(CHSTD)
	#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue_read , r1w1ReadKernel , DIM, NULL, globalSize, localSize, 0, 0, NULL) );
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue_write, r1w1WriteKernel, DIM, NULL, globalSize, localSize, 0, 0, NULL) );
	#else
		CL_SAFE_CALL( clEnqueueTask(queue_read , r1w1ReadKernel , 0, NULL, NULL) );
		CL_SAFE_CALL( clEnqueueTask(queue_write, r1w1WriteKernel, 0, NULL, NULL) );
	#endif
		clFinish(queue_write);
#endif

	// r1w1 kernel
	if (verify || verbose) printf("Executing \"r1w1\" kernel...\n");
	for (int i = 0; i < iter; i++)
	{
		GetTime(start);

#if defined(STD)
	#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue, r1w1Kernel, DIM, NULL, globalSize, localSize, 0, 0, NULL) );
	#else
		CL_SAFE_CALL( clEnqueueTask(queue, r1w1Kernel, 0, NULL, NULL) );
	#endif
		clFinish(queue);
#elif defined(CHSTD)
	#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue_read , r1w1ReadKernel , DIM, NULL, globalSize, localSize, 0, 0, NULL) );
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue_write, r1w1WriteKernel, DIM, NULL, globalSize, localSize, 0, 0, NULL) );
	#else
		CL_SAFE_CALL( clEnqueueTask(queue_read , r1w1ReadKernel , 0, NULL, NULL) );
		CL_SAFE_CALL( clEnqueueTask(queue_write, r1w1WriteKernel, 0, NULL, NULL) );
	#endif
		clFinish(queue_write);
#endif

		GetTime(end);
		totalr1w1Time += TimeDiff(start, end);
	}

	// verify r1w1 kernel
	if (verify)
	{
		// read data back to host
		printf("Reading data back from device...\n");
	#if defined(STD)
		CL_SAFE_CALL(clEnqueueReadBuffer(queue, deviceC, 1, 0, padded_size_Byte, hostC, 0, 0, 0));
		clFinish(queue);
	#elif defined (CHSTD)
		CL_SAFE_CALL(clEnqueueReadBuffer(queue_write, deviceC, 1, 0, padded_size_Byte, hostC, 0, 0, 0));
		clFinish(queue_write);
	#endif

		printf("Verifying \"r1w1\" kernel: ");
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

	// r2w1 kernel
	if (verify || verbose) printf("Executing \"r2w1\" kernel...\n");
	for (int i = 0; i < iter; i++)
	{
		GetTime(start);

#if defined(STD)
	#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue, r2w1Kernel, DIM, NULL, globalSize, localSize, 0, 0, NULL) );
	#else
		CL_SAFE_CALL( clEnqueueTask(queue, r2w1Kernel, 0, NULL, NULL) );
	#endif
		clFinish(queue);
#elif defined(CHSTD)
	#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue_read , r2w1ReadKernel , DIM, NULL, globalSize, localSize, 0, 0, NULL) );
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue_write, r2w1WriteKernel, DIM, NULL, globalSize, localSize, 0, 0, NULL) );
	#else
		CL_SAFE_CALL( clEnqueueTask(queue_read , r2w1ReadKernel , 0, NULL, NULL) );
		CL_SAFE_CALL( clEnqueueTask(queue_write, r2w1WriteKernel, 0, NULL, NULL) );
	#endif
		clFinish(queue_write);
#endif

		GetTime(end);
		totalr2w1Time += TimeDiff(start, end);
	}

	// verify r2w1 kernel
	if (verify)
	{
		// read data back to host
		printf("Reading data back from device...\n");
	#if defined(STD)
		CL_SAFE_CALL(clEnqueueReadBuffer(queue, deviceC, 1, 0, padded_size_Byte, hostC, 0, 0, 0));
		clFinish(queue);
	#elif defined(CHSTD)
		CL_SAFE_CALL(clEnqueueReadBuffer(queue_write, deviceC, 1, 0, padded_size_Byte, hostC, 0, 0, 0));
		clFinish(queue_write);
	#endif

		printf("Verifying \"r2w1\" kernel: ");
		int success = 1;
		#pragma omp parallel for ordered default(none) firstprivate(array_size, pad, hostA, hostB, hostC, verbose) shared(success)
		for (long i = 0; i < array_size; i++)
		{
			float out = hostA[pad + i] + hostB[pad + i];
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

	if (verify || verbose) printf("\n");
	avgr1w1Time = totalr1w1Time / (double)iter;
	avgr2w1Time = totalr2w1Time / (double)iter;
	int extra_halo_x = ((array_size % valid_blk_x >= halo) || (array_size % valid_blk_x == 0)) ? 0 : halo - (array_size % valid_blk_x); // in case the halo width in the last block is not fully traversed
	long totalSize_B = ((num_blk_x * BLOCK_X) - (last_x + 2 * halo - array_size) - extra_halo_x) * sizeof(float);
	printf("r1w1: %.3f GB/s (%.3f GiB/s) @%.1f ms\n", (double)(2 * totalSize_B) / (1.0E6 * avgr1w1Time), (double)(2 * totalSize_B * 1000.0) / (pow(1024.0, 3) * avgr1w1Time), avgr1w1Time);
	printf("r2w1: %.3f GB/s (%.3f GiB/s) @%.1f ms\n", (double)(3 * totalSize_B) / (1.0E6 * avgr2w1Time ), (double)(3 * totalSize_B * 1000.0) / (pow(1024.0, 3) * avgr2w1Time ), avgr2w1Time);

#if defined(STD)
	clReleaseCommandQueue(queue);
#elif defined(CHSTD)
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
	free(kernelSource);
	free(deviceList);
}