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

#ifndef VEC
	#define VEC 1
#endif

#ifndef WGS
	#define WGS 128
#endif

#ifdef LEGACY
	#define MEM_BANK_1 CL_MEM_BANK_1_ALTERA
	#define MEM_BANK_2 CL_MEM_BANK_2_ALTERA
#else
	#define MEM_BANK_1 CL_MEM_BANK_1_INTEL
	#define MEM_BANK_2 CL_MEM_BANK_2_INTEL
#endif

// global variables
static cl_context       context;
static cl_command_queue queue;
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
	queue = clCreateCommandQueue(context, deviceList[0], 0, NULL);
	if(!queue)
	{
		printf("ERROR: clCreateCommandQueue() failed with error code: ");
		display_error_message(error, stdout);
		exit(-1);
	}
	
	free(platforms); // platforms isn't needed in the main function
}

static inline void shutdown()
{
	// release resources
	if(queue) clReleaseCommandQueue(queue);
	if(context) clReleaseContext(context);
	if(deviceList) free(deviceList);
}

void usage(char **argv)
{
	printf("\nUsage: %s -s <buffer size in MiB> -n <number of iterations> -p <number of padding indexes> -v (verbose) --verify\n", argv[0]);
}

int main(int argc, char **argv)
{
	// input arguments
	int size = 100; 								// buffer size, default size is 100 MiB
	int iter = 1;									// number of iterations
	int pad = 0;									// padding
	int verbose = 0, verify = 0;

	// timing measurement
	TimeStamp start, end;
	double totalCopyTime = 0, totalMacTime = 0, avgCopyTime = 0, avgMacTime = 0;

	// for OpenCL errors
	cl_int error = 0;

	int arg = 1;
	while (arg < argc)
	{
		if(strcmp(argv[arg], "-s") == 0)
		{
			size = atoi(argv[arg + 1]);
			arg += 2;
		}
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
		else if (strcmp(argv[arg], "-v") == 0)
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
			exit (-1);
		}
	}

	// set array size based in input buffer size, default is 256k floats (= 100 MiB)
	int array_size = size * 256 * 1024;
	int padded_array_size = array_size + pad;
#ifdef NDR
	// set lcoal and global work size
	size_t localSize[3] = {(size_t)WGS, 1, 1};
	size_t globalSize[3] = {(size_t)array_size, 1, 1};
#endif

	// OpenCL initialization
	init();

	// load kernel file and build program
	size_t kernelFileSize;
#ifdef INTEL_FPGA
	char *kernelSource = read_kernel("fpga-stream-kernel.aocx", &kernelFileSize);
	cl_program prog = clCreateProgramWithBinary(context, deviceCount, deviceList, &kernelFileSize, (const unsigned char**)&kernelSource, NULL, &error);
#else
	char *kernelSource = read_kernel("fpga-stream-kernel-std.cl", &kernelFileSize);
	cl_program prog = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &error);
#endif
	if(error != CL_SUCCESS)
	{
		printf("ERROR: clCreateProgramWithSource/Binary() failed with error: ");
		display_error_message(error, stdout);
		return -1;
	}

	char clOptions[200] = "";

#ifndef INTEL_FPGA
	sprintf(clOptions + strlen(clOptions), "-DVEC=%d -DWGS=%d ", VEC, WGS);
#endif

#ifdef NDR
	sprintf(clOptions + strlen(clOptions), "-DNDR");
#endif

	// compile kernel file
	clBuildProgram_SAFE(prog, deviceCount, deviceList, clOptions, NULL, NULL);

	// create kernel objects
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

	printf("Array size:         %d indexes\n", array_size);
	printf("Buffer size:        %d MiB\n", size);
	printf("Total memory usage: %d MiB\n", 3 * size);
#ifdef NDR
	printf("Work-group size:    %d\n\n", WGS);
#else
	printf("Vector size:        %d\n\n", VEC);
#endif

	// create host buffers
	if (verbose) printf("Creating host buffers...\n");
	float* hostA = alignedMalloc(padded_array_size * sizeof(float));
	float* hostB = alignedMalloc(padded_array_size * sizeof(float));
	float* hostC = alignedMalloc(padded_array_size * sizeof(float));

	// populate host buffers
	if (verbose) printf("Filling host buffers with random data...\n");
	#pragma omp parallel default(none) firstprivate(array_size, pad) shared(hostA, hostB)
	{
		uint seed = omp_get_thread_num();
		#pragma omp for
		for (int i = 0; i < array_size; i++)
		{
			// generate random float numbers between 0 and 1000
			hostA[pad + i] = 1000.0 * (float)rand_r(&seed) / (float)(RAND_MAX);
			hostB[pad + i] = 1000.0 * (float)rand_r(&seed) / (float)(RAND_MAX);
		}
	}

	// create device buffers
	if (verbose) printf("Creating device buffers...\n");
#ifdef NO_INTERLEAVE
	cl_mem deviceA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA , padded_array_size * sizeof(float), NULL, &error);
	if(error != CL_SUCCESS) { printf("ERROR: clCreateBuffer deviceA (size: %d MiB) failed with error: ", size); display_error_message(error, stdout); return -1;}
	cl_mem deviceB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA , padded_array_size * sizeof(float), NULL, &error);
	if(error != CL_SUCCESS) { printf("ERROR: clCreateBuffer deviceB (size: %d MiB) failed with error: ", size); display_error_message(error, stdout); return -1;}
	cl_mem deviceC = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_2_ALTERA, padded_array_size * sizeof(float), NULL, &error);
	if(error != CL_SUCCESS) { printf("ERROR: clCreateBuffer deviceC (size: %d MiB) failed with error: ", size); display_error_message(error, stdout); return -1;}
#else
	cl_mem deviceA = clCreateBuffer(context, CL_MEM_READ_ONLY , padded_array_size * sizeof(float), NULL, &error);
	if(error != CL_SUCCESS) { printf("ERROR: clCreateBuffer deviceA (size: %d MiB) failed with error: ", size); display_error_message(error, stdout); return -1;}
	cl_mem deviceB = clCreateBuffer(context, CL_MEM_READ_ONLY , padded_array_size * sizeof(float), NULL, &error);
	if(error != CL_SUCCESS) { printf("ERROR: clCreateBuffer deviceB (size: %d MiB) failed with error: ", size); display_error_message(error, stdout); return -1;}
	cl_mem deviceC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, padded_array_size * sizeof(float), NULL, &error);
	if(error != CL_SUCCESS) { printf("ERROR: clCreateBuffer deviceC (size: %d MiB) failed with error: ", size); display_error_message(error, stdout); return -1;}
#endif

	//write buffers
	if (verbose) printf("Writing data to device...\n");
	CL_SAFE_CALL(clEnqueueWriteBuffer(queue, deviceA, 1, 0, padded_array_size * sizeof(float), hostA, 0, 0, 0));
	CL_SAFE_CALL(clEnqueueWriteBuffer(queue, deviceB, 1, 0, padded_array_size * sizeof(float), hostB, 0, 0, 0));

	// constValue random float value between 0 and 1 for MAC operation in kernel
	float constValue = (float)rand() / (float)(RAND_MAX);

#ifdef NDR
	CL_SAFE_CALL( clSetKernelArg(copyKernel, 0, sizeof(void*   ), (void*) &deviceA   ) );
	CL_SAFE_CALL( clSetKernelArg(copyKernel, 1, sizeof(void*   ), (void*) &deviceC   ) );
	CL_SAFE_CALL( clSetKernelArg(copyKernel, 2, sizeof(cl_int  ), (void*) &pad       ) );

	CL_SAFE_CALL( clSetKernelArg(macKernel , 0, sizeof(void*   ), (void*) &deviceA   ) );
	CL_SAFE_CALL( clSetKernelArg(macKernel , 1, sizeof(void*   ), (void*) &deviceB   ) );
	CL_SAFE_CALL( clSetKernelArg(macKernel , 2, sizeof(void*   ), (void*) &deviceC   ) );
	CL_SAFE_CALL( clSetKernelArg(macKernel , 3, sizeof(cl_float), (void*) &constValue) );
	CL_SAFE_CALL( clSetKernelArg(macKernel , 4, sizeof(cl_int  ), (void*) &pad       ) );
#else
	CL_SAFE_CALL( clSetKernelArg(copyKernel, 0, sizeof(void*   ), (void*) &deviceA   ) );
	CL_SAFE_CALL( clSetKernelArg(copyKernel, 1, sizeof(void*   ), (void*) &deviceC   ) );
	CL_SAFE_CALL( clSetKernelArg(copyKernel, 2, sizeof(cl_int  ), (void*) &array_size) );
	CL_SAFE_CALL( clSetKernelArg(copyKernel, 3, sizeof(cl_int  ), (void*) &pad       ) );

	CL_SAFE_CALL( clSetKernelArg(macKernel , 0, sizeof(void*   ), (void*) &deviceA   ) );
	CL_SAFE_CALL( clSetKernelArg(macKernel , 1, sizeof(void*   ), (void*) &deviceB   ) );
	CL_SAFE_CALL( clSetKernelArg(macKernel , 2, sizeof(void*   ), (void*) &deviceC   ) );
	CL_SAFE_CALL( clSetKernelArg(macKernel , 3, sizeof(cl_float), (void*) &constValue) );
	CL_SAFE_CALL( clSetKernelArg(macKernel , 4, sizeof(cl_int  ), (void*) &array_size) );
	CL_SAFE_CALL( clSetKernelArg(macKernel , 5, sizeof(cl_int  ), (void*) &pad       ) );
#endif

	// device warm-up
	if (verbose) printf("Device warm-up...\n");
#ifdef NDR
	CL_SAFE_CALL( clEnqueueNDRangeKernel(queue, copyKernel, 1, NULL, globalSize, localSize, 0, 0, NULL) );
#else
	CL_SAFE_CALL( clEnqueueTask(queue, copyKernel, 0, NULL, NULL) );
#endif
	clFinish(queue);

	// copy kernel
	if (verbose) printf("Executing \"Copy\" kernel...\n");
	for (int i = 0; i < iter; i++)
	{
		GetTime(start);

#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue, copyKernel, 1, NULL, globalSize, localSize, 0, 0, NULL) );
#else
		CL_SAFE_CALL( clEnqueueTask(queue, copyKernel, 0, NULL, NULL) );
#endif
		clFinish(queue);

		GetTime(end);
		totalCopyTime += TimeDiff(start, end);
	}

	// verify copy kernel
	if (verify)
	{
		// read data back to host
		printf("Reading data back from device...\n");
		CL_SAFE_CALL(clEnqueueReadBuffer(queue, deviceC, 1, 0, padded_array_size * sizeof(float), hostC, 0, 0, 0));
		clFinish(queue);

		printf("Verifying \"Copy\" kernel: ");
		int success = 1;
		#pragma omp parallel for ordered default(none) firstprivate(array_size, pad, hostA, hostC) shared(success)
		for (int i = 0; i < array_size; i++)
		{
			if (hostA[pad + i] != hostC[pad + i])
			{
				printf("Mismatch at index %d: Expected = %0.6f, Obtained = %0.6f\n", i, hostA[pad + i], hostC[pad + i]);
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

	// MAC kernel
	if (verbose) printf("Executing \"MAC\" kernel...\n");
	for (int i = 0; i < iter; i++)
	{
		GetTime(start);

#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue, macKernel, 1, NULL, globalSize, localSize, 0, 0, NULL) );
#else
		CL_SAFE_CALL( clEnqueueTask(queue, macKernel, 0, NULL, NULL) );
#endif
		clFinish(queue);

		GetTime(end);
		totalMacTime += TimeDiff(start, end);
	}

	// verify mac kernel
	if (verify)
	{
		// read data back to host
		printf("Reading data back from device...\n");
		CL_SAFE_CALL(clEnqueueReadBuffer(queue, deviceC, 1, 0, padded_array_size * sizeof(float), hostC, 0, 0, 0));
		clFinish(queue);

		printf("Verifying \"MAC\" kernel: ");
		int success = 1;
		#pragma omp parallel for ordered default(none) firstprivate(array_size, pad, constValue, verbose, hostA, hostB, hostC) shared(success)
		for (int i = 0; i < array_size; i++)
		{
			float out = constValue * hostA[pad + i] + hostB[pad + i];
			if (fabs(hostC[pad + i] - out) > 0.001)
			{
				if (verbose) printf("Mismatch at index %d: Expected = %0.6f, Obtained = %0.6f\n", i, out, hostC[pad + i]);
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
	avgCopyTime = totalCopyTime / (double)iter;
	avgMacTime = totalMacTime / (double)iter;
	printf("Copy: %.2f GiB/s (%.2f GB/s)\n", (double)(2 * size * 1000.0) / (1024.0 * avgCopyTime), (double)(2 * array_size * sizeof(float)) / (1.0E6 * avgCopyTime));
	printf("MAC : %.2f GiB/s (%.2f GB/s)\n", (double)(3 * size * 1000.0) / (1024.0 * avgMacTime ), (double)(3 * array_size * sizeof(float)) / (1.0E6 * avgMacTime ));

	// OpenCL shutdown
	shutdown();

	clReleaseMemObject(deviceA);
	clReleaseMemObject(deviceB);
	clReleaseMemObject(deviceC);

	free(hostA);
	free(hostB);
	free(hostC);
	free(kernelSource);
}