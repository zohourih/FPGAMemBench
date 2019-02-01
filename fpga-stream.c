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

// global variables
static cl_context       context;
#ifdef STD
static cl_command_queue queue;
#elif CH
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
#ifdef STD
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
		printf("ERROR: clCreateCommandQueue(queue_read) failed with error code: ");
		display_error_message(error, stdout);
		exit(-1);
	}
#endif
	
	free(platforms); // platforms isn't needed in the main function
}

static inline void shutdown()
{
	// release resources
#ifdef STD
	if(queue) clReleaseCommandQueue(queue);
#elif CH
	if(queue_read) clReleaseCommandQueue(queue_read);
	if(queue_write) clReleaseCommandQueue(queue_write);
#endif
	if(context) clReleaseContext(context);
	if(deviceList) free(deviceList);
}

void usage(char **argv)
{
	printf("\nUsage: %s -s <buffer size in MiB> -n <number of iterations> -p <number of padding indexes> --verbose --verify\n", argv[0]);
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
			exit (-1);
		}
	}

	// set array size based in input buffer size, default is 256k floats (= 100 MiB)
	int array_size = size * 256 * 1024;
	int padded_array_size = array_size + pad;
#ifdef NDR
	// set local and global work size
	#ifdef STD
		size_t localSize[3] = {(size_t)WGS, 1, 1};
		size_t globalSize[3] = {(size_t)array_size, 1, 1};
	#elif CH
		size_t localSize[3] = {(size_t)WGS, 1, 1};
		size_t globalSize[3] = {(size_t)(array_size / VEC), 1, 1};
	#endif
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
#ifdef STD
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
#endif
	clReleaseProgram(prog);

#ifdef STD
	printf("Kernel type:        Standard\n");
#elif CH
	printf("Kernel type:        Channelized\n");
#endif
#ifdef NDR
	printf("Kernel model:       NDRange\n");
#else
	printf("Kernel model:       Single Work-item\n");
#endif
	printf("Array size:         %d indexes\n", array_size);
	printf("Buffer size:        %d MiB\n", size);
	printf("Total memory usage: %d MiB\n", 3 * size);
#ifdef NDR
	printf("Work-group size:    %d\n", WGS);
#endif
	printf("Vector size:        %d\n\n", VEC);


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
	cl_mem deviceA = clCreateBuffer(context, CL_MEM_READ_ONLY  | MEM_BANK_1, padded_array_size * sizeof(float), NULL, &error);
	if(error != CL_SUCCESS) { printf("ERROR: clCreateBuffer deviceA (size: %d MiB) failed with error: ", size); display_error_message(error, stdout); return -1;}
	cl_mem deviceB = clCreateBuffer(context, CL_MEM_READ_ONLY  | MEM_BANK_2, padded_array_size * sizeof(float), NULL, &error);
	if(error != CL_SUCCESS) { printf("ERROR: clCreateBuffer deviceB (size: %d MiB) failed with error: ", size); display_error_message(error, stdout); return -1;}
	cl_mem deviceC = clCreateBuffer(context, CL_MEM_WRITE_ONLY | MEM_BANK_2, padded_array_size * sizeof(float), NULL, &error);
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
#ifdef STD
	CL_SAFE_CALL(clEnqueueWriteBuffer(queue, deviceA, 1, 0, padded_array_size * sizeof(float), hostA, 0, 0, 0));
	CL_SAFE_CALL(clEnqueueWriteBuffer(queue, deviceB, 1, 0, padded_array_size * sizeof(float), hostB, 0, 0, 0));
#elif CH
	CL_SAFE_CALL(clEnqueueWriteBuffer(queue_read, deviceA, 1, 0, padded_array_size * sizeof(float), hostA, 0, 0, 0));
	CL_SAFE_CALL(clEnqueueWriteBuffer(queue_read, deviceB, 1, 0, padded_array_size * sizeof(float), hostB, 0, 0, 0));
#endif

	// constValue random float value between 0 and 1 for MAC operation in kernel
	float constValue = (float)rand() / (float)(RAND_MAX);

#ifdef STD
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
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 2, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(copyKernel, 3, sizeof(cl_int  ), (void*) &array_size) );

		CL_SAFE_CALL( clSetKernelArg(macKernel , 0, sizeof(void*   ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 1, sizeof(void*   ), (void*) &deviceB   ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 2, sizeof(void*   ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 3, sizeof(cl_float), (void*) &constValue) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 4, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(macKernel , 5, sizeof(cl_int  ), (void*) &array_size) );
	#endif
#elif CH
	#ifdef NDR
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
		CL_SAFE_CALL( clSetKernelArg(copyReadKernel , 2, sizeof(cl_int  ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(copyWriteKernel, 0, sizeof(void*   ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(copyWriteKernel, 1, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(copyWriteKernel, 2, sizeof(cl_int  ), (void*) &array_size) );

		CL_SAFE_CALL( clSetKernelArg(macReadKernel  , 0, sizeof(void*   ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(macReadKernel  , 1, sizeof(void*   ), (void*) &deviceB   ) );
		CL_SAFE_CALL( clSetKernelArg(macReadKernel  , 2, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(macReadKernel  , 3, sizeof(cl_int  ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(macWriteKernel , 0, sizeof(void*   ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(macWriteKernel , 1, sizeof(cl_float), (void*) &constValue) );
		CL_SAFE_CALL( clSetKernelArg(macWriteKernel , 2, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(macWriteKernel , 3, sizeof(cl_int  ), (void*) &array_size) );
	#endif
#endif
	// device warm-up
	if (verbose) printf("Device warm-up...\n");
#ifdef STD
	#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue, copyKernel, 1, NULL, globalSize, localSize, 0, 0, NULL) );
	#else
		CL_SAFE_CALL( clEnqueueTask(queue, copyKernel, 0, NULL, NULL) );
	#endif
		clFinish(queue);
#elif CH
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

#ifdef STD
	#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue, copyKernel, 1, NULL, globalSize, localSize, 0, 0, NULL) );
	#else
		CL_SAFE_CALL( clEnqueueTask(queue, copyKernel, 0, NULL, NULL) );
	#endif
		clFinish(queue);
#elif CH
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
	#ifdef STD
		CL_SAFE_CALL(clEnqueueReadBuffer(queue, deviceC, 1, 0, padded_array_size * sizeof(float), hostC, 0, 0, 0));
		clFinish(queue);
	#elif CH
		CL_SAFE_CALL(clEnqueueReadBuffer(queue_write, deviceC, 1, 0, padded_array_size * sizeof(float), hostC, 0, 0, 0));
		clFinish(queue_write);
	#endif

		printf("Verifying \"Copy\" kernel: ");
		int success = 1;
		#pragma omp parallel for ordered default(none) firstprivate(array_size, pad, hostA, hostC, verbose) shared(success)
		for (int i = 0; i < array_size; i++)
		{
			if (hostA[pad + i] != hostC[pad + i])
			{
				if (verbose) printf("Mismatch at index %d: Expected = %0.6f, Obtained = %0.6f\n", i, hostA[pad + i], hostC[pad + i]);
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
	if (verify || verbose) printf("Executing \"MAC\" kernel...\n");
	for (int i = 0; i < iter; i++)
	{
		GetTime(start);

#ifdef STD
	#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue, macKernel, 1, NULL, globalSize, localSize, 0, 0, NULL) );
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
	#ifdef STD
		CL_SAFE_CALL(clEnqueueReadBuffer(queue, deviceC, 1, 0, padded_array_size * sizeof(float), hostC, 0, 0, 0));
		clFinish(queue);
	#elif CH
		CL_SAFE_CALL(clEnqueueReadBuffer(queue_write, deviceC, 1, 0, padded_array_size * sizeof(float), hostC, 0, 0, 0));
		clFinish(queue_write);
	#endif

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