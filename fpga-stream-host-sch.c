//====================================================================================================================================
// Memory bandwidth benchmark host for OpenCL-capable FPGAs: Serial channel bandwidth for Nallatech 510T board
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
#define WGS 64

// global variables
static cl_context       context;
static cl_command_queue queue_read, queue_write;
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

	free(platforms); // platforms isn't needed in the main function
}

static inline void usage(char **argv)
{
	printf("\nUsage: %s -s <buffer size in MiB> -n <number of iterations> -pad <array padding indexes> --verbose --verify\n", argv[0]);
}

int main(int argc, char **argv)
{
	// input arguments
	int size_MiB = 100; 							// buffer size, default size is 100 MiB
	int iter = 1;									// number of iterations
	int pad = 0;									// padding
	int verbose = 0, verify = 0;

	// timing measurement
	TimeStamp start, end;
	double totalR1W1Time = 0, avgR1W1Time = 0;

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

	char clOptions[200] = "";

#ifdef NDR
	sprintf(clOptions + strlen(clOptions), "-DNDR");
#endif

	// compile kernel file
	clBuildProgram_SAFE(progFPGA1, 1, &deviceList[0], clOptions, NULL, NULL);
	clBuildProgram_SAFE(progFPGA2, 1, &deviceList[1], clOptions, NULL, NULL);

	// create kernel objects
	cl_kernel R1W1ReadKernel, R1W1WriteKernel;

	R1W1ReadKernel = clCreateKernel(progFPGA1, "R1W1_read", &error);
	if(error != CL_SUCCESS)
	{
		printf("ERROR: clCreateKernel(R1W1_read) failed with error: ");
		display_error_message(error, stdout);
		return -1;
	}

	R1W1WriteKernel= clCreateKernel(progFPGA2, "R1W1_write", &error);
	if(error != CL_SUCCESS)
	{
		printf("ERROR: clCreateKernel(R1W1_write) failed with error: ");
		display_error_message(error, stdout);
		return -1;
	}

	clReleaseProgram(progFPGA1);
	clReleaseProgram(progFPGA2);

	printf("Kernel type:           Nallatech 510T serial channel\n");

#ifdef NDR
	printf("Kernel model:          NDRange\n");
#else
	printf("Kernel model:          Single Work-item\n");
#endif

	printf("Array size:            %ld indexes\n", array_size);
	printf("Buffer size:           %d MiB\n", size_MiB);
	printf("Total memory usage:    %d MiB\n", 3 * size_MiB);
	printf("Vector size:           %d\n", VEC);
	printf("Array padding:         %d\n\n", pad);

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
	CL_SAFE_CALL(clEnqueueWriteBuffer(queue_read, deviceA, 1, 0, padded_size_Byte, hostA, 0, 0, 0));

	#ifdef NDR
		size_t localSize[3] = {(size_t)WGS, 1, 1};
		size_t globalSize[3] = {(size_t)(array_size / VEC), 1, 1};

		CL_SAFE_CALL( clSetKernelArg(R1W1ReadKernel , 0, sizeof(cl_mem  ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(R1W1ReadKernel , 1, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(R1W1WriteKernel, 0, sizeof(cl_mem  ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(R1W1WriteKernel, 1, sizeof(cl_int  ), (void*) &pad       ) );
	#else
		CL_SAFE_CALL( clSetKernelArg(R1W1ReadKernel , 0, sizeof(cl_mem  ), (void*) &deviceA   ) );
		CL_SAFE_CALL( clSetKernelArg(R1W1ReadKernel , 1, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(R1W1ReadKernel , 2, sizeof(cl_long ), (void*) &array_size) );
		CL_SAFE_CALL( clSetKernelArg(R1W1WriteKernel, 0, sizeof(cl_mem  ), (void*) &deviceC   ) );
		CL_SAFE_CALL( clSetKernelArg(R1W1WriteKernel, 1, sizeof(cl_int  ), (void*) &pad       ) );
		CL_SAFE_CALL( clSetKernelArg(R1W1WriteKernel, 2, sizeof(cl_long ), (void*) &array_size) );
	#endif

	// device warm-up
	if (verbose) printf("Device warm-up...\n");
	#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue_read , R1W1ReadKernel , DIM, NULL, globalSize, localSize, 0, 0, NULL) );
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue_write, R1W1WriteKernel, DIM, NULL, globalSize, localSize, 0, 0, NULL) );
	#else
		CL_SAFE_CALL( clEnqueueTask(queue_read , R1W1ReadKernel , 0, NULL, NULL) );
		CL_SAFE_CALL( clEnqueueTask(queue_write, R1W1WriteKernel, 0, NULL, NULL) );
	#endif
		clFinish(queue_write);

	// R1W1 kernel
	if (verify || verbose) printf("Executing \"R1W1\" kernel...\n");
	for (int i = 0; i < iter; i++)
	{
		GetTime(start);

	#ifdef NDR
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue_read , R1W1ReadKernel , DIM, NULL, globalSize, localSize, 0, 0, NULL) );
		CL_SAFE_CALL( clEnqueueNDRangeKernel(queue_write, R1W1WriteKernel, DIM, NULL, globalSize, localSize, 0, 0, NULL) );
	#else
		CL_SAFE_CALL( clEnqueueTask(queue_read , R1W1ReadKernel , 0, NULL, NULL) );
		CL_SAFE_CALL( clEnqueueTask(queue_write, R1W1WriteKernel, 0, NULL, NULL) );
	#endif
		clFinish(queue_write);

		GetTime(end);
		totalR1W1Time += TimeDiff(start, end);
	}

	// verify R1W1 kernel
	if (verify)
	{
		// read data back to host
		printf("Reading data back from device...\n");
		CL_SAFE_CALL(clEnqueueReadBuffer(queue_write, deviceC, 1, 0, padded_size_Byte, hostC, 0, 0, 0));
		clFinish(queue_write);

		printf("Verifying \"R1W1\" kernel: ");
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

	if (verify || verbose) printf("\n");

	avgR1W1Time = totalR1W1Time / (double)iter;
	printf("Channel bandwidth: %.3f GB/s (%.3f GiB/s) @%.1f ms\n", (double)(1 * size_B) / (1.0E6 * avgR1W1Time), (double)(1 * size_MiB * 1000.0) / (1024.0 * avgR1W1Time), avgR1W1Time);
	printf("Memory bandwidth : %.3f GB/s (%.3f GiB/s) @%.1f ms\n", (double)(2 * size_B) / (1.0E6 * avgR1W1Time), (double)(2 * size_MiB * 1000.0) / (1024.0 * avgR1W1Time), avgR1W1Time);

	clReleaseCommandQueue(queue_read);
	clReleaseCommandQueue(queue_write);
	clReleaseContext(context);
	clReleaseMemObject(deviceA);
	clReleaseMemObject(deviceB);
	clReleaseMemObject(deviceC);

	free(hostA);
	free(hostB);
	free(hostC);
	free(kernelSourceFPGA1);
	free(kernelSourceFPGA2);
	free(deviceList);
}