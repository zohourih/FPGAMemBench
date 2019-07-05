//====================================================================================================================================
// Memory bandwidth benchmark kernel for OpenCL-capable FPGAs: Serial channel bandwidth for Nallatech 510T board
// (c) 2019, Hamid Reza Zohouri @ Tokyo Institute of Technology
//====================================================================================================================================

#ifdef LEGACY
	#pragma OPENCL EXTENSION cl_altera_channels : enable
	#define read_channel read_channel_altera
	#define write_channel write_channel_altera
#else
	#pragma OPENCL EXTENSION cl_intel_channels : enable
	#define read_channel read_channel_intel
	#define write_channel write_channel_intel
#endif

#define HALF_VEC VEC/2
#define WGS 64

typedef struct
{
	float data[HALF_VEC];
} CHAN_WIDTH;

#ifdef FPGA_1
channel CHAN_WIDTH sch_copy_out0 __attribute__((depth(16))) __attribute__((io("kernel_output_ch0" )));
channel CHAN_WIDTH sch_copy_out1 __attribute__((depth(16))) __attribute__((io("kernel_output_ch1" )));
#else
channel CHAN_WIDTH sch_copy_in0  __attribute__((depth(16))) __attribute__((io("kernel_input_ch0")));
channel CHAN_WIDTH sch_copy_in1  __attribute__((depth(16))) __attribute__((io("kernel_input_ch1")));
#endif

#ifdef NDR //NDRange kernels

#ifdef FPGA_1
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void R1W1_read(__global const float* restrict a, const int pad)
{
	int tid = get_global_id(0);
	long i = tid * VEC;
	CHAN_WIDTH temp0, temp1;

	#pragma unroll
	for (int j = 0; j < HALF_VEC; j++)
	{
		temp0.data[j] = a[pad + i + j];
		temp1.data[j] = a[pad + i + j + HALF_VEC];
	}

	write_channel(sch_copy_out0, temp0);
	write_channel(sch_copy_out1, temp1);
}

#elif FPGA_2
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void R1W1_write(__global float* restrict c, const int pad)
{
	int tid = get_global_id(0);
	long i = tid * VEC;
	CHAN_WIDTH temp0, temp1;

	temp0 = read_channel(sch_copy_in0);
	temp1 = read_channel(sch_copy_in1);

	#pragma unroll
	for (int j = 0; j < HALF_VEC; j++)
	{
		c[pad + i + j] = temp0.data[j];
		c[pad + i + j + HALF_VEC] = temp1.data[j];
	}
}
#endif

#else // Single Work-item kernels

#ifdef FPGA_1
__attribute__((max_global_work_dim(0)))
__kernel void R1W1_read(__global const float* restrict a, const int pad, const long size)
{
	for (long i = 0; i != size; i += VEC)
	{
		CHAN_WIDTH temp0, temp1;

		#pragma unroll
		for (int j = 0; j < HALF_VEC; j++)
		{
			temp0.data[j] = a[pad + i + j];
			temp1.data[j] = a[pad + i + j + HALF_VEC];
		}

		write_channel(sch_copy_out0, temp0);
		write_channel(sch_copy_out1, temp1);
	}
}

#elif FPGA_2
__attribute__((max_global_work_dim(0)))
__kernel void R1W1_write(__global float* restrict c, const int pad, const long size)
{
	for (long i = 0; i != size; i += VEC)
	{
		CHAN_WIDTH temp0, temp1;

		temp0 = read_channel(sch_copy_in0);
		temp1 = read_channel(sch_copy_in1);

		#pragma unroll
		for (int j = 0; j < HALF_VEC; j++)
		{
			c[pad + i + j] = temp0.data[j];
			c[pad + i + j + HALF_VEC] = temp1.data[j];
		}
	}
}
#endif

#endif