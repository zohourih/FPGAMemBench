//====================================================================================================================================
// Memory bandwidth benchmark kernel for OpenCL-capable FPGAs: Serial channel for Nallatech 510T board - MAC kernel
// Inspired by BabelStream: https://github.com/UoB-HPC/BabelStream/commits/master
// (c) 2019, Hamid Reza Zohouri @ Tokyo Institute of Technology
//====================================================================================================================================

#ifdef LEGACY
	#pragma OPENCL EXTENSION cl_altera_channels : enable;
	#define read_channel read_channel_altera
	#define write_channel write_channel_altera
#else
	#pragma OPENCL EXTENSION cl_intel_channels : enable;
	#define read_channel read_channel_intel
	#define write_channel write_channel_intel
#endif

typedef struct
{
	float data[VEC];
} CHAN_WIDTH;

channel CHAN_WIDTH sch_mac_in0  __attribute__((depth(4))) __attribute__((io("kernel_input_ch0" )));
channel CHAN_WIDTH sch_mac_out0 __attribute__((depth(4))) __attribute__((io("kernel_output_ch0")));
channel CHAN_WIDTH sch_mac_in1  __attribute__((depth(4))) __attribute__((io("kernel_input_ch1" )));
channel CHAN_WIDTH sch_mac_out1 __attribute__((depth(4))) __attribute__((io("kernel_output_ch1")));

#ifdef NDR //NDRange kernels

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void mac_read(__global const float* restrict a, __global const float* restrict b, const int pad)
{
	int tid = get_global_id(0);
	int i = tid * VEC;
	CHAN_WIDTH temp_a, temp_b;

	#pragma unroll
	for (int j = 0; j < VEC; j++)
	{
		temp_a.data[j] = a[pad + i + j];
		temp_b.data[j] = b[pad + i + j];
	}

	write_channel(sch_mac_in0, temp_a);
	write_channel(sch_mac_in1, temp_b);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void mac_write(__global float* restrict c, const float constValue, const int pad)
{
	int tid = get_global_id(0);
	int i = tid * VEC;
	CHAN_WIDTH temp_a, temp_b;

	temp_a = read_channel(sch_mac_out0);
	temp_b = read_channel(sch_mac_out1);

	#pragma unroll
	for (int j = 0; j < VEC; j++)
	{
		c[pad + i + j] = constValue * temp_a.data[j] + temp_b.data[j];
	}
}

#else // Single Work-item kernels

__attribute__((max_global_work_dim(0)))
__kernel void mac_read(__global const float* restrict a, __global const float* restrict b, const int pad, const int size)
{
	for (int i = 0; i != size; i += VEC)
	{
		CHAN_WIDTH temp_a, temp_b;

		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			temp_a.data[j] = a[pad + i + j];
			temp_b.data[j] = b[pad + i + j];
		}

		write_channel(sch_mac_in0, temp_a);
		write_channel(sch_mac_in1, temp_b);
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void mac_write(__global float* restrict c, const float constValue, const int pad, const int size)
{
	for (int i = 0; i != size; i += VEC)
	{
		CHAN_WIDTH temp_a, temp_b;

		temp_a = read_channel(sch_mac_out0);
		temp_b = read_channel(sch_mac_out1);

		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			c[pad + i + j] = constValue * temp_a.data[j] + temp_b.data[j];
		}
	}
}

#endif