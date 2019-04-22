//====================================================================================================================================
// Memory bandwidth benchmark kernel for OpenCL-capable FPGAs: Channelized version
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

channel CHAN_WIDTH ch_copy  __attribute__((depth(16)));
channel CHAN_WIDTH ch_mac_a __attribute__((depth(16)));
channel CHAN_WIDTH ch_mac_b __attribute__((depth(16)));

#ifdef NDR //NDRange kernels

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void copy_read(__global const float* restrict a, const int pad)
{
	int tid = get_global_id(0);
	long i = tid * VEC;
	CHAN_WIDTH temp;

	#pragma unroll
	for (int j = 0; j < VEC; j++)
	{
		temp.data[j] = a[pad + i + j];
	}

	write_channel(ch_copy, temp);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void copy_write(__global float* restrict c, const int pad)
{
	int tid = get_global_id(0);
	long i = tid * VEC;
	CHAN_WIDTH temp;

	temp = read_channel(ch_copy);

	#pragma unroll
	for (int j = 0; j < VEC; j++)
	{
		c[pad + i + j] = temp.data[j];
	}
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void mac_read(__global const float* restrict a, __global const float* restrict b, const int pad)
{
	int tid = get_global_id(0);
	long i = tid * VEC;
	CHAN_WIDTH temp_a, temp_b;

	#pragma unroll
	for (int j = 0; j < VEC; j++)
	{
		temp_a.data[j] = a[pad + i + j];
		temp_b.data[j] = b[pad + i + j];
	}

	write_channel(ch_mac_a, temp_a);
	write_channel(ch_mac_b, temp_b);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void mac_write(__global float* restrict c, const float constValue, const int pad)
{
	int tid = get_global_id(0);
	long i = tid * VEC;
	CHAN_WIDTH temp_a, temp_b;

	temp_a = read_channel(ch_mac_a);
	temp_b = read_channel(ch_mac_b);

	#pragma unroll
	for (int j = 0; j < VEC; j++)
	{
		c[pad + i + j] = constValue * temp_a.data[j] + temp_b.data[j];
	}
}

#else // Single Work-item kernels

__attribute__((max_global_work_dim(0)))
__kernel void copy_read(__global const float* restrict a, const int pad, const long size)
{
	for (long i = 0; i != size; i += VEC)
	{
		CHAN_WIDTH temp;

		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			temp.data[j] = a[pad + i + j];
		}

		write_channel(ch_copy, temp);
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void copy_write(__global float* restrict c, const int pad, const long size)
{
	for (long i = 0; i != size; i += VEC)
	{
		CHAN_WIDTH temp;

		temp = read_channel(ch_copy);

		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			c[pad + i + j] = temp.data[j];
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void mac_read(__global const float* restrict a, __global const float* restrict b, const int pad, const long size)
{
	for (long i = 0; i != size; i += VEC)
	{
		CHAN_WIDTH temp_a, temp_b;

		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			temp_a.data[j] = a[pad + i + j];
			temp_b.data[j] = b[pad + i + j];
		}

		write_channel(ch_mac_a, temp_a);
		write_channel(ch_mac_b, temp_b);
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void mac_write(__global float* restrict c, const float constValue, const int pad, const long size)
{
	for (long i = 0; i != size; i += VEC)
	{
		CHAN_WIDTH temp_a, temp_b;

		temp_a = read_channel(ch_mac_a);
		temp_b = read_channel(ch_mac_b);

		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			c[pad + i + j] = constValue * temp_a.data[j] + temp_b.data[j];
		}
	}
}

#endif