//====================================================================================================================================
// Memory bandwidth benchmark kernel for OpenCL-capable FPGAs: Serial channel bandwidth for Nallatech 510T board
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

#ifdef FPGA_1
channel CHAN_WIDTH sch_copy_out __attribute__((depth(16))) __attribute__((io("kernel_output_ch0")));
#else
channel CHAN_WIDTH sch_copy_in  __attribute__((depth(16))) __attribute__((io("kernel_input_ch0" )));
#endif

#ifdef NDR //NDRange kernels

#ifdef FPGA_1
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void copy_read(__global const float* restrict a, const int pad)
{
	int tid = get_global_id(0);
	int i = tid * VEC;
	CHAN_WIDTH temp;

	#pragma unroll
	for (int j = 0; j < VEC; j++)
	{
		temp.data[j] = a[pad + i + j];
	}

	write_channel(sch_copy_out, temp);
}

#elif FPGA_2
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void copy_write(__global float * restrict c, const int pad)
{
	int tid = get_global_id(0);
	int i = tid * VEC;
	CHAN_WIDTH temp;

	temp = read_channel(sch_copy_in);

	#pragma unroll
	for (int j = 0; j < VEC; j++)
	{
		c[pad + i + j] = temp.data[j];
	}
}
#endif

#else // Single Work-item kernels

#ifdef FPGA_1
__attribute__((max_global_work_dim(0)))
__kernel void copy_read(__global const float* restrict a, const int pad, const int size)
{
	for (int i = 0; i != size; i += VEC)
	{
		CHAN_WIDTH temp;

		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			temp.data[j] = a[pad + i + j];
		}

		write_channel(sch_copy_out, temp);
	}
}

#elif FPGA_2
__attribute__((max_global_work_dim(0)))
__kernel void copy_write(__global float * restrict c, const int pad, const int size)
{
	for (int i = 0; i != size; i += VEC)
	{
		CHAN_WIDTH temp;

		temp = read_channel(sch_copy_in);

		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			c[pad + i + j] = temp.data[j];
		}
	}
}
#endif

#endif