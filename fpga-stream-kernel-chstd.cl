//====================================================================================================================================
// Memory bandwidth benchmark kernel for OpenCL-capable FPGAs: Channelized
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

typedef struct
{
	float data[VEC];
} CHAN_WIDTH;

channel CHAN_WIDTH ch_copy  __attribute__((depth(16)));
channel CHAN_WIDTH ch_mac_a __attribute__((depth(16)));
channel CHAN_WIDTH ch_mac_b __attribute__((depth(16)));

#ifdef NDR //NDRange kernels

__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void r1w1_read(__global const float* restrict a,
                                 const int             pad,
                                 const long            size,
                                 const int             overlap)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	long bx = gidx * (BLOCK_X - overlap);
	long gx = bx + x;
	CHAN_WIDTH temp;

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		long real_x = gx + i;
		long index = pad + real_x;
		if (real_x < size)
		{
			temp.data[i] = a[index];
		}
	}

	write_channel(ch_copy, temp);
}

__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void r1w1_write(__global       float* restrict c,
                                  const int             pad,
                                  const long            size,
                                  const int             overlap)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	long bx = gidx * (BLOCK_X - overlap);
	long gx = bx + x;
	CHAN_WIDTH temp;

	temp = read_channel(ch_copy);

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		long real_x = gx + i;
		long index = pad + real_x;
		if (real_x < size)
		{
			c[index] = temp.data[i];
		}
	}
}

__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void r2w1_read(__global const float* restrict a,
                        __global const float* restrict b,
                                 const int             pad,
                                 const long            size,
                                 const int             overlap)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	long bx = gidx * (BLOCK_X - overlap);
	long gx = bx + x;
	CHAN_WIDTH temp_a, temp_b;

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		long real_x = gx + i;
		long index = pad + real_x;
		if (real_x < size)
		{
			temp_a.data[i] = a[index];
			temp_b.data[i] = b[index];
		}
	}

	write_channel(ch_mac_a, temp_a);
	write_channel(ch_mac_b, temp_b);
}

__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void r2w1_write(__global       float* restrict c,
                                  const int             pad,
                                  const long            size,
                                  const int             overlap)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	long bx = gidx * (BLOCK_X - overlap);
	long gx = bx + x;
	CHAN_WIDTH temp_a, temp_b;

	temp_a = read_channel(ch_mac_a);
	temp_b = read_channel(ch_mac_b);

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		long real_x = gx + i;
		long index = pad + real_x;
		if (real_x < size)
		{
			c[index] = temp_a.data[i] + temp_b.data[i];
		}
	}
}

#else // Single Work-item kernels

__attribute__((max_global_work_dim(0)))
__kernel void r1w1_read(__global const float* restrict a,
                                 const int             pad,
                                 const long            size,
                                 const long            exit,
                                 const int             overlap)
{
	long cond = 0;
	int x = 0;
	long bx = 0;

	while (cond != exit)
	{
		cond++;

		CHAN_WIDTH temp;
		long gx = bx + x;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
			if (real_x < size)
			{
				temp.data[i] = a[index];
			}
		}

		write_channel(ch_copy, temp);

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			bx += BLOCK_X - overlap;
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void r1w1_write(__global       float* restrict c,
                                  const int             pad,
                                  const long            size,
                                  const long            exit,
                                  const int             overlap)
{
	long cond = 0;
	int x = 0;
	long bx = 0;

	while (cond != exit)
	{
		cond++;

		CHAN_WIDTH temp;
		temp = read_channel(ch_copy);
		long gx = bx + x;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
			if (real_x < size)
			{
				c[index] = temp.data[i];
			}
		}

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			bx += BLOCK_X - overlap;
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void r2w1_read(__global const float* restrict a,
                        __global const float* restrict b,
                                 const int             pad,
                                 const long            size,
                                 const long            exit,
                                 const int             overlap)
{
	long cond = 0;
	int x = 0;
	long bx = 0;

	while (cond != exit)
	{
		cond++;

		CHAN_WIDTH temp_a, temp_b;
		long gx = bx + x;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
			if (real_x < size)
			{
				temp_a.data[i] = a[index];
				temp_b.data[i] = b[index];
			}
		}

		write_channel(ch_mac_a, temp_a);
		write_channel(ch_mac_b, temp_b);

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			bx += BLOCK_X - overlap;
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void r2w1_write(__global       float* restrict c,
                                  const int             pad,
                                  const long            size,
                                  const long            exit,
                                  const int             overlap)
{
	long cond = 0;
	int x = 0;
	long bx = 0;

	while (cond != exit)
	{
		cond++;

		CHAN_WIDTH temp_a, temp_b;
		temp_a = read_channel(ch_mac_a);
		temp_b = read_channel(ch_mac_b);
		long gx = bx + x;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
			if (real_x < size)
			{
				c[index] = temp_a.data[i] + temp_b.data[i];
			}
		}

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			bx += BLOCK_X - overlap;
		}
	}
}

#endif