//====================================================================================================================================
// Memory bandwidth benchmark kernel for OpenCL-capable FPGAs: Channelized 2D overlapped blocking
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

__kernel void copy_read(__global const float* restrict a, const int pad, const int cols, const int halo)
{
	int x = get_local_id(0) * VEC;
	int gid = get_group_id(0);
	int y = get_global_id(1);
	int bx = gid * (BSIZE - 2 * halo);
	int gx = bx + x - halo;
	CHAN_WIDTH temp;

	#pragma unroll
	for (int j = 0; j < VEC; j++)
	{
		int real_x = gx + j;
		long index = y * cols + real_x;
		if (real_x >= 0 && real_x < cols)
		{
			temp.data[j] = a[pad + index];
		}
	}

	write_channel(ch_copy, temp);
}

__kernel void copy_write(__global float* restrict c, const int pad, const int cols, const int halo)
{
	int x = get_local_id(0) * VEC;
	int gid = get_group_id(0);
	int y = get_global_id(1);
	int bx = gid * (BSIZE - 2 * halo);
	int gx = bx + x - halo;
	CHAN_WIDTH temp;

	temp = read_channel(ch_copy);

	#pragma unroll
	for (int j = 0; j < VEC; j++)
	{
		int real_x = gx + j;
		long index = y * cols + real_x;
		if (real_x >= 0 && real_x < cols)
		{
			c[pad + index] = temp.data[j];
		}
	}
}

__kernel void mac_read(__global const float* restrict a, __global const float* restrict b, const int pad, const int cols, const int halo)
{
	int x = get_local_id(0) * VEC;
	int gid = get_group_id(0);
	int y = get_global_id(1);
	int bx = gid * (BSIZE - 2 * halo);
	int gx = bx + x - halo;
	CHAN_WIDTH temp_a, temp_b;

	#pragma unroll
	for (int j = 0; j < VEC; j++)
	{
		int real_x = gx + j;
		long index = y * cols + real_x;
		if (real_x >= 0 && real_x < cols)
		{
			temp_a.data[j] = a[pad + index];
			temp_b.data[j] = b[pad + index];
		}
	}

	write_channel(ch_mac_a, temp_a);
	write_channel(ch_mac_b, temp_b);
}

__kernel void mac_write(__global float* restrict c, const float constValue, const int pad, const int cols, const int halo)
{
	int x = get_local_id(0) * VEC;
	int gid = get_group_id(0);
	int y = get_global_id(1);
	int bx = gid * (BSIZE - 2 * halo);
	int gx = bx + x - halo;
	CHAN_WIDTH temp_a, temp_b;

	temp_a = read_channel(ch_mac_a);
	temp_b = read_channel(ch_mac_b);

	#pragma unroll
	for (int j = 0; j < VEC; j++)
	{
		int real_x = gx + j;
		long index = y * cols + real_x;
		if (real_x >= 0 && real_x < cols)
		{
			c[pad + index] = constValue * temp_a.data[j] + temp_b.data[j];
		}
	}
}

#else // Single Work-item kernels

__attribute__((max_global_work_dim(0)))
__kernel void copy_read(__global const float* restrict a, const int pad, const int rows, const int cols, const int exit, const int halo)
{
	int cond = 0;
	int x = 0;
	int y = 0;
	int bx = 0;

	while (cond != exit)
	{
		cond++;

		CHAN_WIDTH temp;
		int gx = bx + x - halo;

		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			int real_x = gx + j;
			long index = y * cols + real_x;

			if (real_x >= 0 && real_x < cols)
			{
				temp.data[j] = a[pad + index];
			}
		}

		write_channel(ch_copy, temp);

		x = (x + VEC) & (BSIZE - 1);

		if (x == 0)
		{
			y++;

			if (y == rows)
			{
				y = 0;
				bx += BSIZE - 2 * halo;
			}
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void copy_write(__global float* restrict c, const int pad, const int rows, const int cols, const int exit, const int halo)
{
	int cond = 0;
	int x = 0;
	int y = 0;
	int bx = 0;

	while (cond != exit)
	{
		cond++;

		CHAN_WIDTH temp;
		temp = read_channel(ch_copy);
		int gx = bx + x - halo;

		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			int real_x = gx + j;
			long index = y * cols + real_x;

			if (real_x >= 0 && real_x < cols)
			{
				c[pad + index] = temp.data[j];
			}
		}

		x = (x + VEC) & (BSIZE - 1);

		if (x == 0)
		{
			y++;

			if (y == rows)
			{
				y = 0;
				bx += BSIZE - 2 * halo;
			}
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void mac_read(__global const float* restrict a, __global const float* restrict b, const int pad, const int rows, const int cols, const int exit, const int halo)
{
	int cond = 0;
	int x = 0;
	int y = 0;
	int bx = 0;

	while (cond != exit)
	{
		cond++;

		CHAN_WIDTH temp_a, temp_b;
		int gx = bx + x - halo;

		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			int real_x = gx + j;
			long index = y * cols + real_x;

			if (real_x >= 0 && real_x < cols)
			{
				temp_a.data[j] = a[pad + index];
				temp_b.data[j] = b[pad + index];
			}
		}

		write_channel(ch_mac_a, temp_a);
		write_channel(ch_mac_b, temp_b);

		x = (x + VEC) & (BSIZE - 1);

		if (x == 0)
		{
			y++;

			if (y == rows)
			{
				y = 0;
				bx += BSIZE - 2 * halo;
			}
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void mac_write(__global float* restrict c, const float constValue, const int pad, const int rows, const int cols, const int exit, const int halo)
{
	int cond = 0;
	int x = 0;
	int y = 0;
	int bx = 0;

	while (cond != exit)
	{
		cond++;

		CHAN_WIDTH temp_a, temp_b;
		temp_a = read_channel(ch_mac_a);
		temp_b = read_channel(ch_mac_b);
		int gx = bx + x - halo;

		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			int real_x = gx + j;
			long index = y * cols + real_x;

			if (real_x >= 0 && real_x < cols)
			{
				c[pad + index] = constValue * temp_a.data[j] + temp_b.data[j];
			}
		}

		x = (x + VEC) & (BSIZE - 1);

		if (x == 0)
		{
			y++;

			if (y == rows)
			{
				y = 0;
				bx += BSIZE - 2 * halo;
			}
		}
	}
}

#endif