//====================================================================================================================================
// Memory bandwidth benchmark kernel for OpenCL-capable FPGAs: Channelized 3D overlapped blocking
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

__kernel void copy_read(__global const float* restrict a, const int pad, const int dim_x, const int dim_y, const int halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	int y = get_local_id(1);
	int gidy = get_group_id(1);
	int z = get_global_id(2);
	int bx = gidx * (BLOCK_X - 2 * halo);
	int by = gidy * (BLOCK_Y - 2 * halo);
	int gx = bx + x - halo;
	int gy = by + y - halo;
	CHAN_WIDTH temp;

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		int real_x = gx + i;
		long index = real_x + gy * dim_x + z * dim_x * dim_y;
		if (real_x >= 0 && gy >= 0 && real_x < dim_x && gy < dim_y)
		{
			temp.data[i] = a[pad + index];
		}
	}

	write_channel(ch_copy, temp);
}

__kernel void copy_write(__global float* restrict c, const int pad, const int dim_x, const int dim_y, const int halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	int y = get_local_id(1);
	int gidy = get_group_id(1);
	int z = get_global_id(2);
	int bx = gidx * (BLOCK_X - 2 * halo);
	int by = gidy * (BLOCK_Y - 2 * halo);
	int gx = bx + x - halo;
	int gy = by + y - halo;
	CHAN_WIDTH temp;

	temp = read_channel(ch_copy);

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		int real_x = gx + i;
		long index = real_x + gy * dim_x + z * dim_x * dim_y;
		if (real_x >= 0 && gy >= 0 && real_x < dim_x && gy < dim_y)
		{
			c[pad + index] = temp.data[i];
		}
	}
}

__kernel void mac_read(__global const float* restrict a, __global const float* restrict b, const int pad, const int dim_x, const int dim_y, const int halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	int y = get_local_id(1);
	int gidy = get_group_id(1);
	int z = get_global_id(2);
	int bx = gidx * (BLOCK_X - 2 * halo);
	int by = gidy * (BLOCK_Y - 2 * halo);
	int gx = bx + x - halo;
	int gy = by + y - halo;
	CHAN_WIDTH temp_a, temp_b;

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		int real_x = gx + i;
		long index = real_x + gy * dim_x + z * dim_x * dim_y;
		if (real_x >= 0 && gy >= 0 && real_x < dim_x && gy < dim_y)
		{
			temp_a.data[i] = a[pad + index];
			temp_b.data[i] = b[pad + index];
		}
	}

	write_channel(ch_mac_a, temp_a);
	write_channel(ch_mac_b, temp_b);
}

__kernel void mac_write(__global float* restrict c, const float constValue, const int pad, const int dim_x, const int dim_y, const int halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	int y = get_local_id(1);
	int gidy = get_group_id(1);
	int z = get_global_id(2);
	int bx = gidx * (BLOCK_X - 2 * halo);
	int by = gidy * (BLOCK_Y - 2 * halo);
	int gx = bx + x - halo;
	int gy = by + y - halo;
	CHAN_WIDTH temp_a, temp_b;

	temp_a = read_channel(ch_mac_a);
	temp_b = read_channel(ch_mac_b);

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		int real_x = gx + i;
		long index = real_x + gy * dim_x + z * dim_x * dim_y;
		if (real_x >= 0 && gy >= 0 && real_x < dim_x && gy < dim_y)
		{
			c[pad + index] = constValue * temp_a.data[i] + temp_b.data[i];
		}
	}
}

#else // Single Work-item kernels

__attribute__((max_global_work_dim(0)))
__kernel void copy_read(__global const float* restrict a, const int pad, const int dim_x, const int dim_y, const int dim_z, const int x_exit, const long loop_exit, const int halo)
{
	long cond = 0;
	int x = 0;
	int y = 0;
	int z = 0;
	int bx = 0;
	int by = 0;

	while (cond != loop_exit)
	{
		cond++;

		CHAN_WIDTH temp;
		int gx = bx + x - halo;
		int gy = by + y - halo;

		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			int real_x = gx + i;
			long index = real_x + gy * dim_x + z * dim_x * dim_y;

			if (real_x >= 0 && gy >= 0 && real_x < dim_x && gy < dim_y)
			{
				temp.data[i] = a[pad + index];
			}
		}

		write_channel(ch_copy, temp);

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			y = (y + 1) & (BLOCK_Y - 1);

			if (y == 0)
			{
				z++;

				if (z == dim_z)
				{
					z = 0;
					bx += BLOCK_X - 2 * halo;

					if (bx == x_exit)
					{
						bx = 0;
						by += BLOCK_Y - 2 * halo;
					}
				}
			}
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void copy_write(__global float* restrict c, const int pad, const int dim_x, const int dim_y, const int dim_z, const int x_exit, const long loop_exit, const int halo)
{
	long cond = 0;
	int x = 0;
	int y = 0;
	int z = 0;
	int bx = 0;
	int by = 0;

	while (cond != loop_exit)
	{
		cond++;

		CHAN_WIDTH temp;
		temp = read_channel(ch_copy);
		int gx = bx + x - halo;
		int gy = by + y - halo;

		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			int real_x = gx + i;
			long index = real_x + gy * dim_x + z * dim_x * dim_y;

			if (real_x >= 0 && gy >= 0 && real_x < dim_x && gy < dim_y)
			{
				c[pad + index] = temp.data[i];
			}
		}

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			y = (y + 1) & (BLOCK_Y - 1);

			if (y == 0)
			{
				z++;

				if (z == dim_z)
				{
					z = 0;
					bx += BLOCK_X - 2 * halo;

					if (bx == x_exit)
					{
						bx = 0;
						by += BLOCK_Y - 2 * halo;
					}
				}
			}
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void mac_read(__global const float* restrict a, __global const float* restrict b, const int pad, const int dim_x, const int dim_y, const int dim_z, const int x_exit, const long loop_exit, const int halo)
{
	long cond = 0;
	int x = 0;
	int y = 0;
	int z = 0;
	int bx = 0;
	int by = 0;

	while (cond != loop_exit)
	{
		cond++;

		CHAN_WIDTH temp_a, temp_b;
		int gx = bx + x - halo;
		int gy = by + y - halo;

		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			int real_x = gx + i;
			long index = real_x + gy * dim_x + z * dim_x * dim_y;

			if (real_x >= 0 && gy >= 0 && real_x < dim_x && gy < dim_y)
			{
				temp_a.data[i] = a[pad + index];
				temp_b.data[i] = b[pad + index];
			}
		}

		write_channel(ch_mac_a, temp_a);
		write_channel(ch_mac_b, temp_b);

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			y = (y + 1) & (BLOCK_Y - 1);

			if (y == 0)
			{
				z++;

				if (z == dim_z)
				{
					z = 0;
					bx += BLOCK_X - 2 * halo;

					if (bx == x_exit)
					{
						bx = 0;
						by += BLOCK_Y - 2 * halo;
					}
				}
			}
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void mac_write(__global float* restrict c, const float constValue, const int pad, const int dim_x, const int dim_y, const int dim_z, const int x_exit, const long loop_exit, const int halo)
{
	long cond = 0;
	int x = 0;
	int y = 0;
	int z = 0;
	int bx = 0;
	int by = 0;

	while (cond != loop_exit)
	{
		cond++;

		CHAN_WIDTH temp_a, temp_b;
		temp_a = read_channel(ch_mac_a);
		temp_b = read_channel(ch_mac_b);
		int gx = bx + x - halo;
		int gy = by + y - halo;

		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			int real_x = gx + i;
			long index = real_x + gy * dim_x + z * dim_x * dim_y;

			if (real_x >= 0 && gy >= 0 && real_x < dim_x && gy < dim_y)
			{
				c[pad + index] = constValue * temp_a.data[i] + temp_b.data[i];
			}
		}

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			y = (y + 1) & (BLOCK_Y - 1);

			if (y == 0)
			{
				z++;

				if (z == dim_z)
				{
					z = 0;
					bx += BLOCK_X - 2 * halo;

					if (bx == x_exit)
					{
						bx = 0;
						by += BLOCK_Y - 2 * halo;
					}
				}
			}
		}
	}
}
#endif