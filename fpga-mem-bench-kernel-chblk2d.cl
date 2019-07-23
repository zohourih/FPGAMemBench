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

#ifndef DEPTH
	#define DEPTH 16
#endif

typedef struct
{
	float data[VEC];
} CHAN_WIDTH;

channel CHAN_WIDTH ch_R1W1   __attribute__((depth(DEPTH)));
channel CHAN_WIDTH ch_R2W1_a __attribute__((depth(DEPTH)));
channel CHAN_WIDTH ch_R2W1_b __attribute__((depth(DEPTH)));
channel CHAN_WIDTH ch_R3W1_a __attribute__((depth(DEPTH)));
channel CHAN_WIDTH ch_R3W1_b __attribute__((depth(DEPTH)));
channel CHAN_WIDTH ch_R3W1_c __attribute__((depth(DEPTH)));
channel CHAN_WIDTH ch_R2W2_a __attribute__((depth(DEPTH)));
channel CHAN_WIDTH ch_R2W2_b __attribute__((depth(DEPTH)));

//=====================================================================
// NDRange Kernels
//=====================================================================
#ifdef NDR

//=======================
// Read One - Write One
//=======================
__kernel void R1W1_read(__global const float* restrict a,
                                 const int             pad,
                                 const int             pad_x,
                                 const int             dim_x,
                                 const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	int y = get_global_id(1);
	int bx = gidx * (BLOCK_X - 2 * halo);
	int gx = bx + x - halo;
	CHAN_WIDTH temp;

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		int real_x = gx + i;
		long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);
		if (real_x >= 0 && real_x < dim_x)
		{
			temp.data[i] = a[index];
		}
	}

	write_channel(ch_R1W1, temp);
}

__kernel void R1W1_write(__global       float* restrict d,
                                  const int             pad,
                                  const int             pad_x,
                                  const int             dim_x,
                                  const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	int y = get_global_id(1);
	int bx = gidx * (BLOCK_X - 2 * halo);
	int gx = bx + x - halo;
	CHAN_WIDTH temp;

	temp = read_channel(ch_R1W1);

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		int real_x = gx + i;
		long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);
		if (real_x >= 0 && real_x < dim_x)
		{
			d[index] = temp.data[i];
		}
	}
}

//=======================
// Read Two - Write One
//=======================
__kernel void R2W1_read(__global const float* restrict a,
                        __global const float* restrict b,
                                 const int             pad,
                                 const int             pad_x,
                                 const int             dim_x,
                                 const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	int y = get_global_id(1);
	int bx = gidx * (BLOCK_X - 2 * halo);
	int gx = bx + x - halo;
	CHAN_WIDTH temp_a, temp_b;

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		int real_x = gx + i;
		long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);
		if (real_x >= 0 && real_x < dim_x)
		{
			temp_a.data[i] = a[index];
			temp_b.data[i] = b[index];
		}
	}

	write_channel(ch_R2W1_a, temp_a);
	write_channel(ch_R2W1_b, temp_b);
}

__kernel void R2W1_write(__global       float* restrict d,
                                  const int             pad,
                                  const int             pad_x,
                                  const int             dim_x,
                                  const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	int y = get_global_id(1);
	int bx = gidx * (BLOCK_X - 2 * halo);
	int gx = bx + x - halo;
	CHAN_WIDTH temp_a, temp_b;

	temp_a = read_channel(ch_R2W1_a);
	temp_b = read_channel(ch_R2W1_b);

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		int real_x = gx + i;
		long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);
		if (real_x >= 0 && real_x < dim_x)
		{
			d[index] = temp_a.data[i] + temp_b.data[i];
		}
	}
}

//=======================
// Read Three - Write One
//=======================
__kernel void R3W1_read(__global const float* restrict a,
                        __global const float* restrict b,
                        __global const float* restrict c,
                                 const int             pad,
                                 const int             pad_x,
                                 const int             dim_x,
                                 const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	int y = get_global_id(1);
	int bx = gidx * (BLOCK_X - 2 * halo);
	int gx = bx + x - halo;
	CHAN_WIDTH temp_a, temp_b, temp_c;

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		int real_x = gx + i;
		long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);
		if (real_x >= 0 && real_x < dim_x)
		{
			temp_a.data[i] = a[index];
			temp_b.data[i] = b[index];
			temp_c.data[i] = c[index];
		}
	}

	write_channel(ch_R3W1_a, temp_a);
	write_channel(ch_R3W1_b, temp_b);
	write_channel(ch_R3W1_c, temp_c);
}

__kernel void R3W1_write(__global       float* restrict d,
                                  const int             pad,
                                  const int             pad_x,
                                  const int             dim_x,
                                  const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	int y = get_global_id(1);
	int bx = gidx * (BLOCK_X - 2 * halo);
	int gx = bx + x - halo;
	CHAN_WIDTH temp_a, temp_b, temp_c;

	temp_a = read_channel(ch_R3W1_a);
	temp_b = read_channel(ch_R3W1_b);
	temp_c = read_channel(ch_R3W1_c);

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		int real_x = gx + i;
		long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);
		if (real_x >= 0 && real_x < dim_x)
		{
			d[index] = temp_a.data[i] + temp_b.data[i] + temp_c.data[i];
		}
	}
}

//=======================
// Read Two - Write Two
//=======================
__kernel void R2W2_read(__global const float* restrict a,
                        __global const float* restrict b,
                                 const int             pad,
                                 const int             pad_x,
                                 const int             dim_x,
                                 const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	int y = get_global_id(1);
	int bx = gidx * (BLOCK_X - 2 * halo);
	int gx = bx + x - halo;
	CHAN_WIDTH temp_a, temp_b;

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		int real_x = gx + i;
		long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);
		if (real_x >= 0 && real_x < dim_x)
		{
			temp_a.data[i] = a[index];
			temp_b.data[i] = b[index];
		}
	}

	write_channel(ch_R2W2_a, temp_a);
	write_channel(ch_R2W2_b, temp_b);
}

__kernel void R2W2_write(__global       float* restrict c,
                         __global       float* restrict d,
                                  const int             pad,
                                  const int             pad_x,
                                  const int             dim_x,
                                  const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	int y = get_global_id(1);
	int bx = gidx * (BLOCK_X - 2 * halo);
	int gx = bx + x - halo;
	CHAN_WIDTH temp_a, temp_b;

	temp_a = read_channel(ch_R2W2_a);
	temp_b = read_channel(ch_R2W2_b);

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		int real_x = gx + i;
		long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);
		if (real_x >= 0 && real_x < dim_x)
		{
			c[index] = temp_a.data[i];
			d[index] = temp_b.data[i];
		}
	}
}

//=====================================================================
// Single Work-item Kernels
//=====================================================================
#else

//=======================
// Read One - Write One
//=======================
__attribute__((max_global_work_dim(0)))
__kernel void R1W1_read(__global const float* restrict a,
                                 const int             pad,
                                 const int             pad_x,
                                 const int             dim_x,
                                 const int             dim_y,
                                 const long            loop_exit,
                                 const int             halo)
{
	long cond = 0;
	int x = 0;
	int y = 0;
	int bx = 0;

	while (cond != loop_exit)
	{
		cond++;

		CHAN_WIDTH temp;

		int gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			int real_x = gx + i;
			long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);

			if (real_x >= 0 && real_x < dim_x)
			{
				temp.data[i] = a[index];
			}
		}

		write_channel(ch_R1W1, temp);

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			y++;

			if (y == dim_y)
			{
				y = 0;
				bx += BLOCK_X - 2 * halo;
			}
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void R1W1_write(__global       float* restrict d,
                                  const int             pad,
                                  const int             pad_x,
                                  const int             dim_x,
                                  const int             dim_y,
                                  const long            loop_exit,
                                  const int             halo)
{
	long cond = 0;
	int x = 0;
	int y = 0;
	int bx = 0;

	while (cond != loop_exit)
	{
		cond++;

		CHAN_WIDTH temp;
		temp = read_channel(ch_R1W1);

		int gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			int real_x = gx + i;
			long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);

			if (real_x >= 0 && real_x < dim_x)
			{
				d[index] = temp.data[i];
			}
		}

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			y++;

			if (y == dim_y)
			{
				y = 0;
				bx += BLOCK_X - 2 * halo;
			}
		}
	}
}

//=======================
// Read Two - Write One
//=======================
__attribute__((max_global_work_dim(0)))
__kernel void R2W1_read(__global const float* restrict a,
                        __global const float* restrict b,
                                 const int             pad,
                                 const int             pad_x,
                                 const int             dim_x,
                                 const int             dim_y,
                                 const long            loop_exit,
                                 const int             halo)
{
	long cond = 0;
	int x = 0;
	int y = 0;
	int bx = 0;

	while (cond != loop_exit)
	{
		cond++;

		CHAN_WIDTH temp_a, temp_b;

		int gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			int real_x = gx + i;
			long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);

			if (real_x >= 0 && real_x < dim_x)
			{
				temp_a.data[i] = a[index];
				temp_b.data[i] = b[index];
			}
		}

		write_channel(ch_R2W1_a, temp_a);
		write_channel(ch_R2W1_b, temp_b);

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			y++;

			if (y == dim_y)
			{
				y = 0;
				bx += BLOCK_X - 2 * halo;
			}
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void R2W1_write(__global       float* restrict d,
                                  const int             pad,
                                  const int             pad_x,
                                  const int             dim_x,
                                  const int             dim_y,
                                  const long            loop_exit,
                                  const int             halo)
{
	long cond = 0;
	int x = 0;
	int y = 0;
	int bx = 0;

	while (cond != loop_exit)
	{
		cond++;

		CHAN_WIDTH temp_a, temp_b;
		temp_a = read_channel(ch_R2W1_a);
		temp_b = read_channel(ch_R2W1_b);

		int gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			int real_x = gx + i;
			long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);

			if (real_x >= 0 && real_x < dim_x)
			{
				d[index] = temp_a.data[i] + temp_b.data[i];
			}
		}

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			y++;

			if (y == dim_y)
			{
				y = 0;
				bx += BLOCK_X - 2 * halo;
			}
		}
	}
}

//=======================
// Read Three - Write One
//=======================
__attribute__((max_global_work_dim(0)))
__kernel void R3W1_read(__global const float* restrict a,
                        __global const float* restrict b,
                        __global const float* restrict c,
                                 const int             pad,
                                 const int             pad_x,
                                 const int             dim_x,
                                 const int             dim_y,
                                 const long            loop_exit,
                                 const int             halo)
{
	long cond = 0;
	int x = 0;
	int y = 0;
	int bx = 0;

	while (cond != loop_exit)
	{
		cond++;

		CHAN_WIDTH temp_a, temp_b, temp_c;

		int gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			int real_x = gx + i;
			long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);

			if (real_x >= 0 && real_x < dim_x)
			{
				temp_a.data[i] = a[index];
				temp_b.data[i] = b[index];
				temp_c.data[i] = c[index];
			}
		}

		write_channel(ch_R3W1_a, temp_a);
		write_channel(ch_R3W1_b, temp_b);
		write_channel(ch_R3W1_c, temp_c);

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			y++;

			if (y == dim_y)
			{
				y = 0;
				bx += BLOCK_X - 2 * halo;
			}
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void R3W1_write(__global       float* restrict d,
                                  const int             pad,
                                  const int             pad_x,
                                  const int             dim_x,
                                  const int             dim_y,
                                  const long            loop_exit,
                                  const int             halo)
{
	long cond = 0;
	int x = 0;
	int y = 0;
	int bx = 0;

	while (cond != loop_exit)
	{
		cond++;

		CHAN_WIDTH temp_a, temp_b, temp_c;
		temp_a = read_channel(ch_R3W1_a);
		temp_b = read_channel(ch_R3W1_b);
		temp_c = read_channel(ch_R3W1_c);

		int gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			int real_x = gx + i;
			long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);

			if (real_x >= 0 && real_x < dim_x)
			{
				d[index] = temp_a.data[i] + temp_b.data[i] + temp_c.data[i];
			}
		}

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			y++;

			if (y == dim_y)
			{
				y = 0;
				bx += BLOCK_X - 2 * halo;
			}
		}
	}
}

//=======================
// Read Two - Write Two
//=======================
__attribute__((max_global_work_dim(0)))
__kernel void R2W2_read(__global const float* restrict a,
                        __global const float* restrict b,
                                 const int             pad,
                                 const int             pad_x,
                                 const int             dim_x,
                                 const int             dim_y,
                                 const long            loop_exit,
                                 const int             halo)
{
	long cond = 0;
	int x = 0;
	int y = 0;
	int bx = 0;

	while (cond != loop_exit)
	{
		cond++;

		CHAN_WIDTH temp_a, temp_b;

		int gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			int real_x = gx + i;
			long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);

			if (real_x >= 0 && real_x < dim_x)
			{
				temp_a.data[i] = a[index];
				temp_b.data[i] = b[index];
			}
		}

		write_channel(ch_R2W2_a, temp_a);
		write_channel(ch_R2W2_b, temp_b);

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			y++;

			if (y == dim_y)
			{
				y = 0;
				bx += BLOCK_X - 2 * halo;
			}
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void R2W2_write(__global       float* restrict c,
                         __global       float* restrict d,
                                  const int             pad,
                                  const int             pad_x,
                                  const int             dim_x,
                                  const int             dim_y,
                                  const long            loop_exit,
                                  const int             halo)
{
	long cond = 0;
	int x = 0;
	int y = 0;
	int bx = 0;

	while (cond != loop_exit)
	{
		cond++;

		CHAN_WIDTH temp_a, temp_b;
		temp_a = read_channel(ch_R2W2_a);
		temp_b = read_channel(ch_R2W2_b);

		int gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			int real_x = gx + i;
			long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);

			if (real_x >= 0 && real_x < dim_x)
			{
				c[index] = temp_a.data[i];
				d[index] = temp_b.data[i];
			}
		}

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			y++;

			if (y == dim_y)
			{
				y = 0;
				bx += BLOCK_X - 2 * halo;
			}
		}
	}
}

#endif