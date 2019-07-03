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

channel CHAN_WIDTH ch_R1W0   __attribute__((depth(16)));
channel CHAN_WIDTH ch_R1W1   __attribute__((depth(16)));
channel CHAN_WIDTH ch_R2W1_a __attribute__((depth(16)));
channel CHAN_WIDTH ch_R2W1_b __attribute__((depth(16)));
channel CHAN_WIDTH ch_R3W1_a __attribute__((depth(16)));
channel CHAN_WIDTH ch_R3W1_b __attribute__((depth(16)));
channel CHAN_WIDTH ch_R3W1_c __attribute__((depth(16)));
channel CHAN_WIDTH ch_R2W2_a __attribute__((depth(16)));
channel CHAN_WIDTH ch_R2W2_b __attribute__((depth(16)));

//=====================================================================
// NDRange Kernels
//=====================================================================
#ifdef NDR

//=======================
// Read One - Write Zero
//=======================
__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void R1W0_read(__global const float* restrict a,
                                 const int             pad,
                                 const long            dim_x,
                                 const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	long bx = gidx * (BLOCK_X - 2 * halo);
	long gx = bx + x - halo;
	CHAN_WIDTH temp;

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		long real_x = gx + i;
		long index = pad + real_x;
		if (real_x >= 0 && real_x < dim_x)
		{
			temp.data[i] = a[index];
		}
	}

	// to prevent the compiler from optimizing out the memory accesses
	if (x == 0 && gidx == 0)
	{
		write_channel(ch_R1W0, temp);
	}
}

__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void R1W0_write(__global       float* restrict d)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	CHAN_WIDTH temp;

	// to prevent the compiler from optimizing out the memory accesses
	if (x == 0 && gidx == 0)
	{
		temp = read_channel(ch_R1W0);

		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			d[i] = temp.data[i];
		}
	}
}

//=======================
// Read One - Write One
//=======================
__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void R1W1_read(__global const float* restrict a,
                                 const int             pad,
                                 const long            dim_x,
                                 const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	long bx = gidx * (BLOCK_X - 2 * halo);
	long gx = bx + x - halo;
	CHAN_WIDTH temp;

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		long real_x = gx + i;
		long index = pad + real_x;
		if (real_x >= 0 && real_x < dim_x)
		{
			temp.data[i] = a[index];
		}
	}

	write_channel(ch_R1W1, temp);
}

__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void R1W1_write(__global       float* restrict d,
                                  const int             pad,
                                  const long            dim_x,
                                  const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	long bx = gidx * (BLOCK_X - 2 * halo);
	long gx = bx + x - halo;
	CHAN_WIDTH temp;

	temp = read_channel(ch_R1W1);

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		long real_x = gx + i;
		long index = pad + real_x;
		if (real_x >= 0 && real_x < dim_x)
		{
			d[index] = temp.data[i];
		}
	}
}

//=======================
// Read Two - Write One
//=======================
__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void R2W1_read(__global const float* restrict a,
                        __global const float* restrict b,
                                 const int             pad,
                                 const long            dim_x,
                                 const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	long bx = gidx * (BLOCK_X - 2 * halo);
	long gx = bx + x - halo;
	CHAN_WIDTH temp_a, temp_b;

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		long real_x = gx + i;
		long index = pad + real_x;
		if (real_x >= 0 && real_x < dim_x)
		{
			temp_a.data[i] = a[index];
			temp_b.data[i] = b[index];
		}
	}

	write_channel(ch_R2W1_a, temp_a);
	write_channel(ch_R2W1_b, temp_b);
}

__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void R2W1_write(__global       float* restrict d,
                                  const int             pad,
                                  const long            dim_x,
                                  const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	long bx = gidx * (BLOCK_X - 2 * halo);
	long gx = bx + x - halo;
	CHAN_WIDTH temp_a, temp_b;

	temp_a = read_channel(ch_R2W1_a);
	temp_b = read_channel(ch_R2W1_b);

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		long real_x = gx + i;
		long index = pad + real_x;
		if (real_x >= 0 && real_x < dim_x)
		{
			d[index] = temp_a.data[i] + temp_b.data[i];
		}
	}
}

//=======================
// Read Three - Write One
//=======================
__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void R3W1_read(__global const float* restrict a,
                        __global const float* restrict b,
                        __global const float* restrict c,
                                 const int             pad,
                                 const long            dim_x,
                                 const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	long bx = gidx * (BLOCK_X - 2 * halo);
	long gx = bx + x - halo;
	CHAN_WIDTH temp_a, temp_b, temp_c;

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		long real_x = gx + i;
		long index = pad + real_x;
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

__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void R3W1_write(__global       float* restrict d,
                                  const int             pad,
                                  const long            dim_x,
                                  const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	long bx = gidx * (BLOCK_X - 2 * halo);
	long gx = bx + x - halo;
	CHAN_WIDTH temp_a, temp_b, temp_c;

	temp_a = read_channel(ch_R3W1_a);
	temp_b = read_channel(ch_R3W1_b);
	temp_c = read_channel(ch_R3W1_c);

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		long real_x = gx + i;
		long index = pad + real_x;
		if (real_x >= 0 && real_x < dim_x)
		{
			d[index] = temp_a.data[i] + temp_b.data[i] + temp_c.data[i];
		}
	}
}

//=======================
// Read Two - Write Two
//=======================
__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void R2W2_read(__global const float* restrict a,
                        __global const float* restrict b,
                                 const int             pad,
                                 const long            dim_x,
                                 const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	long bx = gidx * (BLOCK_X - 2 * halo);
	long gx = bx + x - halo;
	CHAN_WIDTH temp_a, temp_b;

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		long real_x = gx + i;
		long index = pad + real_x;
		if (real_x >= 0 && real_x < dim_x)
		{
			temp_a.data[i] = a[index];
			temp_b.data[i] = b[index];
		}
	}

	write_channel(ch_R2W2_a, temp_a);
	write_channel(ch_R2W2_b, temp_b);
}

__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void R2W2_write(__global       float* restrict c,
                         __global       float* restrict d,
                                  const int             pad,
                                  const long            dim_x,
                                  const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	long bx = gidx * (BLOCK_X - 2 * halo);
	long gx = bx + x - halo;
	CHAN_WIDTH temp_a, temp_b;

	temp_a = read_channel(ch_R2W2_a);
	temp_b = read_channel(ch_R2W2_b);

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		long real_x = gx + i;
		long index = pad + real_x;
		if (real_x >= 0 && real_x < dim_x)
		{
			c[index] = temp_a.data[i];
			d[index] = temp_b.data[i];
		}
	}
}

#else // Single Work-item kernels

//=======================
// Read One - Write Zero
//=======================
__attribute__((max_global_work_dim(0)))
__kernel void R1W0_read(__global const float* restrict a,
                                 const int             pad,
                                 const long            dim_x,
                                 const long            exit,
                                 const int             halo)
{
	long cond = 0;
	int x = 0;
	long bx = 0;

	while (cond != exit)
	{
		cond++;

		CHAN_WIDTH temp;

		long gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
			if (real_x >= 0 && real_x < dim_x)
			{
				temp.data[i] = a[index];
			}
		}

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			// to prevent the compiler from optimizing out the memory accesses
			if (bx == 0)
			{
				write_channel(ch_R1W0, temp);
			}
			bx += BLOCK_X - 2 * halo;
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void R1W0_write(__global float* restrict d)
{
	CHAN_WIDTH temp;
	temp = read_channel(ch_R1W0);

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		d[i] = temp.data[i];
	}
}

//=======================
// Read One - Write One
//=======================
__attribute__((max_global_work_dim(0)))
__kernel void R1W1_read(__global const float* restrict a,
                                 const int             pad,
                                 const long            dim_x,
                                 const long            exit,
                                 const int             halo)
{
	long cond = 0;
	int x = 0;
	long bx = 0;

	while (cond != exit)
	{
		cond++;

		CHAN_WIDTH temp;

		long gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
			if (real_x >= 0 && real_x < dim_x)
			{
				temp.data[i] = a[index];
			}
		}

		write_channel(ch_R1W1, temp);

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			bx += BLOCK_X - 2 * halo;
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void R1W1_write(__global       float* restrict d,
                                  const int             pad,
                                  const long            dim_x,
                                  const long            exit,
                                  const int             halo)
{
	long cond = 0;
	int x = 0;
	long bx = 0;

	while (cond != exit)
	{
		cond++;

		CHAN_WIDTH temp;
		temp = read_channel(ch_R1W1);

		long gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
			if (real_x >= 0 && real_x < dim_x)
			{
				d[index] = temp.data[i];
			}
		}

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			bx += BLOCK_X - 2 * halo;
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
                                 const long            dim_x,
                                 const long            exit,
                                 const int             halo)
{
	long cond = 0;
	int x = 0;
	long bx = 0;

	while (cond != exit)
	{
		cond++;

		CHAN_WIDTH temp_a, temp_b;

		long gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
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
			bx += BLOCK_X - 2 * halo;
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void R2W1_write(__global       float* restrict d,
                                  const int             pad,
                                  const long            dim_x,
                                  const long            exit,
                                  const int             halo)
{
	long cond = 0;
	int x = 0;
	long bx = 0;

	while (cond != exit)
	{
		cond++;

		CHAN_WIDTH temp_a, temp_b;
		temp_a = read_channel(ch_R2W1_a);
		temp_b = read_channel(ch_R2W1_b);

		long gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
			if (real_x >= 0 && real_x < dim_x)
			{
				d[index] = temp_a.data[i] + temp_b.data[i];
			}
		}

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			bx += BLOCK_X - 2 * halo;
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
                                 const long            dim_x,
                                 const long            exit,
                                 const int             halo)
{
	long cond = 0;
	int x = 0;
	long bx = 0;

	while (cond != exit)
	{
		cond++;

		CHAN_WIDTH temp_a, temp_b, temp_c;

		long gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
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
			bx += BLOCK_X - 2 * halo;
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void R3W1_write(__global       float* restrict d,
                                  const int             pad,
                                  const long            dim_x,
                                  const long            exit,
                                  const int             halo)
{
	long cond = 0;
	int x = 0;
	long bx = 0;

	while (cond != exit)
	{
		cond++;

		CHAN_WIDTH temp_a, temp_b, temp_c;
		temp_a = read_channel(ch_R3W1_a);
		temp_b = read_channel(ch_R3W1_b);
		temp_c = read_channel(ch_R3W1_c);

		long gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
			if (real_x >= 0 && real_x < dim_x)
			{
				d[index] = temp_a.data[i] + temp_b.data[i] + temp_c.data[i];
			}
		}

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			bx += BLOCK_X - 2 * halo;
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
                                 const long            dim_x,
                                 const long            exit,
                                 const int             halo)
{
	long cond = 0;
	int x = 0;
	long bx = 0;

	while (cond != exit)
	{
		cond++;

		CHAN_WIDTH temp_a, temp_b;

		long gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
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
			bx += BLOCK_X - 2 * halo;
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void R2W2_write(__global       float* restrict c,
                         __global       float* restrict d,
                                  const int             pad,
                                  const long            dim_x,
                                  const long            exit,
                                  const int             halo)
{
	long cond = 0;
	int x = 0;
	long bx = 0;

	while (cond != exit)
	{
		cond++;

		CHAN_WIDTH temp_a, temp_b;
		temp_a = read_channel(ch_R2W2_a);
		temp_b = read_channel(ch_R2W2_b);

		long gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
			if (real_x >= 0 && real_x < dim_x)
			{
				c[index] = temp_a.data[i];
				d[index] = temp_b.data[i];
			}
		}

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			bx += BLOCK_X - 2 * halo;
		}
	}
}

#endif