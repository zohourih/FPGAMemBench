//====================================================================================================================================
// Memory bandwidth benchmark kernel for OpenCL-capable FPGAs: 2D overlapped blocking
// (c) 2019, Hamid Reza Zohouri @ Tokyo Institute of Technology
//====================================================================================================================================

//=====================================================================
// NDRange Kernels
//=====================================================================
#ifdef NDR

//=======================
// Read One - Write ONE
//=======================
__kernel void R1W1(__global const float* restrict a,
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

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		int real_x = gx + i;
		long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);
		if (real_x >= 0 && real_x < dim_x)
		{
			d[index] = a[index];
		}
	}
}

//=======================
// Read Two - Write One
//=======================
__kernel void R2W1(__global const float* restrict a,
                   __global const float* restrict b,
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

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		int real_x = gx + i;
		long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);
		if (real_x >= 0 && real_x < dim_x)
		{
			d[index] = a[index] + b[index];
		}
	}
}

//=======================
// Read Three - Write One
//=======================
__kernel void R3W1(__global const float* restrict a,
                   __global const float* restrict b,
                   __global const float* restrict c,
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

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		int real_x = gx + i;
		long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);
		if (real_x >= 0 && real_x < dim_x)
		{
			d[index] = a[index] + b[index] + c[index];
		}
	}
}

//=======================
// Read Two - Write Two
//=======================
__kernel void R2W2(__global const float* restrict a,
                   __global const float* restrict b,
                   __global       float* restrict c,
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

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		int real_x = gx + i;
		long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);
		if (real_x >= 0 && real_x < dim_x)
		{
			c[index] = a[index];
			d[index] = b[index];
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
__kernel void R1W1(__global const float* restrict a,
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

		int gx = bx + x - halo;

		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			int real_x = gx + i;
			long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);

			if (real_x >= 0 && real_x < dim_x)
			{
				d[index] = a[index];
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
__kernel void R2W1(__global const float* restrict a,
                   __global const float* restrict b,
                   __global       float* restrict d,
                            const int    pad,
                            const int    pad_x,
                            const int    dim_x,
                            const int    dim_y,
                            const long   loop_exit,
                            const int    halo)
{
	long cond = 0;
	int x = 0;
	int y = 0;
	int bx = 0;

	while (cond != loop_exit)
	{
		cond++;

		int gx = bx + x - halo;

		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			int real_x = gx + i;
			long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);

			if (real_x >= 0 && real_x < dim_x)
			{
				d[index] = a[index] + b[index];
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
__kernel void R3W1(__global const float* restrict a,
                   __global const float* restrict b,
                   __global const float* restrict c,
                   __global       float* restrict d,
                            const int    pad,
                            const int    pad_x,
                            const int    dim_x,
                            const int    dim_y,
                            const long   loop_exit,
                            const int    halo)
{
	long cond = 0;
	int x = 0;
	int y = 0;
	int bx = 0;

	while (cond != loop_exit)
	{
		cond++;

		int gx = bx + x - halo;

		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			int real_x = gx + i;
			long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);

			if (real_x >= 0 && real_x < dim_x)
			{
				d[index] = a[index] + b[index] + c[index];
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
__kernel void R2W2(__global const float* restrict a,
                   __global const float* restrict b,
                   __global       float* restrict c,
                   __global       float* restrict d,
                            const int    pad,
                            const int    pad_x,
                            const int    dim_x,
                            const int    dim_y,
                            const long   loop_exit,
                            const int    halo)
{
	long cond = 0;
	int x = 0;
	int y = 0;
	int bx = 0;

	while (cond != loop_exit)
	{
		cond++;

		int gx = bx + x - halo;

		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			int real_x = gx + i;
			long index = pad + y * (pad_x + dim_x) + (pad_x + real_x);

			if (real_x >= 0 && real_x < dim_x)
			{
				c[index] = a[index];
				d[index] = b[index];
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