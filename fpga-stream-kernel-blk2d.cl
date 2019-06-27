//====================================================================================================================================
// Memory bandwidth benchmark kernel for OpenCL-capable FPGAs: 2D overlapped blocking
// (c) 2019, Hamid Reza Zohouri @ Tokyo Institute of Technology
//====================================================================================================================================

#ifdef NDR //NDRange kernels

__kernel void copy(__global const float* restrict a, __global float* restrict c, const int pad, const int pad_x, const int dim_x, const int halo)
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
		}
	}
}

__kernel void mac(__global const float* restrict a, __global const float* restrict b, __global float* restrict c, const float constValue, const int pad, const int pad_x, const int dim_x, const int halo)
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
			c[index] = constValue * a[index] + b[index];
		}
	}
}

#else // Single Work-item kernels

__attribute__((max_global_work_dim(0)))
__kernel void copy(__global const float* restrict a, __global float* restrict c, const int pad, const int pad_x, const int dim_x, const int dim_y, const long loop_exit, const int halo)
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

__attribute__((max_global_work_dim(0)))
__kernel void mac(__global const float* restrict a, __global const float* restrict b, __global float* restrict c, const float constValue, const int pad, const int pad_x, const int dim_x, const int dim_y, const long loop_exit, const int halo)
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
				c[index] = constValue * a[index] + b[index];
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