//====================================================================================================================================
// Memory bandwidth benchmark kernel for OpenCL-capable FPGAs: 3D overlapped blocking
// (c) 2019, Hamid Reza Zohouri @ Tokyo Institute of Technology
//====================================================================================================================================

#ifdef NDR //NDRange kernels

__kernel void copy(__global const float* restrict a, __global float* restrict c, const int pad, const int pad_x, const int pad_y, const int dim_x, const int dim_y, const int halo)
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

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		int real_x = gx + i;
		long index = pad + z * (pad_x + dim_x) * (pad_y + dim_y) + (gy + pad_y) * (pad_x + dim_x) + (pad_x + real_x);
		if (real_x >= 0 && gy >= 0 && real_x < dim_x && gy < dim_y)
		{
			c[index] = a[index];
		}
	}
}

__kernel void mac(__global const float* restrict a, __global const float* restrict b, __global float* restrict c, const float constValue, const int pad, const int pad_x, const int pad_y, const int dim_x, const int dim_y, const int halo)
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


	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		int real_x = gx + i;
		long index = pad + z * (pad_x + dim_x) * (pad_y + dim_y) + (gy + pad_y) * (pad_x + dim_x) + (pad_x + real_x);
		if (real_x >= 0 && gy >= 0 && real_x < dim_x && gy < dim_y)
		{
			c[index] = constValue * a[index] + b[index];
		}
	}
}

#else // Single Work-item kernels

__attribute__((max_global_work_dim(0)))
__kernel void copy(__global const float* restrict a, __global float* restrict c, const int pad, const int pad_x, const int pad_y, const int dim_x, const int dim_y, const int dim_z, const int last_x, const long loop_exit, const int halo)
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

		int gx = bx + x - halo;
		int gy = by + y - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			int real_x = gx + i;
			long index = pad + z * (pad_x + dim_x) * (pad_y + dim_y) + (gy + pad_y) * (pad_x + dim_x) + (pad_x + real_x);

			if (real_x >= 0 && gy >= 0 && real_x < dim_x && gy < dim_y)
			{
				c[index] = a[index];
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

					if (bx == last_x)
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
__kernel void mac(__global const float* restrict a, __global const float* restrict b, __global float* restrict c, const float constValue, const int pad, const int pad_x, const int pad_y, const int dim_x, const int dim_y, const int dim_z, const int last_x, const long loop_exit, const int halo)
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

		int gx = bx + x - halo;
		int gy = by + y - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			int real_x = gx + i;
			long index = pad + z * (pad_x + dim_x) * (pad_y + dim_y) + (gy + pad_y) * (pad_x + dim_x) + (pad_x + real_x);

			if (real_x >= 0 && gy >= 0 && real_x < dim_x && gy < dim_y)
			{
				c[index] = constValue * a[index] + b[index];
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

					if (bx == last_x)
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