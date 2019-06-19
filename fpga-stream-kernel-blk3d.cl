//====================================================================================================================================
// Memory bandwidth benchmark kernel for OpenCL-capable FPGAs: 3D overlapped blocking
// (c) 2019, Hamid Reza Zohouri @ Tokyo Institute of Technology
//====================================================================================================================================

#ifdef NDR //NDRange kernels

__kernel void copy(__global const float* restrict a, __global float* restrict c, const int pad, const int cols, const int rows, const int halo)
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
	for (int j = 0; j < VEC; j++)
	{
		int real_x = gx + j;
		long index = real_x + gy * cols + z * cols * rows;
		if (real_x >= 0 && gy >= 0 && real_x < cols && gy < rows)
		{
			c[pad + index] = a[pad + index];
		}
	}
}

__kernel void mac(__global const float* restrict a, __global const float* restrict b, __global float* restrict c, const float constValue, const int pad, const int cols, const int rows, const int halo)
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
	for (int j = 0; j < VEC; j++)
	{
		int real_x = gx + j;
		long index = real_x + gy * cols + z * cols * rows;
		if (real_x >= 0 && gy >= 0 && real_x < cols && gy < rows)
		{
			c[pad + index] = constValue * a[pad + index] + b[pad + index];
		}
	}
}

#else // Single Work-item kernels

__attribute__((max_global_work_dim(0)))
__kernel void copy(__global const float* restrict a, __global float* restrict c, const int pad, const int cols, const int rows, const int planes, const int last_col, const int exit, const int halo)
{
	int cond = 0;
	int x = 0;
	int y = 0;
	int z = 0;
	int bx = 0;
	int by = 0;

	while (cond != exit)
	{
		cond++;

		int gx = bx + x - halo;
		int gy = by + y - halo;
		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			int real_x = gx + j;
			long index = real_x + gy * cols + z * cols * rows;

			if (real_x >= 0 && gy >= 0 && real_x < cols && gy < rows)
			{
				printf("x: %02d, y: %02d, z: %02d, index: %03ld, data: %f\n", real_x, gy, z, index, a[pad + index]);
				c[pad + index] = a[pad + index];
			}
		}

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			y = (y + 1) & (BLOCK_Y - 1);

			if (y == 0)
			{
				z++;

				if (z == planes)
				{
					z = 0;
					bx += BLOCK_X - 2 * halo;

					if (bx == last_col)
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
__kernel void mac(__global const float* restrict a, __global const float* restrict b, __global float* restrict c, const float constValue, const int pad, const int cols, const int rows, const int planes, const int last_col, const int exit, const int halo)
{
	int cond = 0;
	int x = 0;
	int y = 0;
	int z = 0;
	int bx = 0;
	int by = 0;

	while (cond != exit)
	{
		cond++;

		int gx = bx + x - halo;
		int gy = by + y - halo;
		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			int real_x = gx + j;
			long index = real_x + gy * cols + z * cols * rows;

			if (real_x >= 0 && gy >= 0 && real_x < cols && gy < rows)
			{
				c[pad + index] = constValue * a[pad + index] + b[pad + index];
			}
		}

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			y = (y + 1) & (BLOCK_Y - 1);

			if (y == 0)
			{
				z++;

				if (z == planes)
				{
					z = 0;
					bx += BLOCK_X - 2 * halo;

					if (bx == last_col)
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