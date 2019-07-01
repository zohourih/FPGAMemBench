//====================================================================================================================================
// Memory bandwidth benchmark kernel for OpenCL-capable FPGAs: Standard
// (c) 2019, Hamid Reza Zohouri @ Tokyo Institute of Technology
//====================================================================================================================================

#ifdef NDR //NDRange kernels

__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void r1w1(__global const float* restrict a, __global float* restrict c, const int pad, const long size, const int overlap)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	long bx = gidx * (BLOCK_X - overlap);
	long gx = bx + x;

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		long index = gx + i;
		if (index < size)
		{
			c[pad + index] = a[pad + index];
		}
	}
}

__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void r2w1(__global const float* restrict a, __global const float* restrict b, __global float* restrict c, const float constValue, const int pad, const long size, const int overlap)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	long bx = gidx * (BLOCK_X - overlap);
	long gx = bx + x;

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		long index = gx + i;
		if (index < size)
		{
			c[pad + index] = constValue * a[pad + index] + b[pad + index];
		}
	}
}

#else // Single Work-item kernels

__attribute__((max_global_work_dim(0)))
__kernel void r1w1(__global const float* restrict a, __global float* restrict c, const int pad, const long size, const long exit, const int overlap)
{
	long cond = 0;
	int x = 0;
	long bx = 0;

	while (cond != exit)
	{
		cond++;

		long gx = bx + x;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long index = gx + i;
			if (index < size)
			{
				c[pad + index] = a[pad + index];
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
__kernel void r2w1(__global const float* restrict a, __global const float* restrict b, __global float* restrict c, const float constValue, const int pad, const long size, const long exit, const int overlap)
{
	long cond = 0;
	int x = 0;
	long bx = 0;

	while (cond != exit)
	{
		cond++;

		long gx = bx + x;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long index = gx + i;
			if (index < size)
			{
				c[pad + index] = constValue * a[pad + index] + b[pad + index];
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