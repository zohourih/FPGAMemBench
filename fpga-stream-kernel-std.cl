//====================================================================================================================================
// Memory bandwidth benchmark kernel for OpenCL-capable FPGAs: Standard
// (c) 2019, Hamid Reza Zohouri @ Tokyo Institute of Technology
//====================================================================================================================================

#ifdef NDR //NDRange kernels

__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void r1w1(__global const float* restrict a,
                   __global       float* restrict c,
                            const int             pad,
                            const long            dim_x,
                            const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	long bx = gidx * (BLOCK_X - 2 * halo);
	long gx = bx + x - halo;

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		long real_x = gx + i;
		long index = pad + real_x;
		if (real_x >= 0 && real_x < dim_x)
		{
			c[index] = a[index];
		}
	}
}

__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void r2w1(__global const float* restrict a,
                   __global const float* restrict b,
                   __global       float* restrict c,
                            const int             pad,
                            const long            dim_x,
                            const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	long bx = gidx * (BLOCK_X - 2 * halo);
	long gx = bx + x - halo;

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		long real_x = gx + i;
		long index = pad + real_x;
		if (real_x >= 0 && real_x < dim_x)
		{
			c[index] = a[index] + b[index];
		}
	}
}

#else // Single Work-item kernels

__attribute__((max_global_work_dim(0)))
__kernel void r1w1(__global const float* restrict a,
                   __global       float* restrict c,
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

		long gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
			if (real_x >= 0 && real_x < dim_x)
			{
				c[index] = a[index];
			}
		}

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			bx += BLOCK_X - halo;
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void r2w1(__global const float* restrict a,
                   __global const float* restrict b,
                   __global       float* restrict c,
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

		long gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
			if (real_x >= 0 && real_x < dim_x)
			{
				c[index] = a[index] + b[index];
			}
		}

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			bx += BLOCK_X - halo;
		}
	}
}

#endif