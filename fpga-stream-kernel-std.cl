//====================================================================================================================================
// Memory bandwidth benchmark kernel for OpenCL-capable FPGAs: Standard
// (c) 2019, Hamid Reza Zohouri @ Tokyo Institute of Technology
//====================================================================================================================================

//=====================================================================
// NDRange Kernels
//=====================================================================
#ifdef NDR 

//=======================
// Read One - Write Zero
//=======================
__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void R1W0(__global const float* restrict a,
                   __global       float* restrict d,
                            const int             pad,
                            const long            dim_x,
                            const int             halo)
{
	int x = get_local_id(0) * VEC;
	int gidx = get_group_id(0);
	long bx = gidx * (BLOCK_X - 2 * halo);
	long gx = bx + x - halo;
	float temp[VEC];

	#pragma unroll
	for (int i = 0; i < VEC; i++)
	{
		long real_x = gx + i;
		long index = pad + real_x;
		if (real_x >= 0 && real_x < dim_x)
		{
			temp[i] = a[index];
		}
	}

	// to prevent the compiler from optimizing out the memory accesses
	if (x == 0 && gidx == 0)
	{
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			d[i] = temp[i];
		}
	}
}

//=======================
// Read One - Write One
//=======================
__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void R1W1(__global const float* restrict a,
                   __global       float* restrict d,
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
			d[index] = a[index];
		}
	}
}

//=======================
// Read Two - Write One
//=======================
__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void R2W1(__global const float* restrict a,
                   __global const float* restrict b,
                   __global       float* restrict d,
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
			d[index] = a[index] + b[index];
		}
	}
}

//=======================
// Read Three - Write One
//=======================
__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void R3W1(__global const float* restrict a,
                   __global const float* restrict b,
                   __global const float* restrict c,
                   __global       float* restrict d,
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
			d[index] = a[index] + b[index] + c[index];
		}
	}
}

//=======================
// Read TWo - Write Two
//=======================
__attribute__((reqd_work_group_size(BLOCK_X / VEC, 1, 1)))
__kernel void R2W2(__global const float* restrict a,
                   __global const float* restrict b,
                   __global       float* restrict c,
                   __global       float* restrict d,
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
			d[index] = b[index];
		}
	}
}

//=====================================================================
// Single Work-item Kernels
//=====================================================================
#else

//=======================
// Read One - Write Zero
//=======================
__attribute__((max_global_work_dim(0)))
__kernel void R1W0(__global const float* restrict a,
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

		float temp[VEC];
		long gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
			if (real_x >= 0 && real_x < dim_x)
			{
				temp[i] = a[index];
			}
		}

		x = (x + VEC) & (BLOCK_X - 1);

		if (x == 0)
		{
			// to prevent the compiler from optimizing out the memory accesses
			if (bx == 0)
			{
				#pragma unroll
				for (int i = 0; i < VEC; i++)
				{
					d[i] = temp[i];
				}
			}
			bx += BLOCK_X - 2 * halo;
		}
	}
}

//=======================
// Read One - Write One
//=======================
__attribute__((max_global_work_dim(0)))
__kernel void R1W1(__global const float* restrict a,
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

		long gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
			if (real_x >= 0 && real_x < dim_x)
			{
				d[index] = a[index];
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
__kernel void R2W1(__global const float* restrict a,
                   __global const float* restrict b,
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

		long gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
			if (real_x >= 0 && real_x < dim_x)
			{
				d[index] = a[index] + b[index];
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
__kernel void R3W1(__global const float* restrict a,
                   __global const float* restrict b,
                   __global const float* restrict c,
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

		long gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
			if (real_x >= 0 && real_x < dim_x)
			{
				d[index] = a[index] + b[index] + c[index];
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
__kernel void R2W2(__global const float* restrict a,
                   __global const float* restrict b,
                   __global       float* restrict c,
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

		long gx = bx + x - halo;
		#pragma unroll
		for (int i = 0; i < VEC; i++)
		{
			long real_x = gx + i;
			long index = pad + real_x;
			if (real_x >= 0 && real_x < dim_x)
			{
				c[index] = a[index];
				d[index] = b[index];
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