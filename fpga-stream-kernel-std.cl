//====================================================================================================================================
// Memory bandwidth benchmark kernel for OpenCL-capable FPGAs: Standard version
// (c) 2019, Hamid Reza Zohouri @ Tokyo Institute of Technology
//====================================================================================================================================

#ifdef NDR //NDRange kernels

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__attribute__((num_simd_work_items(VEC)))
__kernel void copy(__global const float* restrict a, __global float* restrict c, const int pad)
{
	long index = get_global_id(0);
	c[pad + index] = a[pad + index];
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__attribute__((num_simd_work_items(VEC)))
__kernel void mac(__global const float* restrict a, __global const float* restrict b, __global float* restrict c, const float constValue, const int pad)
{
	long index = get_global_id(0);
	c[pad + index] = constValue * a[pad + index] + b[pad + index];
}

#else // Single Work-item kernels

__attribute__((max_global_work_dim(0)))
__kernel void copy(__global const float* restrict a, __global float* restrict c, const int pad, const long size)
{
	for (long i = 0; i != size; i += VEC)
	{
		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			long index = i + j;
			c[pad + index] = a[pad + index];
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void mac(__global const float* restrict a, __global const float* restrict b, __global float* restrict c, const float constValue, const int pad, const long size)
{
	for (long i = 0; i != size; i += VEC)
	{
		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			long index = i + j;
			c[pad + index] = constValue * a[pad + index] + b[pad + index];
		}
	}
}

#endif