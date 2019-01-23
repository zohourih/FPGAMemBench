//====================================================================================================================================
// Memory bandwidth benchmark kernel for OpenCL-capable FPGAs
// Inspired by BabelStream: https://github.com/UoB-HPC/BabelStream/commits/master
// (c) 2019, Hamid Reza Zohouri @ Tokyo Institute of Technology
//====================================================================================================================================

#ifdef NDR //NDRange kernels

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__attribute__((num_simd_work_items(VEC)))
__kernel void copy(__global const float* restrict a, __global float * restrict c, const int pad)
{
	int i = get_global_id(0);
	c[pad + i] = a[pad + i];
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__attribute__((num_simd_work_items(VEC)))
__kernel void mac(__global const float* restrict a, __global const float* restrict b, __global float* restrict c, const float constValue, const int pad)
{
	int i = get_global_id(0);
	c[pad + i] = constValue * a[pad + i] + b[pad + i];
}

#else // Single Work-item kernels

__attribute__((max_global_work_dim(0)))
__kernel void copy(__global const float* restrict a, __global float * restrict c, const int size, const int pad)
{
	for (int i = 0; i != size; i += VEC)
	{
		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			int index = pad + i + j;
			c[index] = a[index];
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void mac(__global const float* restrict a, __global const float* restrict b, __global float* restrict c, const float constValue, const int size, const int pad)
{
	for (int i = 0; i != size; i += VEC)
	{
		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			int index = pad + i + j;
			c[index] = constValue * a[index] + b[index];
		}
	}
}

#endif