//====================================================================================================================================
// Memory bandwidth benchmark kernel for OpenCL-capable FPGAs
// Inspired by BabelStream: https://github.com/UoB-HPC/BabelStream/commits/master
// (c) 2019, Hamid Reza Zohouri @ Tokyo Institute of Technology
//====================================================================================================================================

#ifdef NDR //NDRange kernels

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__attribute__((num_simd_work_items(VEC)))
__kernel void copy(__global const float* restrict a, __global float * restrict c)
{
	int i = get_global_id(0);
	c[i] = a[i];
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__attribute__((num_simd_work_items(VEC)))
__kernel void mac(__global const float* restrict a, __global const float* restrict b, __global float* restrict c, const int constValue)
{
	int i = get_global_id(0);
	c[i] = constValue * a[i] + b[i];
}

#else // Single Work-item kernels

__attribute__((max_global_work_dim(0)))
__kernel void copy(__global const float* restrict a, __global float * restrict c, const int size)
{
	for (int i = 0; i != size; i += VEC)
	{
		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			int index = i + j;
			c[index] = a[index];
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void mac(__global const float* restrict a, __global const float* restrict b, __global float* restrict c, const int constValue, const int size)
{
	for (int i = 0; i != size; i += VEC)
	{
		#pragma unroll
		for (int j = 0; j < VEC; j++)
		{
			int index = i + j;
			c[index] = constValue * a[index] + b[index];
		}
	}
}

#endif