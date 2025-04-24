#ifndef _BOUNDARYQUERY_CUDA_KERNEL
#define _BOUNDARYQUERY_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void boundaryquery_cuda(int m, int nsample, int b, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, at::Tensor offset_tensor, at::Tensor new_offset_tensor, at::Tensor idx_tensor, at::Tensor dist2_tensor, at::Tensor edges_tensor, at::Tensor boundary_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void boundaryquery_cuda_launcher(int m, int nsample, int b, const float *xyz, const float *new_xyz, const int *offset, const int *new_offset, int *idx, float *dist2, const int *edges, const int *boundary);

#ifdef __cplusplus
}
#endif
#endif
