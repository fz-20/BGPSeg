#include <vector>
// #include <THC/THC.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "boundaryquery_cuda_kernel.h"


void boundaryquery_cuda(int m, int nsample, int b, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, at::Tensor offset_tensor, at::Tensor new_offset_tensor, at::Tensor idx_tensor, at::Tensor dist2_tensor, at::Tensor edges_tensor, at::Tensor boundary_tensor)
{
    const float *xyz = xyz_tensor.data_ptr<float>();
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const int *offset = offset_tensor.data_ptr<int>();
    const int *new_offset = new_offset_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();
    float *dist2 = dist2_tensor.data_ptr<float>();
    const int *edges = edges_tensor.data_ptr<int>();
    const int *boundary = boundary_tensor.data_ptr<int>();
    boundaryquery_cuda_launcher(m, nsample, b, xyz, new_xyz, offset, new_offset, idx, dist2, edges, boundary);
}