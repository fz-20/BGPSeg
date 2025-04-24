#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "boundaryquery/boundaryquery_cuda_kernel.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("boundaryquery_cuda", &boundaryquery_cuda, "boundaryquery_cuda");
}
