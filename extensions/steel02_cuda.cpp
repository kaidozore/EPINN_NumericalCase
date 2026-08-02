#include <torch/extension.h>

std::vector<torch::Tensor> steel02_cuda_forward(
    torch::Tensor displacement, torch::Tensor curvature,
    torch::Tensor force_map, torch::Tensor fiber_y,
    torch::Tensor fiber_area, torch::Tensor parameters,
    torch::Tensor state);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &steel02_cuda_forward, "Steel02 history forward (CUDA)");
}
