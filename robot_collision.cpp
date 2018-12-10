#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> robot_collision_cuda_forward(at::Tensor sortedIdx, at::Tensor pos, float RAD, int SORTDIM);

// C++ interface

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> robot_collision_forward(at::Tensor sortedIdx, at::Tensor pos, float RAD, int SORTDIM) {
  CHECK_INPUT(pos);
  CHECK_INPUT(sortedIdx);
  return robot_collision_cuda_forward(sortedIdx, pos, RAD, SORTDIM);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &robot_collision_forward, "Circle Collision Forward (CUDA)");
}