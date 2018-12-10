#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> maze_collision_cuda_forward(at::Tensor pos, at::Tensor maze, float RAD);

// C++ interface

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> maze_collision_forward(at::Tensor pos, at::Tensor maze, float RAD) {
  CHECK_INPUT(pos);
  CHECK_INPUT(maze);
  return maze_collision_cuda_forward(pos, maze, RAD);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &maze_collision_forward, "Maze Collision Forward (CUDA)");
}