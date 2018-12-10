#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <cmath>

#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
    i < (n); \
    i += blockDim.x * gridDim.x)



__global__ void robot_collision_cuda_forward_kernel(
	float* __restrict__ collisionCount,
	float* __restrict__ collisionLoss,
	float* __restrict__ grads,
	const long* __restrict__ sortedIdx,
	const float* __restrict__ pos,
	const size_t NTIME,
	const size_t NROBOTS,
	const size_t NDIM,
	const float RAD,
	const int SORTDIM)
{
	auto timeIdx = [NTIME, NROBOTS](int i) { return i/NROBOTS; };

	CUDA_KERNEL_LOOP(i, NTIME*NROBOTS) {
		const int myRobot = sortedIdx[i];
		const int myTime = timeIdx(i);
		const int myIdx = myTime*NROBOTS*NDIM + myRobot*NDIM;
		const float* myP = &pos[myIdx];

		// now check only the larger indexs, since collisions are pairwise we can check both at once
		for(int j=i+1; j<(myTime+1)*NROBOTS; j++) {
			// j is the index in the sorted list
			// theirIdx is the actual data index
			const int theirRobot = sortedIdx[j];
			const int theirTime = myTime; // guaranteed due to loop boundary
			const int theirIdx = theirTime*NROBOTS*NDIM + theirRobot*NDIM;
			const float* theirP = &pos[theirIdx];

			if (theirP[SORTDIM]-myP[SORTDIM] > RAD) {
				//due to sorting, if this check is true, all subsequent checks will be true so we can exit early
				break; 
			}

			float r = 0;
			for (int dim = 0; dim<NDIM; dim++) {
				r += powf(theirP[dim]-myP[dim], 2.0f);
			}
			r = sqrtf(r);

			if (r <= RAD) {
				atomicAdd(collisionCount, 2);
				atomicAdd(collisionLoss, 2*(RAD-r));

				for(int dim = 0; dim<NDIM; dim++) {
					float g = (theirP[dim]-myP[dim])/r;
					atomicAdd(&grads[myIdx+dim], g);
					atomicAdd(&grads[theirIdx+dim], -g);
				}
			}
		}
	}
}


std::vector<at::Tensor> robot_collision_cuda_forward(at::Tensor sortedIdx, at::Tensor pos, float RAD, int SORTDIM) {

	const auto NTIME = pos.size(0);
	const auto NROBOTS = pos.size(1);
	const auto NDIM = pos.size(2);

	auto collisionCount = at::zeros(pos.type(), 1);
	auto collisionLoss = at::zeros(pos.type(), 1);
	auto grads = at::zeros_like(pos);

	const int threads = 1024;
	const dim3 blocks((int)ceilf((float)NTIME*NROBOTS/(float)threads), 1);

	robot_collision_cuda_forward_kernel<<<blocks, threads>>>(
		collisionCount.data<float>(),
		collisionLoss.data<float>(),
		grads.data<float>(),
		sortedIdx.data<long>(),
		pos.data<float>(),
		NTIME,
		NROBOTS,
		NDIM,
		RAD,
		SORTDIM);

	return {collisionCount, collisionLoss, grads};
}