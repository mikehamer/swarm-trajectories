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



__global__ void maze_collision_cuda_forward_kernel(
	float* __restrict__ collisionCount,
	float* __restrict__ grad,
	const float* __restrict__ pos,
	const uint8_t* __restrict__ maze,
	const size_t NTIME,
	const size_t NROBOTS,
	const size_t NDIM,
	const size_t MAZEX,
	const size_t MAZEY,
	const float RAD)
{
	float RADSQ = RAD*RAD;

	// A wall is defined by its south-west corner location, and extends 1 unit in x and 1 unit in y
	auto isWall = [MAZEX, MAZEY, maze](float x, float y) {
		return x < 0 || x>=MAZEX || y<0 || y>=MAZEY ? true : maze[(int)(floorf(y)+floorf(x)*MAZEY)]==0;
	};

	CUDA_KERNEL_LOOP(rt, NTIME*NROBOTS) {
		int t = rt/NROBOTS;
		int q = rt - NROBOTS*t;

		int ix = t*NROBOTS*NDIM + q*NDIM + 0;
		int iy = t*NROBOTS*NDIM + q*NDIM + 1;

		float x = pos[ix];
		float y = pos[iy];
		
		int coll = 0;

		do { // loop just once through, but allows us to break out early
			if (isWall(x,y)) {
				coll += 2;

				// the center of the wall
				float cx = floorf(x)+0.5;
				float cy = floorf(y)+0.5;

				grad[ix] += 1000*(x>cx ? -1 : 1); // negative gradient should lead out of the wall
				grad[iy] += 1000*(y>cy ? -1 : 1);

				if (fabsf(x-cx) > fabsf(y-cy)) { // x further from the center, ie closer to the border, we assume wall entry was in X
					grad[iy] *= 0.1f;
				} else { // faster to exit via Y
					grad[ix] *= 0.1f;
				}
				break; // we are in a wall... get us out of here before worrying about other conditions
			}
			
			// left wall
			if (isWall(x-1,y) && (x-floorf(x))<RAD) { // we are too close to the wall, so push us outwards
				coll++;
				grad[ix] += -1; // negative gradient should lead away from the wall (i.e. be positive)
			}

			// right wall
			else if (isWall(x+1,y) && (ceilf(x)-x)<=RAD) {
				coll++;
				grad[ix] += 1;
			}

			// bottom wall
			if (isWall(x,y-1) && (y-floorf(y))<=RAD) {
				coll++;
				grad[iy] += -1;
			}

			// top wall
			else if (isWall(x,y+1) && (ceilf(y)-y)<=RAD) {
				coll++;
				grad[iy] += 1;
			}

			if (coll != 0) {
				break; // i.e. we are next to a wall, any gradients resulting from corner detections will therefore be from other blocks that are part of the wall, and should therefore be ignored, so we break here.
			}

			// corner bottom left
			if (isWall(x-1,y-1) && (powf(x-floorf(x), 2.0f)+powf(y-floorf(y), 2.0f) <= RADSQ)) {
				coll += 2;

				float cx = floorf(x);
				float cy = floorf(y);
				float r = sqrtf(powf(x-cx, 2.0f)+powf(y-cy, 2.0f));

				grad[ix] += -1*cx/r;
				grad[iy] += -1*cy/r;
			}

			// corner bottom right
			else if (isWall(x+1,y-1) && (powf(ceilf(x)-x, 2.0f)+powf(y-floorf(y), 2.0f) <= RADSQ)) {
				coll += 2;

				float cx = ceilf(x);
				float cy = floorf(y);
				float r = sqrtf(powf(x-cx, 2.0f)+powf(y-cy, 2.0f));

				grad[ix] +=  1*cx/r;
				grad[iy] += -1*cy/r;
			}

			// corner top left
			else if (isWall(x-1,y+1) && (powf(x-floorf(x), 2.0f)+powf(ceilf(y)-y, 2.0f) <= RADSQ)) {
				coll += 2;

				float cx = floorf(x);
				float cy = ceilf(y);
				float r = sqrtf(powf(x-cx, 2.0f)+powf(y-cy, 2.0f));

				grad[ix] += -1*cx/r;
				grad[iy] +=  1*cy/r;
			}

			// corner top right
			else if (isWall(x+1,y+1) && (powf(ceilf(x)-x, 2.0f)+powf(ceilf(y)-y, 2.0f) <= RADSQ)) {
				coll += 2;

				float cx = ceilf(x);
				float cy = ceilf(y);
				float r = sqrtf(powf(x-cx, 2.0f)+powf(y-cy, 2.0f));

				grad[ix] += 1*cx/r;
				grad[iy] += 1*cy/r;
			}
		} while (0);

		atomicAdd(collisionCount, coll);
	}
}



std::vector<at::Tensor> maze_collision_cuda_forward(at::Tensor pos, at::Tensor maze, float RAD) {

	const auto NTIME = pos.size(0);
	const auto NROBOTS = pos.size(1);
	const auto NDIM = pos.size(2);
	const auto MAZEX = maze.size(0);
	const auto MAZEY = maze.size(1);

	auto collisionCount = at::zeros(pos.type(), 1);
	auto grad = at::zeros_like(pos);

	const int threads = 1024;
	const dim3 blocks((int)ceilf((float)NTIME*NROBOTS/(float)threads), 1);

	maze_collision_cuda_forward_kernel<<<blocks, threads>>>(
		collisionCount.data<float>(),
		grad.data<float>(),
		pos.data<float>(),
		maze.data<uint8_t>(),
		NTIME,
		NROBOTS,
		NDIM,
		MAZEX,
		MAZEY,
		RAD);

	return {collisionCount, grad};
}