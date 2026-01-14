#ifndef CUDA_KERNEL_CUH
#define CUDA_KERNEL_CUH

#include <cuda_runtime.h>
#include "particle.h"
#include "grid.h"

// aspect_ratio = window_width / window_height (physics space is [-aspect, aspect] x [-1, 1])
void init_particles(Particles* d_particles, unsigned int num_particles, unsigned long seed, float aspect_ratio);

// ref_height: reference height for consistent particle pixel size across resizes
void update_and_render(Particles* d_particles, UniformGrid* d_grid, unsigned int num_particles,
                       float4* d_output, unsigned int width, unsigned int height,
                       unsigned int ref_height, float dt);

#endif // CUDA_KERNEL_CUH

