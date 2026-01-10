#ifndef CUDA_KERNEL_CUH
#define CUDA_KERNEL_CUH

#include <cuda_runtime.h>
#include "particle.h"
#include "grid.h"

// Initialize particles with random positions and velocities
void init_particles(Particles* d_particles, unsigned int num_particles);

// Update particle positions and render to output buffer
// ref_width: reference width for consistent particle pixel size across resizes
void update_and_render(Particles* d_particles, UniformGrid* d_grid, unsigned int num_particles,
                       float4* d_output, unsigned int width, unsigned int height,
                       unsigned int ref_width, float dt);

#endif // CUDA_KERNEL_CUH
