#ifndef CPU_API_H
#define CPU_API_H

#include "rgba32f.h"
#include "particle.h"
#include "grid.h"

void init_particles(Particles* particles, unsigned int num_particles, unsigned long seed);

void update_and_render(Particles* particles, UniformGrid* grid, unsigned int num_particles,
                       Rgba32f* output, unsigned int width, unsigned int height,
                       unsigned int ref_width, float dt);

#endif

