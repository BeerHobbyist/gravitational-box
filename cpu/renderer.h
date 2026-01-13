#ifndef CPU_RENDERER_H
#define CPU_RENDERER_H

#include "rgba32f.h"
#include "particle.h"

void clear_screen(Rgba32f* output, unsigned int width, unsigned int height);

void rasterize_particles(const Particles& particles, unsigned int num_particles,
                         Rgba32f* output, unsigned int width, unsigned int height,
                         unsigned int ref_width);

#endif

