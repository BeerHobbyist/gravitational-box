#ifndef CPU_SIMULATION_H
#define CPU_SIMULATION_H

#include "particle.h"
#include "grid.h"

void build_uniform_grid(const Particles& particles, UniformGrid& grid, unsigned int num_particles);

void step_simulation(Particles& particles, UniformGrid& grid, unsigned int num_particles, float dt);

#endif

