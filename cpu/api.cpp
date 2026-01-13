#include "api.h"

#include "renderer.h"
#include "simulation.h"

void update_and_render(Particles* particles, UniformGrid* grid, unsigned int num_particles,
                       Rgba32f* output, unsigned int width, unsigned int height,
                       unsigned int ref_width, float dt) {
    step_simulation(*particles, *grid, num_particles, dt);
    clear_screen(output, width, height);
    rasterize_particles(*particles, num_particles, output, width, height, ref_width);
}

