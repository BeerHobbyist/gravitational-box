#include "renderer.h"

#include <algorithm>

void clear_screen(Rgba32f* output, unsigned int width, unsigned int height) {
    const unsigned int n = width * height;
    const Rgba32f black = rgba32f(0.0f, 0.0f, 0.0f, 1.0f);
    for (unsigned int i = 0; i < n; i++) output[i] = black;
}

void rasterize_particles(const Particles& particles, unsigned int num_particles,
                         Rgba32f* output, unsigned int width, unsigned int height,
                         unsigned int ref_width) {
    const Rgba32f white = rgba32f(1.0f, 1.0f, 1.0f, 1.0f);

    for (unsigned int idx = 0; idx < num_particles; idx++) {
        float px = particles.x[idx];
        float py = particles.y[idx];
        float radius = particles.radius[idx];

        int center_x = (int)((px + 1.0f) * 0.5f * (float)width + 0.5f);
        int center_y = (int)((py + 1.0f) * 0.5f * (float)height + 0.5f);

        float pixel_radius_f = radius * (float)ref_width * 0.5f;
        int pixel_radius = (int)(pixel_radius_f + 0.5f);
        if (pixel_radius < 1) pixel_radius = 1;

        int r2 = pixel_radius * pixel_radius;
        for (int dy = -pixel_radius; dy <= pixel_radius; dy++) {
            for (int dx = -pixel_radius; dx <= pixel_radius; dx++) {
                int px_coord = center_x + dx;
                int py_coord = center_y + dy;
                if (px_coord < 0 || px_coord >= (int)width || py_coord < 0 || py_coord >= (int)height) {
                    continue;
                }
                if (dx * dx + dy * dy < r2) {
                    output[(unsigned int)py_coord * width + (unsigned int)px_coord] = white;
                }
            }
        }
    }
}

