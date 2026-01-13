#include "simulation.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

static inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

void init_particles(Particles* particles, unsigned int num_particles, unsigned long seed) {
    const float golden_angle = 2.39996322f;
    const float circle_radius = 0.8f;

    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::uniform_real_distribution<float> uni01(0.0f, 1.0f);

    for (unsigned int idx = 0; idx < num_particles; idx++) {
        float angle = idx * golden_angle;
        float r_norm = (num_particles > 1) ? std::sqrt((float)idx / (float)(num_particles - 1)) : 0.0f;
        float dist = r_norm * circle_radius;

        particles->x[idx] = dist * std::cos(angle);
        particles->y[idx] = dist * std::sin(angle);

        float u1 = uni01(rng);
        float u2 = uni01(rng);
        particles->vx[idx] = (u1 * 2.0f - 1.0f) * MAX_VELOCITY;
        particles->vy[idx] = (u2 * 2.0f - 1.0f) * MAX_VELOCITY;

        float u3 = uni01(rng);
        float particle_radius = MIN_RADIUS + u3 * (MAX_RADIUS - MIN_RADIUS);
        particles->radius[idx] = particle_radius;
        particles->mass[idx] = particle_radius * particle_radius;
    }
}

void build_uniform_grid(const Particles& particles, UniformGrid& grid, unsigned int num_particles) {
    std::fill(grid.cell_start.begin(), grid.cell_start.end(), -1);
    std::fill(grid.cell_count.begin(), grid.cell_count.end(), 0);

    for (unsigned int idx = 0; idx < num_particles; idx++) {
        float x = particles.x[idx];
        float y = particles.y[idx];

        int cell_x = (int)((x + 1.0f) / grid.cell_size);
        int cell_y = (int)((y + 1.0f) / grid.cell_size);

        cell_x = std::max(0, std::min(cell_x, grid.grid_width - 1));
        cell_y = std::max(0, std::min(cell_y, grid.grid_height - 1));

        int cell_id = cell_y * grid.grid_width + cell_x;
        grid.particle_cell[idx] = cell_id;
        grid.particle_indices[idx] = (int)idx;
    }

    std::vector<int> order(num_particles);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return grid.particle_cell[a] < grid.particle_cell[b];
    });

    std::vector<int> sorted_cell(num_particles);
    std::vector<int> sorted_idx(num_particles);
    for (unsigned int i = 0; i < num_particles; i++) {
        int src = order[i];
        sorted_cell[i] = grid.particle_cell[src];
        sorted_idx[i] = grid.particle_indices[src];
    }

    grid.particle_cell.swap(sorted_cell);
    grid.particle_indices.swap(sorted_idx);

    for (unsigned int i = 0; i < num_particles; i++) {
        int cell_id = grid.particle_cell[i];
        if (grid.cell_start[cell_id] == -1) {
            grid.cell_start[cell_id] = (int)i;
        }
        grid.cell_count[cell_id] += 1;
    }
}

static void integrate_collisions(Particles& particles, const UniformGrid& grid, unsigned int num_particles, float dt) {
    for (unsigned int idx = 0; idx < num_particles; idx++) {
        const float orig_x = particles.x[idx];
        const float orig_y = particles.y[idx];
        const float orig_vx = particles.vx[idx];
        const float orig_vy = particles.vy[idx];
        float radius = particles.radius[idx];
        float mass = particles.mass[idx];

        float corr_x = 0.0f;
        float corr_y = 0.0f;
        float imp_vx = 0.0f;
        float imp_vy = 0.0f;
        int collision_count = 0;

        int cell_x = (int)((orig_x + 1.0f) / grid.cell_size);
        int cell_y = (int)((orig_y + 1.0f) / grid.cell_size);
        cell_x = std::max(0, std::min(cell_x, grid.grid_width - 1));
        cell_y = std::max(0, std::min(cell_y, grid.grid_height - 1));

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int neighbor_x = cell_x + dx;
                int neighbor_y = cell_y + dy;

                if (neighbor_x < 0 || neighbor_x >= grid.grid_width ||
                    neighbor_y < 0 || neighbor_y >= grid.grid_height) continue;

                int neighbor_cell = neighbor_y * grid.grid_width + neighbor_x;
                int start = grid.cell_start[neighbor_cell];
                int count = grid.cell_count[neighbor_cell];
                if (start == -1) continue;

                for (int i = 0; i < count; i++) {
                    int other_idx = grid.particle_indices[start + i];
                    if (other_idx == (int)idx) continue;

                    float other_x = particles.x[other_idx];
                    float other_y = particles.y[other_idx];
                    float other_vx = particles.vx[other_idx];
                    float other_vy = particles.vy[other_idx];
                    float other_radius = particles.radius[other_idx];
                    float other_mass = particles.mass[other_idx];

                    float dxp = other_x - orig_x;
                    float dyp = other_y - orig_y;
                    float dist_sq = dxp * dxp + dyp * dyp;
                    float min_dist = radius + other_radius;

                    if (dist_sq < min_dist * min_dist && dist_sq > 1e-12f) {
                        float dist = std::sqrt(dist_sq);
                        float nx = dxp / dist;
                        float ny = dyp / dist;

                        float dvx = orig_vx - other_vx;
                        float dvy = orig_vy - other_vy;
                        float dvn = dvx * nx + dvy * ny;

                        if (dvn <= 0) continue;

                        const float restitution = 0.95f;
                        float impulse = ((1.0f + restitution) * dvn) / (mass + other_mass);
                        imp_vx -= impulse * other_mass * nx;
                        imp_vy -= impulse * other_mass * ny;

                        const float position_correction_strength = 0.5f;
                        float overlap = min_dist - dist;
                        corr_x -= overlap * 0.5f * position_correction_strength * nx;
                        corr_y -= overlap * 0.5f * position_correction_strength * ny;
                        collision_count++;
                    }
                }
            }
        }

        if (collision_count > 1) {
            float inv_count = 1.0f / (float)collision_count;
            corr_x *= inv_count;
            corr_y *= inv_count;
            imp_vx *= inv_count;
            imp_vy *= inv_count;
        }

        const float max_dv = 0.5f;
        imp_vx = clampf(imp_vx, -max_dv, max_dv);
        imp_vy = clampf(imp_vy, -max_dv, max_dv);

        const float max_corr = 0.05f;
        corr_x = clampf(corr_x, -max_corr, max_corr);
        corr_y = clampf(corr_y, -max_corr, max_corr);

        float x = orig_x + corr_x;
        float y = orig_y + corr_y;
        float vx = orig_vx + imp_vx;
        float vy = orig_vy + imp_vy;

        float gravity = 0.15f;
        vy += -gravity * dt;

        x += vx * dt;
        y += vy * dt;

        if (x - radius < -1.0f) {
            x = -1.0f + radius;
            vx = std::fabs(vx);
        }
        if (x + radius > 1.0f) {
            x = 1.0f - radius;
            vx = -std::fabs(vx);
        }
        if (y - radius < -1.0f) {
            y = -1.0f + radius;
            vy = std::fabs(vy);
        }
        if (y + radius > 1.0f) {
            y = 1.0f - radius;
            vy = -std::fabs(vy);
        }

        particles.x[idx] = x;
        particles.y[idx] = y;
        particles.vx[idx] = vx;
        particles.vy[idx] = vy;
    }
}

void step_simulation(Particles& particles, UniformGrid& grid, unsigned int num_particles, float dt) {
    build_uniform_grid(particles, grid, num_particles);
    integrate_collisions(particles, grid, num_particles, dt);
}

