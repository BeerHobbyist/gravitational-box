#include "cuda_kernel.cuh"
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <thrust/sort.h>


__global__ void init_particles_kernel(Particles particles, unsigned int num_particles, unsigned long seed, float aspect_ratio) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Sunflower/Fibonacci spiral for uniform circle distribution
    const float golden_angle = 2.39996322f;
    const float circle_radius = 0.8f; // Leave margin from edges

    float angle = idx * golden_angle;
    float r_norm = (num_particles > 1) ? sqrtf((float)idx / (float)(num_particles - 1)) : 0.0f;
    float dist = r_norm * circle_radius;

    // Scale X by aspect ratio to fit in the wider physics space
    particles.x[idx] = dist * cosf(angle) * aspect_ratio;
    particles.y[idx] = dist * sinf(angle);

    curandState state;
    curand_init(seed, idx, 0, &state);

    particles.vx[idx] = (curand_uniform(&state) * 2.0f - 1.0f) * MAX_VELOCITY;
    particles.vy[idx] = (curand_uniform(&state) * 2.0f - 1.0f) * MAX_VELOCITY;

    float particle_radius = MIN_RADIUS + curand_uniform(&state) * (MAX_RADIUS - MIN_RADIUS);
    particles.radius[idx] = particle_radius;
    particles.mass[idx] = particle_radius * particle_radius; // Mass proportional to area
}

void init_particles(Particles* d_particles, unsigned int num_particles, unsigned long seed, float aspect_ratio) {
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, init_particles_kernel, 0, 0);
    int gridSize = (num_particles + blockSize - 1) / blockSize;

    init_particles_kernel<<<gridSize, blockSize>>>(*d_particles, num_particles, seed, aspect_ratio);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Particle init error: %s\n", cudaGetErrorString(error));
    }
}

__global__ void assign_particles_to_cells_kernel(Particles particles, UniformGrid grid,
                                                 unsigned int num_particles) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    float x = particles.x[idx];
    float y = particles.y[idx];

    // Use grid's world bounds for cell assignment
    int cell_x = (int)((x - grid.world_min_x) / grid.cell_size);
    int cell_y = (int)((y - grid.world_min_y) / grid.cell_size);

    cell_x = max(0, min(cell_x, grid.grid_width - 1));
    cell_y = max(0, min(cell_y, grid.grid_height - 1));

    int cell_id = cell_y * grid.grid_width + cell_x;

    grid.particle_cell[idx] = cell_id;
    grid.particle_indices[idx] = idx;
}

__global__ void reset_grid_kernel(UniformGrid grid) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= grid.num_cells) return;

    grid.cell_start[idx] = -1;
    grid.cell_count[idx] = 0;
}

__global__ void compute_cell_bounds_kernel(UniformGrid grid, unsigned int num_particles) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    int cell_id = grid.particle_cell[idx];

    // Check if this is the first particle in this cell
    if (idx == 0 || grid.particle_cell[idx - 1] != cell_id) {
        grid.cell_start[cell_id] = idx;
    }

    atomicAdd(&grid.cell_count[cell_id], 1);
}

__global__ void update_particles_kernel(Particles particles, UniformGrid grid,
                                        unsigned int num_particles, float dt) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

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

    // Use grid's world bounds for cell assignment
    int cell_x = (int)((orig_x - grid.world_min_x) / grid.cell_size);
    int cell_y = (int)((orig_y - grid.world_min_y) / grid.cell_size);
    cell_x = max(0, min(cell_x, grid.grid_width - 1));
    cell_y = max(0, min(cell_y, grid.grid_height - 1));

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
                if (other_idx == (int)idx) continue;  // Skip self

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

                if (dist_sq < min_dist * min_dist && dist_sq > 1e-12f) {  // Avoid div by zero
                    float dist = sqrtf(dist_sq);

                    float nx = dxp / dist;
                    float ny = dyp / dist;

                    float dvx = orig_vx - other_vx;
                    float dvy = orig_vy - other_vy;
                    float dvn = dvx * nx + dvy * ny;

                    // Only collide if particles are approaching
                    if (dvn <= 0) continue;

                    // Slightly inelastic collision to prevent energy buildup
                    const float restitution = 0.95f;
                    float impulse = ((1.0f + restitution) * dvn) / (mass + other_mass);
                    imp_vx -= impulse * other_mass * nx;
                    imp_vy -= impulse * other_mass * ny;

                    // Accumulate position correction to separate overlapping particles.
                    // Using a "soft" correction (<1) to avoid big dense clusters exploding.
                    const float position_correction_strength = 0.5f;
                    float overlap = min_dist - dist;
                    corr_x -= overlap * 0.5f * position_correction_strength * nx;
                    corr_y -= overlap * 0.5f * position_correction_strength * ny;
                    collision_count++;
                }
            }
        }
    }

    // Apply accumulated collision response, averaged if multiple collisions
    // This prevents energy explosion when many particles overlap
    if (collision_count > 1) {
        float inv_count = 1.0f / collision_count;
        corr_x *= inv_count;
        corr_y *= inv_count;
        imp_vx *= inv_count;
        imp_vy *= inv_count;
    }

    // Clamp maximum velocity change to prevent explosions
    const float max_dv = 0.5f;
    imp_vx = fmaxf(-max_dv, fminf(max_dv, imp_vx));
    imp_vy = fmaxf(-max_dv, fminf(max_dv, imp_vy));

    // Clamp maximum position correction
    const float max_corr = 0.05f;
    corr_x = fmaxf(-max_corr, fminf(max_corr, corr_x));
    corr_y = fmaxf(-max_corr, fminf(max_corr, corr_y));

    float x = orig_x + corr_x;
    float y = orig_y + corr_y;
    float vx = orig_vx + imp_vx;
    float vy = orig_vy + imp_vy;

    float gravity = 0.15f;
    vy += -gravity * dt;

    x += vx * dt;
    y += vy * dt;

    // Wall collisions using grid's world bounds
    if (x - radius < grid.world_min_x) {
        x = grid.world_min_x + radius;
        vx = fabsf(vx);
    }
    if (x + radius > grid.world_max_x) {
        x = grid.world_max_x - radius;
        vx = -fabsf(vx);
    }
    if (y - radius < grid.world_min_y) {
        y = grid.world_min_y + radius;
        vy = fabsf(vy);
    }
    if (y + radius > grid.world_max_y) {
        y = grid.world_max_y - radius;
        vy = -fabsf(vy);
    }

    particles.x[idx] = x;
    particles.y[idx] = y;
    particles.vx[idx] = vx;
    particles.vy[idx] = vy;
}

__global__ void clear_screen_kernel(float4* output, unsigned int width, unsigned int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    output[y * width + x] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
}

__device__ void draw_circle(float4* output, unsigned int width, unsigned int height,
                            int center_x, int center_y, float pixel_radius_f, float3 color) {
    int pixel_radius = (int)(pixel_radius_f + 0.5f);
    if (pixel_radius < 1) pixel_radius = 1;

    for (int dy = -pixel_radius - 1; dy <= pixel_radius + 1; dy++) {
        for (int dx = -pixel_radius - 1; dx <= pixel_radius + 1; dx++) {
            int px_coord = center_x + dx;
            int py_coord = center_y + dy;

            if (px_coord < 0 || px_coord >= (int)width || py_coord < 0 || py_coord >= (int)height)
                continue;

            float dist = sqrtf((float)(dx * dx + dy * dy));
            float alpha = fminf(1.0f, fmaxf(0.0f, pixel_radius_f + 0.5f - dist));

            if (alpha > 0.0f) {
                float4 existing = output[py_coord * width + px_coord];
                float inv_alpha = 1.0f - alpha;
                output[py_coord * width + px_coord] = make_float4(
                    alpha * color.x + inv_alpha * existing.x,
                    alpha * color.y + inv_alpha * existing.y,
                    alpha * color.z + inv_alpha * existing.z,
                    1.0f);
            }
        }
    }
}

__global__ void render_particles_kernel(Particles particles, unsigned int num_particles,
                                        float4* output, unsigned int width, unsigned int height,
                                        unsigned int ref_height,
                                        float world_min_x, float world_max_x,
                                        float world_min_y, float world_max_y) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    float px = particles.x[idx];
    float py = particles.y[idx];
    float radius = particles.radius[idx];

    float world_width = world_max_x - world_min_x;
    float world_height = world_max_y - world_min_y;
    int center_x = (int)((px - world_min_x) / world_width * width + 0.5f);
    int center_y = (int)((py - world_min_y) / world_height * height + 0.5f);

    float pixel_radius_f = radius * ref_height * 0.5f;
    float3 color = make_float3(1.0f, 1.0f, 1.0f);

    draw_circle(output, width, height, center_x, center_y, pixel_radius_f, color);
}

void update_and_render(Particles* d_particles, UniformGrid* d_grid, unsigned int num_particles,
                       float4* d_output, unsigned int width, unsigned int height,
                       unsigned int ref_height, float dt) {

    int blockSize, minGridSize, gridSize;

    // Grid construction
    {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, reset_grid_kernel, 0, 0);
        gridSize = (d_grid->num_cells + blockSize - 1) / blockSize;
        reset_grid_kernel<<<gridSize, blockSize>>>(*d_grid);

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, assign_particles_to_cells_kernel, 0, 0);
        gridSize = (num_particles + blockSize - 1) / blockSize;
        assign_particles_to_cells_kernel<<<gridSize, blockSize>>>(*d_particles, *d_grid, num_particles);

        thrust::sort_by_key(thrust::device, d_grid->particle_cell, d_grid->particle_cell + num_particles, d_grid->particle_indices);

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, compute_cell_bounds_kernel, 0, 0);
        gridSize = (num_particles + blockSize - 1) / blockSize;
        compute_cell_bounds_kernel<<<gridSize, blockSize>>>(*d_grid, num_particles);
    }

    // Physics update
    {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, update_particles_kernel, 0, 0);
        gridSize = (num_particles + blockSize - 1) / blockSize;
        update_particles_kernel<<<gridSize, blockSize>>>(*d_particles, *d_grid, num_particles, dt);
    }

    // Clear screen (16x16 = 256 threads)
    {
        dim3 blockSize2D(16, 16);
        dim3 gridSize2D((width + 15) / 16, (height + 15) / 16);
        clear_screen_kernel<<<gridSize2D, blockSize2D>>>(d_output, width, height);
    }

    // Render particles
    {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, render_particles_kernel, 0, 0);
        gridSize = (num_particles + blockSize - 1) / blockSize;
        render_particles_kernel<<<gridSize, blockSize>>>(*d_particles, num_particles,
                                                         d_output, width, height, ref_height,
                                                         d_grid->world_min_x, d_grid->world_max_x,
                                                         d_grid->world_min_y, d_grid->world_max_y);
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(error));
    }
}

