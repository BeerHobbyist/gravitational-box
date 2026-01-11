#include "cuda_kernel.cuh"
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <ctime>
#include <thrust/sort.h>

// Query optimal launch configuration based on device properties
struct LaunchConfig {
    int blockSize1D;      // For 1D kernels (particles)
    dim3 blockSize2D;     // For 2D kernels (screen)
    int maxThreadsPerBlock;
    int warpSize;
};

LaunchConfig get_optimal_config() {
    static LaunchConfig config = {0, dim3(0,0), 0, 0};
    
    if (config.maxThreadsPerBlock == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        config.maxThreadsPerBlock = prop.maxThreadsPerBlock;
        config.warpSize = prop.warpSize;
        
        config.blockSize1D = (prop.maxThreadsPerBlock / prop.warpSize) * prop.warpSize;
        
        int block2D = 16;
        while (block2D * block2D > prop.maxThreadsPerBlock) {
            block2D /= 2;
        }
        config.blockSize2D = dim3(block2D, block2D);
        
        printf("Optimal config: 1D blocks=%d, 2D blocks=(%d,%d), warpSize=%d\n",
               config.blockSize1D, block2D, block2D, config.warpSize);
    }
    
    return config;
}

__global__ void init_particles_kernel(Particles particles, unsigned int num_particles, unsigned long seed) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    // Sunflower/Fibonacci spiral for uniform circle distribution
    const float golden_angle = 2.39996322f;
    const float circle_radius = 0.8f; // Leave margin from edges
    
    float angle = idx * golden_angle;
    float r_norm = (num_particles > 1) ? sqrtf((float)idx / (float)(num_particles - 1)) : 0.0f;
    float dist = r_norm * circle_radius;
    
    particles.x[idx] = dist * cosf(angle);
    particles.y[idx] = dist * sinf(angle);
    
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    particles.vx[idx] = (curand_uniform(&state) * 2.0f - 1.0f) * MAX_VELOCITY;
    particles.vy[idx] = (curand_uniform(&state) * 2.0f - 1.0f) * MAX_VELOCITY;
    
    float particle_radius = MIN_RADIUS + curand_uniform(&state) * (MAX_RADIUS - MIN_RADIUS);
    particles.radius[idx] = particle_radius;
    particles.mass[idx] = particle_radius * particle_radius; // Mass proportional to area
}

void init_particles(Particles* d_particles, unsigned int num_particles) {
    LaunchConfig config = get_optimal_config();
    
    int blockSize = config.blockSize1D;
    int gridSize = (num_particles + blockSize - 1) / blockSize;
    
    unsigned long seed = (unsigned long)time(NULL);
    init_particles_kernel<<<gridSize, blockSize>>>(*d_particles, num_particles, seed);
    
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
    
    int cell_x = (int)((x + 1.0f) / grid.cell_size);
    int cell_y = (int)((y + 1.0f) / grid.cell_size);
    
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
    
    // Read original state (used for all collision calculations)
    const float orig_x = particles.x[idx];
    const float orig_y = particles.y[idx];
    const float orig_vx = particles.vx[idx];
    const float orig_vy = particles.vy[idx];
    float radius = particles.radius[idx];
    float mass = particles.mass[idx];
    
    // Accumulators for collision response (applied after all collisions detected)
    float corr_x = 0.0f;
    float corr_y = 0.0f;
    float imp_vx = 0.0f;
    float imp_vy = 0.0f;
    int collision_count = 0;

    int cell_x = (int)((orig_x + 1.0f) / grid.cell_size);
    int cell_y = (int)((orig_y + 1.0f) / grid.cell_size);
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
                if (other_idx == idx) continue;  // Skip self
                
                float other_x = particles.x[other_idx];
                float other_y = particles.y[other_idx];
                float other_vx = particles.vx[other_idx];
                float other_vy = particles.vy[other_idx];
                float other_radius = particles.radius[other_idx];
                float other_mass = particles.mass[other_idx];
                
                float dx = other_x - orig_x;
                float dy = other_y - orig_y;
                float dist_sq = dx * dx + dy * dy;
                float min_dist = radius + other_radius;
                
                if (dist_sq < min_dist * min_dist && dist_sq > 1e-12f) {  // Avoid div by zero
                    float dist = sqrtf(dist_sq);
                    
                    float nx = dx / dist;
                    float ny = dy / dist;
                    
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
    
    if (x - radius < -1.0f) {
        x = -1.0f + radius;
        vx = fabsf(vx);
    }
    if (x + radius > 1.0f) {
        x = 1.0f - radius;
        vx = -fabsf(vx);
    }
    if (y - radius < -1.0f) {
        y = -1.0f + radius;
        vy = fabsf(vy);
    }
    if (y + radius > 1.0f) {
        y = 1.0f - radius;
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

__global__ void render_particles_kernel(Particles particles, unsigned int num_particles,
                                       float4* output, unsigned int width, unsigned int height,
                                       unsigned int ref_width) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    float px = particles.x[idx];
    float py = particles.y[idx];
    float radius = particles.radius[idx];
    
    float3 color = make_float3(1.0f, 1.0f, 1.0f);
    
    // Convert world position to screen coordinates (with proper rounding)
    int center_x = (int)((px + 1.0f) * 0.5f * width + 0.5f);
    int center_y = (int)((py + 1.0f) * 0.5f * height + 0.5f);
    
    // Use reference width for radius so particle pixel size stays constant across resizes
    float pixel_radius_f = radius * ref_width * 0.5f;
    int pixel_radius = (int)(pixel_radius_f + 0.5f);
    if (pixel_radius < 1) pixel_radius = 1;
    
    for (int dy = -pixel_radius; dy <= pixel_radius; dy++) {
        for (int dx = -pixel_radius; dx <= pixel_radius; dx++) {
            int px_coord = center_x + dx;
            int py_coord = center_y + dy;
            
            if (px_coord < 0 || px_coord >= width || py_coord < 0 || py_coord >= height) 
                continue;
            
            if (dx * dx + dy * dy < pixel_radius * pixel_radius) {
                output[py_coord * width + px_coord] = make_float4(color.x, color.y, color.z, 1.0f);
            }
        }
    }
}

void update_and_render(Particles* d_particles, UniformGrid* d_grid, unsigned int num_particles,
                       float4* d_output, unsigned int width, unsigned int height,
                       unsigned int ref_width, float dt) {
    
    LaunchConfig config = get_optimal_config();
    
    {
        int blockSize = config.blockSize1D;
        int gridSize = (d_grid->num_cells + blockSize - 1) / blockSize;
        reset_grid_kernel<<<gridSize, blockSize>>>(*d_grid);
        
        gridSize = (num_particles + blockSize - 1) / blockSize;
        assign_particles_to_cells_kernel<<<gridSize, blockSize>>>(*d_particles, *d_grid, num_particles);
        

        thrust::sort_by_key(thrust::device, d_grid->particle_cell, d_grid->particle_cell + num_particles, d_grid->particle_indices);

        
        compute_cell_bounds_kernel<<<gridSize, blockSize>>>(*d_grid, num_particles);
    }
    
    {
        int blockSize = config.blockSize1D;
        int gridSize = (num_particles + blockSize - 1) / blockSize;
        update_particles_kernel<<<gridSize, blockSize>>>(*d_particles, *d_grid, num_particles, dt);
    }
    
    {
        dim3 blockSize = config.blockSize2D;
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        clear_screen_kernel<<<gridSize, blockSize>>>(d_output, width, height);
    }
    
    {
        int blockSize = config.blockSize1D;
        int gridSize = (num_particles + blockSize - 1) / blockSize;
        render_particles_kernel<<<gridSize, blockSize>>>(*d_particles, num_particles,
                                                         d_output, width, height, ref_width);
    }
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(error));
    }
}
