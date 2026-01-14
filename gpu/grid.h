#ifndef GRID_H
#define GRID_H

#include <cuda_runtime.h>
#include "particle.h"

// Uniform grid for spatial partitioning
// Used for efficient particle-particle collision detection
struct UniformGrid {
    // Grid dimensions
    int grid_width;       // Number of cells in X
    int grid_height;      // Number of cells in Y
    int num_cells;        // Total cells (grid_width × grid_height)
    float cell_size;      // Size of each cell in world space

    // World bounds (physics space)
    float world_min_x;    // Left bound (typically -aspect_ratio)
    float world_max_x;    // Right bound (typically +aspect_ratio)
    float world_min_y;    // Bottom bound (typically -1)
    float world_max_y;    // Top bound (typically +1)

    // Cell data (for each cell, store which particles are in it)
    int* cell_start;      // Starting index in particle_indices for each cell
    int* cell_count;      // Number of particles in each cell

    // Particle to cell mapping
    int* particle_cell;   // Which cell each particle belongs to
    int* particle_indices;// Sorted array of particle indices by cell
};

#define MAX_ASPECT_RATIO 3.0f

// aspect_ratio = window_width / window_height
inline void allocate_grid(UniformGrid* grid, unsigned int num_particles, float aspect_ratio) {
    // Cell size must be at least 2 × MAX_RADIUS so that 3×3 neighborhood
    // check catches all potential collisions between particles
    grid->cell_size = 2.0f * MAX_RADIUS;

    // Physics space: X in [-aspect_ratio, aspect_ratio], Y in [-1, 1]
    grid->world_min_x = -aspect_ratio;
    grid->world_max_x = aspect_ratio;
    grid->world_min_y = -1.0f;
    grid->world_max_y = 1.0f;

    float max_extent = 2.0f * MAX_ASPECT_RATIO;

    grid->grid_width = (int)(max_extent / grid->cell_size) + 1;
    grid->grid_height = (int)(max_extent / grid->cell_size) + 1;
    grid->num_cells = grid->grid_width * grid->grid_height;

    cudaMalloc(&grid->cell_start, grid->num_cells * sizeof(int));
    cudaMalloc(&grid->cell_count, grid->num_cells * sizeof(int));

    cudaMalloc(&grid->particle_cell, num_particles * sizeof(int));
    cudaMalloc(&grid->particle_indices, num_particles * sizeof(int));
}

// Update world bounds when window is resized (no reallocation needed)
inline void update_grid_aspect_ratio(UniformGrid* grid, float aspect_ratio) {
    if (aspect_ratio >= 1.0f) {
        // Landscape: X extends, Y is [-1, 1]
        grid->world_min_x = -aspect_ratio;
        grid->world_max_x = aspect_ratio;
        grid->world_min_y = -1.0f;
        grid->world_max_y = 1.0f;
    } else {
        // Portrait: X is [-1, 1], Y extends
        grid->world_min_x = -1.0f;
        grid->world_max_x = 1.0f;
        grid->world_min_y = -1.0f / aspect_ratio;
        grid->world_max_y = 1.0f / aspect_ratio;
    }
}

inline void free_grid(UniformGrid* grid) {
    cudaFree(grid->cell_start);
    cudaFree(grid->cell_count);
    cudaFree(grid->particle_cell);
    cudaFree(grid->particle_indices);
}

#endif // GRID_H

