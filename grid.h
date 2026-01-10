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
    
    // Cell data (for each cell, store which particles are in it)
    int* cell_start;      // Starting index in particle_indices for each cell
    int* cell_count;      // Number of particles in each cell
    
    // Particle to cell mapping
    int* particle_cell;   // Which cell each particle belongs to
    int* particle_indices;// Sorted array of particle indices by cell
};

// Helper functions to allocate/free grid on device
inline void allocate_grid(UniformGrid* grid, unsigned int num_particles) {
    // Cell size must be at least 2 × MAX_RADIUS so that 3×3 neighborhood
    // check catches all potential collisions between particles
    grid->cell_size = 2.0f * MAX_RADIUS;
    
    grid->grid_width = (int)(2.0f / grid->cell_size) + 1;
    grid->grid_height = (int)(2.0f / grid->cell_size) + 1;
    grid->num_cells = grid->grid_width * grid->grid_height;
    
    cudaMalloc(&grid->cell_start, grid->num_cells * sizeof(int));
    cudaMalloc(&grid->cell_count, grid->num_cells * sizeof(int));
    
    cudaMalloc(&grid->particle_cell, num_particles * sizeof(int));
    cudaMalloc(&grid->particle_indices, num_particles * sizeof(int));
}

inline void free_grid(UniformGrid* grid) {
    cudaFree(grid->cell_start);
    cudaFree(grid->cell_count);
    cudaFree(grid->particle_cell);
    cudaFree(grid->particle_indices);
}

#endif // GRID_H
