#ifndef GRID_H
#define GRID_H

#include <vector>
#include "particle.h"

struct UniformGrid {
    int grid_width;
    int grid_height;
    int num_cells;
    float cell_size;

    std::vector<int> cell_start;
    std::vector<int> cell_count;

    std::vector<int> particle_cell;
    std::vector<int> particle_indices;
};

inline void allocate_grid(UniformGrid* grid, unsigned int num_particles) {
    grid->cell_size = 2.0f * MAX_RADIUS;

    grid->grid_width = (int)(2.0f / grid->cell_size) + 1;
    grid->grid_height = (int)(2.0f / grid->cell_size) + 1;
    grid->num_cells = grid->grid_width * grid->grid_height;

    grid->cell_start.resize(grid->num_cells);
    grid->cell_count.resize(grid->num_cells);
    grid->particle_cell.resize(num_particles);
    grid->particle_indices.resize(num_particles);
}

inline void free_grid(UniformGrid* grid) {
    grid->cell_start.clear();
    grid->cell_count.clear();
    grid->particle_cell.clear();
    grid->particle_indices.clear();
    grid->cell_start.shrink_to_fit();
    grid->cell_count.shrink_to_fit();
    grid->particle_cell.shrink_to_fit();
    grid->particle_indices.shrink_to_fit();
}

#endif

