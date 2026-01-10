#ifndef PARTICLE_H
#define PARTICLE_H

#include <cuda_runtime.h>

// Structure of Arrays for coalesced memory access
struct Particles {
    float* x;        // Position X array
    float* y;        // Position Y array
    float* vx;       // Velocity X array
    float* vy;       // Velocity Y array
    float* radius;   // Radius array
    float* mass;     // Mass array
};

// Simulation constants
#define NUM_PARTICLES 100000
#define MIN_RADIUS 0.001f
#define MAX_RADIUS 0.002f
#define MAX_VELOCITY 0.1f

inline void allocate_particles(Particles* p, unsigned int num) {
    cudaMalloc(&p->x, num * sizeof(float));
    cudaMalloc(&p->y, num * sizeof(float));
    cudaMalloc(&p->vx, num * sizeof(float));
    cudaMalloc(&p->vy, num * sizeof(float));
    cudaMalloc(&p->radius, num * sizeof(float));
    cudaMalloc(&p->mass, num * sizeof(float));
}

inline void free_particles(Particles* p) {
    cudaFree(p->x);
    cudaFree(p->y);
    cudaFree(p->vx);
    cudaFree(p->vy);
    cudaFree(p->radius);
    cudaFree(p->mass);
}

#endif
