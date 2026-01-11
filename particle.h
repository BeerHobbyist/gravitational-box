#ifndef PARTICLE_H
#define PARTICLE_H

#include <cuda_runtime.h>

struct Particles {
    float* x;        // Position X array
    float* y;        // Position Y array
    float* vx;       // Velocity X array
    float* vy;       // Velocity Y array
    float* radius;   // Radius array
    float* mass;     // Mass array
};

// Simulation constants
#define NUM_PARTICLES 120000
#define MIN_RADIUS 0.0028f
#define MAX_RADIUS 0.003f
#define MAX_VELOCITY 0.5f

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
