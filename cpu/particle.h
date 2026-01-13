#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>

struct Particles {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> vx;
    std::vector<float> vy;
    std::vector<float> radius;
    std::vector<float> mass;
};

#define NUM_PARTICLES 120000
#define MIN_RADIUS 0.0028f
#define MAX_RADIUS 0.003f
#define MAX_VELOCITY 0.5f

inline void allocate_particles(Particles* p, unsigned int num) {
    p->x.resize(num);
    p->y.resize(num);
    p->vx.resize(num);
    p->vy.resize(num);
    p->radius.resize(num);
    p->mass.resize(num);
}

inline void free_particles(Particles* p) {
    p->x.clear();
    p->y.clear();
    p->vx.clear();
    p->vy.clear();
    p->radius.clear();
    p->mass.clear();
    p->x.shrink_to_fit();
    p->y.shrink_to_fit();
    p->vx.shrink_to_fit();
    p->vy.shrink_to_fit();
    p->radius.shrink_to_fit();
    p->mass.shrink_to_fit();
}

#endif

