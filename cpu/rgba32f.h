#ifndef CPU_RGBA32F_H
#define CPU_RGBA32F_H

struct Rgba32f {
    float r, g, b, a;
};

constexpr Rgba32f rgba32f(float r, float g, float b, float a) {
    return Rgba32f{r, g, b, a};
}

#endif

