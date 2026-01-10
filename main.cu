#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <iostream>

#include "gl_interop.h"
#include "cuda_kernel.cuh"
#include "particle.h"
#include "grid.h"

// Window configuration
const unsigned int WINDOW_WIDTH = 1440;
const unsigned int WINDOW_HEIGHT = 1024;
const char* WINDOW_TITLE = "Gravitational Box";

// Global state
GLCudaInterop* g_interop = nullptr;
Particles d_particles;
UniformGrid d_grid;

void initCuda() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "CUDA Device: " << deviceProp.name << std::endl;
    std::cout << "Particles: " << NUM_PARTICLES << std::endl;
    
    cudaSetDevice(0);
    
    // Allocate particle arrays on device (SoA for coalesced access)
    allocate_particles(&d_particles, NUM_PARTICLES);
    
    // Allocate uniform grid for spatial partitioning
    allocate_grid(&d_grid, NUM_PARTICLES);
    std::cout << "Grid: " << d_grid.grid_width << "×" << d_grid.grid_height 
              << " cells (cell_size=" << d_grid.cell_size << ")" << std::endl;
    
    // Initialize particles
    init_particles(&d_particles, NUM_PARTICLES);
    cudaDeviceSynchronize();
    
    std::cout << "Particles initialized (SoA layout)\n" << std::endl;
}

void renderFrame(float dt) {
    float4* d_output = g_interop->mapBuffer();
    
    update_and_render(&d_particles, &d_grid, NUM_PARTICLES, d_output,
                     g_interop->getWidth(), g_interop->getHeight(), dt);
    
    g_interop->unmapBuffer();
    g_interop->updateTexture();
    
    // Render fullscreen quad
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_interop->getTextureID());
    
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
    glEnd();
    
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

void cleanup() {
    free_particles(&d_particles);
    free_grid(&d_grid);
    
    if (g_interop) {
        delete g_interop;
        g_interop = nullptr;
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    glfwSetKeyCallback(window, keyCallback);
    
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }
    
    initCuda();
    
    g_interop = new GLCudaInterop(WINDOW_WIDTH, WINDOW_HEIGHT);
    
    std::cout << "Running simulation...\n" << std::endl;
    
    double lastTime = glfwGetTime();
    double fpsTime = lastTime;
    int frameCount = 0;
    
    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        float dt = static_cast<float>(currentTime - lastTime);
        lastTime = currentTime;
        
        // Cap delta time to avoid large jumps
        if (dt > 0.05f) dt = 0.016f;
        
        renderFrame(dt);
        
        glfwSwapBuffers(window);
        glfwPollEvents();
        
        // FPS counter
        frameCount++;
        if (currentTime - fpsTime >= 1.0) {
            printf("FPS: %.1f\n", frameCount / (currentTime - fpsTime));
            frameCount = 0;
            fpsTime = currentTime;
        }
    }
    
    cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();
    
    return 0;
}
