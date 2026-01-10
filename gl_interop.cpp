#include "gl_interop.h"
#include <iostream>
#include <cstdlib>

GLCudaInterop::GLCudaInterop(unsigned int width, unsigned int height)
    : m_width(width), m_height(height), m_pbo(0), m_texture(0), m_cudaResource(nullptr) {
    
    // Create Pixel Buffer Object
    glGenBuffers(1, &m_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
    // Register PBO with CUDA
    checkCudaError(
        cudaGraphicsGLRegisterBuffer(&m_cudaResource, m_pbo, cudaGraphicsMapFlagsWriteDiscard),
        "cudaGraphicsGLRegisterBuffer"
    );
    
    // Create texture
    glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

GLCudaInterop::~GLCudaInterop() {
    if (m_cudaResource) {
        cudaGraphicsUnregisterResource(m_cudaResource);
    }
    if (m_pbo) {
        glDeleteBuffers(1, &m_pbo);
    }
    if (m_texture) {
        glDeleteTextures(1, &m_texture);
    }
}

float4* GLCudaInterop::mapBuffer() {
    float4* d_ptr = nullptr;
    size_t num_bytes;
    
    checkCudaError(
        cudaGraphicsMapResources(1, &m_cudaResource, 0),
        "cudaGraphicsMapResources"
    );
    
    checkCudaError(
        cudaGraphicsResourceGetMappedPointer((void**)&d_ptr, &num_bytes, m_cudaResource),
        "cudaGraphicsResourceGetMappedPointer"
    );
    
    return d_ptr;
}

void GLCudaInterop::unmapBuffer() {
    checkCudaError(
        cudaGraphicsUnmapResources(1, &m_cudaResource, 0),
        "cudaGraphicsUnmapResources"
    );
}

void GLCudaInterop::updateTexture() {
    // Copy from PBO to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void GLCudaInterop::checkCudaError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}
