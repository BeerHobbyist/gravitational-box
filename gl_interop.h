#ifndef GL_INTEROP_H
#define GL_INTEROP_H

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// ============================================================================
// OPENGL-CUDA INTEROP: Handles all the boilerplate
// ============================================================================

class GLCudaInterop {
public:
    GLCudaInterop(unsigned int width, unsigned int height);
    ~GLCudaInterop();
    
    // Map the OpenGL buffer for CUDA access
    float4* mapBuffer();
    
    // Unmap the buffer after CUDA processing
    void unmapBuffer();
    
    // Get the texture ID for rendering
    GLuint getTextureID() const { return m_texture; }
    
    // Update texture from PBO (call after unmapping)
    void updateTexture();
    
    // Get dimensions
    unsigned int getWidth() const { return m_width; }
    unsigned int getHeight() const { return m_height; }
    
    // Resize buffers for new window dimensions
    void resize(unsigned int newWidth, unsigned int newHeight);

private:
    unsigned int m_width;
    unsigned int m_height;
    
    GLuint m_pbo;      // Pixel Buffer Object
    GLuint m_texture;  // Texture for rendering
    
    cudaGraphicsResource* m_cudaResource;
    
    void checkCudaError(cudaError_t error, const char* msg);
};

#endif // GL_INTEROP_H
