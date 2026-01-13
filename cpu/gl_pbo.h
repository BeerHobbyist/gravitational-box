#ifndef GL_PBO_H
#define GL_PBO_H

#include <GL/glew.h>
#include "rgba32f.h"

class GLPboTexture {
public:
    GLPboTexture(unsigned int width, unsigned int height);
    ~GLPboTexture();

    Rgba32f* mapBuffer();
    void unmapBuffer();

    GLuint getTextureID() const { return m_texture; }

    void updateTexture();

    unsigned int getWidth() const { return m_width; }
    unsigned int getHeight() const { return m_height; }

    void resize(unsigned int newWidth, unsigned int newHeight);

private:
    unsigned int m_width;
    unsigned int m_height;

    GLuint m_pbo;
    GLuint m_texture;
    void* m_mappedPtr;
};

#endif

