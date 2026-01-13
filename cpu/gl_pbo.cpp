#include "gl_pbo.h"

#include <cstdlib>
#include <iostream>

GLPboTexture::GLPboTexture(unsigned int width, unsigned int height)
    : m_width(width), m_height(height), m_pbo(0), m_texture(0), m_mappedPtr(nullptr) {

    glGenBuffers(1, &m_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(Rgba32f), nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei)width, (GLsizei)height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

GLPboTexture::~GLPboTexture() {
    if (m_mappedPtr) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        m_mappedPtr = nullptr;
    }
    if (m_pbo) glDeleteBuffers(1, &m_pbo);
    if (m_texture) glDeleteTextures(1, &m_texture);
}

Rgba32f* GLPboTexture::mapBuffer() {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    m_mappedPtr = glMapBufferRange(
        GL_PIXEL_UNPACK_BUFFER,
        0,
        (GLsizeiptr)(m_width * m_height * sizeof(Rgba32f)),
        GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT
    );

    if (!m_mappedPtr) {
        std::cerr << "Failed to map PBO buffer" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return reinterpret_cast<Rgba32f*>(m_mappedPtr);
}

void GLPboTexture::unmapBuffer() {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    m_mappedPtr = nullptr;
}

void GLPboTexture::updateTexture() {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)m_width, (GLsizei)m_height, GL_RGBA, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void GLPboTexture::resize(unsigned int newWidth, unsigned int newHeight) {
    if (newWidth == m_width && newHeight == m_height) return;
    if (newWidth == 0 || newHeight == 0) return;

    if (m_mappedPtr) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        m_mappedPtr = nullptr;
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, newWidth * newHeight * sizeof(Rgba32f), nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei)newWidth, (GLsizei)newHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    m_width = newWidth;
    m_height = newHeight;
}

