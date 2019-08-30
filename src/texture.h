#pragma once

#include <GL/gl.h>
#include "rgb32_image.h"

class Texture
{
public:
    ~Texture() { unload(); }

    void load(const Rgb32Image&);
    void unload();

    bool isLoaded() { return m_id > 0; }

    GLuint id() const { return m_id; }

    int width() const { return m_width; }
    int height() const { return m_height; }
private:
    GLuint m_id = 0;
    int m_width = 0;
    int m_height = 0;
};
