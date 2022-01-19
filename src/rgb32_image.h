#pragma once

#include <utility>

struct Rgb32Image
{
    int width = 0;
    int height = 0;
    unsigned char * data = nullptr;

    Rgb32Image() {}

    Rgb32Image(int w, int h)
        : width(w), height(h)
    {
        data = new unsigned char[w*h*4];
    }

    Rgb32Image(const Rgb32Image&) = delete;
    Rgb32Image(Rgb32Image&& rhs)
    {
        *this = std::move(rhs);
    }

    Rgb32Image& operator=(const Rgb32Image&) = delete;
    Rgb32Image& operator=(Rgb32Image&& rhs)
    {
        if (data) {
            delete[] data;
        }
        this->width = rhs.width;
        this->height = rhs.height;
        this->data = rhs.data;
        rhs.width = 0;
        rhs.height = 0;
        rhs.data = nullptr;
	return *this;
    }

    void resize(int w, int h)
    {
        width = w;
        height = h;
        
        if (data) {
            delete[] data;
        }
        data = new unsigned char[w*h*4];
    }

    ~Rgb32Image()
    {
        if (data) {
            delete[] data;
        }
    }
};
