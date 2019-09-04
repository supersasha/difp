#pragma once

#include <string>
#include <string.h>
#include "rgb32_image.h"

struct Color
{
    float c[4];

    Color() {}
    Color(float r, float g, float b)
    {
        c[0] = r;
        c[1] = g;
        c[2] = b;
    }
};

struct Image
{
    int width = 0;
    int height = 0;

    Color * data = nullptr;

    Image() {}

    Image(int w, int h)
        : width(w), height(h)
    {
        data = new Color[width * h];
    }

    Image(const Image&) = delete;
    Image(Image&& rhs)
    {
        *this = std::move(rhs);
    }

    Image& operator=(const Image&) = delete;
    Image& operator=(Image&& rhs)
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

    Image clone()
    {
        Image img(width, height);
        memcpy(img.data, data, width*height*sizeof(Color));
        return img;
    }

    ~Image()
    {
        if (data) {
            delete[] data;
        }
    }
};

Image load_image_from_raw_file(const std::string& filename);
Rgb32Image convert_image_to_rgb32(const Image& img);
Image bilinear_scale(const Image&, int fit_width, int fit_height);

Image sub_image(const Image&, int left, int top, int width, int height);