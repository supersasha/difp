#pragma once

#include "array_ops.h"

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

    Array<3> to_array() const
    {
        return Array<3> {{c[0], c[1], c[2]}};
    }
};

Color operator*(const Color& c, float);
Color operator*(float, const Color& c);
Color operator+(const Color& c1, const Color& c2);
Color operator-(const Color& c1, const Color& c2);
Color operator-(const Color& c);

Color spectrum_to_xyz(const float * spectrum);
Color xyz_to_srgb(const Color& c);
Color srgb_to_xyz(const Color& c);
Color xyz_to_lab(const Color& c);
double delta_E76_lab(const Color&, const Color&);
double delta_E76_xyz(const Color&, const Color&);

Array<31> daylight_spectrum(double temp);
