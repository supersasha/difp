#pragma once

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

Color operator*(const Color& c, float);
Color operator*(float, const Color& c);
Color operator+(const Color& c1, const Color& c2);
Color operator-(const Color& c1, const Color& c2);
Color operator-(const Color& c);

Color spectrum_to_xyz(const float * spectrum);
Color xyz_to_srgb(const Color& c);
