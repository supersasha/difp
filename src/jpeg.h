#pragma once

#include <string>
#include <turbojpeg.h>
#include "rgb32_image.h"

class Jpeg
{
public:
    Jpeg();
    ~Jpeg();

    Rgb32Image load(const std::string& filename);

    unsigned long bufSize(int w, int h);
    unsigned long compress(const Rgb32Image&,
        unsigned char * jpegImage, int quality = 100);
    unsigned long compress(unsigned char * image, int w, int h,
        unsigned char * jpegImage, int quality = 100);
    Rgb32Image decompress(unsigned char *jpegImage, unsigned long jpegSize);
private:
    tjhandle m_compressor;
    tjhandle m_decompressor;
};

