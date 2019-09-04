#include "jpeg.h"
#include "utils.h"

#include <vector>
#include <iostream>

Jpeg::Jpeg()
{
    m_compressor = tjInitCompress();
    m_decompressor = tjInitDecompress();
}

Jpeg::~Jpeg()
{
    tjDestroy(m_decompressor);
    tjDestroy(m_compressor);
}

unsigned long Jpeg::bufSize(int w, int h)
{
    return tjBufSize(w, h, TJSAMP_444);
}

unsigned long Jpeg::compress(const Rgb32Image& image,
    unsigned char * jpegImage, int quality)
{
    return compress(image.data, image.width, image.height,
                        jpegImage, quality);
}

unsigned long Jpeg::compress(unsigned char * image, int width, int height,
    unsigned char * jpegImage, int quality)
{
    unsigned long jpegSize = bufSize(width, height);
    tjCompress2(m_compressor, image, width, 0, height, TJPF_RGBA,
        &jpegImage, &jpegSize, TJSAMP_444, quality,
        TJFLAG_FASTDCT | TJFLAG_NOREALLOC);
    return jpegSize;
}

Rgb32Image Jpeg::decompress(unsigned char * jpegImage, unsigned long jpegSize)
{
    int jpegSubsamp, width, height, jpegColorspace;

    tjDecompressHeader3(m_decompressor, jpegImage, jpegSize,
        &width, &height, &jpegSubsamp, &jpegColorspace);

    Rgb32Image img(width, height);

    tjDecompress2(m_decompressor, jpegImage, jpegSize, img.data, width, 0, height,
        TJPF_RGBA, TJFLAG_FASTDCT);

    return img;
}

Rgb32Image Jpeg::load(const std::string& filename)
{
    auto jpegImage = read_binary_file(filename);
    return decompress(jpegImage.data(), jpegImage.size());
}

void Jpeg::save(const Rgb32Image& img, const std::string& filename)
{
    unsigned long bSize = bufSize(img.width, img.height);
    std::vector<unsigned char> buf(bSize);
    unsigned long size = compress(img, buf.data(), 75);
    std::cout << "Jpeg size = " << size << "\n";
    buf.resize(size);
    write_binary_file(buf, filename);
}

    /*
See https://stackoverflow.com/questions/9094691/examples-or-tutorials-of-using-libjpeg-turbos-turbojpeg

Compress:
--------

#include <turbojpeg.h>

const int JPEG_QUALITY = 75;
const int COLOR_COMPONENTS = 3;
int _width = 1920;
int _height = 1080;
long unsigned int _jpegSize = 0;
unsigned char* _compressedImage = NULL; //!< Memory is allocated by tjCompress2 if _jpegSize == 0
unsigned char buffer[_width*_height*COLOR_COMPONENTS]; //!< Contains the uncompressed image

tjhandle _jpegCompressor = tjInitCompress();

tjCompress2(_jpegCompressor, buffer, _width, 0, _height, TJPF_RGB,
          &_compressedImage, &_jpegSize, TJSAMP_444, JPEG_QUALITY,
          TJFLAG_FASTDCT);

tjDestroy(_jpegCompressor);

//to free the memory allocated by TurboJPEG (either by tjAlloc(), 
//or by the Compress/Decompress) after you are done working on it:
tjFree(&_compressedImage);

Decompress:
-----------
#include <turbojpeg.h>

long unsigned int _jpegSize; //!< _jpegSize from above
unsigned char* _compressedImage; //!< _compressedImage from above

int jpegSubsamp, width, height;
unsigned char buffer[width*height*COLOR_COMPONENTS]; //!< will contain the decompressed image

tjhandle _jpegDecompressor = tjInitDecompress();

tjDecompressHeader2(_jpegDecompressor, _compressedImage, _jpegSize, &width, &height, &jpegSubsamp);

tjDecompress2(_jpegDecompressor, _compressedImage, _jpegSize, buffer, width, 0, height, TJPF_RGB, TJFLAG_FASTDCT);

tjDestroy(_jpegDecompressor);
     */
