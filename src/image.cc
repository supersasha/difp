#include "image.h"

#include <libraw.h>
#include <iostream>
#include <math.h>

Image load_image_from_raw_file(const std::string& filename)
{
    LibRaw raw;
    raw.imgdata.params.output_color = 5;
    raw.imgdata.params.output_bps = 16;
    raw.imgdata.params.user_qual = 3;
    raw.imgdata.params.highlight = 0;
    raw.imgdata.params.no_auto_bright = 1;
    
    /*
    raw.imgdata.params.use_camera_wb = 0;
    raw.imgdata.params.use_camera_matrix = 0;
    */

    //raw.imgdata.params.auto_bright_thr = 0.0001;
    
    raw.imgdata.params.gamm[0] = 1.0; //1.0 / 2.4;
    raw.imgdata.params.gamm[1] = 1.0; //12.92;
    
    raw.open_file(filename.c_str());
    raw.unpack();
    raw.dcraw_process();
    auto * mi = raw.dcraw_make_mem_image();
    
    /*
    Image img = create_from_buf(mi->data,
        mi->width * mi->colors * (mi->bits >> 3),
        mi->width, mi->height, mi->bits, mi->colors);
    */
    Image img(mi->width, mi->height);

    int srcPixelSize = mi->colors * (mi->bits / 8);
    int srcRedOffset = 0;
    int srcGreenOffset = (mi->bits / 8);
    int srcBlueOffset = 2 * (mi->bits / 8);
    float q = 1.0 / ((1 << mi->bits) - 1);

    for (int row = 0; row < mi->height; row++) {
        int rowOrigin = row * mi->width;
        int srcRowOrigin = row * mi->width * mi->colors * (mi->bits / 8);
        int srcOffset = 0;
        for (int col = 0; col < mi->width; col++, srcOffset += 6) {
            unsigned short * srcPixel =
                (unsigned short *) (mi->data + srcRowOrigin + srcOffset);
            img.data[rowOrigin + col] = Color(
                q * srcPixel[0],
                q * srcPixel[1],
                q * srcPixel[2]
            );
        }
    }
    
    raw.dcraw_clear_mem(mi);
    return img;
}

Rgb32Image convert_image_to_rgb32(const Image& img)
{
    Rgb32Image img32(img.width, img.height);
    for (int row = 0; row < img.height; row++) {
        int srcRowOrigin = row * img.width;
        int destRowOrigin = row * img.width * 4;
        int destOffset = 0;
        for (int col = 0; col < img.width; col++, destOffset += 4) {
            img32.data[destRowOrigin + destOffset] =
                255 * img.data[srcRowOrigin + col].c[0];
            img32.data[destRowOrigin + destOffset + 1] =
                255 * img.data[srcRowOrigin + col].c[1];
            img32.data[destRowOrigin + destOffset + 2] =
                255 * img.data[srcRowOrigin + col].c[2];
            img32.data[destRowOrigin + destOffset + 3] = 255;
        }
    }
    return img32;
}

Image bilinear_scale(const Image& src_img, int fit_width, int fit_height)
{
    std::size_t new_width = 0;
    std::size_t new_height = 0;
    if (fit_height == 0) {
        new_width = fit_width;
        new_height = src_img.height * fit_width / src_img.width;
    } else if (fit_width == 0) {
        new_height = fit_height;
        new_width = src_img.width * fit_height / src_img.height;
    } else {
        if (fit_width * src_img.height < fit_height * src_img.width) {
            new_width = fit_width;
            new_height = src_img.height * fit_width / src_img.width;
        } else {
            new_height = fit_height;
            new_width = src_img.width * fit_height / src_img.height;
        }
    }
    
    std::cout << "width: " << src_img.width << "\n";
    std::cout << "height: " << src_img.height << "\n";

    std::cout << "new width: " << new_width << "\n";
    std::cout << "new height: " << new_height << "\n";

    Image img(new_width, new_height);

    // bilinear interpolation

    /*
    float w_step = float(m_width - 1) / (new_width - 1);
    float h_step = float(m_height - 1) / (new_height - 1);

    std::cout << "w_step = " << w_step << "\n";
    std::cout << "h_step = " << h_step << "\n";

    int ggg = 0;
    */

    for (int h = 0; h < new_height; h++) {
        for (int w = 0; w < new_width; w++) {
            float x = w * (src_img.width - 1) / (new_width - 1);
            float y = h * (src_img.height - 1) / (new_height - 1);
            int x0 = x;
            int y0 = y;
            if (x0 == src_img.width - 1) {
                x0--;
            }
            if (y0 == src_img.height - 1) {
                y0--;
            }

            auto c00 = src_img.data[y0 * src_img.width + x0];
            auto c01 = src_img.data[(y0 + 1) * src_img.width + x0];
            auto c10 = src_img.data[y0 * src_img.width + (x0 + 1)];
            auto c11 = src_img.data[(y0 + 1) * src_img.width + (x0 + 1)];

            float q00 = (x0 - x + 1) * (y0 - y + 1);
            float q10 = (x - x0) * (y0 - y + 1);
            float q01 = (x0 - x + 1) * (y - y0);
            float q11 = (x - x0) * (y - y0);

            auto c = Color(
                c00.c[0]*q00 + c01.c[0]*q01 + c10.c[0]*q10 + c11.c[0]*q11,
                c00.c[1]*q00 + c01.c[1]*q01 + c10.c[1]*q10 + c11.c[1]*q11,
                c00.c[2]*q00 + c01.c[2]*q01 + c10.c[2]*q10 + c11.c[2]*q11
            );
            img.data[h * new_width + w] = c;
        }
    }
    
    /*
    for(float h = 0.0; h < m_height - 1 + 0.1; h += h_step) {
        if(h >= m_height - 1)
            h = m_height - 1 - 0.01;
        std::size_t ih = ((std::size_t)h) * m_width;
        float hh = std::floor(h);
        for(float w = 0.0; w < m_width - 1 + 0.1; w += w_step) {
            if(w >= m_width - 1)
                w = m_width - 1 - 0.01;
            std::size_t iw = (std::size_t)w;
            float ww = std::floor(w);
            auto c00 = (*m_data)[ih + iw];
            auto c01 = (*m_data)[ih + m_width + iw];
            auto c10 = (*m_data)[ih + iw + 1];
            auto c11 = (*m_data)[ih + m_width + iw + 1];

            float q00 = (ww + 1 - w) * (hh + 1 - h);
            float q10 = (w - ww) * (hh + 1 - h);
            float q01 = (ww + 1 - w) * (h - hh);
            float q11 = (w - ww) * (h - hh);

            img.m_data->emplace_back(ColorF(
                c00.c[0]*q00 + c01.c[0]*q01 + c10.c[0]*q10 + c11.c[0]*q11, 
                c00.c[1]*q00 + c01.c[1]*q01 + c10.c[1]*q10 + c11.c[1]*q11, 
                c00.c[2]*q00 + c01.c[2]*q01 + c10.c[2]*q10 + c11.c[2]*q11
            ));

            ggg++;
        }
    }
    */

    //std::cout << "scaled to " << ggg << "\n"; 
    return img;
}

Image sub_image(const Image& img, int left, int top, int width, int height)
{
    Image res(width, height);
    const auto& from_data = img.data;
    auto& to_data = res.data;
    for(int j = 0; j < height; j++) {
        auto from_row_origin = (top + j) * img.width;
        auto to_row_origin = j * width;
        for(int i = 0; i < width; i++) {
            to_data[to_row_origin + i] = from_data[from_row_origin + left + i]; 
        }
    }
    return res;
}