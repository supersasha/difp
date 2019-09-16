#include <iostream>

#include "frame.h"

std::vector<PaperFormat> gPaperFormats = {
    { "8x10", 82, 102, 1035, 1285 },
    { "10x10", 102, 102, 1285, 1285 },
    { "9x13", 85, 127, 1071, 1600 },
    { "10x20", 102, 203, 1285, 2557 },
    { "10x30", 102, 305, 1285, 3843 },
    { "10x13", 95, 127, 1197, 1600 },
    { "10x15", 102, 152, 1285, 1915 },
    { "11x15", 114, 152, 1436, 1915 },
    { "13x19", 127, 190, 1600, 2394 },
    { "15x15", 152, 152, 1915, 1915 },
    { "15x20", 152, 203, 1915, 2557 },
    { "15x21", 152, 215, 1915, 2709 },
    { "15x22", 152, 225, 1915, 2835 },
    { "15x30", 152, 305, 1915, 3843 },
    { "15x45", 152, 457, 1915, 5757 },
    { "20x30", 203, 305, 2557, 3843 },
    { "24x30", 247, 305, 3112, 3843 },
    { "30x30", 305, 305, 3843, 3843 },
    { "30x40", 305, 406, 3843, 5115 },
    { "30x45", 305, 457, 3843, 5757 },
    { "30x60", 305, 610, 3843, 7685 },
    { "30x90", 305, 914, 3843, 11515 }
};

const std::vector<PaperFormat>& get_paper_formats()
{
    return gPaperFormats;
}

IntSize outer_frame(const PaperFormat& pf, float w, float h, float t)
{
    float Q = float(pf.width_px) / pf.height_px;
    float q = w / h;
    if (Q < 1 && q > 1 || Q > 1 && q < 1) {
        Q = 1 / Q;
    }
    float qp = (w + t*w) / (h + t*w);
    float W, H;
    if (Q < qp) {
        W = w * (1 + 2*t);
        H = W / Q;
    } else {
        H = h + 2*t*w;
        W = Q * H;
    }
    return IntSize{ (int)round((W-w)/2), (int)round((H-h)/2) };
}

void inner_frame(const PaperFormat& pf, float q, float fit_w, float fit_h, float t,
        IntSize& wh, IntSize& WH)
{
    float Q = float(pf.width_px) / pf.height_px;
    if (Q < 1 && q > 1 || Q > 1 && q < 1) {
        Q = 1 / Q;
    }
    float W, H;
    if (fit_w / fit_h > Q) {
        H = fit_h;
        W = Q * H;
    } else {
        W = fit_w;
        H = W / Q;
    }
    float qp = (t + 1) / (t + 1/q);
    float w, h;
    if (Q < qp) {
        w = W / (1+2*t);
        h = w / q;
    } else {
        w = q * W / (Q * (2*q*t + 1));
        h = w / q;
    }
    wh = IntSize{ (int)round(w), (int)round(h) };
    WH = IntSize{ (int)round((W-w)/2), (int)round((H-h)/2) };
}
