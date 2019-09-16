#pragma once

#include <string>
#include <vector>
#include <math.h>

#include "utils.h"

struct PaperFormat
{
    std::string name;
    float width_mm;
    float height_mm;
    int width_px;
    int height_px;
};

const std::vector<PaperFormat>& get_paper_formats();

IntSize outer_frame(const PaperFormat& pf, float w, float h, float t);

void inner_frame(const PaperFormat& pf,
                    float q, float fit_w, float fit_h, float t,
                    IntSize& wh, IntSize& WH);
