#pragma once

#include "image.h"
#include "photo_process_opts.h"
#include "data.h"

Image process_photo_old(const Image& img, PhotoProcessOpts& opts);
Image process_photo(const Image& img, SpectrumData& sd, ProfileData& pd, UserOptions& opts);
