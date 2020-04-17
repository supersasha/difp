#pragma once

#include <string>

#include "arrays.h"

struct SpectrumData
{
    Array<2> wp;
    Array<31> light;
    Array2D<3, 31> base;
    Array2D<3, 3> tri_to_v_mtx;
};

SpectrumData load_spectrum_data(const std::string& filename);

struct ProfileData
{
    Array2D<3, 31> film_sense;
    Array2D<3, 31> film_dyes;
    Array2D<3, 31> paper_sense;
    Array2D<3, 31> paper_dyes;

    Array2D<3, 31> couplers;
    Array<31> proj_light;
    Array<31> dev_light;
    Array2D<3, 31> mtx_refl;

    Array<3> neg_gammas;
    Array<3> paper_gammas;
    Array<3> film_max_qs;
};

ProfileData load_profile_data(const std::string& filename);

struct Datasheet
{
    Array2D<3, 31> sense;
    Array2D<3, 31> dyes;
};

Datasheet load_datasheet(const std::string& filename);

enum ProcessingMode
{
    NORMAL = 0,
    NEGATIVE,
    IDENTITY,
    FILM_EXPOSURE,
    GEN_SPECTR,
    FILM_DEV,
    PAPER_EXPOSURE,
    FILM_NEG_LOG_EXP,
    PAPER_NEG_LOG_EXP
};

struct UserOptions
{
    Array<3> color_corr;
    float film_exposure;
    float paper_exposure;
    float paper_contrast;
    float curve_smoo;
    //int negative;
    int mode;
    int channel;
    int frame_horz;
    int frame_vert;
};
