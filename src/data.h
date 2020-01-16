#pragma once

#include <string>
#include <array>

template<size_t N> using Array = std::array<float, N>;
template<size_t ROWS, size_t COLS> using Array2D =
    std::array<std::array<float, COLS>, ROWS>;
template<size_t N, size_t ROWS, size_t COLS> using Array3D =
    std::array<std::array<std::array<float, COLS>, ROWS>, N>;

struct SpectrumData
{
    Array<2> wp;
    Array2D<6, 2> sectors;
    Array<31> light;
    Array3D<7, 3, 31> bases;
    Array3D<7, 3, 3> tri_to_v_mtx;
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
    Array2D<3, 31> mtx_refl;

    Array<3> neg_gammas;
    Array<3> paper_gammas;
    Array<3> film_max_qs;
};

ProfileData load_profile_data(const std::string& filename);

struct UserOptions
{
    Array<3> color_corr;
    float film_exposure;
    float paper_exposure;
    float paper_contrast;
    float curve_smoo;
    int frame_horz;
    int frame_vert;
};
