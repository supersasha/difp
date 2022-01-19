#pragma once

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "spline.h"

const size_t SPECTRUM_SIZE = 65;
const float SPECTRUM_BASE = 380.0;
const float SPECTRUM_STEP = 5.0;

struct Extra
{
    float psr;
    float psg;
    float psb;
    float linear_amp;
    float layer2d;

    float cy;
    float cm;
    float my;

    int stop;
    int data;

    int pixel;
    int frame_horz = 0;
    int frame_vert = 0;
    float film_contrast = 1.0;
    float paper_contrast = 1.0;
    float light_through_film = 0.0;
    float light_on_paper = 0.0;
    float paper_filter[3];
};

struct Debug
{
    std::array<float, SPECTRUM_SIZE> spectrum;
    std::array<float, SPECTRUM_SIZE> film_fall_spectrum;
    std::array<float, SPECTRUM_SIZE> film_pass_spectrum;
    std::array<float, SPECTRUM_SIZE> film_fltr_spectrum;
    std::array<float, SPECTRUM_SIZE> paper_fall_spectrum;
    std::array<float, SPECTRUM_SIZE> paper_refl_spectrum;
    std::array<float, 3> xyz_in;

    std::array<float, 3> film_exposure;
    std::array<float, 3> film_density;
    std::array<float, 3> film_density2;
    std::array<float, 3> film_tdensity;

    std::array<float, 3> paper_exposure;
    std::array<float, 3> paper_density;
    std::array<float, 3> paper_tdensity;

    std::array<float, 3> xyz_out;
    std::array<float, 3> srgb_out;
};

struct CharacteristicCurve
{
    int nodes_cnt = 0;
    float tangent;
    float max;
    float min;
    float bias;
    float smoothness;

    std::array<Spline::S, 100> spline;
    
    float log_density(float x, float exposure_correction = 0) const
    {
        float a = (max - min) / 2;
        float y = tangent * (x + bias + exposure_correction) / a;
        float w = y / pow(1 + pow(fabs(y), 1 / smoothness), smoothness);
        return min + a * (w + 1);
    }
};

struct Illuminant
{
    std::array<float, SPECTRUM_SIZE> v;
};

struct SpectralSensitivity
{
    std::array<float, SPECTRUM_SIZE> v;
};

struct SpectralDyeDensity
{
    std::array<float, SPECTRUM_SIZE> v;
};

struct MaterialLayer
{
    CharacteristicCurve curve;
    SpectralSensitivity sense;
    SpectralDyeDensity dye;
    std::array<float, 3> couplers;
    float amp = 1.0;
    float theta = 0.0;
};

struct Fog
{
    std::array<float, SPECTRUM_SIZE> v;
};

struct PhotoMaterial
{
    Fog fog;
    MaterialLayer red;
    MaterialLayer green;
    MaterialLayer blue;
};

struct PhotoProcessOpts
{
    Illuminant illuminant1;
    Illuminant illuminant2;

    PhotoMaterial film;
    PhotoMaterial paper;

    Extra extra;
    Debug debug;

    float exposure_correction_film;
    float exposure_correction_paper;
};

void from_json(const json& j, Extra& e);
void from_json(const json& j, CharacteristicCurve& cc);
void from_json(const json& j, Illuminant& il);
void from_json(const json& j, SpectralSensitivity& ss);
void from_json(const json& j, SpectralDyeDensity& sdd);
void from_json(const json& j, MaterialLayer& ml);
void from_json(const json& j, Fog& fog);
void from_json(const json& j, PhotoMaterial& pm);
