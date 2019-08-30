#pragma once

#include <json.hpp>
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

    float cy;
    float cm;
    float my;

    int stop;
    int data;

    int pixel;
};

struct CharacteristicCurve
{
    int nodes_cnt = 0;
    float tangent;
    float max;
    float min;
    float bias;
    float smoothness;

    std::array<Spline::S, 30> spline;
    
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
