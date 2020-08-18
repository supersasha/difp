#include <string>
#include <iostream>
#include <vector>
#include <math.h>

#include <nlopt.h>
#include <json.hpp>
using json = nlohmann::json;

#include "data.h"
#include "array_ops.h"
#include "color.h"

Array2D<3, 31> A_1931_64_400_700_10nm = transpose(Array2D<31, 3>({{
    { 0.0191097, 0.0020044, 0.0860109 },
    { 0.084736, 0.008756, 0.389366 },
    { 0.204492, 0.021391, 0.972542 },
    { 0.314679, 0.038676, 1.55348 },
    { 0.383734, 0.062077, 1.96728 },
    { 0.370702, 0.089456, 1.9948 },
    { 0.302273, 0.128201, 1.74537 },
    { 0.195618, 0.18519, 1.31756 },
    { 0.080507, 0.253589, 0.772125 },
    { 0.016172, 0.339133, 0.415254 },
    { 0.003816, 0.460777, 0.218502 },
    { 0.037465, 0.606741, 0.112044 },
    { 0.117749, 0.761757, 0.060709 },
    { 0.236491, 0.875211, 0.030451 },
    { 0.376772, 0.961988, 0.013676 },
    { 0.529826, 0.991761, 0.003988 },
    { 0.705224, 0.99734, 0 },
    { 0.878655, 0.955552, 0 },
    { 1.01416, 0.868934, 0 },
    { 1.11852, 0.777405, 0 },
    { 1.12399, 0.658341, 0 },
    { 1.03048, 0.527963, 0 },
    { 0.856297, 0.398057, 0 },
    { 0.647467, 0.283493, 0 },
    { 0.431567, 0.179828, 0 },
    { 0.268329, 0.107633, 0 },
    { 0.152568, 0.060281, 0 },
    { 0.0812606, 0.0318004, 0 },
    { 0.0408508, 0.0159051, 0 },
    { 0.0199413, 0.0077488, 0 },
    { 0.00957688, 0.00371774, 0 }
}}));

float zigzag(float x, float gamma, float ymax)
{
    if (x >= 0) {
        return ymax;
    }
    float x0 = -ymax/gamma;
    if (x <= x0) {
        return 0;
    }
    return gamma * (x - x0);
}

float zigzag_p(float x, float gamma, float ymax)
{
    if (x < 0) {
        return 0;
    }
    float y = x * gamma;
    if (y > ymax) {
        return ymax;
    }
    return y;
}

float zigzag_from(float x, float ymin, float ymax, float gamma, float bias)
{
    if (x <= bias) {
        return ymin;
    }
    float y = ymin + gamma * (x - bias);
    if (y > ymax) {
        return ymax;
    }
    return y;
}

float zigzag_to(float x, float ymin, float ymax, float gamma, float bias)
{
    if (x >= bias) {
        return ymax;
    }
    float y = ymax + gamma * (x - bias);
    if (y < ymin) {
        return ymin;
    }
    return y;
}

float sigma(float x, float ymin, float ymax, float gamma, float bias, float smoo)
{
    float a = (ymax - ymin) / 2;
    float y = gamma * (x - bias) / a;
    return a * (y / pow(1 + pow(fabs(y), 1/smoo), smoo) + 1) + ymin;
}

float sigma_from(float x, float ymin, float ymax, float gamma, float smoo, float x0)
{
    float avg = (ymax + ymin) / 2;

    // gamma * (x0 - bias) + avg = ymin
    
    float bias = x0 - (ymin - avg) / gamma;
    return sigma(x, ymin, ymax, gamma, bias, smoo);
}

double bell(double a, double mu, double sigma, double x)
{
    double d = (x - mu) / sigma;
    return a * exp(-d*d);
}

Array<3> exposure(const Array2D<3, 31>& logsense, const Array<31>& sp)
{
    return apply(pow, 10, logsense) % sp;
}

Array2D<3, 31>
normalized_sense(const Array2D<3, 31>& logsense, const Array<31>& light)
{
    Array<3> E = exposure(logsense, light);
    Array<3> theta = -apply(log10, E);
    std::cerr << "theta: " << theta << "\n";
    Array2D<3, 31> norm_sense = ~(~logsense + theta);
    std::cerr << "Exposure with normalized sense: " << exposure(norm_sense, light * 10) << "\n";
    return norm_sense;
}

Array2D<3, 31> transmittance_to_xyz_mtx(const Array<31>& light)
{
    const auto N = A_1931_64_400_700_10nm[1] % light;
    return A_1931_64_400_700_10nm * (100.0 / N * light);
}

Array<2> chromaticity(const Array<3>& xyz)
{
    float v = sum(xyz);
    if (fabs(v) < 1e-15) {
        return Array<2> {{1.0/3, 1.0/3}};
    }
    return Array<2> {{xyz[0] / v, xyz[1] / v}};
}

Array<2> white_point(const Array<31>& ill)
{
    Array<3> xyz = A_1931_64_400_700_10nm % ill;
    return chromaticity(xyz);
}

Array<31> transmittance(const Array2D<3, 31>& dyes, const Array<3>& q)
{
    return apply(pow, 10, -(~dyes % q));
}

Array<31> outflux(const Array2D<3, 31>& dyes, const Array<31>& light, const Array<3>& q)
{
    return light * transmittance(dyes, q);
}

float dye_density(const Array2D<3, 31>& dyes, const Array<31>& light, const Array<3>& qs)
{
    Array<31> out = outflux(dyes, light, qs);
    return log10(sum(light) / sum(out));
}


struct normalize_dyes_qs_opt_s
{
    Array2D<3, 31> dyes;
    Array<31> light;
    Array2D<3, 31> tr_mtx;
    Array<2> wp;
    float density;
};

double normalize_dyes_qs_opt_func(unsigned n, const double * q,
        double * gr, void * func_data)
{
    normalize_dyes_qs_opt_s * dt = (normalize_dyes_qs_opt_s *) func_data;
    Array<3> qs = array_from_ptr<double, 3>(q);
    float d = dye_density(dt->dyes, dt->light, qs);
    Array<31> trans = transmittance(dt->dyes, qs);
    Array<3> xyz = dt->tr_mtx % trans;
    Array<2> xy = chromaticity(xyz);
    return (d - dt->density) * (d - dt->density) + ((xy - dt->wp) % (xy - dt->wp));
}

Array<3> normalized_dyes_qs(const Array2D<3, 31>& dyes, const Array<31>& light,
    float density)
{
    normalize_dyes_qs_opt_s dt;
    dt.dyes = dyes;
    dt.light = light;
    dt.tr_mtx = transmittance_to_xyz_mtx(light);
    dt.wp = white_point(light);
    dt.density = density;

    nlopt_opt opt = nlopt_create(NLOPT_LN_PRAXIS, 3);
    nlopt_set_min_objective(opt, normalize_dyes_qs_opt_func, &dt);
    double lb[] = {0, 0, 0};
    double ub[] = {7, 7, 7};
    nlopt_set_lower_bounds(opt, lb);
    nlopt_set_upper_bounds(opt, ub);
    nlopt_set_stopval(opt, 1e-10);
    double opt_x[3] = {0, 0, 0};
    double opt_f;
    auto r = nlopt_optimize(opt, opt_x, &opt_f);
    std::cerr << "r: " << r << "; f: " << opt_f << "\n";

    auto res = array_from_ptr<double, 3>(opt_x);
    nlopt_destroy(opt);
    return res;
}

Array2D<3, 31> normalized_dyes(const Array2D<3, 31>& dyes, const Array<31>& light,
    float density)
{
    return ~(~dyes * normalized_dyes_qs(dyes, light, density));
}

struct ColorW
{
    Color color;
    float weight;
};

std::vector<float> linspace(float start, float end, int n)
{
    float d = (end - start) / (n - 1);
    std::vector<float> res;
    for (int i = 0; i < n; i++) {
        res.emplace_back(start + i * d);
    }
    return res;
}

class ReflGen
{
public:
    void init(const SpectrumData& sd)
    {
        m_tri_to_v_mtx = sd.tri_to_v_mtx;
        m_base = ~sd.base;
        m_light = sd.light;
    }

    Array<31> spectrum_of(const Color& _xyz) const
    {
        return refl_of(_xyz) * m_light;
    }
    
    Array<31> refl_of(const Color& _xyz) const
    {
        auto xyz = _xyz.to_array();
        auto v = m_tri_to_v_mtx % xyz;
        return clip((m_base % v), 1e-15, 1);
    }
private:
    Array<31> m_light;
    Array2D<31, 3> m_base;
    Array2D<3, 3> m_tri_to_v_mtx;
};

std::vector<ColorW> reference_colors(const ReflGen& refl_gen,
    const Array2D<3, 31>& mtx_refl)
{
    /*
    std::vector<ColorW> res, res2;
    auto vs = linspace(0.1, 1.0, 5);
    for (int i = 0; i < vs.size(); i++) {
        auto v = vs[i];
        res.emplace_back(ColorW{srgb_to_xyz(Color(v, 0, 0)), 1});
        res.emplace_back(ColorW{srgb_to_xyz(Color(0, v, 0)), 1});
        res.emplace_back(ColorW{srgb_to_xyz(Color(0, 0, v)), 1});
        res.emplace_back(ColorW{srgb_to_xyz(Color(v, v, 0)), 1});
        res.emplace_back(ColorW{srgb_to_xyz(Color(v, 0, v)), 1});
        res.emplace_back(ColorW{srgb_to_xyz(Color(0, v, v)), 1});
        res.emplace_back(ColorW{Color(95.67970526 * v, 100.0 * v, 92.1480586 * v), 1});
        //res.emplace_back(ColorW{srgb_to_xyz(Color(v, v, v)), 1});
    }
    return res;
    */
    /*
    for (int i = 0; i < res.size(); i++) {
        Color xyz = res[i].color;
        Array<31> refl = refl_gen.refl_of(xyz);
        Array<3> z = mtx_refl % refl;
        Color xyz1 = Color(z[0], z[1], z[2]);
        if (delta_E94_xyz(xyz, xyz1) < 1) {
            res2.emplace_back(res[i]);
        }

    }
    std::cerr << "N colors = " << res2.size() << "\n";
    return res2;
    */
    
    std::vector<ColorW> res;
    res.emplace_back(ColorW{Color(12.08, 19.77, 16.28), 1});
    res.emplace_back(ColorW{Color(20.86, 12.00, 17.97), 1});
    res.emplace_back(ColorW{Color(14.27, 19.77, 26.42), 1});
    res.emplace_back(ColorW{Color( 7.53,  6.55, 34.26), 1});
    res.emplace_back(ColorW{Color(64.34, 59.10, 59.87), 1});
    res.emplace_back(ColorW{Color(58.51, 59.10, 29.81), 1});
    res.emplace_back(ColorW{Color(37.93, 30.05,  4.98), 1});
    res.emplace_back(ColorW{Color(95.67970526, 100.0, 92.1480586), 1});
    res.emplace_back(ColorW{Color(95.67970526 / 4, 100.0 / 4, 92.1480586 / 4), 1});
    res.emplace_back(ColorW{Color(95.67970526 / 16, 100.0 / 16, 92.1480586 / 16), 1});
    res.emplace_back(ColorW{Color(95.67970526 / 64, 100.0 / 64, 92.1480586 / 64), 1});

    res.emplace_back(ColorW{Color(24.3763, 12.752, 3.093), 1});
    res.emplace_back(ColorW{Color(16.6155, 8.47486, 3.12047), 1});

    //res.emplace_back(ColorW{Color(95.05 / 2, 100.0 / 2, 108.9 / 2), 1});
    //res.emplace_back(ColorW{Color(95.05 / 4, 100.0 / 4, 108.9 / 4), 1});
    //res.emplace_back(ColorW{Color(95.05 / 16, 100.0 / 16, 108.9 / 16), 1});
    //res.emplace_back(ColorW{Color(95.05 / 64, 100.0 / 64, 108.9 / 64), 1});
    return res;
}

Array<31> spectrum_of(const SpectrumData& sd, const Color& xyz)
{
    Array<3> axyz = xyz.to_array();
    Array<3> v = sd.tri_to_v_mtx % axyz;
}

double couplers_opt_func(unsigned n, const double * x,
        double * gr, void * func_data);

/*
 * 1. Permanent setup:
 *      - film and paper senses
 *      - film dyes
 *      - basic paper dyes (to be corrected later)
 *      - spectrum generator
 *      - film and paper Chi maximums
 *      - film Chi
 *      - light sources
 *      - reflection Matrix
 * 2. Make couplers and secondary setup based on params:
 *      - paper gammas
 *      - paper Chi
 * 3. Develop function: takes xyz, returns xyz
 */

class Chi
{
public:
    Chi() {}

    Chi(float xmin, float xmax, float ymin, float ymax)
        : m_xmin(xmin), m_xmax(xmax), m_ymin(ymin), m_ymax(ymax)
    {}

    static Chi to(float ymin, float ymax, float gamma, float xmax)
    {
        float xmin = xmax - (ymax - ymin) / gamma;
        return Chi(xmin, xmax, ymin, ymax);
    }

    float operator()(float x) const
    {
        if (x < m_xmin) {
            return m_ymin;
        }
        if (x > m_xmax) {
            return m_ymax;
        }
        return (x - m_xmin) / (m_xmax - m_xmin) * (m_ymax - m_ymin) + m_ymin;
    }

    float gamma() const
    {
        return (m_ymax - m_ymin) / (m_xmax - m_xmin);
    }

    float hmax() const
    {
        return m_xmax;
    }
private:
    float m_xmin;
    float m_xmax;
    float m_ymin;
    float m_ymax;
};

static const int N_FREE_PARAMS = 0;
static constexpr std::array<int, 8> GAUSSIAN_LAYERS = {{0, 0, 1, 1, 1, 1, 2, 2}};
static const int N_GAUSSIANS = GAUSSIAN_LAYERS.size();
static const int N_GAUSSIAN_PARAMS = 3 * N_GAUSSIANS;
static const int N_PARAMS = N_FREE_PARAMS + N_GAUSSIAN_PARAMS;
using ParamVec = Array<N_PARAMS>;

class Developer
{
public:

    Developer()
    {
        std::string film_ds_file =
            //"profiles/datasheets/kodak-vision3-250d-5207-2.datasheet";
            "profiles/datasheets/kodak-vision3-50d-5203-2.datasheet";
        std::string paper_ds_file =
            "profiles/datasheets/kodak-vision-color-print-2383-2.datasheet";
        std::string spectrum_file =
            "research/profile/wthanson/spectra2/spectrum-d55-4.json";

        m_filmds = load_datasheet(film_ds_file);
        m_paperds = load_datasheet(paper_ds_file);
        m_spectrum = load_spectrum_data(spectrum_file);
        m_refl_gen.init(m_spectrum);
        m_mtx_refl = transmittance_to_xyz_mtx(m_refl_light);

        m_film_sense = normalized_sense(m_filmds.sense, m_dev_light);
        m_film_dyes = normalized_dyes(m_filmds.dyes, m_proj_light, 1.0);

        m_paper_dyes0 = normalized_dyes(m_paperds.dyes, m_refl_light, 1.0);
        m_paper_sense = normalized_sense(m_paperds.sense, m_proj_light);

        auto cf = Chi::to(0, m_kfmax, 0.5, 0);
        std::cerr << cf(0) << ", "<< m_kfmax << "\n";
        m_chi_film = {{ cf, cf, cf }};
    }

    void setup(const ParamVec& q)
    {
        make_couplers(q);

        Array<3> hproj_max = beta(-10.0);
        Array<3> hproj_min = beta(1.0);
        
        Array<3> paper_gammas = m_kpmax / (hproj_max - hproj_min);
        m_chi_paper = {{
            Chi::to(0, m_kpmax, paper_gammas[0], hproj_max[0]),
            Chi::to(0, m_kpmax, paper_gammas[1], hproj_max[1]),
            Chi::to(0, m_kpmax, paper_gammas[2], hproj_max[2]),
        }};

        m_paper_dyes = m_paper_dyes0 * (4 / delta(-4));
    }
    
    Color generated_color(const Color& xyz)
    {
        Array<31> refl = m_refl_gen.refl_of(xyz);
        Array<3> z = m_mtx_refl % refl;
        return Color(z[0], z[1], z[2]);
    }

    Color develop(const Color& xyz)
    {
        Array<31> sp = m_refl_gen.spectrum_of(xyz);
        Array<3> H = apply(log10, exposure(m_film_sense, sp));
        auto negative = develop_film(H);
        auto positive = develop_paper(negative);
        auto trans = transmittance(positive, ones<3>());
        Array<3> z = m_mtx_refl % trans;
        return Color(z[0], z[1], z[2]);
    }

    Array2D<3, 31> develop_film(const Array<3>& H)
    {
        Array<3> dev = {{
            m_chi_film[0](H[0]),
            m_chi_film[1](H[1]),
            m_chi_film[2](H[2]),
        }};
        Array2D<3, 31> developed_dyes = dev * m_film_dyes;
        Array<3> cDev = {{
            1 - dev[0] / m_kfmax,
            1 - dev[1] / m_kfmax,
            1 - dev[2] / m_kfmax
        }};
        Array2D<3, 31> developed_couplers = cDev * m_couplers;
        return developed_dyes + developed_couplers;
    }

    Array2D<3, 31> develop_paper(const Array2D<3, 31>& negative)
    {
        float ymax = 4;
        Array<31> trans = transmittance(negative, ones<3>());
        Array<31> sp = trans * m_proj_light;

        // log10(10^paper_sense % sp)
        Array<3> H1 = apply(log10, exposure(m_paper_sense, sp));
        Array<3> dev;
        dev = Array<3> {{
            m_chi_paper[0](H1[0]),
            m_chi_paper[1](H1[1]),
            m_chi_paper[2](H1[2]),
        }};
        //std::cerr << H1[0] << "  --  " << m_chi_paper[0](H1[0]) << " : "<< dev << "\n";
        return dev * m_paper_dyes;
    }

    std::string to_json()
    {
        //auto mtx = transmittance_to_xyz_mtx(m_proj_light);
        json j;
        j["film_sense"] = m_film_sense;
        j["film_dyes"] = m_film_dyes;
        j["paper_sense"] = m_paper_sense;
        j["paper_dyes"] = m_paper_dyes;
        j["couplers"] = m_couplers;
        j["proj_light"] = m_proj_light;
        j["dev_light"] = m_dev_light;
        j["mtx_refl"] = m_mtx_refl; // mtx;  see^
        j["neg_gammas"] = Array<3> {{
                                m_chi_film[0].gamma(),
                                m_chi_film[1].gamma(),
                                m_chi_film[2].gamma(),
                            }};
        j["paper_gammas"] = Array<3> {{
                                m_chi_paper[0].gamma(),
                                m_chi_paper[1].gamma(),
                                m_chi_paper[2].gamma(),
                            }};
        j["film_max_qs"] = Array<3> {{
                                m_chi_paper[0].hmax(),
                                m_chi_paper[1].hmax(),
                                m_chi_paper[2].hmax(),
                            }};
        return j.dump(4);
    }

    ReflGen m_refl_gen;
    Array2D<3, 31> m_mtx_refl;

private:
    const float m_kfmax = 2.5;
    const float m_kpmax = 4.0;

    Datasheet m_filmds;
    Datasheet m_paperds;
    SpectrumData m_spectrum;

    Array<31> m_dev_light = daylight_spectrum(5500);
    Array<31> m_proj_light = daylight_spectrum(5500);
    Array<31> m_refl_light = daylight_spectrum(6500);

    Array2D<3, 31> m_film_sense;
    Array2D<3, 31> m_film_dyes;

    Array2D<3, 31> m_paper_sense;
    Array2D<3, 31> m_paper_dyes0;
    Array2D<3, 31> m_paper_dyes;

    Array2D<3, 31> m_couplers;

    std::array<Chi, 3> m_chi_film;
    std::array<Chi, 3> m_chi_paper;
    
    Array<3> beta(float D)
    {
        float alpha = D;
        Array<3> kfs = {{
            m_chi_film[0](alpha),
            m_chi_film[1](alpha),
            m_chi_film[2](alpha),
        }};
        Array<3> cKfs = 1 - kfs/m_kfmax;
        
        Array<31> trans = apply(pow, 10,
            -(kfs % m_film_dyes
                + cKfs % m_couplers)
        );
        return apply(log10,
            apply(pow, 10, m_paper_sense) % (m_proj_light * trans)
        );
    }

    float delta(float D)
    {
        Array<3> betas = beta(D);
        Array<3> kps = {{
            m_chi_paper[0](betas[0]),
            m_chi_paper[1](betas[1]),
            m_chi_paper[2](betas[2]),
        }};
        Array<31> refl = apply(pow, 10, -kps % m_paper_dyes0);
        return log10(sum(m_refl_light) / (m_refl_light % refl));
    }

    void make_couplers(const ParamVec& q)
    {
        const int B = N_FREE_PARAMS; // Number of params not accounting gaussians
        const int G = N_GAUSSIANS; // Number of gaussians (3 params each) in each (of 3) layer
        m_couplers[0] = zeros<31>();
        m_couplers[1] = zeros<31>();
        m_couplers[2] = zeros<31>();
        for (int i = 0; i < G; i++) {
            double b = B + i * 3;
            double x = 400;
            for (int j = 0; j < 31; j++, x += 10) {
                m_couplers[GAUSSIAN_LAYERS[i]][j] += bell(q[b], q[b+1], q[b+2], x);
            }
        }
    }
};

double couplers_solve_func(unsigned n, const double * x,
        double * gr, void * func_data);

class Solver
{
public:
    ParamVec m_solution;

    Solver()
    {
        m_xyzs = reference_colors(m_dev.m_refl_gen, m_dev.m_mtx_refl);
        /*
        for (int i = 0; i < m_xyzs.size(); i++) {
            m_xyzs[i].color = m_dev.generated_color(m_xyzs[i].color);
        }
        */
    }

    void solve()
    {
        nlopt_opt opt = nlopt_create(NLOPT_GN_ISRES, N_PARAMS);
        //nlopt_set_population(opt, 5000);
        nlopt_set_min_objective(opt, couplers_solve_func, this);
        double lb[N_PARAMS] = {
            0, 350, 20,
            0, 350, 10,
            0, 350, 20,
            0, 350, 10,
            0, 600, 20,
            0, 600, 10,
            0, 500, 20,
            0, 500, 10,
        };
        double ub[N_PARAMS] = {
            2.0, 600, 150,
            1.0, 600, 50,
            2.0, 500, 150,
            1.0, 500, 50,
            2.0, 750, 150,
            1.0, 750, 50,
            2.0, 750, 150,
            1.0, 750, 50,
        };
        double opt_x[N_PARAMS] = {
            0.5, 500, 30,
            0.5, 500, 30,
            0.5, 450, 30,
            0.5, 450, 30,
            0.5, 650, 30,
            0.5, 650, 30,
            0.5, 550, 30,
            0.5, 550, 30,
        };
        nlopt_set_lower_bounds(opt, lb);
        nlopt_set_upper_bounds(opt, ub);
        nlopt_set_maxtime(opt, 60 * 5);

        double opt_f = 0;
        auto r = nlopt_optimize(opt, opt_x, &opt_f);
        std::cerr << "r: " << r << "; f: " << opt_f << "\n";

        //m_solution = array_from_ptr<double, N_PARAMS>(opt_x);
        nlopt_destroy(opt);
        //return res;
    }

    std::string to_json()
    {
        m_dev.setup(m_solution);
        return m_dev.to_json();
    }

    double solve_fun(const ParamVec& q, bool print = false)
    {
        m_dev.setup(q);
        double d = 0;
        for (const auto xyzw: m_xyzs) {
            Color xyz1 = m_dev.develop(xyzw.color);
            double d0 = delta_E94_xyz(xyzw.color, xyz1);
            double d1 = d0 * xyzw.weight;
            d += d1 * d1;
            if (print) {
                std::cerr << xyz_to_srgb(xyzw.color).to_array() << " --> "
                          << xyz_to_srgb(xyz1).to_array() << ": " << d0 << ", " << d1 << "\n";
            }
        }
        return d;
    }

private:
    Developer m_dev;
    std::vector<ColorW> m_xyzs;
};

double couplers_solve_func(unsigned n, const double * x,
        double * gr, void * func_data)
{
    Solver * solver = static_cast<Solver*>(func_data);
    ParamVec q = array_from_ptr<double, N_PARAMS>(x);
    
    static double r = 1e100;
    static int i = 0;
    double r1 = solver->solve_fun(q);
    i++;
    if (r1 < r) {
        r = r1;
        std::cerr << i << ": " << r1 << "; params: " << q << "\n";
        solver->m_solution = q;
    }

    return r1;
}

int main()
{
    Solver solver;
    solver.solve();
    solver.solve_fun(solver.m_solution, true);
    std::cout << solver.to_json() << "\n";
    /*
    Optimizer opt;
    opt.opt();
    std::cout << opt.to_json() << "\n";
    opt.opt_fun(opt.m_solution, true);
    */
}
