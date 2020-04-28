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
    //std::cout << "logsense: " << logsense << "\n";
    Array<3> E = exposure(logsense, light);
    //std::cout << "E: " << E << "\n";
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
    //std::cout << "outflux: " << out << "\n";
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

std::vector<ColorW> reference_colors()
{
    std::vector<ColorW> res;
    /*
    res.emplace_back(ColorW{srgb_to_xyz(Color(1, 0, 0)), 1});
    res.emplace_back(ColorW{srgb_to_xyz(Color(0, 1, 0)), 1});
    res.emplace_back(ColorW{srgb_to_xyz(Color(0, 0, 1)), 1});
    res.emplace_back(ColorW{srgb_to_xyz(Color(1, 1, 1)), 1});
    */
    /*
    auto vs = linspace(0.1, 1, 5);
    for (const auto v: vs) {
        res.emplace_back(ColorW{srgb_to_xyz(Color(v, 0, 0)), 1 + v*4});
        res.emplace_back(ColorW{srgb_to_xyz(Color(0, v, 0)), 1});
        res.emplace_back(ColorW{srgb_to_xyz(Color(v, v, 0)), 1});
        res.emplace_back(ColorW{srgb_to_xyz(Color(v, v, v)), 2});
        
        res.emplace_back(ColorW{srgb_to_xyz(Color(v, 0, 0.3 * v)), 1});

        //res.emplace_back(ColorW{srgb_to_xyz(Color(v, 0, 0)), v*4});
        //res.emplace_back(ColorW{srgb_to_xyz(Color(0, v, 0)), 1});
        //res.emplace_back(ColorW{srgb_to_xyz(Color(0, 0, v)), 1});
        //
        //res.emplace_back(ColorW{srgb_to_xyz(Color(0, v, v)), v});
        //res.emplace_back(ColorW{srgb_to_xyz(Color(v, 0, v)), v});
        //res.emplace_back(ColorW{srgb_to_xyz(Color(v, v, 0)), v});
        //
        //res.emplace_back(ColorW{srgb_to_xyz(Color(v, v, v)), 2});
    }
    */
    auto vs = linspace(0.1, 1, 5);
    for (int i = 0; i < vs.size(); i++) {
        auto v = vs[i];
        res.emplace_back(ColorW{srgb_to_xyz(Color(v, 0, 0)), 1});
        res.emplace_back(ColorW{srgb_to_xyz(Color(0, v, 0)), 1});
        res.emplace_back(ColorW{srgb_to_xyz(Color(v, v, 0)), 1});
        res.emplace_back(ColorW{srgb_to_xyz(Color(v, v, v)), 1 /*(i == 4) ? 5 : 1*/});
    }
    return res;
}

Array<31> spectrum_of(const SpectrumData& sd, const Color& xyz)
{
    Array<3> axyz = xyz.to_array();
    Array<3> v = sd.tri_to_v_mtx % axyz;
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

    Array<31> spectrum_of(const Color& _xyz)
    {
        auto xyz = _xyz.to_array();
        auto v = m_tri_to_v_mtx % xyz;
        return clip((m_base % v), 1e-15, 1) * m_light;
    }
private:
    Array<31> m_light;
    Array2D<31, 3> m_base;
    Array2D<3, 3> m_tri_to_v_mtx;
};

double couplers_opt_func(unsigned n, const double * x,
        double * gr, void * func_data);

class Optimizer
{
public:
    static constexpr int N_FREE_PARAMS = 10;
    static constexpr int N_GAUSSIANS = 2;
    static constexpr int N_COUPLED_LAYERS = 0;
    static constexpr int N_GAUSSIAN_PARAMS = 3 * N_COUPLED_LAYERS * N_GAUSSIANS;
    static constexpr int N_PARAMS = N_FREE_PARAMS + N_GAUSSIAN_PARAMS;
    using ParamVec = Array<N_PARAMS>;
    ParamVec m_solution;

    Optimizer()
    {
        std::string film_ds_file =
            //"profiles/datasheets/kodak-vision-250d-5207.datasheet";
            "profiles/datasheets/kodak-vision3-250d-5207.datasheet";
        std::string paper_ds_file =
            //"profiles/datasheets/kodak-endura.datasheet";
            "profiles/datasheets/kodak-vision-color-print-2383.datasheet";
        std::string spectrum_file =
            "research/profile/wthanson/spectra2/spectrum-d55-4.json";

        m_filmds = load_datasheet(film_ds_file);
        m_paperds = load_datasheet(paper_ds_file);
        m_spectrum = load_spectrum_data(spectrum_file);
        m_refl_gen.init(m_spectrum);
        m_mtx_refl = transmittance_to_xyz_mtx(m_refl_light);

        m_film_sense = normalized_sense(m_filmds.sense, m_dev_light);
        //std::cout << "Film sense (norm): " << m_film_sense << "\n"; 
        m_film_dyes = normalized_dyes(m_filmds.dyes, m_proj_light, 1);
        m_film_max_qs = normalized_dyes_qs(m_film_dyes, m_proj_light, m_max_density);
        //std::cout << "film_max_qs: " << m_film_max_qs << "\n";
        m_film_max_dyes = ~(~m_film_dyes * m_film_max_qs);

        m_paper_dyes = normalized_dyes(m_paperds.dyes, m_refl_light, 1.0);
        Array<31> neg_white = transmittance(m_film_max_dyes, ones<3>()) * m_proj_light;
        std::cerr << "neg_white: " << (m_mtx_refl % (neg_white / m_proj_light)) << "\n";
        std::cerr << "neg_white: " << neg_white << "\n";
        m_paper_sense = normalized_sense(m_paperds.sense, neg_white);
    }

    void opt()
    {
        /*
        auto nd = normalized_dyes_qs(m_filmds.dyes, ones<31>(), 1.0);
        std::cout << "nd: " << nd << "\n";
        auto nd2 = normalized_dyes_qs(nd * m_filmds.dyes, ones<31>(), 1.0);
        std::cout << "nd2: " << nd2 << "\n";
        */
        nlopt_opt opt = nlopt_create(NLOPT_GN_ISRES, N_PARAMS);
        //nlopt_set_population(opt, 5000);
        nlopt_set_min_objective(opt, couplers_opt_func, this);
        double lb[N_PARAMS] = {
            // film: gamma
            0, 0, 0,

            // paper
            //0, 0, 0,

            -6, -6, -6,
            -6, -6, -6,
            //0,

            0.1,
                        
            /*
            0.0001, 350, 20,
            0.0001, 350, 20,
            0.0001, 350, 20,
            0.0001, 350, 20,
            0.0001, 350, 20,
            0.0001, 350, 20,
            */
        };
        double ub[N_PARAMS] = {
            3, 1, 1,
            //5, 5, 5,

            6, 6, 6,
            6, 6, 6,
            //6,

            10,

            /*
            1, 750, 50,
            1, 750, 50,
            1, 750, 50,
            1, 750, 50,
            1, 750, 50,
            1, 750, 50,
            */
        };
        double opt_x[N_PARAMS] = {
            0.5, 0.5, 0.5,
            //1, 1, 1,

            0, 0, 0,
            0, 0, 0,
            //0,

            1,

            /*
            0.5, 500, 30,
            0.5, 500, 30,
            0.5, 500, 30,
            0.5, 500, 30,
            0.5, 500, 30,
            0.5, 500, 30,
            */
        };
        nlopt_set_lower_bounds(opt, lb);
        nlopt_set_upper_bounds(opt, ub);
        nlopt_set_maxtime(opt, 60 * 30);
        /*
        double opt_x[] = {0.5, 0.5, 0.5,
                            0, 0, 0,
                            0.5, 10,
                            0.5, 500, 100,
                            0.5, 500, 100,
                            0.5, 500, 100,
                            0.5, 500, 100,
                            0.5, 500, 100,
                            0.5, 500, 100,
                            0.5, 500, 100,
                            0.5, 500, 100,
                            0.5, 500, 100,
                            0.5, 500, 100,
                            0.5, 500, 100,
                            0.5, 500, 100,
                        };
        */
        double opt_f = 0;
        auto r = nlopt_optimize(opt, opt_x, &opt_f);
        std::cerr << "r: " << r << "; f: " << opt_f << "\n";

        //m_solution = array_from_ptr<double, N_PARAMS>(opt_x);
        nlopt_destroy(opt);
        //return res;
    }

    double opt_fun(const ParamVec& q, bool print = false)
    {
        make_couplers(q);
        double d = 0;
        for (const auto xyzw: m_xyzs) {
            Color xyz1 = develop(xyzw.color, q);
            double d0 = delta_E76_xyz(xyzw.color, xyz1);
            double d1 = d0 * xyzw.weight;
            d += d1 * d1;
            if (print) {
                std::cerr << xyz_to_srgb(xyzw.color).to_array() << " --> "
                          << xyz_to_srgb(xyz1).to_array() << ": " << d0 << ", " << d1 << "\n";
            }
        }
        return d;
    }

    void make_couplers(const ParamVec& q)
    {
        // Overall number of params is B + G * L * 3 = 48
        const int B = N_FREE_PARAMS; // Number of params not accounting gaussians
        const int G = N_GAUSSIANS; // Number of gaussians (3 params each) in each (of 3) layer
        const int L = N_COUPLED_LAYERS; // Number of layers
        m_couplers[0] = zeros<31>();
        m_couplers[1] = zeros<31>();
        m_couplers[2] = zeros<31>();
        for (int i = 0; i < L; i++) {
            double b = B + G * i * 3;
            double x = 400;
            for (int j = 0; j < 31; j++, x += 10) {
                m_couplers[i][j] = 0;
                for (int k = 0; k < G; k++) {
                    double s = b + k * 3;
                    m_couplers[i][j] += bell(q[s], q[s+1], q[s+2], x);
                }
            }
        }
        //std::cout << "couplers: " << m_couplers << "\n";
    }

    Array2D<3, 31> develop_film(const Array<3>& H, const ParamVec& q)
    {
        float ymax = 4;
        Array<3> dev = {{
            /*
            zigzag1(H[0], 0, ymax, q[0], q[1]),
            zigzag1(H[1], 0, ymax, q[2], q[3]),
            zigzag1(H[2], 0, ymax, q[4], q[5])
            */
            zigzag_to(H[0], 0, ymax, q[0], 0),
            zigzag_to(H[1], 0, ymax, q[1], 0),
            zigzag_to(H[2], 0, ymax, q[2], 0)
        }};
        //std::cout << "dev: " << dev << "\n";
        //std::cout << "dyes: " << m_film_max_dyes << "\n";
        Array2D<3, 31> developed_dyes = dev * m_film_max_dyes;
        Array<3> cDev = {{
            1 - dev[0] / ymax,
            1 - dev[1] / ymax,
            1 - dev[2] / ymax
        }};
        //auto m = Array<3>{{q[1], q[5], q[9]}};
        //Array2D<3, 31> developed_couplers = (m - dev) / m * m_couplers;
        Array2D<3, 31> developed_couplers = cDev * m_couplers;
        return developed_dyes + developed_couplers;
    }

    Array2D<3, 31> develop_paper(const Array2D<3, 31>& negative, const ParamVec& q)
    {
        float ymax = 4;
        Array<31> trans = transmittance(negative, ones<3>());
        //std::cout << "trans: " << trans << "\n";
        Array<31> sp = trans * m_proj_light;
        //std::cout << "sp1: " << sp << "\n";

        // log10(10^paper_sense % sp)
        Array<3> H1 = apply(log10, (apply(pow, 10, ~(~m_paper_sense + /*q[N_FREE_PARAMS - 2]*/subarray<N_PARAMS, 6, 3>(q))) % sp)); // * m_paper_gammas;
        //H1[1] += q[3];
        //H1[2] += q[4];
        //std::cout << "H1: " << H1 << "\n";
        /*
        H1[0] = zigzag_p(H1[0], m_paper_gammas[0], 4);
        H1[1] = zigzag_p(H1[1], m_paper_gammas[1], 4);
        H1[2] = zigzag_p(H1[2], m_paper_gammas[2], 4);
        */
        Array<3> dev = {{
            /*
            zigzag1(H1[0], 0, ymax, q[6],  q[7]),
            zigzag1(H1[1], 0, ymax, q[8],  q[9]),
            zigzag1(H1[2], 0, ymax, q[10], q[11])
            */
            zigzag_from(H1[0], 0, ymax, 5, 0),
            zigzag_from(H1[1], 0, ymax, 5, 0),
            zigzag_from(H1[2], 0, ymax, 5, 0)
        }};
        return dev * m_paper_dyes;
    }

    Color develop(const Color& xyz, const ParamVec& q)
    {
        Array<31> sp = m_refl_gen.spectrum_of(xyz);
        //std::cout << "sp: " << sp << "\n";
        Array<3> H = apply(log10, exposure(~(~m_film_sense + subarray<N_PARAMS, 3, 3>(q)), sp));
        //std::cout << "H: " << H << "\n";
        auto negative = develop_film(H, q);
        //std::cout << "negative: " << negative << "\n";
        auto positive = develop_paper(negative, q);
        //std::cout << "positive: " << positive << "\n";
        auto trans = transmittance(positive, ones<3>());
        Array<3> z = m_mtx_refl % trans * q[N_FREE_PARAMS - 1];
        //std::cout << "\n";
        return Color(z[0], z[1], z[2]);
    }

    std::string to_json()
    {
        make_couplers(m_solution);
        json j;
        j["film_sense"] = m_film_sense;
        j["film_dyes"] = m_film_max_dyes;
        j["paper_sense"] = m_paper_sense;
        j["paper_dyes"] = m_paper_dyes;
        j["couplers"] = m_couplers;
        j["proj_light"] = m_proj_light;
        j["dev_light"] = m_dev_light;
        j["mtx_refl"] = m_mtx_refl;
        j["neg_gammas"] = Array<3> {{ m_solution[0], m_solution[1], m_solution[2] }};
        j["paper_gammas"] = m_paper_gammas;
        j["film_max_qs"] = m_film_max_qs;
        return j.dump(4);
    }
private:
    Datasheet m_filmds;
    Datasheet m_paperds;
    SpectrumData m_spectrum;
    ReflGen m_refl_gen;
    Array2D<3, 31> m_mtx_refl;

    Array<31> m_dev_light = daylight_spectrum(5500);
    Array<31> m_proj_light = daylight_spectrum(5500);
    Array<31> m_refl_light = daylight_spectrum(6500);

    double m_max_density = 1.0;
    Array<3> m_paper_gammas = {{ 2.0, 2.0, 2.0 }};
    Array2D<3, 31> m_film_sense;
    Array2D<3, 31> m_film_dyes;
    Array<3> m_film_max_qs;
    Array2D<3, 31> m_film_max_dyes;

    Array2D<3, 31> m_paper_dyes;
    Array2D<3, 31> m_paper_sense;

    Array2D<3, 31> m_couplers;
    std::vector<ColorW> m_xyzs = reference_colors();

};

double couplers_opt_func(unsigned n, const double * x,
        double * gr, void * func_data)
{
    Optimizer * opt = static_cast<Optimizer*>(func_data);
    Optimizer::ParamVec q = array_from_ptr<double, Optimizer::N_PARAMS>(x);
    
    static double r = 1e100;
    static int i = 0;
    double r1 = opt->opt_fun(q);
    i++;
    if (r1 < r) {
        r = r1;
        std::cerr << i << ": " << r1 << "; params: " << q << "\n";
        opt->m_solution = q;
    }

    return r1;
}

int main()
{
    Optimizer opt;
    opt.opt();
    std::cout << opt.to_json() << "\n";
    opt.opt_fun(opt.m_solution, true);
    /*
    double opt_x[] = {0.5, 0.5, 0.5,
                        1, 500, 100,
                        1, 500, 100,
                        1, 500, 100,
                        1, 500, 100,
                        1, 500, 100,
                        1, 500, 100,
                        1, 500, 100,
                        1, 500, 100,
                        1, 500, 100,
                        1, 500, 100,
                        1, 500, 100,
                        1, 500, 100,
                        1, 500, 100,
                        1, 500, 100,
                        1, 500, 100,
                    };
    double of = opt.opt_fun(array_from_ptr<double, 48>(opt_x));
    std::cout << "of: " << of << "\n";
    */
}
