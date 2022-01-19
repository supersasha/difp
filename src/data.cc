#include "data.h"

#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

void from_json(const json& j, SpectrumData& sd)
{
    j.at("wp").get_to(sd.wp);
    j.at("light").get_to(sd.light);
    j.at("base").get_to(sd.base);
    j.at("tri_to_v_mtx").get_to(sd.tri_to_v_mtx);
}

void from_json(const json& j, ProfileData& pd)
{
    j.at("film_sense").get_to(pd.film_sense);
    j.at("film_dyes").get_to(pd.film_dyes);
    j.at("paper_sense").get_to(pd.paper_sense);
    j.at("paper_dyes").get_to(pd.paper_dyes);

    j.at("couplers").get_to(pd.couplers);
    j.at("proj_light").get_to(pd.proj_light);
    j.at("dev_light").get_to(pd.dev_light);
    j.at("mtx_refl").get_to(pd.mtx_refl);

    j.at("neg_gammas").get_to(pd.neg_gammas);
    j.at("paper_gammas").get_to(pd.paper_gammas);
    j.at("film_max_qs").get_to(pd.film_max_qs);
}

Array<31> cv65to31(const Array<65>& a1)
{
    Array<31> a;
    int o = 0;
    for (int i = 4; i < 65; i += 2, o++) {
        a[o] = a1[i];
    }
    return a;
}

void from_json(const json& j, Datasheet& ds)
{
    if (j.count("samples")) {
        int samples = 0;
        j.at("samples").get_to(samples);
        if (samples == 31) {
            j.at("red").at("sense").get_to(ds.sense[0]);
            j.at("green").at("sense").get_to(ds.sense[1]);
            j.at("blue").at("sense").get_to(ds.sense[2]);

            j.at("red").at("dye").at("data").get_to(ds.dyes[0]);
            j.at("green").at("dye").at("data").get_to(ds.dyes[1]);
            j.at("blue").at("dye").at("data").get_to(ds.dyes[2]);
            return;
        }
    }
    Array<65> a;
    j.at("red").at("sense").get_to(a);
    ds.sense[0] = cv65to31(a);
    j.at("green").at("sense").get_to(a);
    ds.sense[1] = cv65to31(a);
    j.at("blue").at("sense").get_to(a);
    ds.sense[2] = cv65to31(a);

    j.at("red").at("dye").at("data").get_to(a);
    ds.dyes[0] = cv65to31(a);
    j.at("green").at("dye").at("data").get_to(a);
    ds.dyes[1] = cv65to31(a);
    j.at("blue").at("dye").at("data").get_to(a);
    ds.dyes[2] = cv65to31(a);
}

SpectrumData load_spectrum_data(const std::string& filename)
{
    std::ifstream fsd(filename);
    json jsd;
    fsd >> jsd;
    return jsd;
}

ProfileData load_profile_data(const std::string& filename)
{
    std::ifstream fpd(filename);
    json jpd;
    fpd >> jpd;
    return jpd;
}

Datasheet load_datasheet(const std::string& filename)
{
    std::ifstream fds(filename);
    json jds;
    fds >> jds;
    return jds;
}
