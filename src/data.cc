#include "data.h"

#include <fstream>
#include <json.hpp>
using json = nlohmann::json;

void from_json(const json& j, SpectrumData& sd) {
    j.at("wp").get_to(sd.wp);
    j.at("sectors").get_to(sd.sectors);
    j.at("light").get_to(sd.light);
    j.at("bases").get_to(sd.bases);
    j.at("tri_to_v_mtx").get_to(sd.tri_to_v_mtx);
}

void from_json(const json& j, ProfileData& pd) {
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

