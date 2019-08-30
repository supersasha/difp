#include "photo_process_opts.h"

void from_json(const json& j, Extra& e)
{
    j.at("psr").get_to(e.psr);
    j.at("psg").get_to(e.psg);
    j.at("psb").get_to(e.psb);

    j.at("cy").get_to(e.cy);
    j.at("cm").get_to(e.cm);
    j.at("my").get_to(e.my);

    j.at("stop").get_to(e.stop);
    j.at("data").get_to(e.data);
}

void from_json(const json& j, CharacteristicCurve& cc)
{
    if (j.count("nodes")) {
        json nodes = j["nodes"];
        Spline::vec xs;
        Spline::vec ys;
        for (int i = 0; i < nodes.size(); i++) {
            xs.emplace_back(nodes[i]["x"]);
            ys.emplace_back(nodes[i]["y"]);
        }
        auto spline = Spline(xs, ys);
        const auto& data = spline.data();
        cc.nodes_cnt = data.size();
        for (int i = 0; i < cc.nodes_cnt; i++) {
            cc.spline[i] = data[i];
        }
    } else {
        j.at("min").get_to(cc.min);
        j.at("max").get_to(cc.max);
        j.at("tangent").get_to(cc.tangent);
        j.at("bias").get_to(cc.bias);
        j.at("smoothness").get_to(cc.smoothness);
    }
}

void from_json(const json& j, Illuminant& il)
{
    j.get_to(il.v);
}

void from_json(const json& j, SpectralSensitivity& ss)
{
    j.get_to(ss.v);
}

void from_json(const json& j, SpectralDyeDensity& sdd)
{
    if (j.at("method") == "points") {
        j.at("data").get_to(sdd.v);
    }
}

void from_json(const json& j, MaterialLayer& ml)
{
    j.at("curve").get_to(ml.curve);
    j.at("sense").get_to(ml.sense);
    if (j.count("dye")) {
        j.at("dye").get_to(ml.dye);
    }
    if (j.count("amp")) {
        j.at("amp").get_to(ml.amp);
        ml.amp = pow(10.0, ml.amp);
    }
    if (j.count("theta")) {
        j.at("theta").get_to(ml.theta);
    }
    if (j.count("couplers")) {
        j.at("couplers").get_to(ml.couplers);
    }

    for(auto &s : ml.sense.v) {
        s = pow(10.0f, s);
    }
}

void from_json(const json& j, Fog& fog)
{
    j.get_to(fog.v);
}

void from_json(const json& j, PhotoMaterial& pm)
{
    if (j.count("fog")) {
        j.at("fog").get_to(pm.fog);
    } else {
        for (auto i = 0; i < SPECTRUM_SIZE; i++) {
            pm.fog.v[i] = 0;
        }
    }
    std::map<std::string, MaterialLayer*> layers = {
        {"bw", &pm.red},
        {"red", &pm.red},
        {"green", &pm.green},
        {"blue", &pm.blue}
    };
    for (auto kv: layers) {
        if (j.count(kv.first)) {
            j.at(kv.first).get_to(*kv.second);
        }
    }
}

