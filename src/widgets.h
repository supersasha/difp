#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <imgui.h>

template <typename F>
bool MiniBrowser(const std::string& id, F f, std::string& item)
{
    bool result = false;
    if (item == "") {
        std::vector<std::string> items = f();
        if (!items.empty()) {
            item = items[0];
            result = true;
        }
    }
    if (ImGui::ArrowButton((std::string("##left") + id).c_str(), ImGuiDir_Left)) {
        std::vector<std::string> items = f();
        auto it = std::find(items.begin(), items.end(), item);
        if (it != items.end()) {
            if (it != items.begin()) {
                it--;
                item = *it;
                result = true;
            }
        }
    }
    ImGui::SameLine();
    if (ImGui::ArrowButton((std::string("##right") + id).c_str(), ImGuiDir_Right)) {
        std::vector<std::string> items = f();
        auto it = std::find(items.begin(), items.end(), item);
        if (it != items.end()) {
            it++;
            if (it != items.end()) {
                item = *it;
                result = true;
            }
        }
    }
    ImGui::SameLine();
    ImGui::Text("%s", item.c_str());
    return result;
}

