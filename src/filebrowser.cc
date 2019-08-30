#include "filebrowser.h"

#include <unistd.h>
#include <dirent.h>

#include <imgui.h>
#include <iostream>
#include <algorithm>

static inline ImVec2 operator-(const ImVec2& lhs, const ImVec2& rhs)            { return ImVec2(lhs.x-rhs.x, lhs.y-rhs.y); }

std::vector<DirPart> split_dir(const std::string& dir)
{
    std::vector<DirPart> res;
    res.emplace_back(DirPart("/", "/"));
    std::string part;
    for (int i = 1; i < dir.length(); i++) {
        if (dir[i] == '/') {
            res.emplace_back(
                DirPart(
                    part,
                    res.back().path + (res.back().path == "/" ? "" : "/") + part
                )
            );
            part = "";
        } else {
            part += dir[i];
        }
    }
    if (!part.empty()) {
        res.emplace_back(
            DirPart(
                part,
                res.back().path + (res.back().path == "/" ? "" : "/") + part
            )
        );
    }
    return res;
}

FileInfo::FileInfo()
{
    char buf[PATH_MAX + 1];
    m_dir = getcwd(buf, PATH_MAX + 1);
    m_parts = split_dir(m_dir);
    obtainEntries();
}

void FileInfo::recomposeDir(std::vector<DirPart>::const_iterator eit)
{
    m_dir = eit->path;

    std::vector<DirPart> parts;
    for (auto it = m_parts.begin(); it != eit; it++) {
        parts.emplace_back(*it);
    }
    parts.emplace_back(*eit);
    m_parts = parts;
    m_filename = "";
    obtainEntries();
}

void FileInfo::dirUp()
{
    if (m_dir == "/") {
        return;
    }
    auto it = m_parts.end();
    it--; it--;
    //std::cout << "dirUp: " << it->part << ", " << it->path << "\n";
    recomposeDir(it);
}

void FileInfo::obtainEntries()
{
    m_entries.clear();
    DIR * dir = opendir(m_dir.c_str());
    dirent * entry;
    while (entry = readdir(dir)) {
        if (entry->d_name[0] == '.') {
            continue;
        }
        if (entry->d_type == DT_DIR || entry->d_type == DT_REG) {
            m_entries.emplace_back(DirEntry(entry->d_name, entry->d_type));
        }
    }
    std::sort(m_entries.begin(), m_entries.end(),
        [](const DirEntry& e1, const DirEntry& e2) {
            if (e1.type == DT_DIR && e2.type == DT_REG) {
                return true;
            } else if (e1.type == DT_REG && e2.type == DT_DIR) {
                return false;
            }
            return e1.name < e2.name;
        }
    );
}

void FileInfo::setDir(const std::string& dir)
{
    if (m_dir != "/") {
        m_dir += "/";
    }
    m_dir += dir;
    m_parts.emplace_back(DirPart(dir, m_dir));
    m_filename = "";
    obtainEntries();
}


bool FileBrowser(const std::string& name, FileInfo& fi)
{
    ImGui::Begin(name.c_str());
    bool isFirst = true;
    for (auto it = fi.dirParts().begin(); it != fi.dirParts().end(); it++) {
        if (!isFirst) {
            ImGui::SameLine();
        } else {
            isFirst = false;
        }
        if (ImGui::Button(it->part.c_str())) {
            fi.recomposeDir(it);
            break;
        }
    }

    ImVec2 size = ImGui::GetContentRegionMax() - ImVec2(0.0f, 120.0f);

	ImGui::BeginChild("##FileDialog_FileList", size);

    if (ImGui::Selectable("..", false)) {
        fi.dirUp();
    }

    for (const auto& entry: fi.entries()) {
        std::string str;
        if (entry.type == DT_DIR) {
            str = "[Dir]";
        } else {
            str = "[File]";
        }
        str += " ";
        str += entry.name;
        if (ImGui::Selectable(str.c_str(), fi.filename() == entry.name))
        {
            if (entry.type == DT_DIR) {
                fi.setDir(entry.name);
                break;
            } else {
                fi.setFilename(entry.name);
            }
        }
    }

    ImGui::EndChild();

    ImGui::Text("File Name : ");

    ImGui::SameLine();
	
    float width = ImGui::GetContentRegionAvailWidth();
	//if (vFilters != 0) width -= 120.0f;
	ImGui::PushItemWidth(width);
	ImGui::Text("%s", fi.filename().c_str());
	ImGui::PopItemWidth();

    bool result = false;
    if (ImGui::Button("Cancel")) {
        fi.setConfirmed(false);
        result = true;
    }
    ImGui::SameLine();
    if(ImGui::Button("OK")) {
        fi.setConfirmed(true);
        result = true;
    }
    ImGui::End();

    return result;
}

