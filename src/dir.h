#pragma once

#include <string>
#include <vector>
#include <algorithm>

#include <dirent.h>

class Dir
{
public:
    Dir(const std::string& path)
        : m_path(path)
    {}

    std::vector<std::string> files() const
    {
        DIR * dir = opendir(m_path.c_str());
        dirent * entry;

        std::vector<std::string> filenames;
        while (entry = readdir(dir)) {
            if (entry->d_name[0] == '.') {
                continue;
            }
            if (entry->d_type == DT_REG) {
                filenames.emplace_back(entry->d_name);
            }
        }
        closedir(dir);
        std::sort(filenames.begin(), filenames.end());
        return filenames;
    }
private:
    std::string m_path;
};

