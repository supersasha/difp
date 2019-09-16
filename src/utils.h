#pragma once

#include <vector>
#include <string>

struct IntSize
{
    int width = 0;
    int height = 0;
};

struct FloatSize
{
    float width = 0;
    float height = 0;
};

std::vector<unsigned char> read_binary_file(const std::string& filename);
void write_binary_file(const std::vector<unsigned char>&, const std::string& filename);

std::string read_text_file(const std::string& filename);
