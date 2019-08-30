#pragma once

#include <vector>
#include <string>

std::vector<unsigned char> read_binary_file(const std::string& filename);
std::string read_text_file(const std::string& filename);
