#include "utils.h"
#include <fstream>
#include <iostream>

std::vector<unsigned char> read_binary_file(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::cout << "JPEG size: " << size << "\n";

    std::vector<unsigned char> buffer(size);
    file.read((char*)buffer.data(), size);

    return buffer;
}

void write_binary_file(const std::vector<unsigned char>& buf,
                       const std::string& filename)
{
    std::ofstream file(filename, std::ios::binary | std::ios::trunc);
    file.write((char*)buf.data(), buf.size());
}

std::string read_text_file(const std::string& filename)
{
    std::ifstream ifs(filename);
    return std::string(std::istreambuf_iterator<char>(ifs),
                        std::istreambuf_iterator<char>());        
}

