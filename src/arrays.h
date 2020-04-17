#pragma once

#include <array>

template<size_t N> using Array = std::array<float, N>;
template<size_t ROWS, size_t COLS> using Array2D =
    std::array<std::array<float, COLS>, ROWS>;
template<size_t N, size_t ROWS, size_t COLS> using Array3D =
    std::array<std::array<std::array<float, COLS>, ROWS>, N>;
