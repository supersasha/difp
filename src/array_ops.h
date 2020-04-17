#pragma once

#include <iostream>
#include "arrays.h"

template <size_t N>
std::ostream& operator<<(std::ostream& os, const Array<N>& a)
{
    os << "[ ";
    for (int i = 0; i < N; i++) {
        if (i > 0) {
            os << ", ";
        }
        os << a[i];
    }
    os << " ]";
    return os;
}

template <size_t N, size_t M>
std::ostream& operator<<(std::ostream& os, const Array2D<N, M>& a)
{
    os << "[ ";
    for (int i = 0; i < N; i++) {
        if (i > 0) {
            os << ", ";
        }
        os << a[i];
    }
    os << " ]";
    return os;
}

/*
 * Array
 */

template <size_t N, size_t K, size_t L>
Array<L> subarray(const Array<N>& a)
{
    Array<L> r;
    for (int i = 0; i < L; i++) {
        r[i] = a[K + i];
    }
    return r;
}

template <size_t N>
float sum(const Array<N>& a)
{
    float s = 0;
    for (int i = 0; i < N; i++) {
        s += a[i];
    }
    return s;
}

template <size_t N>
Array<N> clip(const Array<N>& a1, float lo, float hi)
{
    Array<N> a;
    for (int i = 0; i < N; i++) {
        if (a1[i] < lo) {
            a[i] = lo;
        } else if (a1[i] > hi) {
            a[i] = hi;
        } else {
            a[i] = a1[i];
        }
    }
    return a;
}

template <size_t N>
Array<N> ones()
{
    Array<N> a;
    for (int i = 0; i < N; i++) {
        a[i] = 1.0;
    }
    return a;
}

template <size_t N>
Array<N> zeros()
{
    Array<N> a;
    for (int i = 0; i < N; i++) {
        a[i] = 0.0;
    }
    return a;
}

template <typename D, size_t N>
Array<N> array_from_ptr(const D * d)
{
    Array<N> a;
    for (int i = 0; i < N; i++) {
        a[i] = d[i];
    }
    return a;
}

template <size_t N>
Array<N> operator-(const Array<N>& a1)
{
    Array<N> a;
    for (int i = 0; i < N; i++) {
        a[i] = -a1[i];
    }
    return a;
}

/*
 * Array, Scalar
 */

template<size_t N>
Array<N> operator*(const Array<N>& a1, float d)
{
    Array<N> a;
    for (int i = 0; i < N; i++) {
        a[i] = a1[i] * d;
    }
    return a;
}

template<size_t N>
Array<N> operator*(float d, const Array<N>& a1)
{
    return a1 * d;
}

template<size_t N>
Array<N> operator/(const Array<N>& a1, float d)
{
    return a1 * (1 / d);
}

template<size_t N>
Array<N> operator/(float d, const Array<N>& a1)
{
    Array<N> a;
    for (int i = 0; i < N; i++) {
        a[i] = d / a1[i];
    }
    return a;
}

template<size_t N>
Array<N> operator+(const Array<N>& a1, float d)
{
    Array<N> a;
    for (int i = 0; i < N; i++) {
        a[i] = a1[i] + d;
    }
    return a;
}

template<size_t N>
Array<N> operator+(float d, const Array<N>& a1)
{
    return a1 + d;
}

template<size_t N>
Array<N> operator-(const Array<N>& a1, float d)
{
    return a1 + (-d);
}

template<size_t N>
Array<N> operator-(float d, const Array<N>& a1)
{
    return -(a1 - d);
}

/*
 * Array, Array
 */

template<size_t N>
Array<N> operator+(const Array<N>& a1, const Array<N>& a2)
{
    Array<N> a;
    for (int i = 0; i < N; i++) {
        a[i] = a1[i] + a2[i];
    }
    return a;
}

template<size_t N>
Array<N> operator-(const Array<N>& a1, const Array<N>& a2)
{
    Array<N> a;
    for (int i = 0; i < N; i++) {
        a[i] = a1[i] - a2[i];
    }
    return a;
}

// Dot product
template<size_t N>
float operator%(const Array<N>& a1, const Array<N>& a2)
{
    float r = 0;
    for (int i = 0; i < N; i++) {
        r += a1[i] * a2[i];
    }
    return r;
}

template<size_t N>
Array<N> operator*(const Array<N>& a1, const Array<N>& a2)
{
    Array<N> a;
    for (int i = 0; i < N; i++) {
        a[i] = a1[i] * a2[i];
    }
    return a;
}

template<size_t N>
Array<N> operator/(const Array<N>& a1, const Array<N>& a2)
{
    Array<N> a;
    for (int i = 0; i < N; i++) {
        a[i] = a1[i] / a2[i];
    }
    return a;
}

// Unary function on Array
template<size_t N, typename F>
Array<N> apply(const F& f, const Array<N>& a1)
{
    Array<N> a;
    for (int i = 0; i < N; i++) {
        a[i] = f(a1[i]);
    }
    return a;
}

// Binary function on Array
template<size_t N, typename F>
Array<N> apply(const F& f, const Array<N>& a1, const Array<N>& a2)
{
    Array<N> a;
    for (int i = 0; i < N; i++) {
        a[i] = f(a1[i], a2[i]);
    }
    return a;
}

template<size_t N, typename F>
Array<N> apply(const F& f, const Array<N>& a1, float d)
{
    Array<N> a;
    for (int i = 0; i < N; i++) {
        a[i] = f(a1[i], d);
    }
    return a;
}

template<size_t N, typename F>
Array<N> apply(const F& f, float d, const Array<N>& a1)
{
    Array<N> a;
    for (int i = 0; i < N; i++) {
        a[i] = f(d, a1[i]);
    }
    return a;
}

/*
 * Array2D
 */

template<size_t N, size_t M>
Array2D<N, M> operator-(const Array2D<N, M>& a1)
{
    Array2D<N, M> a;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            a[i][j] = -a1[i][j];
        }
    }
    return a;
}

template<size_t N, size_t M>
Array2D<M, N> transpose(const Array2D<N, M>& a1)
{
    Array2D<M, N> a;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            a[j][i] = a1[i][j];
        }
    }
    return a;
}

template <size_t N, size_t M>
Array2D<M, N> operator~(const Array2D<N, M>& a)
{
    return transpose(a);
}

template<size_t N, size_t M>
Array2D<N, M> operator+(const Array2D<N, M>& a1, const Array2D<N, M>& a2)
{
    Array2D<N, M> a;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            a[i][j] = a1[i][j] + a2[i][j];
        }
    }
    return a;
}

template<size_t N, size_t M>
Array2D<N, M> operator+(const Array2D<N, M>& m, const Array<M>& v)
{
    Array2D<N, M> a;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            a[i] = m[i] + v;
        }
    }
    return a;
}

template<size_t N, size_t M>
Array2D<N, M> operator-(const Array2D<N, M>& a1, const Array2D<N, M>& a2)
{
    Array2D<N, M> a;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            a[i][j] = a1[i][j] - a2[i][j];
        }
    }
    return a;
}

template<size_t N, size_t M>
Array2D<N, M> operator*(const Array2D<N, M>& a1, const Array2D<N, M>& a2)
{
    Array2D<N, M> a;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            a[i][j] = a1[i][j] * a2[i][j];
        }
    }
    return a;
}

template<size_t N, size_t M>
Array2D<N, M> operator*(const Array2D<N, M>& m, const Array<M>& v)
{
    Array2D<N, M> a;
    for (int i = 0; i < N; i++) {
        a[i] = m[i] * v;
    }
    return a;
}

template<size_t N, size_t M>
Array2D<N, M> operator*(const Array<N>& v, const Array2D<N, M>& m)
{
    Array2D<N, M> a;
    for (int i = 0; i < N; i++) {
        a[i] = m[i] * v[i];
    }
    return a;
}

template<size_t N, size_t M>
Array2D<N, M> operator/(const Array2D<N, M>& a1, const Array2D<N, M>& a2)
{
    Array2D<N, M> a;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            a[i][j] = a1[i][j] / a2[i][j];
        }
    }
    return a;
}

template<size_t N, size_t M>
Array<N> operator%(const Array2D<N, M>& m, const Array<M>& a1)
{
    Array<N> a;
    for (int i = 0; i < N; i++) {
        a[i] = m[i] % a1;
    }
    return a;
}

template<size_t N, size_t M>
Array<M> operator%(const Array<N>& a1, const Array2D<N, M>& m)
{
    return transpose(m) * a1;
}

template <typename F, size_t N, size_t M>
Array2D<N, M> apply(const F& f, const Array2D<N, M>& a1)
{
    Array2D<N, M> a;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            a[i][j] = f(a1[i][j]);
        }
    }
    return a;
}

template <typename F, size_t N, size_t M>
Array2D<N, M> apply(const F& f, float d, const Array2D<N, M>& a1)
{
    Array2D<N, M> a;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            a[i][j] = f(d, a1[i][j]);
        }
    }
    return a;
}

template <typename F, size_t N, size_t M>
Array2D<N, M> apply(const F& f, const Array2D<N, M>& a1, float d)
{
    Array2D<N, M> a;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            a[i][j] = f(a1[i][j], d);
        }
    }
    return a;
}

template <typename F, size_t N, size_t M>
Array2D<N, M> apply(const F& f, const Array2D<N, M>& a1, const Array2D<N, M>& a2)
{
    Array2D<N, M> a;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            a[i][j] = f(a1[i][j], a2[i][j]);
        }
    }
    return a;
}

