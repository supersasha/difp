#ifndef __SPLINE_H__
#define __SPLINE_H__

// see https://github.com/josnidhin/CubicSpline

#include <vector>

class Spline
{
public:
    typedef std::vector<float> vec;
    struct S
    {
        float a, b, c, d, x;
    };

    Spline(const vec& xs, const vec& ys)
    {
        auto n = ys.size() - 1;
        vec a(ys);
        vec b(n);
        vec d(n);
        vec h(n);
        for(auto i = 0; i < n; i++)
            h[i] = xs[i+1] - xs[i];
        vec al(n);
        for(auto i = 1; i < n; i++)
            al[i] = 3/h[i]*(a[i+1] - a[i]) - 3/h[i - 1]*(a[i] - a[i-1]);
        vec c(n+1);
        vec l(n+1);
        vec mu(n+1);
        vec z(n+1);
        l[0] = 1;
        mu[0] = z[0] = 0;
        for(auto i = 1; i < n; i++) {
            l[i] = 2*(xs[i+1] - xs[i-1]) - h[i-1]*mu[i-1];
            mu[i] = h[i] / l[i];
            z[i] = (al[i] - h[i-1]*z[i-1]) / l[i];
        }
        l[n] = 1;
        z[n] = c[n] = 0;
        for(int j = n - 1; j >= 0; j--) {
            c[j] = z[j] - mu[j]*c[j+1];
            b[j] = (a[j+1] - a[j]) / h[j] - h[j] * (c[j+1] + 2*c[j]) / 3;
            d[j] = (c[j+1] - c[j]) / (3 * h[j]);
        }
        m_data.reserve(n);
        for(auto i = 0; i < n; i++) {
            m_data.emplace_back(S{a[i], b[i], c[i], d[i], xs[i]});
        }
    }

    float approx(float x) const
    {
        for(auto it = m_data.rbegin(); it != m_data.rend(); it++) {
            if(x >= it->x) {
                auto dx = x - it->x;
                return ((((it->d * dx) + it->c) * dx) + it->b) * dx + it->a;
            }
        }
        return m_data[0].a;
    }

    const std::vector<S>& data()
    {
        return m_data;
    }
private:
    std::vector<S> m_data;
};

#endif
