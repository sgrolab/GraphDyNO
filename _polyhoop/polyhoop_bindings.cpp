/*
Adapted from PolyHoop (https://doi.org/10.17632/4jscxhkd2s.1).
Original copyright notice, retained as the license requires:

PolyHoop
Copyright (c) 2023-2024 Roman Vetter
vetterro@ethz.ch
ETH Zurich

3-Clause BSD License

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Python bindings for PolyHoop
// Using pybind11 with OpenMP parallelization
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <set>
#include <random>
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Forward declarations
struct Vertex;
struct Polygon;
struct Ensemble;

// Simulation parameters (mutable for Python access)
struct Parameters {
    double h = 0.01;          // [L] edge thickness
    double lmin = 0.02;       // [L] minimum edge length
    double lmax = 0.2;        // [L] maximum edge length
    double C = 0.9;           // [-] circularity index
    double alpha_mu = 0;      // [L^2/T] mean area growth rate
    double alpha_CV = 0;      // [-] coefficient of variation of area growth rate
    double beta = 0.9;        // [-] minimum area fraction for growth
    double Amin = 0;          // [L^2] minimum area
    double Amax_mu = INFINITY;// [L^2] mean maximum area
    double Amax_CV = 0;       // [-] coefficient of variation of maximum area
    double gam = 1e3;         // [L/T^2] line tension per vertex mass
    double ka = 1e5;          // [1/(L^2*T^2)] area stiffness per vertex mass
    double kl = 1e4;          // [L/T^2] edge contractility stiffness per vertex mass
    double kb = 0;            // [L^3/T^2] bending stiffness per vertex mass
    double kr = 1e7;          // [1/T^2] repulsion stiffness per vertex mass
    double kh = 1e6;          // [1/T^2] adhesion stiffness per vertex mass
    double sh = 0.01;         // [L] adhesion hardening zone size
    double ss = 0.01;         // [L] adhesion softening zone size
    double theta = 0;         // [-] fusion threshold
    double mu = 0;            // [-] dynamic friction coefficient
    double rho = 0;           // [1/L^2] fluid mass density per vertex mass
    double g = 0;             // [L/T^2] gravitational acceleration
    double gl = 0;            // [L/T^2] edge gravitational acceleration
    double cv = 10;           // [1/T] viscous damping rate
    double cd = 0;            // [-] drag coefficient
    double cc = 30;           // [1/T] collision damping rate
    double dt = 1e-4;         // [T] time step
    double fp = 50;           // [L/T^2] polarity force per vertex mass
    double Dr = 0.5;          // [1/T] rotational diffusion coefficient
    double elongation = 0.5;  // [-] elongation factor
    std::size_t Nr = 0;       // number of rigid polygons
    bool oscillatory_growth = false;  // use oscillatory (true) or linear (false) growth model
    double position_noise_std = 0;  // [L/T^(1/2)] position noise strength (0 = disabled)
    double compress_force = 0;      // [1/T^2] spring constant for radial compression toward center (0 = disabled)
    double compress_cx = 0;         // [L] x-coordinate of compression center
    double compress_cy = 0;         // [L] y-coordinate of compression center
    double simulation_start_time = 0;  // [T] reference time for gam/kl exponential ramps; ramps start from t=simulation_start_time
    // Axis-aligned virtual walls. Active when xmax > xmin AND ymax > ymin.
    // Walls apply the same force formulas as polygon-polygon interaction
    // (kr repulsion, optional wall_kh adhesion, cc damping, mu friction)
    // against each vertex, with the inward normal as the force direction.
    double wall_xmin = 0;
    double wall_xmax = 0;
    double wall_ymin = 0;
    double wall_ymax = 0;
    double wall_kh = 0;  // wall adhesion stiffness (0 = repulsion-only)

    double drmax() const { return h + sh + ss; }
};

// Global parameters instance
static Parameters params;

// Random number generator
static std::mt19937 rng;

struct Point {
    double x, y;

    Point(double x_ = 0, double y_ = 0) : x(x_), y(y_) {}

    Point operator+(const Point& r) const { return {x + r.x, y + r.y}; }
    Point operator-(const Point& r) const { return {x - r.x, y - r.y}; }
    Point cross() const { return {-y, x}; }
    double dot(const Point& r) const { return x * r.x + y * r.y; }
    double wedge(const Point& r) const { return x * r.y - y * r.x; }
    double length2() const { return x * x + y * y; }
    double length() const { return std::sqrt(length2()); }

    void add(double a, const Point& r) {
        #pragma omp atomic
        x += a * r.x;
        #pragma omp atomic
        y += a * r.y;
    }

    Point operator*(double a) const { return {a * x, a * y}; }

    std::string repr() const {
        return "Point(" + std::to_string(x) + ", " + std::to_string(y) + ")";
    }
};

Point operator*(double a, const Point& r) { return {a * r.x, a * r.y}; }

double point_edge_dist2(const Point& r0, const Point& r1, const Point& r2, double& xi, double& xit, Point& dr) {
    const Point r12 = r2 - r1;
    const Point r10 = r0 - r1;
    xi = r10.dot(r12) / r12.dot(r12);
    xit = std::min(std::max(xi, 0.), 1.);
    dr = r10 - xit * r12;
    return dr.dot(dr);
}

struct Vertex {
    Point r, v, a;      // position, velocity, acceleration
    std::size_t p;      // polygon index
    Vertex* next;       // pointer to next vertex in same box
    double l0;          // rest length of edge to the right

    Vertex() : r(), v(), a(), p(0), next(nullptr), l0(0) {}
    Vertex(Point r_, Point v_, Point a_, std::size_t p_)
        : r(r_), v(v_), a(a_), p(p_), next(nullptr), l0(0) {}
};

struct Polygon {
    std::vector<Vertex> vertices;
    bool phase;
    double A0, A, Amax, alpha;
    double area_fluct_period, area_fluct_amp;  // per-cell area fluctuation
    double area_fluct_phase;  // per-cell area oscillation phase offset [T]
    double gam;  // per-cell line tension (0 = use global params.gam)
    double gam_fluct_period, gam_fluct_amp;  // per-cell line tension fluctuation
    double gam_fluct_phase;  // per-cell line tension oscillation phase offset [T]
    double gam_rate;  // per-cell exponential growth rate for gam [1/T]
    double kl;  // per-cell elastic tension (0 = use global params.kl)
    double kl_fluct_period, kl_fluct_amp;  // per-cell elastic tension fluctuation
    double kl_fluct_phase;  // per-cell elastic tension oscillation phase offset [T]
    double kl_rate;  // per-cell exponential growth rate for kl [1/T]
    double kh;  // per-cell adhesion strength (0 = use global params.kh)
    double kh_fluct_period, kh_fluct_amp;  // per-cell adhesion fluctuation
    double kh_fluct_phase;  // per-cell adhesion oscillation phase offset [T]
    double kh_rate;  // per-cell exponential growth rate for kh [1/T]
    double elongation;  // per-cell elongation factor
    Point polarity;

    Polygon() : phase(false), A0(0), A(0), Amax(INFINITY), alpha(0),
                area_fluct_period(0), area_fluct_amp(0), area_fluct_phase(0),
                gam(0), gam_fluct_period(0), gam_fluct_amp(0), gam_fluct_phase(0),
                gam_rate(0),
                kl(0), kl_fluct_period(0), kl_fluct_amp(0), kl_fluct_phase(0),
                kl_rate(0),
                kh(0), kh_fluct_period(0), kh_fluct_amp(0), kh_fluct_phase(0),
                kh_rate(0),
                elongation(0.5), polarity{1, 0} {}

    double area() {
        A = 0;
        for (std::size_t i = vertices.size() - 1, j = 0; j < vertices.size(); i = j++)
            A += vertices[i].r.wedge(vertices[j].r);
        return A /= 2;
    }

    double perimeter() const {
        double L = 0;
        for (std::size_t i = vertices.size() - 1, j = 0; j < vertices.size(); i = j++)
            L += (vertices[j].r - vertices[i].r).length();
        return L;
    }

    double perimeter0() const {
        double L0 = 0;
        for (const auto& v : vertices)
            L0 += v.l0;
        return L0;
    }

    bool contains(const Point& r) const {
        bool in = false;
        for (std::size_t i = vertices.size() - 1, j = 0; j < vertices.size(); i = j++)
            if ((vertices[i].r.y > r.y) != (vertices[j].r.y > r.y) &&
                r.x < (vertices[i].r.x - vertices[j].r.x) * (r.y - vertices[j].r.y) /
                      (vertices[i].r.y - vertices[j].r.y) + vertices[j].r.x)
                in = !in;
        return in;
    }

    Point centroid() const {
        Point c{0, 0};
        double A6 = 0;
        for (std::size_t i = vertices.size() - 1, j = 0; j < vertices.size(); i = j++) {
            const double w = vertices[i].r.wedge(vertices[j].r);
            c.x += (vertices[i].r.x + vertices[j].r.x) * w;
            c.y += (vertices[i].r.y + vertices[j].r.y) * w;
            A6 += w;
        }
        return {c.x / (3 * A6), c.y / (3 * A6)};
    }

    void init_polarity() {
        static std::uniform_real_distribution<> uni_dist;
        const double angle = 2 * M_PI * uni_dist(rng);
        polarity = {std::cos(angle), std::sin(angle)};
    }

    void rotate_polarity(double dtheta) {
        const double c = std::cos(dtheta), s = std::sin(dtheta);
        polarity = {c * polarity.x - s * polarity.y, s * polarity.x + c * polarity.y};
    }

    std::size_t num_vertices() const { return vertices.size(); }

    std::vector<std::pair<double, double>> get_vertex_positions() const {
        std::vector<std::pair<double, double>> positions;
        positions.reserve(vertices.size());
        for (const auto& v : vertices)
            positions.emplace_back(v.r.x, v.r.y);
        return positions;
    }
};

class Ensemble {
public:
    std::vector<Polygon> polygons;
    std::vector<Vertex*> first;
    std::size_t Nx, Ny;
    double x0, y0, bs;
    double t;

    Ensemble() : Nx(0), Ny(0), x0(0), y0(0), bs(0), t(0) {}

    Ensemble(const std::string& name) : t(0) {
        load_off(name);
    }

    void load_off(const std::string& name) {
        std::ifstream file(name);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + name);
        }

        file.ignore(3);
        std::size_t Nv, Np, Ne;
        file >> Nv >> Np >> Ne;

        std::vector<Point> points(Nv);
        std::vector<int> z(Nv);
        for (std::size_t i = 0; i < Nv; ++i)
            file >> points[i].x >> points[i].y >> z[i];

        // Distribution setup
        const double Amax_lnCV = std::log(1 + params.Amax_CV * params.Amax_CV);
        const double alpha_lnCV = std::log(1 + params.alpha_CV * params.alpha_CV);
        std::lognormal_distribution<> Amax_dist(std::log(params.Amax_mu) - Amax_lnCV/2, std::sqrt(Amax_lnCV));
        std::lognormal_distribution<> alpha_dist(std::log(params.alpha_mu) - alpha_lnCV/2, std::sqrt(alpha_lnCV));

        polygons.resize(Np);
        for (std::size_t p = 0, j = 0; p < Np; ++p) {
            file >> Nv;
            for (std::size_t i = 0; i < Nv; ++i) {
                file >> j;
                polygons[p].vertices.push_back({points[j], {0, 0}, {0, 0}, p});
            }
            polygons[p].phase = std::abs(z[j]) % 2;
            polygons[p].A0 = polygons[p].area();
            polygons[p].Amax = std::isinf(params.Amax_mu) ? INFINITY : Amax_dist(rng);
            polygons[p].alpha = params.alpha_mu == 0 ? 0 : alpha_dist(rng);
            polygons[p].init_polarity();
            for (std::size_t i = Nv - 1, k = 0; k < Nv; i = k++)
                polygons[p].vertices[i].l0 = (polygons[p].vertices[k].r - polygons[p].vertices[i].r).length();
        }
    }

    void remove(std::size_t p) {
        polygons[p] = polygons.back();
        for (auto& v : polygons[p].vertices)
            v.p = p;
        polygons.pop_back();
    }

    void boxes() {
        double xmin = polygons[0].vertices[0].r.x, xmax = xmin;
        double ymin = polygons[0].vertices[0].r.y, ymax = ymin;
        double l2max = 0;

        #pragma omp parallel for reduction(min:xmin,ymin) reduction(max:xmax,ymax,l2max)
        for (std::size_t p = 0; p < polygons.size(); ++p) {
            for (std::size_t i = polygons[p].vertices.size() - 1, j = 0; j < polygons[p].vertices.size(); i = j++) {
                if (polygons[p].vertices[i].r.x < xmin) xmin = polygons[p].vertices[i].r.x;
                else if (polygons[p].vertices[i].r.x > xmax) xmax = polygons[p].vertices[i].r.x;
                if (polygons[p].vertices[i].r.y < ymin) ymin = polygons[p].vertices[i].r.y;
                else if (polygons[p].vertices[i].r.y > ymax) ymax = polygons[p].vertices[i].r.y;
                l2max = std::max(l2max, (polygons[p].vertices[j].r - polygons[p].vertices[i].r).length2());
            }
        }
        x0 = xmin;
        y0 = ymin;

        bs = std::sqrt(l2max) + params.drmax();
        Nx = (xmax - x0) / bs + 3;
        Ny = (ymax - y0) / bs + 3;
        first.assign(Nx * Ny, nullptr);

        for (auto& p : polygons) {
            for (auto& v : p.vertices) {
                const std::size_t bx = (v.r.x - x0) / bs + 1;
                const std::size_t by = (v.r.y - y0) / bs + 1;
                const std::size_t b = bx * Ny + by;
                v.next = first[b];
                first[b] = &v;
            }
        }
    }

    void interaction(Vertex* v0, Vertex* v1, Vertex* v1n0, Vertex* v1n1) {
        Vertex* v1n[2] = {v1n0, v1n1};
        double xi[2], xit[2], dr2[2];
        Point dr[2];

        for (unsigned int j = 0; j < 2; ++j)
            dr2[j] = point_edge_dist2(v0->r, v1->r, v1n[j]->r, xi[j], xit[j], dr[j]);
        const unsigned int j = dr2[0] > dr2[1];

        const double drmax = params.drmax();
        if (xi[j] <= 1 && dr2[j] < drmax * drmax) {
            Vertex* v2 = v1n[j];

            if (params.lmin < params.h && v0->p == v1->p) {
                auto& v = polygons[v1->p].vertices;
                const std::size_t j1 = v1 - v.data(), j2 = v2 - v.data();
                for (unsigned int d = 0; d < 2; ++d) {
                    double l = 0;
                    for (std::size_t i = v0 - v.data(), k; l <= params.h && i != j1 && i != j2; i = k) {
                        k = (i + v.size() + 2*d-1) % v.size();
                        l += (v[i].r - v[k].r).length();
                    }
                    if (l <= params.h) return;
                }
            }

            const double dr_abs = std::sqrt(dr2[j]);
            const Point dv = v0->v - v1->v - xit[j] * (v2->v - v1->v);
            const double dndv_dr = dr[j].dot(dv) / dr2[j];
            double da_dr = -params.cc * dndv_dr;

            if (dr_abs < params.h)
                da_dr += params.kr * (params.h / dr_abs - 1);
            else if (v0->p != v1->p) {
                // Per-cell adhesion with geometric-mean pairing. Each cell's
                // kh is computed independently (with optional exponential ramp
                // and oscillation) and the pair coupling is sqrt(kh_a * kh_b).
                // AM-GM guarantees W_AA + W_BB >= 2*W_AB, so cells with higher
                // kh form clusters surrounded by lower-kh cells (Steinberg).
                // If either cell has kh==0, fall back to global params.kh.
                auto cell_kh = [&](std::size_t pi) {
                    double k = polygons[pi].kh;
                    if (k == 0) return params.kh;
                    if (polygons[pi].kh_rate != 0) {
                        k *= std::exp(polygons[pi].kh_rate
                                      * (t - params.simulation_start_time));
                    }
                    if (polygons[pi].kh_fluct_period != 0) {
                        k += polygons[pi].kh_fluct_amp
                             * std::sin(2 * M_PI
                                        * (t - polygons[pi].kh_fluct_phase)
                                        / polygons[pi].kh_fluct_period);
                    }
                    return k;
                };
                const double kh_a = cell_kh(v0->p);
                const double kh_b = cell_kh(v1->p);
                const double kh_eff = (kh_a > 0 && kh_b > 0)
                                          ? std::sqrt(kh_a * kh_b)
                                          : 0.0;
                if (dr_abs < params.h + params.sh)
                    da_dr += kh_eff * (params.h / dr_abs - 1);
                else
                    da_dr += kh_eff * params.sh / params.ss * (1 - drmax / dr_abs);
            }
            Point a = da_dr * dr[j];

            const Point dvt = dv - dndv_dr * dr[j];
            const double dvt_abs = dvt.length();
            if (dvt_abs > 0 && v0->p != v1->p)
                a.add(-params.mu * std::abs(da_dr) * dr_abs / dvt_abs, dvt);

            v0->a.add(1, a);
            v1->a.add(xit[j]-1, a);
            v2->a.add(-xit[j], a);
        }
    }

    void step() {
        // Polygon removal and division
        for (std::size_t p = params.Nr; p < polygons.size(); ++p) {
            auto& v = polygons[p].vertices;

            if (polygons[p].A < params.Amin || v.size() < 3)
                remove(p--);
            else if (polygons[p].A > polygons[p].Amax && v.size() > 5) {
                // Division logic (simplified for bindings)
                divide_polygon(p);
            }
        }

        // Refine/coarsen and compute accelerations
        #pragma omp parallel for
        for (std::size_t p = 0; p < polygons.size(); ++p) {
            auto& v = polygons[p].vertices;

            // Refine or coarsen
            for (std::size_t i = v.size() - 1, k = i - 1, j = 0; j < v.size(); ) {
                const double l2 = (v[j].r - v[i].r).length2();
                if (l2 > params.lmax * params.lmax) {
                    const Point rnew = 0.5 * (v[i].r + v[j].r);
                    const Point vnew = 0.5 * (v[i].v + v[j].v);
                    v[i].l0 /= 2;
                    v.insert(v.begin() + j, {rnew, vnew, {0, 0}, p});
                    v[j].l0 = v[i].l0;
                    if (k > j) ++k;
                    if (i > j) ++i;
                } else if (l2 < params.lmin * params.lmin && p >= params.Nr && v.size() > 3) {
                    v[k].l0 += v[i].l0 / 2;
                    v[i].l0 = v[i].l0 / 2 + v[j].l0;
                    v[i].r = 0.5 * (v[i].r + v[j].r);
                    v[i].v = 0.5 * (v[i].v + v[j].v);
                    v.erase(v.begin() + j);
                    if (k > j) --k;
                    if (i > j) --i;
                } else {
                    k = i;
                    i = j++;
                }
            }

            // Compute accelerations
            polygons[p].area();
            const double ls = polygons[p].perimeter0() / std::sqrt(4 * M_PI * polygons[p].A0 / params.C);

            // Compression force: uniform per-vertex force pointing from the
            // cell centroid toward the compression center (rigid translation).
            Point compress_acc{0, 0};
            if (params.compress_force != 0) {
                const Point c = polygons[p].centroid();
                compress_acc = Point{params.compress_cx - c.x, params.compress_cy - c.y};
                const double dlen = compress_acc.length();
                if (dlen > 0) {
                    compress_acc = (params.compress_force / dlen) * compress_acc;
                }
            }

            for (std::size_t i = v.size() - 1, k = i - 1, j = 0; j < v.size(); k = i, i = j++) {
                const Point e1 = v[i].r - v[k].r;
                const Point e2 = v[j].r - v[i].r;
                const double l1 = e1.length();
                const double l2 = e2.length();
                const Point n = (e1 + e2).cross();

                v[i].a = {0, -params.gl};
                v[i].a.add(params.fp, polygons[p].polarity);
                v[i].a.add(params.ka / 2 * (polygons[p].A - polygons[p].A0), n);
                // Per-cell elastic tension with optional exponential ramp
                // and oscillation. Same simulation_start_time reference as gam.
                double kl_eff = polygons[p].kl;
                if (polygons[p].kl_rate != 0) {
                    kl_eff *= std::exp(polygons[p].kl_rate * (t - params.simulation_start_time));
                }
                if (polygons[p].kl_fluct_period != 0) {
                    kl_eff += polygons[p].kl_fluct_amp
                              * std::sin(2 * M_PI * (t - polygons[p].kl_fluct_phase)
                                         / polygons[p].kl_fluct_period);
                }
                v[i].a.add(-kl_eff, (ls / v[k].l0 - 1 / l1) * e1 - (ls / v[i].l0 - 1 / l2) * e2);

                // Anisotropic line tension (per-cell gam, with optional
                // exponential ramp and oscillation). gam_eff(t) =
                // gam * exp(gam_rate * (t - simulation_start_time)). Python
                // sets simulation_start_time at the start of recording so the
                // ramp clock effectively starts at 0 when data capture begins.
                double gam_eff = polygons[p].gam;
                if (polygons[p].gam_rate != 0) {
                    gam_eff *= std::exp(polygons[p].gam_rate * (t - params.simulation_start_time));
                }
                if (polygons[p].gam_fluct_period != 0) {
                    gam_eff += polygons[p].gam_fluct_amp
                               * std::sin(2 * M_PI * (t - polygons[p].gam_fluct_phase)
                                          / polygons[p].gam_fluct_period);
                }
                const Point& pol = polygons[p].polarity;
                const double align1 = e1.dot(pol) / l1;
                const double align2 = e2.dot(pol) / l2;
                const double gam1 = gam_eff * (1 + polygons[p].elongation * (1 - 2 * align1 * align1));
                const double gam2 = gam_eff * (1 + polygons[p].elongation * (1 - 2 * align2 * align2));
                v[i].a.add(-1, (gam1 / l1) * e1 - (gam2 / l2) * e2);

                // Uniform compression force (rigid translation toward center)
                v[i].a.add(1.0, compress_acc);

                v[i].a.add(params.rho * params.g / 6 * (2*polygons[p].phase-1),
                          (v[k].r.y + v[i].r.y + v[j].r.y) * n - Point{0, e1.wedge(e2)});
                v[i].a.add(-params.cv - params.rho * params.cd / 4 * std::abs(v[i].v.dot(n)), v[i].v);

                // Bending
                const Point e0 = v[k].r - v[(k + v.size() - 1) % v.size()].r;
                const Point e3 = v[(j + 1) % v.size()].r - v[j].r;
                const double l0 = e0.length();
                const double l3 = e3.length();
                const double b0 = l0 * l1 + e0.dot(e1);
                const double b1 = l1 * l2 + e1.dot(e2);
                const double b2 = l2 * l3 + e2.dot(e3);
                const double a0 = e0.wedge(e1) / b0;
                const double a1 = e1.wedge(e2) / b1;
                const double a2 = e2.wedge(e3) / b2;
                v[i].a.add(-8 * params.kb, (a0 / ((l0 + l1) * b0)) * (e0.cross() - a0 * e0)
                          - (a1 / ((l1 + l2) * b1)) * (n - a1 * (e1 - e2))
                          + (a2 / ((l2 + l3) * b2)) * (e3.cross() + a2 * e3));
            }
        }

        // Polygon-polygon interaction
        boxes();
        #pragma omp parallel for
        for (std::size_t p = params.Nr; p < polygons.size(); ++p)
            for (std::size_t i = polygons[p].vertices.size() - 1, k = i - 1, j = 0;
                 j < polygons[p].vertices.size(); k = i, i = j++) {
                const std::size_t bxi = (polygons[p].vertices[i].r.x - x0) / bs + 1;
                const std::size_t byi = (polygons[p].vertices[i].r.y - y0) / bs + 1;
                for (std::size_t bxj = bxi - 1; bxj <= bxi + 1; ++bxj)
                    for (std::size_t byj = byi - 1; byj <= byi + 1; ++byj)
                        for (Vertex* v = first[bxj * Ny + byj]; v; v = v->next)
                            if (v != &polygons[p].vertices[k] && v != &polygons[p].vertices[i] &&
                                v != &polygons[p].vertices[j])
                                interaction(v, &polygons[p].vertices[i],
                                          &polygons[p].vertices[k], &polygons[p].vertices[j]);
            }

        // Virtual axis-aligned walls. Same kr/wall_kh/cc/mu formulas as
        // interaction(), but with the inward normal as the force direction
        // so the force always pushes vertices back into the box (regardless
        // of which side of the wall the vertex sits on).
        if (params.wall_xmax > params.wall_xmin
                && params.wall_ymax > params.wall_ymin) {
            const double drmax_w = params.drmax();
            #pragma omp parallel for
            for (std::size_t p = params.Nr; p < polygons.size(); ++p) {
                for (auto& v : polygons[p].vertices) {
                    // (signed_dist_to_wall, inward_normal). signed_dist > 0
                    // means the vertex is inside the box wrt this wall.
                    const std::pair<double, Point> walls[4] = {
                        {v.r.x - params.wall_xmin, Point{ 1.0,  0.0}},
                        {params.wall_xmax - v.r.x, Point{-1.0,  0.0}},
                        {v.r.y - params.wall_ymin, Point{ 0.0,  1.0}},
                        {params.wall_ymax - v.r.y, Point{ 0.0, -1.0}},
                    };
                    for (const auto& [signed_dist, inward] : walls) {
                        const double dr_abs = std::abs(signed_dist);
                        if (dr_abs >= drmax_w) continue;

                        double da_dr = 0;
                        if (dr_abs < params.h) {
                            da_dr += params.kr * (params.h / dr_abs - 1);
                        } else if (params.wall_kh > 0) {
                            if (dr_abs < params.h + params.sh)
                                da_dr += params.wall_kh
                                         * (params.h / dr_abs - 1);
                            else
                                da_dr += params.wall_kh * params.sh / params.ss
                                         * (1 - drmax_w / dr_abs);
                        }

                        const double v_normal =
                            v.v.x * inward.x + v.v.y * inward.y;
                        const double a_inward =
                            da_dr * dr_abs - params.cc * v_normal;
                        v.a.add(a_inward, inward);

                        if (params.mu != 0) {
                            const double tan_x = v.v.x - v_normal * inward.x;
                            const double tan_y = v.v.y - v_normal * inward.y;
                            const double tan_abs =
                                std::sqrt(tan_x * tan_x + tan_y * tan_y);
                            if (tan_abs > 0) {
                                const double mu_factor =
                                    -params.mu * std::abs(da_dr) * dr_abs
                                    / tan_abs;
                                v.a.add(mu_factor, Point{tan_x, tan_y});
                            }
                        }
                    }
                }
            }
        }

        // Time integration
        static std::normal_distribution<> normal_dist(0.0, 1.0);
        for (std::size_t p = params.Nr; p < polygons.size(); ++p) {
            for (auto& v : polygons[p].vertices) {
                v.v.add(params.dt, v.a);
                v.r.add(params.dt, v.v);
            }
            polygons[p].area();
            double alpha_eff = polygons[p].alpha;
            if (params.oscillatory_growth) {
                if (polygons[p].area_fluct_period != 0) {
                    // Oscillatory mode: alpha_eff = amp * 2π/period * cos(2π*t/period)
                    alpha_eff = polygons[p].area_fluct_amp * 2 * M_PI / polygons[p].area_fluct_period
                                * std::cos(2 * M_PI * (t - polygons[p].area_fluct_phase)
                                           / polygons[p].area_fluct_period);
                } else {
                    // Period = 0 means constant area (no growth, no oscillation)
                    alpha_eff = 0;
                }
            }
            // For oscillatory growth always integrate A0 so phases stay intact;
            // the linear-growth gate (A > beta*A0 or shrinking) would otherwise
            // freeze the oscillation whenever A lags below beta*A0, collapsing
            // per-cell phases into a single effective cycle.
            if (params.oscillatory_growth
                || polygons[p].A > params.beta * polygons[p].A0
                || alpha_eff < 0)
                polygons[p].A0 += alpha_eff * params.dt;

            if (params.fp > 0 && params.Dr > 0) {
                const double dtheta = std::sqrt(2 * params.Dr * params.dt) * normal_dist(rng);
                polygons[p].rotate_polarity(dtheta);
            }
            if (params.position_noise_std > 0) {
                const double dx = params.position_noise_std * normal_dist(rng);
                const double dy = params.position_noise_std * normal_dist(rng);
                const double sqrt_dt = std::sqrt(params.dt);
                for (auto& v : polygons[p].vertices) {
                    v.r.x += dx * sqrt_dt;
                    v.r.y += dy * sqrt_dt;
                }
            }
        }
        t += params.dt;
    }

    void divide_polygon(std::size_t p) {
        auto& v = polygons[p].vertices;

        // Compute centroid and inertia tensor
        Point c{0, 0};
        double Ixx = 0, Iyy = 0, Ixy = 0;
        for (std::size_t i = v.size() - 1, j = 0; j < v.size(); i = j++) {
            const Point rsum = v[i].r + v[j].r;
            const double w = v[i].r.wedge(v[j].r);
            c.add(w, rsum);
            Ixx += w * (rsum.y * rsum.y - v[i].r.y * v[j].r.y);
            Iyy += w * (rsum.x * rsum.x - v[i].r.x * v[j].r.x);
            Ixy += w * (rsum.x * rsum.y + v[i].r.x * v[i].r.y + v[j].r.x * v[j].r.y);
        }
        c = 1 / (6 * polygons[p].A) * c;
        Ixx = Ixx / 12 - c.y * c.y * polygons[p].A;
        Iyy = Iyy / 12 - c.x * c.x * polygons[p].A;
        Ixy = -Ixy / 24 + c.x * c.y * polygons[p].A;

        const double dI = (Ixx - Iyy) / 2;
        const double eig = dI + Iyy + std::sqrt(dI * dI + Ixy * Ixy);
        Point axis = (Ixx < Iyy ? Point{Ixy, eig - Ixx} : Point{eig - Iyy, Ixy});

        const double l = axis.length();
        if (l == 0) {
            static std::uniform_real_distribution<> uni_dist;
            const double angle = 2 * M_PI * uni_dist(rng);
            axis = {std::cos(angle), std::sin(angle)};
        } else {
            axis = (1 / l) * axis;
        }

        // Find division vertices
        std::size_t vend[4];
        for (unsigned int a = 0, i = v.size() - 1, j = 0; a < 4; i = j++) {
            const double d0 = axis.wedge(v[i].r - c);
            const double d1 = axis.wedge(v[j].r - c);
            if (d0 * d1 <= 0) {
                vend[a] = i;
                vend[a+1] = j;
                for (int d = 0; (v[vend[a+1]].r - v[vend[a]].r).length2() < params.h * params.h; d = 1 - d)
                    vend[a+d] = (vend[a+d] + v.size() + 2*d-1) % v.size();
                j = (vend[a+1] + 1) % v.size();
                a += 2;
            }
        }

        const double ls = polygons[p].perimeter() / polygons[p].perimeter0();
        v[vend[0]].l0 = (v[vend[3]].r - v[vend[0]].r).length() / ls;
        v[vend[2]].l0 = (v[vend[1]].r - v[vend[2]].r).length() / ls;

        // Distribution setup for new cells
        const double Amax_lnCV = std::log(1 + params.Amax_CV * params.Amax_CV);
        const double alpha_lnCV = std::log(1 + params.alpha_CV * params.alpha_CV);
        std::lognormal_distribution<> Amax_dist(std::log(params.Amax_mu) - Amax_lnCV/2, std::sqrt(Amax_lnCV));
        std::lognormal_distribution<> alpha_dist(std::log(params.alpha_mu) - alpha_lnCV/2, std::sqrt(alpha_lnCV));

        const double A0 = polygons[p].A0;
        std::vector<Vertex> vold;
        vold.swap(v);

        polygons[p] = Polygon();
        polygons[p].vertices.push_back(vold[vend[2]]);
        polygons[p].phase = vold[vend[2]].p < params.Nr ? false : polygons[vold[vend[2]].p].phase;
        polygons[p].Amax = std::isinf(params.Amax_mu) ? INFINITY : Amax_dist(rng);
        polygons[p].alpha = params.alpha_mu == 0 ? 0 : alpha_dist(rng);
        polygons[p].elongation = params.elongation;
        polygons[p].init_polarity();

        Polygon newpoly;
        newpoly.vertices.push_back(vold[vend[0]]);
        newpoly.phase = polygons[p].phase;
        newpoly.Amax = std::isinf(params.Amax_mu) ? INFINITY : Amax_dist(rng);
        newpoly.alpha = params.alpha_mu == 0 ? 0 : alpha_dist(rng);
        newpoly.elongation = params.elongation;
        newpoly.init_polarity();

        for (std::size_t i = vend[1]; i != vend[2]; i = (i + 1) % vold.size())
            polygons[p].vertices.push_back(vold[i]);
        for (std::size_t i = vend[3]; i != vend[0]; i = (i + 1) % vold.size())
            newpoly.vertices.push_back(vold[i]);

        polygons.push_back(newpoly);
        for (auto& v2 : polygons.back().vertices)
            v2.p = polygons.size() - 1;
        for (auto& v2 : polygons[p].vertices)
            v2.p = p;

        polygons[p].area();
        polygons.back().area();
        polygons[p].A0 = A0 * polygons[p].A / (polygons[p].A + polygons.back().A);
        polygons.back().A0 = A0 - polygons[p].A0;
    }

    void run(std::size_t steps) {
        for (std::size_t s = 0; s < steps; ++s)
            step();
    }

    // Ground-truth contact graph. Mirrors the vertex-edge proximity test from
    // interaction() but only records which polygon pairs are in contact instead
    // of applying forces. Two polygons are "in contact" when any vertex of one
    // lies within drmax = h + sh + ss of an edge of the other (same criterion
    // that gates the adhesion/repulsion force). Rebuilds the spatial hash since
    // positions have advanced since the last step()'s boxes() call.
    std::vector<std::pair<std::size_t, std::size_t>> get_contact_pairs() {
        std::set<std::pair<std::size_t, std::size_t>> pairs;

        if (polygons.empty()) return {};
        boxes();

        const double drmax = params.drmax();
        const double drmax2 = drmax * drmax;

        for (std::size_t p = params.Nr; p < polygons.size(); ++p) {
            auto& verts = polygons[p].vertices;
            for (std::size_t i = verts.size() - 1, k = i - 1, j = 0;
                 j < verts.size(); k = i, i = j++) {
                const std::size_t bxi = (verts[i].r.x - x0) / bs + 1;
                const std::size_t byi = (verts[i].r.y - y0) / bs + 1;
                for (std::size_t bxj = bxi - 1; bxj <= bxi + 1; ++bxj)
                    for (std::size_t byj = byi - 1; byj <= byi + 1; ++byj)
                        for (Vertex* v0 = first[bxj * Ny + byj]; v0; v0 = v0->next) {
                            if (v0->p == p) continue;
                            Vertex* v1 = &verts[i];
                            Vertex* v1n[2] = {&verts[k], &verts[j]};
                            double xi[2], xit[2], dr2[2];
                            Point dr[2];
                            for (unsigned int jj = 0; jj < 2; ++jj)
                                dr2[jj] = point_edge_dist2(v0->r, v1->r, v1n[jj]->r,
                                                           xi[jj], xit[jj], dr[jj]);
                            const unsigned int jj = dr2[0] > dr2[1];
                            if (xi[jj] <= 1 && dr2[jj] < drmax2) {
                                const std::size_t a = std::min(v0->p, p);
                                const std::size_t b = std::max(v0->p, p);
                                pairs.emplace(a, b);
                            }
                        }
            }
        }

        return std::vector<std::pair<std::size_t, std::size_t>>(pairs.begin(), pairs.end());
    }

    void output_vtk(const std::string& filename) {
        std::size_t Nv = 0;
        for (auto& p : polygons)
            Nv += p.vertices.size();

        std::ofstream file(filename);
        file << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        file << "  <PolyData>\n";
        file << "    <Piece NumberOfPoints=\"" << Nv << "\" NumberOfPolys=\"" << polygons.size() << "\">\n";
        file << "      <CellData>\n";
        file << "        <DataArray type=\"Float64\" Name=\"area\" format=\"ascii\">\n";
        for (auto& p : polygons)
            file << p.A << " ";
        file << "\n        </DataArray>\n";
        file << "        <DataArray type=\"Float64\" Name=\"perimeter\" format=\"ascii\">\n";
        for (auto& p : polygons)
            file << p.perimeter() << " ";
        file << "\n        </DataArray>\n";
        file << "        <DataArray type=\"Float64\" Name=\"polarity_x\" format=\"ascii\">\n";
        for (auto& p : polygons)
            file << p.polarity.x << " ";
        file << "\n        </DataArray>\n";
        file << "        <DataArray type=\"Float64\" Name=\"polarity_y\" format=\"ascii\">\n";
        for (auto& p : polygons)
            file << p.polarity.y << " ";
        file << "\n        </DataArray>\n";
        file << "      </CellData>\n";
        file << "      <Points>\n";
        file << "        <DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (auto& p : polygons)
            for (auto& v : p.vertices)
                file << v.r.x << " " << v.r.y << " " << p.phase << " ";
        file << "\n        </DataArray>\n";
        file << "      </Points>\n";
        file << "      <Polys>\n";
        file << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
        for (std::size_t i = 0; i < Nv; ++i)
            file << i << " ";
        file << "\n        </DataArray>\n";
        file << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
        std::size_t offset = 0;
        for (auto& p : polygons)
            file << (offset += p.vertices.size()) << " ";
        file << "\n        </DataArray>\n";
        file << "      </Polys>\n";
        file << "    </Piece>\n";
        file << "  </PolyData>\n";
        file << "</VTKFile>\n";
    }

    std::size_t num_polygons() const { return polygons.size(); }

    std::size_t total_vertices() const {
        std::size_t Nv = 0;
        for (const auto& p : polygons)
            Nv += p.vertices.size();
        return Nv;
    }

    Polygon& get_polygon(std::size_t i) {
        if (i >= polygons.size())
            throw std::out_of_range("Polygon index out of range");
        return polygons[i];
    }

    void add_polygon(const std::vector<std::pair<double, double>>& vertices, bool phase = false) {
        Polygon p;
        p.phase = phase;
        std::size_t idx = polygons.size();
        for (const auto& v : vertices) {
            p.vertices.push_back({{v.first, v.second}, {0, 0}, {0, 0}, idx});
        }
        p.A0 = p.area();

        // Use global parameters for Amax and alpha (with distributions if CV > 0)
        if (std::isinf(params.Amax_mu)) {
            p.Amax = INFINITY;
        } else {
            const double Amax_lnCV = std::log(1 + params.Amax_CV * params.Amax_CV);
            std::lognormal_distribution<> Amax_dist(std::log(params.Amax_mu) - Amax_lnCV/2, std::sqrt(Amax_lnCV));
            p.Amax = Amax_dist(rng);
        }

        if (params.alpha_mu == 0) {
            p.alpha = 0;
        } else {
            const double alpha_lnCV = std::log(1 + params.alpha_CV * params.alpha_CV);
            std::lognormal_distribution<> alpha_dist(std::log(params.alpha_mu) - alpha_lnCV/2, std::sqrt(alpha_lnCV));
            p.alpha = alpha_dist(rng);
        }

        p.elongation = params.elongation;
        p.init_polarity();

        // Set rest lengths
        for (std::size_t i = p.vertices.size() - 1, j = 0; j < p.vertices.size(); i = j++)
            p.vertices[i].l0 = (p.vertices[j].r - p.vertices[i].r).length();

        polygons.push_back(p);
    }
};

// Seed the RNG
void seed_rng(unsigned int seed) {
    rng.seed(seed);
}

PYBIND11_MODULE(polyhoop, m) {
    m.doc() = "Python bindings for PolyHoop - a polygon-based physics simulation";

    // Parameters class
    py::class_<Parameters>(m, "Parameters")
        .def(py::init<>())
        .def_readwrite("h", &Parameters::h, "Edge thickness")
        .def_readwrite("lmin", &Parameters::lmin, "Minimum edge length")
        .def_readwrite("lmax", &Parameters::lmax, "Maximum edge length")
        .def_readwrite("C", &Parameters::C, "Circularity index")
        .def_readwrite("alpha_mu", &Parameters::alpha_mu, "Mean area growth rate")
        .def_readwrite("alpha_CV", &Parameters::alpha_CV, "Coefficient of variation of area growth rate")
        .def_readwrite("beta", &Parameters::beta, "Minimum area fraction for growth")
        .def_readwrite("Amin", &Parameters::Amin, "Minimum area")
        .def_readwrite("Amax_mu", &Parameters::Amax_mu, "Mean maximum area")
        .def_readwrite("Amax_CV", &Parameters::Amax_CV, "Coefficient of variation of maximum area")
        .def_readwrite("gam", &Parameters::gam, "Line tension per vertex mass")
        .def_readwrite("ka", &Parameters::ka, "Area stiffness per vertex mass")
        .def_readwrite("kl", &Parameters::kl, "Edge contractility stiffness per vertex mass")
        .def_readwrite("kb", &Parameters::kb, "Bending stiffness per vertex mass")
        .def_readwrite("kr", &Parameters::kr, "Repulsion stiffness per vertex mass")
        .def_readwrite("kh", &Parameters::kh, "Adhesion stiffness per vertex mass")
        .def_readwrite("sh", &Parameters::sh, "Adhesion hardening zone size")
        .def_readwrite("ss", &Parameters::ss, "Adhesion softening zone size")
        .def_readwrite("theta", &Parameters::theta, "Fusion threshold")
        .def_readwrite("mu", &Parameters::mu, "Dynamic friction coefficient")
        .def_readwrite("rho", &Parameters::rho, "Fluid mass density per vertex mass")
        .def_readwrite("g", &Parameters::g, "Gravitational acceleration")
        .def_readwrite("gl", &Parameters::gl, "Edge gravitational acceleration")
        .def_readwrite("cv", &Parameters::cv, "Viscous damping rate")
        .def_readwrite("cd", &Parameters::cd, "Drag coefficient")
        .def_readwrite("cc", &Parameters::cc, "Collision damping rate")
        .def_readwrite("dt", &Parameters::dt, "Time step")
        .def_readwrite("fp", &Parameters::fp, "Polarity force per vertex mass")
        .def_readwrite("Dr", &Parameters::Dr, "Rotational diffusion coefficient")
        .def_readwrite("elongation", &Parameters::elongation, "Elongation factor")
        .def_readwrite("Nr", &Parameters::Nr, "Number of rigid polygons")
        .def_readwrite("oscillatory_growth", &Parameters::oscillatory_growth, "Use oscillatory (true) or linear (false) growth model")
        .def_readwrite("position_noise_std", &Parameters::position_noise_std, "Position noise standard deviation")
        .def_readwrite("compress_force", &Parameters::compress_force, "Uniform compression force magnitude toward center (0 = disabled)")
        .def_readwrite("compress_cx", &Parameters::compress_cx, "Compression center x-coordinate")
        .def_readwrite("compress_cy", &Parameters::compress_cy, "Compression center y-coordinate")
        .def_readwrite("simulation_start_time", &Parameters::simulation_start_time, "Reference time for gam/kl exponential ramps")
        .def_readwrite("wall_xmin", &Parameters::wall_xmin, "Box wall x-min (walls active when xmax > xmin and ymax > ymin)")
        .def_readwrite("wall_xmax", &Parameters::wall_xmax, "Box wall x-max")
        .def_readwrite("wall_ymin", &Parameters::wall_ymin, "Box wall y-min")
        .def_readwrite("wall_ymax", &Parameters::wall_ymax, "Box wall y-max")
        .def_readwrite("wall_kh", &Parameters::wall_kh, "Wall adhesion stiffness (0 = repulsion-only)")
        .def("drmax", &Parameters::drmax, "Maximum interaction distance");

    // Point class
    py::class_<Point>(m, "Point")
        .def(py::init<double, double>(), py::arg("x") = 0, py::arg("y") = 0)
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y)
        .def("__add__", &Point::operator+)
        .def("__sub__", &Point::operator-)
        .def("cross", &Point::cross, "Perpendicular vector")
        .def("dot", &Point::dot, "Dot product")
        .def("wedge", &Point::wedge, "Wedge (cross) product")
        .def("length2", &Point::length2, "Squared norm")
        .def("length", &Point::length, "Norm")
        .def("__mul__", [](const Point& p, double a) { return p * a; })
        .def("__rmul__", [](const Point& p, double a) { return a * p; })
        .def("__repr__", &Point::repr);

    // Vertex class
    py::class_<Vertex>(m, "Vertex")
        .def(py::init<>())
        .def_readwrite("r", &Vertex::r, "Position")
        .def_readwrite("v", &Vertex::v, "Velocity")
        .def_readwrite("a", &Vertex::a, "Acceleration")
        .def_readwrite("p", &Vertex::p, "Polygon index")
        .def_readwrite("l0", &Vertex::l0, "Rest length of edge to the right");

    // Polygon class
    py::class_<Polygon>(m, "Polygon")
        .def(py::init<>())
        .def_readwrite("vertices", &Polygon::vertices)
        .def_readwrite("phase", &Polygon::phase)
        .def_readwrite("A0", &Polygon::A0, "Target area")
        .def_readwrite("A", &Polygon::A, "Actual area")
        .def_readwrite("Amax", &Polygon::Amax, "Division area")
        .def_readwrite("alpha", &Polygon::alpha, "Area growth rate")
        .def_readwrite("area_fluct_period", &Polygon::area_fluct_period, "Area fluctuation period")
        .def_readwrite("area_fluct_amp", &Polygon::area_fluct_amp, "Area fluctuation amplitude")
        .def_readwrite("area_fluct_phase", &Polygon::area_fluct_phase, "Area oscillation phase offset")
        .def_readwrite("gam", &Polygon::gam, "Per-cell line tension (0 = use global)")
        .def_readwrite("gam_fluct_period", &Polygon::gam_fluct_period, "Line tension fluctuation period")
        .def_readwrite("gam_fluct_amp", &Polygon::gam_fluct_amp, "Line tension fluctuation amplitude")
        .def_readwrite("gam_fluct_phase", &Polygon::gam_fluct_phase, "Line tension oscillation phase offset")
        .def_readwrite("gam_rate", &Polygon::gam_rate, "Per-cell exponential growth rate for gam [1/T]")
        .def_readwrite("kl", &Polygon::kl, "Per-cell elastic tension (0 = use global)")
        .def_readwrite("kl_fluct_period", &Polygon::kl_fluct_period, "Elastic tension fluctuation period")
        .def_readwrite("kl_fluct_amp", &Polygon::kl_fluct_amp, "Elastic tension fluctuation amplitude")
        .def_readwrite("kl_fluct_phase", &Polygon::kl_fluct_phase, "Elastic tension oscillation phase offset")
        .def_readwrite("kl_rate", &Polygon::kl_rate, "Per-cell exponential growth rate for kl [1/T]")
        .def_readwrite("kh", &Polygon::kh, "Per-cell adhesion strength (0 = use global)")
        .def_readwrite("kh_fluct_period", &Polygon::kh_fluct_period, "Adhesion fluctuation period")
        .def_readwrite("kh_fluct_amp", &Polygon::kh_fluct_amp, "Adhesion fluctuation amplitude")
        .def_readwrite("kh_fluct_phase", &Polygon::kh_fluct_phase, "Adhesion oscillation phase offset")
        .def_readwrite("kh_rate", &Polygon::kh_rate, "Per-cell exponential growth rate for kh [1/T]")
        .def_readwrite("elongation", &Polygon::elongation, "Elongation factor")
        .def_readwrite("polarity", &Polygon::polarity, "Cell polarity direction")
        .def("area", &Polygon::area, "Compute and return the polygon area")
        .def("perimeter", &Polygon::perimeter, "Compute and return the perimeter")
        .def("perimeter0", &Polygon::perimeter0, "Compute and return the rest perimeter")
        .def("contains", &Polygon::contains, "Check if a point is inside the polygon")
        .def("centroid", &Polygon::centroid, "Compute and return the centroid")
        .def("init_polarity", &Polygon::init_polarity, "Initialize polarity to random direction")
        .def("rotate_polarity", &Polygon::rotate_polarity, "Rotate polarity by angle dtheta")
        .def("num_vertices", &Polygon::num_vertices, "Number of vertices")
        .def("get_vertex_positions", &Polygon::get_vertex_positions,
             "Get list of (x, y) tuples for all vertices");

    // Ensemble class
    py::class_<Ensemble>(m, "Ensemble")
        .def(py::init<>())
        .def(py::init<const std::string&>(), py::arg("filename"),
             "Load ensemble from OFF file")
        .def("load_off", &Ensemble::load_off, "Load polygons from OFF file")
        .def("step", [](Ensemble& e) {
            py::gil_scoped_release release;
            e.step();
        }, "Perform one simulation step (releases GIL for parallelization)")
        .def("run", [](Ensemble& e, std::size_t steps) {
            py::gil_scoped_release release;
            e.run(steps);
        }, py::arg("steps"), "Run multiple simulation steps (releases GIL for parallelization)")
        .def("output_vtk", &Ensemble::output_vtk, py::arg("filename"),
             "Write current state to VTK file")
        .def("num_polygons", &Ensemble::num_polygons, "Number of polygons")
        .def("total_vertices", &Ensemble::total_vertices, "Total number of vertices")
        .def("get_polygon", &Ensemble::get_polygon, py::return_value_policy::reference,
             "Get polygon by index")
        .def("add_polygon", &Ensemble::add_polygon,
             py::arg("vertices"), py::arg("phase") = false,
             "Add a new polygon from list of (x, y) coordinates")
        .def("get_contact_pairs", [](Ensemble& e) {
            py::gil_scoped_release release;
            return e.get_contact_pairs();
        }, "Return list of (i, j) polygon index pairs (i < j) in ground-truth contact")
        .def_readwrite("polygons", &Ensemble::polygons)
        .def_readwrite("t", &Ensemble::t, "Current simulation time")
        .def("__len__", &Ensemble::num_polygons);

    // Global parameters access
    m.def("get_params", []() -> Parameters& { return params; },
          py::return_value_policy::reference,
          "Get the global parameters object");

    m.def("set_params", [](const Parameters& p) { params = p; },
          "Set the global parameters");

    // RNG seeding
    m.def("seed", &seed_rng, py::arg("seed"),
          "Seed the random number generator");

    // OpenMP thread control
    #ifdef _OPENMP
    m.def("set_num_threads", [](int n) { omp_set_num_threads(n); },
          py::arg("num_threads"),
          "Set the number of OpenMP threads");
    m.def("get_max_threads", []() { return omp_get_max_threads(); },
          "Get the maximum number of OpenMP threads");
    m.def("get_num_procs", []() { return omp_get_num_procs(); },
          "Get the number of available processors");
    m.attr("openmp_enabled") = true;
    #else
    m.def("set_num_threads", [](int) {
        py::print("Warning: OpenMP not enabled, set_num_threads has no effect");
    }, py::arg("num_threads"), "Set the number of OpenMP threads (no-op if OpenMP disabled)");
    m.def("get_max_threads", []() { return 1; },
          "Get the maximum number of OpenMP threads (returns 1 if OpenMP disabled)");
    m.def("get_num_procs", []() { return 1; },
          "Get the number of available processors (returns 1 if OpenMP disabled)");
    m.attr("openmp_enabled") = false;
    #endif

    // Utility function
    m.def("point_edge_dist2", [](const Point& r0, const Point& r1, const Point& r2) {
        double xi, xit;
        Point dr;
        double dist2 = point_edge_dist2(r0, r1, r2, xi, xit, dr);
        return py::make_tuple(dist2, xi, xit, dr);
    }, "Compute squared distance from point to edge segment");

    // Helper function to run full simulation
    m.def("run_simulation", [](const std::string& input_file,
                               std::size_t num_frames,
                               std::size_t steps_per_frame,
                               const std::string& output_prefix,
                               bool verbose) {
        Ensemble ensemble(input_file);

        if (verbose) {
            py::print("frame 0, t=", ensemble.t, ",", ensemble.num_polygons(),
                     "polygons,", ensemble.total_vertices(), "vertices");
        }

        if (!output_prefix.empty()) {
            char name[256];
            snprintf(name, 256, "%s%06zu.vtp", output_prefix.c_str(), (std::size_t)0);
            ensemble.output_vtk(name);
        }

        for (std::size_t f = 1; f <= num_frames; ++f) {
            {
                py::gil_scoped_release release;
                ensemble.run(steps_per_frame);
            }

            if (verbose) {
                py::print("frame", f, ", t=", ensemble.t, ",", ensemble.num_polygons(),
                         "polygons,", ensemble.total_vertices(), "vertices");
            }

            if (!output_prefix.empty()) {
                char name[256];
                snprintf(name, 256, "%s%06zu.vtp", output_prefix.c_str(), f);
                ensemble.output_vtk(name);
            }
        }

        return ensemble;
    },
    py::arg("input_file"),
    py::arg("num_frames") = 1000,
    py::arg("steps_per_frame") = 1000,
    py::arg("output_prefix") = "frame",
    py::arg("verbose") = true,
    R"doc(
Run a complete simulation from an OFF file.

Args:
    input_file: Path to the input OFF file
    num_frames: Number of output frames (default: 1000)
    steps_per_frame: Number of simulation steps between frames (default: 1000)
    output_prefix: Prefix for output VTK files (default: "frame"), empty string to disable output
    verbose: Print progress messages (default: True)

Returns:
    The final Ensemble object after simulation completes
)doc");
}
