#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <numeric>
#include <vector>
#include <memory>
#include <map>
#include <tuple>
#include <exception>
#include <iostream>
#include <functional>
#include <mutex>          // std::mutex
#include "ThreadPool.h"
#include "cubature.h"
#if !defined(NO_CUBA)
#include "cuba.h"
#endif
#include "MultiComplex/MultiComplex.hpp"

#include "nlohmann/json.hpp"

#include "integration.hpp"
#include "molecule.hpp"
#include "evaluator.hpp"

double factorial(double n) {
    return std::tgamma(n + 1);
}

std::mutex mtx;  // mutex for cout

#if !defined(M_PI)
constexpr auto M_PI = 3.14159265358979323846;
#endif

auto geomspace(double xmin, double xmax, int N) {
    std::vector<double> vec;
    double dT = (log(xmax) - log(xmin)) / (N - 1);
    for (auto i = 0; i < N; ++i) {
        vec.push_back(exp(log(xmin) + dT * i));
    }
    return vec;
}

double gr = (sqrt(5) + 1) / 2;

auto gss(std::function<double(double)> f, double a, double b, const double tol = 1e-5) {
    /*
    Golden section search
    C++ translation of https://en.wikipedia.org/wiki/Golden-section_search#Algorithm
    Text is available under the Creative Commons Attribution-ShareAlike License
    */
    auto c = b - (b - a) / gr;
    auto d = a + (b - a) / gr;
    while (abs(c - d) > tol) {
        if (f(c) < f(d)) {
            b = d;
        }
        else {
            a = c;
        }
        // We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr;
        d = a + (b - a) / gr;
    }
    return std::make_tuple((b + a) / 2, f((b + a) / 2));
}

template<typename TYPE> using Molecule = potter::Molecule<TYPE>;
template<typename TYPE> using PotentialEvaluator = potter::PairwisePotentialEvaluator<TYPE>;

/// A helper class to manage evaluation of the various integrand terms 
template<typename TYPE, typename TEMPTYPE>
class IntegrandHelper {
private:
    std::vector<Molecule<TYPE>> mol_sys;
public:
    TEMPTYPE Tstar = -1;
    TYPE rstar = -1;
    
    const PotentialEvaluator<TYPE>& evaltr;

    IntegrandHelper(
        const TEMPTYPE &Tstar,
        const std::vector<Molecule<TYPE>>& mol_sys,
        const PotentialEvaluator<TYPE>& evaltr
    )
    : Tstar(Tstar), mol_sys(mol_sys), evaltr(evaltr) {};

    auto& get_mol(std::size_t i) {
        if (mol_sys.size() < 2) { throw std::invalid_argument("Mol system length must be at least 2!"); }
        return mol_sys[i];
    };

    TYPE eval_pot(const Molecule<TYPE>& molA, const Molecule<TYPE>& molB) {
        return evaltr.eval_pot(molA, molB);
    };

    /**
    * Take the variable a and unpack it into the double buffer
    */
    template<typename T>
    void unpack_f(const T &a, double *fval) {
        if constexpr (std::is_same<T, double>::value) {
            // If T is double (real)
            fval[0] = a;
        }
        else if constexpr (std::is_same<T, std::complex<double>>::value) {
            // If T is a complex number (perhaps for complex step derivatives)
            fval[0] = a.real();
            fval[1] = a.imag();
        }
        else if constexpr (std::is_same<T, MultiComplex<double>>::value) {
            // If T is a multicomplex number
            auto& c = a.get_coef();
            for (auto i = 0; i < c.size(); ++i) {
                fval[i] = c[i];
            }
        }
        else {
            throw std::invalid_argument("temparature type doesn't match");
        }
    }

    /* 
    Given the orientational angles for linear molecules, calculate the integrand
    */
    void oriented_integrand(double theta1, double theta2, double phi, double *fval)
    {
        auto molA = get_mol(0);
        auto molB = get_mol(1);

        // Rotate molecule #1
        molA.reset(); // Back to COM at origin
        molA.rotate_negativey(theta1); // First rotate around -y axis

        // Rotate and move molecule #2
        molB.reset(); // Back to COM at origin
        molB.rotate_negativey(theta2); // First rotate around -y axis
        molB.rotate_negativex(phi); // Then rotate around +x
        molB.translatex(rstar); // Then translate

        auto V = eval_pot(molA, molB); // And finally evaluate the potential
        auto a = (exp(-V/Tstar)-1.0)*sin(theta1)*sin(theta2)*pow(rstar, 2);
        unpack_f<decltype(Tstar)>(a, fval);
    }

    void oriented_integrand_3D(double theta_1_o, double r_12, double theta_2_o, double phi_2_o, double r_13, double theta_3, double phi_3, double theta_3_o, double phi_3_o, double* fval) {

        mol_sys[0].reset(); // Back to COM at origin
        mol_sys[1].reset();
        mol_sys[2].reset();

        // Rotate molecule #1
        mol_sys[0].rotate_negativey(theta_1_o); // First rotate around -y axis

        // Rotate molecule #2
        mol_sys[1].rotate_negativey(theta_2_o); // First rotate around -y axis
        mol_sys[1].rotate_negativex(phi_2_o);   // Then rotate around +x
        mol_sys[1].translatex(r_12);            // Then translate

        // Rotate molecule #2
        mol_sys[2].rotate_negativey(theta_3_o); // First rotate around -y axis
        mol_sys[2].rotate_negativex(phi_3_o);   // Then rotate around +x
        mol_sys[2].translate_3D(r_13, phi_3, theta_3); // Then translate

        // Evaluate the potential for 
        auto V12 = eval_pot(mol_sys[0], mol_sys[1]);
        auto V13 = eval_pot(mol_sys[0], mol_sys[2]);
        auto V23 = eval_pot(mol_sys[1], mol_sys[2]);
        auto SQUARE = [](double x) { return x * x; };
        auto a = sin(theta_1_o)*sin(theta_2_o)*sin(theta_3_o)*sin(theta_3)*(exp(-V12 / Tstar) - 1.0) * (exp(-V13 / Tstar) - 1.0) * (exp(-V23 / Tstar) - 1.0) * SQUARE(r_12) * SQUARE(r_13);
        unpack_f<decltype(Tstar)>(a, fval);
    }

    /*
    Given the separation r, calculate the integrand for B_2 for a spherically-symmetric potential
    for an atomic fluid
    */
    void atomic_B2_integrand(const double r, double* fval) {
        // Get the potential function V(r) that we should use
        auto& pot = this->evaltr.get_potential(0, 0);
        TEMPTYPE Tstar = this->Tstar; // Local reference just for sharing with the lambda function f
        auto f = [pot, Tstar](double r) -> TEMPTYPE { return (exp(-pot(r) / Tstar) - 1.0)*r*r; };
        unpack_f<decltype(Tstar)>(f(r), fval);
    }
    
    /*
    Given the separations and angle, calculate the integrand for B_3 for a spherically-symmetric potential
    for an atomic fluid
    */
    void atomic_B3_integrand(const double r12, const double r13, const double eta_angle, double* fval) {
        // Get the potential function V(r) that we should use
        auto &pot = this->evaltr.get_potential(0, 0);
        TEMPTYPE Tstar = this->Tstar; // Local reference just for sharing with the lambda function f
        auto f = [pot, Tstar](double r) -> TEMPTYPE { return 1.0 - exp(-pot(r)/Tstar); };
        auto SQUARE = [](double x) { return x*x; };
        auto rangle = sqrt(SQUARE(r12) + SQUARE(r13) - 2*r12*r13*eta_angle);
        auto a = SQUARE(r12)*f(r12)*SQUARE(r13)*f(r13)*f(rangle);
        unpack_f<decltype(Tstar)>(a, fval);
    }

    /*
    Given the separations and angles, calculate the integrand for B4_1 for a spherically-symmetric potential
    for an atomic fluid
    */
    void atomic_B4_1_integrand(const double r14, const double r13, const double gamma_angle, const double r12, const double eta_angle, double* fval)
    {
        // Get the potential function V(r) that we should use
        auto &pot = this->evaltr.get_potential(0, 0);
        TEMPTYPE Tstar = this->Tstar; // Local reference just for sharing with the lambda function f
        auto f = [pot, Tstar](double r) -> TEMPTYPE { return 1.0 - exp(-pot(r) / Tstar); };
        auto SQUARE = [](double x) { return x * x; };
        auto sq_r12 = SQUARE(r12);
        auto sq_r13 = SQUARE(r13);
        auto sq_r14 = SQUARE(r14);
        auto rangle_12_13 = sqrt(sq_r12 + sq_r13 - 2 * r12*r13*eta_angle);
        auto rangle_13_14 = sqrt(sq_r14 + sq_r13 - 2 * r14*r13*gamma_angle);

        auto a = sq_r12 * f(r12)*sq_r13*sq_r14*f(r14)*f(rangle_12_13)*f(rangle_13_14);
        unpack_f<decltype(Tstar)>(a, fval);
    }

    /*
    Given the separations and angles, calculate the integrand for B4_2 for a spherically-symmetric potential
    for an atomic fluid
    */
    void atomic_B4_2_integrand(const double eta_angle, const double r12, const double r13, const double gamma_angle, const double r14, double* fval)
    {
        // Get the potential function V(r) that we should use
        auto &pot = this->evaltr.get_potential(0, 0);
        TEMPTYPE Tstar = this->Tstar; // Local reference just for sharing with the lambda function f
        auto f = [pot, Tstar](double r) -> TEMPTYPE { return 1.0 - exp(-pot(r) / Tstar); };
        auto SQUARE = [](double x) { return x * x; };
        auto sq_r12 = SQUARE(r12);
        auto sq_r13 = SQUARE(r13);
        auto sq_r14 = SQUARE(r14);
        auto rangle_12_13 = sqrt(sq_r12 + sq_r13 - 2 * r12*r13*eta_angle);
        auto rangle_13_14 = sqrt(sq_r14 + sq_r13 - 2 * r14*r13*gamma_angle);

        auto a = sq_r12 * f(r12)*sq_r13*f(r13)*sq_r14*f(r14)*f(rangle_12_13)*f(rangle_13_14);
        unpack_f<decltype(Tstar)>(a, fval);
    }

    /*
    Given the separations and angles, calculate the integrand for B4_3 for a spherically-symmetric potential
    for an atomic fluid
    */
    void atomic_B4_3_integrand(const double eta_angle, const double zeta_angle, const double gamma_angle, const double r12, const double r13, const double r14, double* fval)
    {
        // Get the potential function V(r) that we should use
        auto &pot = this->evaltr.get_potential(0, 0);
        TEMPTYPE Tstar = this->Tstar; // Local reference just for sharing with the lambda function f
        auto f = [pot, Tstar](double r) -> TEMPTYPE { return 1.0 - exp(-pot(r) / Tstar); };
        auto SQUARE = [](double x) { return x * x; };
        auto sq_r12 = SQUARE(r12);
        auto sq_r13 = SQUARE(r13);
        auto sq_r14 = SQUARE(r14);
        auto rangle_12_13 = sqrt(sq_r12 + sq_r14 - 2 * r12*r14*eta_angle);
        auto rangle_13_14 = sqrt(sq_r13 + sq_r14 - 2 * r13*r14*gamma_angle);
        auto rangle_12_14 = sqrt(sq_r12 + sq_r13 - 2 * r12*r13*(eta_angle * gamma_angle + sqrt(1.0 - SQUARE(eta_angle))*sqrt(1.0 - SQUARE(gamma_angle))*cos(zeta_angle)));
        auto a = sq_r12 * f(r12)*sq_r13*f(r13)*sq_r14*f(r14)*f(rangle_12_13)*f(rangle_13_14)*f(rangle_12_14);
        unpack_f<decltype(Tstar)>(a, fval);
    }
};

template<typename TYPE>
class Integrator {
private:
    std::unique_ptr<ThreadPool> m_pool;
    nlohmann::json m_conf;
    std::vector<Molecule<TYPE>> mol_sys;
public:
    using EColArray = Eigen::Array<TYPE, Eigen::Dynamic, 1>;
    PotentialEvaluator<TYPE> potcls;

    Integrator(const std::vector<Molecule<TYPE>>& mol_sys) : mol_sys(mol_sys) {};
    
    auto& get_mol(size_t i) const {
        if (mol_sys.size() < 2) {throw std::invalid_argument("Mol system length must be at least 2!");}
        return mol_sys[i];
    };

    auto& get_conf_view() {
        return m_conf;
    }

    /* For a one-dimensional integration for B_2, use trapezoidal integration to calculate B_2 */
    template <typename TEMPTYPE>
    TEMPTYPE radial_integrate_B2(TEMPTYPE Tstar, TYPE rstart, TYPE rend, int N) {
        using arr = Eigen::Array<TYPE, Eigen::Dynamic, 1>;
        using arrT = Eigen::Array<TEMPTYPE, Eigen::Dynamic, 1>;
        arr rv = exp(arr::LinSpaced(N, log(rstart), log(rend)));
        arrT integrand;
        integrand.resize(rv.size());
        
        // Get the potential function V(r) that we should use
        auto& pot = get_evaluator().get_potential(0, 0);

        // Sample the range of r
        for (auto ir = 0; ir < rv.size(); ++ir) {
            auto r = rv[ir];
            auto V = pot(r);
            integrand[ir] = (exp(-V/Tstar)-1.0)*r*r;
        }
        return -2*M_PI*potter::trapz(rv, integrand);
    };
    /* 
    Get a reference to the evaluator class, giving access to matrix of site-site potential functions, for instance
    */
    auto &get_evaluator() {
        return potcls;
    }

    void init_thread_pool(short Nthreads) {
        if (!m_pool || m_pool->GetThreads().size() != Nthreads) {
            // Make a thread pool for the workers
            m_pool = std::unique_ptr<ThreadPool>(new ThreadPool(Nthreads));
        }
    }

    /**
    * A helper function to evaluate the potential given COM separation r and the orientation angles
    */
    double potential(double r, double theta1, double theta2, double phi){
        Molecule<TYPE> molA = get_mol(0), molB = get_mol(1);
        // Rotate molecule #1
        molA.reset(); // Back to COM at origin
        molA.rotate_negativey(theta1); // First rotate around -y axis

        // Rotate and move molecule #2
        molB.reset(); // Back to COM at origin
        molB.rotate_negativey(theta2); // First rotate around -y axis
        molB.rotate_negativex(phi); // Then rotate around +x
        molB.translatex(r); // Then translate

        auto V = potcls.eval_pot(molA, molB); // And finally evaluate the potential in the form V/epsilon
        return V;
    }

    /**
    * Calculate the orientationally-averaged potential
    */
    TYPE orientationally_averaged_potential(TYPE rstar) const {
        using Helper = IntegrandHelper<TYPE, double>;
        Helper helper(0.0, mol_sys, potcls);
        //typedef int (*integrand) (unsigned ndim, const double *x, void *, unsigned fdim, double* fval);
        auto f_integrand = [](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) {
            auto& shared = *((class helper*)(p_shared_data));
            double theta1 = x[0], theta2 = x[1], phi = x[2];
            shared.oriented_integrand(theta1, theta2, phi, fval);
            return 0; // success
        };
        helper.rstar = rstar;
        int ndim = 1;
        std::valarray<double> val(0.0, 4), err(0.0, 4);
        hcubature(ndim, f_integrand, &helper, 3, &(helper.xmin[0]), &(helper.xmax[0]), 100000, 0, 1e-13, ERROR_INDIVIDUAL, &(val[0]), &(err[0]));
        return val[0]/(8*M_PI);
    }

    template<typename T>
    auto allocate_buffer(const T & Tstar) const {
        std::size_t ndim = 0;
        if constexpr (std::is_same<T, double>::value) {
            ndim = 1;
        }
        else if constexpr (std::is_same<T, std::complex<double>>::value) {
            ndim = 2;
        }
        else if constexpr (std::is_same<T, MultiComplex<double>>::value) {
            ndim = static_cast<int>(Tstar.get_coef().size());
        }
        return std::valarray<double>(0.0, ndim);
    }

    /// A helper function to make the output tuple in the right type
    template<typename T>
    auto make_output_tuple(const T& Tstar, const std::valarray<double> &outval, const std::valarray<double> &outerr) const {
        if constexpr (std::is_same_v<TYPE, double>) {
            // If T is double (real)
            return std::make_tuple(outval[0], outerr[0]);
        }
        else if constexpr (std::is_same_v<T, std::complex<double>>) {
            // If T is a complex number (perhaps for complex step derivatives)
            return std::make_tuple(T(outval[0], outval[1]), T(outerr[0], outerr[1]));
        }
        else if constexpr (std::is_same_v<T, MultiComplex<double>>) {
            // If T is a multicomplex number
            return std::make_tuple(T(outval), T(outerr));
        }
        else {
            throw std::invalid_argument("Can't construct output tuple");
            return std::make_tuple(T(outval), T(outerr));
        }
    }

    template <typename TEMPTYPE>
    std::tuple<TEMPTYPE, TEMPTYPE> one_temperature_B2(TEMPTYPE Tstar, TYPE rstart, TYPE rend) const {
        // Some local typedefs to avoid typing
        using SharedData = IntegrandHelper<TYPE, TEMPTYPE>;

        auto outval = allocate_buffer(Tstar), outerr = allocate_buffer(Tstar);
        
        bool is_atomic = (get_mol(0).get_Natoms() == 1);
        SharedData shared(Tstar, mol_sys, potcls);
        bool is_linear = true; // TODO: check if true with principal axes

        if (is_atomic) {
            // The integrand function
            //typedef int (*integrand) (unsigned ndim, const double *x, void *, unsigned fdim, double* fval);
            auto cubature_B2_integrand = [](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) {
                auto& shared = *static_cast<SharedData*>(p_shared_data);
                shared.atomic_B2_integrand(x[0], fval);
                return 0; // success
            };

            std::valarray<double> xmins = { rstart }, xmaxs = { rend }; // Limits on r

            int feval_max = 0;
            if (m_conf.contains("feval_max")) {
                feval_max = static_cast<int>(m_conf["feval_max"]);
            }
            else {
                throw std::invalid_argument("Key \"feval_max\" must be specified in the configuration JSON");
            }
            unsigned naxes = 1; // How many dimensions the integral is taken over (r)
            unsigned fdim = static_cast<unsigned>(outval.size()); // How many output dimensions
            hcubature(fdim, cubature_B2_integrand, &shared, naxes, &(xmins[0]), &(xmaxs[0]), feval_max, 0, 1e-13, ERROR_INDIVIDUAL, &(outval[0]), &(outerr[0]));

            // Copy into output
            outval *= -2*M_PI; outerr *= -2*M_PI;
        }
        else if (is_linear) {

            // The integrand function
            //typedef int (*integrand) (unsigned ndim, const double *x, void *, unsigned fdim, double* fval);
            potter::c_integrand_function integrand = [](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) {
                auto& shared = *static_cast<SharedData*>(p_shared_data);
                double theta1 = x[0], theta2 = x[1], phi = x[2];
                shared.rstar = x[3];
                shared.oriented_integrand(theta1, theta2, phi, fval);
                return 0; // success
            };

            std::valarray<double> xmins = { 0, 0, 0, rstart }, xmaxs = { M_PI, M_PI, 2 * M_PI, rend }; // Limits on theta1, theta2, phi, r

            int feval_max = 0;
            if (m_conf.contains("feval_max")) {
                feval_max = static_cast<int>(m_conf["feval_max"]);
            }
            else {
                throw std::invalid_argument("Key \"feval_max\" must be specified in the configuration JSON");
            }
            unsigned naxes = 4; // How many dimensions the integral is taken over (theta, phi1, phi2, r)
            unsigned fdim = static_cast<unsigned>(outval.size()); // How many output dimensions

#if defined(ENABLE_CUBA)
            auto opt = potter::get_VEGAS_defaults();
            opt["FDIM"] = fdim;
            opt["NDIM"] = naxes;
            opt["MAXEVAL"] = feval_max;
            std::tie(outval, outerr) = potter::VEGAS<decltype(outval)>(integrand, &shared, xmins, xmaxs, opt);
#else
            hcubature(fdim, integrand, &shared, naxes, &(xmins[0]), &(xmaxs[0]), feval_max, 0, 1e-13, ERROR_INDIVIDUAL, &(outval[0]), &(outerr[0]));
#endif

            // Copy into output
            // ....
            // The quadruple integral needs to be divided by 8*pi, but the leading term in the
            // expression for B_2 is -2\pi, so factor becomes -1/4, or -0.25
            outval *= -0.25; outerr *= -0.25;
        }
        else {
            throw std::invalid_argument("Not yet able to handle non-linear molecules");
        }
        return make_output_tuple(Tstar, std::move(outval), std::move(outerr));
    }

    template <typename TEMPTYPE>
    std::tuple<TEMPTYPE, TEMPTYPE> one_temperature_B3(TEMPTYPE Tstar, TYPE rstart, TYPE rend) const {
        // Some local typedefs to avoid typing
        using SharedData = IntegrandHelper<TYPE, TEMPTYPE>;

        auto outval = allocate_buffer(Tstar), outerr = allocate_buffer(Tstar);

        bool is_atomic = (get_mol(0).get_Natoms() == 1);
        SharedData shared(Tstar, mol_sys, potcls);
        bool is_linear = true; // TODO: check if true with principal axes

        int feval_max = 0;
        if (m_conf.contains("feval_max")) {
            feval_max = static_cast<int>(m_conf["feval_max"]);
        }
        else {
            throw std::invalid_argument("Key \"feval_max\" must be specified in the configuration JSON");
        }
        unsigned fdim = static_cast<unsigned>(outval.size()); // How many output dimensions

        if (is_atomic) {

            // The integrand function
            //typedef int (*integrand) (unsigned ndim, const double *x, void *, unsigned fdim, double* fval);
            auto cubature_integrand = [](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) {
                auto& shared = *((SharedData*)(p_shared_data));
                shared.atomic_B3_integrand(x[0], x[1], x[2], fval);
                return 0; // success
            };

            double rbreak = 1.3;
            std::vector<std::valarray<double>> xmins = { { rstart, rstart, -1 }, { rbreak, rstart, -1 } };
            std::vector<std::valarray<double>> xmaxs = { { rbreak, rend, 1 },    { rend, rend, 1 } };

            int naxes = 3; // How many dimensions the integral is taken over (r12, r13, eta)
            for (auto i = 0; i < xmins.size(); ++i) {
                auto vals = allocate_buffer(Tstar), errs = allocate_buffer(Tstar);
                auto xmin = xmins[i];
                auto xmax = xmaxs[i];

                hcubature(fdim, cubature_integrand, &shared, naxes, &(xmin[0]), &(xmax[0]), feval_max, 0, 1e-13, ERROR_INDIVIDUAL, &(vals[0]), &(errs[0]));

                // Copy into output
                outval += vals; outerr += std::abs(errs);
            }
            // Rescale with the leading factor
            double fac = 8*M_PI*M_PI/3;
            outval *= fac; outerr *= fac;
        }
        else if (is_linear) {
            std::valarray<double> xmins = { 0 ,rstart,0,0,rstart,0,0,0,0 };
            std::valarray<double> xmaxs = { M_PI, rend, M_PI, 2*M_PI , rend , M_PI , 2 * M_PI ,  M_PI , 2 * M_PI };

            // The integrand function
            potter::c_integrand_function integrand = [](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) {
                auto& shared = *static_cast<SharedData*>(p_shared_data);
                // particle 1 orientation
                double theta_1_o = x[0];

                // particle 2 location
                double r_12 = x[1];

                // particle 2 orientation
                double theta_2_o = x[2]; double phi_2_o = x[3];

                // particle 3 location
                double r_13 = x[4];
                double theta_3 = x[5]; double phi_3 = x[6];

                // particle 3 orientation
                double theta_3_o = x[7]; double phi_3_o = x[8];

                shared.oriented_integrand_3D(theta_1_o, r_12, theta_2_o, phi_2_o, r_13, theta_3, phi_3, theta_3_o, phi_3_o, fval);
                return 0; // success
            };
            
            unsigned naxes = 9; // How many dimensions the integral is taken over (theta, phi1, phi2, r)

#if defined(ENABLE_CUBA)
            auto opt = potter::get_VEGAS_defaults();
            opt["FDIM"] = fdim;
            opt["NDIM"] = naxes;
            opt["MAXEVAL"] = feval_max;
            for (std::string k : {"NSTART", "NINCREASE", "NBATCH"}){
                if (m_conf.contains(k)){
                    opt[k] = m_conf[k];
                }
            }
            std::tie(outval, outerr) = potter::VEGAS<decltype(outval)>(integrand, &shared, xmins, xmaxs, opt);
#else
            hcubature(fdim, integrand, &shared, naxes, &(xmins[0]), &(xmaxs[0]), feval_max, 0, 1e-13, ERROR_INDIVIDUAL, &(outval[0]), &(outerr[0]));
#endif

            // Copy into output
            // ....
            double fac = -1.0/3.0*(pow(2.0, 3.0)*pow(M_PI, 2.0))/(pow(4.0*M_PI, 3.0));
            outval *= fac; outerr *= fac;
        }
        else {
            throw std::invalid_argument("Not yet able to handle non-linear molecules");
        }
        return make_output_tuple(Tstar, std::move(outval), std::move(outerr));
    }

    template <typename TEMPTYPE>
    std::tuple<TEMPTYPE, TEMPTYPE> one_temperature_B4(TEMPTYPE Tstar, TYPE rstart, TYPE rend) const {
        // Some local typedefs to avoid typing
        using SharedData = IntegrandHelper<TYPE, TEMPTYPE>;
        SharedData shared(Tstar, mol_sys, potcls);

        auto outval = allocate_buffer(Tstar), outerr = allocate_buffer(Tstar);

        bool is_atomic = (get_mol(0).get_Natoms() == 1);
        bool is_linear = true; // TODO: check if true with principal axes

        int feval_max = 0;
        if (m_conf.contains("feval_max")) {
            feval_max = static_cast<int>(m_conf["feval_max"]);
        }
        else {
            throw std::invalid_argument("Key \"feval_max\" must be specified in the configuration JSON");
        }
        

        if (is_atomic) {
            // Fourth virial coefficient: three parts B_4_1,B_4_2,B_4_3
            std::vector<std::valarray<double>> vals = { allocate_buffer(Tstar), allocate_buffer(Tstar), allocate_buffer(Tstar) };
            std::vector<std::valarray<double>> errs = { allocate_buffer(Tstar), allocate_buffer(Tstar), allocate_buffer(Tstar) };
            
            //typedef int (*integrand) (unsigned ndim, const double *x, void *, unsigned fdim, double* fval);
            auto cubature_integrand_1 = [](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) {
                auto& shared = *((SharedData*)(p_shared_data));
                shared.atomic_B4_1_integrand(x[0], x[1], x[2], x[3], x[4], fval); // r14, r13, gamma, r12, eta
                return 0; // success
            };

            auto cubature_integrand_2 = [](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) {
                auto& shared = *((SharedData*)(p_shared_data));
                shared.atomic_B4_2_integrand(x[0], x[1], x[2], x[3], x[4], fval); // eta, r12, r13, eta, r14
                return 0; // success
            };

            auto cubature_integrand_3 = [](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) {
                auto& shared = *((SharedData*)(p_shared_data));
                shared.atomic_B4_3_integrand(x[0], x[1], x[2], x[3], x[4], x[5], fval); // eta, zeta, gamma, r12, r13, r14
                return 0; // success
            };

            unsigned fdim = static_cast<unsigned>(vals[0].size()); // How many output dimensions

            // Limits on r14, r13, gamma, r12, eta
            // Limits on eta, r12, r13, eta, r14
            // Limits on eta,zeta,gamma, r12, r13, r14
            std::vector<std::valarray<double>> xmins = { {rstart, rstart, -1 , rstart, -1 }, { -1, rstart, rstart, -1 , rstart }, { -1,  rstart, -1 , rstart, rstart , rstart } };
            std::vector<std::valarray<double>> xmaxs = { { rend, rend, 1 , rend, 1 }, { 1, rend, rend, 1 , rend }, { 1, 2 * M_PI, 1, rend, rend , rend } };

            int naxes = 5; // How many dimensions the integral is taken over (r14, r13, gamma, r12, eta)
            hcubature(fdim, cubature_integrand_1, &shared, naxes, &(xmins[0][0]), &(xmaxs[0][0]), feval_max, 0, 1e-4, ERROR_INDIVIDUAL, &(vals[0][0]), &(errs[0][0]));
            hcubature(fdim, cubature_integrand_2, &shared, naxes, &(xmins[1][0]), &(xmaxs[1][0]), feval_max, 0, 1e-4, ERROR_INDIVIDUAL, &(vals[1][0]), &(errs[1][0]));

            naxes = 6;
            hcubature(fdim, cubature_integrand_3, &shared, naxes, &(xmins[2][0]), &(xmaxs[2][0]), feval_max, 0, 1e-4, ERROR_INDIVIDUAL, &(vals[2][0]), &(errs[2][0]));

            // prefactors for each contribution 
            std::valarray<double> pre_factors = { -3.0*(27.0/4.0), 3.0*(27.0/2.0), -27.0/(8.0*M_PI) };
            for (auto i = 0; i < pre_factors.size(); ++i) {
                vals[i] *= pre_factors[i];
                errs[i] *= pre_factors[i];
            }

            // Copy into output
            outval = vals[0] + vals[1] + vals[2];
            outerr = std::abs(errs[0]) + std::abs(errs[1]) + std::abs(errs[2]);
        }
        else if (is_linear) {
            throw std::invalid_argument("Not yet able to handle linear molecules");
        }
        else {
            throw std::invalid_argument("Not yet able to handle non-linear molecules");
        }
        return make_output_tuple(Tstar, std::move(outval), std::move(outerr));
    }

    /**
    Do the calculations for one temperature

    @param order The order of the virial coefficient (2=B_2, 3=B_3, etc.)
    @param Tstar The temperature
    @param rstart The initial value of r to be considered in integration
    @param rend The final value of r to be considered in integration
    @returns Tuple of (value, estimated error in value)
    @note The return numerical type maybe be one of double, std::complex<double> or MultiComplex<double>
    */
    template <typename TEMPTYPE>
    std::tuple<TEMPTYPE,TEMPTYPE> one_temperature(int order, TEMPTYPE Tstar, TYPE rstart, TYPE rend) const 
    {
        if (order == 2) {
            return one_temperature_B2(Tstar, rstart, rend);
        }
        else if (order == 3) {
            return one_temperature_B3(Tstar, rstart, rend);
        }
        else if (order == 4) {
            return one_temperature_B4(Tstar, rstart, rend);
        }
        else {
            throw std::invalid_argument("This order is not supported");
        }
    }
    
    std::map<std::string, double> B_and_derivs(int order, int Nderivs, double T, double rstart, double rend){
        
        if (Nderivs == 0) {
            auto [val,esterr] = this->one_temperature(order, T, rstart, rend);
            return {
                {"T", T},
                {"B", val},
                {"error(B)", esterr}
            };
        }
        if (Nderivs == 1) {
            double h = 1e-100;
            auto [val,esterr] = this->one_temperature(order, std::complex<double>(T,h), rstart, rend);
            return {
                {"T", T},
                {"B", val.real()},
                {"error(B)", esterr.real()},
                {"dBdT", val.imag()/h},
                {"error(dBdT)", esterr.imag()/h},
            };
        }
        else {
            std::function<std::tuple<MultiComplex<double>,MultiComplex<double>>(const MultiComplex<double>&)> f(
                [this, order, rstart, rend](const MultiComplex<double>& T) {
                    return this->one_temperature(order, T, rstart, rend);
                });
            bool and_val = true;
            auto [val,esterr] = diff_mcx1(f, T, Nderivs, and_val);
            std::map<std::string, double> o = { {"T", T} };
            for (auto i = 0; i <= Nderivs; ++i) {
                switch (i) {
                case 0:
                    o["B"] = val[0];
                    o["error(B)"] = esterr[0];
                    break;
                case 1:
                    o["dBdT"] = val[1];
                    o["error(dBdT)"] = esterr[1];
                    break;
                default:
                    auto n = std::to_string(i);
                    o["d" + n + "BdT" + n] = val[i];
                    o["error(d" + n + "BdT" + n + ")"] = esterr[i];
                }
            }
            return o;
        }
    }
    auto parallel_B_and_derivs(int order, int Nthreads, int Nderivs, std::vector<double> Tvec, double rstart, double rend)
    {
        init_thread_pool(Nthreads);
        std::vector<double> times(Tvec.size());
        std::vector<std::map<std::string, double>> outputs(Tvec.size());
        std::size_t i = 0;
        for (auto T : Tvec) {
            auto& result = outputs[i];
            auto& time = times[i];
            std::function<void(void)> one_Temp = [this, order, Nderivs, T, rstart, rend, &result, &time]() {
                auto startTime = std::chrono::high_resolution_clock::now();
                result = this->B_and_derivs(order, Nderivs, T, rstart, rend);
                auto endTime = std::chrono::high_resolution_clock::now();
                time = std::chrono::duration<double>(endTime - startTime).count(); 
                {
                    mtx.lock();
                    std::cout << "Done " << T << " in " << time << " seconds\n" ;
                    mtx.unlock();
                }
                result["elapsed / s"] = time;
            };
            m_pool->AddJob(one_Temp);
            i++;
        }
        // Wait until all the threads finish...
        m_pool->WaitAll();
        return outputs;
    }
    //TYPE orientation_averaged_integrate(const TYPE Tstar, const TYPE rstart, const TYPE rend) {
    //    using arr = Eigen::Array<TYPE, Eigen::Dynamic, 1>;
    //    bool parallel = false;
    //    if (parallel){
    //        // Parallel
    //        TYPE result, time;
    //        Molecule<TYPE> mol1 = this->mol1, mol2 = this->mol2;
    //        auto one_temperature_job = [this, Tstar, rstart, rend, mol1, mol2, &result, &time]() {
    //            auto startTime = std::chrono::high_resolution_clock::now();
    //            result = one_temperature<double>(Tstar, rstart, rend, mol1, mol2);
    //            auto endTime = std::chrono::high_resolution_clock::now();
    //            time = std::chrono::duration<double>(endTime - startTime).count();
    //        };
    //        m_pool->AddJob(one_temperature_job);
    //        // Wait until all the threads finish...
    //        m_pool->WaitAll();
    //        return result;
    //    }
    //    else {
    //        // Serial
    //        return one_temperature(Tstar, rstart, rend, mol1, mol2);
    //    }
    //}
};
