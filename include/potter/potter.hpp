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

double factorial(double n) {
    return std::tgamma(n + 1);
}

std::mutex mtx;  // mutex for cout

#if !defined(M_PI)
constexpr auto M_PI = 3.14159265358979323846;
#endif

/* 
Trapezoidal integration -- the workhorse numerical integration routine
*/
template<typename TYPEX, typename TYPEY>
TYPEY trapz(const Eigen::Array<TYPEX, Eigen::Dynamic, 1>& x, 
           const Eigen::Array<TYPEY, Eigen::Dynamic, 1>& y) {
    TYPEY out = 0;
    for (auto i = 0; i < x.size()-1; ++i) {
        auto ymean = (y[i+1]+y[i])/2.0;
        out = out + ymean*(x[i+1] - x[i]);
    }
    return out;
}
/*
Simpson's integration rule
*/
template<typename TYPEX, typename TYPEY>
TYPEY simps(const Eigen::Array<TYPEX, Eigen::Dynamic, 1>& x, 
           const Eigen::Array<TYPEY, Eigen::Dynamic, 1>& f) {
    // C++ translation of https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data
    auto N = x.size() - 1;
    auto h = x.tail(N) - x.head(N);
    auto cube = [](TYPEX x) { return x * x * x; };
    auto sq = [](TYPEX x) { return x * x; };
    TYPEY result = 0.0;
    for (auto i = 1; i < N; i += 2) {
        auto hph = h[i] + h[i - 1];
        result += f[i] * (cube(h[i]) + cube(h[i - 1]) + 3.0 * h[i] * h[i - 1] * hph) / (6 * h[i] * h[i - 1]);
        result += f[i - 1] * (2.0 * cube(h[i - 1.0]) - cube(h[i]) + 3.0 * h[i] * sq(h[i - 1])) / (6 * h[i - 1] * hph);
        result += f[i + 1] * (2.0 * cube(h[i]) - cube(h[i - 1]) + 3.0 * h[i - 1] * sq(h[i])) / (6 * h[i] * hph);
    }
    if ((N + 1) % 2 == 0) {
        result += f[N] * (2 * sq(h[N - 1]) + 3.0 * h[N - 2] * h[N - 1]) / (6 * (h[N - 2] + h[N - 1]));
        result += f[N - 1] * (sq(h[N - 1]) + 3.0 * h[N - 1] * h[N - 2]) / (6 * h[N - 2]);
        result -= f[N - 2] * cube(h[N - 1]) / (6 * h[N - 2] * (h[N - 2] + h[N - 1]));
    }
    return result;
}

template <typename TYPE>
class Molecule {

public:
    using EColArray = Eigen::Array<TYPE, Eigen::Dynamic, 1>;
    using CoordMatrix = Eigen::Array<TYPE, 3, Eigen::Dynamic>;
    CoordMatrix coords, coords_initial;

    Molecule(const std::vector<std::vector<TYPE>>& pts) {
        coords.resize(3, pts.size());
        for (auto i = 0; i < pts.size(); ++i) {
            auto &pt = pts[i];
            for (auto j = 0; j < pt.size(); ++j) {
                coords(j, i) = pt[j];
            }
        }
        coords_initial = coords;
    };
    void reset() {
        coords = coords_initial;
    };
    CoordMatrix rotZ3(TYPE theta) const {
        // Rotation matrix for rotation around the z axis in 3D
        // See https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations
        return (Eigen::ArrayXXd(3,3) <<
            cos(theta), -sin(theta), 0,
            sin(theta), cos(theta),  0,
                0,           0,      1 ).finished();
    };
    CoordMatrix rotY3(const TYPE theta) const {
        // Rotation matrix for rotation around the y axis in 3D
        // See https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations
        const TYPE c = cos(theta), s = sin(theta);
        return (Eigen::ArrayXXd(3,3) <<
            c,  0, s,
            0,  1, 0,
            -s, 0, c).finished();
    }
    CoordMatrix rotX3(TYPE theta) const {
        // Rotation matrix for rotation around the x axis in 3D
        // See https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations
        return (Eigen::ArrayXXd(3,3) <<
            1,      0,          0,
            0, cos(theta), -sin(theta),
            0, sin(theta), cos(theta)).finished();
    }
    void rotate_plusx(TYPE angle) {
        Eigen::Transform<double, 3, Eigen::Affine> rot(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX()));
        coords = rot.linear() * coords.matrix();
        //coords = rotX3(angle).matrix() * coords.matrix(); // Old method
    }
    void rotate_negativex(TYPE angle) {
        rotate_plusx(-angle);
    }
    void rotate_plusy(TYPE angle) {
        Eigen::Transform<double, 3, Eigen::Affine> rot(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()));
        coords = rot.linear() * coords.matrix();
    }
    void rotate_negativey(TYPE angle) {
        rotate_plusy(-angle);
    }
    void rotate_plusz(TYPE angle) {
        Eigen::Transform<double, 3, Eigen::Affine> rot(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()));
        coords = rot.linear()*coords.matrix();
    }
    void rotate_negativez(TYPE angle) {
        rotate_plusz(-angle);
    }
    void translatex(TYPE dx) {
        coords.row(0) += dx;
    }
    void translatey(TYPE dy) {
        coords.row(1) += dy;
    }
    template<typename ArrayType>
    TYPE get_dist(const ArrayType&rA, const ArrayType&rB) const{
        return sqrt((rA - rB).square().sum());
    }
    Eigen::Index get_Natoms() const {
        return coords.cols();
    }
    auto get_xyz_atom(const Eigen::Index i) const {
        return coords.col(i);
    }
};

template<typename TYPE>
class PotentialEvaluator {
public:
    std::map<std::tuple<std::size_t, std::size_t>, std::function<double(double)> > potential_map;

    TYPE eval_pot(const Molecule<TYPE>& molA, const Molecule<TYPE>& molB) const {
        TYPE u = 0.0;
        for (auto iatom = 0; iatom < molA.get_Natoms(); ++iatom) {
            auto xyzA = molA.get_xyz_atom(iatom);
            for (auto jatom = 0; jatom < molB.get_Natoms(); ++jatom) {
                auto xyzB = molB.get_xyz_atom(jatom);
                TYPE distij = molA.get_dist(xyzA, xyzB);
                auto f_potential = get_potential(iatom, jatom);
                u += f_potential(distij);
            }
        }
        return u;
    }
    /*
    Connect up all the site-site potentials, all molecules have the same number of sites
    */
    void connect_potentials(std::function<double(double)>& f,std::size_t Natoms) {
        for (auto iatom = 0; iatom < Natoms; ++iatom) {
            for (auto jatom = 0; jatom < Natoms; ++jatom) {
                potential_map[std::make_tuple(iatom, jatom)] = f;
            }
        }
    }
    /*
    Add a potential for a particular site-site interaction

    @param iatom The index of site on molecule A
    @param jatom The index of site on molecule B
    */
    void add_potential(std::size_t iatom, std::size_t jatom, std::function<double(double)>& f) {
        potential_map[std::make_tuple(iatom, jatom)] = f;
    }
    /*
    Get a reference to the potential function for a particular site-site interaction
    
    @param i The index of site on molecule A
    @param j The index of site on molecule B

    */
    auto& get_potential(std::size_t i, std::size_t j) const {
        auto itf = potential_map.find(std::make_tuple(i, j));
        if (itf != potential_map.end()) {
            return itf->second;
        }
        else {
            throw std::invalid_argument("Bad potential");
        }
    }
};
/// A helper class
template<typename TYPE, typename TEMPTYPE>
class SharedDataBase {
public:
    TEMPTYPE Tstar;
    TYPE rstar;
    Molecule<TYPE> molA, molB;
    std::valarray<TYPE> xmin, xmax;
    const PotentialEvaluator<TYPE>& evaltr;
    SharedDataBase(TEMPTYPE Tstar, TYPE rstar,
        Molecule<TYPE> molA, Molecule<TYPE> molB, 
        const PotentialEvaluator<TYPE>& evaltr, 
        const std::valarray<TYPE> &xmin, const std::valarray<TYPE> &xmax)
        : Tstar(Tstar), rstar(rstar), molA(molA), molB(molB), evaltr(evaltr), xmin(xmin), xmax(xmax) {};

    TYPE eval_pot(const Molecule<TYPE>& molA, const Molecule<TYPE>& molB) {
        return evaltr.eval_pot(molA, molB);
    };

    /* 
    Given the orientational angles, calculate the integrand
    */
    void orient_integrand(double theta1, double theta2, double phi, double *fval)
    {
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
        
        if constexpr (std::is_same<decltype(Tstar), double>::value) {
            // If T is double (real)
            fval[0] = a;
        }
        else if constexpr (std::is_same<decltype(Tstar), std::complex<double>>::value) {
            // If T is a complex number (perhaps for complex step derivatives)
            fval[0] = a.real();
            fval[1] = a.imag();
        }
        else if constexpr (std::is_same<decltype(Tstar), MultiComplex<double>>::value) {
            // If T is a multicomplex number
            auto &c = a.get_coef();
            for (auto i = 0; i < c.size(); ++i) {
                fval[i] = c[i];
            }
        }
    }
    /*
    Given the separations and angle, calculate the integrand for B_3 for a spherically-symmetric potential
    for an atomic fluid
    */
    void atomic_B3_integrand(const double r12, const double r13, const double eta_angle, double* fval)
    {
        // Get the potential function V(r) that we should use
        auto &pot = this->evaltr.get_potential(0, 0);
        TEMPTYPE Tstar = this->Tstar; // Local reference just for sharing with the lambda function f
        auto f = [pot, Tstar](double r) -> TEMPTYPE { return 1.0 - exp(-pot(r)/Tstar); };
        auto SQUARE = [](double x) { return x*x; };
        auto rangle = sqrt(SQUARE(r12) + SQUARE(r13) - 2*r12*r13*eta_angle);
        auto a = SQUARE(r12)*f(r12)*SQUARE(r13)*f(r13)*f(rangle);

        if constexpr (std::is_same<decltype(Tstar), double>::value) {
            // If T is double (real)
            fval[0] = a;
        }
        else if constexpr (std::is_same<decltype(Tstar), std::complex<double>>::value) {
            // If T is a complex number (perhaps for complex step derivatives)
            fval[0] = a.real();
            fval[1] = a.imag();
        }
        else if constexpr (std::is_same<decltype(Tstar), MultiComplex<double>>::value) {
            // If T is a multicomplex number
            auto& c = a.get_coef();
            for (auto i = 0; i < c.size(); ++i) {
                fval[i] = c[i];
            }
        }
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

		if constexpr (std::is_same<decltype(Tstar), double>::value) {
			// If T is double (real)
			fval[0] = a;
		}
		else if constexpr (std::is_same<decltype(Tstar), std::complex<double>>::value) {
			// If T is a complex number (perhaps for complex step derivatives)
			fval[0] = a.real();
			fval[1] = a.imag();
		}
		else if constexpr (std::is_same<decltype(Tstar), MultiComplex<double>>::value) {
			// If T is a multicomplex number
			auto& c = a.get_coef();
			for (auto i = 0; i < c.size(); ++i) {
				fval[i] = c[i];
			}
		}
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

		if constexpr (std::is_same<decltype(Tstar), double>::value) {
			// If T is double (real)
			fval[0] = a;
		}
		else if constexpr (std::is_same<decltype(Tstar), std::complex<double>>::value) {
			// If T is a complex number (perhaps for complex step derivatives)
			fval[0] = a.real();
			fval[1] = a.imag();
		}
		else if constexpr (std::is_same<decltype(Tstar), MultiComplex<double>>::value) {
			// If T is a multicomplex number
			auto& c = a.get_coef();
			for (auto i = 0; i < c.size(); ++i) {
				fval[i] = c[i];
			}
		}
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

		if constexpr (std::is_same<decltype(Tstar), double>::value) {
			// If T is double (real)
			fval[0] = a;
		}
		else if constexpr (std::is_same<decltype(Tstar), std::complex<double>>::value) {
			// If T is a complex number (perhaps for complex step derivatives)
			fval[0] = a.real();
			fval[1] = a.imag();
		}
		else if constexpr (std::is_same<decltype(Tstar), MultiComplex<double>>::value) {
			// If T is a multicomplex number
			auto& c = a.get_coef();
			for (auto i = 0; i < c.size(); ++i) {
				fval[i] = c[i];
			}
		}
	}
};

template<typename TYPE>
class Integrator {
private:
    std::unique_ptr<ThreadPool> m_pool;
public:
    using EColArray = Eigen::Array<TYPE, Eigen::Dynamic, 1>;
    const Molecule<TYPE> mol1, mol2;
    PotentialEvaluator<TYPE> potcls;

    Integrator(const Molecule<TYPE>& mol1, const Molecule<TYPE>& mol2) : mol1(mol1), mol2(mol2) {};

    /* For a one-dimensional integration for B_2, use trapezoidal integration to calculate B_2 */
    template <typename TEMPTYPE>
    TEMPTYPE radial_integrate_B2(TEMPTYPE Tstar, TYPE rstart, TYPE rend, int N) {
        using arr = Eigen::Array<TYPE, Eigen::Dynamic, 1>;
        using arrT = Eigen::Array<TEMPTYPE, Eigen::Dynamic, 1>;
        arr rv = exp(arr::LinSpaced(N, log(rstart), log(rend)));
        arrT integrand;
        integrand.resize(rv.size());
        Molecule<TYPE> mol1 = this->mol1, mol2 = this->mol2;
        for (auto ir = 0; ir < rv.size(); ++ir) {
            auto r = rv[ir];
            mol2.reset();
            mol2.translatex(r);
            auto V = potcls.eval_pot(mol1, mol2);
            integrand[ir] = (exp(-V/Tstar)-1.0)*r*r;
        }
        return -2*M_PI*trapz(rv, integrand);
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
    /* 
    A helper function to evaluate the potential given COM separation r and the orientation angles
    */
    double potential(double r, double theta1, double theta2, double phi){
        Molecule<TYPE> molA = mol1, molB = mol2;
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
    /*
    Calculate the orientationally-averaged potential
    */
    TYPE orient_averaged_potential(TYPE rstar, Molecule<TYPE> mol1, Molecule<TYPE> mol2) const {
        using SharedData = SharedDataBase<TYPE, double>;
        SharedData shared(0.0, 0.0, mol1, mol2, potcls, {0,0,0}, { M_PI, M_PI, 2 * M_PI });
        //typedef int (*integrand) (unsigned ndim, const double *x, void *, unsigned fdim, double* fval);
        auto f_integrand = [](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) {
            auto& shared = *((class SharedDataBase<double, double>*)(p_shared_data));
            auto& molA = shared.molA;
            auto& molB = shared.molB;
            double theta1 = x[0], theta2 = x[1], phi = x[2];
            shared.orient_integrand(theta1, theta2, phi, fval);
            return 0; // success
        };
        shared.rstar = rstar;
        int ndim = 1;
        std::valarray<double> val(0.0, 4), err(0.0, 4);
        hcubature(ndim, f_integrand, &shared, 3, &(shared.xmin[0]), &(shared.xmax[0]), 100000, 0, 1e-13, ERROR_INDIVIDUAL, &(val[0]), &(err[0]));
        return val[0]/(8*M_PI);
    }

    /**
    Do the calculations for one temperature

    @param order The order of the virial coefficient (2=B_2, 3=B_3, etc.)
    @param Tstar The temperature
    @param rstart The initial value of r to be considered in integration
    @param rend The final value of r to be considered in integration
    @param mol1 The first molecule
    @param mol2 The second molecule
    @returns Tuple of (value, estimated error in value)
    @note The return numerical type maybe be one of double, std::complex<double> or MultiComplex<double>
    */
    template <typename TEMPTYPE>
    std::tuple<TEMPTYPE,TEMPTYPE> one_temperature(int order, TEMPTYPE Tstar, TYPE rstart, TYPE rend, Molecule<TYPE> mol1, Molecule<TYPE> mol2) const 
    {
        
        // Some local typedefs to avoid typing
        using SharedData = SharedDataBase<TYPE, TEMPTYPE>;

        std::valarray<double> xmin, xmax;
        std::valarray<double> xmin_1, xmax_1, xmin_2, xmax_2, xmin_3, xmax_3;
        switch (order)
        {
        case 2:
            xmin = { 0, 0, 0, rstart }, xmax = { M_PI, M_PI, 2 * M_PI, rend }; // Limits on theta1, theta2, phi, r
            break;
        case 3:
            xmin = { rstart, rstart, -1 }, xmax = { rend, rend, 1 }; // Limits on r12, r13, eta
            break;
        case 4:
            xmin_1 = { rstart, rstart, -1 , rstart, -1 }, xmax_1 = { rend, rend, 1 , rend, 1 }; // Limits on r14, r13, gamma, r12, eta
            xmin_2 = { -1, rstart, rstart, -1 , rstart }, xmax_2 = { 1, rend, rend, 1 , rend }; // Limits on  eta, r12, r13, eta, r14
            xmin_3 = { -1,  rstart, -1 , rstart, rstart , rstart }, xmax_3 = { 1, 2 * M_PI, 1, rend, rend , rend }; // Limits on  eta,zeta,gamma, r12, r13, r14	
        default:
            break;
        }
        SharedData shared(Tstar, 0.0, mol1, mol2, potcls, xmin, xmax);
        
        int ndim = 1; // If T is a floating point number (default)
        std::valarray<double> val(0.0, 4), err(0.0, 4);
        std::valarray<double> val_1(0.0, 4), val_2(0.0, 4), val_3(0.0, 4), err_1(0.0, 4), err_2(0.0, 4), err_3(0.0, 4);
        if constexpr (std::is_same<decltype(shared.Tstar), std::complex<double>>::value) {
            ndim = 2;
        }
        else if constexpr (std::is_same<decltype(shared.Tstar), MultiComplex<double>>::value) {
            ndim = static_cast<int>(shared.Tstar.get_coef().size());
            val.resize(ndim); err.resize(ndim);
            val = val.apply([](const double x) {return 0.0; });
            err = err.apply([](const double x) {return 0.0; });
        }

#if !defined(NO_CUBA)
        auto Cuba_integrand = [](const int *pndim, const cubareal x[], const int *pncomp, cubareal fval[], void *p_shared_data) {
            auto& shared = *((class SharedDataBase<double, TEMPTYPE>*)(p_shared_data));
            
            double theta1, theta2, phi, r;
            double jacobian = 1.0;
            for (auto i  = 0; i < *pndim; ++i){
                auto range = shared.xmax[i] - shared.xmin[i];
                jacobian *= range;
            }
            theta1 = shared.xmin[0] + x[0]*(shared.xmax[0]-shared.xmin[0]);
            theta2 = shared.xmin[1] + x[1]*(shared.xmax[1]-shared.xmin[1]);
            phi =    shared.xmin[2] + x[2]*(shared.xmax[2]-shared.xmin[2]);
            r   =    shared.xmin[3] + x[3]*(shared.xmax[3]-shared.xmin[3]);
            shared.rstar = r;
            shared.orient_integrand(theta1, theta2, phi, fval);
            for (auto i = 0; i < *pncomp; ++i){
                fval[i] *= jacobian;
            }
            return 0; // success
        };
        
        int NVEC = 1;
        int EPSREL = 1e-8;
        int EPSABS = 1e-12;
        int VERBOSE = 0;
        int LAST = 4;
        int MINEVAL = 0;
        int MAXEVAL = 5000000;
        int NSTART = 1000;
        int NINCREASE = 500;
        int NBATCH = 1000;
        int GRIDNO = 0;
        const char *STATEFILE = nullptr;
        void *SPIN = nullptr;
        int neval, fail;
        cubareal integral[ndim], error[ndim], prob[ndim];

        int nregions;
        int KEY = 0;
        auto startTimeC = std::chrono::high_resolution_clock::now();
        Cuhre(4, ndim, Cuba_integrand, &shared, NVEC,
        EPSREL, EPSABS, VERBOSE | LAST,
        MINEVAL, MAXEVAL, KEY,
        STATEFILE, SPIN,
        &nregions, &neval, &fail, integral, error, prob);
        auto endTimeC = std::chrono::high_resolution_clock::now();
        auto timeC = std::chrono::duration<double>(endTimeC - startTimeC).count(); 

        // The quadruple integral needs to be divided by 8*pi, but the leading term in the
        // expression for B_2 is -2\pi, so factor becomes -1/4, or -0.25
        for (auto i = 0; i < val.size(); ++i) {
            val[i] = -0.25 * integral[i];
            err[i] = -0.25 * error[i];
        }
#else
        // Use cubature to do the integration...

        switch (order) {
        case 2:
        {
            // The integrand function
            //typedef int (*integrand) (unsigned ndim, const double *x, void *, unsigned fdim, double* fval);
            auto cubature_integrand = [](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) {
                auto& shared = *((class SharedDataBase<double, TEMPTYPE>*)(p_shared_data));
                double theta1 = x[0], theta2 = x[1], phi = x[2], r = x[3];
                shared.rstar = r;
                shared.orient_integrand(theta1, theta2, phi, fval);
                return 0; // success
            };

            auto naxes = 4; // How many dimensions the integral is taken over (theta, phi1, phi2, r)
            hcubature(ndim, cubature_integrand, &shared, naxes, &(xmin[0]), &(xmax[0]), 100000, 0, 1e-13, ERROR_INDIVIDUAL, &(val[0]), &(err[0]));

            // The quadruple integral needs to be divided by 8*pi, but the leading term in the
            // expression for B_2 is -2\pi, so factor becomes -1/4, or -0.25
            for (auto i = 0; i < val.size(); ++i) {
                val[i] = -0.25 * val[i]; err[i] = -0.25 * err[i];
            }
            break;
        }
        case 3:
        {
            // The integrand function
            //typedef int (*integrand) (unsigned ndim, const double *x, void *, unsigned fdim, double* fval);
            auto cubature_integrand = [](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) {
                auto& shared = *((class SharedDataBase<double, TEMPTYPE>*)(p_shared_data));
                shared.atomic_B3_integrand(x[0], x[1], x[2], fval);
                return 0; // success
            };

            auto naxes = 3; // How many dimensions the integral is taken over (r12, r13, eta)
            hcubature(ndim, cubature_integrand, &shared, naxes, &(xmin[0]), &(xmax[0]), 1000000, 0, 1e-13, ERROR_INDIVIDUAL, &(val[0]), &(err[0]));

            for (auto i = 0; i < val.size(); ++i) {
                val[i] = 8*M_PI*M_PI/3 * val[i]; 
                err[i] = 8*M_PI*M_PI/3 * err[i];
            }
            break;
        }
		case 4:
		{
			// Fourth virial coefficient: three parts B_4_1,B_4_2,B_4_3
			// The integrand functions
			//typedef int (*integrand) (unsigned ndim, const double *x, void *, unsigned fdim, double* fval);
			auto cubature_integrand_1 = [](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) {
				auto& shared = *((class SharedDataBase<double, TEMPTYPE>*)(p_shared_data));
				shared.atomic_B4_1_integrand(x[0], x[1], x[2], x[3], x[4], fval);
				return 0; // success
			};

			auto cubature_integrand_2 = [](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) {
				auto& shared = *((class SharedDataBase<double, TEMPTYPE>*)(p_shared_data));
				shared.atomic_B4_2_integrand(x[0], x[1], x[2], x[3], x[4], fval);
				return 0; // success
			};

			auto cubature_integrand_3 = [](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) {
				auto& shared = *((class SharedDataBase<double, TEMPTYPE>*)(p_shared_data));
				shared.atomic_B4_3_integrand(x[0], x[1], x[2], x[3], x[4], x[5], fval);
				return 0; // success
			};


			// prefactors for each contribution 
			std::valarray<double> pre_factors = { -3.0*(27.0 / 4.0) , 3.0*(27.0 / 2.0) ,  -27.0 / (8.0*M_PI) };

			auto naxes = 5; // How many dimensions the integral is taken over (r14, r13, gamma, r12, eta)
			hcubature(ndim, cubature_integrand_1, &shared, naxes, &(xmin_1[0]), &(xmax_1[0]), 1E8, 0, 1e-4, ERROR_INDIVIDUAL, &(val_1[0]), &(err_1[0]));
			hcubature(ndim, cubature_integrand_2, &shared, naxes, &(xmin_2[0]), &(xmax_2[0]), 1E8, 0, 1e-4, ERROR_INDIVIDUAL, &(val_2[0]), &(err_2[0]));

			naxes = 6;
			hcubature(ndim, cubature_integrand_3, &shared, naxes, &(xmin_3[0]), &(xmax_3[0]), 1E8, 0, 1e-4, ERROR_INDIVIDUAL, &(val_3[0]), &(err_3[0]));


			for (auto i = 0; i < val.size(); ++i) {
				val_1[i] = pre_factors[0] * val_1[i];
				err_1[i] = pre_factors[0] * err_1[i];

				val_2[i] = pre_factors[1] * val_2[i];
				err_2[i] = pre_factors[1] * err_2[i];

				val_3[i] = pre_factors[2] * val_3[i];
				err_3[i] = pre_factors[2] * err_3[i];
			}


			val = val_1 + val_2 + val_3;
			err = std::abs(err_1) + std::abs(err_2) + std::abs(err_3);

			break;
		}
        default:
            throw -1;
        }
#endif
        if constexpr (std::is_same<decltype(Tstar), double>::value) {
            // If T is double (real)
            return std::make_tuple(val[0],err[0]);
        }
        else if constexpr (std::is_same<decltype(Tstar), std::complex<double>>::value) {
            // If T is a complex number (perhaps for complex step derivatives)
            return std::make_tuple(decltype(Tstar)(val[0], val[1]), decltype(Tstar)(err[0], err[1]));
        }
        else if constexpr (std::is_same<decltype(Tstar), MultiComplex<double>>::value) {
            // If T is a multicomplex number
            return std::make_tuple(decltype(Tstar)(val), decltype(Tstar)(err));
        }
    }
    
    std::map<std::string, double> B_and_derivs(int order, int Nderivs, double T, double rstart, double rend, Molecule<TYPE> mol1, Molecule<TYPE> mol2){
        
        if (Nderivs == 0) {
            auto [val,esterr] = this->one_temperature(order, T, rstart, rend, mol1, mol2);
            return {
                {"T", T},
                {"B", val},
                {"error(B)", esterr}
            };
        }
        if (Nderivs == 1) {
            double h = 1e-100;
            auto [val,esterr] = this->one_temperature(order, std::complex<double>(T,h), rstart, rend, mol1, mol2);
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
                [this, order, rstart, rend, mol1, mol2](const MultiComplex<double>& T) {
                    return this->one_temperature(order, T, rstart, rend, mol1, mol2);
                });
            bool and_val = true;
            auto [val,esterr] = diff_mcx1(f, T, Nderivs, and_val);
            return {
                {"T", T},
                {"B", val[0]},
                {"error(B)", esterr[0]},
                {"dBdT", val[1]},
                {"error(dBdT)", esterr[1]},
                {"d2BdT2", val[2]},
                {"error(d2BdT2)", esterr[2]},
            };
        }
    }
    auto parallel_B_and_derivs(int order, int Nthreads, int Nderivs, std::vector<double> Tvec, double rstart, double rend, Molecule<TYPE> mol1, Molecule<TYPE> mol2)
    {
        init_thread_pool(Nthreads);
        std::vector<double> times(Tvec.size());
        std::vector<std::map<std::string, double>> outputs(Tvec.size());
        std::size_t i = 0;
        for (auto T : Tvec) {
            auto& result = outputs[i];
            auto& time = times[i];
            std::function<void(void)> one_Temp = [this, order, Nderivs, T, rstart, rend, mol1, mol2, &result, &time]() {
                auto startTime = std::chrono::high_resolution_clock::now();
                result = this->B_and_derivs(order, Nderivs, T, rstart, rend, mol1, mol2);
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
    TYPE orientation_averaged_integrate(const TYPE Tstar, const TYPE rstart, const TYPE rend) {
        using arr = Eigen::Array<TYPE, Eigen::Dynamic, 1>;
        bool parallel = false;
        if (parallel){
            // Parallel
            TYPE result, time;
            Molecule<TYPE> mol1 = this->mol1, mol2 = this->mol2;
            auto one_temperature_job = [this, Tstar, rstart, rend, mol1, mol2, &result, &time]() {
                auto startTime = std::chrono::high_resolution_clock::now();
                result = one_temperature<double>(Tstar, rstart, rend, mol1, mol2);
                auto endTime = std::chrono::high_resolution_clock::now();
                time = std::chrono::duration<double>(endTime - startTime).count();
            };
            m_pool->AddJob(one_temperature_job);
            // Wait until all the threads finish...
            m_pool->WaitAll();
            return result;
        }
        else {
            // Serial
            return one_temperature(Tstar, rstart, rend, mol1, mol2);
        }
    }
};
