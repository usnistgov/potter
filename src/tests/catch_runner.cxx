#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch/catch.hpp"

#include <functional>

#include "potter/potter.hpp"
#include "potter/molecules.hpp"
#include "potter/molecules/CO2.hpp"
#include "potter/correlations.hpp"
#include "potter/integration.hpp"

TEST_CASE("Basic integration problems", "[integration]") {

    using OutputType = std::valarray<double>;

    SECTION("hcubature") {
        std::valarray<double> xmins = { 0,0,0 }, xmaxs = { 1,1,1 };

        struct Shared { double c = 10.0; } shared;
        potter::c_integrand_function g = [](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) -> int {
            auto& shared = *((struct Shared*)(p_shared_data));
            fval[0] = shared.c * sin(x[0]) * cos(x[1]) * exp(x[2]);
            return 0;
        };

        auto [val, err] = potter::HCubature<OutputType>(g, &shared, xmins, xmaxs, potter::get_HCubature_defaults());
        auto exact = shared.c * 0.664669679781377;
        CHECK(exact == Approx(val[0]).margin(2 * err[0]));
    }
    SECTION("hcubature with class local wrapping") {
        std::valarray<double> xmins = { 0,0,0 }, xmaxs = { 1,1,1 };
        class Shared {
        public:
            double c;
            int g(unsigned ndim, const double* x, void*, unsigned fdim, double* fval) {
                fval[0] = c * sin(x[0]) * cos(x[1]) * exp(x[2]);
                return 0;
            };
        };
        Shared shared2;
        
        potter::IntegrandType::func = std::bind(&Shared::g, &shared2, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
        potter::c_integrand_function func2 = static_cast<potter::c_integrand_function>(potter::IntegrandType::callback);

        auto [val, err] = potter::HCubature<OutputType>(func2, 0, xmins, xmaxs, potter::get_HCubature_defaults());
        auto exact = shared2.c * 0.664669679781377;
        CHECK(exact == Approx(val[0]).margin(2 * err[0]));
    }

    SECTION("hcubature with lambda") {
        std::valarray<double> xmins = { 0,0,0 }, xmaxs = { 1,1,1 };

        struct Shared { double c = 10.0; } shared;
        auto g = [&shared](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) -> int {
            fval[0] = shared.c * sin(x[0]) * cos(x[1]) * exp(x[2]);
            return 0;
        };
        potter::IntegrandWrapper<decltype(g)> iw(g);
        auto r = iw.ptr();

        auto [val, err] = potter::HCubature<OutputType>(r, 0, xmins, xmaxs, potter::get_HCubature_defaults());
        auto exact = shared.c * 0.664669679781377;
        CHECK(exact == Approx(val[0]).margin(2 * err[0]));
    }
}

TEST_CASE("Benchmark basic integration problems", "[integration]") {

    using OutputType = std::valarray<double>;
    std::valarray<double> xmins = { 0,0,0 }, xmaxs = { 1,1,1 };
    struct Shared { double c = 10.0; } shared;

    // The standard approach of casting the shared data (dangerous, and inconvenient)
    potter::c_integrand_function f = [](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) -> int {
        auto& shared = *((struct Shared*)(p_shared_data));
        fval[0] = shared.c * sin(x[0]) * cos(x[1]) * exp(x[2]);
        return 0;
    };

    class Shared2 {
    public:
        double c;
        int g(unsigned ndim, const double* x, void*, unsigned fdim, double* fval) {
            fval[0] = c * sin(x[0]) * cos(x[1]) * exp(x[2]);
            return 0;
        };
    };
    Shared2 shared2;

    potter::IntegrandType::func = std::bind(&Shared2::g, &shared2, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
    potter::c_integrand_function func2 = static_cast<potter::c_integrand_function>(potter::IntegrandType::callback);

    auto g = [&shared](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) -> int {
        fval[0] = shared.c * sin(x[0]) * cos(x[1]) * exp(x[2]);
        return 0;
    };
    potter::IntegrandWrapper<decltype(g)> iw(g);

    BENCHMARK("hcubature") {
        auto [val, err] = potter::HCubature<OutputType>(f, &shared, xmins, xmaxs, potter::get_HCubature_defaults());
        return val[0];
    };
    BENCHMARK("hcubature with class local wrapping") {
        auto [val, err] = potter::HCubature<OutputType>(func2, 0, xmins, xmaxs, potter::get_HCubature_defaults());
        return val[0];
    };
    BENCHMARK("hcubature with lambda") {
        auto r = iw.ptr();
        auto [val, err] = potter::HCubature<OutputType>(r, 0, xmins, xmaxs, potter::get_HCubature_defaults());
        return val[0];
    };
}

TEST_CASE("Check N_2 values", "[B_2],[N2]") {
    auto pot = get_nitrogen();
    SECTION("Check potential evaluation") {
        for (auto row : validdata_nitrogen_potential()) {
            auto actual = row.V12BkB_K;
            auto check = pot.potential(row.r_A, row.theta1_deg/180*M_PI, row.theta2_deg/180 * M_PI, row.phi_deg / 180 * M_PI);
            CAPTURE(row.r_A);
            CAPTURE(row.theta1_deg);
            CAPTURE(row.theta2_deg);
            CAPTURE(row.phi_deg);
            CHECK(std::abs(actual - check) < 0.001);
        }
    }
}

TEST_CASE("Check CO2 values", "[B_2],[CO2]") {
    auto pot = HellmannCarbonDioxide::get_integrator();
    pot.get_conf_view()["feval_max"] = 1e5;
    SECTION("Check potential evaluation") {
        for (auto row : HellmannCarbonDioxide::potential_validdata()) {
            auto actual = row.V12BkB_K;
            auto check = pot.potential(row.r_A, row.theta1_deg/180*M_PI, row.theta2_deg/180*M_PI, row.phi_deg/180*M_PI);
            CAPTURE(row.r_A);
            CAPTURE(row.theta1_deg);
            CAPTURE(row.theta2_deg);
            CAPTURE(row.phi_deg);
            CHECK(std::abs(actual - check) < 0.001);
        }
    }
    SECTION("Check classical second virials") {
        int i = 0;
        for (auto row : HellmannCarbonDioxide::check_B2cl_vals()) {
            if (i % 10 == 0 && row.T_K > 600) {
                auto actual = row.B2cl_cm3mol;
                auto rmin_A = 2.0;
                auto B = pot.B_and_derivs(2, 1, row.T_K, rmin_A, 100, pot.mol1, pot.mol2)["B"];
                auto check = (B + 2 * M_PI / 3 * pow(rmin_A, 3)) * 6.02214086e23 / 1e24; // cm^3/mol
                CHECK(std::abs(actual - check) < std::abs(actual) * 0.01);
            }
            i += 1;
        }
    }
}

TEST_CASE("Check B_2 values against Singh&Kofke values", "[B_2]") {

    std::vector<std::vector<double>> coords0 = { {0,0,0} };
    Molecule<double> m0(coords0), m1(coords0);
    Integrator<double> i(m0, m1);
    i.get_conf_view()["feval_max"] = 1e6;
    std::function<double(double)> f([](double r) {
        double rn6 = 1/(r*r*r*r*r*r); return 4.0 * (rn6*rn6 - rn6); }
    );
    i.get_evaluator().connect_potentials(f, 1/* number of sites */);

    // Singh and Kofke, PRL, doi:10.1103/PhysRevLett.92.220601
    std::vector<double> B2_over_b = { -5.7578, -4.1757, -2.5381, -1.8359, -1.5842, -1.3759, -1.2010, -0.6275, -0.3126, 0.24332, 0.46087 };
    std::vector<double> standarderr = { 0.0003, 0.0003, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.00004, 0.00003 };
    std::vector<double> Tstar = { 0.625, 0.75, 1.0, 1.2, 1.3, 1.4, 1.5, 2, 2.5, 5, 10 };

    for (auto k = 0; k < Tstar.size(); ++k) {
        int order = 2, Nderivs = 0;
        auto val = i.B_and_derivs(order, Nderivs, Tstar[k], 0.00001, 10000, i.mol1, i.mol2);
        auto B2 = val["B"];
        auto B2err_potter = val["error(B)"];

        auto B2_SK = B2_over_b[k] * (2*M_PI/3);
        auto B2err_SK = standarderr[k] * (2*M_PI/3);

        CHECK(std::abs(B2 - B2_SK) < B2err_SK*2);
    }
}

TEST_CASE("Check B_3 values against Singh&Kofke values", "[B_3]") {

    std::vector<std::vector<double>> coords0 = { {0,0,0} };
    Molecule<double> m0(coords0), m1(coords0);
    Integrator<double> i(m0, m1);
    i.get_conf_view()["feval_max"] = 1e7;
    std::function<double(double)> f([](double r) {
        double rn6 = 1/(r*r*r*r*r*r); return 4.0*(rn6*rn6 - rn6); }
    );
    i.get_evaluator().connect_potentials(f, 1/* number of sites */);

    // Singh and Kofke, PRL, doi:10.1103/PhysRevLett.92.220601
    std::vector<double> B3_over_b2 = { -8.237, -1.7923, 0.4299, 0.5922, 0.5881, 0.5682, 0.5433, 0.43703, 0.38100, 0.31507, 0.28601 };
    std::vector<double> standarderr = { 0.002, 0.0007, 0.0002, 0.0004, 0.0002, 0.0001, 0.0001, 0.00008, 0.00004, 0.00009, 0.0001 };
    std::vector<double> Tstar = { 0.625, 0.75, 1.0, 1.2, 1.3, 1.4, 1.5, 2, 2.5, 5, 10 };

    for (auto k = 0; k < Tstar.size(); ++k) {
        int order = 3, Nderivs = 0;
        auto val = i.B_and_derivs(order, Nderivs, Tstar[k], 0.00001, 10000, i.mol1, i.mol2);
        auto B3 = val["B"];
        auto B3err_potter = val["error(B)"];

        auto B3_SK = B3_over_b2[k] * pow(2*M_PI/3, 2);
        auto B3err_SK = standarderr[k] * pow(2*M_PI/3, 2);

        CHECK(std::abs(B3-B3_SK) < B3err_SK*2);
    }
}

//TEST_CASE("Check B_4 values against Singh&Kofke values", "[B_4]") {
//
//	std::vector<std::vector<double>> coords0 = { {0,0,0} };
//	Molecule<double> m0(coords0), m1(coords0);
//	Integrator<double> i(m0, m1);
//    i.get_conf_view()["feval_max"] = 1e7;
//	std::function<double(double)> f([](double r) {
//		double rn6 = 1 / (r*r*r*r*r*r); return 4.0*(rn6*rn6 - rn6); }
//	);
//	i.get_evaluator().connect_potentials(f, 1/* number of sites */);
//
//	// Singh and Kofke, PRL, doi:10.1103/PhysRevLett.92.220601
//	std::vector<double> B4_over_b3 = { -120.82, -18.77 , -0.2697, 0.3385, 0.3168, 0.2701 , 0.2256, 0.12279, 0.1131, 0.1341, 0.1156};
//	std::vector<double> standarderr = { 0.2, 0.03, 0.002, 0.0005, 0.0005, 0.0004, 0.0003, 0.00007, 0.0001, 0.0001, 0.0002};
//	std::vector<double> Tstar = { 0.625, 0.75, 1.0, 1.2, 1.3, 1.4, 1.5, 2, 2.5, 5, 10 };
//
//	for (auto k = 0; k < Tstar.size(); ++k) {
//		int order = 4, Nderivs = 0;
//		auto val = i.B_and_derivs(order, Nderivs, Tstar[k], 1e-5, 400, i.mol1, i.mol2);
//		auto B4 = val["B"];
//		auto B4err_potter = val["error(B)"];
//
//		auto B4_SK = B4_over_b3[k] * pow(2*M_PI/3, 3);
//		auto B4err_SK = standarderr[k] * pow(2*M_PI/3, 3);
//
//		CHECK(std::abs(B4 - B4_SK) < B4err_SK * 2);
//	}
//} 

TEST_CASE("Benchmark transformations","[bench]") {
    auto integr = get_rigidLJChain(7, 1.0);
    auto molA = integr.mol1, molB = integr.mol2;

    BENCHMARK("reset") {
        molA.reset();
        return molA;
    };
    BENCHMARK("reset+translate") {
        molA.reset();
        molA.translatex(1);
        return molA;
    };
    BENCHMARK("reset+rotate+translate") {
        molA.reset();
        molA.rotate_plusy(1.3);
        molA.translatex(1);
        return molA;
    };
    BENCHMARK("reset+rotate+translate+potential") {
        molA.reset();
        molA.rotate_plusy(1.3);
        molA.rotate_negativex(1.3);
        molA.translatex(1);

        molB.reset();
        molB.rotate_plusy(1.3);

        return integr.get_evaluator().eval_pot(molA, molB);
    };
}

// doi: 10.1039/B302780E
TEST_CASE("Check B3 value against MacDowell et al for 2 center LJF") {
    auto i = get_rigidMieChain(2, 12, 0.6 , 3);
    i.get_conf_view()["feval_max"] = 100000000;
    auto Nthreads = 1; 
    auto Nderiv = 0;
    std::vector<double> T = { 2.59147 };
    std::vector<double> B3_Lit = {8.54};
    std::vector<double> standarderr = {0.01};
    auto val = i.B_and_derivs(3, Nderiv, T[0] , 0.0001, 150 , i.mol1, i.mol2);
    auto B3 = val["B"];
    CHECK(std::abs(B3-B3_Lit[0]) < standarderr[0]*2);
}
