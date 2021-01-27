#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

#include "potter/potter.hpp"
#include "potter/molecules.hpp"
#include "potter/molecules/CO2.hpp"
#include "potter/correlations.hpp"

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
            CHECK(std::abs(actual - check) < std::abs(actual) * 0.01);
        }
    }
}

TEST_CASE("Check CO2 values", "[B_2],[CO2]") {
    auto pot = HellmannCarbonDioxide::get_integrator();
    pot.get_conf_view()["feval_max"] = 1e7;
    SECTION("Check potential evaluation") {
        for (auto row : HellmannCarbonDioxide::potential_validdata()) {
            auto actual = row.V12BkB_K;
            auto check = pot.potential(row.r_A, row.theta1_deg/180*M_PI, row.theta2_deg/180*M_PI, row.phi_deg/180*M_PI);
            CAPTURE(row.r_A);
            CAPTURE(row.theta1_deg);
            CAPTURE(row.theta2_deg);
            CAPTURE(row.phi_deg);
            CHECK(std::abs(actual - check) < std::abs(actual) * 0.01);
        }
    }
    SECTION("Check classical second virials") {
        for (auto row : HellmannCarbonDioxide::check_B2cl_vals()) {
            auto actual = row.B2cl_cm3mol;
            auto rmin_A = 2.0;
            auto B = pot.B_and_derivs(2, 1, row.T_K, rmin_A, 100, pot.mol1, pot.mol2)["B"];
            auto check = (B + 2*M_PI/3*pow(rmin_A, 3)) * 6.02214086e23 / 1e24 ; // cm^3/mol
            CHECK(std::abs(actual - check) < std::abs(actual) * 0.01);
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

TEST_CASE("Check B_4 values against Singh&Kofke values", "[B_4]") {

	std::vector<std::vector<double>> coords0 = { {0,0,0} };
	Molecule<double> m0(coords0), m1(coords0);
	Integrator<double> i(m0, m1);
    i.get_conf_view()["feval_max"] = 1e7;
	std::function<double(double)> f([](double r) {
		double rn6 = 1 / (r*r*r*r*r*r); return 4.0*(rn6*rn6 - rn6); }
	);
	i.get_evaluator().connect_potentials(f, 1/* number of sites */);

	// Singh and Kofke, PRL, doi:10.1103/PhysRevLett.92.220601
	std::vector<double> B4_over_b3 = { -120.82, -18.77 , -0.2697, 0.3385, 0.3168, 0.2701 , 0.2256, 0.12279, 0.1131, 0.1341, 0.1156};
	std::vector<double> standarderr = { 0.2, 0.03, 0.002, 0.0005, 0.0005, 0.0004, 0.0003, 0.00007, 0.0001, 0.0001, 0.0002};
	std::vector<double> Tstar = { 0.625, 0.75, 1.0, 1.2, 1.3, 1.4, 1.5, 2, 2.5, 5, 10 };

	for (auto k = 0; k < Tstar.size(); ++k) {
		int order = 4, Nderivs = 0;
		auto val = i.B_and_derivs(order, Nderivs, Tstar[k], 1e-5, 400, i.mol1, i.mol2);
		auto B4 = val["B"];
		auto B4err_potter = val["error(B)"];

		auto B4_SK = B4_over_b3[k] * pow(2*M_PI/3, 3);
		auto B4err_SK = standarderr[k] * pow(2*M_PI/3, 3);

		CHECK(std::abs(B4 - B4_SK) < B4err_SK * 2);
	}
} 
