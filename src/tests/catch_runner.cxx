#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

#include "potter/potter.hpp"
#include "potter/molecules.hpp"
#include "potter/correlations.hpp"

TEST_CASE("Check B_2 values against Singh&Kofke values", "[B_2]") {

    std::vector<std::vector<double>> coords0 = { {0,0,0} };
    Molecule<double> m0(coords0), m1(coords0);
    Integrator<double> i(m0, m1);
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


TEST_CASE("Check B_4  values against Singh&Kofke values", "[B_4]") {

	std::vector<std::vector<double>> coords0 = { {0,0,0} };
	Molecule<double> m0(coords0), m1(coords0);
	Integrator<double> i(m0, m1);
	std::function<double(double)> f([](double r) {
		double rn6 = 1 / (r*r*r*r*r*r); return 4.0*(rn6*rn6 - rn6); }
	);
	i.get_evaluator().connect_potentials(f, 1/* number of sites */);

	// Singh and Kofke, PRL, doi:10.1103/PhysRevLett.92.220601
	std::vector<double> B4_over_b3 = { -120.82, -18.77 , -0.2697,0.3385 , 0.3168, 0.2701 , 0.2256, 0.12279, 0.1131, 0.1156};
	std::vector<double> standarderr = { 0.002, 0.003, 0.00002, 0.00005, 0.00005, 0.00004, 0.00003, 0.000007, 0.00001, 0.00001, 0.0002 };
	std::vector<double> Tstar = { 0.625, 0.75, 1.0, 1.2, 1.3, 1.4, 1.5, 2, 2.5, 5, 10 };

	for (auto k = 0; k < Tstar.size(); ++k) {
		int order = 4, Nderivs = 0;
		auto val = i.B_and_derivs(order, Nderivs, Tstar[k], 1e-5, 400, i.mol1, i.mol2);
		auto B4 = val["B"];
		auto B4err_potter = val["error(B)"];

		auto B4_SK = B4_over_b3[k] * pow(2 * M_PI / 3, 2);
		auto B4err_SK = standarderr[k] * pow(2 * M_PI / 3, 2);

		CHECK(std::abs(B4 - B4_SK) < B4err_SK * 2);
	}
} 
