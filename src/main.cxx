#include "potter/potter.hpp"
#include "potter/molecules.hpp"
#include "potter/correlations.hpp"

#include <iostream>
#include <fstream>
#include <cmath>

void check_LJ() {
    std::vector<std::vector<double>> coords0 = { {0,0,0} };
    Molecule<double> m0(coords0), m1(coords0);
    Integrator<double> i(m0, m1);
    std::function<double(double)> f([](double r) {
        double rn6 = 1/(r*r*r*r*r*r); return 4.0*(rn6*rn6 - rn6); }
    );
    i.get_evaluator().connect_potentials(f, 1/* number of sites */);
    // B_2^*
    for (auto Tstar = 1.0; Tstar < 10; Tstar *= 2) {
        double h = 1e-100;
        auto radval = i.radial_integrate_B2(std::complex<double>(Tstar, h), 0.01, 1000, 10000);
        int order = 2, Nderivs = 2;
        auto val = i.B_and_derivs(order, Nderivs, Tstar, 0.01, 1000, i.mol1, i.mol2);
        std::cout << Tstar << "," << radval << "," << val["B"] << "," << val["dBdT"] << "," << val["d2BdT2"] << "," << Bstar_Mie(Tstar, 12, 6) << std::endl;
    }
    // B_3^*=B_3/sigma^6
    auto SQUARE = [](double x) { return x*x; };
    for (auto Tstar = 1.0; Tstar < 10; Tstar *= 2) {
        int order = 3, Nderivs = 2;
        auto val = i.B_and_derivs(order, Nderivs, Tstar, 0.00001, 10000, i.mol1, i.mol2);
        std::cout << Tstar << "," << val["B"] << "+-" << val["error(B)"] << std::endl;
    }
}

void Bntable(int order, double Tmin, double Tmax, double NT, const std::string &filename) {
    std::vector<std::vector<double>> coords0 = { {0,0,0} };
    Molecule<double> m0(coords0), m1(coords0);
    Integrator<double> i(m0, m1);
    std::function<double(double)> f([](double r) { double rn6 = 1/(r*r*r*r*r*r); return 4.0*(rn6*rn6 - rn6); });
    i.get_evaluator().connect_potentials(f, 1/* number of sites */);
    // B_3^*=B_3/sigma^6
    auto SQUARE = [](double x) { return x * x; };
    std::vector<double> Tvec; double dT = (log(Tmax)-log(Tmin))/(NT-1); for (auto i = 0; i < NT; ++i){ Tvec.push_back(exp(log(Tmin)+dT*i)); }
    std::ofstream ofs(filename);
    ofs << "T^* B^* dB^*/dT^* d2B^*/dT^*2 elapsed(s)" << std::endl;
    auto results = i.parallel_B_and_derivs(order, 6 /*Nthreads*/, 2 /* Nderiv */, Tvec, 0.02, 1000, i.mol1, i.mol2); // radius in A, B in A^3/molecule
    for (auto val : results) {
        auto B = val["B"];
        auto dBdT = val["dBdT"];
        auto d2BdT2 = val["d2BdT2"];
        auto T = val["T"];
        auto elapsed = val["elapsed"];
        std::cout << T << " " << B << " " << dBdT << " " << d2BdT2 << " " << elapsed << std::endl;
        ofs << T << " " << B << " " << dBdT << " " << d2BdT2 << " " << elapsed << std::endl;
    }
}

void check_N2(const std::string &filename) {
    auto integr = get_nitrogen();
    auto Nthreads = 1;
    auto Nderiv = 2;
    // The temperature values from Hellmann, MP, 2012, supporting information
    std::vector<double> Tvec = { 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150, 160, 170, 180, 
        190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 
        410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 
        760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 
        1400, 1450, 1500, 1600, 1700, 1800, 1900, 2000, 2200, 2400, 2600, 2800, 3000 };
    std::ofstream ofs(filename);
    ofs << "T/K B/cm^3/mol dB^*/dT^* d2B^*/dT^*2 elapsed(s)" << std::endl;
    auto results = integr.parallel_B_and_derivs(2, Nthreads, Nderiv, Tvec, 2, 100, integr.mol1, integr.mol2); // radius in A, B in A^3/molecule
    for (auto val : results) {
        auto B = (val["B"] + 2*M_PI/3*8) * 6.02214086e23 / 1e24;
        auto dBdT = val["dBdT"] * 6.02214086e23 / 1e24;
        auto d2BdT2 = val["d2BdT2"] * 6.02214086e23 / 1e24;
        auto T = val["T"];
        std::cout << T << " " << B << " " << dBdT << " " << d2BdT2  << std::endl;
        ofs << T << " " << B << " " << dBdT << " " << d2BdT2 << std::endl;
    }
}

void LJChain(int N, const std::string &filename) {
    auto i = get_rigidLJChain(N, 1.0);
    std::ofstream ofs(filename);
    ofs << "T^*,N,B^*,dB^*/dT^*,d2B^*/dT^*2,neff,elapsed / s" << std::endl;
    auto Nthreads = 20;
    auto Nderiv = 2;
    using arr = Eigen::Array<double, Eigen::Dynamic, 1>;
    auto Tmin = 1.0, Tmax = 100000.0;
    auto Ntemps = 20;
    auto ETvec = exp(Eigen::ArrayXd::LinSpaced(Ntemps, log(Tmin), log(Tmax)));
    std::vector<double> Tvec(ETvec.size()); for (auto i = 0; i < ETvec.size(); i += 1) { Tvec[i] = ETvec[i]; }
    double rmin = 0.0000001;
    auto results = i.parallel_B_and_derivs(2, Nthreads, Nderiv, Tvec, rmin, 200, i.mol1, i.mol2);
    for (auto val : results) {
        auto B = val["B"]+2*M_PI/3*pow(rmin,3);
        auto dBdT = val["dBdT"];
        auto d2BdT2 = (val.count("d2BdT2") > 0) ? val["d2BdT2"] : -1e30;
        auto Tstar = val["T"];
        auto elapsed = val["elapsed / s"];

        // Defining the term Q_i \equiv (1/T)^i*d^iB_2/d(1/T)^i...
        // The term 1/T is the analog of tau=Tr/T in equation of state land; the 
        // numerator cancels in the derivative
        auto Q0 = B;
        auto Q1 = -Tstar*dBdT;
        auto Q2 = Tstar*Tstar*d2BdT2 + 2*Tstar*dBdT;
        auto neff = -3*(Q0-Q1)/Q2;

        // Estimate the error in neff from propagation from estimated error in 
        // each of the virial coefficient terms
        auto sq = [](double x){ return x*x; };
        auto err_Q0 = val["error(B)"];
        auto err_Q1 = val["error(dBdT)"]*Tstar;
        auto err_Q2 = sqrt(sq(Tstar*Tstar*val["error(d2BdT2)"]) + sq(2*Tstar*val["error(dBdT)"]));

        auto dneff_dQ0 = -3/Q2;
        auto dneff_dQ1 = 3/Q2;
        auto dneff_dQ2 = 3*(Q0-Q1)/pow(Q2,2);
        auto err_neff = sqrt(sq(dneff_dQ0*err_Q0) + sq(dneff_dQ1*err_Q1) + sq(dneff_dQ2*err_Q2));

        std::stringstream out;
        out << Tstar << "," << N << "," << B << "," << dBdT << "," << d2BdT2 << "," << neff << "," << err_neff << "," << elapsed << std::endl;
        std::string sout(out.str());
        ofs << sout;
        std::cout << sout;
    }
}

int main() {
    check_LJ();
    //check_N2("results_N2.txt");
   // for (auto N = 2; N < 20; N *= 2){
   //    LJChain(N, "results" + std::to_string(N) + ".txt");
   //}
}
