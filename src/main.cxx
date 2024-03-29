#include "potter/potter.hpp"
#include "potter/molecules.hpp"
#include "potter/molecules/CO2.hpp"
#include "potter/correlations.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

#include "nlohmann/json.hpp"

// for convenience
using json = nlohmann::json;

void check_EXP6(int order, double alpha, const std::string &filename) {
    std::vector<std::vector<double>> coords0 = { {0,0,0} };
    Molecule<double> m0(coords0), m1(coords0);
    Integrator<double> i({ m0, m1 });

    std::ofstream ofs(filename);

    // Calculate the radius where the potential is at its maximal value
    // by golden section minimization
    double rstarpotmax, valpotmax;
    std::tie(rstarpotmax, valpotmax) = gss([alpha](double rstar) {
        double pot = 1 / (1 - 6 / alpha) * (6 / alpha * exp(alpha * (1 - rstar)) - pow(rstar, -6));;
        return -pot;
        }, 0.1, 1, 1e-10);
    valpotmax *= -1;

    // Connect the potential
    std::function<double(double)> f([rstarpotmax, valpotmax, alpha](double rstar) {
        // rstar here is r/r_m
        return (rstar >= rstarpotmax) ? 1/(1-6/alpha) * (6/alpha*exp(alpha*(1-rstar))-pow(rstar, -6)) : valpotmax;
        }
    );
    i.get_evaluator().connect_potentials(f, 1/* number of sites */);
    i.get_conf_view()["feval_max"] = 1e5;
    
    // B_2^*=B_2/r_m^3
    // B_3^*=B_3/r_m^6
    double Tmin = 1e-1, Tmax = 1e7;
    int NT = 2000;
    std::vector<double> Tvec; double dT = (log(Tmax) - log(Tmin)) / (NT - 1); for (auto i = 0; i < NT; ++i) { Tvec.push_back(exp(log(Tmin) + dT * i)); }
    int Nderivs = 6;
    const auto results = i.parallel_B_and_derivs(order, 6 /*Nthreads*/, Nderivs, Tvec, 0.0001, 200); // radius in sigma, B in r_m^3/molecule

    // write prettified JSON of results to output file
    std::ofstream o(filename);
    o << std::setw(2) << json(results) << std::endl;
}

void check_LJ() {
    std::vector<std::vector<double>> coords0 = { {0,0,0} };
    Molecule<double> m0(coords0), m1(coords0);
    Integrator<double> i({ m0, m1 });
    std::function<double(double)> f([](double r) {
        double rn6 = 1/(r*r*r*r*r*r); return 4.0*(rn6*rn6 - rn6); }
    );
    
    i.get_evaluator().connect_potentials(f, 1/* number of sites */);
    i.get_conf_view()["feval_max"] = 1e5;
    // B_2^*
    for (auto Tstar = 1.0; Tstar < 10; Tstar *= 2) {
        double h = 1e-100;
        auto radval = i.radial_integrate_B2(std::complex<double>(Tstar, h), 0.01, 1000, 10000);
        int order = 2, Nderivs = 2;
        auto val = i.B_and_derivs(order, Nderivs, Tstar, 0.01, 1000);
        std::cout << Tstar << "," << radval << "," << val["B"] << "," << val["dBdT"] << "," << val["d2BdT2"] << "," << Bstar_Mie(Tstar, 12, 6) << std::endl;
    }
    // B_3^*=B_3/sigma^6
    auto SQUARE = [](double x) { return x*x; };
    for (auto Tstar = 1.0; Tstar < 10; Tstar *= 2) {
        int order = 3, Nderivs = 2;
        auto val = i.B_and_derivs(order, Nderivs, Tstar, 0.00001, 10000);
        std::cout << Tstar << "," << val["B"] << "+-" << val["error(B)"] << std::endl;
    }
}

void Bntable(int order, double Tmin, double Tmax, double NT, const std::string &filename) {
    std::vector<std::vector<double>> coords0 = { {0,0,0} };
    Molecule<double> m0(coords0), m1(coords0);
    Integrator<double> i({ m0, m1 });
    std::function<double(double)> f([](double r) { double rn6 = 1/(r*r*r*r*r*r); return 4.0*(rn6*rn6 - rn6); });
    i.get_evaluator().connect_potentials(f, 1/* number of sites */);
    i.get_conf_view()["feval_max"] = 1e7;

    // B_3^*=B_3/sigma^6
    auto SQUARE = [](double x) { return x * x; };
    std::vector<double> Tvec; double dT = (log(Tmax)-log(Tmin))/(NT-1); for (auto i = 0; i < NT; ++i){ Tvec.push_back(exp(log(Tmin)+dT*i)); }
    std::ofstream ofs(filename);
    ofs << "T^* B^* dB^*/dT^* d2B^*/dT^*2 elapsed(s)" << std::endl;
    auto results = i.parallel_B_and_derivs(order, 6 /*Nthreads*/, 2 /* Nderiv */, Tvec, 0.02, 1000); // radius in A, B in A^3/molecule
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
    integr.get_conf_view()["feval_max"] = 1e6;
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
    auto results = integr.parallel_B_and_derivs(2, Nthreads, Nderiv, Tvec, 2, 100); // radius in A, B in A^3/molecule
    for (auto val : results) {
        auto B = (val["B"] + 2*M_PI/3*8) * 6.02214086e23 / 1e24;
        auto dBdT = val["dBdT"] * 6.02214086e23 / 1e24;
        auto d2BdT2 = val["d2BdT2"] * 6.02214086e23 / 1e24;
        auto T = val["T"];
        std::cout << T << " " << B << " " << dBdT << " " << d2BdT2  << std::endl;
        ofs << T << " " << B << " " << dBdT << " " << d2BdT2 << std::endl;
    }
}

/// Check the CO2 classical values against tabulated values
void check_CO2_classical(const std::string& filename) {
    auto integr = HellmannCarbonDioxide::get_integrator();
    integr.get_conf_view()["feval_max"] = 1e6;
    auto Nthreads = 6;
    auto Nderiv = 3;
    std::vector<double> Tvec, Bvals;
    for (auto el : HellmannCarbonDioxide::check_B2cl_vals()) {
        Tvec.push_back(el.T_K);
        Bvals.push_back(el.B2cl_cm3mol);
    }
    auto results = integr.parallel_B_and_derivs(2, Nthreads, Nderiv, Tvec, 2, 100); // radius in A, B in A^3/molecule
    auto i = 0;
    for (auto &val : results) {
        val["B"] += 2*M_PI/3*8;
        val["B2cl(Hellmann) / A^3/molecule"] = Bvals[i]/(6.02214086e23/1e24);
        val["diff"] = val["B2cl(Hellmann) / A^3/molecule"] - val["B"];
        i++;
    }
    // write JSON of results to output file
    std::ofstream o(filename); o << std::setw(2) << json(results) << std::endl;
}

void check_CO2_model(const std::string &model, const std::string& filename) {

    std::map < std::string, std::function<Integrator<double>()>> factoryfunctionmap = {
        {"Errington", CarbonDioxide::get_Errington_integrator},
        {"Vrabec", CarbonDioxide::get_Vrabec_integrator},
        {"Murthy", CarbonDioxide::get_Murthy_integrator},
        {"Moeller", CarbonDioxide::get_Moeller_integrator},
        {"PotoffSiepmann", CarbonDioxide::get_PotoffSiepmann_integrator},
        {"Merker", CarbonDioxide::get_Merker_integrator},
        {"ZhangDuan", CarbonDioxide::get_ZhangDuan_integrator},
        {"HarrisYung", CarbonDioxide::get_HarrisYung_integrator},
        {"Hellmann", HellmannCarbonDioxide::get_integrator}
    };

    auto integrfactory = factoryfunctionmap[model];
    if (!integrfactory) {
        std::cout << "Bad function: " << model << std::endl;
        return;
    }
    auto integr = integrfactory();

    auto& conf = integr.get_conf_view();
    conf["feval_max"] = 1e7;

    auto Nthreads = 8;
    auto Nderiv = 3;
    //std::vector<double> Tvec = { 250,275,300,325,400,500,600,800,1000,2000,4000,6000,8000,10000 };
    std::vector<double> Tvec = geomspace(250, 10000, 200);
    auto results = integr.parallel_B_and_derivs(2, Nthreads, Nderiv, Tvec, 2, 100); // radius in A, B in A^3/molecule
    auto i = 0;
    for (auto& val : results) {
        val["B"] += 2 * M_PI / 3 * 8;
        val["B / A^3/entity"] = val["B"];
        val["B / m^3/mol"] = val["B"] * (6.02214086e23 / 1e30);
        val["B / L/mol"] = val["B / m^3/mol"]*1000;
        auto B = val["B"];
        auto dBdT = val["dBdT"];
        auto d2BdT2 = (val.count("d2BdT2") > 0) ? val["d2BdT2"] : -1e30;
        auto Tstar = val["T"];

        val["neff"] = -3 * (B + Tstar * dBdT) / (Tstar * Tstar * d2BdT2 + 2 * Tstar * dBdT);
        i++;
    }
    // write JSON of results to output file
    std::ofstream o(filename); o << std::setw(2) << json(results) << std::endl;
}

void calculate_CO2(const std::string& filename) {
    auto integr = HellmannCarbonDioxide::get_integrator();
    auto Nthreads = 6;
    auto Nderiv = 3;
    double Tmin = 200, Tmax = 2e4, rmin = 2; // rmin in A
    int NT = 300;
    std::vector<double> Tvec; double dT = (log(Tmax) - log(Tmin)) / (NT - 1); for (auto i = 0; i < NT; ++i) { Tvec.push_back(exp(log(Tmin) + dT * i)); }
    auto results = integr.parallel_B_and_derivs(2, Nthreads, Nderiv, Tvec, 2, 100); // radius in A, B in A^3/molecule
    // Add a hard core contribution for r from zero to 2 A separation
    for (auto& val : results) {
        val["B"] += 2*M_PI/3*std::pow(rmin, 3);
    }
    // write JSON of results to output file
    std::ofstream o(filename); o << std::setw(2) << json(results) << std::endl;
}
void LJChain(int N, double lsigma, const std::string &filename) {
    auto i = get_rigidLJChain(N, lsigma);
    std::ofstream ofs(filename);
    ofs << "T^*,N,B^*,dB^*/dT^*,d2B^*/dT^*2,neff,esterr(neff),elapsed / s" << std::endl;
    auto Nthreads = 6; // Could be more depending on machine...
    auto Nderiv = 2;
    using arr = Eigen::Array<double, Eigen::Dynamic, 1>;
    auto Tmin = 1.0, Tmax = 100000.0;
    auto Ntemps = 200;
    auto ETvec = exp(Eigen::ArrayXd::LinSpaced(Ntemps, log(Tmin), log(Tmax)));
    std::vector<double> Tvec(ETvec.size()); for (auto i = 0; i < ETvec.size(); i += 1) { Tvec[i] = ETvec[i]; }
    double rmin = 0.0000001;
    auto results = i.parallel_B_and_derivs(2, Nthreads, Nderiv, Tvec, rmin, 200);
    for (auto val : results) {
        auto B = val["B"]+2*M_PI/3*pow(rmin,3);
        auto dBdT = val["dBdT"];
        auto d2BdT2 = (val.count("d2BdT2") > 0) ? val["d2BdT2"] : -1e30;
        auto Tstar = val["T"];
        auto elapsed = val["elapsed / s"];
        auto neff = -3*(B+Tstar*dBdT)/(Tstar*Tstar*d2BdT2 + 2*Tstar*dBdT);

        // Estimate the error in neff from propagation from estimated error in 
        // each of the virial coefficient terms
        auto sq = [](double x){ return x*x; };
        auto err_B = val["error(B)"]; auto err_dBdT = val["error(dBdT)"]; auto err_d2BdT2 = val["error(d2BdT2)"];

        // Partials of neff w.r.t. each term
        /* Sympy:
        Tstar,B,dBdT,d2BdT2 = symbols('Tstar,B,dBdT,d2BdT2')
        neff = -3*(B+Tstar*dBdT)/(Tstar**2*d2BdT2 + 2*Tstar*dBdT)
        print(simplify(diff(neff, B)))
        print(simplify(diff(neff, dBdT)))
        print(simplify(diff(neff, d2BdT2)))
        */
        auto dneff_dB = -3/(Tstar*(Tstar*d2BdT2 + 2*dBdT));
        auto dneff_ddBdT = 6*B/(Tstar*sq(Tstar*d2BdT2 + 2*dBdT)) - 3*Tstar*d2BdT2/sq(Tstar*d2BdT2 + 2*dBdT);
        auto dneff_dd2BdT2 = 3*(B + Tstar*dBdT) / sq(Tstar*d2BdT2 + 2*dBdT);

        // Join into overall error estimate
        auto err_neff = sqrt(sq(dneff_dB*err_B) + sq(dneff_ddBdT*err_dBdT) + sq(dneff_dd2BdT2*err_d2BdT2));

        std::stringstream out;
        out << Tstar << "," << N << "," << B << "," << dBdT << "," << d2BdT2 << "," << neff << "," << err_neff << "," << elapsed << std::endl;
        std::string sout(out.str());
        ofs << sout;
        std::cout << sout;
    }
}

int main() {
    for (auto i : {11,12,13,14,15}){
        check_EXP6(2, i, "B2_alpha"+std::to_string(i)+"_EXP6.json");
    }
    
    //check_EXP6(3, 13, "B3_alpha13_EXP6.csv");
//    check_LJ();
    //check_N2("results_N2.txt");
    //check_CO2_classical("classical_CO2.json");
    //check_Singh();
    //calculate_CO2("results_CO2.json");
    //calculate_N2("results_N2.json");
    //check_CO2_model("Errington", "results_CO2_Errington.json");
    //check_CO2_model("Vrabec", "results_CO2_VrabecStollHasse.json"); 
    //check_CO2_model("PotoffSiepmann", "results_CO2_PotoffSiepmann.json");
    //check_CO2_model("Murthy", "results_CO2_Murthy.json");
    //check_CO2_model("Moeller", "results_CO2_Moeller.json");
    //check_CO2_model("Hellmann", "results_CO2_Hellmann.json");
    //check_CO2_model("Merker","results_CO2_Merker.json");
    //check_CO2_model("ZhangDuan", "results_CO2_ZhangDuan.json");
    //check_CO2_model("HarrisYung", "results_CO2_HarrisYung.json");

//    for (auto N = 2; N <= 2; N *= 2){
//        for (auto lsigma = 0.2; lsigma < 1; lsigma += 0.2) {
//            LJChain(N, lsigma, "results" + std::to_string(N) + "_lsigma" + std::to_string(lsigma) + ".csv");
//        }
//    }
}
