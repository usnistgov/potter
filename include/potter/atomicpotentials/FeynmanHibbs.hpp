#pragma once

#include "potter/potter.hpp"

/**
* The potential function to be integrated is defined by 
\f[
\frac{ u }{\epsilon C_{ \epsilon }} = \left(\frac{ \sigma }{r}\right)^ { \lambda_r } - \left(\frac{ \sigma }{r}\right)^ { \lambda_a } + \sum_{ n = 1 }^ {\infty}\frac{ \Lambda^ {2n} }{48 ^ n\pi^ { 2n }(T^*)^ { n }n!}\left(Q_n(\lambda_r)(\frac{ \sigma }{r})^ { \lambda_r + 2n } - Q_n(\lambda_a)(\sigma / r)^ { \lambda_a + 2n }\right)
\f]
with the variables 
\f[
Q_n(\lambda) = \frac{ (\lambda + 2n - 2)!}{(\lambda - 2)!}
\f]
and the de Boer parameter
\f[
\Lambda = \frac{ h }{\sigma\sqrt{ m\epsilon }}
\f]
*/
auto get_MieFeynmanHibbs_potential(double lambda_r, double lambda_a, double deBoer, int Nmax) {
    constexpr double PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;
    std::vector<std::vector<double>> coords0 = { {0,0,0} };
    double C_e = lambda_r/(lambda_r - lambda_a)*pow(lambda_r/lambda_a, lambda_a/(lambda_r-lambda_a));
    
    auto factorial = [](double x) { return tgamma(1 + x); };

    // This potential function depends on both temperature and separation, everything else is captured by value
    auto ff = [lambda_r, lambda_a, C_e, Nmax, PI, factorial, deBoer](const auto &rstar, const auto &Tstar) {
        // The normal, uncorrected contribution from Mie potential
        double Miepart = pow(rstar, -lambda_r) - pow(rstar, -lambda_a);

        // The summation of Feynman-Hibbs correction terms
        std::common_type_t<decltype(rstar), decltype(Tstar)> FHpart = 0.0;
        for (auto n = 1; n <= Nmax; ++n) {
            auto Qn = [n, factorial](double lambda) { return factorial(lambda + 2*n - 2) / factorial(lambda - 2); };
            FHpart += pow(deBoer*deBoer/(48*PI*PI*Tstar), n)/factorial(n)*(Qn(lambda_r)*pow(rstar, -(lambda_r + 2*n)) - Qn(lambda_a)*pow(rstar, -(lambda_a + 2*n)));
        }        
        return C_e * (Miepart + FHpart);
    };
    return ff;
}

/** Get the dimension of the type of the temperature variable */
template<typename T>
int Tdim(const T& a) {
    if constexpr (std::is_same<T, double>::value) {
        return 1;
    }
    else if constexpr (std::is_same<T, std::complex<double>>::value) {
        return 2;
    }
    else {
        throw std::invalid_argument("temperature type doesn't match");
    }
}

template<typename TdepPotential, typename TType>
auto get_BFH(const TdepPotential& pot, double rmin, double rmax, const TType & Tstar) {
    std::valarray<double> xmins = { rmin }, xmaxs = { rmax };

    struct Shared {
        const TType& Temp; 
        const TdepPotential& pot;
        Shared(const TType& Temp, const TdepPotential& pot) : Temp(Temp), pot(pot) {};
    };
    Shared shared(Tstar, pot);
    potter::c_integrand_function g = [](unsigned ndim, const double* x, void* p_shared_data, unsigned fdim, double* fval) -> int {
        constexpr double PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;
        auto& shared = *((struct Shared*)(p_shared_data));
        double rstar = x[0];
        auto& Tstar = shared.Temp;
        auto val = -2*PI*(exp(-shared.pot(rstar, Tstar)/Tstar) - 1.0)*rstar*rstar;
        unpack_f(val, fval);
        return 0;
    };
    using OutputType = std::valarray<double>;
    auto options = potter::get_HCubature_defaults();
    options["fdim"] = Tdim(Tstar);
    return potter::HCubature<OutputType>(g, &shared, xmins, xmaxs, options);
}