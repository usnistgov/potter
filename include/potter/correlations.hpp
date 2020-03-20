#pragma once
#include <cmath>

/// A type-generic factorial function returning a double-precision result
template <typename TYPE> 
double my_factorial(TYPE k) {
    return tgamma(k + 1);
};

/*
@brief Second virial coefficient for the Mie potential

From Sadus, https://doi.org/10.1063/1.5041320, erratum: missing exponent of m

n is the repulsive exponent (the 12 of 12-6 LJ)
m is the attractive exponent (the 6 of 12-6 LJ)

A re-implementation of the derivations of Jones from 1920s
*/
template <typename TYPE>
TYPE Bstar_Mie(TYPE Tstar, int n, int m) {
    auto F = [n, m](TYPE y) -> TYPE{
        auto the_sum = 0.0;
        for (auto i = 1; i < 200; ++i) {
            auto c = tgamma((i*m - 3.0)/n)/my_factorial(i);
            the_sum += c*pow(y, i);
        }
        return pow(y, 3.0/(n - m)) * (tgamma((n - 3.0) / n) - 3.0 / n * the_sum);
    };
    auto yn = pow(n/(n - m), n) * pow((n - m)/m, m) * pow(Tstar, -(n - m)); // y^n, Eq. 9 from Sadus
    auto y = pow(yn, 1.0/n);
    return 2.0*M_PI/3.0 * F(y);
}