#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuba.h"

#include <vector>
#ifndef M_PI
#define M_PI 3.141592654
#endif

const std::vector<cubareal> lower = {0.0001, 0.0001, -1}, upper = {200,200,1};

int Integrand(const int *ndim, const cubareal xx[],
  const int *ncomp, cubareal ff[], void *userdata) {

    double Tstar = 10;
    double rstarmax = 0.24697188338026893, potmax = 7110.0857087661225;
    double alpha = 13;
    auto Vmod = [alpha, rstarmax, potmax](double rstar) { return (rstar >= rstarmax) ? 1/(1-6/alpha)*(6/alpha*exp(alpha*(1 - rstar)) - pow(rstar, -6)) : potmax; };
    auto f = [Vmod, Tstar](double r) {return 1.0 - exp(-Vmod(r) / Tstar); };
    auto SQUARE = [](double x) {return x*x;};
  
    auto r12 = xx[0], r13 = xx[1], eta_angle = xx[2];
    auto rangle = sqrt(SQUARE(r12) + SQUARE(r13) - 2 * r12 * r13 * eta_angle);
    ff[0] = SQUARE(r12) * f(r12) * SQUARE(r13) * f(r13) * f(rangle);
    return 0;
}

int ScaledIntegrand(const int *ndim, const cubareal x[],
  const int *ncomp, cubareal result[], void *userdata) {

    std::vector<cubareal> scaledx(*ndim);
    cubareal jacobian = 1.0;
    for (int dim = 0; dim < *ndim; ++dim){
        auto range = upper[dim] - lower[dim];
        jacobian *= range;
        scaledx[dim] = lower[dim] + x[dim]*range;
    }

    Integrand(ndim, &(scaledx[0]), ncomp, result, userdata);

    for (int comp = 0; comp < *ncomp; ++comp){
        result[comp] *= jacobian;
    }
    return 0;
}

/*********************************************************************/

#define NDIM 3
#define NCOMP 1
#define USERDATA NULL
#define NVEC 1
#define EPSREL 1e-4
#define EPSABS 1e-12
#define VERBOSE 0
#define LAST 4
#define SEED 0
#define MINEVAL 10000
#define MAXEVAL 5000000

#define NSTART 10000
#define NINCREASE 500
#define NBATCH 1000
#define GRIDNO 0
#define STATEFILE NULL
#define SPIN NULL

#define NNEW 1000
#define NMIN 2
#define FLATNESS 25.

#define KEY1 47
#define KEY2 1
#define KEY3 1
#define MAXPASS 5
#define BORDER 0.
#define MAXCHISQ 10.
#define MINDEVIATION .25
#define NGIVEN 0
#define LDXGIVEN NDIM
#define NEXTRA 0

#define KEY 0

int main() {
  int comp, nregions, neval, fail;
  cubareal integral[NCOMP], error[NCOMP], prob[NCOMP];
  auto S = 8*M_PI*M_PI/3;
#if 1
  printf("-------------------- Vegas test --------------------\n");

  Vegas(NDIM, NCOMP, ScaledIntegrand, USERDATA, NVEC,
    EPSREL, EPSABS, VERBOSE, SEED,
    MINEVAL, MAXEVAL, NSTART, NINCREASE, NBATCH,
    GRIDNO, STATEFILE, SPIN,
    &neval, &fail, integral, error, prob);

  printf("VEGAS RESULT:\tneval %d\tfail %d\n",
    neval, fail);
  for( comp = NCOMP-1; comp < NCOMP; ++comp )
    printf("VEGAS RESULT:\t%.8f +- %.8f\tp = %.3f\n",
        S*(double)integral[comp], S*(double)error[comp], (double)prob[comp]);
#endif

#if 1
  printf("\n-------------------- Suave test --------------------\n");

  Suave(NDIM, NCOMP, ScaledIntegrand, USERDATA, NVEC,
    EPSREL, EPSABS, VERBOSE | LAST, SEED,
    MINEVAL, MAXEVAL, NNEW, NMIN, FLATNESS,
    STATEFILE, SPIN,
    &nregions, &neval, &fail, integral, error, prob);

  printf("SUAVE RESULT:\tnregions %d\tneval %d\tfail %d\n",
    nregions, neval, fail);
  for( comp = NCOMP - 1; comp < NCOMP; ++comp )
    printf("SUAVE RESULT:\t%.8f +- %.8f\tp = %.3f\n",
      S*(double)integral[comp], S*(double)error[comp], (double)prob[comp]);
#endif

#if 1
  printf("\n------------------- Divonne test -------------------\n");

  Divonne(NDIM, NCOMP, ScaledIntegrand, USERDATA, NVEC,
    EPSREL, EPSABS, VERBOSE, SEED,
    MINEVAL, MAXEVAL, KEY1, KEY2, KEY3, MAXPASS,
    BORDER, MAXCHISQ, MINDEVIATION,
    NGIVEN, LDXGIVEN, NULL, NEXTRA, NULL,
    STATEFILE, SPIN,
    &nregions, &neval, &fail, integral, error, prob);

  printf("DIVONNE RESULT:\tnregions %d\tneval %d\tfail %d\n",
    nregions, neval, fail);
  for( comp = NCOMP - 1; comp < NCOMP; ++comp )
    printf("DIVONNE RESULT:\t%.8f +- %.8f\tp = %.3f\n",
      S*(double)integral[comp], S*(double)error[comp], (double)prob[comp]);
#endif

#if 1
  printf("\n-------------------- Cuhre test --------------------\n");

  Cuhre(NDIM, NCOMP, ScaledIntegrand, USERDATA, NVEC,
    EPSREL, EPSABS, VERBOSE | LAST,
    MINEVAL, MAXEVAL, KEY,
    STATEFILE, SPIN,
    &nregions, &neval, &fail, integral, error, prob);

  printf("CUHRE RESULT:\tnregions %d\tneval %d\tfail %d\n",
    nregions, neval, fail);
  for( comp = 0; comp < NCOMP; ++comp )
    printf("CUHRE RESULT:\t%.8f +- %.8f\tp = %.3f\n",
      S*(double)integral[comp], S*(double)error[comp], (double)prob[comp]);
#endif

  return 0;
}