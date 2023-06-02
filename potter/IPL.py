"""
Implementation of inverse-power-law potentials
"""

# Python standard library
from __future__ import division
from math import sqrt, pi
from collections import namedtuple
import io

# Packages from the scipy stack
import matplotlib.pyplot as plt
import scipy.integrate
import numpy as np
import mpmath
from scipy.special import gamma as GammaFunc
import pandas

rhobar_f_map = {12: 1.1, 8:1.2, 4:1.2, 1000000: 1.0}

# TABLE VI. Coefficients of fit equation, Eq. (7).
BarlowTableVI = pandas.read_csv(io.StringIO(
"""0 1 2 3 4 5 6 7 8
0 0 0 0 0 0 0 0 0
0 0 0 37.68093 31.84336 12.88986 0.6244331 7.697069 -17.8162
0 0 0 -249.7449 -28.27901 351.9601 547.0779 290.0482 659.0062
0 0 0 1001.704 -143.4443 -2051.451 -2871.629 -998.4549 -3016.937
0 0 0 -995.2711 710.271 3476.869 4530.886 1376.421 4595.473
0 0 0 -642.922 401.1219 2043.935 2823.564 954.9595 3046.475
0 0 0 -249.4429 151.409 761.7278 1090.136 396.4693 1234.401
0 0 0 -79.73547 48.25811 238.9029 350.585 135.0571 413.1661
0 0 0 -23.60437 14.89787 69.17048 102.5482 41.64036 125.0602
0 0 0 -6.366601 4.362856 19.08495 28.573 12.55009 36.4512"""),sep=' ')
BHS = [0,0,2.094395,2.74156,2.636218,2.1213922,1.566904,1.099218,0.7395]

def get_b(n):
    """
    From: Tai Boon Tan, Andrew J. Schultz & David A. Kofke (2011) Virial coefficients,
    equation of state, and solid-fluid coexistence for the soft sphere model, Molecular Physics, 109:1,
    123-132, DOI: 10.1080/00268976.2010.520041

    8-th order approximation

    See also Barlow, JCP, 2012
    """

    if n == 4:
        Bn = [0,0,7.5934596,9.05096,-16.9058,63.934,-325.87,1993,-10712]
    elif n == 12:
        Bn = [0,0,2.566507, 3.79106644,3.527616,2.11492,0.76952,0.09085,-0.0742] # B2 is from Barlow, JCP, 2012
    elif n == 24:
        Bn = [0,0,2.282163,3.19804,3.18751,2.5338,1.752,1.09,0.48] # Barlow
    elif n == 8:
        Bn = [0,0,3.004449,4.55061,3.21509,0.34931,-0.44798,0.2285,0.10611] # Barlow
    elif n == 24:
        Bn = [0,0,2.282163,3.19804,3.18751,2.5338,1.752,1.09,0.48] # Barlow
    elif n == 50:
        Bn = [0,0,2.174826,2.94499,2.90912,2.3801,1.766,1.14,0.57] # Barlow
    elif n == 1000000:
        return BHS
    else:
        raise ValueError(n)
    return Bn

def alphar_SS(gamma, n=12, **kwargs):
    """
    From: Tai Boon Tan, Andrew J. Schultz & David A. Kofke (2011) Virial coefficients,
    equation of state, and solid-fluid coexistence for the soft sphere model, Molecular Physics, 109:1,
    123-132, DOI: 10.1080/00268976.2010.520041

    8-th order approximation

    See also Barlow, JCP, 2012

    Parameters
    ----------
    gamma: float
        The soft-sphere scaling parameter
    """
    Bn = get_b(n)
    # Eq. 11 from Tan et al., but with the added consideration that the effective virial coefficients in terms of gamma need to be employed
    return sum([Bn[m]/(m-1)*gamma**(m-1) for m in range(2,9)])

def neg_sr_over_R(rho_ND, *, n=12, **kwargs):
    Tstar = kwargs.pop('Tstar',1.0)
    sigma = kwargs.get('sigma',1.0)
    gamma = rho_ND*sigma**3*Tstar**(-3/n)
    dgamma_dTstar = (rho_ND*sigma**3*(-3/n)*Tstar**(-3/n-1))
    try:
        Bn = get_b(n)
        alphar = alphar_SS(gamma,**kwargs)
        Zminus1overgamma = sum([Bn[m]*gamma**(m-2) for m in range(2,9)])
        A10 = Zminus1overgamma*(-3/n)*gamma
        sr_over_R = -1*(A10 + alphar)
    except ValueError:
        A10 = Tstar*(Z(rho_ND,n=n,Tstar=Tstar,**kwargs)-1)/gamma*dgamma_dTstar
        def Zminus1overgamma(gamma):
            rho_ND = gamma/(sigma**3*Tstar**(-3/n))
            return (Z(rho_ND,n=n,Tstar=Tstar,**kwargs)-1)/gamma
        alphar,err = scipy.integrate.quad(Zminus1overgamma,0,gamma)
        sr_over_R = -1*(A10 + alphar)

    return -sr_over_R

def _A_l(*,nu=None,n=None,l=2):
    if nu is not None and n is None:
        pass
    elif nu is None and n is not None:
        nu = n+1
    else:
        raise ValueError(dict(nu=nu,n=n))

    def integrand(v0):
        o = lambda v: 1-v**2-2/(nu-1)*(v/v0)**(nu-1)
        def get_v00(v0):
            if isinstance(nu,int) and nu < 30:
                c = np.zeros((nu,))
                c[-1] = 1 # Constant term
                c[-3] = -1 # 
                c[0] = -2/(nu-1)/v0**(nu-1)
                roots = np.roots(c)
                root = roots[np.isreal(roots) & (roots>0)]
                assert(len(root)==1)
                return np.real(root[0])
            else:
                # General treatment with mpmath
                lower, upper = (1e-16, 10)
                return mpmath.findroot(o, (lower, upper), solver='bisect', verbose=False)

        def chi(v0):
            def inner(v):
                return (1-v**2-2/(nu-1)*(v/v0)**(nu-1))**-0.5
            v00 = get_v00(v0)
            val, err = scipy.integrate.quad(inner, 0, v00)
            return np.pi-2*val
        return (1-np.cos(chi(v0))**l)*v0
    val, err = scipy.integrate.quad(integrand, 0, np.inf)
    return val

A_l_vals = {}
def A_l(*,nu=None,n=None,l=2):
    if (nu is not None and n is None):
        pass
    elif nu is None and n is not None:
        nu = n+1
    elif nu is None and n is not None:
        nu = n+1
    else:
        raise ValueError()

    if (nu,l) not in A_l_vals:
        A_l_vals[(nu,l)] = _A_l(nu=nu,l=l)

    return A_l_vals[(nu,l)]

def F(n):
    return 5*(2/n)**(2/n)/(8*sqrt(pi)*A_l(n=n,l=2)*GammaFunc(4-2/n))

def get_etaplus_IPL(n):
    if np.isinf(n):
        return 5/(16*pi**0.5)*(2*pi/3)**(2/3)
    else:
        B2bar = 2*np.pi/3*GammaFunc(1-3/n)
        etaplus = F(n)*(B2bar*(1-3/n))**(2/3)
        return etaplus

def F_D(n):
    return 3*(2/n)**(2/n)/(8*pi**0.5*A_l(n=n,l=1)*GammaFunc(3-2/n))

def get_rhoDplus_IPL(n):
    if np.isinf(n):
        return 6/(16*pi**0.5)*(2*pi/3)**(2/3)
    else:
        B2bar = 2*np.pi/3*GammaFunc(1-3/n)
        rhoDplus = F_D(n)*(B2bar*(1-3/n))**(2/3)
        return rhoDplus