"""
Analysis for the square-well potential defined by

      {oo         r < sigma
V =   {-epsilon   sigma < r < lambda*sigma
      {0          r > lambda*sigma

Collision integrals are as defined in the analysis of:

Eugene M. Holleran, and Hugh M. Hulburt, J. Chem. Phys. 19, 232 (1951); 
https://doi.org/10.1063/1.1748167

Holleran's d is 1/lambda

"""
import numpy as np
from scipy.special import factorial
from scipy.integrate import quad
import matplotlib.pyplot as plt

# *****************************************
#          THERMODYNAMICS 
# *****************************************

def get_B2repatt(*, Tstar, lambda_):
    rep = 2*np.pi/3
    betaepsilon = 1/Tstar
    Y = np.exp(betaepsilon)-1
    att = -2*np.pi/3*Y*(lambda_**3-1)
    return rep, att

def neff_B_total(Tstar, lambda_):
    return 3*Tstar*(Tstar*((1-np.exp(1/Tstar))*(lambda_**3-1)+1) 
        + (lambda_**3-1)*np.exp(1/Tstar))*np.exp(-1/Tstar)/(lambda_**3 - 1)

def neff_B_att(Tstar):
    """ SYMPY:
    lambda_, Tstar = symbols('lambda, T^*')
    Y = exp(1/Tstar)-1
    B2 = 4*(1-(lambda_**3-1)*Y)
    neff = -3*(B2+Tstar*diff(B2, Tstar))/(2*Tstar*diff(B2,Tstar)+Tstar**2*diff(B2,Tstar,2))
    simplify(neff)
    """
    return -3*Tstar**2 + 3*Tstar**2*np.exp(-1/Tstar) + 3*Tstar

def get_TBoyle(lambda_):
    return 1/np.log(1+1/(lambda_**3-1))

# *****************************************
#          COLLISION INTEGRALS 
# *****************************************

def get_theta(*, G, d, l):
    if l == 2:
        if G == 0 and d == 0:
            return 2/5, 0.0
        elif G < d:
            def f(beta):
                """ Eq. 36c """
                return (1-np.cos(2*(np.arcsin(G*beta)-np.arcsin(beta)-np.arcsin(G*beta/d)))**2)*beta
            return 3/2*np.array(quad(f, 0, 1))
        else:
            def f(beta):
                """ Eq. 36d """
                return beta*(1 - np.cos(2*(np.arcsin(G*beta)-np.arcsin(beta)))**2 
                        + d**2/G**2*np.cos(2*(np.arcsin(d*beta)-np.arcsin(d*beta/G)))**2 
                        - d**2/G**2*np.cos(2*(np.arcsin(d*beta)-np.arcsin(beta)-np.arcsin(d*beta/G)))**2) 
            return 3/2*np.array(quad(f, 0, 1))
    else:
        if G == 0 and d == 0:
            return 1/3, 0.0
        raise ValueError()

def get_Omegalk(*, Q, d, l, k):
    def den_integrand(G):
        return np.exp(-Q*G**2/(1-G**2))*G**k/(1-G**2)**((k+3)/2)
    def num_integrand(G):
        return den_integrand(G)*get_theta(G=G,d=d,l=l)[0]
    num = quad(num_integrand, 0, 1, points = [d])
    if k == 7:
        """
        Sympy:
        k=7; Q = symbols('Q', positive=True); G = symbols('G')
        integrate(exp(-Q*G**2/(1-G**2))*G**k/(1-G**2)**((k+3)//2), (G,0,1))
        """
        den_exact = 3/Q**4
    else:
        raise ValueError(k)
    return float(num[0]*Q**((k+1)/2))

def neff_eta_cheb(*, d, Qmin, Qmax, full_output=False):
    chOmega27 = np.polynomial.Chebyshev.interpolate(
        lambda Qs: [get_Omegalk(Q=Q, d=d, l=2, k=7) for Q in Qs], 
        deg=100, 
        domain=[Qmin, Qmax])
    derchOmega27 = chOmega27.deriv(1)
    if not full_output:
        return lambda Q: -2*chOmega27(Q)/(-Q*derchOmega27(Q))
    else:
        return {
        'Omega27': chOmega27,
        'dOmega27dQ': derchOmega27
        }

def plot_neff(lambda_, ofname):
    d = 1/lambda_
    TinvBoyle = np.log(1 + 1/(lambda_**3 - 1))
    TBoyle = 1/TinvBoyle

    fig,ax = plt.subplots(1,1,figsize=(3.5,3))
    Tmin, Tmax = 6e-2, 2000
    Q = 1/np.logspace(np.log10(Tmin), np.log10(Tmax), 1000)
    neff = neff_eta_cheb(d=d, Qmin = 1/Tmax, Qmax=1/Tmin)(Q)
    plt.plot(TinvBoyle/Q, neff, dashes=[3,3])
    
    Tstar = 1/Q
    plt.plot(TinvBoyle/Q, neff_B_total(Tstar, lambda_))
    plt.ylim(0,100)
    plt.xlim(0.01, 5)
    plt.xscale('log')
    plt.xlabel(r'$T/T_{\rm Boyle}$')
    plt.ylabel(r'$n_{\rm eff}$')
    plt.tight_layout(pad=0.2)
    if ofname is not None:
        plt.savefig(ofname)
        plt.close()
    else:
        plt.show()
    
Holleran_TableI = (
"""G theta1(0.0) theta1(0.4) theta1(0.6) theta1(0.8) theta2(0.0) theta2(0.4) theta2(0.6) theta2(0.8)
0.0  0.3333 0.5000 0.5000 0.5000 0.4000 0.5000 0.5000 0.5000
0.2  0.3879 0.3565 0.4312 0.4735 0.5201 0.4044 0.4611 0.4871
0.4  0.2720 0.3058 0.3652 0.4456 0.4705 0.3942 0.4089 0.4693
0.5  0.2117 0.3478 0.3357 0.4305 0.4147 0.4922 0.3850 0.4575
0.6  0.1553 0.2982 0.3260 0.4140 0.3390 0.4658 0.3940 0.4429
0.7  0.1026 0.2305 0.3301 0.3960 0.2471 0.3752 0.4362 0.4253
0.8  0.0560 0.1659 0.2795 0.3809 0.1472 0.2602 0.3692 0.4156
0.86 0.0323 0.1322 0.2445 0.3748 0.0889 0.1914 0.3054 0.4186
0.88 0.0253 0.1221 0.2332 0.3681 0.0708 0.1698 0.2833 0.4090
0.90 0.0190 0.1127 0.2220 0.3606 0.0538 0.1494 0.2620 0.3959
0.92 0.0133 0.1040 0.2119 0.3522 0.0380 0.1304 0.2408 0.3805
0.94 0.0083 0.0962 0.2023 0.3436 0.0241 0.1132 0.2215 0.3639
0.96 0.0043 0.0895 0.1936 0.3341 0.0125 0.0984 0.2041 0.3458
1.00 0.0000 0.0800 0.1800 0.3200 0.0000 0.0800 0.1800 0.3200""")

def validate_theta2():
    for ir, row in pandas.read_csv(io.StringIO(Holleran_TableI), sep=r'\s+').iterrows():
        G = row['G']
        for k, v in dict(row).items():
            if 'theta1' in k or 'G' in k:
                continue
            else:
                d = float(k.split('(')[1].replace(')',''))
                theta, uncert_theta = get_theta(G=G, d=d, l=2)
                if abs(theta-v)> 0.001:
                    print(v, theta, uncert_theta, G, d)

if __name__ == '__main__':
    import pandas, io
    validate_theta2()
    plot_neff(2, None)