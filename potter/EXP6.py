import pandas
import numpy as np 
import os
import matplotlib.pyplot as plt 
import json

import scipy.optimize
import scipy.interpolate

import Mie_zero_density

# Mason, dx.doi/org/10.1063/1.1740026
here = os.path.abspath(os.path.dirname(__file__))
possible_alphas = [12,13,14,15]

# Values from Table I of Mason, JCP 1954
# Could be recalculated more accurately
sigma_over_rm = {
    12: 0.8761,
    13: 0.8832,
    14: 0.8891,
    15: 0.8942
}

EXP6_interpolators = {}
for alpha in possible_alphas:
    fname = here+'/EXP6_data/EXP6_CI_alpha'+str(alpha)+'.csv'
    df = pandas.read_csv(fname, dtype=float, comment='#')
    df = df[df['T*'] > 0]
    df['Omega11'] = df['Z11']/(df['T*']*(1-6/alpha))**(1/3)
    df['Omega22'] = df['Z22']/(df['T*']*(1-6/alpha))**(1/3)
    EXP6_interpolators[(alpha,1,1)] = scipy.interpolate.interp1d(np.log(df['T*']), np.log(df['Omega11']))
    EXP6_interpolators[(alpha,2,2)] = scipy.interpolate.interp1d(np.log(df['T*']), np.log(df['Omega22']))

class BFit(object):
    def __init__(self, JSONfile):
        self.df = pandas.DataFrame(json.load(open(JSONfile)))
        self.Binterpolator = scipy.interpolate.interp1d(self.df['T'], self.df['B'])
        self.dBdTinterpolator = scipy.interpolate.interp1d(self.df['T'], self.df['dBdT'])

    def __call__(self, Tstar):
        return self.Binterpolator(Tstar)

    def deriv(self, Tstar):
        return self.dBdTinterpolator(Tstar)

    def get_Boyle(self):
        return scipy.optimize.newton(self, 2)

# def get_lambdaplus_vec(Tstarvec, *, alpha):
#     BI = BFit(os.path.join(here, 'EXP6_data', f'B2_alpha{alpha}_EXP6.json'))
#     T_B = BI.get_Boyle()
#     BB = (BI(Tstarvec) + BI.deriv(Tstarvec)*Tstarvec)**(2/3)
#     Omega22 = np.array([np.exp(EXP6_interpolators[(alpha,2,2)](np.log(T))) for T in Tstarvec])
#     etastar_over_sqrtTstar = 5/(16*np.pi**0.5*Omega22)
#     return Tstarvec/T_B, etastar_over_sqrtTstar*BB*15/4

def check_CI():
    for alpha in possible_alphas:
        fname = here+'/EXP6_CI_alpha'+str(alpha)+'.csv'
        df = pandas.read_csv(fname, dtype=float, comment='#')
        df = df[df['T*']>0.3]
        df['Omega11'] = df['Z11']/(df['T*']*(1-6/alpha))**(1/3)
        df['Omega22'] = df['Z22']/(df['T*']*(1-6/alpha))**(1/3)
        plt.plot(df['T*'],df['Omega11'])
        plt.plot(df['T*'],df['Omega22'])
        y = [np.exp(EXP6_interpolators[(alpha,2,2)](np.log(T))) for T in df['T*']]
        plt.plot(df['T*'], y, 'o')
        # plt.plot(df['T*'], LennardJones126.get_Omegals(df['T*'], l=1,s=1),'o')
        plt.yscale('log')
        plt.xscale('log')
        plt.title(alpha)
        plt.show()

def plot_EXP6_transport():

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(3.5,4), sharex=True)

    T_B_LJ = Mie_zero_density.get_Boyle(n=12,m=6)
    Tstarvec = np.logspace(np.log10(0.3), np.log10(400), 1000)
    Tred = Tstarvec/T_B_LJ
    LJ_eta_plus = np.array([Mie_zero_density.get_eta_plus(Tstar,n=12) for Tstar in Tstarvec])
    LJ_D_plus = np.array([Mie_zero_density.get_D_plus(Tstar,n=12) for Tstar in Tstarvec])
    ax1.plot(Tred, LJ_eta_plus, dashes=[2,2], color='k', label='LJ 12-6')
    ax2.plot(Tred, LJ_D_plus, dashes=[2,2], color='k', label='LJ 12-6')

    Tstarvec = np.logspace(np.log10(0.41), np.log10(199))
    for alpha in possible_alphas:
        BI = BFit(os.path.join(here, 'EXP6_data', f'B2_alpha{alpha}_EXP6.json')) # These are B_2/sigma^3, but the EXP6 analysis assumes in terms of B_2/rm^3
        T_B = BI.get_Boyle()
        
        Theta2 = (BI(Tstarvec) + BI.deriv(Tstarvec)*Tstarvec)**(2/3)

        Omega22 = np.array([np.exp(EXP6_interpolators[(alpha,2,2)](np.log(T))) for T in Tstarvec])
        etastar_over_sqrtTstar = 5/(16*np.pi**0.5*Omega22)
        Omega11 = np.array([np.exp(EXP6_interpolators[(alpha,1,1)](np.log(T))) for T in Tstarvec])
        rhostarDstar_over_sqrtTstar = 3/(8*np.pi**0.5*Omega11)

        ax1.plot(Tstarvec/T_B, etastar_over_sqrtTstar*Theta2, label=r'$\alpha$: '+str(alpha))
        ax2.plot(Tstarvec/T_B, rhostarDstar_over_sqrtTstar*Theta2, label=r'$\alpha$: '+str(alpha))

    ax1.set_ylabel(r'$\eta^+_{\rho \to 0}$')
    ax1.legend(loc='best',ncol=2)
    ax1.set_xlim(0,8)
    ax1.set_ylim(0.24, 0.3)

    ax2.set_xlabel(r'$T^*/T^*_{\rm Boyle}$')
    ax2.set_ylabel(r'$D^+_{\rho \to 0}$')
    # ax2.legend(loc='best',ncol=2)
    # ax2.set_xlim(0,8)
    ax2.set_ylim(0.30, 0.4)

    plt.tight_layout(pad=0.2)
    plt.savefig('EXP6_potentials.pdf')
    plt.show()

if __name__ == '__main__':
    plot_EXP6_transport()