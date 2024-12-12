"""
This script came from the SI of:

I. Bell, Effective hardness of interaction from thermodynamics and 
viscosity in dilute gases, J. Chem. Phys., 152, 164508 (2020); 
https://doi.org/10.1063/5.0007583

with some slight modifications, especially the addition of units

"""
import numpy as np, scipy.integrate
from scipy.special import factorial
import matplotlib.pyplot as plt
import multicomplex as pymcx
import pandas

k_B = 1.380649e-23 # [J/K]
hbar = 1.054571817e-34 # [J s]
hPlanck = hbar*(2*np.pi)
u = 1.66053906660e-27 # [kg]
N_A = 8.314462618/k_B # [J/(mol*K)]

def exp(x):
    if isinstance(x, np.ndarray) and len(x) > 0 and isinstance(x[0], pymcx.MultiComplex):
        return np.array([x_.exp() for x_ in x])
    else:
        return np.exp(x)

class TangToennies:
    def __init__(self, **params):
        self.__dict__.update(**params)

    def add_recursive(self):
        """ 
        Add the C values by the recurrence relation if they are not provided 
        """
        for n in [6, 7, 8]:
            self.C[2*n] = self.C[2*n-6]*(self.C[2*n-2]/self.C[2*n-4])**3

    def potTT(self, R):
        """
        Return the Tang-Toennies potential V/kB in K as a function of R in nm
        """
        out = self.A*exp(self.a1*R + self.a2*R**2 + self.an1/R + self.an2/R**2)
        bR = self.b*R
        contribs = []
        for n in range(3, self.nmax+1):
            bracket = 1-exp(-bR)*sum(
                [bR**k/factorial(k) for k in range(0, 2*n+1)])
            contribs.append(self.C[2*n]*bracket/R**(2*n))
        out -= sum(contribs)
        return out

    def pot(self, R):
        """
        Return the potential V/kB in K as a function of R in nm

        Also apply the correction function at small separations
        """
        R = np.array(R, ndmin=1) # to array
        out = self.potTT(R)
        mask = R < self.Rcutoff*self.Repsilon
        out[mask] = self.tildeA/R[mask]*exp(-self.tildea*R[mask])
        return out

    def potprimeTT(self, R):
        """
        Return the derivative of the potential V/kB in K with respect to 
        position as a function of R in nm 
        """
        R = np.array(R, ndmin=1) # to array
        v = self.a1*R+self.a2*R**2+self.an1/R+self.an2/R**2
        vprime = (self.a1+2*self.a2*R-self.an1/R**2-2*self.an2/R**3)
        out = self.A*exp(v)*vprime
        summer = 0
        for n in range(3, self.nmax+1):
            bsum = sum([(self.b*R)**k/factorial(k) for k in range(0, 2*n+1)])
            bsumprime = sum(
                [self.b**k*k*R**(k-1)/factorial(k) for k in range(0, 2*n+1)])
            b = 1-exp(-self.b*R)*bsum
            bprime = -exp(-self.b*R)*bsumprime +self.b*exp(-self.b*R)*bsum
            summer += -2*n*self.C[2*n]/R**(2*n+1)*b + self.C[2*n]/R**(2*n)*bprime
        out -= summer
        return out

    def potprime(self, R):
        """
        Return the derivative of the potential V/kB in K with respect to 
        position as a function of R in nm 
        """
        R = np.array(R, ndmin=1) # to array
        out = self.potprimeTT(R)
        mask = R < self.Rcutoff*self.Repsilon
        out[mask] = -self.tildeA*exp(-self.tildea*R[mask])*(
            1/R[mask]**2 + self.tildea/R[mask]
            )
        return out

    def potprime2TT(self, R):
        """
        Return the second derivative of the potential V/kB in K with respect to 
        position as a function of R in nm 
        """
        R = np.array(R, ndmin=1) # to array
        v = self.a1*R+self.a2*R**2+self.an1/R+self.an2/R**2
        vprime = self.a1+2*self.a2*R-self.an1/R**2-2*self.an2/R**3
        vprime2 = 2*self.a2+2*self.an1/R**3+6*self.an2/R**4
        out = self.A*exp(v)*(vprime2 + vprime**2)
        summer = 0
        for n in range(3, self.nmax+1):
            bsum = sum([(self.b*R)**k/factorial(k) for k in range(0, 2*n+1)])
            bsumprime = sum(
                [self.b**k*k*R**(k-1)/factorial(k) for k in range(0, 2*n+1)])
            bsumprime2 = sum(
                [self.b**k*k*(k-1)*R**(k-2)/factorial(k) for k in range(0, 2*n+1)])
            b = 1-exp(-self.b*R)*bsum
            bprime = -exp(-self.b*R)*bsumprime +self.b*exp(-self.b*R)*bsum
            bprime2 = (-exp(-self.b*R)*bsumprime2 
                +2*self.b*exp(-self.b*R)*bsumprime -self.b**2*exp(-self.b*R)*bsum)
            summer += (-4*n*self.C[2*n]/R**(2*n+1)*bprime 
                       +(2*n)*(2*n+1)*self.C[2*n]/R**(2*n+2)*b 
                       +self.C[2*n]/R**(2*n)*bprime2)
        out -= summer
        return out

    def potprime2(self, R):
        """
        Return the second derivative of the potential V/kB in K with respect 
        to position as a function of R in nm 

        Also includes the small separation correction
        """
        R = np.array(R, ndmin=1) # to array
        out = self.potprime2TT(R)
        mask = R < self.Rcutoff*self.Repsilon
        Rm = R[mask]
        out[mask] = self.tildeA*exp(-self.tildea*Rm)*(
            2/Rm**3 + 2*self.tildea/Rm**2 + (Rm*self.tildea)**2/Rm**3)
        return out

    def potprime3TT(self, R):
        """
        Return the third derivative of the Tang-Toennies potential V/kB in K 
        with respect to position as a function of R in nm 
        """
        R = np.array(R, ndmin=1) # to array
        v = self.a1*R+self.a2*R**2+self.an1/R+self.an2/R**2
        vprime = self.a1+2*self.a2*R-self.an1/R**2-2*self.an2/R**3
        vprime2 = 2*self.a2+2*self.an1/R**3+6*self.an2/R**4
        vprime3 = -6*self.an1/R**4-24*self.an2/R**5
        out = self.A*exp(v)*(vprime3 + 3*vprime*vprime2 + vprime**3)
        summer = 0
        for n in range(3, self.nmax+1):
            bsum = sum([(self.b*R)**k/factorial(k) for k in range(0, 2*n+1)])
            bsumprime = sum(
                [self.b**k*k*R**(k-1)/factorial(k) for k in range(0, 2*n+1)])
            bsumprime2 = sum(
                [self.b**k*k*(k-1)*R**(k-2)/factorial(k) for k in range(0, 2*n+1)])
            bsumprime3 = sum(
                [self.b**k*k*(k-1)*(k-2)*R**(k-3)/factorial(k) for k in range(0, 2*n+1)])
            b = 1-exp(-self.b*R)*bsum
            bprime = exp(-self.b*R)*(-bsumprime +self.b*bsum)
            bprime2 = exp(-self.b*R)*(
                -bsumprime2 
                +2*self.b*bsumprime -self.b**2*bsum)
            bprime3 = (exp(-self.b*R)*(
                -bsumprime3 +2*self.b*bsumprime2 -self.b**2*bsumprime) 
                -self.b*exp(-self.b*R)*(-bsumprime2 +2*self.b*bsumprime 
                -self.b**2*bsum))
            summer += (-4*n*self.C[2*n]/R**(2*n+1)*bprime2
                       +4*n*(2*n+1)*self.C[2*n]/R**(2*n+2)*bprime 
                       + (2*n)*(2*n+1)*self.C[2*n]/R**(2*n+2)*bprime 
                       - (2*n)*(2*n+1)*(2*n+2)*self.C[2*n]/R**(2*n+3)*b 
                        + self.C[2*n]/R**(2*n)*bprime3
                        - 2*n*self.C[2*n]/R**(2*n+1)*bprime2
                        )
        out -= summer
        return out

    def potprime3(self, R):
        """
        Return the third derivative of the potential V/kB in K with respect 
        to position as a function of R in nm 

        Also includes the small separation correction
        """
        R = np.array(R, ndmin=1) # to array
        out = self.potprime3TT(R)
        mask = R < self.Rcutoff*self.Repsilon
        Rm = R[mask]
        out[mask] = -self.tildeA*exp(-self.tildea*Rm)*(
            6/Rm**4 + 6*self.tildea/Rm**3 
            +3*(Rm*self.tildea)**2/Rm**4 
            +(Rm*self.tildea)**3/Rm**4)
        return out

    def fit_tildes(self, R):
        pot = self.potTT(R)
        dpot = self.potprimeTT(R)
        def objective(tildes):
            tildeA, tildea = tildes 
            val = tildeA/R*exp(-tildea*R)
            deriv = tildeA*(-tildea*1/R*exp(-tildea*R) + -1/R**2*exp(-tildea*R))
            residues = [val-pot, deriv-dpot]
            return np.array(residues)
        res = scipy.optimize.differential_evolution(
            lambda x: (objective(x)**2).sum(), 
            bounds=[(1e4,1e8),(1e1,1e2)],disp=
            True
            )
        print(res)
        return res.x

NeonTT = TangToennies(
    A =     0.402915058383e+08,
    a1 =   -0.428654039586e+02,
    a2 =   -0.333818674327e+01,
    an1 =  -0.534644860719e-01,
    an2 =   0.501774999419e-02,
    b =     0.492438731676e+02,
    nmax =  8,
    C = {
        6:  0.440676750157e-01,
        8:  0.164892507701e-02,
        10: 0.790473640524e-04,
        12: 0.485489170103e-05,
        14: 0.382012334054e-06,
        16: 0.385106552963e-07
    },
    tildeA = 2.36770343e+06, # Fit in this work
    tildea = 3.93124973e+01, # Fit in this work
    Rcutoff = 0.4,
    mass_rel = 20.1797,
    Repsilon = 0.30894556,
    key = 'Bich-MP-2008-Ne',
    doi = '10.1080/00268970801964207'
)
# print(NeonTT.fit_tildes(NeonTT.Repsilon*NeonTT.Rcutoff))

ArgonTT = TangToennies(
    A =    4.61330146e7,
    a1 =  -2.98337630e1,
    a2 =  -9.71208881,
    an1 =  2.75206827e-2,
    an2 = -1.01489050e-2,
    b =    4.02517211e1,
    nmax = 8,
    C = {
        6: 4.42812017e-1,
        8: 3.26707684e-2,
        10: 2.45656537e-3,
        12: 1.88246247e-4,
        14: 1.47012192e-5,
        16: 1.17006343e-6
    },
    tildeA=9.36167467e5,
    tildea=2.15969557e1,
    epsilonkB=143.123,
    Repsilon=0.376182,
    Rcutoff=0.4,
    sigma = 0.335741,
    mass_rel = 39.948,
    key = 'Vogel-MP-2010-Ar',
    doi = '10.1080/00268976.2010.507557'
)
# print(ArgonTT.fit_tildes(ArgonTT.Repsilon*ArgonTT.Rcutoff))

KryptonTT = TangToennies(
    A =    0.3200711798e8,
    a1 =  -0.2430565544e1  *10,
    a2 =  -0.1435536209    *10**2,
    an1 = -0.4532273868    /10,
    an2 =  0,
    b =    0.2786344368e1  *10,
    nmax = 8,
    C = {
        6: 0.8992209265e6  /10**6,
        8: 0.7316713603e7  /10**8,
        10: 0.7835488511e8 /10**10
    },
    tildeA = 0.8268005465e7 /10,
    tildea = 0.1682493666e1 *10,
    epsilonkB = 200.8753,
    Repsilon = 4.015802     /10,
    Rcutoff = 0.3,
    mass_rel = 83.798,
    key = 'Jaeger-JCP-2016-Kr',
    doi = '10.1063/1.4943959'
)
KryptonTT.add_recursive()

XenonTT = TangToennies(
    A = 0.579317071e8,
    a1 = -0.208311994e1   *10,
    a2 = -0.147746919     *10**2,
    an1 = -0.289687722e1  /10,
    an2 = 0.258976595e1   /10**2,
    b = 0.244337880e1     *10,
    nmax = 8,
    C = {
        6:  0.200298034e7 /10**6, 
        8:  0.199130481e8 /10**8,
        10: 0.286841040e9 /10**10
    },
    tildeA = 4.18081481e+06, # Fit in this work
    tildea = 2.38954061e+01, # Fit in this work
    Rcutoff = 0.3,
    Repsilon = 4.37798    /10,
    mass_rel = 131.293,
    key = 'Hellmann-JCP-2017-Xe',
    doi = '10.1063/1.4994267'
)
XenonTT.add_recursive()
# print(XenonTT.fit_tildes(XenonTT.Repsilon*XenonTT.Rcutoff))

def diffassert(val,thresh, reference=''):
    if val > thresh:
        print(val, thresh, reference)
        assert(val < thresh)

### Check the analytic T-T derivatives w.r.t. R with complex step derivative
h = 1e-100
diffassert(abs(ArgonTT.potprime2TT(0.5+1j*h).imag/h - ArgonTT.potprime3TT(0.5)), 1e-12)
diffassert(abs(ArgonTT.potprimeTT(0.5+1j*h).imag/h - ArgonTT.potprime2TT(0.5)), 1e-12)
diffassert(abs(ArgonTT.potTT(0.5+1j*h).imag/h - ArgonTT.potprimeTT(0.5)), 1e-12, ArgonTT.potprimeTT(0.5))
diffassert(abs(ArgonTT.potprime2(0.1+1j*h).imag/h - ArgonTT.potprime3(0.1)), 1e-30)
diffassert(abs(ArgonTT.potprime(0.1+1j*h).imag/h - ArgonTT.potprime2(0.1))/ArgonTT.potprime2(0.1), 1e-30)
diffassert(abs(ArgonTT.pot(0.1+1j*h).imag/h - ArgonTT.potprime(0.1))/ArgonTT.potprime(0.1), 1e-30)

# Check the potential against values in Table 2 in Hellmann:
diffassert(abs(NeonTT.pot(0.16)- 26879.940), 0.001*26879.940)
x = np.array([1e-2, 0.56, 0.16, 0.56])
diffassert(abs(NeonTT.pot(x)[2]- 26879.940), 0.001*26879.940)
diffassert(abs(NeonTT.pot(0.56)- (-1.631)), 0.001*abs(-1.631))
x = np.array([0.56])
diffassert(abs(NeonTT.pot(x)- (-1.631)), 0.001*abs(-1.631))

### Check the potential against values in Table 2 in Jager:
diffassert(abs(ArgonTT.pot(0.20)- 51406.200), 0.001*51406.200)
diffassert(abs(ArgonTT.pot(0.9)- (-0.918)), 0.004*abs(-0.918))

### Check the potential against values in Table IV in Jager, MP, 2016:
diffassert(abs(KryptonTT.pot(0.24)- 27872.324), 0.004*27872.324)
diffassert(abs(KryptonTT.pot(0.4)- (-200.741)), 0.004*abs(-200.741))
diffassert(abs(KryptonTT.pot(1.00)- (-0.982)), 0.004*abs(-0.982))

### Check the potential against values in Table II in Hellmann:
diffassert(abs(XenonTT.pot(0.26)- 37578.501 ), 0.001*37578.501)
diffassert(abs(XenonTT.pot(0.9)- (-4.343)), 0.004*abs(-4.343))

def get_potvals (Rmin_m, Rmax_m, N, *, pot):
    R_m = np.logspace(np.log10(Rmin_m), np.log10(Rmax_m), N)
    R_nm = R_m*1e9
    V = pot.pot(R_nm)*k_B # [J]
    Vprime = pot.potprime(R_nm)*k_B*1e9 # [J/m]
    Vprime2 = pot.potprime2(R_nm)*k_B*1e9**2 # [J/m^2]
    Vprime3 = pot.potprime3(R_nm)*k_B*1e9**3 # [J/m^3]
    potvals = {
        'R / m': R_m,
        'V': V,
        'Vprime': Vprime,
        'Vprime2': Vprime2,
        'Vprime3': Vprime3,
        'mass_rel': pot.mass_rel
    }
    return potvals

def get_integrand(T_K, *, potvals, quantum=3):
    # Constants
    m = potvals['mass_rel']*u # [kg/atom]

    beta = 1.0/(k_B*T_K) # [1/J]
    lambda_ = hbar**2*beta/(12*m) # [m]

    # These common terms have no temperature dependence
    # and are calculated once and passed into this function
    R_m = potvals['R / m']
    V = potvals['V']
    Vprime = potvals['Vprime']
    Vprime2 = potvals['Vprime2']
    Vprime3 = potvals['Vprime3']

    # Some common terms, calculated once for speed
    enbV = exp(-beta*V)
    if quantum == 0:
        return R_m, R_m**2*(-(enbV-1.0))  # classical
    else:
        betaVprime = beta*Vprime
        betaVprime2 = beta*Vprime2
        betaVprime3 = beta*Vprime3
        integrand = R_m**2*(
            -(enbV-1.0)  # classical
            +lambda_*enbV*(betaVprime)**2 # first quantum correction
            -lambda_**2*enbV*(
                6/5*(betaVprime2)**2 
               +12/(5*R_m**2)*(betaVprime)**2
               +4/(3*R_m)*(betaVprime)**3
               -1/6*(betaVprime)**4) # second quantum correction
            +lambda_**3*enbV*(
                36/35*(betaVprime3)**2
                +216/(35*R_m**2)*(betaVprime2)**2
                +24/21*(betaVprime2)**3
                +24/(5*R_m)*(betaVprime)*(betaVprime2)**2
                +288/(315*R_m**3)*(betaVprime)**3
                -6/5*(betaVprime)**2*(betaVprime2)**2
                -2/(15*R_m**2)*(betaVprime)**4
                -2/(5*R_m)*(betaVprime)**5
                +1/30*(betaVprime)**6
                ) # third quantum correction
            )
        return R_m, integrand

def B2(T_K, *, potvals, quantum):
    R_m, integrand = get_integrand(T_K, potvals=potvals, quantum=quantum)
    return 2.0*np.pi*N_A*scipy.integrate.trapz(x=R_m, y=integrand)

def calc_B2(fluid):

    if fluid.lower() == 'neon':
        pot = NeonTT
    elif fluid.lower() == 'argon':
        pot = ArgonTT
    elif fluid.lower() == 'krypton':
        pot = KryptonTT
    elif fluid.lower() == 'xenon':
        pot = XenonTT
    else:
        raise ValueError('bad fluid'+fluid)

    # Things with only R dependence, no T dependence
    Rmin = 0.01*pot.Repsilon*1e-9
    Rmax = 10000*pot.Repsilon*1e-9
    potvals = get_potvals(Rmin, Rmax, 10**4, pot=pot)

    def ff(T,**kwargs):
        return B2(T, potvals = potvals, **kwargs)

    o = []
    Tvec = np.geomspace(4, 1e4, 100)
    Tvec = np.array([4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,14.00,15.00,16.00,17.00,18.00,19.00,20.00,21.00,22.00,23.00,24.00,25.00,26.00,27.00,28.00,30.00,32.00,34.00,36.00,38.00,40.00,42.00,44.00,46.00,48.00,50.00,55.00,60.00,65.00,70.00,75.00,80.00,85.00,90.00,95.00,100.00,110.00,120.00,130.00,140.00,150.00,160.00,170.00,180.00,190.00,200.00,210.00,220.00,230.00,240.00,250.00,260.00,270.00,273.15,273.16,280.00,290.00,293.15,298.15,300.00,320.00,340.00,360.00,380.00,400.00,420.00,440.00,460.00,480.00,500.00,550.00,600.00,650.00,700.00,750.00,800.00,850.00,900.00,950.00,1000.00,1100.00,1200.00,1300.00,1400.00,1500.00,1600.00,1700.00,1800.00,1900.00,2000.00,2100.00,2200.00,2300.00,2400.00,2500.00,2600.00,2700.00,2800.00,2900.00,3000.00,3200.00,3400.00,3600.00,3800.00,4000.00,4200.00,4400.00,4600.00,4800.00,5000.00])
    for T in Tvec:
        res = {'T / K': T}
        m = potvals['mass_rel']*u # [kg/atom]
        res['lambda_th / m'] = hPlanck/(2*np.pi*m*k_B*T)**0.5 # thermal de Broglie wavelength

        for quantum in [3, 0]:
            B = ff(T, quantum=quantum)
            ders = pymcx.diff_mcx1(lambda T_: ff(T_, quantum=quantum), T, 2)
            TdBdT = ders[0]*T
            T2d2BdT2 = ders[1]*T**2

            gamma0 = 5/3
            betaa = 2*(B + (gamma0-1)*TdBdT + (gamma0-1)**2/(2*gamma0)*T2d2BdT2)
            neff = -3*(B+TdBdT)/(2*TdBdT+T2d2BdT2)

            suffix = '' if quantum == 3 else ' (classical)'
        
            res['B_2 (m^3/mol)'+suffix] = B
            res['B_2 (cm^3/mol)'+suffix] = B*1e6
            res['TdB_2dT (m^3/mol)'+suffix] = TdBdT
            res['T2d2B_2dT2 (m^3/mol)'+suffix] = T2d2BdT2
            res['neff'+suffix] = neff
            res['betaa (m^3/mol)'+suffix] = betaa
            res['Theta_2 (m^3/mol)' + suffix] = B + TdBdT

        o.append(res)
        print(o[-1])

    df = pandas.DataFrame(o)
    df.to_csv(fluid+'_abinitio.csv', index=False)

if __name__ == '__main__':
    calc_B2('Neon')
    calc_B2('Argon')
    # calc_B2('Krypton')
    # calc_B2('Xenon')