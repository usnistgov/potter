"""
Some analytic solutions for Mie n-m potentials
"""
import numpy as np
from scipy.special import gamma as GammaFunc

def get_Bstar_Sadus(Tstar, *, n, m):
    """
    The second virial coefficient B^* = B_2
    # Sadus, https://doi.org/10.1063/1.5041320, erratum: missing exponent of m
    n is the repulsive exponent (the 12 of 12-6 LJ)
    m is the attractive exponent (the 6 of 12-6 LJ)
    """
    def F(y):
        the_sum = 0
        for i in range(1, 200):
            def my_factorial(k):
                return GammaFunc(k+1)
            c = GammaFunc((i*m-3.0)/n)/my_factorial(i)
            the_sum += c*y**i
        return y**(3/(n-m))*(GammaFunc((n-3.0)/n) -3/n*the_sum)
    yn = (n/(n-m))**n*((n-m)/m)**m*Tstar**(-(n-m)) # y**n, Eq. 9
    y = yn**(1/n)
    return 2*np.pi/3*F(y)

def get_dBstardTstar_Sadus(Tstar, *, n, m):
    """
    # Sadus, https://doi.org/10.1063/1.5041320, erratum: missing exponent of m
    m is the attractive exponent (the 6 of 12-6 LJ)
    n is the repulsive exponent (the 12 of 12-6 LJ)
    """
    def dFdy(y):
        the_sum, the_derivsum = 0, 0
        for i in range(1, 200):
            def my_factorial(k):
                return GammaFunc(k+1)
            c = GammaFunc((i*m-3.0)/n)/my_factorial(i)
            the_sum += c*y**i
            the_derivsum += i*c*y**(i-1)
        return y**(3/(n-m))*(-3/n*the_derivsum) + (3/(n-m))*y**(3/(n-m)-1)*(GammaFunc((n-3.0)/n) -3/n*the_sum)
    y = (n/(n-m))*((n-m)/m)**(m/n)*Tstar**(-(n-m)/n) 
    dydT = (n/(n-m))*((n-m)/m)**(m/n)*(-(n-m)/n)*Tstar**(-(n-m)/n-1) 
    return 2*np.pi/3*dFdy(y)*dydT

def get_d2BstardTstar2_Sadus(Tstar, *, n, m):
    """
    # Sadus, https://doi.org/10.1063/1.5041320, erratum: missing exponent of m
    m is the attractive exponent (the 6 of 12-6 LJ)
    n is the repulsive exponent (the 12 of 12-6 LJ)
    """
    def dFdy(y):
        the_sum, the_derivsum = 0, 0
        for i in range(1, 200):
            def my_factorial(k):
                return GammaFunc(k+1)
            c = GammaFunc((i*m-3.0)/n)/my_factorial(i)
            the_sum += c*y**i
            the_derivsum += i*c*y**(i-1)
        return y**(3/(n-m))*(-3/n*the_derivsum) + (3/(n-m))*y**(3/(n-m)-1)*(GammaFunc((n-3.0)/n) -3/n*the_sum)
    def d2Fdy2(y):
        the_sum, the_derivsum, the_2derivsum = 0, 0, 0
        for i in range(1, 200):
            def my_factorial(k):
                return GammaFunc(k+1)
            c = GammaFunc((i*m-3.0)/n)/my_factorial(i)
            the_sum += c*y**i
            the_derivsum += i*c*y**(i-1)
            the_2derivsum += i*(i-1)*c*y**(i-2)
        return (y**(3/(n-m))*(-3/n*the_2derivsum) 
                + (3/(n-m))*y**(3/(n-m)-1)*(-3/n*the_derivsum) 
                + (3/(n-m))*y**(3/(n-m)-1)*(-3/n*the_derivsum)
                + (3/(n-m))*(3/(n-m)-1)*y**(3/(n-m)-2)*(GammaFunc((n-3.0)/n) -3/n*the_sum)
                )

    y = (n/(n-m))*((n-m)/m)**(m/n)*Tstar**(-(n-m)/n) 
    dydT = (n/(n-m))*((n-m)/m)**(m/n)*(-(n-m)/n)*Tstar**(-(n-m)/n-1) 
    d2ydT2 = (n/(n-m))*((n-m)/m)**(m/n)*(-(n-m)/n)*(-(n-m)/n-1) *Tstar**(-(n-m)/n-2) 
    return 2*np.pi/3*(d2Fdy2(y)*dydT**2 + dFdy(y)*d2ydT2)

def get_neff_Sadus(Tstar, *, n, m):
    """ effective hardness from B_2 """
    a = dict(n=n,m=m) # Common keyword args
    B = get_Bstar_Sadus(Tstar,**a)
    dBdT = get_dBstardTstar_Sadus(Tstar,**a)
    d2BdT2 = get_d2BstardTstar2_Sadus(Tstar,**a)
    neff = -3*(B + Tstar*dBdT)/(2*Tstar*dBdT + Tstar**2*d2BdT2)
    return neff

if __name__ == '__main__':
    import scipy.optimize
    f = lambda T: 1/get_neff_Sadus(T, n=12, m=6)
    res = scipy.optimize.minimize(f, 10)
    print(1/f(res.x))