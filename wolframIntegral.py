import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma
from mpmath import csc
from mpmath import hyp1f2
#from mpmath import mpf
#import mpmath as mp

def wolfram_definite_integral(z, nu):
    fraction1 = np.float64(2**( nu-1) * np.pi * z**(1-nu) * csc(np.pi*nu) / gamma(2-nu))
    fraction2 = np.float64(2**(-nu-1) * np.pi * z**(1+nu) * csc(np.pi*nu) / gamma(2+nu))
    F1 = np.float64(hyp1f2((1-nu)/2, 1-nu    , (3-nu)/2, z**2/4))
    F2 = np.float64(hyp1f2((nu+1)/2, (nu+3)/2, nu+1    , z**2/4))
    #print(fraction1, F1, fraction2, F2)
    return fraction1 * F1 - fraction2 * F2

testxbessel = 10**np.linspace(-8,0,50)

definiteIntegral = []
for i,xv in enumerate(testxbessel):
    definiteIntegral.append(wolfram_definite_integral(xv, 5./3))
    #print("For x={}, the integral gives I={}".format(xv, wolfram_definite_integral(xv, 5./3)))

plt.plot(testxbessel, definiteIntegral)
plt.xscale('log')
plt.show()

## Investigate parts of definite integral

def fraction1(z,nu):
    return np.float64(2**( nu-1) * np.pi * z**(1-nu) * csc(np.pi*nu) / gamma(2-nu))

def fraction2(z,nu):
    return np.float64(2**(-nu-1) * np.pi * z**(1+nu) * csc(np.pi*nu) / gamma(2+nu))

def F1(z,nu):
    values = []
    for _z in z:
        values.append(np.float64(hyp1f2((1-nu)/2, 1-nu, (3-nu)/2, _z**2/4)))
    return np.array(values)

def F2(z,nu):
    values = []
    for _z in z:
        values.append(np.float64(hyp1f2((nu+1)/2, (nu+3)/2, nu+1, _z**2/4)))
    return np.array(values)

funclist = [fraction1, fraction2, F1, F2]

xtest = 10**np.linspace(-8,1, 100)
nu = 5./3
for f in funclist:
    ytest = f(xtest, nu)
    negpos = 'positive'
    if np.all(ytest <= 0):
        negpos = 'negative'
        ytest *= -1

    plt.plot(xtest,ytest, '.')
    plt.title(negpos+' of '+f.func_name)
    plt.ylabel("integral part value")
    plt.xlabel("x")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

##

from astropy import units as u
from astropy import constants as cons

# units definitions since astropy doesnt handle [B]_cgs not well
gauss_B = (u.g/u.cm)**(0.5)/u.s
equiv_B = [(u.G, gauss_B, lambda x: x, lambda x: x)]

# physical constants and typical values
me    = cons.m_e.cgs
c     = cons.c.cgs
electron = (cons.e).gauss
Bper  = (4e-6 * gauss_B)
pi    = np.pi
MHz   = 1e6 / u.s
MeV   = 1e6 * u.eV.cgs


def critical_energy(wc):
    return np.sqrt(wc * (2*me**3 *c**5)/(3*electron*Bper)).decompose()

wobs   = 100 * MHz
energy = (critical_energy(wobs/xtest)).to(u.eV)

for f in funclist:
    ytest = f(xtest, nu)
    negpos = 'positive'
    if np.all(ytest <= 0):
        negpos = 'negative'
        ytest *= -1

    plt.plot(energy,ytest, '.')
    plt.title(negpos+' of '+f.func_name)
    plt.ylabel("integral part value")
    plt.xlabel("CRE energy (eV)")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()




