# Main imports and settings
import matplotlib.pyplot as plt
import numpy as np
path = '/home/ysloots/MasterProject/project/figures/'

# Define units and constants
from astropy import units as u
from astropy import constants as cons

# Units definitions since astropy doesnt handle [B]_cgs not well
gauss_B = (u.g/u.cm)**(0.5)/u.s
equiv_B = [(u.G, gauss_B, lambda x: x, lambda x: x)]

# Physical constants and typical values
me    = cons.m_e.cgs
c     = cons.c.cgs
electron = (cons.e).gauss
Bper  = (4e-6 * gauss_B)
pi    = np.pi
MHz   = 1e6 / u.s
eV    = 1.  * u.eV.cgs
MeV   = 1e6 * eV
n0    = 1.  * u.cm**-3

##

from scipy.special import gamma as gammafunc
from numpy.polynomial import Polynomial

def set_polyfitF():
    # Do routine to integrate the besselfunction Kv(x) on startup once
    """ routine here"""
    
    # Do polynomial fit to integration result, using log10y log10x
    domain = [-8., 2.]
    coef   = [-0.67265073,    1.65272252,   -0.12563691,   -0.48138695,
              -0.45289293,    1.08823299,   -1.3740308 ,  -13.9640722 ,
              -6.15950103,   40.96745024,   20.18361224,  -98.78800032,
              -72.82745829,  108.33164766,   87.68206551,  -83.67735104,
              -69.51213597,   31.83594659,   25.89082237,   -6.95991161,
              -4.96558943]
    polyfit_integral = Polynomial(coef, domain)
    
    # Definition of F(x)
    def F(x):
        return x * 10**polyfit_integral(np.log10(x))
    return F

def critical_freq(gamma):
    return (3*electron*Bper/(2*me*c) * (gamma)**2).decompose(bases=u.cgs.bases)

def emission_power_density(gamma, wobs):
    wc   = critical_freq(gamma)
    x    = wobs/wc
    return ((np.sqrt(3)*electron**3*Bper)/(2*pi*me*c**2) * polyfitF(x) ).decompose(bases=u.cgs.bases)

def integrate_total_emissivity(vobs, spectrum, gammas, returnsteps=False):
    wobs = 2*pi*vobs
    ncre = spectrum(gammas).decompose(bases=u.cgs.bases)
    dPw  = emission_power_density(gammas, wobs).decompose(bases=u.cgs.bases)
    dg   = np.ediff1d(gammas)
    if returnsteps:
        return (ncre[:-1]*dPw[:-1]).decompose(bases=u.cgs.bases), dg
    else:
        return 0.5*np.sum((ncre[:-1]*dPw[:-1]*dg).decompose(bases=u.cgs.bases))

## Set global functions

polyfitF = set_polyfitF()

##



def plot_polyfitF():
    wobs       = 2*pi*100 * MHz
    energygrid = 10**np.linspace(6,11,1000)*eV
    gammagrid  = (energygrid/(me*c**2)).decompose(bases=u.cgs.bases)
    x          = wobs/critical_freq(gammagrid)
    plt.plot(x, polyfitF(x), '.')
    #plt.xscale('log')
    plt.show()

    plt.plot(x[:-1], np.ediff1d(x), '.')
    plt.show()

























