# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:09:44 2022

@author: ysloots
"""

# Main imports and settings
import matplotlib.pyplot as plt
import numpy as np
path = '/home/ysloots/masterproject/project/figures/'

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
wobs  = 2*pi*100 * MHz


#%% Numerical functions

from scipy.special import kv as modBessel2nd
from numpy.polynomial import Polynomial

def make_logspace_x(xmin, xmax, steps, xscale=1e6):
    x      = np.empty(steps+1)
    x[:-1] = (xmax-xmin)*(10**np.linspace(-np.log10(xscale),0,steps)-1./xscale) + xmin
    x[-1]  = xmax
    return x

def integralK(xmin, xmax, steps=1000):
    x  = make_logspace_x(xmin, xmax, steps, xscale = 1e6)
    dx = np.ediff1d(x)
    Kv = modBessel2nd(5./3,x)
    dF = Kv[:-1]*dx
    #dF = (Kv[:-1]+Kv[1:])/2 * dx
    #plt.plot(dF)
    #plt.show()
    return np.sum(dF)

def integralF(xgrid):
    """Expects xgrid to be ordered small to large"""
    #print("Evaluating F(x)=xI(x) for x={} up to x={}".format(xgrid.min(),xgrid.max()))
    # subdivide Kv(x) in rectangle areas
    sxgrid = xgrid[1:]
    dI     = []
    for xmin,xmax in zip(xgrid[:-1], sxgrid):
        #print(xmin,xmax)
        dI.append(integralK(xmin, xmax))
    dI = np.array(dI)
    # sum rectangles after x from xgrid
    dF = []
    for i in range(len(dI)):
        dF.append(xgrid[i]*np.sum(dI[i:]))
    dF.append(dF[-1]) # dirty temporary fix to make the array of the same size again
    return np.array(dF)

def set_polyfitF():
    coef = [-6.70416283e-01,  1.61018924e+00, -5.63219709e-01,  4.04961264e+00,
            3.65149319e+01, -1.98431222e+02, -1.19193545e+03,  4.17965143e+03,
            1.99241293e+04, -4.98977562e+04, -1.99974183e+05,  3.72435079e+05,
            1.30455499e+06, -1.85036218e+06, -5.81373137e+06,  6.35262141e+06,
            1.82406280e+07, -1.53885501e+07, -4.09471882e+07,  2.64933708e+07,
            6.60352613e+07, -3.22140638e+07, -7.58569209e+07,  2.70300472e+07,
            6.05473150e+07, -1.48865555e+07, -3.19002452e+07,  4.84145538e+06,
            9.97173698e+06, -7.04507542e+05, -1.40022630e+06]
    domain = [-8.,  2.]
    polyfit_integral = Polynomial(coef, domain)
    
    # Definition of F(x)
    def F(x):
        return 10**polyfit_integral(np.log10(x))
    return F

# Already globaly define polyfitF here for the rest of the script!
polyfitF = set_polyfitF()


#%% Physical functions

def critical_freq(gamma):
    return (3*electron*Bper/(2*me*c) * (gamma)**2).decompose(bases=u.cgs.bases)

def emission_power_density(gamma, wobs, Bper):
    wc = critical_freq(gamma)
    x  = np.flip(wobs/wc)
    F  = np.flip(integralF(x))
    #F  = np.flip(polyfitF(x))
    return ((np.sqrt(3)*electron**3*Bper)/(2*pi*me*c**2) * F).decompose(bases=u.cgs.bases)


#%% Making the plot

# typical observing frequencies and cre energies
dPunit = u.kg*u.m**2/u.s**2
wobs       = 10**np.linspace(0,4,100) * MHz
energygrid = 10**np.linspace(6,11,int(1e2))*eV
gammagrid  = (energygrid/(me*c**2)).decompose(bases=u.cgs.bases)

def calc_power_grid():
    print("Calculating Powergrid")
    powergrid = []
    for wo in wobs:
        powergrid.append(emission_power_density(gammagrid, wo, Bper)/dPunit)
    return powergrid

powergrid = calc_power_grid()
print(np.shape(powergrid))

plt.contourf(energygrid, wobs/(2*pi), powergrid, 30, cmap='Blues')
plt.colorbar()

plt.title("Synchrotron emission power density")
plt.ylabel("Observed frequency (Hz)")
plt.xlabel("CR-electron energy (eV)")
plt.yscale('log')
plt.xscale('log')

plt.savefig(path + "emissionpowerdensity_Integrated.png")
plt.close('all')


















