# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 09:51:17 2021

@author: ysloo
"""

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
wobs  = 2*pi*100 * MHz


#%% Faster integration scheme

from scipy.special import kv as modBessel2nd
from numpy.polynomial import Polynomial

def make_logspace_x(xmin, xmax, steps, xscale=1e8):
    x      = np.empty(steps+1)
    x[:-1] = (xmax-xmin)*(10**np.linspace(-np.log10(xscale),0,steps)-1./xscale) + xmin
    x[-1]  = xmax
    return x

def integralK(xmin, xmax, steps=1000):
    x  = make_logspace_x(xmin, xmax, steps)
    dx = np.ediff1d(x)
    Kv = modBessel2nd(5./3,x)
    #dF = Kv[:-1]*dx
    dF = (Kv[:-1]+Kv[1:])/2 * dx
    #plt.plot(dF)
    #plt.show()
    return np.sum(dF)

def integralF(xgrid):
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
        
    #xbessel = 10**np.linspace(-8,2,int(1e4))
    #ybessel = integralF(xbessel)
    #polyfit_integral = Polynomial.fit(np.log10(xbessel), np.log10(ybessel), 30)
    #print(polyfit_integral.coef)
    #print(polyfit_integral.domain)
    
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

polyfitF = set_polyfitF()

#%% Physical functions

from scipy.special import gamma as gammafunc

def critical_freq(gamma):
    return (3*electron*Bper/(2*me*c) * (gamma)**2).decompose(bases=u.cgs.bases)

def critical_gamma(wc):
    return np.sqrt(wc * (2*me*c)/(3*electron*Bper)).decompose()

def set_cre_spectrum(n0= None, alpha=None, gammas=None, normalize=True):
    if normalize:
        gmin, gmax = np.min(gammas), np.max(gammas)
        Z          = 1./(1+alpha) * (gmax**(1+alpha)-gmin**(1+alpha))
        def spectrum(gamma):
            return n0/Z * gamma**alpha
        return spectrum
    else:
        def spectrum(gamma):
            return n0 * gamma**alpha
        return spectrum

def spectral_integralF(mu):
    return 2**(mu+1)/(mu+2) * gammafunc(mu/2 + 7./3)*gammafunc(mu/2+2./3)

def spectral_total_emissivity(vobs, alpha):
    fraction1 = (np.sqrt(3)*electron**3*Bper*n0/(8*pi*me*c**2)).decompose(bases=u.cgs.bases)
    fraction2 = (4*pi*vobs*me*c/(3*electron*Bper)).decompose(bases=u.cgs.bases)
    integral  = spectral_integralF((-alpha-3)/2)
    return fraction1 * fraction2**((1+alpha)/2) * integral

def emission_power_density(x):
    #F = integralF(x)
    F = polyfitF(x)
    return ((np.sqrt(3)*electron**3*Bper)/(2*pi*me*c**2) * F).decompose(bases=u.cgs.bases)

def integrate_total_emissivity(vobs, spectrum, gammas, returnsteps=False):
    wobs = 2*pi*vobs
    x    = wobs/critical_freq(gammas)
    
    ncre = spectrum(gammas).decompose(bases=u.cgs.bases)
    dPw  = emission_power_density(x).decompose(bases=u.cgs.bases)
    dg   = -np.ediff1d(gammas)
    
    # implement midpoint
    dJleft  = 0.5*np.sum((ncre[:-1]*dPw[:-1]*dg).decompose(bases=u.cgs.bases))
    dJright = 0.5*np.sum((ncre[1:]*dPw[1:]*dg).decompose(bases=u.cgs.bases))
    return (dJleft+dJright)/2

   


#%% Testfunctions

"""
First task is to confirm that the faster integration method works well. We will
check this by doing a comparison to the spectral emissivity function using the
two full integration steps.
"""

def integrated_power_profile(resolution, plotting=True):
    xgrid     = make_logspace_x(xmin=1e-8, xmax=1e2, steps=resolution)
    gammagrid = critical_gamma(wobs/xgrid)
    dPw       = emission_power_density(x=xgrid)
    Pw        = -dPw[:-1]*np.ediff1d(gammagrid)
    spectrum = set_cre_spectrum(n0=n0, alpha=-4, gammas=gammagrid, normalize=False)
    Ncre     = spectrum(gammagrid[:-1])
    dJ       = Ncre * Pw
    
    if plotting:
        plt.plot(gammagrid, dPw, '.')
        plt.xscale('log')
        plt.show()
        
        plt.plot(gammagrid[:-1], Pw, '.')
        plt.xscale('log')
        plt.show()
        
        plt.plot(gammagrid[:-1], dJ, '.')
        plt.title("dJ(gamma) = Ncre(gamma)*dPw(gamma)*dgamma")
        plt.xscale('log')
        plt.show()
        
        print(0.5*np.sum(dJ).decompose(bases=u.cgs.bases))
    else:
        return gammagrid[:-1], dJ

def do_integration_resolution_search():
    reslist = [100,500,1000,5000,10000]
    for res in reslist:
        g, dJ = integrated_power_profile(res, plotting=False)
        plt.plot(g, dJ, label='{} integration steps'.format(res))

    plt.title("Emissivity profile dJi for different integration resolutions")
    plt.xlabel("gamma at index i")
    plt.xscale('log')
    plt.legend()
    plt.show()

def compare_spectral_vs_integral():
    #energygrid = 10**np.linspace(6,11,int(1e3))*eV
    #gammagrid  = (energygrid/(me*c**2)).decompose(bases=u.cgs.bases)
    
    xgrid     = make_logspace_x(xmin=1e-8, xmax=1e2, steps=5000)
    gammagrid = critical_gamma(wobs/xgrid)
    
    Jint = []
    Jspe = []
    steps = 10
    spectralindex = np.linspace(-4,-2,steps)
    for a in spectralindex:
        cre = set_cre_spectrum(n0=n0, alpha=a, gammas=gammagrid, normalize=False)
        Ji  = integrate_total_emissivity(vobs=100*MHz, spectrum = cre, gammas=gammagrid, returnsteps=False)
        Js  = spectral_total_emissivity(vobs=100*MHz, alpha=a)
        Jint.append(Ji/Ji.unit)
        Jspe.append(Js/Js.unit)
    Jint, Jspe = np.array(Jint)*Ji.unit, np.array(Jspe)*Js.unit
    
    plt.plot(spectralindex, Jspe, label = 'analytical')
    plt.plot(spectralindex, Jint, label = 'integrated')
    plt.legend()
    plt.yscale('log')
    plt.show()
    
    plt.plot(spectralindex, Jspe/Jint)
    plt.plot(spectralindex, np.ones(steps))
    plt.title("Ratio Jspectral/Jintegral")
    plt.xlabel("CRe spectral index alpha")
    #plt.ylim([0.9,1.1])
    plt.show()














