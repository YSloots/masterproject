# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 10:36:17 2021
@author: ysloo

============================================
Discription of all functions in this script (except for the most important one)
============================================
'make_logspace_x'      : logarithicly spaced points with different density scales
'integrate_bessel'     : older method for integrating Kv(x) from x to oo
'integralF_old'        : integrate Kv(x) for a full array of xmin
'integralK'            : new method for integrating Kv(x) from xmin to xmax
'integralF'            : integrate Kv(x) for a full array of xmin, more cleverly
'do_integration_method_comparison': compare old with new integration method
'set_polyfitF'         : globaly defines a polynomial fit to integralF
'test_polyfitF'        : do a full integration and fit and compare with saved polyfitF 
'critical_freq'        : calculate wc(gamma)
'critical_gamma'       : calclate gamma(wc)
'cre_suppression'      : define powerlaw funtion with largest value 1
'test_fitted_powerprofile_dJ': calculate dJ for an array of gammas
'set_cre_spectrum'     : define function Ncre(g)=n0 * gamma^alpha
'spectral_integralF'   : part of analytical result for emissivity after assuming powerlaw Ncre
'spectral_total_emissivity': analytical result for emissivity after assuming powerlaw Ncre
'emission_power_density': powerdensity dPw(x) with x=x(gamma, wobs, Bper), contains definition of 
integralF(x) or spectralF(x)
'integrate_total_emissivity': integrate dJ(gamma) over a range of gammas
'do_single_emissivity': test 1 integration of total emissivity and compare with analytical result
this is a usefull function to investigate the effect of different gammagrids

===========================================================
The most imporant function: 'compare_spectral_vs_integral':
===========================================================
This function is the most important function. Here we compare the integration result Jint
with the analytical result Jspe for a range of powerlaw coefs alpha. We pick either the option to
do a full double step integration each time, or we can use the fit polyfitF(x) to speed up the
calculation. If Jspe/Jint = 1 for the full range of alpha we can be confident our integration method
and/or approximation with the fit are accurate.

==========================================
Results of compare_spectral_vs_integral():
==========================================
Using polyfitF() in the defintion of emission_power_density() we get
a fraction Jspe/Jint between [0.9940, 0.995]. Meaning we could correct
with a single factor and obtain a accuracy of at worst about 0.1%

Using integralF() takes a longer time and results in a fraction of
Jspe/Jint between [0.9935, 0.995].

The results are and look slightly different meaning that the fit does
not perfectly cover the integration result. However the order of accuracy
is similar. One thing to note is that for values of alpha closer to zero
the result Jspe/Jint seems to be exponentially increasing. Why?
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


#%% Basic definitions

def make_logspace_x(xmin, xmax, steps, xscale=1e6):
    x      = np.empty(steps+1)
    x[:-1] = (xmax-xmin)*(10**np.linspace(-np.log10(xscale),0,steps)-1./xscale) + xmin
    x[-1]  = xmax
    return x


#%% Test and develop the integration

from scipy.special import kv as modBessel2nd



def integrate_bessel(xmin=None,xmax=None, steps=10000):
    """
    Implementation of a logarithmic variety of the midpoint rule for integration
    of the modified bessel function of the 2nd kind with real order 5/3.
    As a rule of thumb: O(xscale) ~ 2 |O(xmin)|
    """
    # the function decays exponentially, could further implement stopping criteria
    if xmax is None:
        xmax = 1e3 * xmin
    # compacted points along the x-axis (high resolution close to the origin)
    x = make_logspace_x(xmin, xmax, steps)
    y = modBessel2nd(5./3,x)
    # apply midpoint rule
    df = []
    for i in range(steps):
        dx = x[i+1] - x[i]
        df.append(dx*(y[i]+y[i+1])/2)
    return np.sum(df)

def integralF_old(xbessel):
    ybessel = []
    for xv in xbessel:
        ybessel.append(xv*integrate_bessel(xmin = xv))
    return np.array(ybessel)

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


#%% Comparison with old

def do_integration_method_comparison():
    xbessel  = 10**np.linspace(-8,2,1000)
    ybessel1 = integralF_old(xbessel)
    ybessel2 = integralF(xbessel)
    
    plt.plot(xbessel, ybessel1, 'o', label='v1')
    plt.plot(xbessel, ybessel2, label='v2')
    plt.title("Comparing integration methods")
    plt.xscale('log')
    plt.legend()
    plt.show()
    
    plt.plot(xbessel, ybessel2-ybessel1)
    plt.title("Difference methods: new - old")
    plt.xscale('log')
    plt.show()


#%% Better polynomial fit

from numpy.polynomial import Polynomial


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

def test_polyfitF():
    xbessel    = 10**np.linspace(-8,2,int(1e4))
    ybessel    = integralF(xbessel)
    logxbessel = np.log10(xbessel)
    logybessel = np.log10(ybessel)
    deg   = 30
    p     = Polynomial.fit(logxbessel, logybessel, deg)
    yfit  = 10**p(logxbessel)
    plt.plot(xbessel, ybessel,'.',label='integrated')
    plt.plot(xbessel, yfit, label='polyfit')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()

    plt.plot(xbessel, ybessel,'.',label='integrated')
    plt.plot(xbessel, polyfitF(xbessel), label='polyfit saved')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()


#%% What about the supression and amplification of the fit error during the integration?

def critical_freq(gamma):
    return (3*electron*Bper/(2*me*c) * (gamma)**2).decompose(bases=u.cgs.bases)

def critical_gamma(wc):
    return np.sqrt(wc * (2*me*c)/(3*electron*Bper)).decompose()
    
def cre_suppression(gammas, alpha):
    supression = gammas**alpha
    supression /= np.max(supression)
    return supression

#%%

def test_fitted_powerprofile_dJ():
    
    xbessel    = 10**np.linspace(-8,2,int(1e3))
    ybessel    = integralF(xbessel)
    logxbessel = np.log10(xbessel)
    logybessel = np.log10(ybessel)
    gbessel    = critical_gamma(wobs/xbessel)
    supression = cre_suppression(gbessel, alpha=-3)
    dgamma     = -np.ediff1d(gbessel)
    
    #make the same lenght: just ditch the last point
    gbessel    = gbessel[:-1]
    supression = supression[:-1]
    power      = ybessel[:-1]
    
    fig,(ax1,ax2)=plt.subplots(1,2, figsize=(12,6))
    
    # look for the effect of degree on the quality of fit
    degrees = [20,30,40]
    for deg in degrees:
        p        = Polynomial.fit(logxbessel, logybessel, deg)
        fitpower = 10**p(logxbessel)[:-1]
        dJfit    = supression * fitpower * dgamma
        error    = (ybessel[1:]-fitpower)
        dJerror  = supression * error * dgamma
        ax1.plot(gbessel, dJfit  , '.', label='degree {}'.format(deg))
        ax2.plot(gbessel, dJerror, '.', label='degree {}'.format(deg))
    
    # show what the fit should approach
    dJintegral = supression * power * dgamma
    ax1.plot(gbessel, dJintegral, lw = 2, label='integration')

    # dressing up the plots
    fig.suptitle("Relative emissivity profile dJ*dgamma (error suppressed with powerlaw -3)")
    ax1.legend()
    ax2.legend()
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    plt.show()

    plt.plot(gbessel, dJerror*dgamma)
    plt.title("Error of fitted power dPw")
    plt.xscale('log')
    plt.show()


#%% Do a full comparison between integrated and spectral emissivity

from scipy.special import gamma as gammafunc

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

def emission_power_density(gamma, wobs, Bper):
    wc = critical_freq(gamma)
    x  = np.flip(wobs/wc)
    #F  = np.flip(integralF(x))
    F  = np.flip(polyfitF(x))
    return ((np.sqrt(3)*electron**3*Bper)/(2*pi*me*c**2) * F).decompose(bases=u.cgs.bases)

def integrate_total_emissivity(vobs, spectrum, gammas):
    wobs = 2*pi*vobs
    Bper = (4e-6 * gauss_B)
    ncre = spectrum(gammas).decompose(bases=u.cgs.bases)
    dPw  = emission_power_density(gammas, wobs, Bper).decompose(bases=u.cgs.bases)
    dg   = np.ediff1d(gammas)
    
    # implement midpoint
    dJleft  = 0.5*np.sum((ncre[:-1]*dPw[:-1]*dg).decompose(bases=u.cgs.bases))
    dJright = 0.5*np.sum((ncre[1:]*dPw[1:]*dg).decompose(bases=u.cgs.bases))
    return (dJleft+dJright)/2

def do_single_emissivity():
    energygrid = 10**np.linspace(7,13,int(1e4))*eV
    gammagrid  = (energygrid/(me*c**2)).decompose(bases=u.cgs.bases)
    
    a   = -3
    cre = set_cre_spectrum(n0=n0, alpha=a, gammas=gammagrid, normalize=False)
    Ji  = integrate_total_emissivity(vobs=100*MHz, spectrum=cre, gammas=gammagrid)
    Js  = spectral_total_emissivity(vobs=100*MHz, alpha=a)
    print(Ji, Js)

def compare_spectral_vs_integral():
    energygrid = 10**np.linspace(6,11,int(1e3))*eV
    gammagrid  = (energygrid/(me*c**2)).decompose(bases=u.cgs.bases)
    
    Jint = []
    Jspe = []
    spectralindex = np.linspace(-4,-2,10)
    for a in spectralindex:
        cre = set_cre_spectrum(n0=n0, alpha=a, gammas=gammagrid, normalize=False)
        Ji  = integrate_total_emissivity(vobs=100*MHz, spectrum = cre, gammas=gammagrid)
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
    plt.title("Ratio Jspectral/Jintegral")
    plt.xlabel("CRe spectral index alpha")
    plt.show()








