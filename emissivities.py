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

#%%# Definition and imports of functions
from scipy.special import gamma as gammafunc
from numpy.polynomial import Polynomial

def integrate_midpoint(x,y):
    # Remove units
    xu = x.unit
    yu = y.unit
    _x = x/xu
    _y = y/yu
    print(xu, yu)
    # Do the integration
    df = []
    for i in range(len(x)-1):
        dx = _x[i+1] - _x[i]
        df.append(dx*(_y[i]+_y[i+1])/2)
    df = np.array(df)
    # Return sum and restore units
    I = (np.sum(df)*xu*yu).decompose(bases=u.cgs.bases)
    return I

def spectral_integralF(mu):
    return 2**(mu+1)/(mu+2) * gammafunc(mu/2 + 7./3)*gammafunc(mu/2+2./3)

def spectral_total_emissivity(vobs, alpha):
    fraction1 = (np.sqrt(3)*electron**3*Bper*n0/(8*pi*me*c**2)).decompose(bases=u.cgs.bases)
    fraction2 = (4*pi*vobs*me*c/(3*electron*Bper)).decompose(bases=u.cgs.bases)
    #fraction2 = (2*pi*vobs*me*c/(3*electron*Bper)).decompose(bases=u.cgs.bases)
    integral  = spectral_integralF((-alpha-3)/2)
    return fraction1 * fraction2**((1+alpha)/2) * integral

def test_spectral_total_emissivity():
    spectralindex = np.linspace(-4,-2,10)
    Jspectral     = []
    for a in spectralindex:
        J = spectral_total_emissivity(vobs = 100*MHz, alpha=a)
        Jspectral.append(J/J.unit)
    plt.plot(spectralindex, Jspectral)
    plt.yscale('log')
    plt.show()
    
    specintF = spectral_integralF((-spectralindex-3)/2)
    plt.plot(spectralindex, specintF)
    plt.show()
    
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
        return 10**polyfit_integral(np.log10(x))
    return F

def critical_freq(gamma):
    return (3*electron*Bper/(2*me*c) * (gamma)**2).decompose(bases=u.cgs.bases)

def emission_power_density(gamma, wobs):
    wc   = critical_freq(gamma)
    x    = wobs/wc
    return ((np.sqrt(3)*electron**3*Bper)/(2*pi*me*c**2) * polyfitF(x) ).decompose(bases=u.cgs.bases)

def test_emission_power_density():
    wobs = 100 * MHz
    Ecre = 10**np.linspace(7,10,1000) * u.eV
    dP   = (emission_power_density(Ecre, wobs)).decompose(bases=u.cgs.bases)
    plt.plot(Ecre, dP, '.')
    plt.title("Synchrotron emission power density at wobs = {:.2e}".format(wobs))
    plt.ylabel("Power density Jv ({})".format(dP.unit))
    plt.xlabel("CR-energy ({})".format(Ecre.unit))
    plt.xscale('log')
    plt.show()
    return

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

def test_cre_spectrum_normalization():
    energygrid = 10**np.linspace(6,11,1000)*eV
    gammagrid  = energygrid/(me*c**2)
    for a in np.linspace(-4,-2, 5):
        spectrum = set_cre_spectrum(n0=n0, alpha=a, gammas=gammagrid, normalize=True)
        density  = (spectrum(gammagrid)).decompose(bases=u.cgs.bases)
        plt.plot(energygrid/energygrid.unit, density/density.unit, label = "alpha = {}".format(a))
    plt.title("Normalized cre numberdensity spectra")
    plt.ylabel("Number density ({})".format(density.unit))
    plt.xlabel("CR-electron energy ({})".format(energygrid.unit))
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()
    return

def integrate_total_emissivity(vobs, spectrum, gammas, returnsteps=False):
    wobs = 2*pi*vobs
    ncre = spectrum(gammas).decompose(bases=u.cgs.bases)
    dPw  = emission_power_density(gammas, wobs).decompose(bases=u.cgs.bases)
    dg   = np.ediff1d(gammas)
    if returnsteps:
        return (ncre[:-1]*dPw[:-1]).decompose(bases=u.cgs.bases), dg
    else:
        return 0.5*np.sum((ncre[:-1]*dPw[:-1]*dg).decompose(bases=u.cgs.bases))

def test_integrate_total_emissivity():
    energygrid = 10**np.linspace(8,11,1000) * eV
    gammagrid  = (energygrid/(me*c**2)).decompose(bases=u.cgs.bases)
    cre        = set_cre_spectrum(n0=n0, alpha=-4, gammas=gammagrid, normalize=False)
    dJ, dg     = integrate_total_emissivity(vobs=100*MHz, spectrum = cre, gammas=gammagrid, returnsteps=True)
    plt.plot(gammagrid[:-1], dJ, '.')
    #plt.yscale('log')
    #plt.xscale('log')
    plt.show()
    return

def compare_spectral_vs_integral():
    energygrid = 10**np.linspace(8,11,int(1e6))*eV
    gammagrid  = (energygrid/(me*c**2)).decompose(bases=u.cgs.bases)
    
    Jint = []
    Jspe = []
    spectralindex = np.linspace(-4,-2,10)
    for a in spectralindex:
        cre = set_cre_spectrum(n0=n0, alpha=a, gammas=gammagrid, normalize=False)
        Ji  = integrate_total_emissivity(vobs=100*MHz, spectrum = cre, gammas=gammagrid, returnsteps=False)
        Js  = spectral_total_emissivity(vobs=100*MHz, alpha=a)
        Jint.append(Ji/Ji.unit)
        Jspe.append(Js/Js.unit)
    Jint, Jspe = np.array(Jint)*Ji.unit, np.array(Jspe)*Js.unit
    
    #print(Jint, Jspe)
    plt.plot(spectralindex, Jspe, label = 'analytical')
    plt.plot(spectralindex, Jint, label = 'integrated')
    plt.legend()
    plt.yscale('log')
    plt.show()
    
    plt.plot(spectralindex, Jspe/Jint)
    plt.title("Ratio Jspectral/Jintegral")
    plt.xlabel("CRe spectral index alpha")
    plt.show()
    


# Set global functions
polyfitF = set_polyfitF()


#%%# Run test functions

#test_emission_power_density()
#test_cre_spectrum_normalization()
#test_integrate_total_emissivity()
#compare_spectral_vs_integral()

#%%# Inspect integrated total emissivity for range of cre spectral index

def plot_cre_index_dependence():
    wobs          = 2*pi*100 * MHz
    energygrid    = 10**np.linspace(6,11,100)*eV
    gammagrid     = (energygrid/(me*c**2)).decompose(bases=u.cgs.bases)
    spectralindex = np.linspace(-4,-2,100)
    Jtot          = []
    for a in spectralindex:
        cre_spectrum = set_cre_spectrum(n0=n0, alpha=a, gammas=gammagrid, normalize=True)
        J = integrate_total_emissivity(wobs, cre_spectrum, energygrid)
        Jtot.append(J/J.unit)
    Jtot = np.array(Jtot)*J.unit
    
    # Make the plot
    plt.plot(spectralindex, Jtot/Jtot.unit)
    plt.title("Integrated synchrotron emissivty of normalized cre spectrum\n with spectral index alpha")
    plt.ylabel("Emissivity ({})".format(Jtot.unit))
    plt.xlabel("Spectral index alpha")
    #plt.savefig(path+"integratedemissivity.png")
    plt.show()

#plot_cre_index_dependence()

#%%# Inspect total emissivity when using the powerlaw assumption


energygrid = 10**np.linspace(6,11,int(1e6)) * eV
gammagrid  = (energygrid/(me*c**2)).decompose(bases=u.cgs.bases)
cre        = set_cre_spectrum(n0=n0, alpha=-3, gammas=gammagrid, normalize=False)
Jintegral  = integrate_total_emissivity(vobs=100*MHz, spectrum = cre, gammas=gammagrid)
Jspectral  = spectral_total_emissivity(vobs=100*MHz, alpha= -3)


print("Integrated emissivity = {}".format(Jintegral))
print("Spectral emissivty    = {}".format(Jspectral))





#%%# Find better pointset to sample dJ array for integral over energy


from scipy.stats import truncnorm
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
normal = get_truncated_normal(mean = 9, sd=0.5, low=6, upp=11)
#plt.hist(np.sort(normal.rvs(1000)))
#plt.show()

def find_better_sampling():
    #energygrid = 10**np.linspace(6,11,int(1e6)) * eV
    energygrid = 10**np.sort(normal.rvs(1000))*eV
    gammagrid  = (energygrid/(me*c**2)).decompose(bases=u.cgs.bases)
    cre        = set_cre_spectrum(n0=n0, alpha=-3, gammas=gammagrid, normalize=False)
    dJ, dg = integrate_total_emissivity(vobs=100*MHz, spectrum = cre, gammas=gammagrid, returnsteps=True)
    
    plt.plot(gammagrid[:-1], dJ, '.')
    plt.show()
    
    print(np.sum(0.5*dJ*dg))
    
    return



find_better_sampling()





