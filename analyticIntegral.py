import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv as modBessel2nd
from scipy.special import gamma

from numpy.polynomial import Polynomial

from astropy import constants as cons
from astropy import units as u

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
wobs  = 1. * MHz

def critical_freq(E):
    return (3*electron*Bper/(2*me**3*c**5) * E**2).decompose()

def critical_energy(wc):
    return np.sqrt(wc * (2*me**3 *c**5)/(3*electron*Bper)).decompose()


#%%# Definition of the integrator


def integrate_bessel(xmin=None,xmax=None, steps=10000, xscale=1e10):
    """
    Implementation of a logarithmic variety of the midpoint rule for integration
    of the modified bessel function of the 2nd kind with real order 5/3.
    As a rule of thumb: O(xscale) ~ 2 |O(xmin)|
    """
    # the function decays exponentially, could further implement stopping criteria
    if xmax is None:
        xmax = 1e3 * xmin
    # compacted points along the x-axis (high resolution close to the origin)
    x      = np.empty(steps+1)
    x[:-1] = (xmax-xmin)*(10**np.linspace(-np.log10(xscale),0,steps)-1./xscale) + xmin
    x[-1]  = xmax
    y      = modBessel2nd(5./3,x)
    # apply midpoint rule
    df = []
    for i in range(steps):
        dx = x[i+1] - x[i]
        df.append(dx*(y[i]+y[i+1])/2)
    return np.sum(df)


print(integrate_bessel(xmin=0.001, xmax=1))


xbessel = 10**np.linspace(-8,2,400) # think of x as percentage of wc
ybessel = []
for xv in xbessel:
    ybessel.append(xv*integrate_bessel(xmin = xv))
ybessel = np.array(ybessel)

#%%
plt.plot(xbessel,ybessel)
plt.xscale('log')
#plt.yscale('log')
plt.show()

#%%# Direct Polynomial fit to integration


ebessel = (critical_energy(wobs/xbessel)).to(u.eV)
logebessel = np.log10(ebessel/ebessel.unit)
logxbessel = np.log10(xbessel)
logybessel = np.log10(ybessel)

deg   = 20
p     = Polynomial.fit(logxbessel, logybessel, deg)
ypoly = p(logxbessel)

plt.plot(logxbessel, 10**logybessel, '.')
plt.plot(logxbessel, 10**ypoly)
plt.show()

#%%# Short investigation of degree plotting in energy representation

alpha = -3
supression = (ebessel/(me*c**2))**alpha
supression *= 1./np.max(supression)
degrees = [20,30,40,50]
for deg in degrees:
    p     = Polynomial.fit(logxbessel, logybessel, deg)
    ypoly = p(logxbessel)
    error = (10**logybessel-10**ypoly)
    plt.plot(logebessel, error*supression, '.', label='degree: {}'.format(deg))
plt.legend()
#plt.yscale('log')
plt.show()

#%% How is the error effected by the integration over the loge?

ediff = -np.ediff1d(ebessel)
plt.plot(ebessel[:-1], 10**ypoly[:-1] * supression[:-1]*ediff, '.')
plt.plot(ebessel[:-1], 10**ypoly[:-1] * supression[:-1]*ediff + ediff*error[:-1], '.')
#plt.plot(ebessel[:-1], ediff*error[:-1])
plt.xscale('log')
plt.show()


#%%#

alpha = -3
ediff = -np.ediff1d(ebessel)
supression = (ebessel/(me*c**2))**alpha
supression *= 1./np.max(supression)
degrees    = [20,30,40,50]

for deg in degrees:
    p     = Polynomial.fit(logxbessel, logybessel, deg)
    ypoly = p(logxbessel)
    error = (10**logybessel-10**ypoly)
    plt.plot(ebessel[:-1], 10**ypoly[:-1]*supression[:-1]*ediff + ediff*error[:-1], '.', label="degree {}".format(deg))

plt.plot(ebessel[:-1], ybessel[:-1]*supression[:-1]*ediff, lw=4, label="integration")

plt.legend()
plt.xscale('log')

plt.title("Proportionality profiles of dJ = Ncre*dPw*dE \n(withouth physical constants)")
plt.ylabel("")
plt.show()







#%%# What happens to the fit if you go out of fit range?

newE = 10**np.linspace(6, 12, 400)*u.eV
newxbessel = wobs/critical_freq(newE)
newybessel = []
for xv in newxbessel:
    newybessel.append(xv*integrate_bessel(xmin = xv))
newybessel = np.array(newybessel)
plt.plot(xbessel,ybessel)
plt.plot(newxbessel,newybessel, 'o',color='g')
plt.plot(newxbessel,10**p(np.log10(newxbessel)), color='g')
plt.xscale('log')
plt.yscale('log')
plt.show()


#%%# Analytical discription of F(x) = x*Integral(x)

a=1.
b=1.

def wiki_inverse_gamma(x,a,b):
    return b**a/gamma(a)*x**(-a-1) * np.exp(-b/(x))
x = 10**np.linspace(-8,2,100)
plt.plot(x,wiki_inverse_gamma(x,a,b),label='inverse gamma')
plt.plot(xbessel,ybessel,label = 'integration result')
plt.xscale('log')
plt.legend(loc=0)
plt.show()

## Inverse gamma from scipy


from scipy.stats import invgamma
a,loc,scale = invgamma.fit(ybessel)
invgammafit = invgamma(a,loc,scale)
plt.plot(xbessel,ybessel, label='integration result')
plt.plot(x,invgammafit.pdf(x),label='scipy fit')
plt.xscale('log')
plt.legend(loc=0)
plt.show()



## Write my own fitting routine???

from scipy.optimize import curve_fit

def log_inverse_invgamma(logx, a, b, xshift, yshift):
    return 1/(yshift+np.log10(b**a/gamma(a)*(10**(logx+xshift))**(a-0.2) * np.exp(-b*10**(logx+xshift)) ))

# manual fit:
a     = 0.4
b     = 1.0
xshift = 0
yshift = 0.55

logxbessel = np.log10(xbessel)
logybessel = 1/np.log10(ybessel)


popt, pcov = curve_fit(log_inverse_invgamma, logxbessel, logybessel, bounds = ([0,0,0,0],[1,1,1,1]))
yfit       = log_inverse_invgamma(logxbessel, *popt)

plt.plot(10**logxbessel, 10**logybessel, label='integration result')
plt.plot(10**logxbessel, 10**yfit, label = 'fit log_inverse_gamma')
plt.legend(loc=0)
plt.xscale('log')
#plt.yscale('log')
plt.show()

## Make my own function with the same basic idea

def my_func(logx, logyshift, logxshift, logslope, slope):
    x = 10**(logx+logxshift)
    return logyshift + logslope*logx + slope*x

logxbessel = np.log10(xbessel)
logybessel = np.log10(ybessel)

logyshift = 1
logxshift = 0
logslope  = 1./3
slope     = -0.5
params = (logyshift, logxshift, logslope, slope)

popt, pcov = curve_fit(my_func, logxbessel, logybessel)
yfit       = my_func(logxbessel, *popt)

plt.plot(10**logxbessel, 10**(logybessel), label = "integration result")
plt.plot(10**logxbessel, 10**yfit, label = "fit my_func")
plt.legend(loc="lower left")
plt.xscale('log')
plt.show()


## Use the initial guess to fit the inverse in order to force the peak value to be more correct

def inv_my_func(logx, logyshift, logxshift, logslope, slope):
    x = 10**(logx+logxshift)
    return 1/(logyshift + logslope*logx + slope*x)

popt, pcov = curve_fit(inv_my_func, logxbessel, 1/logybessel, p0=popt)
yfit       = 1/inv_my_func(logxbessel, *popt)

plt.plot(10**logxbessel, 10**logybessel, label='integration result')
plt.plot(10**logxbessel, 10**yfit, label = 'fit my_func')
plt.legend(loc=0)
plt.xscale('log')
plt.show()

wobs = 1. * MHz
ebessel = (critical_energy(wobs/xbessel)).to(u.eV)
plt.plot(ebessel, ybessel, '.', label='integration result')
plt.plot(ebessel, 10**yfit, '.', label = 'fit inv_my_func')
plt.legend(loc=0)
plt.xscale('log')
plt.show()

## Polynomial fit to the error


error = ybessel - 10**yfit
loge  = np.log10(ebessel/ebessel.unit)

plt.plot(loge, error, '.')
plt.title("Error dP_integral - dP_fit")
plt.show()

deg = 30
p   = Polynomial.fit(loge, error, deg)
errorfit = p(loge)
plt.title("Fit to error dPw_integral - dPw_fit")
plt.plot(loge, error, '.')
plt.plot(loge, errorfit)
plt.show()


## Show relative error with supression by powerlaw CRE

fig, ax1 = plt.subplots()

ax1.set_xlabel('CRE energy (eV)')
ax1.set_ylabel('F')
ax1.plot(ebessel, ybessel,label='integration result')
ax1.plot(ebessel, 10**yfit,label = 'inv_my_func')
alphalist  = [-1, -2, -3, -4]

for a in alphalist:
    supression = ((ebessel)/(me*c**2)).decompose()**a
    supression = supression/np.max(supression)
    ax1.plot(ebessel, supression, color='black',label="powerlaw index a = {}".format(a))

ax1.set_ylim(-1,1)
ax1.legend(loc="upper right")

ax2 = ax1.twinx()
ax2.plot(ebessel, (ybessel-10**yfit), color='red',label= 'difference')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(-0.025,0.025)
#ax2.set_yscale('log')
ax2.legend(loc="lower right")


fig.tight_layout()
plt.xscale('log')
plt.show()

##

supression = ((ebessel)/(me*c**2)).decompose()**-1
supression = supression/np.max(supression)

fig, ax1 = plt.subplots()

ax1.set_xlabel('CRE energy (eV)')
ax1.set_ylabel('F(E) * Ncre(E)')
ax1.plot(ebessel, ybessel * supression, label='integration result')
ax1.plot(ebessel, 10**yfit * supression, label='integration result')

ax1.set_ylim(-0.07,0.07)
ax1.legend(loc="upper right")

ax2 = ax1.twinx()
ax2.plot(ebessel, (ybessel-10**yfit) * supression, color='red',label= 'error')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(-0.0025,0.0025)
#ax2.set_yscale('log')
ax2.legend(loc="lower right")


fig.tight_layout()
plt.title("Emissivity density dJvobs/dE")
#plt.xscale('log')
plt.show()

## Do the energy integral to obtain the emissivity

def integrate_midpoint(x, y):
    dI = []
    for i in range(len(x)-1):
        dx = np.abs(x[i+1] - x[i])
        dI.append(dx*(y[i]+y[i+1])/2)
    return sum(dI)


alphalist = np.linspace(-4,-2,50)
J      = []
Jerror = []
for a in alphalist:
    supression = ((ebessel)/(me*c**2)).decompose()**a
    supression = supression/np.max(supression)
    Jnum = integrate_midpoint(ebessel, ybessel*supression)
    Jfit = integrate_midpoint(ebessel, 10**yfit*supression)
    J.append(Jnum)
    Jerror.append((Jfit-Jnum)/Jnum)


plt.plot(alphalist, Jerror)
plt.title("Relative error on emissivity due to approximate fit")
plt.ylabel("relative emisivity error")
plt.xlabel("alpha")
plt.show()



##

def log_invgamma(logx, a, b, xshift, yshift):
    return yshift+np.log10(b**a/gamma(a)*(10**(logx+xshift))**(a-0.2) * np.exp(-b*10**(logx+xshift)) )

yfit       = log_invgamma(logxbessel, *popt)

plt.plot(10**logxbessel, ybessel,label='integration result')
plt.plot(10**logxbessel, 10**yfit,label = 'fit inverse_gamma')
plt.legend(loc=0)
plt.xscale('log')
#plt.yscale('log')
plt.show()


## The same plot in energy domain

wobs      = 100 * MHz
energies  = 10**np.linspace(6,11,100)*u.eV.cgs
wc        = critical_freq(energies)
ec        = critical_energy(wc)
xphysical = wobs/wc

ebessel = (critical_energy(wobs/xbessel)).to(u.eV)
plt.plot(ebessel, ybessel,label='integration result')
plt.plot(ebessel, 10**yfit,label = 'fit inverse_gamma')
plt.ylabel("F(x)")
plt.xlabel("CRE energy (eV)")
plt.legend(loc=0)
plt.xscale('log')
#plt.yscale('log')
plt.show()


## Show difference in fit


fig, ax1 = plt.subplots()

ax1.set_xlabel('x')
ax1.set_ylabel('F(x)')
ax1.plot(10**logxbessel, ybessel,label='integration result')
ax1.plot(10**logxbessel, 10**yfit,label = 'fitmethod: log_inverse_invgamma')
ax1.set_ylim(-1,1)
ax1.legend(loc=0)

ax2 = ax1.twinx()
ax2.set_ylabel("diference F(x) - fit(x)",color='red')
ax2.plot(xbessel, ybessel-10**yfit,color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(-0.03,0.03)

fig.tight_layout()
plt.xscale('log')
plt.show()



##

plt.plot(10**logxbessel, ybessel,label='integration result')
plt.plot(10**logxbessel, 10**yfit,label = 'fit log_inverse_gamma')
plt.plot(10**logxbessel, (ybessel-10**yfit)/ybessel, label='fraction difference/total')
plt.legend(loc=0)
plt.xscale('log')
plt.show()






## Adding extra width parameter

def inverse_gamma(x, a, b, xshift, yshift, width):
    return b**a/gamma(a) * ((x+xshift)/width)**(a+1) * np.exp(-b*(x+xshift)/width) + yshift


popt, pcov = curve_fit(inverse_gamma, xbessel, ybessel)
yfit       = inverse_gamma(xbessel, *popt)

plt.plot(xbessel, ybessel, label = 'integration result')
plt.plot(xbessel, yfit, label = 'fit inverse_gamma')
plt.legend(loc=0)
plt.xscale('log')
plt.show()






