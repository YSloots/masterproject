"""
In order to calculate the emitted power of the synchrotron emission of CREs at
a certain energy we need to perform an integral of the moddified bessel function
of the second kind. The lower bound of this integral is given by the observed
synchrotron frequency devided by the critical frequency, eg: xmin = w/wc.

For typical observed frequencies and CRE energies this parameter can be of
order A to B. 

In this exploratory script we find that the value of the integral is indeed
very sensitive to hyper parameters when xmin is below a certain order of magnitude.

Which means that for the application to synchrotron emissivity we are in the
stable/unstable integration area!

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv as modBessel2nd
from astropy import constants as cons
from astropy import units as u

path = '/home/ysloots/MasterProject/project/figures/'
path = 'C:/Users/ysloo/Documents/Masterstage/masterproject/project'

# need to manually acount for cgs units of Bfield
gauss_B = (u.g/u.cm)**(0.5)/u.s
equiv_B = [(u.G, gauss_B, lambda x: x, lambda x: x)]


#%%# Investigate x scales, goal: compact points for small values of x

scales = 10**np.arange(1,10)

n = 0
for s in scales:
    x = 10**np.linspace(-np.log10(s),0,100)-1./s    
    plt.plot(x,n*np.ones(100),'.')
    n+=1

plt.ylim(-0.1*n,n)
plt.title("Different compactness for integration points close to the origin")
plt.show()

#%%# Investigate physical range of xmin


# Equation A5
def critical_freq(E):
    
    #typical values and constants
    Bperp = (1e-6 * gauss_B)
    e     = cons.e.gauss
    me    = cons.m_e.cgs
    c     = cons.c.cgs
    
    return (3*e*Bperp/(2*me*c) * (E/(me*c**2))**2).decompose()

Ecre = 10**np.linspace(6,12,50) * u.eV.cgs
wobs = 10**np.linspace(6,12,50) * u.Hz.cgs

ee, ww = np.meshgrid(Ecre,wobs,sparse=True)
xmin   = ww/critical_freq(ee)
print(xmin.decompose())

levels = [-2]
cplot = plt.contourf(Ecre/u.eV,wobs/u.Hz, np.log10(xmin/xmin.unit),30,cmap='Blues')
clines = plt.contour(Ecre/u.eV,wobs/u.Hz, np.log10(xmin/xmin.unit),levels,colors=('r',),linewidths=(3,))

cbar = plt.colorbar(cplot)
cbar.add_lines(clines)

plt.yscale('log')
plt.xscale('log')
plt.ylabel('wobs')
plt.xlabel('Ecre')
plt.show()


#%%# How does the bessel function look like? -> approach: integrate with midpoint rule

x = np.linspace(0,10,100)
y = modBessel2nd(5./3,x)
xp = 10**np.linspace(-2,1,100)
yp = modBessel2nd(5./3,xp)
plt.plot(x,y)
plt.plot(xp,yp,'.')
plt.yscale('log')
plt.xscale('log')
plt.show()


#%%# Definition of the integrator


def integrate_bessel(xmin=None,xmax=None, steps=10000, xscale=1e6):
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

xbessel = 10**np.linspace(-8,2,100) # think of x as percentage of wc
ybessel = []
for xv in xbessel:
    ybessel.append(xv*integrate_bessel(xmin = xv))
ybessel = np.array(ybessel)
plt.plot(xbessel,ybessel)
plt.xscale('log')
plt.show()



#%%# Analytical discription of F(x) = x*Integral(x)

from scipy.special import gamma

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

#%%# Inverse gamma from scipy


from scipy.stats import invgamma
a,loc,scale = invgamma.fit(ybessel)
invgammafit = invgamma(a,loc,scale)
plt.plot(xbessel,ybessel)
plt.plot(x,invgammafit.pdf(x))
plt.xscale('log')
plt.show()


#%%# Write my own fitting routine???

def log_inverse_gamma(x, a, b, shift, scale):
    return np.log10(scale*b**a/gamma(a)*(x-shift)**(a+1) * np.exp(-b*(x-shift)) )
    
a     = 0.5
b     = 0.5
shift = 0
scale = 1.

plt.plot(x, log_inverse_gamma(x,a,b,shift,scale), label = 'inverse gamma a={}, b={}'.format(a,b))
plt.plot(x,np.log10(ybessel), label='integration result')
plt.xscale('log')
plt.legend(loc=0)
plt.show()


#%%# Gridsearch for sensitivity to hyper parameters

nsteps = 1000*np.arange(6, 10)
scale  = 10**np.arange(6,12)
values = np.empty((len(nsteps),len(scale)))


xmin = 1e-6
for i in range(len(nsteps)):
    for j in range(len(scale)):
        values[i,j] = integrate_bessel(xmin, 2, steps = nsteps[i], xscale=scale[j])

print(np.min(values), np.max(values))
reldif = (values - np.min(values))/np.min(values)

plt.contourf(nsteps,np.log10(scale),values.T,40, cmap='YlOrBr')
plt.xlabel('integration steps')
plt.ylabel('order of magnitude compactness')
plt.title('Integral values for xmin = {}'.format(xmin))
plt.colorbar()
plt.savefig(path+'sensitivityIsmallxmin.png')
plt.show()

#%%#

xmin = 1e-2
for i in range(len(nsteps)):
    for j in range(len(scale)):
        values[i,j] = integrate_bessel(1e-2, 2, steps = nsteps[i], xscale=scale[j])
reldif = (values - np.min(values))/np.min(values)

plt.contourf(nsteps,np.log10(scale),values.T,40, cmap='YlOrBr')

plt.xlabel('integration steps')
plt.ylabel('order of magnitude compactness')
plt.title('Integral values for xmin = {}'.format(xmin))
plt.colorbar()
plt.savefig(path+'sensitivityIlarexmin.png')
plt.show()




