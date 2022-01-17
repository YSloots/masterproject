import matplotlib.pyplot as plt
import numpy as np

path = '/home/ysloots/MasterProject/project/figures/'



## "General" functions
from scipy.special import gamma
from scipy.special import kv as modBessel2nd

def spectral_integralF(mu):
	return 2**(mu+1)*gamma(mu/2 + 7./3)*gamma(mu/2+2./3)/(mu+2)

def spectral_integralG(mu):
	return 2**mu * gamma(mu/2+4./3)*gamma(mu/2+2./3)

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

x = 10**np.linspace(-8,2,200) # think of x as percentage of wc
y = []
for xv in x:
	y.append(xv*integrate_bessel(xmin = xv))
plt.plot(x,y)

plt.title("Behaviour worst case F(x) = x*Integral(x)")
plt.xscale('log')
plt.show()


#%%# Physical functions and definitions
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
n0    = 10 * u.cm**-3
alpha = -3
MHz   = 1e6 / u.s
MeV   = 1e6 * u.eV

def critical_freq(E):
	return (3*electron*Bper/(2*me*c) * (E/(me*c**2))**2).decompose()

def emission_power_density(w,E):

	wc   = critical_freq(E)

	x    = w/wc

	return ((np.sqrt(3)*electron**3*Bper)/(2*pi*me*c**2) * x * integrate_bessel(xmin=x)).decompose()

def Ncre_spectrum(E):
	return n0*(E/E.unit)**alpha

def const_spectrum(E):
	return 1000*np.ones(len(E))*u.cm**-3

#%%# Inspect physical functions

# typical observing frequencies and cre energies
dPunit = u.kg*u.m**2/u.s**2
wobs = 10**np.linspace(0,4,100) * MHz
creE = 10**np.linspace(6,11,100) * u.eV.cgs

def calc_power_grid():
	powergrid = []
	for wo in wobs:
		dP   = []
		for e in creE:
			dP.append(emission_power_density(wo, e)/dPunit)
		powergrid.append(dP)
	return powergrid

#powergrid = calc_power_grid()

plt.contourf(creE, wobs/(2*pi), powergrid, 30, cmap='Blues')
plt.colorbar()

plt.title("Synchrotron emission power density")
plt.ylabel("Observed frequency (Hz)")
plt.xlabel("CR-electron energy (eV)")
plt.yscale('log')
plt.xscale('log')

plt.savefig(path + "emissionpowerdensity.png")

plt.show()


#%%# Final Total Emissivity

def const_spectrum(E):
	return 1000*np.ones(len(E))*u.cm**-3

def emissivity(vobs, cre_spectrum, **kwargs):
	#has to perform some sort of integral over energy
	E    = 10**np.linspace(7,11,200)*u.eV.cgs
	wobs = 2*pi*vobs
	Ncre = cre_spectrum(E)

	Ptot = []
	for e in E:
		Ptot.append(emission_power_density(wobs,e)/dPunit)
	Ptot = np.array(Ptot)
	
	# summing this would be the integral
	Jtot = 0.5*Ncre*Ptot*dPunit
	
	return E, Jtot.decompose()

e,j = emissivity(100*MHz, Ncre_spectrum)

plt.plot(e,j)
plt.xscale('log')
plt.show()


#%%# Final Total Powerlaw Emissivity

# !!! Problem for (alpha-3)/2 = -2 -> alpha =/= -1 which would be a powerlaw in opposite direction ... we are ok

vobs  = 100 * MHz
temp1 = 4*pi*vobs*me*c/(3*electron*Bper)
temp2 = np.sqrt(3)*electron**3*Bper*n0/(8*pi*me*c**2)

def powerlaw_emissivity(vobs, aplha):
	A = (4*pi*vobs*me*c/(3*electron*Bper))**((1-np.abs(alpha))/2)
	B = np.sqrt(3)*electron**3*Bper*n0/(8*pi*me*c**2)
	C = spectral_integralF( (np.abs(alpha)-3)/2 )
	print(A.decompose())
	print(B.decompose())
	print(C)
	
	return (A * B * C).decompose()

jtot = powerlaw_emissivity(100*MHz, 3)








