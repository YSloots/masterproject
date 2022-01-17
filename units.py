"""
Investigation of the units used in the HammurabiX paper: https://arxiv.org/pdf/1907.00207.pdf
The equations that are investigated are from Appendix A. Synchrotron Emission
"""
from astropy import constants as cons
from astropy import units as u
import numpy as np

gauss_B = (u.g/u.cm)**(0.5)/u.s
equiv_B = [(u.G, gauss_B, lambda x: x, lambda x: x)]

## Critical Frequency

me = cons.m_e.cgs
c  = cons.c.cgs
Ecre = 0.511e6*u.eV.cgs # stationary electron
gamma = (Ecre/(me*c**2)).decompose()
print("gamma", gamma)

electron = (cons.e).gauss
print(electron.unit)
Bper  = (1e-6 * gauss_B)
eBper = electron*Bper
mec    = me*c
wc     = (eBper / mec * (Ecre/(me*c**2))**2).decompose()
print(wc, "needs corretion factor of c?")
wc     = (eBper / mec * (Ecre/(me*c**2))**2).decompose()
print(wc)

# Equation A5
def critical_freq(E):
    return (3*electron*Bper/(2*me*c) * (E/(me*c**2))**2).decompose()
    
MeV = 1e6*u.eV.cgs
print("Ecre = 100MeV, gives wc = {}".format( critical_freq(100*MeV)))
print("Ecre = 200Mev, gives wc = {}".format( critical_freq(200*MeV)))
print('\n')

## Emitted Power Density ([Ptot]=[Pdensity(w)*dw])
print("Investigate emitted power")
P = 1*u.erg.cgs/u.s /u.s**-1 # power per freq

def dummy_integralK():
    """Just for the example"""
    return 1.

electron     = (cons.e).gauss
print(electron.unit)
Bper = 1e-6 * gauss_B
pi    = np.pi

e3Bper  = electron**3*Bper
taumec2 = 2*pi*me*c**2
Ptotw   = np.sqrt(3)*e3Bper/taumec2 * dummy_integralK()
print(Ptotw.decompose(), "Should be: ", P.unit.decompose())

# Equation A2
def emission_power(E):
    return (np.sqrt(3)*electron**3*Bper)/(2*pi*me*c**2) * dummy_integralK()



