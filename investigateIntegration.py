# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:00:16 2022

@author: ysloo
"""

# Main imports and settings
#path = '/home/ysloots/MasterProject/project/figures/'
path = 'C:/Users/ysloo/Documents/Masterstage/masterproject/project'

import matplotlib.pyplot as plt
import numpy as np

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

#%% Definitions

from scipy.special import kv as modBessel2nd

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