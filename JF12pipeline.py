#%% Imports and settings

import imagine as img
from imagine.simulators.synchrotronlos import SpectralSynchrotronEmissivitySimulator
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12

import numpy as np
from astropy.coordinates import spherical_to_cartesian
from astropy.coordinates import cartesian_to_spherical

rundir  = 'runs/mockdata'
figpath = 'figures/'
logdir  = 'log/'

import astropy.units as u
MHz   = 1e6 / u.s

# Gobal testing constants
Ndata = 100
observing_frequency = 90*MHz
dunit = u.K/u.kpc

from time import perf_counter
import matplotlib.pyplot as plt

import os


#%%

xmax = 15*u.kpc
ymax = 15*u.kpc
zmax =  2*u.kpc
cartesian_grid = img.fields.UniformGrid(box=[[-xmax, xmax],
                                             [-ymax, ymax],
                                             [-zmax, zmax]],
                                             resolution = [30,30,30])

Bfield = WrappedJF12(grid=cartesian_grid)
Bdata  = Bfield.get_data()
print(dir(Bfield))
print(Bfield.parameter_names)
print(Bfield.parameters)

