#%% Imports and settings
#===================================================================================
# Imagine
import imagine as img
from imagine.simulators.synchrotronlos import SpectralSynchrotronEmissivitySimulator
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12Factory
from imagine.fields.field_utility import MagneticFieldAdder
from imagine.fields.field_utility import ArrayMagneticField

# Utility
import os
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from astropy.coordinates import spherical_to_cartesian
from astropy.coordinates import cartesian_to_spherical
import struct
from scipy.stats import truncnorm

# Directory paths
rundir    = 'runs/mockdata'
figpath   = 'figures/simulator_testing/'
fieldpath = 'arrayfields/'
logdir    = 'log/'

# Gobal testing constants
import astropy.units as u
MHz   = 1e6 / u.s
Ndata = 1
observing_frequency = 90*MHz
dunit = u.K/u.kpc
global_dist_error = 0.5
global_brightness_error = 0.01

print("\n")

#%% Setup 2 los
#===================================================================================

def fill_imagine_dataset(data):
    fake_dset = img.observables.TabularDataset(data,
                                               name='average_los_brightness',
                                               frequency=observing_frequency,
                                               units=dunit,
                                               data_col='brightness',
                                               err_col='err',
                                               lat_col='lat',
                                               lon_col='lon')
    return img.observables.Measurements(fake_dset)

def randrange(minvalue,maxvalue,Nvalues,seed=3145):
    np.random.seed(seed)
    """Returns uniform random values bewteen min and max"""
    return (maxvalue-minvalue)*np.random.rand(Nvalues)+minvalue

def get_label_FB(Ndata, seed=31415):
    np.random.seed(seed)
    NF = int(0.2*Ndata) # 20% of all measurements are front measurements
    F  = np.full(shape=(NF), fill_value='F', dtype=str)
    B  = np.full(shape=(Ndata-NF), fill_value='B', dtype=str)
    FB = np.hstack((F,B))
    np.random.shuffle(FB) 
    #return FB
    return F # only front measurements

def run_test_los():
    """
    This simulation setup assumes the following fields and observing configuration:

    """
    # Produce empty data format
    T     = np.zeros(Ndata)*dunit # placeholder
    T_err = np.zeros(Ndata)*dunit # placeholder
    xmax = 20*u.kpc
    ymax = 20*u.kpc
    zmax =  2*u.kpc
    x = -8.5*np.ones(Ndata)*u.kpc # with observer at x=-8.5
    y = 8.0*np.ones(Ndata)*u.kpc
    z = np.zeros(Ndata)*u.kpc
    hIIdist, lat, lon = cartesian_to_spherical(x+8.5*u.kpc,y,z)
    fake_data = {'brightness':T,'err':T_err,'lat':lat,'lon':lon}
    mea       = fill_imagine_dataset(data=fake_data)
    # Setup the Galactic field models
    cartesian_grid = img.fields.UniformGrid(
        box=[[-xmax, xmax],
             [-ymax, ymax],
             [-zmax, zmax]],
        resolution = [40,40,3]) # skipping x=y=0
    cre = img.fields.ConstantCosmicRayElectrons(
        grid       = cartesian_grid,
        parameters = {'density':1.0e-7*u.cm**-3,'spectral_index':-3.0})
    Bfield = img.fields.ConstantMagneticField(
        grid       = cartesian_grid,
        parameters = {'Bx': 6*u.microgauss,
                      'By': 0*u.microgauss,
                      'Bz': 0*u.microgauss})
    field_list = [cre, Bfield]
    # Setup observing configuration
    observer = np.array([-8.5,0,0])*u.kpc
    dist_err = global_dist_error * hIIdist
    FB       = get_label_FB(Ndata)
    config = {'grid'    :cartesian_grid,
              'observer':observer,
              'dist':hIIdist,
              'e_dist':dist_err,
              'lat':lat,
              'lon':lon,
              'FB':FB} # only front measurements
    # Initialize simulator
    test_sim   = SpectralSynchrotronEmissivitySimulator(measurements=mea, sim_config=config) 
    # Run simulation with field list
    simulation = test_sim(field_list)
    # Print brightness
    key = ('average_los_brightness', 0.09000000000000001, 'tab', None)
    sim_brightness = simulation[key].data[0] * simulation[key].unit
    print(sim_brightness)

run_test_los()