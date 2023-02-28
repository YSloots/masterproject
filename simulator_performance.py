
#%% Imports and settings
#===================================================================================
# Imagine
import imagine as img
from imagine.simulators.synchrotronlos import SpectralSynchrotronEmissivitySimulator
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12


# Utility
import os
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from astropy.coordinates import spherical_to_cartesian
from astropy.coordinates import cartesian_to_spherical
import struct
import seaborn as sns
import pandas as pd

# Directory paths
rundir    = 'runs/performance'
figpath   = 'figures/performance/'
fieldpath = 'arrayfields/'
logdir    = 'log/'

# Gobal testing constants
import astropy.units as u
MHz   = 1e6 / u.s
GHz   = 1e9 / u.s
Ndata = 100
observing_frequency = 74*MHz
dunit = u.K/u.kpc
global_dist_error = 0.001
global_brightness_error = 0.01
key = ('average_los_brightness', 0.07400000000000001, 'tab', None) # simulation key

print("\n")

#%% Controle funtions
#===================================================================================

def randrange(minvalue,maxvalue,Nvalues,seed=3145):
    np.random.seed(seed)
    """Returns uniform random values bewteen min and max"""
    return (maxvalue-minvalue)*np.random.rand(Nvalues)+minvalue

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

def get_label_FB(Ndata, seed=31415):
    np.random.seed(seed)
    NF = int(0.2*Ndata) # 20% of all measurements are front measurements
    F  = np.full(shape=(NF), fill_value='F', dtype=str)
    B  = np.full(shape=(Ndata-NF), fill_value='B', dtype=str)
    FB = np.hstack((F,B))
    np.random.shuffle(FB) 
    return FB

def produce_basic_setup():
    """
    This simulation setup assumes the following fields and observing configuration:
    - The JF12 regular GMF
    - An exponential CRE density profile with constant spectral index alpha=3
    - A 40x40x4 kpc^3 simulation box with resolution [40,40,10]
    - Uniformly distributed HIIregions
    - Randomly assigned 20% front los
    - And an observer located at x=-8.5kpc in Galactocentric coordinates
    """
    # Produce empty data format
    T     = np.zeros(Ndata)*dunit # placeholder
    T_err = np.zeros(Ndata)*dunit # placeholder
    xmax = 20*u.kpc
    ymax = 20*u.kpc
    zmax =  2*u.kpc
    x = randrange(-0.9*xmax,0.9*xmax,Ndata,seed=100) # constant seed
    y = randrange(-0.9*ymax,0.9*ymax,Ndata,seed=200) # constant seed
    z = randrange(-0.9*zmax,0.9*zmax,Ndata,seed=300) # constant seed
    hIIdist, lat, lon = cartesian_to_spherical(x+8.5*u.kpc,y,z)
    fake_data = {'brightness':T,'err':T_err,'lat':lat,'lon':lon}
    mea       = fill_imagine_dataset(data=fake_data)
    # Setup the Galactic field models
    cartesian_grid = img.fields.UniformGrid(box=[[-xmax, xmax],
                                                 [-ymax, ymax],
                                                 [-zmax, zmax]],
                                                 resolution = [40,40,10]) # skipping x=y=0
    cre = img.fields.PowerlawCosmicRayElectrons(
        grid       = cartesian_grid,
        parameters = {'scale_radius':10*u.kpc,
                    'scale_height':1*u.kpc,
                    'central_density':1e-5*u.cm**-3,
                    'spectral_index':-3})
    Bfield = WrappedJF12(
        grid       = cartesian_grid,
        parameters = {'b_arm_1': .1, 'b_arm_2': 3.0, 'b_arm_3': -.9, 'b_arm_4': -.8, 'b_arm_5': -2.,
                    'b_arm_6': -4.2, 'b_arm_7': .0, 'b_ring': .1, 'h_disk': .4, 'w_disk': .27,
                    'Bn': 1.4, 'Bs': -1.1, 'rn': 9.22, 'rs': 16.7, 'wh': .2, 'z0': 5.3, 'B0_X': 4.6,
                    'Xtheta_const': 49, 'rpc_X': 4.8, 'r0_X': 2.9, })
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
              'FB':FB}
    return Bfield, cre, config, mea

#%% Get time statistics
#===================================================================================

def get_simulation_time(Nsimulations):
    Bfield, cre, config, mea = produce_basic_setup()
    field_list = [Bfield, cre]
    
    # Initialize and do simulation
    init_time = []
    simu_time = []
    for n in range(Nsimulations):
        start      = perf_counter()
        test_sim   = SpectralSynchrotronEmissivitySimulator(measurements=mea, sim_config=config) 
        init_time.append(perf_counter()-start)
        simulation = test_sim(field_list)
        simu_time.append(perf_counter()-start-init_time[-1])
        #sim_brightness = simulation[key].data[0] * simulation[key].unit
        #print(sim_brightness)
    init_time = np.array(init_time)
    simu_time = np.array(simu_time)
    np.save(logdir+'simulator_performance.npy', (init_time, simu_time))


def inspect_simulator_performance():
    data = np.load(logdir+'simulator_performance.npy', allow_pickle=True)
    init_time = data[0,:]
    simu_time = data[1,:]
    print(init_time)
    print(simu_time)
    print(np.mean(init_time[1:]), np.std(init_time[1:]))
    print(np.mean(simu_time[1:]), np.std(simu_time[1:]))

#get_simulation_time(Nsimulations=100)
inspect_simulator_performance()
