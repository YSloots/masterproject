
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

# Directory paths
rundir    = 'runs/mockdata'
figpath   = 'figures/simulator_testing/'
fieldpath = 'arrayfields/'
logdir    = 'log/'

# Gobal testing constants
import astropy.units as u
MHz   = 1e6 / u.s
Ndata = 10
observing_frequency = 90*MHz
dunit = u.K/u.kpc

print('\n')

#%% Code reduction functions
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

def produce_mock_data(field_list, mea, config, noise=0.01):
    """
    Runs the simulator once to produce a simulated dataset
    - assume we know the exact positions for the HII regions
    - add some gaussian noise to the brightness temperate with some relative error
    """
    if config['e_dist'] is None:
        mock_config = config
    else:
        mock_config = config.copy()
        mock_config['e_dist'] = None
    test_sim   = SpectralSynchrotronEmissivitySimulator(measurements=mea, sim_config=mock_config) 
    simulation = test_sim(field_list)
    key = ('average_los_brightness', 0.09000000000000001, 'tab', None)
    sim_brightness = simulation[key].data[0] * simulation[key].unit
    brightness_error = noise*sim_brightness
    brightness_error[brightness_error==0]=np.min(brightness_error[np.nonzero(brightness_error)])
    return sim_brightness, brightness_error

def randrange(minvalue,maxvalue,Nvalues,seed=3145):
    np.random.seed(seed)
    """Returns uniform random values bewteen min and max"""
    return (maxvalue-minvalue)*np.random.rand(Nvalues)+minvalue

def get_label_FB(Ndata, seed=31415):
    np.random.seed(seed)
    NF = int(0.5*Ndata) # 20% of all measurements are front measurements
    F  = np.full(shape=(NF), fill_value='F', dtype=str)
    B  = np.full(shape=(Ndata-NF), fill_value='B', dtype=str)
    FB = np.hstack((F,B))
    np.random.shuffle(FB) 
    return FB

def load_JF12rnd(label, shape=(40,40,10,3)):
    with open(fieldpath+"brnd_{}.bin".format(int(label)), "rb") as f:
        arr = f.read()
        arr = struct.unpack("d"*(len(arr)//8), arr[:])
        arr = np.asarray(arr).reshape(shape)
    return arr


#%% Pipeline controllers
#===================================================================================

# simulator testing
def test_simulator(rel_dist_err, Ndata=10, resolution = [40,40,10]):

    # Produce empty data format
    T     = np.zeros(Ndata)*dunit # placeholder
    T_err = np.zeros(Ndata)*dunit # placeholder
    xmax = 20*u.kpc
    ymax = 20*u.kpc
    zmax =  2*u.kpc
    x = randrange(-0.9*xmax,0.9*xmax,Ndata, seed = 1) # constant seed
    y = randrange(-0.9*ymax,0.9*ymax,Ndata, seed = 2)
    z = randrange(-0.9*zmax,0.9*zmax,Ndata, seed = 4)
    hIIdist, lat, lon = cartesian_to_spherical(x,y,z)

    #print("Script distances:\n", hIIdist)
    fake_data = {'brightness':T,'err':T_err,'lat':lat,'lon':lon}
    mea       = fill_imagine_dataset(data=fake_data)
    
    # Setup the Galactic field models
    cartesian_grid = img.fields.UniformGrid(box=[[-xmax, xmax],
                                                 [-ymax, ymax],
                                                 [-zmax, zmax]],
                                                 resolution = resolution) # skipping x=y=0
    cre = img.fields.ConstantCosmicRayElectrons(grid=cartesian_grid,
                                            parameters={'density':1.0e-7*u.cm**-3,'spectral_index':-3.0})
    Bfield = img.fields.ConstantMagneticField(
        grid = cartesian_grid,
        parameters={'Bx': 6*u.microgauss,
                    'By': 0*u.microgauss,
                    'Bz': 0*u.microgauss})
    
    # Setup observing configuration
    observer = np.array([0,0,0])*u.kpc
    dist_err = None
    FB       = get_label_FB(Ndata)
    print(FB)
    config = {'grid'    :cartesian_grid,
              'observer':observer,
              'dist':hIIdist,
              'e_dist':dist_err,
              'lat':lat,
              'lon':lon,
              'FB':FB}

    los_brightness = []
    for err in rel_dist_err:
        config['e_dist'] = err*config['dist']
        # Do simulation
        test_sim   = SpectralSynchrotronEmissivitySimulator(measurements=mea, sim_config=config) 
        simulation = test_sim(field_list=[cre,Bfield])
        key = key = ('average_los_brightness', 0.09000000000000001, 'tab', None)
        sim_brightness = simulation[key].data[0] * simulation[key].unit
        print("Script final brightness: \n", sim_brightness)
        los_brightness.append(np.array(sim_brightness,dtype=np.float64))
    print(los_brightness)
    
test_simulator(rel_dist_err = [0,0.1,1])


