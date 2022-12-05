#%% Imports and settings

# Imagine
import imagine as img
from imagine.simulators.synchrotronlos import SpectralSynchrotronEmissivitySimulator
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12Factory
from imagine.fields.field_utility import MagneticFieldAdder
from imagine.fields.field_utility import ArrayMagneticField
from imagine.fields.field_factory import FieldFactory

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
Ndata = 100
observing_frequency = 90*MHz
dunit = u.K/u.kpc


#===========================================================================

def test_JF12_Field():
    xmax = 15*u.kpc
    ymax = 15*u.kpc
    zmax =  2*u.kpc
    cartesian_grid = img.fields.UniformGrid(
        box=[[-xmax, xmax],
             [-ymax, ymax],
             [-zmax, zmax]],
        resolution = [3,3,3])

    Bfield = WrappedJF12(grid=cartesian_grid)
    print("\nRegular JF12Field()")
    print(Bfield.get_data()) # call to compute_field()
    print(Bfield.parameter_names)

#test_JF12_Field()


#===========================================================================

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
    """Runs the simulator once to produce a simulated dataset"""
    test_sim   = SpectralSynchrotronEmissivitySimulator(measurements=mea, sim_config=config) 
    simulation = test_sim(field_list)
    key = ('average_los_brightness', 0.09000000000000001, 'tab', None)
    sim_brightness = simulation[key].data[0] * simulation[key].unit
    sim_brightness += np.random.normal(loc=0, scale=noise*sim_brightness, size=Ndata)*simulation[key].unit
    return sim_brightness

def randrange(minvalue,maxvalue,Nvalues):
    """Returns uniform random values bewteen min and max"""
    return (maxvalue-minvalue)*np.random.rand(Nvalues)+minvalue

def get_label_FB(Ndata):
    NF = int(0.2*Ndata) # 20% of all measurements are front measurements
    F  = np.full(shape=(NF), fill_value='F', dtype=str)
    B  = np.full(shape=(Ndata-NF), fill_value='B', dtype=str)
    FB = np.hstack((F,B))
    np.random.shuffle(FB) 
    return FB

def load_JF12rnd(label, shape=(40,40,10,3)):
    with open(fieldpath+"brnd_{}.bin".format(label), "rb") as f:
        arr = f.read()
        arr = struct.unpack("d"*(len(arr)//8), arr[:])
        arr = np.asarray(arr).reshape(shape)
    return arr

#===========================================================================

def pipeline_debugger():
    print("\n===Running pipeline_debugger ===")

    # Produce empty data format
    T     = np.zeros(Ndata)*dunit # placeholder
    T_err = np.zeros(Ndata)*dunit # placeholder
    xmax = 20*u.kpc
    ymax = 20*u.kpc
    zmax =  2*u.kpc
    x = randrange(-0.9*xmax,0.9*xmax,Ndata)
    y = randrange(-0.9*ymax,0.9*ymax,Ndata)
    z = randrange(-0.9*zmax,0.9*zmax,Ndata)
    hIIdist, lat, lon = cartesian_to_spherical(x+8.5*u.kpc,y,z)
    fake_data = {'brightness':T,'err':T_err,'lat':lat,'lon':lon}
    mea       = fill_imagine_dataset(data=fake_data)
    
    # Setup the Galactic field models
    cartesian_grid = img.fields.UniformGrid(box=[[-xmax, xmax],
                                                 [-ymax, ymax],
                                                 [-zmax, zmax]],
                                                 resolution = [40,40,10]) # skipping x=y=0
    cre = img.fields.PowerlawCosmicRayElectrons(grid=cartesian_grid,
                                            parameters = {'scale_radius':10*u.kpc,
                                                         'scale_height':1*u.kpc,
                                                         'central_density':1e-5*u.cm**-3,
                                                         'spectral_index':-3})
    
    print("Defining WrappedJF12")
    Bfield1 = WrappedJF12(
        grid = cartesian_grid,
        parameters = {'b_arm_1': .1, 'b_arm_2': 3.0, 'b_arm_3': -.9, 'b_arm_4': -.8, 'b_arm_5': -2.,
                          'b_arm_6': -4.2, 'b_arm_7': .0, 'b_ring': .1, 'h_disk': .4, 'w_disk': .27,
                          'Bn': 1.4, 'Bs': -1.1, 'rn': 9.22, 'rs': 16.7, 'wh': .2, 'z0': 5.3, 'B0_X': 4.6,
                          'Xtheta_const': 49, 'rpc_X': 4.8, 'r0_X': 2.9, })

    #generate_JF12rnd(grid=cartesian_grid) # calls Hammurabi
    Barray  = load_JF12rnd()
    beta    = 1.0
    Bfield2 = ArrayMagneticField(grid=cartesian_grid,
                                parameters = {'array_field_amplitude':beta, 'array_field':Barray*u.microgauss})

    Btotal = MagneticFieldAdder(grid=cartesian_grid,
                                field_1=WrappedJF12,
                                field_2=ArrayMagneticField,
                                parameters = {  'array_field_amplitude':beta,
                                                'b_arm_1': .1, 'b_arm_2': 3.0, 'b_arm_3': -.9, 'b_arm_4': -.8, 'b_arm_5': -2.,
                                                'b_arm_6': -4.2, 'b_arm_7': .0, 'b_ring': .1, 'h_disk': .4, 'w_disk': .27,
                                                'Bn': 1.4, 'Bs': -1.1, 'rn': 9.22, 'rs': 16.7, 'wh': .2, 'z0': 5.3, 'B0_X': 4.6,
                                                'Xtheta_const': 49, 'rpc_X': 4.8, 'r0_X': 2.9, 
                                                'array_field': Barray*u.microgauss})

    Bdata = Btotal.get_data()
    #print("Field data: \n", Bdata)
    print("Parameter_names:\n", Btotal.parameter_names)
    print("Total field name: ", Btotal.NAME)
    
   
    B_factory = FieldFactory(field_class = Btotal, grid=cartesian_grid)
    B_factory.active_parameters = ('b_arm_1','b_arm_2')
    B_factory.priors = {        
    'b_arm_1':img.priors.FlatPrior(xmin=0, xmax=10),
    'b_arm_2':img.priors.FlatPrior(xmin=0, xmax=10),
    }

    print("\n=== Investigating NAME ===")
    print(isinstance(B_factory, FieldFactory))
    print(B_factory.default_parameters)
    print(B_factory.active_parameters)
    print(B_factory.field_class)
    print(B_factory.field_class.NAME)
    print(B_factory.field_name)
    
    print("\n=== These should be the same ===")
    print(Btotal.NAME) # the way a name of a field is accesed
    print(B_factory.field_name) # the way the name of a field in a factory is accesed
    
    return

#pipeline_debugger()

#===========================================================================

# pipeline controller
def JF12pipeline_MagneticFieldAdder():

    # Remove old pipeline
    os.system("rm -r runs/mockdata/*")
    
    # Produce empty data format
    T     = np.zeros(Ndata)*dunit # placeholder
    T_err = np.zeros(Ndata)*dunit # placeholder
    xmax = 20*u.kpc
    ymax = 20*u.kpc
    zmax =  2*u.kpc
    x = randrange(-0.9*xmax,0.9*xmax,Ndata)
    y = randrange(-0.9*ymax,0.9*ymax,Ndata)
    z = randrange(-0.9*zmax,0.9*zmax,Ndata)
    hIIdist, lat, lon = cartesian_to_spherical(x+8.5*u.kpc,y,z)
    fake_data = {'brightness':T,'err':T_err,'lat':lat,'lon':lon}
    mea       = fill_imagine_dataset(data=fake_data)
    
    # Setup the Galactic field models
    cartesian_grid = img.fields.UniformGrid(box=[[-xmax, xmax],
                                                 [-ymax, ymax],
                                                 [-zmax, zmax]],
                                                 resolution = [40,40,10]) # skipping x=y=0
    cre = img.fields.PowerlawCosmicRayElectrons(grid=cartesian_grid,
                                            parameters = {'scale_radius':10*u.kpc,
                                                         'scale_height':1*u.kpc,
                                                         'central_density':1e-5*u.cm**-3,
                                                         'spectral_index':-3})
    
    print("Defining WrappedJF12")
    Bfield1 = WrappedJF12(
        grid = cartesian_grid,
        parameters = {'b_arm_1': .1, 'b_arm_2': 3.0, 'b_arm_3': -.9, 'b_arm_4': -.8, 'b_arm_5': -2.,
                          'b_arm_6': -4.2, 'b_arm_7': .0, 'b_ring': .1, 'h_disk': .4, 'w_disk': .27,
                          'Bn': 1.4, 'Bs': -1.1, 'rn': 9.22, 'rs': 16.7, 'wh': .2, 'z0': 5.3, 'B0_X': 4.6,
                          'Xtheta_const': 49, 'rpc_X': 4.8, 'r0_X': 2.9, })

    Barray  = load_JF12rnd(label=1)
    beta    = 1.0
    Bfield2 = ArrayMagneticField(grid=cartesian_grid,
                            parameters = {'array_field_amplitude':beta,
                                          'array_field':Barray*u.microgauss})

    Btotal = MagneticFieldAdder(grid=cartesian_grid,
                                field_1=WrappedJF12,
                                field_2=ArrayMagneticField,
                                parameters = {  'array_field_amplitude':beta,
                                                'b_arm_1': .1, 'b_arm_2': 3.0, 'b_arm_3': -.9, 'b_arm_4': -.8, 'b_arm_5': -2.,
                                                'b_arm_6': -4.2, 'b_arm_7': .0, 'b_ring': .1, 'h_disk': .4, 'w_disk': .27,
                                                'Bn': 1.4, 'Bs': -1.1, 'rn': 9.22, 'rs': 16.7, 'wh': .2, 'z0': 5.3, 'B0_X': 4.6,
                                                'Xtheta_const': 49, 'rpc_X': 4.8, 'r0_X': 2.9, 
                                                'array_field': Barray*u.microgauss})

    # Setup observing configuration
    observer = np.array([-8.5,0,0])*u.kpc
    dist_err = hIIdist/5
    FB       = get_label_FB(Ndata)
    config = {'grid'    :cartesian_grid,
              'observer':observer,
              'dist':hIIdist,
              'e_dist':dist_err,
              'lat':lat,
              'lon':lon,
              'FB':FB}
    
    # Produce simulated dataset with noise
    mock_data = produce_mock_data(field_list=[cre,Btotal], mea=mea, config=config, noise=0.01)
    sim_data = {'brightness':mock_data,'err':mock_data/10,'lat':config['lat'],'lon':config['lon']}
    sim_mea  = fill_imagine_dataset(sim_data)

    # Setup simulator
    los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)
    
    # Initialize likelihood
    likelihood = img.likelihoods.SimpleLikelihood(sim_mea)    
    
    # Setup field factories and their active parameters
    CRE_factory = img.fields.FieldFactory(field_class = cre, grid=cartesian_grid) 
    
    B_factory   = WrappedJF12Factory(grid=cartesian_grid)
    #B_factory   = img.fields.FieldFactory(
    #    field_class = Btotal,
    #    grid = cartesian_grid,
    #    field_kwargs = {'field_1':WrappedJF12, 'field_2':ArrayMagneticField})
    
    """
    B_factory.active_parameters = ('array_field_amplitude',)
    B_factory.priors = {'array_field_amplitude':img.priors.FlatPrior(xmin=0, xmax=10)}
    
    """
    B_factory.active_parameters = ('b_arm_1',)
    B_factory.priors = {        
    'b_arm_1':img.priors.FlatPrior(xmin=0, xmax=10)
    }
    
    factory_list = [B_factory, CRE_factory]

    # Setup final pipeline
    pipeline = img.pipelines.MultinestPipeline( simulator     = los_simulator,
                                                run_directory = rundir,
                                                factory_list  = factory_list,
                                                likelihood    = likelihood)
    pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}
    
    # Run!
    results = pipeline()
    
    summary = pipeline.posterior_summary
    samples = pipeline.samples
    
    return #samples, summary

start = perf_counter()
JF12pipeline_MagneticFieldAdder()
time_JF12adder = perf_counter()-start

#===========================================================================

# pipeline controller
def JF12pipeline_basic():

    # Remove old pipeline
    os.system("rm -r runs/mockdata/*")
    
    # Produce empty data format
    T     = np.zeros(Ndata)*dunit # placeholder
    T_err = np.zeros(Ndata)*dunit # placeholder
    xmax = 20*u.kpc
    ymax = 20*u.kpc
    zmax =  2*u.kpc
    x = randrange(-0.9*xmax,0.9*xmax,Ndata)
    y = randrange(-0.9*ymax,0.9*ymax,Ndata)
    z = randrange(-0.9*zmax,0.9*zmax,Ndata)
    hIIdist, lat, lon = cartesian_to_spherical(x+8.5*u.kpc,y,z)
    fake_data = {'brightness':T,'err':T_err,'lat':lat,'lon':lon}
    mea       = fill_imagine_dataset(data=fake_data)
    
    # Setup the Galactic field models
    cartesian_grid = img.fields.UniformGrid(box=[[-xmax, xmax],
                                                 [-ymax, ymax],
                                                 [-zmax, zmax]],
                                                 resolution = [40,40,10]) # skipping x=y=0
    cre = img.fields.PowerlawCosmicRayElectrons(grid=cartesian_grid,
                                            parameters = {'scale_radius':10*u.kpc,
                                                         'scale_height':1*u.kpc,
                                                         'central_density':1e-5*u.cm**-3,
                                                         'spectral_index':-3})
    
    Bfield1 = WrappedJF12(
        grid = cartesian_grid,
        parameters = {'b_arm_1': .1, 'b_arm_2': 3.0, 'b_arm_3': -.9, 'b_arm_4': -.8, 'b_arm_5': -2.,
                          'b_arm_6': -4.2, 'b_arm_7': .0, 'b_ring': .1, 'h_disk': .4, 'w_disk': .27,
                          'Bn': 1.4, 'Bs': -1.1, 'rn': 9.22, 'rs': 16.7, 'wh': .2, 'z0': 5.3, 'B0_X': 4.6,
                          'Xtheta_const': 49, 'rpc_X': 4.8, 'r0_X': 2.9, })

    Barray  = load_JF12rnd(label=1)
    beta    = 1.0
    Bfield2 = ArrayMagneticField(grid=cartesian_grid,
                            parameters = {'array_field_amplitude':beta,
                                          'array_field':Barray*u.microgauss})

    # Setup observing configuration
    observer = np.array([-8.5,0,0])*u.kpc
    dist_err = hIIdist/5
    FB       = get_label_FB(Ndata)
    config = {'grid'    :cartesian_grid,
              'observer':observer,
              'dist':hIIdist,
              'e_dist':dist_err,
              'lat':lat,
              'lon':lon,
              'FB':FB}
    
    # Produce simulated dataset with noise
    mock_data = produce_mock_data(field_list=[cre,Bfield1,Bfield2], mea=mea, config=config, noise=0.01)
    sim_data = {'brightness':mock_data,'err':mock_data/10,'lat':config['lat'],'lon':config['lon']}
    sim_mea  = fill_imagine_dataset(sim_data)

    # Setup simulator
    los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)
    
    # Initialize likelihood
    likelihood = img.likelihoods.SimpleLikelihood(sim_mea)    
    
    # Setup field factories and their active parameters
    CRE_factory = img.fields.FieldFactory(field_class = cre, grid=cartesian_grid) 
    
    B_factory1 = WrappedJF12Factory(grid=cartesian_grid)
    B_factory1.active_parameters = ('b_arm_1',)
    B_factory1.priors = {        
    'b_arm_1':img.priors.FlatPrior(xmin=0, xmax=10)
    }
   
    B_factory2 = img.fields.FieldFactory(field_class=Bfield2, grid=cartesian_grid)
    #B_factory2.active_parameters = ('array_field_amplitude',)
    #B_factory2.priors = {'array_field_amplitude':img.priors.FlatPrior(xmin=0, xmax=10)}
    
    factory_list = [B_factory1, B_factory2, CRE_factory]

    # Setup final pipeline
    pipeline = img.pipelines.MultinestPipeline( simulator     = los_simulator,
                                                run_directory = rundir,
                                                factory_list  = factory_list,
                                                likelihood    = likelihood)
    pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}
    
    # Run!
    results = pipeline()
    
    summary = pipeline.posterior_summary
    samples = pipeline.samples
    
    return #samples, summary

start = perf_counter()
JF12pipeline_basic()
time_JF12basic = perf_counter() - start


print("JF12Adder took: {}s".format(time_JF12adder))
print("JF12basic took: {}s".format(time_JF12basic))