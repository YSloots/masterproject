"""
This script is used to do performance testing of the galactic los synchrotron simulator
and the retrieval of model parameters from simulated datasets.



"""

#%% Imports and settings

import imagine as img
from imagine.simulators.synchrotronlos import SpectralSynchrotronEmissivitySimulator
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12

import numpy as np
from astropy.coordinates import spherical_to_cartesian
from astropy.coordinates import cartesian_to_spherical

rundir  = 'runs/mockdata'

import astropy.units as u
MHz   = 1e6 / u.s

# Gobal testing constants
Ndata = 100
observing_frequency = 90*MHz
dunit = u.K/u.kpc

#%% Code reduction functions and pipeline setup controlers

# code duplication reduction
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

def produce_mock_data(field_list, mea, config):
    """Runs the simulator once to produce a simulated dataset"""
    test_sim   = SpectralSynchrotronEmissivitySimulator(measurements=mea, sim_config=config) 
    simulation = test_sim(field_list)
    key = ('average_los_brightness', 0.09000000000000001, 'tab', None)
    sim_brightness = simulation[key].data[0] * simulation[key].unit
    sim_brightness += np.random.normal(loc=0, scale=0.01*sim_brightness, size=Ndata)*simulation[key].unit
    return sim_brightness

def randrange(minvalue,maxvalue,Nvalues):
    """Returns uniform random values bewteen min and max"""
    return (maxvalue-minvalue)*np.random.rand(Nvalues)+minvalue

def get_label_FB():
    NF = int(0.2*Ndata) # 20% of all measurements
    F  = np.full(shape=(NF), fill_value='F', dtype=str)
    B  = np.full(shape=(Ndata-NF), fill_value='B', dtype=str)
    FB = np.hstack((F,B))
    np.random.shuffle(FB) 
    return FB

# pipeline controller
def simple_pipeline():
    """
    Test retrieval of correct cre spectral index for constant GMF and CRE density
    """
    
    # Produce empty data format  
    T     = np.zeros(Ndata)*dunit 
    T_err = np.zeros(Ndata)*dunit
    lat   = 90*np.linspace(-1,1,Ndata)*u.deg
    lon   = 360*np.linspace(0,1,Ndata)*u.deg*300
    fake_data = {'brightness':T,'err':T_err,'lat':lat,'lon':lon}
    mea       = fill_imagine_dataset(data=fake_data)
              
    # Setup the Galactic field models
    box_size       = 15*u.kpc
    cartesian_grid = img.fields.UniformGrid(box=[[-box_size, box_size],
                                                 [-box_size, box_size],
                                                 [-box_size, box_size]],
                                                 resolution = [30,30,30])
    cre = img.fields.ConstantCosmicRayElectrons(grid=cartesian_grid,
                                            parameters={'density':1.0e-7*u.cm**-3,'spectral_index':-3.0})
    Bfield = img.fields.ConstantMagneticField(
        grid = cartesian_grid,
        parameters={'Bx': 6*u.microgauss,
                    'By': 0*u.microgauss,
                    'Bz': 0*u.microgauss})
    
    # Setup observing configuration
    observer = np.array([0,0,0])*u.kpc
    hIIdist  = (box_size-2*u.kpc)*np.random.rand(Ndata) + 1*u.kpc # uniform between [1, max-1] kpc
    dist_err = hIIdist/10
    FB       = get_label_FB()
    config = {'grid'    :cartesian_grid,
              'observer':observer,
              'dist':hIIdist,
              'e_dist':dist_err,
              'lat':lat,
              'lon':lon,
              'FB':FB}    
    
    # Produce simulated dataset with noise
    mock_data = produce_mock_data(field_list=[cre,Bfield], mea=mea, config=config)
    sim_data = {'brightness':mock_data,'err':mock_data/10,'lat':config['lat'],'lon':config['lon']}
    sim_mea  = fill_imagine_dataset(sim_data)
    
    # Setup simulator
    los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)
    
    # Initialize likelihood
    likelihood = img.likelihoods.SimpleLikelihood(sim_mea)    
    
    # Setup field factories and their active parameters
    B_factory  = img.fields.FieldFactory(field_class = Bfield, grid=config['grid'])
    CRE_factory = img.fields.FieldFactory(field_class = cre, grid=config['grid']) 
    CRE_factory.active_parameters=('spectral_index',)
    CRE_factory.priors = {'spectral_index':img.priors.FlatPrior(xmin=-4, xmax=-2.1)}
    factory_list = [B_factory, CRE_factory]
    
    # Setup final pipeline
    pipeline = img.pipelines.MultinestPipeline( simulator     = los_simulator,
                                                run_directory = rundir,
                                                factory_list  = factory_list,
                                                likelihood    = likelihood)
    pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}
    
    # Run!
    results = pipeline()    
    
    return results


# pipeline controller
def JF12constindexCREprofile_setup():
    """
    Test retrieval of correct cre spectral index for regular JF12 magnetic field
    and a exponential number density CRE model with constant spectral index.
    """
    
    # Produce empty data format
    T     = np.zeros(Ndata)*dunit # placeholder
    T_err = np.zeros(Ndata)*dunit # placeholder
    xmax = 15*u.kpc
    ymax = 15*u.kpc
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
                                                 resolution = [30,30,30]) # skipping x=y=0
    cre = img.fields.PowerlawCosmicRayElectrons(grid=cartesian_grid,
                                            parameters = {'scale_radius':10*u.kpc,
                                                         'scale_height':1*u.kpc,
                                                         'central_density':1e-5*u.cm**-3,
                                                         'spectral_index':-3})
    Bfield = WrappedJF12(grid=cartesian_grid)

    # Setup observing configuration
    observer = np.array([-8.5,0,0])*u.kpc
    dist_err = hIIdist/5
    FB       = get_label_FB()
    config = {'grid'    :cartesian_grid,
              'observer':observer,
              'dist':hIIdist,
              'e_dist':dist_err,
              'lat':lat,
              'lon':lon,
              'FB':FB}
    
    # Produce simulated dataset with noise
    mock_data = produce_mock_data(field_list=[cre,Bfield], mea=mea, config=config)
    sim_data = {'brightness':mock_data,'err':mock_data/10,'lat':config['lat'],'lon':config['lon']}
    sim_mea  = fill_imagine_dataset(sim_data)

    # Setup simulator
    los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)
    
    # Initialize likelihood
    likelihood = img.likelihoods.SimpleLikelihood(sim_mea)    
    
    # Setup field factories and their active parameters
    B_factory   = img.fields.FieldFactory(field_class = Bfield, grid=config['grid'])
    CRE_factory = img.fields.FieldFactory(field_class = cre, grid=config['grid']) 
    CRE_factory.active_parameters = ('spectral_index',)
    CRE_factory.priors = {'spectral_index':img.priors.FlatPrior(xmin=-4, xmax=-2.1)}
    factory_list = [B_factory, CRE_factory]
    
    # Setup final pipeline
    pipeline = img.pipelines.MultinestPipeline( simulator     = los_simulator,
                                                run_directory = rundir,
                                                factory_list  = factory_list,
                                                likelihood    = likelihood)
    pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}
    
    # Run!
    results = pipeline()    
    
    return results


# pipeline controller
def JF12spectralhardeningCREprofile_setup():
    """
    Test retrieval of correct CRE hardening slope for a linear hardening model assuming
    the JF12 regular magnetic field.
    """
    
    # Produce empty data format
    T     = np.zeros(Ndata)*dunit # placeholder
    T_err = np.zeros(Ndata)*dunit # placeholder
    xmax = 15*u.kpc
    ymax = 15*u.kpc
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
                                                 resolution = [30,30,30]) # skipping x=y=0
    cre_num = img.fields.CRENumberDensity(grid= cartesian_grid,
                                      parameters={'scale_radius':10*u.kpc,
                                                  'scale_height':1*u.kpc,
                                                  'central_density':1e-5*u.cm**-3})
    cre_alpha = img.fields.SpectralIndexLinearVerticalProfile(
        grid=cartesian_grid,
        parameters={'soft_index':-4, 'hard_index':-2.5, 'slope':1*u.kpc**-1})
    Bfield = WrappedJF12(grid=cartesian_grid)

    
    # Setup observing configuration
    observer = np.array([-8.5,0,0])*u.kpc
    dist_err = hIIdist/5
    FB       = get_label_FB()
    config = {'grid'    :cartesian_grid,
              'observer':observer,
              'dist':hIIdist,
              'e_dist':dist_err,
              'lat':lat,
              'lon':lon,
              'FB':FB}
    
    # Produce simulated dataset with noise
    mock_data = produce_mock_data(field_list=[cre_num,cre_alpha,Bfield], mea=mea, config=config)
    sim_data = {'brightness':mock_data,'err':mock_data/10,'lat':config['lat'],'lon':config['lon']}
    sim_mea  = fill_imagine_dataset(sim_data)

    # Setup simulator
    los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)
    
    # Initialize likelihood
    likelihood = img.likelihoods.SimpleLikelihood(sim_mea)    
    
    # Setup field factories and their active parameters
    B_factory       = img.fields.FieldFactory(field_class = Bfield, grid=config['grid'])
    cre_num_factory = img.fields.FieldFactory(field_class = cre_num, grid=config['grid'])
    alpha_factory   = img.fields.FieldFactory(field_class = cre_alpha, grid=config['grid'])
    #alpha_factory.active_parameters=('slope',)
    #alpha_factory.priors = {'slope':img.priors.FlatPrior(xmin=0*u.kpc**-1, xmax=5*u.kpc**-1)}
    alpha_factory.active_parameters=(('slope','soft_index','hard_index'))
    alpha_factory.priors = {'slope':img.priors.FlatPrior(xmin=0*u.kpc**-1, xmax=5*u.kpc**-1),                            
                            'soft_index':img.priors.FlatPrior(xmin=-5, xmax=-3),
                            'hard_index':img.priors.FlatPrior(xmin=-3, xmax=-2.1)}
    factory_list = [cre_num_factory, alpha_factory, B_factory]  
    
    # Setup final pipeline
    pipeline = img.pipelines.MultinestPipeline( simulator     = los_simulator,
                                                run_directory = rundir,
                                                factory_list  = factory_list,
                                                likelihood    = likelihood)
    pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}
    
    # Run!
    results = pipeline()    
    
    return results    





#%% Choose which setup you want to run

#results = simple_pipeline()
#results = JF12constindexCREprofile_setup()
results = JF12spectralhardeningCREprofile_setup()












