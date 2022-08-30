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
    NF = int(0.2*Ndata) # 20% of all measurements
    F  = np.full(shape=(NF), fill_value='F', dtype=str)
    B  = np.full(shape=(Ndata-NF), fill_value='B', dtype=str)
    FB = np.hstack((F,B))
    np.random.shuffle(FB) 
    return FB


# simulator testing
def test_simulator(Ndata, resolution = [30,30,30], keepinit=False):
    """Do simple speedtesting of the los-simulator"""
    
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
                                                 resolution = resolution)
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
    FB       = get_label_FB(Ndata)
    config = {'grid'    :cartesian_grid,
              'observer':observer,
              'dist':hIIdist,
              'e_dist':dist_err,
              'lat':lat,
              'lon':lon,
              'FB':FB}    
    
    # Do simulation
    start      = perf_counter()
    test_sim   = SpectralSynchrotronEmissivitySimulator(measurements=mea, sim_config=config) 
    inittime   = perf_counter() - start
    simulation = test_sim(field_list=[cre,Bfield])
    stop       = perf_counter()
    if keepinit:
        return stop - start, inittime
    else:
        return stop - start

# pipeline controller
def simple_pipeline(noise=0.1,fakemodel=False):
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
    FB       = get_label_FB(Ndata)
    config = {'grid'    :cartesian_grid,
              'observer':observer,
              'dist':hIIdist,
              'e_dist':dist_err,
              'lat':lat,
              'lon':lon,
              'FB':FB}    
    
    # Produce simulated dataset with noise
    mock_data = produce_mock_data(field_list=[cre,Bfield], mea=mea, config=config, noise=noise)
    sim_data = {'brightness':mock_data,'err':mock_data*noise,'lat':config['lat'],'lon':config['lon']}
    sim_mea  = fill_imagine_dataset(sim_data)
    
    # Setup simulator
    los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)
    
    # Initialize likelihood
    likelihood = img.likelihoods.SimpleLikelihood(sim_mea)    
    
    # Setup field factories and their active parameters
    if fakemodel:
        rundir = 'runs/mockdatafake'
        Bfield = img.fields.ConstantMagneticField(
        grid = cartesian_grid,
        parameters={'Bx': 0*u.microgauss,
                    'By': 6*u.microgauss,
                    'Bz': 0*u.microgauss})
    else:
        rundir = 'runs/mockdata'
        
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
#results = JF12spectralhardeningCREprofile_setup()

#%% Results 3.1 Computational performance

def get_ndatavstime():
    """Save time testing results in binary np array"""
    datapoints = np.arange(10,1010,10)
    ctime      = []
    for nd in datapoints:
        ctime.append(test_simulator(Ndata=nd,resolution=[30,30,30]))
    with open(logdir+'ndatavstime.npy', 'wb') as f:
        np.save(f, datapoints)
        np.save(f, np.array(ctime))
#get_ndatavstime()

def plot_ndatavstime():
    with open(logdir+'ndatavstime.npy', 'rb') as f:
        ndata = np.load(f)
        ctime = np.load(f)
    plt.plot(ndata,ctime)
    plt.ylim([0,3])
    plt.title("Computationtime")
    plt.ylabel("time (s)")
    plt.xlabel("number of los")
    plt.savefig(figpath+'ndatavstime.png')
#plot_ndatavstime()


def get_resolutionvstime():
    res   = [10,20,30,40,50,60,70]
    ctime = [] # total computation time of simulation
    itime = [] # initialization time of the simulator
    for r in res:
        tot, it = test_simulator(Ndata=50,resolution=[r,r,r],keepinit=True)
        ctime.append(tot)
        itime.append(it)
    with open(logdir+'resolutionvstime.npy', 'wb') as f:
        np.save(f, np.array(res))
        np.save(f, np.array(ctime))
        np.save(f, np.array(itime))
#get_resolutionvstime()

def plot_resolutionvstime():
    with open(logdir+'resolutionvstime.npy', 'rb') as f:
        res   = np.load(f)
        ctime = np.load(f)
        itime = np.load(f)
    plt.plot(res,ctime, label='total')
    plt.plot(res,itime, label='initialization')
    plt.legend()
    plt.title("Computationtime")
    plt.ylabel("time (s)")
    plt.xlabel("grid resolution")
    plt.savefig(figpath+'resolutionvstime.png')
    plt.close('all')
    plt.plot(res,ctime-itime, label='total-init')
    plt.legend()
    plt.title("Computationtime")
    plt.ylabel("time (s)")
    plt.xlabel("grid resolution")
    plt.savefig(figpath+'resolutionvstime_relavent.png')
#plot_resolutionvstime()



def get_noisevsevidence():
    rel_brightness_error = 10**np.linspace(-2,0,20)
    reallogZ  = []
    reallogZe = []
    fakelogZ  = []
    fakelogZe = []
    for er in rel_brightness_error:        
        os.system("rm -r runs/mockdata/*")
        os.system("rm -r runs/mockdatafake/*")
        real_results = simple_pipeline(noise=er, fakemodel=False)
        reallogZ.append(real_results['logZ'])
        reallogZe.append(real_results['logZerr'])
        fake_results = simple_pipeline(noise=er, fakemodel=True)
        fakelogZ.append(fake_results['logZ'])
        fakelogZe.append(fake_results['logZerr'])
    with open(logdir+'noisevsevidence.npy', 'wb') as f:
        np.save(f, np.array(rel_brightness_error))
        np.save(f, np.array(reallogZ))
        np.save(f, np.array(reallogZe))
        np.save(f, np.array(fakelogZ))
        np.save(f, np.array(fakelogZe))
#get_noisevsevidence()

def plot_noisevsevidence():
    with open(logdir+'noisevsevidence.npy', 'rb') as f:
        Te   = np.load(f)
        rlZ  = np.load(f)
        rlZe = np.load(f)
        flZ  = np.load(f)
        flZe = np.load(f)
    plt.close('all')
    plt.plot(Te, rlZ, label='real model')
    #plt.fill_between(Te, rlZ-rlZe, rlZ+rlZe,color='gray', alpha=0.2) # errors on the evidence are tiny!!
    plt.plot(Te, rlZ-flZ, label='fake model')
    #plt.fill_between(Te, flZ-flZe, flZ+flZe,color='gray', alpha=0.2)
    plt.xscale('log')
    plt.legend()
    plt.title("Brightness temperature noise performance")
    plt.ylabel("Evidence (logZ)")
    plt.xlabel("Relative error")
    plt.savefig(figpath+'noisevsevidence.png')
plot_noisevsevidence()
















