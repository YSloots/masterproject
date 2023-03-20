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
rundir    = 'runs/characterize_data'
figpath   = 'figures/characterize_data/'
fieldpath = 'arrayfields/'
logdir    = 'log/'

# Gobal testing constants
import astropy.units as u
MHz   = 1e6 / u.s
Ndata = 100
observing_frequency = 74*MHz
dunit = u.K
global_dist_error = 0.000
global_brightness_error = 0.01
key = ('los_brightness_temperature', 0.07400000000000001, 'tab', None)

print("\n")

#%% Code reduction functions
#===================================================================================

def fill_imagine_dataset(data):
    fake_dset = img.observables.TabularDataset(data,
                                               name='los_brightness_temperature',
                                               frequency=observing_frequency,
                                               units=dunit,
                                               data_col='brightness',
                                               err_col='err',
                                               lat_col='lat',
                                               lon_col='lon')
    return img.observables.Measurements(fake_dset)

def produce_mock_data(field_list, mea, config, noise=global_brightness_error):
    """
    Runs the simulator once to produce a simulated dataset
    - assume we know the exact positions for the HII regions
    - add some gaussian noise to the brightness temperate with some relative error
    """
    # Set distance error to None (zero) when computing the los for mock data
    if config['e_dist'] is None:
        mock_config = config
    else:
        mock_config = config.copy()
        mock_config['e_dist'] = None
    test_sim   = SpectralSynchrotronEmissivitySimulator(measurements=mea, sim_config=mock_config) 
    simulation = test_sim(field_list)
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
    NF = int(0.2*Ndata) # 20% of all measurements are front measurements
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

def make_empty_dictionary(scales):
    dictionary = {'scales':scales}
    dictionary['evidence'] = []
    dictionary['evidence_err'] = []
    return dictionary

def save_pipeline_results(pipeline, label, dictionary):
    dictionary['summary_{}'.format(label)] = pipeline.posterior_summary
    dictionary['samples_{}'.format(label)] = pipeline.samples
    dictionary['evidence'].append(pipeline.log_evidence)
    dictionary['evidence_err'].append(pipeline.log_evidence_err)
    return dictionary

def unpack_samples_and_evidence(dictionary={}):
    evidence = np.array(dictionary['evidence'])
    scales   = dictionary['scales']
    meany = np.empty(len(scales))
    stdy  = np.empty(len(scales))
    for i,xvalue in enumerate(scales):
        s = []
        for samples in dictionary['samples_{}'.format(xvalue)]:
            s.append(list(samples)[0])
        meany[i] = np.mean(s,axis=0)
        stdy[i]  = np.std(s,axis=0)
    return scales, meany, stdy, evidence

def make_samples_and_evidence_plot(fig, data=(), xlabel='', ylabel='', title=''):
    x, meany, stdy, evidence = data
    gs  = fig.add_gridspec(2, hspace=0, height_ratios= [3, 1])
    axs = gs.subplots(sharex=True)
    # Top plot: the sampled parameter
    axs[0].plot(x,meany)
    axs[0].fill_between(x, meany-stdy, meany+stdy, alpha=0.2)
    axs[0].set_ylabel(ylabel,fontsize=15)
    # Bottem plot: the evidence
    pos = np.where(evidence>=0)
    signswitch = np.where(evidence*np.roll(evidence,1)<0)[0][1:]
    if np.size(pos) != 0: 
        axs[1].plot(x[pos], evidence[pos],c='tab:blue')
        if np.size(signswitch) != 0:
            print("Plotting transition pieces")
            for i in signswitch: 
                axs[1].plot(x[i-1:i+1],evidence[i-1:i+1],c='tab:blue')
    else:
        print("No positive evidence, plotting full evidence anyway")
        axs[1].plot(x,evidence,c='tab:blue')
    axs[1].set_ylabel("log(Z)",fontsize=15)
    # Correct lables
    axs[1].set_xlabel(xlabel,fontsize=15)
    plt.suptitle(title,fontsize=20)
    plt.tight_layout()
    fig.align_ylabels(axs)

#%% Pipeline controllers
#===================================================================================

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

#==========================

def scale_brightness_error(rel_error = [global_brightness_error]):
    # Basic simulation setup
    Bfield, cre, config, mea = produce_basic_setup()
    # Setup field factories and their active parameters
    B_factory   = img.fields.FieldFactory(field_class = Bfield, grid=config['grid'])
    CRE_factory = img.fields.FieldFactory(field_class = cre, grid=config['grid']) 
    B_factory.active_parameters = ('b_arm_2',)
    B_factory.priors = {'b_arm_2':img.priors.FlatPrior(xmin=0, xmax=10)}
    factory_list = [B_factory, CRE_factory]
    # Simulate data for different noise scales and save results
    results_dictionary = make_empty_dictionary(scales=rel_error)
    for err in rel_error:
        # Clear previous pipeline
        os.system("rm -r {}/*".format(rundir))
        # Produce simulated dataset with temperature noise
        mock_data, error = produce_mock_data(field_list=[cre,Bfield], mea=mea, config=config, noise=err)
        sim_data = {'brightness':mock_data,'err':error,'lat':config['lat'],'lon':config['lon']}
        sim_mea  = fill_imagine_dataset(sim_data)
        # Setup simulator
        los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)
        # Initialize likelihood
        likelihood = img.likelihoods.SimpleLikelihood(sim_mea)    
        # Setup final pipeline
        pipeline = img.pipelines.MultinestPipeline( simulator     = los_simulator, # depends on e_TB, takes sim_mea, which has e_TB
                                                    run_directory = rundir,
                                                    factory_list  = factory_list,
                                                    likelihood    = likelihood) # depends on e_TB, takes sim_mea, which has e_TB
        pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}
        # Run!
        results = pipeline()
        results_dictionary = save_pipeline_results(pipeline=pipeline, label=err, dictionary=results_dictionary)
    return results_dictionary

#==========================

def scale_distance_error(rel_error = [global_dist_error]):
    # Basic simulation setup
    Bfield, cre, config, mea = produce_basic_setup()
    # Setup field factories and their active parameters
    B_factory   = img.fields.FieldFactory(field_class = Bfield, grid=config['grid'])
    CRE_factory = img.fields.FieldFactory(field_class = cre, grid=config['grid']) 
    B_factory.active_parameters = ('b_arm_2',)
    B_factory.priors = {'b_arm_2':img.priors.FlatPrior(xmin=2, xmax=4)}
    factory_list = [B_factory, CRE_factory]
    # Produce simulated dataset with temperature noise
    mock_data, error = produce_mock_data(field_list=[cre,Bfield], mea=mea, config=config)
    sim_data = {'brightness':mock_data,'err':error,'lat':config['lat'],'lon':config['lon']}
    sim_mea  = fill_imagine_dataset(sim_data)
    # Simulate data for different noise scales and save results
    results_dictionary = make_empty_dictionary(scales=rel_error)
    for err in rel_error:
        # Clear previous pipeline
        os.system("rm -r {}/*".format(rundir))
        # Change distance error in config
        config['e_dist'] = err*config['dist']
        # Setup simulator
        los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)
        # Initialize likelihood
        likelihood = img.likelihoods.SimpleLikelihood(sim_mea)
        # Setup final pipeline
        pipeline = img.pipelines.MultinestPipeline( simulator     = los_simulator, # depends on e_dist, takes config, which has e_dist
                                                    run_directory = rundir,
                                                    factory_list  = factory_list,
                                                    likelihood    = likelihood) # depends on e_dist, takes sim_mea, which takes config and mock_data, which have e_dist
        pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}
        # Run!
        results = pipeline()
        results_dictionary = save_pipeline_results(pipeline=pipeline, label=err, dictionary=results_dictionary)
    return results_dictionary

#==========================

def scale_zdist(sigma_z = [0.03]):
    # Basic simulation setup
    Bfield, cre, config, mea = produce_basic_setup()
    # Fake measurements (and config) will be overwritten
    T     = np.zeros(Ndata)*dunit # placeholder
    T_err = np.zeros(Ndata)*dunit # placeholder
    xmax = 20*u.kpc
    ymax = 20*u.kpc
    zmax =  2*u.kpc
    x = randrange(-0.9*xmax,0.9*xmax,Ndata, seed=10) # constant seed
    y = randrange(-0.9*ymax,0.9*ymax,Ndata, seed=20) # constant seed
    #z = randrange(-0.9*zmax,0.9*zmax,Ndata, seed=30) # constant seed
    # Setup field factories and their active parameters
    B_factory   = img.fields.FieldFactory(field_class = Bfield, grid=config['grid'])
    CRE_factory = img.fields.FieldFactory(field_class = cre, grid=config['grid']) 
    B_factory.active_parameters = ('h_disk',)
    B_factory.priors = {'h_disk':img.priors.FlatPrior(xmin=0, xmax=2)}
    factory_list = [B_factory, CRE_factory]
    # Simulate data for different noise scales and save results
    results_dictionary = make_empty_dictionary(scales=sigma_z)
    for stdz in sigma_z:
        # Clear previous pipeline
        os.system("rm -r {}/*".format(rundir))
        # Produce empty data format
        z = truncnorm(a=-zmax/stdz, b=zmax/stdz, scale=stdz/u.kpc).rvs(Ndata)*u.kpc # seed allowed to change
        hIIdist, lat, lon = cartesian_to_spherical(x+8.5*u.kpc,y,z)
        fake_data = {'brightness':T,'err':T_err,'lat':lat,'lon':lon}
        mea       = fill_imagine_dataset(data=fake_data)
        # Setup new observing configuration
        config['dist']   = hIIdist
        config['e_dist'] = global_dist_error * hIIdist
        config['lat']    = lat
        config['lon']    = lon
        # Produce simulated dataset with temperature noise
        mock_data, error = produce_mock_data(field_list=[cre,Bfield], mea=mea, config=config)
        sim_data = {'brightness':mock_data,'err':error,'lat':config['lat'],'lon':config['lon']}
        sim_mea  = fill_imagine_dataset(sim_data)
        # Setup simulator
        los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)
        # Initialize likelihood
        likelihood = img.likelihoods.SimpleLikelihood(sim_mea)
        # Setup final pipeline
        pipeline = img.pipelines.MultinestPipeline( simulator     = los_simulator, # depends on zdist, takes config, which has z position
                                                    run_directory = rundir,
                                                    factory_list  = factory_list,
                                                    likelihood    = likelihood) # depends on zdist, takes sim_mea, which depends on z position
        pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}
        # Run!
        results = pipeline()
        results_dictionary = save_pipeline_results(pipeline=pipeline, label=stdz, dictionary=results_dictionary)
    return results_dictionary

#scale_zdist() # do a run withouth gaussian z but with uniform basic config z
#scale_brightness_error() # do a single run with global brightness error and 


#%% Results 4.4 Sensitivity to data characteristics
#=======================================================================================

# Generate brightness samples
def get_samples_relbrightness_error():
    """Do a sampling of Barm2 with JF12+ConstantAlphaCRE for different e_Te"""
    error_scale        = np.linspace(0.01,1,20)
    #error_scale = np.linspace(0.01, 0.1, 10)
    results_dictionary = scale_brightness_error(rel_error=error_scale)
    np.save(logdir+'samples_relbrightness_err.npy', results_dictionary)
#get_samples_relbrightness_error()

# Plot results
def plot_samples_relbrightness_error():
    results_dictionary = np.load(logdir+'samples_relbrightness_err.npy', allow_pickle=True).item()
    data = unpack_samples_and_evidence(results_dictionary)
    # Make figure
    plt.close("all")
    figure = plt.figure()
    xlabel = "Relative brightness error (eT/T)"
    ylabel = r"$B_{arm2}$"+"(\N{GREEK SMALL LETTER MU}G)"
    title  = r"Nested sampling results $B_{arm2}$"
    make_samples_and_evidence_plot( fig=figure,
                                    data=data,
                                    xlabel=xlabel,
                                    ylabel=ylabel,
                                    title=title)
    plt.savefig(figpath+'samples_relbrightness_error.png')
    plt.close("all")
plot_samples_relbrightness_error()

#==========================

# Generate distance samples
def get_samples_reldistance_error():
    """Do a sampling of Barm2 with JF12+ConstantAlphaCRE for different e_Te"""
    error_scale        = np.linspace(0.000,0.0015,20)
    results_dictionary = scale_distance_error(rel_error=error_scale)
    np.save(logdir+'samples_reldistance_err.npy', results_dictionary)
#get_samples_reldistance_error()

# Plot results
def plot_samples_reldistance_error():
    results_dictionary = np.load(logdir+'samples_reldistance_err.npy', allow_pickle=True).item()
    data = unpack_samples_and_evidence(results_dictionary)
    # Make figure
    plt.close("all")
    figure = plt.figure()
    xlabel = "Relative distance error  (eD/D)"
    ylabel = r"$B_{arm2}$"+"(\N{GREEK SMALL LETTER MU}G)"
    title  = r"Nested sampling results $B_{arm2}$"
    make_samples_and_evidence_plot( fig=figure,
                                    data=data,
                                    xlabel=xlabel,
                                    ylabel=ylabel,
                                    title=title)
    plt.savefig(figpath+'samples_reldistance_error.png')
    plt.close("all")
plot_samples_reldistance_error()

#==========================

# Generate zdist samples
def get_samples_zdist():
    """Do a sampling of scalehight z0 with JF12+ConstantAlphaCRE for different zdist"""
    sigma_z            = np.linspace(0.01,0.1,20) #kpc
    results_dictionary = scale_zdist(sigma_z=sigma_z)
    np.save(logdir+'samples_zdist.npy', results_dictionary)
#get_samples_zdist()

# Plot results
def plot_samples_zdist():
    results_dictionary = np.load(logdir+'samples_zdist.npy', allow_pickle=True).item()
    data = unpack_samples_and_evidence(results_dictionary)
    # Make figure
    plt.close("all")
    figure = plt.figure()
    xlabel = "\N{GREEK SMALL LETTER SIGMA}"+r"$_{z}$ (kpc)"
    ylabel = r"$h_{disk}$ (kpc)"
    title  = r"Nested sampling results $h_{disk}$"
    make_samples_and_evidence_plot( fig=figure,
                                    data=data,
                                    xlabel=xlabel,
                                    ylabel=ylabel,
                                    title=title)
    plt.savefig(figpath+'samples_zdist_uniform.png')
    plt.close("all")
plot_samples_zdist()

#==========================















#%% Development Code (Probably will go unused)
#=======================================================================================





# plotting routine for different color when evidence<0
def plot_samples_reldistance_error():
    results_dictionary = np.load(logdir+'samples_reldistance_err.npy', allow_pickle=True).item()
    rel_error = results_dictionary['scales']
    meany = np.empty(len(rel_error))
    stdy  = np.empty(len(rel_error))
    for i,err in enumerate(rel_error):
        y = []
        for value in results_dictionary['samples_{}'.format(err)]:
            y.append(list(value)[0])
        meany[i] = np.mean(y,axis=0)
        stdy[i]  = np.std(y,axis=0)
        # Make figure
    plt.close("all")
    fig = plt.figure()
    gs  = fig.add_gridspec(2, hspace=0, height_ratios= [3, 1])
    axs = gs.subplots(sharex=True)
    # Top plot: the sampled parameter
    axs[0].plot(rel_error,meany)
    axs[0].fill_between(rel_error, meany-stdy, meany+stdy, alpha=0.2)
    axs[0].set_ylabel('magnetic field amplitude B_arm2')
    # Bottem plot: the evidence
    evidence   = np.array(results_dictionary['evidence'])
    e_evidence = np.array(results_dictionary['evidence_err'])
    maxe = np.max(evidence)
    mine = np.min(evidence)
    evidence_ticks = [maxe,maxe-(maxe-mine)/2,mine]
    pos = np.where(evidence>=0)
    neg = np.where(evidence<=0)
    axs[1].set_yticks(evidence_ticks)
    axs[1].plot(rel_error[pos], evidence[pos],c='blue')
    axs[1].plot(rel_error[neg], evidence[neg],c='red')
    cswitch = np.where(evidence*np.roll(evidence,1)<0)[0][1:]
    for i in cswitch: axs[1].plot(rel_error[i-1:i+1],evidence[i-1:i+1],c='red')
    #axs[1].fill_between(rel_error, evidence-e_evidence, evidence+e_evidence, alpha=0.2)
    axs[1].set_ylabel("evidence")
    # Correct lables
    axs[1].set_xlabel('relative distance error  (eD/D)')
    plt.suptitle('Nested sampling result B_arm2',fontsize=20)
    plt.tight_layout()
    fig.align_ylabels(axs)
    plt.savefig(figpath+'samples_reldistance_error.png')
