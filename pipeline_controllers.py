
#%% Imports and settings
#===================================================================================
# Imagine
import imagine as img
from imagine.simulators.synchrotronlos import SpectralSynchrotronEmissivitySimulator
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12Factory

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
rundir    = 'runs/performance'
figpath   = 'figures/performance/'
fieldpath = 'arrayfields/'
logdir    = 'log/'

# Gobal testing constants
import astropy.units as u
MHz   = 1e6 / u.s
Ndata = 100
observing_frequency = 74*MHz
dunit = u.K/u.kpc
global_dist_error = 0.000
global_brightness_error = 0.01

print("\n")

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
    key = ('average_los_brightness', 0.07400000000000001, 'tab', None)
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

def make_empty_dictionary(scales):
    dictionary = {'scales':scales}
    dictionary['evidence'] = []
    dictionary['time'] = []
    dictionary['evidence_err'] = []
    return dictionary

def save_pipeline_results(pipeline, label, dictionary, time):
    dictionary['summary_{}'.format(label)] = pipeline.posterior_summary
    dictionary['samples_{}'.format(label)] = pipeline.samples
    dictionary['evidence'].append(pipeline.log_evidence)
    dictionary['time'].append(time)
    dictionary['evidence_err'].append(pipeline.log_evidence_err)
    return dictionary

def unpack_samples_and_evidence(dictionary={}):
    evidence  = np.array(dictionary['evidence'])
    time      = np.array(dictionary['time'])
    scales    = dictionary['scales']
    npsamples = ()
    for xvalue in scales: # collect 2D matrix of parameters for each turbscale
        s = []
        for samples in dictionary['samples_{}'.format(xvalue)]:
            no_unit = []
            for samp in samples:
                no_unit.append(samp/samp.unit)
            s.append(no_unit)
        npsamples += (np.array(s),)
    return scales, npsamples, evidence, time

def make_samples_evidence_time_plots(fig, data=(), pnames=(), true_pval=(), xlabel='', title=''):
    scales, samples, evidence, time = data
    mean_samp = []
    for samp in samples: mean_samp.append(np.mean(samp,axis=0))
    mean_samp = np.array(mean_samp)
    gs  = fig.add_gridspec(3, hspace=0, height_ratios= [3,1,1])
    axs = gs.subplots(sharex=True)
    # Top plot: median parameter estimate of all params
    for i,name in enumerate(pnames):
        axs[0].plot(scales, mean_samp[:,i]/true_pval[i], label=name)
    axs[0].set_ylabel("parameter estimate/truth ")
    axs[0].legend()
    # Middle plot: Evidence
    axs[1].plot(scales, evidence)
    axs[1].set_ylabel("log(Z)")
    # Bottem plot: Computation time
    axs[2].plot(scales, time)
    axs[2].set_ylabel("computation time (s)")
    # Correct lables
    axs[2].set_xlabel(xlabel)
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

def run_pipeline(controller_range=[None], controller_type='', Nparams=''):
    # Basic simulation setup
    Bfield, cre, config, mea = produce_basic_setup()
    # Setup field factories and their active parameters
    if Nparams == 'one':
        B_factory   = img.fields.FieldFactory(field_class = Bfield, grid=config['grid'])
        CRE_factory = img.fields.FieldFactory(field_class = cre, grid=config['grid']) 
        B_factory.active_parameters = ('b_arm_2',)
        B_factory.priors = {'b_arm_2':img.priors.FlatPrior(xmin=0, xmax=5)}
    if Nparams == 'three':
        B_factory = WrappedJF12Factory(grid=config['grid'])
        B_factory.active_parameters = ('b_arm_2','h_disk')
        B_factory.priors = {        
        'b_arm_2':img.priors.FlatPrior(xmin=0, xmax=5.0),
        'h_disk':img.priors.FlatPrior(xmin=0, xmax=2)
        }
        CRE_factory = img.fields.FieldFactory(field_class=cre, grid=config['grid']) 
        CRE_factory.active_parameters = ('spectral_index',)
        CRE_factory.priors = {
        'spectral_index':img.priors.FlatPrior(xmin=-5, xmax=-2.1)
        }
    factory_list = [B_factory, CRE_factory]
    # Produce simulated dataset with temperature noise
    mock_data, error = produce_mock_data(field_list=[cre,Bfield], mea=mea, config=config, noise=global_brightness_error)
    sim_data = {'brightness':mock_data,'err':error,'lat':config['lat'],'lon':config['lon']}
    sim_mea  = fill_imagine_dataset(sim_data)
    # Setup simulator
    los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)
    # Initialize likelihood
    likelihood = img.likelihoods.SimpleLikelihood(sim_mea)
    # Default controle parameters{
    default = {'evidence_tolerance': 0.5,
               'n_live_points': 200,
               'sampling_efficiency':0.8}
    # Run pipeline for different controle parameter ranges
    results_dictionary = make_empty_dictionary(scales=controller_range)
    print("Run pipline for controller: \"{}\" in range {}".format(controller_type, controller_range))
    for value in controller_range:
        # Clear previous pipeline
        os.system("rm -r {}/*".format(rundir))
        # Setup pipeline
        pipeline = img.pipelines.MultinestPipeline(
            simulator     = los_simulator, 
            run_directory = rundir,
            factory_list  = factory_list,
            likelihood    = likelihood)
        default.update({controller_type:value})
        pipeline.sampling_controllers = default
        # Run!
        start   = perf_counter()
        results = pipeline()
        elapsed = perf_counter() - start
        results_dictionary = save_pipeline_results(pipeline=pipeline, label=value, dictionary=results_dictionary, time=elapsed)
    return results_dictionary

#%% Plotting routines for parameter inference runs
#===================================================================================

controller_dictionary = {
    'evidence_tolerance' :np.linspace(0.1,0.5,20),
    'n_live_points'      :np.linspace(100,400,20,dtype=int),
    'sampling_efficiency':np.linspace(0.1,0.9,20)}

# Generate samples
def get_samples(controller_range=[None],controller_type=''):
    results_dictionary = run_pipeline(controller_range=controller_range, controller_type=controller_type, Nparams='three')
    np.save(logdir+'samples_controller_{}_threeparam.npy'.format(controller_type), results_dictionary)
#get_samples()

# Plot results
def plot_samples(controller_type=''):
    results_dictionary = np.load(logdir+'samples_controller_{}_threeparam.npy'.format(controller_type), allow_pickle=True).item()
    #print(results_dictionary)
    data = unpack_samples_and_evidence(results_dictionary)
    # Make figure
    plt.close("all")
    figure = plt.figure()
    xlabel = controller_type
    pnames = ('b_arm_2','h_disk','spectral_index')
    true_pval = (3.0, 0.4, -3.0)
    title  = "Pipeline performance overview"
    make_samples_evidence_time_plots( fig=figure,
                                      data=data,
                                      xlabel=xlabel,
                                      pnames=pnames,
                                      true_pval=true_pval,
                                      title=title)
    plt.savefig(figpath+'samples_controller_{}_threeparam.png'.format(controller_type))
    plt.close("all")
#plot_samples(controller_type='evidence_tolerance')

def do_all_controle_parameters():
    for key,item in controller_dictionary.items():
        #get_samples(controller_range=item,controller_type=key)
        plot_samples(controller_type=key)
do_all_controle_parameters()
