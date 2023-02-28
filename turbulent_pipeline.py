"""
This script is used to do performance testing of the galactic los synchrotron simulator
and the retrieval of model parameters from simulated datasets.



"""

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
rundir    = 'runs/turbulent'
figpath   = 'figures/turbulent_pipeline/'
fieldpath = 'arrayfields/'
logdir    = 'log/'

# Gobal testing constants
import astropy.units as u
MHz   = 1e6 / u.s
GHz   = 1e9 / u.s
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

def load_turbulent_field(fname, shape=(40,40,10,3)):
    print("Loading: "+fieldpath+fname)
    arr = np.load(fieldpath+fname, allow_pickle=True)
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
    evidence  = np.array(dictionary['evidence'])
    scales    = dictionary['scales']
    npsamples = ()
    for xvalue in scales: # collect 2D matrix of parameters for each turbscale
        s = []
        for samples in dictionary['samples_{}'.format(xvalue)]:
            s.append(list(samples))
        npsamples += (np.array(s),)
    return scales, npsamples, evidence



#%% Pipeline setup
#===================================================================================

def turbulentJF12CREprofile_setup(turbscale, Brnd_label):

    # Produce empty data format
    T     = np.zeros(Ndata)*dunit # placeholder
    T_err = np.zeros(Ndata)*dunit # placeholder
    xmax = 20*u.kpc
    ymax = 20*u.kpc
    zmax =  2*u.kpc
    x = randrange(-0.9*xmax,0.9*xmax,Ndata, seed=10)
    y = randrange(-0.9*ymax,0.9*ymax,Ndata, seed=20)
    z = randrange(-0.9*zmax,0.9*zmax,Ndata, seed=30)
    hIIdist, lat, lon = cartesian_to_spherical(x+8.5*u.kpc,y,z)
    fake_data = {'brightness':T,'err':T_err,'lat':lat,'lon':lon}
    mea       = fill_imagine_dataset(data=fake_data)
    
    # Setup the Galactic field models
    cartesian_grid = img.fields.UniformGrid(box=[[-xmax, xmax],
                                                 [-ymax, ymax],
                                                 [-zmax, zmax]],
                                                 resolution = [40,40,10]) # skipping x=y=0
    # Setup observing configuration                                          
    observer = np.array([-8.5,0,0])*u.kpc
    dist_err = hIIdist*global_dist_error
    FB       = get_label_FB(Ndata)
    config = {'grid'    :cartesian_grid,
              'observer':observer,
              'dist':hIIdist,
              'e_dist':dist_err,
              'lat':lat,
              'lon':lon,
              'FB':FB}
    
    cre = img.fields.PowerlawCosmicRayElectrons(grid=cartesian_grid,
                                            parameters = {'scale_radius':10*u.kpc,
                                                         'scale_height':1*u.kpc,
                                                         'central_density':1e-5*u.cm**-3,
                                                         'spectral_index':-3})

    # Do sampling runs with different simulated datasets
    results_dictionary = make_empty_dictionary(scales=turbscale)
    for beta in turbscale:
        # Clear previous pipeline
        os.system("rm -r {}/*".format(rundir))
        # Setup magnetic field
        fname  = 'turbulent_nonuniform{}.npy'.format(Brnd_label)
        Barray = load_turbulent_field(fname=fname)
        Btotal = MagneticFieldAdder(grid=cartesian_grid,
                                    field_1=WrappedJF12,
                                    field_2=ArrayMagneticField,
                                    parameters = {  'b_arm_1': .1, 'b_arm_2': 3.0, 'b_arm_3': -.9, 'b_arm_4': -.8, 'b_arm_5': -2.,
                                                    'b_arm_6': -4.2, 'b_arm_7': .0, 'b_ring': .1, 'h_disk': .4, 'w_disk': .27,
                                                    'Bn': 1.4, 'Bs': -1.1, 'rn': 9.22, 'rs': 16.7, 'wh': .2, 'z0': 5.3, 'B0_X': 4.6,
                                                    'Xtheta_const': 49, 'rpc_X': 4.8, 'r0_X': 2.9, 
                                                    'array_field_amplitude':beta, 'array_field': Barray*u.microgauss})

        # Produce simulated dataset with noise
        mock_data, error = produce_mock_data(field_list=[cre,Btotal], mea=mea, config=config, noise=global_brightness_error)
        sim_data = {'brightness':mock_data,'err':error,'lat':config['lat'],'lon':config['lon']}
        sim_mea  = fill_imagine_dataset(sim_data)

        # Setup simulator
        los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)
        
        # Initialize likelihood
        likelihood = img.likelihoods.SimpleLikelihood(sim_mea)    
        
        # Setup field factories and their active parameters
        B_factory = WrappedJF12Factory(grid=cartesian_grid)
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
        
        # Setup final pipeline
        pipeline = img.pipelines.MultinestPipeline( simulator     = los_simulator,
                                                    run_directory = rundir,
                                                    factory_list  = factory_list,
                                                    likelihood    = likelihood)
        pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}
        
        # Run!
        results = pipeline()
        results_dictionary = save_pipeline_results(pipeline=pipeline, label=beta, dictionary=results_dictionary)
    return results_dictionary

import seaborn as sns
import pandas as pd

def plot_corner_and_evidence(fname,data,colnames):
    x, samp_arrays, evidence = data
    # Make cornerplot
    plt.close("all")
    samp = samp_arrays[0]
    df   = pd.DataFrame(data=samp, columns=colnames)
    fig  = sns.pairplot(data=df, corner=True, kind='kde')
    plt.title("Turbulent scale Brms = {}".format(x[0]))
    plt.savefig(figpath+fname+'_pairplot{}.png'.format(x[0]))
    plt.close("all")
    samp = samp_arrays[-1]
    df   = pd.DataFrame(data=samp, columns=colnames)
    fig  = sns.pairplot(data=df, corner=True, kind='kde')
    plt.title("Turbulent scale Brms = {}".format(x[-1]))
    plt.savefig(figpath+fname+'_pairplot{}.png'.format(x[-1]))
    plt.close("all")
    plt.plot(x, evidence)
    plt.savefig(figpath+fname+'_evidence.png')
    plt.close("all")

#%% Plotting routines for parameter inference runs
#===================================================================================


# Generate samples
def get_samples_turbulence(Brnd_label):
    turbulent_scale    = np.linspace(0,0.5,20)
    results_dictionary = turbulentJF12CREprofile_setup(turbscale=turbulent_scale,
                                                       Brnd_label=Brnd_label)
    np.save(logdir+'samples_turbulence{}.npy'.format(Brnd_label), results_dictionary)
#get_samples_turbulence(Brnd_label=1)

# Plot results
def plot_samples_turbulence(Brnd_label):
    results_dictionary = np.load(logdir+'samples_turbulence{}.npy'.format(Brnd_label), allow_pickle=True).item()
    data = unpack_samples_and_evidence(results_dictionary)
    # Make figure
    fname = 'samples_turbulence{}'.format(Brnd_label)
    pnames = ('b_arm_2','h_disk','spectral_index')
    plot_corner_and_evidence(fname    = fname,
                             data     = data,
                             colnames = pnames)
plot_samples_turbulence(Brnd_label=1)

def do_all_turbulent_models(labels = [1,2,3,4,5]):
    for label in labels:
        #get_samples_turbulence(Brnd_label=label)
        plot_samples_turbulence(Brnd_label=label)
#do_all_turbulent_models()

def plot_all_evidences(labels=[1,2,3,4,5]):
    plt.close("all")
    for label in labels:
        results_dictionary = np.load(logdir+'samples_turbulence{}.npy'.format(label), allow_pickle=True).item()
        scales, dummy, evidence = unpack_samples_and_evidence(results_dictionary)
        plt.plot(scales, evidence, label="Brnd {}".format(label))
    plt.title("The effect of scaling the turbulent GMF")
    plt.ylabel("Evidence log(Z)")
    plt.xlabel("Turbulent Brms (muG)")
    plt.legend(title="Turbulent instance")
    plt.subplots_adjust(bottom=.25, left=.25)
    plt.tight_layout()
    plt.savefig(figpath+"evidence_turbulent_instances.png")
    plt.close("all")
    
#plot_all_evidences()