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
from astropy.io import fits
import seaborn as sns
import pandas as pd

# Directory paths
rundir    = 'runs/hiidata'
figpath   = 'figures/data_pipeline/'
fieldpath = 'arrayfields/'
logdir    = 'log/'
datapath = 'data/'

# Gobal constants
import astropy.units as u
MHz   = 1e6 / u.s
GHz   = 1e9 / u.s
pi    = np.pi


print("\n")

#%% Load HII data set
#===================================================================================

with fits.open(datapath+'HII_LOS.fit') as hdul:
    hiidata = hdul[1].data
head = ['GLON','GLAT','Dist','e_Dist','T','e_T','n_T']

# Data definitions
observing_frequency = 74*MHz
dunit = u.K

# Get data columns
T        = hiidata['T']*dunit 
T_err    = hiidata['e_T']*dunit
FB       = hiidata['n_T']
hiidist  = hiidata['Dist']*u.kpc
dist_err = hiidata['e_Dist']*u.kpc
lat      = hiidata['GLAT']*2*pi/360 * u.rad
lon      = hiidata['GLON']*2*pi/360 * u.rad


# Format imagine data set
data    = {'brightness':T,'err':T_err,'lat':lat,'lon':lon}
imgdata = img.observables.TabularDataset(data,
                                        name='los_brightness_temperature',
                                        frequency=observing_frequency,
                                        units=dunit,
                                        data_col='brightness',
                                        err_col='err',
                                        lat_col='lat',
                                        lon_col='lon')
mea = img.observables.Measurements(imgdata)

#%% Pipeline setup
#===================================================================================
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

# pipeline controller
def run_hiidata_pipeline(rel_error = [0.0], spectral_type=None):
    """
    Retrieve CRE model parameters from data

    note: rel_error is now only used as a dummy label for saving the files
    """
    if spectral_type == None:
        print("Missing required keyword argument \"spectral_type\", should be \"hardening\" or \"constant\"")
    
    # Simulation  box size
    xmax = 20*u.kpc
    ymax = 20*u.kpc
    zmax =  2*u.kpc
    # Setup the Galactic field models
    cartesian_grid = img.fields.UniformGrid(box=[[-xmax, xmax],
                                                 [-ymax, ymax],
                                                 [-zmax, zmax]],
                                                 resolution = [40,40,10]) # skipping x=y=0
    
    # Setup observing configuration
    observer = np.array([-8.5,0,0])*u.kpc
    config = {'grid'    :cartesian_grid,
              'observer':observer,
              'dist':hiidist,
              'e_dist':dist_err,
              'lat':lat,
              'lon':lon,
              'FB':FB}

    # Setup fields and field factories
    Bfield    = WrappedJF12(grid=cartesian_grid)
    B_factory = WrappedJF12Factory(grid=cartesian_grid)

    # Two cases for the CRE fields
    if spectral_type == "hardening": # use linear scaling alpha
        print("Using a scaling spectral index alpha ...\n")
        cre_num = img.fields.CRENumberDensity(
            grid       = cartesian_grid,
            parameters = {'scale_radius':10*u.kpc,
                        'scale_height':1*u.kpc,
                        'central_density':1e-5*u.cm**-3})
        cre_alpha = img.fields.SpectralIndexLinearVerticalProfile(
            grid       = cartesian_grid,
            parameters = {'soft_index':-4, 'hard_index':-2.5, 'slope':1*u.kpc**-1})
        cre_num_factory = img.fields.FieldFactory(field_class=cre_num, grid=cartesian_grid)
        alpha_factory   = img.fields.FieldFactory(field_class=cre_alpha, grid=cartesian_grid)
        alpha_factory.active_parameters=(('slope','soft_index','hard_index'))
        alpha_factory.priors = {
            'slope':img.priors.FlatPrior(xmin=0*u.kpc**-1, xmax=5*u.kpc**-1),                            
            'soft_index':img.priors.FlatPrior(xmin=-5, xmax=-3),
            'hard_index':img.priors.FlatPrior(xmin=-3, xmax=-2.1)}
        factory_list = [cre_num_factory, alpha_factory, B_factory] 
    if spectral_type == "constant": # use costant alpha
        print("Using constant spectral index alpha ...\n")
        cre = img.fields.PowerlawCosmicRayElectrons(grid=cartesian_grid,
                                            parameters = {'scale_radius':10*u.kpc,
                                                         'scale_height':1*u.kpc,
                                                         'central_density':1e-5*u.cm**-3,
                                                         'spectral_index':-3})
        cre_factory = img.fields.FieldFactory(field_class=cre, grid=config['grid']) 
        cre_factory.active_parameters = ('spectral_index',)
        cre_factory.priors = {
            'spectral_index':img.priors.FlatPrior(xmin=-5, xmax=-2.1)}
        factory_list = [cre_factory, B_factory]
    
    # Simulate data for different noise scales and save results
    results_dictionary = make_empty_dictionary(scales=rel_error)
    for err in rel_error:
        # Clear previous pipeline
        os.system("rm -r {}/*".format(rundir+spectral_type))
        # Setup simulator
        los_simulator = SpectralSynchrotronEmissivitySimulator(mea, config)
        # Initialize likelihood
        likelihood = img.likelihoods.SimpleLikelihood(mea)    
        # Setup final pipeline
        pipeline = img.pipelines.MultinestPipeline( simulator     = los_simulator,
                                                    run_directory = rundir+spectral_type,
                                                    factory_list  = factory_list,
                                                    likelihood    = likelihood)
        pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}
        # Run!    
        results = pipeline()
        results_dictionary = save_pipeline_results(pipeline=pipeline, label=err, dictionary=results_dictionary)
    return results_dictionary


def unpack_samples_and_evidence(dictionary={}):
    evidence  = np.array(dictionary['evidence'])
    scales    = dictionary['scales']
    npsamples = ()
    for xvalue in scales: # collect 2D matrix of parameters for scale
        s = []
        for samples in dictionary['samples_{}'.format(xvalue)]:
            samp_unit   = list(samples)
            samp_nounit = []
            for w in samp_unit:
                samp_nounit.append(w/w.unit)
            s.append(samp_nounit)
        npsamples += (np.array(s),)
    return scales, npsamples, evidence

def plot_corner(fname,data,colnames):
    x, samp_arrays, evidence = data
    samp = samp_arrays[0]
    print("Figure at x={} and evidence={}".format(x[-1],evidence[-1]))
    # Make cornerplot
    plt.close("all")
    df  = pd.DataFrame(data=samp, columns=colnames)
    fig = sns.pairplot(data=df, corner=True, kind='kde')
    #plt.title("Parameter estimates,     evidence={}".format(evidence[0]))
    plt.savefig(figpath+fname+'_pairplot.png')
    plt.close("all")

#%% Results
#===================================================================================
spectral_types = ["constant","hardening"]

def get_samples_spectral_hardening():
    for spec in spectral_types:
        results_dictionary = run_hiidata_pipeline(spectral_type=spec)
        np.save(logdir+'hiidata_pipeline_{}.npy'.format(spec), results_dictionary)
#get_samples_spectral_hardening()

def plot_samples_spectral_hardening(spectral_type="hardening"):
    results_dictionary = np.load(logdir+'hiidata_pipeline_{}.npy'.format(spectral_type), allow_pickle=True).item()
    data = unpack_samples_and_evidence(results_dictionary)
    # Make figure
    fname = 'hiidata_pipeline_{}'.format(spectral_type)
    pnames = ('slope','soft_index','hard_index')
    plot_corner(fname    = fname,
                data     = data,
                colnames = pnames)
plot_samples_spectral_hardening()


def plot_samples_spectral_constant(spectral_type="constant"):
    results_dictionary = np.load(logdir+'hiidata_pipeline_{}.npy'.format(spectral_type), allow_pickle=True).item()
    data = unpack_samples_and_evidence(results_dictionary)
    # Make figure
    fname = 'hiidata_pipeline_{}'.format(spectral_type)
    pnames = ('spectral_index',)
    plot_corner(fname    = fname,
                data     = data,
                colnames = pnames)
plot_samples_spectral_constant()

# Plot results
def plot_samples_spectral_constant(spectral_type="constant"):
    results_dictionary = np.load(logdir+'hiidata_pipeline_{}.npy'.format(spectral_type), allow_pickle=True).item()
    data = unpack_samples_and_evidence(results_dictionary)
    x, samp_arrays, evidence = data
    print("Figure at x={} and evidence={}".format(x[-1],evidence[-1]))
    samp   = samp_arrays[-1][:,0] # worst case noise
    # Make figure
    plt.close("all")
    sns.displot(samp,kde=True)
    plt.title("Posterior Constant Spectral Index")
    plt.xlabel("\N{GREEK SMALL LETTER alpha}")
    plt.tight_layout()
    plt.savefig(figpath+'hiidata_pipeline_samples_constant_index.png')
plot_samples_spectral_constant()