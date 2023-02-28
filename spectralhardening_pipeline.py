"""
This script is used to do performance testing of the galactic los synchrotron simulator
and the retrieval of model parameters from SIMULATED datasets.



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
import seaborn as sns
import pandas as pd

# Directory paths
rundir    = 'runs/'
figpath   = 'figures/spectralhardening/'
fieldpath = 'arrayfields/'
logdir    = 'log/'

# Gobal testing constants
import astropy.units as u
MHz   = 1e6 / u.s
Ndata = 100
observing_frequency = 74*MHz
dunit = u.K/u.kpc
global_dist_error = 0.001
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

def plot_corner_and_evidence(fname,data,colnames):
    x, samp_arrays, evidence = data
    samp   = samp_arrays[0]
    print("Figure at x={} and evidence={}".format(x[0],evidence[0]))
    # Make cornerplot
    plt.close("all")
    df  = pd.DataFrame(data=samp, columns=colnames)
    fig = sns.pairplot(data=df, corner=True, kind='kde')
    plt.savefig(figpath+fname+'_pairplot.png')
    plt.close("all")
    plt.plot(x, evidence)
    plt.savefig(figpath+fname+'_evidence.png')

#%% Pipeline setup
#===================================================================================

# pipeline controller
def run_pipeline(rel_error=[global_brightness_error], spectral_type=None):
    """
    Test retrieval of correct CRE hardening slope for a linear hardening model assuming
    the JF12 regular magnetic field.
    """
    if spectral_type == None:
        print("Missing required keyword argument \"spectral_type\", should be \"hardening\" or \"constant\"")

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

    # Setup fields and field factories
    Bfield    = WrappedJF12(grid=cartesian_grid)
    B_factory = WrappedJF12Factory(grid=cartesian_grid)
    cre_num = img.fields.CRENumberDensity(grid= cartesian_grid,
                                        parameters={'scale_radius':10*u.kpc,
                                                    'scale_height':1*u.kpc,
                                                    'central_density':1e-5*u.cm**-3})
    cre_alpha = img.fields.SpectralIndexLinearVerticalProfile(
        grid=cartesian_grid,
        parameters={'soft_index':-4, 'hard_index':-2.5, 'slope':1*u.kpc**-1})
    field_list = [Bfield, cre_num, cre_alpha] # used for simulated data
    if spectral_type == "hardening": # use linear scaling alpha
        print("Using a scaling spectral index alpha ...\n")
        # Field factories
        cre_num_factory = img.fields.FieldFactory(field_class = cre_num, grid=cartesian_grid)
        alpha_factory   = img.fields.FieldFactory(field_class = cre_alpha, grid=cartesian_grid)
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
        # Produce simulated dataset with noise
        mock_data, error = produce_mock_data(field_list=field_list, mea=mea, config=config, noise=err)
        sim_data = {'brightness':mock_data,'err':error,'lat':config['lat'],'lon':config['lon']}
        sim_mea  = fill_imagine_dataset(sim_data)
        # Setup simulator
        los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)
        # Initialize likelihood
        likelihood = img.likelihoods.SimpleLikelihood(sim_mea)    
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

def make_samples_and_evidence_plot(fig, data=(), xlabel='', ylabel='', title=''):
    x, meany, stdy, evidence = data
    gs  = fig.add_gridspec(2, hspace=0, height_ratios= [3, 1])
    axs = gs.subplots(sharex=True)
    # Top plot: the sampled parameter
    axs[0].plot(x,meany)
    axs[0].fill_between(x, meany-stdy, meany+stdy, alpha=0.2)
    axs[0].set_ylabel("         "+ylabel)
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
    axs[1].set_ylabel("Evidence log(Z)    ")
    # Correct lables
    axs[1].set_xlabel(xlabel)
    plt.suptitle(title,fontsize=20)
    plt.tight_layout()
    fig.align_ylabels(axs)

def plot_corner_and_evidence(fname,data,colnames):
    x, samp_arrays, evidence = data
    samp   = samp_arrays[-1]
    print("Figure at x={} and evidence={}".format(x[-1],evidence[-1]))
    # Make cornerplot
    plt.close("all")
    df  = pd.DataFrame(data=samp, columns=colnames)
    fig = sns.pairplot(data=df, corner=True, kind='kde')
    plt.savefig(figpath+fname+'_pairplot.png')
    plt.close("all")
    plt.plot(x, evidence)
    plt.savefig(figpath+fname+'_evidence.png')



#%% Results
#===================================================================================
spectral_types = ["constant","hardening"]

# Generate brightness samples
def get_samples_spectral_hardening():
    error_scale = np.linspace(0.01,1.0,20)
    for spec in spectral_types:
        results_dictionary = run_pipeline(rel_error=error_scale, spectral_type=spec)
        np.save(logdir+'samples_spectral_{}.npy'.format(spec), results_dictionary)
#get_samples_spectral_hardening()

# Plot results
def plot_evidence_spectral_hardening():
    plt.close("all")
    for spec in spectral_types:
        results_dictionary = np.load(logdir+'samples_spectral_{}.npy'.format(spec), allow_pickle=True).item()
        evidence  = results_dictionary['evidence']
        rel_error = results_dictionary['scales'] 
        plt.plot(rel_error, evidence, label=spec)
    plt.ylim([-100, 400])
    plt.title("Model comparison for CRE spectrum properties")
    plt.ylabel("Evidence log(Z)")
    plt.xlabel("Relative brightness error (eTB/TB)")
    plt.legend(title="Spectrum types")
    plt.savefig(figpath+"evidence_spectral_hardening.png")
    plt.close("all")
#plot_evidence_spectral_hardening()

# Plot results
def plot_samples_spectral_hardening(spectral_type="hardening"):
    results_dictionary = np.load(logdir+'samples_spectral_{}.npy'.format(spectral_type), allow_pickle=True).item()
    data = unpack_samples_and_evidence(results_dictionary)
    # Make figure
    fname = 'samples_spectral_{}'.format(spectral_type)
    pnames = ('slope','soft_index','hard_index')
    plot_corner_and_evidence(fname    = fname,
                             data     = data,
                             colnames = pnames)
#plot_samples_spectral_hardening()

# Plot results
def plot_samples_spectral_constant(spectral_type="constant"):
    results_dictionary = np.load(logdir+'samples_spectral_{}.npy'.format(spectral_type), allow_pickle=True).item()
    data = unpack_samples_and_evidence(results_dictionary)

plot_samples_spectral_constant()


"""
    # Make figure
    plt.close("all")
    figure = plt.figure()
    xlabel = "Relative brightness error (eTB/TB)"
    ylabel = "Magnetic field amplitude B_arm2 (mG)"
    title  = "Nested sampling results B_arm2"
    make_samples_and_evidence_plot( fig=figure,
                                    data=data,
                                    xlabel=xlabel,
                                    ylabel=ylabel,
                                    title=title)
    plt.savefig(figpath+'samples_relbrightness_errorV2.png')
    plt.close("all")

"""