"""
The aim of this script is to investigate if a geometry effect is causing the 
extreme sensitivity to the relative distance error. The hypotisis is that allowing
a non zero distance error causes random LOS to probe discontinuities in the model.
Making the integrated brightnesses allong different LOS vary greatly compared to 
the mock data LOS, which were generated with zero distance error.
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

# Directory paths
rundir    = 'runs/mockdata'
figpath   = 'figures/'
fieldpath = 'arrayfields/'
logdir    = 'log/'

# Gobal testing constants
import astropy.units as u
MHz   = 1e6 / u.s
Ndata = 2
observing_frequency = 90*MHz
dunit = u.K/u.kpc
global_dist_error = 0.00
global_brightness_error = 0.001

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
    key = ('average_los_brightness', 0.09000000000000001, 'tab', None)
    sim_brightness = simulation[key].data[0] * simulation[key].unit
    brightness_error = noise*sim_brightness
    brightness_error[brightness_error==0]=np.min(brightness_error[np.nonzero(brightness_error)])
    return sim_brightness, brightness_error


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
            s.append(samples[0]/u.microgauss)
        meany[i] = np.mean(s,axis=0)
        stdy[i]  = np.std(s,axis=0)
    return scales, meany, stdy, evidence

def make_samples_and_evidence_plot(fig, data=(), xlabel='', ylabel='', title=''):
    gs  = fig.add_gridspec(2, hspace=0, height_ratios= [3, 1])
    axs = gs.subplots(sharex=True)
    colors = ['tab:orange','tab:blue']
    for c,d in zip(colors,data):
        x, meany, stdy, evidence = d
        # Top plot: the sampled parameter
        axs[0].plot(x,meany,c=c)
        axs[0].fill_between(x, meany-stdy, meany+stdy, alpha=0.2, color=c)
        # Bottem plot: the evidence
        axs[1].plot(x,evidence,c=c)
    axs[0].set_ylabel("         "+ylabel)
    axs[1].set_ylabel("Evidence log(Z)    ")
    # Correct lables
    axs[1].set_xlabel(xlabel)
    plt.suptitle(title,fontsize=20)
    plt.tight_layout()
    fig.align_ylabels(axs)


#%% Pipeline controllers
#===================================================================================

def low_effect():
    x = np.zeros(Ndata)*u.kpc
    y = np.array([-9,-9])*u.kpc # two LOS well inside
    z = np.zeros(Ndata)*u.kpc
    return x,y,z

def high_effect():
    x = np.zeros(Ndata)*u.kpc
    y = np.array([-9,19])*u.kpc # one LOS well inside, the other close to the edge
    z = np.zeros(Ndata)*u.kpc
    return x,y,z

def test_geometry(rel_error, effect_type = "HIGH"):
    # Produce empty data format
    T     = np.zeros(Ndata)*dunit # placeholder
    T_err = np.zeros(Ndata)*dunit # placeholder
    xmax = 20*u.kpc
    ymax = 20*u.kpc
    zmax =  2*u.kpc
    observer = np.array([0,-19,0])*u.kpc
    # Define endoints
    if effect_type == "HIGH":
        x,y,z = high_effect()
    elif effect_type == "LOW":
        x,y,z = low_effect()
    else:
        print("Must provide valid effect_type: HIGH or LOW")
    hIIdist, lat, lon = cartesian_to_spherical(x,y+19*u.kpc,z)
    # Place holder dataset
    fake_data = {'brightness':T,'err':T_err,'lat':lat,'lon':lon}
    mea       = fill_imagine_dataset(data=fake_data)
    # Setup the Galactic field models
    resolution = [100,100,3]
    cartesian_grid = img.fields.UniformGrid(box=[[-xmax, xmax],
                                                 [-ymax, ymax],
                                                 [-zmax, zmax]],
                                                 resolution = resolution) # skipping x=y=0
    cre = img.fields.ConstantCosmicRayElectrons(grid=cartesian_grid,
                                            parameters={'density':2.0e-7*u.cm**-3,'spectral_index':-3.0})
    Bfield = img.fields.ConstantMagneticField(
        grid = cartesian_grid,
        parameters={'Bx': 100*u.microgauss,
                    'By': 0*u.microgauss,
                    'Bz': 0*u.microgauss})
    B_factory   = img.fields.FieldFactory(field_class = Bfield, grid=cartesian_grid)
    CRE_factory = img.fields.FieldFactory(field_class = cre, grid=cartesian_grid) 
    B_factory.active_parameters = ('Bx',)
    B_factory.priors = {'Bx':img.priors.FlatPrior(xmin=10*u.microgauss, xmax=150*u.microgauss)}
    factory_list = [B_factory, CRE_factory]
    # Setup observing configuration
    FB       = np.full(shape=(Ndata), fill_value='F', dtype=str) # only front measurements
    config = {'grid'    :cartesian_grid,
                'observer':observer,
                'dist':hIIdist,
                'e_dist':None,
                'lat':lat,
                'lon':lon,
                'FB':FB}
    # Simulate data for different noise scales and save results
    results_dictionary = make_empty_dictionary(scales=rel_error)
    for err in rel_error:
        # Clear previous pipeline
        os.system("rm -r runs/mockdata/*")
        # Change distance error in config
        config["e_dist"] = err*config["dist"]
        # Produce simulated dataset with temperature noise
        mock_data, error = produce_mock_data(field_list=[cre,Bfield], mea=mea, config=config)
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

def get_geometry_effect_data():
    error_scale = np.linspace(0.0,0.5,50)
    label = "HIGH" # either HIGH or LOW
    results_dictionary = test_geometry(rel_error=error_scale, effect_type=label)
    np.save(logdir+'geometryeffect_{}.npy'.format(label), results_dictionary)
    label = "LOW" # either HIGH or LOW
    results_dictionary = test_geometry(rel_error=error_scale, effect_type=label)
    np.save(logdir+'geometryeffect_{}.npy'.format(label), results_dictionary)
get_geometry_effect_data()

# Plot results
def plot_geometry_effect():
    results_HIGH = np.load(logdir+'geometryeffect_HIGH.npy', allow_pickle=True).item()
    data_HIGH    = unpack_samples_and_evidence(results_HIGH)
    results_LOW  = np.load(logdir+'geometryeffect_LOW.npy', allow_pickle=True).item()
    data_LOW     = unpack_samples_and_evidence(results_LOW)
    # Make figure
    plt.close("all")
    figure = plt.figure()
    xlabel = "Relative distance error  (eD/D)"
    ylabel = "Magnetic field amplitude Bx (mG)"
    title  = "Nested sampling results Bx"
    make_samples_and_evidence_plot( fig=figure,
                                    data=(data_LOW,data_HIGH),
                                    xlabel=xlabel,
                                    ylabel=ylabel,
                                    title=title)
    plt.savefig(figpath+'geometryeffect.png')
    plt.close("all")
plot_geometry_effect()



