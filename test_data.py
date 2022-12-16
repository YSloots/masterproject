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

def produce_mock_data(field_list, mea, config, noise, seed=31415):
    np.random.seed(seed)
    """Runs the simulator once to produce a simulated dataset"""
    #config['e_dist'] = None
    test_sim   = SpectralSynchrotronEmissivitySimulator(measurements=mea, sim_config=config) 
    simulation = test_sim(field_list)
    key = ('average_los_brightness', 0.09000000000000001, 'tab', None)
    sim_brightness = simulation[key].data[0] * simulation[key].unit
    sim_brightness += np.random.normal(loc=0, scale=noise*sim_brightness, size=Ndata)*simulation[key].unit
    return sim_brightness

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


#%% Pipeline controllers
#===================================================================================

def scale_brightness_error(rel_error = [0]):
    
    # Produce empty data format
    T     = np.zeros(Ndata)*dunit # placeholder
    T_err = np.zeros(Ndata)*dunit # placeholder
    xmax = 20*u.kpc
    ymax = 20*u.kpc
    zmax =  2*u.kpc
    x = randrange(-0.9*xmax,0.9*xmax,Ndata) # constant seed
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
    Bfield = WrappedJF12(
        grid=cartesian_grid,
        parameters = {'b_arm_1': .1, 'b_arm_2': 3.0, 'b_arm_3': -.9, 'b_arm_4': -.8, 'b_arm_5': -2.,
        'b_arm_6': -4.2, 'b_arm_7': .0, 'b_ring': .1, 'h_disk': .4, 'w_disk': .27,
        'Bn': 1.4, 'Bs': -1.1, 'rn': 9.22, 'rs': 16.7, 'wh': .2, 'z0': 5.3, 'B0_X': 4.6,
        'Xtheta_const': 49, 'rpc_X': 4.8, 'r0_X': 2.9, })

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
    
    # Simulate data for different noise scales and save results
    results_dictionary = {'error_scale':rel_error}
    results_dictionary['evidence'] = []
    results_dictionary['evidence_err'] = []
    for err in rel_error:
        # Clear previous pipeline
        os.system("rm -r runs/mockdata/*")
        # Produce simulated dataset with noise
        mock_data = produce_mock_data(field_list=[cre,Bfield], mea=mea, config=config, noise=err)
        sim_data = {'brightness':mock_data,'err':mock_data/10,'lat':config['lat'],'lon':config['lon']}
        sim_mea  = fill_imagine_dataset(sim_data)
        # Setup simulator
        los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)
        # Initialize likelihood
        likelihood = img.likelihoods.SimpleLikelihood(sim_mea)    
        # Setup field factories and their active parameters
        B_factory   = img.fields.FieldFactory(field_class = Bfield, grid=config['grid'])
        CRE_factory = img.fields.FieldFactory(field_class = cre, grid=config['grid']) 
        B_factory.active_parameters = ('b_arm_2',)
        B_factory.priors = {'b_arm_2':img.priors.FlatPrior(xmin=0, xmax=10)}
        factory_list = [B_factory, CRE_factory]
        # Setup final pipeline
        pipeline = img.pipelines.MultinestPipeline( simulator     = los_simulator,
                                                    run_directory = rundir,
                                                    factory_list  = factory_list,
                                                    likelihood    = likelihood)
        pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}
        # Run!
        results = pipeline()
        results_dictionary['summary_{}'.format(err)] = pipeline.posterior_summary
        results_dictionary['samples_{}'.format(err)] = pipeline.samples
        results_dictionary['evidence'].append(pipeline.log_evidence)
        results_dictionary['evidence_err'].append(pipeline.log_evidence_err)
    return results_dictionary
#scale_brightness_error(rel_error=1)




def scale_distance_error(rel_error = [0]):
    
    # Produce empty data format
    T     = np.zeros(Ndata)*dunit # placeholder
    T_err = np.zeros(Ndata)*dunit # placeholder
    xmax = 20*u.kpc
    ymax = 20*u.kpc
    zmax =  2*u.kpc
    x = randrange(-0.9*xmax,0.9*xmax,Ndata) # constant seed
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
    Bfield = WrappedJF12(
        grid=cartesian_grid,
        parameters = {'b_arm_1': .1, 'b_arm_2': 3.0, 'b_arm_3': -.9, 'b_arm_4': -.8, 'b_arm_5': -2.,
        'b_arm_6': -4.2, 'b_arm_7': .0, 'b_ring': .1, 'h_disk': .4, 'w_disk': .27,
        'Bn': 1.4, 'Bs': -1.1, 'rn': 9.22, 'rs': 16.7, 'wh': .2, 'z0': 5.3, 'B0_X': 4.6,
        'Xtheta_const': 49, 'rpc_X': 4.8, 'r0_X': 2.9, })

    # Simulate data for different noise scales and save results
    results_dictionary = {'error_scale':rel_error}
    for err in rel_error:
        # Clear previous pipeline
        os.system("rm -r runs/mockdata/*")
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
        mock_data = produce_mock_data(field_list=[cre,Bfield], mea=mea, config=config, noise=0.01)
        sim_data = {'brightness':mock_data,'err':mock_data/10,'lat':config['lat'],'lon':config['lon']}
        sim_mea  = fill_imagine_dataset(sim_data)
        # Setup simulator
        los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)
        # Initialize likelihood
        likelihood = img.likelihoods.SimpleLikelihood(sim_mea)
        # Setup field factories and their active parameters
        B_factory   = img.fields.FieldFactory(field_class = Bfield, grid=config['grid'])
        CRE_factory = img.fields.FieldFactory(field_class = cre, grid=config['grid']) 
        B_factory.active_parameters = ('b_arm_2',)
        B_factory.priors = {'b_arm_2':img.priors.FlatPrior(xmin=0, xmax=10)}
        factory_list = [B_factory, CRE_factory]
        # Setup final pipeline
        pipeline = img.pipelines.MultinestPipeline( simulator     = los_simulator,
                                                    run_directory = rundir,
                                                    factory_list  = factory_list,
                                                    likelihood    = likelihood)
        pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}
        # Run!
        results = pipeline()
        results_dictionary['summary_{}'.format(err)] = pipeline.posterior_summary
        results_dictionary['samples_{}'.format(err)] = pipeline.samples
    return results_dictionary


def scale_zdist(sigma_z = [0.03]):
    
    # Simulate data for different noise scales and save results
    results_dictionary = {'sigma_z':sigma_z}

    for stdz in sigma_z:

        # Clear previous pipeline
        os.system("rm -r runs/mockdata/*")

        # Produce empty data format
        T     = np.zeros(Ndata)*dunit # placeholder
        T_err = np.zeros(Ndata)*dunit # placeholder
        xmax = 20*u.kpc
        ymax = 20*u.kpc
        zmax =  2*u.kpc
        x = randrange(-0.9*xmax,0.9*xmax,Ndata) # constant seed
        y = randrange(-0.9*ymax,0.9*ymax,Ndata)
        z = truncnorm(a=-zmax/stdz, b=zmax/stdz, scale=stdz/u.kpc).rvs(Ndata)*u.kpc
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
        Bfield = WrappedJF12(
            grid=cartesian_grid,
            parameters = {'b_arm_1': .1, 'b_arm_2': 3.0, 'b_arm_3': -.9, 'b_arm_4': -.8, 'b_arm_5': -2.,
            'b_arm_6': -4.2, 'b_arm_7': .0, 'b_ring': .1, 'h_disk': .4, 'w_disk': .27,
            'Bn': 1.4, 'Bs': -1.1, 'rn': 9.22, 'rs': 16.7, 'wh': .2, 'z0': 5.3, 'B0_X': 4.6,
            'Xtheta_const': 49, 'rpc_X': 4.8, 'r0_X': 2.9, })

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
        mock_data = produce_mock_data(field_list=[cre,Bfield], mea=mea, config=config, noise=0.01)
        sim_data = {'brightness':mock_data,'err':mock_data/10,'lat':config['lat'],'lon':config['lon']}
        sim_mea  = fill_imagine_dataset(sim_data)
        # Setup simulator
        los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config)
        # Initialize likelihood
        likelihood = img.likelihoods.SimpleLikelihood(sim_mea)
        # Setup field factories and their active parameters
        B_factory   = img.fields.FieldFactory(field_class = Bfield, grid=config['grid'])
        CRE_factory = img.fields.FieldFactory(field_class = cre, grid=config['grid']) 
        B_factory.active_parameters = ('h_disk',)
        B_factory.priors = {'h_disk':img.priors.FlatPrior(xmin=0, xmax=2)}
        factory_list = [B_factory, CRE_factory]
        # Setup final pipeline
        pipeline = img.pipelines.MultinestPipeline( simulator     = los_simulator,
                                                    run_directory = rundir,
                                                    factory_list  = factory_list,
                                                    likelihood    = likelihood)
        pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}
        # Run!
        results = pipeline()
        results_dictionary['summary_{}'.format(stdz)] = pipeline.posterior_summary
        results_dictionary['samples_{}'.format(stdz)] = pipeline.samples
    return results_dictionary

#%% Results 4.4 Sensitivity to data characteristics
#=======================================================================================

# Generate brightness samples
def get_samples_relbrightness_error():
    """Do a sampling of Barm2 with JF12+ConstantAlphaCRE for different e_Te"""
    error_scale        = np.linspace(0,1,20)
    results_dictionary = scale_brightness_error(rel_error=error_scale)
    np.save(logdir+'samples_relbrightness_err.npy', results_dictionary)
#get_samples_relbrightness_error()

def unpack_samples_and_evidence(x, result_dict={}):
    meany = np.empty(len(x))
    stdy  = np.empty(len(x))
    for i,xvalue in enumerate(x):
        s = []
        for samples in results_dictionary['samples_{}'.format(xvalue)]:
            s.append(list(samples)[0])
        meany[i] = np.mean(s,axis=0)
        stdy[i]  = np.std(s,axis=0)
    return meany, stdy

# Plot results
def plot_samples_relbrightness_error():
    results_dictionary = np.load(logdir+'samples_relbrightness_err.npy', allow_pickle=True).item()
    rel_error  = results_dictionary['error_scale']
    evidence   = np.array(results_dictionary['evidence'])
    e_evidence = np.array(results_dictionary['evidence_err'])
    maxe = np.max(evidence)
    mine = np.min(evidence)
    evidence_ticks = [maxe,maxe-(maxe-mine)/2,mine]
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
    axs[0].plot(rel_error,meany)
    axs[0].fill_between(rel_error, meany-stdy, meany+stdy, alpha=0.2)
    axs[0].set_ylabel('magnetic field amplitude B_arm2')
    pos = np.where(evidence>=0)
    neg = np.where(evidence<=0)
    axs[1].plot(rel_error[pos], evidence[pos],c='blue')
    axs[1].plot(rel_error[neg], evidence[neg],c='red')
    cswitch = np.where(evidence*np.roll(evidence,1)<0)[0][1:]
    for i in cswitch: axs[1].plot(rel_error[i-1:i+1],evidence[i-1:i+1],c='red')
    #axs[1].fill_between(rel_error, evidence-e_evidence, evidence+e_evidence, alpha=0.2)
    axs[1].set_ylabel("evidence")
    axs[1].set_xlabel('relative brightness error  (eTB/TB)')
    axs[1].set_yticks(evidence_ticks)
    plt.suptitle('Nested sampling result B_arm2',fontsize=20)
    plt.tight_layout()
    fig.align_ylabels(axs)
    plt.savefig(figpath+'samples_relbrightness_error.png')
plot_samples_relbrightness_error()


# Generate distance samples
def get_samples_reldistance_error():
    """Do a sampling of Barm2 with JF12+ConstantAlphaCRE for different e_Te"""
    error_scale        = np.linspace(0.5,2,2)
    results_dictionary = scale_distance_error(rel_error=error_scale)
    np.save(logdir+'samples_reldistance_err.npy', results_dictionary)
#get_samples_reldistance_error()

# Plot results
def plot_samples_reldistance_error():
    results_dictionary = np.load(logdir+'samples_reldistance_err.npy', allow_pickle=True).item()
    rel_error = results_dictionary['error_scale']
    meany = np.empty(len(rel_error))
    stdy  = np.empty(len(rel_error))
    for i,err in enumerate(rel_error):
        y = []
        for value in results_dictionary['samples_{}'.format(err)]:
            y.append(list(value)[0])
        meany[i] = np.mean(y,axis=0)
        stdy[i]  = np.std(y,axis=0)
    plt.close('all')
    plt.plot(rel_error,meany)
    plt.fill_between(rel_error, meany-stdy, meany+stdy, alpha=0.2)
    plt.xlabel('relative distance error  (eD/D)')
    plt.ylabel('magnetic field amplitude B_arm2')
    plt.title('Nested sampling result B_arm2',fontsize=20)
    plt.savefig(figpath+'samples_reldistance_error.png')
#plot_samples_reldistance_error()

# Generate zdist samples
def get_samples_zdist():
    """Do a sampling of scalehight z0 with JF12+ConstantAlphaCRE for different zdist"""
    sigma_z            = np.linspace(0.01,0.5,20)
    results_dictionary = scale_zdist(sigma_z=sigma_z)
    np.save(logdir+'samples_zdist.npy', results_dictionary)
#get_samples_zdist()

# Plot results
def plot_samples_zdist():
    results_dictionary = np.load(logdir+'samples_zdist.npy', allow_pickle=True).item()
    sigma_z = results_dictionary['sigma_z']
    meany = np.empty(len(sigma_z))
    stdy  = np.empty(len(sigma_z))
    for i,s in enumerate(sigma_z):
        y = []
        for value in results_dictionary['samples_{}'.format(s)]:
            y.append(list(value)[0])
        meany[i] = np.mean(y,axis=0)
        stdy[i]  = np.std(y,axis=0)
    plt.close('all')
    plt.plot(sigma_z,meany)
    plt.fill_between(sigma_z, meany-stdy, meany+stdy, alpha=0.2)
    plt.xlabel('sigma zdidst')
    plt.ylabel('JF12 scale height h_disk')
    plt.title('Nested sampling results h_disk',fontsize=20)
    plt.savefig(figpath+'samples_zdist.png')
#plot_samples_zdist()