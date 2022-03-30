# Simulator dependancies
import numpy as np
import imagine as img
from imagine import fields
from imagine.simulators import Simulator
from scipy.special import gamma as gammafunc
import nifty7 as ift

import astropy.units as u
from astropy.coordinates import spherical_to_cartesian
from astropy import constants as cons
MHz   = 1e6 / u.s
GHz   = 1e9 / u.s
me    = cons.m_e.cgs
c     = cons.c.cgs
kb    = cons.k_B.cgs
electron = (cons.e).gauss

# Units definitions since astropy doesnt handle [B]_cgs well
gauss_B  = (u.g/u.cm)**(0.5)/u.s
equiv_B  = [(u.G, gauss_B, lambda x: x, lambda x: x)]
ugauss_B = 1e-6 * gauss_B

# Script dependancies
import matplotlib.pyplot as plt

# Run directory for pipeline
rundir = 'runs/mockdata'


print('\nRunning test_lost_pipeline.py\n\n')

#%% Define the Simulator class

class SpectralSynchrotronEmissivitySimulator(Simulator):
    """
    Simulator for synchrotron emissivity. Assumes a constant powerlaw spectrum
    throughout the entire Galaxy.
    
    Requires the user to define the setup in the los_dictionary    
    
    To run the default __call__ method one requires to input:
    - A 3D magnetic_field instance
    - A cosmic_ray_electron_density instance: 3D scalar density grid and constant spectral_index attribute
    
    """
    
    # Class attributes
    SIMULATED_QUANTITIES = ['average_los_brightness']
    REQUIRED_FIELD_TYPES = ['magnetic_field', 'cosmic_ray_electron_density']
    ALLOWED_GRID_TYPES   = ['cartesian']
    
    def __init__(self, measurements, sim_config={'grid':None,'observer':None,'dist':None,'e_dist':None,'lat':None,'lon':None,'FB':None}):
        
        print("Initializing SynchtrotronEmissivitySimulator")
        # Send the Measurements to the parent class
        super().__init__(measurements) 
        
        # Write assert and test functions to make sure stuf is correct
        # - grid.unit == observer.unit
        # - grid = instance of cartesian grid
        # - all HII regions should be located within the simulation box        
        
        # Stores class-specific attributes (and for now double definitions)
        for key in mea.keys(): self.observing_frequency = key[1] * GHz
        grid = sim_config['grid'] # unpack for readability
        
        # Setup unitvector grid used for mapping perpendicular component
        unit = sim_config['observer'].unit 
        unitvectors = []
        for x,y,z in zip(grid.x.ravel()/unit,grid.y.ravel()/unit,grid.z.ravel()/unit):
            v = np.array([x,y,z])-sim_config['observer']/unit
            normv = np.linalg.norm(v)
            if normv == 0: # special case where the observer is inside one of the grid points
                unitvectors.append(v)
            else:
                unitvectors.append(v/normv)
        # Save unitvector grid as class attribute
        self.unitvector_grid = np.reshape(unitvectors, tuple(grid.resolution)+(3,))
    
        # Cast imagine instance of grid into a RGSpace grid  
        cbox = grid.box
        xmax = (cbox[0][1]-cbox[0][0])/unit
        ymax = (cbox[1][1]-cbox[1][0])/unit
        zmax = (cbox[2][1]-cbox[2][0])/unit       
        box  = np.array([xmax,ymax,zmax])
        grid_distances = tuple([b/r for b,r in zip(box, grid.resolution)])
        #print("Domain paramters: ", grid.resolution, grid_distances)
        self.domain = ift.makeDomain(ift.RGSpace(grid.resolution, grid_distances)) # need this later for the los integration 
    
        # Get lines of sight from data
        lat        = sim_config['lat'].to(u.rad)
        lon        = sim_config['lon'].to(u.rad)
        hIIdist    = sim_config['dist'].to(u.kpc)
        e_hIIdist  = sim_config['e_dist'].to(u.kpc)
        behind     = np.where(np.array(sim_config['FB'])=='B')
        nlos       = len(hIIdist)
        
        # Remember the translation
        translation = np.array([xmax,ymax,zmax])/2 * unit
        translated_observer = sim_config['observer'] + translation

        # Cast start and end points in a Nifty compatible format
        starts = []
        for o in translated_observer: starts.append(np.full(shape=(nlos,), fill_value=o)*u.kpc)
        start_points = np.vstack(starts).T

        ends = []
        los  = spherical_to_cartesian(r=hIIdist, lat=lat, lon=lon)
        for i,axis in enumerate(los): ends.append(axis+translated_observer[i])
        end_points = np.vstack(ends).T

        deltas = end_points - start_points
        clims  = box * np.sign(deltas) * unit
        clims[clims<0]=0 # if los goes in negative direction clim of xyz=0-plane
        
        with np.errstate(divide='ignore'):
            all_lambdas = (clims-end_points)/deltas # possibly divide by zero 
        lambdas = np.min(np.abs(all_lambdas), axis=1) # will drop any inf here

        start_points[behind] = end_points[behind] + np.reshape(lambdas[behind], newshape=(np.size(behind),1))*deltas[behind]
        
        self.los_distances = np.linalg.norm(end_points-start_points, axis=1)
   
        nstarts = self._make_nifty_points(start_points)
        nends   = self._make_nifty_points(end_points)
        #print(translated_observer)
        #print(np.sum(self.los_distances > xmax/2*unit))
        #print(start_points[self.los_distances > xmax/2*unit])
        #print(end_points[self.los_distances > xmax/2*unit])
        #print("Min-max of losdist: ", np.min(self.los_distances), np.max(self.los_distances))
        self.response = ift.LOSResponse(self.domain, nstarts, nends) # domain doesnt know about its units but starts/ends do?

    def _make_nifty_points(self, points, dim=3):
        rows,cols = np.shape(points)
        if cols != dim: # we want each row to be a coordinate (x,y,z)
            points = points.T
            rows   = cols
        npoints = []
        for d in range(dim):
            npoints.append(np.full(shape=(rows,), fill_value=points[:,d]))
        return npoints
    
    def _spectral_integralF(self, mu):
        return 2**(mu+1)/(mu+2) * gammafunc(mu/2 + 7./3) * gammafunc(mu/2+2./3)    
    
    def _spectral_total_emissivity(self, Bper, ncre):
        vobs      = self.observing_frequency
        alpha     = self.field_parameter_values['cosmic_ray_electron_density']['spectral_index']
        fraction1 = (np.sqrt(3)*electron**3*ncre/(8*np.pi*me*c**2))
        fraction2 = (4*np.pi*vobs*me*c/(3*electron))
        integral  = self._spectral_integralF( (-alpha-3)/2 )
        return fraction1 * fraction2**((1+alpha)/2) * Bper**((1-alpha)/2) * integral
    
    def _project_to_perpendicular(self, vectorfield):
        v_parallel      = np.zeros(np.shape(vectorfield)) * vectorfield.unit
        amplitudes      = np.sum(vectorfield * self.unitvector_grid, axis=3)
        v_parallel[:,:,:,0]  = amplitudes * self.unitvector_grid[:,:,:,0]
        v_parallel[:,:,:,1]  = amplitudes * self.unitvector_grid[:,:,:,1]
        v_parallel[:,:,:,2]  = amplitudes * self.unitvector_grid[:,:,:,2]
        v_perpendicular      = vectorfield - v_parallel
        v_perp_amplitude     = np.sqrt(np.sum(v_perpendicular*v_perpendicular,axis=3))
        return v_perp_amplitude
        
    def simulate(self, key, coords_dict, realization_id, output_units):
        """Calculate a 3D box with synchrotron emissivities"""        
        
        # Acces required fields
        alpha     = self.field_parameter_values['cosmic_ray_electron_density']['spectral_index']
        ncre_grid = self.fields['cosmic_ray_electron_density']  # in units cm^-3
        B_grid    = self.fields['magnetic_field']/u.microgauss * ugauss_B # fixing astropy cgs units
              
        # Project to perpendicular component along line of sight
        Bperp_amplitude_grid = self._project_to_perpendicular(B_grid)        
        
        # Calculate grid of emissivity values
        emissivity_grid = self._spectral_total_emissivity(Bperp_amplitude_grid, ncre_grid)
        
        # Do the los integration on the domain defined in init with the new emissivity grid
        HII_LOSemissivities = self.response(ift.Field(self.domain, emissivity_grid)).val_rw()
        HII_LOSemissivities *= emissivity_grid.unit * u.kpc # restore units
        
        # Need to convert to K/kpc units. Note that self.los_distances may not be correct when using e_dist for domain
        HII_LOSbrightness = c**2/(2*kb*self.observing_frequency**2)*HII_LOSemissivities/self.los_distances

        print(HII_LOSbrightness[:5].to(u.K/u.kpc))
        return HII_LOSbrightness # whatever is returned has to have the same shape as the object in measurements


#%% Make a simulated dataset with known model paramters

print("Producing simulated dataset ...\n")

# Mock dataset
Ndata = 100
observing_frequency = 90*MHz
dunit = u.K/u.kpc
T     = np.zeros(Ndata)*dunit # placeholder
T_err = np.zeros(Ndata)*dunit# placeholder
lat   = 90*np.linspace(-1,1,Ndata)*u.deg
lon   = 360*np.linspace(0,360,Ndata)*u.deg

fake_data = {'brightness':T,'err':T_err,'lat':lat,'lon':lon}
fake_dset = img.observables.TabularDataset(fake_data,
                                           name='average_los_brightness',
                                           frequency=observing_frequency,
                                           units=dunit,
                                           data_col='brightness',
                                           err_col='err',
                                           lat_col='lat',
                                           lon_col='lon')
mea = img.observables.Measurements(fake_dset)


# Setup coordinate grid
box_size       = 15*u.kpc
cartesian_grid = img.fields.UniformGrid(box=[[-box_size, box_size],
                                             [-box_size, box_size],
                                             [-box_size, box_size]],
                                             resolution = [31,31,31])

# Cosmic-Ray Electron Model
constCRE = fields.ConstantCosmicRayElectrons(grid=cartesian_grid,
                                             parameters={'density':1.0e-7*u.cm**-3,'spectral_index':-3})


# Magnetic Field Model
Bfield = fields.ConstantMagneticField(
    grid = cartesian_grid,
    ensemble_size= 1,
    parameters={'Bx': 6*u.microgauss,
                'By': 0*u.microgauss,
                'Bz': 0*u.microgauss})


# Simulator configuration
observer = np.array([0,0,0])*u.kpc
hIIdist  = (box_size-2*u.kpc)*np.random.rand(Ndata) + 1*u.kpc # uniform between [1, max-1] kpc
#print("\nSimulation domain: \n", cartesian_grid.box)
#print("Min-max of hIIdist: ", np.min(hIIdist), np.max(hIIdist))
dist_err = hIIdist/10
NF = 20 # number of front los measurements
F  = np.full(shape=(NF), fill_value='F', dtype=str)
B  = np.full(shape=(Ndata-NF), fill_value='B', dtype=str)
FB = np.hstack((F,B))
#np.random.shuffle(FB) # This seems to cause a bug with the nifty integration!!


config = {'grid'    :cartesian_grid,
          'observer':observer,
          'dist':hIIdist,
          'e_dist':dist_err,
          'lat':lat,
          'lon':lon,
          'FB':FB}

test_sim   = SpectralSynchrotronEmissivitySimulator(mea, config) # create instance
simulation = test_sim([constCRE, Bfield])

# Retreive simulated data
key = ('average_los_brightness', 0.09000000000000001, 'tab', None)
sim_brightness = simulation[key].data[0] * simulation[key].unit

# Add noise to the data proportional to T_err
sim_brightness += np.random.normal(loc=0, scale=0.01*sim_brightness, size=Ndata)*simulation[key].unit





#%% Setup full pipeline

# Setup (simulated) dataset
sim_data = {'brightness':sim_brightness,'err':sim_brightness/10,'lat':lat,'lon':lon}
sim_dset = img.observables.TabularDataset(sim_data,
                                           name='average_los_brightness',
                                           frequency=observing_frequency,
                                           units=dunit,
                                           data_col='brightness',
                                           err_col='err',
                                           lat_col='lat',
                                           lon_col='lon')
sim_mea = img.observables.Measurements(sim_dset)

# Setup simulator
los_simulator = SpectralSynchrotronEmissivitySimulator(sim_mea, config) # create instance

# Initialize likelihood
likelihood = img.likelihoods.EnsembleLikelihood(sim_mea)

# Setup field factories
constCRE = fields.ConstantCosmicRayElectrons(grid=cartesian_grid,
                                             parameters={'density':1.0e-7*u.cm**-3,'spectral_index':-3})
CRE_factory = img.fields.FieldFactory(field_class = constCRE) # no active paramters

Bfield = fields.ConstantMagneticField(
    grid          = cartesian_grid,
    ensemble_size = 1,
    parameters    = {'Bx': 0*u.microgauss,'By': 0*u.microgauss,'Bz': 0*u.microgauss})
B_factory                   = img.fields.FieldFactory(field_class=Bfield)
B_factory.active_parameters = ('Bx',)
B_factory.priors            = {'Bx':  img.priors.FlatPrior(xmin=0., xmax=10.)} # what units?


factory_list = [CRE_factory, B_factory]




# Setup pipeline
print("Starting pipeline ... \n")
pipeline = img.pipelines.MultinestPipeline( simulator     = los_simulator,
                                            run_directory = rundir,
                                            factory_list  = factory_list,
                                            likelihood    = likelihood,
                                            )
pipeline.sampling_controllers = {'evidence_tolerance': 0.1, 'n_live_points': 200}

# Run pipeline!
results = pipeline()

print(dir(results))







