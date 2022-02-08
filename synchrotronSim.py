import imagine as img
from imagine import fields
from imagine.simulators import Simulator
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
MHz   = 1e6 / u.s
GHz   = 1e9 / u.s
# Units definitions since astropy doesnt handle [B]_cgs well
gauss_B = (u.g/u.cm)**(0.5)/u.s
equiv_B = [(u.G, gauss_B, lambda x: x, lambda x: x)]

## Define the Simulator class

class SynchrotronEmissivitySimulator(Simulator):
    """
    Simulator for synchrotron emissivity 
    
    
    """
    
    # Class attributes
    SIMULATED_QUANTITIES = ['sync']
    REQUIRED_FIELD_TYPES = ['magnetic_field', 'cosmic_ray_electron_density']
    ALLOWED_GRID_TYPES   = ['cartesian']
    
    def __init__(self, measurements, observing_frequency, observer_position, grid):
        # Send the Measurements to the parent class
        super().__init__(measurements) 
        
        # Stores class-specific attributes (and for now double definitions)
        # Measurements already contains the observing frequency for the sync data
        self.observing_frequency = observing_frequency
        self.resolution          = grid.resolution
        
        # After setting observer position make unit vectorgrid
        unit = position.unit # need to check that units are the same
        unitvectors = []
        for x,y,z in zip(grid.x.ravel()/unit,grid.y.ravel()/unit,grid.z.ravel()/unit):
            v = np.array([x,y,z])-position/unit
            unitvectors.append(v/np.linalg.norm(v))
        #print(tuple(self.resolution)+(3,))
        self.unitvector_grid = np.reshape(unitvectors, tuple(self.resolution)+(3,))
        #print(self.unitvector_grid)
    
    
    
    def simulate(self):
        
        """Calculate a 3D box with synchrotron emissivities"""        
        
        # Get the perpendicular component of the GMF for all grid points
        Bfield          = self.fields['magnetic_field'].get_data()
        Bpara           = np.empty(np.shape(Bfield))
        Bpara[:,:,:,0]  = np.sum(Bgrid*self.unitvector_grid,axis=3)*self.unitvector_grid[:,:,:,0]
        Bpara[:,:,:,1]  = np.sum(Bgrid*self.unitvector_grid,axis=3)*self.unitvector_grid[:,:,:,1]
        Bpara[:,:,:,2]  = np.sum(Bgrid*self.unitvector_grid,axis=3)*self.unitvector_grid[:,:,:,2]
        Bperp           = Bfield - Bpara
        Bperp_amplitude = np.sqrt(np.sum(Bperp*Bperp,axis=3))
        
        # Initiate Galactic grid that will contain our emissivities
        emissivity_grid = np.zeros(self.resolution)
        
        # If ncre follows a single powerlaw use spectral emissivity funciton
        ncre = self.fields['cosmic_ray_electron_density']
        if ncre.NAME == 'powerlaw_cosmicray_electrons':
            emissivity_grid = 0
            
        
        return emissivity_grid


## Setup simple test data and environment

# Fake data dictionary for LOS HII synchrotron measurements
freq      = 0.09*GHz
Npoints   = 10
fake_lat  = 360*np.linspace(0,1,Npoints)*u.deg
fake_lon  = np.zeros(Npoints)
fake_dist = np.arange(1,Npoints+1)
fake_j    = np.arange(10,Npoints+10)
fake_jerr = fake_j/2
fake_data = {'emissivity': fake_j/fake_dist,
             'err': fake_jerr/fake_dist,
             'lat': fake_lat,
             'lon': fake_lon}
fake_dset = img.observables.TabularDataset(fake_data,
                                           name='sync',
                                           tag='I',
                                           frequency=freq,
                                           units=u.erg.cgs/u.cm.cgs,
                                           data_col='emissivity',
                                           err_col='err',
                                           lat_col='lat',
                                           lon_col='lon')
mea = img.observables.Measurements(fake_dset)
print(mea.keys())
print()


# Setup coordinate grid
cartesian_grid = img.fields.UniformGrid(box=[[-15*u.kpc, 15*u.kpc],
                                             [-15*u.kpc, 15*u.kpc],
                                             [-5*u.kpc, 5*u.kpc]],
                                             resolution = [3,3,3])

def make_unitvector_grid(imagine_grid, position):
    unit = position.unit # need to check that units are the same
    unitvectors = []
    for x,y,z in zip(imagine_grid.x.ravel()/unit,imagine_grid.y.ravel()/unit,imagine_grid.z.ravel()/unit):
        v = np.array([x,y,z])-position/unit
        unitvectors.append(v/np.linalg.norm(v))
    return np.reshape(unitvectors, tuple(imagine_grid.resolution)+(3,))*unit

# Cosmic-Ray Electron Model
powerCRE_exponential = fields.PowerlawCosmicRayElectrons(
    grid             = cartesian_grid,
    parameters       ={'scale_radius':5.0*u.kpc,
                       'scale_height':1.0*u.kpc,
                       'spectral_index':-3  })

# Magnetic Field Model
Bfield = fields.ConstantMagneticField(
    grid = cartesian_grid,
    ensemble_size= 1,
    parameters={'Bx': 0*u.microgauss,
                'By': 1*u.microgauss,
                'Bz': 2*u.microgauss})
#print(Bfield.get_data().ravel())

## Call the simulator

vobs     = 90*MHz
position = np.array([-8.5,0,0])*u.kpc

test_sim   = SynchrotronEmissivitySimulator(mea,vobs,position,cartesian_grid) # create instance
#simulation = test_sim([Bfield, powerCRE_exponential]) # call instance again with the required field types


#print(simulation)






















