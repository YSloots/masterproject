# Simulator dependancies
import numpy as np
import imagine as img
from imagine import fields
from imagine.simulators import Simulator
from scipy.special import gamma as gammafunc
import nifty7 as ift

import astropy.units as u
from astropy import constants as cons
MHz   = 1e6 / u.s
GHz   = 1e9 / u.s
me    = cons.m_e.cgs
c     = cons.c.cgs
electron = (cons.e).gauss

# Units definitions since astropy doesnt handle [B]_cgs well
gauss_B  = (u.g/u.cm)**(0.5)/u.s
equiv_B  = [(u.G, gauss_B, lambda x: x, lambda x: x)]
ugauss_B = 1e-6 * gauss_B

# Script dependancies
import matplotlib.pyplot as plt

#%% Usefull functions for inspection 

figpath = 'figures/'

def plot_slices(data, grid, fname=' '):
    unit       = data.unit
    resolution = grid.resolution    
    
    # Take a horizontal and a vertical slice through the middle of the box
    #print("Taking slicing index", int(resolution[2]/2))
    hor_slice = data[:,:,int(resolution[2]/2)]/unit 
    ver_slice = data[int(resolution[0]/2),:,:]/unit
    x = grid.x[:,0,0]/u.kpc
    y = grid.y[0,:,0]/u.kpc
    z = grid.z[0,0,:]/u.kpc

    titles = ['xy-slice z=0','yz-slice x=0']
    slices = [hor_slice, ver_slice]
    coords = [[x,y], [x,z]]
    clabel = [['x','y'], ['x','z']]

    maxvalue = np.max(data)/unit
    
    #print(slices, coords, maxvalue)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5), sharey = False)
    n = 0
    for ax in axes.flat:
        #im = ax.contourf(x, y, slices[n], 40, cmap='RdGy', vmin = -range, vmax = range)
        im = ax.contourf(coords[n][0], coords[n][1], slices[n],
    		     40, cmap='Blues', vmin = 0, vmax = maxvalue)
        ax.title.set_text(titles[n])
        ax.set_xlabel(clabel[n][0]+' kpc')
        ax.set_ylabel(clabel[n][1]+' kpc')
        n += 1
        
        
    # fake subplot for colorbar
    fakez = np.zeros((len(y),len(x)))
    fakez[0,0] = 0 # fake pixels for colorbar
    fakez[0,1] = maxvalue
    ax4 = fig.add_subplot(1,3,1)
    im = ax4.contourf(x, y, fakez, 40, cmap='Blues', vmin = 0, vmax = maxvalue)
    cbar_ax = fig.add_axes([0.92, 0.10, 0.01, 0.8])
    cbar = plt.colorbar(im, cax = cbar_ax)
    cbar.set_label(unit, rotation=0)
    plt.delaxes(ax4)
    
    fig.savefig(figpath+fname)
    plt.close('all')
    
    return


def make_unitvector_grid(imagine_grid, position):
    unit = position.unit # need to check that units are the same
    unitvectors = []
    for x,y,z in zip(imagine_grid.x.ravel()/unit,imagine_grid.y.ravel()/unit,imagine_grid.z.ravel()/unit):
        v = np.array([x,y,z])-position/unit
        unitvectors.append(v/np.linalg.norm(v))
    return np.reshape(unitvectors, tuple(imagine_grid.resolution)+(3,))*unit


#%% Define the Simulator class

class SpectralSynchrotronEmissivitySimulator(Simulator):
    """
    Simulator for synchrotron emissivity. Assumes a constant powerlaw spectrum
    throughout the entire Galaxy.
    
    To run the __call__ method one requires to input:
    - A 3D magnetic_field instance
    - A cosmic_ray_electron_density instance: 3D scalar density grid and constant spectral_index attribute
    
    """
    
    # Class attributes
    SIMULATED_QUANTITIES = ['sync']
    REQUIRED_FIELD_TYPES = ['magnetic_field', 'cosmic_ray_electron_density']
    ALLOWED_GRID_TYPES   = ['cartesian']
    
    def __init__(self, measurements, observing_frequency, observer_position, grid):
        print("Initializing SynchtrotronEmissivitySimulator")
        # Send the Measurements to the parent class
        super().__init__(measurements) 
        
        # Stores class-specific attributes (and for now double definitions)
        # Measurements already contains the observing frequency for the sync data
        self.observing_frequency = observing_frequency
        self.grid                = grid # only needed temporarily or for plotting and testing
        
        # After setting observer position make unit vectorgrid.
        # Only want to do this once since this will never change 
        # even when sampling over different fields.
        unit = observer_position.unit # need to check that grid.unit and position.unit are the same
        unitvectors = []
        for x,y,z in zip(grid.x.ravel()/unit,grid.y.ravel()/unit,grid.z.ravel()/unit):
            v = np.array([x,y,z])-observer_position/unit
            #print(v)
            normv = np.linalg.norm(v)
            if normv == 0: # special case where the observer is inside one of the grid points
                unitvectors.append(v)
            else:
                unitvectors.append(v/normv)
        
        #print(tuple(self.resolution)+(3,))
        self.unitvector_grid = np.reshape(unitvectors, tuple(grid.resolution)+(3,))
        #print(self.unitvector_grid)
    

        # Cast imagine instance of grid into a RGSpace grid  
        unit = grid.x.unit # unit of grid should be the same as unit of observer_position
        cbox = grid.box
        xmax = (cbox[0][1]-cbox[0][0])/unit
        ymax = (cbox[1][1]-cbox[1][0])/unit
        zmax = (cbox[2][1]-cbox[2][0])/unit
        translation = np.array([xmax,ymax,zmax])/2     
        newobserver = observer_position/unit + translation        
        box         = (xmax,ymax,zmax)
        #print("Box from grid: ", box)
        distances   = tuple([b/r for b,r in zip(box, grid.resolution)])
        #print("Distances (dx,dy,dz): ", distances)
        self.domain = ift.makeDomain(ift.RGSpace(grid.resolution, distances)) # need this later for the los integration 
    
        # Get lines of sight from data (for now just test los)
        nlos   = 10
        
        starts = []
        [starts.append(newobserver) for dummy in range(nlos)]
        starts = np.array(starts)
        ends   = []
        [ends.append(translation) for dummy in range(nlos)] # towards Galacic center
        ends   = np.array(ends)
        
        print(starts[0,:], ends[0,:])
        # Define the cube integration domain to be called in simulate
        self.response = self._get_response(starts, ends)
    
    # convenience function
    def _make_nifty_points(self, points, dim=3):
        rows,cols = np.shape(points)
        if cols != dim: # we want each row to be a coordinate (x,y,z)
            points = points.T
            rows   = cols
        npoints = []
        npoints.append(np.full(shape=(rows,), fill_value=points[:,0]))
        npoints.append(np.full(shape=(rows,), fill_value=points[:,1]))
        npoints.append(np.full(shape=(rows,), fill_value=points[:,2]))
        return npoints

    def _get_response(self, starts, ends):
        nstarts = self._make_nifty_points(starts)
        nends   = self._make_nifty_points(ends)
        return ift.LOSResponse(self.domain, nstarts, nends)
        
    def _spectral_integralF(self, mu):
        return 2**(mu+1)/(mu+2) * gammafunc(mu/2 + 7./3) * gammafunc(mu/2+2./3)    
    
    def _spectral_total_emissivity(self, Bper, ncre):
        vobs  = self.observing_frequency
        alpha = self.field_parameter_values['cosmic_ray_electron_density']['spectral_index']

        fraction1 = (np.sqrt(3)*electron**3*ncre/(8*np.pi*me*c**2))#.decompose(bases=u.cgs.bases)
        fraction2 = (4*np.pi*vobs*me*c/(3*electron))#.decompose(bases=u.cgs.bases)
        integral  = self._spectral_integralF( (-alpha-3)/2 )

        emissivities = fraction1 * fraction2**((1+alpha)/2) * Bper**((1-alpha)/2) * integral
        #print(fraction1.unit, ', ', fraction2.unit, ', ', Bper.unit, ', ', emissivities.unit)
        return emissivities
    
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
        print("Calling synchrotron simulator")
        
        # Acces required fields
        alpha     = self.field_parameter_values['cosmic_ray_electron_density']['spectral_index']
        ncre_grid = self.fields['cosmic_ray_electron_density']  # in units cm^-3
        B_grid    = self.fields['magnetic_field']/u.microgauss * ugauss_B # fixing astropy cgs units
              
        # Project to perpendicular component along line of sight
        Bperp_amplitude_grid = self._project_to_perpendicular(B_grid)        
        
        # just for testing 
        plot_slices(Bperp_amplitude_grid/ugauss_B*u.microgauss, self.grid, 'Bperp_amplitudes.png')        
        
        # Calculate grid of emissivity values
        emissivity_grid = self._spectral_total_emissivity(Bperp_amplitude_grid, ncre_grid)
        print(emissivity_grid)        
        
        plot_slices(emissivity_grid, self.grid, 'emissivity_cube.png')
        
        # Do the los integration on the domain defined in init with the new emissivity grid
        HII_LOSemissivities = self.response(ift.Field(self.domain, emissivity_grid)).val_rw()
        
        print(HII_LOSemissivities)
        

        # Initiate Galactic grid that will contain our emissivities
        #emissivity_grid = np.zeros(self.resolution)
        # If ncre follows a single powerlaw use spectral emissivity funciton
        #if ncre_grid.NAME == 'powerlaw_cosmicray_electrons':
        #    emissivity_grid = 0
        #else:
        #    raise TypeError("Incorrect cosmic-ray density field: Expected field of type \'powerlaw_cosmicray_electrons\', received "+ncre_grid.NAME)
        
        return np.arange(Npoints) * u.erg.cgs # whatever is returned has to have the same shape as the object in measurements


#%% Setup simple test data and physical fields

# Fake data dictionary for LOS HII synchrotron measurements
freq      = 0.09*GHz
Npoints   = 10
fake_lat  = 360*np.linspace(0,1,Npoints)*u.deg
fake_lon  = np.zeros(Npoints)
fake_j    = np.arange(Npoints)
fake_jerr = fake_j/10
fake_data = {'emissivity': fake_j,
             'err': fake_jerr,
             'lat': fake_lat,
             'lon': fake_lon}
fake_dset = img.observables.TabularDataset(fake_data,
                                           name='sync',
                                           tag='I',
                                           frequency=freq,
                                           units=u.erg.cgs,
                                           data_col='emissivity',
                                           err_col='err',
                                           lat_col='lat',
                                           lon_col='lon')
mea = img.observables.Measurements(fake_dset)

# Setup coordinate grid
cartesian_grid = img.fields.UniformGrid(box=[[-15*u.kpc, 15*u.kpc],
                                             [-15*u.kpc, 15*u.kpc],
                                             [-15*u.kpc, 15*u.kpc]],
                                             resolution = [3,3,3])

# Cosmic-Ray Electron Model
powerCRE_exponential = fields.PowerlawCosmicRayElectrons(
    grid             = cartesian_grid,
    parameters       ={'scale_radius':5.0*u.kpc,
                       'scale_height':1.0*u.kpc,
                       'central_density':314.5*u.cm**-3,
                       'spectral_index':-3})


# Magnetic Field Model
Bfield = fields.ConstantMagneticField(
    grid = cartesian_grid,
    ensemble_size= 1,
    parameters={'Bx': 1*u.microgauss,
                'By': 0*u.microgauss,
                'Bz': 0*u.microgauss})


#%% Call the simulator

vobs     = 90*MHz
observer = np.array([-20,0,0])*u.kpc

test_sim   = SpectralSynchrotronEmissivitySimulator(mea,vobs,observer,cartesian_grid) # create instance
simulation = test_sim([powerCRE_exponential, Bfield]) # call instance again with the required field types



print(simulation)






















