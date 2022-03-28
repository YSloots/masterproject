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
        
        all_lambdas = (clims-end_points)/deltas # possibly divide by zero
        lambdas     = np.min(np.abs(all_lambdas), axis=1) # will drop any inf here

        start_points[behind] = end_points[behind] + np.reshape(lambdas[behind], newshape=(np.size(behind),1))*deltas[behind]
        
        print("Starts: \n", start_points)   
        print("Ends:   \n", end_points)
        
        self.los_distances = np.linalg.norm(end_points-start_points, axis=1)
        print("LOS lengths: ", self.los_distances)        
        
        nstarts = self._make_nifty_points(start_points)
        nends   = self._make_nifty_points(end_points)

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
        print("Calling synchrotron simulator\n")
        
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
        #print(emissivity_grid, '\n')        
        
        plot_slices(emissivity_grid, self.grid, 'emissivity_cube.png')
        
        # Do the los integration on the domain defined in init with the new emissivity grid
        HII_LOSemissivities = self.response(ift.Field(self.domain, emissivity_grid)).val_rw()
        HII_LOSemissivities *= emissivity_grid.unit * u.kpc # restore units
        print("Emissivities: ")
        print(HII_LOSemissivities, '\n') 
        
        # Need to convert to K/kpc units. Note that self.los_distances may not be correct when using e_dist for domain
        HII_LOSbrightness = c**2/(2*kb*self.observing_frequency**2)*HII_LOSemissivities/self.los_distances
        print("Average brightness temperature: ")
        print(HII_LOSbrightness.to(u.K/u.kpc), '\n')
        
        return np.arange(Npoints)*u.K/u.kpc # whatever is returned has to have the same shape as the object in measurements


#%% Setup simple test data and physical fields


# Setup coordinate grid
cartesian_grid = img.fields.UniformGrid(box=[[-15*u.kpc, 15*u.kpc],
                                             [-15*u.kpc, 15*u.kpc],
                                             [-15*u.kpc, 15*u.kpc]],
                                             resolution = [61,61,61])

# Fake data dictionary for LOS HII synchrotron measurements
freq      = 0.09*GHz
Npoints   = 4
fake_lat  = np.zeros(Npoints)*u.deg
fake_lon  = np.zeros(Npoints)*u.deg

# custom los for testing:
fake_lon[2] = 90*u.deg  # positive y-direction
fake_lon[3] = -90*u.deg # negative y-direction
fake_lat[1] = 90*u.deg  # positive z-direction
observer    = np.array([-15,0,0])*u.kpc # from the middle of the 3x3 cube side
# -> integrating allong the x=-15-plane should give 15-central voxel

fake_j    = np.arange(Npoints) * u.K/u.kpc
fake_jerr = fake_j/10
fake_data = {'emissivity': fake_j,
             'err': fake_jerr,
             'lat': fake_lat,
             'lon': fake_lon}
fake_dset = img.observables.TabularDataset(fake_data,
                                           name='average_los_brightness',
                                           frequency=freq,
                                           units=u.K/u.kpc,
                                           data_col='emissivity',
                                           err_col='err',
                                           lat_col='lat',
                                           lon_col='lon')
mea = img.observables.Measurements(fake_dset)



# Cosmic-Ray Electron Model
constCRE = fields.ConstantCosmicRayElectrons(grid      =cartesian_grid,
                                             parameters={'density':1.0e37/7.01499*u.cm**-3,
                                                         'spectral_index':-3})
credata = constCRE.get_data()

# Magnetic Field Model
Bfield = fields.ConstantMagneticField(
    grid = cartesian_grid,
    ensemble_size= 1,
    parameters={'Bx': 1*u.microgauss,
                'By': 0*u.microgauss,
                'Bz': 0*u.microgauss})


#%% Call the simulator

vobs     = 90*MHz
#observer = np.array([-15,0,0])*u.kpc # observer is defined where lon and lat are too

config = {'grid'    :cartesian_grid,
          'observer':observer,
          'dist':np.array([30,15,10,10])*u.kpc,
          'e_dist':np.ones(Npoints)*0.3*u.kpc,
          'lat':fake_lat,
          'lon':fake_lon,
          'FB':['F','F','B','B']}
          
#print(mea)
test_sim   = SpectralSynchrotronEmissivitySimulator(mea, config) # create instance
simulation = test_sim([constCRE, Bfield]) # call instance again with the required field types



#print(simulation)






















