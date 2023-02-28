
#%% Imports and settings
#===================================================================================
# Imagine
import imagine as img
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12
from imagine.simulators import Simulator
from scipy.special import gamma as gammafunc
import nifty7 as ift
# Utility
import os
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import spherical_to_cartesian
from astropy.coordinates import cartesian_to_spherical
from astropy import constants as cons
import struct
# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# Directory paths
rundir    = 'runs/'
figpath   = 'figures/'
fieldpath = 'arrayfields/'
logdir    = 'log/'

# Gobal testing constants
import astropy.units as u
MHz   = 1e6 / u.s
GHz   = 1e9 / u.s
Ndata = 50
observing_frequency = 74*MHz
dunit = u.K/u.kpc
global_dist_error = 0.001
global_brightness_error = 0.01
key = ('average_los_brightness', 0.07400000000000001, 'tab', None) # simulation key

print("\n")



MHz   = 1e6 / u.s
GHz   = 1e9 / u.s
me    = cons.m_e.cgs
c     = cons.c.cgs
kb    = cons.k_B.cgs
electron = cons.e.gauss

# Units definitions since astropy doesnt handle [B]_cgs well
gauss_B  = (u.g/u.cm)**(0.5)/u.s
equiv_B  = [(u.G, gauss_B, lambda x: x, lambda x: x)]
ugauss_B = 1e-6 * gauss_B


#%% Plotting routine
#==============================================================================================================================================================
# Better plotting routine that also plots the inside voxels taken from:
# https://stackoverflow.com/questions/40853556/3d-discrete-heatmap-in-matplotlib

def cuboid_data(center, size=(1,1,1)):
    # code taken from
    # http://stackoverflow.com/questions/30715083/python-plotting-a-wireframe-3d-cuboid?noredirect=1&lq=1
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  # x coordinate of points in inside surface
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]    # y coordinate of points in inside surface
    z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                # z coordinate of points in inside surface
    return np.array(x), np.array(y), np.array(z)

def plotCubeAt(pos, alpha, ax=None):
    # Plotting N cube elements at position pos
    if ax !=None:
        X, Y, Z = cuboid_data( (pos[0],pos[1],pos[2]) )
        ax.plot_surface(X, Y, Z, color='blue', rstride=1, cstride=1, alpha=alpha, linewidth=0)

def plotMatrix(ax, x, y, z, data):
    normdata = data/np.max(data)
    alphadat = normdata.value/5 # make alpha difference more pronounced
    for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                for k, zi, in enumerate(z):
                    if alphadat[i,j,k] > 1e-4:
                        plotCubeAt(pos=(xi, yi, zi), alpha=alphadat[i,j,k],  ax=ax)         

def plot_LOScubeV2(grid,data,starts,ends,fname):

    print("Plotting and saving LOS-cubeV2 figure as: "+fname)

    # Setup xyz coordinates where the first voxel goes from 0,0,0 to 
    cbox = grid.box
    cres = grid.resolution
    xmax = (cbox[0][1]-cbox[0][0])
    ymax = (cbox[1][1]-cbox[1][0])
    zmax = (cbox[2][1]-cbox[2][0])      
    x = np.linspace(0,1,cres[0])*xmax/xmax.unit
    y = np.linspace(0,1,cres[1])*ymax/ymax.unit
    z = np.linspace(0,1,cres[2])*zmax/zmax.unit
    
    # Open canvas
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111, projection='3d')
    #ax = fig.add_axes([0.1, 0.1, 0.7, 0.8], projection='3d')
    #ax_cb = fig.add_axes([0.8, 0.3, 0.05, 0.45])


    # Plot all individual voxels
    plotMatrix(ax, x, y, z, data)

    # Setup lines and start/end points
    segm = []
    [segm.append(np.vstack((s,e))) for s,e in zip(starts/xmax.unit,ends/xmax.unit)]
    ax.scatter(starts[:,0],starts[:,1],starts[:,2])
    ax.scatter(ends[:,0],ends[:,1],ends[:,2])
    ax.add_collection3d(Line3DCollection(segments=segm,colors='k'))

    # Dresup the plot
    ax.set_title("Simulated emissivity cube with LOS collection")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Save figure
    plt.savefig(figpath+fname)
    plt.close('all') # not sure if need but just to be sure

def plot_hor_slice(data, grid, fname=' '):
    unit       = data.unit
    resolution = grid.resolution    
    
    # Take a horizontal and a vertical slice through the middle of the box
    hor_slice = data[:,:,int(resolution[2]/2)]/unit 
    x = grid.x[:,0,0]/u.kpc
    y = grid.y[0,:,0]/u.kpc
    z = grid.z[0,0,:]/u.kpc
    # Make figure
    plt.contourf(x,y,hor_slice.T,20,cmap='Blues') #we need .T because matplotlib plots to SouthEast instead NorthEast
    plt.title("title")
    plt.ylabel('y kpc')
    plt.xlabel('x kpc')
    plt.colorbar()
    print("Saving figure")
    plt.savefig(figpath+fname)
    plt.close('all')

#==============================================================================================================================================================
#%% Define the Simulator class
class SpectralSynchrotronEmissivitySimulator(Simulator):
   
    # Class attributes
    SIMULATED_QUANTITIES = ['average_los_brightness']
    REQUIRED_FIELD_TYPES = ['magnetic_field','cosmic_ray_electron_density']
    OPTIONAL_FIELD_TYPES = ['cosmic_ray_electron_spectral_index']
    ALLOWED_GRID_TYPES   = ['cartesian']
    
    def __init__(self,measurements,sim_config={'grid':None,'observer':None,'dist':None,'e_dist':None,'lat':None,'lon':None,'FB':None}):
        
        print("Initializing SynchtrotronEmissivitySimulator")
        # Send the Measurements to the parent class
        super().__init__(measurements) 
        
        # Stores class-specific attributes (and for now double definitions)
        for key in measurements.keys(): self.observing_frequency = key[1] * GHz
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
        if sim_config['e_dist'] != None:
            e_hIIdist  = sim_config['e_dist'].to(u.kpc)
        else:
            e_hIIdist = None
        behind     = np.where(np.array(sim_config['FB'])=='B')
        nlos       = len(hIIdist)
        
        # Remember the translation
        translation = np.array([xmax,ymax,zmax])/2 * unit
        translated_observer = sim_config['observer'] + translation

        # Cast start and end points in a Nifty compatible format
        starts = []
        for o in translated_observer:
            starts.append(np.full(shape=(nlos,), fill_value=o)*u.kpc)
        start_points = np.vstack(starts).T

        ends = []
        los  = spherical_to_cartesian(r=hIIdist, lat=lat, lon=lon)
        for i,axis in enumerate(los):
            ends.append(axis+translated_observer[i])
        end_points = np.vstack(ends).T

        # Do Front-Behind selection
        deltas = end_points - start_points
        clims  = box * np.sign(deltas) * unit
        clims[clims<0]=0 # if los goes in negative direction clim of xyz=0-plane
        with np.errstate(divide='ignore'):
            all_lambdas = (clims-end_points)/deltas   # possibly divide by zero 
        lambdas = np.min(np.abs(all_lambdas), axis=1) # will drop any inf here
        start_points[behind] = end_points[behind] + np.reshape(lambdas[behind], newshape=(np.size(behind),1))*deltas[behind]     
        
        # Final integration distances
        self.los_distances = np.linalg.norm(end_points-start_points, axis=1)
   
        # convenience for bugfixing:
        behindTF = np.zeros(nlos)*u.kpc
        behindTF[behind] = 1*u.kpc
        self.start_points = start_points
        self.behindTF = behindTF
        nstarts = self._make_nifty_points(start_points)
        nends   = self._make_nifty_points(end_points)

        self.response = ift.LOSResponse(self.domain, nstarts, nends, sigmas=e_hIIdist, truncation=3.) # domain doesnt know about its units but starts/ends do?

        # Just needed for testing can remove these in later version
        self.start = start_points
        self.end   = end_points
        self.box   = box
        self.grid  = grid
    
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
        # Check constant units and remove them for the calculation
        e     = (cons.e).gauss/(cons.e).gauss.unit
        me    = cons.m_e.cgs/cons.m_e.cgs.unit
        c     = cons.c.cgs/cons.c.cgs.unit
        # Check argument units and remove them for the calculation        
        vobs  = self.observing_frequency.to(1/u.s)*u.s
        ncre  = ncre.to(u.cm**-3)*u.cm**3
        Bper  = Bper.to(u.G)/u.G
        # Handle two spectral index cases:
	# -> fieldlist is not provided on initialization so we opt for a runtime check of alpha type
        try: # alpha is a constant spectral index globally 
            alpha = self.field_parameter_values['cosmic_ray_electron_density']['spectral_index']
        except: pass
        try: # alpha is an instance of a 3D scalar field
            alpha = self.fields['cosmic_ray_electron_spectral_index']
        except: pass
        # Calculate emissivity grid
        fraction1 = (np.sqrt(3)*e**3*ncre/(8*np.pi*me*c**2))
        fraction2 = (4*np.pi*vobs*me*c/(3*e))
        integral  = self._spectral_integralF( (-alpha-3)/2 )
        emissivity = fraction1 * fraction2**((1+alpha)/2) * Bper**((1-alpha)/2) * integral
        assert emissivity.unit == u.dimensionless_unscaled
        # Return emissivty and restore correct units
        return fraction1*fraction2**((1+alpha)/2)*Bper**((1-alpha)/2)*integral * u.kg/(u.m*u.s**2)
    
    def _project_to_perpendicular(self, vectorfield):
        """
        This function takes in a 3D vector field and uses the initialized unitvector_grid
        to project each vector on the unit vector perpendicular to the los to that position.
        """
        v_parallel      = np.zeros(np.shape(vectorfield)) * vectorfield.unit
        amplitudes      = np.sum(vectorfield * self.unitvector_grid, axis=3)
        v_parallel[:,:,:,0]  = amplitudes * self.unitvector_grid[:,:,:,0]
        v_parallel[:,:,:,1]  = amplitudes * self.unitvector_grid[:,:,:,1]
        v_parallel[:,:,:,2]  = amplitudes * self.unitvector_grid[:,:,:,2]
        v_perpendicular      = vectorfield - v_parallel
        v_perp_amplitude     = np.sqrt(np.sum(v_perpendicular*v_perpendicular,axis=3))
        return v_perp_amplitude

    def simulate(self, key, coords_dict, realization_id, output_units): 
        # Acces field data
        ncre_grid = self.fields['cosmic_ray_electron_density']  # in units cm^-3
        B_grid    = self.fields['magnetic_field'] # fixing is now done inside emissivity calculation
        # Project to perpendicular component to line of sight
        Bperp_amplitude_grid = self._project_to_perpendicular(B_grid)
        plot_hor_slice(data=Bperp_amplitude_grid,grid=self.grid,fname='JF12perpamp.png')
        #plot_LOScubeV2(grid=self.grid,data=Bperp_amplitude_grid,starts=self.start,ends=self.end,fname='losboxtest_B.png')
        # Calculate grid of emissivity values
        emissivity_grid = self._spectral_total_emissivity(Bperp_amplitude_grid, ncre_grid)
        plot_LOScubeV2(grid=self.grid,data=emissivity_grid,starts=self.start,ends=self.end,fname='losboxtest.png')
        plot_hor_slice(data=emissivity_grid,grid=self.grid,fname='brightness.png')
        # Do the los integration on the domain defined in init with the new emissivity grid
        HII_LOSemissivities = self.response(ift.Field(self.domain, emissivity_grid)).val_rw()
        HII_LOSemissivities *= emissivity_grid.unit * u.kpc # restore units: domain is assumed to be in kpc
        # Need units to be in K/kpc, average brightness temperature allong the line of sight
        HII_LOSbrightness = c**2/(2*kb*self.observing_frequency**2)*HII_LOSemissivities/self.los_distances
        return HII_LOSbrightness

#%% Controle funtions
#===================================================================================

def randrange(minvalue,maxvalue,Nvalues,seed=3145):
    np.random.seed(seed)
    """Returns uniform random values bewteen min and max"""
    return (maxvalue-minvalue)*np.random.rand(Nvalues)+minvalue

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

def get_label_FB(Ndata, seed=31415):
    np.random.seed(seed)
    NF = int(0.2*Ndata) # 20% of all measurements are front measurements
    F  = np.full(shape=(NF), fill_value='F', dtype=str)
    B  = np.full(shape=(Ndata-NF), fill_value='B', dtype=str)
    FB = np.hstack((F,B))
    np.random.shuffle(FB) 
    return FB

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
                    'b_arm_6': -4.2, 'b_arm_7': .0, 'b_ring': .0, 'h_disk': .4, 'w_disk': .27,
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

#%% Get time statistics
#===================================================================================

def do_simulation():
    Bfield, cre, config, mea = produce_basic_setup()
    field_list = [Bfield, cre]
    test_sim   = SpectralSynchrotronEmissivitySimulator(measurements=mea, sim_config=config) 
    simulation = test_sim(field_list)
    key = ('average_los_brightness', 0.07400000000000001, 'tab', None)
    sim_brightness = simulation[key].data[0] * simulation[key].unit
    print(sim_brightness)
do_simulation()