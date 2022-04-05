# Simulator dependancies
import numpy as np
import imagine as img
from imagine import fields
from imagine.simulators import Simulator
from scipy.special import gamma as gammafunc
import nifty7 as ift

from imagine.fields.hamx import BregLSA, BregLSAFactory
from imagine.fields.hamx import CREAna, CREAnaFactory

import astropy.units as u
from astropy.coordinates import spherical_to_cartesian
from astropy.coordinates import cartesian_to_spherical
from astropy import constants as cons
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

# Script dependancies
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# Run directory for pipeline
rundir  = 'runs/mockdata'
figpath = 'figures/'


print('\nRunning test_lost_pipeline.py\n\n')

#%% Diagnostic and plotting functions

def plot_LOScube(axes, starts, ends, fname, data=None):
    print("Plotting and saving LOS-cube setup as: "+fname)    
    
    # The integration domian has to have integer limits in order to plot correctly!
    axes = axes.astype(int)
    unit = starts.unit
    
    # Open canvas
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
        
    # Setup cube
    cube      = np.ones(tuple(axes), dtype=bool)
    colors    = np.empty(list(axes)+[4], dtype=np.float32)
    if data == None:
        alpha     = 0.3
        colors[:] = [0,0,1,alpha]
    else:
        print("Plotting with emissivity data ... ")
        normdata = data/np.max(data)
        print(np.max(normdata))
        colors[:]       = [0,0,1,0]
        colors[:,:,:,3] = normdata**3/2 # moke structure better visible
    ax.voxels(cube, facecolors=colors)

    # Setup lines and start/end points
    segm = []
    [segm.append(np.vstack((s,e))) for s,e in zip(starts/unit,ends/unit)]
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
    alphadat = normdata.value/100 # make alpha difference more pronounced
    for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                for k, zi, in enumerate(z):
                    if alphadat[i,j,k] > 1e-3:
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
    REQUIRED_FIELD_TYPES = ['magnetic_field','cosmic_ray_electron_density']
    #OPTIONAL_FIELD_TYPES = ['dummy','magnetic_field'] #aslo cant do this
    # Do we want the simulator to be able to acces certain Hamx fields? Is this possible?
    
    #REQUIRED_FIELD_TYPES = []
    #OPTIONAL_FIELD_TYPES = ['dummy','magnetic_field',
    #                        'cosmic_ray_electron_density']
    #Shouldnt do this because we need a parameter spectral_index in the CRE field
    
    ALLOWED_GRID_TYPES   = ['cartesian']
    
    def __init__(self,measurements,sim_config={'grid':None,'observer':None,'dist':None,'e_dist':None,'lat':None,'lon':None,'FB':None},plotting=False):
        
        print("Initializing SynchtrotronEmissivitySimulator")
        # Send the Measurements to the parent class
        super().__init__(measurements) 
        
        # Write assert and test functions to make sure stuf is correct
        # - grid.unit == observer.unit
        # - grid = instance of cartesian grid
        # - all HII regions should be located within the simulation box        
        
        
        # Asses field types and set data-acces function
        #self.get_field_data()        
        
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
        
        if plotting:
            #plot_LOScube(axes=box, starts=start_points, ends=end_points, fname='losbox_testsim.pdf')        
            plot_LOScube(axes=grid.resolution,starts=start_points,ends=end_points,fname='losbox_testsim.pdf')        
        
        self.los_distances = np.linalg.norm(end_points-start_points, axis=1)
   
        nstarts = self._make_nifty_points(start_points)
        nends   = self._make_nifty_points(end_points)
        #print(translated_observer)
        #print(np.sum(self.los_distances > xmax/2*unit))
        #print(start_points[self.los_distances > xmax/2*unit])
        #print(end_points[self.los_distances > xmax/2*unit])
        #print("Min-max of losdist: ", np.min(self.los_distances), np.max(self.los_distances))
        self.response = ift.LOSResponse(self.domain, nstarts, nends) # domain doesnt know about its units but starts/ends do?

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
        alpha = self.field_parameter_values['cosmic_ray_electron_density']['spectral_index']
        vobs  = self.observing_frequency.to(1/u.s)*u.s
        ncre  = ncre.to(u.cm**-3)*u.cm**3
        Bper  = Bper.to(u.G)/u.G
        # Calculate emissivity grid
        fraction1 = (np.sqrt(3)*e**3*ncre/(8*np.pi*me*c**2))
        fraction2 = (4*np.pi*vobs*me*c/(3*e))
        integral  = self._spectral_integralF( (-alpha-3)/2 )
        emissivity = fraction1 * fraction2**((1+alpha)/2) * Bper**((1-alpha)/2) * integral
        assert emissivity.unit == u.dimensionless_unscaled
        # Return emissivty and restore correct units
        return fraction1*fraction2**((1+alpha)/2)*Bper**((1-alpha)/2)*integral * u.kg/(u.m*u.s**2)
    
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
        
        # Acces field data
        alpha     = self.field_parameter_values['cosmic_ray_electron_density']['spectral_index']
        ncre_grid = self.fields['cosmic_ray_electron_density']  # in units cm^-3
        #B_grid    = self.fields['magnetic_field']/u.microgauss * ugauss_B # fixing astropy cgs units
        B_grid    = self.fields['magnetic_field'] # fixing is now done inside emissivity calculation        
        
        # Project to perpendicular component along line of sight
        Bperp_amplitude_grid = self._project_to_perpendicular(B_grid)        
        
        # Calculate grid of emissivity values
        emissivity_grid = self._spectral_total_emissivity(Bperp_amplitude_grid, ncre_grid)
        #plot_LOScube(axes=self.grid.resolution,starts=self.start,ends=self.end,fname='losbox_emissivity_IMGModels.pdf')        
        #plot_LOScubeV2(grid=self.grid,data=emissivity_grid,starts=self.start,ends=self.end,fname='losbox2_emissivity_cutoffBestpdf.pdf')
        
        # Do the los integration on the domain defined in init with the new emissivity grid
        HII_LOSemissivities = self.response(ift.Field(self.domain, emissivity_grid)).val_rw()
        HII_LOSemissivities *= emissivity_grid.unit * u.kpc # restore units: domain is measured in kpc
        
        # Need to convert to K/kpc units. Note that self.los_distances may not be correct when using e_dist for domain
        HII_LOSbrightness = c**2/(2*kb*self.observing_frequency**2)*HII_LOSemissivities/self.los_distances

        return HII_LOSbrightness # whatever is returned has to have the same shape as the object in measurements


#%% Make a simulated dataset with known model paramters


"""
print("Producing simple simulated dataset ...\n")

# Mock dataset
Ndata = 100
observing_frequency = 90*MHz
dunit = u.K/u.kpc
T     = np.zeros(Ndata)*dunit # placeholder
T_err = np.zeros(Ndata)*dunit# placeholder
lat   = 90*np.linspace(-1,1,Ndata)*u.deg
lon   = 360*np.linspace(0,1,Ndata)*u.deg*300

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
cre = fields.ConstantCosmicRayElectrons(grid=cartesian_grid,
                                        parameters={'density':1.0e-7*u.cm**-3,'spectral_index':-3.0})


# Magnetic Field Model
Bfield = fields.ConstantMagneticField(
    grid = cartesian_grid,
    parameters={'Bx': 6*u.microgauss,
                'By': 0*u.microgauss,
                'Bz': 0*u.microgauss})


observer = np.array([0,0,0])*u.kpc
hIIdist  = (box_size-2*u.kpc)*np.random.rand(Ndata) + 1*u.kpc # uniform between [1, max-1] kpc
#print("\nSimulation domain: \n", cartesian_grid.box)
#print("Min-max of hIIdist: ", np.min(hIIdist), np.max(hIIdist))
dist_err = hIIdist/10
NF = int(0.2*Ndata) # 20% of all measurements
F  = np.full(shape=(NF), fill_value='F', dtype=str)
B  = np.full(shape=(Ndata-NF), fill_value='B', dtype=str)
FB = np.hstack((F,B))
np.random.shuffle(FB) # This seems to cause a bug with the nifty integration!!


config = {'grid'    :cartesian_grid,
          'observer':observer,
          'dist':hIIdist,
          'e_dist':dist_err,
          'lat':lat,
          'lon':lon,
          'FB':FB}
"""

#%% More complicated models


def randrange(minvalue,maxvalue,Nvalues):
    return (maxvalue-minvalue)*np.random.rand(Nvalues)+minvalue


print("Producing more realistic simulated dataset ...\n")


# Less redundancy in coordinate grid
xmax = 15*u.kpc
ymax = 15*u.kpc
zmax =  2*u.kpc
cartesian_grid = img.fields.UniformGrid(box=[[-xmax, xmax],
                                             [-ymax, ymax],
                                             [-zmax, zmax]],
                                             resolution = [31,31,31])

# Mock dataset
Ndata = 50
observing_frequency = 90*MHz
dunit = u.K/u.kpc
T     = np.zeros(Ndata)*dunit # placeholder
T_err = np.zeros(Ndata)*dunit # placeholder
x = randrange(-0.9*xmax,0.9*xmax,Ndata)
y = randrange(-0.9*ymax,0.9*ymax,Ndata)
z = randrange(-0.9*zmax,0.9*zmax,Ndata)
hIIdist, lat, lon = cartesian_to_spherical(x+8.5*u.kpc,y,z)
#hIIdist  = np.sqrt((x+8.5)**2+y**2+z**2)*u.kpc
observer = np.array([-8.5,0,0])*u.kpc
dist_err = hIIdist/10

# Define front or behind measurements
NF = int(0.2*Ndata) # 20% of all measurements
F  = np.full(shape=(NF), fill_value='F', dtype=str)
B  = np.full(shape=(Ndata-NF), fill_value='B', dtype=str)
FB = np.hstack((F,B))
np.random.shuffle(FB) # This seems to cause a bug with the nifty integration!!

# Save the full simulation configuration
config = {'grid'    :cartesian_grid,
          'observer':observer,
          'dist':hIIdist,
          'e_dist':dist_err,
          'lat':lat,
          'lon':lon,
          'FB':FB}

# Save placeholder dataset
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



# Fieldsetup:
cre = fields.PowerlawCosmicRayElectrons(grid=cartesian_grid,
                                        parameters = {'scale_radius':10*u.kpc,
                                                     'scale_height':1*u.kpc,
                                                     'central_density':1e-5*u.cm**-3,
                                                     'spectral_index':-3})


# Magnetic Field Model
Bfield = fields.ConstantMagneticField(
    grid = cartesian_grid,
    parameters={'Bx': 6*u.microgauss,
                'By': 0*u.microgauss,
                'Bz': 0*u.microgauss})



#%% Produce simulated data

test_sim   = SpectralSynchrotronEmissivitySimulator(mea, config, plotting=False) # create instance
simulation = test_sim([cre, Bfield])

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
likelihood = img.likelihoods.SimpleLikelihood(sim_mea)

#print("Running likelihood: \n")
#print(likelihood(simulation))

# Setup field factories
constCRE = fields.ConstantCosmicRayElectrons(grid=cartesian_grid,
                                             parameters={'density':1.0e-7*u.cm**-3,'spectral_index':0})
CRE_factory = img.fields.FieldFactory(field_class = constCRE, grid=cartesian_grid) # no active paramters
CRE_factory.active_parameters=('spectral_index',)
CRE_factory.priors = {'spectral_index':img.priors.FlatPrior(xmin=-4, xmax=-2.1)}

Bfield = fields.ConstantMagneticField(
    grid          = cartesian_grid,
    parameters    = {'Bx': 6*u.microgauss,'By': 0*u.microgauss,'Bz': 0*u.microgauss})
B_factory = img.fields.FieldFactory(field_class=Bfield, grid=cartesian_grid)
B_factory.active_parameters = ('Bx',)
B_factory.priors = {'Bx':img.priors.FlatPrior(xmin=0.*u.microgauss, xmax=10.*u.microgauss)} # what units?
factory_list = [CRE_factory, B_factory]


"""
# Setup pipeline
print("Starting pipeline ... \n")
pipeline = img.pipelines.MultinestPipeline( simulator     = los_simulator,
                                            run_directory = rundir,
                                            factory_list  = factory_list,
                                            likelihood    = likelihood,
                                            )
pipeline.sampling_controllers = {'evidence_tolerance': 0.5, 'n_live_points': 200}

# Run pipeline!
results = pipeline()

print(dir(results))
"""