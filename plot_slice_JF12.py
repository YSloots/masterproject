#%% Imports and settings

# Utility
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import cartesian_to_spherical
import struct

# IMAGINE imports
import imagine as img
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12
from imagine.fields.field_utility import MagneticFieldAdder
from imagine.fields.field_utility import ArrayMagneticField

# Directory paths
figpath   = 'figures/'
fieldpath = 'arrayfields/'

#===========================================================================
#%% Functions

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
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5), sharey = False)
    n = 0
    for ax in axes.flat:
        #im = ax.contourf(x, y, slices[n], 40, cmap='RdGy', vmin = -range, vmax = range)
        im = ax.contourf(coords[n][0], coords[n][1], slices[n],
    		     3, cmap='Blues', vmin = 0, vmax = maxvalue)
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

def get_unit_vectors(observer, grid):
        unit = observer.unit 
        unitvectors = []
        for x,y,z in zip(grid.x.ravel()/unit,grid.y.ravel()/unit,grid.z.ravel()/unit):
            v = np.array([x,y,z])-observer/unit
            normv = np.linalg.norm(v)
            if normv == 0: # special case where the observer is inside one of the grid points
                unitvectors.append(v)
            else:
                unitvectors.append(v/normv)
        # Save unitvector grid as class attribute
        return np.reshape(unitvectors, tuple(grid.resolution)+(3,))

def project_to_perpendicular(vectorfield, unitvector_grid):
    v_parallel      = np.zeros(np.shape(vectorfield)) * vectorfield.unit
    amplitudes      = np.sum(vectorfield * unitvector_grid, axis=3)
    v_parallel[:,:,:,0]  = amplitudes * unitvector_grid[:,:,:,0]
    v_parallel[:,:,:,1]  = amplitudes * unitvector_grid[:,:,:,1]
    v_parallel[:,:,:,2]  = amplitudes * unitvector_grid[:,:,:,2]
    v_perpendicular      = vectorfield - v_parallel
    v_perp_amplitude     = np.sqrt(np.sum(v_perpendicular*v_perpendicular,axis=3))
    return v_perp_amplitude

def load_JF12rnd(label = 1, shape=(40,40,10,3)):
#def load_JF12rnd(label = 1, shape=(100,100,10,3)):
    print("Loading: "+fieldpath+"brnd_{}.bin".format(int(label)))
    with open(fieldpath+"brnd_{}.bin".format(int(label)), "rb") as f:
    #with open(fieldpath+"brnd_local.bin".format(int(label)), "rb") as f:
        arr = f.read()
        arr = struct.unpack("d"*(len(arr)//8), arr[:])
        arr = np.asarray(arr).reshape(shape)
    return arr

def plot_JF12_orientation():
    """Just a testing function!!!!"""
    # Setup coordinate grid
    cartesian_grid = img.fields.UniformGrid(box=[[-15*u.kpc, 15*u.kpc],
                                                [-15*u.kpc, 15*u.kpc],
                                                [-5*u.kpc, 5*u.kpc]],
                                                resolution = [3,3,3])
    # Get field data
    Bfield = WrappedJF12(grid=cartesian_grid,        
                    parameters = {'b_arm_1': .1, 'b_arm_2': 3.0, 'b_arm_3': -.9, 'b_arm_4': -.8, 'b_arm_5': -2.,
                        'b_arm_6': -4.2, 'b_arm_7': .0, 'b_ring': .1, 'h_disk': .4, 'w_disk': .27,
                        'Bn': 1.4, 'Bs': -1.1, 'rn': 9.22, 'rs': 16.7, 'wh': .2, 'z0': 5.3, 'B0_X': 4.6,
                        'Xtheta_const': 49, 'rpc_X': 4.8, 'r0_X': 2.9, }) # default parameters
    Bdata = Bfield.get_data()
    unit  = Bdata.unit
    Bdata = np.zeros(shape=(3,3,3,3))*unit
    print('\n')
    print(Bdata)
    Bdata[:,:,1,0] = 1*unit
    print(Bdata[0,0,0,:])
    print(Bdata[0,0,1,:])
    
    #make a left spiral
    Bdata[2,1,1,0] = 10*unit #[2=largest x, 1=middle y, 1=middle z, 0=x component]
    Bdata[2,2,1,0] = 9*unit

    #print(Bdata[np.abs(Bdata/Bdata.unit) > 1e-12])
    #print(np.mean(Bdata))
    #print(np.sum(np.isnan(Bdata)))
    #Bdata[:,:,:,2] = np.zeros(cartesian_grid.resolution) # set field in z-direction to zero
    
    azimuth  = get_unit_vectors(observer=np.array([0,0,0])*u.kpc, grid=cartesian_grid)
    Bazimuth = project_to_perpendicular(vectorfield=Bdata, unitvector_grid=azimuth)
    #Bazimuth = Bdata[:,:,:,0]

    # Make figure
    x = cartesian_grid.x[:,0,0]/u.kpc
    y = cartesian_grid.y[0,:,0]/u.kpc
    z = cartesian_grid.z[0,0,:]/u.kpc
    hor_slice = Bazimuth[:,:,int(len(z)/2)]/Bazimuth.unit
    #plt.contourf(x,y,hor_slice.T,10,cmap='Blues') #we need .T because matplotlib plots to SouthEast instead NorthEast
    plt.imshow(hor_slice)
    plt.yticks(np.arange(len(y)),y)
    plt.xticks(np.arange(len(x)),x)
    plt.ylabel('y kpc')
    plt.xlabel('x kpc')
    plt.colorbar()

    print("Saving figure")
    plt.savefig(figpath+'JF12regular_test.png')
    plt.close('all')
#plot_JF12_orientation()

#===========================================================================
#%% Plot single JF12 arms (+ always arm 8)

def get_single_arm_parameters(non_zero_arm):
    parameters = {'b_arm_1': 0, 'b_arm_2': 0, 'b_arm_3': 0, 'b_arm_4': 0, 'b_arm_5': 0,
    'b_arm_6': 0, 'b_arm_7': 0, 'b_ring': 1, 'h_disk': .4, 'w_disk': .27,
    'Bn': 0, 'Bs': 0, 'rn': 9.22, 'rs': 16.7, 'wh': .2, 'z0': 5.3, 'B0_X': 0,
    'Xtheta_const': 49, 'rpc_X': 4.8, 'r0_X': 2.9, } # default parameters

    parameters['b_arm_{}'.format(non_zero_arm)] = 1
    print(parameters)
    return parameters

def plot_JF12_regular_arms():
    # Setup coordinate grid
    cartesian_grid = img.fields.UniformGrid(box=[[-15*u.kpc, 15*u.kpc],
                                                [-15*u.kpc, 15*u.kpc],
                                                [-5*u.kpc, 5*u.kpc]],
                                                resolution = [100,100,3])
    # Get field data
    for arm_index in np.arange(1,8):
        parameters = get_single_arm_parameters(non_zero_arm=arm_index)
        Bfield = WrappedJF12(grid=cartesian_grid, parameters=parameters)
        Bdata = Bfield.get_data()
        azimuth  = get_unit_vectors(observer=np.array([0,0,0])*u.kpc, grid=cartesian_grid)
        Bazimuth = project_to_perpendicular(vectorfield=Bdata, unitvector_grid=azimuth)
        # Make figure
        x = cartesian_grid.x[:,0,0]/u.kpc
        y = cartesian_grid.y[0,:,0]/u.kpc
        z = cartesian_grid.z[0,0,:]/u.kpc
        hor_slice = Bazimuth[:,:,int(len(z)/2)]/Bazimuth.unit
        plt.contourf(x,y,hor_slice.T,20,cmap='Blues') #we need .T because matplotlib plots to SouthEast instead NorthEast
        plt.title("JF12 magnetic field azimuthal component")
        plt.ylabel('y kpc')
        plt.xlabel('x kpc')
        plt.colorbar()
        print("Saving figure arm {}".format(arm_index))
        plt.savefig(figpath+'JF12regular_arm{}.png'.format(arm_index))
        plt.close('all')
#plot_JF12_regular_arms()

#===========================================================================
#%% Make plot for regular JF12 field

def plot_JF12_regular_arms():
    # Setup coordinate grid
    cartesian_grid = img.fields.UniformGrid(box=[[-15*u.kpc, 15*u.kpc],
                                                [-15*u.kpc, 15*u.kpc],
                                                [-5*u.kpc, 5*u.kpc]],
                                                resolution = [100,100,3])
    # Get field data
    parameters = {'b_arm_1': .1, 'b_arm_2': 3.0, 'b_arm_3': -.9, 'b_arm_4': -.8, 'b_arm_5': -2.,
    'b_arm_6': -4.2, 'b_arm_7': .0, 'b_ring': .1, 'h_disk': .4, 'w_disk': .27,
    'Bn': 1.4, 'Bs': -1.1, 'rn': 9.22, 'rs': 16.7, 'wh': .2, 'z0': 5.3, 'B0_X': 4.6,
    'Xtheta_const': 49, 'rpc_X': 4.8, 'r0_X': 2.9, } # default parameters
    Bfield = WrappedJF12(grid=cartesian_grid, parameters=parameters)
    Bdata = Bfield.get_data()
    azimuth  = get_unit_vectors(observer=np.array([0,0,0])*u.kpc, grid=cartesian_grid)
    Bazimuth = project_to_perpendicular(vectorfield=Bdata, unitvector_grid=azimuth)
    # Make figure
    x = cartesian_grid.x[:,0,0]/u.kpc
    y = cartesian_grid.y[0,:,0]/u.kpc
    z = cartesian_grid.z[0,0,:]/u.kpc
    hor_slice = Bazimuth[:,:,int(len(z)/2)]/Bazimuth.unit
    plt.contourf(x,y,hor_slice.T,20,cmap='Blues') #we need .T because matplotlib plots to SouthEast instead NorthEast
    plt.title("JF12 magnetic field azimuthal component")
    plt.ylabel('y kpc')
    plt.xlabel('x kpc')
    plt.colorbar()
    print("Saving figure")
    plt.savefig(figpath+'JF12regular.png')
    plt.close('all')
#plot_JF12_regular()

#===========================================================================
#%% Make plot JF12turbulent

def plot_JF12_turbulent():
    # Setup coordinate grid
    cartesian_grid = img.fields.UniformGrid(box=[[-20*u.kpc, 20*u.kpc],
                                                [-20*u.kpc, 20*u.kpc],
                                                [-2*u.kpc, 2*u.kpc]],
                                                resolution = [40,40,10])
    # Setup turbulent grid
    #generate_JF12rnd(grid=cartesian_grid) # calls Hammurabi (not implemented yet)
    for label in [1,2,3,4,5]:
        Barray = load_JF12rnd(label = label)
        Bfield = ArrayMagneticField(grid = cartesian_grid,
                                    parameters = {'array_field': Barray*1e6*u.microgauss,
                                                'array_field_amplitude': 1.0})
        Bdata    = Bfield.get_data()
        azimuth  = get_unit_vectors(observer=np.array([0,0,0])*u.kpc, grid=cartesian_grid)
        Bazimuth = project_to_perpendicular(vectorfield=Bdata, unitvector_grid=azimuth)
        # Make figure
        x = cartesian_grid.x[:,0,0]/u.kpc
        y = cartesian_grid.y[0,:,0]/u.kpc
        z = cartesian_grid.z[0,0,:]/u.kpc
        hor_slice = Bazimuth[:,:,int(len(z)/2)]/Bazimuth.unit
        plt.contourf(x,y,hor_slice,20,cmap='Blues')
        plt.title('JF12 turbulent field {}'.format(int(label)))
        plt.ylabel('y kpc')
        plt.xlabel('x kpc')
        plt.colorbar()
        print("Saving figure")
        plt.savefig(figpath+'JF12turbulent{}.png'.format(int(label)))
        plt.close('all')
plot_JF12_turbulent()

#===========================================================================
#%% Make plot for JF12regular + JF12turbulent

def plot_JF12_total():
    # Setup coordinate grid
    cartesian_grid = img.fields.UniformGrid(box=[[-20*u.kpc, 20*u.kpc],
                                                [-20*u.kpc, 20*u.kpc],
                                                [-2*u.kpc, 2*u.kpc]],
                                                resolution = [40,40,10])

    # Setup turbulent grid
    #generate_JF12rnd(grid=cartesian_grid) # calls Hammurabi (not implemented yet)
    Barray  = load_JF12rnd()
    print("\nBarray directly after loading:\n",Barray[39,0,0,:])
    beta   = 0.0 # Barray scale
    Btotal = MagneticFieldAdder(grid=cartesian_grid,
                                field_1=WrappedJF12,
                                field_2=ArrayMagneticField,
                                parameters = {  'array_field_amplitude':beta,
                                                'b_arm_1': .1, 'b_arm_2': 3.0, 'b_arm_3': -.9, 'b_arm_4': -.8, 'b_arm_5': -2.,
                                                'b_arm_6': -4.2, 'b_arm_7': .0, 'b_ring': .1, 'h_disk': .4, 'w_disk': .27,
                                                'Bn': 1.4, 'Bs': -1.1, 'rn': 9.22, 'rs': 16.7, 'wh': .2, 'z0': 5.3, 'B0_X': 4.6,
                                                'Xtheta_const': 49, 'rpc_X': 4.8, 'r0_X': 2.9, 
                                                'array_field': Barray*1e6*u.microgauss})
    Bdata    = Btotal.get_data()
    azimuth  = get_unit_vectors(observer=np.array([0,0,0])*u.kpc, grid=cartesian_grid)
    Bazimuth = project_to_perpendicular(vectorfield=Bdata, unitvector_grid=azimuth)
    # Make figure
    x = cartesian_grid.x[:,0,0]/u.kpc
    y = cartesian_grid.y[0,:,0]/u.kpc
    z = cartesian_grid.z[0,0,:]/u.kpc
    hor_slice = Bazimuth[:,:,int(len(z)/2)]/Bazimuth.unit
    plt.contourf(x,y,hor_slice,10,cmap='Blues')
    plt.title('JF12 total horizontal field, beta = {}'.format(beta))
    plt.ylabel('y kpc')
    plt.xlabel('x kpc')
    plt.colorbar()
    print("Saving figure")
    plt.savefig(figpath+'JF12total_horizontal{}.png'.format(beta))
    plt.close('all')
#plot_JF12_total()














