import imagine as img
import numpy as np
import astropy.units as u
from imagine.fields.hamx.breg_jf12 import BregJF12
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12
import matplotlib.pyplot as plt
from astropy.coordinates import cartesian_to_spherical

figpath = 'figures/'
print('\n')

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
    
#%%

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


# Setup coordinate grid
cartesian_grid = img.fields.UniformGrid(box=[[-15*u.kpc, 15*u.kpc],
                                             [-15*u.kpc, 15*u.kpc],
                                             [-5*u.kpc, 5*u.kpc]],
                                             resolution = [50,50,50])
# Get field data
Bfield = WrappedJF12(grid=cartesian_grid) # default parameters
Bdata  = Bfield.get_data()
print(Bdata)
print(Bdata[Bdata/Bdata.unit > 1e-12])
print(np.mean(Bdata))
#print(np.sum(np.isnan(Bdata)))
#Bdata[:,:,:,2] = np.zeros(cartesian_grid.resolution) # set field in z-direction to zero
azimuth  = get_unit_vectors(observer=np.array([0,0,0])*u.kpc, grid=cartesian_grid)
Bazimuth = project_to_perpendicular(vectorfield=Bdata, unitvector_grid=azimuth)


# Make figure
x = cartesian_grid.x[:,0,0]/u.kpc
y = cartesian_grid.y[0,:,0]/u.kpc
z = cartesian_grid.z[0,0,:]/u.kpc
hor_slice = Bazimuth[:,:,int(len(z)/2)]/Bazimuth.unit
plt.contourf(x,y,hor_slice,10,cmap='Blues')
plt.title('JF12 magnetic field perpendicular LOS component')
plt.ylabel('y kpc')
plt.xlabel('x kpc')
plt.colorbar()

print("Saving figure")
plt.savefig(figpath+'JF12_horizontalsliceGalCentr.png')
plt.close('all')



















