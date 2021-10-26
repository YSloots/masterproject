import imagine as img
import numpy as np
import astropy.units as u
from imagine import fields
import matplotlib.pyplot as plt

#%% Functions and global definitions

path = 'figures/'

def plot_slices(data, grid, fname=' '):
    
    # -> there must be a better way to slice this instead of using the index labeling
    hor_slice = data[:,:,15] 
    ver_slice = data[15,:,:]
    x = grid.x[:,0,0]/u.kpc
    y = grid.y[0,:,0]/u.kpc
    z = grid.z[0,0,:]/u.kpc

    titles = ['xy-slice z=0','yz-slice x=0']
    slices = [hor_slice, ver_slice]
    coords = [[x,y], [x,z]]

    maxvalue = np.max(data)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,5), sharey = False)
    n = 0
    for ax in axes.flat:
        #im = ax.contourf(x, y, slices[n], 40, cmap='RdGy', vmin = -range, vmax = range)
        im = ax.contourf(coords[n][0], coords[n][1], slices[n],
    		     40, cmap='YlOrBr', vmin = 0, vmax = maxvalue)
        ax.title.set_text(titles[n])
        ax.set_xlabel('kpc')
        ax.set_ylabel('kpc')
        n += 1
        
        
    # fake subplot for colorbar
    fakez = np.zeros((len(y),len(x)))
    fakez[0,0] = 0 # fake pixels for colorbar
    fakez[0,1] = maxvalue
    ax4 = fig.add_subplot(1,3,1)
    im = ax4.contourf(x, y, fakez, 40, cmap='YlOrBr', vmin = 0, vmax = maxvalue)
    cbar_ax = fig.add_axes([0.92, 0.10, 0.01, 0.8])
    cbar = plt.colorbar(im, cax = cbar_ax)
    cbar.set_label('cm**-3', rotation=0)
    plt.delaxes(ax4)
    
    fig.savefig(fname)
    plt.close('all')
    
    return
    

#%% First try a normal thermal electron grid

# Setup coordinate grid
cartesian_grid = img.fields.UniformGrid(box=[[-15*u.kpc, 15*u.kpc],
                                             [-15*u.kpc, 15*u.kpc],
                                             [-5*u.kpc, 5*u.kpc]],
                                             resolution = [31,31,31])


# ==== First try the usual thermal electron field ====
# Setup field
ne_exponential = fields.ExponentialThermalElectrons(
	grid       = cartesian_grid,
	parameters = { 'central_density':1.0*u.cm**-3,'scale_radius':5.0*u.kpc,'scale_height':1.0*u.kpc})

ne_data   = ne_exponential.get_data() * u.cm**3
#astropyunits are in the way of converting to floats how is this handled by other controurplots in imagine?

# Plot slices of the field
plot_slices(ne_data, cartesian_grid, path+'thermalelectrons.png')

print("hi")


#%% Now inspect CRE grid



CRE_exponential = fields.ExponentialCosmicRayElectrons(
    grid      = cartesian_grid,
    parameters ={'central_density':1.0*u.cm**-3,
                       'scale_radius':5.0*u.kpc,
                       'scale_height':1.0*u.kpc})

CRE_data = CRE_exponential.get_data()*u.cm**3

plot_slices(CRE_data, cartesian_grid, path+'comsicrayelectrons.png')






























