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
ne_exponential_profile = fields.ExponentialThermalElectrons(
	grid       = cartesian_grid,
	parameters = {'central_density':1.0*u.cm**-3,
                   'scale_radius':5.0*u.kpc,
                   'scale_height':1.0*u.kpc})


ne_data   = ne_exponential_profile.get_data() * u.cm**3
#ne_data   = ne_exponential_profile.get_data().to_value()
#astropyunits are in the way of converting to floats how is this handled by other controurplots in imagine?

# Plot slices of the field
plot_slices(ne_data, cartesian_grid, path+'thermalelectrons.png')



#%% Now inspect CRE grid constant index


CRE_exponential = fields.PowerlawCosmicRayElectrons(
    grid            = cartesian_grid,
    parameters      ={'scale_radius':5.0*u.kpc,
                      'scale_height':1.0*u.kpc,
                      'spectral_index': -3})

# Acces and plot profiles
CRE_data = CRE_exponential.get_data()#*u.cm**3
#plot_slices(CRE_data, cartesian_grid, path+'comsicrayelectrons.png')

# Check spectral index
#print(dir(CRE_exponential),'\n')
print(CRE_exponential.parameters['spectral_index'],'\n')



#%% CRE grid with function for spectral index


def spectral_index_function(x,y,z):
    # index range
    amax = 2
    amin = 4
    # galaxy scale
    z_scale = 2
    r_scale = 10
    # now scale profile to square
    r           = np.sqrt(x**2 + y**2)
    scaled_dist = np.sqrt((z/z_scale)**2 + (r/r_scale)**2)/2
    return -(amin + (amax-amin)*scaled_dist)

alpha = spectral_index_function

CRE_exponential = fields.PowerlawCosmicRayElectrons(
    grid            = cartesian_grid,
    parameters      ={'scale_radius':5.0*u.kpc,
                      'scale_height':1.0*u.kpc,
                      'spectral_index': alpha})

# Acces and plot profiles
CRE_data = CRE_exponential.get_data()#*u.cm**3
#plot_slices(CRE_data, cartesian_grid, path+'comsicrayelectrons.png')

# Check spectral index
print(dir(CRE_exponential),'\n')
print(CRE_exponential.parameters['spectral_index'],'\n')
print(CRE_exponential.spectral_index_grid)

cube_alphas = CRE_exponential.spectral_index_grid
x = cartesian_grid.x[:,0,0]/u.kpc
z = cartesian_grid.z[0,0,:]/u.kpc
y = cartesian_grid.z[0,:,0]/u.kpc
slice_alphas = cube_alphas[:,int(len(y)/2),:]

plt.figure(figsize=(20,5))
plt.contourf(x,z,slice_alphas, 40, cmap='YlOrBr')
plt.colorbar()
plt.savefig(path+'spectral_index_grid.png')
plt.close('all')

#%%




















