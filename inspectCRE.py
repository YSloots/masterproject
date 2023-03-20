import imagine as img
import numpy as np
import astropy.units as u
from imagine import fields
import matplotlib.pyplot as plt

#%% Functions and global definitions

figpath = 'figures/'
print('\n')

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
    		     20, cmap='Blues', vmin = 0, vmax = maxvalue)
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

def pretty_slice_plot(data, grid, fname=' '):
    unit       = data.unit
    resolution = grid.resolution    
    
    # Take a horizontal and a vertical slice through the middle of the box
    #print("Taking slicing index", int(resolution[2]/2))
    hor_slice = data[:,:,int(resolution[2]/2)]/unit 
    ver_slice = data[int(resolution[0]/2),:,:]/unit
    x = grid.x[:,0,0]/u.kpc
    y = grid.y[0,:,0]/u.kpc
    z = grid.z[0,0,:]/u.kpc
    maxvalue = np.max(data)/unit

    """
    from matplotlib import gridspec
    gs = gridspec.GridSpec(2,2, hspace=0.1, height_ratios= [4,1])
    
    # first graph
    axes = plt.subplot(gs[0,0])
    pc = plt.contourf(x,y,hor_slice.T, 40, cmap='Blues', vmin = 0, vmax = maxvalue)
    plt.xticks([], [])
    # second graph
    axes = plt.subplot(gs[1,0])
    plt.contourf(x,z,ver_slice.T, 40, cmap='Blues', vmin = 0, vmax = maxvalue)
    # colorbar
    #axes = plt.subplot(gs[0,0])
    #plt.colorbar(pc,ax=axes)

    """
    # Make figure
    plt.close("all")
    fig = plt.figure()
    gs  = fig.add_gridspec(2, hspace=0.1, height_ratios= [5,1])
    axs = gs.subplots(sharex=True)
    # Top plot:
    axs[0].contourf(x,y,hor_slice.T, 40, cmap='Blues', vmin = 0, vmax = maxvalue)
    axs[0].set_ylabel("y (kpc)")
    # Middle plot: 
    axs[1].contourf(x,z,ver_slice.T, 40, cmap='Blues', vmin = 0, vmax = maxvalue)
    axs[1].set_ylabel("z (kpc)")
    # Correct lables
    axs[1].set_xlabel("x (kpc)")
    plt.suptitle("CRE density profiles",fontsize=20)
    #plt.tight_layout()
    fig.align_ylabels(axs)
    # Add colorbar

    # Save figure
    plt.savefig(figpath+fname)
    plt.close('all')

def pretty_slice_plotV2(data, grid, fname=' '):  
    # Take a horizontal and a vertical slice through the middle of the box
    unit       = data.unit
    resolution = grid.resolution  
    hor_slice = data[:,:,int(resolution[2]/2+0.5)]/unit 
    ver_slice = data[int(resolution[0]/2+0.5),:,:]/unit
    x = grid.x[:,0,0]/u.kpc
    y = grid.y[0,:,0]/u.kpc
    z = grid.z[0,0,:]/u.kpc
    maxvalue = np.max(data)/unit
    import matplotlib.gridspec as gridspec
    # Make figure
    fig = plt.figure(figsize=(8,4))
    gs = gridspec.GridSpec(2,2,figure=fig, height_ratios=[5,1])
    
    ax2 = fig.add_subplot(gs[1, 0]) 
    ax2.contourf(x,z,ver_slice.T, 40, cmap='Blues', vmin = 0, vmax = maxvalue)
    ax2.set_ylabel("z (kpc)")
    ax2.set_xlabel("x (kpc)")


    ax1 = fig.add_subplot(gs[0, 0], sharex=ax2)
    ax1.contourf(x,y,hor_slice.T, 40, cmap='Blues', vmin = 0, vmax = maxvalue)
    ax1.set_ylabel("y (kpc)")
    ax1.xaxis.set_visible(False)

    ax3 = fig.add_subplot(gs[:, 1])
    cbarplot = ax3.contourf(x,z,ver_slice.T, 40, cmap='Blues', vmin = 0, vmax = maxvalue)
    cbar = plt.colorbar(cbarplot, ax=ax3)
    cbar.set_label(unit, rotation=0)
    plt.delaxes(ax3)
    
    # Polish up figure
    #plt.tight_layout()
    plt.suptitle("CRE density profiles",fontsize=20)
    
    # Save figure
    plt.savefig(figpath+fname)
    plt.close('all')

#%% First try a normal thermal electron grid

# Setup coordinate grid
cartesian_grid = img.fields.UniformGrid(box=[[-20*u.kpc, 20*u.kpc],
                                             [-20*u.kpc, 20*u.kpc],
                                             [-5*u.kpc, 5*u.kpc]],
                                             resolution = [41,41,41])


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
#plot_slices(ne_data, cartesian_grid, 'thermalelectrons.png')



#%% Now inspect CRE grid constant index

# fields.BrokenPowerlawCosmicRayElectrons()


#( fields.GalpropCosmicRayElectrons( requires to be on the same grid ) maybe dont do this)

CRE_exponential = fields.PowerlawCosmicRayElectrons(
    grid            = cartesian_grid,
    parameters      ={'scale_radius':10.0*u.kpc,
                      'scale_height':1.0*u.kpc,
                      'spectral_index': -3,
                      'central_density':1e-5*u.cm**-3})

# Acces and plot profiles
CRE_data = CRE_exponential.get_data()#*u.cm**3
#print(CRE_data)



#plot_slices(CRE_data, cartesian_grid, 'comsicrayelectrons.png')
pretty_slice_plotV2(CRE_data, cartesian_grid, 'pretty_comsicrayelectrons_V2.png')

# Check spectral index
#print(dir(CRE_exponential),'\n')
print(CRE_exponential.parameters['spectral_index'],'\n')



#%% CRE grid with function for spectral index

"""
def spectral_index_function(x,y,z):
    # index range
    amax = -2
    amin = -4
    # galaxy scale
    z_scale = 2
    r_scale = 10
    # now scale profile to square
    r           = np.sqrt(x**2 + y**2)
    scaled_dist = np.sqrt((z/z_scale)**2 + (r/r_scale)**2)/2
    return amin + (amax-amin)*scaled_dist

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

"""
#%%




















