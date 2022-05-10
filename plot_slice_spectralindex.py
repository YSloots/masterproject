import imagine as img
import numpy as np
import astropy.units as u
from imagine import fields
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
    
# Setup coordinate grid
cartesian_grid = img.fields.UniformGrid(box=[[-15*u.kpc, 15*u.kpc],
                                             [-15*u.kpc, 15*u.kpc],
                                             [-2*u.kpc, 2*u.kpc]],
                                             resolution = [30,30,30])
# Setup field
cre_alpha = fields.SpectralIndexLinearVerticalProfile(
    grid = cartesian_grid,
    parameters = {'soft_index':-4, 'hard_index':-2.5, 'slope':1*u.kpc**-1})
alpha_data = cre_alpha.get_data()
#print(alpha_data[:,:,0])
#print(alpha_data[:,:,1])
#print(alpha_data[:,:,2])
#print(alpha_data[:,:,3])
#print(alpha_data)


# Make figure
x = cartesian_grid.x[:,0,0]/u.kpc
y = cartesian_grid.y[0,:,0]/u.kpc
z = cartesian_grid.z[0,0,:]/u.kpc
ver_slice = alpha_data[:,int(len(y)/2),:]/alpha_data.unit
plt.contourf(x,z,ver_slice.T,10,cmap='Blues')
plt.title('JF12 magnetic field perpendicular LOS component')
plt.ylabel('z kpc')
plt.xlabel('x kpc')
plt.colorbar()

print("Saving figure")
plt.savefig(figpath+'spectralindexprofile.png')
plt.close('all')


















