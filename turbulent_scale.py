#%% Imports and settings

# Utility
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import cartesian_to_spherical
import struct
import nifty7 as ift

# IMAGINE imports
import imagine as img
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12
from imagine.fields.field_utility import MagneticFieldAdder
from imagine.fields.field_utility import ArrayMagneticField

# Directory paths
figpath   = 'figures/'
fieldpath = 'arrayfields/'


def load_JF12rnd(fname, shape=(40,40,10,3)):
    print("Loading: "+fieldpath+fname)
    with open(fieldpath+fname, "rb") as f:
        arr = f.read()
        arr = struct.unpack("d"*(len(arr)//8), arr[:])
        arr = np.asarray(arr).reshape(shape)
    return arr

def make_nifty_points(points, dim=3):
    rows,cols = np.shape(points)
    if cols != dim: # we want each row to be a coordinate (x,y,z)
        points = points.T
        rows   = cols
    npoints = []
    for d in range(dim):
        npoints.append(np.full(shape=(rows,), fill_value=points[:,d]))
    return npoints


def plot_turbulentscale_dep(highres = False):
    
    # Load correct Brnd grid and set output names
    label  = 3
    if highres:
        resolution = [100,100,10] # high resolution
        Barray = load_JF12rnd(fname="highres/brnd_{}.bin".format(label), shape=(100,100,10,3))
        figname = 'turbulent_scale{}_highres.png'.format(label)
    else:
        resolution = [40,40,10]
        Barray = load_JF12rnd(fname="brnd_{}.bin".format(label), shape=(40,40,10,3))
        figname = 'turbulent_scale{}.png'.format(label)

    # Setup coordinate grid
    xmax = 20
    ymax = 20
    zmax = 2      
    box  = np.array([2*xmax,2*ymax,2*zmax]) # must be unitless
    grid_distances = tuple([b/r for b,r in zip(box, resolution)])
    domain = ift.makeDomain(ift.RGSpace(resolution, grid_distances))
    cartesian_grid = img.fields.UniformGrid(box=[[-20*u.kpc, 20*u.kpc],
                                                 [-20*u.kpc, 20*u.kpc],
                                                 [ -2*u.kpc,  2*u.kpc]],
                                                 resolution = resolution)
    
    # Setup turbulent grid
    Bfield = ArrayMagneticField(grid = cartesian_grid,
                                parameters = {'array_field': Barray*1e6*u.microgauss,
                                            'array_field_amplitude': 1.0})
    Bdata = Bfield.get_data()
    Bamp  = np.linalg.norm(Bdata,axis=3)

    # Create start and endpoints for the integration translated to domain
    nlos = 100
    start_points = np.zeros((nlos, 3))
    start_points[:,0] = 1
    start_points[:,1] = np.linspace(1,2*ymax-1,nlos)
    nstarts  = make_nifty_points(start_points)

    dres = 100
    los_distances = np.linspace(0,30,dres+1) # up to 30 kpc
    Brms = []
    for d in los_distances[1:]: # dont want los of distance 0
        end_points = np.copy(start_points)
        end_points[:,0] += d
        nends = make_nifty_points(end_points)
        # Do the nifty integration
        response = ift.LOSResponse(domain, nstarts, nends)
        Brms.append(response(ift.Field(domain, Bamp)).val_rw()/d)

    Brms  = np.array(Brms)
    meany = np.mean(Brms, axis=1)
    stdy  = np.std(Brms, axis=1)
    plt.plot(los_distances[1:],meany)
    plt.fill_between(los_distances[1:], meany-stdy, meany+stdy, alpha=0.2)
    plt.xlabel('los segment lenght (kpc)')
    plt.ylabel('average Brms (muG/kpc)')
    plt.title('LOS-distance sensitivity to turbulence')
    plt.savefig(figpath+figname)

plot_turbulentscale_dep(highres=True)




