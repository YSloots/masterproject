# Simulator dependancies
import numpy as np
import imagine as img
from imagine import fields
from imagine.simulators import Simulator
from scipy.special import gamma as gammafunc
import nifty7 as ift

import astropy.units as u
from astropy import constants as cons
MHz   = 1e6 / u.s
GHz   = 1e9 / u.s
me    = cons.m_e.cgs
c     = cons.c.cgs
electron = (cons.e).gauss

# Units definitions since astropy doesnt handle [B]_cgs well
gauss_B  = (u.g/u.cm)**(0.5)/u.s
equiv_B  = [(u.G, gauss_B, lambda x: x, lambda x: x)]
ugauss_B = 1e-6 * gauss_B

# Script dependancies
import matplotlib.pyplot as plt

figpath = 'figures/'


print('\n')

#%% Projecting function

def project_to_unitvectors(field, grid, observer_position):
    gunit = observer_position.unit # need to check that grid.unit and position.unit are the same
    unitvectors = []
    for x,y,z in zip(grid.x.ravel()/gunit,grid.y.ravel()/gunit,grid.z.ravel()/gunit):
        v = np.array([x,y,z])-observer_position/gunit
        #print(v)
        normv = np.linalg.norm(v)
        if normv == 0: # special case where the observer is inside one of the grid points
            unitvectors.append(v)
        else:
            unitvectors.append(v/normv)
    unitvector_grid = np.reshape(unitvectors, tuple(grid.resolution)+(3,))

    v_parallel      = np.zeros(np.shape(field)) * field.unit
    amplitudes      = np.sum(field * unitvector_grid, axis=3)
    v_parallel[:,:,:,0]  = amplitudes * unitvector_grid[:,:,:,0]
    v_parallel[:,:,:,1]  = amplitudes * unitvector_grid[:,:,:,1]
    v_parallel[:,:,:,2]  = amplitudes * unitvector_grid[:,:,:,2]
    #print(v_parallel)
    v_perpendicular      = field - v_parallel
    v_perp_amplitude     = np.sqrt(np.sum(v_perpendicular*v_perpendicular,axis=3))
    return v_perp_amplitude


def plot_slice(data, grid, plane, index, fname=' '):
    unit       = data.unit
    data       = data/unit
    resolution = grid.resolution    
    
    x = grid.x[:,0,0]/u.kpc
    y = grid.y[0,:,0]/u.kpc
    z = grid.z[0,0,:]/u.kpc

    planes = ['x','y','z']
    if planes.index(plane) == 0:
        dslice = data[index,:,:]
        plt.imshow(dslice)
        plt.yticks(np.arange(len(z)),z)
        plt.xticks(np.arange(len(y)),y)
        plt.ylabel('z')
        plt.xlabel('y')
    if planes.index(plane) == 1: 
        dslice = data[:,index,:]
        plt.imshow(dslice)
        plt.yticks(np.arange(len(z)),z)
        plt.xticks(np.arange(len(x)),x)
        plt.ylabel('z')
        plt.xlabel('x')
    if planes.index(plane) == 2:
        dslice = data[:,:,index]
        plt.imshow(dslice)
        plt.yticks(np.arange(len(y)),y)
        plt.xticks(np.arange(len(x)),x)
        plt.ylabel('y')
        plt.xlabel('x')

    plt.title('Slice through '+plane+'-plane of cube at index {}'.format(index))    
    plt.colorbar()

    plt.savefig(figpath+fname)
    plt.close('all')

    print("Saved figure: " + figpath+fname)


#%% Test setup

# Setup coordinate grid
cartesian_grid = img.fields.UniformGrid(box=[[-15*u.kpc, 15*u.kpc],
                                             [-15*u.kpc, 15*u.kpc],
                                             [-15*u.kpc, 15*u.kpc]],
                                             resolution = [3,3,3])

# Magnetic Field Model
Bfield = fields.ConstantMagneticField(
    grid = cartesian_grid,
    ensemble_size= 1,
    parameters={'Bx': 1*u.microgauss,
                'By': 0*u.microgauss,
                'Bz': 0*u.microgauss})
#print(dir(Bfield))
#print(Bfield)
#print(Bfield.units)
#print(Bfield.get_data())
#print(Bfield.get_data().unit)

#%% Investigate grid 

vobs     = 90*MHz
observer = np.array([-15,0,0])*u.kpc

print("Observer at: ", observer)
Bperp = project_to_unitvectors(field=Bfield.get_data(), grid=cartesian_grid, observer_position=observer)
print(Bperp)
print(Bperp[:,0,:])


plot_slice(
    data=Bperp,
    grid=cartesian_grid,
    plane='z',
    index=1,
    fname='zplane_index1.png')

plot_slice(
    data=Bperp,
    grid=cartesian_grid,
    plane='x',
    index=0,
    fname='xplane_index0.png')

Bperp[0,0,0] = 0

plot_slice(
    data=Bperp,
    grid=cartesian_grid,
    plane='z',
    index=1,
    fname='zplane_index1_edit.png')

plot_slice(
    data=Bperp,
    grid=cartesian_grid,
    plane='x',
    index=0,
    fname='xplane_index0_edit.png')



