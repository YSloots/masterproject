from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12
import imagine as img
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt




xmax = 15*u.kpc
ymax = 15*u.kpc
zmax =  2*u.kpc
r    = 9
cartesian_grid = img.fields.UniformGrid(box=[[-xmax, xmax],
                                             [-ymax, ymax],
                                             [-zmax, zmax]],
                                             resolution = [r,r,r])


Bfield = WrappedJF12(grid=cartesian_grid)
Bdata  = Bfield.get_data()
print(np.sum(np.isnan(Bdata)))
print(Bdata[int(r/2),int(r/2),:,:])

indexNaN = np.argwhere(np.isnan(Bdata)==True)

for index in indexNaN:
    print(index)
    print(cartesian_grid.coordinates['x'][index[0],index[1],index[2]])
    print(cartesian_grid.coordinates['y'][index[0],index[1],index[2]])
    print(cartesian_grid.coordinates['z'][index[0],index[1],index[2]])


print(dir(cartesian_grid))
#print(cartesian_grid.coordinates)