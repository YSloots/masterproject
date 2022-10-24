import imagine as img
from imagine.fields.field_utility import FieldAdder
from imagine.fields.field_utility import ArrayMagneticField
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12
import astropy.units as u

# Stored binary fields
#fieldpath = '~/masterproject/devimagine/imagine/fields/arrayfields/'
fieldpath = 'arrayfields/'

#%% Test FieldAdder

def inspect_field(field):
        print("\n\nInspecting ", field)
        print("TYPE:", field.TYPE)
        print("parameter_names: \n", field.parameter_names)
        print("parameters: \n", field.parameters)


box_size       = 15*u.kpc
cartesian_grid = img.fields.UniformGrid(box=[[-box_size, box_size],
                                                [-box_size, box_size],
                                                [-box_size, box_size]],
                                                resolution = [30,30,30])
Bfield1   = WrappedJF12(grid=cartesian_grid)
B_factory = img.fields.FieldFactory(field_class = Bfield1, grid=cartesian_grid)
B_factory.active_paramters = ('b_arm_1',)
B_factory.priors = {'b_arm_1':img.priors.FlatPrior(xmin=0*u.microgauss, xmax=10*u.microgauss)}
inspect_field(Bfield1)

Bfield2 = img.fields.ConstantMagneticField(
        grid = cartesian_grid,
        parameters={'Bx': 6*u.microgauss,
                    'By': 0*u.microgauss,
                    'Bz': 0*u.microgauss})
inspect_field(Bfield2)

Bsum = FieldAdder(grid=cartesian_grid, summand_1 = Bfield1, summand_2 = Bfield2)
inspect_field(Bsum)


B_factory = img.fields.FieldFactory(field_class = Bsum, grid=cartesian_grid)


#%% Test ArrayMagneticField

"""
import struct
import numpy as np

with open(fieldpath+"brnd.bin", "rb") as f:
	arr = f.read()
	arr = struct.unpack("d"*(len(arr)//8), arr[:])
	arr = np.asarray(arr).reshape((30,30,30,3))
#print(arr)
#print(np.shape(arr))


Bfield2 = ArrayMagneticField(grid=cartesian_grid,
			     array_field=arr*u.microgauss,
			     scale=2.0,
			     name="BrndJF12")
print(Bfield2.get_data())
print(Bfield2.parameter_names)
print(Bfield2.parameters)
print(np.all(2.0*arr*u.microgauss==Bfield2.get_data()))
print(dir(Bfield2))

Bsum = FieldAdder(grid=cartesian_grid, summand_1 = Bfield1, summand_2 = Bfield2)
print(Bsum.get_data())
print(Bsum.parameters)
"""



