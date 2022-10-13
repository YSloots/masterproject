import imagine as img
from imagine.fields.field_utility import FieldAdder
from imagine.fields.field_utility import ArrayField
from imagine.fields.cwrapped.wrappedjf12 import WrappedJF12
from imagine.fields.hamx.brnd_jf12 import BrndJF12

import astropy.units as u

box_size       = 15*u.kpc
cartesian_grid = img.fields.UniformGrid(box=[[-box_size, box_size],
                                                [-box_size, box_size],
                                                [-box_size, box_size]],
                                                resolution = [30,30,30])

Bfield1 = BrndJF12(grid=cartesian_grid)
B1array = Bfield.get_data()

Bfield2 = WrappedJF12(grid=cartesian_grid)

Bsum = FieldAdder(grid=cartesian_grid, summand_1 = Bfield1, summand_2 = Bfield2)

arrayfield = ArrayField(field = Bfield1)
print(arrayfield.array)
print(arrayfield.parameter_names)
#print(Bsum.get_data())