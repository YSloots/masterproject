import imagine as img
import numpy as np
import astropy.units as u
from imagine import fields
import matplotlib.pyplot as plt


# Setup coordinate grid
cartesian_grid = img.fields.UniformGrid(box=[[-15*u.kpc, 15*u.kpc],
                                             [-15*u.kpc, 15*u.kpc],
                                             [-5*u.kpc, 5*u.kpc]],
                                             resolution = [31,31,31])

# Cosmic-Ray Electron Model
CRE_exponential = fields.PowerlawCosmicRayElectrons(
    spectral_index  = -3
    grid            = cartesian_grid,
    parameters      ={'scale_radius':5.0*u.kpc,
                      'scale_height':1.0*u.kpc,
                      'spectral_index':-3  })

# Magnetic Field Model
Bfield = fields.ConstantMagneticField(
    grid = cartesian_grid,
    ensemble_size= 1,
    parameters={'Bx': 0.5*u.microgauss,
                'By': 0.5*u.microgauss,
                'Bz': 0.5*u.microgauss}))

class SynchrotronEmissivitySimulator(Simulator):
    """
    Test simulator for synchrotron emissivity
    """
    
    # Class attributes
    SIMULATED_QUANTITIES = ['testRM']
    REQUIRED_FIELD_TYPES = ['magnetic_field', 'cosmic_ray_electron_density']
    ALLOWED_GRID_TYPES = ['cartesian']
    
    def __init__(self, observing_frequency)
        self.frequency = observing_frequency
        self.position  = observing_position
    
    def simulate(self):
        B = self.fields['magnetic_field']
        
        return emissivity_grid