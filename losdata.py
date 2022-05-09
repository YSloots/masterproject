import numpy as np
import imagine as img
import astropy.units as u
from astropy.coordinates import spherical_to_cartesian
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

print('\n')


#%% Downloading data from online source

#import astroquery
#from astroquery.vizier import Vizier
#HIIlosdata = Vizier.get_catalogs('J/A+A/636/A2')
#print(dir(HIIlosdata))
#print(HIIlosdata)


#%% Loading data from directory

from astropy.io import fits

datapath = 'data/'
fname    = 'HII_LOS.fit'
with fits.open(datapath+fname) as hdul:
    hiidata = hdul[1].data

#%% Make 2 datasets one for the distances and one for the emissivities


head = ['GLON','GLAT','Dist','e_Dist','n_T','eps','e_eps']

print(hiidata['n_T'][:20])
print("We have {} Hii regions!".format(len(hiidata['Dist'])))

"""
This is an old idea. Now we use a sim_config dictionary to initialize the simulator. 
The errors on the distances are accounted for by the Nifty los integrator.

# Data dictionary for distances table
dist_data = {'Dist': np.ones(len(hiidata['Dist'])),
             'e_Dist': hiidata['e_Dist'],
             'lat': hiidata['GLAT'],
             'lon': hiidata['GLON'],
             'fb' : hiidata['n_T']} # what do do with this column? does this work?
dist_dset = img.observables.TabularDataset(data=dist_data,
                                       name='distances',
                                       units=u.kpc,
                                       data_col='Dist',
                                       err_col='e_Dist',
                                       lat_col='lat',
                                       lon_col='lon')
#print(dist_dset.keys())
"""

"""

# Data dictionary for emissivty table
em_data = {'emissivity': hiidata['eps'],
           'err': hiidata['e_eps'],
           'lat': hiidata['GLAT'],
           'lon': hiidata['GLON']}
em_dset = img.observables.TabularDataset(data=em_data,
                                       name='average emissivty',
                                       tag='I',
                                       frequency=0.09*GHz,
                                       units=u.K/u.kpc,
                                       data_col='emissivity',
                                       err_col='err',
                                       lat_col='lat',
                                       lon_col='lon')


mea = img.observables.Measurements(em_dset)

for key in mea.keys():
    print(key)
    print(mea[key].unit)
    print(key[1])


#%% Accessing the stored data

def investigate_acces():

    for key in mea.keys():
        print(key, '\n', type(mea[key]), '\n', dir(mea[key]), '\n')
    
    for k in mea.cov.keys():
        print(key,'\n', mea.cov[key].data)

    emkey = ('average emissivty', 0.09000000000000001, 'tab', 'I')
    print(type(mea[emkey].data), '\n', mea[emkey].data)     
    print(dir(mea[emkey])    )
    
investigate_acces()


"""


#%% Investigate distance errors



hiidist   = hiidata['Dist'] * u.kpc
e_hiidist = hiidata['e_Dist']  * u.kpc
rel_err   = e_hiidist/hiidist
print(hiidist)
print(e_hiidist)
print(rel_err)










