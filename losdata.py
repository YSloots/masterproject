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
    data = hdul[1].data

#print(type(data), dir(data))
#print(np.shape(data))
#print(data.columns)
#print(data[0])

#print(data.names)




#%% Make 2 datasets one for the distances and one for the emissivities


head = ['GLON','GLAT','Dist','e_Dist','n_T','eps','e_eps']

#print(data['n_T'][:20])


# Data dictionary for distances table
dist_data = {'Dist': np.ones(len(data['Dist'])),
             'e_Dist': data['e_Dist'],
             'lat': data['GLAT'],
             'lon': data['GLON'],
             'fb' : data['n_T']} # what do do with this column? does this work?
dist_dset = img.observables.TabularDataset(dist_data,
                                       name='distances',
                                       units=u.kpc,
                                       data_col='Dist',
                                       err_col='e_Dist',
                                       lat_col='lat',
                                       lon_col='lon')
mea = img.observables.Measurements(dist_dset)

# Data dictionary for emissivty table
em_data = {'emissivity': data['eps'],
           'err': data['e_eps'],
           'lat': data['GLAT'],
           'lon': data['GLON']}
em_dset = img.observables.TabularDataset(em_data,
                                       name='sync',
                                       tag='I',
                                       frequency=0.09*GHz,
                                       units=u.erg.cgs,
                                       data_col='emissivity',
                                       err_col='err',
                                       lat_col='lat',
                                       lon_col='lon')
mea.append(dataset=em_dset)

#%% Accessing the stored data

def investigate_acces():

    for key in mea.keys():
        print(key, '\n', type(mea[key]), '\n', dir(mea[key]), '\n')
    
    for k in mea.cov.keys():
        print(k,'\n')
    
    distkey = ('distances', None, 'tab', None)
    print(mea[distkey].data)
    print(mea[distkey].coords)
    print(mea.cov[distkey].data.shape)

#investigate_acces()

#%% Line of sight function



# convenience function
def _make_nifty_points(points, dim=3):
    rows,cols = np.shape(points)
    if cols != dim: # we want each row to be a coordinate (x,y,z)
        points = points.T
        rows   = cols
    npoints = []
    npoints.append(np.full(shape=(rows,), fill_value=points[:,0]))
    npoints.append(np.full(shape=(rows,), fill_value=points[:,1]))
    npoints.append(np.full(shape=(rows,), fill_value=points[:,2]))
    return npoints

def _get_response(starts):
    nstarts = _make_nifty_points(starts)
    return nstarts

observer = np.array([-8.5,1,0]) * u.kpc
nlos = 10
starts = []
[starts.append(np.array([-1,-2,-3])) for dummy in range(nlos)]
starts = np.array(starts)

print(starts)
print(_get_response(starts))

def get_startpoints(observer, nlos):
    starts = []
    for o in observer: starts.append( np.full(shape=(nlos,), fill_value=o) * observer.unit)

    return starts

starts = get_startpoints(observer, nlos = 5)
print(starts)

def get_endpoints(gal_lat, gal_lon, distances, dist_error = None):
    """
    Returns numpy ndarray where each row is a cartesian 
    galactocentric coordinate in kpc.    
    """
    
    #start = self.observerposition    
    observer = np.array([-8.5, 0, 0]) * u.kpc
    los      = spherical_to_cartesian(r=distances, lat=gal_lat, lon=gal_lon)
    
    #shift to observer
    ends = ()
    for i,axis in enumerate(los): ends += (axis-observer[i],)   
    
    if type(dist_error) != None:
        elos = spherical_to_cartesian(r=dist_error, lat=gal_lat, lon=gal_lon)
        return ends, elos #will subtract observer from each row in los
    else:
        return ends


distkey = ('distances', None, 'tab', None)
lat = mea[distkey].coords['lat'][:5]/u.deg * np.pi/180 * u.rad
lon = mea[distkey].coords['lon'][:5]/u.deg * np.pi/180 * u.rad
distances = mea[distkey].data[0,:5] * u.kpc
e_dist    = mea.cov[distkey].data[:5,:5].diagonal() * u.kpc

ends, e_ends = get_endpoints(gal_lat=lat, gal_lon=lon, distances=distances, dist_error=e_dist)
#print(ends)
#print(e_ends)


















