#%% Imports and settings

# Imagine
import imagine as img

# Utility
import numpy as np
from astropy.coordinates import spherical_to_cartesian
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
from scipy.stats import poisson

# Gobal testing constants
import astropy.units as u
from astropy import constants as cons
MHz   = 1e6 / u.s
GHz   = 1e9 / u.s
me    = cons.m_e.cgs
c     = cons.c.cgs
electron = (cons.e).gauss
tau = 2*np.pi

gauss_B  = (u.g/u.cm)**(0.5)/u.s
equiv_B  = [(u.G, gauss_B, lambda x: x, lambda x: x)]
ugauss_B = 1e-6 * gauss_B


# Directory paths
figpath  = 'figures/'
datapath = 'data/'

print('\n')

#%% Functions
#===========================================================================

def sample_poisson(data, n_samples):
    mu = np.median(data)
    return poisson.rvs(mu, size=n_samples)

def sample_gaussian(data, n_samples):
    mu    = np.mean(data)
    sigma = np.std(data)
    return np.random.normal(loc=mu,scale=sigma,size=n_samples)

#%% Loading data from directory
#===========================================================================

fname = 'HII_LOS.fit'
with fits.open(datapath+fname) as hdul:
    hiidata = hdul[1].data

head = ['GLON','GLAT','Dist','e_Dist','n_T','eps','e_eps','n_eps']

#print(hiidata['n_T'][:20])
#print("We have {} Hii regions!".format(len(hiidata['Dist'])))
#print(len(hiidata['n_eps']))

#%% Determine reasonable integration box
#===========================================================================
# We know rdist max ~ 17.5 kpc -> xy minmax +-20kpc
# We know zdist max ~  0.5 kpc ->  z minmax +- 2kpc
# Resolution will depend on turbulent scale trying (40,40,10)


#%% Plotting functions
#===========================================================================


def plot_temperature():
    T = hiidata['eps'] * u.K/u.kpc
    print(T)
    plt.close('all')
    sns.displot(T,bins=30,kde=True)
    plt.title("Brightness temperature distribution")
    plt.xlabel("Brightness temperature T (K/kpc)")
    plt.tight_layout()
    plt.savefig(figpath+'brightness.png')
#plot_temperature()

def plot_temperature_error():
    e_T = hiidata['e_eps'] * u.K/u.kpc
    print(e_T)
    plt.close('all')
    sns.displot(e_T,bins=30,kde=True)
    plt.title("Temperature error distribution")
    plt.xlabel("Temperature error e_T (K/kpc)")
    plt.tight_layout()
    plt.savefig(figpath+'brightnesserror.png')
#plot_temperature_error()

def plot_relative_temperror():
    T   = hiidata['eps'] * u.K/u.kpc
    e_T = hiidata['e_eps']  * u.K/u.kpc
    rel_err   = e_T/T
    plt.close('all')
    sns.displot(rel_err,bins=30,kde=True)
    plt.title("Brightness temperature error distribution")
    plt.xlabel("Relative error (e_T/T)")
    plt.tight_layout()
    plt.savefig(figpath+'relativeTerr.png')
#plot_relative_temperror()

def plot_rdist():
    hiidist = hiidata['Dist'] * u.kpc
    plt.close('all')
    sns.displot(hiidist,bins=30,kde=True)
    plt.title("HII-region distances")
    plt.xlabel("Distance (kpc)")
    plt.tight_layout()
    plt.savefig(figpath+'rdist.png')
#plot_rdist()

def plot_relative_disterr():
    hiidist   = hiidata['Dist'] * u.kpc
    e_hiidist = hiidata['e_Dist']  * u.kpc
    rel_err   = e_hiidist/hiidist
    plt.close('all')
    sns.displot(rel_err,bins=30,kde=True)
    plt.title("HII-region distance error distribution")
    plt.xlabel("Relative error (e_Dist/Dist)")
    plt.tight_layout()
    plt.savefig(figpath+'relativedisterr.png')
#plot_relative_disterr()

from scipy.stats import norm as gaussian

def plot_zdist():
    hiidist = hiidata['Dist'] * u.kpc
    lat     = hiidata['GLAT'] * tau/360 * u.rad
    z       = np.sin(lat)*hiidist
    zgrid   = np.linspace(np.min(z),np.max(z),200)
    sigmas  = np.array([0.25,0.5,1,1.5])*np.std(z)
    plt.close('all')
    sns.displot(z,bins=30,stat='density')
    for s in sigmas:
        plt.plot(zgrid, gaussian.pdf(zgrid,0,s), label='sigma_z={0:.2}'.format(s))
    plt.legend()
    plt.title("HII-region Galactic z-hight distribution")
    plt.xlabel("Z-hight (kpc)")
    plt.tight_layout()
    plt.savefig(figpath+'zdist_extra.png')
#plot_zdist()


def plot_losdist():

    # Data
    hiidist   = hiidata['Dist'] * u.kpc
    e_hiidist = hiidata['e_Dist']  * u.kpc
    lat       = hiidata['GLAT'] * tau/360 * u.rad
    lon       = hiidata['GLON'] * tau/360 * u.rad
    behind    = np.where(np.array(hiidata['n_eps'])=='B')
    nlos      = len(hiidist)
    
    # Simulation setup
    xmax = 20
    ymax = 20
    zmax =  2
    box  = np.array([2*xmax,2*ymax,2*zmax])*u.kpc
    observer    = np.array([-8.5,0,0])*u.kpc
    translation = np.array([xmax,ymax,zmax])*u.kpc #not /2 because the box is 2*max
    translated_observer = observer + translation

    # Cast start and end points in a Nifty compatible format
    starts = []
    for o in translated_observer: starts.append(np.full(shape=(nlos,), fill_value=o)*u.kpc)
    start_points = np.vstack(starts).T

    ends = []
    los  = spherical_to_cartesian(r=hiidist, lat=lat, lon=lon)
    for i,axis in enumerate(los): ends.append(axis+translated_observer[i])
    end_points = np.vstack(ends).T

    # Do arithmatic to find behind los segments
    deltas = end_points - start_points
    clims  = box * np.sign(deltas) # why times box?
    clims[clims<0]=0 # if los goes in negative direction clim of xyz=0-plane
    
    with np.errstate(divide='ignore'):
        all_lambdas = (clims-end_points)/deltas   # possibly divide by zero 
    lambdas = np.min(np.abs(all_lambdas), axis=1) # will drop any inf here

    start_points[behind] = end_points[behind] + np.reshape(lambdas[behind], newshape=(np.size(behind),1))*deltas[behind]     
    
    los_distances = np.linalg.norm(end_points-start_points, axis=1)

    plt.close('all')
    sns.displot(los_distances,bins=30,kde=True)
    plt.title("LOS segment distances")
    plt.xlabel("Distance (kpc)")
    plt.tight_layout()
    plt.savefig(figpath+'losdist.png')
#plot_losdist()


#%% Generate typical and tunable simulated dataset
#===========================================================================

from scipy.stats import truncnorm

def generate_rdist():
    hiidist = hiidata['Dist'] * u.kpc

    r_samples = sample_poisson(hiidist,1000)

    plt.close('all')
    sns.displot(r_samples,kde=True)
    plt.title("Generated HII-region distances")
    plt.xlabel("Distance (kpc)")
    plt.tight_layout()
    plt.savefig(figpath+'generated_rdist.png')
#generate_rdist()

def generate_zdist():
    hiidist = hiidata['Dist'] * u.kpc
    lat     = hiidata['GLAT'] * tau/360 * u.rad
    z       = np.sin(lat)*hiidist

    stdz  = np.std(z)
    meanz = np.mean(z)
    print(stdz, meanz)
    zmax  = 2*u.kpc
    z_gen = truncnorm(a=-zmax/stdz, b=zmax/stdz, scale=stdz/u.kpc).rvs(100)*u.kpc
    z_gen[0]  = np.min(z)
    z_gen[-1] = np.max(z)
    plt.close('all')
    sns.displot(z_gen,bins=30,kde=True)
    plt.title("Generated HII-region z coordinates")
    plt.xlabel("Z Distance (kpc)")
    plt.tight_layout()
    plt.savefig(figpath+'generated_zdist.png')
#generate_zdist()