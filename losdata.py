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
tau = 2*np.pi

# Units definitions since astropy doesnt handle [B]_cgs well
gauss_B  = (u.g/u.cm)**(0.5)/u.s
equiv_B  = [(u.G, gauss_B, lambda x: x, lambda x: x)]
ugauss_B = 1e-6 * gauss_B

import matplotlib.pyplot as plt
import seaborn as sns

figpath = 'figures/'

print('\n')


#%% Loading data from directory
#===========================================================================

from astropy.io import fits

datapath = 'data/'
fname    = 'HII_LOS.fit'
with fits.open(datapath+fname) as hdul:
    hiidata = hdul[1].data


head = ['GLON','GLAT','Dist','e_Dist','n_T','eps','e_eps']

#print(hiidata['n_T'][:20])
#print("We have {} Hii regions!".format(len(hiidata['Dist'])))


#%% Determine reasonable integration box
#===========================================================================
# We know rdist max ~ 17.5 kpc -> xy minmax +-20kpc
# We know zdist max ~  0.5 kpc ->  z minmax +- 2kpc
# Resolution will depend on turbulent scale trying (40,40,10)


#%% Plotting functions
#===========================================================================

def plot_relative_temperror():
    T   = hiidata['eps'] * u.K/u.kpc
    e_T = hiidata['e_eps']  * u.K/u.kpc
    rel_err   = e_T/T
    plt.close('all')
    sns.displot(rel_err,kde=True)
    plt.title("Brightness temperature error distribution")
    plt.xlabel("Relative error (e_T/T)")
    plt.tight_layout()
    plt.savefig(figpath+'relativeTerr.png')
#plot_relative_temperror()

def plot_rdist():
    hiidist = hiidata['Dist'] * u.kpc
    plt.close('all')
    sns.displot(hiidist,kde=True)
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
    sns.displot(rel_err,kde=True)
    plt.title("HII-region distance error distribution")
    plt.xlabel("Relative error (e_Dist/Dist)")
    plt.tight_layout()
    plt.savefig(figpath+'relativedisterr.png')
#plot_relative_disterr()

def plot_zdist():
    hiidist = hiidata['Dist'] * u.kpc
    lat     = hiidata['GLAT'] * tau/360 * u.rad
    z       = np.sin(lat)*hiidist
    plt.close('all')
    sns.displot(z,kde=True)
    plt.title("HII-region Galactic z-hight distribution")
    plt.xlabel("Z-hight (kpc)")
    plt.tight_layout()
    plt.savefig(figpath+'zdist.png')
#plot_zdist()







