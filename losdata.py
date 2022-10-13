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

import matplotlib.pyplot as plt
import seaborn as sns

figpath = 'figures/'

print('\n')


#%% Loading data from directory

from astropy.io import fits

datapath = 'data/'
fname    = 'HII_LOS.fit'
with fits.open(datapath+fname) as hdul:
    hiidata = hdul[1].data




head = ['GLON','GLAT','Dist','e_Dist','n_T','eps','e_eps']

print(hiidata['n_T'][:20])
print("We have {} Hii regions!".format(len(hiidata['Dist'])))



def plot_relative_disterr():
    
    hiidist   = hiidata['Dist'] * u.kpc
    e_hiidist = hiidata['e_Dist']  * u.kpc
    rel_err   = e_hiidist/hiidist
    print(hiidist)
    print(e_hiidist)
    print(rel_err)
    plt.close('all')
    sns.distplot(rel_err)
    plt.savefig(figpath+'relativedisterr.png')
plot_relative_disterr()









