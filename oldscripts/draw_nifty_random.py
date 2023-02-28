#%% Imports and settings

# Utility
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import cartesian_to_spherical
import struct
import nifty7 as ift

# IMAGINE imports
import imagine as img
from imagine.fields.field_utility import ArrayMagneticField



figpath   = 'figures/'
fieldpath = 'arrayfields/'

def draw_random(npix, distances, a_p, a_q, s_p, s_q, seed=42, check_divergence=False):
    ift.random.push_sseq(seed)
    domain = ift.makeDomain(ift.RGSpace(npix, distances))
    harmonic_domain = ift.makeDomain(domain[0].get_default_codomain())
    ht = ift.HarmonicTransformOperator(harmonic_domain[0], domain[0])

    d_dx = PartialDerivative(harmonic_domain, 0, 1)
    d_dy = PartialDerivative(harmonic_domain, 1, 1)
    d_dz = PartialDerivative(harmonic_domain, 2, 1)

    def pow_spec_q(k):
        import pylab as pl
        pl.ioff()
        pl.plot(np.log10(k + 1.5*k[1]), np.log10(a_q/(k + 1.5*k[1])**s_q))

        pl.savefig(figpath+'nifty_random/pspec_q')
        plt.close()
        return a_q/(k + 1.5*k[1])**s_q

    def pow_spec_p(k):
        if check_divergence:
            import pylab as pl
            pl.ioff()
            pl.plot(np.log10(k + 1.5*k[1]), np.log10(a_p/(k + 1.5*k[1])**s_p))

            pl.savefig(figpath+'nifty_random/pspec_p')
            plt.close()
        return a_p/(k + 1.5*k[1])**s_p

    # 1D spectral space on which the power spectrum is defined
    power_domain = ift.PowerSpace(harmonic_domain[0])

    # Mapping to (higher dimensional) harmonic space
    PD = ift.PowerDistributor(harmonic_domain, power_domain)

    # Apply the mapping
    correlation_structure_q = PD(ift.PS_field(power_domain, pow_spec_q))
    correlation_structure_p = PD(ift.PS_field(power_domain, pow_spec_p))

    S_q = ift.makeOp(correlation_structure_q)
    S_p = ift.makeOp(correlation_structure_p)

    position_q = ift.from_random(harmonic_domain, dtype=float)
    position_p = ift.from_random(harmonic_domain, dtype=float)

    q_k = S_q(position_q)
    p_k = S_p(position_p)

    grad_q_x = (ht@d_dx)(q_k)
    grad_q_y = (ht@d_dy)(q_k)
    grad_q_z = (ht@d_dz)(q_k)

    grad_p_x = (ht@d_dx)(p_k)
    grad_p_y = (ht@d_dy)(p_k)
    grad_p_z = (ht@d_dz)(p_k)

    b_x = (grad_q_y*grad_p_z - grad_q_z*grad_p_y)
    b_y = (grad_q_z*grad_p_x - grad_q_x*grad_p_z)
    b_z = (grad_q_x*grad_p_y - grad_q_y*grad_p_x)

    if check_divergence:
        div_B = ht((d_dx@ht.adjoint)(b_x) + (d_dy@ht.adjoint)(b_y) + (d_dz@ht.adjoint)(b_z))
        abs_B = (b_x**2 + b_y**2 + b_z**2).sqrt()
        rel_div = div_B/abs_B
        print('Checking divergence: ')
        print('relative mean value: ', rel_div.val.mean())
        print('relative max value: ', rel_div.val.max())
        print('relative min value: ', rel_div.val.min())
        print('Note that some residual divergence is to be expected')
        import pylab as pl
        pl.ioff()
        pl.imshow(rel_div.val[:, :, 5])
        pl.colorbar()
        pl.savefig(figpath+'nifty_random/rel_div')
        pl.close()

        pl.ioff()
        pl.imshow(ht(q_k).val[:, :, 5])
        pl.colorbar()
        pl.savefig(figpath+'nifty_random/q')
        pl.close()

        pl.imshow(ht(p_k).val[:, :, 5])
        pl.colorbar()
        pl.savefig(figpath+'nifty_random/p')
        pl.close()

        pl.imshow(b_x.val[:, :, 5])
        pl.colorbar()
        pl.savefig(figpath+'nifty_random/b_x')
        pl.close()

        pl.imshow(b_y.val[:, :, 5])
        pl.colorbar()
        pl.savefig(figpath+'nifty_random/b_y')
        pl.close()

        pl.imshow(b_z.val[:, :, 5])
        pl.colorbar()
        pl.savefig(figpath+'nifty_random/b_z')
        pl.close()

    return (b_x.val, b_y.val, b_z.val)


class PartialDerivative(ift.EndomorphicOperator):
    def __init__(self,  domain, direction, order):
        self._domain = ift.makeDomain(domain)
        assert self._domain[0].harmonic, 'This operator works in the harmonic domain'
        self.nax = len(self._domain[0].shape)
        assert direction <= self.nax, 'number of spatial dimensions smaller then direction given'
        self._direction = direction
        self.distance = self._domain[0].distances[direction]
        self.co_distance = self._domain[0].get_default_codomain().distances[direction]
        kfield = np.arange(0, domain.shape[direction])
        idx = kfield > domain.shape[direction]/2
        kfield[idx] = (kfield[idx][::] - domain.shape[direction])
        if len(domain.shape) > 1:
            i = 0
            while i < self._direction:
                kfield = np.repeat(kfield.reshape(1, *kfield.shape), domain.shape[self._direction - i - 1], axis=0)
                i += 1
            i += 1
            while i > self._direction and i < len(domain.shape):
                kfield = np.repeat(kfield.reshape(*kfield.shape, 1), domain.shape[i], axis=-1)
                i += 1

        self.kfield = kfield
        assert isinstance(order, int), 'only non fractional derivatives are supported'
        self.order = order
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        npix = self._domain[0].shape[self._direction]

        sign = 1. if mode == self.TIMES else -1.

        part = x.val * (sign * 2 * np.pi * self.kfield / self.distance / npix)**self.order
        if (self.order % 2) == 0 or (npix % 2) == 1:
            return ift.Field(self.domain, part)
        # Fixing the Nyquist frequency for even grids
        part[np.where(self.kfield == npix / 2)] = 0
        return ift.Field(self.domain, part)

def load_JF12rnd(fname, shape=(40,40,10,3)):
    print("Loading: "+fieldpath+fname)
    with open(fieldpath+fname, "rb") as f:
        arr = f.read()
        arr = struct.unpack("d"*(len(arr)//8), arr[:])
        arr = np.asarray(arr).reshape(shape)
    return arr

def plot_slice(fname, arr, zindex = 0):
    import matplotlib.pyplot as plt
    print("Creating figure \n", fname)
    plt.close()
    plt.imshow(arr[:, :, int(zindex)])
    plt.colorbar()
    plt.savefig(figpath+'nifty_random/'+fname)
    plt.close()

def get_slices(large_arr):

    # Fix z direction: Get grid of 10 points in z direction
    print(np.shape(large_arr))
    small_arr = large_arr[:,:,:20,:]
    print(np.shape(small_arr))
    zcollection = ()
    for index in 2*np.arange(10):# filter out 1 in every 2 layers
        zcollection += (small_arr[:,:,int(index),:],)
    small_arr = np.stack(zcollection, axis=2)
    print(np.shape(small_arr))

    # Fix x direction: Get section of 40 points in x direction
    xcollection = ()
    for index in 5*np.arange(40):# filter out 1 in every 5 layers
        xcollection += (small_arr[int(index),:,:,:],)
    small_arr = np.stack(xcollection, axis=0)
    print(np.shape(small_arr))

    # Fix y direction: GEt section of 40 points in y direction
    ycollection = ()
    for index in 5*np.arange(40):# filter out 1 in every 5 layers
        ycollection += (small_arr[:,int(index),:,:],)
    small_arr = np.stack(ycollection, axis=1)
    print(np.shape(small_arr))
    return small_arr

def scale_rms(vector_field, target_rms):
    amplitude_field = np.linalg.norm(vector_field,axis=3)
    return vector_field/np.mean(amplitude_field)*target_rms

def load_turbulent_field(fname, shape=(40,40,10,3)):
    print("Loading: "+fieldpath+fname)
    arr = np.load(fieldpath+fname, allow_pickle=True)
    return arr

def make_nifty_points(points, dim=3):
    rows,cols = np.shape(points)
    if cols != dim: # we want each row to be a coordinate (x,y,z)
        points = points.T
        rows   = cols
    npoints = []
    for d in range(dim):
        npoints.append(np.full(shape=(rows,), fill_value=points[:,d]))
    return npoints


def plot_turbulentscale_dep(label=1):
    plt.close()
    # Load correct Brnd grid and set output names
    resolution = [40,40,10]
    Barray = load_turbulent_field(fname="turbulent_sliced{}.npy".format(label), shape=(40,40,10,3))
    figname = 'turbulent_scale_sliced{}.png'.format(label)
    # Setup coordinate grid
    xmax = 20
    ymax = 20
    zmax = 2      
    box  = np.array([2*xmax,2*ymax,2*zmax]) # must be unitless
    grid_distances = tuple([b/r for b,r in zip(box, resolution)])
    domain = ift.makeDomain(ift.RGSpace(resolution, grid_distances))
    cartesian_grid = img.fields.UniformGrid(box=[[-20*u.kpc, 20*u.kpc],
                                                 [-20*u.kpc, 20*u.kpc],
                                                 [ -2*u.kpc,  2*u.kpc]],
                                                 resolution = resolution)
    # Setup turbulent grid
    Bfield = ArrayMagneticField(grid = cartesian_grid,
                                parameters = {'array_field': Barray*1e6*u.microgauss,
                                            'array_field_amplitude': 1.0})
    Bdata = Bfield.get_data()
    Bamp  = np.linalg.norm(Bdata,axis=3)
    # Create start and endpoints for the integration translated to domain
    nlos = 100
    start_points = np.zeros((nlos, 3))
    start_points[:,0] = 1
    start_points[:,1] = np.linspace(1,2*ymax-1,nlos)
    nstarts  = make_nifty_points(start_points)
    dres = 100
    los_distances = np.linspace(0,30,dres+1) # up to 30 kpc
    Brms = []
    for d in los_distances[1:]: # dont want los of distance 0
        end_points = np.copy(start_points)
        end_points[:,0] += d
        nends = make_nifty_points(end_points)
        # Do the nifty integration
        response = ift.LOSResponse(domain, nstarts, nends)
        Brms.append(response(ift.Field(domain, Bamp)).val_rw()/d)

    Brms  = np.array(Brms)
    meany = np.mean(Brms, axis=1)
    stdy  = np.std(Brms, axis=1)
    plt.plot(los_distances[1:],meany)
    plt.fill_between(los_distances[1:], meany-stdy, meany+stdy, alpha=0.2)
    plt.xlabel('los segment lenght (kpc)')
    plt.ylabel('average Brms (muG/kpc)')
    plt.title('LOS-distance sensitivity to turbulence')
    plt.savefig(figpath+figname)
    plt.close()

if __name__ == "__main__":
    #def draw_random(npix, distances, a_p, a_q, s_p, s_q, seed=42, check_divergence=False):
    for sd in [1,2,3,4,5]:
        bx,by,bz   = draw_random((200, 200, 200), (0.2, 0.2, 0.2), .0005, .0005, 4, 4, seed = 42*sd, check_divergence=True)
        Barr_nifty = np.stack((bx,by,bz), axis=3)
        Bamp_nifty = np.linalg.norm(Barr_nifty,axis=3)
        plot_slice(fname='nifty_turbulent_amp_highres{}.png'.format(sd), arr=Bamp_nifty, zindex=0)
        Barr_nifty = get_slices(large_arr=Barr_nifty)
        Barr_nifty = scale_rms(Barr_nifty, target_rms=1.0)
        Bamp_nifty = np.linalg.norm(Barr_nifty,axis=3)
        plot_slice(fname='nifty_turbulent_amp_sliced{}.png'.format(sd), arr=Bamp_nifty, zindex=0)
        
        np.save(fieldpath+'turbulent_sliced{}'.format(sd), Barr_nifty)

        plot_turbulentscale_dep(label=sd)

    #Barray    = load_JF12rnd(fname="brnd_1.bin", shape=(40,40,10,3))
    #Bamp_HamX = np.linalg.norm(Barray,axis=3)
    #plot_slice(fname='hamx_turbulent_amp.png', arr=Bamp_HamX, zindex=4)
    