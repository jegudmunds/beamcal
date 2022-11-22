import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin

h = 6.6260755e-34
kb = 1.380658e-23
c = 2.99792458e8
Tcmb= 2.7255

def drone_beam(theta, FWHM=20):
    '''
    theta: Input angle in radians
    '''
    sigma = np.radians(FWHM) / np.sqrt(8 * np.log(2))
    return np.exp(-0.5 * theta**2 / sigma**2)

def planck_func(nu, T=2.7255, deriv=False):

    x = h*nu/kb/T
    if deriv:
        Bnu = (2*kb*nu**2)/c**2 * ((x**2 * np.exp(x))/(np.exp(x)-1)**2)
    else:
        Bnu = (2*h*nu**3)/c**2 / (np.exp(x)-1)

    return Bnu

def dPdTcmb(T0=2.725, v0=145e9, BW=0.25, pol=True):

    v1 = v0 * (1 - BW/2.)
    v2 = v0 * (1 + BW/2.)
    (out,err) = quad(lambda v: planck_func(v, deriv=True, pol=pol)*\
        (c/v)**2, v1, v2)

    return out

def dqdt(nu, bw=0.3, pol=True):

    return 1e12*dPdTcmb(v0=nu*1e9, BW=bw, pol=pol)

def rj_func(nu, T=1):

    return 2 * (nu**2) * kb * T / c**2

def rj_signal(nus, T):

    return np.trapz(rj_func(nus, T=T), nus)

class Telescope(object):

    def __init__(self, D=0.38):

        self.D = D
        self.A = np.pi * (D/2.0)**2


class Detector(Telescope):

    def __init__(self, nu=150e9, bw=0.25, nep=3.0e-16,
        oe=0.3, N=1001, telescope=None, D=0.38):
        '''
        oe: Optical efficiency [1]
        nep: Noise equivalent power [Watts/sqrt(Hz)]
        wavel: Band center wavelength [m]

        '''

        # Detector inherits from Telescope
        super().__init__(D=D)

        self.nu = nu
        self.nus = np.linspace((1-bw/2.0)*nu, (1+bw/2.0)*nu, N)
        self.N = N
        self.wavel = c/nu
        self.bw = bw
        self.telescope = telescope
        self.oe = oe
        self.nep = nep
        self.aomega = self.wavel**2
        self.fwhm = self.wavel/self.D
        self.sigma = self.fwhm / np.sqrt(8 * np.log(2))
        self.omega = 2 * np.pi * self.sigma**2
        self.Aeff = self.aomega / self.omega

    def __str__(self):
        #print()
        str_out = '  FWHM: {:3e}\n'.format(self.fwhm)
        str_out += '  Center wavelength: {:.3e} [m] \n'.format(self.wavel)
        str_out += '  Area: {:.2e} [m^2]'.format(self.A)

        return str_out

class Drone(object):

    def __init__(self, nu=150, Pout=0.001, FWHM=20.0, R=100.,
        Dsource=0.1, Tsource=400,
        microwave_source=True, thermal_source=False):

        if microwave_source and thermal_source:
            raise ValueError('Drone can only have 1 source equipped')

        self.microwave_source = microwave_source
        self.thermal_source = thermal_source

        self.nu = nu
        self.Pout = Pout
        self.FWHM = FWHM
        self.sigma = np.radians(FWHM) / np.sqrt(8 * np.log(2))
        self.beam_omega = 2 * np.pi * self.sigma **2
        self.Dsource = Dsource
        self.Tsource = Tsource
        self.source_omega = (np.arctan(Dsource/2.0/R))**2 * np.pi

        self.R = R

class Planet(object):
    '''
    d: Diameter of the planet disc as seen from Earth in radians
       Default value: 40 arcseconds

    '''

    def __init__(self, d=np.radians(40/3600.), T=200):
        self.d = d
        self.T = T
        self.omega = (d/2.0)**2 * np.pi


class System(object):

    def __init__(self, telescope, drone=None, planet=None):
        self.telescope = telescope
        self.drone = drone
        self.planet = planet

    def pwr_microwave_source(self):
        '''
        Power received by the detector from a drone [Watts]
        '''
        if self.drone is None:
            raise ValueError('Drone is not set')

        return self.telescope.oe * self.drone.Pout * self.telescope.A / \
            (self.drone.beam_omega * self.drone.R**2)

    def pwr_thermal_source(self):
        '''
        Power received by the detector from a drone [Watts]
        '''
        if self.drone is None:
            raise ValueError('Drone is not set')

        return self.telescope.oe * self.telescope.Aeff * self.drone.source_omega * \
            rj_signal(self.telescope.nus, self.drone.Tsource)

    def pwr_planet(self):
        '''
        Power received by the detector from a planet [Watts]
        '''        
        if self.planet is None:
            raise ValueError('Planet is not set')        
        return self.telescope.oe * self.telescope.Aeff * self.planet.omega * \
            rj_signal(self.telescope.nus, self.planet.T)

        #planet_power = oe * Aeff * omega_planet * rj_signal(nus, T=200)

def test_class():
    det = Detector()
    print(det)

def main():
    test_class()

    telescope = Telescope()
    drone1 = Drone(thermal_source=True, microwave_source=False,
        Tsource=500, Dsource=0.1, R=40.)
    drone2 = Drone(microwave_source=True)
    det = Detector()
    jupiter = Planet()

    sys1 = System(det, drone=drone1)
    sys2 = System(det, planet=jupiter)
    sys3 = System(det, drone=drone2)

    print('Power received from microwave source on Drone: {:2e} pW'.\
        format(1e12 * sys1.pwr_microwave_source()))

    print('Power received from Planet: {:2e} pW'.\
        format(1e12 * sys2.pwr_planet()))

    print('Power received from thermal source on Drone: {:2e} pW'.\
        format(1e12 * sys3.pwr_thermal_source()))

    # test()

def test():

    theta = np.radians(np.linspace(0, 90, 100))
    plt.plot(np.degrees(theta), drone_beam(theta))
    plt.axvline(10, ls=':')
    plt.axhline(0.5, ls=':')
    print(np.radians(20))
    print(np.radians(10))

    R = 100 # Distance between telescopes and drone [meters]

    # Detector and telescopes properties
    D = 0.3 # Diameter of telescopes aperture [m]
    A = np.pi * D**2 / 4.0 #[m^2]
    nu = 150e9 # Band center [Hz]
    bw = 0.25
    N = 1000
    nus = np.linspace((1-bw/2.0)*nu, (1+bw/2.0)*nu, N)
    wavel = c/nu # Band center wavelength [m]
    oe = 0.3 # Optical efficiency [1]
    nep = 3.0e-16 # [Watts/sqrt(Hz)]
    aomega = wavel ** 2
    fwhm = wavel/D
    sigma = fwhm / np.sqrt(8 * np.log(2))
    omega = 2 * np.pi * sigma**2
    Aeff = aomega / omega
    omega_planet = np.pi * np.radians((40/2.0)/(60*60))**2

    print('FWHM: {:3e}'.format(fwhm))
    print('Center wavelength: {:.3e} [m]'.format(wavel))
    print('Effective area: {:.2e} [m^2]'.format(Aeff))
    print('Area: {:.2e} [m^2]'.format(A))
    # System properties
    sa_aperture = A / (2 * np.pi * R**2)
    angular_size = np.arctan(D/R)

    print('Solid angle subtended by aperture: {:.2e} srad'.format(sa_aperture))
    print('Angular size of telescope seen from sources: {:.2f} arcmin'.format(np.degrees(60 * angular_size)))

    # Power output of transmitter [W]
    Pout = 0.010
    Preceived = Pout * sa_aperture
    Pdetectior = oe * Preceived

    ## Signal strength from a planet
    # print(rj_func(nu, T=100))
    # print(rj_signal(nus, T=100))
    # print(planet_power)

    planet_power = oe * Aeff * omega_planet * rj_signal(nus, T=200)
    print('Microwave source power received at aperture: {:.2e} W'.format(Preceived))
    print('Planet power received at aperture: {:.2e} W'.format(planet_power))
    print('Ratio {}'.format(10* np.log10(Preceived/planet_power)))
    print('Signal to noise ratio: {} dB'.format(10*np.log10(Preceived/nep)))
    print('Signal to noise ratio: {} dB'.format(10*np.log10(planet_power/nep)))

if __name__ == '__main__':
    main()
