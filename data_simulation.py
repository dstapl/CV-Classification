# ---
# title:  dm-dt detection simulation
# author: Daniel Stapleton (dstapleton3@sheffield.ac.uk)
# date:   2024-11-28
# description: Want to find lowest dm that can be detected at 3\sigma significance level.
# ---

## Imports
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

class CCD:
    px_area = 0.28 * u.arcsec
    R = 12 / (u.pixel)**0.5
    dark_current = 0.002 / u.pixel/ u.s

class Filters:
    def __init__(self, bandwidths, quantum_efficiency):
        self.filter_set = ["ha", "b", "v", "r", "i"]

        self._bandpass = dict(bandwidths)
        self._Q = dict(quantum_efficiency)

        assert set(self._Q.keys()) == set(self.filter_set), "Filter mismatch when initialising bandwidths and quantum efficiency"
        assert set(self._bandpass.keys()) == set(self.filter_set), "Filter mismatch when initialising bandwidths and quantum efficiency"

        self._sky_brightness = dict()

    def Q(self, filter_name):
        """
        Attempts to get quantum_efficiency of chosen filter. None on failure.
        """
        return self._Q.get(filter_name)
    def filter(self, filter_name):
        return self._bandpass.get(filter_name)

    @property
    def sky(self):
        return self._sky_brightness

    @sky.setter
    def sky(self, values):
        new_sky = dict(values)
        assert set(new_sky.keys()) == set(["dark", "grey", "bright"]), "One or more of `dark`, `grey`, `bright` not included in new sky brightness"
        for brightness in new_sky.values():
            assert list(brightness.keys()) == self.filter_set, "Filter mismatch setting new sky brightness "
        self._sky_brightness = new_sky


class Telescope:
    def __init__(self, filters: Filters, ccd: CCD):
        self.tel_area = 1850.4 * u.cm**2
        self.filters = filters
        self.ccd = ccd

class Observing:
    # vega fluxes
    f_vega_unit = u.photon/u.s/u.cm**2/u.Angstrom
    f_vega = dict(
            ha=717*f_vega_unit, b=1170*f_vega_unit, v=1002*f_vega_unit, 
            r=717*f_vega_unit, i=471*f_vega_unit)

    def __init__(self, telescope, t_exp, binning, filter):
        self.telescope = telescope
        assert set(self.f_vega.keys()) == set(self.telescope.filters.filter_set), "Filter mismatch between vega fluxes and telescope filter set"

        self.t_exp = t_exp
        self.binning = binning
        self.filter=  filter.lower()

    def pixel_area(self):
        # pixel size is 0.28
        return (self.telescope.ccd.px_area * self.binning)**2

    def npix(self, aperture_area):
        """Effective number of pixels with binning."""
        return aperture_area / self.pixel_area() * u.pixel

    def dark_current(self, aperture_area):
        return self.telescope.ccd.dark_current * self.t_exp * self.npix(aperture_area)
    def total_readout(self, aperture_area):
        return self.npix(aperture_area)*self.telescope.ccd.R**2

    def snr(self, sky_type, mag, aperture_area):
        N = self.f_vega[self.filter] * 10**(-0.4*mag) * self.telescope.tel_area * self.telescope.filters.filter(self.filter) * self.t_exp * self.telescope.filters.Q(self.filter)

        # from sky
        m_sky = self.telescope.filters.sky[sky_type][self.filter]

        S = (self.f_vega[self.filter] * 10**(-0.4*m_sky) * self.telescope.tel_area * self.telescope.filters.filter(self.filter)
             * self.t_exp * self.telescope.filters.Q(self.filter) * aperture_area.to_value(u.arcsec**2))


        D = self.dark_current(aperture_area)
        total_R = self.total_readout(aperture_area)

        return N/np.sqrt(N+S+D+total_R)

class CV:
    def __init__(self, period, m0, dm, sigma_m):
        self.period = period
        self.m0 = m0
        self.stddev = sigma_m
        self.dm = dm
        self._size = 3*u.arcsec**2

    @property
    def size(self):
        return self._size

    def model(self):
        """Returns a function describing the superhump shape."""
        return self._y

    def _y(self, t):
        """
        Internal model used to represent the superoutburst shape
        t is time series of observations.
        """
        # Use a sine wave with constant frequency
        return self.m0 + self.dm*np.sin(t * 2*np.pi / self.period)


def calc_dm(observing_setup: Observing, object: CV, sky_type: str):
    snr = observing_setup.snr(sky_type, object.m0, object.size)
    return snr


def main():
    print("Start")

    # efficiency
    Q = dict(ha=0.41/u.photon, b=0.37/u.photon, v=0.42/u.photon, 
             r=0.42/u.photon, i=0.22/u.photon)

    # bandwidths 
    banddpass= dict(
            ha=70*u.Angstrom, b=720*u.Angstrom, v=860*u.Angstrom, 
            r=1330*u.Angstrom, i=1400*u.Angstrom)

    sky_brightness = dict(
            bright=dict(ha=21.2, b=18.7, v=17.7, r=18.0, i=18.3),
            grey=dict(ha=23.3, b=21.7,v=20.7, r=20.1, i=19.6),                
            dark=dict(ha=24.1, b=22.7, v=21.7, r=20.9, i=20.0),
            )

    filter_set = Filters(banddpass, Q)
    filter_set.sky = sky_brightness

    ccd = CCD()
    telescope = Telescope(filter_set, ccd)

    observing_setup = Observing(telescope, t_exp = 30*u.s, binning = 1, filter = "V")
    cv_object = CV(period = 90*u.min, m0 = 15, dm = 0.5, sigma_m = 0)

    snr = observing_setup.snr("dark", cv_object.m0, cv_object.size)
    print(snr)
#    results = calc_dm(observing_setup, cv_object, "dark")
#    print(results)



if __name__ == "__main__":
    main()
