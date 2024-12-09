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
from astropy.timeseries import LombScargle
from scipy import stats

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
        N = self.f_vega[self.filter] * 10**(-0.4*mag.to(u.mag).value) * self.telescope.tel_area * self.telescope.filters.filter(self.filter) * self.t_exp * self.telescope.filters.Q(self.filter)

        # from sky
        m_sky = self.telescope.filters.sky[sky_type][self.filter]

        S = (self.f_vega[self.filter] * 10**(-0.4*m_sky.to(u.mag).value) * self.telescope.tel_area * self.telescope.filters.filter(self.filter)
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

    def model(self, sigma_m=0*u.mag):
        """Returns a function describing the superhump shape."""
        if sigma_m.to(u.mag) == 0*u.mag:
            return self._y

        # Otherwise generate random noise to add
        def gen_noise(t):
            return np.random.normal(loc = 0, scale = sigma_m.to(u.mag).value, size = len(t))

        return lambda t: self._y(t) + gen_noise(t)*u.mag

    def _y(self, t):
        """
        Internal model used to represent the superoutburst shape
        t is time series of observations.
        """
        # Use a sine wave with constant frequency
        w = 2*np.pi / self.period.to(u.s)
        return self.m0 + self.dm*np.sin((t * w * u.dimensionless_unscaled).value)


def calc_alpha(timeseries, magnitude_series, cv_object: CV, sky_type: str):
    #snr = observing_setup.snr(sky_type, object.m0, object.size)


    # Generate time series
    # Assuming continuous sampling

    # Calculate Lomb-Scargle periodogram
    (ls, freq, power) = Lomb(timeseries, magnitude_series, cv_object.period, factor=3)

    # Calculate false-alarm probabilities
    (alpha_realistic, alpha_optimal) = FAP(ls, power) # alpha_optimal is included for completeness
    

    return ((freq,power), alpha_realistic)

def FAP(ls: LombScargle, power):
    """Calculates the false-alarm probabilities from a Lomb-Scargle periodogram.
    
    Returns:
        - expected_alpha: Realistic probability
        - upper_alpha: Optimal probability using the approximation described by "Baluev"
    """
    # Observations will only take place at night
    # Expect a 1 day signal: f_alias = f_true + n*f_window (c.f. Astropy Lomb-Scargle Periodograms)
    upper_alpha = ls.false_alarm_probability(power.max(), method = "baluev")
    realistic_alpha = ls.false_alarm_probability(power.max(), method = "bootstrap") # Boostrap sampling method
    return (realistic_alpha, upper_alpha)

def Lomb(t, magnitude_series, object_period, factor = 5):
    """Calculates the Lomb-Scargle periodogram of the timeseries data t.

    Returns:
        - ls: Periodogram settings (LombScargle object)
        - freq: Frequency range evaluated
        - power: Power spectrum values
    """
    ls = LombScargle(t, magnitude_series)
    # Produced frequencies are NOT angular frequency
    # Period is related by T = 1/f
    # Chosen method "slow", or "cython", "fast"* can handle data point errors
    # "chi2", and "fastchi2" can not handle errors but compute fourier terms
    freq, power = ls.autopower(method = "auto", maximum_frequency = factor/(object_period.to(u.s))) # Nyquist factor (at least 2)
    return (ls, freq, power)


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
            bright=dict(ha=21.2*u.mag, b=18.7*u.mag, v=17.7*u.mag, r=18.0*u.mag, i=18.3*u.mag),
            grey=dict(ha=23.3*u.mag, b=21.7*u.mag,v=20.7*u.mag, r=20.1*u.mag, i=19.6*u.mag),                
            dark=dict(ha=24.1*u.mag, b=22.7*u.mag, v=21.7*u.mag, r=20.9*u.mag, i=20.0*u.mag),
            )

    filter_set = Filters(banddpass, Q)
    filter_set.sky = sky_brightness

    ccd = CCD()
    telescope = Telescope(filter_set, ccd)

    observing_setup = Observing(telescope, t_exp = 30*u.s, binning = 1, filter = "V")
    cv_object = CV(period = 90*u.min, m0 = 15*u.mag, dm = 0.5*u.mag, sigma_m = 0*u.mag)

    # SNR calculation 
    print(f"Calculating SNR for: {observing_setup = }; {cv_object = }")
    snr = observing_setup.snr("dark", cv_object.m0, cv_object.size)
    print(snr)

    # false-alarm-probability
    print(f"Calculating t-test for false-alarm probabilities")
    t = np.arange(0, cv_object.period.to(u.s).value, observing_setup.t_exp.to(u.s).value) * u.s

    alpha_list = []
    spectrum_list = []
    max_i = 5
    for i in range(max_i):
        # spectrum = (freq,power)
        mag_series = cv_object.model(sigma_m = cv_object.dm*2)(t)
        (spectrum, alpha_realistic)  = calc_alpha(t,mag_series, cv_object, sky_type = "grey")
        alpha_list.append(alpha_realistic)
        spectrum_list.append(spectrum)
        print(f"Finished object {i+1} out of {max_i}")

    
    print(f"Probabilities: {alpha_list = }")
    # Perform t-test: 3sigma significance against 0
    t_test_result = stats.ttest_1samp(alpha_list, popmean = 0, alternative = "greater")
    print(f"{t_test_result = }")
    pvalue = t_test_result.pvalue

    # Calculate confidence interval at 3sigma
    sigma_n = {1: 68.27, 2:95.45, 3: 99.73}
    conf_level = sigma_n[3]/100
    conf = t_test_result.confidence_interval(conf_level)
    if pvalue > (1 - conf_level):
        print(f"OVERALL {pvalue = } **IS SIGNIFICANT**")
    print(f"{t_test_result = };{conf = }")

    # Calculate number of simulations which passed significance level
    contains_mean = (conf.low < 1) & (conf.high > 1)
    print(f"{contains_mean.sum()} / {len(alpha_list)}")

    # Plot stuff
    #fig, ax = plt.subplots(2)
    #ax[0].plot(t, mag_series)
    # Lomb-Scargle
    #ax[1].plot(freq,power)


    plt.show()
if __name__ == "__main__":
    main()
