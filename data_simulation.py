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
import statsmodels.api as sm

import multiprocessing
from tqdm import tqdm
#from pathos.multiprocessing import ProcessingPool as Pool
import pathos

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

    def sky(self, sky_type: str):
        """Sky background magnitude in chosen filter"""
        return self.telescope.filters.sky[sky_type][self.filter]

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

    def snr(self, sky_type, mag, aperture_area, mag_above_noise = False):
        factor = self.f_vega[self.filter] * self.telescope.tel_area * self.telescope.filters.filter(self.filter) * self.t_exp * self.telescope.filters.Q(self.filter)


        # from sky
        m_sky = self.sky(sky_type)

        S = factor * 10**(-0.4*m_sky.to(u.mag).value) * aperture_area.to_value(u.arcsec**2)


        D = self.dark_current(aperture_area)
        total_R = self.total_readout(aperture_area)
        
        if mag_above_noise:
            mag = m_sky + mag
        N = factor * 10**(-0.4*mag.to(u.mag).value) 
    
        total_noise_variance = N + S + D + total_R

        #print(f"{N = };{total_noise_variance = }")
        return N/np.sqrt(total_noise_variance)

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

    def model(self, mag_noise: bool = False):
        """Returns a function describing the superhump shape.

        Returns:
            - Model y-data
            - Noise associated with each error bar (int = constant, list = irregular)
        """
        if (self.stddev.to(u.mag) == 0*u.mag) or (not mag_noise):
            return lambda t: (self._y(t), None)

        return self._noisy_y

    # Otherwise generate random noise to add
    def _gen_noise(self, t):
        #return np.absolute(np.random.normal(loc = 0, scale = self.stddev.to(u.mag).value, size = len(t)))
        sigma = self.stddev.to(u.mag).value

        # TODO Set sigma of noise
        #noise = np.random.normal(loc = sigma, scale = 1, size = len(t))
        #noise = np.random.normal(loc = 0, scale = sigma, size = len(t))
        noise = np.random.poisson(lam=sigma, size = len(t))

        return noise


    def _noisy_y(self, t):
        noise = self._gen_noise(t)*u.mag
        return (self._y(t) + noise, np.absolute(noise))

    def _y(self, t):
        """
        Internal model used to represent the superoutburst shape
        t is time series of observations.
        """
        # Use a sine wave with constant frequency
        w = 2*np.pi / self.period.to(u.s)
        return self.m0 + self.dm*np.sin((t * w * u.dimensionless_unscaled).value)


def calc_alpha(timeseries, magnitude_series, cv_object: CV, noise, sky_type: str):
    """Noise on each data point - constant or list for periodogram"""
    #snr = observing_setup.snr(sky_type, object.m0, object.size)


    # Generate time series
    # Assuming continuous sampling

    # Calculate Lomb-Scargle periodogram
    # Factor is the sampling frequency = factor*the nyquist frequency
    period = cv_object.period
    nyquist_freq = 0.5 * 1/period
    t_exp = timeseries[1] - timeseries[0]
    sampling_freq = 1/t_exp

    factor = sampling_freq / nyquist_freq
    factor = factor.to(u.dimensionless_unscaled).value
    #print(f"{factor = }")
    # TODO: Set factor
    max_freq = 1e+2*nyquist_freq.to(1/u.s) # pre-factor should = 1, but periodogram is an approximation so leaving room for errors
    (ls, freq, power) = Lomb(timeseries, magnitude_series, errors = noise, factor=factor, max_freq =max_freq)

    alpha_realistic = FAP(ls, power) # alpha_optimal is included for completenes


    return ((freq,power), alpha_realistic)

def FAP(ls: LombScargle, power, method = "baluev"):
    """Calculates the false-alarm probabilities from a Lomb-Scargle periodogram.
   
    Args:
        - method: Ideally `bootstrap` otherwise `baluev`
    Returns:
        - expected_alpha: Realistic probability
    """
    # Observations will only take place at night
    # Expect a 1 day signal: f_alias = f_true + n*f_window (c.f. Astropy Lomb-Scargle Periodograms)
    #upper_alpha = ls.false_alarm_probability(power.max(), method = method)

    realistic_alpha = ls.false_alarm_probability(power.max(), method = method) # Ideally Boostrap sampling
    return realistic_alpha

def Lomb(t, magnitude_series, errors = 0, factor = 1, max_freq = 1/u.s):
    """Calculates the Lomb-Scargle periodogram of the timeseries data t.

    Returns:
        - ls: Periodogram settings (LombScargle object)
        - freq: Frequency range evaluated
        - power: Power spectrum values
    """
    ls = LombScargle(t, magnitude_series, errors)

    # Produced frequencies are NOT angular frequency
    # Period is related by T = 1/f
    # Chosen method "slow", or "cython", "fast"* can handle data point errors
    # "chi2", and "fastchi2" can not handle errors but compute fourier terms
    freq, power = ls.autopower(method = "fast", nyquist_factor = factor, maximum_frequency=max_freq) # Nyquist factor (at least 2)

    return (ls, freq, power)

def simulate(observing_setup, cv_object, iter_N:int, sky_type = "dark", alpha_threshold = 1e-4, conf_level = 99.73/100, log=False):
    if log:
        print("Calculating for {observing_setup = }; {cv_object = }")
    # SNR calculation 
    if log:
        print(f"Calculating SNR for: {observing_setup = }; {cv_object = }")
    snr = observing_setup.snr(sky_type, cv_object.m0, cv_object.size)
    #print(f"Mean flux to noise snr: {snr = }")
    if log: 
        print(snr)

    # false-alarm-probability
    if log:
        print(f"Calculating t-test for false-alarm probabilities")
    t = np.arange(0, cv_object.period.to(u.s).value, observing_setup.t_exp.to(u.s).value) * u.s

    # TODO: E.g. Noise is 10% of mean flux m0 so noise = 10% * m0 = 0.1 * m0 --> 0.1 = noise/m0 = 1/snr
    # from SNR of object (using mean flux): snr = m0 / noise --> 1/snr = noise / m0 
    # so dm_snr = (dm/m0) / (noise / m0) = dm/noise
    #dm_snr = observing_setup.snr(sky_type, cv_object.stddev, cv_object.size, mag_above_noise = True)
    dm_snr = (cv_object.stddev / cv_object.m0) * snr

    # max magnitude is sky_magnitude
    max_mag = observing_setup.sky(sky_type)

    #print(f"Amplitude to mean flux SNR: {dm_snr = }\n")
    mag_list = []
    alpha_list = []
    spectrum_list = []
    #percentage = 10
    #N_percent = iter_N // percentage
    #if N_percent == 0:
    #    N_percent = 1 # to stop division by zero errors
    for i in range(iter_N):
        # spectrum = (freq,power)
        # Measurement uncertainty is based on SNR of dm to m0
        # dm / sigma_m = SNR ==> sigma_m = dm / SNR
        #unc =  cv_object.stddev / snr

        mag_series, mag_changes  = cv_object.model(mag_noise = True)(t)
        # max magnitude is sky_magnitude
        mag_series = np.clip(mag_series, a_min = -100*u.mag, a_max = max_mag)

        #unc = cv_object.m0 / cv_object.stddev

        # unc is dm/snr
        unc = cv_object.stddev/dm_snr
        # Now replace changes in magnitude with measurement uncertainty
        #errors = mag_changes
        if unc is None:
            errors = None
        else:
            try:
                unc = unc.to(u.mag)
            except:
                unc = unc*u.mag
            #errors = np.repeat(unc, len(mag_series))
            errors = unc
        mag_list.append((mag_series, errors))

        # Want the uncertainty on measurements
        (spectrum, alpha_realistic)  = calc_alpha(t,mag_series, cv_object, noise = errors, sky_type = sky_type)
        alpha_list.append(alpha_realistic)
        spectrum_list.append(spectrum)
    #    if i % N_percent == 0:
    #        print(f"Completed {i/N_percent *percentage}% ({i}/{iter_N})")
        if log:
            print(f"Finished object {i+1} out of {iter_N}")

    alpha_list = np.array(alpha_list)
    if log:
        print(f"Probabilities: {alpha_list = }")
    # Perform t-test: 3sigma significance against 0
    alpha_list = alpha_list[~np.isnan(alpha_list)]


    # alpha has outliers from some probabilities being close to 1
    # need to use nonparametric test
    #t_test_result = stats.ttest_1samp(alpha_list, popmean = alpha_threshold, alternative = "less")
    # Wilcoxon uses one-sample so need to test against variable X - mean
    test_result = stats.wilcoxon(alpha_list - alpha_threshold, correction = False, alternative = "less")

    if log:
        print(f"{test_result = }")
    pvalue = test_result.pvalue

#    conf = t_test_result.confidence_interval(conf_level)
#    if log and (pvalue > (1 - conf_level)):
#        print(f"OVERALL {pvalue = } **IS SIGNIFICANT(ly BAD)**")
#    if log:
#        print(f"{t_test_result = };{conf = }")
#
#    # Calculate number of simulations which passed significance level
#    # *GOOD* results have alpha (probability) = 0
#    contains_mean = (alpha_list > conf.low) & (alpha_list < conf.high)
#    proportion_mean = contains_mean.sum() / len(alpha_list)
#    if log:
#        print(f"{proportion_mean = }")

    return (alpha_list, (t, mag_list, spectrum_list), pvalue)#, conf, proportion_mean)

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

    observing_setup = Observing(telescope, t_exp = 30*u.s, binning = 2, filter = "V")

    ### INPUTS


    #m0 = 17 # Mean magnitude
    sky = "bright"
    m0 = 17.1*u.mag
    dm = 0.6*u.mag
    cv_object = CV(period = 90*u.min, m0 = m0, dm = dm, sigma_m = dm) # Initial value of no noise
#    SNR = observing_setup.snr(sky_type = sky, mag = cv_object.m0, aperture_area=cv_object.size )
#    print(f"{SNR = }")
    # (https://slittlefair.staff.shef.ac.uk/teaching/phy241/lectures/l09/)
    # SNR = (dm/mean flux)% / (noise/mean flux)% = dm / noise
    # TODO: Need to find minimum SNR such that the peak is still detected
    # dm / sigma_m = SNR ==> sigma_m = dm / SNR


    # TODO: Uncomment?
    # sigma_m = (dm/SNR)
    # cv_object.stddev= sigma_m 

    # Calculate confidence interval at 3sigma
    sigma_n = {1: 68.27, 2:95.45, 3: 99.73}
    n_sigma_away = 3
    conf_level = sigma_n[n_sigma_away]/100
    conf_level = round(conf_level, 10) # Floating point innacuracies
    alpha_threshold = 1e-3

    iter_N = 1000
    #res = simulate(observing_setup, cv_object, iter_N = iter_N, sky_type = sky, alpha_threshold = alpha_threshold,  conf_level = conf_level, log=False)
    #(alpha_list,(t, mag_series_list, spectrum_list),pvalue) = res#,ci,prop_mean) = res


    # TODO: Create heatmap 

    # Plot m0, dm with p-value as heat
    # For each sky (so 3 plots overall)
    # Bounds: sky=[dark, grey, bright], m0=[10, sky_mag[sky_type]], dm = [0, 1]A
    sky_values = ["dark", "grey", "bright"]

    sigma_n = {1: 68.27, 2:95.45, 3: 99.73}
    n_sigma_away = 3
    conf_level = sigma_n[n_sigma_away]/100
    conf_level = round(conf_level, 10) # Floating point innacuracies

    def pvaltosigma(pvalue):
        "Converts p to n-sigma significant; 0 indicates not significant i.e., failed"
        for value in sigma_n.values():
            if pvalue < (1 - value / 100):
                return list(sigma_n.keys())[list(sigma_n.values()).index(value)]
        else:
            return 0
         

    alpha_threshold = 1e-3

    iter_N = 1000 # For each innermost loop
    
     
    fig, ax = plt.subplots(3)



    # Create a list of points

    for idx, sky_type in enumerate(sky_values, start=0):
        sky_mag = observing_setup.sky(sky_type).value
        m0_step = 0.1
        m0_values = np.arange(10, sky_mag+m0_step, m0_step)
        dm_step = 0.01
        dm_values = np.arange(0, 1+dm_step, dm_step)
    
        X, Y = np.meshgrid(m0_values, dm_values)
        points = [(x, y) for x, y in zip(X.ravel(), Y.ravel())]# Not sure if ravel is actually needed
       
        def compute(point):
            (m0, dm) = point
            cv_object = CV(period = 90*u.min, m0 = m0*u.mag, dm = dm*u.mag, sigma_m = dm*u.mag) # Initial value of no noise

            res = simulate(observing_setup, cv_object, iter_N = iter_N, sky_type = sky, alpha_threshold = alpha_threshold,  conf_level = conf_level, log=False)
            (_,_,pvalue) = res

            #heat = pvaltosigma(pvalue)
            heat = 1 if pvalue <= (1-conf_level) else 0
            return heat
        with pathos.multiprocessing.ProcessingPool() as pool:
            Z = list(
                    tqdm(
                        pool.imap(compute, points),
                        total=len(points)
                    )
                )

        Z = np.array(Z).reshape(Y.shape)


        sky_heatmap = ax[idx].imshow(Z, extent=[min(m0_values), max(m0_values), min(dm_values), max(dm_values)],
               origin='lower', aspect='auto', cmap='viridis')

        sky_cbar = fig.colorbar(sky_heatmap, ax=ax[idx], label=f"Sigma significance")
        ax[idx].set_xlabel("m0")
        ax[idx].set_ylabel("dm")
        ax[idx].set_title(f"Pr(NOT FALSE ALARM) = True peak for **{sky_type} SKY**")


        continue
        Z = np.zeros((len(m0_values), len(dm_values))) # (len Y, len X)
        percentage = 10
        N_percent = len(m0_values) // percentage # Outerloop
        if N_percent == 0:
            N_percent = 1 # to stop division by zero errors

        for i, m0 in enumerate(m0_values):
            for j, dm in enumerate(dm_values):
                #sky = "bright"
                #m0 = 17.1*u.mag
                #dm = 0.6*u.mag
                cv_object = CV(period = 90*u.min, m0 = m0*u.mag, dm = dm*u.mag, sigma_m = dm*u.mag) # Initial value of no noise

                res = simulate(observing_setup, cv_object, iter_N = iter_N, sky_type = sky, alpha_threshold = alpha_threshold,  conf_level = conf_level, log=False)
                (_,_,pvalue) = res

                #heat = pvaltosigma(pvalue)
                heat = 1 if pvalue < (1-conf_level) else 0
                Z[i,j] = heat
    
            if i % N_percent == 0:
                print(f"Completed {i/N_percent *percentage}% ({i}/{len(m0_values)}) of sky type {idx+1}/{len(sky_values)}")

        # Draw heatmap
        sky_heatmap = ax[idx].imshow(Z, extent=[min(m0_values), max(m0_values), min(dm_values), max(dm_values)],
               origin='lower', aspect='auto', cmap='viridis')

        sky_cbar = fig.colorbar(sky_heatmap, ax=ax[idx], label=f"Sigma significance")
        ax[idx].set_xlabel("m0")
        ax[idx].set_ylabel("dm")
        ax[idx].set_title(f"Pr(NOT FALSE ALARM) = True peak for **{sky_type} SKY**")


    plt.show()
    # Want to draw contour line of 3 sigma 

    return 0 
    
    
    
    
    print(f"\nMean uncertainty on measurements = {np.mean(np.array(mag_series_list,dtype=object)[:,1])}") 
    print(f"\n{np.random.choice(alpha_list, 20) = }")
    print(f"False alarm list: Mean = {np.mean(alpha_list)}, variance = {np.var(alpha_list)}")
    print(f"Median false-alarm alpha = {np.median(alpha_list)}; {alpha_threshold = }; Difference = {np.median(alpha_list) - alpha_threshold}")
    print(f"\n{pvalue = }")
    #print(f"\n{pvalue = };{ci = }; GOOD {prop_mean*100 = }%")

    if (pvalue < (1 - sigma_n[n_sigma_away]/100)):
        # Manually mult by 100
        print(f"{pvalue = } FALSE ALARM IS **{n_sigma_away} SIGMA SIGNIFICANT** (GOOD)")
    else:
        print(f"{pvalue = } FALSE ALARM IS **NOT {n_sigma_away} SIGMA SIGNIFICANT** (BAD)")
        for value in sigma_n.values():
            if pvalue < (1 - value / 100):
                actual_away = list(sigma_n.keys())[list(sigma_n.values()).index(value)]
                print(f"ONLY **{actual_away} SIGMA SIGNIICANT**")
                break
        else:
            print("NOT SIGNIFICANT **AT ALL** (VERY BAD)")

    print(f"\nSETUP USED {repr(observing_setup)}; {repr(cv_object)}")
    # Plot stuff
    choice = np.random.randint(0, iter_N)
    (mag_series, errors)= mag_series_list[choice]
    (freq, power) = spectrum_list[choice]
    fig, ax = plt.subplots(4)

    #Standardise
    mean = np.mean(alpha_list)
    std = np.std(alpha_list)
    alpha_list = (alpha_list - mean)/std


    # Compute histogram of false alarm list
    #alpha_list = alpha_list[alpha_list < hist_range[1]]
    # Freedman-Diaconis rule
    q25, q75 = np.percentile(alpha_list, [25, 75])
    bin_width = 2 * (q75 - q25) * (len(alpha_list) ** (-1/3))
    print(f"{bin_width = }")
    bins = round((alpha_list.max() - alpha_list.min()) / bin_width)
    print(f"{bins = } {len(alpha_list) = }; difference = {bins - len(alpha_list)}")
    
    #percentiles = np.percentile(alpha_list, np.linspace(0, 100, 81))  # 20 adaptive bins
    transformed= alpha_list
    #transformed = -np.log(alpha_list + 0.5) / np.log(10)
    ax[0].hist(transformed, density = False, bins = 12)
   
    sm.qqplot(transformed, line="45", ax = ax[1])
    #print(f"{errors = }")
    predicted_freq = freq[power == power.max()][0]
    print(f"\nShown maximum power freq --> Period = {(1/predicted_freq).to(u.min)}")
    ax[2].errorbar(t, mag_series, errors)
    # Lomb-Scargle
    ax[3].plot(freq,power)


    plt.show()
if __name__ == "__main__":
    main()
