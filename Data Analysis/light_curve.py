import pandas as pd
import numpy as np
import glob
import os.path

import astropy.time as astro_time
import astropy.timeseries as timeseries
import astropy.units as u
import matplotlib.pyplot as plt
import scipy

import astropy.stats

#def get_values(series):
#	return list(map(lambda t: t.value, series))

def get_time_values(ts, time_format):
    """
    Time format must match values
    set in Time or TimeDelta depending on use.
    """
    return list(map(lambda t: t.to_value(time_format), ts))


def make_power_spectrum(ls, max_freq):
    return ls.autopower(method = "fast", maximum_frequency = max_freq) # Nyquist factor (at least 2)

def non_aliased_power_spectrum(ls, sampling_frequency):
    nyquist_frequency = 0.5*sampling_frequency 
    max_freq = 5 * nyquist_frequency
    (freq, power) = make_power_spectrum(ls, max_freq)

    # Cut off below the sampling frequency to do with aliasing


    freq_idx = np.argwhere(abs(freq - nyquist_frequency) < 2/u.day)[0][0] # Property of periodogram is that it is symmetric
    # print(freq_idx)
    freq = freq[:freq_idx]
    power = power[:freq_idx]


    # ax.plot(freq, power, label="after")
    return (freq, power)	

def process_file(file_name, zero_pad_amount = 0):
    data = pd.read_csv(file_name, delimiter = "\t")

    data = data[[
        "J.D.-2400000",
        "AIRMASS",
        "EXPTIME",
        "FWHM_Mean",
        "rel_flux_T1",
        "rel_flux_SNR_T1",
        "Source-Sky_T1",
        #"N_Src_Pixels_T1",
        "Source_Error_T1",
        "Source_SNR_T1",
        "Peak_T1",
        "Mean_T1",
        "Sky/Pixel_T1",
        #"N_Sky_Pixels_T1",
        "FWHM_T1",
        "Width_T1",
        "X-Width_T1",
        "Y-Width_T1",
        "Angle_T1",
        "Roundness_T1"
        ]]


    time = data["J.D.-2400000"].to_numpy()
    mean_interval = np.median(np.diff(time))
    null = [time[0] - n*mean_interval for n in range(1, zero_pad_amount + 1)]

    time = np.concatenate([null, time])
    time = astro_time.Time(time, format = "mjd")

    zero_data = np.zeros(zero_pad_amount)
    print(len(null), len(zero_data))
    ts_data = {name: np.concatenate([zero_data, data[name]]) for name in data.keys()}
    ts = timeseries.TimeSeries(time = time, data = ts_data)

    #ts = ts.fold(period = 90*u.s)


    flux_names = ["Mean_T1", "Source_Error_T1"]
    fluxes = [ts[name] for name in flux_names]

    flux_name = "Mean_T1"

    ls = timeseries.LombScargle(ts["time"], ts[flux_name])

    # Want to remove frequency associated with the sampling frequency

    sampling_delta = abs(ts["time"][1] - ts["time"][0])
    print(f"{sampling_delta = }")
    sampling_frequency = 1/sampling_delta
    sampling_model = ls.model(ts["time"], frequency = sampling_frequency)


    ts[flux_name] = ts[flux_name] - sampling_model
    ls = timeseries.LombScargle(ts["time"], ts[flux_name])




    (freq, power) = non_aliased_power_spectrum(ls, sampling_frequency)


    nyquist_frequency = 0.5*sampling_frequency 
    max_freq = 5 * nyquist_frequency
    # Cut off below the sampling frequency to do with aliasing

    freq_idx = np.abs(freq - nyquist_frequency).argmin() # Property of periodogram is that it is symmetric

    print(f"{freq_idx = }")
    freq = freq[:freq_idx]
    print(f"{freq.max() = }")
    power = power[:freq_idx]

    return [ts, [freq, power], ls]

import astropy.timeseries.periodograms.lombscargle._statistics as _lombstats
def custom_false_alarm_probability(
        self,
        power,
        method="baluev",
        samples_per_peak=5,
        nyquist_factor=5,
        minimum_frequency=None,
        maximum_frequency=None,
        method_kwds=None,
        ):
    """False alarm probability of periodogram maxima under the null hypothesis.

    This gives an estimate of the false alarm probability given the height
    of the largest peak in the periodogram, based on the null hypothesis
    of non-varying data with Gaussian noise.

    Parameters
    ----------
    power : array-like
        The periodogram value.
    method : {'baluev', 'davies', 'naive', 'bootstrap'}, optional
        The approximation method to use.
    maximum_frequency : float
        The maximum frequency of the periodogram.
    method_kwds : dict, optional
        Additional method-specific keywords.

    Returns
    -------
    false_alarm_probability : np.ndarray
        The false alarm probability

    Notes
    -----
    The true probability distribution for the largest peak cannot be
    determined analytically, so each method here provides an approximation
    to the value. The available methods are:

    - "baluev" (default): the upper-limit to the alias-free probability,
      using the approach of Baluev (2008) [1]_.
    - "davies" : the Davies upper bound from Baluev (2008) [1]_.
    - "naive" : the approximate probability based on an estimated
      effective number of independent frequencies.
    - "bootstrap" : the approximate probability based on bootstrap
      resamplings of the input data.

    Note also that for normalization='psd', the distribution can only be
    computed for periodograms constructed with errors specified.

    See Also
    --------
    distribution
    false_alarm_level

    References
    ----------
    .. [1] Baluev, R.V. MNRAS 385, 1279 (2008)
    """
    if method != "bootstrap":
        raise NotImplementedError(
                "custom method only implemented for bootstrap method"
                )
    if self.nterms != 1:
        raise NotImplementedError(
                "false alarm probability is not implemented for multiterm periodograms."
                )
    if not (self.fit_mean or self.center_data):
        raise NotImplementedError(
                "false alarm probability is implemented "
                "only for periodograms of centered data."
                )

    fmin, fmax = self.autofrequency(
            samples_per_peak=samples_per_peak,
            nyquist_factor=nyquist_factor,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            return_freq_limits=True,
            )
    return fap_bootstrap(
            power,
            fmax=fmax,
            t=self._trel,
            y=self.y,
            dy=self.dy,
            normalization=self.normalization,
            )

def fap_bootstrap(
        Z, fmax, t, y, dy, normalization="standard", n_bootstraps=1000, random_seed=None
        ):
    # """Copied from https://github.com/astropy/astropy/blob/main/astropy/timeseries/periodograms/lombscargle/_statistics.py
    #   To return the entire list (distribution) of pvalues
    #   """
    """Bootstrap estimate of the false alarm probability."""
    pmax = _lombstats._bootstrap_max(t, y, dy, fmax, normalization, random_seed, n_bootstraps)

    # pmax is power
    # np.searchsorted finds index in array to place the power Z
    # so the fraction is the number of power values in pmax which pass(TODO?) the bootstrap
    # so to get a distribution of pmax values, taking only the values less than the power
    return pmax[pmax <= Z] / len(pmax)#1 - np.searchsorted(pmax, Z) / len(pmax)

def calculate_probability(snr, FAP_list, p_thresh = 0.05, conf_level = 99.7/100):
    """
    """
    # power_n = 1.5 # average value
    # prob_noise_only = 1/(1 + snr**power_n)

    # TODO: Not hardcoded; use the confidence level
    # This is the probability of getting the data in the first place.
    # Since the data has been sliced to be around the peak, this is the probability of getting a peak.

    # Photon statistics are Poisson by nature, which are asymptotically Gaussian
    # by the central limit theorem.
    # TODO: Determine value of const
    #   Should produce 1 if perfect signal (snr >~ 100)
    #   Should produce 0 if all (gaussian) noise
    #   Should produce 0.5 if amplitude (HWHM) of noise equal to dm
    print(f"{snr = }")
    const = 0.0001 # 2.77073586
    prob_signal_present = const/np.clip(6.5 - 2.5 * np.log10(snr), a_min = 0, a_max = np.inf) # negative values to 0
    print(f"{prob_signal_present = }")
    # Restrict probability to range [0,1]
    prob_signal_present = np.clip(prob_signal_present, a_min=0, a_max=1)
    print(f"{prob_signal_present = }")

    prob_noise_only = 1 - prob_signal_present
    prior_factor = prob_noise_only / prob_signal_present


    # TODO: Run bootstrap (Under signal-present assumption) to find the correct value
    prob_data_given_signal_present = 0.5
    # (Technically a list of bayes factors)
    bayes_factor = prob_data_given_signal_present / FAP_list

    prob_noise_only_given_peak = 1/(1 + bayes_factor * prior_factor)

    pvalue_dist = prob_noise_only_given_peak

    assert np.sum(pvalue_dist > 1) == 0, "Invalid adjusted probabilities: greater than 1"

    probability_threshold = p_thresh


    # p < p_thresh <==> p - p_thresh < 0
    test_result = scipy.stats.wilcoxon(pvalue_dist - probability_threshold, alternative = "less",
                                       correction = False, nan_policy = "omit")
    pvalue = test_result.pvalue

    return (pvalue, pvalue_dist)

def main(file_name_list, zero_padding = 0):
    print(file_name_list)
    results_list = [process_file(file, zero_padding) for file in file_name_list]

    results_list = results_list


    time_series_list = [x[0] for x in results_list]
    periodogram_list = [x[1] for x in results_list]
    lomb_scargle_list = [x[2] for x in results_list]

    n_periodograms = len(periodogram_list)

    # Rel_flux vs time, periodogram, FAP, pvalue
    fig, ax = plt.subplots(nrows = n_periodograms, ncols = 4)

    # Iterating over multiple axes so needs to be a list
    if n_periodograms == 1:
        ax = [ax]
    
    for (idx, axis) in enumerate(ax, start = 0):
        ts = time_series_list[idx]
        (freq, power) = periodogram_list[idx]
        ls = lomb_scargle_list[idx]

        # Analysis of highest peak (Hopefully that associated with the true signal)
        max_power = power.max()

        print(f"Default implementation {ls.false_alarm_probability(power.max(), method = 'bootstrap') = }")
        FAP_list = custom_false_alarm_probability(ls, max_power, method = "bootstrap")

        #"rel_flux_SNR_T1",
        #"Source_SNR_T1",
        snr = np.median(ts["Source_SNR_T1"])

        pvalue, pvalue_dist = calculate_probability(snr, FAP_list, p_thresh = 0.05, conf_level = 99.7/100)


        hist, bin_edges = astropy.stats.histogram(FAP_list, bins = "blocks")
        axis[2].stairs(hist, bin_edges)
        axis[2].set_title("False Alarm Probability")

        hist, bin_edges = astropy.stats.histogram(pvalue_dist, bins = "blocks")
        axis[3].stairs(hist, bin_edges)
        axis[3].set_title("Probability that Peak is Produced by Noise")


        print(f"{FAP_list.mean() = }; {pvalue = }; {pvalue_dist.mean() = }")

        # NOTE: Bayesian is *strictly* only for a *pure* sine wave
        power = np.exp(power)

        axis[1].plot(freq, power)
        axis[1].set_title("Lomb-Scargle Periodogram")

        time = get_time_values(ts["time"], time_format = "iso")
        flux = ts["rel_flux_T1"]

        axis[0].plot(time, flux)

        filename = os.path.basename(file_name_list[idx])
        axis[0].set_title(filename)

    plt.show()

if __name__ == "__main__":
    # Directory filter for varying workspaces
    REMOVE_CODING = False

    files = glob.glob(r"..\**\*measurement*.xls", recursive = True)
    files.extend(glob.glob(r"..\**\*measurement*.csv", recursive = True))

    if REMOVE_CODING:
        files = list(filter(lambda s: not ("Coding" in s), files))

    assert files != [], "No measurement files found"
#    print(list(filter(lambda s: "abe" in s, files)))
    # exit()
    zero_padding = 0

    main(files, zero_padding = int(zero_padding))
