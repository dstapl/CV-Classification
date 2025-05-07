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

from matplotlib.ticker import MaxNLocator

#def get_values(series):
#	return list(map(lambda t: t.value, series))

def get_time_values(ts, time_format):
    """
    Time format must match values
    set in Time or TimeDelta depending on use.
    """
    return list(map(lambda t: t.to_value(time_format), ts))


def make_power_spectrum(ls, max_freq, samples_per_peak = 5):
    return ls.autopower(method = "fast", maximum_frequency = max_freq, # Nyquist factor (at least 2)
                        samples_per_peak = samples_per_peak)

def non_aliased_power_spectrum(ls, sampling_frequency = None, factor = 5, max_freq = None,
                               samples_per_peak = 5):
    """If max_freq is set, this function will take
    the minimum of `factor * sampling_frequency` and `max_freq`
    """
    if max_freq is None:
        max_freq = 1000/u.day

    if sampling_frequency is None:
        (freq, power) = make_power_spectrum(ls, max_freq, samples_per_peak)
        return (freq, power)

    nyquist_frequency = 0.5*sampling_frequency 
    max_freq = min([factor * nyquist_frequency, max_freq])
    (freq, power) = make_power_spectrum(ls, max_freq, samples_per_peak)

    # Cut off below the sampling frequency to do with aliasing


    freq_idx = np.argwhere(abs(freq - nyquist_frequency) < 2/u.day)[0][0] # Property of periodogram is that it is symmetric
    # print(freq_idx)
    freq = freq[:freq_idx]
    power = power[:freq_idx]


    # ax.plot(freq, power, label="after")
    return (freq, power)	


def read_necessary_columns(file_name, target_id = "T1", sep = "\t"):
    data = pd.read_csv(file_name, sep = sep)

    #print(data.keys())

    #print(data.keys())

    data = data[[
        "J.D.-2400000",
        "AIRMASS",
        "EXPTIME",
        "FWHM_Mean",
        "rel_flux_"+target_id,
        "rel_flux_err_"+target_id,
        "rel_flux_SNR_"+target_id,
        "Source-Sky_"+target_id,
        #"N_Src_Pixels_"+target_id,
        "Source_Error_"+target_id,
        "Source_SNR_"+target_id,
        # "Peak_"+target_id,
        "Mean_"+target_id,
        "Sky/Pixel_"+target_id,
        #"N_Sky_Pixels_"+target_id,
        "FWHM_"+target_id,
        "Width_"+target_id,
        "X-Width_"+target_id,
        "Y-Width_"+target_id,
        "Angle_"+target_id,
        "Roundness_"+target_id
        ]]

    #print(f"Accessed keys for {file_name}")

    return data


def remove_max_power_freq(ls, freq, power, n = 1, n_bootstraps = 1000, double_bootstrap = False):
    """MODIFIES LOMBSCARGLE INSTANCE

    Remove the frequency associated with the nth current highest powers
    """
    if n < 1:
        return ls, (freq, power), []

    removed_values = [] # (freq, power)
    for i in range(n):
        max_power = power.max()

        power_freq = freq[power == max_power].mean()

        (data_fap_value, lower_conf, upper_conf) = process_false_alarm(ls, power, n_bootstraps = n_bootstraps, double_bootstrap = double_bootstrap)

        if 0 < lower_conf:
            print(f"Halted at {i = }. False-alarm was too high: {data_fap_value} +/- {np.mean([lower_conf, upper_conf])}")
            break

        ls = remove_freq_signal(ls, power_freq)
        removed_values.append((power_freq, max_power, (data_fap_value, lower_conf, upper_conf)))

        freq, power = non_aliased_power_spectrum(ls, max_freq = freq.max())
    return ls, (freq, power), removed_values


def remove_freq_signal(ls, freq):
    """MODIFIES LOMBSCARGLE INSTANCE"""
    # Make sure to not modify original values yet
    #time = np.array(ls.t)
    time = ls.t
    #print(time)
    flux = np.array(ls.y)
    #errors = np.array(ls.dy)

    # try:
    #     print(f"{freq = }")
    #     freq = freq.tolist()
    # except:
    #     pass

    #print(f"{freq = }; {freq.ndim = }")
    sampling_model = ls.model(time, frequency = freq)


    ls.y = flux - sampling_model
    #ls = timeseries.LombScargle(time, ls.y)
    return ls # Not needed but just for sanity

def process_file(file_name, samples_per_peak = 5):
    file_errors = []
    target_id_list = ["T1", "C1"]
    target_id = ""

    found_separator = False

    data = []
    for target in ["T1", "C1"]:
        for sep in ["\t", ",", "\t\t"]:
            try: 
                data = read_necessary_columns(file_name, target_id = target,
                                              sep = "\t")
                target_id = target
                found_separator = True
    #            print(f"Separator was {repr(sep)}.")
                break
            except Exception as e:
                # try:
                #     data = read_necessary_columns(file_name, sep = ",")
                #     target_id = target
                #     break
                # except Exception as e2:
                #     file_errors.append(e2)
                file_errors.append(e)

        if found_separator:
            break
    else:
        if target_id == "":
            raise Exception(f"Target id not in list: {target_id_list}")
        else:
            assert len(data) != 0, "File error but there was still data"
            # No `continue` statements were procesed so
            # file is unexpected / malformed
            raise Exception(file_errors)

    #exit()
    time = data["J.D.-2400000"].to_numpy()
    #mean_interval = np.median(np.diff(time))

    time = astro_time.Time(time, format = "mjd")

    #ts_data = {name: np.concatenate([zero_data, data[name]]) for name in data.keys()}
    ts_data = {name: data[name] for name in data.keys()}
    ts = timeseries.TimeSeries(time = time, data = ts_data)

    #ts = ts.fold(period = 90*u.s)


    flux_names = ["Mean_"+target_id, "Source_Error_"+target_id]
    fluxes = [ts[name] for name in flux_names]

    flux_name = "Mean_"+target_id

    err_flux = ts["rel_flux_err_"+target_id]
    ls = timeseries.LombScargle(ts["time"], ts[flux_name],
                                dy = err_flux)

    # Want to remove frequency associated with the sampling frequency



    sampling_delta = abs(ts["time"][1] - ts["time"][0])
    #print(f"{sampling_delta = }")
    sampling_frequency = 1/sampling_delta

    ls = remove_freq_signal(ls, 1/sampling_delta)




    (freq, power) = non_aliased_power_spectrum(ls, sampling_frequency,
                                               samples_per_peak = samples_per_peak)


    nyquist_frequency = 0.5*sampling_frequency 
    max_freq = 5 * nyquist_frequency
    # Cut off below the sampling frequency to do with aliasing

    freq_idx = np.abs(freq - nyquist_frequency).argmin() # Property of periodogram is that it is symmetric

    #print(f"{freq_idx = }")
    freq = freq[:freq_idx]
    #print(f"{freq.max() = }")
    power = power[:freq_idx]

    return [ts, [freq, power], ls, target_id]


# MODIFIED FROM https://github.com/astropy/astropy/blob/main/astropy/timeseries/periodograms/lombscargle/_statistics.py
def _bootstrap_max(t, y, dy, fmax, normalization, random_seed, n_bootstraps=1000, double_bootstrap = False):
    """Generate a sequence of bootstrap estimates of the max."""
    #from .core import LombScargle

    rng = np.random.default_rng(random_seed)
    power_max = []

    if double_bootstrap:
        std_err = []

    fap_inner = []

    for _ in range(n_bootstraps):
        s = rng.integers(0, len(y), len(y))  # sample with replacement
        ls_boot = timeseries.LombScargle(
                t, y[s], dy if dy is None else dy[s], normalization=normalization
                )
        freq, power = ls_boot.autopower(maximum_frequency=fmax)

        sample_max_power = power.max()

        if double_bootstrap:
            inner_boot = []
            for _ in range(n_bootstraps):
                k = rng.integers(0, len(power), len(power))
                power_boot = power[k]

                inner_boot.append(power_boot.max())
            inner_boot = u.Quantity(inner_boot)
            inner_boot.sort()


            fap = 1 - np.searchsorted(inner_boot, sample_max_power) / len(inner_boot) # Fraction of values that pass

            fap_inner.append(fap)
            std_err.append(np.std(inner_boot))

        power_max.append(power.max())

    power_max = u.Quantity(power_max)
    power_max.sort()

    if double_bootstrap:
        std_err = u.Quantity(std_err)
        std_err.sort()

    if double_bootstrap == False:
        return power_max
    else:
        return (power_max, (std_err, fap_inner))

# MODIFIED FROM ASTROPY SOURCE TODO: LINK
def custom_false_alarm_probability(
        self,
        power,
        method="baluev",
        samples_per_peak=5,
        nyquist_factor=5,
        minimum_frequency=None,
        maximum_frequency=None,
        n_bootstraps = 1000,
        double_bootstrap = False,
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
    return fap_bootstrap(self, 
                         power,
                         fmax=fmax,
                         t=self._trel,
                         y=self.y,
                         dy=self.dy,
                         normalization=self.normalization,
                         n_bootstraps=n_bootstraps,
                         double_bootstrap = double_bootstrap,
                         )

def fap_bootstrap(ls,
                  Z, fmax, t, y, dy, normalization="standard", n_bootstraps=1000, random_seed=None,

                  significance = 0.05,
                  double_bootstrap = False,
                  ):
    # """Copied from https://github.com/astropy/astropy/blob/main/astropy/timeseries/periodograms/lombscargle/_statistics.py
    #   To return the entire list (distribution) of pvalues
    #   """
    """Bootstrap estimate of the false alarm probability."""

    print(double_bootstrap)
    res =  _bootstrap_max(t, y, dy, fmax, normalization, random_seed, n_bootstraps, double_bootstrap = double_bootstrap)
    if double_bootstrap:
        pmax, (std_err_dist, fap_inner_dist) = res
    else:
        pmax = res

    # pmax is power
    # np.searchsorted finds index in array to place the power Z
    # so the fraction is the number of power values in pmax which pass(TODO?) the bootstrap
    # so to get a distribution of pmax values, taking only the values less than the power
    # TODO: This is incorrect
    bootstrap_fap_value = 1 - np.searchsorted(pmax, Z) / len(pmax) # Fraction of values that pass

    print(f"{bootstrap_fap_value = }")
    if bootstrap_fap_value == 0:
        lower_conf, upper_conf, std_err, data_fap_value = 0,0,0, 0
    else:
        # Find confidence intervals for 95% of data (2.5% and 97.5%)

        n = len(y)

        if double_bootstrap:
            SE_boot = std_err_dist
        else:
            SE_boot = np.std(pmax)


        if n > 1:
            narrow_factor = np.sqrt(1 - 1/n) # sqrt( (n - 1)/n )
        else:
            # n = 1 is "infinite" narrowness which is indeterminate
            narrow_factor = 1

        # Correct for narrow bias ( Only really relevant for small n ) 
        SE_boot = SE_boot / narrow_factor

        # Input Z is power.max()
        data_fap_value = ls.false_alarm_probability(Z, method = "bootstrap", 
                                               method_kwds = {"n_bootstraps": n_bootstraps})

        #bias = fap_value - data_mean

        if double_bootstrap:
            t_dist_boot = (fap_inner_dist - data_fap_value) / SE_boot
            t_dist_boot.sort()
            #print(f"{t_dist_boot = }")
        else:
            t_dist_boot = (bootstrap_fap_value - data_fap_value) / SE_boot

        # upper_q = 1 - np.searchsorted(t_dist_boot, lower_power)/ len(pmax)
        # lower_q = 1 - np.searchsorted(t_dist_boot, upper_power)/ len(pmax)
        t_lower = np.percentile(t_dist_boot, significance / 2)
        t_upper = np.percentile(t_dist_boot, 1 - significance / 2)


        #print(f"{t_lower = }; {t_upper = }")
        # If one or both are negative

        # mean - q_1-alpha/2 * SE, mean + q_alpha/2 * SE
        # mean - upper *SE, mean + lower *SE
        if double_bootstrap:
            SE_boot = SE_boot.mean()

        lower_conf = data_fap_value - t_upper * SE_boot
        upper_conf = data_fap_value + t_lower * SE_boot


        if upper_conf < lower_conf:
            lower_conf, upper_conf = upper_conf, lower_conf

    print(bootstrap_fap_value, data_fap_value, lower_conf, upper_conf)

    if not double_bootstrap:
        fap_inner_dist = []

    return (data_fap_value, lower_conf, upper_conf), (bootstrap_fap_value, fap_inner_dist, pmax)

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

    pvalue_list = prob_noise_only_given_peak

    assert np.sum(pvalue_list > 1) == 0, "Invalid adjusted probabilities: greater than 1"

    probability_threshold = p_thresh


    # p < p_thresh <==> p - p_thresh < 0
    test_result = scipy.stats.wilcoxon(pvalue_list - probability_threshold, alternative = "less",
                                       correction = False, nan_policy = "omit")
    pvalue = test_result.pvalue

    return (pvalue, pvalue_list)

def process_false_alarm(ls, power, n_bootstraps = 1000, double_bootstrap = False):
    # Analysis of highest peak (Hopefully that associated with the true signal)
    max_power = power.max()

    print(f"Default implementation {ls.false_alarm_probability(power.max(), method = 'bootstrap') = }")

    (data_fap_value, lower_conf, upper_conf), _  = custom_false_alarm_probability(ls, max_power, method = "bootstrap", n_bootstraps = n_bootstraps, double_bootstrap = double_bootstrap)
     

    return (data_fap_value, lower_conf, upper_conf)

def plot_rows_columns(file_name_list: list[str],
                      results_list,
                      rows = "all",
                      columns = "all",
                      ax_list = [],
                      bins = "blocks",
                      xticks = 5,
                      yticks = 5,
                      n_bootstraps = 1000,
                      double_bootstrap = False):
    """
    Rows accepts either "all" to plot all results
    or part of a filename associated with measurement files.

    Column accepts the following:
    `["all", "signal", "periodogram", "false-alarm", "pvalue"]`

    Bins is number of histogram bins, or method such as Bayesian blocks (`"blocks"`)

    xticks is number of ticks to display on x-axis
    yticks is number of ticks to display on y-axis

    NOTE: `ax_list` assumes all axes come from the same figure
    """
    assert type(columns[0]) == str, f"Column or first element of columns is not str: {columns = }"
    # Python short circuits boolean operators

    assert type(columns[0]) == str, f"Column or first element of columns is not str: {columns = }"
    #assert (type(rows) == str) or len(list(filter(lambda x: type(x) != int, rows))) == 0, f"Rows is not a valid argument: {rows = }"


    n_plots = len(results_list)
    assert n_plots != 0, "Results list is empty for plotting"

    if type(rows) == str:
        assert rows == "all", f"Only `str` value for rows is `all`. Did you mean a list? {rows  = }"
    elif type(rows) == list:
        all_text = "\t".join(rows)
        for row in rows:
            assert row in all_text, f"Row value `{row}` not found in processed files: {file_name_list = }"


    allowed_columns = ["signal", "periodogram", "false-alarm", "pvalue"]
    if type(columns) == str:
        assert columns == "all", f"Only `str` value for column is `all`. Did you mean a list? {columns = }"
    elif type(columns) == list:
        for column in columns:
            assert column in allowed_columns, f"Column value `{column}` is not allowed. The options for column are {allowed_columns}"

    if type(rows) == list:
        rows = list(filter(lambda s: s != "all", rows))
        if len(rows) == 0:
            return # Nothing to plot

    if type(columns) == list:
        columns  = list(filter(lambda s: s != "all", columns))
        if len(columns) == 0:
            return # Nothing to plot





    if rows == "all":
        rows = list(range(n_plots))
    if columns == "all":
        columns = ["signal", "periodogram", "false-alarm", "pvalue"]


    n_rows, n_column = len(rows), len(columns)
    print(f"{n_rows = }, {n_column = }")
    if len(ax_list) == 0:
        fig, ax_list = plt.subplots(nrows = n_rows, ncols = n_column,
                                    figsize = (16, 8))
    else:
        fig = plt.gcf()


    # Iterating over multiple axes so needs to be a list
    if n_rows == 1:
        ax_list = [ax_list]



    long_cache = {obj_name: None for obj_name in rows}

    for (row_idx, axis) in enumerate(ax_list, start = 0):
        row_name = rows[row_idx]
        if not (row_name in rows):
            print(f"skipping {row_name = }")
            continue

        # Single column idx will always be 0
        if n_column == 1:
            axis = [axis]
        ts, (freq,power), ls, target_id = results_list[row_idx]


        cache = {column_name: None for column_name in allowed_columns}

        # TODO: CACHE RESULTS!
        for (column_idx, column) in enumerate(columns, start = 0):
            if column == "signal":
                if cache[column] is None:
                    time = get_time_values(ls.t, time_format = "unix")
                    flux = ts["rel_flux_"+target_id]
                    if True:#if (ls.dy is None) or (len(ls.dy) == 0):
                        err_flux = ts["rel_flux_err_"+target_id]
                    else:
                        err_flux = ls.dy
                        print(f"{err_flux = }")

                    cache[column] = (time, flux, err_flux)
                else:
                    time, flux, err_flux = cache[column]

                axis[column_idx].errorbar(time, flux, yerr = err_flux, fmt = "x")

            elif column == "periodogram":
                BAYESIAN = True
                if cache[column] is None:
                    if BAYESIAN:
                        power = np.exp(power)
                    else:
                        power = power
                    cache[column] = (freq, power)
                else:
                    freq, power = cache[column]

                # # NOTE: Bayesian is *strictly* only for a *pure* sine wave
                # power = np.exp(power)
                USE_PERIOD = True

                if USE_PERIOD:
                    period = 1/freq
                    axis[column_idx].plot(period, power)

                    axis[column_idx].set_xscale("log")
                    # axis[column_idx].set_yscale("log")

                    axis[column_idx].set_xlabel("Period [days] (log scale)")
                else:
                    axis[column_idx].plot(freq, power)

                    axis[column_idx].set_xscale("log")
                    # axis[column_idx].set_yscale("log")

                    axis[column_idx].set_xlabel("Frequency [1/day] (log scale)")

                axis[column_idx].set_title("Lomb-Scargle Periodogram")

            # elif column == "false-alarm":
            #     if cache[column] is None:
            #         (data_fap_value, lower_conf, upper_conf) = process_false_alarm(ls, power, n_bootstraps=n_bootstraps, double_bootstrap = double_bootstrap)
            #         cache[column] = FAP_list
            #     else:
            #         FAP_list = cache[column]
            #
            #     if len(FAP_list) != 0:
            #         print(f"{FAP_list.mean() = }")
            #         hist, bin_edges = astropy.stats.histogram(FAP_list, bins = bins)
            #         axis[column_idx].stairs(hist, bin_edges)
            #     else:
            #         print("WARNING false-alarm list is EMPTY")
            #
            #     axis[column_idx].set_title("False Alarm Probability")
            #
            # elif column == "pvalue":
            #     # Need to compute pvalue for false-alarm so combine both
            #
            #     if cache[column] is None:
            #         if cache["false-alarm"] is None:
            #             FAP_list = process_false_alarm(ls, power, n_bootstraps=n_bootstraps, double_bootstrap = double_bootstrap)
            #             cache["false-alarm"] = FAP_list
            #         else:
            #             FAP_list = cache["false-alarm"]
            #
            #         #"rel_flux_SNR_T1",
            #         #"Source_SNR_T1",
            #         snr = np.median(ts["Source_SNR_"+target_id])
            #         if len(FAP_list) != 0:
            #             pvalue, pvalue_list = calculate_probability(snr, FAP_list, p_thresh = 0.05, conf_level = 99.7/100)
            #         else:
            #             print("WARNING false-alarm list is EMPTY")
            #             pvalue, pvalue_list = (0, [])
            #
            #         cache[column] = (pvalue, pvalue_list, snr)
            #
            #     else:
            #         pvalue, pvalue_list, snr = cache[column]
            #
            #
            #     if len(pvalue_list) != 0:
            #         print(f"{pvalue = }; {pvalue_list.mean() = }")
            #         hist, bin_edges = astropy.stats.histogram(pvalue_list, bins = bins)
            #         axis[column_idx].stairs(hist, bin_edges)
            #     else:
            #         print("WARNING pvalue list is EMPTY")
            #
            #     axis[column_idx].set_title("Probability that Peak is Produced by Noise")
            #
            #
            # Set title of plot
            if column_idx == 0:
                curr_title = axis[column_idx].get_title()
                filename = os.path.basename(file_name_list[row_idx])
                axis[column_idx].set_title(filename + "\n" + curr_title)

        for subplot in axis:
            subplot.xaxis.set_major_locator(MaxNLocator(xticks))
            subplot.yaxis.set_major_locator(MaxNLocator(yticks))
        #     plt.setp(subplot.get_xticklabels(), rotation=30, ha='right')

        long_cache[row_name] = (cache, ts, ls, target_id)
    fig.tight_layout()

    return fig, ax_list, long_cache

def main(file_name_list):
    print(f"{file_name_list = }")

    results_list = []

    for file in file_name_list:
        try:
            result = process_file(file, samples_per_peak = 10)
        except Exception as e:
            print(f"ERROR in {file = }. SKIPPING\n\n", e, "\n\n")
            continue
        results_list.append(result)

    # n_periodograms = len(results_list)
    #
    # # Rel_flux vs time, periodogram, FAP, pvalue
    # fig, ax_list = plt.subplots(nrows = n_periodograms, ncols = 4,
    #                        figsize = (18, 8))
    #
    # # Iterating over multiple axes so needs to be a list
    # if n_periodograms == 1:
    #     ax_list = [ax_list]
    #

    #rows = "all"
    #columns = "all"
    rows = "all"#["18bfs", "19abe"]
    columns = ["signal", "periodogram", "signal"]#["signal", "periodogram", "pvalue"]
    n_xticks = 5
    n_yticks = 5
    bins = "blocks" # 15
    n_bootstraps = 500 # WARNING: For *ALL* periodogram false-alarm probabilities

    double_bootstrap = True
    fig, ax_list, long_cache = plot_rows_columns(file_name_list, results_list,
                                                 rows = rows,
                                                 columns = columns,
                                                 xticks = n_xticks,
                                                 yticks = n_yticks,
                                                 bins = bins,
                                                 n_bootstraps = n_bootstraps,
                                                 double_bootstrap = double_bootstrap
                                                 )

    if rows == "all":
        rows = list(range(len(file_name_list)))

    for (obj, axis) in enumerate(ax_list):
        #print("\n\n\n", long_cache)
        (cache, ts, ls, target_id) = long_cache[rows[obj]]

        (freq_list, power_list) = cache["periodogram"]
        (ls, (freq_list, power_list), removed_values) = remove_max_power_freq(
                ls, freq_list, power_list, n = 6,
                n_bootstraps = n_bootstraps,
                double_bootstrap = double_bootstrap,
                )

        removed_freq = u.Quantity([x[0] for x in removed_values])
        removed_power = [x[1] for x in removed_values]
        removed_fap = [x[2] for x in removed_values]


        try:
            len(removed_fap[0]) # test
            # Hope it's not a single eleent list...
            for i in range(len(removed_fap)):
                removed_fap[i] = np.array(removed_fap[i])
        except:
            removed_fap = [np.array(removed_fap)]

        for (freq, fap) in zip(removed_freq, removed_fap):
            period = (1/freq).to(u.min)
            print(f"{period = }; {fap.mean()*100 = }%")

        axis[1].cla() # wipe previous iteration
        axis[1].plot(1/freq_list, power_list)
        axis[1].set_xscale("log")
        #axis[1].set_yscale("log")

        axis[2].cla() # wipe previous iteration
        time = get_time_values(ts["time"], time_format = "unix")
        #axis[2].errorbar(time, ls.y, yerr = ls.dy.value.tolist(), fmt = "x")
        axis[2].plot(time, ls.y)


    plt.show()

if __name__ == "__main__":
    # Directory filter for varying workspaces
    REMOVE_CODING = False

    files = glob.glob(r"..\**\*measurement*.xls", recursive = True)
    files.extend(glob.glob(r"..\**\*measurement*.csv", recursive = True))

    if REMOVE_CODING:
        files = list(filter(lambda s: not ("Coding" in s), files))

    main(files)
