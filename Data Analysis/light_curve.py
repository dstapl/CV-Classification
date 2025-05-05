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


def process_file(file_name, zero_pad_amount = 0):
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
    mean_interval = np.median(np.diff(time))
    null = [time[0] - n*mean_interval for n in range(1, zero_pad_amount + 1)]

    time = np.concatenate([null, time])
    time = astro_time.Time(time, format = "mjd")

    zero_data = np.zeros(zero_pad_amount)
    print(len(null), len(zero_data))
    ts_data = {name: np.concatenate([zero_data, data[name]]) for name in data.keys()}
    ts = timeseries.TimeSeries(time = time, data = ts_data)

    #ts = ts.fold(period = 90*u.s)


    flux_names = ["Mean_"+target_id, "Source_Error_"+target_id]
    fluxes = [ts[name] for name in flux_names]

    flux_name = "Mean_"+target_id

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

    return [ts, [freq, power], ls, target_id]

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


def plot_rows_columns(file_name_list: list[str], results_list, rows = "all", columns = "all", ax_list = [], bins = "blocks", xticks = 5, yticks = 5):
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


    def process_false_alarm(ls, power):
        # Analysis of highest peak (Hopefully that associated with the true signal)
        max_power = power.max()

        print(f"Default implementation {ls.false_alarm_probability(power.max(), method = 'bootstrap') = }")

        FAP_list = custom_false_alarm_probability(ls, max_power, method = "bootstrap")

        #print(f"{FAP_list.mean() = }; {pvalue = }; {pvalue_dist.mean() = }")

        return FAP_list


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
                    time = get_time_values(ts["time"], time_format = "unix")
                    flux = ts["rel_flux_"+target_id]
                    err_flux = ts["rel_flux_err_"+target_id]

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

                axis[column_idx].plot(freq, power)
                axis[column_idx].set_title("Lomb-Scargle Periodogram")

            elif column == "false-alarm":
                if cache[column] is None:
                    FAP_list = process_false_alarm(ls, power)
                    cache[column] = FAP_list
                else:
                    FAP_list = cache[column]

                print(f"{FAP_list.mean() = }")

                hist, bin_edges = astropy.stats.histogram(FAP_list, bins = bins)
                axis[column_idx].stairs(hist, bin_edges)
                axis[column_idx].set_title("False Alarm Probability")

            elif column == "pvalue":
                # Need to compute pvalue for false-alarm so combine both

                if cache[column] is None:
                    if cache["false-alarm"] is None:
                        FAP_list = process_false_alarm(ls, power)
                        cache["false-alarm"] = FAP_list
                    else:
                        FAP_list = cache["false-alarm"]

                    #"rel_flux_SNR_T1",
                    #"Source_SNR_T1",
                    snr = np.median(ts["Source_SNR_"+target_id])

                    pvalue, pvalue_dist = calculate_probability(snr, FAP_list, p_thresh = 0.05, conf_level = 99.7/100)

                    cache[column] = (pvalue, pvalue_dist, snr)
                else:
                    pvalue, pvalue_dist, snr = cache[column]

                print(f"{pvalue = }; {pvalue_dist.mean() = }")


                hist, bin_edges = astropy.stats.histogram(pvalue_dist, bins = bins)
                axis[column_idx].stairs(hist, bin_edges)
                axis[column_idx].set_title("Probability that Peak is Produced by Noise")


            # Set title of plot
            if column_idx == 0:
                curr_title = axis[column_idx].get_title()
                filename = os.path.basename(file_name_list[row_idx])
                axis[column_idx].set_title(filename + "\n" + curr_title)

        for subplot in axis:
            subplot.xaxis.set_major_locator(MaxNLocator(xticks))
            subplot.yaxis.set_major_locator(MaxNLocator(yticks))
        #     plt.setp(subplot.get_xticklabels(), rotation=30, ha='right')

    fig.tight_layout()


def main(file_name_list, zero_padding = 0):
    print(f"{file_name_list = }")

    results_list = []

    for file in file_name_list:
        try:
            result = process_file(file, zero_padding)
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
    rows = ["18bfs", "19abe"]
    columns = ["signal", "periodogram", "pvalue"]
    n_xticks = 5
    n_yticks = 5
    bins = "blocks" # 15
    plot_rows_columns(file_name_list, results_list,
                      rows = rows,
                      columns = columns,
                      xticks = n_xticks,
                      yticks = n_yticks,
                      bins = bins
    )

    plt.show()

if __name__ == "__main__":
    # Directory filter for varying workspaces
    REMOVE_CODING = False

    files = glob.glob(r"..\**\*measurement*.xls", recursive = True)
    files.extend(glob.glob(r"..\**\*measurement*.csv", recursive = True))

    if REMOVE_CODING:
        files = list(filter(lambda s: not ("Coding" in s), files))

    assert files != [], "No measurement files found"

    zero_padding = 0

    main(files, zero_padding = int(zero_padding))
