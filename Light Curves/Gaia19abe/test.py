import pandas as pd
import numpy as np

def main(filename, target_id = "T1", sep = "\t"):
    data = pd.read_csv(filename, sep = sep)

    #print(vars(data.keys()))
    print(data.keys()._data[0])

    data = data[[
        "J.D.-2400000",
        "AIRMASS",
        # "EXPTIME",
        # "FWHM_Mean",
        # "rel_flux_"+target_id,
        # "rel_flux_SNR_"+target_id,
        # "Source-Sky_"+target_id,
        # #"N_Src_Pixels_"+target_id,
        # "Source_Error_"+target_id,
        # "Source_SNR_"+target_id,
        # "Peak_"+target_id,
        # "Mean_"+target_id,
        # "Sky/Pixel_"+target_id,
        # #"N_Sky_Pixels_"+target_id,
        # "FWHM_"+target_id,
        # "Width_"+target_id,
        # "X-Width_"+target_id,
        # "Y-Width_"+target_id,
        # "Angle_"+target_id,
        # "Roundness_"+target_id
        ]]


if __name__ == "__main__":
    filename = "Gaia19abe_2019_01_04_measurements.xls"
    main(filename, "C1", "\t")
