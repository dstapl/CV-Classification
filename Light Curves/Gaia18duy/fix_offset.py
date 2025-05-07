import pandas as pd
import numpy as np

def main(filename: str):
    data = pd.read_csv(filename, delimiter = "\t")
    
    flux = data.loc[:, ("rel_flux_T1")]

    jump_idx = np.where(flux >= 2)[0][0]

    curr = flux[jump_idx]
    prev = flux[jump_idx - 1]
    print(curr, prev)
    offset = curr - prev
    data.loc[jump_idx:, ("rel_flux_T1")] -= offset


    flux = data["rel_flux_T1"]

    #print(flux.mean())
   
    ext = filename.rfind(".NOTFINAL")
    out_name = filename[:ext]
    data.to_csv(out_name, sep = "\t")

if __name__ == "__main__":
        filename = "Gaia18duy_2018_12_24_BIGJUMP_measurements.xls.NOTFINAL"
        main(filename)
