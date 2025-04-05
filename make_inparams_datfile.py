import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import matplotlib as mpl
import astropy.units as u
from astropy.table import Table, Column, MaskedColumn
from numpy import ma

from setup_ext_data import get_stars_list

def get_f19tostarname(f19starname):
    if "(" in f19starname:
        index = [idx for idx,s in enumerate(f19starname) if s == "("][0]
        f19starname = f19starname[:index-1]

    if f19starname.startswith('HD'):
        starname = f19starname[:2].lower()+f19starname[2:]
        starname = starname.replace(" ", "")
    elif f19starname.startswith('BD'):
        starname = f19starname[:2].lower() + f19starname[2:]
        starname = starname.replace(" ", "d")
    elif f19starname.startswith("CPD"):
        starname = f19starname[:3].lower()+f19starname[3:]
        starname = starname.replace(" ", "d")
    elif f19starname.startswith("NGC"):
        starname = f19starname[:3].lower()+f19starname[4:]
        starname = starname.replace(" ", "-vs")
    elif f19starname.startswith("Trump"):
        starname = f19starname.replace("Trumpler ", "trumpler")
        starname = starname.replace(" ", "-")
    elif f19starname.startswith("VS"):
        starname = f19starname[:3].lower()+"-"+f19starname[4:8].lower()+f19starname[9:]
    else:
        starname = f19starname[:3].lower()+f19starname[3:] 
    
    return starname


def redo_fullfile(data_set, file_path = "/Users/cgunasekera/extstar_data/", overwrite=False):
    if data_set == "F19": 
        starname_col=0
    elif data_set == "HighLowRv": 
        starname_col=1

    stars_list = get_stars_list(file_path, data_set=data_set, starname_col=starname_col)

    inpath =  f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/"
    inparam_filename = f"{inpath}{data_set}_new_inparams.dat" #remove _new after debug

    find_file = glob.glob(inparam_filename)
    if find_file != [] and overwrite==False:
        print(f"Abort! {inparam_filename} file already exists.")
        return

    # stellar
    logTeff_default=4.458226974274963
    logg_default=3.25
    logZ_default=0.0 #-0.8494850021680094
    vturb_default=6.0
    velocity_default=0.0 # km/s
    windamp_default=0.0
    windalpha_default=2.0

    # dust - values, bounds, and priors based on VCG04 and FM07 MW samples (expect Av)
    Av_default=1.0
    Rv_default=3.1
    C2_default=0.73
    B3_default=3.6
    C4_default=0.4
    xo_default=4.59
    gamma_default=0.89

    # gas
    vel_MW_default=0.0  # km/s
    logHI_MW_default=20.0
    vel_exgal_default=0.0  # km/s
    logHI_exgal_default=16.0

    # normalization value (puts model at the same level as data)
    #   value is depends on the stellar radius and distance
    #   radius would require adding stellar evolutionary track info
    norm_default=1.0

    paramnames = ["logTeff", "logg", "logZ", 
                  "vturb", "velocity", "windamp", "windalpha",
                  "Av", "Rv", "C2", "B3", "C4", "xo", "gamma",
                  "vel_MW", "logHI_MW", "vel_exgal", "logHI_exgal",
                  "norm"]

    default_vals = [logTeff_default, logg_default, logZ_default, 
                    vturb_default, velocity_default, windamp_default, windalpha_default,
                    Av_default, Rv_default, C2_default, B3_default, C4_default, xo_default, gamma_default,
                    vel_MW_default, logHI_MW_default, vel_exgal_default, logHI_exgal_default,
                    norm_default]
    
    default_params = ""
    for val in default_vals: 
        ext_val = f"{val}".ljust(17)
        default_params+=f"\t{ext_val}"
    
    header = "#star\t".ljust(17)
    for name in paramnames: header+=f"\t{name.ljust(17)}"

    with open(inparam_filename, "w") as writefile:
        writefile.write(header+"\n")

    for star in stars_list:
        with open(inparam_filename, "a") as writefile:
            writefile.write(star.ljust(17)+default_params+"\n")
        
    print(f"File written: {inparam_filename}")


if __name__ == "__main__":
    data_set = "HighLowRv"
    redo_fullfile(data_set=data_set, overwrite=True)