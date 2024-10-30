import os
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


def redo_fullfile(data_set, file_path = "/Users/cgunasekera/extstar_data/"):
    if data_set == "F19": 
    #    stars_file = "f19_star_inparams.dat"
        starname_col=0
    elif data_set == "HighLowRv": 
    #    stars_file = "HighLowRv_HST.dat"
        starname_col=1

    stars_list = get_stars_list(file_path, data_set=data_set, starname_col=starname_col)

    inpath =  f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/"
    inparam_filename = f"{inpath}{data_set}_inparams.dat"
    
    i = 0
    with open(inparam_filename, "w") as writefile:
        writefile.write("#star\tlogTeff\tlogg\tlogZ\tAv\tRv\tC2\tC3\tC4\tx0\tgamma\tHI_gal\tHI_mw\n")

    for (current_dir, dirs, files) in os.walk(inpath, topdown = 'true'):
        if current_dir == inpath:
            for file in files:
                starname = file.split("_")[0]
                if file == f"{starname}_fit_params_optimizer.dat" and (starname in stars_list): #and (starname[:-4] not in not_tlusty):
                    with open(inpath+file) as datfile:
                        g=datfile.readlines()
                    parameters = g[1:]
                    if len(parameters) == 1:
                        continue
                    for j,param in enumerate(parameters): 
                       parameters[j] = param.split()[0].ljust(25," ")
                    parameters = "\t".join(parameters) + "\n"
                    starname = starname.ljust(14, " ")
                    with open(inparam_filename, "a") as writefile:
                        writefile.write(starname+"\t"+parameters)
                    i += 1
    
    print(i, "stars added")




if __name__ == "__main__":
    data_set = "HighLowRv"
    redo_fullfile(data_set=data_set)
    