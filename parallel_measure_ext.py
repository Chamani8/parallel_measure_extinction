import glob
import os
#import pkg_resources
#import argparse
import subprocess
from multiprocessing import Process
import multiprocessing as mp
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import matplotlib as mpl
import astropy.units as u
import emcee
import warnings

from dust_extinction.averages import B92_MWAvg
from dust_extinction.averages import G03_SMCBar
from dust_extinction.parameter_averages import G23

from read_data import DATstarname
from read_data import get_model_data

from plot_extinction_curves import plot_data_model
from plot_extinction_curves import ext_fit_plot
from setup_ext_data import get_stars_list

from measure_extinction.stardata import StarData
from measure_extinction.extdata import ExtData
from measure_extinction.utils.fit_model import FitInfo
from astropy.modeling.fitting import LevMarLSQFitter, FittingWithOutlierRemoval

from measure_extinction.utils.fit_model import get_best_fit_params, get_percentile_params
from astropy.table import QTable


def fix_photom_data(fstarname, path, band):
    with open(path+fstarname) as datfile:
            f=datfile.readlines()

    i=0
    for line in f:
        if line.startswith("#"+band) or line.startswith("# "+band):
            Vpos = line.find(band)
            f[i] = f[i][Vpos:]

        if f[i].startswith(band):
            eqpos = line[:7].find("=")
            pmpos = line[:7].find("+/-")
            colpos = line[:7].find(";")

            if eqpos < 0:
                flist = f[i].split(" ")
                f[i] = flist[0] + " = " + " ".join(flist[1:])

            if pmpos < 0:
                flist = f[i][:colpos].split()
                if len(flist) > 3:
                    f[i] = " ".join(flist[:3]) + " +/- " + flist[-1]
                elif len(flist) == 3:
                    f[i] = " ".join(flist[:3]) + " +/- 0.03"
                f[i] += line[colpos:]
        i+=1
    
    with open(path+fstarname, 'w') as datfile:
         datfile.writelines(f)
    
    


def get_f19starname(starname):
    if starname.startswith('hd'):
        f19starname = starname[:2].upper()+' '+starname[2:]
    elif starname.startswith('bd'):
        f19starname = starname[:2].upper()
        if starname[2] == "+":
            f19starname += starname[2:]
        else:
            f19starname += "+"+starname[2:] #need to edit later
        f19starname = f19starname.replace("d", " ")
    elif starname.startswith("cpd"):
        f19starname = starname[:3].upper()+starname[3:6]+" "+starname[7:]
    elif starname.startswith("ngc"):
        f19starname = starname[:3].upper()+" "+starname[3:]
        f19starname = f19starname.replace("-vs", " ")
    elif starname.startswith("trump"):
        f19starname = starname.replace("trumpler", "Trumpler ")
        f19starname = f19starname.replace("-", " ")
    else:
        f19starname = starname[:3].upper()+starname[3:] 
    
    return f19starname

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


def evaluate_SED_fit(reddened_star, modinfo, params, fit_range="g23", velocity=0):
    warnings.filterwarnings("ignore")
    
    # intrinsic sed
    modsed = modinfo.stellar_sed(params[0:3], velocity=velocity)

    # dust_extinguished sed
    ext_modsed = modinfo.dust_extinguished_sed(params[3:10], modsed, fit_range=fit_range)

    # hi_abs sed
    hi_ext_modsed = modinfo.hi_abs_sed(
        params[10:12], [velocity, 0.0], ext_modsed
    )

    residual_max_full = []
    for cspec in modinfo.fluxes.keys():
        if cspec == "BAND":
            continue

        norm_model = np.average(hi_ext_modsed["BAND"])
        norm_data = np.average(reddened_star.data["BAND"].fluxes).value

        residual_fluxes = []
        for idx, wave in enumerate(modinfo.waves[cspec]):
            for ind,star_wave in enumerate(reddened_star.data[cspec].waves):
                if star_wave == wave:# and star_wave.value > 0.35 and star_wave.value < 0.8:
                    residual_fluxes.append((reddened_star.data[cspec].fluxes.value[ind]-(hi_ext_modsed[cspec][idx] * norm_data / norm_model))*100/reddened_star.data[cspec].fluxes.value[ind])
   
        residual_fluxes = np.array(residual_fluxes)
        max_resid = max(residual_fluxes[~np.isnan(residual_fluxes)])
        residual_fluxes [residual_fluxes == -np.inf] = 0
        min_resid = abs(min(residual_fluxes[~np.isnan(residual_fluxes)]))

        residual_max_full.append(max(max_resid, min_resid))
    
    max_resid = np.max(residual_max_full)

    return max_resid
    

def get_sptype(fstarname, file_path, sptype_name="sptype"):
    lines = "a a"
    with open(f"{file_path}DAT_files/"+fstarname, "r") as f:
        while not lines.split()[0] == sptype_name:
            lines = f.readline()

            if lines == "":
                if fstarname != "bd_71d92.dat": print("Error: No Spectral Type Found! \t", fstarname)
                break

    if fstarname == "bd_71d92.dat":
        return "B9V"
    
    sptype = lines.split()[2]
    letter_type = sptype[0:2]
    found_roman = False
    j = 0
    for i,k in enumerate(sptype):
        if (k == "I" or k == "V" or k == "X") and found_roman == False:
            j=i
            found_roman = True
    
    if len(sptype) > j and (sptype[j] == "I" or sptype[j] == "V"): letter_type += sptype[j]
    if len(sptype) > j+1 and (sptype[j+1] == "I" or sptype[j+1] == "V"): letter_type += sptype[j+1]
    if len(sptype) > j+2 and (sptype[j+2] == "I" or sptype[j+2] == "V"): letter_type += sptype[j+2]
    
    if len(letter_type) < 3: letter_type += "V"

    return letter_type

def match_sptype(fstarname, file_path):
    lines = "nan nan"
    sptype = get_sptype(fstarname, file_path)

    try:
        with open(f"{file_path}stellar_classification_table.dat") as f:
            while not lines.split()[0].startswith(sptype):
                lines = f.readline()
    
    except IndexError:
        uvsptype = get_sptype(fstarname, file_path, "uvsptype")
        
        if uvsptype == "missing":
            return -1, -1

        lines = "nan nan"
        with open(f"{file_path}stellar_classification_table.dat") as f:
            while not lines.split()[0].startswith(uvsptype):
                lines = f.readline()
    
    except:
        raise IndexError("Oops! Could not find matching spectral type:"+sptype)
        #return -1, -1

    mass = float(lines.split()[1]) # Mstar/Msun
    radius = float(lines.split()[3]) # Rstar/Rsun
    Teff = float(lines.split()[4]) # Teff [K]

    #     G M* 
    # g = ----
    #      r^2
    Grav_const = 6.67430e-11 # [ m^3 kg^-1 s^-2 ]  https://physics.nist.gov/cgi-bin/cuu/Value?bg
    Rsun = 6.957e8 # [m] https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
    Msun = 1.989e+30 # [kg] https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html

    g = Grav_const*mass*Msun/((radius*Rsun)**2) #[m/s^2]
    logg = np.log10(g*1e2)

    return np.log10(Teff), logg

def plot_ext(extdata, params):
    fontsize = 18

    fig, ax = plt.subplots(figsize=(13, 10))

    # convert from E(l-V) to A(l)/A(V)
    extdata.columns["AV"] = (params[3], 0.0)
    extdata.trans_elv_alav()

    #print(extdata.waves["STIS_Opt"].to(u.micron).value > 0.3)
    #print(0.3*u.micron)

    extdata.plot(ax)#, linestyle="o-", color="yellowgreen", linewidth=2) #, alax=True)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$ [$\mu m$]", fontsize=1.3 * fontsize)
    ax.set_ylim(0.2, 1.8)
    ax.set_ylabel(r"$A(\lambda)/A(V)$", fontsize=1.3 * fontsize)
    ax.tick_params("both", length=10, width=2, which="major")
    ax.tick_params("both", length=5, width=1, which="minor")

    # plot known exitnction curves
    mod_x = np.arange(1.0 / 0.7873, 1.0 / 0.3402) / u.micron
    known_curve = B92_MWAvg() #G03_SMCBar()
    #ax.plot(1.0 / mod_x, known_curve(mod_x), "k:", label="B92_MWAvg")

    mod_x = np.arange(1.0, 1.0 / 0.3) / u.micron
    known_curve = G03_SMCBar()
    #ax.plot(1.0 / mod_x, known_curve(mod_x), "m:", label="G03_SMCBar")

    g23_rv31 = G23(Rv=params[4])
    ax.plot(1.0 / mod_x, g23_rv31(mod_x), "--", color="olivedrab", label="g23_rv"+str(params[4])[0:4], linewidth=2)
    g23_rv31 = G23(Rv=3.1)
    ax.plot(1.0 / mod_x, g23_rv31(mod_x), "-", color="olivedrab", label="g23_rv31", linewidth=2)

    ax.legend()


def create_ext(modinfo, reddened_star, fit_params, fit_range="g23", velocity=0, relband="V"):
    #Calculate and save the extinction curve
    # intrinsic sed
    modsed = modinfo.stellar_sed(fit_params[0:3], velocity=velocity)

    # create a StarData object for the best fit SED
    modsed_stardata = modinfo.SED_to_StarData(modsed)

    # create an extincion curve and save it
    extdata = ExtData()
    extdata.calc_elx(reddened_star, modsed_stardata, rel_band=relband) # E(lambda - V)
    col_info = {"av": fit_params[3], "rv": fit_params[4]}
    
    return extdata, col_info








def optimizer_fit(reddened_star, modinfo, fitinfo, params, exclude="BAND", initial_maxiter = 1000, fit_range="g23"):
    
    def nll(*args):
        return -fitinfo.lnprob(*args)

    # run the fit
    
    result = op.minimize(
        nll, params, method="Nelder-Mead", options={"maxiter": initial_maxiter}, args=(reddened_star, modinfo, fitinfo, fit_range)
    )

    maxiter = initial_maxiter
    while result["message"] == "Maximum number of iterations has been exceeded." and maxiter < 10000:
        maxiter += initial_maxiter/2
        result = op.minimize(
            nll, params, method="Nelder-Mead", options={"maxiter": maxiter}, args=(reddened_star, modinfo, fitinfo, fit_range)
        )

    # check the fit output
    #print(result["message"], "Max iterations = ", maxiter)

    return result["x"]


def get_inparams(starname, inparam_filepath = None):
    if inparam_filepath == None:
        inparam_filepath = file_path
    
    inparam_file = glob.glob(inparam_filepath+"*inparams.dat")[0]

    with open(inparam_file, "r") as f:
        line = f.readlines()

    i = 0
    try:
        while not starname in line[i].split():
            i +=1 
    except IndexError:
        print("Intial params not in file. Using default set.")
        return 0

    initial_params = [float(param) for param in line[i].split()[1:]]

    if len(initial_params) == 1:
        print("Incorrect value found for initial params. Using default set.")
        return 0
    else:
        return initial_params




def measure_extinction(starname, file_path,
                       # initial starting position; **customize for each star**
                       logZ=[0.15], Av=[1.4], Rv=[3.1], C2=[0.0], C3=[1.7], C4=[0.5], x0=[4.8], gamma=[0.8], HI_gal=[18.0], HI_mw=[18.0],
                       nwalkers = 100, nsteps = 500, burn = 500,
                       #nwalkers = 2 * ndim, nsteps = 50, burn = 10,
                       velocity=0, relband="V", exclude="BAND",
                       show_plots=False, do_save=True, do_mcmc=False, print_process=False,
                       mcmc_save_prefix="", save_prefix=""):

    not_tlusty = ["bd44d1080", "bd69d1231", "bd_71d92", "hd17443", "hd29647", "hd110336",
                  "hd112607",  "hd142165", "hd146285", "hd147196", "hd282485", "vss-viii10"]

    dstarname = DATstarname(starname)
    fstarname = f"{dstarname}.dat"

    if starname in not_tlusty:
        fit_range = "ALL"
    else:
        fit_range = "G23"

    if save_prefix != "": save_prefix = "_"+save_prefix
    if mcmc_save_prefix != "": mcmc_save_prefix = "_"+mcmc_save_prefix

    ## STAR DATA
    if print_process == True: print(starname, "Reading the star data")
    #Read in the star data
    photo_band = "K" #"J"
    reddened_star = StarData(fstarname, path=f"{file_path}DAT_files/", only_bands=photo_band)
    reddened_star_V = StarData(fstarname, path=f"{file_path}DAT_files/")#, only_bands=f"V,{photo_band}")

    if photo_band not in reddened_star.data["BAND"].get_band_names():
        reddened_star = StarData(fstarname, path=f"{file_path}DAT_files/", only_bands="B")
        reddened_star_V = StarData(fstarname, path=f"{file_path}DAT_files/", only_bands="V,B")

    band_names = reddened_star.data["BAND"].get_band_names() #these are: "U", "B", "V", "J", "H", "K", etc.
    band_names_V = reddened_star_V.data["BAND"].get_band_names()
    data_names = reddened_star_V.data.keys() #these are the data thats available, i.e. "STIS_Opt", "IUE", etc.

    if "V" not in reddened_star_V.data["BAND"].get_band_names():
        print("issue getting photometry: ", starname)
        return
    #    fix_photom_data(fstarname, path=f"{file_path}DAT_files/", band="V")

    reddened_star_masked = reddened_star

    # Using spectra classification to find the intial guesses for Teff, log g:
    logTeff, logg = match_sptype(fstarname, file_path)

    ## MODEL DATA
    if print_process == True: print(starname, "Gathering the model data")
    modinfo = get_model_data(file_path, data_names, logTeff, band_names=band_names[0])
    modinfo_V = get_model_data(file_path, data_names, logTeff, band_names=band_names_V)

    if print_process == True: print(starname, "\tcollected masked reddened_star data")

    ## FIT PARAMETERS
    if print_process == True: print(starname, "\tSetting up inital guesses for fit parameters")
    # parameter names
    pnames = ["logT","logg","logZ","Av","Rv","C2","C3","C4","x0","gamma","HI_gal","HI_mw"]
    # parameter dict
    params_dict = {"logTeff":[logTeff], "logg":[logg], "logZ":logZ, "Av":Av, 
                   "Rv":Rv, "C2":C2, "C3":C3, "C4":C4, "x0":x0, 
                   "gamma":gamma, "HI_gal":HI_gal, "HI_mw":HI_mw}
    
    # Get initial parameters from file
    params = get_inparams(starname, inparam_filepath=f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/")
    if params == 0:
        params=[0.]*12
        i=0
        for param_name,param_value in params_dict.items():
            if len(param_value) == 1:
                params[i] = param_value[0]
                if print_process == True: print(param_name, param_value)
            i+=1

    # some are based on the min/max of the stellar atmosphere grid
    plimits = [
            [modinfo.temps_min, modinfo.temps_max],  # log(Teff)
            [modinfo.gravs_min, modinfo.gravs_max],  # log(g)
            [modinfo.mets_min, modinfo.mets_max],    # log(Z)
            [0.0, 4.5],   # Av
            [2.7, 3.3], #[2.2, 5.7],   # Rv
            [-0.1, 5.0],  # C2
            [0.0, 2.5],   # C3
            [0.0, 1.0],   # C4
            [4.5, 4.9],   # xo
            [0.6, 1.5],   # gamma
            [17.0, 24.0], # log(HI) internal to galaxy
            [17.0, 22.0], # log(HI) MW foreground
        ]
    
    # add Gaussian priors based on prior knowledge
    ppriors = {}
    #ppriors["logZ"] = (0.15, 0.1)
    #ppriors["Rv"] = (3.1,0.01) #needed to keep within limit otherwise error

    #the following are a set of wavelength regions to not use
    # format: [1/lambda min, 1/lambda max]
    ex_regions = [
        [8.23 - 0.1, 8.23 + 0.1],  # geocoronal line
        [8.7, 10.0],  # bad data from STIS
        [3.55, 3.6],
        [3.80, 3.90],
        [4.15, 4.3],
        [6.4, 6.6],
        [7.1, 7.3],
        [7.45, 7.55],
        [7.65, 7.75],
        [7.9, 7.95],
        [8.05, 8.1],
    ] / u.micron

    if print_process == True: print(starname, "\tPreparing the weights for bad data")
    weights = {}
    for cspec in data_names:
        try:
            weights[cspec] = np.full(len(reddened_star_masked.data[cspec].fluxes), 0.0)
            gvals = reddened_star_masked.data[cspec].npts > 0

            weights[cspec][gvals] = 1.0 / reddened_star_masked.data[cspec].uncs[gvals].value

            x = 1.0 / reddened_star_masked.data[cspec].waves
            for cexreg in ex_regions:
                weights[cspec][np.logical_and(x >= cexreg[0], x <= cexreg[1])] = 0.0
        
        except AttributeError:
            weights[cspec] = 0.0
            continue

    #params = [4.276330582403116, # logT
    #          4.187052627352481, # logg
    #          0.17819755049681846, # logZ
    #          1.003260986202839, # Av
    #          3.2999999999933536, # Rv
    #          0.00021436620995938648, # C2
    #          1.444397708840993, # C3
    #          0.5368114877272627, # C4
    #          4.669612723610933, # x0
    #          0.8348369234206559, # gamma
    #          18.899086120531816, # HI_gal
    #          18.5242525755772] # HI_mw

    if print_process==True: print(starname, '\tRunning fit optimizer')
#    if run_grid == False:
#        if print_process==True: print(starname, 'Fitting single set of parameters')
    ppriors["logT"] = (params[0], 0.1)
    ppriors["logg"] = (params[1], 0.1)

    fitinfo = FitInfo(
            pnames,
            plimits,
            weights,
            parameter_priors=ppriors,
            stellar_velocity=velocity,
        )
    
    optimized_params = optimizer_fit(reddened_star_masked, modinfo, fitinfo, params, exclude=exclude, fit_range=fit_range)

    fit_params = optimized_params
    params_best = optimized_params
    inparams_best = params
    pnames_extra = pnames
    
    lnprob = fitinfo.lnprob(optimized_params, reddened_star_masked, modinfo, fitinfo, fit_range=fit_range)
    max_resid = evaluate_SED_fit(reddened_star, modinfo, fit_params, fit_range=fit_range, velocity=velocity)

#    print(logTeff, lnprob, max_resid)

    if max_resid < 20:
        run_grid = False
    else:
        run_grid = True
        
    if run_grid == True:
        if print_process==True: print(starname, 'Fitting grid of intial parameters')
        Teff_grid = np.linspace(np.log10((10**logTeff) - 3000), np.log10((10**logTeff) + 3000), 4)
        lnprob = 0.
        grid_i = 0
        for T in Teff_grid:
            params[0] = T
            ppriors["logT"] = (params[0], 0.1)

            #Package the fit info needed. FitInfo class defines the likelihood functions as well.
            fitinfo = FitInfo(
                pnames,
                plimits,
                weights,
                parameter_priors=ppriors,
                stellar_velocity=velocity,
            )
            optimized_params = optimizer_fit(reddened_star_masked, modinfo, fitinfo, params, exclude=exclude, fit_range=fit_range)

            new_max_resid = evaluate_SED_fit(reddened_star, modinfo, optimized_params, fit_range=fit_range, velocity=velocity)

            if max_resid > new_max_resid:
                fit_params = optimized_params
                params_best = optimized_params
                inparams_best = params
                best_grid_index = grid_i
                pnames_extra = pnames
                max_resid = new_max_resid
                lnprob = fitinfo.lnprob(optimized_params, reddened_star_masked, modinfo, fitinfo, fit_range=fit_range)
            grid_i += 1
    params = params_best

    # print the best 
    if print_process == True or do_save == True:
        if do_save == True:
            f = open(f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/{starname}_fit_params_optimizer{save_prefix}.dat", "w")
            f.write("# best fit\n")
        print("Optimizer best fit params")
        for k, val in enumerate(params_best):
            print("{} # {}".format(val, pnames_extra[k]))
            if do_save == True: f.write("{} # {}\n".format(val, pnames_extra[k]))
        print('lnprob:\t', lnprob)
    
    #Plot the spectra
    # plot optimizer/minimizer best fit
    plot_data_model(reddened_star, modinfo, params, fit_range=fit_range, velocity=velocity, plot_title="Initial guess, Optimizer Fit")
    if do_save == True:
        plt.savefig(f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/plots/{starname}_stis_Opt_optimizer{save_prefix}_{datetime.today().strftime('%Y-%m-%d')}.png")

    ####################################
    #       FIT EXTINCTION CURVE       #
    ####################################

    # Create the extinction curve
    #returns(waves, E(waves - relband), uncs, npts, names)
    extdata, col_info = create_ext(modinfo_V, reddened_star_V, fit_params, fit_range=fit_range, velocity=velocity, relband=relband)
    extdata.save(f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/{starname}{save_prefix}_ext_optimizer"+".fits")
    # Fit the extinction curve
    ext_fit_plot(extdata=extdata, plot_title=starname)#f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/", starname+save_prefix+"_ext_optimizer"+".fits", show_plots=show_plots, do_save=do_save)

    ##Plot the extinction curve
    plot_ext(extdata, params)
    if do_save == True:
        plt.savefig(f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/plots/{starname}_ext_optimizer{save_prefix}.png")

    if show_plots == True:
            plt.show()
        
    if do_mcmc == False:
        return

    ###############################################################
    #####   Run emcee MCMC sampler to define uncertainties   ######
    ###############################################################
    if print_process==True: print(starname, 'Starting MCMC fitting')
    #params = save_init_params
    p0 = params
    ndim = len(p0)
    nwalkers = 2 * ndim
    
    #nsteps = 50
    #burn = 10

    #nwalkers = 700
    #nsteps = 800
    #burn = 1500

    if print_process == True: 
        print("Doing:  nwalkers=",nwalkers,"\tnsteps=",nsteps,"\tburn=",burn)
        print("setting up the walkers to start \"nea\" the inital guess")
    # setting up the walkers to start "near" the inital guess
    p = [p0 * (1 + 0.01 * np.random.normal(0, 1.0, ndim)) for k in range(nwalkers)]
    if print_process == True: print("Setting up the MCMC sampler")
    # setup the sampler
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, fitinfo.lnprob, args=(reddened_star_masked, modinfo, fitinfo)
    )

    # burn in the walkers
    if print_process == True: print("Burning in the walkers")
    
    pos, prob, state = sampler.run_mcmc(p, burn)

    if print_process == True: print("Rest the sampler")
    # rest the sampler
    sampler.reset()
    
    if print_process == True: print('Running full MCMC')
    # do the full sampling
    pos, prob, state = sampler.run_mcmc(pos, nsteps, rstate0=state)
    
    prob = [p_value for p_value in prob if p_value > -np.inf]
    prob = np.mean(prob)
    #print("prob", prob)
    
    # create the samples variable for later use
    samples = sampler.chain.reshape((-1, ndim))
    
    # get the best fit values
    pnames_extra = pnames + ["E(B-V)", "N(HI)/A(V)", "N(HI)/E(B-V)"]
    params_best = get_best_fit_params(sampler)
    fit_params = params_best

    if print_process == True: 
        print("MCMC best params")
        print(params_best)

    # get the 16, 50, and 84 percentiles
    params_per = get_percentile_params(samples)

    # save the best fit and p50 +/- uncs values to a file
    # save as a single row table to provide a uniform format
    if do_save == True:
        f = open(f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/{starname}_fit_params_mcmc.dat", "w")
        f.write("# best fit, p50, +unc, -unc\n")
        for k, val in enumerate(params_per):
            if print_process == True: 
                print(
                    "{} {} {} {} # {}".format(
                        params_best[k], val[0], val[1], val[2], pnames_extra[k]
                    )
                )
            f.write(
                "{} {} {} {} # {}\n".format(
                    params_best[k], val[0], val[1], val[2], pnames_extra[k]
                )
            )
    
    # intrinsic sed
    extdata, col_info = create_ext(modinfo, reddened_star, fit_params, fit_range=fit_range, velocity=velocity, relband=relband)
    if do_save == True:
        extdata.save(f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/"+starname+mcmc_save_prefix+"_ext.fits", column_info=col_info)

    plot_data_model(reddened_star, modinfo, fit_params, velocity, plot_title="MCMC Fitting")
    if do_save == True:
        plt.savefig(f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/plots/{starname}_stis_Opt_mcmc{mcmc_save_prefix}.png")

    if do_save == True:
        plt.savefig(f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/plots/{starname}_residual_mcmc{mcmc_save_prefix}.png")

    plot_ext(extdata, params)
    if do_save == True:
        plt.savefig(f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/plots/{starname}_ext_mcmc{mcmc_save_prefix}.png")

    if show_plots == True:
        plt.show()

    
def parse_stars_list(file_path, stars_list):
    for star in stars_list:
        try:
            print("Measuring Extinction for ", star)
            measure_extinction(star, file_path)
        except:
            print(star, "failed to run!")




############################################
######       GLOBAL VARIABLES       ########
############################################

file_path = "/Users/cgunasekera/extstar_data/"
data_set = "HighLowRv"
# OPTIONS:
#   "F19"
#   "HighLowRv"

inpath = f"{file_path}DAT_files/STIS_data/" #location of original reduced data, *.mrg files
# OPTIONS:
#   F19 STIS:   "DAT_files/STIS_Data/Orig/Opt/"
#   HighLowRv:  "DAT_files/HighLowRv_HST_data/Orig/"

to_do = "Measure Extinction"  # options:
                # 1. "Convert .mrg to .fits"
                # 2. "Measure Extinction"


###########################################
################  MAIN BODY ###############
###########################################

if __name__ == "__main__":

    stars_list = ["csi-27-07550"]
    #, 'hd38087', 'hd54439', 'hd62542', 'hd68633', 'hd70614'] #VSS-viii10
    #['hd13338', 'hd14250', 'hd14321', 'hd27778', 'hd29647']
    exclude_stars = ["hd37023", "hd200775", "walker67"]
#    not_tlusty = ["bd44d1080", "bd69d1231", "bd_71d92", "hd17443", "hd29647", "hd110336",
#                  "hd112607",  "hd142165", "hd146285", "hd147196", "hd282485"
#                  #, "ngc2244-vs32", "bd04d1299s",
#                  #"cpd-69d416", "ngc2264-vas4", "hd164865", "trumpler14-2", "hd99872", "hd73882", "hd281159"
#                  ]
    
    #TODO:
    # - record if fit needs v2 or v10 get from file.

    if data_set == "F19": starname_col = 0
    elif data_set == "HighLowRv": starname_col = 1

    sample_stars_list = get_stars_list(file_path=file_path, data_set=data_set, starname_col=starname_col)

    t0 = time.time()

    velocity = 0 # radial velocity from SIMBAD
    relband = "V"
    process_list = []

    # Else do for all stars in directory:
    if len(stars_list) == 1:
       measure_extinction(stars_list[0], file_path,
                          nsteps = 2000, burn = 100,
                          #Rv=[3.3], #logZ=[0.0, 0.],
                          show_plots=True,
                          do_mcmc=False,
                          print_process=True,
                          do_save=False,
                          save_prefix="",
                          mcmc_save_prefix=""
                          )

    else:
        if len(stars_list) < 1:
            for star in sample_stars_list:
                star_data = glob.glob(inpath+"{}*.fits".format(star))
                if len(star_data) > 0 and star not in exclude_stars:
                    stars_list.append(star)
                #elif len(star_data) == 0 and star not in exclude_stars:
                #    print(star)
                #    print(star_data)

        num_processes = mp.cpu_count()
        stars_per_process = int(len(stars_list)/num_processes)
        cpu_count2 = len(stars_list) - (num_processes*stars_per_process)
            
        j=0
        while not j==cpu_count2:
            sub_stars_list = stars_list[j*(stars_per_process+1):(j+1)*(1+stars_per_process)]
            p = Process(target=parse_stars_list, args=( file_path, sub_stars_list, 
                                                           )
                            )
            process_list.append(p)
            j += 1

        k=0
        if len(stars_list) > num_processes:
            while not k==num_processes-cpu_count2:
                sub_stars_list = stars_list[(k*stars_per_process)+(j*(stars_per_process+1)):((k+1)*stars_per_process)+(j*(stars_per_process+1))]
                p = Process(target=parse_stars_list, args=(file_path, sub_stars_list, ) 
                            )
                process_list.append(p)
                k += 1
        else:
            while not (k+j)==len(stars_list):
                sub_stars_list = stars_list[(k*stars_per_process)+(j*(stars_per_process+1)):((k+1)*stars_per_process)+(j*(stars_per_process+1))]
                p = Process(target=parse_stars_list, args=(file_path, sub_stars_list, )
                            )
                process_list.append(p)
                k += 1

        for i,p in enumerate(process_list):
            #print('Starting to fit ',stars_list[i])
            p.start()

        for i,p in enumerate(process_list):
            p.join()
        
    #for star in stars_list:
    #    print('Finished fitting ', star)

    t1 = time.time()
    total_time = t1-t0
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)
    print('Total time taken ', int(h), ' hours', int(m), 'minutes', int(s), 'seconds')

# blue plot = intrinisic SED of the stellar model without dust
# green plot = final best fit model with dust