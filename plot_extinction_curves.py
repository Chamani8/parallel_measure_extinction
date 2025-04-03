#################### PREAMBLES ###################
#import os
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import os
import glob
import warnings
from datetime import datetime
import pandas as pd
import astropy.units as u


from read_data import DATstarname
from read_data import get_model_data
from read_data import get_mask
from read_data import get_fit_band

from measure_extinction.stardata import StarData
from measure_extinction.extdata import ExtData
from measure_extinction.modeldata import ModelData
from measure_extinction.extdata import AverageExtData
from measure_extinction.model import MEModel

from dust_extinction.shapes import FM90

from astropy.stats import sigma_clip
from astropy.modeling.fitting import LevMarLSQFitter, FittingWithOutlierRemoval
from astropy.table import QTable
from astropy.modeling.models import (
    Drude1D,
    Polynomial1D,
    PowerLaw1D,
    # Legendre1D,
)


##################################
########     FUNCTIONS    ########
##################################
def get_modeldata(reddened_star,
         modtype="obsstars",
         modpath="./", #path to the model files
         modstr=None, #Alternative model string for grid (expert)
         picmodel=True, #Set to read model grid from pickle file
        ):
    if "BAND" in reddened_star.data.keys():
        band_names = reddened_star.data["BAND"].get_band_names()
    else:
        band_names = []
    data_names = list(reddened_star.data.keys())

    # model data
#    start_time = time.time()
    #print("Reading model files")
    if modtype == "whitedwarfs":
        modstr = "wd_hubeny_"
    else:
        modstr = "tlusty_"
    if modstr is not None:
        modstr = modstr

    bandnames=""
    datanames=""
    for band in band_names: bandnames += band
    for data in data_names:
        if data=="BAND":
            continue
        else:
            datanames += data

    picmodel_file = f"{modpath}/{modstr}{bandnames}_{datanames}_modinfo.pkl"
    picmodel_path = glob.glob(picmodel_file)

    if picmodel and picmodel_path != []:
        modinfo = pickle.load(open(picmodel_file, "rb"))
    else:
        try:
            files_and_dirs = os.listdir(modpath)
            tlusty_models_fullpath = glob.glob(f"{modpath}/{modstr}*.dat")
        except PermissionError:
            print(f"PermissionError: not permitted access {modpath}")
            return
        tlusty_models = [
            tfile[tfile.rfind("/") + 1 : len(tfile)] for tfile in tlusty_models_fullpath
        ]
        if len(tlusty_models) > 1:
            print(f"{len(tlusty_models)} model files found.")
        else:
            raise ValueError("no model files found.")

        # get the models with just the reddened star band data and spectra
        modinfo = ModelData(
            tlusty_models,
            path=f"{modpath}/",
            band_names=band_names,
            spectra_names=data_names,
        )
        pickle.dump(modinfo, open(picmodel_file, "wb"))
        print("Finished reading model files")
#        seconds = time.time() - start_time
#        hours, remainder = divmod(seconds, 3600)
#        minutes, seconds = divmod(remainder, 60)
#        hours = f"{int(hours)} hours, " if hours > 0 else ""
#        minutes = f"{int(minutes)} minutes, " if minutes > 0 else ""
#        print(f"Models pickled to {modstr}{starname}_modinfo.pkl, {hours}{minutes}{int(seconds)} seconds".strip())

    return modinfo

def float_param(par):
    try: 
        fpar = float(par.split()[0]) 
    except: 
        fpar = float(par.split("F")[0])
    return fpar

def read_fit_params(param_filename, fpath):
#    if to_execute.upper().split()[-2].startswith("PLOT") or to_execute.upper().split()[-2].startswith("OPT"): 
#        fit_type = "optimizer"
#    elif to_execute.upper().split()[-2].startswith("MCMC"):
#        fit_type = "mcmc"
#    filename = f"{starname}_fit_params_{fit_type}.dat"

    with open(fpath+param_filename, "r") as datfile:
            f=datfile.readlines()
    
    params = [float_param(par) for par in f[1:]]

    return params


def plot_residual(ax, reddened_star, modinfo, params, fit_range="g23", velocity=0, plot_title=""):

#    # plotting setup for easier to read plots
    fontsize = 18
#    font = {"size": fontsize}
#    mpl.rc("font", **font)
#    mpl.rc("lines", linewidth=1)
#    mpl.rc("axes", linewidth=2)
#    mpl.rc("xtick.major", width=2)
#    mpl.rc("xtick.minor", width=2)
#    mpl.rc("ytick.major", width=2)
#    mpl.rc("ytick.minor", width=2)
#
#    # setup the plot
#    fig, ax = plt.subplots(figsize=(13, 10))

    ax.hlines(0, 0.3, 2, linewidth = 2, colors = "grey")
    #ax.vlines(0.3645, 0.5, 2, linestyle="--", colors = "k")

    # intrinsic sed
    modsed = modinfo.stellar_sed(params[0:3], velocity=velocity)

    # dust_extinguished sed
    ext_modsed = modinfo.dust_extinguished_sed(params[3:10], modsed, fit_range=fit_range)

    # hi_abs sed
    hi_ext_modsed = modinfo.hi_abs_sed(
        params[10:12], [velocity, 0.0], ext_modsed
    )

    for cspec in modinfo.fluxes.keys():
        if cspec == "BAND":
            color = "y"
        else:
            color = "m"

        #mask = get_mask(reddened_star.data[cspec].fluxes)
        #reddened_star.data[cspec].fluxes[mask] = np.nan

        norm_model = np.average(hi_ext_modsed["BAND"])
        norm_data = np.average(reddened_star.data["BAND"].fluxes).value

        residual_fluxes = []
        residual_waves = []
        for idx, wave in enumerate(modinfo.waves[cspec]):
            for ind,star_wave in enumerate(reddened_star.data[cspec].waves):
                if star_wave == wave:
                    residual_fluxes.append((reddened_star.data[cspec].fluxes.value[ind]-(hi_ext_modsed[cspec][idx] * norm_data / norm_model))*100/reddened_star.data[cspec].fluxes.value[ind])
                    residual_waves.append(reddened_star.data[cspec].waves.value[ind])

        ax.plot(residual_waves, residual_fluxes, "o", label = cspec)

        if cspec == "STIS_Opt":
            #annotation_fontsize = 
            # Balmer Jump wavelen = 0.3645 micron
            # Paschen Jump wavelen = 0.8204 micron
            # H alpha wavelen = 0.656281 micron
            # H beta wavelen = 0.4861 micron
            # H gamma wavelen = 0.4340 micron
            lambda_shift = 0.0#05
            index_BaJ = [ind for ind,wave in enumerate(modinfo.waves[cspec].value) if wave > 0.3642 and wave < 0.3648]
            BaJ_flux = hi_ext_modsed[cspec][index_BaJ] * norm_data / norm_model
            ax.axvline(0.3645, color="grey")
            ax.annotate("Balmer\nJump", xy=(0.3645+lambda_shift,BaJ_flux[-1]*0.8), #xytext=(0.3645+lambda_shift, BaJ_flux[-1]*0.2),
                        #arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5, headlength=6),
                        horizontalalignment='center'
                        )
            
            index_PaJ = [ind for ind,wave in enumerate(modinfo.waves[cspec].value) if wave > 0.8200 and wave < 0.8208]
            PaJ_flux = hi_ext_modsed[cspec][index_PaJ] * norm_data / norm_model
            ax.axvline(0.8204, color="grey")
            ax.annotate("Paschen\nJump", xy=(0.8204+lambda_shift*2,PaJ_flux[0]*0.85), #xytext=(0.8204+lambda_shift*2, PaJ_flux[0]*0.3),
                        #arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5, headlength=4),
                        horizontalalignment='center'
                        )
            
            index_Ha = [ind for ind,wave in enumerate(modinfo.waves[cspec].value) if wave > 0.6558 and wave < 0.6566]
            Ha_flux = hi_ext_modsed[cspec][index_Ha] * norm_data / norm_model
            ax.axvline(0.656281, color="grey")
            ax.annotate(r'H$\alpha$', xy=(0.656281,Ha_flux[0]*0.88), #xytext=(0.656281, Ha_flux[0]*0.5),
                        #arrowprops=dict(facecolor='black', arrowstyle="-"),
                        horizontalalignment='center', verticalalignment='center'
                        )
            
            index_Hb = [ind for ind,wave in enumerate(modinfo.waves[cspec].value) if wave > 0.4857 and wave < 0.4865]
            Hb_flux = hi_ext_modsed[cspec][index_Hb] * norm_data / norm_model
            ax.axvline(0.4861, color="grey")
            ax.annotate(r'H$\beta$', xy=(0.4861,Hb_flux[-1]*0.85), #xytext=(0.4861, Hb_flux[-1]*0.5),
                        #arrowprops=dict(facecolor='black', arrowstyle="-"),
                        horizontalalignment='center'
                        )
            
            index_Hg = [ind for ind,wave in enumerate(modinfo.waves[cspec].value) if wave > 0.4336 and wave < 0.4344]
            Hg_flux = hi_ext_modsed[cspec][index_Hg] * norm_data / norm_model
            ax.axvline(0.4340, color="grey")
            ax.annotate(r'H$\gamma$', xy=(0.4340,Hg_flux[-1]*0.85), #xytext=(0.4340, Hg_flux[-1]*0.5),
                        #arrowprops=dict(facecolor='black', arrowstyle="-"),
                        horizontalalignment='center'
                        )
    
    # finish configuring the plot
    ax.set_xlabel(r"$\lambda$ [$\mu m$]", fontsize=1.3 * fontsize)
    ax.set_ylabel("Percentage Residual", fontsize=1.3 * fontsize)
    ax.tick_params("both", length=10, width=2, which="major")
    ax.tick_params("both", length=5, width=1, which="minor")

def plot_star_sed(ax, fitmod, obsdata, modinfo, starname="", offset = 0):
    modsed = fitmod.stellar_sed(modinfo)

    ext_modsed = fitmod.dust_extinguished_sed(modinfo, modsed)

    hi_ext_modsed = fitmod.hi_abs_sed(modinfo, ext_modsed)
    
    for cspec in obsdata.data.keys():
        if cspec == "BAND":
            ptype = "o"
            rcolor = "g"
        else:
            ptype = "-"
            rcolor = "k"
        cwaves = modinfo.waves[cspec]

        # nan models where no data or excluded
        nvals = np.full(len(cwaves), 1.0)
        nvals[fitmod.weights[cspec] == 0.0] = np.nan
        multval = fitmod.norm.value * np.power(cwaves, 4.0) * nvals
#        ax.plot(cwaves, modsed[cspec] * multval, "b" + ptype)
        ax.plot(cwaves, ext_modsed[cspec] * multval, "g" + ptype)
        ax.plot(cwaves, hi_ext_modsed[cspec] * multval, "r" + ptype)

        gvals = obsdata.data[cspec].fluxes > 0.0
        ax.plot(
            obsdata.data[cspec].waves[gvals],
            obsdata.data[cspec].fluxes[gvals]
            * np.power(obsdata.data[cspec].waves[gvals], 4.0) + offset,
            "k" + ptype,
            label="data",
            alpha=0.7,
        )

    if starname != "":
        cspec = "STIS_Opt"
        x = 0.8
        val = obsdata.data[cspec].waves.value > x
        y = obsdata.data[cspec].fluxes[val].value * np.power(obsdata.data[cspec].waves[val].value, 4.0)
        ax.annotate(starname,
                    xy=(x, y[0]),
                    xytext=(x, y[0]*1.4),
                    fontsize=15,
                    #textcoords='offset points',
                    ha='center', va='center'
                    )


def plot_data_model(reddened_star_fitband, reddened_star, modinfo, params, fit_range="g23", velocity=0, plot_title=""):
    # plotting setup for easier to read plots
    fontsize = 18
    font = {"size": fontsize}
    mpl.rc("font", **font)
    mpl.rc("lines", linewidth=1)
    mpl.rc("axes", linewidth=2)
    mpl.rc("xtick.major", width=2)
    mpl.rc("xtick.minor", width=2)
    mpl.rc("ytick.major", width=2)
    mpl.rc("ytick.minor", width=2)

    # setup the plot
    fig, ax = plt.subplots(2, sharex=True, 
                        gridspec_kw={
            "wspace": 0.00,
            "hspace": 0.00,
        }, figsize=(13, 10), height_ratios=[2, 1])

    # create a StarData object for the best fit SED
    #modsed_stardata = modinfo.SED_to_StarData(modsed)

    # plot the bands and all spectra for this star
    #TODO: add an assert that band_data is in the star.data.keys()
    for i,cspec in enumerate(modinfo.fluxes.keys()):
        if cspec == "BAND":
            ptype = "o"
        else:
            ptype = "-"
        mask = get_mask(reddened_star.data[cspec].fluxes)
        reddened_star.data[cspec].fluxes[mask] = np.nan

        if i == 0:
            ax[0].plot(
                reddened_star.data[cspec].waves,
                reddened_star.data[cspec].fluxes,
                "k" + ptype,
                label="data",
            )
        else:
            ax[0].plot(
                reddened_star.data[cspec].waves,
                reddened_star.data[cspec].fluxes,
                "k" + ptype,
            )
    
    # intrinsic sed
    modsed = modinfo.stellar_sed(params[0:3], velocity=velocity)

    # dust_extinguished sed
    ext_modsed = modinfo.dust_extinguished_sed(params[3:10], modsed, fit_range=fit_range) #reddened_star, )

    # hi_abs sed
    hi_ext_modsed = modinfo.hi_abs_sed(
        params[10:12], [velocity, 0.0], ext_modsed
    )
    
    for j,cspec in enumerate(modinfo.fluxes.keys()):
        if cspec == "BAND":
            ptype = "o"
        else:
            ptype = "-"

        mask = get_mask(reddened_star_fitband.data[cspec].fluxes)
        reddened_star_fitband.data[cspec].fluxes[mask] = np.nan

        norm_model = np.average(hi_ext_modsed["BAND"])
        norm_data = np.average(reddened_star_fitband.data["BAND"].fluxes).value

        if j == 0:
            ax[0].plot(
                modinfo.waves[cspec], modsed[cspec] * norm_data / norm_model, "b" + ptype, label=cspec+", Intrinic Model SED w/o dust"
            )
            ax[0].plot(
                modinfo.waves[cspec],
                ext_modsed[cspec] * norm_data / norm_model,
                "r" + ptype,
                label=cspec,
            )
            ax[0].plot(
                modinfo.waves[cspec],
                hi_ext_modsed[cspec] * norm_data / norm_model,
                "g" + ptype,
                label=cspec+"final best fit model w/ dust",
            )
        else:
            ax[0].plot(
                modinfo.waves[cspec], modsed[cspec] * norm_data / norm_model, "b" + ptype
            )
            ax[0].plot(
                modinfo.waves[cspec],
                ext_modsed[cspec] * norm_data / norm_model,
                "r" + ptype
            )
            ax[0].plot(
                modinfo.waves[cspec],
                hi_ext_modsed[cspec] * norm_data / norm_model,
                "g" + ptype
            )

        if cspec == "STIS_Opt":
            #annotation_fontsize = 
            # Balmer Jump wavelen = 0.3645 micron
            # Paschen Jump wavelen = 0.8204 micron
            # H alpha wavelen = 0.656281 micron
            # H beta wavelen = 0.4861 micron
            # H gamma wavelen = 0.4340 micron
            lambda_shift = 0.0#05
            index_BaJ = [ind for ind,wave in enumerate(modinfo.waves[cspec].value) if wave > 0.3642 and wave < 0.3648]
            BaJ_flux = hi_ext_modsed[cspec][index_BaJ] * norm_data / norm_model
            ax[0].annotate("Balmer\nJump", xy=(0.3645+lambda_shift,BaJ_flux[-1]*0.8), xytext=(0.3645+lambda_shift, BaJ_flux[-1]*0.2),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5, headlength=6),
                        horizontalalignment='center'
                        )
            
            index_PaJ = [ind for ind,wave in enumerate(modinfo.waves[cspec].value) if wave > 0.8200 and wave < 0.8208]
            PaJ_flux = hi_ext_modsed[cspec][index_PaJ] * norm_data / norm_model
            ax[0].annotate("Paschen\nJump", xy=(0.8204+lambda_shift*2,PaJ_flux[0]*0.85), xytext=(0.8204+lambda_shift*2, PaJ_flux[0]*0.3),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5, headlength=4),
                        horizontalalignment='center'
                        )
            
            index_Ha = [ind for ind,wave in enumerate(modinfo.waves[cspec].value) if wave > 0.6558 and wave < 0.6566]
            Ha_flux = hi_ext_modsed[cspec][index_Ha] * norm_data / norm_model
            ax[0].annotate(r'H$\alpha$', xy=(0.656281,Ha_flux[0]*0.88), xytext=(0.656281, Ha_flux[0]*0.5),
                        arrowprops=dict(facecolor='black', arrowstyle="-"),
                        horizontalalignment='center', verticalalignment='center'
                        )
            
            index_Hb = [ind for ind,wave in enumerate(modinfo.waves[cspec].value) if wave > 0.4857 and wave < 0.4865]
            Hb_flux = hi_ext_modsed[cspec][index_Hb] * norm_data / norm_model
            ax[0].annotate(r'H$\beta$', xy=(0.4861,Hb_flux[-1]*0.85), xytext=(0.4861, Hb_flux[-1]*0.5),
                        arrowprops=dict(facecolor='black', arrowstyle="-"),
                        horizontalalignment='center'
                        )
            
            index_Hg = [ind for ind,wave in enumerate(modinfo.waves[cspec].value) if wave > 0.4336 and wave < 0.4344]
            Hg_flux = hi_ext_modsed[cspec][index_Hg] * norm_data / norm_model
            ax[0].annotate(r'H$\gamma$', xy=(0.4340,Hg_flux[-1]*0.85), xytext=(0.4340, Hg_flux[-1]*0.5),
                        arrowprops=dict(facecolor='black', arrowstyle="-"),
                        horizontalalignment='center'
                        )

    plot_residual(ax[1], reddened_star_fitband, modinfo, params, fit_range=fit_range, velocity=velocity, plot_title="Initial guess, Optimizer Fit")

    # finish configuring the plot
    a1 = np.array(hi_ext_modsed[cspec] * norm_data / norm_model)
    a1 = a1[~np.isnan(a1)]
    a2 = np.array(reddened_star.data[cspec].fluxes)
    a2 = a2[~np.isnan(a2)]
    b = np.array(modsed[cspec] * norm_data / norm_model)
    b = b[~np.isnan(b)]
    ymax = max(b)
    ymin = min(min(a1),min(a2))

    ax[0].set_ylim(0, ymax*1.1) #8e5 * norm_data / norm_model, 2e8 * norm_data / norm_model)
    ax[1].set_ylim(-10,10)
    #ax[0].set_xlim(0.29,1.4)
    ax[0].set_yscale("log")
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$\lambda$ [$\mu m$]", fontsize=1.3 * fontsize)
    ax[0].set_ylabel(r"$F(\lambda)$ [$ergs\ cm^{-2}\ s\ \AA$]", fontsize=1.3 * fontsize)
    ax[0].tick_params("both", length=10, width=2, which="major")
    ax[0].tick_params("both", length=5, width=1, which="minor")
    ax[0].set_title(plot_title)

    ax[0].legend()

    # use the whitespace better
    fig.tight_layout()


def plot_sed(data_set, show_plot=True, save_plot=False, save_prefix="optimizer"):
    print("Plotting SED of ", starname)

    dstarname = DATstarname(starname)
    fstarname = f"{dstarname}.dat"

    # TODO read in params and 
    if to_execute.upper().split()[-2].startswith("PLOT") or to_execute.upper().split()[-2].startswith("OPT"): 
        fit_type = "optimizer"
    elif to_execute.upper().split()[-2].startswith("MCMC"):
        fit_type = "mcmc"
    params = read_fit_params(param_filename=f"{starname}_fit_params_{fit_type}.dat",fpath =file_path+f"/DAT_files/STIS_Data/fitting_results/{data_set}/")

    #Read in the star data
    reddened_star = StarData(fstarname, path=f"{file_path}DAT_files/")

    #Following is needed to normalize model data with reddened star data,
    #check which photo band is used in the fitting in read_data:71
    fit_band = "J"
    reddened_star_J = StarData(fstarname, path=f"{file_path}DAT_files/", only_bands="J")
    band_names = reddened_star_J.data["BAND"].get_band_names()
    data_names = reddened_star_J.data.keys()
    modinfo = get_model_data(file_path, data_names, logTeff=params[0], logg=params[1],
                             band_names=band_names[0])

    plot_data_model(reddened_star_J, reddened_star, modinfo, params, fit_range=fit_range, velocity=velocity, plot_title="Initial guess, Optimizer Fit")
    #if show_plot: plt.show()
    if save_plot:
        save_prefix = f"{save_prefix}_{datetime.today().strftime('%Y-%m-%d')}"
        plt.savefig(f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/plots/{starname}_{list(data_names)[1]}_{save_prefix}.png")


def stack_sed(star_list,
              path,
              modpath,
              fit_params_path,
              fit_type,
              plot_range = "ALL",
              plot_bands = []
              ):

    # plotting setup for easier to read plots
    fontsize = 18
    font = {"size": fontsize}
    mpl.rc("font", **font)
    mpl.rc("lines", linewidth=1)
    mpl.rc("axes", linewidth=2)
    mpl.rc("xtick.major", width=2)
    mpl.rc("xtick.minor", width=2)
    mpl.rc("ytick.major", width=2)
    mpl.rc("ytick.minor", width=2)

    # setup the plot
    fig, ax = plt.subplots(
        figsize=(10, 8),
        sharex=True,
    )

    for star in star_list:
        fstarname = f"{star}.dat"
        best_fit_params = read_fit_params(param_filename=f"{star}_fit_params_{fit_type}_{plot_range.lower()}.dat", fpath=fit_params_path)
        if plot_range == "OPT":
            only_data = ["STIS_Opt"]
        else:
            only_data = plot_range
        reddened_star = StarData(fstarname, path=path, only_bands=plot_bands, only_data=only_data)
        modinfo = get_modeldata(reddened_star, modpath=modpath)
        fitmod = MEModel(modinfo=modinfo)
        fitmod.velocity.fixed = False
        fitmod.windamp.fixed = False
        fitmod.windalpha.fixed = False
        fitmod.fit_to_parameters(best_fit_params)
        fitmod.fit_weights(reddened_star)
        fitmod.set_initial_norm(reddened_star, modinfo)
#        fitmod.pprint_parameters()
        plot_star_sed(ax, fitmod, reddened_star, modinfo, starname=star, offset = 0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\lambda^4 F(\lambda)$ [RJ units]", fontsize=1.3 * fontsize)
    ax.tick_params("both", length=10, width=2, which="major")
    ax.tick_params("both", length=5, width=1, which="minor")

    fig.tight_layout()

    plt.show()
        



##################################
# EXTINCTION FIT AND PLOT BELOW  #
##################################

def plot_resid(ax, data, dindx, model, color, dataname):
    """
    Plot the residuals to the model
    """
    for cspec in dataname:
        if cspec == "BAND": 
            continue

        bvals = data.npts[cspec] <= 0
        data.exts[cspec][bvals] = np.nan

        fitx = 1.0 / data.waves[cspec].value

        ax.plot(
            data.waves[cspec],
            data.exts[cspec] - model[cspec](fitx),
            linestyle="dotted",
            color=color,
            alpha=0.75,
        )

        ax.fill_between(
            data.waves[cspec].value,
            data.exts[cspec] - model[cspec](fitx) - data.uncs[cspec],
            data.exts[cspec] - model[cspec](fitx) + data.uncs[cspec],
            color=color,
            alpha=0.25,
        )

def fit_ext_curve(model, data, wrange, dataname): #, no_weights=False):
    """
    Do the fits and plot the fits and residuals
    """
    warnings.filterwarnings("ignore")
    models = {}
    models["g22opt"] = (
            Polynomial1D(4)
            + Drude1D(amplitude=0.1, x_0=2.288, fwhm=0.243)
            + Drude1D(amplitude=0.1, x_0=2.054, fwhm=0.179)
            + Drude1D(amplitude=0.1, x_0=1.587, fwhm=0.243)
        )
    for i in range(1,4,1):
        models["g22opt"][i].x_0.fixed = True
        models["g22opt"][i].fwhm.fixed = True
        models["g22opt"][i].amplitude.bounds = (0.0, None)
    
    models["fm90"] = FM90()
    models["irpow"] = PowerLaw1D()
    #models["g21mod"] = G21mod()
    
    #else:
    #    raise ValueError("Please enter model, or give string \'g22opt\' as argument for fit_ext_curve().")
    
    npts = []
    waves = []
    intercepts = []
    intercepts_unc = []
    #slopes = []
    #slopes_unc = []
    cmodelfit = {}
    mask = {}
    model = list(model)
    if "BAND" in dataname: model.insert(0,"BAND")
    for i,cspec in enumerate(dataname):
        if cspec == "BAND":
            continue

        npts = data.npts[cspec]
        waves = data.waves[cspec]
        intercepts = data.exts[cspec]
        intercepts_unc = data.uncs[cspec]

        sindxs = np.argsort(waves)
        waves = waves[sindxs]
        npts = npts[sindxs]
        intercepts = intercepts[sindxs]
        intercepts_unc = intercepts_unc[sindxs]

        ## Exclude bad regions of the data:
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
        ] #/ u.micron

        for k, wave in enumerate(waves):
            for cexreg in ex_regions:
                if wave.value >= cexreg[0] and wave.value <= cexreg[1]:
                    npts[k] = 0

        gvals = (npts > 0) & (waves.value >= wrange[0]) & (waves.value <= wrange[1])

        fit = LevMarLSQFitter()
        or_fit = FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=5.0)

        fitx = 1.0 / waves[gvals].value
        try: 
            cmodelfit[cspec], mask[cspec] = or_fit(
            models[model[i]], fitx, intercepts[gvals], weights=1.0 / intercepts_unc[gvals]
            )
        except ValueError:
            print("Ooops! Make sure dataname and model have the same order.")

    return cmodelfit, mask


def ext_fit_plot(extdata, ax, star_name=None, offset=0):
    if "AV" not in extdata.columns.keys():
        extdata.calc_AV_JHK()
    if "RV" not in extdata.columns.keys():
        extdata.calc_RV()
    RV = extdata.columns["RV"]
    AV = extdata.columns["AV"]
    print(f"extdata type: {extdata.type}")
    print(f"RV = {RV[0]:.3f}+/-{RV[1]:.3f}, AV = {AV[0]:.3f}+/-{AV[1]:.3f}")

    plot_title=f"{starname}, Rv = {RV[0]:.3f}+/-{RV[1]:.3f}, AV = {AV[0]:.3f}+/-{AV[1]:.3f}"

    if extdata.type != "alav" or extdata.type != "alax":
        extdata.trans_elv_alav() #Transform E(lambda-V) to A(lambda)/A(V)

    plot_data_type = extdata.waves.keys()
    models = []
    xrange = [100,0.]
    for i,cspec in enumerate(plot_data_type):
        if cspec == "BAND":
            continue
        if cspec == "STIS" or cspec == "IUE":
            models.append("fm90")
            if xrange[0] > 0.1: xrange[0] = 0.1
            if xrange[1] < 0.3: xrange[1] = 0.3
        elif cspec == "STIS_Opt":
            models.append("g22opt")
            if xrange[0] > 0.3: xrange[0] = 0.3
            if xrange[1] < 1.0: xrange[1] = 1.0

#    n_data = len(plot_data_type) - 1 if "BAND" in extdata.waves.keys() else len(plot_data_type)
#    models = ["x"]*n_data
#    xrange = [0.,0.]
#    for i, data_type in enumerate(plot_data_type):
#        if data_type == "BAND":
#            continue
#        if "BAND" in plot_data_type: i -= 1
#        if data_type == "STIS" or data_type == "IUE":
#            models[i]="fm90"
#            xrange = [0.1, 0.3]
#        if data_type == "STIS_Opt":
#            models[i]="g22opt"
#            if xrange[0] == 0.0 or xrange[0] > 0.3: xrange[0] = 0.3
#            if xrange[1] < 1.0: xrange[1] = 1.0
#
#    models = []
#    xrange = [0.,0.]
#    if "STIS"  in plot_data_type or "IUE" in plot_data_type:
#        models.append("fm90")
#        xrange = [0.1, 0.3]
#    
#    if "STIS_Opt" in plot_data_type:
#        models.append("g22opt")
#        if xrange[0] == 0.0 or xrange[0] > 0.3: xrange[0] = 0.3
#        if xrange[1] < 1.0: xrange[1] = 1.0

    #TODO: add in clauses to include xrange for IR data
    #fm90 = FM90()

    cmodelfit, mask = fit_ext_curve(
            models,  # Make sure dataname and model have the same order!
            extdata,
            wrange=xrange,
            dataname=plot_data_type
            #no_weights=True, 
            )

    fit19_color = "dimgray"

    try:
        if len(ax) > 1:
            ax0 = ax[0]
            ax1 = ax[1]
            plot_resid(ax1, extdata, 2, cmodelfit, fit19_color, plot_data_type)
    except:
        ax0 = ax
        ax1 = None

    rejsym = "kx"

    for i,cspec in enumerate(plot_data_type):
        if cspec == "BAND":
            ptype = "o"
        else:
            ptype = "-"
        npts = extdata.npts[cspec]
        waves = extdata.waves[cspec]
        intercepts = extdata.exts[cspec]
        intercepts_unc = extdata.uncs[cspec]

        bvals = npts <= 0
        intercepts[bvals] = np.nan

        gvals = (npts > 0) & (waves.value >= xrange[0]) & (waves.value <= xrange[1])
        fitx = 1.0 / waves[gvals].value

        if cspec!="BAND": filtered_data = np.ma.masked_array(intercepts[gvals], mask=~mask[cspec])
        if i == 1 or cspec == "BAND":
            ax0.plot(waves, intercepts+offset, "k"+ptype, linewidth=1, label = "data")#, alpha=0.75)
            if cspec!="BAND":
                ax0.plot(waves[gvals], cmodelfit[cspec](fitx)+offset, "k"+ptype, alpha=0.5, label = "model")
                ax0.plot(waves[gvals], filtered_data+offset, rejsym, label="rejected")
        else:
            ax0.plot(waves, intercepts+offset, "k"+ptype, linewidth=1)#, alpha=0.75)
            if cspec!="BAND":
                ax0.plot(waves[gvals], cmodelfit[cspec](fitx)+offset, "k"+ptype, alpha=0.5)
                ax0.plot(waves[gvals], filtered_data+offset, rejsym)
    
        if cspec!="BAND" and ax1 != None:
            filtered_data2 = np.ma.masked_array(intercepts[gvals] - cmodelfit[cspec](fitx), mask=~mask[cspec])
            if i == 1: ax[1].plot(waves[gvals], filtered_data2, rejsym, label="rejected")
            else: ax[1].plot(waves[gvals], filtered_data2, rejsym)

        if cspec == "STIS" or cspec == "IUE":
            fitted_models = [cmodelfit[cspec]]
            # plotting the components
            modx = np.linspace(0.09, 0.33, 100) * u.micron
            tmodel = copy.deepcopy(fitted_models[0])
            tmodel.C3 = 0.0
            tmodel.C4 = 0.0
            ax0.plot(modx, tmodel(modx)+offset, "r--", alpha=0.5, label = "C2")

            tmodel = copy.deepcopy(fitted_models[0])
            tmodel.C3 = 0.0
            ax0.plot(modx, tmodel(modx)+offset, "r:", alpha=0.5, label = "C2,C4")

            tmodel = copy.deepcopy(fitted_models[0])
            tmodel.C4 = 0.0
            ax0.plot(modx, tmodel(modx)+offset, "r:", alpha=0.5, label = "C2,C3")

        elif cspec == "STIS_Opt":
            fitted_models = [cmodelfit[cspec]]
            modx = np.linspace(0.30, 1.0, 100)
            ax0.plot(modx, fitted_models[0][0](1.0 / modx)+offset, "b--", linewidth=2, alpha=0.5, label="4D Polynomial")
            for k in range(3):
                ax0.plot(
                    modx,
                    fitted_models[0][0](1.0 / modx) + fitted_models[0][k + 1](1.0 / modx)+offset,
                    "b:",
                    linewidth=2,
                    alpha=0.5, 
                    label = f"4D Poly. + $\lambda=${fitted_models[0][k + 1].x_0.value} Drude"
                )
            
            if star_name != None:
                x = 0.7
                val = waves.value > x
                fitx = 1.0 / waves[val].value
                y = cmodelfit[cspec](fitx)+offset
                print(star_name)
                ax0.annotate(f"{star_name} RV={RV[0]:0.3f}",
                            xy=(x, y[0]),
                            xytext=(x, y[0]*2),
                            fontsize=15,
                            #textcoords='offset points',
                            ha='left', va='center'
                            )
        elif cspec != "BAND":
            print("Oops! Write needed model fits for missing models here.")

    known_ISS = [0.437, 0.487, 0.637] # in G23 paper
    new_ISS = [0.54, 0.769] # in GAIA
    f19_dibs = [0.443, 0.487] # in F19 paper
    ca2_absorption = [0.3968, 0.3933]
    colors=["lightseagreen", "darkorchid", "royalblue", "mediumvioletred"]
    
    return plot_title, xrange


def plot_ext(extdata, plot_title=None, show_plot=False, save_plot=False):
    # Setup plot
    fontsize = 16
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=3)
    plt.rc("axes", linewidth=3)
    plt.rc("xtick.major", width=3, size=10)
    plt.rc("xtick.minor", width=2, size=5)
    plt.rc("ytick.major", width=3, size=10)
    plt.rc("ytick.minor", width=2, size=5)

    fig, ax = plt.subplots(2,
        figsize=(12, 9),
        sharex="col",
        gridspec_kw={
            "wspace": 0.,
            "hspace": 0.,
        },
        constrained_layout=True,
    )

    # Fit and plot the extinction curve read above
    print("Plotting Extinction of ", starname)
    plot_title2, xrange = ext_fit_plot(extdata, ax)
    if plot_title == None: plot_title = plot_title2

    leg_loc = "upper right"    
    ax[0].legend(ncol=2, loc=leg_loc, fontsize=0.8 * fontsize)

    ax[1].set_xscale("log")
    ax[0].set_yscale("linear")
    ax[1].set_yscale("linear")

    ax[1].set_xlabel(r"$\lambda$ [$\mu$m]")
    ax[0].set_ylabel(r"A($\lambda$)/A(V)")
    ax[1].set_ylabel("A($\lambda$)/A(V) Residual")

    #if plot_title == None: plot_title = f"{starname}"
    ax[0].set_title(plot_title)

    yrange_a_type = "linear"
    yrange_a = [0,2] #[-1.0, 1.5]
    yrange_b = [-0.1, 0.1]
    yrange_s = [0.0, 0.08]
    xticks = [0.3, 0.35, 0.45, 0.55, 0.7, 0.9, 1.0]

    ax[1].set_xlim(xrange[0], 2.0)#xrange[1])

    #ax[0].set_ylim(yrange_a)
    ax[1].set_ylim(yrange_b)

    for i in range(2):
        ax[i].axhline(linestyle="--", alpha=0.25, color="k", linewidth=2)

    if xrange == [0.30, 1.0]:
        plot_Hlines(ax[0], y_loc=0.1)
        plot_Hlines(ax[1])
        plot_ISSfeatures(ax[1])

    # Show or save extinction plot
    if save_plot: plt.savefig(f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/plots/{starname}_ext_fit.png")
    if show_plot: plt.show()


def plot_Hlines(ax, y_loc=None, color="grey", linestyle="--", linewidth=1):
    H_line_waves = [0.3645, 0.8204, 0.656281, 0.4861, 0.4340]
    H_line_names = ["Balmer\nJump", "Paschen\nJump", r'H$\alpha$', r'H$\beta$', r'H$\gamma$']

    ymax = 0 #np.max(np.array(intercepts[~np.isnan(intercepts)]))
    for Hline,wave in zip(H_line_names, H_line_waves):
        ax.axvline(wave, linestyle=linestyle, color=color, linewidth=linewidth)
        if not y_loc == None:
            ax.annotate(Hline, xy=(wave, y_loc), 
                           horizontalalignment='center'
                          )

def plot_ISSfeatures(ax, 
                     known_ISS = [0.437, 0.487, 0.637], # in G23 paper
                     new_ISS = [0.54, 0.769], # in GAIA
                     f19_dibs = [0.4428, 0.4882, 0.54503, 0.54875, 0.5535, 0.578045, 0.6283], # in F19 paper, Massa 2020
                     colors=["lightseagreen", "darkorchid", "royalblue", "mediumvioletred"]):

    ca2_absorption = [0.396847, 0.393366] # Massa 2020
    na1_absorption = [0.588995, 0.589692] # Massa 2020
    k1_absorption = [0.766491, 0.769897] # Massa 2020

    for iss in known_ISS: 
        #ax[1].axvline(iss, ymin=0, ymax=0.075, linestyle="-", color="blue", linewidth=1, label="known ISS")
        ax.annotate("known\nISS", xy=(iss, 0.), xytext=(iss, 0.075),
                       horizontalalignment='center', color=colors[0],
                       arrowprops=dict(facecolor=colors[0], arrowstyle="->", color=colors[0])
                      )

    for iss in new_ISS: 
        #ax[1].axvline(iss, linestyle="-", color=colors[1], linewidth=1, label = "new GAIA ISS")
        ax.annotate("new\nISS", xy=(iss, 0.), xytext=(iss, 0.075),
                       horizontalalignment='center', color=colors[1],
                       arrowprops=dict(facecolor=colors[1], arrowstyle="->", color=colors[1])
                      )

    for dibs in f19_dibs: 
        #ax[1].axvline(dibs, linestyle="-", color=colors[2], linewidth=1, label = "F19 dibs")
        ax.annotate("DIB", xy=(dibs, 0.0), xytext=(dibs, -0.04),
                       horizontalalignment='center', color=colors[2],
                       arrowprops=dict(facecolor=colors[2], arrowstyle="->", color=colors[2])
                      )

    # These are 2s1/2 <- 2p3/2,1/2 transitions; Ca II is K iso-electronic seq
    i=0
    for caii in ca2_absorption: 
        #ax[1].axvline(caii, linestyle="-", color=colors[3], linewidth=1, label = "Ca II")
        ax.annotate("Ca II", xy=(caii, 0.), xytext=(caii, 0.04+i),
                       horizontalalignment='center', color=colors[3],
                       arrowprops=dict(facecolor=colors[3], arrowstyle="->", color=colors[3])
                      )
        i+=0.02

    # These are 4s1/2 <- 4p3/2,1/2 transitions
    i=0
    for nai in na1_absorption: 
        #ax[1].axvline(caii, linestyle="-", color=colors[3], linewidth=1, label = "Ca II")
        ax.annotate("Na I", xy=(nai, 0.), xytext=(nai, 0.04+i),
                       horizontalalignment='center', color=colors[3],
                       arrowprops=dict(facecolor=colors[3], arrowstyle="->", color=colors[3])
                      )
        i+=0.02

    # These are 4s1/2 <- 4p3/2,1/2 transitions
    i=0
    for ki in k1_absorption: 
        #ax[1].axvline(caii, linestyle="-", color=colors[3], linewidth=1, label = "Ca II")
        ax.annotate("K I", xy=(ki, 0.), xytext=(ki, 0.04+i),
                       horizontalalignment='center', color=colors[3],
                       arrowprops=dict(facecolor=colors[3], arrowstyle="->", color=colors[3])
                      )
        i+=0.02


def plot_avg(data_set,dataname='STIS_Opt'):
    ext_waves_all = {}
    ext_resid_all = {}
    old_waves = np.array([0.0,0.0,0.0])
    color = 'dimgray'

    if dataname == "STIS_Opt": xrange = [0.30, 1.0]

    for star in lstarname:
        #extdata = QTable.read(f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/gun25_{star}_ext_optimizer.fits", hdu=2)
        file = f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/gun25_{star}_ext_optimizer.fits"
        fit19_extdata = ExtData(file)

        cmodelfit, mask = fit_ext_curve(
                "g22opt",
                fit19_extdata,
                wrange=xrange,
                dataname=dataname
                #no_weights=True, 
                )

        bvals = fit19_extdata.npts[dataname] <= 0
        fit19_extdata.exts[dataname][bvals] = np.nan

        fitx = 1.0 / fit19_extdata.waves[dataname].value

        ext_waves = fit19_extdata.waves[dataname]
        ext_residual = fit19_extdata.exts[dataname] - cmodelfit(fitx)

        ext_waves_all[star] = ext_waves
        ext_resid_all[star] = ext_residual

    df_waves = pd.DataFrame(data=ext_waves_all)
    df_resid = pd.DataFrame(data=ext_resid_all)

    row_avg = df_resid.aggregate('mean', axis=1)

    fontsize = 10
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=1.5)
    plt.rc("xtick.major", width=1.5, size=7)
    plt.rc("xtick.minor", width=1.5, size=5)
    plt.rc("ytick.major", width=1.5, size=7)
    plt.rc("ytick.minor", width=1.5, size=5)

    fig, ax = plt.subplots(1,
        figsize=(7.5, 4),
        #sharex="col",
        #gridspec_kw={
        #    "wspace": 0.01,
        #    "hspace": 0.01,
        #},
        constrained_layout=True,
        dpi=200
    )

    ax.plot(ext_waves, row_avg,
             linestyle="dotted",
             color=color,
             alpha=0.75)

    plot_Hlines(ax, color="goldenrod", y_loc=-0.09, linewidth=0.8)
    plot_ISSfeatures(ax)

    yrange_b = [-0.1, 0.1]
    ax.set_xscale("linear")
    #ax.set_xlim(xrange)
    ax.set_xlabel(r"$\lambda$ [$\mu$m]")

    ax.set_ylim(yrange_b)
    ax.set_ylabel(r"Average A($\lambda$)/A(V) Residual")

    ax.axhline(linestyle="--", alpha=0.25, color="k", linewidth=2)
    ax.vlines(0.514, ymin=-0.1, ymax=0, linestyle="-", color="green", alpha=0.5, linewidth = 1, label=r"$\lambda\approx0.514$")
    #ax = ax.axes
    #ax.set_xticks(list(ax.get_xticks()) + [0.514])

    ax.set_title(f"Averaged Residuals for {len(lstarname)} Stars")
    ax.legend(loc="upper right")

    #plt.show()

    fname = "stis_opt_ext_averaged_residuals"
    fig.savefig(f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/plots/{fname}.png")

    #ext_fit_plot(file_path+"DAT_files/STIS_Data/fitting_results/{data_set}/", starname+"_ext_optimizer"+".fits", show_plot=show_plot, save_plot=False)

def stack_ext(star_list):
    # Setup plot
    fontsize = 16
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=3)
    plt.rc("axes", linewidth=3)
    plt.rc("xtick.major", width=3, size=10)
    plt.rc("xtick.minor", width=2, size=5)
    plt.rc("ytick.major", width=3, size=10)
    plt.rc("ytick.minor", width=2, size=5)

    fig, ax = plt.subplots(1,
        figsize=(12, 9),
        constrained_layout=True
    )

    offset_list = []
    for i,star in enumerate(star_list):
        print(i)
        file = f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/plots/gun25_{star}_ext.fits"
        # Read the calculated extinction curve as an ExtData object 
        extdata = ExtData(file)
        ext_fit_plot(extdata, ax, star_name=star, offset=i/2)

    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel(r"A($\lambda$)/A(V) + offset")

    plt.show()

    



############################################
######       GLOBAL VARIABLES       ########
############################################

#TODO make into user command line input
data_set = "HighLowRv"
starname = "als882"

lstarname = ["hd46106",
             "hd18352",
             "hd13338",
             "bd+56d576",
             "hd14250",
             "cpd-57d3507",
             "hd14250",
             "hd92044"]

to_execute = "Stacked ext" # "Plot extinction", "Plot sed", "Plot average", "Stacked sed"

save_prefix = ""
file_path = "/Users/cgunasekera/extstar_data/"
velocity = 0
show_plot = True
save_plot = False

dstarname = DATstarname(starname)
fstarname = f"{dstarname}.dat"

not_tlusty = ["bd44d1080", "bd69d1231", "bd_71d92", "hd17443", "hd29647", "hd110336",
                  "hd112607",  "hd142165", "hd146285", "hd147196", "hd282485", "vss-viii10"]
    
if starname in not_tlusty:
    fit_range = "all"
else:
    fit_range = "g23"

###########################################
################  MAIN BODY ###############
###########################################

if __name__ == "__main__":

    if to_execute.upper().split()[0].startswith("PLOT"):
        if to_execute.upper().split()[-1].startswith("SED"):
            plot_sed(data_set=data_set, show_plot=show_plot, save_plot=save_plot)

        elif (to_execute.upper()).split()[-1].startswith("EXT"):
            fit_type = f"_optimizer"
            file = f"{file_path}DAT_files/STIS_Data/fitting_results/{data_set}/gun25_{starname}_ext{fit_type}.fits" #gun25_"+starname+mcmc_save_prefix+"_ext.fits"

            # Read the calculated extinction curve as an ExtData object 
            extdata = ExtData(file)

            plot_ext(extdata) #data_set=data_set, show_plot=show_plot, save_plot=save_plot)

        elif (to_execute.upper()).split()[-1].startswith("BOTH"):
            plot_sed(data_set=data_set, show_plot=show_plot, save_plot=save_plot)
            plot_ext(data_set=data_set, show_plot=show_plot, save_plot=save_plot)

        elif (to_execute.upper()).split()[-1].startswith("AVE"):
            plot_avg(data_set=data_set, show_plot=show_plot, save_plot=save_plot) # average of residuals

    elif to_execute.upper().split()[0].startswith("STACK"):
        star_list = ["hd200775", "als18098", "als15273"]#"hd303313", "hd306234", "als6028", "als6206", "als6213", "als15273"]
        path = "/Users/cgunasekera/extstar_data/DAT_files"

        if to_execute.upper().split()[-1].startswith("SED"):
            stack_sed(star_list,
                      path,
                      modpath="/Users/cgunasekera/extstar_data/Models",
                      fit_params_path = path+f"/STIS_Data/fitting_results/{data_set}/",
                      fit_type = "optimizer",
                      plot_range = "ALL",
                      plot_bands = []
                      )
        
        elif to_execute.upper().split()[-1].startswith("EXT"):
            stack_ext(star_list#, fit_type="opt"
            )