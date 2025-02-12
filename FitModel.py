import os
import glob
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt

from measure_extinction.stardata import StarData
from measure_extinction.modeldata import ModelData
from measure_extinction.model import MEModel

def FitModel(starname,
         path, #Path to star data
         modtype="obsstars", #Pick the type of model grid; choices "obstars", "whitedwarfs"
         wind=False, #add IR (lambda > 1 um) wind emission model
         modpath="./", #path to the model files
         modstr=None, #Alternative model string for grid (expert)
         picmodel=True, #Set to read model grid from pickle file
         mcmc=False, #run EMCEE MCMC fitting
         mcmc_nsteps=1000, #number of MCMC steps
         savepath=None,
         showfit=False, #display the best fit model plot
         png=False, #save plots as png instead of pdf
         exregions=[]
        ):

    if savepath == None: savepath=path

    outname = f"{starname}_mefit"
    if png:
        outtype = "png"
    else:
        outtype = "pdf"

    # get data
    fstarname = f"{starname}.dat"
    print(f"Reading in {starname} data")
    reddened_star = StarData(fstarname, path=f"{path}")

    if "BAND" in reddened_star.data.keys():
        band_names = reddened_star.data["BAND"].get_band_names()
    else:
        band_names = []
    data_names = list(reddened_star.data.keys())

    #TODO: set bad values to nan so that it won't be used in the fitting or plotting
#    for cspec in data_names:
#        if cspec == "BAND":
#            continue
#        sudo_bval = reddened_star.data[cspec].fluxes.value < 5.4e-15
#        plt.plot(reddened_star.data[cspec].waves, reddened_star.data[cspec].fluxes, color="black")
#        reddened_star.data[cspec].fluxes[sudo_bval] = np.nan
#        plt.plot(reddened_star.data[cspec].waves, reddened_star.data[cspec].fluxes)
#    plt.yscale("log")
#    plt.xscale("log")
#    plt.show()

    # model data
    start_time = time.time()
    print("Reading model files")
    if modtype == "whitedwarfs":
        modstr = "wd_hubeny_"
    else:
        modstr = "tlusty_"
    if modstr is not None:
        modstr = modstr
    picmodel_path = glob.glob(f"{modstr}_modinfo.pkl")
    if picmodel and picmodel_path != []:
        modinfo = pickle.load(open(f"{modstr}_modinfo.pkl", "rb"))
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
        pickle.dump(modinfo, open(f"{modstr}_modinfo.pkl", "wb"))
    print("Finished reading model files")
    seconds = time.time() - start_time
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    hours = f"{int(hours)} hours, " if hours > 0 else ""
    minutes = f"{int(minutes)} minutes, " if minutes > 0 else ""
    print(f"Models pickled to {modstr}_modinfo.pkl, {hours}{minutes}{int(seconds)} seconds".strip())

    start_time = time.time()
    print("Start fitting")

    fitmod, result = FitModel_opt(reddened_star, modinfo, modtype, wind)

    seconds = time.time() - start_time
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    hours = f"{int(hours)} hours, " if hours > 0 else ""
    minutes = f"{int(minutes)} minutes, " if minutes > 0 else ""
    print(f"Finished fitting, took {hours}{minutes}{int(seconds)} seconds".strip())

    print(result["message"])
    if not result["message"].endswith("successfully."):
        # check the fit output
        return

    print("best parameters")
    fitmod.pprint_parameters()

    os.chdir(savepath)
    fitmod.plot(reddened_star, modinfo)
    plt.savefig(f"{savepath}{outname}_minimizer.{outtype}")
    print(f"Saved optimizer fit in: {savepath}{outname}_minimizer.{outtype}")
    plt.close()

    if mcmc:
        fitmod = FitModel_mcmc(fitmod, reddened_star, modinfo, mcmc_nsteps, outname, outtype)

    if showfit:
        fitmod.plot(reddened_star, modinfo)
        plt.show()

def FitModel_opt(reddened_star, modinfo, modtype, wind):
        # setup the model
    # memod = MEModel(modinfo=modinfo, obsdata=reddened_star)  # use to activate logf fitting
    print("Running MEModel")
    memod = MEModel(modinfo=modinfo)

    print("Fixing Teff, logg, Z, velocity and setting up weights")
    if "Teff" in reddened_star.model_params.keys():
        memod.logTeff.value = np.log10(float(reddened_star.model_params["Teff"]))
        memod.logTeff.fixed = True
    if "logg" in reddened_star.model_params.keys():
        memod.logg.value = float(reddened_star.model_params["logg"])
        memod.logg.fixed = True
    if "Z" in reddened_star.model_params.keys():
        memod.logZ.value = np.log10(float(reddened_star.model_params["Z"]))
        memod.logZ.fixed = True
    if "velocity" in reddened_star.model_params.keys():
        memod.velocity.value = float(reddened_star.model_params["velocity"])
        memod.velocity.fixed = True

    memod.fit_weights(reddened_star)

    if modtype == "whitedwarfs":
        memod.vturb.value = 0.0
        memod.vturb.fixed = True
        memod.Av.value = 0.5
        memod.weights["BAND"] *= 10.0
        memod.weights["STIS"] *= 10.0
    if wind:
        memod.windamp.value = 1e-3
        memod.windamp.fixed = False
        memod.windalpha.fixed = False

    print("Normalizing the model data to match observed data")
    memod.set_initial_norm(reddened_star, modinfo)

    print("Initial parameters")
    memod.pprint_parameters()

    fitmod, result = memod.fit_minimizer(reddened_star, modinfo, maxiter=10000)

    return fitmod, result

def FitModel_mcmc(fitmod,
                  reddened_star,
                  modinfo,
                  mcmc_nsteps,
                  outname,
                  outtype
                  ):
        print("Starting MCMC sampling")
        # using an MCMC sampler to define nD probability function
        # use best fit result as the starting point
        fitmod2, flat_samples, sampler = fitmod.fit_sampler(
            reddened_star,
            modinfo,
            nsteps=mcmc_nsteps)

        print("Finished MCMC sampling")

        print("p50 parameters")
        fitmod2.pprint_parameters()

        fitmod2.plot(reddened_star, modinfo)
        plt.savefig(f"{outname}_mcmc.{outtype}")
        print(f"Saved mcmc fit in: {outname}_minimizer.{outtype}")
        plt.close()

        fitmod2.plot_sampler_chains(sampler)
        plt.savefig(f"{outname}_mcmc_chains.{outtype}")
        print(f"Saved mcmc sampler chians in: {outname}_minimizer.{outtype}")
        plt.close()

        fitmod2.plot_sampler_corner(flat_samples)
        plt.savefig(f"{outname}_mcmc_corner.{outtype}")
        print(f"Saved mcmc sampler corner plots in: {outname}_minimizer.{outtype}")
        plt.close()

        return fitmod2


if __name__ == "__main__":
    starname = "csi-27-07550"
    path = "/Users/cgunasekera/extstar_data/DAT_files"
    FitModel(starname.lower(), path, modpath="/Users/cgunasekera/extstar_data/Models",
             showfit=True, savepath="/Users/cgunasekera/extstar_data/DAT_files/STIS_Data/fitting_results/HighLowRv/plots")