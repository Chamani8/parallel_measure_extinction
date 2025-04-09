import os
import glob
import pickle
import time
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from measure_extinction.stardata import StarData
from measure_extinction.modeldata import ModelData
from measure_extinction.model import MEModel
from measure_extinction.extdata import ExtData

from setup_ext_data import get_stars_list

from plot_extinction_curves import plot_ext

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
    start_time = time.time()
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
        seconds = time.time() - start_time
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        hours = f"{int(hours)} hours, " if hours > 0 else ""
        minutes = f"{int(minutes)} minutes, " if minutes > 0 else ""
        print(f"Models pickled to {modstr}{starname}_modinfo.pkl, {hours}{minutes}{int(seconds)} seconds".strip())

    return modinfo

def Save_params(fitmod, starname, save_path, fit_type):
    f = open(f"{save_path}/{starname}_fit_params_{fit_type}.dat", "w")
    f.write(f"# {starname} best fit params\n")

    pnames = [
        "logTeff", "logg", "logZ", "vturb", "velocity", "windamp", "windalpha",
        "Av", "Rv", "C2", "B3", "C4", "xo", "gamma",
        "vel_MW", "logHI_MW", "vel_exgal", "logHI_exgal"
    ]

    for cname in fitmod.paramnames:
        param = getattr(fitmod, cname)
        if (cname != "norm"):
            if param.unc is not None:
                ptxt = rf"${param.value:f} \pm {param.unc:f}$ #{cname}\n"
            else:
                ptxt = f"{param.value:f} #{cname}\n"
            
            f.write(ptxt)
    f.close()

def FitModel(starname,
         reddened_star,
         modinfo,
         path, #Path to star data
         modtype="obsstars", #Pick the type of model grid; choices "obstars", "whitedwarfs"
         wind=False, #add IR (lambda > 1 um) wind emission model
         mcmc=False, #run EMCEE MCMC fitting
         mcmc_nsteps=1000, #number of MCMC steps
         savepath=None,
         showfit=False, #display the best fit model plot
         png=False, #save plots as png instead of pdf
         print_process=False, #for debugging purposes
         fitmod_Opt=None,
         inparam_file=None
        ):

    if savepath == None:
        param_savepath = None
        savepath=path
    else:
        #to edit later; this specifically only works for 'path/to/save/.../plots/'
        param_savepath = savepath[:-5]

    outname = f"{starname}_mefit"
    if png:
        outtype = "png"
    else:
        outtype = "pdf"

    data_names = list(reddened_star.data.keys())

    #Set bad values to nan so that it won't be used in the fitting or plotting
    for cspec in data_names:
        if cspec == "BAND":
            continue
        elif cspec == "STIS":
            allowed_dev = 0.27
        elif cspec == "STIS_Opt" or cspec == "IUE":
            allowed_dev = 0.3

#        plt.scatter(reddened_star.data[cspec].waves, reddened_star.data[cspec].fluxes, color="black")
        for i,val in enumerate(reddened_star.data[cspec].fluxes.value):
            if i+1 >= len(reddened_star.data[cspec].fluxes.value):
                continue
            if abs(val - reddened_star.data[cspec].fluxes[i-1].value)/val > allowed_dev and abs(val - reddened_star.data[cspec].fluxes[i+1].value)/val > allowed_dev:
                reddened_star.data[cspec].fluxes.value[i] = np.nan

            if starname=="bd+56-510":
                if cspec == "STIS_Opt" and reddened_star.data[cspec].waves[i].value<0.292:
                    reddened_star.data[cspec].fluxes.value[i] = np.nan
                if cspec == "IUE" and reddened_star.data[cspec].waves[i].value>0.3153:
                    reddened_star.data[cspec].fluxes.value[i] = np.nan
                if reddened_star.data[cspec].waves[i].value>0.2186 and reddened_star.data[cspec].waves[i].value<0.2197:
                    reddened_star.data[cspec].fluxes.value[i] = np.nan
            if starname=="als18098":
                if reddened_star.data[cspec].waves[i].value>0.18092 and reddened_star.data[cspec].waves[i].value<0.18164:
                    reddened_star.data[cspec].fluxes.value[i] = np.nan
                if reddened_star.data[cspec].waves[i].value>0.20063 and reddened_star.data[cspec].waves[i].value<0.201981:
                    reddened_star.data[cspec].fluxes.value[i] = np.nan
                if cspec == "STIS_Opt" and reddened_star.data[cspec].waves[i].value<0.293912:
                    reddened_star.data[cspec].fluxes.value[i] = np.nan
            if starname=="hd036982":
                if cspec == "STIS_Opt" and (reddened_star.data[cspec].waves[i].value<0.293516):# or reddened_star.data[cspec].waves[i].value>1.01934):
                    reddened_star.data[cspec].fluxes.value[i] = np.nan
                if cspec == "IUE" and reddened_star.data[cspec].waves[i].value>0.324:
                    reddened_star.data[cspec].fluxes.value[i] = np.nan
            if starname=="hd37021":
                if cspec == "STIS_Opt" and (reddened_star.data[cspec].waves[i].value<0.295614):# or reddened_star.data[cspec].waves[i].value>1.01934):
                    reddened_star.data[cspec].fluxes.value[i] = np.nan
                if cspec == "IUE" and reddened_star.data[cspec].waves[i].value>0.316:
                    reddened_star.data[cspec].fluxes.value[i] = np.nan
            if starname == "hd038087":
                if cspec == "IUE" and reddened_star.data[cspec].waves[i].value>0.315:
                    reddened_star.data[cspec].fluxes.value[i] = np.nan
            if starname == "hd200775":
                if cspec == "STIS_Opt" and reddened_star.data[cspec].waves[i].value>0.65176 and reddened_star.data[cspec].waves[i].value<0.66173:# or reddened_star.data[cspec].waves[i].value>1.01934):
                    reddened_star.data[cspec].fluxes.value[i] = np.nan
                if cspec == "STIS_Opt" and reddened_star.data[cspec].waves[i].value>0.48509 and reddened_star.data[cspec].waves[i].value<0.4872:# or reddened_star.data[cspec].waves[i].value>1.01934):
                    reddened_star.data[cspec].fluxes.value[i] = np.nan
                if cspec == "STIS_Opt" and reddened_star.data[cspec].waves[i].value>0.84118 and reddened_star.data[cspec].waves[i].value<0.84823:# or reddened_star.data[cspec].waves[i].value>1.01934):
                    reddened_star.data[cspec].fluxes.value[i] = np.nan
            if starname == "walker67":
                if cspec == "IUE" and (reddened_star.data[cspec].waves[i].value>0.3305 or reddened_star.data[cspec].waves[i].value<0.1235):
                    reddened_star.data[cspec].fluxes.value[i] = np.nan
                if cspec == "STIS_Opt" and reddened_star.data[cspec].waves[i].value<0.2941:# or reddened_star.data[cspec].waves[i].value>1.01934):
                    reddened_star.data[cspec].fluxes.value[i] = np.nan

    if inparam_file!=None:
        inparams=read_inparams(starname, inparam_file)
        if print_process: print("Read initial parameters from file")
    else:
        inparams=[]

    start_time = time.time()

    fitmod, result = FitModel_opt(reddened_star, modinfo, modtype, wind, print_process, inparams=inparams, fitmod_Opt=fitmod_Opt)
    seconds = time.time() - start_time
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    hours = f"{int(hours)} hours, " if hours > 0 else ""
    minutes = f"{int(minutes)} minutes, " if minutes > 0 else ""
    if print_process: print(f"Time: {hours}{minutes}{int(seconds)} seconds".strip())

    print(result["message"])
    if not result["message"].endswith("successfully."):
        # check the fit output
        return

    if print_process:
        print("best parameters")
        fitmod.pprint_parameters()

    ## lnprob = make positive then average
    ## do grid for 1x10^-{np.arange([number of decimals])} = 0.1, 0.01, 0.001, ...
    ## each time it runs it should improve the fit
    os.chdir(savepath)
    fitmod.plot(reddened_star, modinfo)
    plt.savefig(f"{savepath}/{outname}_minimizer.{outtype}")
    print(f"Saved optimizer plot fit in: {savepath}/{outname}_minimizer.{outtype}")
    plt.close()

    if "IUE" in data_names or "STIS" in data_names:
        fit_type = "all"
    else:
        fit_type = "opt"

    if mcmc:
        mcmc_nsteps = int(inparams["mcmc_nsteps"])
        fitmod = FitModel_mcmc(fitmod, reddened_star, modinfo, mcmc_nsteps, outname, outtype, savepath, print_process)
        Save_params(fitmod, starname, param_savepath, fit_type=f"mcmc_{fit_type}")
    elif mcmc!=None and param_savepath != None:
        Save_params(fitmod, starname, param_savepath, fit_type=f"optimizer_{fit_type}")

    if showfit:
        fitmod.plot(reddened_star, modinfo)
        plt.show()

    return fitmod

def FitModel_opt(reddened_star, modinfo, modtype, wind, print_process, inparams=[], fitmod_Opt=None):
    # setup the model
    # memod = MEModel(modinfo=modinfo, obsdata=reddened_star)  # use to activate logf fitting

    if fitmod_Opt != None:
        fitmod_Opt_logTeff = fitmod_Opt.logTeff.value
        fitmod_Opt_logg = fitmod_Opt.logg.value
        fitmod_Opt_vturb = fitmod_Opt.vturb.value
        fitmod_Opt_velocity = fitmod_Opt.velocity.value
        fitmod_Opt_Av = fitmod_Opt.Av.value
        fitmod_Opt_Rv = fitmod_Opt.Rv.value

    if print_process: print("Running MEModel")
    memod = MEModel(modinfo=modinfo)

    if print_process: print("Fixing Teff, logg, Z, velocity and setting up weights")
    if "Teff" in reddened_star.model_params.keys():
        memod.logTeff.value = np.log10(float(reddened_star.model_params["Teff"]))
        memod.logTeff.fixed = True
    if "logg" in reddened_star.model_params.keys():
        memod.logg.value = float(reddened_star.model_params["logg"])
    if "Z" in reddened_star.model_params.keys():
        memod.logZ.value = np.log10(float(reddened_star.model_params["Z"]))
        memod.logZ.fixed = True
    if "velocity" in reddened_star.model_params.keys():
        memod.velocity.value = float(reddened_star.model_params["velocity"])
        memod.velocity.fixed = True

#    if starname=="hd200775":
#        memod.add_exclude_region([1/0.667082, 1/0.647051]/u.micron)
    if starname=="hd164906":
        memod.add_exclude_region([1/0.661696, 1/0.652988]/u.micron)
    if starname=="hd294264":
        memod.add_exclude_region([1/0.298, 1/0.290515]/u.micron)
        memod.add_exclude_region([1/1.024, 1/0.9915]/u.micron)
    if starname=="als18098":
        memod.add_exclude_region([1/0.2975, 1/0.28945]/u.micron)
        memod.add_exclude_region([1/0.36, 1/0.35785]/u.micron)
    if starname=="gsc04023-00972":
        memod.add_exclude_region([1/0.124, 1/0.114458]/u.micron)

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

    if print_process: print("Normalizing the model data to match observed data")
    memod.set_initial_norm(reddened_star, modinfo)

    # stellar
    if "logTeff" in inparams:
        memod.logTeff.value = inparams["logTeff"]
    if "logTeff_bound" in inparams and inparams["logTeff_bound"]==True:
        memod.logTeff.bounds = (memod.logTeff.bounds[0], inparams["logTeff"]+0.05)
    if "logg" in inparams:
        memod.logg.value = inparams["logg"]
        if starname == "als882": memod.logg.bounds = (memod.logg.bounds[0], 4.2)
        if starname == "hd303313": memod.logg.bounds = (memod.logg.bounds[0], 4.00)
    if "logZ" in inparams:
        memod.logZ.value = inparams["logZ"]
    if "vturb" in inparams:
        memod.vturb.value = inparams["vturb"]
#        if starname == "hd204827": memod.vturb.bounds = (memod.vturb.bounds[0], 7.47)
    if "velocity" in inparams:
        memod.velocity.value = inparams["velocity"]
        #memod.velocity.bounds = (-1.0, 1.0)
    if "windamp" in inparams:
        memod.windamp.value = inparams["windamp"]
    if "windalpha" in inparams:
        memod.windalpha.value = inparams["windalpha"]
    if starname == "als6028":
        memod.velocity.bounds = (-20, 20)
    else:   
        memod.velocity.bounds = (-250, 250)

    # dust - values, bounds, and priors based on VCG04 and FM07 MW samples (expect Av)
    if "Av" in inparams:
        memod.Av.value = inparams["Av"]
#    if starname == "als18098": memod.Av.bounds = (2.0, memod.Av.bounds[1])
    if "Rv" in inparams:
        memod.Rv.value = inparams["Rv"]
    if "C2" in inparams: 
        memod.C2.value = inparams["C2"]
    if "B3" in inparams:
        memod.B3.value = inparams["B3"]
        if starname == "hd210121": memod.B3.bounds = (0.0, inparams["B3"]+0.5)
        if starname == "hd37021": memod.B3.bounds = (memod.B3.bounds[0], 1.6)
    if "C4" in inparams:
        memod.C4.value = inparams["C4"]
    if "xo" in inparams:
        memod.xo.value = inparams["xo"]
    if "gamma" in inparams:
        memod.gamma.value = inparams["gamma"] #UV

    # gas
    if "vel_MW" in inparams:
        memod.vel_MW.value = inparams["vel_MW"]
    if "logHI_MW" in inparams:
        memod.logHI_MW.value = inparams["logHI_MW"]
        if starname == "hd210121": memod.logHI_MW.bounds = ( memod.logHI_MW.bounds[0], inparams["logHI_MW"]+0.5)
        memod.logHI_MW.bounds = (20.0, memod.logHI_MW.bounds[1])
    if "vel_exgal" in inparams:
        memod.vel_exgal.value = inparams["vel_exgal"]
    if "logHI_exgal" in inparams:
        memod.logHI_exgal.value = inparams["logHI_exgal"]

    if fitmod_Opt != None:
        memod.logTeff.value  = fitmod_Opt_logTeff
        memod.logg.value = fitmod_Opt_logg
        memod.vturb.value = fitmod_Opt_vturb
        memod.velocity.value = fitmod_Opt_velocity
        memod.Av.value = fitmod_Opt_Av
        memod.Rv.value = fitmod_Opt_Rv
    memod.logZ.fixed = True
    memod.vturb.fixed = False
    memod.velocity.fixed = False
    #memod.Av.fixed = True
    #memod.Rv.fixed = False


    if "STIS" in reddened_star.data.keys() or "IUE" in reddened_star.data.keys():
        memod.logTeff.fixed = True
        memod.logg.fixed = True
        if starname == "hd192660":
            memod.Av.prior = (memod.Av.value, 1e-5)
        if starname == "hd210121": 
            memod.Av.prior = (memod.Av.value, 1e-5)
            memod.xo.prior = (memod.xo.value, 1e-4)
            memod.vturb.fixed = (memod.vturb.value, 1e-5)
        if starname == "hd37021": memod.xo.prior = (memod.xo.value, 1e-4)
        memod.C2.fixed = False
        memod.B3.fixed = False
        memod.C4.fixed = False
        memod.xo.fixed = False
        memod.gamma.fixed = False
        memod.vel_MW.fixed = False
        memod.logHI_MW.fixed = False
    else:
        memod.logTeff.fixed = False
        memod.logg.fixed = False
        memod.C2.fixed = True
        memod.B3.fixed = True
        memod.C4.fixed = True
        memod.xo.fixed = True
        memod.gamma.fixed = True
        memod.vel_MW.fixed = True
        memod.logHI_MW.fixed = True

    print("Initial fit params: \n",
          memod.logTeff.value,
          memod.logg.value,
          memod.logZ.value,
          memod.vturb.value,
          memod.velocity.fixed,
          memod.windamp.fixed,
          memod.windalpha.fixed,
          memod.Av.value,
          memod.Rv.value,
          memod.C2.value,
          memod.B3.value,
          memod.C4.value,
          memod.xo.value,
          memod.gamma.value,
          memod.vel_MW.value,
          memod.logHI_MW.value,
          memod.vel_exgal.value,
          memod.logHI_exgal.value
          )

    if print_process:
        print("Fitting initial parameters")
        memod.pprint_parameters()

    fitmod, result = memod.fit_minimizer(reddened_star, modinfo, maxiter=10000)

    fitmod.logTeff.fixed = False
    fitmod.logg.fixed = False

    return fitmod, result

def FitModel_mcmc(fitmod,
                  reddened_star,
                  modinfo,
                  mcmc_nsteps,
                  outname,
                  outtype,
                  savepath,
                  print_process
                  ):
        if print_process: print("Starting MCMC sampling")
        # using an MCMC sampler to define nD probability function
        # use best fit result as the starting point
        fitmod2, flat_samples, sampler = fitmod.fit_sampler(
            reddened_star,
            modinfo,
            nsteps=mcmc_nsteps,
            burnfrac=0.2)

        if print_process:
            print("Finished MCMC sampling")
            print("p50 parameters")
            fitmod2.pprint_parameters()

        fitmod2.plot(reddened_star, modinfo)
        plt.savefig(f"{savepath}/{outname}_mcmc.{outtype}")
        if print_process: print(f"Saved mcmc fit in: {savepath}/{outname}_minimizer.{outtype}")
        plt.close()

        fitmod2.plot_sampler_chains(sampler)
        plt.savefig(f"{savepath}/{outname}_mcmc_chains.{outtype}")
        if print_process: print(f"Saved mcmc sampler chians in: {savepath}/{outname}_minimizer.{outtype}")
        plt.close()

        fitmod2.plot_sampler_corner(flat_samples)
        plt.savefig(f"{savepath}/{outname}_mcmc_corner.{outtype}")
        if print_process: print(f"Saved mcmc sampler corner plots in: {savepath}/{outname}_minimizer.{outtype}")
        plt.close()

        return fitmod2


def read_inparams(starname, inparam_file):
    inparam_file = glob.glob(inparam_file)
    if inparam_file == []:
        print("Oops, couldn't find inparams file, using default.")
        return []

    with open(inparam_file[0], "r") as f:
        line = f.readlines()

    header = line[0].split()
    if "#star" in header: header.remove("#star")

    i = 0
    try:
        while not starname in line[i].split():
            i +=1
    except IndexError:
        print("Intial params not in file. Using default set.")
        return 0

    initial_params={}
    for paramname, param in zip(header, line[i].split()[1:]):
        if param == "True": initial_params[paramname]= True
        elif param == "False": initial_params[paramname]= False
        else:
            initial_params[paramname]= float(param)

    if len(initial_params) == 1:
        print("Incorrect value found for initial params. Using default set.")
        return []
    else:
        return initial_params


def create_ext_curve(starname,
        path,
        modpath,
        savepath,
        fitmod,
        modtype="obsstars",
        relband="V"): # 0.55*u.micron):
    #TODO: use data types that match what is in fitmod
    reddened_star = StarData(f"{starname}.dat", path=f"{path}")#, only_data=["IUE", "STIS_Opt"])

    data_names = list(reddened_star.data.keys())
    band_names = reddened_star.data["BAND"].get_band_names()
    bandnames=""
    for band in band_names: bandnames += band
    datanames=""
    for data in data_names:
        if data=="BAND":
            continue
        else:
            datanames += data
    modinfo = get_modeldata(reddened_star, modpath=modpath, modtype=modtype)

    modsed = fitmod.stellar_sed(modinfo)
    modsed_stardata = modinfo.SED_to_StarData(modsed)
    extdata = ExtData()
    extdata.calc_elx(reddened_star, modsed_stardata, rel_band=relband)

    date = datetime.today().strftime('%d%b%Y')
    save_file = f"{savepath}/gun25_{starname}_ext.fits"

    if "AV" not in extdata.columns.keys():
        extdata.calc_AV_JHK()
    if "RV" not in extdata.columns.keys():
        extdata.calc_RV()

    RV = extdata.columns["RV"]
    AV = extdata.columns["AV"]
    col_info = {"av": AV[0], "rv": RV[0]}
    extdata.save(save_file, column_info=col_info)
    print(f"File written to: {save_file}")

    plot_ext(extdata, plot_title=f"{starname}, Rv = {RV[0]}+/-{RV[1]}")
#    print(f"RV = {RV[0]}+/-{RV[1]}, AV = {AV[0]}+/-{AV[1]}")

    plt.savefig(f"{savepath}/{starname}_ext_fit.png")
#    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A simple command-line argument parser.")
    parser.add_argument('-s', '--starname', type=str, help="Target name", default=None)
    parser.add_argument('-p', '--datpath', type=str, help=".dat file location", default=".")
    parser.add_argument('-m', '--modpath', type=str, help=".dat file location", default="../Models/")
    parser.add_argument('-v', '--savepath', type=str, help=".dat file location", default=".")
    parser.add_argument('-i', '--inparam', type=str, help=".dat file location", default=None)

    args = parser.parse_args()

    if args.starname == None:
        starname = "walker67"
        path = "/Users/cgunasekera/extstar_data/DAT_files"
        modpath = "/Users/cgunasekera/extstar_data/Models"
        savepath = "/Users/cgunasekera/extstar_data/DAT_files/STIS_Data/fitting_results/HighLowRv/plots"
        inparam_file="/Users/cgunasekera/extstar_data/DAT_files/STIS_Data/fitting_results/HighLowRv/HighLowRv_inparams.dat"
    else:
        starname = args.starname
        path = args.datpath
        modpath = args.modpath
        savepath = args.savepath
        inparam_file = args.inparam

    fstarname = f"{starname}.dat"
    print(f"Fitting & measuring extinction of {starname}")
    reddened_star = StarData(fstarname, path=f"{path}", only_bands=[], only_data=["STIS_Opt"])
    modinfo = get_modeldata(reddened_star, modpath=modpath)

    fitmod_Opt = FitModel(starname, #args.starname.lower(),
             reddened_star,
             modinfo,
             path,
             showfit=True, 
             savepath=savepath,
             inparam_file=inparam_file,
#             mcmc=True,
             )
#    exit(0)
    fstarname = f"{starname}.dat"
    reddened_star = StarData(fstarname, path=f"{path}", only_bands=[], only_data="ALL")
    modinfo = get_modeldata(reddened_star, modpath=modpath)

    #fitmod = fitmod_Opt
    fitmod = FitModel(starname,
             reddened_star,
             modinfo,
             path,
             showfit=True, 
             savepath=savepath,
             inparam_file=inparam_file,
             fitmod_Opt=fitmod_Opt,
#             mcmc=True,
             )
#    exit(0)
    create_ext_curve(starname,
                     path,
                     modpath=modpath,
                     savepath=savepath,
                     fitmod=fitmod
                     )