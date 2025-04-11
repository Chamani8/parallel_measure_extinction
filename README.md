# FitModel.py - Runs minimizer and mcmc to obtain best fit stellar parameters, and calculate the extinction curve.

FitModel.py uses measure_extinction package to fit the stellar and dust parameters of a star and saves the best fit parameters.
It then calculates the the extinction curve and saves it.

FitModel.py can be run with an editor like VSCode, or from command line.

## Getting started

First make sure you have the .dat file for your target, with the photometry and the locations to the observed data (i.e. files in extstar_data/DAT_files/).

Prepare a file that has the initial guess stellar + dust parameters to be used in the fitting. The script will read in any order of
parameters given, but the rows must have the input parameters for each star, and the columns are for each input parameters, for example:
#star	   	logTeff          	logg     logZ    ...     mcmc_nsteps
hd111934    4.327226974274963	3.98     0.0     ...     1000

The possible columns for this file are:
#star, logTeff, logTeff_bound, logg, logZ, vturb, velocity, windamp, windalpha, Av, Rv, C2, B3, C4, xo, gamma, vel_MW, logHI_MW, vel_exgal, logHI_exgal, norm, mcmc_nsteps

Make sure to giving the heading for in the file marked by a hashtag `#`, otherwise the script won't know what columns it is reading in.

## Running the fit

# Option 1: Editor method

Edit FitModel.py lines 548 - 552, to provide the script: the starname, path to stellar model files, path to save plotted figures, and the path
where your initial guess parameter file is. Then run the python script in your favorite editor.

# Option 2: Command line method

Run the following command from where the .dat file of the star is located (most likely from extstar_data/):
>> python3 FitModel.py -s <starname> -p </path/to/stellar_model/files/> -v </path/to/save/> -i </path/to/input/param/file/>

You can also run the script from a different location to the .dat files; simply specify where the .dat files are:
>> python3 FitModel.py -s <starname> -p </path/to/stellar_model/files/> -v </path/to/save/> -i </path/to/input/param/file/> -p </path/to/datfiles/>

## What the script does

It first fixes the dust parameters and runs an optimizer fit on just the optical data.
Then it fixes logTeff and logg, and runs an optimizer fit for the dust parameters on UV + optical data.
Then it uses this fit to calculate the extinction curve.

Currently this script explicitly deals with STIS_Opt, STIS, and IUE data. Further updates are needed to allow more flexibility.