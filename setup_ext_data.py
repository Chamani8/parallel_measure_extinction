import glob
import os
import subprocess

from astropy.io import fits
import matplotlib.pyplot as plt

from read_data import DATstarname

def AddDatSTISpath(starname, waveregion, outpath, sample_summary):
    dirs = outpath.split('/')
    del dirs[-2]
    datpath = '/'.join(dirs)
    dstarname = DATstarname(starname)
    fstarname = f"{dstarname}.dat"

    fits_file_path = glob.glob(outpath+"{}*{}.fits".format(starname,waveregion))
    fits_filename = fits_file_path[0].split('/')[-1]
    get_band_name = fits_filename.replace(starname+'_', '')
    band_name = get_band_name.replace('.fits','')
    band_name = band_name.split('_')
    band_name[0] = band_name[0].upper()
    band_name = '_'.join(band_name)

    star = ""
    if sample_summary == "F19":
        f19_path = outpath.split('/')[:-3]
        f19_path = '/'.join(f19_path)+'/'
        pstarname = get_f19starname(starname)
    
        with open(f"{file_path}fitzpatrick_2019_stars.dat", "r") as f:
            while not star.startswith(pstarname):
                star = f.readline()
                if star.startswith("# source"):
                    print("Oops! Did not find matching star,"+pstarname+", in F19 file!")
                    return 

        if star.split()[-5].startswith("B") or star.split()[-5].startswith("O"):
            sptype = "sptype = " + "".join(star.split()[-5:-3]) + " ; ref = 2019ApJ...886..108F\n"
        elif star.split()[-4].startswith("B") or star.split()[-4].startswith("O"):
            sptype = "sptype = " + star.split()[-4] + " ; ref = 2019ApJ...886..108F\n"

    try:
        with open(datpath+fstarname) as datfile:
            f=datfile.readlines()
        
        index = [idx for idx, s in enumerate(f) if 'sptype' in s]
        if index == []:
            sptype = ""
        if sample_summary == "F19" and "···" in star.split()[-5:-3]:
            sptype = f[index[0]]
                
    except FileNotFoundError:
        print("Oops .dat file for {} not found. \nCreating new {}.dat file.".format(dstarname, dstarname))

        sptype_placeholder = "sptype = \n"
        if sample_summary == "F19" and "···" in star.split()[-5:-3]:
            print("Oops! Please manually enter star spectral type from SIMBAD.")
            sptype = sptype_placeholder

        data = []
        data.append("# data file for observations of "+dstarname.upper()+"\n")
        data.append(sptype_placeholder)
        
        datfile = open(datpath+fstarname, "a")
        datfile.writelines(data)

        print("Please fill in the photometric data from SIMBAD.")

        with open(datpath+fstarname) as datfile:
            f=datfile.readlines()

    if not any(band_name in s for s in f):
        try:
            index = [idx for idx, s in enumerate(f) if 'fits' in s][0]
        except IndexError:
            index = -1
        if sample_summary == "F19":
            stis_path = 'STIS_Data/'+f'{starname}_stis_Opt.fits'
        else:
            stis_path = "STIS_Data/"+fits_filename
        f.insert(index, band_name+' = '+stis_path+'\n')

    for i,line in enumerate(f):
        line = line.split(' ')
        if sample_summary == "F19" and line[0] == 'sptype':
            f[i] = sptype

        # Comment out any data types that are not needed
        if len(line[0]) != 1 and not line[0].startswith(band_name.split("_")[0]) and line[0][0] != '#' and line[0] != 'sptype' and line[0] != 'uvsptype':
            f[i] = '#'+f[i]

    print(starname, f"Adding {band_name}.fits path .dat")
 
    with open(datpath+fstarname, 'w') as datfile:
         datfile.writelines(f)

def mrg2fits(starname, inpath, outpath, waveregion="Opt", outstarname=None):
    os.chdir('/Users/cgunasekera/measure_extinction/measure_extinction/utils/')

    # Running the command line command:
    # python3 merge_stis_spec.py --ralph --inpath=[inpath] --outpath=[outpath] [starname]
    
    command_args = ["python3",'-W ignore', "merge_stis_spec.py",'--inpath='+inpath,'--outpath='+outpath,'--png']

    if data_set == "F19": command_args.append("--ralph")
    if outstarname != None: command_args.append('--outname='+outstarname)
    command_args.append('--waveregion='+waveregion)
    command_args.append(starname)
    print(command_args)
    subprocess.run(command_args)
    if outstarname != None: starname = outstarname
    print(starname, f"Merged data to create {starname}_stis_{waveregion}.fits")

    starname = starname.split("_")[0]

    #Search and add stis data file path to .dat file
#    AddDatSTISpath(starname, waveregion, outpath, sample_summary=data_set)

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

def get_stars_list(file_path, data_set, starname_col = 0):
    if data_set == "F19": stars_file = "fitzpatrick_2019_stars.dat"
    elif data_set == "HighLowRv": stars_file = "HighLowRv_HST.dat"
    with open(f"{file_path}{stars_file}") as datfile:
            f=datfile.readlines()
    
    stars_list = []
    for star in f:
        if star.startswith("#"):
            continue
        star_line = star.split()
        if data_set == "F19":
            index = [idx for idx, s in enumerate(star.split()[1:]) if s.startswith("B") or s.startswith("O") or s.startswith("···")][0]+1
            starname = get_f19tostarname(" ".join(star_line[:index]))
        else: #if data_set == "HighLowRv_HST.dat":
            starname = star_line[starname_col].lower()
        stars_list.append(starname)
    
    return stars_list


def prep_obs_data(data_set, file_path, inpath, outpath):
    stars_list = []
    program_stars_list = get_stars_list(file_path=file_path, data_set=data_set)

#    i=0
#    for (current_dir, dirs, files) in os.walk(inpath, topdown = 'true'):
#        for file in files:
#            starname = file.split('.')[0]
#            if data_set == "F19" and file.endswith(".mrg") and starname in program_stars_list:# and starname not in not_tlusty:
#                    stars_list.append(starname)
#            elif data_set == "HighLowRv" and file.endswith(".fits"):# and starname in program_stars_list:
#                stars_list.append(starname)
#
#    print(len(stars_list)," stars found.")
#    missing_data_stars = [star for star in program_stars_list if star not in stars_list]
#    #print("Missing data for stars:", missing_data_stars)
#
#    if data_set == "F19":
#        for starname in stars_list:
#            #if not starname == "hd210121":
#            #    continue
#            print("Creating STIS .fits file for ", starname)
#            mrg2fits(starname, inpath, outpath)
#            i+=1
#    
#    elif data_set == "HighLowRv":
    outstarname = program_stars_list #get_stars_list(file_path=file_path, data_set=data_set, starname_col=1)
    for (current_dir, dirs, files) in os.walk(inpath, topdown = 'true'):
        i=0
        merged_stars = []
        if current_dir != inpath:
            continue
        for file in files:
            if file.endswith(".fits") and file[:6] not in merged_stars:
                merged_stars.append(file[:6])
                hdul = fits.open(inpath+file)
                starname = hdul[0].header["TARGNAME"]
                file_name = file.split("_")[0]

                if not starname == "HD200775": #starname.endswith("37023"):
                    continue

                optical_element = hdul[0].header["OPT_ELEM"]

                wave_region = ""
                if optical_element == "G750L" or optical_element == "G430L":
                    wave_region = "Opt"
                else:
                    wave_region == "UV"

                if starname.startswith("HD"):
                    starname = "HD"+starname[2:]
                    print(starname)
                if starname.lower() in program_stars_list:
                    index = [idx for idx,s in enumerate(program_stars_list) if s == starname.lower()][0]
                    print("DOING THIS 1")
                    suffix1 = file.split(".")[0].split("_")[-1]
                    suffix2 = file.split("_")[0].split("j")[-1]
                    mrg2fits(file_name[:6], inpath, outpath, waveregion="UV", outstarname=f"{outstarname[index]}")
                    mrg2fits(file_name[:6], inpath, outpath, waveregion="Opt", outstarname=f"{outstarname[index]}")
                    i+=1
                elif starname.lower() == "ngc2264-vas47":
                    mrg2fits(file_name[:6], inpath, outpath, waveregion="UV", outstarname="walker67")
                    mrg2fits(file_name[:6], inpath, outpath, waveregion="Opt", outstarname="walker67")
                    i+=1
                elif starname.lower() == "hd200775":# or starname.lower() == "ngc2264-vas47":
                    #mrg2fits(file_name[:6], inpath, outpath, waveregion="UV", outstarname="hd200775")
                    mrg2fits(file_name[:6], inpath, outpath, waveregion="Opt", outstarname="hd200775")
                    i+=1

    print(f"{i} stars fits files done.")


###########################################
################  MAIN BODY ###############
###########################################

if __name__ == "__main__":

    file_path = "/Users/cgunasekera/extstar_data/"
    data_set = "HighLowRv"
    # OPTIONS:
    #   "F19"
    #   "HighLowRv"

    inpath = f"{file_path}DAT_files/STIS_Data/HighLowRv_Orig/" #location of original reduced data, *.mrg files
    # OPTIONS:
    #   F19 STIS:   "DAT_files/STIS_Data/F19_Orig/Opt/"
    #   HighLowRv:  "DAT_files/STIS_Data/HighLowRv_Orig/"

    outpath = f"{file_path}DAT_files/STIS_Data/" #location of files to output, *.fits files

    #parser = argparse.ArgumentParser(description="Run merge_stis_spec on set of data, and add .fits path to .dat file.")
    #parser.add_argument('file_path', type=str, default=None, help="")
    #parser.add_argument('data_set', type=str, help=".")
    #parser.add_argument('--inpath', type=str, default=None, help="")
    #parser.add_argument('--outpath', type=str, default=None, help="")

    #args = parser.parse_args()

    #prep_obs_data(args.data_set, args.file_path, args.inpath, args.outpath)
    prep_obs_data(data_set, file_path, inpath, outpath)
