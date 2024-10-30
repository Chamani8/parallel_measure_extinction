#import os
import glob
import re
from measure_extinction.stardata import StarData
from measure_extinction.modeldata import ModelData

def DATstarname(starname):
    if starname.startswith("hd") and len(re.split(r'(\d+)', starname)[1]) < 6:
        starcode = re.split(r'(\d+)', starname)[1]
        starcode = starcode.zfill(6)
        dstarname = re.split(r'(\d+)', starname)[0] + starcode
    else:
        dstarname = starname
    
    return dstarname

def get_model_data(file_path, data_names, logTeff=None, logg=None, 
                   band_names = ["U", "B", "V", "J", "H", "K"]):
    #Note: the code will complain if have less than six band_names

    #fake_dict = {"BAND":-1, dataname:-1}
    #data_names = fake_dict.keys()
    
    add_str = ""
    if logTeff != None and logTeff > 0:
        sTeff = f"{10**logTeff}"
        if int(sTeff[1]) > 2 and int(sTeff[1]) < 8:
            add_str += "*t"+sTeff[0] 
    
    if logg != None and logg > 0:
        slogg = f"{logg}"
        if int(slogg[2]) > 2 and int(slogg[2]) < 8:
            add_str += "*g"+slogg[0]
    
    #Get the model data
    tlusty_models_fullpath = glob.glob("{}/Models/tlusty*v*.dat".format(file_path))
    tlusty_models = [
                        tfile[tfile.rfind("/") + 1 : len(tfile)] for tfile in tlusty_models_fullpath
                    ]
    
    # get the models with just the reddened star band data and spectra
    modinfo = ModelData(
        tlusty_models,
        path="{}/Models/".format(file_path),
        band_names=band_names,
        spectra_names=data_names,
    )

    return modinfo

def get_mask(reddened_star_column, invert = False):
    mask = []
    
    for dat in reddened_star_column:
        if dat == 0.0:
            mask.append(True)
        else:
            mask.append(False)

    if invert == True:
        mask = np.array(mask, dtype = bool)
        mask = list(~mask)
    
    return mask

if __name__ == "__main__":
    file_path = "/Users/cgunasekera/extstar_data/"
    starname = "hd18352"
    velocity = 0

    dstarname = DATstarname(starname)
    fstarname = f"{dstarname}.dat"

    not_tlusty = ["bd44d1080", "bd69d1231", "bd_71d92", "hd17443", "hd29647", "hd110336",
                  "hd112607",  "hd142165", "hd146285", "hd147196", "hd282485", "vss-viii10"]
    
    if starname in not_tlusty:
        fit_range = "all"
    else:
        fit_range = "g23"
    
    params = [4.418135498425232, 3.6984466059775913, 0.15, 1.4, 3.1, 0.0, 1.7, 0.5, 4.8, 0.8, 18.0, 18.0]

    fake_dict = {"BAND":-1, "STIS_Opt":-1}
    data_names = fake_dict.keys()
    modinfo = get_model_data(file_path, data_names, logTeff=params[0], logg=params[1],
                             band_names="J")

