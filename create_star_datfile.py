# Create missing .dat files with data locations and photometry

from read_data import DATstarname
import os

if __name__ == "__main__":
    file_path = "/Users/cgunasekera/extstar_data/DAT_files/HighLowRv_HST_data/"
    
    for (current_dir, dirs, files) in os.walk(file_path, topdown = 'true'):
        for file in files:
            starname = file.split('.')[0]
            print(starname)
    
    #starname = "hd18352"
    #velocity = 0
#
    #dstarname = DATstarname(starname)
    #fstarname = f"{dstarname}.dat"
#
    #print(fstarname)