import h5py as h5
from pyedflib.highlevel import read_edf
import pandas as pd
import os
# signal, signal_header, file_header = read_edf(edf_file=file_path)


SNU_PATH = ""
CHB_PATH = ""

SNU_filename = os.listdir(SNU_PATH)
CHB_filename = os.listdir(CHB_PATH)

print(SNU_filename)
