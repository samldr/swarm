import torch
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = "r10.dat"
path = "./augmented/"

headers = ["I-2 (A)", 
    "I-1 (A)", 
    "I-plate", 
    "I+4 (A)", 
    "I+5 (A)", 
    "Vs (V)", 
    "ne (m^-3)", 
    "T_e (eV)", 
    "Ti (eV)", 
    "vx (m/s)", 
    "mef", 
    "n_O+", 
    "n_N+", 
    "n_H+", 
    "n_He+",
    "I-plate M", 
    "I-plate P", 
    "I-Pedro", 
    "Simulation Name"]




def main():
# get dataset as a pandas DataFrame
    dataset = pd.read_table(path + file, header=None, comment='#', sep='\s+', usecols = list(range(0, 19)), names=headers)
    pd.set_option('display.precision', 8)
    # get the currents into a numpy array
    currents = pd.DataFrame(dataset[headers[:5]]).to_numpy()
    



main()