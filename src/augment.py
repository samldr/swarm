import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random
import scipy.constants as const
from shutil import copyfileobj
from os import remove
import argparse


def augment(current, e, tau, r):

    t = 0
    for i in range(0,12):
        w = random() - 0.5
        t += w

    sigma_0 = math.sqrt(abs(current*e/tau)) 

    augmented_current = current + t * sigma_0 * r

    return augmented_current

def histogram(dataset):

    iqr = np.subtract(*np.percentile(dataset, [75, 25]))
    binwidth = 2 * iqr / (len(dataset)**(1/3))
    bins = math.floor((max(dataset) - min(dataset))/binwidth)
    
    
    plt.hist(dataset, bins=bins)
    plt.xlabel("Current")
    plt.ylabel("Frequency")
    plt.show()

def organize_augmented(original_dataframe, augmented_tensor, headers, new_entries):

    
    augmented_nparray = pd.DataFrame(original_dataframe[headers]).to_numpy()
    original_nparray = augmented_nparray

    for index, row in enumerate(augmented_nparray):
        for new_row in range(0, new_entries):
            augmented_currents = []
            for index2, current in enumerate(row[:5]):
                augmented_currents.append(augmented_tensor[index][index2][new_row])
            

            augmented_row = np.concatenate((np.array(augmented_currents), original_nparray[index][5:]))
            augmented_nparray = np.insert(augmented_nparray, (index+1) + index*new_entries + new_row, augmented_row, axis=0)


    augmented_dataset = pd.DataFrame(augmented_nparray)
    for index in range(18):
        augmented_dataset[index] = augmented_dataset[index].apply(pd.to_numeric, downcast='float').fillna(0)
    
    return augmented_dataset

    # pd.DataFrame.to_csv(augmented_dataset, "./r"+ str(r) +".dat", sep="\t", index=False, header=False, encoding='utf-8', float_format="%.3e")
        
def write_dat(dataframe, training_ndata, training_file, output_dir, new_entries, r):
    with open(training_file) as input:
        with open(output_dir + 'header.dat', 'a') as output:
            lines = input.readlines()
            for line in lines:
                if line.startswith("# ndata"):
                    output.write("# ndata=" + str(training_ndata + training_ndata * new_entries ) + "\n")
                elif line.startswith("#ndata"):
                    output.write("#ndata=" + str(training_ndata + training_ndata * new_entries) + "\n")
                elif line.startswith("#"):
                    output.write(line)
                else:
                    break

    pd.DataFrame.to_csv(dataframe, tempoutput:= output_dir + "tempr"+ str(r) +".dat", sep="\t", index=False, header=False, encoding='utf-8', float_format="%.3e")

    with open(output_dir + 'r'+ str(r) + '.dat','wb') as wfd:
        for f in [output_dir + 'header.dat',tempoutput]:
            with open(f,'rb') as fd:
                copyfileobj(fd, wfd)

    remove(output_dir + "header.dat")
    remove(tempoutput)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="input file path")
    parser.add_argument("output", type=str, help="output directory path")
    parser.add_argument("r", type=int, help="r value")
    parser.add_argument("-t", "--tau", type=float, help="sampling time", default=10e-3)
    parser.add_argument("-e", "--entries", type=int, help="agumented data entries added per entry of training data", default=10)
    args = parser.parse_args()

    training_file = args.input
    output_dir = args.output
    r = args.r
    tau = args.tau
    new_entries = args.entries

    augmented = [] # placeholder

    e = const.e 

    augmentedfile = "./r" + str(r)
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
    


    # get dataset as a pandas DataFrame
    dataset = pd.read_table(training_file, header=None, comment='#', sep='\s+', usecols = list(range(0, 19)), names=headers)
    pd.set_option('display.precision', 8)
    # get the currents into a numpy array
    currents = pd.DataFrame(dataset[headers[:5]]).to_numpy()
    training_ndata = len(currents)
    
    augmented_tensor = []
    for row in currents:

        augmented_matrix = []
        for current in row:
            
            maxaugcur = 0
            augmented_row = []
            for index in range(new_entries):
                augmented_row.append(augcur := augment(current, e, tau, r))

            # percent = abs(( augcur - current) / current) * 100
            # print(percent)
            
                
            augmented_matrix.append(augmented_row)

        
        # print(maxaugcur)
        augmented_tensor.append(augmented_matrix)


    augmented_dataframe = organize_augmented(dataset, augmented_tensor, headers, new_entries)

    write_dat(augmented_dataframe, training_ndata, training_file, output_dir, new_entries, r)

if __name__ == '__main__':
    main()