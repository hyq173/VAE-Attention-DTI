from collections import OrderedDict
import json
import math
import numpy as np
import pandas as pd
from datahelper import  *
from hyperparameter import *

def read_davis_kiba(hp):
    ligands = json.load(open(hp.dataset_path + "/ligands_iso.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(hp.dataset_path + "/proteins.txt"), object_pairs_hook=OrderedDict)
    ligands = orderdict_list(ligands)  # 拿到的是value值
    proteins = orderdict_list(proteins)  # 拿到的是value值
    if 'davis' in hp.dataset_path:
        affinities = pd.read_csv(hp.dataset_path + '/interaction.txt', sep='\s+', header=None, encoding='latin1')
        affinities = -(np.log10(affinities / (math.pow(10, 9))))
    else:
        affinities = pd.read_csv(hp.dataset_path + '/interaction.txt', sep='\s+', header=None, encoding='latin1')
    affinities = np.array(affinities)
    XD, XT = [],[]
    for d in ligands:
        XD.append(label_smiles(d, hp.max_smi_len, CHARISOSMISET))
    for t in proteins:
        XT.append(label_sequence(t, hp.max_seq_len, CHARPROTSET))
    return XD, XT, affinities

def get_avg_var_davis_kiba(hp, label_row_inds, label_col_inds):
    XD, XT, affinities = read_davis_kiba(hp)
    if hp.sub == 'daivs':
        mean = np.mean(affinities)
        var = np.var(affinities)
    else: # kiba
        tem = np.zeros(len(label_col_inds))
        for ind in range(len(label_row_inds)):
            tem[ind] = affinities[label_row_inds[ind]][label_col_inds[ind]]
        mean = np.mean(tem)
        var = np.var(tem)
    return mean, var

# 0-4:5类
def get_classify_davis(hp, dataset, ind, Y, label_row_inds, label_col_inds, mean, var, count):
    Y_value = pd.Series(Y[label_row_inds[ind]][label_col_inds[ind]])
    if hp.sub == 'davis':
        if Y_value.between(mean - 1 * var, 1 * mean + var).bool():
            dataset[ind].append(np.array([1,0,0,0,0,], dtype=np.float32))
            count[0] += 1
        elif Y_value.between(mean - 2 * var, mean + 2 * var).bool():
            dataset[ind].append(np.array([0,1,0,0,0,], dtype=np.float32))
            count[1] += 1
        elif Y_value.between(mean - 3 * var, mean + 3 * var).bool():
            dataset[ind].append(np.array([0,0,1,0,0,], dtype=np.float32))
            count[2] += 1
        elif Y_value.between(mean - 4 * var, mean + 4 * var).bool():
            dataset[ind].append(np.array([0,0,0,1,0,], dtype=np.float32))
            count[3] += 1
        # elif Y_value.between(mean - 5 * var, mean + 5 * var).bool():
        #     dataset[ind].append(np.array([0,0,0,0,1,0,0], dtype=np.float32))
        #     count[4] += 1
        # elif Y_value.between(mean - 6 * var, mean + 6 * var).bool():
        #     dataset[ind].append(np.array([0,0,0,0,0,1,0], dtype=np.float32))
        #     count[5] += 1
        else:
            dataset[ind].append(np.array([0,0,0,0,1], dtype=np.float32))
            count[4] += 1
    else:  # kiba
        # if Y_value.between(mean - 2 * var, 2 * mean + var).bool():
        #     dataset[ind].append(np.array([1,0,0,0], dtype=np.float32))
        #     count[0] += 1
        # elif Y_value.between(mean - 4 * var, mean + 4 * var).bool():
        #     dataset[ind].append(np.array([0,1,0,0], dtype=np.float32))
        #     count[1] += 1
        # elif Y_value.between(mean - 6 * var, mean + 6 * var).bool():
        #     dataset[ind].append(np.array([0,0,1,0], dtype=np.float32))
        #     count[2] += 1
        # else:
        #     dataset[ind].append(np.array([0,0,0,1], dtype=np.float32))
        #     count[3] += 1

        if Y_value.between(mean - 1 * var, 1 * mean + var).bool():
            dataset[ind].append(np.array([1,0,0,0,0], dtype=np.float32))
            count[0] += 1
        elif Y_value.between(mean - 2 * var, mean + 2 * var).bool():
            dataset[ind].append(np.array([0,1,0,0,0], dtype=np.float32))
            count[1] += 1
        elif Y_value.between(mean - 3 * var, mean + 3 * var).bool():
            dataset[ind].append(np.array([0,0,1,0,0], dtype=np.float32))
            count[2] += 1
        elif Y_value.between(mean - 4 * var, mean + 4 * var).bool():
            dataset[ind].append(np.array([0,0,0,1,0], dtype=np.float32))
            count[3] += 1
        # elif Y_value.between(mean - 5 * var, mean + 5 * var).bool():
        #     dataset[ind].append(np.array([0,0,0,0,1,0,0], dtype=np.float32))
        #     count[4] += 1
        # elif Y_value.between(mean - 6 * var, mean + 6 * var).bool():
        #     dataset[ind].append(np.array([0,0,0,0,0,1,0], dtype=np.float32))
        #     count[5] += 1
        else:
            dataset[ind].append(np.array([0,0,0,0,1], dtype=np.float32))
            count[4] += 1

def get_dataset_dk(hp):
    XD, XT, affinities = read_davis_kiba(hp)
    label_row_inds, label_col_inds = np.where(np.isnan(affinities) == False)
    mean, var = get_avg_var_davis_kiba(hp, label_row_inds, label_col_inds) # 11.72,0.7
    dataset = [[]]
    count = np.zeros(hp.classify)
    for ind in range(len(label_row_inds)):
        dataset[ind].append(np.array(XD[label_row_inds[ind]], dtype=np.float32))
        dataset[ind].append(np.array(XT[label_col_inds[ind]], dtype=np.float32))
        get_classify_davis(hp, dataset, ind, affinities, label_row_inds, label_col_inds, mean, var, count)
        if ind < len(label_row_inds) -1 :
            dataset.append([])
    print(f'the number of each category:{count}')
    return dataset

# hp = HyperParameter()
# get_dataset_dk(hp)
