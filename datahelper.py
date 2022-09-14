import numpy as np
import os

# char iso-smiles set
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
CHARISOSMILEN = 64  # char iso-smiles set

#char protein set
CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21,
               "V": 22, "Y": 23, "X": 24,
               "Z": 25}
CHARPROTLEN = 25  #char protein set

def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind)))  # +1

  
    # seasons = ['Spring', 'Summer', 'Fall', 'Winter'] ==> list(enumerate(seasons)) ==> [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i, (smi_ch_ind[ch] - 1)] = 1
    return X

def one_hot_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind)))
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i, (smi_ch_ind[ch]) - 1] = 1
    return X

# [('CHEMBL230654', 'CCCC1=C(NC=N1)CNC2=CC(=C3C(=C2)C(=C(C=N3)C#N)NC4=CC(=C(C=C4)F)Cl)Cl'),(),()...]
def orderdict_list(dict):
    x = []
    for key in dict.keys():
        x.append(dict[key])
    return x

def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):  # x, smi_ch_ind, y
        X[i] = smi_ch_ind[ch]
    return X

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros(MAX_SEQ_LEN)
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def sample_to_number(hp, ligands, proteins):
    XD = []
    XT = []
    for d in ligands:
        XD.append(label_smiles(d,hp.max_smi_len, CHARISOSMISET))
    for t in proteins:
        XT.append(label_smiles(t,hp.max_seq_len, CHARPROTSET))
    return XD,XT

def read_dataset(hp, sub, dataset):
    chem = []
    chem_repr= []
    protein = []
    protein_repr = []
    with open(os.path.join(hp.dataset_path,sub,'chem'), 'r') as f:
        for line in f:
            chem.append(line.rstrip('\n'))
    with open(os.path.join(hp.dataset_path,sub,'chem.repr'), 'r') as f:
        for line in f:
            chem_repr.append(line.rstrip('\n'))
    with open(os.path.join(hp.dataset_path,sub,'protein'), 'r') as f:
        for line in f:
            protein.append(line.rstrip('\n'))
    protein_vocab = [i.strip() for i in open(os.path.join(hp.dataset_path,sub,'protein.vocab'), 'r').readlines()]
    with open(os.path.join(hp.dataset_path,sub,'protein.repr'), 'r') as f:
        for line in f:
            pro = [protein_vocab[int(i)] for i in line.rstrip('\n').split()]  
            pro = ''.join(pro)  
            protein_repr.append(pro)
    pos_edge = [i.strip().split(',')[1::2] for i in open(os.path.join(hp.dataset_path,sub,'edges.pos'), 'r').readlines()]
    neg_edge = [i.strip().split(',')[1::2] for i in open(os.path.join(hp.dataset_path,sub,'edges.neg'), 'r').readlines()]
    if 'train' == sub:
        dataset.train_chem = chem
        dataset.train_chem_repr = chem_repr
        dataset.train_protein = protein
        dataset.train_protein_repr = protein_repr
        dataset.train_pos_edge = pos_edge
        dataset.train_neg_edge = neg_edge
    elif 'dev' == sub:
        dataset.dev_chem = chem
        dataset.dev_chem_repr = chem_repr
        dataset.dev_protein = protein
        dataset.dev_protein_repr = protein_repr
        dataset.dev_pos_edge = pos_edge
        dataset.dev_neg_edge = neg_edge
    else:
        dataset.test_chem = chem
        dataset.test_chem_repr = chem_repr
        dataset.test_protein = protein
        dataset.test_protein_repr = protein_repr
        dataset.test_pos_edge = pos_edge
        dataset.test_neg_edge = neg_edge

def get_row_col_inds(hp, pos_edge, neg_edge, chem, protein):
    pos_drug_key = np.array(pos_edge)[:, 0]
    pos_protein_key = np.array(pos_edge)[:, 1]
    neg_drug_key = np.array(neg_edge)[:, 0]
    neg_protein_key = np.array(neg_edge)[:, 1]
    drug_key = np.concatenate((pos_drug_key, neg_drug_key), axis=0)
    protein_key = np.concatenate((pos_protein_key, neg_protein_key), axis=0)
    drug_index = []
    protein_index = []
    drug_index.extend([chem.index(d) for d in drug_key])
    protein_index.extend([protein.index(t) for t in protein_key])
    return drug_index, protein_index

def get_Y(hp, XD,XT,pos_edge, neg_edge, chem, protein):
    pos_drug_key = np.array(pos_edge)[:, 0]
    pos_protein_key = np.array(pos_edge)[:, 1]
    neg_drug_key = np.array(neg_edge)[:, 0]
    neg_protein_key = np.array(neg_edge)[:, 1]

    pos_drug_index = []
    pos_protein_index = []
    neg_drug_index = []
    neg_protein_index = []
    pos_drug_index.extend([chem.index(d) for d in pos_drug_key])
    pos_protein_index.extend([protein.index(t) for t in pos_protein_key])
    neg_drug_index.extend([chem.index(d) for d in neg_drug_key])
    neg_protein_index.extend([protein.index(t) for t in neg_protein_key])
    Y = [[-1] * len(XT)] * len(XD)
    for i in range(len(pos_drug_index)):
        Y[pos_drug_index[i]][pos_protein_index[i]] = 1
    for j in range(len(neg_drug_index)):
        Y[neg_drug_index[j]][neg_protein_index[j]] = 0
    return Y

def process_data(hp, sub, dataset):
    if 'train' == sub:
        XD, XT = sample_to_number(hp, dataset.train_chem_repr, dataset.train_protein_repr)
        XD = np.asarray(XD)
        XT = np.asarray(XT)
        # Y = np.asarray(get_Y(hp, XD, XT, dataset.train_pos_edge, dataset.train_neg_edge, dataset.train_chem, dataset.train_protein))
        drugcount = XD.shape[0]
        targetcount = XT.shape[0]
        # drug_index, protein_index = get_row_col_inds(hp, dataset.train_pos_edge, dataset.train_neg_edge, dataset.train_chem, dataset.train_protein)

        dataset.train_XD = XD
        dataset.train_XT = XT
        # dataset.train_Y = Y
        dataset.train_drugcount = drugcount
        dataset.train_targetcount = targetcount
        # dataset.train_label_row_inds = drug_index
        # dataset.train_label_col_inds = protein_index
    elif 'dev' == sub:
        XD, XT = sample_to_number(hp, dataset.dev_chem_repr, dataset.dev_protein_repr)
        XD = np.asarray(XD)
        XT = np.asarray(XT)
        # Y = np.asarray(get_Y(hp, XD, XT, dataset.dev_pos_edge, dataset.dev_neg_edge, dataset.dev_chem, dataset.dev_protein))
        drugcount = XD.shape[0]
        targetcount = XT.shape[0]
        # drug_index, protein_index = get_row_col_inds(hp, dataset.dev_pos_edge, dataset.dev_neg_edge, dataset.dev_chem, dataset.dev_protein)

        dataset.dev_XD = XD
        dataset.dev_XT = XT
        # dataset.dev_Y = Y
        dataset.dev_drugcount = drugcount
        dataset.dev_targetcount = targetcount
        # dataset.dev_label_row_inds = drug_index
        # dataset.dev_label_col_inds = protein_index
    else:
        XD, XT = sample_to_number(hp, dataset.test_chem_repr, dataset.test_protein_repr)
        XD = np.asarray(XD)
        XT = np.asarray(XT)
        # Y = np.asarray(get_Y(hp, XD, XT, dataset.test_pos_edge, dataset.test_neg_edge, dataset.test_chem, dataset.test_protein))
        drugcount = XD.shape[0]
        targetcount = XT.shape[0]
        # drug_index, protein_index = get_row_col_inds(hp, dataset.test_pos_edge, dataset.test_neg_edge, dataset.test_chem, dataset.test_protein)

        dataset.test_XD = XD
        dataset.test_XT = XT
        # dataset.test_Y = Y
        dataset.test_drugcount = drugcount
        dataset.test_targetcount = targetcount
        # dataset.test_label_row_inds = drug_index
        # dataset.test_label_col_inds = protein_index

def get_datasets(XD, XT, pos_edge, neg_edge, chem, protein):
    pos_drug_key = np.array(pos_edge)[:, 0]
    pos_protein_key = np.array(pos_edge)[:, 1]
    neg_drug_key = np.array(neg_edge)[:, 0]
    neg_protein_key = np.array(neg_edge)[:, 1]

    pos_drug_index = []
    pos_protein_index = []
    neg_drug_index = []
    neg_protein_index = []
    pos_drug_index.extend([chem.index(d) for d in pos_drug_key])
    pos_protein_index.extend([protein.index(t) for t in pos_protein_key])
    neg_drug_index.extend([chem.index(d) for d in neg_drug_key])
    neg_protein_index.extend([protein.index(t) for t in neg_protein_key])
    drug_index = np.concatenate((pos_drug_index, neg_drug_index), axis = 0)
    protein_index = np.concatenate((pos_protein_index, neg_protein_index), axis = 0)

    dataset = [[]]
    for ind in range(len(drug_index)):
        dataset[ind].append(np.array(XD[drug_index[ind]], dtype=np.float32))
        dataset[ind].append(np.array(XT[protein_index[ind]], dtype=np.float32))
        if ind < len(pos_drug_index):
            dataset[ind].append(np.array(1, dtype=np.float32))
        else:
            dataset[ind].append(np.array(0, dtype=np.float32))
        if ind < len(drug_index) - 1:
            dataset.append([])
    return dataset

def save_train_loss(hp, acc, auc, loss):
    with open(hp.train_acc_path, "a") as fw:
        fw.write("%f\n" % acc)
    with open(hp.train_auc_path, "a") as fw:
        fw.write("%f\n" % auc)
    with open(hp.train_loss_path, "a") as fw:
        fw.write("%f\n" % loss)

def prepare_interaction_pairs(XD, XT, Y, rows, cols):
    dataset = [[]]
    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        dataset[pair_ind].append(np.array(drug, dtype=np.float32))
        target = XT[cols[pair_ind]]
        dataset[pair_ind].append(np.array(target, dtype=np.float32))
        dataset[pair_ind].append(np.array(Y[rows[pair_ind], cols[pair_ind]], dtype=np.float32))
        if pair_ind < len(rows) - 1:
            dataset.append([])
    return dataset

class DataSet(object):
    def __init__(self):
        pass
