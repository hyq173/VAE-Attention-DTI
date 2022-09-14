from datahelper import  *

def read_human_celegans(hp):
    chems, proteins, affinities = [], [], []
    with open(os.path.join(hp.dataset_path,'data.txt'), 'r') as f:
        for line in f:
            chem, protein, affinity = line.rstrip('\n').split()
            chems.append(label_smiles(chem, hp.max_smi_len, CHARISOSMISET))
            proteins.append(label_sequence(protein, hp.max_seq_len, CHARPROTSET))
            affinities.append(int(affinity))
    return chems, proteins, affinities

def get_dataset_human_celegans(hp):
    chems, proteins, affinities = read_human_celegans(hp)
    dataset = [[]]
    for ind in range(len(chems)):
        dataset[ind].append(np.array(chems[ind], dtype=np.float32))
        dataset[ind].append(np.array(proteins[ind], dtype=np.float32))
        if(hp.classify == 1):
            dataset[ind].append(np.array(affinities[ind], dtype=np.float32))
        elif(hp.classify == 2):
            if affinities[ind] == 0:
                dataset[ind].append(np.array([1,0], dtype=np.float32))
            else:
                dataset[ind].append(np.array([0,1], dtype=np.float32))
        if ind < len(chems) - 1:
            dataset.append([])
    return dataset
