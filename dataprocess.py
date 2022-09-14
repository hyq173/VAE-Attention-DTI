import os

def process_bindingDB():
    for sub in ['dev','test','train']:
        chem = []
        chem_repr = []
        with open(os.path.join('data/bindingDB',sub,'chem'), 'r') as f:
            for line in f: chem.append(line.rstrip('\n'))
        with open(os.path.join(r'data/bindingDB',sub,'chem.repr'), 'rU') as f:
            for line in f: chem_repr.append(line.rstrip('\n'))
        drug = dict(zip(chem, chem_repr))  # 等长列表转字典
        save_data(os.path.join('data/bindingDB',sub,'ligands_iso.txt'),drug)

        protein = []
        protein_repr = []
        with open(os.path.join('data/bindingDB',sub,'protein'), 'r') as f:
            for line in f: protein.append(line.rstrip('\n'))
        protein_vocab = [i.strip() for i in open(os.path.join('data/bindingDB',sub, 'protein.vocab'), 'r').readlines()]
        with open(os.path.join('data/bindingDB',sub,'protein.repr'), 'r') as f:
            for line in f:
                pro = [protein_vocab[int(i)] for i in line.rstrip('\n').split()]  # 去掉回车、去掉空格
                pro = ''.join(pro)  # 列表转为字符串
                protein_repr.append(pro)
        proteins = dict(zip(protein, protein_repr))  # 登场列表转字典
        save_data(os.path.join('data/bindingDB',sub,'proteins.txt'), proteins)

        # '''
        # 处理边
        # '''
        # posEdge = [i.strip().split(',')[1::2] for i in open(os.path.join('data/bindingDB', sub, 'edges.pos'), 'r').readlines()]
        # negEdge = [i.strip().split(',')[1::2] for i in open(os.path.join('data/bindingDB', sub, 'edges.neg'), 'r').readlines()]

# def read_drug_protein(drug_path, protein_path, edges_pos, edges_neg):
#     chem_repr = []
#     protein_repr = []
#     with open(drug_path, 'r') as f:
#         for line in f:
#             chem_repr.append(line.rstrip('\n'))
#     with open(protein_path, 'r') as f:
#         for line in f:
#             protein_repr.append(line.rstrip('\n'))
#
#     pos_edge = [i.strip().split(',')[1::2] for i in open(edges_pos, 'r').readlines()]
#     neg_edge = [i.strip().split(',')[1::2] for i in open(edges_neg, 'r').readlines()]
#     return chem_repr, protein_repr, pos_edge, neg_edge

def save_data(fpath, content):
    with open( fpath, "w" ) as fw: fw.write("%s\n" % content)

if __name__ == '__main__':
    process_bindingDB()


