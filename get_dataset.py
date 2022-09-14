'''
封装数据到pkl文件中，方便下次读取
'''
import pickle
from process_human_cele import *
from process_davis_kiba import *
from hyperparameter import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def binddingDB():
    dataset = DataSet()
    hp = HyperParameter()
    for sub in ['train', 'dev', 'test']:
        print('读取%s数据' % sub)
        read_dataset(hp, sub, dataset)
        print('处理%s数据' % sub)
        process_data(hp, sub, dataset)
    train_dataset = get_datasets(dataset.train_XD, dataset.train_XT, dataset.train_pos_edge, dataset.train_neg_edge,
                                 dataset.train_chem, dataset.train_protein)
    train_loader = DataLoader(dataset=train_dataset, batch_size=hp.batch_size, shuffle=True)
    valid_dataset = get_datasets(dataset.dev_XD, dataset.dev_XT, dataset.dev_pos_edge, dataset.dev_neg_edge,
                                 dataset.dev_chem, dataset.dev_protein)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=hp.batch_size)
    test_dataset = get_datasets(dataset.test_XD, dataset.test_XT, dataset.test_pos_edge, dataset.test_neg_edge,
                                dataset.test_chem, dataset.test_protein)
    test_loader = DataLoader(dataset=test_dataset, batch_size=hp.batch_size)
    dataset.train_loader = train_loader
    dataset.valid_loader = valid_loader
    dataset.test_loader = test_loader
    pickle.dump(dataset, file=open(hp.pkl_path, 'wb'))
    print(f'处理结束,pkl文件保存在{hp.pkl_path}')
    print('已保存')

def human_celegans():
    dataset = DataSet()
    hp = HyperParameter()
    human_dataset = get_dataset_human_celegans(hp)
    train, valid = train_test_split(human_dataset,test_size=0.4)
    valid, test = train_test_split(valid, test_size=0.5)
    train_loader = DataLoader(dataset=train, batch_size=hp.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid, batch_size=hp.batch_size)
    test_loader = DataLoader(dataset=test, batch_size=hp.batch_size)
    dataset.train_loader = train_loader
    dataset.valid_loader = valid_loader
    dataset.test_loader = test_loader
    pickle.dump(dataset, file=open(hp.pkl_path, 'wb'))
    print(f'pkl文件保存在{hp.pkl_path}')
    print('已保存')

def davis_kiba():
    dataset = DataSet()
    hp = HyperParameter()
    dk_dataset = get_dataset_dk(hp)
    train, valid = train_test_split(dk_dataset, test_size=0.4)
    valid, test = train_test_split(valid, test_size=0.5)
    train_loader = DataLoader(dataset=train, batch_size=hp.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid, batch_size=hp.batch_size)
    test_loader = DataLoader(dataset=test, batch_size=hp.batch_size)
    dataset.train_loader = train_loader
    dataset.valid_loader = valid_loader
    dataset.test_loader = test_loader
    pickle.dump(dataset, file=open(hp.pkl_path, 'wb'))
    print(f'pkl文件保存在{hp.pkl_path}')
    print('已保存')


if __name__ == '__main__':
    # davis_kiba()
    binddingDB()
