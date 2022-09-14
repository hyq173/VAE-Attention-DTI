# Traditional machine learning

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import metrics as skmetrics
import pickle
from process_davis_kiba import *
from hyperparameter import *

# def my_KNN():
#     iris = datasets.load_iris()
#     X = iris.data
#     y = iris.target
#     print(X, y)
#     KNeighborsClassifier(n_neighbors=3)
#
# my_KNN()
################################【davis dataset】################################
def get_davis():
    # hp = HyperParameter()
    # dataset = get_dataset_dk(hp)
    # pickle.dump(dataset, file=open('other/temp_davis.pkl', 'wb'))
    dataset = pickle.load(file=open('other/temp_davis.pkl', 'rb'))
    data = []
    target = []
    for d in range(len(dataset)):
        data.append(np.hstack((dataset[d][0], dataset[d][1])))
        target.append(np.flatnonzero(dataset[d][2]))
    data = np.array(data)
    target = np.array(target)
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=2003)
    y_test_mul = []
    for i in range(len(y_test)):
        if(y_test[i] == 0):
            y_test_mul.append([1, 0, 0, 0, 0])
        elif(y_test[i] == 1):
            y_test_mul.append([0, 1, 0, 0, 0])
        elif (y_test[i] == 2):
            y_test_mul.append([0, 0, 1, 0, 0])
        elif (y_test[i] == 3):
            y_test_mul.append([0, 0, 0, 1, 0])
        elif (y_test[i] == 4):
            y_test_mul.append([0, 0, 0, 0, 1])
    y_test_mul = np.array(y_test_mul)
    return X_train, X_test, y_train, y_test, y_test_mul
def davis_KNN():
    X_train, X_test, y_train, y_test, y_test_mul = get_davis()

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    pre = clf.predict(X_test)
    pre_mul = []
    for j in range(len(pre)):
        if (pre[j] == 0):
            pre_mul.append([1, 0, 0, 0, 0])
        elif (pre[j] == 1):
            pre_mul.append([0, 1, 0, 0, 0])
        elif (pre[j] == 2):
            pre_mul.append([0, 0, 1, 0, 0])
        elif (pre[j] == 3):
            pre_mul.append([0, 0, 0, 1, 0])
        elif (pre[j] == 4):
            pre_mul.append([0, 0, 0, 0, 1])
    correct = 0
    for d in range(len(pre)):
        if(pre[d] == y_test[d]):
            correct+=1
    acc = correct / len(X_test)
    auc = skmetrics.roc_auc_score(y_test_mul, pre_mul, multi_class = 'ovo', labels = [0, 1, 2, 3, 4])
    precision = skmetrics.precision_score(y_test_mul, pre_mul, average='macro')
    recall = skmetrics.recall_score(y_test_mul, pre_mul, average='macro')
    F1 = skmetrics.f1_score(y_test_mul, pre_mul, average='macro')
    print('===========================Davis-KNN==================================')
    # print(f" Accuracy is:{acc:3f}\n AUC is : {auc:3f}")
    print(f" acc is:{acc:3f}\n AUC is:{auc:3f}\n precision is : {precision:3f}\n recall is:{recall:3f}\n F1 is:{F1:3f}")
    return


def davis_RF():
    X_train, X_test, y_train, y_test, y_test_mul = get_davis()
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    pre = rf.predict(X_test)
    pre_mul = []
    for j in range(len(pre)):
        if (pre[j] == 0):
            pre_mul.append([1, 0, 0, 0, 0])
        elif (pre[j] == 1):
            pre_mul.append([0, 1, 0, 0, 0])
        elif (pre[j] == 2):
            pre_mul.append([0, 0, 1, 0, 0])
        elif (pre[j] == 3):
            pre_mul.append([0, 0, 0, 1, 0])
        elif (pre[j] == 4):
            pre_mul.append([0, 0, 0, 0, 1])
    correct = 0
    for d in range(len(pre)):
        if(pre[d] == y_test[d]):
            correct+=1
    acc = correct / len(X_test)
    auc = skmetrics.roc_auc_score(y_test_mul, pre_mul, multi_class = 'ovo', labels = [0, 1, 2, 3, 4])
    precision = skmetrics.precision_score(y_test_mul, pre_mul, average='macro')
    recall = skmetrics.recall_score(y_test_mul, pre_mul, average='macro')
    F1 = skmetrics.f1_score(y_test_mul, pre_mul, average='macro')
    print('===========================davis-RF==================================')
    # print(f" Accuracy is:{acc:3f}\n AUC is : {auc:3f}")
    print(f" acc is:{acc:3f}\n AUC is:{auc:3f}\n precision is : {precision:3f}\n recall is:{recall:3f}\n F1 is:{F1:3f}")
    return


def davis_L2():
    X_train, X_test, y_train, y_test, y_test_mul = get_davis()
    r = Ridge(solver='sag')
    r.fit(X_train, y_train)
    pre = r.predict(X_test)
    pre = [1 if i >= 0.5 else 0 for i in pre]
    pre_mul = []
    for j in range(len(pre)):
        if (pre[j] == 0):
            pre_mul.append([1, 0, 0, 0, 0])
        elif (pre[j] == 1):
            pre_mul.append([0, 1, 0, 0, 0])
        elif (pre[j] == 2):
            pre_mul.append([0, 0, 1, 0, 0])
        elif (pre[j] == 3):
            pre_mul.append([0, 0, 0, 1, 0])
        elif (pre[j] == 4):
            pre_mul.append([0, 0, 0, 0, 1])
    correct = 0
    for d in range(len(pre)):
        if (pre[d] == y_test[d]):
            correct += 1
    acc = correct / len(X_test)
    auc = skmetrics.roc_auc_score(y_test_mul, pre_mul, multi_class='ovo', labels=[0, 1, 2, 3, 4])
    precision = skmetrics.precision_score(y_test_mul, pre_mul, average='macro')
    recall = skmetrics.recall_score(y_test_mul, pre_mul, average='macro')
    F1 = skmetrics.f1_score(y_test_mul, pre_mul, average='macro')
    print('===========================davis-L2==================================')
    # print(f" Accuracy is:{acc:3f}\n AUC is : {auc:3f}")
    print(f" acc is:{acc:3f}\n AUC is:{auc:3f}\n precision is : {precision:3f}\n recall is:{recall:3f}\n F1 is:{F1:3f}")
    return


def davis_SVM():
    X_train, X_test, y_train, y_test, y_test_mul = get_davis()
    svm = SVC(kernel='rbf')
    svm.fit(X_train, y_train)
    pre = svm.predict(X_test)

    pre_mul = []
    for j in range(len(pre)):
        if (pre[j] == 0):
            pre_mul.append([1, 0, 0, 0, 0])
        elif (pre[j] == 1):
            pre_mul.append([0, 1, 0, 0, 0])
        elif (pre[j] == 2):
            pre_mul.append([0, 0, 1, 0, 0])
        elif (pre[j] == 3):
            pre_mul.append([0, 0, 0, 1, 0])
        elif (pre[j] == 4):
            pre_mul.append([0, 0, 0, 0, 1])
    correct = 0
    for d in range(len(pre)):
        if (pre[d] == y_test[d]):
            correct += 1
    acc = correct / len(X_test)
    auc = skmetrics.roc_auc_score(y_test_mul, pre_mul, multi_class='ovo', labels=[0, 1, 2, 3, 4])
    precision = skmetrics.precision_score(y_test_mul, pre_mul, average='macro')
    recall = skmetrics.recall_score(y_test_mul, pre_mul, average='macro')
    F1 = skmetrics.f1_score(y_test_mul, pre_mul, average='macro')
    print('===========================davis-SVM==================================')
    print(f" Accuracy is:{acc:3f}\n AUC is : {auc:3f}")
    print(f" acc is:{acc:3f}\n AUC is:{auc:3f}\n precision is : {precision:3f}\n recall is:{recall:3f}\n F1 is:{F1:3f}")
    return


################################【kiba dataset】################################
def get_kiba():
    # hp = HyperParameter()
    # dataset = get_dataset_dk(hp)
    # pickle.dump(dataset, file=open('other/temp_kiba.pkl', 'wb'))
    dataset = pickle.load(file=open('other/temp_kiba.pkl', 'rb'))
    data = []
    target = []
    for d in range(len(dataset)):
        data.append(np.hstack((dataset[d][0], dataset[d][1])))
        target.append(np.flatnonzero(dataset[d][2]))
    data = np.array(data)
    target = np.array(target)
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=2003)
    y_test_mul = []
    for i in range(len(y_test)):
        if (y_test[i] == 0):
            y_test_mul.append([1, 0, 0, 0, 0])
        elif (y_test[i] == 1):
            y_test_mul.append([0, 1, 0, 0, 0])
        elif (y_test[i] == 2):
            y_test_mul.append([0, 0, 1, 0, 0])
        elif (y_test[i] == 3):
            y_test_mul.append([0, 0, 0, 1, 0])
        elif (y_test[i] == 4):
            y_test_mul.append([0, 0, 0, 0, 1])
    y_test_mul = np.array(y_test_mul)
    return X_train, X_test, y_train, y_test, y_test_mul
def kiba_KNN():
    X_train, X_test, y_train, y_test, y_test_mul = get_kiba()
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    pre = clf.predict(X_test)
    pre_mul = []
    for j in range(len(pre)):
        if (pre[j] == 0):
            pre_mul.append([1, 0, 0, 0, 0])
        elif (pre[j] == 1):
            pre_mul.append([0, 1, 0, 0, 0])
        elif (pre[j] == 2):
            pre_mul.append([0, 0, 1, 0, 0])
        elif (pre[j] == 3):
            pre_mul.append([0, 0, 0, 1, 0])
        elif (pre[j] == 4):
            pre_mul.append([0, 0, 0, 0, 1])
    correct = 0
    for d in range(len(pre)):
        if(pre[d] == y_test[d]):
            correct+=1
    acc = correct / len(X_test)
    auc = skmetrics.roc_auc_score(y_test_mul, pre_mul, multi_class = 'ovo', labels = [0, 1, 2, 3, 4])
    precision = skmetrics.precision_score(y_test_mul, pre_mul, average='macro')
    recall = skmetrics.recall_score(y_test_mul, pre_mul, average='macro')
    F1 = skmetrics.f1_score(y_test_mul, pre_mul, average='macro')
    print('===========================KIBA-KNN==================================')
    # print(f" Accuracy is:{acc:3f}\n AUC is : {auc:3f}")
    print(f" acc is:{acc:3f}\n AUC is:{auc:3f}\n precision is : {precision:3f}\n recall is:{recall:3f}\n F1 is:{F1:3f}")

def kiba_RF():
    X_train, X_test, y_train, y_test, y_test_mul = get_kiba()
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    pre = rf.predict(X_test)
    pre_mul = []
    for j in range(len(pre)):
        if (pre[j] == 0):
            pre_mul.append([1, 0, 0, 0, 0])
        elif (pre[j] == 1):
            pre_mul.append([0, 1, 0, 0, 0])
        elif (pre[j] == 2):
            pre_mul.append([0, 0, 1, 0, 0])
        elif (pre[j] == 3):
            pre_mul.append([0, 0, 0, 1, 0])
        elif (pre[j] == 4):
            pre_mul.append([0, 0, 0, 0, 1])
    correct = 0
    for d in range(len(pre)):
        if (pre[d] == y_test[d]):
            correct += 1
    acc = correct / len(X_test)
    auc = skmetrics.roc_auc_score(y_test_mul, pre_mul, multi_class='ovo', labels=[0, 1, 2, 3, 4])
    precision = skmetrics.precision_score(y_test_mul, pre_mul, average='macro')
    recall = skmetrics.recall_score(y_test_mul, pre_mul, average='macro')
    F1 = skmetrics.f1_score(y_test_mul, pre_mul, average='macro')
    print('===========================kiba-RF==================================')
    # print(f" Accuracy is:{acc:3f}\n AUC is : {auc:3f}")
    print(f" acc is:{acc:3f}\n AUC is:{auc:3f}\n precision is : {precision:3f}\n recall is:{recall:3f}\n F1 is:{F1:3f}")
    return


def kiba_L2():
    X_train, X_test, y_train, y_test, y_test_mul = get_kiba()
    r = Ridge(solver='sag')
    r.fit(X_train, y_train)
    pre = r.predict(X_test)
    pre = [1 if i >= 0.5 else 0 for i in pre]
    pre_mul = []
    for j in range(len(pre)):
        if (pre[j] == 0):
            pre_mul.append([1, 0, 0, 0, 0])
        elif (pre[j] == 1):
            pre_mul.append([0, 1, 0, 0, 0])
        elif (pre[j] == 2):
            pre_mul.append([0, 0, 1, 0, 0])
        elif (pre[j] == 3):
            pre_mul.append([0, 0, 0, 1, 0])
        elif (pre[j] == 4):
            pre_mul.append([0, 0, 0, 0, 1])
    correct = 0
    for d in range(len(pre)):
        if (pre[d] == y_test[d]):
            correct += 1
    acc = correct / len(X_test)
    auc = skmetrics.roc_auc_score(y_test_mul, pre_mul, multi_class='ovo', labels=[0, 1, 2, 3, 4])
    precision = skmetrics.precision_score(y_test_mul, pre_mul, average='macro')
    recall = skmetrics.recall_score(y_test_mul, pre_mul, average='macro')
    F1 = skmetrics.f1_score(y_test_mul, pre_mul, average='macro')
    print('===========================kiba-L2==================================')
    # print(f" Accuracy is:{acc:3f}\n AUC is : {auc:3f}")
    print(f" acc is:{acc:3f}\n AUC is:{auc:3f}\n precision is : {precision:3f}\n recall is:{recall:3f}\n F1 is:{F1:3f}")
    return


def kiba_SVM():
    X_train, X_test, y_train, y_test, y_test_mul = get_kiba()
    svm = SVC(kernel='rbf')
    svm.fit(X_train, y_train)
    pre = svm.predict(X_test)

    pre_mul = []
    for j in range(len(pre)):
        if (pre[j] == 0):
            pre_mul.append([1, 0, 0, 0, 0])
        elif (pre[j] == 1):
            pre_mul.append([0, 1, 0, 0, 0])
        elif (pre[j] == 2):
            pre_mul.append([0, 0, 1, 0, 0])
        elif (pre[j] == 3):
            pre_mul.append([0, 0, 0, 1, 0])
        elif (pre[j] == 4):
            pre_mul.append([0, 0, 0, 0, 1])
    correct = 0
    for d in range(len(pre)):
        if (pre[d] == y_test[d]):
            correct += 1
    acc = correct / len(X_test)
    auc = skmetrics.roc_auc_score(y_test_mul, pre_mul, multi_class='ovo', labels=[0, 1, 2, 3, 4])
    precision = skmetrics.precision_score(y_test_mul, pre_mul, average='macro')
    recall = skmetrics.recall_score(y_test_mul, pre_mul, average='macro')
    F1 = skmetrics.f1_score(y_test_mul, pre_mul, average='macro')
    print('===========================kiba-SVM==================================')
    # print(f" Accuracy is:{acc:3f}\n AUC is : {auc:3f}")
    print(f" acc is:{acc:3f}\n AUC is:{auc:3f}\n precision is : {precision:3f}\n recall is:{recall:3f}\n F1 is:{F1:3f}")
    return


################################【bindingDB dataset】################################
def get_bindingDB():
    # dataset = DataSet()
    # hp = HyperParameter()
    # for sub in ['train', 'dev', 'test']:
    #     print('读取%s数据' % sub)
    #     read_dataset(hp, sub, dataset)
    #     print('处理%s数据' % sub)
    #     process_data(hp, sub, dataset)
    # train_dataset = get_datasets(dataset.train_XD, dataset.train_XT, dataset.train_pos_edge, dataset.train_neg_edge,
    #                              dataset.train_chem, dataset.train_protein)
    # pickle.dump(train_dataset, file=open('other/temp_bindingDB_train.pkl', 'wb'))
    # test_dataset = get_datasets(dataset.test_XD, dataset.test_XT, dataset.test_pos_edge, dataset.test_neg_edge,
    #                             dataset.test_chem, dataset.test_protein)
    # pickle.dump(test_dataset, file=open('other/temp_bindingDB_test.pkl', 'wb'))
    dataset_train = pickle.load(file=open('other/temp_bindingDB_train.pkl', 'rb'))
    dataset_test = pickle.load(file=open('other/temp_bindingDB_test.pkl', 'rb'))
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for d in range(len(dataset_train)):
        X_train.append(np.hstack((dataset_train[d][0], dataset_train[d][1])))
        y_train.append(dataset_train[d][2])
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train)

    for d2 in range(len(dataset_test)):
        X_test.append(np.hstack((dataset_test[d2][0], dataset_test[d2][1])))
        y_test.append(dataset_test[d2][2])
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test
def bindingDB_KNN():
    X_train, X_test, y_train, y_test  = get_bindingDB()
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    pre = clf.predict(X_test)

    auc = skmetrics.roc_auc_score(y_test, pre)
    precision = skmetrics.precision_score(y_test, pre)
    recall = skmetrics.recall_score(y_test, pre)
    F1 = skmetrics.f1_score(y_test, pre)
    print('===========================BindingDB-KNN==================================')
    print(f" AUC is:{auc:3f}\n precision is : {precision:3f}\n recall is:{recall:3f}\n F1 is:{F1:3f}")
    return


def bindingDB_RF():
    X_train, X_test, y_train, y_test = get_bindingDB()
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    pre = rf.predict(X_test)

    auc = skmetrics.roc_auc_score(y_test, pre)
    precision = skmetrics.precision_score(y_test, pre)
    recall = skmetrics.recall_score(y_test, pre)
    F1 = skmetrics.f1_score(y_test, pre)
    print('===========================bindingDB-RF==================================')
    print(f" AUC is:{auc:3f}\n precision is : {precision:3f}\n recall is:{recall:3f}\n F1 is:{F1:3f}")
    return


def bindingDB_SVM():
    X_train, X_test, y_train, y_test = get_bindingDB()
    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    pre = clf.predict(X_test)

    auc = skmetrics.roc_auc_score(y_test, pre)
    precision = skmetrics.precision_score(y_test, pre)
    recall = skmetrics.recall_score(y_test, pre)
    F1 = skmetrics.f1_score(y_test, pre)
    print('===========================bindingDB-SVM==================================')
    print(f" AUC is:{auc:3f}\n precision is : {precision:3f}\n recall is:{recall:3f}\n F1 is:{F1:3f}")
    return


def bindingDB_L2():
    X_train, X_test, y_train, y_test = get_bindingDB()
    r = Ridge(solver='sag')
    r.fit(X_train, y_train)
    pre = r.predict(X_test)
    pre = [1 if i >= 0.5 else 0 for i in pre]

    auc = skmetrics.roc_auc_score(y_test, pre)
    precision = skmetrics.precision_score(y_test, pre)
    recall = skmetrics.recall_score(y_test, pre)
    F1 = skmetrics.f1_score(y_test, pre)
    print('===========================bindingDB_L2==================================')
    print(f" AUC is:{auc:3f}\n precision is : {precision:3f}\n recall is:{recall:3f}\n F1 is:{F1:3f}")
    return


davis_KNN()
davis_RF()
davis_L2()
davis_SVM()


bindingDB_KNN()
bindingDB_RF()
bindingDB_SVM()
bindingDB_L2()


kiba_KNN()
kiba_RF()
kiba_L2()
kiba_SVM()
