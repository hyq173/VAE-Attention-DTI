import numpy as np
from sklearn import metrics as skmetrics

def deal_data(Y_pre, Y):
    Y_S, Y_pre_S = [], []
    for line in Y:
        if line.any():
            b = [i for i, e in enumerate(line) if e != 0]
            Y_S.extend(b)
    for line in Y_pre:
        if line.any():
            ind = line.tolist().index(max(line.tolist()))
            Y_pre_S.append(ind)
    return np.array(Y_pre_S), np.array(Y_S)
'''
最后的维度是1,sigmod
'''
def ACC_1(Y_pre, Y):
    Y_pre = [1 if i >= 0.5 else 0 for i in Y_pre]
    return (Y_pre==Y).sum() / len(Y)

def AUC_1(Y_pre, Y):
    return skmetrics.roc_auc_score(Y, Y_pre)

def Recall_1(Y_pre, Y):
    Y_pre = [1 if i >= 0.5 else 0 for i in Y_pre]
    return skmetrics.recall_score(Y, Y_pre)

def Precision_1(Y_pre, Y):
    Y_pre = [1 if i >= 0.5 else 0 for i in Y_pre]
    return skmetrics.precision_score(Y, Y_pre)

def F1_1(Y_pre, Y):
    Y_pre = [1 if i >= 0.5 else 0 for i in Y_pre]
    return skmetrics.f1_score(Y, Y_pre)

def get_all_metrics_1(Y_pre, Y):
    acc = ACC_1(Y_pre, Y)
    auc = AUC_1(Y_pre, Y)
    precision = Precision_1(Y_pre, Y)
    recall = Recall_1(Y_pre, Y)
    f1 = F1_1(Y_pre, Y)
    return acc, auc, precision, recall, f1
'''
最后的维度是2,softmax
'''
def ACC_2(Y_pre, Y):
    Y_pre, Y = deal_data(Y_pre, Y)
    return (Y_pre==Y).sum() / len(Y)

def AUC_2(Y_pre, Y):
    y_pre = []
    y = []
    for line in Y:
        y.extend([i for i, e in enumerate(line) if e != 0])
    for line in Y_pre:
        y_pre.append(max(line))
    return skmetrics.roc_auc_score(y, y_pre)

def Precision_2(Y_pre, Y):
    Y_pre, Y = deal_data(Y_pre, Y)
    return skmetrics.precision_score(Y, Y_pre)

def Recall_2(Y_pre, Y):
    Y_pre, Y = deal_data(Y_pre, Y)
    return skmetrics.recall_score(Y, Y_pre)

def F1_2(Y_pre, Y):
    Y_pre, Y = deal_data(Y_pre, Y)
    return skmetrics.f1_score(Y, Y_pre)

def get_all_metrics_2(Y_pre, Y):
    acc = ACC_2(Y_pre, Y)
    auc = AUC_2(Y_pre, Y)
    precision = Precision_2(Y_pre, Y)
    recall = Recall_2(Y_pre, Y)
    f1 = F1_2(Y_pre, Y)
    return acc, auc, precision, recall, f1
'''
多分类
'''
def mul_ACC(Y_pre, Y):
    Y_pre, Y = deal_data(Y_pre, Y)
    acc = (Y_pre == Y).sum() / len(Y)
    return acc

def mul_AUC(Y_pre, Y, hp):
    y = []
    for line in Y:
        y.extend([i for i, e in enumerate(line) if e != 0])
    return skmetrics.roc_auc_score(y, Y_pre, multi_class='ovo', labels=hp.davis_labels)

def mul_Precision(Y_pre, Y, hp):
    Y_pre, Y = deal_data(Y_pre, Y)
    return skmetrics.precision_score(Y, Y_pre, average='macro')

def mul_Recall(Y_pre, Y, hp):
    Y_pre, Y = deal_data(Y_pre, Y)
    return skmetrics.recall_score(Y, Y_pre, average='macro')

def mul_F1(Y_pre, Y, hp):
    Y_pre, Y = deal_data(Y_pre, Y)
    return skmetrics.f1_score(Y, Y_pre, average='macro')

def get_all_mul_metrics(Y_pre, Y, hp):
    acc = mul_ACC(Y_pre, Y)
    auc = mul_AUC(Y_pre, Y, hp)
    precision = mul_Precision(Y_pre, Y, hp)
    recall = mul_Recall(Y_pre, Y, hp)
    f1 = mul_F1(Y_pre,Y, hp)
    return acc, auc, precision, recall, f1