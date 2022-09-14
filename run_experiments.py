## This program is based on the DeepDTA model(https://github.com/hkmztrk/DeepDTA)
import datetime
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.init as init
import os
from tqdm import tqdm
from model import net
from datahelper import *
from hyperparameter import *
from metrics import *
from tensorboardX import SummaryWriter
os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
warnings.filterwarnings("ignore")

def loss_f(recon_x, x, mu, logvar):
    cit = nn.CrossEntropyLoss(reduction='none')
    cr_loss = torch.sum(cit(recon_x.permute(0, 2, 1), x), 1)  # reconstruct loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)  # KL loss
    return torch.mean(cr_loss + KLD)

def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
        if isinstance(m, nn.BatchNorm1d):  # 批量norm
            init.constant_(m.weight.data, 1)
            init.constant_(m.bias.data, 0)
        if isinstance(m, nn.LSTM):
            init.orthogonal_(m.all_weights[0][0])
            init.orthogonal_(m.all_weights[0][1])
        if isinstance(m, nn.Conv1d):
            init.xavier_normal_(m.weight.data)  # 参数初始化
            m.bias.data.fill_(0)  # bias置为0

def m_focal_loss(pred, target):
    prob = pred.clamp(min=0.0001, max=1.0).cuda()
    # target_ = torch.zeros(pred.size(0), pred.size(1)).cuda()
    # target_.scatter_(1, target.view(-1, 1).long(), 1.)
    target_ = target
    defy_target_ = 1 - target_
    p_loss = - 0.25 * torch.pow(1 - prob,2) * prob.log() * target_
    n_loss = - 0.75 * torch.pow(prob,2) * (1 - prob).log() * defy_target_
    loss = p_loss + n_loss
    loss = loss.mean()
    return loss

def train(train_loader, model, hp, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2, lamda, epochind, writer):
    model.train()
    # loss_func = nn.MSELoss()
    if hp.classify == 1: loss_func = nn.BCEWithLogitsLoss()
    # loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    epoch_acc, epoch_auc, epoch_loss = [], [], []
    with tqdm(train_loader) as t:
        for drug_SMILES, target_protein, affinity in t:
            drug_SMILES = torch.Tensor(drug_SMILES)
            target_protein = torch.Tensor(target_protein)
            affinity = torch.Tensor(affinity)
            optimizer.zero_grad()  # optimizer.zero_grad()意思是把梯度置零

            affinity = Variable(affinity).cuda()
            pre_affinity, new_drug, new_target, drug, target, mu_drug, logvar_drug, mu_target, logvar_target = model(
                drug_SMILES, target_protein, hp, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2)
            np.savetxt('pre_affinity.txt',np.array(pre_affinity.cpu().detach().numpy()))
            if hp.classify == 1:
                loss_affinity = loss_func(pre_affinity, affinity)  # 分类误差
            else:
                loss_affinity = m_focal_loss(pre_affinity, affinity)
            loss_drug = loss_f(new_drug, drug, mu_drug, logvar_drug)  # drug的重构误差
            loss_target = loss_f(new_target, target, mu_target, logvar_target)  # target的重构误差 交叉熵Loss+KLD散度
            loss = loss_affinity + 10 ** lamda * (loss_drug + hp.max_smi_len / hp.max_seq_len * loss_target)
            epoch_loss.append(loss.cpu().detach().numpy())
            if hp.classify == 1:
                acc = ACC_1(pre_affinity.cpu().detach().numpy(), affinity.cpu().detach().numpy())
                auc = AUC_1(pre_affinity.cpu().detach().numpy(), affinity.cpu().detach().numpy())
            elif hp.classify == 2:
                acc = ACC_2(pre_affinity.cpu().detach().numpy(), affinity.cpu().detach().numpy())
                auc = AUC_2(pre_affinity.cpu().detach().numpy(), affinity.cpu().detach().numpy())
            else:
                acc = mul_ACC(pre_affinity.cpu().detach().numpy(), affinity.cpu().detach().numpy())
                auc = mul_AUC(pre_affinity.cpu().detach().numpy(), affinity.cpu().detach().numpy(), hp)
            epoch_acc.append(acc)
            epoch_auc.append(auc)
            loss.backward()  # 反向传播，计算当前梯度
            optimizer.step()  # 根据梯度更新网络参数
            bi_classify_loss = loss_affinity.item()
            t.set_postfix(train_total_loss=loss.item(), bi_classify_loss=bi_classify_loss, epochind=epochind, lr=optimizer.param_groups[0]['lr'])
        visual_log_train(writer, np.mean(np.array(epoch_acc)), np.mean(np.array(epoch_auc)), np.mean(np.array(epoch_loss)), epochind)
        save_train_loss(hp, np.mean((epoch_acc)), np.mean((epoch_auc)), np.mean((epoch_loss)) )
    return model

def test(model, valid_loader, hp, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2, save_result):
    #  model.eval()  ：不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化，
    # pytorch框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层影响结果。
    model.eval()
    affinities = []
    pre_affinities = []
    with torch.no_grad():
        for i, (drug_SMILES, target_protein, affinity) in enumerate(valid_loader):
            pre_affinity, new_drug, new_target, drug, target, mu_drug, logvar_drug, mu_target, logvar_target = model(
                drug_SMILES, target_protein, hp, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2)
            pre_affinities += pre_affinity.cpu().detach().numpy().tolist()
            affinities += affinity.cpu().detach().numpy().tolist()
        pre_affinities = np.array(pre_affinities)
        affinities = np.array(affinities)
        if save_result:
            np.savetxt(hp.save_affinities, affinities)
            np.savetxt(hp.save_preaffinities, pre_affinities)
        if hp.classify == 1:
            acc, auc, precision, recall, f1 = get_all_metrics_1(pre_affinities, affinities)
        elif hp.classify == 2:
            acc, auc, precision, recall, f1 = get_all_metrics_2(pre_affinities, affinities)
        else:
            acc, auc, precision, recall, f1 = get_all_mul_metrics(pre_affinities, affinities, hp)
    return acc, auc, precision, recall, f1

def visual_log_train(writer, acc, auc, loss, epochind):
    writer.add_scalar('train ACC ', acc, epochind)
    writer.add_scalar('train AUC ', auc, epochind)
    writer.add_scalar('train loss', loss, epochind)

def my_train(hp, dataset):
    writer = SummaryWriter(logdir=hp.visual_log)
    model = net(hp, hp.num_windows, hp.smi_window_lengths, hp.seq_window_lengths).cuda()
    model.apply(weights_init)
    best_acc, stopSteps = 0.0, 0
    for epochind in range(hp.num_epoch):
        model = train(dataset.train_loader, model, hp, hp.num_windows, hp.smi_window_lengths, hp.seq_window_lengths, hp.lamda, epochind, writer)
        hp.logging(f'========== Epoch:{epochind + 1:5d} ==========')
        acc, auc, precision, recall, f1 = test(model, dataset.train_loader, hp, hp.num_windows, hp.smi_window_lengths, hp.seq_window_lengths, False)
        hp.logging(f'[Total Train] ACC={acc:5f}; AUC={auc:5f}; precision={precision:5f}; recall={recall:5f}, f1={f1:5f}')
        acc, auc, precision, recall, f1 = test(model, dataset.valid_loader, hp, hp.num_windows, hp.smi_window_lengths, hp.seq_window_lengths, False)
        hp.logging(f'[Total valid] ACC={acc:5f}; AUC={auc:5f}; precision={precision:5f}; recall={recall:5f}, f1={f1:5f}')
        hp.logging('=================================')
        if(acc > best_acc):
            hp.logging(f'Get a better Model with val ACC: {acc:.5f}!!!')
            best_acc = acc
            torch.save(model, hp.checkpoint_path)
            stopSteps = 0
        else:
            stopSteps += 1
            if stopSteps >= hp.earlyStop:
                hp.logging(f'The val AUC has not improved for more than {hp.earlyStop} steps in epoch {epochind + 1}, stop training.')
                break
    model = torch.load(hp.checkpoint_path)
    hp.logging(f'============ Result ============')
    acc, auc, precision, recall, f1 = test(model, dataset.train_loader, hp, hp.num_windows, hp.smi_window_lengths, hp.seq_window_lengths, False)
    hp.logging(f'[Total Train] ACC={acc:5f}; AUC={auc:5f}; Precision={precision:5f}; Recall={recall:5f}; f1={f1:5f}')
    acc, auc, precision, recall, f1 = test(model, dataset.valid_loader, hp, hp.num_windows, hp.smi_window_lengths, hp.seq_window_lengths, False)
    hp.logging(f'[Total valid] ACC={acc:5f}; AUC={auc:5f}; Precision={precision:5f}; Recall={recall:5f}; f1={f1:5f}')
    acc, auc, precision, recall, f1 = test(model, dataset.test_loader, hp, hp.num_windows, hp.smi_window_lengths, hp.seq_window_lengths, True)
    hp.logging(f'[Total test ] ACC={acc:5f}; AUC={auc:5f}; Precision={precision:5f}; Recall={recall:5f}; f1={f1:5f}')
    hp.logging(f'============= Finish ===================\n\n\n\n\n')

if __name__ == "__main__":
    hp = HyperParameter()
    hp.logging(str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    dataset = pickle.load(file=open(hp.pkl_path,'rb'))
    print(f'{hp.sub}-读取结束')
    print('开始训练')
    my_train(hp, dataset)