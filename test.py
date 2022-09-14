import torch
from torch.autograd import Variable
import torch.nn as nn

def focal_loss(pred, target):
    pred = pred.view(-1, 1)
    target = target.view(-1, 1)
    pred = torch.cat((1 - pred, pred), dim=1)
    class_mask = torch.zeros(pred.shape[0], pred.shape[1]).cuda()
    class_mask.scatter_(1, target.view(-1, 1).long(), 1.)
    probs = (pred * class_mask).sum(dim=1).view(-1, 1)
    probs = probs.clamp(min=0.0001, max=1.0)
    log_p = probs.log()
    alpha = torch.ones(pred.shape[0], pred.shape[1]).cuda()
    alpha[:, 0] = alpha[:, 0] * (1 - 0.25)
    alpha[:, 1] = alpha[:, 1] * 0.25
    alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)
    batch_loss = -alpha * (torch.pow((1 - probs), 2)) * log_p
    loss = batch_loss.mean()
    return loss

def mul_focal_loss(pred, target):
    loss = torch.zeros(1, target.shape[1]).cuda()
    for label in range(target.shape[1]):
        batch_loss = focal_loss(pred[:, label], target[:, label])
        loss[0, label] = batch_loss.mean()
    loss = loss.mean()
    return loss

def m_focal_loss(pred, target):
    prob = nn.Softmax(dim=1)(pred.view(-1,pred.size(1)))
    prob = prob.clamp(min=0.0001, max=1.0).cuda()
    # for i in range(pred.shape[0]):
    #     for j in range(pred.shpae[1]):
    #         if target[i][j] == 1:
    target_ = torch.zeros(pred.size(0), pred.size(1)).cuda()
    target_.scatter_(1, target.view(-1, 1).long(), 1.)
    defy_target_ = 1 - target_
    p_loss = - 0.25 * torch.pow(1 - prob,2) * prob.log() * target_
    n_loss = - 0.75 * torch.pow(prob,2) * (1 - prob).log() * defy_target_
    loss = p_loss + n_loss
    loss = loss.mean()
    return loss

import  numpy as np
if __name__ == '__main__':
    # torch.manual_seed(50)  # 随机种子确保每次input tensor值是一样的
    # input = torch.randn(4, 3, dtype=torch.float32)
    # targets = torch.randint(3, (4,))
    # input = Variable(input).cuda()
    # targets = Variable(targets).cuda()
    # loss = m_focal_loss(input, targets)

    act = ['a 55 012', 'ab 556 0123', 'ac 552 0122']
    actives = [[x.split(' ')[0], 1] for x in act]
    print(actives)
