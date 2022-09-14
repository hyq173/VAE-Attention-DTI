import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self, num_filters, k_size, att=100):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # nn.Sequential 一个 有序 的容器；把特定神经网络模块按照在传入构造器的顺序依次被添加到计算图中
            # Number of filters in encoder ：32*1;32*2;32*3
            nn.Conv1d(in_channels=128, out_channels=num_filters * 2, kernel_size=k_size, stride=1, padding=k_size // 2),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(0.2), # 0.2->0.5
            nn.Conv1d(num_filters, num_filters * 4, k_size, 1, k_size // 2),
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(0.2), # 0.2->0.5
            nn.Conv1d(num_filters * 2, num_filters * 6, k_size, 1, k_size // 2),
        )
        self.out = nn.AdaptiveAvgPool1d(1)  # 自适应均值池化层
        self.layer1 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
    # 重参数技巧
    def reparametrize(self, mean, logvar):
        # 所有带"_"都是inplace的 意思就是操作后 原数也会改动
        # 不带 "_" 的 只在操作适时候改变数据，操作结束后数据变回原状
        std = logvar.mul(0.5).exp_()  # mul:logvar中的每一个元素/2 ; std标准差
        eps = torch.cuda.FloatTensor(std.size()).normal_(0, 0.1)  # 产生正态分布
        eps = Variable(eps)
        return eps.mul(std).add_(mean)  # z = u + δ * α；1. 【*】是每个元素对应相乘2. α 服从标准正态分布
    def forward(self, x):
        x = self.conv1(x)  # a = np.array([[1,2,3],[4,5,6]]) ==》np.size(a)==》6，若np.size(a,1)==》3； 其中axis=0/1按行/列统计个数
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)  # 门控卷积 注意 * 符号，看论文

        x = self.conv2(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)

        x = self.conv3(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)

        output = self.out(x)  # 调用了上面的方法：nn.AdaptiveAvgPool1d(1)  # 自适应均值池化层
        output = output.squeeze()  # 移除数组中维度为1的维度 例如：(1,3,1)==>(3,)

        output1 = self.layer1(output)  # 第一个 full-connect 得到 均值u
        output2 = self.layer2(output)  # 第二个 full-connect 得到 对数的方差 log δ^2
        output = self.reparametrize(output1, output2)  # 重参数技巧：z = u + δ * α
        return output, output1, output2
class CNN_my(nn.Module):
    def __init__(self, num_filters, k_size, att=100):
        super(CNN_my, self).__init__()
        self.conv1 = nn.Sequential(  # nn.Sequential 一个 有序 的容器；把特定神经网络模块按照在传入构造器的顺序依次被添加到计算图中
            # Number of filters in encoder ：32*1;32*2;32*3
            nn.Conv1d(in_channels=128, out_channels=num_filters * 2, kernel_size=k_size, stride=1, padding=k_size // 2),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(0.2), # 0.2->0.5
            nn.Conv1d(num_filters, num_filters * 4, k_size, 1, k_size // 2),
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(0.2), # 0.2->0.5
            nn.Conv1d(num_filters * 2, num_filters * 6, k_size, 1, k_size // 2),
        )
        self.out = nn.AdaptiveAvgPool1d(1)  # 自适应均值池化层
        self.layer1 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
    # 重参数技巧
    def reparametrize(self, mean, logvar):
        # 所有带"_"都是inplace的 意思就是操作后 原数也会改动
        # 不带 "_" 的 只在操作适时候改变数据，操作结束后数据变回原状
        std = logvar.mul(0.5).exp_()  # mul:logvar中的每一个元素/2 ; std标准差
        eps = torch.cuda.FloatTensor(std.size()).normal_(0, 0.1)  # 产生正态分布
        eps = Variable(eps)
        return eps.mul(std).add_(mean)  # z = u + δ * α；1. 【*】是每个元素对应相乘2. α 服从标准正态分布
    def forward(self, x):
        x = self.conv1(x)  # a = np.array([[1,2,3],[4,5,6]]) ==》np.size(a)==》6，若np.size(a,1)==》3； 其中axis=0/1按行/列统计个数
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)  # 门控卷积 注意 * 符号，看论文

        x = self.conv2(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)

        x = self.conv3(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)

        output = self.out(x)  # 调用了上面的方法：nn.AdaptiveAvgPool1d(1)  # 自适应均值池化层
        output = output.squeeze()  # 移除数组中维度为1的维度 例如：(1,3,1)==>(3,)

        output1 = self.layer1(output)  # 第一个 full-connect 得到 均值u
        output2 = self.layer2(output)  # 第二个 full-connect 得到 对数的方差 log δ^2
        output = self.reparametrize(output1, output2)  # 重参数技巧：z = u + δ * α
        return output, output1, output2

class decoder(nn.Module):
    def __init__(self, init_dim, num_filters, k_size,size):
        super(decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3 * (init_dim - 3 * (k_size - 1))),
            nn.ReLU()
        )
        self.convt = nn.Sequential(
            nn.ConvTranspose1d(num_filters * 3, num_filters * 2, k_size, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters * 2, num_filters, k_size, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters, 128, k_size, 1, 0),
            nn.ReLU(),
        )
        self.layer2 = nn.Linear(128, size)
    def forward(self, x, init_dim, num_filters, k_size):
        x = self.layer(x)
        x = x.view(-1, num_filters * 3, init_dim - 3 * (k_size - 1))
        x = self.convt(x)
        x = x.permute(0,2,1)
        x = self.layer2(x)
        return x

class net(nn.Module):
    def __init__(self, hp, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
        super(net, self).__init__()
        self.embedding1 = nn.Embedding(hp.charsmiset_size, 128)  # FLAGS.charsmiset_size = 64  drug-smiles
        self.embedding2 = nn.Embedding(hp.charseqset_size, 128)  # FLAGS.charseqset_size= 25   protein
        self.cnn1 = CNN(NUM_FILTERS, FILTER_LENGTH1, att=100)  # CNN(num_filters, kernel_size)
        self.cnn2 = CNN(NUM_FILTERS, FILTER_LENGTH2, att=1000)  # CNN(num_filters, kernel_size)
        self.reg = net_reg_my(NUM_FILTERS, hp)
        self.decoder1 = decoder(hp.max_smi_len, NUM_FILTERS, FILTER_LENGTH1, hp.charsmiset_size)
        self.decoder2 = decoder(hp.max_seq_len, NUM_FILTERS, FILTER_LENGTH2, hp.charseqset_size)
    def forward(self, x, y, hp, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
        x_init = Variable(x.long()).cuda()  # 对模型和相应的数据进行.cuda()处理;就可以将内存中的数据复制到GPU的显存中去
        x = self.embedding1(x_init)
        x_embedding = x.permute(0, 2, 1)  # # 将tensor的维度换位
        y_init = Variable(y.long()).cuda()
        y = self.embedding2(y_init)
        y_embedding = y.permute(0, 2, 1)
        x, mu_x, logvar_x = self.cnn1(x_embedding)
        y, mu_y, logvar_y = self.cnn2(y_embedding)

        out = self.reg(x, y, hp).squeeze()

        x = self.decoder1(x, hp.max_smi_len, NUM_FILTERS, FILTER_LENGTH1)
        y = self.decoder2(y, hp.max_seq_len, NUM_FILTERS, FILTER_LENGTH2)
        return out, x, y, x_init, y_init, mu_x, logvar_x, mu_y, logvar_y

class net_reg(nn.Module):
    def __init__(self, num_filters):
        super(net_reg, self).__init__()
        self.reg = nn.Sequential(
            nn.Linear(num_filters * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.1), # 0.1->0.5
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1), # # 0.1->0.5
            nn.Linear(512, 1)
        )
        self.reg1 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
        self.reg2 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
    def forward(self, A, B):
        A = self.reg1(A)
        B = self.reg2(B)
        x = torch.cat((A, B), 1)  # 将两个张量拼接，默认是0：行拼接
        x = self.reg(x)
        return x

class net_reg_my(nn.Module):
    def __init__(self, num_filters, hp):
        super(net_reg_my, self).__init__()
        self.atten = selfattention(num_filters * 3)
        self.sig = nn.Sigmoid()
        self.sof = nn.Softmax()
        # self.lstm = lstm_p(num_filters * 3, 32, num_filters * 3, 1)
        self.reg = nn.Sequential(
            nn.Linear(num_filters * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.1), # # 0.1->0.5
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1), # # 0.1->0.5
            # nn.Linear(512, 1)
            nn.Linear(512, hp.classify)
        )
        self.reg1 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
        self.reg2 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
    def forward(self, A, B, hp):
        A = self.reg1(A)
        B = self.reg2(B)
        A = self.atten(A)
        B = self.atten(B)
        # A = self.reg1(A)
        # B = self.reg2(B)
        x = torch.cat((A, B), 1)  # 将两个张量拼接，默认是0：行拼接
        x = self.reg(x)
        np.savetxt('x.txt',np.array(x.cpu().detach().numpy()))
        if(hp.classify == 1):
            x = self.sig(x)
        elif(hp.classify == 2):
            x = self.sof(x)
        else:
            x = self.sof(x)
        return x

class selfattention(nn.Module):
    def __init__(self, in_channels):
        super(selfattention, self).__init__()
        self.in_channels = in_channels

        self.w_query = nn.Linear(in_features=in_channels, out_features=in_channels, bias=False)
        self.w_key = nn.Linear(in_features=in_channels, out_features=in_channels, bias=False)
        self.w_value = nn.Linear(in_features=in_channels, out_features=in_channels, bias=False)

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, input):
        q = self.w_query(input)
        k = self.w_key(input)
        v = self.w_value(input)

        attention_matrix = torch.matmul(q, k.transpose(0, 1))
        attention_matrix = self.softmax(attention_matrix / (self.in_channels ** 0.5))

        out = torch.matmul(attention_matrix, v)
        return out

class lstm_p(nn.Module):
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super(lstm_p, self).__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)
        return x