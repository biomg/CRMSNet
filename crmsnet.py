#!/usr/bin/python
#-- coding:utf8 --
import sys
import math
import os
import numpy as np
np.set_printoptions(threshold=np.inf)
import pdb

import torch    
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
import random
import gzip
import pickle
import timeit
import argparse
from seq_motifs import get_motif


if torch.cuda.is_available():
        cuda = True
        #torch.cuda.set_device(1)
        print('===> Using GPU')
else:
        cuda = False
        print('===> Using CPU')
#cuda = False        
def padding_sequence_new(seq, window_size = 101, repkey = 'N'):
    seq_len = len(seq)
    new_seq = seq
    if seq_len < window_size:
        gap_len = window_size -seq_len
        new_seq = seq + repkey * gap_len
    return new_seq

def read_rna_dict(rna_dict = 'rna_dict'):
    odr_dict = {}
    with open(rna_dict, 'r') as fp:
        for line in fp:
            values = line.rstrip().split(',')
            for ind, val in enumerate(values):
                val = val.strip()
                odr_dict[val] = ind
    
    return odr_dict

def padding_sequence(seq, max_len = 501, repkey = 'N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq

def get_RNA_seq_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    #for seq in seqs:
    #for key, seq in seqs.iteritems():
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
    return new_array

def split_overlap_seq(seq, window_size):
    overlap_size = 50
    #pdb.set_trace()
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - window_size)/(window_size - overlap_size) + 1
        remain_ins = (seq_len - window_size)%(window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(int(num_ins)):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        seq1 = seq
        pad_seq = padding_sequence_new(seq1, window_size)
        bag_seqs.append(pad_seq)
    else:
        if remain_ins > 10:
            seq1 = seq[-window_size:]
            pad_seq = padding_sequence_new(seq1, window_size)
            bag_seqs.append(pad_seq)
    return bag_seqs

def read_seq_graphprot(seq_file, label = 1):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
                labels.append(label)
    
    return seq_list, labels


def get_bag_data(data, channel = 7, window_size = 101):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        #pdb.set_trace()
        bag_seqs = split_overlap_seq(seq, window_size = window_size)
        #flat_array = []
        bag_subt = []
        for bag_seq in bag_seqs:
            tri_fea = get_RNA_seq_concolutional_array(bag_seq)
            bag_subt.append(tri_fea.T)
        num_of_ins = len(bag_subt)
        if num_of_ins > channel:
            start = (num_of_ins - channel)/2
            bag_subt = bag_subt[start: start + channel]
        if len(bag_subt) <channel:
            rand_more = channel - len(bag_subt)
            for ind in range(rand_more):
                #bag_subt.append(random.choice(bag_subt))
                tri_fea = get_RNA_seq_concolutional_array('N'*window_size)
                bag_subt.append(tri_fea.T)
        bags.append(np.array(bag_subt))

    return bags, labels


def get_bag_data_1_channel(data, max_len = 501):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        #pdb.set_trace()
        #bag_seqs = split_overlap_seq(seq)
        bag_seq = padding_sequence(seq, max_len = max_len)
        #flat_array = []
        bag_subt = []
        #for bag_seq in bag_seqs:
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        # print tri_fea
        bag_subt.append(tri_fea.T)
        # print tri_fea.T
        
        bags.append(np.array(bag_subt))
        # print bags
        
    return bags, labels

def batch(tensor, batch_size = 1000):
    tensor_list = []
    length = tensor.shape[0]
    i = 0
    while True:
        if (i+1) * batch_size >= length:
            tensor_list.append(tensor[i * batch_size: length])
            return tensor_list
        tensor_list.append(tensor[i * batch_size: (i+1) * batch_size])
        i += 1

class Estimator(object):

    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_f = loss

    def _fit(self, train_loader):
        """
        train one epoch
        """
        loss_list = []
        acc_list = []
        for idx, (X, y) in enumerate(train_loader):
            #for X, y in zip(X_train, y_train):
            #X_v = Variable(torch.from_numpy(X.astype(np.float32)))
             #y_v = Variable(torch.from_numpy(np.array(ys)).long())
            X_v = Variable(X)
            y_v = Variable(y)
            if cuda:
                X_v = X_v.cuda()
                y_v = y_v.cuda()
            
            self.optimizer.zero_grad()
            y_pred = self.model(X_v)
            loss = self.loss_f(y_pred, y_v)
            loss.backward()
            self.optimizer.step()

            ## for log
            loss_list.append(loss.item()) # need change to loss_list.append(loss.item()) for pytorch v0.4 or above

        return sum(loss_list) / len(loss_list)

    def fit(self, X, y, batch_size=32, nb_epoch=10, validation_data=()):
        #X_list = batch(X, batch_size)
        #y_list = batch(y, batch_size)
        #pdb.set_trace()
        print (X.shape)
        train_set = TensorDataset(torch.from_numpy(X.astype(np.float32)),
                              torch.from_numpy(y.astype(np.float32)).long().view(-1))
        # val_set = TensorDataset(torch.from_numpy(X_val.astype(np.float32)),
        #                       torch.from_numpy(y_val.astype(np.float32)).long().view(-1))
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
        self.model.train()
        train_loss_print = []
        
        for t in range(nb_epoch):
            loss = self._fit(train_loader)
            train_loss_print.append(loss)
            # print ('%.5f'%loss)
            # print("Epoch %s/%s loss: %06.4f - acc: %06.4f %s" % (t, nb_epoch, loss, acc, val_log))
            print("Epoch %s/%s loss: %06.4f" % (t, nb_epoch, loss))
        return train_loss_print
    def evaluate(self, X, y, batch_size=32):
        
        y_pred = self.predict(X)

        y_v = Variable(torch.from_numpy(y).long(), requires_grad=False)
        if cuda:
            y_v = y_v.cuda()
        loss = self.loss_f(y_pred, y_v)
        predict = y_pred.data.cpu().numpy()[:, 1].flatten()
        auc = roc_auc_score(y, predict)
        #lasses = torch.topk(y_pred, 1)[1].data.numpy().flatten()
        #cc = self._accuracy(classes, y)
        return loss.item(), auc

    def _accuracy(self, y_pred, y):
        return float(sum(y_pred == y)) / y.shape[0]

    def predict(self, X):
        X = Variable(torch.from_numpy(X.astype(np.float32)))
        if cuda:
            X= X.cuda()        
        y_pred = self.model(X)
        return y_pred        

    def predict_proba(self, X):
        self.model.eval()
        return self.model.predict_proba(X)
        

# CRMSNet(Convolution Residual multi-headed self-attention Net)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes , stride, window_size, mhsa = False,  kernel_size = (1,3)):   # 输入频道、输出频道、步长
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=(0,1), bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=(0,1), bias=False)
        # 把第二个卷积层换成mhsa层
        if not mhsa:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=(0,1), bias=False)
        else:
            if stride == 2:
                self.conv2 = nn.Sequential(
                    #MHSA(planes, width=7, height=7),
                    MHSA(planes, width= 14, height=14),
                    # MHSA(planes, width=8, height=8), # for CIFAR10
                    nn.AvgPool2d(2, 2, ceil_mode=True),
                )
            else:
                self.conv2 = nn.Sequential(
                    #MHSA(planes, width=4, height=4),
                    MHSA(planes, width= 1, height = window_size),               # 宽度高度 1*？
                    # MHSA(planes, width=4, height=4), # for CIFAR10
                )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=kernel_size, stride=stride, padding=(0,1), bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # print(x.shape)
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(x))
        # print(out.shape)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CRMSNet (nn.Module):
    def __init__(self, block, num_blocks, nb_filter, channel, labcounts, window_size, pool_size = (1, 3), num_classes=2, hidden_size = 200):
        super(CRMSNet , self).__init__()
        self.in_planes = nb_filter
        self.conv1 = nn.Conv2d(channel, nb_filter, kernel_size=(4,10), stride=1, padding=(0,1), bias=False)
        cnn1_size = window_size - 7
        self.bn1 = nn.BatchNorm2d(nb_filter)
        self.layer1 = self._make_layer(block, nb_filter, num_blocks[0], 1, cnn1_size, True)   # layer输出是64
        self.avg_pool = nn.AvgPool2d(pool_size)
        avgpool2_size = int((cnn1_size - (pool_size[1] - 1) - 1)/pool_size[1] + 1)
        last_layer_size = nb_filter*avgpool2_size*block.expansion
        self.fc = nn.Linear(last_layer_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, window_size, mhsa=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, window_size, mhsa))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 卷积、BN、relu
        out = self.layer1(out)   # transformer, BN, 残差， relu, maxPooling
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        # x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        temp = out.data.cpu().numpy()
        return temp
        
    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        #x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]

class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14):
        super(MHSA, self).__init__()

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, n_dims, height, 1]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims, 1, width]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, C, -1)
        k = self.key(x).view(n_batch, C, -1)
        v = self.value(x).view(n_batch, C, -1)

        content_content = torch.bmm(q.permute(0, 2, 1), k)
        content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
        content_position = torch.matmul(content_position, q)
        energy = content_content + content_position 
        attention = self.softmax(energy)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, width, height)
        return out


# CRT-5(Convolution Residual Transformer Net)
class BasicBlock4(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes , stride, window_size, mhsa = False,  kernel_size = (1,3)):   # 输入频道、输出频道、步长
        super(BasicBlock4, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=(0,1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=(0,1), bias=False)
        # 把第二个卷积层换成mhsa层
        if not mhsa:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=(0,1), bias=False)
        else:
            if stride == 2:
                self.conv2 = nn.Sequential(
                    #MHSA(planes, width=7, height=7),
                    MHSA(planes, width= 14, height=14),
                    # MHSA(planes, width=8, height=8), # for CIFAR10
                    nn.AvgPool2d(2, 2, ceil_mode=True),
                )
            else:
                self.conv2 = nn.Sequential(
                    #MHSA(planes, width=4, height=4),
                    MHSA(planes, width= 1, height = window_size),               # 宽度高度 1*？
                    # MHSA(planes, width=4, height=4), # for CIFAR10
                )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=kernel_size, stride=stride, padding=(0,1), bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # print(out.shape)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CRTNet_5(nn.Module):
    def __init__(self, block, num_blocks, nb_filter, channel, labcounts, window_size, pool_size = (1, 3), num_classes=2, hidden_size = 200):
        super(CRTNet_5, self).__init__()
        self.in_planes = nb_filter
        self.conv1 = nn.Conv2d(channel, nb_filter, kernel_size=(4,10), stride=1, padding=(0,1), bias=False)
        cnn1_size = window_size - 7
        self.bn1 = nn.BatchNorm2d(nb_filter)
        self.layer1 = self._make_layer(block, nb_filter, num_blocks[0], 1, cnn1_size, True)   # layer输出是64
        self.avg_pool = nn.AvgPool2d(pool_size)
        avgpool2_size = int((cnn1_size - (pool_size[1] - 1) - 1)/pool_size[1] + 1)
        last_layer_size = nb_filter*avgpool2_size*block.expansion
        self.fc = nn.Linear(last_layer_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, window_size, mhsa=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, window_size, mhsa))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 卷积、BN、relu
        out = self.layer1(out)   # transformer, BN, 残差， relu, maxPooling
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        # x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        temp = out.data.cpu().numpy()
        return temp
        
    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        #x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]






# CT(Convolution Transformer Net)
class CTNet(nn.Module):
    def __init__(self, nb_filter, channel = 7, num_classes = 2, kernel_size = (4, 10), pool_size = (1, 3), labcounts = 32, window_size = 12, hidden_size = 200, stride = (1, 1), padding = (0,1)):
        super(CTNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        # self.pool1 = nn.MaxPool2d(pool_size, stride = stride, padding=(0,1))
        out1_size = int((window_size + 2*padding[1] - (kernel_size[1] - 1) - 1)/stride[1] + 1)
        # maxpool_size = int((out1_size + 2*padding[1] - (pool_size[1] - 1) - 1)/stride[1] + 1)

        self.layer2 = nn.Sequential(MHSA(nb_filter, width= 1, height = int((window_size)/10) * 10))    # 宽度高度 1*？
        self.bn2 = nn.BatchNorm2d(nb_filter)
        self.pool2 = nn.MaxPool2d(pool_size, stride = stride)
        maxpool_size2 = int((out1_size - (pool_size[1] - 1) - 1)/stride[1] + 1)

        self.drop1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(maxpool_size2*nb_filter, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        # out = self.pool1(out)
        out = F.relu(self.bn2(self.layer2(out)))
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        # x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp
    
    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        # x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]



# T(Transformer Net)
class TNet(nn.Module):
    def __init__(self, nb_filter, channel = 7, num_classes = 2, kernel_size = (4, 10), pool_size = (1, 3), labcounts = 32, window_size = 12, hidden_size = 200, stride = (1, 1), padding = (0,1)):
        super(TNet, self).__init__()
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(channel, nb_filter, kernel_size, stride = stride, padding = padding),
        #     nn.BatchNorm2d(nb_filter),
        #     nn.ReLU())
        # self.pool1 = nn.MaxPool2d(pool_size, stride = stride, padding=(0,1))
        # out1_size = int((window_size + 2*padding[1] - (kernel_size[1] - 1) - 1)/stride[1] + 1)
        # maxpool_size = int((out1_size + 2*padding[1] - (pool_size[1] - 1) - 1)/stride[1] + 1)

        self.layer2 = nn.Sequential(MHSA(nb_filter, width= 1, height = window_size))    # 宽度高度 1*？
        self.bn2 = nn.BatchNorm2d(nb_filter)
        self.pool2 = nn.MaxPool2d(pool_size, stride = stride)
        maxpool_size2 = int((window_size - (pool_size[1] - 1) - 1)/stride[1] + 1)

        self.drop1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(maxpool_size2*nb_filter, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # out = self.layer1(x)
        # out = self.pool1(out)
        out = F.relu(self.bn2(self.layer2(x)))
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        # x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp
    
    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        # x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]


# RT(Residual Transformer Net)
class RTNet(nn.Module):
    def __init__(self, block, num_blocks, nb_filter, channel, labcounts, window_size, pool_size = (1, 3), num_classes=2, hidden_size = 200):
        super(RTNet, self).__init__()
        self.in_planes = nb_filter
        self.conv1 = nn.Conv2d(channel, nb_filter, kernel_size=(1,1), stride=1, padding=(0,0), bias=False)
        # cnn1_size = window_size - 7
        self.bn1 = nn.BatchNorm2d(nb_filter)
        # self.pool1 = nn.MaxPool2d(pool_size, stride = 1, padding=(0,1))
        # maxpool_size = int((cnn1_size + 2 - (pool_size[1] - 1) - 1)/1 + 1)


        self.layer1 = self._make_layer(block, nb_filter, num_blocks[0], 1, window_size, True)   # layer输出是64
        # BN, ReLU
        self.avg_pool = nn.AvgPool2d(pool_size, stride = 1)
        avgpool2_size = int((window_size - (pool_size[1] - 1) - 1)/1 + 1)
        last_layer_size = 4*nb_filter*avgpool2_size*block.expansion

        # self.drop1 = nn.Dropout(p=0.25)
        self.fc = nn.Linear(last_layer_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, window_size, mhsa=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, window_size, mhsa))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 卷积、BN、relu
        # out = self.pool1(out)

        out = self.layer1(out)   # transformer, BN, 残差， relu, maxPooling
        out = self.avg_pool(out)

        out = out.view(out.size(0), -1)

        # out = self.drop1(out)
        out = self.fc(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        # x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        temp = out.data.cpu().numpy()
        return temp
        
    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        #x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes , stride, window_size, mhsa = False,  kernel_size = (1,3)):   # 输入频道、输出频道、步长
        super(BasicBlock2, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=(0,1), bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=(0,1), bias=False)
        # 把第二个卷积层换成mhsa层
        if not mhsa:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=(0,1), bias=False)
        else:
            if stride == 2:
                self.conv2 = nn.Sequential(
                    #MHSA(planes, width=7, height=7),
                    MHSA(planes, width= 14, height=14),
                    # MHSA(planes, width=8, height=8), # for CIFAR10
                    nn.AvgPool2d(2, 2, ceil_mode=True),
                )
            else:
                self.conv2 = nn.Sequential(
                    #MHSA(planes, width=4, height=4),
                    MHSA(planes, width= 1, height = 4*window_size),               # 宽度高度 1*？
                    # MHSA(planes, width=4, height=4), # for CIFAR10
                )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=kernel_size, stride=stride, padding=(0,1), bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # print(x.shape)
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(x))
        # print(out.shape)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# CR(Convolution Residual Net)
class BasicBlock3(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes , stride, window_size, mhsa = False,  kernel_size = (1,3)):   # 输入频道、输出频道、步长
        super(BasicBlock3, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=(0,1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=kernel_size, stride=stride, padding=(0,1), bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CRNet(nn.Module):
    def __init__(self, block, num_blocks, nb_filter, channel, labcounts, window_size, pool_size = (1, 3), num_classes=2, hidden_size = 200):
        super(CRNet, self).__init__()
        self.in_planes = nb_filter
        self.conv1 = nn.Conv2d(channel, nb_filter, kernel_size=(4,10), stride=1, padding=(0,1), bias=False)
        cnn1_size = window_size - 7
        self.bn1 = nn.BatchNorm2d(nb_filter)
        # self.pool1 = nn.MaxPool2d(pool_size, stride = 1, padding=(0,1))
        # maxpool_size = int((cnn1_size + 2 - (pool_size[1] - 1) - 1)/1 + 1)


        self.layer1 = self._make_layer(block, nb_filter, num_blocks[0], 1, window_size, True)   # layer输出是64
        # BN, ReLU
        self.avg_pool = nn.AvgPool2d(pool_size, stride = 1)
        avgpool2_size = int((cnn1_size - (pool_size[1] - 1) - 1)/1 + 1)
        last_layer_size = nb_filter*avgpool2_size*block.expansion

        # self.drop1 = nn.Dropout(p=0.25)
        self.fc = nn.Linear(last_layer_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, window_size, mhsa=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, window_size, mhsa))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 卷积、BN、relu
        # out = self.pool1(out)

        out = self.layer1(out)   # transformer, BN, 残差， relu, maxPooling
        out = self.avg_pool(out)

        out = out.view(out.size(0), -1)

        # out = self.drop1(out)
        out = self.fc(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        # x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        temp = out.data.cpu().numpy()
        return temp
        
    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        #x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]




def read_data_file(posifile, negafile = None, train = True):
    data = dict()
    seqs, labels = read_seq_graphprot(posifile, label = 1)
    if negafile:
        seqs2, labels2 = read_seq_graphprot(negafile, label = 0)
        seqs = seqs + seqs2
        labels = labels + labels2
        # print(labels)
        
    data["seq"] = seqs
    data["Y"] = np.array(labels)
    
    return data

def get_data(posi, nega = None, channel = 7,  window_size = 101, train = True):
    data = read_data_file(posi, nega, train = train)
    if channel == 1:
        train_bags, label = get_bag_data_1_channel(data, max_len = window_size)

    else:
        train_bags, label = get_bag_data(data, channel = channel, window_size = window_size)
    
    return train_bags, label

def detect_motifs(model, test_seqs, X_train, output_dir = 'motifs', channel = 1):
    if channel == 1:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for param in model.parameters():
            layer1_para =  param.data.cpu().numpy()
            break
        	#test_data = load_graphprot_data(protein, train = True)
        	#test_seqs = test_data["seq"]
        N = len(test_seqs)
        if N > 15000: # do need all sequence to generate motifs and avoid out-of-memory
        	sele = 15000
        else:
        	sele = N
        ix_all = np.arange(N)
        np.random.shuffle(ix_all)
        ix_test = ix_all[0:sele]
        
        X_train = X_train[ix_test, :, :, :]
        test_seq = []
        for ind in ix_test:
        	test_seq.append(test_seqs[ind])
        test_seqs = test_seq
        filter_outs = model.layer1out(X_train)[:,:, 0, :]
        get_motif(layer1_para[:,0, :, :], filter_outs, test_seqs, dir1 = output_dir)

def train_network(model_type, X_train, y_train, channel = 7, window_size = 107, model_file = 'model.pkl', batch_size = 100, n_epochs = 50, num_filters = 16, motif = False, motif_seqs = [], motif_outdir = 'motifs'):
    print ('model training for ', model_type)

    if model_type == 'CTNet':
        model = CTNet(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'CRMSNet':
        model = CRMSNet(BasicBlock, [1], nb_filter = num_filters, labcounts = 4, channel = channel , window_size = window_size)
    elif model_type == 'TNet':
        model = CTNet(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'RTNet':
        model = RTNet(BasicBlock2, [1], nb_filter = num_filters, labcounts = 4, channel = channel , window_size = window_size)
    elif model_type == 'CRNet':
        model = CRNet(BasicBlock3, [1], nb_filter = num_filters, labcounts = 4, channel = channel , window_size = window_size)
    elif model_type == 'CRTNet_5':
        model = CRTNet_5(BasicBlock4, [1], nb_filter = num_filters, labcounts = 4, channel = channel , window_size = window_size)
    else:
        print ('only support CRMSNet model')
    if cuda:
        model = model.cuda()
    clf = Estimator(model)
    clf.compile(optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.001),
                loss=nn.CrossEntropyLoss())
    train_loss = clf.fit(X_train, y_train, batch_size=batch_size, nb_epoch=n_epochs)
    if motif and channel == 1:
        detect_motifs(model, motif_seqs, X_train, motif_outdir)
    torch.save(model.state_dict(), model_file)
    #print 'predicting'         
    #pred = model.predict_proba(test_bags)
    return model, train_loss

def predict_network(model_type, X_test, channel = 7, window_size = 107, model_file = 'model.pkl', batch_size = 100, n_epochs = 50, num_filters = 16):
    print ('model training for ', model_type)

    if model_type == 'CTNet':
        model = CTNet(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'CRMSNet':
        model = CRMSNet(BasicBlock, [1], nb_filter = num_filters, labcounts = 4, channel = channel , window_size = window_size)
    elif model_type == 'TNet':
        model = CTNet(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'RTNet':
        model = RTNet(BasicBlock2, [1], nb_filter = num_filters, labcounts = 4, channel = channel , window_size = window_size)
    elif model_type == 'CRNet':
        model = CRNet(BasicBlock3, [1], nb_filter = num_filters, labcounts = 4, channel = channel , window_size = window_size)
    elif model_type == 'CRTNet_5':
        model = CRTNet_5(BasicBlock4, [1], nb_filter = num_filters, labcounts = 4, channel = channel , window_size = window_size)
    else:
        print ('only support CRMSNet model')
    if cuda:
        model = model.cuda()   
    model.load_state_dict(torch.load(model_file))
    try:
        pred = model.predict_proba(X_test)
    except: #to handle the out-of-memory when testing
        test_batch = batch(X_test)
        pred = []
        for test in test_batch:
            pred_test1 = model.predict_proba(test)[:, 1]
            pred = np.concatenate((pred, pred_test1), axis = 0)
    return pred
        
def run(parser):
    #data_dir = './GraphProt_CLIP_sequences/'
    posi = parser.posi
    nega = parser.nega
    motif = parser.motif
    model_type = parser.model_type
    out_file = parser.out_file
    train = parser.train
    model_file = parser.model_file
    predict = parser.predict
    batch_size = parser.batch_size
    n_epochs = parser.n_epochs
    num_filters = parser.num_filters
    testfile = parser.testfile
    motif_outdir = parser.motif_dir
    start_time = timeit.default_timer()
    
    #pdb.set_trace() 
    if predict:
        train = False
        if testfile == '':
            print ('you need specify the fasta file for predicting when predict is True')
            return
    if train:
        if posi == '' or nega == '':
            print ('you need specify the training positive and negative fasta file for training when train is True')
            return

    



    if train:
        file_out = open('time_train.txt','a')
        # print("ResNet-transformer-4")
        motif_seqs = []
        data = read_data_file(posi, nega)
        motif_seqs = data['seq']

        print("301")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel = 2, window_size = 301)
        model, train_loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 301 + 6, model_file = model_file + '.301', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, motif = motif, motif_seqs = motif_seqs, motif_outdir = motif_outdir)
        
       
        end_time = timeit.default_timer()
        file_out.write(str(round(float(end_time - start_time),3))+'\n')
        file_out.close()
        # print ("Training final took: %.2f min" % float((end_time - start_time)/60))
    elif predict:
        fw = open(out_file, 'w')
        file_out = open('pre_auc.txt','a')
        file_out2 = open('time_test.txt', 'a')

        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 301)
        predict = predict_network(model_type, np.array(X_test), channel = 2, window_size = 301 + 6, model_file = model_file+ '.301', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters) 

        predict = predict
        # print predict
	    # pdb.set_trace()
        auc = roc_auc_score(X_labels, predict)
        print ('AUC:{:.3f}'.format(auc))        
        myprob = "\n".join(map(str, predict))  
        fw.write(myprob)
        fw.close()
        file_out.write(str(round(float(auc),3))+'\n')
        file_out.close()
        end_time = timeit.default_timer()
        file_out2.write(str(round(float(end_time - start_time),3))+'\n')
        file_out2.close()
    elif motif:
        motif_seqs = []
        data = read_data_file(posi, nega)
        motif_seqs = data['seq']
        if posi == '' or nega == '':
            print ('To identify motifs, you need training positive and negative sequences using global CNNs.')
        train_bags, train_labels = get_data(posi, nega, channel = 1, window_size = 501)
        model, train_loss = train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 501 + 6, model_file = model_file + '.501', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters, motif = motif, motif_seqs = motif_seqs, motif_outdir = motif_outdir)

    else:
        print ('please specify that you want to train the mdoel or predict for your own sequences')


def parse_arguments(parser):
    parser.add_argument('--posi', type=str, metavar='<postive_sequecne_file>', help='The fasta file of positive training samples')
    parser.add_argument('--nega', type=str, metavar='<negative_sequecne_file>', help='The fasta file of negative training samples')
    parser.add_argument('--model_type', type=str, default='CRT', help='The default model is CRT')
    parser.add_argument('--out_file', type=str, default='prediction.txt', help='The output file used to store the prediction probability of the testing sequences')
    parser.add_argument('--train', type=bool, default=False, help='The path to the Pickled file containing durations between visits of patients. If you are not using duration information, do not use this option')
    parser.add_argument('--model_file', type=str, default='model.pkl', help='The file to save model parameters. Use this option if you want to train on your sequences or predict for your sequences')
    parser.add_argument('--predict', type=bool, default=False,  help='Predicting the RNA-protein binding sites for your input sequences, if using train, then it will be False')
    parser.add_argument('--testfile', type=str, default='',  help='the test fast file for sequences you want to predict for, you need specify it when using predict')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of a single mini-batch (default value: 100)')
    parser.add_argument('--num_filters', type=int, default=16, help='The number of filters for CNNs (default value: 16)')
    parser.add_argument('--n_epochs', type=int, default=50, help='The number of training epochs (default value: 50)')
    parser.add_argument('--motif', type=bool, default=False, help='It is used to identify binding motifs from sequences.')
    parser.add_argument('--motif_dir', type=str, default='motifs', help='The dir used to store the prediction binding motifs.')
    args = parser.parse_args()    #解析添加的参数
    return args

parser = argparse.ArgumentParser()
args = parse_arguments(parser)
print (args)
run(args)



