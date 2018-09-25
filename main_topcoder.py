#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/10/18 4:48 PM
# @Author  : renxiaoming@julive.com
# @Site    : 
# @File    : main.py.py
# @Software: PyCharm

import torch
import yaml
import glob
import time
import PIL.Image as Image
import torch.utils.data as data
import Levenshtein
#from torchviz import make_dot
import os
if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import argparse
from logger import Logger

parser = argparse.ArgumentParser()
#parser.add_argument('--root_dir', type=str,default='/Users/comjia/Downloads/code/pytorch_seq2seq/pytorch_seq2seq',help='??aishell????????path')
parser.add_argument("--mode", type=str, default='train',help="train | dev | test")
parser.add_argument("--datalist_path", type=str, default="/movie/audio/topcoder",help="声谱图列表文件所在路径")
parser.add_argument("--use_gpu", type=int, default=0,help="use gpu = 1; ")
parser.add_argument("--use_pretrained", type=int, default=0,help="use_pretrained = 1; ")
parser.add_argument("--model_path", type=str, default="checkpoint/spoken_Lang_id_topcoder",help="model path ")
parser.add_argument("--img_path", type=str, default="",help="用于预测的声谱图路径 ")

opt = parser.parse_args()
print opt

class batch_gen_imgdata(data.Dataset):
    def __init__(self,data_listfile):
        # self.root = os.path.expanduser(root)
        # self.processed_imglist = glob.glob(os.path.join(data_path,"*.png"))
        with open(data_listfile, "r") as f:
            self.labels = f.readlines()
        self.num_samples = len(self.labels)

    def __getitem__(self, index):
        img_path_label = self.labels[index]
        #print img_path_label
        img_split = img_path_label.split(",")
        label = int(img_split[1][:-1])#kind_map[img_path.split("/")[-1].split("_")[0]]
        img = Image.open(img_split[0])
        img = np.array(img)
        return img if img.shape[1] == 154 else img[:,:-1],label,img_split[0]

    def __len__(self):
        return self.num_samples

    def collate_fn(self,batch):
        # batch.sort(key=lambda x: len(x[1]), reverse=True)
        imgs, labels, img_paths = zip(*batch)

        #64 * 256 * 858
        batch_imgs = np.array(imgs)
        # 64 * 1
        batch_labels = np.array(labels)[:,np.newaxis]
        return batch_imgs,batch_labels

class Network_CNN_RNN(nn.Module):
    def __init__(self,rnn_input_size,rnn_hidden_size,rnn_num_layers,use_gpu =False):
        super(Network_CNN_RNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters
                kernel_size=7,      # filter size
                stride=1,           # filter movement/step
                padding=0,      # , padding=(kernel_size-1)/2 ? stride=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=(3,3),stride=(2,2),padding=1),    # 16 * 125 *212
        )
        self.conv2 = nn.Sequential(  # input shape (16, 125, 212)
            nn.Conv2d(16, 32, 5, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # activation
            nn.MaxPool2d(3,2,1),  # output shape (32, 61 ,104)
        )
        self.conv3 = nn.Sequential(  # input shape (32, 61 ,104)
            nn.Conv2d(32, 32, 3, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # activation
            nn.MaxPool2d(3,2,1),  # output shape (32,30,51)
        )
        self.conv4 = nn.Sequential(  # input shape (32,30,51)
            nn.Conv2d(32, 32, 3, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # activation
            nn.MaxPool2d(3,2),  # output shape (32, 13, 24)
        )
        self.gru = nn.GRU(input_size=rnn_input_size, hidden_size=rnn_hidden_size,
                          num_layers = rnn_num_layers,batch_first=True)
        #self.batch_norm = nn.BatchNorm2d()
        self.out = nn.Linear(rnn_hidden_size, lang_num)


        self.use_gpu = use_gpu
        if self.use_gpu:
            self = self.cuda()

    def forward(self, x):
        #x: 100 * 1 * 256 * 858
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # input: batchsize * 32 * 13 *51
        x = x.view(x.size(0),-1,x.size(-1))
        x = x.transpose(1,2)
        gru_output,_ = self.gru(x)
        gru_output = F.dropout(gru_output, training=self.training)
        # gru_output = self.batch_norm(gru_output)
        output = self.out(gru_output[:,-1,:])
        #output = F.log_softmax(output, dim=1)
        return output

def train(network):
    best_model_acc = 0.0
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01,weight_decay=1e-5)
    scheduler_ = MultiStepLR(optimizer, milestones=[5,10,100], gamma=0.1)
    objective = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        network.train()
        tr_loss = 0.0
        tr_acc = 0.0
        scheduler_.step()
        for batch_index, (batch_imgs,batch_labels) in enumerate(train_set):
            batch_imgs = Variable(torch.FloatTensor(batch_imgs))
            batch_labels = Variable(torch.LongTensor(batch_labels))

            if use_gpu:
                batch_imgs = batch_imgs.cuda()
                batch_labels = batch_labels.cuda()


            pred = network(batch_imgs.unsqueeze(1))

            loss = objective(input = pred,target = batch_labels.squeeze(1))

            optimizer.zero_grad()
            loss.backward() # backpropagation, compute gradients
            optimizer.step() # apply gradients
            batch_loss = loss.cpu().data.numpy() #/ batch_labels.shape[0]

            _, keys = torch.topk(pred, 1)
            pre = keys.cpu().data.numpy().T.tolist()[0]
            tar = np.array(batch_labels).reshape([-1])
            batch_acc = np.mean(np.equal(pre, tar))

            tr_loss += batch_loss
            tr_acc += batch_acc

            # ========================= Log ======================
            step = epoch * imgdata_train.num_samples + batch_index
            # (1) Log the scalar values
            info = {'loss': batch_loss, 'accuracy': batch_acc}

            for tag, value in info.items():
                logger_train.scalar_summary(tag, value, step)


            if (batch_index+1) % verbose_step == 0:
                print(
                training_msg.format(time.asctime(), epoch + 1, batch_index + 1,
                                    tr_loss / verbose_step, tr_acc / verbose_step))
                tr_loss = 0.0
                tr_acc = 0.0

        if epoch % per_epoch_save_model == 0:
            dev_acc = dev(network, epoch=epoch)
            if dev_acc > best_model_acc:
                torch.save(net.state_dict(), model_path)

def dev(network,epoch = 0):
    network.eval()
    sum_acc = 0.0
    for batch_index, (batch_imgs,batch_labels) in enumerate(test_set):
        batch_imgs = Variable(torch.FloatTensor(batch_imgs))
        batch_labels = Variable(torch.LongTensor(batch_labels))

        if use_gpu:
            batch_imgs = batch_imgs.cuda()
            batch_labels = batch_labels.cuda()

        pred = network(batch_imgs.unsqueeze(1))
        _, keys = torch.topk(pred, 1)
        pre = keys.cpu().data.numpy().T.tolist()[0]
        tar = np.array(batch_labels).reshape([-1])
        batch_acc = np.mean(np.equal(pre, tar))
        sum_acc += batch_acc * len(batch_labels)
        print("step_{:3d}_acc_{:.4f}".format(batch_index + 1,batch_acc))

        # ========================= Log ======================
        step = epoch * imgdata_test.num_samples + batch_index
        # (1) Log the scalar values
        info = { 'accuracy': batch_acc}

        for tag, value in info.items():
            logger_dev.scalar_summary(tag, value, step)
    sum_acc = sum_acc/imgdata_test.num_samples
    print "sum_acc = %.4f"%(sum_acc)
    return sum_acc

def test():
    None

def predict(network,img_path):
    network.eval()

    batch_imgs = Image.open(img_path)
    batch_imgs = Variable(torch.FloatTensor(batch_imgs))
    if use_gpu:
        batch_imgs = batch_imgs.cuda()

    pred = network(batch_imgs.unsqueeze(1))
    _, keys = torch.topk(pred, 1)
    pre = keys.cpu().data.numpy().T.tolist()[0]
    print pre


if __name__ == '__main__':

    use_gpu = True if 1 == opt.use_gpu else False
    use_pretrained = True if opt.use_pretrained == 1 else False
    model_path = opt.model_path
    lang_num = 179
    rnn_hidden_size = 500
    rnn_num_layers = 1
    rnn_input_size = 416#448
    num_epochs = 100
    verbose_step = 10
    per_epoch_save_model = 1

    logger_train = Logger('./log_topcoder/logs_train')
    logger_dev = Logger('./log_topcoder/logs_dev')
    net = Network_CNN_RNN(rnn_input_size = rnn_input_size,rnn_hidden_size = rnn_hidden_size,rnn_num_layers = rnn_num_layers,use_gpu=use_gpu)
    print net
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("all params num:" + str(k))


    if use_pretrained:
        net.load_state_dict(torch.load(model_path))


    training_msg = 'time_{}_epoch_{:2d}_step_{:3d}_TrLoss_{:.4f}_acc_{:.4f}'
    imgdata_train = batch_gen_imgdata(data_listfile=os.path.join(opt.datalist_path,"trainEqual.csv"))
    train_set = torch.utils.data.DataLoader(imgdata_train,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=1,
                                              collate_fn=imgdata_train.collate_fn)

    imgdata_test = batch_gen_imgdata(data_listfile=os.path.join(opt.datalist_path,"valEqaul.csv"))
    test_set = torch.utils.data.DataLoader(imgdata_test,
                                              batch_size=64,
                                              shuffle=False,
                                              num_workers=1,
                                              collate_fn=imgdata_test.collate_fn)

    if opt.mode == "train":
        train(net)
    elif opt.mode == "dev":
        dev(net)
    elif opt.mode == "test":
        test()
    elif opt.mode == "predict":
        predict(net,opt.img_path)
