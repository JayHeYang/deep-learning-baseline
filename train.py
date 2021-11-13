import torch.nn as nn
import torch as t
from tqdm._tqdm import trange
from torch.optim.lr_scheduler import StepLR
from torch.utils import data

from dataset.my_dataset import Cifar10_Dataset
from utils.config import Config
from utils.tools import calculation_accuracy, training_process_visualization
from model.my_net import ResNet18


# 初始化超参数实例
opt = Config()

# 初始化训练集、测试集加载器实例
trainset = Cifar10_Dataset(data_roots=opt.train_roots, trans=opt.train_trans)
trainset_loader = data.DataLoader(trainset, opt.bs, num_workers=opt.nw, shuffle=True)
testset = Cifar10_Dataset(data_roots=opt.test_roots, trans=opt.test_trans)
testset_loader = data.DataLoader(testset, opt.bs, num_workers=opt.nw)

# 定义模型、优化器、损失函数、学习率调整器
net = ResNet18(opt.num_classes)
optimizer = t.optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.m, weight_decay=opt.wd)
loss_func = nn.CrossEntropyLoss()
lr_adjust = StepLR(optimizer, step_size=30, gamma=0.1)  

# 用于保存训练过程中的损失和准确率
train_loss = []
train_acc = []
test_acc = []

if t.cuda.is_available():
    net.cuda()
    loss_func.cuda()

# 更改为tqdm模块内的trange函数以了解训练时间
for epoch in trange(1, opt.epochs+1):
    net.train()
    for i, (data, label) in enumerate(trainset_loader):
        if t.cuda.is_available():
            data, label = data.cuda(), label.cuda()

        optimizer.zero_grad()
        pred = net(data)

        loss = loss_func(pred, label)
        acc = calculation_accuracy(pred, label)
        loss.backward()
        optimizer.step()

         # 如果没有用GPU加速，则把.cpu()删除
        train_loss.append(loss.cpu().detach().numpy())
        train_acc.append(acc)
        # print('loss:', loss.cpu().detach().numpy(), "acc", acc)
    lr_adjust.step()

    
    # 每训练完一轮进行一次测试
    net.eval()
    for j, (data, label) in enumerate(testset_loader):
        if t.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        test_pred = net(data)
        
        all_pred = test_pred if j == 0 else t.vstack((all_pred, test_pred))
        all_label = label if j == 0 else t.cat((all_label, label))
    
    acc = calculation_accuracy(all_pred, all_label)
    test_acc.append(acc)


data = {'train_loss':train_loss, 'train_acc':train_acc, 'test_acc':test_acc}
training_process_visualization(data)
 





