# -*- coding:utf-8 -*-
# @time :2020.02.09
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim


from dataset import train_dataloader,train_datasets
import cfg
from warmup_lr import adjust_learning_rate
from build_model import Resnet50, Resnet101, Resnext101_32x8d,Resnext101_32x16d, Densenet121, Densenet169, Mobilenetv2, Efficientnet


##命令行交互，设置一些基本的参数
parser = argparse.ArgumentParser("Train the densenet")

parser.add_argument('-max', '--max_epoch', default=92,
                    help = 'maximum epoch for training')

parser.add_argument('-ng', '--ngpu', default=2,
                    help = 'use multi gpu to train')

parser.add_argument('--resume_epoch', default=40,
                    help = 'resume training from resume_epoch')

parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('-lr', '--learning_rate', default=5e-4,
                    help = 'initial learning rate for training')

##训练保存模型的位置
parser.add_argument('--save_folder', default='./weights',
                    help='the dir to save trained model ')

args = parser.parse_args()

model_name = 'resnext101_32x16d'
##创建训练模型参数保存的文件夹
save_folder = args.save_folder + model_name
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    return model

#####build the network model
if not args.resume_epoch:
    model = Resnext101_32x16d(num_classes=cfg.NUM_CLASSES)
    # 冻结前边一部分层不训练
    ct = 0
    for child in model.children():
        ct += 1
        # print(child)
        if ct < 8:
            print(child)
            for param in child.parameters():
                param.requires_grad = False
    # print(model)
if args.resume_epoch:
    print('***** Resume training from epoch {{}} *******'.format(args.resume_epoch)）
    model = load_checkpoint(os.path.join(save_folder, 'epoch_{}.pth'.format(args.resume_epoch)))

    # state_dict = torch.load(os.path.join(save_folder, 'epoch_{}.pth'.format(args.resume_epoch)))['model_state_dict']
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     head = k[:7]
    #     if head == 'module.':
    #         name = k[7:] # remove `module.`
    #     else:
    #         name = k
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)

# c = 0
# for name, p in model.named_parameters():
#     c += 1
#     if c >=450:
#         break
#     p.requires_grad = True


##进行多gpu的并行计算
if args.ngpu:
    print('***** Multiple gpus parallel training.*********')
    model = nn.DataParallel(model,device_ids=list(range(args.ngpu)))

print("...... Initialize the network done!!! .......")

###模型放置在gpu上进行计算
if torch.cuda.is_available():
    model.cuda()
# for p in model.parameters():
#     print(p.requires_grad)
##定义优化器与损失函数
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
# optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
#                       momentum=args.momentum, weight_decay=args.weight_decay)

criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()

lr = args.learning_rate
# for epoch in range(args.max_epoch):
batch_size = cfg.BATCH_SIZE

#每一个epoch含有多少个batch
max_batch = len(train_datasets)//batch_size
warmup_epoch=5
epoch_size = len(train_datasets) // batch_size
max_iter = args.max_epoch * epoch_size
warmup_steps = warmup_epoch * epoch_size
print(warmup_steps)
##训练max_epoch个epoch
start_iter = args.resume_epoch * epoch_size
model.train()
epoch = args.resume_epoch
global_step = 0
for iteration in range(start_iter, max_iter):
    global_step += 1

    ##更新迭代器
    if iteration % epoch_size == 0:
        # create batch iterator
        batch_iterator = iter(train_dataloader)
        loss = 0

        ###保存模型
        if epoch % 10 == 0 and epoch > 0:
            checkpoint = {'model': model.module,
                          'model_state_dict': model.module.state_dict(),
                          # 'optimizer_state_dict': optimizer.state_dict(),
                          'epoch': epoch}
            torch.save(checkpoint, os.path.join(save_folder, 'epoch_{}.pth'.format(epoch)))
            # torch.save(model.state_dict(), save_folder +'/'+ model_name + '_' + repr(epoch) + '.pth')
        epoch += 1

    ## 调整学习率
    lr = adjust_learning_rate(optimizer, global_step=global_step,
                              learning_rate_base=args.learning_rate,
                              total_steps=max_iter,
                              warmup_steps=warmup_steps)
    ## 获取image 和 label
    try:
        images, labels = next(batch_iterator)
    except:
        continue

    ##在pytorch0.4之后将Variable 与tensor进行合并，所以这里不需要进行Variable封装
    if torch.cuda.is_available():
        images, labels = images.cuda(), labels.cuda()

    out = model(images)
    loss = criterion(out, labels)

    optimizer.zero_grad()  # 清空梯度信息，否则在每次进行反向传播时都会累加
    loss.backward()  # loss反向传播
    optimizer.step()  ##梯度更新

    prediction = torch.max(out, 1)[1]
    train_correct = (prediction == labels).sum()
    ##这里得到的train_correct是一个longtensor型，需要转换为float
    # print(train_correct.type())
    train_acc = (train_correct.float()) / batch_size

    if iteration % 10 == 0:
        print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
              + '|| Totel iter ' + repr(iteration) + ' || Loss: %.6f||' % (loss.item()) + 'ACC: %.3f ||' %(train_acc * 100) + 'LR: %.8f' % (lr))





