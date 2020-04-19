# -*- coding:utf-8 -*-
# @time :2019.09.07
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju


##数据集的类别
NUM_CLASSES = 100

#训练时batch的大小
BATCH_SIZE = 32

#网络默认输入图像的大小
INPUT_SIZE = 300

##预训练模型的存放位置
LOCAL_PRETRAINED = {
    'resnext101_32x8d': '/home/lxztju/weights/resnext101_32x8.pth',
    'resnext101_32x16d': '/home/lxztju/weights/resnext101_32x16.pth',
    'resnet50': '/home/lxztju/weights/resnet50.pth',
    'resnet101': '/home/lxztju/weights/resnet101.pth',
    'densenet121': '/home/lxztju/weights/densenet121.pth',
    'densenet169': '/home/lxztju/weights/densenet169.pth',
    'moblienetv2': '/home/lxztju/weights/mobilenetv2.pth',
}

##训练完成，权重文件的保存路径,默认保存在trained_model下
TRAINED_MODEL = './weights/resnext101_32x16d/epoch_40.pth'


TRAINED_MODELS = ['./weights/resnext101_32x8d/epoch_10.pth',
                  './weights/resnext101_32x8d/epoch_10.pth',
                  './weights/resnext101_32x8d/epoch_10.pth']

#数据集的存放位置
TRAIN_LABEL_DIR = './datasets/train.txt'
# TRAIN_LABEL_DIR = '/media/luxiangzhe/disk/trashdata/train.txt'
VAL_LABEL_DIR = './datasets/val.txt'

labels_to_classes = {
    1: 'pos',
    0: 'neg'
}
