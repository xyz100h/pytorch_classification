# -*- coding:utf-8 -*-
# @time :2019.03.15
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju

import torch
import cfg
import os
from PIL import Image
from transform import get_test_transform
import pandas as pd
from tqdm import tqdm
import numpy as np

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def predict(model):
    # 读入模型
    model = load_checkpoint(model)
    print('..... Finished loading model! ......')
    ##将模型放置在gpu上运行
    if torch.cuda.is_available():
        model.cuda()
    pred_list, _id = [], []
    for i in tqdm(range(len(imgs))):
        img_path = imgs[i].strip()
        # print(img_path)
        _id.append(int(os.path.basename(img_path).split('.')[0]))
        img = Image.open(img_path).convert('RGB')
        # print(type(img))
        img = get_test_transform(size=cfg.INPUT_SIZE)(img).unsqueeze(0)

        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            out = model(img)
        prediction = torch.argmax(out, dim=1).cpu().item()
        pred_list.append(prediction)
    return _id, pred_list


if __name__ == "__main__":

    trained_model = cfg.TRAINED_MODEL
    labels2classes = cfg.labels_to_classes
    model_name = trained_model.split('/')[-2]
    with open(cfg.VAL_LABEL_DIR,  'r')as f:
        imgs = f.readlines()
    # print(len(imgs))

    _id, pred_list = predict(trained_model)
    # print(_id)
    # print(pred_list)
    submission = pd.DataFrame({"ID": _id, "Label": pred_list})
    submission.to_csv('/home/luxiangzhe/competition/102_flowers/weights/' + '{}_submission.csv'
                      .format(model_name), index=False, header=False)



