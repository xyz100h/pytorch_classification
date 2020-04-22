# -*- coding:utf-8 -*-
# @time :2020.04.20
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju

import torch
import cfg
import os
from PIL import Image
from transform import tta_test_transform
import pandas as pd
from tqdm import tqdm
from collections import Counter


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    # print(checkpoint)
    model = checkpoint['model']  # 提取网络结构
    # print(model)
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
        img1 = Image.open(img_path).convert('RGB')
        # print(type(img))
        pred = []
        for i in range(8):
            img = tta_test_transform(size=cfg.INPUT_SIZE)(img1).unsqueeze(0)

            if torch.cuda.is_available():
                img = img.cuda()
            with torch.no_grad():
                out = model(img)
            prediction = torch.argmax(out, dim=1).cpu().item()
            pred.append(prediction)
        res = Counter(pred).most_common(1)[0][0]
        pred_list.append(res)
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
    submission.to_csv('/home/luxiangzhe/competition/102_flowers/weights/' + '{}_tta_submission.csv'
                      .format(model_name), index=False, header=False)



