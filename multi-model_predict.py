# _*_coding:utf-8 _*_
# @author: lxztju
# @time: 2020/4/10 18:46
# @github: https://github.com/lxztju

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
    ##进行模型测试时，eval（）会固定下BN与Dropout的参数
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

        img = get_test_transform(size=cfg.INPUT_SIZE)(img).unsqueeze(0)
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            out = model(img)
        # print(out)
        # prediction = torch.argmax(out, dim=1).cpu().item()
        pred_list.extend(out.cpu().numpy())
    return _id, pred_list

def multi_model_predict(model_list, mode):
    preds_dict = dict()
    for model in model_list:
        model_name = model.split('/')[-2]
        #得到每个模型的预测值
        _id, pred_list = predict(model)
        #将得到的结果存储在csv中
        submission = pd.DataFrame({"ID":_id, "Label":pred_list})
        submission.to_csv('./results/' + '{}_submission.csv'
                          .format(model_name), index=False, header=False)
        if 'ID' in preds_dict:
            pass
        else:
            preds_dict['ID'] = _id
        preds_dict['{}'.format(model_name)] = pred_list
    pred_list = get_pred_list(preds_dict, mode)
    submission = pd.DataFrame({"id": range(len(pred_list)), "label": pred_list})
    submission.to_csv('./results/' + 'submission.csv', index=False, header=False)


def get_pred_list(preds_dict, mode = 'weight'):
    """
    对得到的每个模型的csv文件，然后及逆行加权或者投票融合
    :param preds_dict: 预测得到的模型预测值，格式为字典，键值为model_name，值为预测值
    :param mode:  可选的值为vote或者weight，为加权融合或者投票融合
    :return: 融合后的预测值
    """
    pred_list = []
    if mode == 'weight':
        for i in range(len(preds_dict['ID'])):
            prob = None
            for model in model_list:
                model_name =  model.split('/')[-2]
                if prob is None:
                    prob = preds_dict['{}'.format(model_name)][i] * ratio_dict[model_name]
                else:
                    prob += preds_dict['{}'.format(model_name)][i] * ratio_dict[model_name]
            pred_list.append(np.argmax(prob))
    else:
        for i in range(len(preds_dict['ID'])):
            preds = []
            for model in model_list:
                model_name = model.split('/')[-2]
                prob = preds_dict['{}'.format(model_name)][i]
                pred = np.argmax(prob)
                preds.append(pred)
            #挑选出预测结果出现的次数的最大值
            pred_list.append(max(preds, key=preds.count))
    return pred_list



if __name__ == "__main__":

    model_list = cfg.TRAINED_MODELS
    labels2classes = cfg.labels_to_classes
    #获取数据集图像的存储列表
    with open(cfg.VAL_LABEL_DIR,  'r')as f:
        imgs = f.readlines()[:10]
    #加权融合的权重
    ratio_list=[1,2,3,4]
    ratio_dict = dict(zip(model_list, ratio_list))
    multi_model_predict(model_list, mode='weight')

