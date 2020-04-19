import os
import glob

traindata_path = '/home/luxiangzhe/competition/maskclass/datasets/train'

labels = os.listdir(traindata_path)

valdata_path = '/home/luxiangzhe/competition/maskclass/datasets/toPredict/'


##写train.txt文件
txtpath = '/home/luxiangzhe/competition/maskclass/datasets/'
print(labels)
for index, label in enumerate(labels):
    imglist = glob.glob(os.path.join(traindata_path,label, '*.jpg'))
    # print(imglist)
    with open(txtpath + 'train.txt', 'a')as f:
        for img in imglist:
            # print(img + ' ' + str(index))
            f.write(img + ' ' + str(index))
            f.write('\n')
    # print(imglist)


imglist = glob.glob(os.path.join(valdata_path, '*.jpg'))

with open(txtpath + 'val.txt', 'a')as f:
    for img in imglist:
        f.write(img)
        f.write('\n')