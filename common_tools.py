import random
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms


def set_seed(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def transform_invert(img_: torch.Tensor, transform_train: transforms.Compose):
    #如果有标准化操作
    if 'Normalize' in str(transform_train):
        #取出标准化的transform
        norm_transform = list(
            filter(lambda x: isinstance(x, transforms.Normalize),
                   transform_train.transforms))
        #取出均值
        mean = torch.tensor(norm_transform[0].mean,
                            dtype=img_.dtype,
                            device=img_.device)
        #取出标准差
        std = torch.tensor(norm_transform[0].std,
                           dtype=img_.dtype,
                           device=img_.device)
        #乘以标准差再加上均值
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    #将c*h*w转为h*w*c
    img_ = img_.transpose(0, 2).transpose(0, 1)
    #print(type(img_))
    #将0~1的值变为0-255
    img_ = img_.detach().numpy() * 255
    #img_ = np.array(img_) * 255
    #print(type(img_))
    #如果是rpg图
    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    #如果是灰度图
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception(
            "Invalid img shape, excepted 1 or 3 in axis 2, but got {}".format(
                img_.shape[2]))
    return img_
