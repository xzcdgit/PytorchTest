import time

import torch.optim.optimizer
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model import MyModule

if __name__ == "__main__":
    #======================== 1. 准备数据 ========================
    test_data = torchvision.datasets.CIFAR10(
        '../dataset',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True)
    test_data_size = len(test_data)
    print('test_data_size: {}'.format(test_data_size))

    #======================== 2. 加载数据数据 ========================
    test_dataloader = DataLoader(test_data, batch_size=10)

    #======================== 3. 加载模型 ========================
    mode_name = 'epoch_9_model.pkl'
    model = torch.load(mode_name)
    model.eval()

    #======================== 4. 定义损失函数 ========================
    loss_fn = nn.CrossEntropyLoss()

    #======================== 4. 测试 ========================
    total_test_step = 0
    same_count = 0
    total_test_loss = 0
    writer = SummaryWriter('logs_test')
    for data in test_dataloader:
        imgs, targets = data
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        total_test_loss = total_test_loss + loss
        preds = outputs.argmax(1)
        same_count += (preds == targets).sum()
        total_test_step = total_test_step + 1
        writer.add_scalar('test_loss', loss.item(), total_test_step)
        #plt.imshow(img)
        #plt.pause(2)
        #plt.cla()
    acc = same_count / test_data_size
    print(acc)
    print('整体测试集上的Loss: {}'.format(total_test_loss))
