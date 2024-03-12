import torch.optim.optimizer
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from model import MyModule
from tensorboardX import SummaryWriter

if __name__ == '__main__':

    #======================== 1. 准备数据 ========================
    train_data = torchvision.datasets.CIFAR10(
        '../dataset',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True)

    train_data_size = len(train_data)

    print('train_data_size:{}'.format(train_data_size))

    #======================== 2. 加载数据 ========================
    train_dataloader = DataLoader(train_data, batch_size=64)

    #======================== 3. 创建网络模型 ========================
    my_module = MyModule()

    #======================== 4. 损失函数 ========================
    loss_fn = nn.CrossEntropyLoss()

    #======================== 5. 优化器 ========================
    LEARN_RATE = 1e-2
    optimizer = torch.optim.SGD(my_module.parameters(), lr=LEARN_RATE)

    #======================== 6. 迭代 ========================
    total_train_step = 0

    EPOCH_MAX = 10
    writer = SummaryWriter('logs_train')
    #训练 EPOCH_MAX次
    my_module.train()
    for iteration in range(EPOCH_MAX):
        #每批64张图片，targets为标签
        for data in train_dataloader:
            imgs, targets = data
            outputs = my_module(imgs)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_step += 1
            #记录损失函数
            writer.add_scalar('train_loss', loss.item(), total_train_step)
            print("训练次数: {},loss: {}".format(total_train_step, loss.item()))
        torch.save(my_module, 'epoch_{}_model.pkl'.format(iteration))
