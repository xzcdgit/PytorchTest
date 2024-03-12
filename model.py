import torch
from torch import nn

#标签注释
Label_Names = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


#构造神经网络
class MyModule(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 32, 5, padding=2),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(32, 32, 5, padding=2),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(32, 64, 5, padding=2),
                                   nn.MaxPool2d(2), nn.Flatten(),
                                   nn.Linear(1024, 64), nn.Linear(64, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


#测试
if __name__ == '__main__':
    my_module = MyModule()
    input = torch.ones((64, 3, 32, 32))
    output = my_module(input)
    print(output.shape)
