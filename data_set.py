import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class LabelCreator():

    def __init__(self) -> None:
        self.text = ''
        self.filepath = ''

    #args 分类文件夹的路径，根据数量自动打上0、1、2、3的标签
    def append_text(self, *args: str, read_limit: int = 0):
        for i, folder_path in enumerate(args):
            files = os.listdir(folder_path)
            for j, file in enumerate(files):
                if read_limit != 0 and j >= read_limit:
                    break
                else:
                    self.text = self.text + file + " " + str(i) + "\n"
        return self.text

    def clean_all(self):
        self.text = ''

    def get_text(self):
        return self.text
    
    def save_text(self, filepath:str):
        self.filepath = filepath
        with open(filepath,'w') as f:
            f.write(self.text)


class MyDataset(Dataset):

    def __init__(self, txtpath:str, *datapath:str) -> None:
        super().__init__()
        self.txtpath = txtpath
        self.datapath = datapath
        imgs = []
        with open(txtpath,'r') as f:
            for line in f.readlines():
                tmp = line.split()
                imgs.append(tuple(tmp))
        print(imgs)
        self.imgs = imgs
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        pic,label = self.imgs[index]
        label = int(label)
        pic = Image.open(self.datapath[label]+'\\'+pic)
        pic = transforms.ToTensor()(pic)
        pic = transforms.CenterCrop((320,320))(pic)
        return pic,label
        

            
        

if __name__ == '__main__':
    text_creator = LabelCreator()
    txt = text_creator.append_text(r'D:\Code\Python\TorchData\cats',r'D:\Code\Python\TorchData\dogs',read_limit=2)
    #print(txt)
    filepath = "test.txt"
    text_creator.save_text()
    
    txt_path = r'D:\Code\Python\PytorchTest\test.txt'
    
    data = MyDataset(txt_path,r'D:\Code\Python\TorchData\cats',r'D:\Code\Python\TorchData\dogs')
    
    data_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    for pic,label in data_loader:
        print(pic.shape, label.item())
        img = pic.squeeze()
        img = img.numpy()
        img = np.transpose(img, (1,2,0))
        plt.imshow(img)
        plt.pause(2)
        plt.cla()
    
    
