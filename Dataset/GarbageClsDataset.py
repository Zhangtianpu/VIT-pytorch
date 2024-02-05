import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import numpy as np
import os
from PIL import Image


class GarbageDataset(Dataset):
    def __init__(self, data_folder, resize=(256, 256), is_train=True):
        self.data_folder = data_folder
        self.is_train = is_train
        self.resize = resize
        if is_train:
            self.data_folder = os.path.join(self.data_folder, "TRAIN")
        else:
            self.data_folder = os.path.join(self.data_folder, 'TEST')
        self.label_index_map = {'O': 0, 'R': 1}
        self.label_list = []
        self.image_list = []

        self.load_file_label(self.data_folder)

        """
        data argumentation
        """

        img_resize = transforms.Resize(size=self.resize)
        colorJitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        vertificalFlip = transforms.RandomVerticalFlip()
        horizonFlip = transforms.RandomHorizontalFlip()
        rotation = transforms.RandomRotation(degrees=(0, 360))

        self.trans = transforms.Compose([
            img_resize,
            colorJitter,
            vertificalFlip,
            horizonFlip,
            rotation,
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, item):
        image=self.image_list[item]
        label=self.label_list[item]

        image=Image.open(image).convert('RGB')
        image=self.trans(image)
        return image/255,label

    def load_file_label(self, path):
        folder_path = os.path.join(path, 'O')
        filenames = os.listdir(folder_path)
        for filename in filenames:
            self.label_list.append(0)
            self.image_list.append(os.path.join(folder_path, filename))

        folder_path = os.path.join(path, 'R')
        filenames = os.listdir(folder_path)
        for filename in filenames:
            self.label_list.append(1)
            self.image_list.append(os.path.join(folder_path, filename))

if __name__ == '__main__':
    root_folder='/home/ztp/workspace/dataset/garbage_classfication_dataset'
    garbage_dataset=GarbageDataset(root_folder,resize=(256,256),is_train=True)
    dataloader=DataLoader(garbage_dataset,batch_size=4,shuffle=True)
    for image,label in dataloader:
        print(image)
        print(label)
