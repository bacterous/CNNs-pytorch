import os
from random import shuffle
from PIL import Image
from PIL import ImageFile
from torch.utils import data
import numpy as np
from torchvision import transforms as T


class Echocardiography(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        self.test = test

        # data/test/902c9-23f99e.jpg
        # data/train/1/d923e-3094f9c.jpg
        if self.test:
            imgs = [os.path.join(root, img) for img in os.listdir(root)]
            imgs = sorted(imgs, key=lambda x: x.split('.')[-2].split('\\')[-1])

            # classes = [os.path.join(root, cla) for cla in os.listdir(root)]
            # imgs = [os.path.join(cla, img) for cla in classes for img in os.listdir(cla)]
            # imgs = sorted(imgs, key=lambda x: x.split('.')[-2].split('\\')[-1])
            self.imgs = imgs
        else:
            classes = [os.path.join(root, cla) for cla in os.listdir(root)]
            imgs = [os.path.join(cla, img) for cla in classes for img in os.listdir(cla)]
            imgs = sorted(imgs, key=lambda x: x.split('.')[-2].split('\\')[-1])
            shuffle(imgs)
            self.imgs = imgs
            imgs_num = len(imgs)
            if train:
                self.imgs = imgs[:int(0.8 * imgs_num)]
            else:
                self.imgs = imgs[int(0.8 * imgs_num):]

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_path = self.imgs[index]
        if self.test:
            label = img_path.split('\\')[-1]
        else:
            label = int(img_path.split('\\')[-2]) - 1

        with Image.open(img_path) as data:
            data = data.convert('RGB')
            data = self.transforms(data)
            return data, label

    def __len__(self):
        return len(self.imgs)
