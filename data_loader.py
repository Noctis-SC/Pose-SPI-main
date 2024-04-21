import torch
import torchvision.transforms
from torch.utils.data import DataLoader

import os
from PIL import Image

# Define custom dataset class
class CocoKeypointsDataset(torch.utils.data.Dataset):
    """Class for loading coco-style constructed dataset of keypoints
    Args:
        coco - COCO dataset object
        image_dir - string, directory of images"""
    def __init__(self, coco, image_dir, transform=None):
        self.coco = coco
        self.image_dir = image_dir
        self.transform = transform
        self.ids = list(coco.imgs.keys())
        self.mask_transform = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        keypoints = target[0]['keypoints']
        image_id = target[0]['image_id']
        class_id = target[0]['category_id'] #20 for person in toy dataset

        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        return img, keypoints, image_id

    def __len__(self):
        return len(self.ids)



