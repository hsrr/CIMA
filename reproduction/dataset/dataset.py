from distutils.command.config import config
import json
import os
import random

from torch.utils.data import Dataset
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption
import os
from torchvision.transforms.functional import hflip, resize

import math
import random
from random import random as rand

def load_annotations(paths):
    anns = []
    decoder = json.JSONDecoder()
    for fpath in paths:
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            continue

        # Case 1: standard JSON (list or dict)
        try:
            obj = json.loads(content)
            if isinstance(obj, list):
                anns.extend(obj)
            else:
                anns.append(obj)
            continue
        except json.JSONDecodeError:
            pass

        # Case 2: concatenated JSON blocks in one file
        idx = 0
        parsed_any = False
        while idx < len(content):
            while idx < len(content) and content[idx].isspace():
                idx += 1
            if idx >= len(content):
                break
            try:
                obj, end = decoder.raw_decode(content, idx)
            except json.JSONDecodeError:
                break
            parsed_any = True
            if isinstance(obj, list):
                anns.extend(obj)
            else:
                anns.append(obj)
            idx = end
        if parsed_any:
            continue

        # Case 3: JSONL (one json object per line)
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, list):
                    anns.extend(obj)
                else:
                    anns.append(obj)
    return anns

class DGM4_Dataset(Dataset):
    def __init__(self, config, ann_file, transform, max_words=30, is_train=True): 
        
        self.root_dir = './datasets'
        self.ann = load_annotations(ann_file)
        if 'dataset_division' in config:
            self.ann = self.ann[:int(len(self.ann)/config['dataset_division'])]

        self.transform = transform
        self.max_words = max_words
        self.image_res = config['image_res']

        self.is_train = is_train
        
    def __len__(self):
        return len(self.ann)

    def get_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        return int(xmin), int(ymin), int(w), int(h)    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        img_dir = ann['image']    
        image_dir_all = f'{self.root_dir}/{img_dir}'

        try:
            image = Image.open(image_dir_all).convert('RGB')   
        except Warning:
            raise ValueError("### Warning: fakenews_dataset Image.open")   
                         
        W, H = image.size
        has_bbox = False
        try:
            x, y, w, h = self.get_bbox(ann['fake_image_box'])
            has_bbox = True
        except:
            fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)

        do_hflip = False
        if self.is_train:
            if rand() < 0.5:
                # flipped applied
                image = hflip(image)
                do_hflip = True

            image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
        image = self.transform(image)
            
        if has_bbox:
            # flipped applied
            if do_hflip:  
                x = (W - x) - w  # W is w0

            # resize applied
            x = self.image_res / W * x
            w = self.image_res / W * w
            y = self.image_res / H * y
            h = self.image_res / H * h

            center_x = x + 1 / 2 * w
            center_y = y + 1 / 2 * h

            fake_image_box = torch.tensor([center_x / self.image_res, 
                        center_y / self.image_res,
                        w / self.image_res, 
                        h / self.image_res],
                        dtype=torch.float)

        label = ann['fake_cls']
        caption = pre_caption(ann['text'], self.max_words)
        fake_text_pos = ann['fake_text_pos']

        fake_text_pos_list = torch.zeros(self.max_words)

        for i in fake_text_pos:
            if i<self.max_words:
                fake_text_pos_list[i]=1
        
                
        return image, label, caption, fake_image_box, fake_text_pos_list, W, H


class Clip_Dataset(Dataset):
    def __init__(self, config, ann_file, max_words=30, is_train=True): 
        
        self.root_dir = './datasets/'
        self.ann = load_annotations(ann_file)
        if 'dataset_division' in config:
            self.ann = self.ann[:int(len(self.ann)/config['dataset_division'])]

        self.max_words = max_words
        self.image_res = config['image_res']

        self.is_train = is_train

        self.processor = CLIPProcessor.from_pretrained("./datasets/clip-vit-large-patch14/")
        
    def __len__(self):
        return len(self.ann)

    def get_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        return int(xmin), int(ymin), int(w), int(h)    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        img_dir = ann['image']    
        image_dir_all = f'{self.root_dir}/{img_dir}'

        try:
            image = Image.open(image_dir_all).convert('RGB')   
        except Warning:
            raise ValueError("### Warning: fakenews_dataset Image.open")   
                         
        W, H = image.size
        has_bbox = False
        try:
            x, y, w, h = self.get_bbox(ann['fake_image_box'])
            has_bbox = True
        except:
            fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)

        do_hflip = False
        if self.is_train:
            if rand() < 0.5:
                # flipped applied
                image = hflip(image)
                do_hflip = True
                
        caption = pre_caption(ann['text'], self.max_words)
        image = self.processor(images=image,return_tensors="pt")

        if has_bbox:
            # flipped applied
            if do_hflip:  
                x = (W - x) - w  # W is w0

            # resize applied
            x = self.image_res / W * x
            w = self.image_res / W * w
            y = self.image_res / H * y
            h = self.image_res / H * h

            center_x = x + 1 / 2 * w
            center_y = y + 1 / 2 * h

            fake_image_box = torch.tensor([center_x / self.image_res, 
                        center_y / self.image_res,
                        w / self.image_res, 
                        h / self.image_res],
                        dtype=torch.float)

        label = ann['fake_cls']
        fake_text_pos = ann['fake_text_pos']

        fake_text_pos_list = torch.zeros(self.max_words)

        for i in fake_text_pos:
            if i<self.max_words:
                fake_text_pos_list[i]=1
        
                
        return image,label,caption, fake_image_box, fake_text_pos_list, W, H


class Vilt_Dataset(Dataset):
    def __init__(self, config, ann_file, max_words=30, is_train=True):

        self.root_dir = './DGM4/datasets'
        self.ann = load_annotations(ann_file)
        if 'dataset_division' in config:
            self.ann = self.ann[:int(len(self.ann) / config['dataset_division'])]

        self.max_words = max_words
        self.image_res = config['image_res']

        self.is_train = is_train

    def __len__(self):
        return len(self.ann)

    def get_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        return int(xmin), int(ymin), int(w), int(h)

    def __getitem__(self, index):

        ann = self.ann[index]
        img_dir = ann['image']
        image_dir_all = f'{self.root_dir}/{img_dir}'

        try:
            image = Image.open(image_dir_all).convert('RGB')
        except Warning:
            raise ValueError("### Warning: fakenews_dataset Image.open")

        W, H = image.size
        has_bbox = False
        try:
            x, y, w, h = self.get_bbox(ann['fake_image_box'])
            has_bbox = True
        except:
            fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)

        do_hflip = False
        if self.is_train:
            if rand() < 0.5:
                # flipped applied
                image = hflip(image)
                do_hflip = True

        caption = pre_caption(ann['text'], self.max_words)
        image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        # inputs = self.processor(images = image, text = caption, max_length=40, padding='max_length',truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False)
        if has_bbox:
            # flipped applied
            if do_hflip:
                x = (W - x) - w  # W is w0

            # resize applied
            x = self.image_res / W * x
            w = self.image_res / W * w
            y = self.image_res / H * y
            h = self.image_res / H * h

            center_x = x + 1 / 2 * w
            center_y = y + 1 / 2 * h

            fake_image_box = torch.tensor([center_x / self.image_res,
                                           center_y / self.image_res,
                                           w / self.image_res,
                                           h / self.image_res],
                                          dtype=torch.float)

        label = ann['fake_cls']
        fake_text_pos = ann['fake_text_pos']

        fake_text_pos_list = torch.zeros(self.max_words)

        for i in fake_text_pos:
            if i < self.max_words:
                fake_text_pos_list[i] = 1

        return image, caption, label, fake_image_box, fake_text_pos_list, W, H
