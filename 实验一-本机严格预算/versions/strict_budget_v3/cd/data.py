"""两期图像与标签共享几何增强；弱强分支只改变颜色，不改变坐标。"""
import random
from collections import OrderedDict
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


def rgb_tensor(image):
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy().transpose(2, 0, 1)) / 255.0


def strong_color(image, rng=random):
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.7, 1.3))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.7, 1.3))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 1.0)))
    return image


class PairDataset(Dataset):
    def __init__(self, records, crop=256, unlabeled=False, seed=None, cache_pairs=0):
        self.records, self.crop, self.unlabeled = records, crop, unlabeled
        self.rng = random.Random(seed) if seed is not None else random
        self.cache_pairs = max(0, cache_pairs)
        self.cache = OrderedDict()
        if not records:
            raise ValueError('训练列表为空。')
        if unlabeled and any('label' in x for x in records):
            raise ValueError('无标签清单不得包含label字段。')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        key = r['a']
        if key in self.cache:
            images = self.cache.pop(key)
            self.cache[key] = images
        else:
            with Image.open(r['a']) as im:a = im.convert('RGB')
            with Image.open(r['b']) as im:b = im.convert('RGB')
            images = [a, b]
            if not self.unlabeled:
                with Image.open(r['label']) as im:images.append(im.convert('L'))
            if self.cache_pairs:
                self.cache[key] = images
                while len(self.cache) > self.cache_pairs:self.cache.popitem(last=False)
        a,b = images[:2]
        w, h = a.size
        if min(w, h) < self.crop:
            raise ValueError('图像小于crop；请减小crop，不自动缩放标签。')
        x, y = self.rng.randrange(w - self.crop + 1), self.rng.randrange(h - self.crop + 1)
        images = [im.crop((x, y, x + self.crop, y + self.crop)) for im in images]
        if self.rng.random() < 0.5:
            images = [im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for im in images]
        rotations = self.rng.randrange(4)
        images = [im.rotate(90 * rotations) for im in images]
        a, b = images[:2]
        if self.unlabeled:
            return rgb_tensor(a), rgb_tensor(b), rgb_tensor(strong_color(a, self.rng)), rgb_tensor(strong_color(b, self.rng))
        mask = torch.from_numpy((np.asarray(images[2]) > 0).astype(np.float32).copy())[None]
        return rgb_tensor(strong_color(a, self.rng)), rgb_tensor(strong_color(b, self.rng)), mask
