import os
import cv2
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import random

random.seed(10)


class ImageDataTrain(data.Dataset):
    def __init__(self, rgbd_image_root, rgbd_depth_root, rgbd_gt_root, image_size):
        self.image_size = image_size

        # load rgbd inputs
        self.rgbd_images = [os.path.join(rgbd_image_root, image) for image in os.listdir(rgbd_image_root)]
        self.rgbd_depths = [os.path.join(rgbd_depth_root, depth) for depth in os.listdir(rgbd_depth_root)]
        self.rgbd_gts = [os.path.join(rgbd_gt_root, gt) for gt in os.listdir(rgbd_gt_root)]

        self.rgbd_images = sorted(self.rgbd_images)
        self.rgbd_depths = sorted(self.rgbd_depths)
        self.rgbd_gts = sorted(self.rgbd_gts)

        self.sal_rgbd_num = len(self.rgbd_images)

    def __getitem__(self, item):
        # sal data loading
        rgbd_image_name = self.rgbd_images[item]
        rgbd_depth_name = self.rgbd_depths[item]
        rgbd_gt_name = self.rgbd_gts[item]

        rgbd_sal_image = load_image(rgbd_image_name, self.image_size)
        rgbd_sal_depth = load_image(rgbd_depth_name, self.image_size)
        rgbd_sal_label = load_sal_label(rgbd_gt_name, self.image_size)

        rgbd_sal_image, rgbd_sal_depth, rgbd_sal_label = \
            cv_random_flip(rgbd_sal_image, rgbd_sal_depth, rgbd_sal_label)

        rgbd_sal_image = torch.Tensor(rgbd_sal_image)
        rgbd_sal_depth = torch.Tensor(rgbd_sal_depth)
        rgbd_sal_label = torch.Tensor(rgbd_sal_label)

        sample = {'rgbd_image': rgbd_sal_image, 'rgbd_depth': rgbd_sal_depth, 'rgbd_label': rgbd_sal_label}
        return sample

    def __len__(self):
        return self.sal_rgbd_num


class ImageDataTest(data.Dataset):
    def __init__(self, image_root, depth_root, test_size):
        self.image_root = image_root
        self.depth_root = depth_root
        self.test_size = test_size
        self.images = [os.path.join(self.image_root, image) for image in os.listdir(self.image_root)]
        self.depths = [os.path.join(self.depth_root, depth) for depth in os.listdir(self.depth_root)]

        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.image_num = len(self.images)

    def __getitem__(self, item):
        image, im_size = load_image_test(self.images[item], self.test_size)
        depth, de_size = load_image_test(self.depths[item], self.test_size)
        depth = torch.Tensor(depth)
        image = torch.Tensor(image)
        return {'image': image, 'depth': depth, 'name': os.path.split(self.images[item])[-1], 'size': im_size}

    def __len__(self):
        return self.image_num


def get_loader(config, mode='train', pin=True):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(config.rgbd_image_root, config.rgbd_depth_root, config.rgbd_gt_root, config.image_size)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle,
                                      num_workers=config.num_thread, pin_memory=pin)
    else:
        dataset = ImageDataTest(config.rgbd_image_root, config.rgbd_depth_root, config.test_size)
        data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle,
                                      num_workers=0, pin_memory=pin)
    return data_loader


def load_image(path, img_size=None):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    if img_size:
        in_ = cv2.resize(in_, dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
    in_ = in_.transpose((2, 0, 1))
    return in_


def load_image_test(path, img_size=None):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    if img_size:
        in_ = cv2.resize(in_, dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
    in_ = in_.transpose((2, 0, 1))
    return in_, im_size


def load_sal_label(path, img_size=None):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # bgr mode
    label = np.array(im, dtype=np.float32)

    if img_size:
        label = cv2.resize(label, dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)

    label = label / 255.
    label = label[np.newaxis, ...]
    return label


def load_edge_label(path, image_size):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # bgr mode
    label = np.array(im, dtype=np.float32)
    label = cv2.resize(label, (image_size, image_size))
    label = label / 255.0
    label = label[np.newaxis, ...]
    return label


def cv_random_flip(img, depth, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img[:, :, ::-1].copy()
        depth = depth[:, :, ::-1].copy()
        label = label[:, :, ::-1].copy()
    return img, depth, label


def cv_random_flip_rgb(img, edge, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img[:, :, ::-1].copy()
        edge = edge[:, :, ::-1].copy()
        label = label[:, :, ::-1].copy()
    return img, edge, label

