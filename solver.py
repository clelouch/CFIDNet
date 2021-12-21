import torch
from torch.optim import Adam
from model import Net
import numpy as np
import os
import cv2
import time, datetime
from loss import bce_iou_loss
import logging
from tensorboardX import SummaryWriter


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)
    return dn


RGBD_Dataset_List = ['DES', 'DUT', 'LFSD', 'NJU2K', 'SIP', 'SSD', 'STERE']


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.lr_decay_epoch = [60, ]
        self.build_model()
        if self.config.loss == 'iou':
            self.loss = bce_iou_loss
        else:
            self.loss = torch.nn.BCELoss()
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            if self.config.cuda:
                self.net.load_state_dict(torch.load(self.config.model))
            else:
                self.net.load_state_dict(torch.load(self.config.model, map_location='cpu'))
            self.net.eval()
        else:
            logging.basicConfig(filename=self.config.save_folder + 'log.log',
                                format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                                level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
            logging.info("CFIDNet-Train")
            logging.info("Config")
            logging.info(
                'epoch:{};lr:{};batchsize:{};trainsize:{};save_path:{};decay_epoch:{}'.format(
                    self.config.epoch, self.config.lr, self.config.batch_size, self.config.image_size,
                    self.config.save_folder,
                    self.lr_decay_epoch))
            self.writer = SummaryWriter(self.config.save_folder, 'summary')

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        self.net = Net(self.config)
        if self.config.cuda:
            self.net = self.net.cuda()

        self.net.train()
        # self.net.eval()
        if self.config.load == '':
            self.net.rgb_net.load_state_dict(torch.load(self.config.pretrained_model))
            self.net.depth_net.load_state_dict(torch.load(self.config.pretrained_model))
        else:
            self.net.load_state_dict(torch.load(self.config.load))

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr,
                              weight_decay=self.wd)

    def test(self):
        self.net.eval()

        for i, data_batch in enumerate(self.test_loader):
            images, depth, name, im_size = \
                data_batch['image'], data_batch['depth'], data_batch['name'][0], np.asarray(data_batch['size'])
            im_size = im_size[1], im_size[0]
            # print(im_size)
            with torch.no_grad():
                if self.config.cuda:
                    images, depth = images.cuda(), depth.cuda()
                    sal_pred = self.net(images, depth)

                sal_pred = normPRED(sal_pred[1])

                sal_pred = np.squeeze(sal_pred.cpu().data.numpy())
                sal_pred = 255 * sal_pred
                sal_pred = cv2.resize(sal_pred, dsize=im_size, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(self.config.sal_save, name[:-4] + '.png'), sal_pred)
        print(self.config.sal_mode + ' Test Done!')

    def deep_supervision_loss(self, preds, gt):
        losses = []
        sum_loss = 0
        for pred in preds:
            if self.config.loss == 'iou':
                losses.append(self.loss(pred, gt))
            else:
                losses.append(self.loss(pred, gt))
        for i in range(self.config.decoders):
            sum_loss += losses[i]
        sum_loss += losses[-4] / 2 + losses[-3] / 4 + losses[-2] / 8 + losses[-1] / 8
        return sum_loss

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        for epoch in range(self.config.epoch):
            self.net.zero_grad()
            # record time
            start_time = time.time()

            for i, data_batch in enumerate(self.train_loader):
                rgbd_image, rgbd_depth, rgbd_label = \
                    data_batch['rgbd_image'], data_batch['rgbd_depth'], data_batch['rgbd_label'],
                rgbd_image, rgbd_depth, rgbd_label = rgbd_image.cuda(), rgbd_depth.cuda(), rgbd_label.cuda()

                sal_preds = self.net(rgbd_image, rgbd_depth)
                sal_loss = self.deep_supervision_loss(sal_preds, rgbd_label)

                self.optimizer.zero_grad()
                sal_loss.backward()
                self.optimizer.step()

                if i % (self.show_every // self.config.batch_size) == 0:
                    end_time = time.time()
                    duration_time = end_time - start_time
                    time_second_avg = duration_time / self.show_every
                    eta_sec = time_second_avg * (
                            (self.config.epoch - epoch - 1) * len(self.train_loader) * self.config.batch_size + (
                            len(self.train_loader) - i) * self.config.batch_size)
                    eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

                    print('epoch: [%3d/%3d], iter: [%5d/%5d], eta: %s  || rgbd_sal_loss: %10.4f' % (
                        epoch, self.config.epoch, i, iter_num, eta_str, sal_loss.cpu().data))
                    logging.info(
                        '#TRAIN#:Epoch [{%3d}/{%3d}], Step [{%4d}/{%4d}], RGBD_sal_loss: {%10.4f}' %
                        (epoch, self.config.epoch, i, iter_num, sal_loss.cpu().data,))

                    self.writer.add_scalar('RGBD_Sal_Loss', sal_loss.cpu().data, global_step=epoch * iter_num + i)
                    print('Learning rate: ' + str(self.lr))
                    start_time = time.time()

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr,
                                      weight_decay=self.wd)

        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_folder)
