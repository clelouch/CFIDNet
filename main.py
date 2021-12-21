import argparse
import os
from dataset import get_loader
from solver import Solver
import time


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def get_info(config):
    if config.mode == 'train':
        config.rgbd_image_root = os.path.join(config.rgbd_data_root, 'RGBD_for_train', 'RGB')
        config.rgbd_depth_root = os.path.join(config.rgbd_data_root, 'RGBD_for_train', 'depth')
        config.rgbd_edge_root = os.path.join(config.rgbd_data_root, 'RGBD_for_train', 'edges')
        config.rgbd_gt_root = os.path.join(config.rgbd_data_root, 'RGBD_for_train', 'GT')
    else:
        print(config.sal_mode, ': rgbd test')

        config.rgbd_image_root = os.path.join(config.rgbd_data_root, 'RGBD_for_test', config.sal_mode, 'RGB')
        config.rgbd_depth_root = os.path.join(config.rgbd_data_root, 'RGBD_for_test', config.sal_mode, 'depth')
        config.sal_save = os.path.join('./CFIDNet_test_dir', config.sal_mode,
                                       config.name)
        if not os.path.exists(config.sal_save):
            os.makedirs(config.sal_save)


def main(config):
    get_info(config)
    if config.mode == 'train':
        train_loader = get_loader(config)
        run = 0
        save_root = './'
        while os.path.exists("%s/run-%d" % (save_root, run)):
            run += 1
        os.makedirs("%s/run-%d" % (save_root, run))
        os.mkdir("%s/run-%d/models" % (save_root, run))
        config.save_folder = "%s/run-%d" % (save_root, run)
        train = Solver(train_loader, None, config)
        train.train()

        with open('%s/args.txt' % (config.save_folder), 'w') as f:
            for arg in vars(config):
                print('%s: %s' % (arg, getattr(config, arg)), file=f)
    elif config.mode == 'test':
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config.sal_save): os.makedirs(config.sal_save)
        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':
    resnet_path = 'pretrained/resnet50_caffe.pth'
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.00003)  # Learning rate resnet:5e-5
    parser.add_argument('--wd', type=float, default=0.0005)  # Weight decay
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--cuda', type=bool, default=True)

    # Training settings
    parser.add_argument('--pretrained_model', type=str, default=resnet_path)  # pretrained backbone model
    parser.add_argument('--epoch', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--norm', type=str, default='gn')
    parser.add_argument('--num_thread', type=int, default=4)
    parser.add_argument('--load', type=str, default='')  # resume training from a snapshot, '':  training from start
    parser.add_argument('--save_folder', type=str, default='')
    parser.add_argument('--epoch_save', type=int, default=5)
    parser.add_argument('--iter_size', type=int, default=10)
    parser.add_argument('--show_every', type=int, default=50)
    parser.add_argument('--ratio', type=int, default=8)

    # Train data
    parser.add_argument('--rgbd_data_root', type=str, default='E:\sal_rgbd_datasets\dataset')
    # '/home/omnisky/diskB/datasets/RGBD_Datasets')

    # Testing settings
    parser.add_argument('--model', type=str, default='')  # checkpoint for test
    parser.add_argument('--sal_mode', type=str, default='STERE')  # Test image dataset
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--sal_save', type=str, default='')
    parser.add_argument('--test_size', type=int, default=320)

    # Architecture settings
    parser.add_argument('--branches', type=int, default=4)  # multi-scale features branches
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--loss', type=str, default='iou')

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # Device
    parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    config = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    main(config)
