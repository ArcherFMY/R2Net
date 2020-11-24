import argparse
import os
from dataset.dataset_image import get_loader
from solver import Solver
from torchvision import transforms

def get_test_info(sal_mode='duts'):
    if sal_mode == 'duts':
        image_root = './data/DUTS-TE/DUTS-TE-Image/' 
        image_source = './data/DUTS-TE/DUTS-TE_test.lst'
    elif sal_mode == 'ecssd':
        image_root = './data/ECSSD/ECSSD-Image/'
        image_source = './data/ECSSD/ECSSD_test.lst'
    else:
        image_root = []
        image_source = []
        raise Exception("please set the paths for dataset %s" % sal_mode) 
        
    return image_root, image_source

def main(config):
    if config.mode == 'train':
        train_loader = get_loader(config)
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_folder, run)):
            run += 1
        os.mkdir("%s/run-%d" % (config.save_folder, run))
        os.mkdir("%s/run-%d/models" % (config.save_folder, run))
        config.save_folder = "%s/run-%d" % (config.save_folder, run)
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        config.test_root, config.test_list = get_test_info(config.sal_mode)
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
        if not os.path.exists(config.test_fold + config.sal_mode):
            os.mkdir(config.test_fold + config.sal_mode)
        config.test_fold = config.test_fold + config.sal_mode + '/'

        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")

if __name__ == '__main__':

    vgg_path = './dataset/pretrained/vgg16_20M.pth'
    resnet_path = './dataset/pretrained/resnet50_caffe.pth'

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5) # Learning rate resnet:5e-5, vgg:1e-4
    parser.add_argument('--wd', type=float, default=0.0005) # Weight decay
    parser.add_argument('--cuda', type=bool, default=True)
    # Training settings
    parser.add_argument('--arch', type=str, default='resnet') # resnet or vgg, note that the vgg version is not implemented yet
    parser.add_argument('--pretrained_model', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=10) 
    parser.add_argument('--num_thread', type=int, default=8)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_folder', type=str, default='./results')
    parser.add_argument('--epoch_save', type=int, default=3)
    parser.add_argument('--iter_size', type=int, default=1)
    parser.add_argument('--show_every', type=int, default=50)
    parser.add_argument('--input_size', type=int, default=384)
    parser.add_argument('--train_on_gpus', type=list, default=[0])
    # do data aug
    parser.add_argument('--do_aug', type=int, default=1)
    parser.add_argument('--do_h_flip', type=int, default=1)
    parser.add_argument('--do_v_flip', type=int, default=1)

    # Train data
    parser.add_argument('--train_root', type=str, default='./data/image_training/')
    parser.add_argument('--train_list', type=str, default='./data/image_training/train.lst')

    # Testing settings
    parser.add_argument('--model', type=str, default='results/run-0/models/final.pth') # Snapshot
    parser.add_argument('--test_fold', type=str, default='results/R2Net/ARMI/') # Test results saving folder
    parser.add_argument('--sal_mode', type=str, default='duts') # Test image dataset
    parser.add_argument('--test_on_gpus', type=list, default=[0])

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()
        # Get test set info
    test_root, test_list = get_test_info(config.sal_mode)
    config.test_root = test_root
    config.test_list = test_list

    main(config)
