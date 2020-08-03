import os
import cv2
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, default='DUTS-TE')
config = parser.parse_args()
save_root = config.dataset_name + '/'
im_root = save_root + config.dataset_name + '-Image/'

im_names = os.listdir(im_root)
file = open( save_root + config.dataset_name + '_test.lst', 'w+')

for i in range(0, len(im_names)):
    print( config.dataset_name + '/' + im_names[i])

    file.write(im_names[i])
    file.write('\r\n')
file.close()
