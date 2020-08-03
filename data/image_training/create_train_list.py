import os
import cv2
import random

im_root = 'DUTS-TR-Image/'
gt_root = 'DUTS-TR-Mask/'
file = open('train.lst', 'w+')

im_names = os.listdir(gt_root)
im_names.sort()
    
for im_name in im_names:
    print(im_name)

    file.write(im_root + ' ' + gt_root + ' ' + im_name[:-4])
       
       
    file.write('\r\n')
        
        
file.close()
