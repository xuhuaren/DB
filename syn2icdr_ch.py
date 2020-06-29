# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:39:18 2020

@author: xuhuaren
"""

import os
import scipy.io as io
from tqdm import tqdm
import cv2


raw_path = './SynthText'
im_root = './SynthText/images'
txt_root = './SynthText/gts'
list_file = './SynthText/list.txt'


os.makedirs(im_root, exist_ok=True)
os.makedirs(txt_root, exist_ok=True)


print('reading data from {}'.format(raw_path))
gt = io.loadmat(os.path.join(raw_path, 'gt.mat').replace('\\', '/'))
print('Done.')

img_lists = []
for i, imname in enumerate(tqdm(gt['imnames'][0])):
    
    imname = imname[0]
    im_path_read = os.path.join(raw_path, imname)
    
    try:
        img_matrix = cv2.imread(im_path_read)
    except:
        continue
    
    im_path_write = os.path.join(im_root, str(i) + '_en.jpg')
    cv2.imwrite(im_path_write, img_matrix)
    
    txt_path_write = os.path.join(txt_root, str(i) + '_en.jpg.txt')
    
    temp = gt['wordBB'][0,i]

    if len(temp.shape) == 2:
        annots = temp.transpose(1, 0).reshape(-1, 8).astype(int)
    else:
        annots = temp.transpose(2, 1, 0).reshape(-1, 8).astype(int)
        
    if len(temp.shape)>=2:
        img_lists.append(str(i) + '_en.jpg')
        
    contents = []    
    for val in gt['txt'][0,i]:
        v = [x.split("\n") for x in val.strip().split(" ")]
        contents.extend(sum(v, []))  
    contents = [v for v in contents if v!='']
    
    with open(txt_path_write, 'w') as f:
        for ann_idx, annot in enumerate(annots):
            try:
                sub_text = contents[ann_idx]
            except:
                sub_text = '###'
                
                
            str_write = ','.join(annot.astype(str).tolist())
            str_write += "," + sub_text
            f.write(str_write + '\n')
    f.close()
    
    

        
with open(list_file, 'w') as f:
    for img_list in img_lists:
        f.write(img_list + '\n')
f.close()   
     
print('End.')