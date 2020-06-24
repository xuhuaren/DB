# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:39:18 2020

@author: xuhuaren
"""

import os
from tqdm import tqdm
import cv2
import pymatreader as pm
import numpy as np

im_save = 'train_images'
txt_save = 'train_gts'
list_save = 'train_list.txt'
lists = ['train', 'extra']

os.makedirs(im_save, exist_ok=True)
os.makedirs(txt_save, exist_ok=True)
cnt = 0
img_lists = []
for list_ in lists:
    im_root = os.path.join(list_,'images')
    txt_root = os.path.join(list_,'gts')
    df = pm.read_mat(os.path.join(list_, "digitStruct.mat"))
    bbox = df["digitStruct"]["bbox"]
    name = df["digitStruct"]["name"]

    for i, imname in enumerate(tqdm(name)):
    
        im_path_read = os.path.join(im_root, imname)
        
        try:
            img_matrix = cv2.imread(im_path_read)
        except:
            print("read error")
            continue
        
        cnt += 1
        
        im_path_write = os.path.join(im_save, str(cnt) + '.jpg')
        cv2.imwrite(im_path_write, img_matrix)
        txt_path_write = os.path.join(txt_save, str(cnt) + '.jpg' + '.txt')
        
        label = bbox[i]['label']
        if isinstance(label, float):
            label_str = str(int(label))
        else:
            label = [str(int(ddd)) for ddd in label]
            label_str="".join(label)
            
        h, w = img_matrix.shape[:2]
        left = int(np.min(bbox[i]['left']))
        top = int(np.min(bbox[i]['top']))
        
        if not isinstance(bbox[i]['top'], float) and len(bbox[i]['top']) >= 2:
            if abs(bbox[i]['top'][0] - bbox[i]['top'][1]) < 0.5 * abs(bbox[i]['left'][0] - bbox[i]['left'][1]):
                width = min(int(np.sum(bbox[i]['width'])), int(w - left - 1))        
                height = min(int(np.max(bbox[i]['height'])), int(h - top - 1)) 
            else:
                width = min(int(np.sum(bbox[i]['width'])), int(w - left - 1))        
                height = min(int(np.sum(bbox[i]['height'])), int(h - top - 1)) 
    
        else:
            width = min(int(np.sum(bbox[i]['width'])), int(w - left - 1))        
            height = min(int(np.max(bbox[i]['height'])), int(h - top - 1))              
        
        #left,top, right, top, right, bottom, left, bottom
        annots = [left, top, left+width, top, left+width, top+height, left, top+height]
        annots = [str(ss) for ss in annots]
        if len(annots)>=1:
            img_lists.append(str(cnt) + '.jpg')
            
     
        with open(txt_path_write, 'w') as f:
            str_write = ','.join(annots)
            str_write += "," + label_str
            f.write(str_write + '\n')
        f.close()
    
 
 
with open(list_save, 'w') as f:
    for img_list in img_lists:
        f.write(img_list + '\n')
f.close()   
     
print('End.')
