#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import nibabel as nib
import numpy as np
import pandas as pd
import time
import multiprocessing

def extraction(templet_arr,name,path):
    img = nib.load(path + name + '.nii')
    img = np.array(img.get_fdata())
    for i in range(61):
        for j in range(73):
            for k in range(61):
                if templet_arr[i][j][k]==0:
                    img[i][j][k]*=0
    img_sum = img.sum(axis=0)
    img_sum = img_sum.sum(axis=0)
    img_sum = img_sum.sum(axis=0)
    img_sum = img_sum / (templet_arr.sum() / np.max(templet_arr))
    return img_sum

def process(region):
    global name
    TimeStr = time.asctime(time.localtime(time.time()))
    print(TimeStr + '\t' + name + '\t' + region)
    img = nib.load('/home/zmx/fMRI/Template/area/' + region + '.nii')
    img_arr = np.array(img.get_fdata())
    img_res = extraction(img_arr,name,path)
    varDict[region+'_res'] = list(img_res)

#sub = [1,3,5,6,7,8,9,10,13,14,15,18,21,22,23]
sub = [1]
area = ['Hippocampus_L','Hippocampus_R','Parahippo_L','Parahippo_R','Fusiform_L','Fusiform_R',\
        'Precuneus_L','Precuneus_R','Parietal_Inf_L','Parietal_Inf_R','Parietal_Sup_L','Parietal_Sup_R',\
        'Angular_L','Angular_R','Cingulum_Ant_L','Cingulum_Ant_R','Cingulum_Mid_L','Cingulum_Mid_R',\
        'Cingulum_Post_L','Cingulum_Post_R','Frontal_Inf_Oper_L','Frontal_Inf_Oper_R','Frontal_Inf_Orb_L','Frontal_Inf_Orb_R',\
        'Frontal_Inf_Tri_L','Frontal_Inf_Tri_R','SupraMarginal_L','SupraMarginal_R']
#task = ['loc','mot_1','mot_2','mot_3','prememory','postmemory']
task = ['loc']

varDict = multiprocessing.Manager().dict()

for t in task:
    path = '/home/zmx/ds002311/preprocessed_4D/' + t + '/'
    for i in sub:
        if i < 10:
            name = 'bsub-0' + str(i) + '_' + t
        else:
            name = 'sub-' + str(i) + '_' + t
        
        pool = multiprocessing.Pool(processes = 2)

        for region in area:
            pool.apply_async(process, (region,))

        pool.close()
        pool.join()
        varDict = dict(varDict)
        dataframe = pd.DataFrame(varDict)
        dataframe.to_csv(path + name + ".csv",index=False)