#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os

sub = [1,3,5,6,7,8,9,10,13,14,15,18,21,22,23]
phase= ['loc','mot_1','mot_2','mot_3','prememory','postmemory']
for index in sub:
	task = 'mrcat'
	n = index
	if index < 10:
		index = '0' + str(index)
	else:
		index = str(index)

	for i in range(1,406):
		if i < 10:
			i = str(i)
			task = task + ' /home/zmx/ds002311/sub-' + index + '/func/mot_3/swrasub-' + index + '_task-mot_run-03_bold_0000'+ i +'.nii '
		elif i < 100:
			i = str(i)
			task = task + ' /home/zmx/ds002311/sub-' + index + '/func/mot_3/swrasub-' + index + '_task-mot_run-03_bold_000'+ i +'.nii '
		else:
			i = str(i)
			task = task + ' /home/zmx/ds002311/sub-' + index + '/func/mot_3/swrasub-' + index + '_task-mot_run-03_bold_00'+ i +'.nii '
	#print(task)		
	task = task + '/home/zmx/ds002311/func_net/sub-' + index + '_mot_3.nii'
	os.system(task)
