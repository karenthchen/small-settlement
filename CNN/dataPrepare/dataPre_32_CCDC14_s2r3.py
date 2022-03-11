# @Date:   2021-08-03
# @Email:  karen.t.chen@yale.edu
# @Last modified time: 2021-09-15
# 1. use most of patches, including low fraction ones
# 2. regression, augmentation by part
# 3. sparate validation dataset by stratified sampling
# 4. Include reduced amount of snow


import sys
import os
sys.path.append(os.path.abspath('/home/tc768/project/HKH/DL_seg_regression/code/modelPredict'))
from img2mapC05 import img2mapC

import numpy as np
import time
sys.path.append(os.path.abspath('/home/tc768/project/HKH/DL_seg_regression/DL/code/dataPrepare'))
import basic4dataPre
import h5py
import os
import glob2
import scipy.io as sio
from scipy import stats
import scipy.ndimage
import numpy.matlib
from numpy import argmax
from tensorflow.keras.utils import to_categorical
import skimage.measure
import re #re.sub
import glob #glob.glob

#image folder
foldRef='/home/tc768/project/HKH/DL_seg_regression/images/'

#change here
#stride to cut patches
step=16 #change here to be half of patch_shape
nband=24

patch_shape = (32, 32, nband)
#new line
img_shape = (32, 32)

#save folder
foldS='/home/tc768/project/HKH/DL_seg_regression/patch/patch_32_CCDC14_s2r3/'

params = {'dim_x': patch_shape[0],
		   'dim_y': patch_shape[1],
		   'dim_z': patch_shape[2],
		   'step': step,
		   'Bands': list(range(nband)), #two topographical
		   'scale':1.0,
		   'ratio':1,
		   'isSeg':0,
		   'nanValu':0,
		   'dim_x_img': img_shape[0],#the actuall extracted image patch
		   'dim_y_img': img_shape[1]}



labels=sorted(glob.glob(foldRef+"is*_30m*N*.tif"))
predis=sorted(glob.glob(foldRef+"CCDC_c1-4/CCDC*_c30m*N*.tif"))
predis = [ x for x in predis if ("N51" not in x) and ("N52" not in x) ]
labels = [ x for x in labels if ("N51" not in x) and ("N52" not in x) ]
print(len(predis))
print(len(labels))

labelsn = [re.sub(".tif","",os.path.basename(x)) for x in labels] 
predisn = [re.sub(".tif","",os.path.basename(x)) for x in predis] 

print(labelsn)
print(predisn)


#tra and vali patch numbers of each images
patchNum = np.zeros((2,len(labels)), dtype= np.int64) ;
#class number of each class by distribution
classNum = np.zeros((len(labels),11), dtype= np.int64) ; #change here
classNum_val = np.zeros((len(labels),11), dtype= np.int64) ; #change here

if not os.path.exists(foldS+'trai/'):
    os.makedirs(foldS+'trai/')
if not os.path.exists(foldS+'vali/'):
    os.makedirs(foldS+'vali/')
###########training patch#################
for id in np.arange(len(labels)):

	print(labelsn[id])
	params['Bands'] = [0]
	params['scale'] = 1
	img2mapCLass=img2mapC(**params);

	###lcz to patches
	#load  file
	prj0, trans0, ref0= img2mapCLass.loadImgMat(labels[id])
	print('ref0 size', ref0.shape)
	ref = np.double(ref0)
	#print('lcz file size', ref.shape, trans0, ref.dtype)
	# to patches
	patchLCZ, R, C = img2mapCLass.label2patches_all(ref, 1)
	print('lcz patches, beginning', patchLCZ.shape, patchLCZ.dtype)

	#load img
	file = predis[id]
	params['Bands'] = list(range(nband))#change here band number
	params['scale'] = 1.0#
	img2mapCLass=img2mapC(**params);
	prj0, trans0, img_= img2mapCLass.loadImgMat(file)
	print('img size', img_.shape)
	#image to patches
	patch_summer, R, C, idxNan = img2mapCLass.Bands2patches(img_, 1)
	print('image patches', patch_summer.shape, patch_summer.dtype)
        #try not delete idxNan (by Karen)
	print('lcz patches, before delete idxNan', patchLCZ.shape, patchLCZ.dtype)

	patchLCZ = np.delete(patchLCZ, idxNan, axis=0)
	print('lcz patches, after delete idxNan', patchLCZ.shape, patchLCZ.dtype)

	############manipulate the patches############
	#delete patches with low urban fraction
	#change here:one patch has three pixels with urban fraction = 0.1

	c3Idx=basic4dataPre.patch2labelInx_lt(patchLCZ, 0, patchLCZ.shape[1],  0) 
	patchLCZ = np.delete(patchLCZ, c3Idx, axis=0)
	print('lcz patches, after delete noLCZ', patchLCZ.shape, patchLCZ.dtype)
	patch_summer = np.delete(patch_summer, c3Idx, axis=0)
	print('image patches, after delete noLCZ', patch_summer.shape, patch_summer.dtype)

	c3Idx=basic4dataPre.patch2labelInx_between(patchLCZ, 0, patchLCZ.shape[1],  0.1*5,  0.1*20)
	patchLCZ_llUF = patchLCZ[c3Idx,:,:,:] #5-20
	patch_summer_llUF = patch_summer[c3Idx,:,:,:]

	c3Idx=basic4dataPre.patch2labelInx_between(patchLCZ, 0, patchLCZ.shape[1],  0.1*20,  0.1*100)
	patchLCZ_lUF = patchLCZ[c3Idx,:,:,:] #20-100
	patch_summer_lUF = patch_summer[c3Idx,:,:,:]

	c3Idx=basic4dataPre.patch2labelInx_between(patchLCZ, 0, patchLCZ.shape[1],  0.1*100,  0.1*200)
	patchLCZ_hUF = patchLCZ[c3Idx,:,:,:] #100-200
	patch_summer_hUF = patch_summer[c3Idx,:,:,:]

	c3Idx=basic4dataPre.patch2labelInx_lt(patchLCZ, 0, patchLCZ.shape[1],  0.1*200) 
	patchLCZ_hhUF = np.delete(patchLCZ, c3Idx, axis=0) #>200
	patch_summer_hhUF = np.delete(patch_summer, c3Idx, axis=0)

	c3Idx=basic4dataPre.patch2labelInx_between(patchLCZ, 0, patchLCZ.shape[1],  0,  0.1*5) #>=0
	patchLCZ = patchLCZ[c3Idx,:,:,:] #0-5
	patch_summer = patch_summer[c3Idx,:,:,:]


	print('lcz patches, 20-100high fraction', patchLCZ_lUF.shape, patchLCZ_lUF.dtype)
	print('image patches, 20-100high fraction', patch_summer_lUF.shape, patch_summer_lUF.dtype) 
	print('lcz patches, 1-200high fraction', patchLCZ_hUF.shape, patchLCZ_hUF.dtype)
	print('image patches, 1-200high fraction', patch_summer_hUF.shape, patch_summer_hUF.dtype) 
	print('lcz patches, 200high fraction', patchLCZ_hhUF.shape, patchLCZ_hhUF.dtype)
	print('image patches, 200high fraction', patch_summer_hhUF.shape, patch_summer_hhUF.dtype)
   
    #augmentation
	patch_summer, patchLCZ = basic4dataPre.augmentation_vary2(patch_summer, patchLCZ, p=0.1)
	patch_summer_llUF, patchLCZ_llUF = basic4dataPre.augmentation_vary2(patch_summer_llUF, patchLCZ_llUF, p=0.2) 
	patch_summer_lUF, patchLCZ_lUF = basic4dataPre.augmentation_vary2(patch_summer_lUF, patchLCZ_lUF, p=0.5) 
	patch_summer_hUF, patchLCZ_hUF = basic4dataPre.augmentation_vary2(patch_summer_hUF, patchLCZ_hUF, p=0.8) #randomly keep 80% of the augmentation, else means not including self
	patch_summer_hhUF, patchLCZ_hhUF = basic4dataPre.augmentation_vary2(patch_summer_hhUF, patchLCZ_hhUF, p=1) #randomly keep 80% of the augmentation
	print('image patches, 100-200 fractionafter augmentation', patch_summer_hUF.shape, patch_summer_hUF.dtype)
	print('image patches, >200 fractionafter augmentation', patch_summer_hhUF.shape, patch_summer_hhUF.dtype)
	
	patchLCZ=np.concatenate((patchLCZ, patchLCZ_llUF, patchLCZ_lUF, patchLCZ_hUF, patchLCZ_hhUF), axis=0);
	patch_summer=np.concatenate((patch_summer,patch_summer_llUF, patch_summer_lUF, patch_summer_hUF, patch_summer_hhUF), axis=0);
	indexRandom=np.arange(patchLCZ.shape[0])
	np.random.shuffle(indexRandom)
	patchLCZ = patchLCZ[indexRandom,:,:,:]  
	patch_summer = patch_summer[indexRandom,:,:,:]

	print('image patches, after combine', patch_summer.shape, patch_summer.dtype)
	print('lcz patches, after combine', patchLCZ.shape, patchLCZ.dtype)

	ValIdx=basic4dataPre.patch2ValInx_stratified(patchLCZ, 0, patchLCZ.shape[1],  [0.1*20,0.1*50,0.1*100,0.1*150,0.1*200,0.1*250,1*1024],0.25)
    
	patchLCZ_val = patchLCZ[ValIdx,:,:,:] #validation dataset
	patch_summer_val = patch_summer[ValIdx,:,:,:]     
	patchLCZ = np.delete(patchLCZ, ValIdx, axis=0) #training dataset
	patch_summer = np.delete(patch_summer, ValIdx, axis=0)
	print('lcz val patches', patchLCZ_val.shape, patchLCZ_val.dtype)
	print('lcz tra patches', patchLCZ.shape, patchLCZ.dtype)

	#NOT downsample to have a 90m gt
        #keep original 90m because of the new inputs of label has resoluiton at 90m
	#patchLCZ=skimage.measure.block_reduce(patchLCZ, (1,3,3,1), np.mean)
	#patchLCZ=skimage.measure.block_reduce(patchLCZ, (1,1,1,1), np.mean)
	print('downsampled patchHSE:', patchLCZ.shape)

	###statistic of class number
	tmp=patchLCZ.reshape((-1,1))
	tmp2=patchLCZ_val.reshape((-1,1))
	for c in np.arange(0,11): 
		idx_ = np.where((tmp > (c-1)/10)&(tmp <= c/10))
		idx  = idx_[0]
		classNum[id, c]=idx.shape[0]
		idx_ = np.where((tmp2 > (c-1)/10)&(tmp2 <= c/10))
		idx  = idx_[0]
		classNum_val[id, c]=idx.shape[0]
        
	#reset the labels
	#patchLCZ=np.double(patchLCZ); #keep fraction
	print('print(range(patchLCZ))',np.max(patchLCZ.flatten()),np.min(patchLCZ.flatten()))
	print('shape', patchLCZ.shape, patch_summer.shape)

	patchNum_tra =basic4dataPre.savePatch_fold_single(patch_summer, patchLCZ, foldS+'trai/', predisn[id])
	patchNum_val =basic4dataPre.savePatch_fold_single(patch_summer_val, patchLCZ_val, foldS+'vali/', predisn[id])
    
	patchNum[0,id]=patchNum_tra
	patchNum[1,id]=patchNum_val
    
	print(patchNum, classNum)

traNum=np.sum(patchNum[0,:]) ;
valNum=np.sum(patchNum[1,:]) ;
traClass=np.sum(classNum[:len(labels),],axis=0); #change here
valClass=np.sum(classNum_val[:len(labels),],axis=0); #change here
print('total patch size Tra', traNum)
print('total patch size Val', valNum)
print('total class size Tra', traClass)
print('total class size Val', valClass)


sio.savemat((foldS +'patchNum.mat'), {'patchNum': patchNum, 'classNum':classNum})
