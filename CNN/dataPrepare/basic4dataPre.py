# @Date:   2018-11-04T13:39:47+01:00
# @Email:  chunping.qiu@tum.de
# @Last modified time: 2019-01-02T19:01:13+01:00

import sys
import numpy as np
import time

# from keras.optimizers import Nadam
# from keras.callbacks import EarlyStopping
# from keras.callbacks import ModelCheckpoint
# from keras.callbacks import ReduceLROnPlateau
# from keras.models import load_model
import h5py
import os
import glob2
import scipy.io as sio
from scipy import stats
import scipy.ndimage
import numpy.matlib
from numpy import argmax
from tensorflow.keras.utils import to_categorical
np.random.seed(5)

def augmentation_(x):
    #x_Aug1=np.empty((len(idx4Aug), x.shape[1], x.shape[2], x.shape[3] ))
    x_Aug2=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug3=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug4=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug5=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug1=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)

    for idx in np.arange(x.shape[0]):
                   original_image=x[idx,:,:,:];
          #print(original_image)
                   x_Aug1[idx,:,:,:]=original_image[:, ::-1, :]
                   x_Aug5[idx,:,:,:]=original_image[::-1, :, :]#Vertical flip
                   x_Aug2[idx,:,:,:]=np.rot90(original_image, 2)
                   x_Aug3[idx,:,:,:]=np.rot90(original_image)
                   x_Aug4[idx,:,:,:]=np.rot90(original_image, 3)

    Xaug=np.concatenate((x_Aug1, x_Aug2), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug3), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug4), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug5), axis=0);
    #Xaug=np.concatenate((Xaug, x), axis=0);

    return Xaug


def augmentation(x, y):
    #x_Aug1=np.empty((len(idx4Aug), x.shape[1], x.shape[2], x.shape[3] ))
    x_Aug2=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug3=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug4=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug5=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug1=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)

    y_Aug2=np.empty((x.shape[0], x.shape[1], x.shape[2], y.shape[3] ), dtype=y.dtype)
    y_Aug3=np.empty((x.shape[0], x.shape[1], x.shape[2], y.shape[3] ), dtype=y.dtype)
    y_Aug4=np.empty((x.shape[0], x.shape[1], x.shape[2], y.shape[3] ), dtype=y.dtype)
    y_Aug5=np.empty((x.shape[0], x.shape[1], x.shape[2], y.shape[3] ), dtype=y.dtype)
    y_Aug1=np.empty((x.shape[0], x.shape[1], x.shape[2], y.shape[3] ), dtype=y.dtype)

    for idx in np.arange(x.shape[0]):
                   original_image=x[idx,:,:,:];
          #print(original_image)
                   x_Aug1[idx,:,:,:]=original_image[:, ::-1, :]
                   x_Aug5[idx,:,:,:]=original_image[::-1, :, :]#Vertical flip

                   x_Aug2[idx,:,:,:]=np.rot90(original_image, 2)
                   x_Aug3[idx,:,:,:]=np.rot90(original_image)
                   x_Aug4[idx,:,:,:]=np.rot90(original_image, 3)

                   original_image=y[idx,:,:,:];
                   y_Aug2[idx,:,:,:]=np.rot90(original_image, 2)
                   y_Aug3[idx,:,:,:]=np.rot90(original_image)
                   y_Aug4[idx,:,:,:]=np.rot90(original_image, 3)

                   y_Aug1[idx,:,:,:]=original_image[:, ::-1, :]
                   y_Aug5[idx,:,:,:]=original_image[::-1, :, :]#Vertical flip

    Xaug=np.concatenate((x_Aug1, x_Aug2), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug3), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug4), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug5), axis=0);
    #Xaug=np.concatenate((Xaug, x), axis=0);

    Yaug=np.concatenate((y_Aug1, y_Aug2), axis=0);
    Yaug=np.concatenate((Yaug, y_Aug3), axis=0);
    Yaug=np.concatenate((Yaug, y_Aug4), axis=0);
    Yaug=np.concatenate((Yaug, y_Aug5), axis=0);
    #Yaug=np.concatenate((Yaug, y), axis=0);

    return Xaug, Yaug

def augmentation_vary(x, y):
    #x_Aug1=np.empty((len(idx4Aug), x.shape[1], x.shape[2], x.shape[3] ))
    x_Aug2=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug3=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug4=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug5=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug1=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)

    y_Aug2=np.empty((y.shape[0], y.shape[1], y.shape[2], y.shape[3] ), dtype=y.dtype)
    y_Aug3=np.empty((y.shape[0], y.shape[1], y.shape[2], y.shape[3] ), dtype=y.dtype)
    y_Aug4=np.empty((y.shape[0], y.shape[1], y.shape[2], y.shape[3] ), dtype=y.dtype)
    y_Aug5=np.empty((y.shape[0], y.shape[1], y.shape[2], y.shape[3] ), dtype=y.dtype)
    y_Aug1=np.empty((y.shape[0], y.shape[1], y.shape[2], y.shape[3] ), dtype=y.dtype)

    for idx in np.arange(x.shape[0]):
                   original_image=x[idx,:,:,:];
          #print(original_image)
                   x_Aug1[idx,:,:,:]=original_image[:, ::-1, :]
                   x_Aug5[idx,:,:,:]=original_image[::-1, :, :]#Vertical flip

                   x_Aug2[idx,:,:,:]=np.rot90(original_image, 2)
                   x_Aug3[idx,:,:,:]=np.rot90(original_image)
                   x_Aug4[idx,:,:,:]=np.rot90(original_image, 3)

                   original_image=y[idx,:,:,:];
                   y_Aug2[idx,:,:,:]=np.rot90(original_image, 2)
                   y_Aug3[idx,:,:,:]=np.rot90(original_image)
                   y_Aug4[idx,:,:,:]=np.rot90(original_image, 3)

                   y_Aug1[idx,:,:,:]=original_image[:, ::-1, :]
                   y_Aug5[idx,:,:,:]=original_image[::-1, :, :]#Vertical flip

    Xaug=np.concatenate((x_Aug1, x_Aug2), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug3), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug4), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug5), axis=0);
    #Xaug=np.concatenate((Xaug, x), axis=0);

    Yaug=np.concatenate((y_Aug1, y_Aug2), axis=0);
    Yaug=np.concatenate((Yaug, y_Aug3), axis=0);
    Yaug=np.concatenate((Yaug, y_Aug4), axis=0);
    Yaug=np.concatenate((Yaug, y_Aug5), axis=0);
    #Yaug=np.concatenate((Yaug, y), axis=0);

    return Xaug, Yaug


def augmentation_vary2(x, y, p):
    #x_Aug1=np.empty((len(idx4Aug), x.shape[1], x.shape[2], x.shape[3] ))
    x_Aug2=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug3=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug4=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug5=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug1=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)

    y_Aug2=np.empty((y.shape[0], y.shape[1], y.shape[2], y.shape[3] ), dtype=y.dtype)
    y_Aug3=np.empty((y.shape[0], y.shape[1], y.shape[2], y.shape[3] ), dtype=y.dtype)
    y_Aug4=np.empty((y.shape[0], y.shape[1], y.shape[2], y.shape[3] ), dtype=y.dtype)
    y_Aug5=np.empty((y.shape[0], y.shape[1], y.shape[2], y.shape[3] ), dtype=y.dtype)
    y_Aug1=np.empty((y.shape[0], y.shape[1], y.shape[2], y.shape[3] ), dtype=y.dtype)

    for idx in np.arange(x.shape[0]):
                   original_image=x[idx,:,:,:];
          #print(original_image)
                   x_Aug1[idx,:,:,:]=original_image[:, ::-1, :]
                   x_Aug5[idx,:,:,:]=original_image[::-1, :, :]#Vertical flip

                   x_Aug2[idx,:,:,:]=np.rot90(original_image, 2)
                   x_Aug3[idx,:,:,:]=np.rot90(original_image)
                   x_Aug4[idx,:,:,:]=np.rot90(original_image, 3)

                   original_image=y[idx,:,:,:];
                   y_Aug2[idx,:,:,:]=np.rot90(original_image, 2)
                   y_Aug3[idx,:,:,:]=np.rot90(original_image)
                   y_Aug4[idx,:,:,:]=np.rot90(original_image, 3)

                   y_Aug1[idx,:,:,:]=original_image[:, ::-1, :]
                   y_Aug5[idx,:,:,:]=original_image[::-1, :, :]#Vertical flip

    Xaug=np.concatenate((x_Aug1, x_Aug2), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug3), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug4), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug5), axis=0);
    Xaug=np.concatenate((Xaug, x), axis=0);

    Yaug=np.concatenate((y_Aug1, y_Aug2), axis=0);
    Yaug=np.concatenate((Yaug, y_Aug3), axis=0);
    Yaug=np.concatenate((Yaug, y_Aug4), axis=0);
    Yaug=np.concatenate((Yaug, y_Aug5), axis=0);
    Yaug=np.concatenate((Yaug, y), axis=0);

    #take only 60% (delete 40%)
    indexRandom=np.arange(Yaug.shape[0])
    np.random.shuffle(indexRandom)
    deleteIdx=indexRandom[:int(round(Yaug.shape[0]*(1-p)))] #p=0.6
    
    Yaug = np.delete(Yaug, deleteIdx, axis=0)
    Xaug = np.delete(Xaug, deleteIdx, axis=0)
    return Xaug, Yaug

def augmentation_vary2else(x, y, p):
    #x_Aug1=np.empty((len(idx4Aug), x.shape[1], x.shape[2], x.shape[3] ))
    x_Aug2=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug3=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug4=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug5=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug1=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)

    y_Aug2=np.empty((y.shape[0], y.shape[1], y.shape[2], y.shape[3] ), dtype=y.dtype)
    y_Aug3=np.empty((y.shape[0], y.shape[1], y.shape[2], y.shape[3] ), dtype=y.dtype)
    y_Aug4=np.empty((y.shape[0], y.shape[1], y.shape[2], y.shape[3] ), dtype=y.dtype)
    y_Aug5=np.empty((y.shape[0], y.shape[1], y.shape[2], y.shape[3] ), dtype=y.dtype)
    y_Aug1=np.empty((y.shape[0], y.shape[1], y.shape[2], y.shape[3] ), dtype=y.dtype)

    for idx in np.arange(x.shape[0]):
                   original_image=x[idx,:,:,:];
          #print(original_image)
                   x_Aug1[idx,:,:,:]=original_image[:, ::-1, :]
                   x_Aug5[idx,:,:,:]=original_image[::-1, :, :]#Vertical flip

                   x_Aug2[idx,:,:,:]=np.rot90(original_image, 2)
                   x_Aug3[idx,:,:,:]=np.rot90(original_image)
                   x_Aug4[idx,:,:,:]=np.rot90(original_image, 3)

                   original_image=y[idx,:,:,:];
                   y_Aug2[idx,:,:,:]=np.rot90(original_image, 2)
                   y_Aug3[idx,:,:,:]=np.rot90(original_image)
                   y_Aug4[idx,:,:,:]=np.rot90(original_image, 3)

                   y_Aug1[idx,:,:,:]=original_image[:, ::-1, :]
                   y_Aug5[idx,:,:,:]=original_image[::-1, :, :]#Vertical flip

    Xaug=np.concatenate((x_Aug1, x_Aug2), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug3), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug4), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug5), axis=0);
    #Xaug=np.concatenate((Xaug, x), axis=0);

    Yaug=np.concatenate((y_Aug1, y_Aug2), axis=0);
    Yaug=np.concatenate((Yaug, y_Aug3), axis=0);
    Yaug=np.concatenate((Yaug, y_Aug4), axis=0);
    Yaug=np.concatenate((Yaug, y_Aug5), axis=0);
    #Yaug=np.concatenate((Yaug, y), axis=0);

    #take only 60% (delete 40%)
    indexRandom=np.arange(Yaug.shape[0])
    np.random.shuffle(indexRandom)
    deleteIdx=indexRandom[:int(round(Yaug.shape[0]*(1-p)))] #p=0.6
    
    Yaug = np.delete(Yaug, deleteIdx, axis=0)
    Xaug = np.delete(Xaug, deleteIdx, axis=0)
    return Xaug, Yaug



def augmentation_r3(x, y):
    x_Aug2=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug3=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug4=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)

    y_Aug2=np.empty((x.shape[0], x.shape[1], x.shape[2], y.shape[3] ), dtype=y.dtype)
    y_Aug3=np.empty((x.shape[0], x.shape[1], x.shape[2], y.shape[3] ), dtype=y.dtype)
    y_Aug4=np.empty((x.shape[0], x.shape[1], x.shape[2], y.shape[3] ), dtype=y.dtype)

    for idx in np.arange(x.shape[0]):
                   original_image=x[idx,:,:,:];
                   x_Aug2[idx,:,:,:]=np.rot90(original_image, 2)
                   x_Aug3[idx,:,:,:]=np.rot90(original_image)
                   x_Aug4[idx,:,:,:]=np.rot90(original_image, 3)

                   original_image=y[idx,:,:,:];
                   y_Aug2[idx,:,:,:]=np.rot90(original_image, 2)
                   y_Aug3[idx,:,:,:]=np.rot90(original_image)
                   y_Aug4[idx,:,:,:]=np.rot90(original_image, 3)
    Xaug=np.concatenate((x_Aug2, x_Aug3), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug4), axis=0);
    Yaug=np.concatenate((y_Aug2, y_Aug3), axis=0);
    Yaug=np.concatenate((Yaug, y_Aug4), axis=0);

    return Xaug, Yaug

def aug(x, y):

    b=np.sum(y[:,:,:,0], axis=1);
    #print(np.sum(b, axis=1).reshape(y.shape[0],1))
    percentageB=np.sum(b, axis=1).reshape(y.shape[0],1)
    #print(np.sum(a[:,:,:,0], axis=1))
    #idx4Aug=np.where(y==0)[0] ;

    #print(percentageB, x.shape[2]*x.shape[1]*0.5)
    print(np.max(percentageB))
    idx_densB=np.where(percentageB>(x.shape[2]*x.shape[1]*0.1))
    print('dense osm patches', idx_densB[0].shape[0])

    if idx_densB[0].shape[0]>0:
        idx = idx_densB[0]#.reshape((-1,1))
        xA, yA=augmentation(x[idx, :,:,:], y[idx, :,:,:])
        print('augmentation ed ', yA.shape[0])

        # x=np.concatenate((xA, x), axis=0);
        # y=np.concatenate((yA, y), axis=0);
        x=np.concatenate((xA, x[idx, :,:,:]), axis=0);
        y=np.concatenate((yA, y[idx, :,:,:]), axis=0);


    indexRandom=np.arange(y.shape[0])#.reshape(y_tra.shape[0],1)
    np.random.shuffle(indexRandom)
    y = y[indexRandom ,:]
    x = x[indexRandom ,:]
    #del indexRandom

    return x, y, np.int64( idx_densB[0].shape[0] )

def augmentation4classi(x, y):
    #x_Aug1=np.empty((len(idx4Aug), x.shape[1], x.shape[2], x.shape[3] ))
    x_Aug2=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug3=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug4=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug5=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)
    x_Aug1=np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3] ), dtype=x.dtype)

    # y_Aug2=np.empty((x.shape[0], 1 ), dtype=y.dtype)
    # y_Aug3=np.empty((x.shape[0], 1 ), dtype=y.dtype)
    # y_Aug4=np.empty((x.shape[0], 1 ), dtype=y.dtype)
    # y_Aug5=np.empty((x.shape[0], 1 ), dtype=y.dtype)
    # y_Aug1=np.empty((x.shape[0], 1 ), dtype=y.dtype)

    for idx in np.arange(x.shape[0]):
                   original_image=x[idx,:,:,:];
          #print(original_image)
                   x_Aug1[idx,:,:,:]=original_image[:, ::-1, :]
                   x_Aug5[idx,:,:,:]=original_image[::-1, :, :]#Vertical flip

                   x_Aug2[idx,:,:,:]=np.rot90(original_image, 2)
                   x_Aug3[idx,:,:,:]=np.rot90(original_image)
                   x_Aug4[idx,:,:,:]=np.rot90(original_image, 3)

                   # original_image=y[idx,:,:,:];
                   # y_Aug2[idx,:,:,:]=np.rot90(original_image, 2)
                   # y_Aug3[idx,:,:,:]=np.rot90(original_image)
                   # y_Aug4[idx,:,:,:]=np.rot90(original_image, 3)
                   #
                   # y_Aug1[idx,:,:,:]=original_image[:, ::-1, :]
                   # y_Aug5[idx,:,:,:]=original_image[::-1, :, :]#Vertical flip

    Xaug=np.concatenate((x_Aug1, x_Aug2), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug3), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug4), axis=0);
    Xaug=np.concatenate((Xaug, x_Aug5), axis=0);
    #Xaug=np.concatenate((Xaug, x), axis=0);

    Yaug=np.concatenate((y, y), axis=0);
    Yaug=np.concatenate((Yaug, y), axis=0);
    Yaug=np.concatenate((Yaug, y), axis=0);
    Yaug=np.concatenate((Yaug, y), axis=0);
    #Yaug=np.concatenate((Yaug, y), axis=0);

    return Xaug, Yaug


def aug4classi(x, y):

    idx_B=np.where(y.argmax(axis=1)==1)
    print('dense osm patches', idx_B[0].shape[0])

    if idx_B[0].shape[0]==0:
        print('no building class in this row! no aug')
        return x,y,0


    idx = idx_B[0]#.reshape((-1,1))
    xA, yA=augmentation4classi(x[idx, :,:,:], y[idx, :])
    print('augmentation ed ', yA.shape[0])

    x=np.concatenate((xA, x), axis=0);
    y=np.concatenate((yA, y), axis=0);

    indexRandom=np.arange(y.shape[0])#.reshape(y_tra.shape[0],1)
    np.random.shuffle(indexRandom)
    y = y[indexRandom ,:]
    x = x[indexRandom ,:]
    #del indexRandom

    return x, y, np.int64(idx_B[0].shape[0] )

def patch2label(y, startPixel, endPixel):

    label = np.ones((y.shape[0], 1), dtype=np.int8)

    b=np.sum(y[:, startPixel:endPixel, startPixel:endPixel, 0], axis=1);
    percentageB=np.sum(b, axis=1).reshape(y.shape[0],1)

    idx_densB=np.where(percentageB==0)
    idx = idx_densB[0]#.reshape((-1,1))

    label[idx, 0]=0

    label_hot = np.ones((y.shape[0], 1), dtype=np.int8)
    label_hot = np.concatenate((np.int8(label_hot-label), label), axis=1);


    return label, label_hot

def patch2labelInx(y, startPixel, endPixel, sumEqual2):

    #label = np.ones((y.shape[0], 1), dtype=np.int8)

    b=np.sum(y[:, startPixel:endPixel, startPixel:endPixel, 0], axis=1);
    percentageB=np.sum(b, axis=1).reshape(y.shape[0],1)

    idx_densB=np.where(percentageB==sumEqual2)
    idx = idx_densB[0]#.reshape((-1,1))
    #
    # label[idx, 0]=0
    #
    # label_hot = np.ones((y.shape[0], 1), dtype=np.int8)
    # label_hot = np.concatenate((np.int8(label_hot-label), label), axis=1);

    return idx

def patch2labelInx_gt(y, startPixel, endPixel, lgValue):

    #label = np.ones((y.shape[0], 1), dtype=np.int8)

    b=np.sum(y[:, startPixel:endPixel, startPixel:endPixel, 0], axis=1);
    percentageB=np.sum(b, axis=1).reshape(y.shape[0],1)

    idx_densB=np.where(percentageB>lgValue)
    idx = idx_densB[0]#.reshape((-1,1))
    #
    # label[idx, 0]=0
    #
    # label_hot = np.ones((y.shape[0], 1), dtype=np.int8)
    # label_hot = np.concatenate((np.int8(label_hot-label), label), axis=1);

    return idx
def patch2labelInx_max_lt(y, startPixel, endPixel, ltValue):
    b=np.max(y[:, startPixel:endPixel, startPixel:endPixel, 0], axis=1);
    percentageB=np.max(b, axis=1).reshape(y.shape[0],1)

    idx_densB=np.where(percentageB<ltValue)
    idx = idx_densB[0]#.reshape((-1,1))
    return idx


def patch2labelInx_lt(y, startPixel, endPixel, ltValue):

    #label = np.ones((y.shape[0], 1), dtype=np.int8)

    b=np.sum(y[:, startPixel:endPixel, startPixel:endPixel, 0], axis=1);
    percentageB=np.sum(b, axis=1).reshape(y.shape[0],1)

    idx_densB=np.where(percentageB<ltValue)
    idx = idx_densB[0]#.reshape((-1,1))
    #
    # label[idx, 0]=0
    #
    # label_hot = np.ones((y.shape[0], 1), dtype=np.int8)
    # label_hot = np.concatenate((np.int8(label_hot-label), label), axis=1);

    return idx

def patch2labelInx_between(y, startPixel, endPixel, egtValue, ltValue):

    #label = np.ones((y.shape[0], 1), dtype=np.int8)

    b=np.sum(y[:, startPixel:endPixel, startPixel:endPixel, 0], axis=1);
    percentageB=np.sum(b, axis=1).reshape(y.shape[0],1)

    idx_densB=np.where((percentageB<ltValue) & (percentageB>=egtValue))
    idx = idx_densB[0]#.reshape((-1,1))

    return idx

def patch2ValInx_stratified(y, startPixel, endPixel, breaks,val_p): #percentage of validation data

    #label = np.ones((y.shape[0], 1), dtype=np.int8)

    b=np.sum(y[:, startPixel:endPixel, startPixel:endPixel, 0], axis=1);
    #print("b.shape",b.shape)
    percentageB=np.sum(b, axis=1).reshape(y.shape[0],1)
    #print("percentageB",percentageB)
    #print("percentageB.shape",percentageB.shape)
    idx_val=np.array([], dtype=np.int64)
    fre_str=np.zeros((len(breaks)-1), dtype= np.int64)
    
    for i in np.arange(len(breaks)-1):
        idx_temp=np.where((percentageB>=breaks[i]) & (percentageB<breaks[i+1]))
        idx_temp=idx_temp[0]
        #print("idx_temp",idx_temp)
        fre_str[i]=len(idx_temp)
        #print("len(idx_temp)",len(idx_temp))
        np.random.shuffle(idx_temp)
        idx_temp=idx_temp[:int(round(len(idx_temp)*(val_p)))]
        idx_val=np.concatenate((idx_val,idx_temp),axis=0)
        #print("idx_val",idx_val)
        
    print("fre_str",fre_str)
    indexRandom=np.arange(idx_val.shape[0])
    np.random.shuffle(indexRandom)
    idx_val=idx_val[indexRandom]
    print("idx_val shuffle",idx_val)
    
    #idx = idx_val[0]#.reshape((-1,1))
    #print("idx_val.shape",idx_val.shape)
    return idx_val


def savePatch(patch, patchY, saveFolder, saveFolderV):

    patchNum_val = np.zeros(3, dtype=np.int64).reshape((1,3)) ;
    patchNum_tra = np.zeros(3, dtype=np.int64).reshape((1,3)) ;

#based on the central 8*8, assign the label
    label, label_hot=patch2label(patchY, 13, 21)
    del patchY

    idx_nB=np.where(label==0)
    idx = idx_nB[0]#.reshape((-1,1))
    num_B=label_hot.shape[0]-idx.shape[0]

    if num_B==0:
        print('no building class in this row!')
        deleteNum=np.int64( idx.shape[0]*0.9 )
        np.random.shuffle(idx)
        deleteIdx=idx[0:deleteNum]

        patch = np.delete(patch, deleteIdx, axis=0)
        label_hot = np.delete(label_hot, deleteIdx, axis=0)
    elif idx.shape[0] > (num_B*6) :#randomly delete class 0, to make sure that the ratio is less than 1:6

        print('idx.shape[0] > (num_B*6)')
        deleteNum=np.int64( idx.shape[0]-num_B*6 )
        np.random.shuffle(idx)
        deleteIdx=idx[0:deleteNum]

        patch = np.delete(patch, deleteIdx, axis=0)
        label_hot = np.delete(label_hot, deleteIdx, axis=0)
    else:
        print('no delete in this row!')


#randomly choose 20% for validation
    num=patch.shape[0]
    numT=np.int64( np.round(num*0.6) )
    numV=np.int64( num-numT )
    indexRandom=np.arange(num, dtype=np.int64)#.reshape(y_tra.shape[0],1)
    np.random.shuffle(indexRandom)
    patch_T = patch[indexRandom[0:(numT)], :, :, :]
    label_T = label_hot[indexRandom[0:(numT)], :]

    patch_V = patch[indexRandom[(numT):], :, :, :]
    label_V = label_hot[indexRandom[(numT):], :]

    #print(num, numT, numV)
    #print(patch_T.shape[0], patch_V.shape[0])

    x,y,denseN=aug4classi(patch_T, label_T)
    patchNum_tra[0,0]=numT-denseN
    patchNum_tra[0,1]=denseN
    patchNum_tra[0,2]=x.shape[0]

    #print('print(np.unique(y))',np.unique(y), y.dtype)
    hf = h5py.File(saveFolder, 'w')
    hf.create_dataset('x', data=x)
    hf.create_dataset('y', data=y)
    hf.close()
    del x
    del y

    x,y,denseN=aug4classi(patch_V, label_V)
    patchNum_val[0,0]=numV-denseN
    patchNum_val[0,1]=denseN
    patchNum_val[0,2]=x.shape[0]

    #print('print(np.unique(y))',np.unique(y), y.dtype)
    hf = h5py.File(saveFolderV, 'w')
    hf.create_dataset('x', data=x)
    hf.create_dataset('y', data=y)
    hf.close()

    return patchNum_tra, patchNum_val

def saveTra_Val(c2_x, y0, traRatio, saveFolder, saveFolderV):

    c2_y=np.matlib.repmat(y0, c2_x.shape[0], 1)

    num=c2_x.shape[0]
    numT=np.int64( np.round(num*traRatio) )
    numV=np.int64( num-numT )
    indexRandom=np.arange(num, dtype=np.int64)#.reshape(y_tra.shape[0],1)
    np.random.shuffle(indexRandom)

    # saveFolder=foldS+'train/'+cities[idCity]
    # if not os.path.exists(saveFolder):
    #     os.makedirs(saveFolder)
    #saveFolder=saveFolder +'/'+cities[idCity]+'_2'+'.h5'

    hf = h5py.File(saveFolder, 'w')
    hf.create_dataset('x', data= c2_x[indexRandom[0:(numT)], :, :, :])
    hf.create_dataset('y', data= c2_y[indexRandom[0:(numT)], :])
    hf.close()

    # saveFolderV=foldS+'vali/'+cities[idCity]
    # if not os.path.exists(saveFolderV):
    #     os.makedirs(saveFolderV)
    #saveFolderV=saveFolderV +'/'+cities[idCity]+'_2'+'.h5'
    hf = h5py.File(saveFolderV, 'w')
    hf.create_dataset('x', data= c2_x[indexRandom[(numT):], :, :, :])
    hf.create_dataset('y', data= c2_y[indexRandom[(numT):], :])
    hf.close()

'''
      # save data for training  and validation
      # input:
              patchX: n*patchShape_x*patchShape_y*channels
              patchY: n*patchShape_x*patchShape_y*num_classes
              traRatio: ratio to split the training data, 0.8 is randomly 0.8 data for training
              saveFolder: file name to save training data
              saveFolderV: file name to save validation data
      # output:
              numT: number of patches of training
              numV: number of patches of validation
'''
def saveTra_Val_patch_(patchX, patchY, traRatio, saveFolder, saveFolderV):

    num=patchX.shape[0]
    numT=np.int64( np.round(num*traRatio) )
    numV=np.int64( num-numT )
    indexRandom=np.arange(num, dtype=np.int64)#.reshape(y_tra.shape[0],1)
    np.random.shuffle(indexRandom)

    hf = h5py.File(saveFolder, 'w')
    hf.create_dataset('x', data= patchX[indexRandom[0:(numT)], :, :, :])
    hf.create_dataset('y', data= patchY[indexRandom[0:(numT)], :, :, :])
    hf.close()

    hf = h5py.File(saveFolderV, 'w')
    hf.create_dataset('x', data= patchX[indexRandom[(numT):], :, :, :])
    hf.create_dataset('y', data= patchY[indexRandom[(numT):], :, :, :])
    hf.close()

    return numT, numV

def saveTra_Val_patch_1(patchX, patch_dense_1, patch_dense_2, patch_dense_3, patchY, traRatio, saveFolder, saveFolderV):
    num=patchX.shape[0]
    numT=np.int64( np.round(num*traRatio) )
    numV=np.int64( num-numT )
    indexRandom=np.arange(num, dtype=np.int64)#.reshape(y_tra.shape[0],1)
    np.random.shuffle(indexRandom)

	# patchX = patchX[indexRandom ,:]
	# patchY = patchY[indexRandom ,:]
	# patch_dense_1 = patch_dense_1[indexRandom ,:]
	# patch_dense_2 = patch_dense_2[indexRandom ,:]
	# patch_dense_3 = patch_dense_3[indexRandom ,:]
	# del indexRandom
    save2MultipleFile(patchX[indexRandom[0:(numT)], :, :, :], patch_dense_1[indexRandom[0:(numT)], :, :, :], patch_dense_2[indexRandom[0:(numT)], :, :, :], patch_dense_3[indexRandom[0:(numT)], :, :, :], patchY[indexRandom[0:(numT)], :, :, :], saveFolder)
    save2MultipleFile(patchX[indexRandom[(numT):], :, :, :], patch_dense_1[indexRandom[(numT):], :, :, :], patch_dense_2[indexRandom[(numT):], :, :, :], patch_dense_3[indexRandom[(numT):], :, :, :], patchY[indexRandom[(numT):], :, :, :], saveFolderV)

    return numT, numV

def save2MultipleFile(patchX, patch_dense_1, patch_dense_2, patch_dense_3, patchY, saveFolder, numFile=100):
    num=patchX.shape[0]
    numEach=np.int64( np.round(num*1/numFile) )

    idxStart=0
    for f in np.arange(numFile):
        file=saveFolder+'_'+str(f)+'.h5'
        hf = h5py.File(file, 'w')

        if f==numFile:
            hf.create_dataset('x_0', data= patchX[idxStart:, :, :, :])
            hf.create_dataset('x_1', data= patch_dense_1[idxStart:, :, :, :])
            hf.create_dataset('x', data= patch_dense_2[idxStart:, :, :, :])
            hf.create_dataset('x_3', data= patch_dense_3[idxStart:, :, :, :])
            hf.create_dataset('y', data= patchY[idxStart:, :, :, :])

        else:
            hf.create_dataset('x_0', data= patchX[idxStart:(idxStart+numEach), :, :, :])
            hf.create_dataset('x_1', data= patch_dense_1[idxStart:(idxStart+numEach), :, :, :])
            hf.create_dataset('x', data= patch_dense_2[idxStart:(idxStart+numEach), :, :, :])
            hf.create_dataset('x_3', data= patch_dense_3[idxStart:(idxStart+numEach), :, :, :])
            hf.create_dataset('y', data= patchY[idxStart:(idxStart+numEach), :, :, :])

        hf.close()

        idxStart=idxStart+numEach

#save files into 10 fold
def savePatch_fold(patchX, patchLCZ, saveFolder, city):
    num=patchX.shape[0]
    numV=np.int64( np.round(num*0.05) )
    indexRandom=np.arange(num, dtype=np.int64)#.reshape(y_tra.shape[0],1)
    np.random.shuffle(indexRandom)

    save2MultipleFile_(patchX[indexRandom, :, :, :], patchLCZ[indexRandom, :, :, :], saveFolder, city)

    return num-numV, numV

def savePatch_fold_single(patchX, patchLCZ, saveFolder, city):
    num=patchX.shape[0]
    #numV=np.int64( np.round(num*0.05) )
    indexRandom=np.arange(num, dtype=np.int64)#.reshape(y_tra.shape[0],1)
    np.random.shuffle(indexRandom)

    save2MultipleFile_single(patchX[indexRandom, :, :, :], patchLCZ[indexRandom, :, :, :], saveFolder, city, numFile=20) #change here h5 number

    return num

def savePatch_fold_singlev(patchX, patchLCZ, saveFolder, city):
    num=patchX.shape[0]
    #numV=np.int64( np.round(num*0.05) )
    indexRandom=np.arange(num, dtype=np.int64)#.reshape(y_tra.shape[0],1)
    np.random.shuffle(indexRandom)

    save2MultipleFile_single(patchX[indexRandom, :, :, :], patchLCZ[indexRandom, :, :, :], saveFolder, city, numFile=25) #change here h5 number

    return num



def save2MultipleFile_(patchX, patchLCZ, saveFolder, city, numFile=100):
    num=patchX.shape[0]
    numEach=np.int64( np.round(num*1/numFile) )

    idxStart=0
    for f in np.arange(numFile):

        if f<5:
            file=saveFolder+'vali/'+city+'_'+str(f)+'.h5'
        else:
            file=saveFolder+'trai/'+city+'_'+str(f)+'.h5'

        hf = h5py.File(file, 'w')

        if f==numFile:

            hf.create_dataset('x', data= patchX[idxStart:, :, :, :])
            hf.create_dataset('y', data= patchLCZ[idxStart:, :, :, :])
        else:
            hf.create_dataset('x', data= patchX[idxStart:(idxStart+numEach), :, :, :])
            hf.create_dataset('y', data= patchLCZ[idxStart:(idxStart+numEach), :, :, :])

        hf.close()
        idxStart=idxStart+numEach

def save2MultipleFile_single(patchX, patchLCZ, saveFolder, city, numFile=25):
    num=patchX.shape[0]
    numEach=np.int64( np.round(num*1/numFile) )

    idxStart=0
    for f in np.arange(numFile):

        #if f<5:
        #    file=saveFolder+'vali/'+city+'_'+str(f)+'.h5'
        #else:
        #    file=saveFolder+'trai/'+city+'_'+str(f)+'.h5'

        file=saveFolder+city+'_'+str(f)+'.h5'
        hf = h5py.File(file, 'w')

        if f==numFile:

            hf.create_dataset('x', data= patchX[idxStart:, :, :, :])
            hf.create_dataset('y', data= patchLCZ[idxStart:, :, :, :])
        else:
            hf.create_dataset('x', data= patchX[idxStart:(idxStart+numEach), :, :, :])
            hf.create_dataset('y', data= patchLCZ[idxStart:(idxStart+numEach), :, :, :])

        hf.close()
        idxStart=idxStart+numEach




def saveTra_Val_patch(patchX, patchY, traRatio, tstRatio, saveFolder, saveFolderV, saveFolderT):

    num=patchX.shape[0]
    numT=np.int64( np.round(num*traRatio) )
    numTst=np.int64( np.round(num*tstRatio) )
    numV=np.int64( num-numT-numTst )
    indexRandom=np.arange(num, dtype=np.int64)#.reshape(y_tra.shape[0],1)
    np.random.shuffle(indexRandom)

    hf = h5py.File(saveFolder, 'w')
    hf.create_dataset('x', data= patchX[indexRandom[0:(numT)], :, :, :])
    hf.create_dataset('y', data= patchY[indexRandom[0:(numT)], :, :, :])
    hf.close()

    hf = h5py.File(saveFolderT, 'w')
    hf.create_dataset('x', data= patchX[indexRandom[(numT):(numT+numTst)], :, :, :])
    hf.create_dataset('y', data= patchY[indexRandom[(numT):(numT+numTst)], :, :, :])
    hf.close()

    hf = h5py.File(saveFolderV, 'w')
    hf.create_dataset('x', data= patchX[indexRandom[(numT+numTst):], :, :, :])
    hf.create_dataset('y', data= patchY[indexRandom[(numT+numTst):], :, :, :])
    hf.close()

    return numT, numV

def patchIndex2oneHot(patchY, num_classes):

    num=patchY.shape[0]
    encodedM=np.zeros((patchY.shape[0], patchY.shape[1], patchY.shape[2], num_classes), dtype=np.int8)

    for i in np.arange(num):
        ref0=patchY[i,:,:,:]
        #print(ref0.shape)

        data = ref0.reshape(ref0.shape[0]*ref0.shape[1], 1)
        encoded = to_categorical(data, num_classes=num_classes)
        encoded = np.int8(encoded)
        #print(encoded, encoded.shape)
        #print(argmax(encoded, axis=1))
        ref=encoded.reshape(ref0.shape[0], ref0.shape[1], num_classes)
        #print(np.max(argmax(ref, axis=2)-ref0[:,:,0]))
        #print('ref size', ref.shape, ref.dtype)

        encodedM[i,:,:,:]=ref

    return encodedM
