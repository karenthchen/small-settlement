# @Date:   2018-07-16T20:51:18+02:00
# @Email:  chunping.qiu@tum.de
# @Last modified time: 2018-12-21T09:17:56+01:00


import sys
import os
import numpy as np
from skimage.util.shape import view_as_windows
import glob
from osgeo import gdal,osr
import glob2
from scipy import stats
import scipy.ndimage
#from memprof import *
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from keras.models import Model
from keras.layers import Input
'load img tif and get patches from it;  save as tif file after getting predictions'
class img2mapC(object):

  def __init__(self, dim_x, dim_y, dim_z, step, Bands, scale, ratio, isSeg, nanValu, dim_x_img, dim_y_img):
	  self.dim_x = dim_x#shape of the patch
	  self.dim_y = dim_y
	  self.dim_z = dim_z
	  self.step = step#lcz resolution (in pixel): step
	  self.Bands = Bands#bands selected from the image files, list
	  self.scale = scale#the number used to divided the pixel value by
	  self.ratio = ratio#the ratio between size of label pixel and size of image pixel
	  self.isSeg = isSeg#whether segmentation
	  self.nanValu = nanValu
	  self.dim_x_img = dim_x_img#the actually extracted image patch, can be different from the patch size input to the network
	  self.dim_y_img = dim_y_img

  '''
    # cut a matrix into patches
    # input:
            imgMat: matrix containing the bands of the image
    # output:
            patch: the patches, one patch is with the shape: dim_x, dim_y, dim_z
            R: the size of the final lcz map
            C: the size of the final lcz map
            idxNan: the index of no data area.
  '''
  def Bands2patches(self, imgMat, upSampleR):

	  print('imgMat', imgMat.shape, imgMat.dtype)
	  for band in np.arange(imgMat.shape[2]):

		  arr = imgMat[:,:,band]
		  if upSampleR!=1:
		            arr=scipy.ndimage.zoom(arr, [upSampleR,  upSampleR], order=1)#Bilinear interpolation would be order=1
		            print('upsampling',arr.shape)

		  patch0, R, C= self.__img2patch(arr)#'from band to patches'
		  #print('patch0', patch0.shape, patch0.dtype)


		  if band==0:

			#find the nodata area
			  patch0Tmp=np.amin(patch0, axis=1);
			#print(b) # axis=1；每行的最小值
			  indica=np.amin(patch0Tmp, axis=1);

			  idxNan = np.where( (indica<0.000001) )
			  idxNan = idxNan[0].reshape((-1,1))
			  print(idxNan.shape)

			  patch=np.zeros(((patch0.shape[0]-idxNan.shape[0]), self.dim_x_img, self.dim_y_img, imgMat.shape[2]), dtype=imgMat.dtype);

		  patch0 = np.delete(patch0, idxNan, axis=0)
		  if self.scale == -1:#scale with a fucntion
			  patch[:,:,:,band]=self.scaleBand(patch0)

		  else:
			  patch[:,:,:,band]=patch0/self.scale ;

	  return patch, R, C, idxNan


# #from a multi bands mat to patches, without considering the nan area
  def Bands2patches_all(self, imgMat, upSampleR):
	  for band in np.arange(imgMat.shape[2]):

		  arr = imgMat[:,:,band]
		  if upSampleR!=1:
		            arr=scipy.ndimage.zoom(arr, [upSampleR,  upSampleR], order=1)#Bilinear interpolation would be order=1
		  patch0, R, C= self.__img2patch(arr)#'from band to patches'

		  if band==0:
			  patch=np.zeros(((patch0.shape[0]), self.dim_x, self.dim_y, imgMat.shape[2]), dtype=imgMat.dtype);

		  #print('self.scale', self.scale)
		  if self.scale == -1:#scale with a fucntion
			  patch[:,:,:,band]=self.scaleBand(patch0)

		  else:
			  patch[:,:,:,band]=patch0/self.scale ;

	  return patch, R, C


  '''
    # load all relevent bands of a image file
    # input:
            imgFile: image file
    # output:
            prj: projection data
            trans: projection data
            matImg: matrix containing the bands of the image
  '''

#change to match the size of data
# #from a label to patches, without considering the nan area
  def label2patches_all(self, imgMat, upSampleR):
	  for band in np.arange(imgMat.shape[2]):
		

		  print('band',np.arange(imgMat.shape[2]))
		  arr = imgMat[:,:,band]
		  patch0, R, C= self.__label2patch(arr)#'from label to patches' #change here

		  if band==0:
			  patch=np.zeros(((patch0.shape[0]), self.dim_x, self.dim_y, imgMat.shape[2]), dtype=imgMat.dtype);

		  #print('self.scale', self.scale)
		  if self.scale == -1:#scale with a fucntion
			  patch[:,:,:,band]=self.scaleBand(patch0)

		  else:
			  patch[:,:,:,band]=patch0/self.scale ;

	  return patch, R, C


  '''
    # load all relevent bands of a image file
    # input:
            imgFile: image file
    # output:
            prj: projection data
            trans: projection data
            matImg: matrix containing the bands of the image

  '''

  def loadImgMat(self, imgFile):
	  src_ds = gdal.Open( imgFile )

	  if src_ds is None:
		  print('Unable to open INPUT.tif')
		  sys.exit(1)
	  prj=src_ds.GetProjection()
	  trans=src_ds.GetGeoTransform()
	  #print("[ RASTER BAND COUNT ]: ", src_ds.RasterCount)
	  #print(prj)
	  #print(trans)

	  #print(self.Bands)

	  bandInd=0
	  print(self.Bands)
	  for band in self.Bands:
		  band += 1
		  srcband = src_ds.GetRasterBand(band)

		  if srcband is None:
			  print('srcband is None'+str(band)+imgFile)
			  continue

		  #print('srcband read:'+str(band))

		  arr = srcband.ReadAsArray()
		  #print(np.unique(arr))

		  if bandInd==0:
			  R=arr.shape[0]
			  C=arr.shape[1]
			  #print(arr.shape)
			  matImg=np.zeros((R, C, len(self.Bands)), dtype=np.float32);
		  matImg[:,:,bandInd]=np.float32(arr)#/self.scale ;

		  bandInd += 1

	  return prj, trans, matImg


  '''
     # create a list of files dir for all the cities needed to be produced
     # input:
             fileD: cities path
             cities: a list of cities under the fileD
     # output:
             files: all the files
             imgNum_city: the image number of each city
  '''
  def createFileList_cities(self, fileD, cities):
	  files = []
	  imgNum_city = np.zeros((len(cities),1), dtype=np.uint8)
	  for j in np.arange(len(cities)):
			 #all seasons
			 file = sorted(glob2.glob(fileD+ cities[j] +'/**/*_'  + '*.tif'))
			 files.extend(file)
			 imgNum_city[j] = len(file)

	  return files, imgNum_city

  '''
      # create a list of files dir for all the images in different seasons of the input city dir
      # input:
              fileD: the absolute path of one city
      # output:
              files: all the files corresponding to different seasons
  '''
  def createFileList(self, fileD):
	  files = []
	  imgNum_city = np.zeros((1,1), dtype=np.uint8)

	 #all seasons
	  file = sorted(glob2.glob(fileD +'/**/*_'  + '*.tif'))
	  files.extend(file)
	  return files


  'use the imgmatrix to get patches'
# input:
#        mat: a band of the image
# output:
		# patch: the patch of this input image, to feed to the classifier
		# R: the size of the final lcz map
		# C: the size of the final lcz map
  def  __img2patch(self, mat):

      #mat=np.pad(mat, ((np.int(self.dim_x_img/2), np.int(self.dim_x_img/2)), (np.int(self.dim_y_img/2), np.int(self.dim_y_img/2))), 'reflect')

      window_shape = (self.dim_x_img, self.dim_y_img)#self.dim_x_img

	  #window_shape = (self.dim_x, self.dim_y)#self.dim_x_img
      B = view_as_windows(mat, window_shape, self.step)#B = view_as_windows(A, window_shape,2)
      #print(B.shape)

      patches=np.reshape(B, (-1, window_shape[0], window_shape[1]))
      #print(patches.shape)

      #patches=scipy.ndimage.zoom(patches, [1, (self.dim_x/window_shape[0]),  (self.dim_y/window_shape[1])], order=1)#Bilinear interpolation would be order=1

      R=B.shape[0]#the size of the final map
      C=B.shape[1]

      return patches, R, C

  def  __label2patch(self, mat):

      window_shape = (self.dim_x, self.dim_y)#self.dim_x_img
      print('window_shape',window_shape)

      #B = view_as_windows(mat, window_shape, self.step) #B = view_as_windows(A, window_shape,2)
      B = view_as_windows(mat, window_shape, int(self.step/self.ratio))#8#change here
      #print('self.step',self.step)
      print('B.shape',B.shape)

      patches=np.reshape(B, (-1, window_shape[0], window_shape[1]))
      print('patches.shape',patches.shape)

      #patches=scipy.ndimage.zoom(patches, [1, (self.dim_x/window_shape[0]),  (self.dim_y/window_shape[1])], order=1)#Bilinear interpolation would be order=1

      R=B.shape[0]#the size of the final map
      C=B.shape[1]

      return patches, R, C



  '''
    # save prediction as tif
    # input:
            yPre0: the vector of the predictions
            R: the size of the final lcz map
            C: the size of the final lcz map
            prj: projection data
            trans: projection data
            mapFile: the file to save the produced map
            idxNan: the index of no data area.
    # output:
            no
  '''
  def predic2tif_vector(self, yPre0, R, C, prj, trans, mapFile, idxNan):
	  totalNum=R*C;

	  if self.isSeg==1:
		  xres =trans[1]*1
		  yres= trans[5]*1
		  geotransform = (trans[0]+xres*(1-1)/2.0, xres, 0, trans[3]+xres*(1-1)/2.0, 0, yres)
	  else:
		  xres =trans[1]*self.step
		  yres= trans[5]*self.step
          #geotransform = (trans[0]+xres*(self.self.dim_x_img-1)/2.0, xres, 0, trans[3]+yres*(self.dim_y_img-1)/2.0, 0, yres)#no padding
		  geotransform = (trans[0]-xres*(2-1)/2.0, xres, 0, trans[3]-yres*(2-1)/2.0, 0, yres)#with padding

	  dimZ=np.shape(yPre0)[1]

	 # create the dimZ raster file
	  dst_ds = gdal.GetDriverByName('GTiff').Create(mapFile, C, R, dimZ, gdal.GDT_UInt16)#gdal.GDT_Byte .GDT_Float32 gdal.GDT_Float32
	  dst_ds.SetGeoTransform(geotransform) # specify coords
	  dst_ds.SetProjection(prj)

	  for i in np.arange(dimZ):
		  yPre=np.zeros((totalNum,1), dtype=np.uint16 ) + self.nanValu + 1;
		  yPre[idxNan]= self.nanValu;# set no data value
		  tmp = np.where( (yPre== self.nanValu + 1 ) )

		  yPre[ tmp[0]]=yPre0[:,i].reshape((-1,1));

		  map=np.reshape(yPre, (R, C))
		  dst_ds.GetRasterBand(int(i+1)).WriteArray(map)   # write band to the raster
	  dst_ds.FlushCache()                     # write to disk
	  dst_ds = None

  '''
    # save a map as tif
    # input:
            mat: the matrix to be saved
            prj: projection data
            trans: projection data
            mapFile: the file to save the produced map
    # output:
            no
  '''
  def predic2tif(self, mat, prj, trans, mapFile):

	  R=mat.shape[0]
	  C=mat.shape[1]

	  totalNum=R*C;

	  if self.isSeg==1:
		  xres =trans[1]*1
		  yres= trans[5]*1
		  geotransform = (trans[0]+xres*(1-1)/2.0, xres, 0, trans[3]+xres*(1-1)/2.0, 0, yres)
		  print(geotransform)
	  else:
		  xres =trans[1]*self.step
		  yres= trans[5]*self.step
		  #geotransform = (trans[0]+xres*(self.step-1)/2.0, xres, 0, trans[3]+xres*(self.step-1)/2.0, 0, yres)

		  #geotransform = (trans[0] + trans[1]*(2-1)/2.0 -xres*(2-1)/2.0, xres, 0, trans[3] + trans[5]*(2-1)/2.0-yres*(2-1)/2.0, 0, yres)#no padding
		  geotransform = (trans[0] + trans[1]*(self.dim_x-1)/2.0, xres, 0, trans[3] + trans[5]*(self.dim_y-1)/2.0, 0, yres)#no padding

		  #geotransform = (trans[0]-xres*(2-1)/2.0, xres, 0, trans[3]-yres*(2-1)/2.0, 0, yres)#with padding

		  print(geotransform)

	  dimZ=mat.shape[2]

	 # create the dimZ raster file
	  dst_ds = gdal.GetDriverByName('GTiff').Create(mapFile, C, R, dimZ, gdal.GDT_UInt16)#gdal.GDT_Byte .GDT_Float32 gdal.GDT_Float32
	  dst_ds.SetGeoTransform(geotransform) # specify coords
	  dst_ds.SetProjection(prj)

	  for i in np.arange(dimZ):
		  map=mat[:,:,i]
		  dst_ds.GetRasterBand(int(i+1)).WriteArray(map)   # write band to the raster
	  dst_ds.FlushCache()                     # write to disk
	  dst_ds = None

  def file2prediction(self, file, model):
  		  prj, trans, img= self.loadImgMat(file)
  		  R=img.shape[0]
  		  C=img.shape[1]
  		  x_test, mapR, mapC, idxNan = self.Bands2patches(img,1)
  		  print('x_test:', x_test.shape)
  		  y_pre0 = model.predict(x_test, batch_size = 16, verbose=1)

  		  return y_pre0, mapR, mapC, prj, trans, idxNan

  def file2prediction_(self, file, model):
  		  prj, trans, img= self.loadImgMat(file)
  		  R=img.shape[0]
  		  C=img.shape[1]
  		  print('img:', R, C)
  		  x_test, mapR, mapC = self.Bands2patches_all(img,1)
  		  print('x_test:', x_test.shape)
  		  y_pre0 = model.predict(x_test, batch_size = 128, verbose=1)

  		  return y_pre0, mapR, mapC, prj, trans

  '''
    # from a season to one map and several proba files, returned
    # input:
            files: the files of the images in this season
            model: the trained model
            proFile: the path (and name) to save the prob.
    # output:
            no
  '''
  def season2map_(self, files, model, proFile):

	  numImg=len(files);
	  for idSeason in np.arange(numImg):
		  file = files[idSeason]
		  print(file)

		  y_pre0, mapR, mapC, prj, trans, idxNan = self.file2prediction(file, model)

		  y_pre = self.predict_classes((y_pre0))
		  print(np.unique((np.int8(y_pre))))
		  y_pre=y_pre.reshape(-1,1) ;
		  print('map size:', mapR, mapC)

		  '''save soft prob'''
		  proFile0 = proFile+file[file.rfind('_'):]
		  self.predic2tif_vector(y_pre0*10000, mapR, mapC, prj, trans, proFile0, idxNan)

		  totalNum=mapR*mapC;
		  if idSeason==0:
			  yPreAll=np.empty((totalNum, np.uint8(numImg)  ))

		  yPre=np.zeros((totalNum,1), dtype=np.int8 ) + self.nanValu + 1;
		  yPre[idxNan]= self.nanValu;# set no data value
		  tmp = np.where( (yPre== self.nanValu + 1 ) )
		  yPre[ tmp[0]]=y_pre.reshape((-1,1)) ;

		  yPreAll[:,idSeason]=yPre.reshape(yPre.shape[0]);

	  '''majority voting of seasons'''
	  m = stats.mode(yPreAll, 1)
	  mapConfi=np.empty((mapR, mapC, 2), dtype=np.int8)

	  print(np.unique((m[0])))
	  mapConfi[:,:,0]=np.reshape(m[0], (mapR, mapC))
	  mapConfi[:,:,1]=np.reshape(m[1], (mapR, mapC))

	  return prj, trans, mapConfi

  '''
    # from a season to one map and several proba files, saved
    # input:
            files: the files of the images in this season
            model: the trained model
            proFile: the path (and name) to save the prob.
            mapFile: the file to save the produced map
    # output:
            no
  '''
  #@memprof(plot = True)
  def season2map(self, files, model, proFile, mapFile):
	  prj, trans, mapConfi=self.season2map_(files, model, proFile)
	  self.predic2tif(mapConfi, prj, trans, mapFile+'.tif')


  def season2Bdetection(self, files, model, proFile, mapFile):
	  prj, trans, mapConfi=self.season2map_(files, model, proFile)
	  mapConfi[mapConfi== 2] = self.nanValu
	  self.predic2tif(mapConfi, prj, trans, mapFile+'.tif')

  def scaleBand(self,patches):
      patches_=np.zeros(patches.shape, dtype=np.float32)
      #for b in np.arange(patches.shape[-1]):

      patch=patches.reshape(-1,1)
        #print(patch.shape)
      scaler = StandardScaler().fit(patch)
        #print(scaler.mean_.shape)
      patches_=scaler.transform(patch).reshape(patches.shape[0],patches.shape[1], patches.shape[2])

      return patches_

#
  def img2Bdetection(self, file, model, proFile, mapFile):
      y_pre0, mapR, mapC, prj, trans = self.file2prediction_(file[0], model)

      print('y_pre0.shape', y_pre0.shape)
      y=y_pre0.argmax(axis=3)+1
      print(np.unique(y))
      y[y== 2] = self.nanValu#to make sure only the buildings are labeled
      print('y.shape', y.shape)
      del y_pre0

      mapPatch_shape=y.shape[1]
      B_=np.reshape(y, (mapR, mapC, y.shape[1], y.shape[2]))
      print('B_.shape', B_.shape)
      del y

      C=B_.transpose(0,2,1,3).reshape(-1,B_.shape[1]*B_.shape[3])
      print('C.shape', C.shape)
      del B_

      mapConfi=np.zeros((C.shape[0], C.shape[1], 1), dtype=np.int8)
      mapConfi[:,:,0]=C;

      if mapPatch_shape*2==self.dim_x:
          print('downsampling by 2!')
          trans0 =trans[0]+trans[1]*(2-1)/2.0
          trans3= trans[3]+trans[5]*(2-1)/2.0
          trans1 =trans[1]*2
          trans5= trans[5]*2
          trans = (trans0, trans1, 0, trans3, 0, trans5)
      self.predic2tif(mapConfi, prj, trans, mapFile+'.tif')

  def img2Bdetection_ovlp(self, file, model, mapFile, out=1, nn=0):
      prj, trans, img= self.loadImgMat(file[0])
      R=img.shape[0]
      C=img.shape[1]
      print('img:', R, C)

      imgN=len(file);

      if self.dim_x_img==48:
          paddList=[0,12,24,36]
      if self.dim_x_img==32:
          paddList=[0,8,16,24]
      if self.dim_x_img==20:
          paddList=[0,5,10,20]
      if self.dim_x_img==64:
          paddList=[0,16,32,48]

      # else:
      #     paddList=[0,32,64,96]


      for padding in paddList:#[0,16,32,48,64,80,96,112]
#prepare x_test
          if imgN==1:
              if padding==0:
                  img1=img
              else:
                  img1=np.pad(img, ((padding, 0), (padding, 0), (0,0)), 'reflect')
              print(img1.shape)
              x_test, mapR, mapC = self.Bands2patches_all(img1,1)
              print('x_test:', x_test.shape)
              #y = model.predict(x_test, batch_size = 4, verbose=1)
          if imgN==4:
              prj, trans, img_1= self.loadImgMat(file[1])
              prj, trans, img_2= self.loadImgMat(file[2])
              prj, trans, img_3= self.loadImgMat(file[3])
              if padding==0:
                  img0=img
                  img1=img_1
                  img2=img_2
                  img3=img_3
              else:
                  img0=np.pad(img, ((padding, 0), (padding, 0), (0,0)), 'reflect')
                  img1=np.pad(img_1, ((padding, 0), (padding, 0), (0,0)), 'reflect')
                  img2=np.pad(img_2, ((padding, 0), (padding, 0), (0,0)), 'reflect')
                  img3=np.pad(img_3, ((padding, 0), (padding, 0), (0,0)), 'reflect')
              #print(img1.shape)
              x_test0, mapR, mapC = self.Bands2patches_all(img0,1)
              x_test1, mapR, mapC = self.Bands2patches_all(img1,1)
              x_test2, mapR, mapC = self.Bands2patches_all(img2,1)
              x_test3, mapR, mapC = self.Bands2patches_all(img3,1)
              print('x_test0:', x_test0.shape)
              x_test=[x_test0, x_test1, x_test2, x_test3];

              print('direct out 2 prediction:')
              y0, y1 = model.predict(x_test, batch_size = 16, verbose=1)

              C0, mapPatch_shape_0 =self.pro_from_x(mapR, mapC, y0, padding)
              C1, mapPatch_shape_1 =self.pro_from_x(mapR, mapC, y1, padding)
              OS0 = np.int( self.dim_x_img/ mapPatch_shape_0 )   #ratio between the input and the output
              OS1 = np.int( self.dim_x_img/ mapPatch_shape_1 )

              if padding==0:
                  r0=C0.shape[0]
                  c0=C0.shape[1]
                  Pro0=C0[0:(r0-mapPatch_shape_0),0:(c0-mapPatch_shape_0),:]
                  r1=C1.shape[0]
                  c1=C1.shape[1]
                  Pro1=C1[0:(r1-mapPatch_shape_1),0:(c1-mapPatch_shape_1),:]
              else:
                  Pro0=Pro0+C0[np.int(padding/OS0):(r0-mapPatch_shape_0+np.int(padding/OS0)), np.int(padding/OS0):(c0-mapPatch_shape_0+np.int(padding/OS0)), :]
                  Pro1=Pro1+C1[np.int(padding/OS1):(r1-mapPatch_shape_1+np.int(padding/OS1)), np.int(padding/OS1):(c1-mapPatch_shape_1+np.int(padding/OS1)), :]

          if out==1:
              y = model.predict(x_test, batch_size = 16, verbose=1)
              print('pre:', y.shape)
              print('print(np.unique(y))',np.unique(y.argmax(axis=3)+1))

              C, mapPatch_shape =self.pro_from_x(mapR, mapC, y, padding)
              OS = np.int( self.dim_x_img/ mapPatch_shape )   #ratio between the input and the output
              if padding==0:
                  r=C.shape[0]
                  c=C.shape[1]
                  Pro=C[0:(r-mapPatch_shape),0:(c-mapPatch_shape),:]
              else:
                  Pro=Pro+C[np.int(padding/OS):(r-mapPatch_shape+np.int(padding/OS)), np.int(padding/OS):(c-mapPatch_shape+np.int(padding/OS)), :]

      if out==1:
          print(Pro.shape)
          self.save_pre_pro(prj, trans, Pro, mapFile, mapPatch_shape)


  ''' get y of the targeting shape'''
  def pro_from_x(self, mapR, mapC, y, padding):

      mapPatch_shape=y.shape[1]
      print('class num:', y.shape[-1])

      B_=np.reshape(y, (mapR, mapC, y.shape[1], y.shape[2], y.shape[-1]))
      print('B_.shape', B_.shape)
      del y

      C=np.zeros((B_.shape[0]*B_.shape[2], B_.shape[1]*B_.shape[3], B_.shape[4]), dtype=float)
      for dim in np.arange(B_.shape[4]):
          B_1=B_[:,:,:,:,dim]
          C[:,:,dim]=B_1.transpose(0,2,1,3).reshape(-1,B_1.shape[1]*B_1.shape[3])
          del B_1
      return C, mapPatch_shape

  ''' get middle output'''
  def layerPredict(model, x_test):
      layer_name = 'hse'
      o0Pre = K.function([model.layers[0].input, K.learning_phase()],
                                      [model.get_layer(layer_name).output])
      layer_name = 'lcz'
      o1Pre = K.function([model.layers[0].input, K.learning_phase()],
                                      [model.get_layer(layer_name).output])

      o0 = o0Pre([x_test, 0])[0]
      o1 = o1Pre([x_test, 0])[0]

      return o0, o1

  ''' save predictions and pro'''
  def save_pre_pro(self, prj, trans, Pro, mapFile, mapPatch_shape):

      y=Pro.argmax(axis=2)+1
      print('print(np.unique(y))',np.unique(y))
      #y[y== 2] = 0#self.nanValu#to make sure only the buildings are labeled

      mapConfi=np.zeros((y.shape[0], y.shape[1], 1), dtype=np.int16)
      mapConfi[:,:,0]=y;

      mapPro=np.zeros((y.shape[0], y.shape[1], 1), dtype=np.int16)
      mapPro= Pro*10000;

      #if mapPatch_shape*2==self.dim_x:
      ratio=self.dim_x / mapPatch_shape;
      print('downsampling by: ', ratio)
      trans0 =trans[0]+trans[1]*(ratio-1)/2.0
      trans3= trans[3]+trans[5]*(ratio-1)/2.0
      trans1 =trans[1]* ratio
      trans5= trans[5]* ratio
      trans = (trans0, trans1, 0, trans3, 0, trans5)

      self.predic2tif(mapConfi, prj, trans, mapFile+'.tif')
      self.predic2tif(mapPro, prj, trans, mapFile+'_pro.tif')


  ''' generate class prediction from the input samples'''
  def predict_classes(self, x):
	  y=x.argmax(axis=1)+1
	  return y
