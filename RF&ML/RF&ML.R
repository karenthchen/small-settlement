#Author: karen.t.chen@yale.edu (Tzu-Hsin Karen Chen)
#Script index:
#1. Random forest training
#2. Random forest prediction
#3. Maximum likelihood training
#4. Maximum likelihood prediction

library(raster)
library(randomForest)
rm(list=ls())


#1. Random forest training
setwd("C:/.") #set the folder dictionary
#Multispectral imagery
data_a=read.csv("./data/RF&ML/train_v5_50m_ls2.csv")
#CCDC features
data_a=read.csv("./data/RF&ML/train_v5_50m_ccdc1-4_2.csv")

names(data_a)
dim(data_a)

y<-data_a[,4]
x<-sapply(data_a[,6:59], as.numeric) #landsat
x<-sapply(data_a[,6:221], as.numeric) #ccdc


dim(x)

model4=randomForest(x=x,y=y,ntree=500,importance=TRUE,oob.prox =FALSE,keep.forest=T)
model4$importance[order(model4$importance[,1]),]
(model4$importance/model4$importanceSD)[order((model4$importance/model4$importanceSD)[,1]),]

save(model4, file = "./prediction/model4_ccdc14_2_33.RData")
save(model4, file = "./prediction/model4_ls_2_33.RData")



load("./prediction/model4_ls_2_33.RData") #ls for prediction
load("./prediction/model4_ccdc14_2_33.RData") #ccdc for prediction

#2. Random forest prediction

Files=list.files("./data/RF&ML/landsat_33kernel",pattern="^RF_.*.tif$",full.names=T) #multispectral images
Files=list.files("./data/RF&ML/ccdc_33kernel",pattern="^RF_CCDC.*.tif$",full.names=T) #CCDC features

for (i in 1:length(Files)){
  out=gsub("./data/RF&ML/landsat_33kernel/RF","./prediction/UF_RF",Files[i])
  print(Files[i])
  img=brick(Files[i])
  names(img)=names(model4$forest$ncat)
  start.time=Sys.time()
  print("start prediction")
  pre<-predict(img,model4)
  print("start write file")
  writeRaster(pre,out)
  end.time=Sys.time()
  print(end.time-start.time) 
}
plot(pre)
plot(pre2)




#3. Maximum likelihood training

#Multispectral imagery
data_a=read.csv("./data/RF&ML/train_v5_50m_ls2.csv")
#CCDC features
data_a=read.csv("./data/RF&ML/train_v5_50m_ccdc1-4_2.csv")

data_a$shuffle=sample(1:(dim(data_a)[1]),replace=F)
data_a=data_a[order(data_a$shuffle),]

y<-data_a[,4]
x<-data.frame(sapply(data_a[,6:59], as.numeric))#landsat 3*3 kernel
x<-data.frame(sapply(data_a[,6:221], as.numeric))#ccdc 3*3 kernel
dim(data_a)

dim(x)

model4_lr = glm(y ~ ., family = binomial,
                        data = x)
hist(logit2prob(fitted(model4_lr)))
summary(model4_lr)


save(model4_lr, file = "./prediction/model4_LR_ls_33.RData")
save(model4_lr, file = "./prediction/model4_LR_ccdc14.RData")



#4. Maximum likelihood prediction
load("./prediction/model4_LR_ls2_33.RData")
summary(model4_ols)

Files=list.files("./data/RF&ML/landsat_33kernel",pattern="^RF_.*.tif$",full.names=T) #multispectral images
Files=list.files("./data/RF&ML/ccdc_33kernel",pattern="^RF_CCDC.*.tif$",full.names=T) #CCDC features


for (i in 1:64){#length(Files)
  print(i)
  (out=gsub("./data/RF&ML/landsat_33kernel/RF","./prediction/LR_Landsat/UF_LRv5s2",Files[i]))
  img=brick(Files[i])

  names(img)=names(model4_lr$coefficients)[2:length(names(model4_lr$coefficients))]
  print("start prediction")
  pre<-predict(img,model4_lr)
  values(pre)=logit2prob(values(pre))
  writeRaster(pre,out)
}


