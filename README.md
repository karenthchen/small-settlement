# Mapping mountain settlements using a convolutional neural network: a subpixel approach

Note: This repository is now available for review process and will be publicly avialbale once the paper is published.


## 1. Data collection

## Multispectral image time series
https://code.earthengine.google.com/e43dc60380c7c127d17886802a92ad47

- To Loop over the study regions by small ROI:

upload fishnet to GEE

https://www.dropbox.com/s/yemtlof19iv9tdf/fishnetLS4_correctutm_HKH.zip?dl=0


## CCDC features
https://code.earthengine.google.com/936a98293f7f0679e6e29c0e29a2dab5


## Training data
https://www.dropbox.com/sh/4x06q33g6a7291s/AAC5MeOwsPDuuYEvQYue5fwBa?dl=0

## 2. Models
## Convolutional neural networks - semantic segmentation (UNet)
- Use python (jupyter notebook recommended)
- Run segmentation_regression_multispectral.ipynb for using raw image as input (the best model)
- Run segmentation_regression_CCDC.ipynb for using CCDC as input


## Random forests
- Use R
- Run RF&ML.R (sections 1 and 2)




