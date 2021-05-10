import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import lm
from scipy import ndimage
import pyximport
pyximport.install()
import description
import utilities
import logging
import glob
import random
from sklearn import svm

# create list of filenames
files = glob.glob(r"D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20-unpack\png"+r"\*.png")
# specific range
frange = [91398,150000]
files = files[frange[0]:frange[1]]

# setup logger
log = logging.getLogger(__name__)
# set formatting style
logging.basicConfig(format="[{filename}:{lineno}:{levelname} - {funcName}() ] {message}",style='{')
log.setLevel(logging.INFO)

# parameters for SLICO algorithm
region_size = 4
ruler = 15.0
ratio = 0.075
min_element_size = 4
num_iterations = 10

# One-class SVM settings
# upper bound on the fraction of training errors and a lower bound of the fraction of support vectors
nu=0.1
# kernel type used in algorithm
kernel="rbf"
# kernel coefficient for the chosen type
gamma=0.1

def trainSVM(trainFiles,testFiles):
    # inform user of the parameter settings
    nfiles = len(trainFiles)
    ntfiles = len(testFiles)
    
    logging.info(f"Settings : {nfiles} training images, {len(testFiles)} test images")
    superf = None
    # iterate over training files
    for fi,f in enumerate(trainFiles):
        print(f"{fi}/{nfiles} ({fi/nfiles})")
        # get image
        frame = cv2.imread(f)
        # slico
        superpix = cv2.ximgproc.createSuperpixelSLIC(frame,cv2.ximgproc.SLICO,region_size,ruler)
        superpix.iterate(num_iterations)
        if min_element_size>0:
            superpix.enforceLabelConnectivity(min_element_size)
        # get labels
        # auto generated number controlled by min_element_size
        labels = superpix.getLabels()
        # calculate color features
        log.debug("Calculating color features")
        colF = description.calculateAllColorFeatures(frame,labels)
        log.debug("normalizing the color features")
        colF = description.normalizeFeatures(colF)
        log.debug("Building texture filter bank")
        # build texture filter bank
        fbank = lm.makeLMfilters()
        log.debug("Calculating texture features")
        # calculate response to texture filters
        textF = description.calculateTextureFeatureVector(frame,labels,fbank)
        log.debug("normalizing the texture vectors")
        textF = description.normalizeFeatures(textF)
        # build array of labels as additional feature
        logging.debug("Building label vector")
        ul = np.unique(labels)
        logging.debug("Concatenating the feature sets together")
        # concatenate features together to form feature vector
        fvector = np.concatenate((ul.reshape(-1,1),colF,textF),axis=1)
        if fi==0:
            superf = fvector.copy()
        else:
            superf = np.vstack((superf,fvector))
    # iterate over testing files
    testf = None
    for fi,f in enumerate(testFiles):
        print(f"{fi}/{ntfiles} ({fi/ntfiles})")
        # get image
        frame = cv2.imread(f)
        # slico
        superpix = cv2.ximgproc.createSuperpixelSLIC(frame,cv2.ximgproc.SLICO,region_size,ruler)
        superpix.iterate(num_iterations)
        if min_element_size>0:
            superpix.enforceLabelConnectivity(min_element_size)
        # get labels
        # auto generated number controlled by min_element_size
        labels = superpix.getLabels()
        # calculate color features
        log.debug("Calculating color features")
        colF = description.calculateAllColorFeatures(frame,labels)
        log.debug("normalizing the color features")
        colF = description.normalizeFeatures(colF)
        log.debug("Building texture filter bank")
        # build texture filter bank
        fbank = lm.makeLMfilters()
        log.debug("Calculating texture features")
        # calculate response to texture filters
        textF = description.calculateTextureFeatureVector(frame,labels,fbank)
        log.debug("normalizing the texture vectors")
        textF = description.normalizeFeatures(textF)
        # build array of labels as additional feature
        logging.debug("Building label vector")
        ul = np.unique(labels)
        logging.debug("Concatenating the feature sets together")
        # concatenate features together to form feature vector
        fvector = np.concatenate((ul.reshape(-1,1),colF,textF),axis=1)
        if fi==0:
            testf = fvector.copy()
        else:
            testf = np.vstack((testf,fvector))

    ## feed into SVM
    # setup SVM
    clf = svm.OneClassSVM(nu=nu,kernel=kernel,gamma=gamma)
    logging.info(f"SVM Settings: nu {nu}, kernel type {kernel}, gamma {gamma}")
    clf.fit(superf)
    ## get output data
    # using training features
    y_pred_train = clf.predict(superf)
    y_pred_test = clf.predict(testf)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    return clf,y_pred_train,y_pred_test,n_error_train,n_error_test

if __name__ == "__main__":
    # split images into training and validation set
    split = 0.5
    # split list
    trainFiles = files[:int(len(files)*split)]
    # suffle
    random.shuffle(trainFiles)
    # get validation images
    validFiles = files[int(len(files)*split):]
    model,y_pred_train,y_pred_test,n_error_train,n_error_test=trainSVM(trainFiles,validFiles)
    
