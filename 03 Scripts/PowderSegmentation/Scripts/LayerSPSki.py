import os
import descriptors
import numpy as np
import utilities
import lm
from functools import reduce
import cv2

from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
from skimage.segmentation import join_segmentations

import matplotlib.pyplot as plt

class SkiLayeredSLICO:
    class SPID:
        def __init__(self):
            pass
    
    def __init__(self,np,**kwargs):
        # make filter bank
        self.fbank = lm.makeLMfilters()
        # test if number of superpixels per pass is an iterable
        try:
            iter(np)
        except TypeError:
            if "nl" not in kwargs:
                raise ValueError("Missing number of passes! Required when number of superpixels is a single value")
                return
            else:
                self.__np = [np]*kwargs["nl"]
                self.__nl = kwargs["nl"]
        else:
            self.__np = np
            self.__nl = len(np)

        # check to see if slic compactness parameter is set
        if "compactness" not in kwargs:
            self.compact = 1.
        else:
            self.compact = float(kwargs["compactness"])
        # set convert to lab space flag to True
        self.convert2lab = True
    # function to get the number of superpixels per layer list
    def numSuperpixels(self):
        return self.__np
    # (in theory) the total number of unique pixels
    def getTotalSPs(self):
        return np.product(self.__np)
    # function to get the number of passes to eb performed
    def numPasses(self):
        return self.__nl
    # call method to iterate over 
    def __call__(self,img,verbose=False,index_only=False):
        # check to see if it's multichannel
        is_mc = len(img.shape)>2
        # build arguments matrix
        self.slic_args = dict(compactness=self.compact,slic_zero=True,convert2lab=True,multichannel=is_mc)
        if verbose:
            print(f"SLIC args : {self.slic_args}")
        # for labels matrix
        segments = np.zeros(img.shape[:2],np.int64)
        # for top level labels
        topLabels = np.zeros(img.shape[:2],np.int64)
        # image msak
        mask = np.zeros(img.shape[:2],img.dtype)
        if verbose:
            print(f"mask : {mask.shape},{mask.dtype}")
        # for each pass
        for ni,nc in enumerate(self.__np):
            # variable to keep track of the max id of the superpixels at present
            # updated to max id of subpixel region
            maxid=0
            if verbose:
                print(f"pass {ni}, nc {nc}")
            # for each unique superpixel
            for vi,v in enumerate(np.unique(segments)):
                if verbose:
                    print(f"pass {ni} for top level superpixel {v}")
                # build mask for target superpixel
                mask[topLabels==v]=255
                # find contours in the mask 
                ct,_ = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
                # estimate bounding box around masked area
                tx,ty,w,h = cv2.boundingRect(ct[0])
                # get superpixels for sub area
                segSub = slic(frame[int(ty):int(ty+h),int(tx):int(tx+w),:],n_segments=nc,**self.slic_args)
                print(segSub.max())
                # if not the first superpixel
                # update the ids with the max  of the previous superpixel plus 1
                if vi>0:
                    print(f"adding {maxid}")
                    segSub += maxid+1
                    print(f"max sub id {segSub.max()}")
                    # update max id
                    maxid = segSub.max()
                    print(f"new max id {maxid}")
                # find where the top level superpixel mask is 255
                smx,smy = np.where(mask==255)
                # update the segments array with the sub segments array
                # the segments array is updated such that only the top levels pixels are updated
                segments[smx,smy] = segSub[smx-smx.min(),smy-smy.min()]
                #### DEBUGGING ###########
                plt.imshow(segments,cmap="gray")
                plt.show()
                #cv2.imwrite(f"test_n{ni}_sp{v}.png",cv2.normalize(segments,segments.copy(),0,255,cv2.NORM_MINMAX))
            # update top level segments
            topLabels = segments.copy()
            # if first pass
            if ni==0:
                # if the user has requested the features
                if not index_only:
                    if verbose:
                        print("getting textures")
                    # get texture features for top level superpixels
                    textF = descriptors.calculateTextureFeatureVector(img,topLabels,self.fbank)

        # iterate over ids 
        ## when all the passes have finished
        # save finished labels matrix
        self.labels = segments.copy()
        if verbose:
            print(f"final labels, min {self.labels.min()}, max {self.labels.max()}")
        if not index_only:
            # get color features after finding all superpixels
            self.colF = descriptors.calculateAllColorFeatures(img,segments)
