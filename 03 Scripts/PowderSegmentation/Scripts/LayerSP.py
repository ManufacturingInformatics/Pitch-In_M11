import cv2
import os
import descriptors
import numpy as np
import utilities
import lm
from functools import reduce

kwargs = dict(region_size = 100,
ruler = 15.0,
ratio = 0.075,
min_element_size = 25,
num_iterations = 10)

class LayeredSLICO:
    # nl : number of layers/tiers
    # np : number of superpixels per layer. single or iterable
    def __init__(self,nl=2,np=30,**kwargs):
        # bank of filters
        self.fbank = lm.makeLMfilters()
        # number of layers
        self.nl = nl
        # test number of superpixels for each layer to see if it's an iterable
        try:
            iter(np)
        except TypeError:
            # if not an iterator
            # make list of superpixels for each layer with each element being the same
            self.np = [np]*nl
        else:
            # if the length of the passed sequelnce is not the same as the number of layers
            # i.e. the number of superpixels for each layer is not defined
            if len(np)!=nl:
                raise ValueError(f"Number of superpixels per layer not properly defined! Length of sequelnce needs to be same length as the number of layers")
                return
            else:
                self.np = list(np)

        # kwargs is the arguments passed to the OpenCV SLICO method
        # and parameters used for spatial regularization
        if "ruler" not in kwargs:
            raise ValueError("Missing ruler argument for SLICO!")
            return
        else:
            self.ruler = kwargs["ruler"]

        if "num_iterations" not in kwargs:
            self.nit = 10
        else:
            self.nit = kwargs["num_iterations"]

        if "min_element_size" not in kwargs:
            self.min_size = 0
        else:
            self.min_size = int(kwargs["min_element_size"])
        # test to see if spatial regularization is required
        self.spatial_reg = self.min_size>0

    # get the SLICO labels matrix for the given frame and target number of clusters
    # class attributes control the settings for SLIC
    # SLICO labels start at 0. min_label is added in case they need to be unique
    def getSLICOLabels(self,image,nc,min_label=1):
        # calculate the required region size for the desired number of clusters
        region_size = utilities.calcRegionSize(image.shape,nc)
        # perform superpixels segmentation
        superpix = cv2.ximgproc.createSuperpixelSLIC(fmask,cv2.ximgproc.SLICO,region_size,self.ruler)
        # iterate SLICO
        superpix.iterate(self.nit)
        # force spatial regularization
        if spatial_reg:
            superpix.enforceLabelConnectivity(self.min_size)
        # get labels
        return superpix.getLabels()+(min_label)
    
    # method for performing layered SLICO on the given image
    def __call__(self,img):
        # mask array to update
        mask = np.zeros(img.shape[:2],img.dtype)
        # labels matrix
        labels = np.zeros(img.shape[:2],np.int32)
        self.__origLabels = labels.copy()
        # save a copy of the labels to iterate over
        self.__origLabels = labels.copy()
        # calculate features vector for the larger superpixels
        # feature textures used for all sub pixels
        textF = descriptors.calculateTextureFeatureVector(frame,self.__origLabels,self.fbank)
        # initialize features array
        # number of rows set as total number of pixels to be found
        self.features = np.empty((reduce(lambda x,y: x*y,np),0))
        # max id
        # used to update detailed sub regions to maintain uniqueness
        maxid = 0
        # iterate for each "layer"
        for ni,nc in enumerate(np):
            # iterate over each superpixel
            for v in np.unique(labels):
                # create mask for the unique superpixel
                mask[self.__origLabels==v] = 255
                # find contours in the mask 
                ct,_ = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
                # estimate bounding box around masked area
                tx,ty,w,h = cv2.boundingRect(ct[0])
                # get area inside rectangle
                fmask = frame[int(ty):int(ty+h),int(tx):int(tx+w),:]
                # get superpixels for masked area 
                lsub = self.getSLICOLabels(fmask,nc)
                # calculate color features for detailed labels array
                colF = descriptors.calculateAllColorFeatures(fmask,lsub)
                # build textures for the subpixels
                tempF = np.repeat(textF[ni,:],lsub.max()+1,axis=0)
                # append to larger features array
                self.features = np.concatenate((self.features,tempF),axis=1)
                # get the ids where the mask is 255
                y,x = np.where(mask==255)
                # zero the coordinates
                # assuming top corner of bounding box lines up with superpixel mask
                y -= y.min()
                x -= x.min()
                # update labels array with the new detailed subpixels
                # use image pask to update only the ids within current target superpixel
                labels[mask==255] = lsub[x,y]
                # update ids to maintain uniqueness
                labels[mask==255] += (maxid+1)
                # if the first iteration maintain original copy of top level labels to iterate over
                if ni==0:
                    self.__origLabels = lsub
                # update max id
                maxid = np.max(labels[mask==255])
        # update class 
        self.labels = labels-lablels.min()
