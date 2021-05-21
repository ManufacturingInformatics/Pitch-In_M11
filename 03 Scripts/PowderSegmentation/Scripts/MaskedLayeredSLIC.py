import matplotlib.pyplot as plt

import maskslic as seg
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.io import imread
import lm

class LayeredMaskedSLIC:
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
        self.slic_args = dict(compactness=self.compact,seed_type='nplace',slic_zero=False,convert2lab=True,multichannel=is_mc,enforce_connectivity=False,recompute_seeds=True)
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
            # reset max id
            maxid=0
            # iterate over each unique superpixel
            for vi,v in enumerate(np.unique(segments)):
                if verbose:
                    print(f"pass {ni} for top level superpixel {v},nc {nc}")
                # build mask for target superpixel
                mask= topLabels==v
                # apply segmentation across masked area
                subSeg = seg.slic(img,n_segments=nc,mask=mask,**self.slic_args)
                print(f"min {subSeg.min()}, max {subSeg.max()}")
                # update ids to ensure uniqueness
                subSeg += maxid+1
                # update segmentations
                segments[mask]=subSeg[mask]+1
                # update max id
                if v>0:
                    maxid = subSeg[mask].max()
                    print(f"max id updated to {maxid}")
            # update the top level ids
            topLabels = segments.copy()
        # update internal copy of labels to for access afterwards
        self.labels = segments.copy()

if __name__ == "__main__":
    import utilities
    # get screenshot
    frame,_,_ = utilities.useScreenshot()
    # perform layered segmentation with two layers
    ski = LayeredMaskedSLIC([10,10])
    ski(frame,True,True)
    # show segmentations result
    plt.imshow(ski.labels,cmap="gray")
    plt.show()
