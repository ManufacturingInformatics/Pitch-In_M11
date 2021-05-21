import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider,Button
from skimage.filters import sobel,threshold_otsu
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import Rbf
from scipy.special import xlogy
import cv2

# manager class for plot and its data
# used for displaying the source data for RBF
# scatter plot only
class SourcePlot:
    def __init__(self,plotax=None):
        """ Base class for storing and managing the data shown on a plot

            plotax : 2D plotting axes to manage
        """
        # stored data
        self.x = np.empty(0)
        self.y = np.empty(0)
        self.T = np.empty(0)
        # type of plot
        self.cmap="viridis"
        # set axes
        self.plotax = plotax
        # list of children plots dependent on source
        self._children = []
    def setAxes(self,ax):
        """ Convience method to update plot axes managed by class

            ax : New 2D axes to update when the data changes
        """
        self.plotax = ax
    def addChild(self,ch):
        """ Add new RBFPlot child for this class to update

            ch : New child class

            Adds the new child to an internal list that is iterated over in
            plot method
        """
        self._children.append(ch)
    def plot(self):
        """ Clear and redraw the data being shown in the managed axes

            Clears the axes and redraws the data stored internally.
            Calls plot method of all children RBFPlot currently stored.
        """
        # clear plot
        self.plotax.clear()
        # plot data as scatter plot
        self.plotax.scatter(self.x,self.y,c=self.T,cmap=self.cmap)
        self.plotax.set_title("Half BBox Data")
        # invert y axis so the laser portion is at the top
        self.plotax.invert_yaxis()
        # update children plots
        for rbfp in self._children:
            rbfp.plot()

# manager class for rbf results plot
# uses the data in set SourcePlot class to generate RBF and results
# whenever the source plot is updated, the RBFs are updated
class RbfPlot:
    def __init__(self,source=None,plotax=None,method="multiquadric",cmap="viridis"):
        """ Manager class for plotting the results of fitting an RBF to data stored in
            set SourcePlot class.

            source : SourcePlot class that the data is extracted from
            plotax : Axes to plot the RBF contourf plot on
            method : String or function that is valid to scipy Rbf class
        """
        self.cmap = cmap
        # set source of data
        self.source = source
        # type of rbf
        self.rbf_type = method
        # resolution scale
        self.res_scale = 2
        # create placeholder for rbf
        self.rbf = None
        # create placeholder for axes
        self.plotax = plotax
    def setCmap(self,cmap):
        self.cmap = cmap
    def setSource(self,source):
        """ Convienence method to update the SourcePlot used to supply data"""
        self.source = source
    def setRBFMethod(self,method):
        """ Update the RBF method used to create scipy.interpolate.Rbf class """
        self.rbf_type = method
    def setResScale(self,sc):
        """ Set resolution scale for grid data used in generating RBF results"""
        self.res_scale = sc
    def buildRBF(self):
        """ Build RBF using current settings and data in set SourcePlot

            Returns 1 if successful and non-zero if there was a problem
        """
        #rr = Rbf(self.source.x,self.source.y,self.source.T,function=self.rbf_type)
        try:
            # fit rbf to data
            # it if can't rbf to data, the exception is triggered and the internal rbf state isn't updated
            rr = Rbf(self.source.x,self.source.y,self.source.T,function=self.rbf_type)
            # update internal copy of rbf if successful
            self.rbf = rr
            return 1
        except Exception as e:
            if hasattr(e,'message'):
                print(e.message())
            else:
                print(e)
            print("Failed to build RBF of type {} to source data!".format(self.rbf_type))
            return 0

    def plot(self,buildRBF=True):
        """ Plot generate RBF results using currently built RBF and PlotSource

            buildRBF : Flag to call buildRBF method to fit RBF to source data with current settings. Default: True
        """
        # if flag is set, call buildRBF
        # if there's an error during call, message is printed
        if buildRBF:
            if self.buildRBF() <1:
                print("RBF Build Error!")
                return
            
        # check if rbf has been set
        if self.rbf is not None:
            # if source data has been set
            if self.source is not None:
                # generate grid of value to interpolate over
                # the resolution of the grid is a multiple of the original resolution
                self.x,self.y = np.meshgrid(np.linspace(self.source.x.min(),self.source.x.max(),self.source.x.shape[0]*self.res_scale),np.linspace(self.source.y.min(),self.source.y.max(),self.source.y.shape[0]*self.res_scale))
                # use rbf to interpolate
                self.T = self.rbf(self.x,self.y)
                # plot data
                self.plotax.clear()
                self.plotax.contourf(self.x,self.y,self.T,cmap=self.cmap)
                # update plot labels
                # title is set as the method if it's a string
                # it it's a custom function, then the function's name is used
                self.plotax.set(title=(self.rbf_type if type(self.rbf_type)==str else self.rbf_type.__name__))
        else:
            print("RBF has not been built yet!")

class SliderManager:
    def __init__(self,path,slax,imax,imin=0,source=None,imgax=None):
        # set axes for slider
        self.ax = slax
        # set SourcePlot that the slider will update
        self.source = source
        # set axes to display original data with bounding box
        self.imgax = imgax
        # set path to HDF5 file
        self.path = path
        # set index limits
        self.imax = imax
        self.imin = imin
        # set tolerance for filtering data
        self.tol = 20.0
        # create slider for index
        self.sidx = Slider(self.ax,'Index',self.imin,self.imax,valinit=self.imin,dragging=True,valstep=1,valfmt='%0.0f')
        # set update method
        self.sidx.on_changed(self.updateSource)

    def setSource(self,source):
        self.source = source
    def setSliderLim(self,imax,imin):
        # update internal copy of limits
        self.imax = self.imax
        self.imin = self.imin
        ## rebuild slider
        self.sidx = Slider(self.ax,'Index',self.imin,self.imax,valinit=self.imin,dragging=True,valstep=1,valfmt='%0.0f')
        # set update method
        self.sidx.on_changed(self.updateSource)
    def setTol(self,tol):
        self.tol = tol
    def setDataPath(self,path):
        self.path = path
        
    @staticmethod
    def findBB(frame):
        # perform sobel operation to find edges on frame
        # assumes frame is normalized
        sb = sobel(frame)
        # clear values outside of known range of target area
        sb[:,:15] = 0
        sb[:,25:] = 0
        # get otsu threshold value
        thresh = threshold_otsu(sb)
        # create mask for thresholded values
        img = (sb > thresh).astype('uint8')*255
        # perform morph open to try and close gaps and create a more inclusive mask
        img=cv2.morphologyEx(img,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        # search for contours in the thresholded image
        ct = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
        # if a contour was found
        if len(ct)>0:
            # if there is more than one contour, sort contours by size
            if len(ct)>1:
                ct.sort(key=lambda x : cv2.contourArea(x),reverse=True)
            # return the bounding box for the largest contour and the largest contour
            return cv2.boundingRect(ct[0]),ct[0]
        
    def updateSource(self,*args):
        if self.source is not None:
            # get frame from hdf5 file
            with h5py.File(self.path,'r') as file:
                frame = file['pi-camera-1'][:,:,self.sidx.val]
            # remove any nans or infs
            np.nan_to_num(frame,copy=False)
            # convert data to 8-bit image
            frame_norm = (frame-frame.min())/(frame.max()-frame.min())
            frame_norm *= 255
            frame_norm = frame_norm.astype('uint8')
            frame_norm = np.dstack((frame_norm,frame_norm,frame_norm))
            # find bounding box
            bb,ct = self.findBB(frame)
            # draw rectangle on frame
            cv2.rectangle(frame_norm,(bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),(255,0,0),1)
            # find location of max value
            maxrc = np.unravel_index(np.argmax(frame),frame.shape)[:2]
            # draw cross indicating location
            print(maxrc)
            frame_norm = cv2.drawMarker(frame_norm,maxrc[::-1],color=[0,255,0],markerType=cv2.MARKER_CROSS,markerSize=2,thickness=1)
            # rotate 90 degs clockwise
            frame_norm = cv2.rotate(frame_norm,cv2.ROTATE_90_CLOCKWISE)
            # show image with bounding box
            self.imgax.imshow(frame_norm,cmap='gray')
            # get values within bounding box
            img = frame[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
            # make a copy of the image
            img_c = img.copy()
            # get all cols on second half
            img_c = img_c[img_c.shape[0]//2:,:]
            # get location of values above 20.0
            self.source.x,self.source.y = np.where(img_c>self.tol)
            # get temperature values for marked areas
            self.source.T = img_c[self.source.x,self.source.y]
            # update SourcePlot to show new data
            self.source.plot()
        else:
            print("Slider Manager source plot not set!")

## function for polyharmonic RBF
# based off pseudo code proposed in
# https://github.com/scipy/scipy/issues/9904


# adapted from
# https://en.wikipedia.org/wiki/Radial_basis_function
# implementation is designed to avoid ln(0 = -inf errors
def polyHarmonic7(self,r):
    return r**(7-1) * np.log(r**r)

def polyHarmonic9(self,r):
    return r**(9-1) * np.log(r**r)

# class for generating individual polyHarmonic functions
class polyHarmonic:
    def __init__(self,k):
        self.k = k
    def __call__(self,r):
        return r**(self.k-1) * np.log(r**r)

#polyHarmonic7 = polyHarmonic(7)

# path to hdf5 file
path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"

# create figure
f = plt.figure()
f.suptitle("Comparison of RBF methods for Interpolating")
# create grid of subplots
gs = gridspec.GridSpec(3,4,f)
## set plots for data and image
axdata = f.add_subplot(gs[:2,-1])
aximg = f.add_subplot(gs[-1,-1])
# hide axes text for image axes, keep border
aximg.axes.get_xaxis().set_ticks([])
aximg.axes.get_yaxis().set_ticks([])
# set title
aximg.set_title("Original")
# create axes for slider
# location of bottom left corner, width, height
axidx = f.add_axes([0.1, 0.01, 0.8, 0.03])
## create plot source objects
# scatter data plot
spdata = SourcePlot(axdata)

## add subplots for RBFs
# RBF methods to generate plots for
rbf_ms = ["multiquadric","inverse","gaussian","linear","cubic","quintic","thin_plate",polyHarmonic7,polyHarmonic9]
# get the dimensions of the grid spec
r,c = gs.get_geometry()
# array to hold axes and RBF plots
rbf_ax_obj = np.empty((r,c-1),dtype='object')
# iterate over dimensions
for rr in range(r):
    # ignoring the final column as that's where the source data and image axes are
    for cc in range(c-1):
        # create axes for plot
        ax = f.add_subplot(gs[rr,cc])
        # create RBFPlot object
        rbf_o = RbfPlot(spdata,ax,method=rbf_ms[rr*(c-1)+cc])
        print("Building RbfPlot for {}".format(rbf_ms[rr*(c-1)+cc]))
        # add as child to manage of spdata
        spdata.addChild(rbf_o)
        # add to array
        rbf_ax_obj[rr,cc] = [ax,rbf_o]

# create slider manager which will update the source plot whenever the slider is used
# this will cause a chain reaction of updates
mg = SliderManager(path,axidx,150000,91399,spdata,aximg)
plt.show()



        
