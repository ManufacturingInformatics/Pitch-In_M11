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
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
from skimage import measure

show_mcubes = True

def convPointCloudToVoxel(x,y,z,res_inc=10):
    from pyntcloud import PyntCloud
    import pandas as pd
    # convert separate coordinate vectors to array
    points = np.array([[xx,yy,zz] for xx,yy,zz in zip(x,y,z)])
    # convert numpy.ndarray->pandas.DataFrame->PyntCloud
    cloud = PyntCloud(pd.DataFrame(points,columns=['x','y','z']))
    # generate voxel id grid
    # set the number of cubes based on the range of data plus res_inc
    vid = cloud.add_structure("voxelgrid",n_x=int(x.max()-x.min())+res_inc,n_y=int(y.max()-y.min())+res_inc,n_z=int(z.max()-z.min())+res_inc)
    # return the boxel id vector, binary
    return cloud.structures[vid].get_feature_vector()

def compute_sizev(x,y,z):
    from scipy.spatial import cKDTree
    # combine vectors together
    vv = np.dstack([x,y,z])[0]
    # form a kd tree
    tree = cKDTree(vv)
    # initialize size list to update
    sz = []
    # for each data point find the nearest neighbour
    # use the neighbour to compute the size of the cube
    for i in range(vv.shape[0]):
        # get the two nearest neighbours to this point
        # the first will be the point itself as it belongs to the tree
        # the second will be the next nearest
        qq=tree.query(vv[i,:],k=2)[1][1]
        sz.append(np.abs(vv[qq,:]-vv[i,:]).tolist())
    # update zero sized cubes to min size
    sz = np.array(sz)
    sz[sz==0] += 0.05
    return np.array(sz)
    
def customCube(loc,size):
    #print(size)
    # generate a unit cube
    # each set in the list is the 4 points for a face
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]    
    X = np.array(X).astype(float)
    # increase cube to desired size
    for i in range(3):
        X[:,:,i] *= size[i]
    # shift the position of the cube so it is in the target location
    X += np.array(loc)
    return X

def customCubeCentre(loc,size):
    #print(size)
    # generate a unit cube
    # each set in the list is the 4 points for a face
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]    
    X = np.array(X).astype(float)
    # increase cube to desired size
    for i in range(3):
        X[:,:,i] *= size[i]
    # shift the position of the cube so centre is in target position
    loca = np.array(loc)
    X += loca*0.5
    return X

def genCubesVoxelID(xx,yy,zz,T,sz=None,vmin=None,vmax=None,c='bl'):
    cubes = []
    cols = []
    cmap = cm.get_cmap('viridis')
    if vmin is None or vmax is None:
        norm = Normalize(vmax=T.max(),vmin=T.min())
    else:
        norm = Normalize(vmax=vmax,vmin=vmin)
            
    # if a single size to apply to all
    if len(sz) == 3:
        #print(sz)
        for x,y,z,t in zip(xx,yy,zz,T):
            if c=='bl':
                cubes.append(customCube([x,y,z],sz))
            elif c=='ct':
                cubes.append(customCube([x,y,z],sz))
            cols.append(np.repeat([cmap(norm(t))],6,axis=0))
    elif len(sz) > 3:
        if len(sz) != xx.shape[0]:
            raise ValueError("Shape of size matrix does not match number of data points")
        else:
            for x,y,z,t,ss in zip(xx,yy,zz,T,sz):
                if c=='bl':
                    cubes.append(customCube([x,y,z],ss))
                elif c=='ct':
                    cubes.append(customCubeCentre([x,y,z],ss))
                cols.append(np.repeat([cmap(norm(t))],6,axis=0))
    #print(cols[T.argmax()])
    return Poly3DCollection(np.concatenate(cubes),facecolor=np.concatenate(cols),edgecolor='k')


class SliderManager:
    def __init__(self,ax,limax,ax2,limax2):
        # data shared between sliders
        self.x = np.empty(0)
        self.y = np.empty(0)
        self.z = np.empty(0)
        self.T = np.empty(0)
        self.Tmin = 0.0
        self.Tmax = 0.0
        # function for rbf
        self.rbf_method = 'multiquadric'
        # colormap for plot
        self.cmap = 'viridis'
        # axes for sliders
        self.ax = ax
        self.ax2 = ax2
        # save limits for retrieval later
        self.slider_limits = [limax,limax2]
        # create sliders
        self.sidx = Slider(self.ax,'Index',limax[0],limax[1],valinit=limax[0],dragging=True,valstep=1,valfmt='%0.0f')
        self.sfilt = Slider(self.ax2,'Filter',limax2[0],limax2[1],valinit=limax2[0],dragging=True,valstep=0.01,valfmt='%1.2f')
        # assign slider update methods as internal class methods
        self.sidx.on_changed(self.updateIdx)
        self.sfilt.on_changed(self.updateFilt)
        # create options menu
        self.opts_menu = FittingMenu(self)
        
    def getSliderLimits(self):
        return self.slider_limits

    def setDrawAxes(self,sparseRot,filledRot,sparseData,img):
        # set the axes to update in the slider methods
        self.sparseRotAx = sparseRot
        self.sparseData = sparseData
        self.filledRotAx = filledRot
        self.imgAx = img

    def updateIdx(self,val):
        # get wanted frame index
        ff = self.sidx.val
        # clear required axes
        self.sparseRotAx.clear()
        self.filledRotAx.clear()
        self.sparseData.clear()
        # get frame
        with h5py.File(path,'r') as file:
            frame = file['pi-camera-1'][:,:,ff]
        # convert data to 8-bit image
        frame_norm = (frame-frame.min())/(frame.max()-frame.min())
        frame_norm *= 255
        frame_norm = frame_norm.astype('uint8')
        frame_norm = np.dstack((frame_norm,frame_norm,frame_norm))
        # find bounding box
        bb,ct = findBB(frame)
        # draw rectangle on frame
        cv2.rectangle(frame_norm,(bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),(255,0,0),1)
        # rotate 90 degs clockwise
        frame_norm = cv2.rotate(frame_norm,cv2.ROTATE_90_CLOCKWISE)
        # show image with bounding box
        self.imgAx.imshow(frame_norm,cmap='gray')
        # get values within bounding box
        img = frame[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
        img_c = img.copy()
        # get all cols on second half
        img_c = img_c[img_c.shape[0]//2:,:]
        # get location of values above 20.0
        xx,yy = np.where(img_c>temp_tol)
        # get temperature values for marked areas
        zz = img_c[xx,yy]
        # create copies of the data to update
        xxr = xx.copy()
        yyr = yy.copy()
        zzr = np.zeros(xxr.shape,xxr.dtype)
        dd = zz.copy()
        # form coordinate data set, xyz
        data  = np.array([[x,y,0.0] for x,y in zip(xx,yy)])
        # resolution of rotation
        rot_res = 10.0
        ## rotate the dataset around the y axis
        for r in np.arange(0.0,360.0,rot_res):
            # generate rotation object
            rot = R.from_euler('xyz',[0.0,r,0.0],degrees=True)
            # apply to matrix
            vv = rot.apply(data)
            # udpate position matrix
            xxr = np.concatenate((xxr,vv[:,0]))
            yyr = np.concatenate((yyr,vv[:,1]))
            zzr = np.concatenate((zzr,vv[:,2]))
            # add the temperature values onto the end
            dd = np.concatenate((dd,zz))

        # plot the original data frame as a reference
        self.sparseData.scatter(xx,yy,c=zz,cmap=self.cmap)
        self.sparseData.set_title("Bounding Box Values >{}($^\circ$C),Half Set".format(temp_tol))
        # invert the y axis to "flip" the data so that the top of the data matches the
        # top of the reference image and dataset
        self.sparseData.invert_yaxis()
        
        # plot the rotated dataset where color indicates temperature
        self.sparseRotAx.scatter3D(xxr,yyr,zzr,c=dd,depthshade=False,cmap=self.cmap)
        self.sparseRotAx.set_title("Temperature Vol from Half Set")
        self.sparseRotAx.set(xlabel='X',ylabel='Y',zlabel='Z')

        ## fit rbf to data
        #xx,yy = np.where(img>20.0)
        #zz = img[xx,yy]
        #xx,yy = np.where(frame>20.0)
        #zz = frame[xx,yy]
        # fit rbf to the bounding box dataset whose value is above 20.0
        #print(xx.shape,yy.shape,zz.shape)
        rr = Rbf(xx,yy,zz,function=self.rbf_method)
        # generate a meshgrid for the inbetween values
        XX,YY = np.meshgrid(np.linspace(xx.min(),xx.max(),xx.shape[0]),np.linspace(yy.min(),yy.max(),yy.shape[0]))
        # use rbf to predict temperature across the entire x,y range
        zz = rr(XX,YY)
        # convert coordinate meshgrids to 1d coordinate vectors and make a copy to update
        self.x = XX.ravel()
        self.y = YY.ravel()
        # as the data is in a X-Y plane the z coordinate is 0.0
        self.z = np.zeros(self.x.shape,self.x.dtype)
        # make a copy of the temperature prediction to update
        self.T = zz.ravel()
        self.Tmin = self.T.min()
        self.Tmax = self.T.max()
        # form coordinate data set, xyz
        data  = np.array([[x,y,0.0] for x,y in zip(XX.ravel(),YY.ravel())])
        # resolution of rotation
        rot_res = 10.0
        ## rotate the dataset around the y axis
        for r in np.arange(0.0,360.0,rot_res):
            # generate rotation object
            rot = R.from_euler('xyz',[0.0,r,0.0],degrees=True)
            # apply to matrix
            vv = rot.apply(data)
            # udpate position matrix
            self.x = np.concatenate((self.x,vv[:,0]))
            self.y = np.concatenate((self.y,vv[:,1]))
            self.z = np.concatenate((self.z,vv[:,2]))
            # add the temperature values onto the end
            self.T = np.concatenate((self.T,zz.ravel()))
        # draw rotated data on a scatter plot 
        #self.filledRotAx.scatter3D(self.x,self.y,self.z,c=self.z,depthshade=False,cmap=self.cmap)
        sz = (np.abs(np.diff(self.x).min())*0.5,np.abs(np.diff(self.y).min())*0.5,np.abs(np.diff(self.z).min())*0.5)

        if show_mcubes:
            vg = convPointCloudToVoxel(self.x,self.y,self.z)
            verts,faces,normals,values = measure.marching_cubes_lewiner(vg)
            mesh = Poly3DCollection(verts[faces])
            mesh.set_edgecolor('k')
            self.filledRotAx.add_collection3d(mesh)

            # store axis limits before filtering
            self.filledRotAx.set_xlim3d(verts[:,0].min(),verts[:,0].max())
            self.filledRotAx.set_ylim3d(verts[:,1].min(),verts[:,1].max())
            self.filledRotAx.set_zlim3d(verts[:,2].min(),verts[:,2].max())
            self.orig_limits = self.getAxLimits(self.filledRotAx)
        else:
            self.sz = compute_sizev(self.x,self.y,self.z)
            self.filledRotAx.add_collection3d(genCubesVoxelID(self.x,self.y,self.z,self.T,self.sz,c='ct'))
            self.filledRotAx.set_xlim3d(self.x.min(),self.x.max())
            self.filledRotAx.set_ylim3d(self.y.min(),self.y.max())
            self.filledRotAx.set_zlim3d(self.z.min(),self.z.max())
            self.orig_limits = self.getAxLimits(self.filledRotAx)
        # if the filter slider is set, call the update method for that
        # if it's set at 0.0 then there's no point
        if self.sfilt.val>0.0:
            self.updateFilt(0.0)
        # if filter slider update method isn't to be called, then redraw the figure for the rotation axes
        # the figure is redrawn in the updateFilt method so there's no need to have the figure be redrawn twice
        else:
            self.filledRotAx.figure.canvas.draw()

    def getAxLimits(self,ax):
        return [*ax.axis(),*(ax.get_zlim() if isinstance(ax,Axes3D) else [None,None])]

    def updateFilt(self,val):
        # if data has not been updated, exit early
        if self.x.shape[0]==0:
            return
        # clear axis
        self.filledRotAx.clear()
        # get the temperature limit to filter by
        lim = self.Tmin + (self.Tmax-self.Tmin)*self.sfilt.val
        # generate plot using filtered data
        # colormap norm limits are set to the unfiltered dataset so the color scheme remains the same
        #self.filledRotAx.scatter3D(self.x[self.T>=lim],self.y[self.T>=lim],self.z[self.T>=lim],c=self.T[self.T>=lim],cmap=self.cmap,vmin=self.Tmin,vmax=self.Tmax)
        sz = (np.abs(np.diff(self.x).min())*0.5,np.abs(np.diff(self.y).min())*0.5,np.abs(np.diff(self.z).min())*0.5)
        #print(sz)
        # convert point cloud to voxel index data
        vg = convPointCloudToVoxel(self.x[self.T>=lim],self.y[self.T>=lim],self.z[self.T>=lim])
        # apply marching cubes algorithm to generate surface
        verts,faces,normals,values = measure.marching_cubes_lewiner(vg)
        
        # build poly objects from faces and verticies
        mesh = Poly3DCollection(verts[faces],edgecolor='k')
        # add collection of objects to 
        self.filledRotAx.add_collection3d(mesh)

        # store axis limits before filtering
        #self.orig_limits = self.getAxLimits(self.filledRotAx)
        self.filledRotAx.set_xlim3d(verts[:,0].min()-5,verts[:,0].max()+5)
        self.filledRotAx.set_ylim3d(verts[:,1].min()-5,verts[:,1].max()+5)
        self.filledRotAx.set_zlim3d(verts[:,2].min()-5,verts[:,2].max()+5)
        # force redraw canvas
        self.filledRotAx.figure.canvas.draw()

class FittingMenu:
    def __init__(self,slidermg):
        # set slide manager
        self.slidermg = slidermg
        # create figure
        self.fig = plt.figure(num="RBF Fitting and Plotting Options")
        # methods for RBF, values will be text sizes
        self.rbf_methods = ["gaussian","multiquadric","inverse"]
        # colormaps, values will be text sizes
        self.cmaps = ["viridis","hot"]
        # create lists to store buttons and arrays
        # a dictionary is used to make indexing clearer
        self.buttons = {"RBF" : [None]*len(self.rbf_methods),"CMAP":[None]*len(self.cmaps)}
        self.axes = {"RBF" : [None]*len(self.rbf_methods),"CMAP":[None]*len(self.cmaps)}
        self.rbf_title = self.fig.text(0.06,0.9,"RBF Fitting Method",bbox=dict(facecolor='red', alpha=0.5))
        self.cmap_title = self.fig.text(0.06,0.65,"Colormap",bbox=dict(facecolor='red', alpha=0.5))
        # get the max number of labels
        # used to determine global spacing of buttons
        max_sz = max(len(self.rbf_methods),len(self.cmaps))
        ## create axes and buttons
        # horizontal positioning of buttons is set so that they are horizontally justified
        # width of buttons is set based on size of text, or should be
        for i,k in enumerate(self.rbf_methods):
            self.axes["RBF"][i] = self.fig.add_axes([0.05+round((max_sz**-1)*(i),2),0.75,float(self.getTextSize(k))/50,0.1])
            self.buttons["RBF"][i] = Button(self.axes["RBF"][i],k)
            self.buttons["RBF"][i].on_clicked(self.methodClick)

        for i,k in enumerate(self.cmaps):
            self.axes["CMAP"][i] = self.fig.add_axes([0.05+round((max_sz**-1)*(i),2),0.5,float(self.getTextSize(k))/50,0.1])
            self.buttons["CMAP"][i] = Button(self.axes["CMAP"][i],k)
            self.buttons["CMAP"][i].on_clicked(self.cmapClick)


    def getTextSize(self,t,**kwargs):
        from matplotlib.text import Text
        # get the size of the text in terms of units
        return Text(text=t,**kwargs).get_fontsize()

    def methodClick(self,event):
        # find the index of the axes where the button press event occured
        # as self.buttons and self.axes are indexed the same way, we can use the index to access the button pressed
        # update the RBF method used in the slider manager class
        self.slidermg.rbf_method = self.buttons["RBF"][self.axes["RBF"].index(event.inaxes)].label.get_text()

    def cmapClick(self,event):
        # find the index of the axes where the button press event occured
        # as self.buttons and self.axes are indexed the same way, we can use the index to access the button pressed
        # update the colormap method used in the slider manager class
        self.slidermg.cmap = self.buttons["CMAP"][self.axes["CMAP"].index(event.inaxes)].label.get_text()
    
# data index to start at
starter_f = 91398
# end index to finish at
end_f = 150000
# tolerance used for temperature filtering
temp_tol = 20.0
# path to file
path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"

# function to search for the bounding box area
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

## create plotting axes
# create figure
f = plt.figure(num="Heating Area Volume")
f.suptitle("Estimated Temperature Volume")
# create grid spec based off figure settings
gs = gridspec.GridSpec(2,3,f)
ax = f.add_subplot(gs[0,0],projection='3d')
ax.set(xlabel='X',ylabel='Y',zlabel='Z')
ax.view_init(-84,-90)
axrbf = f.add_subplot(gs[1,0],projection='3d')

axorig = f.add_subplot(gs[:,1])
axorig.set(xlabel='X',ylabel='Y',title="Data Inside Box")
aximg = f.add_subplot(gs[:,2])
aximg.set_title("Original Image")
# add the axes that would need clearing to a list
fax_clear = [ax,axorig,axrbf]

# added slider axes
axidx = f.add_axes([0.25, 0.01, 0.65, 0.03])
axfilt = f.add_axes([0.25,0.045,0.65,0.03])
# adding a button to reset the view of the rotated scatter points
axreset = f.add_axes([0.02, 0.01, 0.1, 0.035])
rview = Button(axreset,"RESET")
rview.on_clicked(lambda x : ax.view_init(-84,-90))

# create slider manager class
mg = SliderManager(axidx,[starter_f,end_f],axfilt,[0.0,1.0])
mg.setDrawAxes(ax,axrbf,axorig,aximg)

# add button for moving index slider by 1
axincr = f.add_axes([0.13,0.01,0.08,0.035])
incridx = Button(axincr,">>")
incridx.on_clicked(lambda x : mg.sidx.set_val(mg.sidx.val+1))

plt.show()
