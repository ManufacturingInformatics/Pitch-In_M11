import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from skimage.filters import sobel,threshold_otsu
import cv2
from scipy.spatial.transform import Rotation as R
from matplotlib.tri.triangulation import Triangulation as mtri
import vtk
from vtk.util import numpy_support
from scipy.interpolate import Rbf

starter_f = 91399
ff = starter_f
filt = 0.6
path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"


print("Getting dataset limits")
path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"
#path = "D:\BEAM\RPiWaifu\pi-camera-data-127001-2019-10-14T12-41-20-waifu-noise-scale-91398-150000.hdf5"
if "waifu" not in path:
    with h5py.File(path,'r') as file:
        mmax,mmin = np.nanmax(file['pi-camera-1'][()],axis=(0,1,2)),np.nanmin(file['pi-camera-1'][()],axis=(0,1,2))
else:
    with h5py.File(path,'r') as file:
        mmax,mmin = np.nanmax(file['pi-camera-1-scale'][()],axis=(0,1,2)),np.nanmin(file['pi-camera-1-scale'][()],axis=(0,1,2))

def norm2image(T,vmax=mmax,vmin=mmin):
    return int((T-vmin)*((2**16)/(vmax-vmin)))


ren = vtk.vtkRenderer()
renwin = vtk.vtkRenderWindow()
renwin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
# define interactor
iren.SetRenderWindow(renwin)

def findBB(frame):
    sb = sobel(frame)
    sb[:,:15] = 0
    sb[:,25:] = 0
    thresh = threshold_otsu(sb)
    img = (sb > thresh).astype('uint8')*255
    img=cv2.morphologyEx(img,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    ct = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(ct)>0:
        if len(ct)>1:
            ct.sort(key=lambda x : cv2.contourArea(x),reverse=True)
        return cv2.boundingRect(ct[0]),ct[0]

def getDataInCT(frame,ct):
    mask = np.zeros(frame.shape,np.uint8)
    mask = cv2.drawContours(mask,[ct],0,1,-1)
    return frame*mask

# convert numpy float array -> vtkFloatArray
#vk_data = numpy_support.numpy_to_vtk(num_array=ct_data_filt.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

# dataset that will be updated by function readPoints
pointSource = vtk.vtkProgrammableSource()
def readPoints():
    output = pointSource.GetPolyDataOutput()
    points = vtk.vtkPoints()
    output.SetPoints(points)
    # collect frame
    with h5py.File(path,'r') as file:
        frame = file['pi-camera-1'][:,:,ff]
    np.nan_to_num(frame,copy=False)
    # find bounding box and contour used to generate it
    bb,ct = findBB(frame)
    # get the data masked to inside the contour
    #img_ct = getDataInCT(frame,ct)
    img_ct = frame[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
    # get the location of the data points
    xx,yy = np.where(img_ct>20.0)
    # get the respective values
    zz = img_ct[xx,yy]
    # have the height touch 0
    zz -= zz.min()
    # generate and add mirroring data
    xx = np.concatenate((xx,xx))
    yy = np.concatenate((yy,yy))
    zz = np.concatenate((zz,-zz))
    # update points set
    for x,y,z in zip(xx,yy,zz):
        points.InsertNextPoint(float(z),float(y),float(x))
##
def readPointsRotRbf():
    output = pointSource.GetPolyDataOutput()
    points = vtk.vtkPoints()
    output.SetPoints(points)
    # collect frame
    with h5py.File(path,'r') as file:
        frame = file['pi-camera-1'][:,:,ff]
    np.nan_to_num(frame,copy=False)
    # find bounding box and contour used to generate it
    bb,ct = findBB(frame)
    # get the data masked to inside the contour
    #img_ct = getDataInCT(frame,ct)
    img = frame[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
    img_c = img.copy()
    # get all cols on second half
    img_c = img_c[img_c.shape[0]//2:,:]
    # get location of values above 20.0
    xx,yy = np.where(img_c>20.0)
    # get temperature values for marked areas
    zz = img_c[xx,yy]

    rr = Rbf(xx,yy,zz)
    interp_shape = xx.shape[0]*2
    XX,YY = np.meshgrid(np.linspace(xx.min(),xx.max(),interp_shape),np.linspace(yy.min(),yy.max(),interp_shape))
    zz = rr(XX,YY)

    # create copies to update
    dd = zz.ravel()
    zz = zz.ravel()
    xxr = XX.ravel()
    yyr = YY.ravel()
    zzr = np.zeros(xxr.shape,xxr.dtype)
    # compile data into xyz array
    data  = np.array([[x,y,0.0] for x,y in zip(xxr,yyr)])
    # rotate according to set resolution
    for r in np.arange(0.0,360.0,1.0):
        # generate rotation object
        rot = R.from_euler('xyz',[0.0,r,0.0],degrees=True)
        # apply to matrix
        vv = rot.apply(data)
        # udpate position matrix
        xxr = np.concatenate((xxr,vv[:,0]))
        yyr = np.concatenate((yyr,vv[:,1]))
        zzr = np.concatenate((zzr,vv[:,2]))
        dd = np.concatenate((dd,zz))

    lim = zz.min() + (zz.max()-zz.min())*filt
    for x,y,z in zip(xxr[dd>=lim],yyr[dd>=lim],zzr[dd>=lim]):
        points.InsertNextPoint(float(z),float(y),float(x))

######
def readPointsRotRbfBk():
    print("Updating data")
    output = pointSource.GetPolyDataOutput()
    points = vtk.vtkPoints()
    output.SetPoints(points)
    # collect frame
    with h5py.File(path,'r') as file:
        frame = file['pi-camera-1'][:,:,ff]
    np.nan_to_num(frame,copy=False)
    # find bounding box and contour used to generate it
    bb,ct = findBB(frame)
    # get the data masked to inside the contour
    #img_ct = getDataInCT(frame,ct)
    img = frame[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
    img_c = img.copy()
    # get all cols on second half
    img_c = img_c[img_c.shape[0]//2:,:]
    # get location of values above 20.0
    xx,yy = np.where(img_c>20.0)
    # get temperature values for marked areas
    zz = img_c[xx,yy]

    rr = Rbf(xx,yy,zz)
    interp_shape = xx.shape[0]*2
    XX,YY = np.meshgrid(np.linspace(xx.min(),xx.max(),interp_shape),np.linspace(yy.min(),yy.max(),interp_shape))
    zz = rr(XX,YY)

    # create copies to update
    dd = zz.ravel()
    zz = zz.ravel()
    xxr = XX.ravel()
    yyr = YY.ravel()
    zzr = np.zeros(xxr.shape,xxr.dtype)
    # compile data into xyz array
    data  = np.array([[x,y,0.0] for x,y in zip(xxr,yyr)])
    # rotate according to set resolution
    for r in np.arange(0.0,360.0,1.0):
        # generate rotation object
        rot = R.from_euler('xyz',[0.0,r,0.0],degrees=True)
        # apply to matrix
        vv = rot.apply(data)
        # udpate position matrix
        xxr = np.concatenate((xxr,vv[:,0]))
        yyr = np.concatenate((yyr,vv[:,1]))
        zzr = np.concatenate((zzr,vv[:,2]))
        dd = np.concatenate((dd,zz))

    print("Filtering data from background")
    dd -= dd.min()
    lim = np.diff(dd).max()*0.5
    print(lim)
    print(xxr[dd>lim].shape,yyr[dd>lim].shape,zzr[dd>lim].shape)
    print(dd.min(),dd.max())
    # background values are a base value plus some noise
    # to ignore tha background, we ignore values less than 1
    # this should reveal the values that stand out from the bk
    for x,y,z in zip(xxr[dd>lim],yyr[dd>lim],zzr[dd>lim]):
        points.InsertNextPoint(float(z),float(y),float(x))  

def readPointsRotEdge():
    output = pointSource.GetPolyDataOutput()
    points = vtk.vtkPoints()
    output.SetPoints(points)
    # collect frame
    with h5py.File(path,'r') as file:
        frame = file['pi-camera-1'][:,:,ff]
    np.nan_to_num(frame,copy=False)
    # find bounding box and contour used to generate it
    bb,ct = findBB(frame)
    # get the data masked to inside the contour
    #img_ct = getDataInCT(frame,ct)
    img = frame[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
    img_c = img.copy()
    # get all cols on second half
    img_c = img_c[img_c.shape[0]//2:,:]
    # get location of values above 20.0
    xx,yy = np.where(img_c>20.0)
    # get temperature values for marked areas
    zz = img_c[xx,yy]

    edge = []
    # for each of the unique column values
    for c in set(yy):
        # get the highest row value, i.e. location of edge value
        edge.append(xx[yy==c].max()+bb[3]//2)

    xx = np.array(edge)
    yy = np.unique(yy)
    zz = img[xx,yy]
    # create copies to update
    dd = zz.copy()
    xxr = xx.copy()
    yyr = yy.copy()
    zzr = np.zeros(xxr.shape,xxr.dtype)
    # compile data into xyz array
    data  = np.array([[x,y,0.0] for x,y in zip(xx,yy)])
    # rotate according to set resolution
    for r in np.arange(0.0,360.0,10.0):
        # generate rotation object
        rot = R.from_euler('xyz',[0.0,r,0.0],degrees=True)
        # apply to matrix
        vv = rot.apply(data)
        # udpate position matrix
        xxr = np.concatenate((xxr,vv[:,0]))
        yyr = np.concatenate((yyr,vv[:,1]))
        zzr = np.concatenate((zzr,vv[:,2]))
        dd = np.concatenate((dd,zz))
        
    for x,y,z in zip(xxr,yyr,zzr):
        points.InsertNextPoint(float(z),float(y),float(x))

def readPointsRot():
    output = pointSource.GetPolyDataOutput()
    points = vtk.vtkPoints()
    output.SetPoints(points)
    # collect frame
    with h5py.File(path,'r') as file:
        frame = file['pi-camera-1'][:,:,ff]
    np.nan_to_num(frame,copy=False)
    # find bounding box and contour used to generate it
    bb,ct = findBB(frame)
    # get the data masked to inside the contour
    #img_ct = getDataInCT(frame,ct)
    img = frame[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
    img_c = img.copy()
    # get all cols on second half
    img_c = img_c[img_c.shape[0]//2:,:]
    # get location of values above 20.0
    xx,yy = np.where(img_c>20.0)
    # get temperature values for marked areas
    zz = img_c[xx,yy]
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

    for x,y,z in zip(xxr,yyr,zzr):
        points.InsertNextPoint(float(z),float(y),float(x))
    np.nan_to_num(dd,copy=False)
    tt = numpy_support.numpy_to_vtk(dd,array_type=vtk.VTK_FLOAT)
    tt.SetName("Temperature")
    pointSource.GetOutput().GetPointData().AddArray(tt)
    pointSource.GetOutput().GetPointData().SetActiveScalars("Temperature")

pointSource.SetExecuteMethod(readPointsRotRbf)

# convert point source to voxel
vox = vtk.vtkVoxelModeller()
#vox.SetModelBounds(pointSource.GetOutput().GetBounds())
vox.SetSampleDimensions(20,20,20)
vox.SetScalarTypeToFloat()
vox.SetInputConnection(pointSource.GetOutputPort())
vox.Update()

# voluume mapper
mapper = vtk.vtkSmartVolumeMapper()
mapper.SetBlendModeToComposite()
mapper.SetInputConnection(vox.GetOutputPort())
mapper.Update()

# volume properties
prop = vtk.vtkVolumeProperty()
prop.ShadeOff()
prop.SetInterpolationType(vtk.VTK_LINEAR_INTERPOLATION)

# color transfer function
# color
color = vtk.vtkColorTransferFunction()
color.SetColorSpaceToRGB()
# build the color transfer function using values from matplotlib colormap
def buildFunctionFromColormap(ctf,name='hot',numpoints=1000):
    import matplotlib.cm as cmap
    # get the desired color map
    cm = cmap.get_cmap(name)
    # get the range of temperature values assigned to the Scalar array in
    # pointSource
    lims = pointSource.GetOutput().GetScalarRange()
    # iterate over a simulated range of scalar (temperature) values
    # normalize each value and extract the color from the colormap
    # the colormap returns BGRA
    for cc in np.linspace(mmin,mmax,num=numpoints):
        ctf.AddRGBPoint(cc,*cm((cc-lims[0])*(lims[1]-lims[0])**-1)[:-1])
    # build functions using the newly added values
    ctf.Build()
    return
    
buildFunctionFromColormap(color)
prop.SetColor(color)
prop.Modified()

# volume
vol = vtk.vtkVolume()
vol.SetMapper(mapper)
vol.SetProperty(prop)
ren.AddViewProp(vol)
ren.ResetCamera()

iren.Initialize()

print("Setting background color")
ren.SetBackground(1.0,1.0,1.0)
renwin.SetSize(500, 500)
renwin.Render()

iren.Start()
