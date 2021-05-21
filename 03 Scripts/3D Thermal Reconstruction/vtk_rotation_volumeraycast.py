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
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.numpy_interface import algorithms as alg
  
from scipy.interpolate import Rbf

# index to start on
starter_f = 93199
# initial ff
ff = starter_f
# filtering limit
filt = 0.0
# rotation resolution
rot_res = 10.0
# path to hdf5 file
path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"

print("Getting data limts")
with h5py.File(path,'r') as file:
    mmax = np.nanmax(file['pi-camera-1'][()])
    mmin = np.nanmin(file['pi-camera-1'][()])

print("Getting data integer limits")
with h5py.File(path,'r') as file:
    gg = np.nanmax(file['pi-camera-1'][()],axis=(0,1))
    imax = np.unique(np.sort(gg).astype('int'))[-2]

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
    else:
        return -1,-1

def getDataInCT(frame,ct):
    mask = np.zeros(frame.shape,np.uint8)
    mask = cv2.drawContours(mask,[ct],0,1,-1)
    return frame*mask

pointSource = vtk.vtkProgrammableSource()
def readPointsRotRbf():
    global ff
    # clear data and register blank point set
    output = pointSource.GetPolyDataOutput()
    points = vtk.vtkPoints()
    output.SetPoints(points)
    # collect frame
    with h5py.File(path,'r') as file:
        frame = file['pi-camera-1'][:,:,ff]
    np.nan_to_num(frame,copy=False)
    # find bounding box and contour used to generate it
    bb,ct = findBB(frame)
    if (bb == -1) or (bb is None):
        print("Failed to find contours or bounding box")
        return np.empty(0),np.empty(0),np.empty(0)
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
    for r in np.arange(0.0,360.0,rot_res):
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
    np.nan_to_num(dd,copy=False)
    tt = numpy_support.numpy_to_vtk(dd,array_type=vtk.VTK_FLOAT)
    tt.SetName("Temperature")
    pointSource.GetOutput().GetPointData().AddArray(tt)
    pointSource.GetOutput().GetPointData().SetActiveScalars("Temperature")
    print("Finished updating points")

# setup collection of daata
pointSource.SetExecuteMethod(readPointsRotRbf)

# convert point data to triangles
##tri = vtk.vtkDataSetTriangleFilter()
##tri.SetInputConnection(pointSource.GetOutputPort())
##tri.SetTetrahedraOnly(1)
##tri.GetOutput().GetPointData().SetActiveScalars("Temperature")
##tri.Update()

delny = vtk.vtkDelaunay3D()
delny.SetInputConnection(pointSource.GetOutputPort())
delny.SetTolerance(10.0)
delny.SetAlpha(0.2)
delny.BoundingTriangulationOff()

# color
color = vtk.vtkColorTransferFunction()
color.SetColorSpaceToRGB()
#color.AddRGBPoint(0.0,0.0,0.0,1.0)
#color.Build()
def buildFunctionFromColormap(ctf,name='hot',numpoints=1000):
    import matplotlib.cm as cmap
    cm = cmap.get_cmap(name)
    lims = pointSource.GetOutput().GetScalarRange()
    for cc in np.linspace(mmin,mmax,num=numpoints):
        ctf.AddRGBPoint(cc,*cm((cc-lims[0])*(lims[1]-lims[0])**-1)[:-1])
    ctf.Build()
    
#buildFunctionFromColormap(color)

color.SetColorSpaceToHSV()
color.HSVWrapOff()
color.AddRGBPoint(0.0,0.0,0.0,1.0)
color.AddRGBPoint(imax,1.0,0.0,0.0)

# opacity
opacity = vtk.vtkPiecewiseFunction()
# hide values below 15.0 C
##opacity.AddPoint(0.0,0.0)
##opacity.AddPoint(14.0,0.0)
### make everything above half opaque
##opacity.AddPoint(15.0,0.5)

opacity.AddPoint(0.0,0.0)
opacity.AddPoint(imax,1.0)

# volume properties
volprop = vtk.vtkVolumeProperty()
volprop.SetScalarOpacity(opacity)
volprop.SetColor(color)
volprop.ShadeOff()
volprop.SetInterpolationTypeToLinear()

# mapper
mapper = vtk.vtkUnstructuredGridVolumeRayCastMapper()
#mapper = vtk.vtkUnstructuredGridVolumeZSweepMapper()
#mapper = vtk.vtkProjectedTetrahedraMapper()
mapper.SetBlendModeToMaximumIntensity()
mapper.SetInputConnection(delny.GetOutputPort())

# volume
vol = vtk.vtkVolume()
vol.SetMapper(mapper)
vol.SetProperty(volprop)

## create graphics stuff
ren = vtk.vtkRenderer()
ren.SetBackground(0,0,0)
renWin = vtk.vtkRenderWindow()
renWin.SetSize(500,500)
renWin.AddRenderer(ren)

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

style = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(style)

ren.AddVolume(vol)

# reset camera
ren.ResetCameraClippingRange()
ren.ResetCamera()

iren.Initialize()
iren.Start()



