import numpy as np
import h5py
import vispy.color
from skimage.filters import sobel,threshold_otsu
from numba import jit
from scipy.interpolate import Rbf
import vtk
import cv2

def findBB(frame):
    from skimage.filters import sobel,threshold_otsu
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

@jit(parallel=True,forceobj=True)
def rotate_data(xxr,yyr,zzr,T,rot_res=1.0):
    zz = zzr.copy()
    # form coordinate data set, xyz
    data  = np.array([[x,y,z] for x,y,z in zip(xxr,yyr,zzr)])
    ## rotate the dataset around the y axis
    for r in np.arange(0.0,360.0,rot_res):
        # generate rotation matrix for y rotation
        rot = np.array([[np.cos(180.0*(np.pi**-1)*r),    0,      np.sin(180.0*(np.pi**-1)*r)  ],
                    [0,                     1,      0                   ],
                    [-np.sin(180.0*(np.pi**-1)*r),   0,      np.cos(180.0*(np.pi**-1)*r)  ]
                    ])
        # apply rotation matrix
        vv = np.dot(data,rot.T)
        # udpate position matrix
        xxr = np.concatenate((xxr,vv[:,0]))
        yyr = np.concatenate((yyr,vv[:,1]))
        zzr = np.concatenate((zzr,vv[:,2]))
        # add the temperature values onto the end
        T = np.concatenate((T,zz.ravel()))
    return xxr,yyr,zzr,T

# create vtk data source to update
pointSource = vtk.vtkProgrammableSource()
# index to start at
starter_f = 91399
# path to hdf5 file to access
path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"
# function to collect a frame of data and process
def getData(f,filt=0.45):
    output = pointSource.GetPolyDataOutput()
    points = vtk.vtkPoints()
    output.SetPoints(points)
    with h5py.File(path,'r') as file:
        #tmax = np.nanmax(file['pi-camera-1'][()],axis=(0,1,2))
        #tmin = np.nanmin(file['pi-camera-1'][()],axis=(0,1,2))
        frame = file['pi-camera-1'][:,:,f]

    frame_norm = (frame-frame.min())*(frame.max()-frame.min())**-1
    frame_norm *= 255
    frame_norm = frame_norm.astype('uint8')
    bb,ct = findBB(frame_norm)
    # get values within bounding box
    img = frame[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
    img_c = img.copy()
    # get all cols on second half
    img_c = img_c[img_c.shape[0]//2:,:]
    # get location of values above 20.0
    temp_tol = 20.0
    xx,yy = np.where(img_c>temp_tol)
    print(xx.shape,yy.shape)
    # get temperature values for marked areas
    zz = img_c[xx,yy]
    # fit rbf to the position and temperature data
    rr = Rbf(xx,yy,zz)
    # generate meshgrid for interpolating
    interp_shape = xx.shape[0]*2
    XX,YY = np.meshgrid(np.linspace(xx.min(),xx.max(),interp_shape),np.linspace(yy.min(),yy.max(),interp_shape))
    # use rbf to predict data over meshgrid
    zz = rr(XX,YY)
    # convert coordinate meshgrids to 1d coordinate vectors and make a copy to update
    xxr = XX.ravel()
    yyr = YY.ravel()
    # as the data is in a X-Y plane the z coordinate is 0.0
    zzr = np.zeros(xxr.shape,xxr.dtype)
    # make a copy of the temperature prediction to update
    T= zz.ravel()
    Tmin = T.min()
    Tmax = T.max()
    # rotate the data about the y axis to form a 3d shape
    xxr,yyr,zzr,T = rotate_data(xxr,yyr,zzr,T)
    # generate limit to filter data by
    #lim = Tmin + (Tmax-Tmin)*filt
    lim=0.0
    for x,y,z in zip(zzr[T>=lim],yyr[T>=lim],zzr[T>=lim]):
        points.InsertNextPoint(float(z),float(y),float(x))

# convert data to vtk data source
pointSource.SetExecuteMethod(getData(starter_f,filt=0.0))
tri = vtk.vtkDataSetTriangleFilter()
tri.SetInputConnection(pointSource.GetOutputPort())
tri.SetTetrahedraOnly(1)
tri.Update()
output = tri.GetOutput()

iss = output.GetPointData().SetActiveScalars("Point Label")
#assert(iss>-1)

drange = [0,1]

# Create transfer mapping scalar value to opacity.
opacityFunction = vtk.vtkPiecewiseFunction()
opacityFunction.AddPoint(drange[0], 0.0)
opacityFunction.AddPoint(drange[1], 1.0)

# Create transfer mapping scalar value to color.
colorFunction = vtk.vtkColorTransferFunction()
colorFunction.SetColorSpaceToHSV()
colorFunction.HSVWrapOff()
colorFunction.AddRGBPoint(drange[0], 0.0, 0.0, 1.0)
colorFunction.AddRGBPoint(drange[1], 1.0, 0.0, 0.0)

volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetScalarOpacity(opacityFunction)
volumeProperty.SetColor(colorFunction)
volumeProperty.ShadeOff()
volumeProperty.SetInterpolationTypeToLinear()
# volumeProperty.SetScalarOpacityUnitDistance(options.unit)

#volumeMapper = vtk.vtkUnstructuredGridVolumeRayCastMapper()
volumeMapper = vtk.vtkUnstructuredGridVolumeZSweepMapper()
# volumeMapper = vtk.vtkProjectedTetrahedraMapper()
# volumeMapper.SetBlendModeToMaximumIntensity()
volumeMapper.SetInputData(output)

volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

# create a rendering window and renderer
renderer = vtk.vtkRenderer()
renderer.SetBackground(0,0,0)

window = vtk.vtkRenderWindow()
window.SetSize(512,512)
window.AddRenderer(renderer)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)

style = vtk.vtkInteractorStyleTrackballCamera();
interactor.SetInteractorStyle(style);

renderer.AddVolume(volume)

scalarBar = vtk.vtkScalarBarActor()
scalarBar.SetLookupTable(colorFunction);
scalarBar.SetOrientationToVertical();
scalarBar.SetPosition( 0.85, 0.7 );
scalarBar.SetPosition2( 0.1, 0.3 );
propT = vtk.vtkTextProperty()
propL = vtk.vtkTextProperty()
propT.SetFontFamilyToArial()
propT.ItalicOff()
propT.BoldOn()
propL.BoldOff()
scalarBar.SetTitleTextProperty(propT);
scalarBar.SetLabelTextProperty(propL);
scalarBar.SetLabelFormat("%5.2f")
renderer.AddActor(scalarBar)

renderer.ResetCameraClippingRange()
renderer.ResetCamera()

interactor.Initialize()
window.Render()
interactor.Start()



