import vtk
import glob
import os
import matplotlib.cm

# get list of filenames
files = glob.glob(r"D:\BEAM\RPiWaifu\Pngs"+r"\*.png")
# select subset
ff = []
for i in range(0,len(files),100):
    ff.append(files[i])

print("Number of frames: ",len(ff))
# setup vtk file names list
print("Sorting out VTK file list")
filePath = vtk.vtkStringArray()
files.sort(key=lambda x : int(os.path.splitext(os.path.basename(x))[0].split('f')[1]))
filePath.SetNumberOfValues(len(ff))
for i in range(0,len(ff),1):
    filePath.SetValue(i,ff[i])

# setup vtk png reader
print("Setting up reader")
reader = vtk.vtkPNGReader()
reader.SetFileNames(filePath)
reader.SetDataSpacing(1,1,1)
# calling update gets the reader to read in the images
reader.Update()

print("Setting up color and opacity options")
colorFunc = vtk.vtkColorTransferFunction()
colorFunc.SetColorSpaceToDiverging()
def buildMatplotlibCTF(ctf,cmap='hot'):
    # get the target colormap
    cm = matplotlib.cm.get_cmap(cmap)
    # images are 16-bit
    ctf.AddRGBPoint(0.0,0.0,0.0,0.0)
    ctf.AddRGBPoint(1.0,*cm(1.0)[:-1])
    #ctf.SetColorSpaceToRGB()
    # add points to function to interpolate between from the colormap
    # essentially rebuild the colormap
    for i in range(256):        
        colorFunc.AddRGBPoint(i,*cm(float(i)/255.0)[:-1])
    # build the transfer function
    ctf.Build()
    return colorFunc
## setup filter to customise data
buildMatplotlibCTF(ctf=colorFunc)
# opacity
opacity = vtk.vtkPiecewiseFunction()
opacity.AdjustRange([0.0,2**16])
opacity.AddPoint(0.0,0.0)
opacity.AddPoint(2**16,1.0)

# properties of volume
volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(colorFunc)
volumeProperty.SetScalarOpacity(opacity)
volumeProperty.SetInterpolationTypeToNearest()
volumeProperty.SetIndependentComponents(2)

print("Setting up mapper")
## create mapper for stack of images from reader
volumeMapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
volumeMapper.SetInputConnection(reader.GetOutputPort())
volumeMapper.SetBlendModeToMaximumIntensity()

print("Creating volume object")
## create volume object
volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

print("Creating display")
## create renderer
ren = vtk.vtkRenderer()
ren.AddVolume(volume)
ren.SetBackground(0,0,0)

## adding axes to display
print("Adding axes")
ax = vtk.vtkAxesActor()
transform = vtk.vtkTransform()
transform.Translate(10.0,10.0,10.0)
transform.Scale(200.0,200.0,200.0)
ax.SetUserTransform(transform)
ax.SetXAxisLabelText("X")
ren.AddActor(ax)

## create window + attach renderer
win = vtk.vtkRenderWindow()
win.AddRenderer(ren)
win.SetSize(600,600)

## create interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(win)

# display
interactor.Initialize()
win.Render()
interactor.Start()



