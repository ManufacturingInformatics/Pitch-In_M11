import vtk
import glob
import os
import matplotlib.cm
import h5py
import numpy as np

# get list of filenames
files = glob.glob(r"D:\BEAM\RPiWaifu\Pngs"+r"\*.png")
# select subset
ff = []
for i in range(0,len(files),1):
    ff.append(files[i])

print("Getting dataset limits")
path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"
with h5py.File(path,'r') as file:
    mmax,mmin = np.nanmax(file['pi-camera-1'][()],axis=(0,1,2)),np.nanmin(file['pi-camera-1'][()],axis=(0,1,2))

def norm2image(T,vmax=mmax,vmin=mmin):
    return int((T-vmin)*((2**16)/(vmax-vmin)))

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
#reader.AddObserver("ProgressEvent",lambda obj,event : print("{:.2f}".format(obj.GetProgress()),end='\r',sep=''))
reader.SetFileNames(filePath)
# set how the images are stacked next to each other
reader.SetDataSpacing(1,1,.1)
# calling update gets the reader to read in the images
reader.Update()

print("Setting up flying edges")
fedge = vtk.vtkFlyingEdges3D()
fedge.SetInputConnection(reader.GetOutputPort())
bb = reader.GetOutput().GetScalarRange()
# generate 5 contours across the range
fedge.GenerateValues(5,bb[0],bb[1])
fedge.ComputeNormalsOn()
fedge.ComputeGradientsOn()
fedge.Update()

print("Setting up Stripper and extractor")
# create triangle strips from isosurface
# faster rendering
stripper = vtk.vtkStripper()
stripper.SetInputConnection(fedge.GetOutputPort())
# take the data and generate poly data
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(stripper.GetOutputPort())
mapper.ScalarVisibilityOff()
# pass connection onto actor to produce the object
actor = vtk.vtkActor()
actor.SetMapper(mapper)
# setup color
#actor.GetProperty().SetColor(0.0,1.0,0.0)
actor.GetProperty().SetSpecular(.3)
actor.GetProperty().SetSpecularPower(20)
actor.GetProperty().SetOpacity(.5)

## setting up the renderer and window
print("Setting up renderer and window")
ren = vtk.vtkRenderer()
# set background color
ren.SetBackground(0.0,0.0,0.0)
renwin = vtk.vtkRenderWindow()
renwin.AddRenderer(ren)
renwin.SetSize(500,500)
iren = vtk.vtkRenderWindowInteractor()
# define interactor
iren.SetRenderWindow(renwin)
# add actor to renderer
ren.AddActor(actor)
# initialize window interactor
iren.Initialize()

print("setting up VRML exporter")
exp = vtk.vtkVRMLExporter()
exp.SetFileName("flying_edges.wrl")
exp.SetRenderWindow(renwin)
exp.Update()

print("Setting camera")
# set camera to centre of the start of the stack
bounds = fedge.GetOutput().GetBounds()
cam = ren.GetActiveCamera()
cam.SetRoll(168.20386785906155)
cam.SetFocalPoint(0.0,0.0,0.0)
cam.SetPosition(-163.88639442833895, -154.80614053319522, -612.5128369126282)
cam.SetDistance(652.6834353714366)
cam.SetViewAngle(30.0)
# add an observer to print event type and camera positioin
print("Starting...")
# generate window
renwin.Render()
renwin.SetWindowName("Image Stack -> Flying Edges")

print("exporting model")
# exporting model
exp.Write()
# start interactor
iren.Start()
