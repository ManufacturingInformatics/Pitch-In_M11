import vtk
import glob
import os
import matplotlib.cm
import h5py
import numpy as np

class MouseInteractorHighlightActor(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self,parent=None):
        self.AddObserver("LeftButtonPressEvent",self.generateBoxClick)
        self.AddObserver("RightButtonPressEvent",self.rightButtonPressEvent)
        self.LastPickedActor = None
        self.cubeSelect = None
        self.LastPickedProperty = vtk.vtkProperty()
        self.no_image=True

    def setStackBounds(self,bb):
        # pass on the limits of the generated volume to the class
        self.bounds = bb

    def setFilenames(self,fnames):
        self.imgnames = fnames

    def setImageViewer(self,view):
        self.imageViewer = view
        
    def generateBoxClick(self,obj,event):
        # get the location of click
        clickPos = self.GetInteractor().GetEventPosition()
        picker = vtk.vtkPropPicker()
        picker.Pick(clickPos[0],clickPos[1],0,self.GetDefaultRenderer())
        # see if something was clicked
        if picker.GetActor() is not None:
            # get click location in terms of world coordinates
            clickPos = picker.GetPickPosition()
            # update slice position/image index of image reader
            self.imageViewer.SetZSlice(click2Index(clickPos[2]))
            # update to force change to viewer
            self.imageViewer.Render()
            print("Index: {}".format(click2Index(clickPos[2])))
            # check if the cube has been created previously, if not create it
            if self.cubeSelect is None:
                # create cube source
                self.cubeSelect = vtk.vtkCubeSource()
                # set the size of the cube
                self.cubeSelect.SetBounds(self.bounds[0],self.bounds[1],self.bounds[2],self.bounds[3],0.0,0.1)
                ## set the color of each face
                # create color array assigning a color to each vertex
                cols = vtk.vtkFloatArray()
                for i in range(8):
                    cols.InsertValue(i,i)
                # convert to poly data
                box = self.cubeSelect.GetOutput()
                box.GetPointData().SetScalars(cols)
                # create mapper to turn it into an object
                self.cubeMapper = vtk.vtkPolyDataMapper()
                self.cubeMapper.SetInputConnection(self.cubeSelect.GetOutputPort())
                # create actor to generate it
                self.cubeActor = vtk.vtkActor()
                self.cubeActor.SetMapper(self.cubeMapper)
                # add actor to renderer
                self.GetDefaultRenderer().AddActor(self.cubeActor)
            # get and move the centre of the cube to the selected z axis coordinate
            cc = list(self.cubeSelect.GetCenter())
            # convert click position to list so it can be edited
            cc[2] = list(clickPos)[2]
            self.cubeSelect.SetCenter(cc)
        self.OnLeftButtonDown()
        return
        
    def leftButtonPressEvent(self,obj,event):
        ## get the actor selected and adjust it's color to make it appear highlighted
        # get the location of the click
        clickPos = self.GetInteractor().GetEventPosition()
        print(clickPos)
        # create object that can select things plotted
        picker = vtk.vtkPropPicker()
        # pick the actor that is in the current location
        picker.Pick(clickPos[0],clickPos[1],0,self.GetDefaultRenderer())
        # get and save the desired frame index
        self.frame_index = click2Index(picker.GetPickPosition()[2])
        # get the thing that was selected
        self.NewPickedActor = picker.GetActor()
        # if something was selected
        if self.NewPickedActor:
            # if something was selected previously, reset it's properties
            print(self.LastPickedActor)
            if self.LastPickedActor is not None:
                self.LastPickedActor.GetProperty().DeepCopy(self.LastPickedProperty)
            # save properties of picked actors
            self.LastPickedProperty.DeepCopy(self.NewPickedActor.GetProperty())
            ## highlight actor
            self.NewPickedActor.GetProperty().SetColor(1.0,0.0,0.0)
            self.NewPickedActor.GetProperty().SetDiffuse(1.0)
            self.NewPickedActor.GetProperty().SetSpecular(0.0)
            # save last picked actor
            self.LastPickedActor = self.NewPickedActor
        self.OnLeftButtonDown()
        return

    def rightButtonPressEvent(self,obj,event):
        # on right click "deslect"/stop highlighting the actor
        # restore it's properties
        if self.LastPickedActor is not None:
            self.LastPickedActor.GetProperty().DeepCopy(self.LastPickedProperty)
        self.OnRightButtonDown()
        return

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

# save number of frames
num_frames = len(ff)
print("Number of frames: ",num_frames)
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

print("Setting up Marching Cubes")
# extract isosurface from images
marching = vtk.vtkMarchingCubes()
marching.ComputeScalarsOff()
marching.SetInputConnection(reader.GetOutputPort())
# setting the starting value of the surface to 15 degrees C when saved as an imag
# starts building the surface around the target temperature values
marching.SetValue(0,norm2image(15.0))
marching.Update()

print("Setting up Stripper and extractor")
# create triangle strips from isosurface
# faster rendering
stripper = vtk.vtkStripper()
stripper.SetInputConnection(marching.GetOutputPort())
# take the data and generate poly data
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(stripper.GetOutputPort())
mapper.ScalarVisibilityOff()
mapper.Update()
# get bounds of formed shape
bounds = mapper.GetBounds()
# function for estimating frame index
def click2Index(clickz,bb=bounds,ff=num_frames):
    # click is the world position coordinates where the click occured
    # bounds is the limit of the shape
    # the click z position is normalized to the image index range
    # [zmin, zmax] |-> [num_frames, 0]
    return int((clickz-bounds[-2])*((-num_frames)/(bounds[-1]-bounds[-2]))+num_frames)
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

print("Setting up highlight manager")
# attach button event handler
style = MouseInteractorHighlightActor()
style.SetDefaultRenderer(ren)
style.setStackBounds(mapper.GetBounds())
style.setFilenames(ff)
iren.SetInteractorStyle(style)

print("Setting up grayscale lookup table")
lut = vtk.vtkLookupTable()
# setting color range to only generate grayscale values
lut.SetHueRange(0.0,0.0)
lut.SetSaturationRange(0.0,0.0)
lut.SetValueRange(0.0,1.0)
# setting range of values to range of possible image values
lut.SetNumberOfTableValues(int(reader.GetOutput().GetScalarTypeMax()))
lut.SetRange(0.0,reader.GetOutput().GetScalarTypeMax())
lut.Build()

print("Setting up image recoloror")
colormap = vtk.vtkImageMapToColors()
colormap.SetLookupTable(lut)
colormap.PassAlphaToOutputOn()
colormap.SetInputConnection(reader.GetOutputPort())

print("Setting up image viewer")
imgIRen = vtk.vtkRenderWindowInteractor()
# create image viewer
imgViewer = vtk.vtkImageViewer()
imgViewer.SetInputConnection(reader.GetOutputPort())
# attach interactor with image viewer
# imageviewer is a wrapper class for a window and renderer
imgViewer.SetupInteractor(imgIRen)
imgViewer.Render()
# assign observer to png reader so whenever the slices are changed
# the image viewer is updated
reader.AddObserver("EndEvent",lambda obj,event : imgViewer.Render())
style.setImageViewer(imgViewer)

# add actor to renderer
ren.AddActor(actor)
# initialize window interactor
iren.Initialize()
print("Starting...")
# generate window
renwin.Render()
# set window name
renwin.SetWindowName("Waifu2x Scaled Arrow Shape")
# start viewer
imgIRen.Start()
# start interactor
iren.Start()


