import vtk
import os

renwin = vtk.vtkRenderWindow()
#renwin.AddObserver("EndEvent",lambda obj,event: print("window: ",event))
# create interactor
iren = vtk.vtkRenderWindowInteractor()
iren.AddObserver("ExitEvent",lambda obj,event: print("interactor: ",event))
iren.SetRenderWindow(renwin)
ren = vtk.vtkRenderer()
renwin.AddRenderer(ren)

## define button
button_img = vtk.vtkImageData()

def createImage(img,color1,color2):
    img.SetDimensions(10,10,1)
    img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR,3)
    dims = img.GetDimensions()

    for y in range(dims[1]):
        for x in range(dims[0]):
            if (x<5):
                img.SetScalarComponentFromDouble(x,y,0,0,color1[0])
                img.SetScalarComponentFromDouble(x,y,0,1,color1[1])
                img.SetScalarComponentFromDouble(x,y,0,2,color1[2])
            else:
                img.SetScalarComponentFromDouble(x,y,0,0,color2[0])
                img.SetScalarComponentFromDouble(x,y,0,1,color2[1])
                img.SetScalarComponentFromDouble(x,y,0,2,color2[2])

gen_icons = False

# data objects for what the button will look like in the different states
icon1 = vtk.vtkImageData()
icon2 = vtk.vtkImageData()
# colors to use
col1 = [ 227, 207, 87 ]
col2 = [255, 99, 71]
# create the icons
if gen_icons:
    createImage(icon1,col1,col2)
    createImage(icon2,col2,col1)
else:
    reader = vtk.vtkPNGReader()
    reader.SetFileName("D:\BEAM\Scripts\BoundingBoxOptsGUI\camera-icon-blank.png")
    reader.Update()
    icon1 = reader.GetOutput()
    reader.SetFileName("D:\BEAM\Scripts\BoundingBoxOptsGUI\camera-icon-click.png")
    reader.Modified()
    icon2 = reader.GetOutput()
    
# create the look of the button
button_rep = vtk.vtkTexturedButtonRepresentation2D()
button_rep.SetNumberOfStates(2)
button_rep.SetButtonTexture(0,icon1)
button_rep.SetButtonTexture(1,icon2)

# create the buttonm
button = vtk.vtkButtonWidget()
#button_rep.AddObserver("AnyEvent",lambda obj,event: print("button rep: "))
button.AddObserver("AnyEvent",lambda obj,event: print("button: ",event,event,obj.GetRepresentation().GetState()))
button.SetInteractor(iren)
button.SetRepresentation(button_rep)
# force to state 0
#button.GetRepresentation().SetState(0)

ren.SetBackground(.1,.2,.5)
renwin.Render()

## place button in window in upper right hand corner
upperRight = vtk.vtkCoordinate()
upperRight.SetCoordinateSystemToNormalizedDisplay()
# starting coordinate is bottom left
upperRight.SetValue(1.0,1.0)
# size of button
sz = 50.0
# bounds
bds = [0.0]*6
bds[0] = upperRight.GetComputedDisplayValue(ren)[0]-sz
bds[1] = bds[0]+sz
bds[2] = upperRight.GetComputedDisplayValue(ren)[1]-sz
bds[3] = bds[2] + sz
bds[4] = bds[5] = 0.0

button_rep.SetPlaceFactor(1)
button_rep.PlaceWidget(bds)
button.On()

iren.Start()
