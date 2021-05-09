import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from GridFrame import ClickGridCanvas
from tkinter import Tk,Button,filedialog,Toplevel,Image,Frame,Scrollbar,Checkbutton,TOP,BOTTOM,BOTH,RAISED,SUNKEN,Menu,Scale,HORIZONTAL,N,S,E,W,CENTER,Label,Entry,StringVar,DoubleVar,BooleanVar,IntVar
from tkinter.ttk import Notebook,Treeview,Separator
from scipy.interpolate import Rbf
import h5py
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from skimage.filters import sobel,threshold_otsu
import cv2
import vtk


""" Bounding Box Data Selection GUI

David Miller 2019, The University of Sheffield

Created December 3rd 2019

A Tkinter GUI to aid in investigating and recording the impact of using different regions of data to generate the 3D rendered shape

"""

# class for keeping track of VTK error messages
class VTKWindowMessageLog(Frame):
    def __init__(self,master=None):
        # perform initialization 
        Frame.__init__(self,master)
        # save parent
        self.master=master
        # define Treeview to host messages
        self.msg_tree = Treeview(self,columns=("MessageType","Message","Location"),show="tree")
        # define scrollbar
        self.tree_scroll = Scrollbar(self,orient='vertical',command=self.msg_tree.yview)
        # setup y scrolling action with scrollbar
        self.msg_tree.configure(yscrollcommand=self.tree_scroll)
        
        # add tree view which will be updated with messages
        self.msg_tree['show']='headings'
        # set column headings
        self.msg_tree.heading("MessageType",text="Type")
        self.msg_tree.heading("Message",text="Message")
        self.msg_tree.heading("Location",text="Location")
        # set initial width and anchor
        self.msg_tree.column("MessageType",width=200,anchor=CENTER)
        self.msg_tree.column("Message",width=200,anchor=CENTER)
        self.msg_tree.column("Location",width=200,anchor=CENTER)

        # capture output and funnel into a variable that can be accessed and parsed
        self.file_win = vtk.vtkStringOutputWindow()
        self.file_win.PromptUserOff()
        # assign output window to the file class
        vtk.vtkOutputWindow.SetInstance(self.file_win)
        # assign observer to update tree everytime the file is updated
        self.file_win.AddObserver("WarningEvent",self.updateDispLog)
        self.file_win.AddObserver("ErrorEvent",self.updateDispLog)
        self.file_win.AddObserver("EndEvent",lambda: print("Finished event!"))
        
        self.msg_tree.pack(side='left',fill=BOTH,expand=1)
        self.tree_scroll.pack(side='right',fill=BOTH,expand=1)
    # update tree when a new message comes in
    def updateDispLog(self,obj,event):
        # output is an increasing multi-line string representing the set of errors
        # split by new line and remove empty strings
        # get last two entries
        log_new = [s for s in self.file_win.GetOutput().splitlines() if s][-2:]
        # parse it into different sections
        msg_type,loc = log_new[0].split(": In ")
        # add to tree
        self.msg_tree.insert('','end',text="",values=(msg_type,log_new[1],loc))
        self.msg_tree['show']='tree headings'

# class for managing screenshots
# assign render window at construct or using set render window method
class RenderWindowScreenshot(vtk.vtkPNGWriter):
    def __init__(self,renwin=None):
        import os
        super().__init__()
        # create class to convert window to image
        self.imgRen = vtk.vtkWindowToImageFilter()
        # if render window is set, attach window to image renderer
        if renwin is not None:
            self.imgRen.SetInput(renwin)
            # capture transparency channel
            self.imgRen.SetInputBufferTypeToRGB()
            self.imgRen.ReadFrontBufferOff()
            self.imgRen.Update()
            # attach to writer
            self.SetInputConnection(self.imgRen.GetOutputPort())
            self.Update()
        # counter for number of screenshots taken
        # used in default fileFormat
        self._ct = 0
        # file format
        # set to f string based off file where the class is used and image counter
        self.SetFilePattern("{:d}.png")
        # file path
        # initialized to current working directory
        self.filePath = os.getcwd()
    # function for generating a new filename each click
    # called inside click
    # saves repeated calls to set filename
    def genNewFileName(self):
        return self.GetFilePattern().format(self.getNumberOfImagesTaken())
    # method for setting the file path
    # is concatenated against the current file format
    def setFilepath(self,newp):
        self.filePath = newp
    # fuunction for setting/updating the render window to capture from
    def setRenderWindow(self,renwin):
        self.imgRen.SetInput(renwin)
        self.imgRen.SetInputBufferTypeToRGB()
        self.imgRen.ReadFrontBufferOff()
        self.imgRen.Update()
        # attach to writer
        self.SetInputConnection(self.imgRen.GetOutputPort())
        self.Update()
    # return the render window assigned to the internal instance of vtkWindowToImageFilter
    def getRenderWindow(self):
        return self.imgRen.GetInput()
    # take screenshot and write to file
    # generates new filename each call based on self.fileFormat
    # extra arguments are so that it can be used as an observer callback
    def click(self,obj=None,event=None):
        # update image converter
        self.imgRen.Modified()
        self.imgRen.Update()
        # update filename
        self.SetFileName(os.path.join(self.filePath,self.genNewFileName()).replace('\\','/'))
        self.Update()
        # save image
        self.Write()
        # increment counter
        self._ct+=1
    # return the number of screenshots taken based off internal counter
    def getNumberOfImagesTaken(self):
        return self._ct

class OptionsMenu:
    def __init__(self,master,controller):
        self.master = master
        self.master.title("Edit Options")
        # save copy of reference
        self.controller = controller
        # create notebook for tabbed interface
        self.nb = Notebook(self.master)
        ## create frames to group options
        self.file_opts = Frame(self.nb)
        self.data_opts = Frame(self.nb)
        self.vtk_opts = Frame(self.nb)
        ## add stuff to frames
        # file options
        self.path_label = Label(self.file_opts,text="File Path",font='Arial 10 bold')
        self.path_box = Entry(self.file_opts,width=100,textvariable=self.controller.curr_file)
        # add button to access file selection dialog
        self.path_select = Button(self.file_opts,text="...",command=self.selectFile)
        # screenshot folder options
        self.sshot_label = Label(self.file_opts,text="Screenshot Path",font='Arial 10 bold')
        self.sshot_box = Entry(self.file_opts,width=100,textvariable=self.controller.sshot_folder)
        # add button to access file selection dialog
        self.sshot_select = Button(self.file_opts,text="...",command=self.selectSShot)
        # screenshot filename format
        self.sshot_name_format = StringVar(value=self.controller.sshot.GetFilePattern())
        self.sshot_name_format.trace('w',self.updateScreenshotFormat)
        self.sshot_name_label = Label(self.file_opts,text="Screenshots Filenames",font='Arial 10 bold')
        self.sshot_name_entry = Entry(self.file_opts,width=100,textvariable=self.sshot_name_format)
        self.sshot_name_help = Button(self.file_opts,text="?",command=self.showSShotFnameFormatHelp)

        ## Data options
        # changet filter variable for data
        self.filt_label = Label(self.data_opts,text="Filter",font='Arial 10 bold')
        self.filt_box = Entry(self.data_opts,textvariable=self.controller.filt)
        self.filt_help = Button(self.data_opts,text="?",command=self.showFilterVariableHelp)
        # change rotation resolution
        self.rotres_label = Label(self.data_opts,text="Rotation Resolution",font='Arial 10 bold')
        self.rotres_box = Entry(self.data_opts,textvariable=self.controller.rot_res)
        self.rotres_help = Button(self.data_opts,text="?",command=self.showRotResVariableHelp)
        # Checkbuttones for using either static or calculated bounding box
        self.bounding_box_label = Label(self.data_opts,text="Bounding Box:",font='Arial 10 bold')
        self.use_dbox_var = BooleanVar(value=not self.controller.use_fixed_box.get())
        self.use_dynamic_box = Checkbutton(self.data_opts,text="Dynamic",variable=self.use_dbox_var,command=self.updateDynamicBox)
        self.use_static_box = Checkbutton(self.data_opts,text="Static",variable=self.controller.use_fixed_box,command=self.updateStaticBox)

        # frame for grouping static box entries
        self.static_box_ctrls = Frame(self.data_opts)
        ## static box location controls
        # variables for saving values
        self.width_var = IntVar(value=10)
        self.height_var = IntVar(value=10)
        # top left corner of bounding box
        self.tl_row = IntVar(value=0)
        self.tl_col = IntVar(value=0)
        # labels
        self.static_box_wlabel = Label(self.static_box_ctrls,text="Width")
        self.static_box_hlabel = Label(self.static_box_ctrls,text="Height")
        self.static_box_tlrow_lb = Label(self.static_box_ctrls,text="TL Row")
        self.static_box_tlcol_lb = Label(self.static_box_ctrls,text="TL Column")
        # boxex to modify values
        self.static_box_width = Entry(self.static_box_ctrls,textvariable=self.width_var)
        self.static_box_height = Entry(self.static_box_ctrls,textvariable=self.height_var)
        self.static_box_tlrow = Entry(self.static_box_ctrls,textvariable=self.tl_row)
        self.static_box_tlcol = Entry(self.static_box_ctrls,textvariable=self.tl_col)
        # update button to set bounding box
        self.update_static_box = Button(self.static_box_ctrls,text="Update",command=self.updateBoundingBox)

        ## if the use dynamic checkbox is ticked, disable the static one and entries
        if self.use_dbox_var.get():
            # ensure the other box is cleared
            self.controller.use_fixed_box.set(False)
            # disable static controls
            self.setFrameState(self.static_box_ctrls,False)
        # if static box has been enabled
        elif self.controller.use_fixed_box.get():
            # clear the other box
            self.use_dbox_var.set(False)
            # ensure the static box entries are enabled
            self.setFrameState(self.static_box_ctrls,True)

        ## VTK Options
        self.bk_label = Label(self.vtk_opts,text="Background",font='Arial 10 bold')
        # intialize array of variables
        self.col_vals = [DoubleVar(value=c) for c in controller.vtk_bk]
        for c,col in enumerate(self.col_vals):
            col.trace('w',lambda *args,idx=c:self.checkBkColLimits(idx=idx))
        # initialize array of entries setting variable
        self.channels = [Entry(self.vtk_opts,textvariable=self.col_vals[cc]) for cc in range(3)]
        # add button that updates the background color
        self.bk_update = Button(self.vtk_opts,text="Update",command=self.updateBkColor)

        self.delny_col_label = Label(self.vtk_opts,text="Delauny Color",font='Arial 10 bold')
        # intialize array of variables
        self.delny_col_vals = [DoubleVar(value=c) for c in controller.dlny_col]
        for c,col in enumerate(self.delny_col_vals):
            col.trace('w',lambda *args,idx=c:self.checkDelnyColLimits(idx=idx))
        # initialize array of entries setting variable
        self.delny_col_channels = [Entry(self.vtk_opts,textvariable=self.delny_col_vals[cc]) for cc in range(3)]
        # add button that updates the background color
        self.delny_col_update = Button(self.vtk_opts,text="Update",command=self.updateDelnyColor)
        ## VTK window size
        self.vtk_size_label = Label(self.vtk_opts,text="Window Size",font='Arial 10 bold')
        self.vtk_size_vals = [IntVar(value=int(v)) for v in controller.vtk_size]
        self.vtk_size_entry = [Entry(self.vtk_opts,textvariable=v) for v in self.vtk_size_vals]
        self.vtk_size_update = Button(self.vtk_opts,text="Update",command=self.updateVTKWindowSize)

        ## geometry manager
        self.bounding_box_label.grid(row=2,column=0)
        self.use_dynamic_box.grid(row=3,column=0)
        self.use_static_box.grid(row=3,column=1)
        # pack the static box controls inside sub frame
        self.static_box_tlrow_lb.grid(row=0,column=0)
        self.static_box_tlrow.grid(row=0,column=1)
        self.static_box_wlabel.grid(row=0,column=2)
        self.static_box_width.grid(row=0,column=3)
        self.static_box_tlcol_lb.grid(row=1,column=0)
        self.static_box_tlcol.grid(row=1,column=1)
        self.static_box_hlabel.grid(row=1,column=2)
        self.static_box_height.grid(row=1,column=3)
        self.update_static_box.grid(row=0,column=4)
        # place frame holding static box controls
        self.static_box_ctrls.grid(row=4,column=0,columnspan=2)

        # path entries and labels
        self.path_label.grid(row=0,column=0)
        self.path_box.grid(row=0,column=1)
        self.path_select.grid(row=0,column=2)

        self.sshot_label.grid(row=1,column=0)
        self.sshot_box.grid(row=1,column=1)
        self.sshot_select.grid(row=1,column=2)
        
        self.sshot_name_label.grid(row=2,column=0)
        self.sshot_name_entry.grid(row=2,column=1)
        self.sshot_name_help.grid(row=2,column=2)

        # filtering parameter and rotation resolution
        self.filt_label.grid(row=0,column=0)
        self.filt_box.grid(row=0,column=1)
        self.filt_help.grid(row=0,column=2)
        self.rotres_label.grid(row=1,column=0)
        self.rotres_box.grid(row=1,column=1)
        self.rotres_help.grid(row=1,column=2)

        # VTK color settings
        self.vtk_size_label.grid(row=2,column=0)
        self.vtk_size_entry[0].grid(row=2,column=1)
        self.vtk_size_entry[1].grid(row=2,column=2)
        self.vtk_size_update.grid(row=2,column=4)

        # iterating over color variable entries
        self.delny_col_label.grid(row=1,column=0)
        for c in range(1,4):
            self.delny_col_channels[c-1].grid(row=1,column=c)
        self.delny_col_update.grid(row=1,column=c+1)

        self.bk_label.grid(row=0,column=0)
        for c in range(1,4):
            self.channels[c-1].grid(row=0,column=c)
        self.bk_update.grid(row=0,column=c+1)
        
        # add to notebook
        self.nb.add(self.file_opts,text='File')
        self.nb.add(self.data_opts,text='Data')
        self.nb.add(self.vtk_opts,text='VTK')

        # setup notebook
        self.nb.grid(row=0,column=0,sticky=N+S+E+W)

    def updateBoundingBox(self):
        # update the bounding box variable
        # if the user changes back to dynamic box, it is overriden in updatePlot method
        # variables are in OpenCV arrangement, (tl row, tl col, width, height)
        self.controller.bb = [self.tl_row.get(),self.tl_col.get(),self.width_var.get(),self.height_var.get()]

    @staticmethod
    def setFrameState(f,enable=False,enableState="normal",disableState="disabled"):
        # iterate over all children in frame setting the state to 
        for child in f.winfo_children():
            # if the child has a state option in config
            # use it to update the state
            if "state" in child.configure().keys():
                child.configure(state=(disableState if not enable else enableState))
            # if the child is in fact another widget containing children
            # iterate over that setting the state
            elif len(child.winfo_children())>0:
                setFrameState(child,enable=enable)

    def updateDynamicBox(self):
        # if dynamic box is enabled
        if self.use_dbox_var.get():
            # clear the other one
            self.controller.use_fixed_box.set(False)
            # disable static box entries
            self.setFrameState(self.static_box_ctrls,False)
        else:
            # if both checkbox are False, ensure the dbox one is enabled
            # as it was last enable
            if not self.controller.use_fixed_box.get():
                self.use_dbox_var.set(True)
            
    def updateStaticBox(self):
        # if use static box has been enabled
        if self.controller.use_fixed_box.get():
            # disable the other one
            self.use_dbox_var.set(False)
            # renable static box entries
            self.setFrameState(self.static_box_ctrls,True)
        else:
            # if both are false, renable static box as it was last set
            if not self.use_dbox_var.get():
                self.controller.use_fixed_box.set(True)

    # check and update new file format for screenshots
    def updateScreenshotFormat(self):
        new_format = self.sshot_name_format.get()
        # ensure file extension is png
        if ".png" not in new_format:
            new_format += ".png"
            self.sshot_name_format.set(self.sshot_name_format.get()+".png")
        # update the file pattern of RenderWindowScreenshot instance inside main GUI using passed reference controller
        self.controller.sshot.SetFilePattern(new_format)
        # update instance to ensure things are up to date
        self.controller.sshot.Update()

    # open help menu describing the format of generated screenshot filenames
    def showSShotFnameFormatHelp(self):
        self.format_help = Toplevel()
        self.format_help.title("Screenshot Filename Format Help")
        help_label = Label(self.format_help,justify='left',text="""Screenshot filenames are generated based on a formatted Python 3 string containing an integer formatter.
The integer is replaced by an internal counter keeping track of the number of
screenshots taken so far starting from 0, so by default the first screenshot will be called
{0}, the second {1}, third {2} etc. The format MUST contain a :d formatter and the file extension
is ALWAYS png.""".format(self.controller.sshot.GetFilePattern().format(0),self.controller.sshot.GetFilePattern().format(1),self.controller.sshot.GetFilePattern().format(2)))
        self.format_help_link = Label(self.format_help,text=r"https://docs.python.org/3.1/library/string.html#format-string-syntax",fg='blue')
        self.format_help_link.bind("<Button-1>",self.open_help_link)
        
        # geometry
        help_label.grid(row=0,column=0,columnspan=2)
        self.format_help_link.grid(row=1,column=0)

    def showFilterVariableHelp(self):
        self.filt_help = Toplevel()
        self.filt_help.title("Filter Variable Help")
        help_label = Label(self.filt_help,justify='left',text="""The filter variable is used to try and filter down the temperature data to just the values we're interested in.
It is used as a tolerance and only the temperature values that are above it are used in the 3D reconstruction. The variable represents the percentage of the dataset temperature
range the tolerance is set as the minimum temperature plus the set percentage of the temperature range. It is initialized to 0.6 as early experiments showed it to be a good
starting point. An adaptive means of determining this has yet to be found as there's no suitable quality metric to evaluate the output.""")
        help_label.grid(row=0,column=0)

    def showRotResVariableHelp(self):
        self.rotres_help = Toplevel()
        self.rotres_help.title("Rotation Resolution Variable Help")
        help_label = Label(self.rotres_help,justify='left',text="""The filtered data is rotated to form a 3D shape by rotating it about the Y-axis. This variable controls the
resolution of the rotation. The units of the rotation are in degrees. Setting this variable to a small value drastically increases the rendering time.""")
        help_label.grid(row=0,column=0)
        
    # open help link
    def open_help_link(self,*args):
        import webbrowser
        # open help link in web browser
        webbrowser.open_new(self.format_help_link.cget("text"))

    def updateVTKWindowSize(self):
        # update the desired window size
        # used in the next call to setupVTK
        self.controller.vtk_size = [v.get() for v in self.vtk_size_vals]

    # trace callback for checking if updated VTk background color is within limits
    def checkBkColLimits(self,idx,*args):
        if self.col_vals[idx].get()>1.0:
            self.col_vals[idx].set(1.0)
        elif self.col_vals[idx].get()<0.0:
            self.col_vals[idx].set(0.0)

    # trace callback for checking if updated VTK Delaunay color is within limits
    def checkDelnyColLimits(self,idx,*args):
        if self.delny_col_vals[idx].get()>1.0:
            self.delny_col_vals[idx].set(1.0)
        elif self.delny_col_vals[idx].get()<0.0:
            self.delny_col_vals[idx].set(0.0)

    # callback for background button to update background color value
    def updateBkColor(self):
        self.controller.vtk_bk = [self.col_vals[c].get() for c in range(3)]
        if hasattr(self.controller,'ren'):
            self.controller.ren.SetBackground(self.controller.vtk_bk)
            self.controller.ren.Update()

    # callback for delaunay button to update delaunay color value
    def updateDelnyColor(self):
        self.controller.dlny_col = [self.delny_col_vals[c].get() for c in range(3)]
        if hasattr(self.controller,'triangulation'):
            self.controller.triangulation.SetColor(self.controller.dlny_col)
            self.controller.triangulation.Update()

    ## independent update 
    def selectFile(self):
        # open file dialog and allow the user to only select HDF5 data file
        # updating string variable triggers updatePath callback which updated the main menu current file path
        temp = filedialog.askopenfilename(initialdir=os.path.dirname(self.controller.curr_file.get()),title="Select HDF5 file to inspect",filetypes=[("HDF5 files (*.hdf5)","*.hdf5")])
        # double check that the path points to a file
        if(os.path.isfile(temp)):
            # update path variable for label
            self.controller.curr_file.set(temp)
            # open file and get the number frames
            # rebuild the slider to the number of frames
            with h5py.File(self.path_var.get(),'r') as file:
                self.controller.sindex.config(to=file['pi-camera-1'].shape[2])
            # update plot with first frame of file
            self.controller.updatePlot(None)

    def selectSShot(self):
        # open directory dialog and ask user to select new folder for screenshots
        # updating string variable triggers updateSShot callback which updated the main menu current file path
        temp = filedialog.askdirectory(initialdir=os.path.dirname(self.controller.sshot_folder.get()),title="Select folder for screenshots to be saved to")
        if os.path.isdir(temp):
            self.controller.sshot_folder.set(temp)

# wrapper gui for displaying the current data stored in vtkMassProperties 
class ShowMassProperties:
    def __init__(self,master,props,units=['pixels^3','pixels^3','pixels^2']):
        self.master = master
        self.master.title("Mass Properties")
        self.props = props
        # title labels
        self.vol_title = Label(master,text="Volume {}:".format(units[0]),font='Arial 10 bold')
        self.proj_vol_title = Label(master,text="Projected Volume {}:".format(units[1]),font='Arial 10 bold')
        self.surf_area_title = Label(master,text="Surface Area {}:".format(units[2]),font='Arial 10 bold')
        # label variables
        self.vol_var = StringVar(value="{:.2f}".format(props.GetVolume()))
        self.proj_vol_var = StringVar(value="{:.2f}".format(props.GetVolumeProjected()))
        self.surf_area_val = StringVar(value="{:.2f}".format(props.GetSurfaceArea()))
        # set variables
        # value labels
        self.vol_val = Label(master,textvariable=self.vol_var,font='Arial 10 bold')
        self.proj_vol_val = Label(master,textvariable=self.proj_vol_var,font='Arial 10 bold')
        self.surf_area_val = Label(master,textvariable=self.surf_area_val,font='Arial 10 bold')
        # geometrry manager
        self.vol_title.grid(row=0,column=0,sticky='nsew')
        self.proj_vol_title.grid(row=1,column=0,sticky='nsew')
        self.surf_area_title.grid(row=2,column=0,sticky='nsew')
        self.vol_val.grid(row=0,column=1,sticky='nsew')
        self.proj_vol_val.grid(row=1,column=1,sticky='nsew')
        self.surf_area_val.grid(row=2,column=1,sticky='nsew')
    # update values in show mass properties
    # pulls values from the current reference
    def update(self):
        self.vol_var.set("{:.2f}".format(self.props.GetVolume()))
        self.proj_vol_var.set("{:.2f}".format(props.GetVolumeProjected()))
        self.surf_area_val.set("{:.2f}".format(props.GetSurfaceArea()))

class AboutWindow:
    def __init__(self,master):
        self.master = master
        self.master.title("About")

        # frame to hold all information
        self.info_frame = Frame(master,relief='sunken')

        ## information to display
        self.descr_label = Label(self.info_frame,text="Description")
        self.author_label = Label(self.info_frame,text="David Miller, Copyright 2019")
        self.org_label = Label(self.info_frame,text="The University of Sheffield")
        self.email_label = Label(self.info_frame,text="d.b.miller@sheffield.ac.uk")
        # added a separator to split the basic information from the buttons to access other windows
        self.desc_sep = Separator(self.info_frame,orient=HORIZONTAL)
        # button to open a window that displayed global dependencies and their respective versions
        self.dep_button = Button(self.info_frame,text="Libraries",command=self.showDependencies)
        # button for displaying copyright information 
        self.copyright_button = Button(self.info_frame,text="Copyright")
        self.copyright_sep = Separator(self.info_frame,orient=HORIZONTAL)
        # current version of this GUI
        self.version_label = Label(self.info_frame,text="Version 1.0.0")

        # set layout
        self.descr_label.grid(row=0,column=0)
        self.author_label.grid(row=1,column=0)
        self.org_label.grid(row=2,column=0)
        self.email_label.grid(row=3,column=0)
        self.desc_sep.grid(row=4,column=0)
        self.dep_button.grid(row=5,column=0)
        self.copyright_sep.grid(row=6,column=0)
        self.version_label.grid(row=7,column=0)
        self.info_frame.grid(row=0,column=0)

    def showDependencies(self):
        """ Function for creating and displaying a window that shows the program's global dependencies"""
        import types
        import pkg_resources
        # row number for labels
        r=1
        # new window
        self.depend_win = Toplevel()
        # set title
        self.depend_win.title("GUI Dependencies")
        # add subtitle
        self.depend_win.sub_title = Label(self.depend_win,text="Global Dependencies",font='Arial 10 bold')
        self.depend_win.sub_title.grid(row=0,column=0)
        # list of labels
        self.depend_win.labels = []
        # iterate over global imports
        for name,val in globals().items():
            if isinstance(val,types.ModuleType):
                # add label and set text to module name and version imported
                # if the package has a version attribute, use it to get the version number
                if hasattr(val,'__version__'):
                    self.depend_win.labels.append(Label(self.depend_win,text="{} {}".format(val.__name__,val.__version__)))
                # if not, use pkg_resources to get version
                else:
                    try:
                        self.depend_win.labels.append(Label(self.depend_win,text="{} {}".format(val.__name__,pkg_resources.get_distribution(val.__name__).version)))
                    except pkg_resources.DistributionNotFound:
                        self.depend_win.labels.append(Label(self.depend_win,text="{} {}".format(val.__name__,"N/A")))
                # set row of label
                self.depend_win.labels[-1].grid(row=r,column=0)
                # increase row number
                r+=1
        
class BoundingBoxGUI:
    def __init__(self,master):
        # set master
        self.master = master
        # update title
        self.master.title("Bounding Box Options GUI")
        # variable controlling how the bounding box is split in half 
        self.box_split=0
        self.split_opts = {"geo": 0, "max":1}

        # flag to indicate that the VTK window has been destroyed
        self.vtk_dead = True

        ## paths
        # create string variables
        self.curr_file = StringVar(value=os.getcwd().replace('\\','/'))
        self.sshot_folder = StringVar(value=os.getcwd().replace('\\','/'))

         # create screenshot class and attach to window
        self.sshot = RenderWindowScreenshot()
        # intialize folder
        self.sshot.setFilepath(self.sshot_folder.get())
        # set the trace on the path string variable to update screenshot class as well
        self.sshot_folder.trace('w',self.updateSShotPath)

        # flag to force use of fixed
        self.use_fixed_box = BooleanVar(value=False)
        
        ## data options
        # settings for filtering data
        self.rot_res = DoubleVar(value=10.0)
        # rotation resolution
        self.filt = DoubleVar(value=0.6)

        ## vtk options
        # window background color
        self.vtk_bk = [1.0,1.0,1.0]
        # delauny object color
        self.dlny_col = [1.0,0.0,0.0]
        # size of the window
        self.vtk_size = [500,500]
        
        ## create a menu bar
        # create top level menu
        self.menuBar = Menu(self.master)
        # assign to root
        self.master.config(menu=self.menuBar)
        
        # create file menu
        self.filemenu = Menu(self.menuBar,tearoff=0)
        # add option to open file
        # calls file dialog to allow user to select file
        self.filemenu.add_command(label='Open New HDF5..',command=self.selectFile)
        # open screenshots folder only if it's a valid folder
        self.filemenu.add_command(label='Open Screenshots..',command=lambda: os.startfile(self.sshot_folder.get()) if os.path.isdir(self.sshot_folder.get()) else False)
        # add to top level menu
        self.menuBar.add_cascade(label='File',menu=self.filemenu)
        
        # create Edit menu to adjust settings of things
        # assign menu to top level
        self.editMenu = Menu(self.menuBar,tearoff=0)
        # open dialog to select directory
        self.editMenu.add_command(label='Set Screenshot Folder..',command=self.selectSShot)
        # add to upper menu
        self.menuBar.add_cascade(label='Edit',menu=self.editMenu)

        # Add Options menu
        self.optionsMenu = Menu(self.menuBar,tearoff=0)
        # create options menu to display and edit options
        self.optionsMenu.add_command(label='Open Menu..',command=self.createOptsMenu)
        self.menuBar.add_cascade(label='Options',menu=self.optionsMenu)

        # Add View menu
        self.viewMenu = Menu(self.menuBar,tearoff=0)
        self.viewMenu.add_command(label='Mass Props..',command=self.showMassProps)
        self.viewMenu.add_command(label='VTK Log..',command=self.showVTKLog)
        self.menuBar.add_cascade(label='View',menu=self.viewMenu)

        # Add about Menu
        self.helpMenu = Menu(self.menuBar,tearoff=0)
        self.helpMenu.add_command(label="About GUI",command=self.showAboutGUI)
        self.menuBar.add_cascade(label='Help',menu=self.helpMenu)
        
        ## create click grid 
        self.cgrid = ClickGridCanvas(self.master,2,2,width=500,height=500,borderwidth=5,background='white')
        
        # allow only a max of 2 cells to be selected out of the grid
        self.cgrid.max_select =2
        # override left click handler with internal method
        # performs ClickGridCanvas callback method and then overrides it with check
        self.cgrid.bind("<Button-1>",self._gridcallback)

        ## Matplotlib frame
        # create wrapper frame so figure can be used wtih grid
        self.fig_frame = Frame(self.master)
        # create figure
        self.fig = Figure(figsize=(5,5),dpi=100)
        # add axes
        self.axes = self.fig.add_subplot(111)
        # as we're going to display an image, hide the axes
        self.axes.get_xaxis().set_ticks([])
        self.axes.get_yaxis().set_ticks([])
        # create canvas
        # initialize with fig and assigm to fig_frame
        self.fig_canvas = FigureCanvasTkAgg(self.fig,self.fig_frame)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(side=TOP,fill=BOTH,expand=True)
        # add navigation toolbar
        self.fig_toolbar = NavigationToolbar2Tk(self.fig_canvas,self.fig_frame)
        self.fig_toolbar.update()
        self.fig_canvas._tkcanvas.pack(side=TOP,fill=BOTH,expand=True)
        # add key press handler
        self.fig_canvas.mpl_connect("key_press_event",self.on_key_press)
        # buttons to set which mid point to use
        self.geo_half = Button(self.master,text="Geometric",font='Arial 16 bold',bg='light gray',command=lambda:self.change_boxsplit("geo"),relief=('sunken' if self.box_split==0 else 'raised'))
        self.max_half = Button(self.master,text="Maximum",font='Arial 16 bold',bg='light gray',highlightbackground='#3E4149',command=lambda:self.change_boxsplit("max"),relief=('sunken' if self.box_split==1 else 'raised'))

        # slider for selecting frame
        self.sindex_label = Label(self.master,text="File Frame Index",font='Arial 16 bold')
        # buttons for moving by one frame
        self.sindex = Scale(self.master,from_=0,to=1,resolution=1,orient=HORIZONTAL,command=self.updatePlot)
        self.incr_frame = Button(self.master,text=">>",height=2,width=5,bg='light gray',font='Arial 10 bold',command=lambda: self.changeSindex(val=1))
        self.decr_frame = Button(self.master,text="<<",height=2,width=5,bg='light gray',font='Arial 10 bold',command=lambda: self.changeSindex(val=-1))
        self.render_frame = Button(self.master,text="Render",height=2,bg='light gray',width=5,font='Arial 10 bold',command=lambda: self.updatePlot())
        
        ## geometry manager
        self.geo_half.grid(row=0,rowspan=2,column=0,padx=(10,10),pady=(10,10))
        self.max_half.grid(row=0,rowspan=2,column=1,padx=(10,10),pady=(10,10))
        self.fig_frame.grid(row=2,rowspan=2,columnspan=2,sticky='nsew')
        self.sindex_label.grid(row=0,column=2,columnspan=3)
        self.sindex.grid(row=2,column=2,columnspan=3,sticky='ew')
        self.incr_frame.grid(row=1,column=2,padx=(10,10),pady=(10,10),sticky='ew')
        self.decr_frame.grid(row=1,column=4,padx=(10,10),pady=(10,10),sticky='ew')
        self.render_frame.grid(row=1,column=3,padx=(10,10),pady=(10,10),sticky='ew')
        self.cgrid.grid(row=3,column=2,columnspan=3,sticky='nsew')

        # assign weights to the top level so it expands
        num_cols,num_rows = master.grid_size()
        for c in range(num_cols):
            master.columnconfigure(c,weight=1)
        for r in range(num_rows):
            master.rowconfigure(r,weight=1)

        self.master.update()

        # store min size of the window
        self.minwidth = master.winfo_width()
        self.minheight = master.winfo_height()

        # force min size to saved size
        master.after_idle(lambda: master.minsize(self.minwidth, self.minheight))

    def showAboutGUI(self):
        t = Toplevel()
        self.help_win = AboutWindow(t)

    def updateSShotPath(self,*args):
        self.sshot.setFilepath(self.sshot_folder.get())

    def changeSindex(self,val):
        self.sindex.set(self.sindex.get()+val)

    def showVTKLog(self):
        if hasattr(self,'vtk_log'):
            # create new window
            t = Toplevel()
            # clone configuration of frame new new window
            frame_copy = self.cloneTo(self.vtk_log,t)
            # get the tree inside copy
            # winfo_children returns list of widgets inside frame_copy
            # filter searches list for widgets that are of type Treeview. this returns an iterator
            tree_copy = next(filter(lambda x: isinstance(x,Treeview),frame_copy.winfo_children()))
            # iterate over items in the logger transferring them over to the other tree
            for child in self.vtk_log.msg_tree.get_children():
                tree_copy.insert("",'end',text=self.vtk_log.msg_tree.item(child)["text"],values=self.vtk_log.msg_tree.item(child)["values"])
            # add button to export the file
            exportButton = Button(t,text="Export",command=self.exportLog)
            exportButton.pack(side=TOP,fill=BOTH,expand=1,padx=(5,5),pady=(5,5))
            # pack
            frame_copy.pack(side=BOTTOM,fill=BOTH,expand=1,padx=(5,5),pady=(5,5))

    def exportLog(self):
        from datetime import datetime
        if hasattr(self,"vtk_log"):
            # get user to select location, filename and type of file to export log to
            # returns a file object
            with filedialog.asksaveasfile(mode='w',initialdir=os.path.dirname(os.getcwd()),initialfile=datetime.now().strftime("vtklog-%d-%m-%y-%H-%M-%S.txt"),title="Select what to export the log as",filetypes=[("All files (*.*)","*.*")]) as file:
                # iterate over entries in treeview
                for item in self.vtk_log.msg_tree.get_children():
                    # iterate over values writing each as a column of delimited strings
                    # current delimiter is ","
                    for v in self.vtk_log.msg_tree.item(item)['values']:
                        file.write(str(v)+",")
                    file.write("\n")

    # from https://stackoverflow.com/a/46507018
    # create duplicate of the target widget assigned to the same parent
    def clone(self,widget):
        # importing _tkinter gives us access to tkinter specific errors
        from tkinter import _tkinter
        # determine parent of widget
        parent = widget.nametowidget(widget.winfo_parent())
        # get class
        cls = widget.__class__
        # intialize class, eq to __init__(self,master=parent)
        clone = cls(parent)
        # iterate over configuration options and copy the settings from one to the other
        for key in widget.configure():
            # attempt to access configure options
            # if it's read only, a TclError is generated and caught
            try:
                clone.configure({key: widget.cget(key)})
            except _tkinter.TclError:
                continue
        return clone

    # clone widget and initialize contents on new parent
    def cloneTo(self,widget,newP):
        # importing _tkinter gives us access to tkinter specific errors
        from tkinter import _tkinter
        # get class
        cls = widget.__class__
        # intialize class, eq to __init__(self,master=parent)
        clone = cls(newP)
        # iterate over configuration options and copy the settings from one to the other
        for key in widget.configure():
            # attempt to access configure options
            # if it's read only, a TclError is generated and caught
            try:
                clone.configure({key: widget.cget(key)})
            except _tkinter.TclError:
                continue
        return clone

    def showMassProps(self):
        if hasattr(self,'vol'):
            t = Toplevel()
            self.vol.Update()
            self.mass_show = ShowMassProperties(t,self.vol)

    def setupVTK(self):
        # clear flag to indicate that the vtk window has been created
        print("clearing death flag")
        self.vtk_dead = False
        print("Setting up VTK...")
        print("Setting up renderer, window and interactor...")
        ## VTK update portion
        # create VTK objects if necessary
        self.ren = vtk.vtkRenderer()
        # create window
        self.renwin = vtk.vtkRenderWindow()
        self.renwin.AddRenderer(self.ren)
        # create interactor
        self.iren = vtk.vtkRenderWindowInteractor()
        # assign exit event to render window interactor
        # triggered when the render window it is attached to closes
        self.iren.AddObserver("ExitEvent",self.__setVTKDeadFlag)
        self.iren.SetRenderWindow(self.renwin)
        self.iren.Initialize()

        print("Setting up VTK message log")
        # generate hidden frame that is never packed/shown
        # log can now be updated without needing to be rendered
        self.log_frame = Frame(self.master)
        # assign message log to frame so it has somewhere to sit and update
        self.vtk_log = VTKWindowMessageLog(self.log_frame)

        print("Setting up screenshot class")
        self.sshot.setRenderWindow(self.renwin)
        # ensure file path is updated
        self.sshot.setFilepath(self.sshot_folder.get())

        print("Setting up button for taking screenshots")
        # object controlling how the button looks
        sshot_rep = vtk.vtkTexturedButtonRepresentation2D()
        # set number of states button can have
        sshot_rep.SetNumberOfStates(2)
        ## read in icons
        reader = vtk.vtkPNGReader()
        reader.SetFileName(os.path.join(os.getcwd(),"camera-icon-blank.png"))
        reader.Update()
        sshot_rep.SetButtonTexture(0,reader.GetOutputDataObject(0))
        # reader has to be rebuilt
        # update/modify methods doesn't update it properly
        reader = vtk.vtkPNGReader()
        reader.SetFileName(os.path.join(os.getcwd(),"camera-icon-click.png"))
        reader.Update()
        sshot_rep.SetButtonTexture(1,reader.GetOutputDataObject(0))
        
        # create widget assigning representation
        self.sshot_button = vtk.vtkButtonWidget()
        # add handler for when the button is pressed
        self.sshot_button.AddObserver("StateChangedEvent",self.takeSShot)
        self.sshot_button.SetRepresentation(sshot_rep)
        self.sshot_button.SetInteractor(self.iren)
        self.sshot_button.EnabledOn()

        # create source of data
        print("Setting up pointSource")
        self.pointSource = vtk.vtkProgrammableSource()
        # set update method as updatePlot
        self.pointSource.SetExecuteMethod(self.updatePlot)
        print("Setting up Delaunay")
        # create Delaunay object
        self.delny = vtk.vtkDelaunay3D()
        self.delny.SetInputConnection(self.pointSource.GetOutputPort())
        self.delny.SetTolerance(10.0)
        self.delny.SetAlpha(0.2)
        self.delny.BoundingTriangulationOff()
        
        # Shrink the result to help see it better.
        print("Setting up shrink filter")
        self.shrink = vtk.vtkShrinkFilter()
        self.shrink.SetInputConnection(self.delny.GetOutputPort())
        self.shrink.SetShrinkFactor(0.9)

        print("Setting up mass properties and associated classes")
        ## setup mass properties
        self.geo = vtk.vtkGeometryFilter()
        self.geo.SetInputConnection(self.delny.GetOutputPort())
        self.geo.Update()

        self.tri = vtk.vtkTriangleFilter()
        self.tri.SetInputConnection(self.geo.GetOutputPort())
        self.tri.Update()

        self.vol = vtk.vtkMassProperties()
        self.vol.SetInputConnection(self.tri.GetOutputPort())
        self.vol.Update()
        self.vol.Modified()

        # set mapper to handle the shrinking
        print("Setting up data mapper")
        self.mapper = vtk.vtkDataSetMapper()
        self.mapper.SetInputConnection(self.geo.GetOutputPort())
        
        # setup actor
        print("Setting up actor")
        self.triangulation = vtk.vtkActor()
        self.triangulation.SetMapper(self.mapper)
        self.triangulation.GetProperty().SetColor(self.dlny_col)

        # setup background
        self.ren.AddActor(self.triangulation)
        # set bk color
        self.ren.SetBackground(self.vtk_bk)
        # set size of the window
        self.renwin.SetSize(self.vtk_size[0],self.vtk_size[1])
        
        print("Rendering result")
        # rendering the window so the title can be changed the button placed
        self.renwin.Render()
        # set window name
        # informs user of which frame has been shown
        self.renwin.SetWindowName("Delaunay 3D of Frame {}".format(self.sindex.get()))
        
        # rotate camera so laser bit is at the top
        print("Setting up camera")
        cam1 = self.ren.GetActiveCamera()
        cam1.Zoom(1.0)
        cam1.Roll(180.0)

        print("Starting VTK")
        # initialize interactor
        self.iren.Initialize()
        # set the size of the window
        self.renwin.SetSize(500,500)
        # render window
        self.renwin.Render()

        print("Placing button")
        # placing in upper right corner
        self.sshot_loc = vtk.vtkCoordinate()
        self.sshot_loc.SetCoordinateSystemToNormalizedDisplay()
        self.sshot_loc.SetValue(1.0,1.0)
        # setting size
        bds = [0.0]*6
        # set the bounds to place the button in the top right hand corner of the frame
        bds[0] = self.sshot_loc.GetComputedDisplayValue(self.ren)[0] - 50.0
        bds[1] = bds[0] + 50.0
        bds[2] = self.sshot_loc.GetComputedDisplayValue(self.ren)[1] - 50.0
        bds[3] = bds[2] + 50.0
        bds[4] = bds[5] = 0.0

        # place button on screen
        sshot_rep.SetPlaceFactor(1)
        sshot_rep.PlaceWidget(bds)
        # enable button
        self.sshot_button.On()

        self.iren.Start()
        print("Finished setting up")

    def takeSShot(self,obj,event):
        # hide button for picture
        self.sshot_button.GetRepresentation().SetVisibility(0)
        # take the picture
        self.sshot.click(obj=obj,event=event)
        # reset button
        obj.GetRepresentation().SetState(0)
        # show button
        self.sshot_button.GetRepresentation().SetVisibility(1)
        
    # method used to catch ExitEvent by renderer interactor to indicate
    # the render window has been closed
    def __setVTKDeadFlag(self,obj,event):
        print("\tvtk dead: ",event)
        self.vtk_dead = True

    def createOptsMenu(self):
        t = Toplevel()
        self.opts_gui = OptionsMenu(master=t,controller=self)
        
    def on_key_press(self,event):
        key_press_handler(event,self.fig_canvas,self.fig_toolbar)

    def change_boxsplit(self,name):
        # change other button to make it appear pressed
        if name=="geo":
            self.max_half.config(relief='raised')
            self.geo_half.config(relief='sunken')
        elif name=="max":
            self.geo_half.config(relief="raised")
            self.max_half.config(relief='sunken')
        self.box_split = self.split_opts[name]
                
    def _gridcallback(self,event):
        # run callback of click grid
        self.cgrid.callback(event)
        # get cells that were selected
        sc = self.cgrid.getSelected()
        # if two tiles were selected, check them
        if len(sc)==2:
            # check if the selected cells are adjacent
            # checks if either the difference in rows is 1 or difference in cols is 1
            # if both are 1 then diagonal cells have been selected
            # the bool call ensures what's returned are booleans. Bitwise XOR returns the correct result
            # as bools are ultimately ints restricted to 0 and 1
            if not (bool(abs(sc[0][0]-sc[1][0])==1) ^ bool(abs(sc[0][1]-sc[1][1])==1)):
                # deslect las tile on list
                self.cgrid.delete(self.cgrid.tiles[sc[-1][0]][sc[-1][1]])
                # update tile id list
                self.cgrid.tiles[sc[-1][0]][sc[-1][1]] = None
        self.cgrid.update_idletasks()
        self.master.after_idle(lambda: self.master.minsize(self.minwidth, self.minheight))

    def updatePlot(self,event=None):
        print("Updating plot!")
        # check if the path is pointing to a valid file
        # mostly here in case the user interacts with the slider before the path is set
        if os.path.isfile(self.curr_file.get()):
            # disable increment frame, decrement frame and render current frame buttons
            # user can't use the buttons anyway as VTK get the main thread, so this is mainly for visual purposes
            self.render_frame.config(relief=SUNKEN,state='disabled')
            self.incr_frame.config(state='disabled')
            self.decr_frame.config(state='disabled')
        
            self.axes.clear()
            print("Getting data")
            # get the data for the current frame
            with h5py.File(self.curr_file.get(),'r') as file:
                frame = file['pi-camera-1'][:,:,int(self.sindex.get())]
            np.nan_to_num(frame,copy=False)
            # convert data to 8-bit image
            frame_norm = (frame-frame.min())/(frame.max()-frame.min())
            frame_norm *= 255
            frame_norm = frame_norm.astype('uint8')
            frame_norm = np.dstack((frame_norm,frame_norm,frame_norm))
            # rotate 90 degs clockwise so it's the right way up
            frame_norm = cv2.rotate(frame_norm,cv2.ROTATE_90_CLOCKWISE)
            # find bounding box
            print("Getting bounding box")
            # if the flag to use static box is not set
            # calculate bounding box
            # if it was set, then the variable would have been set with the options menu
            if not self.use_fixed_box.get():
               self.bb,ct = self.findBB(frame)
            
            # if a bounding box was found
            if (self.bb is not None):
                print("Found a box! ","(",self.bb[:2],")")
                # draw rectangle on frame
                cv2.rectangle(frame_norm,(self.bb[1],self.bb[0]),(self.bb[1]+self.bb[3],self.bb[0]+self.bb[2]),(255,0,0),1)
                # find location of max value
                maxrc = np.unravel_index(np.argmax(frame),frame.shape)[:2]
                # draw cross indicating location
                frame_norm = cv2.drawMarker(frame_norm,maxrc,color=[0,255,0],markerType=cv2.MARKER_CROSS,markerSize=2,thickness=1)
                # show image with bounding box
                # get values within bounding box
                img = frame[self.bb[1]:self.bb[1]+self.bb[3],self.bb[0]:self.bb[0]+self.bb[2]]
                # make a copy of the image
                img_c = img.copy()
                # split bounding box according to geometric settings
                if self.box_split==0:
                    # draw line indicating division along box half
                    cv2.line(frame_norm,(self.bb[1]+int((self.bb[3]//2)),self.bb[0]),(self.bb[1]+int((self.bb[3]//2)),self.bb[0]+self.bb[2]),(0,0,255),1)
                    # get selected cells
                    sc = self.cgrid.getSelected()
                    # if any were selected
                    if len(sc)==2:
                        # sort them by minimum row
                        sc.sort(key=lambda x : x[0])
                        # get data within range
                        img_c = img_c[sc[0][0]*(self.bb[2]//2):(sc[1][0]+1)*(self.bb[2]),sc[0][1]*(self.bb[3]//2):(sc[1][1]+1)*(self.bb[3]//2)]
                        # update display
                    elif len(sc)==1:
                        img_c = img_c[sc[0][0]*(self.bb[2]//2):(sc[0][0]+1)*(self.bb[2]),sc[0][1]*(self.bb[3]//2):(sc[0][1]+1)*(self.bb[3]//2)]
                        # update display
                # if based off location of max
                elif self.box_split==1:
                    # check that max is inside bounding box
                    if ((maxrc[0]>=self.bb[1]) and (maxrc[0]<=(self.bb[1]+self.bb[3]))) and ((maxrc[1]>=self.bb[0]) and (maxrc[1]<=(self.bb[0]+self.bb[2]))):
                        # draw line indicating division is is where maximum is 
                        cv2.line(frame_norm,(self.bb[0],maxrc[1]),(self.bb[0]+self.bb[2],maxrc[1]),(255,0,0),1)
                        # get selected cells
                        sc = self.cgrid.getSelected()
                        # construct grid of sizes for each zone
                        # width and height
                        # arranged so [r][c] notation can be used when selecting it
                        sz = [[[abs(self.bb[0]-maxrc[0]),abs(self.bb[1]-maxrc[1])],[abs((self.bb[0]+self.bb[2])-maxrc[0]),abs(self.bb[1]-maxrc[1])]],
                              [[abs(self.bb[0]-maxrc[0]),abs((self.bb[1]+self.bb[3])-maxrc[1])],[abs((self.bb[0]+self.bb[2])-maxrc[0]),abs((self.bb[1]+self.bb[3])-maxrc[1])]]]
                        # if two cells were selected
                        if len(sc)==2:
                            # sort them by minimum row so the ranges go from left hand side to the right
                            sc.sort(key=lambda x : x[0])
                            # get data within selected cells
                            img_c = img_c[sc[0][0]*sz[sc[0][0]][sc[0][1]][0]: # starting row position, row# x width of first cell
                                               ## end position
                                               sc[0][0]*sz[sc[0][0]][sc[0][1]][0]+ # starting position + 
                                               (sc[1][1]-sc[0][1])*sz[sc[1][0]][sc[1][1]][0], # difference in row positions (either 0 or 1) x width of second cell
                                               sc[0][1]*sz[sc[0][0]][sc[0][1]][1]:
                                               sc[0][1]*sz[sc[0][0]][sc[0][1]][1]+
                                               (sc[1][0]-sc[0][0])*sz[sc[1][0]][sc[1][1]][1]]
                        # if one cell was selected
                        elif len(sc)==1:
                            img_c = img_c[((sc[0][0]-1)>0)*(sz[sc[0][0]-1][sc[0][1]][0]): # starting point is either the begining of the row (0) or size of the adjacent cell if row number is >1
                                          ((sc[0][0]-1)>0)*(sz[sc[0][0]-1][sc[0][1]][0])+sz[sc[0][0]][sc[0][1]][0], # end point is starting point + respective dimension of selected cell
                                          ((sc[0][1]-1)>0)*(sz[sc[0][0]-1][sc[0][1]][1]):((sc[0][1]-1)>0)*(sz[sc[0][0]-1][sc[0][1]][1])+sz[sc[0][0]][sc[0][1]][1]]

                # update image display with added line
                self.axes.get_xaxis().set_ticks([])
                self.axes.get_yaxis().set_ticks([])
                self.axes.imshow(frame_norm,cmap='gray')
                self.fig_canvas.draw()

                # if cells have been selected, setup vtk stuff
                if len(sc)>0:
                    # print shape of the selected range
                    print("User selected range: ",img_c.shape)
                    if self.vtk_dead:
                        self.setupVTK()
                    
                    print("Updating VTK")
                    # get output of point source to update
                    output = self.pointSource.GetPolyDataOutput()
                    points = vtk.vtkPoints()
                    # clear the output
                    output.SetPoints(points)

                    # filter data
                    xx,yy = np.where(img_c>20.0)
                    zz = img_c[xx,yy]
                    # fit rbf
                    print("Data points above 20 C: ",xx.shape,yy.shape)
                    try:
                        print("Fitting RBF")
                        self.rbf = Rbf(xx,yy,zz)
                    except Exception as e:
                        print("Failed to fit RBF to data! Bailing")
                        return
                    # generate meshgrid for interpolation data
                    XX,YY = np.meshgrid(np.linspace(xx.min(),xx.max(),xx.shape[0]*2),np.linspace(yy.min(),yy.max(),yy.shape[0]*2))
                    print("Size of meshgrid: ",XX.shape,YY.shape)
                    # use rbf to generate new data
                    print("Generating new RBF data")
                    zz = self.rbf(XX,YY)

                    # create copies to update
                    dd = zz.ravel()
                    zz = zz.ravel()
                    xxr = XX.ravel()
                    yyr = YY.ravel()
                    zzr = np.zeros(xxr.shape,xxr.dtype)
                    # compile data into xyz array
                    data  = np.array([[x,y,0.0] for x,y in zip(xxr,yyr)])
                    # rotate according to set resolution
                    print("Rotating data")
                    for r in np.arange(0.0,360.0,self.rot_res.get()):
                        # generate rotation object
                        rot = R.from_euler('xyz',[0.0,r,0.0],degrees=True)
                        # apply to matrix
                        vv = rot.apply(data)
                        # udpate position matrix
                        xxr = np.concatenate((xxr,vv[:,0]))
                        yyr = np.concatenate((yyr,vv[:,1]))
                        zzr = np.concatenate((zzr,vv[:,2]))
                        dd = np.concatenate((dd,zz))

                    # filter data in an attempt to remove noise
                    print("Filtering data and updating data set")
                    lim = zz.min() + (zz.max()-zz.min())*self.filt.get()
                    for x,y,z in zip(xxr[dd>=lim],yyr[dd>=lim],zzr[dd>=lim]):
                        points.InsertNextPoint(float(z),float(y),float(x))

            # renable buttons
            self.render_frame.config(relief=RAISED,state='normal')
            self.incr_frame.config(state='normal')
            self.decr_frame.config(state='normal')
            print("Finished updating dataset")

    def selectFile(self):
        # open file dialog and allow the user to only select HDF5 data file
        temp = filedialog.askopenfilename(initialdir=os.path.dirname(self.curr_file.get()),title="Select HDF5 file to inspect",filetypes=[("HDF5 files (*.hdf5)","*.hdf5")])
        if temp is not '':
            self.curr_file.set(temp)
        # double check that the path points to a file
        if(os.path.isfile(self.curr_file.get())):
            # open file and get the number frames
            # rebuild the slider to the number of frames
            with h5py.File(self.curr_file.get(),'r') as file:
                self.sindex.config(to=file['pi-camera-1'].shape[2])
            self.updatePlot()

    def selectSShot(self):
        # open directory dialog and ask user to select new folder for screenshots
        # updating string variable triggers updateSShot callback which updated the main menu current file path
        temp = filedialog.askdirectory(initialdir=os.path.dirname(self.sshot_folder.get()),title="Select folder for screenshots to be saved to")
        if os.path.isdir(temp):
            self.sshot_folder.set(temp)

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
        else:
            return None,None
        
if __name__ == "__main__":         
    root = Tk()
    view = BoundingBoxGUI(root)
    root.mainloop() 

    
