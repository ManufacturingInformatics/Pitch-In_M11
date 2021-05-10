from tkinter import Tk,Label,Frame,Button,Entry,filedialog,messagebox,N,S,E,W,CENTER,TOP,BOTH,StringVar,IntVar,BooleanVar,DoubleVar,LabelFrame,Checkbutton
from tkinter.ttk import Combobox
from tkinter import simpledialog
from tkinter import messagebox

import threading
from queue import Queue

import sounddevice as sd

from sounddevicegui import SoundDevicesDialog

import numpy as np
# import path drawing and selection gui
from CustomExcitationPathsGUI import ComboboxDialog, ScrollableFrame, PathOrderWindow, ChangeFontWindow, ExcitationOrderGUI
# eit solvers and data generators
from pyeit import mesh 
from pyeit.eit.utils import eit_scan_lines
from pyeit.eit.greit import GREIT as greit
from pyeit.eit.bp import BP as bp
from pyeit.eit.jac import JAC as jacobian
from pyeit.eit.interp2d import sim2pts
from pyeit.eit.fem import Forward
from skimage.draw import line as ll
from skimage.transform import radon,iradon_sart,iradon
# log# logging class
import logging
# setup logger
logger = logging.getLogger(__name__)
# matplotlib tkinter plotter classes
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.pyplot import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm,Normalize

# import worker class
from pyEITWorker import ReconstructionWorkerEIT

try:
    import h5py
    hdf5Support = True
except ImportError:
    print("Couldn't find h5py!")
    hdf5Support = False

import os

# recorder class
# used as base for specific file type recorders
class RecorderClass:
    # constructor
    def __init__(self,fname,breakEmpty=False):
        # filename
        self.fname = fname
        # data queue to be popoulated by some process
        self.dataQ = Queue()
        # file object
        self.file = None
        # flag to indicate if the recorder should stop when the input Q is empty
        self.__breakStop = breakEmpty
        # open file object
        # sets self.file
        self.openFile()
        # worker thread performing writing
        self.worker = threading.Thread(target=self.update,args=())
        # set as daemon thread
        # does not start queue as it expects user to setup write method as
        # it varies between data type
        self.worker.daemon=True
        # flag to control inf loop of update
        self.__isrunning = False

    # method for checking if the thread is running and therefore if the class is recording
    def isRecording(self):
        return self.worker.isAlive()

    # method that is called by the worker
    # checks to see if there's data in the queue and calls write method if there is
    def update(self):
        while self.__isrunning:
            if not self.dataQ.empty():
                self.write(self.dataQ.get())
                # indicate that the data has been processed
                self.dataQ.task_done()
            else:
                if self.__breakStop:
                    break
        self.stop()

    # function to start thread
    # sets flag to trigger inf loop
    def start(self):
        self.__isrunning=True
        self.worker.start()

    # process of writing data to the file
    # set by user
    def write(self,data):
        pass

    # function to open file
    # to be set by user
    def openFile(self):
        pass

    # close file method 
    def closeFile(self):
        if self.file is not None:
            if hasattr(self.file,'close'):
                self.file.close()

    # overriden delete handler to ensure that the file object is closed
    # calls closeFile
    def __delete__(self,instance):
        self.closeFile()

    # method to stop the worker
    # clears running flag to stop inf loop
    # then waits for worker to finish
    def stop(self):
        self.__isrunning = False
        if self.worker.isAlive():
            self.worker.join()
        self.closeFile()

# hdf5 recorder class
# derived from RecorderClass
# designed to open and manage a HDF5 file and resizable dataset
class HDF5Recorder(RecorderClass):
    # overriden constructor to receive name of dataset to use when the file is opened
    def __init__(self,fname,dname,initshape=None,breakEmpty=False):
        self.dname = dname
        self.ishape = initshape
        super().__init__(fname,breakEmpty)

    # overriden openFile method to create the file object and reesizable dataset to write to
    def openFile(self):
        # open file
        self.file = h5py.File(self.fname,mode='w')
        # create dataset if shape was given
        if self.ishape is not None:
            self.dset = self.file.create_dataset(self.dname,(*self.ishape,1),maxshape=(*self.ishape,None),dtype='float32')

    # overriden write method for writing to the target dataset and resizing it for the next call
    def write(self,data):
        # if a dataset has not been created
        # create it using data shape and type 
        if not hasattr(self,'dset'):
            self.dset = self.file.create_dataset(self.dname,(*data.shape,1),maxshape=(*data.shape,None),dtype=data.dtype)
            self.ishape = data.shape
        # write data to last index
        self.dset[...,-1] = data
        # increase size of the dataset by one frame
        self.dset.resize(self.dset.shape[:-1] + (self.dset.shape[-1]+1,))

# hdf5 recorder class designed for multiple writers
# derived from recorder class
# designed to open and manage HDF5 file with multiple resizable datasets
class HDF5MultiRecorder(RecorderClass):
    # overriden constructor to accept 
    def __init__(self,fname,dnames,initshapes=None,breakEmpty=False):
        # list of dataset names
        self.__dnames = dnames
        # if initial shapes were provided
        # create dictionary associating initial dataset shape with names
        if initshapes is None:
            self.initshapes = {k:None for k in dnames}
        else:
            self.initshapes = {k:v for k,v in zip(dnames,initshapes)}
        super().__init__(fname,breakEmpty)
        # turn data queue into a dictionary of data queues
        # key corresponds to dataset name
        self.dataQ = {k:Queue() for k in self.__dnames}
        # dictionary of datasets
        self.dsets = {}

    # overriden open method
    def openFile(self):
        # create a file
        # datasets can be added onto it if necessary
        self.file = h5py.File(self.fname,'w')
        # if initial dataset shapes were given on creation
        # create list of boolean checking if all initial shapes are not None
        # if all shape are not None, create datasets 
        if all([v is not None for v in self.initshapes.values()]):
            # populate dictionary with dataset objects initialized with target dataset names and their respective sizes.
            # each dataset is set to be resizable
            self.dsets = {name:self.file.create_dataset(name,(*ishape,1),maxshape=(*ishape,None),dtype='float32') for name,ishape in self.initshapes.items()}

    # overriden write method to specify target dataset
    def write(self,dname,data):
        # if there's no dataset associated with the target name
        # create one and add to dictionary
        # useful if initial datashape size is not known
        if dname not in self.dsets:
            self.dsets[dname] = self.file.create_dataset(dname,(*data.shape,1),maxshape=(*data.shape,None),dtype=data.dtype)
            # create a queue for the dataset
            self.dataQ[dname] = Queue()
            
        # write data to dataset
        self.dsets[dname][...,-1] = data
        # increase dataset size by 1 frame 
        self.dsets[dname].resize(dset.shape[:-1] + (dset.shape[-1]+1,))

    # overriden update method to instead iterate over queues
    # if queue is not empty call write method specifying dataset
    def update(self):
        while self.__isrunning:
            # iterate over data queues
            for k,v in self.dataQ.items():
                # if queue is not empty
                if not v.empty():
                    # write data to dataset
                    # if the dataset doesn't exist, it is created inside write method
                    self.write(k,v.get())
                    v.task_done()
                elif self.__breakFlag:
                    break
        self.stop()
                    
# basic file recorder class
# derived from RecorderClass
# designed to manage basic file types such as csv and txt
class BasicRecorder(RecorderClass):
    # overriden open file method
    # opens in append mode to the data can be added onto it as the worker runs
    def openFile(self):
        self.file = open(self.fname,mode='a')

    # overriden write method
    # uses numpy savetxt function to write the data as it handles datasets well
    # writes as comma separated values
    def write(self,data):
        np.savetxt(self.file,data,delimiter=',')

# data viewing window
# generates a Matplotlib figure plot for displaying the results of a worker thread
class ReconstructionResultsWindow(Frame):
    def __init__(self,master,worker):
        self.master = master
        # run Frame initialization method
        super().__init__(master)
        # flag for running update loop
        self.__updateFlag = False
        # save worker
        self.worker = worker
        # if a worker was given
        if worker is not None:
            # get mesh data
            # used in plotting
            self.x,self.y,self.tri,self.perm,self.elpos = worker.getMeshData()
            # obtain solver type
            self.solverType = worker.getSolverType()

        # recorder class assigned to it
        self.recorder = None

        # flag for initial plot
        self.remakePlot = True
        
        # setup figure
        self.fig = Figure(figsize=(3,3),dpi=100)
        self.ax = self.fig.add_subplot(111)

        # create axes for colorbar
        div = make_axes_locatable(self.ax)
        self.cax = div.append_axes("right",size="5%",pad=0.05)
        # create canvas to attach figure to
        self.figCanvas = FigureCanvasTkAgg(self.fig,self)
        self.figCanvas.draw()
        self.figCanvas.get_tk_widget().pack(side=TOP,fill=BOTH,expand=True)

        self.ax.set_title(rf"$\Delta$ Conductivity Map, {self.solverType}")

        # add toolbar
        self.figToolbar = NavigationToolbar2Tk(self.figCanvas,self)
        self.figToolbar.update()
        # add to gui. always one row below the canvas
        self.figCanvas._tkcanvas.pack(side=TOP,fill=BOTH,expand=True)
        ## add key press handlers
        self.figCanvas.mpl_connect("keypressevent",self.on_key_press)

        # set title
        if hasattr(self.master,'title'):
            self.master.title(self.solverType + " Results")

        # make the main window contents resizable 
        numcols,numrows = master.grid_size()
        for c in range(numcols):
            master.columnconfigure(c,weight=1)
        for r in range(numrows):
            master.rowconfigure(r,weight=1)

    # method for stopping the update loop of the plots
    def stopUpdate(self):
        self.__updateFlag = False

    # method for starting the update loop of the plots
    def startUpdate(self):
        self.__updateFlag = True
        self.update()

    # method for checking if the update loop is running
    def isUpdating(self):
        return self.__updateFlag

    # method to set flag to force remake the plot and colorbar objects
    def rebuildPlots(self):
        self.remakePlot = True

    # method for updating the plot
    # checks to see if the output queue is empty
    # if it isn't, the plot is cleared, new data obtained and the results plotted
    def update(self):
        # if the data queue is not empty
        if not self.worker.outputQ.empty():
            # get new data from worker output Q
            img = self.worker.outputQ.get()
            # if there's a recorder attached to the plot and it is running
            # add data to queue to be written to file
            if self.recorder is not None:
                if self.recorder.isRecording():
                    self.recorder.dataQ.put(img)
            
            ## resolve and update plot according to solver type
            if self.solverType == "GREIT":
                try:
                    # if the plot is being remade
                    # generate plot and set axes parameters
                    if self.remakePlot:
                        # plot results
                        self.im = self.ax.imshow(img,interpolation='none',cmap=cm.rainbow)
                        if hasattr(self,'cb'):
                            self.cv.remove()
                        # generate colorbar
                        self.cb = self.fig.colorbar(self.im,cax=self.cax,orientation='vertical')
                        self.ax.axis('equal')
                        self.remakePlot = False
                    # if the plot has been made then it is just being updated
                    else:
                        # update image data
                        self.im.set_data(img)
                        # change the limits on the colorbar
                        self.im.set_clim(img.min(),img.max())
                        # indicate that the object has been modified
                        self.im.changed()
                        self.ax.axis('equal')
                    # update canvas to ensure results are displayed
                    self.fig.canvas.draw()
                    return
                except Exception as err:
                    messagebox.showerror(title="GREIT Solving Error",message=str(err))
                    return
            # for jacobian solver method
            elif self.solverType == "JAC":
                # run solver
                try:
                    if self.remakePlot:
                        # plot results
                        self.ax.clear()
                        # x,y,tri defines the triangles
                        # img is the values and is converted to colors
                        self.im = self.ax.tripcolor(self.x,self.y,self.tri,img,shading='flat',cmap=cm.rainbow)
                        if hasattr(self,'cb'):
                            self.cb.remove()
                        # setup colorbar and labels
                        self.cb = self.fig.colorbar(self.im,cax=self.cax,orientation='vertical')
                        self.ax.set_title(rf"$\Delta$ Conductivity Map, {self.solverType}")
                        self.ax.axis('equal')
                        #self.remakePlot = False
                    else:
                        self.im.set_array(img)
                        self.im.set_clim(img.min(),img.max())
                    # update canvas to ensure results are displayed
                    self.fig.canvas.draw()
                    return
                except Exception as err:
                    messagebox.showerror(title="JAC Solving Error", message=str(err))
                    return
            # for back propogation method
            elif self.solverType == "BP":
                try:
                    if self.remakePlot:
                        # plot results
                        self.ax.clear()
                        self.im = self.ax.tripcolor(self.x,self.y,self.tri,img,cmap=cm.rainbow)
                        if hasattr(self,'cb'):
                            self.cb.remove()
                        self.cb = self.fig.colorbar(self.im,cax=self.cax,orientation='vertical')
                        self.ax.set_title(rf"$\Delta$ Conductivity Map, {self.solverType}")
                        self.ax.axis('equal')
                        #self.remakePlot=False
                    else:
                        self.im.set_array(img)
                        self.im.set_clim(img.min(),img.max())
                    # update canvas to ensure results are displayed
                    self.fig.canvas.draw()
                    return
                except Exception as err:
                    messagebox.showerror(title="BP Solving Error",message=str(err))
                    return
            # for inverse radon transform
            elif self.solverType == "RADON":
                try:
                    if self.remakePlot:
                        # plot results
                        self.im = self.ax.imshow(img,interpolation='none',cmap=cm.rainbow)
                        self.cb = self.fig.colorbar(self.im,cax=self.cax,orientation='vertical')
                        if hasattr(self,'cb'):
                            self.cb.remove()
                        self.ax.axis('equal')
                        self.remakePlot = False
                    else:
                        self.im.set_data(img)
                        self.im.set_clim(img.min(),img.max())
                    # update canvas to ensure results are displayed
                    self.fig.canvas.draw()
                    return
                except Exception as err:
                    messagebox.showerror(title="RADON Solving Error",message=str(err))
                    return
                
    # key press handler for canvas
    def on_key_press(self,event):
        key_press_handler(event,self.figCanvas,self.figToolbar)

# dialog for adding a reconstruction worker plot to the gui
# based off combobxdialog
class PlotLocationDialog(simpledialog.Dialog):
    def __init__(self,parent,opts,title,rowinit=0,colinit=0):
        # save options
        # used to setup combobox
        self.opts = opts
        self.rowinit=rowinit
        self.colinit=colinit
        super().__init__(parent,title=title)

    # contents of the dialog
    # combobox for choosing solver type
    # entry boxes for entering row and column to place the plot inb
    def body(self,master):
        # label for combobox
        self.optsLabel = Label(master,text="Solver Types",font="Arial 10 bold")
        # combobox representing the different patterns available
        self.optsBox = Combobox(master,state="readonly",values=self.opts)
        # initialize combobox to the first element
        self.optsBox.current(0)

        # entry boxes for row and column of the plot
        self.rowLabel = Label(master,text="Row",font="Arial 10 bold")
        self.colLabel = Label(master,text="Col",font="Arial 10 bold")
        self.row = IntVar(value=self.rowinit)
        self.col = IntVar(value=self.colinit)
        self.rowEntry = Entry(master,textvariable=self.row)
        self.colEntry = Entry(master,textvariable=self.col)

        # geometry manager
        self.optsLabel.grid(row=0,column=0,columnspan=2)
        self.optsBox.grid(row=1,column=0,columnspan=2)
        self.rowLabel.grid(row=2,column=0)
        self.rowEntry.grid(row=3,column=0)
        self.colLabel.grid(row=2,column=1)
        self.colEntry.grid(row=3,column=1)

        # return solver combobox as widget of interest
        # set focus to it on creation
        return self.optsBox

    # on clicking OK
    # returns the chosen solver type, row and column to place the plot in
    def apply(self):
        rtype = self.optsBox.get()
        r = int(self.row.get())
        c = int(self.col.get())
        self.result = rtype,r,c

# dialog wrapper for ExcitationOrderGUI
# user uses it to setup the number of coils, distance and step
class CoilsPathsDialog(simpledialog.Dialog):
    def __init__(self,parent,title):
        self.result = None
        super().__init__(parent,title=title)

    # contents of the dialog
    def body(self,master):
        # draw frame to host gui
        self.drawFrame = Frame(master)
        self.drawPaths = ExcitationOrderGUI(self.drawFrame,nc=16)
        # pack it
        self.drawFrame.grid(row=0,column=0)

    # set the result of the dialog as the number of coils, distance between pairs, step value, excitation order
    def apply(self):
        self.result = self.drawPaths.numCoils.get(),self.drawPaths.dist.get(),self.drawPaths.step.get(),self.drawPaths.exciteOrder

# dialog for setting the properties of the excitation signal
class SetExcitePropertiesDialog(simpledialog.Dialog):
    def __init__(self,parent,title="Set the signal"):
        super().__init__(parent,title=title)

    def body(self,master):
        # labels for the prompt and the entry boxes
        self.promptLabel = Label(master,text="Set the Frequency and Amplitude of the excitation signal",font="Arial 10 bold")
        self.freqLabel = Label(master,text="Frequency (Hz)")
        self.ampLabel = Label(master,text="Amplitude (Peak V)")
        # variable for the entry boxes
        self.freqVar = DoubleVar(value=0.0)
        self.ampVar = DoubleVar(value=0.0)
        # boxes for user to enter parameters
        self.freqEntry = Entry(master,textvariable=self.freqVar)
        self.ampEntry = Entry(master,textvariable=self.ampVar)
        # result set to None
        # if not edited it remains None
        self.result = None

        # arranging widgets
        self.promptLabel.grid(row=0,column=0,columnspan=2,sticky=N+S+E+W)
        self.freqLabel.grid(row=1,column=0,sticky=N+S+E+W)
        self.ampLabel.grid(row=1,column=1,sticky=N+S+E+W)
        self.freqEntry.grid(row=2,column=0,sticky=N+S+E+W)
        self.ampEntry.grid(row=2,column=1,sticky=N+S+E+W)

    # on OK the result is set as the values
    # if cancel is clicked, then the result is left as None
    def apply(self):
        self.result = (self.freqVar.get(),self.ampVar.get())

# wrapper class for generating the excitation signal
class ExcitationSignalGenerator:
    def __init__(self,freq=None,amp=None):
        # set initial device as default output
        self.device = sd.query_devices(kind="output")
        # get device id
        self.devID = sd.default.device[1]
        # if a freq and amp value was given
        if (freq is not None) and (amp is not None):
            self.__stream = sd.OutputStream(device=self.devID,channels=self.device["max_output_channels"],samplerate=self.device["default_samplerate"],
                                callback=self.__sine_callback)
        else:
             self.__stream = None
        # file logger properties
        self.fname = ''
        self.__loggerFile = None
        # signal properties
        self.__amp = 0.0
        self.__freq = 0.0
        # status of the stream
        self.status = None
        ## variables for producing the signal
        # index in the sine signal to produce the next batch of data
        self.__start_idx = 0

    # setup the logger file
    def setLoggerFile(self,file):
        # ensure file is closed
        if self.__loggerFile is not None:
            self.__loggerFile.close()
        # if the file is a path
        if type(file) == str:
            # update local copy
            self.fname = file.name
            # open file
            self.__loggerFile = open(self.fname,mode='a')
        # if the file is something with a write attribute, assume it's a file obj
        elif hasattr(file,'write'):
            # if it has a name attribute
            # use it to update local copy of filename
            if hasattr(file,'name'):
                self.fname = file.name
            self.__loggerFile = file

    # method for closing the logger file
    def closeLoggerFile(self):
        if self.__loggerFile is not None:
            self.__loggerFile.close()

    # method for opening the currently set logger file
    def openLoggerFile(self,mode='w'):
        if self.__loggerFile is not None:
            self.__loggerFile = open(self.fname,mode=mode)
        else:
            if self.fname != '':
                self.__loggerFile = open(self.fname,mode=mode)

    # method for checking if the logging file is open
    def isLogging(self):
        return not self.__loggerFile.closed

    # change frequency of the signal
    # frequency cannot be faster than the sample rate of the audio otherwise you won't get a proper signal
    # just a flat line
    def changeFreq(self,nfreq):
        if nfreq>self.device["default_samplerate"]:
            raise ValueError("Frequency cannot be higher than the sampling rate of the device!")
        else:
            self.__freq = nfreq

    # change amplitude
    def changeAmp(self,namp):
        self.__amp = namp

    # function for changing target device
    def changeDevice(self,devID):
        # update copy of device id
        self.devID = devID
        # get info on device
        self.device = sd.query_devices(self.devID)
        # generate output stream object
        self.__stream = sd.OutputStream(device=self.devID,channels=self.device["max_output_channels"],samplerate=self.device["default_samplerate"],
                                callback=self.__sine_callback)
        
    # method for checking if the stream is active
    def isOn(self):
        return self.__stream.active
    
    # method for starting the stream
    def start(self):
        if self.__stream is not None:
            self.__stream.start()

    # method for stopping the stream
    def stop(self):
        if self.__stream is not None:
            self.__stream.stop()

    # callback for the output stream
    # produces the signal
    def __sine_callback(self,outdata,frames,time,status):
        # update local copy of status
        self.status = status
        # generate new time data based off settings
        t = (self.__start_idx + np.arange(frames))/self.stream.samplerate
        # reshape time vector
        t = t.reshape(-1,1)
        # generate new output data, overrides outdata content
        outdata[:] = self.__amp * np.sin(2.0*np.pi*self.__freq*t)
        # if the file is recording save the time and channel data to the file
        # uses file handler directly and leaves numpy to handle formatting
        if (self.recFile is not None) and (not self.recFile.closed):
            np.savetxt(self.__loggerFile,np.concatenate((t,outdata),axis=1),delimiter=',')
        # update start index for next call
        self.__start_idx += frames
    

# gui for online eit solving of data
# measurement data can come from a file or from device connections
# data is passed to ReconstructionWorkers which process the data according to the internal solver and place on an output Queue
# workers run as daemon threads crunching the data, same as recorders
# as matplotlib is not thread safe, the plots are updated in the main tkinter thread
# the output Queue is read and plotted in a ReconstructionResults window
# the data written to the plot can also be recorded to a variety of file types inc HDF5 (if h5py can be imported)
class PathPatternSolverWorkerGUI:
    def __init__(self,master):
        self.master = master
        # set title
        self.master.title("EIT Solver (Worker Edition)")
        # variable for holding the number of coils
        self.numCoils = IntVar()
        ## excitation pattern variables
        # distance between coils pairs
        self.dist = IntVar(value=0)
        # distance between coils
        self.step = IntVar(value=1)
        
        ## excitation signal variables
        # manager object
        self.signalGen = ExcitationSignalGenerator()
        # frequency var
        self.exFreq = DoubleVar(value=0)
        # trace function to update the signal generator frequency everytime the variable is changed
        self.exFreq.trace('w',lambda name,idx,mode,self=self: self.signalGen.changeFreq(self.exFreq.get()))
        # amplitude variable
        self.exAmp = DoubleVar(value=0)
        self.exAmp.trace('w',lambda name,idx,mode,self=self: self.signalGen.changeAmp(self.exFreq.get()))
        
        # path of read in file
        self.datapath = ''
        
        # collection of workers
        self.dataWorkers = []
        # collection of plots in plot frame
        self.plots = []
        # collection of recorders to be managed
        self.recorders = []

        # global update flag for plots
        self.__updatePlotsGlobal = False

        # excitation order
        self.ex_mat = None

        # number of times the plots have been updated for the current data source
        # used to stop workers when we've reached the end of a file
        self.numPlotUpdates = 0

        # variable describing source type
        # 0 - File
        # 1 - Device
        self.sourceType = None

        # variable as a general indicator for stating if data is being processed or not
        self.isProcessing = BooleanVar(value=False)
        self.isRecording = BooleanVar(value=False)
        # variable showing if an input is being generated
        self.isExcited = BooleanVar(value=False)
        
        # handler for window closing
        # stops recorders to ensure results are saved
        self.master.protocol("WM_DELETE_WINDOW",self.handleExit)

        # string variables for updating the recording and processing button text
        self.procString = StringVar(value="Start Processing")
        self.recString = StringVar(value="Start Recording")
        # string variable for excitation signal button
        self.exciteString = StringVar(value="Start Excitation")

        # set size of button width
        # used for all buttons to ensure they're the same size
        buttonWidth = 50
        # frame for file controls
        self.fileControls = LabelFrame(self.master,text="File Controls",font="Arial 10 bold italic")
        # button for reading in a new file
        self.importMButton = Button(self.fileControls,text="Import Measurements",command = self.readDataFile,width=buttonWidth)
        self.importMButton.grid(row=0,column=0,sticky=N+E+W+S)
        # button for exporting the current reverse results
        self.recordingButton = Button(self.fileControls,textvariable=self.recString,command=self.handleRecording,width=buttonWidth)
        self.recordingButton.grid(row=1,column=0,sticky=N+E+W+S)
        # frame for setting up solver
        self.solverControls = LabelFrame(self.master,text="Solver Controls",font="Arial 10 bold italic")
        # button to start workers
        self.processingButton = Button(self.solverControls,textvariable=self.procString,command=self.handleProc,width=buttonWidth)
        self.processingButton.grid(row=0,column=0,sticky=N+S+E+W)
        # button to add a solver plot and setup worker
        self.addPlotButton = Button(self.solverControls,text="Add Solver",command=self.addPlot,width=buttonWidth)
        self.addPlotButton.grid(row=1,column=0,sticky=N+S+E+W)
        # button to delete a solver
        self.deletePlotButton = Button(self.solverControls,text="Delete Solver",command=self.deletePlot,width=buttonWidth)
        self.deletePlotButton.grid(row=2,column=0,sticky=N+S+E+W)
        # button to open path drawing canvas
        self.setupPathsButton = Button(self.solverControls,text="Setup Coils & Paths",command=self.setupCoilsPaths,width=buttonWidth)
        self.setupPathsButton.grid(row=3,column=0,sticky=N+S+E+W)
        # frame for device controls
        self.deviceControls = LabelFrame(self.master,text="Device Controls",font="Arial 10 bold italic")
        # button to set excitation signal
        self.setSignalButton = Button(self.deviceControls,text="Set Excitation",command=self.setSignal,width=buttonWidth)
        self.setSignalButton.grid(row=0,column=0,sticky=N+S+E+W)
        # button to set the device the excitation signal is sent from
        self.setOutDevButton = Button(self.deviceControls,text="Set Output Device",command=self.setOutputDevice,width=buttonWidth)
        self.setOutDevButton.grid(row=1,column=0,sticky=N+S+E+W)
        # button to stop excitation signal
        self.ctrlSignalButton = Button(self.deviceControls,text="Start Excitation",command=self.handleExcite,width=buttonWidth)
        self.ctrlSignalButton.grid(row=2,column=0,sticky=N+S+E+W)
        # label frame for showing the status if the signal
        self.signalStatus = LabelFrame(self.deviceControls,text="Signal Status")
        # pack signal status
        self.signalStatus.grid(row=3,column=0,sticky=N+S+E+W)
        # text status message
        self.signalStatusStr = StringVar(value="OFF")
        self.statusText = Label(self.signalStatus,textvariable=self.signalStatusStr,width=buttonWidth//2,font="Arial 8 bold")
        self.statusText.grid(row=0,column=0,sticky=N+S+E+W)
        # color status label
        self.statusCol = Label(self.signalStatus,text=" ",width=buttonWidth//2,bg="red")
        self.statusCol.grid(row=0,column=1,sticky=N+S+E+W)
        
        # supported solvers
        self.supportedSolvers = ["GREIT","JAC","BP","RADON"]
    
        # frame for plots
        # potentially going to hold plots for different workers
        self.plotFrame = Frame(self.master)

        # pack widgets
        self.solverControls.grid(row=0,column=0,sticky=N+S+E+W)
        self.fileControls.grid(row=1,column=0,sticky=N+S+E+W)
        self.deviceControls.grid(row=2,column=0,sticky=N+E+W+S)
        self.plotFrame.grid(row=0,column=1,rowspan=2,sticky=N+S+E+W)

        # make all plots resizable
        num_cols,num_rows = self.plotFrame.grid_size()
        for c in range(num_cols):
            self.plotFrame.columnconfigure(c,weight=1)
        for r in range(num_rows):
            self.plotFrame.rowconfigure(r,weight=1)

        ## make the controls resizable
        num_cols,num_rows = self.solverControls.grid_size()
        for c in range(num_cols):
            self.solverControls.columnconfigure(c,weight=1)
        for r in range(num_rows):
            self.solverControls.rowconfigure(r,weight=1)

        num_cols,num_rows = self.fileControls.grid_size()
        for c in range(num_cols):
            self.fileControls.columnconfigure(c,weight=1)
        for r in range(num_rows):
            self.fileControls.rowconfigure(r,weight=1)
        
        # make the main window contents resizable 
        num_cols,num_rows = master.grid_size()
        for c in range(num_cols):
            master.columnconfigure(c,weight=1)
        for r in range(num_rows):
            master.rowconfigure(r,weight=1)

        # set minimum size to size on creation
        self.master.update_idletasks()
        self.master.minsize(self.master.winfo_width(),self.master.winfo_height())

    # creates a dialog allowing the user to select a target output device
    def setOutputDevice(self):
        # create dialog
        devID = SoundDevicesDialog(self.master).result
        # if the user selected something
        if devID is not None:
            # update target device
            self.signalGen.changeDevice(devID)

    # method for asking the user for the properties of the excitation signal
    def setSignal(self):
        # produce dialog
        propRes = SetExcitePropertiesDialog(self.master)
        # if the user clicked OK
        if propRes is not None:
            # separate the variables
            freq,amp = propRes.result
            # update values
            self.exFreq.set(freq)
            self.exAmp.set(amp)

    # method for stopping the excitation signal
    def stopSignal(self):
        self.signalGen.stop()
        self.statusCol.config(bg="red")

    # method for starting the excitation signal
    def startSignal(self):
        self.signalGen.start()
        self.statusCol.config(bg="green")

    # method for handling clicking the Start/Stop excitation button
    def handleExcite(self):
        if self.isExcited.get():
            self.stopSignal()
            self.isExcited.set(False)
            self.exciteString.set("Start Excitation")
        else:
            self.startSignal()
            self.isExcited.set(True)
            self.exciteString.set("Stop Excitation")

    # function for calling update method of each plot
    def updateManager(self):
        # update plots
        for p in self.plots:
            p.update()
        # if the update flag is set
        if self.__updatePlotsGlobal:
            # call this function again after a certain period of time
            self.master.after(0,self.updateManager)

    # recording button handler
    # function called based on self.isRecording flag
    def handleRecording(self):
        # if flag is false
        # call startRecording
        if not self.isRecording.get():
            flag = self.startRecording()
        else:
            flag = self.stopRecording()

        # toggle flag
        self.isRecording.set(not self.isRecording.get())
        # update text of button
        if self.isRecording.get():
            self.recString.set("Stop Recording")
        else:
            self.recString.set("Start Recording")

    # processing button handler
    # function called based on self.isProcessing flag
    def handleProc(self):
        # if flag is false
        # start workers
        if not self.isProcessing.get():
            print(f"started {len(self.dataWorkers)} workers")
            flag = self.startWorkers()
        # if flag is true
        # stop workers
        else:
            print(f"stopping {len(self.dataWorkers)} workers")
            flag = self.stopWorkers()

        # toggle flag
        self.isProcessing.set(not self.isProcessing.get())
        # check new value
        # update button text based on value
        if self.isProcessing.get():
            self.procString.set("Stop Processing")
        else:
            self.procString.set("Start Processing")
            
    # handler for window closing
    def handleExit(self):
        # stop updating plots
        self.__updatePlotsGlobal = False
        # ensure recorders have been stopped
        for r in self.recorders:
            r.stop()
        # stop workers
        for w in self.dataWorkers:
            w.stopRecon()
        # destroy root
        self.master.destroy()
        
    # method to create dialog that allows user to setup the coil arrangement and excitation order
    def setupCoilsPaths(self):
        # create dialog
        cpathsDialog = CoilsPathsDialog(self.master,"Setup the Excitation Order")
        # if something was setup
        if cpathsDialog.result is not None:
            # parse result and update variables
            nc,dist,step,ex_mat = cpathsDialog.result
            # number of coils
            self.numCoils.set(nc)
            # distance between coils in pair
            self.dist.set(nc)
            # distance between coil pairs
            self.step.set(step)
            # excitation order
            self.ex_mat = np.array(ex_mat)

    # method for setting up and starting recording of reconstruction results
    def startRecording(self):
        # stop any current recorders
        self.stopRecording()
        # get number of rows and columns in the plot frame
        num_cols,num_rows = self.plotFrame.grid_size()
        # if there are plots
        if (num_cols==0) and (num_rows==0):
            messagebox.showerror("No plots to record",message="No Plots to record!")
            return False
        else:
            # find which plots have been created
            currentSolvers = [p.solverType for p in self.plots]
            # add ALL option onto the list
            currentSolvers.append("ALL")
            # open a combobox dialog
            chosenplot = StringVar()
            # ask user which plot they want to record
            ComboboxDialog(self.master,currentSolvers,chosenplot,"Choose which plot to record","Current Plots")
            if chosenplot.get() is '':
                return False
            else:
                # if all option was chosen, save the data to the same file
                # only file type that can support all the files at once is hdf5
                # if there's more than one plot and there's no hdf5 support, inform user that it cannot be recorded
                if chosenplot.get() == "ALL":
                    # if hdf5 not supported
                    if not hdf5Support:
                        # iterate over plots and names 
                        for name,p in zip(currentSolvers[:-1],self.plots):
                            # if there's only one plot to record
                            # ask user for filepath
                            fname = filedialog.asksaveasfilename(title="Set Recording Location",initialdir=os.path.dirname(self.datapath),defaultextension='.txt',filetypes=[("Text Data File (*.txt)",'*.txt'),("Log Data File (*.csv)","*.csv"),])
                            # if a filepath was chosen
                            if fname is '':
                                return False
                            else:
                                # create recorder and add to list
                                # recorders are managed from main window
                                self.recorders.append(BasicRecorder(fname))
                                # assign recorder to plot
                                p.recorder = self.recorders[-1]
                                # start recorder
                                p.recorder.start()
                        return True
                    # if hdf5 is supported
                    # ask for filename with hdf5 option shown
                    else:
                        fname = filedialog.asksaveasfilename(title="Set Recording Location",initialdir=os.path.dirname(self.datapath),defaultextension='.txt',filetypes=[("Text Data File (*.txt)",'*.txt'),("Log Data File (*.csv)","*.csv"),("HDF5 File (*.hdf5)","*.hdf5"),])
                        # if a filename was chosen
                        if fname is '':
                            return False
                        else:
                            # create a hdf5 recorder that will manage each plot
                            self.recorders.append(HDF5MultiRecorder(fname,currentSolvers[:-1]))
                            # attach recorder to each of the plots
                            # start recording
                            for p in self.plots:
                                p.recorder = self.recorders[-1]
                                p.recorder.start()
                            return True
                # if only a specific plot was targeted
                else:
                    # hdf5 is not supported
                    if not hdf5Support:
                        # ask user for filepath
                        fname = filedialog.asksaveasfilename(title="Set Recording Location",initialdir=os.path.dirname(self.datapath),defaultextension='.txt',filetypes=[("Text Data File (*.txt)",'*.txt'),("Log Data File (*.csv)","*.csv"),])
                        # if a filepath was chosen
                        if fname is '':
                            return False
                        else:
                            self.recorders.append(BasicRecorder(fname))
                            # find plot, assign recorder and start it
                            for p in self.plots:
                                if p.solverType == chosenplot.get():
                                    p.recorder = self.recorders[-1]
                                    p.recorder.start()
                                    return True
                    # if hdf5 is supported
                    else:
                        # ask user for filepath including hdf5 extension as an option
                        fname = filedialog.asksaveasfilename(title="Set Recording Location",initialdir=os.path.dirname(self.datapath),defaultextension='.txt',filetypes=[("Text Data File (*.txt)",'*.txt'),("Log Data File (*.csv)","*.csv"),("HDF5 File (*.hdf5)","*.hdf5"),])
                        # if a filename was chosen
                        if fname is '':
                            return False
                        else:
                            # create recorder
                            self.recorders.append(HDF5Recorder(fname,chosenplot.get()))
                            # find plot and assign recorder
                            for p in self.plots:
                                if p.solverType == chosenplot.get():
                                    p.recorder = self.recorders[-1]
                                    # recorder is started
                                    p.recorder.start()
                                    return True

    # method for stopping all current recorders
    # iterates over recorders stopping them and destroying them
    def stopRecording(self):
        for rec in self.recorders:
            # call stop button of recorder
            # also closed the file
            rec.stop()
            # removed recorder from gui
            self.recorders.remove(rec)

    # method to start workers
    # starts plots as well
    def startWorkers(self):
        # if there are no workers inform user
        if len(self.dataWorkers)==0:
            messagebox.showinfo("No plots","There are no plots or workers to start")
            return False
        # check if the threads or plots are not running
        # if not, start loop
        for w,p in zip(self.dataWorkers,self.plots):
            if not p.isUpdating():
                p.startUpdate()
            w.startRecon()
        # set global update loop for plots to true
        self.__updatePlotsGlobal = True
        # start update loop
        self.updateManager()
        # return true to indicate that this was completed
        return True

    # method to stop all workers and plots
    def stopWorkers(self):
        for w,p in zip(self.dataWorkers,self.plots):
            p.stopUpdate()
            w.stopRecon()
        # set flag to false
        # this will stop the update of each plot
        self.__updatePlotsGlobal = False
        return True

    # iterate over workers calling pause method
    # whilst paused, workers do not process new data in the input queue
    def pauseWorkers(self):
        for w in self.dataWorkers:
            w.pause()

    # iterate over workers calling resume method
    def resumeWorkers(self):
        for w in self.dataWorkers:
            w.resume()
            
    # function for reading in data files
    def readDataFile(self):
        # ask user to select and open a file
        fname = filedialog.askopenfilename(initialfile=self.datapath,filetypes=(["Text Data File",'*.txt'],["Log Data File","*.csv"]))
        # if user selected a file
        if fname is '':
            return
        else:
            # update local copy of filename
            self.datapath = fname
            # read in data
            with open(self.datapath,'r') as dataFile:
                # read in all the lines in a file
                lines = dataFile.readlines()
            # if it's a text file, attempt to convert each line to a readable form
            # then convert result to a numpy dataset
            if os.path.splitext(os.path.basename(self.datapath))[1] == '.txt':
                # attempt to convert data to numpy array
                # if it fails, an error messagebox is displayed asking if the user wants to try again
                try:
                    # parse the lines and convert into a dataset
                    # if it fails to parse the data then the exception is thrown
                    # if it fails the current stored data is not affected
                    temp = np.array([self.parseLine(l) for l in lines],dtype='float')
                    self.data = temp
                except ValueError as e:
                    messagebox.showerror(title="Failed to parse text data file!",message=f"Failed to parse target data file\n{fname}\n\n{e}")
                    return
            # if it's a csv file, read in file and parse with comma as delimiter
            elif os.path.splitext(os.path.basename(self.datapath))[1] == '.csv':
                try:
                    temp = np.genfromtxt(self.datapath,dtype='float',delimiter=',')
                    self.data = temp
                except ValueError as e:
                    messagebox.showerror(title="Failed to parse csv data file!",message=f"Failed to parse target data file\n{fname}\n\n{e}")
                    return
            # if importing from a hdf5 file
            elif os.path.splitext(os.path.basename(self.datapath))[1] == '.hdf5':
                # and there's no hdf5 support
                # inform user and exit
                if not hdf5Support:
                    messagebox.showerror(title="Cannot read in file",message="Cannot read in HDF5 files!\nh5py not installed")
                    return
                else:
                    # open file
                    with h5py.File(self.datapath,'r') as file:
                        # get root level datasets
                        dsets = [k for k in file.keys() if isinstance(file[k],h5py.Dataset)]
                        # if there are no root level datasets
                        # inform user and exit
                        if len(dsets)==0:
                            messagebox.showerror("No datasets!","No root level datasets available in\n{self.datapath}\nCannot in investigate groups at the moment!")
                            return
                        # if there's only one dataset
                        # load in that that dataset
                        elif len(dsets)==1:
                            self.data = file[dsets[0]][()]
                        # if there's more than one dataset
                        # present options to user 
                        elif len(dsets)>1:
                            chosenDSet = StringVar()
                            ComboboxDialog(self.master,dsets,chosenDSet,"Choose dataset to import","Available datasets")
                            # if the user chose an option
                            if chosenDSet.get() is not '':
                                self.data = file[chosenDSet.get()]

            # set source type to indicate that it's a file
            self.sourceType = 0
            # clear update index
            self.numPlotUpdates = 0

            # populate input buffers of the workers
            for w in self.dataWorkers:
                for r in range(self.data.shape[0]):
                    w.inputQ.put(self.data[r,:])
        
    # method for parsing a line of data from an appropriately formatted data text file
    @staticmethod
    def parseLine(line):
        # attempt to split the data from the semicolon
        # removes the magnitude line header
        try:
            _, data = line.split(":", 1)
        except ValueError:
            return None
        # split the comma separated values removing any special characters
        # convert them to floats
        # return as a numpy array
        items = []
        for item in data.split(","):
            # remove new lines
            item = item.strip()
            # if no item after stripping
            # start next loop
            if not item:
                continue
            # attempt to convert value to float
            try:
                items.append(float(item))
            except ValueError:
                return None
        # convert list of items into a numpy array and return
        return np.array(items)
                    
    # methods for deleting a specific plot
    # stops the worker 
    def deletePlot(self,rtype=None):
        # pause data workers
        self.pauseWorkers()
        # if the plots are currently being updated
        if self.__updatePlotsGlobal:
            # stop updating the plots
            self.__updatePlotsGlobal = False
            # set local flag to indicate that some plots have been stopped
            plotsStopped = True
        else:
            plotsStopped = False
            
        # stop the workers so the dialogs can be handled
        # get the number of rows and cols in the grid of plot_frame
        # tells us the number of plots added
        num_cols,num_rows = self.plotFrame.grid_size()
        # if there are no plots
        if (num_cols==0) and (num_rows==0):
            messagebox.showinfo("No plots to delete",message="No Plots to delete!")
        else:
            # find which solvers are currently being plotted
            currentSolvers = [w.solverType for w in self.plots]
            # open a combobox dialog
            chosenplot = StringVar()
            if rtype == None:
                ComboboxDialog(self.master,currentSolvers,chosenplot,"Choose which plot type to delete","Current Plots")
            else:
                # if the user specified a plot that's not present
                # rause exception and exit
                if rtype not in currentSolvers:
                    raise ValueError(f"No plot corresponding to type {rtype}!")

                # if it's a valid type, update local StringVar
                chosenplot.set(rtype)
            # if a target plot was not chosen
            if chosenplot.get() != '':
                ## search for location of plot
                # iterate of widgets inside plotFrame
                for child in self.plotFrame.grid_slaves():
                    print(child)
                    if isinstance(child,Label):
                        if child.cget("text") == chosenplot.get():
                            print(child.cget("text"))
                            child.destroy()
                    elif isinstance(child,ReconstructionResultsWindow):
                        if child.solverType == chosenplot.get():
                            print(child.solverType)
                            child.destroy()
                # delete worker from list
                for w in self.dataWorkers:
                    if w.getSolverType() == chosenplot.get():
                        # ensure worker is stopped
                        w.stopRecon()
                        self.dataWorkers.remove(w)
                        break
                # delete plot from list
                for p in self.plots:
                    if p.solverType == chosenplot.get():
                        self.plots.remove(p)
                        break
                # force redraw of main window to show that the plot has been removed
                self.master.update()
        # resume workers
        self.resumeWorkers()
        # if the plots were stopped previously
        # and there are still plots now
        # restart update of plots
        if plotsStopped and (len(self.plots)>0):
            # reset plot update flag
            self.__updatePlotsGlobal = True
            # restart updates
            self.updateManager()

    # method for adding a plot for a worker to the gui
    # can specify plot and location for automated use
    # if a plot type or location is not specified, then PlotLocationDialog is opened to prompt user
    def addPlot(self,rtype=None,r=None,c=None):
        # pause data workers
        self.pauseWorkers()
        # if the plots are currently being updated
        if self.__updatePlotsGlobal:
            # stop updating the plots
            self.__updatePlotsGlobal = False
            # set local flag to indicate that some plots have been stopped
            plotsStopped = True
        else:
            plotsStopped = False
            
        # if the number of coils has not been set
        if self.numCoils.get() == 0:
            messagebox.showerror("No coils!","Coil setup not defined!\nNumber of coils and excitation order need to be set in order to initialize solvers")
            return
        # if a type wasn't specified in the constructor
        if (rtype is None) or (r is None) or (c is None):
            # prompt user to select the type of solver they want and where to put it in the plot frame
            plotdialog = PlotLocationDialog(self.master,opts=self.supportedSolvers,title="Choose which solver to use and where to put it",rowinit=0,colinit=0)
            # get tesult
            res = plotdialog.result
        else:
            if (r<0) or (c<0):
                raise ValueError("Target row and column cannot be zero")
            # form target into tuple to be parsed
            res = (rtype,r,c)
        # check that a result was returned 
        if res is None:
            return
        # if an option was selected, process it
        else:
            rtype,r,c = res
            # create worker to process data 
            self.dataWorkers.append(ReconstructionWorkerEIT(self.numCoils.get(),self.dist.get(),rtype=rtype,ex_mat=self.ex_mat))
            # create plot and initialize with worker
            # worker is not started here
            newPlot = ReconstructionResultsWindow(self.plotFrame,self.dataWorkers[-1])
            # add to list
            self.plots.append(newPlot)
            newPlot.grid(row=r,column=c)
            # ensure plots are resizable
            newPlot.rowconfigure(r,weight=1)
            newPlot.columnconfigure(c,weight=1)

        # resume workers
        self.resumeWorkers()
        # if the plots were stopped previously
        # and there are still plots now
        # restart update of plots
        if plotsStopped and (len(self.plots)>0):
            # reset plot update flag
            self.__updatePlotsGlobal = True
            # restart updates
            self.updateManager()
        
if __name__ == "__main__":
    root = Tk()
    view = PathPatternSolverWorkerGUI(root)
    root.mainloop()
