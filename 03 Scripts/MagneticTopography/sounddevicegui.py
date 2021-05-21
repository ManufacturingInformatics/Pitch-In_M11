from tkinter import Tk,filedialog,Button,Toplevel,Frame,Scrollbar,TOP,BOTTOM,BOTH,RAISED,SUNKEN,Menu,HORIZONTAL,N,S,E,W,CENTER,Label,Entry,StringVar,DoubleVar,IntVar,BooleanVar
from tkinter.ttk import Treeview,Separator
from tkinter.simpledialog import Dialog
import numpy as np
import sounddevice as sd
import os

""" Sine Wave Generator for Audio

    David Miller, 2019
    University of Sheffield

    GUI for generating and sending a sine wave to a target audio device
"""

class AboutWindow:
    """ Window displaying information about the author and the GUI"""
    def __init__(self,master):
        self.master = master
        self.master.title("About")

        # frame to hold all information
        self.info_frame = Frame(master,relief='sunken')

        ## information to displa
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
                    self.depend_win.labels.append(Label(self.depend_win,text="{} {}".format(val.name__,val.version__)))
                # if not, use pkg_resources to get version
                else:
                    try:
                        self.depend_win.labels.append(Label(self.depend_win,text="{} {}".format(val.name__,pkg_resources.get_distribution(val.name__).version)))
                    except pkg_resources.DistributionNotFound:
                        self.depend_win.labels.append(Label(self.depend_win,text="{} {}".format(val.name__,"N/A")))
                # set row of label
                self.depend_win.labels[-1].grid(row=r,column=0)
                # increase row number
                r+=1

class SoundDevicesGUI:
    """ Window for choosing and selecting a new target audio device. Links with SoundGUI"""
    def __init__(self,selectedDevRef,master):
        self.master = master
        # save reference to IntVar on master GUI for target device ID
        self.devIDRef = selectedDevRef
        # variable for holding selected device branch
        self.selectedBranch = None
        # set title for window
        self.master.title("Sound Devices")
        # add frame for hosting treeview and scrollbar
        self.frame = Frame(self.master)
        # add title
        self.frame_title = Label(self.frame,text="Found Output Devices",font="Arial 10 bold")
        # get devices
        devs = sd.query_devices(kind="output")
        
        # string that is updated whenever the device ID is updated
        self.updateStr = StringVar(value="                     ")
        self.updateText = Label(self.master,textvariable=self.updateStr,font="Arial 10 bold")
        # button for checking the selected device branch and updating the target device ID if necessary
        self.updateButton = Button(self.master,text="Update ID",command=self.updateID)
        
        # add treeview
        # columns based off keys used to describe devices
        self.dev_tree = Treeview(self.frame,columns=("Property","Value"),show="tree")
        # add scrollbar
        self.tree_scroll = Scrollbar(self.frame,orient='vertical',command=self.dev_tree.yview)
        self.dev_tree.configure(yscrollcommand=self.tree_scroll)
        # add select handlder
        self.dev_tree.bind("<<TreeviewSelect>>",self.selectDevice)

        ## setup the size and headings of the entries
        self.dev_tree['show']='headings'
        # set the heading to be the key with the underscores replaced by spaces to make it more readable
        self.dev_tree.heading("Property",text="Property")
        # set the width and placement
        self.dev_tree.column("Property",width=200,anchor=CENTER)
        # set the heading to be the key with the underscores replaced by spaces to make it more readable
        self.dev_tree.heading("Value",text="Value")
        # set the width and placement
        self.dev_tree.column("Value",width=200,anchor=CENTER)

        ## add the output devices to the string
        # add super branch for easy filtering and viewing
        brid = self.dev_tree.insert('','end',text="Outputs")
        
        # if the list of output devices found is an instance of DeviceList class then there are multiple
        # devices to be added
        if isinstance(devs,sd.DeviceList):
            # add each device to the tree
            for di,dd in enumerate(devs):
                self.addDeviceToTree(self.dev_tree,dd,di,brid)
        # if it's a dictionary, then it's a single device to be added
        # add to tree using findAudioDeviceID method to search for it's index/deviceID
        elif isinstance(devs,dict):
            self.addDeviceToTree(self.dev_tree,devs,self.findAudioDeviceID(devs),brid)

        # add all other devices as the default output device doesn't always work
        brid = self.dev_tree.insert('','end',text="All")
        # get all devices to add
        devs = sd.query_devices()
        # if there is more than one in the form of a deviceList
        # iterate over each of them adding them to the tree under the All branch
        if isinstance(devs,sd.DeviceList):
            for di,dd in enumerate(devs):
                self.addDeviceToTree(self.dev_tree,dd,di,brid)
        # if the list of found devices is in the form of a dict, then it is a single device
        # add to tree using findAudioDeviceID method to search for it's index/deviceID
        elif isinstance(devs,dict):
            self.addDeviceToTree(self.dev_tree,devs,self.findAudioDeviceID(devs),brid)

        # expand tree to show new entries
        self.dev_tree['show']="tree headings"

        ## arrange entries
        # update button and text, outside of frame
        self.updateButton.grid(row=0,column=0,sticky=E+W)
        self.updateText.grid(row=0,column=1,sticky=E+W)
        
        ## frame holding treeview and title
        # frame title, inside self.frame
        self.frame_title.grid(row=0,column=0,sticky=E+W)
        # treeview, inside self.frame
        self.dev_tree.grid(row=1,column=0,sticky=N+S+E+W)
        self.tree_scroll.grid(row=1,column=1,sticky=N+S)
        # frame
        self.frame.grid(row=1,column=0,columnspan=2,sticky=N+S+E+W)

        # add weights so they can be resized
        num_cols,num_rows = master.grid_size()
        for c in range(num_cols):
            master.columnconfigure(c,weight=1)
        for r in range(num_rows):
            master.rowconfigure(r,weight=1)
            
    # get the branch selected by the user, single click highlighting
    # update internal copy of branch ID to be used later
    def selectDevice(self,event):
        self.selectedBranch = event.widget.selection()

    def updateID(self):
        # if a branch has been selected
        if self.selectedBranch is not None:
            ## check what kind of branch was selected
            # if the user has not selected the top level branches
            if self.dev_tree.item(self.selectedBranch)['text'] not in ("Outputs", "All"):
                # if the branch has a text title, then it is the name of the device
                if self.dev_tree.item(self.selectedBranch)['text'] is not '':
                    # get idx for properties child branch
                    for prop in self.dev_tree.get_children(self.selectedBranch):
                        # iterate over children to search for hostapi id
                        for child in self.dev_tree.get_children(prop):
                            if 'Device ID' in self.dev_tree.item(child)['values']:
                                # update target device id on master GUI using passed reference
                                self.devIDRef.set(int(self.dev_tree.item(child)['values'][1]))
                                # set text to show user that the value has been updated
                                self.updateStr.set("Target device updated!")
                                # break from forloop
                                break

    # metbod for searching the sound devices for the device id
    @staticmethod
    def findAudioDeviceID(devDict):
        for di,dev in enumerate(sd.query_devices()):
            if dev == devDict:
                return di

    # method for parsing sounddevice dictionaries on devices and adding them to a Treeview
    @staticmethod
    def addDeviceToTree(tree,devDict,devID,branch=''):
        """ Add to the specified treeview or similar object the dictioanry devDict under the specified branch

            tree : Tkinter.tk Treeview reference. This is can be the entire treeview or a specific branch instance inside of it
            devDict : Dictionary object assumed to be formatted as in sounddevice documentation. Does not check if the keys match the documentation
            devID: Device ID. The index of it's position in the DeviceList
            branch : ID or string to the branch under tree that it will be added to. The ID or string must be valid under the object tree given

            This function does not return anything

            Adds the information stored inside devDict under the specified branch inside tree. Designed to work with a created instance of Tkinter.tk
            Treeview whose columns have been formatted as Property and Value respectively. It does not set the text attribute of the entry only the values.

            Under the specified branch a name branch is created where the text attribute is set to the value under the key 'name' in the dictionary. Under this
            name branch a Properties sub-branch is created to store the information about the device's properties. It is arranged this way so that the user can
            easily inspect what devices are there without investigating the properties of each device

            Below is the structure of what is added under the branch

            E.g.

            tree
            |
            |- branch
                |
                |- name
                     |
                     |- Properties
                             |
                             |- Property #1 : Value of Property #1
                             |- Property #2 : Value of Property #2
                             |- Property #3 : Value of Property #3
                             |- Device ID : Index of device in DeviceList returned by query_devices()
        """
        # iterate over the keys and values in the dictioanry describing the device
        for k,v in devDict.items():
            # if it's the name of the device
            if k is "name":
                # create a branch to represent the device
                # text is the name of the device
                # iow each gets a device based off their name
                name_brid=tree.insert(branch,'end',text=v)
                # create a branch under it to store the properties
                brid = tree.insert(name_brid,'end',text="Properties")
                # move onto next item
                continue
            else:
                # add property under Properties branch created earlier
                # the name of the property and its respective value are added
                tree.insert(brid,'end',text="",values=(k,v))
        tree.insert(brid,'end',text="",values=("Device ID",devID))

# dialog version of the SoundDevicesGUI
# used to select a target device
# updates result component of class with the device id and information dictionary
class SoundDevicesDialog(Dialog):
    def __init__(self,parent,title="Select target sound device"):
        self.result = None
        super().__init__(parent,title=title)

    def body(self,master):
        # variable for holding selected device branch
        self.selectedBranch = None
        # add frame for hosting treeview and scrollbar
        self.frame = Frame(master)
        # add title
        self.frame_title = Label(self.frame,text="Found Output Devices",font="Arial 10 bold")
        # get devices
        devs = sd.query_devices(kind="output")
        
        # add treeview
        # columns based off keys used to describe devices
        self.dev_tree = Treeview(self.frame,columns=("Property","Value"),show="tree")
        # add scrollbar
        self.tree_scroll = Scrollbar(self.frame,orient='vertical',command=self.dev_tree.yview)
        self.dev_tree.configure(yscrollcommand=self.tree_scroll)
        # add select handlder
        self.dev_tree.bind("<<TreeviewSelect>>",self.selectDevice)

        ## setup the size and headings of the entries
        self.dev_tree['show']='headings'
        # set the heading to be the key with the underscores replaced by spaces to make it more readable
        self.dev_tree.heading("Property",text="Property")
        # set the width and placement
        self.dev_tree.column("Property",width=200,anchor=CENTER)
        # set the heading to be the key with the underscores replaced by spaces to make it more readable
        self.dev_tree.heading("Value",text="Value")
        # set the width and placement
        self.dev_tree.column("Value",width=200,anchor=CENTER)

        ## add the output devices to the string
        # add super branch for easy filtering and viewing
        brid = self.dev_tree.insert('','end',text="Outputs")
        
        # if the list of output devices found is an instance of DeviceList class then there are multiple
        # devices to be added
        if isinstance(devs,sd.DeviceList):
            # add each device to the tree
            for di,dd in enumerate(devs):
                self.addDeviceToTree(self.dev_tree,dd,di,brid)
        # if it's a dictionary, then it's a single device to be added
        # add to tree using findAudioDeviceID method to search for it's index/deviceID
        elif isinstance(devs,dict):
            self.addDeviceToTree(self.dev_tree,devs,self.findAudioDeviceID(devs),brid)

        # add all other devices as the default output device doesn't always work
        brid = self.dev_tree.insert('','end',text="All")
        # get all devices to add
        devs = sd.query_devices()
        # if there is more than one in the form of a deviceList
        # iterate over each of them adding them to the tree under the All branch
        if isinstance(devs,sd.DeviceList):
            for di,dd in enumerate(devs):
                self.addDeviceToTree(self.dev_tree,dd,di,brid)
        # if the list of found devices is in the form of a dict, then it is a single device
        # add to tree using findAudioDeviceID method to search for it's index/deviceID
        elif isinstance(devs,dict):
            self.addDeviceToTree(self.dev_tree,devs,self.findAudioDeviceID(devs),brid)

        # expand tree to show new entries
        self.dev_tree['show']="tree headings"

        ## arrange entries
        ## frame holding treeview and title
        # frame title, inside self.frame
        self.frame_title.grid(row=0,column=0,sticky=E+W)
        # treeview, inside self.frame
        self.dev_tree.grid(row=1,column=0,sticky=N+S+E+W)
        self.tree_scroll.grid(row=1,column=1,sticky=N+S)
        # frame
        self.frame.grid(row=1,column=0,columnspan=2,sticky=N+S+E+W)

        # add weights so they can be resized
        num_cols,num_rows = master.grid_size()
        for c in range(num_cols):
            master.columnconfigure(c,weight=1)
        for r in range(num_rows):
            master.rowconfigure(r,weight=1)

    # get the branch selected by the user, single click highlighting
    # update internal copy of branch ID to be used later
    def selectDevice(self,event):
        self.selectedBranch = event.widget.selection()

    # apply method of dialog
    # gets the selected device and information associated with it and updates the result parameter of the widget
    def apply(self):
        # if a branch has been selected
        if self.selectedBranch is not None:
            ## check what kind of branch was selected
            # if the user has not selected the top level branches
            if self.dev_tree.item(self.selectedBranch)['text'] not in ("Outputs", "All"):
                # if the branch has a text title, then it is the name of the device
                if self.dev_tree.item(self.selectedBranch)['text'] is not '':
                    # get idx for properties child branch
                    for prop in self.dev_tree.get_children(self.selectedBranch):
                        # iterate over children to search for hostapi id
                        for child in self.dev_tree.get_children(prop):
                            if 'Device ID' in self.dev_tree.item(child)['values']:
                                self.result = int(self.dev_tree.item(child)['values'][1])
                                return

    # metbod for searching the sound devices for the device id
    @staticmethod
    def findAudioDeviceID(devDict):
        for di,dev in enumerate(sd.query_devices()):
            if dev == devDict:
                return di

    # method for parsing sounddevice dictionaries on devices and adding them to a Treeview
    @staticmethod
    def addDeviceToTree(tree,devDict,devID,branch=''):
        """ Add to the specified treeview or similar object the dictioanry devDict under the specified branch

            tree : Tkinter.tk Treeview reference. This is can be the entire treeview or a specific branch instance inside of it
            devDict : Dictionary object assumed to be formatted as in sounddevice documentation. Does not check if the keys match the documentation
            devID: Device ID. The index of it's position in the DeviceList
            branch : ID or string to the branch under tree that it will be added to. The ID or string must be valid under the object tree given

            This function does not return anything

            Adds the information stored inside devDict under the specified branch inside tree. Designed to work with a created instance of Tkinter.tk
            Treeview whose columns have been formatted as Property and Value respectively. It does not set the text attribute of the entry only the values.

            Under the specified branch a name branch is created where the text attribute is set to the value under the key 'name' in the dictionary. Under this
            name branch a Properties sub-branch is created to store the information about the device's properties. It is arranged this way so that the user can
            easily inspect what devices are there without investigating the properties of each device

            Below is the structure of what is added under the branch

            E.g.

            tree
            |
            |- branch
                |
                |- name
                     |
                     |- Properties
                             |
                             |- Property #1 : Value of Property #1
                             |- Property #2 : Value of Property #2
                             |- Property #3 : Value of Property #3
                             |- Device ID : Index of device in DeviceList returned by query_devices()
        """
        # iterate over the keys and values in the dictioanry describing the device
        for k,v in devDict.items():
            # if it's the name of the device
            if k is "name":
                # create a branch to represent the device
                # text is the name of the device
                # iow each gets a device based off their name
                name_brid=tree.insert(branch,'end',text=v)
                # create a branch under it to store the properties
                brid = tree.insert(name_brid,'end',text="Properties")
                # move onto next item
                continue
            else:
                # add property under Properties branch created earlier
                # the name of the property and its respective value are added
                tree.insert(brid,'end',text="",values=(k,v))
        tree.insert(brid,'end',text="",values=("Device ID",devID))

class SoundGUI:
    """ Main window for defining, producing and sending an audio sine wave to a target audio device.
        Links with SoundDevicesGUI and AboutWindow
    """
    def __init__(self,master):
        self.master = master
        # set the title of the GUI
        self.master.title("Sound Device GUI")

        # variables
        self.freq = DoubleVar(value=500.0)
        self.freq_val = self.freq.get()
        self.amp = DoubleVar(value=1.0)
        self.amp_val = self.amp.get()
        # set the default device ID as the current default output device ID
        self.devID = IntVar(value=sd.default.device[1])
        # set the samplerate and number of channels based off the found output devices
        # if query returns a DeviceList use the first device in the list
        # if the query returns a dictionary, i.e. a single device
        devs = sd.query_devices(kind='output')
        if isinstance(devs,sd.DeviceList):
            self.samplerate = devs[0]['default_samplerate']
            self.channels = devs[0]['max_output_channels']
        elif isinstance(devs,dict):
            self.samplerate = devs['default_samplerate']
            self.channels = devs['max_output_channels']
        # stream status
        self.status_var = BooleanVar(value=False)
        self.status_label = Label(self.master,text="",bg="red")
        # update the color of the status display whenever the variable is changed
        self.status_var.trace("w",self.updateStrStatus)
        
        # index for where in the signal waveform to generate data from
        self.start_idx = 0

        # record controls
        self.record_start = Button(self.master,text="REC START",command=self.startRec,font="Arial 10 bold")
        self.record_stop = Button(self.master,text="REC STOP",command=self.stopRec,font="Arial 10 bold")
        self.record_stop.config(state="disabled")
        # file handler
        self.recFile = None
        # file path
        from datetime import datetime
        # initialize file path to current datetime 
        self.recFilePath = "audrecord-{}.csv".format(datetime.now().strftime("%Y-%m-%d"))
        # check if the file exists
        if os.path.exists(self.recFilePath):
            i=1
            orig = os.path.splitext(self.recFilePath)[0]
            # add numbers onto the end to generate a new file name
            while os.path.exists("{}-({}).csv".format(orig,i)):
                i+=1
            # update filename
            self.recFilePath = "{}-({}).csv".format(orig,i)
            
        ## controls
        # button for starting and stopping the audio stream
        self.start_button = Button(self.master,text="Start",font="Arial 10 bold",command=self.startStream)
        self.stop_button = Button(self.master,text="Stop",font="Arial 10 bold",command=self.stopStream)
        # collection of controls
        self.controls = Frame(self.master)
        # controls for specific variables inside super frame
        self.freqControls = Frame(self.controls)
        self.ampControls = Frame(self.controls)
        self.devControls = Frame(self.controls)

        # frequency controls
        self.freqTitle = Label(self.freqControls,text="Frequency",font="Arial 10 bold")
        self.freqEntry = Entry(self.freqControls,textvariable=self.freq)
        self.freqUpdate = Button(self.freqControls,text="Update",command=self.updateFreq)
        # layout
        self.freqTitle.grid(row=0,column=0,sticky=N+S+E+W)
        self.freqEntry.grid(row=1,column=0,sticky=N+S+E+W)
        self.freqUpdate.grid(row=2,column=0,sticky=N+S+E+W)
        
        # amplitude controls
        self.ampTitle = Label(self.ampControls,text="Amplitude",font="Arial 10 bold")
        self.ampEntry = Entry(self.ampControls,textvariable=self.amp)
        self.ampUpdate = Button(self.ampControls,text="Update",command=self.updateAmp)
        # layout
        self.ampTitle.grid(row=0,column=0,sticky=N+S+E+W)
        self.ampEntry.grid(row=1,column=0,sticky=N+S+E+W)
        self.ampUpdate.grid(row=2,column=0,sticky=N+S+E+W)

        # device ID controls
        self.devTitle = Label(self.devControls,text="Device ID",font="Arial 10 bold")
        self.devEntry = Entry(self.devControls,textvariable=self.devID)
        self.devSelectButton = Button(self.devControls,text="Select...",command=self.selectDevice)
        # layout
        self.devTitle.grid(row=0,column=0,columnspan=2)
        self.devEntry.grid(row=1,column=0)
        self.devSelectButton.grid(row=1,column=1)

        ## create menu bar
        self.menuBar = Menu(self.master)
        self.master.config(menu=self.menuBar)

        # file menu
        self.fileMenu = Menu(self.menuBar,tearoff=0)
        self.fileMenu.add_command(label="Set filename",command=self.setFileName)
        self.menuBar.add_cascade(label="File",menu=self.fileMenu)

        # help menu
        self.helpMenu = Menu(self.menuBar,tearoff=0)
        self.helpMenu.add_command(label="About...",command=self.showAboutGUI)
        self.menuBar.add_cascade(label="Help",menu=self.helpMenu)
                                  
        # controls layout
        self.start_button.grid(row=0,column=0,sticky=E+W)
        self.stop_button.grid(row=0,column=1,sticky=E+W)
        self.controls.grid(row=1,column=0,columnspan=2)
        self.freqControls.grid(row=0,column=0,sticky=N+S+E+W)
        self.ampControls.grid(row=0,column=1,sticky=N+S+E+W)
        self.devControls.grid(row=0,column=2,sticky=N+S+E+W)
        
        # place the audio stream status label
        self.status_label.grid(row=2,column=0,columnspan=2,sticky=N+S+E+W)

        # recording controls
        self.record_start.grid(row=3,column=0,sticky=N+S+E+W)
        self.record_stop.grid(row=3,column=1,sticky=N+S+E+W)
        
        # handler for when the user closes or deletes the window
        # ensures that everthing has been stopped or closed
        self.master.protocol("WM_DELETE_WINDOW",self.__close_handler)

        # add weights so they can be resized
        num_cols,num_rows = master.grid_size()
        for c in range(num_cols):
            master.columnconfigure(c,weight=1)
        for r in range(num_rows):
            master.rowconfigure(r,weight=1)

    def setFileName(self):
        # ask user to select the path and filename of the recording
        newfpath = filedialog.asksaveasfilename(initialdir=os.getcwd(),initialfile=self.recFilePath,confirmoverwrite=False,defaultextension="*.csv",title="Select file path and name for recording",filetypes=[("CSV file","*.csv")])
        # check if returned path is empty
        # empty string evaluates to false
        # if not empty update filepath
        if newfpath.strip():
            self.recFilePath = newfpath

    # start recording button handler
    def startRec(self):
        self.recFile = open(self.recFilePath,'a')
        # enable stop button and disable start button
        self.record_start.config(state="disabled")
        self.record_stop.config(state="normal")

    # stop recording button handler
    def stopRec(self):
        from io import TextIOWrapper
        if isinstance(self.recFile,TextIOWrapper):
            if not self.recFile.closed:
                self.recFile.close()
        # renable button
        self.record_start.config(state="normal")
        self.record_stop.config(state="disabled")

    # handler to ensure that the sound stream was stopped when the window is
    # closed
    def __close_handler(self):
        from io import TextIOWrapper
        # stop stream cleaning buffers
        if hasattr(self,'stream'):
            if self.stream.active:
                self.stream.stop()
        # ensure file is closed
        # checking if the file is a valid file handler just in case
        if isinstance(self.recFile,TextIOWrapper):
            # check if file is open
            if not self.recFile.closed:
                self.recFile.close()

        # destroy window
        self.master.destroy()

    # update the freq used in the callback with the freq in the entry box
    def updateAmp(self):
        self.amp_val = self.amp.get()

    # update the amp used in the callback with the amp in the entry box
    def updateFreq(self):
        self.freq_val = self.freq.get()
        
    # callback for updating the stream status variable
    def updateStrStatus(self,*args):
        # if the audio stream is running, set the label background to green 
        if self.status_var.get():
            self.status_label.config(bg="green")
        # if not running, set the background to red
        else:
            self.status_label.config(bg="red")

    def startStream(self):
        # create stream using set settings
        self.stream = sd.OutputStream(device=self.devID.get(),channels=self.channels,samplerate=self.samplerate,
                                callback=self.sine_callback) # set callback to generate data
        # disable device ID so it can't be edited
        self.devEntry.configure(state="disabled")
        # start stream
        # starts generating requests and thus calls to sine_callback
        self.stream.start()
        self.status_var.set(True)
        
    def stopStream(self):
        # clear start idx to start the next signal at the start next run
        self.start_idx = 0
        # check if a stream has been created in a previous call
        if hasattr(self,'stream'):
            # check if the audio stream is active
            # if active, stop the stream
            if self.stream.active:
                # terminal processing without waiting for buffers to finish
                self.stream.abort()
                # also calling stop to clear buffers
                self.stream.stop()
                # check that stream has stopped
                # update some other status flags or something
                if self.stream.stopped:
                    print("Audio stream stopped!")
                    # change the status variable
                    self.status_var.set(False)
                    # renable device ID
                    self.devEntry.configure(state="normal")
                # if it hasn't finished, print an error message
                # not sure what else can be done in the event of this error
                else:
                    import sys
                    print("ERROR: Audio stream still running!!",file=sys.stderr)

    # audio callback for generating new data on requests
    def sine_callback(self,outdata,frames,time,status):
        self.auddata = outdata
        # if a status update was sent back
        if status:
            import sys
            # print it along error for now
            print(status,file=sys.stderr)
        # generate new time data based off settings
        t = (self.start_idx + np.arange(frames))/self.samplerate
        # reshape time vector
        t = t.reshape(-1,1)
        # generate new output data, overrides outdata content
        outdata[:] = self.amp_val * np.sin(2*np.pi*self.freq_val*t)
        # if the file is recording save the time and channel data to the file
        # uses file handler directly and leaves numpy to handle formatting
        if (self.recFile is not None) and (not self.recFile.closed):
            np.savetxt(self.recFile,np.concatenate((t,outdata),axis=1),delimiter=',')
        # update start index for next call
        self.start_idx += frames

    # create and display AboutGUIU
    def showAboutGUI(self):
        t = Toplevel()
        self.help_win = AboutWindow(t)

    # create and display SoundDevicesGUI
    def selectDevice(self):
        t = Toplevel()
        self.dev_list =SoundDevicesGUI(self.devID,t)

if __name__ == "__main__":
    root = Tk()
    view = SoundGUI(root)
    root.mainloop()
