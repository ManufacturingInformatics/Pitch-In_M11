from tkinter import Tk,Entry,Label,StringVar,Button,StringVar,Listbox,filedialog,Scrollbar,Frame,END,CENTER,W,BOTTOM,TOP,BOTH,Toplevel,X
from tkinter.ttk import Treeview
import javabridge
import bioformats
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

class AttribViewer:
    def __init__(self,master,attribs):
        self.master = master
        self.master.title("XML Attributes Viewer")
        # create list box
        self.attrib_tree = Treeview(master,columns=("key","value"),show="tree")
        self.attrib_tree.pack(fill=BOTH,expand=1)
        # dimensions of the columns
        self.attrib_tree.column("key",width=200,anchor=CENTER,stretch=True)
        self.attrib_tree.column("value",width=200,anchor=CENTER,stretch=True)
        # text to display in headings
        self.attrib_tree.heading("key",text="Key")
        self.attrib_tree.heading("value",text="Value")
        self.attrib_tree['show']='headings'

        for k,v in attribs.items():
            self.attrib_tree.insert('','end',text='',values=(k.split('}')[1],v),open=False)

        # finish any idle tasks and set the minimum size of the window to cur
        master.update_idletasks()
        master.after_idle(lambda: master.minsize(master.winfo_width(), master.winfo_height()))
        

class DataViewer:
    ''' DataViewer
        ======================
        David Miller, 2019
        The University of Sheffield 2019

        Wrapper GUI for a Matplotlib Figure showing the images stored in the specified seroes. Called by BioViewer when the user
        double clicks on a Image Set.

        This GUI is designed a means of performing basic inspections of images stored in CXD files. If users want to
        perform something intensive or particular custom, they are advised to do so elsewhere.

        On creation, the DataViewer uses the reader class created when the file was scanned to access images stored in it

        The title above the Figure displays the name of the dataset and which index, if any is being displayed.

        Methods
        -------------------------
        on_key_press(event):
            Handler for key presses used on the Matploblib canvas

        scroll_data(self,*args):
            Handler for changing which slice of a 3D dataset is displayed. A new data index is chosen based on where
            the scrollbar cursor is dragged to. The index is calculated by multiplying the scrollbar positon [0,1] by
            the number of frames in the dataset and then converted to an integer.
            Updates the title and canvas on exit. Sets the scrollbar position to where the user left it.
            
            Currently this only has support for dragging the scrollbar cursor. Other operations are
            ignored.
    '''
    # using filename and name of dataset
    # create a figure and display the data
    def __init__(self,master,seriesNum,filename):
        self.master = master
        # save creation options
        self.series = seriesNum
        self.filename = filename
        # set title
        self.master.title("Data Viewer")
        # label for graph
        self.title = StringVar()
        self.title.set('Displaying Series {}'.format(self.series))
        self.graph_title = Label(master,textvariable=self.title)
        self.graph_title.pack(side=TOP,pady=10,padx=10)
        # reader for iamges
        self.reader = bioformats.ImageReader(self.filename)
        wrapper = bioformats.formatreader.make_image_reader_class()()
        # set up readers to read contents of file if necessary
        wrapper.setId(self.filename)
        # get number of images
        self.data_shape = (wrapper.getSizeX(),wrapper.getSizeY(),wrapper.getImageCount())
        # create figure
        self.fig = Figure(figsize=(5,5),dpi=100)
        self.axes = self.fig.add_subplot(111)
        # get data from dataset and plot data
        # create scroll bar for viewing different slices
        self.plot_scroll=Scrollbar(master,orient="horizontal",command=self.scroll_data)
        # add too gui
        self.plot_scroll.pack(side=TOP,fill=BOTH,expand=True)
        # plot first slice of data
        self.axes.contourf(self.reader.read(index=0,series=0,rescale=True).T,cmap='gray')
        # create index for current depth index
        self.depth_index = 0
        self.title.set("Displaying {} [{}]".format(self.series,self.depth_index))
        # create canvas to render figure
        self.fig_canvas = FigureCanvasTkAgg(self.fig,self.master)
        # update result
        self.fig_canvas.draw()
        # update canvas to set position and expansion options
        self.fig_canvas.get_tk_widget().pack(side=TOP,fill=BOTH,expand=True)

        ## add matplotlib toolbar
        self.fig_toolbar = NavigationToolbar2Tk(self.fig_canvas,self.master)
        self.fig_toolbar.update()
        # add to gui. always one row below the canvas
        self.fig_canvas._tkcanvas.pack(side=TOP,fill=BOTH,expand=True)
        ## add key press handlers
        self.fig_canvas.mpl_connect("key_press_event",self.on_key_press)
        # ensure elements are expandable in grid
        num_cols,num_rows = master.grid_size()
        for c in range(num_cols):
            master.columnconfigure(c,weight=1)
        for r in range(num_rows):
            master.rowconfigure(r,weight=1)
        # finish any idle tasks and set the minimum size of the window to cur
        master.update_idletasks()
        master.after_idle(lambda: master.minsize(master.winfo_width(), master.winfo_height()))
        
    # handler for matplotlib keypress events
    def on_key_press(event):
        key_press_handler(event,self.fig_canvas,self.fig_toolbar)

    # handler for using the scrollbar to view slices of data
    def scroll_data(self,*args):
        # if the user has dragged the scrollbar
        if args[0] == "moveto":
            # args is only one element in this case and is a number between 0 and 1
            # 0 is left most position and 1 is right most position
            # the data index is calculated as this number times depth and converted to integer
            self.depth_index = int(float(args[1])*self.data_shape[2])
            #print("target f ",self.depth_index)
            # set the scrollbar position to where the user dragged it to
            self.plot_scroll.set(float(args[1]),self.plot_scroll.get()[1])
            # reopen file
            self.axes.contourf(self.reader.read(index=self.depth_index,series=self.series,rescale=True).T,cmap='gray')
            # update canvas
            self.fig_canvas.draw()
            # update title
            self.title.set("Displaying Series {} [{}]".format(self.series,self.depth_index))
            
class BioViewer:
    ''' HD5Viewer
        ======================
        David Miller, 2019
        The University of Sheffield 2019
        
        Presents a graphical means for users to view and inspect the contents of a Bioformats file.
        
        The user points the GUI to a valid Bioformats file by clicking on the 'Open' button, travelling to a folder and either
        double clicking on a file or selecting a file and clicking the 'Open' button.

        Once the file is selected, the user clicks the 'Scan' button which opens the file in read-only mode in
        a context manager and iterates through the file. The GUI populates a Treeview in the GUI with the names of
        objects, their type, shape and data type (datasets only) in a way that mirrows the structures of the file.
        If the type column describes the type of object inside the HDF5 file. If the object is a Group, the contents
        under the shape column represents the number of items stored one level under it. If the object is a Dataset,
        the shape represents the shape of the array stored in the dataset. The data type column is only updated for
        datasets and is the type of data stored in the dataset.

        This class keeps control of the specified file for the minimum amount of time only accessing it for the
        duration of scan_file.

        If the user double clicks on a dataset, a DataViewer GUI is opened in another window. This attempts to display
        the dataset, choosing the appropriate plots based on the shape of the dataset.

        Methods:
        --------------------------
        open_file(self):
            Opens a file dialog for the user to select the HDF5 file they want to inspect.
            
        explore_group(self,item,parent):
            Iterates through the objects stored under item and updates the Treeview with the information it finds
            under the node/leaft with the ID parent. If it finds a Group while iterating, explore_group is called
            with the newly discovered Group passed as the item to explore and the parent node to update under.
            Used in scan_file.

        scan_file(self):
            Attempts to open the file specified by the user. If a file path has yet to be specified it returns.
            If it's successful in opening the file, it iterates through its contents updating the Treeview with the=
            information it finds. It uses the function explore_group to iterate through Groups it finds under root.
    '''
        
    def __init__(self,master):
        self.master = master
        # set title of image
        master.title("Bioformats File Viewer")

        ## initialize internal variables used
        # set current file as blank
        self.curr_file = "/"
        self.status = StringVar()
        
        ## initialize widgets
        # status label indicating progress or errors
        self.status_label = Label(master,textvariable=self.status)
        self.status.set("Waiting for filename...")
        # button to scan target HDF5 file
        self.scan_button = Button(master,text="Scan File",command=self.scan_file,padx=2)
        # button to chose hdf5 file
        self.openfile_button = Button(master,text="Open File",command=self.open_file,padx=2)
        # box to display current filename
        self.name_display = Entry(master,text="Current filename")
        ## setup tree headings
        # tree view for file layout
        self.file_tree = Treeview(master,columns=("htype","shape","dtype"),show="tree")
        # add double click handler
        # <Double-1> double left click handler
        self.file_tree.bind("<Double-1>",self.create_viewer)
        # dimensions of the columns
        self.file_tree.column("htype",width=200,anchor=CENTER)
        self.file_tree.column("shape",width=200,anchor=CENTER)
        self.file_tree.column("dtype",width=200,anchor=CENTER)
        # text to display in headings
        self.file_tree.heading("htype",text="Item Type")
        self.file_tree.heading("shape",text="Shape")
        self.file_tree.heading("dtype",text="Data Type")
        self.file_tree['show']='headings'

        # create Treeview for XML data
        self.xml_tree = Treeview(master,columns=("url","text","nchild","attribs"),show="tree")
        # <Double-1> double left click handler
        self.xml_tree.bind("<Double-1>",self.show_attribs)
        self.xml_tree.column("url",width=200,anchor=CENTER)
        self.xml_tree.column("text",width=200,anchor=CENTER)
        self.xml_tree.column("nchild",width=200,anchor=CENTER)
        self.xml_tree.column("attribs",width=200,anchor=CENTER)
        # text to display in headings
        self.xml_tree.heading("url",text="URL")
        self.xml_tree.heading("text",text="Text")
        self.xml_tree.heading("nchild",text="No. of Children")
        self.xml_tree.heading("attribs",text="Attributes")
        self.xml_tree['show']='headings'
        
        ## add scrollbar for Treeview
        # define scrollbar and set the action associated with moving the scrollbar to changing
        # the yview of the tree
        self.file_scroll=Scrollbar(master,orient="vertical",command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=self.file_scroll)

        self.xml_scroll=Scrollbar(master,orient="vertical",command=self.xml_tree.yview)
        self.xml_tree.configure(yscrollcommand=self.xml_scroll)
        
        # set grid layout for widgets using grid
        self.status_label.grid(columnspan=3,row=0)
        self.scan_button.grid(column=0,row=1,sticky='we',padx=10)
        self.openfile_button.grid(column=1,row=1,sticky='we',padx=10)
        self.name_display.grid(column=2,row=1,sticky='we',padx=10)
        self.xml_tree.grid(column=0,row=2,columnspan=3,sticky='nswe')
        self.xml_scroll.grid(column=3,row=2,sticky='ns')
        self.file_tree.grid(column=0,row=3,columnspan=3,sticky='nswe')
        self.file_scroll.grid(column=3,row=3,sticky='ns')
        # set weight parameters for control how the elements are resized when the user changes the window size
        num_cols,num_rows = master.grid_size()
        for c in range(num_cols):
            master.columnconfigure(c,weight=1)
        for r in range(num_rows):
            master.rowconfigure(r,weight=1)
        # finish any idle tasks and set the minimum size of the window to cur
        master.update_idletasks()
        master.after_idle(lambda: master.minsize(master.winfo_width(), master.winfo_height()))

    def findAttribs(self,node,target):
        if target in node.tag.split('}')[1]:
            return node.attrib
        else:
            for cc in node.getchildren():
                return findAttribs(cc,target)

    def show_attribs(self,event):
        if len(self.xml_tree.get_children())>0:
            iid = self.xml_tree.identify('item',event.x,event.y)
            tag = self.xml_tree.item(iid,"text")
            attrs = self.findAttribs(self.root,tag)
            # initialize window inside new child window
            t = Toplevel()
            self.attrib_view = AttribViewer(t,attrs)
    
    # function for the user to select the HDF5 file to explore
    # opens file dialog for user to explore
    def open_file(self):
        self.status.set("Waiting for user to select file...")
        # open dialog to search for hdf5 file
        self.curr_file = filedialog.askopenfilename(initialdir="/",title="Select CXD file to inspect",filetypes=[("CXD files","*.cxd")])
        self.name_display.delete(0,END)
        self.name_display.insert(0,self.curr_file)
        
    # function to explore CXD XML metadata and update the tree
    def explore_group(self,item,parent):
        if item is not None:
            #print(item.tag)
            # add new entry under parent item setting text as
            #print("Adding {} under {}".format(item.tag,self.xml_tree.item(parent,'text')))
            pkn=self.xml_tree.insert(parent,'end',text=item.tag.split('}')[1],values=(item.tag.split('}')[0][1:],item.text,len(item.getchildren()),"<{} attributes>".format(len(item.attrib))),open=False)
            for child in item.getchildren():
                self.explore_group(child,pkn)
            self.xml_tree['show']='tree headings'
        else:
            return 0

    # explores target hdf5 file and displays the the keys of each entry
    # it the entry is a group, then it calls explore_group to explore further
    def scan_file(self):
        # if target file is set
        if self.curr_file != "/":
            self.scan_button.config(state='disabled')
            self.openfile_button.config(state='disabled')
            # clear tree
            self.status.set("Scanning...")
            self.master.update_idletasks()
            self.file_tree.delete(*self.file_tree.get_children())
            self.xml_tree.delete(*self.xml_tree.get_children())

            ## SCAN XML DOCUMENT ##
            # get xml data as ET and get root
            self.status.set("Getting XML data as ElementTree")
            self.master.update_idletasks()
            self.root = ET.ElementTree(ET.fromstring(bioformats.get_omexml_metadata(self.curr_file))).getroot()
            # add entry for XML data
            pkn=self.xml_tree.insert('','end',text=self.root.tag.split('}')[1],values=(self.root.tag.split('}')[0][1:],self.root.text,len(self.root.getchildren()),"<{} attributes>".format(len(self.root.attrib))),open=False)
            # scan xml data
            for child in self.root.getchildren():
                self.explore_group(child,pkn)

            ## SCAN IMAGES ##
            self.status.set("Collecting information about image series")
            self.master.update_idletasks()
            # get information about images
            reader = bioformats.formatreader.make_image_reader_class()()
            # set up readers to read contents of file if necessary
            reader.setId(self.curr_file)
            # get number of images
            data_shape = (reader.getSizeX(),reader.getSizeY(),reader.getImageCount())
            # set the number of image series in the file
            self.num_series = reader.getSeriesCount()
            # change reader to image reader so it can be used later
            reader = bioformats.ImageReader(self.curr_file)
            for ss in range(self.num_series):
                # add entry for image type
                self.file_tree.insert('','end',text="Image Series {}".format(ss),values=("Image Set",str(data_shape),str(reader.read(index=0,series=ss).dtype)),open=False)
            # update tree display
            self.file_tree['show']='tree headings'
            self.status.set("Finished scanning .../{}".format(self.curr_file[self.curr_file.rfind('/')+1:]))
            # re-enable buttons
            self.scan_button.config(state='normal')
            self.openfile_button.config(state='normal')
        else:
            self.status.set("No file set!")
        # perform any remainins tasks
        self.master.update_idletasks()
        # account for any resizing that occurs
        self.master.after_idle(lambda: self.master.minsize(self.master.winfo_width(), self.master.winfo_height()))
            
    def create_viewer(self,event):
        if self.curr_file != "/":
            # get the item selected
            iid = self.file_tree.identify('item',event.x,event.y)
            # get the values of the item to check if a dataset or group was selected
            if 'Image Set' in self.file_tree.item(iid,"values")[0]:
                self.status.set("Creating view for {}".format(self.file_tree.item(iid,"text")))
                # create new child window
                t = Toplevel()
                # initialize window inside new child window
                self.data_viewer = DataViewer(t,self.file_tree.item(iid,"text").split("Image Series")[1],self.curr_file)

if __name__ == "__main__":         
    root = Tk()
    javabridge.start_vm(class_path=bioformats.JARS)
    view = BioViewer(root)
    root.mainloop()
    javabridge.kill_vm()

        
