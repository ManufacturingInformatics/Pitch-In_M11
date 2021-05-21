from tkinter import Tk,Entry,Label,StringVar,Button,StringVar,filedialog,Scrollbar,Frame,END,CENTER,W,BOTTOM,TOP,BOTH,Toplevel,X
from tkinter.ttk import Treeview
import h5py
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

class DataViewer:
    ''' DataViewer
        ======================
        David Miller, 2019
        The University of Sheffield 2019

        Wrapper GUI for a Matplotlib Figure showing the data given on creation. Called by HDF5Viewer when the user
        double clicks on a dataset.

        This GUI is designed a means of performing basic inspections of data stored in HDF5 files. If users want to
        perform something intensive or particular custom, they are advised to do so elsewhere.

        On creation, the DataViewer opens up the file specified by filename and accesses the dataset specified by the
        path dataName. It then decides how to plot the data based on the number of dimensions in the dataset. The used
        plots are as follows using their respective default options:

        |No. dims | Plots    |
        |---------|----------|
        |   1     | Line     |
        |   2     | Contourf |
        |   3     | Contourf |

        In the case of three dimensions, a scrollbar is added on top of the plot and provides a means for the user
        to select which 2D slice of the 3D dataset to show.

        The scrollbar only supports drag operations.

        Any higher dimensional data is ignored.

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
            
            Currently this only has support for clicking and dragging the scrollbar cursor. Other operations are
            ignored.
    '''
    # using filename and name of dataset
    # create a figure and display the data
    def __init__(self,master,dataName,filename):
        self.master = master
        # save creation options
        self.dataName = dataName
        self.filename = filename
        # set title
        self.master.title("Data Viewer")
        # label for graph
        self.title = StringVar()
        self.title.set('Displaying {}'.format(dataName))
        self.graph_title = Label(master,textvariable=self.title)
        self.graph_title.pack(side=TOP,pady=10,padx=10)
        # create figure
        self.fig = Figure(figsize=(5,5),dpi=100)
        self.axes = self.fig.add_subplot(111)
        # get data from dataset and plot data
        with h5py.File(filename,mode='r') as f:
            self.data_shape = f[dataName].shape
            # if the data is 1D, plot as line
            if len(self.data_shape)==1:
                self.axes.plot(f[dataName][()])
            # if data is 2D, plot as filled contour
            elif len(self.data_shape)==2:
                self.axes.contourf(f[dataName][()])
            # if data is 3D plot as contourf, but also add a scrollbar for navigation
            elif len(self.data_shape)==3:
                # create scroll bar for viewing different slices
                self.plot_scroll=Scrollbar(master,orient="horizontal",command=self.scroll_data)
                # add too gui
                self.plot_scroll.pack(side=TOP,fill=BOTH,expand=True)
                # plot first slice of data
                self.axes.contourf(f[dataName][:,:,0])
                # create index for current depth index
                self.depth_index = 0
                self.title.set("Displaying {} [{}]".format(dataName,self.depth_index))
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
            # set the scrollbar position to where the user dragged it to
            self.plot_scroll.set(float(args[1]),self.plot_scroll.get()[1])
            # reopen file
            with h5py.File(self.filename,mode='r') as f:
                self.axes.contourf(f[self.dataName][:,:,self.depth_index])
            # update canvas
            self.fig_canvas.draw()
            # update title
            self.title.set("Displaying {} [{}]".format(self.dataName,self.depth_index))
        
class HDF5Viewer:
    ''' HD5Viewer
        ======================
        David Miller, 2019
        The University of Sheffield 2019
        
        Presents a graphical means for users to view and inspect the contents of a HDF5 file.
        
        The user points the GUI to a HDF5 file by clicking on the 'Open' button, travelling to a folder and either
        double clicking on a file or selecting a file and clicking the 'Open' button.

        Once the file is selected, the user clicks the 'Scan' button which opens the file in read-only mode in
        a context manager and iterates through the file. The GUI populates a treeview in the GUI with the names of
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
            Iterates through the objects stored under item and updates the treeview with the information it finds
            under the node/leaft with the ID parent. If it finds a Group while iterating, explore_group is called
            with the newly discovered Group passed as the item to explore and the parent node to update under.
            Used in scan_file.

        scan_file(self):
            Attempts to open the file specified by the user. If a file path has yet to be specified it returns.
            If it's successful in opening the file, it iterates through its contents updating the treeview with the=
            information it finds. It uses the function explore_group to iterate through Groups it finds under root.
    '''
        
    def __init__(self,master):
        self.master = master
        # set title of image
        master.title("HDF5 File Viewer")

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
        
        ## add scrollbar for treeview
        # define scrollbar and set the action associated with moving the scrollbar to changing
        # the yview of the tree
        self.tree_scroll=Scrollbar(master,orient="vertical",command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=self.tree_scroll)
        
        # set grid layout for widgets using grid
        self.status_label.grid(columnspan=3,row=0)
        self.file_tree.grid(column=0,row=2,columnspan=3,sticky='nswe')
        self.scan_button.grid(column=0,row=1,sticky='we',padx=10)
        self.openfile_button.grid(column=1,row=1,sticky='we',padx=10)
        self.name_display.grid(column=2,row=1,sticky='we',padx=10)
        self.tree_scroll.grid(column=3,row=2,sticky='ns')
        # set weight parameters for control how the elements are resized when the user changes the window size
        master.columnconfigure(0,weight=1)
        master.columnconfigure(1,weight=1)
        master.columnconfigure(2,weight=1)
        master.columnconfigure(3,weight=1)
        master.rowconfigure(0,weight=1)
        master.rowconfigure(1,weight=1)
        master.rowconfigure(2,weight=1)
        master.rowconfigure(3,weight=1)
        # finish any idle tasks and set the minimum size of the window to cur
        master.update_idletasks()
        master.after_idle(lambda: master.minsize(master.winfo_width(), master.winfo_height()))
        
    # function for the user to select the HDF5 file to explore
    # opens file dialog for user to explore
    def open_file(self):
        self.status.set("Waiting for user to select file...")
        # open dialog to search for hdf5 file
        self.curr_file = filedialog.askopenfilename(initialdir="/",title="Select HDF5 file to inspect",filetypes=[("HDF5 files","*.hdf5")])
        self.name_display.delete(0,END)
        self.name_display.insert(0,self.curr_file)
    # function to explore HDF5 group and update tree
    # if it finds another HDF5 group it calls the functions to explore that group
    def explore_group(self,item,parent):
        self.status.set("Exploring {}...".format(item.name))
        #print("Exploring {}...".format(item.name))
        # iterate through items
        for v in item.values():
            #print(v.name,str(type(v)))
            # if it's a dataset, update shape entry with shape of dataset
            if isinstance(v,h5py.Dataset):
                self.file_tree.insert(parent,'end',text=v.name,values=(str(type(v)),str(v.shape),str(v.dtype)),open=True)
                self.file_tree['show']='tree headings'
            # if it's a group, call function to investiage it passing last group as parent to add new nodes to
            elif isinstance(v,h5py.Group):
                pkn = self.file_tree.insert(parent,'end',text=v.name,values=(str(type(v)),"({},)".format(len(v.keys()))),open=True)
                self.explore_group(v,pkn)           
    # explores target hdf5 file and displays the the keys of each entry
    # it the entry is a group, then it calls explore_group to explore further
    def scan_file(self):
        # if target file is set
        if self.curr_file != "/":
            # clear tree
            self.file_tree.delete(*self.file_tree.get_children())
            # open file in read mode and iterate through values
            with h5py.File(self.curr_file,'r') as file:
                for v in file.values():
                    # if it's a dataset, update shape entry with shape of dataset
                    if isinstance(v,h5py.Dataset):
                        self.file_tree.insert('','end',text=v.name,values=(str(type(v)),str(v.shape),str(v.dtype)),open=True)
                    # if it's a group, call function to investiage it
                    elif isinstance(v,h5py.Group):
                        pkn = self.file_tree.insert('','end',text=v.name,values=(str(type(v)),"({},)".format(len(v.keys()))),open=True)
                        self.explore_group(v,pkn)
            # update tree display
            self.file_tree['show']='tree headings'
            self.status.set("Finished scanning .../{}".format(self.curr_file[self.curr_file.rfind('/')+1:]))
            # finish idle tasks and set minimum window size to final window size
            self.master.update_idletasks()
            self.master.after_idle(lambda: self.master.minsize(self.master.winfo_width(), self.master.winfo_height()))
        else:
            self.status.set("No fime set!")
    def create_viewer(self,event):
        if self.curr_file != "/":
            # get the item selected
            iid = self.file_tree.identify('item',event.x,event.y)
            # get the values of the item to check if a dataset or group was selected
            if 'Dataset' in self.file_tree.item(iid,"values")[0]:
                self.status.set("Creating view for {}".format(self.file_tree.item(iid,"text")))
                # create new child window
                t = Toplevel()
                # initialize window inside new child window
                self.data_viewer = DataViewer(t,self.file_tree.item(iid,"text"),self.curr_file)

if __name__ == "__main__":         
    root = Tk()
    view = HDF5Viewer(root)
    root.mainloop()

        
