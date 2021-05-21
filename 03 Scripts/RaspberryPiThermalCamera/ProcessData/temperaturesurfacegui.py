from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
import h5py
from tkinter import Tk,Label,Button,Scrollbar,Frame,END,CENTER,W,BOTTOM,TOP,BOTH,Toplevel,X
from tkinter import Grid
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

class SurfaceViewer:
    def __init__(self,master):
        self.master = master
        # set title of GUI
        master.title("Surface Viewer")
        # set file to be opened
        # updated later
        self.curr_file = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"

        # stop and play buttons
        self.play = Button(master,text='Play',command=self.play_data,padx=2)
        self.stop = Button(master,text='Stop',command=self.stop_data,padx=2)

        # scrollbar for more freely going through data
        self.data_scroll = Scrollbar(master,orient="horizontal",command=self.update_data)
        # current index being displayed
        self.curr_idx = 0

        self.fig = Figure(figsize=(5,5),dpi=100)
        self.ax = self.fig.add_subplot(111,projection='3d')
        # create canvas to render figure
        with h5py.File(self.curr_file,mode='r') as file:
            # find the maximum of the dataset
            dset_max = file['pi-camera-1'][()].max((0,1))
            np.nan_to_num(dset_max,copy=False)
            self.zmax = dset_max[dset_max<700.0].max()
            # get shape of the dataset
            self.data_shape = file['pi-camera-1'].shape
            # generate data for plotting
            self.xx,self.yy = np.meshgrid(np.arange(0,self.data_shape[1],1),np.arange(0,self.data_shape[0],1))
            # plot first element as surface to initialize plot
            self.ax.plot_surface(self.xx,self.yy,file['pi-camera-1'][:,:,0],cmap='magma')
            self.fig_canvas.draw()
        self.ax.set(xlabel='Row',ylabel='Column',zlabel='Temperature (C)')

        self.fig_canvas = FigureCanvasTkAgg(self.fig,self.master)
        self.fig_canvas.draw()
        # create frame to display toolbar
        self.toolbar_frame = Frame(master)
        self.fig_toolbar = NavigationToolbar2Tk(self.fig_canvas,self.toolbar_frame)
        self.fig_toolbar.update()
        self.fig_canvas.mpl_connect("key_press_event",self.on_key_press)
        
        ## widget layout
        self.play.grid(column=0,row=0,sticky='we',padx=10)
        self.stop.grid(column=1,row=0,sticky='we',padx=10)
        self.data_scroll.grid(column=0,row=1,columnspan=2,sticky='we')
        self.fig_canvas.get_tk_widget().grid(column=0,row=2,columnspan=3,sticky='nsew')
        self.toolbar_frame.grid(column=0,row=2,columnspan=3,sticky='nsew')

        # update weights of grid elements so they can be resized properly by user
        r,c = master.grid_size()
        for rr in range(r):
            master.columnconfigure(rr,weight=1)
        for cc in range(c):
            master.columnconfigure(cc,weight=1)

        master.update_idletasks()
        master.after_idle(lambda: master.minsize(master.winfo_width(), master.winfo_height()))

    # handler for toolbar and canvas key presses
    def on_key_press(event):
        key_press_handler(event,self.fig_canvas,self.fig_toolbar)

    # handler for updating display
    def update_data():
        pass

    # handler for playing data like a video
    def play_data():
        pass

    # handler for stopping playback
    def stop_data():
        pass
        

if __name__ == "__main__":
    root = Tk()
    view = SurfaceViewer(root)
    root.mainloop()
