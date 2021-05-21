from tkinter import Tk,Toplevel,Label,Frame,Button,filedialog,messagebox,N,S,E,W,CENTER,TOP,BOTH,StringVar,BooleanVar,LabelFrame,Checkbutton
from tkinter.ttk import Combobox, Progressbar

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
# logging class
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

try:
    import h5py
    hdf5Support = True
except ImportError:
    print("Couldn't find h5py!")
    hdf5Support = False

import os

# class for radon solving
# produces weird results
class RadonReconstruction:
    def __init__(self,n_el,dist=1,h0=0.08):
        from itertools import combinations
        # problem, if I go down to say, 10, then some of the lines
        # aren't displayed.
        self.image_pixels = 100

        #self.img = np.zeros((self.image_pixels, self.image_pixels),
        #                    dtype=np.float)

        self.img = np.full((self.image_pixels, self.image_pixels),h0,
                            dtype=np.float)

        # Above should be calculated elsewhere and is only for
        # plotting purposes.
        self.x_center = self.image_pixels/2
        self.y_center = self.image_pixels/2
        self.radius = self.image_pixels/2 - self.image_pixels/10
        #
        # generate combinations of coils according to specification
        # excitation order
        #self.logfile = list(combinations(range(n_el), 2))
        self.logfile = eit_scan_lines(n_el,dist)
        # specify the angle of the electrode's positions
        #self.theta_points = [np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4,
        #                     0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.theta_points = [np.pi + (i*np.pi/(n_el/2)) if (np.pi + (i*np.pi/(n_el/2)))%2*np.pi != 0 else 0 for i in range(n_el)]

    def makeimages(self, data):
        #if len(data) != len(self.logfile):
        #    raise ValueError(
        #        "the datasets must match the logfile specification"
        #    )

        # calculate the positions of the electrodes in the image space
        n1 = np.add(self.x_center*np.ones(len(self.theta_points)),
                    self.radius*np.cos(self.theta_points))
        n2 = np.add(self.y_center*np.ones(len(self.theta_points)),
                    self.radius*np.sin(self.theta_points))

        x = n1.astype(np.int)
        y = n2.astype(np.int)

        d = dict()
        for i, (point1, point2) in enumerate(self.logfile):

            # get the gradient angle theta
            g1 = x[point2] - x[point1]
            g2 = y[point2] - y[point1]
            angle = np.rad2deg(np.arctan2(g2, g1))

            if angle < 0:
                angle = angle + 180
            elif angle >= 180:
                angle = 0.0

            # get the line coordinates for the connection of the two
            # considered electrodes
            l_x, l_y = ll(x[point1], y[point1], x[point2], y[point2])

            # if we are close to an existing angle reuse this
            for a in d:
                if abs(a-angle) < 5:
                    d[a][l_x, l_y] += data[i]
                    break
            else:  # create a new array in this slot
                img = np.zeros((self.image_pixels, self.image_pixels),
                               dtype=np.float)
                img[l_x, l_y] = data[i]
                d[angle] = img

        deg = list(sorted(d))

        return d, deg

    def reconstruct(self, d, deg):
        interp_projections = []

        # now interpolate each angled projection, to get an
        # approximate radon transform from the EIT data
        for i, degi in enumerate(deg):
            projections = radon(d[degi], theta=deg, circle=True)
            p = projections[:, i]

            # problem is lines at angle, or indices next to each other
            # should just be one value..
            #
            # sift through p. if
            for t in range(len(p)):
                if p[t] > 0:
                    if p[t+1] > p[t]:
                        p[t] = 0
                    if p[t] < p[t-1] or p[t] < p[t-2]:
                        p[t] = 0

            nonzeroind = np.nonzero(p)[0]
            xp = nonzeroind
            yp = p[nonzeroind]

            xp = np.append([0], nonzeroind)
            yp = np.append(yp[0], yp)

            xp = np.append(xp, [len(p)-1])
            yp = np.append(yp, yp[-1])

            # Now interpolate to acquire the preimages for the inverse
            # radon transform
            xnew = np.linspace(0, len(p), len(p))
            yinterp = np.interp(xnew, xp, yp)
            interp_projections.append(yinterp)

        interp_projections = np.array(interp_projections).transpose()

        reconstruction = iradon(interp_projections, theta=deg, circle=True)

        # SART reconstruction with two iterations for improved
        # accuracy
        reconstruction_sart = iradon_sart(interp_projections,
                                          theta=np.array(deg))
        reconstruction_sart2 = iradon_sart(interp_projections,
                                           theta=np.array(deg),
                                           image=reconstruction_sart)

        image = reconstruction_sart2
        return image

    def eit_reconstruction(self, data):
        d, deg = self.makeimages(data)
        return self.reconstruct(d, deg)

# class for provifing a way for users to monitor the export progress of the GUI
# has two progress bars for local and global progress
# also has a status message meant for title of current local task being performed
class ExportProgressWindow:
    def __init__(self,master):
        self.master = master
        self.master.title("Export progress")
        # add label used as title
        self.labelTitle = Label(self.master,text="Export Progress",font="Arial 10 bold")
        # add label that acts as a status message
        self.status = StringVar(value='')
        self.statusLabel = Label(self.master,textvariable=self.status)
        # local progress bar for a local operation
        self._localProg = Progressbar(self.master,length=200,orient="horizontal",mode="determinate",maximum=1.0)
        # global progress bar
        self._globalProg = Progressbar(self.master,length=200,orient="horizontal",mode="determinate",maximum=1.0)

        # pack widgets
        self.labelTitle.grid(row=0,column=0)
        self.statusLabel.grid(row=1,column=0)
        self._localProg.grid(row=2,column=0)
        self._globalProg.grid(row=3,column=0)

        self.master.update_idletasks()

    ## methods for setting the progress bar and status messages
    def setLocal(self,lp):
        self._localProg["value"] = lp
        self.master.update_idletasks()

    def setGlobal(self,gp):
        self._globalProg["value"] = gp
        self.master.update_idletasks()

    def setStatus(self,news):
        self.status.set(news)
        self.master.update_idletasks()


# reverse results displaying window
# can be used in combination with a solver to solve measurements and display the result
# users can also normalize the data using the checkboxes provided
# to display data as is without any solver
# - set solver to None
# - set results to None
# - assign a supported solver type using keyword displayType
# e.g. ReverseResultsWindow(Toplevel(self.master),None,data,displayType="GREIT")
# slider iterates over data row wise
class ReverseResultsWindow:
    def __init__(self,master,solver,data,**kwargs):
        self.master = master

        ## setup checkboxes for normalizing of the data
        # normalization class passed to the plotting command
        self.norm = None
        # variables for tracking box state
        self.normCheckVar = BooleanVar(self.master,value=0)
        self.logCheckVar = BooleanVar(self.master,value=0)
        self.logMeanCheckVar = BooleanVar(self.master,value=0)
        # check boxes inside a label frame
        self.checkFrame = LabelFrame(self.master,text="Normalization")
        self.normCheck = Checkbutton(self.checkFrame,text="Normalize",variable=self.normCheckVar,command=lambda nt="norm": self.changeNorm(nt))
        self.logCheck = Checkbutton(self.checkFrame,text="Log Normalize",variable=self.logCheckVar,command=lambda nt="lognorm": self.changeNorm(nt))
        self.logMeanCheck = Checkbutton(self.checkFrame,text="Log (Mean) Normalize",variable=self.logMeanCheckVar,command=lambda nt ="lognormmean": self.changeNorm(nt))
        # arrange checkboxes
        self.normCheck.grid(row=0,column=0,sticky=N+S+E+W)
        self.logCheck.grid(row=0,column=1,sticky=N+S+E+W)
        self.logMeanCheck.grid(row=0,column=2,sticky=N+S+E+W)
        # pack checkbox container
        # has to be check as matplotlib tkinter widgets do not support grid
        self.checkFrame.pack(side=TOP,fill=BOTH,expand=True)
        
        # setup figure
        self.fig = Figure(figsize=(5,5),dpi=100)
        self.ax = self.fig.add_subplot(111)

        # create axes for colorbar
        div = make_axes_locatable(self.ax)
        self.cax = div.append_axes("right",size="5%",pad=0.05)
        # create canvas to attach figure to
        self.fig_canvas = FigureCanvasTkAgg(self.fig,self.master)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(side=TOP,fill=BOTH,expand=True)

        # add toolbar
        self.fig_toolbar = NavigationToolbar2Tk(self.fig_canvas,self.master)
        self.fig_toolbar.update()
        # add to gui. always one row below the canvas
        self.fig_canvas._tkcanvas.pack(side=TOP,fill=BOTH,expand=True)
        ## add key press handlers
        self.fig_canvas.mpl_connect("key_press_event",self.on_key_press)
        
        # save solver to be used later
        self.solver = solver
        # save reference to data to iterate over
        self.data = data

        # if no solver is passed, rely on results of keyword to state the type of results
        if solver is None:
            if "displayType" in kwargs.keys():
                self.solverType = kwargs["displayType"]
            # if no type was passed, destroy master window?
            else:
                messagebox.showerror("Wrong display type!","Solver type not stated for the supplied data!")
                self.master.destroy()
                return
        else:
            if type(solver) == greit:
                self.solverType = "GREIT"
            elif type(solver) == jacobian:
                self.solverType = "JAC"
            elif type(solver) == bp:
                self.solverType = "BP"
            elif type(solver) == RadonReconstruction:
                self.solverType = "RADON"
                
        # separate mesh structure of solver into separate parts for
        # easier syntax
        if hasattr(solver,'mesh'):
            mesh_obj = solver.mesh
            self.pts = mesh_obj['node']
            self.tri = mesh_obj['element']
            self.x = self.pts[:, 0]
            self.y = self.pts[:, 1]
        # if just displaying results
        # ask user how many coils there 
        elif 'mesh' in kwargs.keys():
            mesh_obj = kwargs['mesh']
            self.pts = mesh_obj['node']
            self.tri = mesh_obj['element']
            self.x = self.pts[:, 0]
            self.y = self.pts[:, 1]

        print("generating results for",self.solverType)

        ## create slider
        self.axsl = self.fig.add_axes([0.15, 0.01, 0.65, 0.03])
        # if a solver was given, then the limits of the slider are based off the data array given
        if solver is not None:
            self.sl = Slider(self.axsl,"M. Cycle",1,data.shape[0]-1,valinit=1,valfmt="%d",valstep=1)
        # if no solver was given, then we're displaying results so the limits are taken from the results data array
        else:
            self.sl = Slider(self.axsl,"M. Cycle",0,data.shape[-1]-1,valinit=0,valfmt="%d",valstep=1)
        # assugn slider callback
        self.sl.on_changed(self.update)

        # set title
        if hasattr(self.master,'title'):
            self.master.title(self.solverType + " Results")

        # make the main window contents resizable 
        num_cols,num_rows = master.grid_size()
        for c in range(num_cols):
            master.columnconfigure(c,weight=1)
        for r in range(num_rows):
            master.rowconfigure(r,weight=1)

    # methood for changing what type of normalization to perform on the data
    def changeNorm(self,normtype):
        # update the type of normalization based on the state of the checkboxes
        if normtype=="norm" and self.normCheckVar.get():
            print("norm type changed to basic norm")
            self.norm=Normalize()
            self.logCheckVar.set(0)
            self.logMeanCheckVar.set(0)
        elif normtype=="lognorm" and (self.logCheckVar.get()):
            print("norm type changed to basic log norm")
            self.norm=LogNorm()
            self.normCheckVar.set(0)
            self.logMeanCheckVar.set(0)
        elif normtype=="lognormmean" and (self.logMeanCheckVar.get()):
            print("norm type changed to log norm based off mean and std")
            self.norm=2
            self.normCheckVar.set(0)
            self.logCheckVar.set(0)
        elif (not self.normCheckVar.get()) and (not self.logCheckVar.get()) and (not self.logMeanCheckVar.get()):
            print("norm type cleared")
            self.norm=None
        # run update for current index
        self.update(0)

    # callback function for slider
    # gets row of dataset corresponding to index of the slider
    # reference data for solvers currently fixed to first row of dataset
    def update(self,val):
        # remove colorbar and clear axes
        if hasattr(self,'cb'):
            self.cb.remove()
        self.ax.clear()

        # rebuild axes for colobar
        div = make_axes_locatable(self.ax)
        self.cax = div.append_axes("right",size="5%",pad=0.05)
        
        ## resolve and update plot according to solver type
        if self.solverType == "GREIT":
            try:
                # if a solver was given
                if self.solver is not None:
                    # set reference as first line of data
                    self.f0 = self.data[0,:]
                    # set new data to compare against f0
                    self.f1 = self.data[int(self.sl.val),:]
                    # solve
                    ds = self.solver.solve(self.f1,self.f0,normalize=False)
                    # parse results
                    # remove NaNs
                    _,_,ds = self.solver.mask_value(ds,mask_value=np.NAN)
                    # get data
                    img = np.real(ds)
                # if there is no solver, then just display the results
                # the solver type controls what type of plotting is needed
                else:
                    img = np.real(self.data[:,:,int(self.sl.val)])
                ## update normalizer if necessary
                if self.norm==2:
                    self.norm = LogNorm(img.mean() + 0.5 * img.std(), img.max(), clip='True')
                # plot results
                im = self.ax.imshow(img,interpolation='none',cmap=cm.rainbow,norm=self.norm)
                # generate colorbar
                self.cb = self.fig.colorbar(im,cax=self.cax,orientation='vertical')
                self.ax.axis('equal')
                self.ax.set_title(rf"$\Delta$ Conductivity Map, {self.solverType}")
                return
            except Exception as err:
                messagebox.showerror(title="GREIT Solving Error",message=str(err))
                return
        # for jacobian solver method
        elif self.solverType == "JAC":
            # run solver
            try:
                if self.solver is not None:
                    # set reference as first line of data
                    self.f0 = self.data[0,:]
                    # set new data to compare against f0
                    self.f1 = self.data[int(self.sl.val),:]
                    # run solver
                    ds = self.solver.solve(self.f1,self.f0,normalize=False)
                    # ?
                    ds_n = sim2pts(pts,tri,np.real(ds))
                # if there is no solver, then just display the results
                # the solver type controls what type of plotting is needed
                else:
                    ds_n = self.data[:,int(self.sl.val)]
                # update normalizer if necessary
                if self.norm==2:
                    self.norm = LogNorm(ds_n.mean() + 0.5 * ds_n.std(), ds_n.max(), clip='True')
                # plot results
                im = self.ax.tripcolor(self.x,self.y,self.tri,ds_n,shading='flat',cmap=cm.rainbow,norm=self.norm)
                # setup colorbar and labels
                self.cb = self.fig.colorbar(im,cax=self.cax,orientation='vertical')
                self.ax.set_title(rf"$\Delta$ Conductivity Map, {self.solverType}")
                self.ax.set_aspect('equal')
                return
            except Exception:
                messagebox.showerror(title="JAC Solving Error", message=str(err))
                return
        # for back propogation method
        elif self.solverType == "BP":
            try:
                if self.solver is not None:
                    # set reference as first line of data
                    self.f0 = self.data[0,:]
                    # set new data to compare against f0
                    self.f1 = self.data[int(self.sl.val),:]
                    # run solver
                    # don't know what weight is for
                    ds = self.solver.solve(self.f1,self.f0)
                # if there is no solver, then just display the results
                # the solver type controls what type of plotting is needed
                else:
                    ds = np.real(self.data[:,int(self.sl.val)])
                # update normalizer if necessary
                if self.norm==2:
                    self.norm = LogNorm(ds.mean() + 0.5 * ds.std(), ds.max(), clip='True')
                    
                # plot results
                im = self.ax.tripcolor(self.x,self.y,self.tri,ds,cmap=cm.rainbow,norm=self.norm)
                self.cb = self.fig.colorbar(im,cax=self.cax,orientation='vertical')
                self.ax.set_title(rf"$\Delta$ Conductivity Map, {self.solverType}")
                self.ax.axis('equal')
                return
            except RuntimeError as err:
                messagebox.showerror(title="BP Solving Error",message=str(err))
                return
        # inverse radon solver
        # see RadonReconstruction class
        elif self.solverType == "RADON":
            try:
                if self.solver is not None:
                    # set new data to compare against f0
                    self.f1 = self.data[int(self.sl.val),:]
                    # reconstruct the image using new data
                    img = self.solver.eit_reconstruction(self.f1)
                # if there is no solver, then just display the results
                # the solver type controls what type of plotting is needed
                else:
                    img = np.real(self.data[:,:,int(self.sl.val)])

                # update solver if necessary
                if self.norm==2:
                    self.norm = LogNorm(img.mean() + 0.5 * img.std(), img.max(), clip='True')
                # plot results
                im = self.ax.imshow(img,interpolation='none',cmap=cm.rainbow,norm=self.norm)
                self.cb = self.fig.colorbar(im,cax=self.cax,orientation='vertical')
                self.ax.axis('equal')
                self.ax.set_title(rf"$\Delta$ Conductivity Map, {self.solverType}")
            except RuntimeError as err:
                messagebox.showerror(title="RADON Solving Error",message=str(err))
                return

    # handler for tk canvas key presses
    def on_key_press(self,event):
        key_press_handler(event,self.fig_canvas,self.fig_toolbar)

# window for displaying simulated excitation lines
class ForwardResultsWindow:
    def __init__(self,master,mesh,ex_pos,results,solver,ex_mat):
        self.master = master

        self.master.title("Forward Results")
        
        # parse data
        self.pts = mesh['node']
        self.tri = mesh['element']
        self.perm = mesh['perm']
        self.solver = solver
        self.ex_mat = ex_mat
        self.ex_pos = ex_pos
        self.results = results
        
        # setup figure
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        # create canvas to attach figure to
        self.fig_canvas = FigureCanvasTkAgg(self.fig,self.master)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(side=TOP,fill=BOTH,expand=True)

        # add toolbar
        self.fig_toolbar = NavigationToolbar2Tk(self.fig_canvas,self.master)
        self.fig_toolbar.update()
        # add to gui. always one row below the canvas
        self.fig_canvas._tkcanvas.pack(side=TOP,fill=BOTH,expand=True)
        ## add key press handlers
        self.fig_canvas.mpl_connect("key_press_event",self.on_key_press)
        
        # create vector to display results?
        self.vf = np.linspace(min(self.results),max(self.results),32)
        # draw lines to show lines of predicted magnetic potetntial
        self.ax.tricontour(self.pts[:,0],self.pts[:,1],self.tri,self.results,self.vf,cmap=cm.viridis)
        # draw mesh structure
        self.ax.tripcolor(self.pts[:,0],self.pts[:,1],self.tri,np.real(mesh['perm']),edgecolor='k',shading='flat',alpha=0.5,cmap=cm.Greys)
        # draw electrodes
        self.ax.plot(self.pts[ex_pos,0],self.pts[ex_pos,1],'ro')
        for i,e in enumerate(ex_pos):
            self.ax.text(self.pts[e,0],self.pts[e,1],str(i+1),size=12)
        self.ax.set_title(f"Equi-potential lines, Pair {self.ex_mat[0]}")
        # cleanup
        self.ax.set_aspect('equal')
        self.ax.set_ylim([-1.2, 1.2])
        self.ax.set_xlim([-1.2, 1.2])
        self.fig.set_size_inches(6, 6)

        ## create slider
        self.axsl = self.fig.add_axes([0.25, 0.01, 0.65, 0.03])
        self.sl = Slider(self.axsl,"Pair Idx",1,ex_mat.shape[0],valinit=0,valfmt="%d",valstep=1)
        self.sl.on_changed(self.update)

        # make the main window contents resizable 
        num_cols,num_rows = master.grid_size()
        for c in range(num_cols):
            master.columnconfigure(c,weight=1)
        for r in range(num_rows):
            master.rowconfigure(r,weight=1)

    # function for updating the plot based on the position of the slider
    # moving the slider changes which coil pair to use in simulation
    def update(self,val):
        # get next pair
        ex_line = self.ex_mat[int(self.sl.val-1)].ravel()
        # run solver
        f,_ = self.solver.solve(ex_line,self.perm)
        f = np.real(f)
        # clear axes
        self.ax.clear()
        # replit
        self.ax.tricontour(self.pts[:,0],self.pts[:,1],self.tri,f,self.vf,cmap=cm.viridis)
        self.ax.tripcolor(self.pts[:,0],self.pts[:,1],self.tri,np.real(self.perm),edgecolor='k',shading='flat',alpha=0.5,cmap=cm.Greys)
        # draw electrodes
        self.ax.plot(self.pts[self.ex_pos,0],self.pts[self.ex_pos,1],'ro')
        # assign labels
        for i,e in enumerate(self.ex_pos):
            self.ax.text(self.pts[e,0],self.pts[e,1],str(i+1),size=12)
        # set title
        self.ax.set_title(f"Equi-potential lines, Pair {self.ex_mat[int(self.sl.val-1)]}")
        # setup size of the plot to make it more visually pleasing
        self.ax.set_aspect('equal')
        self.ax.set_ylim([-1.2, 1.2])
        self.ax.set_xlim([-1.2, 1.2])
        # complete idle tasks for figure canvas to ensure that the results are displayed
        self.fig.canvas.draw_idle()

    # handler for key presses on the figure canvas
    def on_key_press(self,event):
        key_press_handler(event,self.fig_canvas,self.fig_toolbar)

# class that's a combination of the ExcitationOrderGUI and the solving examples 
class PathPatternSolverGUI:
    def __init__(self,master):
        self.master = master
        if hasattr(master,'title'):
            master.title("EIT Offline Solver GUI")
            
        # create frame to hold ExcitationOrderGUI
        self.gui_frame = Frame(self.master)
        # add gui to frame
        self.path_gui = ExcitationOrderGUI(self.gui_frame,32)
        ## file variables
        # path to data file
        self.datapath = ''
        # data matrx
        self.data = np.empty((0,0),dtype='float')
        # method to use for reverse problem
        self.method = StringVar(value="GREIT")
        # background value of permeability
        self.bkh = 0.08
        # flag to indicate if the forward problem has been solved
        # used as a check for if the background mesh object has been built
        self.fwdSolved = False
        self.revSolved = False
        # solvers
        self.fwdSolver = None
        self.revSolver = None
        
        # frame for solver controls
        self.solverControls = LabelFrame(self.master,text="Solver Controls",font="Arial 10 bold italic")
        # frame for file controls
        self.fileControls = LabelFrame(self.master,text="File Controls",font="Arial 10 bold italic")
        
        ## combobox to set reverse solver type
        self.method_button = Label(self.solverControls,text="Reverse Solver",font="Arial 10 bold").grid(row=0,column=0,sticky=N+S+E+W)
        # combobx of options
        self.solver_cbox = Combobox(self.solverControls,state="readonly",values=["GREIT","JAC","BP","RADON","ALL"])
        # set initial value to first option
        self.solver_cbox.current(0)
        self.solver_cbox.grid(row=1,column=0,sticky=N+S+E+W)
        # handler for when the user selects a value
        # update the method value used with solver methods
        self.solver_cbox.bind("<<ComboboxSelected>>",lambda event,self=self:self.method.set(self.solver_cbox.get()))
        # button for forward solving
        self.start_forward = Button(self.solverControls,text="Solve Forward",command=self.forwardSolve).grid(row=2,column=0,sticky=N+E+W+S)
        # button for reverse solving
        self.start_reverse = Button(self.solverControls,text="Solve Reverse",command=self.reverseSolve).grid(row=3,column=0,sticky=N+E+W+S)

        # button for reading in a new file
        self.file_button = Button(self.fileControls,text="Import Measurments",command = self.readDataFile).grid(row=0,column=0,sticky=N+E+W+S)
        # button for exporting the current forward results
        self.export_fwddata = Button(self.fileControls,text="Export Forward Results",command=self.exportFwdResults).grid(row=1,column=0,sticky=N+E+W+S)
        # button for exporting the current reverse results
        self.export_revdata = Button(self.fileControls,text="Export Reverse Results",command=self.exportRevResults).grid(row=2,column=0,sticky=N+E+W+S)
        # button for importing reverse results from an external file 
        self.import_revdata = Button(self.fileControls,text="Import Reverse Results",command=self.importRevResults).grid(row=3,column=0,sticky=N+E+W+S)
        
        ## geometry manager
        # pack gui frame
        self.gui_frame.grid(row=0,column=1,rowspan=2,sticky=N+S+E+W)
        # pack controls
        self.solverControls.grid(row=0,column=0,sticky=N+E+W+S)
        self.fileControls.grid(row=1,column=0,sticky=N+E+W+S)
        
        # set minimum size to size on creation
        self.master.minsize(self.master.winfo_width(),self.master.winfo_height())

        ## make everything resizable
        # make the solver controls resizable
        num_cols,num_rows = self.solverControls.grid_size()
        for c in range(num_cols):
            self.solverControls.columnconfigure(c,weight=1)
        for r in range(num_rows):
            self.solverControls.rowconfigure(r,weight=1)
        # make the file controls resizable with window
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

        self.master.update_idletasks()

    # shortcut for getting the number of coils
    # simplifies syntax
    def numCoils(self):
        return self.path_gui.numCoils.get()

    # export forward solver results to a file
    def exportFwdResults(self):
        # if the forward problem has not been solved for the current coil arrangement
        # solve it for the first pair
        if not self.fwdSolved:
            self.forwardSolve(False)
            # the only exception thrown by the forwardSolve method is when there are no paths
            # as the exception cannot be caught at this level
            # the number of paths has to be checked here in order to exit properly
            if len(self.path_gui.paths)==0:
                return

        # open file dialog asking user for path to save forward results
        if not hdf5Support:
            fname = filedialog.asksaveasfilename(title="Export forward results",initialdir=os.path.dirname(self.datapath),defaultextension='.txt',filetypes=[("Text Data File (*.txt)",'*.txt'),("Log Data File (*.csv)","*.csv"),])
        else:
            fname = filedialog.asksaveasfilename(title="Export forward results",initialdir=os.path.dirname(self.datapath),defaultextension='.txt',filetypes=[("Text Data File (*.txt)",'*.txt'),("Log Data File (*.csv)","*.csv"),("HDF5 File (*.hdf5)","*.hdf5"),])
        # if a filepath was given, use numpy to solve it
        if fname is not '':
            if os.path.splitext(os.path.basename(fname))[1] != '.hdf5':
                # open file in write mode
                with open(fname,'w') as file:
                    # iterate over each coil pair
                    for i in range(self.ex_mat.shape[0]):
                        # get first coil pair
                        ex_line = self.ex_mat[i].ravel()
                        # solve for ith coil pair
                        f,_ = self.fwdSolver.solve(ex_line,self.mesh_bk['perm'])
                        # remove imaginary components and save data
                        np.savetxt(file,self.fwdData,delimiter=',',newline='\n')
            else:
                # open file in append mode
                with h5py.File(fname,'a') as file:
                    try:
                        # try and create forward results
                        hdset = file.create_dataset("Forward",(self.fwdData.shape[0],self.ex_mat.shape[0]),dtype=self.fwdData.dtype)
                    except RuntimeError:
                        del file["Forward"]
                        hdset = file.create_dataset("Forward",(self.fwdData.shape[0],self.ex_mat.shape[0]),dtype=self.fwdData.dtype)
                    # iterate over each coil pair
                    for i in range(1,self.ex_mat.shape[0]):
                        # get ith coil pair
                        ex_line = self.ex_mat[i].ravel()
                        # solve for ith coil pair
                        f,_ = self.fwdSolver.solve(ex_line,self.mesh_bk['perm'])
                        # save to file
                        hdset[:,i] = f

                    # ask the user if they want to export the mesh as well
                    if messagebox.askyesno(title="Export mesh?",message="Do you want to export the mesh object to the file as well?"):
                        try:
                            # if yes, create a group labelled mesh
                            hgroup = file.create_group("mesh")
                        except ValueError:
                            del file["mesh"]
                            hgroup = file.create_group("mesh")
                        # as h5py does not support dictionaries, each value within the mesh dict object needs to have it's own dataset
                        for k,v in self.mesh_bk.items():
                            hdset = hgroup.create_dataset(k,v.shape,dtype=v.dtype,data=v)

            messagebox.showinfo("Finished exporting",message=f"Finished exporting the forward results to {fname}")

    # create instance of ExportProgressWindow
    def createExportWindow(self):
        # create progress window
        r = Toplevel(self.master)
        self.exportProg = ExportProgressWindow(r)
        self.master.update_idletasks()

    # export reverse solver results to a file
    def exportRevResults(self):
        # check that some offline data has been read in
        if self.data.shape[0]==0:
            messagebox.showerror(title="No offline data!",message="No offline data has been read in!\n Please use the Read in File button to select the target data")
            return

        # check that paths were drawn on the canvas
        # if not then there's no excitation path to use in solver
        # inform user and exit
        if len(self.path_gui.paths) == 0:
            messagebox.showerror(title="No paths!",message="No paths on canvas!\nUse canvas to draw excitation order for imported data")
            return
        
        # open file dialog asking user for path to save forward results
        # if the system has hdf5 support, show hdf5 file option
        if not hdf5Support:
            fname = filedialog.asksaveasfilename(title="Export reverse results",initialdir=os.path.dirname(self.datapath),defaultextension='.txt',filetypes=[("Text Data File (*.txt)",'*.txt'),("Log Data File (*.csv)","*.csv"),])
        else:
            fname = filedialog.asksaveasfilename(title="Export reverse results",initialdir=os.path.dirname(self.datapath),defaultextension='.txt',filetypes=[("Text Data File (*.txt)",'*.txt'),("Log Data File (*.csv)","*.csv"),("HDF5 File (*.hdf5)","*.hdf5"),])
        # if a filepath is not to a hdf5 file
        # use numpy to export file
        if fname is not '':
            # if the selected file is not HDF5 and the data to be exported is at least 2D
            # warn the user that the exported data will only be the first frame as you can't store 3D informtion in a CSV or TXT file
            if (os.path.splitext(os.path.basename(fname))[1] != '.hdf5'):
                if self.method.get() == "ALL":
                    messagebox.showerror("Cannot Export Data",f"Cannot export results from all methods to TXT or CSV!")
                    return
                else:
                    showwarning(title="Export Warning",message="This file format does not support multidimensional data and\n therefore cannot store all the frames.\nOnly the first results frame will be stored")
                    np.savetxt(fname,self.revData,delimiter=',',newline='\n')
            # if hdf5, iterate over each of the frames adding them to the dataset
            else:
                # if the user previously chose to use all the solvers
                # ask them if they want to generate and export all the results or a specific one
                if self.method.get() == "ALL":
                    if not messagebox.askyesno("Query Export All","Are you sure you want to export the results for ALL methods?"):
                        # if the user does not want to export all results, ask which solver they want to use
                        temp = StringVar()
                        ComboboxDialog(self.master,self.solver_cbox.cget("values"),temp,"Choose a solver","Which solver do you want to use to generate results?\nClosing the dialog will exit the export process")
                        # if the user has selected something, set the export methods as the selected one
                        if temp.get() is not '':
                            export_methods = [temp.get()]
                        # if the user closed the dialog
                        # exit from export process
                        else:
                            return
                    # if the user wants to export the results for all the methods
                    # set the export methods as the options in the combobox except the ALL option assumed to be last option
                    else:
                        export_methods = self.solver_cbox.cget("values")[:-1]
                else:
                    export_methods = [self.method.get()]

                # create an instance of ExportProgressWindow
                self.createExportWindow()
    
                # open file in append mode
                # that way new datasets can be added to old files
                with h5py.File(fname,'a') as file:
                    # iterate over the solver types to be used
                    # index started at 1 to avoid div 0 errors when setting progress
                    for mi,m in enumerate(export_methods,start=1):
                        # set status message of progress window to current method
                        self.exportProg.setStatus(f"{m}")
                        # update global progress as progress through
                        # denominator is larger than the number of methods as there's the additional task of exporting
                        # mesh and coil information
                        self.exportProg.setGlobal(float(mi)/(len(export_methods)+1))
                        # set the method string
                        self.method.set(m)
                        # perform an initial solve rebuilding the class for the method
                        # sets the shape of each output data frame which is required to initialize the datasets
                        self.reverseSolve(showRes=False,forceRebuild=True)
                        # create hdf5 dataset with the name based off the type of solver
                        # set size of the datasset as the size of one frame by the number of rows in the dataset minus one as we're not comparing row 0 against row 0
                        # if the dataset already exists for the solver type, delete it and recreate it
                        # user consented to override data when they selected the file
                        try:
                            hdset = file.create_dataset(self.method.get(),(*self.revData.shape,self.data.shape[0]),dtype=self.data.dtype)
                        except:
                            del file[self.method.get()]
                            hdset = file.create_dataset(self.method.get(),(*self.revData.shape,self.data.shape[0]),dtype=self.data.dtype)

                        # greit solver
                        if self.method.get() == "GREIT":
                            # iterate over each of the rows in the dataset
                            for i in range(1,self.data.shape[0]):
                                # set local progress bar
                                self.exportProg.setLocal(float(i)*(self.data.shape[0]**-1.0))
                                # attempt to solve the dataset
                                try:
                                    self.reverseSolve(dataidx=i,showRes=False,forceRebuild=False)
                                    # add data to dataset
                                    hdset[...,i-1] = self.revData[:,:]
                                # if it failed to solve the data
                                # inform the user
                                # frame in dataset will be empty
                                except Exception as err:
                                    messagebox.showerror(title="GREIT Solving Error",message=str(err))
                        # for jacobian solver method
                        elif self.method.get() == "JAC":
                            for i in range(1,self.data.shape[0]):
                                # set local progress bar
                                self.exportProg.setLocal(float(i)/self.data.shape[0])
                                # run solver
                                try:
                                    self.reverseSolve(dataidx=i,showRes=False,forceRebuild=False)
                                    # add to dataset
                                    hdset[...,i-1] = self.revData[:]
                                except Exception:
                                    messagebox.showerror(title="JAC Solving Error", message=str(err))
                        # for back propogation method
                        elif self.method.get() == "BP":
                            for i in range(1,self.data.shape[0]):
                                # set local progress bar
                                self.exportProg.setLocal(float(i)/self.data.shape[0])
                                try:
                                    self.reverseSolve(dataidx=i,showRes=False,forceRebuild=False)
                                    hdset[...,i-1] = self.revData[:]
                                except Exception as err:
                                    messagebox.showerror(title="BP Solving Error",message=str(err))
                        # inverse radon solver
                        elif self.method.get() == "RADON":
                            for i in range(1,self.data.shape[0]):
                                # set local progress bar
                                self.exportProg.setLocal(float(i)/self.data.shape[0])
                                try:
                                    self.reverseSolve(dataidx=i,showRes=False,forceRebuild=False)
                                    hdset[...,-1] = self.revData[:,:]
                                except Exception as err:
                                    messagebox.showerror(title="RADON Solving Error",message=str(err))
                            
                    # export mesh
                    self.exportProg.setStatus("Exporting mesh")
                    self.exportProg.setLocal(0.0)
                    try:
                        hgroup = file.create_group("mesh")
                    except ValueError:
                        del file["mesh"]
                        hgroup = file.create_group("mesh")
                    # as h5py does not support dictionaries, each value within the mesh dict object needs to have it's own dataset
                    for k,v in self.mesh_bk.items():
                        hdset = hgroup.create_dataset(k,v.shape,dtype=v.dtype,data=v)
                    self.exportProg.setLocal(1.0)
                    
                    ## export information about the coils inc excitation order
                    ## information needed for plotting
                    # export number of coils
                    self.exportProg.setStatus("Exporting coil data")
                    self.exportProg.setLocal(0.0)
                    try:
                        hdset = file.create_dataset("nc",(1,),dtype='int',data=self.numCoils())
                    except RuntimeError:
                        del file["nc"]
                        hdset = file.create_dataset("nc",(1,),dtype='int',data=self.numCoils())
                    self.exportProg.setLocal(3.0**-1.0)
                    
                    # export excitation orde
                    try:
                        hdset = file.create_dataset("ex_mat",self.ex_mat.shape,dtype='int',data=self.ex_mat)
                    except RuntimeError:
                        del file["ex_mat"]
                        hdset = file.create_dataset("ex_mat",self.ex_mat.shape,dtype='int',data=self.ex_mat)
                    self.exportProg.setLocal(2.0*(3.0**-1.0))
                    # export coil position order
                    try:
                        hdset = file.create_dataset("ex_pos",self.ex_pos.shape,dtype='int',data=self.ex_pos)
                    except RuntimeError:
                        del file["ex_pos"]
                        hdset = file.create_dataset("ex_pos",self.ex_pos.shape,dtype='int',data=self.ex_pos)

                    # set progress bars to finished
                    # while not strictly necessary, it's put in for completeness
                    self.exportProg.setLocal(1.0)
                    self.exportProg.setGlobal(1.0)

            # get rid of progress window
            self.exportProg.master.destroy()
            # show message box informing user that the export process has finished
            messagebox.showinfo("Finished exporting",message=f"Finished exporting the reverse results to {fname}")

    # import reverse results and display in window
    # prompts user for which dataset to display
    def importRevResults(self):
        # ask user for target file
        # support flags affects whether hdf5 files are supported
        if not hdf5Support:
            fname = filedialog.askopenfilename(title="Export reverse results",initialdir=os.path.dirname(self.datapath),defaultextension='.txt',filetypes=[("Text Data File (*.txt)",'*.txt'),("Log Data File (*.csv)","*.csv"),])
        else:
            fname = filedialog.askopenfilename(title="Export reverse results",initialdir=os.path.dirname(self.datapath),defaultextension='.txt',filetypes=[("Text Data File (*.txt)",'*.txt'),("Log Data File (*.csv)","*.csv"),("HDF5 File (*.hdf5)","*.hdf5"),])

        # get supported solver types
        supp = self.solver_cbox.cget("values")
                
        # if the user chose a file
        if fname is not '':
            # if it is a hdf5 file
            if os.path.splitext(os.path.basename(fname))[1] == '.hdf5':
                # get the datasets in the file
                with h5py.File(fname,'r') as file:
                    fdsets = list(file.keys())
                # find overlap with supported solver types
                choices = [c for c in fdsets if c in supp]
                # construct variable to hold choice
                cdset = StringVar()
                # prompt user to state which dataset they want to inspect
                ComboboxDialog(self.master,choices,cdset,title="Choose data to inspect",label="Choose which dataset in the file to inspect")
                # if the user has not chosen something
                # return
                if cdset.get() is '':
                    return
                # the user chose something with the dialog
                # get the dataset
                else:
                    # import mesh object and coil information
                    with h5py.File(fname,'r') as file:
                        # get mesh
                        if 'mesh' in file:
                            # reconstruct dictionary from items in mesh group
                            self.mesh_bk = {k:v[()] for k,v in file['mesh'].items()}
                        # if not present and is requires for the solver type
                        # display error and exit
                        else:
                            if (cdset.get() == "JAC") or (cdset.get() == "BP"):
                                messagebox.showerror("Mesh missing",f"Mesh object missing from imported data!\nIt is required in order to plot results for {cdset.get()}")
                                return
                        
                        # get dataset
                        with h5py.File(fname,'r') as file:
                            self.data = file[cdset.get()][()]

                    # get coil information in file and update canvas
                    with h5py.File(fname,'r') as file:
                        # update number of coils
                        self.path_gui.numCoils.set(file['nc'][()][0])
                        # update excitation order
                        self.ex_mat = file['ex_mat'][()]
                        # call methods to redraw paths and coils
                        self.path_gui.redrawCoils()
                        self.path_gui.exciteOrder = self.ex_mat.tolist()
                        self.path_gui.redrawPaths()

                    # construct window to show results
                    # pass along the read in mesh object
                    r = Toplevel(self.master)
                    ReverseResultsWindow(r,None,self.data,displayType=cdset.get(),mesh=self.mesh_bk)
                    return
                
            # if the data is not a hdf5 file
            # ask user which solver was used to generate the data
            else:
                rtype = StringVar()
                ComboboxDialog(self.master,supp,rtype,title="Choose solver type",label="Choose which solver was used to generate the data")
                # check returned results
                if rtype.get() is '':
                    return
                else:
                    # attempt to convert file to array
                    # assumed to be comma separated values
                    # try except is to handle badly formatted or unsupported data
                    try:
                        dd = np.genfromtxt(rtype.get(),delimiter=',',dtype='float')
                    except ValueError as err:
                        messagebox.showerror("Failed to read in data",f"Failed to convert file to array!\nReason: {err}")
                        return

                    # save data locally
                    self.data = dd
                    # create window for results
                    ReverseResultsWindow(Toplevel(self.master),None,None,seld.data,displayType=rtype.get())
                    return
        
    # function for solving the forward problem of excitation data
    # pidx is the index of which coil pair in ex_mat to use for solving
    # forceRebuild flag forcefully rebuilds the forward solver and mesh object
    def forwardSolve(self,pidx=0,showRes=True,forceRebuild=False):
        if (len(self.path_gui.exciteOrder)==0) or (len(self.path_gui.paths)==0):
            messagebox.showerror(title="No Excitation Path Set!",message="No Excitation Path Set!\n Please Use Controls to Define the Coil Excitation Path")
            return
        else:
            # if fwdSolver has not been created 
            if (fwdSolver is None):
                # create solver class
                self.fwdSolver = Forward(self.mesh_bk,self.ex_pos)
                # build array objects
                self.mesh_bk,self.ex_pos = mesh.create(self.numCoils(),h0 =self.bkh)
                # setup excitation order
                self.ex_mat = np.array(self.path_gui.exciteOrder)
            # else if the forceRebuild flag is set, rebuild the mesh and forward solver
            elif forceRebuild:
                self.fwdSolver = Forward(self.mesh_bk,self.ex_pos)
                # build array objects
                self.mesh_bk,self.ex_pos = mesh.create(self.numCoils(),h0 =self.bkh)
                # setup excitation order
                self.ex_mat = np.array(self.path_gui.exciteOrder)
                
            # get first coil pair
            ex_line = self.ex_mat[int(pidx)].ravel()
            f,_ = self.fwdSolver.solve(ex_line,self.mesh_bk['perm'])
            # remove imaginary components and save data
            self.fwdData = np.real(f)

            #set flag to indicate that the forward solve was performed
            self.fwdSolved = True
            # if flag is set, display results
            if showRes:
                # create and display results in window
                r = Tk()
                self.fwdWin = ForwardResultsWindow(r,self.mesh_bk,self.ex_pos,self.fwdData,self.fwd,self.ex_mat)

    # function for generating simulated measurement data based off the mesh and excitation order
    # used for generating reference data, f0, for reverse solving
    # doesn't have a showRes flag as it can't be displayed in the current form for some solvers
    def forwardEITSolve(self,forceRebuild=False):
        if (len(self.path_gui.exciteOrder)==0) or (len(self.path_gui.paths)==0):
            messagebox.showerror(title="No Excitation Path Set!",message="No Excitation Path Set!\n Please Use Controls to Define the Coil Excitation Path")
            return
        else:
            # build array objects
            self.mesh_bk,self.ex_pos = mesh.create(self.numCoils(),h0 =self.bkh)
            # setup excitation order
            self.ex_mat = np.array(self.path_gui.exciteOrder)
            if (fwdSolver is None):
                # create solver class
                self.fwdSolver = Forward(self.mesh_bk,self.ex_pos)
            elif forceRebuild:
                self.fwdSolver = Forward(self.mesh_bk,self.ex_pos)
                
            # get simulated measurement data
            # save to f0 rather than fwdData
            self.f0 = self.fwdSolver.solve_eit(self.ex_mat,step=self.path_gui.step,perm=self.mesh_bk['perm'])
        
    # function for solving the reverse problem of predicting material conductivities from data
    # looks at what has changed between two readings set by refidx and dataidx respectively
    # refidx and dataidx sets the index for reference data and the data to compare against it in solver
    # updates internal revData, used in exporting the data or can be obtained later by another method
    # used as row indicies in self.data array
    # showRes flag controls whether an instance of ReverseResultsWindow is created for the solver
    # the window gives user a slider to control which rows from self.data are fed into the slider
    # See ReverseResultsWindow for more information
    # forceRebuild flag forces rebuild of solver class
    def reverseSolve(self,refidx=0,dataidx=1,showRes=True,forceRebuild=False):
        # clear reverse problem solved flag
        self.revSolved = False
        # check if a file has been read in
        if self.data.shape[0]==0:
            messagebox.showerror(title="No offline data!",message="No offline data has been read in!\n Please use the Read in File button to select the target data")
            return False

        # check that the number of excitations set in the canvas is not greater than the number of columns in the dataset (the number of excitations in the dataset)
        if len(self.path_gui.exciteOrder)>self.data.shape[1]:
            messagebox.showerror(title="Incorrect number of coils!",message="The number of excitations on the canvas is greater than the number of excitations in the file.\nAre you sure that you have correctly setup the excitiations?")
            return False

        # check that the user has drawn some paths
        if len(self.path_gui.paths) == 0:
            messagebox.showerror(title="No paths!",message="No paths on canvas!\nUse canvas to draw excitation order for imported data")
            return False
        
        if not self.fwdSolved:
            # create permeability mesh
            self.mesh_bk,self.ex_pos = mesh.create(self.numCoils(),h0 =self.bkh)
            # setup excitation order
            self.ex_mat = np.array(self.path_gui.exciteOrder)
            # create forward solving object
            self.fwd = Forward(self.mesh_bk,self.ex_pos)
            
        # set reference as first line of data
        self.f0 = self.data[refidx,:]
        # set new data to compare against f0
        self.f1 = self.data[dataidx,:]
            
        ## solve reverse problem using designated solver
        # GREIT
        if self.method.get() == "GREIT":
            # if a solver has not been built
            if (self.revSolver is None):
                self.revSolver = greit(self.mesh_bk,self.ex_pos,self.ex_mat,parser='std')
                # setup parameters ?
                self.revSolver.setup(p=0.50,lamb=0.05,n=self.numCoils())
            elif forceRebuild:
                self.revSolver = greit(self.mesh_bk,self.ex_pos,self.ex_mat,parser='std')
                # setup parameters ?
                self.revSolver.setup(p=0.50,lamb=0.05,n=self.numCoils())
                
            try:
                # solve
                # when showRes is set, this acts more like a test of the solver
                ds = self.revSolver.solve(self.f1,self.f0,normalize=False)
                # parse results
                # remove NaNs
                gx,gy,ds = self.revSolver.mask_value(ds,mask_value=np.NAN)
                # get data
                self.revData = np.real(ds)
                # show results if flag is set
                if showRes:
                    # create window for results
                    r = Tk()
                    # generate window passing along required objects inc solver and results
                    # solver provides the mesh etc.
                    self.revWin = ReverseResultsWindow(r,self.revSolver,self.data)
            except Exception as err:
                messagebox.showerror(title="GREIT Solving Error",message=str(err))
        # for jacobian solver method
        elif self.method.get() == "JAC":
            # build solver if it doesn't exist
            if (self.revSolver is None):
                self.revSolver = jacobian(self.mesh_bk,self.ex_pos,self.ex_mat,parser='std')
                # setup solver
                self.revSolver.setup(p=0.50,lamb=0.01,method='kotre')
            elif forceRebuild:
                self.revSolver = jacobian(self.mesh_bk,self.ex_pos,self.ex_mat,parser='std')
                # setup solver
                self.revSolver.setup(p=0.50,lamb=0.01,method='kotre')
            # run solver
            try:
                # run solver
                ds = self.revSolver.solve(self.f1,self.f0,normalize=False)
                # ?
                self.revData = sim2pts(self.mesh_bk['node'],self.mesh_bk['element'],np.real(ds))
                # create window for results if flag is set
                if showRes:
                    r = Tk()
                    self.revWin = ReverseResultsWindow(r,self.revSolver,self.data)
            except Exception as err:
                messagebox.showerror(title="JAC Solving Error", message=str(err))
        # for back propogation method
        elif self.method.get() == "BP":
            if (self.revSolver is None):
                self.revSolver = bp(self.mesh_bk,self.ex_pos,self.ex_mat,parser='std')
                self.revSolver.setup(weight='none')
            elif forceRebuild:
                self.revSolver = bp(self.mesh_bk,self.ex_pos,self.ex_mat,parser='std')
                self.revSolver.setup(weight='none')
            try:
                # run solver
                # don't know what weight is for
                ds = self.revSolver.solve(self.f1,self.f0)
                self.revData = ds
                if showRes:
                    # create window for results
                    r = Tk()
                    self.revWin = ReverseResultsWindow(r,self.revSolver,self.data)
            except Exception as err:
                messagebox.showerror(title="BP Solving Error",message=str(err))
        # inverse radon transform
        # see ReadonReconstruction method
        elif self.method.get() == "RADON":
            if (self.revSolver is None):
                # create the solver class initializing it with the number of coils
                self.revSolver = RadonReconstruction(self.numCoils(),view.path_gui.dist)
            elif forceRebuild:
                # create the solver class initializing it with the number of coils
                self.revSolver = RadonReconstruction(self.numCoils(),view.path_gui.dist)
                
            try:
                # reconstruct the image using Radon transform and measurements
                img = self.revSolver.eit_reconstruction(self.f1)
                self.revData = img
                # send the results to the results window to plot
                if showRes:
                    r = Tk()
                    self.revWin = ReverseResultsWindow(r,self.revSolver,self.data)
            except Exception as err:
                messagebox.showerror(title="RADON Solving Error",message=str(err))
        elif self.method.get() == "ALL":
            # get all the methods except all which is assumed to be at the end
            for m in self.solver_cbox.cget("values")[:-1]:
                # set the method
                self.method.set(m)
                # run reverse solve method with updated method
                # the respective path will handle the method
                self.reverseSolve(showRes,showRes=showRes)
                # after each method, clear reverse solver so it's rebuilt each time
                self.reverseSolver = None
            self.method.set("ALL")
                
        self.revSolved = True
        return True
        
    # function for reading in measurement data files
    def readDataFile(self):
        # ask user to select and open a file
        fname = filedialog.askopenfilename(initialfile=self.datapath,filetypes=(["Text Data File",'*.txt'],["Log Data File","*.csv"]))
        # if user selected a file
        if fname is not None and fname is not '':
            # update local copy of filename
            self.datapath = fname
            # read in data
            with open(self.datapath,'r') as dataFile:
                # read in all the lines in a file
                lines = dataFile.readlines()
            # if it's a text file, attempt to convert each line to a readable form
            # then convert result to a numpy dataset
            if '.txt' in self.datapath:
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
            elif '.csv' in self.datapath:
                try:
                    temp = np.genfromtxt(self.datapath,dtype='float',delimiter=',')
                    self.data = temp
                except ValueError as e:
                    messagebox.showerror(title="Failed to parse csv data file!",message=f"Failed to parse target data file\n{fname}\n\n{e}")
                    return

            # clear forward and reverse solved flag to indicate that the forward problem has not been solved yet with the current data
            self.fwdSolved = False
            self.revSolved = False
            # clear forward and reverse solvers
            # rebuilt on calls to forwardSolve and reverseSolve
            self.fwdSolver = None
            self.revSolver = None
                
    # method for parsing a line of data from an appropriately formatted data text file
    @staticmethod
    def parseLine(line):
        # attempt to split the data from the semicolon
        try:
            _, data = line.split(":", 1)
        except ValueError:
            return None
        # split the comma separated values removing any special characters
        # convert them to floats
        # return as a numpy array
        items = []
        for item in data.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                items.append(float(item))
            except ValueError:
                return None
        return np.array(items)

if __name__ == "__main__":
    root = Tk()
    view = PathPatternSolverGUI(root)
    root.mainloop()
        
