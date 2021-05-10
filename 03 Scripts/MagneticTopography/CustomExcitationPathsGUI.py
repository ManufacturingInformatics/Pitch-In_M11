from tkinter import Tk,Canvas,Label,Button,Frame,Scrollbar,filedialog,N,S,E,W,CENTER,colorchooser,IntVar,StringVar,filedialog,font,LabelFrame
from tkinter.simpledialog import askinteger,Dialog
from tkinter.messagebox import askyesno,showerror
from tkinter.ttk import Combobox
import numpy as np
from PIL import ImageGrab
import os
from pyeit.eit.utils import eit_scan_lines

class ComboboxWindow:
    def __init__(self,master,opts,chosen,updatefn):
        self.master = master
        # save reference to chosen option
        self.chosenPattern = chosen
        # save update function
        self.updatefn = updatefn
        # create a combobox showing the options
        self.optsLabel = Label(self.master,text="Patterns",font="Arial 10 bold")
        self.optsBox = Combobox(self.master,state="readonly",values=opts)
        self.optsBox.current(0)
        # create a button to confirm choice and exit
        self.exitButton = Button(self.master,text="Confirm",command=self.confirmChoice)

        self.master.protocol("WM_DELETE_WINDOW",self.confirmChoice)
        
        # as it's a simple window, pack is used instead of grid
        self.optsLabel.pack()
        self.optsBox.pack()
        self.exitButton.pack()

    def confirmChoice(self):
        print("choice confirmed and set")
        if self.optsBox.get() is None or self.optsBox.get() == '':
            self.master.destroy()
            return
        else:
            self.chosenPattern.set(self.optsBox.get())
            print("running update")
            self.updatefn()
            print("destroying window")
            self.master.destroy()

class ComboboxDialog(Dialog):
    def __init__(self,parent,opts,chosen,title="Path Pattern Options",label="Patterns"):
        # pass along the variable to update represneting chosen pattern
        self.chosenPattern = chosen
        # pass along the available options
        self.opts = opts
        self.label = label
        # initialize dialog
        super().__init__(parent,title=title)

    # create contents of the dialog
    def body(self,master):
        # label for combobix
        self.optsLabel = Label(master,text=self.label,font="Arial 10 bold")
        self.optsLabel.grid(row=0,column=0)
        # combobox representing the different patterns available
        self.optsBox = Combobox(master,state="readonly",values=self.opts)
        self.optsBox.grid(row=1,column=0)
        # initialize combobox to the first element
        self.optsBox.current(0)
        # return combobox as the initial widget of focus
        return self.optsBox

    # method for handling whenn the OK button is pressed
    def apply(self):
        # update variable value with the selected combobox value
        self.chosenPattern.set(self.optsBox.get())

## scrollable frame class
class ScrollableFrame(Frame):
    def __init__(self,master,**kwargs):
        super().__init__(master,**kwargs)
        self.master = master

        # create a canvas
        self.canvas = Canvas(self)
        # add scrollbars to containing frame
        self.yscrollbar = Scrollbar(self,orient="vertical",command=self.canvas.yview)
        # configure canvas to move with scrollbars
        self.canvas.configure(yscrollcommand=self.yscrollbar.set)

        # add to be scrollable frame to canvas
        self.scroll_frame = Frame(self.canvas)
        # add code to change what's being show in the canvas according to the scrollbar
        self.scroll_frame.bind("<Configure>",lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        # draw frame inside canvas
        # put corner in top left of canvas
        self.canvas.create_window((0,0),window=self.scroll_frame,anchor="nw")
        
        ## pack it
        # place canvas in the centre of self
        self.canvas.grid(row=0,column=0,sticky=N+S+E+W)
        # place scrollbar to the right of canvas
        self.yscrollbar.grid(row=0,column=1,sticky=N+S)

class ScrollableLabelFrame(LabelFrame):
    def __init__(self,master,**config):
        super().__init__(master,**config)
        self.master = master

        # create a canvas
        self.canvas = Canvas(self)
        # add scrollbars to containing frame
        self.yscrollbar = Scrollbar(self,orient="vertical",command=self.canvas.yview)
        # configure canvas to move with scrollbars
        self.canvas.configure(yscrollcommand=self.yscrollbar.set)

        # add to be scrollable frame to canvas
        self.scroll_frame = Frame(self.canvas)
        # add code to change what's being show in the canvas according to the scrollbar
        self.scroll_frame.bind("<Configure>",lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        # draw frame inside canvas
        # put corner in top left of canvas
        self.canvas.create_window((0,0),window=self.scroll_frame,anchor="nw")
        
        ## pack it
        # place canvas in the centre of self
        self.canvas.grid(row=0,column=0,sticky=N+S+E+W)
        # place scrollbar to the right of canvas
        self.yscrollbar.grid(row=0,column=1,sticky=N+S)
        
# window for displaying the excitation order
class PathOrderWindow:
    def __init__(self,master,paths):
        self.master = master
        self.master.title("Order of Excitation")
        ## creating a scrollable frame
        self.frame = ScrollableFrame(self.master)
        # add entries to frame
        if len(paths)>0:
            # each coil pair is a label with a string
            for pi,pp in enumerate(paths):
                Label(self.frame.scroll_frame,text=str(pp)).grid(row=pi,column=0,sticky=N+E+W)
        # if there are no paths in the list
        # display a message instead
        else:
            Label(self.frame.scroll_frame,text="None Selected!").grid(row=0,column=0,sticky=N+E+W)
        #  pack the scrollable frame
        self.frame.grid(row=0,column=0,sticky=N+S+E+W)

# window for displaying the different font and size options
class ChangeFontWindow:
    def __init__(self,master,default,winRef):
        self.master = master
        self.winRef = winRef
        self.master.title("Choose new Font and Size")
        # save current font dict
        self.currFont = default
        #self.newFont.trace('w',self.onFamilyChange)
        # label for combobox
        self.fontOptsLabel = Label(self.master,text="Font Options",font="Arial 10 bold")
        # create combobox to show font options
        self.fontOptions = Combobox(self.master,state="readonly",values=font.families(root=self.master))
        
        # move displayed value to the current font
        self.fontOptions.current(self.fontOptions.cget("values").index(self.currFont["family"]))

        ## new size 
        self.sizeOptsLabel = Label(self.master,text="Size Options",font="Arial 10 bold")
        # create combobox to show font options
        self.sizeOptions = Combobox(self.master,values=list(range(1,30,1)))
        # move displayed value to the current font
        self.sizeOptions.current(self.sizeOptions.cget("values").index(str(self.currFont["size"])))

        # arrangement
        self.fontOptsLabel.grid(row=0,column=0)
        self.fontOptions.grid(row=1,column=0)
        self.sizeOptsLabel.grid(row=0,column=1)
        self.sizeOptions.grid(row=1,column=1)

        # attach binding handlers for when a new item is selected
        self.fontOptions.bind("<<ComboboxSelected>>",self.onFamilyChange)
        self.sizeOptions.bind("<<ComboboxSelected>>",self.onSizeChange)

        # on exit force update of fonts in ExcitationOrderGUI
        self.master.protocol("WM_DELETE_WINDOW",self.forceUpdateLabels)

    def forceUpdateLabels(self):
        print("Calling forced update")
        self.winRef.updateTextLabelsFont()
        self.master.destroy()

    def onFamilyChange(self,event):
        self.currFont['family'] = self.fontOptions.get()

    def onSizeChange(self,event):
        self.currFont['size'] = int(self.sizeOptions.get())
        
class ExcitationOrderGUI:
    # data structure to make the syntax of managing drawn paths easier
    # defined as an inner class as it isn't going to be used elsewhere
    class CanvasObj:
        def __init__(self,oid=0,x0=0,x1=0,y0=0,y1=0):
            self.x = [x0,x1]
            self.y = [y0,y1]
            self.oid = oid

        def xy(self):
            # create a list of coordinate pair tuples (x,y)
            t = [(x,y) for x,y in zip(self.x,self.y)]
            # unpack the tuples so they are next to each other (x0,y0,x1,y1...)
            # canvas coords method accepts this
            return [a for tup in t for a in tup]
            
    def __init__(self,master,nc=32):
        self.master = master
        # set title of the window
        #self.master.title("Custom Coil Excitation")
        
        # number of coils
        self.numCoils = IntVar(value=nc)

        # flag to use the first element in nearest coil list or find closest using distance
        self.useFirstObject = True
        
        # radius of the electrodes drawn
        self.coilRadius = 10
        # resolution of the lines
        self.lineRes = 20
        # resolution of the circle
        self.circleRes = 50
        ## line thickness
        self.circleThick = 5
        self.pathThick = 1
        # padding for the canvas
        self.padx = self.pady = 10
        # size of the canvas
        self.cwidth = 400 + self.padx
        self.cheight = 400 + self.pady
        # centre of the boundary circle
        self.circleCentre = [(self.cwidth/2),(self.cheight/2)]
        # radius of the boundary circle
        self.circleRadius = (self.cheight/2)-self.coilRadius-self.padx
        
        # generated excitation path
        self.exciteOrder = []
        # distance between coils, to be set later based on generated pattern
        # remains 0 for custom patterns as it cannot be determined if it's regular or not
        self.dist = IntVar(value=0)
        # distance between each coil in the sequence
        # e.g. step = 2, dist=1 : [0,1] ... [2,0] etc.
        self.step = 1

        ## colors for the canvas items
        self.coilColor = "red"
        self.circleColor = "black"
        self.pathColor = "blue"
        self.textColor = "white"

        ## supported generatable patterns
        self.patterns = ("opposite","opposite v2","adjacent","adjacent+2","all","custom")

        # canvas for drawing
        self.canvas = Canvas(self.master,width=self.cwidth,height=self.cheight,bg="white")
        # collection to hold the drawn path data
        # so it can be deleted and more easily managed
        self.paths = []
        # collection to hold coil ids
        self.coilIds = []
        # collectiion to gold the coil text labels
        self.coilLabels = []
        ## click and drag handlers
        # left cliick to start a path
        self.canvas.bind("<Button-1>",self.leftClick)
        # drag to draw the line
        self.canvas.bind("<B1-Motion>",self.drawPath)
        # button to finish the line 
        self.canvas.bind("<ButtonRelease-1>",self.leftRelease)
        # resize handler
        self.canvas.bind("<Configure>",self.onResize)
        
        # draw boundary circle
        # id and data of the circle boundary
        self.boundaryCircleId = self.canvas.create_oval(self.circleCentre[0]-self.circleRadius,self.circleCentre[1]-self.circleRadius,self.circleCentre[0]+self.circleRadius,self.circleCentre[1]+self.circleRadius,tags="boundary",width=self.circleThick,outline=self.circleColor)
        # draw coils
        self.redrawCoils()
        
        ## control panel
        self.controlPanel = LabelFrame(self.master,text="Canvas Controls",font="Arial 10 bold italic")
        # button to save the drawing as an image
        self.saveCanvasButton = Button(self.controlPanel,text="Save Canvas",command=self.saveCanvasAsPNG).grid(row=0,column=0,sticky=N+E+W+S)
        # button to change the color of the objects
        self.changeLineColButton = Button(self.controlPanel,text="Change Line Color",command=self.changeLineColor).grid(row=1,column=0,sticky=N+E+W+S)
        self.changePathWidthButton = Button(self.controlPanel,text="Change Line Width",command=self.changePathWidth).grid(row=2,column=0,sticky=N+E+W+S)
        self.changeCircleColButton = Button(self.controlPanel,text="Change Circle Color",command=self.changeCircleColor).grid(row=3,column=0,sticky=N+E+W+S)
        self.changeCoilColButton = Button(self.controlPanel,text="Change Coil Color",command=self.changeCoilColor).grid(row=4,column=0,sticky=N+E+W+S)
        self.changeTextColButton = Button(self.controlPanel,text="Change Text Color",command=self.changeTextColor).grid(row=5,column=0,sticky=N+E+W+S)
        self.changeTextOptButton = Button(self.controlPanel,text="Change Text Font",command=self.showChangeFont).grid(row=6,column=0,sticky=N+E+W+S)
        # button to clear the paths
        self.clearPathsButton = Button(self.controlPanel,text="Clear Paths",command=self.clearPaths).grid(row=7,column=0,sticky=N+E+W+S)
        # button to show current excite path
        self.showPathsButton = Button(self.controlPanel,text="Show Excite Order",command=self.showPathsWindow).grid(row=8,column=0,sticky=N+E+W+S)
        # export it to a text file so it can be read in by a different program
        self.exportPathsButton = Button(self.controlPanel,text="Export Excite Order",command=self.exportExciteOrder).grid(row=9,column=0,sticky=N+E+W+S)
        # import excite order and draw
        self.importPathsButton = Button(self.controlPanel,text="Import Excite Order",command=self.importExciteOrder).grid(row=10,column=0,sticky=N+E+W+S)
        # generate a pattern rather than draw one
        self.drawPatternButton = Button(self.controlPanel,text="Draw Pattern",command=self.showPatternOptions).grid(row=11,column=0,sticky=N+E+W+S)
        # change the number of coils
        self.changeCoilsButton = Button(self.controlPanel,text="Change Coils",command=self.changeNumCoils).grid(row=12,column=0,sticky=N+E+W+S)
        
        ## geometry manager
        self.canvas.grid(row=0,column=0,sticky=N+S+E+W)
        # add control panel
        self.controlPanel.grid(row=0,column=1,sticky=N+S+E+W)

        # set minimum size to size on creation
        self.master.update()
        if hasattr(self.master,'minsize'):
            self.master.minsize(self.master.winfo_width(),self.master.winfo_height())
        
        ## make everything resizables
        # make controls resizable
        num_cols,num_rows = self.controlPanel.grid_size()
        for c in range(num_cols):
            self.controlPanel.columnconfigure(c,weight=1)
        for r in range(num_rows):
            self.controlPanel.rowconfigure(r,weight=1)
        # make the main 
        num_cols,num_rows = master.grid_size()
        for c in range(num_cols):
            master.columnconfigure(c,weight=1)
        for r in range(num_rows):
            master.rowconfigure(r,weight=1)

    # import a previously exported excitation order file
    # comma separated integer pairs on each row
    def importExciteOrder(self):
        # if there are currently paths drawn on the canvas, warn user that they will be cleared
        if len(self.paths)>0:
            if not askyesno(title="Found paths",message="There are paths on the canvas!\nImporting an excitation file will result in the canvas being cleared\nAre you sure you want to continue?"):
                return
        # prompt user for location of file
        efile = filedialog.askopenfile(mode='r',initialdir=os.getcwd())
        # if a file was selected
        if efile is not None:
            # attempt to read in and parse file
            try:
                temp = np.genfromtxt(efile,dtype=np.int32,delimiter=',')
            except:
                showerror(title="Failed to read in file",message=f"Failed to read in excitation file\n{efile.name}")
                return
            # update number of coils
            self.numCoils.set(temp.max()+1)
            # redraw coils for new number
            # clears paths in the process
            self.redrawCoils()
            # update new excitation order
            # it is clared as part of redrawCoils
            self.exciteOrder = temp.tolist()
            # iterate over new excitation order drawing paths between nodes 
            for c0,c1 in self.exciteOrder:
                # create a path joining one coil to another
                self.paths.append(self.CanvasObj(self.canvas.create_line(self.coilIds[c0].x[0],self.coilIds[c0].y[0], # start
                                                            self.coilIds[c1].x[0],self.coilIds[c1].y[0], # end
                                                            fill=self.pathColor, # color
                                                            splinesteps=self.lineRes, # number of points that make up the line
                                                            tags="path", # tag to be used when searching for it
                                                            smooth=True, # smooth the line for irregular points
                                                            width=self.pathThick, # set thickness
                                                            activewidth=1.5*self.pathThick) # set increased thickness when highlighted
                                                            ,x0=self.coilIds[c0].x[0],y0=self.coilIds[c0].y[0],
                                                             x1=self.coilIds[c1].x[0],y1=self.coilIds[c1].y[0])) # create data structure
            # update canvas to make sure the results are displayed
            self.canvas.update()
            
    # handler for allowing the user to change the number of coils
    def changeNumCoils(self):
        # if there are paths in the canvas, prompt user to confirm they want to change the coils
        # redrawCoils method clears the canvas
        if len(self.paths)>0:
            if not askyesno(title="Are you sure you want to redraw?",message="There are paths present on the canvas.\nChanging the number of coils will clear the canvas.\nAre you sure you want to change the number of coils?"):
                return
        # prompt user for new number of coils
        temp = askinteger(title="Choose the new number of coils",prompt="Choose the new number of coils")
        # if the user selected cancel, return
        if temp is None:
            return
        # if the number of coils is 0, show error and return
        elif temp == 0:
            showerror(title="Invalid Number of Coils",message="Invalid number of coils!\nNumber of coils must be positive")
            return
        # if the number of coils is odd, show error and return
        elif (temp%2)>0:
            showerror(title="Invalid Number of Coils",message="Invalid Number of coils!\nNumber of coils must be event")
            return
        else:
            self.numCoils.set(temp)
            self.redrawCoils()
            

    # handler for resizing canvas and conents with respect to window size
    def onResize(self,event):
        wscale = float(event.width)/self.cwidth
        hscale = float(event.height)/self.cheight
        self.cwidth = event.width
        self.cheight = event.height
        # adjust stored coil radius
        # use the largest scaling factor
        self.coilRadius *= max(wscale,hscale)
        # adjust stored boundary circle radius
        self.circleRadius *= max(wscale,hscale)
        #resize canvas
        #self.canvas.config(width=event.width,height=event.height)
        # resize canvas contents
        self.canvas.scale("all",0,0,wscale,hscale)
        # get the location of the coils
        for c in self.canvas.find_withtag("coil"):
            # coordinates are given as as a bounding box marking the UL and BR corners
            coilCoords = self.canvas.coords(c)
            # find the index of the coil in local list matching the target id
            for ci in self.coilIds:
                if ci.oid == c:
                    # update the stored centre of the circle
                    ci.x[0] = coilCoords[0] + (coilCoords[2]-coilCoords[0])*0.5
                    ci.y[0] = coilCoords[1] + (coilCoords[3]-coilCoords[1])*0.5

    # create a window to display the current excitation order
    def showPatternOptions(self):
        # create variable to get the chosen option
        self.chosenPattern = StringVar()
        ## using dialog class
        # temporary variable to hold result
        # allows us to check if an option has been set
        temp = StringVar()
        # create dialog allowing the use to choose the generated pattern
        ComboboxDialog(self.master,self.patterns,temp)
        # if an option was selected
        # update the chosen pattern and run generate coil pattern
        if temp.get() is not '':
            self.chosenPattern.set(temp.get())
            self.generateCoilPattern()
        
    # static method for generating coil pair indexes with a certain offset between them
    # for example an opposiite configuration would set the offset to be half the number of coils
    @staticmethod
    def genCPairs(numCoils,offset=1):
        for i in range(numCoils):
            yield i, (i+offset)%numCoils

    # generate and display a common coil excitation patterns
    def generateCoilPattern(self):
        print("Generating pattern",self.chosenPattern.get())
        # remove all paths
        self.clearPaths()
        # clear excitation order
        self.exciteOrder.clear()
        ## process users desired configuration pattern
        ## set as checking keywords in case alternative terms/nicknames are to be added
        # if user wants an opposite configuration to be generated
        if self.chosenPattern.get() in ("opposite",):
            self.dist.set(self.numCoils.get()//2)
        elif self.chosenPattern.get() in ("opposite v2",):
            self.dist.set((self.numCoils.get()//2) -1)
        # if the user wants an adjacent pattern to be generated
        elif self.chosenPattern.get() in ("adjacent",):
            self.dist.set(1)
        # if the user wants an adjacent pattern with an offset
        # set the distance to the number after the plus
        # allows for expansion later on or by others
        elif self.chosenPattern.get() in ("adjacent+2",):
            self.dist.set(int(self.chosenPattern.get().split('+')[1]))
        # special case for generating all possible combinations of numbers
        elif self.chosenPattern.get() in ("all",):
            from itertools import combinations
            self.dist.set(1)
            # generate all possible combinations of coil pairs
            # spun off as separate case as most would involve combinations of pairs with offset
            print("num coils: ",self.numCoils.get())
            for c0,c1 in combinations(range(self.numCoils.get()),2):
                # draw paths between coils
                # create a path joining one coil to another
                self.paths.append(self.CanvasObj(self.canvas.create_line(self.coilIds[c0].x[0],self.coilIds[c0].y[0], # start
                                                            self.coilIds[c1].x[0],self.coilIds[c1].y[0], # end
                                                            fill=self.pathColor, # color
                                                            splinesteps=self.lineRes, # number of points that make up the line
                                                            tags="path", # tag to be used when searching for it
                                                            smooth=True,
                                                            width=self.pathThick,
                                                            activewidth=1.5*self.pathThick) # smooth the line for irregular points
                                                            ,x0=self.coilIds[c0].x[0],y0=self.coilIds[c0].y[0],
                                                             x1=self.coilIds[c1].x[0],y1=self.coilIds[c1].y[0])) # create data structure
                # update excitation order
                self.exciteOrder.append([c0,c1])
            return
        elif self.chosenPattern.get() in ("custom",):
            tempInt = askinteger("Enter distance between coils","Enter distance between coils in custom pattern (uses eit_scan_lines).\n (Step is fixed to 1 at the moment)")
            if tempInt is None:
                return
            else:
                self.dist.set(tempInt)
            # use utility function to generate order
            self.exciteOrder = eit_scan_lines(self.numCoils.get(),self.dist.get()).tolist()
            for c0,c1 in self.exciteOrder:
                # create a path joining one coil to another
                self.paths.append(self.CanvasObj(self.canvas.create_line(self.coilIds[c0].x[0],self.coilIds[c0].y[0], # start
                                                            self.coilIds[c1].x[0],self.coilIds[c1].y[0], # end
                                                            fill=self.pathColor, # color
                                                            splinesteps=self.lineRes, # number of points that make up the line
                                                            tags="path", # tag to be used when searching for it
                                                            smooth=True,
                                                            width=self.pathThick,
                                                            activewidth=1.5*self.pathThick) # smooth the line for irregular points
                                                            ,x0=self.coilIds[c0].x[0],y0=self.coilIds[c0].y[0],
                                                             x1=self.coilIds[c1].x[0],y1=self.coilIds[c1].y[0])) # create data structure
            return
        # if the user tries to use an unsupported pattern
        # raise exception
        else:
            raise ValueError(f"Unsupported pattern name {self.chosenPattern.get()}! Supported filenames are {self.patterns}")
            return
        
        ## generate paths for configurations that can use the genCPairs function
        # iterate over coils
        for c0,c1 in self.genCPairs(self.numCoils.get(),offset=self.dist.get()):
            # create a path joining one coil to another
            self.paths.append(self.CanvasObj(self.canvas.create_line(self.coilIds[c0].x[0],self.coilIds[c0].y[0], # start
                                                        self.coilIds[c1].x[0],self.coilIds[c1].y[0], # end
                                                        fill=self.pathColor, # color
                                                        splinesteps=self.lineRes, # number of points that make up the line
                                                        tags="path", # tag to be used when searching for it
                                                        smooth=True,
                                                        width=self.pathThick,
                                                        activewidth=1.5*self.pathThick) # smooth the line for irregular points
                                                        ,x0=self.coilIds[c0].x[0],y0=self.coilIds[c0].y[0],
                                                         x1=self.coilIds[c1].x[0],y1=self.coilIds[c1].y[0])) # create data structure
            # add to excitation order
            self.exciteOrder.append([c0,c1])

    # create window to allow user to change the font of the coil labels
    def showChangeFont(self):
        t = Tk()
        # get current font
        self.currFont = font.nametofont(self.canvas.itemconfig(self.canvas.find_withtag("label")[0])["font"][-1])
        # create window passing along current settings
        # settings are adjusted inside the program
        self.fontChangeWin = ChangeFontWindow(t,self.currFont,self)
        # print return to show new settings
        print(self.currFont)

    # function to iterate over labels and update the font 
    def updateTextLabelsFont(self):
        # get the ids of all the labels
        for ll in self.canvas.find_withtag("label"):
            # update the font, size etc with the settings 
            self.canvas.itemconfig(ll,font=self.currFont)

    # function for exporting the currently drawn excitation order to a file
    # writes them to a file as comma separated pairs
    def exportExciteOrder(self):
        from datetime import datetime
        # open save dialog and initialize filename as formatted date time
        expfile = filedialog.asksaveasfile(mode='w',initialfile=datetime.now().strftime("%Y-%m-%d-excitepath"),defaultextension='.txt',filetypes=[("Text File","*.txt")])
        # if a location was selected
        if expfile is not None:
            # iterate over the files
            for p0,p1 in self.exciteOrder:
                expfile.write(f"{p0},{p1}\n")
            expfile.close()

    # function for iterating over labels and chaning the text color
    def changeTextColor(self):
        self.textColor = colorchooser.askcolor(color=self.textColor)[1]
        print(self.textColor)
        for l in self.canvas.find_withtag("label"):
            self.canvas.itemconfig(l,fill=self.textColor)

    # function for choosing and updating the width of the paths
    def changePathWidth(self):
        temp = askinteger("Enter new width","Enter the new line thickness")
        # check for zero and negative numbers
        if temp >=0:
            self.pathThick = temp
        # iterate over paths updating line thickness
        for p in self.canvas.find_withtag("path"):
            self.canvas.itemconfig(p,width=self.pathThick)

    # crate and display window that shows path order
    def showPathsWindow(self):
        t = Tk()
        self.pathsWin = PathOrderWindow(t,self.exciteOrder)

    # clear all tagged paths in the canvas and excitation order
    def clearPaths(self):
        self.canvas.delete("path")
        self.paths.clear()
        # clear stored exitation path
        self.exciteOrder = []

    # draw the circles representing the coils
    # wrapped as a function so it can be used if the number of coils changes
    def redrawCoils(self):
        # delete all coils
        self.canvas.delete("coil")
        # remove coil labels
        self.canvas.delete("label")
        # remove paths
        self.clearPaths()
            
        # clear local lists
        self.coilIds.clear()
        self.coilLabels.clear()
        
        # iterate along the boundary of the circle drawing circles representing coils
        startposx = self.circleCentre[0]
        startposy = self.circleCentre[1]
        # iterate around the circle drawing the coil icons
        for di,delta in enumerate(np.linspace(0.0,2.0*np.pi,self.numCoils.get()+1)):
            # calculate offset from center
            x = self.circleRadius * np.cos(delta)
            y = self.circleRadius * np.sin(delta)
            #print(di,delta*(180.0/np.pi),x,y)
            # create and store the coils as CanvasObj
            # store the circle's centres as the first coordinates of the x and y members
            # the second coordinates are 0's
            self.coilIds.append(self.CanvasObj(self.canvas.create_oval(startposx+x-self.coilRadius,startposy+y-self.coilRadius,startposx+x+self.coilRadius,startposy+y+self.coilRadius,tags="coil",width=1,fill=self.coilColor),
                                               x0=startposx+x,y0=startposy+y))
            self.coilLabels.append(self.canvas.create_text(startposx+x,startposy+y,anchor=CENTER,text=str(di-1),tags="label",fill=self.textColor))
        # force update of canvas
        self.canvas.update()

    # redraw the excitation paths between coils
    # used if exciteOrder are changed by an outside method and need to be updated/redrawn
    def redrawPaths(self):
        # delete all paths
        self.canvas.delete("path")
        # re add paths
        for c0,c1 in self.exciteOrder:
            # create a path joining one coil to another
            self.paths.append(self.CanvasObj(self.canvas.create_line(self.coilIds[c0].x[0],self.coilIds[c0].y[0], # start
                                                        self.coilIds[c1].x[0],self.coilIds[c1].y[0], # end
                                                        fill=self.pathColor, # color
                                                        splinesteps=self.lineRes, # number of points that make up the line
                                                        tags="path", # tag to be used when searching for it
                                                        smooth=True, # smooth the line for irregular points
                                                        width=self.pathThick, # set thickness
                                                        activewidth=1.5*self.pathThick) # set increased thickness when highlighted
                                                        ,x0=self.coilIds[c0].x[0],y0=self.coilIds[c0].y[0],
                                                         x1=self.coilIds[c1].x[0],y1=self.coilIds[c1].y[0])) # create data structure

    # open window allowing user to change the color of the paths
    def changeLineColor(self):
        self.pathColor = colorchooser.askcolor(color=self.pathColor)[1]
        for p in self.paths:
            self.canvas.itemconfig(p.oid,fill=self.pathColor)

    # change the color representing coils
    def changeCoilColor(self):
        self.coilColor = colorchooser.askcolor(color=self.coilColor)[1]

    # change the color representing the circle
    def changeCircleColor(self):
        self.circleColor = colorchooser.askcolor(color=self.circleColor)[1]
        self.canvas.itemconfig(self.boundaryCircleId,outline=self.circleColor)

    # left click handler for the canvas
    # starts path
    def leftClick(self,event):
        # create a line
        # add to collection of paths
        self.paths.append(self.CanvasObj(self.canvas.create_line(event.x,event.y, # start
                                                            event.x,event.y, # end
                                                            fill=self.pathColor, # color
                                                            splinesteps=self.lineRes, # number of points that make up the line
                                                            tags="path", # tag to be used when searching for it
                                                            smooth=True,
                                                            width=self.pathThick,
                                                            activewidth=1.5*self.pathThick) # smooth the line for irregular points
                                                            ,x0=event.x,x1=event.x,y0=event.y,y1=event.y)) # create data structure
        # find the closest coil within a certain range
        # search area is a rectangle
        self.clobj = self.canvas.find_enclosed(event.x-25,event.y-25,event.x+25,event.y+25)
        # only get the objects with the coil tag
        self.clobj = [oid for oid in self.clobj if 'coil' in self.canvas.gettags(oid)]
        self.event = event
        # if there's more than one object within range
        if len(self.clobj)>0:
            # if there's more than one item, find out which one is closer
            # only check if flag is set
            if (not self.useFirstObject) and self.len(self.clobj)>1:
                # calculate the distance between the left click and the other coil's centres within range
                # then sort the list to find the closest one
                clickDist = [(((event.x - p.x[0])**2.0 + (event.y - p.y[0])**2.0)**0.5,p.oid) for p in self.coilIds if p.oid in self.clobj]
                # sort the distances in ascending order with distance as key
                clickDist.sort(key=lambda x : x[0])
                # set target first coil
                self.firstcoil = clickDist[0][1]
            # if self.useFirstObject flag is set, the first object in the found object list
            else:
                self.firstcoil = self.clobj[0]

    # handler for left click release
    # complete path and auto connect to nearest coils
    def leftRelease(self,event):
        # search for the nearests canvas object within range
        self.clobj = self.canvas.find_enclosed(event.x-25,event.y-25,event.x+25,event.y+25)
        # only get the objects with the coil tag
        self.clobj = [oid for oid in self.clobj if 'coil' in self.canvas.gettags(oid)]
        self.event=event
        # if one was found
        if len(self.clobj)>0:
            # if the first coil has been set
            # only draw line if both ends are set
            if hasattr(self,'firstcoil'):
                # if there's more than one coil nearby for the other end
                if (not self.useFirstObject) and len(self.clobj)>1:
                    # calculate the distance between the left click and the other coil's centres within range
                    # then sort the list to find the closest one
                    # centres of the circles are stored in the first values of the x and y vectors
                    clickDist = [(((event.x - p.x[0])**2.0 + (event.y - p.y[0])**2.0)**0.5,p.oid) for p in self.coilIds if p.oid in self.clobj]
                    # sort the distances in ascending order
                    clickDist.sort(key=lambda x : x[0])
                    # set the secondcoil id as the closest object
                    self.secondcoil = clickDist[0][1]
                # if the flag isn't set, use the id
                else:
                    self.secondcoil = self.clobj[0]
                # can't excite and receive from the same coil
                # if they're the same coil, set the second coil as the next one in the sequence
                if self.secondcoil==self.firstcoil:
                    # get the index of the first selected coil
                    idx = self.getCoilNumber(self.firstcoil)
                    # retrieve the next one in the sequence
                    # if the last coil was selected, use the first one
                    # else use the next one in the sequence
                    if idx==len(self.coilIds):
                        self.secondcoil = self.coilIds[0].oid
                    else:
                        self.secondcoil = self.coilIds[idx+1].oid                
                # update the end points of lines to go to the coils
                # change start point of the most recent path
                self.paths[-1].x[0] = self.coilIds[self.getCoilNumber(self.firstcoil)].x[0]
                self.paths[-1].y[0] = self.coilIds[self.getCoilNumber(self.firstcoil)].y[0]
                # change other end point
                self.paths[-1].x[-1] = self.coilIds[self.getCoilNumber(self.secondcoil)].x[0]
                self.paths[-1].y[-1] = self.coilIds[self.getCoilNumber(self.secondcoil)].y[0]

                # update drawing of paths
                # pass coordinates as list of coordinate pairs
                self.canvas.coords(self.paths[-1].oid,self.paths[-1].xy())

                # update excitation path
                self.exciteOrder.append([self.firstcoil,self.secondcoil])
                # sort by excitation coil index
                self.exciteOrder.sort(key=lambda x : x[0])
            # if a first coil wasn't found
            # delete path fromm canvas
            else:
                self.canvas.delete(self.paths.pop().oid)
        # if potential second coils aren't within range, delete the path
        else:
            self.canvas.delete(self.paths.pop().oid)

    # get the coil number associated with the target object id
    def getCoilNumber(self,oid):
        return [pi for pi,p in enumerate(self.coilIds) if p.oid == oid][0]

    # left button drag handler
    # append location to path 
    def drawPath(self,event):
        # add the new position to the coordinates
        self.paths[-1].x.append(event.x)
        self.paths[-1].y.append(event.y)
        # update the coordinates of the last path created
        # the end point is updated based on the current mouse position
        self.canvas.coords(self.paths[-1].oid,self.paths[-1].xy())

    # take a screenshot of the canvas and save as an image
    def saveCanvasAsPNG(self):
        # get the size of the main window and the starting coordinates of the canvas within the window
        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y = self.master.winfo_rooty() + self.canvas.winfo_y()
        # get the end coordinates of the bounding box
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        # get the image and save it under a dummy name for now
        path = filedialog.asksaveasfilename(initialdir=os.getcwd(),initialfile="test.png",confirmoverwrite=True,defaultextension="*.png",title="Choose where to save the screenshot to",filetypes=[("PNG","*.png"),("JPEG","*.png")])
        ImageGrab.grab().crop((x,y,x1,y1)).save(path)
        
if __name__ == "__main__":
    root = Tk()
    view = ExcitationOrderGUI(root)
    root.mainloop()
