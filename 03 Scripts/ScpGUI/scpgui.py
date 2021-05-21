from tkinter import Tk,Entry,StringVar,Button,Label,BooleanVar,Checkbutton,Frame,LabelFrame,Canvas,Scrollbar,N,S,E,W
from tkinter.ttk import Progressbar
from tkinter.simpledialog import Dialog
from tkinter.messagebox import showerror,showinfo
from tkinter.filedialog import askdirectory
import paramiko
import socket
from scp import SCPClient
import os

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

# scrollable label frame
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

# dialog for asking the user to entry the username and password of the pi they're copying from
# usernames and passwords are not stored locally and have to be entered
# each time
class LoginDialog(Dialog):
    def __init__(self,parent):
        self.pwd = StringVar()
        self.usr = StringVar(value='pi')
        self.result=None

        super().__init__(parent,title="Entry login details")
        
    def body(self,master):
        self.pwdEntry = Entry(master,textvariable=self.pwd,show="*")
        self.usrEntry = Entry(master,textvariable=self.usr)
        self.pwdLabel = Label(master,text="Password")
        self.usrLabel = Label(master,text="Username")

        self.usrLabel.grid(row=0,column=0,sticky=N+S+E+W)
        self.usrEntry.grid(row=0,column=1,sticky=N+S+E+W)
        self.pwdLabel.grid(row=1,column=0,sticky=N+S+E+W)
        self.pwdEntry.grid(row=1,column=1,sticky=N+S+E+W)

        return self.pwdEntry

    def apply(self):
        self.result = self.usr.get(),self.pwd.get()

# class for defininf a SCP retrieval request
# allows the user to set the search term, destination and options for speeding up searching
class ScpRequest(Frame):
    def __init__(self,master,dinit='',rec=False,**kwargs):
        self.master = master
        super().__init__(master,**kwargs)
        # variable for holding the file search term
        self.fileSearch = StringVar()
        # variable for holding the destination where the file/s will be copied to
        if dinit is '':
            self.destPath = StringVar(value=os.getcwd())
        else:
            self.destPath = StringVar(value=dinit)
        # boolean variable for indicating if the search is recursive
        self.isRec = BooleanVar(value=rec)
        
        # entry boxes for entering the search terms and destination paths
        self.fileSearchEntry = Entry(self,textvariable=self.fileSearch)
        self.destPathEntry = Entry(self,textvariable=self.destPath)
        # button opening a filedialog allowing the user to set the destination directory
        self.findDestButton = Button(self,text="...",command=self.setDestPath)
        # checkbox for indicating if the search is recursive
        # used to set flag in function call
        self.isRecursiveBox = Checkbutton(self,text="Recursive",variable=self.isRec)
        # button to delete self
        self.removeReqButton = Button(self,text="X",command=self.removeReq)

        ## geometry manager
        self.fileSearchEntry.grid(row=0,column=0,sticky=N+S+E+W)
        self.destPathEntry.grid(row=0,column=1,sticky=N+S+E+W)
        self.findDestButton.grid(row=0,column=2,sticky=N+S+E+W)
        self.isRecursiveBox.grid(row=0,column=3,sticky=N+S+E+W)
        self.removeReqButton.grid(row=0,column=4,sticky=N+S+E+W)

    # callback for findDestButton
    # opens filedialog for the user to select a target directory for where the request will be copied to
    def setDestPath(self):
        dname = askdirectory(title="Set the destination of the request",initialdir=self.destPath.get(),mustexist=True)
        if dname is not '':
            self.destPath.set(dname)

    def getTarget(self):
        return self.fileSearch.get()
    
    def getDest(self):
        return self.destPath.get()

    def setDest(self,ndest):
        return self.destPath.set(ndest)

    def isRecursive(self):
        return self.isRec.get()

    def setRecursive(self,flag):
        return self.isRec.set(flag)

    # button to remove the requests
    def removeReq(self):
        self.destroy()

# frame showing the requested files and their destinations in a queue
class ScpRequestQueue(ScrollableLabelFrame):
    def __init__(self,master,**kwargs):
        if "text" not in kwargs:
            kwargs["text"] = "Requests"
        # initialize frame
        super().__init__(master,**kwargs)

        ## boolean variables
        # setting all queries as recursive
        self.allRecursive = BooleanVar(value=False)
        # setting all queries with the same destination
        self.sameDestPath = BooleanVar(value=False)

        # locally stored number of requests in queue
        self.__numReqs = 0

        ## checkboxes for setting boolean variables
        # recursive
        self.recursiveCheck = Checkbutton(self.scroll_frame,text="All Recursive",command=self.setAllRecursive,variable=self.allRecursive)
        # same destination path
        self.sameDestCheck = Checkbutton(self.scroll_frame,text="Same Path",command=self.setAllSameDest,variable=self.sameDestPath)

        # button to clear queue
        self.clearQueueButton = Button(self.scroll_frame,text="Clear Queue",command=self.clearQueue)
        
        ## geometry manager
        # checkboxes at the top
        self.recursiveCheck.grid(row=0,column=0)
        self.sameDestCheck.grid(row=0,column=1)
        self.clearQueueButton.grid(row=0,column=2)

    # method to clear queue
    def clearQueue(self):
        for req in self.getAllRequests(True):
            req.destroy()
            self.__numReqs -=1

    def setAllRecursive(self):
        for req in self.getAllRequests(gen=True):
            req.setRecursive(self.allRecursive.get())

    def setAllSameDest(self):
        # intiialize variables
        dest = ''
        # if there are other requests
        if self.__numReqs==0:
            return
        else:
            # get the first request's destination
            dest = self.getRequest(0).getDest()

        for req in self.getAllRequests(gen=True):
            req.setDest(dest)

    # get all requests in the queue and return as list
    def getAllRequests(self,gen=False):
        if gen:
            return (child for child in self.winfo_children() if isinstance(child,ScpRequest))
        else:
            return [child for child in self.winfo_children() if isinstance(child,ScpRequest)]
    # get a specific request denoted by index
    def getRequest(self,ii):
        if self.__numReqs ==0:
            return None
        
        for ci,child in enumerate(self.winfo_children()):
            if ci==ii:
                return child

    # returns the current number of requests in the queue      
    def getNumRequests(self):
        return self.__numReqs

    # add a ScpRequest to the queue
    def addRequest(self):
        print("adding request")
        # intiialize variables
        dest = ''
        isRec = self.allRecursive.get()
        # if the user has set all the requests to have the same destination path
        if self.sameDestPath.get():
            # if there are other requests
            if self.__numReqs >0:
                # get the first request's destination
                dest = self.getRequest(0).getDest()

        # build request class
        # initial variables are set based on current settings
        req = ScpRequest(self.scroll_frame,dinit=dest,rec=isRec)
        # place in new row
        # first row contains the checkboxes so new requests are always placed one row beneath that
        req.grid(row=self.__numReqs+1,column=0,columnspan=3,sticky=N+S+E+W)
        # update frame
        self.update_idletasks()
        self.__numReqs+=1

# SCP class for connecting to a remote server and making basic SCP copy retrieval requests
# the user sets the login information and target IP and a set of requests they want to perform
# each request is the search term they want to perform, where the results are going to be placed and if it's a recursive request
# when the start button is pressed each reque
class SCPGUI:
    def __init__(self,master):
        self.master = master
        self.master.title("SCP Raspberry Pi")

        # variables for login information
        self.__usr = StringVar()
        self.__pwd = StringVar()
        # ip address
        self.__ip = StringVar(value='raspberrypi.local')
        self.__isConnected = False

        ## scp client classes
        # ssh client
        self.__sshClient = paramiko.SSHClient()
        self.__sshClient.load_system_host_keys()

        
        # button to open the username-password dialog
        self.setLoginButton = Button(self.master,text="Set Login",command=self.getLoginInfo)
        
        # frames to put controls in
        self.controlFrame = Frame(self.master)
        # entry box to set ip
        self.ipEntry = Entry(self.controlFrame,textvariable=self.__ip)
        # frame to put controls in
        # label for ip box
        self.ipLabel = Label(self.controlFrame,text="IP")
        # button to ping ip using client
        self.checkIPButton = Button(self.controlFrame,text="Check IP",command=self.checkIp)
        # label whose color indicates status
        self.connectStatus = Label(self.controlFrame,text="",bg="red")
        
        # scp queue window
        self.reqQueue = ScpRequestQueue(self.master)
        # add one request to queue
        self.reqQueue.addRequest()
        # button to process requests in queue
        self.procQueueButton = Button(self.master,text="Start",command=self.processQueue)
        # scrollable frame to show progress of each request
        self.progressQueueWin = ScrollableLabelFrame(master,text="Progress")
        # button to add requests to the queue
        self.addReqButton = Button(self.master,text="Add",command=self.addRequest)

        ## event handlers
        # window destroyed
        # ensure clients are closed
        self.master.bind("WM_DELETE_WINDOW",self.__closeWindowHandler)
        # handler to manage a progressbar for each request added
        self.reqQueue.bind("<Configure>",self.updateProgress)

        ## geometry manager
        self.setLoginButton.grid(row=0,column=0,columnspan=2,sticky=E+W)
        self.ipLabel.grid(row=0,column=0,sticky=N+E+W)
        self.ipEntry.grid(row=0,column=1,sticky=N+E+W)
        self.checkIPButton.grid(row=1,column=0,sticky=N+E+W)
        self.connectStatus.grid(row=1,column=1,sticky=N+E+W)
        self.controlFrame.grid(row=1,column=0,sticky=N+E+W)
        
        self.addReqButton.grid(row=0,column=2,sticky=E+W)
        self.procQueueButton.grid(row=0,column=3,sticky=E+W)
        self.reqQueue.grid(row=1,column=2,columnspan=2,rowspan=2)

        # make the main window contents resizable 
        num_cols,num_rows = master.grid_size()
        for c in range(num_cols):
            master.columnconfigure(c,weight=1)
        for r in range(num_rows):
            master.rowconfigure(r,weight=1)

    # method for checking the connected flag
    # flag indicates if the last connection attempt was successful
    def connected(self):
        return self.

    def addRequest(self):
        self.reqQueue.addRequest()
        self.master.update_idletasks()

    # handler for when the window is closed
    # ensures that the ssh client is closed
    def __closeWindowHandler(self,event):
        self.__sshClient.close()
        self.__scpClient.close()
        self.master.destroy()

    def getLoginInfo(self):
        di = LoginDialog(self.master)
        if di.result is not None:
            __usr,__pwd = di.result
            self.__usr.set(__usr)
            self.__pwd.set(__pwd)

    # button handler for procQueueButton
    # iterate over the queue trying each request
    def processQueue(self):
        # if connection flag is false
        if not self.__isConnected:
            # attempt connection
            self.checkIp()
            # if still not connected
            # exit and do not process queue
            if not self.__isConnected:
                return
        # if connection is successful
        # iterate over queue items
        # retrieve target, destination and recursive flag passing it to a get request using SCP client
        # handle exceptions by displaying error box 
        for req in self.reqQueue.getAllRequests(gen=True):
            try:
                self.__scpClient.get(req.getTarget(),req.getDest(),req.isRecursive())
            except scp.SCPException as err:
                showerror(title=f"SCP Error for {req.getTarget()}",message=str(err))

    # button for connecting the ssh client to the target ip address
    # updates status label based on result
    def checkIp(self):
        # set status indicator to red to show as disconnected or a bad connection
        self.connectStatus.config(bg="red")
        # close current connection
        self.__sshClient.close()
        # clear connection flag
        self.__isConnected = False
        try:
            # attempt to connect to set ip address
            self.__sshClient.connect(hostname=self.__ip.get(),username=self.__usr.get(),password=self.__pwd.get())
            # scp client
            self.__scpClient = SCPClient(self.__sshClient.get_transport())
            # if successful
            # show window to inform user
            showinfo(title="Success!",message="Connected to IP address")
            # update status label
            self.connectStatus.config(bg="green")
            self.__isConnected = True
        except paramiko.BadHostKeyException as err:
            showerror(title="Failed to connect",message=str(err))
        except paramiko.AuthenticationException as err:
            showerror(title="Incorrect login",message=str(err))
        except socket.error as err:
            showerror(title="Socket Error",message=str(err))
        except Exception as err:
            showerror(title="Error",message=str(err))

    # handler for moving the progress bar showing progress of each copy
    def updateProgress(self,event):
        print(event.type)

if __name__ == "__main__":
    r = Tk()
    view = SCPGUI(r)
    r.mainloop()
