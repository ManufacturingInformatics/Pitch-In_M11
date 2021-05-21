from tkinter import Tk,Entry,Label,Button,Scrollbar,Frame,CENTER
from tkinter.ttk import Treeview
from tkinter.simpledialog import Dialog
import h5py
import os

class HDF5DatasetDialog(Dialog):
    ''' HDF5DatasetDialog
        ======================
        David Miller, 2020
        The University of Sheffield 2020
        
        Tkinter dialog for inspecting and selecting a dataset inside a target HDF5 file. The full path to the target
        dataset inside is returned. The path can be used to access the dataset when reopening the file.

        The purpose is to provide a filedialog inspired way of selecting data inside files that could be arranged in a
        complex or nested manner.

        On start, the file is accessed and the Treeview is populated with the contents of the file. The name of the entries
        is set as the path to the object and the columns are object type, shape of contents and data type of contents if a
        dataset. The nested structure of the file is represented as nested nodes on the Treeview so a Group's contents are
        represented as nodes underneath the Group on the tree.

        The Treeview's select mode is set to browse so only one item may be selected.

        Arguments:
        --------------------------
        fpath : Full valid path to target HDF5 file to inspect
        title : Title of the dialog window
        message : Message displayed above the Treeview

        Returns:
        --------------------------
        HDF5 file path to target dataset or '' if Cancel, no item is selected or a non-Dataset item is selected.

        Methods:
        --------------------------          
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
        
    def __init__(self,parent,fpath,title=None,message=None):
        # if the target path doesn't point anywhere
        # raise an error
        if not os.path.exists(fpath):
            raise ValueError("Filename is not valid! Does not point to an existing path!")
            return
        # set current file as blank
        self.fpath = fpath
        # initialize result to none
        self.result  = None
        # save message
        self.message=message
        # initialize dialog
        if title is None:
            super().__init__(parent,title="HDF5 Dataset Selecter")
        else:
            super().__init__(parent,title=title)

    def body(self,master):
        # message label
        self.mssg_label = Label(master,text=self.message,font="Arial 10 bold")
        ## setup tree headings
        # tree view for file layout
        self.file_tree = Treeview(master,columns=("htype","shape","dtype"),show="tree",selectmode="browse")
        # left button release handler
        self.file_tree.bind("<<TreeviewSelect>>",self.select_item)
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

        # explore target file and populate file tree
        self.scan_file()
        
        # set grid layout for widgets using grid
        self.mssg_label.grid(column=0,row=0,sticky='nsew')
        self.file_tree.grid(column=0,row=1,sticky='nswe')
        self.tree_scroll.grid(column=1,row=1,sticky='nsew')
        # set weight parameters for control how the elements are resized when the user changes the window size
        master.columnconfigure(0,weight=1)
        master.columnconfigure(1,weight=1)
        master.rowconfigure(0,weight=1)
        master.rowconfigure(1,weight=1)
        # set minimum size to size on creation
        self.master.update_idletasks()
        self.master.minsize(self.master.winfo_width(),self.master.winfo_height())
        # set focus on the tree
        return self.file_tree

    # handler for when user selects an item in the treeview
    def select_item(self,event):
        # get the selectID item
        # because of how the selectmode was set only one item can be selectID
        selectID = event.widget.selection()[0]
        # if the selectID item is a dataset
        # update result
        if 'Dataset' in self.file_tree.item(selectID,"values")[0]:
            self.currResult = self.file_tree.item(selectID,"text")
        else:
            self.currResult = None

    # behaviour of OK button
    # transfer current result to class result
    # that way if cancel is pressed nothing is returned
    def apply(self):
        self.result = self.currResult
        
    # string version of class when inspected
    # e.g. typing name pressing enter in the shell
    def __repr__(self):
        return f"HDF5DatasetDialog(result={self.result},fpath={self.fpath})"

    # string version of string when converted to a string or printed
    # returns the name of the selected item i.e. hdf5 file path
    def __str__(self):
        if self.result is None:
            return ''
        else:
            return self.result

    # handler for equality comparisons
    # compares target object to contents of result
    def __eq__(self,other):
        return self.result == other

    # handler for inequality comparisons
    # compares target object to contents of result
    def __neq__(self,other):
        return self.result != other

    # when used as a key in a dict or hdf5 file
    # returns the currently selected result
    def __hash__(self):
        return self.result

    # function to explore HDF5 group and update tree
    # if it finds another HDF5 group it calls the functions to explore that group
    def explore_group(self,item,parent):
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
        # open file in read mode and iterate through values
        with h5py.File(self.fpath,'r') as file:
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
        # finish idle tasks and set minimum window size to final window size
        self.master.update_idletasks()

class HDF5GroupDialog(Dialog):
    ''' HDF5GroupDialog
        ======================
        David Miller, 2020
        The University of Sheffield 2020
        
        Tkinter dialog for inspecting and selecting a Group inside a target HDF5 file. The full path to the target
        Group inside is returned. The path can be used to access the dataset when reopening the file.

        The purpose is to provide a filedialog inspired way of selecting data inside files that could be arranged in a
        complex or nested manner.

        On start, the file is accessed and the Treeview is populated with the contents of the file. The name of the entries
        is set as the path to the object and the columns are object type, shape of contents and data type of contents if a
        dataset. The nested structure of the file is represented as nested nodes on the Treeview so a Group's contents are
        represented as nodes underneath the Group on the tree.

        The Treeview's select mode is set to browse so only one item may be selected.

        Arguments:
        --------------------------
        fpath : Full valid path to target HDF5 file to inspect
        title : Title of the dialog window
        message : Text displayed above the Treeview

        Returns:
        --------------------------
        HDF5 file path to target Group or '' if Cancel, no item is selected or a non-Group item is selected.

        Methods:
        --------------------------          
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
        
    def __init__(self,parent,fpath,title=None,message=None):
        # if the target path doesn't point anywhere
        # raise an error
        if not os.path.exists(fpath):
            raise ValueError("Filename is not valid! Does not point to an existing path!")
            return
        # set current file as blank
        self.fpath = fpath
        # initialize result to none
        self.result  = None
        # save message
        self.message=message
        # initialize dialog
        if title is None:
            super().__init__(parent,title="HDF5 Dataset Selecter")
        else:
            super().__init__(parent,title=title)

    def body(self,master):
        # message label
        self.mssg_label = Label(master,text=self.message,font="Arial 10 bold")
        ## setup tree headings
        # tree view for file layout
        self.file_tree = Treeview(master,columns=("htype","shape","dtype"),show="tree",selectmode="browse")
        # left button release handler
        self.file_tree.bind("<<TreeviewSelect>>",self.select_item)
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

        # explore target file and populate file tree
        self.scan_file()
        
        # set grid layout for widgets using grid
        self.mssg_label.grid(column=0,row=0,sticky='nsew')
        self.file_tree.grid(column=0,row=1,sticky='nswe')
        self.tree_scroll.grid(column=1,row=1,sticky='nsew')
        # set weight parameters for control how the elements are resized when the user changes the window size
        master.columnconfigure(0,weight=1)
        master.columnconfigure(1,weight=1)
        master.rowconfigure(0,weight=1)
        master.rowconfigure(1,weight=1)
        # set minimum size to size on creation
        self.master.update_idletasks()
        self.master.minsize(self.master.winfo_width(),self.master.winfo_height())
        # set focus on the tree
        return self.file_tree

    # handler for when user selects an item in the treeview
    def select_item(self,event):
        # get the selectID item
        # because of how the selectmode was set only one item can be selectID
        selectID = event.widget.selection()[0]
        # if the selectID item is a dataset
        # update result
        if 'Group' in self.file_tree.item(selectID,"values")[0]:
            self.currResult = self.file_tree.item(selectID,"text")
        else:
            self.currResult = None

    # behaviour of OK button
    # transfer current result to class result
    # that way if cancel is pressed nothing is returned
    def apply(self):
        self.result = self.currResult
        
    # string version of class when inspected
    # e.g. typing name pressing enter in the shell
    def __repr__(self):
        return f"HDF5GroupDialog(result={self.result},fpath={self.fpath})"

    # string version of string when converted to a string or printed
    # returns the name of the selected item i.e. hdf5 file path
    def __str__(self):
        if self.result is None:
            return ''
        else:
            return self.result

    # handler for equality comparisons
    # compares target object to contents of result
    def __eq__(self,other):
        return self.result == other

    # handler for inequality comparisons
    # compares target object to contents of result
    def __neq__(self,other):
        return self.result != other

    # when used as a key in a dict or hdf5 file
    # returns the currently selected result
    def __hash__(self):
        return self.result

    # function to explore HDF5 group and update tree
    # if it finds another HDF5 group it calls the functions to explore that group
    def explore_group(self,item,parent):
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
        # open file in read mode and iterate through values
        with h5py.File(self.fpath,'r') as file:
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
        # finish idle tasks and set minimum window size to final window size
        self.master.update_idletasks()
        
if __name__ == "__main__":
    testpath = r"D:\BEAM\Scripts\MagneticTopography\Scripts\test.hdf5"
    # root tk
    r = Tk()
    # open dialog passing test path
    res = HDF5DatasetDialog(r,testpath,"Select a dataset","Select a dataset from the file")
    # destroy root
    r.destroy()
    # print target result
    print(res)
    # use result to get data
    with h5py.File(testpath,'r') as file:
        print(file[str(res)].shape)
