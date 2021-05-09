from tkinter import *


# code extended from https://stackoverflow.com/questions/26988204/using-2d-array-to-create-clickable-tkinter-canvas
class ClickGridCanvas(Canvas):
    def __init__(self,master,nrows,ncols,**cv_args):
        # initialize super using passed canvas arguments
        super().__init__(master,**cv_args)

        # save number of rows and cols
        # stored privately to keep a reference
        self.__nrows = nrows
        self.__ncols = ncols

        # set the maximum number of selected cells to the number of cells
        self.max_select = nrows*ncols

        # color for selected tile
        self.selected_col = "black"

        # create a nested list to keep track of the current state of the tiles
        # each inner list is the columns for the specific row
        self.tiles = [[None for _ in range(self.__ncols)] for _ in range(self.__nrows)]

        self.bind("<Button-1>",self.callback)
        self.grid(row=0,column=0)

    def callback(self,event):
        # update known width and heights of the columns
        col_width = self.winfo_width()/self.__ncols
        row_height = self.winfo_height()/self.__nrows
        # calculate row and column number
        # event.x and event.y are where the user clicked
        col = int(event.x//col_width)
        row = int(event.y//row_height)
        ## Update tile state and color
        # if tile is not filled
        if not self.tiles[row][col]:
            # cover portion of the canvas with a colored rectangle indicating that it has been selected
            # update tile references
            self.tiles[row][col] = self.create_rectangle(col*col_width,row*row_height,(col+1)*col_width,(row+1)*row_height,fill=self.selected_col)
            # check if the number of selected cells exceeds the limit
            # if it does, undo last change
            if len(self.getSelected())>self.max_select:
                self.delete(self.tiles[row][col])
                self.tiles[row][col] = None
        else:
            # delete the created rectangle using the stored object id
            self.delete(self.tiles[row][col])
            # clear entry
            self.tiles[row][col] = None
        self.update_idletasks()
        
    def getSelected(self):
        # iterate over tiles reference list finding which indicies have been set
        # the indicies are in 1D 
        return [[ii%self.__nrows,ii//self.__ncols] for ii in (i for i,x in enumerate(sum(self.tiles,[])) if x)]

class TestWindowForClickGridCanvas:
    def __init__(self,master,nrows=2,ncols=2):
        self.master = master
        # create instance of click grid object
        self.grid = ClickGridCanvas(self.master,nrows,ncols,width=500,height=500,borderwidth=5,background='white')
        self.grid.pack()

#if __name__ == "__main__":         
    #root = Tk()
    #view = TestWindowForClickGridCanvas(root)
    #oot.mainloop() 
        
