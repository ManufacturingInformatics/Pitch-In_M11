import h5py
import numpy as np
from cv2 import imread, IMREAD_ANYDEPTH
from skimage.io import imread as sk_imread
import os
from pathlib import Path

def printFileStructure(item,leading='  '):
    """ Print the h5 or dictionary file structure

        Prints the keys of each item as an indented list.
        If the item is a dataset, then the shape is printed.
        Level of indentation indicates level of depth in the tree
    """
    import h5py
    # code adapted from
    #https://stackoverflow.com/questions/34330283/how-to-differentiate-between-hdf5-datasets-and-groups-with-h5py
    for key in item:
        # if object is a dataset print its name followed by shape
        if isinstance(item[key], h5py.Dataset):
            print(leading + key + ": " + str(item[key].shape))
            #get attributes
            atts = item[key].attrs.items()
            # if there are attributes
            if len(atts)>0:
                # print name and value
                # equals sign indicates its a attribute
                # prints it at a different indent level for ease of reading
                for k,v in atts:
                    print('  '+leading+k, "= ",v)
        # if it's something else
        else:
            print(leading + key)
            printFileStructure(item[key],leading=leading+'  ')

# path to upper directory where the images are stored
#path = "D:\BEAM\Scripts\CallWaifu2x\pi-camera-data-192168146-2019-09-05T10-45-18\ScaledResults"
path = "D:\BEAM\Scripts\EdgeDetection\ScharrSumResults"
dir_name = path[path.rfind('\\')+1:]
# open hdf5 file via context manager
with h5py.File('{}.hdf5'.format(dir_name),'w') as file:
    # iterate through directories and files
    for root,dirs,files in os.walk(path):
        # get/create group for images
        # path choice is designed to mirror the directory tree beneath starting path
        grp = file.require_group(str(Path(root).relative_to(path)))
        # for each file found
        num_imgs = len(files)
        # iterate through files
        for ff in files:
            # print current file being written and where it's being written to
            print("{0} : {1}                \r".format(grp.name,ff),end='')
            # if it's a png, read it in and save it to a compressed dataset
            if '.csv' in ff:
                # read in image, returns empty matrix if failed
                data = np.genfromtxt(os.path.join(root,ff))
                # check if img was read in
                if data.shape[0] != 0:
                    if "CSV" not in grp:
                        #create the resizable dataset
                        dset = grp.create_dataset("CSV",(*data.shape,1),data.dtype,data,maxshape=(*data.shape,None),compression='gzip',compression_opts=9)
                    else:
                        dset = grp["CSV"]
                    # if not the first data, increase dataset size by 1
                    if dset.shape[2]>=1:
                        # increase by one size
                        dset.resize(dset.shape[2]+1,axis=2)
                    # add data
                    dset[:,:,-1]=data
            # given past troubles with corrupted images, read in TIF images in a try catch
            # if it can't be read in, move on
            elif ('.xlxs' in ff) :
                data = np.genfromtxt(os.path.join(root,ff))
                # check if img was read in
                if data.shape[0] != 0:
                    if "EXCEL" not in grp:
                        #create the resizable dataset
                        dset = grp.create_dataset("EXCEL",(*data.shape,1),data.dtype,data,maxshape=(*data.shape,None),compression='gzip',compression_opts=9)
                    else:
                        dset = grp["EXCEL"]
                        
                    if dset.shape[2]>=1:
                        # increase by one size
                        dset.resize(dset.shape[2]+1,axis=2)
                    # add data
                    dset[:,:,-1]=data
            
