import h5py
import numpy as np
from cv2 import imwrite
from skimage.io import imsave as sk_imsave
import os
import argparse

def printFileStructure(item,leading='  '):
    """ Print the h5 or dictionary file structure

        Prints the keys of each item as an indented list.
        If the item is a dataset, then the shape is printed.
        Level of indentation indicates level of depth in the tree
    """
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
            
def padFrameNumber(ff,depth):
    """ Convert the frame number to a padded string based off the number of images to write """
    return ''.join(['0']*(len(str(depth))-len(str(ff))))+str(ff)

# path to upper directory where the images are stored
#path = "D:\BEAM\Scripts\CallWaifu2x\pi-camera-data-192168146-2019-09-05T10-45-18\ScaledResults"
#path = r"arrowshape-temperature-HDF5.hdf5"

parser = argparse.ArgumentParser(description="Iterate through the datasets inside target HDF5 file and write the data as grayscale images to a folder based off the filename, <filename>-unpack.\nSet mirror flag to have the created folder to have the same structure as the HDF5 file.")
parser.add_argument('-p',type=str,dest='path',help='Path to target HDF5 file')
parser.add_argument('--m',type=bool,nargs='?',dest='mirr',help='Flag to have created folder structure mimic HDF5 file structure. If False, the files are written to a single folder. Default True',default=True)

args = parser.parse_args()
path = args.path
mirror = args.mirr
print("Settings : path = {}, mirror = {}".format(path,mirror))
# create folder for results based off hdf5 name
fname = os.path.basename(path)
fname,ext = os.path.splitext(fname)
fname += "-global-unpack"
print("Creating destination folder {}".format(fname))
target_dir = os.path.join(os.path.dirname(path),fname)
# mirror tree
#mirror = True
# open hdf5 file via context manager
with h5py.File(path,'r') as file:
    # create directory
    os.makedirs(target_dir,exist_ok=True)
    print("File structure...")
    printFileStructure(file)
    # iterate through items
    for key,value in file.items():
        print("Searching {}...\r".format(value.name),end='')
        # if the value is a group
        if isinstance(value,h5py.Group):
            # if flag is set to mirror hdf5 structure as directory tree
            if mirror:
                print("Creating folder for group {}".format(value.name))
                # create folder based on directory tree of HDF5
                fname = os.path.join(target_dir,value.name)
                os.makedirs(target_dir,exist_ok=True)
        # if value is a dataset
        elif isinstance(value,h5py.Dataset):
            # create tuple to ensure global max is obtained
            axx = tuple([a for a in range(len(value.shape))])
            max_dset = value[()].max(axx[:-2])
            min_dset = value[()].min(axx[:-2])
            # remove nans
            np.nan_to_num(max_dset,copy=False)
            np.nan_to_num(min_dset,copy=False)
            # find the max that is below 700
            mmax = max_dset[max_dset<700].max()
            mmin = min_dset.min()
            ## iterate through and save images
            # use opencv for 8-bit and 16-bit 
            if (value.dtype == np.uint8) or (value.dtype == np.uint16):
                for ff in range(value.shape[2]):
                    path = os.path.join(fname,"{0}-f{1}.png".format(key,padFrameNumber(ff,value.shape[2])))
                    cv2.imwrite(path,value[ff])
            # use skimage for 32 float and 64 float
            elif (value.dtype == np.float32) or (value.dtype == np.float64):
                for ff in range(value.shape[2]):
                    frame = value[:,:,ff]
                    # normalize to [0,1]
                    frame = (frame-frame.min())/(mmax-frame.min())
                    # save as tif
                    path = os.path.join(fname,"{0}-f{1}.tif".format(key,padFrameNumber(ff,value.shape[2])))
                    sk_imsave(path,frame)
