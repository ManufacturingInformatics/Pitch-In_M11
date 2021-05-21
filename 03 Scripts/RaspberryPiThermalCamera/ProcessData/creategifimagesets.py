from shutil import copyfile
import os

def padFrameNumber(ff,depth):
    """ Convert the frame number to a padded string based off the number of images to write """
    return ''.join(['0']*(len(str(depth))-len(str(ff))))+str(ff)

    
data_ranges = [[47500,48050],[47500,49500],[60000,60450],[60000,62000],[68000,71000],[68200,68820],[90000,93000],[96800,100750],[100750,131750],[90000,150000]]
path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20-global-unpack\png"
os.makedirs("{}gifs".format(("global" if "global" in path else "")),exist_ok=True)
for dd in data_ranges:
    print("Copying image range {} to {}".format(dd[0],dd[1]))
    os.makedirs(os.path.join("{}gifs".format(("global" if "global" in path else "")),"{}-{}".format(dd[0],dd[1])),exist_ok=True)
    for ff in range(dd[0],dd[1]):
        #print("Searching for image {}".format(padFrameNumber(ff,data_ranges[-1][1])))
        # check if the paths have been set correctly
        copyfile(os.path.join(path,"pi-camera-1-f{}.png".format(padFrameNumber(ff,data_ranges[-1][1]))),os.path.join("{}gifs".format(("global" if "global" in path else "")),"{}-{}".format(dd[0],dd[1]),"pi-camera-1-f{}.png".format(padFrameNumber(ff,data_ranges[-1][1]))))
