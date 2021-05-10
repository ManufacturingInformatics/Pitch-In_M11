import cv2
import h5py
import os
import numpy as np
from skimage.io import imread

def scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)  # see below for Python 2.x
        elif '.tif' in entry.name:
            yield entry

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20-unpack"
fname = os.path.splitext(os.path.basename(path))[0]
fname += "-video.mp4"
vw = None
for entry in scantree(path):
    img = imread(entry.path,plugin='tifffile',as_gray=True)
    if vw is None:
        vw = cv2.VideoWriter(fname,cv2.VideoWriter_fourcc(*'mp4v'),1109,img.shape[::-1])
        if vw.isOpened():
            print("Opened videowriter")

    if vw is not None:
        if vw.isOpened():
            #print("\rWriting {}".format(entry.name),end='',sep='')
            vw.write(img.astype('uint8'))

if vw is not None:
    vw.release()
