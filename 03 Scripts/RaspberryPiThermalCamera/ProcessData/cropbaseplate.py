import h5py
import numpy as np
import cv2
import os

data_ranges = [[47500,48050],[47500,49500],[60000,60450],[60000,62000],[68000,71000],[68200,68820],[90000,93000],[96800,100750],[100750,131750],[90000,150000]]
path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20-global-unpack\png"

with h5py.File(path,'r') as file:
    max_dset = file['pi-camera-1'][()].max((0,1))
    min_dset = file['pi-camera-1'][()].min((0,1))
    np.nan_to_num(max_dset,copy=False)
    np.nan_to_num(min_dset,copy=False)
    mmax,mmin = max_dset.max(),min_dset.min()
    for dd in data_ranges:
        os.makedirs(os.path.join("CroppedBasePlate","{}-{}".format(dd[0],dd[1])),exist_ok=True)
        for ff in range(dd[0],dd[1],1):
           frame = file['pi-camera-1'][:,:,ff]
           frame = (frame-mmin)*(1/(mmax-mmin))
           cv2.imwrite(os.path.join("CroppedBasePlate","{}-{}".format(dd[0],dd[1]),"pi-camera-crop-f{}.png".format(ff)),(frame[:,15:25]*255).astype('uint8'))

for dd in data_ranges:
    os.makedirs(os.path.join("CroppedBasePlate","{}-{}".format(dd[0],dd[1]),"Contours"),exist_ok=True)
    os.makedirs(os.path.join("CroppedBasePlate","{}-{}".format(dd[0],dd[1]),"Threshold"),exist_ok=True)
    for ff in range(dd[0],dd[1],1):
        frame = cv2.imread(os.path.join("CroppedBasePlate","{}-{}".format(dd[0],dd[1]),"pi-camera-crop-f{}.png".format(ff)),cv2.IMREAD_GRAYSCALE)
        thresh = cv2.threshold(frame,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.imwrite(os.path.join("CroppedBasePlate","{}-{}".format(dd[0],dd[1]),"Threshold","pi-camera-crop-otsu-f{}.png".format(ff)),thresh)
        # search for contours in the thresholded image
        ct = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
        # sort in descending order with respect to size
        ct.sort(key=lambda x : cv2.contourArea(x),reverse=True)
        # stack to create a color image so coloured lines can be drawn on
        frame = np.dstack((frame,frame,frame))
        # assign a random color to each conntour
        cols = np.random.randint(50,255,size=len(ct),dtype='uint8')
        # draw contours on image
        for ci,cc in zip(range(len(ct)),cols):
            c = tuple(map(int,cc))
            frame = cv2.drawContours(frame,ct,ci,c,thickness=cv2.FILLED)
        cv2.imwrite(os.path.join("CroppedBasePlate","{}-{}".format(dd[0],dd[1]),"Contours","pi-camera-crop-contour-f{}.png".format(ff)),frame)
