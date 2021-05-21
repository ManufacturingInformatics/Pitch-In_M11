import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import h5py

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20-unpack\png"
  
def processImages():
    # get the filenames
    fnames = [os.path.join(path,p) for p in os.listdir(path) if p.endswith(".png")]
    # get number of file names
    numImgs = len(fnames)
    print(f"Detected {numImgs} files")
    # read in first image
    frame = cv2.imread(fnames[0])
    if frame is None:
        print(f"Failed to read in {fnames[0]}")
        return
    print(f"Assuming each image is {frame.shape}")
    # foreground and background model
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)    
    mask = np.zeros(frame.shape[:2],np.uint8)
    rect = (0,0,*frame.shape[:2])
    # setup fn to process
    def grab_cut(img):
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        # process mask to get foreground and background,2,0 respectively
        mask2 = np.where((mask==2)|(mask==0),0,1).astype("uint8")
        # apply mask to image to get result
        return cv2.bitwise_and(img,frame,mask=mask2)

    f,ax = plt.subplots()
    imgPlot = ax.imshow(grab_cut(frame),cmap="gray")
    # function for updating plot
    def update(pp):
        frame = cv2.imread(pp)
        if frame is None:
            print(f"Failed to read in {pp}")
            return imgPlot
        imgPlot.set_array(grab_cut(frame))
        f.tight_layout()
        return imgPlot,
    # setup animation object
    print("creating animation object")
    anim = animation.FuncAnimation(f,update,fnames,interval=1000/60,blit=True)
    # save animation
    print("saving animation")
    anim.save("grabcut-thermal-timelapse.mp4",writer="ffmpeg",fps=60)

def processSizeHDF5(path):
    with h5py.File(path,'r') as file:
        dset = file['pi-camera-1']
        print(f"dataset : {dset.shape}")
        # foreground and background model
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)    
        mask = np.zeros(dset.shape[:2],np.uint8)
        rect = (0,0,*dset.shape[:2])
        def processFrame(ff):
            print(ff)
            # get frame
            frame = dset[:,:,ff]
            # replace nans
            frame[np.isnan(frame)]=0.0
            # normalize
            frame *= (frame.max(axis=(0,1)))**-1.0
            # rescale to 0-255
            frame *= 255
            frame = frame.astype("uint8")
            # perform grabcut
            cv2.grabCut(np.dstack((frame,)*3),mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
            # process mask to get foreground and background,2,0 respectively
            mask2 = np.where((mask==2)|(mask==0),0,1).astype("uint8")
            #cv2.imshow("mask",mask2*255)
            #cv2.waitKey(0)
            # find contours in mask
            ct,_ = cv2.findContours(mask2,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
            if len(ct)>0:
                # find largest contour
                ct = max(ct,key=cv2.contourArea)
                #print(ct)
                return cv2.contourArea(ct)
            else:
                return 0.0

        sz = [processFrame(ff) for ff in range(dset.shape[2])]
    # covert to numpy array and save
    np.savetxt("grabcut-size.csv",np.array(sz),delimiter=',')

def getImage(ff):
    # get the filenames
    fnames = [os.path.join(path,p) for p in os.listdir(path) if p.endswith(".png")]
    if (ff<0) or (ff>len(fnames)):
        print(f"Target frame {ff} outside number of frames or negative")
        return
    # attempts to read in frame
    frame = cv2.imread(fnames[ff])
    if fnames is None:
        print(f"Failed to read in {fnames[ff]}")
        return
    # foreground and background model
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)    
    mask = np.zeros(frame.shape[:2],np.uint8)
    rect = (0,0,*frame.shape[:2])
    # perform grab cut on the target frame
    cv2.grabCut(frame,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    # apply generated mask to image
    mask2 = np.where((mask==2)|(mask==0),0,1).astype("uint8")
    # apply mask to image to get result
    return cv2.bitwise_and(frame,frame,mask=mask2)

def getImageHDF5(ff,path):
    with h5py.File(path,'r') as file:
        # get frame
        frame = file['pi-camera-1'][:,:,ff]
    # replace nans
    frame[np.isnan(frame)]=0.0
    # normalize
    frame *= (frame.max(axis=(0,1)))**-1.0
    # rescale to 0-255
    frame *= 255
    frame = frame.astype("uint8")
    # foreground and background model
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)    
    mask = np.zeros(frame.shape[:2],np.uint8)
    rect = (0,0,*frame.shape[:2])
    # perform grab cut on the target frame
    cv2.grabCut(frame,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    # apply generated mask to image
    mask2 = np.where((mask==2)|(mask==0),0,1).astype("uint8")
    # apply mask to image to get result
    return cv2.bitwise_and(frame,frame,mask=mask2)
