import cv2
import numpy as np
import os

useHDF5=False

if useHDF5:
    print("using hdf5 file")
    import h5py
    def getFrame(ff):
        with h5py.File("D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5",'r') as file:
            frame = file['pi-camera-1'][:,:,ff]
            return (255 * (frame * (frame.max()**-1.0))).astype("uint8")
    with h5py.File("D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5",'r') as file:
        nframes = file['pi-camera-1'].shape[2]
    
else:
    print("using images")
    def getFrame(ff):
        return cv2.imread("D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20-unpack\png\pi-camera-1-f{:06d}.png".format(ff))
    
    import glob
    nframes = len(glob.glob("D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20-unpack\png\*.png"))

print(f"total {nframes} images")

def find_sobel():
    # data type of sobel gradients
    ddepth = cv2.CV_16S
    # create results output
    os.makedirs("WeightedSobel",exist_ok=True)

    for ff in range(nframes):
        # get frame.
        frame = getFrame(ff)
        # reduce noise
        frame = cv2.GaussianBlur(frame,(3,3),0)
        # convert to grayscale
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ## perform sobel edge detection
        # x direction
        grad_x = cv2.Sobel(gray,ddepth,1,0,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
        # y direction
        grad_y = cv2.Sobel(gray,ddepth,0,1,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
        # convert output to 8-bit
        grad_x = cv2.convertScaleAbs(grad_x)
        gray_y = cv2.convertScaleAbs(grad_y)
        # blend the two together
        grad = cv2.addWeighted(grad_x,0.5,gray_y,0.5,0)
        # save results
        cv2.imwrite(os.path.join("WeightedSobel",f"pi-camera-sobel-f{ff:06d}.png"),grad)

def find_area():
    # data type of sobel gradients
    ddepth = cv2.CV_16S
    # create results output
    os.makedirs("SobelContours",exist_ok=True)

    for ff in range(nframes):
        # get frame.
        frame = getFrame(ff)
        # reduce noise
        frame = cv2.GaussianBlur(frame,(3,3),0)
        # convert to grayscale
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ## perform sobel edge detection
        # x direction
        grad_x = cv2.Sobel(gray,ddepth,1,0,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
        # y direction
        grad_y = cv2.Sobel(gray,ddepth,0,1,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
        # convert output to 8-bit
        grad_x = cv2.convertScaleAbs(grad_x)
        gray_y = cv2.convertScaleAbs(grad_y)
        # blend the two together
        grad = cv2.addWeighted(grad_x,0.5,gray_y,0.5,0)
        # threshold
        _,thresh = cv2.threshold(grad,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #print(thresh.dtype)
        #cv2.imwrite(os.path.join("SobelContours",f"pi-camera-sobel-thresh-f{ff:06d}.png"),thresh)
        # find contours
        ct,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        if len(ct)==0:
            print(f"Failed to find contours for {ff}")
            continue
        
        # find 2nd largest contour
        ct.sort(key=cv2.contourArea,reverse=True)
        # stack gradient image to make it a color image
        grad = np.dstack((thresh,thresh,thresh))
        # if there's only one contour
        # draw contour
        draw = cv2.drawContours(grad,[ct[0]],-1,(0,0,255),1)
        # save image
        cv2.imwrite(os.path.join("SobelContours",f"pi-camera-sobel-contour-f{ff:06d}.png"),grad)

def save_area():
    # data type of sobel gradients
    ddepth = cv2.CV_16S
    # size dataset
    harea = []
    for ff in range(nframes):
        # get frame.
        frame = getFrame(ff)
        # reduce noise
        frame = cv2.GaussianBlur(frame,(3,3),0)
        # convert to grayscale
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ## perform sobel edge detection
        # x direction
        grad_x = cv2.Sobel(gray,ddepth,1,0,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
        # y direction
        grad_y = cv2.Sobel(gray,ddepth,0,1,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
        # convert output to 8-bit
        grad_x = cv2.convertScaleAbs(grad_x)
        gray_y = cv2.convertScaleAbs(grad_y)
        # blend the two together
        grad = cv2.addWeighted(grad_x,0.5,gray_y,0.5,0)
        # threshold
        _,thresh = cv2.threshold(grad,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #print(thresh.dtype)
        #cv2.imwrite(os.path.join("SobelContours",f"pi-camera-sobel-thresh-f{ff:06d}.png"),thresh)
        # find contours
        ct,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        if len(ct)==0:
            print(f"Failed to find contours for {ff}")
            continue
        
        # get the size of the largest contour
        ct.sort(key=cv2.contourArea,reverse=True)
        # add to array
        harea.append([ff,cv2.contourArea(ct[0])])
    # convert to numpy array
    # and save
    np.savetxt("thermalContourArea.csv",np.array(harea),delimiter=',')
if __name__ == "__main__":
    pass
    
    
