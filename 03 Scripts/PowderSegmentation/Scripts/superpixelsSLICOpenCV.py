import cv2
import numpy as np
import matplotlib.pyplot as plt

# generate LUT labels
# good for <=256 labels
# from https://stackoverflow.com/a/57080906
def genLUTLabels(labels):
    label_range = np.linspace(0,1,256)
    lut = np.uint8(plt.cm.viridis(label_range)[:,2::-1]*256).reshape(256, 1, 3) # replace viridis with a matplotlib colormap of your choice
    return cv2.LUT(cv2.merge((labels, labels, labels)), lut)

def useCmap(labels,cmap=cv2.COLORMAP_RAINBOW):
    lnorm = labels.copy()
    cv2.normalize(labels,lnorm,0,255,cv2.NORM_MINMAX)
    return cv2.applyColorMap(lnorm,cmap)
    

#### based off http://amroamroamro.github.io/mexopencv/opencv_contrib/superpixels_demo.html
# function applying the segment_colorfulness to the live footage
# and producing a coloured display
def useCameraSP(sp='SEEDS',smooth=True,convert2Lab=True,scale=1.0):
    # get the webcam/default camera
    cam = cv2.VideoCapture(0)
    # if not opened
    # exit
    if not cam.isOpened():
        print("failed to open camera")
        cam.release()
        return

    ret,frame = cam.read()
    if not ret:
        print("failed to get frame")
        cam.release()
        return

    # superpixel segmentation parameters
    num_superpixels = 20 #  SEEDS Number of Superpixels
    num_levels = 4 #  SEEDS Number of Levels
    prior = 2 #  SEEDS Smoothing Prior
    num_histogram_bins = 5 #  SEEDS histogram bins
    double_step = False #  SEEDS two steps
    region_size = 20 #  SLIC/SLICO/MSLIC/LSC average superpixel size
    ruler = 15.0 #  SLIC/MSLIC smoothness (spatial regularization)
    ratio = 0.075 #  LSC compactness
    min_element_size = 25 #  SLIC/SLICO/MSLIC/LSC minimum component size percentage
    num_iterations = 10 #  Iterations

    # create object based on target type
    if sp == "SEEDS":
        superpix = cv2.ximgproc.createSuperpixelSEEDS(frame.shape[1],frame.shape[0],frame.shape[2],num_superpixels,num_levels,prior,num_histogram_bins,double_step)
    elif sp in ["SLIC","SLICO","MSLIC"]:
        algo = {"SLIC":cv2.ximgproc.SLIC,
                "SLICO":cv2.ximgproc.SLICO,
                "MSLIC":cv2.ximgproc.MSLIC}
        superpix = cv2.ximgproc.createSuperpixelSLIC(frame,algo[sp],region_size,ruler)
    elif sp == "LSC":
        superpix = cv2.ximgproc.createSuperpixelLSC(frame,region_size,ratio)
    else:
        print("unknwon algorithm")
        return
    
    while True:
        # get frame
        ret,frame = cam.read()
        if not ret:
            print("failed to get frame")
            cam.release()
            return

        # process image according to parameters
        if scale != 1.0:
            frame = cv2.resize(frame,(0,0),fx=scale,fy=scale)

        if smooth:
            fraeme = cv2.GaussianBlur(frame,(3,3),0)
            
        # make a copy of the frame to process
        processed = frame.copy()
        # merge small superpixels to neighbouring ones
        if sp == "SEEDS":
            superpix.iterate(processed,num_iterations)
        else:
            # pass to segmenters
            superpix.iterate(num_iterations)
            if min_element_size >0:
                superpix.enforceLabelConnectivity(min_element_size)
            
        # get results
        labels = superpix.getLabels()
        # generate colors
        #L = genLUTLabels(labels.astype("uint8"))
        L = useCmap(labels.astype("uint8"),cv2.COLORMAP_TWILIGHT_SHIFTED)
        # show results
        cv2.imshow("Frame",frame)
        cv2.imshow("Labels",L)

        # get key press and draw result
        key = cv2.waitKey(1) & 0xff
        # if ESC, exit
        if key == 27:
            print("ESC presssed")
            break
        # if lower case R
        # rebuild class
        elif key == ord('r'):
            print("rebuilding class")
            if sp == "SEEDS":
                superpix = cv2.ximgproc.createSuperpixelSEEDS(frame.shape[1],frame.shape[0],frame.shape[2],num_superpixels,num_levels,prior,num_histogram_bins,double_step)
            elif sp in ["SLIC","SLICO","MSLIC"]:
                algo = {"SLIC":cv2.ximgproc.SLIC,
                        "SLICO":cv2.ximgproc.SLICO,
                        "MSLIC":cv2.ximgproc.MSLIC}
                superpix = cv2.ximgproc.createSuperpixelSLIC(frame,algo[sp],region_size,ruler)
            elif sp == "LSC":
                superpix = cv2.ximgproc.createSuperpixelLSC(frame,region_size,ratio)

    cam.release()

if __name__ == "__main__":
    useCameraSP()
    cv2.destroyAllWindows()
            
