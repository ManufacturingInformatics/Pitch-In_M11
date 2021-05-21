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

## based off https://www.pyimagesearch.com/2017/06/26/labeling-superpixel-colorfulness-opencv-python/
# adapted for live images
# function for labelling super pixel colourfulness
def segment_colorfulness(image,mask):
    # split the channels
    (B,G,R) = cv2.split(image.astype('float'))
    # build masks
    # only pixels within masks are therefore computed
    R = np.ma.masked_array(R,mask=mask)
    G = np.ma.masked_array(G,mask=mask)
    B = np.ma.masked_array(B,mask=mask)

    # compute difference between Red and Green channels
    rg = np.abs(R-G)

    # compute YB = 0.5*(R+G)-B
    yb = np.abs(0.5*(R+G)-B)

    # compute mean and standard deviation of results
    stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
    meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))
    # calculate the "colorfulness"
    return stdRoot + (0.3 * meanRoot)

# calculate colorfulness of gray image
# based off the colorful metrics above
def segment_color_gray(image,mask):
    mm = np.ma.masked_array(image,mask=mask)
    # computer mean and standard deviation
    stdRoot = np.sqrt(mm.std()**2)
    meanRoot = np.sqrt(mm.mean()**2)
    return stdRoot + (0.3*meanRoot)

# perform SLICO using OpenCV class on the image
# uses color image
def useCamera():
    # parameters
    region_size = 100
    ruler = 15.0
    ratio = 0.075
    min_element_size = 25
    num_iterations = 10

    print("getting camera")
    # get the webcam/default camera
    cam = cv2.VideoCapture(0)
    # if not opened
    # exit
    if not cam.isOpened():
        print("failed to open camera")
        cam.release()
        return

    print("getting test frame")
    ret,frame = cam.read()
    if not ret:
        print("failed to get frame")
        cam.release()
        return

    print("building algorithm")
    # create algorithm
    superpix = cv2.ximgproc.createSuperpixelSLIC(frame,cv2.ximgproc.SLICO,region_size,ruler)
    # create simplified lambda function for passing new frames to it with the same settings
    pass_image = lambda src : cv2.ximgproc.createSuperpixelSLIC(src,cv2.ximgproc.SLICO,region_size,ruler)
    # transparency of blending
    alpha = 0.6
    ## values of supported keys
    # s
    s_key = ord('s')
    # create array used for masking labels
    mask = np.ones(frame.shape[:2])
    # create visiblity results
    vis = np.zeros(frame.shape[:2],dtype='float')
    # main loop
    print("starting loop")
    while True:
        # get frame
        ret,frame = cam.read()
        if not ret:
            print("failed to get frame")
            cam.release()
            return
        # pass new frame
        superpix = pass_image(frame)
        # calculate superpixel segmentation on the previous given image
        superpix.iterate(num_iterations)
        # min_element_size enforces a minimum size of the superpixels
        # merges smaller superpixels into a giant one
        # min_element_size >>, << superpixels
        if min_element_size>0:
            superpix.enforceLabelConnectivity(min_element_size)

        # get labels
        # auto generated number controlled by min_element_size
        labels = superpix.getLabels()
        #print("building label colormap")
        # iterate over labels generating mask and calculating colorfulness metric for each label
        ### MAJOR PERFORMANCE HOG FOR HIGH NUMBER OF SUPERPIXELS ###
        for v in np.unique(labels):
            # reset mask
            mask[...]=1
            # update mask for superpixel
            # 1 is masked and 0 is not masked
            mask[labels==v]=0
            # compute colorfulness
            # mask channels
            vis[labels==v] = segment_colorfulness(frame,mask)
        # rescale colourfulness to display
        vis = cv2.normalize(vis,vis,0,255,cv2.NORM_MINMAX).astype("uint8")
        # overlay colorfulness results with frame to form overlay
        overlay = np.dstack([vis]*3)
        output = frame.copy()
        cv2.addWeighted(overlay,alpha,output,1-alpha,0,output)
        #print("displaying")
        cv2.imshow("Frame",frame)
        cv2.imshow("Vis",vis)
        cv2.imshow("Output",output)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("ESC pressed")
            break
        elif key == s_key:
            cv2.imwrite("opencv_slic_overlay_screenshot.png",np.concatenate((frame,np.dstack([vis]*3),overlay),axis=1))

    cam.release()

def useCameraGray():
    # parameters
    region_size = 50
    ruler = 15.0
    ratio = 0.075
    min_element_size = 25
    num_iterations = 5

    print("getting camera")
    # get the webcam/default camera
    cam = cv2.VideoCapture(0)
    # if not opened
    # exit
    if not cam.isOpened():
        print("failed to open camera")
        cam.release()
        return

    print("getting test frame")
    ret,frame = cam.read()
    if not ret:
        print("failed to get frame")
        cam.release()
        return
    # convert image to grayscale
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    print("building algorithm")
    # create algorithm
    superpix = cv2.ximgproc.createSuperpixelSLIC(frame,cv2.ximgproc.SLICO,region_size,ruler)
    # create simplified lambda function for passing new frames to it with the same settings
    pass_image = lambda src : cv2.ximgproc.createSuperpixelSLIC(src,cv2.ximgproc.SLICO,region_size,ruler)
    # transparency of blending
    alpha = 0.6
    ## values of supported keys
    # s
    s_key = ord('s')
    # create array used for masking labels
    mask = np.ones(frame.shape[:2])
    # create visiblity results
    vis = np.zeros(frame.shape[:2],dtype='float')
    # main loop
    print("starting loop")
    while True:
        # get frame
        ret,frame = cam.read()
        if not ret:
            print("failed to get frame")
            cam.release()
            return
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # pass new frame
        superpix = pass_image(frame)
        # calculate superpixel segmentation on the previous given image
        superpix.iterate(num_iterations)
        # min_element_size enforces a minimum size of the superpixels
        # merges smaller superpixels into a giant one
        # min_element_size >>, << superpixels
        if min_element_size>0:
            superpix.enforceLabelConnectivity(min_element_size)

        # get labels
        # auto generated number controlled by min_element_size
        labels = superpix.getLabels()
        #print("building label colormap")
        # iterate over labels generating mask and calculating colorfulness metric for each label
        ### MAJOR PERFORMANCE HOG FOR HIGH NUMBER OF SUPERPIXELS ###
        for v in np.unique(labels):
            # reset mask
            mask[...]=1
            # update mask for superpixel
            # 1 is masked and 0 is not masked
            mask[labels==v]=0
            # compute colorfulness
            # mask channels
            vis[labels==v] = segment_color_gray(frame,mask)
        # rescale colourfulness to display
        vis = cv2.normalize(vis,vis,0,255,cv2.NORM_MINMAX).astype("uint8")
        # overlay colorfulness results with frame to form overlay
        overlay = vis.copy()
        output = frame.copy()
        cv2.addWeighted(overlay,alpha,output,1-alpha,0,output)
        #print("displaying")
        cv2.imshow("Frame",frame)
        cv2.imshow("Vis",vis)
        cv2.imshow("Output",output)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("ESC pressed")
            break
        elif key == s_key:
            cv2.imwrite("opencv_slic_overlay_screenshot.png",np.concatenate((frame,vis,overlay),axis=1))

    cam.release()
    
if __name__ == "__main__":
    useCamera()
    #useCameraGray()
    cv2.destroyAllWindows()
