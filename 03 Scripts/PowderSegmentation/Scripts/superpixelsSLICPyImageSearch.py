import cv2
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io

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

# function applying the segment_colorfulness to the live footage
# and producing a coloured display
def useCamera():
    print("getting camera")
    # get the webcam/default camera
    cam = cv2.VideoCapture(0)
    # if not opened
    # exit
    if not cam.isOpened():
        print("failed to open camera")
        return
    
    # transparency transparency level
    alpha = 0.6
    # main looop
    # loop until escape is pressed
    print("starting loop")
    while True:
        ret,frame = cam.read()
        if not ret:
            cam.release()
            print("failed to get frame")
            return
        # convert image froom BGR to RGB
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # create mask
        vis = np.zeros(frame.shape[:2],dtype='float')
        
        # cut the frame up into segments/superpixels
        # number of segments is currently fixed
        # slic_zero uses the zero parameter version of slic that does not require
        # manual sepection of parameters
        segments = slic(img_as_float(frame),n_segments=100,slic_zero=True)

        # iterate over each unique superpixel
        for v in np.unique(segments):
            # create a mask for the superpixel
            mask = np.ones(frame.shape[:2])
            # update mask for superpixel
            # 1 is masked, 0 is not masked
            mask[segments==v]=0
            # compute colorfullness for the superpixel
            # the mask is to ensure that stats are only computed for the masked
            # region
            C = segment_colorfulness(frame,mask)
            vis[segments==v] = C

        # rescale intensity so we can display it
        # convert to 8-bit
        vis = rescale_intensity(vis,out_range=(0,255)).astype('uint8')

        # overlay colourfulness on the original image
        overlay = np.dstack([vis]*3)
        output = frame.copy()
        cv2.addWeighted(overlay,alpha,output,1-alpha,0,output)

        cv2.imshow("Frame",frame)
        cv2.imshow("Vis",vis)
        cv2.imshow("Output",output)
        
        # key press handler
        # allows opencv to draw
        # press escape to
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("ESC exit")
            break
        elif key == ord('s'):
            print("saving screenshot")
            cv2.imwrite("slic_pyimgsearch_screenshot.png",np.concatenate((frame,np.dstack((vis,vis,vis)),output),axis=1))

    cam.release()

if __name__ == "__main__":
    useCamera()

    cv2.destroyAllWindows()
