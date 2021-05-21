import numpy as np
from matplotlib import pyplot as plt
import cv2

def useCamera(cam,rect):
    # foreground and background model
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    # mask
    ret,frame = cam.read()
    if not ret:
        print("failed to get frame")
        cam.release()
        return
    
    mask = np.zeros(frame.shape[:2],np.uint8)
    
    while True:
        ret,frame = cam.read()
        if not ret:
            print("failed to get frame")
            cam.release()
            return
        # perform graph cut with rectangle as initializer
        cv2.grabCut(frame,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        # process mask to get foreground and background,2,0 respectively
        mask2 = np.where((mask==2)|(mask==0),0,1).astype("uint8")
        # apply mask to image to get result
        res = cv2.bitwise_and(frame,frame,mask=mask2)
        # update windows
        cv2.imshow("frame",frame)
        cv2.imshow("res",res)
        # handle key press + draw
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("ESC exit")
            break
        elif key == ord('s'):
            cv2.imwrite("opencv_grabcut_screenshot.png",cv2.concatenate((frame,res),axis=1))

    cam.release()

if __name__ == "__main__":
    # setup camera
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("failed to get camera")
    else:
        ret,frame = cam.read()
        if not ret:
            print("failed to get test frame")
        else:            
            roi = cv2.selectROI("Select ROI",frame)
            cv2.destroyWindow("Select ROI")
            useCamera(cam,roi)
            
    cam.release()
    cv2.destroyAllWindows()
