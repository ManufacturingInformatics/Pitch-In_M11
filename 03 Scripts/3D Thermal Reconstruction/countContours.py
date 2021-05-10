import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# path for video file
path = r"C:\Users\david\Documents\CoronaWork\Data\waifu2x-flyingedge-surfacearea-ref15.mp4"
# bounding box for masking
bb = (103, 68, 292, 357)
# masking ranges for red bounding box
l0_lower_red = np.array([0,50,50])
l0_upper_red = np.array([10,255,255])
l1_lower_red = np.array([170,50,50])
l1_upper_red = np.array([180,255,255])

# function for conting the number of rings in the 
def countContours(frame):
    #cv2.imshow("input",frame)
    # mask frame within bounding box
    fcrop = frame[int(bb[1]):int(bb[1]+bb[3]),int(bb[0]):int(bb[0]+bb[2])]
    # convert to gray scale
    fgray = cv2.cvtColor(fcrop, cv2.COLOR_BGR2GRAY)
    # remove the red bounding box that sometimes appears in the centre
    fhsv = cv2.cvtColor(fcrop,cv2.COLOR_BGR2HSV)
    mask0 = cv2.inRange(fhsv,l0_lower_red,l0_upper_red)
    mask1 = cv2.inRange(fhsv,l1_lower_red,l1_upper_red)
    mask=mask0+mask1
    #print(f"mask {mask.min()}, {mask.max()}")
    # use mask to remove red pixels
    fgray[np.where(mask==255)]=0
    #cv2.imshow("red mask",fgray)
    ## search for contours
    # threshold to binary
    thresh = cv2.threshold(fgray,50,255,cv2.THRESH_BINARY)[1]
    #cv2.imshow("thresh",thresh)
    ct = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_KCOS)[0]
    # draw contours for debug
    zz = np.zeros(thresh.shape,thresh.dtype)
    drawct=cv2.drawContours(zz,ct,-1,(255,255,255),1)
    #cv2.waitKey(1)
    # if no contours are found, return 0
    if ct is None:
        return 0,drawct
    else:
        return len(ct),drawct

def testThresholding(frame):
    cv2.imshow("input",frame)
    ut = 50
    # mask frame within bounding box
    fcrop = frame[int(bb[1]):int(bb[1]+bb[3]),int(bb[0]):int(bb[0]+bb[2])]
    # convert to gray scale
    fgray = cv2.cvtColor(fcrop, cv2.COLOR_BGR2GRAY)
    # remove the red bounding box that sometimes appears in the centre
    fhsv = cv2.cvtColor(fcrop,cv2.COLOR_BGR2HSV)
    mask0 = cv2.inRange(fhsv,l0_lower_red,l0_upper_red)
    mask1 = cv2.inRange(fhsv,l1_lower_red,l1_upper_red)
    mask=mask0+mask1
    #print(f"mask {mask.min()}, {mask.max()}")
    # use mask to remove red pixels
    fgray[np.where(mask==255)]=0
    ## search for contours
    # threshold to binary
    thresh = cv2.threshold(fgray,ut,255,cv2.THRESH_BINARY)[1]
    cv2.imshow("thresh",thresh)
    key = cv2.waitKey(1)&0xff
    while key!=27:
        ut = int(input("enter new threshold: "))
        if ut<0:
            print("invalid threshold")
            continue
        else:
            print(f"trying {ut}")
            thresh = cv2.threshold(fgray,ut,255,cv2.THRESH_BINARY)[1]
            ct = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_KCOS)[0]
            print(f"len ct using TC89_KCOS {len(ct)}")
            ct = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
            print(f"len ct using SIMPLE {len(ct)}")
            ct = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
            print(f"len ct using NONE {len(ct)}")
            ct = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)[0]
            print(f"len ct using TC89_L1 {len(ct)}")
            cv2.imshow("thresh",thresh)
            key = cv2.waitKey(1)&0xff

def countVideo(path):
    # open video
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Failed to open file")
        cap.release()
        return
    else:
        # create list to populate
        numct = []
        # setup videowriter for drawn contours
        vwrite = cv2.VideoWriter('found_ct.mp4',cv2.VideoWriter_fourcc(*'MJPG'),30.0,(bb[2],bb[3]),False)
        if not vwrite.isOpened():
            print("Failed to open videowriter")
            return
        else:
            print(f"video writer opened {vwrite.isOpened()}")
            input('')
            ret,frame = cap.read()
            while ret:
                nn,drawct = countContours(frame)
                numct.append(nn)
                print(f"{drawct.shape}")
                #drawct = np.dstack((drawct,)*3)
                print(drawct.shape)
                vwrite.write(drawct)
                print(f"len of ct {nn}")
                ret,frame = cap.read()
            cap.release()
            vwrite.release()
            return numct
if __name__ == "__main__":
    numct = countVideo(path)
    np.savetxt("numContoursFlyingEdge.csv",numct,delimiter=',')
    # generate plot
    plt.plot(np.arange(91398,150001),numct)
    plt.gca().set(xlabel='Frame Index',ylabel='Estimated number of contours',title='Estimated number of contours found from\nFlying Edge slice, thresh=50, method=TC89_KCOS')
    plt.gcf().savefig("est-contours-flyingedge.png")
