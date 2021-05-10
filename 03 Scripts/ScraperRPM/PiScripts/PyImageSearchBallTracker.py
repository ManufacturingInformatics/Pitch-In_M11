from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import argparse
import matplotlib.pyplot as plt

import RotatingDotGen as RD

class BasicTracker:
    def __init__(self,imtype="BGR"):
        # tracked centre points queue
        self.pts = deque(maxlen=100)
        # current centre
        self.center = None
        # resize factor
        self.rf = 1.0
        # image conversion factor when changing images
        if imtype == "BGR":
            self.__convF = cv2.COLOR_BGR2HSV
        elif imtype == "RGB":
            self.__convF = cv2.COLOR_RGB2HSV
        # time of measurement
        self.__t = time.time()
        # current estimate of velocity
        self.__v = 0.0
        # minimum radius of enclosed circle
        self.minR = 5
        ## flags for drawing
        # show centroid
        self.showCentroid = False
        # show tail
        self.showTail = True
        # show boundary
        self.showBounds = True
    # change the size of the history buffer
    def changeHistSize(self,nsz):
        nhist = deque(maxlen=nsz)
        for i in range(min(nsz,len(self.pts))):
            nhist.appendleft(self.pts[i])
        self.pts = nhist
    # get the current estimate of linear velocity
    def getV(self):
        return self.__v
    # get the most recent centre added to the history queue
    def getLastCenter(self):
        return self.pts[0]
    # perform tracking on the current frame
    def track(self,frame):
        # handler for null frames
        if frame is None:
            print("failed to get frame")
            return None
        # downsize frame
        frame = imutils.resize(frame,width=int(frame.shape[1]*self.rf))
        # blur frame to reduce high freq noise
        blur = cv2.GaussianBlur(frame,(11,11),0)
        # convert to hsv color space
        hsv = cv2.cvtColor(blur,self.__convF)
        # perform erosions and dialations to remove remaining small artifacts
        mask = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        # erode and dialate to remove noise and other objects that might confuse the find contours
        mask = cv2.erode(hsv,None,iterations=2)
        mask = cv2.dilate(mask,None,iterations=2)
        # find the edges in our mask
        cts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cts = imutils.grab_contours(cts)
        # create image to draw on
        draw = np.zeros(frame.shape,dtype=frame.dtype)
        # if a contour was found
        if len(cts)>0:
            # find the largest contour
            c = max(cts,key=cv2.contourArea)
            # find the circle that encloses it
            ((x,y),r) = cv2.minEnclosingCircle(c)
            # check if the radius meets a minimum size
            # if it does calculate new center
            if r > self.minR:
                # get current time
                # taken as time new centre was measured
                t = time.time()
                # update duration between old time and new time
                self.__dur = t - self.__t
                # update time to be used next iteration
                self.__t = t
                # find the centre of the circle based off moment
                M = cv2.moments(c)
                self.center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
                # if the show bounding circle is set
                # draw circle
                if self.showBounds:
                    # draw circle showing enclosed circle
                    # colored yellow
                    cv2.circle(draw,(int(x),int(y)),int(r),(0,255,255),2)
                # if show centroid flag is set, draw centroid of the circle
                if self.showCentroid:
                    # filled circle for centroid
                    # colored red
                    cv2.circle(draw,self.center,1,(0,0,255),-1)
                ## update velocity
                # if there's at least one other point in the queue
                # then there's a center we can compare against
                if len(self.pts)>1:
                    ## calculating angular velocity
                    # centre of rotation
                    # taking it to be centre of image for now
                    cr = [sz//2 for sz in frame.shape[:2][::-1]]
                    # new centroid relative to centre of rotation
                    cpr = [b-a for b,a in zip(cr,self.center)]
                    # angle
                    self.angle = np.arctan2(cpr[1],cpr[0])
                    # old position relative
                    ocpr = [b-a for b,a in zip(cr,self.pts[0])]
                    #print(ocpr)
                    ## calculate directional velocity relative to point of rotation
                    # difference between the previous and new centres
                    diff = [b-a for b,a in zip(cpr,ocpr)]
                    # velocity in the respective directions
                    # try-except to get div 0 error
                    try:
                        vel = [diff[0]/self.__dur,diff[1]/self.__dur]
                    except ZeroDivisionError:
                        print("div 0 error")
                        vel = [0.0,0.0]
                    ## calculate angular velocity
                    # numerator
                    num = (cpr[0]*vel[1]) - (vel[0]*cpr[1])
                    # demominator
                    den = (cpr[0]**2.0) + (cpr[1]**2.0)
                    # print result
                    self.__v = num/den
        # add center to queue
        # if a new center is not found, the old one is added
        # fixed length of queue
        self.pts.appendleft(self.center)
        # draw tail of previous points
        if self.showTail:
            ## drawing results
            # loop over queue of pts
            for i in range(1,len(self.pts)):
                # if the current point of previous point are None
                # then target was not found in frame
                # ignore index
                if (self.pts[i-1] is None) or (self.pts[i] is None):
                    continue
                ## if target was found
                # calculate thickness of the line based on the number
                # of points in the buffer and out current index
                # result is a decreasing trail width
                tk = int(((len(self.pts)/float(i+1))**0.5)*2.5)
                # draw lines connecting the two centres
                cv2.line(draw,self.pts[i-1],self.pts[i],(0,0,255),tk)
        # return drawn on result
        return draw

    # perform tracking on the current frame
    def debugTrack(self,frame):
        # handler for null frames
        if frame is None:
            print("failed to get frame")
            return None
        # downsize frame
        frame = imutils.resize(frame,width=int(frame.shape[1]*self.rf))
        # blur frame to reduce high freq noise
        blur = cv2.GaussianBlur(frame,(11,11),0)
        # convert to hsv color space
        hsv = cv2.cvtColor(blur,self.__convF)
        # perform erosions and dialations to remove remaining small artifacts
        mask = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        pred = mask.copy()
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        mask = cv2.erode(hsv,None,iterations=2)
        mask = cv2.dilate(mask,None,iterations=2)
        postd = mask.copy()
        # find the edges in our mask
        cts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cts = imutils.grab_contours(cts)
        # create image to draw on
        draw = np.zeros(frame.shape,dtype=frame.dtype)
        # if a contour was found
        if len(cts)>0:
            # find the largest contour
            c = max(cts,key=cv2.contourArea)
            # find the circle that encloses it
            ((x,y),r) = cv2.minEnclosingCircle(c)
            # check if the radius meets a minimum size
            # if it does calculate new center
            if r > self.minR:
                # get current time
                # taken as time new centre was measured
                t = time.time()
                # update duration between old time and new time
                self.__dur = t - self.__t
                # update time to be used next iteration
                self.__t = t
                # find the centre of the circle based off moment
                M = cv2.moments(c)
                self.center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
                # if the show bounding circle is set
                # draw circle
                if self.showBounds:
                    # draw circle showing enclosed circle
                    # colored yellow
                    cv2.circle(draw,(int(x),int(y)),int(r),(0,255,255),2)
                # if show centroid flag is set, draw centroid of the circle
                if self.showCentroid:
                    # filled circle for centroid
                    # colored red
                    cv2.circle(draw,self.center,1,(0,0,255),-1)
                ## update velocity
                # if there's at least one other point in the queue
                # then there's a center we can compare against
                if len(self.pts)>1:
                    ## calculating angular velocity
                    # centre of rotation
                    # taking it to be centre of image for now
                    cr = [sz//2 for sz in frame.shape[:2][::-1]]
                    # new centroid relative to centre of rotation
                    cpr = [b-a for b,a in zip(cr,self.center)]
                    # angle
                    self.angle = np.arctan2(cpr[1],cpr[0])
                    # old position relative
                    ocpr = [b-a for b,a in zip(cr,self.pts[0])]
                    ## calculate directional velocity relative to point of rotation
                    # difference between the previous and new centres
                    diff = [b-a for b,a in zip(cpr,ocpr)]
                    # velocity in the respective directions
                    # try-except to get div 0 error
                    try:
                        vel = [diff[0]/self.__dur,diff[1]/self.__dur]
                    except ZeroDivisionError:
                        print("div 0 error")
                        vel = [0.0,0.0]
                    ## calculate angular velocity
                    # numerator
                    num = (cpr[0]*vel[1]) - (vel[0]*cpr[1])
                    # demominator
                    den = (cpr[0]**2.0) + (cpr[1]**2.0)
                    # print result
                    self.__v = num/den
        # add center to queue
        # if a new center is not found, the old one is added
        # fixed length of queue
        self.pts.appendleft(self.center)
        # draw tail of previous points
        if self.showTail:
            ## drawing results
            # loop over queue of pts
            for i in range(1,len(self.pts)):
                # if the current point of previous point are None
                # then target was not found in frame
                # ignore index
                if (self.pts[i-1] is None) or (self.pts[i] is None):
                    continue
                ## if target was found
                # calculate thickness of the line based on the number
                # of points in the buffer and out current index
                # result is a decreasing trail width
                tk = int(((len(self.pts)/float(i+1))**0.5)*2.5)
                # draw lines connecting the two centres
                cv2.line(draw,self.pts[i-1],self.pts[i],(0,0,255),tk)
        # pre and post dialation images are grayscale and need to be converted to 3-channel images in order to be stacked
        # with frame and draw
        pred = np.dstack((pred,pred,pred))
        postd = np.dstack((postd,postd,postd))
        # add text onto each of the images to describe what each one is
        H,W = frame.shape[:2]
        cv2.putText(pred,"In-Range",(10,H-20),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255))
        cv2.putText(postd,"Denoised",(10,H-20),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255))
        dframe = frame.copy()
        cv2.putText(dframe,"Frame",(10,H-20),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255))
        ddraw = draw.copy()
        cv2.putText(ddraw,"Results",(10,H-20),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255))
        debugFrame = np.concatenate([np.concatenate((dframe,pred),axis=1),np.concatenate((postd,ddraw),axis=1)],axis=0)
        # return drawn on result
        return debugFrame

###### adapted from https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
# class adapted from the code in the above link
# segments an image to identify a target object within the color range tlow,thigh
# by comparing two frames, the angular velocity is estimated
class BallTracker:
    def __init__(self,tlow,thigh,imtype="BGR",**kwargs):
        # target color limits in HSV space
        self.lw = tuple(tlow)
        self.up = tuple(thigh)
        # keyword for controlling the size of history queue
        if "histLen" in kwargs:
            self.pts = deque(maxlen=kwargs["histLen"])
        else:
            # tracked centre points queue
            self.pts = deque(maxlen=100)
        # current centre
        self.center = None
        # resize factor
        self.rf = 1.0
        # image conversion factor when changing images
        if imtype == "BGR":
            self.__convF = cv2.COLOR_BGR2HSV
        elif imtype == "RGB":
            self.__convF = cv2.COLOR_RGB2HSV
        # time of measurement
        self.__t = time.time()
        # current estimate of velocity
        self.__v = 0.0
        # minimum radius of identified enclosed circle
        self.minR = 5
        ## flags for drawing
        # show centroid
        self.showCentroid = False
        # show tail
        self.showTail = True
        # show boundary
        self.showBounds = True
        # angle
        self.angle = None
        # last drawn frame
        self.res = None
        # flag describing if the last frame was found
        self.__found = False
    # function to check state of found flag
    def wasFound(self):
        return self.__found
    # get the most recent centre added to the queue
    def getLastCenter(self):
        return self.pts[0]
    # change the size of the history buffer
    def changeHistSize(self,nsz):
        nhist = deque(maxlen=nsz)
        for i in range(min(nsz,len(self.pts))):
            nhist.appendleft(self.pts[i])
        self.pts = nhist
    # update color targets
    def updateTargets(self,**kwargs):
        if "tlow" in kwargs:
            self.lw = tuple(kwargs["tlow"])
        if "thigh" in kwargs:
            self.up = tuple(kwargs["thigh"])
    # get the current estimate of linear velocity
    def getV(self):
        return self.__v
    # perform tracking on the current frame
    def track(self,frame):
        self.__found = False
        # handler for null frames
        if frame is None:
            print("failed to get frame")
            return None
        # downsize frame
        frame = imutils.resize(frame,width=int(frame.shape[1]*self.rf))
        # blur frame to reduce high freq noise
        blur = cv2.GaussianBlur(frame,(11,11),0)
        # convert to hsv color space
        hsv = cv2.cvtColor(blur,self.__convF)
        # construct a mask to search for our target color
        #print(f"hsv shape {hsv.shape}")
        #print(f"lw {self.lw}, up {self.up}")
        mask = cv2.inRange(hsv,self.lw,self.up)
        # perform erosions and dialations to remove remaining small artifacts
        mask = cv2.erode(mask,None,iterations=2)
        mask = cv2.dilate(mask,None,iterations=2)
        #cv2.imshow("mask",mask)
        # find the edges in our mask
        cts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cts = imutils.grab_contours(cts)
        #print(len(cts))
        # create image to draw on
        draw = np.zeros(frame.shape,dtype=frame.dtype)
        # if a contour was found
        if len(cts)>0:
            # find the largest contour
            c = max(cts,key=cv2.contourArea)
            # find the circle that encloses it
            ((x,y),r) = cv2.minEnclosingCircle(c)
            # check if the radius meets a minimum size
            # if it does calculate new center
            if r > self.minR:
                self.__found=True
                # get current time
                # taken as time new centre was measured
                t = time.time()
                # update duration between old time and new time
                self.__dur = t - self.__t
                # update time to be used next iteration
                self.__t = t
                # find the centre of the circle based off moment
                M = cv2.moments(c)
                self.center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
                # if the show bounding circle is set
                # draw circle
                if self.showBounds:
                    # draw circle showing enclosed circle
                    # colored yellow
                    cv2.circle(draw,(int(x),int(y)),int(r),(0,255,255),2)
                # if show centroid flag is set, draw centroid of the circle
                if self.showCentroid:
                    # filled circle for centroid
                    # colored red
                    cv2.circle(draw,self.center,1,(0,0,255),-1)
                ## update velocity
                # if there's at least one other point in the queue
                # then there's a center we can compare against
                if len(self.pts)>1:
                    ## calculating angular velocity
                    # centre of rotation
                    # taking it to be centre of image for now
                    cr = [sz//2 for sz in frame.shape[:2][::-1]]
                    # new centroid relative to centre of rotation
                    cpr = [b-a for b,a in zip(cr,self.center)]
                    self.angle = np.arctan2(cpr[1],cpr[0])
                    # old position relative
                    ocpr = [b-a for b,a in zip(cr,self.pts[0])]
                    ## calculate directional velocity relative to point of rotation
                    # difference between the previous and new centres
                    diff = [b-a for b,a in zip(cpr,ocpr)]
                    # velocity in the respective directions
                    # try-except to get div 0 error
                    try:
                        vel = [diff[0]/self.__dur,diff[1]/self.__dur]
                    except ZeroDivisionError:
                        print("div 0 error")
                        vel = [0.0,0.0]
                    ## calculate angular velocity
                    # numerator
                    num = (cpr[0]*vel[1]) - (vel[0]*cpr[1])
                    # demominator
                    den = (cpr[0]**2.0) + (cpr[1]**2.0)
                    # print result
                    self.__v = num/den
        # add center to queue
        # if a new center is not found, the old one is added
        self.pts.appendleft(self.center)
        # draw tail of previous points
        if self.showTail:
            ## drawing results
            # loop over queue of pts
            for i in range(1,len(self.pts)):
                # if the current point or previous point are None
                # then target was not found in frame
                if (self.pts[i-1] is None) or (self.pts[i] is None):
                    continue
                ## if target was found
                # calculate thickness of the line based on the number of pts in buffer
                tk = int(((len(self.pts)/float(i+1))**0.5)*2.5)
                # draw lines connecting the two centres
                cv2.line(draw,self.pts[i-1],self.pts[i],(0,0,255),tk)
        # update res
        self.res = draw.copy()
        # return drawn on result
        return draw,self.__found
    def debugTrack(self,frame):
        # handler for null frames
        if frame is None:
            print("failed to get frame")
            return None
        # downsize frame
        frame = imutils.resize(frame,width=int(frame.shape[1]*self.rf))
        # blur frame to reduce high freq noise
        blur = cv2.GaussianBlur(frame,(11,11),0)
        # convert to hsv color space
        hsv = cv2.cvtColor(blur,self.__convF)
        # construct a mask to search for our target color
        mask = cv2.inRange(hsv,self.lw,self.up)
        pred = mask.copy()
        # perform erosions and dialations to remove remaining small artifacts
        mask = cv2.erode(mask,None,iterations=2)
        mask = cv2.dilate(mask,None,iterations=2)
        postd = mask.copy()
        #cv2.imshow("mask",mask)
        # find the edges in our mask
        cts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cts = imutils.grab_contours(cts)
        #print(len(cts))
        # create image to draw on
        draw = np.zeros(frame.shape,dtype=frame.dtype)
        # if a contour was found
        if len(cts)>0:
            # find the largest contour
            c = max(cts,key=cv2.contourArea)
            # find the circle that encloses it
            ((x,y),r) = cv2.minEnclosingCircle(c)
            # check if the radius meets a minimum size
            # if it does calculate new center
            if r > self.minR:
                # get current time
                # taken as time new centre was measured
                t = time.time()
                # update duration between old time and new time
                self.__dur = t - self.__t
                # update time to be used next iteration
                self.__t = t
                # find the centre of the circle based off moment
                M = cv2.moments(c)
                self.center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
                # if the show bounding circle is set
                # draw circle
                if self.showBounds:
                    # draw circle showing enclosed circle
                    # colored yellow
                    cv2.circle(draw,(int(x),int(y)),int(r),(0,255,255),2)
                # if show centroid flag is set, draw centroid of the circle
                if self.showCentroid:
                    # filled circle for centroid
                    # colored red
                    cv2.circle(draw,self.center,1,(0,0,255),-1)
                ## update velocity
                # if there's at least one other point in the queue
                # then there's a center we can compare against
                if len(self.pts)>1:
                    ## calculating angular velocity
                    # centre of rotation
                    # taking it to be centre of image for now
                    cr = [sz//2 for sz in frame.shape[:2][::-1]]
                    # new centroid relative to centre of rotation
                    cpr = [b-a for b,a in zip(cr,self.center)]
                    self.angle = np.arctan2(cpr[1],cpr[0])
                    # old position relative
                    ocpr = [b-a for b,a in zip(cr,self.pts[0])]
                    ## calculate directional velocity relative to point of rotation
                    # difference between the previous and new centres
                    diff = [b-a for b,a in zip(cpr,ocpr)]
                    # velocity in the respective directions
                    # try-except to get div 0 error
                    try:
                        vel = [diff[0]/self.__dur,diff[1]/self.__dur]
                    except ZeroDivisionError:
                        print("div 0 error")
                        vel = [0.0,0.0]
                    ## calculate angular velocity
                    # numerator
                    num = (cpr[0]*vel[1]) - (vel[0]*cpr[1])
                    # demominator
                    den = (cpr[0]**2.0) + (cpr[1]**2.0)
                    # print result
                    self.__v = num/den
        # add center to queue
        # if a new center is not found, the old one is added
        self.pts.appendleft(self.center)
        # draw tail of previous points
        if self.showTail:
            ## drawing results
            # loop over queue of pts
            for i in range(1,len(self.pts)):
                # if the current point of previous point are None
                # then target was not found in frame
                if (self.pts[i-1] is None) or (self.pts[i] is None):
                    continue
                ## if target was found
                # calculate thickness of the line based on the number of pts in buffer
                tk = int(((len(self.pts)/float(i+1))**0.5)*2.5)
                # draw lines connecting the two centres
                cv2.line(draw,self.pts[i-1],self.pts[i],(0,0,255),tk)
        # pre and post dialation images are grayscale and need to be converted to 3-channel images in order to be stacked
        # with frame and draw
        pred = np.dstack((pred,pred,pred))
        postd = np.dstack((postd,postd,postd))
        # add text onto each of the images to describe what each one is
        H,W = frame.shape[:2]
        cv2.putText(pred,"In-Range",(10,H-20),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255))
        cv2.putText(postd,"Denoised",(10,H-20),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255))
        dframe = frame.copy()
        cv2.putText(dframe,"Frame",(10,H-20),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255))
        ddraw = draw.copy()
        cv2.putText(ddraw,"Results",(10,H-20),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255))
        debugFrame = np.concatenate([np.concatenate((dframe,pred),axis=1),np.concatenate((postd,ddraw),axis=1)],axis=0)
        # return drawn on result
        return debugFrame

def track_app(vs,lw,up):
    # define a queue to hold the data
    pts = deque(maxlen=100)
    center = None
    ## main loop
    while True:
        # get frame
        frame = vs.read()
        if frame is None:
            print("failed to get frame")
            vs.stop()
            break
        # downsize frame
        # by factor of 2
        frame = imutils.resize(frame,width=int(frame.shape[1]*0.5))
        # blur frame to reduce high freq noise
        blur = cv2.GaussianBlur(frame,(11,11),0)
        # convert to hsv color space
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        # construct a mask to search for our target color
        mask = cv2.inRange(hsv,lw,up)
        # perform erosions and dialations to remove remaining small artifacts
        # in the image
        mask = cv2.erode(mask,None,iterations=2)
        mask = cv2.dilate(mask,None,iterations=2)

        # find the edges in our mask
        cts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cts = imutils.grab_contours(cts)

        # make a copy of frame to draw on
        draw = frame.copy()

        # if a contour was found
        if len(cts)>0:
            # find the largest contour
            c = max(cts,key=cv2.contourArea)
            # find the circle that encloses it
            ((x,y),r) = cv2.minEnclosingCircle(c)
            # check if the radius meets a minimum size
            if r > 10:
                # find the centre of the circle
                M = cv2.moments(c)
                center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
                # draw circle and centroid
                cv2.circle(draw,(int(x),int(y)),int(r),
                           (0,255,255),2) # color and thickness
                # filled circle for centroid
                cv2.circle(draw,center,5,(0,0,255),-1)
        # add center to queue
        # if a new center is not found, the old one is added
        # fixed length of queue
        pts.appendleft(center)

        ## drawing results
        # loop over queue of pts
        for i in range(1,len(pts)):
            # if the current point of previous point are None
            # then target was not found in frame
            # ignore index
            if (pts[i-1] is None) or (pts[i] is None):
                continue

            ## if target was found
            # calculate thickness of the line based on the number
            # of points in the buffer and out current index
            # result is a decreasing trail width
            tk = int(((len(pts)/float(i+1))**0.5)*2.5)
            # draw lines connecting the two centres
            cv2.line(draw,pts[i-1],pts[i],(0,0,255),tk)

        # combine raw frame and masked output to show input and output
        show = np.hstack((frame,np.dstack((mask,mask,mask)),draw))
        # show frame
        cv2.imshow("Frame",show)

        # check for exit key
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            print("exit")
            break

# use the webcam with the ball tracker
def use_camera():
    # define the color range of target in HSV colorspace
    colLower = (0,150,50)
    colUpper = (15,255,255)

    # get reference to webcam
    # start webcam
    print("starting webcam")
    camObj = VideoStream(src=0).start()
    # wait for camera to warm up
    time.sleep(2.0)
    # start app to track object
    track_app(camObj,colLower,colUpper)
    camObj.stop()

# utility function for converting an RGB color to HSV color
def RGB2HSV(col):
    # create a small image of 3 channels
    img = np.zeros((1,1,3),np.uint8)
    # populate image with data
    img[:,:,:] = col
    # use opencv to convert color to HSV
    # convert output to list. returns twice nested list
    # return opened list
    return cv2.cvtColor(img,cv2.COLOR_RGB2HSV).tolist()[0][0]

def BGR2HSV(col):
    # create a small image of 3 channels
    img = np.zeros((1,1,3),np.uint8)
    # populate image with data
    img[:,:,:] = col
    # use opencv to convert color to HSV
    # convert output to list. returns twice nested list
    # return opened list
    return cv2.cvtColor(img,cv2.COLOR_BGR2HSV).tolist()[0][0]

# use the rotating dot animation with the tracker
def use_dot_anim():
    # define rotating dot manager
    rd = RD.RotatingDot(15,100)
    # define rotating animation manager
    # update at a rate of 60 FPS
    anim = RD.RotatingAnimator(rd,dur=60.0**-1.0,dangle=np.deg2rad(5.0))
    # set color limits of target
    # needs a fairly narrow tolerance on it
    # ref:
    # black : target[2] = 0-50
    # white : target[2] = 200-255
    colLower = (0,0,0)
    colUpper = (255,255,50)
    # create tracker class
    tracker = BallTracker(colLower,colUpper,"RGB")
    # start animation
    anim.start()
    # loop until esc is pressed
    while True:
        # get frame
        frame = anim.get()
        # invert frame
        #frame = cv2.bitwise_not(frame)
        # feed into tracker
        res,found = tracker.track(frame)
        cv2.imshow("Found",res)
        cv2.imshow("Frame",frame)
        # break on exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    # stop animation
    anim.stop()

# use rotating dot animation and estitmating velocity
def use_dot_anim_vel():
    # define rotating dot manager
    rd = RD.RotatingDot(15,100)
    # define rotating animation manager
    # update at a rate of 60 FPS
    anim = RD.RotatingAnimator(rd,dur=60.0**-1.0,dangle=np.deg2rad(5.0))
    print(f"Animator set at {anim.getDuration()**-1.0} FPS {anim.getDuration()}, {anim.getAngleChange()} rads and {rd.getDist()} px")
    # get distance from centre
    d = rd.getDist()
    # print target angular velocity
    # target calculated as the angle change per update divided by the 
    print(f"target vel (rad/s): {anim.getAngleChange()/anim.getDuration()}")
    targetVel = anim.getAngleChange()/anim.getDuration()
    # set color limits of target
    # needs a fairly narrow tolerance on it
    # ref:
    # black : target[2] = 0-50
    # white : target[2] = 200-255
    colLower = (0,0,0)
    colUpper = (255,255,50)
    # create tracker class stating that the inputs are in RGB colorspace
    tracker = BallTracker(colLower,colUpper,"RGB")
    # create vector to hold velocity data
    vel = np.zeros((1,))
    # starting time
    tstart = time.time()
    # dur
    dur = anim.getDuration()
    # current timestamp
    ots = tstart
    # old animation angle
    orot = rd.getCurrAngle()
    # start animation
    anim.start()
    # loop until esc is pressed
    while True:
        t = time.time()
        if (t-ots)>=dur:
            # get frame
            frame = anim.get()
            # invert frame
            #frame = cv2.bitwise_not(frame)
            # add noise
            frame = cv2.blur(frame,(5,5))
            # feed into tracker
            res,found = tracker.track(frame)
            # show results
            cv2.imshow("Found",res)
            cv2.imshow("Frame",frame)
            # update velocity vector
            vel = np.append(vel,tracker.getV())
            rot = rd.getCurrAngle()
            #print(np.rad2deg(rot))
            #print("Actual diff: ",rot-orot,", Target: ",anim.getAngleChange())
            orot = rot
            ots = t
        # escape on exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        #time.sleep(0.1)

    anim.stop()
    f,ax = plt.subplots()
    ax.plot(vel,'b-',label="Est. Velocity")
    ax.plot(np.full(vel.shape,targetVel),'r--',label="Target Velocity")
    f.legend()
    ax.set(xlabel="Data Index",ylabel="Velocity (rad/s)",title=f"Estimated Angular Velocity using Ball Tracking")
    plt.show()

# use rotating dot animation and estitmating velocity
# apply occlusion
def dot_vel_occ(recRes=False):
    # define rotating dot manager
    rd = RD.RotatingDot(15,100)
    # define rotating animation manager
    # update at a rate of 60 FPS
    anim = RD.RotatingAnimator(rd,dur=60.0**-1.0,dangle=np.deg2rad(5.0))
    print(f"Animator set at {anim.getDuration()**-1.0} FPS {anim.getDuration()}, {anim.getAngleChange()} rads and {rd.getDist()} px")
    # get distance from centre
    d = rd.getDist()
    # print target angular velocity
    # target calculated as the angle change per update divided by the 
    print(f"target vel (rad/s): {anim.getAngleChange()/anim.getDuration()}")
    targetVel = anim.getAngleChange()/anim.getDuration()
    # set color limits of target
    # needs a fairly narrow tolerance on it
    # ref:
    # black : target[2] = 0-50
    # white : target[2] = 200-255
    colLower = (0,0,0)
    colUpper = (255,255,50)
    # create tracker class stating that the inputs are in RGB colorspace
    tracker = BallTracker(colLower,colUpper,"RGB")
    # get information about rotating dot to design occlusion
    H,W,_ = rd.getFrameSize()
    # define occlusion
    fac = RD.OcclusionFactory()
    # ellipse in the top right hand corner
    occ = fac.build("ellipse",col=(255,255,255),center=(int((W//2)+d*0.9),int((H//2)-d*0.9)),angle=-45,axes=(d//2,d))
    # create vector to hold velocity data
    vel = np.zeros((1,))
    # starting time
    tstart = time.time()
    # dur
    dur = anim.getDuration()
    # current timestamp
    ots = tstart
    # old animation angle
    orot = rd.getCurrAngle()
    # setup recorder
    h,w,d = rd.getImg().shape
    if recRes:
        rec = cv2.VideoWriter("rotating_dot_balltracker_occlusion.avi",cv2.VideoWriter_fourcc(*'MJPG'),60,(h,w),True)
        if not rec.isOpened():
            print("Failed to open video file")
    # start animation
    anim.start()
    # loop until esc is pressed
    while True:
        t = time.time()
        if (t-ots)>=dur:
            # get frame
            frame = anim.get()
            # invert frame
            #frame = cv2.bitwise_not(frame)
            # add noise
            frame = cv2.blur(frame,(5,5))
            # add occlusion
            frame = occ.draw(frame)
            # feed into tracker
            res,found = tracker.track(frame)
            # show results
            cv2.imshow("Found",res)
            cv2.imshow("Frame",frame)
            # update velocity vector
            # if found use current estimate
            if found:
                vel = np.append(vel,tracker.getV())
            # if lost use previous estimate
            else:
                vel = np.append(vel,vel[-1])
            # write to file
            if recRes:
                if rec.isOpened():
                    rec.write(res)
            rot = rd.getCurrAngle()
            #print(np.rad2deg(rot))
            #print("Actual diff: ",rot-orot,", Target: ",anim.getAngleChange())
            orot = rot
            ots = t
        # escape on exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        #time.sleep(0.1)
    # stop animation
    anim.stop()
    # release recorder
    if recRes:
        rec.release()

    # display estimated velocity
    f,ax = plt.subplots()
    ax.plot(vel,'b-',label="Est. Velocity")
    ax.plot(np.full(vel.shape,targetVel),'r--',label="Target Velocity")
    f.legend()
    ax.set(xlabel="Data Index",ylabel="Velocity (rad/s)",title=f"Estimated Angular Velocity using Ball Tracking")
    plt.show()

# use the ball tracking class to track the rotating dot
# class debugTrack method is use to show the processing at each step
def ball_track_debug(recRes=False):
    # define rotating dot manager
    rd = RD.RotatingDot(15,100)
    # define rotating animation manager
    # update at a rate of 60 FPS
    anim = RD.RotatingAnimator(rd,dur=60.0**-1.0,dangle=np.deg2rad(5.0))
    print(f"Animator set at {anim.getDuration()**-1.0} FPS {anim.getDuration()}, {anim.getAngleChange()} rads and {rd.getDist()} px")
    # get distance from centre
    d = rd.getDist()
    # print target angular velocity
    # target calculated as the angle change per update divided by the 
    print(f"target vel (rad/s): {anim.getAngleChange()/anim.getDuration()}")
    targetVel = anim.getAngleChange()/anim.getDuration()
    # set color limits of target
    # needs a fairly narrow tolerance on it
    # ref:
    # black : target[2] = 0-50
    # white : target[2] = 200-255
    colLower = (0,0,0)
    colUpper = (255,255,50)
    # create tracker class stating that the inputs are in RGB colorspace
    tracker = BallTracker(colLower,colUpper,"RGB")
    # starting time
    tstart = time.time()
    # dur
    dur = anim.getDuration()
    # current timestamp
    ots = tstart
    # old animation angle
    orot = rd.getCurrAngle()
    # setup recorder
    h,w,d = rd.getImg().shape
    if recRes:
        rec = cv2.VideoWriter("rotating_dot_balltracker_debug.avi",cv2.VideoWriter_fourcc(*'MJPG'),60,(h*2,w*2),True)
        if not rec.isOpened():
            print("Failed to open video file")
    # start animation
    anim.start()
    # loop until esc is pressed
    while True:
        t = time.time()
        if (t-ots)>=dur:
            # get frame
            frame = anim.get()
            # invert frame
            #frame = cv2.bitwise_not(frame)
            # add noise
            frame = cv2.blur(frame,(5,5))
            # feed into tracker
            res = tracker.debugTrack(frame)
            # show results
            cv2.imshow("Found",res)
            cv2.imshow("Frame",frame)
            # record if set
            if recRes:
                if rec.isOpened():
                    rec.write(res)
            rot = rd.getCurrAngle()
            #print(np.rad2deg(rot))
            #print("Actual diff: ",rot-orot,", Target: ",anim.getAngleChange())
            orot = rot
            ots = t
        # escape on exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        #time.sleep(0.1)
    if recRes:
        rec.release()
    anim.stop()

def save_dot_anim_vel(recRes=False):
    # define rotating dot manager
    rd = RD.RotatingDot(15,100,start_angle=np.pi/2)
    # define rotating animation manager
    # update at a rate of 60 FPS
    anim = RD.RotatingAnimator(rd,dur=60.0**-1.0,dangle=np.deg2rad(5.0))
    print(f"Animator set at {anim.getDuration()**-1.0} FPS {anim.getDuration()}, {anim.getAngleChange()} rads and {rd.getDist()} px")
    # get distance from centre
    d = rd.getDist()
    # print target angular velocity
    # target calculated as the angle change per update divided by the update period
    print(f"target vel (rad/s): {anim.getAngleChange()/anim.getDuration()}")
    # set color limits of target
    # needs a fairly narrow tolerance on it
    # ref:
    # black : target[2] = 0-50
    # white : target[2] = 200-255
    colLower = (0,0,0)
    colUpper = (255,255,50)
    # create tracker class stating that the inputs are in RGB colorspace
    tracker = BallTracker(colLower,colUpper,"RGB")
    # create vector to hold velocity data
    vel = np.zeros((0,2))
    # starting time
    tstart = time.time()
    # old timestamp
    ots = tstart
    # run time
    # number of rotations divided by velocity
    tlim = 4*(2.0*np.pi)*((anim.getAngleChange()/anim.getDuration())**-1.0)
    print("set run time to ",tlim, " s")
    # get frame shape
    ss = list(rd.getFrameSize()[:2][::-1])
    ss[0] *= 2
    print(ss)
    #setup recorder
    if recRes:
        rec = cv2.VideoWriter("rotating_dot_balltracker.avi",cv2.VideoWriter_fourcc(*'MJPG'),60,tuple(ss))
        if not rec.isOpened():
            print("Failed to open video file")
    # start animation
    anim.start()
    # loop until esc is pressed
    while True:
        t = time.time()
        # get frame
        frame = anim.get()
        # invert frame
        #frame = cv2.bitwise_not(frame)
        # add noise
        frame = cv2.blur(frame,(5,5))
        # feed into tracker
        res,found = tracker.track(frame)
        # show results
        cv2.imshow("Found",res)
        cv2.imshow("Frame",frame)
        # update velocity vector
        vel = np.append(vel,[[t-tstart,tracker.getV()]])
        # write combine frames and write to video
        ww = np.hstack((frame,res))
        if recRes:
            if rec.isOpened():
                rec.write(ww)
        # update time stamp
        ots = t
        # check if the run time has reached limit
        if (t-tstart)>=tlim:
            print("finished anim")
            break
        cv2.waitKey(1)
        #time.sleep(0.1)
    # stop animation
    anim.stop()
    # stop video writer
    rec.release()
    # save velocity results
    f,ax = plt.subplots()
    ax.plot(vel)
    ax.set(xlabel="Data Index",ylabel="Velocity (rad/s)",title=f"Estimated Angular Velocity using Ball Tracking")
    f.savefig("rotating_dot_balltracker_est_vel.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implements a ball tracker program based off the code example from PyImageSearch")
    parser.add_argument("--anim",dest="anim",action="store_true",help="Run the ball tracker example using the rotating dot animation")
    parser.add_argument("--animvel",dest="animvel",action="store_true",help="Run code using the rotating dot animation and predict velocity")
    parser.add_argument("--debug",dest="debug",action="store_true",help="Run the code using the default camera")
    parser.add_argument("--rec",dest="record",action="store_true",help="Flag to record the generated footage. Use in combination with debug and occ flags")
    parser.add_argument("--occ",dest="occ",action="store_true",help="Run the code with the dot animation including an occlusion")

    # use_camera
    parser.set_defaults(anim=False,animvel=False,debug=True,record=True,occ=False)
    args = parser.parse_args()
    if args.anim:
        use_dot_anim()
    elif args.animvel:
        use_dot_anim_vel()
    elif args.debug:
        ball_track_debug(args.record)
    elif args.occ:
        dot_vel_occ(args.record)
    # destroy opencv windows
    cv2.destroyAllWindows()
