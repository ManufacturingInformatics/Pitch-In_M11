import numpy as np
import cv2
import RotatingDotGen as RD
import matplotlib.pyplot as plt
import time
from PyImageSearchBallTracker import BallTracker,BGR2HSV
import argparse

# adapted from
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

# wrapper class for lucas kanade optic flow
class LKTracker:
    def __init__(self,ft_params=None,lk_params=None):
        # set parameters for features
        if ft_params is None:
            self.feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
        else:
            self.feature_params = ft_params

        # set parameters for tracker
        if lk_params is None:
            self.lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        else:
            self.lk_params = lk_params

        # set colors to use for points
        self.color = np.random.randint(0,255,(100,3))
        # flag to shuffle colors each run
        self.changeCols = False
        # reference frame
        self.old_gray = None
        self.old_frame = None
        # how often to update reference
        self.rs = 15
        # counter for how many frames have been processed
        # used to update how often to update reference
        self.__ct = 0
        # time the reference was set
        self.__refTime = time.time()
        # time new value was received
        self.__newTime = time.time()
        self.__v = 0.0
        self.p1 = None

    # function to change which colors are used for the points
    def shuffleColors(self):
        color = np.random.randint(0,255,(100,3))
    # perform optic flow analysis on the given frame
    def update(self,frame):
        # convert to gray scale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # update reference if
        #  - there is no reference at the moment
        #  - it's time to update according to internal counter and set limit
        if (self.old_gray is None) or ((self.__ct%self.rs)==0):
            # make a copy of the frame
            self.old_frame = frame.copy()
            # convert to gray
            self.old_gray = frame_gray.copy()
            # find some corners to track as defined by feature params
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask = None, **self.feature_params)
            # Create a mask that will be added to the input data to show the found results
            self.mask = np.zeros(self.old_frame.shape,dtype=self.old_frame.dtype)
            # update time reference was updated
            self.__refTime = time.time()
            # set counter to 1 so the reference isn't reset next round
            self.__ct = 1
            # find moments of data point
            M0 = cv2.moments(self.p0)
            # find the centroid
            try:
                self.c0 = int(M0["m10"]/M0["m00"]),int(M0["m01"]/M0["m00"])
            except ZeroDivisionError:
                self.c0 = None
            # exit early otherwise the same frame will be compared and this can cause
            # errors
            return frame

        # update colors if set
        if self.changeCols:
            self.shuffleColors()

        # calculate optical flow
        # passing the original frame and the new frame for comparison
        # OpenCV error can sometimes happen. in that event, the frame is returned
        # time is only updated if the results were obtained
        try:
            self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
        except cv2.error:
            print("caught cv2 error")
            return frame

        # create a copy of frame to draw on
        draw = frame.copy()
        # if points were found
        if self.p1 is not None:
            if self.p1.shape[0]>0:
                # get time new data points were confirmed to be found
                self.__newTime = time.time()
                ## Select good points
                # new frame
                good_new = self.p1[st==1]
                # old frame
                good_old = self.p0[st==1]
                # draw the tracks
                # tracks are the history of good points found in the current set of results and the reference data
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    self.mask = cv2.line(self.mask, (a,b),(c,d), self.color[i].tolist(), 2)
                    cv2.circle(draw,(a,b),5,self.color[i].tolist(),-1)

                ## estimate velocity
                # set points as relative to centre
                # if the reference and new data are different sizes
                # uses the min number of points to perform a valid subtraction
                sz = min(self.p1.shape[0],self.p0.shape[0])
                # set centre of rotation as centre of image
                cr = [sz//2 for sz in frame.shape[:2][::-1]]
                # find points positions relative to point of rotation
                pc0 = self.p0-cr
                pc1 = self.p1-cr
                # find difference between found points
                diff = self.p1[:sz,:,:] - self.p0[:sz,:,:]
                # find the velocity of each component of each point
                # sometimes the time passed is zero which would cause a div 0 error
                # in that case, the velocity is set to None
                if (self.__newTime-self.__refTime)==0.0:
                    print("zero dur")
                    print(f"ref was updated {self.__ct==0}")
                    self.__v = None
                # if it's not None
                # calculate angular velocity of each point
                else:
                    vel = diff/(self.__newTime-self.__refTime)
                    # calculate angular velocity of each point
                    num = (pc1[:,:,0]*vel[:,:,1]) - (vel[:,:,0]*pc1[:,:,1])
                    den = (pc1[:,:,0]**2.0) + (pc1[:,:,1]**2.0)
                    self.__v = abs(num/den)

                ## estimate centroid of these points
                # find the moments of the new data points
                M1 = cv2.moments(self.p1)
                # find the centroid
                try:
                    self.c1 = int(M1["m10"]/M1["m00"]),int(M1["m01"]/M1["m00"])
                    temp = frame.copy()
                    cv2.drawMarker(temp,self.c1,(0,255,0),cv2.MARKER_DIAMOND,5,2)
                    cv2.imshow("centroid",temp)
                except ZeroDivisionError:
                    self.c1 = None

        # increment counter
        # used for controlling when to update the reference
        self.__ct +=1
        # return results
        return cv2.add(draw,self.mask)

    # estimate velocity of the object based on the array of optic flow points
    # result is processed by passing to proc function
    def estVel(self,proc):
        if self.__v is None:
            return None
        else:
            return proc(self.__v)

# flow histories
uhist = np.empty((0,2))
vhist = np.empty((0,2))
# attempt at estimating velocity about centre
ahist = np.empty((0,))
class GFTracker:
    def __init__(self,params=None):
        if params is None:
            self.params = (0.5, 3, 16, 3, 5, 1.2, 0)
        else:
            self.params = None
        # reference frame
        self.prvs = None
        # timestamps for estimating "overall" angular velocity
        self.ots = time.time()
        # previous velocity estimate using centroid method
        self.oldVTest = 0.0
        # current estimate of centroid
        self.cd = None
        ## flag indicating color scheme to be used
        self.useRed = False

    def update(self,frame):
        if not hasattr(self,'cimg'):
            # find centre of the image
            cimg = frame.shape[:2][::-1]
            self.cimg = [pt//2 for pt in cimg]

        #if this is the first time and the reference has not been set yet.
        # use current frame
        if self.prvs is None:
            # convert to gray scale
            self.prvs = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            # set reference a blank image
            #self.prvs = np.zeros(frame.shape[:2],frame.dtype)
            # create matrix of zeros based off frame
            self.hsv = np.zeros(frame.shape,dtype=frame.dtype)
            # set saturation channel to max
            self.hsv[...,1] = 255

        self.t = time.time()
        self.dur = self.t-self.ots
        # convert frame to gray scale
        nx = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        # find dense optical flow comparing current to previous
        # two channel array [u,v], change in x and y between frames
        self.flow = cv2.calcOpticalFlowFarneback(self.prvs,nx, None, *self.params)
        #print(self.flow.any())
        # convert cartesian positions to polar positions
        self.mag, self.ang = cv2.cartToPolar(self.flow[...,0], self.flow[...,1])

        # if there's a previous flow
        # calculate change in angle between velocity components
        if hasattr(self,"oldflow"):
            self.oldmag,oldang = cv2.cartToPolar(self.oldflow[...,0],self.oldflow[...,1])
            # find difference in positions
            # at a specific point in the angular rotation
            # the angle flips its sign causing a large jump and an incorrect velocity value
            # to get an accurate velocity we need to find the absolute difference between the two
            diffAng = np.abs(np.abs(self.ang)-np.abs(oldang))
            # estimate velocity
            self.velAng = diffAng/self.dur
        # if oldflow does not exist then neither does the velocity matrix
        else:
            self.velAng = np.zeros(self.ang.shape,self.ang.dtype)
        ## convert values to colormap
        # direction is hue, angular direction
        if self.useRed:
            self.hsv[...,0]=0
        else:
            self.hsv[...,0] = self.ang*180.0*(np.pi**-1.0)*0.5
        # magnitude of travel is value
        self.hsv[...,2] = cv2.normalize(self.mag,None,0,255,cv2.NORM_MINMAX)
        # convert hsv results to to BGR so they can be saved
        rgb = cv2.cvtColor(self.hsv,cv2.COLOR_HSV2BGR)
        # save previous flow
        self.oldflow = self.flow.copy()
        # use current points as reference
        self.prvs[:] = nx
        self.ots = self.t
        # return drawn results
        return rgb

    # estimate angular velocity of the collective set of points
    # by processing them with a passed meth
    def estimateVel(self,proc=np.mean):
        # flow matrix is a two channel matrix containing the est velocity for x and y
        # change position to be relative to
        # it's an array of different points
        #return proc(((self.flow[...,0]**2.0) + (self.flow[...,1]**2.0))**0.5)
        return proc(self.velAng)

    # estimate angular velocity by comparing the centroid of the flow
    # of previous and past frames
    def estVelCentroid(self,printAngles=False):
        if not hasattr(self,'oldmag'):
            return 0.0
        ## second method centroid
        # normalize old magnitude matrix into an image
        oldFImg = np.zeros(self.oldmag.shape,np.uint8)
        oldFImg = cv2.normalize(self.oldmag,oldFImg,127,255,cv2.NORM_MINMAX)
        #print(oldFImg.any())
        # find moment
        M = cv2.moments(oldFImg,binaryImage=True)
        # find centroid
        # if a divide by zero error occurs
        # return the old velocity
        try:
            ocd = int(M["m10"] / M["m00"]),int(M["m01"] / M["m00"])
        except ZeroDivisionError:
            return self.oldVTest
        # change to be relative to centre
        ocd = [cimgpt-cdpt for cimgpt,cdpt in zip(self.cimg,ocd)]
        # find angle
        t0 = cv2.cartToPolar(ocd[0],ocd[1])[1][0][0]
        #t0 = np.arctan2(ocd[1],ocd[0])
        # normalize new magnitude matrix
        newFImg = np.zeros(self.mag.shape,np.uint8)
        newFImg = cv2.normalize(self.mag,newFImg,0,255,cv2.NORM_MINMAX)
        # threshold to make non-zero values 255
        newFImg = cv2.threshold(newFImg,127,255,cv2.THRESH_BINARY)[1]
        #cv2.imshow("flow threshold",newFImg)
        # find centroid
        M = cv2.moments(newFImg,binaryImage=True)
        try:
            ncd = int(M["m10"] / M["m00"]),int(M["m01"] / M["m00"])
        except ZeroDivisionError:
            return self.oldVTest
        # save new centroid for plotting
        self.cd = ncd
        # change to be relative to centre
        ncd = [cimgpt-cdpt for cimgpt,cdpt in zip(self.cimg,ncd)]
        # find angle
        t1 = cv2.cartToPolar(ncd[0],ncd[1])[1][0][0]
        #t1 = np.arctan2(ncd[1],ncd[0])
        # find difference between angles
        # the difference is absoluted due to the large changes in angles
        diffA = abs(abs(t1)-abs(t0))
        # if the difference between the angles is
        # more than pi/2 (90 degs)
        #print(f"old: {t0},{t1},{diffA}")
        if diffA>=(np.pi*0.5):
            maxT = max(t1,t0)
            # find which angle is larger
            maxTi = [t0,t1].index(maxT)
            # find which angle segment the largest angle is closest to
            segments = [0.5*np.pi,np.pi,2.0*np.pi]
            diff = [abs(maxT-seg) for seg in segments]
            di = diff.index(min(diff))
            # adjust angle
            if maxTi==0:
                t0 = abs(segments[di]-t0)
            elif maxTi==1:
                t1 = abs(segments[di]-t1)
            # recalculate difference between angles
            diffA = abs(t1-t0)
        # estimate angular velocity as change in angle over time
        velTest = diffA/self.dur
        if printAngles:
            print(t0,t1,diffA,velTest)
        # update previous velocity value
        self.oldVTest = velTest
        # return result
        return velTest

def GF_dot():
    # define rotating dot manager
    rd = RD.RotatingDot(15,100)
    # define rotating animation manager
    # update at a rate of 60 FPS
    anim = RD.RotatingAnimator(rd,dur=60.0**-1.0,dangle=np.deg2rad(5.0))
    # define tracker class
    # use default parameters
    of = GFTracker()
    # start animation
    anim.start()
    # loop
    while True:
        # get frame from animator
        frame = anim.get()
        # add noise
        frame = cv2.blur(frame,(5,5))
        # invert
        # most algorithms prefer black background
        frame = cv2.bitwise_not(frame)
        # process
        res = of.update(frame)
        # show results
        cv2.imshow("frame",frame)
        cv2.imshow("results",res)
        # break on escape
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    # cleanup
    anim.stop()
    cv2.destroyAllWindows()

    return of.flow

def GF_dot_vel(recRes=False):
    # define rotating dot manager
    rd = RD.RotatingDot(15,100)
    d = rd.getDist()
    # define rotating animation manager
    # update at a rate of 60 FPS
    anim = RD.RotatingAnimator(rd,dur=60.0**-1.0,dangle=np.deg2rad(5.0))
    # define tracker class
    # use default parameters
    of = GFTracker()
    procVel = np.min
    # velocity matrix
    vel = np.zeros((1,))
    # timestamp
    fps = 30
    # sampling period
    T = fps**-1.0
    # setup recorder
    if recRes:
        rec = cv2.VideoWriter("rotating_dot_opticflow_GF.avi",cv2.VideoWriter_fourcc(*'MJPG'),int(T**-1.0),rd.getFrameSize()[:2][::-1],True)
        if not rec.isOpened():
            print("failed to open recorder")
            rec.release()
    # timestamps
    t = time.time()
    ots = t
    # start animation
    anim.start()
    # loop
    while True:
        t = time.time()
        if (t-ots)>=T:
            # get frame from animator
            frame = anim.get()
            # add noise
            frame = cv2.blur(frame,(5,5))
            # invert
            frame = cv2.bitwise_not(frame)
            # process
            res = of.update(frame)
            if recRes:
                if rec.isOpened():
                    rec.write(res)
            # get velocity
            vel = np.append(vel,of.estimateVel(proc=procVel))
            # show results
            cv2.imshow("frame",frame)
            cv2.imshow("results",res)
            ots = t
        # break on escape
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    # cleanup
    anim.stop()
    if recRes:
        rec.release()

    # destroy opencv windows
    cv2.destroyAllWindows()

    # display estimated velocity data
    f,ax = plt.subplots()
    ax.plot(vel)
    ax.set(xlabel="Data Index",ylabel="Velocity (rad/s)",title=f"Estimated {procVel.__name__.upper()} Angular Velocity using Gunnar-Farneback Optic Flow")
    f.savefig("rotating-dot-opticflow-GF-{procVel.__name__.lower()}-vel.png")
    plt.show()

def GF_dot_vel_cent(recRes=False):
    # define rotating dot manager
    rd = RD.RotatingDot(15,100)
    d = rd.getDist()
    # define rotating animation manager
    # update at a rate of 60 FPS
    anim = RD.RotatingAnimator(rd,dur=60.0**-1.0,dangle=np.deg2rad(5.0))
    # define tracker class
    # use default parameters
    of = GFTracker()
    # set colorscheme flag
    of.useRed = False
    # velocity matrix
    vel = np.zeros((1,))
    tvec = np.zeros((1,))
    tang = np.zeros((1,))
    # timestamp
    fps = 30
    # sampling period
    T = fps**-1.0
    # setup recorder
    if recRes:
        rec = cv2.VideoWriter("rotating_dot_opticflow_centroid_GF.avi",cv2.VideoWriter_fourcc(*'MJPG'),int(T**-1.0),rd.getFrameSize()[:2][::-1],True)
        if not rec.isOpened():
            print("failed to open recorder")
            rec.release()
    # timestamps
    t = time.time()
    ots = t
    tstart=t
    # start animation
    anim.start()
    # loop
    while True:
        t = time.time()
        if (t-ots)>=T:
            tvec = np.append(tvec,t-tstart)
            # get frame from animator
            frame = anim.get()
            # add noise
            frame = cv2.blur(frame,(5,5))
            # invert
            frame = cv2.bitwise_not(frame)
            # process
            res = of.update(frame)
            # get velocity
            vel = np.append(vel,of.estVelCentroid(False))
            #tang = np.append(tang,np.mean(of.ang))
            if of.cd is not None:
                tang = np.append(tang,of.ang[of.cd[1],of.cd[0]])
            else:
                tang = np.append(tang,0.0)
            # draw centroid
            if of.cd is not None:
                #print(of.cd)
                # draw centroid
                cv2.drawMarker(res,of.cd,(255,0,0),cv2.MARKER_DIAMOND,10,2)
            if recRes:
                if rec.isOpened():
                    rec.write(res)
            # show results
            cv2.imshow("frame",frame)
            cv2.imshow("results",res)
            ots = t
        # break on escape
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    # cleanup
    anim.stop()
    if recRes:
        rec.release()

    # destroy opencv windows
    cv2.destroyAllWindows()

    # display estimated velocity data
    f,ax = plt.subplots()
    ax.plot(tvec,vel)
    ax.set(xlabel="Time (s)",ylabel="Velocity (rad/s)",title=f"Estimated Angular Velocity using Gunnar-Farneback \nOptic Flow and Flow Centroids")
    if recRes:
        f.savefig("rotating-dot-opticflow-GF-centroid-vel.png")

    f2,ax2 = plt.subplots()
    ax2.plot(tang)
    plt.show()
    return tvec,vel,tang

def gf_vel_ball_tracker():
    print("running GF with ball tracker")
    # define rotating dot manager
    rd = RD.RotatingDot(15,100)
    d = rd.getDist()
    # define rotating animation manager
    # update at a rate of 60 FPS
    anim = RD.RotatingAnimator(rd,dur=60.0**-1.0,dangle=np.deg2rad(5.0))
    # ball tracker to identify bounding box of flow
    # initialize target as black
    # will be updated later
    ball = BallTracker((0,0,0),(0,0,255))
    # define tracker class
    # use default parameters
    of = GFTracker()
    procVel = np.mean
    # velocity matrix
    vel = np.zeros((1,))
    # start animation
    anim.start()
    # loop
    while True:
        # get frame from animator
        frame = anim.get()
        # add noise
        frame = cv2.blur(frame,(5,5))
        # invert
        frame = cv2.bitwise_not(frame)
        # process
        res = of.update(frame)
        # update targets of ball tracker with min max values
        #nlow = (0,0,0)
        #nhigh = tuple(BGR2HSV(res.max()))
        #print(nlow,nhigh)
        #ball.updateTargets(tlow=nlow,thigh=nhigh)
        # pass frame
        #bres,found = ball.track(res)
        bres = ball.debugTrack(res)
        # get velocity
        vel = np.append(vel,ball.getV())
        # show results
        cv2.imshow("frame",frame)
        cv2.imshow("results",res)
        cv2.imshow("ball",bres)
        # break on escape
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    # cleanup
    anim.stop()
    cv2.destroyAllWindows()

    f,ax = plt.subplots()
    ax.plot(vel)
    ax.set(xlabel="Data Index",ylabel="Velocity (rad/s)",title=f"Estimated {procVel.__name__.upper()} Angular Velocity using Gunnar-Farneback Optic Flow")
    f.savefig("rotating-dot-opticflow-GF-{procVel.__name__.lower()}-vel.png")
    plt.show()

# using lucas kanade algorithm with rotating dot
def LK_dot():
    # define rotating dot manager
    rd = RD.RotatingDot(15,100)
    # define rotating animation manager
    # update at a rate of 60 FPS
    anim = RD.RotatingAnimator(rd,dur=60.0**-1.0,dangle=np.deg2rad(5.0))
    # define tracker class
    # use default parameters
    of = LKTracker()
    # start animation
    anim.start()
    # loop
    while True:
        # get frame from animator
        frame = anim.get()
        # invert
        #frame = cv2.bitwise_not(frame)
        # add noise
        frame = cv2.blur(frame,(5,5))
        # process
        res = of.update(frame)
        # show results
        cv2.imshow("frame",frame)
        cv2.imshow("results",res)
        # break on escape
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    # cleanup
    anim.stop()
    cv2.destroyAllWindows()
    return of.p1

# use lucas kanade algorithm /w rotating dot and estimate velocity
def LK_dot_vel(recRes = False):
    # define rotating dot manager
    rd = RD.RotatingDot(15,100)
    # get distance from center
    d = rd.getDist()
    # define rotating animation manager
    # update at a rate of 60 FPS
    anim = RD.RotatingAnimator(rd,dur=60.0**-1.0,dangle=np.deg2rad(5.0))
    # estimate angular velocirt of rotation
    print(f"target vel (rad/s): {anim.getAngleChange()/anim.getDuration()}")
    # simulated fps of the camera
    fps = 30
    # sampling period
    T = (2.0*fps)**-1.0
    # recorder
    if recRes:
        rec = cv2.VideoWriter("rotating_dot_opticflow_LK.avi",cv2.VideoWriter_fourcc(*'MJPG'),int(T**-1.0),rd.getFrameSize()[:2][::-1],True)
        if not rec.isOpened():
            print("failed to open recorder")
            rec.release()
    # define tracker class
    # use default parameters
    of = LKTracker()
    # flag to control color scheme
    of.useRed = False
    # create vector to hold velocity
    vel = np.empty((0,))
    angle = np.empty((0,))
    avgwin = np.empty((0,))
    win = []
    winsize = 10
    # function used to process velocities
    procVel = np.mean
    # timestamps to control sampling
    ots = time.time()
    # start animation
    anim.start()
    # loop
    while True:
        # get time
        t = time.time()
        # get frame from animator
        frame = anim.get()
        # invert
        frame = cv2.bitwise_not(frame)
        # add noise
        frame = cv2.blur(frame,(5,5))
        cv2.imshow("frame",frame)
        # if the set sampling period has passed
        # pass frame to tracker
        # get velocity
        if (t-ots)>=T:
            ots = t
            # calculate optic flow
            res = of.update(frame)
            # write to file
            if recRes:
                if rec.isOpened():
                    rec.write(res)
            # show results
            cv2.imshow("results",res)
            # estimate velocity of object by processing the velocity of each of the found
            # points using the given proc function
            v = of.estVel(proc=procVel)
            # if the velocity is not None
            # add to vector
            if v is not None:
                vel = np.append(vel,v)
                win.append(v)
            # else use the previous value
            else:
                vel = np.append(vel,vel[-1])
                win.append(vel[-1])
            if len(win)==winsize:
                avgwin = np.append(avgwin,np.mean(win))
                win.clear()
            angle = np.append(angle,rd.getCurrAngle())
        # break on escape
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    # cleanup
    anim.stop()
    # release recorder
    if recRes:
        rec.release()
    # destroy opencv windows
    cv2.destroyAllWindows()

    # plot results
    f,ax = plt.subplots()
    ax.plot(vel)
    ax.set(xlabel="Data Index",ylabel="Velocity (rad/s)",title=f"Estimated {procVel.__name__.upper()} Angular Velocity using Lucas-Kanade Optic Flow")

    f2,ax2 = plt.subplots(nrows=2,sharex=True)
    ax2[0].plot(vel)
    ax2[1].plot(angle)
    ax2[0].set(xlabel="Data Index",ylabel="Velocity (rad/s)")
    ax2[1].set(xlabel="Data Index",ylabel="Angular Position (rads)")
    f2.suptitle(f"Estimated {procVel.__name__.upper()} Angular Velocity and actual Angular Position\n from Lucas-Kanade Optic Flow")

    a,b = plt.subplots()
    b.plot(avgwin)
    b.set(xlabel="Data Index",ylabel="Average Velocity (rad/s",title=f"Moving Averaged Estimated Angular Velocity\n using Lucas-Kanade Optic Flow. Window Size: {winsize}")
    plt.show()

## lucas kanade algorithm
# tracks movement for some points in the frame
def lucas_kanade():
    cap = cv2.VideoCapture(0)

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret,old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    # find some corners to track as defined by feature params
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    rs = 5
    ct = 0
    while True:
        # get data and convert to gray scale
        ret,frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        # passing the original frame and the new frame for comparison
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        ## Select good points
        # new frame
        good_new = p1[st==1]
        # old frame
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        ct +=1
        if (ct%rs)==0:
            ct =0
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)

    cv2.destroyAllWindows()
    cap.release()

## gunner farneback algorithm
# tracks all points in the frame
def gunner_farneback():
    # setup camera
    cap = cv2.VideoCapture(0)
    # read data
    ret, frame1 = cap.read()
    # convert to gray scale
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    # create matrix of zeros based off frame
    hsv = np.zeros_like(frame1)
    # set saturation channel to max
    hsv[...,1] = 255

    while True:
        # get next frame
        ret, frame2 = cap.read()
        # convert to gray scale
        nx = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        # find dense optical flow
        # two channel array [u,v]
        flow = cv2.calcOpticalFlowFarneback(prvs,nx, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # convert cartesian positions to polar positions
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        # convert values to color
        # direction is hue, angular direction
        #hsv[...,0] = ang*180/np.pi/2
        hsv[...,0]=0
        # magnitude of travel is value
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        # convert hsv results to RGB to BGR so they can be saved
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        # show result
        cv2.imshow('frame2',rgb)
        # handle key presses
        # esc to exit
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # s to save to local directory
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',rgb)
        prvs = nx

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set of classes for implementing Optical Flow Analysis for the purpose of estimating the velocity of an object")
    parser.add_argument("--lk",dest="lkonly",action="store_true",help="Flag to trigger the running of Lucas-Kanade optic flow model using the default camera")
    parser.add_argument("--gf",dest="gfonly",action="store_true",help="Flag to trigger the running of the Gunnar-Farneback optic flow model using the default camera")
    parser.add_argument("--lkdot",dest="lkdot",action="store_true",help="Flag to trigger running the Lucas-Kanade Model using the rotating dot animation as the source. Useful in debugging")
    parser.add_argument("--gfdot",dest="gfdot",action="store_true",help="Flag to trigger running the Gunnar-Farneback model with the rotating dot animation as source")
    parser.add_argument("--lkdotvel",dest="lkdotvel",action="store_true",help="Flag to trigger the running of the Lucas-Kanade model to track a rotating dot animation and estimate velocity")
    parser.add_argument("--gfdotvel",dest="gfdotvel",action="store_true",help="Flag to trigger the running of the Gunnar-Farneback model to track a rotating dot and estimate the rotating velocity")
    parser.add_argument("--gfdotcent",dest="gfdotcent",action="store_true",help="Flag to trigger running the Gunnar-Farneback model on a rotating dot animation. Differs from gfdot as the velocity is based off tracking the centroid of the largest object.")
    parser.add_argument("--rec",dest="rec",action="store_true",help="Flag to record the results")

    args = parser.parse_args()
    if args.lkonly:
        lucas_kanade()
    elif args.gfonly:
        gunner_farneback()
    elif args.lkdot:
        LK_dot()
    elif args.gfdot:
        GF_dot()
    elif args.lkdotvel:
        LK_dot_vel(args.rec)
    elif args.gfdotvel:
        GF_dot_vel(args.rec)
    elif args.gfdotcent:
        GF_dot_vel_cent(args.rec)
