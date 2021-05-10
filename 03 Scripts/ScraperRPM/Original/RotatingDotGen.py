import cv2
import numpy as np
import time
import threading
##
# class for defining an occlusion that can be applied to an image
class Occlusion:
    # constructor where the user can pass the set of points defining the shape
    # and the color of the shape
    def __init__(self,pts=[],col=(255,255,255)):
        self.pts = pts
        self.col = col
    # method to draw the defined shape on the given frame
    # returns the results of drawing on the frame
    def draw(self,frame):
        return cv2.fillConvexPoly(frame,np.array(self.pts,np.int32),self.col)
        
# class for building occlusions of a known defined shape
# meant to be a quicker method for defining the points used in Occlusion class
# e.g. quick define a arc or circle to act as an occlusion
class OcclusionFactory:
    def __init__(self):
        return
    # build an occlusion of the target type passing along the necessary parameters to define it
    # constructs the list of points to define the shape and the builds the Occlusion class
    # returns the constructed occlusion class
    def build(self,t,col=(0,0,0),**kwargs):
        # if the user wants an arc to be drawn
        if t.lower() == "arc":
            # radius of the arc must be defined
            if "radius" not in kwargs:
                raise ValueError("Radius not defined")
                return
            # if the angle the arc is to be rotated to is not defined
            # set to 0
            if "angle" not in kwargs:
                kwargs["angle"] = 0
            # if the starting angle of the arc is not defined
            # set to zero
            if "arcStart" not in kwargs:
                kwargs["arcStart"] = 0
            # check if the ending angle of the arc is not defined
            # if not raise error
            if "arcEnd" not in kwargs:
                raise ValueError("Maximum angle of the arc is not defined")
                return
            # use ellipse2poly to define the arc portion
            # assume that the user has given the appropriate arguments
            # to define the ellipse
            pts = cv2.ellipse2Poly(kwargs["center"],(kwargs["radius"],)*2,kwargs["angle"],kwargs["arcStart"],kwargs["arcEnd"],delta=5)
            # add a point to define the centre
            pts = pts.tolist()
            pts.append(kwargs["center"])
        # if the user wants an ellipse to be drawn
        if t.lower() == "ellipse":
            # if the user does not specify thhe arcStart and arcEnd parameters
            # it is assumed that they want a complete ellipse and therefore
            # sets the paramters as such
            # if box is a keyword then the box can structure can be passed as is
            if "box" not in kwargs:
                if "arcStart" not in kwargs:
                    kwargs["arcStart"] = 0

                if "arcEnd" not in kwargs:
                    kwargs["arcEnd"]=360
                    
            # build the points
            if "delta" not in kwargs:
                pts = cv2.ellipse2Poly(**kwargs,delta=5)
            else:
                pts = cv2.ellipse2Poly(**kwargs)
        # wants a circle to be drawn
        elif t.lower() == "circle":
            # as a circle is an ellipse with the same axes sizes
            # combine into single target parameter radius
            # check if one was passed
            # if not raise exception and exit
            if "radius" not in kwargs:
                raise ValueError("Radius not defined")
                return
            if "delta" not in kwargs:
                pts = cv2.ellipse2Poly(kwargs["center"],(kwargs["radius"],)*2,0,0,360,5)
            else:
                pts = cv2.ellipse2Poly(kwargs["center"],(kwargs["radius"],)*2,0,0,360,kwargs["delta"])
        # define a rotated box 
        elif (t.lower() == "box"):
            # box is defined as in opencv
            # top left corner followed by size
            if "tl" not in kwargs:
                raise ValueError("Missing top left corner of bounding area")
                return

            if "size" not in kwargs:
                raise ValueError("Size of bounding area not defined")
                return

            # if no angle was defined then assume not rotated
            if "angle" not in kwargs:
                shiftx = 0
                shifty = 0
            # if an angle was defined
            # calculate the change in the x and y positions that needs to be applied
            else:
                # calculate shift in points if the rectangle is rotated about the top left corner
                shiftx = int(kwargs["size"][0]*np.cos(kwargs["angle"]))
                shifty = int(kwargs["size"][1]*np.size(kwargs["angle"]))
            # define corner points with shift
            pts = [kwargs["tl"],[kwargs["tl"][0]+kwargs["size"][0]+shiftx,kwargs["tl"][1]+shifty],
                   [kwargs["tl"][0]+shiftx,kwargs["tl"][1]+kwargs["size"][1]+shifty],
                   [kwargs["tl"][0]+kwargs["size"][0]+shiftx,kwargs["tl"][1]+kwargs["size"][1]+shifty]]
            
        return Occlusion(pts,col)
        
# class for generating a rotating dot
class RotatingDot:
    def __init__(self,sz,dist,start_angle=0.0,w=480,h=480,addNoise=False):
        # check distance
        if (dist>(h//2)) or (dist>(w//2)):
            raise ValueError("Distance is too large! Dot will be drawn outisde of the image")
            return
        # initialize current image as blank image
        self.__img = np.ones((h,w,3),dtype=np.uint8)*255
        self.__bk = (255,255,255)
        # reference point for rotation
        self.__ref = (w//2,h//2)
        # store distance from rference
        self.__dist = dist
        # starting dot position
        self.__dotStart = (w//2,h//2-dist)
        # initialize current dot position
        self.__dot = self.__dotStart
        # size of the dot
        self.__dotSz = sz
        # current angle of rotation
        # set to desired starting angle
        self.__rot = start_angle
        self.__aRot = start_angle
        # color of the drawn dot
        self.dotCol = (0,0,0)
        # collection of occlusions
        # draw dot at starting position
        self.rotate(start_angle,False)

    # method to get frame size
    def getFrameSize(self):
        return self.__img.shape

    # get current angle of rotation
    def getCurrAngle(self):
        return self.__rot

    # get set distance from centre of rotation
    def getDist(self):
        return self.__dist

    # get copy of current image
    def getImg(self):
        return self.__img.copy()

    # change reference point for rotation
    def changeRef(self,x,y):
        self.__ref = (x,y)

    # change the size of the dot
    def changeDotSize(self,sz):
        if sz<=0:
            raise ValueError("Dot size cannot be zero or negative!")
            return
        else:
            self.__dotSz = sz

    # change the size of the generated image
    def changeImgSize(self,shape):
        # reinitialize image
        self.__img = np.zeros(shape,self.__img.dtype)
        # set as target background color
        self.__img[:] = self.__bk
        # redraw dot
        cv2.circle(self.__img,self.__dot,self.__dotSz,self.dotCol,-1,8)
        
    # rotate the dot by a certain amount, default radians
    # flag indicates if angle is degrees or not
    def rotate(self,d,deg=False):
        angle = d
        # if degrees convert to radians
        if deg:
            angle *= np.pi*180**-1
        # get distances from reference
        dx = self.__dot[0]-self.__ref[0]
        dy = self.__dot[1]-self.__ref[1]
        # rotate
        # rotation matrix
        nx = dx*np.cos(angle)-dy*np.sin(angle)
        ny = dx*np.sin(angle)+dy*np.cos(angle)
        # update dot position
        self.__dot = (int(nx+self.__ref[0]),int(ny+self.__ref[1]))
        # create blank image
        mask = np.zeros(self.__img.shape,self.__img.dtype)
        # fill with background color
        mask[:] = self.__bk
        # draw dot
        cv2.circle(mask,self.__dot,self.__dotSz,self.dotCol,-1,8)
        # update img
        self.__img[:,:,:] = mask
        # update actual rotation angle
        # based off converting dot position relative to centre ?
        # from cartesian to polar coordinates
        self.__rot = cv2.cartToPolar(nx,ny)[1][0][0]

# thread that executes a function every period
class RepeatTimedThread(threading.Thread):
    # initialize passing the interval between calls, the target function and any required arguments
    def __init__(self,interval,target,*args,**kwargs):
        # initialize thread
        threading.Thread.__init__(self)
        # clear flag for daemon thread
        self.daemon=False
        # set the stopping condition as an event
        self.stopped = threading.Event()
        # store 
        self.interval = interval
        self.execute = target
        self.args = args
        self.kwargs = kwargs

    # stopping method
    def stop(self):
        # sets the event acting as a stopping condition
        self.stopped.set()
        # waits for the thread to stop
        self.join()

    # running method
    # overrides the normal run method of the Thread Class
    def run(self):
        # check state of stopped
        # checking after a set period of time
        while not self.stopped.wait(self.interval):
            self.execute(*self.args,**self.kwargs)

# class for generating rotating animations on a timer
class RotatingAnimator:
    def __init__(self,frameGen,dur,dangle):
        # initialize parameters
        self.__gen = frameGen
        # store time duration between updates
        self.__dur = dur
        # store change in angle each time
        self.__alpha = dangle
        # create worker to update the images
        # rotates the dot by a set amount after the set period of time
        self.__worker = RepeatTimedThread(dur,self.__gen.rotate,dangle)

    # get the next frame of the animation
    def get(self):
        return self.__gen.getImg()

    # start the animation
    def start(self):
        self.__worker.start()

    # stop the animation
    def stop(self):
        self.__worker.stop()

    # change the update rate of the worker
    def changeDuration(self,dur):
        self.__dur = dur
        self.__worker.stop()
        self.__worker.interval = dur

    # get the current angle the dot is rotated by
    def getAngleChange(self):
        return self.__alpha

    # get the current set time between updates
    def getDuration(self):
        return self.__dur

    # change the angle the dot is updated by
    def changeAngleChange(self,dangle,deg=False):
        da = dangle
        # if degrees convert to radians
        if deg:
            da *= np.pi*180**-1
        # update stored angle change
        self.__alpha = da
        # rebuild worker with new argument
        self.__worker = RepeatTimedThread(dur,self.rotate,dangle)

def dotRecord():
    import matplotlib.pyplot as plt
    # define rotating dot manager
    rd = RotatingDot(15,100)
    # define rotating animator
    anim = RotatingAnimator(rd,dur=60.0**-1.0,dangle=np.deg2rad(10.0))
    # print set change on rotation
    tch = anim.getAngleChange()
    # print set change in angle per second
    print("Target change: ",tch," rads")
    # print target velocity
    tv = anim.getAngleChange()/anim.getDuration()
    print(f"Target velocity: {tv} rad/s")
    # start animation
    anim.start()
    # old rotation position to update
    tstart = time.time()
    # old time stamp
    ots = tstart
    dur = anim.getDuration()
    # setup recorder
    rec = cv2.VideoWriter("rotating_dot_animation.avi",cv2.VideoWriter_fourcc(*'MJPG'),int(dur**-1.0),rd.getFrameSize()[:2][::-1],True)
    if not rec.isOpened():
        print("failed to open recorder")
        rec.release()
    # loop getting the new frames and showing whem
    while True:
        t = time.time()
        # update 
        if (t-ots)>=dur:
            # get new image and display
            frame = anim.get()
            # record if setup
            if rec.isOpened():
                rec.write(frame)
            cv2.imshow("Frame",frame)
            # update old position
            ots = t
        # wait for key press
        key = cv2.waitKey(1)
        # break on ESC
        if key == 27:
            break
    # stop animation
    anim.stop()
    cv2.destroyAllWindows()
    rec.release()

def dotDebug(recRes=False):
    import matplotlib.pyplot as plt
    # define rotating dot manager
    rd = RotatingDot(15,100)
    # define rotating animator
    anim = RotatingAnimator(rd,dur=60.0**-1.0,dangle=np.deg2rad(10.0))
    # print set change on rotation
    tch = anim.getAngleChange()
    # print set change in angle per second
    print("Target change: ",tch," rads")
    # print target velocity
    tv = anim.getAngleChange()/anim.getDuration()
    print(f"Target velocity: {tv} rad/s")
    # start animation
    anim.start()
    # old rotation position to update
    tstart = time.time()
    orot = 0.0
    # old time stamp
    ots = 0.0
    dur = anim.getDuration()
    # velocity vector
    vel = np.zeros((1,2))
    # angle vector
    ang = np.zeros((1,2))
    # setup recorder
    if recRes:
        rec = cv2.VideoWriter("rotating_dot_animation.avi",cv2.VideoWriter_fourcc(*'MJPG'),60,rd.getFrameSize()[:2][::-1],True)
        if not rec.isOpened():
            print("failed to open recorder")
            rec.release()
    # loop getting the new frames and showing whem
    while True:
        t = time.time()
        # get new image and display
        frame = anim.get()
        # record if setup
        if recRes:
            if rec.isOpened():
                rec.write(frame)
        cv2.imshow("Frame",frame)
        ## print change in rotational position
        if (t-ots)>=dur:
            # get current angular position
            rot = rd.getCurrAngle()
            # find difference
            #print(abs(rot-orot),tch)
            #print(f"{abs(rot-orot)} rad,{t-ots} s,{abs(rot-orot)/(t-ots)} rad/s)")
            vel = np.concatenate((vel,[[t-tstart,abs(rot-orot)/(t-ots)]]),axis=0)
            ang = np.concatenate((ang,[[t-tstart,rot]]),axis=0)
            # update old position
            orot = rot
            ots = t
        # wait for key press
        key = cv2.waitKey(1)
        # break on ESC
        if key == 27:
            break
    # stop animation
    anim.stop()
    if recRes:
        rec.release()
    
    # destroy windows
    cv2.destroyAllWindows()
    # plot the collected velocity and target values against the target
    fv,axv = plt.subplots()
    axv.plot(vel[:,0],vel[:,1],'b-',vel[:,0],np.full(vel.shape[0],tv),'r--')
    axv.set(xlabel="Time (s)",ylabel="Estimated rotation velocity (rad/s)",title="Estimated Angular Velocity of Animation versus Target")
    fv.legend(("Est. Vel","Target Vel"))

    fa,axa = plt.subplots()
    axa.plot(ang[:,0],ang[:,1],'b-')
    axa.set(xlabel="Time (s)",ylabel="Angular Position of Dot (rads)",title="Angular Position of Rotating Dot")
    plt.show()

def dot_occlusion_test():
    # define rotating dot manager
    rd = RotatingDot(15,100)
    # define rotating animator
    anim = RotatingAnimator(rd,dur=60.0**-1.0,dangle=np.deg2rad(10.0))
    # print set change on rotation
    tch = anim.getAngleChange()
    # print set change in angle per second
    print("Target change: ",tch," rads")
    # print target velocity
    tv = anim.getAngleChange()/anim.getDuration()
    print(f"Target velocity: {tv} rad/s")
    # get information about rotating dot to design occlusion
    dist = rd.getDist()
    H,W,_ = rd.getFrameSize()
    # define occlusion
    fac = OcclusionFactory()
    # ellipse in the top right hand corner
    occ = fac.build("ellipse",col=(0,0,0),center=(int((W//2)+dist*0.9),int((H//2)-dist*0.9)),angle=-45,axes=(dist//2,dist))
    # old rotation position to update
    tstart = time.time()
    orot = 0.0
    # old time stamp
    ots = 0.0
    dur = anim.getDuration()
    # velocity vector
    vel = np.zeros((1,2))
    # angle vector
    ang = np.zeros((1,2))
    # start animation
    anim.start()
    # loop getting the new frames and showing whem
    while True:
        t = time.time()
        # get new image
        frame = anim.get()
        # add occlusion
        frame = occ.draw(frame)
        
        cv2.imshow("Frame",frame)
        ## print change in rotational position
        if (t-ots)>=dur:
            # get current angular position
            rot = rd.getCurrAngle()
            # find difference
            #print(abs(rot-orot),tch)
            #print(f"{abs(rot-orot)} rad,{t-ots} s,{abs(rot-orot)/(t-ots)} rad/s)")
            vel = np.concatenate((vel,[[t-tstart,abs(rot-orot)/(t-ots)]]),axis=0)
            ang = np.concatenate((ang,[[t-tstart,rot]]),axis=0)
            # update old position
            orot = rot
            ots = t
        # wait for key press
        key = cv2.waitKey(1)
        # break on ESC
        if key == 27:
            break
    # stop animation
    anim.stop()
    # destroy windows
    cv2.destroyAllWindows()
if __name__ == "__main__":
    #dotDebug(True)
    dotRecord()
    #dot_occlusion_test()
