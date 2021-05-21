import numpy as np
import cv2
import CameraStream
from PyImageSearchBallTracker import BallTracker,BGR2HSV
from cvOpticalFlow import LKTracker, GFTracker
from KalmanTracker import KalmanDotTracker, KalmanDotAccTracker
import argparse
import inspect
import time
import os

# function to construct all trackers defined by a build_tracker_* function
def build_tracker_all():
    # find all functions that start with build_tracker
    # avoids itself
    buildfns = [obj for name,obj in inspect.getmembers(sys.modules[__name__]) if (inspect.isfunction(obj) and name.startswith('build_tracker') and ("all" not in name))]
    # create list of trackers and return them
    return [f() for f in buildfns]

# function to construct a tracker based off lucas-kanade optic flow
def build_tracker_LK():
    oTrack = LKTracker()
    oTrack.useRed = False
    return oTrack

# function to construct a tracker based off gunnar farneback optic flow
def build_tracker_GF():
    oTrack = GFTracker()
    return oTrack
    
# function to construct tracker based off BallTracker method
def build_tracker_BallTracker(clow=(0,0,0),cupper=(255,255,255)):
    tracker = BallTracker(clow,cupper,"RGB")
    return tracker
    
# function to construct Kalman Tracker using the BallTracker to supply measurement
def build_tracker_Kalman(clow=(0,0,0),cupper=(255,255,255),T=60**-1.0):
    # define ball tracker
    btrack = BallTracker(clow,cupper,"RGB")
    # build kalman filter using balltracker as measurement method
    tracker = KalmanDotTracker(tracker=btrack,T=60**-1.0)
    return tracker

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # arguments controlling where to save the footage to
    parser.add_argument("-rec",dest="rec",type=str,help="File path for output images")
    parser.add_argument("-cam",dest="cam",type=str,help="File path for the input camera footage")
    parser.add_argument("-data",dest="data",type=str,help="File path for the estimated position and velocity from each tracker")
    # flags for stopping certain operations
    parser.add_argument("-no-track",dest="noTrack",action="store_true",help="Flag to not use any trackers. Used to test camera feed",default=False)
    parser.add_argument("-no-display",dest="noDisplay",action="store_true",help="Flag to not display anything",default=False)
    # arguments controlling which method to use
    parser.add_argument("-all",dest="all",action="store_true",help="Estimate using ALL methods at default settings")
    parser.add_argument("-lk",dest="lk",action="store_true",help="Estimate using  Lucas-Kanade method")
    parser.add_argument("-gf",dest="gf",action="store_true",help="Estimate using  Gunnar-Farneback method")
    parser.add_argument("-kal",dest="kal",action="store_true",help="Estimate using  Kalman Filter")
    parser.add_argument("-ball",dest="ball",action="store_true",help="Estimate using  BallTracker class")
    # argument for setting tracker parameters
    parser.add_argument("-clow",dest="clow",nargs="+",type=int,help="Lower color value threshold used with BallTracker and KalmanTracker. Default (0,0,0)",default=[0,0,0])
    parser.add_argument("-chigh",dest="chigh",nargs="+",type=int,help="Higher color value threshold used with BallTracker and KalmanTracker, Default (255,255,255)",default=[255,255,255])
    parser.add_argument("-period","-t",dest="t",type=float,help=f"Update period for the Kalman Filter. Based off frame rate of the camera capture. Default 60 FPS {60**-1.0}",default=60.0**-1.0)
    
    args = parser.parse_args()
    # function to check paths
    def checkPath(path):
        # get folder from path
        folder = os.path.dirname(path)
        ## check if the folder exists
        # if there is no folder, then assume local and return True
        if not folder:
            return False
        # else check if the target folder exists
        else:
            return not os.path.isdir(folder)

    # check if lower and upper color has been correctly defined
    if ((len(args.clow)<3) or (len(args.clow)>3)):
        raise ValueError(f"Lower color threshold is the wrong size. Given size is {len(args.clow)}")
    if ((len(args.chigh)<3) or (len(args.chigh)>3)):
        raise ValueError(f"Higher color threshold is the wrong size. Given size is {len(args.chigh)}")
    # check if values are within range
    # colors are defined as RGB
    if min(args.clow)<0 or max(args.clow)>255:
        raise ValueError(f"Lower color threshold is outside permissible range! {args.clow}")
    if min(args.chigh)<0 or max(args.chigh)>255:
        raise ValueError(f"Higher color threshold is outside permissible range! {args.chigh}")
    # check if target time period
    if args.t<0.0:
        raise ValueError("Target time period for KalmanFilter cannot be negative!")

    ## build trackers
    trackers = []
    # get stated trackers
    if args.all:
        trackers = build_tracker_all()
    else:
        if args.lk:
            trackers.append(build_tracker_LK())
        if args.gf:
            trackers.append(build_tracker_GF())
        if args.ball:
            trackers.append(build_tracker_BallTracker(args.clow,args.chigh))
        if args.kal:
            trackers.append(build_tracker_Kalman(args.clow,args.chigh,args.t))
    print(f"{len(trackers)} trackers defined!")
    # create camera
    # default camera, break on Error and buffer length of 100
    cam = CameraStream.CameraStreamBuffer(0,True,100)
    # define fourcc
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    ## create file objects
    # check target paths
    if not args.notRec:
        if checkPath(args.rec):
            raise ValueError(f"Target directory for output recording {args.rec} does not exist!")

        if checkPath(args.cam):
            raise ValueError(f"Target directory for input camera footage {args.cam} does not exist!")

        if checkPath(args.data):
            raise ValueError(f"Target directory for estimated position and velocity {args.data} does not exist!")
    # intialize file objects
    recFile = None
    camFile = None
    dataFile = None
    # attempt to open files
    if not args.cam:
        camFile = cv2.VideoWriter(args.cam,fourcc,60,cam.frame.shape[:2])
        if not camFile.isOpened():
            raise ValueError("Failed to open input recording file!")
    if not args.rec:
        recFile = cv2.VideoWriter(args.rec,fourcc,60,(cam.frame.shape[0],cam.frame.shape[1]*len(trackers)))
        if not recFile.isOpened():
            raise ValueError("Failed to open output camera recording file")
    # start camera
    cam.start()
    # get list of frames
    res = deque(maxlen=len(trackers))
    # get start time
    t0 = time.time()
    # open data file
    with open(args.data,'w'):
        while True:
            # get frame
            frame = cam.read()
            cv2.imshow("Cam",frame)
            # if a frame was obtained
            if frame is not None:
                # write camera frame to file
                if (recFile is not None) and recFile.isOpened():
                    recFile.write(frame)
                # write timestamp
                tdiff = time.time()-t0
                dataFile.write(str(tdiff))
                # pass frame to each tracker
                for tt in trackers:
                    tt.track(frame)
                    # get results
                    res.append(tt.res)
                    # write angle to file
                    dataFile.write(',')
                    dataFile.write(tt.angle)
                dataFile.write('\n')
                # write frames to files
                # stack the tracker results horizontally to form frame
                rec = np.hstack(res)
                if (recFile is not None) and recFile.isOpened():
                    recFile.write(rec)
                # display results
                cv2.imshow("Rec",rec)
            # key press for emergency exit
            key = cv2.waitKey(1) & 0xff
            if key==27:
                break
        # stop camera thread
        cam.stop()
        # iterate over queue of buffered frames
        while cam.q:
            frame = cam.read()
            # pass frame to each tracker
            for tt in trackers:
                tt.track(frame)
                # write angle to file
                dataFile.write(',')
                dataFile.write(tt.angle)
            dataFile.write('\n')
    # close recording files
    recFile.release()
    camFile.release()