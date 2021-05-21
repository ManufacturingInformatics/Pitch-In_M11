from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time

import matplotlib.pyplot as plt

from RotatingDotGen import RotatingDot,RotatingAnimator,Occlusion,OcclusionFactory
# adapted from,
#https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/
# collection of tracker creator classes
OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
}

def objectTrackAllDotVel(recRes=False):
    rec = None
    #create dot generator and manager
    rd = RotatingDot(15,100)
    # create animator
    anim = RotatingAnimator(rd,dur=60.0**-1.0,dangle=np.deg2rad(5.0))
    # get frame
    frame = rd.getImg()
    # get user to select bounding box to search
    initBB = cv2.selectROI("Choose new box",frame,fromCenter=False,showCrosshair=True)
    cv2.destroyWindow("Choose new box")
    # check stated bounding box
    if not all(initBB):
        print("no initial bounding box")
        return
    
    # build list of trackers
    print("building trackers")
    trackers = []
    for v in OPENCV_OBJECT_TRACKERS.values():
        try:
            trackers.append(v())
            trackers[-1].init(frame,initBB)
        except:
            print(f"failed to initialize {str(v)}")

    # define array to hold velocity and angle estimates
    angleMat = np.empty((1,len(trackers)),np.float32)
    velMat = np.empty((1,len(trackers)),np.float32)
    tvec = np.empty((1,),np.float32)
    angles = []
    vels = []
    # timestamps for estimating velocity
    ts = time.time()
    ots = ts
    startt = ts
    # resize factor of image
    rf = 0.5
    # resize test frame
    #frame = imutils.resize(frame,width=int(frame.shape[1]*rf))
    # get new size
    (H,W) = frame.shape[:2]
    # define blank frame to be used as dummy frames
    blank_frame = np.zeros(frame.shape,frame.dtype)
    H,W,D = blank_frame.shape
    # define centre of the image
    center = [W//2,H//2]
    # create collection to hold results
    res = []
    # time stamps
    # start animaator
    anim.start()
    print("starting main loop")
    # loop
    while True:
        ts = time.time()
        tvec = np.append(tvec,[ts-startt],axis=0)
        # get new frame
        frame = anim.get()
        # add noise
        frame = cv2.blur(frame,(5,5))
        # resize frame maintaining aspect ratio
        #frame = imutils.resize(frame,width=int(frame.shape[1]*rf))
        #print(frame.shape)
        # clear collection of frames
        res.clear()
        # iterate over trackers
        #print("iterating over trackers")
        angles.clear()
        vels.clear()
        
        for ti,t in enumerate(trackers):
            # make a copy of frame to draw on
            ft = frame.copy()
            # attempt to update tracker with resized new frame
            try:
                (success, box) = t.update(ft)
            except cv2.error as err:
                print(f"cv2 error for {str(t).split[0][1:]}")
                # on error set success flag to false
                success = False
            #print("updated tracker")
            # check to see if the tracking was a success
            # if it was draw a rectangle around target
            if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(ft, (x, y), (x + w, y + h),
                            (0, 255, 0), 2)
                    ## find angle in relation to the centre of the image
                    # define centre of the found box
                    cbox = (x+(w//2),y+(h//2))
                    # draw cross
                    cv2.drawMarker(ft,cbox,(255,0,0),cv2.MARKER_DIAMOND,10,2)
                    # find distance from centre of the image
                    dist = [cc-pt for cc,pt in zip(center,cbox)]
                    # find angle
                    a = np.arctan2(dist[1],dist[0])
                    angles.append(a)
                    # find velocity if there's at least one other angle to compare against
                    if angleMat.shape[0]>1:
                        if (bool(a<0) ^ bool(angleMat[-1,ti]<0)):
                            vels.append(abs(abs(a)-abs(angleMat[-1,ti]))/(ts-ots))
                        else:   
                            vels.append((a-angleMat[-1,ti])/(ts-ots))
                    else:
                        vels.append(np.NAN)
            else:
                angles.append(np.NAN)
                vels.append(np.NAN)

            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                    ("Success", "Yes" if success else "No"), # if tracking with the current frame is successful
                    ("Type ",str(t).split()[0][1:]), # type of tracker
            ]
            #print("drawing info")
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(ft, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, ((0, 0, 255) if not success else (0,255,0)), 2)
            #print("adding to list")
            res.append(ft)
        # append estimates to matrix
        #print(len(angles))
        #print(angleMat.shape)
        angleMat = np.append(angleMat,[angles],axis=0)
        velMat = np.append(velMat,[vels],axis=0)

##        print(len(res))
##        for r in res:
##            print(r.shape)
            
        if len(res) ==0:
            print("All trackers failed")
            continue
        
        #print("generating result")
        # check number of frames in result
        if (len(res)%2) != 0:
            res.append(blank_frame)

        # concatenate to form super result
        #draw = np.concatenate(res,axis=1)
        r0 = np.concatenate(res[:len(res)//2],axis=1)
        r1 = np.concatenate(res[len(res)//2:],axis=1)
        draw = np.concatenate((r0,r1),axis=0)
        # if the flag is set to record the footage
        if recRes:
            if rec is None:
                rec = cv2.VideoWriter("rotating_dot_trackers_vel.avi",cv2.VideoWriter_fourcc(*'MJPG'),15,draw.shape[:2][::-1])
                if not rec.isOpened():
                    print("failed to open recorder")
                    rec.release()
            else:
                if rec.isOpened():
                    rec.write(draw)
        # show
        cv2.imshow("Frame",draw)
        # wait for key
        # allows opencv to draw
        key = cv2.waitKey(1) & 0xFF
        # if q or ESC was pressed, exit from loop
        if (key == ord('q')) or (key == 27):
            print("exit")
            break
        # if s is pressed it allows the user to select a new bounding box to feed into the trackers
        elif key == ord('s'):
            # update bounding box with selected target box
            initBB = cv2.selectROI("Choose new box",frame,fromCenter=False,showCrosshair=True)
        
            if all(initBB):
                print("updating bounding box")
                for t in trackers:
                    # update tracker
                    t.init(frame,initBB)
            # close window
            cv2.destroyWindow("Choose new box")

        ots = ts
        
    ## cleanup
    # close camera
    anim.stop()
    if recRes:
        rec.release()

    # plot data
    fv,axv = plt.subplots()
    axv.plot(tvec,velMat)
    fv.legend([str(t).split()[0][1:] for t in trackers])
    axv.set(xlabel="Time (s)",ylabel="Estimated Velocity (rad/s)",title="Estimated Velocity from the Different Trackers")

    fa,axa = plt.subplots()
    axa.plot(tvec,angleMat)
    fa.legend([str(t).split()[0][1:] for t in trackers])
    axa.set(xlabel="Data Index",ylabel="Estimated Angle (rads)",title="Estimated Angular Position from the Different Trackers")

    # plot different trackers separately
    f,ax = plt.subplots()
    for ti,t in enumerate(trackers):
        ax.clear()
        ax.plot(tvec,angleMat[:,ti],'-')
        ax.set(xlabel="Time (s)",ylabel="Estimated Angle (rads)",title=f"Estimated Angular Position using {str(t).split()[0][1:]}")
        f.savefig(f"rotating-dot-tracker-ang-{str(t).split()[0][1:]}.png")

    for ti,t in enumerate(trackers):
        ax.clear()
        ax.plot(tvec,velMat[:,ti],'-')
        ax.set(xlabel="Time (s)",ylabel="Estimated Velocity (rad/s)",title=f"Estimated Angular Velocity using {str(t).split()[0][1:]}")
        f.savefig(f"rotating-dot-tracker-vel-{str(t).split()[0][1:]}.png")
    plt.show()
    cv2.destroyAllWindows()
    return velMat,angleMat
    
def objectTrackAllDot(recRes=False):
    rec = None
    #create dot generator and manager
    rd = RotatingDot(15,100)
    # create animator
    anim = RotatingAnimator(rd,dur=60.0**-1.0,dangle=np.deg2rad(5.0))
    # get frame
    frame = rd.getImg()
    # get user to select bounding box to search
    initBB = cv2.selectROI("Choose new box",frame,fromCenter=False,showCrosshair=True)
    cv2.destroyWindow("Choose new box")
    # check stated bounding box
    if not all(initBB):
        print("no initial bounding box")
        return
    
    # build list of trackers
    print("building trackers")
    trackers = []
    for v in OPENCV_OBJECT_TRACKERS.values():
        try:
            trackers.append(v())
            trackers[-1].init(frame,initBB)
        except:
            print(f"failed to initialize {str(v)}")
    # resize factor
    rf = 0.5
    # resize test frame
    #frame = imutils.resize(frame,width=int(frame.shape[1]*rf))
    # get new size
    (H,W) = frame.shape[:2]
    # define blank frame to be used as dummy frames
    blank_frame = np.zeros(frame.shape,frame.dtype)
    # create collection to hold results
    res = []
    # start animaator
    anim.start()
    print("starting main loop")
    # loop
    while True:
        # get new frame
        frame = anim.get()
        # add noise
        frame = cv2.blur(frame,(5,5))
        # resize frame maintaining aspect ratio
        #frame = imutils.resize(frame,width=int(frame.shape[1]*rf))
        #print(frame.shape)
        # clear collection of frames
        res.clear()
        # iterate over trackers
        #print("iterating over trackers")
        for t in trackers:
            # make a copy of frame to draw on
            ft = frame.copy()
            # attempt to update tracker with resized new frame
            try:
                (success, box) = t.update(ft)
            except cv2.error as err:
                print(f"cv2 error for {str(t).split[0][1:]}")
                # on error set success flag to false
                success = False
            #print("updated tracker")
            # check to see if the tracking was a success
            # if it was draw a rectangle around target
            if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(ft, (x, y), (x + w, y + h),
                            (0, 255, 0), 2)
            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                    ("Success", "Yes" if success else "No"), # if tracking with the current frame is successful
                    ("Type ",str(t).split()[0][1:]), # type of tracker
            ]
            #print("drawing info")
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(ft, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, ((0, 0, 255) if not success else (0,255,0)), 2)
            #print("adding to list")
            res.append(ft)

##        print(len(res))
##        for r in res:
##            print(r.shape)
            
        if len(res) ==0:
            print("All trackers failed")
            continue
        
        #print("generating result")
        # check number of frames in result
        if (len(res)%2) != 0:
            res.append(blank_frame)

        # concatenate to form super result
        #draw = np.concatenate(res,axis=1)
        r0 = np.concatenate(res[:len(res)//2],axis=1)
        r1 = np.concatenate(res[len(res)//2:],axis=1)
        draw = np.concatenate((r0,r1),axis=0)
        # if the flag is set to record the footage
        if recRes:
            if rec is None:
                rec = cv2.VideoWriter("rotating_dot_trackers.avi",cv2.VideoWriter_fourcc(*'MJPG'),15,draw.shape[:2][::-1])
                if not rec.isOpened():
                    print("failed to open recorder")
                    rec.release()
            else:
                if rec.isOpened():
                    rec.write(draw)
        # show
        cv2.imshow("Frame",draw)
        if recRes:
            if rec.isOpened():
                rec.write
        # wait for key
        # allows opencv to draw
        key = cv2.waitKey(1) & 0xFF
        # if q or ESC was pressed, exit from loop
        if (key == ord('q')) or (key == 27):
            print("exit")
            break
        # if s is pressed it allows the user to select a new bounding box to feed into the trackers
        elif key == ord('s'):
            # update bounding box with selected target box
            initBB = cv2.selectROI("Choose new box",frame,fromCenter=False,showCrosshair=True)
        
            if all(initBB):
                print("updating bounding box")
                for t in trackers:
                    # update tracker
                    t.init(frame,initBB)
            # close window
            cv2.destroyWindow("Choose new box")
    ## cleanup
    # close camera
    anim.stop()
    if recRes:
        rec.release()

# object tracker using ALL classes and webcam
def objectTrackAllApp():
    # start webcam
    cam = VideoStream(0).start()
    # get frame
    frame = cam.read()
    if frame is None:
        print("failed to get frame")
        cam.stop()
        return
    else:
        initBB = cv2.selectROI("Choose new box",frame,fromCenter=False,showCrosshair=True)

    if not all(initBB):
        print("no initial bounding box")
        cam.stop()
        return
    
    # build list of trackers
    trackers = []
    for v in OPENCV_OBJECT_TRACKERS.values():
        try:
            trackers.append(v())
            trackers[-1].init(frame,initBB)
        except:
            print(f"failed to initialize {str(v)}")
    # resize factor
    rf = 0.5
    # resize test frame
    frame = imutils.resize(frame,width=int(frame.shape[1]*rf))
    # get new size
    (H,W) = frame.shape[:2]
    # define blank frame to be used as dummy frames
    blank_frame = np.zeros(frame.shape,frame.dtype)
    # create collection to hold results
    res = []
    # define size of 
    # flag to indicate a tracker error
    trackerError = False
    # loop
    while True:
        # get new frame
        frame = cam.read()
        # on fail
        if frame is None:
            print("failed to get frame")
            break
        # resize frame maintaining aspect ratio
        frame = imutils.resize(frame,width=int(frame.shape[1]*rf))
        # clear collection of frames
        res.clear()
        # iterate over trackers
        for t in trackers:
            # make a copy of frame to draw on
            ft = frame.copy()
            # attempt to update tracker with resized new frame
            try:
                (success, box) = t.update(ft)
            except cv2.error as err:
                # on error set success flag to false
                success = False

            # check to see if the tracking was a success
            # if it was draw a rectangle around target
            if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(ft, (x, y), (x + w, y + h),
                            (0, 255, 0), 2)
            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                    ("Success", "Yes" if success else "No"), # if tracking with the current frame is successful
                    ("Type ",str(t).split()[0][1:]), # type of tracker
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(ft, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, ((0, 0, 255) if not success else (0,255,0)), 2)
            res.append(ft)

        # check number of frames in result
        if len(res)%2 != 0:
            res.append(blank_frame)
        # concatenate to form super result
        #draw = np.concatenate(res,axis=1)
        r0 = np.concatenate(res[:len(res)//2],axis=1)
        r1 = np.concatenate(res[len(res)//2:],axis=1)
        draw = np.concatenate((r0,r1),axis=0)
        # check number of trackers
        
        # show
        cv2.imshow("Frame",draw)
        # wait for key
        # allows opencv to draw
        key = cv2.waitKey(1) & 0xFF
        # if q or ESC was pressed, exit from loop
        if (key == ord('q')) or (key == 27):
            print("exit")
            break
        # if s is pressed it allows the user to select a new bounding box to feed into the trackers
        elif key == ord('s'):
            # update bounding box with selected target box
            initBB = cv2.selectROI("Choose new box",frame,fromCenter=False,showCrosshair=True)
        
            if all(initBB):
                print("updating bounding box")
                for t in trackers:
                    # update tracker
                    t.init(frame,initBB)
            # close window
            cv2.destroyWindow("Choose new box")
        
    ## cleanup
    # close camera
    cam.stop()

# object tracker demonstrator using the webcam
def objectTrackApp(ttype="csrt"):
    # start webcam
    cam = VideoStream(0).start()
    # get frame
    frame = cam.read()
    # if no frame was read
    if frame is None:
        print("failed to get frame")
        cam.stop()
        return
    else:
        initBB = cv2.selectROI("Frame",frame,fromCenter=False,showCrosshair=True)

    # if no bounding box was given then initBB is set to a tuple of four 0's
    # all converts them to boolean values and checks if they're all True
    # 0's evaluate to False
    if not all(initBB):
        print("no initial bounding box")
        cam.stop()
        return
    # get tracker creator
    try:
        tracker = OPENCV_OBJECT_TRACKERS[ttype]()
        tracker.init(frame,initBB)
    except KeyError:
        print("Invalid tracker type")
        cam.stop()
        return

    while True:
        # get frame
        frame = cam.read()
        if frame is None:
            print("failed to get frame")
            return

        # resize for speed
        frame = imutils.resize(frame,width=frame.shape[1]//2)
        (H,W) = frame.shape[:2]
          
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)

        # check to see if the tracking was a success
        # if it was draw a rectangle around target
        if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                        (0, 255, 0), 2)

        # initialize the set of information we'll be displaying on
        # the frame
        info = [
                ("Success", "Yes" if success else "No"),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ((0, 0, 255) if not success else (0,255,0)), 2)

        cv2.imshow(ttype,frame)
        # wait for key
        key = cv2.waitKey(1) & 0xFF
        # if q or ESC was pressed, exit from loop
        if (key == ord('q')) or (key == 27):
            print("exit")
            break
        elif key == ord('s'):
            # update bounding box with selected target box
            initBB = cv2.selectROI("Frame",frame,fromCenter=False,showCrosshair=True)
            # if a new bounding box was selected
            if all(initBB):
                # update tracker
                tracker.init(frame,initBB)
    # stop and release camera
    cam.stop()
    
if __name__ == "__main__":
    #objectTrackApp()
    #objectTrackAllApp()
    #objectTrackAllDot(False)
    v,a=objectTrackAllDotVel(True)
    cv2.destroyAllWindows()
