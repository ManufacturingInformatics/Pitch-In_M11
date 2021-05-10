import cv2
import numpy as np
import time
from RotatingDotGen import RotatingDot, RotatingAnimator
from PyImageSearchBallTracker import BallTracker

import matplotlib.pyplot as plt

def KFXYPos():
    rd = RotatingDot(15,100)
    d = rd.getDist()
    anim = RotatingAnimator(rd,60.0**-1.0,np.deg2rad(5.0))
    colLower = (0,0,0)
    colUpper = (255,255,50)
    # create tracker class stating that the inputs are in RGB colorspace
    tracker = BallTracker(colLower,colUpper,"RGB")
    colLower = (0,0,0)
    colUpper = (255,255,50)
    # create tracker class stating that the inputs are in RGB colorspace
    tracker = BallTracker(colLower,colUpper,"RGB",histLen=10)
    # set fps to simulate a camera
    fps = 30.0
    # time between frames
    T = fps**-1.0
    # adjust to manipulate sampling rate
    T *= 2.0
    # timestamps to maintain fps
    t = time.time()
    ots = t
    # old angle
    oa = 0.0
    # old velocity
    ovel = 0.0
    # kalman filter
    # states : x, y, x dot, y dot
    # measurement : x, y
    # no control feedback so no ctrl variables
    kalman = cv2.KalmanFilter(4,2,0)
    # initial state
    state = np.random.randn(4,1)
    #state = np.random.random_sample((3,1))*2.0*np.pi
    #kalman.statePre = np.random.random_sample((3,1))*2.0*np.pi
    ## kalman matricies
    ## initialize Kalman parameters
    # transition matrix taking to account acceleration
    kalman.transitionMatrix = np.array([[1.0,0.0,T,0.0],
                                        [0.0,1.0,0.0,T],
                                        [0.0,0.0,1.0,0.0],
                                        [0.0,0.0,0.0,1.0]])
    # measurement gain matrix
    kalman.measurementMatrix = np.array([[1.0,1.0,0.0,0.0]])
    #print(kalman.measurementMatrix.shape)
    kalman.processNoiseCov = 1e-4 * np.eye(4) # noise covariance of process
    kalman.measurementNoiseCov = 1e-5 * np.ones((1,1)) # measurement noise, added by tracker?
    #kalman.measurementNoiseCov = 1e-5 * np.eye(2)
    # ?
    kalman.errorCovPost = 1. * np.ones((4,4)) 
    kalman.statePost = 0.1 * np.random.randn(4,1)

    # define window to update
    cv2.namedWindow("Kalman")
    # define point of rotation as centre of image
    frame = rd.getImg()
    ptr = np.array(frame.shape[:2])//2
    # start animation
    anim.start()
    # main loop
    while True:
        t = time.time()
        dur = t-ots
        if dur>=T:
            ots = t 
            # get frame
            frame = anim.get()
            # add noise
            frame = cv2.blur(frame,(5,5))
            # put into tracker
            res,found = tracker.track(frame)
            # if found
            if found:
                # run a prediction
                prediction = kalman.predict()
                print(prediction.shape)

                # draw marker indicating center
                cv2.drawMarker(res,(int(prediction[0,0]),int(prediction[1,0])),(255,0,0),cv2.MARKER_DIAMOND,10,2)
                cv2.imshow("Kalman",res)

                # update state
                # ?
                process_noise = np.sqrt(kalman.processNoiseCov[0,0]) * np.random.randn(4, 1)
                #process_noise = 0.0
                # state times transition + noise of the process
                state = np.dot(kalman.transitionMatrix,state)+process_noise

                measurement = np.dot(kalman.measurementMatrix,np.array([tracker.center[0],tracker.center[1]]))
                #print(measurement.shape)

                kalman.correct(measurement)

            # press escape to exit
            key = cv2.waitKey(1) &0xff
            if key == 27:
                break

    anim.stop()
                
            

# kalman filter taking into account acceleration
def KFAccDot():
    rd = RotatingDot(15,100)
    d = rd.getDist()
    anim = RotatingAnimator(rd,60.0**-1.0,np.deg2rad(5.0))
    colLower = (0,0,0)
    colUpper = (255,255,50)
    # create tracker class stating that the inputs are in RGB colorspace
    tracker = BallTracker(colLower,colUpper,"RGB")
    colLower = (0,0,0)
    colUpper = (255,255,50)
    # create tracker class stating that the inputs are in RGB colorspace
    tracker = BallTracker(colLower,colUpper,"RGB",histLen=10)
    # set fps to simulate a camera
    fps = 30.0
    # time between frames
    T = fps**-1.0
    # adjust to manipulate sampling rate
    T *= 2.0
    # timestamps to maintain fps
    t = time.time()
    ots = t
    # old angle
    oa = 0.0
    # old velocity
    ovel = 0.0
    # kalman filter
    # states : theta, theta dot and theta dot^2
    # measurement : theta, (theta dot, theta dot^2)
    # no control feedback so no ctrl variables
    kalman = cv2.KalmanFilter(3,1,0)
    # initial state
    state = np.random.randn(3,1)
    #state = np.random.random_sample((3,1))*2.0*np.pi
    #kalman.statePre = np.random.random_sample((3,1))*2.0*np.pi
    ## kalman matricies
    ## initialize Kalman parameters
    # transition matrix taking to account acceleration
    kalman.transitionMatrix = np.array([[1., T,  0.5*T**2.0],
                                        [0.0,1.0,T],
                                        [0.0,0.0,1.0]])
    # measurement gain matrix
    kalman.measurementMatrix = np.array([[1.0,0.0,0.0]]) 
    kalman.processNoiseCov = 1e-4 * np.eye(3) # noise covariance of process
    kalman.measurementNoiseCov = 1e-5 * np.ones((1,1)) # measurement noise, added by tracker?
    # ?
    kalman.errorCovPost = 1. * np.ones((3,3)) 
    kalman.statePost = 0.1 * np.random.randn(3,1)
    
    # define window to update
    cv2.namedWindow("Kalman")
    # define point of rotation as centre of image
    frame = rd.getImg()
    ptr = np.array(frame.shape[:2])//2
    ## vectors to hold results
    # kalman state vector ?
    state_ang = np.zeros((1,))
    state_vel = np.zeros((1,))
    state_acc = np.zeros((1,))
    # data from tracker. measurements
    meas_ang = np.zeros((1,))
    meas_vel = np.zeros((1,))
    meas_acc = np.zeros((1,))
    # predictions from kalman filter
    pred_ang = np.zeros((1,))
    pred_vel = np.zeros((1,))
    pred_acc = np.zeros((1,))
    # raw velocity estimate from the change in estimated angle
    raw_ang = np.zeros((1,))
    raw_vel = np.zeros((1,))
    raw_acc = np.zeros((1,))
    # start animation
    anim.start()
    # main loop
    while True:
        t = time.time()
        dur = t-ots
        if dur>=T:
            ots = t 
            # get frame
            frame = anim.get()
            # add noise
            frame = cv2.blur(frame,(5,5))
            # put into tracker
            res,found = tracker.track(frame)
            # get the estimated angle and angular veloicity from tracker
            ang = tracker.angle
            # if the angle has not been updated
            # or the angle has yet to be found
            if ang is not None:
                raw_ang = np.append(raw_ang,ang)
                #print(ang)
                vel = tracker.getV()
                
                # predict using kalman filter
                prediction = kalman.predict()
                pang = prediction[0,0]
                pvel = prediction[1,0]
                pacc = prediction[2,0]
                
                ## estimate velocity from differences in angle and sampling period
                # find difference between angles
                da = abs(ang-oa)
                # if the difference betweeen angles is more than 300 degrees
                # then the new angle is the start of a new rotation
                # subtract 2 pi to find actual difference between the two
                if da>6.0:
                    da -= 2.0*np.pi
                    da = abs(da)
                # calculate velocity and add to vector
                raw_vel = np.append(raw_vel,da/dur)
                if raw_vel.shape[0]>1:
                    raw_acc = np.append(raw_acc,(raw_vel[-1]-raw_vel[-2])/dur)
                oa = ang

                # update state
                # ?
                process_noise = np.sqrt(kalman.processNoiseCov[0,0]) * np.random.randn(3, 1)
                #process_noise = 0.0
                # state times transition + noise of the process
                state = np.dot(kalman.transitionMatrix,state)+process_noise

                # add to logging vectors
                meas_ang = np.append(meas_ang,ang)
                meas_vel = np.append(meas_vel,vel)
                # if there's at least one other velocity point
                # then acceleration can be calculated
                if meas_vel.shape[0]>1:
                    meas_acc = np.append(meas_acc,(vel-meas_vel[-2])/dur)
                #print(meas_acc.shape)

                # generate measurement vector
                # no noise on measurement
                # raw velocity and acceleration are fed in rather than the tracker ones as they tend to be very noisy
                measurement = np.dot(kalman.measurementMatrix,np.array([raw_ang[-1],raw_vel[-1],raw_acc[-1]]))
                #measurement = np.dot(kalman.measurementMatrix,np.array([meas_ang[-1],meas_vel[-1],meas_acc[-1]]))
                #measurement = np.dot(kalman.measurementMatrix,state)+np.array([raw_ang[-1],raw_vel[-1],raw_acc[-1]])
                print(measurement.shape)
                # correct filter with new measurement
                kalman.correct(measurement)
                
                pred_ang = np.append(pred_ang,pang)
                pred_vel = np.append(pred_vel,pvel)
                pred_acc = np.append(pred_acc,pacc)
                
                state_ang = np.append(state_ang,state[0,0])
                state_vel = np.append(state_vel,state[1,0])
                state_acc = np.append(state_acc,state[2,0])

                ## draw predicted Kalman location
                ## current prediction only
                # get latest tracker center
                cc = tracker.getLastCenter()
                #print(dist)
                # calculate predicted position using angle
                a,b = cv2.polarToCart(d,pang)
                #print(a[0],b[0])
                # draw marker
                #cv2.drawMarker(res,(int(ptr[0]-a[0]),int(ptr[1]-b[0])),(0,255,0),cv2.MARKER_DIAMOND,5,2)
                cv2.circle(res,(int(ptr[0]-a[0]),int(ptr[1]-b[0])),5,(0,255,0),2)
            cv2.imshow("Kalman",res)
        # press escape to exit
        key = cv2.waitKey(1) &0xff
        if key == 27:
            break
    # stop animation
    anim.stop()
    # print vector sizes
    print("\n\nEnd data size")
    print(state_ang.shape)
    print(meas_ang.shape)
    print(pred_ang.shape)

    ## plot results
    f,ax = plt.subplots(ncols=3,nrows=2)
    # plot position data
    ax[0,0].plot(meas_ang,'r-',label="Est. angle")
    ax[0,0].set(xlabel="Data Index",ylabel="Angle (rads)",title="Estimated Angle from Tracker")
    ax[1,0].plot(pred_ang,'g-',label="Pred. angle")
    ax[1,0].set(xlabel="Data Index",ylabel="Angle (rads)",title="Kalman Predicted Angle")
    # plot velocity data
    ax[0,1].plot(meas_vel,'r-',label="Est. Vel")
    ax[0,1].set(xlabel="Data Index",ylabel="Angular Velocity (rads/s)",title="Estimated Angular Velocity from Tracker")
    ax[1,1].plot(pred_vel,'g-',label="Pred. Vel")
    ax[1,1].set(xlabel="Data Index",ylabel="Angular Velocity (rad/s)",title="Kalman Predicted Angular Velocity")
    # plot acceleration data
    ax[0,2].plot(meas_acc,'r-',label="Est. Acc")
    ax[0,2].set(xlabel="Data Index",ylabel="Angular Acceleration ($rad/s^2$)",title="Estimated Angular Acceleration from Tracker")
    ax[1,2].plot(pred_acc,'g-',label="Pred. Vel")
    ax[1,2].set(xlabel="Data Index",ylabel="Angular Acceleration ($rad/s^2$)",title="Kalman Predicted Angular Acceleration")
    
    # plot estimated velocity data
    f2,ax2 = plt.subplots()
    ax2.plot(raw_vel)
    ax2.set(xlabel="Data Index",ylabel="Angular Velocity (rad/s)",title="Predicted Angular Velocity from Differences in Angles")

    facc,axacc = plt.subplots()
    axacc.plot(raw_acc)
    axacc.set(xlabel="Data Index",ylabel="Angular Acceleration ($rad/s^2$)",title="Prediction Angular Acceleration from Differences in Velocity")

    plt.show()

# kalman filter using rotating dot
# estimates angular position
# plots at the end
def KFDot(recRes=False):
    # define rotating dot
    rd = RotatingDot(15,100)
    d = rd.getDist()
    # define animator
    anim = RotatingAnimator(rd,dur=60.0**-1.0,dangle=np.deg2rad(5.0))
    # define tracker
    # version of ball tracker without color target
    colLower = (0,0,0)
    colUpper = (255,255,50)
    # create tracker class stating that the inputs are in RGB colorspace
    # setting history length to a smaller amount as it is not being updated as often
    tracker = BallTracker(colLower,colUpper,"RGB",histLen=10)

    # set fps to simulate a camera
    fps = 30.0
    # time between frames
    T = fps**-1.0
    # adjust to manipulate sampling rate
    T *= 2.0
    # timestamps to maintain fps
    t = time.time()
    ots = t
    # old angle
    oa = 0.0

    if recRes:
        rec = cv2.VideoWriter("rotating_dot_kalman.avi",cv2.VideoWriter_fourcc(*'MJPG'),int(T**-1.0),rd.getFrameSize()[:2][::-1],True)
        if not rec.isOpened():
            print("failed to open recorder")
            rec.release()
    
    # define filter
    # dimensionality of state, dimensionality of measurement, control parameters
    # states: angle and angle dot
    # measurements : angle (and angle dot)
    kalman = cv2.KalmanFilter(2,1,0)
    ## initialize Kalman parameters
    # transition matrix using simulated FPS
    kalman.transitionMatrix = np.array([[1., T], [0., 1.]])
    kalman.measurementMatrix = np.array([[1.0,0.0]])
    #kalman.processNoiseCov = 1e-5 * np.eye(2) # noise covariance
    kalman.processNoiseCov = np.array([[1e-5,0.0],[0.0,1e-4]])
    kalman.measurementNoiseCov = 0.0 * np.ones((1, 1)) # measurement noise
    # ?
    kalman.errorCovPost = 1. * np.ones((2,2)) 
    kalman.statePost = 0.1 * np.random.randn(2,1)
    # define window to update
    cv2.namedWindow("Kalman")
    # define point of rotation as centre of image
    frame = rd.getImg()
    ptr = np.array(frame.shape[:2])//2
    ## vectors to hold results
    # data from tracker. measurements
    meas_ang = np.zeros((0,))
    meas_vel = np.zeros((0,))
    # predictions from kalman filter
    pred_ang = np.zeros((0,))
    pred_vel = np.zeros((0,))
    # raw velocity estimate from the change in estimated angle
    raw_vel = np.zeros((0,))
    # start animation
    anim.start()
    # main loop
    while True:
        t = time.time()
        dur = t-ots
        if dur>=T:
            ots = t 
            # get frame
            frame = anim.get()
            # add noise
            frame = cv2.blur(frame,(5,5))
            # put into tracker
            res,found = tracker.track(frame)
            # get the estimated angle and angular veloicity from tracker
            ang = tracker.angle
            # if the angle has not been updated
            # or the angle has yet to be found
            if ang is not None:
                #print(ang)
                # velocity from tracker
                vel = tracker.getV()
                # predict angle and angular velocity using kalman
                pang,pvel = kalman.predict()
                
                ## estimate velocity from differences in angle and sampling period
                # find difference between angles
                da = abs(ang-oa)
                # if the difference betweeen angles is more than 300 degrees
                # then the new angle is the start of a new rotation
                # subtract 2 pi to find actual difference between the two
                if da>=6.0:
                    da -= 2.0*np.pi
                    da = abs(da)
                # calculate velocity and add to vector
                raw_vel = np.append(raw_vel,da/dur)
                oa = ang

                measurement = np.dot(kalman.measurementMatrix,np.array([ang,vel]))
                #measurement = np.dot(kalman.measurementMatrix,np.array([ang,raw_vel[-1]]))
                #measurement = np.dot(kalman.measurementMatrix,np.array([ang,(ang-oa)/dur]))
                # correct filter with new measurement
                kalman.correct(measurement)

                # add to logging vectors
                meas_ang = np.append(meas_ang,ang)
                meas_vel = np.append(meas_vel,vel)
                pred_ang = np.append(pred_ang,pang[0])
                pred_vel = np.append(pred_vel,pvel[0])

                ## draw predicted Kalman location
                ## current prediction only
                # get latest tracker center
                cc = tracker.getLastCenter()
                #print(dist)
                # calculate predicted position using angle
                a,b = cv2.polarToCart(d,pang[0])
                #print(a[0],b[0])
                # draw marker
                #cv2.drawMarker(res,(int(ptr[0]-a[0]),int(ptr[1]-b[0])),(0,255,0),cv2.MARKER_DIAMOND,5,2)
                cv2.circle(res,(int(ptr[0]-a[0]),int(ptr[1]-b[0])),5,(0,255,0),2)
            if recRes:
                if rec.isOpened():
                    rec.write(res)
            # update results window
            cv2.imshow("Kalman",res)
        # press escape to exit
        key = cv2.waitKey(1) &0xff
        if key == 27:
            break
    # stop animation
    anim.stop()

    if recRes:
        rec.release()
    # print vector sizes
    print("\n\nEnd data size")
    print(meas_ang.shape)
    print(pred_ang.shape)

    ## plot results
    f,ax = plt.subplots(ncols=2,nrows=2,constrained_layout=True)
    # plot position data
    ax[0,0].plot(meas_ang,'r-',label="Est. angle")
    ax[0,0].set(xlabel="Data Index",ylabel="Angle (rads)",title="Estimated Angle from Tracker")
    ax[1,0].plot(pred_ang,'g-',label="Pred. angle")
    ax[1,0].set(xlabel="Data Index",ylabel="Angle (rads)",title="Kalman Predicted Angle")
    # plot velocity data
    ax[0,1].plot(meas_vel,'r-',label="Est. Vel")
    ax[0,1].set(xlabel="Data Index",ylabel="Vel (rads/s)",title="Estimated Angular Velocity from Tracker")
    ax[1,1].plot(pred_vel,'g-',label="Pred. Vel")
    ax[1,1].set(xlabel="Data Index",ylabel="Vel (rad/s)",title="Kalman Predicted Angular Velocity")

    # plot estimated velocity data
    f2,ax2 = plt.subplots()
    ax2.plot(raw_vel)
    ax2.set(xlabel="Data Imdex",ylabel="Vel (rad/s)",title="Predicted Angular Velocity from Differences in Angles")
    
    plt.show()

    return pred_ang,meas_ang

def KFCVExample(recRes=False):
    # define rotating dot
    rd = RotatingDot(15,100)
    d = rd.getDist()
    # define animator
    anim = RotatingAnimator(rd,dur=60.0**-1.0,dangle=np.deg2rad(5.0))
    # define tracker
    # version of ball tracker without color target
    colLower = (0,0,0)
    colUpper = (255,255,50)
    # create tracker class stating that the inputs are in RGB colorspace
    # setting history length to a smaller amount as it is not being updated as often
    tracker = BallTracker(colLower,colUpper,"RGB",histLen=10)

    # set fps to simulate a camera
    fps = 30.0
    # time between frames
    T = fps**-1.0
    # adjust to manipulate sampling rate
    T *= 2.0
    # timestamps to maintain fps
    t = time.time()
    ots = t
    # old angle
    oa = 0.0

    if recRes:
        rec = cv2.VideoWriter("rotating_dot_kalman_opencveg.avi",cv2.VideoWriter_fourcc(*'MJPG'),int(T**-1.0),rd.getFrameSize()[:2][::-1],True)
        if not rec.isOpened():
            print("failed to open recorder")
            rec.release()
    
    # define filter
    # dimensionality of state, dimensionality of measurement, control parameters
    # states: angle and angle dot
    # measurements : angle (and angle dot)
    kalman = cv2.KalmanFilter(2,1,0)
    # state vector
    state = 0.1 * np.random.randn(2,1)
    ## initialize Kalman parameters
    # transition matrix using simulated FPS
    kalman.transitionMatrix = np.array([[1., T], [0., 1.]])
    kalman.measurementMatrix = 1. * np.ones((1,2))
    kalman.processNoiseCov = 1.0 * np.eye(2)
    kalman.measurementNoiseCov = 1e-6 * np.ones((1, 1)) # measurement noise
    # ?
    kalman.errorCovPost = 1. * np.ones((2,2)) 
    kalman.statePost = 0.1 * np.random.randn(2,1)
    # define window to update
    cv2.namedWindow("Kalman")
    # define point of rotation as centre of image
    frame = rd.getImg()
    ptr = np.array(frame.shape[:2])//2
    ## vectors to hold results
    # data from tracker. measurements
    meas_ang = np.zeros((0,))
    meas_vel = np.zeros((0,))
    # predictions from kalman filter
    pred_ang = np.zeros((0,))
    pred_vel = np.zeros((0,))
    # raw velocity estimate from the change in estimated angle
    raw_vel = np.zeros((0,))
    # start animation
    anim.start()
    # main loop
    while True:
        t = time.time()
        dur = t-ots
        if dur>=T:
            ots = t 
            # get frame
            frame = anim.get()
            # add noise
            frame = cv2.blur(frame,(5,5))
            # put into tracker
            res,found = tracker.track(frame)
            # get the estimated angle and angular veloicity from tracker
            ang = tracker.angle
            # if the angle has not been updated
            # or the angle has yet to be found
            if ang is not None:
                #print(ang)
                # velocity from tracker
                vel = tracker.getV()
                # predict angle and angular velocity using kalman
                pang,pvel = kalman.predict()

                # process measurement
                measurement = kalman.measurementNoiseCov*np.random.randn(1,1)
                measurement = np.dot(kalman.measurementMatrix,state)+measurement
                print(measurement.shape)

                ## draw predicted Kalman location
                ## current prediction only
                # get latest tracker center
                cc = tracker.getLastCenter()
                #print(dist)
                # calculate predicted position using angle
                a,b = cv2.polarToCart(d,pang[0])
                #print(a[0],b[0])
                # draw marker
                #cv2.drawMarker(res,(int(ptr[0]-a[0]),int(ptr[1]-b[0])),(0,255,0),cv2.MARKER_DIAMOND,5,2)
                cv2.circle(res,(int(ptr[0]-a[0]),int(ptr[1]-b[0])),5,(0,255,0),2)

                cv2.imshow("Kalman",res)

                # update kalman filter
                kalman.correct(measurement)

                # update state
                process_noise = np.sqrt(kalman.processNoiseCov[0,0])*np.random.randn(2,1)
                state = np.dot(kalman.transitionMatrix,state) + process_noise

                key = cv2.waitKey(1)
                if key == 27:
                    break

    anim.stop()

    if recRes:
        rec.release()

if __name__ == "__main__":
    #pa=KFDot(False)
    #KFAccDot()
    #KFXYPos()
    KFCVExample()
    cv2.destroyAllWindows()

