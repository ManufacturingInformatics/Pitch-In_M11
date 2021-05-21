import cv2
import numpy as np

# wrapper class for Kalman filter being updated using an image tracker
# tracker measures position and is used to update filter
# extends cv2.KalmanFilter
class KalmanDotTracker(cv2.KalmanFilter):
    # initialize tracker by passing the tracker class and update period
    def __init__(self,tracker,T=60**-1.0):
        # build kalman filter
        super().__init__(2,1,0)
        # setup initial state as random position, random velocity
        self.state = 0.1* np.random.randn(2,1).astype("float32")
        # store tracker
        self.__tracker = tracker
        # current angle from tracker
        self.__angle = 0.0
        # update rate of the model
        self.__T = T
        # create transition matrix
        self.transitionMatrix = np.array([[1., T], [0., 1.]],dtype="float32")
        # create measurement matrix
        self.measurementMatrix = np.array([1.0,0.0],dtype='float32')
        # covariances
        self.processNoiseCov = 1.0 * np.eye(2,dtype="float32")
        self.measurementNoiseCov = 1e-6 * np.ones((1, 1),dtype="float32") # measurement noise
        # post
        self.errorCovPost = 1. * np.ones((2,2),dtype="float32")
        self.statePost = 0.1 * np.random.randn(2,1).astype("float32")
    # return the attributes of the tracker
    def getTrackerSettings(self):
        return self.__tracker.__dict__
    # process passed image using set tracker
    # if the dot position is found, update kalman filter
    # class state is always updated
    def track(self,frame):
        # pass image to tracker
        res,found = self.__tracker.track(frame)
        # if the point was not found
        if not found:
            # predict next point using filter
            pang,_ = self.predict()
            # update
            self.__angle = pang
        # if the point was found
        else:
            # update class angle
            self.__angle = self.__tracker.angle
            # measurement from tracker
            measurement = np.array([[self.__tracker.angle]],dtype='float')
            #measurement[np.isnan(measurement)]=0.0
            #measurement = np.dot(self.measurementMatrix,self.state)+measurement
            #print(f"{np.dot(self.measurementMatrix,self.state).shape}")
            # correct filter
            measurement[np.isnan(measurement)]=0.0
            measurement = measurement.astype("float32")
            print(measurement,measurement.dtype)
            self.correct(measurement)
        # update state
        process_noise = sqrt(self.processNoiseCov[0,0]) * np.random.randn(2, 1)
        self.state = np.dot(self.transitionMatrix, state) + process_noise
        # return current estimate of angle for ease of use
        return self.__angle

    # method to get angle
    def angle(self):
        return self.__angle

# wrapper class for Kalman filter being updated using an image tracker
# tracker measures position and is used to update filter
# this version of filter also tracks acceleration
# extends cv2.KalmanFilter
class KalmanDotAccTracker(cv2.KalmanFilter):
    # initialize tracker by passing the tracker class and update period
    def __init__(self,tracker,T=60**-1.0):
        # build kalman filter
        super().__init__(3,1,0)
        # setup initial state as random position, random velocity
        self.state = 0.1* np.random.randn(3,1)
        # store tracker
        self.__tracker = tracker
        # current angle from tracker
        self.__angle = 0.0
        # update rate of the model
        self.__T = T
        # create transition matrix
        # position, velocity, acceleration
        self.transitionMatrix = np.array([[1.0,T,0.5*T**2.0],[0.0, 1., T], [0.0, 0., 1.]])
        # create measurement matrix
        # measuring position only
        self.measurementMatrix = [1.0,0.0,0.0]
        # covariances
        self.processNoiseCov = 1.0 * np.eye(3)
        self.measurementNoiseCov = 1e-6 * np.ones((1, 1)) # measurement noise
        # post
        self.errorCovPost = 1. * np.ones((3,3))
        self.statePost = 0.1 * np.random.randn(3,1)
    # process passed image using set tracker
    # if the dot position is found, update kalman filter
    # class state is always updated
    def track(self,frame):
        # pass image to tracker
        res,found = self.__tracker.track(frame)
        # if the point was not found
        if not found:
            # predict next point using filter
            pang,_ = self.predict()
            # update
            self.__angle = pang
        # if the point was found
        else:
            # measurement from tracker
            measurement = np.array([[self.__tracker.angle]],dtype='float')
            # correct filter
            kalman.correct(measurement)
            # update class angle
            self.__angle = self.__tracker.angle
        # update state
        process_noise = sqrt(self.processNoiseCov[0,0]) * np.random.randn(2, 1)
        self.state = np.dot(self.transitionMatrix, state) + process_noise
        # return current estimate of angle for ease of use
        return self.__angle

    def angle(self):
        return self.__angle

# save matricies of kalman filter as an NPY file
def saveKalmanNPY(kalman,path):
    # create custom datatype to store matricies of kalman filter
    ktype = np.dtype([('transitionMatrix',kalman.transitionMatrix.dtype,kalman.transitionMatrix.shape),
                    ('measurementMatrix',kalman.measurementMatrix.dtype,kalman.measurementMatrix.shape),
                    ('processNoiseCov',kalman.processNoiseCov.dtype,kalman.processNoiseCov.shape),
                    ('measurementNoiseCov',kalman.measurementNoiseCov.dtype,kalman.measurementNoiseCov.shape),
                    ('errorCovPost',kalman.errorCovPost.dtype,kalman.errorCovPost.shape),
                    ('statePost',kalman.statePost.dtype,kalman.statePost.shape),
                    ('gain',kalman.gain.dtype,kalman.gain.shape)])
    # create and populate matrix
    km = np.empty(1,dtype=dd)
    for ff in ktype.fields:
        km[ff] = getattr(kalman,ff)
    # save structured matrix
    np.save(path,km)

# load cv2.KalmanFilter saved as NPY file
def loadKalmanNPY(path):
    # load file
    km = np.load(path)
    # create kalman object
    kalman = cv2.KalmanFilter()
    for ff in km.dtype.fields:
        setattr(kalman,ff,km[ff][0])
    return kalman
