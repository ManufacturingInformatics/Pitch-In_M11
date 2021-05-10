#### Based off https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
from threading import Thread
from imutils.video import FPS
import cv2
from collections import deque
import argparse
# multi threaded camera class
# connects to target source and captures frame on separate thread
class CameraStream:
    def __init__(self,src=0,breakOnError=True):
        print(f"connection to source {src}")
        # init camera stream
        self.stream = cv2.VideoCapture(src)
        # setup FPS estimator
        self.fps = FPS()
        # check if opened
        # if not return None
        if not self.stream.isOpened():
            print("Failed to open stream")
            return None
        # get test frame
        self.grabbed,self.frame = self.stream.read()
        if not self.grabbed:
            print("Failed to get test frame from source")
            return None
        # stop flag
        self.stopped = False
        # error flag
        self.__error = False
        # flag to break on error
        self.onError = breakOnError
    # start capturing
    def start(self):
        # clear stop flag
        self.stopped = False
        self.error = False
        # start capture thread
        Thread(target=self.update,args=()).start()
        return self
    # update method for capturing the next frame from source
    def update(self):
        self.fps.start()
        # while the stop flag is not set read next frame
        while not self.stopped:
            self.grabbed,self.frame = self.stream.read()
            # if it failed to get a frame
            if (not self.grabbed and self.onError):
                print("Failed to capture frame! Exiting")
                # set stopped flag
                self.stopped = True
                # set error flag
                self.__error = True
                return
            self.fps.update()
        self.fps.stop()
    # return the current frame
    def read(self):
        return self.frame
    # set stop flag to kill capture process
    def stop(self):
        self.stopped=True
    # returns state of error flag
    def isError(self):
        return self.__error

# multi threaded camera class with buffer
# connects to target source, captures frame on separate thread and feeds into capture buffer
# size of buffer controlled by qlen parameter. Default 100
class CameraStreamBuffer(CameraStream):
    def __init__(self,src=0,breakOnError=True,qlen=100):
        # initialize CameraStream class
        super().__init__(src,breakOnError)
        # initialize stream
        self.q = deque(maxlen=qlen)
    # update method for thread
    def update(self):
        self.fps.start()
        while not self.stopped:
            self.grabbed,self.frame = self.stream.read()
            # if it failed to get a frame
            if (not self.grabbed and self.onError):
                print("Failed to capture frame! Exiting")
                # set stopped flag
                self.stopped = True
                self.__error = True
                return
            # append to frame
            self.q.append(self.frame)
            self.fps.update()
        self.fps.stop()
    # return latest frame
    def read(self):
        # if the frame buffer is not empty, get frame
        if self.q:
            return self.q.pop()
        # if the queue is empty return last frame
        else:
            return self.frame

# script to demonstrate using the classes
def demo(src,withBuffer=False):
    print("starting demo")
    # setup class with buffer
    if withBuffer:
        cam = CameraStreamBuffer(src)
        cam.start()
    # else setup class without buffer
    else:
        cam = CameraStream(src)
        cam.start()
    print(cam)
    # start capture loop
    while True:
        # get latest frame
        frame = cam.read()
        # if not empty, display it
        if frame is not None:
            cv2.imshow("demo",frame)
        # if the camera has been stopped
        if cam.stopped or cam.isError():
            break
        # break on ESC key
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    # close all windows
    cv2.destroyAllWindows()
    # stop camera
    cam.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-buff","--b",dest="useBuffer",action="store_true",help="Flag to use the buffered version of camera class in demonstration")
    parser.add_argument("-src",dest="src",type=int,default=0,help="Source index for target camera")
    parser.add_argument("-no-demo",dest="noDemo",action="store_true",help="Import the classes without running the demo") 
    args = parser.parse_args()
    if not args.noDemo:
        demo(args.src,args.useBuffer)
