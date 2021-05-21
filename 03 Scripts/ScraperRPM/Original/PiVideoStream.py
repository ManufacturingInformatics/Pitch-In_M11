### based off https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/ ###
# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
from collections import deque
import argparse
import cv2

# multithreaded class for capturing frames from a Raspberry Pi camera
class PiVideoStream:
	def __init__(self, resolution=(320, 240), framerate=32):
		# initialize the camera and stream
		self.camera = PiCamera()
        # store resolution
		self.camera.resolution = resolution
		self.camera.framerate = framerate
        # initialize raw cap
		self.rawCapture = PiRGBArray(self.camera, size=resolution)
        # setup continuous capture method as rawCapture
		self.stream = self.camera.capture_continuous(self.rawCapture,
			format="bgr", use_video_port=True)
		# initialize the frame and the variable used to indicate
		# if the thread should be stopped
		self.frame = None
		self.stopped = False
        # error flag
        self.__error = False
    # return state of error flag
    def isError(self):
        return self.__error
    # start capture thread
	def start(self):
        self.stopped = False
        self.__error = False
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self
    # update loop used for capturing next frame
	def update(self):
		# keep looping infinitely until the thread is stopped
		for f in self.stream:
			# grab the frame from the stream and clear the stream in
			# preparation for the next frame
			self.frame = f.array
			self.rawCapture.truncate(0)
            # if frame is empty set error flag
            if self.frame is None:
                self.__error = True
                self.stopped = True
			# if the thread indicator variable is set, stop the thread
			# and resource camera resources
			if self.stopped:
				self.stream.close()
				self.rawCapture.close()
				self.camera.close()
				return
    # return latest frame
	def read(self):
		# return the frame most recently read
		return self.frame
    # stop capture thread by setting stopped flag
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
        
# multithreaded class for capturing frames from a Raspberry Pi camera with an internal buffer
# captured frames are stored in an internal buffer
def PiBufferStream(PiVideoStream):
    def __init__(self,resolution=(320,240),framerate=32,qlen=100):
        super().__init__(self,resolution,framerate)
        # initialize queue
        self.q = deque(maxlen=qlen)
    # update method run by main thread
    def update(self):
        for f in self.stream:
            self.frame = f.array
            self.rawCapture.truncate(0)
            # add to queue
            self.q.append(self.frame)
            # if frame is empty set error flag
            if self.frame is None:
                self.__error = True
                self.stopped = True
			# if the thread indicator variable is set, stop the thread
			# and resource camera resources
			if self.stopped:
				self.stream.close()
				self.rawCapture.close()
				self.camera.close()
				return
    # return latest frame
    def read(self):
        if self.q:
            return self.frame
        else:
            return self.q.pop()

# function for demonstrating using the capture classes
def demo(useBuffer=True):
    if useBuffer:
        vs = PiBufferStream().start()
    else:
        # create camera class and start
        vs = PiVideoStream().start()
    # capture loop
    while True:
        # get frame
        frame = vs.read()
        # display
        cv2.imshow("Demo",frame)
        # get key press
        key = cv2.waitKey(1)&0xFF
        # exit on ESC
        if k==27:
            break
    # close windows
    cv2.destroyAllWindows()
    # stop camera
    vs.stop()
		
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
    parser.add_argument("-buffer","--b",dest="useBuffer",action="store_true",help="Flag to use the buffered version of the capture class")
    args = parser.parse_args()
    demo(args.useBuffer)