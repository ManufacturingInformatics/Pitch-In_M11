#!/usr/bin/env python3

import time
from datetime import datetime
import subprocess
import argparse
import h5py
import os
import numpy as np
import sys
import struct
import netifaces as ni

SKIP_FRAMES = 2           # Frames to skip before starting recording
FPS = 4                   # Should match the FPS value in examples/rawval.cpp
RAW_RGB_PATH = "./examples/rawval"

# create and setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--fps',metavar='FPS', type=int, help='Framerate to capture. Default: 4',
                    choices=[1, 2, 4, 8, 16, 32, 64], default=FPS)
parser.add_argument('--skip',metavar='S', type=int, help='Frames to skip. Default: 2', default=SKIP_FRAMES)
parser.add_argument('--time',metavar='T', type=int, help='Time limit of script in seconds. Default inf',default=0)
args = parser.parse_args()

# get the values supplied by user
fps = args.fps
if fps<=0:
    raise RuntimeError("FPS Error. Received {} FPS. Cannot be negative or zero".format(fps))

skip_frames = args.skip
if skip_frames<0:
    raise RuntimeError("Skip Frame Error. Received {} skip frames. Cannot be negative".format(skip_frames))

timelim = args.time
if timelim<0:
    raise RuntimeError("Time Error. Received {} time limit. Cannot be zero or negative".format(timelim))

#print program settings
print("Settings: FPS {0}, skip frames {1}, time limit {2} secs".format(fps,skip_frames,timelim if timelim!=0 else "no limit set"))

if not os.path.isfile(RAW_RGB_PATH):
    raise RuntimeError("{} doesn't exist, did you forget to run \"make\"?".format(RAW_RGB_PATH))

print("""log-to-hdf5.py - output a hdf5 file based off the current date and time using ./rawval command.

hdf5 file name is pi-camera-data-IP-YYYY-MM-DDTHH-MM-SS-mmmmmm.hdf5 using date time in isoformat and the wlan0 network address.

Use ./rawval to grab temperature values as a binary string from the MLX90640

You must have built the "rawval" executable first with "make"

Press Ctrl+C to save & exit!

""")

try:
    print("Creating h5py file")
    # create hdf5 file based off the current datetime in isoformat
    f = h5py.File("pi-camera-data-{0}-{1}.hdf5".
        format(str(ni.ifaddresses('wlan0')[2][0]['addr']).replace(".",""),
               datetime.now().isoformat().replace(':','-').replace('.','-',1)))
    # create extentable datastet for camera footagte
    dset = f.create_dataset("pi-camera-1",(24,32,1),maxshape=(24,32,None),dtype=np.float16)    
    # get current start time to keep track of run time and if time limit has been reached
    startt = datetime.now()
    # call rawval executable to get the output sent along stdout
    with subprocess.Popen(["sudo", RAW_RGB_PATH, "{}".format(fps)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as camera:
       while True:
                # Despite the docs, we use read() here since we want to poll
                # the process for chunks of 3072 bytes, each of which is a frame
                frame = camera.stdout.read(3072)
                print("Got {} bytes of data!".format(len(frame)))
                # skip the set number of frames
                if skip_frames > 0:
                    time.sleep(1.0 / fps)
                    skip_frames -= 1
                    continue
                # data is received as a bytes class
                # each 4 bytes is a float
                # bytes2float converts each group of 4 to a float
                # the result is resized to a frame and used to update dataset
                dset[:,:,-1]=np.resize(np.frombuffer(frame,dtype='float16'),(24,32))
                dset.resize((24,32,dset.shape[2]+1))
                ## check time limit
                # if a timelimit is set
                if timelim!=0:
		    # get time passed
                    dur = datetime.now()-startt
                    # if elapsed time has exceeded limit, exit loop
                    if dur.total_seconds()>=timelim:
                        print("Time limit reached!")
                        break

		# wait for the appropriate time between frame
                time.sleep(1.0 / fps)

except KeyboardInterrupt:
    print("Keyboard interrupt!")
    pass
finally:
    # show runtime
    dur = datetime.now() - startt
    dur_s = dur.total_seconds()
    # runtime in mins and seconds
    print("{:.2f} mins {:.5f} secs".format(*divmod(dur_s,60)))
    # est. fps using first dataset size, used as a performance measure
    # takes into account skipped frames
    print("Estimated {:.2f} FPS ({:.2f} FPS)".format(f["pi-camera-1"].shape[2]/(dur_s-(skip_frames/fps)),fps))
    ## print status information about the file
    # number of datasets collected i.e. number of cameras
    print("Collected {0} datasets".format(len(f.keys())))
    for v in f.values():
        print("{0} : {1} : {2} Mb".format(v.name,v.shape,v.size/(1024.0**2.0)))
    # close the file so it can be read later
    f.close()
