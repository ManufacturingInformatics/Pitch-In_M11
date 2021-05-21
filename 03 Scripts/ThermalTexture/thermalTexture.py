import cv2
import lm
from scipy import ndimage
import os
import numpy as np
import glob
from itertools import product

# create list of filenames
files = glob.glob(r"D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20-unpack\png"+r"\*.png")
# specific range
frange = [91398,150000]
files = files[frange[0]:frange[1]]
# create output directory
opath = "pi-data-lm-response"
os.makedirs(opath,exist_ok=True)
# make filterbank
fbank = lm.makeLMfilters()
# make filternames
#filter_names = [str(i) for i in range(fbank.shape[2])]
gauss_names = [f"gaussian_{f}" for f in range(1,5)]
blob_names  = [f"blob_LoG_{f}" for f in range(1,9)]
edge_names =  [f"edge_firstGauss_or{o}_sc{s}" for s in range(1,4) for o in range(1,7)]
bar_names = [f"bar_{b}" for b in range(1,19)]
filter_names = gauss_names + blob_names + edge_names + bar_names

# convolve a frame with a given filter f
# returns the weighted log normed response
def computeFilterResponse(frame,f):
    log.debug(f"for frame {frame.shape} and filter {f.shape}")
    # calculate the response for each channel
    if len(frame.shape)>2:
        log.debug("convolving 3d frame with filter")
        response = np.array([ndimage.convolve(frame[:,:,i],f) for i in range(frame.shape[2])])
    else:
        log.debug("convolving 2d frame with filter")
        response = ndimage.convolve(frame,f)
    log.debug(f"calculated colvolution response {response.shape}")
    # find the norm
    rnorm = np.sqrt(np.sum(np.power(response,2)))
    log.debug(f"rnorm is {rnorm}")
    # check for zeros and INFs
    if (rnorm==0) or (abs(rnorm)==np.Inf):
        log.warning(f"rnorm is {rnorm}. setting response to zeros")
        response = np.zeros(response.shape)
    else:
        # weight the response ?
        # all multiplications as they are slightly faster then divisions in python
        log.debug("finding log of response")
        response = (response * (np.log(1+rnorm)*(0.03**-1.0))) * (rnorm**-1.0)
    log.debug(f"reshaping response to {frame.shape}")
    return response.reshape(frame.shape)

# calculate and save the thermal camera image response to the texture filters
def calcThermalTexture():
    # anonymous function for calculating and saving the texture response for a specific frame
    # identified by the filename f
    def calcTextFor(frame,f):
        print(f)
        # convert to grayscale as it is known that the thermal responses are grayscale
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # create directory path inside the zip file for the image
        filePath = os.path.join(opath,os.path.splitext(os.path.basename(f))[0])
        os.makedirs(filePath,exist_ok=True)
        # compute filter responses
        for ff,fname in zip(range(fbank.shape[2]),filter_names):
            response = computeFilterResponse(frame,fbank[...,ff])
            # normalize filter response so it can be saved
            response = cv2.normalize(response,response,0,255,cv2.NORM_MINMAX).astype("uint8")
            # save the image
            if not cv2.imwrite(os.path.join(filePath,f"fr_{fname}.png"),response):
                print(f"failed to save filter response for {f}")
    # iterate over files
    # list comprehension is faster
    [calcTextFor(cv2.imread(f),f) for f in files]

# read back in the images produced by calcThermalTexture, combine them into a 7x7 array and save as a video
# uses OpenCV
# DOES NOT WORK. COULDN'T FIND THE RIGHT FOURCC
def buildTextureVideo(verbose=False):
    # set image fourcc
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
    # iterate over the filenames
    for fi,f in enumerate(files):
        if verbose:
            print(fi,f)
        # get the original image
        frame = cv2.imread(f)
        # if failed to read in image
        if frame is None:
            print(f"failed to read in image {f}")
            return
        # if it's the first image
        # setup the video writer
        if fi==0:
            vWriter = cv2.VideoWriter("pi-texture-response.avi",fourcc,1.0,(frame.shape[0]*7,frame.shape[1]*7),True)
            # if the writer failed to open
            if not vWriter.isOpened():
                print("Failed to open video writer!")
                return
            elif verbose:
                print(f"opened video writer at pi-texture-response.avi for images of size {(frame.shape[0]*7,frame.shape[1]*7)}")
        filePath = os.path.join(opath,os.path.splitext(os.path.basename(f))[0])
        # iterate over the filter results in order of filter names
        responses =[cv2.imread(os.path.join(filePath,f"fr_{fname}.png")) for fname in filter_names]
        responses = [frame]+responses
        # get frame size
        h,w,d = frame.shape
        # create matrix of zeros
        image = np.zeros((h*7,w*7,d),frame.dtype)
        # fill with white
        image.fill(255)
        # iterate over images and update the image matrix
        for (yi,xi),img in zip(product(range(7),range(7)),responses):
            # calculate location of current image in "super" image
            x = xi * w
            y = yi * h
            # update "super" image
            image[y:y+h,x:x+w,:]=img
        # write image to video
        vWriter.write(image)
    # close writer
    vWriter.release()

# read back in the images produced by calcThermalTexture, combine them into a 7x7 array and save as a video
# uses Matplotlib
def buildPltTextureVideo():
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    # define axes
    f,ax = plt.subplots()
    im = ax.imshow(getTestImage(),cmap="gray")
    ax.axis('off')
    # create writer
    metadata = dict(artist="DBM")
    writer = animation.writers["ffmpeg"](fps=20,metadata=metadata)
    # open the writer
    with writer.saving(f,"pi-texture-response.mp4",100):
        # iterate over filenames
        for f in files:
            # get the original image
            frame = cv2.imread(f)
            # if failed to read in image
            if frame is None:
                print(f"failed to read in image {f}")
                return
            filePath = os.path.join(opath,os.path.splitext(os.path.basename(f))[0])
            # iterate over the filter results in order of filter names
            responses =[cv2.imread(os.path.join(filePath,f"fr_{fname}.png")) for fname in filter_names]
            responses = [frame]+responses
            # get frame size
            h,w,d = frame.shape
            # create matrix of zeros
            image = np.zeros((h*7,w*7,d),frame.dtype)
            # fill with white
            image.fill(255)
            # iterate over images and update the image matrix
            for (yi,xi),img in zip(product(range(7),range(7)),responses):
                # calculate location of current image in "super" image
                x = xi * w
                y = yi * h
                # update "super" image
                image[y:y+h,x:x+w,:]=img
            # update the axes
            im.set_array(image)
            # grab frame and save to file
            writer.grab_frame()

# build and return the fi th results image
# used for debugging
def getTestImage(fi=0):
    f = files[fi]
    # get the original image
    frame = cv2.imread(f)
    # if failed to read in image
    if frame is None:
        print(f"failed to read in image {f}")
        return

    filePath = os.path.join(opath,os.path.splitext(os.path.basename(f))[0])
    # iterate over the filter results in order of filter names
    responses =[cv2.imread(os.path.join(filePath,f"fr_{fname}.png")) for fname in filter_names]
    responses = [frame]+responses
    # get frame size
    h,w,d = frame.shape
    # create matrix of zeros
    image = np.zeros((h*7,w*7,d),frame.dtype)
    # fill with white
    image.fill(255)
    # iterate over images and update the image matrix
    for (yi,xi),img in zip(product(range(7),range(7)),responses):
        # calculate location of current image in "super" image
        x = xi * w
        y = yi * h
        # update "super" image
        image[y:y+h,x:x+w,:]=img
    return image
