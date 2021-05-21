import cv2
import os
import logging
import numpy as np

# setup logger
log = logging.getLogger(__name__)
# set formatting style
logging.basicConfig(format="[{filename}:{lineno}:{levelname} - {funcName}() ] {message}",style='{')
log.setLevel(logging.INFO)

# normalize each filter in the 3D array filter bank and save as an image to the set path
# filter filenames are built as <opath>/filter_{fname}_image.png
# filter names can be set by a list of strings or is built based as string version of filter indicies
def saveFiltersAsImages(fbank,opath="filters",filter_names = None):
    log.debug(f"saving {fbank.shape[2]} filters to {opath}")
    # create directory for results
    os.makedirs(opath,exist_ok=True)
    # build filter names
    if filter_names is None:
        log.debug("building filter names based off index")
        filter_names = [str(f) for f in range(fbank.shape[2])]
    # iterate over each filter in the 3D bank array
    for f,fname in zip(range(fbank.shape[2]),filter_names):
        log.debug(f"normalizing filter {f}, {fname}")
        # normalize and cast filter to image of type uint8
        fimage = cv2.normalize(fbank[:,:,f],0,255,cv2.NORM_MINMAX).astype("uint8")
        # build image path
        ipath = os.path.join(opath,f"filterbank_{fname}.png")
        # if failed to write image to target path
        if not cv2.imwrite(ipath,fimage):
            log.error(f"failed to save filter {fname} to {ipath}")
   
# function to save filter bank responses to images
# image filenames generated as <opath>/frame_filterResp_<filter name>_channel_<channel number>.png if channels have been split
# or <opath>/frame_filterResp_<filter name>_allChannels.png if not
#   - responses : Iterable list of responses 
#   - opath : Directory where the images will be written. Directory is created if it doesn't exists. Default ./filterResponse.
#   - split_channels : Flag to split the responses if they are a multi-channel response. If single channel, it is ignored. Default True
#   - filter_names : Custom names of the filters. Used in naming the output files. If None, a list of names is built based off index
def saveFilterResponseAsImages(responses,opath='.',split_channels=True,filter_names=None):
    # if no filter names have been given
    # build a list of names corresponding to index
    if filter_names is None:
        log.info("generating filter names")
        filter_names = [str(i) for i in range(len(responses))]
    # iterate over responses normalizing and saving them as images
    log.debug("starting iteration over responses")
    for r,fname in zip(responses,filter_names): 
        # if split channels flag is set
        if split_channels:
            # if the image has multiple channels
            if len(r.shape)==3:
                # split the channels
                channels = cv2.split(r)
                # iterate over channels
                for ci,c in enumerate(channels):
                    # normalize and cast to 8-bit
                    c = cv2.normalize(c,c,0,255,cv2.NORM_MINMAX).astype("uint8")
                    # save to directory
                    log.debug(f"saving filter {fname}, channel {ci} to frame_filterResp_{fname}_channel_{ci}.png")
                    # build image save path
                    ipath = os.path.join(opath,f"filterResp_{fname}_channel_{ci}.png")
                    # if failed to save image
                    if not cv2.imwrite(ipath,c):
                        log.error(f"failed to save filter {fname}, channel {ci} to {ipath}")  
        # if the flag is false or the image cannot be split
        # normalize the image
        c = cv2.normalize(r,r,0,255,cv2.NORM_MINMAX).astype("uint8")
        # build image path
        ipath = os.path.join(opath,f"frame_filterResp_{fname}_allChannels.png")
        log.debug(f"saving filter {fname}, all channels to {ipath}")
        # write image to path
        # if failed update logger
        if not cv2.imwrite(ipath,c):
            log.error(f"failed to save filter {fname} all channels to {ipath}")
    log.info("finished writing responses")

# draw the boundaries given my bmask on the target frame
# this differs from screenshotCentes as you can control the thickness
# based off drawContours
def drawBoundaries(frame,bmask,**kwargs):
    # if kwargs is empty, this evaluates to false
    if not kwargs:
        kwargs = {"color" : (0,255,255),
                  "thickness" : 1,
                  "lineType" : cv2.LINE_8,
                  }
    # if kwargs is not empty
    # initialize missing parameters for draw contours that require a value
    else:
        # if contour is missing
        # set color to yellow
        if "color" not in kwargs:
            kwargs["color"] = (0,255,255)

    log.debug(f"parameters {kwargs}")
    # get where the mask is set to 255
    y,x = np.where(bmask==255)
    log.debug(f"found {y.shape[0]} places where boundary is 255")
    pts = np.array([[[xx,yy]] for xx,yy in zip(x,y)],dtype='int32')
    log.debug(f"formed contour array of shape {pts.shape}")
    # draw contours and return result
    return cv2.drawContours(frame,pts,-1,**kwargs)

# find the boundaries of each unique labels region in the labels matrix and draw them on copy of frame
# kwargs controls the parameters passed to OpenCV drawContours function
def drawBoundariesLabels(frame,labels,**kwargs):
    # if kwargs is empty, this evaluates to false
    if not kwargs:
        kwargs = {"color" : (0,255,255),
                  "thickness" : 1,
                  "lineType" : cv2.LINE_8,
                  }
    # if kwargs is not empty
    # initialize missing parameters for draw contours that require a value
    else:
        # if contour is missing
        # set color to yellow
        if "color" not in kwargs:
            kwargs["color"] = (0,255,255)
    # create copy of frame to draw on
    draw = frame.copy()
    # iterate over unique labels
    for v in np.unique(labels):
        mask = (labels==v).astype("uint8")
        # find contours
        ct,_ = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        # draw contours on canvas
        draw = cv2.drawContours(draw,ct,-1,**kwargs)
    return draw

# calculate number of superpixels based off region_size
def calcNumRegions(fshape,rs):
    return int((fshape[0]*fshape[1])//(rs**2))

# calculate the region size required for the target number of pixels
def calcRegionSize(fshape,nc):
    return int(((fshape[0]*fshape[1])//nc)**0.5)

# generate images to display the superpixel labels (labels) and samples classifier labels (tlabels)
# builds and returns three images
#   - overlay of the superpixels labels matrix with the frame. Color is based off label value
#   - superpixels labels matrix when grouped according to the samples classifier labels
#   - overlay of the classifier labels matrix wtih the frame. Color is based off label value
def createOverlay(frame,labels,tlabels,alpha=0.6):
    log.info("building overlays")
    log.debug(f"for frame {frame.shape}, labels {labels.shape}, classifier labels {tlabels.shape}, alpha {alpha}")
    ## build classifier labels matrix
    # make a copy of the superpixels labels matrix to update with the classifier results
    dlabels = labels.copy()
    # iterate over the enumeration of the classifier labels returned by a trained EM result
    # updating the copy of the labels matrix with the classifier results
    for vi,v in enumerate(tlabels):
        log.debug(f"superpixel label {vi}, classifier label {v}")
        dlabels[labels==vi] = v[0]
    ## overlay classifier labels matrix ontop of frame
    # normalize the classifier labels matrix so it's an image
    # convert result to unsigned 8-bit so it's a valid image
    log.debug(f"normalizing classifier labels matrix and converting to type {frame.dtype}")
    dlabels = cv2.normalize(dlabels,dlabels,0,255,cv2.NORM_MINMAX).astype(frame.dtype)
    log.debug(f"classifier labels matrix image colors {np.unique(dlabels).tolist()}")
    # make a copy of frame
    classover = frame.copy()
    # overlay classifier labels matrix on top of copy of frame
    # alpha parameter controls level of transparency
    log.debug(f"overlaying classifier labels matrix ontop of copy of frame with alpha {alpha}")
    cv2.addWeighted(np.dstack([dlabels]*3),alpha,classover,1-alpha,0,classover)
    ## overlay superpixels labels ontop of image
    # normalize the superpixel labels matrix so it can be interprested as an image
    slabels = labels.copy()
    log.debug(f"normalizing superpixel labels matrix and converting to type {frame.dtype}")
    slabels = cv2.normalize(slabels,slabels,0,255,cv2.NORM_MINMAX).astype(frame.dtype)
    log.debug(f"superpixels labels matrix image colors {np.unique(dlabels).tolist()}")
    # overlay superpixel labels on top of copy of frame
    superover = frame.copy()
    log.debug(f"overlaying superpixel labels matrix ontop of copy of frame with alpha {alpha}")
    cv2.addWeighted(np.dstack([slabels.astype(frame.dtype)]*3),alpha,superover,1-alpha,0,superover)
    log.debug(f"returning images superpixel overlay {superover.shape}, classifier labels matrix {dlabels.shape}, classifier overlay {classover.shape}")
    # return images
    return superover,dlabels,classover

# open the camera, grabbed a frame and perform SLICO segmentation
# the frame and the labels matrix are returned
def useScreenshot(**kwargs):
    log.info("starting with {**kwargs}")
    # get the webcam/default camera
    log.debug("Getting camera 0")
    cam = cv2.VideoCapture(0)
    # if not opened
    # exit
    if not cam.isOpened():
        log.critical("failed to open camera")
        cam.release()
        return
    log.debug("successfully opened camera 0")
    # get a frame
    ret,frame = cam.read()
    log.debug(f"retrieved frame ({frame.shape}) and ret ({ret})")
    if not ret:
        log.critical("failed to get frame")
        cam.release()
        return

    ## checks the kwargs for the required entries
    # if no arguments were given
    if not kwargs:
        region_size = 100
        ruler = 15.0
        min_element_size = 25
        num_iterations = 10
    else:
        # check if region size has been specified
        # if not raise a ValueError and return
        if "region_size" not in kwargs:
            log.error("Missing target region size! Required for the superpixels algorithm")
            raise ValueError("Missing target region size from arguments!")
            return
        else:
            region_size = kwargs["region_size"]
        # check if number of SLICO iterations has been set
        if ("num_iterations" not in kwargs):
            log.error("Missing number of iterations! Required for superpixel algorithm")
            raise ValueError("Missing max number of iterations for OpenCV superpixels algorithmm")
        else:
            num_iterations = kwargs["num_iterations"]
        # check if the minimum element size after spatial regularization has been applied
        if "min_element_size" not in kwargs:
            # if the minimumm size has not been specified
            # check if the number of clusters has been specified
            if "number_clusters" not in kwargs:
                log.error("Missing minimum size and number of clusters! Required for superpixel algorithm")
                raise ValueError("Missing minimum element size and number of clusters for OpenCV superpixels algorithm")
            # if the number of clusters has been specified
            # calculate region size based off the number of clusters
            else:
                region_size = calcRegionSize(frame.shape,kwargs["number_clusters"])
                # set the minimum element size as a quarter (arbitrary) of the region size
                min_element_size = region_size//4
        else:
            min_element_size = kwargs["min_element_size"]
            
        if "ruler" not in kwargs:
            log.error("Missing ruler parameter! Required for superpixel algorithm")
            raise ValueError("Missing ruler parameter for OpenCV superpixels algorithm")
            return
        else:
            ruler = kwargs["ruler"]
            
    log.info("performing superpixel segmentation")
    log.debug(f"performing SLICO ({cv2.ximgproc.SLICO}) OpenCV superpixel segmentation, region_size : {region_size}, ruler : {ruler}")
    # perform SLICO algorithm on the frame with the target settings
    superpix = cv2.ximgproc.createSuperpixelSLIC(frame,cv2.ximgproc.SLICO,region_size,ruler)
    # perform a set number of iterations on the current frames
    log.debug(f"iterating superpix {num_iterations} iters")
    superpix.iterate(num_iterations)
    # force a minimum size of superpixels
    if min_element_size>0:
        log.debug(f"forcing minimum superpixel size to {min_element_size}")
        superpix.enforceLabelConnectivity(min_element_size)
    # get the labels matrix
    labels = superpix.getLabels()
    log.info(f"finished superpixels {np.unique(labels).shape}")
    log.debug("releasing camera")
    # release camera
    cam.release()
    # return color image and the superpixels label matrix
    return frame,labels,superpix

# iterate over unique labels and draw the string version in the centre each label mask
# draws on a copy of the frame passed and returns the result
def drawLabelRegions(frame,labels):
    # make create copy of frame to draw on
    draw = frame.copy()
    # iterate over unique labels
    for v in np.unique(labels):
        ## find centres
        # mask for superpixel
        mask = 255*(labels==v).astype("uint8")
        # find bounding box
        ct,_ = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        tx,ty,w,h = cv2.boundingRect(ct[0])
        ## draw text on image
        ## from https://gist.github.com/xcsrz/8938a5d4a47976c745407fe2788c813a
        # calculate size of text
        txt = str(v)
        textsize = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,1,2)[0]
        # get coordinate based on boundary
        textX = (w-textsize[0])//2
        textY = (h-textsize[1])//2
        # add text
        cv2.putText(draw,txt,(tx+textX,ty+textY),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
    return draw
