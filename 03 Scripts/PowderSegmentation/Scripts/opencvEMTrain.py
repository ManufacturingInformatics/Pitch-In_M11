import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import lm
from scipy import ndimage
import descriptors
import utilities
import logging

# setup logger
log = logging.getLogger(__name__)
# set formatting style
logging.basicConfig(format="[{filename}:{lineno}:{levelname} - {funcName}() ] {message}",style='{')
log.setLevel(logging.INFO)

# parameters for SLICO algorithm
region_size = 100
ruler = 15.0
ratio = 0.075
min_element_size = 25
num_iterations = 10

# open the default camera, collect frames, perform SLICO superpixels and classify them
def useCamera():
    log.info("Getting camera")
    # get the webcam/default camera
    log.debug("opening camera")
    cam = cv2.VideoCapture(0)
    # if not opened
    # exit
    if not cam.isOpened():
        log.critical("Failed to get camera!")
        log.debug("releasing camera")
        cam.release()
        return

    log.info("Getting test frame")
    ret,frame = cam.read()
    if not ret:
        log.critical("Failed to get test frame!")
        log.debug("releasing camera")
        cam.release()
        return

    log.debug("building lambda to simplify passing to SLICO")
    # create simplified lambda function for passing new frames to it with the same settings
    pass_image = lambda src : cv2.ximgproc.createSuperpixelSLIC(src,cv2.ximgproc.SLICO,region_size,ruler)
    # transparency of blending
    alpha = 0.6
    ## values of supported keys
    # s
    s_key = ord('s')
    log.debug("building mask and visibility arrays")
    # create array used for masking labels
    mask = np.ones(frame.shape[:2])
    # create visiblity results
    vis = np.zeros(frame.shape[:2],dtype='float')
    
    log.info("Building texture filter bank")
    # build texture filter bank
    fbank = lm.makeLMfilters()
    log.info("Training EM")
    logging.debug("Building EM class")
    # create EM algorithm class
    em = cv2.ml.EM_create()
    # set the number of classes/clusters
    logging.debug(f"Setting the number of clusters to {nc}")
    em.setClustersNumber(nc)
    # main loop
    log.info("starting loop")
    log.debug("entering main collection loop")
    while True:
        log.debug("getting new frame")
        # get frame
        ret,frame = cam.read()
        log.debug(f"retrieved frame ({frame.shape}) and ret ({ret})")
        if not ret:
            logging.error("failed to get frame")
            cam.release()
            return
        # pass new frame
        log.debug(f"performing SLICO superpixels with region size {region_size} and ruler {ruler}")
        superpix = pass_image(frame)
        # calculate superpixel segmentation on the previous given image
        log.debug(f"iterating SLICO {num_iterations} iters")
        superpix.iterate(num_iterations)
        # min_element_size enforces a minimum size of the superpixels
        # merges smaller superpixels into a giant one
        # min_element_size >>, << superpixels
        if min_element_size>0:
            log.debug(f"enforcing minimum superpixel size to {min_element_size}")
            superpix.enforceLabelConnectivity(min_element_size)
        # get labels
        # auto generated number controlled by min_element_size
        log.debug("retrieving superpixels labels matrix")
        labels = superpix.getLabels()
        if em.isTrained():
            # calculate color features
            log.info("Calculating color features")
            colF = descriptors.calculateGrayEnergy2D(frame,labels)
            log.info("Calculating texture features")
            # calculate response to texture filters
            textF = descriptors.calculateTextureFeatureVector(frame,labels,fbank)
            # build array of labels as additional feature
            log.info("Building feature vector")
            logging.debug("Building label vector")
            ul = np.unique(labels)
            logging.debug("Concatenating the feature sets together")
            # concatenate features together to form feature vector
            fvector = np.concatenate((ul.reshape((ul.shape[0],1)),colF,textF),axis=1)
            # train using built feature vector and returns the following:
            #   - retval :  flag indicating if it was successful
            #   - logLikelihoods : likelihood logarithm value for each sample
            #   - tlabels : Class label of each sample
            #   - probs : Optional output matrix that contains posterior probabilities of each gaussian mixture
            logging.debug("predicting using EM algorithm")
            retval,results = em.predict(fvector)
            print(results.shape)
            ##superover,dlabels,classover = createOverlay(frame,labels,tlabels)
##            # update display windows
##            log.debug("updating OpenCV display windows")
##            spimg = np.concatenate((np.dstack((labels,)*3).astype(frame.dtype),superover),axis=1)
##            cv2.imshow("Superpixels",spimg)
##            climg = np.concatenate((np.dstack((dlabels,)*3).astype(frame.dtype),classover),axis=1)
##            cv2.imshow("Classifier",climg)

        cv2.imshow("Frame",frame)

        ## handle key presses
        # wait 1ms to get user key press and update displays
        key = cv2.waitKey(1) & 0xFF
        log.debug(f"opencv key press {key}")
        # if ESC was pressed
        if key == 27:
            log.info("ESC pressed")
            log.debug("exiting main loop")
            break
        # if lowercase s was pressed
        elif key == s_key:
            log.info("saving screenshot")
            # concat the superpixels overlay on top of the classifier overlay
            screenshot = np.concatenate((frame,superover,classover),axis=0)
            log.debug(f"saving screenshot {screenshot.shape} to ./opencv_slic_classifier_screenshot.png")
            cv2.imwrite("opencv_slic_classifier_screenshot.png",screenshot)
    log.debug("exited main loop")
    # release camera
    log.debug("releasing camera")
    cam.release()

# perform SLICO on grayscaled frames collected from the default camera
def useCameraGray(nc=2):
    log.info("getting camera")
    # get the webcam/default camera
    log.debug("creating OpenCV::VideoCapture class for default camera")
    cam = cv2.VideoCapture(0)
    # if not opened
    # exit
    if not cam.isOpened():
        log.critical("failed to open camera")
        cam.release()
        return
    log.info("camera opened")
    log.debug("camera idx 0 opened")

    log.info("getting test frame")
    ret,frame = cam.read()
    log.debug(f"retrieved frame ({frame.shape}) and ret ({ret})")
    if not ret:
        log.critical("failed to get frame")
        cam.release()
        return
    # convert image to grayscale
    log.debig("converting frame to grayscale")
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # create simplified lambda function for passing new frames to it with the same settings
    log.info("building algorithm")
    log.debug("building lambda for performing superpixels")
    pass_image = lambda src : cv2.ximgproc.createSuperpixelSLIC(src,cv2.ximgproc.SLICO,region_size,ruler)
    # transparency of blending
    alpha = 0.6
    ## values of supported keys
    # s
    s_key = ord('s')
    log.debug("creating mask and visibility matricies")
    # create array used for masking labels
    mask = np.ones(frame.shape[:2])
    # create visiblity results
    vis = np.zeros(frame.shape[:2],dtype='float')
    # create expected maximization trainer
    log.debug("creating EM class")
    em = cv2.ml.EM_create()
    log.debug(f"Setting number of clusters {nc}")
    em.setClustersNumber(nc)
    # main loop
    log.info("starting loop")
    while True:
        log.debug("getting frame")
        # get frame
        ret,frame = cam.read()
        log.debug(f"retrieved frame ({frame.shape}) and ret ({ret})")
        if not ret:
            logging.error("failed to get frame")
            cam.release()
            return
        # convert frame to grayscale
        log.debug("converting frame to gray")
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # pass new frame and perform superpixels
        log.debug("performing SLICO superpixels")
        superpix = pass_image(frame)
        # calculate superpixel segmentation on the previous given image
        log.debug(f"iterating SLICO {num_iterations} iter")
        superpix.iterate(num_iterations)
        # min_element_size enforces a minimum size of the superpixels
        # merges smaller superpixels into a giant one
        # min_element_size >>, << superpixels
        if min_element_size>0:
            log.debug(f"forcing min superpixel size to {min_element_size}")
            superpix.enforceLabelConnectivity(min_element_size)

        # on each of the pixels
        log.debug("updating OpenCV display windows")
        cv2.imshow("Frame",frame)
        cv2.imshow("Vis",vis)
        cv2.imshow("Output",output)

        ## handle key presses
        # get key press and allow opencv to draw results
        key = cv2.waitKey(1) & 0xFF
        log.debug(f"received OpenCV key press {key}")
        # if ESC was pressed
        if key == 27:
            log.info("Exiting")
            log.debug(f"User pressed {key}")
            # break from loop and handle exit
            break
    log.debug("releasing camera")
    log.info("exiting")
    cam.release()

# train a EM algorithm with just the frame
# training based off RGB values
def trainEMWithRGB(frame,nc=2):
    log.info("starting")
    log.debug(f"received frame ({frame.shape})")
    # reshape the array into a vector of colours
    # rows * cols x 3
    cols = frame.reshape(-1,frame.shape[-1])
    log.debug(f"reshaped frame into vector of samples ({cols.shape})")
    # create EM object
    log.debug("creating EM class")
    log.info("creating EM class")
    em = cv2.ml.EM_create()
    log.debug(f"setting the number of classed to {nc}")
    em.setClustersNumber(nc)
    # train EM
    # colors passed as individual samples
    log.info("training EM")
    log.debug(f"training EM with samples ({cols.shape}")
    return em.trainEM(cols)

# take a screenshot from the default camera, perform SLICO, calculate the features of each superpixels and train an EM algorithm using it
# the SLICO parameters are set at the top of the program
# function accepts the number of clusters it should be separated by
# returns the following
#   - frame : Image obtained
#   - labels : Labels matrix generated by superpixels
#   - probs : Posteriori probabilities of each Gaussian Mixture component
#   - em    : Trained EM class
def trainEMUsingScreenshotFeatures(nc=2):
    # call useScreenshot to get a frame from the camera and apply SLICO superpixels
    # parameters for SLICO are set at the top of the file
    log.info("Getting screenshot")
    frame,labels,sp = utilities.useScreenshot()
    # calculate color features
    log.info("Calculating color features")
    colF = descriptors.calculateAllColorFeatures(frame,labels)
    log.debug("normalizing the color features")
    colF = descriptors.normalizeFeatures(colF)
    log.info("Building texture filter bank")
    # build texture filter bank
    fbank = lm.makeLMfilters()
    log.info("Calculating texture features")
    # calculate response to texture filters
    textF = descriptors.calculateTextureFeatureVector(frame,labels,fbank)
    log.debug("normalizing the texture vectors")
    textF = descriptors.normalizeFeatures(textF)
    # build array of labels as additional feature
    log.info("Building feature vector")
    logging.debug("Building label vector")
    ul = np.unique(labels)
    logging.debug("Concatenating the feature sets together")
    # concatenate features together to form feature vector
    fvector = np.concatenate((ul.reshape((ul.shape[0],1)),colF,textF),axis=1)
    log.info("Training EM")
    logging.debug("Building EM class")
    # create EM algorithm class
    em = cv2.ml.EM_create()
    # set the number of classes/clusters
    logging.debug(f"Setting the number of clusters to {nc}")
    em.setClustersNumber(nc)
    # train using built feature vector and returns the following:
    #   - retval :  flag indicating if it was successful
    #   - logLikelihoods : likelihood logarithm value for each sample
    #   - tlabels : Class label of each sample
    #   - probs : Optional output matrix that contains posterior probabilities of each gaussian mixture
    logging.debug("Training EM algorithm")
    retval, logLikelihoods, tlabels, probs = em.trainEM(fvector)
    log.info("Finished training EM algorithm")
    log.debug(f"retval {retval}, logLikelihoods {logLikelihoods.shape}, tlabels {tlabels.shape}, probs {probs.shape}")
    # build overlays
    log.info("building overlay images")
    return frame,labels,logLikelihoods,tlabels,probs
        
if __name__ == "__main__":
    frame,labels,logLikelihoods,tlabels,probs = trainEMUsingScreenshotFeatures()
    superover,dlabels,classover = utilities.createOverlay(frame,labels,tlabels)
    # update display windows
    log.debug("updating OpenCV display windows")
    cv2.imshow("Frame",frame)
    spimg = np.concatenate((np.dstack((labels,)*3).astype(frame.dtype),superover),axis=1)
    cv2.imshow("Superpixels",spimg)
    climg = np.concatenate((np.dstack((dlabels,)*3).astype(frame.dtype),classover),axis=1)
    cv2.imshow("Classifier",climg)
