import cv2 
import numpy as np
import os
import lm
from scipy import ndimage
import logging

# setup logger
log = logging.getLogger(__name__)
# set formatting style
logging.basicConfig(format="[{filename}:{lineno}:{levelname} - {funcName}() ] {message}",style='{')
log.setLevel(logging.INFO)

# normalize the feature matrix elementwise using the mean and standard deviation
# the formula for normalizing is as follows:
#           x - mean(x)
#   x = ---------------------
#       2 x (3*std(x)+1)
# where the mean and std are the respective metrics over the whole dataset
def normalizeFeatures(fmatrix):
    # find the mean of each column in the feature matrix
    # IOW find the mean of each feature across the samples
    # result is reshaped into a column vector
    mean = np.mean(fmatrix,axis=0)
    std = np.std(fmatrix,axis=0)
    return (fmatrix-mean)/(2.0*((3.0*std)+1.0))

# calculate the color energy for each label in the image
# for each label is returns a 3 element vector describing the "energy" of each channel
# the function returns an array of energy vectors for each label
############ from https://github.com/Borda/pyImSegm/imsegm/descriptors.py ###################
def calculateColorEnergy(frame,labels):
    # get the number of labels
    num_labels = labels.max()+1
    log.debug(f"total {num_labels} labels")
    # create matricies
    energy = np.zeros((num_labels,3))
    counts = np.zeros(num_labels)
    # iterate over labels image
    log.debug("starting matrix iteration")
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            # get the label value for pixel i,j
            l = labels[i,j]
            # calculate energy
            energy[l,:] += frame[i,j,:]**2
            # update counts
            counts[l] += 1
            log.debug(f"({i},{j}), label {l}, energy {energy[l,:]}, counts {counts[l]}")
    # to prevent div 0 errors
    counts[counts==0] = -1
    # ?
    energy = (energy / np.tile(counts, (3, 1)).T.astype('float64'))
    log.debug(f"returning energy matrix of shape ({energy.shape})")
    return energy

# calculate the gray energy of a color image for each label in the image
# for each label is returns a 1 element vector describing the combined (?) "energy" of the channels
# the function returns an array of energy vectors for each label
############ adapted from https://github.com/Borda/pyImSegm/imsegm/descriptors.py ###################
def calculateGrayEnergy(frame,labels):
    log.debug(f"for frame {frame.shape} and labels {labels.shape}")
    # get the number of labels
    num_labels = labels.max()+1
    log.debug(f"total {num_labels} labels")
    # get the number of
    # create matricies
    energy = np.zeros(num_labels)
    counts = np.zeros(num_labels)
    # iterate over labels image
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            # iterate over channels
            for k in range(labels.shape[2]):
                # get the label value for pixel i,j
                l = labels[i,j,k]
                # calculate energy
                energy[l,:] += frame[i,j,k]**2
                # update counts
                counts[l] += 1
                log.debug(f"({i},{j},{k}), label {l}, energy {energy[l,:]}, counts {counts[l]}")
    # to prevent div 0 errors
    counts[counts==0] = -1
    # ?
    energy = (energy / counts.astype('float64'))
    log.debug(f"returning energy matrix of shape {energy.shape}")
    return energy

# calculate the "energy" of a single channel image 
def calculateGrayEnergy2D(frame,labels):
    log.debug(f"for frame {frame.shape} and labels {labels.shape}")
    # get the number of labels
    num_labels = labels.max()+1
    log.debug(f"total {num_labels} labels")
    # get the number of
    # create matricies
    energy = np.zeros(num_labels)
    counts = np.zeros(num_labels)
    # iterate over labels image
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            # get the label value for pixel i,j
            l = labels[i,j]
            # calculate energy
            energy[l] += frame[i,j]**2
            # update counts
            counts[l] += 1
            log.debug(f"({i},{j}, label {l}, energy {energy[l]}, counts {counts[l]}")
    # to prevent div 0 errors
    counts[counts==0] = -1
    # ?
    energy = (energy / counts.astype('float'))
    log.debug(f"returning energy matrix of shape {energy.shape}")
    return energy

# calculate the median color value of each channel for each label
# does not alter channel order
# returns an array of number of labls x 3
# i.e. superpixel label 0 has a channel mean of [100 for Blue, 40 Green and 90 Blue]
def calculateColorMedian(frame,labels):
    log.debug(f"for frame {frame.shape} and labels {labels.shape}")
    # get number of labels
    num_labels = labels.max()+1
    log.debug(f"total {num_labels} labels")
    # build matrix to hold medians value
    medians = np.zeros((num_labels,3))
    # iterate over labels
    for v in np.unique(labels):
        # create mask for the label
        mask = (labels!=v).astype("uint8")
        # mask the image
        # mask is converted to a 3 channel mask so it can be applied to the image
        fmask = np.ma.masked_array(frame,mask=np.dstack((mask,)*3))
        # find the median
        # special operation required for masked array
        medians[v,:] = np.ma.median(np.ma.median(fmask,axis=0),axis=0).data
        log.debug(f"label {v}: median {medians[v:]}")
    log.debug(f"returning median matrix of shape {medians.shape}")
    return medians

# calculate the mean color value of each channel for each label
# does not alter channel order
# returns an array of number of labls x 3
# i.e. superpixel label 0 has a channel mean of [100 for Blue, 40 Green and 90 Blue]
def calculateColorMean(frame,labels):
    log.debug(f"for frame {frame.shape} and labels {labels.shape}")
    # get number of labels
    num_labels = labels.max()+1
    log.debug(f"total {num_labels} labels")
    # build matrix to hold mean value
    means = np.zeros((num_labels,3))
    # iterate over labels
    for v in np.unique(labels):
        # create mask for the label
        mask = (labels!=v).astype("uint8")
        # mask the image
        # mask is converted to a 3 channel mask so it can be applied to the image
        # masked entries are ignored 
        fmask = np.ma.masked_array(frame,mask=np.dstack((mask,)*3))
        # find the mean
        means[v,:] = fmask.mean(axis=0).mean(axis=0).data
        log.debug(f"label {v}: mean {means[v:]}")
    log.debug(f"returning means matrix of shape {means.shape}")
    return means

# calculate the standard deviation color value of each channel for each label
# does not alter channel order
# returns an array of number of labls x 3
# i.e. superpixel label 0 has a channel standard deviation of [100 for Blue, 40 Green and 90 Blue]
def calculateColorStd(frame,labels):
    log.debug(f"for frame {frame.shape} and labels {labels.shape}")
    # get number of labels
    num_labels = labels.max()+1
    log.debug(f"total {num_labels} labels")
    # build matrix to hold stds value
    stds = np.zeros((num_labels,3))
    # iterate over labels
    for v in np.unique(labels):
        # create mask for the label
        mask = (labels!=v).astype("uint8")
        # mask the image
        # mask is converted to a 3 channel mask so it can be applied to the image
        fmask = np.ma.masked_array(frame,mask=np.dstack((mask,)*3))
        # find the standard deviation
        stds[v,:] = fmask.std(axis=0).std(axis=0).data
        log.debug(f"label {v}: std {stds[v:]}")
    log.debug(f"returning stds matrix of shape {stds.shape}")
    return stds

# calculate the mean color gradient for each label in the labels matrix
# returns a number of labels x 3 of the mean gradient for each channel
# channel order of input frame not changed
def calculateColorMeanGrad(frame,labels):
    log.debug(f"for frame {frame.shape} and labels {labels.shape}")
    # get number of labels
    num_labels = labels.max()+1
    log.debug(f"total {num_labels} labels")
    # create matrix to hold gradient of the image
    grad = np.zeros(frame.shape,dtype='float')
    # iterate over image row wise and calculate gradient
    for i in range(frame.shape[0]):
        log.debug(f"finding gradient of row {i}")
        grad[i,:,:] = np.sum(np.gradient(frame[i]),axis=0)
    # create matrix to hold means
    grad_mean = np.zeros((num_labels,3))
    # iterate over unique labels
    for v in np.unique(labels):
        # build mask
        mask = (labels!=v).astype("uint8")
        # mask gradient 
        fmask = np.ma.masked_array(grad,mask=np.dstack((mask,)*3))
        # calculate mean
        grad_mean[v,:]=fmask.mean(axis=0).mean(axis=0).data
        log.debug("label {v}, mean gradient {grad_mean[v,:]}")
    log.debug("returning mean gradient matrix of shape {grad_mean.shape}")
    return grad_mean

# calculate the feature vector for the given frame for each unique label in labels
# current features in order are
#   - channel 0 mean, channel 1 mean, 2 mean
#   - channel 0 median, channel 1 median, 2 median
#   - channel 0 std, channel 1 std, 2 std
#   - channel 0 energy, channel 1 energy, 2 energy
#   - channel 0 mean gradient, channel 1 mean gradient, 2 mean gradient
# It returns a number of labels x number of features array
def calculateAllColorFeatures(frame,labels):
    log.debug(f"for frame {frame.shape} and labels {labels.shape}")
    # get number of labels
    num_labels = labels.max()+1
    log.debug(f"total {num_labels} labels")
    # create matrix to hold gradient of the image
    grad = np.zeros(frame.shape,dtype='float')
    # create matrix to hold means
    grad_mean = np.zeros((num_labels,3))
    # build matrix to hold stds value
    stds = np.zeros((num_labels,3))
    # build matrix to hold mean value
    means = np.zeros((num_labels,3))
    # build matrix to hold medians value
    medians = np.zeros((num_labels,3))
    # iterate over image row wise and calculate gradient
    for i in range(frame.shape[0]):
        log.debug(f"finding gradient of row {i}")
        grad[i,:,:] = np.sum(np.gradient(frame[i]),axis=0)
    # iterate over each unique label
    for v in np.unique(labels):
        # build mask
        mask = (labels!=v).astype("uint8")
        # mask gradient
        fmask = np.ma.masked_array(grad,mask=np.dstack((mask,)*3))
        # calculate mean
        grad_mean[v,:]=fmask.mean(axis=0).mean(axis=0).data
        # mask image
        fmask = np.ma.masked_array(frame,mask=np.dstack((mask,)*3))
        # find the standard deviation
        stds[v,:] = fmask.std(axis=0).std(axis=0).data
        # find the mean
        means[v,:] = fmask.mean(axis=0).mean(axis=0).data
        # find the median
        # special operation required for masked array
        medians[v,:] = np.ma.median(np.ma.median(fmask,axis=0),axis=0).data
        log.debug("label {v}, mean gradient {grad_mean[v,:]}, stds {stds[v,:]}, means {means[v,:]}, medians {medians[v,:]}")
    # calculate energy matrix
    energy = calculateColorEnergy(frame,labels)
    log.debug(f"finished calculating color energy, end shape {energy.shape}")
    log.debug(f"finished calculating color features for {num_labels}. concatentating results and returning")
    # concatenate the results together
    return np.concatenate((means,medians,stds,energy,grad_mean),axis=1)

# calculate the feature vector for the given frame for each unique label in labels
# it doesn't call each function separately but performs the same operations within one loop
# current features in order are
#   - channel 0 mean
#   - channel 0 median
#   - channel 0 std
#   - channel 0 mean gradient
# It returns a number of labels x number of features array
def calculateAllGrayFeatures(frame,labels):
    log.debug(f"for frame {frame.shape} and labels {labels.shape}")
    # get number of labels
    num_labels = labels.max()+1
    log.debug(f"total {num_labels} labels")
    # create matrix to hold gradient of the image
    grad = np.zeros(frame.shape,dtype='float64')
    # create matrix to hold means
    grad_mean = np.zeros(num_labels,dtype='float64')
    # build matrix to hold stds value
    stds = np.zeros(num_labels,dtype='float64')
    # build matrix to hold mean value
    means = np.zeros(num_labels,dtype='float64')
    # build matrix to hold medians value
    medians = np.zeros(num_labels,dtype='float64')
    # iterate over image row wise and calculate gradient
    for i in range(frame.shape[0]):
        log.debug(f"finding gradient of row {i}")
        grad[i,...] = np.sum(np.gradient(frame[i]),axis=0)
    # iterate over each unique label
    for v in np.unique(labels):
        # build mask
        mask = (labels!=v).astype("uint8")
        # mask gradient 
        fmask = np.ma.masked_array(grad,mask=mask)
        # calculate mean
        grad_mean[v,...]=fmask.mean(axis=0).mean(axis=0).data
        # mask image
        fmask = np.ma.masked_array(frame,mask=mask)
        # find the standard deviation
        stds[v,...] = fmask.std(axis=0).std(axis=0).data
        # find the mean
        means[v,...] = fmask.mean(axis=0).mean(axis=0).data
        # find the median
        # special operation required for masked array
        medians[v,...] = np.ma.median(np.ma.median(fmask,axis=0),axis=0).data
        log.debug("label {v}, mean gradient {grad_mean[v,:]}, stds {stds[v,:]}, means {means[v,:]}, medians {medians[v,:]}")
    energy = calculateGrayEnergy2D(frame,labels)
    log.debug(f"finished calculating gray energy, end shape {energy.shape}")
    log.debug(f"finished calculating gray features for {num_labels}. concatentating results and returning")
    # concatenate the results together
    return np.hstack((means,medians,stds,grad_mean))

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

# compute the texture feature vector using the given filter bank
# filter bank is assumed to be a 3D array of filters
def calculateTextureFeatureVector(frame,labels,fbank):
    log.debug(f"calculating texture feature vector for frame {frame.shape}, labels {labels.shape} and fbank {fbank.shape}")
    # compute filter response
    log.debug("computing filter responses for ALL filters")
    responses = [computeFilterResponse(frame,fbank[...,f]) for f in range(fbank.shape[2])]
    log.debug(f"computed {len(responses)} responses each of shape {responses[0].shape}")
    # to achieve rotational invariance, the maximum response for each channel across all rotational filters is used
    # replaces the sublist across entries 13:30 with the single element by result
    log.debug("finding maximum response to orientation filters, filter range [13:20]")
    responses[13:30] = [np.max(np.stack(responses[13:30],axis=0),axis=0,keepdims=True)[0,...]]
    # response is of the size channels x rows x columns
    # it is reshaped to be the same size as frame so features can be calculated
    # calculate the features for the filter response
    log.debug("finding color features for the modified responses")
    features = [calculateAllColorFeatures(r,labels) for r in responses]
    log.debug(f"computed {len(features)} feature vectors each of size {features[0].shape}")
    # features is a list of num_labels x num_features arrays
    # they are combined together to forma  num_labels x (num_features * num_filters) array
    features = np.concatenate(features,axis=1)
    log.debug(f"concatentated features together into matrix of shape {features.shape}")
    # remove Nans
    features[np.isnan(features)]=0
    log.debug("removing NaNs from features matrix")
    # return the features array
    return features

# segment the labels mask using the class labels returned by em class after running predict method
def segmentLabels(labels,res):
    clabels = labels.copy()
    for v in np.unique(clabels):
        clabels[clabels==v] = res[v].argmax()
    return clabels
