import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import os

# create list of filenames
files = glob.glob(r"D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20-unpack\png"+r"\*.png")
# specific range
frange = [91398,150000]
files = files[frange[0]:frange[1]]

# parameters
region_size = 4
ruler = 15.0
ratio = 0.075
min_element_size = 4
num_iterations = 10

alpha = 0.6
    
# generate LUT labels
# good for <=256 labels
# from https://stackoverflow.com/a/57080906
def genLUTLabels(labels):
    label_range = np.linspace(0,1,256)
    lut = np.uint8(plt.cm.viridis(label_range)[:,2::-1]*256).reshape(256, 1, 3) # replace viridis with a matplotlib colormap of your choice
    return cv2.LUT(cv2.merge((labels, labels, labels)), lut)

def useCmap(labels,cmap=cv2.COLORMAP_RAINBOW):
    lnorm = labels.copy()
    cv2.normalize(labels,lnorm,0,255,cv2.NORM_MINMAX)
    return cv2.applyColorMap(lnorm,cmap)

## based off https://www.pyimagesearch.com/2017/06/26/labeling-superpixel-colorfulness-opencv-python/
# adapted for live images
# function for labelling super pixel colourfulness
def segment_colorfulness(image,mask):
    # split the channels
    (B,G,R) = cv2.split(image.astype('float'))
    # build masks
    # only pixels within masks are therefore computed
    R = np.ma.masked_array(R,mask=mask)
    G = np.ma.masked_array(G,mask=mask)
    B = np.ma.masked_array(B,mask=mask)

    # compute difference between Red and Green channels
    rg = np.abs(R-G)

    # compute YB = 0.5*(R+G)-B
    yb = np.abs(0.5*(R+G)-B)

    # compute mean and standard deviation of results
    stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
    meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))
    # calculate the "colorfulness"
    return stdRoot + (0.3 * meanRoot)

# calculate colorfulness of gray image
# based off the colorful metrics above
def segment_color_gray(image,mask):
    mm = np.ma.masked_array(image,mask=mask)
    # computer mean and standard deviation
    stdRoot = np.sqrt(mm.std()**2)
    meanRoot = np.sqrt(mm.mean()**2)
    return stdRoot + (0.3*meanRoot)

# perform SLICO on each image in files
# normalize the labels produced and place them side by side
# save the result as a video
def useImageSet():
    # create axes
    f,ax = plt.subplots()
    # get the first image
    # same size as output
    frame = cv2.imread(files[0])
    if frame is None:
        print(f"Failed to read in {files[0]}")
        return
    # plot to get plot object
    im = ax.imshow(np.hstack((frame,frame)),cmap="gray")
    ax.axis('off')
    # create writer
    metadata=dict(artist="DBM")
    writer = animation.writers["ffmpeg"](fps=20,metadata=metadata)
    # define lambda expression to simplify passing images
    pass_image = lambda src : cv2.ximgproc.createSuperpixelSLIC(src,cv2.ximgproc.SLICO,region_size,ruler)
   
    with writer.saving(f,"pi-therma-slico.mp4",100):
        for f in files:
            frame = cv2.imread(f)
            if frame is None:
                print(f"Failed to read in {f}")
                return
            # perform slico
            superpix = pass_image(frame)
            # iterate
            superpix.iterate(num_iterations)
            if min_element_size>0:
                superpix.enforceLabelConnectivity(min_element_size)
            # get labels
            # auto generated number controlled by min_element_size
            labels = superpix.getLabels()
            # normalize
            labelImg = cv2.normalize(labels,labels,0,255,cv2.NORM_MINMAX).astype("uint8")
            # concat the two side by side
            # update axes
            im.set_array(np.hstack((frame,np.dstack((labelImg,)*3))))
            writer.grab_frame()

# perform SLICO on each image in files
# calculate the colorfulness of each superpixel
# overlay the colorfulness matrix on top of the image
# save the result as a video
def useImageSetColor():
    # create axes
    f,ax = plt.subplots()
    # get the first image
    # same size as output
    frame = cv2.imread(files[0])
    if frame is None:
        print(f"Failed to read in {files[0]}")
        return
    # plot to get plot object
    im = ax.imshow(frame,cmap="gray")
    ax.axis('off')
    # create writer
    metadata=dict(artist="DBM")
    writer = animation.writers["ffmpeg"](fps=20,metadata=metadata)
    # define lambda expression to simplify passing images
    pass_image = lambda src : cv2.ximgproc.createSuperpixelSLIC(src,cv2.ximgproc.SLICO,region_size,ruler)
    ## masks for calculating colorfulness
    # mask for masking image for each label
    mask = np.ones(frame.shape[:2])
    # create colorfulness results
    vis = np.zeros(frame.shape[:2],dtype='float')
    # open wrte in a non-interactive way
    with writer.saving(f,"pi-therma-slico-color.mp4",100):
        # iterate over files
        for f in files:
            # get image
            frame = cv2.imread(f)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # if failed to read in image
            if frame is None:
                print(f"Failed to read in {f}")
                return
            # perform slico
            superpix = pass_image(frame)
            # iterate
            superpix.iterate(num_iterations)
            # combine smaller regions into regions to achieve min size
            if min_element_size>0:
                superpix.enforceLabelConnectivity(min_element_size)
            # get labels
            # auto generated number controlled by min_element_size
            labels = superpix.getLabels()
            # iterate over each label
            for v in np.unique(labels):
                # reset mask
                mask[...]=1
                # update mask for superpixel
                # 1 is masked and 0 is not masked
                mask[labels==v]=0
                # compute colorfulness
                # mask channels
                vis[labels==v] = segment_color_gray(frame,mask)
            # normalize
            vis = cv2.normalize(vis,vis,0,255,cv2.NORM_MINMAX).astype("uint8")
            # overlay colorfulness results
            overlay = vis.copy()
            output = frame.copy()
            #print(overlay.shape,output.shape)
            cv2.addWeighted(overlay,alpha,output,1-alpha,0,output)
            # update axes
            im.set_array(np.hstack((frame,overlay,output)))
            writer.grab_frame()

# perform SLICO on a target image and return the image, labels and colorfulness matrix
def useImageSingle(fi=0):
    frame = cv2.imread(files[fi])
    if frame is None:
        print(f"Failed to read in {files[fi]}")
        return
    # mask for masking image for each label
    mask = np.ones(frame.shape[:2])
    # create colorfulness results
    vis = np.zeros(frame.shape[:2],dtype='float')
    # create lambda expression to simplify passing image
    pass_image = lambda src : cv2.ximgproc.createSuperpixelSLIC(src,cv2.ximgproc.SLICO,region_size,ruler)
    # perform slico
    superpix = pass_image(frame)
    # iterate
    superpix.iterate(num_iterations)
    if min_element_size>0:
        superpix.enforceLabelConnectivity(min_element_size)
    # get labels
    # auto generated number controlled by min_element_size
    labels = superpix.getLabels()
    # iterate over each label
    for v in np.unique(labels):
        # reset mask
        mask[...]=1
        # update mask for superpixel
        # 1 is masked and 0 is not masked
        mask[labels==v]=0
        # compute colorfulness
        # mask channels
        vis[labels==v] = segment_colorfulness(frame,mask)
    # normalize
    labelImg = cv2.normalize(labels,labels,0,255,cv2.NORM_MINMAX).astype("uint8")
    return frame,labelImg
        
if __name__ == "__main__":
    #useImageSet()
    cv2.destroyAllWindows()
