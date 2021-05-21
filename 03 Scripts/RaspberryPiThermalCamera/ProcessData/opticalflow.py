import numpy as np
import cv2
import h5py
import os

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

path = "D:\BEAM\Scripts\PackImagesToHDF5\pi-camera-data-127001-2019-10-14T12-41-20.hdf5"

color = np.random.randint(0,255,(100,3))

os.makedirs("OpticalFlow",exist_ok=True)
with h5py.File(path,'r') as file:
    w,h,d = file['pi-camera-1'].shape
    # choose a good frame to represent the objects to track
    frame_ref = file['pi-camera-1'][:,:,91399]
    # normalize the frame
    frame_ref = (frame_ref-frame_ref.min())/(frame_ref.max()-frame_ref.min())
    frame_ref = frame_ref.astype('uint8')
    sobel_ref = cv2.Sobel(frame_ref,cv2.CV_8U,1,1,(3,3))
    # find corners of the image
    p0 = cv2.goodFeaturesToTrack(sobel_ref,mask=None,**feature_params)
    # create mask to draw line of movement on
    mask = np.zeros((w,h,3),np.uint8)
    hsv = np.zeros((w,h,3),np.uint8)
    hsv[...,1]=255
    for ff in range(91399,93000):
        # get frame
        frame = file['pi-camera-1'][:,:,ff]
        # norm frame
        frame = (frame-frame.min())/(frame.max()-frame.min())
        frame = frame.astype('uint8')
        #sb = cv2.Sobel(frame,cv2.CV_8U,1,1,(3,3))
        # perform optical flow using reference frame to determine what's changed
        p1,st,err = cv2.calcOpticalFlowPyrLK(frame_ref,frame,p0,None,**lk_params)

        if p1 is not None:
            # select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            # draw tracks
            frame_col = np.dstack((frame,frame,frame))
            # iterate through found reference points
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                # draw line to indicate movement
                mask[...]=0.0
                mask = cv2.line(mask,(a,b),(c,d),color[i].tolist(),2)
                # circle to indicate starting location
                frame_col = cv2.circle(frame_col,(a,b),5,color[i].tolost(),-1)

            # add the two drawings togather
            img = cv2.add(frame_col,mask)

        # update reference frame and feature points
        frame_ref = frame.copy()
        p0 = good_new.reshape(-1,1,2)

        cv2.imwrite(os.path.join("OpticalFlow","optical-flow-f{}.png".format(ff)),img)
        
        flow = cv2.calcOpticalFlowFarneback(frame_ref,frame,None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag,ang = cv2.cartToPolar(flow[...,0],flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2]=cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        cv2.imwrite(os.path.join("OpticalFlow","optical-flow--farneback-f{}.png".format(ff)),img)
        
        
            
